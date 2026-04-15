# paper_decomposer (package)

The implementation of the paper decomposition pipeline. One PDF in, one [`PaperDecomposition`](schema.py#L418) JSON out.

## Module map

| File | Responsibility |
|------|----------------|
| [`__init__.py`](__init__.py) | Re-exports `ConfigError`, `get_config`, `load_config` for convenience. |
| [`__main__.py`](__main__.py) | `python -m paper_decomposer` entry point. Calls `cli.main()` and propagates its exit code. |
| [`cli.py`](cli.py) | `argparse`-based CLI. Accepts a single PDF or a directory of PDFs. Supports `--config <path>` and `--dry-run`. Runs `decompose_paper` inside `asyncio.run`. |
| [`config.py`](config.py) | Loads `config.yaml`, validates it into `PaperDecomposerConfig`, resolves each model tier into `RuntimeModelConfig`, and bundles everything as `AppSettings`. `load_config()` is uncached; `get_config()` is `lru_cache`-cached by path. Raises `ConfigError` for missing file, malformed YAML, invalid data, or missing `TOGETHER_API_KEY`. |
| [`schema.py`](schema.py) | Every Pydantic model used across the pipeline. Config models, enums, wire-format models (`Flat*` variants for API I/O), claim/facet/tree models, and the final `PaperDecomposition`. See "Schema" below. |
| [`models.py`](models.py) | The Together API client. `call_model(tier, messages, response_schema, config)` is the single entry point for LLM calls. Handles structured-output schema injection, JSON extraction (with fence stripping and embedded-object recovery), repair-suffix retry, exponential backoff, per-tier cost accounting. Also exports `preflight_model_tiers`, `reset_cost_tracker`, `get_cost_tracker`, and the `flat_claim_to_raw` adapter. |
| [`pdf_parser.py`](pdf_parser.py) | Phase 0. `parse_pdf(pdf_path, config) -> PaperDocument`. Uses PyMuPDF (`fitz`) to extract text blocks with font metadata, detect two-column layout, segment sections by numbered/unnumbered headers, classify rhetorical role from header keywords, strip references, and extract figure/table/equation captions into `EvidenceArtifact`s. |
| [`pipeline.py`](pipeline.py) | Orchestrator. `decompose_paper(pdf_path, config_path)` runs phases 0 → 0b → 1 → 2 → 2b → 3 → 4 with a `rich.Progress` display. Contains the ID normalization (`_normalize_claim_id_scheme`), parent-section lookup (`_find_section_for_claim`), tree-stat helper, and the fallback tree builder used when phase 4 raises. |
| [`prompts/`](prompts/) | One sub-module per LLM-driven phase. See [`prompts/README.md`](prompts/README.md). |

## Pipeline data flow

```
 PDF
  │
  ▼ parse_pdf (phase 0)
 PaperDocument(metadata, sections, all_artifacts)
  │
  ▼ preflight_model_tiers (phase 0b, fail-fast)
  │
  ▼ extract_seed (phase 1, abstract → seed claims, retry with intro if <2)
 list[RawClaim]  (seed)
  │
  ▼ extract_section_claims (phase 2, per-section, parallel, semaphore)
 list[RawClaim]  (seed + section claims, uniquified + re-numbered C#/M#/R#/A#/N#)
  │
  ▼ extract_facets (phase 2b, METHOD claims only, parallel)
 list[FacetedClaim]
  │
  ▼ chunked_dedup (phase 3, within-type batches → canonical list + groups + cross-type parent links)
 list[RawClaim]  (deduplicated) + list[ClaimGroup]
  │
  ▼ assemble_tree (phase 4, heavy tier assigns parent_id + depends_on)
 PaperDecomposition(metadata, one_liner, claim_tree, negative_claims, all_artifacts, extraction_cost_usd)
  │
  ▼ write JSON to pipeline.output.output_dir
```

Each phase logs incremental cost via `_cost_delta(phase_before, phase_after)`. If phase 1 raises, the pipeline proceeds without seed claims (with a warning). If phase 2 returns zero claims across all non-abstract sections, the run aborts with `RuntimeError("Phase 2 extracted zero claims …")`. If phase 3 raises, deduplication is skipped and the raw claims pass through. If phase 4 raises, `_build_fallback_decomposition` wires CONTEXT → METHOD → RESULT/ASSUMPTION/NEGATIVE deterministically from the claim list.

## Schema (schema.py)

The schema is divided into four layers:

| Layer | Models |
|-------|--------|
| **Config** | `ApiConfig`, `ModelTierConfig`, `ModelsConfig`, `PdfPipelineConfig`, `PipelineConfig`, `PaperDecomposerConfig` (raw YAML shape, `extra="allow"`), `RuntimeModelConfig`, `RuntimePipelineConfig`, `AppSettings` (bundle of `raw` + `model_tiers` + `pipeline` + `api_key` + `config_path`). |
| **Enums** | `ClaimType` (context/method/result/assumption/negative), `RhetoricalRole` (abstract/introduction/background/method/theory/evaluation/discussion/appendix/other), `InterventionType` (architecture/objective/algorithm/representation/data/systems/theory/evaluation/pipeline), `ScopeOfChange` (drop_in/module/system/paradigm), `GroundingType`, `StackLayer`. |
| **Parsed document** | `EvidenceArtifact`, `Section`, `PaperMetadata`, `PaperDocument`. |
| **Extraction** | `RawClaim`, `EvidencePointer`, flat wire variants (`FlatClaim`, `FlatSeedOutput`, `FlatSectionOutput`, `FlatUniversalFacets`, `FlatInterventionClassification`) used for Together's JSON-schema responses; nested equivalents (`SeedOutput`, `SectionExtractionOutput`, etc.); facet models (`UniversalFacets`, `SystemsFacets`, `ArchitectureFacets`, `ObjectiveFacets`, `AlgorithmFacets`, `TheoryFacets`, `RepresentationFacets`, `EvaluationFacets`, `PipelineFacets`); `FacetedClaim` (claim + universal + exactly one domain facet set); dedup models (`ClaimGroup`, `DedupOutput`, `ParentChildLink`, `CrossTypeOutput`, `DedupBatchOutput`); tree models (`TreeNodeAssignment`, `TreeAssemblyOutput`, `ClaimNode`, `OneLiner`, `PaperDecomposition`). |

**Flat vs nested.** Together's structured-output endpoint chokes on `$ref`/nested enums for some models, so prompts request the `Flat*` variants (strings for enum fields, flat `evidence_ids` list) and [`models.py`](models.py)'s `flat_claim_to_raw` plus the facet normalizers convert back into the strict nested models.

**`StackLayer` normalization.** `SystemsFacets.s3_stack_layer` accepts a range of model-generated synonyms ("hw", "kernel", "inference runtime", "application level", …) via a `field_validator` that folds them into the canonical enum values.

## Cost tracker

`models._COST_TRACKER` is a module-global dict: `total_calls`, `prompt_tokens`, `completion_tokens`, `input_cost_usd`, `output_cost_usd`, `total_cost_usd`. Per-million rates come from `config.yaml`'s `models.<tier>.input_cost_per_m` / `output_cost_per_m`. `pipeline.decompose_paper` resets the tracker at the start of every run and stores the final total in `PaperDecomposition.extraction_cost_usd`.

## Config surface

Phase-specific config is read out of `config.yaml → pipeline.<phase>` via small `_resolve_model_tier` helpers duplicated across each prompt module. They accept either an `AppSettings`/`RuntimePipelineConfig`-style object (attribute access) or a raw dict (mapping access), and fall back to a per-phase default (`small` for seed/section/facets, `medium` for dedup, `heavy` for tree).
