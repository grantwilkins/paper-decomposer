# prompts/

One module per LLM-driven pipeline phase. Each module holds:

1. The **system prompt** (and any per-role variants).
2. A `build_*_prompt(...)` function that assembles the message list (`list[dict[str, str]]`) for the phase.
3. An async `extract_*` / `assemble_*` / `deduplicate_*` entry point that calls [`models.call_model`](../models.py) / `call_model_with_fallback` with the right tier and response schema, then converts flat wire-format output back into the strict nested schema.
4. A local `_resolve_model_tier(config)` that reads `config.yaml → pipeline.<phase>.model_tier` (with a phase-specific default).

All entry points are invoked from [`pipeline.decompose_paper`](../pipeline.py).

## Modules

### [`seed.py`](seed.py) — Phase 1: abstract → seed skeleton
- System prompt: `SEED_SYSTEM_PROMPT`. Instructs the model to extract 3–7 typed claims (`context` / `method` / `result` / `assumption`) from the abstract, preserving quantitative specifics and split compound sentences.
- `build_seed_prompt(abstract_text) -> messages`.
- `extract_seed(abstract_text, config) -> SeedOutput`. Calls `call_model_with_fallback` on `FlatSeedOutput`, then maps each `FlatClaim` → `RawClaim` via `flat_claim_to_raw(fallback_section="abstract")`.
- Default tier: `small`. Pipeline retries with the intro body if fewer than 2 claims come back.

### [`section.py`](section.py) — Phase 2: per-section claim extraction
- Five role-specific instruction strings:
  - `METHOD_INSTRUCTIONS` — used for METHOD and THEORY sections. Extracts METHOD sub-claims and explicit NEGATIVE statements.
  - `EVALUATION_INSTRUCTIONS` — used for EVALUATION and APPENDIX. Preserves quantitative findings, flags failed ablations as NEGATIVE.
  - `INTRODUCTION_INSTRUCTIONS` — CONTEXT gaps + anything novel not in the seed skeleton.
  - `DISCUSSION_INSTRUCTIONS` — ASSUMPTION + NEGATIVE with a strict negative-calibration rule.
  - `BACKGROUND_INSTRUCTIONS` — contrastive positioning CONTEXT claims only, grouped 1–3.
  - `UNKNOWN_SECTION_INSTRUCTIONS` — explicit-only fallback for `RhetoricalRole.other`.
- `get_instructions_for_role(role)` is the dispatch table.
- `build_section_prompt(section, seed_claims, artifacts) -> messages`. Injects: (a) the numbered seed skeleton, (b) every paper artifact + caption, (c) the section body (truncated to 8000 chars with a marked middle cut), (d) the strict output contract. Uses `FlatSectionOutput` as the response schema.
- `extract_section_claims(section, seed_claims, artifacts, config) -> SectionExtractionOutput`. Normalizes each flat claim with fallback IDs `sec_<idx>` when the model returns empty IDs.
- `MAX_SECTION_CHARS_FOR_PROMPT = 8000` (~2000 tokens). Default tier: `small`.

### [`facets.py`](facets.py) — Phase 2b: method-claim classification + facet extraction
Runs **three** sequential LLM calls per METHOD claim (all at the `small` tier by default):

1. **Classify** (`build_classify_prompt`, schema `FlatInterventionClassification`) — pick exactly one `InterventionType` value.
2. **Domain facets** — dispatch on the classification to one of the `build_<type>_facet_prompt` builders:
   - `architecture` → `ArchitectureFacets` (A1–A5)
   - `objective` → `ObjectiveFacets` (O1–O4)
   - `algorithm` → `AlgorithmFacets` (G1–G5), with `_normalize_algorithm_facets` forcing `g3` to `UNSPECIFIED` if the answer looks like an empirical result rather than a formal guarantee.
   - `theory` → `TheoryFacets` (T1–T5)
   - `representation` → `RepresentationFacets` (P1–P4)
   - `data` → `EvaluationFacets` (reuses eval schema per config.yaml comment)
   - `systems` → `SystemsFacets` (S1–S6), with `s3_stack_layer` normalized via `StackLayer` aliases.
   - `evaluation` → `EvaluationFacets` (E1–E3)
   - `pipeline` → `PipelineFacets` (L1–L3)
3. **Universal facets** (`build_universal_prompt`, schema `FlatUniversalFacets`) — intervention_types (1–2), scope, improves_or_replaces, core_tradeoff, grounding, analogy_source. `_to_universal_facets` post-processes: ensures the classified type is in the list, normalizes scope/grounding tokens, clamps to 2 types.

Entry point: `extract_facets(claim, source_section, config) -> FacetedClaim`. **Only accepts METHOD claims** — raises `ValueError` otherwise.

Dispatch tables: `_FACET_PROMPT_BUILDERS`, `_FACET_SCHEMA_BY_TYPE`, `_FACET_FIELD_BY_TYPE`. Public lookups: `get_facet_prompt(type)`, `get_facet_schema(type)`.

### [`dedup.py`](dedup.py) — Phase 3: claim deduplication
Two-pass pipeline:

**Pass 1 — within-type batches.** `chunked_dedup` groups the canonical claim list by `claim_type.value`, chunks each group into batches of up to `_MAX_WITHIN_TYPE_BATCH = 20`, and kicks off one `dedup_type_batch` task per chunk with `asyncio.gather(..., return_exceptions=True)`. Each batch uses `WITHIN_TYPE_DEDUP_PROMPT` (schema `DedupBatchOutput`). Types with ≤2 claims skip the LLM entirely and become singleton groups.

**Pass 2 — cross-type parent links.** If the deduplicated canonical set is ≤40 claims, one `CROSS_TYPE_PROMPT` call (schema `CrossTypeOutput`) infers parent/child links across types (general RESULT → specific RESULT; top-level METHOD → sub-mechanism). Failure here is swallowed — pass 1 results stand.

- `apply_dedup(original_claims, DedupOutput) -> list[RawClaim]` keeps one canonical claim per group.
- `_normalize_groups` resolves `parent_id` pointers through the member-to-canonical map and rejects unresolvable parents with `ValueError`.
- Default tier: `medium`.

### [`tree.py`](tree.py) — Phase 4: tree assembly
- `TREE_SYSTEM_PROMPT` states the tree contract (CONTEXT roots → METHOD children → RESULT/ASSUMPTION/NEGATIVE leaves) and eight dependency rules (every RESULT depends on a METHOD, every METHOD addresses a CONTEXT, avoid star graphs, related-work positioning grouped under umbrella CONTEXT, etc.).
- `build_tree_prompt(metadata, claims, faceted_claims, negatives, artifacts)` organizes the prompt body into five labeled blocks: CONTEXT / METHOD (with facet summaries) / RESULT / ASSUMPTION / NEGATIVE / EVIDENCE ARTIFACTS. Uses `_facet_summary` to distill each `FacetedClaim` into a one-line intervention + 1-2 specified domain facets + scope + tradeoff.
- `assemble_tree(...)` calls the **heavy** tier with schema `TreeAssemblyOutput` (`one_liner` + `nodes: list[TreeNodeAssignment]`), then runs `_build_tree_nodes` which:
  1. Validates every `parent_id` / `depends_on` against the known claim IDs; drops unknowns and self-loops.
  2. Cycle-checks every parent assignment via `_would_create_parent_cycle`.
  3. Supplies fallback parents for claims the model left unassigned: METHOD → first CONTEXT; RESULT/ASSUMPTION/NEGATIVE → best-matching METHOD by `_method_affinity_score` (evidence overlap + section match + entity overlap + token overlap + rejected-token overlap), else first METHOD, else first CONTEXT.
  4. Groups related-work CONTEXT roots under a single umbrella if more than one exists.
  5. Re-attaches RESULT/ASSUMPTION/NEGATIVE children whose current parent is significantly worse than the best method match (or whose parent has ≥8 children — the "star graph" rule).
  6. Populates `depends_on` with the immediate METHOD parent where applicable; falls back to the parent ID otherwise.
- Default tier: `heavy`.

### [`__init__.py`](__init__.py)
Re-exports every public builder, entry point, and system prompt constant from the five modules so callers can `from paper_decomposer.prompts import build_section_prompt, extract_facets, assemble_tree, …`.
