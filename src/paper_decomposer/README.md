# paper_decomposer (package)

The implementation of the paper-decomposer pipeline. PDF in -> parsed `PaperDocument` -> graph-first validated paper-local extraction JSON, with the storage schema ready for DB persistence.

## Module map

| File | Responsibility |
|------|----------------|
| [`__init__.py`](__init__.py) | Re-exports `ConfigError`, `get_config`, `load_config`. |
| [`__main__.py`](__main__.py) | `python -m paper_decomposer` entry point. Calls `cli.main()` and propagates its exit code. |
| [`cli.py`](cli.py) | `argparse`-based CLI. Accepts a single PDF or a directory of PDFs. Supports `--config <path>`, `--dry-run`, `--extract`, and `--output-json`. |
| [`config.py`](config.py) | Loads `config.yaml`, validates it into `PaperDecomposerConfig`, resolves each model tier into `RuntimeModelConfig`, and bundles everything as `AppSettings`. Raises `ConfigError` for missing file, malformed YAML, invalid data, or missing `TOGETHER_API_KEY`. |
| [`schema.py`](schema.py) | Pydantic models: `ApiConfig`, `ModelTierConfig`, `ModelsConfig`, `PdfPipelineConfig`, `PipelineConfig`, `DatabaseConfig`, `PaperDecomposerConfig`, `AppSettings`, `RhetoricalRole`, `EvidenceArtifact`, `Section`, `PaperMetadata`, `PaperDocument`. |
| [`models.py`](models.py) | The Together API client. `call_model(tier, messages, response_schema, config)` is the single entry point for LLM calls. Handles structured-output schema injection, JSON extraction, repair-suffix retry, exponential backoff, per-tier cost accounting. Also exports `preflight_model_tiers`, `reset_cost_tracker`, `get_cost_tracker`. |
| [`pdf_parser.py`](pdf_parser.py) | `parse_pdf(pdf_path, config) -> PaperDocument`. Uses PyMuPDF to extract text blocks with font metadata, detect two-column layout, segment sections by numbered/unnumbered headers, classify rhetorical role from header keywords, strip references, and extract figure/table/algorithm/theorem/lemma/definition captions into `EvidenceArtifact`s. |
| [`pipeline.py`](pipeline.py) | `ingest_paper(pdf_path, config_path)` parses only. `extract_paper(pdf_path, config_path)` parses, selects evidence spans, runs staged extraction, applies deterministic cleanup, runs configured heavy cleanup, and validates paper-local JSON. |
| [`extraction/`](extraction/) | Paper-local extraction contracts, evidence selection, prompts, staged LLM calls, deterministic sanitization, validators, assembler, and DB write-plan mapping. |
| [`db/schema.sql`](db/schema.sql) | Postgres DDL: papers, extraction runs, evidence spans/links, methods (DAG), settings (DAG), method-setting links, outcomes, claims. Requires `pgcrypto`, `pg_trgm`, `vector`. |
| [`db/client.py`](db/client.py) | `PaperDecomposerDB` — async psycopg connection pool. `apply_schema()` is implemented; row-level transaction helpers are intentionally absent until the live DB writer lands. |

## Pipeline data flow (current)

```
 PDF
  |
  v parse_pdf
 PaperDocument(metadata, sections, all_artifacts)
  |
  v select_evidence_spans
 EvidenceSpan[]
  |
  v staged graph-first extraction + validation
 PaperExtraction(graph.systems, graph.methods, graph.settings, outcomes, claims)
```

The live DB transaction layer is still separate. The current extraction code produces validated JSON and a schema-aware write plan that preserves paper-local IDs and evidence. Claims-only output is a draft and fails validation until graph nodes and attachments exist. Normal compression and repair use configured cheap/reliable tiers; the configured heavy adjudication tier now runs once as final cleanup before final validation.

## Cost tracking

`models._COST_TRACKER` is a module-global dict: `total_calls`, `prompt_tokens`, `completion_tokens`, `input_cost_usd`, `output_cost_usd`, `total_cost_usd`. Per-million rates come from `config.yaml`'s `models.<tier>.input_cost_per_m` / `output_cost_per_m`.
