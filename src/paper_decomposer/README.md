# paper_decomposer (package)

The implementation of the paper-decomposer pipeline. PDF in → parsed `PaperDocument` out, with the storage layer ready for the next-PR extraction stage that lands DAG nodes in Postgres.

## Module map

| File | Responsibility |
|------|----------------|
| [`__init__.py`](__init__.py) | Re-exports `ConfigError`, `get_config`, `load_config`. |
| [`__main__.py`](__main__.py) | `python -m paper_decomposer` entry point. Calls `cli.main()` and propagates its exit code. |
| [`cli.py`](cli.py) | `argparse`-based CLI. Accepts a single PDF or a directory of PDFs. Supports `--config <path>` and `--dry-run`. |
| [`config.py`](config.py) | Loads `config.yaml`, validates it into `PaperDecomposerConfig`, resolves each model tier into `RuntimeModelConfig`, and bundles everything as `AppSettings`. Raises `ConfigError` for missing file, malformed YAML, invalid data, or missing `TOGETHER_API_KEY`. |
| [`schema.py`](schema.py) | Pydantic models: `ApiConfig`, `ModelTierConfig`, `ModelsConfig`, `PdfPipelineConfig`, `PipelineConfig`, `DatabaseConfig`, `PaperDecomposerConfig`, `AppSettings`, `RhetoricalRole`, `EvidenceArtifact`, `Section`, `PaperMetadata`, `PaperDocument`. |
| [`models.py`](models.py) | The Together API client. `call_model(tier, messages, response_schema, config)` is the single entry point for LLM calls. Handles structured-output schema injection, JSON extraction, repair-suffix retry, exponential backoff, per-tier cost accounting. Also exports `preflight_model_tiers`, `reset_cost_tracker`, `get_cost_tracker`. |
| [`pdf_parser.py`](pdf_parser.py) | `parse_pdf(pdf_path, config) -> PaperDocument`. Uses PyMuPDF to extract text blocks with font metadata, detect two-column layout, segment sections by numbered/unnumbered headers, classify rhetorical role from header keywords, strip references, and extract figure/table/algorithm/theorem/lemma/definition captions into `EvidenceArtifact`s. |
| [`pipeline.py`](pipeline.py) | `ingest_paper(pdf_path, config_path)` — currently a thin shell that calls `parse_pdf`. The next PR fills in extraction → DB persistence. |
| [`db/schema.sql`](db/schema.sql) | Postgres DDL: papers, methods (DAG), settings (DAG), outcomes, claims. Requires `pgcrypto`, `pg_trgm`, `vector`. |
| [`db/client.py`](db/client.py) | `PaperDecomposerDB` — async psycopg connection pool. `apply_schema()` is implemented; the upsert/insert helpers are stubs that the next PR will fill. |

## Pipeline data flow (current)

```
 PDF
  │
  ▼ parse_pdf
 PaperDocument(metadata, sections, all_artifacts)
```

The next stage will take that `PaperDocument` and produce DAG nodes (methods, settings), outcomes, and claims, persisting them via `PaperDecomposerDB`.

## Cost tracking

`models._COST_TRACKER` is a module-global dict: `total_calls`, `prompt_tokens`, `completion_tokens`, `input_cost_usd`, `output_cost_usd`, `total_cost_usd`. Per-million rates come from `config.yaml`'s `models.<tier>.input_cost_per_m` / `output_cost_per_m`.
