# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the decomposer on a single PDF
python -m paper_decomposer <path/to/paper.pdf> [--config config.yaml] [--dry-run]

# Batch mode: process all PDFs in a directory
python -m paper_decomposer <directory/> [--config config.yaml]

# Run all tests (excluding API-hitting tests)
uv run pytest -m "not api"

# Run a single test
uv run pytest tests/test_pipeline.py::test_pipeline_aborts_when_preflight_fails

# Run API integration tests (requires TOGETHER_API_KEY)
uv run pytest -m api
```

## Environment

`TOGETHER_API_KEY` must be set. The API calls use Together.ai's OpenAI-compatible endpoint (`https://api.together.xyz/v1`). The config loader raises `ConfigError` immediately if the key is missing.

## Architecture

The package lives in `src/paper_decomposer/`. `pyproject.toml` sets `pythonpath = ["src"]` for pytest.

**Data flow** — one PDF in, one `PaperDecomposition` JSON out, via five sequential phases:

| Phase | File | Input → Output |
|-------|------|----------------|
| 0 | `pdf_parser.py` | PDF → `PaperDocument` (sections + artifacts) |
| 1 | `prompts/seed.py` | Abstract text → seed `RawClaim` list |
| 2 | `prompts/section.py` | Each section → `RawClaim` list (parallel, semaphore-capped) |
| 2b | `prompts/facets.py` | Each method `RawClaim` → `FacetedClaim` (parallel) |
| 3 | `prompts/dedup.py` | All claims → deduplicated `RawClaim` list |
| 4 | `prompts/tree.py` | Claims + facets → `PaperDecomposition` with `ClaimNode` tree |

**Key modules:**
- `schema.py` — all Pydantic models used across the pipeline (`RawClaim`, `FacetedClaim`, `ClaimNode`, `PaperDecomposition`, etc.)
- `config.py` — loads `config.yaml` into `AppSettings` (which wraps `PaperDecomposerConfig` + resolved `RuntimeModelConfig` per tier); `load_config()` is uncached, `get_config()` is `lru_cache`-cached
- `models.py` — `call_model(tier, messages, response_schema, config)` wraps the Together API; handles JSON extraction, structured-output repair retries, exponential backoff, and global cost tracking via `_COST_TRACKER`
- `pipeline.py` — `decompose_paper(pdf_path, config_path)` orchestrates all phases with a `rich.Progress` display; contains the fallback tree builder used when Phase 4 fails
- `cli.py` — `main()` parses args; `--dry-run` only runs Phase 0 and prints a section table

**Model tiers** (`config.yaml → models.*`): `small` / `medium` / `heavy`. Each phase declares its tier. `preflight_model_tiers()` pings all required tiers before Phase 1 to fail fast on bad credentials or unavailable models.

**Structured output**: `call_model` passes `response_format: {type: "json_object", schema: ...}` to Together. On parse failure it appends a JSON repair suffix to the last user/system message and retries once before raising.

**Facet routing**: `config.yaml → facet_routing` maps `InterventionType` values to facet question sets (A1–A5, O1–O4, etc.). Facet Pydantic models live in `schema.py` (`ArchitectureFacets`, `SystemsFacets`, etc.).

**Output**: JSON files written to `config.yaml → pipeline.output.output_dir` (default `./output/`), named from the paper title. Existing files get a numeric suffix rather than being overwritten.
