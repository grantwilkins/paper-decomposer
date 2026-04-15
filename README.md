# paper-decomposer

A pipeline that converts a research paper PDF into a structured **claim decomposition**: a JSON tree of typed claims (context / method / result / assumption / negative), with domain-specific facets attached to method claims, dependency edges between claims, and a three-part "one-liner" summarizing what the paper achieved, how, and why.

The target domain is ML / systems papers. All LLM calls go through the Together.ai serverless inference API (OpenAI-compatible endpoint).

---

## Repository layout

| Path | Purpose |
|------|---------|
| [src/paper_decomposer/](src/paper_decomposer/) | Primary package source: pipeline, schema, models client, PDF parser, prompts. |
| [paper_decomposer/](paper_decomposer/) | Thin shim package at repo root that re-exports `src/paper_decomposer` so `python -m paper_decomposer …` works without editable install. |
| [src/paper_decomposer/prompts/](src/paper_decomposer/prompts/) | One module per pipeline phase (seed, section, facets, dedup, tree). Each holds prompt strings, prompt builders, and the `extract_*` / `assemble_*` entry points. |
| [tests/](tests/) | Unit tests + `@pytest.mark.api` integration tests that hit Together. |
| [fixtures/](fixtures/) | Real paper PDFs used as inputs for parser and integration tests. |
| [output/](output/) | Default destination for `PaperDecomposition` JSON files (written by the pipeline). Not checked in as authoritative results. |
| [config.yaml](config.yaml) | Sole runtime configuration file. Model tiers, pipeline phase settings, facet routing. |
| [pyproject.toml](pyproject.toml) | Python project metadata, dependencies, pytest config. Sets `pythonpath = ["src"]`. |
| [CLAUDE.md](CLAUDE.md) | Orientation for Claude Code sessions (commands, architecture summary). |
| [implementation.md](implementation.md) | Long-form spec of the decomposition schema, facet question sets, and pipeline design. |
| [tasks.md](tasks.md) | Tracking file for iterative validation tasks. |
| [main.py](main.py) | Placeholder script from `uv init`; not part of the pipeline. |

---

## Prerequisites

- Python `>=3.10` (see `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- A valid `TOGETHER_API_KEY` environment variable. [config.py](src/paper_decomposer/config.py) raises `ConfigError` at load time if it is missing.

---

## Install

```bash
uv sync
```

This creates `.venv/` and installs everything declared in [pyproject.toml](pyproject.toml) (`pymupdf`, `pyyaml`, `pydantic>=2.0`, `openai>=1.50`, `rich`, `tiktoken`) plus the dev group (`pytest>=8.0`).

---

## Run

**Single PDF:**

```bash
python -m paper_decomposer path/to/paper.pdf --config config.yaml
```

**Directory of PDFs (batch):**

```bash
python -m paper_decomposer path/to/dir/ --config config.yaml
```

**Dry run** (phase 0 only, prints the section table and exits):

```bash
python -m paper_decomposer path/to/paper.pdf --dry-run
```

The output is a JSON file in `pipeline.output.output_dir` (default `./output/`), named from the sanitized paper title. Pre-existing names get a numeric suffix (`_2`, `_3`, …) instead of overwriting.

---

## Pipeline at a glance

One PDF in → one [`PaperDecomposition`](src/paper_decomposer/schema.py#L418) out, via five sequential phases. See [src/paper_decomposer/README.md](src/paper_decomposer/README.md) for the detailed flow.

| Phase | Module | Model tier | Input → Output |
|-------|--------|------------|----------------|
| 0   | [pdf_parser.py](src/paper_decomposer/pdf_parser.py) | — | PDF → `PaperDocument` (sections + artifacts) |
| 0b  | [models.py](src/paper_decomposer/models.py) `preflight_model_tiers` | all required | Ping each required tier to fail fast. |
| 1   | [prompts/seed.py](src/paper_decomposer/prompts/seed.py) | `small` | Abstract → 3–7 seed `RawClaim`s |
| 2   | [prompts/section.py](src/paper_decomposer/prompts/section.py) | `small` | Each non-abstract/non-reference section → `RawClaim`s (parallel, semaphore-capped) |
| 2b  | [prompts/facets.py](src/paper_decomposer/prompts/facets.py) | `small` | Each method claim → classify intervention → domain facets + universal facets |
| 3   | [prompts/dedup.py](src/paper_decomposer/prompts/dedup.py) | `medium` | All claims → canonical claim list + groups |
| 4   | [prompts/tree.py](src/paper_decomposer/prompts/tree.py) | `heavy` | Claims + facets + negatives + artifacts → `PaperDecomposition` with tree + `OneLiner` |

---

## Tests

```bash
# Fast tests only (no network)
uv run pytest -m "not api"

# Full suite including live Together calls (requires TOGETHER_API_KEY)
uv run pytest -m api

# Single test
uv run pytest tests/test_pipeline.py::test_pipeline_aborts_when_preflight_fails
```

Integration tests are marked with `@pytest.mark.api`. See [tests/README.md](tests/README.md).

---

## Cost tracking

Every `call_model` invocation updates a process-global `_COST_TRACKER` dict in [models.py](src/paper_decomposer/models.py). `reset_cost_tracker()` is called at the start of every `decompose_paper`; the accumulated total is written into `PaperDecomposition.extraction_cost_usd` and logged at the end of the run.
