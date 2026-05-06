# paper-decomposer

A pipeline that ingests ML / systems research papers (PDFs) into a Postgres-backed **methods / settings / outcomes / claims DAG**, designed to scale to hundreds of thousands of papers across the field.

This repository is in the middle of a reset. PDF parsing, the LLM client, runtime configuration, and the first paper-local extraction stage are in place. The remaining DB work is the transaction layer that persists validated extraction JSON through the schema in [src/paper_decomposer/db/schema.sql](src/paper_decomposer/db/schema.sql).

---

## Repository layout

| Path | Purpose |
|------|---------|
| [src/paper_decomposer/](src/paper_decomposer/) | Primary package source: pipeline, schema, models client, PDF parser, DB layer. |
| [src/paper_decomposer/db/](src/paper_decomposer/db/) | Postgres schema + async client for the methods/settings/outcomes/claims DAG. |
| [paper_decomposer/](paper_decomposer/) | Thin shim package at repo root so `python -m paper_decomposer …` works without an editable install. |
| [tests/](tests/) | Unit tests + `@pytest.mark.api` integration tests that hit Together. |
| [fixtures/](fixtures/) | Real paper PDFs used by parser tests. |
| [config.yaml](config.yaml) | Sole runtime configuration file: API + model tiers + PDF parser + database. |
| [pyproject.toml](pyproject.toml) | Dependencies and pytest config. Sets `pythonpath = ["src"]`. |
| [main.py](main.py) | Placeholder script from `uv init`; not part of the pipeline. |

---

## Prerequisites

- Python `>=3.10` (see `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- A valid `TOGETHER_API_KEY` environment variable. [config.py](src/paper_decomposer/config.py) raises `ConfigError` at load time if it is missing.
- For DB operations: a Postgres instance with `pgvector` and `pg_trgm` extensions available, exposed via the `PAPER_DECOMPOSER_DSN` environment variable.

---

## Install

```bash
uv sync
```

This creates `.venv/` and installs everything in [pyproject.toml](pyproject.toml): `pymupdf`, `pyyaml`, `pydantic`, `openai`, `rich`, `tiktoken`, `psycopg[binary,pool]`, `pgvector`, plus `pytest` in the dev group.

---

## Run

**Parse a PDF (no DB writes yet):**

```bash
python -m paper_decomposer path/to/paper.pdf --config config.yaml
```

**Dry run** (parse + print the section table only):

```bash
python -m paper_decomposer path/to/paper.pdf --dry-run
```

**Extraction dry run** (parse + validated paper-local extraction JSON; no DB writes):

```bash
python -m paper_decomposer path/to/paper.pdf --extract --output-json extraction.json
```

**Apply the database schema:**

```bash
psql "$PAPER_DECOMPOSER_DSN" -f src/paper_decomposer/db/schema.sql
```

---

## Storage shape

The DAG schema lives in [src/paper_decomposer/db/schema.sql](src/paper_decomposer/db/schema.sql). At a glance:

- **`methods`** — DAG nodes for the methods hierarchy (multi-parent allowed). `canonical_parent_id` selects one parent for tree-style display; `method_edges` carries the full DAG.
- **`settings`** — DAG nodes for datasets / tasks / applications / workloads / hardware / model artifacts / metrics. Same pattern as `methods`.
- **`method_setting_links`** — cross-family method-to-setting applicability. `applies_to` does not belong in `method_edges`.
- **`outcomes`** — `(paper, method, setting, metric, value, delta, baseline_method)` rows.
- **`claims`** — typed, scored claims with optional embeddings, linked to one or more methods/settings/outcomes via `claim_links`.
- **`evidence_spans` / `evidence_links`** — text-grounded provenance for paper-local nodes, edges, settings, outcomes, and claims.

Indexing: B-tree for canonical lookups, `pg_trgm` GIN for fuzzy-name lookups across aliases, HNSW (pgvector) for semantic dedup. DAG traversal uses recursive CTEs over the `*_edges` tables.

---

## Tests

```bash
# Fast tests only (no network, no live DB)
uv run pytest -m "not api"

# API tests only (live Together; requires TOGETHER_API_KEY)
uv run pytest -m api

# Full suite
uv run pytest
```

See [tests/README.md](tests/README.md) for the per-file breakdown.

---

## Cost tracking

Every `call_model` invocation updates a process-global `_COST_TRACKER` dict in [models.py](src/paper_decomposer/models.py). Per-million rates come from `config.yaml`'s `models.<tier>.input_cost_per_m` / `output_cost_per_m`. Reset and inspect with `reset_cost_tracker()` / `get_cost_tracker()`.
