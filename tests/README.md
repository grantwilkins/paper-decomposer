# tests/

Pytest suite. `pyproject.toml` declares:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["api: marks tests that hit the Together API"]
```

so imports like `from paper_decomposer.pipeline import ingest_paper` resolve directly to `src/paper_decomposer/`.

## Running

```bash
# Offline / CI-friendly: skip every test that would call Together
uv run pytest -m "not api"

# Integration only (requires TOGETHER_API_KEY in env)
uv run pytest -m api

# All tests
uv run pytest
```

Integration tests use `pytest.mark.skipif(not os.getenv("TOGETHER_API_KEY"), ...)` in addition to the `@pytest.mark.api` marker, so they self-skip when the key is absent.

## Files

| File | Subject | Needs `TOGETHER_API_KEY`? |
|------|---------|---------------------------|
| [`test_config.py`](test_config.py) | `load_config` happy path, all three tiers present, medium extraction default, lower-output-cost small tier, missing-API-key error message. | No |
| [`test_config_semantics.py`](test_config_semantics.py) | Rejection of malformed YAML / non-mapping roots, per-tier values flow to the right runtime fields, `get_config` `lru_cache` behavior, `ConfigError` wrapping. | No |
| [`test_pdf_parser.py`](test_pdf_parser.py) | Real parse of [fixtures/vllm.pdf](../fixtures/vllm.pdf), plus synthetic uniform-font fixtures built with `fitz` to exercise numbered-header detection, hyphenation cleanup, page-number stripping, min/max section-chars enforcement, artifact capture. | No |
| [`test_extraction_cli.py`](test_extraction_cli.py) | CLI extraction JSON output remains opt-in and does not require a database. | No |
| [`test_extraction_db_write_plan.py`](test_extraction_db_write_plan.py) | Local-ID DB write plans preserve paper-local IDs, keep `applies_to` out of method edges, and reject blocking validation failures. | No |
| [`test_extraction_evidence.py`](test_extraction_evidence.py) | Evidence span selection is stable, high-signal, caption-aware, and does not fabricate unavailable section pages. | No |
| [`test_extraction_repair.py`](test_extraction_repair.py) | Extraction repair runs once after blocking validation failures, respects model-call budgets, and includes validation context in repair prompts. | No |
| [`test_extraction_sanitize.py`](test_extraction_sanitize.py) | Sanitization demotes invalid repaired method nodes, normalizes local IDs, deduplicates settings, materializes outcomes, and prunes dangling local references. | No |
| [`test_extraction_stages.py`](test_extraction_stages.py) | Extraction stages map `cheap` to `small`, keep compression/repair on configured cheap tiers, and include evidence span IDs in prompts. | No |
| [`test_extraction_validators.py`](test_extraction_validators.py) | Deterministic validation catches evidence-only outputs, node bags without edges or claims, missing mechanism sentences, bad endpoints, promoted demoted items, paper mismatches, and numeric grounding issues. | No |
| [`test_models.py`](test_models.py) | `call_model`, cost tracking, and model preflight happy paths against Together (live). | Yes (all tests `@api`) |
| [`test_models_semantics.py`](test_models_semantics.py) | With Together mocked: correct tier params picked, Together structured-output payload shape is `{"type": "json_object", "schema": ...}`, retries honor `max_retries` and exponential backoff, cost tracker uses per-1M rates from `AppSettings.raw.models`. | No |
| [`test_db_schema.py`](test_db_schema.py) | Schema-file smoke test: required Postgres extensions are declared, every required table has a `CREATE TABLE` statement. Catches typos and accidental file moves; does not require a live DB. | No |

## Conventions

- **Monkeypatching the Together client.** `test_models_semantics.py` replaces the module-global `_client` / the `_get_client` factory and returns stubbed responses. Do not hit the network from non-`@api` tests.
- **API-key fixtures.** Pure-config tests use `monkeypatch.setenv("TOGETHER_API_KEY", "test-key")`; they do not make network calls.
- **Cache clearing.** `test_config_semantics.py` has an autouse fixture that calls `get_config.cache_clear()` before and after each test to prevent cross-test leakage through the `lru_cache`.
