# tests/

Pytest suite. `pyproject.toml` declares:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["api: marks tests that hit the Together API"]
```

so imports like `from paper_decomposer.pipeline import decompose_paper` resolve directly to `src/paper_decomposer/`.

## Running

```bash
# Offline / CI-friendly: skip every test that would call Together
uv run pytest -m "not api"

# Integration only (requires TOGETHER_API_KEY in env)
uv run pytest -m api

# All tests
uv run pytest

# A single test
uv run pytest tests/test_pipeline.py::test_pipeline_aborts_when_preflight_fails
```

Integration tests use `pytest.mark.skipif(not os.getenv("TOGETHER_API_KEY"), ...)` in addition to the `@pytest.mark.api` marker, so they self-skip when the key is absent even if the marker filter is not applied.

## Files

| File | Subject | Needs `TOGETHER_API_KEY`? |
|------|---------|---------------------------|
| [`test_config.py`](test_config.py) | `load_config` happy path, all three tiers present, missing-API-key error message. | No |
| [`test_config_semantics.py`](test_config_semantics.py) | Rejection of malformed YAML / non-mapping roots, per-tier values flow to the right runtime fields, `get_config` `lru_cache` behavior, `ConfigError` wrapping. | No |
| [`test_schema.py`](test_schema.py) | Every Pydantic model instantiates with a realistic example; all round-trip through `model_dump()` → reconstruction; every structured-output model yields a valid JSON Schema. | No |
| [`test_schema_semantics.py`](test_schema_semantics.py) | Enum/Literal contracts enforced, recursive `ClaimNode` typing preserved, `PaperDecomposerConfig` still accepts extra top-level keys, no shared mutable defaults. | No |
| [`test_pdf_parser.py`](test_pdf_parser.py) | Real parse of [fixtures/vllm.pdf](../fixtures/vllm.pdf), plus synthetic uniform-font fixtures built with `fitz` to exercise numbered-header detection, hyphenation cleanup, page-number stripping, min/max section-chars enforcement, artifact capture. | No |
| [`test_section.py`](test_section.py) | Section prompt builders include correct role-specific instructions, artifact rosters, seed skeletons. `@api` test extracts real claims from a vLLM method section. | Only `@api` tests |
| [`test_facets.py`](test_facets.py) | Routing tables map every `InterventionType` to the right prompt + schema + facet field. Classification + universal + domain flow produces a coherent `FacetedClaim`. `@api` tests call Together. | Only `@api` tests |
| [`test_dedup.py`](test_dedup.py) | Prompt formats numbered claim list with IDs/types. `apply_dedup` keeps only canonical claims and normalizes `parent_id` pointers. Unknown canonical IDs raise `ValueError`. Singleton groups preserve claims when dedup returns nothing. | Only `@api` tests |
| [`test_tree.py`](test_tree.py) | `build_tree_prompt` emits labeled blocks for CONTEXT / METHOD / RESULT / ASSUMPTION / NEGATIVE / EVIDENCE ARTIFACTS; method blocks carry facet summaries. `assemble_tree` uses the heavy tier and the right structured schema; fallback wiring enforces RESULT → METHOD dependency where possible. | Only `@api` tests |
| [`test_models.py`](test_models.py) | `call_model` plain-text + structured-output happy path against Together (live). | Yes (all tests `@api`) |
| [`test_models_semantics.py`](test_models_semantics.py) | With Together mocked: correct tier params picked, Together structured-output payload shape is `{"type": "json_object", "schema": ...}`, retries honor `max_retries` and exponential backoff, cost tracker uses per-1M rates from `AppSettings.raw.models` (the priced tier) not `AppSettings.model_tiers`. | No |
| [`test_pipeline.py`](test_pipeline.py) | End-to-end `@api` test that runs `decompose_paper` on [fixtures/vllm.pdf](../fixtures/vllm.pdf) and asserts tree roots ≥ 3, negatives ≥ 1, cost in (0, $0.50). Two offline tests assert the pipeline aborts on preflight failure and on zero phase-2 claims. | Only `@api` test |

## Test-authoring conventions

- **Docstrings.** Every semantics-heavy file starts with a top-of-file docstring labeled `Claim:` / `Plausible wrong implementations:`. New tests in these files should strengthen those invariants, not just cover lines.
- **Runtime config shim.** Tests that need a fake `AppSettings`-like object build a `types.SimpleNamespace` with `pipeline.<phase>` sub-namespaces. See `_test_runtime_config` in [`test_pipeline.py`](test_pipeline.py).
- **Monkeypatching the Together client.** `test_models_semantics.py` replaces the module-global `_client` / the `_get_client` factory, records request kwargs, and returns a stubbed `SimpleNamespace(choices=..., usage=...)`. Do not hit the network from non-`@api` tests.
- **API-key fixtures.** Pure-config tests use `monkeypatch.setenv("TOGETHER_API_KEY", "test-key")`; they do not make network calls.
- **Cache clearing.** `test_config_semantics.py` has an autouse fixture that calls `get_config.cache_clear()` before and after each test to prevent cross-test leakage through the `lru_cache`.
