# src/

This directory is the package root used by the project. `pyproject.toml` sets:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

so pytest can import `paper_decomposer` without needing an editable install (`uv pip install -e .`).

## Contents

| Path | What it is |
|------|------------|
| [paper_decomposer/](paper_decomposer/) | The real package. All pipeline logic, schema, and prompts live here. See [paper_decomposer/README.md](paper_decomposer/README.md). |

There is intentionally only one subdirectory. A second package named `paper_decomposer` exists at the repo root ([../paper_decomposer/](../paper_decomposer/)); it is a runtime shim that extends its `__path__` to point here, so `python -m paper_decomposer …` from the repo root resolves to this code without requiring an installed distribution.

## Import contract

- Production code: `from paper_decomposer.pipeline import decompose_paper`, `from paper_decomposer.schema import …`, etc.
- Tests: same imports, because `pythonpath = ["src"]` is applied by pytest.
- CLI: `python -m paper_decomposer` invokes `__main__.py` which calls `cli.main()`.

Do not import via the `src.paper_decomposer.*` path — that would only work when the working directory is the repo root, and would bypass the shim's `__path__` extension.
