# output/

Default destination for pipeline output JSON files. Controlled by `config.yaml → pipeline.output.output_dir` (see [../config.yaml](../config.yaml)); the built-in default is `./output`.

## What gets written here

For every successful `decompose_paper(pdf_path, config_path)` run, one JSON file is written:

- **Filename.** Sanitized from `PaperDocument.metadata.title` — whitespace collapsed, non-`[A-Za-z0-9._ -]` characters stripped, truncated to 60 characters, spaces replaced with `_`. If that name is empty, `pdf_path.stem` is used. If the name already exists, a numeric suffix is appended (`_2`, `_3`, …) — existing files are **never** overwritten. See `_sanitize_filename` / `_resolve_output_dir` in [../src/paper_decomposer/pipeline.py](../src/paper_decomposer/pipeline.py).
- **Contents.** A [`PaperDecomposition`](../src/paper_decomposer/schema.py#L418) serialized via `model_dump_json(indent=2)`:
  - `metadata` — `PaperMetadata` (title, authors, venue, year, doi).
  - `one_liner` — `{achieved, via, because}` from phase 4, or `"UNSPECIFIED"` fallbacks.
  - `claim_tree` — list of root `ClaimNode`s. Each node has `claim_id`, `claim_type`, `statement`, `evidence`, `facets` (on METHOD nodes only), `children` (recursive), `depends_on`, and negative fields (`rejected_what`, `rejected_why`) when applicable.
  - `negative_claims` — flat list of NEGATIVE `RawClaim`s.
  - `all_artifacts` — every `EvidenceArtifact` (figures/tables/equations) extracted from the PDF in phase 0.
  - `extraction_cost_usd` — cumulative Together spend for this run, computed from the per-tier rates in `config.yaml`.

## Current contents

Files in this directory are local artifacts, not canonical results. They persist across runs because of the numeric-suffix policy. Purge manually when they stop being useful:

```bash
rm output/*.json
```

## Git status

This directory is tracked so pipeline runs on a fresh clone have somewhere to write. Individual JSON outputs are typically not committed — treat new files here as untracked by default and only commit a specific output if you need to reference it from documentation or a test.

## Troubleshooting

- **Empty or missing file after a run.** Check the CLI output; the pipeline prints `Saved decomposition to <path>` on success. Failure at phase 4 still writes a file using `_build_fallback_decomposition` (deterministic wiring), so a missing file means failure before phase 4 or a write-time exception.
- **Unexpected `_<n>.json` suffixes.** That's the anti-overwrite policy, not a bug. Delete the older copies if they're stale.
- **Cost field suspiciously low.** The cost tracker is reset at the start of every `decompose_paper` call. If you call internal pipeline phases directly (e.g., from a script), reset it yourself with `models.reset_cost_tracker()` beforehand.
