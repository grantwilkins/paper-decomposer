# Goal

Implement and persist a graph-first extraction stage.

A valid extraction is not a list of claims about a paper. A valid extraction is a paper-local method/settings graph with source-grounded claims and outcomes attached to the most specific graph nodes.

The first milestone is to make papers like vLLM and ORCA produce their central graph spine from abstract + introduction + method overview text:

- vLLM -> uses PagedAttention
- ORCA -> uses iteration-level scheduling
- ORCA -> uses selective batching

Claims may be extracted only after graph nodes exist. If claims exist but no graph nodes or attachments exist, validation must fail.

## Current State

- PDF parsing into `PaperDocument` is implemented.
- Evidence span selection is implemented.
- Staged extraction contracts, prompts, and model calls are implemented.
- Deterministic extraction validation is implemented.
- Final `PaperExtraction` emits promoted graph fields under `graph`, not final `candidates`.
- Claims-only extractions fail validation with graph and attachment errors.
- Deterministic cleanup normalizes method/system IDs, collapses duplicate named settings, repairs obvious vLLM-like topology, demotes component details, and materializes explicit numeric outcomes when claims contain enough text.
- Normal compression and optional repair use cheap/reliable configured tiers; the heavy adjudication tier runs once as final cleanup before final validation.
- CLI extraction dry run is implemented with `--extract --output-json`.
- A local-ID DB write plan is implemented.
- The schema has extraction runs, evidence spans, evidence links, setting edges, method-setting links, model artifacts, and metrics.
- Experimental extraction now treats discovery as open-world and paper-local, with top-level problems, evidence classes, proposition-only claims, outcome rows, and local entity resolution tasks for later global matching.

## Remaining Persistence Work

- Implement the live DB writer that resolves local extraction IDs to database UUIDs.
- Persist papers and extraction runs.
- Persist evidence spans and resolve evidence links after target rows are inserted.
- Persist method nodes without cross-paper deduplication.
- Persist method aliases from node metadata into `method_aliases`.
- Persist method edges after translating `uses` to `composes`.
- Persist settings, including `model_artifact` and `metric`.
- Persist setting edges.
- Persist method-setting applicability through `method_setting_links`.
- Persist outcomes from multi-method and multi-setting extraction output.
- Decide whether outcomes expand to one row per method-setting pair or whether the schema should support multi-target outcome links.
- Preserve raw numeric text in outcome metadata when schema numeric fields cannot represent ranges or units cleanly.
- Persist claims and resolve claim links to methods, settings, and outcomes.
- Decide whether `claim_evidence` remains a compatibility table or whether generic `evidence_links` fully replaces it.
- Decide how live DB persistence should store top-level paper-local problems before global problem resolution exists.
- Decide whether `evidence_class` becomes a first-class DB column or remains write-plan metadata.

## Remaining Extraction Work

- Add fixture smoke expectations for experimental DeepSeek-vs-MiniMax extraction on vLLM and ORCA, checking graph shape rather than exact JSON.
- Define deterministic comparison scoring for candidate extractions: blocking errors, warning taxonomy, grounded graph completeness, and attached claim/outcome counts.
- Decide whether future DB persistence should store both experimental candidate outputs or only a selected extraction.
- Preserve grounded settings, claims, and explicit outcomes through compression and repair; the vLLM fixture should keep model artifacts, workloads, hardware, throughput claims, and memory claims when they appear in selected evidence.
- Add fixture smoke expectations that vLLM extraction retains the system -> PagedAttention -> reusable KV-cache mechanism shape, demotes implementation details, and preserves at least one grounded performance or memory claim.
- Narrow final heavy cleanup into targeted adjudication packets once the validator failure taxonomy is stable.
- Decide whether `require_numeric_grounding` should block extraction by default or only in DB-write mode.
- Add optional preflight for only the model tiers used in the current run.
- Keep visual figure extraction out of scope unless a future explicit feature adds it.
- Replace remaining vLLM-specific deterministic cleanup rules with field-general graph compression once multi-paper fixtures are available.
- Implement the global resolver pass that consumes local entity resolution tasks and assigns same_as/variant_of/distinct/uncertain decisions without mutating paper-local extraction records.
- Add a dedicated `figure_label` evidence class if component labels and plot/table labels need different downstream policies.
- Add persistence/resolution support for mechanism signature supporting concepts so block tables, reference counts, and logical/physical blocks are queryable without becoming promoted methods.

## Remaining CLI Work

- Add DB-write mode after the live DB writer exists.
- Report validation warnings separately from blocking failures.
- Surface the exact model tiers used and the total call count.
- Keep parse-only `--dry-run` independent of API credentials.

## Remaining Tests

- Add DB writer tests with a fake or temporary Postgres-compatible boundary.
- Add local-ID to UUID resolution tests for methods, settings, edges, claims, outcomes, and evidence links.
- Add API smoke coverage for a tiny selected-section fixture.
- Add ORCA-like and vLLM-like smoke tests that check graph shape, not exhaustive exact output.

## Future Work

- Global canonicalization across papers.
- Cross-paper deduplication and merge adjudication.
- Retrieval-card generation for methods, settings, outcomes, and claims.
- Better parser-extracted table normalization.
- Optional visual figure support after text-grounded extraction is stable.
