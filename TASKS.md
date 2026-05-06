# Goal

Implement the extraction stage that turns parsed paper sections and artifacts into paper-local methods, settings, outcomes, and claims records that can be written into the existing Postgres DAG schema. The implementation should be cheap, staged, mostly text-grounded, and designed for reusable method and evidence retrieval rather than paper summarization.

## Non-goals for the First Extraction PR

- No OCR.
- No chart reconstruction.
- No visual figure decomposition.
- No full global canonicalization system.
- No exhaustive paper summarization.
- No repeated whole-paper LLM passes.
- No attempt to extract every implementation component as a method.
- No promotion of paper section headings into method nodes.
- No schema changes unless inspection shows the current schema cannot preserve required provenance.

## Inspect and Stabilize Existing Contracts

- Inspect parser output models in `src/paper_decomposer/schema.py`.
- Inspect section and artifact construction in `src/paper_decomposer/pdf_parser.py`.
- Inspect the Together client, structured-output repair behavior, and cost tracker in `src/paper_decomposer/models.py`.
- Inspect runtime configuration loading in `src/paper_decomposer/config.py`.
- Inspect the CLI entrypoint in `src/paper_decomposer/cli.py`.
- Inspect the ingest shell in `src/paper_decomposer/pipeline.py`.
- Inspect the Postgres schema and DB client in `src/paper_decomposer/db/`.
- Identify where parsed section text, captions, table text, page numbers, and artifact IDs are represented.
- Identify where DB writes should be added after parsing and validation.
- Document any contract mismatch that blocks extraction, keeping any documentation update lightweight.

## Define Extraction Data Contracts

- Add typed extraction contracts under `src/paper_decomposer/extraction/`.
- Add `EvidenceSpan` with paper, section, page, text, artifact, and source-kind provenance.
- Add `CandidateNode` for raw model-proposed systems, mechanisms, settings, scenarios, metrics, and artifacts.
- Add `ExtractedNode` for promoted paper-local method-family nodes only: systems, methods, and method categories.
- Add `ExtractedEdge` for paper-local graph relationships with endpoints, relation kind, evidence, and confidence.
- Add `ExtractedSetting` for datasets, tasks, applications, workloads, hardware, model artifacts, and metrics.
- Add `ExtractedOutcome` for explicit metric/value/baseline rows grounded in supplied text or parser-extracted table text.
- Add `ExtractedClaim` with raw text, normalized finding, claim type, linked methods/settings/outcomes, evidence span IDs, and confidence.
- Add `DemotedItem` with name, reason demoted, stored-under target, and evidence span IDs.
- Add `PaperExtraction` as the top-level paper-local JSON envelope.
- Add `ExtractionValidationError` for deterministic validation failures.
- Add `ValidationSeverity` with `error` and `warning`.
- Preserve each claim's raw text and evidence span IDs.
- Require each promoted method node to have a mechanism sentence stating inputs, outputs, and operative move.
- Require each demoted item to store the promoted node or setting where it belongs.

## Select Sections and Build Evidence Spans

- Select high-signal abstract text.
- Select introduction and contribution paragraphs.
- Select method, design, architecture, algorithm, system, and implementation-overview sections.
- Select evaluation prose and result-discussion paragraphs.
- Select conclusion text when it contains claims or limitations.
- Include captions when the parser already extracts them.
- Include parser-extracted table text when available.
- Include explicit numeric values from prose near tables or figures.
- Preserve paper ID, section title, rhetorical role, page range, paragraph index, artifact ID, and source kind.
- Chunk selected text into stable evidence spans that can be cited by model outputs and validators.
- Avoid sending the full paper repeatedly to the model.
- Add budget-aware limits for selected characters per stage.

## Implement Cheap LLM Extraction Stages

- Implement a paper metadata and contribution sketch extractor.
- Implement a problem statement span extractor.
- Implement a candidate method and setting extractor.
- Implement a method-node test classifier.
- Implement a scenario and setting separator.
- Implement a claim and outcome extractor.
- Implement a compression and claim attachment stage.
- Implement a demotion auditor.
- Combine logical extractors into 3-5 actual model calls for the normal path.
- Use cheap Together-hosted models by default.
- Use a cheap retry for malformed JSON or simple schema failures.
- Escalate only for repeated validation failure, paper-outline-shaped graphs, ambiguous claim attachment, or high-value papers.

## Validate Deterministically

- Validate JSON and Pydantic schema shape.
- Validate that every edge endpoint exists.
- Validate required fields on nodes, settings, outcomes, claims, and demoted items.
- Validate that every method node has a mechanism sentence.
- Validate that every method node has a granularity rationale.
- Validate that every claim has at least one evidence span.
- Validate that numeric values appear in evidence text when possible.
- Validate node count caps for systems papers.
- Validate that demoted items do not reappear as first-class nodes.
- Validate that the graph is not a paper section outline.
- Validate that generic categories are not method nodes unless explicitly justified.
- Validate that scenarios are usually settings, tasks, workloads, or claim contexts rather than method nodes.
- Classify validation failures as blocking errors or non-blocking warnings.
- Treat missing evidence, missing edge endpoints, invalid enums, missing method mechanism sentences, and nonexistent claim targets as blocking errors.
- Treat high node counts, unusual system node counts, and unverifiable numeric grounding from table formatting as warnings unless configured as errors.

## Write Extracted Records to the Database

- Map promoted system and method nodes to `methods`.
- Map aliases to `method_aliases`.
- Decide how paper-local nodes are namespaced before writing to shared tables.
- Do not globally deduplicate methods or settings by canonical name in the first PR.
- Preserve `paper_id`, `local_node_id`, and `extraction_run_id` for every extracted node.
- Map paper-local `uses` relationships to `method_edges.composes`.
- Map paper-local `is_a` relationships to `method_edges.is_a`.
- Reserve paper-local `refines` relationships for `method_edges.refines` when present.
- Do not map cross-family `applies_to` edges into `method_edges`.
- Represent method-setting applicability through a `method_setting_links` table, claim links, outcome context, or an explicitly documented MVP fallback.
- Map datasets, tasks, applications, workloads, hardware, model artifacts, and metrics to `settings`.
- Prefer adding `model_artifact` or `model` to `settings.kind` before DB writing.
- Map setting hierarchy and composition to `setting_edges`, translating relation kinds into schema-supported values.
- Map explicit normalized metric rows to `outcomes`.
- Store outcome evidence, original values, units, and parser provenance in outcome metadata when supported.
- Map typed claims to `claims`.
- Store claim raw text, extracted finding, linked local IDs, confidence, and evidence metadata in claim metadata when supported.
- Map claim relationships to `claim_links`.
- Preserve claim provenance in `claim_evidence`.
- Add or identify a persistence path for evidence spans attached to methods, settings, method edges, setting edges, and method-setting links.
- Prefer `evidence_spans` plus generic `evidence_links` for node and edge provenance.
- Accept metadata JSONB on methods, settings, method edges, and setting edges only as a concise MVP fallback.
- Do not drop method, setting, or edge evidence after validation.
- Keep the first implementation paper-local and avoid overbuilding cross-paper deduplication.

## Integrate the CLI

- Add a CLI flag that runs extraction after parsing.
- Add a dry-run extraction mode that prints or writes paper-local extraction JSON without DB writes.
- Add a DB-write mode that persists validated extraction output.
- Add validation failure reporting with actionable messages.
- Add a cost summary using the existing model cost tracker.
- Add configurable model-call and token or character budgets.
- Add machine-readable extraction caps for systems, methods, settings, claims, outcomes, and demoted items.
- Add configuration for cheap default extraction and optional adjudication tiers.
- Keep parse-only dry run behavior intact.

## Test Extraction Behavior

- Add contract serialization tests for extraction models.
- Add prompt output parsing tests with mocked LLM payloads.
- Add validator tests for invalid edges, missing evidence, missing mechanism sentences, excessive nodes, paper-outline graphs, generic nodes, and demoted-node promotion.
- Add schema mapping tests for methods, method edges, settings, setting edges, outcomes, claims, claim links, and claim evidence.
- Add mocked stage tests for frontmatter sketching, method DAG extraction, claim extraction, compression, repair, and demotion auditing.
- Add CLI dry-run extraction tests.
- Add `@pytest.mark.api` integration coverage for a tiny selected-section fixture.
- Add ORCA-like and vLLM-like smoke fixtures that check graph shape rather than exact exhaustive output.
- Add tests that prove OCR and visual figure interpretation are not required.

## Future Work

- Add global canonicalization across papers.
- Add cross-paper deduplication and merge adjudication.
- Add retrieval-card generation for methods, settings, outcomes, and claims.
- Add stronger model adjudication for high-value or ambiguous papers.
- Improve table extraction and table normalization.
- Add optional visual figure support after the text-grounded extractor is stable.
- Add richer evidence browsing for autonomous research agents.
