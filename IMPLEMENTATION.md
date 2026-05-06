# Architecture Overview

The extraction stage should turn a parsed paper into a paper-local method and evidence graph before any global merge or deduplication work happens.

```text
ParsedPaper
  -> EvidenceSpan selection
  -> Cheap staged extraction
  -> Paper-local extraction JSON
  -> Deterministic validation
  -> Optional repair/adjudication
  -> DB mapping
  -> Postgres DAG records
```

The design has two layers:

- Paper-local extraction graph: the first implementation target. It should preserve the paper's reusable methods, settings, explicit outcomes, claims, and source evidence.
- Global canonical method graph: future work. It should merge equivalent concepts across papers after the paper-local extractor is stable.

The database should support reusable method and evidence retrieval, not paper summaries. Research agents should be able to ask what methods exist, what evidence supports a method, where a method was evaluated, what claims are attached to it, and which papers introduced, used, compared against, or improved on it.

# Proposed Package Layout

```text
src/paper_decomposer/extraction/
  __init__.py
  contracts.py
  evidence.py
  prompts.py
  stages.py
  validators.py
  assembler.py
  db_writer.py
```

- `contracts.py`: Pydantic models for evidence spans, method-family nodes, settings, outcomes, claims, demoted items, extraction envelopes, validation severity, and validation errors.
- `evidence.py`: section selection, artifact inclusion, evidence span chunking, and provenance preservation.
- `prompts.py`: short schema-bound prompt builders and extraction rules.
- `stages.py`: budget-aware calls into `call_model` for sketching, method extraction, claim extraction, compression, and repair.
- `validators.py`: deterministic validation of graph shape, evidence grounding, node granularity, numeric grounding, and schema integrity.
- `assembler.py`: merge stage outputs into one paper-local extraction JSON document.
- `db_writer.py`: translate validated paper-local records into the existing database schema.
- `src/paper_decomposer/pipeline.py`: extraction orchestration after `parse_pdf` and before optional DB persistence.

# Data Contracts

## EvidenceSpan

- `span_id`: stable local identifier.
- `paper_id`: paper identifier from the local extraction run or database paper row.
- `section_title`: parser section title.
- `section_kind`: parser rhetorical role or normalized extraction kind.
- `page_start`: first page for the source text when known.
- `page_end`: final page for the source text when known.
- `text`: selected text supplied to the model.
- `artifact_id`: optional parser artifact identifier.
- `source_kind`: optional source type such as `paragraph`, `table_text`, `caption`, `abstract`, `contribution`, or `conclusion`.

## ExtractedNode

- `local_node_id`: stable paper-local node identifier.
- `kind`: `system`, `method`, or `method_category`.
- `canonical_name`: normalized paper-local name.
- `aliases`: names used in the paper.
- `description`: concise text-grounded description.
- `status`: `claimed_new`, `reference`, or `uncertain`.
- `introduced_by`: paper-local introduction status or source paper mention when available.
- `granularity_rationale`: explanation for why this should be a first-class node.
- `mechanism_sentence`: required for methods; one sentence specifying inputs, outputs, and operative move.
- `evidence_span_ids`: cited evidence spans.
- `confidence`: extraction confidence.

Create method nodes only for reusable mechanisms that pass the method-node test: can one sentence specify the method's inputs, outputs, and operative move?

`PaperExtraction.nodes` should contain only method-family objects. `PaperExtraction.settings` should contain applications, tasks, datasets, workloads, hardware, model artifacts, and metrics. Keeping the two families separate matches the current database shape and avoids ambiguity during assembly and persistence.

## ExtractedEdge

- `parent_id`: local parent node identifier.
- `child_id`: local child node identifier.
- `relation_kind`: `uses`, `applies_to`, or `is_a` in paper-local JSON.
- `evidence_span_ids`: cited evidence spans.
- `confidence`: extraction confidence.

The current database allows `is_a`, `specializes`, `composes`, and `refines` for method and setting edges. Method-family relationships should use `uses`, `is_a`, or `refines`. Cross-family applicability should use `applies_to` and should not be written to `method_edges`.

## ExtractedSetting

- `local_setting_id`: stable paper-local setting identifier.
- `kind`: `application`, `task`, `dataset`, `workload`, `hardware`, `model_artifact`, or `metric`.
- `canonical_name`: normalized setting name.
- `aliases`: names used in the paper.
- `description`: concise grounded description.
- `evidence_span_ids`: cited evidence spans.
- `confidence`: extraction confidence.

The current `settings.kind` enum supports `dataset`, `task`, `application`, `workload`, and `hardware`. The DB writer should not leave model artifacts ambiguous. Prefer adding `model_artifact` or `model` to the enum before persistence so outcomes can be retrieved by model-conditioned settings.

## ExtractedOutcome

- `outcome_id`: stable local outcome identifier.
- `paper_id`: paper identifier.
- `method_ids`: related method nodes.
- `setting_ids`: related setting nodes.
- `metric`: metric name exactly enough to support retrieval.
- `value`: explicit numeric value when present.
- `delta`: explicit change when present.
- `baseline`: baseline method or system text when present.
- `comparator`: compared method, system, setting, or paper text when present.
- `units`: units when present.
- `evidence_span_ids`: cited evidence spans.
- `confidence`: extraction confidence.

Outcomes should be extracted only when values are explicit in prose or parser-extracted table text. MVP extraction may miss values that exist only inside plots.

## ExtractedClaim

- `claim_id`: stable local claim identifier.
- `paper_id`: paper identifier.
- `claim_type`: normalized type such as performance, memory, scalability, quality, capability, limitation, or comparison.
- `raw_text`: source-grounded claim text.
- `finding`: concise normalized finding.
- `metric`: metric text when applicable.
- `value`: exact value text when applicable.
- `delta`: exact delta text when applicable.
- `baseline`: baseline text when applicable.
- `comparator`: comparator text when applicable.
- `method_ids`: linked method node IDs.
- `setting_ids`: linked setting IDs.
- `outcome_ids`: linked outcome IDs.
- `evidence_span_ids`: cited evidence spans.
- `confidence`: extraction confidence.

Each claim must retain raw text and evidence span IDs. Claims should attach to the most specific valid method, setting, or outcome.

`raw_text` should copy source text from supplied evidence spans when possible. `finding` is the normalized paraphrase. Do not put paraphrases in `raw_text` unless the source is a table or caption fragment that cannot be copied cleanly.

## DemotedItem

- `name`: demoted candidate name.
- `reason_demoted`: why it failed the method-node test or belongs elsewhere.
- `stored_under`: promoted node or setting ID where the item is preserved.
- `evidence_span_ids`: cited evidence spans.

Demoted items preserve useful implementation details, scenarios, and components without making them first-class method nodes.

## ExtractionValidationError

- `message`: concise validation finding.
- `severity`: `error` or `warning`.
- `code`: stable behavior-named identifier.
- `object_kind`: affected object family when known.
- `object_id`: local object identifier when known.
- `evidence_span_ids`: related evidence spans when relevant.

Blocking errors should prevent DB writes. Warnings should be reported and may become errors through configuration.

# LLM Stage Design

The implementation should describe logical extractors but run only 3-5 actual model calls for a normal paper.

## Frontmatter and Contribution Sketch

Input:

- Title block.
- Abstract.
- Introduction.
- Contribution bullets.
- Method overview headings.

Output:

- Paper metadata.
- Central problem candidates.
- Contribution spans.
- Candidate systems, methods, settings, and scenarios.
- High-level central primitive guess.

Model tier: cheap.

## Method DAG Candidate Extraction

Input:

- High-signal method, design, architecture, algorithm, system, and implementation-overview sections.
- Frontmatter and contribution sketch output.

Output:

- Candidate method nodes.
- Candidate setting nodes.
- Method-node-test decisions.
- Demoted items.
- Draft edges.

Model tier: cheap.

## Claims and Outcomes Extraction

Input:

- Evaluation sections.
- Conclusion.
- Parser-extracted table text and captions when available.
- Candidate node names from prior stages.

Output:

- Claims.
- Outcomes.
- Metrics.
- Baselines.
- Settings.
- Evidence spans.

Model tier: cheap.

## Compression and Claim Attachment

Input:

- Prior stage outputs.
- Extraction rules.
- Method-node test.
- Spine-plus-adapters guidance.

Output:

- Final paper-local extraction JSON.
- Claims attached to the most specific valid node.
- Demoted items preserved.
- Graph compressed away from paper-outline structure.

Model tier: cheap or medium.

## Repair and Adjudication

Run only when deterministic validation fails.

- Retry malformed JSON or simple schema failure with the cheap model once.
- Use a medium model only for repeated validation failure, oversized graphs, paper-outline-shaped graphs, ambiguous claim attachment, or high-value papers.

# Prompting Strategy

Prompts should be short, explicit, and schema-bound.

Hard rules:

- Return JSON only.
- Do not invent evidence.
- Do not promote generic categories.
- Do not create nodes for paper sections.
- Do not create nodes for implementation details unless they pass the method-node test.
- Claim `raw_text` fields must copy source text from supplied evidence spans when possible.
- Use `finding` for normalized claim paraphrases.
- Numeric values should be preserved exactly.
- Ranges should not be collapsed.
- Scenarios should usually be settings, tasks, workloads, or claim contexts rather than methods.
- Prefer a spine-plus-adapters graph rather than a paper outline.

Spine-plus-adapters structure:

```text
system
`-- central primitive
    |-- reusable supporting mechanism
    |-- reusable supporting mechanism
    |-- applies_to scenario
    `-- applies_to scenario
```

Avoid separate method nodes for every scenario. For example, prefer a method node such as `block-level KV cache sharing` with scenarios like parallel sampling, beam search, and shared-prefix prompting.

# Validation Strategy

Deterministic validators should run before any DB write and before escalation to a stronger model.

Checks:

- Every edge endpoint exists.
- Every claim has at least one evidence span.
- Every method node has a mechanism sentence.
- Every method node has a granularity rationale.
- Every numeric value appears in evidence text when possible.
- Node count is not suspiciously high.
- Paper section headings are not method nodes.
- Demoted items do not also appear as promoted nodes.
- Generic categories are not method nodes unless explicitly justified.
- `system` node count is usually 1-2.
- Landmark systems papers usually have 5-10 method nodes, not 30 or more.
- Scenario variants are not promoted into separate method nodes unless each passes the method-node test.
- Claims do not rely on visual impressions from figures.

Blocking errors:

- Missing evidence span.
- Missing edge endpoint.
- Invalid enum value.
- Method missing mechanism sentence.
- Claim references nonexistent node, setting, or outcome.

Warnings:

- Method node count is high.
- System node count is unusual.
- Numeric grounding cannot be verified because table text formatting changed.
- Claim attachment is valid but broad.

# DB Mapping

The first implementation should map validated paper-local extraction output into the existing schema without assuming a global canonicalization system.

- `system` and `method` nodes map to `methods`.
- Method aliases map to `method_aliases`.
- Paper-local `uses` relationships map to `method_edges.composes`.
- Paper-local `is_a` relationships map to `method_edges.is_a`.
- Paper-local `refines` relationships map to `method_edges.refines`.
- Paper-local `applies_to` relationships do not map to `method_edges`.
- Method-setting applicability maps to `method_setting_links`.
- `application`, `task`, `dataset`, `workload`, `hardware`, `model_artifact`, and `metric` map to `settings`.
- Setting relationships map to `setting_edges` after translating paper-local relations into schema-supported relations.
- Explicit normalized metric rows map to `outcomes`.
- Typed claims map to `claims`.
- Claim relationships map to `claim_links`.
- Claim provenance maps to `claim_evidence`.

The schema includes `extraction_runs`, `evidence_spans`, `evidence_links`, and metadata JSON columns on methods, method edges, settings, setting edges, method-setting links, outcomes, and claims. Do not validate evidence and then discard method, setting, or edge provenance.

## Paper-local versus Global IDs

The first extraction PR writes paper-local graph objects. A `canonical_name` is canonical only within a paper extraction run. It must not be used for cross-paper deduplication. Global entity resolution is future work.

Each persisted extraction object should preserve:

- `paper_id`.
- `extraction_run_id`.
- `local_node_id` or local object ID.
- Source evidence span IDs.
- Extraction confidence.

Do not implement `upsert_method(canonical_name)` or `upsert_setting(kind, canonical_name)` in a way that silently merges unrelated concepts across papers.

# Cost Controls

Cost efficiency is a first-class requirement.

- Never send the entire PDF repeatedly.
- Select high-signal sections first.
- Limit maximum characters or tokens per stage.
- Cap model calls per paper.
- Use cheap Together-hosted models by default.
- Retry malformed JSON once with the cheap model.
- Escalate only on validation failure, ambiguous attachment, oversized graph shape, or high-value papers.
- Track cost using the existing cost tracker in `models.py`.
- Expose cost summary in the CLI.
- Make model tiers configurable in `config.yaml`.

Suggested configuration:

```yaml
extraction:
  enabled: true
  max_model_calls_per_paper: 5
  max_input_chars_per_stage: 50000
  default_model_tier: cheap
  adjudication_model_tier: medium
  enable_visual_figure_extraction: false
  include_captions: true
  include_table_text: true
  require_numeric_grounding: true
  caps:
    max_system_nodes: 2
    max_method_nodes: 12
    max_setting_nodes: 30
    max_claims: 25
    max_outcomes: 40
    max_demoted_items: 40
```

If the repository continues to use `small`, `medium`, and `heavy` as model tier names, map `cheap` to the configured `small` tier or rename the suggested key before implementation.

# Figure and Table Policy

- Captions are allowed.
- Parser-extracted table text is allowed.
- Prose around figures is allowed.
- Explicit numeric values from nearby prose are allowed.
- Visual chart interpretation is out of scope.
- OCR is out of scope.
- Reconstruction of plotted curves is out of scope.
- Missing plot-only numbers is acceptable for MVP extraction.
- Claims must not be invented from visual impressions.

# Example Calibration

These examples calibrate graph shape. They are not hardcoded expected outputs.

ORCA-style decomposition:

```text
ORCA
|-- uses iteration-level scheduling
|   |-- uses request-pool based iteration admission
|   `-- uses pipeline-preserving iteration scheduling
`-- uses selective batching
    |-- uses per-request Attention execution
    `-- uses batched non-Attention execution
```

vLLM-style decomposition:

```text
vLLM
`-- uses PagedAttention
    |-- uses block-wise KV cache address translation
    |-- uses on-demand KV block allocation
    |-- uses block-level KV cache sharing
    |   `-- uses KV block copy-on-write
    `-- uses sequence-group preemption
```

Useful granularity examples:

- `Scheduling` is too broad.
- `Iteration-level scheduling` is a valid method.
- `Memory management` is too broad.
- `PagedAttention` is a valid method.
- `Block-wise KV cache address translation` is a valid method.
- `Fused block copy kernel` is usually an implementation detail unless the paper presents it as a reusable mechanism with clear inputs, outputs, and operative move.

# Testing Strategy

The tests should be small, behavior-named, and grounded in concrete extraction responsibilities.

- Contract serialization tests for evidence spans, nodes, edges, settings, outcomes, claims, demoted items, and top-level extraction JSON.
- Prompt output parsing tests for valid JSON, wrapped JSON, malformed JSON repair, and schema rejection.
- Validator tests for missing edge endpoints, missing evidence spans, missing mechanism sentences, missing granularity rationales, suspicious node counts, paper-outline nodes, generic category nodes, numeric grounding failures, and demoted item promotion.
- Mocked stage tests for frontmatter sketching, method DAG extraction, claim extraction, compression, repair, and demotion auditing.
- DB mapping tests for method rows, method aliases, method edges, setting rows, setting edges, outcome rows, claim rows, claim links, and claim evidence.
- CLI dry-run extraction tests that require no database and no live API.
- `@pytest.mark.api` integration test with a tiny selected-section fixture.
- ORCA-like and vLLM-like smoke tests that check graph shape rather than exact exhaustive output.

Smoke expectations:

- The graph is not a paper outline.
- The graph has a system node.
- The graph has central method nodes.
- Claims are grounded in evidence spans.
- Demoted implementation details are preserved.
- Visual figure extraction is not required.

# Definition of Done

The extraction-stage planning docs are complete when a future coding agent can answer:

- What code needs to be added?
- What are the data contracts?
- How many LLM calls should a paper usually need?
- Which model tiers should be used?
- How are claims grounded?
- How are method nodes distinguished from settings, scenarios, and components?
- How does extracted JSON map to the existing Postgres schema?
- What is explicitly out of scope?
- How will this be tested without spending much money?

The implementation that follows these docs should produce a cheap, staged, mostly text-grounded extractor using 3-5 model calls for normal papers plus deterministic validators.
