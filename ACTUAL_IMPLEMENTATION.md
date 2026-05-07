# Actual Implementation

This repo currently implements a local PDF-to-extraction pipeline for research
papers. It parses a PDF, selects evidence spans, asks configured LLM tiers for a
paper-local methods/settings/outcomes/claims graph, then runs deterministic
cleanup and validation before emitting JSON.

It is not yet a complete Postgres ingester. The schema and DB write-plan builder
exist, but the live transaction layer that resolves local IDs to database UUIDs
and persists validated extractions is still future work.

## Current Data Flow

```text
PDF
  -> parse_pdf()
  -> PaperDocument(metadata, sections, artifacts)
  -> select_evidence_spans()
  -> LLM extraction draft
  -> sanitize + attach + validate
  -> PaperExtraction JSON
```

The default `config.yaml` extraction strategy is `big_model_draft`: one compact
medium-tier draft call, deterministic validation, and at most one targeted repair
call. The older staged strategy is still implemented and runs frontmatter,
method-graph, claims/outcomes, compression, optional repair, and optional heavy
cleanup calls.

## Main Entry Points

- `python -m paper_decomposer <pdf>` parses a PDF and prints parse counts.
- `--dry-run` parses and prints a section table.
- `--extract --output-json out.json` writes a validated extraction JSON for one
  PDF.
- `--experimental-big-model-compare --output-json comparison.json` runs the
  configured candidate tiers and writes per-tier extraction, validation, and cost
  results.
- `src/paper_decomposer/pipeline.py` is the main orchestration layer:
  `ingest_paper`, `extract_paper`, `extract_document`, and
  `compare_paper_extractions`.

## Core Objects

- `PaperDocument`: parser output with metadata, sections, and artifacts.
- `EvidenceSpan`: normalized text evidence with span IDs, section context,
  optional page/artifact provenance, source kind, and evidence class.
- `PaperExtraction`: final paper-local artifact containing evidence spans,
  `PaperGraph`, problems, outcomes, claims, and demoted items.
- `PaperGraph`: systems, methods, method edges, settings, setting edges, and
  method-setting links.
- `ExtractedClaim` and `ExtractedOutcome`: grounded findings and measurements
  attached back to graph nodes and evidence spans.

## Code Map

| Path | What it does |
| --- | --- |
| `src/paper_decomposer/pdf_parser.py` | Uses PyMuPDF to turn PDFs into `PaperDocument` sections and artifacts. |
| `src/paper_decomposer/models.py` | Wraps Together's OpenAI-compatible API, structured JSON output, retry behavior, and cost tracking. |
| `src/paper_decomposer/extraction/evidence.py` | Chooses high-signal paper spans and targeted repair context. |
| `src/paper_decomposer/extraction/prompts.py` | Builds the compact JSON-only extraction prompts. |
| `src/paper_decomposer/extraction/stages.py` | Defines intermediate draft contracts and model-call wrappers. |
| `src/paper_decomposer/extraction/assembler.py` | Converts model drafts into `PaperExtraction`. |
| `src/paper_decomposer/extraction/sanitize.py` | Performs deterministic cleanup, graph preservation, ID normalization, demotions, and claim/outcome repairs. |
| `src/paper_decomposer/extraction/validators.py` | Produces blocking errors and warnings for malformed or weak extractions. |
| `src/paper_decomposer/extraction/db_write_plan.py` | Builds unresolved local-ID rows for a future DB writer. |
| `src/paper_decomposer/db/schema.sql` | Defines the Postgres DAG schema, evidence links, claims, outcomes, and indexes. |

## Validation Model

A usable extraction must have a real graph, grounded claims, valid evidence
references, valid local-node references, and no blocking validation errors.
Warnings are allowed and are used to flag quality issues such as too many nodes,
missing page provenance, noisy evidence, or weak graph shape.

The comparison command can intentionally return `ok=false` results. That path is
for diagnosing model behavior, so it records the invalid extraction and the full
validation report instead of raising immediately.

## Storage State

Implemented:

- Postgres schema for papers, methods/settings DAGs, evidence, claims, outcomes,
  and links.
- Async DB client that can apply the schema.
- DB write-plan builder that refuses extractions with blocking validation errors.

Not implemented yet:

- Local-ID to UUID resolution.
- Upserts/transactions for persisting a full validated extraction.
- Global cross-paper deduplication and entity resolution.

## Known Limits

- Extraction quality is still model-dependent; deterministic validators catch
  many bad outputs but do not make every model draft good.
- Figure and table handling is text-based. The system does not infer visual plot
  data when the parser only exposes partial captions or labels.
- Many section-derived spans do not yet carry page provenance.
- `comparison_*.json` files are diagnostic artifacts, not persisted database
  records.
