"""
Claim:
Targeted repair evidence selection sends spans tied to deterministic validation
failures and only adds graph-rebuild context when the graph itself is missing
or structurally unusable.

Plausible wrong implementations:
- Ignore failing object IDs and send arbitrary early spans.
- Include unrelated long evaluation evidence for a local node repair.
- Fail to include method/frontmatter context when the graph must be rebuilt.
- Treat missing target IDs as evidence span IDs.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedClaim,
    ExtractedNode,
    ExtractionValidationError,
    PaperExtraction,
    ValidationSeverity,
)
from paper_decomposer.extraction.evidence import select_targeted_repair_spans


def test_targeted_repair_uses_failing_object_evidence_not_unrelated_long_spans() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            _span("s1", "Method", "method", "TinyAttention maps logical blocks to physical blocks."),
            _span("s2", "Evaluation", "evaluation", "Unrelated row text. " * 400),
        ],
        nodes=[
            ExtractedNode(
                local_node_id="m1",
                kind="method",
                canonical_name="TinyAttention",
                description="Block mapping.",
                granularity_rationale="Reusable cache mapping mechanism.",
                evidence_span_ids=["s1"],
            )
        ],
    )
    errors = [
        ExtractionValidationError(
            code="method_missing_mechanism_sentence",
            message="Method node must state inputs, outputs, and operative move.",
            severity=ValidationSeverity.error,
            object_kind="node",
            object_id="m1",
        )
    ]

    selected = select_targeted_repair_spans(extraction, errors, max_chars=100)

    assert [span.span_id for span in selected] == ["s1"]


def test_targeted_repair_adds_graph_context_for_graph_wide_failures() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            _span("s1", "Abstract", "abstract", "Tiny System introduces TinyAttention.", source_kind="abstract"),
            _span("s2", "Method", "method", "TinyAttention maps logical blocks to physical blocks."),
            _span("s3", "Evaluation", "evaluation", "Unrelated row text. " * 400),
        ],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id="paper-1",
                claim_type="capability",
                raw_text="Tiny System introduces TinyAttention.",
                finding="Tiny System introduces TinyAttention.",
                evidence_span_ids=["s1"],
            )
        ],
    )
    errors = [
        ExtractionValidationError(
            code="no_graph_nodes",
            message="Extraction has claims but no promoted systems, methods, or settings.",
            severity=ValidationSeverity.error,
        )
    ]

    selected = select_targeted_repair_spans(extraction, errors, max_chars=160)

    assert [span.span_id for span in selected] == ["s1", "s2"]


def _span(
    span_id: str,
    section_title: str,
    section_kind: str,
    text: str,
    *,
    source_kind: str = "paragraph",
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        paper_id="paper-1",
        section_title=section_title,
        section_kind=section_kind,
        text=text,
        source_kind=source_kind,
    )
