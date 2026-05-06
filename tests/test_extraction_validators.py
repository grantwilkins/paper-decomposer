"""
Claim:
Extraction validation blocks semantically unsafe graph writes while allowing
explicitly reported warnings for suspicious but recoverable paper-local output.

Plausible wrong implementations:
- Accept method nodes without mechanism sentences.
- Let edges, claims, or outcomes reference nonexistent local IDs.
- Treat demoted implementation details as first-class nodes.
- Normalize or invent numeric evidence instead of checking cited text.
- Promote paper section headings as method nodes.
"""

from __future__ import annotations

from pydantic import ValidationError

from paper_decomposer.extraction.contracts import (
    DemotedItem,
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedOutcome,
    ExtractedSetting,
    PaperExtraction,
)
from paper_decomposer.extraction.validators import validate_extraction


def _valid_extraction() -> PaperExtraction:
    span = EvidenceSpan(
        span_id="s1",
        paper_id="paper-1",
        section_title="Design",
        section_kind="method",
        text="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
    )
    return PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[span],
        nodes=[
            ExtractedNode(
                local_node_id="m1",
                kind="method",
                canonical_name="PagedAttention",
                description="KV cache management mechanism.",
                granularity_rationale="It defines a reusable KV block mapping mechanism.",
                mechanism_sentence="Given logical KV blocks, PagedAttention outputs physical block addresses by mapping blocks on demand.",
                evidence_span_ids=["s1"],
            )
        ],
    )


def test_method_edges_cannot_use_cross_family_applicability_relation() -> None:
    try:
        ExtractedEdge(parent_id="m1", child_id="m2", relation_kind="applies_to", evidence_span_ids=["s1"])
    except ValidationError as exc:
        assert "applies_to" in str(exc)
    else:
        raise AssertionError("applies_to must not be accepted as a method edge relation")


def test_validator_blocks_missing_mechanism_and_missing_edge_endpoint() -> None:
    extraction = _valid_extraction()
    extraction.nodes[0].mechanism_sentence = None
    extraction.edges.append(ExtractedEdge(parent_id="m1", child_id="missing", relation_kind="uses", evidence_span_ids=["s1"]))

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "method_missing_mechanism_sentence" in codes
    assert "edge_endpoint_missing" in codes
    assert not report.ok


def test_validator_blocks_demoted_item_promoted_as_node() -> None:
    extraction = _valid_extraction()
    extraction.demoted_items.append(
        DemotedItem(
            name="PagedAttention",
            reason_demoted="Duplicate of promoted method.",
            stored_under="m1",
            evidence_span_ids=["s1"],
        )
    )

    report = validate_extraction(extraction)

    assert "demoted_item_promoted" in {error.code for error in report.blocking_errors}


def test_validator_warns_when_numeric_value_is_not_in_cited_evidence() -> None:
    extraction = _valid_extraction()
    extraction.claims.append(
        ExtractedClaim(
            claim_id="c1",
            paper_id="paper-1",
            claim_type="performance",
            raw_text="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
            finding="PagedAttention improves throughput by 2-4x.",
            value="2-4x",
            method_ids=["m1"],
            evidence_span_ids=["s1"],
        )
    )

    report = validate_extraction(extraction)

    assert report.ok
    assert "numeric_grounding_unverified" in {warning.code for warning in report.warnings}


def test_validator_blocks_nonexistent_claim_outcome_and_method_setting_targets() -> None:
    extraction = _valid_extraction()
    extraction.settings.append(
        ExtractedSetting(
            local_setting_id="set1",
            kind="model_artifact",
            canonical_name="LLaMA-13B",
            description="Evaluated language model.",
            evidence_span_ids=["s1"],
        )
    )
    extraction.method_setting_links.append(
        ExtractedMethodSettingLink(
            method_id="m1",
            setting_id="missing",
            relation_kind="applies_to",
            evidence_span_ids=["s1"],
        )
    )
    extraction.outcomes.append(
        ExtractedOutcome(
            outcome_id="o1",
            paper_id="paper-1",
            metric="throughput",
            method_ids=["missing"],
            evidence_span_ids=["s1"],
        )
    )
    extraction.claims.append(
        ExtractedClaim(
            claim_id="c1",
            paper_id="paper-1",
            claim_type="comparison",
            raw_text="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
            finding="PagedAttention is compared on LLaMA-13B.",
            outcome_ids=["missing"],
            evidence_span_ids=["s1"],
        )
    )

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "method_setting_endpoint_missing" in codes
    assert "outcome_method_missing" in codes
    assert "claim_outcome_missing" in codes


def test_validator_blocks_claim_and_outcome_from_wrong_paper() -> None:
    extraction = _valid_extraction()
    extraction.outcomes.append(
        ExtractedOutcome(
            outcome_id="o1",
            paper_id="other-paper",
            metric="throughput",
            method_ids=["m1"],
            evidence_span_ids=["s1"],
        )
    )
    extraction.claims.append(
        ExtractedClaim(
            claim_id="c1",
            paper_id="other-paper",
            claim_type="comparison",
            raw_text="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
            finding="PagedAttention is compared against baselines.",
            method_ids=["m1"],
            evidence_span_ids=["s1"],
        )
    )

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "outcome_paper_mismatch" in codes
    assert "claim_paper_mismatch" in codes
