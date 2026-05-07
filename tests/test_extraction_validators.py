"""
Claim:
Extraction validation blocks semantically unsafe graph writes while allowing
explicitly reported warnings for suspicious but recoverable paper-local output.

Plausible wrong implementations:
- Accept evidence-only output as a completed extraction.
- Accept claims-only output as a completed extraction.
- Accept a node bag without a method DAG root, edges, or claims.
- Accept method nodes without mechanism sentences.
- Let edges, claims, or outcomes reference nonexistent local IDs.
- Treat demoted implementation details as first-class nodes.
- Normalize or invent numeric evidence instead of checking cited text.
- Promote paper section headings as method nodes.
"""

from __future__ import annotations

from pydantic import ValidationError

from paper_decomposer.extraction.contracts import (
    CandidateNode,
    DemotedItem,
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedOutcome,
    ExtractedSetting,
    ExtractedSettingEdge,
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
        candidates=[
            CandidateNode(
                name="vLLM",
                candidate_kind="system",
                rationale="Top-level introduced serving system.",
                evidence_span_ids=["s1"],
            ),
            CandidateNode(
                name="PagedAttention",
                candidate_kind="method",
                rationale="Reusable KV cache block mapping mechanism.",
                evidence_span_ids=["s1"],
            ),
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system",
                kind="system",
                canonical_name="vLLM",
                description="LLM serving system.",
                granularity_rationale="The paper presents vLLM as the composed serving system.",
                evidence_span_ids=["s1"],
            ),
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
        edges=[ExtractedEdge(parent_id="system", child_id="m1", relation_kind="uses", evidence_span_ids=["s1"])],
        claims=[
            ExtractedClaim(
                claim_id="c0",
                paper_id="paper-1",
                claim_type="capability",
                raw_text="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                finding="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                method_ids=["m1"],
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
    extraction.nodes[1].mechanism_sentence = None
    extraction.edges.append(ExtractedEdge(parent_id="m1", child_id="missing", relation_kind="uses", evidence_span_ids=["s1"]))

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "method_missing_mechanism_sentence" in codes
    assert "edge_endpoint_missing" in codes
    assert not report.ok


def test_validator_blocks_evidence_only_output() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="vLLM improves throughput with PagedAttention.",
            )
        ],
    )

    report = validate_extraction(extraction)

    assert "extraction_graph_missing" in {error.code for error in report.blocking_errors}


def test_validator_blocks_graph_references_without_declared_nodes() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="vLLM uses PagedAttention.",
            )
        ],
        edges=[ExtractedEdge(parent_id="system_vLLM", child_id="central_PagedAttention", relation_kind="uses", evidence_span_ids=["s1"])],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="vLLM uses PagedAttention.",
                finding="vLLM uses PagedAttention.",
                method_ids=["system_vLLM"],
                evidence_span_ids=["s1"],
            )
        ],
    )

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "graph_references_without_nodes" in codes
    assert "edge_endpoint_missing" in codes
    assert "claim_method_missing" in codes


def test_validator_blocks_claim_draft_without_graph_or_attachments() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="vLLM improves throughput by 2-4x with the same level of latency.",
            )
        ],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="vLLM improves throughput by 2-4x with the same level of latency.",
                finding="vLLM improves throughput by 2-4x.",
                evidence_span_ids=["s1"],
                confidence=0.0,
            )
        ],
    )

    report = validate_extraction(extraction)

    blocking_codes = {error.code for error in report.blocking_errors}
    warning_codes = {warning.code for warning in report.warnings}
    assert "no_graph_nodes" in blocking_codes
    assert "claims_unattached" in blocking_codes
    assert "zero_confidence_default" in warning_codes


def test_validator_blocks_node_bag_without_edges_or_claims() -> None:
    extraction = _valid_extraction()
    nodes = [
        node.model_copy(update={"local_node_id": "m2", "canonical_name": "KV Cache Manager"})
        if node.local_node_id == "m1"
        else node
        for node in extraction.nodes
    ]
    extraction.graph = extraction.graph.model_copy(
        update={
            "systems": [node for node in nodes if node.kind == "system"],
            "methods": [node for node in nodes if node.kind != "system"],
            "method_edges": [],
        }
    )
    extraction.claims = []

    report = validate_extraction(extraction)

    codes = {error.code for error in report.blocking_errors}
    assert "method_edges_missing" in codes
    assert "claims_missing" in codes


def test_validator_blocks_method_graph_without_system_root() -> None:
    extraction = _valid_extraction()
    nodes = [node for node in extraction.nodes if node.kind != "system"]
    extraction.graph = extraction.graph.model_copy(
        update={
            "systems": [],
            "methods": nodes,
            "method_edges": [],
        }
    )

    report = validate_extraction(extraction)

    assert "system_node_missing" in {error.code for error in report.blocking_errors}


def test_validator_allows_named_method_that_matches_section_title() -> None:
    extraction = _valid_extraction()
    extraction.evidence_spans[0].section_title = "PagedAttention"

    report = validate_extraction(extraction)

    assert report.ok
    assert "section_heading_promoted" not in {error.code for error in report.errors}


def test_validator_blocks_generic_section_heading_promoted_as_node() -> None:
    extraction = _valid_extraction()
    extraction.evidence_spans[0].section_title = "Method"
    extraction.nodes[0].canonical_name = "Method"

    report = validate_extraction(extraction)

    assert "section_heading_promoted" in {error.code for error in report.blocking_errors}


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


def test_validator_blocks_concrete_reusable_mechanism_demoted_for_missing_sentence() -> None:
    extraction = _valid_extraction()
    extraction.demoted_items.append(
        DemotedItem(
            name="KV block copy-on-write",
            reason_demoted="Method node lacks a grounded mechanism sentence with inputs, outputs, and operative move.",
            stored_under="m1",
            evidence_span_ids=["s1"],
        )
    )

    report = validate_extraction(extraction)

    assert "concrete_method_demoted_for_missing_mechanism" in {error.code for error in report.blocking_errors}


def test_final_extraction_serializes_graph_without_candidates() -> None:
    extraction = _valid_extraction()

    payload = extraction.model_dump(mode="json")

    assert "candidates" not in payload
    assert [node["canonical_name"] for node in payload["graph"]["systems"]] == ["vLLM"]
    assert [node["canonical_name"] for node in payload["graph"]["methods"]] == ["PagedAttention"]


def test_validator_warns_for_understructured_comparison_claim_and_zero_confidence() -> None:
    extraction = _valid_extraction()
    extraction.evidence_spans[0].text = "vLLM can sustain 2x higher request rates than Orca."
    extraction.claims[0].raw_text = "vLLM can sustain 2x higher request rates than Orca."
    extraction.claims[0].finding = "vLLM can sustain 2x higher request rates than Orca."
    extraction.claims[0].confidence = 0.0

    report = validate_extraction(extraction)

    warning_codes = {warning.code for warning in report.warnings}
    assert "claim_structured_fields_missing" in warning_codes
    assert "zero_confidence_default" in warning_codes


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


def test_validator_blocks_nonexistent_claim_outcome_setting_edge_and_method_setting_targets() -> None:
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
    extraction.graph.setting_edges.append(
        ExtractedSettingEdge(
            parent_id="set1",
            child_id="missing",
            relation_kind="specializes",
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
    assert "setting_edge_endpoint_missing" in codes
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
