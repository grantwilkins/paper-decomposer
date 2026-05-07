"""
Claim:
Extraction sanitization demotes deterministically invalid method-family nodes
after model repair and removes dangling local references before final validation.

Plausible wrong implementations:
- Keep section-heading method nodes after repair.
- Keep methods without mechanism sentences after repair.
- Remove invalid nodes but leave edges, links, claims, or outcomes pointing to them.
- Drop useful evidence instead of preserving invalid nodes as demoted items.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedOutcome,
    ExtractedSetting,
    PaperExtraction,
)
from paper_decomposer.extraction.sanitize import demote_invalid_method_nodes
from paper_decomposer.extraction.validators import validate_extraction


def test_invalid_method_nodes_are_demoted_and_references_are_pruned() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Method",
                section_kind="method",
                text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
            )
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system",
                kind="system",
                canonical_name="Tiny",
                description="Tiny serving system.",
                granularity_rationale="The paper presents Tiny as the system.",
                evidence_span_ids=["s1"],
            ),
            ExtractedNode(
                local_node_id="bad-section",
                kind="method",
                canonical_name="Method",
                description="Section-shaped node.",
                granularity_rationale="Incorrectly promoted section heading.",
                mechanism_sentence="This sentence is long enough but the node is only a section heading.",
                evidence_span_ids=["s1"],
            ),
            ExtractedNode(
                local_node_id="bad-mechanism",
                kind="method",
                canonical_name="Cache handling",
                description="Underspecified cache handling.",
                granularity_rationale="The model failed to state a reusable mechanism.",
                evidence_span_ids=["s1"],
            ),
        ],
        edges=[
            ExtractedEdge(parent_id="system", child_id="bad-section", relation_kind="uses", evidence_span_ids=["s1"]),
            ExtractedEdge(parent_id="bad-section", child_id="bad-mechanism", relation_kind="uses", evidence_span_ids=["s1"]),
        ],
        settings=[
            ExtractedSetting(
                local_setting_id="setting-1",
                kind="workload",
                canonical_name="Serving workload",
                description="Request serving workload.",
                evidence_span_ids=["s1"],
            )
        ],
        method_setting_links=[
            ExtractedMethodSettingLink(
                method_id="bad-mechanism",
                setting_id="setting-1",
                relation_kind="applies_to",
                evidence_span_ids=["s1"],
            )
        ],
        outcomes=[
            ExtractedOutcome(
                outcome_id="outcome-1",
                paper_id="paper-1",
                metric="throughput",
                method_ids=["bad-mechanism"],
                setting_ids=["setting-1"],
                evidence_span_ids=["s1"],
            )
        ],
        claims=[
            ExtractedClaim(
                claim_id="claim-1",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                finding="Tiny improves throughput.",
                method_ids=["bad-section"],
                setting_ids=["setting-1"],
                outcome_ids=["outcome-1"],
                evidence_span_ids=["s1"],
            )
        ],
    )

    sanitized = demote_invalid_method_nodes(extraction)

    assert [node.local_node_id for node in sanitized.nodes] == ["system"]
    assert sanitized.edges == []
    assert sanitized.method_setting_links == []
    assert sanitized.outcomes[0].method_ids == []
    assert sanitized.claims[0].method_ids == []
    assert {item.name for item in sanitized.demoted_items} == {"Method", "Cache handling"}
    assert all(item.evidence_span_ids == ["s1"] for item in sanitized.demoted_items)
    assert validate_extraction(sanitized).ok


def test_named_method_matching_section_title_is_not_demoted() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="TinyAttention",
                section_kind="method",
                text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
            )
        ],
        nodes=[
            ExtractedNode(
                local_node_id="m1",
                kind="method",
                canonical_name="TinyAttention",
                description="Cache-block translation mechanism.",
                granularity_rationale="It defines a reusable cache-block translation mechanism.",
                mechanism_sentence=(
                    "Given logical cache blocks, TinyAttention outputs physical cache blocks by translating "
                    "block addresses on demand."
                ),
                evidence_span_ids=["s1"],
            )
        ],
    )

    sanitized = demote_invalid_method_nodes(extraction)

    assert [node.canonical_name for node in sanitized.nodes] == ["TinyAttention"]
    assert sanitized.demoted_items == []


def test_missing_mechanism_sentence_can_be_grounded_from_cited_evidence() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="TinyAttention",
                section_kind="method",
                text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
            )
        ],
        nodes=[
            ExtractedNode(
                local_node_id="m1",
                kind="method",
                canonical_name="TinyAttention",
                description="Cache-block translation mechanism.",
                granularity_rationale="It defines a reusable cache-block translation mechanism.",
                evidence_span_ids=["s1"],
            )
        ],
    )

    sanitized = demote_invalid_method_nodes(extraction)

    assert [node.canonical_name for node in sanitized.nodes] == ["TinyAttention"]
    assert sanitized.nodes[0].mechanism_sentence == (
        "TinyAttention maps logical cache blocks to physical cache blocks on demand."
    )
    assert sanitized.demoted_items == []
