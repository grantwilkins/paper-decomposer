"""
Claim:
Support-detail attachment and the paper-local scorecard should reflect the new
backbone: residue anchors to promoted claims, and scorecard metrics expose the
ontology-stability failure modes the rewrite is targeting.

Plausible wrong implementations:
- Leave support details unresolved even when a clear promoted anchor exists.
- Report scorecard fields from the old pipeline instead of the new invariants.
- Miss abstraction-tier or dependency-direction violations.
"""

from __future__ import annotations

from paper_decomposer.pipeline import _attach_support_details, _scorecard
from paper_decomposer.schema import AbstractionLevel, ClaimNode, ClaimType, ResultSubtype, SemanticRole, SupportDetail, SupportDetailType, SupportRelationshipType, RawClaim


def _claim(claim_id: str, claim_type: ClaimType, statement: str, source_section: str) -> RawClaim:
    return RawClaim(claim_id=claim_id, claim_type=claim_type, statement=statement, source_section=source_section)


def test_attach_support_details_remaps_to_best_claim() -> None:
    claims = [
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4.2"),
    ]
    support = SupportDetail(
        support_detail_id="SD_1",
        detail_type=SupportDetailType.implementation_fact,
        text="Logical blocks map to non-contiguous physical blocks.",
        source_section="4.2",
        anchor_claim_id=None,
        candidate_anchor_ids=[],
        relationship_type=SupportRelationshipType.implements,
        confidence=0.2,
        evidence_ids=[],
    )

    attached = _attach_support_details([support], claims)

    assert attached[0].anchor_claim_id in {"M1", "M2"}
    assert attached[0].candidate_anchor_ids
    assert attached[0].confidence >= 0.2


def test_attach_support_details_enforces_anchor_legality_and_allows_no_anchor() -> None:
    claims = [
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4.2"),
        _claim("R1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca at the same latency.", "5.2"),
    ]
    support = [
        SupportDetail(
            support_detail_id="SD_num",
            detail_type=SupportDetailType.numeric_support,
            text="vLLM reaches 2-4x higher throughput than Orca at the same latency.",
            source_section="5.2",
            anchor_claim_id="M2",
            candidate_anchor_ids=["M2"],
            relationship_type=SupportRelationshipType.measures,
            confidence=0.3,
            evidence_ids=[],
        ),
        SupportDetail(
            support_detail_id="SD_none",
            detail_type=SupportDetailType.numeric_support,
            text="The Alpaca trace contains long requests.",
            source_section="5.1",
            anchor_claim_id="M1",
            candidate_anchor_ids=["M1"],
            relationship_type=SupportRelationshipType.measures,
            confidence=0.3,
            evidence_ids=[],
        ),
    ]

    attached = _attach_support_details(support, claims)

    assert attached[0].anchor_claim_id == "R1"
    assert attached[0].candidate_anchor_ids == ["R1"]
    assert attached[1].anchor_claim_id is None
    assert attached[1].candidate_anchor_ids == []


def test_scorecard_reports_new_ontology_stability_metrics() -> None:
    promoted = [
        _claim("C1", ClaimType.context, "Fragmentation limits batch size.", "1"),
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "Another PagedAttention primitive restatement.", "4.1"),
        _claim("R1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca.", "5.2"),
    ]
    support = [
        SupportDetail(
            support_detail_id="SD_1",
            detail_type=SupportDetailType.implementation_fact,
            text="A fused CUDA kernel gathers discontinuous blocks.",
            source_section="5.1",
            anchor_claim_id="M1",
            candidate_anchor_ids=["M1"],
            relationship_type=SupportRelationshipType.implements,
            confidence=0.7,
            evidence_ids=[],
        )
    ]
    claim_tree = [
        ClaimNode(
            claim_id="C1",
            claim_type=ClaimType.context,
            abstraction_level=AbstractionLevel.problem,
            semantic_role=SemanticRole.problem,
            canonical_label="fragmentation_limits_batch_size",
            normalized_statement="fragmentation limits batch size",
            statement="Fragmentation limits batch size.",
            children=[
                ClaimNode(
                    claim_id="M1",
                    claim_type=ClaimType.method,
                    abstraction_level=AbstractionLevel.primitive,
                    semantic_role=SemanticRole.method_core,
                    canonical_label="pagedattention_pages_kv_blocks",
                    normalized_statement="pagedattention pages kv blocks",
                    statement="PagedAttention is an attention algorithm that pages KV blocks.",
                    children=[
                        ClaimNode(
                            claim_id="R1",
                            claim_type=ClaimType.result,
                            abstraction_level=AbstractionLevel.not_applicable,
                            semantic_role=SemanticRole.headline_result,
                            canonical_label="higher_throughput_than_orca",
                            normalized_statement="higher throughput than orca",
                            result_subtype=ResultSubtype.headline_result,
                            statement="vLLM reaches 2-4x higher throughput than Orca.",
                            depends_on=["M1"],
                            children=[],
                        )
                    ],
                    depends_on=["C1"],
                )
            ],
            depends_on=[],
        )
    ]

    scorecard = _scorecard(
        promoted_claims=promoted,
        support_details=support,
        claim_tree=claim_tree,
        compression_diagnostics={"assumption_candidates": 2, "assumption_rejections_as_mechanism": 1},
    )

    assert set(scorecard) == {
        "promoted_nodes_by_type",
        "concept_family_collisions",
        "abstraction_tier_violations",
        "illegal_dependency_directions",
        "implementation_detail_support_fraction",
        "assumption_mechanism_rejection_fraction",
    }
    assert scorecard["concept_family_collisions"] >= 1
    assert scorecard["abstraction_tier_violations"] >= 1
    assert scorecard["illegal_dependency_directions"] == 0
    assert scorecard["assumption_mechanism_rejection_fraction"] == 0.5
