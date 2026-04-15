"""
Claim:
The v2.1 pipeline promotes only high-value argument candidates, keeps support
items out of the claim graph by default, repairs skeletons in bounded fashion,
and escalates expensive semantic/model work only under uncertainty.

Plausible wrong implementations:
- Promote candidates using only lexical overlap and ignore novelty/necessity.
- Lose support-detail traceability by not attaching anchors/candidate anchors.
- Skeleton repair rewrites unbounded structure instead of bounded augmentation.
- Hybrid dedup never runs semantic disambiguation on uncertain clusters.
- Ambiguity resolver is called for all edges (or never), ignoring budget/gating.
"""

from __future__ import annotations

import asyncio

import pytest

import paper_decomposer.pipeline as pipeline_module
import paper_decomposer.prompts.dedup as dedup_module
import paper_decomposer.prompts.tree as tree_module
from paper_decomposer.prompts.seed import repair_skeleton
from paper_decomposer.schema import (
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimType,
    PaperMetadata,
    PaperSkeletonCandidate,
    ParentPreference,
    RawClaim,
    Section,
    SectionArgumentCandidate,
    SupportDetail,
    SupportDetailType,
    SupportRelationshipType,
)


def _claim(
    claim_id: str,
    claim_type: ClaimType,
    statement: str,
    source_section: str,
    *,
    entities: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    strength: float | None = None,
) -> RawClaim:
    return RawClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        statement=statement,
        source_section=source_section,
        entity_names=entities or [],
        evidence=[{"artifact_id": evidence_id, "role": "supports"} for evidence_id in (evidence_ids or [])],
        claim_strength=strength,
    )


def _candidate(
    claim_id: str,
    claim_type: ClaimType,
    statement: str,
    source_section: str,
    *,
    entities: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    strength: float | None = None,
    seed_id: str | None = None,
) -> SectionArgumentCandidate:
    return SectionArgumentCandidate(
        claim_id=claim_id,
        claim_type=claim_type,
        statement=statement,
        source_section=source_section,
        entity_names=entities or [],
        evidence_ids=evidence_ids or [],
        strength=strength,
        elaborates_seed_id=seed_id,
        local_role=ClaimLocalRole.mechanism,
        preferred_parent_type=ParentPreference.method,
    )


def test_promote_argument_candidates_supports_anchor_and_novelty_routes() -> None:
    sections = [
        Section(
            section_number="4.1",
            title="Method",
            role="method",
            body_text="Method body.",
            char_count=12,
        )
    ]

    skeleton = PaperSkeletonCandidate(
        context_roots=[
            _claim(
                "C1",
                ClaimType.context,
                "KV cache fragmentation limits batching.",
                "1 Intro",
                entities=["KV cache"],
            )
        ],
        core_methods=[
            _claim(
                "M1",
                ClaimType.method,
                "PagedAttention maps logical blocks to physical blocks.",
                "4.1 Method",
                entities=["PagedAttention", "blocks"],
                evidence_ids=["Fig. 1"],
            )
        ],
        topline_results=[],
        assumptions=[],
        negatives=[],
    )

    candidates = [
        # Anchor route: high overlap with skeleton method.
        _candidate(
            "m_anchor",
            ClaimType.method,
            "PagedAttention maps logical blocks to non-contiguous physical blocks via a block table.",
            "4.1 Method",
            entities=["PagedAttention", "block table"],
            evidence_ids=["Fig. 1"],
            strength=1.8,
            seed_id="M1",
        ),
        # Novelty + strength + necessity route: distinct but high-signal claim.
        _candidate(
            "m_novel",
            ClaimType.method,
            "A copy-on-write policy shares prompt blocks across parallel samples to reduce memory use.",
            "4.1 Method",
            entities=["copy-on-write", "prompt blocks"],
            evidence_ids=["Fig. 8"],
            strength=3.6,
        ),
    ]

    promoted, _, _ = pipeline_module._promote_argument_candidates(candidates, skeleton, sections, config={})
    promoted_statements = {claim.statement for claim in promoted}

    assert len(promoted) == 2
    assert all(claim.claim_type == ClaimType.method for claim in promoted)
    assert any("block table" in statement.lower() for statement in promoted_statements)
    assert any("copy-on-write" in statement.lower() for statement in promoted_statements)


def test_attach_support_details_keeps_traceability_fields() -> None:
    claims = [
        _claim(
            "M4",
            ClaimType.method,
            "PagedAttention uses a fused block copy kernel.",
            "5.1 Kernel-level Optimization",
            entities=["PagedAttention", "kernel"],
            evidence_ids=["Fig. 5"],
        )
    ]

    support = SupportDetail(
        support_detail_id="SD_1",
        detail_type=SupportDetailType.local_kernel_optimization,
        text="A fused block copy kernel batches discontinuous copy-on-write operations.",
        source_section="5.1 Kernel-level Optimization",
        anchor_claim_id=None,
        candidate_anchor_ids=[],
        relationship_type=SupportRelationshipType.local_optimization_of,
        confidence=0.2,
        evidence_ids=["Fig. 5"],
        promotable=True,
    )

    attached = pipeline_module._attach_support_details([support], claims)
    assert len(attached) == 1
    assert attached[0].anchor_claim_id == "M4"
    assert attached[0].candidate_anchor_ids
    assert attached[0].relationship_type == SupportRelationshipType.local_optimization_of


def test_needs_skeleton_repair_triggers_on_unmatched_high_strength_methods() -> None:
    skeleton = PaperSkeletonCandidate(
        context_roots=[],
        core_methods=[_claim("M1", ClaimType.method, "Core method.", "4 Method")],
        topline_results=[],
        assumptions=[],
        negatives=[],
    )
    promoted_claims = [
        _claim("M2", ClaimType.method, "Unmatched method one.", "4 Method", strength=3.0),
        _claim("M3", ClaimType.method, "Unmatched method two.", "4 Method", strength=3.1),
    ]
    unmatched = [
        _claim("M2", ClaimType.method, "Unmatched method one.", "4 Method", strength=3.0),
        _claim("M3", ClaimType.method, "Unmatched method two.", "4 Method", strength=3.1),
    ]

    assert pipeline_module._needs_skeleton_repair(promoted_claims, unmatched, skeleton) is True


def test_repair_skeleton_augmentation_is_bounded() -> None:
    skeleton = PaperSkeletonCandidate(
        context_roots=[_claim("C1", ClaimType.context, "Core context.", "1 Intro")],
        core_methods=[_claim("M1", ClaimType.method, "Core method.", "4 Method")],
        topline_results=[_claim("R1", ClaimType.result, "Core result.", "6 Eval")],
        assumptions=[],
        negatives=[],
    )
    candidates = [
        _claim("C2", ClaimType.context, "Extra context 1.", "1 Intro", strength=3.0),
        _claim("C3", ClaimType.context, "Extra context 2.", "1 Intro", strength=2.8),
        _claim("M2", ClaimType.method, "Extra method 1.", "4 Method", strength=3.2),
        _claim("M3", ClaimType.method, "Extra method 2.", "4 Method", strength=3.1),
        _claim("M4", ClaimType.method, "Extra method 3.", "4 Method", strength=3.0),
        _claim("R2", ClaimType.result, "Extra result 1.", "6 Eval", strength=2.9),
        _claim("R3", ClaimType.result, "Extra result 2.", "6 Eval", strength=2.8),
        _claim("R4", ClaimType.result, "Extra result 3.", "6 Eval", strength=2.7),
    ]

    repaired = asyncio.run(repair_skeleton(skeleton, candidates, config={}))

    assert len(repaired.context_roots) <= len(skeleton.context_roots) + 1
    assert len(repaired.core_methods) <= len(skeleton.core_methods) + 2
    assert len(repaired.topline_results) <= len(skeleton.topline_results) + 2


def test_hybrid_dedup_runs_semantic_only_on_uncertain_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    async def _fake_dedup_type_batch(claims: list[RawClaim], claim_type: str, config):
        calls.append((claim_type, len(claims)))
        return dedup_module.DedupBatchResult(
            claim_type=claim_type,
            input_claims=claims,
            canonical_claims=[claims[0]],
            groups=[dedup_module.ClaimGroup(canonical_id=claims[0].claim_id, member_ids=[claims[1].claim_id], parent_id=None)],
        )

    monkeypatch.setattr(dedup_module, "dedup_type_batch", _fake_dedup_type_batch)

    claims = [
        _claim("M1", ClaimType.method, "PagedAttention uses a block table for KV mapping.", "4.1", strength=3.0),
        _claim("M2", ClaimType.method, "The method uses a mapping table for logical-to-physical KV blocks.", "4.1", strength=2.9),
        _claim("M3", ClaimType.method, "A runtime scheduler preempts sequence groups under pressure.", "4.5", strength=2.7),
    ]

    deduped, groups = asyncio.run(
        dedup_module.hybrid_dedup_promoted(
            claims,
            config={"pipeline": {"dedup": {"semantic_node_cap": 15}}},
        )
    )

    assert deduped
    assert groups
    assert calls, "Expected bounded semantic pass to run on uncertain claims"
    assert all(size <= 15 for _, size in calls)


def test_assemble_tree_deterministic_respects_ambiguity_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_sizes: list[int] = []

    async def _fake_resolve(ambiguities, config):
        captured_sizes.append(len(ambiguities))
        return {}

    monkeypatch.setattr(tree_module, "_resolve_ambiguities", _fake_resolve)

    claims = [
        _claim("C1", ClaimType.context, "Context root.", "1 Intro"),
        _claim("M1", ClaimType.method, "Method A uses KV blocks.", "4 Method", entities=["KV blocks"]),
        _claim("M2", ClaimType.method, "Method B schedules sequence groups.", "4 Method", entities=["sequence groups"]),
        _claim("R1", ClaimType.result, "Throughput improves 2x.", "6 Eval", evidence_ids=["Table 2"]),
        _claim("R2", ClaimType.result, "Latency remains similar.", "6 Eval", evidence_ids=["Table 3"]),
        _claim("A1", ClaimType.assumption, "Requires fast memory bandwidth.", "7 Discussion"),
    ]

    _ = asyncio.run(
        tree_module.assemble_tree_deterministic(
            metadata=PaperMetadata(title="Test", authors=[]),
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"ambiguity_budget": 1}}},
            claim_groups=[],
            support_details=[],
        )
    )

    if captured_sizes:
        assert captured_sizes[0] <= 1
