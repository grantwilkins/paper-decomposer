"""
Claim:
Section digest extraction should stay recall-oriented for argument candidates
while routing obvious implementation residue into typed support details.

Plausible wrong implementations:
- Drop core context/method/result candidates because they are not high-confidence.
- Let API/framework/procedural/kernel residue into the promoted candidate lane.
- Preserve the removed late-promotion field on support details.
"""

from __future__ import annotations

import asyncio

import pytest

import paper_decomposer.prompts.section as section_prompts
from paper_decomposer.schema import ClaimStructuralHints, ClaimType, PaperSkeletonCandidate, ParentPreference, RawClaim, Section, SectionExtractionOutput, SupportDetailType


def _method_claim(claim_id: str, statement: str, *, role: str | None = None) -> RawClaim:
    hints = ClaimStructuralHints(preferred_parent_type=ParentPreference.method)
    if role is not None:
        hints = hints.model_copy(update={"local_role": role})
    return RawClaim(
        claim_id=claim_id,
        claim_type=ClaimType.method,
        statement=statement,
        source_section="4.5 Implementation Details",
        structural_hints=hints,
    )


def test_extract_section_digest_routes_obvious_residue_to_support(monkeypatch: pytest.MonkeyPatch) -> None:
    section = Section(section_number="4.5", title="Implementation Details", role="method", body_text="Details.", char_count=8)
    extracted = [
        _method_claim("m_api", "The frontend exposes an OpenAI-compatible API interface."),
        _method_claim("m_fw", "The runtime uses NCCL, FastAPI, and PyTorch."),
        _method_claim("m_proc", "For each decode step, the runtime selects requests, allocates blocks, updates tables, and stores outputs."),
        _method_claim(
            "m_system",
            "vLLM is a serving engine with a centralized scheduler and a KV cache manager shared by GPU workers.",
        ),
        _method_claim(
            "m_blocks",
            "vLLM maps logical KV blocks to physical KV blocks via a block table in GPU workers.",
            role="implementation_detail",
        ),
        _method_claim("m_kernel", "A fused CUDA kernel gathers discontinuous blocks with coalesced reads.", role="implementation_detail"),
        _method_claim("m_core", "PagedAttention reduces fragmentation through paged KV allocation."),
        RawClaim(claim_id="c1", claim_type=ClaimType.context, statement="Fragmentation limits batch size.", source_section="1 Intro"),
        RawClaim(claim_id="r1", claim_type=ClaimType.result, statement="The system reaches 2x higher throughput.", source_section="5 Eval"),
    ]

    async def _fake_extract(*args, **kwargs):
        _ = (args, kwargs)
        return SectionExtractionOutput(claims=extracted)

    monkeypatch.setattr(section_prompts, "extract_section_claims", _fake_extract)
    digest = asyncio.run(section_prompts.extract_section_digest(section, PaperSkeletonCandidate(), [], {}))

    argument_ids = {candidate.claim_id for candidate in digest.argument_candidates}
    support_types = {detail.detail_type for detail in digest.support_details}

    assert argument_ids == {"m_core", "m_system", "c1", "r1"}
    assert support_types == {
        SupportDetailType.api_surface,
        SupportDetailType.framework_dependency,
        SupportDetailType.implementation_fact,
        SupportDetailType.procedural_step,
        SupportDetailType.local_kernel_optimization,
    }
    assert all(not hasattr(detail, "promotable") for detail in digest.support_details)
