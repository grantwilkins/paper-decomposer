"""
Claim:
Section digest extraction must separate argumentative claims from support details,
and classify support details with promotability constraints that prevent known
non-argumentative detail types from entering the main claim graph.

Plausible wrong implementations:
- Route API/framework/procedural statements into argument candidates.
- Mark non-promotable support types as promotable.
- Drop support details instead of preserving typed records.
"""

from __future__ import annotations

import asyncio

import pytest

import paper_decomposer.prompts.section as section_prompts
from paper_decomposer.schema import (
    ClaimStructuralHints,
    ClaimType,
    PaperSkeletonCandidate,
    ParentPreference,
    RawClaim,
    Section,
    SectionExtractionOutput,
    SupportDetailType,
)


def _method_claim(claim_id: str, statement: str) -> RawClaim:
    return RawClaim(
        claim_id=claim_id,
        claim_type=ClaimType.method,
        statement=statement,
        source_section="4.5 Implementation Details",
        structural_hints=ClaimStructuralHints(preferred_parent_type=ParentPreference.method),
    )


def test_extract_section_digest_routes_non_argumentative_items_to_support(monkeypatch: pytest.MonkeyPatch) -> None:
    section = Section(
        section_number="4.5",
        title="Implementation Details",
        role="method",
        body_text="Details.",
        char_count=8,
    )

    extracted = [
        _method_claim("m_api", "The frontend extends the OpenAI-compatible API interface."),
        _method_claim("m_fw", "The runtime uses NCCL, FastAPI, and PyTorch."),
        _method_claim(
            "m_proc",
            "For each decode step, the runtime selects requests, allocates blocks, fetches inputs, appends tokens, updates tables, and stores outputs.",
        ),
        _method_claim(
            "m_arg",
            "PagedAttention uses block-table indirection to reduce fragmentation at the cost of small lookup overhead.",
        ),
    ]

    async def _fake_extract(*args, **kwargs):
        _ = (args, kwargs)
        return SectionExtractionOutput(claims=extracted)

    monkeypatch.setattr(section_prompts, "extract_section_claims", _fake_extract)

    digest = asyncio.run(
        section_prompts.extract_section_digest(
            section=section,
            skeleton=PaperSkeletonCandidate(),
            artifacts=[],
            config={},
        )
    )

    argument_ids = {candidate.claim_id for candidate in digest.argument_candidates}
    support_by_id = {detail.support_detail_id: detail for detail in digest.support_details}
    support_types = {detail.detail_type for detail in digest.support_details}

    assert "m_arg" in argument_ids
    assert len(support_by_id) == 3
    assert SupportDetailType.api_surface in support_types
    assert SupportDetailType.framework_dependency in support_types
    assert SupportDetailType.procedural_step in support_types
    assert all(detail.promotable is False for detail in digest.support_details)
