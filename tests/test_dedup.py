"""
Claim:
`build_dedup_prompt` formats a stable numbered claim list for grouping,
and `apply_dedup` keeps only canonical claims while preserving valid
parent-child group structure.

Plausible wrong implementations:
- Prompt omits claim IDs or claim types, breaking traceability.
- Prompt changes ordering/numbering, causing unstable grouping.
- apply_dedup returns all member claims instead of only canonicals.
- parent_id points to a non-canonical member and is not normalized.
- Canonical IDs missing from original claims pass silently.
- Empty dedup groups drop all claims instead of preserving originals.
- deduplicate_claims uses the wrong model tier or schema contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import paper_decomposer.prompts.dedup as dedup_prompts
from paper_decomposer.prompts.dedup import apply_dedup, build_dedup_prompt, deduplicate_claims
from paper_decomposer.schema import ClaimGroup, ClaimType, CrossTypeOutput, DedupBatchOutput, DedupOutput, RawClaim


def _claims_fixture() -> list[RawClaim]:
    return [
        RawClaim(
            claim_id="s1",
            claim_type=ClaimType.result,
            statement="vLLM achieves 2-4x throughput improvement",
            source_section="6 Evaluation",
        ),
        RawClaim(
            claim_id="s2",
            claim_type=ClaimType.result,
            statement="Our system improves throughput by 2-4x over baselines",
            source_section="6 Evaluation",
        ),
        RawClaim(
            claim_id="s3",
            claim_type=ClaimType.result,
            statement="On OPT-13B, vLLM achieves 2x over Orca",
            source_section="6.2 Basic Sampling",
        ),
        RawClaim(
            claim_id="s4",
            claim_type=ClaimType.context,
            statement="KV cache wastes memory via fragmentation",
            source_section="3 Memory Challenge",
        ),
    ]


def test_build_dedup_prompt_formats_numbered_claims() -> None:
    claims = _claims_fixture()
    messages = build_dedup_prompt(claims)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    system_text = messages[0]["content"]
    text = messages[1]["content"]

    assert '1. [s1] RESULT: "vLLM achieves 2-4x throughput improvement"' in text
    assert '2. [s2] RESULT: "Our system improves throughput by 2-4x over baselines"' in text
    assert '3. [s3] RESULT: "On OPT-13B, vLLM achieves 2x over Orca"' in text
    assert '4. [s4] CONTEXT: "KV cache wastes memory via fragmentation"' in text
    assert "Ignore claim_id naming style" in system_text
    assert "close paraphrases" in system_text


def test_build_dedup_prompt_empty_claims_boundary() -> None:
    messages = build_dedup_prompt([])
    assert len(messages) == 2
    assert "(no claims provided)" in messages[1]["content"]


def test_build_dedup_prompt_flattens_multiline_statements() -> None:
    multiline_claim = RawClaim(
        claim_id="m_multiline",
        claim_type=ClaimType.method,
        statement="KV cache uses blocks\nand indirection for mapping.",
        source_section="4.1 PagedAttention",
    )
    text = build_dedup_prompt([multiline_claim])[1]["content"]
    assert '1. [m_multiline] METHOD: "KV cache uses blocks and indirection for mapping."' in text


def test_apply_dedup_keeps_canonicals_and_group_relationships() -> None:
    claims = _claims_fixture()

    dedup_output = DedupOutput(
        groups=[
            ClaimGroup(canonical_id="s2", member_ids=["s1", "s2"], parent_id=None),
            ClaimGroup(canonical_id="s3", member_ids=["s3"], parent_id="s1"),
            ClaimGroup(canonical_id="s4", member_ids=["s4"], parent_id=None),
        ]
    )

    group_by_canonical = {group.canonical_id: group for group in dedup_output.groups}

    # s1 and s2 are grouped together.
    assert set(group_by_canonical["s2"].member_ids) == {"s1", "s2"}
    # s3 is a child of the s1/s2 group (via s1 member reference).
    assert group_by_canonical["s3"].parent_id == "s1"
    # s4 remains independent.
    assert "s4" not in group_by_canonical["s2"].member_ids

    deduplicated = apply_dedup(claims, dedup_output)
    dedup_ids = {claim.claim_id for claim in deduplicated}

    # Output keeps only canonicals: 3 claims, not 4.
    assert len(deduplicated) == 3
    assert dedup_ids == {"s2", "s3", "s4"}


def test_apply_dedup_empty_groups_returns_original_claims_copy() -> None:
    claims = _claims_fixture()
    deduplicated = apply_dedup(claims, DedupOutput(groups=[]))

    assert deduplicated == claims
    assert deduplicated is not claims


def test_apply_dedup_rejects_missing_canonical_claim() -> None:
    claims = _claims_fixture()
    dedup_output = DedupOutput(groups=[ClaimGroup(canonical_id="missing", member_ids=["s1"], parent_id=None)])

    try:
        apply_dedup(claims, dedup_output)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing canonical_id.")


def test_apply_dedup_rejects_unresolved_parent_group() -> None:
    claims = _claims_fixture()
    dedup_output = DedupOutput(
        groups=[
            ClaimGroup(canonical_id="s2", member_ids=["s1", "s2"], parent_id=None),
            ClaimGroup(canonical_id="s3", member_ids=["s3"], parent_id="nonexistent_parent"),
        ]
    )

    with pytest.raises(ValueError, match="does not resolve to a canonical group"):
        apply_dedup(claims, dedup_output)


def test_apply_dedup_prefers_section_provenance_over_seed_placeholder() -> None:
    claims = [
        RawClaim(
            claim_id="seed_r1",
            claim_type=ClaimType.result,
            statement="The method improves throughput.",
            source_section="abstract",
        ),
        RawClaim(
            claim_id="sec_r1",
            claim_type=ClaimType.result,
            statement="The method improves throughput by 2-4x on OPT-13B and LLaMA.",
            source_section="5.2 Throughput Evaluation",
            evidence=[{"artifact_id": "table_2", "role": "supports"}],
        ),
    ]

    dedup_output = DedupOutput(
        groups=[
            ClaimGroup(
                canonical_id="seed_r1",
                member_ids=["seed_r1", "sec_r1"],
                parent_id=None,
            )
        ]
    )

    deduplicated = apply_dedup(claims, dedup_output)
    assert [claim.claim_id for claim in deduplicated] == ["sec_r1"]


def test_deduplicate_claims_uses_configured_tier_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, type[Any] | None]] = []

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        calls.append((tier, response_schema))
        assert messages and messages[0]["role"] == "system"
        if response_schema is DedupBatchOutput:
            text = messages[1]["content"]
            groups = [ClaimGroup(canonical_id="s2", member_ids=["s1", "s2"], parent_id=None)]
            if "[s3]" in text:
                groups.append(ClaimGroup(canonical_id="s3", member_ids=["s3"], parent_id="s1"))
            return DedupBatchOutput(groups=groups)
        if response_schema is CrossTypeOutput:
            return CrossTypeOutput(parent_child_links=[])
        raise AssertionError(f"Unexpected response schema: {response_schema}")

    monkeypatch.setattr(dedup_prompts, "call_model", _fake_call_model)
    result = asyncio.run(
        deduplicate_claims(
            _claims_fixture(),
            config={"pipeline": {"dedup": {"model_tier": "heavy"}}},
        )
    )

    assert isinstance(result, DedupOutput)
    by_canonical = {group.canonical_id: group for group in result.groups}
    assert {"s2", "s3", "s4"} <= set(by_canonical)
    assert "s1" in by_canonical["s2"].member_ids
    assert calls
    assert all(tier == "heavy" for tier, _ in calls)
    assert any(schema is DedupBatchOutput for _, schema in calls)
    assert any(schema is CrossTypeOutput for _, schema in calls)


def test_deduplicate_claims_defaults_to_medium_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        calls.append(tier)
        if response_schema is DedupBatchOutput:
            return DedupBatchOutput(groups=[])
        if response_schema is CrossTypeOutput:
            return CrossTypeOutput(parent_child_links=[])
        raise AssertionError(f"Unexpected response schema: {response_schema}")

    monkeypatch.setattr(dedup_prompts, "call_model", _fake_call_model)
    _ = asyncio.run(deduplicate_claims(_claims_fixture(), config={"pipeline": {"dedup": {"model_tier": "bad"}}}))
    _ = asyncio.run(deduplicate_claims(_claims_fixture(), config=None))

    assert calls
    assert all(tier == "medium" for tier in calls)
