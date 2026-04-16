"""
Claim:
Section extraction keeps argument-level claims and suppresses low-worth
implementation inventory, API-surface descriptions, and evaluation restatements.

Plausible wrong implementations:
- Keep implementation inventory claims because they contain method verbs.
- Keep framework/API usage statements even when they have no argumentative force.
- Keep evaluation-section mechanism restatements that are not findings.
- Apply filtering only before retagging and let low-worth claims survive afterward.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import paper_decomposer.prompts.section as section_prompts
from paper_decomposer.config import load_config
from paper_decomposer.pdf_parser import parse_pdf
from paper_decomposer.prompts.seed import extract_seed
from paper_decomposer.prompts.section import (
    BACKGROUND_INSTRUCTIONS,
    DISCUSSION_INSTRUCTIONS,
    UNKNOWN_SECTION_INSTRUCTIONS,
    build_section_prompt,
    extract_section_claims,
)
from paper_decomposer.schema import (
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimType,
    EvidenceArtifact,
    FlatClaim,
    FlatSectionOutput,
    ParentPreference,
    PaperDocument,
    RhetoricalRole,
    RawClaim,
    Section,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
FIXTURE_PDF = ROOT / "fixtures" / "vllm.pdf"

requires_api_key = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY is not set.",
)


def _find_method_section(document: PaperDocument) -> Section:
    paged_attention = [
        section
        for section in document.sections
        if "pagedattention" in section.title.lower() or "pagedattention" in section.body_text.lower()
    ]
    if paged_attention:
        return max(paged_attention, key=lambda section: section.char_count)

    method_like = [
        section
        for section in document.sections
        if section.role in {RhetoricalRole.method, RhetoricalRole.theory, RhetoricalRole.other}
    ]
    if not method_like:
        raise AssertionError("Could not find a method-like section in parsed document.")
    return max(method_like, key=lambda section: section.char_count)


def _find_evaluation_section(document: PaperDocument) -> Section:
    eval_sections = [section for section in document.sections if section.role == RhetoricalRole.evaluation]
    if eval_sections:
        return max(eval_sections, key=lambda section: section.char_count)
    raise AssertionError("Could not find an evaluation section in parsed document.")


def _get_abstract_text(document: PaperDocument) -> str:
    abstracts = [section.body_text for section in document.sections if section.role == RhetoricalRole.abstract]
    if abstracts:
        return "\n\n".join(abstracts)
    if not document.sections:
        raise AssertionError("Parsed document had no sections.")
    return document.sections[0].body_text


def test_extract_section_claims_accepts_empty_flat_claim_list(monkeypatch: pytest.MonkeyPatch) -> None:
    section = Section(
        section_number="3.7",
        title="Miscellaneous Notes",
        role=RhetoricalRole.other,
        body_text="This section is mostly transition text.",
        char_count=43,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        assert messages[0]["content"] == UNKNOWN_SECTION_INSTRUCTIONS
        return FlatSectionOutput(claims=[])

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    output = asyncio.run(extract_section_claims(section, [], [], {"pipeline": {"section_extraction": {}}}))
    assert output.claims == []


def test_extract_section_claims_normalizes_flat_claim_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    section = Section(
        section_number="4.1",
        title="PagedAttention",
        role=RhetoricalRole.method,
        body_text="PagedAttention maps logical blocks to physical GPU blocks.",
        char_count=58,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="m4_1",
                    claim_type=ClaimType.method.value,
                    statement="PagedAttention uses block-table indirection.",
                    evidence_ids=["Fig. 1", "Fig. 1", "Table 2"],
                    entity_names=["PagedAttention", "KV cache manager"],
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [
        EvidenceArtifact(artifact_type="figure", artifact_id="Fig. 1", caption="Kernel layout.", source_page=4),
        EvidenceArtifact(artifact_type="table", artifact_id="Table 2", caption="Throughput.", source_page=8),
    ]
    output = asyncio.run(extract_section_claims(section, [], artifacts, {"pipeline": {"section_extraction": {}}}))

    assert len(output.claims) == 1
    claim = output.claims[0]
    assert claim.claim_id == "m4_1"
    assert claim.claim_type == ClaimType.method
    assert claim.source_section == "4.1 PagedAttention"
    assert [pointer.artifact_id for pointer in claim.evidence] == ["Fig. 1", "Table 2"]
    assert claim.entity_names == ["PagedAttention", "KV cache manager"]


def test_extract_section_claims_retags_eval_method_like_claim_and_sets_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="5.2",
        title="Throughput",
        role=RhetoricalRole.evaluation,
        body_text="vLLM achieves 2-4x throughput gains over baselines on OPT models.",
        char_count=72,
    )
    seed_claims = [
        RawClaim(
            claim_id="s_method",
            claim_type=ClaimType.method,
            statement="PagedAttention uses block indirection for KV cache management.",
            source_section="Abstract",
            structural_hints=ClaimStructuralHints(
                local_role=ClaimLocalRole.top_level,
                preferred_parent_type=ParentPreference.context,
            ),
        )
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        _ = (tier, messages, response_schema, config)
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="x1",
                    claim_type="method",
                    statement="vLLM achieves 2-4x higher throughput than baselines.",
                    source_section="",
                    elaborates_seed_id="s_method",
                    local_role="mechanism",
                    preferred_parent_type="method",
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    output = asyncio.run(extract_section_claims(section, seed_claims, [], {"pipeline": {"section_extraction": {}}}))

    assert len(output.claims) == 1
    claim = output.claims[0]
    assert claim.claim_type == ClaimType.result
    assert claim.source_section == "5.2 Throughput"
    assert claim.structural_hints is not None
    assert claim.structural_hints.local_role == ClaimLocalRole.mechanism
    assert claim.structural_hints.preferred_parent_type == ParentPreference.method
    assert claim.structural_hints.elaborates_seed_id == "s_method"


def test_discussion_and_background_prompt_guardrails_present() -> None:
    discussion = Section(
        section_number="7",
        title="Discussion",
        role=RhetoricalRole.discussion,
        body_text="We discuss limitations and implications.",
        char_count=39,
    )
    background = Section(
        section_number="2",
        title="Related Work",
        role=RhetoricalRole.background,
        body_text="Prior systems differ from ours in assumptions and setting.",
        char_count=62,
    )

    discussion_messages = build_section_prompt(discussion, [], [])
    background_messages = build_section_prompt(background, [], [])

    assert discussion_messages[0]["content"] == DISCUSSION_INSTRUCTIONS
    assert "A claim is NEGATIVE only if the statement itself asserts a failure" in discussion_messages[0]["content"]
    assert "is NOT NEGATIVE even if it appears in the same paragraph" in discussion_messages[0]["content"]
    assert "claim_type must be determined from the statement content itself" in discussion_messages[1]["content"]

    assert background_messages[0]["content"] == BACKGROUND_INSTRUCTIONS
    assert "Prefer 1-3 grouped positioning claims" in background_messages[0]["content"]


def test_method_prompt_includes_granularity_guardrails() -> None:
    method_section = Section(
        section_number="4",
        title="Method",
        role=RhetoricalRole.method,
        body_text="Method details.",
        char_count=15,
    )

    messages = build_section_prompt(method_section, [], [])
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "argument-level, not operation-level" in system_prompt
    assert "Do NOT split a single capability into many sibling claims" in system_prompt
    assert "Avoid sibling explosions" in user_prompt


def test_extract_section_claims_compacts_duplicates_and_procedural_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="4.1",
        title="PagedAttention",
        role=RhetoricalRole.method,
        body_text="PagedAttention details.",
        char_count=22,
    )
    seed_claims = [
        RawClaim(
            claim_id="s1",
            claim_type=ClaimType.method,
            statement="PagedAttention partitions each sequence KV cache into fixed-size blocks.",
            source_section="Abstract",
        )
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="m_seed_dup",
                    claim_type=ClaimType.method.value,
                    statement="PagedAttention partitions each sequence KV cache into fixed-size blocks.",
                    evidence_ids=["Fig. 1"],
                    entity_names=["PagedAttention"],
                ),
                FlatClaim(
                    claim_id="m_core_a",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "PagedAttention partitions each sequence KV cache into fixed-size blocks "
                        "and maps logical blocks to physical blocks."
                    ),
                    evidence_ids=["Fig. 1"],
                    entity_names=["PagedAttention", "block table"],
                ),
                FlatClaim(
                    claim_id="m_core_b",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "PagedAttention partitions each sequence KV cache into fixed-size blocks "
                        "and maps logical blocks to physical blocks via a block table."
                    ),
                    evidence_ids=["Fig. 1"],
                    entity_names=["PagedAttention", "block table"],
                ),
                FlatClaim(
                    claim_id="m_impl_list",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "For each decoding step, the runtime selects candidate sequences, allocates "
                        "physical KV blocks, concatenates input tokens, dispatches the attention "
                        "kernel, updates block tables, and stores outputs."
                    ),
                    evidence_ids=["Fig. 1"],
                    entity_names=["runtime"],
                ),
                FlatClaim(
                    claim_id="r_mistyped",
                    claim_type=ClaimType.context.value,
                    statement="vLLM achieves 2-4x throughput over FasterTransformer.",
                    evidence_ids=["Table 2"],
                    entity_names=["vLLM"],
                ),
                FlatClaim(
                    claim_id="m_unknown_evidence",
                    claim_type=ClaimType.method.value,
                    statement="A scheduler allocates blocks on demand.",
                    evidence_ids=["Fig. 99"],
                    entity_names=["scheduler"],
                ),
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [
        EvidenceArtifact(artifact_type="figure", artifact_id="Fig. 1", caption="Mechanism.", source_page=4),
        EvidenceArtifact(artifact_type="table", artifact_id="Table 2", caption="Results.", source_page=8),
    ]

    output = asyncio.run(
        extract_section_claims(
            section,
            seed_claims,
            artifacts,
            {"pipeline": {"section_extraction": {}}},
        )
    )

    statements = [claim.statement for claim in output.claims]
    ids = {claim.claim_id for claim in output.claims}
    by_id = {claim.claim_id: claim for claim in output.claims}

    assert "m_seed_dup" not in ids
    assert not any("For each decoding step" in statement for statement in statements)
    assert sum("maps logical blocks to physical blocks" in statement for statement in statements) == 1
    assert by_id["r_mistyped"].claim_type == ClaimType.result
    assert [pointer.artifact_id for pointer in by_id["m_unknown_evidence"].evidence] == []


def test_extract_section_claims_suppresses_inventory_and_api_surface_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="4.5",
        title="Implementation Details",
        role=RhetoricalRole.method,
        body_text="Implementation details.",
        char_count=23,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="inv_loc",
                    claim_type=ClaimType.method.value,
                    statement="The engine is written in 8.5K lines of code.",
                    entity_names=["engine"],
                ),
                FlatClaim(
                    claim_id="inv_api",
                    claim_type=ClaimType.method.value,
                    statement="The frontend extends the OpenAI-compatible API interface for users.",
                    entity_names=["frontend"],
                ),
                FlatClaim(
                    claim_id="inv_libs",
                    claim_type=ClaimType.method.value,
                    statement="The runtime uses NCCL, FastAPI, and PyTorch.",
                    entity_names=["runtime"],
                ),
                FlatClaim(
                    claim_id="inv_ops",
                    claim_type=ClaimType.method.value,
                    statement="Append appends a token and free deletes a sequence.",
                    entity_names=["allocator"],
                ),
                FlatClaim(
                    claim_id="m_core",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "PagedAttention uses block-table indirection to reduce KV-cache "
                        "fragmentation while preserving near-zero waste."
                    ),
                    entity_names=["PagedAttention", "block table"],
                    evidence_ids=["Fig. 1"],
                ),
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [
        EvidenceArtifact(artifact_type="figure", artifact_id="Fig. 1", caption="Mechanism.", source_page=4),
    ]
    output = asyncio.run(extract_section_claims(section, [], artifacts, {"pipeline": {"section_extraction": {}}}))

    ids = {claim.claim_id for claim in output.claims}
    assert ids == {"m_core"}


def test_extract_section_claims_evaluation_requires_findings_not_restatements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="5.1",
        title="Main Results",
        role=RhetoricalRole.evaluation,
        body_text="Evaluation details.",
        char_count=20,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="eval_restate",
                    claim_type=ClaimType.method.value,
                    statement="PagedAttention maps logical blocks to physical blocks with a block table.",
                    entity_names=["PagedAttention"],
                ),
                FlatClaim(
                    claim_id="eval_finding",
                    claim_type=ClaimType.result.value,
                    statement="vLLM improves throughput by 2.4x over FasterTransformer on OPT-13B.",
                    evidence_ids=["Table 2"],
                    entity_names=["vLLM", "FasterTransformer", "OPT-13B"],
                ),
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [
        EvidenceArtifact(artifact_type="table", artifact_id="Table 2", caption="Throughput results.", source_page=8),
    ]
    output = asyncio.run(extract_section_claims(section, [], artifacts, {"pipeline": {"section_extraction": {}}}))

    ids = {claim.claim_id for claim in output.claims}
    assert ids == {"eval_finding"}
    assert output.claims[0].claim_type == ClaimType.result


def test_extract_section_claims_demotes_observational_method_to_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="1",
        title="Introduction",
        role=RhetoricalRole.introduction,
        body_text="Motivation.",
        char_count=10,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="m_obs",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "The number of requests that can be batched together is constrained by GPU memory "
                        "capacity, making the system memory-bound."
                    ),
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    output = asyncio.run(extract_section_claims(section, [], [], {"pipeline": {"section_extraction": {}}}))
    claim_by_id = {claim.claim_id: claim for claim in output.claims}

    assert "m_obs" in claim_by_id
    assert claim_by_id["m_obs"].claim_type == ClaimType.context


def test_extract_section_claims_keeps_agency_mechanism_as_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="4",
        title="Method",
        role=RhetoricalRole.method,
        body_text="Method.",
        char_count=7,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="m_agency",
                    claim_type=ClaimType.context.value,
                    statement=(
                        "We design a block-table mapping mechanism that routes logical KV blocks to "
                        "physical blocks during decoding."
                    ),
                    entity_names=["block table"],
                    evidence_ids=["Fig. 1"],
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [EvidenceArtifact(artifact_type="figure", artifact_id="Fig. 1", caption="Method.", source_page=4)]
    output = asyncio.run(extract_section_claims(section, [], artifacts, {"pipeline": {"section_extraction": {}}}))
    claim_by_id = {claim.claim_id: claim for claim in output.claims}

    assert "m_agency" in claim_by_id
    assert claim_by_id["m_agency"].claim_type == ClaimType.method


def test_extract_section_claims_suppresses_stack_inventory_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="5",
        title="Implementation Details",
        role=RhetoricalRole.method,
        body_text="Implementation.",
        char_count=14,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="m_stack",
                    claim_type=ClaimType.method.value,
                    statement=(
                        "Control-related components, including the scheduler and block manager, are "
                        "developed in Python while custom CUDA kernels are used for PagedAttention."
                    ),
                    entity_names=["scheduler", "block manager", "Python", "CUDA"],
                ),
                FlatClaim(
                    claim_id="m_core",
                    claim_type=ClaimType.method.value,
                    statement="We implement a cache manager that allocates paged KV blocks to reduce fragmentation.",
                    entity_names=["cache manager", "paged KV blocks"],
                    evidence_ids=["Fig. 2"],
                ),
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    artifacts = [EvidenceArtifact(artifact_type="figure", artifact_id="Fig. 2", caption="Core method.", source_page=5)]
    output = asyncio.run(extract_section_claims(section, [], artifacts, {"pipeline": {"section_extraction": {}}}))

    ids = {claim.claim_id for claim in output.claims}
    assert "m_core" in ids
    assert "m_stack" not in ids


def test_extract_section_claims_drops_nonconditional_assumption_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="6",
        title="Discussion",
        role=RhetoricalRole.discussion,
        body_text="Discussion.",
        char_count=11,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="a_non_conditional",
                    claim_type=ClaimType.assumption.value,
                    statement="Future work could improve generalization to more domains.",
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    output = asyncio.run(extract_section_claims(section, [], [], {"pipeline": {"section_extraction": {}}}))
    assert output.claims == []


def test_extract_section_claims_drops_negative_without_explicit_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = Section(
        section_number="6",
        title="Discussion",
        role=RhetoricalRole.discussion,
        body_text="Discussion.",
        char_count=11,
    )

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type | None = None,
        config: dict | None = None,
    ) -> FlatSectionOutput:
        assert tier == "small"
        assert response_schema is FlatSectionOutput
        return FlatSectionOutput(
            claims=[
                FlatClaim(
                    claim_id="n_weak",
                    claim_type=ClaimType.negative.value,
                    statement="There are limitations.",
                )
            ]
        )

    monkeypatch.setattr(section_prompts, "call_model", _fake_call_model)
    output = asyncio.run(extract_section_claims(section, [], [], {"pipeline": {"section_extraction": {}}}))
    assert output.claims == []


@pytest.mark.api
@requires_api_key
def test_extract_section_claims_for_method_and_evaluation() -> None:
    settings = load_config(CONFIG_PATH)
    document = parse_pdf(str(FIXTURE_PDF), settings)

    abstract_text = _get_abstract_text(document)
    seed_output = asyncio.run(extract_seed(abstract_text, settings))
    assert seed_output.claims

    method_section = _find_method_section(document)
    evaluation_section = _find_evaluation_section(document)

    method_output = asyncio.run(
        extract_section_claims(method_section, seed_output.claims, document.all_artifacts, settings)
    )
    evaluation_output = asyncio.run(
        extract_section_claims(evaluation_section, seed_output.claims, document.all_artifacts, settings)
    )

    method_claims = [claim for claim in method_output.claims if claim.claim_type == ClaimType.method]
    evaluation_result_claims = [
        claim for claim in evaluation_output.claims if claim.claim_type == ClaimType.result
    ]

    assert len(method_claims) >= 2
    assert len(evaluation_result_claims) >= 2

    combined_claims = method_output.claims + evaluation_output.claims
    assert any(claim.evidence for claim in combined_claims)

    print(f"\nMETHOD SECTION: {method_section.section_number} {method_section.title}".strip())
    for claim in method_output.claims:
        evidence_ids = [pointer.artifact_id for pointer in claim.evidence]
        print(f"- {claim.claim_type.value.upper()} | {claim.statement} | evidence={evidence_ids}")

    print(f"\nEVALUATION SECTION: {evaluation_section.section_number} {evaluation_section.title}".strip())
    for claim in evaluation_output.claims:
        evidence_ids = [pointer.artifact_id for pointer in claim.evidence]
        print(f"- {claim.claim_type.value.upper()} | {claim.statement} | evidence={evidence_ids}")
