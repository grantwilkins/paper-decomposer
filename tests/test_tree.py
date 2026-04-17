"""
Claim:
Deterministic tree assembly should build structure only from the stable promoted
set: primitive under context, system under primitive, and results/assumptions/
negatives under the best method with deterministic subtype and label assignment.

Plausible wrong implementations:
- Flatten primitive and system methods as siblings.
- Attach results to context instead of method.
- Fail to assign result subtypes or unique canonical labels.
- Preserve the removed `negative_claims` side channel.
"""

from __future__ import annotations

import asyncio

from paper_decomposer.prompts.tree import assemble_tree_deterministic, build_tree_prompt
from paper_decomposer.schema import ClaimType, EvidenceArtifact, PaperMetadata, RawClaim


def _claim(claim_id: str, claim_type: ClaimType, statement: str, source_section: str) -> RawClaim:
    return RawClaim(claim_id=claim_id, claim_type=claim_type, statement=statement, source_section=source_section)


def test_build_tree_prompt_lists_claims_stably() -> None:
    prompt = build_tree_prompt(
        metadata=PaperMetadata(title="PagedAttention", authors=[]),
        claims=[_claim("c1", ClaimType.context, "Fragmentation limits batch size.", "1 Intro")],
        faceted=[],
        negatives=[],
        artifacts=[],
    )
    assert prompt[0]["content"]
    assert "[c1] context: Fragmentation limits batch size." in prompt[1]["content"]


def test_assemble_tree_deterministic_builds_primitive_system_hierarchy() -> None:
    metadata = PaperMetadata(title="PagedAttention", authors=[])
    claims = [
        _claim("C1", ClaimType.context, "Fragmentation limits batch size.", "1 Intro"),
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4.2"),
        _claim("R1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca.", "5.2"),
        _claim("A1", ClaimType.assumption, "Benefits depend on concurrent requests and GPU memory capacity.", "6"),
        _claim("N1", ClaimType.negative, "Without paging, prior runtimes waste KV memory through fragmentation.", "1 Intro"),
    ]
    artifacts = [EvidenceArtifact(artifact_type="table", artifact_id="t1", caption="Results", source_page=1)]

    output = asyncio.run(
        assemble_tree_deterministic(metadata=metadata, claims=claims, faceted=[], negatives=[], artifacts=artifacts, config={})
    )

    root = output.claim_tree[0]
    primitive = root.children[0]
    system = primitive.children[0]
    child_ids = {child.claim_id for child in system.children}
    labels = []
    stack = list(output.claim_tree)
    while stack:
        node = stack.pop()
        labels.append(node.canonical_label)
        stack.extend(node.children)

    assert root.claim_type == ClaimType.context
    assert primitive.claim_type == ClaimType.method
    assert system.claim_type == ClaimType.method
    assert child_ids == {"R1", "A1", "N1"}
    assert next(child for child in system.children if child.claim_id == "R1").result_subtype is not None
    assert len(labels) == len(set(labels))
    assert not hasattr(output, "negative_claims")


def test_assemble_tree_deterministic_one_liner_prefers_broad_headline_result() -> None:
    metadata = PaperMetadata(title="PagedAttention", authors=[])
    claims = [
        _claim("C1", ClaimType.context, "Fragmentation limits batch size.", "1 Intro"),
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4.2"),
        _claim("R1", ClaimType.result, "Beam search throughput improves over Orca by 2.3x.", "5.4"),
        _claim("R2", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca at the same latency.", "5.2"),
    ]

    output = asyncio.run(
        assemble_tree_deterministic(metadata=metadata, claims=claims, faceted=[], negatives=[], artifacts=[], config={})
    )

    assert output.one_liner.achieved == "vLLM reaches 2-4x higher throughput than Orca at the same latency."
    assert output.one_liner.via == "vLLM is a serving runtime built around PagedAttention."


def test_assemble_tree_deterministic_one_liner_prefers_problem_root_for_because() -> None:
    metadata = PaperMetadata(title="PagedAttention", authors=[])
    claims = [
        _claim("C1", ClaimType.context, "Complex decoding algorithms create heterogeneous KV-cache sharing opportunities.", "4.4"),
        _claim(
            "C2",
            ClaimType.context,
            "Existing LLM serving systems waste KV-cache memory through fragmentation and duplication, limiting batch size and throughput.",
            "1 Intro",
        ),
        _claim("M1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("M2", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4.2"),
        _claim("R1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca at the same latency.", "5.2"),
    ]

    output = asyncio.run(
        assemble_tree_deterministic(metadata=metadata, claims=claims, faceted=[], negatives=[], artifacts=[], config={})
    )

    assert output.one_liner.because == (
        "Existing LLM serving systems waste KV-cache memory through fragmentation and duplication, limiting batch size and throughput."
    )
