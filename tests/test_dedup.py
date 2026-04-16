"""
Claim:
Stage 2 compression is the only semantic decision stage: it must keep one
problem root, at most one promoted method per concept family and abstraction
tier, reject mechanism-like assumptions, and compact results by functional
family rather than topical proximity.

Plausible wrong implementations:
- Keep both primitive paraphrases as peer core methods.
- Promote kernel/scheduler residue as core methods.
- Let mechanism descriptions survive as assumptions.
- Keep multiple results from the same functional family.
"""

from __future__ import annotations

from paper_decomposer.prompts.dedup import classify_method_abstraction, classify_result_family, compress_claims_to_skeleton
from paper_decomposer.schema import ClaimType, RawClaim


def _claim(claim_id: str, claim_type: ClaimType, statement: str, source_section: str) -> RawClaim:
    return RawClaim(claim_id=claim_id, claim_type=claim_type, statement=statement, source_section=source_section)


def test_compression_prefers_one_primitive_one_system_and_demotes_submechanisms() -> None:
    claims = [
        _claim("c1", ClaimType.context, "KV cache fragmentation limits batch size.", "1 Intro"),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "Abstract"),
        _claim("m2", ClaimType.method, "PagedAttention partitions the KV cache into fixed-size blocks for non-contiguous attention.", "4.1 PagedAttention"),
        _claim("m3", ClaimType.method, "vLLM is a serving runtime built around PagedAttention.", "4 System"),
        _claim("m4", ClaimType.method, "A fused attention kernel gathers discontinuous blocks with coalesced reads.", "5.1 Kernel"),
    ]

    result = compress_claims_to_skeleton(claims)
    methods = [claim for claim in result.promoted_claims if claim.claim_type == ClaimType.method]
    residual_statements = {claim.statement for claim in result.residual_claims}

    assert len(methods) == 2
    assert {classify_method_abstraction(claim) for claim in methods} == {"primitive", "system_realization"}
    assert "A fused attention kernel gathers discontinuous blocks with coalesced reads." in residual_statements
    assert sum(1 for claim in methods if classify_method_abstraction(claim) == "primitive") == 1


def test_compression_rejects_mechanism_like_assumptions_and_compacts_results_by_family() -> None:
    claims = [
        _claim("c1", ClaimType.context, "KV cache fragmentation limits batch size.", "1 Intro"),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("a1", ClaimType.assumption, "The runtime allocates a new block only when prior blocks are full.", "4.2"),
        _claim("a2", ClaimType.assumption, "Benefits depend on concurrent requests and GPU memory capacity.", "6 Limitations"),
        _claim("r1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca at similar latency.", "5.2"),
        _claim("r2", ClaimType.result, "The system sustains higher request rates than Orca on ShareGPT.", "5.2"),
        _claim("r3", ClaimType.result, "PagedAttention reduces KV-cache memory waste to near zero.", "5.1"),
        _claim("r4", ClaimType.result, "Kernel indirection adds 20-26% higher attention latency.", "5.3"),
        _claim("r5", ClaimType.result, "Beam search throughput improves over Orca by 2.3x.", "5.4"),
    ]

    result = compress_claims_to_skeleton(claims)
    promoted_assumptions = [claim.statement for claim in result.promoted_claims if claim.claim_type == ClaimType.assumption]
    promoted_results = [claim for claim in result.promoted_claims if claim.claim_type == ClaimType.result]
    residual_statements = {claim.statement for claim in result.residual_claims}

    assert promoted_assumptions == ["Benefits depend on concurrent requests and GPU memory capacity."]
    assert "The runtime allocates a new block only when prior blocks are full." in residual_statements
    assert len(promoted_results) == 4
    assert {classify_result_family(claim) for claim in promoted_results} == {
        "headline_comparative_performance",
        "memory_mechanism_validation",
        "constraint_observation",
        "decoding_mode_improvement",
    }
