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

from paper_decomposer.prompts.dedup import classify_method_abstraction, classify_result_family, compress_claims_to_skeleton, select_result_for_one_liner
from paper_decomposer.schema import ClaimLocalRole, ClaimStructuralHints, ClaimType, RawClaim


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


def test_compression_keeps_headline_result_family_under_tight_cap() -> None:
    claims = [
        _claim("c1", ClaimType.context, "KV cache fragmentation limits batch size.", "1 Intro"),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim("r1", ClaimType.result, "vLLM reaches 2-4x higher throughput than Orca at the same latency.", "5.2"),
        _claim("r2", ClaimType.result, "vLLM achieves 53.13% average memory saving under parallel sampling on Alpaca.", "5.4"),
    ]

    result = compress_claims_to_skeleton(claims, {"pipeline": {"dedup": {"result_family_cap": 1}}})
    promoted_results = [claim for claim in result.promoted_claims if claim.claim_type == ClaimType.result]

    assert len(promoted_results) == 1
    assert promoted_results[0].statement == "vLLM reaches 2-4x higher throughput than Orca at the same latency."
    assert select_result_for_one_liner(promoted_results).statement == promoted_results[0].statement


def test_classify_method_abstraction_requires_runtime_level_system_scope() -> None:
    runtime_claim = _claim(
        "m_runtime",
        ClaimType.method,
        "vLLM is a serving runtime that manages KV cache allocation and sharing across requests.",
        "4.2",
    )
    block_mapping_claim = _claim(
        "m_blocks",
        ClaimType.method,
        "vLLM maps logical KV blocks to physical KV blocks via a block table in GPU workers.",
        "4.3",
    )

    assert classify_method_abstraction(runtime_claim) == "system_realization"
    assert classify_method_abstraction(block_mapping_claim) == "submechanism"


def test_compression_prefers_problem_root_over_scoped_decoding_observation() -> None:
    claims = [
        _claim(
            "c_problem",
            ClaimType.context,
            "Existing LLM serving systems waste KV-cache memory through fragmentation and duplication, limiting batch size and throughput.",
            "1 Introduction",
        ),
        _claim(
            "c_scoped",
            ClaimType.context,
            "Complex decoding algorithms create heterogeneous KV-cache sharing opportunities across prompt sharing and beam search.",
            "4.4",
        ),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
    ]

    result = compress_claims_to_skeleton(claims, {"pipeline": {"dedup": {"context_cap": 1}}})
    promoted_contexts = [claim for claim in result.promoted_claims if claim.claim_type == ClaimType.context]

    assert [claim.claim_id for claim in promoted_contexts] == ["C1"]
    assert promoted_contexts[0].statement == (
        "Existing LLM serving systems waste KV-cache memory through fragmentation and duplication, limiting batch size and throughput."
    )


def test_select_result_for_one_liner_prefers_broader_headline_over_specific_benchmark() -> None:
    results = [
        _claim(
            "r_benchmark",
            ClaimType.result,
            "vLLM can sustain 2x higher request rates than the three Orca baselines on ShareGPT with 1024-token prompts.",
            "5.4",
        ),
        _claim(
            "r_broad",
            ClaimType.result,
            "vLLM delivers 2-4x higher throughput than state-of-the-art systems at comparable latency, with larger gains on longer sequences and larger models.",
            "abstract",
        ),
    ]

    assert select_result_for_one_liner(results).claim_id == "r_broad"


def test_compression_rejects_method_restatement_assumptions() -> None:
    claims = [
        _claim("c1", ClaimType.context, "KV cache fragmentation limits batch size.", "1 Intro"),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        _claim(
            "a_bad",
            ClaimType.assumption,
            "The approach assumes that KV-cache memory can be managed using virtual-memory-style paging.",
            "abstract",
        ),
        _claim(
            "a_good",
            ClaimType.assumption,
            "Benefits depend on concurrent requests and available GPU memory.",
            "6 Discussion",
        ),
    ]

    result = compress_claims_to_skeleton(claims)

    assert [claim.statement for claim in result.promoted_claims if claim.claim_type == ClaimType.assumption] == [
        "Benefits depend on concurrent requests and available GPU memory."
    ]


def test_compression_keeps_runtime_system_realization_when_present() -> None:
    claims = [
        _claim("c1", ClaimType.context, "KV cache fragmentation limits batch size.", "1 Intro"),
        _claim("m1", ClaimType.method, "PagedAttention is an attention algorithm that pages KV blocks.", "4.1"),
        RawClaim(
            claim_id="m2",
            claim_type=ClaimType.method,
            statement="vLLM is a serving engine with a centralized scheduler and a KV cache manager shared by GPU workers.",
            source_section="4.0",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.implementation_detail),
        ),
        _claim("m3", ClaimType.method, "vLLM maps logical KV blocks to physical KV blocks via a block table.", "4.3"),
    ]

    result = compress_claims_to_skeleton(claims)
    methods = [claim for claim in result.promoted_claims if claim.claim_type == ClaimType.method]

    assert {classify_method_abstraction(claim) for claim in methods} == {"primitive", "system_realization"}
    assert any("serving engine" in claim.statement for claim in methods)
