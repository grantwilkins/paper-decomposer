"""
Claim:
Extraction sanitization demotes deterministically invalid method-family nodes
after model repair and removes dangling local references before final validation.

Plausible wrong implementations:
- Keep section-heading method nodes after repair.
- Keep methods without mechanism sentences after repair.
- Remove invalid nodes but leave edges, links, claims, or outcomes pointing to them.
- Drop useful evidence instead of preserving invalid nodes as demoted items.
- Preserve a graph that is structurally valid but semantically too flat for DB writes.
- Leave end-to-end system claims attached only to supporting methods.
- Keep numeric claims as prose without outcome rows.
- Keep scenario-specific adapters, implementation kernels, or category labels as main DAG methods.
- Demote method bloat but fail to retarget claims/outcomes to the retained reusable mechanism.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import (
    CandidateNode,
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
from paper_decomposer.extraction.sanitize import preserve_graph_and_attach_claims
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
        candidates=[
            CandidateNode(
                name="Method",
                candidate_kind="method",
                rationale="Rejected section-shaped method candidate.",
                evidence_span_ids=["s1"],
            ),
            CandidateNode(
                name="Cache handling",
                candidate_kind="method",
                rationale="Rejected underspecified cache candidate.",
                evidence_span_ids=["s1"],
            ),
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


def test_graph_quality_repair_tightens_vllm_topology_settings_claims_and_outcomes() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="abstract",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="To address this problem, we propose PagedAttention. On top of it, we build vLLM, an LLM serving system.",
            ),
            EvidenceSpan(
                span_id="method",
                paper_id="paper-1",
                section_title="PagedAttention",
                section_kind="method",
                text="The KV cache manager uses block tables for parallel sampling and beam search with copy-on-write.",
            ),
            EvidenceSpan(
                span_id="preemption",
                paper_id="paper-1",
                section_title="Scheduling and Preemption",
                section_kind="method",
                text="Sequence groups are preempted together; recovery uses swapping or recomputation.",
            ),
            EvidenceSpan(
                span_id="eval",
                paper_id="paper-1",
                section_title="Evaluation",
                section_kind="evaluation",
                text=(
                    "vLLM can sustain 2x higher request rates than Orca. "
                    "vLLM achieves 6.1%-9.8% memory saving on parallel sampling and 37.6%-55.2% on beam search. "
                    "PagedAttention incurs 20-26% higher attention kernel latency compared to FasterTransformer."
                ),
            ),
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system:vLLM",
                kind="system",
                canonical_name="vLLM",
                description="LLM serving system.",
                status="uncertain",
                granularity_rationale="Top-level system.",
                evidence_span_ids=["abstract"],
            ),
            ExtractedNode(
                local_node_id="method:PagedAttention",
                kind="method",
                canonical_name="PagedAttention",
                description="Paged KV attention.",
                status="uncertain",
                granularity_rationale="Central primitive.",
                mechanism_sentence="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                evidence_span_ids=["abstract"],
            ),
            ExtractedNode(
                local_node_id="method:blockwise_KV_blocks",
                kind="method",
                canonical_name="Block-wise KV cache",
                description="KV cache uses blocks.",
                granularity_rationale="Reusable representation.",
                mechanism_sentence="The KV cache is partitioned into fixed-size blocks for memory management.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:block_level_sharing",
                kind="method",
                canonical_name="Block-level KV cache sharing",
                description="KV blocks are shared.",
                granularity_rationale="Reusable sharing mechanism.",
                mechanism_sentence="Physical KV blocks are shared across logical blocks to reduce duplicate memory.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:copy_on_write",
                kind="method",
                canonical_name="Copy-on-write",
                description="Shared blocks are copied before mutation.",
                granularity_rationale="Reusable sharing support.",
                mechanism_sentence="When a shared physical block is modified, a private copy is created before writing.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:kv_cache_swapping",
                kind="method",
                canonical_name="KV-cache swapping",
                description="KV blocks can be swapped.",
                granularity_rationale="Reusable recovery mechanism.",
                mechanism_sentence="Evicted KV cache blocks are copied to CPU memory and restored later.",
                evidence_span_ids=["preemption"],
            ),
            ExtractedNode(
                local_node_id="method:kv_cache_recomputation",
                kind="method",
                canonical_name="KV-cache recomputation",
                description="KV blocks can be recomputed.",
                granularity_rationale="Reusable recovery mechanism.",
                mechanism_sentence="Evicted KV cache blocks are recomputed from prior tokens when the sequence resumes.",
                evidence_span_ids=["preemption"],
            ),
        ],
        edges=[
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:blockwise_KV_blocks", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:block_level_sharing", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:copy_on_write", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:kv_cache_swapping", relation_kind="uses", evidence_span_ids=["preemption"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:kv_cache_recomputation", relation_kind="uses", evidence_span_ids=["preemption"]),
        ],
        settings=[
            ExtractedSetting(
                local_setting_id="setting:KV_cache_memory_inefficiency",
                kind="application",
                canonical_name="KV-cache memory inefficiency",
                description="Problem context.",
                evidence_span_ids=["abstract"],
            ),
            ExtractedSetting(
                local_setting_id="setting:decoding_scenarios",
                kind="application",
                canonical_name="Decoding scenarios",
                description="Coarse bucket.",
                evidence_span_ids=["method"],
            ),
        ],
        method_setting_links=[
            ExtractedMethodSettingLink(
                method_id="method:PagedAttention",
                setting_id="setting:KV_cache_memory_inefficiency",
                relation_kind="applies_to",
                evidence_span_ids=["abstract"],
            ),
            ExtractedMethodSettingLink(
                method_id="method:block_level_sharing",
                setting_id="setting:decoding_scenarios",
                relation_kind="applies_to",
                evidence_span_ids=["method"],
            ),
        ],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="vLLM can sustain 2x higher request rates than Orca.",
                finding="vLLM can sustain 2x higher request rates than Orca.",
                method_ids=["method:PagedAttention"],
                setting_ids=["setting:KV_cache_memory_inefficiency"],
                metric="request rate",
                delta="2x",
                comparator="Orca",
                evidence_span_ids=["eval"],
            ),
            ExtractedClaim(
                claim_id="c2",
                paper_id="paper-1",
                claim_type="memory",
                raw_text="vLLM achieves 6.1%-9.8% memory saving on parallel sampling and 37.6%-55.2% on beam search.",
                finding="vLLM saves memory on parallel sampling and beam search.",
                method_ids=["method:block_level_sharing"],
                setting_ids=["setting:decoding_scenarios"],
                metric="memory saving",
                delta="6.1%-55.2%",
                comparator="without sharing",
                evidence_span_ids=["eval"],
            ),
            ExtractedClaim(
                claim_id="c3",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="PagedAttention incurs 20-26% higher attention kernel latency compared to FasterTransformer.",
                finding="PagedAttention incurs 20-26% higher attention kernel latency.",
                method_ids=["method:PagedAttention"],
                metric="attention kernel latency",
                delta="20-26%",
                comparator="FasterTransformer",
                evidence_span_ids=["eval"],
            ),
        ],
    )

    repaired = preserve_graph_and_attach_claims(extraction)

    edge_pairs = {(edge.parent_id, edge.child_id) for edge in repaired.edges}
    assert ("sys_vllm", "meth_pagedattention") in edge_pairs
    assert ("meth_block_level_sharing", "meth_copy_on_write") in edge_pairs
    assert ("meth_sequence_group_preemption", "meth_kv_cache_swapping") in edge_pairs
    assert ("meth_sequence_group_preemption", "meth_kv_cache_recomputation") in edge_pairs

    method_names = {node.canonical_name for node in repaired.graph.methods}
    assert "Block-wise KV cache address translation" in method_names
    assert "Sequence-group preemption" in method_names

    setting_ids = {setting.local_setting_id for setting in repaired.settings}
    assert "setting:KV_cache_memory_inefficiency" not in setting_ids
    assert {"setting:llm_serving", "setting:parallel_sampling", "setting:beam_search"} <= setting_ids
    assert {item.name for item in repaired.demoted_items} >= {
        "KV-cache memory inefficiency",
        "Decoding scenarios",
        "KV cache manager",
    }

    claim_by_id = {claim.claim_id: claim for claim in repaired.claims}
    assert claim_by_id["c1"].method_ids == ["sys_vllm"]
    assert "setting:llm_serving" in claim_by_id["c1"].setting_ids
    assert {"setting:parallel_sampling", "setting:beam_search"} <= set(claim_by_id["c2"].setting_ids)
    assert claim_by_id["c3"].claim_type == "overhead"
    assert all(claim.outcome_ids for claim in repaired.claims)
    assert {outcome.outcome_id for outcome in repaired.outcomes} == {
        "outcome:c1",
        "outcome:c2:parallel_sampling",
        "outcome:c2:beam_search",
        "outcome:c3",
    }

    assert validate_extraction(repaired).ok


def test_cleanup_collapses_vllm_scenario_kernels_and_categories_out_of_main_dag() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="abstract",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="We propose PagedAttention and build vLLM, an LLM serving system.",
            ),
            EvidenceSpan(
                span_id="method",
                paper_id="paper-1",
                section_title="PagedAttention",
                section_kind="method",
                text=(
                    "PagedAttention supports block-level KV cache sharing for parallel sampling, "
                    "beam search, and shared-prefix prompting with reference counts and copy-on-write."
                ),
            ),
            EvidenceSpan(
                span_id="implementation",
                paper_id="paper-1",
                section_title="Implementation",
                section_kind="method",
                text="The implementation includes a fused block copy kernel and helper APIs.",
            ),
            EvidenceSpan(
                span_id="eval",
                paper_id="paper-1",
                section_title="Evaluation",
                section_kind="evaluation",
                text=(
                    "vLLM achieves 6.1%-9.8% memory saving on parallel sampling and "
                    "37.6%-55.2% on beam search. PagedAttention incurs 20-26% higher "
                    "attention kernel latency compared to FasterTransformer."
                ),
            ),
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system:vLLM",
                kind="system",
                canonical_name="vLLM",
                description="LLM serving system.",
                granularity_rationale="Top-level system.",
                evidence_span_ids=["abstract"],
            ),
            ExtractedNode(
                local_node_id="cat_kv_cache_management",
                kind="method_category",
                canonical_name="KV cache management",
                description="Global category label.",
                granularity_rationale="Cheap model promoted a category as a node.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:PagedAttention",
                kind="method",
                canonical_name="PagedAttention",
                description="Paged KV attention.",
                granularity_rationale="Central primitive.",
                mechanism_sentence="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                evidence_span_ids=["abstract", "method"],
            ),
            ExtractedNode(
                local_node_id="method:block_level_kv_cache_sharing",
                kind="method",
                canonical_name="Block-level KV cache sharing",
                description="KV blocks are shared across related sequences.",
                granularity_rationale="Reusable sharing mechanism.",
                mechanism_sentence="Physical KV blocks are shared across logical blocks to reduce duplicate memory.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:parallel_sampling_prompt_kv_sharing",
                kind="method",
                canonical_name="parallel-sampling prompt KV sharing",
                description="Scenario-specific use of KV block sharing.",
                granularity_rationale="Cheap model over-split a scenario adapter.",
                mechanism_sentence="Parallel-sampling prompt KV sharing reuses shared prompt KV blocks across sampled outputs.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:beam_search_kv_block_sharing",
                kind="method",
                canonical_name="beam-search KV block sharing",
                description="Scenario-specific use of KV block sharing.",
                granularity_rationale="Cheap model over-split a scenario adapter.",
                mechanism_sentence="Beam-search KV block sharing reuses common candidate KV blocks across beams.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:shared_prefix_kv_block_caching",
                kind="method",
                canonical_name="shared-prefix KV block caching",
                description="Scenario-specific use of KV block sharing.",
                granularity_rationale="Cheap model over-split a scenario adapter.",
                mechanism_sentence="Shared-prefix KV block caching reuses common prefix KV blocks across requests.",
                evidence_span_ids=["method"],
            ),
            ExtractedNode(
                local_node_id="method:fused_block_copy_kernel",
                kind="method",
                canonical_name="fused block copy kernel",
                description="Kernel implementation detail.",
                granularity_rationale="Cheap model promoted implementation support.",
                mechanism_sentence="The fused block copy kernel copies KV cache blocks in GPU memory for implementation efficiency.",
                evidence_span_ids=["implementation"],
            ),
        ],
        edges=[
            ExtractedEdge(parent_id="system:vLLM", child_id="cat_kv_cache_management", relation_kind="uses", evidence_span_ids=["abstract"]),
            ExtractedEdge(parent_id="cat_kv_cache_management", child_id="method:PagedAttention", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:block_level_kv_cache_sharing", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:block_level_kv_cache_sharing", child_id="method:parallel_sampling_prompt_kv_sharing", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:block_level_kv_cache_sharing", child_id="method:beam_search_kv_block_sharing", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:block_level_kv_cache_sharing", child_id="method:shared_prefix_kv_block_caching", relation_kind="uses", evidence_span_ids=["method"]),
            ExtractedEdge(parent_id="method:PagedAttention", child_id="method:fused_block_copy_kernel", relation_kind="uses", evidence_span_ids=["implementation"]),
        ],
        claims=[
            ExtractedClaim(
                claim_id="c_memory",
                paper_id="paper-1",
                claim_type="memory",
                raw_text=(
                    "vLLM achieves 6.1%-9.8% memory saving on parallel sampling and "
                    "37.6%-55.2% on beam search."
                ),
                finding="vLLM saves memory on parallel sampling and beam search.",
                method_ids=["method:parallel_sampling_prompt_kv_sharing", "method:beam_search_kv_block_sharing"],
                metric="memory saving",
                comparator="without sharing",
                evidence_span_ids=["eval"],
            ),
            ExtractedClaim(
                claim_id="c_kernel",
                paper_id="paper-1",
                claim_type="performance",
                raw_text="PagedAttention incurs 20-26% higher attention kernel latency compared to FasterTransformer.",
                finding="PagedAttention incurs attention-kernel overhead.",
                method_ids=["method:fused_block_copy_kernel"],
                metric="attention kernel latency",
                delta="20-26%",
                comparator="FasterTransformer",
                evidence_span_ids=["eval"],
            ),
        ],
    )

    repaired = preserve_graph_and_attach_claims(extraction)

    method_names = {node.canonical_name for node in repaired.graph.methods}
    assert "Block-level KV cache sharing" in method_names
    assert "parallel-sampling prompt KV sharing" not in method_names
    assert "beam-search KV block sharing" not in method_names
    assert "shared-prefix KV block caching" not in method_names
    assert "fused block copy kernel" not in method_names
    assert "KV cache management" not in method_names
    tags_by_method = {node.local_node_id: node.category_tags for node in repaired.graph.methods}
    assert "kv_cache_management" in tags_by_method["meth_pagedattention"]

    link_pairs = {(link.method_id, link.setting_id) for link in repaired.method_setting_links}
    assert {
        ("meth_block_level_kv_cache_sharing", "setting:parallel_sampling"),
        ("meth_block_level_kv_cache_sharing", "setting:beam_search"),
        ("meth_block_level_kv_cache_sharing", "setting:shared_prefix_prompting"),
    } <= link_pairs

    claim_by_id = {claim.claim_id: claim for claim in repaired.claims}
    assert claim_by_id["c_memory"].method_ids == ["meth_block_level_kv_cache_sharing"]
    assert {"setting:parallel_sampling", "setting:beam_search"} <= set(claim_by_id["c_memory"].setting_ids)
    assert claim_by_id["c_kernel"].method_ids == ["meth_pagedattention"]
    assert claim_by_id["c_kernel"].claim_type == "overhead"
    assert {outcome.outcome_id for outcome in repaired.outcomes} >= {
        "outcome:c_memory:parallel_sampling",
        "outcome:c_memory:beam_search",
        "outcome:c_kernel",
    }
    assert {item.name for item in repaired.demoted_items} >= {
        "parallel-sampling prompt KV sharing",
        "beam-search KV block sharing",
        "shared-prefix KV block caching",
        "fused block copy kernel",
        "KV cache management",
    }

    assert validate_extraction(repaired).ok


def test_cleanup_normalizes_ids_dedupes_settings_and_splits_numeric_outcomes() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Abstract",
                section_kind="abstract",
                text="We propose PagedAttention and build vLLM.",
            ),
            EvidenceSpan(
                span_id="s2",
                paper_id="paper-1",
                section_title="Evaluation",
                section_kind="evaluation",
                text=(
                    "For OPT-13B, vLLM can sustain 1.7x-2.7x higher request rates compared to "
                    "Orca (Oracle) and 2.7x-8x compared to Orca (Max)."
                ),
            ),
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system:vLLM",
                kind="system",
                canonical_name="vLLM",
                description="LLM serving system.",
                granularity_rationale="Top-level system.",
                evidence_span_ids=["s1"],
            ),
            ExtractedNode(
                local_node_id="method:PagedAttention",
                kind="method",
                canonical_name="PagedAttention",
                description="Paged KV attention.",
                granularity_rationale="Central primitive.",
                mechanism_sentence="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                evidence_span_ids=["s1"],
            ),
        ],
        edges=[
            ExtractedEdge(
                parent_id="system:vLLM",
                child_id="method:PagedAttention",
                relation_kind="uses",
                evidence_span_ids=["s1"],
            )
        ],
        settings=[
            ExtractedSetting(
                local_setting_id="set_model_opt13b",
                kind="model_artifact",
                canonical_name="OPT-13B",
                description="Duplicate cheap-model setting.",
                evidence_span_ids=["s2"],
            ),
            ExtractedSetting(
                local_setting_id="setting:opt_13b",
                kind="model_artifact",
                canonical_name="OPT-13B",
                description="Canonical deterministic setting.",
                evidence_span_ids=["s2"],
            ),
        ],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id="paper-1",
                claim_type="performance",
                raw_text=(
                    "vLLM can sustain 1.7x-2.7x higher request rates compared to Orca (Oracle) "
                    "and 2.7x-8x compared to Orca (Max)."
                ),
                finding="vLLM improves request rates over Orca.",
                method_ids=["system:vLLM"],
                setting_ids=["set_model_opt13b", "setting:opt_13b"],
                evidence_span_ids=["s2"],
            )
        ],
    )

    repaired = preserve_graph_and_attach_claims(extraction)

    assert {node.local_node_id for node in repaired.nodes} == {"sys_vllm", "meth_pagedattention"}
    assert [setting.local_setting_id for setting in repaired.settings if setting.canonical_name == "OPT-13B"] == [
        "setting:opt_13b"
    ]
    claim = repaired.claims[0]
    assert claim.method_ids == ["sys_vllm"]
    assert claim.setting_ids.count("setting:opt_13b") == 1
    assert claim.metric == "request rate"
    assert claim.comparator == "Orca (Oracle/Max)"
    assert set(claim.outcome_ids) == {"outcome:c1:orca_oracle", "outcome:c1:orca_max"}
    outcome_by_id = {outcome.outcome_id: outcome for outcome in repaired.outcomes}
    assert outcome_by_id["outcome:c1:orca_oracle"].delta == "1.7×–2.7×"
    assert outcome_by_id["outcome:c1:orca_oracle"].comparator == "Orca (Oracle)"
    assert outcome_by_id["outcome:c1:orca_max"].delta == "2.7×–8×"
    assert outcome_by_id["outcome:c1:orca_max"].comparator == "Orca (Max)"
