"""
Claim:
`build_tree_prompt` must provide a structured summary of deduplicated claims,
facets, negatives, and artifacts; `assemble_tree` must request a
PaperDecomposition using the tree tier and return a decomposition with
non-trivial hierarchy and dependency edges.

Plausible wrong implementations:
- Prompt omits one or more required claim sections (context/method/result/negative).
- Method facet summaries are dropped, so mechanism details are unavailable.
- Tree assembly uses the wrong model tier or wrong structured schema.
- Returned tree is flat (no hierarchy) despite available parent-child structure.
- Dependencies are omitted, losing "which claim supports which" semantics.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from typing import Any

import pytest

import paper_decomposer.prompts.tree as tree_prompts
from paper_decomposer.prompts.tree import (
    TREE_SYSTEM_PROMPT,
    assemble_tree,
    assemble_tree_deterministic,
    build_tree_prompt,
)
from paper_decomposer.schema import (
    AbstractionLevel,
    CanonicalLabelAssignment,
    CanonicalLabelOutput,
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimGroup,
    ClaimNode,
    ClaimType,
    EvidenceArtifact,
    EvidencePointer,
    FacetedClaim,
    GroundingType,
    InterventionType,
    OneLiner,
    PaperDecomposition,
    PaperMetadata,
    ParentPreference,
    RawClaim,
    ResultSubtype,
    SemanticRole,
    ScopeOfChange,
    SystemsFacets,
    TreeAssemblyOutput,
    TreeNodeAssignment,
    UniversalFacets,
)


def _evidence(*artifact_ids: str) -> list[EvidencePointer]:
    return [EvidencePointer(artifact_id=artifact_id, role="supports") for artifact_id in artifact_ids]


def _claim(
    claim_id: str,
    claim_type: ClaimType,
    statement: str,
    source_section: str,
    evidence_ids: tuple[str, ...] = (),
    *,
    rejected_what: str | None = None,
    rejected_why: str | None = None,
) -> RawClaim:
    return RawClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        statement=statement,
        source_section=source_section,
        evidence=_evidence(*evidence_ids),
        entity_names=["PagedAttention", "KV cache"] if claim_type == ClaimType.method else [],
        rejected_what=rejected_what,
        rejected_why=rejected_why,
    )


def _fixtures() -> tuple[PaperMetadata, list[RawClaim], list[FacetedClaim], list[RawClaim], list[EvidenceArtifact]]:
    metadata = PaperMetadata(
        title="Efficient Memory Management for LLM Serving with PagedAttention",
        authors=["Woosuk Kwon", "Zhuohan Li", "Eric Xing"],
        venue="SOSP",
        year=2023,
    )

    artifacts = [
        EvidenceArtifact(
            artifact_type="figure",
            artifact_id="fig_1",
            caption="Overview of PagedAttention and block-table indirection.",
            source_page=2,
        ),
        EvidenceArtifact(
            artifact_type="table",
            artifact_id="table_1",
            caption="KV-cache fragmentation under contiguous allocation.",
            source_page=3,
        ),
        EvidenceArtifact(
            artifact_type="figure",
            artifact_id="fig_3",
            caption="Memory waste comparison across serving systems.",
            source_page=5,
        ),
        EvidenceArtifact(
            artifact_type="table",
            artifact_id="table_2",
            caption="Serving throughput across models and baselines.",
            source_page=8,
        ),
        EvidenceArtifact(
            artifact_type="table",
            artifact_id="table_3",
            caption="Latency overhead of non-contiguous gather.",
            source_page=11,
        ),
    ]

    claims = [
        _claim(
            "c1",
            ClaimType.context,
            "Autoregressive serving wastes KV-cache memory because contiguous allocation fragments.",
            "2.2 Memory Management Challenge",
            ("table_1",),
        ),
        _claim(
            "c2",
            ClaimType.context,
            "Fragmentation limits feasible batch size and throughput on fixed-memory GPUs.",
            "2.2 Memory Management Challenge",
            ("table_1",),
        ),
        _claim(
            "c3",
            ClaimType.context,
            "Existing runtimes assume contiguous KV layout that hinders sharing and preemption.",
            "3 Design Motivation",
            ("fig_1",),
        ),
        _claim(
            "m1",
            ClaimType.method,
            "PagedAttention partitions each sequence KV cache into fixed-size blocks.",
            "4.1 PagedAttention",
            ("fig_1",),
        ),
        _claim(
            "m2",
            ClaimType.method,
            "A block table maps logical KV blocks to physical blocks during attention.",
            "4.1 PagedAttention",
            ("fig_1",),
        ),
        _claim(
            "m3",
            ClaimType.method,
            "The runtime allocator performs on-demand block allocation and reclamation.",
            "4.2 Runtime Allocator",
            ("fig_1",),
        ),
        _claim(
            "m4",
            ClaimType.method,
            "Copy-on-write sharing lets parallel sampling reuse prompt KV blocks safely.",
            "4.3 Parallel Sampling",
            ("fig_1",),
        ),
        _claim(
            "m5",
            ClaimType.method,
            "A custom attention kernel gathers non-contiguous blocks with low overhead.",
            "4.4 CUDA Kernel",
            ("table_3",),
        ),
        _claim(
            "r1",
            ClaimType.result,
            "PagedAttention keeps KV-cache memory waste below 4 percent across workloads.",
            "5.1 Memory Efficiency",
            ("fig_3",),
        ),
        _claim(
            "r2",
            ClaimType.result,
            "vLLM reaches 2-4x higher throughput than FasterTransformer and Orca.",
            "5.2 Throughput",
            ("table_2",),
        ),
        _claim(
            "r3",
            ClaimType.result,
            "Block-table indirection adds negligible decode-time latency overhead.",
            "5.3 Latency",
            ("table_3",),
        ),
        _claim(
            "r4",
            ClaimType.result,
            "Larger effective batch sizes are sustained before OOM events.",
            "5.2 Throughput",
            ("table_2", "fig_3"),
        ),
        _claim(
            "a1",
            ClaimType.assumption,
            "Benefits assume gather kernels can maintain memory coalescing on target GPUs.",
            "6.4 Limitations",
            ("table_3",),
        ),
    ]

    negatives = [
        _claim(
            "n1",
            ClaimType.negative,
            "Online KV compaction was too expensive at serving scale.",
            "2.3 Design Alternatives",
            (),
            rejected_what="online KV compaction",
            rejected_why="compaction cost scales with cache size and stalls decoding",
        ),
        _claim(
            "n2",
            ClaimType.negative,
            "Very small block sizes increase kernel overhead and reduce throughput.",
            "6.4 Limitations",
            (),
        ),
    ]

    method_claims_by_id = {claim.claim_id: claim for claim in claims if claim.claim_type == ClaimType.method}
    faceted_claims = [
        FacetedClaim(
            claim=method_claims_by_id["m1"],
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.systems],
                scope=ScopeOfChange.system,
                improves_or_replaces="contiguous KV allocation",
                core_tradeoff="block indirection overhead for large memory savings",
                grounding=GroundingType.empirical_controlled,
                analogy_source="virtual memory paging",
            ),
            systems_facets=SystemsFacets(
                s1_resource="gpu_memory",
                s2_alloc_unit="fixed-size KV block",
                s3_stack_layer="inference runtime",
                s4_mapping="logical KV block -> physical block via block table",
                s5_policy="allocate blocks on demand",
                s6_hw_assumption="fast random-access reads on modern GPUs",
            ),
        ),
        FacetedClaim(
            claim=method_claims_by_id["m2"],
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.systems],
                scope=ScopeOfChange.system,
                improves_or_replaces="pointer-free contiguous indexing",
                core_tradeoff="address translation cost for flexible placement",
                grounding=GroundingType.empirical_controlled,
                analogy_source="virtual page table",
            ),
            systems_facets=SystemsFacets(
                s1_resource="gpu_memory",
                s2_alloc_unit="logical KV block index",
                s3_stack_layer="attention kernel interface",
                s4_mapping="logical block index -> physical block id via table lookup",
                s5_policy="lazy mapping updates per decode step",
                s6_hw_assumption="lookup metadata fits in fast memory",
            ),
        ),
        FacetedClaim(
            claim=method_claims_by_id["m3"],
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.systems],
                scope=ScopeOfChange.system,
                improves_or_replaces="static per-request reservation",
                core_tradeoff="allocator bookkeeping for less reserved slack",
                grounding=GroundingType.empirical_controlled,
                analogy_source="slab allocator",
            ),
            systems_facets=SystemsFacets(
                s1_resource="gpu_memory",
                s2_alloc_unit="free block",
                s3_stack_layer="runtime scheduler",
                s4_mapping="active request -> set of allocated block ids",
                s5_policy="allocate on decode demand and reclaim on completion",
                s6_hw_assumption="sufficient atomic support for allocator metadata",
            ),
        ),
        FacetedClaim(
            claim=method_claims_by_id["m4"],
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.systems],
                scope=ScopeOfChange.system,
                improves_or_replaces="duplicated prompt KV for each sample",
                core_tradeoff="copy-on-write checks for significant memory sharing",
                grounding=GroundingType.empirical_controlled,
                analogy_source="copy-on-write pages",
            ),
            systems_facets=SystemsFacets(
                s1_resource="gpu_memory",
                s2_alloc_unit="shared prompt block",
                s3_stack_layer="sampling runtime",
                s4_mapping="request branch -> shared prompt block refs with COW metadata",
                s5_policy="share until mutation then fork block",
                s6_hw_assumption="branching workloads with overlapping prompts",
            ),
        ),
        FacetedClaim(
            claim=method_claims_by_id["m5"],
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.systems],
                scope=ScopeOfChange.module,
                improves_or_replaces="contiguous-only attention kernel",
                core_tradeoff="extra gathers for tolerance to fragmented layout",
                grounding=GroundingType.empirical_controlled,
                analogy_source="scatter-gather IO",
            ),
            systems_facets=SystemsFacets(
                s1_resource="gpu_compute_and_memory_bandwidth",
                s2_alloc_unit="block gather operation",
                s3_stack_layer="CUDA attention kernel",
                s4_mapping="query tile -> gathered physical KV blocks by table",
                s5_policy="batch gathers by locality before attention compute",
                s6_hw_assumption="high-bandwidth memory and warp-level primitives",
            ),
        ),
    ]

    return metadata, claims, faceted_claims, negatives, artifacts


def _iter_nodes(nodes: list[ClaimNode]) -> Iterable[ClaimNode]:
    for node in nodes:
        yield node
        yield from _iter_nodes(node.children)


def _node_depth(node: ClaimNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_node_depth(child) for child in node.children)


def _tree_depth(roots: list[ClaimNode]) -> int:
    if not roots:
        return 0
    return max(_node_depth(root) for root in roots)


def _parent_map(roots: list[ClaimNode]) -> dict[str, str | None]:
    parent_by_child: dict[str, str | None] = {}

    def walk(nodes: list[ClaimNode], parent_id: str | None = None) -> None:
        for node in nodes:
            parent_by_child[node.claim_id] = parent_id
            walk(node.children, node.claim_id)

    walk(roots)
    return parent_by_child


def _print_tree(nodes: list[ClaimNode], indent: int = 0) -> None:
    for node in nodes:
        dependency = f" depends_on={node.depends_on}" if node.depends_on else ""
        print(f"{'  ' * indent}- {node.claim_id} ({node.claim_type.value}){dependency}: {node.statement}")
        _print_tree(node.children, indent + 1)


def _section(text: str, title: str) -> str:
    marker = f"=== {title} ==="
    after_marker = text.split(marker, maxsplit=1)[1]
    next_header = after_marker.split("\n=== ", maxsplit=1)[0]
    return next_header.strip()


def _mock_tree_output(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted_claims: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
) -> TreeAssemblyOutput:
    _ = (metadata, claims, faceted_claims, negatives, artifacts)
    return TreeAssemblyOutput(
        one_liner=OneLiner(
            achieved="2-4x higher serving throughput with under-4-percent KV memory waste",
            via="paged KV blocks, block-table indirection, and on-demand allocation",
            because="fragmentation in contiguous KV allocation bottlenecks batch size and GPU utilization",
        ),
        nodes=[
            TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
            TreeNodeAssignment(claim_id="c2", parent_id=None, depends_on=[]),
            TreeNodeAssignment(claim_id="c3", parent_id=None, depends_on=[]),
            TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
            TreeNodeAssignment(claim_id="m2", parent_id="m1", depends_on=["m1"]),
            TreeNodeAssignment(claim_id="m3", parent_id="c1", depends_on=["c1"]),
            TreeNodeAssignment(claim_id="m4", parent_id="c2", depends_on=["c2"]),
            TreeNodeAssignment(claim_id="m5", parent_id="c3", depends_on=["c3"]),
            TreeNodeAssignment(claim_id="r1", parent_id="m1", depends_on=["m1"]),
            TreeNodeAssignment(claim_id="r2", parent_id="m4", depends_on=["m4"]),
            TreeNodeAssignment(claim_id="r3", parent_id="m5", depends_on=["m5"]),
            TreeNodeAssignment(claim_id="r4", parent_id="m3", depends_on=["m3"]),
            TreeNodeAssignment(claim_id="a1", parent_id="m5", depends_on=["m5"]),
            TreeNodeAssignment(claim_id="n1", parent_id="m3", depends_on=["m3"]),
            TreeNodeAssignment(claim_id="n2", parent_id="m5", depends_on=["m5"]),
        ],
    )


def test_build_tree_prompt_routes_claims_to_correct_sections_with_facets_and_evidence() -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()
    messages = build_tree_prompt(metadata, claims, faceted_claims, negatives, artifacts)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == TREE_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"

    text = messages[1]["content"]
    context_block = _section(text, "CONTEXT CLAIMS")
    method_block = _section(text, "METHOD CLAIMS (with facets)")
    result_block = _section(text, "RESULT CLAIMS")
    negative_block = _section(text, "NEGATIVE CLAIMS")
    artifact_block = _section(text, "EVIDENCE ARTIFACTS")

    claim_by_id = {claim.claim_id: claim for claim in claims}
    assert claim_by_id["c1"].statement in context_block
    assert claim_by_id["m1"].statement in method_block
    assert claim_by_id["r1"].statement in result_block
    assert claim_by_id["a1"].statement in _section(text, "ASSUMPTION CLAIMS")
    assert claim_by_id["m1"].statement not in context_block
    assert claim_by_id["r1"].statement not in method_block

    assert "[systems | resource: gpu_memory" in method_block
    assert "mapping: logical KV block -> physical block via block table" in method_block
    assert "[evidence: Table 1]" in context_block
    assert "[evidence: Fig. 3]" in result_block
    assert "N1 [id=n1]: REJECTED online KV compaction - compaction cost scales with cache size and stalls decoding" in (
        negative_block
    )
    assert "Fig. 1: Overview of PagedAttention and block-table indirection." in artifact_block
    assert 'Return JSON with "one_liner" and "nodes" only.' in text
    assert "MOST SPECIFIC method" in messages[0]["content"]
    assert "Avoid star graphs" in messages[0]["content"]
    assert "Hard parent grammar" in messages[0]["content"]


def test_build_tree_prompt_negative_fallback_and_override_boundary() -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    fallback_messages = build_tree_prompt(
        metadata=metadata,
        claims=[*claims, negatives[0]],
        faceted_claims=faceted_claims,
        negatives=[],
        artifacts=artifacts,
    )
    fallback_negative_block = _section(fallback_messages[1]["content"], "NEGATIVE CLAIMS")
    assert "online KV compaction" in fallback_negative_block

    override_negative = _claim(
        "n_override",
        ClaimType.negative,
        "Pinned-memory fallback was rejected.",
        "2.3 Design Alternatives",
        (),
        rejected_what="pinned-memory fallback",
        rejected_why="cross-device transfer bottleneck",
    )
    override_messages = build_tree_prompt(
        metadata=metadata,
        claims=[*claims, negatives[0]],
        faceted_claims=faceted_claims,
        negatives=[override_negative],
        artifacts=artifacts,
    )
    override_negative_block = _section(override_messages[1]["content"], "NEGATIVE CLAIMS")
    assert "pinned-memory fallback" in override_negative_block
    assert "online KV compaction" not in override_negative_block


def test_assemble_tree_returns_valid_hierarchical_decomposition(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()
    calls: list[tuple[str, type[Any] | None]] = []

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        calls.append((tier, response_schema))
        assert messages and messages[0]["content"] == TREE_SYSTEM_PROMPT
        assert "=== METHOD CLAIMS (with facets) ===" in messages[1]["content"]
        return _mock_tree_output(metadata, claims, faceted_claims, negatives, artifacts)

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    assert isinstance(output, PaperDecomposition)
    assert calls == [("heavy", TreeAssemblyOutput)]
    assert output.metadata == metadata
    assert output.negative_claims == negatives
    assert output.all_artifacts == artifacts

    assert output.one_liner.achieved.strip()
    assert output.one_liner.via.strip()
    assert output.one_liner.because.strip()

    assert len(output.claim_tree) >= 2
    assert _tree_depth(output.claim_tree) > 1
    assert any(node.children for node in _iter_nodes(output.claim_tree))
    assert any(node.depends_on for node in _iter_nodes(output.claim_tree))

    print("\nClaim tree:")
    _print_tree(output.claim_tree)


def test_assemble_tree_defaults_to_heavy_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()
    tiers: list[str] = []

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        tiers.append(tier)
        return _mock_tree_output(metadata, claims, faceted_claims, negatives, artifacts)

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)

    _ = asyncio.run(assemble_tree(metadata, claims, faceted_claims, negatives, artifacts, config={}))
    _ = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "invalid"}}},
        )
    )

    assert tiers == ["heavy", "heavy"]


def test_assemble_tree_handles_invalid_parents_and_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="m1", parent_id="m2", depends_on=["m2"]),
                TreeNodeAssignment(claim_id="m2", parent_id="m1", depends_on=["m1"]),
                TreeNodeAssignment(claim_id="r1", parent_id="missing", depends_on=["missing"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    all_nodes = list(_iter_nodes(output.claim_tree))
    assert all_nodes
    assert any(node.claim_id == "m1" for node in all_nodes)
    assert any(node.claim_id == "m2" for node in all_nodes)
    assert any(node.claim_id == "r1" for node in all_nodes)
    assert all(node.claim_id not in {child.claim_id for child in node.children} for node in all_nodes)


def test_assemble_tree_enforces_parent_type_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="r1", depends_on=["r1"]),  # invalid for METHOD
                TreeNodeAssignment(claim_id="r1", parent_id="c1", depends_on=["c1"]),  # invalid for RESULT
                TreeNodeAssignment(claim_id="n1", parent_id="r1", depends_on=["r1"]),  # invalid for NEGATIVE
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child: dict[str, str | None] = {}

    def walk(nodes: list[ClaimNode], parent_id: str | None = None) -> None:
        for node in nodes:
            parent_by_child[node.claim_id] = parent_id
            walk(node.children, node.claim_id)

    walk(output.claim_tree)
    method_ids = {claim.claim_id for claim in claims if claim.claim_type == ClaimType.method}

    assert parent_by_child.get("m1") != "r1"
    assert parent_by_child.get("r1") in method_ids
    assert parent_by_child.get("n1") != "r1"


def test_assemble_tree_uses_canonicalization_links_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, response_schema, config)
        assert "=== CANONICALIZATION LINKS ===" in messages[1]["content"]
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                # m2 intentionally omitted to exercise canonicalization fallback.
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
            claim_groups=[ClaimGroup(canonical_id="m2", member_ids=[], parent_id="m1")],
        )
    )

    parent_by_child: dict[str, str | None] = {}

    def walk(nodes: list[ClaimNode], parent_id: str | None = None) -> None:
        for node in nodes:
            parent_by_child[node.claim_id] = parent_id
            walk(node.children, node.claim_id)

    walk(output.claim_tree)
    assert parent_by_child["m2"] == "m1"


def test_assemble_tree_repairs_star_like_result_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m2", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m3", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m4", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m5", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r1", parent_id="m1", depends_on=["m1"]),
                TreeNodeAssignment(claim_id="r2", parent_id="m1", depends_on=["m1"]),
                TreeNodeAssignment(claim_id="r3", parent_id="m1", depends_on=["m1"]),
                TreeNodeAssignment(claim_id="r4", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child: dict[str, str | None] = {}

    def walk(nodes: list[ClaimNode], parent_id: str | None = None) -> None:
        for node in nodes:
            parent_by_child[node.claim_id] = parent_id
            walk(node.children, node.claim_id)

    walk(output.claim_tree)

    assert parent_by_child["r3"] is not None
    assert parent_by_child["r3"] != "m1"


def test_assemble_tree_backfills_missing_parents_for_non_context_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, claims, faceted_claims, negatives, artifacts = _fixtures()

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id=claim.claim_id, parent_id=None, depends_on=[])
                for claim in [*claims, *negatives]
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted_claims,
            negatives,
            artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    roots = output.claim_tree
    assert roots
    assert all(root.claim_type == ClaimType.context for root in roots)

    parent_by_child = _parent_map(roots)
    for node in _iter_nodes(roots):
        if node.claim_type == ClaimType.context:
            continue
        assert parent_by_child[node.claim_id] is not None


def test_assemble_tree_reattaches_low_level_method_under_method_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="PagedAttention Mini", authors=["A"])
    artifacts = [
        EvidenceArtifact(
            artifact_type="figure",
            artifact_id="fig_1",
            caption="PagedAttention overview.",
            source_page=1,
        )
    ]
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Contiguous KV allocation fragments memory under dynamic serving workloads.",
            source_section="1 Motivation",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We introduce PagedAttention to decouple logical and physical KV placement.",
            source_section="4.1 PagedAttention",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["PagedAttention", "KV cache"],
        ),
        RawClaim(
            claim_id="m2",
            claim_type=ClaimType.method,
            statement="A block table maps logical KV blocks to physical blocks at decode time.",
            source_section="4.1 PagedAttention",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["PagedAttention", "KV block table"],
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="The method improves throughput by 2x on long-context workloads.",
            source_section="5.2 Throughput",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m2", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r1", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata,
            claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["m2"] == "m1"


def test_assemble_tree_reattaches_low_level_method_using_structural_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving systems waste memory via fragmentation.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_top",
            claim_type=ClaimType.method,
            statement="We introduce paged KV cache management.",
            source_section="3 Method",
            structural_hints=ClaimStructuralHints(
                local_role=ClaimLocalRole.top_level,
                preferred_parent_type=ParentPreference.context,
            ),
        ),
        RawClaim(
            claim_id="m_low",
            claim_type=ClaimType.method,
            statement="A block table maps logical KV blocks to physical blocks.",
            source_section="3.1 Implementation",
            structural_hints=ClaimStructuralHints(
                elaborates_seed_id="m_top",
                local_role=ClaimLocalRole.implementation_detail,
                preferred_parent_type=ParentPreference.method,
            ),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_top", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_low", parent_id="c1", depends_on=["c1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child: dict[str, str | None] = {}

    def walk(nodes: list[ClaimNode], parent_id: str | None = None) -> None:
        for node in nodes:
            parent_by_child[node.claim_id] = parent_id
            walk(node.children, node.claim_id)

    walk(output.claim_tree)
    assert parent_by_child["m_top"] == "c1"
    assert parent_by_child["m_low"] == "m_top"


def test_assemble_tree_prefers_top_level_method_for_top_line_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    artifacts = [
        EvidenceArtifact(
            artifact_type="table",
            artifact_id="table_1",
            caption="End-to-end throughput across serving baselines.",
            source_page=6,
        )
    ]
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving throughput is limited by memory management overhead.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="We introduce a paged serving system for KV cache management.",
            source_section="3 Method",
            structural_hints=ClaimStructuralHints(
                local_role=ClaimLocalRole.top_level,
                preferred_parent_type=ParentPreference.context,
            ),
        ),
        RawClaim(
            claim_id="m_kernel",
            claim_type=ClaimType.method,
            statement="A block table maps logical KV blocks to physical blocks during decoding.",
            source_section="3.1 Kernel",
            structural_hints=ClaimStructuralHints(
                elaborates_seed_id="m_system",
                local_role=ClaimLocalRole.mechanism,
                preferred_parent_type=ParentPreference.method,
            ),
        ),
        RawClaim(
            claim_id="r_top",
            claim_type=ClaimType.result,
            statement="The system delivers 2-4x higher end-to-end throughput than baselines at the same latency.",
            source_section="5 Results",
            evidence=_evidence("table_1"),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_system", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_kernel", parent_id="m_system", depends_on=["m_system"]),
                TreeNodeAssignment(claim_id="r_top", parent_id="m_kernel", depends_on=["m_kernel"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["r_top"] == "m_system"


def test_assemble_tree_prefers_submechanism_method_for_local_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    artifacts = [
        EvidenceArtifact(
            artifact_type="table",
            artifact_id="table_2",
            caption="Kernel overhead versus block size.",
            source_page=7,
        )
    ]
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving efficiency depends on managing KV memory without large decode overheads.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="We introduce a paged serving system for KV cache management.",
            source_section="3 Method",
            structural_hints=ClaimStructuralHints(
                local_role=ClaimLocalRole.top_level,
                preferred_parent_type=ParentPreference.context,
            ),
        ),
        RawClaim(
            claim_id="m_kernel",
            claim_type=ClaimType.method,
            statement="A custom kernel uses block-wise gather to keep decode overhead low.",
            source_section="3.1 Kernel",
            structural_hints=ClaimStructuralHints(
                elaborates_seed_id="m_system",
                local_role=ClaimLocalRole.mechanism,
                preferred_parent_type=ParentPreference.method,
            ),
        ),
        RawClaim(
            claim_id="r_kernel",
            claim_type=ClaimType.result,
            statement="Block size ablations show kernel overhead rises for very small blocks.",
            source_section="5.3 Ablations",
            evidence=_evidence("table_2"),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_system", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_kernel", parent_id="m_system", depends_on=["m_system"]),
                TreeNodeAssignment(claim_id="r_kernel", parent_id="m_system", depends_on=["m_system"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["r_kernel"] == "m_kernel"


def test_assemble_tree_suppresses_non_argumentative_method_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    artifacts = [
        EvidenceArtifact(
            artifact_type="figure",
            artifact_id="fig_1",
            caption="Main method overview.",
            source_page=2,
        )
    ]
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving latency is constrained by cache management overhead.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_main",
            claim_type=ClaimType.method,
            statement="We introduce adaptive cache partitioning to reduce synchronization overhead.",
            source_section="3 Method",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["adaptive cache partitioning"],
        ),
        RawClaim(
            claim_id="m_stack",
            claim_type=ClaimType.method,
            statement="The service uses Python API handlers.",
            source_section="Appendix Implementation Details",
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="The method reduces p99 latency by 30 percent.",
            source_section="5 Results",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_main", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_stack", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r1", parent_id="m_stack", depends_on=["m_stack"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    all_nodes = list(_iter_nodes(output.claim_tree))
    node_ids = {node.claim_id for node in all_nodes}
    parent_by_child = _parent_map(output.claim_tree)

    assert "m_stack" not in node_ids
    assert parent_by_child["r1"] == "m_main"


def test_assemble_tree_suppresses_low_strength_method_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    artifacts = [
        EvidenceArtifact(
            artifact_type="figure",
            artifact_id="fig_1",
            caption="Main system overview.",
            source_page=2,
        )
    ]
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving systems need better memory management to sustain throughput.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_main",
            claim_type=ClaimType.method,
            statement="We introduce adaptive cache partitioning to reduce synchronization overhead.",
            source_section="3 Method",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["adaptive cache partitioning"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
            claim_strength=3.8,
        ),
        RawClaim(
            claim_id="m_detail",
            claim_type=ClaimType.method,
            statement="The cache manager exposes tuning hooks.",
            source_section="Appendix Implementation Details",
            structural_hints=ClaimStructuralHints(
                elaborates_seed_id="m_main",
                local_role=ClaimLocalRole.implementation_detail,
                preferred_parent_type=ParentPreference.method,
            ),
            claim_strength=1.4,
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="The system improves throughput by 35 percent on long-context workloads.",
            source_section="5 Results",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_main", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_detail", parent_id="m_main", depends_on=["m_main"]),
                TreeNodeAssignment(claim_id="r1", parent_id="m_detail", depends_on=["m_detail"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    all_nodes = list(_iter_nodes(output.claim_tree))
    node_ids = {node.claim_id for node in all_nodes}
    parent_by_child = _parent_map(output.claim_tree)

    assert "m_detail" not in node_ids
    assert parent_by_child["r1"] == "m_main"


def test_assemble_tree_folds_suppressed_method_evidence_into_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Inference systems face memory pressure during long decoding.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_main",
            claim_type=ClaimType.method,
            statement="We introduce hierarchical cache compaction for long-context decoding.",
            source_section="3 Method",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["hierarchical cache compaction"],
        ),
        RawClaim(
            claim_id="m_short",
            claim_type=ClaimType.method,
            statement="We use PyTorch APIs.",
            source_section="Appendix Implementation Details",
            evidence=[EvidencePointer(artifact_id="table_9", role="supports")],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_main", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_short", parent_id="m_main", depends_on=["m_main"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    all_nodes = list(_iter_nodes(output.claim_tree))
    node_by_id = {node.claim_id: node for node in all_nodes}
    m_main_evidence_ids = {pointer.artifact_id for pointer in node_by_id["m_main"].evidence}

    assert "m_short" not in node_by_id
    assert {"fig_1", "table_9"} <= m_main_evidence_ids


def test_assemble_tree_repairs_primitive_under_system_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Contiguous KV allocation wastes memory and constrains throughput.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_primitive",
            claim_type=ClaimType.method,
            statement="PagedAttention is an attention algorithm that maps logical KV blocks to physical blocks.",
            source_section="4.1 PagedAttention",
            entity_names=["PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.mechanism),
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="vLLM leverages PagedAttention in an end-to-end serving system.",
            source_section="4 Method",
            entity_names=["vLLM", "PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_system", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_primitive", parent_id="m_system", depends_on=["m_system"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["m_primitive"] != "m_system"
    assert parent_by_child["m_primitive"] == "c1"


def test_assemble_tree_sanitizes_descendant_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="GPU memory pressure limits serving throughput.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We introduce paged KV cache management for serving.",
            source_section="4 Method",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
            claim_strength=3.6,
        ),
        RawClaim(
            claim_id="m2",
            claim_type=ClaimType.method,
            statement="A scheduler preempts sequence groups under memory pressure.",
            source_section="4.5 Scheduling",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.implementation_detail),
            evidence=[EvidencePointer(artifact_id="fig_sched", role="supports")],
            claim_strength=3.2,
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["m2"]),
                TreeNodeAssignment(claim_id="m2", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["m1"].depends_on == ["c1"]
    assert "m2" in node_by_id


def test_assemble_tree_blocks_observational_method_as_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="KV cache waste limits throughput.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="m_obs",
            claim_type=ClaimType.method,
            statement=(
                "The number of requests that can be batched together is constrained by GPU memory "
                "capacity, making serving memory-bound."
            ),
            source_section="3 Memory Challenges",
        ),
        RawClaim(
            claim_id="m_main",
            claim_type=ClaimType.method,
            statement="vLLM leverages paged KV cache management to increase batch size.",
            source_section="4 Method",
            entity_names=["vLLM"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="vLLM achieves 2x higher throughput than Orca at similar latency.",
            source_section="5 Results",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_obs", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_main", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r1", parent_id="m_obs", depends_on=["m_obs"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    parent_by_child = _parent_map(output.claim_tree)
    assert node_by_id["m_obs"].claim_type == ClaimType.context
    assert parent_by_child["r1"] == "m_main"


def test_assemble_tree_reattaches_derivative_context_under_core_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Toy Paper", authors=["A"], venue="Conf", year=2025)
    claims = [
        RawClaim(
            claim_id="c_core",
            claim_type=ClaimType.context,
            statement="Existing systems waste KV cache memory due to fragmentation bottlenecks.",
            source_section="1 Motivation",
        ),
        RawClaim(
            claim_id="c_derivative",
            claim_type=ClaimType.context,
            statement="Only 20-40 percent of KV cache memory stores actual token states in practice.",
            source_section="3.1 Memory Characterization",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We introduce paged KV cache management to reduce memory waste.",
            source_section="4 Method",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c_core", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="c_derivative", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c_core", depends_on=["c_core"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["c_derivative"] == "c_core"


def test_assemble_tree_deterministic_assigns_typed_semantic_contract() -> None:
    metadata = PaperMetadata(title="Typed Contract", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Existing runtimes waste KV cache memory due to fragmentation bottlenecks.",
            source_section="1 Introduction",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We implement a block-table mapping to allocate KV blocks on demand.",
            source_section="4 Method",
            entity_names=["block table", "KV cache"],
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="vLLM achieves 2x higher throughput than Orca at similar latency.",
            source_section="5 Evaluation",
            entity_names=["vLLM", "Orca"],
        ),
        RawClaim(
            claim_id="a1",
            claim_type=ClaimType.assumption,
            statement="The gain requires sufficient GPU memory bandwidth under heavy batching.",
            source_section="6 Discussion",
        ),
        RawClaim(
            claim_id="n1",
            claim_type=ClaimType.negative,
            statement="Compaction was rejected because it stalls decoding.",
            source_section="3 Design Choices",
            rejected_what="KV compaction",
            rejected_why="stalls decoding",
        ),
    ]

    output = asyncio.run(
        assemble_tree_deterministic(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[claims[-1]],
            artifacts=[],
            config={"pipeline": {"tree": {"ambiguity_budget": 0}}},
            claim_groups=[],
            support_details=[],
        )
    )

    for node in _iter_nodes(output.claim_tree):
        assert node.canonical_label
        assert node.normalized_statement
        assert re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", node.canonical_label)
        if node.claim_type == ClaimType.result:
            assert node.result_subtype is not None
        else:
            assert node.result_subtype is None

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["c1"].abstraction_level == AbstractionLevel.problem
    assert node_by_id["c1"].semantic_role == SemanticRole.problem
    assert node_by_id["m1"].abstraction_level in {
        AbstractionLevel.primitive,
        AbstractionLevel.system_realization,
        AbstractionLevel.submechanism,
    }
    assert node_by_id["a1"].semantic_role == SemanticRole.assumption
    assert node_by_id["n1"].semantic_role == SemanticRole.limitation


def test_assemble_tree_result_subtype_workload_precedence_over_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Subtype Boundary", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Workload variation affects serving behavior.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We implement paged KV block management.",
            source_section="4 Method",
            entity_names=["paged KV"],
        ),
        RawClaim(
            claim_id="r_workload",
            claim_type=ClaimType.result,
            statement=(
                "On the ShareGPT workload, long prompt-length distribution dominates request mix and "
                "exposes a memory-bound regime."
            ),
            source_section="5 Evaluation",
            entity_names=["ShareGPT"],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r_workload", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["r_workload"].result_subtype == ResultSubtype.workload_characterization
    assert node_by_id["r_workload"].semantic_role == SemanticRole.scoped_result


def test_assemble_tree_compacts_restatement_pair_into_support_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Compaction", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="KV cache fragmentation reduces throughput.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_parent",
            claim_type=ClaimType.method,
            statement="vLLM manages KV cache memory with block-table mapping.",
            source_section="4 Method",
            entity_names=["vLLM", "KV cache"],
        ),
        RawClaim(
            claim_id="m_child",
            claim_type=ClaimType.method,
            statement="vLLM manages KV cache memory with block-table mapping and on-demand allocation.",
            source_section="4 Method",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
            entity_names=["vLLM", "block table", "KV cache"],
        ),
    ]
    artifacts = [EvidenceArtifact(artifact_type="figure", artifact_id="fig_1", caption="Method.", source_page=3)]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_parent", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_child", parent_id="m_parent", depends_on=["m_parent"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=artifacts,
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert "m_child" in node_by_id
    assert "m_parent" not in node_by_id
    assert any(
        detail.anchor_claim_id == "m_child" and "manages KV cache memory" in detail.text
        for detail in output.support_details
    )


def test_assemble_tree_keeps_top_level_instantiation_in_structure_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Instantiation", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving systems waste KV cache memory due to fragmentation.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_primitive",
            claim_type=ClaimType.method,
            statement="PagedAttention is a primitive attention algorithm for paged KV management.",
            source_section="4.1 PagedAttention",
            entity_names=["PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="vLLM builds on PagedAttention to realize an end-to-end serving system.",
            source_section="4.2 System",
            entity_names=["vLLM", "PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_primitive", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_system", parent_id="m_primitive", depends_on=["m_primitive"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["m_system"] == "m_primitive"
    assert node_by_id["m_system"].depends_on == []


def test_assemble_tree_does_not_force_instantiation_with_contradictory_section_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Instantiation Boundary", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving systems waste KV cache memory due to fragmentation.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_primitive",
            claim_type=ClaimType.method,
            statement="PagedAttention is a primitive attention algorithm for paged KV management.",
            source_section="5.1 PagedAttention",
            entity_names=["PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="vLLM builds on PagedAttention to realize an end-to-end serving system.",
            source_section="3.1 Early Design",
            entity_names=["vLLM", "PagedAttention"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_primitive", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_system", parent_id="m_primitive", depends_on=["m_primitive"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["m_system"] == "c1"


def test_assemble_tree_biases_eviction_policy_to_submechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Abstraction", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Memory pressure limits serving throughput.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="vLLM is an end-to-end serving system built on paged KV memory management.",
            source_section="4 Method",
            entity_names=["vLLM"],
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="m_policy",
            claim_type=ClaimType.method,
            statement="An all-or-nothing eviction policy preempts sequence groups under memory pressure.",
            source_section="4.5 Scheduling and Preemption",
            entity_names=["all-or-nothing eviction policy"],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_system", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_policy", parent_id="c1", depends_on=["c1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["m_policy"] == "m_system"
    assert node_by_id["m_policy"].abstraction_level == AbstractionLevel.submechanism
    assert node_by_id["m_policy"].semantic_role == SemanticRole.method_support


def test_assemble_tree_reattaches_headline_result_from_submechanism_to_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Headline Reattach", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Memory waste limits throughput.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m_system",
            claim_type=ClaimType.method,
            statement="vLLM provides a serving system for paged KV cache management.",
            source_section="4 Method",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="m_kernel",
            claim_type=ClaimType.method,
            statement="A kernel-level block-table gather implementation reduces local overhead.",
            source_section="4.4 Kernel",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.implementation_detail),
        ),
        RawClaim(
            claim_id="r_headline",
            claim_type=ClaimType.result,
            statement="vLLM achieves 2-4x higher throughput than Orca at the same latency.",
            source_section="5 Results",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m_system", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="m_kernel", parent_id="m_system", depends_on=["m_system"]),
                TreeNodeAssignment(claim_id="r_headline", parent_id="m_kernel", depends_on=["m_kernel"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    parent_by_child = _parent_map(output.claim_tree)
    assert parent_by_child["r_headline"] == "m_system"
    assert node_by_id["r_headline"].result_subtype == ResultSubtype.headline_result


def test_assemble_tree_result_subtype_boundaries_for_scoped_and_constraint_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Subtype Boundaries", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Memory and kernel costs both impact serving performance.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We implement paged KV cache management.",
            source_section="4 Method",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
        ),
        RawClaim(
            claim_id="r_scoped",
            claim_type=ClaimType.result,
            statement="On the ShareGPT dataset, vLLM sustains 2x higher request rates than Orca.",
            source_section="5 Evaluation",
        ),
        RawClaim(
            claim_id="r_constraint",
            claim_type=ClaimType.result,
            statement="Attention kernel latency overhead is 20-26% higher than FasterTransformer.",
            source_section="5 Evaluation",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="r_scoped", parent_id="m1", depends_on=["m1"]),
                TreeNodeAssignment(claim_id="r_constraint", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["r_scoped"].result_subtype == ResultSubtype.mechanism_validation
    assert node_by_id["r_scoped"].semantic_role == SemanticRole.scoped_result
    assert node_by_id["r_constraint"].result_subtype == ResultSubtype.constraint_observation


def test_assemble_tree_retags_mechanism_assumption_to_method_with_method_abstraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Assumption Retag", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Serving memory pressure limits throughput.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="We introduce paged KV cache management.",
            source_section="4 Method",
            structural_hints=ClaimStructuralHints(local_role=ClaimLocalRole.top_level),
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
        RawClaim(
            claim_id="a_mech",
            claim_type=ClaimType.assumption,
            statement=(
                "The runtime allocates KV blocks on demand with a block table and copy-on-write sharing."
            ),
            source_section="4.3 Decoding",
            evidence=[EvidencePointer(artifact_id="fig_1", role="supports")],
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, response_schema, config)
        return TreeAssemblyOutput(
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            nodes=[
                TreeNodeAssignment(claim_id="c1", parent_id=None, depends_on=[]),
                TreeNodeAssignment(claim_id="m1", parent_id="c1", depends_on=["c1"]),
                TreeNodeAssignment(claim_id="a_mech", parent_id="m1", depends_on=["m1"]),
            ],
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"model_tier": "heavy"}}},
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["a_mech"].claim_type == ClaimType.method
    assert node_by_id["a_mech"].abstraction_level in {
        AbstractionLevel.primitive,
        AbstractionLevel.system_realization,
        AbstractionLevel.submechanism,
    }
    assert node_by_id["a_mech"].semantic_role in {SemanticRole.method_core, SemanticRole.method_support}


def test_assemble_tree_deterministic_accepts_valid_model_relabels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Label Relabel", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Fragmentation causes KV cache memory bottlenecks.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="PagedAttention partitions KV cache into fixed-size blocks.",
            source_section="4 Method",
            entity_names=["PagedAttention", "KV cache"],
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="The system improves throughput by 2x over baselines.",
            source_section="5 Results",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, config)
        assert response_schema is CanonicalLabelOutput
        return CanonicalLabelOutput(
            labels=[
                CanonicalLabelAssignment(claim_id="c1", canonical_label="kv_cache_fragmentation_bottleneck"),
                CanonicalLabelAssignment(claim_id="m1", canonical_label="pagedattention_kv_block_partitioning"),
                CanonicalLabelAssignment(claim_id="r1", canonical_label="throughput_gain_over_baselines"),
            ]
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree_deterministic(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"ambiguity_budget": 0, "canonical_label_relabel_tier": "small"}}},
            claim_groups=[],
            support_details=[],
        )
    )

    node_by_id = {node.claim_id: node for node in _iter_nodes(output.claim_tree)}
    assert node_by_id["c1"].canonical_label == "kv_cache_fragmentation_bottleneck"
    assert node_by_id["m1"].canonical_label == "pagedattention_kv_block_partitioning"
    assert node_by_id["r1"].canonical_label == "throughput_gain_over_baselines"


def test_assemble_tree_deterministic_falls_back_when_model_relabels_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = PaperMetadata(title="Label Fallback", authors=["A"])
    claims = [
        RawClaim(
            claim_id="c1",
            claim_type=ClaimType.context,
            statement="Fragmentation causes KV cache memory bottlenecks.",
            source_section="1 Intro",
        ),
        RawClaim(
            claim_id="m1",
            claim_type=ClaimType.method,
            statement="PagedAttention partitions KV cache into fixed-size blocks.",
            source_section="4 Method",
            entity_names=["PagedAttention", "KV cache"],
        ),
        RawClaim(
            claim_id="r1",
            claim_type=ClaimType.result,
            statement="The system improves throughput by 2x over baselines.",
            source_section="5 Results",
        ),
    ]

    async def _fake_call_model(
        tier: str,
        messages: list[dict[str, str]],
        response_schema: type[Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, config)
        assert response_schema is CanonicalLabelOutput
        return CanonicalLabelOutput(
            labels=[
                CanonicalLabelAssignment(claim_id="c1", canonical_label="sharegpt_dataset"),
                CanonicalLabelAssignment(claim_id="m1", canonical_label="bad label with spaces"),
                CanonicalLabelAssignment(claim_id="r1", canonical_label="sharegpt_dataset"),
            ]
        )

    monkeypatch.setattr(tree_prompts, "call_model", _fake_call_model)
    output = asyncio.run(
        assemble_tree_deterministic(
            metadata=metadata,
            claims=claims,
            faceted=[],
            negatives=[],
            artifacts=[],
            config={"pipeline": {"tree": {"ambiguity_budget": 0, "canonical_label_relabel_tier": "small"}}},
            claim_groups=[],
            support_details=[],
        )
    )

    labels = [node.canonical_label for node in _iter_nodes(output.claim_tree)]
    assert all(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", label) for label in labels)
    assert "sharegpt_dataset" not in labels
    assert len(labels) == len(set(labels))
