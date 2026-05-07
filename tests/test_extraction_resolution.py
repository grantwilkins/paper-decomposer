"""
Claim:
The resolver stub emits paper-local entity tasks for later global resolution
without assigning global IDs or collapsing local discovery.

Plausible wrong implementations:
- Treat paper-local IDs as global canonical IDs.
- Drop mechanism signatures needed for later matching.
- Ignore problems as resolvable local entities.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedNode,
    ExtractedSetting,
    MechanismSignature,
    PaperExtraction,
    ProblemStatement,
)
from paper_decomposer.extraction.resolution import build_local_entity_resolution_tasks


def test_resolution_tasks_cover_local_entities_without_global_identity() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Example",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Method",
                section_kind="method",
                text="PagedAttention addresses KV-cache fragmentation by mapping logical blocks.",
            )
        ],
        problems=[
            ProblemStatement(
                problem_id="local:problem:kv_fragmentation",
                statement="KV-cache fragmentation wastes memory.",
                evidence_span_ids=["s1"],
            )
        ],
        nodes=[
            ExtractedNode(
                local_node_id="local:system:vllm",
                kind="system",
                canonical_name="vLLM",
                description="Paper-local serving system.",
                problem_ids=["local:problem:kv_fragmentation"],
                granularity_rationale="Top-level artifact.",
                evidence_span_ids=["s1"],
            ),
            ExtractedNode(
                local_node_id="local:method:pagedattention",
                kind="method",
                canonical_name="PagedAttention",
                description="Paged KV-cache management.",
                problem_ids=["local:problem:kv_fragmentation"],
                granularity_rationale="Reusable method.",
                mechanism_sentence="Given logical blocks, it outputs physical block mappings.",
                mechanism_signature=MechanismSignature(
                    problem="KV-cache fragmentation wastes memory.",
                    inputs=["logical KV blocks"],
                    outputs=["physical block mappings"],
                    operative_move="Map logical blocks to physical blocks on demand.",
                ),
                evidence_span_ids=["s1"],
            ),
        ],
        settings=[
            ExtractedSetting(
                local_setting_id="local:setting:llm_serving",
                kind="application",
                canonical_name="LLM serving",
                description="Serving application.",
                evidence_span_ids=["s1"],
            )
        ],
    )

    tasks = build_local_entity_resolution_tasks(extraction)

    assert [task.local_entity_id for task in tasks] == [
        "local:system:vllm",
        "local:method:pagedattention",
        "local:setting:llm_serving",
        "local:problem:kv_fragmentation",
    ]
    method = tasks[1]
    assert method.mechanism_signature is not None
    assert method.allowed_relations == [
        "same_as",
        "variant_of",
        "uses",
        "extends",
        "subsumes",
        "is_subsumed_by",
        "reimplements",
        "compared_against",
        "distinct",
        "uncertain",
    ]
    assert not hasattr(method, "global_node_id")
