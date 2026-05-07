"""
Claim:
The DB write plan preserves paper-local identity and evidence while translating
only method-family relations into method_edges.

Plausible wrong implementations:
- Globally deduplicate by canonical_name and lose paper-local IDs.
- Encode method-to-setting applicability as a method edge.
- Drop evidence span IDs after validation succeeds.
- Write invalid extraction output despite blocking validation errors.
"""

from __future__ import annotations

import pytest

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedSetting,
    PaperExtraction,
)
from paper_decomposer.extraction.db_write_plan import ExtractionPersistenceError, build_db_write_plan


def _extraction() -> PaperExtraction:
    return PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="vLLM",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Design",
                section_kind="method",
                text="PagedAttention maps logical KV cache blocks to physical blocks on demand for LLaMA-13B.",
            )
        ],
        nodes=[
            ExtractedNode(
                local_node_id="system",
                kind="system",
                canonical_name="vLLM",
                description="LLM serving system.",
                granularity_rationale="The paper introduces it as the top-level system.",
                evidence_span_ids=["s1"],
            ),
            ExtractedNode(
                local_node_id="m1",
                kind="method",
                canonical_name="PagedAttention",
                description="KV cache management mechanism.",
                granularity_rationale="It is a reusable mechanism for KV cache address translation.",
                mechanism_sentence="Given logical KV blocks, PagedAttention outputs physical block addresses by mapping blocks on demand.",
                evidence_span_ids=["s1"],
            ),
        ],
        edges=[ExtractedEdge(parent_id="system", child_id="m1", relation_kind="uses", evidence_span_ids=["s1"])],
        settings=[
            ExtractedSetting(
                local_setting_id="model-1",
                kind="model_artifact",
                canonical_name="LLaMA-13B",
                description="Evaluated model artifact.",
                evidence_span_ids=["s1"],
            )
        ],
        method_setting_links=[
            ExtractedMethodSettingLink(
                method_id="m1",
                setting_id="model-1",
                relation_kind="applies_to",
                evidence_span_ids=["s1"],
            )
        ],
        claims=[
            ExtractedClaim(
                claim_id="claim-1",
                paper_id="paper-1",
                claim_type="capability",
                raw_text="PagedAttention maps logical KV cache blocks to physical blocks on demand for LLaMA-13B.",
                finding="PagedAttention maps logical KV cache blocks to physical blocks on demand.",
                method_ids=["m1"],
                setting_ids=["model-1"],
                evidence_span_ids=["s1"],
            )
        ],
    )


def test_write_plan_preserves_local_ids_and_evidence() -> None:
    plan = build_db_write_plan(_extraction())

    method = next(row for row in plan.methods if row["local_node_id"] == "m1")
    assert method["paper_id"] == "paper-1"
    assert method["extraction_run_id"] == "run-1"
    assert method["canonical_name"] == "PagedAttention"
    assert method["metadata"]["evidence_span_ids"] == ["s1"]

    setting = plan.settings[0]
    assert setting["kind"] == "model_artifact"
    assert setting["local_setting_id"] == "model-1"
    assert setting["metadata"]["evidence_span_ids"] == ["s1"]

    assert any(
        link["target_kind"] == "method" and link["local_target_id"] == "m1"
        for link in plan.local_evidence_links
    )
    assert any(
        link["target_kind"] == "setting" and link["local_target_id"] == "model-1"
        for link in plan.local_evidence_links
    )


def test_write_plan_keeps_applicability_out_of_method_edges() -> None:
    plan = build_db_write_plan(_extraction())

    assert plan.method_edges == [
        {
            "local_edge_id": "system->m1:uses",
            "parent_local_node_id": "system",
            "child_local_node_id": "m1",
            "relation": "composes",
            "confidence": 0.0,
            "metadata": {"paper_local_relation": "uses", "evidence_span_ids": ["s1"]},
        }
    ]
    assert plan.method_setting_links[0]["relation"] == "applies_to"


def test_write_plan_rejects_blocking_validation_errors() -> None:
    extraction = _extraction()
    extraction.edges[0].child_id = "missing"

    with pytest.raises(ExtractionPersistenceError) as exc_info:
        build_db_write_plan(extraction)

    assert "edge_endpoint_missing" in {error.code for error in exc_info.value.errors}
