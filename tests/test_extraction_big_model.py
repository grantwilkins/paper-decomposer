"""
Claim:
The experimental big-model extraction path runs one compact paper-local draft
per candidate tier, preserves candidate order for MiniMax-vs-DeepSeek
comparison, and uses at most one targeted repair call for invalid drafts.

Plausible wrong implementations:
- Accidentally call the old staged sketch/graph/claims/compression chain.
- Keep the old four-call minimum and reject the one-call big-model path.
- Repair every draft, even when deterministic validation already passes.
- Resend unrelated long evidence during repair instead of targeted spans.
- Sort or deduplicate comparison candidates in a way that changes configured order.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from paper_decomposer import pipeline
from paper_decomposer.extraction.contracts import (
    BigModelComparison,
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedNode,
    ExtractionCaps,
    ExtractionValidationReport,
    PaperExtraction,
)
from paper_decomposer.extraction.stages import ExtractionDraft
from paper_decomposer.schema import PaperDocument, PaperMetadata, RhetoricalRole, Section


def test_big_model_strategy_uses_one_compact_draft_without_staged_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    draft_calls: list[dict[str, Any]] = []

    async def fail_staged_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("staged extraction should not run for big_model_draft strategy")

    async def fake_big_model_draft(spans, *, config, tier, caps):
        draft_calls.append({"span_ids": [span.span_id for span in spans], "tier": tier, "caps": caps})
        return _valid_draft("paper-id", spans[0].span_id)

    async def fail_repair(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("valid draft should not be repaired")

    monkeypatch.setattr(pipeline, "uuid5", lambda namespace, name: "paper-id")
    monkeypatch.setattr(pipeline, "uuid4", lambda: "run-id")
    monkeypatch.setattr(pipeline, "extract_frontmatter_sketch", fail_staged_call)
    monkeypatch.setattr(pipeline, "extract_method_graph", fail_staged_call)
    monkeypatch.setattr(pipeline, "extract_claims_and_outcomes", fail_staged_call)
    monkeypatch.setattr(pipeline, "compress_paper_extraction", fail_staged_call)
    monkeypatch.setattr(pipeline, "extract_big_model_draft", fake_big_model_draft)
    monkeypatch.setattr(pipeline, "repair_paper_extraction", fail_repair)

    extraction = asyncio.run(
        pipeline.extract_document(
            document,
            config=_config(
                {
                    "strategy": "big_model_draft",
                    "max_model_calls_per_paper": 1,
                    "big_model_draft_tier": "heavy",
                }
            ),
        )
    )

    assert draft_calls == [
        {
            "span_ids": ["paper-id:section:1:span:1"],
            "tier": "heavy",
            "caps": ExtractionCaps(),
        }
    ]
    assert extraction.extraction_run_id == "run-id"
    assert [node.canonical_name for node in extraction.nodes] == ["Tiny System", "TinyAttention"]


def test_big_model_strategy_repairs_invalid_draft_with_targeted_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="Tiny System"),
        sections=[
            Section(
                title="Method",
                role=RhetoricalRole.method,
                body_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                char_count=75,
            ),
            Section(
                title="Evaluation",
                role=RhetoricalRole.evaluation,
                body_text="Unrelated evaluation detail. " * 600,
                char_count=16_800,
            ),
        ],
    )
    repair_calls: list[list[str]] = []

    async def fake_big_model_draft(spans, *, config, tier, caps):
        return ExtractionDraft()

    async def fake_repair(extraction: PaperExtraction, validation_errors, *, config, tier=None):
        repair_calls.append([span.section_title for span in extraction.evidence_spans])
        return _valid_draft(extraction.paper_id, extraction.evidence_spans[0].span_id)

    monkeypatch.setattr(pipeline, "uuid5", lambda namespace, name: "paper-id")
    monkeypatch.setattr(pipeline, "uuid4", lambda: "run-id")
    monkeypatch.setattr(pipeline, "extract_big_model_draft", fake_big_model_draft)
    monkeypatch.setattr(pipeline, "repair_paper_extraction", fake_repair)

    extraction = asyncio.run(
        pipeline.extract_document(
            document,
            config=_config(
                {
                    "strategy": "big_model_draft",
                    "max_model_calls_per_paper": 2,
                    "big_model_draft_tier": "medium",
                    "targeted_repair_max_chars": 500,
                }
            ),
        )
    )

    assert repair_calls == [["Method"]]
    assert extraction.nodes[1].canonical_name == "TinyAttention"


def test_big_model_comparison_preserves_candidate_order_and_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    run_tiers: list[str] = []
    reset_calls = 0

    async def fake_run_big_model_extraction(document, *, config, draft_tier):
        run_tiers.append(draft_tier)
        extraction = PaperExtraction(paper_id="paper-id", extraction_run_id=f"run-{draft_tier}", title="Tiny System")
        return extraction, ExtractionValidationReport(), draft_tier == "heavy"

    def fake_reset_cost_tracker() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(pipeline, "_run_big_model_extraction", fake_run_big_model_extraction)
    monkeypatch.setattr(pipeline, "reset_cost_tracker", fake_reset_cost_tracker)
    monkeypatch.setattr(pipeline, "get_cost_tracker", lambda: {"total_cost_usd": 0.25})

    comparison = asyncio.run(
        pipeline.compare_document_extractions(
            document,
            config=SimpleNamespace(
                models=SimpleNamespace(
                    medium=SimpleNamespace(model="MiniMaxAI/MiniMax-M2.7"),
                    heavy=SimpleNamespace(model="deepseek-ai/DeepSeek-V4-Pro"),
                ),
                pipeline=SimpleNamespace(extraction={"comparison_model_tiers": ["medium", "heavy", "medium"]}),
            ),
        )
    )

    assert isinstance(comparison, BigModelComparison)
    assert run_tiers == ["medium", "heavy"]
    assert reset_calls == 2
    assert [result.model for result in comparison.results] == [
        "MiniMaxAI/MiniMax-M2.7",
        "deepseek-ai/DeepSeek-V4-Pro",
    ]
    assert [result.used_repair for result in comparison.results] == [False, True]


def _document() -> PaperDocument:
    return PaperDocument(
        metadata=PaperMetadata(title="Tiny System"),
        sections=[
            Section(
                title="Method",
                role=RhetoricalRole.method,
                body_text=(
                    "TinyAttention maps logical cache blocks to physical cache blocks on demand. "
                    "Tiny System improves request rate."
                ),
                char_count=110,
            )
        ],
    )


def _config(overrides: dict[str, Any]) -> SimpleNamespace:
    extraction = {
        "max_input_chars_per_big_model": 50_000,
        "include_captions": True,
        "include_table_text": True,
        "caps": ExtractionCaps().model_dump(),
        **overrides,
    }
    return SimpleNamespace(pipeline=SimpleNamespace(extraction=extraction))


def _valid_draft(paper_id: str, span_id: str) -> ExtractionDraft:
    return ExtractionDraft(
        nodes=[
            ExtractedNode(
                local_node_id="sys_tiny",
                kind="system",
                canonical_name="Tiny System",
                description="A compact serving system.",
                granularity_rationale="The paper presents Tiny System as the composed artifact.",
                evidence_span_ids=[span_id],
            ),
            ExtractedNode(
                local_node_id="meth_tinyattention",
                kind="method",
                canonical_name="TinyAttention",
                description="Logical-to-physical cache block mapping.",
                granularity_rationale="It is the reusable cache mapping mechanism.",
                mechanism_sentence=(
                    "Given logical cache blocks, TinyAttention outputs physical cache blocks by mapping "
                    "addresses on demand."
                ),
                evidence_span_ids=[span_id],
            ),
        ],
        edges=[
            ExtractedEdge(
                parent_id="sys_tiny",
                child_id="meth_tinyattention",
                relation_kind="uses",
                evidence_span_ids=[span_id],
            )
        ],
        claims=[
            ExtractedClaim(
                claim_id="c1",
                paper_id=paper_id,
                claim_type="performance",
                raw_text="Tiny System improves request rate.",
                finding="Tiny System improves request rate.",
                method_ids=["sys_tiny"],
                evidence_span_ids=[span_id],
            )
        ],
    )
