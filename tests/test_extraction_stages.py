"""
Claim:
LLM extraction stages use cheap/default tiers for normal draft work, reserve
repair_model_tier for the optional repair call, reserve adjudication_model_tier
for final heavy cleanup, and never leak the nonexistent `cheap` tier into the
Together client.

Plausible wrong implementations:
- Pass `cheap` directly to `call_model` even though runtime tiers are small,
  medium, and heavy.
- Ignore extraction config and always call a stronger model.
- Route normal compression or cheap repair through the large adjudication tier.
- Route final cleanup through the normal draft tier instead of the adjudication tier.
- Build stage prompts without evidence span IDs, making grounding impossible.
- Ignore the explicit big-model comparison tier and fall back to default_model_tier.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractionCaps,
    ExtractionValidationError,
    PaperExtraction,
    ValidationSeverity,
)
from paper_decomposer.extraction.stages import (
    ClaimsOutcomesDraft,
    ExtractionDraft,
    FrontmatterSketch,
    MethodGraphDraft,
    cleanup_paper_extraction,
    compress_paper_extraction,
    extract_big_model_draft,
    extract_frontmatter_sketch,
    repair_paper_extraction,
)


def test_frontmatter_stage_maps_cheap_to_small_and_includes_span_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call_model(tier: str, messages: list[dict[str, str]], response_schema: type, config: Any) -> Any:
        calls.append({"tier": tier, "messages": messages, "response_schema": response_schema})
        return FrontmatterSketch(central_problem_candidates=["serving throughput"])

    monkeypatch.setattr("paper_decomposer.extraction.stages.call_model", fake_call_model)
    config = SimpleNamespace(pipeline=SimpleNamespace(extraction={"default_model_tier": "cheap"}))
    spans = [
        EvidenceSpan(
            span_id="s1",
            paper_id="paper-1",
            section_title="Abstract",
            section_kind="abstract",
            text="ORCA improves LLM serving with iteration-level scheduling.",
            source_kind="abstract",
        )
    ]

    result = asyncio.run(extract_frontmatter_sketch(spans, config=config))

    assert result.central_problem_candidates == ["serving throughput"]
    assert calls[0]["tier"] == "small"
    assert "[s1]" in calls[0]["messages"][1]["content"]
    assert calls[0]["response_schema"] is FrontmatterSketch


def test_compression_uses_default_tier_not_large_adjudication(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call_model(tier: str, messages: list[dict[str, str]], response_schema: type, config: Any) -> Any:
        calls.append({"tier": tier, "messages": messages, "response_schema": response_schema})
        return ExtractionDraft()

    monkeypatch.setattr("paper_decomposer.extraction.stages.call_model", fake_call_model)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "default_model_tier": "medium",
                "repair_model_tier": "medium",
                "adjudication_model_tier": "heavy",
                "enable_large_model_adjudication": False,
            }
        )
    )

    asyncio.run(compress_paper_extraction(MethodGraphDraft(), ClaimsOutcomesDraft(), config=config))

    assert calls[0]["tier"] == "medium"
    assert calls[0]["response_schema"] is ExtractionDraft


def test_big_model_draft_uses_explicit_tier_and_full_evidence_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call_model(tier: str, messages: list[dict[str, str]], response_schema: type, config: Any) -> Any:
        calls.append({"tier": tier, "messages": messages, "response_schema": response_schema})
        return ExtractionDraft()

    monkeypatch.setattr("paper_decomposer.extraction.stages.call_model", fake_call_model)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "default_model_tier": "small",
                "big_model_draft_tier": "medium",
            }
        )
    )
    spans = [
        EvidenceSpan(
            span_id="s1",
            paper_id="paper-1",
            section_title="Abstract",
            section_kind="abstract",
            text="Tiny System introduces TinyAttention.",
            source_kind="abstract",
        ),
        EvidenceSpan(
            span_id="s2",
            paper_id="paper-1",
            section_title="Evaluation",
            section_kind="evaluation",
            text="Tiny System improves request rate by 2x.",
        ),
    ]

    asyncio.run(extract_big_model_draft(spans, config=config, tier="heavy", caps=ExtractionCaps(max_claims=8)))

    assert calls[0]["tier"] == "heavy"
    assert calls[0]["response_schema"] is ExtractionDraft
    assert "[s1]" in calls[0]["messages"][1]["content"]
    assert "[s2]" in calls[0]["messages"][1]["content"]
    assert "compact claims <= 8" in calls[0]["messages"][1]["content"]


def test_repair_uses_repair_tier_not_large_adjudication(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call_model(tier: str, messages: list[dict[str, str]], response_schema: type, config: Any) -> Any:
        calls.append({"tier": tier, "messages": messages, "response_schema": response_schema})
        return ExtractionDraft()

    monkeypatch.setattr("paper_decomposer.extraction.stages.call_model", fake_call_model)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "default_model_tier": "medium",
                "repair_model_tier": "cheap",
                "adjudication_model_tier": "heavy",
                "enable_large_model_adjudication": False,
            }
        )
    )
    extraction = PaperExtraction(paper_id="paper-1", extraction_run_id="run-1", title="Tiny")
    errors = [
        ExtractionValidationError(
            code="claims_unattached",
            message="Claim must attach to graph.",
            severity=ValidationSeverity.error,
        )
    ]

    asyncio.run(repair_paper_extraction(extraction, errors, config=config))

    assert calls[0]["tier"] == "small"
    assert calls[0]["response_schema"] is ExtractionDraft


def test_cleanup_uses_adjudication_tier_and_includes_validation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call_model(tier: str, messages: list[dict[str, str]], response_schema: type, config: Any) -> Any:
        calls.append({"tier": tier, "messages": messages, "response_schema": response_schema})
        return ExtractionDraft()

    monkeypatch.setattr("paper_decomposer.extraction.stages.call_model", fake_call_model)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "default_model_tier": "medium",
                "repair_model_tier": "medium",
                "adjudication_model_tier": "heavy",
            }
        )
    )
    extraction = PaperExtraction(
        paper_id="paper-1",
        extraction_run_id="run-1",
        title="Tiny",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="paper-1",
                section_title="Evaluation",
                section_kind="evaluation",
                text="Tiny System sustains 2x higher request rate.",
            )
        ],
    )
    issues = [
        ExtractionValidationError(
            code="structured_claim_without_outcome",
            message="Claim has metric and comparison fields but no outcome.",
            severity=ValidationSeverity.warning,
            object_kind="claim",
            object_id="c1",
        )
    ]

    asyncio.run(cleanup_paper_extraction(extraction, issues, config=config))

    assert calls[0]["tier"] == "heavy"
    assert calls[0]["response_schema"] is ExtractionDraft
    assert "structured_claim_without_outcome" in calls[0]["messages"][1]["content"]
    assert "[s1]" in calls[0]["messages"][1]["content"]
