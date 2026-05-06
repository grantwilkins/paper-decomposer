"""
Claim:
LLM extraction stages use the configured cheap model tier without leaking the
nonexistent `cheap` tier into the Together client, and prompts stay focused on
the selected evidence spans.

Plausible wrong implementations:
- Pass `cheap` directly to `call_model` even though runtime tiers are small,
  medium, and heavy.
- Ignore extraction config and always call a stronger model.
- Build stage prompts without evidence span IDs, making grounding impossible.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from paper_decomposer.extraction.contracts import EvidenceSpan
from paper_decomposer.extraction.stages import FrontmatterSketch, extract_frontmatter_sketch


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
