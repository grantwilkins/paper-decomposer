"""
Claim:
Extraction repair runs once after blocking validation failures and can replace
paper-outline-shaped method output with a valid paper-local graph.

Plausible wrong implementations:
- Raise immediately on the first deterministic validation failure.
- Retry the full extraction instead of repairing the invalid final JSON.
- Keep invalid section-heading nodes after repair.
- Accept repaired methods that still lack mechanism sentences.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedNode,
    ExtractionCaps,
    PaperExtraction,
)
from paper_decomposer.extraction.stages import ExtractionDraft
from paper_decomposer.pipeline import extract_document
from paper_decomposer.schema import PaperDocument, PaperMetadata, RhetoricalRole, Section


def test_extract_document_repairs_invalid_method_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="Tiny System"),
        sections=[
            Section(
                title="Method",
                role=RhetoricalRole.method,
                body_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                char_count=75,
            )
        ],
    )

    async def fake_frontmatter(spans, *, config):
        return SimpleNamespace(candidates=[], model_dump_json=lambda: "{}")

    async def fake_method_graph(spans, sketch, *, config):
        return SimpleNamespace(model_dump_json=lambda *args, **kwargs: "{}")

    async def fake_claims(spans, graph, *, config):
        return SimpleNamespace(model_dump_json=lambda: "{}")

    async def fake_compress(graph, claims, *, config):
        return ExtractionDraft(
            nodes=[
                ExtractedNode(
                    local_node_id="bad-method",
                    kind="method",
                    canonical_name="Method",
                    description="Section-shaped node.",
                    granularity_rationale="Incorrectly promoted section heading.",
                    evidence_span_ids=["paper-id:section:1:span:1"],
                )
            ]
        )

    async def fake_repair(extraction: PaperExtraction, validation_errors, *, config):
        assert {error.code for error in validation_errors} == {
            "method_missing_mechanism_sentence",
            "section_heading_promoted",
        }
        return ExtractionDraft(
            nodes=[
                ExtractedNode(
                    local_node_id="m1",
                    kind="method",
                    canonical_name="TinyAttention",
                    description="Cache-block address translation mechanism.",
                    granularity_rationale="It defines a reusable block translation mechanism.",
                    mechanism_sentence=(
                        "Given logical cache blocks, TinyAttention outputs physical cache blocks by translating "
                        "block addresses on demand."
                    ),
                    evidence_span_ids=[extraction.evidence_spans[0].span_id],
                )
            ]
        )

    monkeypatch.setattr("paper_decomposer.pipeline.uuid5", lambda namespace, name: "paper-id")
    monkeypatch.setattr("paper_decomposer.pipeline.uuid4", lambda: "run-id")
    monkeypatch.setattr("paper_decomposer.pipeline.extract_frontmatter_sketch", fake_frontmatter)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_method_graph", fake_method_graph)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_claims_and_outcomes", fake_claims)
    monkeypatch.setattr("paper_decomposer.pipeline.compress_paper_extraction", fake_compress)
    monkeypatch.setattr("paper_decomposer.pipeline.repair_paper_extraction", fake_repair)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "max_model_calls_per_paper": 5,
                "max_input_chars_per_stage": 50_000,
                "caps": ExtractionCaps().model_dump(),
            }
        )
    )

    extraction = asyncio.run(extract_document(document, config=config))

    assert [node.canonical_name for node in extraction.nodes] == ["TinyAttention"]
    assert extraction.nodes[0].mechanism_sentence is not None


def test_extract_document_respects_repair_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="Tiny System"),
        sections=[
            Section(
                title="Method",
                role=RhetoricalRole.method,
                body_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                char_count=75,
            )
        ],
    )

    async def fake_frontmatter(spans, *, config):
        return SimpleNamespace(candidates=[], model_dump_json=lambda: "{}")

    async def fake_method_graph(spans, sketch, *, config):
        return SimpleNamespace(model_dump_json=lambda *args, **kwargs: "{}")

    async def fake_claims(spans, graph, *, config):
        return SimpleNamespace(model_dump_json=lambda: "{}")

    async def fake_compress(graph, claims, *, config):
        return ExtractionDraft(
            nodes=[
                ExtractedNode(
                    local_node_id="bad-method",
                    kind="method",
                    canonical_name="Method",
                    description="Section-shaped node.",
                    granularity_rationale="Incorrectly promoted section heading.",
                    evidence_span_ids=["paper-id:section:1:span:1"],
                )
            ]
        )

    async def fake_repair(extraction: PaperExtraction, validation_errors, *, config):
        raise AssertionError("repair should not run when max_model_calls_per_paper is 4")

    monkeypatch.setattr("paper_decomposer.pipeline.uuid5", lambda namespace, name: "paper-id")
    monkeypatch.setattr("paper_decomposer.pipeline.uuid4", lambda: "run-id")
    monkeypatch.setattr("paper_decomposer.pipeline.extract_frontmatter_sketch", fake_frontmatter)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_method_graph", fake_method_graph)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_claims_and_outcomes", fake_claims)
    monkeypatch.setattr("paper_decomposer.pipeline.compress_paper_extraction", fake_compress)
    monkeypatch.setattr("paper_decomposer.pipeline.repair_paper_extraction", fake_repair)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "max_model_calls_per_paper": 4,
                "max_input_chars_per_stage": 50_000,
                "caps": ExtractionCaps().model_dump(),
            }
        )
    )

    with pytest.raises(ValueError, match="outside the model-call budget"):
        asyncio.run(extract_document(document, config=config))


def test_repair_prompt_names_validation_errors_and_evidence() -> None:
    from paper_decomposer.extraction.contracts import ExtractionValidationError, ValidationSeverity
    from paper_decomposer.extraction.prompts import repair_prompt

    extraction = PaperExtraction(
        paper_id="p1",
        extraction_run_id="r1",
        title="Tiny",
        evidence_spans=[
            EvidenceSpan(
                span_id="s1",
                paper_id="p1",
                section_title="Method",
                section_kind="method",
                text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
            )
        ],
    )
    messages = repair_prompt(
        extraction.model_dump_json(),
        [
            ExtractionValidationError(
                code="method_missing_mechanism_sentence",
                message="Method node must state inputs, outputs, and operative move.",
                severity=ValidationSeverity.error,
                object_kind="node",
                object_id="m1",
            )
        ],
        extraction.evidence_spans,
    )

    content = messages[1]["content"]
    assert "method_missing_mechanism_sentence" in content
    assert "[s1]" in content
    assert "mechanism_sentence" in content
