"""
Claim:
When large-model adjudication is enabled, extraction runs a final heavy cleanup
after the draft has passed deterministic normalization and validation.

Plausible wrong implementations:
- Skip the final cleanup even though the config enables it.
- Run cleanup before deterministic graph preservation and claim attachment.
- Spend a cleanup call when the configured model-call budget cannot cover it.
- Let cleanup bypass the final deterministic validation gate.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from paper_decomposer.extraction.contracts import (
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedNode,
    ExtractionCaps,
    PaperExtraction,
)
from paper_decomposer.extraction.stages import ExtractionDraft, MethodGraphDraft
from paper_decomposer.pipeline import extract_document
from paper_decomposer.schema import PaperDocument, PaperMetadata, RhetoricalRole, Section


def test_extract_document_runs_enabled_heavy_cleanup_after_valid_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _tiny_document()
    cleanup_inputs: list[PaperExtraction] = []

    async def fake_frontmatter(spans, *, config):
        return SimpleNamespace(candidates=[], model_dump_json=lambda: "{}")

    async def fake_method_graph(spans, sketch, *, config):
        return _tiny_method_graph("paper-id:section:1:span:1")

    async def fake_claims(spans, graph, *, config):
        return SimpleNamespace(model_dump_json=lambda: "{}")

    async def fake_compress(graph, claims, *, config):
        return ExtractionDraft(
            claims=[
                ExtractedClaim(
                    claim_id="c1",
                    paper_id="paper-id",
                    claim_type="capability",
                    raw_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                    finding="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                    evidence_span_ids=["paper-id:section:1:span:1"],
                )
            ]
        )

    async def fake_repair(extraction: PaperExtraction, validation_errors, *, config):
        raise AssertionError("valid draft should not need medium repair before heavy cleanup")

    async def fake_cleanup(extraction: PaperExtraction, validation_issues, *, config):
        cleanup_inputs.append(extraction)
        assert extraction.claims[0].method_ids == ["m1"]
        return ExtractionDraft(
            graph=extraction.graph,
            claims=[
                extraction.claims[0].model_copy(
                    update={
                        "metric": "cache-block translation",
                        "finding": "TinyAttention performs on-demand cache-block translation.",
                    }
                )
            ],
        )

    _patch_common_stages(monkeypatch, fake_frontmatter, fake_method_graph, fake_claims, fake_compress, fake_repair)
    monkeypatch.setattr("paper_decomposer.pipeline.cleanup_paper_extraction", fake_cleanup)

    extraction = asyncio.run(
        extract_document(
            document,
            config=_config(max_calls=5, enable_large_model_adjudication=True),
        )
    )

    assert len(cleanup_inputs) == 1
    assert extraction.claims[0].metric == "cache-block translation"
    assert extraction.claims[0].method_ids == ["m1"]


def test_extract_document_respects_heavy_cleanup_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_frontmatter(spans, *, config):
        return SimpleNamespace(candidates=[], model_dump_json=lambda: "{}")

    async def fake_method_graph(spans, sketch, *, config):
        return _tiny_method_graph("paper-id:section:1:span:1")

    async def fake_claims(spans, graph, *, config):
        return SimpleNamespace(model_dump_json=lambda: "{}")

    async def fake_compress(graph, claims, *, config):
        return ExtractionDraft(
            graph=_tiny_method_graph("paper-id:section:1:span:1").graph,
            claims=[
                ExtractedClaim(
                    claim_id="c1",
                    paper_id="paper-id",
                    claim_type="capability",
                    raw_text="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                    finding="TinyAttention maps logical cache blocks to physical cache blocks on demand.",
                    method_ids=["m1"],
                    evidence_span_ids=["paper-id:section:1:span:1"],
                )
            ],
        )

    async def fake_repair(extraction: PaperExtraction, validation_errors, *, config):
        raise AssertionError("valid draft should not need repair")

    async def fake_cleanup(extraction: PaperExtraction, validation_issues, *, config):
        raise AssertionError("cleanup should not run outside the model-call budget")

    _patch_common_stages(monkeypatch, fake_frontmatter, fake_method_graph, fake_claims, fake_compress, fake_repair)
    monkeypatch.setattr("paper_decomposer.pipeline.cleanup_paper_extraction", fake_cleanup)

    with pytest.raises(ValueError, match="large-model cleanup"):
        asyncio.run(
            extract_document(
                _tiny_document(),
                config=_config(max_calls=4, enable_large_model_adjudication=True),
            )
        )


def _tiny_document() -> PaperDocument:
    return PaperDocument(
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


def _tiny_method_graph(span_id: str) -> MethodGraphDraft:
    return MethodGraphDraft(
        nodes=[
            ExtractedNode(
                local_node_id="system",
                kind="system",
                canonical_name="Tiny System",
                description="Tiny serving system.",
                granularity_rationale="The paper presents Tiny System as the composed serving system.",
                evidence_span_ids=[span_id],
            ),
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
                evidence_span_ids=[span_id],
            ),
        ],
        edges=[
            ExtractedEdge(
                parent_id="system",
                child_id="m1",
                relation_kind="uses",
                evidence_span_ids=[span_id],
            )
        ],
    )


def _config(*, max_calls: int, enable_large_model_adjudication: bool) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            extraction={
                "max_model_calls_per_paper": max_calls,
                "max_input_chars_per_stage": 50_000,
                "enable_large_model_adjudication": enable_large_model_adjudication,
                "caps": ExtractionCaps().model_dump(),
            }
        )
    )


def _patch_common_stages(
    monkeypatch: pytest.MonkeyPatch,
    fake_frontmatter,
    fake_method_graph,
    fake_claims,
    fake_compress,
    fake_repair,
) -> None:
    monkeypatch.setattr("paper_decomposer.pipeline.uuid5", lambda namespace, name: "paper-id")
    monkeypatch.setattr("paper_decomposer.pipeline.uuid4", lambda: "run-id")
    monkeypatch.setattr("paper_decomposer.pipeline.extract_frontmatter_sketch", fake_frontmatter)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_method_graph", fake_method_graph)
    monkeypatch.setattr("paper_decomposer.pipeline.extract_claims_and_outcomes", fake_claims)
    monkeypatch.setattr("paper_decomposer.pipeline.compress_paper_extraction", fake_compress)
    monkeypatch.setattr("paper_decomposer.pipeline.repair_paper_extraction", fake_repair)
