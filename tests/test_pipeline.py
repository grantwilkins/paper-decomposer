from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import paper_decomposer.pipeline as pipeline_module
from paper_decomposer.models import get_cost_tracker
from paper_decomposer.pipeline import decompose_paper
from paper_decomposer.schema import (
    ClaimGroup,
    ClaimLocalRole,
    ClaimNode,
    ClaimType,
    ParentPreference,
    OneLiner,
    PaperDecomposition,
    PaperDocument,
    PaperMetadata,
    RawClaim,
    Section,
    SectionExtractionOutput,
    SeedOutput,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
FIXTURE_PDF = ROOT / "fixtures" / "vllm.pdf"

requires_api_key = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY is not set.",
)


def _iter_nodes(nodes: list[ClaimNode]):
    for node in nodes:
        yield node
        yield from _iter_nodes(node.children)


def _tree_depth(nodes: list[ClaimNode]) -> int:
    if not nodes:
        return 0

    def node_depth(node: ClaimNode) -> int:
        if not node.children:
            return 1
        return 1 + max(node_depth(child) for child in node.children)

    return max(node_depth(node) for node in nodes)


def _test_runtime_config(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            output={"output_dir": str(output_dir)},
            seed={"model_tier": "small"},
            section_extraction={"model_tier": "small", "max_concurrent": 1},
            dedup={"model_tier": "medium"},
            tree={"model_tier": "heavy"},
        )
    )


@pytest.mark.api
@requires_api_key
def test_pipeline_integration_vllm(tmp_path: Path) -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    config_data["pipeline"]["output"]["output_dir"] = str(tmp_path)
    test_config = tmp_path / "config.pipeline.test.yaml"
    test_config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    result = asyncio.run(decompose_paper(str(FIXTURE_PDF), str(test_config)))

    assert isinstance(result, PaperDecomposition)

    output_files = sorted(tmp_path.glob("*.json"))
    assert output_files, "Expected pipeline to write at least one output JSON file."

    output_path = output_files[-1]
    parsed = PaperDecomposition.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert len(parsed.claim_tree) >= 3
    assert len(parsed.negative_claims) >= 1
    assert 0.0 < parsed.extraction_cost_usd < 0.50

    claim_count = sum(1 for _ in _iter_nodes(parsed.claim_tree))
    tree_depth = _tree_depth(parsed.claim_tree)
    cost_breakdown = get_cost_tracker()

    print(f"\nOutput JSON: {output_path}")
    print(
        "Cost breakdown: "
        f"calls={cost_breakdown['total_calls']} "
        f"prompt_tokens={cost_breakdown['prompt_tokens']} "
        f"completion_tokens={cost_breakdown['completion_tokens']} "
        f"total_usd={cost_breakdown['total_cost_usd']:.4f}"
    )
    print(
        "Tree summary: "
        f"roots={len(parsed.claim_tree)} "
        f"nodes={claim_count} "
        f"depth={tree_depth} "
        f"negatives={len(parsed.negative_claims)}"
    )


def test_pipeline_aborts_when_preflight_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    config = _test_runtime_config(tmp_path)
    document = PaperDocument(
        metadata=PaperMetadata(title="Test Paper", authors=["A"]),
        sections=[
            Section(
                section_number=None,
                title="Abstract",
                role="abstract",
                body_text="Abstract text.",
                char_count=14,
            ),
            Section(
                section_number="1",
                title="Method",
                role="method",
                body_text="Method details.",
                char_count=15,
            ),
        ],
        all_artifacts=[],
    )

    monkeypatch.setattr(pipeline_module, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda *_: document)

    async def _fail_preflight(*args, **kwargs):
        raise RuntimeError("preflight down")

    monkeypatch.setattr(pipeline_module, "preflight_model_tiers", _fail_preflight)

    with pytest.raises(RuntimeError, match="preflight down"):
        asyncio.run(decompose_paper(str(fake_pdf), "ignored.yaml"))


def test_pipeline_aborts_on_zero_phase2_claims(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    config = _test_runtime_config(tmp_path)
    document = PaperDocument(
        metadata=PaperMetadata(title="Test Paper", authors=["A"]),
        sections=[
            Section(
                section_number=None,
                title="Abstract",
                role="abstract",
                body_text="Abstract text.",
                char_count=14,
            ),
            Section(
                section_number="1",
                title="Method",
                role="method",
                body_text="Method details.",
                char_count=15,
            ),
        ],
        all_artifacts=[],
    )

    monkeypatch.setattr(pipeline_module, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda *_: document)

    async def _ok_preflight(*args, **kwargs):
        return None

    async def _seed(*args, **kwargs):
        return SeedOutput(
            claims=[
                RawClaim(
                    claim_id="s1",
                    claim_type=ClaimType.context,
                    statement="Need better memory management.",
                    source_section="Abstract",
                )
            ]
        )

    async def _empty_section(*args, **kwargs):
        return SectionExtractionOutput(claims=[])

    monkeypatch.setattr(pipeline_module, "preflight_model_tiers", _ok_preflight)
    monkeypatch.setattr(pipeline_module, "extract_seed", _seed)
    monkeypatch.setattr(pipeline_module, "extract_section_claims", _empty_section)

    with pytest.raises(RuntimeError, match="zero claims"):
        asyncio.run(decompose_paper(str(fake_pdf), "ignored.yaml"))


def test_pipeline_passes_deduplicated_negatives_to_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    config = _test_runtime_config(tmp_path)
    document = PaperDocument(
        metadata=PaperMetadata(title="Test Paper", authors=["A"]),
        sections=[
            Section(
                section_number=None,
                title="Abstract",
                role="abstract",
                body_text="Abstract text.",
                char_count=14,
            ),
            Section(
                section_number="1",
                title="Limitations",
                role="discussion",
                body_text="Discussion details.",
                char_count=20,
            ),
        ],
        all_artifacts=[],
    )

    seed_context = RawClaim(
        claim_id="c1",
        claim_type=ClaimType.context,
        statement="Memory fragmentation hurts serving efficiency.",
        source_section="Abstract",
    )
    negative_a = RawClaim(
        claim_id="n1",
        claim_type=ClaimType.negative,
        statement="Online compaction is too expensive.",
        source_section="1 Limitations",
    )
    negative_b = RawClaim(
        claim_id="n2",
        claim_type=ClaimType.negative,
        statement="Online compaction is too expensive.",
        source_section="1 Limitations",
    )

    monkeypatch.setattr(pipeline_module, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda *_: document)

    async def _ok_preflight(*args, **kwargs):
        _ = (args, kwargs)
        return None

    async def _seed(*args, **kwargs):
        _ = (args, kwargs)
        return SeedOutput(claims=[seed_context])

    async def _section(*args, **kwargs):
        _ = (args, kwargs)
        return SectionExtractionOutput(claims=[negative_a, negative_b])

    async def _dedup(*args, **kwargs):
        _ = (args, kwargs)
        return [seed_context, negative_a], [ClaimGroup(canonical_id="n1", member_ids=["n2"], parent_id=None)]

    captured: dict[str, object] = {}

    async def _assemble_tree(
        *,
        metadata,
        claims,
        faceted,
        negatives,
        artifacts,
        config,
        claim_groups,
    ):
        _ = (faceted, artifacts, config)
        captured["claims"] = list(claims)
        captured["negatives"] = list(negatives)
        captured["claim_groups"] = list(claim_groups or [])
        return PaperDecomposition(
            metadata=metadata,
            one_liner=OneLiner(achieved="ok", via="ok", because="ok"),
            claim_tree=[],
            negative_claims=list(negatives),
            all_artifacts=[],
            extraction_cost_usd=0.0,
        )

    monkeypatch.setattr(pipeline_module, "preflight_model_tiers", _ok_preflight)
    monkeypatch.setattr(pipeline_module, "extract_seed", _seed)
    monkeypatch.setattr(pipeline_module, "extract_section_claims", _section)
    monkeypatch.setattr(pipeline_module, "chunked_dedup", _dedup)
    monkeypatch.setattr(pipeline_module, "assemble_tree", _assemble_tree)

    _ = asyncio.run(decompose_paper(str(fake_pdf), "ignored.yaml"))

    negatives = captured["negatives"]
    assert isinstance(negatives, list)
    assert [claim.claim_id for claim in negatives] == ["n1"]

    groups = captured["claim_groups"]
    assert isinstance(groups, list)
    assert len(groups) == 1
    assert groups[0].canonical_id == "n1"


def test_normalize_claim_semantics_retags_context_claim_with_result_signal() -> None:
    claims = [
        RawClaim(
            claim_id="c_raw",
            claim_type=ClaimType.context,
            statement="vLLM improves throughput by 2-4x while keeping latency comparable to baselines.",
            source_section="Abstract",
        )
    ]

    normalized = pipeline_module._normalize_claim_semantics(claims)

    assert len(normalized) == 1
    assert normalized[0].claim_type == ClaimType.result
    assert normalized[0].structural_hints is not None
    assert normalized[0].structural_hints.local_role == ClaimLocalRole.empirical_finding
    assert normalized[0].structural_hints.preferred_parent_type == ParentPreference.method


def test_collapse_claim_duplicates_prefers_claim_with_stronger_provenance() -> None:
    weak = RawClaim(
        claim_id="m1",
        claim_type=ClaimType.method,
        statement="PagedAttention maps logical KV blocks to physical blocks.",
        source_section="abstract",
        evidence=[],
        entity_names=["PagedAttention"],
    )
    strong = RawClaim(
        claim_id="m2",
        claim_type=ClaimType.method,
        statement="PagedAttention maps logical KV blocks to physical blocks.",
        source_section="4.1 PagedAttention",
        evidence=[
            {"artifact_id": "fig_1", "role": "supports"},
            {"artifact_id": "table_2", "role": "supports"},
        ],
        entity_names=["PagedAttention", "KV cache"],
    )

    collapsed = pipeline_module._collapse_claim_duplicates([weak, strong])

    assert len(collapsed) == 1
    assert collapsed[0].claim_id == "m2"
