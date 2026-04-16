"""
Claim:
The rewritten pipeline should use Stage 2 compression as the sole semantic
selection step, route residue into support details, and emit a stable final
structure without semantic fallback stages.

Plausible wrong implementations:
- Keep more than two promoted methods because late stages still reinterpret nodes.
- Drop residual claims instead of converting them to support details.
- Fail to anchor support details back to promoted claims.
- Preserve removed output fields from the old pipeline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import paper_decomposer.pipeline as pipeline_module
from paper_decomposer.schema import (
    AppSettings,
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimType,
    FacetedClaim,
    GroundingType,
    InterventionType,
    OneLiner,
    PaperDecomposerConfig,
    PaperDocument,
    PaperMetadata,
    ParentPreference,
    RawClaim,
    RuntimeModelConfig,
    RuntimePipelineConfig,
    ScopeOfChange,
    Section,
    SectionArgumentCandidate,
    SectionDigestOutput,
    SeedOutput,
    SupportDetail,
    SupportDetailType,
    SupportRelationshipType,
    UniversalFacets,
)


@pytest.fixture()
def settings(tmp_path: Path) -> AppSettings:
    raw = PaperDecomposerConfig.model_validate(
        {
            "api": {"provider": "together", "base_url": "https://api.together.xyz/v1"},
            "models": {
                "small": {"model": "small", "temperature": 0.1, "max_tokens": 64},
                "medium": {"model": "medium", "temperature": 0.1, "max_tokens": 64},
                "heavy": {"model": "heavy", "temperature": 0.1, "max_tokens": 64},
            },
            "pipeline": {
                "pdf": {"min_section_chars": 1, "max_section_chars": 100},
                "seed": {},
                "section_extraction": {},
                "dedup": {},
                "tree": {},
                "output": {"output_dir": str(tmp_path)},
            },
        }
    )
    return AppSettings(
        config_path="config.yaml",
        api_key="test-key",
        model_tiers={
            "small": RuntimeModelConfig(model="small", temperature=0.1, max_tokens=64),
            "medium": RuntimeModelConfig(model="medium", temperature=0.1, max_tokens=64),
            "heavy": RuntimeModelConfig(model="heavy", temperature=0.1, max_tokens=64),
        },
        pipeline=RuntimePipelineConfig(
            parser="pymupdf",
            extract_captions=False,
            extract_equations=False,
            min_section_chars=1,
            max_section_chars=100,
            output={"output_dir": str(tmp_path)},
        ),
        raw=raw,
    )


def test_decompose_paper_uses_bounded_semantic_backbone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: AppSettings) -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="PagedAttention", authors=["A"]),
        sections=[
            Section(section_number="0", title="Abstract", role="abstract", body_text="Abstract text", char_count=20),
            Section(section_number="4", title="Method", role="method", body_text="Method text", char_count=20),
            Section(section_number="5", title="Eval", role="evaluation", body_text="Eval text", char_count=20),
        ],
        all_artifacts=[],
    )

    seed_output = SeedOutput(
        claims=[
            RawClaim(claim_id="seed_c", claim_type=ClaimType.context, statement="Fragmentation limits batch size.", source_section="Abstract"),
            RawClaim(claim_id="seed_m", claim_type=ClaimType.method, statement="PagedAttention is an attention algorithm that pages KV blocks.", source_section="Abstract"),
        ]
    )

    section_digest = SectionDigestOutput(
        argument_candidates=[
            SectionArgumentCandidate(
                claim_id="cand_sys",
                claim_type=ClaimType.method,
                statement="vLLM is a serving runtime built around PagedAttention.",
                source_section="4 Method",
                local_role=ClaimLocalRole.top_level,
                preferred_parent_type=ParentPreference.method,
            ),
            SectionArgumentCandidate(
                claim_id="cand_kernel",
                claim_type=ClaimType.method,
                statement="A fused CUDA kernel gathers discontinuous blocks with coalesced reads.",
                source_section="4 Method",
                local_role=ClaimLocalRole.implementation_detail,
                preferred_parent_type=ParentPreference.method,
            ),
            SectionArgumentCandidate(
                claim_id="cand_res",
                claim_type=ClaimType.result,
                statement="vLLM reaches 2-4x higher throughput than Orca.",
                source_section="5 Eval",
            ),
            SectionArgumentCandidate(
                claim_id="cand_bad_assumption",
                claim_type=ClaimType.assumption,
                statement="The runtime allocates a new block only when prior blocks are full.",
                source_section="4 Method",
            ),
            SectionArgumentCandidate(
                claim_id="cand_good_assumption",
                claim_type=ClaimType.assumption,
                statement="Benefits depend on concurrent requests and GPU memory capacity.",
                source_section="6 Limitations",
            ),
        ],
        support_details=[
            SupportDetail(
                support_detail_id="SD_existing",
                detail_type=SupportDetailType.implementation_fact,
                text="Logical blocks map to non-contiguous physical blocks.",
                source_section="4 Method",
                anchor_claim_id=None,
                candidate_anchor_ids=[],
                relationship_type=SupportRelationshipType.implements,
                confidence=0.4,
                evidence_ids=[],
            )
        ],
    )

    async def _noop_preflight(*args, **kwargs):
        _ = (args, kwargs)
        return None

    async def _fake_seed(*args, **kwargs):
        _ = (args, kwargs)
        return seed_output

    async def _fake_section(*args, **kwargs):
        _ = (args, kwargs)
        return section_digest

    async def _fake_facets(claim: RawClaim, source, config):
        _ = (source, config)
        return FacetedClaim(
            claim=claim,
            universal_facets=UniversalFacets(
                intervention_types=[InterventionType.algorithm],
                scope=ScopeOfChange.module,
                improves_or_replaces="baseline attention",
                core_tradeoff="UNSPECIFIED",
                grounding=GroundingType.qualitative,
                analogy_source=None,
            ),
        )

    monkeypatch.setattr(pipeline_module, "load_config", lambda path: settings)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda path, config: document)
    monkeypatch.setattr(pipeline_module, "preflight_model_tiers", _noop_preflight)
    monkeypatch.setattr(pipeline_module, "extract_seed", _fake_seed)
    monkeypatch.setattr(pipeline_module, "extract_section_digest", _fake_section)
    monkeypatch.setattr(pipeline_module, "extract_facets", _fake_facets)

    result = asyncio.run(pipeline_module.decompose_paper(str(tmp_path / "paper.pdf"), "ignored.yaml"))
    all_nodes = []
    stack = list(result.claim_tree)
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        stack.extend(node.children)

    method_nodes = [node for node in all_nodes if node.claim_type == ClaimType.method]

    assert len(result.claim_tree) == 1
    assert len(method_nodes) <= 2
    assert any(detail.anchor_claim_id for detail in result.support_details)
    assert not hasattr(result, "negative_claims")
    assert any(path.name.startswith("PagedAttention") for path in tmp_path.iterdir())
