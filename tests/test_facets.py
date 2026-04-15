"""
Claim:
`extract_facets` classifies a METHOD claim, routes to the correct
domain facet schema, and returns a coherent FacetedClaim where
universal facets and domain facets agree.

Plausible wrong implementations:
- Uses wrong routing map (e.g., systems claim filled with architecture facets).
- Classifies from irrelevant text (title/noise) instead of claim + section context.
- Populates multiple domain facet blocks instead of exactly one.
- Fills unsupported facet answers with guesses instead of "UNSPECIFIED".
- Returns universal scope inconsistent with the classified intervention type.
- Ignores facet-specific model tier config and silently uses the wrong tier.
- Accepts non-METHOD claims and produces semantically invalid facet objects.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

import paper_decomposer.prompts.facets as facet_prompts
from paper_decomposer.config import load_config
from paper_decomposer.models import call_model
from paper_decomposer.prompts.facets import (
    build_classify_prompt,
    build_systems_facet_prompt,
    build_universal_prompt,
    extract_facets,
    get_facet_prompt,
    get_facet_schema,
)
from paper_decomposer.schema import (
    AlgorithmFacets,
    ArchitectureFacets,
    ClaimType,
    EvaluationFacets,
    FlatInterventionClassification,
    FlatUniversalFacets,
    InterventionClassification,
    InterventionType,
    ObjectiveFacets,
    PipelineFacets,
    RawClaim,
    RepresentationFacets,
    RhetoricalRole,
    ScopeOfChange,
    Section,
    SystemsFacets,
    TheoryFacets,
    UniversalFacets,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

requires_api_key = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY is not set.",
)

PAGED_ATTENTION_CLAIM = (
    "PagedAttention partitions each sequence KV cache into fixed-size blocks and maps "
    "logical blocks to non-contiguous physical blocks."
)

PAGED_ATTENTION_CLAIM_PARAPHRASE = (
    "The method stores KV cache in fixed blocks and uses indirection from logical "
    "block indices to physical memory blocks."
)

PAGED_ATTENTION_SECTION_SNIPPET = """
To address the memory challenges in §3, we introduce PagedAttention, an attention
algorithm inspired by paging in operating systems. Unlike traditional attention,
PagedAttention allows storing continuous keys and values in non-contiguous memory
space. It partitions each sequence KV cache into fixed-size KV blocks. During
attention computation, the kernel identifies and fetches different KV blocks
separately. The mechanism keeps a mapping from logical KV blocks to physical
KV blocks through a block table and enables dynamic block allocation.
""".strip()

PAGED_ATTENTION_TRIMMED_SNIPPET = """
PagedAttention partitions each sequence KV cache into fixed-size blocks and tracks
which logical blocks correspond to which physical blocks through a block table.
New logical blocks are mapped as decoding proceeds. The text here omits hardware
or platform assumptions.
""".strip()


def _method_claim(statement: str, claim_id: str = "m_pagedattention") -> RawClaim:
    return RawClaim(
        claim_id=claim_id,
        claim_type=ClaimType.method,
        statement=statement,
        source_section="4.1 PagedAttention",
    )


def _method_section(text: str) -> Section:
    return Section(
        section_number="4.1",
        title="PagedAttention",
        role=RhetoricalRole.method,
        body_text=text,
        char_count=len(text),
    )


@pytest.mark.api
@requires_api_key
def test_extract_facets_api_integration_primary_fixture() -> None:
    settings = load_config(CONFIG_PATH)
    claim = _method_claim(PAGED_ATTENTION_CLAIM)
    section = _method_section(PAGED_ATTENTION_SECTION_SNIPPET)

    classified = asyncio.run(
        call_model(
            "small",
            build_classify_prompt(claim, section),
            response_schema=InterventionClassification,
            config=settings,
        )
    )
    assert isinstance(classified, InterventionClassification)
    assert classified.intervention_type == InterventionType.systems

    faceted = asyncio.run(extract_facets(claim, section, settings))
    assert faceted.systems_facets is not None
    assert faceted.architecture_facets is None
    assert faceted.objective_facets is None
    assert faceted.algorithm_facets is None
    assert faceted.theory_facets is None
    assert faceted.representation_facets is None
    assert faceted.evaluation_facets is None
    assert faceted.pipeline_facets is None

    s1_resource = faceted.systems_facets.s1_resource.lower()
    assert ("memory" in s1_resource) or ("gpu" in s1_resource)

    s4_mapping = faceted.systems_facets.s4_mapping.lower()
    assert "block" in s4_mapping
    assert ("logical" in s4_mapping) or ("physical" in s4_mapping)

    assert faceted.universal_facets.scope == ScopeOfChange.system
    print("\nPrimary faceted claim:\n", faceted.model_dump_json(indent=2))


def test_facet_routing_schema_boundary_without_api() -> None:
    expected_schema_map = {
        InterventionType.architecture: ArchitectureFacets,
        InterventionType.objective: ObjectiveFacets,
        InterventionType.algorithm: AlgorithmFacets,
        InterventionType.representation: RepresentationFacets,
        InterventionType.data: EvaluationFacets,
        InterventionType.systems: SystemsFacets,
        InterventionType.theory: TheoryFacets,
        InterventionType.evaluation: EvaluationFacets,
        InterventionType.pipeline: PipelineFacets,
    }

    for intervention_type, schema_cls in expected_schema_map.items():
        assert callable(get_facet_prompt(intervention_type))
        assert get_facet_schema(intervention_type) is schema_cls

    # Boundary: string routing should be case/space tolerant.
    assert callable(get_facet_prompt(" SYSTEMS "))
    assert get_facet_schema("ThEoRy") is TheoryFacets

    with pytest.raises((KeyError, ValueError)):
        get_facet_prompt("unsupported_type")
    with pytest.raises((KeyError, ValueError)):
        get_facet_schema("unsupported_type")


def test_classify_prompt_contract_includes_all_types_and_context() -> None:
    claim = _method_claim("CLAIM_SENTINEL: paging-based KV indirection")
    section = _method_section("SECTION_SENTINEL: logical blocks map to physical blocks.")
    messages = build_classify_prompt(claim, section)

    all_text = "\n".join(message["content"] for message in messages)
    assert "CLAIM_SENTINEL" in all_text
    assert "SECTION_SENTINEL" in all_text
    for intervention_type in InterventionType:
        assert intervention_type.value in all_text


def test_domain_and_universal_prompts_require_unspecified_boundary() -> None:
    claim = _method_claim("CLAIM_SENTINEL: paging-based KV indirection")
    section = _method_section("SECTION_SENTINEL: logical blocks map to physical blocks.")

    systems_messages = build_systems_facet_prompt(claim, section)
    universal_messages = build_universal_prompt(claim, section)

    systems_text = "\n".join(message["content"] for message in systems_messages)
    universal_text = "\n".join(message["content"] for message in universal_messages)

    assert "CLAIM_SENTINEL" in systems_text
    assert "SECTION_SENTINEL" in systems_text
    assert "UNSPECIFIED" in systems_text
    assert "Never invent hardware assumptions" in systems_text
    assert "CLAIM_SENTINEL" in universal_text
    assert "SECTION_SENTINEL" in universal_text
    assert "UNSPECIFIED" in universal_text
    assert "Do not invent analogies" in universal_text


def test_extract_facets_routes_schema_uses_facets_tier_and_enforces_single_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _method_claim(PAGED_ATTENTION_CLAIM, "m_stubbed")
    section = _method_section(PAGED_ATTENTION_SECTION_SNIPPET)
    call_log: list[tuple[str, Any]] = []

    async def _fake_call_typed_model(
        *,
        tier: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        config: Any | None = None,
    ) -> Any:
        call_log.append((tier, schema))
        assert messages

        if schema is FlatInterventionClassification:
            return FlatInterventionClassification(intervention_type="systems")
        if schema is SystemsFacets:
            return SystemsFacets(
                s1_resource="gpu memory",
                s2_alloc_unit="fixed-size KV block",
                s3_stack_layer="runtime",
                s4_mapping="logical block -> physical block via block table",
                s5_policy="on-demand allocation",
                s6_hw_assumption="UNSPECIFIED",
            )
        if schema is FlatUniversalFacets:
            # Deliberately omit systems to ensure extract_facets repairs coherence.
            return FlatUniversalFacets(
                intervention_types=["algorithm"],
                scope=ScopeOfChange.system.value,
                improves_or_replaces="contiguous KV allocation",
                core_tradeoff="indirection overhead for memory efficiency",
                grounding="empirical_demo",
                analogy_source="virtual memory paging",
            )
        raise AssertionError(f"Unexpected schema request: {schema}")

    monkeypatch.setattr(facet_prompts, "_call_typed_model", _fake_call_typed_model)

    result = asyncio.run(
        extract_facets(
            claim,
            section,
            config={"pipeline": {"facets": {"model_tier": "medium"}}},
        )
    )

    assert call_log == [
        ("medium", FlatInterventionClassification),
        ("medium", SystemsFacets),
        ("medium", FlatUniversalFacets),
    ]

    assert result.systems_facets is not None
    assert result.architecture_facets is None
    assert result.objective_facets is None
    assert result.algorithm_facets is None
    assert result.theory_facets is None
    assert result.representation_facets is None
    assert result.evaluation_facets is None
    assert result.pipeline_facets is None
    assert result.universal_facets.intervention_types[0] == InterventionType.systems


def test_extract_facets_data_classification_routes_to_evaluation_facets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _method_claim("We curate a new benchmark dataset with strict filtering.", "m_data")
    section = _method_section("Dataset construction and filtering pipeline details.")

    async def _fake_call_typed_model(
        *,
        tier: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        config: Any | None = None,
    ) -> Any:
        if schema is FlatInterventionClassification:
            return FlatInterventionClassification(intervention_type="data")
        if schema is EvaluationFacets:
            return EvaluationFacets(
                e1="dataset coverage quality",
                e2="web and curated corpora",
                e3="human validation protocol",
            )
        if schema is FlatUniversalFacets:
            return FlatUniversalFacets(
                intervention_types=["data"],
                scope=ScopeOfChange.module.value,
                improves_or_replaces="unfiltered corpus collection",
                core_tradeoff="quality gains for added curation cost",
                grounding="empirical_controlled",
                analogy_source="UNSPECIFIED",
            )
        raise AssertionError(f"Unexpected schema request: {schema}")

    monkeypatch.setattr(facet_prompts, "_call_typed_model", _fake_call_typed_model)
    result = asyncio.run(extract_facets(claim, section, config={}))

    assert result.evaluation_facets is not None
    assert result.systems_facets is None
    assert result.architecture_facets is None
    assert result.objective_facets is None
    assert result.algorithm_facets is None
    assert result.theory_facets is None
    assert result.representation_facets is None
    assert result.pipeline_facets is None
    assert InterventionType.data in result.universal_facets.intervention_types


def test_extract_facets_sanitizes_unsupported_facet_content_to_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _method_claim(PAGED_ATTENTION_CLAIM_PARAPHRASE, "m_sanitize")
    section = _method_section(PAGED_ATTENTION_TRIMMED_SNIPPET)

    async def _fake_call_typed_model(
        *,
        tier: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, config)
        if schema is FlatInterventionClassification:
            return FlatInterventionClassification(intervention_type="systems")
        if schema is SystemsFacets:
            return SystemsFacets(
                s1_resource="traffic control lanes",
                s2_alloc_unit="delivery convoy",
                s3_stack_layer="firmware",
                s4_mapping="cars -> roads via dispatcher",
                s5_policy="like a taxi dispatcher",
                s6_hw_assumption="NVIDIA A100 GPUs with NVLink",
            )
        if schema is FlatUniversalFacets:
            return FlatUniversalFacets(
                intervention_types=["systems"],
                scope="system",
                improves_or_replaces="urban traffic routing",
                core_tradeoff="like dispatching taxis across a city",
                grounding="empirical_demo",
                analogy_source="Like a traffic controller coordinating cars",
            )
        raise AssertionError(f"Unexpected schema request: {schema}")

    monkeypatch.setattr(facet_prompts, "_call_typed_model", _fake_call_typed_model)
    result = asyncio.run(extract_facets(claim, section, config={}))

    assert result.systems_facets is not None
    assert result.systems_facets.s1_resource == "UNSPECIFIED"
    assert result.systems_facets.s2_alloc_unit == "UNSPECIFIED"
    assert result.systems_facets.s3_stack_layer == "UNSPECIFIED"
    assert result.systems_facets.s4_mapping == "UNSPECIFIED"
    assert result.systems_facets.s5_policy == "UNSPECIFIED"
    assert result.systems_facets.s6_hw_assumption == "UNSPECIFIED"
    assert result.universal_facets.improves_or_replaces == "UNSPECIFIED"
    assert result.universal_facets.core_tradeoff == "UNSPECIFIED"
    assert result.universal_facets.analogy_source is None


def test_extract_facets_preserves_grounded_analogy_and_hardware_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _method_claim(PAGED_ATTENTION_CLAIM, "m_grounded")
    section = _method_section(PAGED_ATTENTION_SECTION_SNIPPET)

    async def _fake_call_typed_model(
        *,
        tier: str,
        messages: list[dict[str, str]],
        schema: type[Any],
        config: Any | None = None,
    ) -> Any:
        _ = (tier, messages, config)
        if schema is FlatInterventionClassification:
            return FlatInterventionClassification(intervention_type="systems")
        if schema is SystemsFacets:
            return SystemsFacets(
                s1_resource="KV cache memory",
                s2_alloc_unit="fixed-size KV blocks",
                s3_stack_layer="runtime",
                s4_mapping="logical blocks -> physical blocks via block table",
                s5_policy="dynamic block allocation",
                s6_hw_assumption="GPU memory for KV blocks",
            )
        if schema is FlatUniversalFacets:
            return FlatUniversalFacets(
                intervention_types=["systems"],
                scope="system",
                improves_or_replaces="contiguous KV cache allocation",
                core_tradeoff="block table indirection for flexible allocation",
                grounding="empirical_demo",
                analogy_source="paging in operating systems",
            )
        raise AssertionError(f"Unexpected schema request: {schema}")

    monkeypatch.setattr(facet_prompts, "_call_typed_model", _fake_call_typed_model)
    result = asyncio.run(extract_facets(claim, section, config={}))

    assert result.systems_facets is not None
    assert result.systems_facets.s1_resource != "UNSPECIFIED"
    assert result.systems_facets.s4_mapping != "UNSPECIFIED"
    assert result.systems_facets.s6_hw_assumption != "UNSPECIFIED"
    assert result.universal_facets.improves_or_replaces != "UNSPECIFIED"
    assert result.universal_facets.core_tradeoff != "UNSPECIFIED"
    assert result.universal_facets.analogy_source == "paging in operating systems"


def test_extract_facets_rejects_non_method_claim() -> None:
    claim = RawClaim(
        claim_id="r1",
        claim_type=ClaimType.result,
        statement="Throughput improves by 2x.",
        source_section="6.2 Basic Sampling",
    )
    section = _method_section(PAGED_ATTENTION_SECTION_SNIPPET)

    with pytest.raises(ValueError, match="METHOD"):
        asyncio.run(extract_facets(claim, section, config={}))


@pytest.mark.api
@requires_api_key
def test_extract_facets_metamorphic_paraphrase_classification() -> None:
    settings = load_config(CONFIG_PATH)
    section = _method_section(PAGED_ATTENTION_SECTION_SNIPPET)

    claim_1 = _method_claim(PAGED_ATTENTION_CLAIM, "m1")
    claim_2 = _method_claim(PAGED_ATTENTION_CLAIM_PARAPHRASE, "m2")

    classified_1 = asyncio.run(
        call_model(
            "small",
            build_classify_prompt(claim_1, section),
            response_schema=InterventionClassification,
            config=settings,
        )
    )
    classified_2 = asyncio.run(
        call_model(
            "small",
            build_classify_prompt(claim_2, section),
            response_schema=InterventionClassification,
            config=settings,
        )
    )
    assert isinstance(classified_1, InterventionClassification)
    assert isinstance(classified_2, InterventionClassification)
    assert classified_1.intervention_type == InterventionType.systems
    assert classified_2.intervention_type == InterventionType.systems

    first = asyncio.run(extract_facets(claim_1, section, settings))
    second = asyncio.run(extract_facets(claim_2, section, settings))

    assert first.systems_facets is not None
    assert second.systems_facets is not None
    assert InterventionType.systems in first.universal_facets.intervention_types
    assert InterventionType.systems in second.universal_facets.intervention_types

    print("\nMetamorphic faceted claim #1:\n", first.model_dump_json(indent=2))
    print("\nMetamorphic faceted claim #2:\n", second.model_dump_json(indent=2))


@pytest.mark.api
@requires_api_key
def test_extract_facets_unknown_information_boundary_hw_unspecified() -> None:
    settings = load_config(CONFIG_PATH)
    claim = _method_claim(PAGED_ATTENTION_CLAIM_PARAPHRASE, "m_unknown")
    section = _method_section(PAGED_ATTENTION_TRIMMED_SNIPPET)

    faceted = asyncio.run(extract_facets(claim, section, settings))
    assert faceted.systems_facets is not None
    hw_assumption = faceted.systems_facets.s6_hw_assumption.strip()
    assert hw_assumption == "UNSPECIFIED" or "UNSPECIFIED" in hw_assumption.upper()

    print("\nUnknown-information faceted claim:\n", faceted.model_dump_json(indent=2))
