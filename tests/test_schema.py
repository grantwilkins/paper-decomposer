import json
from collections.abc import Iterable

import pytest
from pydantic import BaseModel

from paper_decomposer.schema import (
    AlgorithmFacets,
    ArchitectureFacets,
    ClaimGroup,
    ClaimNode,
    ClaimType,
    DedupOutput,
    EvaluationFacets,
    EvidenceArtifact,
    EvidencePointer,
    FacetedClaim,
    GroundingType,
    InterventionClassification,
    InterventionType,
    ObjectiveFacets,
    OneLiner,
    PaperDecomposition,
    PaperDocument,
    PaperMetadata,
    PipelineFacets,
    RawClaim,
    RepresentationFacets,
    RhetoricalRole,
    ScopeOfChange,
    Section,
    SectionExtractionOutput,
    SeedOutput,
    SystemsFacets,
    TheoryFacets,
    UniversalFacets,
)


def _model_examples() -> list[tuple[type[BaseModel], BaseModel]]:
    artifact = EvidenceArtifact(
        artifact_type="figure",
        artifact_id="fig_1",
        caption="Overview of the method.",
        source_page=2,
    )
    section = Section(
        section_number="1",
        title="Introduction",
        role=RhetoricalRole.introduction,
        body_text="This section introduces the paper.",
        artifacts=[artifact],
        char_count=36,
    )
    metadata = PaperMetadata(
        title="PagedAttention for Efficient Serving",
        authors=["Alice Example", "Bob Example"],
        venue="SOSP",
        year=2023,
        doi="10.1234/example.doi",
    )
    document = PaperDocument(metadata=metadata, sections=[section], all_artifacts=[artifact])

    pointer = EvidencePointer(artifact_id="fig_1", role="supports")
    claim = RawClaim(
        claim_id="m1",
        claim_type=ClaimType.method,
        statement="PagedAttention maps logical KV blocks to physical memory blocks.",
        source_section="4.1 PagedAttention",
        evidence=[pointer],
        entity_names=["PagedAttention", "KV cache"],
        rejected_what=None,
        rejected_why=None,
    )
    negative_claim = RawClaim(
        claim_id="n1",
        claim_type=ClaimType.negative,
        statement="Compaction was rejected.",
        source_section="2.3 Design",
        evidence=[],
        entity_names=["compaction"],
        rejected_what="compaction",
        rejected_why="too expensive at serving scale",
    )
    seed_output = SeedOutput(claims=[claim])
    section_output = SectionExtractionOutput(claims=[claim, negative_claim])

    universal = UniversalFacets(
        intervention_types=[InterventionType.systems],
        scope=ScopeOfChange.system,
        improves_or_replaces="replaces contiguous KV cache allocation",
        core_tradeoff="small lookup overhead for much lower fragmentation",
        grounding=GroundingType.formal_and_empirical,
        analogy_source="virtual memory paging",
    )
    systems = SystemsFacets(
        s1_resource="gpu_memory",
        s2_alloc_unit="fixed-size KV block",
        s3_stack_layer="inference runtime",
        s4_mapping="logical block table to physical blocks",
        s5_policy="on-demand block allocation",
        s6_hw_assumption="high-bandwidth GPU memory access",
    )
    architecture = ArchitectureFacets(
        a1="decoder-only serving stack",
        a2="modular memory manager",
        a3="block table and scheduler",
        a4="shared KV cache pool",
        a5="request-level isolation",
    )
    objective = ObjectiveFacets(
        o1="maximize throughput",
        o2="reduce memory waste",
        o3="maintain generation quality",
        o4="preserve compatibility",
    )
    algorithm = AlgorithmFacets(
        g1="block allocation",
        g2="request admission",
        g3="block reclaim",
        g4="lookup path",
        g5="eviction behavior",
    )
    theory = TheoryFacets(
        t1="fragmentation bounds",
        t2="allocation invariants",
        t3="asymptotic overhead",
        t4="correctness argument",
        t5="failure modes",
    )
    representation = RepresentationFacets(
        p1="token-to-block index",
        p2="compact block metadata",
        p3="logical-physical indirection",
        p4="batched lookup tensors",
    )
    evaluation = EvaluationFacets(
        e1="throughput benchmarks",
        e2="memory usage curves",
        e3="ablation comparisons",
    )
    pipeline = PipelineFacets(
        l1="prefill stage",
        l2="decode stage",
        l3="resource arbitration",
    )
    classification = InterventionClassification(intervention_type=InterventionType.systems)
    faceted_claim = FacetedClaim(
        claim=claim,
        universal_facets=universal,
        systems_facets=systems,
    )

    group = ClaimGroup(canonical_id="m1", member_ids=["m1", "m1_dup"], parent_id=None)
    dedup = DedupOutput(groups=[group])

    child_node = ClaimNode(
        claim_id="r1",
        claim_type=ClaimType.result,
        statement="The method improves throughput by 2x.",
        evidence=[pointer],
        facets=None,
        children=[],
        depends_on=["m1"],
    )
    root_node = ClaimNode(
        claim_id="m1",
        claim_type=ClaimType.method,
        statement=claim.statement,
        evidence=[pointer],
        facets=faceted_claim,
        children=[child_node],
        depends_on=[],
    )
    one_liner = OneLiner(
        achieved="higher serving throughput",
        via="paged KV cache allocation",
        because="fragmentation is reduced while preserving access locality",
    )
    decomposition = PaperDecomposition(
        metadata=metadata,
        one_liner=one_liner,
        claim_tree=[root_node],
        negative_claims=[negative_claim],
        all_artifacts=[artifact],
        extraction_cost_usd=1.25,
    )

    return [
        (EvidenceArtifact, artifact),
        (Section, section),
        (PaperMetadata, metadata),
        (PaperDocument, document),
        (EvidencePointer, pointer),
        (RawClaim, claim),
        (SeedOutput, seed_output),
        (SectionExtractionOutput, section_output),
        (UniversalFacets, universal),
        (SystemsFacets, systems),
        (ArchitectureFacets, architecture),
        (ObjectiveFacets, objective),
        (AlgorithmFacets, algorithm),
        (TheoryFacets, theory),
        (RepresentationFacets, representation),
        (EvaluationFacets, evaluation),
        (PipelineFacets, pipeline),
        (InterventionClassification, classification),
        (FacetedClaim, faceted_claim),
        (ClaimGroup, group),
        (DedupOutput, dedup),
        (ClaimNode, root_node),
        (OneLiner, one_liner),
        (PaperDecomposition, decomposition),
    ]


MODEL_EXAMPLES = _model_examples()


@pytest.mark.parametrize(("model_cls", "example"), MODEL_EXAMPLES)
def test_models_instantiate(model_cls: type[BaseModel], example: BaseModel) -> None:
    assert isinstance(example, model_cls)


@pytest.mark.parametrize(("model_cls", "example"), MODEL_EXAMPLES)
def test_model_round_trip(model_cls: type[BaseModel], example: BaseModel) -> None:
    rebuilt = model_cls(**example.model_dump())
    assert rebuilt == example


STRUCTURED_OUTPUT_MODELS: Iterable[type[BaseModel]] = [
    SeedOutput,
    SectionExtractionOutput,
    InterventionClassification,
    UniversalFacets,
    SystemsFacets,
    ArchitectureFacets,
    ObjectiveFacets,
    AlgorithmFacets,
    TheoryFacets,
    RepresentationFacets,
    EvaluationFacets,
    PipelineFacets,
    DedupOutput,
    PaperDecomposition,
]


@pytest.mark.parametrize("model_cls", STRUCTURED_OUTPUT_MODELS)
def test_structured_output_models_json_schema(model_cls: type[BaseModel]) -> None:
    schema = model_cls.model_json_schema()
    serialized = json.loads(json.dumps(schema))

    assert isinstance(serialized, dict)
    assert serialized.get("type") == "object"
    assert "properties" in serialized
