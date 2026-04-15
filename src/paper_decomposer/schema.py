from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

ModelTier = Literal["small", "medium", "heavy"]

_NONE_LIKE_TEXT = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "unspecified",
    "unknown",
    "not specified",
    "not stated",
}


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, (int, float, bool)):
        return _clean_text(str(value))
    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "statement",
            "reason",
            "why",
            "because",
            "artifact_id",
            "id",
            "label",
            "name",
            "value",
        )
        for key in preferred_keys:
            nested = value.get(key)
            if nested is None:
                continue
            nested_text = _coerce_string(nested)
            if nested_text:
                return nested_text
        nested_parts = [_coerce_string(item) for item in value.values()]
        return _clean_text(" ".join(part for part in nested_parts if part))
    if isinstance(value, (list, tuple, set)):
        nested_parts = [_coerce_string(item) for item in value]
        return _clean_text(" ".join(part for part in nested_parts if part))
    return _clean_text(str(value))


def _coerce_optional_string(value: Any) -> str | None:
    cleaned = _coerce_string(value)
    if cleaned.lower() in _NONE_LIKE_TEXT:
        return None
    return cleaned or None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part for part in re.split(r"[,\n;]+", value) if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        parts = [_coerce_string(item) for item in value]
    elif isinstance(value, dict):
        parts = [_coerce_string(value)]
    else:
        parts = [_coerce_string(value)]

    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_text(part)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in _NONE_LIKE_TEXT:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


class ApiConfig(BaseModel):
    provider: str
    base_url: str
    env_key_var: str = "TOGETHER_API_KEY"
    max_retries: int = 3
    retry_backoff_base: float = 2.0


class ModelTierConfig(BaseModel):
    model: str
    context_length: int | None = None
    supports_structured_output: bool | None = None
    input_cost_per_m: float | None = None
    output_cost_per_m: float | None = None
    temperature: float
    max_tokens: int
    notes: str | None = None


class ModelsConfig(BaseModel):
    small: ModelTierConfig
    medium: ModelTierConfig
    heavy: ModelTierConfig


class PdfPipelineConfig(BaseModel):
    parser: str = "pymupdf"
    extract_captions: bool = False
    extract_equations: bool = False
    min_section_chars: int
    max_section_chars: int


class PipelineConfig(BaseModel):
    pdf: PdfPipelineConfig
    seed: dict[str, Any] = Field(default_factory=dict)
    section_extraction: dict[str, Any] = Field(default_factory=dict)
    dedup: dict[str, Any] = Field(default_factory=dict)
    tree: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class PaperDecomposerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    api: ApiConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    facet_routing: dict[str, list[str]] = Field(default_factory=dict)


class RuntimeModelConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int


class RuntimePipelineConfig(BaseModel):
    parser: str
    extract_captions: bool
    extract_equations: bool
    min_section_chars: int
    max_section_chars: int
    seed: dict[str, Any] = Field(default_factory=dict)
    section_extraction: dict[str, Any] = Field(default_factory=dict)
    dedup: dict[str, Any] = Field(default_factory=dict)
    tree: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class AppSettings(BaseModel):
    config_path: str
    api_key: str
    model_tiers: dict[ModelTier, RuntimeModelConfig]
    pipeline: RuntimePipelineConfig
    raw: PaperDecomposerConfig

    def tier(self, tier: ModelTier) -> RuntimeModelConfig:
        return self.model_tiers[tier]


class ClaimType(str, Enum):
    context = "context"
    method = "method"
    result = "result"
    assumption = "assumption"
    negative = "negative"


class ClaimLocalRole(str, Enum):
    top_level = "top_level"
    mechanism = "mechanism"
    implementation_detail = "implementation_detail"
    empirical_finding = "empirical_finding"
    assumption = "assumption"
    limitation = "limitation"
    context_gap = "context_gap"
    positioning = "positioning"
    other = "other"


class ParentPreference(str, Enum):
    context = "context"
    method = "method"
    none = "none"


class RhetoricalRole(str, Enum):
    abstract = "abstract"
    introduction = "introduction"
    background = "background"
    method = "method"
    theory = "theory"
    evaluation = "evaluation"
    discussion = "discussion"
    appendix = "appendix"
    other = "other"


class InterventionType(str, Enum):
    architecture = "architecture"
    objective = "objective"
    algorithm = "algorithm"
    representation = "representation"
    data = "data"
    systems = "systems"
    theory = "theory"
    evaluation = "evaluation"
    pipeline = "pipeline"


class ScopeOfChange(str, Enum):
    drop_in = "drop_in"
    module = "module"
    system = "system"
    paradigm = "paradigm"


class GroundingType(str, Enum):
    formal_only = "formal_only"
    formal_and_empirical = "formal_and_empirical"
    empirical_controlled = "empirical_controlled"
    empirical_demo = "empirical_demo"
    qualitative = "qualitative"


class StackLayer(str, Enum):
    hardware = "hardware"
    os_kernel = "os_kernel"
    runtime = "runtime"
    application_level = "application_level"


class EvidenceArtifact(BaseModel):
    artifact_type: str
    artifact_id: str
    caption: str
    source_page: int


class Section(BaseModel):
    section_number: str | None = None
    title: str
    role: RhetoricalRole
    body_text: str
    artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    char_count: int


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None


class PaperDocument(BaseModel):
    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    all_artifacts: list[EvidenceArtifact] = Field(default_factory=list)


class EvidencePointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    role: str


class ClaimStructuralHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elaborates_seed_id: str | None = None
    local_role: ClaimLocalRole | None = None
    preferred_parent_type: ParentPreference | None = None


class RawClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: ClaimType
    statement: str
    source_section: str
    evidence: list[EvidencePointer] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    rejected_what: str | None = None
    rejected_why: str | None = None
    # Internal pipeline hinting; excluded from serialized final output.
    structural_hints: ClaimStructuralHints | None = Field(default=None, exclude=True)


# Flat API response schemas (no nesting, no $refs).
class FlatClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(validation_alias=AliasChoices("claim_id", "id"))
    claim_type: str = Field(validation_alias=AliasChoices("claim_type", "type"))
    statement: str = Field(validation_alias=AliasChoices("statement", "claim", "claim_text", "text"))
    source_section: str = Field(
        default="",
        validation_alias=AliasChoices("source_section", "section", "source", "section_title", "section_name"),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "evidence_ids",
            "evidence_artifact_ids",
            "evidence_id_list",
            "evidence",
            "artifacts",
        ),
    )
    entity_names: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("entity_names", "entities", "entity_list", "components"),
    )
    rejected_what: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "rejected_what",
            "rejection_what",
            "rejected_item",
            "rejected_alternative",
        ),
    )
    rejected_why: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "rejected_why",
            "rejection_why",
            "rejected_reason",
            "rejection_reason",
            "why_rejected",
        ),
    )
    elaborates_seed_id: str = Field(
        default="",
        validation_alias=AliasChoices("elaborates_seed_id", "seed_claim_id", "seed_id", "elaborates"),
    )
    local_role: str = Field(
        default="",
        validation_alias=AliasChoices("local_role", "claim_local_role", "role"),
    )
    preferred_parent_type: str = Field(
        default="",
        validation_alias=AliasChoices(
            "preferred_parent_type",
            "parent_preference",
            "preferred_parent",
            "parent_type",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_flat_claim(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        payload = dict(value)

        def first_present(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
            return default

        drift_aliases = {
            "sourceSection": "source_section",
            "sourceSectionTitle": "source_section",
            "evidenceArtifactIds": "evidence_ids",
            "entityNames": "entity_names",
            "rejectedWhat": "rejected_what",
            "rejectedWhy": "rejected_why",
            "elaboratesSeedId": "elaborates_seed_id",
            "localRole": "local_role",
            "preferredParentType": "preferred_parent_type",
        }
        for alias_key, canonical_key in drift_aliases.items():
            if alias_key in payload and canonical_key not in payload:
                payload[canonical_key] = payload[alias_key]

        return {
            "claim_id": first_present("claim_id", "id"),
            "claim_type": first_present("claim_type", "type"),
            "statement": first_present("statement", "claim", "claim_text", "text"),
            "source_section": _coerce_string(
                first_present(
                    "source_section",
                    "section",
                    "source",
                    "section_title",
                    "section_name",
                    default="",
                )
            ),
            "evidence_ids": _coerce_string_list(
                first_present(
                    "evidence_ids",
                    "evidence_artifact_ids",
                    "evidence_id_list",
                    "evidence",
                    "artifacts",
                )
            ),
            "entity_names": _coerce_string_list(
                first_present("entity_names", "entities", "entity_list", "components")
            ),
            "rejected_what": _coerce_optional_string(
                first_present(
                    "rejected_what",
                    "rejection_what",
                    "rejected_item",
                    "rejected_alternative",
                )
            ),
            "rejected_why": _coerce_optional_string(
                first_present(
                    "rejected_why",
                    "rejection_why",
                    "rejected_reason",
                    "rejection_reason",
                    "why_rejected",
                )
            ),
            "elaborates_seed_id": _coerce_string(
                first_present("elaborates_seed_id", "seed_claim_id", "seed_id", "elaborates", default="")
            ),
            "local_role": _coerce_string(first_present("local_role", "claim_local_role", "role", default="")),
            "preferred_parent_type": _coerce_string(
                first_present(
                    "preferred_parent_type",
                    "parent_preference",
                    "preferred_parent",
                    "parent_type",
                    default="",
                )
            ),
        }


class FlatSeedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[FlatClaim] = Field(default_factory=list)


class FlatSectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[FlatClaim] = Field(default_factory=list)


class SeedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[RawClaim] = Field(default_factory=list)


class SectionExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[RawClaim] = Field(default_factory=list)


class SectionExtractionClaimItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: ClaimType
    statement: str
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    rejected_what: str | None = None
    rejected_why: str | None = None


class SectionExtractionModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[SectionExtractionClaimItem] = Field(default_factory=list)


class UniversalFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_types: list[InterventionType] = Field(default_factory=list)
    scope: ScopeOfChange
    improves_or_replaces: str
    core_tradeoff: str
    grounding: GroundingType
    analogy_source: str | None = None


class FlatUniversalFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_types: list[str] = Field(default_factory=list)
    scope: str
    improves_or_replaces: str
    core_tradeoff: str
    grounding: str
    analogy_source: str = ""


class SystemsFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s1_resource: str
    s2_alloc_unit: str
    s3_stack_layer: str
    s4_mapping: str
    s5_policy: str
    s6_hw_assumption: str

    @field_validator("s3_stack_layer", mode="before")
    @classmethod
    def _normalize_stack_layer(cls, value: Any) -> Any:
        if isinstance(value, StackLayer):
            return value.value
        if not isinstance(value, str):
            return value

        lowered = value.strip().lower()
        if lowered in {"hardware", "hw", "gpu", "cpu", "accelerator", "device"}:
            return StackLayer.hardware.value
        if lowered in {"os_kernel", "kernel", "os", "operating system"}:
            return StackLayer.os_kernel.value
        if lowered in {
            "runtime",
            "inference runtime",
            "attention kernel interface",
            "runtime scheduler",
            "sampling runtime",
            "cuda attention kernel",
            "serving runtime",
        }:
            return StackLayer.runtime.value
        if lowered in {
            "application_level",
            "application",
            "application level",
            "llm serving system",
            "service layer",
        }:
            return StackLayer.application_level.value
        return lowered


class ArchitectureFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a1: str
    a2: str
    a3: str
    a4: str
    a5: str


class ObjectiveFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    o1: str
    o2: str
    o3: str
    o4: str


class AlgorithmFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    g1: str
    g2: str
    g3: str
    g4: str
    g5: str


class TheoryFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t1: str
    t2: str
    t3: str
    t4: str
    t5: str


class RepresentationFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p1: str
    p2: str
    p3: str
    p4: str


class EvaluationFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    e1: str
    e2: str
    e3: str


class PipelineFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l1: str
    l2: str
    l3: str


class InterventionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_type: InterventionType


class FlatInterventionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_type: str


class FacetedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: RawClaim
    universal_facets: UniversalFacets
    systems_facets: SystemsFacets | None = None
    architecture_facets: ArchitectureFacets | None = None
    objective_facets: ObjectiveFacets | None = None
    algorithm_facets: AlgorithmFacets | None = None
    theory_facets: TheoryFacets | None = None
    representation_facets: RepresentationFacets | None = None
    evaluation_facets: EvaluationFacets | None = None
    pipeline_facets: PipelineFacets | None = None


class ClaimGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_id: str = Field(
        validation_alias=AliasChoices("canonical_id", "canonical", "canonical_claim_id", "id"),
    )
    member_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("member_ids", "members", "duplicate_ids"),
    )
    parent_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("parent_id", "parent", "parent_claim_id"),
    )

    @field_validator("member_ids", mode="before")
    @classmethod
    def _normalize_member_ids(cls, value: Any) -> Any:
        return _coerce_string_list(value)

    @field_validator("parent_id", mode="before")
    @classmethod
    def _normalize_parent_id(cls, value: Any) -> Any:
        return _coerce_optional_string(value)


class DedupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[ClaimGroup] = Field(default_factory=list)


class ParentChildLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str = Field(validation_alias=AliasChoices("parent_id", "parent", "from_id"))
    child_id: str = Field(validation_alias=AliasChoices("child_id", "child", "to_id"))
    relationship: str = Field(validation_alias=AliasChoices("relationship", "type", "relation"))


class DedupBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[ClaimGroup] = Field(default_factory=list)


class CrossTypeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_child_links: list[ParentChildLink] = Field(default_factory=list)


class ClaimNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: ClaimType
    statement: str
    evidence: list[EvidencePointer] = Field(default_factory=list)
    facets: FacetedClaim | None = None
    children: list[ClaimNode] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    rejected_what: str | None = None
    rejected_why: str | None = None


class OneLiner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    achieved: str = Field(validation_alias=AliasChoices("achieved", "what", "result"))
    via: str = Field(validation_alias=AliasChoices("via", "how", "method"))
    because: str = Field(validation_alias=AliasChoices("because", "why", "motivation"))


class TreeNodeAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(validation_alias=AliasChoices("claim_id", "id", "node_id"))
    parent_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("parent_id", "parent", "parent_claim_id"),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("depends_on", "dependencies", "depends"),
    )

    @field_validator("depends_on", mode="before")
    @classmethod
    def _normalize_depends_on(cls, value: Any) -> Any:
        return _coerce_string_list(value)

    @field_validator("parent_id", mode="before")
    @classmethod
    def _normalize_parent_id(cls, value: Any) -> Any:
        return _coerce_optional_string(value)


class TreeAssemblyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    one_liner: OneLiner
    nodes: list[TreeNodeAssignment] = Field(default_factory=list)


class PaperDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: PaperMetadata
    one_liner: OneLiner
    claim_tree: list[ClaimNode] = Field(default_factory=list)
    negative_claims: list[RawClaim] = Field(default_factory=list)
    all_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    extraction_cost_usd: float = 0.0


__all__ = [
    "ApiConfig",
    "ModelTier",
    "ModelTierConfig",
    "ModelsConfig",
    "PdfPipelineConfig",
    "PipelineConfig",
    "PaperDecomposerConfig",
    "RuntimeModelConfig",
    "RuntimePipelineConfig",
    "AppSettings",
    "ClaimType",
    "ClaimLocalRole",
    "ParentPreference",
    "RhetoricalRole",
    "InterventionType",
    "ScopeOfChange",
    "GroundingType",
    "StackLayer",
    "EvidenceArtifact",
    "Section",
    "PaperMetadata",
    "PaperDocument",
    "EvidencePointer",
    "ClaimStructuralHints",
    "RawClaim",
    "FlatClaim",
    "FlatSeedOutput",
    "FlatSectionOutput",
    "SeedOutput",
    "SectionExtractionOutput",
    "SectionExtractionClaimItem",
    "SectionExtractionModelOutput",
    "UniversalFacets",
    "FlatUniversalFacets",
    "SystemsFacets",
    "ArchitectureFacets",
    "ObjectiveFacets",
    "AlgorithmFacets",
    "TheoryFacets",
    "RepresentationFacets",
    "EvaluationFacets",
    "PipelineFacets",
    "InterventionClassification",
    "FlatInterventionClassification",
    "FacetedClaim",
    "ClaimGroup",
    "DedupOutput",
    "ParentChildLink",
    "DedupBatchOutput",
    "CrossTypeOutput",
    "ClaimNode",
    "OneLiner",
    "TreeNodeAssignment",
    "TreeAssemblyOutput",
    "PaperDecomposition",
]
