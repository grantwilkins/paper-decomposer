from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MethodNodeKind = Literal["system", "method", "method_category"]
SettingKind = Literal["application", "task", "dataset", "workload", "hardware", "model_artifact", "metric"]
MethodRelationKind = Literal["uses", "is_a", "refines", "composes"]
SettingRelationKind = Literal["is_a", "composes", "specializes", "refines"]
MethodSettingRelationKind = Literal["applies_to", "evaluated_on", "uses_artifact"]
ClaimType = Literal[
    "performance",
    "memory",
    "scalability",
    "quality",
    "capability",
    "limitation",
    "comparison",
    "overhead",
    "other",
]
NodeStatus = Literal["claimed_new", "reference", "uncertain"]
SourceKind = Literal["abstract", "paragraph", "caption", "table_text", "contribution", "conclusion"]
EvidenceClass = Literal[
    "prose",
    "caption",
    "table",
    "component_label",
    "example_text",
    "frontmatter",
    "formula_fragment",
]
ResolutionRelationKind = Literal[
    "same_as",
    "variant_of",
    "uses",
    "extends",
    "subsumes",
    "is_subsumed_by",
    "reimplements",
    "compared_against",
    "distinct",
    "uncertain",
]


class ValidationSeverity(str, Enum):
    error = "error"
    warning = "warning"


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    paper_id: str
    section_title: str
    section_kind: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    artifact_id: str | None = None
    source_kind: SourceKind = "paragraph"
    evidence_class: EvidenceClass = "prose"

    @field_validator("span_id", "paper_id", "section_title", "section_kind", "text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned


class CandidateNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    candidate_kind: str
    rationale: str
    evidence_span_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class ProblemStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    statement: str
    description: str | None = None
    evidence_span_ids: list[str]
    confidence: float | None = None

    @field_validator("problem_id", "statement")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned


class MechanismSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    operative_move: str
    preconditions: list[str] = Field(default_factory=list)
    state_modified: list[str] = Field(default_factory=list)
    failure_modes_or_tradeoffs: list[str] = Field(default_factory=list)
    typical_settings: list[str] = Field(default_factory=list)
    supporting_concepts: list[str] = Field(default_factory=list)

    @field_validator("operative_move")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_node_id: str
    kind: MethodNodeKind
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    category_tags: list[str] = Field(default_factory=list)
    description: str
    status: NodeStatus = "uncertain"
    introduced_by: str | None = None
    problem_ids: list[str] = Field(default_factory=list)
    granularity_rationale: str
    evidence_span_ids: list[str]
    confidence: float | None = None
    mechanism_sentence: str | None = None
    mechanism_signature: MechanismSignature | None = None

    @field_validator("local_node_id", "canonical_name", "description", "granularity_rationale")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned


class ExtractedSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_setting_id: str
    kind: SettingKind
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str
    evidence_span_ids: list[str]
    confidence: float | None = None


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str
    child_id: str
    relation_kind: MethodRelationKind
    evidence_span_ids: list[str]
    confidence: float | None = None


class ExtractedSettingEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str
    child_id: str
    relation_kind: SettingRelationKind
    evidence_span_ids: list[str]
    confidence: float | None = None


class ExtractedMethodSettingLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method_id: str
    setting_id: str
    relation_kind: MethodSettingRelationKind
    evidence_span_ids: list[str]
    confidence: float | None = None


class ExtractedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_id: str
    paper_id: str
    metric: str
    method_ids: list[str] = Field(default_factory=list)
    setting_ids: list[str] = Field(default_factory=list)
    value: str | None = None
    delta: str | None = None
    baseline: str | None = None
    comparator: str | None = None
    units: str | None = None
    evidence_span_ids: list[str]
    confidence: float | None = None


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    paper_id: str
    claim_type: ClaimType
    raw_text: str
    finding: str
    method_ids: list[str] = Field(default_factory=list)
    setting_ids: list[str] = Field(default_factory=list)
    problem_ids: list[str] = Field(default_factory=list)
    outcome_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str]
    confidence: float | None = None


class DemotedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    reason_demoted: str
    stored_under: str
    evidence_span_ids: list[str]


class ExtractionCaps(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_system_nodes: int = 2
    max_method_nodes: int = 12
    max_setting_nodes: int = 30
    max_claims: int = 25
    max_outcomes: int = 40
    max_demoted_items: int = 40


class PaperGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    systems: list[ExtractedNode] = Field(default_factory=list)
    methods: list[ExtractedNode] = Field(default_factory=list)
    method_edges: list[ExtractedEdge] = Field(default_factory=list)
    settings: list[ExtractedSetting] = Field(default_factory=list)
    setting_edges: list[ExtractedSettingEdge] = Field(default_factory=list)
    method_setting_links: list[ExtractedMethodSettingLink] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_graph_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        lifted = dict(data)
        nodes = lifted.pop("nodes", None)
        edges = lifted.pop("edges", None)
        if nodes is not None and "systems" not in lifted and "methods" not in lifted:
            lifted["systems"] = [node for node in nodes if _node_kind(node) == "system"]
            lifted["methods"] = [node for node in nodes if _node_kind(node) != "system"]
        if edges is not None and "method_edges" not in lifted:
            lifted["method_edges"] = edges
        return lifted

    @model_validator(mode="after")
    def _validate_node_families(self) -> PaperGraph:
        wrong_systems = [node.local_node_id for node in self.systems if node.kind != "system"]
        wrong_methods = [node.local_node_id for node in self.methods if node.kind == "system"]
        if wrong_systems:
            raise ValueError(f"systems must contain only system nodes: {wrong_systems}")
        if wrong_methods:
            raise ValueError(f"methods must not contain system nodes: {wrong_methods}")
        return self

    @property
    def nodes(self) -> list[ExtractedNode]:
        return [*self.systems, *self.methods]


class PaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    extraction_run_id: str
    title: str
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    graph: PaperGraph = Field(default_factory=PaperGraph)
    problems: list[ProblemStatement] = Field(default_factory=list)
    outcomes: list[ExtractedOutcome] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    demoted_items: list[DemotedItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_final_graph_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        lifted = dict(data)
        graph = dict(lifted.get("graph") or {})
        legacy_keys = {
            "nodes": "nodes",
            "edges": "edges",
            "settings": "settings",
            "setting_edges": "setting_edges",
            "method_setting_links": "method_setting_links",
        }
        for old_key, graph_key in legacy_keys.items():
            if old_key in lifted and graph_key not in graph:
                graph[graph_key] = lifted.pop(old_key)
        lifted.pop("candidates", None)
        if graph:
            lifted["graph"] = graph
        return lifted

    @property
    def nodes(self) -> list[ExtractedNode]:
        return self.graph.nodes

    @property
    def edges(self) -> list[ExtractedEdge]:
        return self.graph.method_edges

    @property
    def settings(self) -> list[ExtractedSetting]:
        return self.graph.settings

    @property
    def setting_edges(self) -> list[ExtractedSettingEdge]:
        return self.graph.setting_edges

    @property
    def method_setting_links(self) -> list[ExtractedMethodSettingLink]:
        return self.graph.method_setting_links


def _node_kind(node: object) -> str | None:
    if isinstance(node, ExtractedNode):
        return node.kind
    if isinstance(node, dict):
        kind = node.get("kind")
        return str(kind) if kind is not None else None
    kind = getattr(node, "kind", None)
    return str(kind) if kind is not None else None


class ExtractionValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    severity: ValidationSeverity
    code: str
    object_kind: str | None = None
    object_id: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)


class ExtractionValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    errors: list[ExtractionValidationError] = Field(default_factory=list)

    @property
    def blocking_errors(self) -> list[ExtractionValidationError]:
        return [error for error in self.errors if error.severity is ValidationSeverity.error]

    @property
    def warnings(self) -> list[ExtractionValidationError]:
        return [error for error in self.errors if error.severity is ValidationSeverity.warning]

    @property
    def ok(self) -> bool:
        return not self.blocking_errors


class BigModelExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_tier: str
    model: str | None = None
    ok: bool
    used_repair: bool = False
    validation_report: ExtractionValidationReport | None = None
    cost: dict[str, float | int] = Field(default_factory=dict)
    extraction: PaperExtraction | None = None
    error: str | None = None


class BigModelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    results: list[BigModelExtractionResult] = Field(default_factory=list)


class LocalEntityResolutionTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_entity_id: str
    entity_kind: Literal["system", "method", "setting", "problem"]
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    problem_ids: list[str] = Field(default_factory=list)
    mechanism_signature: MechanismSignature | None = None
    allowed_relations: list[ResolutionRelationKind] = Field(
        default_factory=lambda: [
            "same_as",
            "variant_of",
            "uses",
            "extends",
            "subsumes",
            "is_subsumed_by",
            "reimplements",
            "compared_against",
            "distinct",
            "uncertain",
        ]
    )
    evidence_span_ids: list[str] = Field(default_factory=list)


__all__ = [
    "BigModelComparison",
    "BigModelExtractionResult",
    "CandidateNode",
    "ClaimType",
    "DemotedItem",
    "EvidenceClass",
    "EvidenceSpan",
    "ExtractedClaim",
    "ExtractedEdge",
    "ExtractedMethodSettingLink",
    "ExtractedNode",
    "ExtractedOutcome",
    "ExtractedSetting",
    "ExtractedSettingEdge",
    "ExtractionCaps",
    "ExtractionValidationError",
    "ExtractionValidationReport",
    "MethodNodeKind",
    "MethodRelationKind",
    "MethodSettingRelationKind",
    "MechanismSignature",
    "NodeStatus",
    "PaperGraph",
    "PaperExtraction",
    "ProblemStatement",
    "ResolutionRelationKind",
    "LocalEntityResolutionTask",
    "SettingRelationKind",
    "SettingKind",
    "SourceKind",
    "ValidationSeverity",
]
