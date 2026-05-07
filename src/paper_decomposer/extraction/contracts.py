from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MethodNodeKind = Literal["system", "method", "method_category"]
SettingKind = Literal["application", "task", "dataset", "workload", "hardware", "model_artifact", "metric"]
MethodRelationKind = Literal["uses", "is_a", "refines"]
MethodSettingRelationKind = Literal["applies_to", "evaluated_on"]
ClaimType = Literal[
    "performance",
    "memory",
    "scalability",
    "quality",
    "capability",
    "limitation",
    "comparison",
    "other",
]
NodeStatus = Literal["claimed_new", "reference", "uncertain"]
SourceKind = Literal["abstract", "paragraph", "caption", "table_text", "contribution", "conclusion"]


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
    confidence: float = 0.0


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_node_id: str
    kind: MethodNodeKind
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str
    status: NodeStatus = "uncertain"
    introduced_by: str | None = None
    granularity_rationale: str
    evidence_span_ids: list[str]
    confidence: float = 0.0
    mechanism_sentence: str | None = None

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
    confidence: float = 0.0


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str
    child_id: str
    relation_kind: MethodRelationKind
    evidence_span_ids: list[str]
    confidence: float = 0.0


class ExtractedMethodSettingLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method_id: str
    setting_id: str
    relation_kind: MethodSettingRelationKind
    evidence_span_ids: list[str]
    confidence: float = 0.0


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
    confidence: float = 0.0


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    paper_id: str
    claim_type: ClaimType
    raw_text: str
    finding: str
    method_ids: list[str] = Field(default_factory=list)
    setting_ids: list[str] = Field(default_factory=list)
    outcome_ids: list[str] = Field(default_factory=list)
    metric: str | None = None
    value: str | None = None
    delta: str | None = None
    baseline: str | None = None
    comparator: str | None = None
    evidence_span_ids: list[str]
    confidence: float = 0.0


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


class PaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    extraction_run_id: str
    title: str
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    candidates: list[CandidateNode] = Field(default_factory=list)
    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    settings: list[ExtractedSetting] = Field(default_factory=list)
    method_setting_links: list[ExtractedMethodSettingLink] = Field(default_factory=list)
    outcomes: list[ExtractedOutcome] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    demoted_items: list[DemotedItem] = Field(default_factory=list)


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


__all__ = [
    "CandidateNode",
    "ClaimType",
    "DemotedItem",
    "EvidenceSpan",
    "ExtractedClaim",
    "ExtractedEdge",
    "ExtractedMethodSettingLink",
    "ExtractedNode",
    "ExtractedOutcome",
    "ExtractedSetting",
    "ExtractionCaps",
    "ExtractionValidationError",
    "ExtractionValidationReport",
    "MethodNodeKind",
    "MethodRelationKind",
    "MethodSettingRelationKind",
    "NodeStatus",
    "PaperExtraction",
    "SettingKind",
    "SourceKind",
    "ValidationSeverity",
]
