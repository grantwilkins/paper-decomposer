from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_decomposer.models import call_model
from paper_decomposer.schema import ModelTier

from .contracts import (
    CandidateNode,
    DemotedItem,
    EvidenceSpan,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedOutcome,
    ExtractedSetting,
    ExtractionCaps,
    ExtractionValidationError,
    PaperGraph,
    PaperExtraction,
)
from .prompts import (
    big_model_compact_prompt,
    claims_outcomes_prompt,
    cleanup_prompt,
    compression_prompt,
    frontmatter_prompt,
    method_graph_prompt,
    repair_prompt,
)


class GraphSketch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_metadata: dict[str, Any] = Field(default_factory=dict)
    central_problem_candidates: list[str] = Field(default_factory=list)
    contribution_span_ids: list[str] = Field(default_factory=list)
    candidates: list[CandidateNode] = Field(default_factory=list)
    graph: PaperGraph = Field(default_factory=PaperGraph)
    central_primitive_guess: str | None = None
    demoted_items: list[DemotedItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_graph_fields(cls, data: object) -> object:
        return _lift_graph_payload(data)

    @property
    def nodes(self) -> list[ExtractedNode]:
        return self.graph.nodes

    @property
    def edges(self) -> list[ExtractedEdge]:
        return self.graph.method_edges


FrontmatterSketch = GraphSketch


class MethodGraphDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateNode] = Field(default_factory=list)
    graph: PaperGraph = Field(default_factory=PaperGraph)
    demoted_items: list[DemotedItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_graph_fields(cls, data: object) -> object:
        return _lift_graph_payload(data)

    @property
    def nodes(self) -> list[ExtractedNode]:
        return self.graph.nodes

    @property
    def edges(self) -> list[ExtractedEdge]:
        return self.graph.method_edges


class ClaimsOutcomesDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: list[ExtractedSetting] = Field(default_factory=list)
    outcomes: list[ExtractedOutcome] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: PaperGraph = Field(default_factory=PaperGraph)
    outcomes: list[ExtractedOutcome] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    demoted_items: list[DemotedItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_graph_fields(cls, data: object) -> object:
        lifted = _lift_graph_payload(data)
        if isinstance(lifted, dict):
            lifted.pop("candidates", None)
        return lifted


async def extract_frontmatter_sketch(
    spans: list[EvidenceSpan],
    *,
    config: Any,
) -> FrontmatterSketch:
    return await call_model(
        _default_model_tier(config),
        frontmatter_prompt(spans),
        response_schema=FrontmatterSketch,
        config=config,
    )


async def extract_method_graph(
    spans: list[EvidenceSpan],
    sketch: FrontmatterSketch,
    *,
    config: Any,
) -> MethodGraphDraft:
    return await call_model(
        _default_model_tier(config),
        method_graph_prompt(spans, sketch.model_dump_json()),
        response_schema=MethodGraphDraft,
        config=config,
    )


async def extract_claims_and_outcomes(
    spans: list[EvidenceSpan],
    graph: MethodGraphDraft,
    *,
    config: Any,
) -> ClaimsOutcomesDraft:
    graph_json = graph.model_dump_json(exclude={"candidates", "demoted_items"})
    return await call_model(
        _default_model_tier(config),
        claims_outcomes_prompt(spans, graph_json),
        response_schema=ClaimsOutcomesDraft,
        config=config,
    )


async def extract_big_model_draft(
    spans: list[EvidenceSpan],
    *,
    config: Any,
    tier: ModelTier | None = None,
    caps: ExtractionCaps | None = None,
) -> ExtractionDraft:
    return await call_model(
        tier or _big_model_draft_tier(config),
        big_model_compact_prompt(spans, caps or ExtractionCaps()),
        response_schema=ExtractionDraft,
        config=config,
    )


async def compress_paper_extraction(
    graph: MethodGraphDraft,
    claims: ClaimsOutcomesDraft,
    *,
    config: Any,
) -> ExtractionDraft:
    return await call_model(
        _default_model_tier(config),
        compression_prompt(graph.model_dump_json(exclude={"candidates", "demoted_items"}), claims.model_dump_json()),
        response_schema=ExtractionDraft,
        config=config,
    )


async def repair_paper_extraction(
    extraction: PaperExtraction,
    validation_errors: list[ExtractionValidationError],
    *,
    config: Any,
    tier: ModelTier | None = None,
) -> ExtractionDraft:
    return await call_model(
        tier or _repair_model_tier(config),
        repair_prompt(extraction.model_dump_json(), validation_errors, extraction.evidence_spans),
        response_schema=ExtractionDraft,
        config=config,
    )


async def cleanup_paper_extraction(
    extraction: PaperExtraction,
    validation_issues: list[ExtractionValidationError],
    *,
    config: Any,
) -> ExtractionDraft:
    return await call_model(
        _adjudication_model_tier(config),
        cleanup_prompt(
            extraction.model_dump_json(exclude={"evidence_spans"}),
            validation_issues,
            extraction.evidence_spans,
        ),
        response_schema=ExtractionDraft,
        config=config,
    )


def _default_model_tier(config: Any) -> ModelTier:
    return _model_tier_from_config(config, key="default_model_tier", default="small")


def _repair_model_tier(config: Any) -> ModelTier:
    return _model_tier_from_config(config, key="repair_model_tier", default=_default_model_tier(config))


def _adjudication_model_tier(config: Any) -> ModelTier:
    return _model_tier_from_config(config, key="adjudication_model_tier", default=_default_model_tier(config))


def _big_model_draft_tier(config: Any) -> ModelTier:
    return _model_tier_from_config(config, key="big_model_draft_tier", default=_default_model_tier(config))


def _model_tier_from_config(config: Any, *, key: str, default: ModelTier) -> ModelTier:
    extraction = _extraction_config(config)
    raw_tier = extraction.get(key, default)
    if raw_tier == "cheap":
        return "small"
    if raw_tier in {"small", "medium", "heavy"}:
        return raw_tier
    return default


def _extraction_config(config: Any) -> dict[str, Any]:
    pipeline = getattr(config, "pipeline", None)
    extraction = getattr(pipeline, "extraction", None)
    if isinstance(extraction, dict):
        return extraction
    if isinstance(config, dict):
        raw = config.get("pipeline", {}).get("extraction", {})
        if isinstance(raw, dict):
            return raw
    return {}


def _lift_graph_payload(data: object) -> object:
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
    if graph:
        lifted["graph"] = graph
    return lifted


__all__ = [
    "ClaimsOutcomesDraft",
    "ExtractionDraft",
    "FrontmatterSketch",
    "GraphSketch",
    "MethodGraphDraft",
    "cleanup_paper_extraction",
    "compress_paper_extraction",
    "extract_big_model_draft",
    "extract_claims_and_outcomes",
    "extract_frontmatter_sketch",
    "extract_method_graph",
    "repair_paper_extraction",
]
