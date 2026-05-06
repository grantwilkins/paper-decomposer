from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ExtractionValidationError, PaperExtraction
from .validators import validate_extraction

_METHOD_EDGE_RELATION = {
    "uses": "composes",
    "is_a": "is_a",
    "refines": "refines",
}


class DBWritePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_spans: list[dict[str, object]] = Field(default_factory=list)
    methods: list[dict[str, object]] = Field(default_factory=list)
    method_edges: list[dict[str, object]] = Field(default_factory=list)
    settings: list[dict[str, object]] = Field(default_factory=list)
    method_setting_links: list[dict[str, object]] = Field(default_factory=list)
    outcomes: list[dict[str, object]] = Field(default_factory=list)
    claims: list[dict[str, object]] = Field(default_factory=list)
    evidence_links: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[ExtractionValidationError] = Field(default_factory=list)


class ExtractionPersistenceError(RuntimeError):
    def __init__(self, errors: list[ExtractionValidationError]) -> None:
        self.errors = errors
        message = "; ".join(error.code for error in errors)
        super().__init__(f"Extraction has blocking validation errors: {message}")


def build_db_write_plan(extraction: PaperExtraction) -> DBWritePlan:
    report = validate_extraction(extraction)
    if report.blocking_errors:
        raise ExtractionPersistenceError(report.blocking_errors)

    plan = DBWritePlan(warnings=report.warnings)
    plan.evidence_spans = [_evidence_span_row(extraction, span) for span in extraction.evidence_spans]

    for node in extraction.nodes:
        plan.methods.append(
            {
                "paper_id": extraction.paper_id,
                "extraction_run_id": extraction.extraction_run_id,
                "local_node_id": node.local_node_id,
                "canonical_name": node.canonical_name,
                "description": node.description,
                "metadata": {
                    "kind": node.kind,
                    "aliases": node.aliases,
                    "status": node.status,
                    "introduced_by": node.introduced_by,
                    "granularity_rationale": node.granularity_rationale,
                    "mechanism_sentence": node.mechanism_sentence,
                    "confidence": node.confidence,
                    "evidence_span_ids": node.evidence_span_ids,
                },
            }
        )
        plan.evidence_links.extend(_evidence_links(node.evidence_span_ids, "method", node.local_node_id))

    for edge in extraction.edges:
        relation = _METHOD_EDGE_RELATION[edge.relation_kind]
        local_edge_id = f"{edge.parent_id}->{edge.child_id}:{edge.relation_kind}"
        plan.method_edges.append(
            {
                "local_edge_id": local_edge_id,
                "parent_local_node_id": edge.parent_id,
                "child_local_node_id": edge.child_id,
                "relation": relation,
                "confidence": edge.confidence,
                "metadata": {
                    "paper_local_relation": edge.relation_kind,
                    "evidence_span_ids": edge.evidence_span_ids,
                },
            }
        )
        plan.evidence_links.extend(_evidence_links(edge.evidence_span_ids, "method_edge", local_edge_id))

    for setting in extraction.settings:
        plan.settings.append(
            {
                "paper_id": extraction.paper_id,
                "extraction_run_id": extraction.extraction_run_id,
                "local_setting_id": setting.local_setting_id,
                "kind": setting.kind,
                "canonical_name": setting.canonical_name,
                "description": setting.description,
                "metadata": {
                    "aliases": setting.aliases,
                    "confidence": setting.confidence,
                    "evidence_span_ids": setting.evidence_span_ids,
                },
            }
        )
        plan.evidence_links.extend(_evidence_links(setting.evidence_span_ids, "setting", setting.local_setting_id))

    for link in extraction.method_setting_links:
        local_link_id = f"{link.method_id}->{link.setting_id}:{link.relation_kind}"
        plan.method_setting_links.append(
            {
                "local_link_id": local_link_id,
                "paper_id": extraction.paper_id,
                "method_local_node_id": link.method_id,
                "setting_local_setting_id": link.setting_id,
                "relation": link.relation_kind,
                "confidence": link.confidence,
                "metadata": {"evidence_span_ids": link.evidence_span_ids},
            }
        )
        plan.evidence_links.extend(_evidence_links(link.evidence_span_ids, "method_setting_link", local_link_id))

    for outcome in extraction.outcomes:
        plan.outcomes.append(
            {
                "paper_id": extraction.paper_id,
                "local_outcome_id": outcome.outcome_id,
                "method_local_node_ids": outcome.method_ids,
                "setting_local_setting_ids": outcome.setting_ids,
                "metric_name": outcome.metric,
                "value": outcome.value,
                "delta_value": outcome.delta,
                "baseline": outcome.baseline,
                "units": outcome.units,
                "metadata": {
                    "comparator": outcome.comparator,
                    "confidence": outcome.confidence,
                    "evidence_span_ids": outcome.evidence_span_ids,
                },
            }
        )
        plan.evidence_links.extend(_evidence_links(outcome.evidence_span_ids, "outcome", outcome.outcome_id))

    for claim in extraction.claims:
        plan.claims.append(
            {
                "paper_id": extraction.paper_id,
                "local_claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "statement": claim.finding,
                "strength": claim.confidence,
                "metadata": {
                    "raw_text": claim.raw_text,
                    "method_ids": claim.method_ids,
                    "setting_ids": claim.setting_ids,
                    "outcome_ids": claim.outcome_ids,
                    "metric": claim.metric,
                    "value": claim.value,
                    "delta": claim.delta,
                    "baseline": claim.baseline,
                    "comparator": claim.comparator,
                    "evidence_span_ids": claim.evidence_span_ids,
                },
            }
        )
        plan.evidence_links.extend(_evidence_links(claim.evidence_span_ids, "claim", claim.claim_id))

    return plan


def _evidence_span_row(extraction: PaperExtraction, span: object) -> dict[str, object]:
    return {
        "paper_id": extraction.paper_id,
        "extraction_run_id": extraction.extraction_run_id,
        "local_span_id": getattr(span, "span_id"),
        "section_title": getattr(span, "section_title"),
        "section_kind": getattr(span, "section_kind"),
        "page_start": getattr(span, "page_start"),
        "page_end": getattr(span, "page_end"),
        "artifact_id": getattr(span, "artifact_id"),
        "source_kind": getattr(span, "source_kind"),
        "text": getattr(span, "text"),
    }


def _evidence_links(span_ids: list[str], target_kind: str, target_id: str) -> list[dict[str, object]]:
    return [
        {
            "local_span_id": span_id,
            "target_kind": target_kind,
            "local_target_id": target_id,
        }
        for span_id in span_ids
    ]


__all__ = ["DBWritePlan", "ExtractionPersistenceError", "build_db_write_plan"]
