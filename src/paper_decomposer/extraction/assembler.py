from __future__ import annotations

from typing import TypeVar

from .contracts import (
    CandidateNode,
    DemotedItem,
    EvidenceSpan,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    PaperExtraction,
)
from .stages import ClaimsOutcomesDraft, ExtractionDraft, FrontmatterSketch, MethodGraphDraft

T = TypeVar("T")


def assemble_extraction(
    *,
    paper_id: str,
    extraction_run_id: str,
    title: str,
    evidence_spans: list[EvidenceSpan],
    sketch: FrontmatterSketch,
    graph: MethodGraphDraft,
    claims: ClaimsOutcomesDraft,
    final: ExtractionDraft | None = None,
) -> PaperExtraction:
    if final is not None:
        return PaperExtraction(
            paper_id=paper_id,
            extraction_run_id=extraction_run_id,
            title=title,
            evidence_spans=evidence_spans,
            candidates=final.candidates,
            nodes=final.nodes,
            edges=final.edges,
            settings=final.settings,
            method_setting_links=final.method_setting_links,
            outcomes=final.outcomes,
            claims=final.claims,
            demoted_items=final.demoted_items,
        )

    return PaperExtraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=title,
        evidence_spans=evidence_spans,
        candidates=_dedupe_by_name([*sketch.candidates, *graph.candidates]),
        nodes=_dedupe_by_id(graph.nodes, key="local_node_id"),
        edges=_dedupe_edges(graph.edges),
        settings=_dedupe_by_id([*graph.settings, *claims.settings], key="local_setting_id"),
        method_setting_links=_dedupe_method_setting_links(graph.method_setting_links),
        outcomes=_dedupe_by_id(claims.outcomes, key="outcome_id"),
        claims=_dedupe_by_id(claims.claims, key="claim_id"),
        demoted_items=_dedupe_demoted_items(graph.demoted_items),
    )


def _dedupe_by_name(items: list[CandidateNode]) -> list[CandidateNode]:
    seen: set[str] = set()
    result: list[CandidateNode] = []
    for item in items:
        key = item.name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_by_id(items: list[T], *, key: str) -> list[T]:
    seen: set[str] = set()
    result: list[T] = []
    for item in items:
        item_id = str(getattr(item, key))
        if item_id in seen:
            continue
        seen.add(item_id)
        result.append(item)
    return result


def _dedupe_edges(edges: list[ExtractedEdge]) -> list[ExtractedEdge]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ExtractedEdge] = []
    for edge in edges:
        key = (edge.parent_id, edge.child_id, edge.relation_kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def _dedupe_method_setting_links(
    links: list[ExtractedMethodSettingLink],
) -> list[ExtractedMethodSettingLink]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ExtractedMethodSettingLink] = []
    for link in links:
        key = (link.method_id, link.setting_id, link.relation_kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(link)
    return result


def _dedupe_demoted_items(items: list[DemotedItem]) -> list[DemotedItem]:
    seen: set[tuple[str, str]] = set()
    result: list[DemotedItem] = []
    for item in items:
        key = (item.name.casefold(), item.stored_under)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


__all__ = ["assemble_extraction"]
