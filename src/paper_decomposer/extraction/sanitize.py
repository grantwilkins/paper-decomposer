from __future__ import annotations

import re

from .contracts import DemotedItem, ExtractedClaim, ExtractedNode, PaperExtraction, PaperGraph

_GENERIC_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "overview",
    "method",
    "methods",
    "design",
    "evaluation",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "implementation",
    "appendix",
    "scheduling",
    "memory management",
    "optimization",
    "parallelism",
    "batching",
    "caching",
    "architecture",
    "system design",
}


def demote_invalid_method_nodes(extraction: PaperExtraction) -> PaperExtraction:
    """Remove method-family nodes that validators can identify without model help."""
    section_titles = {_normalize(span.section_title) for span in extraction.evidence_spans}
    evidence_by_id = {span.span_id: span for span in extraction.evidence_spans}
    kept_nodes: list[ExtractedNode] = []
    demoted_items = list(extraction.demoted_items)

    for node in extraction.nodes:
        if node.kind == "method" and not _has_mechanism_sentence(node.mechanism_sentence):
            node = _with_grounded_mechanism_sentence(node, evidence_by_id)
        reason = _demotion_reason(node, section_titles)
        if reason is None:
            kept_nodes.append(node)
            continue
        demoted_items.append(
            DemotedItem(
                name=node.canonical_name,
                reason_demoted=reason,
                stored_under=_stored_under(kept_nodes),
                evidence_span_ids=node.evidence_span_ids,
            )
        )

    kept_node_ids = {node.local_node_id for node in kept_nodes}
    setting_ids = {setting.local_setting_id for setting in extraction.settings}
    outcome_ids = {outcome.outcome_id for outcome in extraction.outcomes}
    graph = extraction.graph.model_copy(
        update={
            "systems": [node for node in kept_nodes if node.kind == "system"],
            "methods": [node for node in kept_nodes if node.kind != "system"],
            "method_edges": [
                edge
                for edge in extraction.edges
                if edge.parent_id in kept_node_ids and edge.child_id in kept_node_ids
            ],
            "setting_edges": [
                edge
                for edge in extraction.setting_edges
                if edge.parent_id in setting_ids and edge.child_id in setting_ids
            ],
            "method_setting_links": [
                link
                for link in extraction.method_setting_links
                if link.method_id in kept_node_ids and link.setting_id in setting_ids
            ],
        }
    )
    return extraction.model_copy(
        update={
            "graph": graph,
            "outcomes": [
                outcome.model_copy(
                    update={
                        "method_ids": [method_id for method_id in outcome.method_ids if method_id in kept_node_ids],
                        "setting_ids": [setting_id for setting_id in outcome.setting_ids if setting_id in setting_ids],
                    }
                )
                for outcome in extraction.outcomes
            ],
            "claims": [
                claim.model_copy(
                    update={
                        "method_ids": [method_id for method_id in claim.method_ids if method_id in kept_node_ids],
                        "setting_ids": [setting_id for setting_id in claim.setting_ids if setting_id in setting_ids],
                        "outcome_ids": [outcome_id for outcome_id in claim.outcome_ids if outcome_id in outcome_ids],
                    }
                )
                for claim in extraction.claims
            ],
            "demoted_items": demoted_items,
        }
    )


def preserve_graph_and_attach_claims(
    extraction: PaperExtraction,
    *,
    fallback_graph: PaperGraph | None = None,
) -> PaperExtraction:
    """Keep the graph-first invariant intact across compression and repair drafts."""
    graph = _merge_graph(extraction.graph, fallback_graph)
    extraction = extraction.model_copy(update={"graph": graph})
    return extraction.model_copy(
        update={"claims": [_attach_claim(claim, extraction.graph) for claim in extraction.claims]}
    )


def _merge_graph(graph: PaperGraph, fallback_graph: PaperGraph | None) -> PaperGraph:
    if fallback_graph is None:
        return graph
    return graph.model_copy(
        update={
            "systems": graph.systems or fallback_graph.systems,
            "methods": graph.methods or fallback_graph.methods,
            "method_edges": graph.method_edges or fallback_graph.method_edges,
            "settings": graph.settings or fallback_graph.settings,
            "setting_edges": graph.setting_edges or fallback_graph.setting_edges,
            "method_setting_links": graph.method_setting_links or fallback_graph.method_setting_links,
        }
    )


def _attach_claim(claim: ExtractedClaim, graph: PaperGraph) -> ExtractedClaim:
    if claim.method_ids or claim.setting_ids or claim.outcome_ids:
        return claim

    text = _normalize_identifier(f"{claim.raw_text} {claim.finding}")
    matching_nodes = [
        node
        for node in graph.nodes
        if _name_appears_in_text(node.canonical_name, text)
        or any(_name_appears_in_text(alias, text) for alias in node.aliases)
    ]
    if matching_nodes:
        node = max(matching_nodes, key=lambda node: len(_normalize_identifier(node.canonical_name)))
        return claim.model_copy(update={"method_ids": [node.local_node_id]})

    matching_settings = [
        setting
        for setting in graph.settings
        if _name_appears_in_text(setting.canonical_name, text)
        or any(_name_appears_in_text(alias, text) for alias in setting.aliases)
    ]
    if matching_settings:
        setting = max(matching_settings, key=lambda setting: len(_normalize_identifier(setting.canonical_name)))
        return claim.model_copy(update={"setting_ids": [setting.local_setting_id]})

    if len(graph.systems) == 1:
        return claim.model_copy(update={"method_ids": [graph.systems[0].local_node_id]})

    return claim


def _with_grounded_mechanism_sentence(
    node: ExtractedNode,
    evidence_by_id: dict[str, object],
) -> ExtractedNode:
    for span_id in node.evidence_span_ids:
        span = evidence_by_id.get(span_id)
        text = str(getattr(span, "text", ""))
        sentence = _sentence_mentioning(text, node.canonical_name)
        if sentence is not None:
            return node.model_copy(update={"mechanism_sentence": sentence})
    return node


def _sentence_mentioning(text: str, name: str) -> str | None:
    normalized_name = _normalize(name)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cleaned = " ".join(sentence.strip().split())
        if normalized_name not in _normalize(cleaned):
            continue
        if _has_mechanism_sentence(cleaned):
            return cleaned
    cleaned_text = " ".join(text.strip().split())
    if normalized_name in _normalize(cleaned_text) and _has_mechanism_sentence(cleaned_text):
        return cleaned_text[:300].rsplit(" ", 1)[0].strip()
    return None


def _demotion_reason(node: ExtractedNode, section_titles: set[str]) -> str | None:
    normalized_name = _normalize(node.canonical_name)
    if normalized_name in section_titles and normalized_name in _GENERIC_SECTION_HEADINGS:
        return "Paper section heading is not a reusable method-family node."
    if node.kind == "method" and not _has_mechanism_sentence(node.mechanism_sentence):
        return "Method node lacks a grounded mechanism sentence with inputs, outputs, and operative move."
    return None


def _stored_under(kept_nodes: list[ExtractedNode]) -> str:
    for node in reversed(kept_nodes):
        if node.kind in {"method", "system"}:
            return node.local_node_id
    return "paper"


def _has_mechanism_sentence(value: str | None) -> bool:
    if not value:
        return False
    words = re.findall(r"[A-Za-z0-9_%-]+", value)
    return len(words) >= 8


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _name_appears_in_text(name: str, normalized_text: str) -> bool:
    normalized_name = _normalize_identifier(name)
    if not normalized_name:
        return False
    return f" {normalized_name} " in f" {normalized_text} "


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


__all__ = ["demote_invalid_method_nodes", "preserve_graph_and_attach_claims"]
