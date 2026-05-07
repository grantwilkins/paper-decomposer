from __future__ import annotations

import re

from .contracts import (
    DemotedItem,
    ExtractedClaim,
    ExtractedEdge,
    ExtractedMethodSettingLink,
    ExtractedNode,
    ExtractedOutcome,
    ExtractedSetting,
    PaperExtraction,
    PaperGraph,
)

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

_PROBLEM_SETTING_TERMS = (
    "inefficiency",
    "challenge",
    "problem",
    "limitation",
    "bottleneck",
    "fragmentation issue",
)

_NAMED_SETTINGS = (
    (
        "setting:llm_serving",
        "LLM serving",
        "application",
        ("llm serving", "serving systems", "serving system"),
        "Serving large language model requests.",
    ),
    (
        "setting:basic_sampling",
        "basic sampling",
        "task",
        ("basic sampling", "one sample per request"),
        "Single-sample generation requests.",
    ),
    (
        "setting:parallel_sampling",
        "parallel sampling",
        "task",
        ("parallel sampling", "parallel generation"),
        "Generating multiple sampled outputs for one prompt.",
    ),
    (
        "setting:beam_search",
        "beam search",
        "task",
        ("beam search", "beam width"),
        "Beam-search decoding.",
    ),
    (
        "setting:shared_prefix_prompting",
        "shared-prefix prompting",
        "task",
        ("shared prefix", "prefix sharing", "shared-prefix"),
        "Requests whose prompts share a reusable prefix.",
    ),
    (
        "setting:chatbot_serving",
        "chatbot serving",
        "application",
        ("chatbot",),
        "Chatbot request serving.",
    ),
    ("setting:sharegpt", "ShareGPT", "dataset", ("sharegpt",), "ShareGPT workload dataset."),
    ("setting:alpaca", "Alpaca", "dataset", ("alpaca",), "Alpaca workload dataset."),
    (
        "setting:wmt16_english_to_german",
        "WMT16 English-to-German",
        "dataset",
        ("wmt16", "english-to-german", "english to german"),
        "WMT16 English-to-German translation workload.",
    ),
    ("setting:opt_13b", "OPT-13B", "model_artifact", ("opt-13b",), "OPT-13B model artifact."),
    ("setting:opt_66b", "OPT-66B", "model_artifact", ("opt-66b",), "OPT-66B model artifact."),
    ("setting:opt_175b", "OPT-175B", "model_artifact", ("opt-175b",), "OPT-175B model artifact."),
    ("setting:llama_13b", "LLaMA-13B", "model_artifact", ("llama-13b",), "LLaMA-13B model artifact."),
    ("setting:nvidia_a100", "NVIDIA A100", "hardware", ("nvidia a100", "a100"), "NVIDIA A100 GPU hardware."),
)

_COMPONENT_DEMOTION_SPECS = (
    ("KV cache manager", ("kv cache manager",), "Component too broad for a reusable method node."),
    ("centralized scheduler", ("centralized scheduler",), "System component; preserve under the relevant scheduling method."),
    ("distributed GPU workers", ("gpu workers", "distributed gpu workers"), "Implementation plumbing for distributed execution."),
    ("fused reshape and block write", ("fused reshape and block write",), "Kernel implementation detail."),
    ("fused block copy", ("fused block copy",), "Kernel implementation detail."),
    ("FastAPI frontend", ("fastapi",), "Serving frontend implementation detail."),
    ("OpenAI-compatible API", ("openai api",), "Serving API compatibility detail."),
    ("fork / append / free helper methods", ("fork method", "append method", "free method"), "Helper API, not a first-class method node."),
)


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
    extraction = _repair_vllm_graph_shape(extraction)
    extraction = _repair_setting_taxonomy(extraction)
    extraction = _demote_component_details(extraction)
    extraction = extraction.model_copy(update={"claims": [_repair_claim(claim, extraction.graph) for claim in extraction.claims]})
    return _materialize_claim_outcomes(extraction)


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


def _repair_claim(claim: ExtractedClaim, graph: PaperGraph) -> ExtractedClaim:
    claim = _retarget_overall_system_claim(claim, graph)
    claim = _retarget_claim_settings(claim, graph)
    if _is_overhead_claim(claim):
        claim = claim.model_copy(update={"claim_type": "overhead"})
    return _attach_claim(claim, graph)


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


def _repair_vllm_graph_shape(extraction: PaperExtraction) -> PaperExtraction:
    graph = extraction.graph
    methods = [_repair_node_status(_tighten_blockwise_node(node), extraction) for node in graph.methods]
    systems = [_repair_node_status(node, extraction) for node in graph.systems]
    graph = graph.model_copy(update={"systems": systems, "methods": methods})

    system = _first_node_matching(graph.systems, ("vllm",))
    paged_attention = _first_node_matching(graph.methods, ("pagedattention", "paged attention"))
    block_sharing = _first_node_matching(graph.methods, ("block level", "sharing"))
    copy_on_write = _first_node_matching(graph.methods, ("copy on write", "copy-on-write"))
    swapping = _first_node_matching(graph.methods, ("swapping",))
    recomputation = _first_node_matching(graph.methods, ("recomputation",))

    methods = list(graph.methods)
    preemption = _first_node_matching(methods, ("sequence group preemption", "preemption"))
    if preemption is None and (swapping is not None or recomputation is not None):
        preemption = _sequence_group_preemption_node(extraction, swapping=swapping, recomputation=recomputation)
        methods.append(preemption)

    edges = list(graph.method_edges)
    if system is not None and paged_attention is not None:
        edges = _ensure_edge(edges, system.local_node_id, paged_attention.local_node_id, _combined_evidence(system, paged_attention))
    if block_sharing is not None and copy_on_write is not None:
        edges = _reparent_edge(
            edges,
            child_id=copy_on_write.local_node_id,
            new_parent_id=block_sharing.local_node_id,
            evidence_span_ids=copy_on_write.evidence_span_ids,
        )
    if system is not None and preemption is not None:
        edges = _ensure_edge(edges, system.local_node_id, preemption.local_node_id, preemption.evidence_span_ids)
    if preemption is not None and swapping is not None:
        edges = _reparent_edge(
            edges,
            child_id=swapping.local_node_id,
            new_parent_id=preemption.local_node_id,
            evidence_span_ids=swapping.evidence_span_ids,
        )
    if preemption is not None and recomputation is not None:
        edges = _reparent_edge(
            edges,
            child_id=recomputation.local_node_id,
            new_parent_id=preemption.local_node_id,
            evidence_span_ids=recomputation.evidence_span_ids,
        )

    graph = graph.model_copy(update={"methods": methods, "method_edges": _dedupe_edges(edges)})
    return extraction.model_copy(update={"graph": graph})


def _repair_setting_taxonomy(extraction: PaperExtraction) -> PaperExtraction:
    text_by_span = {span.span_id: span.text for span in extraction.evidence_spans}
    evidence_text = _normalize_identifier(" ".join(text_by_span.values()))
    graph = extraction.graph
    demoted_items = list(extraction.demoted_items)
    settings_by_id = {setting.local_setting_id: setting for setting in graph.settings}

    for spec in _NAMED_SETTINGS:
        setting_id, name, kind, markers, description = spec
        if setting_id in settings_by_id:
            continue
        span_ids = _matching_span_ids(extraction, markers)
        if not span_ids and not any(_marker_in_text(marker, evidence_text) for marker in markers):
            continue
        settings_by_id[setting_id] = ExtractedSetting(
            local_setting_id=setting_id,
            kind=kind,  # type: ignore[arg-type]
            canonical_name=name,
            description=description,
            evidence_span_ids=span_ids or [extraction.evidence_spans[0].span_id],
        )

    removed_setting_ids: set[str] = set()
    kept_settings: list[ExtractedSetting] = []
    coarse_scenario_ids: set[str] = set()
    for setting in settings_by_id.values():
        normalized_name = _normalize_identifier(setting.canonical_name)
        if _is_problem_setting_name(normalized_name):
            removed_setting_ids.add(setting.local_setting_id)
            demoted_items.append(
                DemotedItem(
                    name=setting.canonical_name,
                    reason_demoted="Problem/challenge context is not a task, application, workload, hardware, dataset, model artifact, or metric setting.",
                    stored_under=_first_system_id(graph) or "paper",
                    evidence_span_ids=setting.evidence_span_ids,
                )
            )
            continue
        if normalized_name in {"decoding scenarios", "decoding scenario"}:
            coarse_scenario_ids.add(setting.local_setting_id)
            demoted_items.append(
                DemotedItem(
                    name=setting.canonical_name,
                    reason_demoted="Coarse scenario bucket split into named task settings.",
                    stored_under="paper",
                    evidence_span_ids=setting.evidence_span_ids,
                )
            )
            continue
        kept_settings.append(setting)

    replacement_ids = _present_named_setting_ids(kept_settings)
    method_setting_links = _repair_method_setting_links(
        graph.method_setting_links,
        removed_setting_ids=removed_setting_ids,
        coarse_scenario_ids=coarse_scenario_ids,
        replacement_ids=replacement_ids,
    )
    setting_edges = [
        edge
        for edge in graph.setting_edges
        if edge.parent_id not in removed_setting_ids | coarse_scenario_ids
        and edge.child_id not in removed_setting_ids | coarse_scenario_ids
    ]
    graph = graph.model_copy(
        update={
            "settings": kept_settings,
            "setting_edges": setting_edges,
            "method_setting_links": method_setting_links,
        }
    )
    extraction = extraction.model_copy(update={"graph": graph, "demoted_items": _dedupe_demotions(demoted_items)})
    return _drop_removed_setting_references(extraction, removed_setting_ids | coarse_scenario_ids)


def _demote_component_details(extraction: PaperExtraction) -> PaperExtraction:
    promoted_names = {_normalize_identifier(node.canonical_name) for node in extraction.nodes}
    existing_demotions = {_normalize_identifier(item.name) for item in extraction.demoted_items}
    demoted_items = list(extraction.demoted_items)
    for name, markers, reason in _COMPONENT_DEMOTION_SPECS:
        normalized_name = _normalize_identifier(name)
        if normalized_name in promoted_names or normalized_name in existing_demotions:
            continue
        span_ids = _matching_span_ids(extraction, markers)
        if not span_ids:
            continue
        demoted_items.append(
            DemotedItem(
                name=name,
                reason_demoted=reason,
                stored_under=_component_storage_target(extraction.graph, name),
                evidence_span_ids=span_ids[:3],
            )
        )
        existing_demotions.add(normalized_name)
    return extraction.model_copy(update={"demoted_items": demoted_items})


def _retarget_overall_system_claim(claim: ExtractedClaim, graph: PaperGraph) -> ExtractedClaim:
    system = _first_system(graph)
    if system is None:
        return claim
    text = _normalize_identifier(f"{claim.raw_text} {claim.finding}")
    if not _name_appears_in_text(system.canonical_name, text):
        return claim
    metric_text = _normalize_identifier(" ".join(part for part in [claim.metric, claim.raw_text, claim.finding] if part))
    if not any(term in metric_text for term in ("request rate", "throughput", "latency", "requests")):
        return claim
    if _is_overhead_claim(claim):
        return claim
    return claim.model_copy(update={"method_ids": [system.local_node_id]})


def _retarget_claim_settings(claim: ExtractedClaim, graph: PaperGraph) -> ExtractedClaim:
    text = _normalize_identifier(f"{claim.raw_text} {claim.finding}")
    matching_ids = [
        setting.local_setting_id
        for setting in graph.settings
        if _setting_matches_claim(setting, text)
    ]
    if not matching_ids and _claim_mentions_single_system(claim, graph):
        matching_ids = [
            setting.local_setting_id
            for setting in graph.settings
            if setting.local_setting_id == "setting:llm_serving"
        ]
    if not matching_ids:
        return claim
    merged = [setting_id for setting_id in claim.setting_ids if setting_id in {setting.local_setting_id for setting in graph.settings}]
    for setting_id in matching_ids:
        if setting_id not in merged:
            merged.append(setting_id)
    return claim.model_copy(update={"setting_ids": merged})


def _materialize_claim_outcomes(extraction: PaperExtraction) -> PaperExtraction:
    outcomes_by_id = {outcome.outcome_id: outcome for outcome in extraction.outcomes}
    claims: list[ExtractedClaim] = []
    for claim in extraction.claims:
        if not _claim_requires_outcome(claim):
            claims.append(claim)
            continue
        outcome_id = f"outcome:{claim.claim_id}"
        if outcome_id not in outcomes_by_id:
            outcomes_by_id[outcome_id] = ExtractedOutcome(
                outcome_id=outcome_id,
                paper_id=extraction.paper_id,
                metric=claim.metric or "unspecified metric",
                method_ids=claim.method_ids,
                setting_ids=claim.setting_ids,
                value=claim.value,
                delta=claim.delta,
                baseline=claim.baseline,
                comparator=claim.comparator,
                evidence_span_ids=claim.evidence_span_ids,
                confidence=claim.confidence,
            )
        outcome_ids = list(claim.outcome_ids)
        if outcome_id not in outcome_ids:
            outcome_ids.append(outcome_id)
        claims.append(claim.model_copy(update={"outcome_ids": outcome_ids}))
    return extraction.model_copy(update={"claims": claims, "outcomes": list(outcomes_by_id.values())})


def _tighten_blockwise_node(node: ExtractedNode) -> ExtractedNode:
    normalized_name = _normalize_identifier(node.canonical_name)
    if normalized_name not in {"block wise kv cache", "blockwise kv blocks", "block wise kv blocks"}:
        return node
    return node.model_copy(
        update={
            "canonical_name": "Block-wise KV cache address translation",
            "description": (
                "Block-table mapping from a request's logical KV blocks to non-contiguous "
                "physical KV blocks."
            ),
            "mechanism_sentence": (
                "Given logical KV blocks and a block table, the method outputs physical KV "
                "block addresses for attention reads and cache writes."
            ),
        }
    )


def _repair_node_status(node: ExtractedNode, extraction: PaperExtraction) -> ExtractedNode:
    if node.status != "uncertain":
        return node
    evidence_text = " ".join(
        span.text
        for span in extraction.evidence_spans
        if span.span_id in node.evidence_span_ids
    )
    normalized_text = _normalize_identifier(evidence_text)
    normalized_name = _normalize_identifier(node.canonical_name)
    if not normalized_name:
        return node
    novelty_markers = ("we propose", "we build", "we design", "we implement", "we develop", "we introduce")
    if any(marker in normalized_text for marker in novelty_markers) and _name_appears_in_text(node.canonical_name, normalized_text):
        return node.model_copy(update={"status": "claimed_new"})
    return node


def _sequence_group_preemption_node(
    extraction: PaperExtraction,
    *,
    swapping: ExtractedNode | None,
    recomputation: ExtractedNode | None,
) -> ExtractedNode:
    span_ids = _matching_span_ids(extraction, ("preempt", "sequence group", "swapping", "recomputation"))
    if not span_ids:
        span_ids = [span_id for node in (swapping, recomputation) if node is not None for span_id in node.evidence_span_ids]
    return ExtractedNode(
        local_node_id="method:sequence_group_preemption",
        kind="method",
        canonical_name="Sequence-group preemption",
        description="Scheduling policy that preempts whole sequence groups and recovers their KV cache later.",
        status="claimed_new",
        introduced_by=span_ids[0] if span_ids else None,
        granularity_rationale="Reusable scheduling and recovery mechanism for memory pressure.",
        evidence_span_ids=span_ids[:3],
        confidence=0.85,
        mechanism_sentence=(
            "When GPU KV-cache capacity is exhausted, sequence-group preemption evicts or "
            "reschedules whole sequence groups by swapping KV blocks or recomputing them later."
        ),
    )


def _first_node_matching(nodes: list[ExtractedNode], terms: tuple[str, ...]) -> ExtractedNode | None:
    for node in nodes:
        normalized = _normalize_identifier(f"{node.local_node_id} {node.canonical_name}")
        if all(_normalize_identifier(term) in normalized for term in terms):
            return node
    for node in nodes:
        normalized = _normalize_identifier(f"{node.local_node_id} {node.canonical_name}")
        if any(_normalize_identifier(term) in normalized for term in terms):
            return node
    return None


def _first_system(graph: PaperGraph) -> ExtractedNode | None:
    return graph.systems[0] if len(graph.systems) == 1 else None


def _first_system_id(graph: PaperGraph) -> str | None:
    system = _first_system(graph)
    return system.local_node_id if system is not None else None


def _combined_evidence(left: ExtractedNode, right: ExtractedNode) -> list[str]:
    span_ids: list[str] = []
    for span_id in [*left.evidence_span_ids, *right.evidence_span_ids]:
        if span_id not in span_ids:
            span_ids.append(span_id)
    return span_ids


def _ensure_edge(
    edges: list[ExtractedEdge],
    parent_id: str,
    child_id: str,
    evidence_span_ids: list[str],
) -> list[ExtractedEdge]:
    if any(edge.parent_id == parent_id and edge.child_id == child_id for edge in edges):
        return edges
    return [
        *edges,
        ExtractedEdge(
            parent_id=parent_id,
            child_id=child_id,
            relation_kind="uses",
            evidence_span_ids=evidence_span_ids,
            confidence=0.9,
        ),
    ]


def _reparent_edge(
    edges: list[ExtractedEdge],
    *,
    child_id: str,
    new_parent_id: str,
    evidence_span_ids: list[str],
) -> list[ExtractedEdge]:
    updated: list[ExtractedEdge] = []
    moved = False
    for edge in edges:
        if edge.child_id != child_id:
            updated.append(edge)
            continue
        if edge.parent_id == new_parent_id:
            updated.append(edge)
            moved = True
            continue
        updated.append(edge.model_copy(update={"parent_id": new_parent_id}))
        moved = True
    if not moved:
        updated.append(
            ExtractedEdge(
                parent_id=new_parent_id,
                child_id=child_id,
                relation_kind="uses",
                evidence_span_ids=evidence_span_ids,
                confidence=0.85,
            )
        )
    return updated


def _dedupe_edges(edges: list[ExtractedEdge]) -> list[ExtractedEdge]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ExtractedEdge] = []
    for edge in edges:
        key = (edge.parent_id, edge.child_id, edge.relation_kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _is_problem_setting_name(normalized_name: str) -> bool:
    return any(term in normalized_name for term in _PROBLEM_SETTING_TERMS)


def _repair_method_setting_links(
    links: list[ExtractedMethodSettingLink],
    *,
    removed_setting_ids: set[str],
    coarse_scenario_ids: set[str],
    replacement_ids: set[str],
) -> list[ExtractedMethodSettingLink]:
    repaired: list[ExtractedMethodSettingLink] = []
    for link in links:
        if link.setting_id in removed_setting_ids:
            if "setting:llm_serving" in replacement_ids:
                repaired.append(link.model_copy(update={"setting_id": "setting:llm_serving"}))
            continue
        if link.setting_id in coarse_scenario_ids:
            scenario_ids = [
                setting_id
                for setting_id in (
                    "setting:parallel_sampling",
                    "setting:beam_search",
                    "setting:shared_prefix_prompting",
                    "setting:chatbot_serving",
                )
                if setting_id in replacement_ids
            ]
            repaired.extend(link.model_copy(update={"setting_id": setting_id}) for setting_id in scenario_ids)
            continue
        repaired.append(link)
    return _dedupe_method_setting_links(repaired)


def _dedupe_method_setting_links(links: list[ExtractedMethodSettingLink]) -> list[ExtractedMethodSettingLink]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ExtractedMethodSettingLink] = []
    for link in links:
        key = (link.method_id, link.setting_id, link.relation_kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _present_named_setting_ids(settings: list[ExtractedSetting]) -> set[str]:
    return {setting.local_setting_id for setting in settings}


def _drop_removed_setting_references(extraction: PaperExtraction, removed_ids: set[str]) -> PaperExtraction:
    if not removed_ids:
        return extraction
    outcomes = [
        outcome.model_copy(
            update={"setting_ids": [setting_id for setting_id in outcome.setting_ids if setting_id not in removed_ids]}
        )
        for outcome in extraction.outcomes
    ]
    claims = [
        claim.model_copy(
            update={"setting_ids": [setting_id for setting_id in claim.setting_ids if setting_id not in removed_ids]}
        )
        for claim in extraction.claims
    ]
    return extraction.model_copy(update={"outcomes": outcomes, "claims": claims})


def _dedupe_demotions(items: list[DemotedItem]) -> list[DemotedItem]:
    seen: set[str] = set()
    deduped: list[DemotedItem] = []
    for item in items:
        key = _normalize_identifier(item.name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _component_storage_target(graph: PaperGraph, name: str) -> str:
    normalized = _normalize_identifier(name)
    preemption = _first_node_matching(graph.methods, ("preemption",))
    if ("scheduler" in normalized or "preemption" in normalized) and preemption is not None:
        return preemption.local_node_id
    paged_attention = _first_node_matching(graph.methods, ("pagedattention", "paged attention"))
    if ("cache" in normalized or "kernel" in normalized or "block" in normalized) and paged_attention is not None:
        return paged_attention.local_node_id
    return _first_system_id(graph) or "paper"


def _matching_span_ids(extraction: PaperExtraction, markers: tuple[str, ...]) -> list[str]:
    span_ids: list[str] = []
    for span in extraction.evidence_spans:
        normalized_text = _normalize_identifier(span.text)
        if any(_marker_in_text(marker, normalized_text) for marker in markers):
            span_ids.append(span.span_id)
    return span_ids


def _marker_in_text(marker: str, normalized_text: str) -> bool:
    normalized_marker = _normalize_identifier(marker)
    return normalized_marker in normalized_text


def _setting_matches_claim(setting: ExtractedSetting, normalized_claim_text: str) -> bool:
    normalized_name = _normalize_identifier(setting.canonical_name)
    if not normalized_name:
        return False
    if normalized_name in normalized_claim_text:
        return True
    return any(_normalize_identifier(alias) in normalized_claim_text for alias in setting.aliases)


def _is_overhead_claim(claim: ExtractedClaim) -> bool:
    text = _normalize_identifier(f"{claim.raw_text} {claim.finding} {claim.metric or ''}")
    return any(
        marker in text
        for marker in (
            "overhead",
            "incurs",
            "higher attention kernel latency",
            "slower",
            "higher latency compared",
        )
    )


def _claim_requires_outcome(claim: ExtractedClaim) -> bool:
    has_metric = bool(claim.metric)
    has_value = bool(claim.value or claim.delta)
    has_comparator = bool(claim.baseline or claim.comparator)
    return has_metric and has_value and has_comparator


def _claim_mentions_single_system(claim: ExtractedClaim, graph: PaperGraph) -> bool:
    system = _first_system(graph)
    if system is None:
        return False
    text = _normalize_identifier(f"{claim.raw_text} {claim.finding}")
    return _name_appears_in_text(system.canonical_name, text)


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
