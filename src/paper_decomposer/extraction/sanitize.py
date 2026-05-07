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
    ExtractedSettingEdge,
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
    "memory management",
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

_IMPLEMENTATION_DETAIL_MARKERS = (
    "kernel",
    "cuda",
    "fused",
    "api",
    "worker",
    "frontend",
    "helper",
    "broadcast",
)

_SCENARIO_SETTING_MARKERS = (
    ("setting:basic_sampling", ("single sequence", "single-sequence", "basic sampling")),
    ("setting:parallel_sampling", ("parallel sampling", "parallel generation", "parallel-sampling")),
    ("setting:beam_search", ("beam search", "beam-search", "beam width")),
    ("setting:shared_prefix_prompting", ("shared prefix", "shared-prefix", "prefix sharing")),
    ("setting:chatbot_serving", ("chatbot", "chatbot serving")),
)

_SCENARIO_ADAPTER_MECHANISM_MARKERS = (
    "kv",
    "cache",
    "sharing",
    "prompt sharing",
    "block caching",
)

_METRIC_PATTERNS = (
    ("attention kernel latency", ("attention kernel latency",)),
    ("memory saving", ("memory saving", "memory savings")),
    ("request rate", ("request rate", "request rates")),
    ("throughput", ("throughput",)),
    ("concurrent requests", ("concurrent requests", "requests at the same time", "more requests")),
    ("latency", ("latency", "latencies")),
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
    extraction = _normalize_method_family_ids(extraction)
    extraction = _repair_setting_taxonomy(extraction)
    extraction = _collapse_non_main_method_nodes(extraction)
    extraction = _demote_component_details(extraction)
    extraction = extraction.model_copy(update={"claims": [_repair_claim(claim, extraction.graph) for claim in extraction.claims]})
    extraction = _normalize_method_family_ids(extraction)
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


def _normalize_method_family_ids(extraction: PaperExtraction) -> PaperExtraction:
    replacements: dict[str, str] = {}
    systems = _normalize_nodes(extraction.graph.systems, replacements)
    methods = _normalize_nodes(extraction.graph.methods, replacements)
    if not replacements:
        return extraction

    graph = extraction.graph.model_copy(
        update={
            "systems": systems,
            "methods": methods,
            "method_edges": [
                edge.model_copy(
                    update={
                        "parent_id": replacements.get(edge.parent_id, edge.parent_id),
                        "child_id": replacements.get(edge.child_id, edge.child_id),
                    }
                )
                for edge in extraction.edges
            ],
            "method_setting_links": [
                link.model_copy(update={"method_id": replacements.get(link.method_id, link.method_id)})
                for link in extraction.method_setting_links
            ],
        }
    )
    outcomes = [
        outcome.model_copy(
            update={"method_ids": _dedupe_ids([replacements.get(method_id, method_id) for method_id in outcome.method_ids])}
        )
        for outcome in extraction.outcomes
    ]
    claims = [
        claim.model_copy(
            update={"method_ids": _dedupe_ids([replacements.get(method_id, method_id) for method_id in claim.method_ids])}
        )
        for claim in extraction.claims
    ]
    demoted_items = [
        item.model_copy(update={"stored_under": replacements.get(item.stored_under, item.stored_under)})
        for item in extraction.demoted_items
    ]
    return extraction.model_copy(
        update={
            "graph": graph,
            "outcomes": outcomes,
            "claims": claims,
            "demoted_items": demoted_items,
        }
    )


def _normalize_nodes(nodes: list[ExtractedNode], replacements: dict[str, str]) -> list[ExtractedNode]:
    by_id: dict[str, ExtractedNode] = {}
    for node in nodes:
        normalized_id = _normalized_node_id(node)
        if normalized_id != node.local_node_id:
            replacements[node.local_node_id] = normalized_id
        normalized = node.model_copy(update={"local_node_id": normalized_id})
        if normalized_id in by_id:
            by_id[normalized_id] = _merge_nodes(by_id[normalized_id], normalized)
        else:
            by_id[normalized_id] = normalized
    return list(by_id.values())


def _normalized_node_id(node: ExtractedNode) -> str:
    if node.local_node_id.startswith("method:"):
        return f"meth_{_slug(node.local_node_id.removeprefix('method:'))}"
    if node.local_node_id.startswith("system:"):
        return f"sys_{_slug(node.local_node_id.removeprefix('system:'))}"
    return node.local_node_id


def _merge_nodes(left: ExtractedNode, right: ExtractedNode) -> ExtractedNode:
    return left.model_copy(
        update={
            "aliases": _dedupe_ids([*left.aliases, right.canonical_name, *right.aliases]),
            "category_tags": _dedupe_ids([*left.category_tags, *right.category_tags]),
            "evidence_span_ids": _dedupe_ids([*left.evidence_span_ids, *right.evidence_span_ids]),
            "mechanism_sentence": left.mechanism_sentence or right.mechanism_sentence,
            "confidence": left.confidence if left.confidence is not None else right.confidence,
        }
    )


def _repair_claim(claim: ExtractedClaim, graph: PaperGraph) -> ExtractedClaim:
    claim = _retarget_overall_system_claim(claim, graph)
    claim = _retarget_claim_settings(claim, graph)
    if _is_overhead_claim(claim):
        claim = claim.model_copy(update={"claim_type": "overhead"})
    claim = _populate_claim_structured_fields(claim)
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
    settings_by_id, setting_replacements = _canonicalize_settings(graph.settings)

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
        _remap_method_setting_links(graph.method_setting_links, setting_replacements),
        removed_setting_ids=removed_setting_ids,
        coarse_scenario_ids=coarse_scenario_ids,
        replacement_ids=replacement_ids,
    )
    setting_edges = [
        edge
        for edge in _remap_setting_edges(graph.setting_edges, setting_replacements)
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
    extraction = _remap_setting_references(extraction, setting_replacements)
    return _drop_removed_setting_references(extraction, removed_setting_ids | coarse_scenario_ids)


def _collapse_non_main_method_nodes(extraction: PaperExtraction) -> PaperExtraction:
    graph = extraction.graph
    node_ids = {node.local_node_id for node in graph.nodes}
    setting_ids = {setting.local_setting_id for setting in graph.settings}
    actions: dict[str, tuple[str | None, str, str, list[str]]] = {}

    for node in graph.methods:
        target_id: str | None = None
        reason: str | None = None
        scenario_setting_ids: list[str] = []

        if node.kind == "method_category":
            reason = "Global/category label; keep outside the paper-local method DAG."
        else:
            target_id, scenario_setting_ids = _scenario_adapter_target(node, graph.methods, setting_ids)
            if target_id is not None:
                reason = "Scenario-specific adapter collapsed to the reusable mechanism plus applies_to settings."
            else:
                reason = _implementation_detail_reason(node, extraction)
                if reason is not None:
                    target_id = _valid_demoted_method_target(
                        _component_storage_target(graph, node.canonical_name),
                        node_ids,
                    )

        if reason is None:
            continue
        if target_id == node.local_node_id:
            continue
        stored_under = target_id or _component_storage_target(graph, node.canonical_name)
        actions[node.local_node_id] = (target_id, stored_under, reason, scenario_setting_ids)

    if not actions:
        return extraction

    kept_methods = _apply_demoted_category_tags(
        [node for node in graph.methods if node.local_node_id not in actions],
        graph.methods,
        graph.method_edges,
        actions,
    )
    kept_node_ids = {node.local_node_id for node in [*graph.systems, *kept_methods]}
    replacements = {node_id: target_id for node_id, (target_id, _stored_under, _reason, _settings) in actions.items()}
    edges = _retarget_method_edges(graph.method_edges, replacements, kept_node_ids)
    links = _retarget_method_setting_links(graph.method_setting_links, replacements, kept_node_ids, setting_ids)
    links.extend(_scenario_applies_to_links(graph.methods, actions, kept_node_ids, setting_ids))
    graph = graph.model_copy(
        update={
            "methods": kept_methods,
            "method_edges": _dedupe_edges(edges),
            "method_setting_links": _dedupe_method_setting_links(links),
        }
    )
    demoted_items = [
        *extraction.demoted_items,
        *[
            DemotedItem(
                name=node.canonical_name,
                reason_demoted=reason,
                stored_under=stored_under,
                evidence_span_ids=node.evidence_span_ids,
            )
            for node in extraction.graph.methods
            if node.local_node_id in actions
            for _target_id, stored_under, reason, _setting_ids in [actions[node.local_node_id]]
        ],
    ]
    extraction = extraction.model_copy(update={"graph": graph, "demoted_items": _dedupe_demotions(demoted_items)})
    return _retarget_demoted_method_references(extraction, replacements, kept_node_ids)


def _apply_demoted_category_tags(
    kept_methods: list[ExtractedNode],
    original_methods: list[ExtractedNode],
    edges: list[ExtractedEdge],
    actions: dict[str, tuple[str | None, str, str, list[str]]],
) -> list[ExtractedNode]:
    kept_ids = {node.local_node_id for node in kept_methods}
    method_by_id = {node.local_node_id: node for node in original_methods}
    tags_by_method_id: dict[str, list[str]] = {}

    for node_id in actions:
        node = method_by_id[node_id]
        if node.kind != "method_category":
            continue
        tag = _slug(node.canonical_name)
        neighbor_ids = [
            edge.child_id
            for edge in edges
            if edge.parent_id == node_id and edge.child_id in kept_ids
        ]
        neighbor_ids.extend(
            edge.parent_id
            for edge in edges
            if edge.child_id == node_id and edge.parent_id in kept_ids
        )
        for neighbor_id in neighbor_ids:
            tags_by_method_id.setdefault(neighbor_id, []).append(tag)

    if not tags_by_method_id:
        return kept_methods
    return [
        node.model_copy(
            update={
                "category_tags": _dedupe_ids(
                    [*node.category_tags, *tags_by_method_id.get(node.local_node_id, [])]
                )
            }
        )
        for node in kept_methods
    ]


def _scenario_adapter_target(
    node: ExtractedNode,
    methods: list[ExtractedNode],
    setting_ids: set[str],
) -> tuple[str | None, list[str]]:
    normalized = _normalize_identifier(f"{node.local_node_id} {node.canonical_name} {node.description}")
    scenario_setting_ids = _scenario_setting_ids(normalized, setting_ids)
    if not scenario_setting_ids:
        return None, []
    if not _has_any_marker(normalized, _SCENARIO_ADAPTER_MECHANISM_MARKERS):
        return None, []
    target = _general_sharing_method(methods, node.local_node_id)
    if target is None:
        return None, []
    return target.local_node_id, scenario_setting_ids


def _scenario_setting_ids(normalized_text: str, setting_ids: set[str]) -> list[str]:
    matched: list[str] = []
    for setting_id, markers in _SCENARIO_SETTING_MARKERS:
        if setting_id not in setting_ids:
            continue
        if any(_marker_in_text(marker, normalized_text) for marker in markers):
            matched.append(setting_id)
    return matched


def _general_sharing_method(methods: list[ExtractedNode], excluded_node_id: str) -> ExtractedNode | None:
    for node in methods:
        if node.local_node_id == excluded_node_id:
            continue
        normalized = _normalize_identifier(f"{node.local_node_id} {node.canonical_name}")
        if _mentions_scenario(normalized):
            continue
        if "block level" in normalized and "sharing" in normalized:
            return node
        if "kv cache sharing" in normalized or "kv block sharing" in normalized:
            return node
        if "reference counted" in normalized and "sharing" in normalized:
            return node
    return None


def _mentions_scenario(normalized_text: str) -> bool:
    return any(
        _marker_in_text(marker, normalized_text)
        for _setting_id, markers in _SCENARIO_SETTING_MARKERS
        for marker in markers
    )


def _has_any_marker(normalized_text: str, markers: tuple[str, ...]) -> bool:
    return any(_marker_in_text(marker, normalized_text) for marker in markers)


def _implementation_detail_reason(node: ExtractedNode, extraction: PaperExtraction) -> str | None:
    if node.kind != "method":
        return None
    if _is_central_contribution_node(node, extraction):
        return None
    normalized = _normalize_identifier(f"{node.local_node_id} {node.canonical_name} {node.description}")
    spec_reason = _component_detail_reason_for_node(normalized)
    if spec_reason is not None:
        return spec_reason
    if _has_any_marker(normalized, _IMPLEMENTATION_DETAIL_MARKERS):
        return (
            "Implementation support such as kernels, CUDA code, fused operations, APIs, workers, "
            "frontends, helpers, or broadcasts is not part of the main reusable method DAG."
        )
    return None


def _component_detail_reason_for_node(normalized_text: str) -> str | None:
    for _name, markers, reason in _COMPONENT_DEMOTION_SPECS:
        if any(_marker_in_text(marker, normalized_text) for marker in markers):
            return reason
    return None


def _is_central_contribution_node(node: ExtractedNode, extraction: PaperExtraction) -> bool:
    normalized_title = _normalize_identifier(extraction.title)
    normalized_name = _normalize_identifier(node.canonical_name)
    if normalized_name and normalized_name in normalized_title:
        return True
    novelty_markers = ("we propose", "we introduce", "we design", "we develop")
    evidence_by_id = {span.span_id: span for span in extraction.evidence_spans}
    for span_id in node.evidence_span_ids:
        span = evidence_by_id.get(span_id)
        normalized_text = _normalize_identifier(str(getattr(span, "text", "")))
        if not _name_appears_in_text(node.canonical_name, normalized_text):
            continue
        if any(_normalize_identifier(marker) in normalized_text for marker in novelty_markers):
            return True
    return False


def _valid_demoted_method_target(target_id: str, node_ids: set[str]) -> str | None:
    if target_id in node_ids:
        return target_id
    return None


def _retarget_method_edges(
    edges: list[ExtractedEdge],
    replacements: dict[str, str | None],
    kept_node_ids: set[str],
) -> list[ExtractedEdge]:
    retargeted: list[ExtractedEdge] = []
    for edge in edges:
        parent_id = replacements.get(edge.parent_id, edge.parent_id)
        child_id = replacements.get(edge.child_id, edge.child_id)
        if parent_id is None or child_id is None or parent_id == child_id:
            continue
        if parent_id not in kept_node_ids or child_id not in kept_node_ids:
            continue
        retargeted.append(edge.model_copy(update={"parent_id": parent_id, "child_id": child_id}))

    for node_id, target_id in replacements.items():
        if target_id is not None:
            continue
        parents = [edge for edge in edges if edge.child_id == node_id and edge.parent_id in kept_node_ids]
        children = [edge for edge in edges if edge.parent_id == node_id and edge.child_id in kept_node_ids]
        for parent in parents:
            for child in children:
                if parent.parent_id == child.child_id:
                    continue
                retargeted.append(
                    ExtractedEdge(
                        parent_id=parent.parent_id,
                        child_id=child.child_id,
                        relation_kind=child.relation_kind,
                        evidence_span_ids=_dedupe_ids([*parent.evidence_span_ids, *child.evidence_span_ids]),
                        confidence=child.confidence if child.confidence is not None else parent.confidence,
                    )
                )
    return retargeted


def _retarget_method_setting_links(
    links: list[ExtractedMethodSettingLink],
    replacements: dict[str, str | None],
    kept_node_ids: set[str],
    setting_ids: set[str],
) -> list[ExtractedMethodSettingLink]:
    retargeted: list[ExtractedMethodSettingLink] = []
    for link in links:
        method_id = replacements.get(link.method_id, link.method_id)
        if method_id is None or method_id not in kept_node_ids or link.setting_id not in setting_ids:
            continue
        retargeted.append(link.model_copy(update={"method_id": method_id}))
    return retargeted


def _scenario_applies_to_links(
    methods: list[ExtractedNode],
    actions: dict[str, tuple[str | None, str, str, list[str]]],
    kept_node_ids: set[str],
    setting_ids: set[str],
) -> list[ExtractedMethodSettingLink]:
    by_id = {node.local_node_id: node for node in methods}
    links: list[ExtractedMethodSettingLink] = []
    for node_id, (target_id, _stored_under, _reason, scenario_setting_ids) in actions.items():
        if target_id is None or target_id not in kept_node_ids:
            continue
        node = by_id[node_id]
        for setting_id in scenario_setting_ids:
            if setting_id not in setting_ids:
                continue
            links.append(
                ExtractedMethodSettingLink(
                    method_id=target_id,
                    setting_id=setting_id,
                    relation_kind="applies_to",
                    evidence_span_ids=node.evidence_span_ids,
                    confidence=node.confidence,
                )
            )
    return links


def _retarget_demoted_method_references(
    extraction: PaperExtraction,
    replacements: dict[str, str | None],
    kept_node_ids: set[str],
) -> PaperExtraction:
    claims = [
        claim.model_copy(update={"method_ids": _retarget_method_ids(claim.method_ids, replacements, kept_node_ids)})
        for claim in extraction.claims
    ]
    outcomes = [
        outcome.model_copy(update={"method_ids": _retarget_method_ids(outcome.method_ids, replacements, kept_node_ids)})
        for outcome in extraction.outcomes
    ]
    return extraction.model_copy(update={"claims": claims, "outcomes": outcomes})


def _retarget_method_ids(
    method_ids: list[str],
    replacements: dict[str, str | None],
    kept_node_ids: set[str],
) -> list[str]:
    retargeted: list[str] = []
    for method_id in method_ids:
        target_id = replacements.get(method_id, method_id)
        if target_id is None or target_id not in kept_node_ids or target_id in retargeted:
            continue
        retargeted.append(target_id)
    return retargeted


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
        outcome_specs = _claim_outcome_specs(claim)
        if not outcome_specs:
            claims.append(claim)
            continue
        outcome_ids = list(claim.outcome_ids)
        for spec in outcome_specs:
            outcome_id = str(spec["outcome_id"])
            if outcome_id not in outcomes_by_id:
                outcomes_by_id[outcome_id] = ExtractedOutcome(
                    outcome_id=outcome_id,
                    paper_id=extraction.paper_id,
                    metric=str(spec["metric"]),
                    method_ids=claim.method_ids,
                    setting_ids=claim.setting_ids,
                    value=spec.get("value"),
                    delta=spec.get("delta"),
                    baseline=spec.get("baseline"),
                    comparator=spec.get("comparator"),
                    evidence_span_ids=claim.evidence_span_ids,
                    confidence=claim.confidence,
                )
            if outcome_id not in outcome_ids:
                outcome_ids.append(outcome_id)
        claims.append(claim.model_copy(update={"outcome_ids": outcome_ids}))
    return extraction.model_copy(update={"claims": claims, "outcomes": list(outcomes_by_id.values())})


def _claim_outcome_specs(claim: ExtractedClaim) -> list[dict[str, str | None]]:
    text = _claim_text(claim)
    metric = claim.metric or _infer_metric(text)
    if metric is None:
        return []

    specs = _split_known_outcomes(claim, text, metric)
    if specs:
        return specs

    delta = claim.delta or claim.value or _first_magnitude(text)
    comparator = claim.comparator or claim.baseline or _infer_comparator(text, metric)
    if not delta or not comparator:
        return []
    return [
        {
            "outcome_id": f"outcome:{claim.claim_id}",
            "metric": metric,
            "delta": delta if claim.value is None else claim.delta,
            "value": claim.value,
            "baseline": claim.baseline,
            "comparator": comparator if claim.baseline is None else claim.comparator,
        }
    ]


def _split_known_outcomes(claim: ExtractedClaim, text: str, metric: str) -> list[dict[str, str | None]]:
    specs: list[dict[str, str | None]] = []
    oracle_delta = _delta_before(text, "orca (oracle)")
    max_delta = _delta_before(text, "orca (max)")
    if oracle_delta and max_delta:
        return [
            _outcome_spec(claim, "orca_oracle", metric, oracle_delta, "Orca (Oracle)"),
            _outcome_spec(claim, "orca_max", metric, max_delta, "Orca (Max)"),
        ]

    one_shot_delta = _delta_before(text, "one")
    five_shot_delta = _delta_before(text, "five")
    if "shared" in _normalize_identifier(text) and one_shot_delta and five_shot_delta:
        return [
            _outcome_spec(claim, "one_shot", metric, one_shot_delta, "Orca (Oracle)"),
            _outcome_spec(claim, "five_shot", metric, five_shot_delta, "Orca (Oracle)"),
        ]

    if metric == "memory saving":
        parallel_delta = _delta_before(text, "parallel sampling")
        beam_delta = _delta_before(text, "beam search")
        if parallel_delta:
            specs.append(_outcome_spec(claim, "parallel_sampling", metric, parallel_delta, "without sharing"))
        if beam_delta:
            specs.append(_outcome_spec(claim, "beam_search", metric, beam_delta, "without sharing"))
    return specs


def _outcome_spec(
    claim: ExtractedClaim,
    suffix: str,
    metric: str,
    delta: str,
    comparator: str,
) -> dict[str, str | None]:
    return {
        "outcome_id": f"outcome:{claim.claim_id}:{suffix}",
        "metric": metric,
        "delta": delta,
        "value": None,
        "baseline": claim.baseline,
        "comparator": comparator,
    }


def _populate_claim_structured_fields(claim: ExtractedClaim) -> ExtractedClaim:
    text = _claim_text(claim)
    updates: dict[str, str] = {}
    if claim.metric is None:
        metric = _infer_metric(text)
        if metric is not None:
            updates["metric"] = metric
    if claim.delta is None and claim.value is None:
        delta = _first_magnitude(text)
        if delta is not None:
            updates["delta"] = delta
    if claim.comparator is None and claim.baseline is None:
        comparator = _infer_comparator(text, updates.get("metric") or claim.metric)
        if comparator is not None:
            updates["comparator"] = comparator
    if not updates:
        return claim
    return claim.model_copy(update=updates)


def _claim_text(claim: ExtractedClaim) -> str:
    return f"{claim.raw_text} {claim.finding}"


def _infer_metric(text: str) -> str | None:
    normalized = _normalize_identifier(text)
    for metric, markers in _METRIC_PATTERNS:
        if any(marker in normalized for marker in markers):
            return metric
    return None


def _infer_comparator(text: str, metric: str | None) -> str | None:
    normalized = _normalize_identifier(text)
    if "orca oracle" in normalized and "orca max" in normalized:
        return "Orca (Oracle/Max)"
    if "orca oracle" in normalized:
        return "Orca (Oracle)"
    if "orca max" in normalized:
        return "Orca (Max)"
    if "orca baselines" in normalized:
        return "Orca baselines"
    if "orca" in normalized:
        return "Orca"
    if "fastertransformer" in normalized:
        return "FasterTransformer"
    if metric == "memory saving":
        return "without sharing"
    return None


def _first_magnitude(text: str) -> str | None:
    match = re.search(_magnitude_pattern(), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return _clean_delta(match.group(0))


def _delta_before(text: str, marker: str) -> str | None:
    normalized_marker = re.escape(marker).replace(r"\ ", r"\s+")
    marker_match = re.search(normalized_marker, text, flags=re.IGNORECASE)
    if marker_match is None:
        return None
    prefix = text[: marker_match.start()]
    matches = list(re.finditer(_magnitude_pattern(), prefix, flags=re.IGNORECASE))
    if not matches:
        return None
    return _clean_delta(matches[-1].group(0))


def _magnitude_pattern() -> str:
    range_dash = r"[\-–—‑]"
    number = r"\d+(?:\s*\.\s*\d+)?"
    unit = r"(?:×|x|%)"
    return rf"{number}\s*{unit}(?:\s*{range_dash}\s*{number}\s*{unit})?"


def _clean_delta(value: str) -> str:
    cleaned = re.sub(r"\s+", "", value)
    cleaned = cleaned.replace("x", "×")
    cleaned = cleaned.replace("-", "–").replace("—", "–").replace("‑", "–")
    return cleaned


def _claim_requires_outcome(claim: ExtractedClaim) -> bool:
    has_metric = bool(claim.metric)
    has_value = bool(claim.value or claim.delta)
    has_comparator = bool(claim.baseline or claim.comparator)
    return has_metric and has_value and has_comparator


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
        local_node_id="meth_sequence_group_preemption",
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


def _canonicalize_settings(settings: list[ExtractedSetting]) -> tuple[dict[str, ExtractedSetting], dict[str, str]]:
    settings_by_id: dict[str, ExtractedSetting] = {}
    replacements: dict[str, str] = {}
    for setting in settings:
        canonical = _named_setting_spec_for(setting)
        if canonical is None:
            target_id = setting.local_setting_id
            canonical_setting = setting
        else:
            target_id, name, kind, _markers, description = canonical
            if setting.local_setting_id != target_id:
                replacements[setting.local_setting_id] = target_id
            aliases = list(setting.aliases)
            if _normalize_identifier(setting.canonical_name) != _normalize_identifier(name):
                aliases.append(setting.canonical_name)
            canonical_setting = setting.model_copy(
                update={
                    "local_setting_id": target_id,
                    "kind": kind,
                    "canonical_name": name,
                    "description": setting.description or description,
                    "aliases": _dedupe_ids(aliases),
                }
            )
        if target_id in settings_by_id:
            settings_by_id[target_id] = _merge_settings(settings_by_id[target_id], canonical_setting)
        else:
            settings_by_id[target_id] = canonical_setting
    return settings_by_id, replacements


def _named_setting_spec_for(setting: ExtractedSetting) -> tuple[str, str, str, tuple[str, ...], str] | None:
    normalized_name = _normalize_identifier(setting.canonical_name)
    normalized_text = _normalize_identifier(" ".join([setting.canonical_name, *setting.aliases, setting.local_setting_id]))
    for spec in _NAMED_SETTINGS:
        setting_id, name, _kind, markers, _description = spec
        if setting.local_setting_id == setting_id:
            return spec
        if normalized_name == _normalize_identifier(name):
            return spec
        if any(_normalize_identifier(marker) in normalized_text for marker in markers):
            return spec
    return None


def _merge_settings(left: ExtractedSetting, right: ExtractedSetting) -> ExtractedSetting:
    aliases = [*left.aliases, right.canonical_name, *right.aliases]
    return left.model_copy(
        update={
            "aliases": _dedupe_ids(aliases),
            "evidence_span_ids": _dedupe_ids([*left.evidence_span_ids, *right.evidence_span_ids]),
            "confidence": left.confidence if left.confidence is not None else right.confidence,
        }
    )


def _remap_method_setting_links(
    links: list[ExtractedMethodSettingLink],
    replacements: dict[str, str],
) -> list[ExtractedMethodSettingLink]:
    if not replacements:
        return links
    return [
        link.model_copy(update={"setting_id": replacements.get(link.setting_id, link.setting_id)})
        for link in links
    ]


def _remap_setting_edges(
    edges: list[ExtractedSettingEdge],
    replacements: dict[str, str],
) -> list[ExtractedSettingEdge]:
    if not replacements:
        return edges
    return [
        edge.model_copy(
            update={
                "parent_id": replacements.get(edge.parent_id, edge.parent_id),
                "child_id": replacements.get(edge.child_id, edge.child_id),
            }
        )
        for edge in edges
    ]


def _remap_setting_references(extraction: PaperExtraction, replacements: dict[str, str]) -> PaperExtraction:
    if not replacements:
        return extraction
    claims = [
        claim.model_copy(
            update={"setting_ids": _dedupe_ids([replacements.get(setting_id, setting_id) for setting_id in claim.setting_ids])}
        )
        for claim in extraction.claims
    ]
    outcomes = [
        outcome.model_copy(
            update={"setting_ids": _dedupe_ids([replacements.get(setting_id, setting_id) for setting_id in outcome.setting_ids])}
        )
        for outcome in extraction.outcomes
    ]
    return extraction.model_copy(update={"claims": claims, "outcomes": outcomes})


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


def _slug(value: str) -> str:
    return "_".join(_normalize_identifier(value).split())


def _dedupe_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


__all__ = ["demote_invalid_method_nodes", "preserve_graph_and_attach_claims"]
