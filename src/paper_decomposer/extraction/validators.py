from __future__ import annotations

import re

from .contracts import (
    ExtractionCaps,
    ExtractionValidationError,
    ExtractionValidationReport,
    PaperExtraction,
    ValidationSeverity,
)

_GENERIC_METHOD_NAMES = {
    "scheduling",
    "memory management",
    "optimization",
    "parallelism",
    "batching",
    "caching",
    "evaluation",
    "implementation",
    "architecture",
    "system design",
}

_GENERIC_SECTION_HEADINGS = _GENERIC_METHOD_NAMES | {
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
}

_CONCRETE_REUSABLE_METHOD_NAMES = {
    "block wise kv cache address translation",
    "on demand kv block allocation",
    "block level kv cache sharing",
    "kv block copy on write",
    "sequence group preemption",
    "kv cache swapping",
    "kv cache recomputation",
}


def validate_extraction(
    extraction: PaperExtraction,
    *,
    caps: ExtractionCaps | None = None,
    require_numeric_grounding: bool = False,
) -> ExtractionValidationReport:
    caps = caps or ExtractionCaps()
    errors: list[ExtractionValidationError] = []

    evidence_by_id = {span.span_id: span for span in extraction.evidence_spans}
    node_by_id = {node.local_node_id: node for node in extraction.nodes}
    setting_by_id = {setting.local_setting_id: setting for setting in extraction.settings}
    outcome_by_id = {outcome.outcome_id: outcome for outcome in extraction.outcomes}
    section_titles = {_normalize(span.section_title) for span in extraction.evidence_spans}

    errors.extend(_graph_shape_errors(extraction))

    for node in extraction.nodes:
        errors.extend(_missing_evidence_errors("node", node.local_node_id, node.evidence_span_ids, evidence_by_id))
        if node.kind == "method" and not _has_mechanism_sentence(node.mechanism_sentence):
            errors.append(
                _error(
                    "method_missing_mechanism_sentence",
                    "Method node must state inputs, outputs, and operative move.",
                    object_kind="node",
                    object_id=node.local_node_id,
                )
            )
        normalized_name = _normalize(node.canonical_name)
        if normalized_name in section_titles and normalized_name in _GENERIC_SECTION_HEADINGS:
            errors.append(
                _error(
                    "section_heading_promoted",
                    "Paper section heading was promoted as a method-family node.",
                    object_kind="node",
                    object_id=node.local_node_id,
                )
            )
        if node.kind == "method" and _normalize(node.canonical_name) in _GENERIC_METHOD_NAMES:
            errors.append(
                _warning(
                    "generic_method_name",
                    "Generic method category needs explicit justification before persistence.",
                    object_kind="node",
                    object_id=node.local_node_id,
                )
            )

    for setting in extraction.settings:
        errors.extend(
            _missing_evidence_errors("setting", setting.local_setting_id, setting.evidence_span_ids, evidence_by_id)
        )
        if _is_problem_setting(setting.canonical_name, setting.kind):
            errors.append(
                _warning(
                    "problem_stored_as_setting",
                    "Problem/challenge context should not be stored as an application, task, or workload setting.",
                    object_kind="setting",
                    object_id=setting.local_setting_id,
                    evidence_span_ids=setting.evidence_span_ids,
                )
            )
        if _is_coarse_scenario_setting(setting.canonical_name, extraction.evidence_spans):
            errors.append(
                _warning(
                    "scenario_bucket_setting",
                    "Generic scenario bucket should be split into named task/application settings.",
                    object_kind="setting",
                    object_id=setting.local_setting_id,
                    evidence_span_ids=setting.evidence_span_ids,
                )
            )

    for edge in extraction.edges:
        if edge.parent_id not in node_by_id or edge.child_id not in node_by_id:
            errors.append(
                _error(
                    "edge_endpoint_missing",
                    "Method edge endpoint does not exist in method-family nodes.",
                    object_kind="edge",
                    object_id=f"{edge.parent_id}->{edge.child_id}",
                    evidence_span_ids=edge.evidence_span_ids,
                )
            )
        errors.extend(_missing_evidence_errors("edge", f"{edge.parent_id}->{edge.child_id}", edge.evidence_span_ids, evidence_by_id))
        if _is_explicit_zero(edge.confidence) and edge.evidence_span_ids:
            errors.append(
                _warning(
                    "zero_confidence_default",
                    "Grounded edge has confidence=0.0; use null for unscored confidence.",
                    object_kind="edge",
                    object_id=f"{edge.parent_id}->{edge.child_id}",
                    evidence_span_ids=edge.evidence_span_ids,
                )
            )

    for edge in extraction.setting_edges:
        if edge.parent_id not in setting_by_id or edge.child_id not in setting_by_id:
            errors.append(
                _error(
                    "setting_edge_endpoint_missing",
                    "Setting edge endpoint does not exist in settings.",
                    object_kind="setting_edge",
                    object_id=f"{edge.parent_id}->{edge.child_id}",
                    evidence_span_ids=edge.evidence_span_ids,
                )
            )
        errors.extend(
            _missing_evidence_errors(
                "setting_edge",
                f"{edge.parent_id}->{edge.child_id}",
                edge.evidence_span_ids,
                evidence_by_id,
            )
        )

    for link in extraction.method_setting_links:
        if link.method_id not in node_by_id or link.setting_id not in setting_by_id:
            errors.append(
                _error(
                    "method_setting_endpoint_missing",
                    "Method-setting link endpoint does not exist.",
                    object_kind="method_setting_link",
                    object_id=f"{link.method_id}->{link.setting_id}",
                    evidence_span_ids=link.evidence_span_ids,
                )
            )
        errors.extend(
            _missing_evidence_errors(
                "method_setting_link",
                f"{link.method_id}->{link.setting_id}",
                link.evidence_span_ids,
                evidence_by_id,
            )
        )
        if _is_explicit_zero(link.confidence) and link.evidence_span_ids:
            errors.append(
                _warning(
                    "zero_confidence_default",
                    "Grounded method-setting link has confidence=0.0; use null for unscored confidence.",
                    object_kind="method_setting_link",
                    object_id=f"{link.method_id}->{link.setting_id}",
                    evidence_span_ids=link.evidence_span_ids,
                )
            )

    for outcome in extraction.outcomes:
        if outcome.paper_id != extraction.paper_id:
            errors.append(
                _error(
                    "outcome_paper_mismatch",
                    "Outcome paper_id must match the extraction paper_id.",
                    object_kind="outcome",
                    object_id=outcome.outcome_id,
                )
            )
        for method_id in outcome.method_ids:
            if method_id not in node_by_id:
                errors.append(_missing_target_error("outcome_method_missing", "outcome", outcome.outcome_id, method_id))
        for setting_id in outcome.setting_ids:
            if setting_id not in setting_by_id:
                errors.append(_missing_target_error("outcome_setting_missing", "outcome", outcome.outcome_id, setting_id))
        errors.extend(_missing_evidence_errors("outcome", outcome.outcome_id, outcome.evidence_span_ids, evidence_by_id))
        errors.extend(
            _numeric_grounding_findings(
                "outcome",
                outcome.outcome_id,
                [outcome.value, outcome.delta],
                outcome.evidence_span_ids,
                evidence_by_id,
                require_numeric_grounding=require_numeric_grounding,
            )
        )

    for claim in extraction.claims:
        if claim.paper_id != extraction.paper_id:
            errors.append(
                _error(
                    "claim_paper_mismatch",
                    "Claim paper_id must match the extraction paper_id.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                )
            )
        for method_id in claim.method_ids:
            if method_id not in node_by_id:
                errors.append(_missing_target_error("claim_method_missing", "claim", claim.claim_id, method_id))
        for setting_id in claim.setting_ids:
            if setting_id not in setting_by_id:
                errors.append(_missing_target_error("claim_setting_missing", "claim", claim.claim_id, setting_id))
        for outcome_id in claim.outcome_ids:
            if outcome_id not in outcome_by_id:
                errors.append(_missing_target_error("claim_outcome_missing", "claim", claim.claim_id, outcome_id))
        errors.extend(_missing_evidence_errors("claim", claim.claim_id, claim.evidence_span_ids, evidence_by_id))
        errors.extend(
            _numeric_grounding_findings(
                "claim",
                claim.claim_id,
                [claim.value, claim.delta],
                claim.evidence_span_ids,
                evidence_by_id,
                require_numeric_grounding=require_numeric_grounding,
            )
        )
        if not _appears_in_evidence(claim.raw_text, claim.evidence_span_ids, evidence_by_id):
            errors.append(
                _warning(
                    "claim_raw_text_not_copied",
                    "Claim raw_text should copy source text from evidence when possible.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )
        if _claim_needs_structured_fields(claim) and not any(
            [claim.metric, claim.value, claim.delta, claim.baseline, claim.comparator]
        ):
            errors.append(
                _warning(
                    "claim_structured_fields_missing",
                    "Comparison claim should expose metric, delta/value, baseline, or comparator fields.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )
        if _claim_requires_outcome(claim):
            errors.append(
                _warning(
                    "structured_claim_without_outcome",
                    "Claim has metric, value/delta, and comparator/baseline but is not linked to an explicit outcome row.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )
        if _overall_system_claim_attached_too_low(claim, extraction):
            errors.append(
                _warning(
                    "system_claim_attached_to_method_only",
                    "End-to-end system claim should attach to the system node, not only to a supporting method.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )
        if not (claim.method_ids or claim.setting_ids or claim.outcome_ids):
            errors.append(
                _error(
                    "claims_unattached",
                    "Claim must attach to at least one method-family node, setting, or outcome.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )
        if (
            _is_explicit_zero(claim.confidence)
            and claim.evidence_span_ids
            and _appears_in_evidence(claim.raw_text, claim.evidence_span_ids, evidence_by_id)
        ):
            errors.append(
                _warning(
                    "zero_confidence_default",
                    "Grounded claim has confidence=0.0; use null for unscored confidence.",
                    object_kind="claim",
                    object_id=claim.claim_id,
                    evidence_span_ids=claim.evidence_span_ids,
                )
            )

    promoted_names = {_normalize(node.canonical_name) for node in extraction.nodes}
    promoted_names.update(_normalize(setting.canonical_name) for setting in extraction.settings)
    for item in extraction.demoted_items:
        errors.extend(_missing_evidence_errors("demoted_item", item.name, item.evidence_span_ids, evidence_by_id))
        if _normalize(item.name) in promoted_names:
            errors.append(
                _error(
                    "demoted_item_promoted",
                    "Demoted item reappears as a first-class node or setting.",
                    object_kind="demoted_item",
                    object_id=item.name,
                    evidence_span_ids=item.evidence_span_ids,
                )
            )
        if _is_concrete_reusable_method_demoted_for_missing_mechanism(item.name, item.reason_demoted):
            errors.append(
                _error(
                    "concrete_method_demoted_for_missing_mechanism",
                    "Concrete reusable method was demoted for a missing mechanism sentence; synthesize the mechanism from evidence instead.",
                    object_kind="demoted_item",
                    object_id=item.name,
                    evidence_span_ids=item.evidence_span_ids,
                )
            )

    errors.extend(_cap_warnings(extraction, caps))
    errors.extend(_evidence_span_warnings(extraction))
    errors.extend(_graph_quality_warnings(extraction, evidence_by_id))
    return ExtractionValidationReport(errors=errors)


def _graph_shape_errors(extraction: PaperExtraction) -> list[ExtractionValidationError]:
    if not extraction.evidence_spans:
        return []

    errors: list[ExtractionValidationError] = []
    graph_node_count = len(extraction.nodes) + len(extraction.settings)
    if graph_node_count == 0 and not extraction.edges and not extraction.claims:
        errors.append(
            _error(
                "extraction_graph_missing",
                "Extraction preserved evidence spans but produced no method-family nodes, edges, or claims.",
            )
        )
        return errors

    if graph_node_count == 0 and extraction.claims:
        errors.append(
            _error(
                "no_graph_nodes",
                "Extraction has claims but no promoted systems, methods, or settings.",
            )
        )

    if not extraction.nodes and (extraction.edges or any(claim.method_ids for claim in extraction.claims)):
        errors.append(
            _error(
                "graph_references_without_nodes",
                "Edges or method-linked claims require declared method-family nodes.",
            )
        )

    if extraction.nodes and not any(node.kind == "system" for node in extraction.nodes):
        errors.append(
            _error(
                "system_node_missing",
                "Method graph must include the paper-local system or introduced artifact root.",
            )
        )

    if extraction.graph.systems and extraction.graph.methods:
        system_ids = {node.local_node_id for node in extraction.graph.systems}
        if not any(edge.parent_id in system_ids for edge in extraction.edges):
            errors.append(
                _error(
                    "system_method_edge_missing",
                    "Method graph must connect the system root to at least one method node.",
                )
            )

    method_family_count = len(extraction.nodes)
    if method_family_count >= 2 and not extraction.edges:
        errors.append(
            _error(
                "method_edges_missing",
                "Multiple method-family nodes require method DAG edges.",
            )
        )

    if extraction.nodes and not extraction.claims:
        errors.append(
            _error(
                "claims_missing",
                "Method graph must attach at least one grounded claim.",
            )
        )
    return errors


def _missing_evidence_errors(
    object_kind: str,
    object_id: str,
    evidence_span_ids: list[str],
    evidence_by_id: dict[str, object],
) -> list[ExtractionValidationError]:
    if not evidence_span_ids:
        return [
            _error(
                "evidence_missing",
                "Object must cite at least one evidence span.",
                object_kind=object_kind,
                object_id=object_id,
            )
        ]
    errors: list[ExtractionValidationError] = []
    for span_id in evidence_span_ids:
        if span_id not in evidence_by_id:
            errors.append(
                _error(
                    "evidence_span_missing",
                    "Referenced evidence span does not exist.",
                    object_kind=object_kind,
                    object_id=object_id,
                    evidence_span_ids=[span_id],
                )
            )
    return errors


def _numeric_grounding_findings(
    object_kind: str,
    object_id: str,
    values: list[str | None],
    evidence_span_ids: list[str],
    evidence_by_id: dict[str, object],
    *,
    require_numeric_grounding: bool,
) -> list[ExtractionValidationError]:
    findings: list[ExtractionValidationError] = []
    for value in values:
        if not value:
            continue
        if not _contains_digit(value):
            continue
        if _appears_in_evidence(value, evidence_span_ids, evidence_by_id):
            continue
        builder = _error if require_numeric_grounding else _warning
        findings.append(
            builder(
                "numeric_grounding_unverified",
                "Numeric value does not appear exactly in cited evidence.",
                object_kind=object_kind,
                object_id=object_id,
                evidence_span_ids=evidence_span_ids,
            )
        )
    return findings


def _evidence_span_warnings(extraction: PaperExtraction) -> list[ExtractionValidationError]:
    warnings: list[ExtractionValidationError] = []
    noisy_span_ids = [span.span_id for span in extraction.evidence_spans if _looks_like_figure_label_noise(span.text)]
    if noisy_span_ids:
        warnings.append(
            _warning(
                "figure_label_noise",
                "Evidence spans contain isolated plot labels or tick values.",
                evidence_span_ids=noisy_span_ids,
            )
        )
    missing_page_ids = [
        span.span_id
        for span in extraction.evidence_spans
        if span.page_start is None or span.page_end is None
    ]
    if missing_page_ids:
        warnings.append(
            _warning(
                "no_page_provenance",
                "Evidence spans have null page_start/page_end.",
                evidence_span_ids=missing_page_ids,
            )
        )
    return warnings


def _graph_quality_warnings(
    extraction: PaperExtraction,
    evidence_by_id: dict[str, object],
) -> list[ExtractionValidationError]:
    warnings: list[ExtractionValidationError] = []
    warnings.extend(_preemption_topology_warnings(extraction))
    warnings.extend(_empty_demotion_warnings(extraction))
    warnings.extend(_status_warnings(extraction, evidence_by_id))
    return warnings


def _preemption_topology_warnings(extraction: PaperExtraction) -> list[ExtractionValidationError]:
    node_by_id = {node.local_node_id: node for node in extraction.nodes}
    warnings: list[ExtractionValidationError] = []
    for edge in extraction.edges:
        parent = node_by_id.get(edge.parent_id)
        child = node_by_id.get(edge.child_id)
        if parent is None or child is None:
            continue
        parent_name = _normalize_identifier(parent.canonical_name)
        child_name = _normalize_identifier(child.canonical_name)
        if "pagedattention" not in parent_name and "paged attention" not in parent_name:
            continue
        if "swapping" not in child_name and "recomputation" not in child_name:
            continue
        warnings.append(
            _warning(
                "preemption_topology_suspicious",
                "Swapping/recomputation should usually sit under sequence-group preemption, not directly under the attention algorithm.",
                object_kind="edge",
                object_id=f"{edge.parent_id}->{edge.child_id}",
                evidence_span_ids=edge.evidence_span_ids,
            )
        )
    return warnings


def _empty_demotion_warnings(extraction: PaperExtraction) -> list[ExtractionValidationError]:
    if extraction.demoted_items:
        return []
    component_markers = ("manager", "scheduler", "worker", "kernel", "api", "frontend")
    span_ids = [
        span.span_id
        for span in extraction.evidence_spans
        if span.section_kind in {"method", "other"} and any(marker in span.text.casefold() for marker in component_markers)
    ]
    if not span_ids:
        return []
    return [
        _warning(
            "demoted_items_empty_for_component_heavy_paper",
            "Method/design evidence contains implementation components but demoted_items is empty.",
            evidence_span_ids=span_ids[:10],
        )
    ]


def _status_warnings(
    extraction: PaperExtraction,
    evidence_by_id: dict[str, object],
) -> list[ExtractionValidationError]:
    warnings: list[ExtractionValidationError] = []
    for node in extraction.nodes:
        if node.status != "uncertain":
            continue
        evidence_text = " ".join(str(getattr(evidence_by_id.get(span_id), "text", "")) for span_id in node.evidence_span_ids)
        normalized_text = _normalize_identifier(evidence_text)
        if not _name_appears_in_text(node.canonical_name, normalized_text):
            continue
        if not any(marker in normalized_text for marker in ("we propose", "we build", "we design", "we implement", "we introduce", "we develop")):
            continue
        warnings.append(
            _warning(
                "paper_local_status_uncertain",
                "Evidence says the paper proposes/builds/designs this node; paper-local status should usually be claimed_new.",
                object_kind="node",
                object_id=node.local_node_id,
                evidence_span_ids=node.evidence_span_ids,
            )
        )
    return warnings


def _cap_warnings(extraction: PaperExtraction, caps: ExtractionCaps) -> list[ExtractionValidationError]:
    system_count = len(extraction.graph.systems)
    method_count = len(extraction.graph.methods)
    checks = (
        ("system_node_count_high", "system nodes", system_count, caps.max_system_nodes),
        ("method_node_count_high", "method nodes", method_count, caps.max_method_nodes),
        ("setting_node_count_high", "settings", len(extraction.settings), caps.max_setting_nodes),
        ("claim_count_high", "claims", len(extraction.claims), caps.max_claims),
        ("outcome_count_high", "outcomes", len(extraction.outcomes), caps.max_outcomes),
        ("demoted_item_count_high", "demoted items", len(extraction.demoted_items), caps.max_demoted_items),
    )
    warnings: list[ExtractionValidationError] = []
    for code, label, count, cap in checks:
        if count <= cap:
            continue
        warnings.append(_warning(code, f"Extraction has {count} {label}, above configured cap {cap}."))
    return warnings


def _has_mechanism_sentence(value: str | None) -> bool:
    if not value:
        return False
    words = re.findall(r"[A-Za-z0-9_%-]+", value)
    return len(words) >= 8


def _appears_in_evidence(value: str, evidence_span_ids: list[str], evidence_by_id: dict[str, object]) -> bool:
    needle = _normalize_for_grounding(value)
    if not needle:
        return False
    for span_id in evidence_span_ids:
        span = evidence_by_id.get(span_id)
        text = getattr(span, "text", "")
        if needle in _normalize_for_grounding(str(text)):
            return True
    return False


def _contains_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)


def _is_explicit_zero(value: float | None) -> bool:
    return value == 0.0


def _looks_like_figure_label_noise(value: str) -> bool:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return False
    if re.match(r"^(fig\.?|figure|table)\b", cleaned, flags=re.IGNORECASE):
        return False
    if re.search(r"[.!?]\s*$", cleaned):
        return False
    words = re.findall(r"[A-Za-z]+", cleaned)
    if re.fullmatch(r"[\d.,]+[kKmM]?", cleaned):
        return True
    if len(words) <= 4 and re.search(r"\b(GB|MB|token/s|tokens/s|requests?|batch size|memory usage)\b", cleaned, flags=re.IGNORECASE):
        return True
    if len(words) <= 3 and re.search(r"[\d#%()]", cleaned):
        return True
    if len(words) <= 3 and cleaned.casefold() in {"others", "parameter size", "existing systems vllm", "vllm"}:
        return True
    return False


def _claim_needs_structured_fields(claim: object) -> bool:
    text = f"{getattr(claim, 'raw_text', '')} {getattr(claim, 'finding', '')}".casefold()
    if not _contains_digit(text):
        return False
    comparison_markers = ("compared", "than", "higher", "lower", "versus", " vs ", "×", "x", "%")
    return any(marker in text for marker in comparison_markers)


def _claim_requires_outcome(claim: object) -> bool:
    return bool(
        getattr(claim, "metric", None)
        and (getattr(claim, "value", None) or getattr(claim, "delta", None))
        and (getattr(claim, "baseline", None) or getattr(claim, "comparator", None))
        and not getattr(claim, "outcome_ids", [])
    )


def _overall_system_claim_attached_too_low(claim: object, extraction: PaperExtraction) -> bool:
    if not extraction.graph.systems:
        return False
    method_ids = set(getattr(claim, "method_ids", []))
    if any(system.local_node_id in method_ids for system in extraction.graph.systems):
        return False
    if not method_ids:
        return False
    text = _normalize_identifier(f"{getattr(claim, 'raw_text', '')} {getattr(claim, 'finding', '')}")
    if not any(_name_appears_in_text(system.canonical_name, text) for system in extraction.graph.systems):
        return False
    metric_text = _normalize_identifier(
        f"{getattr(claim, 'metric', '')} {getattr(claim, 'raw_text', '')} {getattr(claim, 'finding', '')}"
    )
    if "overhead" in metric_text or "kernel latency" in metric_text:
        return False
    return any(term in metric_text for term in ("request rate", "throughput", "latency", "requests"))


def _is_problem_setting(name: str, kind: str) -> bool:
    if kind not in {"application", "task", "workload"}:
        return False
    normalized = _normalize_identifier(name)
    problem_terms = (
        "inefficiency",
        "challenge",
        "problem",
        "limitation",
        "bottleneck",
        "fragmentation issue",
        "memory management",
    )
    return any(term in normalized for term in problem_terms)


def _is_coarse_scenario_setting(name: str, spans: list[object]) -> bool:
    normalized_name = _normalize_identifier(name)
    if normalized_name not in {"decoding scenarios", "decoding scenario"}:
        return False
    text = _normalize_identifier(" ".join(str(getattr(span, "text", "")) for span in spans))
    named_scenarios = ("parallel sampling", "beam search", "shared prefix", "chatbot")
    return any(scenario in text for scenario in named_scenarios)


def _name_appears_in_text(name: str, normalized_text: str) -> bool:
    normalized_name = _normalize_identifier(name)
    if not normalized_name:
        return False
    return f" {normalized_name} " in f" {normalized_text} "


def _is_concrete_reusable_method_demoted_for_missing_mechanism(name: str, reason: str) -> bool:
    normalized_name = _normalize_identifier(name)
    normalized_reason = _normalize_identifier(reason)
    if "mechanism sentence" not in normalized_reason:
        return False
    return any(method_name in normalized_name for method_name in _CONCRETE_REUSABLE_METHOD_NAMES)


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalize_for_grounding(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _missing_target_error(code: str, object_kind: str, object_id: str, target_id: str) -> ExtractionValidationError:
    return _error(
        code,
        "Referenced target does not exist.",
        object_kind=object_kind,
        object_id=object_id,
        evidence_span_ids=[target_id],
    )


def _error(
    code: str,
    message: str,
    *,
    object_kind: str | None = None,
    object_id: str | None = None,
    evidence_span_ids: list[str] | None = None,
) -> ExtractionValidationError:
    return ExtractionValidationError(
        code=code,
        message=message,
        severity=ValidationSeverity.error,
        object_kind=object_kind,
        object_id=object_id,
        evidence_span_ids=evidence_span_ids or [],
    )


def _warning(
    code: str,
    message: str,
    *,
    object_kind: str | None = None,
    object_id: str | None = None,
    evidence_span_ids: list[str] | None = None,
) -> ExtractionValidationError:
    return ExtractionValidationError(
        code=code,
        message=message,
        severity=ValidationSeverity.warning,
        object_kind=object_kind,
        object_id=object_id,
        evidence_span_ids=evidence_span_ids or [],
    )


__all__ = ["validate_extraction"]
