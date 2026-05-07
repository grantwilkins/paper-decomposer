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
        if _normalize(node.canonical_name) in section_titles:
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

    errors.extend(_cap_warnings(extraction, caps))
    return ExtractionValidationReport(errors=errors)


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


def _cap_warnings(extraction: PaperExtraction, caps: ExtractionCaps) -> list[ExtractionValidationError]:
    system_count = sum(1 for node in extraction.nodes if node.kind == "system")
    method_count = sum(1 for node in extraction.nodes if node.kind == "method")
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


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


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
