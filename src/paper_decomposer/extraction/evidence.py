from __future__ import annotations

from collections.abc import Iterable
import re

from paper_decomposer.schema import EvidenceArtifact, PaperDocument, RhetoricalRole, Section

from .contracts import EvidenceClass, EvidenceSpan, ExtractionValidationError, PaperExtraction, SourceKind

_HIGH_SIGNAL_ROLES = {
    RhetoricalRole.abstract,
    RhetoricalRole.introduction,
    RhetoricalRole.method,
    RhetoricalRole.evaluation,
    RhetoricalRole.discussion,
}
_HIGH_SIGNAL_TITLE_PATTERN = re.compile(
    r"\b(abstract|introduction|contribution|method|design|architecture|algorithm|system|implementation|"
    r"evaluation|experiment|result|discussion|conclusion)\b",
    re.IGNORECASE,
)


def select_evidence_spans(
    document: PaperDocument,
    *,
    paper_id: str,
    max_chars_per_stage: int = 50_000,
    include_captions: bool = True,
    include_table_text: bool = True,
) -> list[EvidenceSpan]:
    """Select stable text-grounded evidence spans without inventing provenance."""
    spans: list[EvidenceSpan] = []
    used_chars = 0

    for section_index, section in enumerate(document.sections, start=1):
        if not _is_high_signal_section(section):
            continue
        source_kind = _section_source_kind(section)
        for chunk_index, text in enumerate(_paragraph_chunks(section.body_text), start=1):
            if not text:
                continue
            if _is_isolated_visual_fragment(text):
                continue
            if used_chars + len(text) > max_chars_per_stage and spans:
                return spans
            spans.append(
                EvidenceSpan(
                    span_id=f"{paper_id}:section:{section_index}:span:{chunk_index}",
                    paper_id=paper_id,
                    section_title=section.title,
                    section_kind=section.role.value,
                    text=text,
                    source_kind=source_kind,
                    evidence_class=_evidence_class(text, source_kind=source_kind),
                )
            )
            used_chars += len(text)

        if include_captions:
            for artifact in section.artifacts:
                if not _include_artifact(artifact, include_table_text=include_table_text):
                    continue
                caption_text = artifact.caption.strip()
                if not caption_text:
                    continue
                if used_chars + len(caption_text) > max_chars_per_stage and spans:
                    return spans
                spans.append(
                    EvidenceSpan(
                        span_id=f"{paper_id}:artifact:{artifact.artifact_id}",
                        paper_id=paper_id,
                        section_title=section.title,
                        section_kind=section.role.value,
                        text=caption_text,
                        page_start=artifact.source_page,
                        page_end=artifact.source_page,
                        artifact_id=artifact.artifact_id,
                        source_kind=_artifact_source_kind(artifact),
                        evidence_class=_evidence_class(
                            caption_text,
                            source_kind=_artifact_source_kind(artifact),
                        ),
                    )
                )
                used_chars += len(caption_text)

    return spans


def select_model_draft_spans(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    """Keep evidence classes suitable for method/claim/outcome drafting."""
    allowed: set[EvidenceClass] = {"prose", "caption", "table", "frontmatter"}
    return [span for span in spans if span.evidence_class in allowed]


def select_targeted_repair_spans(
    extraction: PaperExtraction,
    validation_errors: list[ExtractionValidationError],
    *,
    max_chars: int = 16_000,
) -> list[EvidenceSpan]:
    """Select the smallest useful evidence packet for repair prompts."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")

    span_by_id = {span.span_id: span for span in extraction.evidence_spans}
    candidate_ids: list[str] = []

    for error in validation_errors:
        for span_id in error.evidence_span_ids:
            if span_id in span_by_id:
                candidate_ids.append(span_id)
        candidate_ids.extend(_object_evidence_span_ids(extraction, error.object_kind, error.object_id))

    if _needs_graph_rebuild_context(validation_errors) or not candidate_ids:
        candidate_ids.extend(_graph_rebuild_context_span_ids(extraction.evidence_spans))

    if not candidate_ids:
        candidate_ids.extend(span.span_id for span in extraction.evidence_spans)

    selected: list[EvidenceSpan] = []
    seen: set[str] = set()
    used_chars = 0
    for span_id in candidate_ids:
        if span_id in seen:
            continue
        span = span_by_id.get(span_id)
        if span is None:
            continue
        span_chars = len(span.text)
        if selected and used_chars + span_chars > max_chars:
            continue
        selected.append(span)
        seen.add(span_id)
        used_chars += span_chars
        if used_chars >= max_chars:
            break

    return selected or extraction.evidence_spans[:1]


def _is_high_signal_section(section: Section) -> bool:
    if section.role in _HIGH_SIGNAL_ROLES:
        return True
    return bool(_HIGH_SIGNAL_TITLE_PATTERN.search(section.title))


def _object_evidence_span_ids(
    extraction: PaperExtraction,
    object_kind: str | None,
    object_id: str | None,
) -> list[str]:
    if object_kind is None or object_id is None:
        return []

    if object_kind == "node":
        return [span_id for node in extraction.nodes if node.local_node_id == object_id for span_id in node.evidence_span_ids]
    if object_kind == "setting":
        return [
            span_id
            for setting in extraction.settings
            if setting.local_setting_id == object_id
            for span_id in setting.evidence_span_ids
        ]
    if object_kind == "outcome":
        return [
            span_id
            for outcome in extraction.outcomes
            if outcome.outcome_id == object_id
            for span_id in outcome.evidence_span_ids
        ]
    if object_kind == "claim":
        return [span_id for claim in extraction.claims if claim.claim_id == object_id for span_id in claim.evidence_span_ids]
    if object_kind == "demoted_item":
        return [
            span_id
            for item in extraction.demoted_items
            if item.name == object_id
            for span_id in item.evidence_span_ids
        ]
    if object_kind in {"edge", "setting_edge", "method_setting_link"}:
        return _edge_evidence_span_ids(extraction, object_kind, object_id)
    return []


def _edge_evidence_span_ids(extraction: PaperExtraction, object_kind: str, object_id: str) -> list[str]:
    if "->" not in object_id:
        return []
    parent_id, child_id = object_id.split("->", 1)
    if object_kind == "edge":
        return [
            span_id
            for edge in extraction.edges
            if edge.parent_id == parent_id and edge.child_id == child_id
            for span_id in edge.evidence_span_ids
        ]
    if object_kind == "setting_edge":
        return [
            span_id
            for edge in extraction.setting_edges
            if edge.parent_id == parent_id and edge.child_id == child_id
            for span_id in edge.evidence_span_ids
        ]
    if object_kind == "method_setting_link":
        return [
            span_id
            for link in extraction.method_setting_links
            if link.method_id == parent_id and link.setting_id == child_id
            for span_id in link.evidence_span_ids
        ]
    return []


def _needs_graph_rebuild_context(validation_errors: list[ExtractionValidationError]) -> bool:
    graph_rebuild_codes = {
        "extraction_graph_missing",
        "no_graph_nodes",
        "graph_references_without_nodes",
        "system_node_missing",
        "system_method_edge_missing",
        "method_edges_missing",
        "claims_missing",
    }
    return any(error.code in graph_rebuild_codes for error in validation_errors)


def _graph_rebuild_context_span_ids(spans: list[EvidenceSpan]) -> list[str]:
    selected = [
        span.span_id
        for span in spans
        if span.source_kind in {"abstract", "contribution"}
        or span.section_kind in {"abstract", "introduction", "method", "theory"}
    ]
    return selected or [span.span_id for span in spans[:4]]


def _section_source_kind(section: Section) -> SourceKind:
    if section.role is RhetoricalRole.abstract:
        return "abstract"
    if re.search(r"\bcontribution", section.title, flags=re.IGNORECASE):
        return "contribution"
    if re.search(r"\bconclusion", section.title, flags=re.IGNORECASE):
        return "conclusion"
    return "paragraph"


def _evidence_class(text: str, *, source_kind: SourceKind) -> EvidenceClass:
    if source_kind in {"abstract", "contribution", "conclusion"}:
        return "frontmatter"
    if source_kind == "caption":
        return "caption"
    if source_kind == "table_text":
        return "table"
    if _looks_like_formula_fragment(text):
        return "formula_fragment"
    if _looks_like_component_label(text):
        return "component_label"
    if _looks_like_example_text(text):
        return "example_text"
    return "prose"


def _paragraph_chunks(text: str, *, max_chars: int = 2400) -> Iterable[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    for paragraph in paragraphs:
        cleaned = " ".join(paragraph.split())
        if len(cleaned) <= max_chars:
            yield cleaned
            continue
        start = 0
        while start < len(cleaned):
            end = min(start + max_chars, len(cleaned))
            if end < len(cleaned):
                split_at = cleaned.rfind(" ", start, end)
                if split_at > start:
                    end = split_at
            yield cleaned[start:end].strip()
            start = end


def _include_artifact(artifact: EvidenceArtifact, *, include_table_text: bool) -> bool:
    artifact_type = artifact.artifact_type.lower()
    if artifact_type == "table":
        return include_table_text
    return True


def _artifact_source_kind(artifact: EvidenceArtifact) -> SourceKind:
    if artifact.artifact_type.lower() == "table":
        return "table_text"
    return "caption"


def _is_isolated_visual_fragment(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return True
    if re.match(r"^(fig\.?|figure|table)\b", cleaned, flags=re.IGNORECASE):
        return False
    if re.search(r"[.!?]\s*$", cleaned):
        return False
    words = re.findall(r"[A-Za-z]+", cleaned)
    if re.fullmatch(r"[\d.,]+[kKmM]?", cleaned):
        return True
    if len(words) <= 4 and re.search(
        r"\b(GB|MB|token/s|tokens/s|requests?|batch size|memory usage)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return True
    if len(words) <= 3 and re.search(r"[\d#%()]", cleaned):
        return True
    if len(words) <= 3 and cleaned.casefold() in {
        "others",
        "parameter size",
        "existing systems vllm",
        "vllm",
    }:
        return True
    return False


def _looks_like_component_label(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False
    if re.search(r"[.!?]\s*$", cleaned):
        return False
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    if len(words) > 6:
        return False
    component_markers = (
        "scheduler",
        "manager",
        "engine",
        "worker",
        "frontend",
        "backend",
        "shard",
        "cache",
        "api",
        "kernel",
    )
    normalized = cleaned.casefold()
    return any(marker in normalized for marker in component_markers)


def _looks_like_example_text(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    normalized = cleaned.casefold()
    if "four score and seven" in normalized or "brought forth" in normalized:
        return True
    if re.match(r"^(example|prompt|response|input|output)\s*[:：]", cleaned, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_formula_fragment(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) > 160:
        return False
    has_formula_operator = bool(re.search(r"[=∑∏≤≥≈±√→←↔]", cleaned))
    has_latex_marker = "\\" in cleaned or "$" in cleaned
    return has_formula_operator and (has_latex_marker or len(re.findall(r"[A-Za-z]", cleaned)) <= 12)


__all__ = ["select_evidence_spans", "select_model_draft_spans", "select_targeted_repair_spans"]
