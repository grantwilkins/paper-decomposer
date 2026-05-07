from __future__ import annotations

from collections.abc import Iterable
import re

from paper_decomposer.schema import EvidenceArtifact, PaperDocument, RhetoricalRole, Section

from .contracts import EvidenceSpan, SourceKind

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
                    )
                )
                used_chars += len(caption_text)

    return spans


def _is_high_signal_section(section: Section) -> bool:
    if section.role in _HIGH_SIGNAL_ROLES:
        return True
    return bool(_HIGH_SIGNAL_TITLE_PATTERN.search(section.title))


def _section_source_kind(section: Section) -> SourceKind:
    if section.role is RhetoricalRole.abstract:
        return "abstract"
    if re.search(r"\bcontribution", section.title, flags=re.IGNORECASE):
        return "contribution"
    if re.search(r"\bconclusion", section.title, flags=re.IGNORECASE):
        return "conclusion"
    return "paragraph"


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


__all__ = ["select_evidence_spans"]
