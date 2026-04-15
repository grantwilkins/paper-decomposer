from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import fitz

from .schema import EvidenceArtifact, PaperDocument, PaperMetadata, RhetoricalRole, Section

HEADER_PATTERN = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\s+)?([A-Z][A-Za-z\s:,\-&]+)$"
)
APPENDIX_HEADER_PATTERN = re.compile(
    r"^([A-Z](?:\.\d+)*)\s+([A-Z][A-Za-z\s:,\-&]+)$"
)

CAPTION_PATTERN = re.compile(
    r"(?im)(?:^|\n)\s*"
    r"(Figure|Fig\.|Table|Algorithm|Theorem|Lemma|Definition)"
    r"\s*\.?\s*(\d+(?:\.\d+)*)[^\n]*"
)

PAGE_NUMBER_LINE_PATTERN = re.compile(r"(?m)^\s*\d+\s*$")
HYPHENATED_BREAK_PATTERN = re.compile(r"(\w)-\n(\w)")
EXCESS_NEWLINE_PATTERN = re.compile(r"\n{3,}")

ROLE_KEYWORDS: dict[str, list[str]] = {
    "ABSTRACT": ["abstract"],
    "INTRODUCTION": ["introduction"],
    "BACKGROUND": ["background", "preliminary", "preliminaries", "related work", "prior work"],
    "METHOD": ["method", "approach", "design", "architecture", "algorithm", "framework", "model", "system", "our approach", "proposed"],
    "THEORY": ["theory", "analysis", "theoretical", "proof", "convergence", "bound"],
    "EVALUATION": ["experiment", "evaluation", "result", "empirical", "benchmark", "performance", "ablation"],
    "DISCUSSION": ["discussion", "limitation", "future work", "broader impact", "conclusion", "concluding"],
    "APPENDIX": ["appendix", "supplement", "additional"],
}

ROLE_ENUM: dict[str, RhetoricalRole] = {
    "ABSTRACT": RhetoricalRole.abstract,
    "INTRODUCTION": RhetoricalRole.introduction,
    "BACKGROUND": RhetoricalRole.background,
    "METHOD": RhetoricalRole.method,
    "THEORY": RhetoricalRole.theory,
    "EVALUATION": RhetoricalRole.evaluation,
    "DISCUSSION": RhetoricalRole.discussion,
    "APPENDIX": RhetoricalRole.appendix,
}

KNOWN_UNNUMBERED_HEADERS = {
    "abstract",
    "acknowledgement",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
    "conclusion",
    "conclusions",
}

AUTHOR_FILTER_TERMS = (
    "university",
    "institute",
    "school",
    "research",
    "independent",
    "copyright",
    "licensed",
    "sosp",
    "conference",
    "uc ",
)
AUTHOR_NAME_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,2}\b"
)
AUTHOR_SPLIT_PATTERN = re.compile(r"(?:,|;|\||\band\b)", re.IGNORECASE)
CAPITALIZED_TOKEN_PATTERN = re.compile(r"^(?:[A-Z][a-z]+|[A-Z])\.?$")
INITIAL_TOKEN_PATTERN = re.compile(r"^[A-Z]\.$")


@dataclass(slots=True)
class TextBlock:
    idx: int
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    max_font_size: float


@dataclass(slots=True)
class HeaderMatch:
    section_number: str | None
    title: str


@dataclass(slots=True)
class SectionChunk:
    section_number: str | None
    title: str
    body_blocks: list[TextBlock] = field(default_factory=list)


@dataclass(slots=True)
class ArtifactRecord:
    artifact: EvidenceArtifact
    block_idx: int


def parse_pdf(pdf_path: str, config: Any) -> PaperDocument:
    doc = fitz.open(pdf_path)
    try:
        page_blocks, page_widths, font_sizes = _extract_page_blocks(doc)
        body_font_size = _compute_body_font_size(font_sizes)
        use_two_column_layout = _is_two_column_document(page_blocks, page_widths)
        ordered_blocks = _order_blocks(page_blocks, page_widths, use_two_column_layout)
        chunks, preamble_blocks = _segment_sections(ordered_blocks, body_font_size)

        title = _extract_title(ordered_blocks)
        authors = _extract_authors(preamble_blocks, title)
        artifact_records = _extract_artifacts(ordered_blocks)
        min_chars, max_chars = _resolve_section_limits(config)
        sections = _build_sections(chunks, artifact_records, min_chars, max_chars)

        return PaperDocument(
            metadata=PaperMetadata(title=title, authors=authors),
            sections=sections,
            all_artifacts=[record.artifact for record in artifact_records],
        )
    finally:
        doc.close()


def _extract_page_blocks(doc: fitz.Document) -> tuple[dict[int, list[TextBlock]], dict[int, float], Counter[float]]:
    page_blocks: dict[int, list[TextBlock]] = {}
    page_widths: dict[int, float] = {}
    font_sizes: Counter[float] = Counter()

    for page_number, page in enumerate(doc, start=1):
        page_widths[page_number] = float(page.rect.width)
        blocks_for_page: list[TextBlock] = []

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue

            line_texts: list[str] = []
            block_sizes: list[float] = []
            for line in block.get("lines", []):
                span_texts: list[str] = []
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if not text:
                        continue
                    size = round(float(span.get("size", 0.0)), 1)
                    font_sizes[size] += 1
                    block_sizes.append(size)
                    span_texts.append(text)
                if span_texts:
                    line_texts.append(" ".join(span_texts))

            text = "\n".join(line_texts).strip()
            if not text or not block_sizes:
                continue

            x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            blocks_for_page.append(
                TextBlock(
                    idx=-1,
                    page=page_number,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    text=text,
                    max_font_size=max(block_sizes),
                )
            )

        page_blocks[page_number] = blocks_for_page

    return page_blocks, page_widths, font_sizes


def _compute_body_font_size(font_sizes: Counter[float]) -> float:
    if not font_sizes:
        return 10.0
    return float(font_sizes.most_common(1)[0][0])


def _order_blocks(
    page_blocks: dict[int, list[TextBlock]],
    page_widths: dict[int, float],
    use_two_column_layout: bool,
) -> list[TextBlock]:
    ordered: list[TextBlock] = []
    next_idx = 0

    for page_number in sorted(page_blocks):
        blocks = page_blocks[page_number]
        if not blocks:
            continue

        page_width = page_widths[page_number]
        if use_two_column_layout:
            midpoint = page_width / 2.0
            blocks = sorted(
                blocks,
                key=lambda block: (0 if block.x0 < midpoint else 1, block.y0, block.x0),
            )
        else:
            blocks = sorted(blocks, key=lambda block: (block.y0, block.x0))

        for block in blocks:
            block.idx = next_idx
            ordered.append(block)
            next_idx += 1

    return ordered


def _is_two_column_page(blocks: list[TextBlock], page_width: float) -> bool:
    if len(blocks) < 6:
        return False

    midpoint = page_width / 2.0
    margin = page_width * 0.03
    left = sum(1 for block in blocks if block.x0 < (midpoint - margin))
    right = sum(1 for block in blocks if block.x0 > (midpoint + margin))
    ratio = (left + right) / len(blocks)
    return ratio > 0.55 and (left / len(blocks)) > 0.2 and (right / len(blocks)) > 0.1


def _is_two_column_document(
    page_blocks: dict[int, list[TextBlock]],
    page_widths: dict[int, float],
) -> bool:
    candidates = [
        _is_two_column_page(page_blocks[page_number], page_widths[page_number])
        for page_number in sorted(page_blocks)
        if len(page_blocks[page_number]) >= 6
    ]
    if not candidates:
        return False
    return sum(1 for value in candidates if value) >= (len(candidates) / 2)


def _segment_sections(blocks: list[TextBlock], body_font_size: float) -> tuple[list[SectionChunk], list[TextBlock]]:
    chunks: list[SectionChunk] = []
    preamble: list[TextBlock] = []
    current: SectionChunk | None = None

    for block in blocks:
        header = _detect_header(block, body_font_size)
        if header is not None:
            if current is not None:
                chunks.append(current)
            current = SectionChunk(section_number=header.section_number, title=header.title)
            continue

        if current is None:
            preamble.append(block)
        else:
            current.body_blocks.append(block)

    if current is not None:
        chunks.append(current)

    return chunks, preamble


def _detect_header(block: TextBlock, body_font_size: float) -> HeaderMatch | None:
    text = _normalize_inline_whitespace(block.text)
    parsed = _parse_header_text(text)
    if parsed is None:
        return None

    has_section_number = parsed.section_number is not None
    known_unnumbered = parsed.title.lower() in KNOWN_UNNUMBERED_HEADERS
    larger_font = block.max_font_size > (body_font_size * 1.2)

    # Do not treat the paper title as a section header.
    likely_title = (
        block.page <= 2
        and parsed.section_number is None
        and len(parsed.title.split()) >= 6
        and block.max_font_size >= body_font_size * 1.6
    )
    if likely_title:
        return None

    if has_section_number or known_unnumbered:
        return parsed

    if larger_font:
        return parsed

    return None


def _parse_header_text(text: str) -> HeaderMatch | None:
    match = HEADER_PATTERN.fullmatch(text)
    if match:
        return HeaderMatch(
            section_number=match.group(1).strip() if match.group(1) else None,
            title=match.group(2).strip(),
        )

    appendix_match = APPENDIX_HEADER_PATTERN.fullmatch(text)
    if appendix_match:
        return HeaderMatch(
            section_number=appendix_match.group(1).strip(),
            title=appendix_match.group(2).strip(),
        )

    return None


def _extract_artifacts(blocks: list[TextBlock]) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    seen: set[tuple[str, str, str, int]] = set()

    for block in blocks:
        for match in CAPTION_PATTERN.finditer(block.text):
            label = match.group(1).strip()
            number = match.group(2).strip()
            caption = _normalize_inline_whitespace(match.group(0))
            if not caption:
                continue

            artifact_type = _normalize_artifact_type(label)
            artifact_id = f"{label.rstrip('.')} {number}"
            key = (artifact_type, artifact_id, caption, block.page)
            if key in seen:
                continue
            seen.add(key)

            records.append(
                ArtifactRecord(
                    artifact=EvidenceArtifact(
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        caption=caption,
                        source_page=block.page,
                    ),
                    block_idx=block.idx,
                )
            )

    return records


def _build_sections(
    chunks: list[SectionChunk],
    artifact_records: list[ArtifactRecord],
    min_chars: int,
    max_chars: int,
) -> list[Section]:
    artifacts_by_block_idx: dict[int, list[EvidenceArtifact]] = defaultdict(list)
    for record in artifact_records:
        artifacts_by_block_idx[record.block_idx].append(record.artifact)

    sections: list[Section] = []
    chunk_roles = _resolve_chunk_roles(chunks)

    for chunk, role in zip(chunks, chunk_roles, strict=False):
        if _is_reference_section(chunk.title):
            continue

        body_text = _clean_text(_join_block_text(chunk.body_blocks))
        if not body_text:
            continue

        chunk_artifacts: list[EvidenceArtifact] = []
        for block in chunk.body_blocks:
            chunk_artifacts.extend(artifacts_by_block_idx.get(block.idx, []))

        section_max_chars = min(max_chars, 1000) if role == RhetoricalRole.abstract else max_chars
        for part_idx, part_text in enumerate(_split_text_by_max_chars(body_text, section_max_chars), start=1):
            part_text = _clean_text(part_text)
            if not part_text:
                continue
            if len(part_text) < min_chars:
                continue

            part_title = chunk.title if part_idx == 1 else f"{chunk.title} (Part {part_idx})"
            sections.append(
                Section(
                    section_number=chunk.section_number,
                    title=part_title,
                    role=role,
                    body_text=part_text,
                    artifacts=chunk_artifacts if part_idx == 1 else [],
                    char_count=len(part_text),
                )
            )

    return sections


def _resolve_chunk_roles(chunks: list[SectionChunk]) -> list[RhetoricalRole]:
    direct_roles = [_assign_role(chunk.title) for chunk in chunks]
    major_roles: dict[str, RhetoricalRole] = {}

    for chunk, role in zip(chunks, direct_roles, strict=False):
        if not chunk.section_number:
            continue
        major_number = chunk.section_number.split(".", maxsplit=1)[0]
        if "." in chunk.section_number:
            continue
        if role == RhetoricalRole.other:
            continue
        major_roles[major_number] = role

    for chunk, role in zip(chunks, direct_roles, strict=False):
        if not chunk.section_number:
            continue
        if role == RhetoricalRole.other:
            continue
        major_number = chunk.section_number.split(".", maxsplit=1)[0]
        major_roles.setdefault(major_number, role)

    resolved_roles: list[RhetoricalRole] = []
    for chunk, direct_role in zip(chunks, direct_roles, strict=False):
        role = direct_role
        if chunk.section_number and "." in chunk.section_number:
            parent_number = chunk.section_number.split(".", maxsplit=1)[0]
            parent_role = major_roles.get(parent_number)
            if parent_role is not None and parent_role != RhetoricalRole.other:
                role = parent_role
        resolved_roles.append(role)

    return resolved_roles


def _extract_title(blocks: list[TextBlock]) -> str:
    title_candidates = [block for block in blocks if block.page <= 2]
    if not title_candidates:
        return "Unknown Title"

    max_font = max(block.max_font_size for block in title_candidates)
    biggest = [block for block in title_candidates if block.max_font_size >= (max_font - 0.1)]
    biggest.sort(key=lambda block: (-len(_normalize_inline_whitespace(block.text)), block.page, block.y0))

    title = _normalize_inline_whitespace(biggest[0].text)
    return title or "Unknown Title"


def _extract_authors(preamble_blocks: list[TextBlock], title: str) -> list[str]:
    authors: list[str] = []
    seen: set[str] = set()

    for block in preamble_blocks:
        if block.page > 2:
            continue
        if _normalize_inline_whitespace(block.text) == title:
            continue

        for raw_line in block.text.splitlines():
            line = _clean_author_line(raw_line)
            if not line:
                continue
            lower_line = line.lower()
            if any(term in lower_line for term in AUTHOR_FILTER_TERMS):
                continue
            if "@" in line:
                continue
            if len(line.split()) < 2:
                continue

            for segment in _split_author_line(line):
                for name in _extract_author_names(segment):
                    if name in seen:
                        continue
                    seen.add(name)
                    authors.append(name)

    return authors


def _clean_author_line(text: str) -> str:
    cleaned = text.replace("*", " ")
    cleaned = cleaned.replace("∗", " ")
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    return _normalize_inline_whitespace(cleaned)


def _split_author_line(line: str) -> list[str]:
    parts = [part.strip() for part in AUTHOR_SPLIT_PATTERN.split(line) if part.strip()]
    return parts if parts else [line]


def _looks_like_capitalized_sequence(tokens: list[str]) -> bool:
    return all(CAPITALIZED_TOKEN_PATTERN.fullmatch(token) for token in tokens)


def _split_compound_author_segment(segment: str) -> list[str]:
    tokens = [token for token in segment.split() if token]
    if len(tokens) < 4:
        return [segment]
    if not _looks_like_capitalized_sequence(tokens):
        return [segment]
    names: list[str] = []
    idx = 0
    while idx < len(tokens):
        remaining = len(tokens) - idx
        window = tokens[idx:]
        if remaining >= 3 and INITIAL_TOKEN_PATTERN.fullmatch(window[1]):
            names.append(" ".join(window[:3]))
            idx += 3
            continue
        if remaining == 3:
            names.append(" ".join(window[:3]))
            idx += 3
            continue
        if remaining >= 2:
            names.append(" ".join(window[:2]))
            idx += 2
            continue
        names.append(window[0])
        idx += 1
    return names if names else [segment]


def _extract_author_names(segment: str) -> list[str]:
    candidates = _split_compound_author_segment(segment)
    names: list[str] = []
    for candidate in candidates:
        normalized = _normalize_inline_whitespace(candidate)
        tokens = normalized.split()
        if len(tokens) in {2, 3} and _looks_like_capitalized_sequence(tokens):
            names.append(normalized)
            continue
        match_names = [match.group(0).strip() for match in AUTHOR_NAME_PATTERN.finditer(candidate)]
        if match_names:
            names.extend(match_names)
            continue
    return names


def _assign_role(title: str) -> RhetoricalRole:
    normalized_title = title.lower()
    for role_name, keywords in ROLE_KEYWORDS.items():
        if any(keyword in normalized_title for keyword in keywords):
            return ROLE_ENUM[role_name]
    return RhetoricalRole.other


def _resolve_section_limits(config: Any) -> tuple[int, int]:
    min_chars = 0
    max_chars = 12_000
    candidates = _config_candidates(config)

    resolved_min = _read_int_field(candidates, "min_section_chars")
    if resolved_min is not None:
        min_chars = max(resolved_min, 0)

    resolved_max = _read_int_field(candidates, "max_section_chars")
    if resolved_max is not None:
        max_chars = max(resolved_max, 1)

    if max_chars < min_chars:
        max_chars = max(min_chars, 1)

    return min_chars, max_chars


def _config_candidates(config: Any) -> list[Any]:
    candidates: list[Any] = []
    if config is None:
        return candidates

    candidates.append(config)

    pipeline = _read_field(config, "pipeline")
    if pipeline is not None:
        candidates.append(pipeline)

    pdf = _read_field(pipeline, "pdf") if pipeline is not None else None
    if pdf is not None:
        candidates.append(pdf)

    if isinstance(config, dict):
        pipeline_dict = config.get("pipeline")
        if pipeline_dict is not None:
            candidates.append(pipeline_dict)
            if isinstance(pipeline_dict, dict) and pipeline_dict.get("pdf") is not None:
                candidates.append(pipeline_dict["pdf"])
        if config.get("pdf") is not None:
            candidates.append(config["pdf"])

    return candidates


def _read_int_field(candidates: list[Any], key: str) -> int | None:
    for candidate in candidates:
        value = _read_field(candidate, key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _read_field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _split_text_by_max_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return _split_hard(text, max_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_hard(paragraph, max_chars))
            continue

        added_len = len(paragraph) if not current else len(paragraph) + 2
        if current and current_len + added_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += added_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def _split_hard(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        trial = " ".join(current + [word])
        if current and len(trial) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _is_reference_section(title: str) -> bool:
    lowered = title.lower()
    return "reference" in lowered or "bibliograph" in lowered


def _clean_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = HYPHENATED_BREAK_PATTERN.sub(r"\1\2", cleaned)
    cleaned = PAGE_NUMBER_LINE_PATTERN.sub("", cleaned)
    cleaned = EXCESS_NEWLINE_PATTERN.sub("\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _join_block_text(blocks: list[TextBlock]) -> str:
    if not blocks:
        return ""
    return "\n\n".join(block.text for block in blocks)


def _normalize_artifact_type(label: str) -> str:
    token = label.lower().rstrip(".")
    if token == "fig":
        return "figure"
    return token
