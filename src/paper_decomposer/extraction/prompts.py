from __future__ import annotations

from .contracts import EvidenceSpan

RULES = (
    "Return JSON only. Use only supplied evidence span IDs. Do not invent evidence. "
    "Do not create paper-section nodes. Method nodes need inputs, outputs, and operative move. "
    "Put scenarios, datasets, hardware, models, and metrics in settings. "
    "Claim raw_text must copy source text when possible; finding is the paraphrase. "
    "Do not infer OCR, plots, charts, or visual content."
)


def frontmatter_prompt(spans: list[EvidenceSpan]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RULES},
        {
            "role": "user",
            "content": "Sketch paper metadata, problem, contribution spans, and candidate reusable mechanisms.\n\n"
            + _format_spans(spans),
        },
    ]


def method_graph_prompt(spans: list[EvidenceSpan], sketch_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RULES},
        {
            "role": "user",
            "content": "Apply the method-node test and draft method/settings graph JSON.\n\n"
            f"Sketch:\n{sketch_json}\n\nEvidence:\n{_format_spans(spans)}",
        },
    ]


def claims_outcomes_prompt(spans: list[EvidenceSpan], graph_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RULES},
        {
            "role": "user",
            "content": "Extract grounded claims and explicit outcomes. Preserve numeric text exactly.\n\n"
            f"Graph:\n{graph_json}\n\nEvidence:\n{_format_spans(spans)}",
        },
    ]


def compression_prompt(graph_json: str, claims_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RULES},
        {
            "role": "user",
            "content": "Compress to final paper-local extraction JSON. Keep method-family nodes separate from settings. "
            "Do not put applies_to in method edges.\n\n"
            f"Graph:\n{graph_json}\n\nClaims and outcomes:\n{claims_json}",
        },
    ]


def _format_spans(spans: list[EvidenceSpan], *, max_span_chars: int = 1800) -> str:
    lines: list[str] = []
    for span in spans:
        text = span.text
        if len(text) > max_span_chars:
            text = text[:max_span_chars].rsplit(" ", 1)[0].strip()
        lines.append(f"[{span.span_id}] {span.section_title} ({span.source_kind}): {text}")
    return "\n".join(lines)


__all__ = [
    "RULES",
    "claims_outcomes_prompt",
    "compression_prompt",
    "frontmatter_prompt",
    "method_graph_prompt",
]
