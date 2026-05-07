from __future__ import annotations

from .contracts import EvidenceSpan, ExtractionCaps, ExtractionValidationError

RULES = """
Return JSON only.

Extraction contract:
1. Use only supplied evidence span IDs; never infer unseen figure, OCR, chart, or plot content.
2. Discover paper-local objects only; do not assign global identity. Use typed slug IDs from canonical names:
   local:system:*, local:method:*, local:setting:*, local:problem:*, local:outcome:*.
3. Build a compact graph, not a paper outline. Systems/methods go in graph; problems go in problems;
   datasets, tasks, workloads, hardware, model artifacts, applications, and metrics go in settings.
4. Methods are reusable mechanisms. Give each method a grounded mechanism_sentence. Use category_tags
   for broad labels; do not emit method_category nodes.
5. Claims are compact propositions. Put measurements in outcomes, not claim fields. Link each claim to
   the most specific methods, settings, problems, and outcomes.
6. Outcomes are measurement rows. Split rows when baseline/comparator, metric, dataset, task, model,
   or hardware differs.
7. Baseline systems are reference system nodes. Demote only implementation support or duplicate details.
8. Claim evidence must be grounded source text, not component labels, examples, formula fragments, or
   isolated figure labels.
""".strip()

CALIBRATION = """
Graph calibration:
1. Prefer system -> central primitive -> reusable mechanisms.
2. Keep scenario-specific variants as settings/adapters unless the paper introduces a reusable mechanism.
3. Do not demote concrete mechanisms only because a mechanism_sentence is missing; write one from evidence.
4. Attach end-to-end claims to the system, mechanism claims to the responsible mechanism, and overhead
   claims to the mechanism causing the cost.
5. Split named tasks, datasets, models, hardware, and workloads into settings and attach outcomes to them.
""".strip()

BIG_MODEL_COMPACT_RULES = """
One-pass extraction:
1. Return complete ExtractionDraft JSON matching the schema: graph, problems, outcomes, claims, demoted_items.
2. Use slugged local IDs based on canonical names, for example local:method:paged_attention.
3. Build the method spine first; keep scenario names as settings unless they introduce reusable mechanisms.
4. Prefer few reusable methods and compact claims; do not create one claim per numeric row.
5. Create separate outcome rows for each baseline/comparator x dataset x task x model x hardware x metric combination.
6. Claims with outcomes must also attach to the responsible method/system unless they are purely background/problem claims.
7. Synthesize mechanism_sentence for concrete mechanisms; do not demote them only because the sentence was missing.
8. Internally verify all references resolve, referenced outcomes are unique, numeric text is exact when possible,
   and claim evidence is not noisy.
""".strip()


def frontmatter_prompt(spans: list[EvidenceSpan]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Produce GraphSketch JSON: paper metadata, central problems, systems, methods, settings, "
            "and tentative graph edges. Do not extract claims yet.\n\n"
            + _format_spans(spans),
        },
    ]


def method_graph_prompt(spans: list[EvidenceSpan], sketch_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Refine the GraphSketch into a compact method/settings graph. Do not extract claims yet.\n\n"
            f"Sketch:\n{sketch_json}\n\nEvidence:\n{_format_spans(spans)}",
        },
    ]


def claims_outcomes_prompt(spans: list[EvidenceSpan], graph_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Extract grounded claims and explicit outcomes against the supplied graph. Preserve numeric text exactly. "
            "Do not emit floating claims. Claims with outcome_ids must also include responsible method_ids unless "
            "they are purely background/problem claims.\n\n"
            f"Graph:\n{graph_json}\n\nEvidence:\n{_format_spans(spans)}",
        },
    ]


def big_model_compact_prompt(spans: list[EvidenceSpan], caps: ExtractionCaps) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"{_system_prompt()}\n\n{BIG_MODEL_COMPACT_RULES}"},
        {
            "role": "user",
            "content": "Produce complete compact ExtractionDraft JSON from the supplied evidence. "
            "Return only schema fields; do not add validation_notes or commentary.\n\n"
            "Output caps:\n"
            f"- system nodes <= {caps.max_system_nodes}\n"
            f"- method nodes <= {caps.max_method_nodes}\n"
            f"- settings <= {caps.max_setting_nodes}\n"
            f"- compact claims <= {caps.max_claims}\n"
            f"- outcomes <= {caps.max_outcomes}\n"
            f"- demoted items <= {caps.max_demoted_items}\n\n"
            "Hard checks:\n"
            "- every referenced method_id exists in graph.systems or graph.methods\n"
            "- every referenced setting_id exists in graph.settings\n"
            "- every referenced problem_id exists in problems\n"
            "- every referenced outcome_id exists exactly once in outcomes\n"
            "- every numeric value appears exactly in cited evidence when possible\n"
            "- claims with outcomes also cite responsible methods unless they are background/problem-only\n"
            "- claim_type matches the metric and finding\n"
            "- baseline systems are reference system nodes, not demoted items\n"
            "- component_label, example_text, and formula_fragment evidence is not used as claim evidence\n\n"
            f"Evidence:\n{_format_spans(spans, max_span_chars=2200)}",
        },
    ]


def compression_prompt(graph_json: str, claims_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Compress to final PaperExtraction JSON: graph.systems, graph.methods, graph.method_edges, "
            "graph.settings, graph.setting_edges, graph.method_setting_links, problems, claims, outcomes, demoted_items. "
            "Do not include final candidates. Keep methods separate from settings. Every method needs mechanism_sentence. "
            "Preserve major claims and attach them to the most specific graph objects.\n\n"
            f"Graph:\n{graph_json}\n\nClaims and outcomes:\n{claims_json}",
        },
    ]


def repair_prompt(
    extraction_json: str,
    validation_errors: list[ExtractionValidationError],
    evidence_spans: list[EvidenceSpan],
) -> list[dict[str, str]]:
    errors = "\n".join(
        f"- {error.code}: {error.object_kind or ''} {error.object_id or ''} {error.message}".strip()
        for error in validation_errors
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Repair only the listed validation errors and return ExtractionDraft JSON. "
            "Keep valid graph objects and supplied evidence IDs. If the graph is missing, create a sparse "
            "system -> method DAG with grounded claims. If a concrete method lacks mechanism_sentence, write one "
            "from evidence instead of demoting it. If a claim/outcome reference is missing or duplicated, retarget "
            "it or create the missing grounded row. Remove section-heading nodes and implementation details.\n\n"
            f"Validation errors:\n{errors}\n\nExtraction JSON:\n{extraction_json}\n\nEvidence:\n{_format_spans(evidence_spans, max_span_chars=1000)}",
        },
    ]


def cleanup_prompt(
    extraction_json: str,
    validation_issues: list[ExtractionValidationError],
    evidence_spans: list[EvidenceSpan],
) -> list[dict[str, str]]:
    issues = "\n".join(
        f"- {issue.code}: {issue.object_kind or ''} {issue.object_id or ''} {issue.message}".strip()
        for issue in validation_issues
    )
    if not issues:
        issues = "- none"
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Run final cleanup and return complete ExtractionDraft JSON. Preserve valid structure and "
            "major grounded claims. Fix only extraction-contract issues: slug IDs, setting deduplication, "
            "claim method attachments, outcome rows, method topology, problem modeling, demotions, and noisy evidence. "
            "Do not invent evidence; keep only supplied evidence IDs.\n\n"
            f"Validation issues and warnings:\n{issues}\n\nExtraction JSON:\n{extraction_json}\n\n"
            f"Evidence:\n{_format_spans(evidence_spans, max_span_chars=1200)}",
        },
    ]


def _format_spans(spans: list[EvidenceSpan], *, max_span_chars: int = 1800) -> str:
    lines: list[str] = []
    for span in spans:
        text = span.text
        if len(text) > max_span_chars:
            text = text[:max_span_chars].rsplit(" ", 1)[0].strip()
        lines.append(f"[{span.span_id}] {span.section_title} ({span.source_kind}/{span.evidence_class}): {text}")
    return "\n".join(lines)


def _system_prompt() -> str:
    return RULES


__all__ = [
    "CALIBRATION",
    "BIG_MODEL_COMPACT_RULES",
    "RULES",
    "big_model_compact_prompt",
    "claims_outcomes_prompt",
    "cleanup_prompt",
    "compression_prompt",
    "frontmatter_prompt",
    "method_graph_prompt",
    "repair_prompt",
]
