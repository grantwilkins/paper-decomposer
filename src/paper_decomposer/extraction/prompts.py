from __future__ import annotations

from .contracts import EvidenceSpan, ExtractionValidationError

RULES = (
    "Return JSON only. The extractor is graph-first: build the paper-local method/settings graph before claims. "
    "Use only supplied evidence span IDs. Do not invent evidence. "
    "Do not create paper-section nodes. Method nodes need inputs, outputs, and operative move. "
    "Return a sparse method DAG, not a paper outline or evidence-span list. "
    "Put scenarios, datasets, hardware, models, and metrics in graph.settings. "
    "Use graph.method_edges only for method/system relationships; use graph.setting_edges for setting relationships; "
    "use graph.method_setting_links for applies_to, evaluated_on, and uses_artifact. "
    "Claim raw_text must copy source text when possible; finding is the paraphrase. "
    "Claims must attach to the most specific existing graph node, setting, or outcome. "
    "Do not infer OCR, plots, charts, or visual content."
)

CALIBRATION = """
Graph calibration:
- Prefer a system -> central primitive -> reusable mechanisms shape.
- For vLLM-like papers, vLLM is the system, PagedAttention is the central primitive, and reusable
  mechanisms include block-wise KV cache address translation, on-demand KV block allocation,
  block-level KV cache sharing, KV block copy-on-write, sequence-group preemption,
  KV-cache swapping, and KV-cache recomputation.
- Do not demote those concrete mechanisms merely because a mechanism_sentence is missing. Synthesize
  a grounded mechanism_sentence from supplied evidence, citing the spans that describe logical and
  physical KV blocks, block tables, on-demand allocation, reference counts, copy-on-write, all-or-none
  eviction, swapping, or recomputation.
- Treat decoding scenarios such as single-sequence generation, parallel sampling, beam search,
  shared-prefix prompting, and chatbot serving as applications/settings or claim contexts, not
  first-class method nodes unless the paper introduces a new reusable mechanism for that scenario.
- Demote implementation support such as fused kernels, fork/append/free APIs, frontend frameworks,
  scheduler message plumbing, code size, and implementation language.
- Attach composed end-to-end throughput/request-rate claims to the system. Attach memory-sharing
  claims to the most specific sharing mechanism. Attach kernel-overhead claims to PagedAttention
  or its most specific supporting mechanism, and use claim_type=overhead for costs/slowdowns.
- Split named task/settings nodes such as parallel sampling, beam search, shared-prefix prompting,
  chatbot serving, ShareGPT, Alpaca, WMT16 English-to-German, OPT-13B, LLaMA-13B, and NVIDIA A100.
  Do not store problems such as KV-cache memory inefficiency as application settings.
- For comparison claims, fill metric, delta/value, baseline, and comparator when present in raw text.
- For claims with metric plus delta/value plus comparator/baseline, create explicit outcome rows and
  link the claims to those outcomes.
- Use meaningful confidence for evidence-copied claims and cited edges; do not emit 0.0 for clearly
  grounded objects.
""".strip()


def frontmatter_prompt(spans: list[EvidenceSpan]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Produce GraphSketch JSON from frontmatter: paper metadata, central problem, systems, methods, settings, "
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
            "Do not emit floating claims; every claim must link to method_ids, setting_ids, or outcome_ids.\n\n"
            f"Graph:\n{graph_json}\n\nEvidence:\n{_format_spans(spans)}",
        },
    ]


def compression_prompt(graph_json: str, claims_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Compress to final PaperExtraction JSON with graph.systems, graph.methods, graph.method_edges, "
            "graph.settings, graph.setting_edges, graph.method_setting_links, claims, outcomes, and demoted_items. "
            "Do not include final candidates. Required: system root, method nodes, method DAG edges, and grounded claims. "
            "Keep method-family nodes separate from settings. "
            "Do not put applies_to in method edges. Every method node must include mechanism_sentence. "
            "Do not create nodes named after section headings. Preserve major claims from the claims input.\n\n"
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
            "content": "Repair this extraction JSON so deterministic validation passes. "
            "If no_graph_nodes, claims_unattached, extraction_graph_missing, system_node_missing, method_edges_missing, or claims_missing appears, "
            "produce a sparse paper-local method DAG with a system root, method edges, and grounded claims. "
            "For method_missing_mechanism_sentence, add a grounded mechanism_sentence with inputs, outputs, "
            "and operative move when the node names a concrete reusable mechanism. Only demote vague section-shaped "
            "or implementation-detail methods. For concrete_method_demoted_for_missing_mechanism, promote the item "
            "back to a method node and synthesize the mechanism_sentence from evidence. For section_heading_promoted, demote/remove "
            "the section-heading node and repair affected edges/links/claims. Keep only supplied evidence IDs.\n\n"
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
            "content": "Run a final heavy-model cleanup pass on this validated paper-local extraction. "
            "Return a complete ExtractionDraft JSON with graph, outcomes, claims, and demoted_items. "
            "Preserve valid graph structure and grounded claims unless supplied evidence requires a specific fix. "
            "Improve only extraction-contract issues: ID consistency, setting deduplication, claim attachment, "
            "method topology, metric/value/baseline/comparator fields, outcome rows, and demotion decisions. "
            "Do not invent evidence, do not drop major claims, and keep only supplied evidence IDs.\n\n"
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
        lines.append(f"[{span.span_id}] {span.section_title} ({span.source_kind}): {text}")
    return "\n".join(lines)


def _system_prompt() -> str:
    return f"{RULES}\n\n{CALIBRATION}"


__all__ = [
    "CALIBRATION",
    "RULES",
    "claims_outcomes_prompt",
    "cleanup_prompt",
    "compression_prompt",
    "frontmatter_prompt",
    "method_graph_prompt",
    "repair_prompt",
]
