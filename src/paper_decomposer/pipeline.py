from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4, uuid5, NAMESPACE_URL

from .config import load_config
from .models import get_cost_tracker, reset_cost_tracker
from .extraction.assembler import assemble_extraction
from .extraction.contracts import BigModelComparison, BigModelExtractionResult, ExtractionCaps, PaperExtraction
from .extraction.evidence import select_evidence_spans, select_model_draft_spans, select_targeted_repair_spans
from .extraction.sanitize import demote_invalid_method_nodes, preserve_graph_and_attach_claims
from .extraction.stages import (
    cleanup_paper_extraction,
    compress_paper_extraction,
    extract_big_model_draft,
    extract_claims_and_outcomes,
    extract_frontmatter_sketch,
    extract_method_graph,
    repair_paper_extraction,
)
from .extraction.validators import validate_extraction
from .pdf_parser import parse_pdf
from .schema import ModelTier, PaperDocument


async def ingest_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperDocument:
    """Parse a paper PDF into a `PaperDocument` without extraction or DB writes."""
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return document


async def extract_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperExtraction:
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return await extract_document(document, config=config)


async def compare_paper_extractions(pdf_path: str, config_path: str = "config.yaml") -> BigModelComparison:
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return await compare_document_extractions(document, config=config)


async def extract_document(document: PaperDocument, *, config: Any) -> PaperExtraction:
    extraction_config = _extraction_config(config)
    _ensure_extraction_supported(extraction_config)
    strategy = str(extraction_config.get("strategy", "staged")).strip().casefold()
    if strategy in {"big_model_draft", "experimental_big_model_draft"}:
        tier = _model_tier_from_value(extraction_config.get("big_model_draft_tier", extraction_config.get("default_model_tier", "medium")))
        extraction, report, _used_repair = await _run_big_model_extraction(
            document,
            config=config,
            draft_tier=tier,
        )
        if report.blocking_errors:
            codes = ", ".join(error.code for error in report.blocking_errors)
            raise ValueError(f"Big-model extraction failed deterministic validation: {codes}")
        return extraction
    if strategy not in {"staged", "multi_stage"}:
        raise ValueError(f"Unknown extraction strategy: {strategy}")

    max_model_calls = int(extraction_config.get("max_model_calls_per_paper", 5))
    heavy_cleanup_enabled = _heavy_cleanup_enabled(extraction_config)
    if max_model_calls < 4:
        raise ValueError("Extraction requires at least 4 model calls per paper.")

    paper_id = _paper_id(document)
    extraction_run_id = str(uuid4())
    spans = _select_extraction_spans(
        document,
        paper_id=paper_id,
        extraction_config=extraction_config,
        max_chars=int(extraction_config.get("max_input_chars_per_stage", 50_000)),
    )

    sketch = await extract_frontmatter_sketch(_frontmatter_spans(spans), config=config)
    graph = await extract_method_graph(_method_spans(spans), sketch, config=config)
    fallback_graph = getattr(graph, "graph", None)
    claims = await extract_claims_and_outcomes(_evaluation_spans(spans), graph, config=config)
    final = await compress_paper_extraction(graph, claims, config=config)
    extraction = assemble_extraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=document.metadata.title,
        evidence_spans=spans,
        final=final,
    )
    extraction = preserve_graph_and_attach_claims(extraction, fallback_graph=fallback_graph)

    caps = _extraction_caps(extraction_config)
    report = validate_extraction(
        extraction,
        caps=caps,
        require_numeric_grounding=bool(extraction_config.get("require_numeric_grounding", False)),
    )
    used_repair = False
    if report.blocking_errors:
        required_calls = 6 if heavy_cleanup_enabled else 5
        if max_model_calls < required_calls:
            codes = ", ".join(error.code for error in report.blocking_errors)
            repair_path = "repair plus large-model cleanup" if heavy_cleanup_enabled else "repair"
            raise ValueError(
                f"Extraction failed deterministic validation and {repair_path} is outside the model-call budget: "
                f"{codes}"
            )
        repaired = await repair_paper_extraction(extraction, report.blocking_errors, config=config)
        used_repair = True
        extraction = assemble_extraction(
            paper_id=paper_id,
            extraction_run_id=extraction_run_id,
            title=document.metadata.title,
            evidence_spans=spans,
            final=repaired,
        )
        extraction = preserve_graph_and_attach_claims(extraction, fallback_graph=fallback_graph)
        extraction = demote_invalid_method_nodes(extraction)
        extraction = preserve_graph_and_attach_claims(extraction)
        report = validate_extraction(
            extraction,
            caps=caps,
            require_numeric_grounding=bool(extraction_config.get("require_numeric_grounding", False)),
        )
        if report.blocking_errors:
            if not heavy_cleanup_enabled:
                codes = ", ".join(error.code for error in report.blocking_errors)
                raise ValueError(f"Extraction failed deterministic validation after repair: {codes}")
    if heavy_cleanup_enabled:
        required_calls = 6 if used_repair else 5
        if max_model_calls < required_calls:
            raise ValueError("Extraction large-model cleanup is outside the model-call budget.")
        fallback_graph = extraction.graph if extraction.nodes or extraction.settings else fallback_graph
        cleaned = await cleanup_paper_extraction(extraction, report.errors, config=config)
        extraction = assemble_extraction(
            paper_id=paper_id,
            extraction_run_id=extraction_run_id,
            title=document.metadata.title,
            evidence_spans=spans,
            final=cleaned,
        )
        extraction = preserve_graph_and_attach_claims(extraction, fallback_graph=fallback_graph)
        extraction = demote_invalid_method_nodes(extraction)
        extraction = preserve_graph_and_attach_claims(extraction, fallback_graph=fallback_graph)
        report = validate_extraction(
            extraction,
            caps=caps,
            require_numeric_grounding=bool(extraction_config.get("require_numeric_grounding", False)),
        )
        if report.blocking_errors:
            codes = ", ".join(error.code for error in report.blocking_errors)
            raise ValueError(f"Extraction failed deterministic validation after large-model cleanup: {codes}")
    return extraction


async def compare_document_extractions(document: PaperDocument, *, config: Any) -> BigModelComparison:
    extraction_config = _extraction_config(config)
    _ensure_extraction_supported(extraction_config)
    candidate_tiers = _comparison_model_tiers(extraction_config)
    if not candidate_tiers:
        raise ValueError("At least one comparison candidate tier is required.")

    results: list[BigModelExtractionResult] = []
    for tier in candidate_tiers:
        reset_cost_tracker()
        try:
            extraction, report, used_repair = await _run_big_model_extraction(
                document,
                config=config,
                draft_tier=tier,
            )
            results.append(
                BigModelExtractionResult(
                    candidate_tier=tier,
                    model=_model_name(config, tier),
                    ok=report.ok,
                    used_repair=used_repair,
                    validation_report=report,
                    cost=get_cost_tracker(),
                    extraction=extraction,
                )
            )
        except Exception as exc:
            results.append(
                BigModelExtractionResult(
                    candidate_tier=tier,
                    model=_model_name(config, tier),
                    ok=False,
                    cost=get_cost_tracker(),
                    error=str(exc),
                )
            )

    return BigModelComparison(
        paper_id=_paper_id(document),
        title=document.metadata.title,
        results=results,
    )


async def _run_big_model_extraction(
    document: PaperDocument,
    *,
    config: Any,
    draft_tier: ModelTier,
) -> tuple[PaperExtraction, Any, bool]:
    extraction_config = _extraction_config(config)
    max_model_calls = int(extraction_config.get("max_model_calls_per_paper", 2))
    if max_model_calls < 1:
        raise ValueError("Big-model extraction requires at least 1 model call per paper.")

    paper_id = _paper_id(document)
    extraction_run_id = str(uuid4())
    caps = _extraction_caps(extraction_config)
    spans = _select_extraction_spans(
        document,
        paper_id=paper_id,
        extraction_config=extraction_config,
        max_chars=int(
            extraction_config.get(
                "max_input_chars_per_big_model",
                extraction_config.get("max_input_chars_per_stage", 120_000),
            )
        ),
    )

    draft_spans = select_model_draft_spans(spans)
    final = await extract_big_model_draft(draft_spans or spans, config=config, tier=draft_tier, caps=caps)
    extraction = _normalize_big_model_extraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=document.metadata.title,
        evidence_spans=spans,
        final=final,
    )
    report = validate_extraction(
        extraction,
        caps=caps,
        require_numeric_grounding=bool(extraction_config.get("require_numeric_grounding", False)),
    )
    if not report.blocking_errors:
        return extraction, report, False

    if max_model_calls < 2:
        return extraction, report, False

    repair_spans = select_targeted_repair_spans(
        extraction,
        report.blocking_errors,
        max_chars=int(extraction_config.get("targeted_repair_max_chars", 16_000)),
    )
    repair_input = extraction.model_copy(update={"evidence_spans": repair_spans})
    repair_tier = _model_tier_from_value(extraction_config.get("big_model_repair_tier", draft_tier))
    repaired = await repair_paper_extraction(
        repair_input,
        report.blocking_errors,
        config=config,
        tier=repair_tier,
    )
    repaired_extraction = _normalize_big_model_extraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=document.metadata.title,
        evidence_spans=spans,
        final=repaired,
    )
    repaired_report = validate_extraction(
        repaired_extraction,
        caps=caps,
        require_numeric_grounding=bool(extraction_config.get("require_numeric_grounding", False)),
    )
    return repaired_extraction, repaired_report, True


def _normalize_big_model_extraction(
    *,
    paper_id: str,
    extraction_run_id: str,
    title: str,
    evidence_spans: list[Any],
    final: Any,
) -> PaperExtraction:
    extraction = assemble_extraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=title,
        evidence_spans=evidence_spans,
        final=final,
    )
    extraction = preserve_graph_and_attach_claims(extraction)
    extraction = demote_invalid_method_nodes(extraction)
    return preserve_graph_and_attach_claims(extraction)


def _frontmatter_spans(spans: list[Any]) -> list[Any]:
    selected = [
        span
        for span in spans
        if span.source_kind in {"abstract", "contribution"} or span.section_kind in {"introduction", "abstract"}
    ]
    return selected or spans[: min(4, len(spans))]


def _method_spans(spans: list[Any]) -> list[Any]:
    selected = [span for span in spans if span.section_kind in {"method", "theory"}]
    return selected or spans


def _evaluation_spans(spans: list[Any]) -> list[Any]:
    selected = [
        span
        for span in spans
        if span.section_kind in {"evaluation", "discussion"} or span.source_kind in {"caption", "table_text", "conclusion"}
    ]
    return selected or spans


def _select_extraction_spans(
    document: PaperDocument,
    *,
    paper_id: str,
    extraction_config: dict[str, Any],
    max_chars: int,
) -> list[Any]:
    spans = select_evidence_spans(
        document,
        paper_id=paper_id,
        max_chars_per_stage=max_chars,
        include_captions=bool(extraction_config.get("include_captions", True)),
        include_table_text=bool(extraction_config.get("include_table_text", True)),
    )
    if not spans:
        raise ValueError("No high-signal evidence spans were selected for extraction.")
    return spans


def _extraction_config(config: Any) -> dict[str, Any]:
    extraction = getattr(getattr(config, "pipeline", None), "extraction", None)
    if isinstance(extraction, dict):
        return extraction
    return {}


def _ensure_extraction_supported(extraction_config: dict[str, Any]) -> None:
    if extraction_config.get("enabled") is False:
        raise ValueError("Extraction is disabled by pipeline.extraction.enabled.")
    if extraction_config.get("enable_visual_figure_extraction") is True:
        raise ValueError("Visual figure extraction is not implemented; use captions and parser-extracted text.")


def _heavy_cleanup_enabled(extraction_config: dict[str, Any]) -> bool:
    return bool(extraction_config.get("enable_large_model_adjudication", False))


def _extraction_caps(extraction_config: dict[str, Any]) -> ExtractionCaps:
    caps = extraction_config.get("caps", {})
    if isinstance(caps, dict):
        return ExtractionCaps.model_validate(caps)
    return ExtractionCaps()


def _paper_id(document: PaperDocument) -> str:
    return str(uuid5(NAMESPACE_URL, document.metadata.title))


def _comparison_model_tiers(extraction_config: dict[str, Any]) -> list[ModelTier]:
    nested = extraction_config.get("experimental_big_model_compare", {})
    raw_tiers: Any = None
    if isinstance(nested, dict):
        raw_tiers = nested.get("candidate_tiers")
    if raw_tiers is None:
        raw_tiers = extraction_config.get("comparison_model_tiers")
    if raw_tiers is None:
        raw_tiers = [extraction_config.get("big_model_draft_tier", "medium"), extraction_config.get("adjudication_model_tier", "heavy")]
    if not isinstance(raw_tiers, list):
        raise ValueError("comparison_model_tiers must be a list.")

    tiers: list[ModelTier] = []
    for raw_tier in raw_tiers:
        tier = _model_tier_from_value(raw_tier)
        if tier not in tiers:
            tiers.append(tier)
    return tiers


def _model_tier_from_value(value: Any) -> ModelTier:
    if value == "cheap":
        return "small"
    if value in {"small", "medium", "heavy"}:
        return value
    raise ValueError(f"Unknown model tier: {value}")


def _model_name(config: Any, tier: ModelTier) -> str | None:
    raw_config = getattr(config, "raw", config)
    models = _read_value(raw_config, "models")
    model_config = _read_value(models, tier)
    model = _read_value(model_config, "model")
    return str(model) if model is not None else None


def _read_value(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


__all__ = [
    "compare_document_extractions",
    "compare_paper_extractions",
    "extract_document",
    "extract_paper",
    "ingest_paper",
]
