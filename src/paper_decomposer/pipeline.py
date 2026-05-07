from __future__ import annotations

from typing import Any
from uuid import uuid4, uuid5, NAMESPACE_URL

from .config import load_config
from .extraction.assembler import assemble_extraction
from .extraction.contracts import ExtractionCaps, PaperExtraction
from .extraction.evidence import select_evidence_spans
from .extraction.sanitize import demote_invalid_method_nodes, preserve_graph_and_attach_claims
from .extraction.stages import (
    compress_paper_extraction,
    extract_claims_and_outcomes,
    extract_frontmatter_sketch,
    extract_method_graph,
    repair_paper_extraction,
)
from .extraction.validators import validate_extraction
from .pdf_parser import parse_pdf
from .schema import PaperDocument


async def ingest_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperDocument:
    """Parse a paper PDF into a `PaperDocument` without extraction or DB writes."""
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return document


async def extract_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperExtraction:
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return await extract_document(document, config=config)


async def extract_document(document: PaperDocument, *, config: Any) -> PaperExtraction:
    extraction_config = _extraction_config(config)
    max_model_calls = int(extraction_config.get("max_model_calls_per_paper", 5))
    if extraction_config.get("enabled") is False:
        raise ValueError("Extraction is disabled by pipeline.extraction.enabled.")
    if extraction_config.get("enable_visual_figure_extraction") is True:
        raise ValueError("Visual figure extraction is not implemented; use captions and parser-extracted text.")
    if max_model_calls < 4:
        raise ValueError("Extraction requires at least 4 model calls per paper.")

    paper_id = str(uuid5(NAMESPACE_URL, document.metadata.title))
    extraction_run_id = str(uuid4())
    spans = select_evidence_spans(
        document,
        paper_id=paper_id,
        max_chars_per_stage=int(extraction_config.get("max_input_chars_per_stage", 50_000)),
        include_captions=bool(extraction_config.get("include_captions", True)),
        include_table_text=bool(extraction_config.get("include_table_text", True)),
    )
    if not spans:
        raise ValueError("No high-signal evidence spans were selected for extraction.")

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
    if report.blocking_errors:
        if max_model_calls < 5:
            codes = ", ".join(error.code for error in report.blocking_errors)
            raise ValueError(
                "Extraction failed deterministic validation and repair is outside the model-call budget: "
                f"{codes}"
            )
        repaired = await repair_paper_extraction(extraction, report.blocking_errors, config=config)
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
            codes = ", ".join(error.code for error in report.blocking_errors)
            raise ValueError(f"Extraction failed deterministic validation after repair: {codes}")
    return extraction


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


def _extraction_config(config: Any) -> dict[str, Any]:
    extraction = getattr(getattr(config, "pipeline", None), "extraction", None)
    if isinstance(extraction, dict):
        return extraction
    return {}


def _extraction_caps(extraction_config: dict[str, Any]) -> ExtractionCaps:
    caps = extraction_config.get("caps", {})
    if isinstance(caps, dict):
        return ExtractionCaps.model_validate(caps)
    return ExtractionCaps()


__all__ = ["extract_document", "extract_paper", "ingest_paper"]
