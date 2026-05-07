from __future__ import annotations

from .contracts import EvidenceSpan, PaperExtraction
from .stages import ExtractionDraft


def assemble_extraction(
    *,
    paper_id: str,
    extraction_run_id: str,
    title: str,
    evidence_spans: list[EvidenceSpan],
    final: ExtractionDraft,
) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        extraction_run_id=extraction_run_id,
        title=title,
        evidence_spans=evidence_spans,
        candidates=final.candidates,
        nodes=final.nodes,
        edges=final.edges,
        settings=final.settings,
        method_setting_links=final.method_setting_links,
        outcomes=final.outcomes,
        claims=final.claims,
        demoted_items=final.demoted_items,
    )


__all__ = ["assemble_extraction"]
