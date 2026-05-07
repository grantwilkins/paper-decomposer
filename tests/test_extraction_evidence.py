"""
Claim:
Evidence selection creates stable, text-grounded spans from high-signal parser
sections and captions without fabricating provenance that the parser does not
provide.

Plausible wrong implementations:
- Include low-signal background text and waste the model budget.
- Fabricate page ranges for section text even though Section has no page fields.
- Drop parser-provided artifact page numbers and artifact IDs.
- Produce unstable span IDs across repeated runs on the same document.
- Send isolated plot labels and tick values as ordinary prose spans.
"""

from __future__ import annotations

from paper_decomposer.extraction.evidence import select_evidence_spans
from paper_decomposer.schema import EvidenceArtifact, PaperDocument, PaperMetadata, RhetoricalRole, Section


def _section(title: str, role: RhetoricalRole, body_text: str) -> Section:
    return Section(title=title, role=role, body_text=body_text, char_count=len(body_text))


def test_evidence_spans_are_stable_and_do_not_fabricate_section_pages() -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="PagedAttention"),
        sections=[
            _section("Abstract", RhetoricalRole.abstract, "PagedAttention manages KV cache in fixed-size blocks."),
            _section("Background", RhetoricalRole.background, "General transformer serving background."),
            Section(
                title="System Design",
                role=RhetoricalRole.method,
                body_text="The runtime maps logical KV blocks to physical blocks on demand.",
                artifacts=[
                    EvidenceArtifact(
                        artifact_type="figure",
                        artifact_id="fig-1",
                        caption="Figure 1: PagedAttention address translation.",
                        source_page=4,
                    )
                ],
                char_count=66,
            ),
        ],
    )

    first = select_evidence_spans(document, paper_id="paper-1")
    second = select_evidence_spans(document, paper_id="paper-1")

    assert [span.span_id for span in first] == [span.span_id for span in second]
    assert "General transformer serving background." not in {span.text for span in first}

    section_spans = [span for span in first if span.artifact_id is None]
    assert section_spans
    assert all(span.page_start is None and span.page_end is None for span in section_spans)

    caption = next(span for span in first if span.artifact_id == "fig-1")
    assert caption.page_start == 4
    assert caption.page_end == 4
    assert caption.source_kind == "caption"


def test_table_captions_obey_table_text_policy() -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="ORCA"),
        sections=[
            Section(
                title="Evaluation",
                role=RhetoricalRole.evaluation,
                body_text="ORCA improves serving throughput in ShareGPT workloads.",
                artifacts=[
                    EvidenceArtifact(
                        artifact_type="table",
                        artifact_id="tbl-1",
                        caption="Table 1: Throughput improves by 2.2x.",
                        source_page=7,
                    )
                ],
                char_count=56,
            )
        ],
    )

    with_tables = select_evidence_spans(document, paper_id="paper-1", include_table_text=True)
    without_tables = select_evidence_spans(document, paper_id="paper-1", include_table_text=False)

    assert any(span.artifact_id == "tbl-1" for span in with_tables)
    assert all(span.artifact_id != "tbl-1" for span in without_tables)


def test_isolated_visual_fragments_are_filtered_but_captions_remain() -> None:
    document = PaperDocument(
        metadata=PaperMetadata(title="vLLM"),
        sections=[
            Section(
                title="Introduction",
                role=RhetoricalRole.introduction,
                body_text=(
                    "vLLM proposes PagedAttention for high-throughput LLM serving.\n\n"
                    "Memory usage (GB)\n\n"
                    "1.2k\n\n"
                    "Batch size (# requests)\n\n"
                    "Figure 1. vLLM reduces KV cache waste in serving workloads."
                ),
                char_count=154,
            )
        ],
    )

    spans = select_evidence_spans(document, paper_id="paper-1")
    texts = {span.text for span in spans}

    assert "vLLM proposes PagedAttention for high-throughput LLM serving." in texts
    assert "Figure 1. vLLM reduces KV cache waste in serving workloads." in texts
    assert "Memory usage (GB)" not in texts
    assert "1.2k" not in texts
    assert "Batch size (# requests)" not in texts
