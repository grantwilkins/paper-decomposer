"""
Claim:
Extraction is an explicit CLI mode that does not change parse-only dry-run
behavior, and extraction JSON can be written without requiring a database.

Plausible wrong implementations:
- Run extraction during parse-only dry run.
- Require a database for extraction dry-run JSON.
- Ignore the requested output path and only print to the console.
"""

from __future__ import annotations

from pathlib import Path

from paper_decomposer import cli
from paper_decomposer.extraction.contracts import EvidenceSpan, PaperExtraction


def test_extract_cli_writes_json_without_database(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_path = tmp_path / "extraction.json"

    async def fake_extract_paper(pdf_path_arg: str, config_path: str) -> PaperExtraction:
        return PaperExtraction(
            paper_id="paper-1",
            extraction_run_id="run-1",
            title="Tiny Paper",
            evidence_spans=[
                EvidenceSpan(
                    span_id="s1",
                    paper_id="paper-1",
                    section_title="Abstract",
                    section_kind="abstract",
                    text="Tiny evidence.",
                    source_kind="abstract",
                )
            ],
        )

    monkeypatch.setattr(cli, "extract_paper", fake_extract_paper)
    monkeypatch.setattr(cli, "reset_cost_tracker", lambda: None)
    monkeypatch.setattr(cli, "get_cost_tracker", lambda: {"total_cost_usd": 0.0})

    status = cli.main([str(pdf_path), "--extract", "--output-json", str(output_path)])

    assert status == 0
    text = output_path.read_text(encoding="utf-8")
    assert '"paper_id": "paper-1"' in text
    assert '"evidence_spans"' in text
