"""
Claim:
`parse_pdf` converts a research PDF into a semantically structured `PaperDocument`
with section boundaries, rhetorical roles, cleaned text, extracted artifacts, and
config-driven section-size constraints.

Plausible wrong implementations:
- Require larger header font and miss numbered headers in uniform-font PDFs.
- Fail to remove references, causing bibliography text to leak into sections.
- Ignore `min_section_chars` / `max_section_chars`, so oversized sections are not
  split or short sections are retained.
- Keep broken text cleanup (hyphenated line breaks and standalone page numbers).
- Produce malformed artifacts (missing captions/pages or out-of-range page indices).
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import pytest

from paper_decomposer.config import load_config
from paper_decomposer.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
FIXTURE_PDF = ROOT / "fixtures" / "vllm.pdf"


def _build_uniform_font_fixture(pdf_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((50, 70), "Synthetic Paper Title", fontsize=12)
    page.insert_text((50, 90), "Alice Example Bob Example", fontsize=10)

    page.insert_text((50, 120), "Abstract", fontsize=10)
    page.insert_textbox(
        fitz.Rect(50, 140, 545, 220),
        "This abstract line includes page marker line.\n123\nShort summary text.",
        fontsize=10,
    )

    page.insert_text((50, 250), "1 Introduction", fontsize=10)
    page.insert_textbox(
        fitz.Rect(50, 270, 545, 360),
        "This intro has a hyphen-\nated token and should be cleaned.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(50, 370, 545, 560),
        (
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
            "Second paragraph with enough words to force a split when max chars is small. "
        ),
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(50, 565, 545, 620),
        "Figure 1. Synthetic caption for artifact extraction.",
        fontsize=10,
    )

    page.insert_text((50, 640), "2 Evaluation", fontsize=10)
    page.insert_textbox(
        fitz.Rect(50, 660, 545, 705),
        "Evaluation text with benchmark and result discussion.",
        fontsize=10,
    )

    page.insert_text((50, 710), "3 Tiny", fontsize=10)
    page.insert_textbox(fitz.Rect(50, 725, 545, 745), "tiny", fontsize=10)

    page.insert_text((50, 760), "References", fontsize=10)
    page.insert_textbox(fitz.Rect(50, 780, 545, 830), "[1] Some citation", fontsize=10)

    doc.save(pdf_path)
    doc.close()


@pytest.fixture
def app_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    return load_config(CONFIG_PATH)


def test_parse_vllm_extracts_core_structure(app_settings) -> None:
    document = parse_pdf(str(FIXTURE_PDF), app_settings)

    assert len(document.sections) >= 8
    assert document.metadata.title
    assert "PagedAttention" in document.metadata.title or "Memory Management" in document.metadata.title

    roles = {section.role.value for section in document.sections}
    assert {"abstract", "introduction", "method", "evaluation"}.issubset(roles)

    assert all(section.body_text.strip() == section.body_text for section in document.sections)
    assert all(section.char_count == len(section.body_text) for section in document.sections)

    assert all("reference" not in section.title.lower() for section in document.sections)
    assert all("bibliograph" not in section.title.lower() for section in document.sections)

    abstracts = [section for section in document.sections if section.role.value == "abstract"]
    assert abstracts
    assert all(section.char_count <= 1000 for section in abstracts)


def test_parse_vllm_promotes_subsection_roles_and_preserves_numeric_order(app_settings) -> None:
    document = parse_pdf(str(FIXTURE_PDF), app_settings)
    sections_by_number = {section.section_number: section for section in document.sections if section.section_number}

    for key in ("4.1", "4.2", "4.3", "6.2", "6.5", "7.2"):
        assert key in sections_by_number
        assert sections_by_number[key].role.value != "other"

    section_numbers = [section.section_number for section in document.sections if section.section_number]
    assert section_numbers.index("7.3") < section_numbers.index("8")
    assert section_numbers.index("7.3") < section_numbers.index("9")


def test_parse_vllm_extracts_clean_author_list(app_settings) -> None:
    document = parse_pdf(str(FIXTURE_PDF), app_settings)

    authors = document.metadata.authors
    assert len(authors) >= 8
    assert "Woosuk Kwon" in authors
    assert "Zhuohan Li" in authors
    assert "Ion Stoica" in authors
    assert "Siyuan Zhuang Ying" not in authors
    assert "Joseph E. Gonzalez Hao" not in authors
    assert all(2 <= len(author.split()) <= 3 for author in authors)


def test_parse_vllm_artifacts_have_structural_validity(app_settings) -> None:
    document = parse_pdf(str(FIXTURE_PDF), app_settings)
    with fitz.open(FIXTURE_PDF) as source_pdf:
        page_count = len(source_pdf)

    assert len(document.all_artifacts) >= 10

    artifact_types = {artifact.artifact_type for artifact in document.all_artifacts}
    assert "figure" in artifact_types
    assert "table" in artifact_types

    for artifact in document.all_artifacts:
        assert artifact.caption.strip() == artifact.caption
        assert re.search(r"\d", artifact.artifact_id)
        assert 1 <= artifact.source_page <= page_count


def test_parser_detects_numbered_headers_without_font_bump_and_cleans_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "uniform-font.pdf"
    _build_uniform_font_fixture(pdf_path)

    document = parse_pdf(
        str(pdf_path),
        {"pipeline": {"pdf": {"min_section_chars": 1, "max_section_chars": 12_000}}},
    )

    titles = [section.title for section in document.sections]
    assert "Introduction" in titles
    assert "Evaluation" in titles
    assert all("References" != title for title in titles)

    roles = {section.role.value for section in document.sections}
    assert {"abstract", "introduction", "evaluation"}.issubset(roles)

    introduction = next(section for section in document.sections if section.title == "Introduction")
    abstract = next(section for section in document.sections if section.title == "Abstract")
    assert "hyphenated token" in introduction.body_text
    assert "\n123\n" not in f"\n{abstract.body_text}\n"


def test_parser_applies_section_size_limits_and_filters_short_sections(tmp_path: Path) -> None:
    pdf_path = tmp_path / "uniform-font-limits.pdf"
    _build_uniform_font_fixture(pdf_path)

    document = parse_pdf(
        str(pdf_path),
        {"pipeline": {"pdf": {"min_section_chars": 40, "max_section_chars": 220}}},
    )

    assert document.sections
    assert all(section.char_count >= 40 for section in document.sections)
    assert all(section.char_count <= 220 for section in document.sections)
    assert all(section.title != "Tiny" for section in document.sections)
    assert any(section.title.endswith("(Part 2)") for section in document.sections)
