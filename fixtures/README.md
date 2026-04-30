# fixtures/

Input PDFs used by unit tests, integration tests, and manual runs of the pipeline.

## Contents

| File | Paper | Used by |
|------|-------|---------|
| [`vllm.pdf`](vllm.pdf) | Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023). | [tests/test_pdf_parser.py](../tests/test_pdf_parser.py). |

## Why these fixtures exist

- **Parser coverage.** `vllm.pdf` is a two-column conference paper with numbered headers, figures, tables, equations, and a references section — enough surface area to shake out every branch in `pdf_parser.py` (two-column ordering, header detection, caption pattern, reference stripping, artifact capture). Synthetic `fitz`-built PDFs in [`test_pdf_parser.py`](../tests/test_pdf_parser.py) cover additional edge cases (uniform font, short sections, hyphenated line breaks) that are awkward to find in real papers.

## Adding a new fixture

1. Drop the PDF here.
2. Prefer conference papers with clear numbered headers. Arxiv preprints with non-standard layouts will stress-test the parser but also fail more often; fine as a parser fixture, risky as an integration fixture.
3. If the PDF will back an integration test, pick a paper whose decomposition you have independently reviewed — the integration test asserts on structure (roots, negatives, cost), and you want those bounds to reflect a known-good run.
4. Reference the file path via `ROOT / "fixtures" / "<name>.pdf"` in the test, using `Path(__file__).resolve().parents[1]` as `ROOT` (see existing tests for the pattern).

## Licensing note

These PDFs are third-party copyrighted research papers retained here for reproducibility of local tests. They are not redistributed with the package. Do not publish this directory to PyPI or any public artifact.
