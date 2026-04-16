from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table
import yaml

from .pdf_parser import parse_pdf
from .pipeline import decompose_paper

console = Console()


def _load_pdf_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return raw


def _print_dry_run_summary(pdf_path: Path, config_path: str) -> None:
    config = _load_pdf_config(config_path)
    document = parse_pdf(str(pdf_path), config)

    console.print(f"[bold]Dry run:[/bold] {pdf_path}")
    console.print(
        f"Title: {document.metadata.title}\n"
        f"Sections: {len(document.sections)} | Artifacts: {len(document.all_artifacts)}"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Role")
    table.add_column("Section")
    table.add_column("Chars", justify="right")

    for idx, section in enumerate(document.sections, start=1):
        label = (
            f"{section.section_number} {section.title}"
            if section.section_number
            else section.title
        )
        table.add_row(str(idx), section.role.value, label, str(section.char_count))

    console.print(table)


def _run_single(pdf_path: Path, config_path: str, dry_run: bool) -> None:
    if dry_run:
        _print_dry_run_summary(pdf_path, config_path)
        return

    result = asyncio.run(decompose_paper(str(pdf_path), config_path))
    console.print(
        f"[green]Done[/green] {pdf_path.name} | "
        f"roots={len(result.claim_tree)} support_details={len(result.support_details)} "
        f"cost=${result.extraction_cost_usd:.4f}"
    )


def _run_batch(directory: Path, config_path: str, dry_run: bool) -> int:
    pdfs = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {directory}[/yellow]")
        return 1

    failures = 0
    for pdf in pdfs:
        try:
            _run_single(pdf, config_path, dry_run)
        except Exception as exc:  # pragma: no cover - behavior validated via integration execution
            failures += 1
            console.print(f"[red]Failed[/red] {pdf.name}: {exc}")

    if failures:
        console.print(f"[yellow]Completed with {failures} failures.[/yellow]")
        return 1

    console.print(f"[green]Processed {len(pdfs)} PDF(s).[/green]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decompose ML papers into structured claim trees")
    parser.add_argument("input", help="PDF file or directory containing PDFs")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Parse PDFs and print sections only")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            console.print(f"[red]Input file is not a PDF:[/red] {input_path}")
            return 1
        _run_single(input_path, args.config, args.dry_run)
        return 0

    if input_path.is_dir():
        return _run_batch(input_path, args.config, args.dry_run)

    console.print(f"[red]Unsupported input path:[/red] {input_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
