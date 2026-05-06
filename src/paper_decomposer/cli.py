from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table
import yaml

from .models import get_cost_tracker, reset_cost_tracker
from .pdf_parser import parse_pdf
from .pipeline import extract_paper, ingest_paper

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


def _print_section_summary(pdf_path: Path, config_path: str) -> None:
    config = _load_pdf_config(config_path)
    document = parse_pdf(str(pdf_path), config)

    console.print(f"[bold]{pdf_path}[/bold]")
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


def _run_single(
    pdf_path: Path,
    config_path: str,
    dry_run: bool,
    extract: bool,
    output_json: Path | None,
) -> None:
    if dry_run and not extract:
        _print_section_summary(pdf_path, config_path)
        return

    if extract:
        reset_cost_tracker()
        extraction = asyncio.run(extract_paper(str(pdf_path), config_path))
        payload = extraction.model_dump(mode="json")
        output_text = json.dumps(payload, indent=2, sort_keys=True)
        if output_json is not None:
            output_json.write_text(output_text + "\n", encoding="utf-8")
            console.print(f"[green]Wrote extraction JSON[/green] {output_json}")
        else:
            console.print(output_text)
        costs = get_cost_tracker()
        console.print(
            "[green]Extracted[/green] "
            f"{pdf_path.name} | nodes={len(extraction.nodes)} settings={len(extraction.settings)} "
            f"claims={len(extraction.claims)} outcomes={len(extraction.outcomes)} "
            f"cost=${float(costs['total_cost_usd']):.6f}"
        )
        return

    document = asyncio.run(ingest_paper(str(pdf_path), config_path))
    console.print(
        f"[green]Parsed[/green] {pdf_path.name} | "
        f"sections={len(document.sections)} artifacts={len(document.all_artifacts)}"
    )


def _run_batch(directory: Path, config_path: str, dry_run: bool, extract: bool) -> int:
    pdfs = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {directory}[/yellow]")
        return 1

    failures = 0
    for pdf in pdfs:
        try:
            _run_single(pdf, config_path, dry_run, extract, output_json=None)
        except Exception as exc:
            failures += 1
            console.print(f"[red]Failed[/red] {pdf.name}: {exc}")

    if failures:
        console.print(f"[yellow]Completed with {failures} failures.[/yellow]")
        return 1

    console.print(f"[green]Processed {len(pdfs)} PDF(s).[/green]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest ML papers into the methods/settings/outcomes/claims DAG")
    parser.add_argument("input", help="PDF file or directory containing PDFs")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Parse PDFs and print sections only")
    parser.add_argument("--extract", action="store_true", help="Run validated extraction after parsing")
    parser.add_argument("--output-json", type=Path, help="Write extraction JSON for a single PDF")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            console.print(f"[red]Input file is not a PDF:[/red] {input_path}")
            return 1
        _run_single(input_path, args.config, args.dry_run, args.extract, args.output_json)
        return 0

    if input_path.is_dir():
        if args.output_json is not None:
            console.print("[red]--output-json is only supported for a single PDF.[/red]")
            return 1
        return _run_batch(input_path, args.config, args.dry_run, args.extract)

    console.print(f"[red]Unsupported input path:[/red] {input_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
