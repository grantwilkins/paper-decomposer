from __future__ import annotations

from .config import load_config
from .pdf_parser import parse_pdf
from .schema import PaperDocument


async def ingest_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperDocument:
    """Parse a paper PDF into a `PaperDocument`.

    The follow-up PR will: extract methods/settings/outcomes/claims from the
    parsed sections, embed them, and persist via `db.PaperDecomposerDB`.
    """
    config = load_config(config_path)
    document = parse_pdf(pdf_path, config)
    return document


__all__ = ["ingest_paper"]
