from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Allow running from repo root without editable install by extending package lookup
# to src/paper_decomposer.
__path__ = extend_path(__path__, __name__)
_src_package = Path(__file__).resolve().parent.parent / "src" / "paper_decomposer"
if _src_package.exists():
    __path__.append(str(_src_package))

from .config import ConfigError, get_config, load_config

__all__ = ["ConfigError", "get_config", "load_config"]
