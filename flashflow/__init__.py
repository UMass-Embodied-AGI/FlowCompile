"""Development-time namespace shim for the vendored FlashFlow package."""
from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_vendored = Path(__file__).resolve().parent.parent / "3rdparty" / "flashflow" / "flashflow"
if _vendored.exists():
    vendored_path = str(_vendored)
    if vendored_path not in __path__:
        __path__.append(vendored_path)
