"""
CoChem-SpycFit: Interactive Bayesian Active Learning & ML-Augmented Assignment Module.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Prevent stale external scratch directories named `__agentic` on sys.path from
# shadowing `inspect2` when optional dependencies (`zarr` -> `deprecated`) import.
_sys.path[:] = [
    _entry for _entry in _sys.path if _Path(_entry).name.lower() != "__agentic"
]

__version__ = "0.1.0"

__all__: list[str] = [
    "__version__",
]
