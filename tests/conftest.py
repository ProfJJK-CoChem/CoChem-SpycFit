"""Pytest configuration and environment hygiene for CoChem-SPYFY."""

from __future__ import annotations

import sys
from pathlib import Path

# Guard against stale developer scratch directories (e.g. D:\__CoChem\__agentic)
# injected via external .pth files containing top-level scripts named `inspect2.py`
# that crash `deprecated.params` -> `numcodecs` -> `zarr` during test collection.
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry).name.lower() != "__agentic"
]
