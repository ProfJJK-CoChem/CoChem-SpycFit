"""
CoChem-SpycFit: Smart Scan Navigator Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_navigator import (
        ScanRecommendation,
        SmartScanNavigator,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_navigator import (  # type: ignore[no-redef]
        ScanRecommendation,
        SmartScanNavigator,
    )

__all__ = [
    "ScanRecommendation",
    "SmartScanNavigator",
]
