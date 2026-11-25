"""
CoChem-SPYFY: Smart Scan Navigator Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spyfy.physics_core.cochem_spyfy_navigator import (
        ScanRecommendation,
        SmartScanNavigator,
    )
except ImportError:
    from cochem_spyfy.physics_core.cochem_spyfy_navigator import (  # type: ignore[no-redef]
        ScanRecommendation,
        SmartScanNavigator,
    )

__all__ = [
    "ScanRecommendation",
    "SmartScanNavigator",
]
