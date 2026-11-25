"""
CoChem-SPYFY: Quantum Feature Extractor Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spyfy.physics_core.cochem_spyfy_features import (
        QuantumFeatureExtractor,
    )
except ImportError:
    from cochem_spyfy.physics_core.cochem_spyfy_features import (  # type: ignore[no-redef]
        QuantumFeatureExtractor,
    )

__all__ = [
    "QuantumFeatureExtractor",
]
