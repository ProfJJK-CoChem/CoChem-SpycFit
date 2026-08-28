"""
CoChem-SpycFit: Quantum Feature Extractor Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_features import (  # type: ignore[no-redef]
        QuantumFeatureExtractor,
    )

__all__ = [
    "QuantumFeatureExtractor",
]
