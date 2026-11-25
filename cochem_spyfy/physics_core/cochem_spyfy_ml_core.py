"""
CoChem-SPYFY: Gaussian Process ML Core Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spyfy.physics_core.cochem_spyfy_ml_core import (
        ProvenanceTag,
        ResidualPrediction,
        SpycFitGPRegressor,
    )
except ImportError:
    from cochem_spyfy.physics_core.cochem_spyfy_ml_core import (  # type: ignore[no-redef]
        ProvenanceTag,
        ResidualPrediction,
        SpycFitGPRegressor,
    )

__all__ = [
    "ProvenanceTag",
    "ResidualPrediction",
    "SpycFitGPRegressor",
]
