"""
CoChem-SpycFit: Re-export proxy for Core ML-Augmented Execution Engine.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.cochem_spycfit_ml_engine import (
        DualEngineParityBridge,
        GPRManifoldEngine,
        JAXHamiltonianDriver,
        QuantumFeatureExtractor,
        SmartScanNavigator,
        discover_hardware_tier,
        enforce_fp64_precision,
    )
except ImportError:
    from cochem_spycfit.cochem_spycfit_ml_engine import (  # type: ignore
        DualEngineParityBridge,
        GPRManifoldEngine,
        JAXHamiltonianDriver,
        QuantumFeatureExtractor,
        SmartScanNavigator,
        discover_hardware_tier,
        enforce_fp64_precision,
    )

__all__: list[str] = [
    "DualEngineParityBridge",
    "GPRManifoldEngine",
    "JAXHamiltonianDriver",
    "QuantumFeatureExtractor",
    "SmartScanNavigator",
    "discover_hardware_tier",
    "enforce_fp64_precision",
]
