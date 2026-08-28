"""
CoChem-SpycFit: Re-export proxy for ML-Augmented Assignment Schema & Data Models.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.cochem_spycfit_ml_schema import (
        AssignmentTag,
        DipoleComponents,
        DipoleType,
        FitState,
        HardwareScanChunk,
        IsotopeRetrievalError,
        MLSchemaError,
        MultiTierResourceLimits,
        ParityStatus,
        ParityWarning,
        ProvenanceError,
        ProvenanceLedger,
        QuantumNumberError,
        SpectroscopicTransition,
        TransitionQuantumNumbers,
        compute_bytes_sha256,
        compute_dynamic_molecular_mass,
        compute_file_sha256,
        compute_object_sha256,
        get_dynamic_isotope_mass,
    )
except ImportError:
    from cochem_spycfit.cochem_spycfit_ml_schema import (  # type: ignore
        AssignmentTag,
        DipoleComponents,
        DipoleType,
        FitState,
        HardwareScanChunk,
        IsotopeRetrievalError,
        MLSchemaError,
        MultiTierResourceLimits,
        ParityStatus,
        ParityWarning,
        ProvenanceError,
        ProvenanceLedger,
        QuantumNumberError,
        SpectroscopicTransition,
        TransitionQuantumNumbers,
        compute_bytes_sha256,
        compute_dynamic_molecular_mass,
        compute_file_sha256,
        compute_object_sha256,
        get_dynamic_isotope_mass,
    )

__all__: list[str] = [
    "AssignmentTag",
    "DipoleComponents",
    "DipoleType",
    "FitState",
    "HardwareScanChunk",
    "IsotopeRetrievalError",
    "MLSchemaError",
    "MultiTierResourceLimits",
    "ParityStatus",
    "ParityWarning",
    "ProvenanceError",
    "ProvenanceLedger",
    "QuantumNumberError",
    "SpectroscopicTransition",
    "TransitionQuantumNumbers",
    "compute_bytes_sha256",
    "compute_dynamic_molecular_mass",
    "compute_file_sha256",
    "compute_object_sha256",
    "get_dynamic_isotope_mass",
]
