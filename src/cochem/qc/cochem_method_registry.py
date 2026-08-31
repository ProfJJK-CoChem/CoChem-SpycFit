"""
CoChem Quantum Chemistry Method Taxonomy & Auxiliary Fitting Basis Registry.

Provenance:
- Method Matrix [M]: Functional classification, Karlsruhe def2/J and Dunning cc-pVnZ/JK auxiliary sets (REQ-QC-052).
- Grid Presets: GridX 5 with ! NoFinalGridX for routine Opt; GridX 6 for analytical/numerical Freq (REQ-QC-053).
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Dict, List, Literal, Optional, Set, Tuple

from cochem.core.models import (
    IncompatibleAuxiliaryBasisError,
    RIConfigurationPayload,
)


class FunctionalType(str, Enum):
    """Enumeration of quantum chemical functional families and electron correlation treatments."""

    PURE_GGA = "PURE_GGA"
    META_GGA = "META_GGA"
    HYBRID = "HYBRID"
    DOUBLE_HYBRID = "DOUBLE_HYBRID"
    POST_HF = "POST_HF"


# Canonical taxonomy dictionaries
FUNCTIONAL_TAXONOMY: Dict[str, FunctionalType] = {
    # PURE_GGA
    "blyp": FunctionalType.PURE_GGA,
    "pbe": FunctionalType.PURE_GGA,
    "bp86": FunctionalType.PURE_GGA,
    "revpbe": FunctionalType.PURE_GGA,
    "olyp": FunctionalType.PURE_GGA,
    "b97-d": FunctionalType.PURE_GGA,
    "b97-d3": FunctionalType.PURE_GGA,
    "b97-3c": FunctionalType.PURE_GGA,
    # META_GGA
    "tpss": FunctionalType.META_GGA,
    "r2scan": FunctionalType.META_GGA,
    "r2scan-3c": FunctionalType.META_GGA,
    "m06-l": FunctionalType.META_GGA,
    "revm06-l": FunctionalType.META_GGA,
    "scan": FunctionalType.META_GGA,
    # HYBRID
    "b3lyp": FunctionalType.HYBRID,
    "pbe0": FunctionalType.HYBRID,
    "tpssh": FunctionalType.HYBRID,
    "wb97x-d4": FunctionalType.HYBRID,
    "wb97x-d3": FunctionalType.HYBRID,
    "wb97x-3c": FunctionalType.HYBRID,
    "m06-2x": FunctionalType.HYBRID,
    "cam-b3lyp": FunctionalType.HYBRID,
    "b3pw91": FunctionalType.HYBRID,
    "pbeh-3c": FunctionalType.HYBRID,
    "hf-3c": FunctionalType.HYBRID,
    # DOUBLE_HYBRID
    "b2plyp": FunctionalType.DOUBLE_HYBRID,
    "dsd-pbep86": FunctionalType.DOUBLE_HYBRID,
    "revdsd-pbep86": FunctionalType.DOUBLE_HYBRID,
    "pwpb95": FunctionalType.DOUBLE_HYBRID,
    # POST_HF
    "hf": FunctionalType.POST_HF,
    "mp2": FunctionalType.POST_HF,
    "ri-mp2": FunctionalType.POST_HF,
    "dlpno-ccsd(t)": FunctionalType.POST_HF,
    "ccsd(t)": FunctionalType.POST_HF,
    "ccsd": FunctionalType.POST_HF,
    "ccsdt": FunctionalType.POST_HF,
    "caspt2": FunctionalType.POST_HF,
}


def classify_functional(functional_name: str) -> FunctionalType:
    """
    [REQ-QC-052] Classify an electronic structure functional or method into its theoretical family.

    Args:
        functional_name: Functional name string (e.g. 'B3LYP', 'r2SCAN', 'DLPNO-CCSD(T)').

    Returns:
        FunctionalType enum value.
    """
    clean_name = functional_name.strip().lower().replace(" ", "").replace("_", "-")

    if clean_name in FUNCTIONAL_TAXONOMY:
        return FUNCTIONAL_TAXONOMY[clean_name]

    # Heuristic substring classification
    if any(dh in clean_name for dh in ["b2plyp", "dsd-", "pwpb95", "double-hybrid"]):
        return FunctionalType.DOUBLE_HYBRID
    if any(ph in clean_name for ph in ["mp2", "ccsd", "caspt2", "post-hf"]):
        return FunctionalType.POST_HF
    if any(hyb in clean_name for hyb in ["b3", "pbe0", "tpssh", "wb97", "m06-2x", "cam-", "hybrid"]):
        return FunctionalType.HYBRID
    if any(mgga in clean_name for mgga in ["scan", "tpss", "m06-l"]):
        return FunctionalType.META_GGA

    return FunctionalType.PURE_GGA


def get_recommended_grid(is_frequency_job: bool = False) -> Literal["GridX 5", "GridX 6"]:
    """
    [REQ-QC-053] Get recommended numerical integration grid preset.

    Args:
        is_frequency_job: True if configuring a vibrational frequency job.

    Returns:
        GridX level string ('GridX 5' or 'GridX 6').
    """
    return "GridX 6" if is_frequency_job else "GridX 5"


def resolve_auxiliary_basis(
    primary_basis: str,
    functional: str,
    max_atomic_number: int = 1,
    is_freq: bool = False,
) -> RIConfigurationPayload:
    """
    [REQ-QC-052, REQ-QC-053] Resolve the appropriate RI / RIJCOSX auxiliary fitting basis sets and grid level.

    Resolution Rules:
    - Karlsruhe primary (def2-SVP, def2-TZVP, def2-QZVP) -> def2/J (Coulomb), def2-TZVP/C (MP2 correlation).
    - Dunning primary (cc-pVDZ, cc-pVTZ, cc-pVQZ) -> cc-pVnZ/JK (Coulomb/Exchange), cc-pVnZ/C (correlation).
    - Pople (6-31G*, 6-311+G**) or heavy elements (Z > 36) without matching basis -> AutoAux.

    Args:
        primary_basis: Primary orbital basis set string.
        functional: Target functional or electronic method string.
        max_atomic_number: Maximum nuclear charge Z in system (default 1).
        is_freq: Whether the calculation is a frequency evaluation.

    Returns:
        RIConfigurationPayload instance.

    Raises:
        IncompatibleAuxiliaryBasisError: If basis set cannot be resolved.
    """
    family = classify_functional(functional)
    basis_clean = primary_basis.strip()
    basis_lower = basis_clean.lower()

    aux_coulomb: Optional[str] = None
    aux_correlation: Optional[str] = None

    # Karlsruhe def2 families
    if "def2" in basis_lower:
        aux_coulomb = "def2/J"
        if family in (FunctionalType.DOUBLE_HYBRID, FunctionalType.POST_HF):
            if "qzvp" in basis_lower:
                aux_correlation = "def2-QZVPP/C"
            elif "tzvp" in basis_lower:
                aux_correlation = "def2-TZVP/C"
            else:
                aux_correlation = "def2-SVP/C"

    # Dunning cc-pV families
    elif "cc-pv" in basis_lower or "aug-cc-pv" in basis_lower:
        # Extract cardinal zeta (D, T, Q, 5)
        if "cc-pvdz" in basis_lower:
            zeta = "cc-pVDZ"
        elif "cc-pvtz" in basis_lower:
            zeta = "cc-pVTZ"
        elif "cc-pvqz" in basis_lower:
            zeta = "cc-pVQZ"
        elif "cc-pv5z" in basis_lower:
            zeta = "cc-pV5Z"
        else:
            zeta = "cc-pVTZ"

        aux_coulomb = f"{zeta}/JK"
        if family in (FunctionalType.DOUBLE_HYBRID, FunctionalType.POST_HF):
            aux_correlation = f"{zeta}/C"

    # Pople 6-31G / 6-311G or legacy sets -> AutoAux
    elif any(pople in basis_lower for pople in ["6-31", "sto-3g", "lanl", "dzp", "tzp"]):
        aux_coulomb = "AutoAux"
        if family in (FunctionalType.DOUBLE_HYBRID, FunctionalType.POST_HF):
            aux_correlation = "AutoAux"

    # Heavy element fallback
    elif max_atomic_number > 36:
        aux_coulomb = "AutoAux"
        if family in (FunctionalType.DOUBLE_HYBRID, FunctionalType.POST_HF):
            aux_correlation = "AutoAux"

    else:
        # Default to AutoAux if unsupported
        aux_coulomb = "AutoAux"

    if not aux_coulomb:
        raise IncompatibleAuxiliaryBasisError(
            primary_basis=primary_basis,
            functional=functional,
        )

    grid_level = get_recommended_grid(is_frequency_job=is_freq)
    no_final_grid_x = not is_freq

    return RIConfigurationPayload(
        functional_name=functional,
        functional_family=family.value,
        primary_basis=basis_clean,
        aux_coulomb_basis=aux_coulomb,
        aux_correlation_basis=aux_correlation,
        grid_level=grid_level,
        no_final_grid_x=no_final_grid_x,
        allow_exact_coulomb=False,
    )
