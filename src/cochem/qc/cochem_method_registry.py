"""
CoChem Quantum Chemistry Method Taxonomy & Auxiliary Fitting Basis Registry.

Provenance:
- Method Matrix [M]: Functional classification, Karlsruhe def2/J and Dunning cc-pVnZ/JK auxiliary sets (REQ-QC-052).
- Grid Presets: GridX 5 with ! NoFinalGridX for routine Opt; GridX 6 for analytical/numerical Freq (REQ-QC-053).
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from cochem.core.models import (
    IncompatibleAuxiliaryBasisError,
    RIConfigurationPayload,
)


class DispersionConfig(BaseModel):
    """Configuration for empirical dispersion corrections."""
    model_config = ConfigDict(frozen=True)
    scheme: Optional[str] = Field(default=None, description="Dispersion scheme, e.g. D3BJ, D4, None")
    damping: Optional[str] = Field(default=None, description="Damping function, e.g. BJ, zero")
    s6: Optional[float] = Field(default=None)
    s8: Optional[float] = Field(default=None)
    a1: Optional[float] = Field(default=None)
    a2: Optional[float] = Field(default=None)


class SolvationSpec(BaseModel):
    """Specification of implicit continuum solvation model."""
    model_config = ConfigDict(frozen=True)
    model_name: str = Field(description="Solvation model name, e.g. SMD, CPCM")
    solvent_name: str = Field(description="Solvent name, e.g. water, benzene")
    dielectric_constant: Optional[float] = Field(default=None, description="Static dielectric constant eps")
    probe_radius_angstrom: Optional[float] = Field(default=None, description="Solvent probe radius in Angstroms")


class LevelOfTheory(BaseModel):
    """Canonical representation of a quantum mechanical Level of Theory."""
    model_config = ConfigDict(frozen=True)
    hamiltonian: str = Field(..., description="Electronic Hamiltonian or DFT functional, e.g. B3LYP, wB97X-D4")
    basis_set: str = Field(..., description="Primary GTO orbital basis set, e.g. def2-TZVP")
    auxiliary_basis_set: Optional[str] = Field(default=None, description="Auxiliary Coulomb/correlation fitting basis set")
    dispersion: Optional[Union[str, DispersionConfig]] = Field(default=None, description="Dispersion correction or configuration")
    grid: Optional[str] = Field(default="DefGrid2", description="Numerical integration quadrature grid")
    solvation: Optional[Union[str, SolvationSpec]] = Field(default=None, description="Solvation model or specification")
    max_gradient_residual_au: Optional[float] = Field(default=None, ge=0.0)


class IncompatibleLevelOfTheoryError(ValueError):
    """Raised when an unphysical or incompatible Level-of-Theory combination is requested."""
    pass


def compute_lot_fingerprint(lot: Union[LevelOfTheory, Dict[str, Any]]) -> str:
    """
    [D] Generate deterministic, RFC 8785-compliant canonical SHA-256 fingerprint for a LevelOfTheory.
    """
    if isinstance(lot, LevelOfTheory):
        lot_dict = lot.model_dump(exclude_none=True)
    elif isinstance(lot, dict):
        lot_dict = {k: v for k, v in lot.items() if v is not None}
    else:
        raise TypeError(f"Expected LevelOfTheory or dict, got {type(lot)}")

    norm_dict = {}
    for k, v in lot_dict.items():
        if isinstance(v, str):
            norm_dict[k.lower()] = v.strip().lower()
        elif isinstance(v, dict):
            norm_dict[k.lower()] = {ik.lower(): (iv.strip().lower() if isinstance(iv, str) else iv) for ik, iv in v.items()}
        else:
            norm_dict[k.lower()] = v

    canonical_json = json.dumps(norm_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# Alias for compute_lot_fingerprint
compute_canonical_lot_hash = compute_lot_fingerprint


class FunctionalType(str, Enum):
    """Enumeration of quantum chemical functional families and electron correlation treatments."""

    PURE_GGA = "PURE_GGA"
    META_GGA = "META_GGA"
    HYBRID = "HYBRID"
    GLOBAL_HYBRID = "GLOBAL_HYBRID"
    RANGE_SEPARATED_HYBRID = "RANGE_SEPARATED_HYBRID"
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
    "b3lyp": FunctionalType.GLOBAL_HYBRID,
    "pbe0": FunctionalType.GLOBAL_HYBRID,
    "tpssh": FunctionalType.GLOBAL_HYBRID,
    "wb97x-d4": FunctionalType.RANGE_SEPARATED_HYBRID,
    "wb97x-d3": FunctionalType.RANGE_SEPARATED_HYBRID,
    "wb97x-3c": FunctionalType.RANGE_SEPARATED_HYBRID,
    "m06-2x": FunctionalType.GLOBAL_HYBRID,
    "cam-b3lyp": FunctionalType.RANGE_SEPARATED_HYBRID,
    "b3pw91": FunctionalType.GLOBAL_HYBRID,
    "pbeh-3c": FunctionalType.GLOBAL_HYBRID,
    "hf-3c": FunctionalType.GLOBAL_HYBRID,
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
