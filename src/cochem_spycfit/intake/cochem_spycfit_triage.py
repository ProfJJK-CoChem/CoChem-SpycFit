"""
CoChem-SpycFit: Stage 1.0 Physical & Isotopic Resonance Filter.

Module: cochem_spycfit/intake/cochem_spycfit_triage.py
Role: Physical & Isotopic Resonance Filter and Mass Matrix Transformation Engine.

Capabilities:
1. Mendeleev SQLite Local Resolution & Air-Gap Verification:
   - Resolves the local Mendeleev SQLite database (elements.db) to guarantee offline execution.
   - Verifies SQLite magic binary header and queries isotopic properties without network access.
2. Natural Abundance Filter & Isotopic Triage:
   - Evaluates isotopologue natural abundance product from isotopic constituent abundances.
   - Categorizes molecular systems into Method Matrix v4 Product Classes:
     * Product A: Parent isotopologue (most abundant, de novo)
     * Product B: Singly-substituted isotopologues (semi-experimental template / observable)
     * Product C: Multi-substituted difference targets
     * Trace: Rare sub-threshold isotopologues (< 0.01% abundance) flagged for rejection/warning.
3. Method Matrix v4 Mass Matrix Transformation:
   - Re-weights Cartesian Hessian matrices via M^(-1/2) F_cart M^(-1/2) in strict FP64 precision.
   - Solves the mass-weighted secular equation to obtain harmonic vibrational wavenumbers (cm^-1),
     distinguishing translation/rotation zero modes, genuine vibrational modes, and imaginary saddle modes.
4. Resonance & Polyad Trap:
   - Scans harmonic vibrational frequencies for resonance conditions:
     * Fermi Type I: omega_i ~ 2 * omega_j
     * Fermi Type II: omega_i ~ omega_j + omega_k
     * Coriolis Coupling: omega_i ~ omega_j (near-degenerate mode pairs)
     * Darling-Dennison: 2 * omega_i ~ 2 * omega_j
   - Clusters interacting modes into coupled Polyad blocks for effective Hamiltonian construction.
5. Unified Spectroscopic Triage Pipeline:
   - Executes comprehensive multi-stage spectroscopic intake validation.
   - Generates deterministic SHA-256 provenance hashes for auditable execution records.
6. Pydantic v2 Schemas:
   - Strict data validation models with extra-field forbidding.

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE Version 4, August 2026)
- Task 2 File Inventory & Deliverable Capabilities Manifest
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Triage")

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        CoChemPrecisionError,
    )
except ImportError:

    class CoChemPrecisionError(RuntimeError):  # type: ignore[no-redef]
        """Raised when numerical float precision is violated."""

        pass

    class CoChemIntegrityError(ValueError):  # type: ignore[no-redef]
        """Raised when cryptographic checksum or binary integrity check fails."""

        pass


# =============================================================================
# 1. PHYSICAL CONSTANTS AND UNIT CONVERSION FACTORS
# =============================================================================

# Standard IUPAC/CODATA physical constants (SI and Atomic Units)
HARTREE_TO_JOULE: float = 4.3597447222071e-18  # 1 E_h in Joules (FP64)
BOHR_RADIUS_METER: float = 0.529177210903e-10  # 1 a_0 in meters (FP64)
AMU_TO_KG: float = 1.66053906660e-27  # 1 u in kg (FP64)
SPEED_OF_LIGHT_CM_S: float = 29979245800.0  # Speed of light c in cm/s (FP64)
PLANCK_CONSTANT_JS: float = 6.62607015e-34  # Planck constant h in J*s (FP64)
EV_TO_JOULE: float = 1.602176634e-19  # 1 eV in Joules (FP64)
ANGSTROM_TO_METER: float = 1.0e-10  # 1 Angstrom in meters (FP64)

# Analytical conversion factor from sqrt(Hartree / (bohr^2 * u)) to wavenumber (cm^-1):
# conv = (1 / (2 * pi * c_cm)) * sqrt(E_h / (a_0^2 * u))
# conv = 5140.487143611564 cm^-1
AU_TO_WAVENUMBER_CONVERSION: float = 5140.487143611564


# =============================================================================
# 2. ENUMS
# =============================================================================


class ResonanceType(str, Enum):
    """Vibrational and rovibrational resonance classifications."""

    FERMI_TYPE_I = (
        "Fermi_Type_I"  # Fundamental couples to overtone: omega_i ~ 2 * omega_j
    )
    FERMI_TYPE_II = "Fermi_Type_II"  # Fundamental couples to combination band: omega_i ~ omega_j + omega_k
    DARLING_DENNISON = (
        "Darling_Dennison"  # Overtones/combinations couple: 2 * omega_i ~ 2 * omega_j
    )
    CORIOLIS = "Coriolis"  # Rovibrational Coriolis coupling: omega_i ~ omega_j


class ProductClass(str, Enum):
    """Method Matrix v4 Spectroscopic Product Classifications."""

    PRODUCT_A = "Product A"  # Absolute de novo prediction (parent isotopologue)
    PRODUCT_B = (
        "Product B"  # Semi-experimental / template-anchored (singly-substituted)
    )
    PRODUCT_C = "Product C"  # Difference targets (shifts, conformers, poly-substituted)
    TRACE = "Trace"  # Rare/sub-threshold isotopologues (< 0.01% natural abundance)


class TriageStatus(str, Enum):
    """Triage execution status."""

    PASSED = "PASSED"
    WARNING_RESONANCE = "WARNING_RESONANCE"
    REJECTED_RARE_ISOTOPE = "REJECTED_RARE_ISOTOPE"
    REJECTED_IMAGINARY_FREQUENCY = "REJECTED_IMAGINARY_FREQUENCY"
    ERROR = "ERROR"


# =============================================================================
# 3. PYDANTIC V2 SCHEMAS
# =============================================================================


class IsotopeSpec(BaseModel):
    """Validated atomic isotope specification."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        ..., description="Chemical element symbol (e.g. 'C', 'H', 'O')."
    )
    mass_number: int = Field(
        ..., gt=0, description="Mass number A (number of nucleons)."
    )
    atomic_number: int = Field(
        ..., gt=0, description="Atomic number Z (number of protons)."
    )
    mass_amu: float = Field(
        ...,
        gt=0.0,
        description="Atomic isotopic mass in unified atomic mass units u (FP64).",
    )
    abundance_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Natural isotopic abundance percentage (0.0 to 100.0).",
    )
    nuclear_spin: Optional[str] = Field(
        default=None,
        description="Nuclear spin quantum number string (e.g. '1/2', '1').",
    )
    is_stable: bool = Field(default=True, description="Whether isotope is stable.")


class IsotopologueData(BaseModel):
    """Collection of constituent isotopes defining an isotopologue."""

    model_config = ConfigDict(extra="forbid")

    formula: Optional[str] = Field(
        default=None, description="Chemical formula or label."
    )
    name: Optional[str] = Field(default=None, description="Descriptive system name.")
    isotopes: List[IsotopeSpec] = Field(
        ..., min_length=1, description="Ordered list of constituent atomic isotopes."
    )
    custom_masses: Optional[List[float]] = Field(
        default=None, description="Optional override masses in amu (FP64)."
    )


class IsotopeTriageResult(BaseModel):
    """Output report from isotopic natural abundance analysis."""

    model_config = ConfigDict(extra="forbid")

    isotopologue_name: str = Field(
        ..., description="Canonical formula or label of the isotopologue."
    )
    total_mass_amu: float = Field(
        ..., gt=0.0, description="Total molecular mass in amu (FP64)."
    )
    abundance_percent: float = Field(
        ..., ge=0.0, description="Product natural abundance percentage (FP64)."
    )
    abundance_fraction: float = Field(
        ...,
        ge=0.0,
        description="Product natural abundance fraction (0.0 to 1.0) (FP64).",
    )
    is_observable: bool = Field(
        ..., description="True if abundance satisfies the triage threshold."
    )
    product_class: ProductClass = Field(
        ..., description="Method Matrix v4 Product classification."
    )
    substitution_count: int = Field(
        ..., ge=0, description="Number of isotopic substitutions relative to parent."
    )
    parent_formula: str = Field(..., description="Formula of the parent isotopologue.")
    isotope_details: List[IsotopeSpec] = Field(
        ..., description="Constituent isotope properties."
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Reason for rejection if sub-threshold."
    )


class HarmonicVibrationalResult(BaseModel):
    """Harmonic vibrational analysis output from mass-weighted Cartesian Hessian."""

    model_config = ConfigDict(extra="forbid")

    frequencies_cm1: List[float] = Field(
        ..., description="All computed harmonic frequencies in cm^-1 (FP64)."
    )
    vibrational_frequencies_cm1: List[float] = Field(
        ..., description="Genuine vibrational normal mode frequencies in cm^-1 (FP64)."
    )
    imaginary_frequencies_count: int = Field(
        ..., ge=0, description="Number of imaginary frequencies (saddle points)."
    )
    zero_modes_count: int = Field(
        ..., ge=0, description="Number of translational and rotational zero modes."
    )
    eigenvalues: List[float] = Field(
        ..., description="Secular equation eigenvalues (FP64)."
    )
    mass_weighted_hessian_shape: Tuple[int, int] = Field(
        ..., description="Dimensions of the mass-weighted Hessian."
    )
    reduced_masses_amu: Optional[List[float]] = Field(
        default=None, description="Mode reduced masses in amu (FP64)."
    )
    normal_modes: Optional[List[List[float]]] = Field(
        default=None, description="Normal mode displacement vectors."
    )


class ResonanceCondition(BaseModel):
    """Detected resonance interaction between normal modes."""

    model_config = ConfigDict(extra="forbid")

    resonance_type: ResonanceType = Field(
        ..., description="Classification of resonance interaction."
    )
    mode_indices: List[int] = Field(
        ..., description="Indices of interacting vibrational normal modes (0-indexed)."
    )
    frequencies_cm1: List[float] = Field(
        ..., description="Harmonic frequencies of the involved modes in cm^-1."
    )
    energy_difference_cm1: float = Field(
        ..., ge=0.0, description="Energy detuning |Delta E| in cm^-1 (FP64)."
    )
    description: str = Field(
        ..., description="Spectroscopic description of the resonance condition."
    )
    severity: str = Field(
        ...,
        description="Interaction strength classification: 'STRONG', 'MODERATE', or 'WEAK'.",
    )


class PolyadBlock(BaseModel):
    """Coupled vibrational subspace (Polyad) requiring multi-state diagonalization."""

    model_config = ConfigDict(extra="forbid")

    block_id: int = Field(..., ge=0, description="Unique polyad block index.")
    mode_indices: List[int] = Field(
        ..., description="List of mutually coupled mode indices."
    )
    frequencies_cm1: List[float] = Field(
        ..., description="Frequencies of the modes within this block."
    )
    resonance_types: List[ResonanceType] = Field(
        ..., description="Types of resonance couplings present."
    )
    dimension: int = Field(
        ..., ge=2, description="Dimension of the interacting polyad state subspace."
    )
    frequency_span_cm1: float = Field(
        ..., ge=0.0, description="Max frequency spread across modes in cm^-1."
    )
    recommended_treatment: str = Field(
        ...,
        description="Theoretical treatment recommendation (e.g. Polyad Hamiltonian).",
    )


class ResonanceReport(BaseModel):
    """Comprehensive resonance scan and polyad partition report."""

    model_config = ConfigDict(extra="forbid")

    total_resonances_count: int = Field(
        ..., ge=0, description="Total number of detected resonance conditions."
    )
    fermi_resonances: List[ResonanceCondition] = Field(
        default_factory=list, description="Detected Fermi resonances."
    )
    coriolis_resonances: List[ResonanceCondition] = Field(
        default_factory=list, description="Detected Coriolis resonances."
    )
    darling_dennison_resonances: List[ResonanceCondition] = Field(
        default_factory=list, description="Detected Darling-Dennison resonances."
    )
    polyad_blocks: List[PolyadBlock] = Field(
        default_factory=list, description="Constructed Polyad blocks."
    )
    has_strong_resonances: bool = Field(
        ..., description="True if any strong resonance condition is active."
    )
    recommended_hamiltonian: str = Field(
        ..., description="Recommended Hamiltonian model for downstream VPT2/fitting."
    )


class TriageConfig(BaseModel):
    """Configuration options for spectroscopic intake triage."""

    model_config = ConfigDict(extra="forbid")

    abundance_threshold_percent: float = Field(
        default=0.01,
        ge=0.0,
        description="Natural abundance cut-off percentage (e.g. 0.01% = 1e-4).",
    )
    fermi_type_1_threshold_cm1: float = Field(
        default=30.0,
        ge=0.0,
        description="Max energy detuning for Fermi Type I resonance in cm^-1.",
    )
    fermi_type_2_threshold_cm1: float = Field(
        default=30.0,
        ge=0.0,
        description="Max energy detuning for Fermi Type II resonance in cm^-1.",
    )
    coriolis_threshold_cm1: float = Field(
        default=20.0,
        ge=0.0,
        description="Max frequency difference for Coriolis resonance in cm^-1.",
    )
    darling_dennison_threshold_cm1: float = Field(
        default=30.0,
        ge=0.0,
        description="Max energy detuning for Darling-Dennison resonance in cm^-1.",
    )
    zero_frequency_threshold_cm1: float = Field(
        default=30.0,
        ge=0.0,
        description="Threshold below which modes are treated as translation/rotation zeros.",
    )
    hessian_units: str = Field(
        default="hartree_bohr2",
        description="Units of input Cartesian Hessian ('hartree_bohr2', 'si', 'ev_angstrom2').",
    )
    mendeleev_db_path: Optional[str] = Field(
        default=None,
        description="Optional custom path to local Mendeleev SQLite database.",
    )
    allow_imaginary_frequencies: bool = Field(
        default=False,
        description="Whether to permit imaginary frequencies (transition states) without error.",
    )


class TriageReport(BaseModel):
    """Master audit and triage report for a molecular spectroscopic candidate."""

    model_config = ConfigDict(extra="forbid")

    config: TriageConfig = Field(
        ..., description="Configuration used for this triage run."
    )
    status: TriageStatus = Field(..., description="Overall triage execution status.")
    passed: bool = Field(
        ...,
        description="Whether system passed all triage gates for downstream processing.",
    )
    product_class: ProductClass = Field(
        ..., description="Method Matrix v4 Product classification."
    )
    isotopologue_triage: Optional[IsotopeTriageResult] = Field(
        default=None, description="Isotopic natural abundance triage."
    )
    vibrational_triage: Optional[HarmonicVibrationalResult] = Field(
        default=None, description="Harmonic vibrational analysis result."
    )
    resonance_triage: Optional[ResonanceReport] = Field(
        default=None, description="Resonance and polyad analysis report."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Informational warnings emitted during triage.",
    )
    summary: str = Field(..., description="Human-readable summary of triage decisions.")
    sha256_provenance: str = Field(
        ...,
        description="Cryptographic SHA-256 hash digest of input parameters and outputs.",
    )


# =============================================================================
# 4. MENDELEEV DATABASE RESOLUTION AND AIR-GAP VERIFICATION
# =============================================================================


def resolve_mendeleev_database(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolves the absolute path to the local Mendeleev SQLite database (elements.db).
    Verifies that the database file exists locally, has non-zero size, and begins with
    the valid SQLite format 3 binary header, guaranteeing air-gapped offline operation.

    Args:
        custom_path: Optional user-supplied path to elements.db.

    Returns:
        Path: Absolute resolved path to verified elements.db.

    Raises:
        FileNotFoundError: If the database file does not exist on disk.
        ValueError: If the file is not a valid SQLite database.
    """
    resolved: Optional[Path] = None

    if custom_path is not None:
        p = Path(custom_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Specified Mendeleev database path not found: {p}")
        resolved = p
    else:
        # Check environment variables
        env_path = os.environ.get("COCHEM_MENDELEEV_DB") or os.environ.get(
            "MENDELEEV_DATABASE_PATH"
        )
        if env_path:
            p = Path(env_path).resolve()
            if p.is_file():
                resolved = p

    if resolved is None:
        try:
            import mendeleev.db

            dbpath_str = mendeleev.db.get_package_dbpath()
            p = Path(dbpath_str).resolve()
            if p.is_file():
                resolved = p
        except Exception as exc:
            logger.debug(f"Failed to query mendeleev.db.get_package_dbpath(): {exc}")

    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(
            "Could not resolve local Mendeleev SQLite database (elements.db)."
        )

    # Air-gap binary verification: check file size and SQLite header
    file_size = resolved.stat().st_size
    if file_size < 100000:
        raise ValueError(
            f"Mendeleev database file at {resolved} is corrupted or incomplete ({file_size} bytes)."
        )

    with open(resolved, "rb") as f:
        magic_bytes = f.read(16)

    if not magic_bytes.startswith(b"SQLite format 3\x00"):
        raise ValueError(f"File at {resolved} is not a valid SQLite 3 database.")

    logger.debug(
        f"Resolved air-gapped Mendeleev SQLite database: {resolved} ({file_size} bytes)"
    )
    return resolved


def get_isotope_properties(
    symbol: str,
    mass_number: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> IsotopeSpec:
    """
    Queries local Mendeleev database to retrieve atomic and isotopic properties in FP64 precision.

    Args:
        symbol: Chemical element symbol (e.g. 'C', 'H', 'O', 'Cl').
        mass_number: Nucleon number A. If None, resolves to the most naturally abundant stable isotope.
        db_path: Optional custom path to elements.db.

    Returns:
        IsotopeSpec: Validated Pydantic model containing isotope physics.

    Raises:
        ValueError: If element symbol or mass number is not found.
    """
    resolve_mendeleev_database(db_path)

    from mendeleev import element

    clean_symbol = symbol.strip().capitalize()
    try:
        elem = element(clean_symbol)
    except Exception as exc:
        raise ValueError(
            f"Unknown chemical element symbol '{clean_symbol}': {exc}"
        ) from exc

    atomic_number = int(elem.atomic_number)

    if mass_number is not None:
        target_a = int(mass_number)
        matching = [iso for iso in elem.isotopes if iso.mass_number == target_a]
        if not matching:
            raise ValueError(
                f"Isotope '{clean_symbol}-{target_a}' not found in Mendeleev database."
            )
        selected_iso = matching[0]
    else:
        # Resolve to most abundant isotope
        valid_isotopes = [iso for iso in elem.isotopes if iso.abundance is not None]
        if valid_isotopes:
            selected_iso = max(
                valid_isotopes, key=lambda iso: float(iso.abundance or 0.0)
            )
        else:
            # Fallback to first isotope if no abundances recorded
            selected_iso = elem.isotopes[0]

    # Extract mass and abundance in FP64
    iso_mass = (
        float(selected_iso.mass)
        if selected_iso.mass is not None
        else float(elem.atomic_weight or selected_iso.mass_number)
    )
    iso_abundance = (
        float(selected_iso.abundance) if selected_iso.abundance is not None else 0.0
    )
    spin_str = str(selected_iso.spin) if selected_iso.spin is not None else None
    is_stable = not bool(getattr(selected_iso, "is_radioactive", False))

    return IsotopeSpec(
        symbol=clean_symbol,
        mass_number=int(selected_iso.mass_number),
        atomic_number=atomic_number,
        mass_amu=np.float64(iso_mass),
        abundance_percent=np.float64(iso_abundance),
        nuclear_spin=spin_str,
        is_stable=is_stable,
    )


# =============================================================================
# 5. NATURAL ABUNDANCE FILTER AND ISOTOPIC TRIAGE
# =============================================================================


def parse_isotopologue_input(
    isotopologue: Union[
        IsotopologueData,
        Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]],
        Dict[str, Any],
        str,
    ],
    db_path: Optional[Union[str, Path]] = None,
) -> List[IsotopeSpec]:
    """
    Parses diverse isotopologue representations into a standard list of IsotopeSpec models.

    Supported formats:
    - IsotopologueData Pydantic model
    - Single dict: {"symbol": "C", "mass_number": 12} or {"isotopes": [...]}
    - List of tuples: [("C", 12), ("O", 16), ("O", 16)]
    - List of dicts: [{"symbol": "C", "mass_number": 12}, ...]
    - List of IsotopeSpec models
    - String with hyphenated isotopes: "13C-1H-35Cl" or formula "12C16O2"
    """
    if isinstance(isotopologue, IsotopologueData):
        return list(isotopologue.isotopes)

    if isinstance(isotopologue, dict):
        if "isotopes" in isotopologue and isinstance(isotopologue["isotopes"], list):
            return parse_isotopologue_input(isotopologue["isotopes"], db_path=db_path)
        sym_dict = str(isotopologue.get("symbol") or isotopologue.get("element") or "")
        a_dict = isotopologue.get("mass_number") or isotopologue.get("a")
        if sym_dict:
            mass_num: Optional[int] = int(a_dict) if a_dict is not None else None
            return [get_isotope_properties(sym_dict, mass_num, db_path=db_path)]
        raise ValueError(f"Unrecognized isotope dictionary format: {isotopologue}")

    specs: List[IsotopeSpec] = []

    if isinstance(isotopologue, str):
        text = isotopologue.strip()
        # Case 1: Hyphen-separated isotope tokens, e.g. "13C-1H-35Cl"
        if "-" in text:
            tokens = [t.strip() for t in text.split("-") if t.strip()]
            for tok in tokens:
                match = re.match(r"^(\d+)?([A-Za-z]+)(\d+)?$", tok)
                if match:
                    prefix_a, sym, suffix_a = match.groups()
                    mass_no = (
                        int(prefix_a)
                        if prefix_a
                        else (int(suffix_a) if suffix_a else None)
                    )
                    specs.append(get_isotope_properties(sym, mass_no, db_path=db_path))
                else:
                    raise ValueError(
                        f"Cannot parse isotope token '{tok}' from string '{text}'."
                    )
            return specs

        # Case 2: Space-separated tokens, e.g. "12C 16O2"
        if " " in text:
            tokens = [t.strip() for t in text.split() if t.strip()]
            for tok in tokens:
                specs.extend(parse_isotopologue_input(tok, db_path=db_path))
            return specs

        # Case 3: Parenthesized isotope formula, e.g. "(12C)(16O)2"
        if "(" in text:
            pattern = re.compile(r"(?:\((\d+)?([A-Za-z]+)\)|([A-Z][a-z]*))(\d*)")
            matches = pattern.findall(text)
            if not matches:
                raise ValueError(
                    f"Unrecognized parenthesized isotopologue formula: '{text}'."
                )
            for prefix_a, sym1, sym2, count_str in matches:
                sym = sym1 or sym2
                if not sym:
                    continue
                mass_no = int(prefix_a) if prefix_a else None
                count = int(count_str) if count_str else 1
                iso_prop = get_isotope_properties(sym, mass_no, db_path=db_path)
                for _ in range(count):
                    specs.append(iso_prop)
            return specs

        # Case 4: Standard chemical formula (e.g. "H2O", "CO2") or single token "13C"
        pattern = re.compile(r"(\d+)?([A-Z][a-z]*)(\d*)")
        matches = pattern.findall(text)
        if not matches:
            raise ValueError(f"Unrecognized isotopologue formula string: '{text}'.")
        for prefix_a, sym, count_str in matches:
            if not sym:
                continue
            mass_no = int(prefix_a) if prefix_a else None
            count = int(count_str) if count_str else 1
            iso_prop = get_isotope_properties(sym, mass_no, db_path=db_path)
            for _ in range(count):
                specs.append(iso_prop)
        return specs

    # Sequence of elements/tuples/dicts
    for item in isotopologue:
        if isinstance(item, IsotopeSpec):
            specs.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            t_sym, t_a = item
            specs.append(get_isotope_properties(str(t_sym), int(t_a), db_path=db_path))
        elif isinstance(item, dict):
            d_sym = str(item.get("symbol") or item.get("element") or "")
            d_a = item.get("mass_number") or item.get("a")
            d_mass_no: Optional[int] = int(d_a) if d_a is not None else None
            specs.append(get_isotope_properties(d_sym, d_mass_no, db_path=db_path))
        elif isinstance(item, str):
            specs.extend(parse_isotopologue_input(item, db_path=db_path))
        else:
            raise ValueError(f"Unsupported isotope specification object: {item}")

    return specs


def evaluate_isotopologue_abundance(
    isotopes: Sequence[Union[IsotopeSpec, Tuple[str, int], Dict[str, Any], str]],
    threshold_percent: float = 0.01,
    db_path: Optional[Union[str, Path]] = None,
) -> IsotopeTriageResult:
    """
    Evaluates the natural abundance product of an isotopologue and categorizes it into
    Method Matrix v4 Product Classes (Product A, B, C, or Trace).

    Args:
        isotopes: Constituent atomic isotope definitions.
        threshold_percent: Abundance threshold cut-off percentage (default 0.01% = 1e-4).
        db_path: Optional path to Mendeleev SQLite database.

    Returns:
        IsotopeTriageResult: Structured triage classification.
    """
    specs: List[IsotopeSpec] = []
    for item in isotopes:
        if isinstance(item, IsotopeSpec):
            specs.append(item)
        else:
            parsed = parse_isotopologue_input([item], db_path=db_path)
            specs.extend(parsed)

    if not specs:
        raise ValueError("Cannot evaluate empty isotopologue specification.")

    # Calculate molecular mass and abundance product in FP64
    total_mass = np.float64(sum(s.mass_amu for s in specs))

    # Product of fractional abundances: prod(abundance_percent / 100)
    abundance_fraction = np.float64(1.0)
    for s in specs:
        fraction_i = np.float64(s.abundance_percent) / np.float64(100.0)
        abundance_fraction *= fraction_i

    abundance_percent = np.float64(abundance_fraction * 100.0)

    # Determine parent isotopologue and count substitutions
    substitution_count = 0
    parent_symbols: List[str] = []
    current_symbols: List[str] = []

    for s in specs:
        current_symbols.append(f"{s.mass_number}{s.symbol}")
        parent_iso = get_isotope_properties(s.symbol, None, db_path=db_path)
        parent_symbols.append(f"{parent_iso.mass_number}{parent_iso.symbol}")
        if s.mass_number != parent_iso.mass_number:
            substitution_count += 1

    isotopologue_name = "-".join(current_symbols)
    parent_formula = "-".join(parent_symbols)

    # Classification according to Method Matrix v4 standards:
    # Product A: Parent isotopologue (0 substitutions)
    # Product B: Singly-substituted isotopologue (1 substitution)
    # Product C: Multiply-substituted difference target (>= 2 substitutions)
    # Trace: Natural abundance below threshold_percent
    is_observable = bool(abundance_percent >= threshold_percent)

    if not is_observable:
        prod_class = ProductClass.TRACE
        rejection_reason = f"Natural abundance {abundance_percent:.6e}% is below threshold {threshold_percent:.4f}%."
    elif substitution_count == 0:
        prod_class = ProductClass.PRODUCT_A
        rejection_reason = None
    elif substitution_count == 1:
        prod_class = ProductClass.PRODUCT_B
        rejection_reason = None
    else:
        prod_class = ProductClass.PRODUCT_C
        rejection_reason = None

    return IsotopeTriageResult(
        isotopologue_name=isotopologue_name,
        total_mass_amu=total_mass,
        abundance_percent=abundance_percent,
        abundance_fraction=abundance_fraction,
        is_observable=is_observable,
        product_class=prod_class,
        substitution_count=substitution_count,
        parent_formula=parent_formula,
        isotope_details=specs,
        rejection_reason=rejection_reason,
    )


def check_isotope_abundance(
    isotopologue_data: Union[
        IsotopologueData,
        Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]],
        str,
        Dict[str, Any],
    ],
    threshold_percent: float = 0.01,
    db_path: Optional[Union[str, Path]] = None,
) -> IsotopeTriageResult:
    """
    High-level entry point to evaluate natural abundance and triage rare isotopologues.
    """
    specs = parse_isotopologue_input(isotopologue_data, db_path=db_path)
    return evaluate_isotopologue_abundance(
        specs, threshold_percent=threshold_percent, db_path=db_path
    )


# =============================================================================
# 6. METHOD MATRIX V4 MASS MATRIX TRANSFORMATION (M^(-1/2) F_cart M^(-1/2))
# =============================================================================


def mass_matrix_transformation(
    F_cart: Union[np.ndarray, Sequence[Sequence[float]]],
    masses: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """
    Implements the Method Matrix v4 Mass Matrix Transformation:
    H_mw = M^(-1/2) * F_cart * M^(-1/2)

    Re-weights the Cartesian force constant Hessian matrix by atomic/coordinate masses
    in strict FP64 precision.

    Args:
        F_cart: Square Cartesian Hessian matrix of shape (3N, 3N) or (D, D).
        masses: Array of atomic masses (length N, broadcast to 3N) or coordinate masses (length 3N).

    Returns:
        np.ndarray: Mass-weighted Hessian matrix H_mw of dtype np.float64.

    Raises:
        ValueError: On dimension mismatch or non-square matrix.
        CoChemPrecisionError: If input cannot be converted to FP64.
    """
    f_arr = np.asarray(F_cart, dtype=np.float64)
    if f_arr.ndim != 2 or f_arr.shape[0] != f_arr.shape[1]:
        raise ValueError(
            f"Cartesian Hessian F_cart must be a 2D square matrix, got shape {f_arr.shape}."
        )

    dim = f_arr.shape[0]
    m_arr = np.asarray(masses, dtype=np.float64).flatten()

    if m_arr.size == dim:
        coord_masses = m_arr
    elif m_arr.size * 3 == dim:
        # Atomic masses provided (length N) -> expand 3 times for (x, y, z) coordinates
        coord_masses = np.repeat(m_arr, 3)
    else:
        raise ValueError(
            f"Mass vector size ({m_arr.size}) is incompatible with Hessian dimension ({dim}). "
            f"Expected either {dim} coordinate masses or {dim // 3} atomic masses."
        )

    if np.any(coord_masses <= 0.0):
        raise ValueError("All atomic/coordinate masses must be strictly positive.")

    # Enforce strict symmetry on input Cartesian Hessian: F = 0.5 * (F + F^T)
    f_sym = 0.5 * (f_arr + f_arr.T)

    # Compute inverse square root mass vector M^(-1/2) in FP64
    inv_sqrt_mass = 1.0 / np.sqrt(coord_masses)

    # Mass-weighted transformation: H_mw = M^(-1/2) F_cart M^(-1/2)
    # Element-wise: H_mw[i, j] = F[i, j] / sqrt(M_i * M_j)
    h_mw = f_sym * np.outer(inv_sqrt_mass, inv_sqrt_mass)

    return np.asarray(h_mw, dtype=np.float64)


def compute_harmonic_vibrational_frequencies(
    F_cart: Union[np.ndarray, Sequence[Sequence[float]]],
    masses: Union[Sequence[float], np.ndarray],
    units: str = "hartree_bohr2",
    zero_threshold_cm1: float = 30.0,
) -> HarmonicVibrationalResult:
    """
    Computes harmonic vibrational frequencies (cm^-1) and normal modes by diagonalizing
    the mass-weighted Cartesian Hessian matrix in strict FP64 precision.

    Conversion units:
    - 'hartree_bohr2': F_cart in Hartree / bohr^2, masses in unified atomic mass units u.
                       Frequency factor = 5140.487143611564 cm^-1.
    - 'si': F_cart in N/m (J/m^2), masses in kg.
            Frequency factor = 1 / (2 * pi * c_cm) cm^-1.
    - 'ev_angstrom2': F_cart in eV / Angstrom^2, masses in u.

    Args:
        F_cart: Cartesian Hessian matrix.
        masses: Atomic or coordinate masses.
        units: Unit convention for input Hessian ('hartree_bohr2', 'si', 'ev_angstrom2').
        zero_threshold_cm1: Threshold below which modes are considered translation/rotation zeros.

    Returns:
        HarmonicVibrationalResult: Complete vibrational spectrum report.
    """
    h_mw = mass_matrix_transformation(F_cart, masses)
    dim = h_mw.shape[0]

    # Diagonalize symmetric mass-weighted Hessian matrix (FP64)
    eigenvalues, eigenvectors = np.linalg.eigh(h_mw)

    # Determine conversion factor from eigenvalue to wavenumber (cm^-1)
    if units.lower() in ["hartree_bohr2", "hartree/bohr^2", "au"]:
        # lambda in Hartree / (bohr^2 * u)
        # freq_cm1 = sign(lambda) * sqrt(|lambda|) * AU_TO_WAVENUMBER_CONVERSION
        scale_factor = AU_TO_WAVENUMBER_CONVERSION
    elif units.lower() in ["si", "n/m", "j/m2"]:
        # lambda in s^-2 (N / (m * kg))
        scale_factor = 1.0 / (2.0 * math.pi * SPEED_OF_LIGHT_CM_S)
    elif units.lower() in ["ev_angstrom2", "ev/a^2"]:
        # Convert eV/Angstrom^2/u -> Hartree/bohr^2/u
        ev_to_hartree = EV_TO_JOULE / HARTREE_TO_JOULE
        angstrom_to_bohr = ANGSTROM_TO_METER / BOHR_RADIUS_METER
        unit_scale = math.sqrt(ev_to_hartree / (angstrom_to_bohr**2))
        scale_factor = AU_TO_WAVENUMBER_CONVERSION * unit_scale
    else:
        raise ValueError(
            f"Unsupported Hessian unit system '{units}'. Supported: 'hartree_bohr2', 'si', 'ev_angstrom2'."
        )

    frequencies_cm1: List[float] = []
    vibrational_frequencies_cm1: List[float] = []
    imaginary_count = 0
    zero_count = 0

    for val in eigenvalues:
        if val < -1e-8:
            # Imaginary frequency mode (saddle point)
            imag_freq = -float(scale_factor * math.sqrt(abs(val)))
            frequencies_cm1.append(imag_freq)
            imaginary_count += 1
        elif abs(val) <= 1e-8:
            frequencies_cm1.append(0.0)
            zero_count += 1
        else:
            freq = float(scale_factor * math.sqrt(val))
            frequencies_cm1.append(freq)
            if freq < zero_threshold_cm1:
                zero_count += 1
            else:
                vibrational_frequencies_cm1.append(freq)

    # Sort genuine vibrational frequencies in ascending order
    vibrational_frequencies_cm1.sort()

    # Calculate reduced masses for normal modes
    m_arr = np.asarray(masses, dtype=np.float64).flatten()
    coord_masses = m_arr if m_arr.size == dim else np.repeat(m_arr, 3)
    inv_sqrt_m = 1.0 / np.sqrt(coord_masses)

    reduced_masses: List[float] = []
    normal_mode_displacements: List[List[float]] = []

    for k in range(dim):
        # Mass-weighted eigenvector L_k
        l_k = eigenvectors[:, k]
        # Cartesian displacement Q_cart = M^(-1/2) * L_k
        q_cart = l_k * inv_sqrt_m
        norm_q = float(np.linalg.norm(q_cart))
        if norm_q > 1e-12:
            q_cart_norm = q_cart / norm_q
            # Reduced mass mu_k = sum_i(m_i * q_cart_norm_i^2) = 1 / sum_i(L_ik^2 / m_i) = 1 / norm_q^2
            mu_k_val: float = float(np.sum(coord_masses * (q_cart_norm**2)))
        else:
            q_cart_norm = q_cart
            mu_k_val = 1.0

        reduced_masses.append(float(mu_k_val))
        normal_mode_displacements.append([float(x) for x in q_cart_norm])

    return HarmonicVibrationalResult(
        frequencies_cm1=[float(f) for f in frequencies_cm1],
        vibrational_frequencies_cm1=vibrational_frequencies_cm1,
        imaginary_frequencies_count=imaginary_count,
        zero_modes_count=zero_count,
        eigenvalues=[float(v) for v in eigenvalues],
        mass_weighted_hessian_shape=(dim, dim),
        reduced_masses_amu=reduced_masses,
        normal_modes=normal_mode_displacements,
    )


# =============================================================================
# 7. RESONANCE & POLYAD TRAP
# =============================================================================


def scan_fermi_resonances(
    frequencies: Sequence[float],
    type1_tol: float = 30.0,
    type2_tol: float = 30.0,
) -> List[ResonanceCondition]:
    """
    Scans harmonic vibrational frequencies for Fermi resonance conditions:
    - Fermi Type I: omega_i ~ 2 * omega_j
    - Fermi Type II (triad): omega_i ~ omega_j + omega_k

    Args:
        frequencies: List of fundamental vibrational frequencies in cm^-1.
        type1_tol: Max energy detuning |omega_i - 2*omega_j| in cm^-1.
        type2_tol: Max energy detuning |omega_i - (omega_j + omega_k)| in cm^-1.

    Returns:
        List[ResonanceCondition]: Detected Fermi resonance interactions.
    """
    valid_freqs = [(idx, float(f)) for idx, f in enumerate(frequencies) if f > 0.0]
    resonances: List[ResonanceCondition] = []

    # 1. Fermi Type I Scan: omega_i ~ 2 * omega_j
    for i, w_i in valid_freqs:
        for j, w_j in valid_freqs:
            if i == j:
                continue
            detuning = abs(w_i - 2.0 * w_j)
            if detuning <= type1_tol:
                severity = (
                    "STRONG"
                    if detuning <= 10.0
                    else ("MODERATE" if detuning <= 20.0 else "WEAK")
                )
                desc = (
                    f"Fermi Type I: Mode {i} ({w_i:.1f} cm^-1) couples to overtone 2*nu_{j} "
                    f"({2.0 * w_j:.1f} cm^-1), |Delta E| = {detuning:.2f} cm^-1"
                )
                resonances.append(
                    ResonanceCondition(
                        resonance_type=ResonanceType.FERMI_TYPE_I,
                        mode_indices=[i, j],
                        frequencies_cm1=[w_i, w_j],
                        energy_difference_cm1=float(detuning),
                        description=desc,
                        severity=severity,
                    )
                )

    # 2. Fermi Type II Scan: omega_i ~ omega_j + omega_k (with j < k)
    n = len(valid_freqs)
    for idx_i, (i, w_i) in enumerate(valid_freqs):
        for idx_j in range(n):
            for idx_k in range(idx_j + 1, n):
                j, w_j = valid_freqs[idx_j]
                k, w_k = valid_freqs[idx_k]
                if i in (j, k):
                    continue
                combination = w_j + w_k
                detuning = abs(w_i - combination)
                if detuning <= type2_tol:
                    severity = (
                        "STRONG"
                        if detuning <= 10.0
                        else ("MODERATE" if detuning <= 20.0 else "WEAK")
                    )
                    desc = (
                        f"Fermi Type II: Mode {i} ({w_i:.1f} cm^-1) couples to combination band "
                        f"nu_{j}+nu_{k} ({combination:.1f} cm^-1), |Delta E| = {detuning:.2f} cm^-1"
                    )
                    resonances.append(
                        ResonanceCondition(
                            resonance_type=ResonanceType.FERMI_TYPE_II,
                            mode_indices=[i, j, k],
                            frequencies_cm1=[w_i, w_j, w_k],
                            energy_difference_cm1=float(detuning),
                            description=desc,
                            severity=severity,
                        )
                    )

    return resonances


def scan_coriolis_resonances(
    frequencies: Sequence[float],
    coriolis_tol: float = 20.0,
) -> List[ResonanceCondition]:
    """
    Scans harmonic vibrational frequencies for Coriolis resonance conditions between
    nearly degenerate fundamental modes (omega_i ~ omega_j).

    Args:
        frequencies: List of fundamental vibrational frequencies in cm^-1.
        coriolis_tol: Max frequency difference |omega_i - omega_j| in cm^-1.

    Returns:
        List[ResonanceCondition]: Detected Coriolis resonance interactions.
    """
    valid_freqs = [(idx, float(f)) for idx, f in enumerate(frequencies) if f > 0.0]
    resonances: List[ResonanceCondition] = []
    n = len(valid_freqs)

    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            i, w_i = valid_freqs[idx_i]
            j, w_j = valid_freqs[idx_j]
            diff = abs(w_i - w_j)
            if diff <= coriolis_tol:
                severity = (
                    "STRONG"
                    if diff <= 5.0
                    else ("MODERATE" if diff <= 10.0 else "WEAK")
                )
                desc = (
                    f"Coriolis Resonance: Mode {i} ({w_i:.1f} cm^-1) and Mode {j} ({w_j:.1f} cm^-1) "
                    f"are near-degenerate, |Delta E| = {diff:.2f} cm^-1"
                )
                resonances.append(
                    ResonanceCondition(
                        resonance_type=ResonanceType.CORIOLIS,
                        mode_indices=[i, j],
                        frequencies_cm1=[w_i, w_j],
                        energy_difference_cm1=float(diff),
                        description=desc,
                        severity=severity,
                    )
                )

    return resonances


def scan_darling_dennison_resonances(
    frequencies: Sequence[float],
    dd_tol: float = 30.0,
) -> List[ResonanceCondition]:
    """
    Scans overtone states for Darling-Dennison resonance conditions (2*omega_i ~ 2*omega_j).

    Args:
        frequencies: List of fundamental vibrational frequencies in cm^-1.
        dd_tol: Max energy detuning |2*omega_i - 2*omega_j| in cm^-1.

    Returns:
        List[ResonanceCondition]: Detected Darling-Dennison resonance interactions.
    """
    valid_freqs = [(idx, float(f)) for idx, f in enumerate(frequencies) if f > 0.0]
    resonances: List[ResonanceCondition] = []
    n = len(valid_freqs)

    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            i, w_i = valid_freqs[idx_i]
            j, w_j = valid_freqs[idx_j]
            diff = abs(2.0 * w_i - 2.0 * w_j)
            if diff <= dd_tol:
                severity = (
                    "STRONG"
                    if diff <= 10.0
                    else ("MODERATE" if diff <= 20.0 else "WEAK")
                )
                desc = (
                    f"Darling-Dennison Resonance: Overtone 2*nu_{i} ({2.0 * w_i:.1f} cm^-1) and "
                    f"2*nu_{j} ({2.0 * w_j:.1f} cm^-1), |Delta E| = {diff:.2f} cm^-1"
                )
                resonances.append(
                    ResonanceCondition(
                        resonance_type=ResonanceType.DARLING_DENNISON,
                        mode_indices=[i, j],
                        frequencies_cm1=[w_i, w_j],
                        energy_difference_cm1=float(diff),
                        description=desc,
                        severity=severity,
                    )
                )

    return resonances


def identify_polyad_blocks(
    resonances: Sequence[ResonanceCondition],
    frequencies: Sequence[float],
) -> List[PolyadBlock]:
    """
    Clusters mutually resonant vibrational modes into coupled Polyad blocks using
    connected component graph partitioning.

    Args:
        resonances: All detected resonance interactions.
        frequencies: Normal mode frequencies in cm^-1.

    Returns:
        List[PolyadBlock]: Partitioned coupled polyad blocks.
    """
    if not resonances:
        return []

    # Build adjacency list across interacting mode indices
    adjacency: Dict[int, Set[int]] = {}
    mode_resonance_types: Dict[int, Set[ResonanceType]] = {}

    for r in resonances:
        for m in r.mode_indices:
            if m not in adjacency:
                adjacency[m] = set()
                mode_resonance_types[m] = set()
            mode_resonance_types[m].add(r.resonance_type)
            for other_m in r.mode_indices:
                if m != other_m:
                    adjacency[m].add(other_m)

    # Find connected components via breadth-first search
    visited: Set[int] = set()
    components: List[List[int]] = []

    for node in sorted(adjacency.keys()):
        if node in visited:
            continue
        comp = []
        queue = [node]
        visited.add(node)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adjacency.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(comp))

    # Construct PolyadBlock models
    polyad_blocks: List[PolyadBlock] = []
    freq_list = list(frequencies)

    for block_idx, comp in enumerate(components):
        block_freqs = [freq_list[m] for m in comp if m < len(freq_list)]
        span = float(max(block_freqs) - min(block_freqs)) if block_freqs else 0.0

        # Collect resonance types active within this component
        types_in_block: Set[ResonanceType] = set()
        for r in resonances:
            if any(m in comp for m in r.mode_indices):
                types_in_block.add(r.resonance_type)

        dimension = max(len(comp), 2)
        treatment = (
            f"Coupled Polyad Diagonalization (Block Dimension {dimension}x{dimension}) "
            f"via Effective Rovibrational Hamiltonian."
        )

        polyad_blocks.append(
            PolyadBlock(
                block_id=block_idx,
                mode_indices=comp,
                frequencies_cm1=block_freqs,
                resonance_types=sorted(list(types_in_block), key=lambda x: x.value),
                dimension=dimension,
                frequency_span_cm1=span,
                recommended_treatment=treatment,
            )
        )

    return polyad_blocks


def trap_resonances(
    frequencies: Union[Sequence[float], np.ndarray],
    fermi_tol: float = 30.0,
    coriolis_tol: float = 20.0,
    dd_tol: float = 30.0,
) -> ResonanceReport:
    """
    Master entry point for Resonance & Polyad Trap.
    Scans theoretical harmonic frequencies for Fermi, Coriolis, and Darling-Dennison
    resonances and groups interacting modes into coupled Polyad blocks.

    Args:
        frequencies: Fundamental harmonic frequencies in cm^-1.
        fermi_tol: Detuning tolerance for Fermi resonances (cm^-1).
        coriolis_tol: Detuning tolerance for Coriolis resonances (cm^-1).
        dd_tol: Detuning tolerance for Darling-Dennison resonances (cm^-1).

    Returns:
        ResonanceReport: Comprehensive report of detected resonances and polyad blocks.
    """
    freq_list = [float(f) for f in frequencies]

    fermi_res = scan_fermi_resonances(
        freq_list, type1_tol=fermi_tol, type2_tol=fermi_tol
    )
    coriolis_res = scan_coriolis_resonances(freq_list, coriolis_tol=coriolis_tol)
    dd_res = scan_darling_dennison_resonances(freq_list, dd_tol=dd_tol)

    all_res = fermi_res + coriolis_res + dd_res
    polyads = identify_polyad_blocks(all_res, freq_list)

    has_strong = any(r.severity == "STRONG" for r in all_res)

    if has_strong:
        recommended_hamiltonian = (
            "De-perturbed Second-Order Vibrational Perturbation Theory (DVPT2) "
            "with Coupled Polyad Effective Hamiltonian Matrix Diagonalization."
        )
    elif all_res:
        recommended_hamiltonian = (
            "Standard Second-Order Vibrational Perturbation Theory (VPT2) "
            "with Resonance Detuning Monitoring."
        )
    else:
        recommended_hamiltonian = (
            "Standard Rayleigh-Schroedinger Perturbation Theory (VPT2)."
        )

    return ResonanceReport(
        total_resonances_count=len(all_res),
        fermi_resonances=fermi_res,
        coriolis_resonances=coriolis_res,
        darling_dennison_resonances=dd_res,
        polyad_blocks=polyads,
        has_strong_resonances=has_strong,
        recommended_hamiltonian=recommended_hamiltonian,
    )


# =============================================================================
# 8. PROVENANCE HASHING AND MASTER SPECTROSCOPIC TRIAGE PIPELINE
# =============================================================================


def calculate_triage_provenance_hash(payload: Any) -> str:
    """
    Generates a deterministic SHA-256 cryptographic hash digest over spectroscopic triage data.
    """
    if isinstance(payload, BaseModel):
        dump_bytes = json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode(
            "utf-8"
        )
    elif isinstance(payload, (dict, list, tuple)):
        dump_bytes = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    elif isinstance(payload, np.ndarray):
        dump_bytes = payload.tobytes()
    else:
        dump_bytes = str(payload).encode("utf-8")

    return hashlib.sha256(dump_bytes).hexdigest()


def triage_spectroscopic_system(
    isotopologue: Optional[Union[IsotopologueData, Sequence, str, Dict]] = None,
    F_cart: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    masses: Optional[Union[Sequence[float], np.ndarray]] = None,
    config: Optional[TriageConfig] = None,
) -> TriageReport:
    """
    Master unified pipeline for Stage 1.0 Physical & Isotopic Resonance Triage.

    Executes:
    1. Natural abundance isotopic triage against Mendeleev air-gapped database.
    2. Mass matrix transformation M^(-1/2) F_cart M^(-1/2) and harmonic frequency analysis.
    3. Resonance & Polyad Trap scanning for Fermi, Coriolis, and Darling-Dennison couplings.
    4. Deterministic cryptographic SHA-256 provenance hashing.

    Args:
        isotopologue: Constituent isotope definitions or formula string.
        F_cart: Cartesian Hessian matrix (optional).
        masses: Atomic or coordinate masses (optional).
        config: Triage configuration parameters (optional).

    Returns:
        TriageReport: Master spectroscopic triage report.
    """
    cfg = config or TriageConfig()
    warnings: List[str] = []
    overall_status = TriageStatus.PASSED
    product_class = ProductClass.PRODUCT_A

    # 1. Isotopic Triage
    iso_result: Optional[IsotopeTriageResult] = None
    if isotopologue is not None:
        try:
            iso_result = check_isotope_abundance(
                isotopologue_data=isotopologue,
                threshold_percent=cfg.abundance_threshold_percent,
                db_path=cfg.mendeleev_db_path,
            )
            product_class = iso_result.product_class
            if not iso_result.is_observable:
                overall_status = TriageStatus.REJECTED_RARE_ISOTOPE
                warnings.append(
                    f"Isotopologue '{iso_result.isotopologue_name}' abundance ({iso_result.abundance_percent:.4e}%) "
                    f"is below threshold ({cfg.abundance_threshold_percent:.4f}%)."
                )
        except Exception as exc:
            logger.error(f"Error during isotopic abundance triage: {exc}")
            overall_status = TriageStatus.ERROR
            warnings.append(f"Isotopic triage error: {exc}")

    # 2. Harmonic Vibrational Triage
    vib_result: Optional[HarmonicVibrationalResult] = None
    if F_cart is not None:
        try:
            # Resolve masses from isotopologue if not explicitly supplied
            resolved_masses = masses
            if resolved_masses is None and iso_result is not None:
                resolved_masses = [iso.mass_amu for iso in iso_result.isotope_details]

            if resolved_masses is None:
                raise ValueError(
                    "Masses must be provided directly or inferred from isotopologue specification."
                )

            vib_result = compute_harmonic_vibrational_frequencies(
                F_cart=F_cart,
                masses=resolved_masses,
                units=cfg.hessian_units,
                zero_threshold_cm1=cfg.zero_frequency_threshold_cm1,
            )

            if vib_result.imaginary_frequencies_count > 0:
                if not cfg.allow_imaginary_frequencies:
                    overall_status = TriageStatus.REJECTED_IMAGINARY_FREQUENCY
                    warnings.append(
                        f"Detected {vib_result.imaginary_frequencies_count} imaginary frequency modes. "
                        "System is a transition state or saddle point."
                    )
                else:
                    warnings.append(
                        f"Permitted {vib_result.imaginary_frequencies_count} imaginary modes under configuration."
                    )
        except Exception as exc:
            logger.error(f"Error during vibrational frequency calculation: {exc}")
            overall_status = TriageStatus.ERROR
            warnings.append(f"Vibrational analysis error: {exc}")

    # 3. Resonance & Polyad Trap
    res_result: Optional[ResonanceReport] = None
    if vib_result is not None and vib_result.vibrational_frequencies_cm1:
        try:
            res_result = trap_resonances(
                frequencies=vib_result.vibrational_frequencies_cm1,
                fermi_tol=cfg.fermi_type_1_threshold_cm1,
                coriolis_tol=cfg.coriolis_threshold_cm1,
                dd_tol=cfg.darling_dennison_threshold_cm1,
            )
            if (
                res_result.has_strong_resonances
                and overall_status == TriageStatus.PASSED
            ):
                overall_status = TriageStatus.WARNING_RESONANCE
                warnings.append(
                    f"Strong vibrational resonances detected ({res_result.total_resonances_count} total). "
                    f"Partitioned into {len(res_result.polyad_blocks)} Polyad blocks."
                )
        except Exception as exc:
            logger.error(f"Error during resonance scanning: {exc}")
            overall_status = TriageStatus.ERROR
            warnings.append(f"Resonance analysis error: {exc}")

    # Determine passing status
    passed = overall_status in [TriageStatus.PASSED, TriageStatus.WARNING_RESONANCE]

    # Assemble summary
    summary_parts = [
        f"Triage Status: {overall_status.value} (Product Class: {product_class.value})."
    ]
    if iso_result:
        summary_parts.append(
            f"Isotopologue: {iso_result.isotopologue_name} (Mass: {iso_result.total_mass_amu:.4f} amu, "
            f"Abundance: {iso_result.abundance_percent:.4f}%)."
        )
    if vib_result:
        summary_parts.append(
            f"Vibrational Modes: {len(vib_result.vibrational_frequencies_cm1)} "
            f"(Imaginary: {vib_result.imaginary_frequencies_count}, Zero modes: {vib_result.zero_modes_count})."
        )
    if res_result:
        summary_parts.append(
            f"Resonances: {res_result.total_resonances_count} detected "
            f"(Polyad blocks: {len(res_result.polyad_blocks)})."
        )
    summary_text = " ".join(summary_parts)

    # Compute provenance hash
    provenance_dict = {
        "status": overall_status.value,
        "product_class": product_class.value,
        "passed": passed,
        "iso": iso_result.model_dump(mode="json") if iso_result else None,
        "vib": {
            "frequencies": vib_result.frequencies_cm1,
            "imag_count": vib_result.imaginary_frequencies_count,
        }
        if vib_result
        else None,
        "res": {
            "total_count": res_result.total_resonances_count,
            "polyad_count": len(res_result.polyad_blocks),
        }
        if res_result
        else None,
    }
    provenance_hash = calculate_triage_provenance_hash(provenance_dict)

    return TriageReport(
        config=cfg,
        status=overall_status,
        passed=passed,
        product_class=product_class,
        isotopologue_triage=iso_result,
        vibrational_triage=vib_result,
        resonance_triage=res_result,
        warnings=warnings,
        summary=summary_text,
        sha256_provenance=provenance_hash,
    )
