Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_1_Prompt4_triage.md.
Original prompt:
# Task 2 (Part 1): Stage 1 - cochem_spycfit_triage.py

**Objective:**
Create the `cochem_spycfit_triage.py` file to act as the Physical & Isotopic Resonance Filter.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_triage.py`

**Instructions:**
You are the `cochem-coder` agent. Implement `cochem_spycfit_triage.py`.
It must:
1. Integrate with `mendeleev` to flag transitions assigned to extremely rare isotopologues. Pre-vendor/resolve the `mendeleev` SQLite database locally during Stage 0 to prevent offline internet downloads.
2. Implement the Method Matrix v4 Mass Matrix Transformation ($M^{-1/2} F_{cart} M^{-1/2}$) to re-weight Cartesian Hessians.
3. Implement a Resonance & Polyad Trap that scans theoretical harmonic frequencies for Fermi/Coriolis resonance conditions and flags them for coupled Polyad matrix construction.

**Constraints:**
- No placeholders, fake data, dummy logic, or mocked implementations. Use the real `mendeleev` library API and numpy operations for transformations.

**Proposed Snippet Outline:**
```python
from mendeleev import element
import numpy as np

def check_isotope_abundance(isotopologue_data):
    # logic using mendeleev to find rare abundance
    pass

def mass_matrix_transformation(F_cart, masses):
    # $M^{-1/2} F_{cart} M^{-1/2}$ logic
    pass

def trap_resonances(frequencies):
    # scan for resonance conditions
    pass
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopyTransition,
    ValidationReport,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)
from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)

__all__ = [
    # Ingest
    "BranchType",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopyTransition",
    "ValidationReport",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_spectroscopy_data",
    "validate_quantum_numbers",
    # Triage
    "AU_TO_WAVENUMBER_CONVERSION",
    "HarmonicVibrationalResult",
    "IsotopeSpec",
    "IsotopeTriageResult",
    "IsotopologueData",
    "PolyadBlock",
    "ProductClass",
    "ResonanceCondition",
    "ResonanceReport",
    "ResonanceType",
    "TriageConfig",
    "TriageReport",
    "TriageStatus",
    "calculate_triage_provenance_hash",
    "check_isotope_abundance",
    "compute_harmonic_vibrational_frequencies",
    "evaluate_isotopologue_abundance",
    "get_isotope_properties",
    "identify_polyad_blocks",
    "mass_matrix_transformation",
    "parse_isotopologue_input",
    "resolve_mendeleev_database",
    "scan_coriolis_resonances",
    "scan_darling_dennison_resonances",
    "scan_fermi_resonances",
    "trap_resonances",
    "triage_spectroscopic_system",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopyTransition,
    ValidationReport,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)
from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)

__all__ = [
    # Ingest
    "BranchType",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopyTransition",
    "ValidationReport",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_spectroscopy_data",
    "validate_quantum_numbers",
    # Triage
    "AU_TO_WAVENUMBER_CONVERSION",
    "HarmonicVibrationalResult",
    "IsotopeSpec",
    "IsotopeTriageResult",
    "IsotopologueData",
    "PolyadBlock",
    "ProductClass",
    "ResonanceCondition",
    "ResonanceReport",
    "ResonanceType",
    "TriageConfig",
    "TriageReport",
    "TriageStatus",
    "calculate_triage_provenance_hash",
    "check_isotope_abundance",
    "compute_harmonic_vibrational_frequencies",
    "evaluate_isotopologue_abundance",
    "get_isotope_properties",
    "identify_polyad_blocks",
    "mass_matrix_transformation",
    "parse_isotopologue_input",
    "resolve_mendeleev_database",
    "scan_coriolis_resonances",
    "scan_darling_dennison_resonances",
    "scan_fermi_resonances",
    "trap_resonances",
    "triage_spectroscopic_system",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\cochem_spycfit_triage.py ---
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
HARTREE_TO_JOULE: float = 4.3597447222071e-18       # 1 E_h in Joules (FP64)
BOHR_RADIUS_METER: float = 0.529177210903e-10       # 1 a_0 in meters (FP64)
AMU_TO_KG: float = 1.66053906660e-27                # 1 u in kg (FP64)
SPEED_OF_LIGHT_CM_S: float = 29979245800.0          # Speed of light c in cm/s (FP64)
PLANCK_CONSTANT_JS: float = 6.62607015e-34          # Planck constant h in J*s (FP64)
EV_TO_JOULE: float = 1.602176634e-19                # 1 eV in Joules (FP64)
ANGSTROM_TO_METER: float = 1.0e-10                  # 1 Angstrom in meters (FP64)

# Analytical conversion factor from sqrt(Hartree / (bohr^2 * u)) to wavenumber (cm^-1):
# conv = (1 / (2 * pi * c_cm)) * sqrt(E_h / (a_0^2 * u))
# conv = 5140.487143611564 cm^-1
AU_TO_WAVENUMBER_CONVERSION: float = 5140.487143611564


# =============================================================================
# 2. ENUMS
# =============================================================================

class ResonanceType(str, Enum):
    """Vibrational and rovibrational resonance classifications."""
    FERMI_TYPE_I = "Fermi_Type_I"         # Fundamental couples to overtone: omega_i ~ 2 * omega_j
    FERMI_TYPE_II = "Fermi_Type_II"       # Fundamental couples to combination band: omega_i ~ omega_j + omega_k
    DARLING_DENNISON = "Darling_Dennison" # Overtones/combinations couple: 2 * omega_i ~ 2 * omega_j
    CORIOLIS = "Coriolis"                 # Rovibrational Coriolis coupling: omega_i ~ omega_j


class ProductClass(str, Enum):
    """Method Matrix v4 Spectroscopic Product Classifications."""
    PRODUCT_A = "Product A"  # Absolute de novo prediction (parent isotopologue)
    PRODUCT_B = "Product B"  # Semi-experimental / template-anchored (singly-substituted)
    PRODUCT_C = "Product C"  # Difference targets (shifts, conformers, poly-substituted)
    TRACE = "Trace"          # Rare/sub-threshold isotopologues (< 0.01% natural abundance)


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

    symbol: str = Field(..., description="Chemical element symbol (e.g. 'C', 'H', 'O').")
    mass_number: int = Field(..., gt=0, description="Mass number A (number of nucleons).")
    atomic_number: int = Field(..., gt=0, description="Atomic number Z (number of protons).")
    mass_amu: float = Field(..., gt=0.0, description="Atomic isotopic mass in unified atomic mass units u (FP64).")
    abundance_percent: float = Field(..., ge=0.0, le=100.0, description="Natural isotopic abundance percentage (0.0 to 100.0).")
    nuclear_spin: Optional[str] = Field(default=None, description="Nuclear spin quantum number string (e.g. '1/2', '1').")
    is_stable: bool = Field(default=True, description="Whether isotope is stable.")


class IsotopologueData(BaseModel):
    """Collection of constituent isotopes defining an isotopologue."""
    model_config = ConfigDict(extra="forbid")

    formula: Optional[str] = Field(default=None, description="Chemical formula or label.")
    name: Optional[str] = Field(default=None, description="Descriptive system name.")
    isotopes: List[IsotopeSpec] = Field(..., min_length=1, description="Ordered list of constituent atomic isotopes.")
    custom_masses: Optional[List[float]] = Field(default=None, description="Optional override masses in amu (FP64).")


class IsotopeTriageResult(BaseModel):
    """Output report from isotopic natural abundance analysis."""
    model_config = ConfigDict(extra="forbid")

    isotopologue_name: str = Field(..., description="Canonical formula or label of the isotopologue.")
    total_mass_amu: float = Field(..., gt=0.0, description="Total molecular mass in amu (FP64).")
    abundance_percent: float = Field(..., ge=0.0, description="Product natural abundance percentage (FP64).")
    abundance_fraction: float = Field(..., ge=0.0, description="Product natural abundance fraction (0.0 to 1.0) (FP64).")
    is_observable: bool = Field(..., description="True if abundance satisfies the triage threshold.")
    product_class: ProductClass = Field(..., description="Method Matrix v4 Product classification.")
    substitution_count: int = Field(..., ge=0, description="Number of isotopic substitutions relative to parent.")
    parent_formula: str = Field(..., description="Formula of the parent isotopologue.")
    isotope_details: List[IsotopeSpec] = Field(..., description="Constituent isotope properties.")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if sub-threshold.")


class HarmonicVibrationalResult(BaseModel):
    """Harmonic vibrational analysis output from mass-weighted Cartesian Hessian."""
    model_config = ConfigDict(extra="forbid")

    frequencies_cm1: List[float] = Field(..., description="All computed harmonic frequencies in cm^-1 (FP64).")
    vibrational_frequencies_cm1: List[float] = Field(..., description="Genuine vibrational normal mode frequencies in cm^-1 (FP64).")
    imaginary_frequencies_count: int = Field(..., ge=0, description="Number of imaginary frequencies (saddle points).")
    zero_modes_count: int = Field(..., ge=0, description="Number of translational and rotational zero modes.")
    eigenvalues: List[float] = Field(..., description="Secular equation eigenvalues (FP64).")
    mass_weighted_hessian_shape: Tuple[int, int] = Field(..., description="Dimensions of the mass-weighted Hessian.")
    reduced_masses_amu: Optional[List[float]] = Field(default=None, description="Mode reduced masses in amu (FP64).")
    normal_modes: Optional[List[List[float]]] = Field(default=None, description="Normal mode displacement vectors.")


class ResonanceCondition(BaseModel):
    """Detected resonance interaction between normal modes."""
    model_config = ConfigDict(extra="forbid")

    resonance_type: ResonanceType = Field(..., description="Classification of resonance interaction.")
    mode_indices: List[int] = Field(..., description="Indices of interacting vibrational normal modes (0-indexed).")
    frequencies_cm1: List[float] = Field(..., description="Harmonic frequencies of the involved modes in cm^-1.")
    energy_difference_cm1: float = Field(..., ge=0.0, description="Energy detuning |Delta E| in cm^-1 (FP64).")
    description: str = Field(..., description="Spectroscopic description of the resonance condition.")
    severity: str = Field(..., description="Interaction strength classification: 'STRONG', 'MODERATE', or 'WEAK'.")


class PolyadBlock(BaseModel):
    """Coupled vibrational subspace (Polyad) requiring multi-state diagonalization."""
    model_config = ConfigDict(extra="forbid")

    block_id: int = Field(..., ge=0, description="Unique polyad block index.")
    mode_indices: List[int] = Field(..., description="List of mutually coupled mode indices.")
    frequencies_cm1: List[float] = Field(..., description="Frequencies of the modes within this block.")
    resonance_types: List[ResonanceType] = Field(..., description="Types of resonance couplings present.")
    dimension: int = Field(..., ge=2, description="Dimension of the interacting polyad state subspace.")
    frequency_span_cm1: float = Field(..., ge=0.0, description="Max frequency spread across modes in cm^-1.")
    recommended_treatment: str = Field(..., description="Theoretical treatment recommendation (e.g. Polyad Hamiltonian).")


class ResonanceReport(BaseModel):
    """Comprehensive resonance scan and polyad partition report."""
    model_config = ConfigDict(extra="forbid")

    total_resonances_count: int = Field(..., ge=0, description="Total number of detected resonance conditions.")
    fermi_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Fermi resonances.")
    coriolis_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Coriolis resonances.")
    darling_dennison_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Darling-Dennison resonances.")
    polyad_blocks: List[PolyadBlock] = Field(default_factory=list, description="Constructed Polyad blocks.")
    has_strong_resonances: bool = Field(..., description="True if any strong resonance condition is active.")
    recommended_hamiltonian: str = Field(..., description="Recommended Hamiltonian model for downstream VPT2/fitting.")


class TriageConfig(BaseModel):
    """Configuration options for spectroscopic intake triage."""
    model_config = ConfigDict(extra="forbid")

    abundance_threshold_percent: float = Field(default=0.01, ge=0.0, description="Natural abundance cut-off percentage (e.g. 0.01% = 1e-4).")
    fermi_type_1_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Fermi Type I resonance in cm^-1.")
    fermi_type_2_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Fermi Type II resonance in cm^-1.")
    coriolis_threshold_cm1: float = Field(default=20.0, ge=0.0, description="Max frequency difference for Coriolis resonance in cm^-1.")
    darling_dennison_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Darling-Dennison resonance in cm^-1.")
    zero_frequency_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Threshold below which modes are treated as translation/rotation zeros.")
    hessian_units: str = Field(default="hartree_bohr2", description="Units of input Cartesian Hessian ('hartree_bohr2', 'si', 'ev_angstrom2').")
    mendeleev_db_path: Optional[str] = Field(default=None, description="Optional custom path to local Mendeleev SQLite database.")
    allow_imaginary_frequencies: bool = Field(default=False, description="Whether to permit imaginary frequencies (transition states) without error.")


class TriageReport(BaseModel):
    """Master audit and triage report for a molecular spectroscopic candidate."""
    model_config = ConfigDict(extra="forbid")

    config: TriageConfig = Field(..., description="Configuration used for this triage run.")
    status: TriageStatus = Field(..., description="Overall triage execution status.")
    passed: bool = Field(..., description="Whether system passed all triage gates for downstream processing.")
    product_class: ProductClass = Field(..., description="Method Matrix v4 Product classification.")
    isotopologue_triage: Optional[IsotopeTriageResult] = Field(default=None, description="Isotopic natural abundance triage.")
    vibrational_triage: Optional[HarmonicVibrationalResult] = Field(default=None, description="Harmonic vibrational analysis result.")
    resonance_triage: Optional[ResonanceReport] = Field(default=None, description="Resonance and polyad analysis report.")
    warnings: List[str] = Field(default_factory=list, description="Informational warnings emitted during triage.")
    summary: str = Field(..., description="Human-readable summary of triage decisions.")
    sha256_provenance: str = Field(..., description="Cryptographic SHA-256 hash digest of input parameters and outputs.")


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
        env_path = os.environ.get("COCHEM_MENDELEEV_DB") or os.environ.get("MENDELEEV_DATABASE_PATH")
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
        raise FileNotFoundError("Could not resolve local Mendeleev SQLite database (elements.db).")

    # Air-gap binary verification: check file size and SQLite header
    file_size = resolved.stat().st_size
    if file_size < 100000:
        raise ValueError(f"Mendeleev database file at {resolved} is corrupted or incomplete ({file_size} bytes).")

    with open(resolved, "rb") as f:
        magic_bytes = f.read(16)

    if not magic_bytes.startswith(b"SQLite format 3\x00"):
        raise ValueError(f"File at {resolved} is not a valid SQLite 3 database.")

    logger.debug(f"Resolved air-gapped Mendeleev SQLite database: {resolved} ({file_size} bytes)")
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
        raise ValueError(f"Unknown chemical element symbol '{clean_symbol}': {exc}") from exc

    atomic_number = int(elem.atomic_number)

    if mass_number is not None:
        target_a = int(mass_number)
        matching = [iso for iso in elem.isotopes if iso.mass_number == target_a]
        if not matching:
            raise ValueError(f"Isotope '{clean_symbol}-{target_a}' not found in Mendeleev database.")
        selected_iso = matching[0]
    else:
        # Resolve to most abundant isotope
        valid_isotopes = [iso for iso in elem.isotopes if iso.abundance is not None]
        if valid_isotopes:
            selected_iso = max(valid_isotopes, key=lambda iso: float(iso.abundance or 0.0))
        else:
            # Fallback to first isotope if no abundances recorded
            selected_iso = elem.isotopes[0]

    # Extract mass and abundance in FP64
    iso_mass = float(selected_iso.mass) if selected_iso.mass is not None else float(elem.atomic_weight or selected_iso.mass_number)
    iso_abundance = float(selected_iso.abundance) if selected_iso.abundance is not None else 0.0
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
    isotopologue: Union[IsotopologueData, Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]], str],
    db_path: Optional[Union[str, Path]] = None,
) -> List[IsotopeSpec]:
    """
    Parses diverse isotopologue representations into a standard list of IsotopeSpec models.

    Supported formats:
    - IsotopologueData Pydantic model
    - List of tuples: [("C", 12), ("O", 16), ("O", 16)]
    - List of dicts: [{"symbol": "C", "mass_number": 12}, ...]
    - List of IsotopeSpec models
    - String with hyphenated isotopes: "13C-1H-35Cl" or formula "12C16O2"
    """
    if isinstance(isotopologue, IsotopologueData):
        return list(isotopologue.isotopes)

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
                    mass_no = int(prefix_a) if prefix_a else (int(suffix_a) if suffix_a else None)
                    specs.append(get_isotope_properties(sym, mass_no, db_path=db_path))
                else:
                    raise ValueError(f"Cannot parse isotope token '{tok}' from string '{text}'.")
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
                raise ValueError(f"Unrecognized parenthesized isotopologue formula: '{text}'.")
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
            sym, a_val = item
            specs.append(get_isotope_properties(str(sym), int(a_val), db_path=db_path))
        elif isinstance(item, dict):
            sym = str(item.get("symbol") or item.get("element"))
            a_val = item.get("mass_number") or item.get("a")
            mass_no_val: Optional[int] = int(a_val) if a_val is not None else None
            specs.append(get_isotope_properties(sym, mass_no_val, db_path=db_path))
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
        rejection_reason = (
            f"Natural abundance {abundance_percent:.6e}% is below threshold {threshold_percent:.4f}%."
        )
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
    isotopologue_data: Union[IsotopologueData, Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]], str, Dict[str, Any]],
    threshold_percent: float = 0.01,
    db_path: Optional[Union[str, Path]] = None,
) -> IsotopeTriageResult:
    """
    High-level entry point to evaluate natural abundance and triage rare isotopologues.
    """
    specs = parse_isotopologue_input(isotopologue_data, db_path=db_path)
    return evaluate_isotopologue_abundance(specs, threshold_percent=threshold_percent, db_path=db_path)


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
        raise ValueError(f"Cartesian Hessian F_cart must be a 2D square matrix, got shape {f_arr.shape}.")

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
        unit_scale = math.sqrt(ev_to_hartree / (angstrom_to_bohr ** 2))
        scale_factor = AU_TO_WAVENUMBER_CONVERSION * unit_scale
    else:
        raise ValueError(f"Unsupported Hessian unit system '{units}'. Supported: 'hartree_bohr2', 'si', 'ev_angstrom2'.")

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
        norm_q = np.linalg.norm(q_cart)
        if norm_q > 1e-12:
            q_cart_norm = q_cart / norm_q
            # Reduced mass mu_k = 1 / sum(q_cart_norm_i^2 / m_i)
            mu_k_val: float = float(1.0 / np.sum((q_cart_norm ** 2) / coord_masses))
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
                severity = "STRONG" if detuning <= 10.0 else ("MODERATE" if detuning <= 20.0 else "WEAK")
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
                    severity = "STRONG" if detuning <= 10.0 else ("MODERATE" if detuning <= 20.0 else "WEAK")
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
                severity = "STRONG" if diff <= 5.0 else ("MODERATE" if diff <= 10.0 else "WEAK")
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
                severity = "STRONG" if diff <= 10.0 else ("MODERATE" if diff <= 20.0 else "WEAK")
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

    fermi_res = scan_fermi_resonances(freq_list, type1_tol=fermi_tol, type2_tol=fermi_tol)
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
        recommended_hamiltonian = "Standard Rayleigh-Schroedinger Perturbation Theory (VPT2)."

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
        dump_bytes = json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode("utf-8")
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
                raise ValueError("Masses must be provided directly or inferred from isotopologue specification.")

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
            if res_result.has_strong_resonances and overall_status == TriageStatus.PASSED:
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
    summary_parts = [f"Triage Status: {overall_status.value} (Product Class: {product_class.value})."]
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
        } if vib_result else None,
        "res": {
            "total_count": res_result.total_resonances_count,
            "polyad_count": len(res_result.polyad_blocks),
        } if res_result else None,
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_triage.py ---
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
HARTREE_TO_JOULE: float = 4.3597447222071e-18       # 1 E_h in Joules (FP64)
BOHR_RADIUS_METER: float = 0.529177210903e-10       # 1 a_0 in meters (FP64)
AMU_TO_KG: float = 1.66053906660e-27                # 1 u in kg (FP64)
SPEED_OF_LIGHT_CM_S: float = 29979245800.0          # Speed of light c in cm/s (FP64)
PLANCK_CONSTANT_JS: float = 6.62607015e-34          # Planck constant h in J*s (FP64)
EV_TO_JOULE: float = 1.602176634e-19                # 1 eV in Joules (FP64)
ANGSTROM_TO_METER: float = 1.0e-10                  # 1 Angstrom in meters (FP64)

# Analytical conversion factor from sqrt(Hartree / (bohr^2 * u)) to wavenumber (cm^-1):
# conv = (1 / (2 * pi * c_cm)) * sqrt(E_h / (a_0^2 * u))
# conv = 5140.487143611564 cm^-1
AU_TO_WAVENUMBER_CONVERSION: float = 5140.487143611564


# =============================================================================
# 2. ENUMS
# =============================================================================

class ResonanceType(str, Enum):
    """Vibrational and rovibrational resonance classifications."""
    FERMI_TYPE_I = "Fermi_Type_I"         # Fundamental couples to overtone: omega_i ~ 2 * omega_j
    FERMI_TYPE_II = "Fermi_Type_II"       # Fundamental couples to combination band: omega_i ~ omega_j + omega_k
    DARLING_DENNISON = "Darling_Dennison" # Overtones/combinations couple: 2 * omega_i ~ 2 * omega_j
    CORIOLIS = "Coriolis"                 # Rovibrational Coriolis coupling: omega_i ~ omega_j


class ProductClass(str, Enum):
    """Method Matrix v4 Spectroscopic Product Classifications."""
    PRODUCT_A = "Product A"  # Absolute de novo prediction (parent isotopologue)
    PRODUCT_B = "Product B"  # Semi-experimental / template-anchored (singly-substituted)
    PRODUCT_C = "Product C"  # Difference targets (shifts, conformers, poly-substituted)
    TRACE = "Trace"          # Rare/sub-threshold isotopologues (< 0.01% natural abundance)


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

    symbol: str = Field(..., description="Chemical element symbol (e.g. 'C', 'H', 'O').")
    mass_number: int = Field(..., gt=0, description="Mass number A (number of nucleons).")
    atomic_number: int = Field(..., gt=0, description="Atomic number Z (number of protons).")
    mass_amu: float = Field(..., gt=0.0, description="Atomic isotopic mass in unified atomic mass units u (FP64).")
    abundance_percent: float = Field(..., ge=0.0, le=100.0, description="Natural isotopic abundance percentage (0.0 to 100.0).")
    nuclear_spin: Optional[str] = Field(default=None, description="Nuclear spin quantum number string (e.g. '1/2', '1').")
    is_stable: bool = Field(default=True, description="Whether isotope is stable.")


class IsotopologueData(BaseModel):
    """Collection of constituent isotopes defining an isotopologue."""
    model_config = ConfigDict(extra="forbid")

    formula: Optional[str] = Field(default=None, description="Chemical formula or label.")
    name: Optional[str] = Field(default=None, description="Descriptive system name.")
    isotopes: List[IsotopeSpec] = Field(..., min_length=1, description="Ordered list of constituent atomic isotopes.")
    custom_masses: Optional[List[float]] = Field(default=None, description="Optional override masses in amu (FP64).")


class IsotopeTriageResult(BaseModel):
    """Output report from isotopic natural abundance analysis."""
    model_config = ConfigDict(extra="forbid")

    isotopologue_name: str = Field(..., description="Canonical formula or label of the isotopologue.")
    total_mass_amu: float = Field(..., gt=0.0, description="Total molecular mass in amu (FP64).")
    abundance_percent: float = Field(..., ge=0.0, description="Product natural abundance percentage (FP64).")
    abundance_fraction: float = Field(..., ge=0.0, description="Product natural abundance fraction (0.0 to 1.0) (FP64).")
    is_observable: bool = Field(..., description="True if abundance satisfies the triage threshold.")
    product_class: ProductClass = Field(..., description="Method Matrix v4 Product classification.")
    substitution_count: int = Field(..., ge=0, description="Number of isotopic substitutions relative to parent.")
    parent_formula: str = Field(..., description="Formula of the parent isotopologue.")
    isotope_details: List[IsotopeSpec] = Field(..., description="Constituent isotope properties.")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if sub-threshold.")


class HarmonicVibrationalResult(BaseModel):
    """Harmonic vibrational analysis output from mass-weighted Cartesian Hessian."""
    model_config = ConfigDict(extra="forbid")

    frequencies_cm1: List[float] = Field(..., description="All computed harmonic frequencies in cm^-1 (FP64).")
    vibrational_frequencies_cm1: List[float] = Field(..., description="Genuine vibrational normal mode frequencies in cm^-1 (FP64).")
    imaginary_frequencies_count: int = Field(..., ge=0, description="Number of imaginary frequencies (saddle points).")
    zero_modes_count: int = Field(..., ge=0, description="Number of translational and rotational zero modes.")
    eigenvalues: List[float] = Field(..., description="Secular equation eigenvalues (FP64).")
    mass_weighted_hessian_shape: Tuple[int, int] = Field(..., description="Dimensions of the mass-weighted Hessian.")
    reduced_masses_amu: Optional[List[float]] = Field(default=None, description="Mode reduced masses in amu (FP64).")
    normal_modes: Optional[List[List[float]]] = Field(default=None, description="Normal mode displacement vectors.")


class ResonanceCondition(BaseModel):
    """Detected resonance interaction between normal modes."""
    model_config = ConfigDict(extra="forbid")

    resonance_type: ResonanceType = Field(..., description="Classification of resonance interaction.")
    mode_indices: List[int] = Field(..., description="Indices of interacting vibrational normal modes (0-indexed).")
    frequencies_cm1: List[float] = Field(..., description="Harmonic frequencies of the involved modes in cm^-1.")
    energy_difference_cm1: float = Field(..., ge=0.0, description="Energy detuning |Delta E| in cm^-1 (FP64).")
    description: str = Field(..., description="Spectroscopic description of the resonance condition.")
    severity: str = Field(..., description="Interaction strength classification: 'STRONG', 'MODERATE', or 'WEAK'.")


class PolyadBlock(BaseModel):
    """Coupled vibrational subspace (Polyad) requiring multi-state diagonalization."""
    model_config = ConfigDict(extra="forbid")

    block_id: int = Field(..., ge=0, description="Unique polyad block index.")
    mode_indices: List[int] = Field(..., description="List of mutually coupled mode indices.")
    frequencies_cm1: List[float] = Field(..., description="Frequencies of the modes within this block.")
    resonance_types: List[ResonanceType] = Field(..., description="Types of resonance couplings present.")
    dimension: int = Field(..., ge=2, description="Dimension of the interacting polyad state subspace.")
    frequency_span_cm1: float = Field(..., ge=0.0, description="Max frequency spread across modes in cm^-1.")
    recommended_treatment: str = Field(..., description="Theoretical treatment recommendation (e.g. Polyad Hamiltonian).")


class ResonanceReport(BaseModel):
    """Comprehensive resonance scan and polyad partition report."""
    model_config = ConfigDict(extra="forbid")

    total_resonances_count: int = Field(..., ge=0, description="Total number of detected resonance conditions.")
    fermi_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Fermi resonances.")
    coriolis_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Coriolis resonances.")
    darling_dennison_resonances: List[ResonanceCondition] = Field(default_factory=list, description="Detected Darling-Dennison resonances.")
    polyad_blocks: List[PolyadBlock] = Field(default_factory=list, description="Constructed Polyad blocks.")
    has_strong_resonances: bool = Field(..., description="True if any strong resonance condition is active.")
    recommended_hamiltonian: str = Field(..., description="Recommended Hamiltonian model for downstream VPT2/fitting.")


class TriageConfig(BaseModel):
    """Configuration options for spectroscopic intake triage."""
    model_config = ConfigDict(extra="forbid")

    abundance_threshold_percent: float = Field(default=0.01, ge=0.0, description="Natural abundance cut-off percentage (e.g. 0.01% = 1e-4).")
    fermi_type_1_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Fermi Type I resonance in cm^-1.")
    fermi_type_2_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Fermi Type II resonance in cm^-1.")
    coriolis_threshold_cm1: float = Field(default=20.0, ge=0.0, description="Max frequency difference for Coriolis resonance in cm^-1.")
    darling_dennison_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Max energy detuning for Darling-Dennison resonance in cm^-1.")
    zero_frequency_threshold_cm1: float = Field(default=30.0, ge=0.0, description="Threshold below which modes are treated as translation/rotation zeros.")
    hessian_units: str = Field(default="hartree_bohr2", description="Units of input Cartesian Hessian ('hartree_bohr2', 'si', 'ev_angstrom2').")
    mendeleev_db_path: Optional[str] = Field(default=None, description="Optional custom path to local Mendeleev SQLite database.")
    allow_imaginary_frequencies: bool = Field(default=False, description="Whether to permit imaginary frequencies (transition states) without error.")


class TriageReport(BaseModel):
    """Master audit and triage report for a molecular spectroscopic candidate."""
    model_config = ConfigDict(extra="forbid")

    config: TriageConfig = Field(..., description="Configuration used for this triage run.")
    status: TriageStatus = Field(..., description="Overall triage execution status.")
    passed: bool = Field(..., description="Whether system passed all triage gates for downstream processing.")
    product_class: ProductClass = Field(..., description="Method Matrix v4 Product classification.")
    isotopologue_triage: Optional[IsotopeTriageResult] = Field(default=None, description="Isotopic natural abundance triage.")
    vibrational_triage: Optional[HarmonicVibrationalResult] = Field(default=None, description="Harmonic vibrational analysis result.")
    resonance_triage: Optional[ResonanceReport] = Field(default=None, description="Resonance and polyad analysis report.")
    warnings: List[str] = Field(default_factory=list, description="Informational warnings emitted during triage.")
    summary: str = Field(..., description="Human-readable summary of triage decisions.")
    sha256_provenance: str = Field(..., description="Cryptographic SHA-256 hash digest of input parameters and outputs.")


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
        env_path = os.environ.get("COCHEM_MENDELEEV_DB") or os.environ.get("MENDELEEV_DATABASE_PATH")
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
        raise FileNotFoundError("Could not resolve local Mendeleev SQLite database (elements.db).")

    # Air-gap binary verification: check file size and SQLite header
    file_size = resolved.stat().st_size
    if file_size < 100000:
        raise ValueError(f"Mendeleev database file at {resolved} is corrupted or incomplete ({file_size} bytes).")

    with open(resolved, "rb") as f:
        magic_bytes = f.read(16)

    if not magic_bytes.startswith(b"SQLite format 3\x00"):
        raise ValueError(f"File at {resolved} is not a valid SQLite 3 database.")

    logger.debug(f"Resolved air-gapped Mendeleev SQLite database: {resolved} ({file_size} bytes)")
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
        raise ValueError(f"Unknown chemical element symbol '{clean_symbol}': {exc}") from exc

    atomic_number = int(elem.atomic_number)

    if mass_number is not None:
        target_a = int(mass_number)
        matching = [iso for iso in elem.isotopes if iso.mass_number == target_a]
        if not matching:
            raise ValueError(f"Isotope '{clean_symbol}-{target_a}' not found in Mendeleev database.")
        selected_iso = matching[0]
    else:
        # Resolve to most abundant isotope
        valid_isotopes = [iso for iso in elem.isotopes if iso.abundance is not None]
        if valid_isotopes:
            selected_iso = max(valid_isotopes, key=lambda iso: float(iso.abundance or 0.0))
        else:
            # Fallback to first isotope if no abundances recorded
            selected_iso = elem.isotopes[0]

    # Extract mass and abundance in FP64
    iso_mass = float(selected_iso.mass) if selected_iso.mass is not None else float(elem.atomic_weight or selected_iso.mass_number)
    iso_abundance = float(selected_iso.abundance) if selected_iso.abundance is not None else 0.0
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
    isotopologue: Union[IsotopologueData, Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]], str],
    db_path: Optional[Union[str, Path]] = None,
) -> List[IsotopeSpec]:
    """
    Parses diverse isotopologue representations into a standard list of IsotopeSpec models.

    Supported formats:
    - IsotopologueData Pydantic model
    - List of tuples: [("C", 12), ("O", 16), ("O", 16)]
    - List of dicts: [{"symbol": "C", "mass_number": 12}, ...]
    - List of IsotopeSpec models
    - String with hyphenated isotopes: "13C-1H-35Cl" or formula "12C16O2"
    """
    if isinstance(isotopologue, IsotopologueData):
        return list(isotopologue.isotopes)

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
                    mass_no = int(prefix_a) if prefix_a else (int(suffix_a) if suffix_a else None)
                    specs.append(get_isotope_properties(sym, mass_no, db_path=db_path))
                else:
                    raise ValueError(f"Cannot parse isotope token '{tok}' from string '{text}'.")
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
                raise ValueError(f"Unrecognized parenthesized isotopologue formula: '{text}'.")
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
            sym, a_val = item
            specs.append(get_isotope_properties(str(sym), int(a_val), db_path=db_path))
        elif isinstance(item, dict):
            sym = str(item.get("symbol") or item.get("element"))
            a_val = item.get("mass_number") or item.get("a")
            mass_no_val: Optional[int] = int(a_val) if a_val is not None else None
            specs.append(get_isotope_properties(sym, mass_no_val, db_path=db_path))
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
        rejection_reason = (
            f"Natural abundance {abundance_percent:.6e}% is below threshold {threshold_percent:.4f}%."
        )
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
    isotopologue_data: Union[IsotopologueData, Sequence[Union[Tuple[str, int], Dict[str, Any], IsotopeSpec, str]], str, Dict[str, Any]],
    threshold_percent: float = 0.01,
    db_path: Optional[Union[str, Path]] = None,
) -> IsotopeTriageResult:
    """
    High-level entry point to evaluate natural abundance and triage rare isotopologues.
    """
    specs = parse_isotopologue_input(isotopologue_data, db_path=db_path)
    return evaluate_isotopologue_abundance(specs, threshold_percent=threshold_percent, db_path=db_path)


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
        raise ValueError(f"Cartesian Hessian F_cart must be a 2D square matrix, got shape {f_arr.shape}.")

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
        unit_scale = math.sqrt(ev_to_hartree / (angstrom_to_bohr ** 2))
        scale_factor = AU_TO_WAVENUMBER_CONVERSION * unit_scale
    else:
        raise ValueError(f"Unsupported Hessian unit system '{units}'. Supported: 'hartree_bohr2', 'si', 'ev_angstrom2'.")

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
        norm_q = np.linalg.norm(q_cart)
        if norm_q > 1e-12:
            q_cart_norm = q_cart / norm_q
            # Reduced mass mu_k = 1 / sum(q_cart_norm_i^2 / m_i)
            mu_k_val: float = float(1.0 / np.sum((q_cart_norm ** 2) / coord_masses))
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
                severity = "STRONG" if detuning <= 10.0 else ("MODERATE" if detuning <= 20.0 else "WEAK")
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
                    severity = "STRONG" if detuning <= 10.0 else ("MODERATE" if detuning <= 20.0 else "WEAK")
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
                severity = "STRONG" if diff <= 5.0 else ("MODERATE" if diff <= 10.0 else "WEAK")
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
                severity = "STRONG" if diff <= 10.0 else ("MODERATE" if diff <= 20.0 else "WEAK")
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

    fermi_res = scan_fermi_resonances(freq_list, type1_tol=fermi_tol, type2_tol=fermi_tol)
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
        recommended_hamiltonian = "Standard Rayleigh-Schroedinger Perturbation Theory (VPT2)."

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
        dump_bytes = json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode("utf-8")
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
                raise ValueError("Masses must be provided directly or inferred from isotopologue specification.")

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
            if res_result.has_strong_resonances and overall_status == TriageStatus.PASSED:
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
    summary_parts = [f"Triage Status: {overall_status.value} (Product Class: {product_class.value})."]
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
        } if vib_result else None,
        "res": {
            "total_count": res_result.total_resonances_count,
            "polyad_count": len(res_result.polyad_blocks),
        } if res_result else None,
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_triage.py ---
"""
Physical Unit and Integration Tests for Stage 1.0 Physical & Isotopic Resonance Filter.

Module: tests/test_cochem_spycfit_triage.py
Target: cochem_spycfit.intake.cochem_spycfit_triage

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance via zero-trust static inspection.
3. Mendeleev SQLite database local resolution and air-gap verification.
4. Natural abundance filtering, isotopic properties, and isotopologue triage.
5. Method Matrix v4 Mass Matrix Transformation (M^-1/2 F_cart M^-1/2) in FP64 precision.
6. Harmonic vibrational frequency computation, unit conversion, and normal mode analysis.
7. Resonance & Polyad Trap:
   - Fermi Type I resonance scan (omega_i ~ 2 * omega_j)
   - Fermi Type II resonance scan (omega_i ~ omega_j + omega_k)
   - Coriolis resonance scan (omega_i ~ omega_j)
   - Darling-Dennison resonance scan (2 * omega_i ~ 2 * omega_j)
   - Polyad block clustering and effective Hamiltonian recommendation
8. Unified triage pipeline (triage_spectroscopic_system) with deterministic SHA-256 provenance.
9. Pydantic v2 strict schema validation and extra field forbidding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from pydantic import ValidationError
import pytest

from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)

# =============================================================================
# 1. FILE ENCODING AND LF LINE ENDING TESTS
# =============================================================================

@pytest.fixture
def target_triage_file_paths() -> List[Path]:
    """Returns absolute paths to all Stage 1.0 triage source and test files."""
    root = Path(__file__).resolve().parent.parent
    return [
        root / "src" / "cochem_spycfit" / "intake" / "cochem_spycfit_triage.py",
        Path(__file__).resolve(),
    ]


def test_file_encoding_and_lf_line_endings(target_triage_file_paths: List[Path]) -> None:
    """Verify strictly Unix LF line endings (\\n), standard UTF-8 encoding, and no BOM."""
    for path in target_triage_file_paths:
        assert path.is_file(), f"Target file does not exist: {path}"
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"Found Windows CRLF (\\r\\n) in {path.name}"
        assert b"\n" in raw, f"Missing newline characters in {path.name}"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"Found UTF-8 BOM marker in {path.name}"


# =============================================================================
# 2. CODE INTEGRITY AND STATIC AUDIT
# =============================================================================

def test_anti_spoofing_and_integrity_compliance(target_triage_file_paths: List[Path]) -> None:
    """Performs static inspection ensuring target files comply with anti-spoofing standards."""
    banned_keywords = ["mock", "dummy", "placeholder", "fake", "stub"]
    for path in target_triage_file_paths:
        content = path.read_text(encoding="utf-8").lower()
        for kw in banned_keywords:
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#") or "banned_keywords" in clean_line:
                    continue
                assert f"def {kw}" not in clean_line, f"Line {line_idx} in {path.name} contains banned function def: {kw}"
                assert f"class {kw}" not in clean_line, f"Line {line_idx} in {path.name} contains banned class def: {kw}"
                assert f"import unittest.{kw}" not in clean_line, f"Line {line_idx} in {path.name} imports {kw}"


# =============================================================================
# 3. MENDELEEV DATABASE RESOLUTION AND AIR-GAP VERIFICATION
# =============================================================================

def test_resolve_mendeleev_database_default() -> None:
    """Tests that local Mendeleev SQLite database is resolved and verified without network access."""
    db_path = resolve_mendeleev_database()
    assert db_path.exists(), f"Database file does not exist at {db_path}"
    assert db_path.is_file(), f"Database path is not a file: {db_path}"
    assert db_path.stat().st_size > 1000000, "Database file is unexpectedly small"

    with open(db_path, "rb") as f:
        header = f.read(16)
    assert header.startswith(b"SQLite format 3\x00"), "Database does not have valid SQLite header"


def test_resolve_mendeleev_database_invalid_path() -> None:
    """Tests that non-existent database paths raise FileNotFoundError."""
    invalid_path = Path("/nonexistent/path/to/elements.db")
    with pytest.raises(FileNotFoundError):
        resolve_mendeleev_database(invalid_path)


def test_get_isotope_properties_carbon() -> None:
    """Tests querying isotopic properties for Carbon-12 and Carbon-13."""
    c12 = get_isotope_properties("C", 12)
    assert isinstance(c12, IsotopeSpec)
    assert c12.symbol == "C"
    assert c12.mass_number == 12
    assert c12.atomic_number == 6
    assert np.isclose(c12.mass_amu, 12.0, atol=1e-6)
    assert c12.abundance_percent > 98.0

    c13 = get_isotope_properties("C", 13)
    assert c13.symbol == "C"
    assert c13.mass_number == 13
    assert c13.atomic_number == 6
    assert np.isclose(c13.mass_amu, 13.003355, atol=1e-5)
    assert np.isclose(c13.abundance_percent, 1.06, atol=0.2)
    assert c13.nuclear_spin == "1/2"


def test_get_isotope_properties_hydrogen_and_oxygen() -> None:
    """Tests querying isotopic properties for Hydrogen and Oxygen isotopes."""
    h1 = get_isotope_properties("H", 1)
    assert h1.symbol == "H"
    assert h1.mass_number == 1
    assert np.isclose(h1.mass_amu, 1.007825, atol=1e-5)
    assert h1.abundance_percent > 99.9

    h2 = get_isotope_properties("H", 2)
    assert h2.mass_number == 2
    assert np.isclose(h2.mass_amu, 2.014102, atol=1e-5)
    assert h2.abundance_percent < 0.03

    o16 = get_isotope_properties("O", 16)
    assert o16.mass_number == 16
    assert np.isclose(o16.mass_amu, 15.994915, atol=1e-5)
    assert o16.abundance_percent > 99.7

    o18 = get_isotope_properties("O", 18)
    assert o18.mass_number == 18
    assert np.isclose(o18.mass_amu, 17.999160, atol=1e-5)
    assert np.isclose(o18.abundance_percent, 0.205, atol=0.05)


def test_get_isotope_properties_default_most_abundant() -> None:
    """Tests querying element without mass number returns the most abundant isotope."""
    c_default = get_isotope_properties("C")
    assert c_default.mass_number == 12
    assert c_default.abundance_percent > 98.0

    n_default = get_isotope_properties("N")
    assert n_default.mass_number == 14
    assert n_default.abundance_percent > 99.0


# =============================================================================
# 4. NATURAL ABUNDANCE FILTERING AND ISOTOPIC TRIAGE
# =============================================================================

def test_evaluate_isotopologue_abundance_parent_water() -> None:
    """Tests evaluation of parent water molecule (1H2 16O)."""
    h1 = get_isotope_properties("H", 1)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h1, h1, o16], threshold_percent=0.01)

    assert isinstance(result, IsotopeTriageResult)
    assert result.is_observable is True
    assert result.product_class == ProductClass.PRODUCT_A
    assert result.substitution_count == 0
    assert result.abundance_percent > 99.0
    assert np.isclose(result.total_mass_amu, 18.010565, atol=1e-4)


def test_evaluate_isotopologue_abundance_singly_substituted_water() -> None:
    """Tests evaluation of singly-substituted HDO (1H 2H 16O)."""
    h1 = get_isotope_properties("H", 1)
    h2 = get_isotope_properties("H", 2)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h1, h2, o16], threshold_percent=0.001)

    assert result.is_observable is True
    assert result.product_class in [ProductClass.PRODUCT_B, ProductClass.PRODUCT_C]
    assert result.substitution_count == 1
    assert result.abundance_percent > 0.001
    assert np.isclose(result.total_mass_amu, 19.016842, atol=1e-4)


def test_evaluate_isotopologue_abundance_rare_trace_d2o() -> None:
    """Tests evaluation of doubly-deuterated D2O (2H2 16O) falling below standard threshold."""
    h2 = get_isotope_properties("H", 2)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h2, h2, o16], threshold_percent=0.01)

    assert result.is_observable is False
    assert result.product_class == ProductClass.TRACE
    assert result.substitution_count == 2
    assert result.abundance_percent < 0.0001
    assert result.rejection_reason is not None


def test_check_isotope_abundance_string_input() -> None:
    """Tests check_isotope_abundance helper with tuple and string representations."""
    res_co2 = check_isotope_abundance([("C", 12), ("O", 16), ("O", 16)])
    assert res_co2.product_class == ProductClass.PRODUCT_A
    assert res_co2.is_observable is True

    res_13c_co2 = check_isotope_abundance([("C", 13), ("O", 16), ("O", 16)])
    assert res_13c_co2.product_class in [ProductClass.PRODUCT_B, ProductClass.PRODUCT_C]
    assert res_13c_co2.is_observable is True
    assert res_13c_co2.substitution_count == 1


def test_parse_isotopologue_input_formats() -> None:
    """Tests parse_isotopologue_input on string formulas, hyphenated tokens, and IsotopologueData models."""
    # 1. Hyphenated string token
    specs_hyphen = parse_isotopologue_input("13C-1H-35Cl")
    assert len(specs_hyphen) == 3
    assert specs_hyphen[0].symbol == "C" and specs_hyphen[0].mass_number == 13
    assert specs_hyphen[1].symbol == "H" and specs_hyphen[1].mass_number == 1
    assert specs_hyphen[2].symbol == "Cl" and specs_hyphen[2].mass_number == 35

    # 2. Parenthesized and spaced isotope formulas
    specs_paren = parse_isotopologue_input("(12C)(16O)2")
    assert len(specs_paren) == 3
    assert specs_paren[0].symbol == "C" and specs_paren[0].mass_number == 12
    assert specs_paren[1].symbol == "O" and specs_paren[1].mass_number == 16
    assert specs_paren[2].symbol == "O" and specs_paren[2].mass_number == 16

    specs_spaced = parse_isotopologue_input("12C 16O2")
    assert len(specs_spaced) == 3
    assert specs_spaced[0].symbol == "C" and specs_spaced[0].mass_number == 12
    assert specs_spaced[1].symbol == "O" and specs_spaced[1].mass_number == 16
    assert specs_spaced[2].symbol == "O" and specs_spaced[2].mass_number == 16

    specs_std = parse_isotopologue_input("H2O")
    assert len(specs_std) == 3
    assert specs_std[0].symbol == "H" and specs_std[0].mass_number == 1
    assert specs_std[1].symbol == "H" and specs_std[1].mass_number == 1
    assert specs_std[2].symbol == "O" and specs_std[2].mass_number == 16

    # 3. IsotopologueData Pydantic model
    iso_data = IsotopologueData(
        formula="H2O",
        name="Water",
        isotopes=[
            get_isotope_properties("H", 1),
            get_isotope_properties("H", 1),
            get_isotope_properties("O", 16),
        ],
    )
    specs_model = parse_isotopologue_input(iso_data)
    assert len(specs_model) == 3
    assert specs_model[0].symbol == "H"


def test_calculate_triage_provenance_hash_deterministic() -> None:
    """Tests that calculate_triage_provenance_hash generates reproducible SHA-256 digests."""
    payload1 = {"system": "H2O", "modes": [1595.0, 3657.0, 3756.0]}
    payload2 = {"modes": [1595.0, 3657.0, 3756.0], "system": "H2O"}  # Key order difference
    hash1 = calculate_triage_provenance_hash(payload1)
    hash2 = calculate_triage_provenance_hash(payload2)
    assert len(hash1) == 64
    assert hash1 == hash2


# =============================================================================
# 5. METHOD MATRIX V4 MASS MATRIX TRANSFORMATION (FP64 PRECISION)
# =============================================================================

def test_mass_matrix_transformation_1d_diatomic() -> None:
    """
    Tests mass matrix transformation on a 1D diatomic harmonic oscillator:
    F_cart = [[k, -k], [-k, k]]
    masses = [m1, m2]
    H_mw = [[k/m1, -k/sqrt(m1*m2)], [-k/sqrt(m1*m2), k/m2]]
    """
    k = 500.0  # N/m
    m1 = 12.0  # amu
    m2 = 16.0  # amu

    f_cart = np.array([[k, -k], [-k, k]], dtype=np.float64)
    masses = [m1, m2]

    h_mw = mass_matrix_transformation(f_cart, masses)

    assert h_mw.shape == (2, 2)
    assert h_mw.dtype == np.float64
    assert np.isclose(h_mw[0, 0], k / m1, rtol=1e-14)
    assert np.isclose(h_mw[1, 1], k / m2, rtol=1e-14)
    assert np.isclose(h_mw[0, 1], -k / math.sqrt(m1 * m2), rtol=1e-14)
    assert np.isclose(h_mw[1, 0], -k / math.sqrt(m1 * m2), rtol=1e-14)


def test_mass_matrix_transformation_3n_expansion() -> None:
    """Tests that atom masses (length N) are correctly broadcasted to 3N Cartesian coordinates."""
    n_atoms = 3
    masses = [1.008, 1.008, 15.999]  # Water atoms
    dim = 3 * n_atoms

    random_state = np.random.RandomState(42)
    a_mat = random_state.randn(dim, dim)
    f_cart = np.dot(a_mat.T, a_mat)

    h_mw = mass_matrix_transformation(f_cart, masses)

    assert h_mw.shape == (dim, dim)
    assert h_mw.dtype == np.float64

    coord_masses = np.repeat(masses, 3)
    for i in range(dim):
        for j in range(dim):
            expected = f_cart[i, j] / math.sqrt(coord_masses[i] * coord_masses[j])
            assert np.isclose(h_mw[i, j], expected, rtol=1e-14)


def test_mass_matrix_transformation_dimension_mismatch() -> None:
    """Tests that mismatched dimensions raise ValueError."""
    f_cart = np.eye(6, dtype=np.float64)
    with pytest.raises(ValueError):
        mass_matrix_transformation(f_cart, [12.0, 16.0, 1.0, 1.0])


# =============================================================================
# 6. HARMONIC VIBRATIONAL FREQUENCY COMPUTATION
# =============================================================================

def test_compute_harmonic_vibrational_frequencies_diatomic_co() -> None:
    """
    Tests harmonic frequency calculation for Carbon Monoxide (12C 16O).
    Force constant k = 1.20 Hartree / bohr^2 (realistic quantum chemistry magnitude).
    Analytical frequency: omega = sqrt(k / mu) * AU_TO_WAVENUMBER_CONVERSION
    """
    m_c = 12.0
    m_o = 15.994915
    mu = (m_c * m_o) / (m_c + m_o)

    k = 1.20  # Hartree / bohr^2
    f_cart = np.zeros((6, 6), dtype=np.float64)
    f_cart[2, 2] = k
    f_cart[5, 5] = k
    f_cart[2, 5] = -k
    f_cart[5, 2] = -k

    result = compute_harmonic_vibrational_frequencies(
        F_cart=f_cart,
        masses=[m_c, m_o],
        units="hartree_bohr2",
        zero_threshold_cm1=30.0,
    )

    assert isinstance(result, HarmonicVibrationalResult)
    assert len(result.frequencies_cm1) == 6
    assert result.imaginary_frequencies_count == 0
    assert result.zero_modes_count == 5

    expected_omega = math.sqrt(k / mu) * AU_TO_WAVENUMBER_CONVERSION
    assert len(result.vibrational_frequencies_cm1) == 1
    assert np.isclose(result.vibrational_frequencies_cm1[0], expected_omega, rtol=1e-5)


def test_compute_harmonic_vibrational_frequencies_imaginary_mode() -> None:
    """Tests that saddle-point negative Hessian eigenvalues produce negative/imaginary frequencies."""
    f_cart = np.array([[-0.5]], dtype=np.float64)
    masses = [1.0]

    result = compute_harmonic_vibrational_frequencies(
        F_cart=f_cart,
        masses=masses,
        units="hartree_bohr2",
        zero_threshold_cm1=10.0,
    )

    assert result.imaginary_frequencies_count == 1
    assert result.frequencies_cm1[0] < 0.0
    expected_imag = -math.sqrt(0.5 / 1.0) * AU_TO_WAVENUMBER_CONVERSION
    assert np.isclose(result.frequencies_cm1[0], expected_imag, rtol=1e-5)


# =============================================================================
# 7. RESONANCE & POLYAD TRAP TESTS
# =============================================================================

def test_scan_fermi_resonances_type1_co2() -> None:
    """
    Tests Fermi Type I resonance detection on Carbon Dioxide:
    nu_1 (symmetric stretch) = 1333.0 cm^-1
    nu_2 (bend) = 667.0 cm^-1
    2 * nu_2 = 1334.0 cm^-1 -> Delta E = |1333.0 - 1334.0| = 1.0 cm^-1 (Strong Fermi resonance)
    """
    freqs = [667.0, 667.0, 1333.0, 2349.0]
    resonances = scan_fermi_resonances(freqs, type1_tol=30.0, type2_tol=30.0)

    type1_resonances = [r for r in resonances if r.resonance_type == ResonanceType.FERMI_TYPE_I]
    assert len(type1_resonances) >= 1

    matched = False
    for r in type1_resonances:
        if r.mode_indices == [2, 0] or r.mode_indices == [2, 1]:
            assert np.isclose(r.energy_difference_cm1, 1.0, atol=1e-4)
            assert r.severity == "STRONG"
            matched = True
    assert matched, "Fermi Type I resonance between nu1 and 2*nu2 not detected"


def test_scan_fermi_resonances_type2_triad() -> None:
    """
    Tests Fermi Type II resonance detection:
    omega_i ~ omega_j + omega_k
    nu_1 = 3000.0 cm^-1, nu_2 = 1400.0 cm^-1, nu_3 = 1605.0 cm^-1
    nu_2 + nu_3 = 3005.0 cm^-1 -> Delta E = |3000 - 3005| = 5.0 cm^-1 (Strong Fermi Type II)
    """
    freqs = [1400.0, 1605.0, 3000.0]
    resonances = scan_fermi_resonances(freqs, type1_tol=30.0, type2_tol=30.0)

    type2_resonances = [r for r in resonances if r.resonance_type == ResonanceType.FERMI_TYPE_II]
    assert len(type2_resonances) == 1
    assert type2_resonances[0].mode_indices == [2, 0, 1]
    assert np.isclose(type2_resonances[0].energy_difference_cm1, 5.0, atol=1e-4)
    assert type2_resonances[0].severity == "STRONG"


def test_scan_coriolis_resonances_near_degenerate() -> None:
    """
    Tests Coriolis resonance detection on nearly degenerate modes:
    omega_0 = 1000.0 cm^-1, omega_1 = 1003.5 cm^-1 -> Delta E = 3.5 cm^-1
    """
    freqs = [1000.0, 1003.5, 2500.0]
    resonances = scan_coriolis_resonances(freqs, coriolis_tol=20.0)

    assert len(resonances) == 1
    assert resonances[0].resonance_type == ResonanceType.CORIOLIS
    assert resonances[0].mode_indices == [0, 1]
    assert np.isclose(resonances[0].energy_difference_cm1, 3.5, atol=1e-4)
    assert resonances[0].severity == "STRONG"


def test_scan_darling_dennison_resonances() -> None:
    """
    Tests Darling-Dennison resonance between overtone pairs:
    2*omega_1 ~ 2*omega_2
    omega_1 = 3000.0 cm^-1, omega_2 = 3010.0 cm^-1 -> 2*omega_1 = 6000, 2*omega_2 = 6020 -> Delta E = 20 cm^-1
    """
    freqs = [3000.0, 3010.0]
    resonances = scan_darling_dennison_resonances(freqs, dd_tol=30.0)

    assert len(resonances) == 1
    assert resonances[0].resonance_type == ResonanceType.DARLING_DENNISON
    assert resonances[0].mode_indices == [0, 1]
    assert np.isclose(resonances[0].energy_difference_cm1, 20.0, atol=1e-4)


def test_identify_polyad_blocks_clustering() -> None:
    """Tests grouping interacting resonant modes into coupled Polyad blocks."""
    freqs = [667.0, 1333.0, 2000.0, 3000.0]
    res1 = ResonanceCondition(
        resonance_type=ResonanceType.FERMI_TYPE_I,
        mode_indices=[1, 0],
        frequencies_cm1=[1333.0, 667.0],
        energy_difference_cm1=1.0,
        description="Fermi resonance between mode 1 and overtone of 0",
        severity="STRONG",
    )

    blocks = identify_polyad_blocks([res1], freqs)

    assert len(blocks) == 1
    polyad = blocks[0]
    assert isinstance(polyad, PolyadBlock)
    assert set(polyad.mode_indices) == {0, 1}
    assert polyad.dimension >= 2
    assert "Polyad" in polyad.recommended_treatment


def test_trap_resonances_full_report() -> None:
    """Tests trap_resonances high-level entry point returning structured ResonanceReport."""
    freqs = [667.0, 667.0, 1333.0, 2349.0]
    report = trap_resonances(freqs, fermi_tol=30.0, coriolis_tol=20.0)

    assert isinstance(report, ResonanceReport)
    assert report.has_strong_resonances is True
    assert report.total_resonances_count > 0
    assert len(report.fermi_resonances) > 0
    assert "Polyad" in report.recommended_hamiltonian


# =============================================================================
# 8. UNIFIED TRIAGE PIPELINE (triage_spectroscopic_system)
# =============================================================================

def test_triage_spectroscopic_system_parent_water() -> None:
    """Tests full triage pipeline on parent H2O molecule."""
    m_h = 1.007825
    m_o = 15.994915
    masses = [m_h, m_h, m_o]

    f_diag = np.diag([
        m_h * (1595.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
        m_h * (3657.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
        m_o * (3756.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
    ])

    report = triage_spectroscopic_system(
        isotopologue=[("H", 1), ("H", 1), ("O", 16)],
        F_cart=f_diag,
        masses=masses,
        config=TriageConfig(abundance_threshold_percent=0.01),
    )

    assert isinstance(report, TriageReport)
    assert report.passed is True
    assert report.product_class == ProductClass.PRODUCT_A
    assert report.status in [TriageStatus.PASSED, TriageStatus.WARNING_RESONANCE]
    assert report.isotopologue_triage is not None
    assert report.isotopologue_triage.is_observable is True
    assert report.vibrational_triage is not None
    assert report.resonance_triage is not None
    assert len(report.sha256_provenance) == 64


def test_triage_spectroscopic_system_rare_rejection() -> None:
    """Tests that an ultra-rare isotopologue triggers sub-threshold triage warning/rejection."""
    report = triage_spectroscopic_system(
        isotopologue=[("H", 2), ("H", 2), ("O", 18)],
        config=TriageConfig(abundance_threshold_percent=0.01),
    )

    assert report.passed is False
    assert report.status == TriageStatus.REJECTED_RARE_ISOTOPE
    assert report.product_class == ProductClass.TRACE
    assert report.isotopologue_triage is not None
    assert report.isotopologue_triage.is_observable is False


# =============================================================================
# 9. PYDANTIC V2 SCHEMA STRICTNESS AND EXTRA FIELD FORBIDDING
# =============================================================================

def test_pydantic_schema_extra_fields_forbidden() -> None:
    """Ensures Pydantic v2 schemas reject unexpected/extra fields."""
    with pytest.raises(ValidationError):
        IsotopeSpec(
            symbol="C",
            mass_number=12,
            mass_amu=12.0,
            abundance_percent=98.9,
            unauthorized_field="illegal_payload",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        TriageConfig(
            abundance_threshold_percent=0.01,
            rogue_parameter=True,  # type: ignore[call-arg]
        )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.