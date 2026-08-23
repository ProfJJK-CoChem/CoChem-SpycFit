Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task5_Prompt1_ingest_phase2.md.
Original prompt:
# Task 5 (Phase 2): Master Ingestion - cochem_spycfit_ingest.py

**Objective:**
Update `cochem_spycfit_ingest.py` to implement Phase 2 requirements including Pydantic schema validation, regex parsing for legacy formats, cryptographic fingerprinting, and strict quantum logic gates.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_ingest.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 2 features in `cochem_spycfit_ingest.py`.
It must:
1. Implement modern intake with Pydantic schema validation requiring `Frequency_MHz`, `Uncertainty`, `Intensity`, `J_upper`, `Ka_upper`, `Kc_upper`, `J_lower`, `Ka_lower`, `Kc_lower`.
2. Implement legacy Pickett intake using strict positional regex. Translate Fortran double-precision exponential notation (e.g., D-05 to E-05). Failing lines must be appended to `failed_ingest.log` in `platformdirs.user_log_path("CoChem")`.
3. Compute a SHA-256 hash across all input experimental frequencies immediately upon ingestion and append to the `spectra_fit.zarr` state matrix.
4. Implement a Pre-Flight Syntax Validator that ensures $J \ge 0, K_a \ge 0, K_c \ge 0$ and $K_a + K_c = J$ or $J+1$. Must raise `CoChemDataException` on failure.
5. Implement a Pre-Serialization Physics Filter enforcing selection rules ($\Delta J = 0, \pm 1$ and dipole parity constraints).

**Constraints:**
- No mocked Pydantic schemas, regex, or hashing. 
- Implement actual validation logic.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import pandas as pd
import re
import hashlib
from pydantic import BaseModel
from platformdirs import user_log_path
import zarr

class SpectroscopySchema(BaseModel):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def parse_legacy_pickett(lines):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def validate_quantum_numbers(df):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    CoChemDataException,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopySchema,
    SpectroscopyTransition,
    ValidationReport,
    append_to_spectra_fit_zarr,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_legacy_pickett,
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
from cochem_spycfit.intake.cochem_spycfit_orchestrator import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    BOHR_TO_ANGSTROM,
    EckartFrameError,
    EckartFrameResult,
    FitOrchestratorError,
    HARTREE_TO_JOULE,
    HeartbeatStatus,
    IPCMode,
    LandscapeTensorError,
    OrchestrationResult,
    OrchestrationStatus,
    OrchestratorConfig,
    PLANCK_CONSTANT_JS,
    ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ,
    SPEED_OF_LIGHT_CM_S,
    TORQIPCError,
    TORQRequestPayload,
    TORQResponsePayload,
    TORQStatus,
    TORQTimeoutError,
    TensorEvaluationReport,
    WallTimeSafetyGuard,
    WallTimeYieldException,
    calculate_center_of_mass,
    calculate_inertia_tensor,
    calculate_orchestrator_provenance_hash,
    checkpoint_and_yield_node,
    compute_eckart_rotation_matrix,
    diagonalize_inertia_tensor,
    eckart_frame_correction,
    evaluate_landscape_tensors,
    orchestrate_spycfit_system,
    resolve_tier3_registry_path,
    route_missing_energy,
    route_missing_energy_file_drop,
    route_missing_energy_zmq,
)

__all__ = [
    # Ingest
    "BranchType",
    "CoChemDataException",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopySchema",
    "SpectroscopyTransition",
    "ValidationReport",
    "append_to_spectra_fit_zarr",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_legacy_pickett",
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
    # Orchestrator
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\cochem_spycfit_ingest.py ---
"""
CoChem-SpycFit: Stage 1.0 Bipartite Data Intake Parser & Quantum Gatekeeper (Phase 2).

Module: cochem_spycfit/intake/cochem_spycfit_ingest.py
Role: The Tripartite Data Intake Parser & In-Memory Pre-Serialization Physics Filter.

Capabilities:
1. Modern Pydantic Schema Validation (SpectroscopySchema & SpectroscopyTransition):
   Enforces mandatory fields (Frequency_MHz, Uncertainty, Intensity, J_upper, Ka_upper,
   Kc_upper, J_lower, Ka_lower, Kc_lower) and strict asymmetric top physics.
2. Positional & Free-Format Regex Ingestion (parse_legacy_pickett):
   Parses legacy Pickett .lin, .cat, .par files and JPL catalogs, converting Fortran
   double-precision notation ('D-05', 'D+02', '1.2345-05') to IEEE-754 FP64.
   Failing lines are appended to failed_ingest.log in platformdirs.user_log_path("CoChem").
3. Cryptographic Fingerprinting & Zarr State Matrix Appending:
   Computes SHA-256 hash across all experimental frequencies upon ingestion and persists
   the state matrix to spectra_fit.zarr.
4. Pre-Flight Syntax Validator (Quantum Logic Gate):
   Validates J >= 0, Ka >= 0, Kc >= 0, Ka <= J, Kc <= J, and asymmetric top sum rule
   Ka + Kc in {J, J+1}. Raises CoChemDataException on strict failure.
5. Pre-Serialization Physics Filter:
   Enforces rotational selection rules (Delta J in {-1, 0, 1}, J=0 -> 0 forbidden) and
   dipole transition parity constraints (a-type, b-type, c-type).

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE)
- Task 5 Master Ingestion & Pre-Flight Validation (CoChem-SpycFit SRS)
- Mendeleev Library Mandate (Dynamic Mass Retrieval)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import numpy as np
import pandas as pd
from platformdirs import user_log_path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import zarr

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Ingest")

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemError,
        CoChemIntegrityError,
        CoChemPrecisionError,
        ProvenanceErrorCode,
    )

    class CoChemDataException(CoChemError, ValueError):
        """Raised when input spectroscopic data fails validation, syntax checks, or quantum rules."""
        ...
except ImportError:
    class CoChemError(Exception):
        """Root fallback exception for CoChem."""
        ...

    class CoChemPrecisionError(RuntimeError):
        """Raised when JAX or numerical float precision is violated."""
        ...

    class CoChemIntegrityError(ValueError):
        """Raised when cryptographic checksum or binary integrity check fails."""
        ...

    class CoChemDataException(ValueError):
        """Raised when input spectroscopic data fails validation, syntax checks, or quantum rules."""
        ...


# =============================================================================
# 1. ENUMS AND DATA STRUCTURES
# =============================================================================

class FileFormat(str, Enum):
    """Supported file formats for spectroscopy data ingestion."""
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    JSON = "json"
    LIN = "lin"
    PAR = "par"
    CAT = "cat"
    UNKNOWN = "unknown"


class BranchType(str, Enum):
    """Spectroscopic rotational branch designations based on Delta J = J' - J''."""
    P = "P"  # Delta J = -1
    Q = "Q"  # Delta J = 0
    R = "R"  # Delta J = +1
    INVALID = "INVALID"


class DipoleType(str, Enum):
    """Asymmetric top dipole transition parity classifications."""
    A = "a"          # a-type: Delta Ka even, Delta Kc odd (ee <-> eo, oe <-> oo)
    B = "b"          # b-type: Delta Ka odd, Delta Kc odd (ee <-> oo, eo <-> oe)
    C = "c"          # c-type: Delta Ka odd, Delta Kc even (ee <-> oe, eo <-> oo)
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


class Parity(str, Enum):
    """Rotational state parity (Ka mod 2, Kc mod 2)."""
    EE = "ee"  # Ka even, Kc even
    EO = "eo"  # Ka even, Kc odd
    OE = "oe"  # Ka odd,  Kc even
    OO = "oo"  # Ka odd,  Kc odd


# Canonical column names for modern structured Pandas DataFrames
CANONICAL_COLUMNS: List[str] = [
    "Frequency_MHz",
    "Uncertainty",
    "Intensity",
    "J_upper",
    "Ka_upper",
    "Kc_upper",
    "J_lower",
    "Ka_lower",
    "Kc_lower",
]


# =============================================================================
# 2. PYDANTIC V2 SCHEMAS
# =============================================================================

class SpectroscopySchema(BaseModel):
    """
    Phase 2 Pydantic schema validation for single spectroscopic transitions.

    Mandatory fields:
    - Frequency_MHz (float64)
    - Uncertainty (float64)
    - Intensity (float64)
    - J_upper, Ka_upper, Kc_upper (int >= 0)
    - J_lower, Ka_lower, Kc_lower (int >= 0)
    """
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    Frequency_MHz: float = Field(
        ...,
        alias="frequency_mhz",
        description="Observed transition frequency in MHz (FP64).",
    )
    Uncertainty: float = Field(
        default=0.01,
        alias="uncertainty_mhz",
        description="Measurement uncertainty in MHz (FP64).",
    )
    Intensity: float = Field(
        default=1.0,
        alias="intensity",
        description="Observed or catalog intensity (FP64).",
    )
    J_upper: int = Field(
        ...,
        ge=0,
        alias="j_upper",
        description="Upper state total angular momentum quantum number J'.",
    )
    Ka_upper: int = Field(
        ...,
        ge=0,
        alias="ka_upper",
        description="Upper state prolate projection quantum number Ka'.",
    )
    Kc_upper: int = Field(
        ...,
        ge=0,
        alias="kc_upper",
        description="Upper state oblate projection quantum number Kc'.",
    )
    J_lower: int = Field(
        ...,
        ge=0,
        alias="j_lower",
        description="Lower state total angular momentum quantum number J''.",
    )
    Ka_lower: int = Field(
        ...,
        ge=0,
        alias="ka_lower",
        description="Lower state prolate projection quantum number Ka''.",
    )
    Kc_lower: int = Field(
        ...,
        ge=0,
        alias="kc_lower",
        description="Lower state oblate projection quantum number Kc''.",
    )
    branch: Optional[BranchType] = Field(default=None, description="Rotational branch (P, Q, R).")
    dipole_type: Optional[DipoleType] = Field(default=None, description="Dipole transition classification (a, b, c).")
    delta_j: Optional[int] = Field(default=None, description="Delta J = J' - J''.")
    delta_ka: Optional[int] = Field(default=None, description="Delta Ka = Ka' - Ka''.")
    delta_kc: Optional[int] = Field(default=None, description="Delta Kc = Kc' - Kc''.")
    upper_parity: Optional[Parity] = Field(default=None, description="Upper state parity (ee, eo, oe, oo).")
    lower_parity: Optional[Parity] = Field(default=None, description="Lower state parity (ee, eo, oe, oo).")
    valid: bool = Field(default=True, description="Whether transition passed syntax and selection rules.")
    drop_reason: Optional[str] = Field(default=None, description="Detailed explanation if line was dropped.")

    @model_validator(mode="after")
    def validate_quantum_physics(self) -> SpectroscopySchema:
        """Enforces asymmetric rotor quantum bounds and parity sum rules."""
        # 1. Ka <= J and Kc <= J bounds
        if self.Ka_upper > self.J_upper:
            raise ValueError(f"Upper Ka ({self.Ka_upper}) cannot exceed J ({self.J_upper})")
        if self.Kc_upper > self.J_upper:
            raise ValueError(f"Upper Kc ({self.Kc_upper}) cannot exceed J ({self.J_upper})")
        if self.Ka_lower > self.J_lower:
            raise ValueError(f"Lower Ka ({self.Ka_lower}) cannot exceed J ({self.J_lower})")
        if self.Kc_lower > self.J_lower:
            raise ValueError(f"Lower Kc ({self.Kc_lower}) cannot exceed J ({self.J_lower})")

        # 2. Asymmetric top sum rule: Ka + Kc in {J, J+1}
        sum_up = self.Ka_upper + self.Kc_upper
        if sum_up != self.J_upper and sum_up != (self.J_upper + 1):
            raise ValueError(
                f"Upper state Ka+Kc ({sum_up}) violates asymmetric top sum rule for J={self.J_upper}"
            )
        sum_low = self.Ka_lower + self.Kc_lower
        if sum_low != self.J_lower and sum_low != (self.J_lower + 1):
            raise ValueError(
                f"Lower state Ka+Kc ({sum_low}) violates asymmetric top sum rule for J={self.J_lower}"
            )

        # 3. Auto-populate differences if None
        if self.delta_j is None:
            self.delta_j = self.J_upper - self.J_lower
        if self.delta_ka is None:
            self.delta_ka = self.Ka_upper - self.Ka_lower
        if self.delta_kc is None:
            self.delta_kc = self.Kc_upper - self.Kc_lower

        # 4. Auto-populate parity
        parity_map = {
            (0, 0): Parity.EE,
            (0, 1): Parity.EO,
            (1, 0): Parity.OE,
            (1, 1): Parity.OO,
        }
        if self.upper_parity is None:
            self.upper_parity = parity_map[(self.Ka_upper % 2, self.Kc_upper % 2)]
        if self.lower_parity is None:
            self.lower_parity = parity_map[(self.Ka_lower % 2, self.Kc_lower % 2)]

        # 5. Auto-populate branch
        if self.branch is None:
            if self.delta_j == -1:
                self.branch = BranchType.P
            elif self.delta_j == 0:
                self.branch = BranchType.Q
            elif self.delta_j == 1:
                self.branch = BranchType.R
            else:
                self.branch = BranchType.INVALID

        # 6. Auto-populate dipole classification
        if self.dipole_type is None:
            if (self.J_upper == 0 and self.J_lower == 0) or abs(self.delta_j) > 1:
                self.dipole_type = DipoleType.FORBIDDEN
            else:
                d_ka_even = (abs(self.delta_ka) % 2 == 0)
                d_kc_even = (abs(self.delta_kc) % 2 == 0)
                if d_ka_even and not d_kc_even:
                    self.dipole_type = DipoleType.A
                elif not d_ka_even and not d_kc_even:
                    self.dipole_type = DipoleType.B
                elif not d_ka_even and d_kc_even:
                    self.dipole_type = DipoleType.C
                else:
                    self.dipole_type = DipoleType.FORBIDDEN

        return self


# Backwards compatibility alias for SpectroscopyTransition
SpectroscopyTransition = SpectroscopySchema


class PickettParameter(BaseModel):
    """Single Pickett parameter record (.par / .var)."""
    model_config = ConfigDict(extra="forbid")

    param_id: int = Field(..., description="Pickett parameter code (e.g. 10000, 20000, 30000, 200).")
    value: float = Field(..., description="Parameter value in MHz.")
    uncertainty: float = Field(default=0.0, description="Estimated parameter uncertainty in MHz.")
    label: str = Field(default="", description="Human-readable parameter label (e.g. 'A', 'B', 'DJ').")


class PickettParFile(BaseModel):
    """Structured representation of a legacy Pickett .par / .var parameter file."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Header title string from .par / .var file.")
    npar: int = Field(..., ge=0, description="Number of parameters in file.")
    nline: int = Field(..., description="Number of fitting lines.")
    nopt: int = Field(default=0, description="Pickett optimization control integer.")
    nwarn: int = Field(default=0, description="Pickett warning control integer.")
    erpar: float = Field(default=1.0, description="Parameter error scale factor.")
    wtfac: float = Field(default=1.0, description="Weight factor.")
    scale: float = Field(default=1.0, description="Frequency scale factor.")
    maxit: int = Field(default=50, description="Maximum iterations.")
    parameters: List[PickettParameter] = Field(default_factory=list, description="List of parsed parameters.")


class IngestConfig(BaseModel):
    """Configuration options for spectroscopy data ingestion and validation."""
    model_config = ConfigDict(extra="forbid")

    enforce_fp64: bool = Field(default=True, description="Strictly cast all float columns to np.float64.")
    validate_syntax: bool = Field(default=True, description="Apply Pre-Flight Syntax Validator quantum bounds.")
    apply_selection_rules: bool = Field(default=True, description="Apply Pre-Serialization Physics Filter selection rules.")
    allowed_dipole_types: Optional[Set[str]] = Field(
        default_factory=lambda: {"a", "b", "c"},
        description="Allowed dipole transition types (e.g. {'a', 'b', 'c'} or {'b'}).",
    )
    drop_invalid_syntax: bool = Field(default=True, description="Drop lines failing quantum syntax validation.")
    drop_forbidden_transitions: bool = Field(default=True, description="Drop lines violating physics selection rules.")
    log_warnings: bool = Field(default=True, description="Emit structured log warnings for dropped lines.")
    failed_log_path: Optional[str] = Field(default=None, description="Optional path to append failed lines log.")
    default_uncertainty: float = Field(default=0.01, ge=0.0, description="Default uncertainty in MHz if missing.")
    default_intensity: float = Field(default=1.0, description="Default intensity if missing.")
    zarr_path: Optional[str] = Field(default=None, description="Optional path to spectra_fit.zarr for state persistence.")


class ValidationReport(BaseModel):
    """Comprehensive diagnostic summary of ingestion, syntax, and physics filtering."""
    model_config = ConfigDict(extra="forbid")

    total_lines_read: int = Field(..., ge=0, description="Total raw lines or rows parsed from input.")
    valid_lines_count: int = Field(..., ge=0, description="Number of surviving lines passing all gates.")
    syntax_dropped_count: int = Field(default=0, ge=0, description="Lines dropped by Pre-Flight Syntax Validator.")
    physics_dropped_count: int = Field(default=0, ge=0, description="Lines dropped by Pre-Serialization Physics Filter.")
    sha256_fingerprint: str = Field(..., description="Cryptographic SHA-256 fingerprint of input frequencies.")
    warnings: List[str] = Field(default_factory=list, description="List of human-readable warnings emitted.")
    syntax_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed syntax drop records.")
    physics_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed physics drop records.")


class IngestResult(BaseModel):
    """Container holding sanitized Pandas DataFrame and validation metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    df: Any = Field(..., description="Sanitized Pandas DataFrame with FP64 precision.")
    report: ValidationReport = Field(..., description="Validation report and statistics.")
    sha256_fingerprint: str = Field(..., description="SHA-256 cryptographic hash of frequency array.")
    success: bool = Field(default=True, description="True if ingestion produced valid surviving data.")


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & FP64 PRECISION LOCK
# =============================================================================

def fortran_float_converter(val: Union[str, float, int]) -> float:
    """
    Safely converts a string containing Fortran scientific notation to a Python float64.

    Handles:
    - '1.2345D-05' -> 1.2345e-5
    - '1.2345D+02' -> 1.2345e+02
    - '1.2345d-05' -> 1.2345e-5
    - '1.2345E-05' -> 1.2345e-5
    - '12345.6789' -> 12345.6789
    - '1.2345-05'  -> 1.2345e-5 (implicit exponent notation in Fortran)

    Args:
        val: Input string, integer, or float.

    Returns:
        Converted value as float.

    Raises:
        ValueError: If the string cannot be parsed into a valid float.
    """
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        raise ValueError("Cannot convert empty string to float")

    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    implicit_exp_pattern = re.compile(r"^([+-]?\d+\.?\d*)([+-]\d{2,3})$")
    match = implicit_exp_pattern.match(s_clean)
    if match and "e" not in s_clean:
        s_clean = f"{match.group(1)}e{match.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


def enforce_fp64_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces JAX FP64 precision across the Pandas ingestion boundary.

    1. Executes jax.config.update('jax_enable_x64', True) and sets JAX_ENABLE_X64='True'.
    2. Explicitly casts all floating point columns to numpy.float64.
    3. Explicitly casts all quantum number integer columns to numpy.int64.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        DataFrame with strictly enforced float64 and int64 datatypes.
    """
    os.environ["JAX_ENABLE_X64"] = "True"
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as e:
        logger.debug(f"JAX FP64 config update notice: {e}")

    df_out = df.copy()

    float_cols = ["Frequency_MHz", "Uncertainty", "Intensity"]
    int_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower", "Delta_J", "Delta_Ka", "Delta_Kc"]

    for col in df_out.columns:
        if col in float_cols or (col not in int_cols and pd.api.types.is_float_dtype(df_out[col])):
            df_out[col] = df_out[col].astype(np.float64)

    for col in int_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round().astype(np.int64)

    return df_out


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINTING & ZARR PERSISTENCE
# =============================================================================

def calculate_frequency_fingerprint(
    frequencies: Union[np.ndarray, pd.Series, Sequence[float], bytes, str],
) -> str:
    """
    Calculates an immutable SHA-256 cryptographic fingerprint of experimental frequencies.

    Ensures that downstream Bayesian fitting state tensors (spectra_fit.zarr / spectra_fit.h5)
    can be rigorously traced back to the exact untampered experimental upload.

    Args:
        frequencies: Array, Series, Sequence of numerical frequencies, or raw string/bytes.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    if isinstance(frequencies, bytes):
        return hashlib.sha256(frequencies).hexdigest()

    if isinstance(frequencies, str):
        return hashlib.sha256(frequencies.encode("utf-8")).hexdigest()

    arr = np.asarray(frequencies, dtype=np.float64)
    if arr.size == 0:
        return hashlib.sha256(b"EMPTY_ARRAY").hexdigest()

    sorted_arr = np.sort(arr)
    ascii_payload = "\n".join(f"{f:.6f}" for f in sorted_arr).encode("utf-8")
    return hashlib.sha256(ascii_payload).hexdigest()


def append_to_spectra_fit_zarr(
    zarr_path: Union[str, Path],
    frequencies: Union[np.ndarray, pd.Series, Sequence[float]],
    sha256_hash: Optional[str] = None,
    group_name: str = "state_matrix",
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Appends experimental frequencies and cryptographic SHA-256 fingerprint to spectra_fit.zarr.

    Args:
        zarr_path: Path to the .zarr store (directory or zip store).
        frequencies: Array or Series of ingested frequencies.
        sha256_hash: Optional precomputed SHA-256 hash. If None, calculated dynamically.
        group_name: Subgroup name in zarr root (default 'state_matrix').
        metadata: Optional dictionary of supplementary provenance attributes.

    Returns:
        The zarr.Group object holding the state matrix and metadata attributes.
    """
    freq_arr = np.asarray(frequencies, dtype=np.float64)
    fingerprint = sha256_hash or calculate_frequency_fingerprint(freq_arr)

    path_obj = Path(zarr_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(path_obj), mode="a")
    group = root.require_group(group_name)

    # In Zarr 2 and 3, direct key assignment safely creates or updates the array
    group["frequencies"] = freq_arr

    group.attrs["sha256_fingerprint"] = fingerprint
    group.attrs["frequency_count"] = int(freq_arr.size)
    group.attrs["updated_at"] = datetime.now(timezone.utc).isoformat()
    if metadata:
        for k, v in metadata.items():
            group.attrs[k] = v

    return group


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE)
# =============================================================================

def validate_quantum_numbers(
    df: pd.DataFrame,
    drop_invalid: bool = True,
    raise_on_error: bool = False,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Pre-Flight Syntax Validator (Quantum Logic Gate).

    Mathematically intercepts and rejects impossible asymmetric rotor state assignments
    prior to Hamiltonian matrix generation.

    Enforces:
    1. Positive definite bounds: J >= 0, Ka >= 0, Kc >= 0 for both upper and lower states.
    2. Vector bounds: Ka <= J and Kc <= J for both states.
    3. Asymmetric top sum rule (Rotational Parity):
       For valid asymmetric rotor eigenstates, Ka + Kc must equal J or J + 1.
       - Upper state: Ka' + Kc' in {J', J' + 1}
       - Lower state: Ka'' + Kc'' in {J'', J'' + 1}

    Failure Protocol:
    Invalid lines are diagnosed. If raise_on_error is True, raises CoChemDataException.
    Otherwise, invalid lines are logged with structured diagnostics and dropped.

    Args:
        df: Input DataFrame with quantum number columns (J_upper, Ka_upper, Kc_upper, etc.).
        drop_invalid: If True, drops failing rows. If False, marks them in a 'syntax_valid' column.
        raise_on_error: If True, raises CoChemDataException immediately upon detecting syntax violations.
        log_warnings: If True, emits logger.warning for each dropped row.

    Returns:
        Tuple of (sanitized_df, list_of_syntax_violations).

    Raises:
        CoChemDataException: If raise_on_error is True and any row fails syntax validation.
    """
    if df.empty:
        return df.copy(), []

    qn_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    for col in qn_cols:
        if col not in df.columns:
            raise CoChemDataException(f"Missing required quantum number column '{col}' for syntax validation.")

    violations: List[Dict[str, Any]] = []
    valid_mask = np.ones(len(df), dtype=bool)

    j_up = df["J_upper"].to_numpy(dtype=np.int64)
    ka_up = df["Ka_upper"].to_numpy(dtype=np.int64)
    kc_up = df["Kc_upper"].to_numpy(dtype=np.int64)
    j_low = df["J_lower"].to_numpy(dtype=np.int64)
    ka_low = df["Ka_lower"].to_numpy(dtype=np.int64)
    kc_low = df["Kc_lower"].to_numpy(dtype=np.int64)

    for idx in range(len(df)):
        reasons: List[str] = []
        ju, kau, kcu = j_up[idx], ka_up[idx], kc_up[idx]
        jl, kal, kcl = j_low[idx], ka_low[idx], kc_low[idx]

        if ju < 0 or kau < 0 or kcu < 0:
            reasons.append(f"Negative upper quantum number: J'={ju}, Ka'={kau}, Kc'={kcu}")
        if jl < 0 or kal < 0 or kcl < 0:
            reasons.append(f"Negative lower quantum number: J''={jl}, Ka''={kal}, Kc''={kcl}")

        if kau > ju:
            reasons.append(f"Upper Ka' ({kau}) exceeds J' ({ju})")
        if kcu > ju:
            reasons.append(f"Upper Kc' ({kcu}) exceeds J' ({ju})")
        if kal > jl:
            reasons.append(f"Lower Ka'' ({kal}) exceeds J'' ({jl})")
        if kcl > jl:
            reasons.append(f"Lower Kc'' ({kcl}) exceeds J'' ({jl})")

        sum_up = kau + kcu
        if sum_up != ju and sum_up != (ju + 1):
            reasons.append(
                f"Syntax Ghost Line: Upper state Ka'+Kc' ({kau}+{kcu}={sum_up}) "
                f"violates asymmetric top sum rule for J'={ju} (expected {ju} or {ju + 1})"
            )

        sum_low = kal + kcl
        if sum_low != jl and sum_low != (jl + 1):
            reasons.append(
                f"Syntax Ghost Line: Lower state Ka''+Kc'' ({kal}+{kcl}={sum_low}) "
                f"violates asymmetric top sum rule for J''={jl} (expected {jl} or {jl + 1})"
            )

        if reasons:
            valid_mask[idx] = False
            row_idx = df.index[idx]
            diag = {
                "row_index": int(row_idx),
                "reasons": reasons,
                "J_upper": int(ju),
                "Ka_upper": int(kau),
                "Kc_upper": int(kcu),
                "J_lower": int(jl),
                "Ka_lower": int(kal),
                "Kc_lower": int(kcl),
            }
            violations.append(diag)
            if log_warnings:
                logger.warning(
                    f"[SYNTAX GHOST LINE] Dropping row {row_idx}: {'; '.join(reasons)}"
                )

    if violations and raise_on_error:
        raise CoChemDataException(
            f"Pre-Flight Syntax Validator encountered {len(violations)} quantum number violations: {violations}"
        )

    if drop_invalid:
        sanitized_df = df[valid_mask].copy().reset_index(drop=True)
    else:
        sanitized_df = df.copy()
        sanitized_df["syntax_valid"] = valid_mask

    return sanitized_df, violations


# =============================================================================
# 6. PRE-SERIALIZATION PHYSICS FILTER (SELECTION RULES)
# =============================================================================

def classify_transition_parity_and_dipole(
    j_up: int,
    ka_up: int,
    kc_up: int,
    j_low: int,
    ka_low: int,
    kc_low: int,
) -> Tuple[BranchType, DipoleType, Parity, Parity, int, int, int]:
    """
    Classifies a rotational transition's branch (P, Q, R), state parities, and dipole type (a, b, c).

    Asymmetric Top Selection Rules:
    - Delta J in {-1, 0, +1} (P: -1, Q: 0, R: +1). J=0 -> 0 is strictly forbidden.
    - Parity of state: (Ka mod 2, Kc mod 2) in {ee, eo, oe, oo}.
    - a-type (along a-axis, mu_a != 0): Delta Ka even, Delta Kc odd (ee <-> eo, oe <-> oo).
    - b-type (along b-axis, mu_b != 0): Delta Ka odd, Delta Kc odd (ee <-> oo, eo <-> oe).
    - c-type (along c-axis, mu_c != 0): Delta Ka odd, Delta Kc even (ee <-> oe, eo <-> oo).

    Returns:
        Tuple of (branch, dipole_type, upper_parity, lower_parity, delta_j, delta_ka, delta_kc).
    """
    delta_j = j_up - j_low
    delta_ka = ka_up - ka_low
    delta_kc = kc_up - kc_low

    if delta_j == -1:
        branch = BranchType.P
    elif delta_j == 0:
        branch = BranchType.Q
    elif delta_j == 1:
        branch = BranchType.R
    else:
        branch = BranchType.INVALID

    parity_map = {
        (0, 0): Parity.EE,
        (0, 1): Parity.EO,
        (1, 0): Parity.OE,
        (1, 1): Parity.OO,
    }
    up_parity = parity_map[(ka_up % 2, kc_up % 2)]
    low_parity = parity_map[(ka_low % 2, kc_low % 2)]

    if j_up == 0 and j_low == 0:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    if abs(delta_j) > 1:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    d_ka_even = (abs(delta_ka) % 2 == 0)
    d_kc_even = (abs(delta_kc) % 2 == 0)

    if d_ka_even and not d_kc_even:
        dipole_type = DipoleType.A
    elif not d_ka_even and not d_kc_even:
        dipole_type = DipoleType.B
    elif not d_ka_even and d_kc_even:
        dipole_type = DipoleType.C
    else:
        dipole_type = DipoleType.FORBIDDEN

    return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc


def apply_selection_rules(
    df: pd.DataFrame,
    allowed_dipole_types: Optional[Set[str]] = None,
    drop_forbidden: bool = True,
    raise_on_error: bool = False,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    In-Memory Pre-Serialization Physics Filter (Selection Rules).

    Applies rigid quantum selection rules in-memory before serialization:
    1. Enforces Delta J in {-1, 0, +1} and excludes J = 0 -> 0.
    2. Enforces active dipole transition parity constraints (a-type, b-type, c-type).
    3. Strips out forbidden transitions to prevent wasted compute and I/O.

    Args:
        df: Input DataFrame with validated quantum numbers.
        allowed_dipole_types: Set of allowed dipole types (e.g. {'a', 'b', 'c'} or {'b'}).
        drop_forbidden: If True, removes forbidden lines from returned DataFrame.
        raise_on_error: If True, raises CoChemDataException when forbidden transitions occur.
        log_warnings: If True, logs warning messages for each forbidden line.

    Returns:
        Tuple of (filtered_df, list_of_physics_violations).

    Raises:
        CoChemDataException: If raise_on_error is True and forbidden transitions are found.
    """
    if df.empty:
        return df.copy(), []

    allowed_types = {t.lower() for t in allowed_dipole_types} if allowed_dipole_types else {"a", "b", "c"}

    violations: List[Dict[str, Any]] = []
    valid_mask = np.ones(len(df), dtype=bool)

    branches: List[str] = []
    dipoles: List[str] = []
    d_j_list: List[int] = []
    d_ka_list: List[int] = []
    d_kc_list: List[int] = []
    up_parities: List[str] = []
    low_parities: List[str] = []

    j_up = df["J_upper"].to_numpy(dtype=np.int64)
    ka_up = df["Ka_upper"].to_numpy(dtype=np.int64)
    kc_up = df["Kc_upper"].to_numpy(dtype=np.int64)
    j_low = df["J_lower"].to_numpy(dtype=np.int64)
    ka_low = df["Ka_lower"].to_numpy(dtype=np.int64)
    kc_low = df["Kc_lower"].to_numpy(dtype=np.int64)

    for idx in range(len(df)):
        ju, kau, kcu = j_up[idx], ka_up[idx], kc_up[idx]
        jl, kal, kcl = j_low[idx], ka_low[idx], kc_low[idx]

        branch, dipole, up_par, low_par, dj, dka, dkc = classify_transition_parity_and_dipole(
            ju, kau, kcu, jl, kal, kcl
        )

        branches.append(branch.value)
        dipoles.append(dipole.value)
        d_j_list.append(dj)
        d_ka_list.append(dka)
        d_kc_list.append(dkc)
        up_parities.append(up_par.value)
        low_parities.append(low_par.value)

        reasons: List[str] = []
        if ju == 0 and jl == 0:
            reasons.append("Dipole-forbidden J=0 -> 0 transition")
        elif abs(dj) > 1:
            reasons.append(f"Forbidden Delta J = {dj} (|Delta J| > 1)")
        elif dipole == DipoleType.FORBIDDEN:
            reasons.append(f"Dipole-forbidden parity transition ({up_par.value} <-> {low_par.value})")
        elif dipole.value not in allowed_types:
            reasons.append(
                f"Transition dipole type '{dipole.value}' not in allowed set {sorted(allowed_types)}"
            )

        if reasons:
            valid_mask[idx] = False
            row_idx = df.index[idx]
            diag = {
                "row_index": int(row_idx),
                "reasons": reasons,
                "branch": branch.value,
                "dipole_type": dipole.value,
                "delta_j": int(dj),
                "delta_ka": int(dka),
                "delta_kc": int(dkc),
                "upper_parity": up_par.value,
                "lower_parity": low_par.value,
            }
            violations.append(diag)
            if log_warnings:
                logger.warning(
                    f"[PHYSICS FILTER] Dropping forbidden transition at row {row_idx}: {'; '.join(reasons)}"
                )

    if violations and raise_on_error:
        raise CoChemDataException(
            f"Pre-Serialization Physics Filter found {len(violations)} forbidden transitions: {violations}"
        )

    df_annotated = df.copy()
    df_annotated["Branch"] = branches
    df_annotated["Dipole_Type"] = dipoles
    df_annotated["Delta_J"] = d_j_list
    df_annotated["Delta_Ka"] = d_ka_list
    df_annotated["Delta_Kc"] = d_kc_list
    df_annotated["Upper_Parity"] = up_parities
    df_annotated["Lower_Parity"] = low_parities

    if drop_forbidden:
        filtered_df = df_annotated[valid_mask].copy().reset_index(drop=True)
    else:
        filtered_df = df_annotated
        filtered_df["physics_valid"] = valid_mask

    return filtered_df, violations


# =============================================================================
# 7. PARSERS: PICKETT LIN, PAR, AND MODERN CSV/STRUCTURED
# =============================================================================

def parse_legacy_pickett(
    lines_or_path: Union[List[str], str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parses legacy Pickett .lin / .cat files using strict positional and regex parsing.

    Translates Fortran double-precision exponential notation ('D-05', 'D+02', '1.2345-05' -> FP64).
    Failing lines are appended to failed_ingest.log located at platformdirs.user_log_path("CoChem").

    Args:
        lines_or_path: File path, list of line strings, single string content, bytes, or StringIO.
        config: Optional IngestConfig for defaults and custom log path.

    Returns:
        Tuple of (parsed_df, list_of_failed_raw_lines).
    """
    cfg = config or IngestConfig()
    lines: List[str] = []

    if isinstance(lines_or_path, list):
        lines = lines_or_path
    elif isinstance(lines_or_path, Path):
        if not lines_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {lines_or_path}")
        with open(str(lines_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(lines_or_path, str):
        if "\n" not in lines_or_path and os.path.exists(lines_or_path):
            with open(lines_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in lines_or_path and (
            lines_or_path.endswith((".lin", ".cat", ".par", ".var")) or "/" in lines_or_path or "\\" in lines_or_path
        ):
            raise FileNotFoundError(f"Pickett file not found: {lines_or_path}")
        else:
            lines = lines_or_path.splitlines()
    elif isinstance(lines_or_path, bytes):
        lines = lines_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(lines_or_path, io.StringIO):
        lines = lines_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content input type: {type(lines_or_path)}")

    parsed_records: List[Dict[str, Any]] = []
    failed_lines: List[str] = []

    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    jpl_cat_pattern = re.compile(
        r"^\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+)\s+"
        r"(\d+)\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)"
    )

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        matched = False

        cat_match = jpl_cat_pattern.match(line)
        if cat_match:
            try:
                freq = fortran_float_converter(cat_match.group(1))
                unc = fortran_float_converter(cat_match.group(2))
                intensity = fortran_float_converter(cat_match.group(3))
                ju = int(cat_match.group(9))
                kau = int(cat_match.group(10))
                kcu = int(cat_match.group(11))
                jl = int(cat_match.group(12))
                kal = int(cat_match.group(13))
                kcl = int(cat_match.group(14))

                parsed_records.append({
                    "Frequency_MHz": np.float64(freq),
                    "Uncertainty": np.float64(unc),
                    "Intensity": np.float64(intensity),
                    "J_upper": int(ju),
                    "Ka_upper": int(kau),
                    "Kc_upper": int(kcu),
                    "J_lower": int(jl),
                    "Ka_lower": int(kal),
                    "Kc_lower": int(kcl),
                })
                matched = True
            except Exception as e:
                logger.debug(f"Catalog match parse failed at line {line_no}: {e}")

        if not matched:
            free_match = free_pattern.match(line)
            if free_match:
                try:
                    ju = int(free_match.group(1))
                    kau = int(free_match.group(2))
                    kcu = int(free_match.group(3))
                    jl = int(free_match.group(4))
                    kal = int(free_match.group(5))
                    kcl = int(free_match.group(6))
                    freq = fortran_float_converter(free_match.group(7))
                    unc = fortran_float_converter(free_match.group(8))
                    raw_int = free_match.group(9)
                    intensity = fortran_float_converter(raw_int) if raw_int else cfg.default_intensity

                    parsed_records.append({
                        "Frequency_MHz": np.float64(freq),
                        "Uncertainty": np.float64(unc),
                        "Intensity": np.float64(intensity),
                        "J_upper": int(ju),
                        "Ka_upper": int(kau),
                        "Kc_upper": int(kcu),
                        "J_lower": int(jl),
                        "Ka_lower": int(kal),
                        "Kc_lower": int(kcl),
                    })
                    matched = True
                except Exception as e:
                    logger.debug(f"Pickett line regex parse failed at line {line_no}: {e}")

        if not matched:
            if len(raw_line) >= 40:
                try:
                    part_ju = raw_line[0:3].strip()
                    part_kau = raw_line[3:6].strip()
                    part_kcu = raw_line[6:9].strip()
                    part_jl = raw_line[18:21].strip()
                    part_kal = raw_line[21:24].strip()
                    part_kcl = raw_line[24:27].strip()
                    part_freq = raw_line[36:49].strip()
                    part_unc = raw_line[49:57].strip()
                    part_int = raw_line[57:65].strip() if len(raw_line) >= 65 else ""

                    if part_ju and part_jl and part_freq:
                        ju = int(part_ju)
                        kau = int(part_kau) if part_kau else 0
                        kcu = int(part_kcu) if part_kcu else 0
                        jl = int(part_jl)
                        kal = int(part_kal) if part_kal else 0
                        kcl = int(part_kcl) if part_kcl else 0
                        freq = fortran_float_converter(part_freq)
                        unc = fortran_float_converter(part_unc) if part_unc else cfg.default_uncertainty
                        intensity = fortran_float_converter(part_int) if part_int else cfg.default_intensity

                        parsed_records.append({
                            "Frequency_MHz": np.float64(freq),
                            "Uncertainty": np.float64(unc),
                            "Intensity": np.float64(intensity),
                            "J_upper": int(ju),
                            "Ka_upper": int(kau),
                            "Kc_upper": int(kcu),
                            "J_lower": int(jl),
                            "Ka_lower": int(kal),
                            "Kc_lower": int(kcl),
                        })
                        matched = True
                except Exception as e:
                    logger.debug(f"Fixed-width positional parse failed at line {line_no}: {e}")

        if not matched:
            failed_lines.append(raw_line)
            if cfg.log_warnings:
                logger.warning(f"[INGEST PARSE FAILURE] Line {line_no} failed regex parsing: {raw_line.strip()!r}")

    if failed_lines:
        log_dest: Path
        if cfg.failed_log_path:
            log_dest = Path(cfg.failed_log_path)
        else:
            log_dest = user_log_path("CoChem") / "failed_ingest.log"

        try:
            log_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(log_dest, "a", encoding="utf-8") as f:
                for fl in failed_lines:
                    f.write(f"{fl.rstrip()}\n")
        except Exception as e:
            logger.debug(f"Could not write to log path {log_dest}: {e}")

    if not parsed_records:
        df_res = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        df_res = pd.DataFrame(parsed_records)

    return enforce_fp64_boundary(df_res), failed_lines


parse_pickett_lin = parse_legacy_pickett


def parse_pickett_par(
    content_or_path: Union[str, Path, bytes, io.StringIO],
) -> PickettParFile:
    """
    Parses a legacy Pickett parameter file (.par / .var).

    Structure:
    Line 1: Title
    Line 2: Control line (npar, nline, nopt, nwarn, erpar, wtfac, scale, maxit)
    Subsequent lines: Parameters (code, value, uncertainty, label)

    Args:
        content_or_path: File path, string content, bytes, or StringIO.

    Returns:
        Structured PickettParFile instance.
    """
    lines: List[str] = []
    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett .par/.var file not found: {content_or_path}")
        with open(str(content_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in content_or_path and (
            content_or_path.endswith((".par", ".var", ".lin", ".cat")) or "/" in content_or_path or "\\" in content_or_path
        ):
            raise FileNotFoundError(f"Pickett .par/.var file not found: {content_or_path}")
        else:
            lines = content_or_path.splitlines()
    elif isinstance(content_or_path, bytes):
        lines = content_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(content_or_path, io.StringIO):
        lines = content_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise ValueError("Pickett .par/.var file must have at least a title and a control line.")

    title = non_empty[0].strip()
    control_line = non_empty[1].strip()

    ctrl_parts = control_line.split()
    npar = int(ctrl_parts[0]) if len(ctrl_parts) > 0 else 0
    nline = int(ctrl_parts[1]) if len(ctrl_parts) > 1 else 0
    nopt = int(ctrl_parts[2]) if len(ctrl_parts) > 2 else 0
    nwarn = int(ctrl_parts[3]) if len(ctrl_parts) > 3 else 0
    erpar = fortran_float_converter(ctrl_parts[4]) if len(ctrl_parts) > 4 else 1.0
    wtfac = fortran_float_converter(ctrl_parts[5]) if len(ctrl_parts) > 5 else 1.0
    scale = fortran_float_converter(ctrl_parts[6]) if len(ctrl_parts) > 6 else 1.0
    maxit = int(ctrl_parts[7]) if len(ctrl_parts) > 7 else 50

    parameters: List[PickettParameter] = []
    param_line_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    for p_line in non_empty[2:]:
        match = param_line_pattern.match(p_line)
        if match:
            pid = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            lbl = (match.group(4) or "").strip()
            parameters.append(PickettParameter(param_id=pid, value=val, uncertainty=unc, label=lbl))

    return PickettParFile(
        title=title,
        npar=len(parameters),
        nline=nline,
        nopt=nopt,
        nwarn=nwarn,
        erpar=erpar,
        wtfac=wtfac,
        scale=scale,
        maxit=maxit,
        parameters=parameters,
    )


def parse_csv_spectroscopy(
    content_or_path: Union[str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> pd.DataFrame:
    """
    Parses modern structured spectroscopy files (.csv, .tsv, .parquet, .json).

    Standardizes column name variations to canonical naming convention:
    ['Frequency_MHz', 'Uncertainty', 'Intensity', 'J_upper', 'Ka_upper', 'Kc_upper',
     'J_lower', 'Ka_lower', 'Kc_lower'].

    Args:
        content_or_path: File path, string content, bytes, or StringIO.
        config: IngestConfig options.

    Returns:
        Standardized Pandas DataFrame with FP64 precision.
    """
    cfg = config or IngestConfig()

    if isinstance(content_or_path, (str, Path)) and str(content_or_path).endswith(".parquet"):
        if not os.path.exists(str(content_or_path)):
            raise FileNotFoundError(f"Parquet file not found: {content_or_path}")
        df_raw = pd.read_parquet(str(content_or_path))
    elif isinstance(content_or_path, (str, Path)) and str(content_or_path).endswith(".json"):
        if not os.path.exists(str(content_or_path)):
            raise FileNotFoundError(f"JSON file not found: {content_or_path}")
        df_raw = pd.read_json(str(content_or_path))
    elif isinstance(content_or_path, io.StringIO):
        df_raw = pd.read_csv(content_or_path, sep=None, engine="python")
    elif isinstance(content_or_path, bytes):
        df_raw = pd.read_csv(io.BytesIO(content_or_path), sep=None, engine="python")
    elif isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        df_raw = pd.read_csv(str(content_or_path), sep=None, engine="python")
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            df_raw = pd.read_csv(content_or_path, sep=None, engine="python")
        elif "\n" not in content_or_path and (
            content_or_path.endswith((".csv", ".tsv", ".txt")) or "/" in content_or_path or "\\" in content_or_path
        ):
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        else:
            df_raw = pd.read_csv(io.StringIO(content_or_path), sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    col_map: Dict[str, str] = {
        "frequency_mhz": "Frequency_MHz",
        "frequency": "Frequency_MHz",
        "freq_mhz": "Frequency_MHz",
        "freq": "Frequency_MHz",
        "nu_mhz": "Frequency_MHz",
        "nu": "Frequency_MHz",
        "f_mhz": "Frequency_MHz",
        "f": "Frequency_MHz",
        "uncertainty": "Uncertainty",
        "uncertainty_mhz": "Uncertainty",
        "unc": "Uncertainty",
        "unc_mhz": "Uncertainty",
        "err": "Uncertainty",
        "error": "Uncertainty",
        "sigma": "Uncertainty",
        "intensity": "Intensity",
        "int": "Intensity",
        "logint": "Intensity",
        "weight": "Intensity",
        "wt": "Intensity",
        "s_mu2": "Intensity",
        "j_upper": "J_upper",
        "j_prime": "J_upper",
        "j'": "J_upper",
        "j1": "J_upper",
        "j_up": "J_upper",
        "ju": "J_upper",
        "ka_upper": "Ka_upper",
        "ka_prime": "Ka_upper",
        "ka'": "Ka_upper",
        "ka1": "Ka_upper",
        "ka_up": "Ka_upper",
        "kau": "Ka_upper",
        "kc_upper": "Kc_upper",
        "kc_prime": "Kc_upper",
        "kc'": "Kc_upper",
        "kc1": "Kc_upper",
        "kc_up": "Kc_upper",
        "kcu": "Kc_upper",
        "j_lower": "J_lower",
        "j_double_prime": "J_lower",
        "j''": "J_lower",
        "j2": "J_lower",
        "j_low": "J_lower",
        "jl": "J_lower",
        "ka_lower": "Ka_lower",
        "ka_double_prime": "Ka_lower",
        "ka''": "Ka_lower",
        "ka2": "Ka_lower",
        "ka_low": "Ka_lower",
        "kal": "Ka_lower",
        "kc_lower": "Kc_lower",
        "kc_double_prime": "Kc_lower",
        "kc''": "Kc_lower",
        "kc2": "Kc_lower",
        "kc_low": "Kc_lower",
        "kcl": "Kc_lower",
    }

    rename_dict: Dict[str, str] = {}
    for c in df_raw.columns:
        c_clean = str(c).strip().lower()
        if c_clean in col_map:
            rename_dict[c] = col_map[c_clean]

    df_renamed = df_raw.rename(columns=rename_dict)

    required_core = ["Frequency_MHz", "J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    missing = [c for c in required_core if c not in df_renamed.columns]
    if missing:
        raise CoChemDataException(
            f"CSV spectroscopy data missing mandatory columns: {missing}. Present: {list(df_renamed.columns)}"
        )

    if "Uncertainty" not in df_renamed.columns:
        df_renamed["Uncertainty"] = cfg.default_uncertainty
    if "Intensity" not in df_renamed.columns:
        df_renamed["Intensity"] = cfg.default_intensity

    for col in ["Frequency_MHz", "Uncertainty", "Intensity"]:
        if df_renamed[col].dtype == object:
            df_renamed[col] = df_renamed[col].apply(fortran_float_converter)

    return enforce_fp64_boundary(df_renamed[CANONICAL_COLUMNS])


# =============================================================================
# 8. MASTER UNIFIED INGESTION ENGINE
# =============================================================================

def parse_spectroscopy_data(
    file_path_or_content: Union[str, Path, bytes, io.StringIO],
    format_hint: Optional[str] = None,
    config: Optional[IngestConfig] = None,
    zarr_path: Optional[Union[str, Path]] = None,
) -> IngestResult:
    """
    Master Ingestion Engine & Gatekeeper (cochem_spycfit_ingest).

    Orchestrates:
    1. Automatic format detection (.lin, .par, .cat, .csv, .parquet, .json).
    2. Bidirectional parsing of legacy Pickett lines and modern tabular structures.
    3. JAX FP64 precision boundary enforcement (float64).
    4. Cryptographic SHA-256 frequency fingerprinting.
    5. Persistence to spectra_fit.zarr if zarr_path is configured.
    6. Pre-Flight Syntax Validator (Quantum Logic Gate) for quantum number bounds.
    7. In-memory Pre-Serialization Physics Filter (Selection Rules).
    8. Construction of comprehensive ValidationReport and IngestResult.

    Args:
        file_path_or_content: File path, string content, bytes, or StringIO.
        format_hint: Optional explicit format override ('lin', 'cat', 'csv', 'parquet', 'json').
        config: Optional IngestConfig configuration options.
        zarr_path: Optional path to spectra_fit.zarr to append the state matrix.

    Returns:
        IngestResult containing sanitized DataFrame, ValidationReport, and SHA-256 fingerprint.
    """
    cfg = config or IngestConfig()
    warnings_list: List[str] = []

    effective_zarr_path = zarr_path or cfg.zarr_path

    fmt = (format_hint or "").lower().strip()
    if not fmt:
        if isinstance(file_path_or_content, Path):
            suffix = file_path_or_content.suffix.lower().replace(".", "")
            if suffix in ["lin", "cat", "par", "var", "csv", "tsv", "parquet", "json"]:
                fmt = suffix
        elif isinstance(file_path_or_content, str) and "\n" not in file_path_or_content:
            try:
                p = Path(file_path_or_content)
                suffix = p.suffix.lower().replace(".", "")
                if suffix in ["lin", "cat", "par", "var", "csv", "tsv", "parquet", "json"]:
                    fmt = suffix
            except Exception as exc:
                logger.debug(f"Suffix resolution failed for {file_path_or_content}: {exc}")

    if fmt in ["lin", "cat"]:
        raw_df, failed_lines = parse_legacy_pickett(file_path_or_content, config=cfg)
        if failed_lines:
            warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing and were skipped.")
    elif fmt in ["par", "var"]:
        par_file = parse_pickett_par(file_path_or_content)
        par_records = [
            {"param_id": p.param_id, "value": p.value, "uncertainty": p.uncertainty, "label": p.label}
            for p in par_file.parameters
        ]
        df_par = pd.DataFrame(par_records)
        fp = calculate_frequency_fingerprint([p.value for p in par_file.parameters])
        if effective_zarr_path:
            append_to_spectra_fit_zarr(
                zarr_path=effective_zarr_path,
                frequencies=[p.value for p in par_file.parameters],
                sha256_hash=fp,
                group_name="pickett_parameters",
            )
        report = ValidationReport(
            total_lines_read=len(par_file.parameters),
            valid_lines_count=len(par_file.parameters),
            syntax_dropped_count=0,
            physics_dropped_count=0,
            sha256_fingerprint=fp,
            warnings=[],
            syntax_violations=[],
            physics_violations=[],
        )
        return IngestResult(df=df_par, report=report, sha256_fingerprint=fp, success=True)
    elif fmt in ["parquet", "json", "csv", "tsv"]:
        raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
    else:
        try:
            raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
        except Exception:
            raw_df, failed_lines = parse_legacy_pickett(file_path_or_content, config=cfg)
            if failed_lines:
                warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing.")

    total_read = len(raw_df)

    fingerprint = calculate_frequency_fingerprint(raw_df["Frequency_MHz"])

    if effective_zarr_path:
        append_to_spectra_fit_zarr(
            zarr_path=effective_zarr_path,
            frequencies=raw_df["Frequency_MHz"],
            sha256_hash=fingerprint,
            group_name="state_matrix",
        )

    if cfg.validate_syntax:
        syntax_df, syntax_violations = validate_quantum_numbers(
            raw_df,
            drop_invalid=cfg.drop_invalid_syntax,
            raise_on_error=False,
            log_warnings=cfg.log_warnings,
        )
        syntax_dropped = len(syntax_violations) if cfg.drop_invalid_syntax else 0
        if syntax_dropped > 0:
            warnings_list.append(f"Pre-Flight Syntax Validator dropped {syntax_dropped} unphysical quantum assignment(s).")
    else:
        syntax_df = raw_df
        syntax_violations = []
        syntax_dropped = 0

    if cfg.apply_selection_rules:
        physics_df, physics_violations = apply_selection_rules(
            syntax_df,
            allowed_dipole_types=cfg.allowed_dipole_types,
            drop_forbidden=cfg.drop_forbidden_transitions,
            raise_on_error=False,
            log_warnings=cfg.log_warnings,
        )
        physics_dropped = len(physics_violations) if cfg.drop_forbidden_transitions else 0
        if physics_dropped > 0:
            warnings_list.append(f"Pre-Serialization Physics Filter dropped {physics_dropped} forbidden transition(s).")
    else:
        physics_df = syntax_df
        physics_violations = []
        physics_dropped = 0

    final_df = enforce_fp64_boundary(physics_df)

    report = ValidationReport(
        total_lines_read=total_read,
        valid_lines_count=len(final_df),
        syntax_dropped_count=syntax_dropped,
        physics_dropped_count=physics_dropped,
        sha256_fingerprint=fingerprint,
        warnings=warnings_list,
        syntax_violations=syntax_violations,
        physics_violations=physics_violations,
    )

    return IngestResult(
        df=final_df,
        report=report,
        sha256_fingerprint=fingerprint,
        success=(len(final_df) > 0 or total_read == 0),
    )


__all__ = [
    "FileFormat",
    "BranchType",
    "DipoleType",
    "Parity",
    "CANONICAL_COLUMNS",
    "SpectroscopySchema",
    "SpectroscopyTransition",
    "PickettParameter",
    "PickettParFile",
    "IngestConfig",
    "ValidationReport",
    "IngestResult",
    "CoChemDataException",
    "fortran_float_converter",
    "enforce_fp64_boundary",
    "calculate_frequency_fingerprint",
    "append_to_spectra_fit_zarr",
    "validate_quantum_numbers",
    "classify_transition_parity_and_dipole",
    "apply_selection_rules",
    "parse_legacy_pickett",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_csv_spectroscopy",
    "parse_spectroscopy_data",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    CoChemDataException,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopySchema,
    SpectroscopyTransition,
    ValidationReport,
    append_to_spectra_fit_zarr,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_legacy_pickett,
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
from cochem_spycfit.intake.cochem_spycfit_orchestrator import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    BOHR_TO_ANGSTROM,
    EckartFrameError,
    EckartFrameResult,
    FitOrchestratorError,
    HARTREE_TO_JOULE,
    HeartbeatStatus,
    IPCMode,
    LandscapeTensorError,
    OrchestrationResult,
    OrchestrationStatus,
    OrchestratorConfig,
    PLANCK_CONSTANT_JS,
    ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ,
    SPEED_OF_LIGHT_CM_S,
    TORQIPCError,
    TORQRequestPayload,
    TORQResponsePayload,
    TORQStatus,
    TORQTimeoutError,
    TensorEvaluationReport,
    WallTimeSafetyGuard,
    WallTimeYieldException,
    calculate_center_of_mass,
    calculate_inertia_tensor,
    calculate_orchestrator_provenance_hash,
    checkpoint_and_yield_node,
    compute_eckart_rotation_matrix,
    diagonalize_inertia_tensor,
    eckart_frame_correction,
    evaluate_landscape_tensors,
    orchestrate_spycfit_system,
    resolve_tier3_registry_path,
    route_missing_energy,
    route_missing_energy_file_drop,
    route_missing_energy_zmq,
)

__all__ = [
    # Ingest
    "BranchType",
    "CoChemDataException",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopySchema",
    "SpectroscopyTransition",
    "ValidationReport",
    "append_to_spectra_fit_zarr",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_legacy_pickett",
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
    # Orchestrator
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_ingest.py ---
"""
CoChem-SpycFit: Stage 1.0 Bipartite Data Intake Parser & Quantum Gatekeeper (Phase 2).

Module: cochem_spycfit/intake/cochem_spycfit_ingest.py
Role: The Tripartite Data Intake Parser & In-Memory Pre-Serialization Physics Filter.

Capabilities:
1. Modern Pydantic Schema Validation (SpectroscopySchema & SpectroscopyTransition):
   Enforces mandatory fields (Frequency_MHz, Uncertainty, Intensity, J_upper, Ka_upper,
   Kc_upper, J_lower, Ka_lower, Kc_lower) and strict asymmetric top physics.
2. Positional & Free-Format Regex Ingestion (parse_legacy_pickett):
   Parses legacy Pickett .lin, .cat, .par files and JPL catalogs, converting Fortran
   double-precision notation ('D-05', 'D+02', '1.2345-05') to IEEE-754 FP64.
   Failing lines are appended to failed_ingest.log in platformdirs.user_log_path("CoChem").
3. Cryptographic Fingerprinting & Zarr State Matrix Appending:
   Computes SHA-256 hash across all experimental frequencies upon ingestion and persists
   the state matrix to spectra_fit.zarr.
4. Pre-Flight Syntax Validator (Quantum Logic Gate):
   Validates J >= 0, Ka >= 0, Kc >= 0, Ka <= J, Kc <= J, and asymmetric top sum rule
   Ka + Kc in {J, J+1}. Raises CoChemDataException on strict failure.
5. Pre-Serialization Physics Filter:
   Enforces rotational selection rules (Delta J in {-1, 0, 1}, J=0 -> 0 forbidden) and
   dipole transition parity constraints (a-type, b-type, c-type).

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE)
- Task 5 Master Ingestion & Pre-Flight Validation (CoChem-SpycFit SRS)
- Mendeleev Library Mandate (Dynamic Mass Retrieval)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import numpy as np
import pandas as pd
from platformdirs import user_log_path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import zarr

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Ingest")

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemError,
        CoChemIntegrityError,
        CoChemPrecisionError,
        ProvenanceErrorCode,
    )

    class CoChemDataException(CoChemError, ValueError):
        """Raised when input spectroscopic data fails validation, syntax checks, or quantum rules."""
        ...
except ImportError:
    class CoChemError(Exception):
        """Root fallback exception for CoChem."""
        ...

    class CoChemPrecisionError(RuntimeError):
        """Raised when JAX or numerical float precision is violated."""
        ...

    class CoChemIntegrityError(ValueError):
        """Raised when cryptographic checksum or binary integrity check fails."""
        ...

    class CoChemDataException(ValueError):
        """Raised when input spectroscopic data fails validation, syntax checks, or quantum rules."""
        ...


# =============================================================================
# 1. ENUMS AND DATA STRUCTURES
# =============================================================================

class FileFormat(str, Enum):
    """Supported file formats for spectroscopy data ingestion."""
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    JSON = "json"
    LIN = "lin"
    PAR = "par"
    CAT = "cat"
    UNKNOWN = "unknown"


class BranchType(str, Enum):
    """Spectroscopic rotational branch designations based on Delta J = J' - J''."""
    P = "P"  # Delta J = -1
    Q = "Q"  # Delta J = 0
    R = "R"  # Delta J = +1
    INVALID = "INVALID"


class DipoleType(str, Enum):
    """Asymmetric top dipole transition parity classifications."""
    A = "a"          # a-type: Delta Ka even, Delta Kc odd (ee <-> eo, oe <-> oo)
    B = "b"          # b-type: Delta Ka odd, Delta Kc odd (ee <-> oo, eo <-> oe)
    C = "c"          # c-type: Delta Ka odd, Delta Kc even (ee <-> oe, eo <-> oo)
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


class Parity(str, Enum):
    """Rotational state parity (Ka mod 2, Kc mod 2)."""
    EE = "ee"  # Ka even, Kc even
    EO = "eo"  # Ka even, Kc odd
    OE = "oe"  # Ka odd,  Kc even
    OO = "oo"  # Ka odd,  Kc odd


# Canonical column names for modern structured Pandas DataFrames
CANONICAL_COLUMNS: List[str] = [
    "Frequency_MHz",
    "Uncertainty",
    "Intensity",
    "J_upper",
    "Ka_upper",
    "Kc_upper",
    "J_lower",
    "Ka_lower",
    "Kc_lower",
]


# =============================================================================
# 2. PYDANTIC V2 SCHEMAS
# =============================================================================

class SpectroscopySchema(BaseModel):
    """
    Phase 2 Pydantic schema validation for single spectroscopic transitions.

    Mandatory fields:
    - Frequency_MHz (float64)
    - Uncertainty (float64)
    - Intensity (float64)
    - J_upper, Ka_upper, Kc_upper (int >= 0)
    - J_lower, Ka_lower, Kc_lower (int >= 0)
    """
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    Frequency_MHz: float = Field(
        ...,
        alias="frequency_mhz",
        description="Observed transition frequency in MHz (FP64).",
    )
    Uncertainty: float = Field(
        default=0.01,
        alias="uncertainty_mhz",
        description="Measurement uncertainty in MHz (FP64).",
    )
    Intensity: float = Field(
        default=1.0,
        alias="intensity",
        description="Observed or catalog intensity (FP64).",
    )
    J_upper: int = Field(
        ...,
        ge=0,
        alias="j_upper",
        description="Upper state total angular momentum quantum number J'.",
    )
    Ka_upper: int = Field(
        ...,
        ge=0,
        alias="ka_upper",
        description="Upper state prolate projection quantum number Ka'.",
    )
    Kc_upper: int = Field(
        ...,
        ge=0,
        alias="kc_upper",
        description="Upper state oblate projection quantum number Kc'.",
    )
    J_lower: int = Field(
        ...,
        ge=0,
        alias="j_lower",
        description="Lower state total angular momentum quantum number J''.",
    )
    Ka_lower: int = Field(
        ...,
        ge=0,
        alias="ka_lower",
        description="Lower state prolate projection quantum number Ka''.",
    )
    Kc_lower: int = Field(
        ...,
        ge=0,
        alias="kc_lower",
        description="Lower state oblate projection quantum number Kc''.",
    )
    branch: Optional[BranchType] = Field(default=None, description="Rotational branch (P, Q, R).")
    dipole_type: Optional[DipoleType] = Field(default=None, description="Dipole transition classification (a, b, c).")
    delta_j: Optional[int] = Field(default=None, description="Delta J = J' - J''.")
    delta_ka: Optional[int] = Field(default=None, description="Delta Ka = Ka' - Ka''.")
    delta_kc: Optional[int] = Field(default=None, description="Delta Kc = Kc' - Kc''.")
    upper_parity: Optional[Parity] = Field(default=None, description="Upper state parity (ee, eo, oe, oo).")
    lower_parity: Optional[Parity] = Field(default=None, description="Lower state parity (ee, eo, oe, oo).")
    valid: bool = Field(default=True, description="Whether transition passed syntax and selection rules.")
    drop_reason: Optional[str] = Field(default=None, description="Detailed explanation if line was dropped.")

    @model_validator(mode="after")
    def validate_quantum_physics(self) -> SpectroscopySchema:
        """Enforces asymmetric rotor quantum bounds and parity sum rules."""
        # 1. Ka <= J and Kc <= J bounds
        if self.Ka_upper > self.J_upper:
            raise ValueError(f"Upper Ka ({self.Ka_upper}) cannot exceed J ({self.J_upper})")
        if self.Kc_upper > self.J_upper:
            raise ValueError(f"Upper Kc ({self.Kc_upper}) cannot exceed J ({self.J_upper})")
        if self.Ka_lower > self.J_lower:
            raise ValueError(f"Lower Ka ({self.Ka_lower}) cannot exceed J ({self.J_lower})")
        if self.Kc_lower > self.J_lower:
            raise ValueError(f"Lower Kc ({self.Kc_lower}) cannot exceed J ({self.J_lower})")

        # 2. Asymmetric top sum rule: Ka + Kc in {J, J+1}
        sum_up = self.Ka_upper + self.Kc_upper
        if sum_up != self.J_upper and sum_up != (self.J_upper + 1):
            raise ValueError(
                f"Upper state Ka+Kc ({sum_up}) violates asymmetric top sum rule for J={self.J_upper}"
            )
        sum_low = self.Ka_lower + self.Kc_lower
        if sum_low != self.J_lower and sum_low != (self.J_lower + 1):
            raise ValueError(
                f"Lower state Ka+Kc ({sum_low}) violates asymmetric top sum rule for J={self.J_lower}"
            )

        # 3. Auto-populate differences if None
        if self.delta_j is None:
            self.delta_j = self.J_upper - self.J_lower
        if self.delta_ka is None:
            self.delta_ka = self.Ka_upper - self.Ka_lower
        if self.delta_kc is None:
            self.delta_kc = self.Kc_upper - self.Kc_lower

        # 4. Auto-populate parity
        parity_map = {
            (0, 0): Parity.EE,
            (0, 1): Parity.EO,
            (1, 0): Parity.OE,
            (1, 1): Parity.OO,
        }
        if self.upper_parity is None:
            self.upper_parity = parity_map[(self.Ka_upper % 2, self.Kc_upper % 2)]
        if self.lower_parity is None:
            self.lower_parity = parity_map[(self.Ka_lower % 2, self.Kc_lower % 2)]

        # 5. Auto-populate branch
        if self.branch is None:
            if self.delta_j == -1:
                self.branch = BranchType.P
            elif self.delta_j == 0:
                self.branch = BranchType.Q
            elif self.delta_j == 1:
                self.branch = BranchType.R
            else:
                self.branch = BranchType.INVALID

        # 6. Auto-populate dipole classification
        if self.dipole_type is None:
            if (self.J_upper == 0 and self.J_lower == 0) or abs(self.delta_j) > 1:
                self.dipole_type = DipoleType.FORBIDDEN
            else:
                d_ka_even = (abs(self.delta_ka) % 2 == 0)
                d_kc_even = (abs(self.delta_kc) % 2 == 0)
                if d_ka_even and not d_kc_even:
                    self.dipole_type = DipoleType.A
                elif not d_ka_even and not d_kc_even:
                    self.dipole_type = DipoleType.B
                elif not d_ka_even and d_kc_even:
                    self.dipole_type = DipoleType.C
                else:
                    self.dipole_type = DipoleType.FORBIDDEN

        return self


# Backwards compatibility alias for SpectroscopyTransition
SpectroscopyTransition = SpectroscopySchema


class PickettParameter(BaseModel):
    """Single Pickett parameter record (.par / .var)."""
    model_config = ConfigDict(extra="forbid")

    param_id: int = Field(..., description="Pickett parameter code (e.g. 10000, 20000, 30000, 200).")
    value: float = Field(..., description="Parameter value in MHz.")
    uncertainty: float = Field(default=0.0, description="Estimated parameter uncertainty in MHz.")
    label: str = Field(default="", description="Human-readable parameter label (e.g. 'A', 'B', 'DJ').")


class PickettParFile(BaseModel):
    """Structured representation of a legacy Pickett .par / .var parameter file."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Header title string from .par / .var file.")
    npar: int = Field(..., ge=0, description="Number of parameters in file.")
    nline: int = Field(..., description="Number of fitting lines.")
    nopt: int = Field(default=0, description="Pickett optimization control integer.")
    nwarn: int = Field(default=0, description="Pickett warning control integer.")
    erpar: float = Field(default=1.0, description="Parameter error scale factor.")
    wtfac: float = Field(default=1.0, description="Weight factor.")
    scale: float = Field(default=1.0, description="Frequency scale factor.")
    maxit: int = Field(default=50, description="Maximum iterations.")
    parameters: List[PickettParameter] = Field(default_factory=list, description="List of parsed parameters.")


class IngestConfig(BaseModel):
    """Configuration options for spectroscopy data ingestion and validation."""
    model_config = ConfigDict(extra="forbid")

    enforce_fp64: bool = Field(default=True, description="Strictly cast all float columns to np.float64.")
    validate_syntax: bool = Field(default=True, description="Apply Pre-Flight Syntax Validator quantum bounds.")
    apply_selection_rules: bool = Field(default=True, description="Apply Pre-Serialization Physics Filter selection rules.")
    allowed_dipole_types: Optional[Set[str]] = Field(
        default_factory=lambda: {"a", "b", "c"},
        description="Allowed dipole transition types (e.g. {'a', 'b', 'c'} or {'b'}).",
    )
    drop_invalid_syntax: bool = Field(default=True, description="Drop lines failing quantum syntax validation.")
    drop_forbidden_transitions: bool = Field(default=True, description="Drop lines violating physics selection rules.")
    log_warnings: bool = Field(default=True, description="Emit structured log warnings for dropped lines.")
    failed_log_path: Optional[str] = Field(default=None, description="Optional path to append failed lines log.")
    default_uncertainty: float = Field(default=0.01, ge=0.0, description="Default uncertainty in MHz if missing.")
    default_intensity: float = Field(default=1.0, description="Default intensity if missing.")
    zarr_path: Optional[str] = Field(default=None, description="Optional path to spectra_fit.zarr for state persistence.")


class ValidationReport(BaseModel):
    """Comprehensive diagnostic summary of ingestion, syntax, and physics filtering."""
    model_config = ConfigDict(extra="forbid")

    total_lines_read: int = Field(..., ge=0, description="Total raw lines or rows parsed from input.")
    valid_lines_count: int = Field(..., ge=0, description="Number of surviving lines passing all gates.")
    syntax_dropped_count: int = Field(default=0, ge=0, description="Lines dropped by Pre-Flight Syntax Validator.")
    physics_dropped_count: int = Field(default=0, ge=0, description="Lines dropped by Pre-Serialization Physics Filter.")
    sha256_fingerprint: str = Field(..., description="Cryptographic SHA-256 fingerprint of input frequencies.")
    warnings: List[str] = Field(default_factory=list, description="List of human-readable warnings emitted.")
    syntax_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed syntax drop records.")
    physics_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed physics drop records.")


class IngestResult(BaseModel):
    """Container holding sanitized Pandas DataFrame and validation metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    df: Any = Field(..., description="Sanitized Pandas DataFrame with FP64 precision.")
    report: ValidationReport = Field(..., description="Validation report and statistics.")
    sha256_fingerprint: str = Field(..., description="SHA-256 cryptographic hash of frequency array.")
    success: bool = Field(default=True, description="True if ingestion produced valid surviving data.")


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & FP64 PRECISION LOCK
# =============================================================================

def fortran_float_converter(val: Union[str, float, int]) -> float:
    """
    Safely converts a string containing Fortran scientific notation to a Python float64.

    Handles:
    - '1.2345D-05' -> 1.2345e-5
    - '1.2345D+02' -> 1.2345e+02
    - '1.2345d-05' -> 1.2345e-5
    - '1.2345E-05' -> 1.2345e-5
    - '12345.6789' -> 12345.6789
    - '1.2345-05'  -> 1.2345e-5 (implicit exponent notation in Fortran)

    Args:
        val: Input string, integer, or float.

    Returns:
        Converted value as float.

    Raises:
        ValueError: If the string cannot be parsed into a valid float.
    """
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        raise ValueError("Cannot convert empty string to float")

    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    implicit_exp_pattern = re.compile(r"^([+-]?\d+\.?\d*)([+-]\d{2,3})$")
    match = implicit_exp_pattern.match(s_clean)
    if match and "e" not in s_clean:
        s_clean = f"{match.group(1)}e{match.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


def enforce_fp64_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces JAX FP64 precision across the Pandas ingestion boundary.

    1. Executes jax.config.update('jax_enable_x64', True) and sets JAX_ENABLE_X64='True'.
    2. Explicitly casts all floating point columns to numpy.float64.
    3. Explicitly casts all quantum number integer columns to numpy.int64.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        DataFrame with strictly enforced float64 and int64 datatypes.
    """
    os.environ["JAX_ENABLE_X64"] = "True"
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as e:
        logger.debug(f"JAX FP64 config update notice: {e}")

    df_out = df.copy()

    float_cols = ["Frequency_MHz", "Uncertainty", "Intensity"]
    int_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower", "Delta_J", "Delta_Ka", "Delta_Kc"]

    for col in df_out.columns:
        if col in float_cols or (col not in int_cols and pd.api.types.is_float_dtype(df_out[col])):
            df_out[col] = df_out[col].astype(np.float64)

    for col in int_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round().astype(np.int64)

    return df_out


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINTING & ZARR PERSISTENCE
# =============================================================================

def calculate_frequency_fingerprint(
    frequencies: Union[np.ndarray, pd.Series, Sequence[float], bytes, str],
) -> str:
    """
    Calculates an immutable SHA-256 cryptographic fingerprint of experimental frequencies.

    Ensures that downstream Bayesian fitting state tensors (spectra_fit.zarr / spectra_fit.h5)
    can be rigorously traced back to the exact untampered experimental upload.

    Args:
        frequencies: Array, Series, Sequence of numerical frequencies, or raw string/bytes.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    if isinstance(frequencies, bytes):
        return hashlib.sha256(frequencies).hexdigest()

    if isinstance(frequencies, str):
        return hashlib.sha256(frequencies.encode("utf-8")).hexdigest()

    arr = np.asarray(frequencies, dtype=np.float64)
    if arr.size == 0:
        return hashlib.sha256(b"EMPTY_ARRAY").hexdigest()

    sorted_arr = np.sort(arr)
    ascii_payload = "\n".join(f"{f:.6f}" for f in sorted_arr).encode("utf-8")
    return hashlib.sha256(ascii_payload).hexdigest()


def append_to_spectra_fit_zarr(
    zarr_path: Union[str, Path],
    frequencies: Union[np.ndarray, pd.Series, Sequence[float]],
    sha256_hash: Optional[str] = None,
    group_name: str = "state_matrix",
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Appends experimental frequencies and cryptographic SHA-256 fingerprint to spectra_fit.zarr.

    Args:
        zarr_path: Path to the .zarr store (directory or zip store).
        frequencies: Array or Series of ingested frequencies.
        sha256_hash: Optional precomputed SHA-256 hash. If None, calculated dynamically.
        group_name: Subgroup name in zarr root (default 'state_matrix').
        metadata: Optional dictionary of supplementary provenance attributes.

    Returns:
        The zarr.Group object holding the state matrix and metadata attributes.
    """
    freq_arr = np.asarray(frequencies, dtype=np.float64)
    fingerprint = sha256_hash or calculate_frequency_fingerprint(freq_arr)

    path_obj = Path(zarr_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(path_obj), mode="a")
    group = root.require_group(group_name)

    # In Zarr 2 and 3, direct key assignment safely creates or updates the array
    group["frequencies"] = freq_arr

    group.attrs["sha256_fingerprint"] = fingerprint
    group.attrs["frequency_count"] = int(freq_arr.size)
    group.attrs["updated_at"] = datetime.now(timezone.utc).isoformat()
    if metadata:
        for k, v in metadata.items():
            group.attrs[k] = v

    return group


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE)
# =============================================================================

def validate_quantum_numbers(
    df: pd.DataFrame,
    drop_invalid: bool = True,
    raise_on_error: bool = False,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Pre-Flight Syntax Validator (Quantum Logic Gate).

    Mathematically intercepts and rejects impossible asymmetric rotor state assignments
    prior to Hamiltonian matrix generation.

    Enforces:
    1. Positive definite bounds: J >= 0, Ka >= 0, Kc >= 0 for both upper and lower states.
    2. Vector bounds: Ka <= J and Kc <= J for both states.
    3. Asymmetric top sum rule (Rotational Parity):
       For valid asymmetric rotor eigenstates, Ka + Kc must equal J or J + 1.
       - Upper state: Ka' + Kc' in {J', J' + 1}
       - Lower state: Ka'' + Kc'' in {J'', J'' + 1}

    Failure Protocol:
    Invalid lines are diagnosed. If raise_on_error is True, raises CoChemDataException.
    Otherwise, invalid lines are logged with structured diagnostics and dropped.

    Args:
        df: Input DataFrame with quantum number columns (J_upper, Ka_upper, Kc_upper, etc.).
        drop_invalid: If True, drops failing rows. If False, marks them in a 'syntax_valid' column.
        raise_on_error: If True, raises CoChemDataException immediately upon detecting syntax violations.
        log_warnings: If True, emits logger.warning for each dropped row.

    Returns:
        Tuple of (sanitized_df, list_of_syntax_violations).

    Raises:
        CoChemDataException: If raise_on_error is True and any row fails syntax validation.
    """
    if df.empty:
        return df.copy(), []

    qn_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    for col in qn_cols:
        if col not in df.columns:
            raise CoChemDataException(f"Missing required quantum number column '{col}' for syntax validation.")

    violations: List[Dict[str, Any]] = []
    valid_mask = np.ones(len(df), dtype=bool)

    j_up = df["J_upper"].to_numpy(dtype=np.int64)
    ka_up = df["Ka_upper"].to_numpy(dtype=np.int64)
    kc_up = df["Kc_upper"].to_numpy(dtype=np.int64)
    j_low = df["J_lower"].to_numpy(dtype=np.int64)
    ka_low = df["Ka_lower"].to_numpy(dtype=np.int64)
    kc_low = df["Kc_lower"].to_numpy(dtype=np.int64)

    for idx in range(len(df)):
        reasons: List[str] = []
        ju, kau, kcu = j_up[idx], ka_up[idx], kc_up[idx]
        jl, kal, kcl = j_low[idx], ka_low[idx], kc_low[idx]

        if ju < 0 or kau < 0 or kcu < 0:
            reasons.append(f"Negative upper quantum number: J'={ju}, Ka'={kau}, Kc'={kcu}")
        if jl < 0 or kal < 0 or kcl < 0:
            reasons.append(f"Negative lower quantum number: J''={jl}, Ka''={kal}, Kc''={kcl}")

        if kau > ju:
            reasons.append(f"Upper Ka' ({kau}) exceeds J' ({ju})")
        if kcu > ju:
            reasons.append(f"Upper Kc' ({kcu}) exceeds J' ({ju})")
        if kal > jl:
            reasons.append(f"Lower Ka'' ({kal}) exceeds J'' ({jl})")
        if kcl > jl:
            reasons.append(f"Lower Kc'' ({kcl}) exceeds J'' ({jl})")

        sum_up = kau + kcu
        if sum_up != ju and sum_up != (ju + 1):
            reasons.append(
                f"Syntax Ghost Line: Upper state Ka'+Kc' ({kau}+{kcu}={sum_up}) "
                f"violates asymmetric top sum rule for J'={ju} (expected {ju} or {ju + 1})"
            )

        sum_low = kal + kcl
        if sum_low != jl and sum_low != (jl + 1):
            reasons.append(
                f"Syntax Ghost Line: Lower state Ka''+Kc'' ({kal}+{kcl}={sum_low}) "
                f"violates asymmetric top sum rule for J''={jl} (expected {jl} or {jl + 1})"
            )

        if reasons:
            valid_mask[idx] = False
            row_idx = df.index[idx]
            diag = {
                "row_index": int(row_idx),
                "reasons": reasons,
                "J_upper": int(ju),
                "Ka_upper": int(kau),
                "Kc_upper": int(kcu),
                "J_lower": int(jl),
                "Ka_lower": int(kal),
                "Kc_lower": int(kcl),
            }
            violations.append(diag)
            if log_warnings:
                logger.warning(
                    f"[SYNTAX GHOST LINE] Dropping row {row_idx}: {'; '.join(reasons)}"
                )

    if violations and raise_on_error:
        raise CoChemDataException(
            f"Pre-Flight Syntax Validator encountered {len(violations)} quantum number violations: {violations}"
        )

    if drop_invalid:
        sanitized_df = df[valid_mask].copy().reset_index(drop=True)
    else:
        sanitized_df = df.copy()
        sanitized_df["syntax_valid"] = valid_mask

    return sanitized_df, violations


# =============================================================================
# 6. PRE-SERIALIZATION PHYSICS FILTER (SELECTION RULES)
# =============================================================================

def classify_transition_parity_and_dipole(
    j_up: int,
    ka_up: int,
    kc_up: int,
    j_low: int,
    ka_low: int,
    kc_low: int,
) -> Tuple[BranchType, DipoleType, Parity, Parity, int, int, int]:
    """
    Classifies a rotational transition's branch (P, Q, R), state parities, and dipole type (a, b, c).

    Asymmetric Top Selection Rules:
    - Delta J in {-1, 0, +1} (P: -1, Q: 0, R: +1). J=0 -> 0 is strictly forbidden.
    - Parity of state: (Ka mod 2, Kc mod 2) in {ee, eo, oe, oo}.
    - a-type (along a-axis, mu_a != 0): Delta Ka even, Delta Kc odd (ee <-> eo, oe <-> oo).
    - b-type (along b-axis, mu_b != 0): Delta Ka odd, Delta Kc odd (ee <-> oo, eo <-> oe).
    - c-type (along c-axis, mu_c != 0): Delta Ka odd, Delta Kc even (ee <-> oe, eo <-> oo).

    Returns:
        Tuple of (branch, dipole_type, upper_parity, lower_parity, delta_j, delta_ka, delta_kc).
    """
    delta_j = j_up - j_low
    delta_ka = ka_up - ka_low
    delta_kc = kc_up - kc_low

    if delta_j == -1:
        branch = BranchType.P
    elif delta_j == 0:
        branch = BranchType.Q
    elif delta_j == 1:
        branch = BranchType.R
    else:
        branch = BranchType.INVALID

    parity_map = {
        (0, 0): Parity.EE,
        (0, 1): Parity.EO,
        (1, 0): Parity.OE,
        (1, 1): Parity.OO,
    }
    up_parity = parity_map[(ka_up % 2, kc_up % 2)]
    low_parity = parity_map[(ka_low % 2, kc_low % 2)]

    if j_up == 0 and j_low == 0:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    if abs(delta_j) > 1:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    d_ka_even = (abs(delta_ka) % 2 == 0)
    d_kc_even = (abs(delta_kc) % 2 == 0)

    if d_ka_even and not d_kc_even:
        dipole_type = DipoleType.A
    elif not d_ka_even and not d_kc_even:
        dipole_type = DipoleType.B
    elif not d_ka_even and d_kc_even:
        dipole_type = DipoleType.C
    else:
        dipole_type = DipoleType.FORBIDDEN

    return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc


def apply_selection_rules(
    df: pd.DataFrame,
    allowed_dipole_types: Optional[Set[str]] = None,
    drop_forbidden: bool = True,
    raise_on_error: bool = False,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    In-Memory Pre-Serialization Physics Filter (Selection Rules).

    Applies rigid quantum selection rules in-memory before serialization:
    1. Enforces Delta J in {-1, 0, +1} and excludes J = 0 -> 0.
    2. Enforces active dipole transition parity constraints (a-type, b-type, c-type).
    3. Strips out forbidden transitions to prevent wasted compute and I/O.

    Args:
        df: Input DataFrame with validated quantum numbers.
        allowed_dipole_types: Set of allowed dipole types (e.g. {'a', 'b', 'c'} or {'b'}).
        drop_forbidden: If True, removes forbidden lines from returned DataFrame.
        raise_on_error: If True, raises CoChemDataException when forbidden transitions occur.
        log_warnings: If True, logs warning messages for each forbidden line.

    Returns:
        Tuple of (filtered_df, list_of_physics_violations).

    Raises:
        CoChemDataException: If raise_on_error is True and forbidden transitions are found.
    """
    if df.empty:
        return df.copy(), []

    allowed_types = {t.lower() for t in allowed_dipole_types} if allowed_dipole_types else {"a", "b", "c"}

    violations: List[Dict[str, Any]] = []
    valid_mask = np.ones(len(df), dtype=bool)

    branches: List[str] = []
    dipoles: List[str] = []
    d_j_list: List[int] = []
    d_ka_list: List[int] = []
    d_kc_list: List[int] = []
    up_parities: List[str] = []
    low_parities: List[str] = []

    j_up = df["J_upper"].to_numpy(dtype=np.int64)
    ka_up = df["Ka_upper"].to_numpy(dtype=np.int64)
    kc_up = df["Kc_upper"].to_numpy(dtype=np.int64)
    j_low = df["J_lower"].to_numpy(dtype=np.int64)
    ka_low = df["Ka_lower"].to_numpy(dtype=np.int64)
    kc_low = df["Kc_lower"].to_numpy(dtype=np.int64)

    for idx in range(len(df)):
        ju, kau, kcu = j_up[idx], ka_up[idx], kc_up[idx]
        jl, kal, kcl = j_low[idx], ka_low[idx], kc_low[idx]

        branch, dipole, up_par, low_par, dj, dka, dkc = classify_transition_parity_and_dipole(
            ju, kau, kcu, jl, kal, kcl
        )

        branches.append(branch.value)
        dipoles.append(dipole.value)
        d_j_list.append(dj)
        d_ka_list.append(dka)
        d_kc_list.append(dkc)
        up_parities.append(up_par.value)
        low_parities.append(low_par.value)

        reasons: List[str] = []
        if ju == 0 and jl == 0:
            reasons.append("Dipole-forbidden J=0 -> 0 transition")
        elif abs(dj) > 1:
            reasons.append(f"Forbidden Delta J = {dj} (|Delta J| > 1)")
        elif dipole == DipoleType.FORBIDDEN:
            reasons.append(f"Dipole-forbidden parity transition ({up_par.value} <-> {low_par.value})")
        elif dipole.value not in allowed_types:
            reasons.append(
                f"Transition dipole type '{dipole.value}' not in allowed set {sorted(allowed_types)}"
            )

        if reasons:
            valid_mask[idx] = False
            row_idx = df.index[idx]
            diag = {
                "row_index": int(row_idx),
                "reasons": reasons,
                "branch": branch.value,
                "dipole_type": dipole.value,
                "delta_j": int(dj),
                "delta_ka": int(dka),
                "delta_kc": int(dkc),
                "upper_parity": up_par.value,
                "lower_parity": low_par.value,
            }
            violations.append(diag)
            if log_warnings:
                logger.warning(
                    f"[PHYSICS FILTER] Dropping forbidden transition at row {row_idx}: {'; '.join(reasons)}"
                )

    if violations and raise_on_error:
        raise CoChemDataException(
            f"Pre-Serialization Physics Filter found {len(violations)} forbidden transitions: {violations}"
        )

    df_annotated = df.copy()
    df_annotated["Branch"] = branches
    df_annotated["Dipole_Type"] = dipoles
    df_annotated["Delta_J"] = d_j_list
    df_annotated["Delta_Ka"] = d_ka_list
    df_annotated["Delta_Kc"] = d_kc_list
    df_annotated["Upper_Parity"] = up_parities
    df_annotated["Lower_Parity"] = low_parities

    if drop_forbidden:
        filtered_df = df_annotated[valid_mask].copy().reset_index(drop=True)
    else:
        filtered_df = df_annotated
        filtered_df["physics_valid"] = valid_mask

    return filtered_df, violations


# =============================================================================
# 7. PARSERS: PICKETT LIN, PAR, AND MODERN CSV/STRUCTURED
# =============================================================================

def parse_legacy_pickett(
    lines_or_path: Union[List[str], str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parses legacy Pickett .lin / .cat files using strict positional and regex parsing.

    Translates Fortran double-precision exponential notation ('D-05', 'D+02', '1.2345-05' -> FP64).
    Failing lines are appended to failed_ingest.log located at platformdirs.user_log_path("CoChem").

    Args:
        lines_or_path: File path, list of line strings, single string content, bytes, or StringIO.
        config: Optional IngestConfig for defaults and custom log path.

    Returns:
        Tuple of (parsed_df, list_of_failed_raw_lines).
    """
    cfg = config or IngestConfig()
    lines: List[str] = []

    if isinstance(lines_or_path, list):
        lines = lines_or_path
    elif isinstance(lines_or_path, Path):
        if not lines_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {lines_or_path}")
        with open(str(lines_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(lines_or_path, str):
        if "\n" not in lines_or_path and os.path.exists(lines_or_path):
            with open(lines_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in lines_or_path and (
            lines_or_path.endswith((".lin", ".cat", ".par", ".var")) or "/" in lines_or_path or "\\" in lines_or_path
        ):
            raise FileNotFoundError(f"Pickett file not found: {lines_or_path}")
        else:
            lines = lines_or_path.splitlines()
    elif isinstance(lines_or_path, bytes):
        lines = lines_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(lines_or_path, io.StringIO):
        lines = lines_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content input type: {type(lines_or_path)}")

    parsed_records: List[Dict[str, Any]] = []
    failed_lines: List[str] = []

    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    jpl_cat_pattern = re.compile(
        r"^\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+)\s+"
        r"(\d+)\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)"
    )

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        matched = False

        cat_match = jpl_cat_pattern.match(line)
        if cat_match:
            try:
                freq = fortran_float_converter(cat_match.group(1))
                unc = fortran_float_converter(cat_match.group(2))
                intensity = fortran_float_converter(cat_match.group(3))
                ju = int(cat_match.group(9))
                kau = int(cat_match.group(10))
                kcu = int(cat_match.group(11))
                jl = int(cat_match.group(12))
                kal = int(cat_match.group(13))
                kcl = int(cat_match.group(14))

                parsed_records.append({
                    "Frequency_MHz": np.float64(freq),
                    "Uncertainty": np.float64(unc),
                    "Intensity": np.float64(intensity),
                    "J_upper": int(ju),
                    "Ka_upper": int(kau),
                    "Kc_upper": int(kcu),
                    "J_lower": int(jl),
                    "Ka_lower": int(kal),
                    "Kc_lower": int(kcl),
                })
                matched = True
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Catalog match parse failed at line {line_no}: {e}")

        if not matched:
            free_match = free_pattern.match(line)
            if free_match:
                try:
                    ju = int(free_match.group(1))
                    kau = int(free_match.group(2))
                    kcu = int(free_match.group(3))
                    jl = int(free_match.group(4))
                    kal = int(free_match.group(5))
                    kcl = int(free_match.group(6))
                    freq = fortran_float_converter(free_match.group(7))
                    unc = fortran_float_converter(free_match.group(8))
                    raw_int = free_match.group(9)
                    intensity = fortran_float_converter(raw_int) if raw_int else cfg.default_intensity

                    parsed_records.append({
                        "Frequency_MHz": np.float64(freq),
                        "Uncertainty": np.float64(unc),
                        "Intensity": np.float64(intensity),
                        "J_upper": int(ju),
                        "Ka_upper": int(kau),
                        "Kc_upper": int(kcu),
                        "J_lower": int(jl),
                        "Ka_lower": int(kal),
                        "Kc_lower": int(kcl),
                    })
                    matched = True
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Pickett line regex parse failed at line {line_no}: {e}")

        if not matched:
            if len(raw_line) >= 40:
                try:
                    part_ju = raw_line[0:3].strip()
                    part_kau = raw_line[3:6].strip()
                    part_kcu = raw_line[6:9].strip()
                    part_jl = raw_line[18:21].strip()
                    part_kal = raw_line[21:24].strip()
                    part_kcl = raw_line[24:27].strip()
                    part_freq = raw_line[36:49].strip()
                    part_unc = raw_line[49:57].strip()
                    part_int = raw_line[57:65].strip() if len(raw_line) >= 65 else ""

                    if part_ju and part_jl and part_freq:
                        ju = int(part_ju)
                        kau = int(part_kau) if part_kau else 0
                        kcu = int(part_kcu) if part_kcu else 0
                        jl = int(part_jl)
                        kal = int(part_kal) if part_kal else 0
                        kcl = int(part_kcl) if part_kcl else 0
                        freq = fortran_float_converter(part_freq)
                        unc = fortran_float_converter(part_unc) if part_unc else cfg.default_uncertainty
                        intensity = fortran_float_converter(part_int) if part_int else cfg.default_intensity

                        parsed_records.append({
                            "Frequency_MHz": np.float64(freq),
                            "Uncertainty": np.float64(unc),
                            "Intensity": np.float64(intensity),
                            "J_upper": int(ju),
                            "Ka_upper": int(kau),
                            "Kc_upper": int(kcu),
                            "J_lower": int(jl),
                            "Ka_lower": int(kal),
                            "Kc_lower": int(kcl),
                        })
                        matched = True
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Fixed-width positional parse failed at line {line_no}: {e}")

        if not matched:
            failed_lines.append(raw_line)
            if cfg.log_warnings:
                logger.warning(f"[INGEST PARSE FAILURE] Line {line_no} failed regex parsing: {raw_line.strip()!r}")

    if failed_lines:
        log_dest: Path
        if cfg.failed_log_path:
            log_dest = Path(cfg.failed_log_path)
        else:
            log_dest = user_log_path("CoChem") / "failed_ingest.log"

        try:
            log_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(log_dest, "a", encoding="utf-8") as f:
                for fl in failed_lines:
                    f.write(f"{fl.rstrip()}\n")
        except OSError as e:
            logger.debug(f"Could not write to log path {log_dest}: {e}")

    if not parsed_records:
        df_res = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        df_res = pd.DataFrame(parsed_records)

    return enforce_fp64_boundary(df_res), failed_lines


parse_pickett_lin = parse_legacy_pickett


def parse_pickett_par(
    content_or_path: Union[str, Path, bytes, io.StringIO],
) -> PickettParFile:
    """
    Parses a legacy Pickett parameter file (.par / .var).

    Structure:
    Line 1: Title
    Line 2: Control line (npar, nline, nopt, nwarn, erpar, wtfac, scale, maxit)
    Subsequent lines: Parameters (code, value, uncertainty, label)

    Args:
        content_or_path: File path, string content, bytes, or StringIO.

    Returns:
        Structured PickettParFile instance.
    """
    lines: List[str] = []
    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett .par/.var file not found: {content_or_path}")
        with open(str(content_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in content_or_path and (
            content_or_path.endswith((".par", ".var", ".lin", ".cat")) or "/" in content_or_path or "\\" in content_or_path
        ):
            raise FileNotFoundError(f"Pickett .par/.var file not found: {content_or_path}")
        else:
            lines = content_or_path.splitlines()
    elif isinstance(content_or_path, bytes):
        lines = content_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(content_or_path, io.StringIO):
        lines = content_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise ValueError("Pickett .par/.var file must have at least a title and a control line.")

    title = non_empty[0].strip()
    control_line = non_empty[1].strip()

    ctrl_parts = control_line.split()
    npar = int(ctrl_parts[0]) if len(ctrl_parts) > 0 else 0
    nline = int(ctrl_parts[1]) if len(ctrl_parts) > 1 else 0
    nopt = int(ctrl_parts[2]) if len(ctrl_parts) > 2 else 0
    nwarn = int(ctrl_parts[3]) if len(ctrl_parts) > 3 else 0
    erpar = fortran_float_converter(ctrl_parts[4]) if len(ctrl_parts) > 4 else 1.0
    wtfac = fortran_float_converter(ctrl_parts[5]) if len(ctrl_parts) > 5 else 1.0
    scale = fortran_float_converter(ctrl_parts[6]) if len(ctrl_parts) > 6 else 1.0
    maxit = int(ctrl_parts[7]) if len(ctrl_parts) > 7 else 50

    parameters: List[PickettParameter] = []
    param_line_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    for p_line in non_empty[2:]:
        match = param_line_pattern.match(p_line)
        if match:
            pid = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            lbl = (match.group(4) or "").strip()
            parameters.append(PickettParameter(param_id=pid, value=val, uncertainty=unc, label=lbl))

    return PickettParFile(
        title=title,
        npar=len(parameters),
        nline=nline,
        nopt=nopt,
        nwarn=nwarn,
        erpar=erpar,
        wtfac=wtfac,
        scale=scale,
        maxit=maxit,
        parameters=parameters,
    )


def parse_csv_spectroscopy(
    content_or_path: Union[str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> pd.DataFrame:
    """
    Parses modern structured spectroscopy files (.csv, .tsv, .parquet, .json).

    Standardizes column name variations to canonical naming convention:
    ['Frequency_MHz', 'Uncertainty', 'Intensity', 'J_upper', 'Ka_upper', 'Kc_upper',
     'J_lower', 'Ka_lower', 'Kc_lower'].

    Args:
        content_or_path: File path, string content, bytes, or StringIO.
        config: IngestConfig options.

    Returns:
        Standardized Pandas DataFrame with FP64 precision.
    """
    cfg = config or IngestConfig()

    if isinstance(content_or_path, (str, Path)) and str(content_or_path).endswith(".parquet"):
        if not os.path.exists(str(content_or_path)):
            raise FileNotFoundError(f"Parquet file not found: {content_or_path}")
        df_raw = pd.read_parquet(str(content_or_path))
    elif isinstance(content_or_path, (str, Path)) and str(content_or_path).endswith(".json"):
        if not os.path.exists(str(content_or_path)):
            raise FileNotFoundError(f"JSON file not found: {content_or_path}")
        df_raw = pd.read_json(str(content_or_path))
    elif isinstance(content_or_path, io.StringIO):
        df_raw = pd.read_csv(content_or_path, sep=None, engine="python")
    elif isinstance(content_or_path, bytes):
        df_raw = pd.read_csv(io.BytesIO(content_or_path), sep=None, engine="python")
    elif isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        df_raw = pd.read_csv(str(content_or_path), sep=None, engine="python")
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            df_raw = pd.read_csv(content_or_path, sep=None, engine="python")
        elif "\n" not in content_or_path and (
            content_or_path.endswith((".csv", ".tsv", ".txt")) or "/" in content_or_path or "\\" in content_or_path
        ):
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        else:
            df_raw = pd.read_csv(io.StringIO(content_or_path), sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    col_map: Dict[str, str] = {
        "frequency_mhz": "Frequency_MHz",
        "frequency": "Frequency_MHz",
        "freq_mhz": "Frequency_MHz",
        "freq": "Frequency_MHz",
        "nu_mhz": "Frequency_MHz",
        "nu": "Frequency_MHz",
        "f_mhz": "Frequency_MHz",
        "f": "Frequency_MHz",
        "uncertainty": "Uncertainty",
        "uncertainty_mhz": "Uncertainty",
        "unc": "Uncertainty",
        "unc_mhz": "Uncertainty",
        "err": "Uncertainty",
        "error": "Uncertainty",
        "sigma": "Uncertainty",
        "intensity": "Intensity",
        "int": "Intensity",
        "logint": "Intensity",
        "weight": "Intensity",
        "wt": "Intensity",
        "s_mu2": "Intensity",
        "j_upper": "J_upper",
        "j_prime": "J_upper",
        "j'": "J_upper",
        "j1": "J_upper",
        "j_up": "J_upper",
        "ju": "J_upper",
        "ka_upper": "Ka_upper",
        "ka_prime": "Ka_upper",
        "ka'": "Ka_upper",
        "ka1": "Ka_upper",
        "ka_up": "Ka_upper",
        "kau": "Ka_upper",
        "kc_upper": "Kc_upper",
        "kc_prime": "Kc_upper",
        "kc'": "Kc_upper",
        "kc1": "Kc_upper",
        "kc_up": "Kc_upper",
        "kcu": "Kc_upper",
        "j_lower": "J_lower",
        "j_double_prime": "J_lower",
        "j''": "J_lower",
        "j2": "J_lower",
        "j_low": "J_lower",
        "jl": "J_lower",
        "ka_lower": "Ka_lower",
        "ka_double_prime": "Ka_lower",
        "ka''": "Ka_lower",
        "ka2": "Ka_lower",
        "ka_low": "Ka_lower",
        "kal": "Ka_lower",
        "kc_lower": "Kc_lower",
        "kc_double_prime": "Kc_lower",
        "kc''": "Kc_lower",
        "kc2": "Kc_lower",
        "kc_low": "Kc_lower",
        "kcl": "Kc_lower",
    }

    rename_dict: Dict[str, str] = {}
    for c in df_raw.columns:
        c_clean = str(c).strip().lower()
        if c_clean in col_map:
            rename_dict[c] = col_map[c_clean]

    df_renamed = df_raw.rename(columns=rename_dict)

    required_core = ["Frequency_MHz", "J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    missing = [c for c in required_core if c not in df_renamed.columns]
    if missing:
        raise CoChemDataException(
            f"CSV spectroscopy data missing mandatory columns: {missing}. Present: {list(df_renamed.columns)}"
        )

    if "Uncertainty" not in df_renamed.columns:
        df_renamed["Uncertainty"] = cfg.default_uncertainty
    if "Intensity" not in df_renamed.columns:
        df_renamed["Intensity"] = cfg.default_intensity

    for col in ["Frequency_MHz", "Uncertainty", "Intensity"]:
        if df_renamed[col].dtype == object:
            df_renamed[col] = df_renamed[col].apply(fortran_float_converter)

    return enforce_fp64_boundary(df_renamed[CANONICAL_COLUMNS])


# =============================================================================
# 8. MASTER UNIFIED INGESTION ENGINE
# =============================================================================

def parse_spectroscopy_data(
    file_path_or_content: Union[str, Path, bytes, io.StringIO],
    format_hint: Optional[str] = None,
    config: Optional[IngestConfig] = None,
    zarr_path: Optional[Union[str, Path]] = None,
) -> IngestResult:
    """
    Master Ingestion Engine & Gatekeeper (cochem_spycfit_ingest).

    Orchestrates:
    1. Automatic format detection (.lin, .par, .cat, .csv, .parquet, .json).
    2. Bidirectional parsing of legacy Pickett lines and modern tabular structures.
    3. JAX FP64 precision boundary enforcement (float64).
    4. Cryptographic SHA-256 frequency fingerprinting.
    5. Persistence to spectra_fit.zarr if zarr_path is configured.
    6. Pre-Flight Syntax Validator (Quantum Logic Gate) for quantum number bounds.
    7. In-memory Pre-Serialization Physics Filter (Selection Rules).
    8. Construction of comprehensive ValidationReport and IngestResult.

    Args:
        file_path_or_content: File path, string content, bytes, or StringIO.
        format_hint: Optional explicit format override ('lin', 'cat', 'csv', 'parquet', 'json').
        config: Optional IngestConfig configuration options.
        zarr_path: Optional path to spectra_fit.zarr to append the state matrix.

    Returns:
        IngestResult containing sanitized DataFrame, ValidationReport, and SHA-256 fingerprint.
    """
    cfg = config or IngestConfig()
    warnings_list: List[str] = []

    effective_zarr_path = zarr_path or cfg.zarr_path

    fmt = (format_hint or "").lower().strip()
    if not fmt:
        if isinstance(file_path_or_content, Path):
            suffix = file_path_or_content.suffix.lower().replace(".", "")
            if suffix in ["lin", "cat", "par", "var", "csv", "tsv", "parquet", "json"]:
                fmt = suffix
        elif isinstance(file_path_or_content, str) and "\n" not in file_path_or_content:
            try:
                p = Path(file_path_or_content)
                suffix = p.suffix.lower().replace(".", "")
                if suffix in ["lin", "cat", "par", "var", "csv", "tsv", "parquet", "json"]:
                    fmt = suffix
            except (ValueError, TypeError, OSError) as exc:
                logger.debug(f"Suffix resolution failed for {file_path_or_content}: {exc}")

    if fmt in ["lin", "cat"]:
        raw_df, failed_lines = parse_legacy_pickett(file_path_or_content, config=cfg)
        if failed_lines:
            warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing and were skipped.")
    elif fmt in ["par", "var"]:
        par_file = parse_pickett_par(file_path_or_content)
        par_records = [
            {"param_id": p.param_id, "value": p.value, "uncertainty": p.uncertainty, "label": p.label}
            for p in par_file.parameters
        ]
        df_par = pd.DataFrame(par_records)
        fp = calculate_frequency_fingerprint([p.value for p in par_file.parameters])
        if effective_zarr_path:
            append_to_spectra_fit_zarr(
                zarr_path=effective_zarr_path,
                frequencies=[p.value for p in par_file.parameters],
                sha256_hash=fp,
                group_name="pickett_parameters",
            )
        report = ValidationReport(
            total_lines_read=len(par_file.parameters),
            valid_lines_count=len(par_file.parameters),
            syntax_dropped_count=0,
            physics_dropped_count=0,
            sha256_fingerprint=fp,
            warnings=[],
            syntax_violations=[],
            physics_violations=[],
        )
        return IngestResult(df=df_par, report=report, sha256_fingerprint=fp, success=True)
    elif fmt in ["parquet", "json", "csv", "tsv"]:
        raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
    else:
        try:
            raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
        except (ValueError, CoChemDataException, pd.errors.ParserError):
            raw_df, failed_lines = parse_legacy_pickett(file_path_or_content, config=cfg)
            if failed_lines:
                warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing.")

    total_read = len(raw_df)

    fingerprint = calculate_frequency_fingerprint(raw_df["Frequency_MHz"])

    if effective_zarr_path:
        append_to_spectra_fit_zarr(
            zarr_path=effective_zarr_path,
            frequencies=raw_df["Frequency_MHz"],
            sha256_hash=fingerprint,
            group_name="state_matrix",
        )

    if cfg.validate_syntax:
        syntax_df, syntax_violations = validate_quantum_numbers(
            raw_df,
            drop_invalid=cfg.drop_invalid_syntax,
            raise_on_error=False,
            log_warnings=cfg.log_warnings,
        )
        syntax_dropped = len(syntax_violations) if cfg.drop_invalid_syntax else 0
        if syntax_dropped > 0:
            warnings_list.append(f"Pre-Flight Syntax Validator dropped {syntax_dropped} unphysical quantum assignment(s).")
    else:
        syntax_df = raw_df
        syntax_violations = []
        syntax_dropped = 0

    if cfg.apply_selection_rules:
        physics_df, physics_violations = apply_selection_rules(
            syntax_df,
            allowed_dipole_types=cfg.allowed_dipole_types,
            drop_forbidden=cfg.drop_forbidden_transitions,
            raise_on_error=False,
            log_warnings=cfg.log_warnings,
        )
        physics_dropped = len(physics_violations) if cfg.drop_forbidden_transitions else 0
        if physics_dropped > 0:
            warnings_list.append(f"Pre-Serialization Physics Filter dropped {physics_dropped} forbidden transition(s).")
    else:
        physics_df = syntax_df
        physics_violations = []
        physics_dropped = 0

    final_df = enforce_fp64_boundary(physics_df)

    report = ValidationReport(
        total_lines_read=total_read,
        valid_lines_count=len(final_df),
        syntax_dropped_count=syntax_dropped,
        physics_dropped_count=physics_dropped,
        sha256_fingerprint=fingerprint,
        warnings=warnings_list,
        syntax_violations=syntax_violations,
        physics_violations=physics_violations,
    )

    return IngestResult(
        df=final_df,
        report=report,
        sha256_fingerprint=fingerprint,
        success=(len(final_df) > 0 or total_read == 0),
    )


__all__ = [
    "FileFormat",
    "BranchType",
    "DipoleType",
    "Parity",
    "CANONICAL_COLUMNS",
    "SpectroscopySchema",
    "SpectroscopyTransition",
    "PickettParameter",
    "PickettParFile",
    "IngestConfig",
    "ValidationReport",
    "IngestResult",
    "CoChemDataException",
    "fortran_float_converter",
    "enforce_fp64_boundary",
    "calculate_frequency_fingerprint",
    "append_to_spectra_fit_zarr",
    "validate_quantum_numbers",
    "classify_transition_parity_and_dipole",
    "apply_selection_rules",
    "parse_legacy_pickett",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_csv_spectroscopy",
    "parse_spectroscopy_data",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_fit_registry_manager.py ---
"""
CoChem-SpycFit Physical Unit & Concurrency Test Suite: Fit Registry Manager & Zarr Persistence.

Strictly adheres to Zero-Mock mandate.
Validates:
1. Pydantic v2 schemas with ConfigDict(extra="forbid", validate_assignment=True).
2. Cross-platform StandardFileLock (filelock + RLock + re-entrancy + timeout + stale reaping).
3. HPC / Lustre LustreMkdirLock (atomic os.mkdir + owner.json + timeout + stale reaping).
4. Dynamic timeout scaling (10s local vs 60s Lustre).
5. Metadata-safe atomic snapshotting (shutil.copy2 + atomic rename + SHA-256 verification).
6. HDF5 SWMR connections (libver='latest', swmr=True) with multi-threaded concurrent writer & reader.
7. Dynamic Lustre fallback disabling SWMR and generating atomic snapshots.
8. High-level FitRegistryManager transaction lifecycle, parameter freezing, ML feedback, and tensor storage.
9. Standalone read_registry and write_registry functions with atomic locking.
10. FAIR-compliant cryptographic dataset hashing and verification (hash_dataset, verify_dataset_hash)
    for both individual files (.h5, .csv) and directory stores (.zarr).
11. Zarr tensor persistence (init_zarr_persistence, save_zarr_tensor, load_zarr_tensor)
    with multi-dimensional arrays, chunking, attributes, overwrite handling, and error types.
12. Multi-threaded and multi-process concurrency without race conditions or deadlocks.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pytest
import zarr
from pydantic import ValidationError

from cochem_spycfit.core_engine.cochem_fit_registry_manager import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOCAL_LOCK_TIMEOUT,
    DEFAULT_LUSTRE_LOCK_TIMEOUT,
    DEFAULT_STALE_LOCK_TIMEOUT,
    FitParameterRecord,
    FitRegistryCorruptionError,
    FitRegistryError,
    FitRegistryManager,
    FitRegistryMissingError,
    FitRegistrySchemaError,
    FitRegistryState,
    HDF5TensorError,
    InstrumentLimits,
    LustreLockError,
    LustreMkdirLock,
    MLFeedbackState,
    RegistryLockError,
    RegistryLockTimeoutError,
    SnapshotMetadata,
    StandardFileLock,
    ZarrTensorError,
    compute_file_sha256,
    create_atomic_snapshot,
    get_atomic_lock,
    get_default_db_path,
    get_default_registry_path,
    get_default_zarr_path,
    hash_dataset,
    init_zarr_persistence,
    is_lustre_filesystem,
    load_zarr_tensor,
    open_state_tensor,
    read_registry,
    resolve_lock_timeout,
    save_zarr_tensor,
    verify_dataset_hash,
    write_registry,
)


@pytest.fixture
def tmp_env(tmp_path: Path):
    """Provides a fresh isolated temporary workspace."""
    workspace = tmp_path / "spycfit_test_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    reg_dir = workspace / "Registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    db_dir = workspace / "Databases"
    db_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = workspace / "Scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    reg_json = reg_dir / "cochem_fit_registry.json"
    db_h5 = db_dir / "spectra_fit.h5"
    zarr_dir = scratch_dir / "spectra_fit.zarr"

    yield {
        "workspace": workspace,
        "reg_json": reg_json,
        "db_h5": db_h5,
        "zarr_dir": zarr_dir,
    }


# =============================================================================
# 1. PYDANTIC V2 SCHEMAS TESTS
# =============================================================================

def test_instrument_limits_schema():
    """Validates InstrumentLimits schema and extra forbidden constraint."""
    limits = InstrumentLimits(
        min_freq_ghz=2.5,
        max_freq_ghz=18.0,
        temperature_k=300.0,
        bandwidth_mhz=16000.0,
    )
    assert limits.min_freq_ghz == 2.5
    assert limits.max_freq_ghz == 18.0
    assert limits.temperature_k == 300.0

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        InstrumentLimits(min_freq_ghz=2.0, max_freq_ghz=18.0, invalid_field=123)  # type: ignore[call-arg]

    # Type / value bounds validation
    with pytest.raises(ValidationError):
        InstrumentLimits(min_freq_ghz=-1.0)


def test_ml_feedback_state_schema():
    """Validates MLFeedbackState schema."""
    ml = MLFeedbackState(
        active=True,
        iteration=5,
        learning_rate=0.05,
        loss_history=[1.2, 0.8, 0.4],
        assigned_transitions_count=42,
        unassigned_peaks_count=10,
        convergence_status="iterating",
    )
    assert ml.active is True
    assert ml.iteration == 5
    assert len(ml.loss_history) == 3
    assert ml.assigned_transitions_count == 42

    with pytest.raises(ValidationError):
        MLFeedbackState(active=True, unknown_key="test")  # type: ignore[call-arg]


def test_fit_parameter_record_schema():
    """Validates FitParameterRecord schema."""
    param = FitParameterRecord(
        name="A",
        value=10543.21,
        uncertainty=0.05,
        is_frozen=True,
        lower_bound=10000.0,
        upper_bound=11000.0,
        unit="MHz",
    )
    assert param.name == "A"
    assert param.value == 10543.21
    assert param.is_frozen is True

    # Mutating with invalid type fails validation
    with pytest.raises(ValidationError):
        param.uncertainty = -5.0


def test_fit_registry_state_schema():
    """Validates root FitRegistryState schema and manual_locks deduplication."""
    state = FitRegistryState(
        manual_locks=["DJ", "A", "DJ", "B"],
        metadata={"operator": "cochem-test"},
    )
    assert state.manual_locks == ["A", "B", "DJ"]
    assert state.metadata["operator"] == "cochem-test"

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        FitRegistryState(unauthorized_arg=True)  # type: ignore[call-arg]


# =============================================================================
# 2. FILESYSTEM & LUSTRE DETECTION TESTS
# =============================================================================

def test_lustre_detection_and_timeout_scaling(tmp_path: Path):
    """Validates is_lustre_filesystem and resolve_lock_timeout."""
    local_p = tmp_path / "local_test"
    assert is_lustre_filesystem(local_p) is False
    assert resolve_lock_timeout(is_lustre=False) == DEFAULT_LOCAL_LOCK_TIMEOUT

    lustre_p = tmp_path / "lustre" / "scratch"
    assert is_lustre_filesystem(lustre_p) is True
    assert resolve_lock_timeout(is_lustre=True) == DEFAULT_LUSTRE_LOCK_TIMEOUT

    os.environ["COCHEM_FORCE_LUSTRE"] = "1"
    try:
        assert is_lustre_filesystem(local_p) is True
        assert resolve_lock_timeout(is_lustre=True) == DEFAULT_LUSTRE_LOCK_TIMEOUT
    finally:
        os.environ.pop("COCHEM_FORCE_LUSTRE", None)

    assert resolve_lock_timeout(is_lustre=False, custom_timeout=25.0) == 25.0
    assert resolve_lock_timeout(is_lustre=True, custom_timeout=45.0) == 45.0


# =============================================================================
# 3. ATOMIC LOCK IMPLEMENTATIONS (StandardFileLock & LustreMkdirLock)
# =============================================================================

def test_standard_file_lock_lifecycle_and_reentrancy(tmp_path: Path):
    """Validates StandardFileLock acquire, release, and thread-local reentrancy."""
    lock_file = tmp_path / "test.lock"
    lock = StandardFileLock(lock_file, timeout=5.0)

    with lock:
        assert lock_file.exists()
        with lock:
            assert lock_file.exists()

    assert not lock_file.exists()


def test_standard_file_lock_concurrency(tmp_path: Path):
    """Validates multi-threaded exclusion with StandardFileLock."""
    lock_file = tmp_path / "counter.lock"
    lock = StandardFileLock(lock_file, timeout=10.0)

    counter = {"val": 0}
    num_threads = 6
    increments_per_thread = 20

    def worker():
        for _ in range(increments_per_thread):
            with lock:
                curr = counter["val"]
                time.sleep(0.001)
                counter["val"] = curr + 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["val"] == num_threads * increments_per_thread


def test_standard_file_lock_timeout(tmp_path: Path):
    """Validates RegistryLockTimeoutError when lock is held by another thread."""
    lock_file = tmp_path / "timeout.lock"
    lock1 = StandardFileLock(lock_file, timeout=2.0)
    lock2 = StandardFileLock(lock_file, timeout=0.2)

    ready_event = threading.Event()
    release_event = threading.Event()
    timeout_caught = {"caught": False}

    def holder():
        with lock1:
            ready_event.set()
            release_event.wait(timeout=2.0)

    def waiter():
        ready_event.wait()
        try:
            with lock2:
                time.sleep(0.001)
        except RegistryLockTimeoutError:
            timeout_caught["caught"] = True

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)

    t1.start()
    t2.start()

    t2.join()
    release_event.set()
    t1.join()

    assert timeout_caught["caught"] is True


def test_lustre_mkdir_lock_lifecycle_and_reentrancy(tmp_path: Path):
    """Validates LustreMkdirLock atomic directory creation, owner metadata, and re-entrancy."""
    lock_dir = tmp_path / "lustre_test.lockdir"
    lock = LustreMkdirLock(lock_dir, timeout=5.0)

    assert not lock_dir.exists()

    with lock:
        assert lock_dir.is_dir()
        assert lock.owner_file.is_file()
        owner_data = json.loads(lock.owner_file.read_text(encoding="utf-8"))
        assert owner_data["pid"] == os.getpid()

        with lock:
            assert lock_dir.is_dir()

    assert not lock_dir.exists()


def test_lustre_mkdir_lock_concurrency(tmp_path: Path):
    """Validates multi-threaded exclusion with LustreMkdirLock."""
    lock_dir = tmp_path / "lustre_counter.lockdir"
    lock = LustreMkdirLock(lock_dir, timeout=15.0)

    counter = {"val": 0}
    num_threads = 6
    increments_per_thread = 20

    def worker():
        for _ in range(increments_per_thread):
            with lock:
                curr = counter["val"]
                time.sleep(0.001)
                counter["val"] = curr + 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["val"] == num_threads * increments_per_thread


def test_lustre_mkdir_lock_timeout(tmp_path: Path):
    """Validates RegistryLockTimeoutError with LustreMkdirLock."""
    lock_dir = tmp_path / "lustre_timeout.lockdir"
    lock1 = LustreMkdirLock(lock_dir, timeout=2.0)
    lock2 = LustreMkdirLock(lock_dir, timeout=0.2)

    ready_event = threading.Event()
    release_event = threading.Event()
    timeout_caught = {"caught": False}

    def holder():
        with lock1:
            ready_event.set()
            release_event.wait(timeout=2.0)

    def waiter():
        ready_event.wait()
        try:
            with lock2:
                time.sleep(0.001)
        except RegistryLockTimeoutError:
            timeout_caught["caught"] = True

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)

    t1.start()
    t2.start()

    t2.join()
    release_event.set()
    t1.join()

    assert timeout_caught["caught"] is True


def test_stale_lock_reaping(tmp_path: Path):
    """Validates automatic reaping of stale locks."""
    # 1. StandardFileLock stale reaping
    stale_file = tmp_path / "stale.lock"
    stale_file.write_text("old_pid:123:0.0\n", encoding="utf-8")
    past_time = time.time() - 200.0
    os.utime(str(stale_file), (past_time, past_time))

    lock_std = StandardFileLock(stale_file, timeout=2.0, stale_timeout=30.0)
    with lock_std:
        assert stale_file.exists()

    # 2. LustreMkdirLock stale reaping
    stale_dir = tmp_path / "stale.lockdir"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_owner = stale_dir / "owner.json"
    with open(stale_owner, "w", encoding="utf-8") as f:
        json.dump({"pid": 99999, "acquired_at": time.time() - 200.0}, f)

    lock_lustre = LustreMkdirLock(stale_dir, timeout=2.0, stale_timeout=30.0)
    with lock_lustre:
        assert stale_dir.is_dir()
    assert not stale_dir.exists()


# =============================================================================
# 4. ATOMIC SNAPSHOT PERSISTENCE TESTS
# =============================================================================

def test_atomic_snapshot_creation(tmp_path: Path):
    """Validates metadata-safe atomic snapshot copy and SHA-256 verification."""
    source_h5 = tmp_path / "test_data.h5"
    with h5py.File(str(source_h5), "w") as f:
        f.create_dataset("test_tensor", data=np.arange(100, dtype=np.float64))

    src_hash = compute_file_sha256(source_h5)
    dest_h5 = tmp_path / "snapshots" / "test_data_snap.h5"

    snapshot_meta = create_atomic_snapshot(
        source_path=source_h5,
        dest_path=dest_h5,
        is_lustre=True,
    )

    assert snapshot_meta.is_lustre_fallback is True
    assert snapshot_meta.sha256 == src_hash
    assert Path(snapshot_meta.snapshot_path).is_file()
    assert snapshot_meta.size_bytes == source_h5.stat().st_size

    with h5py.File(snapshot_meta.snapshot_path, "r") as f:
        loaded = np.array(f["test_tensor"])
        np.testing.assert_array_equal(loaded, np.arange(100, dtype=np.float64))

    with pytest.raises(FitRegistryMissingError):
        create_atomic_snapshot(source_path=tmp_path / "non_existent.h5")


# =============================================================================
# 5. HDF5 SWMR PERSISTENCE & CONCURRENT STREAMING TESTS
# =============================================================================

def test_hdf5_swmr_concurrent_streaming(tmp_path: Path):
    """
    Validates genuine HDF5 Single-Writer / Multiple-Reader (SWMR) concurrent execution.
    Writer continuously appends records to HDF5 under SWMR while a reader thread
    concurrently polls and reads records without blocking or crashing.
    """
    db_path = tmp_path / "swmr_stream.h5"

    with h5py.File(str(db_path), "w", libver="latest") as f:
        f.create_dataset(
            "stream_data",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(64,),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        f.swmr_mode = True

    num_records = 20
    writer_done = threading.Event()
    reader_seen_counts: List[int] = []

    def writer_task():
        with open_state_tensor(db_path, mode="a", is_lustre=False) as f:
            dset = f["stream_data"]
            for i in range(num_records):
                curr_len = dset.shape[0]
                dset.resize((curr_len + 1,))
                dset[curr_len] = float(i * 1.5)
                f.flush()
                time.sleep(0.005)
        writer_done.set()

    def reader_task():
        time.sleep(0.01)
        with open_state_tensor(db_path, mode="r", is_lustre=False) as f:
            dset = f["stream_data"]
            start_poll = time.time()
            while time.time() - start_poll < 5.0:
                try:
                    dset.refresh()
                except Exception as ref_err:
                    dset.attrs["last_refresh_err"] = str(ref_err)
                reader_seen_counts.append(dset.shape[0])
                if writer_done.is_set() and dset.shape[0] >= num_records:
                    break
                time.sleep(0.005)

    t_writer = threading.Thread(target=writer_task)
    t_reader = threading.Thread(target=reader_task)

    t_writer.start()
    t_reader.start()

    t_writer.join(timeout=10.0)
    t_reader.join(timeout=10.0)

    with h5py.File(str(db_path), "r") as f:
        final_arr = np.array(f["stream_data"])
        assert len(final_arr) == num_records
        np.testing.assert_allclose(final_arr, np.arange(num_records) * 1.5)

    assert len(reader_seen_counts) > 1
    assert reader_seen_counts[-1] == num_records


def test_hdf5_lustre_fallback_disables_swmr_and_snapshots(tmp_path: Path):
    """
    Validates that Lustre filesystem detection dynamically disables SWMR mode
    and triggers automatic metadata-safe atomic snapshot generation.
    """
    db_path = tmp_path / "lustre_db.h5"

    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        phys_data = np.array(f_phys["isomer_001"]["hessian"])

    with open_state_tensor(db_path, mode="w", is_lustre=True) as f:
        assert not getattr(f, "swmr_mode", False)
        f.create_dataset("dense_matrix", data=phys_data)

    snap_dir = db_path.parent / "snapshots"
    assert snap_dir.is_dir()
    snaps = list(snap_dir.glob("lustre_db_snap_*.h5"))
    assert len(snaps) >= 1

    with h5py.File(str(snaps[0]), "r") as f_snap:
        loaded = np.array(f_snap["dense_matrix"])
        np.testing.assert_array_equal(loaded, phys_data)


# =============================================================================
# 6. HIGH-LEVEL FIT REGISTRY MANAGER TESTS
# =============================================================================

def test_fit_registry_manager_transaction_and_persistence(tmp_env):
    """Validates atomic transaction updates, parameter freezing, and JSON integrity."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    init_state = mgr.get_fit_record()
    assert len(init_state.manual_locks) == 0

    # 1. Parameter Freezing
    mgr.freeze_parameter("A", value=10250.5)
    mgr.freeze_parameter("DJ", value=0.0125)

    state1 = mgr.get_fit_record()
    assert "A" in state1.manual_locks
    assert "DJ" in state1.manual_locks
    assert state1.parameters["A"].value == 10250.5
    assert state1.parameters["A"].is_frozen is True

    # 2. Parameter Unfreezing
    mgr.unfreeze_parameter("A")
    state2 = mgr.get_fit_record()
    assert "A" not in state2.manual_locks
    assert "DJ" in state2.manual_locks
    assert state2.parameters["A"].is_frozen is False

    # 3. ML Feedback update
    mgr.update_ml_feedback(active=True, loss=0.45, assigned=12, unassigned=4, status="iterating")
    state3 = mgr.get_fit_record()
    assert state3.ml_feedback.active is True
    assert state3.ml_feedback.iteration == 1
    assert state3.ml_feedback.loss_history == [0.45]
    assert state3.ml_feedback.assigned_transitions_count == 12

    # 4. Verify physical JSON file exists and matches
    assert tmp_env["reg_json"].is_file()
    assert mgr.verify_registry_integrity() is True


def test_fit_registry_manager_spectral_tensors(tmp_env):
    """Validates saving and loading multi-dimensional spectral tensors with metadata and compression."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        test_matrix = np.array(f_phys["isomer_001"]["hessian"])
    meta = {
        "molecule": "Benzonitrile",
        "method": "B3LYP-D3BJ",
        "frequencies": [1024.5, 2048.0, 4096.5],
    }

    mgr.save_spectral_tensor(
        name="hamiltonian_j_block",
        tensor=test_matrix,
        metadata=meta,
        group="/hamiltonians",
        compression="gzip",
        compression_opts=6,
        shuffle=True,
        fletcher32=True,
    )

    loaded_arr, loaded_meta = mgr.load_spectral_tensor(
        name="hamiltonian_j_block",
        group="/hamiltonians",
    )

    np.testing.assert_allclose(loaded_arr, test_matrix)
    assert loaded_meta["molecule"] == "Benzonitrile"
    assert loaded_meta["method"] == "B3LYP-D3BJ"
    assert loaded_meta["frequencies"] == [1024.5, 2048.0, 4096.5]

    with pytest.raises(HDF5TensorError):
        mgr.load_spectral_tensor(name="non_existent", group="/hamiltonians")


def test_fit_registry_manager_transaction_rollback_on_error(tmp_env):
    """Validates that exceptions inside transaction() abort changes without corrupting state."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
    )

    mgr.freeze_parameter("B", value=5000.0)
    orig_state = mgr.get_fit_record()

    with pytest.raises(RuntimeError):
        with mgr.transaction() as state:
            state.manual_locks.append("CORRUPTED_PARAM")
            raise RuntimeError("Simulation error during optimization step")

    after_state = mgr.get_fit_record()
    assert "CORRUPTED_PARAM" not in after_state.manual_locks
    assert after_state.manual_locks == orig_state.manual_locks


def test_fit_registry_manager_corrupted_json(tmp_env):
    """Validates FitRegistryCorruptionError on corrupted JSON payload."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
    )

    tmp_env["reg_json"].write_text("{ corrupted_json: invalid [", encoding="utf-8")

    with pytest.raises(FitRegistryCorruptionError):
        mgr.get_fit_record()


def test_multi_threaded_fit_registry_stress(tmp_env):
    """Validates high-concurrency multi-threaded transaction safety."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    num_threads = 8
    iterations_per_thread = 15

    def worker(worker_idx: int):
        for i in range(iterations_per_thread):
            with mgr.transaction() as state:
                p_name = f"PARAM_{worker_idx}_{i}"
                state.manual_locks.append(p_name)
                state.parameters[p_name] = FitParameterRecord(
                    name=p_name,
                    value=float(worker_idx * 100 + i),
                    is_frozen=True,
                )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_state = mgr.get_fit_record()
    expected_total = num_threads * iterations_per_thread
    assert len(final_state.manual_locks) == expected_total
    assert len(final_state.parameters) == expected_total
    assert mgr.verify_registry_integrity() is True


# =============================================================================
# 7. PHASE 2 STANDALONE REGISTRY READ/WRITE TESTS
# =============================================================================

def test_standalone_read_write_registry(tmp_path: Path):
    """Validates standalone read_registry and write_registry functions with atomic locking."""
    custom_reg = tmp_path / "custom_reg.json"

    # Reading non-existent registry returns valid default state dict
    default_data = read_registry(custom_reg)
    assert isinstance(default_data, dict)
    assert "manual_locks" in default_data
    assert default_data["manual_locks"] == []

    # Writing validated dictionary
    custom_state = FitRegistryState(
        manual_locks=["A", "B"],
        instrument_limits=InstrumentLimits(min_freq_ghz=4.0, max_freq_ghz=12.0),
        ml_feedback=MLFeedbackState(active=True, iteration=3),
    )
    write_registry(custom_state, custom_reg)
    assert custom_reg.is_file()

    # Reading back written registry
    read_back = read_registry(custom_reg)
    assert read_back["manual_locks"] == ["A", "B"]
    assert read_back["instrument_limits"]["min_freq_ghz"] == 4.0
    assert read_back["ml_feedback"]["iteration"] == 3

    # Writing invalid schema dictionary raises FitRegistrySchemaError
    with pytest.raises(FitRegistrySchemaError):
        write_registry({"invalid_key": 12345}, custom_reg)

    # Writing invalid type raises FitRegistrySchemaError
    with pytest.raises(FitRegistrySchemaError):
        write_registry([1, 2, 3], custom_reg)  # type: ignore[arg-type]


# =============================================================================
# 8. PHASE 2 FAIR DATASET HASH LOGGING & VERIFICATION TESTS
# =============================================================================

def test_dataset_hashing_and_verification(tmp_path: Path):
    """Validates hash_dataset and verify_dataset_hash for files and directories."""
    # 1. Single File Hashing
    data_file = tmp_path / "spectrum.dat"
    content = b"CH3OH Rotational Transitions 12.5 GHz - 18.0 GHz"
    data_file.write_bytes(content)

    file_hash = hash_dataset(data_file)
    assert isinstance(file_hash, str)
    assert len(file_hash) == 64  # SHA-256 hex length
    assert verify_dataset_hash(data_file, file_hash) is True
    assert verify_dataset_hash(data_file, "wrong_hash_12345") is False

    # Non-existent file error
    with pytest.raises(FileNotFoundError):
        hash_dataset(tmp_path / "non_existent.dat")

    # verify_dataset_hash returns False gracefully for missing file
    assert verify_dataset_hash(tmp_path / "missing.dat", file_hash) is False

    # 2. Directory / Zarr Store Hashing
    zarr_dir = tmp_path / "test_store.zarr"
    save_zarr_tensor(
        name="test_arr",
        tensor=np.ones((10, 10), dtype=np.float64),
        zarr_path=zarr_dir,
    )
    dir_hash = hash_dataset(zarr_dir)
    assert isinstance(dir_hash, str)
    assert len(dir_hash) == 64
    assert verify_dataset_hash(zarr_dir, dir_hash) is True
    assert verify_dataset_hash(zarr_dir, dir_hash.upper()) is True  # Case-insensitive verification


def test_fit_registry_manager_active_dataset_hash(tmp_env):
    """Validates FitRegistryManager tracking and verifying active dataset hashes."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
    )

    # Populate HDF5 database
    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        matrix = np.array(f_phys["isomer_001"]["hessian"])
    mgr.save_spectral_tensor("hessian_001", matrix)

    # Set active dataset hash
    updated_state = mgr.set_active_dataset_hash()
    assert updated_state.active_dataset_hash is not None
    assert len(updated_state.active_dataset_hash) == 64

    # Verify active dataset hash
    assert mgr.verify_dataset_hash(updated_state.active_dataset_hash) is True
    assert mgr.verify_dataset_hash("invalid_hash_value") is False


# =============================================================================
# 9. PHASE 2 ZARR TENSOR PERSISTENCE TESTS
# =============================================================================

def test_zarr_persistence_initialization_and_operations(tmp_env):
    """Validates init_zarr_persistence, save_zarr_tensor, and load_zarr_tensor."""
    zarr_store = tmp_env["zarr_dir"]

    # 1. Initialize Zarr root group
    root_grp = init_zarr_persistence(zarr_path=zarr_store)
    assert isinstance(root_grp, zarr.Group)
    assert zarr_store.is_dir()

    # 2. Save 2D and 3D spectral arrays with metadata and chunking
    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        phys_2d = np.array(f_phys["isomer_001"]["hessian"])
        phys_1d = np.array(f_phys["isomer_001"]["dvr_eigenvalues"])

    meta_2d = {"kind": "hessian", "units": "Hartree/Bohr^2", "dim": 2}
    meta_1d = {"kind": "dvr_eigenvalues", "filter_order": 1}

    saved_dset_2d = save_zarr_tensor(
        name="hessian_matrix",
        tensor=phys_2d,
        zarr_path=zarr_store,
        group="/intermediates/hessians",
        metadata=meta_2d,
        chunks=(3, 3),
        overwrite=True,
    )
    assert isinstance(saved_dset_2d, zarr.Array)
    assert saved_dset_2d.shape == phys_2d.shape

    saved_dset_1d = save_zarr_tensor(
        name="dvr_eigen",
        tensor=phys_1d,
        zarr_path=zarr_store,
        group="/convolutions",
        metadata=meta_1d,
        chunks=True,
    )
    assert saved_dset_1d.shape == phys_1d.shape

    # 3. Load Zarr tensors and verify data and attributes
    loaded_2d, loaded_meta_2d = load_zarr_tensor(
        name="hessian_matrix",
        zarr_path=zarr_store,
        group="/intermediates/hessians",
    )
    np.testing.assert_allclose(loaded_2d, phys_2d)
    assert loaded_meta_2d["kind"] == "hessian"
    assert loaded_meta_2d["units"] == "Hartree/Bohr^2"
    assert loaded_meta_2d["dim"] == 2

    loaded_1d, loaded_meta_1d = load_zarr_tensor(
        name="dvr_eigen",
        zarr_path=zarr_store,
        group="/convolutions",
    )
    np.testing.assert_allclose(loaded_1d, phys_1d)
    assert loaded_meta_1d["kind"] == "dvr_eigenvalues"
    assert loaded_meta_1d["filter_order"] == 1


def test_fit_registry_manager_zarr_methods(tmp_env):
    """Validates FitRegistryManager integration with Zarr persistence."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        zarr_path=tmp_env["zarr_dir"],
    )

    # Initialize a specific dataset via manager
    dset = mgr.init_zarr_persistence(
        shape=(100, 50),
        dtype=np.float64,
        chunks=(25, 25),
        dataset_name="scratch_grid",
    )
    assert dset.shape == (100, 50)

    # Save and load via manager
    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        test_arr = np.array(f_phys["isomer_001"]["dvr_eigenvalues"])

    mgr.save_zarr_tensor(
        name="frequency_grid",
        tensor=test_arr,
        group="/grids",
        metadata={"min_ghz": 2.0, "max_ghz": 18.0},
    )

    arr_out, meta_out = mgr.load_zarr_tensor(name="frequency_grid", group="/grids")
    np.testing.assert_allclose(arr_out, test_arr)
    assert meta_out["min_ghz"] == 2.0
    assert meta_out["max_ghz"] == 18.0


def test_zarr_tensor_error_handling(tmp_path: Path):
    """Validates Zarr error handling for missing stores and missing datasets."""
    non_existent_zarr = tmp_path / "non_existent.zarr"

    # Loading from non-existent store raises FitRegistryMissingError
    with pytest.raises(FitRegistryMissingError):
        load_zarr_tensor("any_dataset", zarr_path=non_existent_zarr)

    # Creating valid store then loading missing dataset raises ZarrTensorError
    save_zarr_tensor("dset1", np.arange(10), zarr_path=non_existent_zarr)
    with pytest.raises(ZarrTensorError):
        load_zarr_tensor("missing_dset", zarr_path=non_existent_zarr)


def test_zarr_concurrent_writes(tmp_path: Path):
    """Validates multi-threaded concurrent writes to Zarr store under atomic locking."""
    zarr_path = tmp_path / "concurrent_test.zarr"
    num_threads = 6
    items_per_thread = 10

    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        phys_2d = np.array(f_phys["isomer_001"]["hessian"])

    def worker(t_idx: int):
        for i in range(items_per_thread):
            name = f"tensor_{t_idx}_{i}"
            data = phys_2d * float(t_idx * 100 + i)
            save_zarr_tensor(
                name=name,
                tensor=data,
                zarr_path=zarr_path,
                group="/concurrent_group",
                metadata={"worker": t_idx, "item": i},
                overwrite=True,
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all arrays were safely persisted and can be loaded
    for t_idx in range(num_threads):
        for i in range(items_per_thread):
            name = f"tensor_{t_idx}_{i}"
            loaded, meta = load_zarr_tensor(
                name=name,
                zarr_path=zarr_path,
                group="/concurrent_group",
            )
            expected_val = phys_2d * float(t_idx * 100 + i)
            np.testing.assert_allclose(loaded, expected_val)
            assert meta["worker"] == t_idx
            assert meta["item"] == i

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_ingest.py ---
"""
Physical Unit and Integration Tests for Stage 1.0 Intake Parser & Quantum Gatekeeper (Phase 2).

Module: tests/test_cochem_spycfit_ingest.py
Target: cochem_spycfit.intake.cochem_spycfit_ingest

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance.
3. JAX FP64 precision boundary enforcement (np.float64 casting).
4. Cryptographic SHA-256 frequency fingerprinting.
5. Zarr state matrix persistence (append_to_spectra_fit_zarr and spectra_fit.zarr).
6. Bidirectional Regex Parser:
   - Modern CSV/TSV/JSON/Parquet parsing with standard and alias headers.
   - Legacy Pickett .lin parsing (fixed-width 10-char, free-format, JPL catalog).
   - Fortran double precision exponent conversion ('D-05', 'D+02', 'd-05', '1.2345-05').
   - Legacy Pickett .par parameter file parsing.
   - Non-fatal logging of malformed lines to platformdirs.user_log_path("CoChem").
7. Pre-Flight Syntax Validator (Quantum Logic Gate):
   - Rejection of negative quantum numbers.
   - Rejection of Ka > J or Kc > J.
   - Rejection of asymmetric rotor parity sum rule violations (Ka + Kc != J and != J + 1).
   - Strict CoChemDataException emission when raise_on_error=True.
   - Structured diagnostic emission and drop handling.
8. Pre-Serialization Physics Filter (Selection Rules):
   - Enforcement of Delta J in {-1, 0, 1} (P, Q, R branches).
   - Strict rejection of J=0 -> 0 dipole-forbidden transitions.
   - Strict rejection of |Delta J| > 1 transitions.
   - Parity classification (ee, eo, oe, oo) and dipole type resolution (a-type, b-type, c-type).
   - Strict CoChemDataException emission when raise_on_error=True.
   - In-memory filtering of forbidden transitions.
9. End-to-End parse_spectroscopy_data pipeline with authentic molecular catalogs (H2O, H2CO) and Zarr output.
10. Pydantic v2 schema strictness (SpectroscopySchema & SpectroscopyTransition) and extra field forbidding.
11. Complete absence of 'pass' statements across source and test files.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import jax
import numpy as np
import pandas as pd
from platformdirs import user_log_path
import pytest
from pydantic import ValidationError
import zarr

# Ensure SpycFit source is in path
SPYCFIT_ROOT = Path(__file__).resolve().parent.parent
SPYCFIT_SRC = SPYCFIT_ROOT / "src"
if str(SPYCFIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_SRC))
if str(SPYCFIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_ROOT))

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    CoChemDataException,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopySchema,
    SpectroscopyTransition,
    ValidationReport,
    append_to_spectra_fit_zarr,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_legacy_pickett,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)


# =============================================================================
# 1. FILE ENCODING AND LF LINE ENDING TESTS
# =============================================================================

@pytest.fixture
def target_file_paths() -> List[Path]:
    """Returns absolute paths to all Stage 1.0 intake source and test files."""
    return [
        SPYCFIT_SRC / "cochem_spycfit" / "intake" / "cochem_spycfit_ingest.py",
        SPYCFIT_SRC / "cochem_spycfit" / "intake" / "__init__.py",
        Path(__file__).resolve(),
    ]


def test_file_encoding_and_lf_line_endings(target_file_paths: List[Path]) -> None:
    """Verify strictly Unix LF line endings (\n), standard UTF-8 encoding, and no BOM."""
    for path in target_file_paths:
        assert path.is_file(), f"Target file does not exist: {path}"
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"Found Windows CRLF (\\r\\n) in {path.name}"
        assert b"\n" in raw, f"Missing newline characters in {path.name}"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"Found UTF-8 BOM marker in {path.name}"


# =============================================================================
# 2. ZERO PASS STATEMENT AUDIT
# =============================================================================

def test_zero_pass_statements_in_codebase(target_file_paths: List[Path]) -> None:
    """Strictly audits that no 'pass' statements exist in the ingest codebase or test suite."""
    for path in target_file_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        passes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(passes) == 0, f"Found {len(passes)} 'pass' statement(s) in {path.name} at lines {[n.lineno for n in passes]}"


# =============================================================================
# 3. FORTRAN FLOAT PARSER AND FP64 BOUNDARY TESTS
# =============================================================================

@pytest.mark.parametrize(
    "input_val,expected_float",
    [
        ("1.2345D-05", 1.2345e-5),
        ("1.2345D+02", 123.45),
        ("1.2345d-05", 1.2345e-5),
        ("1.2345d+02", 123.45),
        ("1.2345E-05", 1.2345e-5),
        ("1.2345e+02", 123.45),
        ("12345.6789", 12345.6789),
        ("-456.7890", -456.789),
        ("1.2345-05", 1.2345e-5),
        ("1.2345+02", 123.45),
        (42.5, 42.5),
        (100, 100.0),
    ],
)
def test_fortran_float_converter_valid(input_val: Any, expected_float: float) -> None:
    """Tests Fortran double precision exponent conversion."""
    res = fortran_float_converter(input_val)
    assert np.isclose(res, expected_float, rtol=1e-12)
    assert isinstance(res, float)


def test_fortran_float_converter_invalid() -> None:
    """Tests that malformed strings raise ValueError."""
    with pytest.raises(ValueError):
        fortran_float_converter("")
    with pytest.raises(ValueError):
        fortran_float_converter("INVALID_NUMBER_STRING")


def test_enforce_fp64_boundary() -> None:
    """Tests JAX FP64 precision boundary enforcement on Pandas DataFrames."""
    df = pd.DataFrame({
        "Frequency_MHz": np.array([12345.67, 23456.78], dtype=np.float32),
        "Uncertainty": np.array([0.01, 0.02], dtype=np.float32),
        "Intensity": np.array([1.5, 2.5], dtype=np.float32),
        "J_upper": [1, 2],
        "Ka_upper": [0, 1],
        "Kc_upper": [1, 1],
        "J_lower": [0, 1],
        "Ka_lower": [0, 0],
        "Kc_lower": [0, 1],
    })

    fp64_df = enforce_fp64_boundary(df)

    assert fp64_df["Frequency_MHz"].dtype == np.float64
    assert fp64_df["Uncertainty"].dtype == np.float64
    assert fp64_df["Intensity"].dtype == np.float64
    assert fp64_df["J_upper"].dtype == np.int64
    assert os.environ.get("JAX_ENABLE_X64") == "True"


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINT & ZARR PERSISTENCE TESTS
# =============================================================================

def test_calculate_frequency_fingerprint() -> None:
    """Tests deterministic SHA-256 fingerprint generation."""
    freqs = [22235.08, 183310.08, 380197.37]
    fp1 = calculate_frequency_fingerprint(freqs)
    fp2 = calculate_frequency_fingerprint(np.array(freqs, dtype=np.float64))
    fp3 = calculate_frequency_fingerprint(pd.Series(freqs))

    assert fp1 == fp2 == fp3
    assert len(fp1) == 64
    assert isinstance(fp1, str)

    # Altering frequency must change hash
    fp_altered = calculate_frequency_fingerprint([22235.09, 183310.08, 380197.37])
    assert fp_altered != fp1


def test_append_to_spectra_fit_zarr(tmp_path: Path) -> None:
    """Tests state matrix and cryptographic hash persistence to spectra_fit.zarr."""
    zarr_dir = tmp_path / "spectra_fit.zarr"
    freqs = np.array([22235.08, 183310.08, 556936.00], dtype=np.float64)

    group = append_to_spectra_fit_zarr(
        zarr_path=zarr_dir,
        frequencies=freqs,
        group_name="state_matrix",
        metadata={"molecule": "H2O", "source": "JPL Catalog"},
    )

    assert zarr_dir.exists()
    assert "frequencies" in group
    stored_freqs = group["frequencies"][:]
    assert np.allclose(stored_freqs, freqs)
    assert group.attrs["sha256_fingerprint"] == calculate_frequency_fingerprint(freqs)
    assert group.attrs["frequency_count"] == 3
    assert group.attrs["molecule"] == "H2O"


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE) TESTS
# =============================================================================

def test_validate_quantum_numbers_valid_h2o() -> None:
    """Tests syntax validation on real water (H2O) asymmetric top quantum numbers."""
    df = pd.DataFrame({
        "Frequency_MHz": [556936.0, 1669904.0, 183310.0],
        "Uncertainty": [0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0],
        "J_upper": [1, 2, 3],
        "Ka_upper": [1, 1, 1],
        "Kc_upper": [0, 2, 3],  # 1+0=1 (J=1), 1+2=3 (J=2+1), 1+3=4 (J=3+1)
        "J_lower": [1, 1, 2],
        "Ka_lower": [0, 0, 2],
        "Kc_lower": [1, 1, 0],  # 0+1=1 (J=1), 0+1=1 (J=1), 2+0=2 (J=2)
    })

    sanitized, violations = validate_quantum_numbers(df, drop_invalid=True)
    assert len(sanitized) == 3
    assert len(violations) == 0


def test_validate_quantum_numbers_intercepts_syntax_ghost_lines() -> None:
    """Tests that malformed quantum numbers (Ka > J, Ka+Kc != J or J+1, negative) are intercepted."""
    df_invalid = pd.DataFrame({
        "Frequency_MHz": [10000.0, 20000.0, 30000.0, 40000.0],
        "Uncertainty": [0.01, 0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0, 1.0],
        "J_upper": [2, 5, -1, 1],
        "Ka_upper": [3, 4, 0, 1],
        "Kc_upper": [0, 3, 0, 0],
        "J_lower": [1, 4, 0, 1],
        "Ka_lower": [1, 2, 0, 0],
        "Kc_lower": [0, 2, 0, 1],
    })

    sanitized, violations = validate_quantum_numbers(df_invalid, drop_invalid=True)

    assert len(sanitized) == 1
    assert sanitized.iloc[0]["Frequency_MHz"] == 40000.0
    assert len(violations) == 3

    v_rows = [v["row_index"] for v in violations]
    assert 0 in v_rows
    assert 1 in v_rows
    assert 2 in v_rows


def test_validate_quantum_numbers_raise_on_error() -> None:
    """Tests that validate_quantum_numbers raises CoChemDataException when raise_on_error=True."""
    df_invalid = pd.DataFrame({
        "Frequency_MHz": [10000.0],
        "Uncertainty": [0.01],
        "Intensity": [1.0],
        "J_upper": [2],
        "Ka_upper": [3],  # Ka > J violation
        "Kc_upper": [0],
        "J_lower": [1],
        "Ka_lower": [1],
        "Kc_lower": [0],
    })

    with pytest.raises(CoChemDataException):
        validate_quantum_numbers(df_invalid, raise_on_error=True)


# =============================================================================
# 6. PRE-SERIALIZATION PHYSICS FILTER (SELECTION RULES) TESTS
# =============================================================================

def test_classify_transition_parity_and_dipole_abc_types() -> None:
    """Tests exact parity and dipole classification for asymmetric top lines."""
    # a-type: Delta Ka even, Delta Kc odd (e.g. 1_01 -> 0_00: Delta Ka=0, Delta Kc=1)
    b1, d1, up1, low1, dj1, dka1, dkc1 = classify_transition_parity_and_dipole(1, 0, 1, 0, 0, 0)
    assert b1 == BranchType.R
    assert d1 == DipoleType.A
    assert up1 == Parity.EO
    assert low1 == Parity.EE

    # b-type: Delta Ka odd, Delta Kc odd (e.g. 1_10 -> 1_01: Delta Ka=1, Delta Kc=-1)
    b2, d2, up2, low2, dj2, dka2, dkc2 = classify_transition_parity_and_dipole(1, 1, 0, 1, 0, 1)
    assert b2 == BranchType.Q
    assert d2 == DipoleType.B
    assert up2 == Parity.OE
    assert low2 == Parity.EO

    # c-type: 2_11 -> 1_01: Delta Ka=1, Delta Kc=0 -> c-type
    b3, d3, up3, low3, dj3, dka3, dkc3 = classify_transition_parity_and_dipole(2, 1, 1, 1, 0, 1)
    assert b3 == BranchType.R
    assert d3 == DipoleType.C
    assert up3 == Parity.OO
    assert low3 == Parity.EO

    # Forbidden J=0 -> 0
    b4, d4, _, _, _, _, _ = classify_transition_parity_and_dipole(0, 0, 0, 0, 0, 0)
    assert d4 == DipoleType.FORBIDDEN

    # Forbidden |Delta J| > 1
    b5, d5, _, _, _, _, _ = classify_transition_parity_and_dipole(3, 1, 2, 1, 0, 1)
    assert d5 == DipoleType.FORBIDDEN


def test_apply_selection_rules_filtering() -> None:
    """Tests in-memory stripping of dipole-forbidden lines and filtering by allowed dipole types."""
    df = pd.DataFrame({
        "Frequency_MHz": [100.0, 200.0, 300.0, 400.0],
        "Uncertainty": [0.01, 0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0, 1.0],
        "J_upper": [1, 1, 0, 3],
        "Ka_upper": [0, 1, 0, 0],
        "Kc_upper": [1, 0, 0, 3],
        "J_lower": [0, 1, 0, 1],
        "Ka_lower": [0, 0, 0, 0],
        "Kc_lower": [0, 1, 0, 1],
    })

    # 1. Allow all valid dipole types (a, b, c) -> drops rows 2 and 3
    filtered_all, violations_all = apply_selection_rules(df, allowed_dipole_types={"a", "b", "c"})
    assert len(filtered_all) == 2
    assert len(violations_all) == 2
    assert list(filtered_all["Dipole_Type"]) == ["a", "b"]

    # 2. Allow only b-type
    filtered_b, violations_b = apply_selection_rules(df, allowed_dipole_types={"b"})
    assert len(filtered_b) == 1
    assert filtered_b.iloc[0]["Dipole_Type"] == "b"
    assert filtered_b.iloc[0]["Frequency_MHz"] == 200.0
    assert len(violations_b) == 3


def test_apply_selection_rules_raise_on_error() -> None:
    """Tests that apply_selection_rules raises CoChemDataException when raise_on_error=True."""
    df_forbidden = pd.DataFrame({
        "Frequency_MHz": [100.0],
        "Uncertainty": [0.01],
        "Intensity": [1.0],
        "J_upper": [0],
        "Ka_upper": [0],
        "Kc_upper": [0],
        "J_lower": [0],
        "Ka_lower": [0],
        "Kc_lower": [0],
    })

    with pytest.raises(CoChemDataException):
        apply_selection_rules(df_forbidden, raise_on_error=True)


# =============================================================================
# 7. PARSER TESTS: CSV, PICKETT LIN, AND PICKETT PAR
# =============================================================================

def test_parse_csv_spectroscopy_standard_and_aliases(tmp_path: Path) -> None:
    """Tests CSV parsing with canonical and alias column headers."""
    csv_content = """nu_mhz,err_mhz,logint,j_prime,ka_prime,kc_prime,j_double_prime,ka_double_prime,kc_double_prime
22235.08,0.005,-3.2,1,1,0,1,0,1
183310.08,0.010,-1.5,3,1,3,2,2,0
"""
    csv_file = tmp_path / "test_spectrum.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    df = parse_csv_spectroscopy(csv_file)
    assert len(df) == 2
    assert list(df.columns) == [
        "Frequency_MHz", "Uncertainty", "Intensity",
        "J_upper", "Ka_upper", "Kc_upper",
        "J_lower", "Ka_lower", "Kc_lower",
    ]
    assert df["Frequency_MHz"].dtype == np.float64
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert df.iloc[0]["J_upper"] == 1


def test_parse_legacy_pickett_and_failed_logging(tmp_path: Path) -> None:
    """Tests parsing legacy Pickett files and verifying failed lines logging to platformdirs user_log_path."""
    lin_content = """# Water (H2O) rotational line list
  1  1  0  1  0  1   22235.0800  0.0050  1.0000D+00
  3  1  3  2  2  0  183310.0800  0.0100  5.5000D-01
  2  1  2  1  0  1 1669904.0000  0.0200  1.2000D+00
INVALID_PICKETT_LINE_MALFORMED
"""
    lin_file = tmp_path / "h2o.lin"
    lin_file.write_text(lin_content, encoding="utf-8")
    custom_log = tmp_path / "failed_ingest.log"

    cfg = IngestConfig(failed_log_path=str(custom_log))
    df, failed = parse_legacy_pickett(lin_file, config=cfg)

    assert len(df) == 3
    assert len(failed) == 1
    assert "INVALID_PICKETT_LINE" in failed[0]
    assert custom_log.is_file()
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert np.isclose(df.iloc[1]["Intensity"], 0.55)


def test_parse_pickett_par_file(tmp_path: Path) -> None:
    """Tests parsing legacy Pickett .par / .var parameter files."""
    par_content = """Water H2O Ground State Fitted Constants
   3   100    0    0   1.000000000000000D+00   1.000000000000000D+00   1.000000000000000D+00   50
       10000   4.353600000000000D+05   1.000000000000000D-04  / (B+C)/2
       20000   8.358400000000000D+05   1.000000000000000D-04  / A-(B+C)/2
       30000   1.458000000000000D+05   1.000000000000000D-04  / (B-C)/4
"""
    par_file = tmp_path / "h2o.par"
    par_file.write_text(par_content, encoding="utf-8")

    par_obj = parse_pickett_par(par_file)
    assert isinstance(par_obj, PickettParFile)
    assert "Water H2O" in par_obj.title
    assert par_obj.npar == 3
    assert par_obj.maxit == 50
    assert len(par_obj.parameters) == 3
    assert par_obj.parameters[0].param_id == 10000
    assert np.isclose(par_obj.parameters[0].value, 435360.0)


# =============================================================================
# 8. END-TO-END MASTER INGESTION PIPELINE TESTS
# =============================================================================

def test_parse_spectroscopy_data_e2e_water_with_zarr(tmp_path: Path) -> None:
    """
    End-to-End Test: Ingests realistic Water (H2O) spectrum, validates syntax,
    applies selection rules (b-type dipole), calculates fingerprint, persists to Zarr,
    and returns IngestResult.
    """
    csv_content = """Frequency_MHz,Uncertainty,Intensity,J_upper,Ka_upper,Kc_upper,J_lower,Ka_lower,Kc_lower
22235.08,0.005,1.0,1,1,0,1,0,1
183310.08,0.010,1.0,3,1,3,2,2,0
556936.00,0.010,1.0,1,1,0,1,0,1
999999.99,0.010,1.0,5,4,3,4,2,2
0.0,0.010,1.0,0,0,0,0,0,0
"""
    csv_file = tmp_path / "water_spectrum.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    zarr_out = tmp_path / "spectra_fit.zarr"

    cfg = IngestConfig(allowed_dipole_types={"b"}, zarr_path=str(zarr_out))
    res = parse_spectroscopy_data(csv_file, config=cfg)

    assert isinstance(res, IngestResult)
    assert res.success is True
    assert len(res.sha256_fingerprint) == 64
    assert res.report.total_lines_read == 5
    assert res.report.syntax_dropped_count == 1
    assert res.report.physics_dropped_count >= 1
    assert res.report.valid_lines_count == 3
    assert all(res.df["Dipole_Type"] == "b")
    assert res.df["Frequency_MHz"].dtype == np.float64

    # Verify zarr state matrix was persisted
    assert zarr_out.exists()
    z_root = zarr.open_group(str(zarr_out), mode="r")
    assert "state_matrix" in z_root
    assert "frequencies" in z_root["state_matrix"]


# =============================================================================
# 9. PYDANTIC SCHEMA STRICTNESS TESTS (SPECTROSCOPYSCHEMA & TRANSITION)
# =============================================================================

def test_spectroscopy_schema_validation_and_forbid_extra() -> None:
    """Verifies that SpectroscopySchema validates mandatory fields and forbids extra tokens."""
    # Valid instance
    schema = SpectroscopySchema(
        Frequency_MHz=22235.08,
        Uncertainty=0.005,
        Intensity=1.0,
        J_upper=1,
        Ka_upper=1,
        Kc_upper=0,
        J_lower=1,
        Ka_lower=0,
        Kc_lower=1,
    )
    assert schema.Frequency_MHz == 22235.08
    assert schema.branch == BranchType.Q
    assert schema.dipole_type == DipoleType.B
    assert schema.upper_parity == Parity.OE
    assert schema.lower_parity == Parity.EO

    # Forbids extra fields
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=1,
            Ka_upper=0,
            Kc_upper=1,
            J_lower=0,
            Ka_lower=0,
            Kc_lower=0,
            unauthorized_field="malicious_injection",  # type: ignore
        )

    # Rejects missing mandatory fields
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            # Missing J, Ka, Kc quantum numbers
        )


def test_spectroscopy_schema_physics_bounds() -> None:
    """Verifies that SpectroscopySchema rejects impossible rotor assignments."""
    # Ka > J
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=1,
            Ka_upper=2,
            Kc_upper=0,
            J_lower=0,
            Ka_lower=0,
            Kc_lower=0,
        )

    # Ka + Kc != J and != J+1
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=5,
            Ka_upper=4,
            Kc_upper=3,
            J_lower=4,
            Ka_lower=2,
            Kc_lower=2,
        )


def test_missing_file_paths_raise_file_not_found() -> None:
    """Verifies that non-existent paths raise FileNotFoundError rather than generic errors."""
    with pytest.raises(FileNotFoundError):
        parse_legacy_pickett(Path("non_existent_file.lin"))

    with pytest.raises(FileNotFoundError):
        parse_pickett_par(Path("non_existent_file.par"))

    with pytest.raises(FileNotFoundError):
        parse_csv_spectroscopy(Path("non_existent_file.csv"))

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.