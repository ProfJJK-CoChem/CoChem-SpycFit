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
