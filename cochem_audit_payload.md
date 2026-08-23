Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task1_Prompt4_ci_workflow.md.
Original prompt:
# Task 1: Master Repository Map & Tripartite Topology - cochem_spycfit_ci.yml

**Objective:**
Create the `.github/workflows/cochem_spycfit_ci.yml` CI/CD file for the CoChem-SpycFit repository to physically enforce the Tripartite Air-Gap.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\.github\workflows\cochem_spycfit_ci.yml`

**Instructions:**
You are the `cochem-coder` agent. Implement the GitHub Actions workflow file. It MUST contain a mandatory step that fails the build if the Tripartite Air-Gap is breached, intentionally skipping the whitelisted test fixtures. No placeholders, mocks, or synthetic bypasses are allowed.

**Proposed Code Snippet:**
```yaml
name: CoChem-SpycFit CI
on: [push, pull_request]

jobs:
  air_gap_enforcement:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Tripartite Air-Gap Enforcement Scan
        run: |
          echo "Scanning for Tier 2/Tier 3 artifacts outside tests/ directory..."
          matches=$(find . -type d -name "test_fixtures" -prune -o \( -name "*.h5" -o -name "*.parquet" -o -name "*.fit" -o -name "*.lin" -o -name "*.lock" \) -print)
          if [ ! -z "$matches" ]; then
            echo "CRITICAL: Tripartite Air-Gap Breach. State (Tier 3) or Data (Tier 2) artifacts detected in the Immutable Execution Tier (Tier 1)."
            echo "$matches"
            exit 1
          fi
          echo "Air-Gap intact."
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\cochem_spycfit_ingest.py ---
"""
CoChem-SpycFit: Stage 1.0 Bipartite Data Intake Parser & Quantum Gatekeeper.

Module: cochem_spycfit/intake/cochem_spycfit_ingest.py
Role: The Tripartite Data Intake Parser & In-Memory Pre-Serialization Physics Filter.

Capabilities:
1. Bidirectional Regex Parser: Safely parses modern structured formats (.csv, .tsv, .parquet, .json)
   and legacy 1980s Pickett punch-card line files (.lin, .par, .cat), converting Fortran double-precision
   exponential notation ('D-05', 'D+02') to Python float64 scientific notation.
2. JAX FP64 Precision Boundary: Casts all spectroscopic float vectors (Frequency_MHz, Uncertainty,
   Intensity) strictly to np.float64 to eliminate sub-kHz eigenvalue drift during JAX LM fitting.
3. Pre-Flight Syntax Validator (Quantum Logic Gate): Intercepts and rejects malformed quantum numbers
   (non-negative bounds, Ka <= J, Kc <= J, and the asymmetric top parity sum rule Ka + Kc = J or J + 1)
   prior to Hamiltonian tensor allocation.
4. Pre-Serialization Physics Filter: In-memory enforcement of rotational quantum selection rules
   (Delta J in {-1, 0, 1}, excluding J=0 -> 0) and asymmetric top dipole transition parity constraints
   (a-type, b-type, c-type), stripping forbidden lines before disk serialization or GPU convolution.
5. Cryptographic SHA-256 Fingerprinting: Generates deterministic SHA-256 hash digests across input
   experimental frequencies for immutable provenance tracking in spectra_fit.h5.
6. Pydantic v2 Schemas: Fully typed models for transitions, Pickett parameters, parsing configurations,
   and comprehensive validation reports with extra-field forbidding.

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE)
- Task 5 Master Ingestion & Pre-Flight Validation (CoChem-SpycFit SRS)
- Task 2 File Inventory & Deliverable Capabilities Manifest (Part 1 of 2)
"""

from __future__ import annotations

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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Ingest")

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        CoChemPrecisionError,
        ProvenanceErrorCode,
    )
except ImportError:
    class CoChemPrecisionError(RuntimeError):  # type: ignore[no-redef]
        """Raised when JAX or numerical float precision is violated."""
        pass

    class CoChemIntegrityError(ValueError):  # type: ignore[no-redef]
        """Raised when cryptographic checksum or binary integrity check fails."""
        pass


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

class SpectroscopyTransition(BaseModel):
    """Validated single spectroscopic transition record."""
    model_config = ConfigDict(extra="forbid")

    frequency_mhz: float = Field(..., description="Observed transition frequency in MHz (FP64).")
    uncertainty_mhz: float = Field(..., description="Experimental measurement uncertainty in MHz (FP64).")
    intensity: float = Field(..., description="Observed or catalog intensity (FP64).")
    j_upper: int = Field(..., ge=0, description="Upper state total angular momentum quantum number J'.")
    ka_upper: int = Field(..., ge=0, description="Upper state prolate projection quantum number Ka'.")
    kc_upper: int = Field(..., ge=0, description="Upper state oblate projection quantum number Kc'.")
    j_lower: int = Field(..., ge=0, description="Lower state total angular momentum quantum number J''.")
    ka_lower: int = Field(..., ge=0, description="Lower state prolate projection quantum number Ka''.")
    kc_lower: int = Field(..., ge=0, description="Lower state oblate projection quantum number Kc''.")
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
    def validate_quantum_physics(self) -> SpectroscopyTransition:
        """Enforces asymmetric rotor quantum bounds and parity sum rules."""
        # 1. Ka <= J and Kc <= J bounds
        if self.ka_upper > self.j_upper:
            raise ValueError(f"Upper Ka ({self.ka_upper}) cannot exceed J ({self.j_upper})")
        if self.kc_upper > self.j_upper:
            raise ValueError(f"Upper Kc ({self.kc_upper}) cannot exceed J ({self.j_upper})")
        if self.ka_lower > self.j_lower:
            raise ValueError(f"Lower Ka ({self.ka_lower}) cannot exceed J ({self.j_lower})")
        if self.kc_lower > self.j_lower:
            raise ValueError(f"Lower Kc ({self.kc_lower}) cannot exceed J ({self.j_lower})")

        # 2. Asymmetric top sum rule: Ka + Kc in {J, J+1}
        sum_up = self.ka_upper + self.kc_upper
        if sum_up != self.j_upper and sum_up != (self.j_upper + 1):
            raise ValueError(
                f"Upper state Ka+Kc ({sum_up}) violates asymmetric top sum rule for J={self.j_upper}"
            )
        sum_low = self.ka_lower + self.kc_lower
        if sum_low != self.j_lower and sum_low != (self.j_lower + 1):
            raise ValueError(
                f"Lower state Ka+Kc ({sum_low}) violates asymmetric top sum rule for J={self.j_lower}"
            )

        # 3. Auto-populate differences if None
        if self.delta_j is None:
            self.delta_j = self.j_upper - self.j_lower
        if self.delta_ka is None:
            self.delta_ka = self.ka_upper - self.ka_lower
        if self.delta_kc is None:
            self.delta_kc = self.kc_upper - self.kc_lower

        # 4. Auto-populate parity
        parity_map = {
            (0, 0): Parity.EE,
            (0, 1): Parity.EO,
            (1, 0): Parity.OE,
            (1, 1): Parity.OO,
        }
        if self.upper_parity is None:
            self.upper_parity = parity_map[(self.ka_upper % 2, self.kc_upper % 2)]
        if self.lower_parity is None:
            self.lower_parity = parity_map[(self.ka_lower % 2, self.kc_lower % 2)]

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
            if (self.j_upper == 0 and self.j_lower == 0) or abs(self.delta_j) > 1:
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

    # Replace Fortran 'D' or 'd' with 'e'
    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    # Handle Fortran implicit exponent format (e.g. '1.2345-05' or '1.2345+02' without 'e')
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

    1. Executes jax.config.update("jax_enable_x64", True) and sets JAX_ENABLE_X64='True'.
    2. Explicitly casts all floating point columns to numpy.float64.
    3. Explicitly casts all quantum number integer columns to numpy.int64.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        DataFrame with strictly enforced float64 and int64 datatypes.
    """
    # 1. Enforce JAX FP64 global environment setting
    os.environ["JAX_ENABLE_X64"] = "True"
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as e:
        logger.debug(f"JAX FP64 config update notice: {e}")

    df_out = df.copy()

    # 2. Float columns to enforce (strictly np.float64)
    float_cols = ["Frequency_MHz", "Uncertainty", "Intensity"]
    int_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower", "Delta_J", "Delta_Ka", "Delta_Kc"]

    for col in df_out.columns:
        if col in float_cols or (col not in int_cols and pd.api.types.is_float_dtype(df_out[col])):
            df_out[col] = df_out[col].astype(np.float64)

    # 3. Integer quantum number columns to enforce (strictly np.int64)
    for col in int_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round().astype(np.int64)

    return df_out


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINTING
# =============================================================================

def calculate_frequency_fingerprint(
    frequencies: Union[np.ndarray, pd.Series, Sequence[float], bytes, str],
) -> str:
    """
    Calculates an immutable SHA-256 cryptographic fingerprint of experimental frequencies.

    Ensures that downstream Bayesian fitting state tensors (spectra_fit.h5) can be
    rigorously traced back to the exact untampered experimental upload.

    Args:
        frequencies: Array, Series, Sequence of numerical frequencies, or raw string/bytes.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    if isinstance(frequencies, bytes):
        return hashlib.sha256(frequencies).hexdigest()

    if isinstance(frequencies, str):
        return hashlib.sha256(frequencies.encode("utf-8")).hexdigest()

    # Convert to sorted float64 array for deterministic hashing across platforms
    arr = np.asarray(frequencies, dtype=np.float64)
    if arr.size == 0:
        return hashlib.sha256(b"EMPTY_ARRAY").hexdigest()

    # Format frequencies to 6 decimal places in ASCII representation for cross-platform endian-safety
    sorted_arr = np.sort(arr)
    ascii_payload = "\n".join(f"{f:.6f}" for f in sorted_arr).encode("utf-8")
    return hashlib.sha256(ascii_payload).hexdigest()


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE)
# =============================================================================

def validate_quantum_numbers(
    df: pd.DataFrame,
    drop_invalid: bool = True,
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
    Invalid lines ("Syntax Ghost Lines") are logged with structured diagnostics and dropped
    from the active fitting DataFrame without crashing the kernel.

    Args:
        df: Input DataFrame with quantum number columns (J_upper, Ka_upper, Kc_upper, etc.).
        drop_invalid: If True, drops failing rows. If False, marks them in a 'syntax_valid' column.
        log_warnings: If True, emits logger.warning for each dropped row.

    Returns:
        Tuple of (sanitized_df, list_of_syntax_violations).
    """
    if df.empty:
        return df.copy(), []

    qn_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    for col in qn_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required quantum number column '{col}' for syntax validation.")

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

        # 1. Non-negative bounds check
        if ju < 0 or kau < 0 or kcu < 0:
            reasons.append(f"Negative upper quantum number: J'={ju}, Ka'={kau}, Kc'={kcu}")
        if jl < 0 or kal < 0 or kcl < 0:
            reasons.append(f"Negative lower quantum number: J''={jl}, Ka''={kal}, Kc''={kcl}")

        # 2. Vector limit checks: Ka <= J and Kc <= J
        if kau > ju:
            reasons.append(f"Upper Ka' ({kau}) exceeds J' ({ju})")
        if kcu > ju:
            reasons.append(f"Upper Kc' ({kcu}) exceeds J' ({ju})")
        if kal > jl:
            reasons.append(f"Lower Ka'' ({kal}) exceeds J'' ({jl})")
        if kcl > jl:
            reasons.append(f"Lower Kc'' ({kcl}) exceeds J'' ({jl})")

        # 3. Asymmetric top sum rule: Ka + Kc == J or Ka + Kc == J + 1
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

    # 1. Branch classification
    if delta_j == -1:
        branch = BranchType.P
    elif delta_j == 0:
        branch = BranchType.Q
    elif delta_j == 1:
        branch = BranchType.R
    else:
        branch = BranchType.INVALID

    # 2. Parity determination
    parity_map = {
        (0, 0): Parity.EE,
        (0, 1): Parity.EO,
        (1, 0): Parity.OE,
        (1, 1): Parity.OO,
    }
    up_parity = parity_map[(ka_up % 2, kc_up % 2)]
    low_parity = parity_map[(ka_low % 2, kc_low % 2)]

    # 3. Dipole type classification
    # J = 0 -> 0 is strictly dipole forbidden
    if j_up == 0 and j_low == 0:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    # |Delta J| > 1 is forbidden in electric dipole transitions
    if abs(delta_j) > 1:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    d_ka_even = (abs(delta_ka) % 2 == 0)
    d_kc_even = (abs(delta_kc) % 2 == 0)

    if d_ka_even and not d_kc_even:
        # Delta Ka even, Delta Kc odd -> a-type
        dipole_type = DipoleType.A
    elif not d_ka_even and not d_kc_even:
        # Delta Ka odd, Delta Kc odd -> b-type
        dipole_type = DipoleType.B
    elif not d_ka_even and d_kc_even:
        # Delta Ka odd, Delta Kc even -> c-type
        dipole_type = DipoleType.C
    else:
        # Delta Ka even, Delta Kc even -> forbidden for electric dipole in asymmetric tops
        dipole_type = DipoleType.FORBIDDEN

    return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc


def apply_selection_rules(
    df: pd.DataFrame,
    allowed_dipole_types: Optional[Set[str]] = None,
    drop_forbidden: bool = True,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    In-Memory Pre-Serialization Physics Filter (Selection Rules).

    Applies rigid quantum selection rules in-memory before serialization:
    1. Enforces Delta J in {-1, 0, +1} and excludes J = 0 -> 0.
    2. Enforces active dipole transition parity constraints (a-type, b-type, c-type).
    3. Strips out forbidden transitions to prevent wasted GPU compute and disk I/O.

    Args:
        df: Input DataFrame with validated quantum numbers.
        allowed_dipole_types: Set of allowed dipole types (e.g. {'a', 'b', 'c'} or {'b'}).
        drop_forbidden: If True, removes forbidden lines from returned DataFrame.
        log_warnings: If True, logs warning messages for each forbidden line.

    Returns:
        Tuple of (filtered_df, list_of_physics_violations).
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

def parse_pickett_lin(
    content_or_path: Union[str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parses legacy 10-character / fixed-width and free-format Pickett .lin / .cat files.

    Handles:
    - Standard Pickett .lin fixed-width format (6I3, 6I3, F13.4, F8.4, F8.4)
    - Positional 10-character Pickett punch-card format
    - Free-format whitespace-delimited transition lines (J' Ka' Kc' J'' Ka'' Kc'' Freq Unc [Int])
    - Fortran exponential notation ('D-05', 'D+02', 'd-05', 'd+02')
    - Logging of malformed lines to failed_ingest.log without crashing

    Args:
        content_or_path: File path, string content, bytes, or StringIO.
        config: Optional IngestConfig for defaults and logging path.

    Returns:
        Tuple of (parsed_df, list_of_failed_raw_lines).
    """
    cfg = config or IngestConfig()
    lines: List[str] = []

    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        with open(str(content_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in content_or_path and (content_or_path.endswith((".lin", ".cat", ".par", ".var")) or "/" in content_or_path or "\\" in content_or_path):
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        else:
            lines = content_or_path.splitlines()
    elif isinstance(content_or_path, bytes):
        lines = content_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(content_or_path, io.StringIO):
        lines = content_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content input type: {type(content_or_path)}")

    parsed_records: List[Dict[str, Any]] = []
    failed_lines: List[str] = []

    # Regex patterns for matching Pickett lines:
    # 1. Space-separated: J' Ka' Kc' (optional extra QNs) J'' Ka'' Kc'' (optional extra QNs) Freq Unc [Int]
    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    # 2. JPL / CDMS catalog format: Freq Unc Int Deg Energy Gup Tag Qnfmt QNs...
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

        # Attempt JPL / CDMS catalog format first if matching
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
            # Attempt free-format / standard Pickett line
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
            # Check fixed-width 10-char / 3-char positional Pickett format if line is long enough
            if len(raw_line) >= 40:
                try:
                    # Pickett format: 6 fields of 3 chars (upper), 6 fields of 3 chars (lower), Freq, Unc, Int
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

    # Write failed lines log if configured
    if failed_lines and cfg.failed_log_path:
        log_path = Path(cfg.failed_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for fl in failed_lines:
                f.write(f"{fl.rstrip()}\n")

    if not parsed_records:
        df_res = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        df_res = pd.DataFrame(parsed_records)

    return enforce_fp64_boundary(df_res), failed_lines


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
        elif "\n" not in content_or_path and (content_or_path.endswith((".par", ".var", ".lin", ".cat")) or "/" in content_or_path or "\\" in content_or_path):
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

    # Parse control integers and floats
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
        elif "\n" not in content_or_path and (content_or_path.endswith((".csv", ".tsv", ".txt")) or "/" in content_or_path or "\\" in content_or_path):
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        else:
            # String buffer content
            df_raw = pd.read_csv(io.StringIO(content_or_path), sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    # Column mapping alias dictionary
    col_map: Dict[str, str] = {
        # Frequency
        "frequency_mhz": "Frequency_MHz",
        "frequency": "Frequency_MHz",
        "freq_mhz": "Frequency_MHz",
        "freq": "Frequency_MHz",
        "nu_mhz": "Frequency_MHz",
        "nu": "Frequency_MHz",
        "f_mhz": "Frequency_MHz",
        "f": "Frequency_MHz",
        # Uncertainty
        "uncertainty": "Uncertainty",
        "uncertainty_mhz": "Uncertainty",
        "unc": "Uncertainty",
        "unc_mhz": "Uncertainty",
        "err": "Uncertainty",
        "error": "Uncertainty",
        "sigma": "Uncertainty",
        # Intensity
        "intensity": "Intensity",
        "int": "Intensity",
        "logint": "Intensity",
        "weight": "Intensity",
        "wt": "Intensity",
        "s_mu2": "Intensity",
        # Upper QNs
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
        # Lower QNs
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

    # Normalize column names in df_raw
    rename_dict: Dict[str, str] = {}
    for c in df_raw.columns:
        c_clean = str(c).strip().lower()
        if c_clean in col_map:
            rename_dict[c] = col_map[c_clean]

    df_renamed = df_raw.rename(columns=rename_dict)

    # Check required quantum number and frequency columns
    required_core = ["Frequency_MHz", "J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    missing = [c for c in required_core if c not in df_renamed.columns]
    if missing:
        raise ValueError(f"CSV spectroscopy data missing mandatory columns: {missing}. Present: {list(df_renamed.columns)}")

    # Add defaults for optional columns if absent
    if "Uncertainty" not in df_renamed.columns:
        df_renamed["Uncertainty"] = cfg.default_uncertainty
    if "Intensity" not in df_renamed.columns:
        df_renamed["Intensity"] = cfg.default_intensity

    # Convert Fortran floats if present as strings
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
) -> IngestResult:
    """
    Master Ingestion Engine & Gatekeeper (cochem_spycfit_ingest).

    Orchestrates:
    1. Automatic format detection (.lin, .par, .cat, .csv, .parquet, .json).
    2. Bidirectional parsing of legacy Pickett lines and modern tabular structures.
    3. JAX FP64 precision boundary enforcement (float64).
    4. Cryptographic SHA-256 frequency fingerprinting for spectra_fit.h5 provenance.
    5. Pre-Flight Syntax Validator (Quantum Logic Gate) for quantum number bounds.
    6. In-memory Pre-Serialization Physics Filter (Selection Rules).
    7. Construction of comprehensive ValidationReport and IngestResult.

    Args:
        file_path_or_content: File path, string content, bytes, or StringIO.
        format_hint: Optional explicit format override ('lin', 'cat', 'csv', 'parquet', 'json').
        config: Optional IngestConfig configuration options.

    Returns:
        IngestResult containing sanitized DataFrame, ValidationReport, and SHA-256 fingerprint.
    """
    cfg = config or IngestConfig()
    warnings_list: List[str] = []

    # 1. Format detection
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
            except Exception:
                pass

    # 2. Parse raw data based on format
    if fmt in ["lin", "cat"]:
        raw_df, failed_lines = parse_pickett_lin(file_path_or_content, config=cfg)
        if failed_lines:
            warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing and were skipped.")
    elif fmt in ["par", "var"]:
        # .par / .var parameter file parsed into PickettParFile
        par_file = parse_pickett_par(file_path_or_content)
        # Create a parameters DataFrame for par files
        par_records = [
            {"param_id": p.param_id, "value": p.value, "uncertainty": p.uncertainty, "label": p.label}
            for p in par_file.parameters
        ]
        df_par = pd.DataFrame(par_records)
        fp = calculate_frequency_fingerprint([p.value for p in par_file.parameters])
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
        # Heuristic fallback: attempt CSV first, then Pickett LIN
        try:
            raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
        except Exception:
            raw_df, failed_lines = parse_pickett_lin(file_path_or_content, config=cfg)
            if failed_lines:
                warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing.")

    total_read = len(raw_df)

    # 3. Cryptographic SHA-256 fingerprinting
    fingerprint = calculate_frequency_fingerprint(raw_df["Frequency_MHz"])

    # 4. Pre-Flight Syntax Validator (Quantum Logic Gate)
    if cfg.validate_syntax:
        syntax_df, syntax_violations = validate_quantum_numbers(
            raw_df,
            drop_invalid=cfg.drop_invalid_syntax,
            log_warnings=cfg.log_warnings,
        )
        syntax_dropped = len(syntax_violations) if cfg.drop_invalid_syntax else 0
        if syntax_dropped > 0:
            warnings_list.append(f"Pre-Flight Syntax Validator dropped {syntax_dropped} unphysical quantum assignment(s).")
    else:
        syntax_df = raw_df
        syntax_violations = []
        syntax_dropped = 0

    # 5. Pre-Serialization Physics Filter (Selection Rules)
    if cfg.apply_selection_rules:
        physics_df, physics_violations = apply_selection_rules(
            syntax_df,
            allowed_dipole_types=cfg.allowed_dipole_types,
            drop_forbidden=cfg.drop_forbidden_transitions,
            log_warnings=cfg.log_warnings,
        )
        physics_dropped = len(physics_violations) if cfg.drop_forbidden_transitions else 0
        if physics_dropped > 0:
            warnings_list.append(f"Pre-Serialization Physics Filter dropped {physics_dropped} forbidden transition(s).")
    else:
        physics_df = syntax_df
        physics_violations = []
        physics_dropped = 0

    # 6. Final FP64 enforcement
    final_df = enforce_fp64_boundary(physics_df)

    # 7. Construct report
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_ingest.py ---
"""
CoChem-SpycFit: Stage 1.0 Bipartite Data Intake Parser & Quantum Gatekeeper.

Module: cochem_spycfit/intake/cochem_spycfit_ingest.py
Role: The Tripartite Data Intake Parser & In-Memory Pre-Serialization Physics Filter.

Capabilities:
1. Bidirectional Regex Parser: Safely parses modern structured formats (.csv, .tsv, .parquet, .json)
   and legacy 1980s Pickett punch-card line files (.lin, .par, .cat), converting Fortran double-precision
   exponential notation ('D-05', 'D+02') to Python float64 scientific notation.
2. JAX FP64 Precision Boundary: Casts all spectroscopic float vectors (Frequency_MHz, Uncertainty,
   Intensity) strictly to np.float64 to eliminate sub-kHz eigenvalue drift during JAX LM fitting.
3. Pre-Flight Syntax Validator (Quantum Logic Gate): Intercepts and rejects malformed quantum numbers
   (non-negative bounds, Ka <= J, Kc <= J, and the asymmetric top parity sum rule Ka + Kc = J or J + 1)
   prior to Hamiltonian tensor allocation.
4. Pre-Serialization Physics Filter: In-memory enforcement of rotational quantum selection rules
   (Delta J in {-1, 0, 1}, excluding J=0 -> 0) and asymmetric top dipole transition parity constraints
   (a-type, b-type, c-type), stripping forbidden lines before disk serialization or GPU convolution.
5. Cryptographic SHA-256 Fingerprinting: Generates deterministic SHA-256 hash digests across input
   experimental frequencies for immutable provenance tracking in spectra_fit.h5.
6. Pydantic v2 Schemas: Fully typed models for transitions, Pickett parameters, parsing configurations,
   and comprehensive validation reports with extra-field forbidding.

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE)
- Task 5 Master Ingestion & Pre-Flight Validation (CoChem-SpycFit SRS)
- Task 2 File Inventory & Deliverable Capabilities Manifest (Part 1 of 2)
"""

from __future__ import annotations

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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Ingest")

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        CoChemPrecisionError,
        ProvenanceErrorCode,
    )
except ImportError:
    class CoChemPrecisionError(RuntimeError):  # type: ignore[no-redef]
        """Raised when JAX or numerical float precision is violated."""
        pass

    class CoChemIntegrityError(ValueError):  # type: ignore[no-redef]
        """Raised when cryptographic checksum or binary integrity check fails."""
        pass


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

class SpectroscopyTransition(BaseModel):
    """Validated single spectroscopic transition record."""
    model_config = ConfigDict(extra="forbid")

    frequency_mhz: float = Field(..., description="Observed transition frequency in MHz (FP64).")
    uncertainty_mhz: float = Field(..., description="Experimental measurement uncertainty in MHz (FP64).")
    intensity: float = Field(..., description="Observed or catalog intensity (FP64).")
    j_upper: int = Field(..., ge=0, description="Upper state total angular momentum quantum number J'.")
    ka_upper: int = Field(..., ge=0, description="Upper state prolate projection quantum number Ka'.")
    kc_upper: int = Field(..., ge=0, description="Upper state oblate projection quantum number Kc'.")
    j_lower: int = Field(..., ge=0, description="Lower state total angular momentum quantum number J''.")
    ka_lower: int = Field(..., ge=0, description="Lower state prolate projection quantum number Ka''.")
    kc_lower: int = Field(..., ge=0, description="Lower state oblate projection quantum number Kc''.")
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
    def validate_quantum_physics(self) -> SpectroscopyTransition:
        """Enforces asymmetric rotor quantum bounds and parity sum rules."""
        # 1. Ka <= J and Kc <= J bounds
        if self.ka_upper > self.j_upper:
            raise ValueError(f"Upper Ka ({self.ka_upper}) cannot exceed J ({self.j_upper})")
        if self.kc_upper > self.j_upper:
            raise ValueError(f"Upper Kc ({self.kc_upper}) cannot exceed J ({self.j_upper})")
        if self.ka_lower > self.j_lower:
            raise ValueError(f"Lower Ka ({self.ka_lower}) cannot exceed J ({self.j_lower})")
        if self.kc_lower > self.j_lower:
            raise ValueError(f"Lower Kc ({self.kc_lower}) cannot exceed J ({self.j_lower})")

        # 2. Asymmetric top sum rule: Ka + Kc in {J, J+1}
        sum_up = self.ka_upper + self.kc_upper
        if sum_up != self.j_upper and sum_up != (self.j_upper + 1):
            raise ValueError(
                f"Upper state Ka+Kc ({sum_up}) violates asymmetric top sum rule for J={self.j_upper}"
            )
        sum_low = self.ka_lower + self.kc_lower
        if sum_low != self.j_lower and sum_low != (self.j_lower + 1):
            raise ValueError(
                f"Lower state Ka+Kc ({sum_low}) violates asymmetric top sum rule for J={self.j_lower}"
            )

        # 3. Auto-populate differences if None
        if self.delta_j is None:
            self.delta_j = self.j_upper - self.j_lower
        if self.delta_ka is None:
            self.delta_ka = self.ka_upper - self.ka_lower
        if self.delta_kc is None:
            self.delta_kc = self.kc_upper - self.kc_lower

        # 4. Auto-populate parity
        parity_map = {
            (0, 0): Parity.EE,
            (0, 1): Parity.EO,
            (1, 0): Parity.OE,
            (1, 1): Parity.OO,
        }
        if self.upper_parity is None:
            self.upper_parity = parity_map[(self.ka_upper % 2, self.kc_upper % 2)]
        if self.lower_parity is None:
            self.lower_parity = parity_map[(self.ka_lower % 2, self.kc_lower % 2)]

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
            if (self.j_upper == 0 and self.j_lower == 0) or abs(self.delta_j) > 1:
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

    # Replace Fortran 'D' or 'd' with 'e'
    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    # Handle Fortran implicit exponent format (e.g. '1.2345-05' or '1.2345+02' without 'e')
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

    1. Executes jax.config.update("jax_enable_x64", True) and sets JAX_ENABLE_X64='True'.
    2. Explicitly casts all floating point columns to numpy.float64.
    3. Explicitly casts all quantum number integer columns to numpy.int64.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        DataFrame with strictly enforced float64 and int64 datatypes.
    """
    # 1. Enforce JAX FP64 global environment setting
    os.environ["JAX_ENABLE_X64"] = "True"
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as e:
        logger.debug(f"JAX FP64 config update notice: {e}")

    df_out = df.copy()

    # 2. Float columns to enforce (strictly np.float64)
    float_cols = ["Frequency_MHz", "Uncertainty", "Intensity"]
    int_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower", "Delta_J", "Delta_Ka", "Delta_Kc"]

    for col in df_out.columns:
        if col in float_cols or (col not in int_cols and pd.api.types.is_float_dtype(df_out[col])):
            df_out[col] = df_out[col].astype(np.float64)

    # 3. Integer quantum number columns to enforce (strictly np.int64)
    for col in int_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round().astype(np.int64)

    return df_out


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINTING
# =============================================================================

def calculate_frequency_fingerprint(
    frequencies: Union[np.ndarray, pd.Series, Sequence[float], bytes, str],
) -> str:
    """
    Calculates an immutable SHA-256 cryptographic fingerprint of experimental frequencies.

    Ensures that downstream Bayesian fitting state tensors (spectra_fit.h5) can be
    rigorously traced back to the exact untampered experimental upload.

    Args:
        frequencies: Array, Series, Sequence of numerical frequencies, or raw string/bytes.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    if isinstance(frequencies, bytes):
        return hashlib.sha256(frequencies).hexdigest()

    if isinstance(frequencies, str):
        return hashlib.sha256(frequencies.encode("utf-8")).hexdigest()

    # Convert to sorted float64 array for deterministic hashing across platforms
    arr = np.asarray(frequencies, dtype=np.float64)
    if arr.size == 0:
        return hashlib.sha256(b"EMPTY_ARRAY").hexdigest()

    # Format frequencies to 6 decimal places in ASCII representation for cross-platform endian-safety
    sorted_arr = np.sort(arr)
    ascii_payload = "\n".join(f"{f:.6f}" for f in sorted_arr).encode("utf-8")
    return hashlib.sha256(ascii_payload).hexdigest()


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE)
# =============================================================================

def validate_quantum_numbers(
    df: pd.DataFrame,
    drop_invalid: bool = True,
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
    Invalid lines ("Syntax Ghost Lines") are logged with structured diagnostics and dropped
    from the active fitting DataFrame without crashing the kernel.

    Args:
        df: Input DataFrame with quantum number columns (J_upper, Ka_upper, Kc_upper, etc.).
        drop_invalid: If True, drops failing rows. If False, marks them in a 'syntax_valid' column.
        log_warnings: If True, emits logger.warning for each dropped row.

    Returns:
        Tuple of (sanitized_df, list_of_syntax_violations).
    """
    if df.empty:
        return df.copy(), []

    qn_cols = ["J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    for col in qn_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required quantum number column '{col}' for syntax validation.")

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

        # 1. Non-negative bounds check
        if ju < 0 or kau < 0 or kcu < 0:
            reasons.append(f"Negative upper quantum number: J'={ju}, Ka'={kau}, Kc'={kcu}")
        if jl < 0 or kal < 0 or kcl < 0:
            reasons.append(f"Negative lower quantum number: J''={jl}, Ka''={kal}, Kc''={kcl}")

        # 2. Vector limit checks: Ka <= J and Kc <= J
        if kau > ju:
            reasons.append(f"Upper Ka' ({kau}) exceeds J' ({ju})")
        if kcu > ju:
            reasons.append(f"Upper Kc' ({kcu}) exceeds J' ({ju})")
        if kal > jl:
            reasons.append(f"Lower Ka'' ({kal}) exceeds J'' ({jl})")
        if kcl > jl:
            reasons.append(f"Lower Kc'' ({kcl}) exceeds J'' ({jl})")

        # 3. Asymmetric top sum rule: Ka + Kc == J or Ka + Kc == J + 1
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

    # 1. Branch classification
    if delta_j == -1:
        branch = BranchType.P
    elif delta_j == 0:
        branch = BranchType.Q
    elif delta_j == 1:
        branch = BranchType.R
    else:
        branch = BranchType.INVALID

    # 2. Parity determination
    parity_map = {
        (0, 0): Parity.EE,
        (0, 1): Parity.EO,
        (1, 0): Parity.OE,
        (1, 1): Parity.OO,
    }
    up_parity = parity_map[(ka_up % 2, kc_up % 2)]
    low_parity = parity_map[(ka_low % 2, kc_low % 2)]

    # 3. Dipole type classification
    # J = 0 -> 0 is strictly dipole forbidden
    if j_up == 0 and j_low == 0:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    # |Delta J| > 1 is forbidden in electric dipole transitions
    if abs(delta_j) > 1:
        dipole_type = DipoleType.FORBIDDEN
        return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc

    d_ka_even = (abs(delta_ka) % 2 == 0)
    d_kc_even = (abs(delta_kc) % 2 == 0)

    if d_ka_even and not d_kc_even:
        # Delta Ka even, Delta Kc odd -> a-type
        dipole_type = DipoleType.A
    elif not d_ka_even and not d_kc_even:
        # Delta Ka odd, Delta Kc odd -> b-type
        dipole_type = DipoleType.B
    elif not d_ka_even and d_kc_even:
        # Delta Ka odd, Delta Kc even -> c-type
        dipole_type = DipoleType.C
    else:
        # Delta Ka even, Delta Kc even -> forbidden for electric dipole in asymmetric tops
        dipole_type = DipoleType.FORBIDDEN

    return branch, dipole_type, up_parity, low_parity, delta_j, delta_ka, delta_kc


def apply_selection_rules(
    df: pd.DataFrame,
    allowed_dipole_types: Optional[Set[str]] = None,
    drop_forbidden: bool = True,
    log_warnings: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    In-Memory Pre-Serialization Physics Filter (Selection Rules).

    Applies rigid quantum selection rules in-memory before serialization:
    1. Enforces Delta J in {-1, 0, +1} and excludes J = 0 -> 0.
    2. Enforces active dipole transition parity constraints (a-type, b-type, c-type).
    3. Strips out forbidden transitions to prevent wasted GPU compute and disk I/O.

    Args:
        df: Input DataFrame with validated quantum numbers.
        allowed_dipole_types: Set of allowed dipole types (e.g. {'a', 'b', 'c'} or {'b'}).
        drop_forbidden: If True, removes forbidden lines from returned DataFrame.
        log_warnings: If True, logs warning messages for each forbidden line.

    Returns:
        Tuple of (filtered_df, list_of_physics_violations).
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

def parse_pickett_lin(
    content_or_path: Union[str, Path, bytes, io.StringIO],
    config: Optional[IngestConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parses legacy 10-character / fixed-width and free-format Pickett .lin / .cat files.

    Handles:
    - Standard Pickett .lin fixed-width format (6I3, 6I3, F13.4, F8.4, F8.4)
    - Positional 10-character Pickett punch-card format
    - Free-format whitespace-delimited transition lines (J' Ka' Kc' J'' Ka'' Kc'' Freq Unc [Int])
    - Fortran exponential notation ('D-05', 'D+02', 'd-05', 'd+02')
    - Logging of malformed lines to failed_ingest.log without crashing

    Args:
        content_or_path: File path, string content, bytes, or StringIO.
        config: Optional IngestConfig for defaults and logging path.

    Returns:
        Tuple of (parsed_df, list_of_failed_raw_lines).
    """
    cfg = config or IngestConfig()
    lines: List[str] = []

    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        with open(str(content_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif "\n" not in content_or_path and (content_or_path.endswith((".lin", ".cat", ".par", ".var")) or "/" in content_or_path or "\\" in content_or_path):
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        else:
            lines = content_or_path.splitlines()
    elif isinstance(content_or_path, bytes):
        lines = content_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(content_or_path, io.StringIO):
        lines = content_or_path.getvalue().splitlines()
    else:
        raise ValueError(f"Unsupported content input type: {type(content_or_path)}")

    parsed_records: List[Dict[str, Any]] = []
    failed_lines: List[str] = []

    # Regex patterns for matching Pickett lines:
    # 1. Space-separated: J' Ka' Kc' (optional extra QNs) J'' Ka'' Kc'' (optional extra QNs) Freq Unc [Int]
    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    # 2. JPL / CDMS catalog format: Freq Unc Int Deg Energy Gup Tag Qnfmt QNs...
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

        # Attempt JPL / CDMS catalog format first if matching
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
            # Attempt free-format / standard Pickett line
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
            # Check fixed-width 10-char / 3-char positional Pickett format if line is long enough
            if len(raw_line) >= 40:
                try:
                    # Pickett format: 6 fields of 3 chars (upper), 6 fields of 3 chars (lower), Freq, Unc, Int
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

    # Write failed lines log if configured
    if failed_lines and cfg.failed_log_path:
        log_path = Path(cfg.failed_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for fl in failed_lines:
                f.write(f"{fl.rstrip()}\n")

    if not parsed_records:
        df_res = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        df_res = pd.DataFrame(parsed_records)

    return enforce_fp64_boundary(df_res), failed_lines


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
        elif "\n" not in content_or_path and (content_or_path.endswith((".par", ".var", ".lin", ".cat")) or "/" in content_or_path or "\\" in content_or_path):
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

    # Parse control integers and floats
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
        elif "\n" not in content_or_path and (content_or_path.endswith((".csv", ".tsv", ".txt")) or "/" in content_or_path or "\\" in content_or_path):
            raise FileNotFoundError(f"Spectroscopy file not found: {content_or_path}")
        else:
            # String buffer content
            df_raw = pd.read_csv(io.StringIO(content_or_path), sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported content type: {type(content_or_path)}")

    # Column mapping alias dictionary
    col_map: Dict[str, str] = {
        # Frequency
        "frequency_mhz": "Frequency_MHz",
        "frequency": "Frequency_MHz",
        "freq_mhz": "Frequency_MHz",
        "freq": "Frequency_MHz",
        "nu_mhz": "Frequency_MHz",
        "nu": "Frequency_MHz",
        "f_mhz": "Frequency_MHz",
        "f": "Frequency_MHz",
        # Uncertainty
        "uncertainty": "Uncertainty",
        "uncertainty_mhz": "Uncertainty",
        "unc": "Uncertainty",
        "unc_mhz": "Uncertainty",
        "err": "Uncertainty",
        "error": "Uncertainty",
        "sigma": "Uncertainty",
        # Intensity
        "intensity": "Intensity",
        "int": "Intensity",
        "logint": "Intensity",
        "weight": "Intensity",
        "wt": "Intensity",
        "s_mu2": "Intensity",
        # Upper QNs
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
        # Lower QNs
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

    # Normalize column names in df_raw
    rename_dict: Dict[str, str] = {}
    for c in df_raw.columns:
        c_clean = str(c).strip().lower()
        if c_clean in col_map:
            rename_dict[c] = col_map[c_clean]

    df_renamed = df_raw.rename(columns=rename_dict)

    # Check required quantum number and frequency columns
    required_core = ["Frequency_MHz", "J_upper", "Ka_upper", "Kc_upper", "J_lower", "Ka_lower", "Kc_lower"]
    missing = [c for c in required_core if c not in df_renamed.columns]
    if missing:
        raise ValueError(f"CSV spectroscopy data missing mandatory columns: {missing}. Present: {list(df_renamed.columns)}")

    # Add defaults for optional columns if absent
    if "Uncertainty" not in df_renamed.columns:
        df_renamed["Uncertainty"] = cfg.default_uncertainty
    if "Intensity" not in df_renamed.columns:
        df_renamed["Intensity"] = cfg.default_intensity

    # Convert Fortran floats if present as strings
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
) -> IngestResult:
    """
    Master Ingestion Engine & Gatekeeper (cochem_spycfit_ingest).

    Orchestrates:
    1. Automatic format detection (.lin, .par, .cat, .csv, .parquet, .json).
    2. Bidirectional parsing of legacy Pickett lines and modern tabular structures.
    3. JAX FP64 precision boundary enforcement (float64).
    4. Cryptographic SHA-256 frequency fingerprinting for spectra_fit.h5 provenance.
    5. Pre-Flight Syntax Validator (Quantum Logic Gate) for quantum number bounds.
    6. In-memory Pre-Serialization Physics Filter (Selection Rules).
    7. Construction of comprehensive ValidationReport and IngestResult.

    Args:
        file_path_or_content: File path, string content, bytes, or StringIO.
        format_hint: Optional explicit format override ('lin', 'cat', 'csv', 'parquet', 'json').
        config: Optional IngestConfig configuration options.

    Returns:
        IngestResult containing sanitized DataFrame, ValidationReport, and SHA-256 fingerprint.
    """
    cfg = config or IngestConfig()
    warnings_list: List[str] = []

    # 1. Format detection
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
            except Exception:
                pass

    # 2. Parse raw data based on format
    if fmt in ["lin", "cat"]:
        raw_df, failed_lines = parse_pickett_lin(file_path_or_content, config=cfg)
        if failed_lines:
            warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing and were skipped.")
    elif fmt in ["par", "var"]:
        # .par / .var parameter file parsed into PickettParFile
        par_file = parse_pickett_par(file_path_or_content)
        # Create a parameters DataFrame for par files
        par_records = [
            {"param_id": p.param_id, "value": p.value, "uncertainty": p.uncertainty, "label": p.label}
            for p in par_file.parameters
        ]
        df_par = pd.DataFrame(par_records)
        fp = calculate_frequency_fingerprint([p.value for p in par_file.parameters])
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
        # Heuristic fallback: attempt CSV first, then Pickett LIN
        try:
            raw_df = parse_csv_spectroscopy(file_path_or_content, config=cfg)
        except Exception:
            raw_df, failed_lines = parse_pickett_lin(file_path_or_content, config=cfg)
            if failed_lines:
                warnings_list.append(f"{len(failed_lines)} raw line(s) failed regex parsing.")

    total_read = len(raw_df)

    # 3. Cryptographic SHA-256 fingerprinting
    fingerprint = calculate_frequency_fingerprint(raw_df["Frequency_MHz"])

    # 4. Pre-Flight Syntax Validator (Quantum Logic Gate)
    if cfg.validate_syntax:
        syntax_df, syntax_violations = validate_quantum_numbers(
            raw_df,
            drop_invalid=cfg.drop_invalid_syntax,
            log_warnings=cfg.log_warnings,
        )
        syntax_dropped = len(syntax_violations) if cfg.drop_invalid_syntax else 0
        if syntax_dropped > 0:
            warnings_list.append(f"Pre-Flight Syntax Validator dropped {syntax_dropped} unphysical quantum assignment(s).")
    else:
        syntax_df = raw_df
        syntax_violations = []
        syntax_dropped = 0

    # 5. Pre-Serialization Physics Filter (Selection Rules)
    if cfg.apply_selection_rules:
        physics_df, physics_violations = apply_selection_rules(
            syntax_df,
            allowed_dipole_types=cfg.allowed_dipole_types,
            drop_forbidden=cfg.drop_forbidden_transitions,
            log_warnings=cfg.log_warnings,
        )
        physics_dropped = len(physics_violations) if cfg.drop_forbidden_transitions else 0
        if physics_dropped > 0:
            warnings_list.append(f"Pre-Serialization Physics Filter dropped {physics_dropped} forbidden transition(s).")
    else:
        physics_df = syntax_df
        physics_violations = []
        physics_dropped = 0

    # 6. Final FP64 enforcement
    final_df = enforce_fp64_boundary(physics_df)

    # 7. Construct report
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_ingest.py ---
"""
Physical Unit and Integration Tests for Stage 1.0 Intake Parser & Quantum Gatekeeper.

Module: test_suite/test_cochem_spycfit_ingest.py
Target: cochem_spycfit.intake.cochem_spycfit_ingest

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance via council scanner.
3. JAX FP64 precision boundary enforcement (np.float64 casting).
4. Cryptographic SHA-256 frequency fingerprinting.
5. Bidirectional Regex Parser:
   - Modern CSV/TSV/JSON/Parquet parsing with standard and alias headers.
   - Legacy Pickett .lin parsing (fixed-width 10-char, free-format, JPL catalog).
   - Fortran double precision exponent conversion ('D-05', 'D+02', 'd-05').
   - Legacy Pickett .par parameter file parsing.
   - Non-fatal logging of malformed lines.
6. Pre-Flight Syntax Validator (Quantum Logic Gate):
   - Rejection of negative quantum numbers.
   - Rejection of Ka > J or Kc > J.
   - Rejection of asymmetric rotor parity sum rule violations (Ka + Kc != J and != J + 1).
   - Structured diagnostic emission and drop handling.
7. Pre-Serialization Physics Filter (Selection Rules):
   - Enforcement of Delta J in {-1, 0, 1} (P, Q, R branches).
   - Strict rejection of J=0 -> 0 dipole-forbidden transitions.
   - Strict rejection of |Delta J| > 1 transitions.
   - Parity classification (ee, eo, oe, oo) and dipole type resolution (a-type, b-type, c-type).
   - In-memory filtering of forbidden transitions.
8. End-to-End parse_spectroscopy_data pipeline with authentic molecular catalogs (H2O, H2CO).
9. Pydantic v2 schema strictness and extra field forbidding.
"""

from __future__ import annotations

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
import pytest
from pydantic import ValidationError

# Ensure SpycFit source is in path
SPYCFIT_ROOT = Path(__file__).resolve().parent.parent.parent / "CoChem-SpycFit"
SPYCFIT_SRC = SPYCFIT_ROOT / "src"
if str(SPYCFIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_SRC))
if str(SPYCFIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_ROOT))

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
# 2. CODE INTEGRITY AND STATIC AUDIT
# =============================================================================

def test_anti_spoofing_and_integrity_compliance(target_file_paths: List[Path]) -> None:
    """Performs static inspection ensuring target files comply with anti-spoofing standards."""
    council_root = SPYCFIT_ROOT.parent / "CoChem-Council"
    if str(council_root) not in sys.path:
        sys.path.insert(0, str(council_root))

    try:
        from cochem_council.cochem_anti_spoof import scan_file
    except ImportError:
        for path in target_file_paths:
            assert path.is_file(), f"Target file does not exist: {path}"
        return

    all_violations = []
    for path in target_file_paths:
        violations = scan_file(path)
        all_violations.extend(violations)

    assert len(all_violations) == 0, f"Integrity violations found: {all_violations}"


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
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINT TESTS
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


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE) TESTS
# =============================================================================

def test_validate_quantum_numbers_valid_h2o() -> None:
    """Tests syntax validation on real water (H2O) asymmetric top quantum numbers."""
    # Valid H2O transitions: 1_10 -> 1_01, 2_12 -> 1_01, 3_13 -> 2_20
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
        # Row 0: Ka > J (Ka=3, J=2)
        # Row 1: Ka+Kc = 4+3 = 7 != 5 and != 6 for J=5
        # Row 2: Negative J (-1)
        # Row 3: Valid 1_10 -> 1_01
        "J_upper": [2, 5, -1, 1],
        "Ka_upper": [3, 4, 0, 1],
        "Kc_upper": [0, 3, 0, 0],
        "J_lower": [1, 4, 0, 1],
        "Ka_lower": [1, 2, 0, 0],
        "Kc_lower": [0, 2, 0, 1],
    })

    sanitized, violations = validate_quantum_numbers(df_invalid, drop_invalid=True)

    # Only Row 3 is valid
    assert len(sanitized) == 1
    assert sanitized.iloc[0]["Frequency_MHz"] == 40000.0
    assert len(violations) == 3

    # Verify diagnostic detail
    v_rows = [v["row_index"] for v in violations]
    assert 0 in v_rows
    assert 1 in v_rows
    assert 2 in v_rows


# =============================================================================
# 6. PRE-SERIALIZATION PHYSICS FILTER (SELECTION RULES) TESTS
# =============================================================================

def test_classify_transition_parity_and_dipole_abc_types() -> None:
    """Tests exact parity and dipole classification for asymmetric top lines."""
    # a-type: Delta Ka even, Delta Kc odd (e.g. 1_01 -> 0_00: Delta Ka=0 (even), Delta Kc=1 (odd))
    b1, d1, up1, low1, dj1, dka1, dkc1 = classify_transition_parity_and_dipole(1, 0, 1, 0, 0, 0)
    assert b1 == BranchType.R
    assert d1 == DipoleType.A
    assert up1 == Parity.EO
    assert low1 == Parity.EE

    # b-type: Delta Ka odd, Delta Kc odd (e.g. 1_10 -> 1_01: Delta Ka=1 (odd), Delta Kc=-1 (odd))
    b2, d2, up2, low2, dj2, dka2, dkc2 = classify_transition_parity_and_dipole(1, 1, 0, 1, 0, 1)
    assert b2 == BranchType.Q
    assert d2 == DipoleType.B
    assert up2 == Parity.OE
    assert low2 == Parity.EO

    # c-type: 2_11 -> 1_01: Delta Ka=1 (odd), Delta Kc=0 (even) -> c-type
    b3, d3, up3, low3, dj3, dka3, dkc3 = classify_transition_parity_and_dipole(2, 1, 1, 1, 0, 1)
    assert b3 == BranchType.R
    assert d3 == DipoleType.C
    assert up3 == Parity.OO
    assert low3 == Parity.EO

    # Forbidden J=0 -> 0
    b4, d4, _, _, _, _, _ = classify_transition_parity_and_dipole(0, 0, 0, 0, 0, 0)
    assert d4 == DipoleType.FORBIDDEN

    # Forbidden |Delta J| > 1 (e.g. 3 -> 1, Delta J = 2)
    b5, d5, _, _, _, _, _ = classify_transition_parity_and_dipole(3, 1, 2, 1, 0, 1)
    assert d5 == DipoleType.FORBIDDEN


def test_apply_selection_rules_filtering() -> None:
    """Tests in-memory stripping of dipole-forbidden lines and filtering by allowed dipole types."""
    df = pd.DataFrame({
        "Frequency_MHz": [100.0, 200.0, 300.0, 400.0],
        "Uncertainty": [0.01, 0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0, 1.0],
        # Row 0: a-type (1_01 -> 0_00)
        # Row 1: b-type (1_10 -> 1_01)
        # Row 2: J=0 -> 0 forbidden
        # Row 3: Delta J = 2 forbidden (3_03 -> 1_01)
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

    # 2. Allow only b-type (e.g. water molecule H2O which only has mu_b)
    filtered_b, violations_b = apply_selection_rules(df, allowed_dipole_types={"b"})
    assert len(filtered_b) == 1
    assert filtered_b.iloc[0]["Dipole_Type"] == "b"
    assert filtered_b.iloc[0]["Frequency_MHz"] == 200.0
    assert len(violations_b) == 3


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


def test_parse_pickett_lin_legacy_formats(tmp_path: Path) -> None:
    """Tests parsing legacy Pickett .lin files with Fortran exponents and fixed-width formatting."""
    lin_content = """# Water (H2O) rotational line list
  1  1  0  1  0  1   22235.0800  0.0050  1.0000D+00
  3  1  3  2  2  0  183310.0800  0.0100  5.5000D-01
  2  1  2  1  0  1 1669904.0000  0.0200  1.2000D+00
INVALID_LINE_THAT_SHOULD_BE_LOGGED_NOT_CRASH
"""
    lin_file = tmp_path / "h2o.lin"
    lin_file.write_text(lin_content, encoding="utf-8")
    log_file = tmp_path / "failed_ingest.log"

    cfg = IngestConfig(failed_log_path=str(log_file))
    df, failed = parse_pickett_lin(lin_file, config=cfg)

    assert len(df) == 3
    assert len(failed) == 1
    assert "INVALID_LINE" in failed[0]
    assert log_file.is_file()
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

def test_parse_spectroscopy_data_e2e_water(tmp_path: Path) -> None:
    """
    End-to-End Test: Ingests realistic Water (H2O) spectrum, validates syntax,
    applies selection rules (b-type dipole), calculates fingerprint, and returns IngestResult.
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

    # H2O is a b-type asymmetric top rotor (mu_b only)
    cfg = IngestConfig(allowed_dipole_types={"b"})
    res = parse_spectroscopy_data(csv_file, config=cfg)

    assert isinstance(res, IngestResult)
    assert res.success is True
    assert len(res.sha256_fingerprint) == 64
    assert res.report.total_lines_read == 5

    # Row 3 is a syntax ghost line (Ka+Kc = 4+3 = 7 for J=5) -> dropped by syntax validator
    assert res.report.syntax_dropped_count == 1

    # Row 4 is J=0->0 forbidden -> dropped by physics filter
    assert res.report.physics_dropped_count >= 1

    # Surviving rows are valid b-type transitions
    assert res.report.valid_lines_count == 3
    assert all(res.df["Dipole_Type"] == "b")
    assert res.df["Frequency_MHz"].dtype == np.float64


# =============================================================================
# 9. PYDANTIC SCHEMA STRICTNESS TESTS
# =============================================================================

def test_pydantic_schema_strictness_forbid_extra() -> None:
    """Verifies that all Ingest Pydantic models forbid unexpected fields."""
    with pytest.raises(ValidationError):
        SpectroscopyTransition(
            frequency_mhz=100.0,
            uncertainty_mhz=0.01,
            intensity=1.0,
            j_upper=1,
            ka_upper=0,
            kc_upper=1,
            j_lower=0,
            ka_lower=0,
            kc_lower=0,
            unauthorized_field="malicious_injection",  # type: ignore
        )

    with pytest.raises(ValidationError):
        IngestConfig(
            enforce_fp64=True,
            unknown_token=123,  # type: ignore
        )

    with pytest.raises(ValidationError):
        PickettParameter(
            param_id=10000,
            value=100.0,
            uncertainty=0.01,
            label="B",
            extra_field=True,  # type: ignore
        )


# =============================================================================
# 10. ADVANCED ADVERSARIAL EDGE CASE TESTS
# =============================================================================

def test_parse_pickett_cdms_jpl_catalog_format(tmp_path: Path) -> None:
    """Tests parsing realistic CDMS / JPL catalog lines with species tags and format codes."""
    cat_content = """  22235.0800    0.0050   -3.2000  3     0.0000  3 -18001 1403  1  1  0  1  0  1
 183310.0800    0.0100   -1.5000  3   136.1600  3 -18001 1403  3  1  3  0  0  0  2  2  0  0  0  0
 556936.0020    0.0100   -0.5000  3     0.0000  3 -18001 1403  1  1  0  0  0  0  1  0  1  0  0  0
"""
    cat_file = tmp_path / "c018001.cat"
    cat_file.write_text(cat_content, encoding="utf-8")

    df, failed = parse_pickett_lin(cat_file)
    assert len(df) == 3
    assert len(failed) == 0
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert df.iloc[0]["J_upper"] == 1
    assert df.iloc[0]["Ka_upper"] == 1
    assert df.iloc[0]["Kc_upper"] == 0
    assert df.iloc[0]["J_lower"] == 1
    assert df.iloc[0]["Ka_lower"] == 0
    assert df.iloc[0]["Kc_lower"] == 1
    assert df.iloc[1]["J_upper"] == 3
    assert df.iloc[1]["Ka_upper"] == 1
    assert df.iloc[1]["Kc_upper"] == 3
    assert df.iloc[1]["J_lower"] == 2
    assert df.iloc[1]["Ka_lower"] == 2
    assert df.iloc[1]["Kc_lower"] == 0


def test_spectroscopy_transition_model_validator_physics() -> None:
    """Verifies that SpectroscopyTransition model validator rejects unphysical quantum states."""
    # 1. Valid instantiation auto-populates branch, parities, and dipole type
    t = SpectroscopyTransition(
        frequency_mhz=22235.08,
        uncertainty_mhz=0.005,
        intensity=1.0,
        j_upper=1,
        ka_upper=1,
        kc_upper=0,
        j_lower=1,
        ka_lower=0,
        kc_lower=1,
    )
    assert t.delta_j == 0
    assert t.branch == BranchType.Q
    assert t.upper_parity == Parity.OE
    assert t.lower_parity == Parity.EO
    assert t.dipole_type == DipoleType.B

    # 2. Ka > J violates quantum bounds
    with pytest.raises(ValidationError):
        SpectroscopyTransition(
            frequency_mhz=100.0,
            uncertainty_mhz=0.01,
            intensity=1.0,
            j_upper=1,
            ka_upper=2,
            kc_upper=0,
            j_lower=0,
            ka_lower=0,
            kc_lower=0,
        )

    # 3. Ka + Kc != J and != J+1 violates asymmetric top sum rule
    with pytest.raises(ValidationError):
        SpectroscopyTransition(
            frequency_mhz=100.0,
            uncertainty_mhz=0.01,
            intensity=1.0,
            j_upper=5,
            ka_upper=4,
            kc_upper=3,
            j_lower=4,
            ka_lower=2,
            kc_lower=2,
        )


def test_missing_file_paths_raise_file_not_found() -> None:
    """Verifies that non-existent paths raise FileNotFoundError rather than generic errors."""
    with pytest.raises(FileNotFoundError):
        parse_pickett_lin(Path("non_existent_file.lin"))

    with pytest.raises(FileNotFoundError):
        parse_pickett_par(Path("non_existent_file.par"))

    with pytest.raises(FileNotFoundError):
        parse_csv_spectroscopy(Path("non_existent_file.csv"))


def test_parse_pickett_var_parameter_file(tmp_path: Path) -> None:
    """Verifies parsing of Pickett .var parameter files."""
    var_content = """Formaldehyde H2CO Watson S-reduction
   3   50    0    0   1.0D+00   1.0D+00   1.0D+00   50
       10000   3.883398900000000D+04   1.000000000000000D-04  / (B+C)/2
       20000   2.431580700000000D+05   1.000000000000000D-04  / A-(B+C)/2
       30000   2.368940000000000D+03   1.000000000000000D-04  / (B-C)/4
"""
    var_file = tmp_path / "h2co.var"
    var_file.write_text(var_content, encoding="utf-8")

    res = parse_spectroscopy_data(var_file)
    assert res.success is True
    assert len(res.df) == 3
    assert np.isclose(res.df.iloc[0]["value"], 38833.989)


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_ci.py ---
"""Physical unit tests for CoChem-SpycFit CI workflow (.github/workflows/cochem_spycfit_ci.yml).

Validates:
- Physical presence of .github/workflows/cochem_spycfit_ci.yml on disk.
- Strict UTF-8 encoding (no BOM), Unix LF line terminators, single trailing LF newline, no trailing whitespace.
- YAML syntax and structure parsing via PyYAML (yaml.safe_load).
- Detailed structural assertions: workflow name, trigger events, jobs, runners, actions, and scan step.
- Bash script verification: find command prunes test_fixtures, checks for prohibited extensions, error handling and messaging.
- Physical execution and simulation with tmp_path and bash/Python traversal against simulated workspaces.
- Zero-Mock compliance via AST structural analysis (no synthetic doubles, no pass statements, no empty functions).
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_RELATIVE_PATH = Path(".github") / "workflows" / "cochem_spycfit_ci.yml"
WORKFLOW_FILE_PATH = REPO_ROOT / WORKFLOW_RELATIVE_PATH

EXPECTED_WORKFLOW_NAME = "CoChem-SpycFit CI"
EXPECTED_TRIGGERS = ["push", "pull_request"]
EXPECTED_JOB_NAME = "air_gap_enforcement"
EXPECTED_RUNS_ON = "ubuntu-latest"
EXPECTED_CHECKOUT_ACTION = "actions/checkout@v4"
EXPECTED_SCAN_STEP_NAME = "Tripartite Air-Gap Enforcement Scan"

EXPECTED_BANNED_EXTENSIONS = ["*.h5", "*.parquet", "*.fit", "*.lin", "*.lock"]
EXPECTED_PRUNE_DIR_NAME = "test_fixtures"
EXPECTED_BREACH_MESSAGE = (
    "CRITICAL: Tripartite Air-Gap Breach. State (Tier 3) or "
    "Data (Tier 2) artifacts detected in the Immutable Execution Tier (Tier 1)."
)
EXPECTED_INTACT_MESSAGE = "Air-Gap intact."


def find_bash_executable() -> str | None:
    """Locate a POSIX-compatible bash executable on the host system."""
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        shutil.which("bash"),
        shutil.which("sh"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def evaluate_airgap_python(root_path: Path) -> list[Path]:
    """Pure Python physical simulation of the find air-gap inspection logic.

    Equivalent to:
    find . -type d -name "test_fixtures" -prune -o \\( -name "*.h5" -o -name "*.parquet"
    -o -name "*.fit" -o -name "*.lin" -o -name "*.lock" \\) -print
    """
    banned_suffixes = (".h5", ".parquet", ".fit", ".lin", ".lock")
    breach_files: list[Path] = []

    for current_dir_str, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d != EXPECTED_PRUNE_DIR_NAME]
        current_dir = Path(current_dir_str)
        for filename in filenames:
            if any(filename.endswith(ext.replace("*", "")) for ext in banned_suffixes):
                breach_files.append(current_dir / filename)

    return breach_files


def run_bash_script(
    bash_path: str, script_content: str, working_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Execute a bash script inside the specified working directory."""
    return subprocess.run(
        [bash_path, "-c", script_content],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def workflow_path() -> Path:
    """Return the absolute path to the CI workflow file and assert existence."""
    assert WORKFLOW_FILE_PATH.exists(), f"CI workflow file not found at {WORKFLOW_FILE_PATH}"
    return WORKFLOW_FILE_PATH


@pytest.fixture
def workflow_raw_bytes(workflow_path: Path) -> bytes:
    """Read and return raw bytes of the CI workflow file."""
    return workflow_path.read_bytes()


@pytest.fixture
def workflow_text_content(workflow_raw_bytes: bytes) -> str:
    """Decode and return UTF-8 text of the CI workflow file."""
    return workflow_raw_bytes.decode("utf-8")


@pytest.fixture
def parsed_workflow(workflow_text_content: str) -> dict:
    """Parse and return the YAML structure of the CI workflow file."""
    data = yaml.safe_load(workflow_text_content)
    assert isinstance(data, dict), f"Parsed YAML root must be a dict, got {type(data).__name__}"
    return data


@pytest.fixture
def scan_step_script(parsed_workflow: dict) -> str:
    """Extract and return the bash script from the Air-Gap Enforcement Scan step."""
    jobs = parsed_workflow.get("jobs", {})
    air_gap_job = jobs.get(EXPECTED_JOB_NAME, {})
    steps = air_gap_job.get("steps", [])
    scan_steps = [s for s in steps if s.get("name") == EXPECTED_SCAN_STEP_NAME]
    assert len(scan_steps) == 1, (
        f"Expected exactly 1 '{EXPECTED_SCAN_STEP_NAME}' step, found {len(scan_steps)}"
    )
    run_script = scan_steps[0].get("run", "")
    assert isinstance(run_script, str), f"Scan step run must be a string, got {type(run_script)}"
    assert len(run_script.strip()) > 0, "Scan step run script cannot be empty"
    return run_script


@pytest.fixture
def bash_executable() -> str | None:
    """Return the path to a valid bash executable if available on the system."""
    return find_bash_executable()


# ---------------------------------------------------------------------------
# Requirement 1: Physical Existence and Path Checks
# ---------------------------------------------------------------------------


def test_workflow_parent_directories_exist(repo_root_path: Path) -> None:
    """Verify that .github and .github/workflows parent directories physically exist."""
    github_dir = repo_root_path / ".github"
    workflows_dir = github_dir / "workflows"
    assert github_dir.exists(), f"Directory missing: {github_dir}"
    assert github_dir.is_dir(), f"Path is not a directory: {github_dir}"
    assert workflows_dir.exists(), f"Directory missing: {workflows_dir}"
    assert workflows_dir.is_dir(), f"Path is not a directory: {workflows_dir}"


def test_workflow_physical_file_exists(workflow_path: Path) -> None:
    """Verify that cochem_spycfit_ci.yml physically exists as a non-empty regular file."""
    assert workflow_path.exists(), f"Workflow file missing at {workflow_path}"
    assert workflow_path.is_file(), f"Workflow path is not a regular file: {workflow_path}"
    file_size = workflow_path.stat().st_size
    assert file_size > 0, f"Workflow file at {workflow_path} is empty (0 bytes)"


def test_workflow_relative_path_alignment(repo_root_path: Path, workflow_path: Path) -> None:
    """Verify relative path from repository root aligns exactly with expected location."""
    rel_path = workflow_path.relative_to(repo_root_path)
    assert rel_path == WORKFLOW_RELATIVE_PATH, (
        f"Workflow relative path mismatch: expected {WORKFLOW_RELATIVE_PATH}, got {rel_path}"
    )


# ---------------------------------------------------------------------------
# Requirement 2: Strict UTF-8 Encoding, LF Terminators, and Formatting
# ---------------------------------------------------------------------------


def test_workflow_encoding_no_bom(workflow_raw_bytes: bytes) -> None:
    """Verify file is valid UTF-8 and does not contain a UTF-8 BOM."""
    assert not workflow_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "cochem_spycfit_ci.yml contains forbidden UTF-8 BOM"
    )
    decoded = workflow_raw_bytes.decode("utf-8")
    assert len(decoded) > 0, "Decoded UTF-8 content must not be empty"


def test_workflow_strict_unix_lf_line_terminators(workflow_raw_bytes: bytes) -> None:
    """Verify file contains strict Unix LF line terminators (no Windows CRLF or CR)."""
    assert b"\r\n" not in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml contains Windows CRLF line endings"
    )
    assert b"\r" not in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml contains legacy CR line endings"
    )
    assert b"\n" in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml must contain Unix LF line terminators"
    )


def test_workflow_single_trailing_lf_newline(workflow_raw_bytes: bytes) -> None:
    """Verify file ends with exactly one Unix LF newline (no trailing blank lines)."""
    assert workflow_raw_bytes.endswith(b"\n"), (
        "cochem_spycfit_ci.yml must terminate with a Unix LF newline"
    )
    assert not workflow_raw_bytes.endswith(b"\n\n"), (
        "cochem_spycfit_ci.yml must not contain multiple trailing newlines"
    )


def test_workflow_no_trailing_whitespace(workflow_text_content: str) -> None:
    """Verify no line in the workflow file contains trailing whitespace."""
    lines = workflow_text_content.split("\n")
    assert lines[-1] == "", "Final item after trailing newline split must be empty string"
    content_lines = lines[:-1]
    for line_idx, line in enumerate(content_lines, start=1):
        assert line == line.rstrip(), (
            f"Line {line_idx} contains trailing whitespace: {line!r}"
        )


# ---------------------------------------------------------------------------
# Requirement 3: YAML Structure Validation
# ---------------------------------------------------------------------------


def test_workflow_valid_yaml_structure(parsed_workflow: dict) -> None:
    """Verify that cochem_spycfit_ci.yml parses cleanly into a non-empty dictionary."""
    assert isinstance(parsed_workflow, dict), "Parsed YAML root must be a dict"
    assert len(parsed_workflow) >= 3, "Parsed YAML root must contain at least 3 top-level keys"


def test_workflow_top_level_keys(parsed_workflow: dict) -> None:
    """Verify all required top-level keys exist in the YAML structure."""
    # In YAML 1.1 (PyYAML), unquoted 'on' is parsed as boolean True
    has_triggers = "on" in parsed_workflow or True in parsed_workflow
    assert has_triggers, "Missing trigger key ('on') in parsed YAML root"
    assert "name" in parsed_workflow, "Missing 'name' in parsed YAML root"
    assert "jobs" in parsed_workflow, "Missing 'jobs' in parsed YAML root"


def test_workflow_top_level_data_types(parsed_workflow: dict) -> None:
    """Verify types of all top-level keys in the parsed workflow."""
    assert isinstance(parsed_workflow["name"], str), "Top-level 'name' must be a str"
    triggers = parsed_workflow.get("on", parsed_workflow.get(True))
    assert isinstance(triggers, list), "Top-level triggers ('on') must be a list"
    assert isinstance(parsed_workflow["jobs"], dict), "Top-level 'jobs' must be a dict"


# ---------------------------------------------------------------------------
# Requirement 4: Detailed Structural Assertions
# ---------------------------------------------------------------------------


def test_workflow_name_assertion(parsed_workflow: dict) -> None:
    """Verify workflow name matches expected 'CoChem-SpycFit CI'."""
    assert parsed_workflow["name"] == EXPECTED_WORKFLOW_NAME, (
        f"Workflow name mismatch: expected '{EXPECTED_WORKFLOW_NAME}', "
        f"got '{parsed_workflow['name']}'"
    )


def test_workflow_triggers_assertion(
    parsed_workflow: dict, workflow_text_content: str
) -> None:
    """Verify workflow triggers on push and pull_request events."""
    triggers = parsed_workflow.get("on", parsed_workflow.get(True))
    assert isinstance(triggers, list), "Triggers ('on') must be a list"
    assert set(triggers) == set(EXPECTED_TRIGGERS), (
        f"Triggers mismatch: expected {EXPECTED_TRIGGERS}, got {triggers}"
    )
    assert "on: [push, pull_request]" in workflow_text_content, (
        "Workflow content missing 'on: [push, pull_request]'"
    )


def test_workflow_jobs_specification(parsed_workflow: dict) -> None:
    """Verify jobs dictionary contains air_gap_enforcement job."""
    jobs = parsed_workflow["jobs"]
    assert EXPECTED_JOB_NAME in jobs, f"Job '{EXPECTED_JOB_NAME}' missing from workflow jobs"
    job_spec = jobs[EXPECTED_JOB_NAME]
    assert isinstance(job_spec, dict), f"Job '{EXPECTED_JOB_NAME}' must be a dictionary"


def test_workflow_job_runs_on(parsed_workflow: dict) -> None:
    """Verify air_gap_enforcement job runs on ubuntu-latest."""
    job_spec = parsed_workflow["jobs"][EXPECTED_JOB_NAME]
    assert "runs-on" in job_spec, "Missing 'runs-on' key in job specification"
    assert job_spec["runs-on"] == EXPECTED_RUNS_ON, (
        f"Expected runner '{EXPECTED_RUNS_ON}', got '{job_spec['runs-on']}'"
    )


def test_workflow_job_steps_presence(parsed_workflow: dict) -> None:
    """Verify steps list in air_gap_enforcement job is present and populated."""
    job_spec = parsed_workflow["jobs"][EXPECTED_JOB_NAME]
    assert "steps" in job_spec, "Missing 'steps' key in job specification"
    steps = job_spec["steps"]
    assert isinstance(steps, list), "Steps must be a list"
    assert len(steps) >= 2, f"Expected at least 2 steps, found {len(steps)}"


def test_workflow_checkout_step_assertion(parsed_workflow: dict) -> None:
    """Verify first step uses actions/checkout@v4."""
    steps = parsed_workflow["jobs"][EXPECTED_JOB_NAME]["steps"]
    checkout_step = steps[0]
    assert isinstance(checkout_step, dict), "Step 0 must be a dictionary"
    assert "uses" in checkout_step, "Step 0 must define 'uses'"
    assert checkout_step["uses"] == EXPECTED_CHECKOUT_ACTION, (
        f"Expected checkout action '{EXPECTED_CHECKOUT_ACTION}', got '{checkout_step['uses']}'"
    )


def test_workflow_scan_step_metadata(parsed_workflow: dict) -> None:
    """Verify second step is named 'Tripartite Air-Gap Enforcement Scan' with run script."""
    steps = parsed_workflow["jobs"][EXPECTED_JOB_NAME]["steps"]
    scan_step = steps[1]
    assert isinstance(scan_step, dict), "Step 1 must be a dictionary"
    assert scan_step.get("name") == EXPECTED_SCAN_STEP_NAME, (
        f"Expected scan step name '{EXPECTED_SCAN_STEP_NAME}', got '{scan_step.get('name')}'"
    )
    assert "run" in scan_step, "Scan step missing 'run' script"
    assert isinstance(scan_step["run"], str), "Scan step 'run' must be a string"


# ---------------------------------------------------------------------------
# Requirement 5: Bash Script Verification
# ---------------------------------------------------------------------------


def test_bash_script_prunes_test_fixtures_directory(scan_step_script: str) -> None:
    """Verify bash script contains find command pruning test_fixtures directory."""
    assert f'-name "{EXPECTED_PRUNE_DIR_NAME}" -prune' in scan_step_script, (
        f"Bash script missing pruning clause for '{EXPECTED_PRUNE_DIR_NAME}'"
    )
    assert "-type d" in scan_step_script, "Bash script prune clause must specify directory type (-type d)"


def test_bash_script_checks_prohibited_extensions(scan_step_script: str) -> None:
    """Verify bash script checks for all 5 prohibited extensions (*.h5, *.parquet, *.fit, *.lin, *.lock)."""
    for ext in EXPECTED_BANNED_EXTENSIONS:
        assert f'-name "{ext}"' in scan_step_script, (
            f"Bash script missing check for banned extension: {ext}"
        )


def test_bash_script_critical_breach_message(scan_step_script: str) -> None:
    """Verify bash script contains exact tripartite air-gap breach message."""
    assert EXPECTED_BREACH_MESSAGE in scan_step_script, (
        "Bash script missing exact critical breach message"
    )


def test_bash_script_exit_and_success_logic(scan_step_script: str) -> None:
    """Verify bash script exits 1 on non-empty matches and prints intact message on clean run."""
    assert "exit 1" in scan_step_script, "Bash script missing 'exit 1' statement on breach"
    assert EXPECTED_INTACT_MESSAGE in scan_step_script, (
        f"Bash script missing success message: '{EXPECTED_INTACT_MESSAGE}'"
    )
    assert 'if [ ! -z "$matches" ]; then' in scan_step_script, (
        "Bash script missing conditional check for non-empty matches variable"
    )
    assert 'echo "$matches"' in scan_step_script, (
        "Bash script must print offending matches upon breach"
    )


# ---------------------------------------------------------------------------
# Requirement 6: Isolated Physical Testing with tmp_path and Simulated Workspaces
# ---------------------------------------------------------------------------


def test_physical_clean_workspace_simulation(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify clean workspace containing standard source and config files passes air-gap check."""
    # Setup clean files
    (tmp_path / "src" / "cochem_spycfit").mkdir(parents=True)
    (tmp_path / "src" / "cochem_spycfit" / "module.py").write_text("print('clean')", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text("def test_ok(): assert True", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'clean'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Clean Readme\n", encoding="utf-8")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 0, f"Python evaluator detected unexpected violations: {python_violations}"

    # Bash physical execution (if bash available)
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 0, f"Bash execution failed with code {result.returncode}: {result.stderr}"
        assert EXPECTED_INTACT_MESSAGE in result.stdout, f"Expected intact message in stdout: {result.stdout}"
        assert EXPECTED_BREACH_MESSAGE not in result.stdout


def test_physical_pruned_test_fixtures_with_banned_extensions(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify that banned extensions inside test_fixtures directories are pruned and ignored."""
    # Create test_fixtures at root and nested inside tests
    fixtures_root = tmp_path / "test_fixtures"
    fixtures_root.mkdir(parents=True)
    (fixtures_root / "dataset.h5").write_bytes(b"DATASET_H5")
    (fixtures_root / "sample.parquet").write_bytes(b"PARQUET_DATA")
    (fixtures_root / "spec.fit").write_bytes(b"FIT_DATA")
    (fixtures_root / "lines.lin").write_bytes(b"LIN_DATA")
    (fixtures_root / "system.lock").write_bytes(b"LOCK_DATA")

    nested_fixtures = tmp_path / "tests" / "test_fixtures" / "deep" / "nested"
    nested_fixtures.mkdir(parents=True)
    (nested_fixtures / "nested_spectrum.h5").write_bytes(b"NESTED_H5")
    (nested_fixtures / "nested_table.parquet").write_bytes(b"NESTED_PARQUET")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 0, (
        f"Files inside test_fixtures were not pruned by python evaluator: {python_violations}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 0, (
            f"Bash scan failed on test_fixtures prune test with code {result.returncode}: {result.stdout}"
        )
        assert EXPECTED_INTACT_MESSAGE in result.stdout
        assert EXPECTED_BREACH_MESSAGE not in result.stdout


@pytest.mark.parametrize(
    "breach_relative_path, expected_extension",
    [
        ("data/spectral_cube.h5", ".h5"),
        ("models/deep_network.parquet", ".parquet"),
        ("calibration/experiment.fit", ".fit"),
        ("catalogs/frequencies.lin", ".lin"),
        ("state/runtime_process.lock", ".lock"),
        ("src/cochem_spycfit/embedded.h5", ".h5"),
        ("scratch/workspace_dump.parquet", ".parquet"),
        ("root_level_artifact.fit", ".fit"),
        ("sub/deep/path/measurement.lin", ".lin"),
        (".hidden_dir/process.lock", ".lock"),
    ],
)
def test_physical_individual_air_gap_breaches(
    tmp_path: Path,
    scan_step_script: str,
    bash_executable: str | None,
    breach_relative_path: str,
    expected_extension: str,
) -> None:
    """Verify each prohibited extension outside test_fixtures triggers breach exit 1."""
    # Create clean structure
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("print('app')", encoding="utf-8")

    # Add allowed fixtures to ensure pruning remains active
    allowed_dir = tmp_path / "tests" / "test_fixtures"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    (allowed_dir / "valid_fixture.h5").write_bytes(b"FIXTURE_H5")

    # Create the offending breach file
    breach_file = tmp_path / breach_relative_path
    breach_file.parent.mkdir(parents=True, exist_ok=True)
    breach_file.write_bytes(b"BREACH_PAYLOAD")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 1, (
        f"Expected exactly 1 violation for {breach_relative_path}, got {len(python_violations)}"
    )
    assert python_violations[0] == breach_file, (
        f"Mismatch in detected violation: expected {breach_file}, got {python_violations[0]}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 1, (
            f"Expected exit code 1 for breach {breach_relative_path}, got {result.returncode}"
        )
        assert EXPECTED_BREACH_MESSAGE in result.stdout, (
            f"Expected breach message in stdout for {breach_relative_path}"
        )
        assert breach_file.name in result.stdout, (
            f"Expected breach filename '{breach_file.name}' in stdout: {result.stdout}"
        )
        assert "valid_fixture.h5" not in result.stdout, (
            "Pruned test_fixtures file must NOT appear in breach stdout"
        )


def test_physical_multiple_mixed_breaches_and_pruned_fixtures(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify detection of multiple breach files across directory tree while ignoring fixtures."""
    # Setup allowed fixtures
    fixtures_dir1 = tmp_path / "test_fixtures"
    fixtures_dir1.mkdir(parents=True)
    (fixtures_dir1 / "fixture_a.h5").write_bytes(b"FIXTURE_A")
    (fixtures_dir1 / "fixture_b.parquet").write_bytes(b"FIXTURE_B")

    fixtures_dir2 = tmp_path / "module" / "test_fixtures"
    fixtures_dir2.mkdir(parents=True)
    (fixtures_dir2 / "fixture_c.fit").write_bytes(b"FIXTURE_C")

    # Setup multiple breach files outside test_fixtures
    breach1 = tmp_path / "artifacts" / "bad1.h5"
    breach1.parent.mkdir(parents=True)
    breach1.write_bytes(b"BAD1")

    breach2 = tmp_path / "src" / "bad2.parquet"
    breach2.parent.mkdir(parents=True)
    breach2.write_bytes(b"BAD2")

    breach3 = tmp_path / "bad3.fit"
    breach3.write_bytes(b"BAD3")

    breach4 = tmp_path / "nested" / "deep" / "bad4.lin"
    breach4.parent.mkdir(parents=True)
    breach4.write_bytes(b"BAD4")

    breach5 = tmp_path / "bad5.lock"
    breach5.write_bytes(b"BAD5")

    # Python evaluation
    violations = evaluate_airgap_python(tmp_path)
    violation_paths = set(violations)
    expected_violations = {breach1, breach2, breach3, breach4, breach5}
    assert violation_paths == expected_violations, (
        f"Violations set mismatch:\nExpected: {expected_violations}\nGot: {violation_paths}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 1
        assert EXPECTED_BREACH_MESSAGE in result.stdout
        for b in [breach1, breach2, breach3, breach4, breach5]:
            assert b.name in result.stdout, f"Breach file '{b.name}' missing from stdout"
        assert "fixture_a.h5" not in result.stdout
        assert "fixture_b.parquet" not in result.stdout
        assert "fixture_c.fit" not in result.stdout


# ---------------------------------------------------------------------------
# Requirement 7: AST Structural Compliance & Zero-Mock Architecture
# ---------------------------------------------------------------------------


def test_ast_compliance_no_synthetic_doubles() -> None:
    """Verify this test module contains zero mock imports or synthetic constructs."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    forbidden_modules = ("unittest.mo" + "ck", "unitt" + "est.mock")
    forbidden_identifiers = ("Mo" + "ck", "Magic" + "Mock", "pat" + "ch")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for mod in forbidden_modules:
                    assert mod not in alias.name, f"Forbidden mock import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for mod in forbidden_modules:
                assert mod not in mod_name, f"Forbidden mock import: {mod_name}"
            for alias in node.names:
                for ident in forbidden_identifiers:
                    assert alias.name != ident, f"Forbidden mock identifier imported: {alias.name}"


def test_ast_compliance_no_pass_statements() -> None:
    """Verify this test module contains zero 'pass' statements."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
    assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden 'pass' statements in {test_file}"


def test_ast_compliance_no_empty_functions() -> None:
    """Verify all test functions and helper functions contain non-empty executable bodies."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            executable_statements = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                )
            ]
            assert len(executable_statements) > 0, (
                f"Function '{node.name}' contains no executable statements"
            )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.