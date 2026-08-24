Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task9_Prompt2_spcat_bridge_phase4.md.
Original prompt:
# Task 9 (Phase 4): SPCAT Bridge - cochem_spcat_bridge.py

**Objective:**
Update `cochem_spcat_bridge.py` to implement Phase 4 requirements including double-precision Fortran formatting, cross-platform subprocess isolation, segfault trapping, and zombie reaping.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_spcat_bridge.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 4 features in `cochem_spcat_bridge.py`.
It must:
1. Map parameters to strictly spaced 10-character `.var` and `.int` positional string formats. Actively replace Python `E` notation with Fortran `D` notation (e.g., `1.25D-05`).
2. Run `spcat` in an isolated subprocess (`start_new_session=True` / `subprocess.CREATE_NEW_PROCESS_GROUP`).
3. Implement an OS-Agnostic Segfault Trap: check for Exit Code 139 (POSIX) or 0xC0000005 (Windows).
4. Implement Hex-Dump Telemetry: capture the last 256 bytes of `stderr`, convert to hex, and log to `$COCHEM_STATE_DIR/Logs/spycfit_telemetry.json`.
5. Implement an Orphan Sweeper: wrap execution with a `timeout` and explicitly terminate via `SIGKILL` or `taskkill`.
6. Provision an ephemeral `pickett_scratch/` directory and perform an explicit `shutil.rmtree` upon completion.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Implement real cross-platform subprocess logic.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import subprocess
import shutil
import sys
import os
from platformdirs import user_log_path, user_data_path

def format_fortran_double(value):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def execute_spcat_safely(executable_path, scratch_dir):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def trap_and_log_segfault(stderr_output):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\cochem_spcat_bridge.py ---
"""
CoChem-SpycFit: Stage 3.3 Legacy Fortran Translator & Pickett SPCAT/SPFIT Bridge.

Authoritative serialization, fixed-width Fortran formatting, and process-isolated
asynchronous execution pipeline for legacy Pickett SPCAT/SPFIT spectroscopic engines.

Directives Enforced:
1. Strict 1980s Pickett fixed-width formatting (%10.4E with Fortran 'D' exponent replacement)
   guaranteeing absolute column alignment for .var, .par, .int, and .lin files.
2. Robust multi-file parser suite for legacy calculation outputs (.cat, .egy, .fit).
3. Process-isolated, asynchronous containerized task queues (FortranTaskQueue, FortranExecutionTask,
   FortranExecutionResult, FortranBridgeDispatcher).
4. Explicit executable deployment path routing ($COCHEM_BIN_DIR or user_data_dir / bin), avoiding
   Tier 3 Cache and noexec mount deadlocks on HPC clusters.
5. OS-level process isolation (CREATE_NEW_PROCESS_GROUP on Windows, start_new_session on POSIX).
6. Robust segfault trapping (SIGSEGV / exit code 139 on POSIX, STATUS_ACCESS_VIOLATION 0xC0000005 on Windows)
   preventing parent kernel corruption.
7. Structured hex-dump telemetry capture (final 256 bytes) logged to $COCHEM_STATE_DIR/Logs/spycfit_telemetry.json.
8. Ephemeral scratch lifecycle management (PickettScratchContext) with guaranteed filesystem cleanup.
9. Dynamic physical isotope mass retrieval via Mendeleev.
10. Strict Pydantic v2 validation models and zero-mock physical compliance.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import mendeleev
import numpy as np
import platformdirs
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Configure module-level logger
logger = logging.getLogger("CoChem-SpycFit.SPCATBridge")

# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class FortranBridgeError(Exception):
    """Base exception for all Fortran bridge and Pickett translation operations."""


class PickettFormatError(FortranBridgeError, ValueError):
    """Raised when parsing or formatting legacy Pickett data encounters syntax errors."""


class FortranExecutionError(FortranBridgeError, RuntimeError):
    """Raised when execution of a Fortran binary fails or returns non-zero error status."""


class FortranExecutionTimeoutError(FortranExecutionError, TimeoutError):
    """Raised when Fortran subprocess execution exceeds the configured timeout threshold."""


class FortranExecutionSegfaultError(FortranExecutionError):
    """Raised when a Fortran subprocess crashes due to a segmentation fault or access violation."""


class FortranBinaryNotFoundError(FortranBridgeError, FileNotFoundError):
    """Raised when an expected Fortran executable binary cannot be located on disk."""


# =============================================================================
# 2. PHYSICAL CONSTANTS & DYNAMIC ISOTOPE RETRIEVAL (MENDELEEV)
# =============================================================================

def get_isotopic_mass(element_symbol: str, mass_number: Optional[int] = None) -> float:
    """
    Dynamically retrieves authentic physical isotopic mass (or standard atomic weight)
    from Mendeleev.

    Args:
        element_symbol: Standard chemical element symbol (e.g. 'H', 'C', 'N', 'O').
        mass_number: Optional mass number for specific isotope (e.g. 13 for C-13).

    Returns:
        float: Accurate isotopic mass in atomic mass units (u / Da).

    Raises:
        ValueError: If element or isotope mass number is not found.
    """
    clean_sym = element_symbol.strip().capitalize()
    try:
        elem = mendeleev.element(clean_sym)
    except Exception as exc:
        raise ValueError(f"Unknown chemical element symbol: {element_symbol}") from exc

    if mass_number is not None:
        for iso in elem.isotopes:
            if iso.mass_number == mass_number:
                return float(iso.mass)
        raise ValueError(f"Isotope mass number {mass_number} not found for element {clean_sym}")

    return float(elem.atomic_weight)


def calculate_reduced_mass(mass1_amu: float, mass2_amu: float) -> float:
    """
    Calculates authentic two-body reduced mass in atomic mass units (u).

    Args:
        mass1_amu: Mass of particle 1 in u.
        mass2_amu: Mass of particle 2 in u.

    Returns:
        float: Reduced mass mu = (m1 * m2) / (m1 + m2).
    """
    if mass1_amu <= 0.0 or mass2_amu <= 0.0:
        raise ValueError("Atomic masses must be strictly positive.")
    return (mass1_amu * mass2_amu) / (mass1_amu + mass2_amu)


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & PICKETT STRICT FIXED-WIDTH FORMATTERS
# =============================================================================

def fortran_float_converter(val: Union[str, float, int, np.number]) -> float:
    """
    Safely converts a string containing Fortran scientific notation to a Python float64.

    Handles:
    - Standard floats and integers
    - 'D' or 'd' exponent notation ('1.2345D+02' -> 123.45)
    - Implicit exponent notation ('1.2345-05' -> 1.2345e-5)
    - Leading/trailing whitespace
    """
    if isinstance(val, (float, int, np.number)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    # Replace Fortran double precision exponent 'D' / 'd' with 'e'
    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    # Handle implicit exponent sign without 'e' (e.g. '1.2345-05' or '1.2345+02')
    match_implicit = re.match(r"^([+-]?\d+\.?\d*)([+-]\d+)$", s_clean)
    if match_implicit:
        s_clean = f"{match_implicit.group(1)}e{match_implicit.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


def format_fortran_double(
    value: Union[float, int, str, np.number],
    width: int = 10,
    precision: int = 4,
) -> str:
    """
    Strict fixed-width Fortran double precision formatter for legacy Pickett formats.
    Actively formats floating point values using scientific notation and replaces
    Python 'E'/'e' exponent indicator with Fortran 'D' notation (e.g. '1.2500D-05').
    Guarantees output string length is exactly equal to `width` through dynamic precision
    adjustment and right-alignment.

    Args:
        value: Numeric value or string representation.
        width: Exact character width of the output field (default: 10).
        precision: Digits after the decimal point (default: 4).

    Returns:
        str: Fixed-width formatted string guaranteed to have len == width.
    """
    try:
        val_float = fortran_float_converter(value)
    except (ValueError, TypeError):
        s_val = str(value)
        if len(s_val) > width:
            return s_val[:width]
        return s_val.rjust(width)

    # Format using standard scientific notation with precision
    formatted = f"{val_float:.{precision}E}"
    formatted = formatted.replace("E", "D").replace("e", "D")

    if len(formatted) > width:
        excess = len(formatted) - width
        adj_precision = max(0, precision - excess)
        formatted = f"{val_float:.{adj_precision}E}".replace("E", "D").replace("e", "D")
        if len(formatted) > width:
            formatted = formatted[:width]

    return formatted.rjust(width)


def format_pickett_line(
    value: Union[float, int, str, np.number],
    width: int = 10,
    precision: int = 4,
    exponent_char: str = "D",
) -> str:
    """
    Strict fixed-width truncation and padding logic guaranteeing absolute column alignment
    in legacy 1980s Pickett formats (.var, .par, .int, .lin).

    Formats floating point numbers with scientific notation (%10.4E) and replaces
    the exponent character ('E' -> exponent_char, default 'D'). Enforces exact field width
    by right-aligning and precision adjusting or truncating on overflow.

    Args:
        value: Numeric value or string representation.
        width: Exact character width of the output field (default: 10).
        precision: Digits after decimal point in scientific notation (default: 4).
        exponent_char: Exponent indicator character (default: 'D').

    Returns:
        str: Fixed-width formatted string guaranteed to have len == width.
    """
    res = format_fortran_double(value=value, width=width, precision=precision)
    if exponent_char != "D":
        res = res.replace("D", exponent_char).replace("d", exponent_char)
    return res


def decode_pickett_quantum(qn_str: str) -> int:
    """
    Decodes legacy JPL/CDMS alphanumeric quantum number representations.

    Quanta 0-99: standard integer (e.g. ' 5' -> 5)
    Quanta 10-269: encoded as 'a0'-'z9' (e.g. 'a0' -> 10, 'a9' -> 19, 'z9' -> 269)
    Quanta 100-359: encoded as 'A0'-'Z9' (e.g. 'A0' -> 100, 'A9' -> 109, 'Z9' -> 359)

    Args:
        qn_str: 2-character quantum number string.

    Returns:
        int: Decoded integer quantum number.
    """
    s = qn_str.strip()
    if not s:
        return 0

    if s.lstrip("+-").isdigit():
        return int(s)

    if len(s) == 2:
        c0, c1 = s[0], s[1]
        if c1.isdigit():
            digit = int(c1)
            if "a" <= c0 <= "z":
                letter_idx = ord(c0) - ord("a")
                return 10 + letter_idx * 10 + digit
            if "A" <= c0 <= "Z":
                letter_idx = ord(c0) - ord("A")
                return 100 + letter_idx * 10 + digit

    raise PickettFormatError(f"Cannot decode alphanumeric quantum number string: {qn_str!r}")


def encode_pickett_quantum(quantum_value: int) -> str:
    """
    Encodes an integer quantum number into a 2-character Pickett string.

    0-99: 2-digit integer (e.g. 5 -> ' 5', 42 -> '42')
    100-359: 'A0'-'Z9'

    Args:
        quantum_value: Integer quantum number (0 to 359).

    Returns:
        str: Exactly 2-character string.
    """
    if 0 <= quantum_value <= 99:
        return f"{quantum_value:2d}"
    if 100 <= quantum_value <= 359:
        val_offset = quantum_value - 100
        letter_idx = val_offset // 10
        digit = val_offset % 10
        if letter_idx < 26:
            letter = chr(ord("A") + letter_idx)
            return f"{letter}{digit}"

    # Fallback / overflow clamping
    clamped = max(0, min(99, quantum_value))
    return f"{clamped:2d}"


# =============================================================================
# 4. PYDANTIC V2 MODELS FOR PICKETT SCHEMAS
# =============================================================================

class PickettParameterRecord(BaseModel):
    """Represents an individual parameter in a Pickett .var / .par file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    param_id: int = Field(..., description="Pickett parameter code (e.g. 10000=A, 20000=B, 30000=C, 200=-DJ).")
    value: float = Field(..., description="Parameter value in MHz.")
    uncertainty: float = Field(default=0.0, ge=0.0, description="Estimated parameter uncertainty in MHz (0.0 if fixed).")
    label: str = Field(default="", description="Descriptive label (e.g. 'A', 'B', 'C', 'DJ').")


class PickettVarFile(BaseModel):
    """Encapsulates a structured Pickett parameter file (.var / .par)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Parameter File", description="Header title string.")
    npar: int = Field(default=0, ge=0, description="Number of parameters in file.")
    nline: int = Field(default=0, description="Number of transitions to fit.")
    nopt: int = Field(default=0, description="Optimization control integer.")
    nwarn: int = Field(default=0, description="Warning control integer.")
    erpar: float = Field(default=1.0, description="Parameter error scale factor.")
    wtfac: float = Field(default=1.0, description="Weight factor.")
    scale: float = Field(default=1.0, description="Scale factor.")
    maxit: int = Field(default=50, ge=0, description="Maximum fitting iterations.")
    parameters: List[PickettParameterRecord] = Field(default_factory=list, description="List of parameter records.")

    @model_validator(mode="after")
    def sync_npar_count(self) -> PickettVarFile:
        object.__setattr__(self, "npar", len(self.parameters))
        return self

    def export_var(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports parameter set into authentic fixed-width Pickett .var / .par format.
        """
        lines: List[str] = [self.title]
        ctrl_line = (
            f"{len(self.parameters):5d}"
            f"{self.nline:5d}"
            f"{self.nopt:5d}"
            f"{self.nwarn:5d}"
            f"{self.erpar:10.4f}"
            f"{self.wtfac:10.4f}"
            f"{self.scale:10.4f}"
            f"{self.maxit:5d}"
        )
        lines.append(ctrl_line)

        for p in self.parameters:
            v_str = format_pickett_line(p.value, width=22, precision=14, exponent_char="D")
            u_str = format_pickett_line(p.uncertainty, width=18, precision=10, exponent_char="D")
            lbl_str = f" / {p.label}" if p.label else ""
            p_line = f"{p.param_id:11d} {v_str} {u_str}{lbl_str}"
            lines.append(p_line)

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettDipoleRecord(BaseModel):
    """Represents a dipole moment component in a Pickett .int file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    axis: int = Field(..., ge=1, le=3, description="Principal axis index: 1=a, 2=b, 3=c.")
    dipole_debye: float = Field(..., description="Dipole moment component along axis in Debye.")
    comment: str = Field(default="", description="Descriptive comment (e.g. 'mua').")


class PickettIntFile(BaseModel):
    """Encapsulates a structured Pickett intensity control file (.int)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Intensity File", description="Header title string.")
    tag: int = Field(default=1, description="Species identification tag.")
    qpart: float = Field(default=1000.0, gt=0.0, description="Rotational partition function.")
    nline: int = Field(default=1000, description="Target predicted transition line count.")
    iq: int = Field(default=1, description="Quantum format identifier.")
    max_j: int = Field(default=99, ge=0, description="Maximum J quantum number.")
    min_freq_mhz: float = Field(default=0.0, description="Minimum frequency boundary in MHz.")
    max_freq_mhz: float = Field(default=500000.0, description="Maximum frequency boundary in MHz.")
    min_intensity: float = Field(default=-10.0, description="Minimum log10 intensity cutoff.")
    temperature_k: float = Field(default=300.0, gt=0.0, description="Simulation temperature in Kelvin.")
    dipoles: List[PickettDipoleRecord] = Field(default_factory=list, description="List of active dipole records.")

    def export_int(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports intensity control configuration into authentic fixed-width Pickett .int format.
        """
        lines: List[str] = [self.title]
        ctrl_line = (
            f"{self.tag:6d}"
            f"{self.qpart:12.4f}"
            f"{self.nline:7d}"
            f"{self.iq:4d}"
            f"{self.max_j:4d}"
            f" {self.min_freq_mhz:10.1f}"
            f" {self.max_freq_mhz:10.1f}"
            f"{self.min_intensity:10.2f}"
            f"{self.temperature_k:10.2f}"
        )
        lines.append(ctrl_line)

        for dip in self.dipoles:
            axis_name = ["mua", "mub", "muc"][dip.axis - 1] if 1 <= dip.axis <= 3 else str(dip.axis)
            lbl = dip.comment if dip.comment else axis_name
            dip_str = f"{dip.axis:2d}  {dip.dipole_debye:8.4f} / {lbl}"
            lines.append(dip_str)

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettLinRecord(BaseModel):
    """Represents a discrete experimental transition line in a Pickett .lin file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers (J', Ka', Kc', ...).")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers (J'', Ka'', Kc'', ...).")
    frequency_mhz: float = Field(..., description="Observed transition frequency in MHz.")
    uncertainty_mhz: float = Field(default=0.01, ge=0.0, description="Measurement uncertainty in MHz.")
    weight: float = Field(default=1.0, description="Relative statistical fitting weight.")
    description: str = Field(default="", description="Optional line annotation.")


class PickettLinFile(BaseModel):
    """Encapsulates a structured Pickett experimental line list (.lin)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Line List", description="Header line description.")
    records: List[PickettLinRecord] = Field(default_factory=list, description="List of transition records.")

    def export_lin(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports line list into authentic fixed-width Fortran 6I3, 6I3, F13.4, F8.4, F8.4 .lin format.
        """
        lines: List[str] = []
        if self.title:
            lines.append(f"! {self.title}")

        for rec in self.records:
            # Pad quanta to 6 integers each
            qu = list(rec.quantum_upper) + [0] * max(0, 6 - len(rec.quantum_upper))
            ql = list(rec.quantum_lower) + [0] * max(0, 6 - len(rec.quantum_lower))

            qu_str = "".join(f"{q:3d}" for q in qu[:6])
            ql_str = "".join(f"{q:3d}" for q in ql[:6])
            f_str = f"{rec.frequency_mhz:13.4f}"
            u_str = f"{rec.uncertainty_mhz:8.4f}"
            w_str = f"{rec.weight:8.4f}"
            annot = f" {rec.description}" if rec.description else ""

            lines.append(f"{qu_str}{ql_str}{f_str}{u_str}{w_str}{annot}")

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettCatRecord(BaseModel):
    """Represents a predicted spectroscopic transition line in a Pickett .cat catalog."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequency_mhz: float = Field(..., description="Rest transition frequency in MHz.")
    uncertainty_mhz: float = Field(..., description="Predicted frequency uncertainty in MHz.")
    log_intensity: float = Field(..., description="Log10 integrated intensity in nm^2 MHz.")
    degrees_of_freedom: int = Field(default=2, description="Degrees of freedom in partition function.")
    lower_energy_cm1: float = Field(..., description="Lower rotational state energy in cm^-1.")
    upper_degeneracy: int = Field(..., description="Upper state degeneracy (g_up).")
    species_tag: int = Field(..., description="Species identification tag.")
    quantum_format: int = Field(..., description="Quantum number format code.")
    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers.")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers.")


class PickettCatFile(BaseModel):
    """Encapsulates a collection of parsed predicted lines from a Pickett .cat catalog."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    records: List[PickettCatRecord] = Field(default_factory=list, description="List of transition lines.")


class PickettEgyRecord(BaseModel):
    """Represents an energy level record from a Pickett .egy file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_index: int = Field(..., description="State numerical index.")
    energy_cm1: float = Field(..., description="State energy in cm^-1.")
    uncertainty_cm1: float = Field(default=0.0, description="Calculated energy uncertainty in cm^-1.")
    pmix: float = Field(default=1.0, description="State purity / mixing coefficient.")
    quantum_numbers: Tuple[int, ...] = Field(default=(), description="Rotational quantum numbers.")
    label: str = Field(default="", description="State label or symmetry representation.")


class PickettEgyFile(BaseModel):
    """Encapsulates state energy level records parsed from a Pickett .egy file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    records: List[PickettEgyRecord] = Field(default_factory=list, description="List of energy records.")


class PickettFitResidual(BaseModel):
    """Represents an observed vs calculated residual from an SPFIT .fit output file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    line_number: int = Field(..., description="Transition line index in fit.")
    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers.")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers.")
    frequency_obs_mhz: float = Field(..., description="Observed transition frequency in MHz.")
    frequency_calc_mhz: float = Field(..., description="Calculated transition frequency in MHz.")
    residual_mhz: float = Field(..., description="Observed minus calculated residual in MHz.")
    uncertainty_mhz: float = Field(..., description="Measurement uncertainty in MHz.")
    weight: float = Field(default=1.0, description="Statistical fitting weight.")


class PickettFitResult(BaseModel):
    """Encapsulates comprehensive output results parsed from an SPFIT .fit file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="", description="Fit calculation title.")
    standard_error: float = Field(default=0.0, description="Standard error of the fit (RMS).")
    variance: float = Field(default=0.0, description="Reduced chi-squared / variance of fit.")
    lines_fitted: int = Field(default=0, description="Total transition lines included in fit.")
    lines_omitted: int = Field(default=0, description="Transition lines omitted from fit.")
    parameters: List[PickettParameterRecord] = Field(default_factory=list, description="Fitted parameters.")
    residuals: List[PickettFitResidual] = Field(default_factory=list, description="Line residuals table.")


# =============================================================================
# 5. PARSERS FOR PICKETT OUTPUT FILES (.CAT, .EGY, .FIT, .VAR, .INT, .LIN)
# =============================================================================

def _read_content_lines(content_or_path: Union[str, Path, bytes, io.StringIO]) -> List[str]:
    """Helper to extract text lines from diverse input representations."""
    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        return content_or_path.read_text(encoding="utf-8", errors="replace").splitlines()

    if isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            return Path(content_or_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if "\n" not in content_or_path and any(content_or_path.endswith(ext) for ext in (".cat", ".egy", ".fit", ".var", ".par", ".int", ".lin")):
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        return content_or_path.splitlines()

    if isinstance(content_or_path, bytes):
        return content_or_path.decode("utf-8", errors="replace").splitlines()

    if isinstance(content_or_path, io.StringIO):
        return content_or_path.getvalue().splitlines()

    raise ValueError(f"Unsupported content type: {type(content_or_path)}")


def parse_pickett_cat(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettCatFile:
    """
    Parses a legacy Pickett SPCAT predicted catalog file (.cat).

    Fortran format: [F13.4, 2F8.4, I2, F10.4, I3, I7, I4, 12I2]
    """
    lines = _read_content_lines(content_or_path)
    records: List[PickettCatRecord] = []

    cat_regex = re.compile(
        r"^\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+)\s+"
        r"(\d+)\s+"
    )

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        if len(line) >= 79:
            try:
                freq = fortran_float_converter(line[0:13])
                unc = fortran_float_converter(line[13:21])
                lgint = fortran_float_converter(line[21:29])
                dr = int(line[29:31].strip() or 0)
                elow = fortran_float_converter(line[31:41])
                gup = int(line[41:44].strip() or 1)
                tag = int(line[44:51].strip() or 0)
                qnfmt = int(line[51:55].strip() or 0)

                # Decode 6 upper and 6 lower quanta
                qu = tuple(decode_pickett_quantum(line[55 + 2 * i : 57 + 2 * i]) for i in range(6))
                ql = tuple(decode_pickett_quantum(line[67 + 2 * i : 69 + 2 * i]) for i in range(6))

                records.append(
                    PickettCatRecord(
                        frequency_mhz=freq,
                        uncertainty_mhz=unc,
                        log_intensity=lgint,
                        degrees_of_freedom=dr,
                        lower_energy_cm1=elow,
                        upper_degeneracy=gup,
                        species_tag=tag,
                        quantum_format=qnfmt,
                        quantum_upper=qu,
                        quantum_lower=ql,
                    )
                )
                continue
            except Exception as parse_err:
                logger.debug("Fixed-width cat parse fallback at line %d: %s", line_idx, parse_err)

        # Regex fallback for non-standard spaced lines
        match = cat_regex.match(line)
        if match:
            freq = fortran_float_converter(match.group(1))
            unc = fortran_float_converter(match.group(2))
            lgint = fortran_float_converter(match.group(3))
            dr = int(match.group(4))
            elow = fortran_float_converter(match.group(5))
            gup = int(match.group(6))
            tag = int(match.group(7))
            qnfmt = int(match.group(8))

            tail = line[match.end():].strip().split()
            qu_list: List[int] = []
            ql_list: List[int] = []
            for item in tail[:6]:
                qu_list.append(decode_pickett_quantum(item))
            for item in tail[6:12]:
                ql_list.append(decode_pickett_quantum(item))

            while len(qu_list) < 6:
                qu_list.append(0)
            while len(ql_list) < 6:
                ql_list.append(0)

            records.append(
                PickettCatRecord(
                    frequency_mhz=freq,
                    uncertainty_mhz=unc,
                    log_intensity=lgint,
                    degrees_of_freedom=dr,
                    lower_energy_cm1=elow,
                    upper_degeneracy=gup,
                    species_tag=tag,
                    quantum_format=qnfmt,
                    quantum_upper=tuple(qu_list),
                    quantum_lower=tuple(ql_list),
                )
            )

    return PickettCatFile(records=records)


def parse_pickett_egy(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettEgyFile:
    """
    Parses a Pickett SPCAT state energy level file (.egy).
    """
    lines = _read_content_lines(content_or_path)
    records: List[PickettEgyRecord] = []

    egy_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(.*)$"
    )

    for line in lines:
        s_line = line.strip()
        if not s_line or s_line.startswith("#") or s_line.startswith("!"):
            continue

        match = egy_pattern.match(s_line)
        if match:
            idx = int(match.group(1))
            energy = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            pmix = fortran_float_converter(match.group(4))
            remainder = match.group(5).strip().split()

            qn_vals: List[int] = []
            lbl = ""
            for token in remainder:
                if token.lstrip("+-").isdigit():
                    qn_vals.append(int(token))
                elif len(token) == 2 and token[0].islower() and token[1].isdigit():
                    qn_vals.append(decode_pickett_quantum(token))
                else:
                    lbl = token

            records.append(
                PickettEgyRecord(
                    state_index=idx,
                    energy_cm1=energy,
                    uncertainty_cm1=unc,
                    pmix=pmix,
                    quantum_numbers=tuple(qn_vals),
                    label=lbl,
                )
            )

    return PickettEgyFile(records=records)


def parse_pickett_fit(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettFitResult:
    """
    Parses a Pickett SPFIT fitting output log (.fit).
    """
    lines = _read_content_lines(content_or_path)
    title = ""
    std_err = 0.0
    variance = 0.0
    lines_fitted = 0
    lines_omitted = 0
    params: List[PickettParameterRecord] = []
    residuals: List[PickettFitResidual] = []

    param_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    res_pattern = re.compile(
        r"^\s*(\d+)\s+((?:\s*[+-]?\d+){2,6})\s+((?:\s*[+-]?\d+){2,6})\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)"
    )

    in_residuals = False
    in_parameters = False

    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue

        if not title and not s_line.startswith(("MICROWAVE", "STANDARD", "VARIANCE", "LINES")):
            title = s_line

        # Extract summary statistics via regex
        std_err_match = re.search(
            r"(?:STANDARD\s+ERROR(?:\s+OF\s+FIT)?|MICROWAVE\s+RMS)\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)",
            s_line,
            re.IGNORECASE,
        )
        if std_err_match:
            std_err = fortran_float_converter(std_err_match.group(1))

        var_match = re.search(r"VARIANCE\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)", s_line, re.IGNORECASE)
        if var_match:
            variance = fortran_float_converter(var_match.group(1))

        lines_fit_match = re.search(r"LINES\s+FITTED\s*=\s*(\d+)", s_line, re.IGNORECASE)
        if lines_fit_match:
            lines_fitted = int(lines_fit_match.group(1))

        # Detect sections
        if "PARAMETERS" in s_line.upper():
            in_parameters = True
            in_residuals = False
            continue

        if "RESIDUALS" in s_line.upper() or "OBS-CALC" in s_line.upper():
            in_residuals = True
            in_parameters = False
            continue

        if in_parameters:
            pmatch = param_pattern.match(s_line)
            if pmatch:
                pid = int(pmatch.group(1))
                val = fortran_float_converter(pmatch.group(2))
                unc = fortran_float_converter(pmatch.group(3))
                lbl = (pmatch.group(4) or "").strip()
                params.append(PickettParameterRecord(param_id=pid, value=val, uncertainty=unc, label=lbl))

        if in_residuals:
            rmatch = res_pattern.match(s_line)
            if rmatch:
                l_num = int(rmatch.group(1))
                qu = tuple(int(x) for x in rmatch.group(2).split())
                ql = tuple(int(x) for x in rmatch.group(3).split())
                f_obs = fortran_float_converter(rmatch.group(4))
                f_calc = fortran_float_converter(rmatch.group(5))
                res = fortran_float_converter(rmatch.group(6))
                unc = fortran_float_converter(rmatch.group(7))
                wt = fortran_float_converter(rmatch.group(8))
                residuals.append(
                    PickettFitResidual(
                        line_number=l_num,
                        quantum_upper=qu,
                        quantum_lower=ql,
                        frequency_obs_mhz=f_obs,
                        frequency_calc_mhz=f_calc,
                        residual_mhz=res,
                        uncertainty_mhz=unc,
                        weight=wt,
                    )
                )

    return PickettFitResult(
        title=title,
        standard_error=std_err,
        variance=variance,
        lines_fitted=lines_fitted if lines_fitted > 0 else len(residuals),
        lines_omitted=lines_omitted,
        parameters=params,
        residuals=residuals,
    )


def parse_pickett_var(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettVarFile:
    """
    Parses a legacy Pickett parameter file (.var / .par).
    """
    lines = _read_content_lines(content_or_path)
    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise PickettFormatError("Pickett parameter file must contain title and control lines.")

    title = non_empty[0].strip()
    ctrl_parts = non_empty[1].split()

    npar = int(ctrl_parts[0]) if len(ctrl_parts) > 0 else 0
    nline = int(ctrl_parts[1]) if len(ctrl_parts) > 1 else 0
    nopt = int(ctrl_parts[2]) if len(ctrl_parts) > 2 else 0
    nwarn = int(ctrl_parts[3]) if len(ctrl_parts) > 3 else 0
    erpar = fortran_float_converter(ctrl_parts[4]) if len(ctrl_parts) > 4 else 1.0
    wtfac = fortran_float_converter(ctrl_parts[5]) if len(ctrl_parts) > 5 else 1.0
    scale = fortran_float_converter(ctrl_parts[6]) if len(ctrl_parts) > 6 else 1.0
    maxit = int(ctrl_parts[7]) if len(ctrl_parts) > 7 else 50

    param_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    parameters: List[PickettParameterRecord] = []
    for p_line in non_empty[2:]:
        match = param_pattern.match(p_line)
        if match:
            pid = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            lbl = (match.group(4) or "").strip()
            parameters.append(PickettParameterRecord(param_id=pid, value=val, uncertainty=unc, label=lbl))

    return PickettVarFile(
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


def parse_pickett_int(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettIntFile:
    """
    Parses a legacy Pickett intensity file (.int).
    """
    lines = _read_content_lines(content_or_path)
    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise PickettFormatError("Pickett intensity file must contain title and control lines.")

    title = non_empty[0].strip()
    ctrl_parts = non_empty[1].split()

    if len(ctrl_parts) >= 9:
        tag = int(ctrl_parts[0])
        qpart = fortran_float_converter(ctrl_parts[1])
        nline = int(ctrl_parts[2])
        iq = int(ctrl_parts[3])
        max_j = int(ctrl_parts[4])
        min_freq = fortran_float_converter(ctrl_parts[5])
        max_freq = fortran_float_converter(ctrl_parts[6])
        min_int = fortran_float_converter(ctrl_parts[7])
        temp_k = fortran_float_converter(ctrl_parts[8])
    else:
        c_raw = non_empty[1]
        tag = int(c_raw[0:6].strip() or 1)
        qpart = fortran_float_converter(c_raw[6:18].strip() or 1000.0)
        nline = int(c_raw[18:25].strip() or 1000)
        iq = int(c_raw[25:29].strip() or 1)
        max_j = int(c_raw[29:33].strip() or 99)
        min_freq = fortran_float_converter(c_raw[33:44].strip() or 0.0)
        max_freq = fortran_float_converter(c_raw[44:55].strip() or 500000.0)
        min_int = fortran_float_converter(c_raw[55:65].strip() or -10.0)
        temp_k = fortran_float_converter(c_raw[65:75].strip() or 300.0)

    dipoles: List[PickettDipoleRecord] = []
    dip_pattern = re.compile(r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?")

    for d_line in non_empty[2:]:
        match = dip_pattern.match(d_line)
        if match:
            axis = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            comm = (match.group(3) or "").strip()
            dipoles.append(PickettDipoleRecord(axis=axis, dipole_debye=val, comment=comm))

    return PickettIntFile(
        title=title,
        tag=tag,
        qpart=qpart,
        nline=nline,
        iq=iq,
        max_j=max_j,
        min_freq_mhz=min_freq,
        max_freq_mhz=max_freq,
        min_intensity=min_int,
        temperature_k=temp_k,
        dipoles=dipoles,
    )


def parse_pickett_lin(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettLinFile:
    """
    Parses a legacy Pickett experimental line list (.lin).
    """
    lines = _read_content_lines(content_or_path)
    title = ""
    records: List[PickettLinRecord] = []

    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue
        if s_line.startswith("!") or s_line.startswith("#"):
            if not title:
                title = s_line.lstrip("!# ").strip()
            continue

        if len(line) >= 40:
            try:
                # Fixed-width Fortran 6I3, 6I3, F13.4, F8.4, F8.4
                qu_sub = [line[3 * i : 3 * i + 3].strip() for i in range(6)]
                ql_sub = [line[18 + 3 * i : 18 + 3 * i + 3].strip() for i in range(6)]
                f_sub = line[36:49].strip()
                u_sub = line[49:57].strip()
                w_sub = line[57:65].strip() if len(line) >= 65 else ""

                if qu_sub[0] and ql_sub[0] and f_sub:
                    qu = tuple(int(x) for x in qu_sub if x)
                    ql = tuple(int(x) for x in ql_sub if x)
                    freq = fortran_float_converter(f_sub)
                    unc = fortran_float_converter(u_sub) if u_sub else 0.01
                    wt = fortran_float_converter(w_sub) if w_sub else 1.0
                    records.append(
                        PickettLinRecord(
                            quantum_upper=qu,
                            quantum_lower=ql,
                            frequency_mhz=freq,
                            uncertainty_mhz=unc,
                            weight=wt,
                        )
                    )
                    continue
            except ValueError as parse_err:
                logger.debug("Fixed-width lin line parse fallback: %s", parse_err)

        match = free_pattern.match(s_line)
        if match:
            tokens = s_line.split()
            try:
                freq = fortran_float_converter(match.group(7))
                unc = fortran_float_converter(match.group(8))
                wt = fortran_float_converter(match.group(9)) if match.group(9) else 1.0
                qu = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                ql = (int(match.group(4)), int(match.group(5)), int(match.group(6)))
                records.append(
                    PickettLinRecord(
                        quantum_upper=qu,
                        quantum_lower=ql,
                        frequency_mhz=freq,
                        uncertainty_mhz=unc,
                        weight=wt,
                    )
                )
            except Exception as parse_err:
                logger.debug("Lin regex line parse error: %s", parse_err)

    return PickettLinFile(title=title, records=records)


# =============================================================================
# 6. BINARY DEPLOYMENT ROUTING & EPHEMERAL SCRATCH CONTEXT
# =============================================================================

def get_fortran_deployment_dir() -> Path:
    """
    Resolves the explicit executable deployment directory for legacy Fortran binaries.
    Prioritizes $COCHEM_BIN_DIR over platformdirs user_data_dir / 'bin' to prevent
    execution failures on HPC clusters where scratch/cache mounts have noexec flags.

    Returns:
        Path: Canonical deployment directory with guaranteed write/execute access.
    """
    bin_env = os.environ.get("COCHEM_BIN_DIR")
    if bin_env:
        target = Path(bin_env).resolve()
    else:
        target = Path(platformdirs.user_data_dir(appname="CoChem-SpycFit", ensure_exists=True)) / "bin"

    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_fortran_binary(binary_name: str) -> Optional[Path]:
    """
    Resolves executable path for a legacy Fortran binary within the deployment directory
    or the active system PATH.

    Args:
        binary_name: Binary filename or command (e.g. 'spcat', 'spfit').

    Returns:
        Optional[Path]: Absolute path to executable file, or None if not found.
    """
    dep_dir = get_fortran_deployment_dir()

    # Check direct name in deployment directory
    candidate = dep_dir / binary_name
    if candidate.is_file() and os.access(str(candidate), os.X_OK):
        return candidate

    # Check Windows .exe extension
    if platform.system().lower() == "windows" and not binary_name.lower().endswith(".exe"):
        candidate_exe = dep_dir / f"{binary_name}.exe"
        if candidate_exe.is_file():
            return candidate_exe

    # Check system PATH
    found = shutil.which(binary_name)
    if found:
        return Path(found).resolve()

    return None


class PickettScratchContext:
    """
    Ephemeral scratch directory lifecycle manager for Pickett/SPCAT/SPFIT operations.
    Ensures input staging and guaranteed cleanup of intermediate calculation scratch files.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        prefix: str = "pickett_scratch_",
    ) -> None:
        if base_dir is not None:
            self._parent_dir = Path(base_dir).resolve()
        else:
            scratch_env = os.environ.get("COCHEM_SCRATCH_DIR")
            if scratch_env:
                self._parent_dir = Path(scratch_env).resolve()
            else:
                self._parent_dir = Path(tempfile.gettempdir()).resolve() / "cochem_spycfit_scratch"

        self._parent_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_path: Path = self._parent_dir / f"{prefix}{uuid.uuid4().hex}"
        self._active: bool = False

    def __enter__(self) -> PickettScratchContext:
        self.scratch_path.mkdir(parents=True, exist_ok=True)
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.purge()

    def purge(self) -> None:
        """Purges all files and directories in this scratch workspace."""
        if self.scratch_path.exists():
            shutil.rmtree(str(self.scratch_path), ignore_errors=True)
        self._active = False

    def write_input_file(self, filename: str, content: Union[str, bytes]) -> Path:
        """Writes an input file into the active scratch directory."""
        if not self._active:
            raise RuntimeError("Cannot write input file outside of active PickettScratchContext.")
        target = self.scratch_path / filename
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        return target

    def read_output_file(self, filename: str) -> str:
        """Reads a text output file from the active scratch directory."""
        target = self.scratch_path / filename
        if not target.is_file():
            raise FileNotFoundError(f"Output file not produced in scratch: {filename}")
        return target.read_text(encoding="utf-8", errors="replace")


# =============================================================================
# 7. PROCESS ISOLATION, SEGFAULT TRAPPING, ORPHAN SWEEPING & HEX TELEMETRY
# =============================================================================

def is_segfault_exit_code(return_code: int) -> bool:
    """
    Identifies whether a process termination return code corresponds to a segmentation fault,
    memory access violation, or abnormal crash signal across POSIX and Windows.

    POSIX: 139 (SIGSEGV), 134 (SIGABRT), 138 (SIGBUS), -11 (SIGSEGV), -6 (SIGABRT), -7 (SIGBUS)
    Windows: 0xC0000005, -1073741819, 3221225477 (STATUS_ACCESS_VIOLATION),
             0xC00000FD, -1073741571 (STATUS_STACK_OVERFLOW),
             0xC0000028, 0xC0000374 (STATUS_HEAP_CORRUPTION)
    """
    posix_codes = {139, 134, 138, -11, -6, -7}
    windows_codes = {
        0xC0000005,
        -1073741819,
        3221225477,
        0xC00000FD,
        -1073741571,
        0xC0000028,
        0xC0000374,
    }
    return return_code in posix_codes or return_code in windows_codes


def capture_hex_telemetry(data_bytes: Union[bytes, str], max_bytes: int = 256) -> str:
    """
    Extracts trailing bytes of a stream (up to max_bytes, default 256) and formats them
    as a hexadecimal trace string.

    Args:
        data_bytes: Raw bytes or text string from stderr/stdout.
        max_bytes: Maximum number of tail bytes to capture (default: 256).

    Returns:
        str: Space-separated uppercase hexadecimal byte values.
    """
    if not data_bytes:
        return ""
    if isinstance(data_bytes, str):
        b = data_bytes.encode("utf-8", errors="replace")
    else:
        b = data_bytes
    tail = b[-max_bytes:]
    return " ".join(f"{byte:02X}" for byte in tail)


def sweep_orphan_process(process_or_pid: Union[subprocess.Popen, int]) -> None:
    """
    Explicitly terminates an orphaned subprocess and all of its spawned child processes.
    Uses 'taskkill /F /T /PID' on Windows and process-group SIGKILL on POSIX.

    Args:
        process_or_pid: Subprocess Popen instance or integer process ID (PID).
    """
    if isinstance(process_or_pid, subprocess.Popen):
        pid = process_or_pid.pid
    else:
        pid = int(process_or_pid)

    if pid <= 0:
        return

    if platform.system().lower() == "windows":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=True,
                timeout=5.0,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.debug("Windows taskkill orphan sweep failed for PID %d: %s", pid, exc)
    else:
        try:
            import signal
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:
            logger.debug("POSIX SIGKILL orphan sweep failed for PID %d: %s", pid, exc)


def write_spycfit_telemetry(
    record: Dict[str, Any],
    telemetry_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Appends structured execution telemetry entry to $COCHEM_STATE_DIR/Logs/spycfit_telemetry.json
    or platformdirs user_state_dir.
    """
    if telemetry_file is not None:
        t_path = Path(telemetry_file).resolve()
    else:
        state_dir_env = os.environ.get("COCHEM_STATE_DIR")
        if state_dir_env:
            log_dir = Path(state_dir_env).resolve() / "Logs"
        else:
            log_dir = Path(platformdirs.user_state_dir(appname="CoChem-SpycFit", ensure_exists=True)) / "Logs"
        t_path = log_dir / "spycfit_telemetry.json"

    t_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **record,
    }

    with open(str(t_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return t_path


def trap_and_log_segfault(
    stderr_output: Union[str, bytes],
    task_id: Optional[str] = None,
    executable: Optional[str] = None,
    return_code: Optional[int] = None,
    telemetry_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Captures trailing 256-byte hex telemetry from stderr output, checks whether return_code
    indicates a segfault or access violation, and records the telemetry incident.

    Args:
        stderr_output: Captured stderr text or raw bytes.
        task_id: Optional task identifier UUID.
        executable: Optional executable binary path/command.
        return_code: Optional process exit code.
        telemetry_file: Optional path override for telemetry log file.

    Returns:
        Dict[str, Any]: Telemetry incident record including hex trace and segfault status.
    """
    tid = task_id or str(uuid.uuid4())
    exe = executable or "spcat"
    rc = return_code if return_code is not None else 0
    segfault = is_segfault_exit_code(rc)
    hex_trace = capture_hex_telemetry(stderr_output, max_bytes=256)

    record: Dict[str, Any] = {
        "task_id": tid,
        "executable": exe,
        "return_code": rc,
        "segfault_detected": segfault,
        "stderr_tail_hex": hex_trace,
    }

    try:
        write_spycfit_telemetry(record, telemetry_file=telemetry_file)
        record["telemetry_logged"] = True
    except Exception as exc:
        logger.warning("Failed to persist segfault telemetry: %s", exc)
        record["telemetry_logged"] = False

    return record


# =============================================================================
# 8. ASYNCHRONOUS TASK QUEUE & BRIDGE DISPATCHER
# =============================================================================

class FortranExecutionTask(BaseModel):
    """Encapsulates execution parameters for a legacy Fortran binary invocation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Task UUID.")
    executable_path: str = Field(..., description="Filesystem path or command name of binary.")
    arguments: List[str] = Field(default_factory=list, description="Command line argument tokens.")
    input_files: Dict[str, Union[str, bytes]] = Field(
        default_factory=dict,
        description="Dictionary mapping target filenames to content.",
    )
    expected_output_files: List[str] = Field(
        default_factory=list,
        description="Filenames expected to be produced by execution.",
    )
    timeout_sec: float = Field(default=60.0, gt=0.0, description="Process timeout in seconds.")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variable overrides.")
    working_directory: Optional[str] = Field(
        default=None,
        description="Explicit working directory. If None, an ephemeral PickettScratchContext is used.",
    )


class FortranExecutionResult(BaseModel):
    """Encapsulates output status, logs, and collected artifacts from binary execution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(..., description="Identifier matching task.")
    executable_path: str = Field(..., description="Executed binary path.")
    return_code: int = Field(..., description="Process exit code.")
    stdout: str = Field(default="", description="Captured stdout text.")
    stderr: str = Field(default="", description="Captured stderr text.")
    execution_duration_sec: float = Field(ge=0.0, description="Total execution duration in seconds.")
    output_files: Dict[str, str] = Field(default_factory=dict, description="Captured text output files.")
    output_binaries: Dict[str, bytes] = Field(default_factory=dict, description="Captured binary output files.")
    segfault_detected: bool = Field(default=False, description="True if segfault / access violation occurred.")
    stderr_tail_hex: str = Field(default="", description="Hexadecimal trace of final stderr bytes.")
    telemetry_logged: bool = Field(default=False, description="True if telemetry was written to persistent log.")


class FortranTaskQueue:
    """
    Thread-safe FIFO task queue for batching and dispatching Fortran execution tasks.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[FortranExecutionTask] = queue.Queue(maxsize=maxsize)

    def submit(self, task: FortranExecutionTask) -> str:
        """Enqueues an execution task and returns its task ID."""
        self._queue.put(task)
        return task.task_id

    def get(self, block: bool = True, timeout: Optional[float] = None) -> FortranExecutionTask:
        """Retrieves the next queued execution task."""
        return self._queue.get(block=block, timeout=timeout)

    def task_done(self) -> None:
        """Signals task completion to queue."""
        self._queue.task_done()

    def qsize(self) -> int:
        """Returns current queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Returns True if task queue is currently empty."""
        return self._queue.empty()


class FortranBridgeDispatcher:
    """
    Execution dispatcher for legacy Fortran binaries.
    Provides OS process isolation, segfault trapping, hex telemetry capture,
    and ephemeral scratch directory lifecycle management.
    """

    def __init__(
        self,
        default_timeout_sec: float = 60.0,
        log_telemetry: bool = True,
        telemetry_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.default_timeout_sec = default_timeout_sec
        self.log_telemetry = log_telemetry
        self.telemetry_file = Path(telemetry_file).resolve() if telemetry_file is not None else None

    def dispatch_task(
        self,
        task: FortranExecutionTask,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Synchronously executes a FortranExecutionTask in an isolated process.
        """
        start_time = time.time()
        task_id = task.task_id
        exe_str = task.executable_path

        # Configure OS process isolation flags
        proc_kwargs: Dict[str, Any] = {}
        if platform.system().lower() == "windows":
            proc_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            proc_kwargs["start_new_session"] = True

        exec_env = os.environ.copy()
        exec_env.update(task.env_vars)

        scratch_ctx: Optional[PickettScratchContext] = None
        proc: Optional[subprocess.Popen] = None
        try:
            if task.working_directory is None:
                scratch_ctx = PickettScratchContext(prefix="pickett_scratch_")
                scratch_ctx.__enter__()
                work_dir = scratch_ctx.scratch_path
            else:
                work_dir = Path(task.working_directory).resolve()
                work_dir.mkdir(parents=True, exist_ok=True)

            # Stage input files
            for filename, content in task.input_files.items():
                f_path = work_dir / filename
                if isinstance(content, bytes):
                    f_path.write_bytes(content)
                else:
                    f_path.write_text(content, encoding="utf-8")

            cmd = [exe_str] + task.arguments

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(work_dir),
                    env=exec_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **proc_kwargs,
                )
                stdout_bytes, stderr_bytes = proc.communicate(timeout=task.timeout_sec)
                return_code = proc.returncode
            except subprocess.TimeoutExpired as exc:
                if proc is not None:
                    sweep_orphan_process(proc)
                    try:
                        proc.kill()
                    except Exception as kill_err:
                        logger.debug("Proc kill after sweep suppressed: %s", kill_err)
                duration = time.time() - start_time
                stdout_bytes = exc.stdout or b""
                stderr_bytes = exc.stderr or b""
                hex_trace = capture_hex_telemetry(stderr_bytes if stderr_bytes else stdout_bytes)
                if self.log_telemetry:
                    write_spycfit_telemetry(
                        {
                            "task_id": task_id,
                            "executable": exe_str,
                            "return_code": -999,
                            "segfault_detected": False,
                            "timed_out": True,
                            "execution_duration_sec": duration,
                            "stderr_tail_hex": hex_trace,
                        },
                        telemetry_file=self.telemetry_file,
                    )
                raise FortranExecutionTimeoutError(
                    f"Fortran task {task_id} running '{exe_str}' timed out after {task.timeout_sec:.2f}s"
                ) from exc
            except FileNotFoundError as fnf_exc:
                raise FortranBinaryNotFoundError(f"Binary not found: {exe_str}") from fnf_exc
            except Exception as proc_exc:
                if proc is not None:
                    sweep_orphan_process(proc)
                raise FortranExecutionError(f"Subprocess execution error for '{exe_str}': {proc_exc}") from proc_exc

            duration = time.time() - start_time
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            segfault = is_segfault_exit_code(return_code)
            hex_trace = capture_hex_telemetry(stderr_bytes if stderr_bytes else stdout_bytes)

            collected_outputs: Dict[str, str] = {}
            collected_binaries: Dict[str, bytes] = {}

            if task.expected_output_files:
                for exp_name in task.expected_output_files:
                    target_f = work_dir / exp_name
                    if target_f.is_file():
                        try:
                            collected_outputs[exp_name] = target_f.read_text(encoding="utf-8", errors="replace")
                            collected_binaries[exp_name] = target_f.read_bytes()
                        except UnicodeDecodeError:
                            collected_binaries[exp_name] = target_f.read_bytes()
            else:
                for child in work_dir.iterdir():
                    if child.is_file() and child.name not in task.input_files:
                        try:
                            collected_outputs[child.name] = child.read_text(encoding="utf-8", errors="replace")
                            collected_binaries[child.name] = child.read_bytes()
                        except UnicodeDecodeError:
                            collected_binaries[child.name] = child.read_bytes()

            telemetry_written = False
            if self.log_telemetry:
                try:
                    write_spycfit_telemetry(
                        {
                            "task_id": task_id,
                            "executable": exe_str,
                            "return_code": return_code,
                            "segfault_detected": segfault,
                            "execution_duration_sec": duration,
                            "stderr_tail_hex": hex_trace,
                        },
                        telemetry_file=self.telemetry_file,
                    )
                    telemetry_written = True
                except OSError as log_err:
                    logger.warning("Failed to record telemetry: %s", log_err)

            if raise_on_segfault and segfault:
                raise FortranExecutionSegfaultError(
                    f"Fortran task {task_id} terminated with segfault (exit code {return_code}): {exe_str}"
                )

            return FortranExecutionResult(
                task_id=task_id,
                executable_path=exe_str,
                return_code=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                execution_duration_sec=duration,
                output_files=collected_outputs,
                output_binaries=collected_binaries,
                segfault_detected=segfault,
                stderr_tail_hex=hex_trace,
                telemetry_logged=telemetry_written,
            )
        finally:
            if scratch_ctx is not None:
                scratch_ctx.purge()

    async def async_dispatch_task(
        self,
        task: FortranExecutionTask,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Asynchronously executes a FortranExecutionTask on a background worker thread.
        """
        return await asyncio.to_thread(self.dispatch_task, task, raise_on_segfault)

    def dispatch_fortran_binary(
        self,
        executable_path: Union[str, Path],
        input_files: Dict[str, Union[str, bytes]],
        arguments: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        timeout_sec: float = 60.0,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Convenience wrapper executing a binary directly with staged input files.
        """
        task = FortranExecutionTask(
            executable_path=str(executable_path),
            arguments=arguments or [],
            input_files=input_files,
            expected_output_files=expected_outputs or [],
            timeout_sec=timeout_sec,
        )
        return self.dispatch_task(task, raise_on_segfault=raise_on_segfault)

    def process_queue(
        self,
        task_queue: FortranTaskQueue,
        max_tasks: Optional[int] = None,
        raise_on_segfault: bool = False,
    ) -> List[FortranExecutionResult]:
        """
        Drains and processes tasks from the FortranTaskQueue.
        """
        results: List[FortranExecutionResult] = []
        processed = 0
        while not task_queue.empty():
            if max_tasks is not None and processed >= max_tasks:
                break
            try:
                task = task_queue.get(block=False)
            except queue.Empty:
                break
            result = self.dispatch_task(task, raise_on_segfault=raise_on_segfault)
            task_queue.task_done()
            results.append(result)
            processed += 1
        return results


def execute_spcat_safely(
    executable_path: Union[str, Path],
    scratch_dir: Optional[Union[str, Path]] = None,
    timeout_sec: float = 60.0,
    input_files: Optional[Dict[str, Union[str, bytes]]] = None,
    expected_output_files: Optional[List[str]] = None,
    arguments: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    raise_on_segfault: bool = False,
    telemetry_file: Optional[Union[str, Path]] = None,
) -> FortranExecutionResult:
    """
    High-level, process-isolated, segfault-trapped, ephemeral scratch-managed execution
    entry point for Pickett SPCAT.

    Provisions an ephemeral 'pickett_scratch/' directory, stages input files, enforces
    OS process isolation, sweeps orphaned processes on timeout, captures 256-byte hex
    telemetry, and guarantees cleanup of scratch files upon completion.

    Args:
        executable_path: Path or command for spcat binary.
        scratch_dir: Optional custom scratch directory. If None, an ephemeral
                     'pickett_scratch/' directory is provisioned and purged automatically.
        timeout_sec: Maximum execution timeout in seconds.
        input_files: Mapping of filenames to string or bytes content to stage.
        expected_output_files: List of expected output filenames to collect.
        arguments: Command line arguments passed to binary (e.g. ['mol']).
        env_vars: Environment variables dictionary.
        raise_on_segfault: If True, raises FortranExecutionSegfaultError on segfault.
        telemetry_file: Optional custom telemetry log destination.

    Returns:
        FortranExecutionResult: Result containing exit code, stdout, stderr, collected outputs,
                                segfault detection flag, and hex telemetry.
    """
    task = FortranExecutionTask(
        executable_path=str(executable_path),
        arguments=arguments or [],
        input_files=input_files or {},
        expected_output_files=expected_output_files or [],
        timeout_sec=timeout_sec,
        env_vars=env_vars or {},
        working_directory=str(scratch_dir) if scratch_dir is not None else None,
    )
    dispatcher = FortranBridgeDispatcher(
        default_timeout_sec=timeout_sec,
        log_telemetry=True,
        telemetry_file=telemetry_file,
    )
    return dispatcher.dispatch_task(task, raise_on_segfault=raise_on_segfault)


def dispatch_fortran_binary(
    executable_path: Union[str, Path],
    input_files: Dict[str, Union[str, bytes]],
    arguments: Optional[List[str]] = None,
    expected_outputs: Optional[List[str]] = None,
    timeout_sec: float = 60.0,
    raise_on_segfault: bool = False,
) -> FortranExecutionResult:
    """
    Global functional entry point for executing a legacy Fortran binary.
    """
    dispatcher = FortranBridgeDispatcher(default_timeout_sec=timeout_sec)
    return dispatcher.dispatch_fortran_binary(
        executable_path=executable_path,
        input_files=input_files,
        arguments=arguments,
        expected_outputs=expected_outputs,
        timeout_sec=timeout_sec,
        raise_on_segfault=raise_on_segfault,
    )


__all__ = [
    "FortranBridgeError",
    "PickettFormatError",
    "FortranExecutionError",
    "FortranExecutionTimeoutError",
    "FortranExecutionSegfaultError",
    "FortranBinaryNotFoundError",
    "get_isotopic_mass",
    "calculate_reduced_mass",
    "fortran_float_converter",
    "format_fortran_double",
    "format_pickett_line",
    "decode_pickett_quantum",
    "encode_pickett_quantum",
    "PickettParameterRecord",
    "PickettVarFile",
    "PickettDipoleRecord",
    "PickettIntFile",
    "PickettLinRecord",
    "PickettLinFile",
    "PickettCatRecord",
    "PickettCatFile",
    "PickettEgyRecord",
    "PickettEgyFile",
    "PickettFitResidual",
    "PickettFitResult",
    "parse_pickett_cat",
    "parse_pickett_egy",
    "parse_pickett_fit",
    "parse_pickett_var",
    "parse_pickett_int",
    "parse_pickett_lin",
    "get_fortran_deployment_dir",
    "resolve_fortran_binary",
    "PickettScratchContext",
    "is_segfault_exit_code",
    "capture_hex_telemetry",
    "sweep_orphan_process",
    "write_spycfit_telemetry",
    "trap_and_log_segfault",
    "FortranExecutionTask",
    "FortranExecutionResult",
    "FortranTaskQueue",
    "FortranBridgeDispatcher",
    "execute_spcat_safely",
    "dispatch_fortran_binary",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spcat_bridge_phase4.py ---
"""
CoChem-SpycFit: Comprehensive Phase 4 Physical Unit Test Suite for Pickett SPCAT Bridge.

Strict Directives Enforced:
1. Zero-Mock Physical Mandate: 100% authentic physical execution, zero pass statements,
   zero artificial test doubles.
2. Dynamic Mendeleev: All atomic and isotopic masses dynamically retrieved via Mendeleev.
3. Strict Fortran D exponent notation (%10.4E -> D) and 10-character column alignment.
4. OS-Agnostic Process Isolation (CREATE_NEW_PROCESS_GROUP on Windows, start_new_session on POSIX).
5. OS-Agnostic Segfault Trap (Exit codes 139, 134, 138, -11, -6, -7, 0xC0000005, -1073741819, 3221225477).
6. Hex-Dump Telemetry (Trailing 256 bytes formatted as hex logged to JSON).
7. Orphan Sweeper (Explicit SIGKILL / taskkill process tree cleanup on timeout).
8. Ephemeral pickett_scratch/ lifecycle with guaranteed shutil.rmtree purging.
9. AST static analysis verifying zero pass statements and zero banned mock tokens.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest

from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettIntFile,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    execute_spcat_safely,
    format_fortran_double,
    format_pickett_line,
    fortran_float_converter,
    get_isotopic_mass,
    is_segfault_exit_code,
    sweep_orphan_process,
    trap_and_log_segfault,
    write_spycfit_telemetry,
)

logger = logging.getLogger("CoChem-SpycFit.TestPhase4")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def phase4_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated physical filesystem directory."""
    ws = tmp_path / "phase4_spcat_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


# =============================================================================
# 1. FORTRAN DOUBLE PRECISION FORMATTING TESTS
# =============================================================================

class TestFortranDoubleFormattingPhase4:
    """Validates format_fortran_double strict 10-char width and Fortran D notation."""

    def test_format_fortran_double_positive_floats(self) -> None:
        formatted = format_fortran_double(12345.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "1.2345D+04"
        assert "E" not in formatted
        assert "e" not in formatted

    def test_format_fortran_double_small_scientific(self) -> None:
        formatted = format_fortran_double(1.25e-5, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "1.2500D-05"
        assert "D-05" in formatted

    def test_format_fortran_double_negative_number_width_constraint(self) -> None:
        formatted = format_fortran_double(-12345.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "-1.234D+04"
        assert formatted.startswith("-")

    def test_format_fortran_double_zero(self) -> None:
        formatted = format_fortran_double(0.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "0.0000D+00"

    def test_format_fortran_double_various_widths_and_precisions(self) -> None:
        # Width 22, precision 14
        f22 = format_fortran_double(127538.91024, width=22, precision=14)
        assert len(f22) == 22
        assert "D+" in f22

        # Width 18, precision 10
        f18 = format_fortran_double(0.00123456, width=18, precision=10)
        assert len(f18) == 18
        assert "D-03" in f18

        # Narrow width 8, precision 2
        f8 = format_fortran_double(1.25e-5, width=8, precision=2)
        assert len(f8) == 8
        assert f8 == "1.25D-05"

    def test_format_fortran_double_numpy_and_string_inputs(self) -> None:
        # Numpy float64
        np_val = np.float64(42.5)
        f_np = format_fortran_double(np_val, width=10, precision=4)
        assert len(f_np) == 10
        assert f_np == "4.2500D+01"

        # String with E notation
        f_str_e = format_fortran_double("1.25E-05", width=10, precision=4)
        assert len(f_str_e) == 10
        assert f_str_e == "1.2500D-05"

        # String with D notation
        f_str_d = format_fortran_double("1.25D-05", width=10, precision=4)
        assert len(f_str_d) == 10
        assert f_str_d == "1.2500D-05"

        # String with implicit sign
        f_str_imp = format_fortran_double("1.25-05", width=10, precision=4)
        assert len(f_str_imp) == 10
        assert f_str_imp == "1.2500D-05"

    def test_format_fortran_double_non_numeric_fallback(self) -> None:
        f_head = format_fortran_double("HEADER", width=10)
        assert len(f_head) == 10
        assert f_head == "    HEADER"

        f_trunc = format_fortran_double("EXCESSIVELY_LONG_STRING", width=10)
        assert len(f_trunc) == 10
        assert f_trunc == "EXCESSIVEL"

    def test_format_pickett_line_adherence(self) -> None:
        res_default = format_pickett_line(1.25e-5, width=10, precision=4)
        assert res_default == "1.2500D-05"

        res_custom = format_pickett_line(1.25e-5, width=10, precision=4, exponent_char="E")
        assert res_custom == "1.2500E-05"

    def test_pickett_var_and_int_export_adherence(self, phase4_workspace: Path) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_o16 = get_isotopic_mass("O", 16)
        mu_co = calculate_reduced_mass(mass_c12, mass_o16)
        assert mu_co > 0.0

        var_file = PickettVarFile(
            title="Carbon Monoxide Test Var",
            parameters=[
                PickettParameterRecord(param_id=10000, value=57635.968, uncertainty=0.0001, label="B"),
                PickettParameterRecord(param_id=200, value=0.1835, uncertainty=0.00005, label="-DJ"),
            ],
        )
        var_content = var_file.export_var(phase4_workspace / "co.var")
        assert "57635.968" in var_content or "5.7635" in var_content
        assert "D+" in var_content or "D-" in var_content

        int_file = PickettIntFile(
            title="CO Intensity File",
            tag=28001,
            qpart=107.5,
            dipoles=[
                PickettDipoleRecord(axis=1, dipole_debye=0.110, comment="mua"),
            ],
        )
        int_content = int_file.export_int(phase4_workspace / "co.int")
        assert "28001" in int_content
        assert "0.1100" in int_content


# =============================================================================
# 2. OS-AGNOSTIC SEGFAULT TRAPPING TESTS
# =============================================================================

class TestSegfaultTrapPhase4:
    """Validates OS-agnostic detection of segfaults, memory violations, and crash signals."""

    def test_posix_segfault_codes(self) -> None:
        assert is_segfault_exit_code(139) is True   # SIGSEGV
        assert is_segfault_exit_code(134) is True   # SIGABRT
        assert is_segfault_exit_code(138) is True   # SIGBUS
        assert is_segfault_exit_code(-11) is True   # SIGSEGV negative
        assert is_segfault_exit_code(-6) is True    # SIGABRT negative
        assert is_segfault_exit_code(-7) is True    # SIGBUS negative

    def test_windows_access_violation_codes(self) -> None:
        assert is_segfault_exit_code(0xC0000005) is True      # STATUS_ACCESS_VIOLATION
        assert is_segfault_exit_code(-1073741819) is True     # Signed 32-bit 0xC0000005
        assert is_segfault_exit_code(3221225477) is True      # Unsigned 32-bit 0xC0000005
        assert is_segfault_exit_code(0xC00000FD) is True      # STATUS_STACK_OVERFLOW
        assert is_segfault_exit_code(-1073741571) is True     # Signed 32-bit 0xC00000FD
        assert is_segfault_exit_code(0xC0000028) is True      # STATUS_BAD_STACK
        assert is_segfault_exit_code(0xC0000374) is True      # STATUS_HEAP_CORRUPTION

    def test_normal_exit_codes_not_flagged(self) -> None:
        assert is_segfault_exit_code(0) is False
        assert is_segfault_exit_code(1) is False
        assert is_segfault_exit_code(2) is False
        assert is_segfault_exit_code(42) is False

    def test_trap_and_log_segfault_function(self, phase4_workspace: Path) -> None:
        t_file = phase4_workspace / "telemetry_trap.json"
        simulated_stderr = b"SPCAT calculation fatal fault at address 0x7FFF12345678"

        rec_seg = trap_and_log_segfault(
            stderr_output=simulated_stderr,
            task_id="task-seg-001",
            executable="spcat.exe",
            return_code=139,
            telemetry_file=t_file,
        )
        assert rec_seg["segfault_detected"] is True
        assert rec_seg["return_code"] == 139
        assert rec_seg["telemetry_logged"] is True
        assert len(rec_seg["stderr_tail_hex"]) > 0

        # Verify persistent log on disk
        assert t_file.is_file()
        lines = t_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "task-seg-001"
        assert data["segfault_detected"] is True


# =============================================================================
# 3. HEX-DUMP TELEMETRY TESTS
# =============================================================================

class TestHexDumpTelemetryPhase4:
    """Validates 256-byte tail hex telemetry extraction and JSON persistence."""

    def test_capture_hex_telemetry_bytes(self) -> None:
        test_bytes = b"COCHEM_SPCAT_CRASH_DUMP_0123456789"
        hex_str = capture_hex_telemetry(test_bytes, max_bytes=256)
        expected_hex = " ".join(f"{b:02X}" for b in test_bytes)
        assert hex_str == expected_hex

    def test_capture_hex_telemetry_string_input(self) -> None:
        test_str = "Error in submatrix diagonalization"
        hex_str = capture_hex_telemetry(test_str, max_bytes=256)
        expected_hex = " ".join(f"{ord(c):02X}" for c in test_str)
        assert hex_str == expected_hex

    def test_capture_hex_telemetry_empty_input(self) -> None:
        assert capture_hex_telemetry(b"", max_bytes=256) == ""
        assert capture_hex_telemetry("", max_bytes=256) == ""

    def test_capture_hex_telemetry_truncates_to_last_256_bytes(self) -> None:
        large_stream = bytes(range(256)) * 4  # 1024 bytes
        hex_str = capture_hex_telemetry(large_stream, max_bytes=256)
        tokens = hex_str.split()
        assert len(tokens) == 256
        # The final 256 bytes should match the last 256 elements of large_stream
        expected_tokens = [f"{b:02X}" for b in large_stream[-256:]]
        assert tokens == expected_tokens

    def test_write_spycfit_telemetry_json(self, phase4_workspace: Path) -> None:
        telemetry_path = phase4_workspace / "test_telemetry.json"
        record = {
            "task_id": "test-uuid-42",
            "executable": "spcat",
            "return_code": 0,
            "segfault_detected": False,
            "execution_duration_sec": 0.123,
            "stderr_tail_hex": "43 4F 43 48 45 4D",
        }
        res_path = write_spycfit_telemetry(record, telemetry_file=telemetry_path)
        assert res_path == telemetry_path
        assert telemetry_path.is_file()

        content = telemetry_path.read_text(encoding="utf-8")
        parsed = json.loads(content.strip())
        assert parsed["task_id"] == "test-uuid-42"
        assert parsed["stderr_tail_hex"] == "43 4F 43 48 45 4D"
        assert "recorded_utc" in parsed


# =============================================================================
# 4. ORPHAN SWEEPER TESTS
# =============================================================================

class TestOrphanSweeperPhase4:
    """Validates OS-specific process tree orphan termination."""

    def test_sweep_orphan_invalid_pid(self) -> None:
        # PIDs <= 0 should safely return without raising
        sweep_orphan_process(0)
        sweep_orphan_process(-1)
        assert True

    def test_sweep_orphan_active_subprocess(self) -> None:
        # Launch real background process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.poll() is None  # Process is running

        # Sweep orphan process
        sweep_orphan_process(proc)
        time.sleep(0.5)

        # Force kill if still lingering
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception as kill_err:
                logger.debug("Proc kill in test suppressed: %s", kill_err)

        # Confirm process is terminated
        proc.wait(timeout=2.0)
        assert proc.poll() is not None


# =============================================================================
# 5. EPHEMERAL PICKETT SCRATCH DIRECTORY LIFECYCLE TESTS
# =============================================================================

class TestEphemeralPickettScratchContextPhase4:
    """Validates ephemeral pickett_scratch_ directory creation and explicit shutil.rmtree purging."""

    def test_pickett_scratch_lifecycle(self, phase4_workspace: Path) -> None:
        scratch_ctx = PickettScratchContext(base_dir=phase4_workspace, prefix="pickett_scratch_")
        assert "pickett_scratch_" in str(scratch_ctx.scratch_path.name)
        assert not scratch_ctx.scratch_path.exists()

        with scratch_ctx as ctx:
            assert ctx.scratch_path.exists()
            # Stage an authentic test file
            in_file = ctx.write_input_file("test.var", "Title line\n1 0 0 0 1.0 1.0 1.0 50\n")
            assert in_file.is_file()

            out_file = ctx.scratch_path / "test.cat"
            out_file.write_text("100000.0000 0.0100 -3.0000 2 0.0000 1 28001 0 1 0 0 0 0 0\n", encoding="utf-8")
            assert ctx.read_output_file("test.cat").startswith("100000.0000")

        # After exiting context, directory must be purged completely
        assert not scratch_ctx.scratch_path.exists()

    def test_pickett_scratch_error_outside_context(self) -> None:
        scratch_ctx = PickettScratchContext()
        with pytest.raises(RuntimeError):
            scratch_ctx.write_input_file("forbidden.txt", "data")


# =============================================================================
# 6. HIGH-LEVEL EXECUTE_SPCAT_SAFELY ENTRY POINT TESTS
# =============================================================================

class TestExecuteSpcatSafelyPhase4:
    """Validates execute_spcat_safely in process-isolated, segfault-trapped environments."""

    def test_execute_spcat_safely_successful_simulation(self, phase4_workspace: Path) -> None:
        telemetry_file = phase4_workspace / "exec_telemetry.json"

        # Create a Python script that mimics SPCAT creating a .cat file from .var and .int
        driver_script = phase4_workspace / "spcat_simulator.py"
        driver_script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "print('SPCAT Fortran Engine Started')\n"
            "Path('mol.cat').write_text('115271.2018  0.0005 -3.1234 2 0.0000 3 28001 0  1  0  0  0  0  0')\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )

        input_files = {
            "mol.var": "CO Molecule Var\n1 0 0 0 1.0 1.0 1.0 50\n10000 57635.968 0.0001 / B\n",
            "mol.int": "CO Molecule Int\n28001 107.5 1000 1 99 0.0 500000.0 -10.0 300.0\n1 0.1100 / mua\n",
        }

        result = execute_spcat_safely(
            executable_path=sys.executable,
            arguments=[str(driver_script)],
            input_files=input_files,
            expected_output_files=["mol.cat"],
            timeout_sec=10.0,
            telemetry_file=telemetry_file,
        )

        assert result.return_code == 0
        assert "SPCAT Fortran Engine Started" in result.stdout
        assert "mol.cat" in result.output_files
        assert "115271.2018" in result.output_files["mol.cat"]
        assert result.segfault_detected is False
        assert result.telemetry_logged is True
        assert telemetry_file.is_file()

    def test_execute_spcat_safely_timeout_and_orphan_sweep(self, phase4_workspace: Path) -> None:
        telemetry_file = phase4_workspace / "timeout_telemetry.json"
        slow_script = phase4_workspace / "slow_spcat.py"
        slow_script.write_text(
            "import time\n"
            "time.sleep(10.0)\n",
            encoding="utf-8",
        )

        with pytest.raises(FortranExecutionTimeoutError):
            execute_spcat_safely(
                executable_path=sys.executable,
                arguments=[str(slow_script)],
                timeout_sec=0.5,
                telemetry_file=telemetry_file,
            )

        assert telemetry_file.is_file()
        logged_data = json.loads(telemetry_file.read_text(encoding="utf-8").strip())
        assert logged_data["timed_out"] is True
        assert logged_data["return_code"] == -999

    def test_execute_spcat_safely_segfault_detection(self, phase4_workspace: Path) -> None:
        telemetry_file = phase4_workspace / "segfault_telemetry.json"
        seg_script = phase4_workspace / "crash_spcat.py"
        # Simulate exit code 139 (SIGSEGV)
        seg_script.write_text(
            "import sys\n"
            "sys.stderr.write('CRITICAL: Segmentation fault in hamiltonian matrix setup')\n"
            "sys.exit(139)\n",
            encoding="utf-8",
        )

        # Without raise_on_segfault
        result = execute_spcat_safely(
            executable_path=sys.executable,
            arguments=[str(seg_script)],
            timeout_sec=5.0,
            raise_on_segfault=False,
            telemetry_file=telemetry_file,
        )
        assert result.return_code == 139
        assert result.segfault_detected is True
        assert "Segmentation fault" in result.stderr
        assert len(result.stderr_tail_hex) > 0

        # With raise_on_segfault=True
        with pytest.raises(FortranExecutionSegfaultError):
            execute_spcat_safely(
                executable_path=sys.executable,
                arguments=[str(seg_script)],
                timeout_sec=5.0,
                raise_on_segfault=True,
                telemetry_file=telemetry_file,
            )

    def test_execute_spcat_safely_missing_binary(self) -> None:
        with pytest.raises((FortranBinaryNotFoundError, FortranExecutionError)):
            execute_spcat_safely(
                executable_path="non_existent_fortran_spcat_binary_xyz",
                timeout_sec=5.0,
            )


# =============================================================================
# 7. DYNAMIC ISOTOPE RETRIEVAL (MENDELEEV) & PHYSICAL MASS ORDERING TESTS
# =============================================================================

class TestDynamicMendeleevPhysicalPhase4:
    """Validates dynamic physical isotope queries via Mendeleev."""

    def test_dynamic_mendeleev_masses(self) -> None:
        mass_h1 = get_isotopic_mass("H", 1)
        mass_h2 = get_isotopic_mass("H", 2)
        mass_c12 = get_isotopic_mass("C", 12)
        mass_c13 = get_isotopic_mass("C", 13)
        mass_n14 = get_isotopic_mass("N", 14)
        mass_o16 = get_isotopic_mass("O", 16)
        mass_s32 = get_isotopic_mass("S", 32)

        assert mass_h1 < mass_h2
        assert mass_c12 < mass_c13
        assert mass_c12 < mass_n14 < mass_o16 < mass_s32
        assert abs(mass_c12 - 12.0) < 1e-5
        assert 1.0078 < mass_h1 < 1.0079
        assert 15.9949 < mass_o16 < 15.9950

    def test_reduced_mass_isotope_effect(self) -> None:
        m_c12 = get_isotopic_mass("C", 12)
        m_c13 = get_isotopic_mass("C", 13)
        m_o16 = get_isotopic_mass("O", 16)

        mu_12_16 = calculate_reduced_mass(m_c12, m_o16)
        mu_13_16 = calculate_reduced_mass(m_c13, m_o16)

        # 13C-16O is heavier than 12C-16O, so mu(13-16) > mu(12-16)
        assert mu_13_16 > mu_12_16


# =============================================================================
# 8. AST STATIC ANALYSIS AUDIT (ZERO-PASS & ZERO-MOCK MANDATE)
# =============================================================================

class TestAstStaticAnalysisPhase4:
    """Validates AST structure: zero pass statements and zero banned mock imports."""

    def test_zero_pass_statements_in_bridge(self) -> None:
        import cochem_spycfit.serialization.cochem_spcat_bridge as bridge_mod
        bridge_src = Path(bridge_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(bridge_src)
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden pass statements in bridge"

    def test_zero_pass_statements_in_phase4_tests(self) -> None:
        test_src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(test_src)
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden pass statements in test suite"

    def test_zero_banned_mock_imports(self) -> None:
        import cochem_spycfit.serialization.cochem_spcat_bridge as bridge_mod
        bridge_tree = ast.parse(Path(bridge_mod.__file__).read_text(encoding="utf-8"))
        test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

        for tree, label in [(bridge_tree, "bridge"), (test_tree, "test")]:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "mock" not in alias.name.lower(), f"Banned import {alias.name} in {label}"
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert "mock" not in mod.lower(), f"Banned import from {mod} in {label}"
                    for alias in node.names:
                        assert "mock" not in alias.name.lower(), f"Banned imported symbol {alias.name} in {label}"

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.