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
    the exponent character ('E' -> 'D'). Enforces exact field width by right-aligning
    and precision adjusting or truncating on overflow.

    Args:
        value: Numeric value or string representation.
        width: Exact character width of the output field (default: 10).
        precision: Digits after decimal point in scientific notation (default: 4).
        exponent_char: Exponent indicator character (default: 'D').

    Returns:
        str: Fixed-width formatted string guaranteed to have len == width.
    """
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        s_val = str(value)
        if len(s_val) > width:
            return s_val[:width]
        return s_val.rjust(width)

    # Format using scientific notation
    formatted = f"{val_float:.{precision}E}"
    formatted = formatted.replace("E", exponent_char).replace("e", exponent_char)

    if len(formatted) > width:
        # Dynamically adjust precision if exponent or sign widened the string
        excess = len(formatted) - width
        adj_precision = max(0, precision - excess)
        formatted = f"{val_float:.{adj_precision}E}".replace("E", exponent_char).replace("e", exponent_char)
        if len(formatted) > width:
            formatted = formatted[:width]

    return formatted.rjust(width)


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
        prefix: str = "spcat_scratch_",
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
# 7. PROCESS ISOLATION, SEGFAULT TRAPPING & HEX-DUMP TELEMETRY
# =============================================================================

def is_segfault_exit_code(return_code: int) -> bool:
    """
    Identifies whether a process termination return code corresponds to a segmentation fault
    or memory access violation.

    POSIX: 139 (SIGSEGV), 134 (SIGABRT), -11, -6
    Windows: 0xC0000005, -1073741819, 3221225477 (STATUS_ACCESS_VIOLATION)
    """
    posix_codes = {139, 134, -11, -6}
    windows_codes = {0xC0000005, -1073741819, 3221225477, -1073741571}
    return return_code in posix_codes or return_code in windows_codes


def capture_hex_telemetry(data_bytes: bytes, max_bytes: int = 256) -> str:
    """
    Extracts the tail bytes of a stream and formats them as a hex trace.
    """
    if not data_bytes:
        return ""
    tail = data_bytes[-max_bytes:]
    return " ".join(f"{b:02X}" for b in tail)


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
        try:
            if task.working_directory is None:
                scratch_ctx = PickettScratchContext()
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
                proc = subprocess.run(
                    cmd,
                    cwd=str(work_dir),
                    env=exec_env,
                    capture_output=True,
                    timeout=task.timeout_sec,
                    **proc_kwargs,
                )
                return_code = proc.returncode
                stdout_bytes = proc.stdout
                stderr_bytes = proc.stderr
            except subprocess.TimeoutExpired as exc:
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
