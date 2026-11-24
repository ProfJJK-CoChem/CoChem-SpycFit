"""Live GruLabRotSpec/SURF output parser, reconciler, and Doppler doublet peak picker.

Implements Pillar 3 (`SPYC-SURF-03`) of the CoChem-SpycFit Primary SURF
Integration Plan, grounded verbatim in upstream ``GruLabRotSpec/SURF`` at
commit ``0486b877d2a38f5bbe50be3cb333d34e27c7c22d``:

- ``spectrometer.py:695-745``: immutable per-step ``CavityFiles/<freq>.csv``
  with sparse row-0 metadata (``Center Freq``, ``Cavity Position``) and exact
  inclusive bandwidth slicing ``[center - step_size/2, center + step_size/2]``
  for both ``StepDirection.Up`` and ``StepDirection.Down``.
- ``spectrometer.py:747-758``: ``finalize_csv()`` top-packed metadata
  compaction reconciliation against stitched ``CavityFiles/*.csv`` to FP64
  tolerance, flagging missing steps and Zaber cavity-tracking rail drop-outs
  (``0.0`` mm or ``50.0`` mm).
- ``spectrometer.py:581-595``: cavity-search ``CavityRuns/<freq>MHz.csv`` +
  ``search_settings.toml`` lookup extraction.
- ``gui/analysis_panel.py:228-235``: coaxial Fabry-Perot Doppler doublet
  detection using carrier-gas beam velocity model ``delta_nu_d = 2*nu*v/c``,
  emitting strictly ``AssignmentStatus.PROPOSED`` candidates with SHA-256
  step-file provenance and never silently collapsing unresolved singlets/blends.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.signal import find_peaks

from cochem_spycfit.instruments.surf_schemas import (
    SurfCavitySearchSettings,
    SurfConfig,
    SurfFrequencyScanSettings,
)

# =============================================================================
# 1. CONSTANTS & PHYSICAL PARAMETERS
# =============================================================================

SURF_SPECTRUM_COLUMNS: Tuple[str, str, str, str] = (
    "Frequency (MHz)",
    "Intensity",
    "Center Freq",
    "Cavity Position",
)

SURF_CAVITY_RUN_COLUMNS: Tuple[str, str, str, str, str] = (
    "Zaber Position (mm)",
    "Intensity (V)",
    "Frequency (MHz)",
    "Peaks",
    "Intensity",
)

ZABER_POS_MIN_MM: float = 0.0
ZABER_POS_MAX_MM: float = 50.0
DEFAULT_RAIL_TOLERANCE_MM: float = 1e-6

SPEED_OF_LIGHT_M_S: float = 299_792_458.0

# Terminal supersonic expansion beam velocities (m/s) near 295 K for standard
# coaxial Fabry-Perot FTMW carrier gases.
CARRIER_GAS_BEAM_VELOCITIES_M_S: Dict[str, float] = {
    "Ne": 780.0,
    "Ar": 555.0,
    "He": 1750.0,
    "N2": 785.0,
}

_GAS_ALIAS_MAP: Dict[str, str] = {
    "ne": "Ne",
    "neon": "Ne",
    "ar": "Ar",
    "argon": "Ar",
    "he": "He",
    "helium": "He",
    "n2": "N2",
    "nitrogen": "N2",
}

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CAVITY_RUN_FREQ_RE = re.compile(r"^(?P<freq>[+-]?\d+(?:\.\d+)?)MHz$", re.IGNORECASE)


# =============================================================================
# 2. EXCEPTIONS & ENUMS
# =============================================================================


class SurfIngestError(ValueError):
    """Base exception for SURF output parsing, validation, and peak picking."""


class SurfCsvSchemaError(SurfIngestError):
    """Raised when a SURF CSV violates expected columns, types, or finiteness."""


class SurfBoundsError(SurfIngestError):
    """Raised when physical SURF hardware bounds (e.g. Zaber 0-50 mm) are violated."""


class StepDirection(str, Enum):
    """Mirrors ``StepDirection`` in ``SURF/src/spectrometer_gui/spectrometer.py:38-40``."""

    Down = "Down"
    Up = "Up"

    @classmethod
    def from_bounds(cls, start_freq_mhz: float, end_freq_mhz: float) -> StepDirection:
        """Infer scan step direction from ``start_freq`` and ``end_freq``."""
        if not (math.isfinite(start_freq_mhz) and math.isfinite(end_freq_mhz)):
            raise SurfBoundsError("start_freq_mhz and end_freq_mhz must be finite.")
        if start_freq_mhz < end_freq_mhz:
            return cls.Up
        if start_freq_mhz > end_freq_mhz:
            return cls.Down
        raise SurfBoundsError("start_freq_mhz and end_freq_mhz must not be equal.")


class SurfFeatureKind(str, Enum):
    """Classification of a candidate feature detected by the SURF peak picker."""

    DOPPLER_DOUBLET = "doppler_doublet"
    SINGLET = "singlet"
    UNRESOLVED_BLEND = "unresolved_blend"


# =============================================================================
# 3. INTERNAL VALIDATION & HASHING HELPERS
# =============================================================================


def compute_bytes_sha256(raw: bytes) -> str:
    """Return lowercase 64-character hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(raw).hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute exact SHA-256 digest of a file on disk using chunked streaming."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _decode_utf8_no_bom(raw: bytes, *, source_label: str) -> str:
    """Decode UTF-8 bytes while stripping a leading BOM only if needed for CSV resilience."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SurfCsvSchemaError(f"SURF CSV {source_label} is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    return text


def _parse_finite_fp64(cell: str, *, field_name: str, row_number: int) -> float:
    """Parse a CSV scalar into a strictly finite IEEE-754 float64."""
    stripped = cell.strip()
    if not stripped:
        raise SurfCsvSchemaError(f"Row {row_number}: empty value in mandatory numeric column '{field_name}'.")
    try:
        value = float(stripped)
    except ValueError as exc:
        raise SurfCsvSchemaError(
            f"Row {row_number}: non-numeric value {cell!r} in column '{field_name}'."
        ) from exc
    if not math.isfinite(value):
        raise SurfCsvSchemaError(
            f"Row {row_number}: non-finite value {cell!r} in column '{field_name}'."
        )
    return float(np.float64(value))


def _validate_zaber_position(
    cavity_position_mm: float,
    *,
    row_number: int,
    rail_tolerance_mm: float = DEFAULT_RAIL_TOLERANCE_MM,
) -> bool:
    """Validate ``0.0 <= cavity_position_mm <= 50.0`` and return True if pinned at a rail."""
    if not math.isfinite(cavity_position_mm):
        raise SurfBoundsError(f"Row {row_number}: cavity_position_mm must be finite, got {cavity_position_mm!r}.")
    if cavity_position_mm < ZABER_POS_MIN_MM or cavity_position_mm > ZABER_POS_MAX_MM:
        raise SurfBoundsError(
            f"Row {row_number}: cavity_position_mm={cavity_position_mm} mm is outside "
            f"valid Zaber travel range [{ZABER_POS_MIN_MM}, {ZABER_POS_MAX_MM}] mm."
        )
    return (
        cavity_position_mm <= ZABER_POS_MIN_MM + rail_tolerance_mm
        or cavity_position_mm >= ZABER_POS_MAX_MM - rail_tolerance_mm
    )


def _resolve_proposed_assignment_status() -> Any:
    """Lazily resolve ``AssignmentStatus.PROPOSED`` without circular top-level UI imports."""
    from cochem_spycfit.interfaces.cochem_spycfit_ui import AssignmentStatus

    return AssignmentStatus.PROPOSED


def _validate_strictly_proposed_status(value: Any) -> Any:
    """Enforce that every SURF peak candidate is strictly ``AssignmentStatus.PROPOSED``."""
    from cochem_spycfit.interfaces.cochem_spycfit_ui import AssignmentStatus

    if value is AssignmentStatus.PROPOSED:
        return AssignmentStatus.PROPOSED
    if isinstance(value, str) and value.strip().lower() == AssignmentStatus.PROPOSED.value:
        return AssignmentStatus.PROPOSED
    raise ValueError(
        "SURF peak candidates must strictly carry AssignmentStatus.PROPOSED; "
        f"auto-approval or status {value!r} is forbidden."
    )


def normalize_carrier_gas(gas_name: str) -> str:
    """Normalize a SURF ``experiment.gas_name`` string to canonical symbol (``Ne``, ``Ar``, ``He``, ``N2``)."""
    cleaned = gas_name.strip()
    if not cleaned:
        return ""
    canonical = _GAS_ALIAS_MAP.get(cleaned.lower())
    if canonical is None:
        supported = ", ".join(sorted(CARRIER_GAS_BEAM_VELOCITIES_M_S.keys()))
        raise SurfIngestError(
            f"Unsupported carrier gas {gas_name!r}; expected one of ({supported}) or supply beam_velocity_m_s."
        )
    return canonical


def resolve_beam_velocity_m_s(
    gas_name: str = "Ne",
    beam_velocity_m_s: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    """Resolve canonical carrier-gas label and coaxial beam velocity in m/s."""
    if beam_velocity_m_s is not None:
        if not math.isfinite(beam_velocity_m_s) or beam_velocity_m_s <= 0.0:
            raise SurfIngestError(f"beam_velocity_m_s must be finite and > 0, got {beam_velocity_m_s!r}.")
        cleaned = gas_name.strip()
        canonical = _GAS_ALIAS_MAP.get(cleaned.lower(), cleaned or "custom")
        return canonical, float(beam_velocity_m_s)

    canonical = normalize_carrier_gas(gas_name)
    if not canonical:
        return "", None
    return canonical, CARRIER_GAS_BEAM_VELOCITIES_M_S[canonical]


def expected_doppler_splitting_mhz(
    freq_mhz: float,
    gas_name: str = "Ne",
    *,
    beam_velocity_m_s: Optional[float] = None,
) -> float:
    """Compute coaxial Fabry-Perot Doppler doublet splitting ``2 * nu * (v_beam / c)`` in MHz."""
    if not math.isfinite(freq_mhz) or freq_mhz <= 0.0:
        raise SurfIngestError(f"freq_mhz must be finite and > 0, got {freq_mhz!r}.")
    _, v_beam = resolve_beam_velocity_m_s(gas_name, beam_velocity_m_s=beam_velocity_m_s)
    if v_beam is None:
        raise SurfIngestError("Cannot compute Doppler splitting without a carrier gas or beam_velocity_m_s.")
    return float(2.0 * np.float64(freq_mhz) * (np.float64(v_beam) / np.float64(SPEED_OF_LIGHT_M_S)))


def slice_surf_step_window(
    full_window_freq_mhz: np.ndarray,
    full_window_intensity: np.ndarray,
    *,
    center_freq_mhz: float,
    step_size_mhz: float,
    step_direction: StepDirection = StepDirection.Up,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice ``[center_freq - step_size/2, center_freq + step_size/2]`` matching ``SURF/spectrometer.py:705-730``."""
    if not math.isfinite(step_size_mhz) or step_size_mhz <= 0.0:
        raise SurfBoundsError(f"step_size_mhz must be finite and > 0, got {step_size_mhz!r}.")
    if not math.isfinite(center_freq_mhz):
        raise SurfBoundsError(f"center_freq_mhz must be finite, got {center_freq_mhz!r}.")

    freqs = np.asarray(full_window_freq_mhz, dtype=np.float64)
    ints = np.asarray(full_window_intensity, dtype=np.float64)
    if freqs.ndim != 1 or ints.ndim != 1 or freqs.shape != ints.shape:
        raise SurfCsvSchemaError("Frequency and intensity arrays must be 1D with identical length.")

    half_step = np.float64(step_size_mhz) / np.float64(2.0)
    lower_bound = np.float64(center_freq_mhz) - half_step
    upper_bound = np.float64(center_freq_mhz) + half_step

    mask = (freqs >= lower_bound) & (freqs <= upper_bound)
    sliced_freq = np.ascontiguousarray(freqs[mask], dtype=np.float64)
    sliced_int = np.ascontiguousarray(ints[mask], dtype=np.float64)

    if step_direction == StepDirection.Down:
        sliced_freq = np.ascontiguousarray(sliced_freq[::-1], dtype=np.float64)
        sliced_int = np.ascontiguousarray(sliced_int[::-1], dtype=np.float64)

    return sliced_freq, sliced_int


# =============================================================================
# 4. STRUCTURED PYDANTIC DATA MODELS
# =============================================================================


class SurfStepData(BaseModel):
    """Parsed immutable per-step ``CavityFiles/<total_frequency>.csv`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    file_path: Optional[Path] = None
    file_sha256: str = Field(min_length=64, max_length=64)
    center_freq_mhz: float
    cavity_position_mm: float = Field(ge=ZABER_POS_MIN_MM, le=ZABER_POS_MAX_MM)
    cavity_dropout: bool = False
    full_window_freq_mhz: np.ndarray
    full_window_intensity: np.ndarray
    stitched_freq_mhz: np.ndarray
    stitched_intensity: np.ndarray
    step_size_mhz: Optional[float] = None
    step_direction: StepDirection = StepDirection.Up
    row_count: int = Field(ge=1)

    @field_validator("file_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_HEX_RE.match(value):
            raise ValueError("file_sha256 must be a 64-character lowercase hex digest.")
        return value


class SurfFinalizedCsvData(BaseModel):
    """Parsed multi-step or finalized compacted ``<filename>.csv`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    file_path: Optional[Path] = None
    file_sha256: str = Field(min_length=64, max_length=64)
    freq_mhz: np.ndarray
    intensity: np.ndarray
    center_freqs_mhz: Tuple[float, ...]
    cavity_positions_mm: Tuple[float, ...]
    metadata_row_indices: Tuple[int, ...]
    is_compacted: bool
    cavity_dropout_steps_mhz: Tuple[float, ...]
    row_count: int = Field(ge=1)

    @field_validator("file_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_HEX_RE.match(value):
            raise ValueError("file_sha256 must be a 64-character lowercase hex digest.")
        return value


class SurfReconciliationReport(BaseModel):
    """Reconciliation diagnostics comparing stitched ``CavityFiles/*.csv`` vs finalized ``<filename>.csv``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reconciled: bool
    spectrum_matches: bool
    has_cavity_dropout: bool
    finalized_csv_path: Path
    finalized_csv_sha256: str = Field(min_length=64, max_length=64)
    step_file_count: int = Field(ge=0)
    finalized_metadata_count: int = Field(ge=0)
    finalized_is_compacted: bool
    row_count_match: bool
    stitched_row_count: int = Field(ge=0)
    finalized_row_count: int = Field(ge=0)
    max_freq_abs_diff_mhz: float = Field(ge=0.0)
    max_intensity_abs_diff: float = Field(ge=0.0)
    missing_steps_mhz: Tuple[float, ...] = ()
    extra_steps_mhz: Tuple[float, ...] = ()
    cavity_dropout_steps_mhz: Tuple[float, ...] = ()
    discrepancies: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    config_toml_sha256: Optional[str] = None
    scan_settings_toml_sha256: Optional[str] = None
    step_file_sha256s: Dict[str, str] = Field(default_factory=dict)


class SurfCavityPeak(BaseModel):
    """Single resonant cavity peak recorded in ``CavityRuns/<freq>MHz.csv``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zaber_position_mm: float = Field(ge=ZABER_POS_MIN_MM, le=ZABER_POS_MAX_MM)
    intensity_v: float


class SurfCavityRunEntry(BaseModel):
    """Parsed cavity resonance lookup entry for one frequency in ``CavityRuns/<freq>MHz.csv``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency_mhz: float = Field(gt=0.0)
    resonant_zaber_pos_mm: float = Field(ge=ZABER_POS_MIN_MM, le=ZABER_POS_MAX_MM)
    resonant_intensity_v: float
    peak_source: str = Field(min_length=1)
    all_peaks: Tuple[SurfCavityPeak, ...] = ()
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    cavity_type: Optional[str] = None


class SurfCavityLookupTable(BaseModel):
    """Lookup table mapping ``frequency_mhz -> (resonant_zaber_pos_mm, source_sha256)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: Dict[float, SurfCavityRunEntry] = Field(default_factory=dict)
    search_settings: Optional[SurfCavitySearchSettings] = None
    search_settings_sha256: Optional[str] = None

    def to_position_map(self) -> Dict[float, Tuple[float, str]]:
        """Return canonical ``{frequency_mhz: (resonant_zaber_pos_mm, source_sha256)}`` map."""
        return {
            freq: (entry.resonant_zaber_pos_mm, entry.source_sha256)
            for freq, entry in sorted(self.entries.items())
        }

    def lookup(
        self,
        freq_mhz: float,
        *,
        max_delta_mhz: float = 5.0,
    ) -> Optional[SurfCavityRunEntry]:
        """Return exact or nearest cavity resonance entry within ``max_delta_mhz``."""
        if not self.entries:
            return None
        if freq_mhz in self.entries:
            return self.entries[freq_mhz]
        nearest_freq = min(self.entries.keys(), key=lambda key: abs(key - freq_mhz))
        if abs(nearest_freq - freq_mhz) <= max_delta_mhz:
            return self.entries[nearest_freq]
        return None


class RawSpectralPeak(BaseModel):
    """Single detected spectral peak component preserved without silent collapse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency_mhz: float = Field(gt=0.0)
    intensity: float
    snr: float = Field(gt=0.0)
    uncertainty_mhz: float = Field(gt=0.0)
    bin_index: int = Field(ge=0)
    source_step_sha256: Optional[str] = None

    @field_validator("source_step_sha256")
    @classmethod
    def _validate_optional_sha(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _SHA256_HEX_RE.match(value):
            raise ValueError("source_step_sha256 must be a 64-character lowercase hex digest.")
        return value


class SurfCandidateTransition(BaseModel):
    """Peak candidate emitted by the SURF Doppler picker; strictly ``AssignmentStatus.PROPOSED``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    rest_frequency_mhz: float = Field(gt=0.0)
    uncertainty_mhz: float = Field(gt=0.0)
    intensity: float
    snr: float = Field(gt=0.0)
    feature_kind: SurfFeatureKind
    status: Any = Field(default_factory=_resolve_proposed_assignment_status)
    raw_components: Tuple[RawSpectralPeak, ...] = Field(min_length=1)
    observed_splitting_mhz: Optional[float] = Field(default=None, gt=0.0)
    expected_splitting_mhz: Optional[float] = Field(default=None, gt=0.0)
    intensity_asymmetry: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gas_name: str = ""
    beam_velocity_m_s: Optional[float] = Field(default=None, gt=0.0)
    source_step_sha256: Optional[str] = None
    source_step_sha256s: Tuple[str, ...] = ()
    source_file: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def _enforce_proposed_only(cls, value: Any) -> Any:
        return _validate_strictly_proposed_status(value)

    @field_validator("source_step_sha256")
    @classmethod
    def _validate_sha(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _SHA256_HEX_RE.match(value):
            raise ValueError("source_step_sha256 must be a 64-character lowercase hex digest.")
        return value


class SurfPeakPickResult(BaseModel):
    """Complete output of coaxial Fabry-Perot Doppler doublet peak picking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: Tuple[SurfCandidateTransition, ...]
    raw_peaks: Tuple[RawSpectralPeak, ...]
    baseline_intensity: float
    mad_noise_sigma: float = Field(ge=0.0)
    snr_threshold: float = Field(gt=0.0)
    gas_name: str
    beam_velocity_m_s: Optional[float] = None
    warnings: Tuple[str, ...] = ()


# =============================================================================
# 5. SURF OUTPUT PARSER
# =============================================================================


class SurfOutputParser:
    """Parser and reconciler for native ``GruLabRotSpec/SURF`` disk artifacts."""

    def __init__(
        self,
        *,
        rail_tolerance_mm: float = DEFAULT_RAIL_TOLERANCE_MM,
        stem_tolerance_mhz: float = 1e-3,
    ) -> None:
        if not math.isfinite(rail_tolerance_mm) or rail_tolerance_mm < 0.0:
            raise SurfBoundsError("rail_tolerance_mm must be finite and >= 0.")
        if not math.isfinite(stem_tolerance_mhz) or stem_tolerance_mhz < 0.0:
            raise SurfBoundsError("stem_tolerance_mhz must be finite and >= 0.")
        self._rail_tolerance_mm = float(rail_tolerance_mm)
        self._stem_tolerance_mhz = float(stem_tolerance_mhz)

    @staticmethod
    def is_surf_spectrum_header(header: Sequence[str]) -> bool:
        """Return True iff ``header`` exactly matches SURF's 4-column FFT spectrum CSV header."""
        cleaned = [col.lstrip("\ufeff").strip() for col in header]
        return tuple(cleaned) == SURF_SPECTRUM_COLUMNS

    @staticmethod
    def is_surf_cavity_run_header(header: Sequence[str]) -> bool:
        """Return True iff ``header`` exactly matches SURF's 5-column cavity-search CSV header."""
        cleaned = [col.lstrip("\ufeff").strip() for col in header]
        return tuple(cleaned) == SURF_CAVITY_RUN_COLUMNS

    def parse_step_csv_bytes(
        self,
        raw_bytes: bytes,
        *,
        source_path: Optional[Path] = None,
        step_size_mhz: Optional[float] = None,
        step_direction: StepDirection = StepDirection.Up,
        verify_filename_stem: bool = True,
    ) -> SurfStepData:
        """Parse an immutable per-step ``CavityFiles/<total_frequency>.csv`` payload."""
        source_label = str(source_path) if source_path is not None else "<bytes>"
        file_sha256 = compute_bytes_sha256(raw_bytes)
        text = _decode_utf8_no_bom(raw_bytes, source_label=source_label)

        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SurfCsvSchemaError(f"SURF step CSV {source_label} is empty.") from exc

        if not self.is_surf_spectrum_header(header):
            raise SurfCsvSchemaError(
                f"SURF step CSV {source_label} has invalid header {header!r}; "
                f"expected {list(SURF_SPECTRUM_COLUMNS)!r}."
            )

        freqs: List[float] = []
        intensities: List[float] = []
        center_freq_mhz: Optional[float] = None
        cavity_position_mm: Optional[float] = None

        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 4:
                raise SurfCsvSchemaError(
                    f"Row {row_idx} in {source_label} has {len(row)} column(s); expected 4."
                )

            freq_val = _parse_finite_fp64(row[0], field_name="Frequency (MHz)", row_number=row_idx)
            int_val = _parse_finite_fp64(row[1], field_name="Intensity", row_number=row_idx)
            freqs.append(freq_val)
            intensities.append(int_val)

            cf_cell = row[2].strip()
            cp_cell = row[3].strip()

            if len(freqs) == 1:
                if not cf_cell or not cp_cell:
                    raise SurfCsvSchemaError(
                        f"Row {row_idx} in {source_label} is missing mandatory row-0 "
                        "'Center Freq' or 'Cavity Position' metadata."
                    )
                center_freq_mhz = _parse_finite_fp64(cf_cell, field_name="Center Freq", row_number=row_idx)
                cavity_position_mm = _parse_finite_fp64(cp_cell, field_name="Cavity Position", row_number=row_idx)
            elif cf_cell or cp_cell:
                raise SurfCsvSchemaError(
                    f"Row {row_idx} in step file {source_label} contains unexpected non-empty "
                    "metadata outside row 0 (expected sparse row-0 layout for CavityFiles/*.csv)."
                )

        if not freqs or center_freq_mhz is None or cavity_position_mm is None:
            raise SurfCsvSchemaError(f"SURF step CSV {source_label} contains zero spectral rows.")

        cavity_dropout = _validate_zaber_position(
            cavity_position_mm,
            row_number=1,
            rail_tolerance_mm=self._rail_tolerance_mm,
        )

        if verify_filename_stem and source_path is not None:
            try:
                stem_freq = float(source_path.stem)
            except ValueError as exc:
                raise SurfCsvSchemaError(
                    f"Step CSV filename stem {source_path.stem!r} is not a numeric center frequency."
                ) from exc
            if not math.isfinite(stem_freq) or abs(stem_freq - center_freq_mhz) > self._stem_tolerance_mhz:
                raise SurfCsvSchemaError(
                    f"Step CSV filename stem {source_path.stem!r} does not match row-0 "
                    f"Center Freq {center_freq_mhz} MHz (tolerance {self._stem_tolerance_mhz} MHz)."
                )

        full_freq_arr = np.asarray(freqs, dtype=np.float64)
        full_int_arr = np.asarray(intensities, dtype=np.float64)

        if step_size_mhz is not None:
            stitched_freq, stitched_int = slice_surf_step_window(
                full_freq_arr,
                full_int_arr,
                center_freq_mhz=center_freq_mhz,
                step_size_mhz=step_size_mhz,
                step_direction=step_direction,
            )
        else:
            if step_direction == StepDirection.Down:
                stitched_freq = np.ascontiguousarray(full_freq_arr[::-1], dtype=np.float64)
                stitched_int = np.ascontiguousarray(full_int_arr[::-1], dtype=np.float64)
            else:
                stitched_freq = full_freq_arr.copy()
                stitched_int = full_int_arr.copy()

        return SurfStepData(
            file_path=source_path,
            file_sha256=file_sha256,
            center_freq_mhz=center_freq_mhz,
            cavity_position_mm=cavity_position_mm,
            cavity_dropout=cavity_dropout,
            full_window_freq_mhz=full_freq_arr,
            full_window_intensity=full_int_arr,
            stitched_freq_mhz=stitched_freq,
            stitched_intensity=stitched_int,
            step_size_mhz=step_size_mhz,
            step_direction=step_direction,
            row_count=len(freqs),
        )

    def parse_step_csv(
        self,
        path: Union[str, Path],
        *,
        step_size_mhz: Optional[float] = None,
        step_direction: StepDirection = StepDirection.Up,
        verify_filename_stem: bool = True,
    ) -> SurfStepData:
        """Read and parse a single ``CavityFiles/<total_frequency>.csv`` file from disk."""
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"SURF step CSV not found: {resolved}")
        raw_bytes = resolved.read_bytes()
        return self.parse_step_csv_bytes(
            raw_bytes,
            source_path=resolved,
            step_size_mhz=step_size_mhz,
            step_direction=step_direction,
            verify_filename_stem=verify_filename_stem,
        )

    def parse_finalized_csv_bytes(
        self,
        raw_bytes: bytes,
        *,
        source_path: Optional[Path] = None,
    ) -> SurfFinalizedCsvData:
        """Parse a finalized compacted (or appended) multi-step ``<filename>.csv`` payload."""
        source_label = str(source_path) if source_path is not None else "<bytes>"
        file_sha256 = compute_bytes_sha256(raw_bytes)
        text = _decode_utf8_no_bom(raw_bytes, source_label=source_label)

        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SurfCsvSchemaError(f"SURF spectrum CSV {source_label} is empty.") from exc

        if not self.is_surf_spectrum_header(header):
            raise SurfCsvSchemaError(
                f"SURF spectrum CSV {source_label} has invalid header {header!r}; "
                f"expected {list(SURF_SPECTRUM_COLUMNS)!r}."
            )

        freqs: List[float] = []
        intensities: List[float] = []
        center_freqs: List[float] = []
        cavity_positions: List[float] = []
        metadata_rows: List[int] = []
        dropout_steps: List[float] = []

        data_row_index = 0
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 4:
                raise SurfCsvSchemaError(
                    f"Row {row_idx} in {source_label} has {len(row)} column(s); expected 4."
                )

            freq_val = _parse_finite_fp64(row[0], field_name="Frequency (MHz)", row_number=row_idx)
            int_val = _parse_finite_fp64(row[1], field_name="Intensity", row_number=row_idx)
            freqs.append(freq_val)
            intensities.append(int_val)

            cf_cell = row[2].strip()
            cp_cell = row[3].strip()
            if cf_cell or cp_cell:
                if not (cf_cell and cp_cell):
                    raise SurfCsvSchemaError(
                        f"Row {row_idx} in {source_label} has unpaired 'Center Freq'/'Cavity Position' metadata."
                    )
                cf_val = _parse_finite_fp64(cf_cell, field_name="Center Freq", row_number=row_idx)
                cp_val = _parse_finite_fp64(cp_cell, field_name="Cavity Position", row_number=row_idx)
                is_dropout = _validate_zaber_position(
                    cp_val,
                    row_number=row_idx,
                    rail_tolerance_mm=self._rail_tolerance_mm,
                )
                center_freqs.append(cf_val)
                cavity_positions.append(cp_val)
                metadata_rows.append(data_row_index)
                if is_dropout:
                    dropout_steps.append(cf_val)

            data_row_index += 1

        if not freqs:
            raise SurfCsvSchemaError(f"SURF spectrum CSV {source_label} contains zero spectral rows.")

        is_compacted = bool(metadata_rows) and metadata_rows == list(range(len(metadata_rows)))

        return SurfFinalizedCsvData(
            file_path=source_path,
            file_sha256=file_sha256,
            freq_mhz=np.asarray(freqs, dtype=np.float64),
            intensity=np.asarray(intensities, dtype=np.float64),
            center_freqs_mhz=tuple(center_freqs),
            cavity_positions_mm=tuple(cavity_positions),
            metadata_row_indices=tuple(metadata_rows),
            is_compacted=is_compacted,
            cavity_dropout_steps_mhz=tuple(dropout_steps),
            row_count=len(freqs),
        )

    def parse_finalized_csv(self, path: Union[str, Path]) -> SurfFinalizedCsvData:
        """Read and parse a finalized compacted ``<filename>.csv`` from disk."""
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"SURF finalized CSV not found: {resolved}")
        return self.parse_finalized_csv_bytes(resolved.read_bytes(), source_path=resolved)

    @staticmethod
    def stitch_steps(
        steps: Sequence[SurfStepData],
        *,
        step_direction: StepDirection = StepDirection.Up,
    ) -> Tuple[np.ndarray, np.ndarray, List[SurfStepData]]:
        """Order and concatenate per-step sliced spectra matching ``SURF`` run order."""
        if not steps:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy(), []

        reverse_order = step_direction == StepDirection.Down
        ordered = sorted(steps, key=lambda item: item.center_freq_mhz, reverse=reverse_order)
        stitched_freq = np.concatenate([item.stitched_freq_mhz for item in ordered]).astype(np.float64, copy=False)
        stitched_int = np.concatenate([item.stitched_intensity for item in ordered]).astype(np.float64, copy=False)
        return stitched_freq, stitched_int, ordered

    @staticmethod
    def expected_step_frequencies(scan_settings: SurfFrequencyScanSettings) -> Tuple[float, ...]:
        """Compute nominal step center frequencies from ``scan_settings.toml``."""
        start = float(scan_settings.scan_parameters.start_freq)
        end = float(scan_settings.scan_parameters.end_freq)
        step = float(scan_settings.scan_parameters.step_size)
        direction = StepDirection.from_bounds(start, end)
        sign = 1.0 if direction == StepDirection.Up else -1.0
        n_steps = math.ceil(abs(end - start) / step - 1e-12) + 1
        return tuple(round(start + sign * idx * step, 6) for idx in range(n_steps))

    def reconcile_run(
        self,
        *,
        finalized_csv_path: Union[str, Path],
        step_files: Sequence[Union[str, Path, SurfStepData]],
        scan_settings: Optional[SurfFrequencyScanSettings] = None,
        step_size_mhz: Optional[float] = None,
        step_direction: Optional[StepDirection] = None,
        config_toml_path: Optional[Union[str, Path]] = None,
        scan_settings_toml_path: Optional[Union[str, Path]] = None,
        fp64_atol: float = 1e-9,
        step_match_tol_mhz: float = 1e-4,
    ) -> SurfReconciliationReport:
        """Reconcile finalized compacted ``<filename>.csv`` against stitched ``CavityFiles/*.csv``."""
        resolved_finalized = Path(finalized_csv_path)
        finalized = self.parse_finalized_csv(resolved_finalized)

        if scan_settings is not None:
            effective_step_size = (
                step_size_mhz if step_size_mhz is not None else float(scan_settings.scan_parameters.step_size)
            )
            effective_direction = (
                step_direction
                if step_direction is not None
                else StepDirection.from_bounds(
                    scan_settings.scan_parameters.start_freq,
                    scan_settings.scan_parameters.end_freq,
                )
            )
        else:
            effective_step_size = step_size_mhz
            effective_direction = step_direction or StepDirection.Up

        parsed_steps: List[SurfStepData] = []
        for item in step_files:
            if isinstance(item, SurfStepData):
                if effective_step_size is not None and (
                    item.step_size_mhz != effective_step_size or item.step_direction != effective_direction
                ):
                    sliced_f, sliced_i = slice_surf_step_window(
                        item.full_window_freq_mhz,
                        item.full_window_intensity,
                        center_freq_mhz=item.center_freq_mhz,
                        step_size_mhz=effective_step_size,
                        step_direction=effective_direction,
                    )
                    parsed_steps.append(
                        item.model_copy(
                            update={
                                "stitched_freq_mhz": sliced_f,
                                "stitched_intensity": sliced_i,
                                "step_size_mhz": effective_step_size,
                                "step_direction": effective_direction,
                            }
                        )
                    )
                else:
                    parsed_steps.append(item)
            else:
                parsed_steps.append(
                    self.parse_step_csv(
                        item,
                        step_size_mhz=effective_step_size,
                        step_direction=effective_direction,
                    )
                )

        stitched_freq, stitched_int, ordered_steps = self.stitch_steps(
            parsed_steps,
            step_direction=effective_direction,
        )

        discrepancies: List[str] = []
        warnings: List[str] = []

        step_freqs = [step.center_freq_mhz for step in ordered_steps]
        finalized_freqs = list(finalized.center_freqs_mhz)

        def _has_freq(target: float, pool: Sequence[float]) -> bool:
            return any(abs(candidate - target) <= step_match_tol_mhz for candidate in pool)

        missing_steps: List[float] = []
        extra_steps: List[float] = []

        for freq_val in finalized_freqs:
            if not _has_freq(freq_val, step_freqs):
                missing_steps.append(freq_val)

        if scan_settings is not None:
            for expected_freq in self.expected_step_frequencies(scan_settings):
                if not _has_freq(expected_freq, step_freqs) and not _has_freq(expected_freq, missing_steps):
                    missing_steps.append(expected_freq)
                if not _has_freq(expected_freq, finalized_freqs):
                    discrepancies.append(
                        f"Expected step {expected_freq:.6f} MHz from scan_settings.toml missing in finalized CSV metadata."
                    )

        for freq_val in step_freqs:
            if not _has_freq(freq_val, finalized_freqs):
                extra_steps.append(freq_val)

        if missing_steps:
            discrepancies.append(
                f"Missing CavityFiles step CSV(s) for center frequencies (MHz): {missing_steps}"
            )
        if extra_steps:
            discrepancies.append(
                f"CavityFiles step CSV(s) absent from finalized CSV metadata (MHz): {extra_steps}"
            )

        # Verify metadata pair consistency (center_freq & cavity_position) in run order
        if len(ordered_steps) == len(finalized.center_freqs_mhz):
            for idx, step in enumerate(ordered_steps):
                if abs(step.center_freq_mhz - finalized.center_freqs_mhz[idx]) > step_match_tol_mhz:
                    discrepancies.append(
                        f"Step #{idx} center frequency mismatch: CavityFiles={step.center_freq_mhz} MHz "
                        f"vs finalized={finalized.center_freqs_mhz[idx]} MHz."
                    )
                if abs(step.cavity_position_mm - finalized.cavity_positions_mm[idx]) > 1e-6:
                    discrepancies.append(
                        f"Step #{idx} ({step.center_freq_mhz} MHz) cavity position mismatch: "
                        f"CavityFiles={step.cavity_position_mm} mm vs finalized={finalized.cavity_positions_mm[idx]} mm."
                    )

        # Spectral FP64 array comparison
        row_count_match = int(stitched_freq.size) == int(finalized.freq_mhz.size)
        if row_count_match and stitched_freq.size > 0:
            max_freq_diff = float(np.max(np.abs(stitched_freq - finalized.freq_mhz)))
            max_int_diff = float(np.max(np.abs(stitched_int - finalized.intensity)))
            if max_freq_diff > fp64_atol:
                discrepancies.append(
                    f"Stitched vs finalized Frequency (MHz) max abs diff {max_freq_diff:.3e} exceeds tolerance {fp64_atol:.3e}."
                )
            if max_int_diff > fp64_atol:
                discrepancies.append(
                    f"Stitched vs finalized Intensity max abs diff {max_int_diff:.3e} exceeds tolerance {fp64_atol:.3e}."
                )
        else:
            max_freq_diff = float("inf") if not row_count_match else 0.0
            max_int_diff = float("inf") if not row_count_match else 0.0
            if not row_count_match:
                discrepancies.append(
                    f"Row count mismatch: stitched CavityFiles={stitched_freq.size} vs finalized CSV={finalized.freq_mhz.size}."
                )
                # Keep finite diagnostic numbers for strict JSON serialization
                max_freq_diff = abs(float(stitched_freq.size - finalized.freq_mhz.size))
                max_int_diff = abs(float(stitched_int.size - finalized.intensity.size))

        # Flag cavity-tracking rail drop-outs (0.0 mm or 50.0 mm)
        dropout_steps: List[float] = []
        for step in ordered_steps:
            if step.cavity_dropout and not _has_freq(step.center_freq_mhz, dropout_steps):
                dropout_steps.append(step.center_freq_mhz)
        for freq_val in finalized.cavity_dropout_steps_mhz:
            if not _has_freq(freq_val, dropout_steps):
                dropout_steps.append(freq_val)

        if dropout_steps:
            warnings.append(
                f"Cavity-tracking rail drop-out detected (Zaber pinned at 0.0 or 50.0 mm) at step(s): {dropout_steps} MHz."
            )

        config_sha = compute_file_sha256(config_toml_path) if config_toml_path and Path(config_toml_path).is_file() else None
        settings_sha = (
            compute_file_sha256(scan_settings_toml_path)
            if scan_settings_toml_path and Path(scan_settings_toml_path).is_file()
            else None
        )

        step_sha_map: Dict[str, str] = {}
        for step in ordered_steps:
            key = step.file_path.name if step.file_path is not None else f"{step.center_freq_mhz}.csv"
            step_sha_map[key] = step.file_sha256

        spectrum_matches = (
            row_count_match
            and max_freq_diff <= fp64_atol
            and max_int_diff <= fp64_atol
            and len(discrepancies) == 0
        )
        reconciled = spectrum_matches and len(missing_steps) == 0 and len(extra_steps) == 0

        return SurfReconciliationReport(
            reconciled=reconciled,
            spectrum_matches=spectrum_matches,
            has_cavity_dropout=len(dropout_steps) > 0,
            finalized_csv_path=resolved_finalized,
            finalized_csv_sha256=finalized.file_sha256,
            step_file_count=len(ordered_steps),
            finalized_metadata_count=len(finalized.center_freqs_mhz),
            finalized_is_compacted=finalized.is_compacted,
            row_count_match=row_count_match,
            stitched_row_count=int(stitched_freq.size),
            finalized_row_count=int(finalized.freq_mhz.size),
            max_freq_abs_diff_mhz=max_freq_diff,
            max_intensity_abs_diff=max_int_diff,
            missing_steps_mhz=tuple(missing_steps),
            extra_steps_mhz=tuple(extra_steps),
            cavity_dropout_steps_mhz=tuple(dropout_steps),
            discrepancies=tuple(discrepancies),
            warnings=tuple(warnings),
            config_toml_sha256=config_sha,
            scan_settings_toml_sha256=settings_sha,
            step_file_sha256s=step_sha_map,
        )

    def reconcile_run_directory(
        self,
        run_dir: Union[str, Path],
        *,
        fp64_atol: float = 1e-9,
    ) -> SurfReconciliationReport:
        """Load ``scan_settings.toml``, ``CavityFiles/*.csv``, and finalized ``<filename>.csv`` in ``run_dir``."""
        root = Path(run_dir)
        settings_path = root / "scan_settings.toml"
        if not settings_path.is_file():
            raise FileNotFoundError(f"scan_settings.toml not found in SURF run directory: {root}")
        scan_settings = SurfFrequencyScanSettings.from_surf_toml_file(settings_path)
        stem = scan_settings.output_settings.filename
        finalized_path = root / f"{stem}.csv"
        config_path = root / f"{stem}_config.toml"
        if config_path.is_file():
            SurfConfig.from_surf_toml_file(config_path)

        cavity_dir = root / "CavityFiles"
        step_paths = sorted(cavity_dir.glob("*.csv")) if cavity_dir.is_dir() else []

        return self.reconcile_run(
            finalized_csv_path=finalized_path,
            step_files=step_paths,
            scan_settings=scan_settings,
            config_toml_path=config_path if config_path.is_file() else None,
            scan_settings_toml_path=settings_path,
            fp64_atol=fp64_atol,
        )

    def parse_cavity_run_csv(
        self,
        path: Union[str, Path],
        *,
        cavity_type: Optional[str] = None,
    ) -> SurfCavityRunEntry:
        """Parse a SURF cavity-search ``CavityRuns/<freq>MHz.csv`` file into a lookup entry."""
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"SURF cavity run CSV not found: {resolved}")
        raw_bytes = resolved.read_bytes()
        file_sha256 = compute_bytes_sha256(raw_bytes)
        text = _decode_utf8_no_bom(raw_bytes, source_label=str(resolved))

        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SurfCsvSchemaError(f"SURF cavity run CSV {resolved} is empty.") from exc

        if not self.is_surf_cavity_run_header(header):
            raise SurfCsvSchemaError(
                f"SURF cavity run CSV {resolved} has invalid header {header!r}; "
                f"expected {list(SURF_CAVITY_RUN_COLUMNS)!r}."
            )

        positions: List[float] = []
        trace_intensities: List[float] = []
        freq_values: List[float] = []
        peaks: List[SurfCavityPeak] = []

        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 5:
                raise SurfCsvSchemaError(
                    f"Row {row_idx} in cavity run CSV {resolved} has {len(row)} column(s); expected 5."
                )

            pos_mm = _parse_finite_fp64(row[0], field_name="Zaber Position (mm)", row_number=row_idx)
            _validate_zaber_position(pos_mm, row_number=row_idx, rail_tolerance_mm=self._rail_tolerance_mm)
            int_v = _parse_finite_fp64(row[1], field_name="Intensity (V)", row_number=row_idx)
            freq_mhz = _parse_finite_fp64(row[2], field_name="Frequency (MHz)", row_number=row_idx)
            if freq_mhz <= 0.0:
                raise SurfBoundsError(f"Row {row_idx} in {resolved}: Frequency (MHz) must be > 0.")

            positions.append(pos_mm)
            trace_intensities.append(int_v)
            freq_values.append(freq_mhz)

            peak_pos_cell = row[3].strip()
            peak_int_cell = row[4].strip()
            if peak_pos_cell or peak_int_cell:
                if not (peak_pos_cell and peak_int_cell):
                    raise SurfCsvSchemaError(
                        f"Row {row_idx} in cavity run CSV {resolved} has unpaired Peaks/Intensity cells."
                    )
                peak_pos = _parse_finite_fp64(peak_pos_cell, field_name="Peaks", row_number=row_idx)
                _validate_zaber_position(peak_pos, row_number=row_idx, rail_tolerance_mm=self._rail_tolerance_mm)
                peak_int = _parse_finite_fp64(peak_int_cell, field_name="Intensity", row_number=row_idx)
                peaks.append(SurfCavityPeak(zaber_position_mm=peak_pos, intensity_v=peak_int))

        if not positions:
            raise SurfCsvSchemaError(f"SURF cavity run CSV {resolved} contains zero data rows.")

        nominal_freq = freq_values[0]
        if any(abs(fv - nominal_freq) > self._stem_tolerance_mhz for fv in freq_values):
            raise SurfCsvSchemaError(
                f"SURF cavity run CSV {resolved} contains inconsistent Frequency (MHz) entries."
            )

        stem_match = _CAVITY_RUN_FREQ_RE.match(resolved.stem)
        if stem_match is not None:
            stem_freq = float(stem_match.group("freq"))
            if abs(stem_freq - nominal_freq) > self._stem_tolerance_mhz:
                raise SurfCsvSchemaError(
                    f"Cavity run filename {resolved.name!r} disagrees with Frequency (MHz)={nominal_freq}."
                )

        if peaks:
            best_peak = max(peaks, key=lambda item: item.intensity_v)
            resonant_pos = best_peak.zaber_position_mm
            resonant_int = best_peak.intensity_v
            peak_source = "surf_peaks"
        else:
            max_idx = int(np.argmax(np.asarray(trace_intensities, dtype=np.float64)))
            resonant_pos = positions[max_idx]
            resonant_int = trace_intensities[max_idx]
            peak_source = "trace_argmax"

        return SurfCavityRunEntry(
            frequency_mhz=nominal_freq,
            resonant_zaber_pos_mm=resonant_pos,
            resonant_intensity_v=resonant_int,
            peak_source=peak_source,
            all_peaks=tuple(peaks),
            source_file=resolved.name,
            source_sha256=file_sha256,
            cavity_type=cavity_type,
        )

    def parse_cavity_search_dir(self, search_dir: Union[str, Path]) -> SurfCavityLookupTable:
        """Parse ``search_settings.toml`` (if present) and all ``CavityRuns/<freq>MHz.csv`` files."""
        root = Path(search_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"SURF cavity search directory not found: {root}")

        settings_path = root / "search_settings.toml"
        search_settings: Optional[SurfCavitySearchSettings] = None
        settings_sha: Optional[str] = None
        cavity_type: Optional[str] = None

        if settings_path.is_file():
            search_settings = SurfCavitySearchSettings.from_surf_toml_file(settings_path)
            settings_sha = compute_file_sha256(settings_path)
            cavity_type = search_settings.cavity_type.value

        runs_dir = root / "CavityRuns"
        csv_paths = sorted(runs_dir.glob("*.csv")) if runs_dir.is_dir() else sorted(root.glob("*MHz.csv"))

        entries: Dict[float, SurfCavityRunEntry] = {}
        for csv_path in csv_paths:
            entry = self.parse_cavity_run_csv(csv_path, cavity_type=cavity_type)
            entries[entry.frequency_mhz] = entry

        return SurfCavityLookupTable(
            entries=entries,
            search_settings=search_settings,
            search_settings_sha256=settings_sha,
        )


# =============================================================================
# 6. COAXIAL FABRY-PEROT DOPPLER DOUBLET PEAK PICKER
# =============================================================================


def _estimate_mad_baseline_and_noise(intensities: np.ndarray) -> Tuple[float, float]:
    """Estimate robust spectral baseline and noise standard deviation via MAD."""
    baseline = float(np.median(intensities))
    abs_dev = np.abs(intensities - baseline)
    mad = float(np.median(abs_dev))
    sigma = 1.4826 * mad
    if sigma > 0.0:
        return baseline, sigma

    # Fallback when >= 50% of bins share an identical floor value
    lower_half = intensities[intensities <= np.percentile(intensities, 50.0)]
    if lower_half.size > 1:
        lower_std = float(np.std(lower_half))
        if lower_std > 0.0:
            return baseline, lower_std

    mean_dev = float(np.mean(abs_dev))
    if mean_dev > 0.0:
        return baseline, 1.2533 * mean_dev

    return baseline, 0.0


def _refine_peak_parabolic(freqs: np.ndarray, intensities: np.ndarray, peak_idx: int) -> float:
    """Apply bounded 3-point parabolic apex refinement around a discrete peak bin."""
    if peak_idx <= 0 or peak_idx >= freqs.size - 1:
        return float(freqs[peak_idx])

    y_left = float(intensities[peak_idx - 1])
    y_mid = float(intensities[peak_idx])
    y_right = float(intensities[peak_idx + 1])
    denom = y_left - 2.0 * y_mid + y_right
    if denom >= 0.0 or not math.isfinite(denom):
        return float(freqs[peak_idx])

    delta_bins = 0.5 * (y_left - y_right) / denom
    delta_bins = max(-0.5, min(0.5, delta_bins))
    if delta_bins >= 0.0:
        step = float(freqs[peak_idx + 1] - freqs[peak_idx])
    else:
        step = float(freqs[peak_idx] - freqs[peak_idx - 1])
    return float(freqs[peak_idx] + delta_bins * step)


def _lookup_step_sha_for_freq(
    freq_mhz: float,
    default_sha: Optional[str],
    step_provenance_spans: Optional[Sequence[Tuple[float, float, str]]],
) -> Optional[str]:
    """Resolve the originating step-CSV SHA-256 for a given peak frequency."""
    if step_provenance_spans:
        for low_mhz, high_mhz, sha_val in step_provenance_spans:
            if low_mhz <= freq_mhz <= high_mhz:
                return sha_val
    return default_sha


def pick_surf_doppler_peaks(
    freqs_mhz: Union[np.ndarray, Sequence[float]],
    intensities: Union[np.ndarray, Sequence[float]],
    *,
    gas_name: str = "Ne",
    snr_threshold: float = 5.0,
    beam_velocity_m_s: Optional[float] = None,
    splitting_tolerance_fraction: float = 0.25,
    min_splitting_tol_mhz: float = 0.005,
    max_intensity_asymmetry: float = 0.50,
    min_peak_distance_bins: int = 1,
    source_step_sha256: Optional[str] = None,
    step_provenance_spans: Optional[Sequence[Tuple[float, float, str]]] = None,
    source_file: Optional[str] = None,
) -> SurfPeakPickResult:
    """Detect SURF FFT peaks and pair coaxial Fabry-Perot Doppler doublets.

    - Estimates noise via robust MAD and detects peaks with ``scipy.signal.find_peaks``
      at ``>= snr_threshold`` (default ``5.0``).
    - Pairs isolated, symmetric-intensity doublets matching
      ``delta_nu_d = 2 * nu * (v_beam / c)`` for the carrier gas (``Ne``, ``Ar``,
      ``He``, ``N2``) into midpoint rest frequencies ``(nu_+ + nu_-) / 2`` with
      explicit uncertainty while preserving raw components.
    - Preserves isolated singlets and ambiguous overlapping multiplets/blends
      uncollapsed (never silently merging blends).
    - Emits ALL candidate transitions strictly as ``AssignmentStatus.PROPOSED``
      (never auto-``APPROVED``) with step-CSV SHA-256 provenance.
    """
    if not math.isfinite(snr_threshold) or snr_threshold <= 0.0:
        raise SurfIngestError(f"snr_threshold must be finite and > 0, got {snr_threshold!r}.")
    if not math.isfinite(splitting_tolerance_fraction) or splitting_tolerance_fraction <= 0.0:
        raise SurfIngestError("splitting_tolerance_fraction must be finite and > 0.")
    if not math.isfinite(max_intensity_asymmetry) or not (0.0 <= max_intensity_asymmetry <= 1.0):
        raise SurfIngestError("max_intensity_asymmetry must lie within [0.0, 1.0].")
    if min_peak_distance_bins < 1:
        raise SurfIngestError("min_peak_distance_bins must be >= 1.")

    freq_arr = np.asarray(freqs_mhz, dtype=np.float64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    if freq_arr.ndim != 1 or int_arr.ndim != 1 or freq_arr.shape != int_arr.shape:
        raise SurfCsvSchemaError("freqs_mhz and intensities must be 1D arrays of equal length.")
    if freq_arr.size == 0:
        raise SurfCsvSchemaError("Cannot pick peaks from an empty spectrum.")
    if not (np.all(np.isfinite(freq_arr)) and np.all(np.isfinite(int_arr))):
        raise SurfCsvSchemaError("Spectrum contains non-finite frequency or intensity values.")

    canonical_gas, v_beam = resolve_beam_velocity_m_s(gas_name, beam_velocity_m_s=beam_velocity_m_s)
    warnings: List[str] = []

    # Sort strictly ascending by frequency for peak finding regardless of Up/Down scan direction
    order = np.argsort(freq_arr, kind="mergesort")
    sorted_freqs = freq_arr[order]
    sorted_ints = int_arr[order]

    baseline, sigma_noise = _estimate_mad_baseline_and_noise(sorted_ints)
    if sigma_noise <= 0.0:
        return SurfPeakPickResult(
            candidates=(),
            raw_peaks=(),
            baseline_intensity=baseline,
            mad_noise_sigma=0.0,
            snr_threshold=float(snr_threshold),
            gas_name=canonical_gas,
            beam_velocity_m_s=v_beam,
            warnings=("Flat spectrum with zero variance; no peaks detected.",),
        )

    height_threshold = baseline + float(snr_threshold) * sigma_noise
    prominence_threshold = max(2.0 * sigma_noise, 0.35 * float(snr_threshold) * sigma_noise)

    peak_indices, _ = find_peaks(
        sorted_ints,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=min_peak_distance_bins,
    )

    diffs = np.diff(sorted_freqs)
    positive_diffs = diffs[diffs > 0.0]
    median_bin_mhz = float(np.median(positive_diffs)) if positive_diffs.size > 0 else 0.005

    raw_peaks: List[RawSpectralPeak] = []
    for peak_idx in peak_indices:
        idx = int(peak_idx)
        refined_freq = _refine_peak_parabolic(sorted_freqs, sorted_ints, idx)
        peak_int = float(sorted_ints[idx])
        snr = (peak_int - baseline) / sigma_noise
        local_spacing = (
            float(sorted_freqs[idx + 1] - sorted_freqs[idx])
            if idx + 1 < sorted_freqs.size and sorted_freqs[idx + 1] > sorted_freqs[idx]
            else median_bin_mhz
        )
        unc_mhz = max(local_spacing / max(2.0 * snr, 1.0), 0.1 * local_spacing, 1e-6)
        step_sha = _lookup_step_sha_for_freq(refined_freq, source_step_sha256, step_provenance_spans)
        raw_peaks.append(
            RawSpectralPeak(
                frequency_mhz=refined_freq,
                intensity=peak_int,
                snr=float(snr),
                uncertainty_mhz=float(unc_mhz),
                bin_index=idx,
                source_step_sha256=step_sha,
            )
        )

    if not raw_peaks:
        return SurfPeakPickResult(
            candidates=(),
            raw_peaks=(),
            baseline_intensity=baseline,
            mad_noise_sigma=sigma_noise,
            snr_threshold=float(snr_threshold),
            gas_name=canonical_gas,
            beam_velocity_m_s=v_beam,
            warnings=tuple(warnings),
        )

    # If carrier gas is blank and no beam velocity override was supplied, do not fabricate pairings
    if v_beam is None:
        warnings.append(
            "Carrier gas unspecified; preserving all detected peaks as uncollapsed PROPOSED singlets."
        )
        singlet_candidates = tuple(
            SurfCandidateTransition(
                candidate_id=f"SURF_SGL_{idx:04d}_{peak.frequency_mhz:.4f}",
                rest_frequency_mhz=peak.frequency_mhz,
                uncertainty_mhz=peak.uncertainty_mhz,
                intensity=peak.intensity,
                snr=peak.snr,
                feature_kind=SurfFeatureKind.SINGLET,
                raw_components=(peak,),
                gas_name=canonical_gas,
                beam_velocity_m_s=None,
                source_step_sha256=peak.source_step_sha256,
                source_step_sha256s=(peak.source_step_sha256,) if peak.source_step_sha256 else (),
                source_file=source_file,
            )
            for idx, peak in enumerate(raw_peaks)
        )
        return SurfPeakPickResult(
            candidates=singlet_candidates,
            raw_peaks=tuple(raw_peaks),
            baseline_intensity=baseline,
            mad_noise_sigma=sigma_noise,
            snr_threshold=float(snr_threshold),
            gas_name=canonical_gas,
            beam_velocity_m_s=None,
            warnings=tuple(warnings),
        )

    # Build candidate symmetric Doppler doublet adjacency graph
    n_peaks = len(raw_peaks)
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n_peaks)}
    pair_metrics: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    for i in range(n_peaks):
        peak_i = raw_peaks[i]
        for j in range(i + 1, n_peaks):
            peak_j = raw_peaks[j]
            sep = peak_j.frequency_mhz - peak_i.frequency_mhz
            mid_freq = 0.5 * (peak_i.frequency_mhz + peak_j.frequency_mhz)
            expected_sep = 2.0 * mid_freq * (v_beam / SPEED_OF_LIGHT_M_S)
            tol = max(splitting_tolerance_fraction * expected_sep, min_splitting_tol_mhz)
            if sep > expected_sep + tol:
                break
            if abs(sep - expected_sep) <= tol:
                max_int = max(peak_i.intensity, peak_j.intensity)
                asymmetry = abs(peak_j.intensity - peak_i.intensity) / max_int if max_int > 0.0 else 1.0
                if asymmetry <= max_intensity_asymmetry:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    pair_metrics[(i, j)] = (sep, expected_sep, asymmetry)

    # Find connected components to separate clean isolated doublets from ambiguous blends
    visited = [False] * n_peaks
    candidates: List[SurfCandidateTransition] = []
    blend_indices: set[int] = set()
    clean_pairs: List[Tuple[int, int]] = []

    for start_idx in range(n_peaks):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        component: List[int] = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)

        component.sort()
        if len(component) == 2:
            left_idx, right_idx = component[0], component[1]
            # Check for any intervening detected peak strictly inside (nu_left, nu_right)
            intervening = [
                k for k in range(left_idx + 1, right_idx)
                if raw_peaks[left_idx].frequency_mhz < raw_peaks[k].frequency_mhz < raw_peaks[right_idx].frequency_mhz
            ]
            if intervening:
                blend_indices.add(left_idx)
                blend_indices.add(right_idx)
                for mid_idx in intervening:
                    blend_indices.add(mid_idx)
            else:
                clean_pairs.append((left_idx, right_idx))
        elif len(component) >= 3:
            for member in component:
                blend_indices.add(member)

    paired_indices: set[int] = set()
    for pair_num, (left_idx, right_idx) in enumerate(clean_pairs):
        paired_indices.add(left_idx)
        paired_indices.add(right_idx)
        peak_left = raw_peaks[left_idx]
        peak_right = raw_peaks[right_idx]
        obs_sep, exp_sep, asym = pair_metrics[(left_idx, right_idx)]
        rest_freq = 0.5 * (peak_left.frequency_mhz + peak_right.frequency_mhz)
        combined_unc = math.sqrt(
            (0.5 * peak_left.uncertainty_mhz) ** 2
            + (0.5 * peak_right.uncertainty_mhz) ** 2
            + (0.1 * abs(obs_sep - exp_sep)) ** 2
        )
        sha_list = tuple(
            dict.fromkeys(
                sha
                for sha in (peak_left.source_step_sha256, peak_right.source_step_sha256)
                if sha is not None
            )
        )
        candidates.append(
            SurfCandidateTransition(
                candidate_id=f"SURF_DBL_{pair_num:04d}_{rest_freq:.4f}",
                rest_frequency_mhz=float(rest_freq),
                uncertainty_mhz=float(max(combined_unc, 1e-6)),
                intensity=float(0.5 * (peak_left.intensity + peak_right.intensity)),
                snr=float(min(peak_left.snr, peak_right.snr)),
                feature_kind=SurfFeatureKind.DOPPLER_DOUBLET,
                raw_components=(peak_left, peak_right),
                observed_splitting_mhz=float(obs_sep),
                expected_splitting_mhz=float(exp_sep),
                intensity_asymmetry=float(asym),
                gas_name=canonical_gas,
                beam_velocity_m_s=float(v_beam),
                source_step_sha256=sha_list[0] if sha_list else None,
                source_step_sha256s=sha_list,
                source_file=source_file,
            )
        )

    for idx, peak in enumerate(raw_peaks):
        if idx in paired_indices:
            continue
        is_blend = idx in blend_indices
        kind = SurfFeatureKind.UNRESOLVED_BLEND if is_blend else SurfFeatureKind.SINGLET
        prefix = "SURF_BLD" if is_blend else "SURF_SGL"
        sha_tuple = (peak.source_step_sha256,) if peak.source_step_sha256 else ()
        candidates.append(
            SurfCandidateTransition(
                candidate_id=f"{prefix}_{idx:04d}_{peak.frequency_mhz:.4f}",
                rest_frequency_mhz=peak.frequency_mhz,
                uncertainty_mhz=peak.uncertainty_mhz,
                intensity=peak.intensity,
                snr=peak.snr,
                feature_kind=kind,
                raw_components=(peak,),
                observed_splitting_mhz=None,
                expected_splitting_mhz=float(2.0 * peak.frequency_mhz * (v_beam / SPEED_OF_LIGHT_M_S)),
                intensity_asymmetry=None,
                gas_name=canonical_gas,
                beam_velocity_m_s=float(v_beam),
                source_step_sha256=peak.source_step_sha256,
                source_step_sha256s=sha_tuple,
                source_file=source_file,
            )
        )

    candidates.sort(key=lambda item: item.rest_frequency_mhz)

    return SurfPeakPickResult(
        candidates=tuple(candidates),
        raw_peaks=tuple(raw_peaks),
        baseline_intensity=baseline,
        mad_noise_sigma=sigma_noise,
        snr_threshold=float(snr_threshold),
        gas_name=canonical_gas,
        beam_velocity_m_s=float(v_beam),
        warnings=tuple(warnings),
    )


__all__ = [
    "CARRIER_GAS_BEAM_VELOCITIES_M_S",
    "DEFAULT_RAIL_TOLERANCE_MM",
    "RawSpectralPeak",
    "SPEED_OF_LIGHT_M_S",
    "SURF_CAVITY_RUN_COLUMNS",
    "SURF_SPECTRUM_COLUMNS",
    "StepDirection",
    "SurfBoundsError",
    "SurfCandidateTransition",
    "SurfCavityLookupTable",
    "SurfCavityPeak",
    "SurfCavityRunEntry",
    "SurfCsvSchemaError",
    "SurfFeatureKind",
    "SurfFinalizedCsvData",
    "SurfIngestError",
    "SurfOutputParser",
    "SurfPeakPickResult",
    "SurfReconciliationReport",
    "SurfStepData",
    "ZABER_POS_MAX_MM",
    "ZABER_POS_MIN_MM",
    "compute_bytes_sha256",
    "compute_file_sha256",
    "expected_doppler_splitting_mhz",
    "normalize_carrier_gas",
    "pick_surf_doppler_peaks",
    "resolve_beam_velocity_m_s",
    "slice_surf_step_window",
]
