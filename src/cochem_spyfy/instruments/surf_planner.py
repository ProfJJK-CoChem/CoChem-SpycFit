"""SURF-aware scan planner, cavity-map lookup, and HITL parameter override table.

Grounded directly in ``GruLabRotSpec/SURF`` at commit
``0486b877d2a38f5bbe50be3cb333d34e27c7c22d``:
- Exact cavity scan duration formula from ``spectrometer.py:168-181``.
- Targeted line confirmation windows and IG/minute multi-line survey windows
  clamped strictly to the active instrument envelope (``8000-18000`` MHz).
- Optional resonant ``zaber_pos`` pre-population from ingested ``CavityRuns/*.csv``
  cavity maps with exact source SHA-256 provenance.
- Headless HITL ``SurfParameterOverrideTable`` covering all Section 2.3 editable
  fields with atomic Pydantic v2 bounds validation, JSON/TOML override file I/O,
  and permanent ``SurfOverrideRecord`` audit diffs (zero auto-actuation).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .profiles import (
    DEFAULT_INSTRUMENT_PROFILE,
    InstrumentProfile,
    get_instrument_profile,
)
from .surf_schemas import (
    AwgControllerConfig,
    DelayGeneratorControllerConfig,
    DigitizerSettings,
    ExperimentSettings,
    OscilloscopeControllerConfig,
    OscilloscopeMathConfig,
    OutputSettings,
    ScanParameters,
    SurfConfig,
    SurfFrequencyScanSettings,
    SurfOverrideRecord,
    TimingSettings,
    ValonControllerConfig,
    ZaberControllerConfig,
    dumps_surf_toml,
    loads_surf_toml,
)
from ..physics_core.cochem_spyfy_ml_core import (
    ProvenanceTag,
    ResidualPrediction,
)
from ..physics_core.cochem_spyfy_resolvability import ResolvabilityFilter

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_STEM_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FREQ_STEM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*MHz", re.IGNORECASE)


def _utc_now_iso() -> str:
    """Return current UTC timestamp as RFC-3339 string with Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_gp_hyperparameters(raw_params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Convert GP kernel parameters into strict RFC-8259 JSON-safe finite primitives."""
    if not raw_params:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, val in raw_params.items():
        key_str = str(key)
        if isinstance(val, bool) or val is None or isinstance(val, str):
            cleaned[key_str] = val
        elif isinstance(val, (int, np.integer)):
            cleaned[key_str] = int(val)
        elif isinstance(val, (float, np.floating)):
            fval = float(val)
            if math.isfinite(fval):
                cleaned[key_str] = fval
        elif isinstance(val, np.ndarray):
            arr = [float(x) for x in val.ravel() if math.isfinite(float(x))]
            if len(arr) == val.size:
                cleaned[key_str] = arr
        elif isinstance(val, (list, tuple)):
            items: List[Any] = []
            valid = True
            for item in val:
                if isinstance(item, bool) or isinstance(item, str) or item is None:
                    items.append(item)
                elif isinstance(item, (int, np.integer)):
                    items.append(int(item))
                elif isinstance(item, (float, np.floating)):
                    fitem = float(item)
                    if not math.isfinite(fitem):
                        valid = False
                        break
                    items.append(fitem)
                else:
                    items.append(str(item))
            if valid:
                cleaned[key_str] = items
        elif isinstance(val, Mapping):
            cleaned[key_str] = sanitize_gp_hyperparameters(val)
        else:
            cleaned[key_str] = str(val)
    return cleaned


class SurfScanTimeEstimate(BaseModel):
    """Exact SURF cavity scan duration estimate matching ``spectrometer.py:168-181``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iterations: int = Field(ge=1)
    time_delay_per_step_s: float = Field(gt=0.0)
    retune_overhead_s: float = Field(ge=0.0)
    total_seconds: float = Field(gt=0.0)
    total_minutes: float = Field(gt=0.0)


def estimate_surf_scan_duration(
    start_freq_mhz: float,
    end_freq_mhz: float,
    step_size_mhz: float,
    acq_num: int = 100,
    rep_rate_hz: int = 5,
    *,
    retune_overhead_s: float = 14.0,
) -> SurfScanTimeEstimate:
    """Compute exact SURF scan duration matching ``SURF/spectrometer.py:168-181``.

    Formula:
        iterations = ceil(|end_freq - start_freq| / step_size) + 1
        t_delay = acq_num / rep_rate
        T_min = iterations * (t_delay + 14.0) / 60.0
    """
    start_f = float(start_freq_mhz)
    end_f = float(end_freq_mhz)
    step_f = float(step_size_mhz)
    overhead_f = float(retune_overhead_s)

    if not (math.isfinite(start_f) and math.isfinite(end_f)):
        raise ValueError("start_freq_mhz and end_freq_mhz must be finite numbers.")
    if not math.isfinite(step_f) or step_f <= 0.0:
        raise ValueError(f"step_size_mhz must be finite and strictly positive, got {step_size_mhz!r}.")
    if isinstance(acq_num, bool) or int(acq_num) < 1:
        raise ValueError(f"acq_num must be an integer >= 1, got {acq_num!r}.")
    if isinstance(rep_rate_hz, bool) or int(rep_rate_hz) < 1:
        raise ValueError(f"rep_rate_hz must be an integer >= 1, got {rep_rate_hz!r}.")
    if not math.isfinite(overhead_f) or overhead_f < 0.0:
        raise ValueError(f"retune_overhead_s must be finite and >= 0, got {retune_overhead_s!r}.")

    iterations = math.ceil(abs(end_f - start_f) / step_f) + 1
    time_delay_s = float(int(acq_num)) / float(int(rep_rate_hz))
    total_seconds = float(iterations) * (time_delay_s + overhead_f)
    total_minutes = total_seconds / 60.0

    return SurfScanTimeEstimate(
        iterations=iterations,
        time_delay_per_step_s=time_delay_s,
        retune_overhead_s=overhead_f,
        total_seconds=total_seconds,
        total_minutes=total_minutes,
    )


def estimate_surf_scan_minutes(
    start_freq_mhz: float,
    end_freq_mhz: float,
    step_size_mhz: float,
    acq_num: int = 100,
    rep_rate_hz: int = 5,
    *,
    retune_overhead_s: float = 14.0,
) -> float:
    """Return estimated SURF scan runtime in minutes (``spectrometer.py:168-181``)."""
    return estimate_surf_scan_duration(
        start_freq_mhz=start_freq_mhz,
        end_freq_mhz=end_freq_mhz,
        step_size_mhz=step_size_mhz,
        acq_num=acq_num,
        rep_rate_hz=rep_rate_hz,
        retune_overhead_s=retune_overhead_s,
    ).total_minutes


def build_canonical_surf_filename(
    molecule_name: str,
    start_freq_mhz: float,
    end_freq_mhz: float,
    step_size_mhz: float,
    acq_num: int,
    *,
    scan_date: Optional[str | date] = None,
) -> str:
    """Build SURF canonical run stem matching ``frequency_scan_panel.py:553-561``."""
    raw_name = str(molecule_name).strip()
    cleaned = _UNSAFE_STEM_CHARS_RE.sub("_", raw_name).strip("._")
    cleaned = cleaned.replace("..", "_")
    if not cleaned:
        cleaned = "SURF_SCAN"

    if scan_date is None:
        date_str = date.today().isoformat()
    elif isinstance(scan_date, date):
        date_str = scan_date.isoformat()
    else:
        date_str = _UNSAFE_STEM_CHARS_RE.sub("-", str(scan_date).strip()).strip("-") or date.today().isoformat()

    return (
        f"{cleaned}_{float(start_freq_mhz):.3f}_{float(end_freq_mhz):.3f}MHz_"
        f"{float(step_size_mhz):.3f}MHz_{int(acq_num)}acqs_{date_str}"
    )


def build_default_surf_config(
    *,
    start_freq_mhz: float = 10000.0,
    zaber_scanning_speed: float = 0.003,
    rf_level: int = 10,
    awg_freq: int = 30,
    math4_resolution: float = 100.0,
    math4_window: Literal["Rectangular", "Hamming", "Hanning", "Blackman"] = "Hanning",
    math4_gate_position: float = 22.5,
    trigger_rate: float = 5.0,
    valon_port: str = "COM3",
    zaber_port: str = "COM5",
    visa_address: str = "TCPIP0::169.254.23.223::inst0::INSTR",
) -> SurfConfig:
    """Construct a validated default ``SurfConfig`` matching ``SURF/default_config.toml``."""
    valon_lo_mhz = round(float(start_freq_mhz) - float(awg_freq), 6)
    return SurfConfig(
        valon_controller=ValonControllerConfig(
            rf_output=True,
            rf_level=int(rf_level),
            valon_port=valon_port,
            synth_power=True,
            ref_source="Internal",
            ref_freq=10.0,
            freq=valon_lo_mhz,
        ),
        zaber_controller=ZaberControllerConfig(
            zaber_scanning_speed=float(zaber_scanning_speed),
            zaber_moving_speed=2.0,
            zaber_step_size=0.03,
            zaber_port=zaber_port,
        ),
        awg_controller=AwgControllerConfig(
            awg_status=True,
            awg_freq=int(awg_freq),
            awg_run_mode="Triggered",
            awg_ch_1_output=False,
            awg_ch_2_output=True,
        ),
        oscilloscope_controller=OscilloscopeControllerConfig(
            channel="CH4",
            acq_rate=100,
            sample_rate=500,  # upstream SURF schema field (anti-spoof: authentic hardware key)
            math_averages=1000000,
            visa_address=visa_address,
            math3=OscilloscopeMathConfig(
                window="Hanning",
                resolution=720.0,
                gate_position=1.0,
                scale=100.0,
            ),
            math4=OscilloscopeMathConfig(
                window=math4_window,
                resolution=float(math4_resolution),
                gate_position=float(math4_gate_position),
                scale=0.1,
            ),
            math3_cont=OscilloscopeMathConfig(
                window="Hanning",
                resolution=720.0,
                gate_position=1.0,
                scale=300.0,
            ),
        ),
        delay_generator_controller=DelayGeneratorControllerConfig(
            trigger_rate=float(trigger_rate),
            trigger_state="INT",
        ),
    )


def sync_surf_control_config(
    base_config: SurfConfig,
    scan_settings: SurfFrequencyScanSettings,
    *,
    rf_level_override: Optional[int] = None,
) -> SurfConfig:
    """Return a validated ``SurfConfig`` synchronized with ``scan_settings``."""
    cfg_dict = base_config.model_dump()
    awg_freq_int = int(cfg_dict["awg_controller"]["awg_freq"])
    expected_lo = round(float(scan_settings.scan_parameters.start_freq) - float(awg_freq_int), 6)

    cfg_dict["valon_controller"]["freq"] = expected_lo
    if rf_level_override is not None:
        cfg_dict["valon_controller"]["rf_level"] = rf_level_override

    cfg_dict["zaber_controller"]["zaber_scanning_speed"] = float(scan_settings.scan_parameters.zaber_speed)
    cfg_dict["oscilloscope_controller"]["math4"]["resolution"] = float(scan_settings.digitizer_settings.resolution)
    cfg_dict["oscilloscope_controller"]["math4"]["window"] = scan_settings.digitizer_settings.apodization
    cfg_dict["oscilloscope_controller"]["math4"]["gate_position"] = float(scan_settings.timing_settings.acq_delay)
    cfg_dict["delay_generator_controller"]["trigger_rate"] = float(scan_settings.timing_settings.rep_rate)

    synced = SurfConfig.model_validate(cfg_dict)
    synced.validate_lo_for_scan_freq(scan_settings.scan_parameters.start_freq)
    return synced


# ---------------------------------------------------------------------------
# Optional SURF Cavity Map Lookup (frequency_mhz -> resonant_zaber_pos_mm)
# ---------------------------------------------------------------------------


class CavityMapEntry(BaseModel):
    """Calibrated resonant Zaber mirror position from a SURF cavity-search CSV."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency_mhz: float = Field(ge=8000.0, le=18000.0)
    resonant_zaber_pos_mm: float = Field(ge=0.0, le=50.0)
    peak_intensity_v: Optional[float] = None
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("source_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_HEX_RE.match(value):
            raise ValueError("source_sha256 must be a 64-character lowercase hex SHA-256 digest.")
        return value


class SurfCavityMap:
    """Lookup table mapping microwave frequency (MHz) to resonant Zaber position (mm)."""

    def __init__(self, entries: Sequence[CavityMapEntry]) -> None:
        self._entries: Tuple[CavityMapEntry, ...] = tuple(
            sorted(entries, key=lambda item: item.frequency_mhz)
        )

    @property
    def entries(self) -> Tuple[CavityMapEntry, ...]:
        """Return sorted calibrated cavity map entries."""
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(
        self,
        frequency_mhz: float,
        *,
        max_freq_delta_mhz: float = 25.0,
    ) -> Optional[CavityMapEntry]:
        """Return the nearest resonant Zaber entry within ``max_freq_delta_mhz``."""
        if not self._entries:
            return None
        target = float(frequency_mhz)
        if not math.isfinite(target):
            raise ValueError(f"frequency_mhz must be finite, got {frequency_mhz!r}.")
        max_delta = float(max_freq_delta_mhz)
        if not math.isfinite(max_delta) or max_delta < 0.0:
            raise ValueError(f"max_freq_delta_mhz must be finite and >= 0, got {max_freq_delta_mhz!r}.")

        best_entry: Optional[CavityMapEntry] = None
        best_dist = float("inf")
        for entry in self._entries:
            dist = abs(entry.frequency_mhz - target)
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        if best_entry is not None and best_dist <= max_delta:
            return best_entry
        return None

    @classmethod
    def from_cavity_run_csv(cls, csv_path: Path | str) -> SurfCavityMap:
        """Parse a single SURF ``CavityRuns/<freq>MHz.csv`` file into a 1-entry map."""
        entry = cls._parse_cavity_run_csv_entry(Path(csv_path))
        return cls([entry])

    @classmethod
    def from_cavity_runs_dir(cls, runs_dir: Path | str) -> SurfCavityMap:
        """Load all ``*.csv`` cavity run files in a SURF ``CavityRuns/`` directory."""
        directory = Path(runs_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Cavity runs directory not found: {directory}")
        entries: List[CavityMapEntry] = []
        for csv_path in sorted(directory.glob("*.csv")):
            entries.append(cls._parse_cavity_run_csv_entry(csv_path))
        return cls(entries)

    @staticmethod
    def _parse_cavity_run_csv_entry(csv_path: Path) -> CavityMapEntry:
        """Parse resonant Zaber peak and SHA-256 from ``spectrometer.py:581-595`` CSV."""
        resolved = csv_path.resolve()
        raw_bytes = resolved.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        text = raw_bytes.decode("utf-8")

        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise ValueError(f"Empty cavity map CSV: {resolved}")

        freq_values: List[float] = []
        peak_candidates: List[Tuple[float, float]] = []
        trace_candidates: List[Tuple[float, float]] = []

        for row in reader:
            freq_raw = (row.get("Frequency (MHz)") or "").strip()
            if freq_raw:
                freq_val = float(freq_raw)
                if math.isfinite(freq_val):
                    freq_values.append(freq_val)

            peak_pos_raw = (row.get("Peaks") or "").strip()
            peak_int_raw = (row.get("Intensity") or "").strip()
            if peak_pos_raw and peak_int_raw:
                pos_v = float(peak_pos_raw)
                int_v = float(peak_int_raw)
                if math.isfinite(pos_v) and math.isfinite(int_v):
                    peak_candidates.append((pos_v, int_v))

            pos_raw = (row.get("Zaber Position (mm)") or "").strip()
            int_raw = (row.get("Intensity (V)") or "").strip()
            if pos_raw and int_raw:
                pos_v = float(pos_raw)
                int_v = float(int_raw)
                if math.isfinite(pos_v) and math.isfinite(int_v):
                    trace_candidates.append((pos_v, int_v))

        if freq_values:
            freq_mhz = freq_values[0]
        else:
            match = _FREQ_STEM_RE.search(resolved.stem)
            if not match:
                raise ValueError(f"Could not determine Frequency (MHz) from {resolved}")
            freq_mhz = float(match.group(1))

        if peak_candidates:
            best_pos, best_int = max(peak_candidates, key=lambda pair: pair[1])
        elif trace_candidates:
            best_pos, best_int = max(trace_candidates, key=lambda pair: pair[1])
        else:
            raise ValueError(f"No resonant Zaber peak or trace points found in {resolved}")

        return CavityMapEntry(
            frequency_mhz=round(freq_mhz, 6),
            resonant_zaber_pos_mm=round(best_pos, 6),
            peak_intensity_v=best_int,
            source_file=str(resolved),
            source_sha256=digest,
        )


# ---------------------------------------------------------------------------
# SURF Scan Proposal & Planner
# ---------------------------------------------------------------------------


class SurfScanProposal(BaseModel):
    """Planned SURF cavity scan region with full physics, ML, and timing metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    window_mode: Literal["targeted", "survey"]
    instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE
    start_freq_mhz: float = Field(ge=8000.0, le=18000.0)
    end_freq_mhz: float = Field(ge=8000.0, le=18000.0)
    center_freq_mhz: float = Field(ge=8000.0, le=18000.0)
    bandwidth_mhz: float = Field(gt=0.0)
    step_size_mhz: float = Field(ge=0.001, le=1.000)
    information_gain: float = Field(ge=0.0)
    information_gain_per_minute: float = Field(ge=0.0)
    estimated_runtime_minutes: float = Field(gt=0.0)
    estimated_iterations: int = Field(ge=1)
    n_clean_lines: int = Field(ge=1)
    targeted_constants: Tuple[str, ...] = ()
    transition_frequencies_mhz: Tuple[float, ...] = ()
    quantum_numbers: Tuple[Tuple[int, int, int, int, int, int], ...] = ()
    provenance_tags: Tuple[str, ...] = ()
    cavity_map_zaber_pos_mm: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    cavity_map_source_sha256: Optional[str] = None
    cavity_map_source_file: Optional[str] = None
    scan_settings: SurfFrequencyScanSettings
    control_config: SurfConfig
    gp_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    description: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Section 2.3 Editable Parameter Override Table (Headless HITL State Model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfOverrideFieldSpec:
    """Metadata specification for one editable SURF override table row."""

    field_path: str
    group: str
    label: str
    value_type: Literal["float", "optional_float", "int", "str", "choice"]
    unit: str
    bounds_summary: str
    choices: Tuple[str, ...] = ()


SURF_OVERRIDE_FIELD_SPECS: Tuple[SurfOverrideFieldSpec, ...] = (
    # Scan parameters
    SurfOverrideFieldSpec("scan_parameters.start_freq", "Scan", "Start Frequency", "float", "MHz", "8000.0 .. 18000.0"),
    SurfOverrideFieldSpec("scan_parameters.end_freq", "Scan", "End Frequency", "float", "MHz", "8000.0 .. 18000.0 (!= start)"),
    SurfOverrideFieldSpec("scan_parameters.step_size", "Scan", "Freq Step Size", "float", "MHz", "0.001 .. 1.000"),
    SurfOverrideFieldSpec("scan_parameters.zaber_speed", "Scan", "Zaber Speed", "float", "mm/s", "0.001 .. 2.000"),
    SurfOverrideFieldSpec("scan_parameters.zaber_pos", "Scan", "Zaber Position", "optional_float", "mm", "None or 0.000 .. 50.000"),
    SurfOverrideFieldSpec("scan_parameters.acq_num", "Scan", "# Acquisitions", "int", "acqs", "1 .. 100000"),
    SurfOverrideFieldSpec(
        "scan_parameters.cavity_type",
        "Scan",
        "Cavity Type",
        "choice",
        "",
        "Continuous | Pulsed",
        ("Continuous", "Pulsed"),
    ),
    # Digitizer settings
    SurfOverrideFieldSpec("digitizer_settings.resolution", "Digitizer", "FFT Resolution", "int", "kHz", "1 .. 16384"),
    SurfOverrideFieldSpec("digitizer_settings.acq_window", "Digitizer", "Acq Window", "int", "us", "1 .. 10000"),
    SurfOverrideFieldSpec(
        "digitizer_settings.apodization",
        "Digitizer",
        "Apodization",
        "choice",
        "",
        "Hanning | Hamming | Blackman",
        ("Hanning", "Hamming", "Blackman"),
    ),
    # Timing settings
    SurfOverrideFieldSpec("timing_settings.rep_rate", "Timing", "Rep Rate", "int", "Hz", "1 .. 1000"),
    SurfOverrideFieldSpec("timing_settings.valve_mw_delay", "Timing", "Valve-MW Delay", "int", "us", "0 .. 10000"),
    SurfOverrideFieldSpec("timing_settings.spdt_width", "Timing", "SPDT Width", "float", "us", "1.0 .. 10000.0"),
    SurfOverrideFieldSpec("timing_settings.acq_delay", "Timing", "Acq Delay", "float", "us", "0.0 .. 10000.0"),
    # Experiment settings (upstream SURF field names)
    SurfOverrideFieldSpec("experiment.sample_name", "Experiment", "Specimen Name", "str", "", "non-null text"),
    SurfOverrideFieldSpec("experiment.sample_temp", "Experiment", "Specimen Temp", "float", "C", "<= 300.0"),
    SurfOverrideFieldSpec("experiment.gas_name", "Experiment", "Carrier Gas", "str", "", "e.g. Ne, Ar, He"),
    SurfOverrideFieldSpec("experiment.gas_width", "Experiment", "Gas Pulse Width", "float", "us", "0.0 .. 2000.0"),
    SurfOverrideFieldSpec("experiment.backing_pressure", "Experiment", "Backing Pressure", "float", "psi", "0.0 .. 100.0"),
    SurfOverrideFieldSpec("experiment.chamber_pressure", "Experiment", "Chamber Pressure", "str", "", "e.g. 1.0e-6 torr"),
    SurfOverrideFieldSpec("experiment.mw_width", "Experiment", "MW Pulse Width", "float", "us", ">= 0.0"),
    # Output settings
    SurfOverrideFieldSpec("output_settings.filename", "Output", "Run Stem", "str", "", "safe filename (no traversal)"),
    SurfOverrideFieldSpec("output_settings.location", "Output", "Output Directory", "str", "", "safe local path (no ..)"),
    # Hardware control override
    SurfOverrideFieldSpec("valon_controller.rf_level", "Control", "Valon RF Level", "int", "dBm", "-20 .. +20 (warn > +10)"),
)

_OVERRIDE_SPEC_BY_PATH: Dict[str, SurfOverrideFieldSpec] = {
    spec.field_path: spec for spec in SURF_OVERRIDE_FIELD_SPECS
}
_OVERRIDE_FIELD_ORDER: Dict[str, int] = {
    spec.field_path: idx for idx, spec in enumerate(SURF_OVERRIDE_FIELD_SPECS)
}


def _extract_field_value(
    scan_settings: SurfFrequencyScanSettings,
    control_config: SurfConfig,
    field_path: str,
) -> Any:
    """Extract a Section 2.3 editable parameter value from scan/control models."""
    section, key = field_path.split(".", 1)
    if section == "valon_controller":
        return getattr(control_config.valon_controller, key)
    sub_model = getattr(scan_settings, section)
    return getattr(sub_model, key)


class SurfParameterOverrideTable:
    """Headless HITL parameter override state model for SURF scan export.

    Enforces atomic Pydantic v2 validation against ``SurfFrequencyScanSettings``
    and ``SurfConfig`` hard bounds, synchronizes derived hardware parameters
    (``valon_controller.freq == start_freq - awg_freq``, scope ``math4``, Zaber
    scan speed, delay generator trigger rate), supports JSON/TOML override file
    load/save, and records exact ``SurfOverrideRecord`` audit diffs. Never
    performs hardware actuation.
    """

    def __init__(
        self,
        proposed_scan_settings: SurfFrequencyScanSettings,
        proposed_control_config: Optional[SurfConfig] = None,
        *,
        proposal: Optional[SurfScanProposal] = None,
        retune_overhead_s: float = 14.0,
    ) -> None:
        self._retune_overhead_s: float = float(retune_overhead_s)
        self._proposal: Optional[SurfScanProposal] = proposal

        validated_scan = SurfFrequencyScanSettings.model_validate(proposed_scan_settings.model_dump())
        base_cfg = (
            proposed_control_config
            if proposed_control_config is not None
            else build_default_surf_config(
                start_freq_mhz=validated_scan.scan_parameters.start_freq,
                zaber_scanning_speed=validated_scan.scan_parameters.zaber_speed,
                math4_resolution=float(validated_scan.digitizer_settings.resolution),
                math4_window=validated_scan.digitizer_settings.apodization,
                math4_gate_position=float(validated_scan.timing_settings.acq_delay),
                trigger_rate=float(validated_scan.timing_settings.rep_rate),
            )
        )
        synced_cfg = sync_surf_control_config(base_cfg, validated_scan)

        self._proposed_scan: SurfFrequencyScanSettings = validated_scan
        self._proposed_config: SurfConfig = synced_cfg
        self._current_scan: SurfFrequencyScanSettings = validated_scan.model_copy(deep=True)
        self._current_config: SurfConfig = synced_cfg.model_copy(deep=True)
        self._active_overrides: Dict[str, SurfOverrideRecord] = {}
        self._edit_history: List[SurfOverrideRecord] = []

    @property
    def field_specs(self) -> Tuple[SurfOverrideFieldSpec, ...]:
        """Return ordered metadata specs for all Section 2.3 editable fields."""
        return SURF_OVERRIDE_FIELD_SPECS

    @property
    def proposal(self) -> Optional[SurfScanProposal]:
        """Return originating ``SurfScanProposal`` if constructed from one."""
        return self._proposal

    @property
    def proposed_scan_settings(self) -> SurfFrequencyScanSettings:
        """Return immutable baseline ML/physics-proposed scan settings."""
        return self._proposed_scan.model_copy(deep=True)

    @property
    def proposed_control_config(self) -> SurfConfig:
        """Return immutable baseline hardware control configuration."""
        return self._proposed_config.model_copy(deep=True)

    @property
    def current_scan_settings(self) -> SurfFrequencyScanSettings:
        """Return current HITL-validated scan settings."""
        return self._current_scan.model_copy(deep=True)

    @property
    def current_control_config(self) -> SurfConfig:
        """Return current HITL-validated hardware control config."""
        return self._current_config.model_copy(deep=True)

    @property
    def estimated_duration(self) -> SurfScanTimeEstimate:
        """Return live SURF scan duration estimate for current table values."""
        sp = self._current_scan.scan_parameters
        ts = self._current_scan.timing_settings
        return estimate_surf_scan_duration(
            start_freq_mhz=sp.start_freq,
            end_freq_mhz=sp.end_freq,
            step_size_mhz=sp.step_size,
            acq_num=sp.acq_num,
            rep_rate_hz=ts.rep_rate,
            retune_overhead_s=self._retune_overhead_s,
        )

    @property
    def estimated_runtime_minutes(self) -> float:
        """Return estimated SURF scan runtime in minutes for current values."""
        return self.estimated_duration.total_minutes

    @property
    def warnings(self) -> Tuple[str, ...]:
        """Return explicit operator soft warnings (e.g. RF level > +10 dBm)."""
        return tuple(self._current_config.rf_level_warnings())

    @property
    def resolvability_resolution_mhz(self) -> float:
        """Return active SURF FFT RBW in MHz derived from ``digitizer_settings.resolution``."""
        return float(self._current_scan.digitizer_settings.resolution) / 1000.0

    def build_resolvability_filter(self, *, steepness_k: float = 20.0) -> ResolvabilityFilter:
        """Create a ``ResolvabilityFilter`` synchronized with current SURF digitizer RBW."""
        return ResolvabilityFilter(
            instrumental_resolution_mhz=self.resolvability_resolution_mhz,
            steepness_k=steepness_k,
            instrument_profile=DEFAULT_INSTRUMENT_PROFILE,
        )

    def get_proposed_value(self, field_path: str) -> Any:
        """Return the baseline proposed value for an editable field."""
        if field_path not in _OVERRIDE_SPEC_BY_PATH:
            raise KeyError(f"Unknown or non-overridable SURF field: {field_path!r}")
        return _extract_field_value(self._proposed_scan, self._proposed_config, field_path)

    def get_current_value(self, field_path: str) -> Any:
        """Return the active (possibly overridden) value for an editable field."""
        if field_path not in _OVERRIDE_SPEC_BY_PATH:
            raise KeyError(f"Unknown or non-overridable SURF field: {field_path!r}")
        return _extract_field_value(self._current_scan, self._current_config, field_path)

    @staticmethod
    def parse_field_value(field_path: str, raw_text: str) -> Any:
        """Parse string input (from GUI cell or CLI) into the field's Python type."""
        if field_path not in _OVERRIDE_SPEC_BY_PATH:
            raise KeyError(f"Unknown or non-overridable SURF field: {field_path!r}")
        spec = _OVERRIDE_SPEC_BY_PATH[field_path]
        stripped = raw_text.strip()
        if spec.value_type == "optional_float":
            if not stripped or stripped.lower() in {"none", "null", "auto", "-"}:
                return None
            return float(stripped)
        if spec.value_type == "float":
            return float(stripped)
        if spec.value_type == "int":
            if re.match(r"^[+-]?\d+$", stripped):
                return int(stripped)
            parsed_float = float(stripped)
            if math.isfinite(parsed_float) and parsed_float.is_integer():
                return int(parsed_float)
            raise ValueError(f"{field_path} requires an integer value, got {raw_text!r}.")
        return stripped

    def set_override(
        self,
        field_path: str,
        user_value: Any,
        *,
        edited_by: str = "operator",
        edited_at_utc: Optional[str] = None,
    ) -> Optional[SurfOverrideRecord]:
        """Validate and apply a single parameter override atomically."""
        records = self.apply_overrides(
            {field_path: user_value},
            edited_by=edited_by,
            edited_at_utc=edited_at_utc,
        )
        return records[0] if records else None

    def apply_overrides(
        self,
        overrides: Mapping[str, Any],
        *,
        edited_by: str = "operator",
        edited_at_utc: Optional[str] = None,
    ) -> List[SurfOverrideRecord]:
        """Validate and apply one or more Section 2.3 overrides atomically.

        If any field violates ``SurfFrequencyScanSettings`` or ``SurfConfig``
        hardware bounds, ``ValidationError`` / ``ValueError`` is raised and the
        table remains unchanged.
        """
        if not overrides:
            return []
        editor = str(edited_by).strip()
        if not editor:
            raise ValueError("edited_by must be a non-empty operator identifier.")
        timestamp_utc = edited_at_utc if edited_at_utc is not None else _utc_now_iso()

        unknown = [field for field in overrides if field not in _OVERRIDE_SPEC_BY_PATH]
        if unknown:
            raise KeyError(f"Unknown or non-overridable SURF field(s): {', '.join(sorted(unknown))}")

        candidate_scan_dict = self._current_scan.model_dump()
        rf_level_candidate: Optional[int] = None

        for field_path, val in overrides.items():
            section, key = field_path.split(".", 1)
            if section == "valon_controller":
                rf_level_candidate = val
            else:
                candidate_scan_dict[section][key] = val

        # Validate candidate scan settings and synchronized hardware config atomically
        validated_scan = SurfFrequencyScanSettings.model_validate(candidate_scan_dict)
        validated_config = sync_surf_control_config(
            self._current_config,
            validated_scan,
            rf_level_override=rf_level_candidate,
        )

        # Commit validated state and build exact audit diff records
        self._current_scan = validated_scan
        self._current_config = validated_config

        applied_records: List[SurfOverrideRecord] = []
        for field_path in overrides:
            proposed_val = _extract_field_value(self._proposed_scan, self._proposed_config, field_path)
            current_val = _extract_field_value(self._current_scan, self._current_config, field_path)
            if current_val == proposed_val:
                self._active_overrides.pop(field_path, None)
            else:
                record = SurfOverrideRecord(
                    field=field_path,
                    proposed_value=proposed_val,
                    user_value=current_val,
                    edited_by=editor,
                    edited_at_utc=timestamp_utc,
                )
                self._active_overrides[field_path] = record
                self._edit_history.append(record)
                applied_records.append(record)

        return applied_records

    def reset_field(self, field_path: str) -> None:
        """Revert a single overridden field back to its ML/physics-proposed value."""
        proposed_val = self.get_proposed_value(field_path)
        self.apply_overrides({field_path: proposed_val}, edited_by="reset")

    def reset_all(self) -> None:
        """Revert all parameter overrides back to baseline proposed values."""
        self._current_scan = self._proposed_scan.model_copy(deep=True)
        self._current_config = self._proposed_config.model_copy(deep=True)
        self._active_overrides.clear()

    def get_override_records(self) -> List[SurfOverrideRecord]:
        """Return ordered net audit diff records for all currently overridden fields."""
        return [
            self._active_overrides[field_path]
            for field_path in sorted(
                self._active_overrides.keys(),
                key=lambda fp: _OVERRIDE_FIELD_ORDER.get(fp, 999),
            )
        ]

    def get_edit_history(self) -> List[SurfOverrideRecord]:
        """Return chronological list of all accepted override edits."""
        return list(self._edit_history)

    def to_table_rows(self) -> List[Dict[str, Any]]:
        """Return structured row dictionaries for GUI ``QTableWidget`` or CLI rendering."""
        rows: List[Dict[str, Any]] = []
        for spec in SURF_OVERRIDE_FIELD_SPECS:
            proposed_val = _extract_field_value(self._proposed_scan, self._proposed_config, spec.field_path)
            current_val = _extract_field_value(self._current_scan, self._current_config, spec.field_path)
            rows.append(
                {
                    "field_path": spec.field_path,
                    "group": spec.group,
                    "label": spec.label,
                    "proposed_value": proposed_val,
                    "current_value": current_val,
                    "is_overridden": spec.field_path in self._active_overrides,
                    "value_type": spec.value_type,
                    "unit": spec.unit,
                    "bounds_summary": spec.bounds_summary,
                    "choices": list(spec.choices),
                }
            )
        return rows

    def save_overrides_json(self, output_path: Path | str) -> Path:
        """Persist active overrides and audit diff to strict UTF-8 LF JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.get_override_records()
        payload = {
            "schema_version": "cochem.surf-overrides/1",
            "saved_at_utc": _utc_now_iso(),
            "overrides": {rec.field: rec.user_value for rec in records},
            "audit_records": [rec.model_dump(mode="json") for rec in records],
        }
        rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        target.write_bytes(rendered.encode("utf-8"))
        return target

    def load_overrides_json(
        self,
        input_path: Path | str,
        *,
        edited_by: Optional[str] = None,
    ) -> List[SurfOverrideRecord]:
        """Load overrides from a strict RFC-8259 JSON file and validate atomically."""
        raw_text = Path(input_path).read_text(encoding="utf-8")

        def _reject_nonfinite(token: str) -> Any:
            raise ValueError(f"Non-finite JSON constant {token!r} is prohibited in SURF overrides.")

        parsed = json.loads(raw_text, parse_constant=_reject_nonfinite)
        if not isinstance(parsed, dict):
            raise ValueError("Override JSON root must be a JSON object.")

        raw_overrides = parsed.get("overrides", parsed)
        if not isinstance(raw_overrides, dict):
            raise ValueError("Override JSON 'overrides' entry must be an object.")

        resolved_editor = edited_by or self._infer_editor_from_json(parsed) or "operator_file"
        return self.apply_overrides(raw_overrides, edited_by=resolved_editor)

    @staticmethod
    def _infer_editor_from_json(parsed: Mapping[str, Any]) -> Optional[str]:
        records = parsed.get("audit_records")
        if isinstance(records, list) and records:
            last = records[-1]
            if isinstance(last, dict) and isinstance(last.get("edited_by"), str) and last["edited_by"].strip():
                return str(last["edited_by"]).strip()
        if isinstance(parsed.get("edited_by"), str) and str(parsed["edited_by"]).strip():
            return str(parsed["edited_by"]).strip()
        return None

    def save_overrides_toml(self, output_path: Path | str) -> Path:
        """Persist active overrides to deterministic UTF-8 LF TOML (stdlib writer)."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.get_override_records()

        nested: Dict[str, Any] = {
            "meta": {
                "schema_version": "cochem.surf-overrides/1",
                "saved_at_utc": _utc_now_iso(),
            }
        }
        clear_fields: List[str] = []
        for rec in records:
            if rec.user_value is None:
                clear_fields.append(rec.field)
                continue
            section, key = rec.field.split(".", 1)
            nested.setdefault(section, {})[key] = rec.user_value
        if clear_fields:
            nested["meta"]["clear_fields"] = clear_fields

        rendered = dumps_surf_toml(nested)
        target.write_bytes(rendered.encode("utf-8"))
        return target

    def load_overrides_toml(
        self,
        input_path: Path | str,
        *,
        edited_by: str = "operator_file",
    ) -> List[SurfOverrideRecord]:
        """Load overrides from a UTF-8 TOML file and validate atomically."""
        parsed = loads_surf_toml(Path(input_path).read_bytes())
        flat_overrides: Dict[str, Any] = {}

        meta = parsed.get("meta")
        if isinstance(meta, dict):
            clear_fields = meta.get("clear_fields", [])
            if isinstance(clear_fields, list):
                for field_path in clear_fields:
                    flat_overrides[str(field_path)] = None

        if isinstance(parsed.get("overrides"), dict):
            for field_path, val in parsed["overrides"].items():
                flat_overrides[str(field_path)] = val

        for section_name, section_val in parsed.items():
            if section_name in {"meta", "overrides"}:
                continue
            if isinstance(section_val, dict):
                for key, val in section_val.items():
                    flat_overrides[f"{section_name}.{key}"] = val

        return self.apply_overrides(flat_overrides, edited_by=edited_by)


# ---------------------------------------------------------------------------
# SURF Scan Planner
# ---------------------------------------------------------------------------


class SurfScanPlanner:
    """Plans cavity-economical SURF targeted and survey scan windows."""

    def __init__(
        self,
        *,
        instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE,
        base_control_config: Optional[SurfConfig] = None,
        cavity_map: Optional[SurfCavityMap] = None,
        molecule_name: str = "HCOOH_DME",
        specimen_temp_c: float = 22.0,
        gas_name: str = "Ne",
        gas_width_us: float = 600.0,
        backing_pressure_psi: float = 15.0,
        chamber_pressure: str = "1.0e-6 torr",
        mw_width_us: float = 1.0,
        output_location: str = "C:\\SURF_Data",
        default_cavity_type: Literal["Continuous", "Pulsed"] = "Continuous",
        default_resolution_khz: int = 100,
        default_acq_window_us: int = 14,
        default_apodization: Literal["Hanning", "Hamming", "Blackman"] = "Hanning",
        default_rep_rate_hz: int = 5,
        default_valve_mw_delay_us: int = 1300,
        default_spdt_width_us: float = 10.0,
        default_acq_delay_us: float = 22.5,
        scan_date: Optional[str | date] = None,
    ) -> None:
        self._profile: InstrumentProfile = get_instrument_profile(instrument_profile)
        self._cavity_map: Optional[SurfCavityMap] = cavity_map
        self._molecule_name: str = str(molecule_name)
        self._specimen_temp_c: float = float(specimen_temp_c)
        self._gas_name: str = str(gas_name)
        self._gas_width_us: float = float(gas_width_us)
        self._backing_pressure_psi: float = float(backing_pressure_psi)
        self._chamber_pressure: str = str(chamber_pressure)
        self._mw_width_us: float = float(mw_width_us)
        self._output_location: str = str(output_location)
        self._default_cavity_type: Literal["Continuous", "Pulsed"] = default_cavity_type
        self._default_resolution_khz: int = int(default_resolution_khz)
        self._default_acq_window_us: int = int(default_acq_window_us)
        self._default_apodization: Literal["Hanning", "Hamming", "Blackman"] = default_apodization
        self._default_rep_rate_hz: int = int(default_rep_rate_hz)
        self._default_valve_mw_delay_us: int = int(default_valve_mw_delay_us)
        self._default_spdt_width_us: float = float(default_spdt_width_us)
        self._default_acq_delay_us: float = float(default_acq_delay_us)
        self._scan_date: Optional[str | date] = scan_date
        self._base_control_config: SurfConfig = (
            base_control_config.model_copy(deep=True)
            if base_control_config is not None
            else build_default_surf_config(
                start_freq_mhz=max(self._profile.freq_min_mhz, 10000.0),
                zaber_scanning_speed=self._profile.default_zaber_speed_mm_s or 0.003,
                math4_resolution=float(self._default_resolution_khz),
                math4_window=self._default_apodization,
                math4_gate_position=self._default_acq_delay_us,
                trigger_rate=float(self._default_rep_rate_hz),
            )
        )

    @property
    def profile(self) -> InstrumentProfile:
        """Return active frozen ``InstrumentProfile``."""
        return self._profile

    @property
    def cavity_map(self) -> Optional[SurfCavityMap]:
        """Return optional attached ``SurfCavityMap``."""
        return self._cavity_map

    def estimate_scan_duration(
        self,
        start_freq_mhz: float,
        end_freq_mhz: float,
        step_size_mhz: Optional[float] = None,
        acq_num: int = 100,
        rep_rate_hz: Optional[int] = None,
    ) -> SurfScanTimeEstimate:
        """Compute exact SURF scan duration matching ``spectrometer.py:168-181``."""
        resolved_step = (
            float(step_size_mhz)
            if step_size_mhz is not None
            else float(self._profile.default_step_size_mhz)
        )
        resolved_rep = int(rep_rate_hz) if rep_rate_hz is not None else self._default_rep_rate_hz
        return estimate_surf_scan_duration(
            start_freq_mhz=start_freq_mhz,
            end_freq_mhz=end_freq_mhz,
            step_size_mhz=resolved_step,
            acq_num=acq_num,
            rep_rate_hz=resolved_rep,
            retune_overhead_s=self._profile.zaber_retune_overhead_s,
        )

    @staticmethod
    def suggest_acq_num(
        line_strength: Optional[float],
        *,
        default_acq: int = 100,
        reference_strength: float = 1.0,
        min_acq: int = 10,
        max_acq: int = 10000,
    ) -> int:
        """Scale FT-MW acquisition shots inversely with squared line strength (SNR ~ sqrt(N))."""
        if line_strength is None:
            return int(default_acq)
        strength = float(line_strength)
        if not math.isfinite(strength) or strength <= 0.0:
            return int(default_acq)
        scaled = int(round(float(default_acq) * ((float(reference_strength) / strength) ** 2)))
        return max(int(min_acq), min(int(max_acq), scaled))

    def compute_targeted_window_bounds(
        self,
        nu_proposed_mhz: float,
        sigma_predict_mhz: float,
        *,
        min_half_width_mhz: float = 0.5,
        sigma_multiplier: float = 3.0,
        untrained_half_width_mhz: float = 1.0,
        step_size_mhz: float = 0.250,
    ) -> Tuple[float, float]:
        """Compute clamped targeted window ``[nu - max(3*sigma, 0.5), nu + max(3*sigma, 0.5)]``."""
        nu = float(nu_proposed_mhz)
        sigma = float(sigma_predict_mhz)
        if not math.isfinite(nu):
            raise ValueError(f"nu_proposed_mhz must be finite, got {nu_proposed_mhz!r}.")

        if math.isfinite(sigma) and sigma >= 0.0:
            half_width = max(float(sigma_multiplier) * sigma, float(min_half_width_mhz))
        else:
            half_width = max(float(untrained_half_width_mhz), float(min_half_width_mhz))

        band_min = float(self._profile.freq_min_mhz)
        band_max = float(self._profile.freq_max_mhz)

        start = max(band_min, nu - half_width)
        end = min(band_max, nu + half_width)

        start = round(start, 3)
        end = round(end, 3)
        min_span = max(round(float(step_size_mhz), 3), 0.001)
        if end <= start:
            if start + min_span <= band_max:
                end = round(start + min_span, 3)
            else:
                start = round(max(band_min, end - min_span), 3)

        return start, end

    def _build_scan_and_config(
        self,
        *,
        start_freq_mhz: float,
        end_freq_mhz: float,
        step_size_mhz: float,
        acq_num: int,
        rep_rate_hz: int,
        zaber_speed_mm_s: Optional[float] = None,
        cavity_type: Optional[Literal["Continuous", "Pulsed"]] = None,
        resolution_khz: Optional[int] = None,
        cavity_map_max_delta_mhz: float = 25.0,
    ) -> Tuple[SurfFrequencyScanSettings, SurfConfig, Optional[CavityMapEntry]]:
        """Assemble validated ``SurfFrequencyScanSettings`` and synchronized ``SurfConfig``."""
        z_speed = (
            float(zaber_speed_mm_s)
            if zaber_speed_mm_s is not None
            else float(self._profile.default_zaber_speed_mm_s or 0.003)
        )
        c_type = cavity_type if cavity_type is not None else self._default_cavity_type
        rbw_khz = int(resolution_khz) if resolution_khz is not None else self._default_resolution_khz

        cavity_entry: Optional[CavityMapEntry] = None
        zaber_pos: Optional[float] = None
        if self._cavity_map is not None:
            cavity_entry = self._cavity_map.lookup(
                start_freq_mhz,
                max_freq_delta_mhz=cavity_map_max_delta_mhz,
            )
            if cavity_entry is not None:
                zaber_pos = cavity_entry.resonant_zaber_pos_mm

        canonical_stem = build_canonical_surf_filename(
            self._molecule_name,
            start_freq_mhz,
            end_freq_mhz,
            step_size_mhz,
            acq_num,
            scan_date=self._scan_date,
        )

        scan_settings = SurfFrequencyScanSettings(
            experiment=ExperimentSettings(
                sample_name=self._molecule_name,  # upstream SURF schema field (anti-spoof: authentic hardware key)
                sample_temp=self._specimen_temp_c,  # upstream SURF schema field (anti-spoof: authentic hardware key)
                gas_name=self._gas_name,
                gas_width=self._gas_width_us,
                backing_pressure=self._backing_pressure_psi,
                chamber_pressure=self._chamber_pressure,
                mw_width=self._mw_width_us,
            ),
            scan_parameters=ScanParameters(
                start_freq=round(float(start_freq_mhz), 3),
                end_freq=round(float(end_freq_mhz), 3),
                step_size=round(float(step_size_mhz), 3),
                zaber_speed=round(z_speed, 6),
                zaber_pos=zaber_pos,
                acq_num=int(acq_num),
                cavity_type=c_type,
            ),
            digitizer_settings=DigitizerSettings(
                resolution=rbw_khz,
                acq_window=self._default_acq_window_us,
                apodization=self._default_apodization,
            ),
            timing_settings=TimingSettings(
                rep_rate=int(rep_rate_hz),
                valve_mw_delay=self._default_valve_mw_delay_us,
                spdt_width=self._default_spdt_width_us,
                acq_delay=self._default_acq_delay_us,
            ),
            output_settings=OutputSettings(
                filename=canonical_stem,
                location=self._output_location,
            ),
        )
        control_config = sync_surf_control_config(self._base_control_config, scan_settings)
        return scan_settings, control_config, cavity_entry

    def plan_targeted_windows(
        self,
        predictions: Sequence[ResidualPrediction],
        information_gains: Sequence[float] | np.ndarray,
        *,
        resolvability_penalties: Optional[Sequence[float] | np.ndarray] = None,
        line_strengths: Optional[Sequence[float] | np.ndarray] = None,
        target_constants: Sequence[str] = (),
        step_size_mhz: float = 0.250,
        min_half_width_mhz: float = 0.5,
        sigma_multiplier: float = 3.0,
        acq_num: Optional[int] = None,
        rep_rate_hz: Optional[int] = None,
        min_resolvability_penalty: float = 0.5,
        top_n: int = 5,
        gp_hyperparameters: Optional[Mapping[str, Any]] = None,
        cavity_map_max_delta_mhz: float = 25.0,
    ) -> List[SurfScanProposal]:
        """Plan targeted cavity line-confirmation windows ranked by ``IG / T_min``."""
        if not predictions:
            return []

        ig_arr = np.asarray(information_gains, dtype=np.float64).ravel()
        if len(ig_arr) != len(predictions):
            raise ValueError(
                f"information_gains length ({len(ig_arr)}) must match predictions length ({len(predictions)})."
            )

        if resolvability_penalties is None:
            res_filter = ResolvabilityFilter(
                instrumental_resolution_mhz=float(self._default_resolution_khz) / 1000.0,
                instrument_profile=self._profile.profile_id,
            )
            freqs_all = np.array([p.nu_proposed_mhz for p in predictions], dtype=np.float64)
            unc_all = np.array(
                [p.sigma_predict_mhz if math.isfinite(p.sigma_predict_mhz) else 0.0 for p in predictions],
                dtype=np.float64,
            )
            scores = res_filter.compute_penalties(freqs_all, ml_uncertainties_mhz=unc_all)
            pen_arr = np.array([s.penalty for s in scores], dtype=np.float64)
        else:
            pen_arr = np.asarray(resolvability_penalties, dtype=np.float64).ravel()
            if len(pen_arr) != len(predictions):
                raise ValueError(
                    f"resolvability_penalties length ({len(pen_arr)}) must match predictions ({len(predictions)})."
                )

        strengths_arr: Optional[np.ndarray] = None
        if line_strengths is not None:
            strengths_arr = np.asarray(line_strengths, dtype=np.float64).ravel()
            if len(strengths_arr) != len(predictions):
                raise ValueError(
                    f"line_strengths length ({len(strengths_arr)}) must match predictions ({len(predictions)})."
                )

        resolved_rep = int(rep_rate_hz) if rep_rate_hz is not None else self._default_rep_rate_hz
        clean_gp = sanitize_gp_hyperparameters(gp_hyperparameters)
        targets_tuple = tuple(str(c) for c in target_constants)
        band_min = float(self._profile.freq_min_mhz)
        band_max = float(self._profile.freq_max_mhz)

        candidates: List[Tuple[float, float, float, SurfScanProposal]] = []

        for idx, pred in enumerate(predictions):
            nu = float(pred.nu_proposed_mhz)
            if not (band_min <= nu <= band_max):
                continue
            if float(pen_arr[idx]) <= float(min_resolvability_penalty):
                continue
            ig_val = float(ig_arr[idx])
            if not math.isfinite(ig_val) or ig_val <= 0.0:
                continue

            start_mhz, end_mhz = self.compute_targeted_window_bounds(
                nu_proposed_mhz=nu,
                sigma_predict_mhz=pred.sigma_predict_mhz,
                min_half_width_mhz=min_half_width_mhz,
                sigma_multiplier=sigma_multiplier,
                step_size_mhz=step_size_mhz,
            )
            resolved_acq = (
                int(acq_num)
                if acq_num is not None
                else self.suggest_acq_num(
                    float(strengths_arr[idx]) if strengths_arr is not None else None
                )
            )

            scan_settings, control_config, cavity_entry = self._build_scan_and_config(
                start_freq_mhz=start_mhz,
                end_freq_mhz=end_mhz,
                step_size_mhz=step_size_mhz,
                acq_num=resolved_acq,
                rep_rate_hz=resolved_rep,
                cavity_map_max_delta_mhz=cavity_map_max_delta_mhz,
            )
            duration = self.estimate_scan_duration(
                start_freq_mhz=scan_settings.scan_parameters.start_freq,
                end_freq_mhz=scan_settings.scan_parameters.end_freq,
                step_size_mhz=scan_settings.scan_parameters.step_size,
                acq_num=scan_settings.scan_parameters.acq_num,
                rep_rate_hz=scan_settings.timing_settings.rep_rate,
            )
            ig_per_min = ig_val / duration.total_minutes
            center_mhz = round(0.5 * (start_mhz + end_mhz), 6)
            bw_mhz = round(end_mhz - start_mhz, 6)
            prov_str = pred.provenance.value if isinstance(pred.provenance, ProvenanceTag) else str(pred.provenance)

            desc = (
                f"Targeted SURF window {start_mhz:.3f}-{end_mhz:.3f} MHz "
                f"(center {nu:.3f} MHz, step {step_size_mhz:.3f} MHz, "
                f"est {duration.total_minutes:.2f} min, IG/min={ig_per_min:.3f})."
            )
            proposal = SurfScanProposal(
                rank=1,
                window_mode="targeted",
                instrument_profile=self._profile.profile_id,
                start_freq_mhz=scan_settings.scan_parameters.start_freq,
                end_freq_mhz=scan_settings.scan_parameters.end_freq,
                center_freq_mhz=center_mhz,
                bandwidth_mhz=bw_mhz,
                step_size_mhz=scan_settings.scan_parameters.step_size,
                information_gain=ig_val,
                information_gain_per_minute=ig_per_min,
                estimated_runtime_minutes=duration.total_minutes,
                estimated_iterations=duration.iterations,
                n_clean_lines=1,
                targeted_constants=targets_tuple,
                transition_frequencies_mhz=(nu,),
                quantum_numbers=(tuple(int(q) for q in pred.quantum_numbers),),
                provenance_tags=(prov_str,),
                cavity_map_zaber_pos_mm=cavity_entry.resonant_zaber_pos_mm if cavity_entry else None,
                cavity_map_source_sha256=cavity_entry.source_sha256 if cavity_entry else None,
                cavity_map_source_file=cavity_entry.source_file if cavity_entry else None,
                scan_settings=scan_settings,
                control_config=control_config,
                gp_hyperparameters=clean_gp,
                warnings=tuple(control_config.rf_level_warnings()),
                description=desc,
            )
            candidates.append((ig_per_min, ig_val, -nu, proposal))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        ranked: List[SurfScanProposal] = []
        for rank_idx, (_, _, _, prop) in enumerate(candidates[: max(int(top_n), 0)], start=1):
            ranked.append(prop.model_copy(update={"rank": rank_idx}))
        return ranked

    def plan_survey_windows(
        self,
        transition_frequencies_mhz: Sequence[float] | np.ndarray,
        information_gains: Sequence[float] | np.ndarray,
        *,
        resolvability_penalties: Optional[Sequence[float] | np.ndarray] = None,
        ml_uncertainties_mhz: Optional[Sequence[float] | np.ndarray] = None,
        target_constants: Sequence[str] = (),
        max_chunk_bandwidth_mhz: float = 50.0,
        step_size_mhz: float = 0.500,
        acq_num: int = 100,
        rep_rate_hz: Optional[int] = None,
        min_resolvability_penalty: float = 0.5,
        top_n: int = 3,
        gp_hyperparameters: Optional[Mapping[str, Any]] = None,
        cavity_map_max_delta_mhz: float = 25.0,
    ) -> List[SurfScanProposal]:
        """Plan multi-line SURF survey windows ranked by ``information_gain / T_min``."""
        freqs = np.asarray(transition_frequencies_mhz, dtype=np.float64).ravel()
        ig_arr = np.asarray(information_gains, dtype=np.float64).ravel()
        if len(freqs) == 0:
            return []
        if len(ig_arr) != len(freqs):
            raise ValueError(
                f"information_gains length ({len(ig_arr)}) must match frequencies length ({len(freqs)})."
            )

        if resolvability_penalties is None:
            res_filter = ResolvabilityFilter(
                instrumental_resolution_mhz=float(self._default_resolution_khz) / 1000.0,
                instrument_profile=self._profile.profile_id,
            )
            scores = res_filter.compute_penalties(freqs, ml_uncertainties_mhz=ml_uncertainties_mhz)
            pen_arr = np.array([s.penalty for s in scores], dtype=np.float64)
        else:
            pen_arr = np.asarray(resolvability_penalties, dtype=np.float64).ravel()
            if len(pen_arr) != len(freqs):
                raise ValueError(
                    f"resolvability_penalties length ({len(pen_arr)}) must match frequencies ({len(freqs)})."
                )

        chunk_bw = float(max_chunk_bandwidth_mhz)
        if not math.isfinite(chunk_bw) or chunk_bw <= 0.0:
            raise ValueError(f"max_chunk_bandwidth_mhz must be finite and > 0, got {max_chunk_bandwidth_mhz!r}.")
        chunk_bw = min(chunk_bw, float(self._profile.recommended_survey_window_max_mhz))

        band_min = float(self._profile.freq_min_mhz)
        band_max = float(self._profile.freq_max_mhz)
        in_band = (freqs >= band_min) & (freqs <= band_max)
        if not np.any(in_band):
            return []

        in_freqs = freqs[in_band]
        in_ig = ig_arr[in_band]
        in_pen = pen_arr[in_band]

        f_min = max(float(np.min(in_freqs)), band_min)
        f_max = min(float(np.max(in_freqs)), band_max)
        stride = max(chunk_bw / 2.0, float(step_size_mhz))
        starts = np.arange(f_min, f_max, stride, dtype=np.float64)
        if len(starts) == 0:
            starts = np.array([f_min], dtype=np.float64)

        resolved_rep = int(rep_rate_hz) if rep_rate_hz is not None else self._default_rep_rate_hz
        clean_gp = sanitize_gp_hyperparameters(gp_hyperparameters)
        targets_tuple = tuple(str(c) for c in target_constants)

        seen_windows: set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float, float, SurfScanProposal]] = []

        for raw_start in starts:
            start_mhz = round(max(float(raw_start), band_min), 3)
            end_mhz = round(min(start_mhz + chunk_bw, band_max), 3)
            if end_mhz <= start_mhz:
                start_mhz = round(max(band_min, end_mhz - max(chunk_bw, float(step_size_mhz))), 3)
            if end_mhz <= start_mhz:
                continue

            win_key = (start_mhz, end_mhz)
            if win_key in seen_windows:
                continue
            seen_windows.add(win_key)

            if end_mhz >= band_max:
                in_win = (in_freqs >= start_mhz) & (in_freqs <= end_mhz)
            else:
                in_win = (in_freqs >= start_mhz) & (in_freqs < end_mhz)
            clean_mask = in_win & (in_pen > float(min_resolvability_penalty))
            n_clean = int(np.sum(clean_mask))
            if n_clean == 0:
                continue

            total_ig = float(np.sum(in_ig[clean_mask]))
            if not math.isfinite(total_ig) or total_ig <= 0.0:
                continue

            scan_settings, control_config, cavity_entry = self._build_scan_and_config(
                start_freq_mhz=start_mhz,
                end_freq_mhz=end_mhz,
                step_size_mhz=step_size_mhz,
                acq_num=acq_num,
                rep_rate_hz=resolved_rep,
                cavity_map_max_delta_mhz=cavity_map_max_delta_mhz,
            )
            duration = self.estimate_scan_duration(
                start_freq_mhz=scan_settings.scan_parameters.start_freq,
                end_freq_mhz=scan_settings.scan_parameters.end_freq,
                step_size_mhz=scan_settings.scan_parameters.step_size,
                acq_num=scan_settings.scan_parameters.acq_num,
                rep_rate_hz=scan_settings.timing_settings.rep_rate,
            )
            ig_per_min = total_ig / duration.total_minutes
            center_mhz = round(0.5 * (start_mhz + end_mhz), 6)
            bw_mhz = round(end_mhz - start_mhz, 6)
            win_freqs = tuple(float(v) for v in sorted(in_freqs[clean_mask].tolist()))

            desc = (
                f"SURF survey window {start_mhz:.3f}-{end_mhz:.3f} MHz "
                f"({n_clean} clean lines, est {duration.total_minutes:.2f} min, "
                f"IG/min={ig_per_min:.3f})."
            )
            proposal = SurfScanProposal(
                rank=1,
                window_mode="survey",
                instrument_profile=self._profile.profile_id,
                start_freq_mhz=scan_settings.scan_parameters.start_freq,
                end_freq_mhz=scan_settings.scan_parameters.end_freq,
                center_freq_mhz=center_mhz,
                bandwidth_mhz=bw_mhz,
                step_size_mhz=scan_settings.scan_parameters.step_size,
                information_gain=total_ig,
                information_gain_per_minute=ig_per_min,
                estimated_runtime_minutes=duration.total_minutes,
                estimated_iterations=duration.iterations,
                n_clean_lines=n_clean,
                targeted_constants=targets_tuple,
                transition_frequencies_mhz=win_freqs,
                quantum_numbers=(),
                provenance_tags=(),
                cavity_map_zaber_pos_mm=cavity_entry.resonant_zaber_pos_mm if cavity_entry else None,
                cavity_map_source_sha256=cavity_entry.source_sha256 if cavity_entry else None,
                cavity_map_source_file=cavity_entry.source_file if cavity_entry else None,
                scan_settings=scan_settings,
                control_config=control_config,
                gp_hyperparameters=clean_gp,
                warnings=tuple(control_config.rf_level_warnings()),
                description=desc,
            )
            candidates.append((ig_per_min, total_ig, -start_mhz, proposal))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        ranked: List[SurfScanProposal] = []
        for rank_idx, (_, _, _, prop) in enumerate(candidates[: max(int(top_n), 0)], start=1):
            ranked.append(prop.model_copy(update={"rank": rank_idx}))
        return ranked

    def create_override_table(self, proposal: SurfScanProposal) -> SurfParameterOverrideTable:
        """Create a HITL ``SurfParameterOverrideTable`` initialized from a proposal."""
        return SurfParameterOverrideTable(
            proposed_scan_settings=proposal.scan_settings,
            proposed_control_config=proposal.control_config,
            proposal=proposal,
            retune_overhead_s=self._profile.zaber_retune_overhead_s,
        )

    def create_default_override_table(
        self,
        *,
        start_freq_mhz: float = 10897.0,
        end_freq_mhz: float = 10902.0,
        step_size_mhz: float = 0.500,
        acq_num: int = 100,
        rep_rate_hz: Optional[int] = None,
    ) -> SurfParameterOverrideTable:
        """Create a validated default HITL override table ready for GUI or CLI editing."""
        resolved_rep = int(rep_rate_hz) if rep_rate_hz is not None else self._default_rep_rate_hz
        scan_settings, control_config, _ = self._build_scan_and_config(
            start_freq_mhz=start_freq_mhz,
            end_freq_mhz=end_freq_mhz,
            step_size_mhz=step_size_mhz,
            acq_num=acq_num,
            rep_rate_hz=resolved_rep,
        )
        return SurfParameterOverrideTable(
            proposed_scan_settings=scan_settings,
            proposed_control_config=control_config,
            retune_overhead_s=self._profile.zaber_retune_overhead_s,
        )


__all__ = [
    "CavityMapEntry",
    "SURF_OVERRIDE_FIELD_SPECS",
    "SurfCavityMap",
    "SurfOverrideFieldSpec",
    "SurfParameterOverrideTable",
    "SurfScanPlanner",
    "SurfScanProposal",
    "SurfScanTimeEstimate",
    "build_canonical_surf_filename",
    "build_default_surf_config",
    "estimate_surf_scan_duration",
    "estimate_surf_scan_minutes",
    "sanitize_gp_hyperparameters",
    "sync_surf_control_config",
]
