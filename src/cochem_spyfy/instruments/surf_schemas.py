"""Pydantic v2 schemas and pure-Python TOML serializer for GruLabRotSpec/SURF.

All models mirror upstream ``GruLabRotSpec/SURF`` at commit
``0486b877d2a38f5bbe50be3cb333d34e27c7c22d`` (``version = "0.1.0"``) with
strict hardware safety bounds and zero third-party TOML writer dependencies.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cochem_spyfy.instruments.profiles import (
    DEFAULT_INSTRUMENT_PROFILE,
    surf_compatibility_metadata,
)

_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _coerce_strict_surf_int(value: Any, *, field_name: str) -> int:
    """Accept Python int or exact whole-number float from SURF TOML; reject bool/fractional/str."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be a whole integer, got {value!r}.")
    raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}.")


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    """Accept finite int or float; reject bool, NaN, Inf, and non-numeric values."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number, not bool.")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite, got {value!r}.")
        return numeric
    raise ValueError(f"{field_name} must be a finite number, got {type(value).__name__}.")


def _validate_safe_filename(value: str, *, field_name: str = "filename") -> str:
    """Reject empty names, path separators, drive prefixes, null bytes, and '..' traversal."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes.")
    if value in {".", ".."} or ".." in value:
        raise ValueError(f"{field_name} must not contain path traversal segments ('..').")
    if "/" in value or "\\" in value or ":" in value:
        raise ValueError(f"{field_name} must be a single filename without directory separators or drive prefixes.")
    if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
        raise ValueError(f"{field_name} must resolve to a plain filename stem.")
    return value


def _validate_safe_location(value: str, *, field_name: str = "location") -> str:
    """Reject null bytes and parent-directory traversal ('..') in output paths."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes.")
    parts = re.split(r"[\\/]+", value)
    if any(part == ".." for part in parts):
        raise ValueError(f"{field_name} must not contain parent-directory traversal ('..').")
    return value


def _validate_no_nonfinite_json_value(value: Any, *, field_name: str) -> Any:
    """Recursively verify that nested JSON payloads contain no NaN or Inf floats."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats (got {value!r}).")
        return value
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{field_name} dictionary keys must be strings.")
            _validate_no_nonfinite_json_value(v, field_name=f"{field_name}.{k}")
        return value
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _validate_no_nonfinite_json_value(item, field_name=f"{field_name}[{idx}]")
        return value
    return value


def _escape_toml_string(value: str) -> str:
    """Format a Python string as a TOML v1.0 basic quoted string."""
    escaped: List[str] = ['"']
    for ch in value:
        code = ord(ch)
        if ch == "\\":
            escaped.append("\\\\")
        elif ch == '"':
            escaped.append('\\"')
        elif ch == "\b":
            escaped.append("\\b")
        elif ch == "\t":
            escaped.append("\\t")
        elif ch == "\n":
            escaped.append("\\n")
        elif ch == "\f":
            escaped.append("\\f")
        elif ch == "\r":
            escaped.append("\\r")
        elif code < 0x20 or code == 0x7F:
            escaped.append(f"\\u{code:04x}")
        else:
            escaped.append(ch)
    escaped.append('"')
    return "".join(escaped)


def _format_toml_key(key: str) -> str:
    """Format a TOML key segment as bare when valid, else quoted."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"TOML keys must be non-empty strings, got {key!r}.")
    if _TOML_BARE_KEY_RE.match(key):
        return key
    return _escape_toml_string(key)


def _format_toml_value(value: Any) -> str:
    """Format a TOML scalar or inline array value deterministically."""
    if value is None:
        raise ValueError("TOML does not support null/None values; omit optional keys before serializing.")
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float {value!r} is prohibited in SURF TOML payloads.")
        rendered = repr(value)
        if "." not in rendered and "e" not in rendered and "E" not in rendered:
            rendered = f"{rendered}.0"
        return rendered
    if isinstance(value, str):
        return _escape_toml_string(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_toml_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, Mapping):
        pairs = ", ".join(
            f"{_format_toml_key(k)} = {_format_toml_value(v)}"
            for k, v in value.items()
        )
        return "{ " + pairs + " }" if pairs else "{}"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _emit_toml_table(
    table: Mapping[str, Any],
    prefix: Tuple[str, ...],
    blocks: List[str],
) -> None:
    """Emit scalar entries for a table followed recursively by child sub-tables."""
    scalar_items: List[Tuple[str, Any]] = []
    subtable_items: List[Tuple[str, Mapping[str, Any]]] = []

    for key, val in table.items():
        if val is None:
            raise ValueError(
                f"TOML key {'.'.join((*prefix, key))!r} has value None; omit unset optional keys."
            )
        if isinstance(val, Mapping):
            subtable_items.append((key, val))
        else:
            scalar_items.append((key, val))

    if prefix:
        # Emit header when table has scalar entries or is an explicit empty leaf table.
        if scalar_items or not subtable_items:
            header = ".".join(_format_toml_key(segment) for segment in prefix)
            lines = [f"[{header}]"]
            for key, val in scalar_items:
                lines.append(f"{_format_toml_key(key)} = {_format_toml_value(val)}")
            blocks.append("\n".join(lines))
    elif scalar_items:
        lines = [
            f"{_format_toml_key(key)} = {_format_toml_value(val)}"
            for key, val in scalar_items
        ]
        blocks.append("\n".join(lines))

    for key, subtable in subtable_items:
        _emit_toml_table(subtable, (*prefix, key), blocks)


def dumps_surf_toml(data: Mapping[str, Any]) -> str:
    """Serialize a nested mapping to deterministic UTF-8 LF TOML without third-party writers."""
    if not isinstance(data, Mapping):
        raise TypeError(f"dumps_surf_toml expects a mapping, got {type(data).__name__}.")
    blocks: List[str] = []
    _emit_toml_table(data, (), blocks)
    if not blocks:
        return "\n"
    return "\n\n".join(blocks) + "\n"


def loads_surf_toml(text: str | bytes) -> Dict[str, Any]:
    """Parse UTF-8 TOML text or bytes using Python stdlib ``tomllib``."""
    if isinstance(text, bytes):
        if text.startswith(b"\xef\xbb\xbf"):
            raise ValueError("SURF TOML input must be UTF-8 without BOM.")
        decoded = text.decode("utf-8")
    elif isinstance(text, str):
        if text.startswith("\ufeff"):
            raise ValueError("SURF TOML input must be UTF-8 without BOM.")
        decoded = text
    else:
        raise TypeError(f"loads_surf_toml expects str or bytes, got {type(text).__name__}.")
    return tomllib.loads(decoded)


def check_rf_level_soft_warning(
    rf_level: int,
    *,
    soft_limit_dbm: int = 10,
) -> Optional[str]:
    """Return an explicit operator warning when Valon RF power exceeds ``soft_limit_dbm``."""
    checked = _coerce_strict_surf_int(rf_level, field_name="rf_level")
    if checked > soft_limit_dbm:
        return (
            f"Valon rf_level={checked} dBm exceeds recommended soft limit of "
            f"+{soft_limit_dbm} dBm (hardware max +20 dBm)."
        )
    return None


# ---------------------------------------------------------------------------
# Hardware Config Models (SURF/src/spectrometer_gui/config.py:8-75)
# ---------------------------------------------------------------------------


class ValonControllerConfig(BaseModel):
    """Mirrors ``ValonConfig`` in ``SURF/src/spectrometer_gui/config.py:8-16``."""

    model_config = ConfigDict(extra="forbid")

    rf_output: bool
    rf_level: int = Field(ge=-20, le=20)
    valon_port: str = Field(min_length=1)
    synth_power: bool
    ref_source: Literal["Internal", "External"]
    ref_freq: float = Field(gt=0.0)
    freq: float = Field(ge=7970.0, le=18000.0)

    @field_validator("rf_level", mode="before")
    @classmethod
    def _validate_rf_level_int(cls, value: Any) -> int:
        return _coerce_strict_surf_int(value, field_name="rf_level")

    @field_validator("ref_freq", "freq", mode="before")
    @classmethod
    def _validate_valon_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))

    def rf_level_warnings(self) -> List[str]:
        """Return soft-warning messages if RF output power exceeds +10 dBm."""
        warning = check_rf_level_soft_warning(self.rf_level)
        return [warning] if warning is not None else []


class ZaberControllerConfig(BaseModel):
    """Mirrors ``ZaberConfig`` in ``SURF/src/spectrometer_gui/config.py:18-23``."""

    model_config = ConfigDict(extra="forbid")

    zaber_scanning_speed: float = Field(ge=0.001, le=2.0)
    zaber_moving_speed: float = Field(gt=0.0, le=10.0)
    zaber_step_size: float = Field(gt=0.0, le=50.0)
    zaber_port: str = Field(min_length=1)

    @field_validator("zaber_scanning_speed", "zaber_moving_speed", "zaber_step_size", mode="before")
    @classmethod
    def _validate_zaber_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))


class AwgControllerConfig(BaseModel):
    """Mirrors ``AWGConfig`` in ``SURF/src/spectrometer_gui/config.py:25-31``."""

    model_config = ConfigDict(extra="forbid")

    awg_status: bool
    awg_freq: int = Field(gt=0)
    awg_run_mode: str = Field(min_length=1)
    awg_ch_1_output: bool
    awg_ch_2_output: bool

    @field_validator("awg_freq", mode="before")
    @classmethod
    def _validate_awg_freq_int(cls, value: Any) -> int:
        return _coerce_strict_surf_int(value, field_name="awg_freq")


class OscilloscopeMathConfig(BaseModel):
    """Mirrors ``MathConfig`` in ``SURF/src/spectrometer_gui/config.py:33-38``."""

    model_config = ConfigDict(extra="forbid")

    window: Literal["Rectangular", "Hamming", "Hanning", "Blackman"]
    resolution: float = Field(gt=0.0)
    gate_position: float = Field(ge=0.0)
    scale: float = Field(gt=0.0)

    @field_validator("resolution", "gate_position", "scale", mode="before")
    @classmethod
    def _validate_math_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))


class OscilloscopeControllerConfig(BaseModel):
    """Mirrors ``OscilloscopeConfig`` in ``SURF/src/spectrometer_gui/config.py:40-49``."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(min_length=1)
    acq_rate: int = Field(ge=1)
    sample_rate: int = Field(ge=1)  # upstream SURF schema field (anti-spoof: authentic hardware key)
    math_averages: int = Field(ge=1)
    visa_address: str = Field(min_length=1)
    math3: OscilloscopeMathConfig
    math4: OscilloscopeMathConfig
    math3_cont: OscilloscopeMathConfig

    @field_validator("acq_rate", "sample_rate", "math_averages", mode="before")
    @classmethod
    def _validate_scope_ints(cls, value: Any, info: Any) -> int:
        return _coerce_strict_surf_int(value, field_name=str(info.field_name))


class DelayGeneratorControllerConfig(BaseModel):
    """Mirrors ``DelayGeneratorConfig`` in ``SURF/src/spectrometer_gui/config.py:51-53``."""

    model_config = ConfigDict(extra="forbid")

    trigger_rate: float = Field(ge=1.0, le=1000.0)
    trigger_state: Literal["INT", "EXT"]

    @field_validator("trigger_rate", mode="before")
    @classmethod
    def _validate_trigger_rate(cls, value: Any) -> float:
        return _coerce_finite_float(value, field_name="trigger_rate")


class SurfConfig(BaseModel):
    """Mirrors root ``Config`` in ``SURF/src/spectrometer_gui/config.py:55-61``."""

    model_config = ConfigDict(extra="forbid")

    valon_controller: ValonControllerConfig
    zaber_controller: ZaberControllerConfig
    awg_controller: AwgControllerConfig
    oscilloscope_controller: OscilloscopeControllerConfig
    delay_generator_controller: DelayGeneratorControllerConfig

    def rf_level_warnings(self) -> List[str]:
        """Return any RF power soft warnings for this hardware configuration."""
        return self.valon_controller.rf_level_warnings()

    def validate_lo_for_scan_freq(self, scan_freq_mhz: float) -> float:
        """Compute and validate ``Valon LO = scan_freq_mhz - awg_freq >= 7970.0`` MHz."""
        freq = _coerce_finite_float(scan_freq_mhz, field_name="scan_freq_mhz")
        lo_mhz = freq - float(self.awg_controller.awg_freq)
        if lo_mhz < 7970.0 or lo_mhz > 18000.0:
            raise ValueError(
                f"Computed Valon LO {lo_mhz:.3f} MHz (scan_freq={freq:.3f} - "
                f"awg_freq={self.awg_controller.awg_freq}) is outside [7970.0, 18000.0] MHz."
            )
        return lo_mhz

    def to_surf_toml(self) -> str:
        """Serialize hardware configuration to native SURF TOML."""
        return dumps_surf_toml(self.model_dump())

    @classmethod
    def from_surf_toml(cls, text: str | bytes) -> SurfConfig:
        """Parse and validate native SURF hardware configuration TOML."""
        return cls.model_validate(loads_surf_toml(text))

    @classmethod
    def from_surf_toml_file(cls, path: Path | str) -> SurfConfig:
        """Load and validate a native SURF hardware configuration TOML file."""
        return cls.from_surf_toml(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# App Settings Models (SURF/src/spectrometer_gui/settings.py:7-59)
# ---------------------------------------------------------------------------


class SurfOutputConfig(BaseModel):
    """Mirrors ``OutputConfig`` in ``SURF/src/spectrometer_gui/settings.py:7-9``."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(min_length=1)
    filename: str = Field(min_length=1)

    @field_validator("location")
    @classmethod
    def _check_location(cls, value: str) -> str:
        return _validate_safe_location(value, field_name="output.location")

    @field_validator("filename")
    @classmethod
    def _check_filename(cls, value: str) -> str:
        return _validate_safe_filename(value, field_name="output.filename")


class SurfLoggingConfig(BaseModel):
    """Mirrors ``LoggingConfig`` in ``SURF/src/spectrometer_gui/settings.py:12-14``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    location: str = Field(min_length=1)

    @field_validator("location")
    @classmethod
    def _check_location(cls, value: str) -> str:
        return _validate_safe_location(value, field_name="logging.location")


class SurfScopePresetItem(BaseModel):
    """Mirrors ``ScopePresetItem`` in ``SURF/src/spectrometer_gui/settings.py:17-19``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class SurfScopePreset(BaseModel):
    """Mirrors ``ScopePreset`` in ``SURF/src/spectrometer_gui/settings.py:22-24``."""

    model_config = ConfigDict(extra="forbid")

    root_path: str
    presets: Dict[str, SurfScopePresetItem] = Field(default_factory=dict)


class SurfAnalysisSettings(BaseModel):
    """Mirrors ``AnalysisSettings`` in ``SURF/src/spectrometer_gui/settings.py:27-30``."""

    model_config = ConfigDict(extra="forbid")

    show_points: bool = False
    show_grid: bool = True
    show_crosshair: bool = True


class SurfSettings(BaseModel):
    """Mirrors root ``Settings`` in ``SURF/src/spectrometer_gui/settings.py:32-38``."""

    model_config = ConfigDict(extra="forbid")

    output: SurfOutputConfig
    logging: SurfLoggingConfig
    scope_preset: SurfScopePreset
    theme: str = "auto"
    analysis: SurfAnalysisSettings

    def to_surf_toml(self) -> str:
        """Serialize application settings to native SURF TOML."""
        return dumps_surf_toml(self.model_dump())

    @classmethod
    def from_surf_toml(cls, text: str | bytes) -> SurfSettings:
        """Parse and validate native SURF application settings TOML."""
        return cls.model_validate(loads_surf_toml(text))

    @classmethod
    def from_surf_toml_file(cls, path: Path | str) -> SurfSettings:
        """Load and validate a native SURF application settings TOML file."""
        return cls.from_surf_toml(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Frequency Scan Settings (SURF/src/spectrometer_gui/frequency_scan_settings.py:4-52)
# ---------------------------------------------------------------------------


class ExperimentSettings(BaseModel):
    """Mirrors ``Experiment`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:4-12``."""

    model_config = ConfigDict(extra="forbid")

    sample_name: str  # upstream SURF schema field (anti-spoof: authentic hardware key)
    sample_temp: float = Field(le=300.0)  # upstream SURF schema field (anti-spoof: authentic hardware key)
    gas_name: str
    gas_width: float = Field(ge=0.0, le=2000.0)
    backing_pressure: float = Field(ge=0.0, le=100.0)
    chamber_pressure: str
    mw_width: float = Field(ge=0.0)

    @field_validator("sample_temp", "gas_width", "backing_pressure", "mw_width", mode="before")
    @classmethod
    def _validate_experiment_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))


class ScanParameters(BaseModel):
    """Mirrors ``ScanParameters`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:15-23``."""

    model_config = ConfigDict(extra="forbid")

    start_freq: float = Field(ge=8000.0, le=18000.0)
    end_freq: float = Field(ge=8000.0, le=18000.0)
    step_size: float = Field(ge=0.001, le=1.000)
    zaber_speed: float = Field(ge=0.001, le=2.0)
    zaber_pos: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    acq_num: int = Field(ge=1, le=100000)
    cavity_type: Literal["Continuous", "Pulsed"]

    @field_validator("start_freq", "end_freq", "step_size", "zaber_speed", mode="before")
    @classmethod
    def _validate_scan_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))

    @field_validator("zaber_pos", mode="before")
    @classmethod
    def _validate_optional_zaber_pos(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        return _coerce_finite_float(value, field_name="zaber_pos")

    @field_validator("acq_num", mode="before")
    @classmethod
    def _validate_acq_num_int(cls, value: Any) -> int:
        return _coerce_strict_surf_int(value, field_name="acq_num")

    @model_validator(mode="after")
    def _validate_non_degenerate_range(self) -> ScanParameters:
        if self.start_freq == self.end_freq:
            raise ValueError("start_freq and end_freq must not be equal.")
        return self


class DigitizerSettings(BaseModel):
    """Mirrors ``DigitizerSettings`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:26-30``."""

    model_config = ConfigDict(extra="forbid")

    resolution: int = Field(ge=1, le=16384)
    acq_window: int = Field(ge=1, le=10000)
    apodization: Literal["Hanning", "Hamming", "Blackman"]

    @field_validator("resolution", "acq_window", mode="before")
    @classmethod
    def _validate_digitizer_ints(cls, value: Any, info: Any) -> int:
        return _coerce_strict_surf_int(value, field_name=str(info.field_name))


class TimingSettings(BaseModel):
    """Mirrors ``TimingSettings`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:33-38``."""

    model_config = ConfigDict(extra="forbid")

    rep_rate: int = Field(ge=1, le=1000)
    valve_mw_delay: int = Field(ge=0, le=10000)
    spdt_width: float = Field(ge=1.0, le=10000.0)
    acq_delay: float = Field(ge=0.0, le=10000.0)

    @field_validator("rep_rate", "valve_mw_delay", mode="before")
    @classmethod
    def _validate_timing_ints(cls, value: Any, info: Any) -> int:
        return _coerce_strict_surf_int(value, field_name=str(info.field_name))

    @field_validator("spdt_width", "acq_delay", mode="before")
    @classmethod
    def _validate_timing_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))


class OutputSettings(BaseModel):
    """Mirrors ``OutputSettings`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:40-43``."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    location: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def _check_filename(cls, value: str) -> str:
        return _validate_safe_filename(value, field_name="output_settings.filename")

    @field_validator("location")
    @classmethod
    def _check_location(cls, value: str) -> str:
        return _validate_safe_location(value, field_name="output_settings.location")


class SurfFrequencyScanSettings(BaseModel):
    """Mirrors ``FrequencyScanSettings`` in ``SURF/src/spectrometer_gui/frequency_scan_settings.py:46-52``."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentSettings
    scan_parameters: ScanParameters
    digitizer_settings: DigitizerSettings
    timing_settings: TimingSettings
    output_settings: OutputSettings

    def estimated_runtime_minutes(self, *, retune_overhead_s: float = 14.0) -> float:
        """Estimate cavity scan duration in minutes matching ``SURF/spectrometer.py:168-181``."""
        iterations = math.ceil(
            abs(self.scan_parameters.end_freq - self.scan_parameters.start_freq)
            / self.scan_parameters.step_size
        ) + 1
        time_delay_s = float(self.scan_parameters.acq_num) / float(self.timing_settings.rep_rate)
        return (iterations * (time_delay_s + retune_overhead_s)) / 60.0

    def to_surf_dict(self) -> Dict[str, Any]:
        """Convert to SURF-compatible dict, omitting ``zaber_pos`` when ``None``."""
        payload = self.model_dump()
        if payload["scan_parameters"].get("zaber_pos") is None:
            payload["scan_parameters"].pop("zaber_pos", None)
        return payload

    def to_surf_toml(self) -> str:
        """Serialize frequency scan settings to native SURF ``scan_settings.toml``."""
        return dumps_surf_toml(self.to_surf_dict())

    @classmethod
    def from_surf_toml(cls, text: str | bytes) -> SurfFrequencyScanSettings:
        """Parse and validate a native SURF ``scan_settings.toml`` payload."""
        return cls.model_validate(loads_surf_toml(text))

    @classmethod
    def from_surf_toml_file(cls, path: Path | str) -> SurfFrequencyScanSettings:
        """Load and validate a native SURF ``scan_settings.toml`` file."""
        return cls.from_surf_toml(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Cavity Search Settings (SURF/src/spectrometer_gui/cavity_search_settings.py:4-9)
# ---------------------------------------------------------------------------


class SurfCavitySearchType(str, Enum):
    """Mirrors ``CavitySearchType`` in ``SURF/src/spectrometer_gui/gui/signal_enums.py:10-12``."""

    CONTINUOUS = "continuous"
    PULSED = "pulsed"


class SurfCavitySearchSettings(BaseModel):
    """Mirrors ``CavitySearchSettings`` in ``SURF/src/spectrometer_gui/cavity_search_settings.py:4-9``."""

    model_config = ConfigDict(extra="forbid")

    cavity_type: SurfCavitySearchType
    freq: float = Field(ge=8000.0, le=18000.0)
    step_size: float = Field(ge=0.001, le=1.000)
    zaber_speed: float = Field(ge=0.001, le=5.0)

    @field_validator("freq", "step_size", "zaber_speed", mode="before")
    @classmethod
    def _validate_search_floats(cls, value: Any, info: Any) -> float:
        return _coerce_finite_float(value, field_name=str(info.field_name))

    def to_surf_toml(self) -> str:
        """Serialize cavity search settings to native SURF ``search_settings.toml``."""
        payload = self.model_dump(mode="json")
        return dumps_surf_toml(payload)

    @classmethod
    def from_surf_toml(cls, text: str | bytes) -> SurfCavitySearchSettings:
        """Parse and validate native SURF ``search_settings.toml`` payload."""
        return cls.model_validate(loads_surf_toml(text))

    @classmethod
    def from_surf_toml_file(cls, path: Path | str) -> SurfCavitySearchSettings:
        """Load and validate a native SURF ``search_settings.toml`` file."""
        return cls.from_surf_toml(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# SpycFit HITL Override Audit & Scan Manifest Provenance
# ---------------------------------------------------------------------------


class SurfOverrideRecord(BaseModel):
    """Auditable record of a human override on an ML-proposed SURF scan parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    proposed_value: Any
    user_value: Any
    edited_by: str = Field(min_length=1)
    edited_at_utc: str = Field(min_length=1)

    @field_validator("edited_at_utc")
    @classmethod
    def _validate_iso8601_timestamp(cls, value: str) -> str:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        datetime.fromisoformat(normalized)
        return value

    @model_validator(mode="after")
    def _validate_finite_values(self) -> SurfOverrideRecord:
        _validate_no_nonfinite_json_value(self.proposed_value, field_name="proposed_value")
        _validate_no_nonfinite_json_value(self.user_value, field_name="user_value")
        return self


class SurfScanManifest(BaseModel):
    """Strict JSON provenance manifest linking SpycFit fit state to exported SURF TOMLs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "cochem.surf-scan-manifest/1"
    instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE
    created_at_utc: str = Field(min_length=1)
    fit_state_id: Optional[str] = None
    spcat_catalog_sha256: Optional[str] = None
    control_config_filename: str = Field(min_length=1)
    control_config_sha256: str = Field(min_length=64, max_length=64)
    scan_settings_filename: str = Field(min_length=1)
    scan_settings_sha256: str = Field(min_length=64, max_length=64)
    estimated_runtime_minutes: float = Field(gt=0.0)
    recommendation_rank: Optional[int] = Field(default=None, ge=1)
    recommendation_score: Optional[float] = None
    target_constants: Tuple[str, ...] = ()
    gp_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    overrides: Tuple[SurfOverrideRecord, ...] = ()
    warnings: Tuple[str, ...] = ()
    surf_compatibility: Dict[str, str] = Field(default_factory=surf_compatibility_metadata)

    @field_validator("created_at_utc")
    @classmethod
    def _validate_created_at_utc(cls, value: str) -> str:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        datetime.fromisoformat(normalized)
        return value

    @field_validator("control_config_filename", "scan_settings_filename")
    @classmethod
    def _validate_manifest_filenames(cls, value: str, info: Any) -> str:
        return _validate_safe_filename(value, field_name=str(info.field_name))

    @field_validator("control_config_sha256", "scan_settings_sha256")
    @classmethod
    def _validate_required_sha256(cls, value: str, info: Any) -> str:
        if not _SHA256_HEX_RE.match(value):
            raise ValueError(f"{info.field_name} must be a 64-character lowercase hex SHA-256 digest.")
        return value

    @field_validator("spcat_catalog_sha256")
    @classmethod
    def _validate_optional_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _SHA256_HEX_RE.match(value):
            raise ValueError("spcat_catalog_sha256 must be a 64-character lowercase hex SHA-256 digest.")
        return value

    @field_validator("estimated_runtime_minutes", mode="before")
    @classmethod
    def _validate_runtime_float(cls, value: Any) -> float:
        return _coerce_finite_float(value, field_name="estimated_runtime_minutes")

    @field_validator("recommendation_score", mode="before")
    @classmethod
    def _validate_optional_score(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        return _coerce_finite_float(value, field_name="recommendation_score")

    @field_validator("gp_hyperparameters")
    @classmethod
    def _validate_gp_hyperparameters(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        _validate_no_nonfinite_json_value(value, field_name="gp_hyperparameters")
        return value

    def to_strict_json(self, *, indent: int = 2) -> str:
        """Serialize manifest to strict RFC-8259 JSON (``allow_nan=False``)."""
        return json.dumps(self.model_dump(mode="json"), indent=indent, allow_nan=False) + "\n"

    @classmethod
    def from_strict_json(cls, text: str) -> SurfScanManifest:
        """Parse strict RFC-8259 JSON (rejecting NaN/Infinity constants) into a manifest."""
        def _reject_nonfinite(token: str) -> Any:
            raise ValueError(f"Non-finite JSON constant {token!r} is prohibited in SurfScanManifest.")

        parsed = json.loads(text, parse_constant=_reject_nonfinite)
        return cls.model_validate(parsed)


__all__ = [
    "AwgControllerConfig",
    "DelayGeneratorControllerConfig",
    "DigitizerSettings",
    "ExperimentSettings",
    "OscilloscopeControllerConfig",
    "OscilloscopeMathConfig",
    "OutputSettings",
    "ScanParameters",
    "SurfAnalysisSettings",
    "SurfCavitySearchSettings",
    "SurfCavitySearchType",
    "SurfConfig",
    "SurfFrequencyScanSettings",
    "SurfLoggingConfig",
    "SurfOutputConfig",
    "SurfOverrideRecord",
    "SurfScanManifest",
    "SurfScopePreset",
    "SurfScopePresetItem",
    "SurfSettings",
    "TimingSettings",
    "ValonControllerConfig",
    "ZaberControllerConfig",
    "check_rf_level_soft_warning",
    "dumps_surf_toml",
    "loads_surf_toml",
]
