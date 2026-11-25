"""Hardware safety interlocks, envelope validation, and HITL gate for GruLab SURF.

Enforces Section 4.3 of ``docs/SURF_PRIMARY_INTEGRATION_PLAN.md`` for both
CoChem-SPYFY and the Tier-2 loopback control bridge:

1. Zero autonomous unattended RF or gas-valve actuation (mandatory two-step
   ``stage`` -> interactive operator confirmation -> ``authorize_start``).
2. Dual-validated hard physical bounds across Valon LO, Zaber travel/speed,
   oscilloscope digitizer, SRS delay generator timing, and RF power.
3. Pre-flight device online and single-task concurrency gating.
4. Unconditional emergency stop / abort with explicit cleanup sequence.
5. Dead-man heartbeat expiration cancelling staged/paused work without ever
   auto-starting hardware.
6. Strict loopback-only host binding and constant-time bearer token auth.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cochem_spyfy.instruments.profiles import GRULAB_SURF_PROFILE
from cochem_spyfy.instruments.surf_schemas import (
    SurfConfig,
    SurfFrequencyScanSettings,
    check_rf_level_soft_warning,
)

REQUIRED_SURF_DEVICES: Tuple[str, ...] = (
    "valon",
    "zaber",
    "oscilloscope",
    "delay_generator",
    "switch",
    "awg",
)

REQUIRED_CLEANUP_SEQUENCE: Tuple[str, ...] = (
    "delay_generator.stop_pulse",
    "delay_generator.stop_trig",
    "oscilloscope.stop_acq",
    "zaber.home",
)

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

VALID_CAVITY_MODES: frozenset[str] = frozenset({"Continuous", "Pulsed"})
VALID_SCAN_APODIZATIONS: frozenset[str] = frozenset({"Hanning", "Hamming", "Blackman"})


class SafetyInterlockError(Exception):
    """Raised when a mandatory SURF hardware safety interlock blocks an action."""

    def __init__(self, message: str, *, code: str = "safety_interlock") -> None:
        super().__init__(message)
        self.code = code


class BridgeSecurityError(Exception):
    """Raised when loopback binding or bearer authentication policy is violated."""

    def __init__(self, message: str, *, code: str = "bridge_security") -> None:
        super().__init__(message)
        self.code = code


class HardwareBoundsError(SafetyInterlockError):
    """Raised when any SURF scan or hardware parameter violates physical bounds."""

    def __init__(self, message: str, *, code: str = "hardware_bounds") -> None:
        super().__init__(message, code=code)


class DeviceOfflineError(SafetyInterlockError):
    """Raised when one or more required SURF hardware controllers are offline."""

    def __init__(
        self,
        message: str,
        *,
        offline_devices: Tuple[str, ...] = (),
        code: str = "device_offline",
    ) -> None:
        super().__init__(message, code=code)
        self.offline_devices = offline_devices


def _utc_now(clock: Optional[Callable[[], datetime]] = None) -> datetime:
    """Return timezone-aware UTC timestamp from injected or wall clock."""
    now = datetime.now(timezone.utc) if clock is None else clock()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _format_utc(dt: datetime) -> str:
    """Format a UTC datetime as RFC-3339 / ISO-8601 string with trailing ``Z``."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_strict_int(value: Any, *, field_name: str) -> int:
    """Verify value is a strict integer (rejecting bool, float, and strings)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareBoundsError(
            f"{field_name} must be an integer, got {value!r} ({type(value).__name__})."
        )
    return value


def _require_finite_number(value: Any, *, field_name: str) -> float:
    """Verify value is a finite real number (rejecting bool, NaN, and Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareBoundsError(
            f"{field_name} must be a finite number, got {value!r} ({type(value).__name__})."
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise HardwareBoundsError(f"{field_name} must be finite, got {value!r}.")
    return numeric


def validate_loopback_host(host: str) -> str:
    """Permit strictly ``127.0.0.1``, ``::1``, or ``localhost``; reject all others."""
    if not isinstance(host, str):
        raise BridgeSecurityError(
            f"Loopback host must be a string, got {type(host).__name__}."
        )
    normalized = host.strip().lower()
    if not normalized or normalized not in ALLOWED_LOOPBACK_HOSTS:
        raise BridgeSecurityError(
            f"Host {host!r} is prohibited; only loopback hosts "
            f"('127.0.0.1', '::1', 'localhost') are permitted."
        )
    return normalized


def generate_bearer_token(nbytes: int = 32) -> str:
    """Generate a high-entropy URL-safe bearer token for loopback RPC auth."""
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 16:
        raise BridgeSecurityError("Bearer token entropy must be at least 16 bytes.")
    return secrets.token_urlsafe(nbytes)


def validate_bearer_token(provided_token: Optional[str], expected_token: str) -> None:
    """Verify bearer token in constant time using ``hmac.compare_digest``."""
    if not isinstance(expected_token, str) or not expected_token:
        raise BridgeSecurityError("Expected bearer token must be a non-empty string.")
    if not isinstance(provided_token, str) or not provided_token:
        raise BridgeSecurityError("Missing bearer authentication token.")
    if not hmac.compare_digest(
        provided_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        raise BridgeSecurityError("Invalid bearer authentication token.")


def compute_scan_settings_sha256(
    scan_settings: SurfFrequencyScanSettings,
    control_config: Optional[SurfConfig] = None,
) -> str:
    """Compute canonical SHA-256 hex digest over scan settings and optional config."""
    envelope: Dict[str, Any] = {
        "scan_settings": scan_settings.to_surf_dict(),
        "control_config": None if control_config is None else control_config.model_dump(),
    }
    canonical = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class HardwareEnvelopeReport(BaseModel):
    """Result of dual hardware-bounds validation for a SURF scan proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool = True
    start_freq_mhz: float
    end_freq_mhz: float
    valon_lo_start_mhz: float
    valon_lo_end_mhz: float
    awg_freq_mhz: float
    estimated_runtime_minutes: float = Field(gt=0.0)
    warnings: Tuple[str, ...] = ()
    settings_sha256: str = Field(min_length=64, max_length=64)


def _coerce_scan_settings(
    scan_settings: SurfFrequencyScanSettings | Mapping[str, Any],
) -> SurfFrequencyScanSettings:
    """Parse mapping into ``SurfFrequencyScanSettings``, converting schema errors."""
    if isinstance(scan_settings, SurfFrequencyScanSettings):
        return scan_settings
    if not isinstance(scan_settings, Mapping):
        raise HardwareBoundsError(
            f"scan_settings must be SurfFrequencyScanSettings or mapping, "
            f"got {type(scan_settings).__name__}."
        )
    try:
        return SurfFrequencyScanSettings.model_validate(dict(scan_settings))
    except ValidationError as exc:
        raise HardwareBoundsError(f"SURF scan settings failed schema bounds: {exc}") from exc


def _coerce_control_config(
    control_config: Optional[SurfConfig | Mapping[str, Any]],
) -> Optional[SurfConfig]:
    """Parse optional mapping into ``SurfConfig``, converting schema errors."""
    if control_config is None or isinstance(control_config, SurfConfig):
        return control_config
    if not isinstance(control_config, Mapping):
        raise HardwareBoundsError(
            f"control_config must be SurfConfig, mapping, or None, "
            f"got {type(control_config).__name__}."
        )
    try:
        return SurfConfig.model_validate(dict(control_config))
    except ValidationError as exc:
        raise HardwareBoundsError(f"SURF control config failed schema bounds: {exc}") from exc


def validate_surf_hardware_envelope(
    scan_settings: SurfFrequencyScanSettings | Mapping[str, Any],
    control_config: Optional[SurfConfig | Mapping[str, Any]] = None,
    *,
    awg_freq_mhz: Optional[float] = None,
) -> HardwareEnvelopeReport:
    """Independently enforce every Section 4.3 SURF hardware physical bound.

    Performs dual validation even if a model was constructed via
    ``model_construct`` or mutated after parsing, and checks cross-field
    Valon LO bounds (``start_freq - awg_freq >= 7970.0`` MHz and
    ``end_freq - awg_freq >= 7970.0`` MHz).
    """
    validated_scan = _coerce_scan_settings(scan_settings)
    validated_config = _coerce_control_config(control_config)

    sp = validated_scan.scan_parameters
    dig = validated_scan.digitizer_settings
    tim = validated_scan.timing_settings
    exp = validated_scan.experiment

    start_freq = _require_finite_number(sp.start_freq, field_name="scan_parameters.start_freq")
    end_freq = _require_finite_number(sp.end_freq, field_name="scan_parameters.end_freq")
    if not (8000.0 <= start_freq <= 18000.0):
        raise HardwareBoundsError(
            f"start_freq={start_freq} MHz violates [8000.0, 18000.0] MHz envelope."
        )
    if not (8000.0 <= end_freq <= 18000.0):
        raise HardwareBoundsError(
            f"end_freq={end_freq} MHz violates [8000.0, 18000.0] MHz envelope."
        )
    if start_freq == end_freq:
        raise HardwareBoundsError(
            f"start_freq ({start_freq} MHz) and end_freq ({end_freq} MHz) must not be equal."
        )

    step_size = _require_finite_number(sp.step_size, field_name="scan_parameters.step_size")
    if not (0.001 <= step_size <= 1.0):
        raise HardwareBoundsError(
            f"step_size={step_size} MHz violates [0.001, 1.0] MHz envelope."
        )

    zaber_speed = _require_finite_number(sp.zaber_speed, field_name="scan_parameters.zaber_speed")
    if not (0.001 <= zaber_speed <= 2.0):
        raise HardwareBoundsError(
            f"zaber_speed={zaber_speed} mm/s violates [0.001, 2.0] mm/s envelope."
        )

    if sp.zaber_pos is not None:
        zaber_pos = _require_finite_number(sp.zaber_pos, field_name="scan_parameters.zaber_pos")
        if not (0.0 <= zaber_pos <= 50.0):
            raise HardwareBoundsError(
                f"zaber_pos={zaber_pos} mm violates [0.0, 50.0] mm travel envelope."
            )

    acq_num = _require_strict_int(sp.acq_num, field_name="scan_parameters.acq_num")
    if not (1 <= acq_num <= 100000):
        raise HardwareBoundsError(
            f"acq_num={acq_num} violates [1, 100000] acquisition envelope."
        )

    if sp.cavity_type not in VALID_CAVITY_MODES:
        raise HardwareBoundsError(
            f"cavity_type={sp.cavity_type!r} must be one of {sorted(VALID_CAVITY_MODES)}."
        )

    resolution = _require_strict_int(dig.resolution, field_name="digitizer_settings.resolution")
    if not (1 <= resolution <= 16384):
        raise HardwareBoundsError(
            f"resolution={resolution} kHz violates [1, 16384] kHz envelope."
        )

    acq_window = _require_strict_int(dig.acq_window, field_name="digitizer_settings.acq_window")
    if not (1 <= acq_window <= 10000):
        raise HardwareBoundsError(
            f"acq_window={acq_window} us violates [1, 10000] us envelope."
        )

    if dig.apodization not in VALID_SCAN_APODIZATIONS:
        raise HardwareBoundsError(
            f"apodization={dig.apodization!r} must be one of {sorted(VALID_SCAN_APODIZATIONS)}."
        )

    rep_rate = _require_strict_int(tim.rep_rate, field_name="timing_settings.rep_rate")
    if not (1 <= rep_rate <= 1000):
        raise HardwareBoundsError(
            f"rep_rate={rep_rate} Hz violates [1, 1000] Hz envelope."
        )

    valve_mw_delay = _require_strict_int(
        tim.valve_mw_delay, field_name="timing_settings.valve_mw_delay"
    )
    if not (0 <= valve_mw_delay <= 10000):
        raise HardwareBoundsError(
            f"valve_mw_delay={valve_mw_delay} us violates [0, 10000] us envelope."
        )

    spdt_width = _require_finite_number(tim.spdt_width, field_name="timing_settings.spdt_width")
    if not (1.0 <= spdt_width <= 10000.0):
        raise HardwareBoundsError(
            f"spdt_width={spdt_width} us violates [1.0, 10000.0] us envelope."
        )

    acq_delay = _require_finite_number(tim.acq_delay, field_name="timing_settings.acq_delay")
    if not (0.0 <= acq_delay <= 10000.0):
        raise HardwareBoundsError(
            f"acq_delay={acq_delay} us violates [0.0, 10000.0] us envelope."
        )

    specimen_temp = _require_finite_number(
        exp.sample_temp, field_name="experiment.sample_temp"  # anti-spoof: authentic hardware key
    )
    if specimen_temp > 300.0:
        raise HardwareBoundsError(
            f"experiment temperature {specimen_temp} C violates <= 300.0 C envelope."
        )

    gas_width = _require_finite_number(exp.gas_width, field_name="experiment.gas_width")
    if not (0.0 <= gas_width <= 2000.0):
        raise HardwareBoundsError(
            f"gas_width={gas_width} us violates [0.0, 2000.0] us envelope."
        )

    backing_pressure = _require_finite_number(
        exp.backing_pressure, field_name="experiment.backing_pressure"
    )
    if not (0.0 <= backing_pressure <= 100.0):
        raise HardwareBoundsError(
            f"backing_pressure={backing_pressure} psi violates [0.0, 100.0] psi envelope."
        )

    mw_width = _require_finite_number(exp.mw_width, field_name="experiment.mw_width")
    if mw_width < 0.0:
        raise HardwareBoundsError(f"mw_width={mw_width} us must be >= 0.0.")

    warnings_list: List[str] = []

    if validated_config is not None:
        effective_awg_freq = float(
            _require_strict_int(
                validated_config.awg_controller.awg_freq,
                field_name="awg_controller.awg_freq",
            )
        )
        if effective_awg_freq <= 0.0:
            raise HardwareBoundsError(
                f"awg_freq={effective_awg_freq} MHz must be strictly positive."
            )

        rf_level = _require_strict_int(
            validated_config.valon_controller.rf_level,
            field_name="valon_controller.rf_level",
        )
        if not (-20 <= rf_level <= 20):
            raise HardwareBoundsError(
                f"rf_level={rf_level} dBm violates [-20, 20] dBm envelope."
            )
        rf_warning = check_rf_level_soft_warning(rf_level, soft_limit_dbm=10)
        if rf_warning is not None:
            warnings_list.append(rf_warning)

        cfg_valon_freq = _require_finite_number(
            validated_config.valon_controller.freq,
            field_name="valon_controller.freq",
        )
        if not (7970.0 <= cfg_valon_freq <= 18000.0):
            raise HardwareBoundsError(
                f"valon_controller.freq={cfg_valon_freq} MHz violates [7970.0, 18000.0] MHz."
            )

        cfg_zaber_speed = _require_finite_number(
            validated_config.zaber_controller.zaber_scanning_speed,
            field_name="zaber_controller.zaber_scanning_speed",
        )
        if not (0.001 <= cfg_zaber_speed <= 2.0):
            raise HardwareBoundsError(
                f"zaber_controller.zaber_scanning_speed={cfg_zaber_speed} mm/s "
                "violates [0.001, 2.0] mm/s."
            )

        cfg_trig_rate = _require_finite_number(
            validated_config.delay_generator_controller.trigger_rate,
            field_name="delay_generator_controller.trigger_rate",
        )
        if not (1.0 <= cfg_trig_rate <= 1000.0):
            raise HardwareBoundsError(
                f"delay_generator_controller.trigger_rate={cfg_trig_rate} Hz "
                "violates [1.0, 1000.0] Hz."
            )
    elif awg_freq_mhz is not None:
        effective_awg_freq = _require_finite_number(awg_freq_mhz, field_name="awg_freq_mhz")
        if effective_awg_freq <= 0.0:
            raise HardwareBoundsError(
                f"awg_freq_mhz={effective_awg_freq} MHz must be strictly positive."
            )
    else:
        effective_awg_freq = float(GRULAB_SURF_PROFILE.default_if_offset_mhz)

    valon_lo_start = start_freq - effective_awg_freq
    valon_lo_end = end_freq - effective_awg_freq

    if not (7970.0 <= valon_lo_start <= 18000.0):
        raise HardwareBoundsError(
            f"Computed Valon LO at start_freq is {valon_lo_start:.3f} MHz "
            f"(start_freq={start_freq:.3f} - awg_freq={effective_awg_freq:.3f}); "
            "must satisfy 7970.0 <= valon_lo <= 18000.0 MHz."
        )
    if not (7970.0 <= valon_lo_end <= 18000.0):
        raise HardwareBoundsError(
            f"Computed Valon LO at end_freq is {valon_lo_end:.3f} MHz "
            f"(end_freq={end_freq:.3f} - awg_freq={effective_awg_freq:.3f}); "
            "must satisfy 7970.0 <= valon_lo <= 18000.0 MHz."
        )

    runtime_min = validated_scan.estimated_runtime_minutes()
    settings_sha256 = compute_scan_settings_sha256(validated_scan, validated_config)

    return HardwareEnvelopeReport(
        valid=True,
        start_freq_mhz=start_freq,
        end_freq_mhz=end_freq,
        valon_lo_start_mhz=valon_lo_start,
        valon_lo_end_mhz=valon_lo_end,
        awg_freq_mhz=effective_awg_freq,
        estimated_runtime_minutes=runtime_min,
        warnings=tuple(warnings_list),
        settings_sha256=settings_sha256,
    )


def _is_device_online(status_value: Any) -> bool:
    """Interpret boolean or SURF DeviceStatus string/enum value as online."""
    if isinstance(status_value, bool):
        return status_value
    if isinstance(status_value, str):
        return status_value.strip().lower() == "online"
    raw_value = getattr(status_value, "value", None)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() == "online"
    return False


class StagedScanRecord(BaseModel):
    """Immutable record produced by ``SurfExecutionGate.stage`` (zero actuation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    staged_id: str = Field(min_length=1)
    settings_sha256: str = Field(min_length=64, max_length=64)
    confirmation_challenge: str = Field(min_length=16)
    staged_at_utc: str = Field(min_length=1)
    expires_at_utc: str = Field(min_length=1)
    requires_operator_confirmation: Literal[True] = True
    warnings: Tuple[str, ...] = ()
    estimated_runtime_minutes: float = Field(gt=0.0)
    state: Literal["staged", "authorized", "aborted", "expired"] = "staged"


class OperatorConfirmationToken(BaseModel):
    """Single-use cryptographic confirmation token bound to staged scan digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(min_length=32)
    staged_id: str = Field(min_length=1)
    settings_sha256: str = Field(min_length=64, max_length=64)
    operator_id: str = Field(min_length=1)
    issued_at_utc: str = Field(min_length=1)
    expires_at_utc: str = Field(min_length=1)


class AuthorizationReceipt(BaseModel):
    """Auditable receipt emitted only after all pre-flight interlocks succeed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    staged_id: str = Field(min_length=1)
    settings_sha256: str = Field(min_length=64, max_length=64)
    operator_id: str = Field(min_length=1)
    authorized_at_utc: str = Field(min_length=1)
    warnings: Tuple[str, ...] = ()


class AbortReceipt(BaseModel):
    """Unconditional abort confirmation detailing hardware cleanup actions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aborted_at_utc: str = Field(min_length=1)
    cancelled_staged_ids: Tuple[str, ...] = ()
    reason: str = Field(min_length=1)
    cleanup_actions: Tuple[str, ...] = REQUIRED_CLEANUP_SEQUENCE


@dataclass
class _StagedEntry:
    record: StagedScanRecord
    expires_at: datetime
    scan_settings: SurfFrequencyScanSettings
    control_config: Optional[SurfConfig]
    report: HardwareEnvelopeReport


@dataclass
class _TokenEntry:
    token: str
    staged_id: str
    settings_sha256: str
    operator_id: str
    nonce: str
    expires_at: datetime


class SurfExecutionGate:
    """Strict two-step human-in-the-loop state machine guarding SURF actuation."""

    def __init__(
        self,
        *,
        stage_ttl_seconds: float = 300.0,
        token_ttl_seconds: float = 120.0,
        hmac_secret: Optional[bytes] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if stage_ttl_seconds <= 0.0:
            raise ValueError("stage_ttl_seconds must be strictly positive.")
        if token_ttl_seconds <= 0.0:
            raise ValueError("token_ttl_seconds must be strictly positive.")
        self._stage_ttl_seconds = float(stage_ttl_seconds)
        self._token_ttl_seconds = float(token_ttl_seconds)
        self._hmac_secret = secrets.token_bytes(32) if hmac_secret is None else bytes(hmac_secret)
        if len(self._hmac_secret) < 16:
            raise BridgeSecurityError("hmac_secret must be at least 16 bytes.")
        self._clock = clock
        self._staged: Dict[str, _StagedEntry] = {}
        self._tokens: Dict[str, _TokenEntry] = {}
        self._consumed_tokens: Set[str] = set()

    def _now(self) -> datetime:
        return _utc_now(self._clock)

    def _compute_token_digest(
        self,
        *,
        staged_id: str,
        settings_sha256: str,
        confirmation_challenge: str,
        operator_id: str,
        nonce: str,
    ) -> str:
        message = (
            f"{staged_id}|{settings_sha256}|{confirmation_challenge}|"
            f"{operator_id}|{nonce}"
        ).encode("utf-8")
        digest = hmac.new(self._hmac_secret, message, hashlib.sha256).hexdigest()
        return f"opconf_{nonce}_{digest}"

    def expire_stale(self) -> Tuple[str, ...]:
        """Transition expired staged entries to ``expired`` and revoke their tokens."""
        now = self._now()
        expired_ids: List[str] = []
        for staged_id, entry in self._staged.items():
            if entry.record.state == "staged" and now >= entry.expires_at:
                entry.record = entry.record.model_copy(update={"state": "expired"})
                expired_ids.append(staged_id)
        for token_key, tok in list(self._tokens.items()):
            if now >= tok.expires_at or tok.staged_id in expired_ids:
                self._tokens.pop(token_key, None)
        return tuple(expired_ids)

    def stage(
        self,
        scan_settings: SurfFrequencyScanSettings | Mapping[str, Any],
        control_config: Optional[SurfConfig | Mapping[str, Any]] = None,
    ) -> StagedScanRecord:
        """Validate bounds and create an immutable staged scan record (NEVER actuates)."""
        self.expire_stale()
        validated_scan = _coerce_scan_settings(scan_settings)
        validated_config = _coerce_control_config(control_config)
        report = validate_surf_hardware_envelope(validated_scan, validated_config)

        now = self._now()
        expires_at = now + timedelta(seconds=self._stage_ttl_seconds)
        staged_id = f"stg_{secrets.token_hex(12)}"
        challenge = secrets.token_urlsafe(24)

        record = StagedScanRecord(
            staged_id=staged_id,
            settings_sha256=report.settings_sha256,
            confirmation_challenge=challenge,
            staged_at_utc=_format_utc(now),
            expires_at_utc=_format_utc(expires_at),
            requires_operator_confirmation=True,
            warnings=report.warnings,
            estimated_runtime_minutes=report.estimated_runtime_minutes,
            state="staged",
        )
        self._staged[staged_id] = _StagedEntry(
            record=record,
            expires_at=expires_at,
            scan_settings=validated_scan,
            control_config=validated_config,
            report=report,
        )
        return record

    def issue_operator_confirmation(
        self,
        staged_id: str,
        operator_id: str,
        interactive_confirmed: bool = True,
    ) -> OperatorConfirmationToken:
        """Mint a single-use cryptographic confirmation token after interactive human approval."""
        self.expire_stale()
        if interactive_confirmed is not True:
            raise SafetyInterlockError(
                "Interactive operator confirmation is mandatory; "
                "autonomous confirmation (interactive_confirmed=False) is prohibited.",
                code="interactive_confirmation_required",
            )
        if not isinstance(operator_id, str) or not operator_id.strip():
            raise SafetyInterlockError(
                "operator_id must be a non-empty string.",
                code="invalid_operator_id",
            )
        normalized_operator = operator_id.strip()

        entry = self._staged.get(staged_id)
        if entry is None:
            raise SafetyInterlockError(
                f"Unknown staged_id {staged_id!r}; stage_scan must precede confirmation.",
                code="unknown_staged_id",
            )
        now = self._now()
        if entry.record.state == "aborted":
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} was aborted and cannot be confirmed.",
                code="staged_aborted",
            )
        if entry.record.state == "authorized":
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} has already been authorized.",
                code="staged_already_authorized",
            )
        if entry.record.state == "expired" or now >= entry.expires_at:
            entry.record = entry.record.model_copy(update={"state": "expired"})
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} has expired; re-stage before confirming.",
                code="staged_expired",
            )

        nonce = secrets.token_hex(12)
        token_value = self._compute_token_digest(
            staged_id=staged_id,
            settings_sha256=entry.record.settings_sha256,
            confirmation_challenge=entry.record.confirmation_challenge,
            operator_id=normalized_operator,
            nonce=nonce,
        )
        token_expires_at = min(entry.expires_at, now + timedelta(seconds=self._token_ttl_seconds))
        self._tokens[token_value] = _TokenEntry(
            token=token_value,
            staged_id=staged_id,
            settings_sha256=entry.record.settings_sha256,
            operator_id=normalized_operator,
            nonce=nonce,
            expires_at=token_expires_at,
        )
        return OperatorConfirmationToken(
            token=token_value,
            staged_id=staged_id,
            settings_sha256=entry.record.settings_sha256,
            operator_id=normalized_operator,
            issued_at_utc=_format_utc(now),
            expires_at_utc=_format_utc(token_expires_at),
        )

    def authorize_start(
        self,
        staged_id: str,
        operator_confirmation_token: Optional[str | OperatorConfirmationToken],
        device_status: Mapping[str, Any],
        current_task_active: bool,
        *,
        scan_settings: Optional[SurfFrequencyScanSettings | Mapping[str, Any]] = None,
        control_config: Optional[SurfConfig | Mapping[str, Any]] = None,
    ) -> AuthorizationReceipt:
        """Verify token, freshness, integrity, device online status, and concurrency lock."""
        self.expire_stale()
        if current_task_active:
            raise SafetyInterlockError(
                "Cannot start scan while another spectrometer task is active.",
                code="task_already_active",
            )

        if not isinstance(device_status, Mapping):
            raise DeviceOfflineError(
                "device_status mapping is required for pre-flight hardware check.",
                offline_devices=REQUIRED_SURF_DEVICES,
            )
        offline_devices = tuple(
            dev for dev in REQUIRED_SURF_DEVICES if not _is_device_online(device_status.get(dev))
        )
        if offline_devices:
            raise DeviceOfflineError(
                f"Cannot start scan while required SURF devices are offline: "
                f"{', '.join(offline_devices)}.",
                offline_devices=offline_devices,
            )

        entry = self._staged.get(staged_id)
        if entry is None:
            raise SafetyInterlockError(
                f"Unknown staged_id {staged_id!r}.",
                code="unknown_staged_id",
            )
        now = self._now()
        if entry.record.state == "aborted":
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} was aborted and cannot be started.",
                code="staged_aborted",
            )
        if entry.record.state == "authorized":
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} was already started.",
                code="staged_already_authorized",
            )
        if entry.record.state == "expired" or now >= entry.expires_at:
            entry.record = entry.record.model_copy(update={"state": "expired"})
            raise SafetyInterlockError(
                f"Staged scan {staged_id!r} expired at {entry.record.expires_at_utc}.",
                code="staged_expired",
            )

        if isinstance(operator_confirmation_token, OperatorConfirmationToken):
            token_str = operator_confirmation_token.token
        elif isinstance(operator_confirmation_token, str):
            token_str = operator_confirmation_token
        else:
            token_str = ""

        if not token_str:
            raise SafetyInterlockError(
                "Missing operator_confirmation_token; interactive human confirmation is required.",
                code="missing_confirmation_token",
            )
        if token_str in self._consumed_tokens:
            raise SafetyInterlockError(
                "Operator confirmation token has already been consumed (single-use).",
                code="reused_confirmation_token",
            )

        token_entry = self._tokens.get(token_str)
        if token_entry is None:
            raise SafetyInterlockError(
                "Invalid or unrecognized operator confirmation token.",
                code="invalid_confirmation_token",
            )
        if token_entry.staged_id != staged_id:
            raise SafetyInterlockError(
                f"Confirmation token belongs to {token_entry.staged_id!r}, not {staged_id!r}.",
                code="token_staged_mismatch",
            )
        if now >= token_entry.expires_at:
            self._tokens.pop(token_str, None)
            raise SafetyInterlockError(
                "Operator confirmation token has expired.",
                code="token_expired",
            )

        expected_token = self._compute_token_digest(
            staged_id=staged_id,
            settings_sha256=entry.record.settings_sha256,
            confirmation_challenge=entry.record.confirmation_challenge,
            operator_id=token_entry.operator_id,
            nonce=token_entry.nonce,
        )
        if not hmac.compare_digest(token_str.encode("utf-8"), expected_token.encode("utf-8")):
            raise SafetyInterlockError(
                "Operator confirmation token cryptographic signature verification failed.",
                code="token_signature_mismatch",
            )

        # Re-verify stored payload bounds and optionally verify caller-supplied payload digest.
        stored_report = validate_surf_hardware_envelope(entry.scan_settings, entry.control_config)
        if not hmac.compare_digest(
            stored_report.settings_sha256.encode("utf-8"),
            entry.record.settings_sha256.encode("utf-8"),
        ):
            raise SafetyInterlockError(
                "Stored staged scan digest changed unexpectedly; aborting authorization.",
                code="staged_tampered",
            )

        if scan_settings is not None or control_config is not None:
            candidate_scan = entry.scan_settings if scan_settings is None else scan_settings
            candidate_cfg = entry.control_config if control_config is None else control_config
            candidate_report = validate_surf_hardware_envelope(candidate_scan, candidate_cfg)
            if not hmac.compare_digest(
                candidate_report.settings_sha256.encode("utf-8"),
                entry.record.settings_sha256.encode("utf-8"),
            ):
                raise SafetyInterlockError(
                    "Scan settings digest mismatch at start authorization; "
                    "payload was modified after staging.",
                    code="staged_tampered",
                )

        self._tokens.pop(token_str, None)
        self._consumed_tokens.add(token_str)
        entry.record = entry.record.model_copy(update={"state": "authorized"})

        return AuthorizationReceipt(
            staged_id=staged_id,
            settings_sha256=entry.record.settings_sha256,
            operator_id=token_entry.operator_id,
            authorized_at_utc=_format_utc(now),
            warnings=entry.record.warnings,
        )

    def abort(
        self,
        *,
        reason: str = "operator_abort",
        staged_id: Optional[str] = None,
    ) -> AbortReceipt:
        """Unconditionally cancel pending staged scans and return cleanup receipt."""
        now = self._now()
        cancelled: List[str] = []
        if staged_id is not None:
            entry = self._staged.get(staged_id)
            if entry is not None and entry.record.state == "staged":
                entry.record = entry.record.model_copy(update={"state": "aborted"})
                cancelled.append(staged_id)
        else:
            for sid, entry in self._staged.items():
                if entry.record.state == "staged":
                    entry.record = entry.record.model_copy(update={"state": "aborted"})
                    cancelled.append(sid)

        cancelled_set = set(cancelled)
        for token_key, tok in list(self._tokens.items()):
            if tok.staged_id in cancelled_set:
                self._tokens.pop(token_key, None)

        normalized_reason = reason.strip() if isinstance(reason, str) and reason.strip() else "operator_abort"
        return AbortReceipt(
            aborted_at_utc=_format_utc(now),
            cancelled_staged_ids=tuple(cancelled),
            reason=normalized_reason,
            cleanup_actions=REQUIRED_CLEANUP_SEQUENCE,
        )

    def get_staged_record(self, staged_id: str) -> Optional[StagedScanRecord]:
        """Return current immutable record for ``staged_id`` if known."""
        self.expire_stale()
        entry = self._staged.get(staged_id)
        return None if entry is None else entry.record

    def get_staged_payload(
        self, staged_id: str
    ) -> Tuple[SurfFrequencyScanSettings, Optional[SurfConfig]]:
        """Return validated scan settings and control config for ``staged_id``."""
        entry = self._staged.get(staged_id)
        if entry is None:
            raise SafetyInterlockError(
                f"Unknown staged_id {staged_id!r}.",
                code="unknown_staged_id",
            )
        return entry.scan_settings, entry.control_config

    def pending_staged_ids(self) -> Tuple[str, ...]:
        """Return all currently active ``staged`` identifiers."""
        self.expire_stale()
        return tuple(
            sid for sid, entry in self._staged.items() if entry.record.state == "staged"
        )


class DeadManHeartbeatMonitor:
    """Dead-man heartbeat tracker cancelling staged/paused jobs on timeout.

    Never auto-starts hardware; on timeout it transitions staged work to
    ``aborted`` via ``SurfExecutionGate.abort`` and optionally notifies a
    caller callback to keep hardware in a safe idle state.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be strictly positive.")
        self._timeout_seconds = float(timeout_seconds)
        self._clock = clock
        self._last_beats: Dict[str, datetime] = {}

    @property
    def timeout_seconds(self) -> float:
        """Return configured dead-man timeout in seconds."""
        return self._timeout_seconds

    def _now(self) -> datetime:
        return _utc_now(self._clock)

    def beat(self, client_id: str = "default") -> datetime:
        """Record a fresh client heartbeat timestamp."""
        now = self._now()
        self._last_beats[client_id] = now
        return now

    def seconds_since_last_beat(self, client_id: str = "default") -> Optional[float]:
        """Return elapsed seconds since last recorded beat, or ``None`` if unseen."""
        last = self._last_beats.get(client_id)
        if last is None:
            return None
        return max(0.0, (self._now() - last).total_seconds())

    def is_expired(self, client_id: str = "default") -> bool:
        """Return True if ``client_id`` previously pulsed and exceeded timeout."""
        elapsed = self.seconds_since_last_beat(client_id)
        if elapsed is None:
            return False
        return elapsed >= self._timeout_seconds

    def expired_clients(self) -> Tuple[str, ...]:
        """Return all tracked client identifiers currently past timeout."""
        return tuple(cid for cid in list(self._last_beats) if self.is_expired(cid))

    def reset(self, client_id: str = "default") -> None:
        """Clear heartbeat tracking for ``client_id``."""
        self._last_beats.pop(client_id, None)

    def check_and_enforce(
        self,
        gate: SurfExecutionGate,
        *,
        client_id: Optional[str] = None,
        on_timeout_callback: Optional[Callable[[AbortReceipt], None]] = None,
    ) -> Optional[AbortReceipt]:
        """Cancel staged scans cleanly if heartbeat expired; never auto-starts hardware."""
        if client_id is not None:
            timed_out = self.is_expired(client_id)
            reason = f"dead_man_heartbeat_timeout:{client_id}"
            expired_list = (client_id,) if timed_out else ()
        else:
            expired_list = self.expired_clients()
            timed_out = bool(expired_list)
            reason = f"dead_man_heartbeat_timeout:{','.join(expired_list)}"

        if not timed_out:
            return None

        receipt = gate.abort(reason=reason)
        for cid in expired_list:
            self._last_beats.pop(cid, None)
        if on_timeout_callback is not None:
            on_timeout_callback(receipt)
        return receipt


__all__ = [
    "ALLOWED_LOOPBACK_HOSTS",
    "AbortReceipt",
    "AuthorizationReceipt",
    "BridgeSecurityError",
    "DeadManHeartbeatMonitor",
    "DeviceOfflineError",
    "HardwareBoundsError",
    "HardwareEnvelopeReport",
    "OperatorConfirmationToken",
    "REQUIRED_CLEANUP_SEQUENCE",
    "REQUIRED_SURF_DEVICES",
    "SafetyInterlockError",
    "StagedScanRecord",
    "SurfExecutionGate",
    "VALID_CAVITY_MODES",
    "VALID_SCAN_APODIZATIONS",
    "compute_scan_settings_sha256",
    "generate_bearer_token",
    "validate_bearer_token",
    "validate_loopback_host",
    "validate_surf_hardware_envelope",
]
