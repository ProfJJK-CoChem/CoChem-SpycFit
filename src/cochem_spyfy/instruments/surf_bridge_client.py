"""Tier-1 side-by-side co-runner and Tier-2 loopback control bridge for GruLab SURF.

Implements Sections 4.1-4.3 of ``docs/SURF_PRIMARY_INTEGRATION_PLAN.md`` using
Python standard-library ``asyncio`` exclusively (zero new third-party network
dependencies):

- **Tier 1 (``SurfSideBySideCoordinator``):** Zero-change workflow for unmodified
  ``GruLabRotSpec/SURF@0486b877`` generating operator run cards (Markdown + strict
  JSON checklist), enforcing hardware bounds, and tracking watched run folders
  through ``staged -> operator_running -> completed | aborted``.
- **Tier 2 (``SurfLocalBridgeClient`` + ``SurfLoopbackBridgeServer``):** Real
  ``127.0.0.1`` loopback-only, bearer-authenticated JSON-Lines TCP RPC
  implementing ``ping``, ``get_status``, ``stage_scan``,
  ``confirm_and_start_scan``, ``pause_scan``, ``resume_scan``, ``abort_scan``,
  ``subscribe_events``, and ``heartbeat`` with every ``surf_safety.py`` interlock
  enforced end-to-end.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from cochem_spyfy.instruments.profiles import (
    SURF_SUPPORTED_VERSION_SPEC,
    SURF_VERIFIED_COMMIT_SHA,
)
from cochem_spyfy.instruments.surf_safety import (
    REQUIRED_CLEANUP_SEQUENCE,
    REQUIRED_SURF_DEVICES,
    AbortReceipt,
    BridgeSecurityError,
    DeadManHeartbeatMonitor,
    DeviceOfflineError,
    HardwareBoundsError,
    HardwareEnvelopeReport,
    OperatorConfirmationToken,
    SafetyInterlockError,
    StagedScanRecord,
    SurfExecutionGate,
    generate_bearer_token,
    validate_bearer_token,
    validate_loopback_host,
    validate_surf_hardware_envelope,
)
from cochem_spyfy.instruments.surf_schemas import (
    SurfConfig,
    SurfFrequencyScanSettings,
)

BRIDGE_PROTOCOL_VERSION: str = "cochem.surf-bridge/1"
MAX_JSON_LINE_BYTES: int = 1_048_576


def _utc_iso_now(clock: Optional[Callable[[], datetime]] = None) -> str:
    """Return RFC-3339 UTC timestamp string."""
    now = datetime.now(timezone.utc) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dumps_strict_json_line(payload: Mapping[str, Any]) -> bytes:
    """Serialize mapping as a single strict RFC-8259 JSON line (UTF-8 LF)."""
    rendered = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    encoded = (rendered + "\n").encode("utf-8")
    if len(encoded) > MAX_JSON_LINE_BYTES:
        raise BridgeSecurityError(
            f"JSON-Lines payload exceeds maximum size ({len(encoded)} > {MAX_JSON_LINE_BYTES} bytes)."
        )
    return encoded


def _loads_strict_json_line(raw_line: bytes) -> Dict[str, Any]:
    """Decode a single UTF-8 JSON line rejecting NaN/Infinity and non-object roots."""
    if len(raw_line) > MAX_JSON_LINE_BYTES:
        raise BridgeSecurityError("Received JSON-Lines frame exceeding maximum allowed size.")
    try:
        text = raw_line.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise BridgeSecurityError(f"Invalid UTF-8 frame on loopback bridge: {exc}") from exc
    if not text:
        raise BridgeSecurityError("Empty JSON-Lines frame received on loopback bridge.")

    def _reject_nonfinite(token: str) -> Any:
        raise BridgeSecurityError(f"Non-finite JSON constant {token!r} is prohibited on bridge.")

    try:
        parsed = json.loads(text, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise BridgeSecurityError(f"Malformed JSON-Lines frame: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BridgeSecurityError("JSON-Lines RPC frames must be JSON objects.")
    return parsed


# ---------------------------------------------------------------------------
# Tier 1: Zero-Change Side-by-Side Co-Runner (Unmodified SURF@0486b877)
# ---------------------------------------------------------------------------


class RegionRunStatus(str, Enum):
    """Lifecycle states for a Tier-1 side-by-side SURF scan region."""

    STAGED = "staged"
    OPERATOR_RUNNING = "operator_running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ChecklistStep(BaseModel):
    """Single explicit operator action on a Tier-1 side-by-side run card."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    target_ui: str = Field(min_length=1)
    verification: str = Field(min_length=1)


class OperatorRunCard(BaseModel):
    """Structured Tier-1 operator run card with Markdown and strict JSON checklist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "cochem.surf-runcard/1"
    region_id: str = Field(min_length=1)
    status: RegionRunStatus = RegionRunStatus.STAGED
    created_at_utc: str = Field(min_length=1)
    fit_state_id: Optional[str] = None
    settings_sha256: str = Field(min_length=64, max_length=64)
    watch_directory: str = Field(min_length=1)
    expected_run_prefix: str = Field(min_length=1)
    start_freq_mhz: float
    end_freq_mhz: float
    step_size_mhz: float
    valon_lo_start_mhz: float
    valon_lo_end_mhz: float
    estimated_runtime_minutes: float = Field(gt=0.0)
    warnings: Tuple[str, ...] = ()
    checklist: Tuple[ChecklistStep, ...]
    markdown: str = Field(min_length=1)
    checklist_json: str = Field(min_length=2)


class SideBySideRegionRecord(BaseModel):
    """Tracked state of one Tier-1 side-by-side scan region."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region_id: str = Field(min_length=1)
    status: RegionRunStatus
    settings_sha256: str = Field(min_length=64, max_length=64)
    expected_run_prefix: str = Field(min_length=1)
    watch_directory: str = Field(min_length=1)
    operator_id: Optional[str] = None
    run_directory: Optional[str] = None
    detected_run_directories: Tuple[str, ...] = ()
    detected_step_csv_count: int = Field(default=0, ge=0)
    abort_reason: Optional[str] = None
    updated_at_utc: str = Field(min_length=1)
    run_card: OperatorRunCard


class SurfSideBySideCoordinator:
    """Tier-1 zero-change coordinator for running SpycFit alongside unmodified SURF.

    Never touches hardware ports or auto-starts acquisitions. Enforces full
    ``surf_safety.py`` hardware bounds, produces operator run cards (Markdown +
    JSON checklist), and monitors the watched SURF output folder.
    """

    def __init__(
        self,
        watch_directory: Path | str,
        *,
        default_control_config: Optional[SurfConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._watch_directory = Path(watch_directory)
        self._default_control_config = default_control_config
        self._clock = clock
        self._regions: Dict[str, SideBySideRegionRecord] = {}

    @property
    def watch_directory(self) -> Path:
        """Return the SURF output folder monitored for run directories."""
        return self._watch_directory

    def stage_region(
        self,
        region_id: str,
        scan_settings: SurfFrequencyScanSettings | Mapping[str, Any],
        *,
        control_config: Optional[SurfConfig | Mapping[str, Any]] = None,
        fit_state_id: Optional[str] = None,
        notes: str = "",
    ) -> OperatorRunCard:
        """Validate bounds and create an operator run card for a scan region."""
        if not isinstance(region_id, str) or not region_id.strip():
            raise ValueError("region_id must be a non-empty string.")
        rid = region_id.strip()
        if rid in self._regions:
            raise ValueError(f"Region {rid!r} is already registered in coordinator.")

        effective_config: Optional[SurfConfig | Mapping[str, Any]] = (
            self._default_control_config if control_config is None else control_config
        )
        report: HardwareEnvelopeReport = validate_surf_hardware_envelope(
            scan_settings, effective_config
        )
        validated_scan = (
            scan_settings
            if isinstance(scan_settings, SurfFrequencyScanSettings)
            else SurfFrequencyScanSettings.model_validate(dict(scan_settings))
        )
        validated_cfg = (
            effective_config
            if effective_config is None or isinstance(effective_config, SurfConfig)
            else SurfConfig.model_validate(dict(effective_config))
        )

        now_iso = _utc_iso_now(self._clock)
        sp = validated_scan.scan_parameters
        dig = validated_scan.digitizer_settings
        tim = validated_scan.timing_settings
        out = validated_scan.output_settings
        exp = validated_scan.experiment
        specimen_label = getattr(exp, "sample_name")
        specimen_temp = getattr(exp, "sample_temp")

        checklist_steps: Tuple[ChecklistStep, ...] = (
            ChecklistStep(
                step_number=1,
                action="Load exported *_control_config.toml via SURF File -> Open control options from file...",
                target_ui="SURF MainWindow / Control tab",
                verification=(
                    f"Confirm Valon LO={report.valon_lo_start_mhz:.3f} MHz "
                    f"(awg_freq={report.awg_freq_mhz:.0f} MHz)"
                    + (
                        f" and RF level={validated_cfg.valon_controller.rf_level} dBm."
                        if validated_cfg is not None
                        else "."
                    )
                ),
            ),
            ChecklistStep(
                step_number=2,
                action="Verify all 6 hardware controllers report ONLINE in SURF Status tab.",
                target_ui="SURF Status tab",
                verification=f"All ONLINE: {', '.join(REQUIRED_SURF_DEVICES)}.",
            ),
            ChecklistStep(
                step_number=3,
                action="Enter validated Frequency Scan parameters from the table below.",
                target_ui="SURF Frequency Scan tab",
                verification=(
                    f"{sp.start_freq:.3f} -> {sp.end_freq:.3f} MHz, step {sp.step_size:.3f} MHz, "
                    f"acq_num={sp.acq_num}, cavity={sp.cavity_type}, "
                    f"dir={self._watch_directory}."
                ),
            ),
            ChecklistStep(
                step_number=4,
                action="Review hardware warnings and click Start manually in SURF.",
                target_ui="SURF Frequency Scan -> Start button",
                verification=(
                    f"Verify <watch_dir>/{out.filename}_<k>/CavityFiles/ appears and mark region "
                    "operator_running in SpycFit."
                ),
            ),
            ChecklistStep(
                step_number=5,
                action="Monitor live CavityFiles/*.csv readback; use SURF Cancel for immediate abort.",
                target_ui="SURF Cancel button + SpycFit live viewport",
                verification="On completion or abort, verify cleanup (stop_pulse, stop_trig, stop_acq, zaber.home).",
            ),
        )

        checklist_payload: Dict[str, Any] = {
            "schema_version": "cochem.surf-runcard/1",
            "region_id": rid,
            "status": RegionRunStatus.STAGED.value,
            "created_at_utc": now_iso,
            "fit_state_id": fit_state_id,
            "settings_sha256": report.settings_sha256,
            "surf_verified_commit": SURF_VERIFIED_COMMIT_SHA,
            "surf_version_spec": SURF_SUPPORTED_VERSION_SPEC,
            "watch_directory": str(self._watch_directory),
            "expected_run_prefix": out.filename,
            "hardware_envelope": report.model_dump(mode="json"),
            "scan_settings": validated_scan.to_surf_dict(),
            "warnings": list(report.warnings),
            "notes": notes,
            "checklist": [step.model_dump(mode="json") for step in checklist_steps],
        }
        checklist_json = json.dumps(checklist_payload, indent=2, allow_nan=False) + "\n"

        zaber_pos_text = "Auto (current)" if sp.zaber_pos is None else f"{sp.zaber_pos:.3f} mm"
        warning_lines = (
            "\n".join(f"- WARNING: {msg}" for msg in report.warnings)
            if report.warnings
            else "- None"
        )
        checklist_md = "\n".join(
            f"{step.step_number}. [ ] **{step.action}** ({step.target_ui}) — {step.verification}"
            for step in checklist_steps
        )
        notes_block = f"\n## Operator Notes\n\n{notes.strip()}\n" if notes.strip() else ""

        markdown = (
            f"# SURF Tier-1 Operator Run Card: `{rid}`\n\n"
            f"- **Status:** `{RegionRunStatus.STAGED.value}`\n"
            f"- **Created (UTC):** `{now_iso}`\n"
            f"- **Settings SHA-256:** `{report.settings_sha256}`\n"
            f"- **Fit State ID:** `{fit_state_id or 'N/A'}`\n"
            f"- **Watch Folder:** `{self._watch_directory}`\n"
            f"- **Expected Run Prefix:** `{out.filename}_<k>/`\n"
            f"- **Estimated Runtime:** `{report.estimated_runtime_minutes:.2f} min`\n\n"
            f"## Hardware Bounds & Parameters\n\n"
            f"| Parameter | Value |\n"
            f"|---|---|\n"
            f"| Specimen / Gas | `{specimen_label}` ({specimen_temp:.1f} C) / `{exp.gas_name}` |\n"
            f"| Frequency Range | `{sp.start_freq:.3f} -> {sp.end_freq:.3f} MHz` |\n"
            f"| Valon LO Range | `{report.valon_lo_start_mhz:.3f} -> {report.valon_lo_end_mhz:.3f} MHz` (IF `{report.awg_freq_mhz:.1f} MHz`) |\n"
            f"| Step Size | `{sp.step_size:.3f} MHz` |\n"
            f"| Zaber Speed / Pos | `{sp.zaber_speed:.3f} mm/s` / `{zaber_pos_text}` |\n"
            f"| Acquisitions / Mode | `{sp.acq_num}` / `{sp.cavity_type}` |\n"
            f"| Digitizer | `{dig.resolution} kHz`, `{dig.acq_window} us`, `{dig.apodization}` |\n"
            f"| Timing | `{tim.rep_rate} Hz`, valve-MW `{tim.valve_mw_delay} us`, SPDT `{tim.spdt_width} us`, acq delay `{tim.acq_delay} us` |\n\n"
            f"## Safety Warnings\n\n"
            f"{warning_lines}\n\n"
            f"## Operator Checklist (Zero Autonomous Actuation)\n\n"
            f"{checklist_md}\n"
            f"{notes_block}"
        )

        run_card = OperatorRunCard(
            region_id=rid,
            status=RegionRunStatus.STAGED,
            created_at_utc=now_iso,
            fit_state_id=fit_state_id,
            settings_sha256=report.settings_sha256,
            watch_directory=str(self._watch_directory),
            expected_run_prefix=out.filename,
            start_freq_mhz=sp.start_freq,
            end_freq_mhz=sp.end_freq,
            step_size_mhz=sp.step_size,
            valon_lo_start_mhz=report.valon_lo_start_mhz,
            valon_lo_end_mhz=report.valon_lo_end_mhz,
            estimated_runtime_minutes=report.estimated_runtime_minutes,
            warnings=report.warnings,
            checklist=checklist_steps,
            markdown=markdown,
            checklist_json=checklist_json,
        )

        self._regions[rid] = SideBySideRegionRecord(
            region_id=rid,
            status=RegionRunStatus.STAGED,
            settings_sha256=report.settings_sha256,
            expected_run_prefix=out.filename,
            watch_directory=str(self._watch_directory),
            updated_at_utc=now_iso,
            run_card=run_card,
        )
        return run_card

    def _require_region(self, region_id: str) -> SideBySideRegionRecord:
        record = self._regions.get(region_id)
        if record is None:
            raise KeyError(f"Unknown Tier-1 region_id {region_id!r}.")
        return record

    def mark_operator_running(
        self,
        region_id: str,
        *,
        operator_id: str,
        run_directory: Optional[Path | str] = None,
    ) -> SideBySideRegionRecord:
        """Transition a staged region to ``operator_running`` after manual SURF start."""
        if not isinstance(operator_id, str) or not operator_id.strip():
            raise SafetyInterlockError("operator_id must be a non-empty string.")
        record = self._require_region(region_id)
        if record.status != RegionRunStatus.STAGED:
            raise SafetyInterlockError(
                f"Region {region_id!r} cannot transition from {record.status.value!r} to 'operator_running'."
            )
        now_iso = _utc_iso_now(self._clock)
        updated_card = record.run_card.model_copy(update={"status": RegionRunStatus.OPERATOR_RUNNING})
        updated = record.model_copy(
            update={
                "status": RegionRunStatus.OPERATOR_RUNNING,
                "operator_id": operator_id.strip(),
                "run_directory": None if run_directory is None else str(Path(run_directory)),
                "updated_at_utc": now_iso,
                "run_card": updated_card,
            }
        )
        self._regions[region_id] = updated
        return updated

    def mark_completed(
        self,
        region_id: str,
        *,
        run_directory: Optional[Path | str] = None,
        step_files_count: Optional[int] = None,
    ) -> SideBySideRegionRecord:
        """Transition an ``operator_running`` region to ``completed``."""
        record = self._require_region(region_id)
        if record.status != RegionRunStatus.OPERATOR_RUNNING:
            raise SafetyInterlockError(
                f"Region {region_id!r} must be 'operator_running' before marking completed "
                f"(current status: {record.status.value!r})."
            )
        now_iso = _utc_iso_now(self._clock)
        final_dir = record.run_directory if run_directory is None else str(Path(run_directory))
        final_steps = record.detected_step_csv_count if step_files_count is None else int(step_files_count)
        updated_card = record.run_card.model_copy(update={"status": RegionRunStatus.COMPLETED})
        updated = record.model_copy(
            update={
                "status": RegionRunStatus.COMPLETED,
                "run_directory": final_dir,
                "detected_step_csv_count": final_steps,
                "updated_at_utc": now_iso,
                "run_card": updated_card,
            }
        )
        self._regions[region_id] = updated
        return updated

    def mark_aborted(
        self,
        region_id: str,
        *,
        reason: str = "operator_abort",
    ) -> SideBySideRegionRecord:
        """Unconditionally mark a staged or running region as ``aborted``."""
        record = self._require_region(region_id)
        now_iso = _utc_iso_now(self._clock)
        normalized_reason = reason.strip() if isinstance(reason, str) and reason.strip() else "operator_abort"
        updated_card = record.run_card.model_copy(update={"status": RegionRunStatus.ABORTED})
        updated = record.model_copy(
            update={
                "status": RegionRunStatus.ABORTED,
                "abort_reason": normalized_reason,
                "updated_at_utc": now_iso,
                "run_card": updated_card,
            }
        )
        self._regions[region_id] = updated
        return updated

    def poll_watched_folder(self, region_id: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
        """Scan ``watch_directory`` for SURF ``<prefix>_<k>/`` folders with ``CavityFiles/``."""
        targets = (
            [self._require_region(region_id)]
            if region_id is not None
            else list(self._regions.values())
        )
        discovered: Dict[str, Tuple[str, ...]] = {}
        now_iso = _utc_iso_now(self._clock)

        for record in targets:
            matches: List[str] = []
            step_csv_total = 0
            if self._watch_directory.exists() and self._watch_directory.is_dir():
                prefix = f"{record.expected_run_prefix}_"
                for child in sorted(self._watch_directory.iterdir()):
                    if not child.is_dir():
                        continue
                    if not child.name.startswith(prefix):
                        continue
                    suffix = child.name[len(prefix) :]
                    if not suffix.isdigit():
                        continue
                    cavity_dir = child / "CavityFiles"
                    if (child / "scan_settings.toml").is_file() or cavity_dir.is_dir():
                        matches.append(str(child))
                        if cavity_dir.is_dir():
                            step_csv_total += sum(
                                1 for entry in cavity_dir.iterdir() if entry.is_file() and entry.suffix.lower() == ".csv"
                            )

            match_tuple = tuple(matches)
            discovered[record.region_id] = match_tuple
            latest_dir = match_tuple[-1] if match_tuple else record.run_directory
            self._regions[record.region_id] = record.model_copy(
                update={
                    "detected_run_directories": match_tuple,
                    "detected_step_csv_count": step_csv_total,
                    "run_directory": latest_dir,
                    "updated_at_utc": now_iso,
                }
            )

        return discovered

    def export_run_card_files(self, region_id: str, output_dir: Path | str) -> Tuple[Path, Path]:
        """Write ``<region_id>_runcard.md`` and ``<region_id>_runcard.json`` in UTF-8 LF."""
        record = self._require_region(region_id)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        md_path = out_root / f"{region_id}_runcard.md"
        json_path = out_root / f"{region_id}_runcard.json"
        md_path.write_text(record.run_card.markdown, encoding="utf-8", newline="\n")
        json_path.write_text(record.run_card.checklist_json, encoding="utf-8", newline="\n")
        return md_path, json_path

    def get_region(self, region_id: str) -> SideBySideRegionRecord:
        """Return tracked state for ``region_id``."""
        return self._require_region(region_id)

    def list_regions(self) -> List[SideBySideRegionRecord]:
        """Return all tracked Tier-1 regions in registration order."""
        return list(self._regions.values())


# ---------------------------------------------------------------------------
# Tier 2: Loopback Discovery Metadata & Authenticated JSON-Lines RPC
# ---------------------------------------------------------------------------


class BridgeDiscoveryInfo(BaseModel):
    """Runtime loopback bridge descriptor written to ``%LOCALAPPDATA%\\SURF\\bridge.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)
    pid: int = Field(ge=1)
    auth_token: str = Field(min_length=16)
    surf_version: str = Field(min_length=1)
    protocol_version: str = BRIDGE_PROTOCOL_VERSION
    created_at_utc: str = Field(min_length=1)


def write_bridge_discovery_file(path: Path | str, info: BridgeDiscoveryInfo) -> Path:
    """Persist ``BridgeDiscoveryInfo`` with loopback check and strict UTF-8 LF JSON."""
    validate_loopback_host(info.host)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(info.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    target.write_text(payload, encoding="utf-8", newline="\n")
    if os.name != "nt":
        os.chmod(target, 0o600)
    return target


def read_bridge_discovery_file(path: Path | str) -> BridgeDiscoveryInfo:
    """Load and validate a ``bridge.json`` discovery descriptor."""
    raw = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    info = BridgeDiscoveryInfo.model_validate(parsed)
    validate_loopback_host(info.host)
    return info


@dataclass
class SurfControllerRuntimeState:
    """Authentic in-process hardware state model for the Tier-2 reference harness."""

    device_status: Dict[str, bool] = field(
        default_factory=lambda: {dev: True for dev in REQUIRED_SURF_DEVICES}
    )
    scan_type: Literal["NONE", "FREQUENCY", "CAVITY"] = "NONE"
    current_task_active: bool = False
    paused: bool = False
    zaber_position_mm: float = 12.5
    output_directory: str = "C:\\SURF_Data"
    step_delay_s: float = 0.02
    cleanup_history: List[Tuple[str, ...]] = field(default_factory=list)
    pause_safe_actions: List[str] = field(default_factory=list)
    completed_scans: List[str] = field(default_factory=list)
    aborted_scans: List[str] = field(default_factory=list)


def _map_server_error(error_payload: Mapping[str, Any]) -> Exception:
    """Translate a JSON-Lines error response back to the exact typed exception."""
    err_type = str(error_payload.get("type", "SafetyInterlockError"))
    message = str(error_payload.get("message", "Unknown bridge error"))
    code = str(error_payload.get("code", "bridge_error"))
    if err_type == "BridgeSecurityError":
        return BridgeSecurityError(message, code=code)
    if err_type == "HardwareBoundsError":
        return HardwareBoundsError(message, code=code)
    if err_type == "DeviceOfflineError":
        raw_offline = error_payload.get("offline_devices", ())
        offline = tuple(str(item) for item in raw_offline) if isinstance(raw_offline, list) else ()
        return DeviceOfflineError(message, offline_devices=offline, code=code)
    if err_type == "SafetyInterlockError":
        return SafetyInterlockError(message, code=code)
    return SafetyInterlockError(f"{err_type}: {message}", code=code)


class SurfLoopbackBridgeServer:
    """Reference Tier-2 ``asyncio`` TCP loopback JSON-Lines bridge server.

    Enforces loopback-only binding, constant-time bearer auth, dual hardware
    envelope validation, two-step ``SurfExecutionGate`` HITL authorization,
    cooperative pause/resume, unconditional emergency ``abort_scan`` with full
    hardware cleanup confirmation, dead-man heartbeat watchdog, and live
    ``subscribe_events`` streaming.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        auth_token: Optional[str] = None,
        surf_version: str = "0.1.0",
        gate: Optional[SurfExecutionGate] = None,
        heartbeat_monitor: Optional[DeadManHeartbeatMonitor] = None,
        runtime_state: Optional[SurfControllerRuntimeState] = None,
        heartbeat_timeout_seconds: float = 15.0,
        watchdog_interval_seconds: Optional[float] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._host = validate_loopback_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not (0 <= port <= 65535):
            raise BridgeSecurityError(f"Invalid loopback port {port!r}; expected integer in [0, 65535].")
        self._requested_port = port
        self._auth_token = generate_bearer_token() if auth_token is None else auth_token
        if not isinstance(self._auth_token, str) or not self._auth_token:
            raise BridgeSecurityError("Server auth_token must be a non-empty string.")
        self._surf_version = surf_version
        self._clock = clock
        self._gate = SurfExecutionGate(clock=clock) if gate is None else gate
        self._heartbeat_monitor = (
            DeadManHeartbeatMonitor(timeout_seconds=heartbeat_timeout_seconds, clock=clock)
            if heartbeat_monitor is None
            else heartbeat_monitor
        )
        self._runtime = SurfControllerRuntimeState() if runtime_state is None else runtime_state
        self._watchdog_interval_s = (
            min(0.05, max(0.005, self._heartbeat_monitor.timeout_seconds / 4.0))
            if watchdog_interval_seconds is None
            else float(watchdog_interval_seconds)
        )

        self._server: Optional[asyncio.AbstractServer] = None
        self._bound_port: Optional[int] = None
        self._watchdog_task: Optional[asyncio.Task[None]] = None
        self._active_scan_task: Optional[asyncio.Task[None]] = None
        self._active_staged_id: Optional[str] = None
        self._cancel_event = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._subscribers: Set[asyncio.Queue[Optional[Dict[str, Any]]]] = set()

    @property
    def host(self) -> str:
        """Return validated loopback bind host."""
        return self._host

    @property
    def bound_port(self) -> int:
        """Return OS-assigned TCP port once server is listening."""
        if self._bound_port is None:
            raise RuntimeError("SurfLoopbackBridgeServer is not running.")
        return self._bound_port

    @property
    def auth_token(self) -> str:
        """Return active bearer token required on all RPC calls."""
        return self._auth_token

    @property
    def gate(self) -> SurfExecutionGate:
        """Return the server's ``SurfExecutionGate`` instance."""
        return self._gate

    @property
    def heartbeat_monitor(self) -> DeadManHeartbeatMonitor:
        """Return the server's ``DeadManHeartbeatMonitor`` instance."""
        return self._heartbeat_monitor

    @property
    def runtime_state(self) -> SurfControllerRuntimeState:
        """Return mutable runtime state of the reference controller harness."""
        return self._runtime

    def issue_operator_confirmation(
        self,
        staged_id: str,
        operator_id: str,
        *,
        interactive_confirmed: bool = True,
    ) -> OperatorConfirmationToken:
        """Local interactive operator dialog hook minting a single-use token.

        Not exposed as a remote RPC method so clients can never bypass human
        confirmation over the wire.
        """
        return self._gate.issue_operator_confirmation(
            staged_id=staged_id,
            operator_id=operator_id,
            interactive_confirmed=interactive_confirmed,
        )

    def discovery_info(self) -> BridgeDiscoveryInfo:
        """Build discovery metadata suitable for ``write_bridge_discovery_file``."""
        return BridgeDiscoveryInfo(
            host=self._host,
            port=self.bound_port,
            pid=os.getpid(),
            auth_token=self._auth_token,
            surf_version=self._surf_version,
            protocol_version=BRIDGE_PROTOCOL_VERSION,
            created_at_utc=_utc_iso_now(self._clock),
        )

    async def start(self) -> int:
        """Bind strictly on loopback TCP and launch dead-man watchdog."""
        validate_loopback_host(self._host)
        if self._server is not None:
            return self.bound_port

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._host,
            port=self._requested_port,
        )
        sockets = self._server.sockets
        if not sockets:
            await self.stop()
            raise BridgeSecurityError("Loopback server failed to bind a listening socket.")
        self._bound_port = int(sockets[0].getsockname()[1])
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        return self._bound_port

    async def stop(self) -> None:
        """Stop listening, cancel active background tasks, and close event streams."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                self._watchdog_task = None
            else:
                self._watchdog_task = None

        if self._active_scan_task is not None and not self._active_scan_task.done():
            self._cancel_event.set()
            self._resume_event.set()
            try:
                await asyncio.wait_for(self._active_scan_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._active_scan_task.cancel()
        self._active_scan_task = None

        for queue in list(self._subscribers):
            queue.put_nowait(None)
        self._subscribers.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._bound_port = None

    async def __aenter__(self) -> SurfLoopbackBridgeServer:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    def _broadcast_event(self, event_name: str, payload: Mapping[str, Any]) -> None:
        """Push an event frame to all active ``subscribe_events`` connections."""
        frame: Dict[str, Any] = {
            "event": event_name,
            "payload": dict(payload),
            "timestamp_utc": _utc_iso_now(self._clock),
        }
        for queue in list(self._subscribers):
            queue.put_nowait(frame)

    def _run_hardware_cleanup(self) -> Tuple[str, ...]:
        """Execute deterministic SURF hardware cleanup sequence."""
        self._runtime.cleanup_history.append(REQUIRED_CLEANUP_SEQUENCE)
        self._runtime.zaber_position_mm = 0.0
        self._runtime.paused = False
        return REQUIRED_CLEANUP_SEQUENCE

    async def _watchdog_loop(self) -> None:
        """Periodically enforce dead-man heartbeat expiration without auto-starting."""
        while True:
            await asyncio.sleep(self._watchdog_interval_s)
            receipt = self._heartbeat_monitor.check_and_enforce(self._gate)
            if receipt is not None:
                if self._runtime.paused:
                    self._runtime.pause_safe_actions.append("delay_generator.stop_pulse")
                self._broadcast_event(
                    "heartbeat_timeout",
                    {
                        "reason": receipt.reason,
                        "cancelled_staged_ids": list(receipt.cancelled_staged_ids),
                        "aborted_at_utc": receipt.aborted_at_utc,
                    },
                )

    async def _execute_scan_loop(
        self,
        staged_id: str,
        scan_settings: SurfFrequencyScanSettings,
    ) -> None:
        """Cooperative stepped scan runner mirroring SURF Spectrometer.scan_frequency."""
        sp = scan_settings.scan_parameters
        self._runtime.current_task_active = True
        self._runtime.scan_type = "FREQUENCY"
        self._runtime.paused = False
        self._runtime.output_directory = scan_settings.output_settings.location
        self._active_staged_id = staged_id
        self._cancel_event.clear()
        self._resume_event.set()

        iterations = math.ceil(abs(sp.end_freq - sp.start_freq) / sp.step_size) + 1
        direction = 1.0 if sp.end_freq > sp.start_freq else -1.0
        zaber_pos = self._runtime.zaber_position_mm if sp.zaber_pos is None else float(sp.zaber_pos)

        self._broadcast_event(
            "scanning",
            {"scanning": True, "scan_type": "FREQUENCY", "staged_id": staged_id},
        )
        try:
            for step_idx in range(iterations):
                while not self._resume_event.is_set():
                    if self._cancel_event.is_set():
                        break
                    await asyncio.sleep(0.005)
                if self._cancel_event.is_set():
                    break

                current_freq = sp.start_freq + direction * sp.step_size * step_idx
                zaber_pos = min(50.0, max(0.0, zaber_pos + 0.03 * direction))
                self._runtime.zaber_position_mm = zaber_pos

                self._broadcast_event(
                    "progress",
                    {
                        "fraction": float(step_idx + 1) / float(iterations),
                        "message": f"{step_idx + 1} / {iterations} - Freq {current_freq:.3f} MHz",
                    },
                )
                self._broadcast_event(
                    "detailed_progress",
                    {
                        "current_freq": current_freq,
                        "step_index": step_idx,
                        "total_steps": iterations,
                    },
                )
                self._broadcast_event(
                    "zaber_position",
                    {"position_mm": zaber_pos},
                )
                self._broadcast_event(
                    "step_file_written",
                    {
                        "center_freq_mhz": current_freq,
                        "cavity_position_mm": zaber_pos,
                        "filename": f"{current_freq:.3f}.csv",
                    },
                )

                try:
                    await asyncio.wait_for(
                        self._cancel_event.wait(),
                        timeout=self._runtime.step_delay_s,
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            if self._cancel_event.is_set():
                self._runtime.aborted_scans.append(staged_id)
            else:
                self._runtime.completed_scans.append(staged_id)
        finally:
            self._run_hardware_cleanup()
            self._runtime.current_task_active = False
            self._runtime.scan_type = "NONE"
            self._active_staged_id = None
            self._resume_event.set()
            self._broadcast_event(
                "scanning",
                {"scanning": False, "scan_type": "NONE", "staged_id": staged_id},
            )

    async def _dispatch_rpc(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Execute an authenticated RPC method and return its result mapping."""
        if method == "ping":
            return {
                "pong": True,
                "surf_version": self._surf_version,
                "protocol_version": BRIDGE_PROTOCOL_VERSION,
                "timestamp_utc": _utc_iso_now(self._clock),
            }

        if method == "get_status":
            last_cleanup = (
                list(self._runtime.cleanup_history[-1])
                if self._runtime.cleanup_history
                else []
            )
            return {
                "device_status": dict(self._runtime.device_status),
                "scan_type": self._runtime.scan_type,
                "current_task_active": self._runtime.current_task_active,
                "paused": self._runtime.paused,
                "zaber_position_mm": self._runtime.zaber_position_mm,
                "output_directory": self._runtime.output_directory,
                "staged_ids": list(self._gate.pending_staged_ids()),
                "active_staged_id": self._active_staged_id,
                "last_cleanup_actions": last_cleanup,
            }

        if method == "stage_scan":
            scan_payload = params.get("scan_settings")
            cfg_payload = params.get("control_config")
            record: StagedScanRecord = self._gate.stage(scan_payload, cfg_payload)
            return record.model_dump(mode="json")

        if method == "confirm_and_start_scan":
            staged_id = str(params.get("staged_id", ""))
            token = params.get("operator_confirmation_token")
            receipt = self._gate.authorize_start(
                staged_id=staged_id,
                operator_confirmation_token=token if isinstance(token, str) else None,
                device_status=self._runtime.device_status,
                current_task_active=self._runtime.current_task_active,
            )
            scan_settings, _ = self._gate.get_staged_payload(staged_id)
            self._active_scan_task = asyncio.create_task(
                self._execute_scan_loop(staged_id, scan_settings)
            )
            return {
                "started": True,
                "staged_id": receipt.staged_id,
                "settings_sha256": receipt.settings_sha256,
                "operator_id": receipt.operator_id,
                "authorized_at_utc": receipt.authorized_at_utc,
                "warnings": list(receipt.warnings),
            }

        if method == "pause_scan":
            if not self._runtime.current_task_active:
                raise SafetyInterlockError(
                    "Cannot pause scan when no scan is active.",
                    code="no_active_scan",
                )
            self._runtime.paused = True
            self._resume_event.clear()
            self._runtime.pause_safe_actions.append("delay_generator.stop_pulse")
            self._broadcast_event(
                "scan_paused",
                {
                    "staged_id": self._active_staged_id,
                    "safe_actions": ["delay_generator.stop_pulse"],
                },
            )
            return {
                "paused": True,
                "safe_actions": ["delay_generator.stop_pulse"],
            }

        if method == "resume_scan":
            if not self._runtime.current_task_active or not self._runtime.paused:
                raise SafetyInterlockError(
                    "Cannot resume scan when no scan is currently paused.",
                    code="scan_not_paused",
                )
            self._runtime.paused = False
            self._resume_event.set()
            self._broadcast_event(
                "scan_resumed",
                {"staged_id": self._active_staged_id},
            )
            return {
                "paused": False,
                "resumed": True,
            }

        if method == "abort_scan":
            reason = str(params.get("reason", "operator_abort"))
            abort_receipt: AbortReceipt = self._gate.abort(reason=reason)
            was_active = self._runtime.current_task_active
            if self._active_scan_task is not None and not self._active_scan_task.done():
                self._cancel_event.set()
                self._resume_event.set()
                try:
                    await asyncio.wait_for(self._active_scan_task, timeout=2.0)
                except asyncio.TimeoutError:
                    self._active_scan_task.cancel()
            else:
                self._run_hardware_cleanup()

            self._broadcast_event(
                "scan_aborted",
                {
                    "reason": abort_receipt.reason,
                    "was_active": was_active,
                    "cancelled_staged_ids": list(abort_receipt.cancelled_staged_ids),
                    "cleanup_actions": list(abort_receipt.cleanup_actions),
                },
            )
            return {
                "aborted": True,
                "was_active": was_active,
                "reason": abort_receipt.reason,
                "aborted_at_utc": abort_receipt.aborted_at_utc,
                "cancelled_staged_ids": list(abort_receipt.cancelled_staged_ids),
                "cleanup_actions": list(abort_receipt.cleanup_actions),
            }

        if method == "heartbeat":
            client_id = str(params.get("client_id", "default"))
            beat_time = self._heartbeat_monitor.beat(client_id)
            return {
                "alive": True,
                "client_id": client_id,
                "timeout_seconds": self._heartbeat_monitor.timeout_seconds,
                "server_time_utc": beat_time.isoformat().replace("+00:00", "Z"),
            }

        raise SafetyInterlockError(
            f"Unsupported bridge RPC method {method!r}.",
            code="unsupported_method",
        )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve JSON-Lines RPC requests and event subscriptions on a loopback socket."""
        sub_queue: Optional[asyncio.Queue[Optional[Dict[str, Any]]]] = None
        try:
            while True:
                raw_line = await reader.readline()
                if not raw_line:
                    break

                req_id: Optional[str] = None
                try:
                    frame = _loads_strict_json_line(raw_line)
                    req_id = str(frame.get("id", ""))
                    validate_bearer_token(frame.get("auth_token"), self._auth_token)
                    method = str(frame.get("method", ""))
                    params_raw = frame.get("params", {})
                    if not isinstance(params_raw, dict):
                        raise BridgeSecurityError("RPC 'params' must be a JSON object.")

                    if method == "subscribe_events":
                        sub_queue = asyncio.Queue()
                        self._subscribers.add(sub_queue)
                        ack = {
                            "id": req_id,
                            "ok": True,
                            "result": {"subscribed": True},
                        }
                        writer.write(_dumps_strict_json_line(ack))
                        await writer.drain()
                        while True:
                            event_frame = await sub_queue.get()
                            if event_frame is None:
                                break
                            writer.write(_dumps_strict_json_line(event_frame))
                            await writer.drain()
                        break

                    result = await self._dispatch_rpc(method, params_raw)
                    response: Dict[str, Any] = {
                        "id": req_id,
                        "ok": True,
                        "result": result,
                    }
                except DeviceOfflineError as exc:
                    response = {
                        "id": req_id,
                        "ok": False,
                        "error": {
                            "type": "DeviceOfflineError",
                            "code": exc.code,
                            "message": str(exc),
                            "offline_devices": list(exc.offline_devices),
                        },
                    }
                except HardwareBoundsError as exc:
                    response = {
                        "id": req_id,
                        "ok": False,
                        "error": {
                            "type": "HardwareBoundsError",
                            "code": exc.code,
                            "message": str(exc),
                        },
                    }
                except BridgeSecurityError as exc:
                    response = {
                        "id": req_id,
                        "ok": False,
                        "error": {
                            "type": "BridgeSecurityError",
                            "code": exc.code,
                            "message": str(exc),
                        },
                    }
                except SafetyInterlockError as exc:
                    response = {
                        "id": req_id,
                        "ok": False,
                        "error": {
                            "type": "SafetyInterlockError",
                            "code": exc.code,
                            "message": str(exc),
                        },
                    }
                except Exception as exc:
                    response = {
                        "id": req_id,
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "code": "internal_error",
                            "message": str(exc),
                        },
                    }

                writer.write(_dumps_strict_json_line(response))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            return
        finally:
            if sub_queue is not None:
                self._subscribers.discard(sub_queue)
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()


class SurfLocalBridgeClient:
    """Tier-2 authenticated loopback TCP JSON-Lines client for GruLab SURF."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        auth_token: Optional[str],
        timeout_seconds: float = 5.0,
    ) -> None:
        self._host = validate_loopback_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise BridgeSecurityError(f"Invalid loopback port {port!r}; expected integer in [1, 65535].")
        self._port = port
        self._auth_token = auth_token
        self._timeout_seconds = float(timeout_seconds)

    @classmethod
    def from_discovery_file(
        cls,
        path: Path | str,
        *,
        timeout_seconds: float = 5.0,
    ) -> SurfLocalBridgeClient:
        """Instantiate client from a validated ``bridge.json`` discovery file."""
        info = read_bridge_discovery_file(path)
        return cls(
            host=info.host,
            port=info.port,
            auth_token=info.auth_token,
            timeout_seconds=timeout_seconds,
        )

    @property
    def host(self) -> str:
        """Return validated loopback host."""
        return self._host

    @property
    def port(self) -> int:
        """Return target loopback TCP port."""
        return self._port

    async def _rpc_call(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Open a loopback TCP connection, send one JSON-Lines RPC, and parse reply."""
        validate_loopback_host(self._host)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port),
            timeout=self._timeout_seconds,
        )
        try:
            req_id = f"req_{secrets.token_hex(8)}"
            frame: Dict[str, Any] = {
                "id": req_id,
                "method": method,
                "params": {} if params is None else dict(params),
            }
            if self._auth_token is not None:
                frame["auth_token"] = self._auth_token

            writer.write(_dumps_strict_json_line(frame))
            await writer.drain()

            raw_reply = await asyncio.wait_for(
                reader.readline(),
                timeout=self._timeout_seconds,
            )
            if not raw_reply:
                raise BridgeSecurityError("Loopback bridge closed connection without a response.")
            reply = _loads_strict_json_line(raw_reply)
            if not reply.get("ok", False):
                error_info = reply.get("error", {})
                if not isinstance(error_info, Mapping):
                    raise SafetyInterlockError("Malformed error payload from SURF loopback bridge.")
                raise _map_server_error(error_info)

            result = reply.get("result")
            if not isinstance(result, dict):
                raise SafetyInterlockError("Malformed result payload from SURF loopback bridge.")
            return result
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def ping(self) -> Dict[str, Any]:
        """Check loopback bridge liveness and protocol version."""
        return await self._rpc_call("ping")

    async def get_status(self) -> Dict[str, Any]:
        """Query SURF device online status, active scan state, and Zaber position."""
        return await self._rpc_call("get_status")

    async def stage_scan(
        self,
        scan_settings: SurfFrequencyScanSettings | Mapping[str, Any],
        control_config: Optional[SurfConfig | Mapping[str, Any]] = None,
        *,
        skip_client_validation: bool = False,
    ) -> StagedScanRecord:
        """Validate bounds and stage scan parameters on SURF (zero actuation)."""
        if not skip_client_validation:
            validate_surf_hardware_envelope(scan_settings, control_config)

        scan_payload = (
            scan_settings.to_surf_dict()
            if isinstance(scan_settings, SurfFrequencyScanSettings)
            else dict(scan_settings)
        )
        cfg_payload: Optional[Dict[str, Any]]
        if control_config is None:
            cfg_payload = None
        elif isinstance(control_config, SurfConfig):
            cfg_payload = control_config.model_dump()
        else:
            cfg_payload = dict(control_config)

        result = await self._rpc_call(
            "stage_scan",
            {
                "scan_settings": scan_payload,
                "control_config": cfg_payload,
            },
        )
        return StagedScanRecord.model_validate(result)

    async def confirm_and_start_scan(
        self,
        staged_id: str,
        operator_confirmation_token: Optional[str | OperatorConfirmationToken],
    ) -> Dict[str, Any]:
        """Request scan start using a single-use interactive operator confirmation token."""
        token_value: Optional[str]
        if isinstance(operator_confirmation_token, OperatorConfirmationToken):
            token_value = operator_confirmation_token.token
        else:
            token_value = operator_confirmation_token
        return await self._rpc_call(
            "confirm_and_start_scan",
            {
                "staged_id": staged_id,
                "operator_confirmation_token": token_value,
            },
        )

    async def pause_scan(self) -> Dict[str, Any]:
        """Pause an active scan cooperatively while keeping gas pulsing stopped."""
        return await self._rpc_call("pause_scan")

    async def resume_scan(self) -> Dict[str, Any]:
        """Resume a paused scan."""
        return await self._rpc_call("resume_scan")

    async def abort_scan(self, *, reason: str = "operator_abort") -> Dict[str, Any]:
        """Unconditionally abort any active/staged scan and return cleanup confirmation."""
        return await self._rpc_call("abort_scan", {"reason": reason})

    async def heartbeat(self, *, client_id: str = "spycfit") -> Dict[str, Any]:
        """Send a dead-man heartbeat keepalive to the loopback bridge."""
        return await self._rpc_call("heartbeat", {"client_id": client_id})

    async def subscribe_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream live JSON-Lines ``ScanSignals`` events from the loopback bridge."""
        validate_loopback_host(self._host)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port),
            timeout=self._timeout_seconds,
        )
        try:
            req_id = f"sub_{secrets.token_hex(8)}"
            frame: Dict[str, Any] = {
                "id": req_id,
                "method": "subscribe_events",
                "params": {},
            }
            if self._auth_token is not None:
                frame["auth_token"] = self._auth_token
            writer.write(_dumps_strict_json_line(frame))
            await writer.drain()

            raw_ack = await asyncio.wait_for(reader.readline(), timeout=self._timeout_seconds)
            if not raw_ack:
                raise BridgeSecurityError("Loopback bridge closed subscription before ack.")
            ack = _loads_strict_json_line(raw_ack)
            if not ack.get("ok", False):
                error_info = ack.get("error", {})
                if not isinstance(error_info, Mapping):
                    raise SafetyInterlockError("Malformed error on subscribe_events.")
                raise _map_server_error(error_info)

            while True:
                raw_event = await reader.readline()
                if not raw_event:
                    break
                yield _loads_strict_json_line(raw_event)
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()


__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "BridgeDiscoveryInfo",
    "ChecklistStep",
    "MAX_JSON_LINE_BYTES",
    "OperatorRunCard",
    "RegionRunStatus",
    "SideBySideRegionRecord",
    "SurfControllerRuntimeState",
    "SurfLocalBridgeClient",
    "SurfLoopbackBridgeServer",
    "SurfSideBySideCoordinator",
    "read_bridge_discovery_file",
    "write_bridge_discovery_file",
]
