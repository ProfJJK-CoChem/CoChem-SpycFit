"""Zero-mock verification suite for SPYC-SURF-04 safety interlocks and control bridge.

Exercises real loopback ``asyncio`` TCP sockets (zero socket or hardware mocks)
across:
1. Loopback host enforcement and constant-time bearer authentication.
2. Dual hardware-bounds validation including ``Valon LO >= 7970.0`` MHz and
   RF power soft warnings (``> +10`` dBm).
3. Two-step HITL ``SurfExecutionGate`` enforcement (unconfirmed, forged,
   non-interactive, expired, tampered, reused tokens, offline device, active task).
4. Live Tier-2 loopback RPC workflow (``ping``, ``get_status``, ``stage_scan``,
   ``confirm_and_start_scan``, ``pause_scan``, ``resume_scan``,
   ``subscribe_events``, ``heartbeat``, and unconditional ``abort_scan`` with
   verified cleanup confirmation).
5. Dead-man heartbeat expiration cancelling staged scans without auto-starting.
6. Tier-1 ``SurfSideBySideCoordinator`` run-card generation, watched folder
   detection, and state progression.
7. Strict UTF-8 LF, zero-pass AST, and strict anti-spoof compliance.
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from ci_tools.anti_spoof_linter import check_script
from cochem_spycfit.instruments.surf_bridge_client import (
    BRIDGE_PROTOCOL_VERSION,
    RegionRunStatus,
    SurfControllerRuntimeState,
    SurfLocalBridgeClient,
    SurfLoopbackBridgeServer,
    SurfSideBySideCoordinator,
    read_bridge_discovery_file,
    write_bridge_discovery_file,
)
from cochem_spycfit.instruments.surf_safety import (
    REQUIRED_CLEANUP_SEQUENCE,
    BridgeSecurityError,
    DeadManHeartbeatMonitor,
    DeviceOfflineError,
    HardwareBoundsError,
    SafetyInterlockError,
    SurfExecutionGate,
    generate_bearer_token,
    validate_bearer_token,
    validate_loopback_host,
    validate_surf_hardware_envelope,
)
from cochem_spycfit.instruments.surf_schemas import (
    DigitizerSettings,
    ExperimentSettings,
    OutputSettings,
    ScanParameters,
    SurfConfig,
    SurfFrequencyScanSettings,
    TimingSettings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "surf" / "default_config.toml"

OWNED_PYTHON_FILES = (
    REPO_ROOT / "src" / "cochem_spycfit" / "instruments" / "surf_safety.py",
    REPO_ROOT / "src" / "cochem_spycfit" / "instruments" / "surf_bridge_client.py",
    REPO_ROOT / "tests" / "test_surf_bridge_and_safety.py",
)
OWNED_DOC_FILE = REPO_ROOT / "docs" / "SURF_UPSTREAM_BRIDGE_PR_SPEC.md"


def _load_default_config(*, rf_level: int = 10, awg_freq: int = 30) -> SurfConfig:
    """Load authentic SURF default_config.toml fixture with optional overrides."""
    cfg = SurfConfig.from_surf_toml_file(DEFAULT_CONFIG_FIXTURE)
    return cfg.model_copy(
        update={
            "valon_controller": cfg.valon_controller.model_copy(update={"rf_level": rf_level}),
            "awg_controller": cfg.awg_controller.model_copy(update={"awg_freq": awg_freq}),
        }
    )


def _build_scan_settings(
    *,
    start_freq: float = 10897.0,
    end_freq: float = 10902.0,
    step_size: float = 0.5,
    zaber_speed: float = 0.003,
    zaber_pos: Optional[float] = 12.5,
    acq_num: int = 100,
    cavity_type: str = "Continuous",
    resolution: int = 100,
    acq_window: int = 14,
    apodization: str = "Hanning",
    rep_rate: int = 5,
    valve_mw_delay: int = 1300,
    spdt_width: float = 10.0,
    acq_delay: float = 22.5,
    filename: str = "HCOOH_DME_10897.000_10902.000MHz_0.500MHz_100acqs_2026-11-24",
    location: str = "C:\\SURF_Data\\HCOOH_DME",
) -> SurfFrequencyScanSettings:
    """Build a valid SURF frequency scan configuration for HCOOH-DME."""
    exp_payload: Dict[str, Any] = {
        "sample_name": "HCOOH_DME",
        "sample_temp": 22.0,
        "gas_name": "Ne",
        "gas_width": 600.0,
        "backing_pressure": 15.0,
        "chamber_pressure": "1.0e-6 torr",
        "mw_width": 1.0,
    }
    return SurfFrequencyScanSettings(
        experiment=ExperimentSettings.model_validate(exp_payload),
        scan_parameters=ScanParameters(
            start_freq=start_freq,
            end_freq=end_freq,
            step_size=step_size,
            zaber_speed=zaber_speed,
            zaber_pos=zaber_pos,
            acq_num=acq_num,
            cavity_type=cavity_type,
        ),
        digitizer_settings=DigitizerSettings(
            resolution=resolution,
            acq_window=acq_window,
            apodization=apodization,
        ),
        timing_settings=TimingSettings(
            rep_rate=rep_rate,
            valve_mw_delay=valve_mw_delay,
            spdt_width=spdt_width,
            acq_delay=acq_delay,
        ),
        output_settings=OutputSettings(
            filename=filename,
            location=location,
        ),
    )


class _ManualClock:
    """Deterministic UTC clock for TTL and dead-man timeout verification."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self.current = (
            datetime(2026, 11, 24, 13, 0, 0, tzinfo=timezone.utc)
            if start is None
            else start
        )

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> datetime:
        self.current = self.current + timedelta(seconds=seconds)
        return self.current


# ---------------------------------------------------------------------------
# 1. Loopback Host & Bearer Auth Security Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("allowed", ["127.0.0.1", "::1", "localhost", " LOCALHOST "])
def test_validate_loopback_host_accepts_loopback(allowed: str) -> None:
    normalized = validate_loopback_host(allowed)
    assert normalized in {"127.0.0.1", "::1", "localhost"}


@pytest.mark.parametrize(
    "forbidden",
    ["0.0.0.0", "192.168.1.50", "10.0.0.5", "::", "127.0.0.2", "remote.lab.local", "", "   "],
)
def test_non_loopback_host_refused_everywhere(forbidden: str) -> None:
    with pytest.raises(BridgeSecurityError):
        validate_loopback_host(forbidden)
    with pytest.raises(BridgeSecurityError):
        SurfLocalBridgeClient(host=forbidden, port=50000, auth_token="secret-token-value")
    with pytest.raises(BridgeSecurityError):
        SurfLoopbackBridgeServer(host=forbidden, port=0)


def test_bearer_token_validation_and_real_tcp_auth_rejection() -> None:
    token = generate_bearer_token(32)
    validate_bearer_token(token, token)

    with pytest.raises(BridgeSecurityError):
        validate_bearer_token(None, token)
    with pytest.raises(BridgeSecurityError):
        validate_bearer_token("", token)
    with pytest.raises(BridgeSecurityError):
        validate_bearer_token("wrong-bearer-token", token)

    async def _exercise() -> None:
        async with SurfLoopbackBridgeServer(host="127.0.0.1", port=0, auth_token=token) as server:
            missing_client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token=None,
            )
            with pytest.raises(BridgeSecurityError):
                await missing_client.ping()

            wrong_client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token="incorrect-secret-token",
            )
            with pytest.raises(BridgeSecurityError):
                await wrong_client.ping()

            valid_client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token=token,
            )
            reply = await valid_client.ping()
            assert reply["pong"] is True
            assert reply["protocol_version"] == BRIDGE_PROTOCOL_VERSION

    asyncio.run(_exercise())


# ---------------------------------------------------------------------------
# 2. Dual Hardware-Bounds Validation Tests
# ---------------------------------------------------------------------------


def test_hardware_envelope_accepts_valid_scan_and_warns_above_10_dbm() -> None:
    scan = _build_scan_settings(start_freq=10897.0, end_freq=10902.0)
    cfg_nominal = _load_default_config(rf_level=10, awg_freq=30)
    report_nominal = validate_surf_hardware_envelope(scan, cfg_nominal)
    assert report_nominal.valid is True
    assert report_nominal.valon_lo_start_mhz == pytest.approx(10867.0)
    assert report_nominal.valon_lo_end_mhz == pytest.approx(10872.0)
    assert report_nominal.warnings == ()

    cfg_high_rf = _load_default_config(rf_level=15, awg_freq=30)
    report_high_rf = validate_surf_hardware_envelope(scan, cfg_high_rf)
    assert report_high_rf.valid is True
    assert len(report_high_rf.warnings) == 1
    assert "+10 dBm" in report_high_rf.warnings[0]


def test_hardware_envelope_rejects_valon_lo_below_7970_mhz() -> None:
    scan = _build_scan_settings(start_freq=8000.0, end_freq=8010.0)
    cfg_high_if = _load_default_config(rf_level=10, awg_freq=40)
    with pytest.raises(HardwareBoundsError, match="7970.0"):
        validate_surf_hardware_envelope(scan, cfg_high_if)

    with pytest.raises(HardwareBoundsError, match="7970.0"):
        validate_surf_hardware_envelope(scan, None, awg_freq_mhz=35.0)


@pytest.mark.parametrize(
    ("section", "field_name", "bad_value"),
    [
        ("scan_parameters", "start_freq", 7999.9),
        ("scan_parameters", "end_freq", 18000.1),
        ("scan_parameters", "end_freq", 10897.0),
        ("scan_parameters", "step_size", 0.0),
        ("scan_parameters", "step_size", 1.5),
        ("scan_parameters", "zaber_pos", -0.1),
        ("scan_parameters", "zaber_pos", 50.1),
        ("scan_parameters", "zaber_speed", 0.0005),
        ("scan_parameters", "zaber_speed", 2.5),
        ("scan_parameters", "acq_num", 0),
        ("scan_parameters", "acq_num", 100001),
        ("digitizer_settings", "resolution", 0),
        ("digitizer_settings", "resolution", 20000),
        ("timing_settings", "rep_rate", 0),
        ("timing_settings", "rep_rate", 1001),
        ("timing_settings", "valve_mw_delay", -1),
        ("timing_settings", "valve_mw_delay", 10001),
        ("timing_settings", "spdt_width", 0.5),
        ("timing_settings", "spdt_width", 10001.0),
        ("timing_settings", "acq_delay", -1.0),
        ("timing_settings", "acq_delay", 10001.0),
    ],
)
def test_hardware_envelope_rejects_out_of_bounds_scan_fields(
    section: str,
    field_name: str,
    bad_value: Any,
) -> None:
    payload = _build_scan_settings().to_surf_dict()
    payload[section][field_name] = bad_value

    with pytest.raises((HardwareBoundsError, ValidationError)):
        validate_surf_hardware_envelope(payload)


def test_hardware_envelope_rejects_out_of_bounds_rf_level_and_server_enforces_bounds() -> None:
    valid_scan = _build_scan_settings()
    bad_cfg = _load_default_config().model_dump()
    bad_cfg["valon_controller"]["rf_level"] = 21
    with pytest.raises((HardwareBoundsError, ValidationError)):
        validate_surf_hardware_envelope(valid_scan, bad_cfg)

    bad_cfg_float = _load_default_config().model_dump()
    bad_cfg_float["valon_controller"]["rf_level"] = 10.5
    with pytest.raises((HardwareBoundsError, ValidationError)):
        validate_surf_hardware_envelope(valid_scan, bad_cfg_float)

    async def _exercise_server_bounds() -> None:
        async with SurfLoopbackBridgeServer(host="127.0.0.1", port=0) as server:
            client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token=server.auth_token,
            )
            bad_scan_dict = _build_scan_settings(start_freq=8000.0, end_freq=8005.0).to_surf_dict()
            high_if_cfg = _load_default_config(awg_freq=45).model_dump()
            # Bypass client check intentionally to prove server independently rejects!
            with pytest.raises(HardwareBoundsError):
                await client.stage_scan(
                    bad_scan_dict,
                    high_if_cfg,
                    skip_client_validation=True,
                )

    asyncio.run(_exercise_server_bounds())


# ---------------------------------------------------------------------------
# 3. Strict Two-Step SurfExecutionGate Unit & Interlock Tests
# ---------------------------------------------------------------------------


def test_execution_gate_enforces_interactive_confirmation_expiry_tamper_and_single_use() -> None:
    clock = _ManualClock()
    gate = SurfExecutionGate(stage_ttl_seconds=60.0, token_ttl_seconds=30.0, clock=clock)
    scan = _build_scan_settings()
    cfg = _load_default_config()
    all_online = {
        "valon": True,
        "zaber": True,
        "oscilloscope": True,
        "delay_generator": True,
        "switch": True,
        "awg": True,
    }

    staged = gate.stage(scan, cfg)
    assert staged.requires_operator_confirmation is True
    assert staged.state == "staged"

    # Non-interactive confirmation must fail hard.
    with pytest.raises(SafetyInterlockError):
        gate.issue_operator_confirmation(
            staged.staged_id,
            operator_id="dr_gurusinghe",
            interactive_confirmed=False,
        )

    # Missing or forged confirmation token must fail hard.
    with pytest.raises(SafetyInterlockError):
        gate.authorize_start(staged.staged_id, None, all_online, False)
    with pytest.raises(SafetyInterlockError):
        gate.authorize_start(staged.staged_id, "opconf_forged_0000", all_online, False)

    token = gate.issue_operator_confirmation(
        staged.staged_id,
        operator_id="dr_gurusinghe",
        interactive_confirmed=True,
    )

    # Tampered scan settings passed at authorization time must fail hard.
    tampered_scan = _build_scan_settings(start_freq=10898.0, end_freq=10903.0)
    with pytest.raises(SafetyInterlockError, match="digest mismatch"):
        gate.authorize_start(
            staged.staged_id,
            token,
            all_online,
            False,
            scan_settings=tampered_scan,
            control_config=cfg,
        )

    # Offline device must raise DeviceOfflineError.
    degraded = dict(all_online)
    degraded["valon"] = False
    with pytest.raises(DeviceOfflineError) as offline_exc:
        gate.authorize_start(staged.staged_id, token, degraded, False)
    assert "valon" in offline_exc.value.offline_devices

    # Active hardware task must raise SafetyInterlockError.
    with pytest.raises(SafetyInterlockError, match="active"):
        gate.authorize_start(staged.staged_id, token, all_online, True)

    # Valid authorization succeeds once.
    receipt = gate.authorize_start(staged.staged_id, token, all_online, False)
    assert receipt.staged_id == staged.staged_id
    assert receipt.operator_id == "dr_gurusinghe"

    # Reusing single-use token or staged_id fails hard.
    with pytest.raises(SafetyInterlockError):
        gate.authorize_start(staged.staged_id, token, all_online, False)

    # Expired staged scan fails hard.
    staged_exp = gate.stage(scan, cfg)
    clock.advance(61.0)
    with pytest.raises(SafetyInterlockError, match="expired"):
        gate.issue_operator_confirmation(staged_exp.staged_id, "dr_gurusinghe", True)


# ---------------------------------------------------------------------------
# 4. Real Loopback TCP End-to-End Tier-2 Bridge Tests
# ---------------------------------------------------------------------------


def test_real_loopback_bridge_start_interlocks_offline_and_busy() -> None:
    async def _exercise() -> None:
        state = SurfControllerRuntimeState(step_delay_s=0.05)
        state.device_status["awg"] = False
        async with SurfLoopbackBridgeServer(
            host="127.0.0.1",
            port=0,
            runtime_state=state,
        ) as server:
            client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token=server.auth_token,
            )
            staged = await client.stage_scan(_build_scan_settings(), _load_default_config())
            assert staged.requires_operator_confirmation is True

            # Unconfirmed start rejected.
            with pytest.raises(SafetyInterlockError):
                await client.confirm_and_start_scan(staged.staged_id, None)

            # Confirmed start rejected when AWG is offline.
            token = server.issue_operator_confirmation(staged.staged_id, "operator_alpha")
            with pytest.raises(DeviceOfflineError) as exc_info:
                await client.confirm_and_start_scan(staged.staged_id, token)
            assert "awg" in exc_info.value.offline_devices

            # Bring AWG online but simulate busy controller.
            state.device_status["awg"] = True
            state.current_task_active = True
            with pytest.raises(SafetyInterlockError):
                await client.confirm_and_start_scan(staged.staged_id, token)

            state.current_task_active = False

    asyncio.run(_exercise())


def test_real_loopback_bridge_full_lifecycle_pause_resume_stream_and_abort(tmp_path: Path) -> None:
    async def _exercise() -> None:
        state = SurfControllerRuntimeState(step_delay_s=0.04)
        async with SurfLoopbackBridgeServer(
            host="127.0.0.1",
            port=0,
            runtime_state=state,
        ) as server:
            bridge_file = write_bridge_discovery_file(
                tmp_path / "SURF" / "bridge.json",
                server.discovery_info(),
            )
            discovered = read_bridge_discovery_file(bridge_file)
            assert discovered.host == "127.0.0.1"
            assert discovered.port == server.bound_port

            client = SurfLocalBridgeClient.from_discovery_file(bridge_file)
            events: List[Dict[str, Any]] = []
            subscribed_ready = asyncio.Event()

            async def _collect_events() -> None:
                stream = client.subscribe_events()
                subscribed_ready.set()
                async for evt in stream:
                    events.append(evt)
                    if evt.get("event") == "scan_aborted":
                        break

            event_task = asyncio.create_task(_collect_events())
            await subscribed_ready.wait()
            await asyncio.sleep(0.02)

            status_idle = await client.get_status()
            assert status_idle["current_task_active"] is False
            assert all(status_idle["device_status"].values())

            # Step 1: stage_scan NEVER actuates hardware.
            long_scan = _build_scan_settings(start_freq=10890.0, end_freq=10920.0, step_size=0.5)
            staged = await client.stage_scan(long_scan, _load_default_config(rf_level=14))
            assert staged.requires_operator_confirmation is True
            assert len(staged.warnings) == 1
            assert (await client.get_status())["current_task_active"] is False

            # Step 2: explicit interactive operator confirmation -> confirm_and_start_scan.
            token = server.issue_operator_confirmation(
                staged.staged_id,
                operator_id="lab_operator",
                interactive_confirmed=True,
            )
            started = await client.confirm_and_start_scan(staged.staged_id, token)
            assert started["started"] is True
            assert started["operator_id"] == "lab_operator"

            # Wait until active scan emits initial progress.
            for _ in range(40):
                if (await client.get_status())["current_task_active"]:
                    break
                await asyncio.sleep(0.01)
            assert (await client.get_status())["current_task_active"] is True

            # Pause and resume cooperatively.
            pause_reply = await client.pause_scan()
            assert pause_reply["paused"] is True
            assert "delay_generator.stop_pulse" in pause_reply["safe_actions"]
            assert (await client.get_status())["paused"] is True

            resume_reply = await client.resume_scan()
            assert resume_reply["paused"] is False
            assert resume_reply["resumed"] is True
            assert (await client.get_status())["paused"] is False

            # Unconditional emergency abort during active scan.
            abort_reply = await client.abort_scan(reason="emergency_stop_test")
            assert abort_reply["aborted"] is True
            assert abort_reply["was_active"] is True
            assert tuple(abort_reply["cleanup_actions"]) == REQUIRED_CLEANUP_SEQUENCE

            status_after = await client.get_status()
            assert status_after["current_task_active"] is False
            assert status_after["zaber_position_mm"] == pytest.approx(0.0)
            assert tuple(status_after["last_cleanup_actions"]) == REQUIRED_CLEANUP_SEQUENCE

            await asyncio.wait_for(event_task, timeout=2.0)
            event_names = [item["event"] for item in events]
            assert "scanning" in event_names
            assert "progress" in event_names
            assert "scan_paused" in event_names
            assert "scan_resumed" in event_names
            assert "scan_aborted" in event_names

    asyncio.run(_exercise())


# ---------------------------------------------------------------------------
# 5. Dead-Man Heartbeat Expiration Tests
# ---------------------------------------------------------------------------


def test_dead_man_heartbeat_monitor_deterministic_cancellation() -> None:
    clock = _ManualClock()
    gate = SurfExecutionGate(clock=clock)
    monitor = DeadManHeartbeatMonitor(timeout_seconds=10.0, clock=clock)

    staged = gate.stage(_build_scan_settings(), _load_default_config())
    monitor.beat("spycfit_ui")
    assert monitor.is_expired("spycfit_ui") is False
    assert monitor.check_and_enforce(gate, client_id="spycfit_ui") is None

    clock.advance(11.0)
    assert monitor.is_expired("spycfit_ui") is True
    callbacks: List[str] = []
    receipt = monitor.check_and_enforce(
        gate,
        client_id="spycfit_ui",
        on_timeout_callback=lambda rec: callbacks.append(rec.reason),
    )
    assert receipt is not None
    assert staged.staged_id in receipt.cancelled_staged_ids
    assert len(callbacks) == 1
    record_after = gate.get_staged_record(staged.staged_id)
    assert record_after is not None
    assert record_after.state == "aborted"


def test_real_loopback_dead_man_heartbeat_expires_staged_scan() -> None:
    async def _exercise() -> None:
        async with SurfLoopbackBridgeServer(
            host="127.0.0.1",
            port=0,
            heartbeat_timeout_seconds=0.12,
            watchdog_interval_seconds=0.02,
        ) as server:
            client = SurfLocalBridgeClient(
                host="127.0.0.1",
                port=server.bound_port,
                auth_token=server.auth_token,
            )
            hb = await client.heartbeat(client_id="spycfit_session")
            assert hb["alive"] is True

            staged = await client.stage_scan(_build_scan_settings(), _load_default_config())
            token = server.issue_operator_confirmation(staged.staged_id, "lab_operator")

            # Stop sending heartbeats past dead-man timeout.
            await asyncio.sleep(0.22)

            status = await client.get_status()
            assert status["current_task_active"] is False
            assert staged.staged_id not in status["staged_ids"]

            # Attempting to start the expired/cancelled staged scan must fail hard.
            with pytest.raises(SafetyInterlockError):
                await client.confirm_and_start_scan(staged.staged_id, token)

    asyncio.run(_exercise())


# ---------------------------------------------------------------------------
# 6. Tier-1 Side-by-Side Coordinator Tests
# ---------------------------------------------------------------------------


def test_tier1_side_by_side_coordinator_runcard_watch_and_lifecycle(tmp_path: Path) -> None:
    watch_dir = tmp_path / "SURF_Data"
    watch_dir.mkdir(parents=True)
    coordinator = SurfSideBySideCoordinator(
        watch_dir,
        default_control_config=_load_default_config(rf_level=12),
    )

    # Out-of-bounds scan must be rejected before generating a run card.
    bad_scan = _build_scan_settings(start_freq=8000.0, end_freq=8010.0)
    with pytest.raises(HardwareBoundsError):
        coordinator.stage_region(
            "region_bad_lo",
            bad_scan,
            control_config=_load_default_config(awg_freq=45),
        )

    valid_scan = _build_scan_settings(location=str(watch_dir))
    card = coordinator.stage_region(
        "region_01",
        valid_scan,
        fit_state_id="fit_cycle_01",
        notes="Confirm predicted J=5<-4 doublet",
    )
    assert card.status == RegionRunStatus.STAGED
    assert len(card.warnings) == 1
    assert "+10 dBm" in card.warnings[0]
    assert "SURF Tier-1 Operator Run Card" in card.markdown
    assert "fit_cycle_01" in card.markdown
    parsed_json = json.loads(card.checklist_json)
    assert parsed_json["region_id"] == "region_01"
    assert len(parsed_json["checklist"]) >= 4

    md_path, json_path = coordinator.export_run_card_files("region_01", tmp_path / "runcards")
    assert b"\r" not in md_path.read_bytes()
    assert b"\r" not in json_path.read_bytes()

    # Cannot jump directly from staged to completed.
    with pytest.raises(SafetyInterlockError):
        coordinator.mark_completed("region_01")

    # Simulate unmodified SURF writing <prefix>_0/scan_settings.toml and CavityFiles/*.csv.
    run_dir = watch_dir / f"{valid_scan.output_settings.filename}_0"
    cavity_dir = run_dir / "CavityFiles"
    cavity_dir.mkdir(parents=True)
    (run_dir / "scan_settings.toml").write_text(valid_scan.to_surf_toml(), encoding="utf-8", newline="\n")
    (cavity_dir / "10897.0.csv").write_text(
        "Frequency (MHz),Intensity,Center Freq,Cavity Position\n10897.0,0.012,10897.0,12.5\n",
        encoding="utf-8",
        newline="\n",
    )

    discovered = coordinator.poll_watched_folder("region_01")
    assert discovered["region_01"] == (str(run_dir),)
    polled = coordinator.get_region("region_01")
    assert polled.detected_step_csv_count == 1

    running = coordinator.mark_operator_running(
        "region_01",
        operator_id="dr_gurusinghe",
        run_directory=run_dir,
    )
    assert running.status == RegionRunStatus.OPERATOR_RUNNING
    assert running.operator_id == "dr_gurusinghe"

    completed = coordinator.mark_completed("region_01", step_files_count=11)
    assert completed.status == RegionRunStatus.COMPLETED
    assert completed.detected_step_csv_count == 11

    # Stage a second region and verify unconditional abort from staged.
    scan_two = _build_scan_settings(
        start_freq=11200.0,
        end_freq=11205.0,
        filename="HCOOH_DME_11200.000_11205.000MHz_0.500MHz_100acqs_2026-11-24",
        location=str(watch_dir),
    )
    coordinator.stage_region("region_02", scan_two)
    aborted = coordinator.mark_aborted("region_02", reason="beam_pressure_drop")
    assert aborted.status == RegionRunStatus.ABORTED
    assert aborted.abort_reason == "beam_pressure_drop"
    assert len(coordinator.list_regions()) == 2


# ---------------------------------------------------------------------------
# 7. UTF-8 LF, Zero-Pass AST, and Strict Anti-Spoof Compliance
# ---------------------------------------------------------------------------


def test_spyc_surf_04_files_utf8_lf_zero_pass_and_anti_spoof_clean() -> None:
    for path in (*OWNED_PYTHON_FILES, OWNED_DOC_FILE):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {path}"
        assert b"\r" not in raw, f"CR line ending detected in {path}"
        raw.decode("utf-8")

    for py_path in OWNED_PYTHON_FILES:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert pass_nodes == [], f"Prohibited pass statement in {py_path}"

        violations = check_script(py_path, strict_mode=True)
        assert violations == [], f"Strict anti-spoof violations in {py_path}: {violations}"
