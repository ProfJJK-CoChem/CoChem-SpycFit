# Upstream PR Specification: Minimal Opt-In Local Control Bridge & Safety Fixes for `GruLabRotSpec/SURF`

**Target Upstream Repository:** `GruLabRotSpec/SURF`  
**Verified Base Commit:** `0486b877d2a38f5bbe50be3cb333d34e27c7c22d` (`version = "0.1.0"`, MIT License)  
**Companion Client:** `CoChem-SPYFY` (`src/cochem_spyfy/instruments/surf_bridge_client.py`, `src/cochem_spyfy/instruments/surf_safety.py`)  
**Total Upstream Diff Budget:** ~250 LOC (zero new third-party dependencies; stdlib `asyncio`, `hmac`, `json`, `secrets`, `tomllib` only)  
**Default Activation State:** **DISABLED by default** — requires explicit `--enable-local-bridge` CLI flag or `[bridge] enabled = true` in `settings.toml`.

---

## 1. Design Principles & Non-Negotiable Constraints

1. **Zero New Dependencies:** `SURF` already runs `asyncio` via `qasync.QEventLoop` (`src/spectrometer_gui/__main__.py:3-22`) and `pydantic>=2.12.5` (`pyproject.toml:12`). The bridge uses Python standard-library `asyncio.start_server` exclusively.
2. **SURF Remains Sole Hardware Owner:** No external client opens serial (`COM3`/`COM5`), GPIB/TCPIP VISA, or Mini-Circuits `.NET` DLL handles. All hardware commands execute exclusively through `SpectrometerController` and `Spectrometer`.
3. **Zero Autonomous Hardware Actuation:** Staging scan parameters populates `FrequencyScanPanel` widgets and creates a short-lived `staged_id`, but **never** starts a scan, enables Valon RF, opens the gas valve, or moves the Zaber mirror until an explicit interactive operator confirmation dialog mints a single-use token bound to `(staged_id, settings_sha256)`.
4. **Loopback-Only & Owner-Private Auth:** Server binds strictly to `127.0.0.1` on an ephemeral OS-assigned port (`port=0`), never `0.0.0.0`, and requires constant-time (`hmac.compare_digest`) bearer authentication whose secret is written only to `%LOCALAPPDATA%\SURF\bridge.json` with owner-only ACLs.
5. **Unconditional Abort Priority:** Both `SURF`'s native `Cancel` button (`gui/frequency_scan_panel.py:346-349`) and RPC `abort_scan` immediately trigger `SpectrometerController.cancel_operation()` -> `Spectrometer.cleanup()`.

---

## 2. Deliverable A: `File -> Load scan settings from file...` & `FrequencyScanPanel.apply_scan_settings(...)`

Unmodified `SURF@0486b877` writes `scan_settings.toml` when starting a scan (`spectrometer.py:144-154`), but provides no loader in `MainWindow` (which only loads hardware `Config` TOMLs at `gui/main_window.py:97-99, 143-162`). Adding a clean TOML loader benefits both Tier-1 manual workflows and Tier-2 staging.

### 2.1 `src/spectrometer_gui/frequency_scan_settings.py` (+30 LOC)

Add `load_frequency_scan_settings(path: Path) -> FrequencyScanSettings` below `FrequencyScanSettings` (`lines 46-52`):

```python
import tomllib
from pathlib import Path

def load_frequency_scan_settings(path: Path) -> FrequencyScanSettings:
    """Parse a SURF scan_settings.toml file into FrequencyScanSettings."""
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    scan_raw = dict(raw["scan_parameters"])
    scan_raw.setdefault("zaber_pos", None)
    return FrequencyScanSettings(
        experiment=Experiment(**raw["experiment"]),
        scan_parameters=ScanParameters(**scan_raw),
        digitizer_settings=DigitizerSettings(**raw["digitizer_settings"]),
        timing_settings=TimingSettings(**raw["timing_settings"]),
        output_settings=OutputSettings(**raw["output_settings"]),
    )
```

Note on `TimingSettings.acq_delay` (`frequency_scan_settings.py:38`): upstream annotates `acq_delay: int` even though `frequency_scan_panel.py:255` uses `QDoubleSpinBox` and `default_config.toml:39` uses `22.5`. Update annotation on `line 38` to `acq_delay: float`.

### 2.2 `src/spectrometer_gui/gui/frequency_scan_panel.py` (+35 LOC)

Add `apply_scan_settings(self, settings: FrequencyScanSettings) -> None` adjacent to `get_scan_settings()` (`lines 357-396`). This method populates every widget across all 4 form groups plus output fields **without** triggering `start_scan()`:

```python
def apply_scan_settings(self, settings: FrequencyScanSettings) -> None:
    """Populate FrequencyScanPanel widgets from settings without starting a scan."""
    if self.spec_controller.current_task is not None:
        raise RuntimeError("Cannot apply scan settings while a scan is active.")

    exp = settings.experiment
    sp = settings.scan_parameters
    dig = settings.digitizer_settings
    tim = settings.timing_settings
    out = settings.output_settings

    # Guard against partial filename recalculations while populating spinboxes
    self.sample_name_field.blockSignals(True)
    self.start_freq_field.blockSignals(True)
    self.end_freq_field.blockSignals(True)
    self.step_size_field.blockSignals(True)
    self.acq_field.blockSignals(True)
    try:
        self.sample_name_field.setText(exp.sample_name)
        self.sample_temp_field.setValue(float(exp.sample_temp))
        self.gas_name_field.setText(exp.gas_name)
        self.gas_width_field.setValue(float(exp.gas_width))
        self.backing_pressure_field.setValue(float(exp.backing_pressure))
        self.chamber_pressure_field.setText(exp.chamber_pressure)
        self.mw_width_field.setValue(float(exp.mw_width))

        self.start_freq_field.setValue(float(sp.start_freq))
        self.end_freq_field.setValue(float(sp.end_freq))
        self.step_size_field.setValue(float(sp.step_size))
        self.scanning_speed_field.setValue(float(sp.zaber_speed))
        if sp.zaber_pos is None:
            self.zaber_set_pos_checkbox.setChecked(False)
            self.zaber_pos_field.setEnabled(False)
        else:
            self.zaber_set_pos_checkbox.setChecked(True)
            self.zaber_pos_field.setEnabled(True)
            self.zaber_pos_field.setValue(float(sp.zaber_pos))
        self.cavity_type_field.setCurrentText(sp.cavity_type)
        self.acq_field.setValue(int(sp.acq_num))

        self.resolution_field.setValue(int(dig.resolution))
        self.acq_window_field.setValue(float(dig.acq_window))
        self.apodization_field.setCurrentText(dig.apodization)

        self.rep_rate_field.setValue(int(tim.rep_rate))
        self.valve_mw_delay_field.setValue(int(tim.valve_mw_delay))
        self.spdt_width_field.setValue(float(tim.spdt_width))
        self.acq_delay_field.setValue(float(tim.acq_delay))

        if out.location:
            self.directory_field.setText(out.location)
    finally:
        self.sample_name_field.blockSignals(False)
        self.start_freq_field.blockSignals(False)
        self.end_freq_field.blockSignals(False)
        self.step_size_field.blockSignals(False)
        self.acq_field.blockSignals(False)

    if out.filename:
        self.output_folder_field.setText(out.filename)
    else:
        self._update_output_filename()
```

### 2.3 `src/spectrometer_gui/gui/main_window.py` (+25 LOC)

1. Promote local `frequency_scan` (`line 64`) to `self.frequency_scan_panel = FrequencyScanPanel(self.spec_controller)` and pass `self.frequency_scan_panel` to `self.tab_widget.addTab(..., "Frequency Scan")` (`line 79`).
2. In `setup_menu_bar()` (`lines 97-105`), add immediately after `open_config_action`:
   ```python
   load_scan_action = file_menu.addAction("Load &scan settings from file...")
   load_scan_action.triggered.connect(self.load_scan_settings)
   ```
3. Add `MainWindow.load_scan_settings(self)` beside `open_config()` (`lines 143-162`):
   - Opens `QFileDialog.getOpenFileName(self, "Load Scan Settings from File", "", "SURF scan settings (*.toml)")`.
   - Calls `load_frequency_scan_settings(Path(filename))`, validates hardware bounds, invokes `self.frequency_scan_panel.apply_scan_settings(scan_settings)`, and switches `self.tab_widget.setCurrentWidget(self.frequency_scan_panel)` so the operator can inspect every populated field before clicking `Start`.
4. In `closeEvent(self, event)` (`lines 237-255`), if `self.local_bridge is not None`, schedule `await self.local_bridge.stop()` and unlink `%LOCALAPPDATA%\SURF\bridge.json` cleanly.

---

## 3. Deliverable B: Optional Loopback Bridge Module (`src/spectrometer_gui/local_bridge.py`, ~145 LOC)

### 3.1 Activation & Opt-In Guard (`__main__.py:10-22` and `settings.py:32-38`)

- In `src/spectrometer_gui/settings.py`: add optional `BridgeConfig(BaseModel)` with `enabled: bool = False`, `heartbeat_timeout_s: float = 15.0`, defaulting on `Settings.bridge: BridgeConfig = Field(default_factory=BridgeConfig)` so existing `settings.toml` files remain 100% backward-compatible.
- In `src/spectrometer_gui/__main__.py:10-22`: parse `--enable-local-bridge` via stdlib `argparse.ArgumentParser.parse_known_args(sys.argv[1:])` (forwarding unknown Qt args to `QApplication`). Instantiate and `await bridge.start()` **only** when `args.enable_local_bridge or main_window.settings.bridge.enabled` is `True`.

### 3.2 Socket Binding & Owner-Private Discovery File

- Bind strictly via `await asyncio.start_server(self._handle_client, host="127.0.0.1", port=0)`. Never accept or bind `0.0.0.0` or any non-loopback address.
- Mint per-process secret `auth_token = secrets.token_urlsafe(32)`.
- Write `%LOCALAPPDATA%\SURF\bridge.json` (on POSIX, `$XDG_RUNTIME_DIR/surf/bridge.json` or `~/.local/share/surf/bridge.json` with `0o600` mode):
  ```json
  {
    "host": "127.0.0.1",
    "port": 54321,
    "pid": 12345,
    "auth_token": "<32-byte urlsafe secret>",
    "surf_version": "0.1.0",
    "protocol_version": "cochem.surf-bridge/1",
    "created_at_utc": "2026-11-24T13:00:00Z"
  }
  ```
- **Windows Owner-Only ACLs:** On Windows (`os.name == "nt"`), create `%LOCALAPPDATA%\SURF` and restrict `bridge.json` ACLs to the current interactive user (`icacls <path> /inheritance:r /grant:r "%USERNAME%:(R,W)"` or Win32 `SetNamedSecurityInfoW` via `ctypes`) before writing the bearer token, and delete `bridge.json` in `stop()` / `aboutToQuit`.

### 3.3 Authenticated JSON-Lines Wire Protocol (`cochem.surf-bridge/1`)

- Every frame is a single UTF-8 LF-terminated RFC-8259 JSON object (`allow_nan=False`, max 1 MiB).
- Request: `{"id": "<req_id>", "auth_token": "<bearer>", "method": "<name>", "params": {...}}`
- Success reply: `{"id": "<req_id>", "ok": true, "result": {...}}`
- Error reply: `{"id": "<req_id>", "ok": false, "error": {"type": "<ErrorType>", "code": "<code>", "message": "<msg>"}}`
- Every request verifies `hmac.compare_digest(provided_token, self._auth_token)` prior to dispatch; failures return `BridgeSecurityError`.

### 3.4 Exact RPC Method Mapping to `SpectrometerController` (`gui/spectrometer_controller.py:14-163`)

| Method | Upstream `SURF` Action | Safety / Interlock Enforcement |
|---|---|---|
| `ping` | Returns `{"pong": true, "surf_version": "0.1.0", "protocol_version": "cochem.surf-bridge/1", "timestamp_utc": ...}` | Auth required |
| `get_status` | Reads `spec_controller.spectrometer.get_device_status()` (`spectrometer.py:784-795`), `current_task is not None`, active `ScanType`, `paused`, latest `zaber_position`, and `directory_field.text()` | Read-only |
| `stage_scan` | Dual-validates all Section 4.3 hardware bounds (`8000..18000` MHz, `start_freq != end_freq`, `valon_lo = freq - awg_freq >= 7970` MHz, `0.001 <= step_size <= 1.0`, `0 <= zaber_pos <= 50`, `0.001 <= zaber_speed <= 2.0`, `-20 <= rf_level <= 20` int + warn `> 10`, `acq_num`, `rep_rate`, delays, `spdt_width`, `resolution`); calls `frequency_scan_panel.apply_scan_settings(...)`; records immutable `staged_id` + `settings_sha256` + `expires_at_utc`; opens/highlights confirmation banner in `SURF` | **NEVER calls `run_scan()`**; rejects invalid bounds with `HardwareBoundsError` |
| `confirm_and_start_scan` | Verifies single-use `operator_confirmation_token` minted by interactive operator approval dialog (`QMessageBox` in `SURF` or paired interactive confirmation bound to `staged_id` + `settings_sha256`), verifies `spec_controller.current_task is None` (`spectrometer_controller.py:44`), verifies all 6 devices in `get_device_status()` are `True` (`valon`, `zaber`, `oscilloscope`, `delay_generator`, `switch`, `awg`), then invokes `spec_controller.run_scan(settings)` (`spectrometer_controller.py:67-71`) | Fails hard with `SafetyInterlockError` or `DeviceOfflineError`; rejects reused/expired/forged tokens |
| `pause_scan` | Sets cooperative `spec_controller.pause_event.set()` (checked in `spectrometer.py:294` loop alongside `canceled`), immediately invokes `delay_generator_controller.stop_pulse()` so gas valve never stays open during pause | Fails with `SafetyInterlockError` if no scan active |
| `resume_scan` | Clears `spec_controller.pause_event.clear()` | Fails with `SafetyInterlockError` if not paused |
| `abort_scan` | **Unconditionally permitted**; cancels all pending `staged_id` entries, sets `cancel_event`, clears `pause_event`, calls `spec_controller.cancel_operation()` (`spectrometer_controller.py:122-125`), guaranteeing `delay_generator.stop_pulse`, `delay_generator.stop_trig`, `oscilloscope.stop_acq`, `zaber.home` (`spectrometer.py:627-630`) | Highest priority; never blocked by staging/pause state |
| `subscribe_events` | Connects Qt slots on `spec_controller.signal` (`ScanSignals`, `spectrometer_controller.py:14-22`: `device_status_changed`, `progress`, `detailed_progress`, `scanning`, `update_graph`, `update_cavityTrack`, `zaber_position`) and pushes JSON-Lines event objects to subscribed loopback clients via `asyncio.Queue` | Read-only event stream |
| `heartbeat` | Updates dead-man watchdog timestamp for `client_id`. If missed past `heartbeat_timeout_s`, cancels all staged scans without ever auto-starting hardware and preserves `delay_generator.stop_pulse()` during paused state | Safe-idle dead-man fallback |

### 3.5 Cooperative Pause Hook in `SpectrometerController` & `Spectrometer` (+12 LOC)

- Add `self.pause_event = threading.Event()` to `SpectrometerController.__init__` (`spectrometer_controller.py:45`) and clear it in `finish_run()` (`line 129`) and `cancel_operation()` (`line 125`).
- In `Spectrometer.scan_frequency()` (`spectrometer.py:294`), accept optional `paused: Event | None = None`:
  ```python
  while paused is not None and paused.is_set() and not canceled.is_set():
      self.delay_generator_controller.stop_pulse()
      time.sleep(0.1)
  if canceled.is_set():
      self.logger.logger.info("Scan Canceled")
      break
  ```
- In `Spectrometer.cleanup()` (`spectrometer.py:627-630`), add `self.delay_generator_controller.stop_pulse()` before `self.delay_generator_controller.stop_trig()` so emergency cancellation always closes the gas valve even if cancelled mid-step.

---

## 4. Deliverable C: Three Verified Upstream Safety Bugfixes

Forensic audit of `GruLabRotSpec/SURF@0486b877` uncovered three real defects that directly impact cancellation safety, device pre-flight checks, and division-by-zero protection. All three are included in the minimal PR:

### Bugfix 1: Propagate `cancel_event` into `Spectrometer.cavity_search()`

- **Root Cause:** `CavitySearchPanel.cancel_search()` (`gui/cavity_search_panel.py:123-124`) calls `self.spec_controller.cancel_operation()`, which sets `self.cancel_event` (`gui/spectrometer_controller.py:122-125`). However, `SpectrometerController._run_search_async()` (`gui/spectrometer_controller.py:107-109`) calls `self.spectrometer.cavity_search(self.search_signals, settings)` **without passing `self.cancel_event`**, and `Spectrometer.cavity_search()` (`spectrometer.py:416-625`) has no `canceled` parameter and never checks cancellation inside `while run_bool:` (`lines 524-620`). Clicking `Cavity Map -> Stop` (`cavity_search_panel.py:91-95`) therefore cannot stop an in-progress cavity search.
- **Exact Unified Patch:**
  ```diff
  --- a/src/spectrometer_gui/gui/spectrometer_controller.py
  +++ b/src/spectrometer_gui/gui/spectrometer_controller.py
  @@ -105,7 +105,7 @@ class SpectrometerController(QObject):
       async def _run_search_async(self, settings: CavitySearchSettings):
           try:
               await asyncio.to_thread(
  -                self.spectrometer.cavity_search, self.search_signals, settings
  +                self.spectrometer.cavity_search, self.search_signals, settings, self.cancel_event
               )
               if self.cancel_event.is_set():
                   self.signal.progress.emit(1, "Search cancelled")
  --- a/src/spectrometer_gui/spectrometer.py
  +++ b/src/spectrometer_gui/spectrometer.py
  @@ -413,7 +413,12 @@ class Spectrometer:
           self.finalize_csv()
           self.logger.logger.info("Run is finished")
   
  -    def cavity_search(self, signals: SearchSignals, settings: CavitySearchSettings):
  +    def cavity_search(
  +        self,
  +        signals: SearchSignals,
  +        settings: CavitySearchSettings,
  +        canceled: Event | None = None,
  +    ):
           # This code is meant to scan the whole region from 0 - 40 mm and find all the cavity positions for a set frequency
   
  @@ -522,6 +527,9 @@ class Spectrometer:
           i = 0
   
           while run_bool:
  +            if canceled is not None and canceled.is_set():
  +                self.logger.logger.info("Cavity search canceled before sweep")
  +                break
               
               new_freq = valon_freq + step_size * i + self.awg_controller.awg_freq
  @@ -543,6 +551,10 @@ class Spectrometer:
               max_list.append(self.scan_with_acquisition(end_pos_zaber_mm))
               self.delay_generator_controller.stop_trig()
   
  +            if canceled is not None and canceled.is_set():
  +                self.logger.logger.info("Cavity search canceled after sweep")
  +                break
  +
               for max_lists in max_list:
  ```

### Bugfix 2: Set `self.initialized = True` in `AWGController.initialize()`

- **Root Cause:** `AWGController.__init__()` sets `self.initialized = False` (`awg_controller.py:15`), and `AWGController.initialize()` (`awg_controller.py:17-22`) opens `"GPIB1::3::INSTR"` and calls `self.update_config(config)`, but **never sets `self.initialized = True`**. Consequently, `Spectrometer.get_device_status()["awg"]` (`spectrometer.py:794`) permanently returns `False` (`OFFLINE`) even when the AWG initialized cleanly — breaking both status displays and Tier-2 pre-flight device checks.
- **Exact Unified Patch:**
  ```diff
  --- a/src/spectrometer_gui/awg_controller.py
  +++ b/src/spectrometer_gui/awg_controller.py
  @@ -19,6 +19,7 @@ class AWGController:
           self._rm = visa.ResourceManager()
           self._awg = self._rm.open_resource(self._visa_address)
           self.update_config(config)
  +        self.initialized = True
           print("AWG connected successfully.")
  ```

### Bugfix 3: Guard Against `step_size <= 0` Before Division in `Spectrometer.scan_frequency()`

- **Root Cause:** `FrequencyScanPanel` configures `self.step_size_field` with `minimum=0` (`gui/frequency_scan_panel.py:105-107`), allowing an operator to type `0.000 MHz`. When `Spectrometer.scan_frequency()` executes `iterations = math.ceil(abs(stop_freq_input - start_freq) / step_size) + 1` (`spectrometer.py:178`), Python raises `ZeroDivisionError` *after* the Zaber mirror has already moved (`line 123`), output directories have been created (`line 128`), and delay-generator triggers have been armed (`lines 164-167`), bypassing `self.cleanup()` (`line 412`).
- **Exact Unified Patch:**
  ```diff
  --- a/src/spectrometer_gui/gui/frequency_scan_panel.py
  +++ b/src/spectrometer_gui/gui/frequency_scan_panel.py
  @@ -103,7 +103,7 @@ class FrequencyScanPanel(QWidget):
   
           step_size_label = QLabel("Freq Step Size")
           self.step_size_field = QDoubleSpinBox(
  -            minimum=0, maximum=1, value=0.5, singleStep=0.25, decimals=3, suffix=" MHz"
  +            minimum=0.001, maximum=1.0, value=0.5, singleStep=0.25, decimals=3, suffix=" MHz"
           )
           scan_form.addRow(step_size_label, self.step_size_field)
  --- a/src/spectrometer_gui/spectrometer.py
  +++ b/src/spectrometer_gui/spectrometer.py
  @@ -99,6 +99,11 @@ class Spectrometer:
           step_size = settings.scan_parameters.step_size
           start_pos = settings.scan_parameters.zaber_pos
  +        if not math.isfinite(step_size) or step_size <= 0.0:
  +            raise ValueError(
  +                f"step_size must be a finite value strictly greater than 0 MHz (got {step_size!r})."
  +            )
           self.config.zaber_controller.zaber_scanning_speed = settings.scan_parameters.zaber_speed
  @@ -175,6 +180,8 @@ class Spectrometer:
               + f"each run the oscilloscope will require a time delay of {self._time_delay}",
           )
   
  +        if step_size <= 0.0:
  +            raise ValueError(f"step_size must be strictly positive (got {step_size!r}).")
           iterations = math.ceil(abs(stop_freq_input - start_freq) / step_size) + 1
  ```

---

## 5. LOC Budget & PR Acceptance Checklist

| File | Estimated Delta | Purpose |
|---|---:|---|
| `src/spectrometer_gui/local_bridge.py` (new) | ~145 LOC | Opt-in `127.0.0.1` JSON-Lines server, owner-ACL `bridge.json`, `ScanSignals` forwarder, dead-man watchdog |
| `src/spectrometer_gui/frequency_scan_settings.py` | +30 LOC | `load_frequency_scan_settings(path)` + `acq_delay: float` typing fix |
| `src/spectrometer_gui/gui/frequency_scan_panel.py` | +36 LOC | `apply_scan_settings(...)` + `step_size_field` `minimum=0.001` |
| `src/spectrometer_gui/gui/main_window.py` | +25 LOC | `File -> Load scan settings from file...` + `self.frequency_scan_panel` + `closeEvent` bridge cleanup |
| `src/spectrometer_gui/__main__.py` & `settings.py` | +16 LOC | `--enable-local-bridge` opt-in CLI flag + `[bridge]` default-disabled config |
| `src/spectrometer_gui/spectrometer.py` & `gui/spectrometer_controller.py` & `awg_controller.py` | +20 LOC | `pause_event`, `cleanup()` `stop_pulse()`, Bugfixes 1–3 |
| **Total** | **~272 LOC** | **Zero new dependencies; 100% opt-in; default behavior unchanged** |

### Acceptance Criteria for Upstream PR Review

1. Launching `python -m spectrometer_gui` without `--enable-local-bridge` opens zero listening sockets and creates no `bridge.json`.
2. Launching with `--enable-local-bridge` binds strictly to `127.0.0.1` on an ephemeral port and writes owner-only `%LOCALAPPDATA%\SURF\bridge.json`.
3. `CoChem-SPYFY`'s `SurfLocalBridgeClient` passes all loopback integration checks (`tests/test_surf_bridge_and_safety.py`) against the live bridge: auth rejection, bounds rejection, unconfirmed start refusal, two-step confirmed start, pause/resume, live `ScanSignals` stream, dead-man expiration, and immediate unconditional `abort_scan`.
4. `AWGController.initialize()` reports `is_initialized() == True` after successful VISA open.
5. Clicking `Stop` during `Cavity Map` cleanly breaks `Spectrometer.cavity_search()` and runs `cleanup()`.
6. Passing `step_size <= 0` raises `ValueError` before any Zaber movement, folder creation, or delay-generator arming.
