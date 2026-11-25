# CoChem-SPYFY User Guide

**Package:** `CoChem-SPYFY` (`cochem_spyfy`)  
**Default Instrument Profile:** `grulab_surf` ([`GruLabRotSpec/SURF`](https://github.com/GruLabRotSpec/SURF) @ `0486b877`, 8–18 GHz Cavity Fabry–Perot FTMW)  
**Optional Integration:** `CoChem-BASE` (`cochem.modules` entry point `spyfy = "cochem_spyfy.module_provider:spyfy_provider"`)

---

## 1. Architecture & Scientific Principles

`CoChem-SPYFY` is designed around four non-negotiable scientific principles:

1. **Human-In-The-Loop (HITL) Assignment Authority:** Automated peak pickers, bipartite matchers, and Gaussian Process residual predictors **only** emit `PROPOSED` candidates. An expert spectroscopist must explicitly lock or reject transitions before they enter the next `FitState` commit.
2. **Independent Engine Truthfulness (`SF-CMP-001`–`015`):** Genuine Pickett `SPFIT` and `SPCAT` binaries execute inside isolated temporary sandboxes (`SpfitSandbox`) alongside native FP64 JAX Watson Hamiltonians. `ParityAuditor` compares native vs `SPFIT` constants under explicit per-constant kHz tolerances and explicitly reports `INCOMPLETE`, `DIVERGENT`, or `WITHIN_TOLERANCE`. Missing binaries fail or skip explicitly—never falling back to surrogate approximations.
3. **Zero Autonomous Hardware Actuation:** `CoChem-SPYFY` never opens serial (`COM3`/`COM5`), GPIB/VISA, or RF switch DLL handles directly. `SURF` remains sole hardware owner; scan proposals always pass through an editable override table and Pydantic hardware-bounds gate before export or staging.
4. **Full Cryptographic Provenance:** Every exported TOML, ingested `CavityFiles/*.csv` step, reconciled run, and locked `FitState` carries SHA-256 checksums and lineage links (`CommitManager` DAG + JSONL ledgers).

---

## 2. End-to-End Rotational Fitting Workflow (`SPFIT`/`SPCAT` + JAX + GP)

### 2.1 Configure `SPFIT` and `SPCAT`

Point `CoChem-SPYFY` to your local `calpgm` executables via environment variables (or place them on `PATH` / `~/.local/bin`):

```bash
export SPYFY_CALPGM_DIR=/path/to/calpgm/bin
# Or explicit executable overrides:
export COCHEM_SPFIT_BIN=/path/to/spfit
export COCHEM_SPCAT_BIN=/path/to/spcat
```

Verify runtime discovery:

```python
from cochem_spyfy.module_provider import spyfy_provider

meta = spyfy_provider()
print("SPFIT available:", meta["scientific_apis"]["fit_pickett"]["available"])
print("SPCAT available:", meta["scientific_apis"]["predict_spectrum"]["available"])
print("Default profile:", meta["default_instrument_profile"])
```

### 2.2 Run a Pickett Fit & Catalog Prediction Cycle

```python
from pathlib import Path
from cochem_spyfy.module_provider import fit_pickett, predict_spectrum, audit_fit

# 1. Fit initial locked assignments with genuine SPFIT
fit_res = fit_pickett(
    par_text=Path("molecule.par").read_text(encoding="ascii"),
    lin_text=Path("molecule.lin").read_text(encoding="ascii"),
    stem="molecule_cycle1",
)
print("RMS (MHz):", fit_res["microwave_rms_mhz"])
print("Constants:", fit_res["constants"])

# 2. Predict full rotational catalog with genuine SPCAT
cat_res = predict_spectrum(
    var_text=fit_res["var_text"],
    int_text=Path("molecule.int").read_text(encoding="ascii"),
    stem="molecule_cycle1",
)
print("Predicted transitions:", cat_res["n_transitions"])

# 3. Audit native JAX constants against SPFIT under explicit kHz tolerances
audit = audit_fit(
    native_constants={"A": 5177.7475, "B": 1824.8454, "C": 1808.3907},
    spfit_constants=fit_res["constants"],
    tolerance_khz={"A": 5.0, "B": 5.0, "C": 5.0},
    required_constants=["A", "B", "C"],
)
print("Parity status:", audit["status"], "max delta (kHz):", audit["max_delta_khz"])
```

### 2.3 Gaussian Process `(O - C)` Residual Augmentation & Information Gain

When systematic `(O - C)` residuals exist (e.g., centrifugal distortion or internal-rotation perturbations), train `SpycFitGPRegressor` on quantum features from `QuantumFeatureExtractor` to predict residual shifts `delta_nu_ml_shift_mhz` and 1-sigma predictive uncertainties `sigma_predict_mhz`:

```python
import numpy as np
from cochem_spyfy.physics_core.cochem_spyfy_features import QuantumFeatureExtractor
from cochem_spyfy.physics_core.cochem_spyfy_ml_core import SpycFitGPRegressor
from cochem_spyfy.physics_core.cochem_spyfy_resolvability import ResolvabilityFilter
from cochem_spyfy.physics_core.cochem_spyfy_navigator import SmartScanNavigator

# Extract 14-D quantum-number features and fit GP on locked (O - C) residuals
extractor = QuantumFeatureExtractor()
X_raw = extractor.extract_raw_features(assignments=locked_qns, dipole_components={"mu_a": 1.5})
X_scaled = extractor.fit_transform(X_raw)

gp = SpycFitGPRegressor(n_restarts_optimizer=2, random_state=42)
gp.fit(X_scaled, locked_residuals_mhz)

# Rank candidate 8-18 GHz cavity regions using grulab_surf profile (default)
res_filter = ResolvabilityFilter(instrument_profile="grulab_surf")
navigator = SmartScanNavigator(instrument_profile="grulab_surf")
```

Note on Scientific Hygiene:
- **LOIO Cross-Validation (`cochem_spyfy.convolver.loio_cross_validation`)** requires at least 2 isotopologues (raises `ValueError` for `< 2`) and returns `NaN` for `R^2` when target variance is zero (`ss_tot <= 1e-12`).
- **Jensen Inverse-Inertia Vibrational Averaging (`cochem_spyfy.physics_core.cochem_jensen_averager.generate_vdw_stretch_distribution`)** requires explicit element symbols matching `len(monomer1_coords) + len(monomer2_coords)` so non-`Ar...CO2` complexes (such as `H2O...HF` or `NH3...HF`) preserve exact isotopic identities.

---

## 3. `GruLabRotSpec/SURF` Tier-1 Side-by-Side Workflow

Tier-1 requires **zero modifications to `GruLabRotSpec/SURF`** and works immediately alongside an unmodified `SURF` installation on the laboratory instrument PC.

### Step 1: Plan Cavity-Economical Scan Regions (`SurfScanPlanner`)

`SURF` operates a stepped Fabry–Perot cavity across `8000.0–18000.0 MHz` with ~14 s Zaber retuning overhead per step (`T_min = iterations * (acq_num / rep_rate + 14.0) / 60.0`). `SurfScanPlanner` ranks candidate targeted windows (`±3 * sigma_predict` + Doppler margin) or survey chunks by **Information Gain per minute (`IG / T_min`)**, and optionally seeds resonant Zaber mirror positions from prior `CavityRuns/<freq>MHz.csv` maps.

```python
from pathlib import Path
from cochem_spyfy.instruments import SurfScanPlanner, SurfExporter, SurfCavityMap

cavity_map = SurfCavityMap.from_cavity_runs_dir(Path("./CavityRuns"))
planner = SurfScanPlanner(
    instrument_profile="grulab_surf",
    cavity_map=cavity_map,
    molecule_name="HCOOH_DME",
    specimen_temp_c=22.0,
    gas_name="Ne",
    output_location="C:/SURF_Data/HCOOH_DME",
)
proposals = planner.plan_survey_windows(
    transition_frequencies_mhz=[10874.73, 10899.24, 10924.08],
    information_gains=[2.1, 3.8, 2.5],
    target_constants=["A", "B", "C"],
    max_chunk_bandwidth_mhz=50.0,
    step_size_mhz=0.500,
    acq_num=100,
    rep_rate_hz=5,
)
override_table = planner.create_override_table(proposals[0])
```

### Step 2: Review & Edit Parameters (Mandatory HITL Override Table)

Inspect proposed `SURF` parameters (`start_freq`, `end_freq`, `step_size`, `zaber_speed`, `zaber_pos`, `acq_num`, `cavity_type`, `resolution`, `acq_window`, `apodization`, `rep_rate`, `valve_mw_delay`, `spdt_width`, `acq_delay`, `valon_controller.rf_level`, and experiment gas/pressure fields) and apply operator edits:

```python
override_table.apply_overrides(
    {
        "scan_parameters.acq_num": 150,
        "scan_parameters.step_size": 0.250,
        "digitizer_settings.resolution": 50,
        "valon_controller.rf_level": 10,
    },
    edited_by="j_klaassen",
)
print("Estimated scan duration (min):", override_table.estimated_duration.estimated_minutes)
```

Hardware bounds are enforced strictly by Pydantic v2 (`8000–18000 MHz`, `step_size in [0.001, 1.000] MHz`, `zaber_pos in [0.0, 50.0] mm`, `rf_level in [-20, +20] dBm` with an explicit warning above `+10 dBm`). Every modification is recorded as `{field, proposed_value, user_value, edited_by, edited_at_utc}`.

### Step 3: Export Validated `SURF` TOML Files (`SurfExporter`)

```python
exporter = SurfExporter(instrument_profile="grulab_surf")
export_res = exporter.export_region(
    output_dir=Path("./surf_exports"),
    override_table=override_table,
    fit_state_id=current_fit_state.state_id,
)
```

Three deterministic files are written with SHA-256 receipts:
1. **`<stem>_control_config.toml`**: Validated against `SURF`'s `Config` schema (`valon_controller.freq = start_freq - awg_freq`, strict integer `rf_level` and `awg_freq`). Load directly in `SURF` via **`File -> Open control options from file...`**.
2. **`<stem>_scan_settings.toml`**: Exact 5-table `FrequencyScanSettings` TOML (`[experiment]`, `[scan_parameters]`, `[digitizer_settings]`, `[timing_settings]`, `[output_settings]`) for operator reference or Tier-2 staging.
3. **`<stem>_surf_scan_manifest.json`**: Complete provenance manifest binding the `FitState` ID, `SPCAT` catalog digest, GP hyperparameters, runtime estimate, and every operator override.

### Step 4: Live `CavityFiles/*.csv` Watcher & Cavity Health Monitoring (`SurfRunWatcher`)

Point `SurfRunWatcher` at `SURF`'s output directory while the spectrometer acquires steps:

```python
from cochem_spyfy.instruments import SurfRunWatcher

watcher = SurfRunWatcher(
    Path("C:/SURF_Data/HCOOH_DME"),
    stability_polls=2,
    min_stable_seconds=0.25,
    snr_threshold=5.0,
)
updates = watcher.poll_once()
for upd in updates:
    health = watcher.get_cavity_health(upd.run_dir)
    print(f"Step {upd.step_data.center_freq_mhz:.3f} MHz | Cavity status: {health.status}")
```

- Reads immutable raw `CavityFiles/<center_freq>.csv` step files (never relying on `SURF`'s non-atomic end-of-scan compaction of `<filename>.csv`), applies `SURF`'s exact bandwidth-slice filter `[center - step/2, center + step/2]`, flags cavity rail dropouts (`0.0 mm` / `50.0 mm`), and appends SHA-256 provenance entries to `surf_ingest_provenance.jsonl`.
- At scan completion, `watcher.reconcile_run(run_dir)` verifies FP64 numerical agreement between the live-stitched `CavityFiles` spectrum and `SURF`'s finalized `<filename>.csv`.

### Step 5: Coaxial Doppler Doublet Peak Picking & HITL Lock -> Next Cycle

Because GruLab's cavity FTMW uses a coaxial molecular beam, each rotational transition appears as a symmetric Doppler doublet split by $\Delta\nu_{\text{Doppler}} = \frac{2 v_{\text{beam}}}{c}\,\nu_0$ (carrier gas velocity defaults: `He`, `Ne`, `Ar`, `N2`).

```python
from cochem_spyfy.module_provider import ingest_surf_run
from cochem_spyfy.core_engine.cochem_spyfy_state import CommitManager

ingest = ingest_surf_run(Path("C:/SURF_Data/HCOOH_DME/run_0"), snr_threshold=5.0)
for cand in ingest["proposed_candidates"]:
    assert cand["status"] == "proposed"  # HITL gate: never auto-locked
    print(
        f"{cand['candidate_id']}: {cand['rest_frequency_mhz']:.4f} MHz "
        f"(kind={cand['feature_kind']}, SNR={cand['snr']:.1f}, "
        f"sha256={cand['source_step_sha256'][:12]})"
    )
```

In the GUI (`SpycFitTab` / `SpectralViewport` / `Start_SPYFY.ipynb`), review each `PROPOSED` doublet overlay against `SPCAT` + JAX predictions, click **Approve/Lock** on verified assignments, and commit the updated locked peak list into `CommitManager` to start Cycle $N+1$.

---

## 4. Tier-2 Opt-In Loopback Bridge & Safety Controls

When `GruLabRotSpec/SURF` runs with the optional local bridge enabled (see [`docs/SURF_UPSTREAM_BRIDGE_PR_SPEC.md`](SURF_UPSTREAM_BRIDGE_PR_SPEC.md)), `CoChem-SPYFY` communicates via `SurfLocalBridgeClient` (`src/cochem_spyfy/instruments/surf_bridge_client.py`) protected by `SurfSafetyGate` (`src/cochem_spyfy/instruments/surf_safety.py`).

### Mandatory Safety Interlocks

| Control | Enforcement |
|---|---|
| **Loopback-Only Binding** | Only `127.0.0.1` / `::1` / `localhost` is accepted. Any non-loopback host (`0.0.0.0`, LAN IP) raises `BridgeSecurityError`. |
| **Owner-Private Bearer Auth** | Ephemeral 256-bit bearer token stored in owner-only `bridge.json` and checked with constant-time `hmac.compare_digest`. Missing or wrong tokens fail hard. |
| **Hardware Bounds Gate** | All staged parameters are validated against `GRULAB_SURF_PROFILE` physical bounds before transmission. Out-of-band frequencies (`< 8000` or `> 18000 MHz`), non-positive steps, or out-of-range Zaber positions (`< 0` or `> 50 mm`) are rejected locally and server-side. |
| **Two-Step Operator Confirmation** | `stage_scan` populates `SURF`'s UI widgets and returns a short-lived `(staged_id, settings_sha256)` without touching RF or valves. `start_scan` **refuses** (`SafetyInterlockError`) unless given a single-use `confirmation_token` minted by interactive operator approval bound to that exact `(staged_id, settings_sha256)`. |
| **Heartbeat Watchdog** | `HeartbeatWatchdog` tracks client liveness (`heartbeat()`); expiry transitions the gate into safe-abort state. |
| **Unconditional Abort Priority** | `abort_scan()` works immediately even during an active scan (bypassing normal queue locks) and triggers `SURF`'s `SpectrometerController.cancel_operation()` -> `Spectrometer.cleanup()`. |

### Minimal Tier-2 Client Example

```python
import asyncio
from cochem_spyfy.instruments import SurfLocalBridgeClient, load_bridge_connection_file

async def run_staged_scan() -> None:
    conn = load_bridge_connection_file()  # reads %LOCALAPPDATA%/SURF/bridge.json
    async with SurfLocalBridgeClient.from_connection_file(conn) as client:
        staged = await client.stage_scan(
            control_config=override_table.to_control_config(),
            scan_settings=override_table.to_scan_settings(),
        )
        # Operator explicitly reviews widgets in SURF and confirms:
        token = await client.request_operator_confirmation(staged["staged_id"])
        await client.start_scan(staged_id=staged["staged_id"], confirmation_token=token)
        # Emergency stop at any time:
        # await client.abort_scan(reason="Operator requested immediate stop")
```

---

## 5. `CoChem-BASE` Module Handoff Integration

When invoked through `CoChem-BASE` (`cochem.module-handoff/1`), `cochem_spyfy.module_provider:execute_handoff` supports all six operations with immutable SHA-256 input verification and atomic JSON receipts:

- `predict_spectrum` — Genuine `SPCAT` catalog prediction
- `fit_pickett` — Genuine `SPFIT` rotational parameter fitting
- `parse_pickett` — Pure-Python `.fit`, `.par`/`.var`, and `.cat` parser
- `audit_fit` — `ParityAuditor` cross-engine tolerance audit
- `plan_surf_scan` — `SurfScanPlanner` + HITL overrides + `SurfExporter` TOML bundle
- `ingest_surf_run` — `SurfRunWatcher` + Doppler doublet `PROPOSED` candidate ingest

---

## 6. Troubleshooting & Diagnostics

- **Pickett binary not found:** Set `SPYFY_CALPGM_DIR` (or `COCHEM_SPFIT_BIN` / `COCHEM_SPCAT_BIN`). Run `pytest -q tests/test_cochem_spyfy_provider_and_calpgm.py` to confirm live `calpgm` discovery.
- **Line endings on Windows:** Keep `git config core.autocrlf false` so `pyproject.toml`, `requirements.txt`, and TOML exports maintain strict UTF-8 LF (`\n`).
- **Anti-spoof & air-gap verification:**
  ```bash
  python -B ci_tools/anti_spoof_linter.py --strict src/cochem_spyfy tests
  ```
