# CoChem-SPYFY: Spectroscopic Yield Fitting

**Author/PI:** Dr. Joshua John Klaassen  
**ORCiD:** [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)  
**GitHub Organization:** [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)  
**Primary Collaborator Spectrometer:** [`GruLabRotSpec/SURF`](https://github.com/GruLabRotSpec/SURF) (Gurusinghe Research Lab, Tennessee Tech University; *Rev. Sci. Instrum.* **97**, 024704 (2026), [`doi:10.1063/5.0311736`](https://doi.org/10.1063/5.0311736))  
**User Guide:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## 1. Overview

**CoChem-SPYFY** (`cochem_spyfy`) is a standalone computational rotational and vibrational spectroscopy fitting, assignment, and active-learning package. It combines genuine Pickett `SPFIT`/`SPCAT` (`calpgm`) execution, native FP64 JAX Watson Hamiltonians, Gaussian Process `(O - C)` residual augmentation, and out-of-the-gate **`GruLabRotSpec/SURF`** cavity Fabry–Perot FTMW instrument integration (`default_instrument_profile = "grulab_surf"`).

### Key Capabilities

- **Out-of-the-Gate `GruLabRotSpec/SURF` Default (`grulab_surf`):**
  - Default instrument profile targeting the 8–18 GHz GruLab cavity FTMW spectrometer (`SURF@0486b877`, version `>=0.1.0,<0.2.0`), while retaining `generic_cp_ftmw` and `generic_ir_raman` secondary profiles.
  - **Tier-1 Zero-Change Side-by-Side Workflow:** Plan cavity-economical scan regions (`SurfScanPlanner`), edit every `SURF` scan/timing/digitizer/hardware parameter through a mandatory HITL override table (`SurfParameterOverrideTable`), export validated `*_control_config.toml` + `*_scan_settings.toml` + `surf_scan_manifest.json` (`SurfExporter`), stream live `CavityFiles/*.csv` steps (`SurfRunWatcher`), detect coaxial Doppler doublets (`pick_surf_doppler_peaks`) as strictly `PROPOSED` candidates, and commit human-approved locks into the next fitting cycle (`CommitManager`).
  - **Tier-2 Opt-In Loopback Bridge Safety Controls:** Strictly `127.0.0.1` loopback-only JSON-RPC client (`SurfLocalBridgeClient`) with constant-time bearer auth, hardware envelope guards, mandatory two-step interactive operator confirmation before actuation, heartbeat watchdog, and unconditional `abort_scan`.
- **Dual-Engine Rotational Fitting & Parity Auditing:**
  - Executes genuine `SPFIT` and `SPCAT` binaries inside isolated sandboxes (`SpfitSandbox`), compares native JAX constants against `SPFIT` under explicit per-constant kHz tolerances (`ParityAuditor`), and never substitutes surrogate models when `calpgm` binaries are absent.
- **Standalone or Optional `CoChem-BASE` Add-On:**
  - Runs completely standalone with zero required `CoChem-BASE` runtime dependency, and registers the `cochem.modules` entry point `spyfy = "cochem_spyfy.module_provider:spyfy_provider"` (`cochem.module-provider/1` and `cochem.module-handoff/1`) when installed alongside `CoChem-BASE`.
- **Zero-Mock, Strict Provenance & Scientific Hygiene:**
  - Strict FP64 calculations (`JAX_ENABLE_X64=True`), dynamic Mendeleev isotopic masses, explicit molecular element validation in Jensen inverse-inertia vibrational averaging, and SHA-256 provenance receipts for exported TOMLs, ingested `CavityFiles`, and `CommitManager` DAG states.

---

## 2. Installation

### Standalone Installation

```bash
git clone https://github.com/ProfJJK-CoChem/CoChem-SPYFY.git
cd CoChem-SPYFY
python -m pip install -e ".[dev]"
```

Optional CUDA GPU extras:

```bash
python -m pip install -e ".[gpu]"
```

### Optional `CoChem-BASE` Add-On Installation

Install `CoChem-SPYFY` into the same Python environment as `CoChem-BASE` so `cochem.modules` automatically discovers the `spyfy` provider:

```bash
python -m pip install -e /path/to/CoChem-BASE
python -m pip install -e /path/to/CoChem-SPYFY
```

Verify discovery:

```python
from cochem_spyfy.module_provider import spyfy_provider

meta = spyfy_provider()
assert meta["module_id"] == "spyfy"
assert meta["default_instrument_profile"] == "grulab_surf"
print(meta["operations"], meta["surf_operations"])
```

### Pickett `SPFIT` / `SPCAT` (`calpgm`) Binary Configuration

`CoChem-SPYFY` resolves genuine `spfit` and `spcat` binaries in priority order:
1. `COCHEM_SPFIT_BIN` / `COCHEM_SPCAT_BIN` (explicit executable paths)
2. `SPYFY_CALPGM_DIR` (or legacy `SPYCFIT_CALPGM_DIR`) / `COCHEM_BIN_DIR`
3. System `PATH` and `~/.local/bin/{spfit,spcat}`

---

## 3. Quickstart (Python API, GUI, & Voila Dashboard)

### 3.1 Plan & Export a `GruLabRotSpec/SURF` Scan Region

```python
from pathlib import Path
from cochem_spyfy.module_provider import plan_surf_scan, ingest_surf_run

# 1. Plan a cavity-economical SURF scan region with operator overrides
plan = plan_surf_scan(
    output_dir=Path("./surf_exports"),
    transition_frequencies_mhz=[10874.73, 10899.24, 10924.08, 14531.81],
    information_gains=[2.1, 3.8, 2.5, 5.4],
    target_constants=["A", "B", "C"],
    overrides={"scan_parameters.acq_num": 100, "scan_parameters.step_size": 0.5},
    edited_by="spectroscopist",
)
print("Control Config:", plan["control_config_path"])
print("Scan Settings: ", plan["scan_settings_path"])
print("Est. Runtime:  ", plan["estimated_runtime_minutes"], "min")

# 2. Ingest a completed or live SURF run directory (emits strictly PROPOSED candidates)
ingest = ingest_surf_run(Path("./sample_run_0"), snr_threshold=5.0)
print("Stitched points:", ingest["stitched_spectrum"]["n_points"])
print("Proposed Doppler doublets:", ingest["candidate_count"])
```

### 3.2 Desktop GUI (`SpycFitTab`) & Air-Gapped Voila Dashboard

Launch the interactive Jupyter/Voila HITL dashboard locally (strictly loopback, zero remote CDNs):

```bash
voila Start_SPYFY.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
```

Or use the provided launcher scripts (`launch_voila.sh`, `launch_voila.ps1`, `launch_voila.bat`).

### 3.3 Running the Verification Suite

```bash
python -B -m pytest -q
python -B ci_tools/anti_spoof_linter.py --strict src/cochem_spyfy tests
```

---

## 4. Documentation Index

- **[User Guide (`docs/USER_GUIDE.md`)](docs/USER_GUIDE.md)** — End-to-end rotational fitting (`SPFIT`/`SPCAT` + JAX + GP), `GruLabRotSpec/SURF` Tier-1 side-by-side workflow, and Tier-2 loopback bridge safety controls.
- **[Primary `SURF` Integration Plan (`docs/SURF_PRIMARY_INTEGRATION_PLAN.md`)](docs/SURF_PRIMARY_INTEGRATION_PLAN.md)** — Verified `SURF@0486b877` schemas, hardware envelopes, and closed-loop architecture.
- **[Upstream `SURF` Bridge PR Spec (`docs/SURF_UPSTREAM_BRIDGE_PR_SPEC.md`)](docs/SURF_UPSTREAM_BRIDGE_PR_SPEC.md)** — Minimal opt-in loopback JSON-RPC bridge specification for `GruLabRotSpec/SURF`.
- **[Canonical Pipeline 5.0.0 SRS](docs/SRS.md)** (with non-normative historical specs archived in **[docs/archive/README.md](docs/archive/README.md)**) — Normative scientific, `SURF`, packaging, and independent-engine comparison requirements (`SF-CMP-001`–`015`, `SF-SURF-001`–`008`).
- **[Next-Agent Handoff (`docs/HANDOFF.md`)](docs/HANDOFF.md)** — Architectural conventions and verification baseline notes.

---

## 5. Citation & License

If you use `CoChem-SPYFY` in published research, please cite:
- Klaassen, J. J. et al. *"CoChem: A Differentiable Tensor Framework for Heterogeneous Ab Initio and Spectroscopic Workflows."* (2026)
- Gurusinghe Research Lab (`GruLabRotSpec/SURF`), Tennessee Tech University, *Rev. Sci. Instrum.* **97**, 024704 (2026), `doi:10.1063/5.0311736`.

See [LICENSE](LICENSE) for license terms.
