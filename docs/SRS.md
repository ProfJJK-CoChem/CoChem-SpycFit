# CoChem-SPYFY Software Requirements Specification (SRS) — Pipeline 5.0.0

**Document Version:** `5.0.0` (Canonical Normative Baseline)
**Effective Date:** 2026-11-27
**Repository:** `ProfJJK-CoChem/CoChem-SPYFY` (`CoChem-SPYFY` / `cochem_spyfy`)
**Governing Ecosystem Ruling:** `sys-1126-a3d4` (Pipeline 5.0.0 Ecosystem Consolidation)
**Primary Instrument Profile:** `grulab_surf` ([`GruLabRotSpec/SURF`](https://github.com/GruLabRotSpec/SURF) @ `0486b877d2a38f5bbe50be3cb333d34e27c7c22d`, `>=0.1.0,<0.2.0`; *Rev. Sci. Instrum.* **97**, 024704 (2026), `doi:10.1063/5.0311736`)
**Historical / Non-Normative Archives:** [`docs/archive/README.md`](archive/README.md)

---

## 1. Purpose, Ecosystem Boundary & Package Ownership

**CoChem-SPYFY** (`cochem_spyfy`) is the standalone computational spectroscopy assignment, fitting, active-learning, and instrument-integration module of the **CoChem Pipeline 5.0.0** ecosystem.

### 1.1 Sole Package Ownership (`SF-PKG-001`–`SF-PKG-004`)

| ID | Normative Requirement | Verification Criterion |
|---|---|---|
| `SF-PKG-001` | `src/cochem_spyfy/` is the **single canonical source tree** for `CoChem-SPYFY`. No duplicate root-level `cochem_spyfy/` proxy tree shall exist. | Only `src/cochem_spyfy/` is tracked and packaged. |
| `SF-PKG-002` | `CoChem-SPYFY` shall **never** vendor or package `cochem_base`, `cochem`, `cochem_topos`, `core_engine`, `calc`, or `gui` distributions. `ProfJJK-CoChem/CoChem-BASE` is the sole owner of `cochem_base`. | Built wheel (`*.whl`) contains exclusively `cochem_spyfy/` and `CoChem_SPYFY-*.dist-info/`; zero `cochem_base/` files. |
| `SF-PKG-003` | `CoChem-SPYFY` shall operate standalone without requiring `CoChem-BASE` at runtime, and expose optional discovery via `[project.entry-points."cochem.modules"]` (`spyfy = "cochem_spyfy.module_provider:spyfy_provider"`) conforming to `cochem.module-provider/1` and `cochem.module-handoff/1`. | `spyfy_provider()` executes side-effect free with `cochem_base` absent; `plugin.py` degrades cleanly when `cochem_base` is not installed. |
| `SF-PKG-004` | `pyproject.toml` shall expose the console script `cochem-spyfy = "cochem_spyfy.cli:main"` for version, provider inspection, and instrument profile reporting. | `cochem-spyfy --version`, `cochem-spyfy info`, and `cochem-spyfy profiles` return exit code `0` with valid JSON/text output. |

---

## 2. Scientific Scope & Capability Profiles (`SF-SCOPE-001`–`SF-SCOPE-006`, `SF-MODEL-001`–`SF-MODEL-002`)

1. **Required Research Workflows (`SF-SCOPE-001`):**
   - Broadband and cavity rotational assignment and fitting (including isotopologues and calibrated uncertainty).
   - Joint rotational and resolved rovibrational assignment/fitting with explicit shared physical parameters.
   - Computation-led assignment for flexible/non-covalent molecules using qualified QM and MLFF providers.
   - Vibrational band/line fitting, mode interpretation, and spectral-envelope decomposition (`SF-SCOPE-004`).
2. **First-Class Modalities (`SF-SCOPE-002`):** Microwave/FTMW, THz, IR, near-IR, and Raman workflows share consistent import, inspect, propose, assign, fit, compare, and publish semantics. Unsupported combinations emit explicit diagnostics rather than silent approximations (`SF-MODEL-002`).
3. **Observable & Physical Semantics (`SF-MODEL-001`, `SF-SCOPE-003`):** Spectral region, measurement technique, measured observable (absolute frequency, wavenumber, wavelength, Raman shift; absorbance, transmittance, FFT magnitude, integrated intensity), conditions, and physical transition model are represented independently.
4. **Validation Tracks (`SF-SCOPE-006`):** Gas-phase spectroscopy anchors primary validation; solution and solid-state analyses maintain separate validation records and provenance.
5. **PES/MLFF Boundary (`SF-SCOPE-005`):** Inverse PES/force-field refinement exchanges versioned models, atom mappings, units, domain bounds, and training provenance across explicit provider contracts (delegated to `CoChem-TORQ` / `CoChem-BASE` when external training is required).

---

## 3. Ingestion, Interoperability & Physical Invariants (`SF-IO-001`–`SF-IO-007`, `SF-PHYS-001`–`SF-PHYS-008`)

- **Core Contract (`SF-IO-001`–`SF-IO-003`):** Accepts calibrated spectra, peak lists, and native `SURF` continuous FFT CSVs (`Frequency (MHz)`, `Intensity`, `Center Freq`, `Cavity Position`). Original source files are preserved immutably with SHA-256 digests; missing uncertainties remain explicit (never silently replaced with `0.0`).
- **Pickett Serialization (`SF-IO-004`):** Versioned `.par`/`.var`/`.lin`/`.int`/`.cat`/`.fit` parsers and writers preserve exact Watson reduction/representation sign conventions (e.g., `200 -> -DJ`, `1100 -> -DJK`, `2000 -> -DK`, `41000 -> -dK`; `40100` remains reduction-dependent) and refuse fabricated uncertainties.
- **Dynamic Isotopic Masses & Precision (`SF-IO-005`, `SF-PHYS-001`–`SF-PHYS-004`):** Strict FP64 (`JAX_ENABLE_X64=True`) execution; atomic/isotopic masses resolve dynamically via `mendeleev` with explicit element validation (including Jensen inverse-inertia vibrational averaging across arbitrary complexes).
- **Zero-Mock & Provenance Hygiene (`SF-IO-006`–`SF-IO-007`):** No test doubles (`unittest.mock`, `monkeypatch`, synthetic solver stand-ins), no simulation-mode fabrication, empty `.anti_spoof_amnesty.json`, and full deduplication of shared-source benchmark evidence.

---

## 4. Fitting, Uncertainty, ML Augmentation & HITL Authority (`SF-FIT-001`–`SF-FIT-008`, `SF-ML-001`–`SF-ML-003`, `SF-HITL-001`–`SF-HITL-003`)

- **Immutable Fit Binding (`SF-FIT-001`–`SF-FIT-006`):** Every `FitState` binds an immutable observation revision, expert assignment snapshot, free/fixed parameter mask, weights/uncertainties, optimizer settings, covariance/correlation diagnostics, and sensitivity reports. Experiment-only and computation-informed fits are reported side by side.
- **ML Augmentation (`SF-ML-001`–`SF-ML-003`):** `SpycFitGPRegressor` trains strictly on locked `(O - C)` residuals via `QuantumFeatureExtractor`, emitting `ResidualPrediction` records with explicit `ProvenanceTag` (`TORQ` abstention when untrained or out-of-domain), predictive uncertainty `sigma_predict_mhz`, and D-/A-/G-optimal information gain (`cochem_assign_ml`).
- **Human-In-The-Loop Authority (`SF-HITL-001`–`SF-HITL-003`):** Automated peak pickers, bipartite matchers, and GP predictors **exclusively** emit `PROPOSED` candidates. Only explicit human spectroscopist confirmation locks, rejects, or drags assignments into the `CommitManager` DAG.

---

## 5. Independent `SPFIT`/`SPCAT` Engine Comparison Contract (`SF-CMP-001`–`SF-CMP-015`)

- **Authentic Execution (`SF-CMP-001`, `SF-CMP-013`):** Genuine Pickett `spfit` and `spcat` (`calpgm`) executables run in isolated sandboxes (`SpfitSandbox`), resolved via `COCHEM_SPFIT_BIN`/`COCHEM_SPCAT_BIN`, `SPYFY_CALPGM_DIR`, or `PATH`/`~/.local/bin`. Absent binaries produce explicit `UNAVAILABLE` / `NOT_RUN` outcomes—never surrogate substitution.
- **Explicit Status Taxonomy (`SF-CMP-006`–`SF-CMP-012`):** `ParityAuditor` evaluates native JAX vs `SPFIT` constants under declared per-constant kHz tolerances (missing tolerance raises `ValueError`), distinguishing `NOT_RUN`, `UNAVAILABLE`, `FAILED`, `INCOMPATIBLE_MODEL`, `INCOMPLETE`, `DIVERGENT`, and `WITHIN_TOLERANCE`. Empty overlap or missing statistics cannot yield `WITHIN_TOLERANCE`.
- **Transparent Discrepancy Governance (`SF-CMP-014`–`SF-CMP-015`):** Divergences remain visible throughout exploration and are recorded verbatim with provenance in publication exports (`SF-PUB-001`).

---

## 6. `GruLabRotSpec/SURF` Primary Instrument Integration (`SF-SURF-001`–`SF-SURF-008`)

Detailed engineering design and upstream PR specification live in [`docs/SURF_PRIMARY_INTEGRATION_PLAN.md`](SURF_PRIMARY_INTEGRATION_PLAN.md) and [`docs/SURF_UPSTREAM_BRIDGE_PR_SPEC.md`](SURF_UPSTREAM_BRIDGE_PR_SPEC.md).

| ID | Normative Requirement | Implementation Module |
|---|---|---|
| `SF-SURF-001` | `grulab_surf` (`8000.0`–`18000.0` MHz stepped Fabry–Perot cavity FTMW, `SURF@0486b877`) shall be the default `InstrumentProfile`, while retaining `generic_cp_ftmw` and `generic_ir_raman`. | `src/cochem_spyfy/instruments/profiles.py` |
| `SF-SURF-002` | `SurfScanPlanner` shall cluster high-IG transitions into cavity-economical windows ranked by `IG / T_min` (`T_min = iterations * (acq_num / rep_rate + 14.0) / 60.0`) and optionally seed resonant `zaber_pos` from `CavityRuns/*.csv`. | `src/cochem_spyfy/instruments/surf_planner.py` |
| `SF-SURF-003` | Every scan proposal shall pass through an interactive HITL `SurfParameterOverrideTable` and strict Pydantic v2 hardware-envelope validation (`extra="forbid"`, `8000–18000` MHz, Zaber `0–50` mm, RF `-20..+20` dBm with soft warn `> +10` dBm). | `src/cochem_spyfy/instruments/surf_schemas.py`, `surf_planner.py`, `gui.py` |
| `SF-SURF-004` | Tier-1 zero-change export shall emit valid `*_control_config.toml`, `*_scan_settings.toml`, and SHA-256 `surf_scan_manifest.json` without requiring modifications to upstream `SURF`. | `src/cochem_spyfy/instruments/surf_exporter.py` |
| `SF-SURF-005` | `SurfRunWatcher` and `SurfOutputParser` shall ingest live `CavityFiles/*.csv` and finalized CSVs with non-destructive SHA-256 provenance, and `pick_surf_doppler_peaks` shall emit coaxial Doppler doublet candidates strictly as `PROPOSED`. | `src/cochem_spyfy/instruments/surf_ingest.py`, `surf_watcher.py` |
| `SF-SURF-006` | Tier-2 opt-in local bridge (`SurfLocalBridgeClient`) shall bind/connect strictly to `127.0.0.1` loopback, require constant-time bearer auth, enforce hardware safety envelopes, and never autonomously actuate hardware without two-step operator confirmation bound to `(staged_id, settings_sha256)`. | `src/cochem_spyfy/instruments/surf_bridge_client.py`, `surf_safety.py` |
| `SF-SURF-007` | `abort_scan` shall have unconditional priority across all client/bridge states; `CoChem-SPYFY` shall never directly open serial (`COM3`/`COM5`), GPIB/VISA, or RF switch DLL handles (`SURF` remains sole hardware owner). | `src/cochem_spyfy/instruments/surf_safety.py`, `surf_bridge_client.py` |
| `SF-SURF-008` | `spyfy_provider()` shall publish `default_instrument_profile`, `instrument_profiles`, `surf_compatibility`, and additive handoff operations `plan_surf_scan` and `ingest_surf_run`. | `src/cochem_spyfy/module_provider.py` |

---

## 7. Pipeline 5.0.0 Verification & Release Gates

1. **Zero-Mock & Anti-Spoof Gate:** `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider tests/test_zero_mock_and_no_sim_gate.py` passes with `0` findings and empty `.anti_spoof_amnesty.json`.
2. **Full SPYFY + SURF Verification Suite:** `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider` across all suites declared in `pytest.ini` passes with `0` failures and `0` tracked `.pyc` artifacts.
3. **Clean Packaging Gate:** `python -m build --no-isolation` (or `pip wheel --no-deps`) produces a wheel containing exclusively `cochem_spyfy/` and `CoChem_SPYFY-*.dist-info/` with zero `cochem_base/`, `cochem/`, `cochem_topos/`, or `core/` contamination.
