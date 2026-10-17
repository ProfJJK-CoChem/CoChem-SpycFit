# Phase 0 inventory

| Field | Value |
| :--- | :--- |
| Commit measured | `3f3c61a` (branch `ccr-c1697164-ao6hz4`: `main` at `488dc87` merged into PR #1) |
| Commit after fixes | `e87f6f4` |
| Date | 2026-10-16 |
| Where | A clone outside the repository, in a cloud container. Not CI: GitHub Actions jobs on this account fail before a runner is assigned. |
| Environment | Python 3.13.16; numpy 2.5.3, scipy 1.18.1, jax 0.11.2, pytest 9.1.1, h5py 3.16.0, pydantic 2.13.5, ase 3.29.0; declared dependencies installed unpinned (zarr 3.4.1, pyarrow 26.0.0, plotly 7.1.0, ipywidgets 8.1.9, mendeleev 1.3.0, platformdirs 4.13.3, filelock 4.2.0, pyzmq 27.2.0); no GPU, no cupy |
| Genuine engines | SPFIT and SPCAT built from https://spec.jpl.nasa.gov/ftp/pub/calpgm/ with the shipped Makefile (gcc 13.3); sources and examples match `docs/handoff/calpgm_sources_2026-10-11.sha256`. Binary SHA-256: spfit `784ce379…2012071`, spcat `8d3eb357…a533c`. |

## 1. Installation

- `requirements.txt` pins (`jax==0.4.13`, `jaxlib==0.4.13`, `cupy-cuda12x==12.2.0`, …) cannot be
  installed on Python 3.12 or later: the oldest jaxlib with a Python 3.12 wheel is 0.4.18, and
  `cupy-cuda12x==12.2.0` has wheels for CPython 3.8-3.11 only.
- Imports used by tests but declared nowhere: `scikit-learn` (17 test modules fail to import
  without it), `zstandard`, `networkx`, `PyQt6`/`PySide6`. They were installed for the run except
  the Qt bindings.

## 2. What the configured suite runs

- `pytest.ini` sets `testpaths = tests/test_provenance_fair_security_suite.py`, one file. That
  file imports the vendored `cochem_base.gui` package, which needs PyQt6 or PySide6, so it fails
  at collection: **the configured suite runs no test.**
- `pytest tests` collects 1,193 tests in 61 modules with 2 collection errors:
  `test_provenance_fair_security_suite.py` (PyQt6) and `test_spycfit.py` (it opens
  `tests/fixtures/test_arrays.h5` at import; `*.h5` is gitignored, so the file is not in the
  repository).
- `pytest tests` stops at the collection errors. With `--continue-on-collection-errors` the
  process is killed (exit 137) midway: `TestOrphanSweeperPhase4::test_sweep_orphan_active_subprocess`
  calls `sweep_orphan_process` on a child in pytest's own process group, and the function sent
  SIGKILL to that whole group. Fixed in `e87f6f4`; production calls start SPCAT in a new session
  and were not affected. On POSIX no whole-suite run could have completed before the fix.

## 3. Which package tree is imported

`pytest.ini` puts `src` before `.` on `sys.path`, so tests import `src/cochem_spycfit/`. The flat
`cochem_spycfit/` tree mixes re-export shims (for example `core_engine/cochem_spycfit_init.py`)
with byte-identical copies (for example `serialization/cochem_spycfit_spcat_wrapper.py`,
`cochem_spycfit_export.py`, `cochem_spcat_bridge.py`). `src/` also vendors `cochem_base` and
`cochem` packages that belong to CoChem-BASE; several test modules test them
(`test_cochem_base_*`, `test_tripartite_airgap_topology.py`). Which tree survives is D-SPYC-15.

## 4. Results, module by module

Each module was run as `pytest tests/<module>.py` with a 900 s and 6 GB cap, because the
whole-suite run was killed (§2).

| Run | Tests | Passed | Failed | Errors | Skipped |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline `3f3c61a` | 1,117 | 1,036 | 78 | 2 | 1 |
| Baseline, killed module | 31 tests in `test_cochem_spcat_bridge_phase4.py` produced no result | | | | |
| After `e87f6f4`, `SPYCFIT_CALPGM_DIR` set | 1,178 | 1,097 | 78 | 2 | 1 |

Every test present in both runs has the same outcome. The 61 additional results are the 31
phase-4 tests that now run and 30 new tests (§6). Without `SPYCFIT_CALPGM_DIR`, 13
genuine-engine tests fail by design; they never skip.

The 80 remaining failures and errors, classified from their messages:

| Cause | Count |
| :--- | ---: |
| Test data that is gitignored and absent (`test_arrays.h5`, `landscape.h5`, generated catalogs) | 54 |
| File permissions checks that cannot fail when run as root | 7 |
| Hard-coded Windows paths (`D:\__CoChem\…`) | 6 |
| `nbformat` not installed | 4 |
| No TORQ file-drop or ZeroMQ peer (timeout) | 3 |
| Other: an RI/RIJCOSX functional-family enum (2, vendored ORCA code); a parquet lock test (1) | 3 |
| `test_spfit_live_execution` and its async twin: dictionary-based `run_spfit` (§5) | 2 |
| PyQt6/PySide6 not installed | 1 |

## 5. Misleading evidence found

In addition to the seven findings of amendment §5, all confirmed:

1. **The SPFIT wrapper never ran against genuine SPFIT.** `run_spfit` wrote a `.var` (SPFIT
   reads `.par`) without an option line and with control fields NLINE = 0, NITR = 0,
   ERRTST = 1.0 and CAL = 50; SPFIT exited with code 1. `run_spcat` wrote an `.int` in the wrong
   field order and SPCAT waited on standard input until the timeout. Every "live SPFIT" test
   therefore ended in the ASE/EMT substitution or in a missing-binary error.
2. **The `.fit` parser did not match genuine output.** On the JPL HCO example it would read
   the a priori parameter block as (code, value, error) = (1, 1, 120000000) and match no
   residual line (`calfit.c` prints `NNNNN:`), so its RMS came from `MICROWAVE RMS` alone.
3. **Invented `.fit` fixtures labelled authentic**: `authentic_fit_output_text` in
   `tests/test_cochem_spycfit_spcat_wrapper.py` and `AUTHENTIC_FIT_OUTPUT` in
   `tests/test_cochem_spycfit_serialization_init.py` use a layout SPFIT does not write.
4. **Sign error in names.** Parsed code 200 was named `DJ` although SPFIT stores −Δ_J there
   (likewise 1100, 2000, 40100, 41000).
5. **Silent defaults that invent data:** an unknown parameter name became code 10000 (A); a
   missing a priori error became 0.0, which in SPFIT constrains the parameter; a missing line
   uncertainty became 0.0100 MHz; a missing dipole became μ_a = 1 D and an unknown axis became
   a; the export wrote every `.lin` line with 0.0050 MHz; an empty Fortran field parsed as 0.0.
6. **The export methods paragraph asserted "Watson's A-reduction Hamiltonian in the I^r
   representation"** whatever the fit used; the fit state records neither.
7. **Test inputs presented as authentic**: water frequencies 987654.321 and 1200345.678 MHz in
   `tests/test_cochem_spycfit_physics_core_init.py`; water's dipole placed on the a axis (it
   lies along b) in the wrapper tests; `tests/generate_physical_fixtures.py` lines 48-56
   ("Real H2O transitions (JPL/CDMS)") disagree with JPL `c018003.cat` in ELO, GUP, LGINT and two
   frequencies, then replicate the lines 2,000 times with a frequency ramp (literature review
   R4-11).

## 6. Corrected in `e87f6f4`

Amendment §5 items 1-7, items 1, 2, 4, 5 and 6 above, and the orphan sweep (§2). Tests changed
and why:

| Test | Change |
| :--- | :--- |
| `test_cochem_spycfit_auditor.py` (13 tests) | New status names; explicit test tolerance; `test_disjoint_constant_sets` now expects INCOMPLETE; FAILED reports no maximum difference. 8 tests added. |
| `test_cochem_spycfit_physics_core_init.py::test_parity_auditor_execution` | New status name. |
| `test_cochem_spycfit_spcat_wrapper.py` | ASE/EMT substitution removed from the two live tests (they fail: §4); invented `.fit` fixture removed and `test_parse_fit_file_structure` rewritten on genuine SPFIT output; `.lin` formatting tests pass an explicit uncertainty column; 4 tests added. |
| `test_cochem_spycfit_serialization_init.py` | Name map asserts SPFIT's stored signs; `test_spfit_sandbox_static_fit_parser` uses the genuine `.fit` and checks the invented text is rejected; exporter assertions follow §6 of the export fix. |
| `test_cochem_spycfit_export.py` | Tables, units, unknown uncertainty, absent `.lin`, methods text; the SPFIT stand-in is labelled as an interface stand-in. |
| `test_cochem_spycfit_init.py` | Registry test rejects the empty-input digest; 2 tests added. |
| `test_cochem_spcat_bridge_phase4.py` | Stand-in labelled transport-only; genuine SPCAT test added. |
| `test_cochem_spycfit_pickett_output.py`, `test_cochem_spycfit_sobol.py` | New: 11 genuine-engine and parser tests; 4 analytic-oracle tests. |

## 7. Open follow-ups

- `cochem_spcat_bridge.parse_pickett_fit` and its test still use the invented `.fit` layout.
- The export's legacy `.var` is `name value` text, not a Pickett `.var`.
- A dictionary-to-`.par`/`.int` builder needs explicit option lines and control settings
  (WP-PICK; SRS v2 SF-IO-004, SF-PICK-001).
- The gitignored HDF5 test data (54 failures) must be regenerated from recorded sources or the
  tests rewritten (Phase 2, D-SPYC-15).
- Windows paths, root-dependent permission tests, missing `nbformat`, undeclared dependencies
  and the one-file `testpaths` belong to the Phase 2 hygiene PR.
- The calpgm NO example is DPFIT/DPCAT input. Built with `make dpfit`, DPFIT on the hash-verified
  `no.par`/`no.lin` reads every line and reaches MICROWAVE RMS 0.045678 MHz and RMS ERROR
  1.15158, against 0.045684 MHz and 1.1534 in the shipped 1993 `no.fit`.
