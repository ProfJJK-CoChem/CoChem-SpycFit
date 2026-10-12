# CoChem-SpycFit architecture

| Field | Value |
| :--- | :--- |
| Status | **Draft 0: structure and proposals only.** Nothing here is approved. The PI approves this document together with SRS v2 at gate G1. |
| Depends on | `docs/srs/CoChem-SpycFit_SRS_v2.md`; PI decisions D-SPYC-01, 02, 03, 04, 05, 07, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 and 20 in `docs/handoff/PI_DECISIONS.md`. |
| Model | `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md` §3 (context flowchart and component table with required error behaviour). |
| Proposal source | Review finding ARCH-30 (`docs/wiki/review/CoChem-SpycFit_SRS_Review_findings.json`), corrected by `docs/handoff/REVIEW_ERRATA.md`. |

The text in §1-§5 is a **proposal** for discussion with the PI. Do not implement from it before
G1.

---

## 1. Context

_To write after D-SPYC-01 and D-SPYC-16: one diagram and one table._

Proposed context, consistent with the TORQ SRS ownership statement (§1.2):

- **TOPOS** finds conformers and binding topologies.
- **TORQ** runs quantum chemistry. It exports predicted constants as a versioned HDF5 bundle
  (TORQ-EXP-004) and Pickett prediction inputs (TORQ-EXP-002). It produces predicted catalogs
  (TORQ-SPEC-001 spectral-catalog stage, D12) through its own pinned SPCAT backend.
  - The HDF5 bundle exporter does not exist yet.
  - The legacy `tar.zst` payload exporter in TORQ `Libraries/` is not a contract.
- **BASE** owns the shared services:
  - discovers modules (`cochem.modules` entry point; `spycfit` registered);
  - hands work over with `cochem.module-handoff/1` manifests (`load_module_handoff`), which today
    accept only geometries, Hessians and calculation-result JSON;
  - supplies artifact and scratch roots;
  - hosts the Voila UI;
  - owns durable task state (TORQ D08).
  BASE also declares `pickett_export`, `pgopher_export` and `spectroscopy_snapshot` boundaries
  for SpycFit (`CoChem-BASE/src/cochem_base/interfaces/cochem_pyckett_bridge.py`,
  `cochem_pgopher_bridge.py`, `cochem_vibspyc_snap.py`; not the vendored copy in this
  repository), which SRS v2 §4.4 must accept, rename or retire.
- **The user** supplies the experimental spectrum or peak list, nuclear-spin weights and
  assignment decisions.
- **SpycFit** supports assignment and fits through the pinned SPFIT/SPCAT. It produces catalogs
  from fitted or edited constants (per D-SPYC-16) and exports results to the BASE artifact root.

## 2. Package layout (proposal)

A single package `src/cochem_spycfit/`, pure Python, CPU first. Candidate subpackages:

| Subpackage | Responsibility | Open decisions |
| :--- | :--- | :--- |
| `io/pickett` | Read and write `.par`, `.var`, `.int`, `.lin`, `.cat`, `.egy`, `.fit` with their roles; round-trip against the pinned binaries on scratch copies. | D-SPYC-17 (pin); whether `pyckett` is a runtime dependency or a test-only cross-check |
| `io/experimental` | Spectrum and peak-list input; peak picking with an uncertainty model. | Instrument formats from the PI |
| `contract` | TORQ bundle reader and schema-version negotiation (later); BASE handoff receiver; configuration model. | D-SPYC-19; joint contracts with TORQ and BASE |
| `model` | Only if D-SPYC-18 is B or C: in-process Hamiltonian (Watson A/S, quadrupole hyperfine, multi-state) and eigenvalue labelling. Each term must reproduce SPCAT before its output is shown. | D-SPYC-07, 11, 16, 18 |
| `diag` | κ, inertial defect and planar moments from any set of constants, checked against published values. | D-SPYC-12 |
| `catalog` | Intensities with user spin weights and catalogs at T_rot, through the pinned SPCAT (or `model` where D-SPYC-18 allows). | D-SPYC-13, 16, 17, 20 |
| `fit` | Driver for the pinned SPFIT as fit of record (subprocess, scratch directory, timeouts, parsers); in-process sensitivity and what-if support. | D-SPYC-01 |
| `assign` | Prediction catalogs, search windows, candidate ranking, pattern tools, the human assignment loop, `.lin` generation from approved assignments only. | D-SPYC-13, 14 |
| `campaign` | Isotopologues, conformers, and structure fits gated by fluxionality. | D-SPYC-10 |
| `session` | One versioned project file per campaign; single writer; event log. | none |
| `ui` | One UI stack calling the same headless API. | D-SPYC-05 |
| `export` | Complete Pickett set, publication tables, refinement request, BASE handoff back. | D-SPYC-03; a new BASE handoff artifact kind |

## 3. State (proposal)

- One project file per campaign under the BASE artifact root, written by one writer.
- Kernel memory is a cache, never the record.
- SPFIT/SPCAT work files go under the BASE scratch root.
- The source tree is never written at run time.
- No shared databases and no daemons. SpycFit never writes another module's task state.

## 4. Process boundaries (proposal)

- In process: file I/O, the optional model, assignment, UI and export.
- Subprocess: the pinned SPFIT and SPCAT (and XIAM, PGOPHER or STRFIT if selected), located
  through configuration. A missing program raises a typed error; nothing substitutes for it.
- Never launched by SpycFit: TORQ, ORCA, CFOUR, PySCF or any quantum-chemistry engine.

## 5. Deployment targets (proposal)

- Codespaces under BASE (the default for students).
- A local Linux, WSL2 or macOS workstation.
- An HPC login node.

GitHub Actions is CI only. Each target needs its own acceptance evidence: a clean install, the
SPFIT/SPCAT round trip, and a UI journey on a reference complex.

## 6. Error behaviour

_To write: per component, the typed errors and the explicit missing-value states it can return
(model on the TORQ SRS §3 component table and the TORQ-DATA-002 absent states)._

## 7. Mapping to requirements

_To write after SRS v2 ids exist: component to requirement ids to acceptance tests._
