# CoChem-SpycFit architecture

| Field | Value |
| :--- | :--- |
| Status | **Draft 0: structure and proposals only.** Nothing here is approved. The PI approves this document together with SRS v2 at gate G1. |
| Depends on | `docs/srs/CoChem-SpycFit_SRS_v2.md`; PI decisions D-SPYC-01, 03, 04, 05, 11, 15 in `docs/handoff/PI_DECISIONS.md`. |
| Model | `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md` §3 (context flowchart and component table with required error behaviour). |
| Proposal source | Review finding ARCH-30 (`docs/wiki/review/CoChem-SpycFit_SRS_Review_findings.json`), corrected by `docs/handoff/REVIEW_ERRATA.md`. |

The text in §2-§7 is a **proposal** for discussion with the PI. Do not implement from it before
G1.

---

## 1. Context

_To write after D-SPYC-01: one diagram and one table._

Proposed context, consistent with the TORQ SRS ownership statement:

- **TOPOS** finds conformers and binding topologies.
- **TORQ** runs quantum chemistry and exports predicted constants as a versioned HDF5 bundle
  (TORQ-EXP-004) and Pickett prediction inputs (TORQ-EXP-002). The exporter does not exist yet.
- **BASE** discovers modules (`cochem.modules` entry point), hands work over with
  `cochem.module-handoff/1` manifests (`load_module_handoff`), supplies artifact and scratch
  roots, hosts the Voila UI and owns durable task state (TORQ D08).
- **The user** supplies the experimental spectrum or peak list, nuclear-spin weights and
  assignment decisions.
- **SpycFit** predicts, supports assignment, fits through the pinned SPFIT/SPCAT and exports
  results to the BASE artifact root.

## 2. Package layout (proposal)

A single package `src/cochem_spycfit/`, pure Python, CPU first. Candidate subpackages:

| Subpackage | Responsibility | Open decisions |
| :--- | :--- | :--- |
| `io/pickett` | Read and write `.par`, `.var`, `.int`, `.lin`, `.cat`, `.egy`, `.fit` with their roles; round-trip against the pinned binaries; pyckett compatibility. | none |
| `io/experimental` | Spectrum and peak-list input; peak picking with an uncertainty model. | Instrument formats from the PI |
| `contract` | TORQ bundle reader, schema-version negotiation, BASE handoff receiver, configuration model. | Joint contract with TORQ |
| `model` | Effective Hamiltonian forward model (Watson A/S, quadrupole hyperfine, multi-state); eigenvalue labelling; intensities with user spin weights; diagnostics (κ, inertial defect, planar moments). Must reproduce SPCAT before its output is shown. | D-SPYC-01, 04, 11 |
| `fit` | Driver for the pinned SPFIT as fit of record (subprocess, scratch directory, timeouts, parsers); in-process sensitivity and what-if support. | D-SPYC-01 |
| `assign` | Prediction catalogs, search windows, candidate ranking, pattern tools, the human assignment loop, `.lin` generation from approved assignments only. | D-SPYC-13, 14 |
| `campaign` | Isotopologues, conformers, structure fits gated by fluxionality. | D-SPYC-10 |
| `session` | One versioned project file per campaign; single writer; event log. | none |
| `ui` | One UI stack calling the same headless API. | D-SPYC-05 |
| `export` | Complete Pickett set, publication tables, refinement request, BASE handoff back. | D-SPYC-03 |

## 3. State (proposal)

- One project file per campaign under the BASE artifact root, written by one writer.
- Kernel memory is a cache, never the record.
- SPFIT/SPCAT work files go under the BASE scratch root.
- The source tree is never written at run time.
- No shared databases and no daemons. SpycFit never writes another module's task state.

## 4. Process boundaries (proposal)

- In process: file I/O, forward model, assignment, UI, export.
- Subprocess: the pinned SPFIT and SPCAT (and XIAM, PGOPHER or STRFIT if selected), located
  through configuration.
- Never launched by SpycFit: TORQ, ORCA, CFOUR, PySCF or any quantum-chemistry engine.

## 5. Deployment targets (proposal)

Codespaces under BASE (default for students), local Linux/WSL2/macOS workstation, HPC login node.
GitHub Actions is CI only. Each target needs its own acceptance evidence: clean install,
SPFIT/SPCAT round trip, UI journey on a reference complex.

## 6. Error behaviour

_To write: per component, the typed errors and the explicit missing-value states it can return
(model on TORQ SRS §3 component table and TORQ-DATA-002 absent states)._

## 7. Mapping to requirements

_To write after SRS v2 ids exist: component to requirement ids to acceptance tests._
