# CoChem-SpycFit Software Requirements Specification, version 2

| Field | Value |
| :--- | :--- |
| Status | **Draft 0: structure only.** No section below is approved. Content is written with the PI, one chapter per PR. |
| Author | PI: Dr. Joshua John Klaassen. Agents draft text; only the PI approves it. |
| Approval | The PI's tag `srs-v2.0` on the commit the PI approves (gate G1). |
| Supersedes | `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md` (historical); `.docs/task1_subsystems_architectural_specification.md` (a CoChem-BASE document, not SpycFit's); `docs/legacy/2026-08-07_*` plans. |
| Parent documents | `CoChem-BASE/Method_Matrix.md` v4 (2026-08-09). `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md` (accepted corrections to the Method Matrix). `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`: §1.2 ownership, §9.1 TORQ-SPEC-001 spectral-catalog stage, §12 exports, §15 evidence classes and V-CONSUMER, §16 definition of done. `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md`: D05, D08, D10, D11, D12, ARCH-12. |
| Working inputs | `docs/handoff/PI_DECISIONS.md`; `docs/wiki/review/CoChem-SpycFit_SRS_Review.md` §5 and §8, as corrected by `docs/handoff/REVIEW_ERRATA.md`. |

Drafting rules for this document:

- Each requirement has these fields:
  - id `SPYC-<AREA>-NNN`, never reused and independent of section numbers;
  - title;
  - one SHALL statement with one verifiable predicate;
  - rationale;
  - source (Method Matrix section, TORQ id, or PI decision id);
  - priority (Core / Research profile / Optional);
  - verification method (Test / Analysis / Inspection / Demonstration);
  - acceptance test id `V-*` with pass condition and tolerance;
  - owning module;
  - status.
- Status values:
  - Agents write only `proposed`.
  - The PI sets `approved`.
  - `implemented` and `verified` are never written by hand. They come from the CI-generated
    traceability report (`@pytest.mark.req` markers plus evidence reports), and the PI confirms
    them at G3.
- Until D-SPYC-14 is answered, a requirement's tolerance field reads "TBD (D-SPYC-14)"; the
  requirement is otherwise complete.
- No requirement may depend on a PI decision that is still open. Mark such text "blocked on
  D-SPYC-nn".
- SHALL, SHOULD and MAY are defined in §0 and used only as defined.
- No citation without a DOI or URL and one line stating what it supports.

---

## 0. Document control

_To write: version, date, change history, standards statement (ISO/IEC/IEEE 29148:2018), modal
verb definitions, list of superseded documents._

## 1. Introduction

### 1.1 Purpose

### 1.2 Product definition and scope
_Blocked on D-SPYC-01, D-SPYC-06 and D-SPYC-16. Draft from review §8.6, amended per errata
row 16. Include an explicit "is" list and "is not" list._

### 1.3 Product perspective
_BASE / TOPOS / TORQ / SpycFit data flow and ownership (TORQ SRS §1.2, §3; TORQ D12), including
the catalog boundary decided in D-SPYC-16. Detail lives in the architecture document;
summarise here._

### 1.4 Users and user classes
_PI and research spectroscopists; students in BASE Codespaces; operators; autonomous agents._

### 1.5 Definitions, acronyms, symbols and glossary
_This section is the glossary; Appendix B points here. Terms: B_e, B_0, ΔB_vib = B_0 − B_e;
Watson A/S reductions; I^r, III^r, III^l; Pickett file roles; Product A/B/C; [M]/[D]/[E]; Wang
blocks; s = 4V3/(9F); T_rot; spin statistical weight._

### 1.6 References

### 1.7 Conventions
_Covers:_
- _units (MHz, cm⁻¹, K, D, u, Å);_
- _constants profile and edition, and the isotope-mass source and version (D-SPYC-02);_
- _E′ upper / E″ lower; residual r = ν_obs − ν_calc; weights W = 1/σ²;_
- _lineshape parameters;_
- _sign conventions for distortion constants in Watson form and in SPFIT storage;_
- _SPCAT quantum-number letter encoding (TORQ contract §4)._

## 2. Scientific products and invariants
_What SpycFit claims and does not claim. Computational domain and defaults (D-SPYC-20).
Experimental domain. Uncertainty labelling rules (TORQ contract §3.2: calibrated vs
uncalibrated; Method Matrix bands are not prediction intervals)._

## 3. Use cases
_Candidates from the review: assign a new CP-FTMW spectrum; refine an existing SPFIT fit;
isotopologue campaign and structure; internal rotation or tunnelling fit; refinement request;
publication export; headless batch fit. The PI selects and orders them._

## 4. External interfaces

### 4.1 Pickett files and the pinned SPFIT/SPCAT
_Pin and use rights per D-SPYC-17. SPFIT rewrites `.par`/`.var` in place; always run on scratch
copies._

### 4.2 Experimental data input (spectra, peak lists, `.lin`)

### 4.3 TORQ spectroscopy bundle consumer (TORQ-EXP-004, V-CONSUMER)
_Joint with TORQ. TORQ's HDF5 exporter does not exist yet; its legacy `tar.zst` payload exporter
is not a contract (errata row 8)._

### 4.4 BASE integration
_Covers `cochem.modules`, `cochem.module-handoff/1`, artifact and scratch roots and UI hosting.
BASE already declares `pickett_export`, `pgopher_export` and `spectroscopy_snapshot` boundaries
for SpycFit; accept, rename or retire each. Handoff artifact kinds for spectra, Pickett sets and
refinement requests do not exist in BASE yet. Dependency per D-SPYC-19._

### 4.5 Command-line and Python API

### 4.6 Cross-validation programs (PGOPHER, XIAM / BELGI / RAM36, STRFIT / KRA), if selected

## 5. Functional requirements
_One subsection per area, drafted after D-SPYC-01 (and D-SPYC-16 for HAM and INT). Candidate
areas:_
- _ING: ingestion and validation._
- _HAM: Hamiltonian, labelling, hyperfine, multi-state (D-SPYC-18, D-SPYC-07)._
- _LAM: internal rotation and tunnelling (D-SPYC-04)._
- _INT: intensities, spin weights, selection rules, catalogs (D-SPYC-13)._
- _FIT: SPFIT driver, parameter state, uncertainties (errata row 14)._
- _DIAG: inertial defect and planar moments (D-SPYC-12)._
- _ASSIGN: prediction windows, pattern recognition, the assignment loop._
- _ISO / STR: isotopologues and structure (D-SPYC-10)._
- _SESSION: project state._
- _UI (D-SPYC-05)._
- _OUT: Pickett set and publication tables._
- _FB: refinement request (D-SPYC-03)._

## 6. Non-functional requirements
_Toolchain (D-SPYC-11), numerical precision, determinism, resource bounds on named hosts
(measured), truthful failure (typed errors, explicit missing values, no substitute fallbacks),
provenance, installability, performance targets each with dataset, host and the CI job that
measures it._

## 7. Data requirements
_Records (spectrum, peak list, assignment, fit session, result bundle), missing-value semantics,
provenance fields, schema versions._

## 8. Verification

### 8.1 Evidence classes
_Adopt TORQ SRS §15.1: no fabricated engine output; a claimed agent audit is not evidence;
skips are counted and never satisfy a requirement._

### 8.2 Acceptance catalog (`V-*`)

### 8.3 Benchmark manifest
_Blocked on D-SPYC-14. Tolerances frozen before implementation._

### 8.4 Release scenarios

## 9. Work packages
_Work packages are defined in `docs/plan/CoChem-SpycFit_Plan.md`, which is authoritative. This
chapter lists the requirement ids per package._

## 10. Traceability
_Requirement to source; requirement to acceptance test; test to module. Generated by CI from
`@pytest.mark.req` markers once code exists._

## 11. Decision register
_Summary table of D-SPYC-nn linking to `docs/decisions/D-SPYC-nn.md`, the authoritative records._

## Appendices
_A. Hamiltonian conventions and matrix elements. B. Glossary (see §1.5). C. Module map.
D. Disposition of every review finding (accepted, changed or rejected, with reason). E. Backlog
of unverified suggestions (not requirements)._
