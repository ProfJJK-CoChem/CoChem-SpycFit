# CoChem-SpycFit Software Requirements Specification, version 2

| Field | Value |
| :--- | :--- |
| Status | **Draft 0: structure only.** No section below is approved. Content is written with the PI, one chapter per PR. |
| Author | PI: Dr. Joshua John Klaassen. Agents draft text; only the PI approves it. |
| Approval | Tag `srs-v2.0` on the commit the PI approves (gate G1). |
| Supersedes | `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md` (historical); `.docs/task1_subsystems_architectural_specification.md` (CoChem-BASE document, not SpycFit's); `docs/legacy/2026-08-07_*` plans. |
| Parent documents | `CoChem-BASE/Method_Matrix.md` v4 (2026-08-09); `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`; `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md` (§1.1 ownership, §12 exports, §15 evidence classes); `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md` (D08, D10, D11, ARCH-12). |
| Working inputs | `docs/handoff/PI_DECISIONS.md`; `docs/wiki/review/CoChem-SpycFit_SRS_Review.md` §5 and §8, corrected by `docs/handoff/REVIEW_ERRATA.md`. |

Drafting rules for this document:

- Each requirement has: id `SPYC-<AREA>-NNN`, title, one SHALL statement with one verifiable
  predicate, rationale, source (Method Matrix section, TORQ id, or PI decision id), priority
  (Core / Research profile / Optional), verification method (Test / Analysis / Inspection /
  Demonstration), acceptance test id `V-*` with pass condition and tolerance, owning module,
  status (proposed / approved / implemented / verified).
- SHALL, SHOULD and MAY are defined in §0 and used only as defined.
- Ids are never reused and do not depend on section numbers.
- No requirement may depend on a PI decision that is still open. Mark such text "blocked on
  D-SPYC-nn".
- No citation without a DOI or URL and one line stating what it supports.

---

## 0. Document control

_To write: version, date, change history, standards statement (ISO/IEC/IEEE 29148:2018), modal
verb definitions, list of superseded documents._

## 1. Introduction

### 1.1 Purpose
### 1.2 Product definition and scope
_Blocked on D-SPYC-01 and D-SPYC-06. Draft from review §8.6 once decided. Include an explicit
"is" list and "is not" list._
### 1.3 Product perspective
_BASE / TOPOS / TORQ / SpycFit data flow and ownership (TORQ SRS §1.1, §3). Detail lives in the
architecture document; summarise here._
### 1.4 Users and user classes
_PI and research spectroscopists; students in BASE Codespaces; operators; autonomous agents._
### 1.5 Definitions, acronyms and symbols
_B_e, B_0, ΔB_vib = B_0 − B_e; Watson A/S reductions; I^r, III^r, III^l; Pickett file roles;
Product A/B/C; [M]/[D]/[E]; Wang blocks; s = 4V3/(9F); T_rot; spin statistical weight._
### 1.6 References
### 1.7 Conventions
_Units (MHz, cm⁻¹, K, D, u, Å), constants source and edition (D-SPYC-02), isotope-mass table,
E′ upper / E″ lower, residual r = ν_obs − ν_calc, weights W = 1/σ², lineshape parameters, sign
conventions for distortion constants in Watson form and in SPFIT storage._

## 2. Scientific products and invariants
_What SpycFit claims and does not claim. Computational domain (J_max rule, frequency window,
default T_rot). Experimental domain. Uncertainty labelling rules (TORQ contract §3.2: calibrated
vs uncalibrated)._

## 3. Use cases
_Candidates from the review: assign a new CP-FTMW spectrum; refine an existing SPFIT fit;
isotopologue campaign and structure; internal rotation or tunnelling fit; refinement request;
publication export; headless batch fit. The PI selects and orders them._

## 4. External interfaces
### 4.1 Pickett files and the pinned SPFIT/SPCAT
### 4.2 Experimental data input (spectra, peak lists, `.lin`)
### 4.3 TORQ spectroscopy bundle consumer (TORQ-EXP-004, V-CONSUMER)
_Joint with TORQ. TORQ's exporter does not exist yet (REVIEW_ERRATA #8)._
### 4.4 BASE integration (`cochem.modules`, `cochem.module-handoff/1`, artifact and scratch roots, UI hosting)
### 4.5 Command-line and Python API
### 4.6 Cross-validation programs (PGOPHER, XIAM / BELGI / RAM36, STRFIT / KRA), if selected

## 5. Functional requirements
_One subsection per area, drafted after D-SPYC-01. Candidate areas: ING (ingestion and
validation), HAM (Hamiltonian, labelling, hyperfine, multi-state), LAM (internal rotation,
tunnelling; per D-SPYC-04), INT (intensities, spin weights, selection rules, catalogs), FIT
(fitting, Jacobian, uncertainties, parameter state), DIAG (inertial defect, planar moments),
ASSIGN (prediction windows, pattern recognition, assignment loop), ISO / STR (isotopologues,
structure; per D-SPYC-10), SESSION (project state), UI (per D-SPYC-05), OUT (Pickett set,
publication tables), FB (refinement request; per D-SPYC-03)._

## 6. Non-functional requirements
_Numerical precision, determinism, resource bounds on named hosts (measured), truthful failure
(typed errors, explicit missing values), provenance, installability, performance targets each
with dataset, host and the CI job that measures it._

## 7. Data requirements
_Records (spectrum, peak list, assignment, fit session, result bundle), missing-value
semantics, provenance fields, schema versions._

## 8. Verification
### 8.1 Evidence classes
_Adopt TORQ SRS §15.1: no fabricated engine output; a claimed agent audit is not evidence;
skips are counted._
### 8.2 Acceptance catalog (`V-*`)
### 8.3 Benchmark manifest
_Blocked on D-SPYC-14. Tolerances frozen before implementation._
### 8.4 Release scenarios

## 9. Work packages and definition of done
_Mirror `docs/plan/CoChem-SpycFit_Plan.md` once approved._

## 10. Traceability
_Requirement to source; requirement to acceptance test; test to module. Generated by CI from
`@pytest.mark.req` markers once code exists._

## 11. Decision register
_Summary table of D-SPYC-nn with the PI's answers; full records in `docs/decisions/`._

## Appendices
_A. Hamiltonian conventions and matrix elements. B. Glossary. C. Module map. D. Disposition of
every review finding (accepted / changed / rejected, with reason). E. Backlog of unverified
suggestions (not requirements)._
