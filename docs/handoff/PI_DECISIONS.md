# PI decision sheet for CoChem-SpycFit

Every entry starts as **OPEN**. Agents may add options and evidence, but never answers.

**How an answer is recorded.** When the PI answers, an agent pastes the PI's verbatim words, the
date and the source (PR or issue link, or "conversation YYYY-MM-DD") under the entry, and sets
**ANSWERED**. The agent also writes `docs/decisions/D-SPYC-nn.md` from the template in
`docs/decisions/README.md`. The entry becomes **DECIDED** only when the PI approves the PR that
contains the record, in a PR comment as defined in `AGENTS.md`. Answers relayed by other agents
or scripts are not recorded.

`docs/decisions/D-SPYC-nn.md` is the authoritative record. This sheet and SRS v2 §11 link to it.

Sources:
- "Review": `docs/wiki/review/CoChem-SpycFit_SRS_Review.md`.
- "MM": `CoChem-BASE/Method_Matrix.md` v4.
- "TORQ contract": `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`.
- "TORQ SRS": `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`.
- "TORQ D-nn": `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md`.

## Summary

| Id | Question | Needed before | Recommendation | Status |
| :--- | :--- | :--- | :--- | :--- |
| D-SPYC-01 | What SpycFit is; fitter of record | any functional requirement | A: assignment and fitting workbench; pinned SPFIT/SPCAT is the fit of record | ANSWERED 2026-10-16 |
| D-SPYC-16 | Catalog and Hamiltonian boundary with TORQ | SRS §1.2, §1.3, HAM and INT | A: TORQ predicts from computed constants; SpycFit re-predicts from fitted or edited constants | OPEN |
| D-SPYC-02 | Masses and constants | SRS §1.7 | Atomic masses with source and version; TORQ's CODATA 2022 profile, shared | OPEN |
| D-SPYC-03 | Feedback to TORQ | SRS FB requirements; G1 | A: refinement-request file handed to BASE; a human approves any follow-up | ANSWERED in part 2026-10-16 |
| D-SPYC-04 | Internal rotation and tunnelling | SRS LAM; G1 | C: defer to after the first release, then A or B | ANSWERED in part 2026-10-16 |
| D-SPYC-05 | UI stack and hosting | SRS UI; G1 | ipywidgets + Plotly under Voila as a BASE tab, plus CLI | OPEN |
| D-SPYC-06 | IR / Raman / UV-Vis in SpycFit | SRS §1.2 | None in SpycFit | ANSWERED in part 2026-10-16 and 2026-10-17 (TORQ predicts; UV-Vis open) |
| D-SPYC-07 | Eigenvalue label swaps during a fit | SRS HAM/FIT (only if D-SPYC-18 includes a model) | Halt and show the swap | OPEN |
| D-SPYC-08 | Anti-gaming checks | G2 (hygiene PR) | Keep narrow hygiene checks; gate on literature tests, SPFIT/SPCAT round trip, whole-suite CI, PI review | OPEN |
| D-SPYC-09 | Parallel-execution import ban | G2 | Lift the ban | OPEN |
| D-SPYC-10 | Structure determination in v2 | SRS STR; G1 | In scope, gated by fluxionality | ANSWERED 2026-10-17 (defer to GEOM) |
| D-SPYC-11 | Toolchain baseline | G1 (for SRS §6), Phase 2 code | A: Python 3.12, NumPy/SciPy, no JAX/CuPy, CPU only | OPEN |
| D-SPYC-12 | Inertial-defect sign rule | SRS DIAG | Diagnostic flag, never automatic rejection | OPEN |
| D-SPYC-13 | Dark dipole components | SRS INT/ASSIGN | De-prioritise in search; keep in archival catalogs | OPEN |
| D-SPYC-14 | Benchmark set, data supply and tolerances | G1 | A: minimum set (see entry) | ANSWERED in part 2026-10-16 and 2026-10-17 (panel accepted; tolerances open) |
| D-SPYC-15 | Repository strategy | Phase 2 | A: tag legacy, build fresh in `src/cochem_spycfit/`, port proven modules | OPEN |
| D-SPYC-17 | SPFIT/SPCAT source, pin and use rights | SRS §4.1; Phase 2 CI | A: one ecosystem pin shared with TORQ | ANSWERED in part 2026-10-17 (JPL sources) |
| D-SPYC-18 | In-process Hamiltonian model and its scope | SRS HAM; G1 | B: model limited to what SPCAT reproduces, used for windows and what-if | ANSWERED in part 2026-10-16 |
| D-SPYC-19 | Dependency on the `cochem-base` package | SRS §4.4; Phase 2 | A: pinned dependency | ANSWERED in part 2026-10-17 (BASE required; direct input) |
| D-SPYC-20 | Domain defaults (T_rot, window, J_max) | SRS §2 | T_rot 1.5 K; 2-22 GHz; J_max from population | ANSWERED in part 2026-10-17 (set first, editable) |

---

## D-SPYC-01 Product identity and fitter of record

**Needed before:** any functional requirement. **Status:** ANSWERED 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-01.md`

**Question.** What is SpycFit, and which program's least-squares fit is the published result?

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** A rotational assignment and fitting workbench for the MM §2.1 system class: 5-10 atom weakly bound complexes plus their monomers and isotopologues, measured by CP-FTMW at about 2-22 GHz and T_rot 1-2 K. A pinned SPFIT/SPCAT is the fit of record and export path. SpycFit owns experimental ingestion, prediction windows, the human assignment loop, diagnostics, campaigns and publication export. | Matches MM (the JAX note at BASE l.3632: JAX "not used to replace SPFIT") and the TORQ SRS §1.2 ownership statement. Smallest scientific risk. |
| B. A native fitter of record (own Hamiltonian and least squares), with SPFIT/SPCAT only as a cross-check. | Larger build. Every Hamiltonian term (hyperfine, multi-state, internal rotation) must be implemented and proved equal to SPFIT before release. Review §5.7 FIT-001..005 assume this option (errata row 14). |
| C. A thin front end over PGOPHER, as BASE User Manual Ch. 5 says. | PGOPHER is open source and runs on Windows, macOS and Linux, with a command-line program `pgo` (`--fit`, `--contourfit`) (literature review R3-04; the 2026-10-11 text called it a Windows GUI program). Contradicts the SPFIT-centred deliverable of MM §18: a `.par` + `.int` that runs unmodified through SPCAT. |

**Draft scope statement for option A:** review §8.6, with the amendments in errata row 16. The
text belongs in SRS v2 §1.2. The question of IR/Raman/UV-Vis is D-SPYC-06.

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The agent also asked: "3. When native and SPFIT results both exist, which one a publication table shows first." The PI replied: "3. SPFIT first." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-01.md`. Still open: nothing in this entry. Related open questions are D-SPYC-16 and D-SPYC-18.

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "SpycFit is a computational chemistry-to-assignment/fitting/publication pipeline. It must support rotational assignment/fitting, joint rotational–rovibrational fitting, MLFF-assisted computational assignment, and vibrational band/line/envelope fitting."
> "Run genuine SPCAT/SPFIT alongside native SpycFit for compatible microwave models."
This differs from option A: there is a native path beside SPFIT, and the scope is wider than the MM §2.1 system class. When both exist, publication tables show the SPFIT result first (PI answer above).


## D-SPYC-16 Catalog and Hamiltonian boundary with TORQ

**Needed before:** SRS §1.2, §1.3, HAM and INT requirements. **Status:** OPEN

TORQ's accepted scope includes a "Spectral catalog" stage: TORQ-SPEC-001 "Validated effective
rotational Hamiltonian ... intensity/partition/statistical-weight model", TORQ-EXP-002/003, and
TORQ D12 "the full rotational-spectroscopy route is TORQ scope". TORQ also has its own pinned
SPCAT backend. SpycFit needs catalogs from *fitted* or user-edited constants during assignment.

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** TORQ owns catalogs predicted from computed constants. SpycFit owns catalogs from fitted, experimental or user-edited constants, and the assignment loop. Both use the same pinned SPCAT (D-SPYC-17) and one shared spectrum-equivalence test. | No duplicated physics. Needs agreement with TORQ on the shared pin and test. |
| B. SpycFit owns all catalog generation; TORQ calls SpycFit for its catalogs. | Changes TORQ's accepted scope; needs TORQ's agreement and a reverse dependency. |
| C. Both implement catalogs independently, with cross-validation. | Duplicated work; divergence risk across builds. |

**PI answer:** _none yet_

## D-SPYC-02 Masses and constants

**Needed before:** SRS §1.7. **Status:** OPEN

TORQ-SCI-002 requires "isotope-specific atomic masses with source and version". TORQ implements
constants as `cochem.constants.codata-2022-torq-compatible/1` (CODATA 2022 via SciPy, provider
version recorded; `CoChem-TORQ/docs/development/units_and_constants.md`). TORQ records the
isotope-mass "source database checksum/version"
(`CoChem-TORQ/docs/development/advanced_scientific_results.md`).

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Isotope-specific **atomic** masses with source, version and hash recorded. Constants from TORQ's CODATA 2022 profile, imported or shared rather than copied. h/(8π²) is derived from that profile, never typed in. No nuclear-mass stripping (the old SRS "PCA-32" rule is withdrawn). | One edition across the ecosystem; prediction and fit agree. |
| B. As A, but SpycFit keeps its own copy of the same edition, with a test that compares it against TORQ's values. | Avoids a runtime dependency on TORQ code; duplication is guarded by the test. |
| C. A bundled AME2020 table and CODATA 2018 (the review's proposal). | Diverges from TORQ. Not recommended. |

**PI answer:** _none yet_

## D-SPYC-03 Feedback to TORQ

**Needed before:** SRS feedback requirements; G1. **Status:** ANSWERED in part 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-03.md`

When a fit shows the predictions were wrong (wrong conformer, poor barrier), what does SpycFit
emit?

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** A versioned "refinement request" file in SpycFit's output bundle, handed to BASE. A human approves any TORQ calculation that follows. | Respects TORQ's single task authority (TORQ-ARC-002, TORQ D08). Needs a new artifact kind in BASE's `cochem.module-handoff` schema, which today accepts only geometries, Hessians and calculation-result JSON. That is a cross-repository change agreed with BASE. |
| B. No feedback path in v2. | Simplest; feedback stays manual. |
| C. Direct submission into TORQ or BASE task queues. | Conflicts with TORQ D08 and TORQ-ARC-002. Not recommended. |

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-03.md`. Still open: the hand-off route for refinement requests (options A-C).

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "PES/force-field refinement is required in the overall workflow and may be implemented in another CoChem module."
Still open: the hand-off route (options A-C).


## D-SPYC-04 Internal rotation and tunnelling

**Needed before:** SRS LAM requirements; G1. **Status:** ANSWERED in part 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-04.md`

| Option | Consequence |
| :--- | :--- |
| A. Delegate internal rotation to XIAM / BELGI / RAM36 (or SPFIT's own options) through wrappers; SpycFit ingests their results. | Least code; depends on external programs being installable in CI. |
| B. A native effective torsion-rotation Hamiltonian (RAM or PAM) with XIAM-compatible parameters, validated against XIAM. | More code; one in-process model for assignment. |
| **C (recommended for the first release).** Defer internal rotation; v2 covers semi-rigid species and two-state tunnelling doublets. Revisit A or B afterwards. | Matches TORQ D05's ordering (semi-rigid first, large-amplitude motion later). Leaves NH₃⋯HCOOH (MM §17 system 5; Roehling, Hill, Daly & Kukolich 2024), the only in-class V3 benchmark, out of the first release. |

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-04.md`. Still open: whether internal rotation is delegated to external engines (option A) or built natively (option B), and which engines.

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "Establish independent semirigid comparison first; then separately validate hyperfine, excited-state, and large-amplitude-motion extensions. All remain in scope."
Consistent with option C's ordering; internal rotation stays in scope.


## D-SPYC-05 User interface and hosting

**Needed before:** SRS UI requirements; G1. **Status:** OPEN

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** One stack: ipywidgets + Plotly under Voila, hosted as a BASE tab and runnable as a standalone notebook. Every action calls the same headless Python API, and a CLI handles batch fits. The Qt, Streamlit and plugin variants are deleted. | Matches BASE's Voila hosting; one code path. |
| B. CLI and library only in v2; UI in a later release. | Faster to a tested core; the assignment loop is less usable. |

**PI answer:** _none yet_

## D-SPYC-06 IR / Raman / UV-Vis in SpycFit

**Needed before:** SRS §1.2. **Status:** ANSWERED in part 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-06.md`

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** None in SpycFit. If IR/Raman simulation is still wanted, it moves to TORQ reporting or a separate module (TORQ D12: adjacent domains need explicit sibling handoffs). | Rewrite README §1-3 and ask the BASE maintainer to revise User Manual Ch. 5 (its `%spycfit` block at l.854: `Type = IR  # Options: IR, Raman, UVVIS, Microwave`). |
| B. Keep a minimal IR/Raman stick-spectrum renderer in SpycFit. | Re-opens scope that the 2026-08-07 plan already rejected for UV-Vis/TD-DFT. |

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-06.md`. Still open: UV-Vis, and who produces vibrational and Raman predictions (TORQ or another provider).

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "IR, Raman, THz, near-IR, and microwave are first-class workflows with familiar interactions."
This is option B or wider, against the recommendation.

**PI answer** (conversation 2026-10-17). The agent asked: "**D-SPYC-06** who produces vibrational and Raman predictions: TORQ, another provider or SpycFit". The PI replied: "4. TORQ." On UV-Vis the agent asked: "**D-SPYC-06** UV-Vis: in scope or out (you named IR, Raman, THz, NIR and MW only)". The PI replied: "20. Research then give suggestions, context, pros, cons, and risks. I think the calculations and requirements for fitting UV-Vis share little to no overlap with rotovibrational spectroscopy predictions." Record: `docs/decisions/D-SPYC-06.md`. Still open: UV-Vis.


## D-SPYC-07 Eigenvalue label swaps during a fit

**Needed before:** HAM/FIT requirements, and only if D-SPYC-18 includes an in-process model.
**Status:** OPEN

When a parameter change reorders levels so that a transition's (J, Ka, Kc) label maps to a
different eigenvalue, what should happen?

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Halt the fit and show the swap. | Silent relabelling corrupts residuals; halting forces a human decision. |
| B. Warn and continue. | Faster iteration; risk of unnoticed corruption. |

**PI answer:** _none yet_

## D-SPYC-08 Anti-gaming checks

**Needed before:** the Phase 2 hygiene PR. **Status:** OPEN

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Review §9.4. Keep narrow hygiene checks: no `except: pass`, no tautological asserts, no empty function bodies outside abstract classes. Drop keyword bans and the amnesty file. Gate on literature-value acceptance tests, the SPFIT/SPCAT round trip, whole-suite CI and PI review. | Checks that cannot be gamed replace checks that were. |
| B. Keep the current `ci_tools/anti_spoof_linter.py` rules. | The review documents how these were evaded (GOV-02, GOV-08). |

The parallel-import ban is decided separately in D-SPYC-09.

**PI answer:** _none yet_

## D-SPYC-09 Parallel-execution import ban

**Needed before:** the Phase 2 hygiene PR. **Status:** OPEN

The current linter bans `multiprocessing`, `concurrent.futures`, `dask` and `parsl`; MM uses
Parsl elsewhere.

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Lift the ban. SpycFit's workloads are small, so no new dependency follows. | Removes a rule that conflicts with the ecosystem. |
| B. Keep the ban. | No practical effect on SpycFit, but keeps a conflicting rule. |

**PI answer:** _none yet_

## D-SPYC-10 Structure determination in v2

**Needed before:** SRS STR requirements; G1. **Status:** ANSWERED 2026-10-17; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-10.md`

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Kraitchman r_s, r_0 and r_m fits in v2, gated by a fluxionality check. There is no structure output for species with resolved tunnelling or several populated basins (MM §2.3). The fluxionality information comes from the user until TORQ's bundle carries it. | Covers isotopologue campaigns; the gate prevents meaningless structures for floppy complexes. |
| B. Defer to a later release. | Smaller v2. |
| C. Delegate to Kisiel's STRFIT/KRA. | Less code; depends on external programs. |

**PI answer** (conversation 2026-10-17). The agent asked: "**D-SPYC-10** structure determination (r₀, r_s, r_m): include or defer". The PI replied: "17. Defer to GEOM which is the molecular geometry module." Record: `docs/decisions/D-SPYC-10.md`. GEOM's plan already lists Pickett files and SpycFit JSON payloads as inputs and r₀/r_s/r_m outputs (`CoChem-GEOM/CoChem_GEOM_Workflow_Plan_v4.md`); the payload schema is a cross-repository item.

## D-SPYC-11 Toolchain baseline

**Needed before:** SRS §6 (G1) and Phase 2 code. **Status:** OPEN

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Python 3.12. NumPy/SciPy for any in-process model; matrices are at most a few hundred square, and Hellmann-Feynman derivatives need no autodiff. No JAX, no CuPy, CPU only. | Installs everywhere, runs in Codespaces and CI. Departs from MM's JAX note (BASE l.3632), which keeps JAX for DVR diagonalisation (TORQ's job) and search-space pruning (assignment work), done here with NumPy/SciPy instead. |
| B. Python 3.12 with JAX ≥ 0.11 (FP64 enforced at one entry point) for the model and pruning. | Follows the ecosystem's JAX direction; adds version and precision management. `jax==0.4.13` in `requirements.txt` cannot be installed and lacks `wofz`. |

**PI answer:** _none yet_

## D-SPYC-12 Inertial-defect sign rule

**Needed before:** SRS DIAG requirements. **Status:** OPEN

MM §6.4 makes a sign disagreement between computed and experimental inertial defects a **binding
rejection**. The accepted TORQ contract §3.1 says "a sign discrepancy is a diagnostic, not
sufficient evidence to reject a conformer", because ground-state defects carry vibrational
contributions.

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Follow the TORQ contract. Always compute and show Δ and the planar moments, flag a sign disagreement prominently, and never reject automatically. | Consistent with TORQ; avoids rejecting correct structures of floppy complexes. |
| B. Follow MM §6.4 literally: automatic rejection. | Conflicts with the accepted TORQ contract. |

**PI answer:** _none yet_

## D-SPYC-13 Dark dipole components

**Needed before:** SRS INT and ASSIGN requirements. **Status:** OPEN

MM §6.1 flags |μ_g| < 0.1 D as "dark". The TORQ contract §3.4 says "a 0.1 D display/detection
threshold cannot delete nonzero archival transitions".

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Dark branches are de-prioritised and labelled in search windows, and kept in every archival catalog. The threshold is a user setting. | Consistent with TORQ; no transitions lost. |
| B. Exclude dark branches from predicted catalogs. | Conflicts with the TORQ contract. |

**PI answer:** _none yet_

## D-SPYC-14 Benchmark set, data supply and tolerances

**Needed before:** G1. **Status:** ANSWERED in part 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-14.md`

Review §8.4 proposes 13 systems anchored on MM §17. Several rows depend on other decisions:
rows 6-7 on D-SPYC-04, rows 9-10 on D-SPYC-10, and row 12 on TORQ's exporter. Rows 3-4 need
errata row 1. TORQ D10 requires a preregistered benchmark with observable-specific targets fixed
before execution.

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** A minimum release set: propylene oxide (rigid-rotor regression); Ar-oxazole (¹⁴N hyperfine, near-prolate, b/c-type only); H₂CO-HCl (Cl hyperfine; A fixed and excluded per MM §17); the Pickett HCO example (exact SPCAT/SPFIT regression); constructed exact cases (review row 11). Add SO₂⋯HF once its data are in hand. | Covers the core Hamiltonian, hyperfine and the round trip with modest data collection. |
| B. A plus Ar-ketene (two-state tunnelling, measured dipoles) and an isotopologue set for structure (needs D-SPYC-10). | Broader coverage; more data to obtain. |
| C. The full review §8.4 set, with conditional rows following their decisions. | Most complete; largest data effort. |

**Proposed default tolerances** for the PI to approve or change:
- Refitted constants within the published 1σ (2σ for distortion constants).
- σ_fit within 10 % of the published value.
- SPCAT agreement within the 0.1 kHz print quantum.
- MM §3.1 bands are literature statistics, not prediction intervals (TORQ contract §3.2); they are
  never used as tolerances.

**Data supply:** the PI provides or points to each line list and parameter file. Until the
Phase 2 PR allows fixtures under `tests/`, sources are recorded (DOI or URL, SHA-256) without
committing the files.

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-14.md`. Still open: the benchmark panel and the tolerance values.

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "Establish independent semirigid comparison first" (`docs/HANDOFF.md`); amendment §7 D-03: "Matched semirigid rotational models first, then separately validated extensions with explicit capability profiles."
The literature review (R4-02) proposes a semirigid panel of CO, OCS, SO₂, H₂CO and propyne (SRS v2 §8.3) and notes that HCO is an open-shell case, valid as a file-handling regression but not as semirigid evidence. It also proposes four tolerance tiers (SF-CMP-008a); the forward tier for `.cat` comparisons would be half the 0.1 kHz print quantum plus a numerical bound.

**PI answer** (conversation 2026-10-17). The agent asked: "**D-SPYC-14** first semirigid comparison panel (proposed: CO, OCS, SO₂, H₂CO, propyne)". The PI replied: "2. Yes, proceed." On tolerances the agent asked: "**D-SPYC-14** tolerance tiers and their starting values". The PI replied: "3. Research what limits are considered valid in publications from the last 5 years and give me suggestions." Record: `docs/decisions/D-SPYC-14.md`. Still open: tolerance tiers and values; reconciling the option A set with the accepted panel (R4-12).


## D-SPYC-15 Repository strategy

**Needed before:** Phase 2. **Status:** OPEN

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** The PI tags the current state `legacy-swarm-2026-10`. The new package is built in `src/cochem_spycfit/` from a clean start in this repository, and legacy modules are ported one at a time, only after they pass requirement-linked tests. | History preserved; no ambiguity about which code is current. |
| B. Clean up in place: keep the `src/` tree, delete the rest, fix module by module. | Less churn, but inherits unproven code and the old tests' assumptions. |
| C. A new repository; archive this one. | Cleanest, but breaks links and BASE's registry entry. |

**PI answer:** _none yet_

## D-SPYC-17 SPFIT/SPCAT source, pin and use rights

**Needed before:** SRS §4.1 and the Phase 2 CI work. Decide jointly with the TORQ maintainer.
**Status:** ANSWERED in part 2026-10-17; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-17.md`

TORQ already pins the laserkelvin/Pickett mirror at `f5af2ff` (gcc 14.2, binary hashes recorded,
distribution status `unverified_not_bundled`). Student-image provisioning there is "blocked until
the laboratory establishes appropriate distribution/use rights"
(`CoChem-TORQ/docs/development/pickett_backend.md`). This handoff built the JPL sources instead
(`docs/handoff/calpgm_sources_2026-10-11.sha256`). Those differ from TORQ's tree in `calcat.c`
and the `Makefile`.

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** One ecosystem pin shared with TORQ (TORQ's existing one, or a jointly chosen replacement). One build recipe, one place where verified sources or binaries are kept, and use rights established once. | Cross-module SPCAT parity, which V-CONSUMER and D-SPYC-16 rely on. |
| B. A separate SpycFit pin (for example the JPL files) plus a cross-build catalog-equivalence test against TORQ's build. | Independent, but two builds to maintain and reconcile. |
| C. Kisiel's PROSPE binaries. | Windows-oriented; harder in Linux CI. |

Also decide where verified copies are kept, if JPL or GitHub content changes: a private
repository, a release asset or an Actions cache. Also decide whether `pyckett` is a runtime
dependency or a test-only cross-check.

**PI answer** (conversation 2026-10-17). The agent asked: "**D-SPYC-17** SPFIT/SPCAT source and pin (JPL sources or TORQ's laserkelvin copy), and whether DPFIT/DPCAT are included". The PI replied: "1. JPL sources. What are DPFIT/DPCAT?" Record: `docs/decisions/D-SPYC-17.md`. Conflict: TORQ's validator accepts only laserkelvin/Pickett `f5af2ff` (`CoChem-TORQ/src/cochem_torq/spectroscopy/pickett_backend.py`); whether TORQ moves to the JPL build is a cross-repository question. Still open: DPFIT/DPCAT; where verified copies are kept; `pyckett`; use rights.

## D-SPYC-18 In-process Hamiltonian model and its scope

**Needed before:** SRS HAM requirements; G1. **Status:** ANSWERED in part 2026-10-16; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-18.md`

| Option | Consequence |
| :--- | :--- |
| A. No in-process model in v2. Every prediction and fit goes through SPCAT/SPFIT. | Least code. Interactive what-if views run SPCAT per update, which is fast for these molecules but adds process overhead. |
| **B (recommended).** A model limited to what SPCAT reproduces exactly: Watson A/S in I^r (and III^r / III^l if needed), quadrupole hyperfine for up to two nuclei, multi-state blocks. It is used for windows, assignment and what-if views, never for results of record. Each term must pass a spectrum-equivalence test against SPCAT before it is enabled. | Responsive assignment loop; correctness tied to SPCAT. D-SPYC-07 applies. |
| C. A full native model including internal rotation (only with D-SPYC-01 B or D-SPYC-04 B). | Largest effort. |

TORQ's Pickett backend (`CoChem-TORQ/docs/development/pickett_backend.md`) supports asymmetric-top
J, Ka, Kc with an explicit Watson A or S reduction in I^r. It lists other representations,
hyperfine states, spin weights, tunnelling and multiple vibrational states as unsupported.
Whatever SpycFit needs beyond that is new work under D-SPYC-16 and D-SPYC-17.

**PI answer** (conversation 2026-10-16). The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet." The PI replied: "1. That is correct." The confirmed direction is quoted below. Record: `docs/decisions/D-SPYC-18.md`. Still open: which native terms, in which order.

**Direction confirmed by the PI on 2026-10-16** (first recorded from `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`):

> "Run genuine SPCAT/SPFIT alongside native SpycFit for compatible microwave models."
A native model therefore exists for compatible microwave models (option B or C). Still open: which terms, in which order, and whether native results are ever results of record.


## D-SPYC-19 Dependency on the `cochem-base` package

**Needed before:** SRS §4.4 and Phase 2. **Status:** ANSWERED in part 2026-10-17; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-19.md`

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** `cochem-base` is a pinned dependency, used for `load_module_handoff`, the module registry and shared constants. The vendored copy in `src/cochem_base/` is removed. | One implementation; SpycFit follows BASE releases. |
| B. SpycFit carries copies of the handoff schema and constants, with tests against BASE's versions. | Independent install; duplication guarded by tests. |

**PI answer** (conversation 2026-10-17; dictated, quoted as received). The agent asked: "**D-SPYC-19** declare `cochem-base` as a dependency (it is vendored today)". The PI replied: "13. CoChem-BASE, TOPOS, and TORQ are all dependencies. However, there should be a way to input the predicted um Mhm. That settles that. Welcome to the no-chat or the predicted um orca files or SP um Cat SP fit files and have them read and act as a valid input rather than having to go through topo and torque. CoChem-BASE though is a hard requirement cuz that has the ingestion and the unified GUI." Record: `docs/decisions/D-SPYC-19.md`. Still open: whether TOPOS and TORQ are install-time requirements or optional; which ORCA outputs the direct-input path reads, and whose ORCA parser it uses.

## D-SPYC-20 Domain defaults

**Needed before:** SRS §2. **Status:** ANSWERED in part 2026-10-17; DECIDED when the PI approves the PR containing `docs/decisions/D-SPYC-20.md`

Recommended defaults, all user-overridable and recorded with every catalog and fit:
- T_rot = 1.5 K (allowed 0.5-20 K).
- Frequency window 2-22 GHz (extensible to 26 GHz).
- J_max = the smallest J whose population at T_rot falls below 1×10⁻⁴ of the maximum, capped
  at 30.
- Line width from the instrument; the PI to supply typical CP-FTMW values.

**PI answer** (conversation 2026-10-17). The agent asked: "**D-SPYC-20** default temperature, frequency window and J_max, now needed per technique (the old defaults were CP-FTMW only)". The PI replied: "10. These should be set at the beginning before the predictions start happening, and then they should be boxes that can be modified and changed by the user as need be." Record: `docs/decisions/D-SPYC-20.md`. Still open: whether the fields start empty or pre-filled with per-technique presets, and their values.
