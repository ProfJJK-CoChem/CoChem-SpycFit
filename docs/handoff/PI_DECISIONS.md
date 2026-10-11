# PI decision sheet for CoChem-SpycFit

Status of every entry: **OPEN** until the PI answers. An agent may add options or evidence but
must not change an entry to DECIDED. When the PI answers, record the answer here and in
`docs/decisions/D-SPYC-nn.md` (date, the PI's words, consequences, and any follow-up work).

Decisions marked **blocking** must be answered before SRS v2 functional requirements are
written. The others can be answered as the relevant chapter is drafted.

Sources: "Review" = `docs/wiki/review/CoChem-SpycFit_SRS_Review.md`; "MM" =
`CoChem-BASE/Method_Matrix.md` v4; "TORQ contract" =
`CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`; "TORQ SRS" =
`CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`.

---

## D-SPYC-01 Product identity and fitter of record (blocking)

**Question.** What is SpycFit, and which program's least-squares fit is the published result?

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Rotational assignment and fitting workbench for the MM §2.1 system class (5-10 atom weakly bound complexes, CP-FTMW, about 2-22 GHz, T_rot 1-2 K, plus monomers and isotopologues). A pinned SPFIT/SPCAT is the fit of record and export path. SpycFit owns ingestion, prediction windows, the human assignment loop, diagnostics, campaigns and publication export. Any in-process Hamiltonian model serves prediction, assignment and what-if views and must match SPCAT before its output is shown. | Matches MM (JAX "not used to replace SPFIT") and the TORQ SRS ownership split. Smallest scientific risk. Draft scope statement: Review §8.6. |
| B. Native fitter of record (own Hamiltonian and least squares), with SPFIT/SPCAT used only to cross-check. | Larger build. Every Hamiltonian term (hyperfine, multi-state, internal rotation) must be implemented and proved equal to SPFIT before release. |
| C. Thin front end over PGOPHER (as the BASE User Manual Ch. 5 says). | PGOPHER is a Windows GUI program; automation in Codespaces/CI is harder. Contradicts MM's SPFIT-centred deliverable (§18: a `.par` + `.int` that runs unmodified through SPCAT). |

Also confirm: **no IR, Raman, UV-Vis, NMR, EPR or HITRAN/ExoMol functions in SpycFit** (Review
§8.6 "is not" list), and that README and User Manual Ch. 5 will be rewritten to match.

**PI answer:** _open_

## D-SPYC-02 Mass and constants convention

Recommended: isotope-specific **atomic** masses from a pinned, bundled AME2020 table with
edition and hash recorded (same as TORQ-SCI-002); CODATA 2018 h/(8π²) = 505379.0 MHz·u·Å²; no
nuclear-mass stripping (old SRS "PCA-32" is withdrawn). Option: share a constants module with
BASE/TORQ instead of carrying a SpycFit copy.

**PI answer:** _open_

## D-SPYC-03 Feedback to TORQ

When a fit indicates the predictions were wrong (wrong conformer, poor barrier), what does
SpycFit emit?

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** A versioned "refinement request" file in SpycFit's output bundle, handed to BASE through a module handoff manifest. A human approves any TORQ calculation that follows. | Respects TORQ's single task authority (TORQ-ARC-002, D08). No daemon. |
| B. No feedback path in v2. | Simplest; feedback stays manual. |
| C. Direct submission into TORQ or BASE task queues. | Conflicts with D08 and TORQ-ARC-002. Not recommended. |

**PI answer:** _open_

## D-SPYC-04 Internal rotation and tunnelling

| Option | Consequence |
| :--- | :--- |
| A. Delegate internal rotation to XIAM / BELGI / RAM36 (or SPFIT's own options) through wrappers; SpycFit ingests their results. | Least code; depends on external programs being installable in CI. |
| B. Native effective torsion-rotation Hamiltonian (RAM or PAM) with XIAM-compatible parameters, validated against XIAM. | More code; one in-process model for assignment. |
| C. Defer internal rotation to a later release; v2 covers semi-rigid species and two-state tunnelling doublets only. | Matches TORQ D05 ordering (semi-rigid first, LAM later). |

Recommendation: C for the first release, then A or B. The PI's own NH3-HCOOH work needs A/E
fitting, so the PI should weigh that.

**PI answer:** _open_

## D-SPYC-05 User interface and hosting

Recommended: one stack, ipywidgets + Plotly under Voila, hosted as a BASE tab (and runnable as a
standalone notebook), every action calling the same headless Python API, plus a CLI for batch
fits. Delete the Qt, Streamlit and plugin variants. Alternative: CLI and library only in v2,
UI in a later release.

**PI answer:** _open_

## D-SPYC-06 IR / Raman profile

Recommended: none in SpycFit. The README and User Manual describe IR/Raman spectrum generation.
If that capability is still wanted, it moves to TORQ reporting or a separate module (TORQ D12:
adjacent domains need explicit sibling handoffs).

**PI answer:** _open_

## D-SPYC-07 Eigenvalue label swaps during a fit

When a parameter change reorders levels so that a transition's (J, Ka, Kc) label maps to a
different eigenvalue, should SpycFit halt the fit or warn and continue? Recommended: halt and
show the swap, because silent relabelling corrupts residuals.

**PI answer:** _open_

## D-SPYC-08 Anti-gaming checks

Review §9.4 proposes keeping narrow hygiene checks (no `except: pass`, no tautological asserts,
no empty function bodies outside abstract classes), dropping keyword bans and the amnesty file,
and gating on literature-value acceptance tests, the SPFIT/SPCAT round trip, whole-suite CI and
PI review. Approve, amend or reject.

**PI answer:** _open_

## D-SPYC-09 Parallel-execution imports

The current linter bans `multiprocessing`, `concurrent.futures`, `dask` and `parsl`. The Method
Matrix uses Parsl elsewhere. Recommended: lift the ban; SpycFit's own workloads are small and
need no parallel framework, so no new dependency follows.

**PI answer:** _open_

## D-SPYC-10 Structure determination in v2

Kraitchman r_s, r_0 and r_m fits. Recommended: in scope for v2 but gated by a fluxionality
check (no structure output for species with resolved tunnelling or several populated basins,
MM §2.3). Alternative: defer to a later release or delegate to Kisiel's STRFIT/KRA.

**PI answer:** _open_

## D-SPYC-11 Toolchain baseline (blocking for code, not for the SRS)

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Python 3.12; NumPy/SciPy for the Hamiltonian and fit support (matrices are at most a few hundred square; Hellmann-Feynman derivatives need no autodiff); no JAX, no CuPy; CPU only. | Removes the JAX version trap (`jax==0.4.13` lacks `wofz`), installs everywhere, runs in Codespaces and CI. |
| B. Python 3.12 with JAX ≥ 0.11 (FP64 enforced at one entry point) for the forward model. | Keeps the JAX direction of the ecosystem; adds version and precision management. |

**PI answer:** _open_

## D-SPYC-12 Inertial-defect sign rule

The raw Method Matrix §6.4 makes a sign disagreement between computed and experimental inertial
defects a **binding rejection**. The accepted TORQ contract §3.1 says "a sign discrepancy is a
diagnostic, not sufficient evidence to reject a conformer", because ground-state defects carry
vibrational contributions. Recommended: follow the TORQ contract, so SpycFit always computes
and shows Δ and the planar moments, flags a sign disagreement prominently, and never rejects
automatically. The review's SPYC-DIAG-002 text must be changed either way (see the errata).

**PI answer:** _open_

## D-SPYC-13 Dark dipole components

MM §6.1 flags |μ_g| < 0.1 D as "dark". The TORQ contract §3.4 says "a 0.1 D display/detection
threshold cannot delete nonzero archival transitions". Recommended: dark branches are
de-prioritised in search windows and labelled, but remain in every archival catalog; the
threshold is a user setting.

**PI answer:** _open_

## D-SPYC-14 Benchmark set and data supply (blocking for G1)

Which systems gate the release, with which published data and tolerances? Review §8.4 proposes
13 entries anchored on MM §17. The PI chooses the set, supplies or points to the line lists, and
approves the tolerances before implementation (TORQ D10: preregistered, observable-specific
targets). Note that MM §3.1 accuracy bands are uncalibrated literature statistics (TORQ contract
§3.2) and must not be used as acceptance tolerances.

**PI answer:** _open_

## D-SPYC-15 Repository strategy (blocking for Phase 2)

| Option | Consequence |
| :--- | :--- |
| **A (recommended).** Tag the current state `legacy-swarm-2026-10`, then build the new package in `src/cochem_spycfit/` from a clean start in this repository, porting legacy modules one at a time only after they pass requirement-linked tests. | History preserved; no ambiguity about which code is current. |
| B. Clean up in place: keep the `src/` tree, delete the rest, fix module by module. | Less churn, but the agent inherits unproven code and the old tests' assumptions. |
| C. New repository; archive this one. | Cleanest, but breaks existing links and BASE's registry entry. |

**PI answer:** _open_
