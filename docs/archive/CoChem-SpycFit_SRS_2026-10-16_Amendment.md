# CoChem-SpycFit SRS amendment: expert-directed spectroscopy and independent engine comparison

Date: October 16, 2026 (America/Chicago)

Status: Companion to the consolidated October 16 SRS, containing user-confirmed requirements and policy decisions, proposed acceptance criteria, and a source-evidence review. The three policy questions in section 7 were answered by the user on October 16, 2026. This document specifies required behavior; it does not claim that the behavior is implemented or scientifically validated.

## 1. Baseline and authority

The reviewed GitHub main commit is 7e42507e4668751da393241294ffbb9dd4f0c259, published October 7, 2026. Its Comprehensive SRS is the prior published baseline, preserved unchanged in [the archive](../archive/CoChem-SpycFit_Comprehensive_SRS_7e42507.md). The [consolidated SRS](CoChem-SpycFit_Comprehensive_SRS.md) and the user decisions recorded here supersede conflicting baseline requirements. Section 7 records the resolved policy decisions. Baseline line references in this review refer to the archived publication or original Git commit.

The review used the current source and tracked audit artifacts. No new scientific test suite or live SPCAT/SPFIT experiment was executed for this documentation review. The previous onboarding report recorded 22 passed and 2 failed; its checkout and results were not revalidated in this review and must not be reported as the current baseline.

No dedicated current handoff file was found among tracked filenames. Historical audit claims are distinguished from current source evidence in section 6. Git access succeeded; GitHub issue/PR API access was denied, so this review does not claim to include issue or PR discussions.

## 2. Confirmed product and scientific requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| SF-SCOPE-001 | Support rotational assignment/fitting, joint rotational–rovibrational fitting, computation-led/MLFF-assisted assignment, and vibrational spectrum fitting as required research workflows. | A separate, named demonstration and validation record for each workflow. |
| SF-SCOPE-002 | IR, Raman, THz, near-IR, and microwave shall be first-class workflows with familiar inspection, assignment, fitting, comparison, and publication interactions. | A capability matrix and end-to-end example for every workflow; unsupported interactions are explicit. |
| SF-SCOPE-003 | Model spectral region, measurement technique, measured observable, physical transition model, and experimental conditions separately. | Correct handling of absolute frequency/wavenumber, wavelength, Raman shift, and applicable intensity definitions without silent reinterpretation. |
| SF-SCOPE-004 | Prioritize vibrational band/line fitting, mode assignment, and envelope decomposition: positions, widths, intensities/areas, baseline, and overlap. | Validate identifiable quantities; retain ambiguity for unresolved components and mixed, overtone, combination, or resonant assignments. |
| SF-SCOPE-005 | Support PES/force-field refinement as a required workflow capability through an explicit provider boundary; training/refinement may belong to another CoChem module. | Versioned model, atom mapping, units, domain, training provenance, derivatives/capabilities, and independent validation accompany exchanged artifacts. |
| SF-SCOPE-006 | Use gas-phase benchmarks as the primary track, with solution and solid-state tracks reported separately. | Phase and conditions are retained; cross-phase performance claims require their own evidence. |
| SF-ING-001 | The core accepts calibrated spectra and peak lists. Raw FID, interferogram, and time-trace processing connects through adapters or other CoChem modules. | Direct and adapter imports obey the same observation contract and provenance checks. |
| SF-ING-002 | Preserve source observations, units, calibration information, available conditions, uncertainty, processing history, and missing-data flags. | Derived conversions and corrections are versioned; missing uncertainty is never converted into zero uncertainty. Frequency calibration does not imply absolute intensity calibration. |
| SF-HITL-001 | CoChem proposes, ranks, explains, and cycles assignment candidates; explicit expert action determines acceptance. Autonomous scientific acceptance is not the endpoint. | Accepted records identify the expert action and exact assignment revision. Exploratory calculations cannot silently accept, reject, or replace scientific assignments. |
| SF-HITL-002 | Support accepting, rejecting, dragging/reassigning, undoing, and retaining tentative or competing hypotheses, including blends. | Dragging changes an assignment relationship without moving the underlying measurement. Expert rejections persist unless new evidence is explicitly surfaced. |
| SF-HITL-003 | Reassignment shall show affected residuals, predictions, parameters, uncertainties, and outputs; asynchronous results belong to the revision that generated them. | An older calculation cannot overwrite a newer decision; pending/stale derived results are visibly identified. |
| SF-INF-001 | Report experiment-only and computation-informed fits side by side, with prior/constraint sensitivity checks. | Each result carries its own data, assumptions, parameter values, covariance or posterior description, engine, units, and model revision. |
| SF-INF-002 | Distinguish measurement uncertainty, conditional parameter uncertainty, assignment ambiguity, model discrepancy, computational prediction error, and numerical error. | Output precision or small fit residuals are not substituted for evidence of accuracy, identifiability, or uncertainty calibration. |
| SF-DATA-001 | Build a versioned, broad reference corpus using suitable NIST WebBook, CCCBDB, other curated sources, and primary publications. | Every accepted record includes source identity/version, experimental/computational status, conditions, reuse terms, resolution and uncertainty where available. Shared-source spectra, peaks, and fitted constants are not counted as independent validations. |
| SF-PUB-001 | Publication artifacts shall be reproducible and traceable to observations, expert decisions, model, fit settings, uncertainty interpretation, and validation status. | Every exported number and plot can be traced to one coherent result record; uncertainty from a different engine or fit cannot be silently attached. |

The molecular panel proposed in the planning conversation remains a candidate corpus, not a downloaded or validated dataset. It includes CO, HCl, OCS, CO2, H2O, formaldehyde, SO2, acetonitrile, ammonia, methanol, acetaldehyde, dimethyl ether, ethanol, 1-propanol, and later NO/OH stress tests. Final inclusion requires usable reference records and a declared scientific purpose.

## 3. Required independent SPCAT/SPFIT comparison method

### 3.1 Purpose and program roles

SF-CMP-001: SpycFit shall run genuine SPCAT and SPFIT alongside its native spectroscopy path for supported microwave/rotational models, retain each program's results separately, and expose agreement and divergence to the expert.

SPCAT supplies predicted catalogs from a supplied model. SPFIT fits supplied line assignments and model parameters. This integration does not presume that either program independently discovers an experimental assignment. Assignment comparison means comparing the consistency, residuals, and competing interpretations supported by their predictions/fits against SpycFit hypotheses and expert decisions.

Agreement is evidence of consistency under matched assumptions, not independent proof of physical correctness. Experimental and held-out validation remain necessary.

### 3.2 Comparison stages

| Stage | Controlled inputs | Compared outputs | Interpretation |
|---|---|---|---|
| Forward prediction | Identical model, constants, conventions, states, and transition limits | SpycFit predictions versus SPCAT predictions | Isolates the forward-model implementation and serialization. |
| Experimental fitting | Same observation revision, assignment mapping, uncertainties/weights, free/fixed parameters, starting values, and supported objective | SpycFit experiment-only fit versus SPFIT fit | Detects differences in fitting, constraints, parameterization, and convergence. |
| Cross-prediction | Each fitted parameter set evaluated in both compatible prediction engines | A two-by-two prediction matrix | Helps distinguish forward-model disagreement from fitted-parameter disagreement. |
| Assignment review | Same observations with explicitly identified candidate/accepted mappings | Matched and unmatched transitions, alternatives, residuals, blends, rejected candidates | Presents evidence for expert review without fabricating an autonomous SPFIT/SPCAT assigner. |
| Informed-fit extension | Explicit computational priors/corrections and their provenance | Experiment-only versus informed predictions and fits | Attributes differences to additional information; unsupported objectives are not described as direct numerical parity. |

SF-CMP-002: Comparability shall be checked before numerical agreement is assessed. The record shall specify isotopologue, conformer, vibrational/electronic/torsional state, Hamiltonian reduction and representation, operator/sign conventions, truncation order, axis conventions, state/quantum-number mapping, units, hyperfine treatment, and free/fixed or constrained parameters as applicable.

SF-CMP-003: Intensity comparison additionally requires matched temperature, partition-function/degeneracy conventions, dipole or transition-moment definitions, intensity units, and population assumptions. Frequency agreement does not validate intensity agreement.

SF-CMP-004: Match transitions through explicit state identities and quantum-number conventions, with ambiguity recorded. Nearest-frequency matching alone shall not certify correspondence. Unresolved blends and mixed-state labels require a declared comparison rule.

SF-CMP-005: Incompatible models or objectives shall produce an explicit non-comparable result. Transformations between parameter conventions require a validated mapping and must preserve limitations from finite-order truncation. Where parameter comparison is invalid but observable comparison is meaningful, report that distinction.

### 3.3 Metrics and uncertainty

SF-CMP-006: Report per-transition signed and absolute frequency differences, maximum and RMS differences, observation-minus-calculation residuals for each engine, matched/unmatched counts, and fit/convergence diagnostics. Summaries shall retain links to individual divergent transitions.

SF-CMP-007: Report parameter differences only for physically comparable quantities, with native and normalized units, uncertainty interpretation, fixed/free status, covariance/correlation where supplied, and data/model provenance. Equilibrium constants and vibrationally averaged/effective constants shall remain distinct.

SF-CMP-008: Numerical implementation tolerances, experimental adequacy limits, and predictive-uncertainty criteria are separate. They shall be declared per capability/benchmark/observable with units, rationale, and version before acceptance evaluation. A small SPFIT observation residual alone is not cross-engine agreement.

SF-CMP-009: Do not assume fit results from the same observations are statistically independent. Any uncertainty-normalized comparison shall state its treatment of shared-data covariance or explicitly label its approximation. A missing uncertainty or covariance is unknown, not zero.

SF-CMP-010: Comparisons shall retain disagreement even when each fit separately converges. Different accepted models may yield different constants; the display shall distinguish expected model differences, unresolved mismatch, parsing/convention errors, numerical discrepancies, and failures to explain observations. Divergence does not by itself establish a wrong conformer or deficient PES.

### 3.4 Evidence, statuses, and expert display

SF-CMP-011: Retain executable identity/version, verified artifact digest and provenance, platform, exact input files, command/options, exit status, stdout/stderr, original output files, parser version, run identifier, and comparison configuration. Paths must use supported configuration, not private developer-machine assumptions.

SF-CMP-012: Comparison outcomes shall distinguish NOT_RUN, UNAVAILABLE, FAILED, INCOMPATIBLE_MODEL, INCOMPLETE, DIVERGENT, and WITHIN_TOLERANCE. An empty match set, missing statistics, empty parsed output, or unsupported requirement cannot yield WITHIN_TOLERANCE. Report partial coverage explicitly; it cannot satisfy full acceptance.

SF-CMP-013: Each compared result must actually originate from the named engine. An internal solver, simulator, analytic check, or alternative force field may provide separately labeled evidence, but cannot stand in for execution of SPCAT/SPFIT or convert its failure into a passed comparison.

SF-CMP-014: Provide shared spectral overlays, transition/assignment and constant comparison tables, difference/residual views, coverage, and an inspectable provenance record. Experts can navigate from a discrepancy to the observations, competing assignments, and model assumptions. Display rules must not hide disagreement by rounding.

SF-CMP-015: Applying a changed assignment, parameter constraint, or model selection requires an expert action. Comparison may automatically recompute and flag results without changing that scientific decision. Exploration continues under a visible divergence warning. Publication export requires explicit expert review of the discrepancy and records unresolved disagreement, the review action, and comparison evidence in the export package. Expert review does not relabel a divergent comparison as agreement.

### 3.5 Acceptance evidence

The acceptance suite shall include the following separately identified checks:

1. Actual SPCAT and SPFIT runs against versioned physical reference cases with confirmed nonempty transition/parameter coverage and supported models.
2. Controlled same-input predictions, experiment-only fits, and cross-predictions, with declared tolerances and held-out observations where appropriate.
3. Explicit convention/units/state mismatches that yield an informative incompatibility or divergence rather than silent reconciliation.
4. Empty or partial mappings, missing uncertainty, malformed output, unavailable executables, nonzero process exits, and failed assertions that remain visible failures or incomplete outcomes.
5. Assignment changes and blended features that preserve measurement identity and expert revision history.
6. Publication exports that keep each engine/fit's values, uncertainties, units, and conditions together; unknown uncertainties stay unknown.
7. CPU/GPU numerical agreement for supported native paths under declared tolerances, with hardware and precision recorded.

Transport/parser checks, analytical correctness checks, statistical calibration studies, real external-engine checks, and experimental validation shall retain separate evidence labels. None may be silently substituted for another.

The initial physical capability profile is matched semirigid rotational models, followed by separately validated extensions for hyperfine structure, excited states, and large-amplitude motion. Binary provisioning/version support and numeric tolerance values remain engineering/scientific validation tasks; no universal 0.1 kHz or 1000 kHz criterion is adopted by this amendment.

## 4. Scientific corrections to the published baseline

These corrections follow the confirmed scope and sound comparison semantics; they do not require choosing a preferred scientific outcome:

- Replace automatic parameter freezing solely from a universal Sobol threshold with reported sensitivity/identifiability diagnostics and an expert-approved fit configuration. Global sensitivity is not identical to local identifiability, and a particular prior range can change a sensitivity result.
- Define dimensionless correlation from covariance and parameter standard uncertainties where defined. Off-diagonal covariance values near one are not a valid generic correlation test.
- Diagnose unit/state/convention errors, measurement/calibration problems, assignment ambiguity, underidentified fits, incomplete Hamiltonians, and numerical problems before recommending new theory. Do not automatically reject a conformer from a universal effective-versus-equilibrium constant percentage.
- Treat differentiability, eigenvalue degeneracy/state tracking, numerical precision, and CPU/GPU performance as validation matters. Autodiff does not establish exact physical derivatives in all cases or a universal acceleration claim.
- Preserve lineage and persistent expert decisions. Ingestion failures must return inspectable diagnostics; a silent drop and zero retained source provenance cannot satisfy the confirmed research workflow.
- Remove the assumption that offline execution inherently forbids locally available JIT compilation. Network acquisition and local compilation are different operations.
- Consolidate duplicate chapters and conflicting formulas into a single reviewed scientific definition. The two Chapter 1 copies currently disagree on the energy-difference sign.
- Move authoring transcripts, unresolved council statements, and quota-driven suggestion lists out of normative requirements. Verify citations against the claims they support.
- Replace absolute compatibility, guaranteed determinism, unqualified precision, and fixed hardware performance claims with declared capability profiles and measurable acceptance criteria.

## 5. Current source findings that require implementation follow-up

References below identify the reviewed commit, not a new executed validation result. They are acceptance blockers for claiming the new comparison capability, rather than reasons to substitute another calculation.

| Finding | Evidence at baseline commit 7e42507 | Required correction |
|---|---|---|
| No common parameter names can return VERIFIED. | src/cochem_spycfit/physics_core/cochem_spycfit_auditor.py, lines 129–140 | Return incomplete/non-comparable; require meaningful coverage. |
| A parity flag is assigned from SPFIT residual RMS below 1000 kHz, without another engine. | src/cochem_spycfit/serialization/cochem_spycfit_spcat_wrapper.py, lines 541–558; missing parsed statistics default at line 460 | Separate fit-quality and actual engine-comparison outcomes. |
| Live SPFIT tests catch execution and assertion errors and substitute ASE/EMT water checks. | tests/test_cochem_spycfit_spcat_wrapper.py, lines 284–315 and 322–343 | Preserve actual SPFIT failures; separate physical/adapter tests from independent-engine validation. |
| A process test writes a Python simulator with a canned catalog. | tests/test_cochem_spcat_bridge_phase4.py, lines 376–407 | Label transport-only evidence; require additional genuine executable tests for scientific claims. |
| Export combines fit-state values with SPFIT uncertainties, defaults missing uncertainty to zero, and attaches distortion-unit labels without conversion at this output boundary. | src/cochem_spycfit/serialization/cochem_spycfit_export.py, lines 157–188 | Use coherent per-fit result records and validated unit conversion; preserve unknowns. |
| Default SPCAT/SPFIT artifact hashes are hashes of an empty byte string. | src/cochem_spycfit/core_engine/cochem_spycfit_init.py, lines 116–122 | Supply verified version/platform manifests through the supported provisioning path. |
| A Sobol total-effect calculation varies the complement of the indexed input under its current hybrid-matrix construction. | src/cochem_spycfit/convolver.py, lines 407–414 | Validate sensitivity estimators against independently understood cases before using them in parameter-selection advice. |

Production wrappers do contain real subprocess invocations for SPFIT and SPCAT (wrapper lines 628 and 715). The findings above concern verification meaning, correctness, and reporting; they do not imply that all production execution is emulated.

## 6. Prior-agent concerns and reliability of handoff evidence

- The published SRS includes an explicit warning at baseline line 939 that generating exactly twenty cited LAM improvement suggestions could require fabrication. Treat that section as unfinished authoring, not a completed scientific review.
- The August 27 parallel_council_payload.md claims 757 of 758 tests passed, with one skipped, and reports exception suppression, hardcoded masses, obfuscation patterns, and linter exemptions. Those counts and all listed defects have not been revalidated as a current suite result. HDF5 exception suppression is still visible in src/cochem_spycfit/intake/cochem_spycfit_orchestrator.py near line 1100. Inspected mass handling now uses DynamicMassDict; the historical blanket hardcoded-table claim must not be repeated as a current finding.
- cochem_refine_payload.md claims a Chunk 09 audit pass but links its supporting report to an unavailable Windows-local artifact. It does not establish current SpycFit end-to-end validation.
- ci_tools/Physics_Autopsy_Report.md is an unfilled template, not an actual diagnosed incident or closure record.
- The September 5 commit d62ba7b introduced the fallback testing behavior discussed above. Its governance wording does not establish scientific equivalence between SPFIT and ASE/EMT.
- Existing scientific concerns must become reproducible requirements, test cases, and issue evidence. Agent assertions, audit labels, exemptions, publication, or a clean process exit alone are not acceptance evidence.

## 7. User decisions resolved on October 16, 2026

| Decision | Question | Confirmed decision |
|---|---|---|
| D-01 | What happens when comparison remains divergent at publication export? | Continue with a visible warning; publication export requires expert review and records unresolved disagreement. |
| D-02 | Is physical air-gap mandatory for every deployment? | Offline-capable fitting core with optional connected import/adapters. A strict air-gap profile may receive versioned offline bundles. Physical air-gap is not a universal requirement. |
| D-03 | What physical profile must the first independent SPCAT/SPFIT acceptance suite cover? | Matched semirigid rotational models first, then separately validated extensions with explicit capability profiles. |

All broader modality and research-workflow requirements remain in scope regardless of D-03. A model unsupported by SPCAT/SPFIT needs another qualified reference or experimental validation; it must not be forced into an incompatible comparison.

## 8. Remaining specification work and release gates

No additional high-level user decision is presently required for the comparison architecture. Remaining specification work is:

1. Define the exact molecular/model/transition fixtures, supported executable versions, and versioned capability matrix.
2. Acquire suitable reference records and establish justified numerical tolerances, coverage, and uncertainty-calibration criteria per benchmark before reporting acceptance results.
3. Verify the chosen physical method definitions and citations for each implemented capability. Duplicate chapters and authoring transcripts have been removed from the consolidated normative SRS and retained in the unchanged archive.

Implementation and release validation are separate subsequent gates: verified executable provisioning, correction of the implementation/evidence defects in section 5, actual real-engine comparison runs, and demonstrated expert-review/export behavior. They are not prerequisites for agreeing on the SRS. This separation does not reopen the user decisions already confirmed in sections 2 and 7.
