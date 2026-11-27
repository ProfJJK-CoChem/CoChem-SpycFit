# CoChem-SpycFit: Software Requirements Specification

Revision: October 16, 2026 (America/Chicago)

Status: Consolidated requirements baseline for implementation, incorporating the user's confirmed product and policy decisions. Benchmark-specific records, supported executable versions, and quantitative acceptance criteria remain work items. Scientific capability and acceptance claims require the evidence defined below. Publication of this SRS does not certify the current implementation.

The October 7 published SRS at commit 7e42507e4668751da393241294ffbb9dd4f0c259 is preserved unchanged in the [published baseline archive](../archive/CoChem-SpycFit_Comprehensive_SRS_7e42507.md). This revision replaces its duplicated chapters, conflicting requirements, and authoring transcripts. The [October 16 amendment and evidence review](CoChem-SpycFit_SRS_2026-10-16_Amendment.md) supplies the confirmed decision register, complete SF-CMP-001–015 comparison contract, and source-backed implementation findings. Its confirmed requirements are part of this SRS; historical findings are informative rather than evidence of a new executed validation.

## 1. Purpose, users, and success

SpycFit is an expert-directed computational chemistry-to-spectral-assignment and fitting pipeline. It shall help researchers inspect measured spectra, interpret molecular predictions, compare assignment hypotheses, fit physical models, quantify justified uncertainty, and produce reproducible publication artifacts.

Improvements over existing workflows shall be measured through correctly supported assignments, expert effort, predictive accuracy, calibrated uncertainty, runtime, and traceability. A blanket superiority claim over SPCAT/SPFIT is insufficient. Comparisons shall identify the physical capability, data, objective, hardware, and tolerances.

Required research demonstrations:

1. Broadband rotational assignment and fitting, including isotopologues and uncertainty.
2. Joint rotational and resolved rovibrational assignment/fitting with explicit shared parameters.
3. Computation-led assignment for flexible molecules using qualified computational and MLFF providers.
4. Vibrational band/line fitting, mode interpretation, and spectral-envelope decomposition.

Gas-phase spectroscopy is the primary validation track. Solution and solid-state analyses shall be separately modeled and validated.

## 2. Scope and capability profiles

IR, Raman, THz, near-IR, and microwave are first-class workflows. Each shall expose a familiar import, inspect, propose, assign, fit, compare, and publish sequence. First-class status requires supported models, uncertainty handling, validation examples, and exports rather than merely a viewer.

SF-MODEL-001: Distinguish spectral region, measurement technique, measured observable, conditions, and physical transition model. Raman shift and absolute frequency are different quantities; absorbance, transmittance, integrated intensity, and spectral density require distinct semantics.

SF-MODEL-002: Every solver/backend shall declare supported species/state classes, Hamiltonian or force-field assumptions, operator conventions, truncation/basis limits, observables, precision, and applicability limits. Unsupported combinations shall produce diagnostics rather than an unlabelled approximation.

| Profile | Required role | Validation boundary |
|---|---|---|
| Semirigid rotational | Linear, symmetric-top, and asymmetric-top cases under declared effective models; supported distortion terms | First independent SPCAT/SPFIT acceptance profile; matched conventions and fixtures |
| Extended rotational | Isotopologues, excited vibrational states, resolved hyperfine, and supported state interactions | Separate capability and benchmark records |
| Resolved rovibrational | Band origins, state-dependent constants, assignments, and supported couplings | Transition-level references and state/interaction-specific checks |
| Vibrational bands/profiles | Positions, widths, intensities/areas, baselines, overlap, and mode interpretation | Identifiability and profile/peak evidence; envelope agreement does not certify unique assignments |
| Large-amplitude motion | Internal rotation, inversion, tunnelling, and flexible conformational behavior | Dedicated model/provider contracts and independent references |
| Computational/PES refinement | Validated predictions, corrections, and inverse PES/force-field refinement as a required workflow capability, potentially implemented in another CoChem module | Training/validation separation, domain checks, and observable-level accuracy |

Hyperfine, excited-state, and large-amplitude-motion extensions follow semirigid comparison acceptance under separately validated profiles; all remain in scope.

## 3. Architecture and ownership

| Capability ID | Responsibility | Consumes |
|---|---|---|
| scientific-records | Identities, states, observations, units, provenance, uncertainty, result revisions | Experimental and computational records |
| computational-evidence | Quantum-chemistry import, conformers, tensors/modes, PES/MLFF contracts | Scientific records and provider artifacts |
| spectroscopic-models | Hamiltonians, states, transitions, intrinsic intensities/populations | Scientific records; optional computational evidence |
| signal-models | Instrument response, broadening, baselines, noise, overlapping features | Scientific records; optional predicted transitions |
| assignment-hypotheses | Candidate relationships, alternatives, blends, expert decisions | Observations and model predictions |
| fitting-inference | Experiment-only/informed fits, shared parameters, covariance/posteriors, diagnostics | Versioned assignments, models, and data |
| validation-comparison | Independent engines, held-out evidence, uncertainty coverage, numerical checks | Results and reference records |
| research-outputs | Spectra, tables, constants, figures, reproducible packages | Traceable results, decisions, validation |

These dependencies describe data ownership. A workflow coordinator passes immutable revisions through iterative proposal, fit, comparison, and expert-review cycles. Stateless workers are allowed when their inputs/outputs remain linked to durable provenance and project history.

CoChem orchestration may schedule jobs, manage artifacts, and provide upstream theory. SpycFit owns spectroscopic observation semantics, assignment interaction, fitting interpretation, comparison, and reporting. Raw instrument reduction and PES/MLFF training may be supplied by adapters or separate CoChem modules.

## 4. Ingestion, interoperability, and reference data

SF-IO-001: The core accepts calibrated spectra and peak lists. Raw microwave FIDs, IR interferograms, THz time traces, and other instrument-specific raw formats connect through adapters or other CoChem modules. Both routes shall produce the same observation contract.

SF-IO-002: Preserve stable record identity, source, observable, units, available calibration/processing history, conditions, and uncertainty. Missing metadata remains explicit. Missing uncertainty may limit precision fitting while still permitting inspection; it shall not cause an invisible drop or be invented as zero.

SF-IO-003: Preserve original measurements and source files. Conversion, normalization, baseline correction, masking, subtraction, and resampling create traceable derived records. Distinguish full profiles, extracted peaks, assigned lines, compiled constants, and computed quantities.

SF-IO-004: Supported Pickett formats require versioned adapters with documented coverage and round-trip evidence. Preserve field meanings, state labels, units, uncertainty/weight conventions, and constraints. Unsupported fields/records shall be reported rather than silently discarded.

SF-IO-005: Atomic/isotopic masses shall have explicit isotope identity, units, and a versioned, traceable source. Offline deployments may use verified snapshots without runtime network queries.

SF-IO-006: Build a reference corpus from suitable NIST WebBook, CCCBDB, other curated databases, and primary publications. Preserve record identifiers, attribution, retrieval/version information, reuse terms, conditions/resolution where available, and experimental/computational status.

SF-IO-007: Deduplicate related evidence. A spectrum, its extracted peaks, and constants fitted to those peaks are not independent validations. Benchmark partitions shall prevent relevant molecule/isotopologue/conformer/source leakage and distinguish interpolation from extrapolation.

The candidate molecular panel in the amendment is proposed, not a verified downloaded corpus.

## 5. Molecular and spectroscopic calculations

SF-PHYS-001: Imported geometries, Hessians, dipole derivatives, modes, constants, and other tensors shall retain atom mapping, frames, units, method/version, and provenance. Check shape, symmetry, and relevant physical preconditions.

SF-PHYS-002: Rotational predictions shall identify reduction/representation, operators, sign/unit conventions, state labels, basis/truncation, and numerical convergence. Absorption frequency uses upper-minus-lower energy divided by Planck's constant under declared units.

SF-PHYS-003: Equilibrium, vibrational-state-averaged, and fitted effective constants remain distinct. Vibrational corrections, including VPT2 where applicable, shall identify assumptions and resonance/applicability limits. A universal percentage difference between unlike constants shall not automatically reject a conformer.

SF-PHYS-004: Axis alignment, atom mapping, isotope substitution, and frame transformations shall preserve documented handedness, permutation, and tensor conventions. Address degeneracies explicitly. Isotopic reweighting within a shared electronic model shall state its approximations.

SF-PHYS-005: Internal-axis/principal-axis and perturbative/variational large-amplitude approaches require declared capability, references, convergence, and observable tests before support is claimed. Method names in the archived baseline alone do not establish support.

SF-PHYS-006: Vibrational assignments shall allow combinations, overtones, resonance mixing, ambiguous mode correspondence, and multiple contributors to a feature. A universal one-to-one band/harmonic-mode constraint is not permitted.

SF-PHYS-007: IR absorption and Raman scattering require appropriate intensity models. Represent temperature, populations, partition functions, selection rules, and instrument response where required. Missing intensity calibration limits intensity-based claims.

SF-PHYS-008: Convolution/profile models shall declare normalization, line-shape parameters, truncation/windowing, and instrumental assumptions. Multitemperature/mixed-species predictions shall declare populations. Validate chunked results against a reference computation.

## 6. Fitting, uncertainty, and computational assistance

SF-FIT-001: Bind each fit to an immutable observation revision, expert assignment configuration, model, free/fixed parameters, objective/likelihood, uncertainties/weights, starting values, priors/constraints, optimizer settings, and convergence diagnostics.

SF-FIT-002: Support experiment-only and computation-informed results side by side with prior/constraint sensitivity checks. Computation-derived initialization alone shall not be described as experimental evidence.

SF-FIT-003: Joint fitting shall identify shared physical parameters and preserve data-specific signal models, calibration, and dependencies. Do not double-count related observations or force incompatible phases/states into one model.

SF-FIT-004: Report rank deficiency, correlation, conditioning, and identifiability limits. When defined, normalized correlation is covariance divided by the product of parameter standard uncertainties. Off-diagonal covariance approaching one is not a general correlation criterion.

SF-FIT-005: Sensitivity measures shall identify observable, parameter domain/distribution, estimator, and limitations. Proposed freezing, constraints, exclusions, or priors require an inspectable expert-approved configuration. A universal Sobol cutoff shall not automatically impose a theoretical value.

SF-FIT-006: Distinguish measurement uncertainty, conditional parameter uncertainty, assignment ambiguity, model discrepancy, computational prediction error, and numerical error. State interval conventions/assumptions and propagate available covariance/posterior information. Missing information remains unknown.

SF-FIT-007: Convergence, low RMS, narrow conditional errors, agreement between programs, or plausible reconstructed spectra alone shall not establish correct assignments or accurate predictive uncertainty.

SF-FIT-008: Validate derivative methods for the declared model, including relevant degeneracies and state tracking. Floating-point execution does not imply mathematical exactness. Acceleration claims require measured workloads and precision/accuracy checks.

SF-ML-001: ML may rank candidates, propose corrections/parameters, identify suspicious observations, or recommend informative measurements. Distinguish heuristic scores from calibrated probabilities. Scientific recommendations require domain checks, provenance, limitations, and abstention.

SF-ML-002: MLFF/PES providers shall declare molecular identity, atom mapping, units, supported derivatives, training/reference provenance, domain, and model version. Energy/force error alone does not establish accuracy of spectral frequencies, constants, resonances, or splittings.

SF-ML-003: PES/force-field inverse fitting shall preserve model versions, objectives, constraints, and training/evaluation partitions. Refinement may use another CoChem provider. Poor fit alone shall not diagnose a faulty PES or authorize unbounded theory escalation.

## 7. Expert-directed interaction

SF-UI-001: Explicit expert action determines assignment acceptance. CoChem proposes, ranks, cycles, explains, and recomputes without silently accepting assignments or replacing accepted scientific decisions.

SF-UI-002: Support acceptance, rejection, tentative hypotheses, unresolved blends, dragging/reassignment, and undo consistently across the five workflows. Provide a non-drag path. Dragging an assignment changes a relationship, not the measurement.

SF-UI-003: Show overlays, residuals, alternatives, physical labels, uncertainty meaning, and supporting evidence. Preserve the distinction between an expert-accepted and uniquely demonstrated interpretation.

SF-UI-004: Parameter manipulation may use an explicitly approximate preview; confirmed results require recalculation at the required fidelity. Show pending/stale results and affected assignments, parameters, uncertainties, and outputs.

SF-UI-005: Asynchronous calculations remain attached to their originating revision and cannot overwrite newer decisions. Rejections, masks, and subtraction actions shall remain traceable and reversible.

The baseline Voila/widget frontend is an implementation candidate. The common interaction and scientific contracts shall remain independent of toolkit choice.

## 8. SPCAT/SPFIT comparison and divergence review

The complete SF-CMP-001–015 contract in the [amendment](CoChem-SpycFit_SRS_2026-10-16_Amendment.md#3-required-independent-spcatspfit-comparison-method) is normative.

Required method:

1. Run genuine SPCAT and native SpycFit predictions with the same supported model/constants.
2. Run genuine SPFIT and native SpycFit experiment-only fits with matched observations, assignments, and settings.
3. Evaluate each fitted parameter set through both compatible predictors to separate fitting from forward-model differences.
4. Display transition, assignment-evidence, residual, constant, uncertainty, and coverage comparisons.
5. Preserve actual engine inputs/outputs, executable identity/version, run evidence, mappings, and thresholds.

SPCAT predicts catalogs; SPFIT fits supplied assignments. Their predictions/fits support review of SpycFit hypotheses and expert decisions; the interface shall not invent an autonomous assignment output from them.

Distinguish not run, unavailable, failed, incompatible model, incomplete, divergent, and within tolerance. Empty comparisons, missing statistics, partial coverage, substitute calculations, or simulated executables cannot certify independent-engine agreement.

Exploration continues with visible divergence warnings. Publication export requires expert review and preserves unresolved disagreement and review provenance. Review does not convert divergence into agreement.

Semirigid acceptance comes first. Hyperfine, excited-state, and large-amplitude-motion comparisons follow as separately qualified profiles. Models unsupported by SPCAT/SPFIT require another qualified reference or experimental validation rather than forced parity. The evidence used shall be named explicitly; an implementation comparison alone does not establish experimental accuracy.

## 9. Execution, hardware, deployment, and state

SF-RUN-001: Provide an offline-capable fitting core with optional connected import/adapters. Physical air-gap is an optional profile. Reference acquisition and executable/dependency provisioning shall use supported, traceable routes.

SF-RUN-002: Provide a validated CPU reference path and optional accelerators. Record precision, backend, versions, hardware, seeds where relevant, and numerical tolerances. Backend changes preserve the model and are reported; substitute calculations cannot carry the requested engine's label.

SF-RUN-003: Resource limits/scheduling shall be configurable. Chunking/cache keys shall account for model/data/state, code/compiler, precision, and hardware assumptions. Performance/memory targets require declared workloads rather than an unexplained universal VRAM ceiling.

SF-RUN-004: Failures, cancellation, timeouts, and unsupported capabilities shall be visible and diagnosable. Preserve input/run records. Isolated work directories shall not discard provenance.

SF-RUN-005: Refinement requests shall carry diagnostic evidence, alternatives, intended observables, model revision, requested calculations, and user-approved resource policy. Return a new version with provenance/validation rather than overwriting accepted results.

SF-RUN-006: Integrity checks require genuine verified artifacts/manifests, not placeholder hashes or disabled verification. Locally available compilation/JIT is compatible with offline execution.

## 10. Publication and reproducibility

SF-OUT-001: Export lines, constants, uncertainties, diagnostics, spectra, figures, and tables from coherent result records. Central values and uncertainty from different engines/fits shall remain separate.

SF-OUT-002: Unit labels require actual numerical conversion. Unknown uncertainty remains unknown. Display precision shall be justified while preserving sufficient internal values for reproducibility.

SF-OUT-003: Publication packages identify sources, molecular/state identities, conditions, models/conventions, assignments, expert decisions, experiment-only/informed distinction, priors, uncertainty interpretation, engine/run versions, and validation status.

SF-OUT-004: Preserve relevant unresolved assignments, blends, model limitations, and engine divergence. Expert divergence review is required under section 8.

## 11. Validation and release evidence

Maintain separate evidence classes for parser/process/interface behavior; independently understood numerical correctness; statistical recovery/uncertainty calibration; genuine external-engine comparisons; and experimental assignment/prediction/publication demonstrations.

An analytical check, process simulator, or alternative physical calculation cannot silently replace a failed real-engine or experimental test. Failed assertions remain failures. Acceptance requires nonempty coverage; failed, incomplete, unrun, and unavailable checks remain distinct.

The first versioned panel shall declare semirigid cases, isotopologues/states, transitions, model conventions, objectives, binary versions, and source provenance. Additional profiles cover the broader required workflows.

Before acceptance runs, define justified numerical tolerances, experimental/predictive error criteria, coverage, uncertainty-calibration metrics, and hardware/workload definitions. One arbitrary frequency threshold cannot serve every parameter, state, observable, or experimental regime.

Improvement claims shall compare the documented research workflow, including expert intervention and failed/abstained assignments. Preserve holdouts and identify extrapolation. Agreement between implementations sharing data/assumptions is not independent experimental truth.

## 12. Traceability and remaining work

The amendment's SF-SCOPE, SF-ING, SF-HITL, SF-INF, SF-DATA, SF-PUB, and SF-CMP requirements retain their identifiers and are incorporated here. This document adds SF-MODEL, SF-IO, SF-PHYS, SF-FIT, SF-ML, SF-UI, SF-RUN, and SF-OUT requirements.

| Published topic retained | Consolidated location |
|---|---|
| Ecosystem/TORQ boundary and persistent registry | Sections 3 and 9 |
| Hardware, precision, caching, chunking, CPU path | Sections 5, 6, 9 and 11 |
| Ingestion, Pickett formats, mass data, preflight | Section 4 and comparison contract |
| Quantum tensors, Hamiltonians, vibrational corrections | Sections 2 and 5 |
| Internal rotation and large-amplitude motion | Sections 2, 5 and provider contract |
| Eckart/frame alignment and isotopic substitution | Section 5 |
| Inverse fitting, derivatives, sensitivity | Section 6 |
| Multitemperature convolution | Sections 3 and 5 |
| Expert interaction and comparison displays | Sections 7 and 8 |
| Upstream PES feedback | Sections 6 and 9 |

Three policy questions are resolved: expert-reviewed divergence export, offline-capable operation with optional connected adapters, and semirigid-first independent comparison. No new high-level product preference is required to continue.

Remaining scientific/engineering specification work is to select verified reference records and executable versions, populate the capability/acceptance matrix with justified numerical criteria, and validate method definitions/citations before profiles are implemented or advertised. Archived authoring suggestions are not requirements merely because they appeared in the publication.

Current implementation defects in the amendment require separate correction and real-engine validation. Agreement on this SRS does not claim those defects are fixed, the new feature is implemented, or release acceptance has passed.
