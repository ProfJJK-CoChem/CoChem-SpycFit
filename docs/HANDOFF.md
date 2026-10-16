# SpycFit next-agent handoff

Date: October 16, 2026 (America/Chicago)

## Start here

1. Read the [consolidated SRS](wiki/CoChem-SpycFit_Comprehensive_SRS.md).
2. Read the [amendment, comparison contract, and evidence review](wiki/CoChem-SpycFit_SRS_2026-10-16_Amendment.md).
3. Consult the [unchanged October 7 publication](archive/CoChem-SpycFit_Comprehensive_SRS_7e42507.md) only for historical context. Conflicting archived statements and unverified agent suggestions are not current requirements.

The source review used GitHub main at 7e42507e4668751da393241294ffbb9dd4f0c259. This handoff accompanies the documentation publication requested by the user so another agent can continue. Application source, tests, dependency declarations, and scientific algorithms were not changed by this documentation work.

## User decisions already settled

- SpycFit is a computational chemistry-to-assignment/fitting/publication pipeline. It must support rotational assignment/fitting, joint rotational–rovibrational fitting, MLFF-assisted computational assignment, and vibrational band/line/envelope fitting.
- IR, Raman, THz, near-IR, and microwave are first-class workflows with familiar interactions. Backend model differences are acceptable and must remain scientifically explicit.
- The expert always determines assignment acceptance. CoChem proposes, ranks, cycles, explains, and recalculates. Intuitive dragging/reassignment, undo, ambiguity, blends, and revision history are required.
- The core accepts calibrated spectra and peak lists. Raw instrument reduction belongs in adapters or other CoChem modules.
- Gas phase anchors validation; solution and solid-state spectra have separate tracks.
- Experiment-only and computation-informed fits are reported side by side with sensitivity checks. Keep each result's values, uncertainty, model, data revision, and units together.
- PES/force-field refinement is required in the overall workflow and may be implemented in another CoChem module. Band/mode assignment and envelope fitting take priority within SpycFit.
- Run genuine SPCAT/SPFIT alongside native SpycFit for compatible microwave models. SPCAT predicts catalogs and SPFIT fits supplied assignments; neither is to be presented as an autonomous assignment engine.
- Divergence remains visible. Exploration continues, but publication export requires explicit expert review and records unresolved disagreement. Review does not relabel divergence as agreement.
- The fitting core is offline-capable, with optional connected reference import/adapters. Physical air-gap is not a universal installation requirement.
- Establish independent semirigid comparison first; then separately validate hyperfine, excited-state, and large-amplitude-motion extensions. All remain in scope.

Do not reopen these preferences without new evidence or a change requested by the user.

## What is complete

- Retrieved the latest main source and the October 7 comprehensive SRS using the platform's existing HTTPS Git access.
- Consolidated duplicate chapters and removed generated authoring transcripts from normative requirements while archiving the original byte for byte.
- Recorded 73 unique requirement IDs across the SRS and amendment, including 15 independent-engine comparison requirements.
- Reviewed the current source and historical audit artifacts; documented concrete evidence and distinguished it from unverified prior-agent assertions.
- Checked documentation formatting, local file links, requirement-ID uniqueness, and archive equality; obtained an independent documentation/source-evidence review.

No new scientific suite or live SPCAT/SPFIT comparison was run. No implementation defect was fixed. Historical onboarding results (22 passed, 2 failed) and old council certifications do not establish present readiness.

## Recommended next work

### 1. Establish the current development and test baseline

Use the existing isolated checkout; do not create a Git worktree unless requested. Preserve unrelated changes.

Inspect current package/import paths and dependencies before installation. This repository contains both root-level and src/cochem_spycfit trees; confirm which implementation each test imports. The earlier cloud setup and retained virtual environment predate the source reorganization and must not be assumed current.

The current pytest.ini defaults to a single provenance/security test file; a bare pytest command is not a complete spectroscopy test run. Name intended targets explicitly and record collection/execution counts. The existing GitHub workflow scans artifact placement; it is not evidence of numerical or scientific acceptance.

### 2. Fix misleading comparison and publication outcomes

Detailed paths and baseline line references are in amendment section 5:

1. Empty parameter overlap must not produce VERIFIED; incomplete coverage must remain explicit.
2. Separate SPFIT fit residuals from cross-engine parity. Missing parsed statistics cannot imply agreement.
3. Preserve actual SPFIT execution/assertion failures. ASE/EMT calculations cannot substitute for failed external-engine tests.
4. Separate simulated process/parser checks from genuine SPCAT/SPFIT scientific checks.
5. Keep publication values and uncertainties from the same fit/result. Preserve unknown uncertainty and perform actual unit conversion.
6. Replace placeholder binary digests with verified version/platform manifests.
7. Validate sensitivity estimators before using them to advise parameter selection; investigate the documented Sobol hybrid-matrix/total-effect mismatch.

Suggested first implementation slice: regression coverage and corrections for empty/incomplete comparison status and missing parsed statistics. Follow with coherent result records/export correctness before building stronger scientific comparison claims.

### 3. Qualify real comparison inputs and engines

- Select supported SPCAT/SPFIT builds and preserve verified acquisition/version information.
- Verify a small semirigid molecular/model panel with traceable observed assignments and physically matched conventions.
- Establish prediction, fitting, and cross-prediction fixtures, full match coverage, and justified numerical tolerances before acceptance runs.
- Keep engine identity, exact inputs, original outputs, exit status, parser version, model conventions, and comparison configuration.
- Maintain explicit not-run, unavailable, failed, incompatible, incomplete, divergent, and within-tolerance outcomes.

### 4. Acquire the broader reference corpus and extend capabilities

The proposed molecule panel is not a downloaded or validated corpus. NIST WebBook, CCCBDB, other curated catalogs, and primary publications still need record-level access, metadata, reuse, and suitability checks. Include measurement conditions, uncertainty availability, provenance, and training/evaluation relationships.

Do not count a spectrum, its peak list, and constants derived from it as independent validation. Verify that each record can support the claimed task; envelope data alone may not support high-resolution assignment or calibrated precision claims.

Then extend supported models and modalities under explicit capability profiles while retaining expert control, common interaction patterns, and independent validation.

## Access and evidence notes

- HTTPS Git read access succeeded through injected platform authentication. Do not request a GitHub token merely because token environment variables are unset.
- GitHub CLI issue/PR API reads returned Forbidden during the review; issue/PR discussions were not inspected. Successful Git access does not imply API permission.
- No separate current HANDOFF file was found before this one. The historical Physics_Autopsy_Report.md is an empty template; several old audit reports reference unavailable Windows-local evidence.
- Store source/run provenance durably even if numerical workers are stateless. Do not inherit the archived SRS's silent-ingestion-drop or zero-source-history language.
- Publishing requirements is separate from validating or releasing executable software. Report each outcome according to actual evidence.
