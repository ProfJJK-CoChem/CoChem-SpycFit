# CoChem-SpycFit plan

| Field | Value |
| :--- | :--- |
| Status | **Proposed.** Phases and gates are approved by the PI at G0. The work-package table and definition of done are approved with SRS v2 at G1. |
| Authority | This plan is authoritative for phases, gates and work packages. SRS v2 §9 lists the requirement ids per package. |
| Source | Review §9, superseded and adapted here (`docs/handoff/REVIEW_ERRATA.md` row 15). |
| Rule | Every gate is an explicit PI approval as defined in `AGENTS.md`. No phase advances on an agent's statement. |

## Phases and gates

| Phase | Work | Entry | Exit gate |
| :--- | :--- | :--- | :--- |
| 0. Baseline | The PI creates the tag `legacy-swarm-2026-10` on `7e42507` as a signed tag from their own machine (the route the PI selected on 2026-10-18; G0 amendment 2; D-SPYC-15 is not used to make the tag optional); the PI's G1 and G4 comments name the tag and its SHA. An agent writes `docs/evidence/phase0_inventory.md` from a clone outside the repository. The inventory covers the whole-suite pytest manifest including collection errors, the diff of the two package trees, the `REQ-*` ids per test file, fabricated-success paths and a dependency scan. The former clause "No changes to code, tests or configuration" was removed by the PI on 2026-10-18 ("Remove both completely we are far enough along that they are not needed anymore.", `docs/decisions/D-SPYC-25.md`); `e87f6f4` corrected the amendment §5 defects at the PI's 2026-10-16 request before that removal. If the PI explicitly asks for fabricated fallbacks to be removed before G1, that is a separate PR that only deletes lines listed in the inventory and quotes the PI's request. | This handoff | **G0**: D-SPYC-01 is DECIDED and the PI approves these phases and gates. |
| 1. Specification | SRS v2, architecture and this plan completed with the PI, one chapter per PR, with an acceptance catalog, a benchmark manifest with frozen tolerances, and a disposition of every review finding. Benchmark sources are recorded (DOI or URL, SHA-256) without committing data files. Decisions needed before G1 are listed in `docs/handoff/PI_DECISIONS.md`. | G0 | **G1**: PI approves and tags `srs-v2.0`. The former code gate "No code PR before that tag exists on origin" was removed by the PI on 2026-10-18 (`docs/decisions/D-SPYC-25.md`); a dedicated governance PR, approved by the PI in writing, removes the same wording from `AGENTS.md` and the handoff. The approved SRS still wins over code where they disagree, and code PRs name the requirement ids they implement; G2-G4 stay. |
| 2. Hygiene | One PR under the PI's D-SPYC-15 choice: <ul><li>a single package tree;</li><li>foreign code and committed junk removed, limited to paths listed in the Phase 0 inventory and approved by the PI;</li><li>every deleted test file listed with its reason, and no test deleted, skipped or xfailed to make CI green;</li><li>`pytest.ini` runs the whole suite, registers the `req` marker and enables `--strict-markers`;</li><li>CI runs `pytest tests` and builds SPFIT, SPCAT, DPFIT and DPCAT from manifest-verified JPL sources (D-SPYC-17; SF-RUN-009), commits no binaries and records the HCO and NO acceptance results (SF-RUN-007); the `air_gap_enforcement` `find` step, which fails on any `.lin`, `.fit`, `.h5`, `.parquet` or `.lock` file outside a `test_fixtures` directory, is removed;</li><li>`.gitignore` ignores run outputs by directory and allows `tests/fixtures/` and `tests/benchmarks/`;</li><li>the linter is reduced per D-SPYC-08 and D-SPYC-09;</li><li>the README is rewritten from SRS v2.</li></ul> | G1 | **G2**: PI approves the PR (in a PR comment, per `AGENTS.md`) with CI green, or with a substitute the PI recorded in `docs/decisions/` if GitHub Actions cannot run. B-split (G0 amendment 4, PI 2026-10-18): G2a, a PR the PI permits in writing before it opens, puts the whole suite and the four-program build in CI and publishes the summary; G2b, the hygiene PR, then goes green. |
| 3. Work packages | One PR each, in the order below. | G2 | **G3** per package: PI reads `docs/evidence/WP-<NAME>.md` and approves the PR in a PR comment. |
| 4. Release | Two releases (D-SPYC-24; G0 amendment 5, PI 2026-10-18). For each: preregistered benchmark run; SPFIT/SPCAT round trip; clean-wheel install; acceptance evidence for every profile named at G1, WP-VIB and the electronic-state scope (D-SPYC-06) included; the publication demonstration-cases document (SF-VER-007); UI journey on a reference complex per deployment target except HPC, which is headless (D-SPYC-05). The BASE build mechanism (task 11) and the SF-RUN-009 → WP-PICK mapping sit in the work-package table for G1. | All G3 of that release | **G4** per release: PI signs the release tag. |

State on 2026-10-18 (G0 amendment 6): the Phase 0 inventory exists
(`docs/evidence/phase0_inventory.md`) and lacks the per-test manifest and the `REQ-*` list per
test file that Phase 0 names; the baseline tag is not created; `e87f6f4` changed 21 files before G1
at the PI's 2026-10-16 request (PR #1); on 2026-10-18 the PI removed the Phase 0 no-code clause and
the `srs-v2.0` code gate (D-SPYC-25), the AGENTS.md and handoff wording following in the dedicated
governance PR from `ccr-c1697164-ao6hz4-governance` that the PI permitted; GitHub Actions
runs the workflow's single `find` step and `pytest.ini` runs one file; SRS v2 Draft 1 with its
2026-10-17 and 2026-10-18 updates is in Phase 1 review. On 2026-10-18 the PI chose "G0: B" with all
six amendments; G0 closes when D-SPYC-01 is DECIDED by the PI's PR #1 comment naming the gate and
the commit.

Release staging (D-SPYC-24, PI 2026-10-18; a plan amendment for the PI's G0 approval): Phase 4
runs twice, each release with its own G4. Release 1, in this order: UC-7; UC-8 with Pickett files;
the WP-PICK round trip; UC-1 with SPFIT as the fit of record; UC-9 when GEOM agrees the payload
schema; UC-5 when the engine research lands and native terms meet SF-HAM-001; UC-4 as P8
condensed-phase envelopes only. Release 2: UC-2, UC-3, UC-6, profiles P2-P4, P7 and P15. The
demonstration molecules follow research R6 (per engine component, two isotopologues from reputable
databases); the PI supplies the measured spectra.

## Proposed work packages

Final content and requirement ids come from SRS v2. Each package's acceptance tests and
tolerances are approved by the PI before its implementation PR. Packages whose scope depends on
an open decision say so.

1. **WP-PICK**: Pickett file I/O and round trip through SPFIT, SPCAT, DPFIT and DPCAT built from
   the pinned JPL sources (D-SPYC-17; SF-RUN-006..009a), provisioned by the BASE build mechanism
   the PI asked for (task 11; D-SPYC-26, PI 2026-10-18: BASE owns the provisioner and the pin, the
   build runs automatically in BASE's Codespaces setup script, BASE's in-process SPCAT fallback is
   removed in the same BASE PR, and the `hcotst.par` route satisfies V-PICK-05 while `hco.var` is
   unavailable; `docs/handoff/briefs/BASE-BUILD_2026-10-17.md`). SpycFit reads the receipt
   (SF-RUN-009a); the BASE side is draft PR ProfJJK-CoChem/CoChem-BASE#14 (2026-10-18), with a
   local run recorded in `docs/evidence/pickett_build_2026-10-18.md`.
2. **WP-EXP**: experimental spectrum and peak-list input; peak picking with an uncertainty model.
3. **WP-HAM**: the in-process model, if any (D-SPYC-16, D-SPYC-18): Watson A/S in the chosen
   representations, Wang-block labelling, quadrupole hyperfine and multi-state structure,
   each term matching SPCAT.
4. **WP-FIT**: the SPFIT driver as fit of record (D-SPYC-01): parameter state (fitted, fixed,
   constrained), uncertainties, correlation matrix and residual tables.
5. **WP-INT**: intensities with user spin weights, selection rules and catalogs at T_rot
   (D-SPYC-13: three-state component labels and a relative intensity floor as a recorded view
   that never deletes, preset after research; D-SPYC-16: catalogs split by producer).
6. **WP-DIAG**: inertial defect, planar moments and κ (D-SPYC-12: typed, uncertainty-aware flag
   at k = 2 and never rejection; export review record; off-by-default search window with logged
   drops; ranking only; recorded corrections; the legacy module's seven rejection tests are
   rewritten in Phase 2 with the PI's approval).
7. **WP-ASSIGN**: prediction windows with the D-SPYC-21 per-method-class defaults, candidate
   ranking, pattern tools and the assignment loop.
8. **WP-CAMPAIGN**: isotopologues and conformers, and the constants GEOM needs for structure
   determination (D-SPYC-10: structure determination is GEOM's).
9. **WP-LAM**: internal rotation and tunnelling (D-SPYC-04: engines XIAM with the XIAMALL.FOR
   reference build and ERHAM, later BELGI-Cs/C1, RAM36 and westerfit, built by BASE's mechanism;
   wrap versus native waits for the research the PI requested on 2026-10-17).
10. **WP-UI**: the single UI stack (D-SPYC-05: a notebook rendered by BASE's Voila in a
    SpycFit-environment kernel, H2; HPC headless in the first release; the toolkit follows the
    usability research the PI requested on 2026-10-17).
11. **WP-IF**: BASE integration (D-SPYC-19: BASE is required; D-SPYC-23: SpycFit publishes its own
    operations and provides none of BASE's three declared names), the direct-input path for
    user-supplied ORCA and Pickett files through BASE input kinds (route ii, blocked on BASE), the
    GEOM payload, and the TORQ bundle consumer once TORQ's exporter exists.
12. **WP-OUT**: publication export and the refinement request (D-SPYC-03: a BASE handoff kind
    with a SpycFit-drafted schema, staged; the BASE requirement waits on BASE and a receiver).
13. **WP-CMP**: the native-versus-SPCAT/SPFIT comparison of amendment §3 (SRS v2 §5.6),
    starting with the semirigid profile. The PI chose no gate on 2026-10-17 (D-SPYC-14, follow-up
    4: "No gate: report the comparison only"): SF-CMP-021a and 021b results are reported with
    WP-BENCH's registry and held-out partitions; forward prediction stays gated by SF-HAM-001
    alone.
14. **WP-BENCH**: the benchmark registry: reference records, redistribution classes,
    fetch-and-verify fixtures, lineage partitions (SRS v2 §4.6, §8.3; D-SPYC-14).
15. **WP-VIB**: the vibrational, rovibrational, Raman, THz and near-IR profiles of SRS v2 §1.2,
    each with its own acceptance evidence (PI 2026-10-16; D-SPYC-06).

Packages 13-15 were added on 2026-10-16 to follow the PI's 2026-10-16 scope; their order is a
proposal.

## Definition of done (every package)

TORQ SRS §16: "Every work package must produce executable implementation, tests tied to
requirements, migration notes for affected persisted data, and a measured evidence report. A
checklist, source stub, hardcoded result, synthetic engine output or claimed agent audit cannot
close a package."

Additionally:
- Every SHALL in the package has a passing, non-skipped test marked with its requirement id.
- Scientific packages (PICK, HAM, FIT, INT, DIAG, ASSIGN, CAMPAIGN, LAM, OUT) have at least one
  test comparing against a published value or the pinned SPFIT/SPCAT output, with a tolerance the
  PI approved before the implementation PR.
- For EXP, UI and IF, SRS v2 names the acceptance evidence and the PI approves it before
  implementation. Examples are a PI-supplied spectrum file, a scripted UI journey and a BASE
  handoff round trip.
- The whole suite runs in CI.
- `docs/evidence/WP-<NAME>.md` (two pages at most) records what ran, where, the numbers against
  tolerance, skips and open questions.
- The PI approves the PR in a PR comment, as defined in `AGENTS.md`.

Proposals from review §9.2-§9.3 for the PI to accept or reject at G1 (errata row 15):
- a limit of about 1,500 changed lines per PR;
- an 85 % line-coverage floor on touched files;
- `tests/acceptance/manifest.yaml` with a CI check that tolerances were committed before the
  implementation;
- `docs/agent_tasks/<WP>/` for the prompts given to agents.

Review §9.3's `docs/agent_reports/` is not carried forward (errata row 15).
