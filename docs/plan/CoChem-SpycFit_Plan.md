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
| 0. Baseline | The PI creates the tag `legacy-swarm-2026-10` on `7e42507`. An agent writes `docs/evidence/phase0_inventory.md` from a clone outside the repository. The inventory covers the whole-suite pytest manifest including collection errors, the diff of the two package trees, the `REQ-*` ids per test file, fabricated-success paths and a dependency scan. No changes to code, tests or configuration. If the PI explicitly asks for fabricated fallbacks to be removed before G1, that is a separate PR that only deletes lines listed in the inventory and quotes the PI's request. | This handoff | **G0**: PI answers D-SPYC-01 and approves these phases and gates. |
| 1. Specification | SRS v2, architecture and this plan completed with the PI, one chapter per PR, with an acceptance catalog, a benchmark manifest with frozen tolerances, and a disposition of every review finding. Benchmark sources are recorded (DOI or URL, SHA-256) without committing data files. Decisions needed before G1 are listed in `docs/handoff/PI_DECISIONS.md`. | G0 | **G1**: PI approves and tags `srs-v2.0`. No code PR before that tag exists on origin, created by the PI. |
| 2. Hygiene | One PR under the PI's D-SPYC-15 choice: <ul><li>a single package tree;</li><li>foreign code and committed junk removed, limited to paths listed in the Phase 0 inventory and approved by the PI;</li><li>every deleted test file listed with its reason, and no test deleted, skipped or xfailed to make CI green;</li><li>`pytest.ini` runs the whole suite, registers the `req` marker and enables `--strict-markers`;</li><li>CI runs the tests and builds the pinned SPFIT/SPCAT (D-SPYC-17);</li><li>`.gitignore` ignores run outputs by directory and allows `tests/fixtures/` and `tests/benchmarks/`;</li><li>the linter is reduced per D-SPYC-08 and D-SPYC-09;</li><li>the README is rewritten from SRS v2.</li></ul> | G1 | **G2**: PI approves the PR with CI green, or with a substitute the PI recorded in `docs/decisions/` if GitHub Actions cannot run. |
| 3. Work packages | One PR each, in the order below. | G2 | **G3** per package: PI reads `docs/evidence/WP-<NAME>.md` and approves. |
| 4. Release | Preregistered benchmark run; SPFIT/SPCAT round trip; clean-wheel install; UI journey on a reference complex per deployment target. | All G3 | **G4**: PI signs the release tag. |

## Proposed work packages

Final content and requirement ids come from SRS v2. Each package's acceptance tests and
tolerances are approved by the PI before its implementation PR. Packages whose scope depends on
an open decision say so.

1. **WP-PICK**: Pickett file I/O and round trip through the pinned SPFIT/SPCAT (D-SPYC-17).
2. **WP-EXP**: experimental spectrum and peak-list input; peak picking with an uncertainty model.
3. **WP-HAM**: the in-process model, if any (D-SPYC-16, D-SPYC-18): Watson A/S in the chosen
   representations, Wang-block labelling, quadrupole hyperfine and multi-state structure,
   each term matching SPCAT.
4. **WP-FIT**: the SPFIT driver as fit of record (D-SPYC-01): parameter state (fitted, fixed,
   constrained), uncertainties, correlation matrix and residual tables.
5. **WP-INT**: intensities with user spin weights, selection rules and catalogs at T_rot
   (D-SPYC-13, D-SPYC-16).
6. **WP-DIAG**: inertial defect, planar moments and κ (D-SPYC-12).
7. **WP-ASSIGN**: prediction windows, candidate ranking, pattern tools and the assignment loop.
8. **WP-CAMPAIGN**: isotopologues, conformers and structure (D-SPYC-10).
9. **WP-LAM**: internal rotation and tunnelling (D-SPYC-04).
10. **WP-UI**: the single UI stack (D-SPYC-05).
11. **WP-IF**: BASE integration (D-SPYC-19), and the TORQ bundle consumer once TORQ's exporter
    exists.
12. **WP-OUT**: publication export and the refinement request (D-SPYC-03).

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
- The PI approves the PR.

Proposals from review §9.2-§9.3 for the PI to accept or reject at G1 (errata row 15):
- a limit of about 1,500 changed lines per PR;
- an 85 % line-coverage floor on touched files;
- `tests/acceptance/manifest.yaml` with a CI check that tolerances were committed before the
  implementation;
- `docs/agent_tasks/<WP>/` for the prompts given to agents.
