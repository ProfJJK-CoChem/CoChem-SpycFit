# CoChem-SpycFit plan

| Field | Value |
| :--- | :--- |
| Status | **Proposed.** Awaiting PI approval at gate G0. |
| Source | Review §9 (`docs/wiki/review/CoChem-SpycFit_SRS_Review.md`), adapted in `docs/handoff/AGENT_HANDOFF.md` §5. |
| Rule | Every gate is a PI approval on a PR or a PI-signed tag. No phase advances on an agent's statement. |

## Phases and gates

| Phase | Work | Entry | Exit gate |
| :--- | :--- | :--- | :--- |
| 0. Baseline | Tag `legacy-swarm-2026-10` (PI pushes or approves). Inventory in `docs/evidence/phase0_inventory.md`: whole-suite pytest manifest including collection errors; diff of the two package trees; `REQ-*` ids used by tests; fabricated-success paths; dependency scan. | This handoff | **G0**: PI answers D-SPYC-01 and approves this plan. (D-SPYC-14 is needed before G1; D-SPYC-11 and D-SPYC-15 before Phase 2.) |
| 1. Specification | SRS v2, architecture and this plan completed with the PI, one chapter per PR; acceptance catalog; benchmark manifest with frozen tolerances; disposition of every review finding. | G0 | **G1**: PI approves; tag `srs-v2.0`. No code PR before the tag exists. |
| 2. Hygiene | One PR: single package tree; foreign code and committed junk removed; `pytest.ini` runs the whole suite; CI runs tests and builds SPFIT/SPCAT from the hash-verified sources; `.gitignore` ignores run outputs by directory; linter reduced per D-SPYC-08; README rewritten from SRS v2. | G1 | **G2**: PI approves the PR with CI green. |
| 3. Work packages | One PR each, in dependency order (below). | G2 | **G3** per package: PI reads `docs/evidence/WPnn.md` and approves. |
| 4. Release | Preregistered benchmark run; SPFIT/SPCAT round trip; clean-wheel install; UI journey on a reference complex per deployment target. | All G3 | **G4**: PI signs the release tag. |

## Proposed work-package order

Final content and ids come from SRS v2. Each package's acceptance tests and tolerances are
committed and approved before its implementation PR.

1. **WP-PICK**: Pickett file I/O and round trip through the pinned SPFIT/SPCAT.
2. **WP-EXP**: experimental spectrum and peak-list input; peak picking with an uncertainty model.
3. **WP-HAM**: one Hamiltonian builder (Watson A/S in the chosen representations), Wang-block
   labelling, quadrupole hyperfine, multi-state structure; agreement with SPCAT energies.
4. **WP-FIT**: SPFIT driver as fit of record; parameter state (fitted, fixed, constrained);
   uncertainties, correlation matrix and residual tables.
5. **WP-INT**: intensities with user spin weights, selection rules, catalogs at T_rot.
6. **WP-DIAG**: inertial defect, planar moments, κ.
7. **WP-ASSIGN**: prediction windows, candidate ranking, pattern tools, assignment loop.
8. **WP-CAMPAIGN**: isotopologues, conformers, structure (per D-SPYC-10).
9. **WP-LAM**: internal rotation and tunnelling (per D-SPYC-04).
10. **WP-UI**: the single UI stack (per D-SPYC-05).
11. **WP-IF**: BASE integration; TORQ bundle consumer once TORQ's exporter exists.
12. **WP-OUT**: publication export and refinement request (per D-SPYC-03).

## Definition of done (every package)

TORQ SRS §16: "Every work package must produce executable implementation, tests tied to
requirements, migration notes for affected persisted data, and a measured evidence report. A
checklist, source stub, hardcoded result, synthetic engine output or claimed agent audit cannot
close a package."

Additionally:
- every SHALL in the package has a passing test marked with its requirement id;
- at least one test compares with a published value or pinned SPFIT/SPCAT output, using a
  tolerance committed before the implementation;
- the whole suite runs in CI;
- `docs/evidence/WPnn.md` (two pages at most) records what ran, where, the numbers against
  tolerance, skips and open questions;
- the PI approves the PR.
