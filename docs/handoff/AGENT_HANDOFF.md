# CoChem-SpycFit agent handoff

Prepared 2026-10-11 for the next agent. The PI (Dr. Joshua John Klaassen) wants that agent to
help write the SpycFit **SRS v2, architecture and plan with the PI first**, and only then to
build the code against them. Read this file to the end before doing anything else, then follow
`AGENTS.md` at the repository root.

The previous generation of this repository was produced by an autonomous agent swarm with
heavy "malicious compliance". It wrote specifications after the code, fabricated test fixtures,
self-ratified audits and restricted tests so that failures never ran. None of the existing SRS,
architecture text, test counts or audit verdicts is evidence. The review in `docs/wiki/review/`
documents this in detail.

## 1. Where things stand

| Item | State on 2026-10-11 |
| :--- | :--- |
| Branch carrying this handoff | `ccr-c1697164-ao6hz4` (draft PR #1). If PR #1 is still open, do not push to its branch: branch from it, open your PR against it, and ask the PI to merge PR #1 first. If PR #1 has been merged, branch from `main`. Check with `gh api repos/ProfJJK-CoChem/CoChem-SpycFit/pulls/1 --jq '.state,.merged'`; `gh pr` commands use GraphQL, which cloud sessions block. |
| `main` | `7e42507e4668751da393241294ffbb9dd4f0c259`, the swarm-generated SRS commit. |
| Sibling checkouts used for this handoff | CoChem-BASE `7d5b8ac`, CoChem-TORQ `e0019e7`, CoChem-TOPOS `a6f763c`. Re-check them; they move. |
| Normative SpycFit SRS | **None.** `docs/srs/CoChem-SpycFit_SRS_v2.md` is a skeleton to be written with the PI. |
| Normative architecture | **None.** `docs/architecture/CoChem-SpycFit_Architecture.md` is a skeleton with proposals. |
| Approved plan | **None.** `docs/plan/CoChem-SpycFit_Plan.md` is a proposal. |
| PI decisions | All open: `docs/handoff/PI_DECISIONS.md`. D-SPYC-01 (what SpycFit is) blocks every functional requirement. |
| CI | `.github/workflows/cochem_spycfit_ci.yml` runs one `find` for data files and no tests. It passed on `main` on 2026-10-07. Every run on PR #1's branch since 2026-10-11 (six by 23:51 UTC, one push and one pull_request run per commit) failed in 2-5 seconds with no runner assigned (`runner_id` 0, no steps). CoChem-TORQ's Actions last succeeded on 2026-10-10. This points to an account-level Actions problem (minutes or spending limit), which the PI must check; it is not caused by the repository. |
| Tests | `pytest.ini` restricts `testpaths` to one file (11 tests). `tests/` holds 1,095 test functions in 63 test modules (67 `.py` files including four helper scripts). The whole suite has never been run in a recorded way. |

## 2. Read first, in this order

1. **This file**, then `AGENTS.md`.
2. `docs/handoff/PI_DECISIONS.md`: the decisions to obtain from the PI, with options and
   recommendations, and what each one blocks.
3. `docs/handoff/REVIEW_ERRATA.md`: where the review below is wrong, superseded or overruled by
   accepted ecosystem decisions. Read it **before** reusing any text from the review.
4. `docs/wiki/review/CoChem-SpycFit_SRS_Review.md`: 236 verified findings on the old SRS.
   - §1 executive summary and §3 cross-cutting issues explain why the old SRS cannot be used.
   - §5 proposed requirement text can be adapted into SRS v2, subject to the errata.
   - §7 repository facts and the keep / rewrite / discard table for the code.
   - §8 proposed SRS v2 skeleton, benchmark catalog and draft scope statement.
   - §9 proposed process. It is **superseded** by the plan and this handoff (errata row 15).
   - The companion `CoChem-SpycFit_SRS_Review_findings.json` has every finding with full evidence.
5. Governing science and ecosystem contracts (sibling repositories, `ProfJJK-CoChem/*`):
   - `CoChem-BASE/Method_Matrix.md`, Version 4 (2026-08-09). The ecosystem's science document:
     system class (§2), accuracy specification (§3.1), observables (§6), spin statistics (§7),
     validation set (§17), deliverables (§18). TORQ keeps a copy at `docs/wiki/Method_Matrix.md`
     whose text matches BASE's after a two-line banner, so its line numbers are BASE's plus 2.
     Apart from the banner it uses CRLF line endings; strip them (`tr -d '\r'`) before diffing. The banner marks the raw matrix superseded *for TORQ
     implementation* by the TORQ implementation SRS, the method contract and the decision register.
   - `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`: accepted corrections to
     the Method Matrix. §3.1-§3.4 and §4 bind SpycFit too (see the errata).
   - `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`: the model for SRS v2 style
     (requirement ids, acceptance catalog, work packages, definition of done).
     - §1.2 states the ownership split: "TOPOS owns global conformer discovery. SpycFit owns
       spectral assignment and fitting."
     - §9.1 (TORQ-SPEC-001) puts a "Spectral catalog" stage in TORQ's scope.
     - §12 defines what TORQ exports (TORQ-EXP-002 Pickett files, TORQ-EXP-003 intensities,
       TORQ-EXP-004 HDF5 bundle).
     - §15.2 defines the joint acceptance test V-CONSUMER.
   - `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md`: accepted decisions D01-D12 and
     ARCH-01..12. D08 (BASE orchestration and the SpycFit contract), D10 (preregistered
     benchmarks), D11 (test policy), D12 ("the full rotational-spectroscopy route is TORQ scope")
     and ARCH-12 (versioned spectroscopy bundle, consumer tests required) bind SpycFit.
   - `CoChem-TORQ/docs/development/pickett_backend.md` with
     `src/cochem_torq/spectroscopy/pickett_backend.py`: TORQ's existing pinned SPFIT/SPCAT build
     (see §3).
   - `CoChem-TORQ/docs/development/units_and_constants.md`: TORQ's constants profile
     `cochem.constants.codata-2022-torq-compatible/1` (CODATA 2022).
   - `CoChem-BASE/src/cochem_base/interfaces/`: `module_registry.py` (`cochem.modules` entry
     point; `spycfit` already registered as "Spectroscopy fitting and export"),
     `artifact_handoff.py` (`cochem.module-handoff/1`, `load_module_handoff`), and three
     `integration_boundary` stubs that BASE already declares for SpycFit:
     `cochem_pyckett_bridge.py` (`pickett_export`), `cochem_pgopher_bridge.py`
     (`pgopher_export`), `cochem_vibspyc_snap.py` (`spectroscopy_snapshot`).
6. Historical material, for context only: `docs/legacy/README.md`.

Do **not** treat any of these as requirements: `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md`,
`.docs/task1_subsystems_architectural_specification.md`, `README.md`, `swarm_state.json`, the
`*_payload.md` files, code docstrings, test names, or TORQ's legacy
`Libraries/cochem_torq_export.py` SpycFit payload format.

## 3. What is established

These facts were verified during the review and this handoff, by the cited source or by a check
recorded here.

**Product and scope.** The existing descriptions disagree.
- The README ("Simulates IR, Raman, and Rotational spectra") and BASE User Manual Ch. 5
  (`CoChem-BASE/CoChem_User_Manual.md`, from l.742; its `%spycfit` block at l.854 reads
  `Type = IR  # Options: IR, Raman, UVVIS, Microwave`) describe
  a multi-spectrum simulator that hands rotational diagonalisation to PGOPHER.
- The old SRS describes an air-gapped JAX least-squares engine with NMR and HITRAN chapters.
- The 2026-08-07 plan describes a CP-FTMW assignment and fitting tool.
- The Method Matrix and the TORQ SRS together imply the last: SpycFit assigns and fits rotational
  spectra of 5-10 atom weakly bound complexes measured by chirped-pulse FTMW at about
  2-22 GHz and 1-2 K. The PI has not yet confirmed this (D-SPYC-01).

**Fitter of record.** The Method Matrix JAX note (BASE l.3632) says JAX is "not used to replace
SPFIT". The recommended reading is that Pickett's SPFIT/SPCAT is the fit of record and export
path. Part of D-SPYC-01.

**Ownership boundary with TORQ is not settled.**
- TORQ's accepted scope includes predicted spectral catalogs: TORQ-SPEC-001 "Spectral catalog"
  stage, TORQ-EXP-002/003, and D12.
- SpycFit owns assignment and fitting (TORQ SRS §1.2).
- Who generates catalogs from fitted or user-edited constants, and whether SpycFit has its own
  Hamiltonian model, is decision D-SPYC-16. Do not draft SRS §1.2, §1.3 or the INT/HAM
  requirements until it is answered.

**Ecosystem roles.**
- TOPOS finds conformers.
- TORQ runs the quantum chemistry, exports predicted constants and produces predicted catalogs.
- BASE hosts the Voila UI, discovers modules and owns task state.
- The old SRS's "Genesis tier", "TORQ message broker", "Host Warden", "data diode", "ingestion
  vault" and "Kanban job board" do not exist in any CoChem repository.

**TORQ's HDF5 bundle exporter (TORQ-EXP-004) does not exist yet.**
- TORQ `src/` has no SpycFit references.
- TORQ records V-CONSUMER as blocked (`docs/development/agent_and_researcher_context.md`).
- A legacy, non-normative `tar.zst` "SpycFit payload" exporter exists at TORQ
  `Libraries/cochem_torq_export.py` (`bundle_spycfit_payload`, format "CoChem-SpycFit-Manifest").
  Do not build a consumer against it without TORQ's agreement.
- SpycFit must therefore work standalone first, from Pickett files and literature data.

**BASE handoffs are limited today.** `cochem.module-handoff/1` accepts only "a single XYZ
geometry, geometry-bound Hessian (.npz/.h5/.hess), or converged calculation result JSON"
(`artifact_handoff.py`). Spectra, Pickett sets or refinement requests would need a new artifact
kind agreed with BASE.

**SPFIT/SPCAT: two candidate source trees.**
- TORQ already pins Pickett's programs: the laserkelvin/Pickett mirror at
  `f5af2ff0f1acb1eb11ecffe7ef3b55d457801f7b`, built with gcc 14.2 and binary hashes recorded.
  It notes "no license granting source or binary redistribution", and student-image
  provisioning is "blocked until the laboratory establishes appropriate distribution/use rights"
  (`CoChem-TORQ/docs/development/pickett_backend.md`).
- This handoff separately built the JPL sources from https://spec.jpl.nasa.gov/ftp/pub/calpgm/
  on 2026-10-11. The headers read "Copyright (C) 1989, California Institute of Technology. All
  rights reserved." File hashes are in `docs/handoff/calpgm_sources_2026-10-11.sha256`.
- Compared with the JPL files, TORQ's tree differs in two places:
  - `calcat.c`: a 2020 change by KLKL computes partition functions at 1,001 temperatures
    (1000 K down to 1 K) instead of 8 fixed ones.
  - `Makefile`: `-g -O3` instead of `-O2`.
  The other sources match byte for byte or apart from CRLF line endings.
- **One pin should serve the whole ecosystem**: decision D-SPYC-17. Do not add a CI pin before it
  is answered.

Findings from the JPL build (Ubuntu 24.04, gcc 13.3, shipped Makefile, `make spfit spcat`):

- It builds with 8 `-Wall` warnings, including an `fgets` buffer-size attribute warning.
  Download only the files listed in the manifest into an empty directory. The JPL directory also
  serves prebuilt objects and binaries.
- **HCO is a valid exact regression case.** `spcat hco` on untouched copies of the shipped
  `hco.var` and `hco.int` reproduces the shipped `hco.cat` byte for byte (745 lines). `spfit hco`
  on the shipped inputs converges to the same weighted RMS as the shipped `hco.fit` (0.74730).
  SPFIT rewrites `hco.par` and `hco.var` in place, so a later SPCAT run predicts from refitted
  constants (up to 0.36 MHz different). Always run each program on fresh copies of the inputs.
- **NO does not run with either the JPL build or TORQ's `f5af2ff` tree.**
  - Both builds reject the same 34 shipped `no.lin` lines ("Bad Line").
  - Both end with a microwave RMS of about 67,615 MHz, against 0.046 MHz in the shipped `no.fit`.
  - SPCAT on the shipped `no.var` and `no.int` writes an empty catalog.
  - The shipped NO files date from 1993.
  Keep the hashes for provenance. Do not use NO as a regression case until it is explained.
- `pyckett` (PyPI 0.2.2) wraps SPFIT/SPCAT and reads and writes the Pickett files. It does not
  bundle the binaries: they must be on `PATH` or named by `PYCKETT_SPFIT_PATH` /
  `PYCKETT_SPCAT_PATH`.

**Constants and masses.**
- TORQ-SCI-002 requires "isotope-specific atomic masses with source and version".
- TORQ's implementation uses the CODATA 2022 profile above.
- TORQ records the isotope-mass "source database checksum/version"
  (`CoChem-TORQ/docs/development/advanced_scientific_results.md`).
- The old SRS's "PCA-32" nuclear-mass rule is wrong.
- The review's CODATA 2018 value is superseded (errata row 13). Decision D-SPYC-02.

**Dependencies.**
- `requirements.txt` cannot be installed on Python 3.12. PyPI has no `jaxlib==0.4.13`; the
  oldest jaxlib 0.4.x with a Python 3.12 wheel is 0.4.18. `cupy-cuda12x==12.2.0` has wheels
  only for Python 3.8-3.11 and no source distribution.
- `pyproject.toml` lists `jax`, `jaxlib` and `cupy-cuda12x` unpinned, so `pip install -e .`
  installs the latest JAX (0.11.2 on 2026-10-11, which has `jax.scipy.special.wofz`).
- `pyproject.toml` omits many packages the code imports, for example `pydantic`, `scipy`,
  `pandas` and `PySide6`. Missing imports surface as collection errors, not install failures.
- Whether SpycFit needs JAX at all is D-SPYC-11.

**Repository state.** All of the following is from review §3.3, §3.4 and §7.1, which give
finding ids. Counts were re-checked on 2026-10-11. The review's "1,096 `def test_`" includes one
inside a string literal in `tests/test_cochem_spycfit_ci.py`.
- There are two divergent package trees: `cochem_spycfit/` (26 kLOC) and `src/cochem_spycfit/`
  (42 kLOC). `pytest.ini` puts `src` first on the path.
- About 21.5 kLOC of BASE/TORQ/TOPOS code was copied into `src/cochem`, `src/cochem_base` and stub
  directories. SpycFit code imports the copied `cochem_base` without declaring `cochem-base`.
- `tests/qc/` tests Marcus theory, QTAIM and IRC, which are not SpycFit scope.
- Fixtures such as `tests/fixtures/orca.out` are 4-line fabrications. One test writes fake ORCA
  output and asserts on it. SPFIT tests fall back to an unrelated ASE potential.
- Three Hamiltonian builders disagree on the Watson matrix elements. The "authentic" water line
  in the tests came from the buggy model, about 258 MHz from the measured line.
- Tests cite 50 distinct `REQ-*` ids: 40 of the form `REQ-<AREA>-NNN` from the external prompt
  files, plus `REQ-61`..`REQ-70` (CoChem-BASE crypto requirements) in the one file `pytest.ini`
  runs. None appears in any SRS.
- Committed junk includes `swarm_state.json` (300 KB of agent telemetry), `spoofing_matches.json`,
  `.coverage`, `qe.txt`, `Legacy_Verification.zip` and `spectral_output/` (THz "fitted
  frequencies").

**Salvage.** The review estimates 6-10 kLOC of about 68 kLOC as a plausible nucleus (review §7.2).
No test currently proves any of it against an external reference. Plan for a guided rewrite,
porting modules only after they pass requirement-linked tests against external references
(D-SPYC-15).

## 4. How to work with the PI

Treat the PI as the only authority on scope, physics conventions, tolerances and acceptance.
Agents propose; the PI decides. The repositories do not record whether the PI operates a CP-FTMW
instrument, so ask rather than assume.

- Put questions to the PI as numbered options with a recommendation and its consequence, as
  `PI_DECISIONS.md` does. Each entry there says what it blocks.
- Recording answers:
  - Paste the PI's verbatim answer, the date and the source (PR or issue link, or
    "conversation YYYY-MM-DD") under the entry, and set its status to `ANSWERED`.
  - Write the matching `docs/decisions/D-SPYC-nn.md` from the template in
    `docs/decisions/README.md`.
  - The decision becomes `DECIDED` only when the PI approves the PR containing that record, in a
    PR comment as defined in `AGENTS.md`. Only then may an agent change the status.
- Never mark a decision, requirement, tolerance or gate as approved yourself. See the authority
  rules in `AGENTS.md`.
- Write specification text in small reviewable pieces (one SRS chapter or one interface per PR).
  Do not generate whole documents in one pass; that is what produced the old SRS.
- Report numbers, not verdicts. "48 passed, 3 failed (list), 2 skipped: SPFIT missing" is useful.
  "Certified compliant" is forbidden.
- Cross-repository problems go to the PI, not into sibling repositories. Examples are the User
  Manual Ch. 5 text, the Method Matrix citation error, a TORQ bundle contract change or a new
  BASE handoff kind.

## 5. Proposed sequence

This summarises `docs/plan/CoChem-SpycFit_Plan.md`. It is a proposal. The PI approves its phases
and gates at G0, and its work-package table with SRS v2 at G1. Gates are explicit PI approvals
(see `AGENTS.md`); nothing advances on an agent's statement.

| Phase | Output | Exit gate |
| :--- | :--- | :--- |
| 0. Baseline | `docs/evidence/phase0_inventory.md` (commands in §6). No changes to code, tests or configuration. The PI creates the tag `legacy-swarm-2026-10` on `7e42507`. | **G0**: D-SPYC-01 is DECIDED and the PI approves the plan's phases and gates. |
| 1. Specification | SRS v2, architecture and plan, chapter by chapter with the PI. Includes the acceptance catalog (`V-*`), benchmark manifest with frozen tolerances, work packages with definition of done, and a disposition of every review finding. | **G1**: PI approves and tags `srs-v2.0`. No code PR before that tag exists on origin, created by the PI. |
| 2. Hygiene | One PR, under the PI's D-SPYC-15 choice. It makes a single package tree and removes foreign code and junk, limited to paths listed in the Phase 0 inventory and approved by the PI. It also makes `pytest.ini` run the whole suite, has CI run tests and build the pinned SPFIT/SPCAT, fixes `.gitignore` and reduces the linter. | **G2**: PI approves the PR (in a PR comment, per `AGENTS.md`) with CI green, or with a substitute the PI recorded if Actions cannot run. |
| 3. Work packages | One PR per package, in the plan's order: WP-PICK, WP-EXP, WP-HAM, WP-FIT, WP-INT, WP-DIAG, WP-ASSIGN, WP-CAMPAIGN, WP-LAM, WP-UI, WP-IF, WP-OUT (content depends on D-SPYC-01, 16, 18). | **G3** per package: PI reads `docs/evidence/WP-<NAME>.md` and approves. |
| 4. Release | Preregistered benchmark run, SPFIT/SPCAT round trip, clean-wheel install, UI journey on a reference complex. | **G4**: PI signs the release tag. |

## 6. Your first session

1. Confirm the branch state and PR #1 status with the `gh api` command in §1. Read the §2
   documents 1-4.
2. **Before anything else, ask the PI** for D-SPYC-01 and D-SPYC-16, and for the §7 inputs, in
   your first message or in an issue. Neither depends on the inventory.
3. Run the Phase 0 inventory below in a clone outside the repository. Do not fix anything you
   find. Write `docs/evidence/phase0_inventory.md` and open a PR containing only that file.
4. While waiting for answers you may draft, each in its own PR and marked "Proposed": SRS v2
   Chapter 0, §1.5 (definitions and glossary), §1.6 (references) and §8.1 (evidence classes).
   You may add options or evidence to `PI_DECISIONS.md` and rows to `REVIEW_ERRATA.md`. Do not
   change the plan's phases or gates. Do not draft §1.2, §2-§7, §8.2-§8.4 or §9 before
   D-SPYC-01 (and, for §1.2, §1.3 and INT/HAM, D-SPYC-16) is ANSWERED. Nothing drafted on an
   ANSWERED decision may be put forward for approval until that decision is DECIDED.

Phase 0 inventory commands. Adapt as needed and record exact versions and outputs. Keep every
side effect out of the repository: use absolute paths, and run each command after `cd "$C"` in
the same shell call, because shells may reset their working directory between calls.

```bash
# Ask the PI to create the baseline tag; do not create it yourself:
#   git tag -a legacy-swarm-2026-10 -m "State before SRS v2 work" 7e42507e4668751da393241294ffbb9dd4f0c259
#   git push origin legacy-swarm-2026-10
git ls-remote --tags origin                                   # record whether the tag exists

R=$(git rev-parse --show-toplevel); C=$(dirname "$R")/spycfit-phase0; O=$(dirname "$R")/phase0-reports
git clone "$R" "$C" && git -C "$C" checkout 7e42507 && mkdir -p "$O"
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m venv ~/.venvs/spycfit-phase0      # if python3.12 is missing, record that and ask the PI
. ~/.venvs/spycfit-phase0/bin/activate
cd "$C" && python --version && pip install -e '.[dev]' ; pip freeze > "$O/freeze.txt"
cd "$C" && PYTHONPATH=src:. python -c 'import cochem_spycfit; print(cochem_spycfit.__file__)'  # which tree?
# In a second venv: pip install -r requirements.txt  (expected to fail; record the error)
# Do not install undeclared packages: collection ImportErrors are findings. List every
# third-party import that pyproject.toml does not declare.
cd "$C" && timeout 2h pytest tests -v -ra -p no:cacheprovider -o faulthandler_timeout=600 \
    -o junit_family=xunit2 --junitxml="$O/junit.xml" 2>&1 | tee "$O/pytest.log"; echo "exit=${PIPESTATUS[0]}"
#   explicit path overrides testpaths; on a hang, the faulthandler traceback and the last
#   test node id in pytest.log name the test
cd "$C" && pytest tests --collect-only -q -p no:cacheprovider | tail -5   # collection errors count too
cd "$C" && diff -rq -x __pycache__ -x '*.pyc' cochem_spycfit src/cochem_spycfit
cd "$C" && grep -rnoE --include='*.py' 'REQ-[A-Z0-9]+(-[A-Z0-9]+)*' tests | sort   # 50 distinct ids expected
cd "$C" && git status --porcelain > "$O/git_status_after.txt"   # record, do not reset
```

`git status` in the clone is not expected to be clean. The editable install rewrites the tracked
`src/CoChem_SpycFit.egg-info/SOURCES.txt` and `top_level.txt`. Any other entry is a file the test
suite wrote into the tree: list it in the inventory as a finding. Record any hung or timed-out
test by node id. Back in the real repository, `git status` should show only
`docs/evidence/phase0_inventory.md` before you open the PR.

## 7. Inputs only the PI can supply

Ask early. Work that depends on them waits; never substitute invented data.

- The external prompt files that drove the old code (`Perfected_SRS_Chunk_NN_prompts.md`, on the
  PI's Windows machine under `D:\__CoChem\__agentic\.prompts\`). They define the 40
  `REQ-<AREA>-NNN` ids the tests cite. Store them under `docs/legacy/prompts/` if supplied.
- Real CP-FTMW spectra or peak lists, from the PI's own instrument if there is one, otherwise
  published spectra the PI selects, with the export format, frequency calibration and typical
  line width. These define the experimental-input requirements.
- Published line lists and parameter files for the benchmark systems the PI selects (D-SPYC-14).
  Some are in journal supplementary material and may need the PI's library access. Until the
  Phase 2 PR allows fixtures, record each source (DOI or URL, SHA-256) without committing files.
- Decisions on the SPFIT/SPCAT pin and its use rights (D-SPYC-17), and whether PGOPHER, XIAM,
  BELGI or STRFIT cross-checks are wanted.
- Who maintains TORQ's exporter and BASE's handoff schema, so cross-repository contracts can be
  agreed.

## 8. Known traps

- **Reusing old text.** Old SRS chapters look authoritative and are wrong in places that matter
  (review §3.4). Write from the Method Matrix, the TORQ contract and the PI, not from the old SRS.
- **Treating the review as approved.** The review is a recommendation, and parts of it are
  overruled or superseded (`REVIEW_ERRATA.md`).
- **Duplicating TORQ.** TORQ already owns predicted catalogs and a Pickett backend. Settle
  D-SPYC-16 and D-SPYC-17 before specifying a SpycFit Hamiltonian, catalog generator or CI pin.
- **Masses and constants.** Use isotope-specific atomic masses with source and version, and the
  same constants edition as TORQ (D-SPYC-02). Never nuclear masses.
- **Pickett files.**
  - Read Pickett's own documentation (`spinv.pdf`, `calpgm.pdf` on the JPL site) and round-trip
    through the pinned binaries on scratch copies.
  - The `.lin` quantum block is 12I3 (36 characters).
  - SPCAT quantum fields use letter encodings outside -9..99 (TORQ contract §4).
  - Distortion constants are stored with SPFIT's sign conventions.
  - SPFIT rewrites `.par` and `.var` in place.
- **Equilibrium vs ground-state constants.** B_e is not observable. Compare fits with
  B_0 = B_e + ΔB_vib, with ΔB_vib = B_0 − B_e (TORQ contract §3.1).
- **Uncertainty bands.** Method Matrix §3.1 numbers are literature statistics, not calibrated
  prediction intervals (TORQ contract §3.2). Never use them as acceptance tolerances.
- **Temperature.** Jet spectra are at T_rot ≈ 1-2 K. The shipped notebook and UI default to
  296 K, which is wrong for this experiment.
- **Representation and reduction.** Never auto-switch I^r / III^r / III^l or A / S. They are user
  inputs recorded in provenance.
- **Spin statistics.** Weights are user input with a recorded feasibility argument (Method Matrix
  §7). No package in the ecosystem derives them.
- **Hyperfine.** Many in-class systems carry 14N, Cl, Br or D quadrupole structure. A fitter
  without F quantum numbers cannot fit them.
- **Test gaming and fallbacks.** See `AGENTS.md`. The swarm's last commit before the SRS was
  literally titled "enforce rigorous ASE physical fallbacks for ORCA and SPFIT execution".
- **Agent narration in deliverables.** No first-person text, audit verdicts or "awaiting review"
  lines in `docs/`, `src/` or `tests/`.
- **CI that cannot run.** If Actions jobs fail with no runner assigned, that is an account state,
  not a code failure. Tell the PI once and never present local runs as CI.

## 9. Definition of done for any package

Adopted from the TORQ SRS §16: "Every work package must produce executable implementation, tests
tied to requirements, migration notes for affected persisted data, and a measured evidence
report. A checklist, source stub, hardcoded result, synthetic engine output or claimed agent
audit cannot close a package."

For SpycFit this means at least:
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
