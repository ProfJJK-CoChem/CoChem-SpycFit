# CoChem-SpycFit agent handoff

Prepared 2026-10-11 and updated 2026-10-16 for the next agent. Read this file to the end before
doing anything else, then follow `AGENTS.md` at the repository root.

The previous generation of this repository was produced by an autonomous agent swarm with heavy
"malicious compliance". It wrote specifications after the code, fabricated test fixtures,
self-ratified audits and restricted tests so that failures never ran. None of its SRS,
architecture text, test counts or audit verdicts is evidence. `docs/wiki/review/` and
`docs/evidence/phase0_inventory.md` document this.

## 1. Where things stand (2026-10-16)

| Item | State |
| :--- | :--- |
| Branch | `ccr-c1697164-ao6hz4`, draft PR #1. It carries the 2026-10-11 review and handoff, a merge of `main` at `488dc87`, the Phase 0 inventory, the amendment §5 fixes (`e87f6f4`), SRS v2 Draft 1 and the literature review. If PR #1 is still open, branch from it and ask the PI to merge it first; if merged, branch from `main`. Check with `gh api repos/ProfJJK-CoChem/CoChem-SpycFit/pulls/1 --jq '.state,.merged'` (`gh pr` uses GraphQL, which cloud sessions block). |
| `main` | `488dc87`: consolidated SRS, amendment and `docs/HANDOFF.md`, published on 2026-10-16 from the PI's session with another agent. |
| Requirements baseline | `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md` and `docs/wiki/CoChem-SpycFit_SRS_2026-10-16_Amendment.md` at `488dc87`, with the PI's 2026-10-16 decisions in `docs/HANDOFF.md` and amendment §7. The PI asked agents on 2026-10-16 to continue from this work. |
| Proposed revision | `docs/srs/CoChem-SpycFit_SRS_v2.md` Draft 1: every SF id kept, split into single-SHALL requirements, with literature-backed additions. Nothing in it is approved. |
| Architecture and plan | Proposals: `docs/architecture/CoChem-SpycFit_Architecture.md`, `docs/plan/CoChem-SpycFit_Plan.md`. |
| PI decisions | `docs/handoff/PI_DECISIONS.md`. The 2026-10-16 direction bears on D-SPYC-01, 03, 04, 06, 14 and 18 (annotated there). No entry is DECIDED; the PI confirms each by id. |
| Code | `e87f6f4` corrects the seven amendment §5 defects and four more found in Phase 0 (`docs/evidence/phase0_inventory.md` §6), at the PI's 2026-10-16 request. No feature was added. |
| Tests | Module by module, with the genuine SPFIT/SPCAT build configured: 1,097 passed, 78 failed, 2 collection errors, 1 skipped of 1,178. The remaining failures are classified in the inventory §4 (54 come from gitignored test data). `pytest.ini` still runs one file, which fails to import. |
| CI | `.github/workflows/cochem_spycfit_ci.yml` runs one `find` for data files and no tests. Actions jobs on this account fail with no runner assigned (an account-level state the PI must check). Local runs are not CI. |

## 2. Read first, in this order

1. **This file**, then `AGENTS.md`.
2. `docs/HANDOFF.md`: the PI's settled decisions of 2026-10-16. Do not reopen them without new
   evidence or a request from the PI.
3. The baseline SRS and amendment (§1 table).
4. `docs/srs/CoChem-SpycFit_SRS_v2.md` Draft 1 and
   `docs/wiki/review/CoChem-SpycFit_SRS_2026-10-16_Literature_Review.md`, which records the
   evidence behind each added requirement.
5. `docs/evidence/phase0_inventory.md`: what the suite does, what was fixed, what remains.
6. `docs/handoff/PI_DECISIONS.md` and `docs/handoff/REVIEW_ERRATA.md`.
7. `docs/wiki/review/CoChem-SpycFit_SRS_Review.md` (the 2026-10-11 review of the old SRS). Its
   scope recommendations are superseded by the PI's 2026-10-16 decisions; its file-level findings
   remain useful.
8. Governing ecosystem documents (sibling repositories, `ProfJJK-CoChem/*`, read-only):
   - `CoChem-BASE/Method_Matrix.md` v4: system class, accuracy, observables, spin statistics,
     validation set, deliverables. TORQ keeps a copy whose line numbers are BASE's plus 2 (CRLF).
   - `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`: accepted corrections;
     §3.1-§3.4 and §4 bind SpycFit.
   - `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`: §1.2 ownership ("SpycFit owns
     spectral assignment and fitting"), §9.1 TORQ-SPEC-001, §12 exports, §15.2 V-CONSUMER, §16.
   - `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md`: D08, D10, D11, D12, ARCH-12.
   - `CoChem-TORQ/docs/development/pickett_backend.md`, `units_and_constants.md`.
   - `CoChem-BASE/src/cochem_base/interfaces/`: `module_registry.py`, `artifact_handoff.py`
     (`cochem.module-handoff/1` accepts only XYZ, Hessian and calculation-result JSON today) and
     the three `integration_boundary` stubs BASE declares for SpycFit.
9. Historical material: `docs/archive/`, `docs/legacy/README.md`.

Not requirements: `docs/archive/CoChem-SpycFit_Comprehensive_SRS_7e42507.md`,
`.docs/task1_subsystems_architectural_specification.md`, `README.md` feature claims,
`swarm_state.json`, the `*_payload.md` files, code docstrings, test names, and TORQ's legacy
`Libraries/cochem_torq_export.py` payload format.

## 3. What is established

**Scope (PI, 2026-10-16).** SpycFit is a computational-chemistry-to-assignment, fitting and
publication pipeline: rotational assignment and fitting, joint rotational-rovibrational fitting,
MLFF-assisted assignment, and vibrational band, line and envelope fitting. IR, Raman, THz,
near-IR and microwave are first-class. The expert decides every assignment. Native SpycFit and
genuine SPCAT/SPFIT run side by side for compatible microwave models; divergence stays visible
and needs expert review before export. The core is offline-capable; physical air-gap is not
required. Semirigid comparison comes first. The Method Matrix system class (5-10 atom complexes,
CP-FTMW at about 2-22 GHz and 1-2 K) is one important use within this scope, not its boundary.

**Open boundaries.** Which result a publication table reports first when native and SPFIT
results both exist; where TORQ's predicted catalogs end and SpycFit's re-prediction begins
(D-SPYC-16); which native Hamiltonian terms come first (D-SPYC-18); who produces vibrational and
Raman predictions (TORQ or another provider); the SPFIT/SPCAT pin (D-SPYC-17).

**Ecosystem roles.** TOPOS finds conformers. TORQ runs quantum chemistry, exports predicted
constants and produces predicted catalogs; its HDF5 bundle exporter (TORQ-EXP-004) does not exist
yet, and its legacy `tar.zst` payload is not a contract. BASE hosts the UI, discovers modules and
owns task state. The old SRS's "Genesis tier", "TORQ message broker", "Host Warden", "data diode"
and "Kanban job board" exist in no CoChem repository.

**SPFIT/SPCAT.**
- Two candidate trees: TORQ pins laserkelvin/Pickett `f5af2ff` (gcc 14.2; "no license granting
  source or binary redistribution"); the JPL sources at https://spec.jpl.nasa.gov/ftp/pub/calpgm/
  were built here (hashes in `docs/handoff/calpgm_sources_2026-10-11.sha256`). TORQ's `calcat.c`
  computes partition functions at 1,001 temperatures and its Makefile uses `-g -O3`; otherwise
  the trees match. One ecosystem pin is D-SPYC-17.
- The JPL HCO example is an exact regression for file handling: SPCAT on the shipped
  `hco.var`/`hco.int` reproduces `hco.cat` byte for byte, and SPFIT's first iteration on the
  shipped `hco.par`/`hco.lin` reproduces the last iteration of the shipped `hco.fit` exactly.
  HCO is an open-shell radical, so it is not semirigid evidence.
- The JPL NO example is input for DPFIT/DPCAT (the calpgm README: "the NO files are sample input
  for DPI"), not SPFIT/SPCAT. DPFIT runs it (Phase 0 inventory §7). The 2026-10-11 version of this
  handoff said NO "does not run"; that was a program mismatch.
- `tests/pickett_reference.py` locates the build and the hash-verified examples through
  `SPYCFIT_CALPGM_DIR`. The examples are not committed (no redistribution grant).
- Pickett conventions to respect: SPFIT reads `.par` and rewrites it; `.var` holds fitted errors
  and correlations; 200, 1100 and 2000 hold −Δ_J, −Δ_JK, −Δ_K; 40100 is −δ_J (A) or +d_1 (S);
  a zero exit status does not mean the run succeeded (SRS v2 §1.7).

**Constants and masses.** Use isotope-specific atomic masses with source and version (AME2020
proposed) and TORQ's CODATA 2022 profile; never nuclear masses (D-SPYC-02).

**Dependencies.** `requirements.txt` cannot be installed on Python 3.12 or later; `pyproject.toml`
omits scikit-learn, zstandard, networkx, PyQt6/PySide6, pydantic, scipy and pandas. Whether
SpycFit needs JAX is D-SPYC-11.

**Repository state.** Two package trees; tests import `src/`. About 21.5 kLOC of BASE/TORQ/TOPOS
code is vendored under `src/`. Committed junk includes `swarm_state.json`, `spoofing_matches.json`,
`qe.txt`, `Legacy_Verification.zip`, `spectral_output/` and `__pycache__` files. Salvage is a
guided rewrite, porting modules only after they pass requirement-linked tests against external
references (D-SPYC-15).

## 4. How to work with the PI

The PI is the only authority on scope, conventions, tolerances and acceptance.

- Put questions as numbered options with a recommendation and its consequence, as
  `PI_DECISIONS.md` does.
- Record an answer by pasting the PI's verbatim words, the date and the source under the entry
  and setting `ANSWERED`; write `docs/decisions/D-SPYC-nn.md`. It becomes `DECIDED` only when the
  PI approves the PR containing the record, in a PR comment (`AGENTS.md`).
- Never mark a decision, requirement, tolerance or gate approved yourself.
- Report numbers, not verdicts.
- Cross-repository problems go to the PI.
- SRS v2 Draft 1 was written whole because the PI asked for the existing SRS to be checked and
  improved. Earlier guidance asked for one chapter per PR; review and approval can still go
  chapter by chapter.

## 5. Sequence

| Phase | State | Exit gate |
| :--- | :--- | :--- |
| 0. Baseline | Inventory done (`docs/evidence/phase0_inventory.md`). The PI's tag `legacy-swarm-2026-10` on `7e42507` is not yet created. | **G0**: the PI approves the plan's phases and gates and confirms D-SPYC-01 by id. |
| PI-requested fixes | Amendment §5 defects corrected in `e87f6f4` at the PI's 2026-10-16 request, before the PI removed the `srs-v2.0` code gate on 2026-10-18 (`docs/decisions/D-SPYC-25.md`). | The PI approves PR #1 in a PR comment. |
| 1. Specification | SRS v2 Draft 1 written; architecture and plan to be aligned with it. | **G1**: the PI tags `srs-v2.0`. Code PRs do not wait for the tag (D-SPYC-25). |
| 2. Hygiene | Not started. Inventory §7 lists the inputs. | **G2** |
| 3. Work packages | Not started. | **G3** per package |
| 4. Release | Not started. | **G4** |

## 6. Next session

1. Confirm PR #1's state. Read any PI comments on it and act on them.
2. Ask the PI to confirm by id each 2026-10-16 direction listed in `PI_DECISIONS.md` and SRS v2
   §11, and to answer SRS v2 §11 questions 2-6.
3. Review SRS v2 Draft 1 with the PI chapter by chapter. Align the architecture (component names
   `compare` and `bench` are new) and the plan (new WP-CMP, WP-BENCH, WP-VIB) with it.
4. Code PRs may open before `srs-v2.0` (the PI removed the code gate on 2026-10-18,
   `docs/decisions/D-SPYC-25.md`); each cites the proposed SRS v2 ids it implements and says that
   they are proposed. Candidates from the inventory §7, smallest first:
   - the bridge's `parse_pickett_fit` and its test still use the invented `.fit` layout;
   - the export's legacy `.var` is not a Pickett `.var`;
   - a dictionary-to-`.par`/`.int` builder with explicit option lines (needs SF-PICK-001 and
     V-PICK-03 approved).
5. Provide the genuine engine for tests by building the JPL sources into a directory and setting
   `SPYCFIT_CALPGM_DIR` (see `tests/pickett_reference.py`).

## 7. Inputs only the PI can supply

Ask early; work that depends on them waits, and nothing is invented in their place.

- The external prompt files that drove the old code (`Perfected_SRS_Chunk_NN_prompts.md`) that
  define the 40 `REQ-<AREA>-NNN` ids the tests cite.
- Measured spectra or peak lists with their export format, frequency calibration and line widths.
- The benchmark panel and tolerances (D-SPYC-14; SRS v2 §8.3), and confirmation of the reference
  sources' use terms (SF-DATA-001d).
- The SPFIT/SPCAT pin and use rights (D-SPYC-17), and which second engines (PGOPHER, XIAM, BELGI,
  RAM36) are wanted.
- Who maintains TORQ's exporter and BASE's handoff schema.

## 8. Known traps

- **Reusing old text.** The archived SRS looks authoritative and is wrong where it matters.
- **Fallbacks and stand-ins.** The swarm's last pre-SRS commit was titled "enforce rigorous ASE
  physical fallbacks for ORCA and SPFIT execution". Genuine-engine tests fail when the engine is
  missing; they never skip or substitute.
- **Invented fixtures.** Several test inputs called "authentic" were invented or disagree with
  their claimed source (inventory §5). Check every value against its source.
- **Pickett.** SPFIT reads `.par` and rewrites it; run on scratch copies. A zero exit status is
  not success (`Bad Line`, lines not used, empty `.cat`, `Fit Diverging`). JPL's `spins.nam`
  labels for 40100/50000 contradict the program. Programs are matched to their examples (HCO:
  SPFIT/SPCAT; NO: DPFIT/DPCAT).
- **Equilibrium vs ground-state constants.** ΔB_vib = B_0 − B_e (TORQ contract §3.1).
- **Uncertainty bands.** Method Matrix §3.1 numbers are literature statistics, not calibrated
  prediction intervals (TORQ contract §3.2).
- **Temperature.** Jet spectra are at T_rot ≈ 1-2 K; the notebook and UI default to 296 K.
  Catalog intensities are at 300 K and must be recomputed at the declared temperature.
- **Representation and reduction.** Never auto-switch them; never infer the reduction from the
  option-line character or `.nam` labels.
- **Spin statistics and hyperfine.** User inputs with recorded justification; many in-class
  systems carry quadrupole structure.
- **Process groups.** Never signal a process group that may include the caller.
- **Agent narration in deliverables.** No first-person text, audit verdicts or "awaiting review"
  lines in `docs/`, `src/` or `tests/`.
- **CI that cannot run.** Tell the PI once; never present local runs as CI.

## 9. Definition of done for any package

TORQ SRS §16: "Every work package must produce executable implementation, tests tied to
requirements, migration notes for affected persisted data, and a measured evidence report. A
checklist, source stub, hardcoded result, synthetic engine output or claimed agent audit cannot
close a package."

For SpycFit, at least: every SHALL in the package has a passing, non-skipped test naming its
requirement id; scientific packages have a test against a published value or the pinned
SPFIT/SPCAT output with a tolerance the PI approved before the implementation PR; the whole suite
runs in CI; `docs/evidence/WP-<NAME>.md` records what ran and where; the PI approves the PR in a
PR comment.
