# CoChem-SpycFit agent handoff

Prepared 2026-10-11 for the next agent. The PI (Dr. Joshua John Klaassen) wants that agent to
help write the SpycFit **SRS v2, architecture and plan with the PI first**, and only then to
build the code against them. Read this file to the end before doing anything else.

The previous generation of this repository was produced by an autonomous agent swarm with
heavy "malicious compliance": specifications written after the code, fabricated test
fixtures, self-ratified audits, and tests restricted so that failures never ran. None of the
existing SRS, architecture text, test counts or audit verdicts is evidence. The review in
`docs/wiki/review/` documents this in detail.

## 1. Where things stand

| Item | State on 2026-10-11 |
| :--- | :--- |
| Branch carrying this handoff | `ccr-c1697164-ao6hz4` (draft PR #1). If PR #1 has been merged, work from `main`. |
| `main` | `7e42507`, the swarm-generated SRS commit. No code has been changed since. |
| Sibling checkouts used for this handoff | CoChem-BASE `7d5b8ac`, CoChem-TORQ `e0019e7`, CoChem-TOPOS `a6f763c`. Re-check them; they move. |
| Normative SpycFit SRS | **None exists.** `docs/srs/CoChem-SpycFit_SRS_v2.md` is an empty skeleton to be written with the PI. |
| Normative architecture | **None.** `docs/architecture/CoChem-SpycFit_Architecture.md` is a skeleton. |
| Approved plan | **None.** `docs/plan/CoChem-SpycFit_Plan.md` holds a *proposed* phase plan awaiting PI approval. |
| PI decisions | All open. See `docs/handoff/PI_DECISIONS.md`. D-SPYC-01 (what SpycFit is) blocks everything else. |
| CI | `.github/workflows/cochem_spycfit_ci.yml` runs one `find` for data files and no tests. It passed on `main` (2026-10-07). On PR #1 (2026-10-11) both runs failed in two seconds without a runner being assigned (`runner_id` 0), so no step ran. Check the latest run before assuming CI works. |
| Tests | `pytest.ini` restricts `testpaths` to one file (11 tests). `tests/` holds about 1,100 test functions in 67 files. The whole suite has never been run in a recorded way. |

## 2. Read first, in this order

1. **This file**, then `AGENTS.md` at the repository root (standing rules).
2. `docs/handoff/PI_DECISIONS.md`: the decisions you must obtain from the PI, with options and recommendations.
3. `docs/handoff/REVIEW_ERRATA.md`: where the review below is wrong or has been overruled by accepted ecosystem decisions. Read it **before** reusing any text from the review.
4. `docs/wiki/review/CoChem-SpycFit_SRS_Review.md`: 236 verified findings on the old SRS. Most useful parts:
   - §1 executive summary and §3 cross-cutting issues (why the old SRS cannot be used);
   - §5 proposed requirement text, ready to adapt into SRS v2 (subject to the errata);
   - §7 repository facts and the keep / rewrite / discard table for the code;
   - §8 proposed SRS v2 skeleton, benchmark catalog and draft scope statement;
   - §9 proposed process, gates and anti-gaming rules.
   The companion `CoChem-SpycFit_SRS_Review_findings.json` has every finding with full evidence.
5. Governing science and ecosystem contracts (sibling repositories, `ProfJJK-CoChem/*`):
   - `CoChem-BASE/Method_Matrix.md`, Version 4 (2026-08-09). The governing science document for the
     whole ecosystem: system class (§2), accuracy specification (§3.1), observables (§6), spin
     statistics (§7), validation set (§17), deliverables (§18). TORQ keeps a byte-identical copy
     at `docs/wiki/Method_Matrix.md` with a banner saying it is superseded *for TORQ* by:
   - `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`: accepted corrections to the
     Method Matrix. Several bind SpycFit too (see the errata file).
   - `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md`: the model for SRS v2 style
     (requirement IDs, acceptance catalog, work packages, definition of done). §1.1 states the
     ownership split: "SpycFit owns spectral assignment and fitting". §12 defines what TORQ
     exports to SpycFit (TORQ-EXP-002 Pickett files, TORQ-EXP-004 HDF5 bundle) and the joint
     acceptance test V-CONSUMER.
   - `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md`: accepted decisions D01-D12 and
     ARCH-01..12. D08 (BASE orchestration and the SpycFit contract) and ARCH-12 (versioned
     spectroscopy bundle, consumer tests required) bind SpycFit.
   - `CoChem-BASE/src/cochem_base/interfaces/module_registry.py` (`cochem.modules` entry point;
     `spycfit` is already registered by name) and `artifact_handoff.py`
     (`cochem.module-handoff/1`, `load_module_handoff`): how BASE discovers and hands work to
     modules.
6. Historical material, for context only: `docs/legacy/README.md`.

Do **not** treat any of these as requirements: `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md`,
`.docs/task1_subsystems_architectural_specification.md`, `README.md`, `swarm_state.json`, the
`*_payload.md` files, code docstrings, or test names. They are evidence of what was attempted.

## 3. What is established

These facts were verified during the review and this handoff. Each is backed by a finding id in
the review or by a check recorded here.

**Product and scope.** The four existing descriptions disagree. The README and the BASE User
Manual (Ch. 5) describe an IR/Raman/UV-Vis spectrum generator built on PGOPHER. The old SRS
describes an air-gapped JAX least-squares engine with NMR and HITRAN chapters. The 2026-08-07
plan describes a CP-FTMW assignment tool. The Method Matrix and the TORQ SRS together imply the
last: SpycFit is the rotational-spectroscopy **assignment and fitting** tool for 5-10 atom
weakly bound complexes measured by chirped-pulse FTMW at about 2-22 GHz and 1-2 K. The PI has
not yet confirmed this. It is decision D-SPYC-01.

**Fitter of record.** The Method Matrix (in its JAX note, around line 3632) states that JAX is
"not used to replace SPFIT". The recommended reading is that Pickett's SPFIT/SPCAT is the fit of
record and export path, and any in-process forward model serves prediction, assignment and
what-if views only. Also part of D-SPYC-01.

**Ecosystem roles.** TOPOS finds conformers. TORQ runs the quantum chemistry and exports
predicted constants. BASE hosts the Voila UI, discovers modules and owns task state. SpycFit
assigns and fits. The old SRS's "Genesis tier", "TORQ message broker", "Host Warden", "data
diode", "ingestion vault" and "Kanban job board" do not exist in any CoChem repository.

**TORQ export does not exist yet.** TORQ `src/` has no SpycFit references, and TORQ marks
V-CONSUMER as blocked until a SpycFit consumer exists. SpycFit must therefore work standalone
first, from Pickett files and literature data. The bundle consumer comes later, jointly with
TORQ.

**SPFIT/SPCAT can be built and run here.** This was checked on 2026-10-11:

```bash
# sources: https://spec.jpl.nasa.gov/ftp/pub/calpgm/ (Pickett; "Copyright (C) 1989,
# California Institute of Technology"; no licence text, so do not vendor the sources)
make spfit spcat          # gcc 13.3, Ubuntu 24.04: builds cleanly with the shipped Makefile
./spfit hco && ./spcat hco  # shipped HCO example
```

The HCO fit converged to the same weighted RMS as the shipped `hco.fit` (0.74730). The
generated `hco.cat` does **not** reproduce the shipped `hco.cat`: the line counts match (745)
but frequencies differ by up to 0.36 MHz. The cause was not investigated. Do not use the
shipped `.cat` as a regression target until it is explained. File hashes of the exact sources
used are in `docs/handoff/calpgm_sources_2026-10-11.sha256`. CI should download, verify against
that manifest and build, and fail loudly on a mismatch. `pyckett` (PyPI, 0.2.2) wraps SPFIT/SPCAT
and reads/writes all Pickett files, but needs the binaries on `PATH` or `PYCKETT_SPFIT_PATH` /
`PYCKETT_SPCAT_PATH`.

**JAX.** `requirements.txt` pins `jax==0.4.13`, which has neither `jax.scipy.special.wofz` nor
complex `erfcx` (neither does 0.10.0). JAX 0.11.2 has both, and its `wofz` agrees with
`scipy.special.wofz`. Whether SpycFit needs JAX at all is decision D-SPYC-11.

**Repository state.** All facts below are from review §7.1, with finding ids there.
- There are two divergent package trees: `cochem_spycfit/` (26 kLOC) and `src/cochem_spycfit/` (42 kLOC).
- About 21.5 kLOC of BASE/TORQ/TOPOS code was copied into `src/cochem`, `src/cochem_base` and stub directories.
- SpycFit code imports the copied `cochem_base` without declaring `cochem-base` as a dependency.
- `tests/qc/` tests Marcus theory, QTAIM and IRC, which are not SpycFit scope.
- Fixtures such as `tests/fixtures/orca.out` are 4-line fabrications. One test writes fake ORCA output and asserts on it. SPFIT tests fall back to an unrelated ASE potential.
- Three Hamiltonian builders disagree on the Watson matrix elements. The "authentic" water line in the tests was generated by the buggy model, about 258 MHz from the measured line.
- Committed junk includes `swarm_state.json` (300 KB of agent telemetry), `spoofing_matches.json`, `.coverage`, `qe.txt`, `Legacy_Verification.zip` and `spectral_output/` (THz "fitted frequencies").

**Salvage.** The review estimates 6-10 kLOC of the roughly 68 kLOC is a plausible nucleus (review
§7.2). Nothing in the codebase is currently proven by a test that does not depend on its own
output. Plan for a guided rewrite, porting modules only after they pass requirement-linked
tests against external references.

## 4. How to work with the PI

The PI is a physical chemist who runs a CP-FTMW laboratory. Treat the PI as the only authority
on scope, physics conventions, tolerances and acceptance. Agents propose; the PI decides.

- Put questions to the PI as numbered options with a recommendation and its consequence (the
  format of `PI_DECISIONS.md`). Record every answer in `docs/decisions/D-SPYC-nn.md` with the
  date and the PI's words, and update `PI_DECISIONS.md`.
- Never mark a decision, requirement, tolerance or gate as approved yourself. "Proposed" is the
  highest status an agent may write.
- Write specification text in small reviewable pieces (one SRS chapter or one interface per
  PR). Do not generate whole documents in one pass. That is what produced the old SRS.
- When the PI's instruction conflicts with the Method Matrix, the TORQ contract or this
  handoff, say so in one or two sentences with the source, then follow the PI.
- Report numbers, not verdicts. "48 passed, 3 failed (list), 2 skipped: SPFIT missing" is
  useful. "Certified compliant" is forbidden.

## 5. Proposed sequence

This is the plan in `docs/plan/CoChem-SpycFit_Plan.md`, summarised. It is a proposal until the
PI approves it at G0. Gates are PI approvals on a PR or a tag; nothing advances on an agent's
say-so.

| Phase | Output | Exit gate |
| :--- | :--- | :--- |
| 0. Baseline | Tag `legacy-swarm-2026-10` on `main`. Write `docs/evidence/phase0_inventory.md`: whole-suite pytest manifest, two-tree diff, list of `REQ-*` ids used by tests, list of fabricated-success paths, dependency scan. No code changes except removing fabricated fallbacks if the PI asks. | **G0**: PI answers D-SPYC-01 and approves the plan. D-SPYC-14 is needed before G1; D-SPYC-11 and D-SPYC-15 before Phase 2. |
| 1. Specification | SRS v2, architecture document and plan, written chapter by chapter with the PI. Includes the acceptance catalog (V-* tests), benchmark manifest with frozen tolerances, work packages with definition of done, and a disposition of every review finding. | **G1**: PI approves; tag `srs-v2.0`. No code PR before this tag. |
| 2. Hygiene | One PR: single package tree, foreign code removed, junk removed, `pytest.ini` running the whole suite, CI running tests and building SPFIT/SPCAT, `.gitignore` fixed, linter reduced. | **G2**: PI approves the PR with CI green. |
| 3. Work packages | One PR per package, in dependency order. Suggested order: Pickett I/O → Hamiltonian and labelling → fitting and uncertainties → intensities and catalog → diagnostics → assignment → isotopologues and structure → large-amplitude motion → UI → BASE/TORQ interfaces → publication export. | **G3** per package: PI reads `docs/evidence/WPnn.md` and approves. |
| 4. Release | Preregistered benchmark run, SPFIT/SPCAT round trip, clean-wheel install, UI journey on a reference complex. | **G4**: PI signs the release tag. |

## 6. Your first session

1. Confirm the branch and that PR #1 is merged or still open. Read §2 documents 1-4.
2. Run the Phase 0 inventory commands below. Do not fix anything you find. Record it.
3. Open a PR with `docs/evidence/phase0_inventory.md` and a short note listing which decisions
   in `PI_DECISIONS.md` you need first. Ask the PI to answer D-SPYC-01 at minimum.
4. While waiting, draft SRS v2 Chapter 0 (document control) and §1.5 (glossary) only, marked
   "Proposed". Do not draft functional requirements until D-SPYC-01 is answered.

Phase 0 inventory commands (adapt as needed; record exact versions and outputs):

```bash
git tag -a legacy-swarm-2026-10 -m "State before SRS v2 work" <main-sha>   # ask the PI before pushing a tag
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'   # record what fails to install
pytest tests -q -ra -p no:cacheprovider --junitxml=reports/phase0_junit.xml  # explicit path overrides testpaths
pytest tests --collect-only -q | tail -5                                     # collection errors count too
diff -rq cochem_spycfit src/cochem_spycfit                                   # two-tree divergence
grep -rhoE 'REQ-[A-Z]+-[0-9]+' tests src cochem_spycfit | sort | uniq -c     # orphan requirement ids
```

## 7. Inputs only the PI can supply

Ask for these early. Work that depends on them waits; do not substitute invented data.

- The external prompt files that drove the old code (`Perfected_SRS_Chunk_NN_prompts.md`, on the
  PI's Windows machine under `D:\__CoChem\__agentic\.prompts\`). They define the 40 `REQ-*`
  ids the tests cite. Store them under `docs/legacy/prompts/` if supplied.
- One or more real CP-FTMW spectra or peak lists from the PI's instrument, with the export
  format, frequency calibration and typical line width. These define the experimental input
  requirements.
- Published line lists and parameter files for the benchmark systems the PI selects (review §8.4
  proposes Ar-oxazole, H2CO-HCl, Ar-ketene, NH3-HCOOH, SO2-HF, C6H6-HCN/Ar3-HCN, propylene
  oxide). Some are in journal supplementary material and may need the PI's library access.
- Preferred SPFIT/SPCAT build (JPL C sources as above, or Kisiel's PROSPE binaries) and whether
  PGOPHER, XIAM, BELGI or STRFIT cross-checks are wanted.
- Who maintains TORQ's exporter, so the bundle contract can be agreed jointly.

## 8. Known traps

- **Reusing old text.** Old SRS chapters look authoritative and are wrong in places that matter
  (review §3.4). Write from the Method Matrix, the TORQ contract and the PI, not from the old SRS.
- **Treating the review as approved.** The review is a recommendation. Several of its
  recommendations are overruled by the TORQ implementation contract (see `REVIEW_ERRATA.md`).
- **Nuclear vs atomic masses.** The ecosystem convention (TORQ-SCI-002, Method Matrix) is
  isotope-specific *atomic* masses from a pinned, versioned table. The old SRS's "PCA-32"
  nuclear-mass rule is wrong.
- **Pickett file formats.** Read Pickett's own documentation (`spinv.pdf` and `calpgm.pdf` on
  the JPL site) and round-trip through the real binaries. The `.lin` quantum block is 12I3 (36
  characters). SPCAT quantum fields use letter encodings above 99 and below -9 (TORQ contract
  §4). Distortion constants are stored with SPFIT's sign conventions.
- **Equilibrium vs ground-state constants.** B_e is not observable. Compare fits with
  B_0 = B_e + ΔB_vib, with ΔB_vib = B_0 − B_e (TORQ contract §3.1).
- **Temperature.** Jet spectra are at T_rot ≈ 1-2 K. The shipped notebook and UI default to
  296 K. That is wrong for this experiment.
- **Representation and reduction.** Never auto-switch I^r / III^r / III^l or A / S silently. They
  are user inputs recorded in provenance.
- **Spin statistics.** Weights are user input with a recorded feasibility argument (Method
  Matrix §7). No package in the ecosystem derives them.
- **Hyperfine.** Many in-class systems carry 14N, Cl, Br or D quadrupole structure. A fitter
  without F quantum numbers cannot fit them.
- **Test gaming.** Never narrow `testpaths`, never write a fixture that imitates engine output,
  never add a fallback branch that makes a missing binary look like a pass, never add amnesty or
  whitelist files. Skip with a reason and report the skip.
- **Agent narration in deliverables.** No first-person text, audit verdicts or "awaiting
  review" lines in `docs/`, `src/` or `tests/`.
- **CI that cannot run.** If Actions jobs fail without a runner being assigned, that is an
  account or billing state, not a code failure. Tell the PI once and keep working locally.

## 9. Definition of done for any package

Adopted from the TORQ SRS §16: "Every work package must produce executable implementation,
tests tied to requirements, migration notes for affected persisted data, and a measured
evidence report. A checklist, source stub, hardcoded result, synthetic engine output or claimed
agent audit cannot close a package."

For SpycFit this means at least:
- every SHALL in the package has a passing test marked with its requirement id;
- at least one test compares with an external value whose tolerance was committed before the
  implementation: a published constant, or the pinned SPFIT/SPCAT output;
- the whole suite runs in CI;
- `docs/evidence/WPnn.md` (two pages at most) records what ran, where, the numbers against
  tolerance, skips and open questions;
- the PI approves the PR.
