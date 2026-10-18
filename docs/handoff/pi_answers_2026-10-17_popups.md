# PI answers of 2026-10-17 given through pop-up questions

The PI asked on 2026-10-17: "Provide me the unanswered questions as answer pop-ups so that with
suggestions, pros, cons, context for the suggestion and risks for the suggestion. One prompted at
a time so that I can answer them properly as there are no useful tools for this on the mobile."

Each pop-up named the item, gave context, and offered options with the agent's recommendation
first and marked "(Recommended)". The PI selected an option or typed a reply under "Other". The
selected option label or typed text is quoted verbatim below. Every option label the PI did not
select is listed as not chosen. These answers are the source of the decision records that cite
this file; the records are DECIDED only when the PI approves the PR that contains them.

## D-SPYC-14 follow-ups (nine pop-ups)

1. **T1 limit.** Options: "51 Hz (.cat) / 6 Hz (.fit) (Recommended)"; "50 Hz (.cat) / 5 Hz
   (.fit), as the summary read"; "Compare against .fit only (6 Hz), not .cat".
   Selected: **"51 Hz (.cat) / 6 Hz (.fit) (Recommended)"**.
2. **Does "equal or closer" cover fits as well as predictions?** Options: "Yes: fits and
   predictions (Recommended)"; "Predictions only (SPCAT comparison)"; "Fits only (SPFIT
   comparison)". Selected: **"Yes: fits and predictions (Recommended)"**.
3. **How to judge "equal or closer" on held-out predictions.** Options: "Paired bootstrap test,
   5% level (Recommended)" (resampling by line family); "Fixed margin you set (e.g. SpycFit
   within 1 print quantum of SPCAT per line)"; "Strict: never farther on any line"; "Report only,
   no pass/fail rule". Selected: **"Paired bootstrap test, 5% level (Recommended)"**.
4. **What happens when SpycFit's native fit or prediction falls short.** Options: "Usable for
   exploration, labelled; no export as results until benchmarks pass (Recommended)"; "Strict
   gate: native fits hidden for the profile until benchmarks pass"; "No gate: report the
   comparison only". Selected: **"No gate: report the comparison only"**.
5. **Benchmarks without fit inputs (OCS).** Options: "Fit comparison NOT_EVALUABLE with reason;
   prediction comparison still reported (Recommended)"; "Drop OCS from the panel; replace it with
   a molecule that has fit inputs"; "Treat a missing fit comparison as a shortfall in the report".
   Selected: **"Drop OCS from the panel; replace it with a molecule that has fit inputs"**.
6. **When SpycFit and SPFIT use different models.** Options: "Compare and report only; the rule
   applies to same-model cases (Recommended)"; "Apply the rule to every case, whatever the
   model"; "Refuse different-model comparisons entirely". Selected: **"Compare and report only;
   the rule applies to same-model cases (Recommended)"**.
7. **Expert exclusion of lines below 5u for a recorded physical reason.** Options: "Yes, with a
   recorded reason (Recommended)"; "No: exclusion only above 5u"; "Allowed, but the excluded line
   stays in the fit at 10× its uncertainty instead of being removed". Selected: **"Yes, with a
   recorded reason (Recommended)"**.
8. **Should over-coverage be bounded too?** Options: "Yes: bound both under- and over-coverage
   (Recommended)"; "No: floors only"; "Report over-coverage, do not bound it". Selected: **"Yes:
   bound both under- and over-coverage (Recommended)"**.
9. **Remaining full-brief detail.** The pop-up listed: T1 infrared and high-frequency print
   quanta (`.cat` 500 Hz above 99 999 999 MHz; 5×10⁻⁷ cm⁻¹ in IR mode; `.fit` 5×10⁻⁸ cm⁻¹; `.egy`
   5×10⁻⁷ cm⁻¹), transitions matched by state identity; T2 stopping rules (SPFIT rerun with a
   larger NITR if it stopped at its iteration limit; the native fit stops when σ changes by less
   than 10⁻⁷ relative and max |Δp|/u < 10⁻³; rms(Δν/u) reported beside Δp/u); T3 extra defaults
   when a line has no uncertainty (4 kHz cavity with unresolved hyperfine, 25 kHz Ka-band
   CP-FTMW, 10⁻³ cm⁻¹ wavemeter IR; defaults Type B, k = 1, ranking after source and user values,
   flagging every result that uses them; per-data-set statistic with leverage-corrected degrees of
   freedom); T4 (n < 30 held-out lines gives INCOMPLETE; floors simulated under the joint
   predictive covariance; extrapolated lines reported only). Options: "Confirm all as listed
   (Recommended)"; "Confirm all except the T3 extra defaults (those stay TBD)"; "Confirm nothing
   yet; I will read the record first". Selected: **"Confirm all except the T3 extra defaults
   (those stay TBD)"**.

## D-SPYC-20 detail confirmation

The pop-up listed: the P13/P14 row (pre-fills from the row of the data being fitted); the
uniform-flow entry in the mm-wave row (a flow runs at its own measured temperature, 9.3–30.2 K in
the sources; jet default 7 K); the NIR gas-cell optional intensity cutoff of 1×10⁻²⁵ cm/molecule at
296 K, off by default; published ranges beside each default and the key DOIs; the line-width
definitions (condensed IR/Raman/NIR 10 cm⁻¹ is a Lorentzian HWHM; THz-TDS 5 cm⁻¹ and band contour
1 cm⁻¹ are FWHM). Options: "Confirm all as listed (Recommended)"; "Confirm all, but state every
width as FWHM (condensed IR/Raman/NIR becomes 20 cm⁻¹)"; "Confirm all except the P13/P14 row
(computation-led profiles get no pre-fill)"; "Confirm nothing yet; I will read the record first".
Selected: **"Confirm all as listed (Recommended)"**.

## Module name

Options offered: "SPAFT — SPectral Assignment and FiTting (Recommended)"; "SPYAF — SPectra in
pYthon: Assign and Fit"; "SPEFY — SPEctral Fitting in pYthon ('SPEF-ee')"; "SPEKT — clipped
'Spektrum'"; with SPECQ, SPYFT, ROFIT and SPEFT named as typeable alternatives.
The PI typed: **"Derivative of SPEFY but instead spelled SPYFY, pronounced Spiffy"**.

The agent checked SPYFY the same day (PyPI JSON API and simple index 404 for `spyfy` and `SPYFY`;
no exact or look-alike match in the 915,698-project PyPI index of 2026-10-17; one-letter
near-names pyfy, py-fy, spfy, spycy, spydy, spyfi, spyky; `spiffy` is an unrelated PyPI package;
GitHub search shows four small repositories named Spyfy or spyfy, none scientific; Spy-Fy
(spy-fy.com) sells iPhone camera covers and "Hidden Camera Detector - SpyFy" is an iOS app;
trademark registries not searched) and asked again. Options: "Confirm SPYFY (Recommended)";
"Use SPAFT instead"; "Hold the name; keep SpycFit for now and decide later".
Selected: **"Confirm SPYFY (Recommended)"**.

## Search windows for computed constants (proposed D-SPYC-21)

Options: "Per-method-class defaults, q = 2×RMSD (Recommended)"; "Per-method-class defaults,
q = 1×RMSD (narrower)"; "Method Matrix §3.1 bands as a flat default (errata item 4 as written)";
"No default: the user or TORQ must supply q before a window is drawn".
Selected: **"Per-method-class defaults, q = 2×RMSD (Recommended)"**.

## D-SPYC-16 catalog boundary

Options: "D-SPYC-16: D — split by producer (Recommended)"; "D-SPYC-16: E — as D, plus one shared
runner and receipt (in BASE, or TORQ's backend extended)"; "D-SPYC-16: A — split by origin (TORQ
from computed, SpycFit from fitted or edited constants)"; "D-SPYC-16: B — SpycFit makes all
catalogs; TORQ exports constants only".
Selected: **"D-SPYC-16: D — split by producer (Recommended)"**.

## D-SPYC-02 masses, abundances and constants

Part 1 options: "D-SPYC-02: D — mendeleev pinned exactly, read directly (Recommended)";
"D-SPYC-02: A — call TORQ's units profile and isotope_record at run time"; "D-SPYC-02: B — a
SpycFit-owned snapshot with id and manifest, checked in CI against mendeleev and SciPy";
"D-SPYC-02: E — versioned reference data in BASE, or a new shared package".
Selected: **"D-SPYC-02: D — mendeleev pinned exactly, read directly (Recommended)"**.

Part 2, four sub-questions in one pop-up:
- Isotope default: **"(i) Most-abundant isotope as a named, overridable, recorded default
  (Recommended)"** (not chosen: "(ii) No default: the user labels every atom").
- Abundance for intensities: **"(c) User value per sample, with (a) mendeleev's value as the named
  default (Recommended)"** (not chosen: "(a) mendeleev's value only"; "(b) CIAAW 2024 interval
  midpoint, interval recorded"; "(c) with (b) as the default").
- Constants: **"(f) CODATA 2022 frozen in SpycFit's profile; SciPy as a cross-check
  (Recommended)"** (not chosen: "(p) SciPy pinned below the release that adopts CODATA 2026").
- Nuclear spins and moments: **"Separate decision (Recommended)"** (not chosen: "Include here,
  same source as masses (mendeleev)").

## D-SPYC-03 theory-feedback route

Options: "D-SPYC-03: A, schema (i), staged (Recommended)"; "D-SPYC-03: A2 — request in handoff
options beside a geometry artifact; BASE adds a reviewed operation"; "D-SPYC-03: D — request file
only; the user carries it"; "D-SPYC-03: C2 — a versioned BASE task-service API"; B typeable.
Selected: **"D-SPYC-03: A, schema (i), staged (Recommended)"**.

## D-SPYC-04 internal rotation (four questions in one pop-up)

1. Main option (A wrap / D SPCAT-reproducible native terms / B full native / C defer). The PI
   typed: **"The main value component of SPYFY is that it can utilize GPU to parallel, array,
   and make fitting and calculations dynamic recalculate on the fly. Honestly I don't see how we
   could be useful just copying SPFIT and SPCAT. I need more research and analysis for this on
   what would give the better end result and fit the vision for CoChem"**.
2. Engines named for P4 (multi-select): **"XIAM first (Recommended)"**, **"ERHAM second
   (Recommended)"**, **"BELGI-Cs/C1 and RAM36 as later"**, **"westerfit as later"** (all four
   selected).
3. XIAM reference build: **"XIAMALL.FOR build (the XIAM5 source)"** (not chosen: "XIAM 2.5e
   (XIAM-25E.TGZ), unchanged source (Recommended)"; "Posted XIAM5.EXE (Windows)").
4. BASE builds the P4 engines: **"Yes: XIAM (and ERHAM when its terms allow) in the BASE build
   mechanism (Recommended)"** (not chosen: "No: P4 engines are user-provisioned; SpycFit records
   UNAVAILABLE when absent").

## D-SPYC-05 user interface (three questions in one pop-up)

1. Stack (C Jupyter-widgets UI over the headless API with a measured widget comparison / A
   ipywidgets + Plotly under Voila / D desktop Qt; B typeable). The PI typed: **"What is the best
   and most user friendly option? Research and come up with a low friction and best usability
   option suggestions."**
2. Hosting inside BASE: **"H2: BASE's Voila renders a SpycFit notebook in a SpycFit-environment
   kernel; one port (Recommended)"** (not chosen: "H1: SpycFit's own Voila server, linked from
   BASE; a second forwarded port"; "H3: BASE writes a SpycFit tab over a SpycFit adapter, like
   its TORQ tab").
3. HPC UI journey in the first release: **"No: HPC is headless (API and CLI) in the first release
   (Recommended)"** (not chosen: "Yes: an HPC UI journey is a release requirement").

## D-SPYC-06 UV-Vis (three questions in one pop-up)

1. Scope (A exclude electronic transitions / B later profile P15 / C B plus condensed-phase
   electronic bands / D empirical envelopes). The PI typed: **"Retract my comment on UV-VIS we
   will indeed need to incorporate the full electronic treatment for those specific microwave use
   cases. We still will handle UV-VIS spectra predictions and fitting for another module but will
   incorporate the full electronic microwave spectroscopy predictions and fitting."**
2. O₂ near-IR electronic bands in HITRAN imports (excluded and labelled / excepted). The PI
   typed: **"Give me research and analysis on incorporating UV-VIS spectroscopy as well. The
   spectroscopy should be modularized and allow the user to go as far as they want with what
   predictions and evidence they have."**
3. Reserve an electronic-state field in the state identity (yes / no). The PI typed: **"We are
   going to pivot and include this field and scope."**

## D-SPYC-07 eigenvalue label swaps (four questions in one pop-up; brief `D-SPYC-07_2026-10-17.md`)

1. What the native model does on a swap (D flag and continue with provenance and an export review
   record / A halt and show / B warn and continue / C halt stored fits, previews continue).
   Selected: **"D-SPYC-07: D — flag every swap, continue, record it in provenance; export needs an
   expert-review record (Recommended)"**.
2. 07b, a post-run eigenvector check of finished SPFIT and DPFIT runs. Selected: **"On by default,
   user-overridable, recorded (Recommended)"** (not chosen: "Off by default, available on
   request"; "No such check").
3. 07c, the swap test and near-tie margin (options: "0.5 swap / 0.9 near-tie (Recommended)";
   "Defer the numbers until the stepping test has run"; "Other values"). The PI typed: **"2 but use
   1 as a provisional value that should be adjusted with further evidence. Allow these numbers to
   be changed by advanced users. Once further testing has been completed we will revise these
   defaults. Ensure a testing to be completed and published is flagged in a file. When CoChem is
   coded to a release state we should have documents for what cases need to be run for the
   publications to demonstrate CoChem modules."**
4. E, overlap-continuity labelling with proposed relabel. Selected: **"Not now: keep SPFIT-style
   energy-rank labelling as the native rule (Recommended)"**.

## D-SPYC-12 inertial-defect sign rule (four questions in one pop-up; answered 2026-10-18 UTC; brief `D-SPYC-12_2026-10-17.md`)

1. What SPYFY does on a sign difference (C flag with export review record / A diagnostic only /
   B binding rejection / D display only, comparison in GEOM). Selected: **"D-SPYC-12: C — typed,
   uncertainty-aware display and flag, never automatic rejection, plus an expert-review record
   before export of a flagged same-type sign difference (Recommended)"**.
2. An off-by-default |Δ| window in automated triples searches. Selected: **"Yes: off by default,
   expert-enabled, dropped triples logged (Recommended)"** (not chosen: "No window at all").
3. Same-type Δ agreement ranking candidates. Selected: **"Yes: rank only, never remove, reasons
   shown (Recommended)"** (not chosen: "No: display only, no ranking").
4. A labelled, recorded correction so that Δ_e compares with Δ₀, and the flag threshold k.
   Selected: **"Yes, recorded correction allowed; k = 2 (Recommended)"** (not chosen: "No
   correction: unlike-type pairs are displayed unflagged").

## D-SPYC-13 dark dipole components (four questions in one pop-up; answered 2026-10-18 UTC; brief `D-SPYC-13_2026-10-17.md`)

1. Treatment of lines with a small predicted dipole or transition-moment component (C recorded
   view with three label states and a relative intensity floor / A component label, three states /
   B exclude weak branches). Selected: **"D-SPYC-13: C — visibility as a recorded view: three label
   states plus an intensity floor relative to the strongest predicted line; labels sit beside the
   catalog and re-rank only (Recommended)"**.
2. Pruning labelled-weak lines from automated searches. Selected: **"Expert opt-in, recorded,
   rerunnable (Recommended)"** (not chosen: "Never prune"; "On by default").
3. The preset for the weak-component setting. Selected: **"Research a published detection value
   now, then pre-fill it"** (not chosen: "No preset until a published detection value is researched
   (Recommended)"; "0.1 D, the Method Matrix value").
4. Rank of lines whose label cannot be computed. Selected: **"Normal rank, shown 'not evaluated'
   with the reason (Recommended)"** (not chosen: "Lowest rank").

## SF-IF-004 BASE boundaries and the direct-input route (two questions in one pop-up; answered 2026-10-18 UTC; brief `SF-IF-004_2026-10-17.md`; recorded as D-SPYC-23)

1. Which of BASE's three declared operation names (`pickett_export`, `pgopher_export`,
   `spectroscopy_snapshot`) SPYFY provides (C none, own operations through `cochem.modules` / B
   rename two, retire one / A accept all three / D drop the requirement). Selected: **"SF-IF-004:
   C — provide none of the three names; SPYFY declares its own operations through cochem.modules
   (Recommended)"**.
2. The route for SF-IF-005a files given that the BASE manifest rejects Pickett suffixes.
   Selected: **"Route ii: BASE ingests Pickett files and ORCA output with new multi-file handoff
   kinds; SF-IF-002 keeps its 'only' clause (Recommended)"** (not chosen: "Route i: under hosting
   H2 the user supplies files inside the SPYFY notebook; SF-IF-002's 'only' clause is relaxed").

## UC-SEL use cases of the first release (three questions in one pop-up; answered 2026-10-18 UTC; brief `UC-SEL_2026-10-17.md`; recorded as D-SPYC-24)

1. Which use cases enter the first release and in what order (B two tracks staged / A rotational
   core staged / C one release / D vibrational first). Selected: **"UC-SEL: B — two tracks, staged:
   release 1 = UC-7, UC-8 (Pickett), WP-PICK round trip, UC-1 with SPFIT as fit of record, UC-9 when
   GEOM agrees a schema, UC-5 when R1 lands, plus UC-4 as P8 envelopes only; release 2 = UC-2, UC-3,
   UC-6, P2–P4, P7, P15 (Recommended)"**.
2. The molecule for the release-1 rotational demonstration V-DEMO-1 (options: SO₂ from the §8.3
   panel; H₂CO or CH₃CCH; a quadrupolar molecule the PI names). The PI typed: **"Search and find a
   case to compare for each of the possible engine components that has the main isotope log and
   another isotope log that we can compare. Look for databases that we can pull these from that
   have been from published reputable sources."**
3. UC-2 (joint rotational and rovibrational fit) in release 1. Selected: **"No: UC-2 in release 2
   (Recommended)"**.

## G0 plan phases and gates (two questions in one pop-up; answered 2026-10-18 UTC; brief `G0_2026-10-17.md`)

1. Approve the plan's phases and gates as written (A), with six amendments (B), or defer (C).
   Selected: **"G0: B — approve with the six amendments, written into PR #1 before your comment
   (Recommended, with the B-split variant of item 4)"**. The pop-up stated that this answer is
   direction and that G0 closes only when D-SPYC-01 becomes DECIDED by the PI's comment on PR #1
   naming the gate and the commit.
2. Keep all six amendments or drop some. Selected: **"Keep all six (Recommended)"** (not chosen:
   "Drop item 2: the tag becomes optional under D-SPYC-15"; "Drop the B-split of item 4: one G2 PR
   only").

## G0 amendments 1 and 2: the waiver record and the baseline tag (two questions in one pop-up; answered 2026-10-18 UTC)

1. The waiver sentence for `e87f6f4` (options: "I waive the Phase 0 no-code clause and the
   srs-v2.0 code gate for e87f6f4 only (Recommended)"; "I waive the Phase 0 no-code clause for
   e87f6f4 only"; "No waiver: record e87f6f4 as a gate crossing without waiver"). The PI typed:
   **"Remove both completely we are far enough along that they are not needed anymore."**
2. The baseline tag `legacy-swarm-2026-10` on `7e42507` (options: the PI creates it as a signed tag
   (Recommended); make it optional under D-SPYC-15). The PI typed: **"Make the key for me"**.

## Governance PR branch and the baseline-tag route (two questions in one pop-up; answered 2026-10-18 UTC)

1. The branch for the dedicated governance PR that removes the `srs-v2.0` code-gate wording from
   `AGENTS.md` and `docs/handoff/AGENT_HANDOFF.md`, and how code PRs cite requirements before the
   tag exists (options: "Open the dedicated PR from branch ccr-c1697164-ao6hz4-governance; in it,
   code PRs may cite proposed SRS v2 ids until srs-v2.0 exists, each such PR saying so
   (Recommended)"; "Open the dedicated PR from that branch; keep 'code PRs name approved ids'
   unchanged"; "Do not open a separate PR; hold the AGENTS.md change until srs-v2.0"). Selected:
   **"Open the dedicated PR from branch ccr-c1697164-ao6hz4-governance; in it, code PRs may cite
   proposed SRS v2 ids until srs-v2.0 exists, each such PR saying so (Recommended)"**. The pop-up
   stated that the PI's sentence "Remove both completely we are far enough along that they are not
   needed anymore." is the written approval the AGENTS.md governance rule requires, and that the
   branch permission was needed because this session may otherwise push only to
   `ccr-c1697164-ao6hz4`.
2. The route for the baseline tag `legacy-swarm-2026-10` on `7e42507`, after the agent declined
   "Make the key for me" (a key generated by an agent in a cloud container is not the PI's
   identity; its private half would sit on a shared machine; `AGENTS.md` bars agents from creating
   tags or credentials). Options: "Web UI, no key: GitHub → CoChem-SpycFit → Releases → Draft a
   new release → Choose a tag: type legacy-swarm-2026-10 → Target: pick commit 7e42507 → Publish
   (Recommended)"; "Annotated tag from your machine, no key: git fetch origin && git tag -a
   legacy-swarm-2026-10 7e42507 -m 'Swarm baseline before the SRS v2 work' && git push origin
   legacy-swarm-2026-10"; "Signed tag: ...". Selected: **"Signed tag: generate a key on your
   machine (gpg --quick-generate-key "Your Name <you@example.org>" ed25519 sign 2y), upload the
   public key to GitHub (Settings → SSH and GPG keys), then git tag -s legacy-swarm-2026-10
   7e42507 && git push origin legacy-swarm-2026-10"**. The PI runs these commands on their own
   machine; the agent creates no key and no tag.

## Research the PI asked for in these answers

- R1 (D-SPYC-04, D-SPYC-18, SRS §1.1): GPU-parallel, array-based, on-the-fly native fitting
  against wrapping or copying SPFIT/SPCAT; "what would give the better end result and fit the
  vision for CoChem".
- R2 (D-SPYC-05): the lowest-friction, best-usability user-interface option under hosting H2.
- R3 (D-SPYC-06): incorporating electronic and UV-Vis spectroscopy as a modular ladder, and the
  boundary between SPYFY and "another module".
- R4 (D-SPYC-14 follow-up 5): a replacement for OCS with public SPFIT fit inputs.
- R5 (D-SPYC-13, 2026-10-18): a published detection value for the weak-component preset, to be
  researched and then pre-filled.
- R6 (D-SPYC-24, 2026-10-18): for each engine component (SPFIT/SPCAT, DPFIT/DPCAT, XIAM, ERHAM,
  later BELGI-Cs/C1, RAM36, westerfit), a published case with a main-isotopologue line list and a
  second-isotopologue line list that can be compared, drawn from reputable databases of published
  data, with sources and redistribution class recorded.
