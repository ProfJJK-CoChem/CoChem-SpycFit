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

## Research the PI asked for in these answers

- R1 (D-SPYC-04, D-SPYC-18, SRS §1.1): GPU-parallel, array-based, on-the-fly native fitting
  against wrapping or copying SPFIT/SPCAT; "what would give the better end result and fit the
  vision for CoChem".
- R2 (D-SPYC-05): the lowest-friction, best-usability user-interface option under hosting H2.
- R3 (D-SPYC-06): incorporating electronic and UV-Vis spectroscopy as a modular ladder, and the
  boundary between SPYFY and "another module".
- R4 (D-SPYC-14 follow-up 5): a replacement for OCS with public SPFIT fit inputs.
