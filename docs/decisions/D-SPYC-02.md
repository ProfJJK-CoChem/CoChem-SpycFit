# D-SPYC-02 Masses, abundances and constants

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-02_2026-10-17.md` |

## Decision

The agent asked, as part 1: "D-SPYC-02 (part 1 of 2) — which source and owner for isotope masses,
abundances and physical constants?" with options D, A, B and E (sheet option C typeable). The PI
selected: "D-SPYC-02: D — mendeleev pinned exactly, read directly (Recommended)".

Option D as offered: the Python package mendeleev, pinned to an exact release and read directly;
provenance "AME2020 via mendeleev X.Y.Z" plus the SHA-256 of the extracted table; a CI check of
the isotopes used against the hash-pinned AME2020 file (downloaded, never committed), which also
supplies the estimated-value flag.

The agent asked, as part 2, four sub-questions. The PI selected:

1. Isotope default when an input names an element but not its isotope: "(i) Most-abundant isotope
   as a named, overridable, recorded default (Recommended)".
2. Isotopic abundance for intensities (SF-INT-001a): "(c) User value per sample, with (a)
   mendeleev's value as the named default (Recommended)".
3. Fundamental constants: "(f) CODATA 2022 frozen in SpycFit's profile; SciPy as a cross-check
   (Recommended)".
4. Nuclear spins and moments: "Separate decision (Recommended)".

Common rules stated in the pop-up for every option: atomic masses for isotopologue constants,
including those in Pickett files; nuclear or effective masses only as stated model inputs; values
arriving with an input (TORQ, ORCA, JPL, CDMS, HITRAN) recorded as received, never silently
recomputed; SPCAT's built-in constants recorded. No option changes a mass value.

## Consequences

- SRS v2 §1.7.5 and SF-IO-005 name mendeleev, pinned exactly in the Phase 2 PR, as the mass and
  abundance source, with the AME2020 cross-check in CI and the table hash in provenance. The
  agent's check of 2026-10-17: mendeleev 1.0.0 and 1.3.0 match AME2020 for all 3,557 masses and
  uncertainties, including the 1,008 values AME marks as estimates, none of them a naturally
  occurring isotope; its abundances are CIAAW 2013 values via NUBASE2020, intervals as midpoints
  (brief, Context).
- The most-abundant isotope is a PI-named default under AGENTS.md: user-overridable, recorded in
  provenance as a default, and flagged on every result that uses it.
- Abundance is a per-sample user value; mendeleev's CIAAW 2013 value is the named default,
  recorded as such.
- Constants: CODATA 2022 frozen in SpycFit's constants profile, SciPy's value as a cross-check;
  h/(8π² u Å²) and other derived factors are computed from the profile, not typed in (eight
  modules type them today, one with a value that matches neither CODATA 2018 nor 2022). Editions
  change only with TORQ, which implements `cochem.constants.codata-2022-torq-compatible/1`.
- Nuclear spins and quadrupole moments (hyperfine, Raman spin weights) get their own decision
  line before P2 and P10 work.
- Cross-repository item for the PI: CoChem-BASE ships a different `cochem_torq` package without
  TORQ's `units` and `isotope_record` modules and does not depend on CoChem-TORQ, so D-SPYC-19's
  note that BASE supplies TORQ does not hold today (brief, Context).
- A silent data change in a new mendeleev release is the main risk; the exact pin, the AME2020
  CI check and the table hash reduce it.

## Alternatives not chosen

A (call TORQ's `units` profile and `isotope_record` at run time), B (a SpycFit-owned snapshot
with id and manifest), E (versioned reference data in BASE or a shared package; the agent
suggested E later, if the PI asks BASE for it) and sheet option C (AME2020 with CODATA 2018) were
offered and not selected. For the sub-questions, (ii) no isotope default, (a) mendeleev's
abundance only, (b) CIAAW 2024 midpoints, (c) with (b) as default, (p) SciPy pinned below the
CODATA 2026 release, and including spins in this decision were not selected.
