# D-SPYC-13 Dark dipole components

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent): the preset value waits for the research the PI asked for. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-18 (UTC; pop-up at the end of the 2026-10-17 session) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-13_2026-10-17.md` |

## Decision

The agent asked four questions in one pop-up.

1. "D-SPYC-13 (1 of 4) — dark dipole components: when a predicted dipole or transition-moment
   component is small, how should SPYFY treat the affected lines?" with options C, A and B. The PI
   selected: "D-SPYC-13: C — visibility as a recorded view: three label states plus an intensity
   floor relative to the strongest predicted line; labels sit beside the catalog and re-rank only
   (Recommended)".
2. "D-SPYC-13 (2 of 4) — may labelled-weak lines be pruned from automated searches (never from
   catalogs or comparisons)?" The PI selected: "Expert opt-in, recorded, rerunnable (Recommended)".
3. "D-SPYC-13 (3 of 4) — the preset value for the weak-component setting." The PI selected:
   "Research a published detection value now, then pre-fill it".
4. "D-SPYC-13 (4 of 4) — where do lines whose label cannot be computed (missing dipole, partition
   function or spin weight) rank?" The PI selected: "Normal rank, shown 'not evaluated' with the
   reason (Recommended)".

Option C as offered: three label states (forbidden by symmetry, weak below the user setting,
missing with reason) plus an intensity floor relative to the strongest predicted line, per summed
rotational line with hyperfine components added; labels carry source and uncertainty, sit beside
the catalog and re-rank only; a conformer below the setting on every axis is labelled, never
removed.

## Consequences

- SRS v2 SF-INT-002a to 002e carry the decision. Labels are computed per species, conformer, axis
  system and producer with the frame recorded; the relative floor needs no intensity calibration;
  no label, mask or view filter removes a line from a catalog or a comparison or changes an `.int`
  dipole; pruning in automated searches is an expert opt-in, recorded and rerunnable; unevaluated
  lines rank normally and show "not evaluated" with the reason.
- The preset for the weak-component setting is TBD until the research the PI asked for (R5) finds
  a published detection value; the Method Matrix's 0.1 D is not pre-filled. Until then the field
  starts empty and only the forbidden and missing labels appear.
- Consistent with the accepted TORQ contract §3.4 ("A 0.1 D display/detection threshold cannot
  delete nonzero archival transitions") and REVIEW_ERRATA row 2; Method Matrix §6.1 ("excluded from
  the predicted line list") is not followed, a BASE correction for the PI to request
  (cross-repository item). The legacy code that deletes lines below 0.01 D
  (`cochem_boltzmann_synthesizer.py`; BASE's `cochem_dark_branch_filter.py`) or 0.05 D
  (`cochem_autofit_engine.py`) contradicts it; the SpycFit parts are Phase 2 work, the BASE part a
  PI request.
- In an array or GPU engine (the PI's 2026-10-17 statement under D-SPYC-04) the label is a mask
  column, never a deletion; under D-SPYC-16 D consumed catalogs keep their producer label and the
  labels sit beside them.
- IR, Raman and NIR labels read "not evaluated" until TORQ exports transition moments (SF-IF-006);
  electronic transition moments for P15 wait on the D-SPYC-06 research.
- Acceptance: a test that no SPYFY label, mask or view filter removes a catalog or comparison line
  or changes an `.int` dipole, on `.cat` files made at test time from hash-verified scratch copies
  (SF-DATA-001e); camphor (Kisiel et al. 2003, doi:10.1039/B212029A) as the weak-but-observed case
  once its data availability is checked.

## Alternatives not chosen

A (component label with three states, the sheet's A extended) and B (exclude weak branches, the
Method Matrix as written) were offered and not selected. Never pruning and pruning on by default
were not selected. No preset (the agent's recommendation) and the 0.1 D Matrix value were not
selected; the PI chose to research a published detection value and then pre-fill it. Lowest rank for
unevaluated lines was not selected.
