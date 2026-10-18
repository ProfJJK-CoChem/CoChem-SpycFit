# D-SPYC-12 Inertial-defect sign rule

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-18 (UTC; pop-up at the end of the 2026-10-17 session) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-12_2026-10-17.md` |

## Decision

The agent asked four questions in one pop-up.

1. "D-SPYC-12 (1 of 4) — inertial-defect sign rule: when computed and experimental inertial
   defects Δ = I_c − I_a − I_b differ in sign, what should SPYFY do?" with options C, A, B and D.
   The PI selected: "D-SPYC-12: C — typed, uncertainty-aware display and flag, never automatic
   rejection, plus an expert-review record before export of a flagged same-type sign difference
   (Recommended)".
2. "D-SPYC-12 (2 of 4) — may the expert enable an off-by-default |Δ| window in automated triples
   searches, with dropped candidates logged?" The PI selected: "Yes: off by default,
   expert-enabled, dropped triples logged (Recommended)".
3. "D-SPYC-12 (3 of 4) — may same-type Δ agreement rank conformer candidates (never remove them),
   with the reasons shown?" The PI selected: "Yes: rank only, never remove, reasons shown
   (Recommended)".
4. "D-SPYC-12 (4 of 4) — may DIAG apply a labelled, recorded correction (VPT2 Δ₀ where TORQ
   supplies B₀; Oka's relation from the lowest out-of-plane frequency) so that a computed Δ_e
   becomes comparable with a measured Δ₀? Also the flag threshold k (a new tolerance; proposed
   k = 2)". The PI selected: "Yes, recorded correction allowed; k = 2 (Recommended)".

## Consequences

- SRS v2 SF-DIAG-001 to 006 carry the decision: Δ and the planar moments are computed for every
  constant set with propagated uncertainties and labelled by type, reduction, representation and
  evidence label; a same-type sign difference with |Δ| > 2·u(Δ) for both values is flagged and
  never causes rejection, pruning or a change of acceptance; a flagged result needs an
  expert-review record before export; a computed Δ_e compares with a measured Δ₀ only through a
  labelled, recorded correction whose model and uncertainty are shown; an |Δ| window in automated
  searches is off by default, expert-enabled, recorded, with every dropped candidate logged; Δ
  agreement may rank candidates with the reason shown and never removes one; a P_aa ≥ P_bb ≥ P_cc
  violation is reported as a mislabelled input.
- The decision follows the accepted TORQ contract §3.1 ("A sign discrepancy is a diagnostic, not
  sufficient evidence to reject a conformer") and REVIEW_ERRATA row 1 over Method Matrix §6.4
  ("binding"). The five Method Matrix lines that state the binding rule are a BASE correction for
  the PI to request (cross-repository item). TORQ's contract needs no change.
- The legacy module `src/cochem_spycfit/physics_core/cochem_inertial_defect.py` rejects on sign,
  and seven tests in `tests/test_cochem_inertial_defect.py` assert rejection or filtering. They
  contradict this decision; rewriting them is Phase 2 work in a PR that lists each affected test
  with its reason and that the PI approves (AGENTS.md).
- Evidence for the acceptance cases: Oka 1995 (doi:10.1016/0022-2860(95)08844-L; formamide,
  styrene, 2-fluoronitrobenzene) and Zhou et al. 2019 (doi:10.1063/1.5081683; formamidinium
  formate, whose abstract and Table II constants give Δ of −0.315 and −0.283 amu Å² against the
  published −0.243, a discrepancy the acceptance record states). k = 2 is a new tolerance outside
  the D-SPYC-14 tiers, recorded here.
- WP-DIAG and the `diag` component implement it; GEOM keeps structure determination (D-SPYC-10) and
  receives the typed constants through SF-IF-008.

## Alternatives not chosen

A (diagnostic only, the 2026-10-11 sheet's recommendation), B (binding rejection as the Method
Matrix reads) and D (display only, comparison in GEOM) were offered and not selected. No window at
all, no ranking, and no correction were not selected.
