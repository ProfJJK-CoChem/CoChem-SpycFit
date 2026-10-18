# D-SPYC-07 Eigenvalue label swaps during a fit

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent); the 07c thresholds are provisional by the PI's instruction. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-07_2026-10-17.md` |

## Decision

The agent asked four questions in one pop-up.

1. "D-SPYC-07 (1 of 4) — eigenvalue label swaps: when a parameter change moves an assigned
   transition's label onto a different eigenstate (or changes the eigenvector behind a kept label),
   what should the native model do?" with options D, A (sheet), B (sheet) and C. The PI selected:
   "D-SPYC-07: D — flag every swap, continue, record it in provenance; export needs an
   expert-review record (Recommended)".
2. "D-SPYC-07 (2 of 4) — 07b: should SpycFit also check finished SPFIT and DPFIT runs for the same
   event, by comparing SPCAT/DPCAT eigenvectors (EGYFLG 3) at the start and final parameters?" The
   PI selected: "On by default, user-overridable, recorded (Recommended)".
3. "D-SPYC-07 (3 of 4) — 07c: the swap test and near-tie margin." Options: "0.5 swap / 0.9
   near-tie (Recommended)" (1); "Defer the numbers until the stepping test has run" (2); "Other
   values". The PI typed: "2 but use 1 as a provisional value that should be adjusted with further
   evidence. Allow these numbers to be changed by advanced users. Once further testing has been
   completed we will revise these defaults. Ensure a testing to be completed and published is
   flagged in a file. When CoChem is coded to a release state we should have documents for what
   cases need to be run for the publications to demonstrate CoChem modules."
4. "D-SPYC-07 (4 of 4) — E, a labelling variant for any letter: label native states by
   eigenvector-overlap continuity across parameter steps (PGOPHER's approach) and propose the
   relabel for your acceptance, instead of SPFIT's energy-rank-within-block rule?" The PI
   selected: "Not now: keep SPFIT-style energy-rank labelling as the native rule (Recommended)".

## Consequences

- SRS v2 SF-HAM-002 is rewritten: the native model continues on a swap, flags the transition and
  records the swap in provenance (largest overlap, parameter step, old and new labels, reviewer
  action and time); SF-HAM-002c requires an expert-review record before a result carrying a
  flagged swap is exported, and marks its comparisons INCOMPLETE until then. This follows the PI's
  2026-10-16 pattern for divergence ("Divergence remains visible. Exploration continues, but
  publication export requires explicit expert review") and holds alike for native, SPFIT, batched
  and headless runs, including on-the-fly array evaluation where "between iterations" has no
  single meaning.
- SF-HAM-002a (07b): every SPFIT and DPFIT run gets a post-run eigenvector comparison through
  SPCAT or DPCAT at EGYFLG 3, on by default, user-overridable per run with the choice recorded;
  INCOMPLETE when a `.lin` state lies outside FBGN/FEND/MAXV or the basis changed. Measured cost on
  HCO scratch copies: two extra SPCAT runs of 0.006–0.020 s each (larger blocks unverified).
- SF-HAM-002b (07c): the swap test is the largest-overlap map |⟨old_i|new_j⟩|² not being the
  identity for a state an assigned line uses; 0.5 (two-state crossover) and 0.9 (near-tie margin)
  are provisional values that advanced users may change, with the change recorded. They are
  revised after the stepping test proposed in the brief (an analytical two-state crossing; HCO
  stepped across block 4; SO₂ and H₂CO panel inputs stepped through ±3u in ten steps). The pending
  test is listed in `docs/evidence/pending_tests.md` (SRS v2 SF-VER-006), the register the PI asked
  for ("Ensure a testing to be completed and published is flagged in a file.").
- SF-HAM-003: the native labelling rule stays energy rank within a symmetry block, as SPFIT's
  DIAG 0; overlap-continuity labelling (E) is not adopted now, so SF-CMP-004a state matching needs
  no native-to-DIAG mapping.
- The PI's instruction on publication cases applies to every CoChem module: "When CoChem is coded
  to a release state we should have documents for what cases need to be run for the publications
  to demonstrate CoChem modules." For SpycFit it becomes SRS v2 SF-VER-007 (a publication
  demonstration-cases document before release); for the other modules it is a cross-repository
  item for the PI.
- The halt form that options A and C would have needed (a `HALTED_LABEL_SWAP` run outcome through
  the BASE handoff) is not required.

## Alternatives not chosen

A (halt the fit and show the swap, the 2026-10-11 sheet's recommendation), B (warn and continue)
and C (halt stored native fits, previews continue) were offered and not selected. For 07b, "off by
default" and "no such check" were not selected. For 07c, adopting 0.5/0.9 as settled values and
typing other values were not selected; the PI chose deferral with 0.5/0.9 as provisional,
user-changeable values. E (overlap-continuity labelling) was not adopted now.
