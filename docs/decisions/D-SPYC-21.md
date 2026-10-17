# D-SPYC-21 Search windows for computed constants

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); the search-window section of `docs/handoff/briefs/D-SPYC-14_tolerances_2026-10-17.md` |

## Decision

The agent asked: "Search windows for computed constants (new decision line, proposed id
D-SPYC-21)." The pop-up defined the window as [B_pred/(1+q), B_pred/(1−q)] around each predicted
constant and the transitions it implies, labelled uncalibrated and never an acceptance tolerance,
with a calibrated q from a TORQ bundle (TORQ contract §3.2) taking precedence; the decision sets
the default for constants without one. It stated the conflict with REVIEW_ERRATA item 4 (Method
Matrix §3.1 bands as the default) and gave the 2021–2026 literature values. Options: per-method-class
defaults at q = 2 × RMSD; the same at 1 × RMSD; the Method Matrix §3.1 bands as a flat default;
no default. The PI selected: "Per-method-class defaults, q = 2×RMSD (Recommended)".

The defaults as offered (relative half-width q = 2 × the relative RMSD of the method class where a
set statistic exists, otherwise 1.5 × the largest error seen; sources in the brief):

| Method class of the constants | q |
| :--- | :--- |
| Hybrid DFT: B3LYP-D3(BJ)/def2-TZVP | 4.5 % |
| Hybrid DFT: PBE0-D3(BJ)/def2-TZVP | 5.1 % |
| r2SCAN-3c | 5.4 % |
| B3LYP-D3(BJ)/def2-SVP | 5.5 % |
| PBEh-3c | 6.2 % |
| MP2 (one paper) | 2.1 % |
| Double hybrid | 1.5 %; 2 % with ΔB_vib |
| Composite or CCSD(T) + ΔB_vib | 1 % |
| Weakly bound complexes | 7.5 % |
| Multi-water clusters (one functional) | 16 % |
| Quartic distortion constants | [D/2, 2D]; [D/4, 4D] for a constant below one tenth of its largest sibling |
| Sextic distortion constants | one order of magnitude |

## Consequences

- SRS v2 SF-WIN-001 to SF-WIN-003 carry the window form, the per-method-class defaults and the
  distortion-constant rule. Each default is recorded in provenance as a default with its evidence
  grade; rows resting on one paper (MP2, multi-water clusters) are labelled thin.
- The defaults are uncalibrated. A calibrated q supplied by TORQ (TORQ contract §3.2; Method
  Matrix §17.5 split-conformal protocol) replaces them when a bundle carries one. The user may
  override q at any time; the override is recorded as user-supplied.
- REVIEW_ERRATA item 4 (windows default to the Method Matrix §3.1 bands) is superseded for the
  default; its other content (the bands are statistics, not prediction intervals; the
  [B/(1+q), B/(1−q)] form; never acceptance tolerances) stands.
- Conflict recorded: the Method Matrix §3.1 bands (B₀ 0.3–0.5 % semirigid, 1–2 % floppy) are the
  claimable accuracy of the Matrix's high tiers (composite coupled cluster with ΔB_vib), and the
  Matrix itself says they need local calibration before they are quoted for a new complex
  (`CoChem-BASE/Method_Matrix.md`, "Provenance discipline for this table" below the §3.1 table).
  Applied to hybrid-DFT constants, they would miss the true lines most of the time.
- The values are from 2021–2026 publications and need a dated revision when the literature moves.

## Alternatives not chosen

Per-method-class defaults at 1 × RMSD, the Method Matrix §3.1 bands as a flat default, and no
default (user or TORQ must supply q) were offered and not selected.
