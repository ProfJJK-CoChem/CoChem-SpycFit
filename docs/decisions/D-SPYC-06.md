# D-SPYC-06 IR / Raman / UV-Vis in SpycFit

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-16 (agent question and PI reply quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`):

> "IR, Raman, THz, near-IR, and microwave are first-class workflows with familiar interactions."

## Consequences

- SRS v2 §1.2 capability matrix rows P5-P12 are in scope; the review's and the 2026-10-11
  recommendation ("None in SpycFit") is superseded (errata row 18).
- BASE User Manual Ch. 5 describes SpycFit as an IR/Raman/UV-Vis simulator; its wording should be
  revised with the BASE maintainer (cross-repository item for the PI).
- Still open: UV-Vis, and who produces vibrational and Raman predictions (TORQ or another
  provider; literature review R3-01).

## Alternatives not chosen

Option A (no IR/Raman in SpycFit) was not chosen.
