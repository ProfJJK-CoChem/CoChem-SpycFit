# D-SPYC-03 Feedback to TORQ

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

> "PES/force-field refinement is required in the overall workflow and may be implemented in another CoChem module."

## Consequences

- Refinement requests (SRS v2 SF-RUN-005a/b) and the provider interface (SF-SCOPE-005, §4.3)
  stay in scope.
- Still open: the hand-off route (options A-C in `docs/handoff/PI_DECISIONS.md`).

## Alternatives not chosen

Option B (no feedback path in v2) is not compatible with the direction. Options A and C remain
open.
