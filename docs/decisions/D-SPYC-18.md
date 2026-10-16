# D-SPYC-18 In-process Hamiltonian model and its scope

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

> "Run genuine SPCAT/SPFIT alongside native SpycFit for compatible microwave models."

## Consequences

- A native model exists for microwave models that SPCAT/SPFIT also support; each native term
  must match the pinned SPCAT before it is enabled (SRS v2 SF-HAM-001).
- SPFIT comes first in publication tables (D-SPYC-01), so native results are shown beside, not
  instead of, SPFIT results.
- Still open: which terms, in which order.

## Alternatives not chosen

Option A (no in-process model) was not chosen. Options B and C remain open as to scope.
