# D-SPYC-03 Feedback to TORQ

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (route) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-16 (agent question and PI reply quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`; pop-up answer of 2026-10-17 in `docs/handoff/pi_answers_2026-10-17_popups.md`; brief `docs/handoff/briefs/D-SPYC-03_2026-10-17.md` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`):

> "PES/force-field refinement is required in the overall workflow and may be implemented in another CoChem module."

On 2026-10-17 the agent asked, as a pop-up: "D-SPYC-03 — how should SpycFit send theory-feedback
requests (recompute requests to TORQ; inverse PES or force-field fits to a provider not yet
named)?" with options A (staged, schema (i)), A2, D and C2, B typeable. The PI selected:
"D-SPYC-03: A, schema (i), staged (Recommended)".

Option A, schema (i), staged, as offered: the request file becomes a new BASE handoff kind; a
human approves the receiver's plan; SpycFit drafts the schema and agrees it with each receiver by
consumer tests in both repositories, mapped to `cochem.torq.request/1` for TORQ and extended to
carry assigned observables, uncertainties and `data_usage`. Staged: SF-RUN-005a is written now;
the BASE requirement stays blocked until BASE adds the kind.

## Consequences

- Refinement requests (SRS v2 SF-RUN-005a/b) and the provider interface (SF-SCOPE-005, §4.3)
  stay in scope. SF-RUN-005a is unblocked and names the route; SRS v2 SF-IF-009 states the BASE
  handoff kind and stays blocked on BASE adding it and on a receiver.
- SpycFit owns the request schema (schema owner (i)); TORQ-bound requests map to
  `cochem.torq.request/1`, which today carries scalar experimental anchors with `uncertainty` and
  `data_usage` and needs the assigned-observable extension. Inverse-fit routing waits for a
  provider module.
- Cross-repository items for the PI: BASE must add the handoff kind and validator and relax its
  geometry-only executor check; TORQ needs an SRS change and a consumer. TORQ's task authority
  (TORQ-ARC-002, D08) and plan approval (TORQ-PLAN-003) are kept.
- Observables sent for calibration must never later validate the returned model (TORQ-SCI-001).
  The round trip stays unverified until a receiver answers.

## Alternatives not chosen

Option B (no feedback path in v2) is not compatible with the direction. A2 (request in handoff
`options` beside a geometry artifact), D (request file carried by the user) and C2 (a versioned
BASE task-service API) were offered on 2026-10-17 and not selected; C1 (writing another module's
task database) was excluded because it breaks TORQ-ARC-002.
