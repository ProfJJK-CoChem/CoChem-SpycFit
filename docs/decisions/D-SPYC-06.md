# D-SPYC-06 IR / Raman / UV-Vis in SpycFit

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (prediction owner) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversations 2026-10-16 and 2026-10-17 (agent questions and PI replies quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`):

> "IR, Raman, THz, near-IR, and microwave are first-class workflows with familiar interactions."

On 2026-10-17 the agent asked, as item 4: "**D-SPYC-06** who produces vibrational and Raman
predictions: TORQ, another provider or SpycFit". The PI replied: "4. TORQ."

As item 20 the agent asked: "**D-SPYC-06** UV-Vis: in scope or out (you named IR, Raman, THz,
NIR and MW only)". The PI replied: "20. Research then give suggestions, context, pros, cons, and
risks. I think the calculations and requirements for fitting UV-Vis share little to no overlap
with rotovibrational spectroscopy predictions."

## Consequences

- SRS v2 §1.2 capability matrix rows P5-P12 are in scope; the review's and the 2026-10-11
  recommendation ("None in SpycFit") is superseded (errata row 18).
- BASE User Manual Ch. 5 describes SpycFit as an IR/Raman/UV-Vis simulator; its wording should be
  revised with the BASE maintainer (cross-repository item for the PI).
- TORQ produces vibrational and Raman predictions. SpycFit consumes them through the versioned TORQ
  interface (SRS v2 §4.3) and does not compute them. Literature review R3-01 is resolved for
  ownership; what TORQ exports for these profiles is a cross-repository item for the PI and the
  TORQ maintainer.
- Still open: UV-Vis. The PI asked for a brief and stated a view, quoted above.

## Alternatives not chosen

Option A (no IR/Raman in SpycFit) was not chosen. SpycFit or another provider as the producer
of vibrational and Raman predictions was not chosen.
