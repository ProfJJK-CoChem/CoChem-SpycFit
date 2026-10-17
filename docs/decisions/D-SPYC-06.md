# D-SPYC-06 IR / Raman / UV-Vis in SpycFit

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent): the electronic-state field and the electronic scope are answered; the UV-Vis module boundary waits for research the PI asked for. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (prediction owner); 2026-10-17 (UV-Vis view retracted; electronic scope) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversations 2026-10-16 and 2026-10-17 (agent questions and PI replies quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`; pop-up answers of 2026-10-17 in `docs/handoff/pi_answers_2026-10-17_popups.md`; brief `docs/handoff/briefs/D-SPYC-06-UVVIS_2026-10-17.md` |

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

Later on 2026-10-17 the agent presented the UV-Vis brief as a pop-up with three questions.

1. "D-SPYC-06 UV-Vis (1 of 3) — should SpycFit v2 fit UV-Vis spectra (electronic transitions),
   which your 2026-10-16 list (IR, Raman, THz, near-IR, microwave) leaves out?" with options A
   (exclude electronic transitions, worded by transition type), B (a later profile P15 for
   rotationally resolved gas-phase electronic spectra), C (B plus condensed-phase electronic bands)
   and D (empirical envelopes only). The PI typed: "Retract my comment on UV-VIS we will indeed
   need to incorporate the full electronic treatment for those specific microwave use cases. We
   still will handle UV-VIS spectra predictions and fitting for another module but will incorporate
   the full electronic microwave spectroscopy predictions and fitting."
2. "D-SPYC-06 UV-Vis (2 of 3) — if electronic transitions are excluded, what happens to the O₂
   electronic bands that arrive inside HITRAN near-IR imports?" The PI typed: "Give me research
   and analysis on incorporating UV-VIS spectroscopy as well. The spectroscopy should be
   modularized and allow the user to go as far as they want with what predictions and evidence
   they have."
3. "D-SPYC-06 UV-Vis (3 of 3) — reserve an electronic-state field in the state identity now
   (SF-CMP-004a), so that option B could be added later without migrating stored data?" The PI
   typed: "We are going to pivot and include this field and scope."

## Consequences

- SRS v2 §1.2 capability matrix rows P5-P12 are in scope; the review's and the 2026-10-11
  recommendation ("None in SpycFit") is superseded (errata row 18).
- BASE User Manual Ch. 5 describes SpycFit as an IR/Raman/UV-Vis simulator; its wording should be
  revised with the BASE maintainer (cross-repository item for the PI).
- TORQ produces vibrational and Raman predictions. SpycFit consumes them through the versioned TORQ
  interface (SRS v2 §4.3) and does not compute them. Literature review R3-01 is resolved for
  ownership; what TORQ exports for these profiles is a cross-repository item for the PI and the
  TORQ maintainer.
- The PI's item-20 view that UV-Vis fitting shares "little to no overlap" with rovibrational
  predictions is retracted (2026-10-17). Electronic treatment is in scope: "the full electronic
  microwave spectroscopy predictions and fitting", that is, electronic states in rotational
  spectroscopy. The state identity carries an electronic-state field (SRS v2 SF-CMP-004e), set to
  the ground electronic state for every P1-P14 record. SRS v2 §1.2 gains a placeholder profile P15
  whose rungs (open-shell fine and hyperfine structure, rotational spectra in excited electronic
  states, rotationally resolved electronic transitions) are blocked on the research below.
- UV-Vis spectra prediction and fitting: the PI said this is "for another module" and in the next
  answer asked for "research and analysis on incorporating UV-VIS spectroscopy as well", modular,
  so that "the user [can] go as far as they want with what predictions and evidence they have".
  The research (R3 in `docs/handoff/pi_answers_2026-10-17_popups.md`) lays out a ladder from
  open-shell rotational spectra to broad UV-Vis bands and puts the boundary between SPYFY and the
  other module to the PI as explicit options.
- The O₂ near-IR electronic bands in HITRAN imports and the BASE User Manual Ch. 5 `UVVIS`
  keyword are settled with that research, not before.
- The 2026-10-11 recommendation "None in SpycFit" and the brief's recommendation A are superseded
  by the PI's answers.

## Alternatives not chosen

Option A (no IR/Raman in SpycFit) was not chosen. SpycFit or another provider as the producer
of vibrational and Raman predictions was not chosen. For UV-Vis on 2026-10-17, the brief's options
A (exclude electronic transitions), B, C and D were not selected as offered; the PI's typed
answers replace them.
