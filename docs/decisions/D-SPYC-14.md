# D-SPYC-14 Benchmark set, data supply and tolerances

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (panel) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversations 2026-10-16 and 2026-10-17 (agent questions and PI replies quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`; see also amendment §7 D-03):

> "Establish independent semirigid comparison first; then separately validate hyperfine, excited-state, and large-amplitude-motion extensions. All remain in scope."

On 2026-10-17 the agent asked, as item 2: "**D-SPYC-14** first semirigid comparison panel
(proposed: CO, OCS, SO₂, H₂CO, propyne)". The PI replied: "2. Yes, proceed."

As item 3 the agent asked: "**D-SPYC-14** tolerance tiers and their starting values". The PI
replied: "3. Research what limits are considered valid in publications from the last 5 years and
give me suggestions."

## Consequences

- The first SPCAT/SPFIT comparison panel is semirigid: CO, OCS, SO₂, H₂CO and propyne
  (CH₃CCH), as listed in SRS v2 §8.3.
- The earlier D-SPYC-14 set (propylene oxide, Ar-oxazole, H₂CO-HCl, HCO) and this panel still
  need one reconciled registry (literature review R4-12); HCO stays an I/O and provisioning
  regression (V-PICK-05).
- Still open: the tolerance tiers and their values (SRS v2 SF-CMP-008a). The PI asked for
  suggestions based on limits accepted in publications from the last five years.

## Alternatives not chosen

Not yet decided.
