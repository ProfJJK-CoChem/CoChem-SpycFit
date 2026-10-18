# D-SPYC-24 Use cases and order of the first release (UC-SEL)

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent): the staging and order are answered; the demonstration cases wait for the research the PI asked for. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-18 (UTC; pop-up at the end of the 2026-10-17 session) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/UC-SEL_2026-10-17.md` |

## Decision

The agent asked three questions in one pop-up.

1. "UC-SEL (1 of 3) — which SRS v2 §3 use cases (UC-1 to UC-9) belong in SPYFY's first release,
   and in what order?" with options B, A, C and D. The PI selected: "UC-SEL: B — two tracks,
   staged: release 1 = UC-7, UC-8 (Pickett), WP-PICK round trip, UC-1 with SPFIT as fit of record,
   UC-9 when GEOM agrees a schema, UC-5 when R1 lands, plus UC-4 as P8 envelopes only; release 2 =
   UC-2, UC-3, UC-6, P2–P4, P7, P15 (Recommended)".
2. "UC-SEL (2 of 3) — which molecule carries the release-1 rotational demonstration (V-DEMO-1:
   broadband assignment and fit with isotopologues and uncertainty)?" with the options SO₂, H₂CO or
   CH₃CCH from the §8.3 panel, or a quadrupolar molecule the PI names. The PI typed: "Search and
   find a case to compare for each of the possible engine components that has the main isotope log
   and another isotope log that we can compare. Look for databases that we can pull these from that
   have been from published reputable sources."
3. "UC-SEL (3 of 3) — add UC-2 (joint rotational + resolved rovibrational fit) to release 1?" The PI
   selected: "No: UC-2 in release 2 (Recommended)".

## Consequences

- Release 1, in this order: UC-7 (headless batch); UC-8 with Pickett files through BASE's input kinds
  (D-SPYC-23); the WP-PICK round trip through the pinned SPFIT/SPCAT; UC-1 with SPFIT as the fit of
  record (D-SPYC-01); UC-9 once GEOM agrees the payload schema (SF-IF-008); UC-5 once the GPU
  research (R1) lands and native terms meet SF-HAM-001; UC-4 as P8 condensed-phase envelopes only,
  P7 gas-phase contours being UNAVAILABLE until PGOPHER (rights record needed) or native P6 terms
  exist, so V-DEMO-4 is partial in release 1. Release 2: UC-2, UC-3, UC-6, profiles P2–P4, P7 and
  P15.
- The plan gains a release-staging paragraph: Phase 4 runs twice, each release with its own G4; this
  is a plan amendment for the PI's G0 approval. SRS v2 §3 records the selection and §1.2's order
  column follows it.
- The demonstration cases are not fixed yet. The PI asked for research (R6): for each engine
  component (SPFIT/SPCAT, DPFIT/DPCAT, XIAM, ERHAM, later BELGI-Cs/C1, RAM36 and westerfit), a
  published case with a main-isotopologue line list and a second-isotopologue line list that can be
  compared, drawn from reputable databases of published data (CDMS, JPL, HITRAN and others), with
  sources and redistribution classes recorded (SF-DATA-001a, 001d). The result feeds V-DEMO-1, the
  §8.3 panel's OCS replacement (R4) and the engine acceptance cases.
- The PI still supplies or points to measured spectra for V-DEMO-1 and V-DEMO-4 (AGENT_HANDOFF §7);
  public databases supply constants and line lists, not broadband spectra.
- UC-2 moves to release 2; the IR observation model, HITRAN redistribution terms and the handling of
  HITRAN decade-code uncertainties are release-2 work.

## Alternatives not chosen

A (rotational core staged, without UC-4), C (one release, order only) and D (vibrational first)
were offered and not selected. For V-DEMO-1, SO₂, H₂CO or CH₃CCH from the panel and a quadrupolar
molecule were offered; the PI chose instead to have cases found by research for every engine
component. Adding UC-2 to release 1 was not selected.
