# D-SPYC-25 Removal of the Phase 0 no-code clause and the `srs-v2.0` code gate

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. The governance-file part is carried by a dedicated PR under the AGENTS.md rule for governance files. |
| Answered | 2026-10-18 (UTC) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up question (`docs/handoff/pi_answers_2026-10-17_popups.md`, "G0 amendments 1 and 2"); brief `docs/handoff/briefs/G0_2026-10-17.md` item 1 |

## Decision

The agent asked: "G0 amendment 1 — the waiver record for e87f6f4 (D-SPYC-25). AGENTS.md says a
gate is waived only when you name it and say so, in your own words, recorded in docs/decisions/.
The commit e87f6f4 (2026-10-16) corrected the amendment §5 truthfulness defects in code at your
request; it crossed two written rules: the Phase 0 clause 'No changes to code, tests or
configuration' and the code gate 'No code PR before the srs-v2.0 tag'. Which sentence is your
waiver?" with the options "I waive the Phase 0 no-code clause and the srs-v2.0 code gate for
e87f6f4 only", "I waive the Phase 0 no-code clause for e87f6f4 only" and "No waiver: record
e87f6f4 as a gate crossing without waiver".

The PI typed: "Remove both completely we are far enough along that they are not needed anymore."

## Consequences

- The two rules are removed, not waived for one commit:
  1. the Phase 0 clause "No changes to code, tests or configuration" in
     `docs/plan/CoChem-SpycFit_Plan.md` (removed in the plan on 2026-10-18; the plan's phase table
     is not a governance row);
  2. the code gate "No code PR before the `srs-v2.0` tag" in the plan's Phase 1 row (removed on
     2026-10-18) and its wording in `AGENTS.md` ("only after the PI approves them (the PI's tag
     `srs-v2.0`) to build the code against them") and `docs/handoff/AGENT_HANDOFF.md`. Those two
     files are governance files: AGENTS.md requires a PR whose only purpose is that change,
     approved by the PI in writing before it is opened. The PI's sentence above is that written
     approval; the branch for the dedicated PR is confirmed with the PI before it is opened, because
     this session's instructions allow pushes only to `ccr-c1697164-ao6hz4` without the PI's
     explicit permission.
- `e87f6f4` is recorded as a code change made before the removal, at the PI's 2026-10-16 request
  (PR #1). No waiver sentence exists for it; the PI chose removal of the rules instead.
- G0 amendment 3 of the G0 brief (a governance PR after G1 removing the code-gate wording) is
  brought forward: the PR is opened now, once the branch is confirmed.
- Rules that stay in force, because the PI did not name them: the approval mechanism for SRS v2
  (the `srs-v2.0` tag approves the requirements; agents write only `proposed`); "When the approved
  SRS v2 and the code disagree, the approved SRS wins. Draft or proposed SRS text never justifies a
  code change."; and "A code PR names the approved SRS v2 requirement ids it implements". With the
  code gate removed and no approved SRS yet, a code PR cannot name approved ids; the agent puts this
  consequence to the PI with the branch question (either an interim rule that code PRs cite
  proposed ids pending `srs-v2.0`, or code PRs wait for the tag in practice).
- Phase 2 (hygiene) and the work packages keep their own gates G2–G4 (G0 amendments, PI
  2026-10-18); nothing else in the gate structure changes.

## Alternatives not chosen

The three waiver sentences offered were not selected; the PI chose removal of both rules.
