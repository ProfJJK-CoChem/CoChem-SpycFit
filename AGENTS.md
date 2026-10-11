# Working on CoChem-SpycFit

Start with [the agent handoff](docs/handoff/AGENT_HANDOFF.md). It records the current state, the
documents to read, the decisions the PI still has to make, and the first steps. Then read
[PI decisions](docs/handoff/PI_DECISIONS.md) and [review errata](docs/handoff/REVIEW_ERRATA.md).

The current objective is to write the SpycFit SRS v2, architecture and plan **with the PI**, and
only after the PI approves them (tag `srs-v2.0`) to build the code against them. No normative
SRS exists yet. The old `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md`, the copied
`.docs/task1_subsystems_architectural_specification.md`, `swarm_state.json`, the
`*_payload.md` files and existing test names are historical evidence, not requirements.

SpycFit is the rotational-spectroscopy assignment and fitting module of the CoChem ecosystem
(pending PI confirmation in D-SPYC-01). TOPOS finds conformers, TORQ computes and exports
predicted constants, BASE hosts the UI and owns task state. Respect the accepted TORQ
implementation contract and decision register in `ProfJJK-CoChem/CoChem-TORQ`. Keep
interfaces to BASE and TORQ explicit and versioned. Never write another module's database.

## Scientific integrity

- Use genuine programs and published data. Never fabricate frequencies, constants,
  uncertainties, line lists, fixtures, engine output, test receipts or human approvals.
  Missing quantities stay missing, with a reason.
- Pickett's SPFIT/SPCAT is the reference implementation for Hamiltonian conventions and file
  formats. Check parsers and writers by round trip through the pinned binaries.
- Units, sign conventions, representation (I^r / III^r / III^l), Watson reduction (A / S),
  nuclear-spin weights and isotope masses are explicit inputs recorded in provenance. Never
  switch or infer them silently.
- Report numbers against stated tolerances. Never write verdicts such as PASS, CERTIFIED or
  RATIFIED, and never mark a decision, requirement, tolerance or gate as approved. Only the
  PI approves.

## Code and tests

- Change code by small diffs in reviewable PRs. Each PR names the requirement ids it
  implements and the tests it makes pass.
- Run the whole suite (`pytest tests`) before every PR and paste the summary line. Do not
  narrow `testpaths`, deselect failures to get green, or relabel old receipts as current.
- Do not edit `pytest.ini`, CI workflows, `.gitignore`, linter configuration or dependency
  pins except in a PR the PI has approved for that purpose.
- If SPFIT, SPCAT or another external program is unavailable, skip with a reason and say so.
  Never add a fallback that makes a missing program look like a pass. Never add amnesty or
  whitelist files.
- Test fixtures that claim physical provenance carry a `PROVENANCE.yaml` naming the source
  (DOI, catalog tag, or program version and input hash).

## Documents

- No agent narration, audit verdicts or "awaiting review" lines inside `docs/`, `src/` or
  `tests/`.
- When the PI's instruction conflicts with the Method Matrix, the TORQ contract or the handoff,
  state the conflict briefly with its source, then follow the PI.
- When the SRS v2 and the code disagree, the SRS v2 wins. When the SRS v2 is silent or wrong,
  stop and ask the PI in the PR rather than guessing.
