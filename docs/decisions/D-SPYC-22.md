# D-SPYC-22 Module name

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/module_name_2026-10-17.md` |

## Decision

The PI had asked for a new four- or five-letter all-capitals name for SpycFit. The agent offered
SPAFT, SPYAF, SPEFY and SPEKT, with SPECQ, SPYFT, ROFIT and SPEFT as typeable alternatives. The PI
typed: "Derivative of SPEFY but instead spelled SPYFY, pronounced Spiffy".

The agent checked SPYFY on 2026-10-17 (PyPI JSON API and simple index return 404 for `spyfy` and
`SPYFY`, control `numpy` 200; no exact or look-alike match in the 915,698-project PyPI index;
one-letter near-names pyfy, py-fy, spfy, spycy, spydy, spyfi, spyky; `spiffy` is an unrelated PyPI
package; GitHub search shows four small repositories named Spyfy or spyfy, none scientific; Spy-Fy
at spy-fy.com sells iPhone camera-cover cases and "Hidden Camera Detector - SpyFy" is an iOS app;
trademark registries were not searched) and asked: "Confirm SPYFY?" with options "Confirm SPYFY
(Recommended)", "Use SPAFT instead" and "Hold the name; keep SpycFit for now and decide later".
The PI selected: "Confirm SPYFY (Recommended)".

## Consequences

- The module's name is SPYFY, pronounced "spiffy". SRS v2 §1.1 notes the name; the documents keep
  "SpycFit" until the rename is carried out.
- The rename touches the repository name (CoChem-SpycFit), the Python package `cochem_spycfit` and
  the `pyproject.toml` name, the documents, and the governance files (`AGENTS.md`, `CLAUDE.md`,
  `GEMINI.md`, `.github/copilot-instructions.md`, `docs/handoff/AGENT_HANDOFF.md`,
  `docs/decisions/README.md`), which need a PR whose only purpose is that change, approved by the
  PI in writing before it is opened. `pyproject.toml` is also a Phase 2 file.
- In CoChem-BASE the rename touches the module registry entry `spycfit`
  (`src/cochem_base/interfaces/module_registry.py`), `module_id='spycfit'` in the interface
  bridges and `resource_guard.py`, and `SPYCFIT = "cochem_spycfit_silo"` in
  `cochem_system_config.py`. BASE is read-only for agents; this needs the PI's request to the BASE
  maintainer.
- The SF- requirement ids do not change.
- Trademark status is unverified for SPYFY, as for every candidate; the commercial uses found are
  consumer privacy products, not scientific software.

## Alternatives not chosen

SPAFT, SPYAF, SPEFY, SPEKT, SPECQ, SPYFT, ROFIT and SPEFT (brief ranking and clashes in
`docs/handoff/briefs/module_name_2026-10-17.md`); "Use SPAFT instead" and "Hold the name" at the
confirmation step.
