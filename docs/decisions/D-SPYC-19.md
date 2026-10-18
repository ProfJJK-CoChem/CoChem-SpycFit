# D-SPYC-19 Dependencies on CoChem-BASE, TOPOS and TORQ

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 (two replies) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17 (agent question and PI reply quoted below; the reply was dictated and is quoted as received) |

## Decision

The agent asked, as item 13: "**D-SPYC-19** declare `cochem-base` as a dependency (it is
vendored today)".

The PI replied: "13. CoChem-BASE, TOPOS, and TORQ are all dependencies. However, there should be
a way to input the predicted um Mhm. That settles that. Welcome to the no-chat or the predicted
um orca files or SP um Cat SP fit files and have them read and act as a valid input rather than
having to go through topo and torque. CoChem-BASE though is a hard requirement cuz that has the
ingestion and the unified GUI."

The agent then asked: "**D-SPYC-19:** must TOPOS and TORQ be installed for SpycFit to install,
or are they optional when you load files directly? And which ORCA results should SpycFit read?"

The PI replied: "19 They are mandatory for CoChem BASE anyways so it is mute point."

## Consequences

- CoChem-BASE is a hard requirement: SpycFit installs and runs only with it, because BASE owns
  ingestion and the unified GUI. The vendored copy in `src/cochem_base/` is to be replaced by the
  BASE package (Phase 2).
- TOPOS and TORQ are always present, because CoChem-BASE requires them (BASE's
  `pyproject.toml` packages `cochem_topos` and `cochem_torq`, lines 63-78 at `7d5b8ac`).
  SpycFit therefore needs no separate install-time rule for them. The direct-input path is a
  choice of workflow, not a way to run without TOPOS or TORQ installed.
- A direct-input path accepts predicted ORCA output files and Pickett SPCAT/SPFIT files
  (`.par`, `.var`, `.int`, `.lin`, `.cat`) as valid inputs without going through TOPOS and TORQ.
  Inputs on this path keep their source, format and SHA-256 in provenance, and are labelled as
  user-supplied rather than as TORQ predictions.
- Still open: which ORCA outputs and quantities the direct-input path reads, and whether it uses
  the ORCA parser of TORQ or CoChem-ORCA rather than a new one in SpycFit. Because TORQ is always
  installed, reusing its parser adds no dependency.

## Alternatives not chosen

Option B (SpycFit carries copies of BASE's handoff schema and constants) was not chosen.
