# D-SPYC-19 Dependencies on CoChem-BASE, TOPOS and TORQ

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
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

## Consequences

- CoChem-BASE is a hard requirement: SpycFit installs and runs only with it, because BASE owns
  ingestion and the unified GUI. The vendored copy in `src/cochem_base/` is to be replaced by the
  BASE package (Phase 2).
- TOPOS and TORQ are dependencies of the normal workflow (conformers from TOPOS, predictions
  from TORQ).
- A direct-input path accepts predicted ORCA output files and Pickett SPCAT/SPFIT files
  (`.par`, `.var`, `.int`, `.lin`, `.cat`) as valid inputs without going through TOPOS and TORQ.
  Inputs on this path keep their source, format and SHA-256 in provenance, and are labelled as
  user-supplied rather than as TORQ predictions.
- Still open: whether TOPOS and TORQ are install-time requirements or optional components (the
  direct-input path works without them); which ORCA outputs and quantities the path reads, and
  whether it uses the ORCA parser of TORQ or CoChem-ORCA rather than a new one in SpycFit.

## Alternatives not chosen

Option B (SpycFit carries copies of BASE's handoff schema and constants) was not chosen.
