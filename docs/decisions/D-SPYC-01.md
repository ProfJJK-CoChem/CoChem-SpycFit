# D-SPYC-01 Product identity and fitter of record

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-16 (agent question and PI reply quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The agent also asked: "3. When native and SPFIT results both exist, which one a publication table shows first."

The PI replied: "3. SPFIT first."

The confirmed direction (`docs/HANDOFF.md`, "User decisions already settled"):

> "SpycFit is a computational chemistry-to-assignment/fitting/publication pipeline. It must support rotational assignment/fitting, joint rotational–rovibrational fitting, MLFF-assisted computational assignment, and vibrational band/line/envelope fitting."

> "Run genuine SPCAT/SPFIT alongside native SpycFit for compatible microwave models."

## Consequences

- SRS v2 §1.1-§1.2 scope and the capability matrix stand as the PI's scope. The Method Matrix
  system class (5-10 atom complexes, CP-FTMW) is one use within it (errata row 18).
- A native SpycFit path exists beside genuine SPCAT/SPFIT (SF-CMP-001a/b); its terms are
  D-SPYC-18.
- Publication tables show the SPFIT result first when native and SPFIT results both exist
  (SRS v2 SF-OUT-006, proposed). The export at `e87f6f4` already keeps the two in separate
  tables (`publication_table.tex` from the fit state, `publication_table_spfit.tex` from SPFIT);
  the ordering follows SF-OUT-006 once it is approved.

## Alternatives not chosen

Option A (SPFIT/SPCAT as the only fitter, MM §2.1 scope) and option B (a native fitter of record
with SPFIT as a cross-check) were each partly adopted: both engines run, and SPFIT comes first in
publication tables. Option C (a PGOPHER front end) was not chosen. The PI gave no reasons.
