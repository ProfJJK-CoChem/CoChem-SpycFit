# D-SPYC-10 Structure determination

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17 (agent question and PI reply quoted below) |

## Decision

The agent asked, as item 17: "**D-SPYC-10** structure determination (r₀, r_s, r_m): include or
defer".

The PI replied: "17. Defer to GEOM which is the molecular geometry module."

## Consequences

- Structure determination (r₀, r_s, r_m and related fits) belongs to GEOM, the CoChem molecular
  geometry module. SpycFit has no structure-determination requirements.
- GEOM's plan already takes this role: its Stage 1.0 inputs are "Pickett files (.lin, .par,
  .fit), CoChem-SpycFit JSON payloads", and its fitter writes r₀, r_s, r_m, r_e and
  semi-experimental r_e structures, with "Analytical Kraitchman propagation"
  (`CoChem-GEOM/CoChem_GEOM_Workflow_Plan_v4.md` at `84d5af4`).
- SpycFit supplies GEOM with fitted rotational constants per isotopologue, with uncertainties,
  correlations, reduction, representation and isotopologue identity. The payload schema is to be
  agreed with GEOM and versioned (cross-repository item for the PI).
- Benchmark rows that needed structure output (review §8.4 rows 9-10) move to GEOM.

## Alternatives not chosen

Option A (r_s, r₀ and r_m fits in SpycFit, gated by fluxionality) and option C (SpycFit
delegating to Kisiel's STRFIT/KRA) were not chosen. Option B (defer) is taken in the form of
delegation to GEOM.
