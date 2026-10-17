# D-SPYC-20 Domain defaults (T_rot, frequency window, J_max)

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 (two replies) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17 (agent question and PI reply quoted below) |

## Decision

The agent asked, as item 10: "**D-SPYC-20** default temperature, frequency window and J_max,
now needed per technique (the old defaults were CP-FTMW only)".

The PI replied: "10. These should be set at the beginning before the predictions start
happening, and then they should be boxes that can be modified and changed by the user as need
be."

The agent then asked: "**D-SPYC-20:** should the three boxes start empty, or come pre-filled
with suggested values for each technique?"

The PI replied: "20 prefilled. Research default values from publications"

## Consequences

- Temperature, frequency window and J_max are set when a project or session is set up, before
  any prediction is generated.
- They are editable input fields that the user can change at any time. A change makes every
  prediction, window and catalog made with the old values stale until it is regenerated.
- The values used are recorded in the provenance of every prediction, catalog and fit, with
  whether each value came from the user or from a preset.
- The fields are pre-filled with per-technique values drawn from published practice. The user
  can change every pre-filled value, and provenance records that it was a pre-filled default.
- Still open: the pre-filled values themselves. The PI asked for them to be researched from
  publications. A value counts as a default under AGENTS.md only once this record names it.

## Alternatives not chosen

The fixed CP-FTMW defaults proposed in `docs/handoff/PI_DECISIONS.md` (T_rot 1.5 K, 2-22 GHz,
J_max from population) were not adopted as fixed values.
