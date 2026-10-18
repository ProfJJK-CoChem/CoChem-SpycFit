# D-SPYC-23 BASE's declared boundaries and the direct-input route (SF-IF-004)

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-18 (UTC; pop-up at the end of the 2026-10-17 session) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/SF-IF-004_2026-10-17.md` |

## Decision

The agent asked two questions in one pop-up.

1. "SF-IF-004 (1 of 2) — the three operation names CoChem-BASE declares for SpycFit
   (pickett_export, pgopher_export, spectroscopy_snapshot): which will SPYFY provide, under that
   name or a new one?" with options C, B, A and D. The PI selected: "SF-IF-004: C — provide none of
   the three names; SPYFY declares its own operations through cochem.modules (Recommended)".
2. "SF-IF-004 (2 of 2) — a conflict the brief found: SF-IF-002 says SpycFit receives work 'only
   through cochem.module-handoff/1 manifests', yet that manifest rejects the .par/.var/.lin/.int/.cat
   files that SF-IF-005a (direct input, your D-SPYC-19 answer) requires. Which route?" The PI
   selected: "Route ii: BASE ingests Pickett files and ORCA output with new multi-file handoff
   kinds; SF-IF-002 keeps its 'only' clause (Recommended)".

## Consequences

- SRS v2 SF-IF-004 now states the disposition: none of the three names is provided; SpycFit
  publishes its own operations through the `cochem.modules` entry point, and a test checks that the
  published operations exclude the three names. `pickett_export` is not provided because Pickett
  files are inputs (SF-IF-002a) and outputs (SF-IO-004a) under SRS names; `pgopher_export` is not
  provided because no requirement writes PGOPHER files and D-SPYC-01 did not choose a PGOPHER front
  end; `spectroscopy_snapshot` is not provided because publication packages are SF-PUB-001 and the
  legacy snapshot exporter's fate is D-SPYC-15.
- SF-IF-002 keeps its "only through manifests" clause. New SF-IF-002a: Pickett files (as a
  multi-file set) and ORCA output reach SpycFit as BASE handoff artifacts of input kinds that BASE
  ingests, so every direct input carries BASE's lineage record (D-SPYC-19: "CoChem-BASE though is a
  hard requirement cuz that has the ingestion and the unified GUI").
- Cross-repository items for the PI to request from BASE: input kinds for spectra, peak lists,
  Pickett files and ORCA output; a multi-file handoff (a fit needs `.par` plus `.lin`; today one
  artifact per manifest); an adapter and distribution-manifest entries for SpycFit; and, as a
  separate cleanup, removal of the three shells (`cochem_pyckett_bridge.py`,
  `cochem_pgopher_bridge.py`, `cochem_vibspyc_snap.py`), their test parameters and one documentation
  line. BASE's registry description "Spectroscopy fitting and export" is edited by the D-SPYC-22
  rename request.
- Direct input through BASE waits on BASE's release of those kinds; until then SF-IF-005a and 005b
  are blocked on BASE, and SpycFit records the absence, never a workaround.
- The architecture §1 BASE bullet and plan WP-IF follow.

## Alternatives not chosen

B (rename `pickett_export` as the Pickett input operation and `spectroscopy_snapshot` as the
publication-package operation, retire `pgopher_export`), A (accept all three names) and D (drop the
requirement) were offered and not selected. Route i (files supplied inside the SPYFY notebook under
hosting H2, relaxing SF-IF-002) was not selected.
