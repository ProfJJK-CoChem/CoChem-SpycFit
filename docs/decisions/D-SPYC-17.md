# D-SPYC-17 SPFIT/SPCAT source, pin and use rights

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 (two replies) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17 (agent question and PI reply quoted below) |

## Decision

The agent asked, as item 1 of the list of decisions needed before the SRS can be completed:
"**D-SPYC-17** SPFIT/SPCAT source and pin (JPL sources or TORQ's laserkelvin copy), and whether
DPFIT/DPCAT are included".

The PI replied: "1. JPL sources. What are DPFIT/DPCAT?"

The agent explained DPFIT/DPCAT (Pickett's `calfit.c`/`calcat.c` linked with `dpi.c`, "PACKAGE
FOR DOUBLET PI", for molecules in a ²Π state such as NO and OH) and asked for "D-SPYC-17:
include DPFIT/DPCAT" or "exclude".

The PI replied: "include DPFIT/DPCAT"

The agent had reported that TORQ pins laserkelvin/Pickett and that "Either TORQ switches to the JPL
build, or the two builds need a check that they give the same catalogs." After a preliminary
comparison of the source copies, the PI replied: "Switch to the JPL source".

## Consequences

- SpycFit's SPFIT and SPCAT are built from Pickett's sources at
  `https://spec.jpl.nasa.gov/ftp/pub/calpgm/`, checked against a SHA-256 manifest. The files
  retrieved on 2026-10-11 are listed in `docs/handoff/calpgm_sources_2026-10-11.sha256`.
- The sources carry a Caltech copyright notice ("all rights reserved"). Sources and binaries are
  not committed to any CoChem repository; each installation downloads and builds them
  (SF-RUN-008).
- The CoChem-BASE build mechanism the PI asked for (conversation 2026-10-16: "We need to build
  sp_cat and sp_fit and add a mechanism that can build sp_cat and sp_fit into [CoChem-BASE] so
  that it can install properly") uses these sources.
- Conflict with TORQ, stated as AGENTS.md requires: TORQ pins the laserkelvin/Pickett mirror at
  `f5af2ff0f1acb1eb11ecffe7ef3b55d457801f7b`, and its validator rejects any other upstream URL
  (`CoChem-TORQ/src/cochem_torq/spectroscopy/pickett_backend.py`, `SUPPORTED_COMMIT` and the
  `upstream_url` check). The PI's reply "Switch to the JPL source" is read as moving TORQ to the
  JPL build, so that the ecosystem has one Pickett source. The TORQ change is made on TORQ's own
  branch and pull request; until it is merged, catalog parity between TORQ and SpycFit
  (D-SPYC-16) still needs a cross-build equivalence check.
- DPFIT and DPCAT are built with SPFIT and SPCAT from the same JPL download. The source manifest
  must add `dpi.c`, which the 2026-10-11 manifest omits. Their acceptance run uses JPL's NO
  example (SF-RUN-007). An agent run on 2026-10-17 did not reproduce the shipped 1993 NO outputs
  (DPFIT RMS ERROR 1.15158 against 1.1534; DPCAT 3,900 catalog lines against 1,131), and DPFIT
  crashed when the shipped `no.var` was used as its parameter file. These differences are listed
  in the acceptance record; until they are explained, DPFIT/DPCAT results are not used as
  evidence.
- Still open: where verified copies are kept if the JPL files change; whether `pyckett` is a
  runtime dependency or a test-only cross-check; the use rights to record under SF-RUN-008. The
  PI also asked how the available source copies differ and which is more reputable (brief in
  preparation).

## Alternatives not chosen

TORQ's laserkelvin/Pickett copy as SpycFit's source, and Kisiel's PROSPE binaries (option C),
were not chosen. Excluding DPFIT/DPCAT was not chosen. Whether the JPL build becomes the one ecosystem pin (option A with a jointly
chosen replacement) or a separate SpycFit pin (option B) depends on the TORQ question above.
