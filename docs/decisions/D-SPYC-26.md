# D-SPYC-26 The BASE build mechanism for SPFIT, SPCAT, DPFIT and DPCAT (task 11, BASE-BUILD)

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. The BASE change itself is a CoChem-BASE pull request that the PI requests from the BASE maintainer. |
| Answered | 2026-10-18 (UTC) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | pop-up question (`docs/handoff/pi_answers_2026-10-17_popups.md`, "BASE-BUILD"); brief `docs/handoff/briefs/BASE-BUILD_2026-10-17.md` |

## Decision

The pop-up stated that this is the PI's task 11, recorded by an agent from the 2026-10-16
conversation as "We need to build sp_cat and sp_fit and add a mechanism that can build sp_cat and
sp_fit into [CoChem-BASE] so that it can install properly" (`D-SPYC-17.md`), and that choosing any
option of the first question confirms that wording, with Other to correct it. The PI chose an
option, so the recorded wording stands.

The PI selected:

1. Placement: **"A: BASE owns the provisioner (Recommended)"**. Not chosen: "B: TORQ's existing
   script is the provisioner"; "C: per-module provisioners".
2. Hook: **"H1: automatic in the Codespaces setup script"**. Not chosen: "H2: opt-in command
   `cochem-cli pickett build` (Recommended)"; "H3: off; CI only".
3. BASE's in-process SPCAT fallback: **"F1: remove it in the same BASE PR (Recommended)"**. Not
   chosen: F2 (separate PR); F3 (leave, record as never-SPCAT).
4. The HCO acceptance check while JPL's `hco.var` is unavailable: **"Y: yes, recorded as a listed
   difference (Recommended)"**. Not chosen: "N: wait for hco.var".

## Consequences

- CoChem-BASE owns the Pickett provisioner and the JPL pin: a provision script, a
  `pickett-distribution.json` manifest listing the JPL calpgm files with SHA-256 (the 2026-10-11
  manifest plus `dpi.c` and `hcotst.par`), a receipt `cochem.pickett-installation/1` (source
  inventory and digest; build command, compiler, flags, build-log hash, OS, architecture, libc;
  per-program path, SHA-256 and status `built` or `accepted`; the quoted rights notice and the
  PI's use-rights statement; the HCO and NO acceptance results), `cochem-cli pickett
  build|verify|status`, and a CI workflow that builds the four programs with
  `make -j1 spfit spcat dpfit dpcat`, commits no binary and records the acceptance results
  (SF-RUN-007, SF-RUN-009). TORQ's source inventory follows BASE's pin; a second receipt schema
  for TORQ needs the PI's request to the TORQ maintainer.
- The build runs automatically in BASE's Codespaces setup script. Conflict stated as AGENTS.md
  requires: TORQ's accepted text says "Automatic student-image provisioning remains blocked until
  the laboratory establishes appropriate distribution/use rights"
  (`CoChem-TORQ/docs/development/pickett_backend.md`, lines 46-48 on 2026-10-17). The PI, as the
  laboratory, chose automatic provisioning; the receipt therefore carries the rights basis
  (Caltech's "all rights reserved" notice in the sources; JPL's statement that the programs are
  "made freely available as a service to the spectroscopic community", read 2026-10-17), and the
  PI's own use-rights sentence for SF-RUN-008 is still to be given. The TORQ sentence is amended
  only when the PI requests it from the TORQ maintainer. A failed download or build disables only
  the dependent operations (BASE's optional-engine rule) and never stops image creation; no
  cached copy or in-process model stands in for the programs.
- BASE's in-process SPCAT fallback in `spcat_runner.py` is removed in the same BASE pull request;
  BASE's dipole-code mapping follows the shared parameter-code table (D-SPYC-16).
- V-PICK-05 and SF-RUN-007 are satisfied for HCO by SPFIT on `hcotst.par` + `hco.lin` followed by
  SPCAT reproducing `hco.cat` byte for byte, recorded in the receipt as a listed difference from
  the shipped `.var` route while `hco.var` returns HTTP 500 at JPL. If `hco.var` becomes available
  and differs, the record is amended and the check rerun. The NO differences stay listed
  (`D-SPYC-17.md`); DPFIT and DPCAT stay `built` until they are explained.
- XIAM (XIAMALL.FOR) and ERHAM follow in a second BASE pull request once the PI has their terms
  (D-SPYC-04); that PR adds gfortran to the image.
- D-SPYC-16 option E (one shared runner and receipt) is revisited once the BASE mechanism exists.
- Cross-repository work for the PI to request: the CoChem-BASE pull request above (the BASE hash
  ring and `pytest.ini` are refreshed in the same PR); the TORQ inventory and receipt change; the
  TORQ documentation sentence on automatic provisioning.

## Alternatives not chosen

Placement B and C, hooks H2 and H3, fallback options F2 and F3, and waiting for `hco.var` (N).
