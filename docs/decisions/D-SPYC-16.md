# D-SPYC-16 Catalog and Hamiltonian boundary with TORQ

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-16_2026-10-17.md` |

## Decision

The agent asked: "D-SPYC-16 — where do TORQ's rotational catalogs end and SpycFit's begin, and
should both use one shared SPCAT runner and one catalog-equivalence test?" with options D, E, A
and B. The PI selected: "D-SPYC-16: D — split by producer (Recommended)".

Option D as offered: TORQ owns catalogs from its computed constants; SpycFit owns every catalog it
makes, from any constants, labelled with their origin; consumed TORQ, BASE and user catalogs keep
their producer label, and BASE runner output never counts as SPCAT output. One shared test and one
parameter-ID/sign table.

## Consequences

- SRS v2 §1.3 and SF-IF-001 record the split by producer. SpycFit catalogues computed constants
  from the direct-input path (SF-IF-005a to 005c) and fitted or edited constants, each catalog
  labelled with its producer and the origin of its constants (SF-CMP-013, SF-PHYS-001a).
- TORQ keeps its spectral-catalog stage (TORQ-SPEC-001, D12) for its computed constants. The
  agent's reading of D12 under this split needs the TORQ maintainer's confirmation (cross-repository
  item for the PI).
- Catalogs from the same constants can differ in spin weights, partition function and
  uncertainties; the SRS must say which catalog sets assignment windows (agent proposal: the
  catalog SpycFit makes from the constants it is fitting, labelled as such).
- One shared catalog-equivalence test guards adapter drift: `.cat`, `.egy` and `.str` byte for byte
  on one binary, `.cat` and `.egy` within T1 across builds, `.out` only between same-source builds.
  Its fixtures are downloaded and SHA-256-checked (SF-DATA-001e), never committed. Each TORQ change
  needs a specific PI request and the TORQ maintainer's agreement.
- BASE's `spcat_runner` (in-process diagonaliser fallback at J_max = 5; d1/d2 mapping that differs
  from TORQ's) is a third producer whose output is never labelled SPCAT; this is a cross-repository
  item carried by the BASE build design.
- Option E (one shared runner and receipt) is to be revisited once BASE's build mechanism exists.

## Alternatives not chosen

E (as D plus one shared runner and receipt), A (split by origin, the 2026-10-11 sheet option) and
B (SpycFit makes all catalogs) were offered and not selected. Sheet option C (both implement
catalogs independently) was not offered because D-SPYC-17 chose one source.
