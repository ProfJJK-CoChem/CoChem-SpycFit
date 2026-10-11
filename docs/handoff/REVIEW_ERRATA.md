# Errata and overrides for the SRS review

The review `docs/wiki/review/CoChem-SpycFit_SRS_Review.md` (2026-10-11) used the raw Method
Matrix v4 as its main reference. The CoChem-TORQ repository holds an **accepted** implementation
contract, `docs/wiki/Method_Matrix_Implementation_Contract.md`, that corrects parts of the
Method Matrix, and an accepted decision register, `docs/wiki/review/Decisions_and_Risks.md`.
The review did not consult either. Where they conflict, the accepted TORQ documents win unless
the PI decides otherwise. Line numbers below refer to the review file.

Apply these corrections before copying any proposed requirement text from the review into
SRS v2.

| # | Review location | What the review says | Correction | Source |
| :--- | :--- | :--- | :--- | :--- |
| 1 | l.115, l.198, l.261, l.367 (SPYC-DIAG-002, PHYS-04, GAPS-11) | A sign disagreement between computed and experimental inertial defects is a **binding** rejection (`STRUCTURE_REJECTED`), applied before any rotational-constant comparison. | Compute and always show Δ and the planar moments. Flag a sign disagreement prominently, and never reject a structure or conformer automatically on it. Ground-state defects carry vibrational contributions and can legitimately differ in sign from equilibrium values. | TORQ contract §3.1; decision D-SPYC-12 |
| 2 | l.247, l.393 (SPYC-DIP-002, PHYS-18) | Transitions on an axis with \|μ_g\| < 0.1 D are **excluded** from the predicted line list and from assignment candidates. | De-prioritise and label dark-axis transitions in search and display. Keep them in every archival catalog. The 0.1 D threshold is a user setting, not a deletion rule. | TORQ contract §3.4; D-SPYC-13 |
| 3 | l.207, l.212, l.379, l.596, l.844, l.901 (SPYC-HAM-021, SPYC-IR-004, PHYS-15, benchmark row 6) | Tunnelling splittings carry a "factor-3" band; V3 predictions carry a "±14 %" band. | Neither is a universal or calibrated interval. Carry the uncertainty the TORQ bundle declares, labelled calibrated or uncalibrated. When none is declared, show the value as an uncalibrated estimate with no band. MM numbers may be cited as literature examples only. | TORQ contract §3.2-3.3 |
| 4 | l.291 (SPYC-WIN-001, SCOPE-12) and the assignment requirements in §5 | Default search windows come from the MM §3.1 Product A/B/C accuracy bands. | MM bands are uncalibrated `[D]` statistics. A calibrated relative score q gives the interval B_ref ∈ [B_pred/(1+q), B_pred/(1−q)], not B_pred(1 ± q). Windows may default to the MM bands for display, labelled uncalibrated and user-overridable, until a calibrated q exists. They must never be used as acceptance tolerances. | TORQ contract §3.2; D-SPYC-14 |
| 5 | l.534 (keep table: `cochem_jensen_averager.py` "verify then keep") | Inverse-inertia averaging per MM §5.1. | Verify the module uses **matrix** inversion of each Eckart-aligned inertia tensor before averaging, not elementwise inversion. The copied `.docs` architecture spec says elementwise, which is wrong. The result must not be re-inverted and reported as a frequency. This averaging is a TORQ job; SpycFit consumes its result. | TORQ contract §3.1 |
| 6 | l.453 (SPYC-OUT-001 `.cat` layout `12I2`) | `.cat` quantum fields are 12I2. | Quantum fields use Pickett's letter encoding outside -9..99 (lowercase `a0` = -10 … `z9` = -269; uppercase `A0` = 100 … `Z9` = 359). Upper-state fields occupy columns 56-67 and lower-state fields 68-79. Parsers and writers must round-trip authentic files, not decode as decimal integers. | TORQ contract §4 |
| 7 | l.415, l.463, l.609, l.699 (SPYC-STR-003, SPYC-IF-002) | SpycFit reads a "basin-count / MS-group feasibility record" from the TORQ bundle. | TORQ-EXP-004 has no such field today. Treat it as a proposed addition to be agreed with TORQ. Until it exists, the fluxionality gate takes the information from the user. | TORQ SRS §12 |
| 8 | §8.5, §5.10 (V-CONSUMER) | The consumer test uses a reference bundle "produced by the actual pinned TORQ exporter". | The TORQ exporter does not exist yet (TORQ `src/` has no SpycFit references). V-CONSUMER stays blocked on both sides. SpycFit's first releases must work from Pickett files and literature data alone. | TORQ SRS V-CONSUMER; TORQ `src/` |
| 9 | §5.9, §5.11, §9 (pinned SPFIT/SPCAT in CI) | CI "installs the binaries". | The JPL sources carry "Copyright (C) 1989, California Institute of Technology" and no licence text. Do not vendor them. Download at CI time, verify against `docs/handoff/calpgm_sources_2026-10-11.sha256` and build with the shipped Makefile. The shipped `hco.cat` is not reproduced by a fresh SPFIT+SPCAT run (up to 0.36 MHz differences), so it is not a regression target until explained. | Handoff build check, 2026-10-11 |
| 10 | Throughout | Requirement ids in the recommendation columns of §4 use `SPYC-…` names proposed by individual lenses and overlap in places. | SRS v2 assigns final ids. Do not cite review ids as if they were requirements. | Review §4 note |
| 11 | §2, §8.1, §9.5 | The Method Matrix is the parent document. | Cite both `CoChem-BASE/Method_Matrix.md` v4 and the TORQ implementation contract. TORQ's copy of the Method Matrix is byte-identical apart from a banner marking it superseded for TORQ implementation by that contract. | TORQ `docs/wiki/Method_Matrix.md` |
| 12 | Conformer handling (§4.9 GAPS-20, §5.5) | Multiple conformers overlaid with scale factors. | Consistent with the TORQ contract, with one addition: a conformer mixture is a **sum of spectra**, never a spectrum from averaged constants. An exchange-averaged spectrum needs a separately validated dynamical model. | TORQ contract §3.1 |

## Items the review got right that need action elsewhere

- **Method Matrix citation error.** The Method Matrix credits the SO₂⋯HF spin-spin and D
  quadrupole paper (J. Chem. Phys. 85, 3180, 1986; DOI 10.1063/1.451829) to "Andrews,
  Taleb-Bendiab, LaBarge, Hillig & Kuczkowski". The authors are A. Fillery-Travis and A. C.
  Legon. It appears in MM §3.1 (spin-spin row), §6.6 and reference 15, in both the BASE and TORQ
  copies.
- **Agent instructions.** The review recommended replacing `.github/copilot-instructions.md`.
  The handoff commit did this: the old roster is preserved at
  `docs/legacy/copilot-instructions_2026-08.md`, and `AGENTS.md` now carries the rules.
