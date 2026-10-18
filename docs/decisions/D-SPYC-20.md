# D-SPYC-20 Domain defaults (T_rot, frequency window, J_max)

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 (four replies) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17 (agent question and PI reply quoted below); the detail confirmation by pop-up, `docs/handoff/pi_answers_2026-10-17_popups.md` |

## Decision

The agent asked, as item 10: "**D-SPYC-20** default temperature, frequency window and J_max,
now needed per technique (the old defaults were CP-FTMW only)".

The PI replied: "10. These should be set at the beginning before the predictions start
happening, and then they should be boxes that can be modified and changed by the user as need
be."

The agent then asked: "**D-SPYC-20:** should the three boxes start empty, or come pre-filled
with suggested values for each technique?"

The PI replied: "20 prefilled. Research default values from publications"

The agent then sent a summary in the conversation, pointing to the full brief (a working copy,
committed unchanged as `docs/handoff/briefs/D-SPYC-20_prefills_2026-10-17.md`). The summary
offered D20-TABLE ("per-technique values as above") or D20-FAMILY ("one value per family (jets
2 K, gas cells 300/296 K, condensed 298 K)"), and three J_max options: D20-JPOP (the population
rule), D20-JLINE ("keep every line within 10⁻⁴ of the strongest") and D20-JFIX ("a fixed J_max
per technique"). It ended: "**To answer:** "D-SPYC-20 pre-fills: D20-TABLE, D20-JPOP", plus any
value changes, for example "jet mm-wave 10 K"."

The PI replied: "D-SPYC-20 pre-fills: D20-TABLE, D20-JPOP"

### Table the PI answered (D20-TABLE as presented)

Quoted from the summary. Ranges seen in papers are in brackets; * marks an agent proposal, not a
value from a paper.

| Technique | Temperature | Window | Third box | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| CP-FTMW jet | 2 K (0.4–10) | 2–18 GHz (2–40) | J_max rule | Moderate |
| Cavity FTMW jet | 2 K (0.5–10) | 2–26.5 GHz | J_max rule | Good |
| mm-wave jet | 7 K (5–20) | 60–95 GHz | J_max rule | Thin |
| Room-temperature cell, mm/sub-mm/THz | 300 K | 75–500 GHz; THz 500–1500 GHz | J_max rule | Good |
| LN₂-cooled cell | 130 K | as above | J_max rule | Thin |
| IR, gas cell | 296 K | band ±50 cm⁻¹* | J_max rule | Good |
| IR, jet | 10 K (6–50) | band ±50 cm⁻¹* | J_max rule | Moderate |
| Band contour, gas | 298 K | band ±50 cm⁻¹* | J_max rule; line width 1 cm⁻¹ | Moderate |
| NIR, gas cell | 296 K | 5000–6600 cm⁻¹ | J_max rule | Thin |
| Raman, gas | 296 K | vibrational 0–4200*, rotational 0–300* cm⁻¹ | J_max rule | Thin |
| Condensed IR / Raman / NIR | 298 K | IR 400–4000*, Raman 100–2000, NIR 4000–10000 cm⁻¹ | band width 10 cm⁻¹ (no J_max) | Thin |
| THz-TDS, condensed | 295 K (78 K if cryogenic) | 20–120 cm⁻¹ | band width 5 cm⁻¹ | Thin |

### Detail in the full brief that the summary did not show (confirmed by the PI on 2026-10-17)

The agent put this detail to the PI as a pop-up ("D-SPYC-20 detail confirmation"), listing the
P13/P14 row, the uniform-flow entry, the optional NIR intensity cutoff of 1×10⁻²⁵ cm/molecule at
296 K (off by default), the published ranges and key DOIs, and the line-width definitions (HWHM
for condensed IR/Raman/NIR, FWHM for THz-TDS and band contours), with the options "Confirm all as
listed (Recommended)", "Confirm all, but state every width as FWHM (condensed IR/Raman/NIR becomes
20 cm⁻¹)", "Confirm all except the P13/P14 row (computation-led profiles get no pre-fill)" and
"Confirm nothing yet; I will read the record first". The PI selected: "Confirm all as listed
(Recommended)". Each width box is labelled with its definition (HWHM or FWHM) so that nothing is
converted by hand.

The full brief's table, reproduced below, adds: a row for computation-led work and PES refinement
(P13, P14: "Row of the data fitted"); a uniform-flow entry in the mm-wave row; an optional NIR
intensity cutoff of 1e-25 cm/molecule at 296 K; published ranges for rows where the summary showed
none; key DOIs; and line-width definitions. The summary's "band width 10 cm⁻¹" is a Lorentzian
half width at half maximum (HWHM) in the brief, its "band width 5 cm⁻¹" a Lorentzian full width
at half maximum (FWHM), and its "line width 1 cm⁻¹" an FWHM, so the two condensed-phase entries
differ in definition by a factor of two.

|Profile|Technique|T (range)|Window (range)|Third box|Evidence|Key DOIs|
|---|---|---|---|---|---|---|
|P1–P4|CP-FTMW jet|2 K (0.43–10)|2–18 GHz (2–40)|Rule|Moderate: ~2 K (Ar; 3-bar Ne), ~1 K (other Ne setups); record carrier gas|10.1021/acs.jpca.5c00980; 10.1021/acs.jpclett.6c00413; 10.1021/acs.jpca.6c00891|
|P1–P4|Cavity FTMW jet|2 K (0.5–10)|2–26.5 GHz (2–40)|Rule|Good|10.3390/molecules28083419; 10.1103/js7k-3cxm|
|P1–P5|mm-wave jet or uniform flow|Jet 7 K (5–20); flow at its T (9.3–30.2)|60–95 GHz (59.6–385)|Rule|Thin: Bologna (5–10 K) counts once; Rennes jet ~7 K|10.1093/mnras/stag1769; 10.1051/0004-6361/202661229|
|P1–P5|Cell, room temperature|300 K (253–335)|75–500 GHz; THz 500–1500 (8.7–1600)|Rule|Good|10.1021/acsearthspacechem.4c00360; 10.3847/1538-4357/adfc62|
|P1–P5|Cell, LN2-cooled|130 K (113–133)|As above|Rule|Thin: one discharge cell|10.3847/1538-4357/ad2f9d|
|P6|IR cell (FTIR, synchrotron, laser)|296 K (81–300)|Band ±50 cm⁻¹*|Rule|T good|10.3390/molecules30224389; 10.3390/molecules30091992|
|P6|IR jet|10 K (6–50)|Band ±50 cm⁻¹*|Rule|Moderate; T_vib can exceed T_rot (180 vs 50 K)|10.1021/acs.jpca.3c02272; 10.3390/molecules28104165|
|P7|Band contour, gas|298 K (12–298)|Band ±50 cm⁻¹*|Rule (papers fix 140–200); FWHM 1 cm⁻¹ (0.64–1.28)|Moderate: two groups|10.1038/s42004-020-00447-1; 10.3390/molecules28104165|
|P12|NIR, gas cell|296 K (81–296)|5000–6600 cm⁻¹ (5000–6530)|Rule; optional cutoff 1e-25 cm/molecule at 296 K|Thin: one group; jets use the IR-jet row|10.3390/molecules29225276; 10.1016/j.jqsrt.2024.108905|
|P9, P10|Raman, gas|296 K (135–300)|Vibrational 0–4200*; rotational 0–300* cm⁻¹|Rule|Thin|10.3390/molecules26154523; 10.3390/molecules30214267|
|P8, P9, P12|Condensed IR, Raman, NIR|298 K (78–333)|IR 400–4000*; Raman 100–2000; NIR 4000–10000 cm⁻¹|Lorentzian HWHM 10 cm⁻¹ (4–10); NIR up to 2 quanta|Thin|10.1021/acs.jpca.5c04282; 10.1177/00037028211027951|
|P11|THz-TDS, condensed|295 K; 78 K if cryogenic (20–295)|20–120 cm⁻¹ (5–150)|Lorentzian FWHM 5 cm⁻¹ (5–8)|Thin|10.1021/acs.molpharmaceut.2c00509; 10.1021/acs.cgd.5c00007|
|P13, P14|Computation-led; PES refinement|Row of the data fitted*|—|—|Not a technique|—|

### J_max (D20-JPOP)

**J_max rule (proposed; no paper states it).** P(J) = Σ g·exp(−E/kT) over all predicted levels with rotational quantum number J (N if open-shell); g = (2J+1) × nuclear-spin weight; hyperfine components summed; E from the predicting engine. J_max is the smallest J above the peak with P(J) < 10⁻⁴ × max P. Predict every line with J ≤ J_max, then filter by the window.
- Tests: a linear rigid rotor, B̄ = 0.1 cm⁻¹, 296 K gives 152. Glycidaldehyde (CDMS parameters; an SPCAT build that reproduces the CDMS catalog byte for byte) gives 11, 19, 38 and 149 at 1.5, 5, 20 and 300 K.
- Keeping instead every line within 10⁻⁴ of the strongest (SPCAT to 2000 GHz) gives J_up 11, 21, 42 and 164.
- SPCAT: TEMP = T; FQLIM = window top; FEND = J_max plus the largest total nuclear spin if hyperfine is resolved; QROT = Q(T) summed to far larger J (papers: 200–400); STR0/STR1 off (−100) unless set, since JPL/CDMS cutoffs assume 300 K but SPCAT applies them at TEMP. With IRFLG = 1, FQLIM is in cm⁻¹ and the STR1 term keeps its 300 GHz scaling (calcat.c lines 176-190, 407).
- PGOPHER: Temperature = T, Jmax = J_max; Tvib defaults to T. XIAM/BELGI/RAM36 mappings: not researched. Condensed rows record J_max as "not applicable".

## Consequences

- Temperature, frequency window and J_max are set when a project or session is set up, before
  any prediction is generated.
- They are editable input fields that the user can change at any time. A change makes every
  prediction, window and catalog made with the old values stale until it is regenerated.
- The values used are recorded in the provenance of every prediction, catalog and fit, with
  whether each value came from the user or from a preset.
- The fields are pre-filled with per-technique values drawn from published practice. The user
  can change every pre-filled value, and provenance records that it was a pre-filled default.
- Once this record is DECIDED, the values in the table the PI answered are the named defaults
  under AGENTS.md. Each is recorded in provenance as a pre-filled default together with its
  evidence grade; rows graded "Thin" are shown to the user as provisional until the PI's own
  data replace them (as the summary and the brief proposed). A value outside its bracketed range
  produces a warning, not a block.
- J_max is computed by the D20-JPOP rule above, without a cap. Condensed-phase rows, where J_max
  does not apply, record J_max as "not applicable" and use the table's third-box entry (a
  Lorentzian line width; for condensed NIR, also "up to 2 quanta").
- The full-brief detail listed above is confirmed (2026-10-17): the P13/P14 row takes its
  pre-fills from the row of the data being fitted; a uniform flow runs at its own measured
  temperature; the NIR intensity cutoff is optional and off by default; each box shows its
  published range and the width definition it uses.
- Still open: how temperature and J_max map onto XIAM, ERHAM, BELGI and RAM36 inputs (not
  researched).
- The 300 K reference for microwave cells and 296 K for IR and NIR cells is an agent choice,
  following the JPL/CDMS (300 K) and HITRAN (296 K) conventions.

## Alternatives not chosen

The fixed CP-FTMW defaults proposed in `docs/handoff/PI_DECISIONS.md` (T_rot 1.5 K, 2-22 GHz,
J_max from population, capped at 30) were not adopted. D20-FAMILY (one preset per technique
family), D20-JLINE (line-intensity J_max) and D20-JFIX (fixed J_max per technique) were not
chosen. At the detail confirmation, stating every width as FWHM, dropping the P13/P14 pre-fill
row, and deferring confirmation were not chosen.
