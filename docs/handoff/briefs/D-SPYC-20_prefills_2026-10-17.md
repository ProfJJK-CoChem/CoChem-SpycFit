## D-SPYC-20: pre-filled temperature, window and J_max

**Question.** Which values pre-fill the temperature, window and J_max boxes per technique, and how is J_max set?

**Short answer.** Per-technique values from the table; J_max computed by the rule below, uncapped. CP-FTMW, cavity and room-temperature cell rows rest on several groups, thin rows on one or two.

**Context**
- PI: boxes are set before predictions, editable, and pre-filled from publications (`docs/decisions/D-SPYC-20.md` lines 15-22). The 2026-10-11 proposal (1.5 K, 2–22 GHz, population rule capped at 30) was CP-FTMW only (`docs/handoff/PI_DECISIONS.md` lines 381-386). No CP-FTMW instrument we read spans 2–22 GHz (sample only).
- SPCAT's .int has no THRSH or MAXF field: line 2 holds QROT, FEND (an F limit), STR0/STR1, FQLIM and TEMP (default 300 K) (spinv.pdf). Intensities scale with the QROT given, so QROT must be Q(TEMP) (calcat.c line 175).
- SPFIT (fit) and SPCAT (predict) build from one JPL source set; DPFIT/DPCAT are the same programs linked with dpi.c, same .int fields (Makefile; calpgm.pdf). No DPCAT reference output was reproducible (JPL's no.var returned HTTP 500).
- JPL/CDMS intensities refer to 300 K, HITRAN to 296 K. The 300 K microwave / 296 K IR-NIR split below is an agent choice.
- Fixed caps cut data. 30 cuts cyanocoronene (J = 39 at about 2 K) and MTBE (J'' = 64, in a deliberately warmed 1-bar Ar jet, 7 K + 45 K). 100 cuts the rule's values: glycidaldehyde 149 at 300 K; rigid rotors 152–229 at 296 K (B̄ 0.1–0.05 cm⁻¹).

**Pre-fills.** Parentheses: range in the sources read; * agent proposal. Outside the range, warn, not block (the PI wants boxes changeable "as need be").

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

**J_max rule (proposed; no paper states it).** P(J) = Σ g·exp(−E/kT) over all predicted levels with rotational quantum number J (N if open-shell); g = (2J+1) × nuclear-spin weight; hyperfine components summed; E from the predicting engine. J_max is the smallest J above the peak with P(J) < 10⁻⁴ × max P. Predict every line with J ≤ J_max, then filter by the window.
- Tests: a linear rigid rotor, B̄ = 0.1 cm⁻¹, 296 K gives 152. Glycidaldehyde (CDMS parameters; an SPCAT build that reproduces the CDMS catalog byte for byte) gives 11, 19, 38 and 149 at 1.5, 5, 20 and 300 K.
- Keeping instead every line within 10⁻⁴ of the strongest (SPCAT to 2000 GHz) gives J_up 11, 21, 42 and 164.
- SPCAT: TEMP = T; FQLIM = window top; FEND = J_max plus the largest total nuclear spin if hyperfine is resolved; QROT = Q(T) summed to far larger J (papers: 200–400); STR0/STR1 off (−100) unless set, since JPL/CDMS cutoffs assume 300 K but SPCAT applies them at TEMP. With IRFLG = 1, FQLIM is in cm⁻¹ and the STR1 term keeps its 300 GHz scaling (calcat.c lines 176-190, 407).
- PGOPHER: Temperature = T, Jmax = J_max; Tvib defaults to T. XIAM/BELGI/RAM36 mappings: not researched. Condensed rows record J_max as "not applicable".

**Options.** Presets:
- **D20-TABLE**: per technique. Pro: follows each literature. Con: 13 rows. Risk: thin rows look solid.
- **D20-FAMILY**: jet microwave 2 K, gas cells 300/296 K, condensed 298 K. Pro: fewer numbers. Con: mm-wave and IR jets run at 5–50 K. Risk: silently wrong intensities.

J_max:
- **D20-JPOP**: the rule, uncapped. Pro: needs only T and energies; never below an observed J checked. Con: unpublished; at 300 K drops some lines within 10⁻⁴ of the strongest.
- **D20-JLINE**: the line-intensity definition, uncapped. Pro: keeps every line within 10⁻⁴. Con: needs a first intensity pass; depends on the frequency limit.
- **D20-JFIX**: fixed per technique (e.g. 30 jets, 150 contours). Pro: simple. Con: caps cut published data. Risk: lines beyond the cap look absent.

**Agent recommendation.** D20-TABLE with D20-JPOP: JPOP needs no intensities and, for glycidaldehyde, differs from JLINE by at most 4 at 20 K or below. Prefer JLINE if room-temperature completeness matters more. Record each value as "default" with its evidence grade, and label thin rows "provisional" until your own data replace them.

**How to answer:** `D-SPYC-20 pre-fills: D20-TABLE, D20-JPOP` (add changes, e.g. "jet mm-wave T = 10 K" or "all gas cells 300 K").
