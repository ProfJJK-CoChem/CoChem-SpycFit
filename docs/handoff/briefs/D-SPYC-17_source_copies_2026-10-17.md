# SPFIT/SPCAT sources compared

**Question.** "What is the difference between SPCAT and SPFIT sources. Which is more reputable." (PI, 2026-10-17)

**The programs.** SPFIT fits parameters to assigned lines (.par + .lin → .fit, .var, new .par). SPCAT predicts lines from parameters (.var + .int → .cat, .out, optionally .egy, .str). All four share one source set (DPFIT/DPCAT use dpi.c for spinv.c), so provenance is identical.

**The copies.** One code base (change log ends 6 Nov 2007): JPL is the original, CDMS's C sources are identical, laserkelvin is a modified CDMS copy. JPL and CDMS build byte-identical programs, so choosing between them changes provenance and packaging, not results.

| Source | Who | Code vs JPL | Output vs JPL | Rights wording | Old versions |
|---|---|---|---|---|---|
| JPL calpgm | Pickett (Caltech/JPL); site manager B. Drouin | Original; also prebuilt binaries, 2014 .o files | Reference | "copyrighted, but made freely available as a service to the spectroscopic community"; Caltech header in 23 of 49 .c/.h | Fortran in legacy/ only |
| CDMS quelle | H. S. P. Müller, Cologne | Identical (no ftran.h, iamcalc only); Makefile -g, no -O; Makefile.txt = JPL's | Same binaries at -O2 | None | Six dated archives, 2001–2007; examples |
| laserkelvin f5af2ff | Kelvin Lee; 4 commits, 2020-10-16 | calcat.c Q table; Makefile -g -O3 | .cat/.fit/.var identical; .out Q at 1000…1 K, without 37.5/18.75/9.375 K | No licence | Git history from 2007 code |
| PROSPE; SPIN | Z. Kisiel; L. Bonah (ISMO Orsay) | Binaries; SPIN's source zip = CDMS zip + own Makefile | Not run (task rule) | PROSPE: "The use of these programs is free" | None |

Also found: pyckett (wrapper; runs any spfit/spcat on the PATH); jtbr/spfit-spcat (C++ rewrite, no licence file, same executable names). No Homebrew, conda or PyPI package.

**Evidence**
- With `make CFLAGS="-O2 -Wall"`, JPL, CDMS and the CDMS zip give byte-identical spfit, spcat, dpfit, dpcat; laserkelvin matches on spfit and dpfit. All builds write identical .cat/.fit/.var/.par/.egy/.str for HCO and NO.
- laserkelvin's change cannot alter .cat: intensities use QROT from .int (calcat.c line 175). JPL catalogue QLOG uses the three dropped temperatures.
- Version matters, copy does not: CDMS's archived March 2007 build drops 68 of 745 HCO lines and changes 668 intensities.
- Standing: Kisiel (2024): current versions "are available from http://spec.jpl.nasa.gov"; BASE cites the JPL URL; CDMS uses SPFIT/SPCAT for most predictions. Today's builds reproduce JPL's 2018 hco.cat byte for byte and CDMS's NO entry c030517 within 0.0001 MHz. 15 sampled papers (2021–2026, preprint wording) cite Pickett 1991; none names its copy or version. We found no JPL or CDMS statement naming a canonical copy.
- JPL today: hco.var and no.var return HTTP 500 (02:23 UTC); the 2026-10-11 copies match the manifest. SPFIT on hcotst.par + hco.lin, then SPCAT, reproduces hco.cat byte for byte without hco.var. A mirror keeping server timestamps links the 2014 .o files; build from manifest files only.
- CDMS serves all 21 manifest build files (same SHA-256) and dpi.c, not the 12 HCO/NO examples. The manifest lacks dpi.c and hcotst.par.

**Shared by every copy**
- No redistribution grant; no published checksums; no logged code change since 2007.
- Three crash bugs: calfit.c line 263 (.par with a correlation matrix); subfit.c (.par lines over 80 characters); lsqfit.c line 471 (every line rejected).
- DPI is unvalidated. The 1993 (Fortran-era) NO example is not reproduced: DPFIT starts at RMS 133,595; the refit flips p, q, pD, qD and moves Ω=3/2←1/2 lines about −350 MHz. By code reading, dpi.c and Fortran dpi.f assign the parity sign differently when A > 0. On one line family, current DPCAT lies ~517 MHz and the 1993 catalogue ~167 MHz below CDMS's NO model (suggestive only). Which is right is UNVERIFIED.

**Already answered.** The PI said "JPL sources", "Switch to the JPL source" and "Yes, change TORQ" (docs/decisions/D-SPYC-17.md). TORQ main (e0019e7) pins laserkelvin; draft PR CoChem-TORQ#5 (05bf2b3) pins JPL. TORQ reads .cat, .var and .fit, never .out.

**Options**

S1. JPL for the whole ecosystem (the recorded answers).
- Pros: the original; one pin; BASE already cites it.
- Cons: .var files unavailable today; prebuilt objects beside sources; no older C versions.
- Risks: a JPL outage blocks fresh builds.

S2. JPL for SpycFit/BASE; TORQ stays on laserkelvin with an equivalence check.
- Pros: no TORQ change.
- Cons: reverses "Yes, change TORQ"; two pins; modified, unlicensed copy.
- Risks: the check needs upkeep; later use of .out Q diverges.

S3. CDMS as the pinned source.
- Pros: identical code; dated archives; examples.
- Cons: reverses "JPL sources"; unoptimised Makefile; no HCO/NO examples.
- Risks: a wrong Makefile silently changes flags, not output.

S4. PROSPE/SPIN binaries (D-SPYC-17 C, not chosen): Windows builds need unpublished edits; the Linux "unmodified" claim is unchecked.

**Agent recommendation.** S1, pinned by a SHA-256 manifest of .c/.h files and the JPL Makefile, adding dpi.c and hcotst.par. Main risk: JPL availability. CDMS serving the same hashes as a second official URL needs the PI's ruling, since AGENTS.md forbids substituting cached files for missing inputs. Second risk: DPI for A > 0; record its results as unvalidated until a current reference case exists. Ask JPL for written redistribution terms before anything bundles binaries.

**How to answer:** "D-SPYC-17 source copy: S1/S2/S3; CDMS as second URL for identical hashes: yes/no; DPI results unvalidated until a reference case: yes/no."
