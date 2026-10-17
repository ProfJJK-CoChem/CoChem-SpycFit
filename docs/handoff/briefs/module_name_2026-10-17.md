**Re-check of the TOP-8 name candidates (2026-10-17)**

**How I checked**
- **PyPI:** For each name in both cases I queried `https://pypi.org/pypi/<name>/json` and `https://pypi.org/simple/<name>/`. All 16 returned 404. Known projects (numpy, spektral, specfit) returned 200, so the method works.
- **PyPI page check:** `https://pypi.org/project/<name>/` returns 200 for every name, so that page cannot show whether a project exists. This matches the first pass.
- **Near-names:** I downloaded the full PyPI index (915,698 projects) and compared each name against it. None of the 8 has an exact or look-alike match (look-alikes being names that differ only by o/0, l/i/1 or punctuation). The near-names below differ by one letter.
- **GitHub:** API search and curl were blocked (403). I opened the github.com search page through WebFetch instead. Result counts match the first pass: SPAFT 2, SPYFT 3, SPYAF 2, SPEFT 3, SPEFY 5, ROFIT 44 (all 5 pages read, none scientific), SPECQ 55, SPEKT about 1.3k.

**Collisions the first pass missed or got wrong**
- **SPAFT**
  - PyPI one-letter near-names: spafe (audio features), SpaGFT (spatial transcriptomics), spft, saft (an ML fine-tuning method).
  - Missed: SAFT is the Statistical Associating Fluid Theory equation of state in chemical thermodynamics. I saw this in search results only and did not open the sources.
- **SPYFT**
  - Got wrong: girpas-ulg/spyfit is not recently active. The newest commit on master is dated 15 July 2016. The README says it "hasn't been released yet". The first pass took the "last updated 30 May 2023" date as activity.
  - Missed PyPI near-names: syft (PySyft, latest release 2026-08-26), swyft, shyft, spft, spyfi, pyft.
- **SPEKT**
  - Missed PyPI near-name spektro (v0.0.3, 2023): "A Python package for hyperspectral/multispectral imaging research".
  - Also on PyPI: spektra, Spektrum, spekk, speks, spekti, spekter.
  - On GitHub page 1 the first pass did not mention andreavolpato/spektrafilm (spectral simulation of film photography, about 1k stars).
  - SpekPy (an X-ray tube spectrum toolkit) is another near-name.
- **SPYAF**
  - Missed PyPI near-name pyaf (Python Automatic Forecasting, v5.0).
  - The web search found only ETF tickers (SPYF, SPYA). UNVERIFIED: I did not open those pages.
- **SPEFT**
  - Missed: spefit on PyPI, "Fitting of Single Photoelectron Spectra". It belongs to the CTA consortium, is BSD-3 licensed, and v3.1.0 came out on 2026-08-10. It is an active spectral-fitting package one letter away.
  - Missed: peft (Hugging Face Parameter-Efficient Fine-Tuning, v0.21.2, released 2026-10-01) and scPEFT. SPEFT reads as "S-PEFT", and the ACL 2025 paper uses it that way.
- **SPEFY**
  - Missed PyPI near-names: spexy ("Spectral Exterior Calculus"), spey (particle-physics inference, latest release 2026-07-28), spepy.
- **ROFIT**
  - Missed: PROFIT, NASA HEASARC's "Tool for Visualizing and Modeling High-Resolution Spectra". It fits lines to Gaussians; v2.0.0 dates from 2007-08-07. It is one letter off and is a spectral-fitting tool.
  - Missed PyPI near-names: profit (response-model fitting), gofit ("Global Optimization for Fitting problems", latest release 2026-09-22), royfit (KM3NeT muon fit), refit, rfit, r3fit.
  - ROTFIT is now confirmed: Frasca et al., A&A 594, A39 (2016), DOI 10.1051/0004-6361/201628337, arXiv:1606.09149.
- **SPECQ**
  - Missed: specqp is also on PyPI, not only GitHub. It is v1.1.9, "Quick plotting, correcting and fitting of spectroscopic data".
  - Missed: SpectraQ(TM), quantitative-analysis software for FTIR and FT-NIR from Applied Instrument Technologies (press release 2012-07-31, read via Laser Focus World).
  - Also on PyPI: specql, speq, speck.

**Ranking, fewest to most risks**
1. **SPAFT:** No spectroscopy clash. The only near-name in chemistry is SAFT, which is one letter off.
2. **SPYAF:** No science clash. The near-names are pyaf and ETF tickers; the main problem is that it is awkward to say.
3. **SPEFY:** No spectroscopy clash. The near-names are minor (spexy, spey), but it can be read as a misspelling of "specify".
4. **SPEKT:** No exact match, but the stem is crowded with spectral tools (spektro, SPEKTR, SpekPy, Spekwin32).
5. **SPECQ:** A spectral-fitting package (specqp) and a trademark-marked FTIR product (SpectraQ) are both one step away.
6. **SPYFT:** It is SPFIT with one inserted Y, so native-fitter output could be mistaken for an SPFIT fit, which the project rules forbid. It also sounds like spyfit, although that project has been inactive since 2016.
7. **ROFIT:** Two spectral-fitting codes (PROFIT, ROTFIT) and several "*fit" packages are one letter away. It also collides with fitness apps and the word "profit".
8. **SPEFT:** An actively released spectral-fitting package (spefit) is one letter away, Hugging Face PEFT and the ML method name would crowd search results, and it looks like SPFIT.

### Name suggestions

| Name | Reads as | Pros | Cons | Risks |
|---|---|---|---|---|
| SPAFT | SPectral Assignment And FiTting; "spaft" | Says what the module does; free on PyPI; 2 unrelated GitHub repos | No Python cue; "SPA" reads as single-page app | One letter from SAFT (fluid-theory equation of state); near-names spafe, SpaGFT |
| SPYAF | SPectra in pYthon: Assign and Fit | Python cue; free on PyPI; no science clash | Hard to say; "spy" connotation | Looks like an ETF ticker (SPY); near-name pyaf (forecasting) |
| SPEFY | SPEctral Fitting in pYthon; "SPEF-ee" | Python cue; free on PyPI; no science clash | Meaning needs explaining; looks like "specify" | Near-names spfy, spexy, spey on PyPI |
| SPEKT | Clipped "Spektrum"; "spekt" | Real-word clip like TORQ and GEOM; free on PyPI | No fitting or Python cue | Crowded stem: spektral, spektr, spektro (spectral imaging), SPEKTR (X-ray spectra), Spekwin32 |
| SPECQ | SPECtral Quantum-number assignment; "spec-Q" | Q echoes TORQ; free on PyPI | Often said "speck"; no fitting cue | specqp (PyPI spectral fitting); SpectraQ(TM) FTIR software; two exact-name GitHub repos |
| SPYFT | SPectra in pYthon FiTting; "spy-fit" | Closest to SpycFit; Python cue; free on PyPI | "spy" connotation | One letter from SPFIT, so native-fitter output could be mistaken for SPFIT output; sounds like girpas-ulg/spyfit (FTIR, inactive since 2016) |

For all six, the PyPI JSON API and simple index return 404 in both cases, and none has a look-alike match in the PyPI index of 2026-10-17. Trademark registries (USPTO, EUIPO) were not searched, so trademark status is UNVERIFIED for every name.

Renaming touches the repository name (CoChem-SpycFit), the Python package cochem_spycfit and the pyproject name, and the docs and governance files, which need their own PI-approved PR. In CoChem-BASE it touches the module registry entry 'spycfit' (src/cochem_base/interfaces/module_registry.py:46), module_id='spycfit' in the interface bridges and resource_guard.py, and the SPYCFIT = "cochem_spycfit_silo" entry in cochem_system_config.py. BASE is read-only for agents, so that part needs a PI request. The SF- requirement ids stay unchanged.

Sources: https://pypi.org/pypi/spefit/json, https://pypi.org/pypi/specqp/json, https://pypi.org/pypi/peft/json, https://pypi.org/pypi/gofit/json, https://pypi.org/pypi/spektro/json, https://heasarc.gsfc.nasa.gov/docs/software/profit, https://arxiv.org/abs/1606.09149, https://github.com/girpas-ulg/spyfit/commits/master, https://github.com/0-ml/speft, https://www.laserfocusworld.com/biooptics/biophotonics-tools/article/14193552/analytical-software-for-ftir-and-ft-nir-by-applied-instrument-technologies, https://en.wikipedia.org/wiki/Statistical_associating_fluid_theory (search result only, not opened).
