# Pickett build and acceptance run, 2026-10-18 (local container, not CI)

Record of the run that checked the BASE build mechanism before draft PR
ProfJJK-CoChem/CoChem-BASE#14 was opened (D-SPYC-17, D-SPYC-26; SF-RUN-007, SF-RUN-009,
SF-RUN-009a, V-PICK-05). Host: Ubuntu 24.04 container, gcc 13.3.0, GNU make. Nothing below is
CI; the BASE workflow `pickett_build.yml` repeats the run on GitHub-hosted runners once the PR
runs there.

## Sources

Retrieved 2026-10-18 01:40 UTC from `https://spec.jpl.nasa.gov/ftp/pub/calpgm/`. All 26 source
files and 11 example files matched the pin in `CoChem-BASE/scripts/pickett-distribution.json`
(SHA-256 and size per file). The 25 files shared with TORQ's pin match TORQ's table; the 31 files
shared with `docs/handoff/calpgm_sources_2026-10-11.sha256` match it. Additions since that
manifest:

| File | SHA-256 | Bytes |
| :--- | :--- | ---: |
| dpi.c | a0e7cccff1ff981313b57efcf4ff906f130f0957b82eb6ffd89bcf76df66fdde | 17133 |
| hcotst.par | 76b4c910576009cefecf116b368177060195a067ac16554b2c5381123ecf441d | 1160 |
| README | 278b657d8cae34745ef324393a917b80a9c6e362205e399afd6188866096ec8b | 1396 |
| calpgm.pdf | 48796ff50df151916041aa4a8a68ec3a408a9982098265adc872faf79e49e8fb | 76497 |
| spinv.pdf | 4dd04df5c2fa82e87e4ff0a6942a12e5d7d710be01b09e66f37c6952e2d488e9 | 226739 |
| whats.new | 245713d299a7d89af08c633ef7c1a30f278b0ac935ab4ba60e7aa9006bd1fe0a | 5860 |

`hco.var`, `hcotst.var` and `no.var` returned HTTP 500 (also on 2026-10-17).

## Build

`make -j1 spfit spcat dpfit dpcat` with the shipped Makefile (`CFLAGS=-O2 -Wall`, `BLASLIB`
empty, `dblas.o` supplied): exit 0, 5.5 s wall. Products are ELF x86-64; their hashes depend on
the compiler and libc and are recorded in the receipt, not pinned.

## Acceptance

HCO (library SPINV; route of D-SPYC-26 "Y"): `hcotst.par` copied as `hco.par` with `hco.lin` and
`hco.int`; `spfit hco` exit 0; `spcat hco` exit 0. The generated `hco.cat` is byte-equal to the
shipped `hco.cat` (745 lines). The generated `hco.fit` iteration summary lines equal the shipped
file's: "MICROWAVE RMS = 0.751457 MHz" and "END OF ITERATION 1 OLD, NEW RMS ERROR= 7.52262
0.74456", then "MICROWAVE RMS = 0.070444 MHz" and "END OF ITERATION 2 OLD, NEW RMS ERROR= 0.74827
0.74730". Listed difference: the shipped `hco.var` was unavailable, so SPFIT wrote its own
`hco.var` from `hcotst.par` before SPCAT ran. Receipt status: `spfit` and `spcat` `accepted`.

NO (library DPI): `no.par`, `no.lin`, `no.int`; `dpfit no` exit 0; `dpcat no` exit 0. Listed
differences, as D-SPYC-17 recorded: the generated `no.cat` has 3918 lines against 1131 shipped
and differs from the first line; `no.fit` ends "END OF ITERATION 3 OLD, NEW RMS ERROR= 1.15158
1.15158" with MICROWAVE RMS 0.045678 MHz, where the shipped 1993 file reads "END OF ITERATION 1
... 1.1534 1.1534" with MICROWAVE RMS 0.45684E-01 MHz. Receipt status: `dpfit` and `dpcat`
`built`; their output is not evidence until the differences are explained.

## Open

The PI's use-rights sentence for `pi_use_rights` (SF-RUN-008); the TORQ inventory and
documentation changes; the CI run of `pickett_build.yml`.
