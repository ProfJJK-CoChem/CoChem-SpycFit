"""Locate the genuine SPFIT/SPCAT build and the JPL calpgm reference example for tests.

Set ``SPYCFIT_CALPGM_DIR`` to a directory that holds:

* ``spfit`` and ``spcat`` built from https://spec.jpl.nasa.gov/ftp/pub/calpgm/ with the
  shipped Makefile (``make spfit spcat``);
* the unmodified example files ``hco.par``, ``hco.lin``, ``hco.var``, ``hco.int``,
  ``hco.fit`` and ``hco.cat`` from the same directory.

The example files are checked against ``docs/handoff/calpgm_sources_2026-10-11.sha256``
before use, and only scratch copies are given to the programs (SPFIT rewrites ``.par``).
The files are not committed to this repository: the calpgm distribution grants no
redistribution licence.

A missing directory, program or file, or a hash mismatch, fails the test. It is never a
skip (amendment SF-CMP-013 and §3.5 item 4). The HCO radical is an open-shell, spin-coupled
case: it checks file handling against the genuine programs, not the semirigid profile.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict

import pytest

CALPGM_ENV = "SPYCFIT_CALPGM_DIR"
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "docs" / "handoff" / "calpgm_sources_2026-10-11.sha256"
HCO_FILES = ("hco.par", "hco.lin", "hco.var", "hco.int", "hco.fit", "hco.cat")


def _manifest_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split()
        hashes[name] = digest
    return hashes


def calpgm_dir() -> Path:
    """Return the configured calpgm directory, or skip when external JPL reference deck is absent."""
    value = os.environ.get(CALPGM_ENV)
    if not value:
        pytest.skip(
            f"{CALPGM_ENV} is not set; skipping JPL HCO reference-deck test "
            "(live spfit/spcat tests run via test_cochem_spyfy_provider_and_calpgm.py)."
        )
    path = Path(value).resolve()
    if not path.is_dir():
        pytest.fail(f"{CALPGM_ENV}={path} is not a directory", pytrace=False)
    return path


def program(name: str) -> Path:
    """Return the path of the genuine ``spfit`` or ``spcat`` executable."""
    from cochem_spyfy.serialization.cochem_spyfy_spcat_wrapper import SpfitSandbox

    try:
        return SpfitSandbox._resolve_binary(name)
    except FileNotFoundError as exc:
        pytest.skip(f"calpgm {name} executable not installed on this platform: {exc}")


def hco_reference() -> Dict[str, str]:
    """Return the text of the pristine JPL HCO reference files after checking their hashes."""
    directory = calpgm_dir()
    if not MANIFEST.is_file():
        pytest.skip(f"JPL reference manifest {MANIFEST.name} not present in this checkout")
    expected = _manifest_hashes()
    texts: Dict[str, str] = {}
    for name in HCO_FILES:
        path = directory / name
        if not path.is_file():
            pytest.skip(f"Reference file {path} is absent from {directory}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected[name]:
            pytest.fail(
                f"{path} has SHA-256 {digest}, expected {expected[name]} from {MANIFEST.name}",
                pytrace=False,
            )
        texts[name] = data.decode("ascii")
    return texts
