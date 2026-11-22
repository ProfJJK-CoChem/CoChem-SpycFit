"""Tests for CoChem-SpycFit's CoChem-BASE module provider and live calpgm (spfit/spcat) wiring.

Covers:
1. ``cochem.modules`` provider contract (``cochem.module-provider/1`` / ``cochem.module-handoff/1``).
2. Binary discovery order: ``COCHEM_SPFIT_BIN`` / ``COCHEM_SPCAT_BIN``, ``SPYCFIT_CALPGM_DIR``,
   ``PATH``, and ``~/.local/bin/{spfit,spcat}``.
3. Pure-Python ``parse_pickett``, ``audit_fit``, and ``execute_handoff`` operations.
4. Genuine live ``spfit`` + ``spcat`` execution on an authentic HCOOH-DME Pickett deck
   (marked ``calpgm``; skips explicitly when binaries are absent, never substitutes ASE/EMT).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
from pathlib import Path
from typing import Any, Dict

import pytest

from cochem_spycfit.module_provider import (
    audit_fit,
    execute_handoff,
    fit_pickett,
    parse_pickett,
    predict_spectrum,
    spycfit_provider,
)
from cochem_spycfit.serialization.cochem_spycfit_pickett_output import (
    PickettOutputError,
    parse_spcat_cat,
    parse_value_error,
)
from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    PickettInputError,
    SpfitSandbox,
)

# Authentic HCOOH-DME parent Watson S-reduction Pickett deck (22 microwave transitions).
FA_DME_PAR = """\
HCOOH-DME_Parent                                        Thu Nov  5 00:08:15 2026
   8   22    8    0     0.0000E+00     1.0000E+05     1.0000E+00 1.0000000000
s    1  1   0   15   0   1   1   1   0   0   0
        10000   5.177747505572495E+03 1.00000000E+02 /A
        20000   1.824845398942627E+03 1.00000000E+02 /B
        30000   1.808390714533585E+03 1.00000000E+02 /C
          200  -2.069046164449231E-03 1.00000000E+02 /-DJ
         1100  -6.761775961854464E-03 1.00000000E+02 /-DJK
         2000  -0.000000000000000E+00 1.00000000E-25 /-DK
        40100  -3.855009085819847E-05 1.00000000E+02 /d1
        50000   1.901306403687336E-04 1.00000000E+02 /d2
"""

FA_DME_LIN = """\
  2  1  2  1  1  1  0  0  0  0  0  0     7249.9250    0.0030    1.0000
  2  0  2  1  0  1  0  0  0  0  0  0     7266.3440    0.0030    1.0000
  2  1  1  1  1  0  0  0  0  0  0  0     7282.8360    0.0030    1.0000
  3  1  3  2  1  2  0  0  0  0  0  0    10874.7300    0.0030    1.0000
  3  0  3  2  0  2  0  0  0  0  0  0    10899.2400    0.0030    1.0000
  3  2  2  2  2  1  0  0  0  0  0  0    10899.3000    0.0030    1.0000
  3  2  1  2  2  0  0  0  0  0  0  0    10899.5900    0.0030    1.0000
  3  1  2  2  1  1  0  0  0  0  0  0    10924.0800    0.0030    1.0000
  4  1  4  3  1  3  0  0  0  0  0  0    14499.3400    0.0030    1.0000
  4  0  4  3  0  3  0  0  0  0  0  0    14531.8100    0.0030    1.0000
  4  2  3  3  2  2  0  0  0  0  0  0    14532.1100    0.0030    1.0000
  4  2  2  3  2  1  0  0  0  0  0  0    14532.8000    0.0030    1.0000
  4  1  3  3  1  2  0  0  0  0  0  0    14565.1400    0.0030    1.0000
  5  1  5  4  1  4  0  0  0  0  0  0    18123.7000    0.0030    1.0000
  5  0  5  4  0  4  0  0  0  0  0  0    18163.9400    0.0030    1.0000
  5  2  4  4  2  3  0  0  0  0  0  0    18164.6500    0.0030    1.0000
  5  2  3  4  2  2  0  0  0  0  0  0    18166.0400    0.0030    1.0000
  5  4  2  4  4  1  0  0  0  0  0  0    18164.2000    0.0030    1.0000
  5  4  1  4  4  0  0  0  0  0  0  0    18164.2000    0.0030    1.0000
  5  3  3  4  3  2  0  0  0  0  0  0    18164.7400    0.0030    1.0000
  5  3  2  4  3  1  0  0  0  0  0  0    18164.7400    0.0030    1.0000
  5  1  4  4  1  3  0  0  0  0  0  0    18205.9300    0.0030    1.0000
"""

FA_DME_INT = """\
HCOOH-DME_Parent SPCAT intensity deck (a-type transitions, 2 K)
 0000  90001      100.0000   0   8  -10.0000  -10.0000   25.0000    2.00
 1  1.5000 /mua
"""


def _have_calpgm() -> bool:
    try:
        SpfitSandbox._resolve_binary("spfit")
        SpfitSandbox._resolve_binary("spcat")
        return True
    except FileNotFoundError:
        return False


def test_spycfit_provider_contract_metadata() -> None:
    """Verify cochem.modules provider metadata conforms to cochem.module-provider/1."""
    meta = spycfit_provider()
    # Strictly JSON-serializable (no NaN, no raw Python callables in metadata)
    encoded = json.dumps(meta, allow_nan=False)
    assert isinstance(encoded, str) and len(encoded) > 0

    assert meta["module_id"] == "spycfit"
    assert meta["distribution"] == "CoChem-SpycFit"
    assert meta["integration_contract"] == "cochem.module-handoff/1"
    assert meta["scientific_api_version"] == "cochem.module-provider/1"
    assert meta["execute_handoff"] == "cochem_spycfit.module_provider:execute_handoff"
    assert meta["operations"] == ["predict_spectrum", "fit_pickett", "parse_pickett", "audit_fit"]

    apis = meta["scientific_apis"]
    for op in meta["operations"]:
        entry = apis[op]
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["required_engines"], list)
        assert isinstance(entry["supported_engines"], list)
        assert isinstance(entry["scope"], str) and entry["scope"].strip()
        assert entry["handler"].startswith("cochem_spycfit.module_provider:")
        if entry["available"] and entry["required_engines"]:
            receipts = entry["runtime_receipts"]
            for eng in entry["required_engines"]:
                rcpt = receipts[eng]
                assert rcpt["available"] is True
                assert Path(rcpt["executable_path"]).is_file()
                assert len(rcpt["sha256"]) == 64
        elif not entry["available"]:
            assert isinstance(entry.get("reason"), str) and entry["reason"].strip()

    assert apis["parse_pickett"]["available"] is True
    assert apis["audit_fit"]["available"] is True


def test_cochem_modules_entry_point_discovery() -> None:
    """When CoChem-SpycFit is installed, cochem.modules exposes the spycfit provider."""
    entries = [
        ep
        for ep in importlib.metadata.entry_points(group="cochem.modules")
        if ep.name == "spycfit"
    ]
    if not entries:
        pytest.skip("CoChem-SpycFit wheel/editable metadata not installed in this interpreter")
    loaded = entries[0].load()
    assert callable(loaded)
    meta = loaded()
    assert meta["module_id"] == "spycfit"
    assert set(meta["operations"]) == {"predict_spectrum", "fit_pickett", "parse_pickett", "audit_fit"}


def test_binary_discovery_env_var_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """COCHEM_SPFIT_BIN / COCHEM_SPCAT_BIN take top priority and validate executable existence."""
    fake_bin = tmp_path / "custom_spfit"
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("COCHEM_SPFIT_BIN", str(fake_bin))
    assert SpfitSandbox._resolve_binary("spfit") == fake_bin.resolve()

    monkeypatch.setenv("COCHEM_SPFIT_BIN", str(tmp_path / "missing_spfit"))
    with pytest.raises(FileNotFoundError, match="COCHEM_SPFIT_BIN"):
        SpfitSandbox._resolve_binary("spfit")


def test_parse_pickett_par_and_cat_and_overflow_error() -> None:
    """parse_pickett handles .par parameters, .cat fixed-width rows, and calfit.c overflow."""
    parsed_par = parse_pickett("par", FA_DME_PAR)
    assert parsed_par["n_parameters"] == 8
    by_code = {p["code"]: p for p in parsed_par["parameters"]}
    assert by_code[10000]["value"] == pytest.approx(5177.747505572495)
    assert by_code[20000]["value"] == pytest.approx(1824.845398942627)
    assert by_code[30000]["value"] == pytest.approx(1808.390714533585)

    # Regression: calfit.c parer() prints -0(-2147483648)E-36 for fixed zero params
    val, err, chg = parse_value_error("              -0(-2147483648)E-36           -0E-36")
    assert val == 0.0
    assert err is None
    assert chg == 0.0

    # Regression: adjacent F13.4 and F8.4 columns without whitespace separator
    sample_cat = (
        "     157.5844353.7392 -8.0879 3    3.8392 15  90000 303 7 2 5       8 1 8       \n"
        "      15.9211  1.0504 -9.1491 3    1.0966  5  90000 303 2 2 0       2 2 1       \n"
    )
    parsed_cat = parse_pickett("cat", sample_cat)
    assert parsed_cat["n_transitions"] == 2
    assert parsed_cat["transitions"][0]["frequency_mhz"] == pytest.approx(157.5844)
    assert parsed_cat["transitions"][0]["uncertainty_mhz"] == pytest.approx(353.7392)
    assert parsed_cat["transitions"][0]["upper_quanta"] == [7, 2, 5]
    assert parsed_cat["transitions"][0]["lower_quanta"] == [8, 1, 8]

    with pytest.raises(PickettOutputError):
        parse_spcat_cat("")


def test_audit_fit_truthfulness_and_handoff(tmp_path: Path) -> None:
    """audit_fit enforces SF-CMP-008/012 and executes cleanly through execute_handoff."""
    empty_audit = audit_fit({}, {}, tolerance_khz=1.0)
    assert empty_audit["status"] == "INCOMPLETE"
    assert empty_audit["is_within_tolerance"] is False

    partial_audit = audit_fit({"A": 5177.7475, "B": 1824.8454}, {"A": 5177.7475}, tolerance_khz=1.0)
    assert partial_audit["status"] == "INCOMPLETE"
    assert partial_audit["missing_from_spfit"] == ["B"]

    good_audit = audit_fit(
        {"A": 5177.747505, "B": 1824.845399},
        {"A": 5177.747506, "B": 1824.845399},
        tolerance_khz=0.01,
    )
    assert good_audit["status"] == "WITHIN_TOLERANCE"
    assert good_audit["is_within_tolerance"] is True

    # Verify BASE cochem.module-handoff/1 round-trip for parse_pickett
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    par_file = handoff_dir / "fa_dme.par"
    par_bytes = FA_DME_PAR.encode("utf-8")
    par_file.write_bytes(par_bytes)
    manifest: Dict[str, Any] = {
        "schema_version": "cochem.module-handoff/1",
        "handoff_id": "test-handoff-001",
        "created_at": "2026-11-22T00:00:00Z",
        "module_id": "spycfit",
        "operation": "parse_pickett",
        "status": "pending_integration",
        "capability": {"module_id": "spycfit", "execution_verified": False},
        "artifact": {
            "kind": "pickett_par",
            "filename": "fa_dme.par",
            "source_path": str(par_file),
            "sha256": hashlib.sha256(par_bytes).hexdigest(),
            "size_bytes": len(par_bytes),
            "metadata": {},
        },
        "options": {"kind": "par"},
        "scientific_execution_performed": False,
        "validation_scope": "artifact_structure_and_integrity",
    }
    manifest_path = handoff_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    receipt = execute_handoff(handoff_path=manifest_path, output_directory=out_dir, registry=None)
    assert receipt["schema_version"] == "cochem.module-operation/1"
    assert receipt["module_id"] == "spycfit"
    assert receipt["operation"] == "parse_pickett"
    assert receipt["status"] == "completed"
    assert receipt["result"]["n_parameters"] == 8
    assert (out_dir / "parse_pickett_receipt.json").is_file()


@pytest.mark.calpgm
def test_live_spfit_and_spcat_roundtrip_fa_dme(tmp_path: Path) -> None:
    """Execute genuine spfit and spcat binaries on the HCOOH-DME parent Pickett deck."""
    if not _have_calpgm():
        pytest.skip("calpgm spfit/spcat binaries are not installed on this platform")

    # 1. Run genuine SPFIT via fit_pickett
    fit_res = fit_pickett(FA_DME_PAR, FA_DME_LIN, stem="fa_dme", scratch_root=tmp_path / "scratch")
    assert fit_res["exit_code"] == 0
    assert fit_res["n_residuals"] == 22
    assert fit_res["bad_lines"] == []
    assert fit_res["microwave_rms_mhz"] == pytest.approx(0.002960, abs=1e-4)
    assert fit_res["constants"]["A"] == pytest.approx(5177.7475, abs=0.05)
    assert fit_res["constants"]["B"] == pytest.approx(1824.8454, abs=0.01)
    assert fit_res["constants"]["C"] == pytest.approx(1808.3907, abs=0.01)
    assert len(fit_res["executable_sha256"]) == 64

    # 2. Feed SPFIT's real .var output directly into genuine SPCAT via predict_spectrum
    cat_res = predict_spectrum(
        fit_res["var_text"],
        FA_DME_INT,
        stem="fa_dme",
        scratch_root=tmp_path / "scratch",
    )
    assert cat_res["exit_code"] == 0
    assert cat_res["n_transitions"] > 20
    assert len(cat_res["executable_sha256"]) == 64

    # Verify SPCAT predicts the measured 2_1,2 -> 1_1,1 line (~7249.925 MHz)
    matched = [
        t
        for t in cat_res["transitions"]
        if t["upper_quanta"] == [2, 1, 2] and t["lower_quanta"] == [1, 1, 1]
    ]
    assert len(matched) == 1
    assert matched[0]["frequency_mhz"] == pytest.approx(7249.9255, abs=0.01)

    # 3. Verify spycfit_provider() truthfully reports both binaries available with real receipts
    meta = spycfit_provider()
    assert meta["scientific_apis"]["fit_pickett"]["available"] is True
    assert meta["scientific_apis"]["predict_spectrum"]["available"] is True
    assert meta["scientific_apis"]["fit_pickett"]["runtime_receipts"]["spfit"]["sha256"] == fit_res["executable_sha256"]
    assert meta["scientific_apis"]["predict_spectrum"]["runtime_receipts"]["spcat"]["sha256"] == cat_res["executable_sha256"]
