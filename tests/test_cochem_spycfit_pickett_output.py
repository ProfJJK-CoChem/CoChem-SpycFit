"""Genuine-engine tests for SPFIT/SPCAT execution and output parsing.

Requirements: SF-IO-004 (Pickett adapters with round-trip evidence), SF-CMP-011 (run
evidence), SF-CMP-012/013 (failures stay failures; results come from the named engine),
SF-RUN-004 (visible, diagnosable failures).

Evidence class: genuine external engine. Every expected value comes from a file that JPL
ships with calpgm (see tests/pickett_reference.py), never from numbers written into this
file. The HCO radical is spin-coupled, so these tests check file handling and execution,
not the semirigid comparison profile.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cochem_spycfit.serialization.cochem_spycfit_pickett_output import (
    PickettOutputError,
    parse_pickett_parameters,
    parse_spfit_fit,
    parse_value_error,
)
from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    PARAM_ID_TO_NAME_MAP,
    PickettInputError,
    PickettProgramError,
    SpfitSandbox,
)

from pickett_reference import hco_example, program


@pytest.fixture
def sandbox(tmp_path: Path) -> SpfitSandbox:
    return SpfitSandbox(
        scratch_root=tmp_path / "scratch",
        timeout_seconds=120.0,
        spfit_binary=program("spfit"),
        spcat_binary=program("spcat"),
    )


@pytest.fixture
def hco() -> dict:
    return hco_example()


def test_spfit_first_iteration_reproduces_shipped_fit(sandbox: SpfitSandbox, hco: dict) -> None:
    """Genuine SPFIT on the shipped hco.par/hco.lin reproduces the last iteration in the
    shipped hco.fit: same residual rows, parameter estimates and statistics."""
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="hco")
    shipped = parse_spfit_fit(hco["hco.fit"]).final
    ours = run.report.iterations[0]

    assert run.record.exit_code == 0
    assert run.report.bad_lines == ()
    assert len(ours.residuals) == len(shipped.residuals) > 0
    for mine, ref in zip(ours.residuals, shipped.residuals):
        assert mine.quantum_numbers == ref.quantum_numbers
        assert mine.obs == ref.obs
        assert mine.calc == ref.calc
        assert mine.obs_minus_calc == ref.obs_minus_calc
        assert mine.exp_err == ref.exp_err
    assert [p.text for p in ours.parameters] == [p.text for p in shipped.parameters]
    assert ours.microwave_rms_mhz == shipped.microwave_rms_mhz
    assert ours.microwave_avg_mhz == shipped.microwave_avg_mhz
    assert ours.rms_error_old == shipped.rms_error_old
    assert ours.rms_error_new == shipped.rms_error_new


def test_spfit_run_record_is_complete(sandbox: SpfitSandbox, hco: dict) -> None:
    """The run record names the executable, its digest, the inputs and every output file."""
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="hco")
    record = run.record

    exe = program("spfit")
    assert record.program == "spfit"
    assert Path(record.executable) == exe.resolve()
    assert record.executable_sha256 == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert record.argv[1] == "hco"
    assert record.inputs == {"hco.par": hco["hco.par"], "hco.lin": hco["hco.lin"]}
    assert {"hco.fit", "hco.var", "hco.bak", "hco.par", "hco.lin"} <= set(record.outputs)
    # SPFIT backs up the .par it read before rewriting it.
    assert record.outputs["hco.bak"] == hco["hco.par"]


def test_var_parameters_agree_with_fit_text(sandbox: SpfitSandbox, hco: dict) -> None:
    """Full-precision .var values round to the value(error) text SPFIT prints in the .fit."""
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="hco")
    printed = run.report.final.parameters
    assert [p.code for p in run.parameters] == [p.code for p in printed]
    for full, shown in zip(run.parameters, printed):
        mantissa = shown.text.split("(")[0]
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        exponent = shown.text.split(")")[1].split()[0] if ")E" in shown.text else ""
        last_digit = 10.0 ** (-decimals) * (10.0 ** int(exponent[1:]) if exponent.startswith("E") else 1.0)
        assert abs(full.value - shown.value) <= 0.5 * last_digit + 1e-15
        assert abs(full.error - shown.est_error) <= 0.5 * last_digit + 1e-15


def test_parse_fit_file_reports_spfit_statistics_without_parity(
    sandbox: SpfitSandbox, hco: dict, tmp_path: Path
) -> None:
    """Regression: at 7e42507 the parser could not read genuine output and set a parity
    flag from SPFIT's own RMS (amendment §5). It now reports SPFIT's statistics only."""
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="hco")
    (tmp_path / "hco.fit").write_text(run.record.outputs["hco.fit"], encoding="ascii")
    (tmp_path / "hco.var").write_text(run.record.outputs["hco.var"], encoding="ascii")

    parsed = SpfitSandbox._parse_fit_file_static(tmp_path / "hco.fit")
    final = run.report.final

    assert "parity_status" not in parsed
    assert parsed["parameter_source"] == "var"
    assert parsed["rmsd_khz"] == pytest.approx(final.microwave_rms_mhz * 1000.0)
    assert parsed["rms_error_old"] == final.rms_error_old
    assert parsed["n_lines"] == len(final.residuals)
    assert parsed["missing"] == []
    by_code = {p.code: p for p in run.parameters}
    assert parsed["constants"]["A"] == by_code[10000].value
    assert parsed["uncertainties"]["A"] == by_code[10000].error
    # SPFIT stores -DJ at code 200; the name says so.
    assert PARAM_ID_TO_NAME_MAP[200] == "-DJ"
    assert parsed["constants"]["-DJ"] == by_code[200].value


def test_parse_fit_file_without_var_uses_printed_precision(
    sandbox: SpfitSandbox, hco: dict, tmp_path: Path
) -> None:
    """Without a .var the parser falls back to nothing but the .fit text, and says so."""
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="hco")
    (tmp_path / "hco.fit").write_text(run.record.outputs["hco.fit"], encoding="ascii")

    parsed = SpfitSandbox._parse_fit_file_static(tmp_path / "hco.fit")
    assert parsed["parameter_source"] == "fit"
    assert parsed["constants"]["A"] == run.report.final.parameters[11].value


def test_spcat_reproduces_shipped_catalog(sandbox: SpfitSandbox, hco: dict) -> None:
    """Genuine SPCAT on the shipped hco.var/hco.int writes the shipped hco.cat byte for byte."""
    record = sandbox.run_spcat_files(hco["hco.var"], hco["hco.int"], stem="hco")
    assert record.exit_code == 0
    assert record.outputs["hco.cat"] == hco["hco.cat"]


def test_spfit_failure_is_preserved(sandbox: SpfitSandbox, hco: dict) -> None:
    """SPFIT without its .par exits non-zero; the failure and its record are kept."""
    with pytest.raises(PickettProgramError) as excinfo:
        sandbox.run_pickett("spfit", {"hco.lin": hco["hco.lin"]}, "hco")
    err = excinfo.value
    assert err.program == "spfit"
    assert err.record is not None
    assert err.record.exit_code not in (0, None)
    assert err.record.inputs == {"hco.lin": hco["hco.lin"]}


def test_dict_runners_refuse_incomplete_inputs(sandbox: SpfitSandbox) -> None:
    """Regression: at 7e42507 run_spfit wrote no option line to a .var that SPFIT does not
    read, and run_spcat left SPCAT waiting on standard input. Both now refuse."""
    constants = {"A": 1.0, "B": 1.0, "C": 1.0}
    with pytest.raises(PickettInputError, match="run_spfit_files"):
        sandbox.run_spfit(physical_constants=constants, uncertainties={k: 1e37 for k in constants})
    with pytest.raises(PickettInputError, match="run_spcat_files"):
        sandbox.run_spcat(physical_constants=constants, dipole_components={"mu_a": 1.0})


def test_parse_value_error_examples_from_shipped_fit(hco: dict) -> None:
    """value(error)[E±nn] texts as printed in the shipped hco.fit."""
    texts = [p.text for p in parse_spfit_fit(hco["hco.fit"]).final.parameters]
    value, error, _ = parse_value_error(next(t for t in texts if t.startswith("729365.908(127)")))
    assert value == pytest.approx(729365.908, abs=1e-9)
    assert error == pytest.approx(0.127, abs=1e-12)
    value, error, _ = parse_value_error(next(t for t in texts if ")E-03" in t))
    assert error > 0 and abs(value) < 1e-2


def test_parse_spfit_fit_rejects_text_without_iterations() -> None:
    """Robustness: text that never completed an iteration is not a fit report."""
    with pytest.raises(PickettOutputError, match="No completed SPFIT iteration"):
        parse_spfit_fit("TITLE\nLINES REQUESTED=   76 NUMBER OF PARAMETERS= 21 NUMBER OF ITERATIONS=  2\n")


def test_parse_pickett_parameters_counts_npar(hco: dict) -> None:
    """The .par parser returns exactly NPAR parameter lines, skipping the option line."""
    params = parse_pickett_parameters(hco["hco.par"])
    npar = int(hco["hco.par"].splitlines()[1].split()[0])
    assert len(params) == npar
    assert {10000, 20000, 30000, 200} <= {p.code for p in params}
