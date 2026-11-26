"""Authentic live Multiwfn_noGUI QTAIM and NCI verification tests.

Zero-Mock Compliance:
- Executes genuine PySCF/ORCA wavefunctions and native Multiwfn_noGUI.
- Verifies fail-closed behavior when Multiwfn_noGUI is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cochem.qc.cochem_qtaim import (
    MultiwfnNCIResult,
    MultiwfnQTAIMResult,
    probe_multiwfn,
    resolve_multiwfn_binary,
    run_multiwfn_nci,
    run_multiwfn_qtaim,
)


def _find_orca_binary() -> Path | None:
    for var in ("COCHEM_ORCA_BIN", "ORCA_CMD", "ORCA_BIN"):
        val = os.environ.get(var, "").strip()
        if val and Path(val).is_file():
            return Path(val).resolve()
    found = shutil.which("orca")
    if found:
        return Path(found).resolve()
    default = (
        Path.home()
        / ".local"
        / "share"
        / "cochem-topos"
        / "workstation-20261008"
        / "engines"
        / "orca-native-v1"
        / "orca_6_1_1_linux_x86-64_shared_openmpi418"
        / "orca"
    )
    return default.resolve() if default.is_file() else None


def test_multiwfn_resolution_fails_closed_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Multiwfn_noGUI"):
        resolve_multiwfn_binary(env={}, search_defaults=False)

    missing_exe = tmp_path / "nonexistent_multiwfn"
    with pytest.raises(FileNotFoundError, match="absent or not executable"):
        resolve_multiwfn_binary(missing_exe)

    probe_absent = probe_multiwfn(env={}, search_defaults=False)
    assert probe_absent["available"] is False
    assert "Multiwfn_noGUI" in probe_absent["reason"]

    if probe_multiwfn()["available"]:
        bad_wfn = tmp_path / "water.txt"
        bad_wfn.write_text("not a wavefunction", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported wavefunction"):
            run_multiwfn_qtaim(bad_wfn)


def test_live_multiwfn_qtaim_and_nci_from_pyscf_molden(tmp_path: Path) -> None:
    probe = probe_multiwfn()
    if not probe["available"]:
        pytest.skip(f"Multiwfn_noGUI unavailable: {probe.get('reason')}")
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.tools import molden

    mol = gto.M(
        atom=(
            "O 0.000000 0.000000 0.000000; "
            "H 0.000000 0.757160 0.586260; "
            "H 0.000000 -0.757160 0.586260"
        ),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged is True

    molden_path = tmp_path / "h2o_pyscf.molden"
    molden.from_scf(mf, str(molden_path))
    assert molden_path.is_file() and pyscf.__version__

    qtaim = run_multiwfn_qtaim(molden_path, work_dir=tmp_path / "qtaim_run")
    assert isinstance(qtaim, MultiwfnQTAIMResult)
    assert qtaim.multiwfn_version
    assert len(qtaim.multiwfn_binary_sha256) == 64
    assert len(qtaim.wavefunction_sha256) == 64
    assert len(qtaim.cpprop_sha256) == 64
    assert len(qtaim.result_sha256) == 64
    assert qtaim.n_ncp == 3
    assert qtaim.n_bcp == 2
    assert qtaim.n_rcp == 0
    assert qtaim.n_ccp == 0
    assert qtaim.poincare_hopf_chi == 1
    assert qtaim.is_poincare_hopf_satisfied is True
    assert len(qtaim.topology_graph.bcps) == 2
    for bcp in qtaim.topology_graph.bcps:
        assert bcp.electron_density_au > 0.25
        assert bcp.laplacian_au < 0.0
        assert bcp.total_energy_density_h_au < 0.0
        assert bcp.interaction_classification == "COVALENT_SHARED"
        assert bcp.ellipticity >= 0.0
    assert all(set(pair) == {"O", "H"} for pair in qtaim.bcp_symbol_pairs)

    nci = run_multiwfn_nci(
        molden_path,
        work_dir=tmp_path / "nci_run",
        grid_quality=1,
    )
    assert isinstance(nci, MultiwfnNCIResult)
    assert nci.multiwfn_version == qtaim.multiwfn_version
    assert nci.n_grid_points > 1000
    assert nci.n_attractive_points + nci.n_vdw_points + nci.n_steric_points == nci.n_nci_points
    assert nci.min_rdg >= 0.0
    assert len(nci.nci_output_sha256) == 64
    assert len(nci.result_sha256) == 64


def test_live_multiwfn_qtaim_from_orca_gbw_conversion(tmp_path: Path) -> None:
    probe = probe_multiwfn()
    orca_bin = _find_orca_binary()
    if not probe["available"] or orca_bin is None:
        pytest.skip("Requires installed Multiwfn_noGUI and ORCA 6 binaries.")

    orca_dir = tmp_path / "orca_hf"
    orca_dir.mkdir()
    inp_path = orca_dir / "hf.inp"
    inp_path.write_text(
        "! B3LYP def2-SVP\n* xyz 0 1\nH 0.0 0.0 0.0\nF 0.0 0.0 0.917\n*\n",
        encoding="utf-8",
    )
    run_env = os.environ.copy()
    ld_parts = []
    if run_env.get("COCHEM_ORCA_LD_LIBRARY_PATH"):
        ld_parts.append(run_env["COCHEM_ORCA_LD_LIBRARY_PATH"])
    ld_parts.append(str(orca_bin.parent))
    if run_env.get("LD_LIBRARY_PATH"):
        ld_parts.append(run_env["LD_LIBRARY_PATH"])
    run_env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)
    proc = subprocess.run(
        [str(orca_bin), inp_path.name],
        cwd=str(orca_dir),
        capture_output=True,
        text=True,
        env=run_env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0
    gbw_path = orca_dir / "hf.gbw"
    assert gbw_path.is_file()

    qtaim = run_multiwfn_qtaim(gbw_path, work_dir=tmp_path / "qtaim_gbw")
    assert qtaim.converted_molden_sha256 is not None
    assert len(qtaim.converted_molden_sha256) == 64
    assert qtaim.n_ncp == 2
    assert qtaim.n_bcp == 1
    assert qtaim.poincare_hopf_chi == 1
    assert qtaim.is_poincare_hopf_satisfied is True
    assert len(qtaim.topology_graph.bcps) == 1
    assert set(qtaim.bcp_symbol_pairs[0]) == {"H", "F"}


def test_live_multiwfn_nci_detects_attractive_hbond_in_water_dimer(tmp_path: Path) -> None:
    probe = probe_multiwfn()
    if not probe["available"]:
        pytest.skip(f"Multiwfn_noGUI unavailable: {probe.get('reason')}")
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.tools import molden

    mol = gto.M(
        atom=(
            "O  0.000000  0.000000  0.000000; "
            "H  0.957000  0.000000  0.000000; "
            "H -0.240000  0.927000  0.000000; "
            "O  2.850000  0.000000  0.000000; "
            "H  3.250000  0.760000  0.420000; "
            "H  3.250000 -0.760000  0.420000"
        ),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged is True

    molden_path = tmp_path / "h2o_dimer.molden"
    molden.from_scf(mf, str(molden_path))

    qtaim = run_multiwfn_qtaim(molden_path, work_dir=tmp_path / "dimer_qtaim")
    assert qtaim.n_ncp == 6
    assert qtaim.n_bcp == 5  # 4 covalent O-H + 1 intermolecular O...H H-bond
    assert qtaim.poincare_hopf_chi == 1
    assert qtaim.is_poincare_hopf_satisfied is True
    classes = [b.interaction_classification for b in qtaim.topology_graph.bcps]
    assert classes.count("COVALENT_SHARED") == 4
    assert "NON_COVALENT_CLOSED_SHELL" in classes

    nci = run_multiwfn_nci(molden_path, work_dir=tmp_path / "dimer_nci", grid_quality=1)
    assert nci.n_nci_points > 0
    assert nci.n_attractive_points > 0
    assert nci.attractive_fraction > 0.0
    assert nci.n_attractive_points + nci.n_vdw_points + nci.n_steric_points == nci.n_nci_points

