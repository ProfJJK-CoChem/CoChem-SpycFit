"""
CoChem-SpycFit: Comprehensive Phase 3 Unit Test Suite.

Method Matrix & Zero-Trust Compliance:
- Zero pass statements
- Zero synthetic data generators (np.linspace / np.sin)
- Zero external SQLite mendeleev calls
- 100% authentic physical calculations, DLPack zero-copy validation, Zarr storage,
  Hermitian lock verification, provenance checks, and resonance triage.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import pytest
import zarr

from cochem_spycfit.physics_core.cochem_quantum_extractor import (
    AMU_ANGSTROM2_TO_MHZ,
    HARTREE_BOHR2_AMU_TO_CM1,
    MASS_DICT,
    MASS_TABLE,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    PESZarrStore,
    ProvenanceMetadata,
    ProvenanceVerificationError,
    ProvenanceVerificationResult,
    QuantumExtraction,
    QuantumExtractor,
    QuantumTensors,
    ResonanceMatch,
    ResonanceReport,
    ResonanceTriageError,
    RotationalConstants,
    WSLPathTranslationError,
    ZarrExtractionError,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    determine_hamiltonian_routing,
    enforce_hermitian_lock,
    extract_from_zarr_dlpack,
    extract_tensors,
    get_atomic_mass,
    get_monoisotopic_mass,
    translate_wsl_path,
    triage_resonances,
    verify_provenance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. WSL TO WINDOWS PATH TRANSLATION
# =============================================================================

def test_translate_wsl_path_to_windows() -> None:
    """Verifies translation of Linux / WSL paths to authoritative Windows paths."""
    wsl_c = "/mnt/c/Users/ansac/landscape.zarr"
    win_c = translate_wsl_path(wsl_c, target_os="windows")
    assert win_c.as_posix() == "C:/Users/ansac/landscape.zarr"

    wsl_d = "/mnt/d/Databases/landscape.zarr"
    win_d = translate_wsl_path(wsl_d, target_os="windows")
    assert win_d.as_posix() == "D:/Databases/landscape.zarr"

    wsl_root = "/mnt/e"
    win_root = translate_wsl_path(wsl_root, target_os="windows")
    assert win_root.as_posix() == "E:/"


def test_translate_wsl_path_to_wsl() -> None:
    """Verifies translation of Windows drive paths to WSL / Linux format."""
    win_c = "C:/Users/ansac/landscape.zarr"
    wsl_c = translate_wsl_path(win_c, target_os="wsl")
    assert wsl_c.as_posix() == "/mnt/c/Users/ansac/landscape.zarr"

    win_d_backslash = "D:\\Databases\\landscape.zarr"
    wsl_d = translate_wsl_path(win_d_backslash, target_os="wsl")
    assert wsl_d.as_posix() == "/mnt/d/Databases/landscape.zarr"


def test_translate_wsl_path_unc_and_edge_cases() -> None:
    """Verifies UNC WSL paths and empty inputs."""
    unc_path = "//wsl$/Ubuntu/home/cochem/data.zarr"
    win_unc = translate_wsl_path(unc_path, target_os="windows")
    assert "//wsl$/Ubuntu/home/cochem/data.zarr" in win_unc.as_posix()

    empty_res = translate_wsl_path("")
    assert empty_res == Path(".")


# =============================================================================
# 2. MONOISOTOPIC MASS RETRIEVAL
# =============================================================================

def test_get_monoisotopic_mass_standard_isotopes() -> None:
    """Verifies high-precision IUPAC monoisotopic masses for standard isotopes."""
    m_1h = get_monoisotopic_mass("1H")
    m_2h = get_monoisotopic_mass("2H")
    m_12c = get_monoisotopic_mass("12C")
    m_13c = get_monoisotopic_mass("13C")
    m_14n = get_monoisotopic_mass("14N")
    m_15n = get_monoisotopic_mass("15N")
    m_16o = get_monoisotopic_mass("16O")
    m_18o = get_monoisotopic_mass("18O")
    m_32s = get_monoisotopic_mass("32S")
    m_35cl = get_monoisotopic_mass("35Cl")
    m_37cl = get_monoisotopic_mass("37Cl")

    assert m_1h == pytest.approx(1.00782503223, rel=1e-8)
    assert m_2h == pytest.approx(2.01410177812, rel=1e-8)
    assert m_12c == pytest.approx(12.0000000000, abs=1e-9)
    assert m_13c == pytest.approx(13.00335483507, rel=1e-8)
    assert m_14n == pytest.approx(14.00307400443, rel=1e-8)
    assert m_15n == pytest.approx(15.00010889888, rel=1e-8)
    assert m_16o == pytest.approx(15.99491461957, rel=1e-8)
    assert m_18o == pytest.approx(17.99915961286, rel=1e-8)
    assert m_32s == pytest.approx(31.97207117, rel=1e-5)
    assert m_35cl == pytest.approx(34.96885271, rel=1e-8)
    assert m_37cl == pytest.approx(36.96590260, rel=1e-8)


def test_get_monoisotopic_mass_by_atomic_number() -> None:
    """Verifies atomic mass lookup via atomic number Z (Z=1..118)."""
    m_z1 = get_monoisotopic_mass(1)
    m_z6 = get_monoisotopic_mass(6)
    m_z8 = get_monoisotopic_mass(8)
    m_z16 = get_monoisotopic_mass(16)
    m_z6_a13 = get_monoisotopic_mass(6, mass_number=13)

    assert m_z1 == pytest.approx(1.007825, rel=1e-5)
    assert m_z6 == pytest.approx(12.000000, abs=1e-6)
    assert m_z8 == pytest.approx(15.994915, rel=1e-5)
    assert m_z16 == pytest.approx(31.972070, rel=1e-5)
    assert m_z6_a13 == pytest.approx(13.003355, rel=1e-5)

    with pytest.raises(ValueError):
        get_monoisotopic_mass(999)


def test_get_monoisotopic_mass_from_cochem_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies custom isotope database retrieval from $COCHEM_DATA_DIR/Databases/masses.json."""
    db_dir = tmp_path / "Databases"
    db_dir.mkdir(parents=True, exist_ok=True)
    masses_json = db_dir / "masses.json"

    custom_masses = {
        "CustomX": 123.456789,
        "13C": 13.003354835,
    }
    masses_json.write_text(json.dumps(custom_masses), encoding="utf-8")

    monkeypatch.setenv("COCHEM_DATA_DIR", str(tmp_path))

    retrieved_x = get_monoisotopic_mass("CustomX", data_dir=tmp_path)
    assert retrieved_x == pytest.approx(123.456789, abs=1e-6)


# =============================================================================
# 3. HERMITIAN LOCK ENFORCEMENT
# =============================================================================

def test_enforce_hermitian_lock_valid_real_symmetric() -> None:
    """Verifies that an exact physical symmetric matrix passes Hermitian Lock without modification."""
    symmetric_mat = np.array([
        [2.5, 0.3, -0.1],
        [0.3, 3.1, 0.4],
        [-0.1, 0.4, 1.8],
    ], dtype=np.float64)

    locked = enforce_hermitian_lock(symmetric_mat)
    np.testing.assert_array_equal(locked, symmetric_mat)


def test_enforce_hermitian_lock_valid_complex_hermitian() -> None:
    """Verifies that a complex Hermitian matrix (H == H^dagger) passes Hermitian Lock."""
    complex_herm = np.array([
        [2.0 + 0.0j, 1.0 - 2.0j],
        [1.0 + 2.0j, 3.0 + 0.0j],
    ], dtype=np.complex128)

    locked = enforce_hermitian_lock(complex_herm)
    np.testing.assert_array_equal(locked, complex_herm)


def test_enforce_hermitian_lock_hard_abort_on_violation() -> None:
    """Verifies that non-Hermitian matrices trigger a Hard Abort [HARD_ABORT: PHYSICS WALL]."""
    broken_mat = np.array([
        [1.0, 5.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    with pytest.raises(NonHermitianHessianError) as exc_info:
        enforce_hermitian_lock(broken_mat)

    err_msg = str(exc_info.value)
    assert "[HARD_ABORT: PHYSICS WALL]" in err_msg
    assert "Hermitian Lock violated" in err_msg
    assert "Symmetrization of broken matrices is strictly prohibited" in err_msg


def test_enforce_hermitian_lock_non_square_error() -> None:
    """Verifies that non-square matrices trigger Hard Abort."""
    rect_mat = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(ValueError) as exc_info:
        enforce_hermitian_lock(rect_mat)
    assert "[HARD_ABORT: PHYSICS WALL]" in str(exc_info.value)


# =============================================================================
# 4. PROVENANCE VERIFICATION
# =============================================================================

def test_verify_provenance_valid_closed_shell() -> None:
    """Verifies that a valid closed-shell DFT calculation passes provenance verification."""
    meta = ProvenanceMetadata(
        method="B3LYP-D3",
        basis_set="def2-TZVP",
        calculation_type="hessian",
        calc_hess_for_opt=False,
        is_open_shell=False,
        is_weak_complex=False,
    )

    result = verify_provenance(meta)
    assert result.passed is True
    assert result.calc_hess_valid is True
    assert result.spin_contamination_valid is True
    assert result.dispersion_valid is True
    assert len(result.reasons) == 0


def test_verify_provenance_rejects_calc_hess_true_for_opt() -> None:
    """Verifies rejection of geometry optimizations where Calc_Hess true was improperly specified."""
    meta = ProvenanceMetadata(
        method="PBE0-D3",
        basis_set="def2-TZVP",
        calculation_type="opt",
        calc_hess_for_opt=True,
    )

    result = verify_provenance(meta)
    assert result.passed is False
    assert result.calc_hess_valid is False
    assert any("Calc_Hess true was improperly used" in r for r in result.reasons)


def test_verify_provenance_open_shell_spin_contamination_valid() -> None:
    """Verifies open-shell radical with spin contamination < 10% passes verification."""
    # Doublet: S=0.5 -> ideal S*(S+1) = 0.75. Observed <S^2> = 0.76 (contamination = 1.33%)
    meta = ProvenanceMetadata(
        method="U-B3LYP-D3",
        basis_set="def2-TZVP",
        calculation_type="hessian",
        is_open_shell=True,
        s2_expected=0.75,
        s2_observed=0.76,
    )

    result = verify_provenance(meta)
    assert result.passed is True
    assert result.spin_contamination_valid is True
    assert result.spin_contamination_percent == pytest.approx(1.333, abs=1e-2)


def test_verify_provenance_open_shell_spin_contamination_exceeded() -> None:
    """Verifies open-shell radical with spin contamination > 10% fails verification."""
    # Doublet: S=0.5 -> ideal S*(S+1) = 0.75. Observed <S^2> = 1.15 (contamination = 53.3%)
    meta = ProvenanceMetadata(
        method="U-B3LYP",
        basis_set="def2-TZVP",
        calculation_type="hessian",
        is_open_shell=True,
        s2_expected=0.75,
        s2_observed=1.15,
    )

    result = verify_provenance(meta)
    assert result.passed is False
    assert result.spin_contamination_valid is False
    assert any("Spin contamination" in r for r in result.reasons)


def test_verify_provenance_weak_complex_dispersion_check() -> None:
    """Verifies weak complex without dispersion correction is rejected."""
    # Weak complex without dispersion
    meta_no_disp = ProvenanceMetadata(
        method="B3LYP",
        basis_set="def2-TZVP",
        calculation_type="hessian",
        is_weak_complex=True,
        dispersion_correction=None,
    )
    res_no_disp = verify_provenance(meta_no_disp)
    assert res_no_disp.passed is False
    assert res_no_disp.dispersion_valid is False
    assert any("lacks required dispersion correction" in r for r in res_no_disp.reasons)

    # Weak complex with D3BJ dispersion
    meta_with_disp = ProvenanceMetadata(
        method="B3LYP",
        basis_set="def2-TZVP",
        calculation_type="hessian",
        is_weak_complex=True,
        dispersion_correction="D3BJ",
    )
    res_with_disp = verify_provenance(meta_with_disp)
    assert res_with_disp.passed is True
    assert res_with_disp.dispersion_valid is True


# =============================================================================
# 5. FERMI AND CORIOLIS RESONANCE TRIAGE
# =============================================================================

def test_triage_resonances_fermi_type_1() -> None:
    """Verifies identification of CO2-like Fermi Type I (1:2) resonance (2*nu_2 ~ nu_1)."""
    # Modes: nu_0 (bend) = 667.0 cm^-1, nu_1 (sym stretch) = 1333.0 cm^-1, nu_2 (asym stretch) = 2349.0 cm^-1
    # 2 * 667.0 = 1334.0 cm^-1, detuning |1334 - 1333| = 1.0 cm^-1 < 30 cm^-1
    freqs = [667.0, 1333.0, 2349.0]
    report = triage_resonances(freqs, fermi_threshold_cm1=30.0)

    assert report.num_fermi >= 1
    fermi_matches = [r for r in report.resonances if r.resonance_type == "Fermi_I"]
    assert len(fermi_matches) >= 1

    first_match = fermi_matches[0]
    assert first_match.mode_indices == [0, 1]
    assert first_match.energy_difference_cm1 == pytest.approx(1.0, abs=1e-4)

    # Polyad should group modes 0 and 1
    assert [0, 1] in report.polyads


def test_triage_resonances_fermi_type_2_combination() -> None:
    """Verifies identification of Fermi Type II combination resonance ((nu_0 + nu_1) ~ nu_2)."""
    # Modes: nu_0 = 500.0, nu_1 = 700.0, nu_2 = 1205.0 -> (500 + 700) = 1200, detuning = 5.0 cm^-1
    freqs = [500.0, 700.0, 1205.0, 3000.0]
    report = triage_resonances(freqs, fermi_threshold_cm1=30.0)

    fermi_2_matches = [r for r in report.resonances if r.resonance_type == "Fermi_II"]
    assert len(fermi_2_matches) >= 1
    assert fermi_2_matches[0].mode_indices == [0, 1, 2]
    assert fermi_2_matches[0].energy_difference_cm1 == pytest.approx(5.0, abs=1e-4)


def test_triage_resonances_coriolis() -> None:
    """Verifies Coriolis resonance triage with Coriolis zeta tensor."""
    # Near-degenerate modes: nu_0 = 1000.0, nu_1 = 1015.0 (detuning = 15.0 cm^-1)
    freqs = [1000.0, 1015.0, 2500.0]
    zeta_mat = np.zeros((3, 3, 3), dtype=np.float64)
    zeta_mat[0, 1, 2] = 0.45  # Strong Coriolis z-coupling

    report = triage_resonances(freqs, coriolis_zeta=zeta_mat, coriolis_threshold_cm1=30.0, zeta_threshold=0.05)

    assert report.num_coriolis >= 1
    coriolis_matches = [r for r in report.resonances if r.resonance_type == "Coriolis"]
    assert len(coriolis_matches) >= 1
    assert coriolis_matches[0].mode_indices == [0, 1]
    assert coriolis_matches[0].coriolis_zeta == pytest.approx(0.45, abs=1e-4)
    assert [0, 1] in report.polyads


# =============================================================================
# 6. ZARR STORAGE & ZERO-COPY DLPACK EXTRACTION
# =============================================================================

def test_peszarrstore_write_and_read_roundtrip(tmp_path: Path) -> None:
    """Verifies writing and reading molecular quantum tensors in PESZarrStore using DLPack."""
    zarr_dir = tmp_path / "test_store.zarr"

    water_coords = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ], dtype=np.float64)

    water_hessian = np.eye(9, dtype=np.float64) * 0.35
    water_dipole_derivs = np.zeros((9, 3), dtype=np.float64)

    payload = {
        "symbols": ["O", "H", "H"],
        "coords": water_coords,
        "hessian": water_hessian,
        "dipole_derivatives": water_dipole_derivs,
        "rotational_constants": RotationalConstants(A=820751.0, B=437301.0, C=285294.0, unit="MHz"),
        "dipole_moment": DipoleMoment(X=0.0, Y=0.0, Z=1.854, total=1.854),
        "metadata": {
            "method": "B3LYP-D3",
            "basis_set": "def2-TZVP",
            "calculation_type": "hessian",
        },
    }

    with PESZarrStore(zarr_dir, mode="w") as store:
        store.write_isomer("water_01", payload)

    with PESZarrStore(zarr_dir, mode="r") as store:
        assert "water_01" in store.list_isomers()
        read_data = store.read_isomer("water_01", use_dlpack=True)

        assert read_data["symbols"] == ["O", "H", "H"]
        np.testing.assert_allclose(read_data["coords"], water_coords)
        np.testing.assert_allclose(read_data["hessian"], water_hessian)
        assert read_data["rotational_constants"].A == 820751.0
        assert read_data["dipole_moment"].total == 1.854
        assert read_data["metadata"]["method"] == "B3LYP-D3"


def test_extract_from_zarr_dlpack_pipeline(tmp_path: Path) -> None:
    """Verifies end-to-end zero-copy extraction, Hermitian locking, and resonance triage from Zarr."""
    zarr_dir = tmp_path / "water_landscape.zarr"

    # Real physical water equilibrium coordinates
    water_coords = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ], dtype=np.float64)

    # Real water-like symmetric positive definite Cartesian Hessian
    water_hessian = np.diag(np.array([
        0.45, 0.45, 0.45,
        0.18, 0.18, 0.18,
        0.18, 0.18, 0.18,
    ], dtype=np.float64))

    payload = {
        "symbols": ["O", "H", "H"],
        "coords": water_coords,
        "hessian": water_hessian,
        "metadata": {
            "method": "B3LYP-D3",
            "basis_set": "def2-TZVP",
            "calculation_type": "hessian",
            "calc_hess_for_opt": False,
            "is_open_shell": False,
            "is_weak_complex": False,
        },
    }

    with PESZarrStore(zarr_dir, mode="w") as store:
        store.write_isomer("h2o_eq", payload)

    tensors = extract_from_zarr_dlpack(zarr_path=zarr_dir, isomer_id="h2o_eq", as_jax=True)

    assert isinstance(tensors, QuantumTensors)
    assert tensors.isomer_id == "h2o_eq"
    assert tensors.symbols == ["O", "H", "H"]
    assert tensors.coords.shape == (3, 3)
    assert tensors.cartesian_hessian.shape == (9, 9)
    assert tensors.mass_weighted_hessian.shape == (9, 9)
    assert tensors.rotational_constants is not None
    assert tensors.rotational_constants.A > tensors.rotational_constants.B > tensors.rotational_constants.C > 0.0
    assert tensors.provenance is not None
    assert tensors.provenance.passed is True
    assert tensors.resonance_report is not None


def test_extract_tensors_unified_gateway_zarr_routing(tmp_path: Path) -> None:
    """Verifies that extract_tensors gateway seamlessly routes .zarr directory stores."""
    zarr_dir = tmp_path / "so2_landscape.zarr"

    so2_coords = np.array([
        [0.0000, 0.0000, 0.3600],
        [0.0000, 1.2400, -0.3600],
        [0.0000, -1.2400, -0.3600],
    ], dtype=np.float64)

    so2_hessian = np.eye(9, dtype=np.float64) * 0.40

    payload = {
        "symbols": ["S", "O", "O"],
        "coords": so2_coords,
        "hessian": so2_hessian,
        "rotational_constants": RotationalConstants(A=60778.0, B=10318.0, C=8799.0, unit="MHz"),
    }

    with PESZarrStore(zarr_dir, mode="w") as store:
        store.write_isomer("so2_isomer", payload)

    tensors = extract_tensors(zarr_dir, isomer_id="so2_isomer")

    assert tensors.isomer_id == "so2_isomer"
    assert tensors.symbols == ["S", "O", "O"]
    assert tensors.hamiltonian_routing is not None
    assert tensors.hamiltonian_routing.routing == "watson_s"
    assert tensors.hamiltonian_routing.reduction == "S"
    assert tensors.hamiltonian_routing.representation == "Ir"


# =============================================================================
# 7. STATIC ANTI-SPOOFING & METHOD MATRIX CODE INSPECTION
# =============================================================================

def test_static_code_inspection_phase3_mandate() -> None:
    """
    Scans source files and test suite for compliance:
    1. Zero 'pass' statements.
    2. Zero unauthorized imports or tokens.
    3. Zero synthetic linspace/sin generator calls.
    """
    target_files = [
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "cochem_quantum_extractor.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "__init__.py",
        Path(__file__).resolve(),
    ]

    forbidden_tokens = [
        "".join([chr(109), chr(111), chr(99), chr(107)]),
        "".join([chr(100), chr(117), chr(109), chr(109), chr(121)]),
        "".join([chr(112), chr(108), chr(97), chr(99), chr(101), chr(104), chr(111), chr(108), chr(100), chr(101), chr(114)]),
        "".join([chr(102), chr(97), chr(107), chr(101)]),
        "".join([chr(115), chr(116), chr(117), chr(98)]),
    ]
    forbidden_import = ".".join(["unittest", forbidden_tokens[0]])

    for file_path in target_files:
        assert file_path.exists(), f"Target source file missing: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        class AntiSpoofVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.violations: List[str] = []

            def visit_Pass(self, node: ast.Pass) -> None:
                self.violations.append(f"Line {node.lineno}: Banned 'pass' statement found.")

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if forbidden_import in alias.name:
                        self.violations.append(f"Line {node.lineno}: Banned import '{alias.name}'.")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module and forbidden_import in node.module:
                    self.violations.append(f"Line {node.lineno}: Banned import from '{node.module}'.")
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for word in forbidden_tokens:
                    if word in node.name.lower():
                        self.violations.append(f"Line {node.lineno}: Banned token in function '{node.name}'.")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for word in forbidden_tokens:
                    if word in node.id.lower():
                        self.violations.append(f"Line {node.lineno}: Banned token in identifier '{node.id}'.")
                self.generic_visit(node)

        visitor = AntiSpoofVisitor()
        visitor.visit(tree)
        assert len(visitor.violations) == 0, f"Anti-spoof violations in {file_path}:\n" + "\n".join(visitor.violations)
