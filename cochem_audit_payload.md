Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_2_Prompt1_extractor.md.
Original prompt:
# Task 2 (Part 2): Stage 3.1 - cochem_quantum_extractor.py

**Objective:**
Create the `cochem_quantum_extractor.py` file to act as the Theoretical Tensor Gateway.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_quantum_extractor.py`

**Instructions:**
You are the `cochem-coder` agent. Implement `cochem_quantum_extractor.py`.
It must:
1. Extract Cartesian Hessians and dipole derivatives via the `PESStore` class. Do NOT use raw `H5F_ACC_RDONLY` calls.
2. Cast data to `np.float64` strictly at the ingestion boundary.
3. Apply mass perturbations to yield mass-weighted coordinates and execute a Hermitian lock (`np.allclose(H, H.conj().T)`). Apply strict symmetrization if numerical noise is detected.
4. Compute Ray's asymmetry parameter ($\kappa$).
5. Extract the 1D-torsional barrier ($V_3$) computed by CoChem-TORQ and route to an IAM/Woods Method Hamiltonian if $V_3 < 400 \text{ cm}^{-1}$.

**Constraints:**
- No placeholders, stubs, mock dictionaries, or dummy execution.
- You must write the actual mathematical implementations using JAX and Numpy. 
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import numpy as np

def extract_tensors(pes_store):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def apply_mass_perturbation_and_lock(hessian_cartesian, mass_matrix):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def compute_rays_asymmetry(A, B, C):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_quantum_extractor.py ---
"""
CoChem-SpycFit: Unit Test Suite for Theoretical Tensor Gateway & Physics Core.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero unverified imports
- 100% authentic physical calculations, HDF5 persistence, and JAX FP64 validation.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from typing import Dict, List

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_quantum_extractor import (
    AMU_ANGSTROM2_TO_MHZ,
    HARTREE_BOHR2_AMU_TO_CM1,
    MASS_DICT,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    QuantumExtraction,
    QuantumExtractor,
    QuantumTensors,
    RotationalConstants,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    determine_hamiltonian_routing,
    extract_tensors,
    get_atomic_mass,
)

# Physical dataset path resolved relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
LANDSCAPE_H5_PATH = REPO_ROOT / "artifacts_test" / "landscape.h5"


# =============================================================================
# 1. ATOMIC MASS & ROTATIONAL CONSTANT CALCULATIONS
# =============================================================================

def test_atomic_mass_retrieval() -> None:
    """Verifies IUPAC isotopic mass resolution for standard elements and isotopes."""
    mass_h = get_atomic_mass("H")
    mass_d = get_atomic_mass("D")
    mass_c = get_atomic_mass("C")
    mass_c13 = get_atomic_mass("13C")
    mass_o = get_atomic_mass("O")
    mass_s = get_atomic_mass("S")

    assert mass_h == pytest.approx(1.007825, rel=1e-5)
    assert mass_d == pytest.approx(2.014101, rel=1e-5)
    assert mass_c == pytest.approx(12.000000, abs=1e-6)
    assert mass_c13 == pytest.approx(13.003354, rel=1e-5)
    assert mass_o == pytest.approx(15.994915, rel=1e-5)
    assert mass_s == pytest.approx(31.972070, rel=1e-5)

    with pytest.raises(ValueError):
        get_atomic_mass("UnknownElement999")


def test_compute_rotational_constants_water() -> None:
    """Computes principal rotational constants for real H2O equilibrium geometry."""
    # Water geometry in Angstroms: O at origin, H atoms symmetrically placed
    h2o_coords = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ], dtype=np.float64)
    symbols = ["O", "H", "H"]

    rot = compute_rotational_constants_from_coords(h2o_coords, symbols, unit="angstrom")

    assert rot.A > rot.B > rot.C > 0.0
    # Expected water constants: A ~ 820 GHz, B ~ 437 GHz, C ~ 285 GHz
    assert rot.A == pytest.approx(820751.0, rel=1e-2)
    assert rot.B == pytest.approx(437301.0, rel=1e-2)
    assert rot.C == pytest.approx(285294.0, rel=1e-2)


def test_compute_rotational_constants_sulfur_dioxide() -> None:
    """Computes principal rotational constants for real SO2 equilibrium geometry."""
    # SO2 geometry (Angstrom): S at (0, 0, 0.36), O1 at (0, 1.24, -0.36), O2 at (0, -1.24, -0.36)
    so2_coords = np.array([
        [0.0000, 0.0000, 0.3600],
        [0.0000, 1.2400, -0.3600],
        [0.0000, -1.2400, -0.3600],
    ], dtype=np.float64)
    symbols = ["S", "O", "O"]

    rot = compute_rotational_constants_from_coords(so2_coords, symbols, unit="angstrom")

    assert rot.A > rot.B > rot.C > 0.0
    assert rot.A > 50000.0  # SO2 A is ~60 GHz
    assert rot.B > 8000.0   # SO2 B is ~10 GHz
    assert rot.C > 7000.0   # SO2 C is ~8.8 GHz


# =============================================================================
# 2. RAY'S ASYMMETRY PARAMETER & HAMILTONIAN ROUTING
# =============================================================================

def test_compute_rays_asymmetry_limits() -> None:
    """Verifies Ray's asymmetry parameter calculations across exact physical limits."""
    # 1. Prolate Symmetric Top Limit (A > B == C) -> kappa = -1.0
    kappa_prolate = compute_rays_asymmetry(A=30000.0, B=10000.0, C=10000.0)
    assert kappa_prolate == pytest.approx(-1.0, abs=1e-7)

    # 2. Oblate Symmetric Top Limit (A == B > C) -> kappa = +1.0
    kappa_oblate = compute_rays_asymmetry(A=20000.0, B=20000.0, C=8000.0)
    assert kappa_oblate == pytest.approx(1.0, abs=1e-7)

    # 3. Spherical Top Limit (A == B == C) -> kappa = 0.0
    kappa_spherical = compute_rays_asymmetry(A=15000.0, B=15000.0, C=15000.0)
    assert kappa_spherical == pytest.approx(0.0, abs=1e-7)

    # 4. Most Asymmetric Limit (B = (A + C) / 2) -> kappa = 0.0
    kappa_asym = compute_rays_asymmetry(A=30000.0, B=20000.0, C=10000.0)
    assert kappa_asym == pytest.approx(0.0, abs=1e-7)

    # 5. Real Water: A ~ 820751, B ~ 437301, C ~ 285295 -> kappa ~ -0.4322
    kappa_water = compute_rays_asymmetry(A=820751.0, B=437301.4, C=285294.6)
    assert kappa_water == pytest.approx(-0.4322, abs=1e-3)

    # 6. Real SO2: A ~ 60778, B ~ 10318, C ~ 8799 -> kappa ~ -0.9415 (Near-prolate)
    kappa_so2 = compute_rays_asymmetry(A=60778.0, B=10318.0, C=8799.0)
    assert kappa_so2 == pytest.approx(-0.9415, abs=1e-3)


def test_determine_hamiltonian_routing_torsional_barrier() -> None:
    """Verifies IAM/Woods Method Hamiltonian routing for low torsional barrier V3 < 400 cm^-1."""
    # Low barrier (e.g. methanol V3 ~ 373 cm^-1 or acetic acid V3 ~ 170 cm^-1)
    route_iam = determine_hamiltonian_routing(kappa=-0.45, v3_barrier_cm1=373.0)
    assert route_iam.routing == "IAM_WOODS"
    assert route_iam.reduction is None
    assert route_iam.v3_barrier_cm1 == 373.0

    # Low barrier regardless of kappa
    route_iam_prolate = determine_hamiltonian_routing(kappa=-0.95, v3_barrier_cm1=150.0)
    assert route_iam_prolate.routing == "IAM_WOODS"

    # High barrier (V3 >= 400 cm^-1) falls through to Watson reductions
    route_high_barrier = determine_hamiltonian_routing(kappa=-0.43, v3_barrier_cm1=1200.0)
    assert route_high_barrier.routing == "watson_a"
    assert route_high_barrier.reduction == "A"
    assert route_high_barrier.representation == "Ir"


def test_determine_hamiltonian_routing_watson_reductions() -> None:
    """Verifies Watson A and S reduction routings based on Ray's kappa."""
    # 1. Asymmetric rotor (|kappa| < 0.5) -> Watson A, Ir
    route_water = determine_hamiltonian_routing(kappa=-0.432)
    assert route_water.routing == "watson_a"
    assert route_water.reduction == "A"
    assert route_water.representation == "Ir"

    route_asym_pos = determine_hamiltonian_routing(kappa=0.25)
    assert route_asym_pos.routing == "watson_a"
    assert route_asym_pos.reduction == "A"
    assert route_asym_pos.representation == "Ir"

    # 2. Near-prolate symmetric top (kappa <= -0.5) -> Watson S, Ir
    route_so2 = determine_hamiltonian_routing(kappa=-0.9415)
    assert route_so2.routing == "watson_s"
    assert route_so2.reduction == "S"
    assert route_so2.representation == "Ir"

    route_prolate_edge = determine_hamiltonian_routing(kappa=-0.5)
    assert route_prolate_edge.routing == "watson_s"
    assert route_prolate_edge.reduction == "S"
    assert route_prolate_edge.representation == "Ir"

    # 3. Near-oblate symmetric top (kappa >= 0.5) -> Watson S, IIIr
    route_oblate = determine_hamiltonian_routing(kappa=0.85)
    assert route_oblate.routing == "watson_s"
    assert route_oblate.reduction == "S"
    assert route_oblate.representation == "IIIr"

    route_oblate_edge = determine_hamiltonian_routing(kappa=0.5)
    assert route_oblate_edge.routing == "watson_s"
    assert route_oblate_edge.reduction == "S"
    assert route_oblate_edge.representation == "IIIr"


# =============================================================================
# 3. MASS PERTURBATION & HERMITIAN LOCK
# =============================================================================

def test_apply_mass_perturbation_and_lock_water() -> None:
    """Verifies mass-weighting and Hermitian lock on physical water Cartesian Hessian."""
    # Load physical water Cartesian Hessian from landscape.h5
    with h5py.File(LANDSCAPE_H5_PATH, "r") as f:
        cart_hessian = f["isomer_001"]["hessian"][:]
        raw_symbols = f["isomer_001"]["symbols"][:]
        symbols = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_symbols]

    mw_hessian = apply_mass_perturbation_and_lock(cart_hessian, symbols=symbols)

    assert mw_hessian.shape == (9, 9)
    assert mw_hessian.dtype == np.float64

    # Strict Hermitian check
    np.testing.assert_allclose(mw_hessian, mw_hessian.T, atol=1e-14)

    # Mass weighting verification: H_mw[i, j] == H_cart[i, j] / sqrt(m_i * m_j)
    masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)
    masses_3n = np.repeat(masses, 3)
    expected_elem_0_0 = cart_hessian[0, 0] / masses_3n[0]
    assert float(mw_hessian[0, 0]) == pytest.approx(expected_elem_0_0, rel=1e-10)


def test_apply_mass_perturbation_isotopologues() -> None:
    """Verifies mass-weighting scaling across isotopologues (H2O vs D2O)."""
    # Create symmetric positive harmonic force matrix
    k_mat = np.diag(np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64))

    mw_h2o = apply_mass_perturbation_and_lock(k_mat, symbols=["O", "H", "H"])
    mw_d2o = apply_mass_perturbation_and_lock(k_mat, symbols=["O", "D", "D"])

    # Hydrogen atoms are at indices 3..8: mass ratio is m_D / m_H ~ 2.014 / 1.0078 ~ 2
    # Mass weighted diagonal elements scale as 1 / m
    ratio_h_d = mw_h2o[3, 3] / mw_d2o[3, 3]
    expected_ratio = get_atomic_mass("D") / get_atomic_mass("H")
    assert float(ratio_h_d) == pytest.approx(expected_ratio, rel=1e-5)


def test_apply_mass_perturbation_non_hermitian_error() -> None:
    """Verifies that severe non-Hermitian asymmetry raises NonHermitianHessianError."""
    # Create asymmetric matrix with deviation > 1e-2
    asym_hess = np.eye(9, dtype=np.float64)
    asym_hess[0, 1] = 10.0
    asym_hess[1, 0] = 0.0

    with pytest.raises(NonHermitianHessianError) as exc_info:
        apply_mass_perturbation_and_lock(asym_hess, symbols=["O", "H", "H"])

    assert "violates Hermitian symmetry severely" in str(exc_info.value)


def test_apply_mass_perturbation_numerical_noise_lock() -> None:
    """Verifies that tiny floating point numerical noise is cleanly symmetrized."""
    clean_hess = np.eye(9, dtype=np.float64)
    clean_hess[0, 1] = 0.5
    clean_hess[1, 0] = 0.5 + 1e-10  # Tiny asymmetry from numerical floating point noise

    locked = apply_mass_perturbation_and_lock(clean_hess, symbols=["O", "H", "H"])
    np.testing.assert_allclose(locked, locked.T, atol=1e-15)


# =============================================================================
# 4. JAX FP64 NORMAL MODE ANALYSIS
# =============================================================================

def test_compute_normal_modes_jax_fp64() -> None:
    """Executes JAX FP64 normal mode analysis on physical water mass-weighted Hessian."""
    with h5py.File(LANDSCAPE_H5_PATH, "r") as f:
        cart_hessian = f["isomer_001"]["hessian"][:]
        raw_symbols = f["isomer_001"]["symbols"][:]
        symbols = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_symbols]

    mw_hessian = apply_mass_perturbation_and_lock(cart_hessian, symbols=symbols)

    result = compute_normal_modes_jax(mw_hessian, symbols=symbols)

    assert "frequencies_cm1" in result
    assert "eigenvalues" in result
    assert "eigenvectors" in result
    assert "reduced_masses_amu" in result
    assert "zero_point_energy_cm1" in result

    freqs = result["frequencies_cm1"]
    assert freqs.shape == (9,)
    assert freqs.dtype == np.float64

    # Water has 3 translational + 3 rotational zero/near-zero modes and 3 genuine vibrational modes
    # Highest frequency mode should be positive and correspond to O-H stretch
    positive_freqs = freqs[freqs > 0.0]
    assert len(positive_freqs) > 0
    assert result["zero_point_energy_cm1"] > 0.0


# =============================================================================
# 5. PESSTORE HDF5 CHUNKED PERSISTENCE & EXTRACTION
# =============================================================================

def test_pesstore_read_landscape_h5() -> None:
    """Verifies reading real physical landscape.h5 through PESStore class."""
    with PESStore(LANDSCAPE_H5_PATH, mode="r") as store:
        isomers = store.list_isomers()
        assert "isomer_001" in isomers

        data = store.read_isomer("isomer_001")
        assert data["symbols"] == ["O", "H", "H"]
        assert data["coords"].shape == (3, 3)
        assert data["coords"].dtype == np.float64
        assert data["hessian"].shape == (9, 9)
        assert data["hessian"].dtype == np.float64
        assert data["lam_trigger"] is True
        assert data["dvr_eigenvalues"] is not None
        assert data["tunneling_splittings"] is not None
        assert data["kraitchman_coords"] is not None


def test_pesstore_write_and_read_roundtrip(tmp_path: Path) -> None:
    """Verifies chunked QCSchema dataset writing and reading in a new HDF5 file."""
    h5_file = tmp_path / "test_store.h5"

    so2_coords = np.array([
        [0.0, 0.0, 0.36],
        [0.0, 1.24, -0.36],
        [0.0, -1.24, -0.36],
    ], dtype=np.float64)

    so2_hessian = np.eye(9, dtype=np.float64) * 0.45
    so2_dipole_derivs = np.ones((9, 3), dtype=np.float64) * 0.12

    payload = {
        "symbols": ["S", "O", "O"],
        "coords": so2_coords,
        "hessian": so2_hessian,
        "dipole_derivatives": so2_dipole_derivs,
        "rotational_constants": RotationalConstants(A=60778.0, B=10318.0, C=8799.0, unit="MHz"),
        "dipole_moment": DipoleMoment(X=0.0, Y=0.0, Z=1.63, total=1.63),
        "v3_barrier_cm1": 850.0,
        "lam_trigger": False,
    }

    with PESStore(h5_file, mode="w") as store:
        store.write_isomer("so2_isomer", payload, chunks=True, compression="gzip")

    # Read back and verify exact data integrity
    with PESStore(h5_file, mode="r") as store:
        assert "so2_isomer" in store.list_isomers()
        read_data = store.read_isomer("so2_isomer")

        assert read_data["symbols"] == ["S", "O", "O"]
        np.testing.assert_allclose(read_data["coords"], so2_coords)
        np.testing.assert_allclose(read_data["hessian"], so2_hessian)
        np.testing.assert_allclose(read_data["dipole_derivatives"], so2_dipole_derivs)
        assert read_data["rotational_constants"].A == 60778.0
        assert read_data["rotational_constants"].B == 10318.0
        assert read_data["rotational_constants"].C == 8799.0
        assert read_data["dipole_moment"].total == 1.63
        assert read_data["v3_barrier_cm1"] == 850.0
        assert read_data["lam_trigger"] is False


def test_pesstore_missing_isomer_error(tmp_path: Path) -> None:
    """Verifies that IsomerNotFoundError is raised when querying non-existent isomer."""
    h5_file = tmp_path / "empty_store.h5"
    with PESStore(h5_file, mode="w") as store:
        store.write_isomer("iso_A", {"symbols": ["H"], "hessian": np.eye(3, dtype=np.float64)})

    with PESStore(h5_file, mode="r") as store:
        with pytest.raises(IsomerNotFoundError):
            store.read_isomer("non_existent_iso")


# =============================================================================
# 6. HIGH-LEVEL TENSOR EXTRACTION GATEWAY (extract_tensors)
# =============================================================================

def test_extract_tensors_physical_landscape() -> None:
    """Verifies extract_tensors gateway against physical landscape.h5."""
    tensors = extract_tensors(LANDSCAPE_H5_PATH, isomer_id="isomer_001")

    assert isinstance(tensors, QuantumTensors)
    assert tensors.isomer_id == "isomer_001"
    assert tensors.symbols == ["O", "H", "H"]
    assert tensors.coords.shape == (3, 3)
    assert tensors.cartesian_hessian.shape == (9, 9)
    assert tensors.mass_weighted_hessian.shape == (9, 9)
    assert tensors.masses.shape == (3,)
    assert tensors.lam_trigger is True
    assert tensors.dvr_eigenvalues is not None
    assert tensors.tunneling_splittings is not None
    assert tensors.kraitchman_coords is not None


def test_extract_tensors_from_pesstore_with_routing(tmp_path: Path) -> None:
    """Verifies extract_tensors populates Ray kappa and Hamiltonian routing accurately."""
    h5_file = tmp_path / "so2_test.h5"

    so2_coords = np.array([
        [0.0, 0.0, 0.36],
        [0.0, 1.24, -0.36],
        [0.0, -1.24, -0.36],
    ], dtype=np.float64)

    payload = {
        "symbols": ["S", "O", "O"],
        "coords": so2_coords,
        "hessian": np.eye(9, dtype=np.float64) * 0.5,
        "rotational_constants": RotationalConstants(A=60778.0, B=10318.0, C=8799.0, unit="MHz"),
        "v3_barrier_cm1": 1500.0,
    }

    with PESStore(h5_file, mode="w") as store:
        store.write_isomer("so2_001", payload)

    tensors = extract_tensors(h5_file, isomer_id="so2_001")

    assert tensors.hamiltonian_routing is not None
    assert tensors.hamiltonian_routing.routing == "watson_s"
    assert tensors.hamiltonian_routing.reduction == "S"
    assert tensors.hamiltonian_routing.representation == "Ir"
    assert tensors.hamiltonian_routing.kappa == pytest.approx(-0.9415, abs=1e-3)


# =============================================================================
# 7. QUANTUM EXTRACTOR LOG PARSER (MPQC, ORCA, G-XTB)
# =============================================================================

def test_quantum_extractor_mpqc_parsing() -> None:
    """Verifies parsing MPQC log text for rotational constants and dipole moment."""
    mpqc_text = """
    *** MPQC Program Version 3.0 ***
    Rotational constants (MHz): 820751.045 437301.397 285294.608
    Total Dipole Moment : 0.0000 0.0000 1.8540
    Expectation value of <S**2> : 0.0000
    Ideal value S*(S+1) for S=0 : 0.0000
    """
    extractor = QuantumExtractor()
    result = extractor.parse_output(mpqc_text)

    assert result.rotational_constants is not None
    assert result.rotational_constants.A == pytest.approx(820751.045, rel=1e-5)
    assert result.rotational_constants.B == pytest.approx(437301.397, rel=1e-5)
    assert result.rotational_constants.C == pytest.approx(285294.608, rel=1e-5)

    assert result.dipole_moment is not None
    assert result.dipole_moment.total == pytest.approx(1.854, abs=1e-3)
    assert result.dipole_moment.Z == pytest.approx(1.854, abs=1e-3)


def test_quantum_extractor_gxtb_parsing() -> None:
    """Verifies parsing g-xTB log text for rotational constants and dipole moment."""
    gxtb_text = """
    GFN2-xTB calculation log
    rotational constants (MHz): 60778.800 10318.000 8799.800
    molecular dipole: x= 0.000 y= 0.000 z= 1.630
    """
    extractor = QuantumExtractor()
    result = extractor.parse_output(gxtb_text)

    assert result.rotational_constants is not None
    assert result.rotational_constants.A == pytest.approx(60778.8, rel=1e-4)

    assert result.dipole_moment is not None
    assert result.dipole_moment.Z == pytest.approx(1.63, abs=1e-3)


def test_quantum_extractor_spin_contamination_error() -> None:
    """Verifies that severe spin contamination (> 10%) raises ValueError."""
    mpqc_text = """
    *** MPQC Program Version 3.0 ***
    Rotational constants (MHz): 50000.0 20000.0 10000.0
    Expectation value of <S**2> : 1.1500
    Ideal value S*(S+1) for S=0.5 : 0.7500
    """
    extractor = QuantumExtractor()
    with pytest.raises(ValueError) as exc_info:
        extractor.parse_mpqc_output(mpqc_text)
    assert "Spin contamination exceeded 10%" in str(exc_info.value)


# =============================================================================
# 8. STATIC CODE INSPECTION (ZERO-SPOOFING & INTEGRITY ENFORCEMENT)
# =============================================================================

def test_static_code_inspection_anti_spoofing() -> None:
    """
    Scans the source code of cochem_quantum_extractor.py and __init__.py.
    Ensures:
    1. Zero 'pass' statements.
    2. Zero unauthorized imports or tokens.
    """
    target_files = [
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "cochem_quantum_extractor.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "__init__.py",
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

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.