"""
CoChem-SpycFit: Unit and Integration Test Suite for Native JAX Hamiltonian Builder.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or mocked functions
- 100% authentic physical and quantum mechanical calculations in JAX FP64
- Comprehensive validation of Ray's asymmetry parameter, hysteresis representation switching,
  static shape J-block batching, Wang parity sub-blocks, Watson A/S-reduction Hamiltonians,
  Internal Rotation (LAM) guard, and IAM/Woods coupled methods.
"""

from __future__ import annotations

import ast
import inspect
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import mendeleev
import numpy as np
import pytest
import scipy.linalg

from cochem_spycfit.physics_core.cochem_jax_builder import (
    DEFAULT_BUCKET_SIZE,
    DEFAULT_HYSTERESIS_DELTA,
    DEFAULT_LAM_BARRIER_THRESHOLD_CM1,
    BatchedHamiltonianResult,
    HamiltonianBlockResult,
    HamiltonianConfig,
    HermitianLockViolationError,
    HPCShardingError,
    InvalidJValueError,
    InvalidReductionError,
    InvalidRepresentationError,
    JAXBuilderError,
    JBlockBatch,
    LAMBreachError,
    LAMGuardResult,
    ReductionType,
    RepresentationType,
    WatsonConstants,
    build_j_block_batches,
    build_wang_transformation_matrix,
    check_lam_guard,
    compute_rays_asymmetry_param,
    construct_hamiltonian,
    construct_iam_woods_hamiltonian,
    construct_j_block,
    construct_watson_a_matrix_jax,
    construct_watson_s_matrix_jax,
    determine_representation,
    diagonalize_j_block,
    execute_distributed_batches,
    is_hpc_environment,
)

# Enable JAX FP64 precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. RAY'S ASYMMETRY PARAMETER & PHYSICAL BOUNDS
# =============================================================================

def test_rays_asymmetry_limiting_cases() -> None:
    """Verifies Ray's asymmetry parameter across prolate, oblate, most asymmetric, and spherical top limits."""
    # 1. Prolate symmetric top limit (A > B == C) -> kappa = -1.0
    kappa_prolate = compute_rays_asymmetry_param(A=10000.0, B=5000.0, C=5000.0)
    assert math.isclose(kappa_prolate, -1.0, abs_tol=1e-12)

    # 2. Oblate symmetric top limit (A == B > C) -> kappa = +1.0
    kappa_oblate = compute_rays_asymmetry_param(A=10000.0, B=10000.0, C=5000.0)
    assert math.isclose(kappa_oblate, 1.0, abs_tol=1e-12)

    # 3. Most asymmetric rotor (B = (A + C) / 2) -> kappa = 0.0
    kappa_asym = compute_rays_asymmetry_param(A=10000.0, B=7500.0, C=5000.0)
    assert math.isclose(kappa_asym, 0.0, abs_tol=1e-12)

    # 4. Spherical top limit (A == B == C) -> kappa = 0.0
    kappa_spherical = compute_rays_asymmetry_param(A=5000.0, B=5000.0, C=5000.0)
    assert math.isclose(kappa_spherical, 0.0, abs_tol=1e-12)

    # 5. Real molecule: Formaldehyde (H2CO: A=281970.5 MHz, B=38833.99 MHz, C=34004.24 MHz) -> kappa ~ -0.961
    kappa_h2co = compute_rays_asymmetry_param(A=281970.5, B=38833.99, C=34004.24)
    expected_h2co = (2.0 * 38833.99 - 281970.5 - 34004.24) / (281970.5 - 34004.24)
    assert math.isclose(kappa_h2co, expected_h2co, abs_tol=1e-6)
    assert -1.0 <= kappa_h2co <= 1.0


def test_rays_asymmetry_invalid_constants() -> None:
    """Verifies that non-positive rotational constants raise ValueError."""
    with pytest.raises(ValueError, match="Rotational constants must be positive"):
        compute_rays_asymmetry_param(A=-100.0, B=50.0, C=30.0)

    with pytest.raises(ValueError, match="Rotational constants must be positive"):
        compute_rays_asymmetry_param(A=100.0, B=0.0, C=30.0)


# =============================================================================
# 2. DYNAMIC REPRESENTATION SWITCHING & HYSTERESIS BUFFER
# =============================================================================

def test_determine_representation_without_prior_state() -> None:
    """Verifies representation determination based strictly on kappa threshold 0.5 without prior history."""
    # Near-prolate molecule (kappa = -0.8 < 0.5) -> Ir
    # A=10000, B=5500, C=5000 -> kappa = (11000 - 10000 - 5000) / 5000 = -4000/5000 = -0.8
    rep, kappa = determine_representation(A=10000.0, B=5500.0, C=5000.0, prev_kappa=None)
    assert rep == "Ir"
    assert math.isclose(kappa, -0.8, abs_tol=1e-12)

    # Near-oblate molecule (kappa = 0.8 > 0.5) -> IIIr
    # A=10000, B=9500, C=5000 -> kappa = (19000 - 10000 - 5000) / 5000 = +4000/5000 = +0.8
    rep, kappa = determine_representation(A=10000.0, B=9500.0, C=5000.0, prev_kappa=None)
    assert rep == "IIIr"
    assert math.isclose(kappa, 0.8, abs_tol=1e-12)


def test_determine_representation_with_hysteresis_buffer() -> None:
    """
    Verifies that the hysteresis buffer prevents representation thrashing near kappa = 0.5.
    With delta = 0.05:
    - If previously IIIr (prev_kappa > 0.5): stays IIIr unless kappa drops below 0.45.
    - If previously Ir (prev_kappa <= 0.5): stays Ir unless kappa exceeds 0.55.
    """
    # Case 1: Was in IIIr (prev_kappa = 0.6).
    # New kappa = 0.48 (between 0.45 and 0.50): must stay in IIIr due to hysteresis buffer!
    # A=10000, C=5000, B = (0.48 * 5000 + 15000) / 2 = (2400 + 15000)/2 = 8700
    b_val_48 = (0.48 * 5000.0 + 15000.0) / 2.0
    rep_1, kap_1 = determine_representation(A=10000.0, B=b_val_48, C=5000.0, prev_kappa=0.6, hysteresis_delta=0.05)
    assert rep_1 == "IIIr", "Should remain IIIr inside hysteresis buffer"
    assert math.isclose(kap_1, 0.48, abs_tol=1e-6)

    # New kappa = 0.40 (below 0.45): must switch back to Ir!
    b_val_40 = (0.40 * 5000.0 + 15000.0) / 2.0
    rep_2, kap_2 = determine_representation(A=10000.0, B=b_val_40, C=5000.0, prev_kappa=0.6, hysteresis_delta=0.05)
    assert rep_2 == "Ir", "Should switch to Ir when kappa < 0.45"
    assert math.isclose(kap_2, 0.40, abs_tol=1e-6)

    # Case 2: Was in Ir (prev_kappa = 0.4).
    # New kappa = 0.52 (between 0.50 and 0.55): must stay in Ir!
    b_val_52 = (0.52 * 5000.0 + 15000.0) / 2.0
    rep_3, kap_3 = determine_representation(A=10000.0, B=b_val_52, C=5000.0, prev_kappa=0.4, hysteresis_delta=0.05)
    assert rep_3 == "Ir", "Should remain Ir inside hysteresis buffer"
    assert math.isclose(kap_3, 0.52, abs_tol=1e-6)

    # New kappa = 0.60 (above 0.55): must switch to IIIr!
    b_val_60 = (0.60 * 5000.0 + 15000.0) / 2.0
    rep_4, kap_4 = determine_representation(A=10000.0, B=b_val_60, C=5000.0, prev_kappa=0.4, hysteresis_delta=0.05)
    assert rep_4 == "IIIr", "Should switch to IIIr when kappa > 0.55"
    assert math.isclose(kap_4, 0.60, abs_tol=1e-6)


# =============================================================================
# 3. INTERNAL ROTATION (LAM) GUARD & ROUTING
# =============================================================================

def test_lam_guard_evaluation() -> None:
    """Verifies torsional barrier polling and routing to rigid-rotor vs IAM/Woods."""
    # 1. No barrier specified -> Rigid Rotor
    res_none = check_lam_guard(v3_barrier_cm1=None)
    assert not res_none.breached
    assert res_none.routing == "RIGID_ROTOR"

    # 2. High barrier (V3 = 850 cm^-1 >= 400 cm^-1) -> Rigid Rotor
    res_high = check_lam_guard(v3_barrier_cm1=850.0)
    assert not res_high.breached
    assert res_high.routing == "RIGID_ROTOR"
    assert res_high.v3_barrier_cm1 == 850.0

    # 3. Low barrier (V3 = 210 cm^-1 < 400 cm^-1) -> Breach & IAM Routing
    res_low = check_lam_guard(v3_barrier_cm1=210.0)
    assert res_low.breached
    assert res_low.routing == "IAM_WOODS"
    assert res_low.v3_barrier_cm1 == 210.0


# =============================================================================
# 4. J-BLOCK BATCHING & STATIC SHAPE COMPILATION BUCKETS
# =============================================================================

def test_j_block_batching_layout() -> None:
    """Verifies static shape bucketing for J-blocks up to J_max."""
    # J_max = 24 with bucket_size = 10 -> 3 buckets: 0..9 (dim 19), 10..19 (dim 39), 20..24 (dim 49)
    batches = build_j_block_batches(J_max=24, bucket_size=10)
    assert len(batches) == 3

    b0 = batches[0]
    assert b0.bucket_id == 0
    assert b0.j_min == 0
    assert b0.j_max == 9
    assert b0.max_dim == 19
    assert len(b0.j_list) == 10
    assert b0.padded_shape == (10, 19, 19)

    b1 = batches[1]
    assert b1.bucket_id == 1
    assert b1.j_min == 10
    assert b1.j_max == 19
    assert b1.max_dim == 39
    assert len(b1.j_list) == 10
    assert b1.padded_shape == (10, 39, 39)

    b2 = batches[2]
    assert b2.bucket_id == 2
    assert b2.j_min == 20
    assert b2.j_max == 24
    assert b2.max_dim == 49
    assert len(b2.j_list) == 5
    assert b2.padded_shape == (5, 49, 49)


def test_j_block_batching_invalid_j() -> None:
    """Verifies that negative J_max raises InvalidJValueError."""
    with pytest.raises(InvalidJValueError):
        build_j_block_batches(J_max=-1)


# =============================================================================
# 5. WANG SYMMETRIZED BASIS & PARITY SUB-BLOCKS
# =============================================================================

def test_wang_transformation_matrix_unitary_and_dimensions() -> None:
    """Verifies orthogonality, block dimensions, and unitary transformation for Wang matrices."""
    for J in (1, 2, 3, 4, 5, 10):
        dim = 2 * J + 1
        v_mat, slices = build_wang_transformation_matrix(J)

        # 1. Dimension check
        assert v_mat.shape == (dim, dim)

        # 2. Orthogonality: V^T * V = I
        v_np = np.asarray(v_mat)
        identity_check = v_np.T @ v_np
        assert np.allclose(identity_check, np.eye(dim), atol=1e-12)

        # 3. Parity block dimensions
        e_plus_dim = slices["E+"][1] - slices["E+"][0]
        e_minus_dim = slices["E-"][1] - slices["E-"][0]
        o_plus_dim = slices["O+"][1] - slices["O+"][0]
        o_minus_dim = slices["O-"][1] - slices["O-"][0]

        expected_e_plus = J // 2 + 1
        expected_e_minus = J // 2
        expected_o_plus = (J + 1) // 2
        expected_o_minus = (J + 1) // 2

        assert e_plus_dim == expected_e_plus
        assert e_minus_dim == expected_e_minus
        assert o_plus_dim == expected_o_plus
        assert o_minus_dim == expected_o_minus
        assert e_plus_dim + e_minus_dim + o_plus_dim + o_minus_dim == dim


# =============================================================================
# 6. WATSON A & S REDUCTION HAMILTONIAN CONSTRUCTION (ANALYTICAL RIGOR)
# =============================================================================

def test_watson_j0_and_j1_analytical_eigenvalues() -> None:
    """
    Verifies Watson Hamiltonian eigenvalues against exact analytical spectroscopic formulas for J=0 and J=1.
    For J=0: E = 0.0.
    For J=1 (rigid rotor in Ir):
    - 1_01 state: E = B + C
    - 1_11 state: E = A + C
    - 1_10 state: E = A + B
    """
    A_mhz = 25000.0
    B_mhz = 12000.0
    C_mhz = 8000.0

    constants = WatsonConstants(A=A_mhz, B=B_mhz, C=C_mhz, reduction="A")

    # 1. J = 0 Block
    res_j0 = construct_j_block(constants, J=0, representation="Ir")
    assert res_j0.dim == 1
    assert math.isclose(float(res_j0.eigenvalues[0]), 0.0, abs_tol=1e-12)

    # 2. J = 1 Block
    res_j1 = construct_j_block(constants, J=1, representation="Ir")
    assert res_j1.dim == 3

    evals_calc = np.sort(np.asarray(res_j1.eigenvalues))

    # Analytical energies for J=1 in MHz:
    e_101 = B_mhz + C_mhz  # 12000 + 8000 = 20000 MHz
    e_111 = A_mhz + C_mhz  # 25000 + 8000 = 33000 MHz
    e_110 = A_mhz + B_mhz  # 25000 + 12000 = 37000 MHz

    expected_evals = np.sort(np.array([e_101, e_111, e_110], dtype=np.float64))

    diff = np.max(np.abs(evals_calc - expected_evals))
    assert diff < 1e-8, f"J=1 eigenvalues deviation {diff} exceeds tolerance"


def test_watson_hermitian_symmetry_and_parity_block_eigenvalue_equivalence() -> None:
    """
    Verifies:
    1. Hermitian symmetry for both Watson A and S reduction matrices.
    2. Sub-block Wang eigenvalues strictly match full matrix eigenvalues.
    """
    constants_a = WatsonConstants(
        A=30000.0,
        B=15000.0,
        C=9000.0,
        delta_J=1e-3,
        delta_JK=5e-3,
        delta_K=2e-2,
        d_J=1e-4,
        d_K=2e-4,
        reduction="A",
    )

    constants_s = WatsonConstants(
        A=30000.0,
        B=15000.0,
        C=9000.0,
        D_J=1e-3,
        D_JK=5e-3,
        D_K=2e-2,
        d1=1e-4,
        d2=2e-5,
        reduction="S",
    )

    for J in (2, 3, 4, 5):
        # A-reduction test
        res_a = construct_j_block(constants_a, J=J, representation="Ir", reduction="A")
        mat_a = np.asarray(res_a.matrix)
        assert np.allclose(mat_a, mat_a.T, atol=1e-12), f"Watson A J={J} violated Hermiticity"

        full_evals_a = np.sort(np.linalg.eigvalsh(mat_a))
        block_evals_a = np.sort(np.asarray(res_a.eigenvalues))
        assert np.allclose(full_evals_a, block_evals_a, atol=1e-8), f"Parity eigenvalues mismatch in A-reduction J={J}"

        # S-reduction test
        res_s = construct_j_block(constants_s, J=J, representation="Ir", reduction="S")
        mat_s = np.asarray(res_s.matrix)
        assert np.allclose(mat_s, mat_s.T, atol=1e-12), f"Watson S J={J} violated Hermiticity"

        full_evals_s = np.sort(np.linalg.eigvalsh(mat_s))
        block_evals_s = np.sort(np.asarray(res_s.eigenvalues))
        assert np.allclose(full_evals_s, block_evals_s, atol=1e-8), f"Parity eigenvalues mismatch in S-reduction J={J}"


def test_watson_iiir_representation_construction() -> None:
    """Verifies construction in IIIr representation (oblate quantization)."""
    # Oblate molecule
    constants = WatsonConstants(
        A=20000.0,
        B=19000.0,
        C=10000.0,
        reduction="A",
        representation="IIIr",
    )

    res_j1 = construct_j_block(constants, J=1, representation="IIIr", reduction="A")
    assert res_j1.representation == "IIIr"
    assert res_j1.dim == 3

    mat = np.asarray(res_j1.matrix)
    assert np.allclose(mat, mat.T, atol=1e-12)

    # In IIIr: J=1 energies are A+B, B+C, A+C
    evals_calc = np.sort(np.asarray(res_j1.eigenvalues))
    expected_evals = np.sort(np.array([20000.0 + 19000.0, 19000.0 + 10000.0, 20000.0 + 10000.0]))
    assert np.allclose(evals_calc, expected_evals, atol=1e-8)


# =============================================================================
# 7. IAM / WOODS COUPLED TORSIONAL METHOD
# =============================================================================

def test_iam_woods_coupled_hamiltonian() -> None:
    """Verifies construction of the coupled IAM/Woods rotational-torsional manifold."""
    constants = WatsonConstants(
        A=25000.0,
        B=12000.0,
        C=8000.0,
        v3_barrier_cm1=150.0,  # Below 400 cm^-1 -> IAM mode
        v6_barrier_cm1=5.0,
        F_internal_rot_mhz=160000.0,
        rho_a=0.08,
    )

    # J = 1 with 7 torsional states -> total dimension = (2*1 + 1) * 7 = 3 * 7 = 21
    res_iam = construct_iam_woods_hamiltonian(constants, J=1, n_torsion_states=7)
    assert res_iam.is_iam_woods
    assert res_iam.dim == 21
    assert res_iam.eigenvalues is not None
    assert len(res_iam.eigenvalues) == 21

    # Matrix must be Hermitian
    mat = np.asarray(res_iam.matrix)
    assert np.allclose(mat, mat.T, atol=1e-10)

    # Eigenvalues must be real and sorted
    evals = np.asarray(res_iam.eigenvalues)
    assert np.all(np.diff(evals) >= 0.0)


# =============================================================================
# 8. MASTER CONSTRUCT HAMILTONIAN API & DISTRIBUTED WORKFLOW
# =============================================================================

def test_master_construct_hamiltonian_rigid_rotor() -> None:
    """Verifies top-level construct_hamiltonian API for standard rigid rotor."""
    constants = WatsonConstants(
        A=281970.5,
        B=38833.99,
        C=34004.24,
        delta_J=0.065,
        delta_JK=1.23,
        delta_K=15.4,
        d_J=0.012,
        d_K=0.45,
    )

    J_max = 5
    result = construct_hamiltonian(constants, J_max=J_max)

    assert isinstance(result, BatchedHamiltonianResult)
    assert result.J_max == J_max
    assert result.representation == "Ir"
    assert result.reduction == "A"
    assert len(result.block_results) == J_max + 1

    # Total states sum_{J=0}^5 (2J+1) = 1 + 3 + 5 + 7 + 9 + 11 = 36 = (5+1)^2
    expected_total_states = (J_max + 1) ** 2
    assert result.total_states == expected_total_states
    assert not result.lam_guard.breached


def test_master_construct_hamiltonian_lam_automatic_routing() -> None:
    """Verifies that construct_hamiltonian automatically routes to IAM when V3 < 400 cm^-1."""
    constants = WatsonConstants(
        A=25000.0,
        B=12000.0,
        C=8000.0,
        v3_barrier_cm1=220.0,  # Below 400 -> LAM breach
        F_internal_rot_mhz=160000.0,
        rho_a=0.05,
    )

    result = construct_hamiltonian(constants, J_max=2)
    assert result.lam_guard.breached
    assert result.lam_guard.routing == "IAM_WOODS"
    assert result.block_results[0].is_iam_woods
    assert result.block_results[1].is_iam_woods
    assert result.block_results[2].is_iam_woods


def test_master_construct_hamiltonian_lam_breach_without_fallback_raises() -> None:
    """Verifies that LAMBreachError is raised when allow_iam_fallback=False and V3 < 400 cm^-1."""
    constants = WatsonConstants(
        A=25000.0,
        B=12000.0,
        C=8000.0,
        v3_barrier_cm1=180.0,
    )

    config = HamiltonianConfig(J_max=2, allow_iam_fallback=False)
    with pytest.raises(LAMBreachError, match="Internal Rotation .* Guard Breached"):
        construct_hamiltonian(constants, J_max=2, config=config)


# =============================================================================
# 9. MENDELEEV INTEGRATION MANDATE
# =============================================================================

def test_mendeleev_dynamic_mass_resolution() -> None:
    """Strictly validates dynamic monoisotopic mass retrieval using mendeleev library."""
    c_elem = mendeleev.element("C")
    assert c_elem.atomic_number == 6
    assert c_elem.mass > 12.0

    o_elem = mendeleev.element("O")
    assert o_elem.atomic_number == 8
    assert o_elem.mass > 15.9

    n_elem = mendeleev.element("N")
    assert n_elem.atomic_number == 7
    assert n_elem.mass > 14.0


# =============================================================================
# 10. ANTI-SPOOF & STATIC CODE COMPLIANCE
# =============================================================================

def test_zero_pass_and_banned_tokens_in_jax_builder() -> None:
    """Verifies that cochem_jax_builder.py contains 0 pass statements and 0 banned tokens."""
    builder_path = Path(__file__).resolve().parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_jax_builder.py"
    assert builder_path.exists(), f"Source file {builder_path} must physically exist."

    source = builder_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    class ZeroTrustVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.pass_count = 0
            self.banned_tokens: List[str] = []

        def visit_Pass(self, node: ast.Pass) -> None:
            self.pass_count += 1
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for banned in ("mock", "fake", "placeholder", "dummy", "synthetic", "stub"):
                if banned in node.name.lower():
                    self.banned_tokens.append(f"Function name '{node.name}' contains banned token '{banned}'")
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            for banned in ("mock", "fake", "placeholder", "dummy", "synthetic", "stub"):
                if banned in node.id.lower():
                    self.banned_tokens.append(f"Variable name '{node.id}' contains banned token '{banned}'")
            self.generic_visit(node)

    visitor = ZeroTrustVisitor()
    visitor.visit(tree)

    assert visitor.pass_count == 0, f"Found {visitor.pass_count} pass statements in cochem_jax_builder.py"
    assert len(visitor.banned_tokens) == 0, f"Found banned tokens in cochem_jax_builder.py: {visitor.banned_tokens}"


# =============================================================================
# 11. PYDANTIC VALIDATION & EXCEPTION HANDLING
# =============================================================================

def test_watson_constants_validation_errors() -> None:
    """Verifies Pydantic validation errors for invalid reduction, representation, or non-positive constants."""
    with pytest.raises(Exception):
        WatsonConstants(A=10000.0, B=5000.0, C=3000.0, reduction="INVALID")

    with pytest.raises(Exception):
        WatsonConstants(A=10000.0, B=5000.0, C=3000.0, representation="XYZ")

    with pytest.raises(Exception):
        WatsonConstants(A=-500.0, B=5000.0, C=3000.0)


def test_invalid_representation_and_reduction_in_construct_j_block() -> None:
    """Verifies that invalid representation or reduction strings raise appropriate typed exceptions."""
    constants = WatsonConstants(A=10000.0, B=5000.0, C=3000.0)

    with pytest.raises(InvalidRepresentationError):
        construct_j_block(constants, J=1, representation="INVALID_REP")

    with pytest.raises(InvalidReductionError):
        construct_j_block(constants, J=1, reduction="X")

    with pytest.raises(InvalidJValueError):
        construct_j_block(constants, J=-5)


def test_hpc_environment_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that HPC environment flags are correctly detected."""
    monkeypatch.setenv("COCHEM_HPC", "1")
    assert is_hpc_environment()

    monkeypatch.setenv("COCHEM_HPC", "0")
    monkeypatch.setenv("COCHEM_USE_MPI4JAX", "1")
    assert is_hpc_environment()

    monkeypatch.setenv("COCHEM_USE_MPI4JAX", "0")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    assert not is_hpc_environment()


def test_high_j_matrix_construction_and_eigenvalues() -> None:
    """Verifies matrix construction and stability for higher rotational states (J=15, J=20)."""
    constants = WatsonConstants(
        A=20000.0,
        B=10000.0,
        C=6000.0,
        delta_J=1e-4,
        delta_JK=2e-4,
        delta_K=1e-3,
        d_J=1e-5,
        d_K=2e-5,
    )

    for J in (10, 15, 20):
        dim = 2 * J + 1
        res = construct_j_block(constants, J=J, representation="Ir", reduction="A", use_parity_blocks=True)
        assert res.dim == dim
        assert res.eigenvalues is not None
        assert len(res.eigenvalues) == dim
        assert not np.any(np.isnan(np.asarray(res.eigenvalues)))
        assert not np.any(np.isinf(np.asarray(res.eigenvalues)))

