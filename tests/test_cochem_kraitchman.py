"""
Unit and Integration Test Suite for CoChem Kraitchman Analytical Substitution & Planar Moment Physics Core.

Validates Method Matrix §2.3, §2.4, §6.10 & WBS Phase 2.3:
1. Dynamic atomic and isotopic mass retrieval via Mendeleev library mandate.
2. Analytical calculation of effective reduced mass factor mu = (M * delta_m) / (M + delta_m).
3. Planar moment transformations P_aa, P_bb, P_cc and sum rules.
4. Automated planar moment shift sign verification (Delta P_aa >= 0, Delta P_bb >= 0, Delta P_cc >= 0) and inertial defect sign preservation.
5. Exact Kraitchman substitution coordinate calculations for planar and 3D asymmetric rotors.
6. Costain error bounds estimation (delta_z = 0.0015 / |z|) with small-coordinate cutoff diagnostics.
7. Rigorous Costain uncertainty propagation to bond lengths and bond angles.
8. Automated coordinate sign resolution using structural priors and decoupled 1st-moment center-of-mass conditions.
9. Full r_s substitution structure determination with center-of-mass and cross-moment validation.
10. Pure JAX FP64 hardware-accelerated batch filtering and tensor transformations.
11. Full analytical self-verification benchmark suite on authentic molecular systems (Water, Formaldehyde, Methanol).
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from cochem_spycfit.physics_core.cochem_kraitchman import (
    AMU_ANGSTROM2_TO_MHZ,
    CONV_MHZ_AMU_ANG2,
    CODATA_CONV_EXACT,
    DEFAULT_COSTAIN_PARAM,
    DEFAULT_SMALL_COORD_CUTOFF,
    DEFAULT_PLANAR_TOL,
    DEFAULT_SIGN_VERIFY_TOL,
    DEFAULT_ZERO_TOL,
    PROVENANCE_TAG,
    SECTION_REF,
    BondAngleUncertainty,
    BondLengthUncertainty,
    ImaginaryCoordinateError,
    IncompleteSubstitutionSetError,
    IsotopicSubstitutionData,
    KraitchmanCoordinateResult,
    KraitchmanError,
    KraitchmanStructureReport,
    MomentsOfInertia,
    PlanarMoments,
    PlanarMomentShiftValidation,
    PlanarSignViolationError,
    RotationalConstants,
    SmallCoordinateBreakdownWarning,
    UnphysicalSubstitutionShiftError,
    calculate_reduced_mass_change,
    compute_cartesian_inertia_tensor,
    compute_planar_moments_analytical,
    estimate_costain_error_bounds,
    get_dynamic_atomic_mass,
    jax_calculate_reduced_mass,
    jax_center_of_mass_residual_batch,
    jax_costain_uncertainties_batch,
    jax_kraitchman_coordinates_batch,
    jax_planar_moments,
    moments_from_rotational_constants,
    propagate_bond_angle_uncertainty,
    propagate_bond_length_uncertainty,
    resolve_coordinate_signs_combinatorial,
    resolve_coordinate_signs_with_prior,
    resolve_mass_array,
    rotational_constants_from_moments,
    run_kraitchman_verification_suite,
    solve_full_rs_structure,
    solve_kraitchman_single_substitution,
    verify_planar_moment_shifts,
)


# =============================================================================
# 1. ATOMIC MASS & PHYSICAL CONSTANTS MANDATE TESTS
# =============================================================================

def test_dynamic_atomic_mass_resolution() -> None:
    """Verifies atomic and isotopic masses are retrieved dynamically from mendeleev."""
    m_h = get_dynamic_atomic_mass("H")
    m_d = get_dynamic_atomic_mass("D")
    m_t = get_dynamic_atomic_mass("T")
    m_c = get_dynamic_atomic_mass("C")
    m_13c = get_dynamic_atomic_mass("13C")
    m_o = get_dynamic_atomic_mass("O")
    m_18o = get_dynamic_atomic_mass("18O")

    assert abs(m_h - 1.007825) < 0.01
    assert abs(m_d - 2.014101) < 0.001
    assert abs(m_t - 3.016049) < 0.001
    assert abs(m_c - 12.011) < 0.02
    assert abs(m_13c - 13.00335) < 0.001
    assert abs(m_o - 15.999) < 0.01
    assert abs(m_18o - 17.99916) < 0.001

    # By atomic number
    assert abs(get_dynamic_atomic_mass(1) - 1.008) < 0.01
    assert abs(get_dynamic_atomic_mass(6, mass_number=13) - 13.00335) < 0.001


def test_resolve_mass_array() -> None:
    """Verifies bulk resolution of molecular atomic masses."""
    masses = resolve_mass_array(["C", "O", "H", "H"])
    assert len(masses) == 4
    assert abs(masses[0] - 12.011) < 0.02
    assert abs(masses[1] - 15.999) < 0.01
    assert abs(masses[2] - 1.008) < 0.01


def test_physical_constants_definitions() -> None:
    """Verifies spectroscopic conversion factors comply with CODATA and Method Matrix §2.3."""
    assert abs(AMU_ANGSTROM2_TO_MHZ - 505379.008) < 1e-3
    assert abs(CONV_MHZ_AMU_ANG2 - 505379.008) < 1e-3
    assert abs(CODATA_CONV_EXACT - 505379.00536) < 1e-3
    assert DEFAULT_COSTAIN_PARAM == 0.0015
    assert DEFAULT_SMALL_COORD_CUTOFF == 0.15
    assert PROVENANCE_TAG == "[M]"


# =============================================================================
# 2. REDUCED MASS & CONVERSION EQUATIONS TESTS
# =============================================================================

def test_calculate_reduced_mass_change() -> None:
    """Tests the reduced mass calculation mu = (M * delta_m) / (M + delta_m)."""
    m_parent = 18.015
    delta_m = 1.006277  # H -> D
    mu = calculate_reduced_mass_change(m_parent, delta_m)
    expected_mu = (m_parent * delta_m) / (m_parent + delta_m)
    assert abs(mu - expected_mu) < 1e-12

    with pytest.raises(ValueError, match="Total parent mass must be strictly positive"):
        calculate_reduced_mass_change(-10.0, 1.0)

    with pytest.raises(ValueError, match="Substituted mass cannot be zero or negative"):
        calculate_reduced_mass_change(10.0, -10.0)


def test_rotational_constants_and_moments_conversions() -> None:
    """Verifies bi-directional conversions between moments (amu*A^2) and constants (MHz)."""
    A_in, B_in, C_in = 27000.0, 14000.0, 9000.0
    moms = moments_from_rotational_constants(A_in, B_in, C_in)
    assert moms.Ia > 0.0 and moms.Ib > 0.0 and moms.Ic > 0.0
    assert moms.Ia < moms.Ib < moms.Ic

    rc = rotational_constants_from_moments(moms.Ia, moms.Ib, moms.Ic)
    assert abs(rc.A - A_in) < 1e-6
    assert abs(rc.B - B_in) < 1e-6
    assert abs(rc.C - C_in) < 1e-6

    # Asymmetry parameter
    kappa_moms = moms.ray_asymmetry
    kappa_rc = rc.ray_asymmetry
    assert abs(kappa_moms - kappa_rc) < 1e-10


# =============================================================================
# 3. PLANAR MOMENTS & SIGN VERIFICATION TESTS
# =============================================================================

def test_planar_moments_analytical_and_sum_rules() -> None:
    """Verifies analytical planar moment relationships and sum rules."""
    Ia, Ib, Ic = 10.0, 25.0, 35.0  # Planar molecule where Ic = Ia + Ib
    pm = compute_planar_moments_analytical(Ia, Ib, Ic)

    # P_aa = 0.5 * (-Ia + Ib + Ic) = 25.0
    # P_bb = 0.5 * ( Ia - Ib + Ic) = 10.0
    # P_cc = 0.5 * ( Ia + Ib - Ic) = 0.0
    assert abs(pm.P_aa - 25.0) < 1e-12
    assert abs(pm.P_bb - 10.0) < 1e-12
    assert abs(pm.P_cc - 0.0) < 1e-12
    assert abs(pm.inertial_defect - 0.0) < 1e-12

    # Invert back to principal moments
    Ia_rec, Ib_rec, Ic_rec = pm.to_principal_moments()
    assert abs(Ia_rec - Ia) < 1e-12
    assert abs(Ib_rec - Ib) < 1e-12
    assert abs(Ic_rec - Ic) < 1e-12


def test_verify_planar_moment_shifts_valid_and_invalid() -> None:
    """Verifies automated planar moment shift sign checking (Method Matrix §2.3/§6.10)."""
    p_parent = MomentsOfInertia(Ia=10.0, Ib=20.0, Ic=30.0)
    # Valid substitution: mass added along a and b axes
    p_sub_valid = MomentsOfInertia(Ia=10.5, Ib=21.0, Ic=31.5)
    val_valid = verify_planar_moment_shifts(p_parent, p_sub_valid)
    assert val_valid.is_physically_valid
    assert val_valid.is_P_aa_positive
    assert val_valid.is_P_bb_positive
    assert val_valid.is_P_cc_positive

    # Unphysical substitution: negative planar moment shift
    p_sub_invalid = MomentsOfInertia(Ia=15.0, Ib=18.0, Ic=28.0)
    val_invalid = verify_planar_moment_shifts(p_parent, p_sub_invalid)
    assert not val_invalid.is_physically_valid
    assert len(val_invalid.diagnostic_messages) > 0


# =============================================================================
# 4. COSTAIN ERROR BOUNDS ESTIMATION TESTS
# =============================================================================

def test_estimate_costain_error_bounds() -> None:
    """Verifies Costain error formula delta_z = 0.0015 / |z| and small-coord cutoff."""
    # Normal coordinate >= 0.15 A
    z_large = 1.50
    unc_large, is_small_large = estimate_costain_error_bounds(z_large)
    assert abs(unc_large - 0.001) < 1e-6
    assert not is_small_large

    # Moderate coordinate
    z_med = 0.50
    unc_med, is_small_med = estimate_costain_error_bounds(z_med)
    assert abs(unc_med - 0.003) < 1e-6
    assert not is_small_med

    # Small coordinate < 0.15 A
    z_small = 0.05
    unc_small, is_small_small = estimate_costain_error_bounds(z_small)
    assert is_small_small
    assert abs(unc_small - (0.0015 / 0.05)) < 1e-6

    # Coordinate clamped at floor 1e-4
    z_zero = 0.00001
    unc_zero, is_small_zero = estimate_costain_error_bounds(z_zero)
    assert is_small_zero
    assert abs(unc_zero - (0.0015 / 1e-4)) < 1e-6


# =============================================================================
# 5. CARTESIAN INERTIA TENSOR & KRAITCHMAN EQUATIONS TESTS
# =============================================================================

def test_compute_cartesian_inertia_tensor() -> None:
    """Verifies center of mass translation and inertia tensor diagonalization."""
    # Linear CO molecule: C at (0, 0, 0), O at (0, 0, 1.1283 A)
    m_c = get_dynamic_atomic_mass("C")
    m_o = get_dynamic_atomic_mass("O")
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.1283]
    ])
    masses = np.array([m_c, m_o])

    evals, evecs, com = compute_cartesian_inertia_tensor(coords, masses)
    assert evals[0] < 1e-6  # Linear axis has zero moment of inertia
    assert abs(evals[1] - evals[2]) < 1e-6  # Degenerate transverse axes
    assert com[2] > 0.0 and com[2] < 1.1283


def test_solve_kraitchman_single_substitution_water() -> None:
    """Verifies Kraitchman substitution for H2O -> HDO on authentic molecular geometry."""
    m_O = get_dynamic_atomic_mass("O")
    m_H = get_dynamic_atomic_mass("H")
    m_D = get_dynamic_atomic_mass("D")

    theta_half = math.radians(104.51 / 2.0)
    x_H = 0.9575 * math.sin(theta_half)
    y_H = 0.9575 * math.cos(theta_half)
    raw_coords = np.array([
        [0.0, 0.0, 0.0],
        [x_H, y_H, 0.0],
        [-x_H, y_H, 0.0]
    ])
    masses = np.array([m_O, m_H, m_H])
    evals_parent, evecs, com = compute_cartesian_inertia_tensor(raw_coords, masses)
    pcoords = (raw_coords - np.array(com)) @ evecs

    # Substituted H1 -> D
    masses_sub = masses.copy()
    masses_sub[1] = m_D
    evals_sub, _, _ = compute_cartesian_inertia_tensor(raw_coords, masses_sub)

    sub_data = IsotopicSubstitutionData(
        atom_index=1,
        parent_symbol="H",
        substituted_symbol="D",
        parent_mass=m_H,
        substituted_mass=m_D,
        parent_moments=MomentsOfInertia(Ia=evals_parent[0], Ib=evals_parent[1], Ic=evals_parent[2]),
        substituted_moments=MomentsOfInertia(Ia=evals_sub[0], Ib=evals_sub[1], Ic=evals_sub[2])
    )

    res = solve_kraitchman_single_substitution(
        sub_data,
        total_mass_parent=float(np.sum(masses)),
        prior_coord=pcoords[1]
    )

    assert abs(res.abs_a - abs(pcoords[1, 0])) < 1e-6
    assert abs(res.abs_b - abs(pcoords[1, 1])) < 1e-6
    assert abs(res.abs_c - abs(pcoords[1, 2])) < 1e-6
    assert res.planar_validation.is_physically_valid


# =============================================================================
# 6. UNCERTAINTY PROPAGATION TESTS (BOND LENGTHS & ANGLES)
# =============================================================================

def test_propagate_bond_length_uncertainty() -> None:
    """Verifies analytical error propagation from Costain uncertainties to bond lengths."""
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([1.20, 0.0, 0.0])
    u1 = np.array([0.002, 0.005, 0.005])
    u2 = np.array([0.003, 0.004, 0.004])

    # Bond along x-axis: delta_R = sqrt(u1_x^2 + u2_x^2) = sqrt(0.002^2 + 0.003^2)
    unc = propagate_bond_length_uncertainty(r1, r2, u1, u2)
    expected_unc = math.sqrt(0.002**2 + 0.003**2)
    assert abs(unc - expected_unc) < 1e-6


def test_propagate_bond_angle_uncertainty() -> None:
    """Verifies analytical error propagation from Costain uncertainties to bond angles."""
    r_i = np.array([1.0, 0.0, 0.0])
    r_j = np.array([0.0, 0.0, 0.0])  # Vertex
    r_k = np.array([0.0, 1.0, 0.0])
    u_i = np.array([0.001, 0.001, 0.001])
    u_j = np.array([0.001, 0.001, 0.001])
    u_k = np.array([0.001, 0.001, 0.001])

    unc_deg = propagate_bond_angle_uncertainty(r_i, r_j, r_k, u_i, u_j, u_k)
    assert unc_deg > 0.0
    assert unc_deg < 5.0  # Plausible physical angular uncertainty in degrees


# =============================================================================
# 7. COORDINATE SIGN RESOLUTION & FULL STRUCTURE REPORT TESTS
# =============================================================================

def test_resolve_coordinate_signs_combinatorial() -> None:
    """Verifies combinatorial sign resolution via 1st-moment center of mass minimization."""
    # 1D symmetric diatomic coordinates: C at -0.60, O at +0.60
    masses = np.array([12.0, 12.0])
    abs_coords = np.array([
        [0.60, 0.0, 0.0],
        [0.60, 0.0, 0.0]
    ])
    signed = resolve_coordinate_signs_combinatorial(abs_coords, masses)
    com_residual = np.sum(masses * signed[:, 0])
    assert abs(com_residual) < 1e-6
    assert (signed[0, 0] > 0 and signed[1, 0] < 0) or (signed[0, 0] < 0 and signed[1, 0] > 0)


def test_solve_full_rs_structure_formaldehyde() -> None:
    """Verifies full r_s determination on Formaldehyde (H2CO)."""
    m_C = get_dynamic_atomic_mass("C")
    m_13C = get_dynamic_atomic_mass("13C")
    m_O = get_dynamic_atomic_mass("O")
    m_18O = get_dynamic_atomic_mass("18O")
    m_H = get_dynamic_atomic_mass("H")
    m_D = get_dynamic_atomic_mass("D")

    theta_half = math.radians(116.52 / 2.0)
    raw_coords = np.array([
        [0.0, 0.0, 0.0],                                      # C (atom 0)
        [0.0, 1.2078, 0.0],                                   # O (atom 1)
        [1.1171 * math.sin(theta_half), -1.1171 * math.cos(theta_half), 0.0],  # H1 (atom 2)
        [-1.1171 * math.sin(theta_half), -1.1171 * math.cos(theta_half), 0.0]  # H2 (atom 3)
    ])
    masses = np.array([m_C, m_O, m_H, m_H])
    evals_parent, evecs, com = compute_cartesian_inertia_tensor(raw_coords, masses)
    pcoords = (raw_coords - np.array(com)) @ evecs

    masses_c = masses.copy(); masses_c[0] = m_13C
    evals_c, _, _ = compute_cartesian_inertia_tensor(raw_coords, masses_c)

    masses_o = masses.copy(); masses_o[1] = m_18O
    evals_o, _, _ = compute_cartesian_inertia_tensor(raw_coords, masses_o)

    masses_h = masses.copy(); masses_h[2] = m_D
    evals_h, _, _ = compute_cartesian_inertia_tensor(raw_coords, masses_h)

    subs = [
        IsotopicSubstitutionData(
            atom_index=0, parent_symbol="C", substituted_symbol="13C",
            parent_mass=m_C, substituted_mass=m_13C,
            parent_moments=MomentsOfInertia(Ia=evals_parent[0], Ib=evals_parent[1], Ic=evals_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_c[0], Ib=evals_c[1], Ic=evals_c[2])
        ),
        IsotopicSubstitutionData(
            atom_index=1, parent_symbol="O", substituted_symbol="18O",
            parent_mass=m_O, substituted_mass=m_18O,
            parent_moments=MomentsOfInertia(Ia=evals_parent[0], Ib=evals_parent[1], Ic=evals_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_o[0], Ib=evals_o[1], Ic=evals_o[2])
        ),
        IsotopicSubstitutionData(
            atom_index=2, parent_symbol="H", substituted_symbol="D",
            parent_mass=m_H, substituted_mass=m_D,
            parent_moments=MomentsOfInertia(Ia=evals_parent[0], Ib=evals_parent[1], Ic=evals_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_h[0], Ib=evals_h[1], Ic=evals_h[2])
        ),
    ]

    report = solve_full_rs_structure(
        molecular_formula="H2CO",
        atom_symbols=["C", "O", "H", "H"],
        substitutions=subs,
        prior_geometry=pcoords,
        bonds=[(0, 1), (0, 2), (0, 3)],
        angles=[(2, 0, 3), (1, 0, 2)]
    )

    assert report.all_planar_valid
    assert len(report.bond_lengths) == 3
    assert len(report.bond_angles) == 2
    assert report.provenance == "[M]"
    assert "Method Matrix" in report.method_matrix_ref

    # Check C=O bond length accuracy: 1.2078 A
    co_bond = [b for b in report.bond_lengths if (b.atom_i, b.atom_j) in [(0, 1), (1, 0)]][0]
    assert abs(co_bond.length_angstrom - 1.2078) < 1e-4
    assert co_bond.uncertainty_angstrom > 0.0


# =============================================================================
# 8. PURE JAX FP64 HARDWARE-ACCELERATED VECTORIZATION TESTS
# =============================================================================

def test_jax_fp64_batch_functions() -> None:
    """Verifies pure JAX FP64 batch Kraitchman functions against NumPy reference."""
    total_mass_b = jnp.array([18.015, 30.026], dtype=jnp.float64)
    delta_mass_b = jnp.array([1.006277, 0.99318], dtype=jnp.float64)
    mu_jax = jax_calculate_reduced_mass(total_mass_b, delta_mass_b)
    assert len(mu_jax) == 2

    # Planar moments batch
    Ia = jnp.array([10.0, 20.0], dtype=jnp.float64)
    Ib = jnp.array([25.0, 40.0], dtype=jnp.float64)
    Ic = jnp.array([35.0, 60.0], dtype=jnp.float64)
    Paa, Pbb, Pcc = jax_planar_moments(Ia, Ib, Ic)
    assert jnp.allclose(Paa, jnp.array([25.0, 40.0], dtype=jnp.float64))

    # Costain uncertainties batch
    coords_b = jnp.array([[1.5, 0.5, 0.05], [0.01, 1.0, 2.0]], dtype=jnp.float64)
    unc_b = jax_costain_uncertainties_batch(coords_b)
    assert unc_b.shape == (2, 3)
    assert abs(float(unc_b[0, 0]) - 0.001) < 1e-6


# =============================================================================
# 9. SELF-VERIFICATION SUITE EXECUTION
# =============================================================================

def test_run_kraitchman_verification_suite() -> None:
    """Executes the full built-in analytical self-verification benchmark suite."""
    suite_res = run_kraitchman_verification_suite()
    assert "Water_HDO" in suite_res
    assert suite_res["Water_HDO"]["status"] == "PASSED"
    assert "Formaldehyde_Full_rs" in suite_res
    assert suite_res["Formaldehyde_Full_rs"]["status"] == "PASSED"
    assert "Methanol_3D" in suite_res
    assert suite_res["Methanol_3D"]["status"] == "PASSED"
    assert "JAX_FP64_Acceleration" in suite_res
    assert suite_res["JAX_FP64_Acceleration"]["status"] == "PASSED"
