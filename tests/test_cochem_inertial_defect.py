"""
Unit and Integration Test Suite for CoChem Inertial Defect & Planar Moments.

Validates Method Matrix Step 4 (§1.3, §6.4, §6.10) & WBS Phase 3.1:
1. Analytical computation of inertial defect Delta = I_c - I_a - I_b (in u*Angstrom^2).
2. Analytical computation of planar moments Paa, Pbb, Pcc and validation of sum rules:
   Paa + Pbb = Ic, Paa + Pcc = Ib, Pbb + Pcc = Ia, Delta = -2 * Pcc.
3. Conversion of rotational constants A, B, C (in MHz) to principal moments.
4. Exact physical sign validation: sign(Delta_calc) == sign(Delta_exp).
5. Rejection of unphysical sign inversions (e.g. planar calc vs non-planar exp).
6. Planar moment hierarchy validation: Paa >= Pbb >= Pcc.
7. Axis permutation detection (canonical, ab_swapped, bc_swapped, ac_swapped, unphysical_floor).
8. Structural gate evaluation dictionary and report generation.
9. Filtering unphysical candidate assignments and 3D trajectory coordinate batches.
10. Hardware-accelerated pure-JAX vectorized computations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_inertial_defect import (
    AMU_ANGSTROM2_TO_MHZ,
    DEFAULT_CONV_FACTOR_MHZ,
    DEFAULT_MIN_PCC_FLOOR,
    DEFAULT_PLANAR_TOL,
    DEFAULT_ZERO_TOL,
    InertialDefectError,
    InertialDefectSignError,
    InertialDefectValidator,
    MomentsOfInertia,
    PermutationError,
    PlanarHierarchyError,
    PlanarMoments,
    RotationalConstants,
    StructuralGateReport,
    UnphysicalAssignmentError,
    compute_inertial_defect,
    compute_inertial_defect_from_constants,
    compute_inertial_defect_jax,
    compute_moments_of_inertia,
    compute_planar_moments,
    compute_planar_moments_jax,
    detect_planar_permutation,
    evaluate_structural_gate,
    filter_unphysical_assignments,
    filter_unphysical_constants_jax,
    filter_unphysical_trajectories,
    get_defect_sign,
    get_dynamic_atomic_mass,
    moments_from_rotational_constants,
    resolve_inertial_defect,
    resolve_mass_array,
    resolve_planar_moments,
    rotational_constants_from_moments,
    validate_inertial_defect_sign,
    validate_planar_hierarchy,
)

logger = logging.getLogger("Test.InertialDefect")


# =============================================================================
# 1. ATOMIC MASS RESOLUTION & PHYSICAL CONSTANTS TESTS
# =============================================================================

def test_dynamic_atomic_mass_retrieval() -> None:
    """Verifies atomic masses are retrieved dynamically via mendeleev."""
    o_mass = get_dynamic_atomic_mass("O")
    h_mass = get_dynamic_atomic_mass("H")
    c_mass = get_dynamic_atomic_mass("C")
    n_mass = get_dynamic_atomic_mass("N")

    assert abs(o_mass - 15.999) < 0.01
    assert abs(h_mass - 1.008) < 0.01
    assert abs(c_mass - 12.011) < 0.02
    assert abs(n_mass - 14.007) < 0.01

    # Isotopes
    d_mass = get_dynamic_atomic_mass("D")
    assert abs(d_mass - 2.0141) < 0.001
    c13_mass = get_dynamic_atomic_mass("13C")
    assert abs(c13_mass - 13.0033) < 0.001

    # Atomic number
    fe_mass = get_dynamic_atomic_mass(26)
    assert abs(fe_mass - 55.845) < 0.05


def test_resolve_mass_array() -> None:
    """Verifies mass array construction for molecular systems."""
    elements = ["O", "H", "H"]
    masses = resolve_mass_array(elements)
    assert len(masses) == 3
    assert abs(masses[0] - 15.999) < 0.01
    assert abs(masses[1] - 1.008) < 0.01


# =============================================================================
# 2. INERTIAL DEFECT & PLANAR MOMENTS COMPUTATION TESTS
# =============================================================================

def test_inertial_defect_from_explicit_moments() -> None:
    """Verifies Delta = Ic - Ia - Ib from moments."""
    validator = InertialDefectValidator()

    # Planar system: Ia=10.0, Ib=20.0, Ic=30.0 -> Delta = 0.0
    delta_zero = validator.compute_inertial_defect(10.0, 20.0, 30.0)
    assert abs(delta_zero - 0.0) < 1e-10

    # Non-planar system: Ia=10.0, Ib=20.0, Ic=25.0 -> Delta = -5.0
    delta_neg = validator.compute_inertial_defect(10.0, 20.0, 25.0)
    assert abs(delta_neg - (-5.0)) < 1e-10

    # Vibrational ground state planar: Ia=10.0, Ib=20.0, Ic=30.05 -> Delta = +0.05
    delta_pos = validator.compute_inertial_defect(10.0, 20.0, 30.05)
    assert abs(delta_pos - 0.05) < 1e-10


def test_planar_moments_and_sum_rules() -> None:
    """Verifies planar moments Paa, Pbb, Pcc and algebraic sum rules."""
    validator = InertialDefectValidator()
    Ia, Ib, Ic = 10.0, 20.0, 30.05

    paa, pbb, pcc = validator.compute_planar_moments(Ia, Ib, Ic)

    # Definitions
    assert abs(paa - 0.5 * (-Ia + Ib + Ic)) < 1e-10
    assert abs(pbb - 0.5 * (Ia - Ib + Ic)) < 1e-10
    assert abs(pcc - 0.5 * (Ia + Ib - Ic)) < 1e-10

    # Sum rules
    assert abs((paa + pbb) - Ic) < 1e-10
    assert abs((paa + pcc) - Ib) < 1e-10
    assert abs((pbb + pcc) - Ia) < 1e-10
    assert abs((paa + pbb + pcc) - 0.5 * (Ia + Ib + Ic)) < 1e-10

    # Delta = -2 * Pcc identity
    delta = validator.compute_inertial_defect(Ia, Ib, Ic)
    assert abs(delta - (-2.0 * pcc)) < 1e-10


def test_inertial_defect_from_experimental_rotational_constants() -> None:
    """Verifies Delta from authentic experimental rotational constants (MHz)."""
    validator = InertialDefectValidator()

    # Pyridine: A=6039.2, B=5804.9, C=2959.2 MHz (Planar, Delta > 0)
    delta_pyr = validator.compute_inertial_defect(rot_constants=(6039.2, 5804.9, 2959.2))
    assert delta_pyr > 0.0
    assert abs(delta_pyr - 0.0384) < 0.005

    # Water monomer r0: A=835840.0, B=435350.0, C=278140.0 MHz (Planar, Delta ~ +0.05)
    delta_water = validator.compute_inertial_defect(rot_constants=(835840.0, 435350.0, 278140.0))
    assert delta_water > 0.0
    assert abs(delta_water - 0.0515) < 0.005

    # Water dimer: A=204000.0, B=6160.0, C=6000.0 MHz (Non-planar, Delta < 0)
    delta_dimer = validator.compute_inertial_defect(rot_constants=(204000.0, 6160.0, 6000.0))
    assert delta_dimer < 0.0
    assert abs(delta_dimer - (-0.289)) < 0.02


def test_inertial_defect_from_cartesian_coordinates() -> None:
    """Verifies Delta from 3D Cartesian coordinates and Mendeleev masses."""
    validator = InertialDefectValidator()

    # Planar water geometry (z=0)
    water_coords = np.array([
        [0.000000,  0.000000, 0.000000],
        [0.957200,  0.000000, 0.000000],
        [-0.239987, 0.926627, 0.000000]
    ], dtype=np.float64)
    water_masses = resolve_mass_array(["O", "H", "H"])

    delta_planar = validator.compute_inertial_defect(coords=water_coords, masses=water_masses)
    assert abs(delta_planar) < 1e-8

    # Non-planar ammonia geometry (pyramidal, z != 0)
    nh3_coords = np.array([
        [0.000000,  0.000000,  0.116489],
        [0.000000,  0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808]
    ], dtype=np.float64)
    nh3_masses = resolve_mass_array(["N", "H", "H", "H"])

    delta_nh3 = validator.compute_inertial_defect(coords=nh3_coords, masses=nh3_masses)
    assert delta_nh3 < -0.1


# =============================================================================
# 3. BINDING SIGN VALIDATION TESTS (§6.4 & §6.10)
# =============================================================================

def test_validate_sign_matching_planar() -> None:
    """Verifies sign validation succeeds when both systems are planar."""
    validator = InertialDefectValidator()
    pyridine_rot = (6039.2, 5804.9, 2959.2)
    formaldehyde_rot = (281970.6, 38833.9, 34004.2)
    water_rot = (835840.0, 435350.0, 278140.0)

    assert validator.validate_sign(pyridine_rot, formaldehyde_rot) is True
    assert validator.validate_sign(formaldehyde_rot, water_rot) is True
    assert validator.validate_sign(0.045, 0.052) is True


def test_validate_sign_matching_non_planar() -> None:
    """Verifies sign validation succeeds when both systems are non-planar."""
    validator = InertialDefectValidator()
    water_dimer_rot = (204000.0, 6160.0, 6000.0)
    delta_nh3 = -1.214
    delta_ch4 = -2.85

    assert validator.validate_sign(water_dimer_rot, delta_nh3) is True
    assert validator.validate_sign(delta_nh3, delta_ch4) is True


def test_validate_sign_rejects_sign_inversions() -> None:
    """Verifies sign validation rejects planar vs non-planar contradictions."""
    validator = InertialDefectValidator()
    pyridine_rot = (6039.2, 5804.9, 2959.2)       # Delta > 0 (Planar)
    water_dimer_rot = (204000.0, 6160.0, 6000.0)  # Delta < 0 (Non-planar)

    assert validator.validate_sign(pyridine_rot, water_dimer_rot) is False
    assert validator.validate_sign(water_dimer_rot, pyridine_rot) is False
    assert validator.validate_sign(+0.055, -0.320) is False
    assert validator.validate_sign(-0.320, +0.055) is False


def test_validate_sign_raises_exception_when_requested() -> None:
    """Verifies raise_on_error=True raises InertialDefectSignError on conflict."""
    validator = InertialDefectValidator()
    with pytest.raises(InertialDefectSignError) as exc_info:
        validator.validate_sign(+0.048, -0.550, raise_on_error=True)
    assert "UNPHYSICAL INERTIAL DEFECT SIGN" in str(exc_info.value)


def test_validate_sign_equilibrium_to_ground_state_matching() -> None:
    """Verifies equilibrium planar (Delta_e = 0) matches vibrational ground state (Delta_0 > 0)."""
    validator = InertialDefectValidator()
    delta_re = 0.000000
    delta_r0 = +0.051500

    assert validator.validate_sign(delta_re, delta_r0, allow_equilibrium_planar_match=True) is True
    assert validator.validate_sign(delta_r0, delta_re, allow_equilibrium_planar_match=True) is True
    assert validator.validate_sign(delta_re, -0.500) is False


# =============================================================================
# 4. PLANAR HIERARCHY & PERMUTATION TESTS
# =============================================================================

def test_validate_planar_hierarchy_valid() -> None:
    """Verifies physical ordering Paa >= Pbb >= Pcc on valid systems."""
    validator = InertialDefectValidator()
    assert validator.validate_planar_hierarchy((6039.2, 5804.9, 2959.2)) is True
    assert validator.validate_planar_hierarchy((835840.0, 435350.0, 278140.0)) is True
    assert validator.validate_planar_hierarchy((204000.0, 6160.0, 6000.0)) is True


def test_resolve_planar_moments_formats() -> None:
    """Verifies planar moment resolution across dicts, tuples, and kwargs."""
    validator = InertialDefectValidator()

    # From rotational constants dict
    pyr_dict_rot = {"A": 6039.2, "B": 5804.9, "C": 2959.2}
    paa1, pbb1, pcc1 = validator.resolve_planar_moments(pyr_dict_rot)
    assert paa1 >= pbb1 >= pcc1

    # From moments dict
    pyr_dict_mom = {"Ia": 10.0, "Ib": 20.0, "Ic": 30.0}
    paa2, pbb2, pcc2 = validator.resolve_planar_moments(pyr_dict_mom)
    assert (paa2, pbb2, pcc2) == (20.0, 10.0, 0.0)

    # From explicit kwargs
    paa3, pbb3, pcc3 = validator.resolve_planar_moments(Paa=25.0, Pbb=15.0, Pcc=5.0)
    assert (paa3, pbb3, pcc3) == (25.0, 15.0, 5.0)


def test_detect_planar_permutations() -> None:
    """Verifies detection of canonical ordering and axis swaps."""
    validator = InertialDefectValidator()

    # Canonical: Ia <= Ib <= Ic -> Paa >= Pbb >= Pcc
    perm1, diag1 = validator.detect_planar_permutation((10.0, 20.0, 30.0))
    assert perm1 == "canonical"
    assert diag1["is_canonical"] is True

    # ab_swapped: Ib < Ia <= Ic -> Paa < Pbb
    perm2, diag2 = validator.detect_planar_permutation((30.0, 10.0, 40.0))
    assert perm2 == "ab_swapped"
    assert diag2["ordering_ab_valid"] is False

    # bc_swapped
    perm3, diag3 = validator.detect_planar_permutation((10.0, 50.0, 30.0))
    assert perm3 == "bc_swapped"
    assert diag3["ordering_bc_valid"] is False

    # unphysical_floor: Pcc < min_pcc_floor
    perm4, diag4 = validator.detect_planar_permutation((50.0, 40.0, -1.0), is_planar_moments=True)
    assert perm4 == "unphysical_floor"
    assert diag4["pcc_floor_valid"] is False


def test_validate_planar_hierarchy_rejection() -> None:
    """Verifies rejection of inverted moments and hierarchy violation exception."""
    validator = InertialDefectValidator()
    inverted = (50.0, 20.0, 30.0)
    assert validator.validate_planar_hierarchy(inverted) is False

    with pytest.raises(PlanarHierarchyError) as exc_info:
        validator.validate_planar_hierarchy(inverted, raise_on_error=True)
    assert "PLANAR MOMENT HIERARCHY VIOLATION" in str(exc_info.value)


# =============================================================================
# 5. STRUCTURAL GATE & FILTERING TESTS
# =============================================================================

def test_evaluate_structural_gate() -> None:
    """Verifies comprehensive structural gate report."""
    validator = InertialDefectValidator()
    pyridine_rot = (6039.2, 5804.9, 2959.2)
    formaldehyde_rot = (281970.6, 38833.9, 34004.2)
    water_dimer_rot = (204000.0, 6160.0, 6000.0)

    # Valid matching planar pair
    gate_planar = validator.evaluate_structural_gate(calc_input=pyridine_rot, exp_input=formaldehyde_rot)
    assert gate_planar["is_valid"] is True
    assert gate_planar["sign_valid"] is True
    assert gate_planar["is_planar_calc"] is True
    assert gate_planar["is_planar_exp"] is True
    assert gate_planar["planar_hierarchy_valid"] is True

    # Conflicting pair
    gate_conflict = validator.evaluate_structural_gate(calc_input=pyridine_rot, exp_input=water_dimer_rot)
    assert gate_conflict["is_valid"] is False
    assert gate_conflict["sign_valid"] is False


def test_filter_unphysical_assignments() -> None:
    """Verifies batch candidate filtering for rotational constant candidates."""
    cand1 = (6039.2, 5804.9, 2959.2)        # Valid planar
    cand2 = (281970.6, 38833.9, 34004.2)    # Valid planar
    cand3 = (204000.0, 6160.0, 6000.0)      # Non-planar (sign conflict against planar exp)

    exp_target = (835840.0, 435350.0, 278140.0) # Planar water exp

    filtered, diag = filter_unphysical_assignments(
        [cand1, cand2, cand3],
        exp_reference=exp_target,
        return_diagnostics=True
    )

    assert diag["total_candidates"] == 3
    assert diag["accepted_count"] == 2
    assert diag["rejected_count"] == 1
    assert diag["accepted_indices"] == [0, 1]
    assert diag["rejected_indices"] == [2]


def test_filter_unphysical_trajectories() -> None:
    """Verifies filtering unphysical trajectory steps (overlaps, non-planar distortions)."""
    water_masses = resolve_mass_array(["O", "H", "H"])

    # Frame 0: Valid planar water
    frame_0 = np.array([
        [0.000000,  0.000000, 0.000000],
        [0.957200,  0.000000, 0.000000],
        [-0.239987, 0.926627, 0.000000]
    ], dtype=np.float64)

    # Frame 1: Valid distorted planar water
    frame_1 = np.array([
        [0.000000,  0.000000, 0.000000],
        [0.965000,  0.000000, 0.000000],
        [-0.245000, 0.930000, 0.000000]
    ], dtype=np.float64)

    # Frame 2: Atom overlap (< 0.2 Angstrom)
    frame_2_overlap = np.array([
        [0.000000, 0.000000, 0.000000],
        [0.100000, 0.000000, 0.000000],
        [-0.23998, 0.926627, 0.000000]
    ], dtype=np.float64)

    batch = np.array([frame_0, frame_1, frame_2_overlap], dtype=np.float64)

    filtered, diag = filter_unphysical_trajectories(
        batch,
        masses=water_masses,
        exp_reference=(835840.0, 435350.0, 278140.0),
        return_diagnostics=True
    )

    assert diag["total_frames"] == 3
    assert diag["accepted_count"] == 2
    assert diag["rejected_count"] == 1
    assert diag["accepted_indices"] == [0, 1]
    assert diag["rejected_indices"] == [2]


# =============================================================================
# 6. JAX ACCELERATION TESTS
# =============================================================================

def test_jax_inertial_defect_and_planar_moments() -> None:
    """Verifies pure-JAX accelerated calculation matches NumPy analytical values."""
    A, B, C = 6039.2, 5804.9, 2959.2

    delta_jax = compute_inertial_defect_jax(A, B, C)
    paa_j, pbb_j, pcc_j = compute_planar_moments_jax(A, B, C)

    delta_np = compute_inertial_defect(rot_constants=(A, B, C))
    paa_np, pbb_np, pcc_np = compute_planar_moments(rot_constants=(A, B, C))

    assert abs(float(delta_jax) - delta_np) < 1e-6
    assert abs(float(paa_j) - paa_np) < 1e-6
    assert abs(float(pbb_j) - pbb_np) < 1e-6
    assert abs(float(pcc_j) - pcc_np) < 1e-6


def test_resolve_inertial_defect_dict_moments() -> None:
    """Verifies resolve_inertial_defect handles dictionary of principal moments correctly."""
    validator = InertialDefectValidator()
    # Planar moments: Ia=10.0, Ib=20.0, Ic=30.0 -> Delta = 30 - 10 - 20 = 0.0
    d_mom = {"Ia": 10.0, "Ib": 20.0, "Ic": 30.0}
    delta = validator.resolve_inertial_defect(d_mom)
    assert abs(delta - 0.0) < 1e-10

    # Non-planar moments
    d_mom_np = {"Ia": 15.0, "Ib": 25.0, "Ic": 32.0}
    delta_np = validator.resolve_inertial_defect(d_mom_np)
    assert abs(delta_np - (32.0 - 15.0 - 25.0)) < 1e-10


def test_compute_planar_moments_length_validation() -> None:
    """Verifies compute_planar_moments rejects invalid sequence lengths."""
    with pytest.raises(ValueError, match="rot_constants sequence must contain 3 values"):
        compute_planar_moments(rot_constants=(100.0, 50.0))

    with pytest.raises(ValueError, match="moments sequence must contain 3 values"):
        compute_planar_moments(moments=(10.0, 20.0, 30.0, 40.0))


def test_validator_filter_unphysical_assignments() -> None:
    """Verifies InertialDefectValidator.filter_unphysical_assignments method."""
    validator = InertialDefectValidator(min_pcc_floor=-0.1)
    cand_valid = (6039.2, 5804.9, 2959.2)
    cand_unphysical = (204000.0, 6160.0, 6000.0)
    exp_ref = (835840.0, 435350.0, 278140.0)

    filtered, diag = validator.filter_unphysical_assignments(
        [cand_valid, cand_unphysical],
        exp_reference=exp_ref,
        return_diagnostics=True
    )
    assert diag["accepted_count"] == 1
    assert diag["rejected_count"] == 1
    assert diag["accepted_indices"] == [0]


def test_custom_min_pcc_floor_propagation() -> None:
    """Verifies custom min_pcc_floor correctly propagates to structural gate."""
    # Pcc = -0.3 is below standard floor (-0.25), but above custom floor (-0.5)
    cand = {"Paa": 50.0, "Pbb": 40.0, "Pcc": -0.3}
    custom_floor_res = evaluate_structural_gate(
        calc_input=cand, min_pcc_floor=-0.5
    )
    assert custom_floor_res["planar_hierarchy_valid"] is True

    strict_floor_res = evaluate_structural_gate(
        calc_input=cand, min_pcc_floor=-0.2
    )
    assert strict_floor_res["planar_hierarchy_valid"] is False
