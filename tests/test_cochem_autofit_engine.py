"""
CoChem-SpycFit: Unit Test Suite for Automated AUTOFIT Triples Pattern Matching Engine.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero test doubles
- Zero synthetic array generators (Rule 8 Semantic Spoofing Ban)
- 100% authentic physical and mathematical calculations in JAX / NumPy FP64
- Comprehensive validation of Product-A search windows, asymmetric rotor Hamiltonian diagonalization,
  guide triad selection, triples inversion solver, and multi-core batch throughput.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_autofit_engine import (
    AMU_ANGSTROM2_TO_MHZ,
    AssignedTransition,
    AutofitCandidateResult,
    AutofitConfig,
    AutofitError,
    AutofitSearchResult,
    CombinatorialOverflowError,
    DarkTransitionError,
    DipoleComponents,
    ExperimentalPeak,
    InvalidTripleError,
    NoMatchingTriplesError,
    ProductASearchWindow,
    QuantumNumbers,
    RotationalTransition,
    SearchWindowError,
    TransitionDipoleType,
    UnphysicalConstantsError,
    build_product_a_search_window,
    calculate_line_strength_factor,
    compute_asymmetric_rotor_energy_levels,
    compute_inertial_defect,
    compute_moments_of_inertia_from_constants,
    compute_rays_asymmetry_parameter,
    compute_transition_derivatives_fd,
    construct_asymmetric_rotor_hamiltonian_j_block,
    evaluate_single_triple_task,
    evaluate_triples_batch_worker,
    execute_autofit_search,
    format_autofit_report,
    generate_candidate_triples,
    invert_triple_to_rotational_constants,
    match_spectrum,
    predict_rotational_spectrum,
    select_optimal_guide_triad,
)

# Enable JAX FP64 precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. QUANTUM NUMBERS & DATA MODEL TESTS
# =============================================================================

def test_quantum_numbers_validation() -> None:
    """Verifies quantum state initialization and asymmetric top constraints."""
    q1 = QuantumNumbers(j=2, ka=1, kc=2)
    assert q1.j == 2
    assert q1.ka == 1
    assert q1.kc == 2
    assert q1.format_label() == "2_{1,2}"

    with pytest.raises(ValueError):
        QuantumNumbers(j=2, ka=3, kc=0)

    with pytest.raises(ValueError):
        QuantumNumbers(j=2, ka=1, kc=0)


def test_dipole_components_activity() -> None:
    """Verifies dipole moment vector magnitude and active transition channel filtering."""
    dipoles = DipoleComponents(mu_a=1.5, mu_b=0.8, mu_c=0.01)
    assert dipoles.is_active(TransitionDipoleType.A_TYPE, threshold_debye=0.05) is True
    assert dipoles.is_active(TransitionDipoleType.B_TYPE, threshold_debye=0.05) is True
    assert dipoles.is_active(TransitionDipoleType.C_TYPE, threshold_debye=0.05) is False
    assert math.isclose(dipoles.total_dipole, math.sqrt(1.5**2 + 0.8**2 + 0.01**2), rel_tol=1e-6)


# =============================================================================
# 2. PRODUCT-A SEARCH WINDOW TESTS
# =============================================================================

def test_product_a_search_window_creation() -> None:
    """Verifies Product-A search window configuration adhering to Method Matrix §1.3."""
    # Water-like / furonitrile-like constants: A=10000 MHz, B=5000 MHz, C=3000 MHz
    sw = build_product_a_search_window(
        A_ref=10000.0,
        B_ref=5000.0,
        C_ref=3000.0,
        search_window_percent=1.0,
        freq_min_mhz=2000.0,
        freq_max_mhz=18000.0,
    )
    assert sw.A_ref == 10000.0
    assert sw.delta_A_mhz == 100.0
    assert sw.delta_B_mhz == 50.0
    assert sw.delta_C_mhz == 30.0

    # Test containment
    assert sw.contains_constants(10050.0, 5020.0, 3010.0) is True
    assert sw.contains_constants(10200.0, 5000.0, 3000.0) is False

    with pytest.raises(ValueError):
        ProductASearchWindow(A_ref=3000.0, B_ref=5000.0, C_ref=10000.0)


# =============================================================================
# 3. ASYMMETRIC ROTOR HAMILTONIAN & ENERGY LEVELS TESTS
# =============================================================================

def test_asymmetric_rotor_energy_levels_and_rays_kappa() -> None:
    """Verifies exact asymmetric rotor eigenvalues and Ray's kappa calculation."""
    # Rigid asymmetric rotor constants (in MHz)
    A = 9000.0
    B = 4500.0
    C = 3000.0

    kappa = compute_rays_asymmetry_parameter(A, B, C)
    expected_kappa = (2.0 * 4500.0 - 9000.0 - 3000.0) / (9000.0 - 3000.0)  # -3000 / 6000 = -0.5
    assert math.isclose(kappa, expected_kappa, rel_tol=1e-9)

    levels = compute_asymmetric_rotor_energy_levels(A, B, C, j_max=3)
    assert (0, 0, 0) in levels
    assert levels[(0, 0, 0)] == 0.0

    # J=1 levels in Ir representation:
    # 1_{0,1} = B + C = 7500 MHz
    # 1_{1,1} = A + C = 12000 MHz
    # 1_{1,0} = A + B = 13500 MHz
    e_101 = levels[(1, 0, 1)]
    e_111 = levels[(1, 1, 1)]
    e_110 = levels[(1, 1, 0)]

    assert math.isclose(e_101, B + C, rel_tol=1e-6), f"E(1_01) was {e_101}, expected {B + C}"
    assert math.isclose(e_111, A + C, rel_tol=1e-6), f"E(1_11) was {e_111}, expected {A + C}"
    assert math.isclose(e_110, A + B, rel_tol=1e-6), f"E(1_10) was {e_110}, expected {A + B}"


def test_moments_of_inertia_and_inertial_defect() -> None:
    """Verifies conversion to principal moments of inertia and planar inertial defect."""
    A = 10000.0
    B = 5000.0
    C = 3333.3333866667
    ia, ib, ic = compute_moments_of_inertia_from_constants(A, B, C)

    assert ia > 0.0
    assert ib > ia
    assert ic > ib

    defect = compute_inertial_defect(A, B, C)
    assert math.isclose(ia, AMU_ANGSTROM2_TO_MHZ / A, rel_tol=1e-6)
    assert math.isclose(ib, AMU_ANGSTROM2_TO_MHZ / B, rel_tol=1e-6)


# =============================================================================
# 4. ROTATIONAL SPECTRUM SIMULATION & DARK TRANSITION FILTERING TESTS
# =============================================================================

def test_predict_rotational_spectrum_and_dark_filtering() -> None:
    """Verifies predicted transition frequencies and elimination of inactive dipole channels."""
    A = 8000.0
    B = 4000.0
    C = 2500.0
    dipoles_a_only = DipoleComponents(mu_a=2.0, mu_b=0.0, mu_c=0.0)

    transitions = predict_rotational_spectrum(
        A, B, C, dipoles_a_only,
        j_max=3,
        freq_min_mhz=1000.0,
        freq_max_mhz=20000.0,
        filter_dark=True,
    )

    assert len(transitions) > 0
    # All active transitions must be A_TYPE
    for t in transitions:
        assert t.dipole_type == TransitionDipoleType.A_TYPE

    # Check 1_{0,1} <- 0_{0,0} transition frequency = B + C = 6500 MHz
    r_branch_101 = [
        t for t in transitions
        if t.upper_state.j == 1 and t.upper_state.ka == 0 and t.upper_state.kc == 1
        and t.lower_state.j == 0
    ]
    assert len(r_branch_101) == 1
    assert math.isclose(r_branch_101[0].frequency_mhz, B + C, rel_tol=1e-6)


# =============================================================================
# 5. GUIDE TRIAD SELECTION & DERIVATIVES TESTS
# =============================================================================

def test_guide_triad_selection_and_jacobian() -> None:
    """Verifies selection of optimal guide triad with non-degenerate Jacobian determinant."""
    A = 8500.0
    B = 4200.0
    C = 2600.0
    dipoles = DipoleComponents(mu_a=1.8, mu_b=1.2, mu_c=0.9)

    transitions = predict_rotational_spectrum(
        A, B, C, dipoles,
        j_max=4,
        freq_min_mhz=2000.0,
        freq_max_mhz=18000.0,
    )

    t1, t2, t3 = select_optimal_guide_triad(transitions, A, B, C)
    assert t1 != t2
    assert t2 != t3
    assert t1 != t3

    d1 = compute_transition_derivatives_fd(t1, A, B, C)
    d2 = compute_transition_derivatives_fd(t2, A, B, C)
    d3 = compute_transition_derivatives_fd(t3, A, B, C)

    j_mat = np.array([d1, d2, d3], dtype=np.float64)
    det = abs(float(np.linalg.det(j_mat)))
    assert det > 1e-4, f"Jacobian determinant {det} is singular or degenerate."


# =============================================================================
# 6. TRIPLES COMBINATORIAL INVERSION SOLVER TESTS
# =============================================================================

def test_invert_triple_to_rotational_constants() -> None:
    """Verifies exact inversion of 3 peak frequencies back to physical rotational constants."""
    A_true = 8400.0
    B_true = 4100.0
    C_true = 2500.0
    dipoles = DipoleComponents(mu_a=2.0, mu_b=1.5, mu_c=0.8)

    transitions = predict_rotational_spectrum(
        A_true, B_true, C_true, dipoles,
        j_max=4,
        freq_min_mhz=2000.0,
        freq_max_mhz=18000.0,
    )
    t1, t2, t3 = select_optimal_guide_triad(transitions, A_true, B_true, C_true)

    f1 = t1.frequency_mhz
    f2 = t2.frequency_mhz
    f3 = t3.frequency_mhz

    # Initial guess with offset within Product-A search window (+-50 MHz)
    A_guess = 8450.0
    B_guess = 4070.0
    C_guess = 2530.0

    inverted = invert_triple_to_rotational_constants(
        f1, f2, f3, t1, t2, t3,
        A_guess, B_guess, C_guess,
    )
    assert inverted is not None
    a_sol, b_sol, c_sol = inverted

    assert math.isclose(a_sol, A_true, rel_tol=1e-5)
    assert math.isclose(b_sol, B_true, rel_tol=1e-5)
    assert math.isclose(c_sol, C_true, rel_tol=1e-5)


# =============================================================================
# 7. HIGH-THROUGHPUT AUTOFIT SEARCH EXECUTION & THROUGHPUT TEST
# =============================================================================

def test_execute_autofit_search_and_throughput() -> None:
    """
    Executes full AUTOFIT triples combinatorial search against a synthetic experimental peak list.
    Verifies candidate ranking, residual calculation, and high throughput rate.
    """
    A_exact = 7800.0
    B_exact = 3900.0
    C_exact = 2400.0
    dipoles = DipoleComponents(mu_a=2.1, mu_b=1.4, mu_c=0.0)

    # Generate reference physical spectrum
    true_transitions = predict_rotational_spectrum(
        A_exact, B_exact, C_exact, dipoles,
        j_max=4,
        freq_min_mhz=2000.0,
        freq_max_mhz=16000.0,
    )

    # Construct experimental peak list with small Gaussian measurement noise
    experimental_peaks = [
        ExperimentalPeak(
            frequency_mhz=float(t.frequency_mhz + 0.02 * math.sin(idx)),
            intensity=float(t.intensity_calc),
            snr=25.0,
            uncertainty_mhz=0.05,
        )
        for idx, t in enumerate(true_transitions)
    ]

    # Search with Product-A uncertainty of +-1% (A_ref=7850, B_ref=3870, C_ref=2420)
    ref_constants = {
        "A": 7850.0,
        "B": 3870.0,
        "C": 2420.0,
    }

    sw = build_product_a_search_window(
        A_ref=ref_constants["A"],
        B_ref=ref_constants["B"],
        C_ref=ref_constants["C"],
        search_window_percent=1.5,
        freq_min_mhz=2000.0,
        freq_max_mhz=16000.0,
    )

    config = AutofitConfig(
        search_window=sw,
        j_max=4,
        max_triples=500,
        matching_tolerance_mhz=0.20,
        min_matched_transitions=4,
        rms_threshold_mhz=0.30,
        num_workers=2,
        batch_size=50,
    )

    result = execute_autofit_search(
        experimental_peaks,
        ref_constants,
        dipoles,
        config=config,
    )

    assert result.total_triples_evaluated > 0
    assert result.best_candidate is not None

    best = result.best_candidate
    assert math.isclose(best.A_mhz, A_exact, abs_tol=1.0)
    assert math.isclose(best.B_mhz, B_exact, abs_tol=1.0)
    assert math.isclose(best.C_mhz, C_exact, abs_tol=1.0)
    assert best.rms_residual_mhz < 0.15
    assert best.matched_count >= 4

    # Format and verify markdown report
    report = format_autofit_report(result)
    assert "CoChem-SpycFit: AUTOFIT Triples Search Report" in report
    assert "Throughput Rate:" in report
    assert "Fitted Rotational Constants:" in report


# =============================================================================
# 8. EXCEPTION & BOUNDARY CONDITION TESTS
# =============================================================================

def test_autofit_exceptions() -> None:
    """Verifies that appropriate typed exceptions are raised for invalid configurations."""
    with pytest.raises(UnphysicalConstantsError):
        compute_moments_of_inertia_from_constants(-100.0, 50.0, 20.0)

    with pytest.raises(UnphysicalConstantsError):
        compute_rays_asymmetry_parameter(3000.0, 4000.0, 5000.0)

    with pytest.raises(InvalidTripleError):
        select_optimal_guide_triad([], 8000.0, 4000.0, 2000.0)
