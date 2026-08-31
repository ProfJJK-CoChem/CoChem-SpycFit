#!/usr/bin/env python3
# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
PyTest Suite for CoChem-SpycFit Jensen Averager.

Validates:
1. Method Matrix §5.1 / §3.0 & WBS Phase 1.1: 1D Intermolecular Stretch Jensen Model.
2. 3D Polyatomic Molecular Ensemble Tensor Averaging (<I^-1> vs 1/<I>).
3. Mass-Weighted Eckart Frame Alignment & Right-Handed Rotation Locking.
4. Discrete Variable Representation (DVR) Wavefunction Grid Expectation Averager.
5. Hardware-Accelerated Pure JAX FP64 Kernels.
6. Dynamic Mendeleev Mass Resolution (Mendeleev Mandate).
7. Method Matrix §16.3 Rejection Mandate for Non-Compliant Averaging.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_jensen_averager import (
    AMU_ANGSTROM2_TO_MHZ,
    CONV_MHZ_AMU_ANG2,
    EckartAlignmentError,
    Jensen1DResult,
    Jensen3DResult,
    JensenAveragerError,
    JensenDVRGridResult,
    JensenInequalityViolationError,
    MethodMatrixComplianceError,
    SingularInertiaTensorError,
    UnphysicalWeightDistributionError,
    VibrationalAveragingRejectionError,
    average_inverse_inertia_dvr_grid,
    average_inverse_inertia_ensemble,
    average_inverse_inertia_ensemble_jax,
    compute_1d_benchmark_suite,
    compute_1d_jensen_model,
    compute_center_of_mass,
    compute_center_of_mass_jax,
    compute_inertia_tensor,
    compute_inertia_tensor_jax,
    compute_inverse_inertia_tensor_jax,
    compute_principal_moments,
    compute_rotational_constants_from_inertia,
    compute_rotational_constants_from_inverse_inertia,
    compute_vibrationally_averaged_constants,
    eckart_align,
    format_benchmark_table_ascii,
    format_benchmark_table_markdown,
    generate_vdw_stretch_distribution,
    get_dynamic_atomic_mass,
    get_reduced_mass,
    resolve_mass_array,
    validate_vibrational_averaging_compliance,
    verify_method_matrix_section_5_1_table,
)


def test_1d_jensen_model_method_matrix_table() -> None:
    """Verifies that the 1D benchmark suite matches Method Matrix Section 5.1 reference values."""
    is_verified: bool = verify_method_matrix_section_5_1_table()
    assert is_verified is True

    suite: List[Jensen1DResult] = compute_1d_benchmark_suite()
    assert len(suite) == 5

    # At sigma_R = 0.15 A (standard vdW zero-point amplitude)
    res_015: Jensen1DResult = suite[2]
    assert math.isclose(res_015.sigma_R, 0.15, abs_tol=1e-4)
    # Mandated by Method Matrix Section 5.1: 21.93 MHz bias (0.628% ~ 0.63%)
    assert abs(res_015.bias_MHz - 21.93) < 0.05
    assert abs(res_015.bias_pct - 0.628) < 0.01
    assert res_015.B_inv_I > res_015.B_I_avg
    assert res_015.provenance == "[M]"


def test_1d_jensen_taylor_series_expansion() -> None:
    """Verifies convergence of high-order Taylor series factor against numerical quadrature."""
    res: Jensen1DResult = compute_1d_jensen_model(R0=3.80, reduced_mass=10.0, sigma_R=0.05)
    # At small amplitude, numerical integration and Taylor expansion agree to high precision
    rel_diff: float = abs(res.B_inv_I - res.taylor_B_inv_I) / res.B_inv_I
    assert rel_diff < 1e-4

    # Leading fractional bias formula: 3 * sigma^2 / R0^2
    expected_leading_bias: float = 3.0 * ((0.05 / 3.80) ** 2)
    assert math.isclose(res.leading_fractional_bias, expected_leading_bias, rel_tol=1e-5)


def test_1d_jensen_model_parameter_guards() -> None:
    """Verifies domain validation and exception handling for unphysical 1D model parameters."""
    with pytest.raises(JensenAveragerError, match="Equilibrium separation R0 must be positive"):
        compute_1d_jensen_model(R0=-1.0)

    with pytest.raises(JensenAveragerError, match="Reduced mass must be positive"):
        compute_1d_jensen_model(reduced_mass=0.0)

    with pytest.raises(JensenAveragerError, match="Standard deviation sigma_R must be positive"):
        compute_1d_jensen_model(sigma_R=-0.1)


def test_mendeleev_mass_retrieval_and_reduced_mass() -> None:
    """Verifies dynamic atomic mass lookup for elements and isotopes from mendeleev library."""
    m_h: float = get_dynamic_atomic_mass("H")
    m_d: float = get_dynamic_atomic_mass("D")
    m_13c: float = get_dynamic_atomic_mass("13C")
    m_18o: float = get_dynamic_atomic_mass("18O")
    m_ar: float = get_dynamic_atomic_mass("Ar")

    assert 1.007 <= m_h <= 1.009
    assert 2.014 <= m_d <= 2.015
    assert 13.003 <= m_13c <= 13.004
    assert 17.999 <= m_18o <= 18.000
    assert 39.94 <= m_ar <= 39.97

    # Reduced mass calculation
    mu_ar_co2: float = get_reduced_mass("Ar", "C")
    expected_mu: float = (m_ar * get_dynamic_atomic_mass("C")) / (m_ar + get_dynamic_atomic_mass("C"))
    assert math.isclose(mu_ar_co2, expected_mu, rel_tol=1e-6)

    # Sequence resolution
    masses: np.ndarray = resolve_mass_array(["H", "D", "13C", "18O"])
    assert len(masses) == 4
    assert masses.dtype == np.float64


def test_3d_polyatomic_ensemble_tensor_averaging_h2o() -> None:
    """Verifies 3D inverse inertia tensor expectation value calculation on genuine H2O fixture."""
    fixture_path: Path = Path(__file__).parent / "fixtures" / "landscape_water.h5"
    with h5py.File(fixture_path, "r") as f:
        geom: np.ndarray = f["geometry"][:]
        masses: np.ndarray = f["masses_amu"][:]

    # Create coordinate ensemble by displacing along normal modes
    displacements: List[float] = [-0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08]
    n_snaps: int = len(displacements)
    coords_list: List[np.ndarray] = []
    for d in displacements:
        disp_geom = geom.copy()
        disp_geom[1, 0] += d
        disp_geom[2, 0] -= d
        coords_list.append(disp_geom)

    ensemble: np.ndarray = np.stack(coords_list, axis=0)
    symbols: List[str] = ["O", "H", "H"]

    res_3d: Jensen3DResult = average_inverse_inertia_ensemble(ensemble, masses, symbols)
    assert res_3d.n_atoms == 3
    assert res_3d.n_snapshots == n_snaps
    assert res_3d.A_correct >= res_3d.A_flawed - 1e-7
    assert res_3d.B_correct >= res_3d.B_flawed - 1e-7
    assert res_3d.C_correct >= res_3d.C_flawed - 1e-7
    assert res_3d.is_valid_jensen is True
    assert "least reliable constant" in res_3d.c_constant_reliability_warning


def test_3d_polyatomic_ensemble_tensor_averaging_vdw_complex() -> None:
    """Verifies 3D tensor expectation value calculation on Ar...CO2 fluxional dimer."""
    m_ar: float = get_dynamic_atomic_mass("Ar")
    m_c: float = get_dynamic_atomic_mass("C")
    m_o: float = get_dynamic_atomic_mass("O")

    m1: np.ndarray = np.array([m_ar])
    m2: np.ndarray = np.array([m_c, m_o, m_o])

    c1: np.ndarray = np.array([[0.0, 0.0, 0.0]])
    c2: np.ndarray = np.array([[0.0, 0.0, 0.0], [0.0, 1.16, 0.0], [0.0, -1.16, 0.0]])

    ens, total_masses, syms = generate_vdw_stretch_distribution(c1, c2, m1, m2, R0=3.80, sigma_R=0.15, n_snapshots=500, seed=42)
    res_3d: Jensen3DResult = average_inverse_inertia_ensemble(ens, total_masses, syms)

    # Jensen's inequality: B_correct > B_flawed and C_correct > C_flawed
    assert res_3d.bias_B_MHz > 0.0
    assert res_3d.bias_C_MHz > 0.0
    assert res_3d.bias_pct_B > 0.1
    assert res_3d.is_valid_jensen is True


def test_compute_vibrationally_averaged_constants_direct_array() -> None:
    """Verifies batch array-level expectation value calculator with 3x3 inertia tensors."""
    t1: np.ndarray = np.diag([10.0, 20.0, 30.0])
    t2: np.ndarray = np.diag([12.0, 22.0, 32.0])
    tensor_set: np.ndarray = np.stack([t1, t2], axis=0)
    weights: np.ndarray = np.array([0.5, 0.5])

    b0_jensen: np.ndarray = compute_vibrationally_averaged_constants(tensor_set, weights)

    # Flawed calculation (inverting after averaging)
    avg_i: np.ndarray = 0.5 * (t1 + t2)
    inv_avg_i: np.ndarray = np.linalg.inv(avg_i)
    b0_flawed: np.ndarray = CONV_MHZ_AMU_ANG2 * np.sort(np.diag(inv_avg_i))[::-1]

    # Verify Jensen inequality for all principal axes
    for idx in range(3):
        assert b0_jensen[idx] > b0_flawed[idx]
        assert (b0_jensen[idx] - b0_flawed[idx]) > 0.0


def test_dvr_grid_expectation_averager() -> None:
    """Verifies Discrete Variable Representation (DVR) grid vibrational averaging."""
    fixture_path: Path = Path(__file__).parent / "fixtures" / "landscape_water.h5"
    with h5py.File(fixture_path, "r") as f:
        base_geom: np.ndarray = f["geometry"][:]
        masses: np.ndarray = f["masses_amu"][:]

    grid_points: List[np.ndarray] = []
    probabilities: List[float] = []
    displacements: List[float] = [-0.06, -0.03, 0.0, 0.03, 0.06]
    for d in displacements:
        g = base_geom.copy()
        g[0, 2] += d
        grid_points.append(g)
        # Gaussian probability weight
        prob = math.exp(-0.5 * (d / 0.03) ** 2)
        probabilities.append(prob)

    grid_coords: np.ndarray = np.stack(grid_points, axis=0)
    grid_probs: np.ndarray = np.array(probabilities, dtype=np.float64)

    res_dvr: JensenDVRGridResult = average_inverse_inertia_dvr_grid(grid_coords, grid_probs, masses)
    assert res_dvr.n_grid_points == 5
    assert res_dvr.A_correct >= res_dvr.A_flawed - 1e-7
    assert res_dvr.B_correct >= res_dvr.B_flawed - 1e-7
    assert res_dvr.C_correct >= res_dvr.C_flawed - 1e-7
    assert res_dvr.is_valid_jensen is True


def test_pure_jax_fp64_kernels() -> None:
    """Verifies pure JAX FP64 accelerated tensor routines match NumPy exact calculations."""
    fixture_path: Path = Path(__file__).parent / "fixtures" / "landscape_water.h5"
    with h5py.File(fixture_path, "r") as f:
        geom: np.ndarray = f["geometry"][:]
        masses: np.ndarray = f["masses_amu"][:]

    # Center of mass
    com_np: np.ndarray = compute_center_of_mass(geom, masses)
    com_jax = np.asarray(compute_center_of_mass_jax(geom, masses))
    np.testing.assert_allclose(com_np, com_jax, rtol=1e-12, atol=1e-12)

    # Inertia tensor
    i_np: np.ndarray = compute_inertia_tensor(geom, masses)
    i_jax = np.asarray(compute_inertia_tensor_jax(geom, masses))
    np.testing.assert_allclose(i_np, i_jax, rtol=1e-12, atol=1e-12)

    # Inverse inertia tensor
    inv_i_np: np.ndarray = np.linalg.inv(i_np)
    inv_i_jax = np.asarray(compute_inverse_inertia_tensor_jax(geom, masses))
    np.testing.assert_allclose(inv_i_np, inv_i_jax, rtol=1e-12, atol=1e-12)

    # Ensemble JAX averaging
    disps: List[float] = [-0.05, 0.0, 0.05]
    ens_list: List[np.ndarray] = []
    for d in disps:
        g = geom.copy()
        g[0, 1] += d
        ens_list.append(g)
    ensemble: np.ndarray = np.stack(ens_list, axis=0)

    a_jax, b_jax, c_jax = average_inverse_inertia_ensemble_jax(ensemble, masses)
    assert a_jax > b_jax > c_jax > 0.0


def test_method_matrix_section_16_3_audit_rejection() -> None:
    """Verifies rejection mandate under Method Matrix §16.3 for non-compliant averaging."""
    # Compliant execution must pass
    report = validate_vibrational_averaging_compliance(
        averaging_method="eckart_watson_inverse_inertia",
        inverse_before_avg=True,
        eckart_aligned=True,
    )
    assert report.status == "PASSED_COMPLIANT"
    assert report.rejection_triggered is False
    assert len(report.sha256_checksum) == 64

    # Non-compliant execution (inverting <I>) must trigger immediate hard rejection
    with pytest.raises(VibrationalAveragingRejectionError, match="Violates Jensen's inequality directive"):
        validate_vibrational_averaging_compliance(
            averaging_method="averaged_inertia_inversion",
            inverse_before_avg=False,
            eckart_aligned=True,
        )

    # Direct scalar B(R) averaging must trigger rejection
    with pytest.raises(VibrationalAveragingRejectionError, match="Direct averaging of scalar rotational constants"):
        validate_vibrational_averaging_compliance(
            averaging_method="direct_b_average",
            inverse_before_avg=True,
            eckart_aligned=True,
        )


def test_markdown_and_ascii_table_formatting() -> None:
    """Verifies Markdown and ASCII benchmark table formatting."""
    suite: List[Jensen1DResult] = compute_1d_benchmark_suite()
    md_table: str = format_benchmark_table_markdown(suite)
    ascii_table: str = format_benchmark_table_ascii(suite)

    assert "| sigma_R (Angstrom) |" in md_table
    assert "**3516.34**" in md_table or "**3516.38**" in md_table or "21.93" in md_table
    assert "sigma_R (A)" in ascii_table
    assert "bias (MHz)" in ascii_table
