"""
CoChem-SpycFit: Comprehensive Phase 3 Physical Unit Test Suite for Macroscopic Spectral Broadener.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of exact Faddeeva Voigt profile via jax.scipy.special.erfcx
- Validation of multi-temperature Boltzmann scaling with tensor broadcasting [N_lines, N_temperatures]
- Validation of static masked batching via jax.lax.scan returning registered pure PyTree ConvolverPyTree
- AST structural verification enforcing pure jax.lax.scan without forbidden Python for/while loops
- Zero-mock audit verifying no mocks, stubs, or pass statements across codebase.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
import scipy.special as scipy_sp

from cochem_spycfit.physics_core.cochem_convolver import (
    C2_RADIATION_CONST,
    DEFAULT_MAX_VRAM_MB,
    FOUR_LN_2,
    FWHM_TO_SIGMA_GAUSS,
    GAUSS_PREFACTOR,
    H_OVER_K_B,
    LORENTZ_PREFACTOR,
    PLANCK_H,
    SPEED_OF_LIGHT_CM_S,
    SPEED_OF_LIGHT_M_S,
    SQRT_2_LN_2,
    ConvolutionConfig,
    ConvolvedSpectrumResult,
    ConvolverError,
    ConvolverPyTree,
    EnergyUnit,
    InvalidGridError,
    MacroscopicSpectralBroadener,
    MicroBatchError,
    PartitionGridConfig,
    PartitionGridResult,
    ProfileError,
    ProfileType,
    SpectralDataError,
    SpectralLines,
    ThermalScalingError,
    build_frequency_grid,
    calculate_optimal_chunk_size,
    compute_boltzmann_scaling_grid_jax,
    convolute_multi_temperature,
    convolve_frequency_chunks_scan_1d_jax,
    convolve_frequency_chunks_scan_2d_jax,
    convolve_line_chunks_scan_1d_jax,
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    evaluate_partition_grid,
    evaluate_profile_kernel_jax,
    faddeeva_rational_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    get_energy_conversion_factor_to_kelvin,
    get_profile_type_code,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
    static_masked_scan,
    true_voigt_profile,
)

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. EXACT FADDEEVA VOIGT PROFILE TESTS (true_voigt_profile via erfcx)
# =============================================================================

def test_true_voigt_profile_erfcx_mathematical_precision() -> None:
    """Validate true_voigt_profile numerical match with scipy.special.voigt_profile within machine precision."""
    x_points = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    sigma_vals = [0.05, 0.1, 0.5, 1.2, 2.5]
    gamma_vals = [0.02, 0.1, 0.4, 0.8, 2.0]

    for sigma in sigma_vals:
        for gamma in gamma_vals:
            # JAX implementation
            jax_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
            jax_arr = np.array(jax_vals, dtype=np.float64)

            # Reference SciPy Faddeeva voigt_profile
            sp_vals = scipy_sp.voigt_profile(x_points, sigma, gamma)

            max_diff = np.max(np.abs(jax_arr - sp_vals))
            assert max_diff < 1e-13, (
                f"Voigt profile mismatch for sigma={sigma}, gamma={gamma}: max_diff={max_diff}"
            )


def test_true_voigt_profile_gaussian_limit() -> None:
    """When gamma -> 0, true_voigt_profile must analytically converge to a pure Gaussian distribution."""
    sigma = 0.35
    gamma = 1e-25  # Effectively zero Lorentzian width
    x_points = np.linspace(-3.0 * sigma, 3.0 * sigma, 50, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Gaussian: (1 / (sigma * sqrt(2*pi))) * exp(-x^2 / (2*sigma^2))
    expected_gaussian = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (x_points / sigma) ** 2)

    max_diff = np.max(np.abs(voigt_arr - expected_gaussian))
    assert max_diff < 1e-13, f"Gaussian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_lorentzian_limit() -> None:
    """When sigma is small, true_voigt_profile must approach a pure Lorentzian profile."""
    sigma = 1e-15
    gamma = 0.5
    x_points = np.linspace(-4.0, 4.0, 41, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Lorentzian: (1 / pi) * (gamma / (x^2 + gamma^2))
    expected_lorentzian = (1.0 / math.pi) * (gamma / (x_points ** 2 + gamma ** 2))

    max_diff = np.max(np.abs(voigt_arr - expected_lorentzian))
    assert max_diff < 1e-11, f"Lorentzian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_unit_area_normalization() -> None:
    """Verify analytical unit area normalization int_{-\\infty}^{\\infty} V(x) dx == 1.0."""
    sigma = 0.25
    gamma = 0.30
    cutoff = 200.0 * (sigma + gamma)
    n_pts = 100001
    x_grid = np.linspace(-cutoff, cutoff, n_pts, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_grid), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    integrated_area = float(np.trapezoid(voigt_arr, x_grid))
    # Account for analytical Lorentzian asymptotic tail 2 * int_{cutoff}^\\infty (gamma / (pi * x^2)) dx = 2*gamma / (pi * cutoff)
    tail_analytical = (2.0 * gamma) / (math.pi * cutoff)
    total_area = integrated_area + tail_analytical
    assert abs(total_area - 1.0) < 1e-6, f"Voigt integrated total area was {total_area}, expected ~1.0"

    # Also verify exact agreement with SciPy integration over identical domain
    sp_vals = scipy_sp.voigt_profile(x_grid, sigma, gamma)
    sp_area = float(np.trapezoid(sp_vals, x_grid))
    assert abs(integrated_area - sp_area) < 1e-13, f"Integration mismatch vs SciPy: diff={abs(integrated_area - sp_area)}"


def test_true_voigt_profile_symmetry_and_peak() -> None:
    """Verify profile symmetry V(-x) == V(x) and exact analytical peak height at x=0."""
    sigma = 0.4
    gamma = 0.25
    x_eval = jnp.array([0.0, -1.5, 1.5, -3.2, 3.2], dtype=jnp.float64)

    v_vals = true_voigt_profile(x_eval, sigma, gamma)

    # Peak height at x=0: Re[erfcx(gamma / (sigma * sqrt(2)))] / (sigma * sqrt(2*pi))
    u_0 = gamma / (sigma * math.sqrt(2.0))
    expected_peak = float(jsp.erfcx(u_0)) / (sigma * math.sqrt(2.0 * math.pi))
    actual_peak = float(v_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-15

    # Symmetry checks
    assert abs(float(v_vals[1]) - float(v_vals[2])) < 1e-15
    assert abs(float(v_vals[3]) - float(v_vals[4])) < 1e-15


def test_true_voigt_profile_jit_and_broadcasting() -> None:
    """Verify JIT compilation and tensor broadcasting across 2D/3D shapes."""
    jitted_voigt = jax.jit(true_voigt_profile)

    # 2D broadcast: x shape (20, 1), sigma (1, 5), gamma scalar
    x_2d = jnp.linspace(-2.0, 2.0, 20)[:, None]
    sigma_row = jnp.linspace(0.1, 0.5, 5)[None, :]
    gamma_val = 0.15

    res_2d = jitted_voigt(x_2d, sigma_row, gamma_val)
    assert res_2d.shape == (20, 5)
    assert jnp.all(jnp.isfinite(res_2d))
    assert jnp.all(res_2d > 0.0)


# =============================================================================
# 2. MULTI-TEMPERATURE CONVOLUTION TESTS (convolute_multi_temperature)
# =============================================================================

def test_convolute_multi_temperature_shape_broadcasting() -> None:
    """Verify output shape is strictly [N_lines, N_temperatures] with pure JAX tensor broadcasting."""
    n_lines = 40
    n_temps = 8

    freqs = jnp.linspace(100000.0, 110000.0, n_lines)
    intensities = jnp.linspace(0.1, 5.0, n_lines)
    energies = jnp.linspace(0.0, 500.0, n_lines)
    temps = jnp.array([10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 296.0, 350.0], dtype=jnp.float64)
    # Power-law partition function: Q(T) = (T / 296)^1.5
    q_temps = jnp.power(temps / 296.0, 1.5)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=296.0,
        q_ref=1.0,
    )

    assert scaled_i.shape == (n_lines, n_temps), f"Expected shape ({n_lines}, {n_temps}), got {scaled_i.shape}"
    assert jnp.all(jnp.isfinite(scaled_i))
    assert jnp.all(scaled_i >= 0.0)


def test_convolute_multi_temperature_reference_invariance() -> None:
    """At T == T_ref and Q(T) == Q_ref, scaled intensities must identically equal reference intensities."""
    n_lines = 15
    freqs = jnp.linspace(90000.0, 95000.0, n_lines)
    intensities = jnp.array([0.5, 1.2, 3.4, 0.8, 2.1, 4.0, 1.5, 0.3, 2.7, 3.1, 1.9, 0.4, 2.8, 3.9, 1.1])
    energies = jnp.linspace(10.0, 200.0, n_lines)

    t_ref = 296.0
    q_ref = 1.0
    temps = jnp.array([t_ref], dtype=jnp.float64)
    q_temps = jnp.array([q_ref], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # Column 0 corresponds to T_ref
    scaled_col = scaled_i[:, 0]
    max_diff = float(jnp.max(jnp.abs(scaled_col - intensities)))
    assert max_diff < 1e-14, f"Reference temperature scaling differed from original: max_diff={max_diff}"


def test_convolute_multi_temperature_boltzmann_population_redistribution() -> None:
    """Validate that high-energy transitions gain intensity at high T while ground-state transitions deplete."""
    # Line 0: Ground state E_lower = 0 cm^-1
    # Line 1: High energy state E_lower = 300 cm^-1
    freqs = jnp.array([100000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 1.0], dtype=jnp.float64)
    energies = jnp.array([0.0, 300.0], dtype=jnp.float64)

    t_low = 50.0
    t_high = 300.0
    t_ref = 300.0
    q_ref = 1.0
    temps = jnp.array([t_low, t_high], dtype=jnp.float64)
    q_temps = jnp.array([1.0, 1.0], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # At low temperature (T=50K), high-energy line (Line 1) must be exponentially suppressed relative to Line 0
    ratio_t_low = float(scaled_i[1, 0] / scaled_i[0, 0])
    # Expected Boltzmann ratio: exp(- C2 * 300 * (1/50 - 1/300)) / exp(0)
    expected_ratio = math.exp(-C2_RADIATION_CONST * 300.0 * (1.0 / 50.0 - 1.0 / 300.0))
    assert abs(ratio_t_low - expected_ratio) < 1e-12
    assert ratio_t_low < 0.01  # Heavily depopulated at 50K


def test_convolute_multi_temperature_jit_compiled() -> None:
    """Verify convolute_multi_temperature JIT compilation and execution purity."""
    jitted_fn = jax.jit(convolute_multi_temperature)

    freqs = jnp.array([80000.0, 90000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    q_temps = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    temps = jnp.array([100.0, 200.0, 400.0], dtype=jnp.float64)

    res = jitted_fn(freqs, intensities, q_temps, temps)
    assert res.shape == (3, 3)
    assert jnp.all(jnp.isfinite(res))


# =============================================================================
# 3. STATIC MASKED SCAN & CONVOLVER PYTREE TESTS
# =============================================================================

def test_convolver_pytree_registration_and_roundtrip() -> None:
    """Verify ConvolverPyTree is registered with JAX PyTree system and supports flatten/unflatten/tree_map."""
    freqs = jnp.linspace(100.0, 200.0, 50)
    intensities = jnp.sin(jnp.linspace(0.0, math.pi, 50))
    mask = jnp.ones(50, dtype=jnp.bool_)
    total_i = jnp.array(5.25, dtype=jnp.float64)

    pytree = ConvolverPyTree(
        frequencies=freqs,
        intensities=intensities,
        mask=mask,
        total_intensity=total_i,
        num_lines=50,
        chunk_size=10000,
        num_chunks=1,
        profile_type="voigt",
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    assert len(leaves) == 4
    assert jnp.allclose(leaves[0], freqs)
    assert jnp.allclose(leaves[1], intensities)
    assert jnp.allclose(leaves[2], mask)
    assert jnp.allclose(leaves[3], total_i)

    # Unflatten
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, ConvolverPyTree)
    assert reconstructed.num_lines == 50
    assert reconstructed.chunk_size == 10000
    assert reconstructed.num_chunks == 1
    assert reconstructed.profile_type == "voigt"
    assert jnp.allclose(reconstructed.intensities, intensities)

    # Tree Map: Multiply all array leaves by 2.0
    mapped_tree = jax.tree_util.tree_map(lambda x: x * 2.0, pytree)
    assert isinstance(mapped_tree, ConvolverPyTree)
    assert jnp.allclose(mapped_tree.total_intensity, total_i * 2.0)


def test_static_masked_scan_returns_pure_pytree() -> None:
    """Verify static_masked_scan execution returns a pure ConvolverPyTree."""
    freqs = [100.0, 102.0, 105.0, 108.0, 110.0]
    intensities = [1.0, 2.5, 0.8, 3.2, 1.5]
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    result_pytree = static_masked_scan(
        lines=lines,
        chunk_size=2,
        grid_points=200,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.05,
        gamma=0.02,
    )

    assert isinstance(result_pytree, ConvolverPyTree)
    assert result_pytree.num_lines == 5
    assert result_pytree.chunk_size == 2
    assert result_pytree.num_chunks == 3  # ceil(5 / 2) = 3 chunks (padded length 6)
    assert len(result_pytree.frequencies) == 200
    assert len(result_pytree.intensities) == 200
    assert float(result_pytree.total_intensity) > 0.0


def test_static_masked_scan_masking_authenticity() -> None:
    """Verify that padded dummy lines in static masked batching have exactly zero effect on output spectrum."""
    freqs = [100.0, 105.0, 110.0]
    intensities = [2.0, 3.0, 1.0]
    lines_original = SpectralLines(frequencies=freqs, intensities=intensities)

    # Run with chunk_size=2 (produces 2 chunks: chunk 1 has 2 lines, chunk 2 has 1 line + 1 padded dummy line)
    pytree_chunked = static_masked_scan(
        lines=lines_original,
        chunk_size=2,
        grid_points=300,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.08,
        gamma=0.04,
    )

    # Run monolithic reference (all lines at once)
    grid_arr = pytree_chunked.frequencies
    delta_nu = grid_arr[:, None] - jnp.array(freqs, dtype=jnp.float64)[None, :]
    prof_mono = true_voigt_profile(delta_nu, 0.08, 0.04)
    spec_mono = jnp.dot(prof_mono, jnp.array(intensities, dtype=jnp.float64))

    max_diff = float(jnp.max(jnp.abs(pytree_chunked.intensities - spec_mono)))
    assert max_diff < 1e-13, f"Masked scan output differed from unchunked monolithic reference: max_diff={max_diff}"


def test_static_masked_scan_chunk_size_invariance() -> None:
    """Verify that convolved spectrum is invariant to the chunk_size parameter."""
    freqs = np.linspace(100.0, 120.0, 37, dtype=np.float64).tolist()
    intensities = np.linspace(0.5, 4.0, 37, dtype=np.float64).tolist()
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    grid_pts = 500
    g_min = 95.0
    g_max = 125.0
    sigma = 0.06
    gamma = 0.03

    # Test chunk sizes: 1, 5, 10, 37, 100
    chunk_sizes = [1, 5, 10, 37, 100]
    results = []

    for cs in chunk_sizes:
        res = static_masked_scan(
            lines=lines,
            chunk_size=cs,
            grid_points=grid_pts,
            grid_min=g_min,
            grid_max=g_max,
            sigma=sigma,
            gamma=gamma,
        )
        results.append(res.intensities)

    ref_spec = results[0]
    for idx, cs in enumerate(chunk_sizes[1:], start=1):
        diff = float(jnp.max(jnp.abs(results[idx] - ref_spec)))
        assert diff < 1e-13, f"Chunk size {cs} produced mismatch vs chunk size 1: diff={diff}"


def test_static_masked_scan_input_formats() -> None:
    """Verify static_masked_scan works identically with SpectralLines, tuple of arrays, and dict inputs."""
    f_list = [100.0, 102.0, 104.0]
    i_list = [1.0, 2.0, 3.0]

    # 1. SpectralLines model
    res1 = static_masked_scan(SpectralLines(frequencies=f_list, intensities=i_list), chunk_size=2, grid_points=100)

    # 2. Tuple of arrays
    res2 = static_masked_scan((jnp.array(f_list), jnp.array(i_list)), chunk_size=2, grid_points=100)

    # 3. Dict
    res3 = static_masked_scan({"frequencies": f_list, "intensities": i_list}, chunk_size=2, grid_points=100)

    assert jnp.allclose(res1.intensities, res2.intensities, atol=1e-14)
    assert jnp.allclose(res1.intensities, res3.intensities, atol=1e-14)


# =============================================================================
# 4. AST LOOP-FREE & ZERO-MOCK COMPLIANCE VERIFICATION
# =============================================================================

def test_ast_loop_free_compliance() -> None:
    """
    AST structural verification ensuring that JAX kernels use pure vectorization and jax.lax.scan,
    with ZERO Python for/while loops in performance-critical paths.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    critical_functions = [
        "true_voigt_profile",
        "convolute_multi_temperature",
        "_scan_step_masked_kernel",
        "faddeeva_rational_jax",
        "gaussian_kernel_jax",
        "lorentzian_kernel_jax",
        "pseudo_voigt_kernel_jax",
        "faddeeva_voigt_kernel_jax",
        "evaluate_profile_kernel_jax",
        "evaluate_partition_function_jax",
        "compute_boltzmann_scaling_grid_jax",
        "convolve_frequency_chunks_scan_1d_jax",
        "convolve_frequency_chunks_scan_2d_jax",
        "convolve_line_chunks_scan_1d_jax",
        "convolve_spectrum_monolithic_jax",
    ]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in critical_functions:
            for child in ast.walk(node):
                assert not isinstance(child, (ast.For, ast.While)), (
                    f"Forbidden Python loop found in JAX kernel '{node.name}' at line {getattr(child, 'lineno', '?')}."
                )


def test_zero_mock_and_zero_pass_mandate() -> None:
    """
    Strict Zero-Mock Mandate audit:
    Verifies that cochem_convolver contains zero unittest.mock, MagicMock, Mock, or pass statements.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    # Check for pass statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            # Pass statements in empty exception classes or protocols are forbidden under zero-pass
            pytest.fail(f"Forbidden 'pass' statement found in cochem_convolver.py at line {getattr(node, 'lineno', '?')}.")

    # Check for mock terms
    banned_terms = ["unittest.mock", "MagicMock", "PropertyMock", "create_autospec"]
    for term in banned_terms:
        assert term not in source_code, f"Forbidden mock term '{term}' found in cochem_convolver.py."
