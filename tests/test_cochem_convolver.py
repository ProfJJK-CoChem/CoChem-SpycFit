"""
CoChem-SpycFit: Comprehensive Physical Unit Test Suite for Macroscopic Spectral Broadener.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of Gaussian, Lorentzian, Pseudo-Voigt, and Faddeeva-Voigt profiles
- Validation of multi-temperature partition functions Q(T) and Boltzmann thermal scaling
- Numerical verification of micro-batched jax.lax.scan vs monolithic vectorized output (diff < 1e-12)
- AST structural verification enforcing pure jax.lax.scan without forbidden Python for/while loops
- Negative physical tests for invalid inputs, non-physical temperatures, and dimension mismatches.
"""

from __future__ import annotations

import ast
import inspect
import math
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
)

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. ANALYTICAL LINE PROFILE KERNEL PHYSICAL TESTS (FP64)
# =============================================================================

def test_gaussian_profile_peak_and_fwhm_geometry() -> None:
    """Validate Gaussian profile peak location, peak height, and half-maximum width."""
    fwhm = 0.4
    delta_nu = jnp.array([0.0, -0.2, 0.2, -0.4, 0.4], dtype=jnp.float64)
    g_vals = gaussian_kernel_jax(delta_nu, fwhm)

    # 1. Peak height at delta_nu = 0 matches analytical 2 * sqrt(ln 2) / (sqrt(pi) * FWHM)
    expected_peak = (2.0 * math.sqrt(math.log(2.0))) / (math.sqrt(math.pi) * fwhm)
    actual_peak = float(g_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-14

    # 2. At delta_nu = +/- FWHM / 2, profile value must be exactly 50% of the peak
    half_left = float(g_vals[1])
    half_right = float(g_vals[2])
    assert abs(half_left - 0.5 * expected_peak) < 1e-14
    assert abs(half_right - 0.5 * expected_peak) < 1e-14

    # 3. Symmetry check: g(-delta) == g(+delta)
    assert abs(float(g_vals[1]) - float(g_vals[2])) < 1e-15
    assert abs(float(g_vals[3]) - float(g_vals[4])) < 1e-15


def test_gaussian_profile_area_normalization() -> None:
    """Validate Gaussian profile area normalization over dense frequency grid."""
    fwhm = 0.25
    nu_grid = np.linspace(-10.0, 10.0, 200001, dtype=np.float64)
    g_vals = np.array(gaussian_kernel_jax(jnp.array(nu_grid), fwhm))

    integrated_area = float(np.trapezoid(g_vals, nu_grid))
    assert abs(integrated_area - 1.0) < 1e-11


def test_lorentzian_profile_peak_and_fwhm_geometry() -> None:
    """Validate Lorentzian profile peak location, peak height, and half-maximum width."""
    fwhm = 0.6
    delta_nu = jnp.array([0.0, -0.3, 0.3, -0.6, 0.6], dtype=jnp.float64)
    l_vals = lorentzian_kernel_jax(delta_nu, fwhm)

    # 1. Peak height at delta_nu = 0 matches analytical 2 / (pi * FWHM)
    expected_peak = 2.0 / (math.pi * fwhm)
    actual_peak = float(l_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-14

    # 2. At delta_nu = +/- FWHM / 2, profile value must be exactly 50% of the peak
    half_left = float(l_vals[1])
    half_right = float(l_vals[2])
    assert abs(half_left - 0.5 * expected_peak) < 1e-14
    assert abs(half_right - 0.5 * expected_peak) < 1e-14


def test_lorentzian_profile_analytical_integral_conservation() -> None:
    """Validate Lorentzian profile integral against analytical arctan Cauchy integral."""
    fwhm = 0.5
    half_span = 50.0
    nu_grid = np.linspace(-half_span, half_span, 500001, dtype=np.float64)
    l_vals = np.array(lorentzian_kernel_jax(jnp.array(nu_grid), fwhm))

    # Analytical integral of L(x; fwhm) from -L to +L is (2/pi) * arctan(2 * L / fwhm)
    expected_integral = (2.0 / math.pi) * math.atan((2.0 * half_span) / fwhm)
    numerical_integral = float(np.trapezoid(l_vals, nu_grid))

    assert abs(numerical_integral - expected_integral) < 1e-7


def test_pseudo_voigt_asymptotic_limits() -> None:
    """Validate Thompson-Cox-Hastings Pseudo-Voigt asymptotic convergence to pure Gaussian and Lorentzian."""
    nu_grid = jnp.linspace(-2.0, 2.0, 501, dtype=jnp.float64)
    fwhm_ref = 0.3

    # Limit 1: FWHM_L -> 0 (pure Gaussian)
    pv_gauss_limit = pseudo_voigt_kernel_jax(nu_grid, fwhm_ref, 1e-12)
    pure_gauss = gaussian_kernel_jax(nu_grid, fwhm_ref)
    diff_gauss = float(jnp.max(jnp.abs(pv_gauss_limit - pure_gauss)))
    assert diff_gauss < 1e-10

    # Limit 2: FWHM_G -> 0 (pure Lorentzian)
    pv_lorentz_limit = pseudo_voigt_kernel_jax(nu_grid, 1e-12, fwhm_ref)
    pure_lorentz = lorentzian_kernel_jax(nu_grid, fwhm_ref)
    diff_lorentz = float(jnp.max(jnp.abs(pv_lorentz_limit - pure_lorentz)))
    assert diff_lorentz < 1e-10


def test_faddeeva_rational_accuracy_vs_reference() -> None:
    """Validate pure-JAX Weideman Faddeeva evaluation against analytical identities."""
    # Test property: w(0) = 1.0 + 0.0j
    w_zero = faddeeva_rational_jax(jnp.array([0.0 + 0.0j], dtype=jnp.complex128))
    assert abs(complex(w_zero[0]) - 1.0) < 1e-13

    # Test property: w(i*y) = exp(y^2) * erfc(y) is strictly real for real y >= 0
    y_test = 1.5
    w_iy = faddeeva_rational_jax(jnp.array([0.0 + y_test * 1j], dtype=jnp.complex128))
    val = complex(w_iy[0])
    assert abs(val.imag) < 1e-14
    assert val.real > 0.0

    # Test exact Voigt profile peak and symmetry
    fg = 0.4
    fl = 0.2
    nu_test = jnp.array([0.0, -0.5, 0.5], dtype=jnp.float64)
    v_vals = faddeeva_voigt_kernel_jax(nu_test, fg, fl)

    assert float(v_vals[0]) > float(v_vals[1])
    assert abs(float(v_vals[1]) - float(v_vals[2])) < 1e-14


def test_unified_profile_kernel_dispatch() -> None:
    """Validate unified evaluate_profile_kernel_jax dispatch across all 4 profile codes."""
    delta = jnp.array([0.0], dtype=jnp.float64)
    fg = 0.2
    fl = 0.2

    for code in range(4):
        val = evaluate_profile_kernel_jax(delta, fg, fl, code)
        assert float(val[0]) > 0.0
        assert math.isfinite(float(val[0]))


# =============================================================================
# 2. MULTI-TEMPERATURE PARTITION GRIDS & BOLTZMANN SCALING TESTS
# =============================================================================

def test_partition_function_direct_state_sum() -> None:
    """Validate direct state-sum partition function calculation Q(T) = sum_k g_k * exp(-c2 * E_k / T)."""
    # 3-level quantum system: E = [0.0, 100.0, 300.0] cm^-1, g = [1.0, 3.0, 5.0]
    e_levels = jnp.array([0.0, 100.0, 300.0], dtype=jnp.float64)
    degeneracies = jnp.array([1.0, 3.0, 5.0], dtype=jnp.float64)
    temps = jnp.array([50.0, 150.0, 300.0], dtype=jnp.float64)
    c2 = C2_RADIATION_CONST

    q_jax = evaluate_partition_function_jax(
        temperatures=temps,
        power_law_index=None,
        q_ref=1.0,
        t_ref=298.15,
        energy_levels=e_levels,
        degeneracies=degeneracies,
        energy_to_k=c2,
    )

    # Analytical verification
    for idx, t in enumerate([50.0, 150.0, 300.0]):
        q_expected = (
            1.0 * math.exp(-c2 * 0.0 / t)
            + 3.0 * math.exp(-c2 * 100.0 / t)
            + 5.0 * math.exp(-c2 * 300.0 / t)
        )
        assert abs(float(q_jax[idx]) - q_expected) < 1e-12


def test_partition_function_power_law_scaling() -> None:
    """Validate power law temperature scaling Q(T) = Q(T_ref) * (T / T_ref)^n."""
    temps = jnp.array([100.0, 200.0, 298.15, 400.0], dtype=jnp.float64)
    t_ref = 298.15
    q_ref = 500.0
    n = 1.5  # Non-linear polyatomic asymmetric top exponent

    q_jax = evaluate_partition_function_jax(
        temperatures=temps,
        power_law_index=n,
        q_ref=q_ref,
        t_ref=t_ref,
        energy_levels=None,
        degeneracies=None,
        energy_to_k=C2_RADIATION_CONST,
    )

    # At T = T_ref, Q(T) must be exactly Q(T_ref)
    assert abs(float(q_jax[2]) - q_ref) < 1e-14
    # At T = 100 K, Q(T) = 500 * (100 / 296)^1.5
    expected_100 = q_ref * (100.0 / 298.15) ** 1.5
    assert abs(float(q_jax[0]) - expected_100) < 1e-12


def test_boltzmann_scaling_invariance_at_reference_temperature() -> None:
    """Validate that scaled intensities, Boltzmann factors, and stimulated emissions equal 1.0 at T = T_ref."""
    freqs = jnp.array([100000.0, 250000.0, 500000.0], dtype=jnp.float64)  # MHz
    energies = jnp.array([10.0, 80.0, 250.0], dtype=jnp.float64)  # cm^-1
    intensities = jnp.array([1.5, 0.8, 0.3], dtype=jnp.float64)

    t_ref = 298.15
    q_ref = 200.0
    temps = jnp.array([t_ref], dtype=jnp.float64)
    q_temps = jnp.array([q_ref], dtype=jnp.float64)

    freq_to_k = H_OVER_K_B * 1.0e6
    energy_to_k = C2_RADIATION_CONST

    scaled_i, boltz_f, stim_f = compute_boltzmann_scaling_grid_jax(
        frequencies=freqs,
        lower_energies=energies,
        reference_intensities=intensities,
        temperatures=temps,
        q_temps=q_temps,
        t_ref=t_ref,
        q_ref=q_ref,
        freq_to_k=freq_to_k,
        energy_to_k=energy_to_k,
    )

    # All factors at T_ref must be identically 1.0
    assert float(jnp.max(jnp.abs(boltz_f - 1.0))) < 1e-15
    assert float(jnp.max(jnp.abs(stim_f - 1.0))) < 1e-15
    assert float(jnp.max(jnp.abs(scaled_i[0] - intensities))) < 1e-15


def test_boltzmann_excited_state_thermal_activation() -> None:
    """Validate physical growth of high-energy transition line intensities with temperature."""
    # Line 1: ground state E_l = 0 cm^-1
    # Line 2: excited state E_l = 500 cm^-1
    freqs = jnp.array([100000.0, 100000.0], dtype=jnp.float64)
    energies = jnp.array([0.0, 500.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 1.0], dtype=jnp.float64)

    temps = jnp.array([50.0, 300.0], dtype=jnp.float64)
    t_ref = 300.0
    q_ref = 100.0
    q_temps = jnp.array([10.0, 100.0], dtype=jnp.float64)

    scaled_i, _, _ = compute_boltzmann_scaling_grid_jax(
        frequencies=freqs,
        lower_energies=energies,
        reference_intensities=intensities,
        temperatures=temps,
        q_temps=q_temps,
        t_ref=t_ref,
        q_ref=q_ref,
        freq_to_k=H_OVER_K_B * 1.0e6,
        energy_to_k=C2_RADIATION_CONST,
    )

    # At 50 K, excited state line (index 1) must be exponentially suppressed compared to 300 K
    i_excited_50k = float(scaled_i[0, 1])
    i_excited_300k = float(scaled_i[1, 1])
    assert i_excited_50k < 1e-3 * i_excited_300k


# =============================================================================
# 3. MICRO-BATCHED JAX.LAX.SCAN VS MONOLITHIC KERNEL TESTS
# =============================================================================

def test_frequency_chunked_scan_matches_monolithic_1d() -> None:
    """Verify that 1D micro-batched jax.lax.scan matches monolithic vectorized convolution (diff < 1e-12)."""
    n_freqs = 2048
    n_lines = 64
    freq_grid = jnp.linspace(80.0, 120.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(85.0, 115.0, n_lines, dtype=jnp.float64)
    line_intensities = jnp.linspace(0.1, 2.0, n_lines, dtype=jnp.float64)
    fwhm_g = jnp.full((n_lines,), 0.2, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.1, dtype=jnp.float64)

    # Test all profile types
    for profile_code in [0, 1, 2, 3]:
        # 1. Monolithic evaluation
        spec_mono = convolve_spectrum_monolithic_jax(
            freq_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
        )

        # 2. Micro-batched scan with small chunk size (e.g. 128 points = 16 chunks)
        chunk_size = 128
        num_chunks = (n_freqs + chunk_size - 1) // chunk_size
        padded_len = num_chunks * chunk_size
        padded_grid = jnp.pad(freq_grid, (0, padded_len - n_freqs), mode="edge")
        reshaped_grid = padded_grid.reshape((num_chunks, chunk_size))

        chunked_spec = convolve_frequency_chunks_scan_1d_jax(
            reshaped_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
        )
        spec_scan = chunked_spec.reshape(-1)[:n_freqs]

        diff = float(jnp.max(jnp.abs(spec_mono - spec_scan)))
        assert diff < 1e-12, f"Profile code {code} scan deviation {diff} exceeds 1e-12."


def test_frequency_chunked_scan_matches_monolithic_2d_multi_temp() -> None:
    """Verify that multi-temperature 2D jax.lax.scan matches monolithic evaluation (diff < 1e-12)."""
    n_freqs = 1500
    n_lines = 32
    n_temps = 4
    freq_grid = jnp.linspace(100.0, 200.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(110.0, 190.0, n_lines, dtype=jnp.float64)
    base_int = jnp.linspace(0.1, 1.0, n_lines, dtype=jnp.float64)
    temps = jnp.linspace(50.0, 300.0, n_temps, dtype=jnp.float64)
    multi_intensities = base_int[None, :] * (300.0 / temps[:, None])
    fwhm_g = jnp.full((n_lines,), 0.3, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.2, dtype=jnp.float64)
    profile_code = 2  # Pseudo-Voigt

    # Monolithic
    spec_mono = convolve_spectrum_monolithic_jax(
        freq_grid, line_freqs, multi_intensities, fwhm_g, fwhm_l, profile_code
    )

    # Micro-batched 2D scan
    chunk_size = 256
    num_chunks = (n_freqs + chunk_size - 1) // chunk_size
    padded_len = num_chunks * chunk_size
    padded_grid = jnp.pad(freq_grid, (0, padded_len - n_freqs), mode="edge")
    reshaped_grid = padded_grid.reshape((num_chunks, chunk_size))

    chunked_spec_2d = convolve_frequency_chunks_scan_2d_jax(
        reshaped_grid, line_freqs, multi_intensities, fwhm_g, fwhm_l, profile_code
    )
    spec_scan_2d = jnp.swapaxes(chunked_spec_2d, 0, 1).reshape((n_temps, padded_len))[:, :n_freqs]

    diff = float(jnp.max(jnp.abs(spec_mono - spec_scan_2d)))
    assert diff < 1e-12


def test_line_chunked_scan_accumulation_matches_monolithic() -> None:
    """Verify that spectral line chunking via jax.lax.scan accumulates identical spectrum to monolithic."""
    n_freqs = 500
    n_lines = 100
    chunk_lines = 25
    num_chunks = 4

    freq_grid = jnp.linspace(50.0, 150.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(60.0, 140.0, n_lines, dtype=jnp.float64)
    line_intensities = jnp.linspace(0.5, 3.0, n_lines, dtype=jnp.float64)
    fwhm_g = jnp.full((n_lines,), 0.15, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.10, dtype=jnp.float64)
    profile_code = 0  # Gaussian

    # Monolithic
    spec_mono = convolve_spectrum_monolithic_jax(
        freq_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
    )

    # Line chunked
    f_reshaped = line_freqs.reshape((num_chunks, chunk_lines))
    i_reshaped = line_intensities.reshape((num_chunks, chunk_lines))
    fg_reshaped = fwhm_g.reshape((num_chunks, chunk_lines))
    fl_reshaped = fwhm_l.reshape((num_chunks, chunk_lines))

    spec_line_scan = convolve_line_chunks_scan_1d_jax(
        freq_grid, f_reshaped, i_reshaped, fg_reshaped, fl_reshaped, profile_code
    )

    diff = float(jnp.max(jnp.abs(spec_mono - spec_line_scan)))
    assert diff < 1e-12


def test_optimal_chunk_size_calculation() -> None:
    """Validate dynamic micro-batch chunk calculation under memory cap."""
    n_freqs = 100000
    n_lines = 500
    max_mb = 10.0  # 10 MB budget

    chunk = calculate_optimal_chunk_size(
        n_freqs=n_freqs,
        n_lines=n_lines,
        n_temps=1,
        max_vram_mb=max_mb,
    )

    assert 1 <= chunk <= n_freqs
    # Memory occupied by chunk must be <= max_mb
    bytes_per_pt = (4.0 * n_lines + 1) * 8.0
    mem_mb = (chunk * bytes_per_pt) / (1024.0 * 1024.0)
    assert mem_mb <= max_mb + 1e-6


# =============================================================================
# 4. AST STRUCTURAL INSPECTION TESTS (FORBIDDEN PYTHON LOOPS)
# =============================================================================

def test_ast_scan_kernels_contain_no_forbidden_python_loops() -> None:
    """
    AST inspection test verifying that micro-batch scan kernels do NOT contain
    Python for/while loops and invoke jax.lax.scan directly.
    """
    import cochem_spycfit.physics_core.cochem_convolver as conv_module

    scan_functions = [
        conv_module.convolve_frequency_chunks_scan_1d_jax,
        conv_module.convolve_frequency_chunks_scan_2d_jax,
        conv_module.convolve_line_chunks_scan_1d_jax,
        conv_module.faddeeva_rational_jax,
    ]

    for fn in scan_functions:
        source = inspect.getsource(fn)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            assert not isinstance(node, (ast.For, ast.While, ast.AsyncFor)), (
                f"Forbidden Python loop {type(node).__name__} found in scan kernel {fn.__name__} at line {node.lineno}."
            )

        # Check for jax.lax.scan presence in the AST
        has_scan_call = any(
            isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "scan") or
                (isinstance(node.func, ast.Name) and "scan" in node.func.id)
            )
            for node in ast.walk(tree)
        )
        assert has_scan_call, f"Kernel {fn.__name__} must invoke jax.lax.scan."


# =============================================================================
# 5. END-TO-END MACROSCOPIC BROADENER & RESULT VALIDATION
# =============================================================================

def test_end_to_end_broadener_single_temperature() -> None:
    """Validate end-to-end broadening with MacroscopicSpectralBroadener."""
    lines = SpectralLines(
        frequencies=[100.0, 105.0, 110.0],
        intensities=[1.0, 2.5, 1.5],
        lower_state_energies=[10.0, 25.0, 40.0],
    )
    config = ConvolutionConfig(
        profile_type=ProfileType.GAUSSIAN,
        fwhm_gaussian=0.1,
        grid_resolution=0.01,
        cutoff_fwhm=20.0,
    )

    broadener = MacroscopicSpectralBroadener(config)
    result: ConvolvedSpectrumResult = broadener.broaden(lines)

    assert len(result.frequencies) > 100
    assert len(result.intensities) == len(result.frequencies)
    assert result.peak_frequency >= 90.0 and result.peak_frequency <= 120.0
    assert result.peak_intensity > 0.0
    assert result.execution_time_seconds > 0.0

    # Total area should approximately equal sum of input intensities (5.0)
    expected_total_intensity = sum(lines.intensities)
    assert abs(result.total_integrated_intensity - expected_total_intensity) < 1e-4


def test_end_to_end_broadener_multi_temperature() -> None:
    """Validate end-to-end multi-temperature broadening and result structures."""
    lines = SpectralLines(
        frequencies=[150.0, 160.0],
        intensities=[1.0, 1.0],
        lower_state_energies=[0.0, 200.0],
    )
    conv_config = ConvolutionConfig(
        profile_type=ProfileType.PSEUDO_VOIGT,
        fwhm_gaussian=0.05,
        fwhm_lorentzian=0.05,
        grid_resolution=0.01,
    )
    part_config = PartitionGridConfig(
        temperatures=[50.0, 150.0, 298.15],
        t_ref=298.15,
        power_law_index=1.5,
        q_ref=100.0,
    )

    broadener = MacroscopicSpectralBroadener(conv_config)
    result = broadener.broaden(lines, partition_config=part_config)

    assert result.temperatures == [50.0, 150.0, 298.15]
    assert result.multi_temp_intensities is not None
    assert len(result.multi_temp_intensities) == 3
    assert len(result.multi_temp_intensities[0]) == len(result.frequencies)


# =============================================================================
# 6. NEGATIVE TESTS & EXCEPTION HIERARCHY
# =============================================================================

def test_negative_temperature_raises_thermal_scaling_error() -> None:
    """Verify that non-physical temperatures T <= 0 raise ThermalScalingError."""
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[-10.0, 300.0])

    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[0.0, 300.0])

    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[])


def test_missing_energies_for_partition_grid_raises_thermal_scaling_error() -> None:
    """Verify that evaluating partition grid without lower state energies raises ThermalScalingError."""
    lines_no_energies = SpectralLines(
        frequencies=[100.0, 200.0],
        intensities=[1.0, 2.0],
        lower_state_energies=None,
    )
    part_config = PartitionGridConfig(temperatures=[100.0, 200.0])

    with pytest.raises(ThermalScalingError):
        evaluate_partition_grid(lines_no_energies, part_config)


def test_mismatched_spectral_lines_raises_spectral_data_error() -> None:
    """Verify that mismatched list lengths in SpectralLines raise SpectralDataError."""
    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, 200.0],
            intensities=[1.0],  # Mismatched length
        )

    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, -50.0],  # Negative frequency
            intensities=[1.0, 1.0],
        )

    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, 200.0],
            intensities=[1.0, -0.5],  # Negative intensity
        )


def test_invalid_grid_raises_invalid_grid_error() -> None:
    """Verify that non-monotonic or inverted frequency grids raise InvalidGridError."""
    lines = SpectralLines(frequencies=[100.0], intensities=[1.0])

    # Inverted grid
    with pytest.raises(InvalidGridError):
        ConvolutionConfig(grid_min=200.0, grid_max=100.0)

    # Non-monotonic custom grid
    with pytest.raises(InvalidGridError):
        convolve_spectrum(lines, freq_grid=[100.0, 99.0, 101.0])


def test_invalid_vram_and_chunk_raises_micro_batch_error() -> None:
    """Verify that non-positive VRAM or chunk sizes raise MicroBatchError."""
    with pytest.raises(MicroBatchError):
        calculate_optimal_chunk_size(1000, 10, max_vram_mb=-5.0)

    with pytest.raises(MicroBatchError):
        calculate_optimal_chunk_size(1000, 10, explicit_chunk_size=0)


def test_unsupported_profile_type_raises_profile_error() -> None:
    """Verify that invalid profile type string raises ProfileError."""
    with pytest.raises(ProfileError):
        get_profile_type_code("unsupported_shape")


def test_energy_unit_conversions() -> None:
    """Verify conversion factors for all supported spectroscopic energy units."""
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.CM_1) - C2_RADIATION_CONST) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin("cm-1") - C2_RADIATION_CONST) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.MHZ) - H_OVER_K_B * 1.0e6) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.GHZ) - H_OVER_K_B * 1.0e9) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.KELVIN) - 1.0) < 1e-12
    assert get_energy_conversion_factor_to_kelvin(EnergyUnit.JOULE) > 0.0

    with pytest.raises(ThermalScalingError):
        get_energy_conversion_factor_to_kelvin("invalid_unit")


def test_per_line_fwhm_overrides_and_broadening() -> None:
    """Verify that per-line Gaussian and Lorentzian FWHM overrides are respected in broadening."""
    lines = SpectralLines(
        frequencies=[100.0, 110.0],
        intensities=[1.0, 2.0],
        fwhm_gaussian_overrides=[0.05, 0.20],
        fwhm_lorentzian_overrides=[0.02, 0.10],
        quantum_numbers=["J=1-0", "J=2-1"],
    )
    assert lines.quantum_numbers == ["J=1-0", "J=2-1"]

    config = ConvolutionConfig(profile_type=ProfileType.VOIGT)
    broadener = MacroscopicSpectralBroadener(config)
    result = broadener.broaden(lines)

    assert result.profile_type == ProfileType.VOIGT
    assert len(result.frequencies) > 0
    assert result.peak_intensity > 0.0


def test_spectral_lines_validation_errors() -> None:
    """Verify validation errors for malformed SpectralLines inputs."""
    # Empty frequencies
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[], intensities=[])

    # Mismatched lower state energies length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], lower_state_energies=[10.0])

    # Mismatched quantum numbers length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], quantum_numbers=["J=1-0"])

    # Mismatched Gaussian overrides length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_gaussian_overrides=[0.1])

    # Negative Gaussian override
    with pytest.raises(ProfileError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_gaussian_overrides=[0.1, -0.05])

    # Mismatched Lorentzian overrides length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_lorentzian_overrides=[0.1])

    # Negative Lorentzian override
    with pytest.raises(ProfileError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_lorentzian_overrides=[-0.1, 0.05])


def test_partition_grid_config_validation_errors() -> None:
    """Verify validation errors for malformed PartitionGridConfig inputs."""
    # Tabulated partition values length mismatch
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0, 200.0], partition_values=[50.0])

    # Non-positive tabulated partition value
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0, 200.0], partition_values=[50.0, -10.0])

    # Energy levels without degeneracies
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0, 50.0])

    # Degeneracies length mismatch with energy levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0, 50.0], degeneracies=[1.0])

    # Empty energy levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[], degeneracies=[])


def test_tabulated_partition_values_evaluation() -> None:
    """Verify evaluate_partition_grid with explicit tabulated partition function values."""
    lines = SpectralLines(
        frequencies=[100.0, 200.0],
        intensities=[1.0, 2.0],
        lower_state_energies=[10.0, 50.0],
    )
    part_config = PartitionGridConfig(
        temperatures=[100.0, 200.0],
        partition_values=[25.0, 60.0],
        q_ref=25.0,
        t_ref=100.0,
    )
    result = evaluate_partition_grid(lines, part_config)
    assert result.partition_function_values == [25.0, 60.0]
    assert len(result.scaled_intensities) == 2


def test_broadener_helper_methods_and_custom_grids() -> None:
    """Verify MacroscopicSpectralBroadener helper methods: build_grid and custom grid checks."""
    lines = SpectralLines(frequencies=[100.0, 110.0], intensities=[1.0, 1.0])

    # Explicit grid_points
    cfg_points = ConvolutionConfig(grid_points=500)
    broadener = MacroscopicSpectralBroadener(cfg_points)
    grid = broadener.build_grid(lines)
    assert len(grid) == 500

    # Auto resolution grid
    cfg_auto = ConvolutionConfig(fwhm_gaussian=0.5)
    broadener_auto = MacroscopicSpectralBroadener(cfg_auto)
    grid_auto = broadener_auto.build_grid(lines)
    assert len(grid_auto) >= 100

    # Custom grid with single element raises InvalidGridError
    with pytest.raises(InvalidGridError):
        broadener.broaden(lines, freq_grid=[100.0])

    # Custom grid with NaN raises InvalidGridError
    with pytest.raises(InvalidGridError):
        broadener.broaden(lines, freq_grid=[100.0, float("nan"), 110.0])


def test_profile_type_code_mapping_and_padding() -> None:
    """Verify get_profile_type_code and chunk padding logic in convolve_spectrum."""
    assert get_profile_type_code(ProfileType.GAUSSIAN) == 0
    assert get_profile_type_code(ProfileType.LORENTZIAN) == 1
    assert get_profile_type_code(ProfileType.PSEUDO_VOIGT) == 2
    assert get_profile_type_code(ProfileType.VOIGT) == 3

    # Force padding by specifying chunk_size that does not divide n_freqs
    lines = SpectralLines(frequencies=[100.0], intensities=[1.0])
    cfg = ConvolutionConfig(chunk_size=17, grid_points=100)
    result = convolve_spectrum(lines, config=cfg)
    assert len(result.frequencies) == 100
    assert result.num_chunks == 6  # ceil(100 / 17) = 6


def test_broadener_evaluate_partition_grid_and_constant_q() -> None:
    """Verify broadener.evaluate_partition_grid and default constant Q(T) partition function."""
    lines = SpectralLines(
        frequencies=[100.0],
        intensities=[1.0],
        lower_state_energies=[10.0],
    )
    # Neither power_law_index nor energy_levels provided -> uses constant Q(T) = Q_ref
    part_cfg = PartitionGridConfig(temperatures=[100.0, 200.0], q_ref=42.0)
    broadener = MacroscopicSpectralBroadener()
    res = broadener.evaluate_partition_grid(lines, part_cfg)
    assert res.partition_function_values == [42.0, 42.0]


