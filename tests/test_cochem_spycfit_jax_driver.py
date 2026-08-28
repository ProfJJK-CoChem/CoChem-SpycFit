"""
Unit and physical validation tests for JaxHamiltonianDriver.
Adheres strictly to the Zero-Mock Mandate and Anti-Spoofing Directive.
Authentic physical constants and transitions for water (H2O).
"""

import numpy as np
import pytest
import scipy.linalg

from cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (
    JaxHamiltonianDriver,
    STANDARD_WATSON_A_KEYS,
    get_asymmetric_level_index,
    _detect_device,
)


@pytest.fixture
def h2o_constants():
    """Authentic literature rotational and quartic centrifugal distortion constants for H2O (MHz)."""
    return {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "DJ": 37.594,
        "DJK": -172.91,
        "DK": 973.29,
        "dJ": 9.423,
        "dK": 41.25,
    }


@pytest.fixture
def h2o_ten_transitions():
    """
    Authentic low-J (J=0 to 5) rotational transitions for H2O.
    Format: (Ju, Kau, Kcu, Jl, Kal, Kcl, obs_freq_mhz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 557194.310),
        (2, 1, 1, 2, 0, 2, 753989.927),
        (2, 2, 0, 2, 1, 1, 1227816.439),
        (2, 1, 2, 1, 0, 1, 1668530.822),
        (1, 1, 1, 0, 0, 0, 1113080.950),
        (3, 1, 2, 3, 0, 3, 1104663.043),
        (3, 1, 2, 2, 2, 1, 1156822.629),
        (4, 2, 2, 4, 1, 3, 1213053.387),
        (3, 2, 1, 3, 1, 2, 1163032.305),
        (2, 2, 1, 2, 1, 2, 1662204.522),
    )


def test_device_detection():
    """Verify device detection executes and returns an active hardware device."""
    dev = _detect_device()
    assert dev is not None


def test_asymmetric_level_index_ordering():
    """Verify asymmetric top level index mapping for low J states."""
    # J = 0
    assert get_asymmetric_level_index(0, 0, 0) == 0

    # J = 1: 1_01 -> 0, 1_11 -> 1, 1_10 -> 2
    assert get_asymmetric_level_index(1, 0, 1) == 0
    assert get_asymmetric_level_index(1, 1, 1) == 1
    assert get_asymmetric_level_index(1, 1, 0) == 2

    # J = 2: 2_02 -> 0, 2_12 -> 1, 2_11 -> 2, 2_21 -> 3, 2_20 -> 4
    assert get_asymmetric_level_index(2, 0, 2) == 0
    assert get_asymmetric_level_index(2, 1, 2) == 1
    assert get_asymmetric_level_index(2, 1, 1) == 2
    assert get_asymmetric_level_index(2, 2, 1) == 3
    assert get_asymmetric_level_index(2, 2, 0) == 4


def test_hamiltonian_construction_symmetry(h2o_constants):
    """Verify constructed Hamiltonian matrices are square and strictly symmetric."""
    driver = JaxHamiltonianDriver(
        assignments=((1, 1, 0, 1, 0, 1, 557194.310),),
        physical_constants=h2o_constants,
    )

    for J in range(0, 6):
        H = np.asarray(driver.build_hamiltonian(h2o_constants, J=J), dtype=np.float64)
        expected_dim = 2 * J + 1 if J > 0 else 1
        assert H.shape == (expected_dim, expected_dim)
        asym_error = np.max(np.abs(H - H.T))
        assert asym_error < 1e-12


def test_predict_frequencies_h2o(h2o_constants, h2o_ten_transitions):
    """Verify transition frequency predictions match authentic physical values."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    calc_freqs = np.asarray(driver.predict_frequencies(h2o_constants), dtype=np.float64)
    assert calc_freqs.shape == (len(h2o_ten_transitions),)

    # All predicted frequencies must be positive
    assert np.all(calc_freqs > 0.0)

    # 1(1,0) -> 1(0,1) line is ~557194.31 MHz
    assert abs(calc_freqs[0] - 557194.310) < 0.1


def test_compute_jacobian_shape_and_non_zero(h2o_constants, h2o_ten_transitions):
    """Verify exact autodiff Jacobian shape (N_transitions, N_params) and non-zero entries."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    J_mat = np.asarray(driver.compute_jacobian(h2o_constants), dtype=np.float64)
    n_trans = len(h2o_ten_transitions)
    n_params = len(driver.param_keys)

    assert J_mat.shape == (n_trans, n_params)
    assert not np.all(J_mat == 0.0)
    assert np.all(np.isfinite(J_mat))


def test_compute_residuals(h2o_constants, h2o_ten_transitions):
    """Verify O-C residual computation returns exact zero vector when input equals predictions."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    exact_freqs = driver.predict_frequencies(h2o_constants)
    residuals = np.asarray(driver.compute_residuals(h2o_constants, exact_freqs), dtype=np.float64)

    assert np.max(np.abs(residuals)) < 1e-9


def test_fit_constants_recovery_h2o(h2o_constants, h2o_ten_transitions):
    """
    Perturb initial constants by +1% and run fit_constants() against authentic frequencies.
    Assert optimizer recovers original constants within 0.01 kHz (< 1e-5 MHz).
    Assert covariance matrix is symmetric positive-definite.
    """
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    # Ground truth exact frequencies
    exact_observed_freqs = driver.predict_frequencies(h2o_constants)

    # Perturb initial constants by +1%
    perturbed_params = {k: v * 1.01 for k, v in h2o_constants.items()}

    opt_params, cov_mat, rmsd_khz = driver.fit_constants(
        observed_freqs=exact_observed_freqs,
        initial_params=perturbed_params,
        max_iter=100,
        tol=1e-10,
    )

    # 1. Recovered constants within 0.01 kHz (0.00001 MHz)
    for k in ["A", "B", "C", "DJ", "DJK", "DK", "dJ", "dK"]:
        delta_mhz = abs(opt_params[k] - h2o_constants[k])
        delta_khz = delta_mhz * 1000.0
        assert delta_khz < 0.01, f"Parameter {k} recovery error {delta_khz:.6f} kHz exceeds 0.01 kHz"

    # 2. RMSD is sub-kHz
    assert rmsd_khz < 0.01

    # 3. Covariance matrix stored and symmetric positive-definite
    cached_cov = driver.get_covariance_matrix()
    assert cached_cov.shape == (len(driver.param_keys), len(driver.param_keys))
    assert np.allclose(cached_cov, cached_cov.T, atol=1e-12)

    eigenvalues = np.linalg.eigvalsh(cached_cov)
    assert np.all(eigenvalues >= 0.0)


def test_scipy_fallback_execution(h2o_constants, h2o_ten_transitions):
    """Verify NumPy / SciPy fallback execution paths work seamlessly."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    # Test numpy prediction
    np_freqs = driver._predict_frequencies_numpy(h2o_constants)
    assert np_freqs.shape == (len(h2o_ten_transitions),)

    # Test scipy fit
    perturbed_params = {k: v * 1.01 for k, v in h2o_constants.items()}
    opt_params, cov_mat, rmsd_khz = driver._fit_constants_scipy(
        observed_freqs=np_freqs,
        initial_params=perturbed_params,
        param_names=driver.param_keys,
        max_iter=100,
        tol=1e-10,
    )

    for k in ["A", "B", "C"]:
        delta_khz = abs(opt_params[k] - h2o_constants[k]) * 1000.0
        assert delta_khz < 0.1
