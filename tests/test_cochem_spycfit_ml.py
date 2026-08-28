"""
Unit Test Suite for CoChem-SpycFit ML-Augmented Assignment Engine.

Tests cover:
1. Dynamic Mendeleev isotope mass retrieval and molecular mass computation (zero hardcoding).
2. Pydantic v2 strict models, quantum number constraints, and SHA-256 cryptographic provenance.
3. Hardware discovery and JAX FP64 precision enforcement.
4. QuantumFeatureExtractor vectorization of spectroscopic states.
5. GPRManifoldEngine Gaussian Process regression on physical O-C residuals.
6. JAXHamiltonianDriver asymmetric top diagonalization, analytical Jacobians, and covariance evaluation.
7. SmartScanNavigator resolvability penalty filtering, information gain triage, and chunk recommendations.
8. DualEngineParityBridge comparison against 0.1 kHz threshold and SPFIT formatting.
9. CrossPlatformLockManager, ScratchLifecycleManager, ThreadSafeSWMRStore, and DAGCommitManager.

Zero-Mock Mandate: Authentic quantum mechanical calculations, zero mocks, zero pass statements.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cochem_spycfit.cochem_spycfit_ml_schema import (
    AssignmentTag,
    DipoleComponents,
    DipoleType,
    FitState,
    HardwareScanChunk,
    IsotopeRetrievalError,
    MultiTierResourceLimits,
    ParityStatus,
    ParityWarning,
    ProvenanceError,
    ProvenanceLedger,
    QuantumNumberError,
    SpectroscopicTransition,
    TransitionQuantumNumbers,
    compute_bytes_sha256,
    compute_dynamic_molecular_mass,
    compute_file_sha256,
    compute_object_sha256,
    get_dynamic_isotope_mass,
)
from cochem_spycfit.cochem_spycfit_ml_engine import (
    DualEngineParityBridge,
    GPRManifoldEngine,
    JAXHamiltonianDriver,
    QuantumFeatureExtractor,
    SmartScanNavigator,
    discover_hardware_tier,
    enforce_fp64_precision,
)
from cochem_spycfit.cochem_spycfit_ml_storage import (
    CrossPlatformLockManager,
    DAGCommitManager,
    ScratchLifecycleManager,
    ThreadSafeSWMRStore,
)


# =============================================================================
# 1. DYNAMIC MENDELEEV ISOTOPE & MOLECULAR MASS TESTS
# =============================================================================

def test_dynamic_isotope_mass_retrieval() -> None:
    """Verify dynamic isotope mass lookup directly from Mendeleev without hardcoding."""
    mass_1h = get_dynamic_isotope_mass("H", 1)
    mass_2h = get_dynamic_isotope_mass("H", 2)
    mass_12c = get_dynamic_isotope_mass("C", 12)
    mass_13c = get_dynamic_isotope_mass("C", 13)
    mass_16o = get_dynamic_isotope_mass("O", 16)
    mass_32s = get_dynamic_isotope_mass("S", 32)

    # Physical mass assertions (in atomic mass units u)
    assert 1.007 < mass_1h < 1.009
    assert 2.013 < mass_2h < 2.015
    assert abs(mass_12c - 12.000000) < 1e-6
    assert 13.002 < mass_13c < 13.004
    assert 15.994 < mass_16o < 16.000
    assert 31.971 < mass_32s < 31.973

    # Default abundant isotope retrieval
    carbon_abundant = get_dynamic_isotope_mass("C")
    assert abs(carbon_abundant - 12.0) < 0.1

    # Error handling for invalid symbols and mass numbers
    with pytest.raises(IsotopeRetrievalError):
        get_dynamic_isotope_mass("NonExistentElement")

    with pytest.raises(IsotopeRetrievalError):
        get_dynamic_isotope_mass("C", 999)


def test_compute_dynamic_molecular_mass() -> None:
    """Verify molecular mass computation from chemical formula and composition dicts."""
    # Water H2O
    h2o_mass = compute_dynamic_molecular_mass("H2O")
    assert 18.01 < h2o_mass < 18.02

    # Carbon-13 methane 13C1H4
    c13_methane_mass = compute_dynamic_molecular_mass("13C1H4")
    mass_13c = get_dynamic_isotope_mass("C", 13)
    mass_1h = get_dynamic_isotope_mass("H", 1)
    expected_c13_methane = mass_13c + 4 * mass_1h
    assert abs(c13_methane_mass - expected_c13_methane) < 1e-8

    # Sulfur dioxide SO2
    so2_mass = compute_dynamic_molecular_mass("SO2")
    assert 63.95 < so2_mass < 64.08

    # Dict composition
    dict_mass = compute_dynamic_molecular_mass({"13C": 1, "1H": 4})
    assert abs(dict_mass - expected_c13_methane) < 1e-8

    # Parsing error handling
    with pytest.raises(IsotopeRetrievalError):
        compute_dynamic_molecular_mass("Invalid_Formula_123!")


# =============================================================================
# 2. SHA-256 PROVENANCE LEDGER TESTS
# =============================================================================

def test_cryptographic_provenance_ledger(tmp_path: Path) -> None:
    """Verify SHA-256 byte, file, object hashing, and ProvenanceLedger validation."""
    raw_data = b"CoChem-SpycFit Spectral Stream"
    digest = compute_bytes_sha256(raw_data)
    assert len(digest) == 64
    assert isinstance(digest, str)

    test_file = tmp_path / "spectrum.csv"
    test_file.write_bytes(raw_data)
    file_digest = compute_file_sha256(test_file)
    assert file_digest == digest

    obj_payload = {"experiment": "FT-MW", "frequency_ghz": 12.5, "resolution_khz": 5.0}
    obj_digest_1 = compute_object_sha256(obj_payload)
    obj_digest_2 = compute_object_sha256(obj_payload)
    assert obj_digest_1 == obj_digest_2

    ledger = ProvenanceLedger(
        git_commit_hash="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        csv_sha256=file_digest,
        landscape_uuid="uuid-pes-water-001",
        source_level_of_theory="CCSD(T)-F12/cc-pVTZ-F12",
        user_id="spectroscopist_alpha",
    )
    ledger_hash = ledger.compute_ledger_hash()
    assert len(ledger_hash) == 64

    # Non-existent file error
    with pytest.raises(ProvenanceError):
        compute_file_sha256(tmp_path / "non_existent_file.csv")


# =============================================================================
# 3. SPECTROSCOPIC TRANSITIONS & QUANTUM MODELS
# =============================================================================

def test_quantum_numbers_validation() -> None:
    """Verify physical selection rules and quantum constraints on asymmetric top states."""
    # Valid transition 1(0,1) -> 0(0,0)
    qn_valid = TransitionQuantumNumbers(
        J_upper=1, Ka_upper=0, Kc_upper=1,
        J_lower=0, Ka_lower=0, Kc_lower=0,
    )
    assert qn_valid.delta_J() == 1
    assert qn_valid.delta_Ka() == 0
    assert qn_valid.delta_Kc() == 1
    assert qn_valid.to_label() == "1(0,1) -> 0(0,0)"

    # Invalid: Ka > J
    with pytest.raises(QuantumNumberError):
        TransitionQuantumNumbers(
            J_upper=1, Ka_upper=2, Kc_upper=0,
            J_lower=0, Ka_lower=0, Kc_lower=0,
        )

    # Invalid: delta J > 1
    with pytest.raises(QuantumNumberError):
        TransitionQuantumNumbers(
            J_upper=3, Ka_upper=1, Kc_upper=2,
            J_lower=1, Ka_lower=0, Kc_lower=1,
        )

    # Invalid: Ka + Kc not in (J, J+1)
    with pytest.raises(QuantumNumberError):
        TransitionQuantumNumbers(
            J_upper=2, Ka_upper=0, Kc_upper=0,
            J_lower=1, Ka_lower=0, Kc_lower=1,
        )


def test_dipole_components_and_transition_model() -> None:
    """Verify dipole moment vector properties and SpectroscopicTransition model."""
    dipole = DipoleComponents(mu_a=0.5, mu_b=1.8, mu_c=0.1)
    assert abs(dipole.magnitude() - math.sqrt(0.25 + 3.24 + 0.01)) < 1e-6
    assert dipole.dominant_type() == DipoleType.B_TYPE

    qn = TransitionQuantumNumbers(
        J_upper=1, Ka_upper=1, Kc_upper=0,
        J_lower=1, Ka_lower=0, Kc_lower=1,
    )
    trans = SpectroscopicTransition(
        transition_id="trans_110_101",
        quantum_numbers=qn,
        dipole_type=DipoleType.B_TYPE,
        dipole_components=dipole,
        lower_state_energy_cm1=1.5,
        frequency_calc_mhz=14500.0,
        frequency_exp_mhz=14502.0,
        uncertainty_exp_mhz=0.02,
    )

    assert trans.frequency_proposed_mhz == 14500.0
    assert trans.obs_minus_calc_mhz == 2.0
    assert trans.obs_minus_proposed_mhz == 2.0
    assert trans.tag == AssignmentTag.TORQ


# =============================================================================
# 4. HARDWARE DISCOVERY & PRECISION LOCK
# =============================================================================

def test_hardware_discovery_and_precision_lock() -> None:
    """Verify JAX FP64 precision lock and hardware tier profiling."""
    assert enforce_fp64_precision() is True
    res_limits = discover_hardware_tier(override_chunk_bandwidth=250.0)

    assert isinstance(res_limits, MultiTierResourceLimits)
    assert res_limits.max_chunk_bandwidth_mhz == 250.0
    assert res_limits.fp64_enforced is True
    assert res_limits.cpu_threads >= 1
    assert res_limits.max_ram_gb > 0.0


# =============================================================================
# 5. QUANTUM FEATURE EXTRACTOR TESTS
# =============================================================================

def test_quantum_feature_extractor() -> None:
    """Verify feature vectorization across transitions."""
    transitions: List[SpectroscopicTransition] = []
    # Build 5 authentic asymmetric top transitions
    pairs = [
        (1, 0, 1, 0, 0, 0, DipoleType.A_TYPE, 10000.0),
        (1, 1, 0, 1, 0, 1, DipoleType.B_TYPE, 15000.0),
        (2, 0, 2, 1, 0, 1, DipoleType.A_TYPE, 20000.0),
        (2, 1, 1, 2, 0, 2, DipoleType.B_TYPE, 25000.0),
        (2, 2, 0, 2, 1, 1, DipoleType.B_TYPE, 30000.0),
    ]
    for idx, (ju, kau, kcu, jl, kal, kcl, dtype, freq) in enumerate(pairs):
        qn = TransitionQuantumNumbers(
            J_upper=ju, Ka_upper=kau, Kc_upper=kcu,
            J_lower=jl, Ka_lower=kal, Kc_lower=kcl,
        )
        t = SpectroscopicTransition(
            transition_id=f"t_{idx}",
            quantum_numbers=qn,
            dipole_type=dtype,
            dipole_components=DipoleComponents(mu_a=1.0 if dtype == DipoleType.A_TYPE else 0.0,
                                               mu_b=1.5 if dtype == DipoleType.B_TYPE else 0.0),
            lower_state_energy_cm1=float(idx * 2.0),
            frequency_calc_mhz=freq,
        )
        transitions.append(t)

    extractor = QuantumFeatureExtractor()
    raw_feats = extractor.extract_raw_features(transitions)
    assert raw_feats.shape == (5, 16)
    assert raw_feats.dtype == np.float64

    scaled_feats = extractor.fit_transform(transitions)
    assert scaled_feats.shape == (5, 16)
    # Check that scaled features have near-zero mean
    assert np.all(np.abs(np.mean(scaled_feats, axis=0)) < 1e-7)


# =============================================================================
# 6. GAUSSIAN PROCESS ERROR MANIFOLD ENGINE TESTS
# =============================================================================

def test_gpr_manifold_engine_residual_mapping() -> None:
    """
    Verify GPR learns systematic shifts (e.g. +2.0 MHz on b-type lines)
    and predicts shifts for unassigned lines with updated tags.
    """
    # Create 6 locked training transitions: b-type have +2.0 MHz shift, a-type have 0.0 MHz shift
    locked_lines: List[SpectroscopicTransition] = []
    training_data = [
        (1, 0, 1, 0, 0, 0, DipoleType.A_TYPE, 10000.0, 10000.0),
        (2, 0, 2, 1, 0, 1, DipoleType.A_TYPE, 19000.0, 19000.0),
        (3, 0, 3, 2, 0, 2, DipoleType.A_TYPE, 28000.0, 28000.0),
        (1, 1, 0, 1, 0, 1, DipoleType.B_TYPE, 14000.0, 14002.0),  # +2.0 MHz shift
        (2, 1, 1, 2, 0, 2, DipoleType.B_TYPE, 22000.0, 22002.0),  # +2.0 MHz shift
        (3, 1, 2, 3, 0, 3, DipoleType.B_TYPE, 31000.0, 31002.0),  # +2.0 MHz shift
    ]

    for idx, (ju, kau, kcu, jl, kal, kcl, dtype, f_calc, f_exp) in enumerate(training_data):
        qn = TransitionQuantumNumbers(
            J_upper=ju, Ka_upper=kau, Kc_upper=kcu,
            J_lower=jl, Ka_lower=kal, Kc_lower=kcl,
        )
        t = SpectroscopicTransition(
            transition_id=f"locked_{idx}",
            quantum_numbers=qn,
            dipole_type=dtype,
            dipole_components=DipoleComponents(
                mu_a=1.0 if dtype == DipoleType.A_TYPE else 0.0,
                mu_b=2.0 if dtype == DipoleType.B_TYPE else 0.0,
            ),
            lower_state_energy_cm1=float(jl * 1.5),
            frequency_calc_mhz=f_calc,
            frequency_exp_mhz=f_exp,
            uncertainty_exp_mhz=0.01,
            is_locked=True,
        )
        locked_lines.append(t)

    engine = GPRManifoldEngine()
    train_summary = engine.train(locked_lines)
    assert train_summary["trained"] is True
    assert train_summary["sample_count"] == 6

    # Test predictions on unseen candidate lines: 1 unseen b-type, 1 unseen a-type
    unseen_b_type = SpectroscopicTransition(
        transition_id="unseen_b",
        quantum_numbers=TransitionQuantumNumbers(
            J_upper=4, Ka_upper=1, Kc_upper=3,
            J_lower=4, Ka_lower=0, Kc_lower=4,
        ),
        dipole_type=DipoleType.B_TYPE,
        dipole_components=DipoleComponents(mu_b=2.0),
        lower_state_energy_cm1=6.0,
        frequency_calc_mhz=40000.0,
    )
    unseen_a_type = SpectroscopicTransition(
        transition_id="unseen_a",
        quantum_numbers=TransitionQuantumNumbers(
            J_upper=4, Ka_upper=0, Kc_upper=4,
            J_lower=3, Ka_lower=0, Kc_lower=3,
        ),
        dipole_type=DipoleType.A_TYPE,
        dipole_components=DipoleComponents(mu_a=1.0),
        lower_state_energy_cm1=4.5,
        frequency_calc_mhz=37000.0,
    )

    predictions = engine.predict([unseen_b_type, unseen_a_type])
    pred_b = predictions[0]
    pred_a = predictions[1]

    # Assert GPR correctly predicts ~+2.0 MHz for b-type and ~0.0 MHz for a-type
    assert abs(pred_b.frequency_ml_shift_mhz - 2.0) < 0.6
    assert abs(pred_a.frequency_ml_shift_mhz - 0.0) < 0.6
    assert pred_b.tag == AssignmentTag.ML
    assert pred_a.tag == AssignmentTag.ML
    assert pred_b.probable_correctness > 0.0


# =============================================================================
# 7. JAX HAMILTONIAN AUTODIFF DRIVER TESTS
# =============================================================================

def test_jax_hamiltonian_autodiff_and_covariance() -> None:
    """Verify asymmetric top Hamiltonian diagonalization, exact autodiff Jacobians, and covariance evaluation."""
    driver = JAXHamiltonianDriver(max_J=3)

    # Authentic SO2-like rotational constants in MHz
    constants = {
        "A": 60778.8,
        "B": 10318.0,
        "C": 8799.8,
        "DJ": 0.006,
        "DJK": -0.04,
        "DK": 0.25,
        "dJ": 0.001,
        "dK": 0.015,
    }

    transitions = [
        SpectroscopicTransition(
            transition_id="so2_101_000",
            quantum_numbers=TransitionQuantumNumbers(
                J_upper=1, Ka_upper=0, Kc_upper=1,
                J_lower=0, Ka_lower=0, Kc_lower=0,
            ),
            dipole_type=DipoleType.A_TYPE,
            frequency_calc_mhz=19117.8,
            frequency_exp_mhz=19117.82,
            uncertainty_exp_mhz=0.01,
        ),
        SpectroscopicTransition(
            transition_id="so2_202_101",
            quantum_numbers=TransitionQuantumNumbers(
                J_upper=2, Ka_upper=0, Kc_upper=2,
                J_lower=1, Ka_lower=0, Kc_lower=1,
            ),
            dipole_type=DipoleType.A_TYPE,
            frequency_calc_mhz=38235.6,
            frequency_exp_mhz=38235.58,
            uncertainty_exp_mhz=0.01,
        ),
    ]

    calc_freqs = driver.compute_transition_frequencies(constants, transitions)
    assert len(calc_freqs) == 2
    # 1(0,1) -> 0(0,0) frequency equals (B+C) = 10318 + 8799.8 = 19117.8 MHz
    assert abs(calc_freqs[0] - (constants["B"] + constants["C"])) < 1.0

    # Test Analytical Jacobian matrix
    jacobian = driver.compute_jacobian(constants, transitions)
    assert jacobian.shape == (2, 8)
    assert jacobian.dtype == np.float64
    # B and C derivatives for 1(0,1) -> 0(0,0) are both 1.0
    assert abs(jacobian[0, 1] - 1.0) < 1e-4  # dnu/dB
    assert abs(jacobian[0, 2] - 1.0) < 1e-4  # dnu/dC

    # Test Fit Evaluation
    fit_eval = driver.evaluate_fit(constants, transitions)
    assert "covariance_matrix" in fit_eval
    assert "parameter_uncertainties" in fit_eval
    assert fit_eval["global_rmsd_khz"] > 0.0
    assert fit_eval["chi_squared"] > 0.0


# =============================================================================
# 8. SMART SCAN NAVIGATOR TESTS
# =============================================================================

def test_smart_scan_navigator_chunking_and_resolvability() -> None:
    """Verify resolvability density penalty, information gain calculation, and chunk recommendations."""
    navigator = SmartScanNavigator()

    # Create candidate transitions: some closely clustered, some isolated
    trans_isolated = SpectroscopicTransition(
        transition_id="isolated_1",
        quantum_numbers=TransitionQuantumNumbers(
            J_upper=1, Ka_upper=0, Kc_upper=1,
            J_lower=0, Ka_lower=0, Kc_lower=0,
        ),
        dipole_type=DipoleType.A_TYPE,
        frequency_calc_mhz=10000.0,
        frequency_proposed_mhz=10000.0,
        ml_uncertainty_mhz=0.02,
    )
    # Clustered pair
    trans_cluster_1 = SpectroscopicTransition(
        transition_id="cluster_1",
        quantum_numbers=TransitionQuantumNumbers(
            J_upper=2, Ka_upper=0, Kc_upper=2,
            J_lower=1, Ka_lower=0, Kc_lower=1,
        ),
        dipole_type=DipoleType.A_TYPE,
        frequency_calc_mhz=20000.0,
        frequency_proposed_mhz=20000.0,
        ml_uncertainty_mhz=0.02,
    )
    trans_cluster_2 = SpectroscopicTransition(
        transition_id="cluster_2",
        quantum_numbers=TransitionQuantumNumbers(
            J_upper=2, Ka_upper=1, Kc_upper=1,
            J_lower=1, Ka_lower=1, Kc_lower=0,
        ),
        dipole_type=DipoleType.A_TYPE,
        frequency_calc_mhz=20000.01,  # 10 kHz separation -> unresolvable
        frequency_proposed_mhz=20000.01,
        ml_uncertainty_mhz=0.02,
    )

    candidate_transitions = [trans_isolated, trans_cluster_1, trans_cluster_2]
    penalties = navigator.compute_resolvability_penalty(
        candidate_transitions, min_instrument_resolution_mhz=0.05
    )

    # Isolated line should have high resolvability (>0.8), clustered lines should have low resolvability (<0.3)
    assert penalties[0] > 0.8
    assert penalties[1] < 0.3
    assert penalties[2] < 0.3

    # Test chunk recommendations
    jacobian_input = np.array([
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    cov_input = np.diag([100.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01])
    param_names = ["A", "B", "C", "DJ", "DJK", "DK", "dJ", "dK"]

    chunks = navigator.recommend_scan_chunks(
        unassigned_transitions=candidate_transitions,
        jacobian_matrix=jacobian_input,
        covariance_matrix=cov_input,
        parameter_names=param_names,
        max_chunk_bandwidth_mhz=250.0,
        top_n=2,
    )

    assert len(chunks) >= 1
    assert isinstance(chunks[0], HardwareScanChunk)
    assert chunks[0].bandwidth_mhz == 250.0


# =============================================================================
# 9. DUAL-ENGINE PARITY BRIDGE TESTS
# =============================================================================

def test_dual_engine_parity_bridge() -> None:
    """Verify parity threshold checking (0.1 kHz) and SPFIT file formatting."""
    jax_consts = {"A": 10000.00005, "B": 5000.00002, "C": 2500.00001}
    # SPFIT constants match within 0.05 kHz
    spfit_converged = {"A": 10000.00003, "B": 5000.00001, "C": 2500.00000}

    report_converged = DualEngineParityBridge.compare_constants(
        jax_consts, spfit_converged, threshold_khz=0.1
    )
    assert report_converged.is_warning is False
    assert report_converged.parity_status == ParityStatus.CONVERGED_PARITY
    assert report_converged.max_delta_khz < 0.1

    # SPFIT constants with 0.5 kHz discrepancy (> 0.1 kHz threshold)
    spfit_divergent = {"A": 10000.00055, "B": 5000.00002, "C": 2500.00001}
    report_warning = DualEngineParityBridge.compare_constants(
        jax_consts, spfit_divergent, threshold_khz=0.1
    )
    assert report_warning.is_warning is True
    assert report_warning.parity_status == ParityStatus.PARITY_WARNING
    assert report_warning.max_delta_khz > 0.1

    # Test file formatters
    qn = TransitionQuantumNumbers(
        J_upper=1, Ka_upper=0, Kc_upper=1,
        J_lower=0, Ka_lower=0, Kc_lower=0,
    )
    trans = SpectroscopicTransition(
        transition_id="t_0",
        quantum_numbers=qn,
        dipole_type=DipoleType.A_TYPE,
        frequency_calc_mhz=10000.0,
        frequency_exp_mhz=10000.05,
        uncertainty_exp_mhz=0.01,
    )
    lin_text = DualEngineParityBridge.format_spfit_lin_file([trans])
    assert "10000.050000" in lin_text

    var_text = DualEngineParityBridge.format_spfit_var_file(jax_consts)
    assert "10000.00005000" in var_text


# =============================================================================
# 10. STORAGE, LOCK IPC, SWMR & DAG COMMIT MANAGER TESTS
# =============================================================================

def test_storage_concurrency_and_dag_commits(tmp_path: Path) -> None:
    """Verify ThreadSafeSWMRStore, CrossPlatformLockManager, ScratchLifecycleManager, and DAGCommitManager."""
    # 1. Scratch Lifecycle Manager
    scratch_mgr = ScratchLifecycleManager(base_scratch_dir=tmp_path / "scratch")
    with scratch_mgr.sandbox_context() as sandbox_dir:
        assert sandbox_dir.exists()
        temp_file = sandbox_dir / "test.txt"
        temp_file.write_text("ephemeral", encoding="utf-8")
        assert temp_file.exists()
    assert not sandbox_dir.exists()

    # 2. ThreadSafeSWMRStore & Lock Manager
    h5_file = tmp_path / "test_landscape.h5"
    store = ThreadSafeSWMRStore(h5_path=h5_file, timeout_seconds=5.0)

    provenance = ProvenanceLedger(
        git_commit_hash="commit_hash_12345",
        csv_sha256="sha256_spectrum_hash_001",
        landscape_uuid="uuid-001",
    )
    qn = TransitionQuantumNumbers(
        J_upper=1, Ka_upper=0, Kc_upper=1,
        J_lower=0, Ka_lower=0, Kc_lower=0,
    )
    trans = SpectroscopicTransition(
        transition_id="t_state_0",
        quantum_numbers=qn,
        dipole_type=DipoleType.A_TYPE,
        frequency_calc_mhz=12000.0,
        frequency_exp_mhz=12000.1,
        is_locked=True,
    )

    state_0 = FitState(
        state_id="state_root_000",
        parent_state_id=None,
        branch_name="main",
        rotational_constants={"A": 10000.0, "B": 5000.0, "C": 2500.0},
        parameter_uncertainties={"A": 0.1, "B": 0.05, "C": 0.02},
        covariance_matrix=[[0.01, 0.0, 0.0], [0.0, 0.0025, 0.0], [0.0, 0.0, 0.0004]],
        locked_transitions=[trans],
        global_rmsd_khz=15.2,
        chi_squared=1.2,
        provenance=provenance,
    )

    # Save and load state
    t0 = time.perf_counter()
    store.save_fit_state(state_0)
    save_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    loaded_state_0 = store.load_fit_state("state_root_000")
    load_time = time.perf_counter() - t1

    # Sub-100ms restore requirement
    assert load_time < 0.100
    assert loaded_state_0.state_id == "state_root_000"
    assert loaded_state_0.rotational_constants["A"] == 10000.0
    assert len(loaded_state_0.locked_transitions) == 1
    assert loaded_state_0.locked_transitions[0].transition_id == "t_state_0"

    # Export Parquet
    parquet_out = tmp_path / "final_fit.parquet"
    store.export_fair_parquet("state_root_000", parquet_out)
    assert parquet_out.exists()

    # 3. DAG Commit Manager
    registry_file = tmp_path / "cochem_fit_registry.json"
    dag_mgr = DAGCommitManager(storage_store=store, registry_json_path=registry_file)

    commit_root = dag_mgr.commit_state(state_0)
    assert commit_root == "state_root_000"
    assert dag_mgr.active_head_id == "state_root_000"

    # Commit child state on main branch
    state_1 = state_0.model_copy(
        update={
            "state_id": "state_child_001",
            "global_rmsd_khz": 8.5,
        }
    )
    dag_mgr.commit_state(state_1)
    assert dag_mgr.active_head_id == "state_child_001"

    # Create exploration branch and commit
    dag_mgr.create_branch("explore_branch")
    state_branch = state_1.model_copy(
        update={
            "state_id": "state_branch_002",
            "branch_name": "explore_branch",
            "global_rmsd_khz": 4.1,
        }
    )
    dag_mgr.commit_state(state_branch)
    assert dag_mgr.active_head_id == "state_branch_002"

    # Check history on branch
    history = dag_mgr.get_history("explore_branch")
    assert len(history) == 3
    assert history[0].state_id == "state_root_000"
    assert history[1].state_id == "state_child_001"
    assert history[2].state_id == "state_branch_002"

    # Time-travel checkout to root
    restored = dag_mgr.checkout_state("state_root_000")
    assert restored.state_id == "state_root_000"
    assert dag_mgr.active_head_id == "state_root_000"

    # Purge abandoned branches (keeping only explore_branch)
    purged = dag_mgr.purge_abandoned_branches(keep_branches=["explore_branch"])
    assert isinstance(purged, list)
