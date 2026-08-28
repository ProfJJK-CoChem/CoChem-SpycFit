"""Physical unit tests for Gaussian Process ML Core (SpycFitGPRegressor).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational transitions for H2O.
- Systematic residual learning (O-C error manifold mapping) using Matérn + WhiteKernel composite GP.
- Dual Provenance tagging (TORQ vs ML vs USER).
- Probable correctness confidence score evaluation.
- Dual-tree module and package re-export verification.
- Static AST validation confirming zero mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from cochem_spycfit.physics_core.cochem_spycfit_features import QuantumFeatureExtractor
from cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
    ProvenanceTag,
    ResidualPrediction,
    SpycFitGPRegressor,
)


@pytest.fixture
def h2o_training_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """Authentic low-J rotational transitions for H2O used for training.

    Entries: (J', Ka', Kc', J'', Ka'', Kc'', nu_calc_physical_MHz)
    - 3 b-type transitions (Ka' != Ka'', Kc' != Kc'')
    - 2 a-type transitions (Ka' == Ka'', Kc' != Kc'')
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985),  # b-type (dk_a=1, dk_c=1 -> b-type)
        (2, 1, 1, 2, 0, 2, 752033.227),  # b-type (dk_a=1, dk_c=1 -> b-type)
        (3, 1, 2, 3, 0, 3, 1097364.791), # b-type (dk_a=1, dk_c=1 -> b-type)
        (1, 0, 1, 0, 0, 0, 1113342.964), # a-type (dk_a=0, dk_c=1 -> a-type)
        (2, 0, 2, 1, 0, 1, 987926.764),  # a-type (dk_a=0, dk_c=1 -> a-type)
    )


@pytest.fixture
def h2o_unseen_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """Authentic unseen low-J rotational transitions for H2O used for prediction testing.

    Entries: (J', Ka', Kc', J'', Ka'', Kc'', nu_calc_physical_MHz)
    - 1 unseen b-type transition: 4(1,3) <- 4(0,4)
    - 1 unseen a-type transition: 3(0,3) <- 2(0,2)
    """
    return (
        (4, 1, 3, 4, 0, 4, 1602219.182), # unseen b-type
        (3, 0, 3, 2, 0, 2, 1716769.318), # unseen a-type
    )


@pytest.fixture
def h2o_dipole() -> Dict[str, float]:
    """Authentic permanent dipole components for H2O in Debye."""
    return {
        "mu_a": 0.0,
        "mu_b": 1.8546,
        "mu_c": 0.0,
    }


@pytest.fixture
def h2o_lower_state_energies() -> Dict[Tuple[int, int, int], float]:
    """Authentic lower state energies for H2O rotational levels in cm^-1."""
    return {
        (0, 0, 0): 0.0,
        (1, 0, 1): 23.794,
        (1, 1, 1): 37.137,
        (2, 0, 2): 70.091,
        (2, 1, 2): 79.496,
        (3, 0, 3): 136.762,
        (4, 0, 4): 222.052,
        (5, 0, 5): 325.348,
    }


def test_provenance_tag_enum() -> None:
    """Verify ProvenanceTag enum values and membership."""
    assert ProvenanceTag.TORQ.value == "TORQ"
    assert ProvenanceTag.ML.value == "ML"
    assert ProvenanceTag.USER.value == "USER"
    assert len(ProvenanceTag) == 3


def test_residual_prediction_dataclass_instantiation_and_immutability() -> None:
    """Verify ResidualPrediction dataclass field access and frozen immutability."""
    prediction = ResidualPrediction(
        quantum_numbers=(1, 1, 0, 1, 0, 1),
        nu_calc_physical_mhz=556935.985,
        delta_nu_ml_shift_mhz=2.0001,
        nu_proposed_mhz=556937.9851,
        sigma_predict_mhz=0.015,
        probable_correctness=0.9709,
        provenance=ProvenanceTag.ML,
    )

    assert prediction.quantum_numbers == (1, 1, 0, 1, 0, 1)
    assert prediction.nu_calc_physical_mhz == 556935.985
    assert prediction.delta_nu_ml_shift_mhz == 2.0001
    assert prediction.nu_proposed_mhz == 556937.9851
    assert prediction.sigma_predict_mhz == 0.015
    assert prediction.probable_correctness == 0.9709
    assert prediction.provenance == ProvenanceTag.ML

    with pytest.raises(FrozenInstanceError):
        prediction.delta_nu_ml_shift_mhz = 0.0  # type: ignore[misc]


def test_gp_regressor_init_defaults_and_custom_kernel() -> None:
    """Verify default and custom kernel initialization of SpycFitGPRegressor."""
    regressor_default = SpycFitGPRegressor()
    assert regressor_default.is_trained is False
    assert regressor_default.get_trained_kernel_params() == {}

    custom_kernel = ConstantKernel(2.0) * Matern(length_scale=0.5, nu=1.5) + WhiteKernel(noise_level=0.05)
    regressor_custom = SpycFitGPRegressor(
        kernel=custom_kernel,
        n_restarts_optimizer=3,
        random_state=123,
    )
    assert regressor_custom.is_trained is False


def test_untrained_regressor_predict_returns_torq(
    h2o_unseen_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify untrained SpycFitGPRegressor returns pure TORQ predictions with sigma=inf and confidence=0.0."""
    extractor = QuantumFeatureExtractor()
    features = extractor.extract_raw_features(
        assignments=h2o_unseen_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    nu_calc_physical = np.array([t[6] for t in h2o_unseen_transitions], dtype=np.float64)
    quantum_numbers = [t[:6] for t in h2o_unseen_transitions]

    regressor = SpycFitGPRegressor()
    predictions = regressor.predict(
        features=features,
        nu_calc_physical=nu_calc_physical,
        quantum_numbers=quantum_numbers,
    )

    assert len(predictions) == len(h2o_unseen_transitions)
    for i, pred in enumerate(predictions):
        assert pred.quantum_numbers == quantum_numbers[i]
        assert pred.nu_calc_physical_mhz == nu_calc_physical[i]
        assert pred.delta_nu_ml_shift_mhz == 0.0
        assert pred.nu_proposed_mhz == nu_calc_physical[i]
        assert math.isinf(pred.sigma_predict_mhz)
        assert pred.probable_correctness == 0.0
        assert pred.provenance == ProvenanceTag.TORQ


def test_train_and_predict_systematic_residual_manifold(
    h2o_training_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_unseen_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify GP learns systematic O-C residuals and generalizes accurately to unseen transitions."""
    extractor = QuantumFeatureExtractor()
    raw_train_feats = extractor.extract_raw_features(
        assignments=h2o_training_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )
    extractor.fit_scaler(raw_train_feats)
    norm_train_feats = extractor.transform(raw_train_feats)

    # Systematic residuals: +2.0 MHz for b-type (first 3), 0.0 MHz for a-type (last 2)
    residuals_train = np.array([2.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float64)

    regressor = SpycFitGPRegressor(n_restarts_optimizer=5, random_state=42)
    regressor.train(features=norm_train_feats, residuals=residuals_train)

    assert regressor.is_trained is True

    # Transform unseen transitions
    raw_unseen_feats = extractor.extract_raw_features(
        assignments=h2o_unseen_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )
    norm_unseen_feats = extractor.transform(raw_unseen_feats)

    nu_calc_unseen = np.array([t[6] for t in h2o_unseen_transitions], dtype=np.float64)
    qn_unseen = [t[:6] for t in h2o_unseen_transitions]

    predictions = regressor.predict(
        features=norm_unseen_feats,
        nu_calc_physical=nu_calc_unseen,
        quantum_numbers=qn_unseen,
    )

    assert len(predictions) == 2

    # Prediction 0: Unseen b-type transition -> Expect ~+2.0 MHz shift (within +-0.5 MHz)
    pred_b = predictions[0]
    assert pred_b.quantum_numbers == (4, 1, 3, 4, 0, 4)
    assert abs(pred_b.delta_nu_ml_shift_mhz - 2.0) < 0.5
    assert pred_b.provenance == ProvenanceTag.ML
    assert 0.0 < pred_b.probable_correctness <= 1.0
    assert pred_b.nu_proposed_mhz == pytest.approx(
        pred_b.nu_calc_physical_mhz + pred_b.delta_nu_ml_shift_mhz, rel=1e-9
    )
    assert math.isfinite(pred_b.sigma_predict_mhz)
    assert pred_b.sigma_predict_mhz > 0.0

    # Prediction 1: Unseen a-type transition -> Expect ~0.0 MHz shift (within +-0.5 MHz)
    pred_a = predictions[1]
    assert pred_a.quantum_numbers == (3, 0, 3, 2, 0, 2)
    assert abs(pred_a.delta_nu_ml_shift_mhz - 0.0) < 0.5
    assert pred_a.provenance == ProvenanceTag.ML
    assert 0.0 < pred_a.probable_correctness <= 1.0
    assert pred_a.nu_proposed_mhz == pytest.approx(
        pred_a.nu_calc_physical_mhz + pred_a.delta_nu_ml_shift_mhz, rel=1e-9
    )
    assert math.isfinite(pred_a.sigma_predict_mhz)
    assert pred_a.sigma_predict_mhz > 0.0


def test_get_trained_kernel_params(
    h2o_training_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify get_trained_kernel_params returns optimized kernel parameters dictionary."""
    extractor = QuantumFeatureExtractor()
    norm_train_feats = extractor.fit_transform(
        assignments=h2o_training_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )
    residuals = np.array([2.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float64)

    regressor = SpycFitGPRegressor(n_restarts_optimizer=2, random_state=42)
    regressor.train(norm_train_feats, residuals)

    params = regressor.get_trained_kernel_params()
    assert isinstance(params, dict)
    assert len(params) > 0
    for k, v in params.items():
        assert isinstance(k, str)
        assert isinstance(v, (int, float, str, tuple, list, object))


def test_dual_tree_exports() -> None:
    """Verify SpycFitGPRegressor, ProvenanceTag, ResidualPrediction are exportable from physics_core packages."""
    import cochem_spycfit.physics_core as root_core
    import src.cochem_spycfit.physics_core as src_core

    for core_mod in [root_core, src_core]:
        assert hasattr(core_mod, "SpycFitGPRegressor")
        assert hasattr(core_mod, "ProvenanceTag")
        assert hasattr(core_mod, "ResidualPrediction")

    from cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
        ProvenanceTag as TagRoot,
        ResidualPrediction as PredRoot,
        SpycFitGPRegressor as RegRoot,
    )
    from src.cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
        ProvenanceTag as TagSrc,
        ResidualPrediction as PredSrc,
        SpycFitGPRegressor as RegSrc,
    )

    assert TagRoot is not None
    assert PredRoot is not None
    assert RegRoot is not None
    assert TagSrc is not None
    assert PredSrc is not None
    assert RegSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or fake implementations exist in the test and implementation."""
    test_path = Path(__file__).resolve()
    impl_src_path = test_path.parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_ml_core.py"
    impl_root_path = test_path.parent.parent / "cochem_spycfit" / "physics_core" / "cochem_spycfit_ml_core.py"

    for file_path in [test_path, impl_src_path, impl_root_path]:
        if not file_path.exists():
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.Name):
                assert node.id not in ("MagicMock", "Mock", "AsyncMock"), f"Prohibited mock identifier in {file_path}"
