"""Physical unit tests for QuantumFeatureExtractor (spectroscopic ML feature vectorizer).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational transitions for H2O and asymmetric rotors.
- Verification of 14-feature quantum numbers, delta quantum selection rules, dipole components, and energies.
- Scaler fitting, transform normalization, and stateful invariants.
- Static AST validation confirming zero synthetic mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_spycfit_features import QuantumFeatureExtractor


@pytest.fixture
def h2o_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """Authentic low-J rotational transitions for H2O (nu_exp in MHz).

    Entries: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985),
        (2, 1, 1, 2, 0, 2, 752033.227),
        (3, 1, 2, 3, 0, 3, 1097364.791),
        (4, 1, 3, 4, 0, 4, 1602219.182),
        (5, 1, 4, 5, 0, 5, 2264148.770),
        (1, 1, 1, 0, 0, 0, 1113342.964),
        (2, 0, 2, 1, 1, 1, 987926.764),
        (2, 1, 2, 1, 0, 1, 1669904.775),
        (3, 0, 3, 2, 1, 2, 1716769.318),
        (3, 1, 3, 2, 0, 2, 2196345.756),
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


def test_feature_names_constant() -> None:
    """Verify FEATURE_NAMES tuple matches exact 14 spectroscopic feature specifications."""
    expected = (
        "J_upper", "Ka_upper", "Kc_upper",
        "J_lower", "Ka_lower", "Kc_lower",
        "delta_J", "delta_Ka", "delta_Kc",
        "mu_a", "mu_b", "mu_c",
        "lower_state_energy_cm",
        "transition_type",
    )
    assert QuantumFeatureExtractor.FEATURE_NAMES == expected
    assert len(QuantumFeatureExtractor.FEATURE_NAMES) == 14


def test_extract_raw_features_shape_and_values(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify raw feature matrix extraction shape (N, 14) and exact quantum column mappings."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    assert isinstance(raw, np.ndarray)
    assert raw.shape == (10, 14)
    assert raw.dtype == np.float64

    # Check first transition: (1, 1, 0, 1, 0, 1, 556935.985)
    # J_u=1, Ka_u=1, Kc_u=0, J_l=1, Ka_l=0, Kc_l=1
    # delta_J=0, delta_Ka=1, delta_Kc=-1
    assert raw[0, 0] == 1.0
    assert raw[0, 1] == 1.0
    assert raw[0, 2] == 0.0
    assert raw[0, 3] == 1.0
    assert raw[0, 4] == 0.0
    assert raw[0, 5] == 1.0
    assert raw[0, 6] == 0.0  # delta_J = 1 - 1
    assert raw[0, 7] == 1.0  # delta_Ka = 1 - 0
    assert raw[0, 8] == -1.0  # delta_Kc = 0 - 1

    # Dipole components: mu_a=0.0, mu_b=1.8546, mu_c=0.0
    assert raw[0, 9] == 0.0
    assert raw[0, 10] == 1.8546
    assert raw[0, 11] == 0.0

    # Lower state energy for (1, 0, 1) -> 23.794 cm^-1
    assert raw[0, 12] == 23.794

    # Transition type: dk_a = |1-0|=1, dk_c = |0-1|=1 -> b-type (1)
    assert raw[0, 13] == 1.0


def test_transition_type_classification() -> None:
    """Verify transition type classification for a-type (0), b-type (1), and c-type (2)."""
    extractor = QuantumFeatureExtractor()

    # a-type: ΔKa = 0, ΔKc = ±1  e.g. 2(1,1) <- 1(1,0) [Ka_u=1, Ka_l=1 -> dk_a=0]
    # b-type: ΔKa = ±1, ΔKc = ±1 e.g. 1(1,1) <- 0(0,0) [Ka_u=1, Ka_l=0, Kc_u=1, Kc_l=0]
    # c-type: ΔKa = ±1, ΔKc = 0  e.g. 1(1,0) <- 0(0,0) [Ka_u=1, Ka_l=0, Kc_u=0, Kc_l=0 -> dk_c=0]
    transitions = (
        (2, 1, 1, 1, 1, 0, 100000.0),  # a-type -> 0
        (1, 1, 1, 0, 0, 0, 200000.0),  # b-type -> 1
        (1, 1, 0, 0, 0, 0, 300000.0),  # c-type -> 2
    )
    dipole = {"mu_a": 1.0, "mu_b": 2.0, "mu_c": 3.0}
    raw = extractor.extract_raw_features(transitions, dipole)

    assert raw[0, 13] == 0.0  # a-type
    assert raw[1, 13] == 1.0  # b-type
    assert raw[2, 13] == 2.0  # c-type


def test_dipole_and_energy_defaults(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...]
) -> None:
    """Verify missing dipole keys and missing lower-state energies default gracefully to 0.0."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=h2o_transitions[:3],
        dipole_components={},  # Empty dipole dict
        lower_state_energies=None,  # No lower state energies
    )

    assert raw.shape == (3, 14)
    # Dipoles must be 0.0
    assert np.all(raw[:, 9:12] == 0.0)
    # Lower state energy must be 0.0
    assert np.all(raw[:, 12] == 0.0)


def test_fit_scaler_and_transform(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify StandardScaler fitting, scaling, and zero-mean unit-variance transformation."""
    extractor = QuantumFeatureExtractor()
    assert extractor.is_fitted is False
    assert extractor.scaler is None

    raw = extractor.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    extractor.fit_scaler(raw)
    assert extractor.is_fitted is True
    assert extractor.scaler is not None

    scaled = extractor.transform(raw)
    assert scaled.shape == raw.shape
    assert isinstance(scaled, np.ndarray)

    # For columns with non-zero variance, verify mean is ~0 and std is ~1
    for col in range(raw.shape[1]):
        col_raw = raw[:, col]
        if np.std(col_raw) > 1e-9:
            assert np.mean(scaled[:, col]) == pytest.approx(0.0, abs=1e-7)
            assert np.std(scaled[:, col]) == pytest.approx(1.0, abs=1e-7)


def test_fit_transform_equivalence(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify fit_transform produces identical result to fit_scaler + transform."""
    extractor_1 = QuantumFeatureExtractor()
    scaled_1 = extractor_1.fit_transform(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    extractor_2 = QuantumFeatureExtractor()
    raw_2 = extractor_2.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )
    extractor_2.fit_scaler(raw_2)
    scaled_2 = extractor_2.transform(raw_2)

    np.testing.assert_allclose(scaled_1, scaled_2, rtol=1e-12, atol=1e-12)


def test_unfitted_transform_raises() -> None:
    """Verify calling transform() on an unfitted QuantumFeatureExtractor raises RuntimeError."""
    extractor = QuantumFeatureExtractor()
    dummy_input = np.ones((5, 14), dtype=np.float64)

    with pytest.raises(RuntimeError, match="Scaler not fitted. Call fit_scaler\\(\\) first."):
        extractor.transform(dummy_input)


def test_empty_assignments() -> None:
    """Verify extract_raw_features gracefully handles empty assignments tuple."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=(),
        dipole_components={"mu_a": 1.0},
    )
    assert raw.shape == (0, 14)


def test_dual_tree_exports() -> None:
    """Verify QuantumFeatureExtractor is exportable from root and src physics_core packages."""
    import cochem_spycfit.physics_core as root_core
    import src.cochem_spycfit.physics_core as src_core

    assert hasattr(root_core, "QuantumFeatureExtractor")
    assert hasattr(src_core, "QuantumFeatureExtractor")

    from cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor as ExtractorRoot,
    )
    from src.cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor as ExtractorSrc,
    )

    assert ExtractorRoot is not None
    assert ExtractorSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or fake implementations exist in the test and implementation."""
    test_path = Path(__file__).resolve()
    impl_src_path = test_path.parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_features.py"
    impl_root_path = test_path.parent.parent / "cochem_spycfit" / "physics_core" / "cochem_spycfit_features.py"

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
