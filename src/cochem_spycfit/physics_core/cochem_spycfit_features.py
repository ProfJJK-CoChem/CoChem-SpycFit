"""Quantum feature extractor for ML-driven spectroscopic residual prediction.

Transforms raw spectroscopic quantum numbers, physical dipole moments, and state
energies into standardized NumPy feature matrices suitable for Gaussian Process Regression.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class QuantumFeatureExtractor:
    """Extracts and normalizes quantum-mechanical features for GP regression."""

    FEATURE_NAMES: Tuple[str, ...] = (
        "J_upper",
        "Ka_upper",
        "Kc_upper",
        "J_lower",
        "Ka_lower",
        "Kc_lower",
        "delta_J",
        "delta_Ka",
        "delta_Kc",
        "mu_a",
        "mu_b",
        "mu_c",
        "lower_state_energy_cm",
        "transition_type",  # 0=a-type, 1=b-type, 2=c-type
    )

    def __init__(self) -> None:
        """Initialize the feature extractor with an unfitted state."""
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """Return the internal fitted StandardScaler instance, or None if unfitted."""
        return self._scaler

    @property
    def is_fitted(self) -> bool:
        """Return True if the internal scaler has been successfully fitted."""
        return self._is_fitted

    def extract_raw_features(
        self,
        assignments: Sequence[Tuple[int, int, int, int, int, int, float]],
        dipole_components: Mapping[str, float],
        lower_state_energies: Optional[Mapping[Tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """Extract raw (unnormalized) features from assignment data.

        Parameters
        ----------
        assignments : Sequence of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        dipole_components : Mapping[str, float]
            {'mu_a': float, 'mu_b': float, 'mu_c': float} in Debye.
        lower_state_energies : Mapping[Tuple[int, int, int], float], optional
            Map from (J'', Ka'', Kc'') -> lower-state energy in cm^-1.
            If None or level missing, defaults to 0.0.

        Returns
        -------
        np.ndarray
            Shape (N_assignments, 14). Feature matrix of dtype float64.
        """
        n = len(assignments)
        features = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float64)

        for i, (j_u, ka_u, kc_u, j_l, ka_l, kc_l, _nu) in enumerate(assignments):
            # Quantum numbers
            features[i, 0] = float(j_u)
            features[i, 1] = float(ka_u)
            features[i, 2] = float(kc_u)
            features[i, 3] = float(j_l)
            features[i, 4] = float(ka_l)
            features[i, 5] = float(kc_l)
            features[i, 6] = float(j_u - j_l)  # delta_J
            features[i, 7] = float(ka_u - ka_l)  # delta_Ka
            features[i, 8] = float(kc_u - kc_l)  # delta_Kc

            # Dipole components
            features[i, 9] = float(dipole_components.get("mu_a", 0.0))
            features[i, 10] = float(dipole_components.get("mu_b", 0.0))
            features[i, 11] = float(dipole_components.get("mu_c", 0.0))

            # Lower-state energy
            if lower_state_energies and (j_l, ka_l, kc_l) in lower_state_energies:
                features[i, 12] = float(lower_state_energies[(j_l, ka_l, kc_l)])
            else:
                features[i, 12] = 0.0

            # Transition type classification
            # a-type: ΔKa=0, ΔKc=±1; b-type: ΔKa=±1, ΔKc=±1; c-type: ΔKa=±1, ΔKc=0
            dk_a = abs(ka_u - ka_l)
            dk_c = abs(kc_u - kc_l)
            if dk_a == 0:
                features[i, 13] = 0.0  # a-type
            elif dk_c == 0:
                features[i, 13] = 2.0  # c-type
            else:
                features[i, 13] = 1.0  # b-type

        return features

    def fit_scaler(self, raw_features: np.ndarray) -> None:
        """Fit the StandardScaler on the raw feature matrix.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features). Raw feature matrix from extract_raw_features.
        """
        self._scaler = StandardScaler()
        self._scaler.fit(raw_features)
        self._is_fitted = True

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Scale features using the fitted StandardScaler.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features).

        Returns
        -------
        np.ndarray
            Normalized feature matrix.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self._scaler.transform(raw_features)

    def fit_transform(
        self,
        assignments: Sequence[Tuple[int, int, int, int, int, int, float]],
        dipole_components: Mapping[str, float],
        lower_state_energies: Optional[Mapping[Tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """Extract, fit, and transform features in one atomic call.

        Parameters
        ----------
        assignments : Sequence of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        dipole_components : Mapping[str, float]
            {'mu_a': float, 'mu_b': float, 'mu_c': float} in Debye.
        lower_state_energies : Mapping[Tuple[int, int, int], float], optional
            Map from (J'', Ka'', Kc'') -> lower-state energy in cm^-1.

        Returns
        -------
        np.ndarray
            Normalized feature matrix of shape (N, 14).
        """
        raw = self.extract_raw_features(assignments, dipole_components, lower_state_energies)
        self.fit_scaler(raw)
        return self.transform(raw)


__all__ = [
    "QuantumFeatureExtractor",
]
