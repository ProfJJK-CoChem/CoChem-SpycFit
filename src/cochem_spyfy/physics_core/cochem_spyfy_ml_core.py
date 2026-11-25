"""Gaussian Process Regressor for systematic residual prediction in spectroscopic fitting."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel

logger = logging.getLogger(__name__)


class ProvenanceTag(Enum):
    """Provenance origin tag for transition predictions."""

    TORQ = "TORQ"  # Pure ab initio prediction (no ML correction)
    ML = "ML"  # ML-shifted prediction (GP correction applied)
    USER = "USER"  # User-assigned (drag-to-shift confirmed)


@dataclass(frozen=True)
class ResidualPrediction:
    """Single transition prediction with ML correction and confidence."""

    quantum_numbers: tuple[int, int, int, int, int, int]  # J', Ka', Kc', J'', Ka'', Kc''
    nu_calc_physical_mhz: float  # ν_calc from Hamiltonian (no ML)
    delta_nu_ml_shift_mhz: float  # Predicted ML systematic shift
    nu_proposed_mhz: float  # ν_calc + Δν_ML
    sigma_predict_mhz: float  # GP prediction uncertainty (1σ)
    probable_correctness: float  # Confidence score (0.0–1.0), derived from σ
    provenance: ProvenanceTag  # TORQ or ML


class SpycFitGPRegressor:
    """Gaussian Process Regressor for learning systematic O−C residuals."""

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        kernel : sklearn kernel, optional
            Composite kernel for the GPR. Defaults to Matérn(ν=2.5) + WhiteKernel.
        n_restarts_optimizer : int
            Number of optimizer restarts for kernel hyperparameter tuning.
        random_state : int
            Random seed for reproducibility.
        """
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=1.0, nu=2.5
            ) + WhiteKernel(noise_level=0.01)
        self._gpr: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            normalize_y=True,
        )
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Return whether the GPR model has been trained."""
        return self._is_trained

    @property
    def gpr(self) -> GaussianProcessRegressor:
        """Return the underlying GaussianProcessRegressor instance."""
        return self._gpr

    def train(
        self,
        features: np.ndarray,
        residuals: np.ndarray,
    ) -> None:
        """
        Train the GPR on O−C residuals of locked assignments.

        Parameters
        ----------
        features : np.ndarray
            Shape (N_locked, N_features). Normalized feature matrix from
            QuantumFeatureExtractor.transform().
        residuals : np.ndarray
            Shape (N_locked,). O−C residuals in MHz:
            residuals[i] = nu_observed[i] - nu_calculated_physical[i].
        """
        features_arr = np.asarray(features, dtype=np.float64)
        residuals_arr = np.asarray(residuals, dtype=np.float64)
        self._gpr.fit(features_arr, residuals_arr)
        self._is_trained = True
        logger.info(
            f"GPR trained on {len(residuals_arr)} locked assignments. "
            f"Kernel: {self._gpr.kernel_}"
        )

    def predict(
        self,
        features: np.ndarray,
        nu_calc_physical: Sequence[float] | np.ndarray,
        quantum_numbers: Sequence[tuple[int, int, int, int, int, int]],
    ) -> list[ResidualPrediction]:
        """
        Predict systematic shifts for unassigned transitions.

        Parameters
        ----------
        features : np.ndarray
            Shape (N_unassigned, N_features). Normalized features for unassigned transitions.
        nu_calc_physical : np.ndarray or Sequence[float]
            Shape (N_unassigned,). Hamiltonian-calculated frequencies in MHz.
        quantum_numbers : list of 6-tuples
            Quantum numbers for each unassigned transition.

        Returns
        -------
        list[ResidualPrediction]
            Predictions with ML-shifted frequencies, uncertainties, and provenance tags.
        """
        nu_calc_arr = np.asarray(nu_calc_physical, dtype=np.float64)
        qn_list = [tuple(int(x) for x in qn) for qn in quantum_numbers]

        if not self._is_trained:
            # Return pure TORQ predictions with no ML correction
            return [
                ResidualPrediction(
                    quantum_numbers=qn,
                    nu_calc_physical_mhz=float(nu),
                    delta_nu_ml_shift_mhz=0.0,
                    nu_proposed_mhz=float(nu),
                    sigma_predict_mhz=float("inf"),
                    probable_correctness=0.0,
                    provenance=ProvenanceTag.TORQ,
                )
                for qn, nu in zip(qn_list, nu_calc_arr)
            ]

        # GP prediction with uncertainty
        features_arr = np.asarray(features, dtype=np.float64)
        delta_nu_pred, sigma_pred = self._gpr.predict(features_arr, return_std=True)

        predictions: list[ResidualPrediction] = []
        for i, (qn, nu_phys) in enumerate(zip(qn_list, nu_calc_arr)):
            shift = float(delta_nu_pred[i])
            sigma = float(sigma_pred[i])
            nu_proposed = float(nu_phys) + shift

            # Probable correctness: higher confidence when σ is small relative to shift
            # Use a sigmoid-like mapping: confidence = 1 / (1 + σ/reference_scale)
            reference_scale = 0.5  # MHz — typical acceptable uncertainty
            confidence = 1.0 / (1.0 + sigma / reference_scale)

            predictions.append(
                ResidualPrediction(
                    quantum_numbers=qn,
                    nu_calc_physical_mhz=float(nu_phys),
                    delta_nu_ml_shift_mhz=shift,
                    nu_proposed_mhz=nu_proposed,
                    sigma_predict_mhz=sigma,
                    probable_correctness=round(confidence, 4),
                    provenance=ProvenanceTag.ML,
                )
            )

        return predictions

    def get_trained_kernel_params(self) -> dict[str, float]:
        """
        Return the optimized kernel hyperparameters after training.

        Returns
        -------
        dict
            Hyperparameter names and values.
        """
        if not self._is_trained:
            return {}
        return dict(
            zip(
                self._gpr.kernel_.get_params().keys(),
                [
                    float(v) if isinstance(v, (int, float, np.floating)) else v
                    for v in self._gpr.kernel_.get_params().values()
                ],
            )
        )


__all__ = [
    "ProvenanceTag",
    "ResidualPrediction",
    "SpycFitGPRegressor",
]
