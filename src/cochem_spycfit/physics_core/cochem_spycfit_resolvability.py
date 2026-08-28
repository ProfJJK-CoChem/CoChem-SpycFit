"""Resolvability anti-clustering filter for spectral line density penalty scoring."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvabilityScore:
    """Resolvability assessment for a single predicted transition."""

    frequency_mhz: float
    nearest_neighbor_spacing_mhz: float
    instrumental_resolution_mhz: float
    ml_uncertainty_mhz: float
    effective_threshold_mhz: float  # Res_inst + σ_predict
    penalty: float  # 0.0 (unresolvable) to 1.0 (fully resolved)
    is_resolvable: bool  # True if spacing > effective_threshold


class ResolvabilityFilter:
    """Evaluates spectral line density and assigns resolvability penalties."""

    def __init__(
        self,
        instrumental_resolution_mhz: float = 0.050,
        steepness_k: float = 20.0,
    ) -> None:
        """Initialize the ResolvabilityFilter.

        Parameters
        ----------
        instrumental_resolution_mhz : float
            User's FT-MW spectrometer resolution limit in MHz.
            Default: 50 kHz (0.050 MHz), typical for chirped-pulse instruments.
        steepness_k : float
            Sigmoid steepness factor controlling the transition sharpness around the threshold.
            Default: 20.0.
        """
        self._res_inst: float = float(instrumental_resolution_mhz)
        self._steepness_k: float = float(steepness_k)

    @property
    def instrumental_resolution_mhz(self) -> float:
        """Return the instrumental resolution limit in MHz."""
        return self._res_inst

    @property
    def steepness_k(self) -> float:
        """Return the sigmoid steepness factor."""
        return self._steepness_k

    def compute_penalties(
        self,
        predicted_frequencies_mhz: Union[np.ndarray, Sequence[float]],
        ml_uncertainties_mhz: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> list[ResolvabilityScore]:
        """Compute resolvability penalties for all predicted transitions.

        For each transition, the penalty is computed as:
        - effective_threshold = Res_inst + σ_predict
        - nearest_spacing = min distance to any other predicted transition
        - If nearest_spacing < effective_threshold: penalty approaches 0 (steep decay)
        - If nearest_spacing >= effective_threshold: penalty approaches 1.0 (fully resolved)

        The penalty function uses a numerically stable sigmoid steep decay:
        penalty = 1.0 / (1.0 + exp(-k * (spacing - threshold)))

        Parameters
        ----------
        predicted_frequencies_mhz : Union[np.ndarray, Sequence[float]]
            Shape (N,). All predicted transition frequencies in MHz.
        ml_uncertainties_mhz : Optional[Union[np.ndarray, Sequence[float]]], optional
            Shape (N,). GP prediction uncertainty σ_predict for each transition.
            If None, defaults to 0.0 (only instrumental resolution used).

        Returns
        -------
        list[ResolvabilityScore]
            Penalty scores for each transition, preserving input array ordering.
        """
        freqs = np.asarray(predicted_frequencies_mhz, dtype=np.float64).ravel()
        n = len(freqs)
        if n == 0:
            return []

        if ml_uncertainties_mhz is None:
            unc = np.zeros(n, dtype=np.float64)
        else:
            unc = np.asarray(ml_uncertainties_mhz, dtype=np.float64).ravel()
            if len(unc) != n:
                raise ValueError(
                    f"ml_uncertainties_mhz length ({len(unc)}) must match predicted_frequencies_mhz length ({n})"
                )

        # Sort frequencies for efficient nearest-neighbor computation
        sorted_indices = np.argsort(freqs)
        sorted_freqs = freqs[sorted_indices]

        scores: list[ResolvabilityScore] = [None] * n  # type: ignore[list-item]

        for rank, idx in enumerate(sorted_indices):
            freq = float(sorted_freqs[rank])
            sigma = float(unc[idx])
            threshold = self._res_inst + sigma

            # Find nearest neighbor spacing (left and right in sorted array)
            spacings = []
            if rank > 0:
                spacings.append(freq - float(sorted_freqs[rank - 1]))
            if rank < n - 1:
                spacings.append(float(sorted_freqs[rank + 1]) - freq)

            if spacings:
                nearest = float(min(spacings))
            else:
                nearest = float("inf")

            # Numerically stable sigmoid penalty
            if math.isinf(nearest):
                penalty = 1.0
            else:
                exponent = -self._steepness_k * (nearest - threshold)
                if exponent > 700.0:
                    penalty = 0.0
                elif exponent < -700.0:
                    penalty = 1.0
                else:
                    penalty = 1.0 / (1.0 + float(np.exp(exponent)))
            penalty = float(np.clip(penalty, 0.0, 1.0))

            scores[idx] = ResolvabilityScore(
                frequency_mhz=freq,
                nearest_neighbor_spacing_mhz=nearest,
                instrumental_resolution_mhz=self._res_inst,
                ml_uncertainty_mhz=sigma,
                effective_threshold_mhz=threshold,
                penalty=penalty,
                is_resolvable=bool(nearest > threshold),
            )

        return scores


__all__ = [
    "ResolvabilityFilter",
    "ResolvabilityScore",
]
