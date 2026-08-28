"""Smart Scan Navigator — Information-Gain triage and hardware-aware chunking."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRecommendation:
    """A recommended FT-MW scan region."""

    rank: int
    start_frequency_mhz: float
    end_frequency_mhz: float
    chunk_score: float
    n_clean_lines: int
    targeted_constants: tuple[str, ...]
    description: str  # Human-readable recommendation string


class SmartScanNavigator:
    """Information-Gain engine for strategic FT-MW scan recommendations."""

    def __init__(
        self,
        max_chunk_bandwidth_mhz: float = 250.0,
        top_n_recommendations: int = 3,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_chunk_bandwidth_mhz : float
            Maximum scan chunk bandwidth in MHz. Default 250 MHz.
            Can be overridden by 'max_chunk_bandwidth' in cochem_system_config.json.
        top_n_recommendations : int
            Number of top-ranked scan regions to return.
        config_path : Path, optional
            Path to cochem_system_config.json for overrides.
        """
        self._chunk_bw: float = float(max_chunk_bandwidth_mhz)
        self._top_n: int = int(top_n_recommendations)

        # Override from config if available
        resolved_config = self._resolve_config_path(config_path)
        if resolved_config and resolved_config.exists():
            with open(resolved_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "max_chunk_bandwidth" in config:
                self._chunk_bw = float(config["max_chunk_bandwidth"])

    def identify_target_constants(
        self,
        covariance_matrix: np.ndarray,
        parameter_names: list[str],
        n_targets: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Identify the physical constants with highest variance or correlation.

        Scans the diagonal (parameter uncertainty) and off-diagonal
        (parameter collinearity) elements of the covariance matrix.

        Parameters
        ----------
        covariance_matrix : np.ndarray
            Shape (N_params, N_params). Symmetric covariance matrix from JAX fit.
        parameter_names : list[str]
            Names corresponding to matrix indices (e.g., ['A', 'B', 'C', 'DJ', ...]).
        n_targets : int
            Number of top-priority constants to flag.

        Returns
        -------
        list[tuple[str, float]]
            List of (parameter_name, variance) sorted by descending variance.
        """
        variances = np.diag(covariance_matrix)
        sorted_indices = np.argsort(variances)[::-1]
        targets = [
            (parameter_names[i], float(variances[i]))
            for i in sorted_indices[:n_targets]
        ]
        logger.info(f"Target constants (highest variance): {targets}")
        return targets

    def compute_information_gain(
        self,
        jacobian: np.ndarray,
        parameter_names: list[str],
        target_constants: list[tuple[str, float]],
        resolvability_penalties: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-transition Information Gain weighted by resolvability.

        Information Gain for transition i:
          IG_i = Σ_j (∂ν_i/∂p_j)² * w_j * resolvability_penalty_i
        where w_j = 1 if p_j is a target constant, 0 otherwise.

        Parameters
        ----------
        jacobian : np.ndarray
            Shape (N_transitions, N_params). Analytical Jacobian from JAX driver.
        parameter_names : list[str]
            Parameter names matching Jacobian columns.
        target_constants : list[tuple[str, float]]
            Target constants from identify_target_constants().
        resolvability_penalties : np.ndarray
            Shape (N_transitions,). Penalty scores from ResolvabilityFilter.

        Returns
        -------
        np.ndarray
            Shape (N_transitions,). Per-transition Information Gain scores.
        """
        target_names = {name for name, _ in target_constants}
        target_mask = np.array([
            1.0 if name in target_names else 0.0
            for name in parameter_names
        ], dtype=np.float64)

        # Weighted sum of squared Jacobian entries for target constants
        sensitivity = np.sum((jacobian ** 2) * target_mask[np.newaxis, :], axis=1)
        return sensitivity * resolvability_penalties

    def recommend_scan_regions(
        self,
        transition_frequencies_mhz: np.ndarray,
        information_gains: np.ndarray,
        resolvability_penalties: np.ndarray,
        target_constants: list[tuple[str, float]],
    ) -> list[ScanRecommendation]:
        """
        Slide a bandwidth window across the spectrum and rank regions.

        Chunk Score = Σ(Information Gain of clean lines) within the window,
        where "clean" means resolvability_penalty > 0.5.

        Parameters
        ----------
        transition_frequencies_mhz : np.ndarray
            Shape (N,). Predicted frequencies of all unassigned transitions.
        information_gains : np.ndarray
            Shape (N,). Per-transition IG scores.
        resolvability_penalties : np.ndarray
            Shape (N,). Penalty scores from ResolvabilityFilter.
        target_constants : list[tuple[str, float]]
            Target constants for description generation.

        Returns
        -------
        list[ScanRecommendation]
            Top-N ranked scan region recommendations.
        """
        if len(transition_frequencies_mhz) == 0:
            return []

        freq_min = float(np.min(transition_frequencies_mhz))
        freq_max = float(np.max(transition_frequencies_mhz))

        chunk_starts = np.arange(freq_min, freq_max, self._chunk_bw / 2.0)
        if len(chunk_starts) == 0:
            chunk_starts = np.array([freq_min])

        chunk_results = []

        for start in chunk_starts:
            end = start + self._chunk_bw
            mask = (transition_frequencies_mhz >= start) & (transition_frequencies_mhz < end)
            if not np.any(mask):
                continue

            clean_mask = mask & (resolvability_penalties > 0.5)
            chunk_score = float(np.sum(information_gains[clean_mask]))
            n_clean = int(np.sum(clean_mask))

            if chunk_score > 0 and n_clean > 0:
                chunk_results.append((start, end, chunk_score, n_clean))

        # Sort by chunk score descending
        chunk_results.sort(key=lambda x: x[2], reverse=True)

        target_names = tuple(name for name, _ in target_constants)
        recommendations = []
        for rank, (start, end, score, n_clean) in enumerate(chunk_results[:self._top_n], 1):
            desc = (
                f"Recommendation {rank}: Scan {start:.0f}–{end:.0f} MHz. "
                f"Expected {n_clean} isolated transitions. "
                f"Targets: {', '.join(target_names)}."
            )
            recommendations.append(ScanRecommendation(
                rank=rank,
                start_frequency_mhz=start,
                end_frequency_mhz=end,
                chunk_score=score,
                n_clean_lines=n_clean,
                targeted_constants=target_names,
                description=desc,
            ))

        return recommendations

    @staticmethod
    def _resolve_config_path(user_path: Optional[Path]) -> Optional[Path]:
        """Resolve cochem_system_config.json path."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "Registry" / "cochem_system_config.json"
        return Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"


__all__ = [
    "ScanRecommendation",
    "SmartScanNavigator",
]
