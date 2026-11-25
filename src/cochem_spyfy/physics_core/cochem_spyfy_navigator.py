"""Smart Scan Navigator — Information-Gain triage and hardware-aware chunking."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..instruments.profiles import InstrumentProfile

DEFAULT_INSTRUMENT_PROFILE: str = "grulab_surf"

logger = logging.getLogger(__name__)


def _get_instrument_profile(profile_id: Optional[str] = None) -> Any:
    """Resolve an ``InstrumentProfile`` across both ``src``-layout and repo-root imports."""
    try:
        from ..instruments.profiles import get_instrument_profile

        return get_instrument_profile(profile_id)
    except ModuleNotFoundError:
        cached = sys.modules.get("_cochem_spyfy_instruments_profiles")
        if cached is None:
            profiles_path = Path(__file__).resolve().parent.parent / "instruments" / "profiles.py"
            spec = importlib.util.spec_from_file_location("_cochem_spyfy_instruments_profiles", profiles_path)
            if spec is None or spec.loader is None:
                raise
            cached = importlib.util.module_from_spec(spec)
            sys.modules["_cochem_spyfy_instruments_profiles"] = cached
            spec.loader.exec_module(cached)
        return cached.get_instrument_profile(profile_id)


def _compute_surf_scan_minutes(
    start_freq_mhz: float,
    end_freq_mhz: float,
    step_size_mhz: float,
    acq_num: int,
    rep_rate_hz: int,
    retune_overhead_s: float = 14.0,
) -> float:
    """Compute exact SURF scan duration in minutes matching ``spectrometer.py:168-181``."""
    iterations = math.ceil(abs(float(end_freq_mhz) - float(start_freq_mhz)) / float(step_size_mhz)) + 1
    time_delay_s = float(acq_num) / float(rep_rate_hz)
    return (float(iterations) * (time_delay_s + float(retune_overhead_s))) / 60.0


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
    instrument_profile: Optional[str] = None
    surf_estimated_duration_min: Optional[float] = None
    recommended_step_size_mhz: Optional[float] = None
    window_mode: str = "survey"


class SmartScanNavigator:
    """Information-Gain engine for strategic FT-MW scan recommendations."""

    def __init__(
        self,
        max_chunk_bandwidth_mhz: Optional[float] = None,
        top_n_recommendations: int = 3,
        config_path: Optional[Path] = None,
        instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE,
        surf_step_size_mhz: Optional[float] = None,
        surf_acq_num: int = 100,
        surf_rep_rate_hz: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        max_chunk_bandwidth_mhz : float, optional
            Maximum scan chunk bandwidth in MHz. Default 250 MHz.
            Can be overridden by 'max_chunk_bandwidth' in cochem_system_config.json.
        top_n_recommendations : int
            Number of top-ranked scan regions to return.
        config_path : Path, optional
            Path to cochem_system_config.json for overrides.
        instrument_profile : str
            Active instrument profile ID. Defaults to ``"grulab_surf"``.
        surf_step_size_mhz : float, optional
            Frequency step size (MHz) used for SURF scan-time estimation.
        surf_acq_num : int
            Default acquisitions per step used for SURF scan-time estimation.
        surf_rep_rate_hz : int
            Default valve repetition rate (Hz) used for SURF scan-time estimation.
        """
        self._top_n: int = int(top_n_recommendations)
        self._profile: InstrumentProfile = _get_instrument_profile(instrument_profile)
        self._instrument_profile_id: str = self._profile.profile_id
        self._surf_step_size_mhz: float = (
            float(surf_step_size_mhz)
            if surf_step_size_mhz is not None
            else float(self._profile.default_step_size_mhz)
        )
        self._surf_acq_num: int = int(surf_acq_num)
        self._surf_rep_rate_hz: int = int(surf_rep_rate_hz)

        if max_chunk_bandwidth_mhz is not None:
            self._chunk_bw = float(max_chunk_bandwidth_mhz)
        else:
            self._chunk_bw = 250.0
            resolved_config = self._resolve_config_path(config_path)
            if resolved_config and resolved_config.exists():
                with open(resolved_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if "max_chunk_bandwidth" in config and config["max_chunk_bandwidth"]:
                    self._chunk_bw = float(config["max_chunk_bandwidth"])

        if config_path is not None:
            resolved_config = self._resolve_config_path(config_path)
            if resolved_config and resolved_config.exists():
                with open(resolved_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if "max_chunk_bandwidth" in config and config["max_chunk_bandwidth"]:
                    self._chunk_bw = float(config["max_chunk_bandwidth"])

    @property
    def instrument_profile(self) -> str:
        """Return active instrument profile identifier."""
        return self._instrument_profile_id

    @property
    def profile(self) -> InstrumentProfile:
        """Return active frozen ``InstrumentProfile``."""
        return self._profile

    @property
    def chunk_bandwidth_mhz(self) -> float:
        """Return configured chunk bandwidth in MHz."""
        return self._chunk_bw

    def _estimate_surf_duration(
        self,
        start_mhz: float,
        end_mhz: float,
        *,
        step_size_mhz: Optional[float] = None,
    ) -> Optional[float]:
        """Return SURF duration in minutes when ``grulab_surf`` is active, else ``None``."""
        if self._instrument_profile_id != "grulab_surf":
            return None
        step = float(step_size_mhz) if step_size_mhz is not None else self._surf_step_size_mhz
        return _compute_surf_scan_minutes(
            start_freq_mhz=start_mhz,
            end_freq_mhz=end_mhz,
            step_size_mhz=step,
            acq_num=self._surf_acq_num,
            rep_rate_hz=self._surf_rep_rate_hz,
            retune_overhead_s=self._profile.zaber_retune_overhead_s,
        )

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
        Slide a bandwidth window across the active instrument envelope and rank regions.

        Proposals are clamped strictly to the active profile's
        ``[freq_min_mhz, freq_max_mhz]`` hardware range. When ``grulab_surf``
        is active, each ``ScanRecommendation`` includes ``surf_estimated_duration_min``
        and profile metadata.
        """
        freqs = np.asarray(transition_frequencies_mhz, dtype=np.float64).ravel()
        ig_arr = np.asarray(information_gains, dtype=np.float64).ravel()
        pen_arr = np.asarray(resolvability_penalties, dtype=np.float64).ravel()

        if len(freqs) == 0:
            return []

        prof_min = float(self._profile.freq_min_mhz)
        prof_max = float(self._profile.freq_max_mhz)

        in_band_mask = (freqs >= prof_min) & (freqs <= prof_max)
        if not np.any(in_band_mask):
            return []

        in_freqs = freqs[in_band_mask]
        in_ig = ig_arr[in_band_mask]
        in_pen = pen_arr[in_band_mask]

        freq_min = max(float(np.min(in_freqs)), prof_min)
        freq_max = min(float(np.max(in_freqs)), prof_max)

        chunk_starts = np.arange(freq_min, freq_max, self._chunk_bw / 2.0)
        if len(chunk_starts) == 0:
            chunk_starts = np.array([freq_min])

        seen_windows: set[tuple[float, float]] = set()
        chunk_results: list[tuple[float, float, float, int]] = []

        for raw_start in chunk_starts:
            start = max(float(raw_start), prof_min)
            end = min(start + self._chunk_bw, prof_max)
            if end <= start:
                start = max(prof_min, end - min(self._chunk_bw, prof_max - prof_min))
            if end <= start:
                continue

            window_key = (round(start, 6), round(end, 6))
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)

            if end >= prof_max:
                mask = (in_freqs >= start) & (in_freqs <= end)
            else:
                mask = (in_freqs >= start) & (in_freqs < end)
            if not np.any(mask):
                continue

            clean_mask = mask & (in_pen > 0.5)
            chunk_score = float(np.sum(in_ig[clean_mask]))
            n_clean = int(np.sum(clean_mask))

            if chunk_score > 0 and n_clean > 0:
                chunk_results.append((start, end, chunk_score, n_clean))

        # Sort by chunk score descending
        chunk_results.sort(key=lambda x: x[2], reverse=True)

        target_names = tuple(name for name, _ in target_constants)
        recommendations: list[ScanRecommendation] = []
        for rank, (start, end, score, n_clean) in enumerate(chunk_results[:self._top_n], 1):
            surf_duration = self._estimate_surf_duration(start, end)
            desc = (
                f"Recommendation {rank}: Scan {start:.0f}–{end:.0f} MHz. "
                f"Expected {n_clean} isolated transitions. "
                f"Targets: {', '.join(target_names)}."
            )
            if surf_duration is not None:
                desc = f"{desc} Est. SURF runtime: {surf_duration:.1f} min."
            recommendations.append(ScanRecommendation(
                rank=rank,
                start_frequency_mhz=start,
                end_frequency_mhz=end,
                chunk_score=score,
                n_clean_lines=n_clean,
                targeted_constants=target_names,
                description=desc,
                instrument_profile=self._instrument_profile_id,
                surf_estimated_duration_min=surf_duration,
                recommended_step_size_mhz=self._surf_step_size_mhz,
                window_mode="survey",
            ))

        return recommendations

    def recommend_targeted_windows(
        self,
        transition_frequencies_mhz: Sequence[float] | np.ndarray,
        information_gains: Sequence[float] | np.ndarray,
        resolvability_penalties: Sequence[float] | np.ndarray,
        target_constants: list[tuple[str, float]],
        *,
        ml_uncertainties_mhz: Optional[Sequence[float] | np.ndarray] = None,
        doppler_margin_mhz: float = 0.0,
        min_half_width_mhz: float = 0.5,
        sigma_multiplier: float = 3.0,
        step_size_mhz: float = 0.250,
    ) -> list[ScanRecommendation]:
        """Recommend narrow targeted confirmation windows clamped to active profile bounds."""
        freqs = np.asarray(transition_frequencies_mhz, dtype=np.float64).ravel()
        ig_arr = np.asarray(information_gains, dtype=np.float64).ravel()
        pen_arr = np.asarray(resolvability_penalties, dtype=np.float64).ravel()
        if len(freqs) == 0:
            return []

        if ml_uncertainties_mhz is None:
            unc_arr = np.zeros(len(freqs), dtype=np.float64)
        else:
            unc_arr = np.asarray(ml_uncertainties_mhz, dtype=np.float64).ravel()

        prof_min = float(self._profile.freq_min_mhz)
        prof_max = float(self._profile.freq_max_mhz)
        target_names = tuple(name for name, _ in target_constants)

        candidates: list[tuple[float, float, float, float, Optional[float]]] = []
        for idx, nu_raw in enumerate(freqs):
            nu = float(nu_raw)
            if not (prof_min <= nu <= prof_max):
                continue
            if float(pen_arr[idx]) <= 0.5 or float(ig_arr[idx]) <= 0.0:
                continue

            sigma = float(unc_arr[idx])
            if math.isfinite(sigma) and sigma >= 0.0:
                half_width = max(float(sigma_multiplier) * sigma, float(min_half_width_mhz)) + float(doppler_margin_mhz)
            else:
                half_width = max(1.0, float(min_half_width_mhz)) + float(doppler_margin_mhz)

            start = round(max(prof_min, nu - half_width), 3)
            end = round(min(prof_max, nu + half_width), 3)
            if end <= start:
                end = round(min(prof_max, start + max(float(step_size_mhz), 0.5)), 3)
                if end <= start:
                    start = round(max(prof_min, end - max(float(step_size_mhz), 0.5)), 3)

            surf_duration = self._estimate_surf_duration(start, end, step_size_mhz=step_size_mhz)
            rank_metric = (
                float(ig_arr[idx]) / surf_duration
                if surf_duration is not None and surf_duration > 0.0
                else float(ig_arr[idx])
            )
            candidates.append((rank_metric, float(ig_arr[idx]), start, end, surf_duration))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        recommendations: list[ScanRecommendation] = []
        for rank, (_, score, start, end, surf_duration) in enumerate(candidates[:self._top_n], 1):
            desc = (
                f"Targeted Recommendation {rank}: Scan {start:.3f}–{end:.3f} MHz. "
                f"Targets: {', '.join(target_names)}."
            )
            if surf_duration is not None:
                desc = f"{desc} Est. SURF runtime: {surf_duration:.2f} min."
            recommendations.append(ScanRecommendation(
                rank=rank,
                start_frequency_mhz=start,
                end_frequency_mhz=end,
                chunk_score=score,
                n_clean_lines=1,
                targeted_constants=target_names,
                description=desc,
                instrument_profile=self._instrument_profile_id,
                surf_estimated_duration_min=surf_duration,
                recommended_step_size_mhz=float(step_size_mhz),
                window_mode="targeted",
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
