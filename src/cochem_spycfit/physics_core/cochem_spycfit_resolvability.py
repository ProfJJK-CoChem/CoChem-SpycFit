"""Resolvability anti-clustering filter for spectral line density penalty scoring."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import math
from pathlib import Path
import sys
from typing import Any, Optional, Sequence, TYPE_CHECKING, Union

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
        cached = sys.modules.get("_cochem_spycfit_instruments_profiles")
        if cached is None:
            profiles_path = Path(__file__).resolve().parent.parent / "instruments" / "profiles.py"
            spec = importlib.util.spec_from_file_location("_cochem_spycfit_instruments_profiles", profiles_path)
            if spec is None or spec.loader is None:
                raise
            cached = importlib.util.module_from_spec(spec)
            sys.modules["_cochem_spycfit_instruments_profiles"] = cached
            spec.loader.exec_module(cached)
        return cached.get_instrument_profile(profile_id)


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
        instrumental_resolution_mhz: Optional[float] = None,
        steepness_k: float = 20.0,
        instrument_profile: Optional[str] = None,
    ) -> None:
        """Initialize the ResolvabilityFilter.

        Parameters
        ----------
        instrumental_resolution_mhz : float, optional
            Spectrometer resolution limit in MHz. When omitted and
            ``instrument_profile`` is provided, defaults to the active profile's
            ``instrumental_resolution_mhz`` (``0.100`` MHz for ``grulab_surf``).
            When both are omitted, preserves the historical ``0.050`` MHz default.
        steepness_k : float
            Sigmoid steepness factor controlling transition sharpness around threshold.
        instrument_profile : str, optional
            Registered instrument profile identifier (e.g. ``"grulab_surf"``,
            ``"generic_cp_ftmw"``, ``"generic_ir_raman"``).
        """
        self._profile: Optional[InstrumentProfile] = (
            _get_instrument_profile(instrument_profile)
            if instrument_profile is not None
            else None
        )
        if instrumental_resolution_mhz is not None:
            resolved_res = float(instrumental_resolution_mhz)
            if not math.isfinite(resolved_res) or resolved_res <= 0.0:
                raise ValueError(
                    f"instrumental_resolution_mhz must be finite and > 0, got {instrumental_resolution_mhz!r}."
                )
            self._res_inst: float = resolved_res
        elif self._profile is not None:
            self._res_inst = float(self._profile.instrumental_resolution_mhz)
        else:
            self._res_inst = 0.050
        self._steepness_k: float = float(steepness_k)

    @classmethod
    def from_instrument_profile(
        cls,
        instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE,
        *,
        resolution_khz: Optional[float] = None,
        steepness_k: float = 20.0,
    ) -> ResolvabilityFilter:
        """Build a ``ResolvabilityFilter`` bound to a registered instrument profile.

        Defaults to ``grulab_surf`` (``0.100`` MHz = ``100`` kHz RBW), or uses an
        explicitly overridden digitizer RBW in kHz (e.g. ``10`` kHz -> ``0.010`` MHz).
        """
        override_mhz: Optional[float] = None
        if resolution_khz is not None:
            rbw = float(resolution_khz)
            if not math.isfinite(rbw) or rbw <= 0.0:
                raise ValueError(f"resolution_khz must be finite and > 0, got {resolution_khz!r}.")
            override_mhz = rbw / 1000.0
        return cls(
            instrumental_resolution_mhz=override_mhz,
            steepness_k=steepness_k,
            instrument_profile=instrument_profile,
        )

    @property
    def instrument_profile(self) -> Optional[str]:
        """Return the bound instrument profile ID, or ``None`` if unbound."""
        return self._profile.profile_id if self._profile is not None else None

    @property
    def profile(self) -> Optional[InstrumentProfile]:
        """Return the bound frozen ``InstrumentProfile``, or ``None`` if unbound."""
        return self._profile

    @property
    def instrumental_resolution_mhz(self) -> float:
        """Return the instrumental resolution limit in MHz."""
        return self._res_inst

    @property
    def steepness_k(self) -> float:
        """Return the sigmoid steepness factor."""
        return self._steepness_k

    def update_resolution_khz(self, resolution_khz: float) -> float:
        """Update active instrumental resolution from a digitizer RBW override in kHz."""
        rbw = float(resolution_khz)
        if not math.isfinite(rbw) or rbw <= 0.0:
            raise ValueError(f"resolution_khz must be finite and > 0, got {resolution_khz!r}.")
        self._res_inst = rbw / 1000.0
        return self._res_inst

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
