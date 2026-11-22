"""Constant-level comparison between native SpycFit fits and Pickett SPFIT fits.

This module compares two already-computed parameter sets: one from the native SpycFit
solver and one parsed from a genuine SPFIT run. It reports, per shared parameter, the
absolute difference against a tolerance the caller declares, and it reports which
parameters could not be compared.

Status vocabulary follows SF-CMP-012 (docs/wiki/CoChem-SpycFit_SRS_2026-10-16_Amendment.md):
an empty or partial comparison is INCOMPLETE, never WITHIN_TOLERANCE. No tolerance is
built in; SF-CMP-008 requires tolerances declared per capability and observable.

Comparing constants is meaningful only when both fits used the same model, reduction,
representation, sign convention and parameter set (SF-CMP-002, SF-CMP-005). This module
does not check those conditions; callers record them alongside the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. COMPARISON STATUS & DATA STRUCTURES
# =============================================================================

@enum.unique
class ParityStatus(str, enum.Enum):
    """Comparison outcome, using the SF-CMP-012 vocabulary."""

    NOT_RUN = "NOT_RUN"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    INCOMPATIBLE_MODEL = "INCOMPATIBLE_MODEL"
    INCOMPLETE = "INCOMPLETE"
    DIVERGENT = "DIVERGENT"
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"

    def __str__(self) -> str:
        return self.value


ToleranceSpec = Union[float, Mapping[str, float]]


@dataclass(frozen=True)
class ParityWarning:
    """Immutable report of a constant-level comparison between native SpycFit and SPFIT.

    Attributes:
        status: Comparison outcome (SF-CMP-012 vocabulary).
        max_delta_khz: Largest absolute difference among compared constants in kHz, or
            None when no constant was compared.
        divergent_constants: Compared constants whose difference exceeds their tolerance.
        contrast_table: Per-constant values, differences, tolerances and uncertainties.
        jax_uncertainties: Native parameter standard uncertainties (MHz) as supplied.
        spfit_uncertainties: SPFIT parameter standard uncertainties (MHz) as supplied.
        compared_constants: Constants present and finite in both sets.
        missing_from_native: Required constants absent or non-finite in the native set.
        missing_from_spfit: Required constants absent or non-finite in the SPFIT set.
        tolerance_khz: Tolerance applied to each compared constant, in kHz.
    """

    status: ParityStatus
    max_delta_khz: Optional[float]
    divergent_constants: Tuple[str, ...] = ()
    contrast_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    jax_uncertainties: Optional[Dict[str, float]] = None
    spfit_uncertainties: Optional[Dict[str, float]] = None
    compared_constants: Tuple[str, ...] = ()
    missing_from_native: Tuple[str, ...] = ()
    missing_from_spfit: Tuple[str, ...] = ()
    tolerance_khz: Dict[str, float] = field(default_factory=dict)

    @property
    def is_within_tolerance(self) -> bool:
        """Return True only for a complete comparison with every constant within tolerance."""
        return self.status == ParityStatus.WITHIN_TOLERANCE

    @property
    def has_divergence(self) -> bool:
        """Return True if one or more compared constants exceeded their tolerance."""
        return len(self.divergent_constants) > 0

    @property
    def is_complete(self) -> bool:
        """Return True if every required constant was compared."""
        return (
            len(self.compared_constants) > 0
            and not self.missing_from_native
            and not self.missing_from_spfit
        )

    def to_dataframe(self) -> Any:
        """Format contrast table as a pandas DataFrame."""
        return ParityAuditor.format_contrast_dataframe(self)


def _finite_value(mapping: Mapping[str, Any], key: str) -> Optional[float]:
    """Return mapping[key] as a float if present and finite, else None."""
    if key not in mapping or mapping[key] is None:
        return None
    value = float(np.float64(mapping[key]))
    return value if math.isfinite(value) else None


# =============================================================================
# 2. COMPARISON CORE
# =============================================================================

class ParityAuditor:
    """Compare native SpycFit constants with SPFIT constants under declared tolerances."""

    def __init__(self, default_tolerance_khz: Optional[ToleranceSpec] = None) -> None:
        """Initialize the stateless ParityAuditor.

        Args:
            default_tolerance_khz: Tolerance used when ``audit`` receives none. Either one
                value in kHz for every constant or a mapping from constant name to kHz.
                There is no built-in value (SF-CMP-008).
        """
        self.default_tolerance_khz: Optional[ToleranceSpec] = default_tolerance_khz

    @staticmethod
    def _resolve_tolerances(spec: Optional[ToleranceSpec], keys: Sequence[str]) -> Dict[str, float]:
        """Return a per-constant tolerance in kHz for every key, or raise ValueError."""
        if spec is None:
            raise ValueError(
                "No comparison tolerance was declared. SF-CMP-008 requires a tolerance per "
                "capability and observable; pass tolerance_khz or default_tolerance_khz."
            )
        if isinstance(spec, Mapping):
            missing = [k for k in keys if k not in spec]
            if missing:
                raise ValueError(f"No tolerance declared for constants: {', '.join(missing)}")
            resolved = {k: float(spec[k]) for k in keys}
        else:
            resolved = {k: float(spec) for k in keys}
        for key, tol in resolved.items():
            if not math.isfinite(tol) or tol < 0.0:
                raise ValueError(f"Tolerance for {key} must be finite and non-negative, got {tol}")
        return resolved

    def audit(
        self,
        jax_constants: Mapping[str, float],
        spfit_constants: Optional[Mapping[str, float]] = None,
        jax_uncertainties: Optional[Mapping[str, float]] = None,
        spfit_uncertainties: Optional[Mapping[str, float]] = None,
        tolerance_khz: Optional[ToleranceSpec] = None,
        required_constants: Optional[Sequence[str]] = None,
    ) -> ParityWarning:
        """Compare native constants with SPFIT constants.

        Args:
            jax_constants: Native SpycFit constants (MHz).
            spfit_constants: Constants parsed from a genuine SPFIT run (MHz), or None when
                SPFIT ran and produced no usable constants.
            jax_uncertainties: Optional native parameter standard uncertainties (MHz).
            spfit_uncertainties: Optional SPFIT parameter standard uncertainties (MHz).
            tolerance_khz: Tolerance in kHz, one value or one per constant. Overrides the
                instance default.
            required_constants: Constants the comparison must cover. Defaults to the union
                of both sets, so a constant present in only one set makes the comparison
                INCOMPLETE.

        Returns:
            ParityWarning describing the outcome, coverage and per-constant differences.
        """
        jax_unc = dict(jax_uncertainties) if jax_uncertainties is not None else None
        spfit_unc = dict(spfit_uncertainties) if spfit_uncertainties is not None else None

        if spfit_constants is None:
            logger.warning("SPFIT constants are None: comparison FAILED.")
            return ParityWarning(
                status=ParityStatus.FAILED,
                max_delta_khz=None,
                divergent_constants=(),
                contrast_table={},
                jax_uncertainties=jax_unc,
                spfit_uncertainties=None,
                compared_constants=(),
                missing_from_native=(),
                missing_from_spfit=tuple(jax_constants.keys()),
            )

        if required_constants is None:
            required = list(dict.fromkeys(list(jax_constants.keys()) + list(spfit_constants.keys())))
        else:
            required = list(dict.fromkeys(required_constants))

        missing_native = tuple(k for k in required if _finite_value(jax_constants, k) is None)
        missing_spfit = tuple(k for k in required if _finite_value(spfit_constants, k) is None)
        compared = [k for k in required if k not in missing_native and k not in missing_spfit]

        tolerances = self._resolve_tolerances(
            tolerance_khz if tolerance_khz is not None else self.default_tolerance_khz,
            compared,
        )

        contrast_table: Dict[str, Dict[str, Any]] = {}
        divergent_list: List[str] = []
        max_delta: Optional[float] = None

        for key in compared:
            val_jax = _finite_value(jax_constants, key)
            val_spfit = _finite_value(spfit_constants, key)
            assert val_jax is not None and val_spfit is not None
            tol = tolerances[key]

            delta_khz = abs(val_jax - val_spfit) * 1000.0
            is_divergent = delta_khz > tol and not math.isclose(delta_khz, tol, rel_tol=1e-7, abs_tol=1e-9)
            if is_divergent:
                divergent_list.append(key)
            max_delta = delta_khz if max_delta is None else max(max_delta, delta_khz)

            unc_native = _finite_value(jax_unc, key) if jax_unc is not None else None
            unc_spfit = _finite_value(spfit_unc, key) if spfit_unc is not None else None

            contrast_table[key] = {
                "jax_mhz": val_jax,
                "spfit_mhz": val_spfit,
                "delta_khz": delta_khz,
                "tolerance_khz": tol,
                "status": "⚠️" if is_divergent else "✅",
                "is_divergent": is_divergent,
                "jax_unc": unc_native,
                "spfit_unc": unc_spfit,
                "jax_unc_mhz": unc_native,
                "spfit_unc_mhz": unc_spfit,
            }

        complete = len(compared) > 0 and not missing_native and not missing_spfit
        if divergent_list:
            status = ParityStatus.DIVERGENT
        elif not complete:
            status = ParityStatus.INCOMPLETE
        else:
            status = ParityStatus.WITHIN_TOLERANCE

        if not compared:
            logger.warning("No constant could be compared between the native and SPFIT sets.")

        return ParityWarning(
            status=status,
            max_delta_khz=max_delta,
            divergent_constants=tuple(divergent_list),
            contrast_table=contrast_table,
            jax_uncertainties=jax_unc,
            spfit_uncertainties=spfit_unc,
            compared_constants=tuple(compared),
            missing_from_native=missing_native,
            missing_from_spfit=missing_spfit,
            tolerance_khz=tolerances,
        )

    @staticmethod
    def format_contrast_dataframe(warning: ParityWarning) -> Any:
        """Format contrast table from a ParityWarning into a presentation pandas DataFrame.

        Deferred pandas import to preserve zero-overhead CLI startup.

        Args:
            warning: ParityWarning object to format.

        Returns:
            pandas.DataFrame with columns ['Constant', 'JAX (MHz)', 'SPFIT (MHz)', 'Δ (kHz)', 'Status'].
        """
        import pandas as pd  # type: ignore[import-untyped]

        columns = ["Constant", "JAX (MHz)", "SPFIT (MHz)", "Δ (kHz)", "Status"]
        if not warning.contrast_table:
            return pd.DataFrame(columns=columns)

        rows = []
        for const_name, data in warning.contrast_table.items():
            jax_val = data.get("jax_mhz", np.nan)
            spfit_val = data.get("spfit_mhz", np.nan)
            delta_khz = data.get("delta_khz", np.nan)
            status_emoji = data.get("status", "⚠️" if const_name in warning.divergent_constants else "✅")
            if status_emoji not in ("✅", "⚠️"):
                status_emoji = "⚠️" if const_name in warning.divergent_constants else "✅"

            rows.append({
                "Constant": const_name,
                "JAX (MHz)": jax_val,
                "SPFIT (MHz)": spfit_val,
                "Δ (kHz)": delta_khz,
                "Status": status_emoji,
            })

        return pd.DataFrame(rows, columns=columns)


format_contrast_dataframe = ParityAuditor.format_contrast_dataframe

__all__ = [
    "ParityAuditor",
    "ParityStatus",
    "ParityWarning",
    "format_contrast_dataframe",
]
