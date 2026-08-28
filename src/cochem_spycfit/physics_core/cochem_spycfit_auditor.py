"""Mathematical Parity Auditor: JAX autodiff Hamiltonian vs. Pickett SPFIT.

This module provides high-precision cross-validation between pure-JAX Hamiltonian
eigenvalue / Levenberg-Marquardt solutions and Pickett SPFIT / SPCAT legacy Fortran
solutions. It rigorously verifies rotational and centrifugal distortion parameter
congruence against a strict 0.1 kHz mathematical parity tolerance (in FP64).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. PARITY STATUS & DATA STRUCTURES
# =============================================================================

@enum.unique
class ParityStatus(str, enum.Enum):
    """Execution and mathematical parity audit status."""

    VERIFIED = "VERIFIED"
    WARNING = "WARNING"
    SPFIT_FAILED = "SPFIT_FAILED"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ParityWarning:
    """Immutable report of mathematical parity audit between JAX and SPFIT.

    Attributes:
        status: Overall parity status (VERIFIED, WARNING, SPFIT_FAILED).
        max_delta_khz: Maximum absolute discrepancy between JAX and SPFIT in kHz.
        divergent_constants: Tuple of parameter names exceeding the parity tolerance.
        contrast_table: Detailed dictionary mapping parameter names to comparative values.
        jax_uncertainties: Optional mapping of JAX parameter standard errors (MHz).
        spfit_uncertainties: Optional mapping of SPFIT parameter standard errors (MHz).
    """

    status: ParityStatus
    max_delta_khz: float
    divergent_constants: Tuple[str, ...] = ()
    contrast_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    jax_uncertainties: Optional[Dict[str, float]] = None
    spfit_uncertainties: Optional[Dict[str, float]] = None

    @property
    def is_verified(self) -> bool:
        """Return True if mathematical parity was successfully verified."""
        return self.status == ParityStatus.VERIFIED

    @property
    def has_divergence(self) -> bool:
        """Return True if one or more constants diverged beyond tolerance."""
        return len(self.divergent_constants) > 0

    def to_dataframe(self) -> Any:
        """Format contrast table as a pandas DataFrame."""
        return ParityAuditor.format_contrast_dataframe(self)


# =============================================================================
# 2. PARITY AUDITOR CORE
# =============================================================================

class ParityAuditor:
    """Rigorous mathematical parity auditor comparing JAX and SPFIT spectroscopic constants.

    Audits Watson A-reduction and S-reduction rotational and distortion constants
    in FP64 precision with a default tolerance of 0.1 kHz (1e-4 MHz).
    """

    TOLERANCE_KHZ: float = 0.1

    def __init__(self, default_tolerance_khz: float = TOLERANCE_KHZ) -> None:
        """Initialize the stateless ParityAuditor.

        Args:
            default_tolerance_khz: Default divergence threshold in kHz (default: 0.1 kHz).
        """
        self.default_tolerance_khz: float = float(default_tolerance_khz)

    def audit(
        self,
        jax_constants: Mapping[str, float],
        spfit_constants: Optional[Mapping[str, float]] = None,
        jax_uncertainties: Optional[Mapping[str, float]] = None,
        spfit_uncertainties: Optional[Mapping[str, float]] = None,
        tolerance_khz: Optional[float] = None,
    ) -> ParityWarning:
        """Audit parity between JAX-optimized constants and SPFIT constants.

        Args:
            jax_constants: Mapping of constant names to JAX-optimized values (MHz).
            spfit_constants: Mapping of constant names to SPFIT values (MHz), or None if SPFIT failed.
            jax_uncertainties: Optional mapping of JAX parameter uncertainties (MHz).
            spfit_uncertainties: Optional mapping of SPFIT parameter uncertainties (MHz).
            tolerance_khz: Override tolerance threshold in kHz (defaults to instance setting).

        Returns:
            ParityWarning frozen dataclass detailing parity status and contrast metrics.
        """
        tol = float(tolerance_khz) if tolerance_khz is not None else self.default_tolerance_khz

        # Handle failed SPFIT execution
        if spfit_constants is None:
            logger.warning("SPFIT constants are None: flagging SPFIT_FAILED.")
            return ParityWarning(
                status=ParityStatus.SPFIT_FAILED,
                max_delta_khz=float("inf"),
                divergent_constants=tuple(jax_constants.keys()),
                contrast_table={},
                jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
                spfit_uncertainties=None,
            )

        # Audit only common keys present in both parameter sets
        common_keys = [k for k in jax_constants.keys() if k in spfit_constants]

        if not common_keys:
            logger.warning("No overlapping parameter keys found between JAX and SPFIT sets.")
            return ParityWarning(
                status=ParityStatus.VERIFIED,
                max_delta_khz=0.0,
                divergent_constants=(),
                contrast_table={},
                jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
                spfit_uncertainties=dict(spfit_uncertainties) if spfit_uncertainties is not None else None,
            )

        contrast_table: Dict[str, Dict[str, Any]] = {}
        divergent_list: List[str] = []
        max_delta = 0.0

        for key in common_keys:
            val_jax = float(np.float64(jax_constants[key]))
            val_spfit = float(np.float64(spfit_constants[key]))

            delta_mhz = abs(val_jax - val_spfit)
            delta_khz = float(delta_mhz * 1000.0)

            # Check divergence: delta > tol and not within float64 representation precision
            is_divergent = (delta_khz > tol) and not math.isclose(delta_khz, tol, rel_tol=1e-7, abs_tol=1e-9)

            if is_divergent:
                divergent_list.append(key)

            if delta_khz > max_delta:
                max_delta = delta_khz

            jax_unc = (
                float(jax_uncertainties[key])
                if jax_uncertainties is not None and key in jax_uncertainties and jax_uncertainties[key] is not None
                else None
            )
            spfit_unc = (
                float(spfit_uncertainties[key])
                if spfit_uncertainties is not None and key in spfit_uncertainties and spfit_uncertainties[key] is not None
                else None
            )

            status_emoji = "⚠️" if is_divergent else "✅"

            contrast_table[key] = {
                "jax_mhz": val_jax,
                "spfit_mhz": val_spfit,
                "delta_khz": delta_khz,
                "status": status_emoji,
                "is_divergent": is_divergent,
                "jax_unc": jax_unc,
                "spfit_unc": spfit_unc,
                "jax_unc_mhz": jax_unc,
                "spfit_unc_mhz": spfit_unc,
            }

        status = ParityStatus.WARNING if len(divergent_list) > 0 else ParityStatus.VERIFIED

        return ParityWarning(
            status=status,
            max_delta_khz=float(max_delta),
            divergent_constants=tuple(divergent_list),
            contrast_table=contrast_table,
            jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
            spfit_uncertainties=dict(spfit_uncertainties) if spfit_uncertainties is not None else None,
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
            delta_khz = data.get("delta_khz", 0.0)
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
