"""
CoChem Core Domain Exceptions Hierarchy for Chunk 06.

Establishes the centralized domain exception hierarchy for CoChem-BASE Chunk 06
(REQ-CORE-051 through REQ-DIST-060).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class CoChemError(Exception):
    """Top-level root exception for all CoChem domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class CoChemPerformanceException(CoChemError):
    """Base exception for all CoChem performance, memory heuristic, and numerical execution errors."""

    def __init__(self, message: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message or "CoChem performance exception occurred.", details=details)


class InsufficientMemoryAllocationError(CoChemPerformanceException):
    """Raised when available memory per core is below minimum threshold (e.g. < 512 MB)."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        requested_cores: int = 1,
        available_mb: int = 0,
        min_core_mb: int = 512,
        **kwargs: Any,
    ) -> None:
        self.requested_cores = requested_cores
        self.available_mb = available_mb
        self.min_core_mb = min_core_mb
        msg = message or (
            f"Insufficient memory allocation: usable RAM {available_mb} MB for {requested_cores} cores "
            f"yields {available_mb / max(1, requested_cores):.1f} MB/core, which is below the "
            f"mandatory minimum of {min_core_mb} MB/core."
        )
        super().__init__(
            msg,
            details={
                "requested_cores": requested_cores,
                "available_mb": available_mb,
                "min_core_mb": min_core_mb,
                **kwargs,
            },
        )


class IncompatibleAuxiliaryBasisError(CoChemPerformanceException):
    """Raised when a primary basis set lacks a compatible Coulomb or correlation fitting set."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        primary_basis: str = "",
        functional: str = "",
        **kwargs: Any,
    ) -> None:
        self.primary_basis = primary_basis
        self.functional = functional
        msg = message or (
            f"Incompatible auxiliary basis: primary basis '{primary_basis}' has no resolved "
            f"Coulomb or correlation auxiliary fitting basis for functional '{functional}'."
        )
        super().__init__(
            msg,
            details={
                "primary_basis": primary_basis,
                "functional": functional,
                **kwargs,
            },
        )


class AsymmetricNumericalHessianError(CoChemPerformanceException):
    """Raised when numerical gradient difference asymmetry Delta_asym >= 1e-4."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        delta_asymmetry: float = 0.0,
        threshold: float = 1.0e-4,
        **kwargs: Any,
    ) -> None:
        self.delta_asymmetry = delta_asymmetry
        self.threshold = threshold
        msg = message or (
            f"Asymmetric numerical Hessian detected: maximum gradient difference asymmetry "
            f"Delta_asym={delta_asymmetry:.3e} exceeds allowable threshold of {threshold:.3e}. "
            f"Likely caused by loose SCF convergence or grid instability."
        )
        super().__init__(
            msg,
            details={
                "delta_asymmetry": delta_asymmetry,
                "threshold": threshold,
                **kwargs,
            },
        )


class SCFSymmetryStagnationError(CoChemPerformanceException):
    """Raised when symmetry-constrained SCF exhibits stagnation or non-convergence (> 35 cycles)."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        scf_cycles: int = 0,
        point_group: str = "C1",
        **kwargs: Any,
    ) -> None:
        self.scf_cycles = scf_cycles
        self.point_group = point_group
        msg = message or (
            f"SCF symmetry stagnation encountered under point group '{point_group}' after "
            f"{scf_cycles} SCF iterations without convergence. Coordinate perturbation and symmetry "
            f"release required."
        )
        super().__init__(
            msg,
            details={
                "scf_cycles": scf_cycles,
                "point_group": point_group,
                **kwargs,
            },
        )


class InhomogeneousFeatureSievingError(CoChemPerformanceException):
    """Raised when KD-Tree conformer population features exhibit zero variance (< 1e-12)."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        zero_variance_indices: Optional[List[int]] = None,
        variances: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        self.zero_variance_indices = zero_variance_indices or []
        self.variances = variances or []
        msg = message or (
            f"Inhomogeneous feature sieving failure: feature variance at indices {self.zero_variance_indices} "
            f"is below mathematical numerical stability threshold (< 1e-12)."
        )
        super().__init__(
            msg,
            details={
                "zero_variance_indices": self.zero_variance_indices,
                "variances": self.variances,
                **kwargs,
            },
        )


class DistributedDisplacementExhaustionError(CoChemPerformanceException):
    """Raised when unrecoverable gradient worker dropouts occur during 6N numerical frequencies."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        failed_tasks: int = 0,
        total_tasks: int = 0,
        **kwargs: Any,
    ) -> None:
        self.failed_tasks = failed_tasks
        self.total_tasks = total_tasks
        msg = message or (
            f"Distributed displacement exhaustion: {failed_tasks} of {total_tasks} gradient calculation "
            f"tasks failed unrecoverably across worker pool."
        )
        super().__init__(
            msg,
            details={
                "failed_tasks": failed_tasks,
                "total_tasks": total_tasks,
                **kwargs,
            },
        )
