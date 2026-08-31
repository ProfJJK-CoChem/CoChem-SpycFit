"""
CoChem Core Domain Exceptions Hierarchy.

Establishes the centralized domain exception hierarchy for CoChem-BASE Chunk 05
(REQ-CORE-041 through REQ-TOPOS-050).

Invariants:
- All exceptions derive from CoChemError.
- Subclassed by domain: QuantumChemistryError, HardwareResourceError, MachineLearningError.
- Explicit __init__ constructors with full typing and no empty pass blocks.
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


class QuantumChemistryError(CoChemError):
    """Root domain exception for all ab-initio, DFT, and semi-empirical QC failures."""

    def __init__(self, message: str, method: Optional[str] = None, basis: Optional[str] = None) -> None:
        self.method = method
        self.basis = basis
        details: Dict[str, Any] = {}
        if method is not None:
            details["method"] = method
        if basis is not None:
            details["basis"] = basis
        super().__init__(message, details=details)


class HardwareResourceError(CoChemError):
    """Root domain exception for CPU, GPU, memory, MPI, and concurrency lock errors."""

    def __init__(self, message: str, resource_type: Optional[str] = None, requested: Optional[Any] = None) -> None:
        self.resource_type = resource_type
        self.requested = requested
        details: Dict[str, Any] = {}
        if resource_type is not None:
            details["resource_type"] = resource_type
        if requested is not None:
            details["requested"] = requested
        super().__init__(message, details=details)


class MachineLearningError(CoChemError):
    """Root domain exception for MLIP, neural potential, and clustering errors."""

    def __init__(self, message: str, backend: Optional[str] = None) -> None:
        self.backend = backend
        details: Dict[str, Any] = {}
        if backend is not None:
            details["backend"] = backend
        super().__init__(message, details=details)


class ECPConfigurationError(QuantumChemistryError):
    """
    Raised on unsupported elements or invalid basis/ECP combinations (REQ-CORE-041).
    """

    def __init__(self, element_symbol: str, atomic_number: int, basis: str, reason: str) -> None:
        self.element_symbol = element_symbol
        self.atomic_number = atomic_number
        msg = (
            f"ECP configuration failed for element '{element_symbol}' (Z={atomic_number}) "
            f"with basis '{basis}': {reason}"
        )
        super().__init__(msg, method="ECP", basis=basis)
        self.details.update({
            "element_symbol": element_symbol,
            "atomic_number": atomic_number,
            "reason": reason,
        })


class SaddlePointEscapeError(QuantumChemistryError):
    """
    Raised when saddle-point re-optimization fails after MAX_PIVOT_CYCLES=3 (REQ-QC-042).
    """

    def __init__(self, pivot_cycles: int, lowest_frequency: float, message: Optional[str] = None) -> None:
        self.pivot_cycles = pivot_cycles
        self.lowest_frequency = lowest_frequency
        msg = message or (
            f"Saddle-point escape failed after {pivot_cycles} pivot cycles. "
            f"Lowest vibrational harmonic mode persists at {lowest_frequency:.2f} cm^-1 (< -15.0 cm^-1)."
        )
        super().__init__(msg)
        self.details.update({
            "pivot_cycles": pivot_cycles,
            "lowest_frequency": lowest_frequency,
        })


class BenchmarkMotifMismatchError(QuantumChemistryError):
    """
    Raised when 3D principal moments of inertia deviate > 5% from canonical S22/S66 benchmarks (REQ-BM-043).
    """

    def __init__(
        self,
        benchmark_id: str,
        deviation_pct: float,
        calculated_inertia: Tuple[float, float, float],
        reference_inertia: Tuple[float, float, float],
    ) -> None:
        self.benchmark_id = benchmark_id
        self.deviation_pct = deviation_pct
        self.calculated_inertia = calculated_inertia
        self.reference_inertia = reference_inertia
        msg = (
            f"Benchmark motif mismatch for '{benchmark_id}': principal moments of inertia "
            f"deviate by {deviation_pct:.2f}% (exceeds allowable tolerance of 5.00%). "
            f"Calculated: {calculated_inertia}, Reference: {reference_inertia}."
        )
        super().__init__(msg)
        self.details.update({
            "benchmark_id": benchmark_id,
            "deviation_pct": deviation_pct,
            "calculated_inertia": calculated_inertia,
            "reference_inertia": reference_inertia,
        })


class StericGridLockError(QuantumChemistryError):
    """
    Raised if adaptive step refinement drops below 0.1° without resolving steric clash/SCF divergence (REQ-TORQ-044).
    """

    def __init__(self, dihedral_angle_deg: float, step_deg: float, min_steric_ratio: float) -> None:
        self.dihedral_angle_deg = dihedral_angle_deg
        self.step_deg = step_deg
        self.min_steric_ratio = min_steric_ratio
        msg = (
            f"Steric grid lock encountered at dihedral angle {dihedral_angle_deg:.2f}°: "
            f"step size refined to {step_deg:.4f}° (< 0.1000°) with unresolved steric clash "
            f"(min steric ratio: {min_steric_ratio:.4f})."
        )
        super().__init__(msg)
        self.details.update({
            "dihedral_angle_deg": dihedral_angle_deg,
            "step_deg": step_deg,
            "min_steric_ratio": min_steric_ratio,
        })


class UnsupportedMethodThermoError(QuantumChemistryError):
    """
    Raised when empirical frequency scaling factors are missing for a given functional/basis tuple (REQ-THM-045).
    """

    def __init__(self, method: str, basis: Optional[str] = None, available_methods: Optional[List[str]] = None) -> None:
        self.available_methods = available_methods or []
        msg = (
            f"Empirical thermochemistry frequency scaling factors not found for method '{method}' "
            f"(basis: '{basis}'). Supported methods include: {', '.join(self.available_methods)}."
        )
        super().__init__(msg, method=method, basis=basis)
        self.details.update({
            "available_methods": self.available_methods,
        })


class CudaKernelExecutionError(HardwareResourceError):
    """
    Raised on CUDA runtime, driver mismatch, or out-of-VRAM exceptions (REQ-HW-046).
    """

    def __init__(self, device_id: int, reason: str, vram_free_mb: Optional[int] = None) -> None:
        self.device_id = device_id
        self.reason = reason
        self.vram_free_mb = vram_free_mb
        msg = (
            f"CUDA kernel execution failed on GPU device {device_id}: {reason} "
            f"(Free VRAM: {vram_free_mb} MB)."
        )
        super().__init__(msg, resource_type="GPU_CUDA", requested=device_id)
        self.details.update({
            "device_id": device_id,
            "reason": reason,
            "vram_free_mb": vram_free_mb,
        })


class HardwareAllocationFloorError(HardwareResourceError):
    """
    Raised if scheduler or workstation process calculation yields <= 0 cores (REQ-HW-047).
    """

    def __init__(self, calculated_cores: int, scheduler_type: str, raw_metrics: Dict[str, Any]) -> None:
        self.calculated_cores = calculated_cores
        self.scheduler_type = scheduler_type
        self.raw_metrics = raw_metrics
        msg = (
            f"Hardware allocation floor violation: calculated CPU processes N_procs={calculated_cores} "
            f"is <= 0 for scheduler '{scheduler_type}'. Raw metrics: {raw_metrics}."
        )
        super().__init__(msg, resource_type="CPU_CORES", requested=calculated_cores)
        self.details.update({
            "calculated_cores": calculated_cores,
            "scheduler_type": scheduler_type,
            "raw_metrics": raw_metrics,
        })


class ModelCoverageError(MachineLearningError):
    """
    Raised when input elements exceed the coverage of all NNIP fallback ladder tiers (REQ-ML-048).
    """

    def __init__(self, uncovered_elements: List[str], attempted_backends: List[str]) -> None:
        self.uncovered_elements = uncovered_elements
        self.attempted_backends = attempted_backends
        msg = (
            f"NNIP model coverage failure: elements {uncovered_elements} are not supported "
            f"across any fallback ladder tiers ({', '.join(attempted_backends)})."
        )
        super().__init__(msg, backend=attempted_backends[-1] if attempted_backends else "none")
        self.details.update({
            "uncovered_elements": uncovered_elements,
            "attempted_backends": attempted_backends,
        })


class StationaryPointExtractionError(QuantumChemistryError):
    """
    Raised if no valid local minima are identified on the xTB baseline PES (REQ-TORQ-049).
    """

    def __init__(self, grid_size: int, energy_range_kcal_mol: float, message: Optional[str] = None) -> None:
        self.grid_size = grid_size
        self.energy_range_kcal_mol = energy_range_kcal_mol
        msg = message or (
            f"No valid stationary local minima (grad=0, hessian>0) identified on baseline PES "
            f"with {grid_size} grid points (Energy range: {energy_range_kcal_mol:.2f} kcal/mol)."
        )
        super().__init__(msg)
        self.details.update({
            "grid_size": grid_size,
            "energy_range_kcal_mol": energy_range_kcal_mol,
        })


class ConcurrencyLockTimeoutError(HardwareResourceError):
    """
    Raised when acquiring cross-platform filelock exceeds timeout (REQ-TOPOS-050).
    """

    def __init__(self, lock_file_path: str, timeout_seconds: float) -> None:
        self.lock_file_path = lock_file_path
        self.timeout_seconds = timeout_seconds
        msg = (
            f"Concurrency lock acquisition timed out after {timeout_seconds:.1f}s "
            f"for lock file: '{lock_file_path}'."
        )
        super().__init__(msg, resource_type="FILE_LOCK", requested=lock_file_path)
        self.details.update({
            "lock_file_path": lock_file_path,
            "timeout_seconds": timeout_seconds,
        })
