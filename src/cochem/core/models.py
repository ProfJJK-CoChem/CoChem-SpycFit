"""
CoChem Core Pydantic v2 Schemas & Models for Chunk 06.

Establishes the centralized data models for CoChem-BASE performance, memory heuristics,
and distributed quantum chemistry execution (REQ-CORE-051 to REQ-DIST-060).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

# Re-export domain exceptions for convenience
from cochem.core.exceptions import (
    AsymmetricNumericalHessianError,
    CoChemError,
    CoChemPerformanceException,
    DistributedDisplacementExhaustionError,
    IncompatibleAuxiliaryBasisError,
    InhomogeneousFeatureSievingError,
    InsufficientMemoryAllocationError,
    SCFSymmetryStagnationError,
)


class MemoryAllocationConfig(BaseModel):
    """
    [REQ-CORE-051] Pydantic v2 data contract for hardware memory heuristics and process throttling.
    """

    model_config = ConfigDict(frozen=True)

    total_system_ram_mb: int = Field(gt=0, description="Total physical host RAM in megabytes.")
    cgroup_limit_mb: Optional[int] = Field(default=None, gt=0, description="Linux cgroups memory limit in megabytes.")
    requested_cores: int = Field(default=1, gt=0, le=512, description="Number of CPU cores requested.")
    headroom_fraction: float = Field(default=0.20, ge=0.05, le=0.50, description="Safety headroom reserved for OS/MPI (default 20%).")
    min_core_mb: int = Field(default=512, ge=256, description="Minimum allowable memory per core in megabytes.")

    @property
    def effective_limit_mb(self) -> int:
        """Calculate effective memory upper bound in megabytes."""
        if self.cgroup_limit_mb is not None:
            return min(self.total_system_ram_mb, self.cgroup_limit_mb)
        return self.total_system_ram_mb


class RIConfigurationPayload(BaseModel):
    """
    [REQ-QC-052, REQ-QC-053] Pydantic v2 data contract for RI / RIJCOSX auxiliary basis sets and integration grids.
    """

    model_config = ConfigDict(frozen=True)

    functional_name: str = Field(..., description="Name of the quantum chemical functional or method.")
    functional_family: Literal["PURE_GGA", "META_GGA", "HYBRID", "DOUBLE_HYBRID", "POST_HF"] = Field(
        ..., description="Taxonomy family classification."
    )
    primary_basis: str = Field(..., description="Primary orbital basis set.")
    aux_coulomb_basis: str = Field(..., description="Auxiliary Coulomb fitting basis set (e.g. def2/J, AutoAux).")
    aux_correlation_basis: Optional[str] = Field(default=None, description="Auxiliary correlation fitting basis (e.g. def2-TZVP/C).")
    grid_level: Literal["GridX 4", "GridX 5", "GridX 6"] = Field(default="GridX 5", description="Numerical integration grid level.")
    no_final_grid_x: bool = Field(default=True, description="Whether to include ! NoFinalGridX to eliminate PES ripples.")
    allow_exact_coulomb: bool = Field(default=False, description="Whether exact four-center Coulomb evaluation is permitted.")


class SievingFeatureVector(BaseModel):
    """
    [REQ-TOPOS-055] Pydantic v2 data contract for 12D invariant geometric conformer feature vectors.
    """

    model_config = ConfigDict(frozen=True)

    conformer_id: str = Field(..., description="Unique conformer identifier string.")
    principal_moments: Tuple[float, float, float] = Field(..., description="Principal moments of inertia (Ia, Ib, Ic) in amu*Å^2.")
    coulomb_eigenvalues_top3: Tuple[float, float, float] = Field(..., description="Top 3 eigenvalues of the Coulomb matrix.")
    min_interatomic_distance: float = Field(..., gt=0.0, description="Minimum pairwise interatomic distance in Å.")
    max_interatomic_distance: float = Field(..., gt=0.0, description="Maximum pairwise interatomic distance in Å.")
    radius_of_gyration: float = Field(..., gt=0.0, description="Radius of gyration Rg in Å.")
    estimated_dipole_debye: float = Field(..., ge=0.0, description="Estimated molecular dipole moment magnitude in Debye.")
    spatial_span: float = Field(..., gt=0.0, description="Maximum distance of any atom from the center of mass in Å.")
    ray_asymmetry_kappa: float = Field(..., ge=-1.0, le=1.0, description="Ray's asymmetry parameter kappa.")
    energy_hartree: float = Field(..., description="Conformer electronic energy in Hartree.")


class DistributedNumFreqTask(BaseModel):
    """
    [REQ-DIST-060] Pydantic v2 data contract for 6N Cartesian displacement single-point gradient tasks.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(..., description="Unique task identifier.")
    atom_index: int = Field(..., ge=0, description="0-based atom index being displaced.")
    cartesian_axis: Literal["x", "y", "z"] = Field(..., description="Displacement Cartesian axis.")
    direction: Literal[1, -1] = Field(..., description="+1 for positive displacement, -1 for negative displacement.")
    displacement_angstrom: float = Field(default=0.005, gt=0.0, le=0.02, description="Displacement step size delta in Å.")
    scratch_directory: str = Field(..., description="Local NVMe scratch directory for execution.")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(default="PENDING", description="Task execution status.")


__all__ = [
    "CoChemError",
    "CoChemPerformanceException",
    "InsufficientMemoryAllocationError",
    "IncompatibleAuxiliaryBasisError",
    "AsymmetricNumericalHessianError",
    "SCFSymmetryStagnationError",
    "InhomogeneousFeatureSievingError",
    "DistributedDisplacementExhaustionError",
    "MemoryAllocationConfig",
    "RIConfigurationPayload",
    "SievingFeatureVector",
    "DistributedNumFreqTask",
]
