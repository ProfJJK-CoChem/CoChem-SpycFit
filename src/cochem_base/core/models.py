"""
Immutable Pydantic v2 domain schemas and QCSchema v1.0 data models.

Enforces frozen state immutability across all runtime state representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cochem_base.core.glossary import CalculationFidelity


class MolecularTopology(BaseModel):
    """
    Immutable representation of 3D molecular Cartesian coordinates and quantum properties.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    atomic_numbers: List[int] = Field(..., description="List of nuclear charge integer Z numbers.")
    coordinates_bohr: List[List[float]] = Field(..., description="Cartesian coordinate matrix in Bohr (N_atoms, 3).")
    charge: int = Field(default=0, description="Net molecular charge.")
    multiplicity: int = Field(default=1, description="Spin multiplicity (2S + 1).")
    symbols: Optional[List[str]] = Field(default=None, description="Optional chemical element symbols.")

    @field_validator("coordinates_bohr")
    @classmethod
    def validate_coord_shape(cls, v: List[List[float]], info: Any) -> List[List[float]]:
        for row in v:
            if len(row) != 3:
                raise ValueError("Each Cartesian coordinate vector must contain exactly 3 elements (x, y, z).")
        return v

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    def get_coordinates_angstrom(self) -> List[List[float]]:
        from cochem_base.core.glossary import UnitConversionConstants
        factor = UnitConversionConstants.BOHR_TO_ANGSTROM
        return [[c * factor for c in row] for row in self.coordinates_bohr]


class QCResultsRecord(BaseModel):
    """
    MolSSI QCSchema v1.0 compliant quantum chemistry output state schema.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    energy_hartree: float = Field(..., description="Electronic total energy in Hartree.")
    gradient_bohr: Optional[List[List[float]]] = Field(default=None, description="Cartesian gradient in Hartree/Bohr.")
    hessian: Optional[List[List[float]]] = Field(default=None, description="Cartesian Hessian matrix (3N x 3N).")
    dipole_moment_debye: Optional[List[float]] = Field(default=None, description="Dipole moment vector (x, y, z) in Debye.")
    rotational_constants_ghz: Optional[List[float]] = Field(default=None, description="Principal rotational constants [A, B, C] in GHz.")
    frequencies_cm1: Optional[List[float]] = Field(default=None, description="Harmonic vibrational frequencies in cm^-1.")
    zpve_hartree: Optional[float] = Field(default=None, description="Zero-Point Vibrational Energy in Hartree.")
    enthalpy_hartree: Optional[float] = Field(default=None, description="Total Enthalpy H(T) in Hartree at standard temperature.")
    entropy_cal_mol_k: Optional[float] = Field(default=None, description="Total Entropy S(T) in cal/(mol*K).")
    gibbs_free_energy_hartree: Optional[float] = Field(default=None, description="Total Gibbs Free Energy G(T) in Hartree.")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Metadata and execution provenance record.")


class ConformerRecord(BaseModel):
    """
    Ranked conformational state record.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conformer_id: str = Field(..., description="Unique conformer identifier hash.")
    topology: MolecularTopology = Field(..., description="Molecular coordinate topology.")
    relative_energy_kcal_mol: float = Field(..., description="Relative energy relative to global minimum (kcal/mol).")
    boltzmann_weight: float = Field(..., description="Normalized Boltzmann statistical population fraction [0, 1].")
    rmsd_to_minimum_angstrom: float = Field(default=0.0, description="Heavy-atom Kabsch RMSD relative to global minimum.")


class CalculationJobPayload(BaseModel):
    """
    Subprocess execution payload schema for quantum chemistry calculation dispatch.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    job_id: str = Field(..., description="Unique job UUID.")
    topology: MolecularTopology = Field(..., description="Target molecular topology.")
    fidelity: CalculationFidelity = Field(..., description="Requested calculation fidelity.")
    command_flags: List[str] = Field(default_factory=list, description="Method and keyword command arguments.")
    scratch_dir: Optional[str] = Field(default=None, description="Dedicated scratch directory path.")
    status: str = Field(default="PENDING", description="Execution status state.")
