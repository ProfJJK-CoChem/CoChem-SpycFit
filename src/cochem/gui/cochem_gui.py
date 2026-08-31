"""
CoChem Main Graphical User Interface and Master QC Dashboard (REQ-QC-081).

Provenance:
- Method Matrix [M]: Interactive QC Dashboard coordinating Spin-Parity Pre-Flight checks,
  IRC path trajectories, isotopic labeling, and QTAIM topology rendering.
- Derived Relations [D]: Real-time chemical formula electron counting via mendeleev.
- Empirical [E]: UI accessibility, color badges, and defensive validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import (
    SpinParityValidationResult,
    validate_charge_and_multiplicity,
)


class MolecularChargeMultiplicityInput(BaseModel):
    """Data contract representing user input for molecular charge and spin multiplicity."""
    model_config = ConfigDict(frozen=True)

    symbols: List[str]
    charge: int = 0
    multiplicity: int = 1


class GUIStateModel(BaseModel):
    """Data contract representing the global state of the CoChem interactive QC GUI."""
    model_config = ConfigDict(frozen=True)

    active_molecule_symbols: List[str] = Field(default_factory=list)
    active_coordinates_angstrom: List[Tuple[float, float, float]] = Field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    is_spin_parity_valid: bool = True
    spin_parity_message: str = "Ready"
    active_tab: str = "PreFlight"
    workflow_status: str = "IDLE"
    loaded_results: Dict[str, Any] = Field(default_factory=dict)


class CoChemMasterQCController:
    """
    [REQ-QC-081] Main UI Controller for CoChem Quantum Chemistry Workstation.
    """

    def __init__(self, symbols: Optional[Sequence[str]] = None, coordinates: Optional[Sequence[Tuple[float, float, float]]] = None) -> None:
        self.symbols = list(symbols) if symbols else []
        self.coordinates = list(coordinates) if coordinates else []
        self.charge = 0
        self.multiplicity = 1
        self._state = GUIStateModel(
            active_molecule_symbols=self.symbols,
            active_coordinates_angstrom=self.coordinates,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )

    def load_molecule(self, symbols: Sequence[str], coordinates: Optional[Sequence[Tuple[float, float, float]]] = None) -> GUIStateModel:
        """Load molecular structure into controller and re-validate state."""
        self.symbols = list(symbols)
        self.coordinates = list(coordinates) if coordinates else []
        try:
            res = validate_charge_and_multiplicity(
                symbols=self.symbols,
                charge=self.charge,
                multiplicity=self.multiplicity,
            )
            self._state = GUIStateModel(
                active_molecule_symbols=self.symbols,
                active_coordinates_angstrom=self.coordinates,
                charge=self.charge,
                multiplicity=self.multiplicity,
                is_spin_parity_valid=True,
                spin_parity_message=f"Valid electronic state: N_e={res.total_electrons}, M={self.multiplicity}",
            )
        except Exception as exc:
            self._state = GUIStateModel(
                active_molecule_symbols=self.symbols,
                active_coordinates_angstrom=self.coordinates,
                charge=self.charge,
                multiplicity=self.multiplicity,
                is_spin_parity_valid=False,
                spin_parity_message=f"Spin Parity Violation: {exc}",
            )
        return self._state

    def update_spin_and_charge(self, charge: int, multiplicity: int) -> SpinParityValidationResult:
        """
        Validate and update charge and multiplicity with instant pre-flight validation.
        """
        self.charge = charge
        self.multiplicity = multiplicity

        if not self.symbols:
            self._state = GUIStateModel(
                active_molecule_symbols=[],
                active_coordinates_angstrom=[],
                charge=charge,
                multiplicity=multiplicity,
                is_spin_parity_valid=False,
                spin_parity_message="No molecular structure loaded in GUI controller.",
            )
            raise ValueError("Cannot validate spin/charge: No molecular structure loaded in GUI controller.")

        try:
            res = validate_charge_and_multiplicity(
                symbols=self.symbols,
                charge=charge,
                multiplicity=multiplicity,
            )
            self._state = GUIStateModel(
                active_molecule_symbols=self.symbols,
                active_coordinates_angstrom=self.coordinates,
                charge=self.charge,
                multiplicity=self.multiplicity,
                is_spin_parity_valid=True,
                spin_parity_message=f"Valid electronic state: N_e={res.total_electrons}, M={multiplicity}",
            )
            return res
        except Exception as exc:
            self._state = GUIStateModel(
                active_molecule_symbols=self.symbols,
                active_coordinates_angstrom=self.coordinates,
                charge=self.charge,
                multiplicity=self.multiplicity,
                is_spin_parity_valid=False,
                spin_parity_message=f"Spin Parity Violation: {exc}",
            )
            raise

    def get_state(self) -> GUIStateModel:
        """Retrieve current UI state."""
        return self._state
