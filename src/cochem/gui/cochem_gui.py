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

    # =========================================================================
    # Chunk 10: Publication, Supporting Information, and FAIR Archival Actions
    # (REQ-QC-091 through REQ-QC-100)
    # =========================================================================

    def open_table_editor(self) -> ACSTableEditorWidget:
        """Instantiate and return the reactive ACS Table Editor widget."""
        return ACSTableEditorWidget()

    def open_orbital_viewer(self, cube_file_path: Optional[str] = None) -> OrbitalViewerWidget:
        """Instantiate and return the Orbital Viewer widget."""
        return OrbitalViewerWidget(cube_file_path=cube_file_path)

    def open_html_si_preview(self, html_path: Optional[str] = None) -> HTMLSIPreviewWidget:
        """Instantiate and return the 3D WebGL HTML SI Preview widget."""
        return HTMLSIPreviewWidget(html_path=html_path)

    def open_zenodo_dialog(self, project_dir: Optional[str] = None, api_token: str = "") -> ZenodoPublishDialog:
        """Instantiate and return the Zenodo FAIR publishing dialog."""
        return ZenodoPublishDialog(project_dir=project_dir, api_token=api_token)

    def generate_publication_package(
        self,
        output_dir: str,
        authors: List[ORCIDAuthorProfile],
        title: str = "Quantum Chemical Investigation",
        methods_config: Optional[ComputationalMethodsConfig] = None,
        stationary_points: Optional[List[StationaryPointSIEntry]] = None,
    ) -> Dict[str, Any]:
        """Dispatch asynchronous / local assembly of the complete publication bundle."""
        return assemble_full_publication_package(
            project_dir=".",
            output_dir=output_dir,
            author_profiles=authors,
            title=title,
            methods_config=methods_config,
            stationary_points=stationary_points,
        )

    def export_excel_spreadsheet(
        self,
        records: List[MolecularRecordSpreadsheetRow],
        output_excel_path: str,
        output_csv_path: str,
    ) -> SpreadsheetExportManifest:
        """Export calculation dataset to multi-sheet Excel and CSV."""
        return export_molecular_dataset_to_excel(
            records=records,
            output_excel_path=output_excel_path,
            output_csv_path=output_csv_path,
        )


from cochem.gui.widgets.table_editor import ACSTableEditorWidget
from cochem.gui.widgets.orbital_viewer import OrbitalViewerWidget
from cochem.gui.widgets.html_si_preview import HTMLSIPreviewWidget
from cochem.gui.widgets.zenodo_dialog import ZenodoPublishDialog
from cochem.provenance.cochem_orcid_resolver import ORCIDAuthorProfile
from cochem.reporting.cochem_methods_writer import ComputationalMethodsConfig
from cochem.reporting.cochem_si_builder import StationaryPointSIEntry
from cochem.reporting.cochem_report_generator import assemble_full_publication_package
from cochem.export.cochem_spreadsheet_exporter import (
    MolecularRecordSpreadsheetRow,
    SpreadsheetExportManifest,
    export_molecular_dataset_to_excel,
)

