"""
CoChem GUI ACS Energy Table Preview & Editor Widget (REQ-QC-096).

Provenance:
- Method Matrix [M]: Interactive relative thermochemistry table editor with dynamic reference species selection,
  energy unit toggling (kcal/mol vs kJ/mol), and OS clipboard export for LaTeX booktabs and GFM Markdown.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.reporting.cochem_table_formatter import (
    ACSTableManifest,
    ThermochemicalSpeciesEntry,
    build_acs_energy_table,
)


class TableEditorState(BaseModel):
    """Data contract representing the reactive state of the table editor widget."""
    model_config = ConfigDict(frozen=True)

    reference_species: str = ""
    energy_units: str = "kcal_mol"
    species_count: int = 0
    manifest: Optional[ACSTableManifest] = None
    latex_clipboard_text: str = ""
    markdown_clipboard_text: str = ""


class ACSTableEditorWidget:
    """
    [REQ-QC-096] GUI Table Editor and Clipboard Export Widget.
    """

    def __init__(
        self,
        entries: Optional[List[ThermochemicalSpeciesEntry]] = None,
        default_reference: Optional[str] = None,
        energy_units: str = "kcal_mol",
    ) -> None:
        self.entries: List[ThermochemicalSpeciesEntry] = list(entries) if entries else []
        self.energy_units = energy_units
        self.reference_species = default_reference or (self.entries[0].species_name if self.entries else "")
        self._state: TableEditorState = self._recompute_state()

    def _recompute_state(self) -> TableEditorState:
        if not self.entries or not self.reference_species:
            return TableEditorState(
                reference_species=self.reference_species,
                energy_units=self.energy_units,
                species_count=len(self.entries),
            )

        manifest = build_acs_energy_table(
            entries=self.entries,
            reference_label=self.reference_species,
            energy_units=self.energy_units,
        )

        return TableEditorState(
            reference_species=self.reference_species,
            energy_units=self.energy_units,
            species_count=len(self.entries),
            manifest=manifest,
            latex_clipboard_text=manifest.latex_table_code,
            markdown_clipboard_text=manifest.markdown_table_code,
        )

    def load_entries(self, entries: List[ThermochemicalSpeciesEntry], reference_label: Optional[str] = None) -> TableEditorState:
        """Load new thermochemical entries and refresh the table."""
        self.entries = list(entries)
        if reference_label:
            self.reference_species = reference_label
        elif self.entries and (not self.reference_species or self.reference_species not in [e.species_name for e in self.entries]):
            self.reference_species = self.entries[0].species_name

        self._state = self._recompute_state()
        return self._state

    def set_reference_species(self, species_name: str) -> TableEditorState:
        """Change the reference ground state (0.00) and recompute relative energies."""
        if species_name not in [e.species_name for e in self.entries]:
            raise ValueError(f"Species '{species_name}' is not in the loaded entries list.")
        self.reference_species = species_name
        self._state = self._recompute_state()
        return self._state

    def toggle_energy_units(self, units: str) -> TableEditorState:
        """Toggle between 'kcal_mol' and 'kj_mol'."""
        if units not in ["kcal_mol", "kj_mol"]:
            raise ValueError("Energy units must be 'kcal_mol' or 'kj_mol'")
        self.energy_units = units
        self._state = self._recompute_state()
        return self._state

    def copy_latex_to_clipboard(self) -> str:
        """Return the formatted LaTeX booktabs string for copying to clipboard."""
        return self._state.latex_clipboard_text

    def copy_markdown_to_clipboard(self) -> str:
        """Return the formatted Markdown string for copying to clipboard."""
        return self._state.markdown_clipboard_text

    def export_csv_text(self) -> str:
        """Generate CSV string of current relative energies."""
        if not self._state.manifest:
            return ""
        lines = ["Species,Delta_E,Delta_E0,Delta_H298,Delta_G298,Delta_G_qh,nu_imag"]
        for r in self._state.manifest.rows:
            qh_str = f"{r.delta_g_qh:.2f}" if r.delta_g_qh is not None else ""
            lines.append(f"{r.species_name},{r.delta_e:.2f},{r.delta_e0:.2f},{r.delta_h:.2f},{r.delta_g:.2f},{qh_str},{r.imaginary_frequency_str}")
        return "\n".join(lines)

    @property
    def state(self) -> TableEditorState:
        return self._state
