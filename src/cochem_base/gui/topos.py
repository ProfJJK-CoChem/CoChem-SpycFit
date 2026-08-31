"""
TOPOS Conformation & Energy Landscape Interactive GUI Panel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QFormLayout
    )
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QFormLayout
    )


class ToposControlPanel(QWidget):
    """
    Panel for configuring conformer generation, Kabsch clustering, and Boltzmann analysis.
    """

    conformer_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TOPOS Conformation Explorer")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Search Parameter Form
        form = QFormLayout()
        self.rmsd_edit = QLineEdit("0.25", self)
        self.ewin_edit = QLineEdit("6.0", self)
        self.temp_edit = QLineEdit("298.15", self)
        self.trials_edit = QLineEdit("50", self)

        form.addRow("RMSD Cutoff (Å):", self.rmsd_edit)
        form.addRow("Energy Window (kcal/mol):", self.ewin_edit)
        form.addRow("Temperature (K):", self.temp_edit)
        form.addRow("Search Trials:", self.trials_edit)
        layout.addLayout(form)

        # Run button
        self.search_btn = QPushButton("Launch TOPOS Exploration", self)
        layout.addWidget(self.search_btn)

        # Conformer Results Table
        self.conf_table = QTableWidget(0, 4, self)
        self.conf_table.setHorizontalHeaderLabels(["Conformer ID", "ΔE (kcal/mol)", "Weight (%)", "RMSD (Å)"])
        self.conf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.conf_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.conf_table)

    def _on_selection_changed(self) -> None:
        selected_rows = self.conf_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            cid_item = self.conf_table.item(row, 0)
            if cid_item:
                self.conformer_selected.emit(cid_item.text())

    def display_conformers(self, conformers: List[Dict[str, Any]]) -> None:
        """Display conformer records in table."""
        self.conf_table.setRowCount(len(conformers))
        for row, c in enumerate(conformers):
            self.conf_table.setItem(row, 0, QTableWidgetItem(str(c.get("conformer_id", ""))))
            self.conf_table.setItem(row, 1, QTableWidgetItem(f"{c.get('relative_energy_kcal_mol', 0.0):.3f}"))
            weight_pct = c.get('boltzmann_weight', 0.0) * 100.0
            self.conf_table.setItem(row, 2, QTableWidgetItem(f"{weight_pct:.2f}%"))
            self.conf_table.setItem(row, 3, QTableWidgetItem(f"{c.get('rmsd_to_minimum_angstrom', 0.0):.3f}"))
