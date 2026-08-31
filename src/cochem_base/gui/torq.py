"""
TORQ Torsional Scan and Spectroscopy GUI Panel.
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


class TorqControlPanel(QWidget):
    """
    Panel for setting up torsional dihedral scans, Sinc-DVR tunneling, and rotational barrier views.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TORQ Torsional Spectroscopy")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.dihedral_edit = QLineEdit("0, 1, 2, 3", self)
        self.step_edit = QLineEdit("15.0", self)
        self.reduced_moment_edit = QLineEdit("0.5", self)

        form.addRow("Dihedral Atoms (i, j, k, l):", self.dihedral_edit)
        form.addRow("Scan Step (deg):", self.step_edit)
        form.addRow("Reduced Moment I_red (amu·Å²):", self.reduced_moment_edit)
        layout.addLayout(form)

        self.scan_btn = QPushButton("Execute Dihedral Scan & Sinc-DVR", self)
        layout.addWidget(self.scan_btn)

        # Summary Table
        self.summary_table = QTableWidget(0, 3, self)
        self.summary_table.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.summary_table)

    def display_results(self, barrier_kcal: float, tunneling_cm1: float, q_tor: float) -> None:
        """Populate spectroscopy summary data."""
        self.summary_table.setRowCount(3)
        props = [
            ("Barrier Height (V‡)", f"{barrier_kcal:.3f}", "kcal/mol"),
            ("Tunneling Splitting (ΔE_01)", f"{tunneling_cm1:.4f}", "cm⁻¹"),
            ("Quantum Torsional Q(T)", f"{q_tor:.3f}", "dimensionless"),
        ]
        for row, (p, v, u) in enumerate(props):
            self.summary_table.setItem(row, 0, QTableWidgetItem(p))
            self.summary_table.setItem(row, 1, QTableWidgetItem(v))
            self.summary_table.setItem(row, 2, QTableWidgetItem(u))
