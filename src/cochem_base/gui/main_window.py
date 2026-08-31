"""
Master Desktop QMainWindow Container for CoChem-BASE.
"""

from __future__ import annotations

from typing import Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDockWidget, QStatusBar,
        QMenuBar, QMenu, QVBoxLayout, QLabel
    )
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDockWidget, QStatusBar,
        QMenuBar, QMenu, QVBoxLayout, QLabel
    )

from cochem_base import __version__
from cochem_base.gui.dashboard import CalculationDashboard
from cochem_base.gui.topos import ToposControlPanel
from cochem_base.gui.torq import TorqControlPanel
from cochem_base.gui.scribe import ScribeExportDialog


class CoChemMainWindow(QMainWindow):
    """
    Primary desktop application shell integrating dockable calculation panels.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"CoChem-BASE Computational Ecosystem v{__version__}")
        self.resize(1200, 800)
        self._init_ui()

    def _init_ui(self) -> None:
        # Central Viewport Widget Area
        central = QWidget(self)
        c_layout = QVBoxLayout(central)
        c_label = QLabel("3D Molecular Viewport Active", central)
        c_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter)
        c_layout.addWidget(c_label)
        self.setCentralWidget(central)

        # Dockable Panels
        self.dashboard_dock = QDockWidget("Calculations Dashboard", self)
        self.dashboard_widget = CalculationDashboard(self.dashboard_dock)
        self.dashboard_dock.setWidget(self.dashboard_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea if hasattr(Qt, "DockWidgetArea") else Qt.RightDockWidgetArea, self.dashboard_dock)

        self.topos_dock = QDockWidget("TOPOS Conformation", self)
        self.topos_widget = ToposControlPanel(self.topos_dock)
        self.topos_dock.setWidget(self.topos_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea if hasattr(Qt, "DockWidgetArea") else Qt.LeftDockWidgetArea, self.topos_dock)

        self.torq_dock = QDockWidget("TORQ Torsional Scan", self)
        self.torq_widget = TorqControlPanel(self.torq_dock)
        self.torq_dock.setWidget(self.torq_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea if hasattr(Qt, "DockWidgetArea") else Qt.LeftDockWidgetArea, self.torq_dock)

        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Open Structure (XYZ/PDB)...")
        file_menu.addAction("Export SI Report (Scribe)...", self.open_scribe)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        calc_menu = menubar.addMenu("&Calculations")
        calc_menu.addAction("Dispatch Quantum Job...")
        calc_menu.addAction("Conformer Search (TOPOS)...")
        calc_menu.addAction("Torsional Scan (TORQ)...")

        # Status Bar
        self.statusbar = QStatusBar(self)
        self.statusbar.showMessage(f"CoChem-BASE v{__version__} Ready | Zero-Mock Invariant Active")
        self.setStatusBar(self.statusbar)

    def open_scribe(self) -> None:
        """Open Scribe dialog."""
        dialog = ScribeExportDialog(self)
        dialog.exec()
