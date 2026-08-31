"""
Scribe Automated Publication & Supporting Information Exporter Dialog.
"""

from __future__ import annotations

from typing import Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QCheckBox, QComboBox, QTextEdit
    )
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QCheckBox, QComboBox, QTextEdit
    )


class ScribeExportDialog(QDialog):
    """
    Wizard dialog for selecting SI sections and compiling publication-ready reports.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scribe - Automated SI Exporter")
        self.resize(600, 450)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select SI Document Sections:", self))
        self.chk_methods = QCheckBox("1. Computational Chemistry Methodology", self)
        self.chk_coords = QCheckBox("2. Optimized 3D Cartesian Coordinates", self)
        self.chk_thermo = QCheckBox("3. Thermochemistry & Frequencies Summary", self)
        self.chk_prov = QCheckBox("4. W3C PROV-O FAIR Provenance Block", self)
        self.chk_bib = QCheckBox("5. Curated BibTeX References", self)

        for chk in [self.chk_methods, self.chk_coords, self.chk_thermo, self.chk_prov, self.chk_bib]:
            chk.setChecked(True)
            layout.addWidget(chk)

        # Format selector
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Export Format:", self))
        self.fmt_combo = QComboBox(self)
        self.fmt_combo.addItems(["Markdown (.md)", "LaTeX (.tex)", "HTML (.html)"])
        fmt_layout.addWidget(self.fmt_combo)
        layout.addLayout(fmt_layout)

        # Preview area
        layout.addWidget(QLabel("Live Document Preview:", self))
        self.preview_text = QTextEdit(self)
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlainText("# Supporting Information Preview\n\nReady to compile.")
        layout.addWidget(self.preview_text)

        # Actions
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Preview", self)
        self.export_btn = QPushButton("Export to File", self)
        self.close_btn = QPushButton("Close", self)
        self.close_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
