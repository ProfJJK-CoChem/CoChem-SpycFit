"""
Real-Time Quantum Chemistry Calculation Job Monitoring Dashboard Widget.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
    )
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, QTimer, Signal, Slot
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
    )

import psutil


class CalculationDashboard(QWidget):
    """
    Real-time dashboard displaying active jobs, hardware utilization, and telemetry status.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calculation Dashboard")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Header & Hardware meters
        header_layout = QHBoxLayout()
        self.cpu_label = QLabel("CPU: 0%", self)
        self.ram_label = QLabel("RAM: 0%", self)
        self.cpu_bar = QProgressBar(self)
        self.ram_bar = QProgressBar(self)
        self.cpu_bar.setRange(0, 100)
        self.ram_bar.setRange(0, 100)

        header_layout.addWidget(self.cpu_label)
        header_layout.addWidget(self.cpu_bar)
        header_layout.addWidget(self.ram_label)
        header_layout.addWidget(self.ram_bar)
        layout.addLayout(header_layout)

        # Job Table
        self.job_table = QTableWidget(0, 4, self)
        self.job_table.setHorizontalHeaderLabels(["Job ID", "Fidelity", "Status", "Energy (Eh)"])
        self.job_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.job_table)

        # Controls
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Status", self)
        self.refresh_btn.clicked.connect(self.update_telemetry)
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)

        self.update_telemetry()

    def update_telemetry(self) -> None:
        """Poll host system memory and CPU utilization."""
        cpu_pct = int(psutil.cpu_percent())
        mem = psutil.virtual_memory()
        ram_pct = int(mem.percent)

        self.cpu_label.setText(f"CPU: {cpu_pct}%")
        self.cpu_bar.setValue(cpu_pct)
        self.ram_label.setText(f"RAM: {ram_pct}% ({mem.used // (1024**2)} MB)")
        self.ram_bar.setValue(ram_pct)

    def populate_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """Populate active/historical jobs in the table."""
        self.job_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            self.job_table.setItem(row, 0, QTableWidgetItem(str(job.get("job_id", ""))))
            self.job_table.setItem(row, 1, QTableWidgetItem(str(job.get("fidelity", ""))))
            self.job_table.setItem(row, 2, QTableWidgetItem(str(job.get("status", ""))))
            self.job_table.setItem(row, 3, QTableWidgetItem(str(job.get("energy", "N/A"))))
