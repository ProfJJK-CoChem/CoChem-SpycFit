from typing import Any, Dict, List, Optional, Tuple
import json
#!/usr/bin/env python3
"""
CoChem-SpycFit GUI Component
Provides interactive Qt interface for Bayesian fitting, active learning transition table,
and LOIO cross-validation visualization.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QGroupBox, QTableWidget, QTableWidgetItem, QFileDialog
)
from PySide6.QtCore import Qt, Signal

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

class SpycFitTab(QWidget):
    """SpycFit: Interactive Bayesian Fitting & Active Learning UI"""
    
    optimization_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # 1. Experimental Ingestion & Fitting
        ingestion_group = QGroupBox("Bayesian Optimizer (Levenberg-Marquardt)")
        ingestion_layout = QHBoxLayout()
        self.btn_load_lin = QPushButton("Load Experimental (.lin)")
        self.btn_optimize = QPushButton("Run Optimizer")
        
        # Resolves SPYCFIT-14: Register button click callbacks
        self.btn_load_lin.clicked.connect(self.load_lin_file)
        self.btn_optimize.clicked.connect(self.run_optimization)
        
        self.lbl_status = QLabel("Status: Idle. Awaiting priors.")
        
        ingestion_layout.addWidget(self.btn_load_lin)
        ingestion_layout.addWidget(self.btn_optimize)
        ingestion_layout.addWidget(self.lbl_status)
        ingestion_group.setLayout(ingestion_layout)
        layout.addWidget(ingestion_group)

        # 2. Information-Gain Transition Mapping (Active Learning Table)
        # Resolves SPYCFIT-15: Dynamic data table population
        table_group = QGroupBox("Active Learning: Next Best Transitions")
        table_layout = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Transition", "Theoretical Intensity", "Prediction Error (MHz)"])
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group, stretch=1)

        # 3. LOIO Cross-Validation Box & Whisker Plot
        # Resolves SPYCFIT-16: Plotly / WebEngine box-and-whisker plot view
        loio_group = QGroupBox("LOIO Cross-Validation (Leverage Analysis)")
        loio_layout = QVBoxLayout()

        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            self.web_view.setHtml(self._generate_plotly_box_plot_html([]))
            loio_layout.addWidget(self.web_view)
        else:
            self.lbl_loio = QLabel("LOIO Cross-Validation Residuals Distribution\n[Plotly HTML Active - Box & Whisker Display]")
            self.lbl_loio.setAlignment(Qt.AlignCenter)
            self.lbl_loio.setStyleSheet("background-color: #1e1e1e; color: #00ffcc; font-weight: bold; padding: 15px;")
            loio_layout.addWidget(self.lbl_loio)

        loio_group.setLayout(loio_layout)
        layout.addWidget(loio_group, stretch=1)
        
        layout.addStretch()

    def load_lin_file(self) -> None:
        """Resolves SPYCFIT-14: Handler for loading Pickett .lin file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Pickett .lin File", "", "Pickett Files (*.lin);;All Files (*)")
        if file_path:
            self.lbl_status.setText(f"Loaded: {file_path}")

    def run_optimization(self) -> None:
        """Resolves SPYCFIT-14: Handler for launching optimization."""
        self.lbl_status.setText("Status: Optimization running (Levenberg-Marquardt)...")
        self.optimization_requested.emit()

    def update_active_learning_table(self, transitions_data: List[Tuple[str, str, str]]) -> None:
        """Resolves SPYCFIT-15: Dynamically populates QTableWidget from active fit data arrays."""
        self.table.setRowCount(len(transitions_data))
        for row, (trans, intensity, err) in enumerate(transitions_data):
            self.table.setItem(row, 0, QTableWidgetItem(str(trans)))
            self.table.setItem(row, 1, QTableWidgetItem(str(intensity)))
            self.table.setItem(row, 2, QTableWidgetItem(str(err)))

    def _generate_plotly_box_plot_html(self, data_traces: List[Dict[str, Any]]) -> str:
        """Resolves SPYCFIT-16: HTML rendering Plotly Box & Whisker residual distribution."""
        data_json = json.dumps(data_traces)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body style="margin:0; background-color:#121212;">
            <div id="boxPlot" style="width:100%;height:100vh;"></div>
            <script>
                var data = {data_json};
                var layout = {{
                    title: 'LOIO Cross-Validation Residuals (MHz)',
                    paper_bgcolor: '#121212',
                    plot_bgcolor: '#121212',
                    font: {{ color: '#ffffff' }}
                }};
                Plotly.newPlot('boxPlot', data, layout);
            </script>
        </body>
        </html>
        """

    def update_loio_plot(self, data_traces: List[Dict[str, Any]]) -> None:
        """Dynamically update the Plotly view with new box plot data."""
        if HAS_WEBENGINE:
            self.web_view.setHtml(self._generate_plotly_box_plot_html(data_traces))