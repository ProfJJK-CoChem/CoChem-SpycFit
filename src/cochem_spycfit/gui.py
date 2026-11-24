#!/usr/bin/env python3
"""
CoChem-SpycFit GUI Component
Provides interactive Qt interface for Bayesian fitting, active learning transition table,
LOIO cross-validation visualization, Instrument Profile selection (defaulting to
``grulab_surf``), and the HITL SURF User Parameter Override Table + native TOML exporter.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ValidationError

from .instruments.profiles import (
    DEFAULT_INSTRUMENT_PROFILE,
    InstrumentProfile,
    get_instrument_profile,
    list_instrument_profiles,
)
from .instruments.surf_exporter import SurfExporter, SurfExportResult
from .instruments.surf_planner import (
    SurfParameterOverrideTable,
    SurfScanPlanner,
    SurfScanProposal,
)

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    QWidget = object  # type: ignore[assignment,misc]

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


INSTRUMENT_PROFILE_LABELS: Dict[str, str] = {
    "grulab_surf": "GruLab SURF (8–18 GHz Cavity FTMW)",
    "generic_cp_ftmw": "Generic CP-FTMW (2–18 GHz Broadband)",
    "generic_ir_raman": "Generic IR / Raman Spectrometer",
}


def format_instrument_profile_label(profile_id: str) -> str:
    """Return user-facing label for an instrument profile ID."""
    if profile_id in INSTRUMENT_PROFILE_LABELS:
        return INSTRUMENT_PROFILE_LABELS[profile_id]
    return get_instrument_profile(profile_id).display_name


def _format_cell_value(value: Any) -> str:
    """Format an override value cleanly for QTableWidget cell display."""
    if value is None:
        return "None"
    return str(value)


def _format_profile_summary(profile: InstrumentProfile) -> str:
    """Build a concise hardware envelope summary string for the GUI status bar."""
    rbw_khz = profile.default_fft_rbw_khz
    parts = [
        f"Band: {profile.freq_min_mhz:.0f}–{profile.freq_max_mhz:.0f} {profile.spectral_unit}",
        f"RBW: {rbw_khz:.1f} kHz",
    ]
    if profile.verified_commit_sha:
        parts.append(f"SURF@{profile.verified_commit_sha[:8]}")
    return " | ".join(parts)


class PlotlyTrace(BaseModel):
    """Pydantic model for Plotly trace validation."""

    y: List[float]
    type: str = "box"
    name: str


class TransitionData(BaseModel):
    """Pydantic model for Active Learning table row."""

    transition: str
    theoretical_intensity: float
    prediction_error: float


class SurfOverridePanelController:
    """Headless-safe controller managing Instrument Profile selection and SURF overrides."""

    def __init__(
        self,
        *,
        instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE,
        override_table: Optional[SurfParameterOverrideTable] = None,
        operator_id: str = "operator",
    ) -> None:
        self._profile: InstrumentProfile = get_instrument_profile(instrument_profile)
        self._operator_id: str = str(operator_id)
        self._override_table: SurfParameterOverrideTable = (
            override_table
            if override_table is not None
            else SurfScanPlanner(instrument_profile=DEFAULT_INSTRUMENT_PROFILE).create_default_override_table()
        )
        self._exporter: SurfExporter = SurfExporter(instrument_profile=DEFAULT_INSTRUMENT_PROFILE)
        self._last_error: Optional[str] = None
        self._last_export: Optional[SurfExportResult] = None

    @property
    def profile(self) -> InstrumentProfile:
        """Return currently selected ``InstrumentProfile``."""
        return self._profile

    @property
    def instrument_profile(self) -> str:
        """Return currently selected instrument profile identifier."""
        return self._profile.profile_id

    @property
    def override_table(self) -> SurfParameterOverrideTable:
        """Return active ``SurfParameterOverrideTable`` state model."""
        return self._override_table

    @property
    def last_error(self) -> Optional[str]:
        """Return last validation error encountered on an edit, if any."""
        return self._last_error

    @property
    def last_export(self) -> Optional[SurfExportResult]:
        """Return most recent ``SurfExportResult``, if any."""
        return self._last_export

    def select_instrument_profile(self, profile_id: str) -> InstrumentProfile:
        """Switch active instrument profile after validating against registry."""
        self._profile = get_instrument_profile(profile_id)
        return self._profile

    def load_override_table(self, table: SurfParameterOverrideTable) -> None:
        """Attach an externally planned ``SurfParameterOverrideTable``."""
        self._override_table = table
        self._last_error = None

    def load_proposal(self, proposal: SurfScanProposal) -> SurfParameterOverrideTable:
        """Initialize the override table from a ``SurfScanProposal``."""
        self._override_table = SurfParameterOverrideTable(
            proposed_scan_settings=proposal.scan_settings,
            proposed_control_config=proposal.control_config,
            proposal=proposal,
            retune_overhead_s=get_instrument_profile(DEFAULT_INSTRUMENT_PROFILE).zaber_retune_overhead_s,
        )
        self._last_error = None
        return self._override_table

    def apply_cell_edit(
        self,
        field_path: str,
        raw_value: Any,
        *,
        edited_by: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Parse and validate a user edit; returns ``(ok, status_message)``."""
        editor = edited_by if edited_by is not None else self._operator_id
        try:
            parsed_value = (
                SurfParameterOverrideTable.parse_field_value(field_path, raw_value)
                if isinstance(raw_value, str)
                else raw_value
            )
            self._override_table.set_override(field_path, parsed_value, edited_by=editor)
            self._last_error = None
            return True, f"Updated {field_path} = {self._override_table.get_current_value(field_path)!r}"
        except (ValidationError, ValueError, KeyError) as exc:
            self._last_error = str(exc)
            return False, f"Validation rejected {field_path}: {exc}"

    def export_surf_files(
        self,
        output_dir: Union[str, Path],
        *,
        region_stem: Optional[str] = None,
        fit_state_id: Optional[str] = None,
        spcat_catalog_sha256: Optional[str] = None,
        gp_hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> SurfExportResult:
        """Export validated native SURF TOMLs and strict manifest JSON."""
        result = self._exporter.export_region(
            output_dir=output_dir,
            override_table=self._override_table,
            region_stem=region_stem,
            fit_state_id=fit_state_id,
            spcat_catalog_sha256=spcat_catalog_sha256,
            gp_hyperparameters=gp_hyperparameters,
        )
        self._last_export = result
        return result


if HAS_PYSIDE6:

    class SpycFitTab(QWidget):
        """SpycFit: Interactive Bayesian Fitting, Active Learning & SURF Override UI."""

        optimization_requested = Signal()
        instrument_profile_changed = Signal(str)
        surf_export_completed = Signal(object)

        def __init__(
            self,
            *,
            instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE,
            override_table: Optional[SurfParameterOverrideTable] = None,
            operator_id: str = "operator",
        ) -> None:
            super().__init__()
            self._controller = SurfOverridePanelController(
                instrument_profile=instrument_profile,
                override_table=override_table,
                operator_id=operator_id,
            )
            self._populating_override_table: bool = False
            self._override_row_by_field: Dict[str, int] = {}

            layout = QVBoxLayout(self)

            # 0. Instrument Profile Selector (Default: GruLab SURF 8-18 GHz Cavity FTMW)
            profile_group = QGroupBox("Instrument Profile")
            profile_layout = QHBoxLayout()
            self.profile_selector = QComboBox()
            for pid in list_instrument_profiles():
                self.profile_selector.addItem(format_instrument_profile_label(pid), pid)
            default_idx = self.profile_selector.findData(self._controller.instrument_profile)
            if default_idx >= 0:
                self.profile_selector.setCurrentIndex(default_idx)
            self.lbl_profile_summary = QLabel(_format_profile_summary(self._controller.profile))
            self.profile_selector.currentIndexChanged.connect(self._on_profile_index_changed)

            profile_layout.addWidget(QLabel("Active Instrument:"))
            profile_layout.addWidget(self.profile_selector)
            profile_layout.addWidget(self.lbl_profile_summary, stretch=1)
            profile_group.setLayout(profile_layout)
            layout.addWidget(profile_group)

            # 1. Experimental Ingestion & Fitting
            ingestion_group = QGroupBox("Bayesian Optimizer (Levenberg-Marquardt)")
            ingestion_layout = QHBoxLayout()
            self.btn_load_lin = QPushButton("Load Experimental (.lin)")
            self.btn_optimize = QPushButton("Run Optimizer")

            # Resolves SPYCFIT-14: Register button click callbacks
            self.btn_load_lin.clicked.connect(self.load_lin_file)
            self.btn_optimize.clicked.connect(self.run_optimization)

            self.lbl_status = QLabel("Status: Idle.")

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
            self.table.setHorizontalHeaderLabels(
                ["Transition", "Theoretical Intensity", "Prediction Error (MHz)"]
            )
            table_layout.addWidget(self.table)
            table_group.setLayout(table_layout)
            layout.addWidget(table_group, stretch=1)

            # 3. Interactive SURF User Parameter Override Table (HITL Gate)
            surf_group = QGroupBox("SURF Scan Parameter Override Table (HITL Gate)")
            surf_layout = QVBoxLayout()
            self.surf_override_table = QTableWidget(0, 5)
            self.surf_override_table.setHorizontalHeaderLabels(
                ["Group", "Parameter", "Proposed", "Override Value", "Bounds / Units"]
            )
            self.surf_override_table.itemChanged.connect(self._on_override_cell_changed)
            surf_layout.addWidget(self.surf_override_table)

            surf_controls = QHBoxLayout()
            self.btn_export_surf = QPushButton("Export Native SURF Files...")
            self.btn_reset_surf_overrides = QPushButton("Reset Overrides")
            self.btn_export_surf.clicked.connect(self._on_export_surf_clicked)
            self.btn_reset_surf_overrides.clicked.connect(self.reset_surf_overrides)
            self.lbl_surf_runtime = QLabel("")
            self.lbl_surf_warnings = QLabel("")
            self.lbl_surf_override_status = QLabel("SURF HITL Gate: Ready (zero auto-actuation).")

            surf_controls.addWidget(self.btn_export_surf)
            surf_controls.addWidget(self.btn_reset_surf_overrides)
            surf_controls.addWidget(self.lbl_surf_runtime)
            surf_controls.addWidget(self.lbl_surf_warnings, stretch=1)
            surf_layout.addLayout(surf_controls)
            surf_layout.addWidget(self.lbl_surf_override_status)
            surf_group.setLayout(surf_layout)
            layout.addWidget(surf_group, stretch=1)

            # 4. LOIO Cross-Validation Box & Whisker Plot
            # Resolves SPYCFIT-16: Plotly / WebEngine box-and-whisker plot view
            loio_group = QGroupBox("LOIO Cross-Validation (Leverage Analysis)")
            loio_layout = QVBoxLayout()

            if HAS_WEBENGINE:
                self.web_view = QWebEngineView()
                self.web_view.setHtml(self._generate_plotly_box_plot_html([]))
                loio_layout.addWidget(self.web_view)
            else:
                self.lbl_loio = QLabel(
                    "LOIO Cross-Validation Residuals Distribution\n"
                    "[Plotly HTML Active - Box & Whisker Display]"
                )
                self.lbl_loio.setAlignment(Qt.AlignCenter)
                self.lbl_loio.setStyleSheet(
                    "background-color: #1e1e1e; color: #00ffcc; font-weight: bold; padding: 15px;"
                )
                loio_layout.addWidget(self.lbl_loio)

            loio_group.setLayout(loio_layout)
            layout.addWidget(loio_group, stretch=1)

            layout.addStretch()
            self._refresh_surf_override_table()

        @property
        def surf_override_model(self) -> SurfParameterOverrideTable:
            """Return underlying ``SurfParameterOverrideTable`` state model."""
            return self._controller.override_table

        @property
        def last_override_error(self) -> Optional[str]:
            """Return last override validation error message, if any."""
            return self._controller.last_error

        def selected_instrument_profile(self) -> str:
            """Return currently selected instrument profile ID."""
            return self._controller.instrument_profile

        def set_instrument_profile(self, profile_id: str) -> InstrumentProfile:
            """Programmatically select an instrument profile by ID."""
            profile = self._controller.select_instrument_profile(profile_id)
            idx = self.profile_selector.findData(profile.profile_id)
            if idx >= 0 and idx != self.profile_selector.currentIndex():
                self.profile_selector.setCurrentIndex(idx)
            self.lbl_profile_summary.setText(_format_profile_summary(profile))
            self.instrument_profile_changed.emit(profile.profile_id)
            return profile

        def _on_profile_index_changed(self, index: int) -> None:
            pid = self.profile_selector.itemData(index)
            if isinstance(pid, str):
                profile = self._controller.select_instrument_profile(pid)
                self.lbl_profile_summary.setText(_format_profile_summary(profile))
                self.instrument_profile_changed.emit(profile.profile_id)

        def load_surf_override_table(self, table: SurfParameterOverrideTable) -> None:
            """Load a prepared ``SurfParameterOverrideTable`` into the Qt widget."""
            self._controller.load_override_table(table)
            self._refresh_surf_override_table()

        def load_surf_proposal(self, proposal: SurfScanProposal) -> SurfParameterOverrideTable:
            """Populate the override table from a planned ``SurfScanProposal``."""
            table = self._controller.load_proposal(proposal)
            self._refresh_surf_override_table()
            return table

        def _refresh_surf_override_table(self) -> None:
            """Synchronize ``QTableWidget`` cells and status labels with the state model."""
            self._populating_override_table = True
            try:
                rows = self._controller.override_table.to_table_rows()
                self.surf_override_table.setRowCount(len(rows))
                self._override_row_by_field.clear()

                for row_idx, row_data in enumerate(rows):
                    field_path = str(row_data["field_path"])
                    self._override_row_by_field[field_path] = row_idx

                    group_item = QTableWidgetItem(str(row_data["group"]))
                    param_item = QTableWidgetItem(f"{row_data['label']} ({field_path})")
                    param_item.setData(Qt.ItemDataRole.UserRole, field_path)
                    proposed_item = QTableWidgetItem(_format_cell_value(row_data["proposed_value"]))
                    override_item = QTableWidgetItem(_format_cell_value(row_data["current_value"]))
                    bounds_text = str(row_data["bounds_summary"])
                    if row_data["unit"]:
                        bounds_text = f"{bounds_text} [{row_data['unit']}]"
                    bounds_item = QTableWidgetItem(bounds_text)

                    for readonly_item in (group_item, param_item, proposed_item, bounds_item):
                        readonly_item.setFlags(readonly_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                    self.surf_override_table.setItem(row_idx, 0, group_item)
                    self.surf_override_table.setItem(row_idx, 1, param_item)
                    self.surf_override_table.setItem(row_idx, 2, proposed_item)
                    self.surf_override_table.setItem(row_idx, 3, override_item)
                    self.surf_override_table.setItem(row_idx, 4, bounds_item)

                self._refresh_surf_summary_labels()
            finally:
                self._populating_override_table = False

        def _refresh_surf_summary_labels(self) -> None:
            """Update estimated runtime and soft-warning labels."""
            est = self._controller.override_table.estimated_duration
            self.lbl_surf_runtime.setText(
                f"Est. SURF Runtime: {est.total_minutes:.2f} min ({est.iterations} steps)"
            )
            warnings = self._controller.override_table.warnings
            if warnings:
                self.lbl_surf_warnings.setText("WARNING: " + " | ".join(warnings))
            else:
                self.lbl_surf_warnings.setText("Warnings: None")

        def _on_override_cell_changed(self, item: QTableWidgetItem) -> None:
            """Validate edited cells through ``SurfParameterOverrideTable``."""
            if self._populating_override_table or item.column() != 3:
                return
            row_idx = item.row()
            param_item = self.surf_override_table.item(row_idx, 1)
            if param_item is None:
                return
            field_path = param_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(field_path, str):
                return

            ok, status_msg = self._controller.apply_cell_edit(field_path, item.text())
            self.lbl_surf_override_status.setText(status_msg)
            self._populating_override_table = True
            try:
                current_val = self._controller.override_table.get_current_value(field_path)
                item.setText(_format_cell_value(current_val))
            finally:
                self._populating_override_table = False
            if ok:
                self._refresh_surf_summary_labels()

        def apply_override_from_ui(
            self,
            field_path: str,
            raw_value: Any,
            *,
            edited_by: Optional[str] = None,
        ) -> Tuple[bool, str]:
            """Apply an override through the UI model and synchronize table display."""
            ok, msg = self._controller.apply_cell_edit(field_path, raw_value, edited_by=edited_by)
            self.lbl_surf_override_status.setText(msg)
            self._refresh_surf_override_table()
            return ok, msg

        def reset_surf_overrides(self) -> None:
            """Reset all user overrides back to ML/physics-proposed baseline values."""
            self._controller.override_table.reset_all()
            self.lbl_surf_override_status.setText("SURF HITL Gate: Overrides reset to proposed defaults.")
            self._refresh_surf_override_table()

        def _on_export_surf_clicked(self) -> None:
            """Interactive button handler for exporting native SURF files."""
            if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
                self.lbl_surf_override_status.setText(
                    "Export skipped in offscreen mode without explicit output_dir."
                )
                return
            self.export_surf_files()

        def export_surf_files(
            self,
            output_dir: Optional[Union[str, Path]] = None,
            *,
            region_stem: Optional[str] = None,
            fit_state_id: Optional[str] = None,
            spcat_catalog_sha256: Optional[str] = None,
            gp_hyperparameters: Optional[Dict[str, Any]] = None,
        ) -> Optional[SurfExportResult]:
            """Export validated SURF TOMLs and strict manifest JSON from the HITL table."""
            resolved_dir: Optional[Union[str, Path]] = output_dir
            if resolved_dir is None:
                if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
                    raise ValueError("output_dir must be specified when exporting in headless/offscreen mode.")
                default_dir = os.environ.get("COCHEM_ARTIFACTS", str(Path.home()))
                selected = QFileDialog.getExistingDirectory(self, "Select SURF Export Directory", default_dir)
                if not selected:
                    return None
                resolved_dir = selected

            result = self._controller.export_surf_files(
                output_dir=resolved_dir,
                region_stem=region_stem,
                fit_state_id=fit_state_id,
                spcat_catalog_sha256=spcat_catalog_sha256,
                gp_hyperparameters=gp_hyperparameters,
            )
            self.lbl_surf_override_status.setText(
                f"Exported {result.region_stem} (config SHA-256: {result.control_config_sha256[:12]}...)"
            )
            self.surf_export_completed.emit(result)
            return result

        def load_lin_file(self) -> None:
            """Resolves SPYCFIT-14: Handler for loading Pickett .lin file."""
            default_dir = os.environ.get("COCHEM_ARTIFACTS", str(Path.home()))
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Pickett .lin File", default_dir, "Pickett Files (*.lin);;All Files (*)"
            )
            if file_path:
                self.lbl_status.setText(f"Loaded: {file_path}")

        def run_optimization(self) -> None:
            """Resolves SPYCFIT-14: Handler for launching optimization."""
            self.lbl_status.setText("Status: Optimization running (Levenberg-Marquardt)...")
            self.optimization_requested.emit()

        def update_active_learning_table(
            self, transitions_data: List[Union[dict, Tuple[str, Any, Any], List[Any]]]
        ) -> None:
            """Resolves SPYCFIT-15: Dynamically populates QTableWidget from active fit data arrays."""
            self.table.setRowCount(len(transitions_data))
            for row, item in enumerate(transitions_data):
                if isinstance(item, (tuple, list)):
                    if len(item) >= 3:
                        try:
                            trans_str = str(item[0])
                            int_float = float(item[1])
                            err_float = float(item[2])
                        except (ValueError, TypeError):
                            continue
                    else:
                        continue
                else:
                    try:
                        parsed = TransitionData(**item)
                        trans_str = parsed.transition
                        int_float = parsed.theoretical_intensity
                        err_float = parsed.prediction_error
                    except ValidationError as exc:
                        logging.error(f"Validation error: {exc}")
                        continue

                self.table.setItem(row, 0, QTableWidgetItem(trans_str))
                self.table.setItem(row, 1, QTableWidgetItem(f"{int_float:.4f}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{err_float:.4f}"))

        def _generate_plotly_box_plot_html(self, data_traces: List[Dict[str, Any]]) -> str:
            """Resolves SPYCFIT-16: HTML rendering Plotly Box & Whisker residual distribution."""
            validated_traces = []
            for trace in data_traces:
                try:
                    if "type" not in trace:
                        trace["type"] = "box"
                    valid_trace = PlotlyTrace(**trace)
                    validated_traces.append(valid_trace.model_dump())
                except ValidationError as exc:
                    logging.warning(f"Trace validation failed, ignoring: {exc}")

            data_json = json.dumps(validated_traces)
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


__all__ = [
    "HAS_PYSIDE6",
    "HAS_WEBENGINE",
    "INSTRUMENT_PROFILE_LABELS",
    "PlotlyTrace",
    "SpycFitTab",
    "SurfOverridePanelController",
    "TransitionData",
    "format_instrument_profile_label",
]
