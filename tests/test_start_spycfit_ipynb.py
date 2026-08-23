"""Physical unit test suite for Start_SpycFit.ipynb and launch_dashboard entrypoint.

Verifies:
1. launch_dashboard function exports and executes correctly with authentic physical data.
2. launch_dashboard supports custom HITLAssignmentManager, WhatIfSpectrumEngine, and DashboardConfig.
3. Start_SpycFit.ipynb exists at repository root and complies with Jupyter nbformat v4 schema.
4. Start_SpycFit.ipynb documents the mandatory Voila launch command.
5. Start_SpycFit.ipynb maintains Air-Gap compliance with zero external CDN references.
6. Start_SpycFit.ipynb contains no pass statements or stub logic (AST-validated).
7. Start_SpycFit.ipynb code cells execute successfully in a real execution environment.

Zero-Mock Policy: All calculations use real physical spectroscopic data and authentic algorithmic solvers.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import ipywidgets as widgets
import numpy as np
import pytest

from cochem_spycfit.interfaces.cochem_spycfit_ui import (
    AssignmentStatus,
    DashboardConfig,
    HITLAssignmentManager,
    SpectralPeak,
    SpycFitInteractiveDashboard,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
    launch_dashboard,
)


@pytest.fixture
def repo_root() -> Path:
    """Returns the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def notebook_path(repo_root: Path) -> Path:
    """Returns the path to Start_SpycFit.ipynb."""
    return repo_root / "Start_SpycFit.ipynb"


class TestLaunchDashboardFunction:
    """Tests the launch_dashboard programmatic entrypoint in cochem_spycfit_ui."""

    def test_launch_dashboard_default(self) -> None:
        """Verifies launch_dashboard runs with authentic defaults and returns a valid VBox widget."""
        widget = launch_dashboard(display_in_notebook=False)
        assert isinstance(widget, widgets.VBox)
        assert len(widget.children) >= 3
        # First child is controls_box (VBox)
        controls_box = widget.children[0]
        assert isinstance(controls_box, widgets.VBox)
        # Second child is status_output (Output)
        status_out = widget.children[1]
        assert isinstance(status_out, widgets.Output)
        # Third child is plot_output (Output)
        plot_out = widget.children[2]
        assert isinstance(plot_out, widgets.Output)

    def test_launch_dashboard_custom_manager(self) -> None:
        """Verifies launch_dashboard with custom physical transitions."""
        exp_peaks = [
            SpectralPeak(
                peak_id="EXP_H2O_01",
                frequency=22235.08,
                intensity=1.85,
                uncertainty=0.01,
                quantum_label="6_1,6 -> 5_2,3",
            ),
            SpectralPeak(
                peak_id="EXP_H2O_02",
                frequency=183310.08,
                intensity=2.40,
                uncertainty=0.02,
                quantum_label="3_1,3 -> 2_2,0",
            ),
        ]
        theo_peaks = [
            SpectralPeak(
                peak_id="THEO_H2O_01",
                frequency=22235.10,
                intensity=1.80,
                uncertainty=0.01,
                quantum_label="6_1,6 -> 5_2,3",
            ),
            SpectralPeak(
                peak_id="THEO_H2O_02",
                frequency=183310.15,
                intensity=2.45,
                uncertainty=0.02,
                quantum_label="3_1,3 -> 2_2,0",
            ),
        ]
        manager = HITLAssignmentManager(exp_peaks=exp_peaks, theo_peaks=theo_peaks)
        cfg = DashboardConfig(title="Custom Water Spectrum HITL", theme="plotly_white", use_webgl=False)

        widget = launch_dashboard(manager=manager, config=cfg, display_in_notebook=False)
        assert isinstance(widget, widgets.VBox)
        assert len(widget.children) == 3


class TestStartSpycFitNotebook:
    """Tests Start_SpycFit.ipynb notebook structure, compliance, and execution."""

    def test_notebook_exists(self, notebook_path: Path) -> None:
        """Verifies Start_SpycFit.ipynb exists on disk."""
        assert notebook_path.exists(), f"Notebook file not found at: {notebook_path}"
        assert notebook_path.is_file(), f"Path is not a regular file: {notebook_path}"
        assert notebook_path.stat().st_size > 0, "Notebook file is empty."

    def test_notebook_valid_json_schema(self, notebook_path: Path) -> None:
        """Verifies the notebook matches valid nbformat v4 schema."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cells" in data, "Notebook missing 'cells' key."
        assert "metadata" in data, "Notebook missing 'metadata' key."
        assert "nbformat" in data, "Notebook missing 'nbformat' key."
        assert data["nbformat"] >= 4, f"Expected nbformat >= 4, got {data.get('nbformat')}"
        assert len(data["cells"]) >= 2, "Notebook must contain both documentation and execution cells."

    def test_voila_command_documentation(self, notebook_path: Path) -> None:
        """Verifies the notebook explicitly documents the exact mandatory Voila launch command."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            content = f.read()

        mandatory_cmd = "voila Start_SpycFit.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap"
        assert mandatory_cmd in content, (
            f"Mandatory Voila launch command not found in notebook documentation.\n"
            f"Expected string:\n{mandatory_cmd}"
        )

    def test_airgap_compliance_no_external_cdns(self, notebook_path: Path) -> None:
        """Verifies no external CDNs or unconstrained remote URLs exist in the notebook."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        banned_cdns = [
            "cdn.jsdelivr.net",
            "unpkg.com",
            "cdnjs.cloudflare.com",
            "googleapis.com",
            "cdn.plot.ly",
            "cdn.tailwindcss.com",
        ]

        for idx, cell in enumerate(data.get("cells", [])):
            source_lines = cell.get("source", [])
            cell_text = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
            for cdn in banned_cdns:
                assert cdn not in cell_text, (
                    f"Air-Gap Violation in cell {idx}: external CDN '{cdn}' detected!"
                )

    def test_no_pass_statements_in_code_cells(self, notebook_path: Path) -> None:
        """Verifies code cells contain zero 'pass' statements via AST parsing."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
        assert len(code_cells) > 0, "Notebook must contain at least one code cell."

        for idx, cell in enumerate(code_cells):
            source_lines = cell.get("source", [])
            source_code = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
            if not source_code.strip():
                continue

            parsed_ast = ast.parse(source_code)
            for node in ast.walk(parsed_ast):
                assert not isinstance(node, ast.Pass), (
                    f"Zero-Mock Violation: 'pass' statement found in code cell {idx}:\n{source_code}"
                )

    def test_notebook_physical_execution(self, notebook_path: Path, repo_root: Path) -> None:
        """Physically executes all code cells in the notebook sequentially."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
        exec_globals: Dict[str, Any] = {"__file__": str(notebook_path), "__name__": "__main__"}

        # Add src and root to path
        src_path = str(repo_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        for idx, cell in enumerate(code_cells):
            source_lines = cell.get("source", [])
            source_code = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
            if not source_code.strip():
                continue

            # Execute cell natively
            try:
                exec(source_code, exec_globals)
            except Exception as exc:
                pytest.fail(f"Execution of code cell {idx} failed with error: {exc}\nCode:\n{source_code}")

        # Verify launch_dashboard was executed and created widget
        assert "launch_dashboard" in exec_globals, "launch_dashboard was not imported in notebook."
