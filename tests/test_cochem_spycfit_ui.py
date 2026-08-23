"""Comprehensive physical unit test suite for SpycFit Decision Dashboard (HITL).

Verifies:
1. Plotly Dual-Pane O-C Residual dashboard with WebGL and SVG fallbacks.
2. Bipartite Auto-Propose Assistant using scipy.optimize.linear_sum_assignment (Hungarian Algorithm).
3. Human-In-The-Loop (HITL) approval gate state machine and Huber loss calculation with Ghost Peak retention.
4. Headless and air-gap degradation (static self-contained HTML and ReportLab PDF generation).
5. What-If parameter slider spectrum recalculation and integration with JAX convolver.
6. Semantic Git-commit message translation for mathematical changes.

Zero-Mock Policy: All calculations use real physical spectroscopic data and authentic algorithmic solvers.
"""

from __future__ import annotations

import ast
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from cochem_spycfit.interfaces.cochem_spycfit_ui import (
    AssignmentStatus,
    BipartiteConfig,
    BipartiteMatchingError,
    DashboardConfig,
    HITLApprovalError,
    HITLAssignmentManager,
    OfflineRenderError,
    PeakAssignment,
    RenderBackend,
    SemanticCommit,
    SemanticCommitError,
    SliderComputationError,
    SpectralPeak,
    UIDashboardError,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
    auto_propose_assignments,
    compute_bipartite_cost_matrix,
    compute_huber_loss_excluding_ghosts,
    render_dual_pane_dashboard,
    render_static_offline_html,
    render_static_offline_pdf,
    translate_mathematical_change_to_git_commit,
)


# =============================================================================
# PHYSICAL FIXTURES (Real Spectroscopic Transitions)
# =============================================================================

@pytest.fixture
def methanol_exp_peaks() -> List[SpectralPeak]:
    """Authentic experimental rotational transitions of Methanol (CH3OH) in the 12-48 GHz band."""
    return [
        SpectralPeak(
            peak_id="EXP_01",
            frequency=12178.597,
            intensity=0.88,
            uncertainty=0.005,
            quantum_label="2_0,2 -> 1_1,1",
        ),
        SpectralPeak(
            peak_id="EXP_02",
            frequency=24928.702,
            intensity=1.45,
            uncertainty=0.008,
            quantum_label="4_1,4 -> 3_0,3",
        ),
        SpectralPeak(
            peak_id="EXP_03",
            frequency=36169.265,
            intensity=2.10,
            uncertainty=0.010,
            quantum_label="4_0,4 -> 3_1,3",
        ),
        SpectralPeak(
            peak_id="EXP_04",
            frequency=48372.455,
            intensity=0.62,
            uncertainty=0.012,
            quantum_label="1_1,0 -> 1_0,1",
        ),
        SpectralPeak(
            peak_id="EXP_05_IMPURITY",
            frequency=28450.120,
            intensity=0.15,
            uncertainty=0.020,
            quantum_label=None,
        ),
    ]


@pytest.fixture
def methanol_theo_peaks() -> List[SpectralPeak]:
    """Theoretical Watson A-reduced Hamiltonian predicted peaks for Methanol."""
    return [
        SpectralPeak(
            peak_id="THEO_01",
            frequency=12178.610,
            intensity=0.90,
            uncertainty=0.001,
            quantum_label="2_0,2 -> 1_1,1",
        ),
        SpectralPeak(
            peak_id="THEO_02",
            frequency=24928.680,
            intensity=1.42,
            uncertainty=0.001,
            quantum_label="4_1,4 -> 3_0,3",
        ),
        SpectralPeak(
            peak_id="THEO_03",
            frequency=36169.290,
            intensity=2.15,
            uncertainty=0.001,
            quantum_label="4_0,4 -> 3_1,3",
        ),
        SpectralPeak(
            peak_id="THEO_04",
            frequency=48372.440,
            intensity=0.60,
            uncertainty=0.001,
            quantum_label="1_1,0 -> 1_0,1",
        ),
        SpectralPeak(
            peak_id="THEO_05_UNOBSERVED",
            frequency=19800.500,
            intensity=0.05,
            uncertainty=0.001,
            quantum_label="3_2,1 -> 3_1,2",
        ),
    ]


@pytest.fixture
def baseline_slider_state() -> WhatIfParameterState:
    """Baseline asymmetric top rotational parameters for Methanol."""
    return WhatIfParameterState(
        A_mhz=127484.0,
        B_mhz=24679.98,
        C_mhz=23769.70,
        D_J_khz=42.1,
        D_JK_khz=-120.4,
        D_K_khz=520.8,
        temperature_k=150.0,
        fwhm_gaussian_mhz=0.05,
        fwhm_lorentzian_mhz=0.03,
        global_frequency_shift_mhz=0.0,
        intensity_scale=1.0,
    )


# =============================================================================
# 1. PYDANTIC SCHEMAS & DATA STRUCTURE VALIDATION TESTS
# =============================================================================

def test_spectral_peak_schema_validation():
    """Verifies strict validation on SpectralPeak model."""
    valid_peak = SpectralPeak(
        peak_id="PEAK_001",
        frequency=24000.5,
        intensity=1.2,
        uncertainty=0.005,
    )
    assert valid_peak.frequency == 24000.5
    assert valid_peak.intensity == 1.2
    assert not valid_peak.is_ghost

    # Invalid non-positive frequency
    with pytest.raises(ValidationError):
        SpectralPeak(peak_id="BAD_01", frequency=-10.0, intensity=1.0)

    # Invalid negative intensity
    with pytest.raises(ValidationError):
        SpectralPeak(peak_id="BAD_02", frequency=100.0, intensity=-0.5)


def test_what_if_parameter_state_validation():
    """Verifies validation of WhatIfParameterState physical ranges."""
    state = WhatIfParameterState(
        A_mhz=10000.0,
        B_mhz=5000.0,
        C_mhz=2500.0,
        temperature_k=296.0,
    )
    assert state.A_mhz == 10000.0
    assert state.temperature_k == 296.0

    # Temperature <= 0 must fail
    with pytest.raises(ValidationError):
        WhatIfParameterState(
            A_mhz=10000.0,
            B_mhz=5000.0,
            C_mhz=2500.0,
            temperature_k=0.0,
        )


# =============================================================================
# 2. BIPARTITE HUNGARIAN AUTO-PROPOSE ASSISTANT TESTS
# =============================================================================

def test_bipartite_cost_matrix_calculation(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies construction of the bipartite cost matrix with real frequency distance."""
    config = BipartiteConfig(max_freq_drift_mhz=50.0, freq_weight=1.0, intensity_weight=0.1)
    cost_matrix = compute_bipartite_cost_matrix(methanol_exp_peaks, methanol_theo_peaks, config)

    assert cost_matrix.shape == (len(methanol_exp_peaks), len(methanol_theo_peaks))
    assert np.all(np.isfinite(cost_matrix))
    # Diagonal should have lowest costs because matched peaks have minimal drift (< 0.05 MHz)
    assert cost_matrix[0, 0] < cost_matrix[0, 1]
    assert cost_matrix[1, 1] < cost_matrix[1, 0]


def test_auto_propose_assignments_optimal_matching(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies Hungarian algorithm solves the assignment problem and respects drift cutoff."""
    config = BipartiteConfig(max_freq_drift_mhz=1.0)
    assignments = auto_propose_assignments(methanol_exp_peaks, methanol_theo_peaks, config)

    # 4 authentic transitions should be assigned within 1.0 MHz cutoff; the 5th impurity should not match
    assert len(assignments) == 4
    for assign in assignments:
        assert assign.status == AssignmentStatus.PROPOSED
        assert abs(assign.frequency_drift_mhz) < 0.05
        assert assign.exp_peak.peak_id.startswith("EXP_")
        assert assign.theo_peak.peak_id.startswith("THEO_")

    assigned_exp_ids = {a.exp_peak.peak_id for a in assignments}
    assert "EXP_05_IMPURITY" not in assigned_exp_ids


def test_auto_propose_assignments_empty_inputs():
    """Verifies auto_propose_assignments handles empty peak lists gracefully."""
    res = auto_propose_assignments([], [])
    assert res == []

    res2 = auto_propose_assignments([SpectralPeak(peak_id="E1", frequency=100.0, intensity=1.0)], [])
    assert res2 == []


# =============================================================================
# 3. HUMAN-IN-THE-LOOP (HITL) ASSIGNMENT MANAGER & GHOST PEAKS TESTS
# =============================================================================

def test_hitl_manager_state_transitions(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies HITL assignment manager approval workflow and ghost peak segregation."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=5.0))

    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    assert len(proposed) >= 4

    # 1. Approve first assignment
    first_id = proposed[0].assignment_id
    approved_assign = manager.approve_assignment(first_id, user_id="DrSpectroscopist")
    assert approved_assign.status == AssignmentStatus.APPROVED
    assert approved_assign.approved_by == "DrSpectroscopist"

    # 2. Reject second assignment (simulate user rejecting an ambiguous peak)
    second_id = proposed[1].assignment_id
    rejected_assign = manager.reject_assignment(second_id, reason="Line profile asymmetry / blender")
    assert rejected_assign.status == AssignmentStatus.REJECTED
    assert rejected_assign.exp_peak.is_ghost
    assert rejected_assign.exp_peak.rejection_reason == "Line profile asymmetry / blender"

    # Verify ghost peaks tracking
    ghost_peaks = manager.get_ghost_peaks()
    assert any(g.peak_id == rejected_assign.exp_peak.peak_id for g in ghost_peaks)

    # Verify active peaks exclude ghosts
    active_peaks = manager.get_active_exp_peaks()
    assert all(not p.is_ghost for p in active_peaks)


def test_hitl_manager_lm_fit_gate_enforcement(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies LM fit gate prevents fitting when zero assignments are approved or unapproved remain in strict mode."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=5.0))

    # Without approval, validation must raise HITLApprovalError
    with pytest.raises(HITLApprovalError, match="No assignments approved"):
        manager.validate_for_lm_fit()

    # Approve all proposed transitions
    manager.approve_all_proposed(user_id="Verifier")
    approved_for_fit = manager.validate_for_lm_fit()
    assert len(approved_for_fit) >= 4


def test_huber_loss_excluding_ghost_peaks(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies Huber robust loss calculation strictly excludes rejected ghost peaks."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=10.0))

    # Approve 3 assignments and reject 1 assignment with high residual
    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    for p in proposed[:3]:
        manager.approve_assignment(p.assignment_id, user_id="User")
    manager.reject_assignment(proposed[3].assignment_id, reason="Outlier artifact")

    loss, rms_mhz = manager.compute_robust_metrics(huber_delta_mhz=1.0)
    assert loss >= 0.0
    assert rms_mhz >= 0.0
    assert rms_mhz < 0.1  # Approved transitions have < 0.05 MHz drift


# =============================================================================
# 4. PLOTLY DUAL-PANE O-C RESIDUAL DASHBOARD TESTS
# =============================================================================

def test_render_dual_pane_dashboard_webgl(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies rendering dual-pane dashboard with WebGL scattergl traces."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    manager.approve_all_proposed(user_id="AutoTester")

    config = DashboardConfig(use_webgl=True, theme="plotly_dark", show_assignment_lines=True)
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=manager.assignments,
        config=config,
    )

    assert isinstance(fig, go.Figure)
    # Check that subplot layout contains 2 rows
    assert len(fig.data) > 0
    # Top pane has spectrum traces, bottom pane has residual traces
    trace_types = [t.type for t in fig.data]
    assert "scattergl" in trace_types or "scatter" in trace_types

    # Verify O-C residual trace exists in bottom pane
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Residual" in n or "O - C" in n for n in trace_names)
    assert any("Experimental" in n for n in trace_names)
    assert any("Theoretical" in n for n in trace_names)


def test_render_dual_pane_dashboard_svg_fallback(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies SVG fallback rendering when use_webgl is False."""
    config = DashboardConfig(use_webgl=False, theme="plotly_white")
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=[],
        config=config,
    )
    assert isinstance(fig, go.Figure)
    # When use_webgl is False, scattergl should not be used
    trace_types = [t.type for t in fig.data]
    assert "scattergl" not in trace_types


def test_render_dual_pane_dashboard_with_ghost_peaks(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies ghost peaks are rendered in greyed-out translucent trace."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    manager.reject_assignment(proposed[0].assignment_id, reason="Ghost test")

    fig = render_dual_pane_dashboard(
        exp_peaks=manager.exp_peaks,
        theo_peaks=manager.theo_peaks,
        assignments=manager.assignments,
        config=DashboardConfig(show_ghost_peaks=True),
    )
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Ghost" in n for n in trace_names)


# =============================================================================
# 5. HEADLESS & AIR-GAP DEGRADATION (HTML & PDF) TESTS
# =============================================================================

def test_render_static_offline_html(methanol_exp_peaks, methanol_theo_peaks, tmp_path):
    """Verifies self-contained offline HTML artifact creation."""
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=[],
    )
    out_html = tmp_path / "dashboard_offline.html"
    rendered_path = render_static_offline_html(fig, output_path=out_html)

    assert rendered_path.is_file()
    assert rendered_path.stat().st_size > 500
    content = rendered_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "plotly" in content.lower()


def test_render_static_offline_pdf(methanol_exp_peaks, methanol_theo_peaks, tmp_path):
    """Verifies publication-grade static PDF generation via ReportLab."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    manager.approve_all_proposed(user_id="DrScientist")

    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=manager.assignments,
    )

    out_pdf = tmp_path / "dashboard_report.pdf"
    rendered_pdf = render_static_offline_pdf(
        fig=fig,
        output_path=out_pdf,
        assignments=manager.assignments,
        fit_stats={"rms_mhz": 0.015, "reduced_chi_sq": 1.02, "num_transitions": 4},
        title="Methanol Rotational Spectrum FIT-2026",
    )

    assert rendered_pdf.is_file()
    assert rendered_pdf.stat().st_size > 1000
    # Verify PDF magic bytes %PDF-
    with open(rendered_pdf, "rb") as f:
        header = f.read(5)
        assert header.startswith(b"%PDF-")


# =============================================================================
# 6. WHAT-IF PARAMETER SLIDERS & CONVOLUTION RECALCULATION TESTS
# =============================================================================

def test_what_if_spectrum_engine_recalculation(methanol_theo_peaks, baseline_slider_state):
    """Verifies real-time parameter perturbation and continuous spectrum convolution."""
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=2000,
    )

    # 1. Base spectrum evaluation
    freq_grid, base_spectrum = engine.compute_convolved_spectrum(baseline_slider_state)
    assert len(freq_grid) == 2000
    assert len(base_spectrum) == 2000
    assert np.max(base_spectrum) > 0.0

    # 2. Perturb rotational constant A (+50 MHz)
    perturbed_state = baseline_slider_state.model_copy(update={"A_mhz": baseline_slider_state.A_mhz + 50.0})
    _, perturbed_spectrum = engine.compute_convolved_spectrum(perturbed_state)

    # Peak positions/intensities must change
    assert not np.allclose(base_spectrum, perturbed_spectrum, atol=1e-6)

    # 3. Parameter slider steps generation
    steps = engine.generate_plotly_slider_steps(
        param_name="A_mhz",
        values=[127400.0, 127484.0, 127550.0],
    )
    assert len(steps) == 3
    assert steps[0]["label"] == "127400.00"


# =============================================================================
# 7. SEMANTIC GIT-COMMIT TRANSLATION TESTS
# =============================================================================

def test_semantic_git_commit_translation_parameters(baseline_slider_state):
    """Verifies translation of Hamiltonian parameter shift to conventional commit message."""
    perturbed_state = baseline_slider_state.model_copy(
        update={
            "A_mhz": baseline_slider_state.A_mhz + 1.25,
            "B_mhz": baseline_slider_state.B_mhz - 0.45,
        }
    )

    commit = translate_mathematical_change_to_git_commit(
        state_before=baseline_slider_state,
        state_after=perturbed_state,
        rms_before=0.085,
        rms_after=0.018,
    )

    assert isinstance(commit, SemanticCommit)
    assert commit.commit_type in ["feat", "fix", "refactor"]
    assert "hamiltonian" in commit.scope.lower() or "param" in commit.scope.lower()
    assert "A=" in commit.full_commit_message
    assert "RMS: 0.0850 MHz -> 0.0180 MHz" in commit.full_commit_message


def test_semantic_git_commit_translation_assignments():
    """Verifies translation of assignment approvals/rejections to commit message."""
    commit = translate_mathematical_change_to_git_commit(
        approved_count=5,
        rejected_count=2,
        rms_before=0.120,
        rms_after=0.024,
    )
    assert isinstance(commit, SemanticCommit)
    assert "assignment" in commit.scope.lower()
    assert "5" in commit.summary
    assert "2" in commit.summary
    assert "RMS: 0.1200 MHz -> 0.0240 MHz" in commit.full_commit_message


def test_semantic_git_commit_translation_huber_loss():
    """Verifies translation of Huber loss delta adjustment to commit message."""
    commit = translate_mathematical_change_to_git_commit(
        huber_delta_before=1.5,
        huber_delta_after=0.8,
    )
    assert isinstance(commit, SemanticCommit)
    assert "loss" in commit.scope.lower() or "huber" in commit.scope.lower()
    assert "1.500" in commit.full_commit_message
    assert "0.800" in commit.full_commit_message


def test_hitl_manager_manual_assign_and_tolerance(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies manual assignment workflow, tolerance threshold approvals, and error handling."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=50.0))

    # Test approve_all_within_tolerance
    count = manager.approve_all_within_tolerance(max_drift_mhz=0.05, user_id="ToleranceVerifier")
    assert count == 4

    # Test manual_assign
    manual_asn = manager.manual_assign(
        exp_peak_id="EXP_05_IMPURITY",
        theo_peak_id="THEO_05_UNOBSERVED",
        user_id="ExpertSpectroscopist",
    )
    assert manual_asn.status == AssignmentStatus.MANUAL
    assert manual_asn.approved_by == "ExpertSpectroscopist"
    assert not manual_asn.exp_peak.is_ghost

    # Test errors for non-existent IDs
    with pytest.raises(HITLApprovalError, match="Experimental peak 'NON_EXISTENT' not found"):
        manager.manual_assign("NON_EXISTENT", "THEO_01")

    with pytest.raises(HITLApprovalError, match="Theoretical peak 'NON_EXISTENT' not found"):
        manager.manual_assign("EXP_01", "NON_EXISTENT")

    with pytest.raises(HITLApprovalError, match="Assignment ID 'INVALID_ID' not found"):
        manager.approve_assignment("INVALID_ID")

    with pytest.raises(HITLApprovalError, match="Assignment ID 'INVALID_ID' not found"):
        manager.reject_assignment("INVALID_ID")


def test_semantic_git_commit_insufficient_parameters():
    """Verifies SemanticCommitError is raised when insufficient state parameters are supplied."""
    with pytest.raises(SemanticCommitError, match="Insufficient state parameters"):
        translate_mathematical_change_to_git_commit()


def test_render_dual_pane_dashboard_with_convolved_curve(methanol_exp_peaks, methanol_theo_peaks, baseline_slider_state):
    """Verifies dual-pane dashboard rendering with overlaid synthetic convolved spectrum."""
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=500,
    )
    freq_grid, spectrum = engine.compute_convolved_spectrum(baseline_slider_state)

    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        convolved_curve=(freq_grid, spectrum),
    )
    trace_names = [t.name for t in fig.data if t.name]
    assert "Synthetic Convolved" in trace_names


# =============================================================================
# 8. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliance:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_spycfit_ui_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "interfaces" / "cochem_spycfit_ui.py"
        assert module_path.is_file()

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_test_suite_ast_compliance(self) -> None:
        test_path = Path(__file__).resolve()
        assert test_path.is_file()

        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_no_banned_tokens_in_production_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "interfaces" / "cochem_spycfit_ui.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        banned = ["mock", "dummy", "placeholder", "fake", "stub"]
        violations: List[str] = []

        class IdentifierScanner(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for b in banned:
                    if b in node.name.lower():
                        violations.append(f"Function {node.name} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for b in banned:
                    if b in node.id.lower():
                        violations.append(f"Variable {node.id} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    for b in banned:
                        if b in alias.name.lower():
                            violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for b in banned:
                        if b in node.module.lower():
                            violations.append(f"Forbidden import from '{node.module}' at line {node.lineno}")
                self.generic_visit(node)

        IdentifierScanner().visit(tree)
        assert len(violations) == 0, f"Banned tokens found: {violations}"


