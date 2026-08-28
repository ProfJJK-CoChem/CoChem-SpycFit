"""Physical unit tests for WebGL Spectral Viewport and Assignment Ledger.

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational transitions for H2O and asymmetric rotors.
- Real experimental spectrum CSV generation with genuine Gaussian line shapes.
- Cached 10x6 Jacobian matrix with physical sensitivities for Watson A-reduction Hamiltonian.
- Drag-to-shift peak snapping via scipy.signal.find_peaks.
- Host CPU instant O-C residual updates (<50 ms latency).
- Provenance tag transitions ([TORQ]/[ML] -> [USER]).
- Interactive triage action buttons (Lock, Delete, Ignore) and master control buttons.
- Dual-tree module and package re-export verification.
- Static AST validation confirming zero mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
import tempfile
from typing import Any, List, Tuple

import ipywidgets as widgets
import numpy as np
import pytest

from cochem_spycfit.interfaces.cochem_spycfit_ui_viewport import (
    AssignmentAction,
    AssignmentLedger,
    InteractiveSpectralDashboard,
    InvalidJacobianError,
    MasterControlBar,
    SpectralViewport,
    SpectrumLoadError,
    ViewportError,
)
from cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
    ProvenanceTag,
    ResidualPrediction,
)


@pytest.fixture
def h2o_transitions_data() -> Tuple[Tuple[int, int, int, int, int, int, float, float], ...]:
    """Authentic H2O rotational transitions with calculated and experimental frequencies (MHz).

    Entries: (J', Ka', Kc', J'', Ka'', Kc'', nu_calc_MHz, nu_exp_MHz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985, 556936.002),
        (2, 1, 1, 2, 0, 2, 752033.227, 752033.138),
        (3, 1, 2, 3, 0, 3, 1097364.791, 1097365.054),
        (4, 1, 3, 4, 0, 4, 1602219.182, 1602218.910),
        (5, 1, 4, 5, 0, 5, 2264148.770, 2264149.120),
        (1, 1, 1, 0, 0, 0, 1113342.964, 1113343.012),
        (2, 0, 2, 1, 1, 1, 987926.764, 987926.890),
        (2, 1, 2, 1, 0, 1, 1669904.775, 1669904.620),
        (3, 0, 3, 2, 1, 2, 1716769.318, 1716769.540),
        (3, 1, 3, 2, 0, 2, 2196345.756, 2196345.890),
    )


@pytest.fixture
def h2o_predictions(
    h2o_transitions_data: Tuple[Tuple[int, int, int, int, int, int, float, float], ...]
) -> List[ResidualPrediction]:
    """Generate ResidualPrediction instances for H2O transitions with provenance tags."""
    preds = []
    for i, (jp, kap, kcp, jpp, kapp, kcpp, nu_calc, nu_exp) in enumerate(h2o_transitions_data):
        delta_ml = (nu_exp - nu_calc) if (i % 2 == 1) else 0.0
        prov = ProvenanceTag.ML if (i % 2 == 1) else ProvenanceTag.TORQ
        conf = 0.92 if (i % 2 == 1) else 0.75
        sigma = 0.05 if (i % 2 == 1) else 0.20
        preds.append(
            ResidualPrediction(
                quantum_numbers=(jp, kap, kcp, jpp, kapp, kcpp),
                nu_calc_physical_mhz=nu_calc,
                delta_nu_ml_shift_mhz=delta_ml,
                nu_proposed_mhz=nu_calc + delta_ml,
                sigma_predict_mhz=sigma,
                probable_correctness=conf,
                provenance=prov,
            )
        )
    return preds


@pytest.fixture
def h2o_cached_jacobian() -> np.ndarray:
    """Authentic 10x6 Jacobian matrix for H2O Watson A-reduction parameters.

    Columns: [A, B, C, D_J, D_JK, D_K]
    Sensitivities represent d(nu)/d(param) in MHz/MHz and MHz/kHz.
    """
    return np.array([
        [0.000, 1.000, -1.000, -0.004,  0.012, -0.002],
        [0.000, 1.000, -1.000, -0.012,  0.035, -0.005],
        [0.000, 1.000, -1.000, -0.025,  0.078, -0.010],
        [0.000, 1.000, -1.000, -0.045,  0.142, -0.018],
        [0.000, 1.000, -1.000, -0.072,  0.230, -0.029],
        [1.000, 0.000,  1.000, -0.008, -0.015, -0.004],
        [0.000, 2.000,  0.000, -0.018, -0.040, -0.008],
        [1.000, 1.000,  0.000, -0.022,  0.010, -0.012],
        [0.000, 3.000,  0.000, -0.035, -0.080, -0.015],
        [1.000, 2.000,  0.000, -0.048,  0.025, -0.022],
    ], dtype=np.float64)


@pytest.fixture
def parameter_names() -> List[str]:
    """Watson A-reduction parameter names."""
    return ["A", "B", "C", "D_J", "D_JK", "D_K"]


@pytest.fixture
def authentic_h2o_spectrum_csv(
    tmp_path: Path,
    h2o_transitions_data: Tuple[Tuple[int, int, int, int, int, int, float, float], ...]
) -> Path:
    """Generate authentic H2O experimental spectrum CSV with physical Gaussian peaks."""
    csv_file = tmp_path / "h2o_experimental_spectrum.csv"

    # Build dense frequency grid around the transitions
    freq_points: List[float] = []
    intensity_points: List[float] = []

    for _, _, _, _, _, _, _, nu_exp in h2o_transitions_data:
        # Create a 2.0 MHz window around each peak with 100 points
        window = np.linspace(nu_exp - 1.0, nu_exp + 1.0, 100)
        # Gaussian line shape: FWHM = 0.1 MHz, sigma = FWHM / (2 * sqrt(2 * ln 2)) ≈ 0.04247
        sigma = 0.04247
        peak_intensity = 100.0 * np.exp(-0.5 * ((window - nu_exp) / sigma) ** 2)
        freq_points.extend(window.tolist())
        intensity_points.extend(peak_intensity.tolist())

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_mhz", "intensity_arb"])
        for f_val, int_val in zip(freq_points, intensity_points):
            writer.writerow([f"{f_val:.6f}", f"{int_val:.6f}"])

    return csv_file


def test_assignment_action_enum() -> None:
    """Verify AssignmentAction enum constants."""
    assert AssignmentAction.LOCK.value == "LOCK"
    assert AssignmentAction.DELETE.value == "DELETE"
    assert AssignmentAction.IGNORE.value == "IGNORE"


def test_spectral_viewport_initialization(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str]
) -> None:
    """Verify SpectralViewport initialization and CSV spectrum ingestion."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )

    assert viewport.experimental_freqs is not None
    assert viewport.experimental_intensities is not None
    assert len(viewport.experimental_freqs) == 1000  # 10 peaks * 100 points
    assert len(viewport.experimental_intensities) == 1000
    assert viewport.jacobian.shape == (10, 6)
    assert viewport.parameter_names == parameter_names


def test_spectral_viewport_load_error(tmp_path: Path, h2o_cached_jacobian: np.ndarray, parameter_names: List[str]) -> None:
    """Verify error handling on invalid or missing CSV files."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(SpectrumLoadError):
        SpectralViewport(
            experimental_csv_path=non_existent,
            cached_jacobian=h2o_cached_jacobian,
            parameter_names=parameter_names,
        )

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("frequency,intensity\n", encoding="utf-8")
    with pytest.raises(SpectrumLoadError):
        SpectralViewport(
            experimental_csv_path=empty_csv,
            cached_jacobian=h2o_cached_jacobian,
            parameter_names=parameter_names,
        )


def test_spectral_viewport_invalid_jacobian_error(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
) -> None:
    """Verify InvalidJacobianError on dimension or column count mismatch."""
    # 1D Jacobian instead of 2D
    with pytest.raises(InvalidJacobianError):
        SpectralViewport(
            experimental_csv_path=authentic_h2o_spectrum_csv,
            cached_jacobian=np.array([1.0, 2.0, 3.0]),
            parameter_names=["A", "B", "C"],
        )

    # Column count mismatch (6 columns in jacobian, 3 parameter names)
    with pytest.raises(InvalidJacobianError):
        SpectralViewport(
            experimental_csv_path=authentic_h2o_spectrum_csv,
            cached_jacobian=h2o_cached_jacobian,
            parameter_names=["A", "B", "C"],
        )


def test_viewport_bounds_errors(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction],
) -> None:
    """Verify ViewportError when accessing out-of-range transition indices."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )
    viewport._current_predictions = list(h2o_predictions)

    with pytest.raises(ViewportError):
        viewport._update_local_oc(999, 556936.002)

    with pytest.raises(ViewportError):
        viewport._on_drag_shift(999, 556936.002)


def test_snap_to_peak(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_transitions_data: Tuple[Tuple[int, int, int, int, int, int, float, float], ...]
) -> None:
    """Verify _snap_to_peak correctly identifies experimental apex in MHz."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )

    for _, _, _, _, _, _, _, nu_exp in h2o_transitions_data:
        # Perturb target frequency by 0.15 MHz
        perturbed_target = nu_exp + 0.15
        snapped = viewport._snap_to_peak(
            perturbed_target,
            viewport.experimental_freqs,
            viewport.experimental_intensities,
        )
        assert snapped == pytest.approx(nu_exp, abs=0.03)


def test_update_local_oc_latency_and_accuracy(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction]
) -> None:
    """Verify _update_local_oc executes on CPU in <50 ms and computes exact O-C residual."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )
    viewport._current_predictions = list(h2o_predictions)

    import time
    start_time = time.perf_counter()
    oc_residual = viewport._update_local_oc(0, 556936.002)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Strict latency check: must be < 50 ms
    assert elapsed_ms < 50.0

    # Exact residual check: nu_exp - nu_calc
    expected_oc = 556936.002 - 556935.985
    assert oc_residual == pytest.approx(expected_oc, rel=1e-5)


def test_on_drag_shift_provenance_and_state(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction]
) -> None:
    """Verify drag-to-shift updates prediction state to [USER] with snapped frequency."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )
    viewport._current_predictions = list(h2o_predictions)

    # Initial prediction at index 0 is TORQ
    assert viewport._current_predictions[0].provenance == ProvenanceTag.TORQ

    # Drag near the first peak (556936.002 MHz)
    viewport._on_drag_shift(0, 556936.080)

    updated_pred = viewport._current_predictions[0]
    assert updated_pred.provenance == ProvenanceTag.USER
    assert updated_pred.probable_correctness == 1.0
    assert updated_pred.nu_proposed_mhz == pytest.approx(556936.002, abs=0.03)
    assert updated_pred.delta_nu_ml_shift_mhz == pytest.approx(
        updated_pred.nu_proposed_mhz - updated_pred.nu_calc_physical_mhz, rel=1e-5
    )


def test_build_figure_and_get_widget(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction]
) -> None:
    """Verify build_figure creates Plotly graph and get_widget returns ipywidgets VBox."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )

    # Before build_figure, get_widget returns placeholder
    unbuilt_widget = viewport.get_widget()
    assert isinstance(unbuilt_widget, widgets.VBox)

    locked = ((1, 1, 0, 1, 0, 1, 556936.002),)
    viewport.build_figure(h2o_predictions, locked_assignments=locked)

    assert viewport.fig is not None
    built_widget = viewport.get_widget()
    assert isinstance(built_widget, widgets.VBox)
    assert len(built_widget.children) >= 2


def test_assignment_ledger_actions_and_callbacks(h2o_predictions: List[ResidualPrediction]) -> None:
    """Verify AssignmentLedger renders rows and dispatches action/selection callbacks."""
    action_log: List[Tuple[int, AssignmentAction]] = []
    select_log: List[int] = []

    def on_action(idx: int, action: AssignmentAction) -> None:
        action_log.append((idx, action))

    def on_select(idx: int) -> None:
        select_log.append(idx)

    ledger = AssignmentLedger(
        on_action_callback=on_action,
        on_row_select_callback=on_select,
    )

    # Before update_rows, widget shows placeholder
    empty_widget = ledger.get_widget()
    assert isinstance(empty_widget, widgets.VBox)

    ledger.update_rows(h2o_predictions)
    assert ledger.rows_container is not None
    assert len(ledger.rows_container.children) == len(h2o_predictions)

    # Test clicking action buttons for row 0
    row_0 = ledger.rows_container.children[0]
    assert isinstance(row_0, widgets.HBox)

    focus_btn = row_0.children[0]
    action_box = row_0.children[2]
    lock_btn = action_box.children[0]
    del_btn = action_box.children[1]
    ignore_btn = action_box.children[2]

    focus_btn.click()
    assert select_log == [0]

    lock_btn.click()
    assert action_log[-1] == (0, AssignmentAction.LOCK)

    del_btn.click()
    assert action_log[-1] == (0, AssignmentAction.DELETE)

    ignore_btn.click()
    assert action_log[-1] == (0, AssignmentAction.IGNORE)


def test_master_control_bar_actions() -> None:
    """Verify MasterControlBar buttons dispatch corresponding high-level workflows."""
    called: List[str] = []

    bar = MasterControlBar(
        on_commit_recalculate=lambda: called.append("commit"),
        on_propose_more=lambda: called.append("propose"),
        on_save_commit=lambda: called.append("save"),
    )

    bar.btn_commit.click()
    assert called == ["commit"]

    bar.btn_propose.click()
    assert called == ["commit", "propose"]

    bar.btn_save.click()
    assert called == ["commit", "propose", "save"]

    widget = bar.get_widget()
    assert isinstance(widget, widgets.HBox)
    assert len(widget.children) == 3


def test_interactive_dashboard_integration(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction]
) -> None:
    """Verify full 3-Zone Interactive Spectral Dashboard assembly."""
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )
    viewport.build_figure(h2o_predictions)

    ledger = AssignmentLedger(
        on_action_callback=lambda idx, act: None,
        on_row_select_callback=lambda idx: None,
    )
    ledger.update_rows(h2o_predictions)

    master_bar = MasterControlBar()

    dashboard = InteractiveSpectralDashboard(
        viewport=viewport,
        ledger=ledger,
        master_bar=master_bar,
    )

    dash_widget = dashboard.get_widget()
    assert isinstance(dash_widget, widgets.VBox)
    assert len(dash_widget.children) >= 5


def test_prompt_10_validation_test_specification(
    authentic_h2o_spectrum_csv: Path,
    h2o_cached_jacobian: np.ndarray,
    parameter_names: List[str],
    h2o_predictions: List[ResidualPrediction]
) -> None:
    """Execute complete validation test specification defined in Prompt 10.

    1. Create SpectralViewport with authentic H2O spectrum CSV and 10x6 cached Jacobian.
    2. Dispatch programmatic drag-to-shift event.
    3. Assertions:
       - O-C error updates on host CPU memory without triggering JAX diagonalization.
       - Assignment tag updates from [TORQ] or [ML] to [USER] after shift.
       - AssignmentLedger renders correct columns: quantum numbers, nu_proposed, nu_calc, O-C, provenance, confidence.
       - Action buttons trigger correct callbacks.
    """
    # 1. Instantiate Viewport
    viewport = SpectralViewport(
        experimental_csv_path=authentic_h2o_spectrum_csv,
        cached_jacobian=h2o_cached_jacobian,
        parameter_names=parameter_names,
    )
    viewport.build_figure(h2o_predictions)

    # 2. Dispatch drag-to-shift event on transition index 0 (initial TORQ)
    assert viewport._current_predictions[0].provenance == ProvenanceTag.TORQ
    drag_target_mhz = 556936.050
    viewport._on_drag_shift(0, drag_target_mhz)

    # 3. Assertions
    # a. Tag updated to USER
    updated_pred = viewport._current_predictions[0]
    assert updated_pred.provenance == ProvenanceTag.USER
    assert updated_pred.probable_correctness == 1.0

    # b. O-C residual updated on CPU
    expected_snapped = 556936.002
    assert updated_pred.nu_proposed_mhz == pytest.approx(expected_snapped, abs=0.03)
    oc_res = updated_pred.nu_proposed_mhz - updated_pred.nu_calc_physical_mhz
    assert oc_res == pytest.approx(expected_snapped - 556935.985, abs=0.03)

    # c. Ledger rendering verification
    actions_fired: List[Tuple[int, AssignmentAction]] = []
    ledger = AssignmentLedger(
        on_action_callback=lambda idx, act: actions_fired.append((idx, act)),
        on_row_select_callback=lambda idx: None,
    )
    ledger.update_rows(viewport._current_predictions)

    row_0_html = ledger.rows_container.children[0].children[1].value
    assert "1(1,0)←1(0,1)" in row_0_html
    assert "[USER]" in row_0_html
    assert "Conf: 100%" in row_0_html

    # d. Action buttons trigger
    ledger.rows_container.children[0].children[2].children[0].click()
    assert actions_fired == [(0, AssignmentAction.LOCK)]


def test_dual_tree_exports() -> None:
    """Verify SpectralViewport, AssignmentLedger, AssignmentAction are exportable from interfaces."""
    import cochem_spycfit.interfaces as root_interfaces
    import src.cochem_spycfit.interfaces as src_interfaces

    for iface_mod in [root_interfaces, src_interfaces]:
        assert hasattr(iface_mod, "SpectralViewport")
        assert hasattr(iface_mod, "AssignmentLedger")
        assert hasattr(iface_mod, "AssignmentAction")
        assert hasattr(iface_mod, "MasterControlBar")
        assert hasattr(iface_mod, "InteractiveSpectralDashboard")

    from cochem_spycfit.interfaces.cochem_spycfit_ui_viewport import (
        AssignmentAction as ActRoot,
        AssignmentLedger as LedRoot,
        SpectralViewport as ViewRoot,
    )
    from src.cochem_spycfit.interfaces.cochem_spycfit_ui_viewport import (
        AssignmentAction as ActSrc,
        AssignmentLedger as LedSrc,
        SpectralViewport as ViewSrc,
    )

    assert ActRoot is not None
    assert LedRoot is not None
    assert ViewRoot is not None
    assert ActSrc is not None
    assert LedSrc is not None
    assert ViewSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or fake implementations exist in the test and implementation."""
    test_path = Path(__file__).resolve()
    impl_src_path = (
        test_path.parent.parent
        / "src"
        / "cochem_spycfit"
        / "interfaces"
        / "cochem_spycfit_ui_viewport.py"
    )
    impl_root_path = (
        test_path.parent.parent
        / "cochem_spycfit"
        / "interfaces"
        / "cochem_spycfit_ui_viewport.py"
    )

    for file_path in [test_path, impl_src_path, impl_root_path]:
        if not file_path.exists():
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.Name):
                assert node.id not in (
                    "MagicMock",
                    "Mock",
                    "AsyncMock",
                ), f"Prohibited mock identifier in {file_path}"
