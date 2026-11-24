"""Zero-mock end-to-end test suite for SPYC-SURF-02 planner, override table, exporter, and UI.

Verifies:
1. Authentic HCOOH-DME rotational constants, analytical Jacobian, and GP O-C residual
   predictions -> ``SmartScanNavigator`` (``grulab_surf``) -> ``ResolvabilityFilter``
   -> ``SurfScanPlanner`` (targeted confirmation windows & IG/min survey windows).
2. Exact SURF scan duration estimator matching ``SURF/spectrometer.py:168-181``.
3. Optional cavity map CSV (``CavityRuns/<freq>MHz.csv``) resonant Zaber lookup with
   source SHA-256 provenance.
4. Headless HITL ``SurfParameterOverrideTable`` across all 24 Section 2.3 parameters:
   atomic Pydantic hard-bounds validation, out-of-bounds ``ValidationError`` rejection,
   ``rf_level > 10`` dBm soft warning, JSON/TOML override file round-trip, and exact
   ``SurfOverrideRecord`` audit diffs.
5. ``SurfExporter`` emitting ``*_control_config.toml``, ``*_scan_settings.toml``, and
   ``*_surf_manifest.json`` cross-validated with ``tomllib`` against upstream
   ``SURF/config.py:8-61`` and ``SURF/frequency_scan_settings.py:4-52`` rules.
6. ``SpycFitTab`` (PySide6 offscreen ``QTableWidget`` + profile selector + export trigger)
   and ``SpycFitInteractiveDashboard`` profile selector & override export wiring.
7. Strict UTF-8 LF line endings, zero ``pass`` statements, and strict anti-spoof cleanliness.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from ci_tools.anti_spoof_linter import check_script
from cochem_spycfit.core_engine.cochem_spycfit_state import FitState
from cochem_spycfit.gui import (
    HAS_PYSIDE6,
    INSTRUMENT_PROFILE_LABELS,
    SpycFitTab,
    SurfOverridePanelController,
)
from cochem_spycfit.instruments.profiles import (
    DEFAULT_INSTRUMENT_PROFILE,
    GRULAB_SURF_PROFILE,
    get_instrument_profile,
)
from cochem_spycfit.instruments.surf_exporter import SurfExporter
from cochem_spycfit.instruments.surf_planner import (
    SURF_OVERRIDE_FIELD_SPECS,
    SurfCavityMap,
    SurfParameterOverrideTable,
    SurfScanPlanner,
    estimate_surf_scan_duration,
    estimate_surf_scan_minutes,
)
from cochem_spycfit.instruments.surf_schemas import (
    SurfConfig,
    SurfFrequencyScanSettings,
    SurfScanManifest,
)
from cochem_spycfit.interfaces.cochem_spycfit_ui import (
    DashboardConfig,
    HITLAssignmentManager,
    SpectralPeak,
    SpycFitInteractiveDashboard,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
)
from cochem_spycfit.physics_core.cochem_assign_ml import (
    InformationCriterion,
    calculate_information_gain,
)
from cochem_spycfit.physics_core.cochem_spycfit_features import QuantumFeatureExtractor
from cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
    ProvenanceTag,
    SpycFitGPRegressor,
)
from cochem_spycfit.physics_core.cochem_spycfit_navigator import SmartScanNavigator
from cochem_spycfit.physics_core.cochem_spycfit_resolvability import ResolvabilityFilter

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Authentic HCOOH-DME Rotational Model (Prolate Asymmetric Top, 8-18 GHz Band)
# ---------------------------------------------------------------------------

HCOOH_DME_CONSTANTS: Dict[str, float] = {
    "A": 3826.420,
    "B": 1295.810,
    "C": 992.540,
    "DJ": 0.00048,
    "DJK": 0.00215,
}
HCOOH_DME_DIPOLES: Dict[str, float] = {
    "mu_a": 1.65,
    "mu_b": 0.92,
    "mu_c": 0.00,
}


def _compute_hcooh_dme_transition_and_jacobian(
    qn: Tuple[int, int, int, int, int, int],
    constants: Dict[str, float],
) -> Tuple[float, np.ndarray]:
    """Compute analytical transition frequency (MHz) and Jacobian wrt [A, B, C, DJ, DJK]."""
    j_u, ka_u, _kc_u, j_l, ka_l, _kc_l = qn
    a_c = constants["A"]
    b_c = constants["B"]
    c_c = constants["C"]
    dj = constants["DJ"]
    djk = constants["DJK"]

    half_bc = 0.5 * (b_c + c_c)

    def _energy(j_val: int, ka_val: int) -> float:
        jj1 = float(j_val * (j_val + 1))
        ka2 = float(ka_val * ka_val)
        return (
            half_bc * jj1
            + (a_c - half_bc) * ka2
            - dj * (jj1 * jj1)
            - djk * jj1 * ka2
        )

    nu = _energy(j_u, ka_u) - _energy(j_l, ka_l)

    d_jj1 = float(j_u * (j_u + 1) - j_l * (j_l + 1))
    d_ka2 = float(ka_u * ka_u - ka_l * ka_l)
    d_jj1_sq = float((j_u * (j_u + 1)) ** 2 - (j_l * (j_l + 1)) ** 2)
    d_jj1_ka2 = float(j_u * (j_u + 1) * (ka_u * ka_u) - j_l * (j_l + 1) * (ka_l * ka_l))

    d_nu_da = d_ka2
    d_nu_db = 0.5 * (d_jj1 - d_ka2)
    d_nu_dc = 0.5 * (d_jj1 - d_ka2)
    d_nu_ddj = -d_jj1_sq
    d_nu_ddjk = -d_jj1_ka2

    jac_row = np.array([d_nu_da, d_nu_db, d_nu_dc, d_nu_ddj, d_nu_ddjk], dtype=np.float64)
    return float(nu), jac_row


def _assert_matches_upstream_surf_config_contract(parsed_cfg: Dict[str, Any]) -> None:
    """Validate parsed TOML dict against every rule of ``SURF/src/spectrometer_gui/config.py:8-61``."""
    assert set(parsed_cfg.keys()) == {
        "valon_controller",
        "zaber_controller",
        "awg_controller",
        "oscilloscope_controller",
        "delay_generator_controller",
    }

    valon = parsed_cfg["valon_controller"]
    assert set(valon.keys()) == {
        "rf_output",
        "rf_level",
        "valon_port",
        "synth_power",
        "ref_source",
        "ref_freq",
        "freq",
    }
    assert type(valon["rf_output"]) is bool
    assert type(valon["rf_level"]) is int
    assert -20 <= valon["rf_level"] <= 20
    assert isinstance(valon["valon_port"], str) and len(valon["valon_port"]) > 0
    assert type(valon["synth_power"]) is bool
    assert valon["ref_source"] in {"Internal", "External"}
    assert isinstance(valon["ref_freq"], (int, float)) and valon["ref_freq"] > 0.0
    assert isinstance(valon["freq"], (int, float)) and 7970.0 <= float(valon["freq"]) <= 18000.0

    zaber = parsed_cfg["zaber_controller"]
    assert set(zaber.keys()) == {
        "zaber_scanning_speed",
        "zaber_moving_speed",
        "zaber_step_size",
        "zaber_port",
    }
    assert 0.001 <= float(zaber["zaber_scanning_speed"]) <= 2.0
    assert float(zaber["zaber_moving_speed"]) > 0.0
    assert float(zaber["zaber_step_size"]) > 0.0
    assert isinstance(zaber["zaber_port"], str) and len(zaber["zaber_port"]) > 0

    awg = parsed_cfg["awg_controller"]
    assert set(awg.keys()) == {
        "awg_status",
        "awg_freq",
        "awg_run_mode",
        "awg_ch_1_output",
        "awg_ch_2_output",
    }
    assert type(awg["awg_status"]) is bool
    assert type(awg["awg_freq"]) is int and awg["awg_freq"] > 0
    assert isinstance(awg["awg_run_mode"], str)
    assert type(awg["awg_ch_1_output"]) is bool
    assert type(awg["awg_ch_2_output"]) is bool

    scope = parsed_cfg["oscilloscope_controller"]
    assert set(scope.keys()) == {
        "channel",
        "acq_rate",
        "sample_rate",
        "math_averages",
        "visa_address",
        "math3",
        "math4",
        "math3_cont",
    }
    assert isinstance(scope["channel"], str)
    assert type(scope["acq_rate"]) is int and scope["acq_rate"] >= 1
    assert type(scope["sample_rate"]) is int and scope["sample_rate"] >= 1
    assert type(scope["math_averages"]) is int and scope["math_averages"] >= 1
    assert isinstance(scope["visa_address"], str)
    for math_key in ("math3", "math4", "math3_cont"):
        math_tbl = scope[math_key]
        assert set(math_tbl.keys()) == {"window", "resolution", "gate_position", "scale"}
        assert math_tbl["window"] in {"Rectangular", "Hamming", "Hanning", "Blackman"}
        assert float(math_tbl["resolution"]) > 0.0
        assert float(math_tbl["gate_position"]) >= 0.0
        assert float(math_tbl["scale"]) > 0.0

    dg = parsed_cfg["delay_generator_controller"]
    assert set(dg.keys()) == {"trigger_rate", "trigger_state"}
    assert 1.0 <= float(dg["trigger_rate"]) <= 1000.0
    assert dg["trigger_state"] in {"INT", "EXT"}


def _assert_matches_upstream_surf_scan_settings_contract(
    parsed_scan: Dict[str, Any],
    *,
    expect_zaber_pos: bool,
) -> None:
    """Validate parsed TOML dict against ``SURF/src/spectrometer_gui/frequency_scan_settings.py:4-52``."""
    assert set(parsed_scan.keys()) == {
        "experiment",
        "scan_parameters",
        "digitizer_settings",
        "timing_settings",
        "output_settings",
    }
    assert set(parsed_scan["experiment"].keys()) == {
        "sample_name",
        "sample_temp",
        "gas_name",
        "gas_width",
        "backing_pressure",
        "chamber_pressure",
        "mw_width",
    }
    expected_scan_keys = {
        "start_freq",
        "end_freq",
        "step_size",
        "zaber_speed",
        "acq_num",
        "cavity_type",
    }
    if expect_zaber_pos:
        expected_scan_keys.add("zaber_pos")
    assert set(parsed_scan["scan_parameters"].keys()) == expected_scan_keys
    assert set(parsed_scan["digitizer_settings"].keys()) == {
        "resolution",
        "acq_window",
        "apodization",
    }
    assert set(parsed_scan["timing_settings"].keys()) == {
        "rep_rate",
        "valve_mw_delay",
        "spdt_width",
        "acq_delay",
    }
    assert set(parsed_scan["output_settings"].keys()) == {"filename", "location"}


# ---------------------------------------------------------------------------
# 1. Exact SURF Scan-Time Estimator & Cavity Map Lookup
# ---------------------------------------------------------------------------


def test_surf_scan_time_estimator_exact_formula() -> None:
    """Verify exact formula matching ``SURF/src/spectrometer_gui/spectrometer.py:168-181``."""
    est = estimate_surf_scan_duration(
        start_freq_mhz=10897.0,
        end_freq_mhz=10902.0,
        step_size_mhz=0.5,
        acq_num=100,
        rep_rate_hz=5,
    )
    # iterations = ceil(|10902 - 10897| / 0.5) + 1 = 11
    # t_delay = 100 / 5 = 20.0 s; T_min = 11 * (20.0 + 14.0) / 60.0 = 374 / 60
    assert est.iterations == 11
    assert est.time_delay_per_step_s == pytest.approx(20.0)
    assert est.retune_overhead_s == pytest.approx(14.0)
    assert est.total_seconds == pytest.approx(374.0)
    assert est.total_minutes == pytest.approx(374.0 / 60.0)
    assert estimate_surf_scan_minutes(10902.0, 10897.0, 0.5, 100, 5) == pytest.approx(374.0 / 60.0)

    # Non-integer division rounds up via math.ceil
    est_ceil = estimate_surf_scan_duration(10000.0, 10001.1, 0.5, acq_num=50, rep_rate_hz=10)
    assert est_ceil.iterations == math.ceil(1.1 / 0.5) + 1
    assert est_ceil.total_minutes == pytest.approx((4 * (5.0 + 14.0)) / 60.0)

    # Guard against step_size <= 0 and invalid timing
    with pytest.raises(ValueError, match="step_size_mhz"):
        estimate_surf_scan_duration(10000.0, 10010.0, 0.0)
    with pytest.raises(ValueError, match="rep_rate_hz"):
        estimate_surf_scan_duration(10000.0, 10010.0, 0.5, rep_rate_hz=0)


def test_cavity_map_csv_lookup_and_sha256_provenance(tmp_path: Path) -> None:
    """Verify ``SurfCavityMap`` parses ``SURF/spectrometer.py:581-595`` CSVs with exact SHA-256."""
    runs_dir = tmp_path / "CavityRuns"
    runs_dir.mkdir()
    csv_path = runs_dir / "11440.0MHz.csv"
    csv_text = (
        "Zaber Position (mm),Intensity (V),Frequency (MHz),Peaks,Intensity\n"
        "10.000,0.002,11440.0,22.450,0.415\n"
        "22.450,0.415,11440.0,37.120,0.190\n"
        "37.120,0.190,11440.0,,\n"
    )
    csv_path.write_bytes(csv_text.encode("utf-8"))
    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    cmap = SurfCavityMap.from_cavity_runs_dir(runs_dir)
    assert len(cmap) == 1

    hit = cmap.lookup(11441.5, max_freq_delta_mhz=10.0)
    assert hit is not None
    assert hit.frequency_mhz == pytest.approx(11440.0)
    assert hit.resonant_zaber_pos_mm == pytest.approx(22.450)
    assert hit.peak_intensity_v == pytest.approx(0.415)
    assert hit.source_sha256 == expected_sha

    assert cmap.lookup(15000.0, max_freq_delta_mhz=25.0) is None


# ---------------------------------------------------------------------------
# 2. Navigator & Resolvability Profile Integration
# ---------------------------------------------------------------------------


def test_navigator_and_resolvability_grulab_surf_integration() -> None:
    """Verify profile clamping, SURF duration metadata, and profile-aware RBW."""
    # ResolvabilityFilter preserves legacy 0.050 default, adopts 0.100 MHz for grulab_surf
    legacy_filter = ResolvabilityFilter()
    assert legacy_filter.instrumental_resolution_mhz == pytest.approx(0.050)

    surf_filter = ResolvabilityFilter(instrument_profile="grulab_surf")
    assert surf_filter.instrument_profile == "grulab_surf"
    assert surf_filter.instrumental_resolution_mhz == pytest.approx(0.100)

    cp_filter = ResolvabilityFilter(instrument_profile="generic_cp_ftmw")
    assert cp_filter.instrumental_resolution_mhz == pytest.approx(0.025)

    fine_surf_filter = ResolvabilityFilter.from_instrument_profile("grulab_surf", resolution_khz=10.0)
    assert fine_surf_filter.instrumental_resolution_mhz == pytest.approx(0.010)
    assert fine_surf_filter.update_resolution_khz(25.0) == pytest.approx(0.025)

    # SmartScanNavigator defaults to grulab_surf and clamps out-of-band transitions
    nav_surf = SmartScanNavigator(max_chunk_bandwidth_mhz=100.0, top_n_recommendations=3)
    assert nav_surf.instrument_profile == DEFAULT_INSTRUMENT_PROFILE == "grulab_surf"

    freqs = np.array([5000.0, 9153.2, 11441.5, 17960.0, 19500.0], dtype=np.float64)
    ig = np.array([999.0, 12.0, 48.0, 35.0, 999.0], dtype=np.float64)
    penalties = np.ones_like(freqs)
    recs = nav_surf.recommend_scan_regions(freqs, ig, penalties, [("A", 50.0)])

    assert len(recs) >= 2
    for rec in recs:
        assert rec.instrument_profile == "grulab_surf"
        assert 8000.0 <= rec.start_frequency_mhz < rec.end_frequency_mhz <= 18000.0
        assert rec.surf_estimated_duration_min is not None and rec.surf_estimated_duration_min > 0.0
        expected_min = estimate_surf_scan_minutes(
            rec.start_frequency_mhz,
            rec.end_frequency_mhz,
            GRULAB_SURF_PROFILE.default_step_size_mhz,
        )
        assert rec.surf_estimated_duration_min == pytest.approx(expected_min)

    # Top recommendation must NOT pick the out-of-band 5000 or 19500 MHz transitions
    assert recs[0].start_frequency_mhz <= 11441.5 <= recs[0].end_frequency_mhz

    # Secondary profile generic_cp_ftmw permits 2-18 GHz (includes 5000 MHz!)
    nav_cp = SmartScanNavigator(
        max_chunk_bandwidth_mhz=100.0,
        top_n_recommendations=2,
        instrument_profile="generic_cp_ftmw",
    )
    cp_recs = nav_cp.recommend_scan_regions(freqs, ig, penalties, [("A", 50.0)])
    assert cp_recs[0].instrument_profile == "generic_cp_ftmw"
    assert cp_recs[0].surf_estimated_duration_min is None
    assert cp_recs[0].start_frequency_mhz <= 5000.0 <= cp_recs[0].end_frequency_mhz

    # Targeted windows on SmartScanNavigator
    targeted = nav_surf.recommend_targeted_windows(
        freqs,
        ig,
        penalties,
        [("A", 50.0)],
        ml_uncertainties_mhz=np.array([0.1, 0.12, 0.25, 0.15, 0.1]),
    )
    assert len(targeted) >= 1
    assert all(t.window_mode == "targeted" for t in targeted)
    assert all(8000.0 <= t.start_frequency_mhz < t.end_frequency_mhz <= 18000.0 for t in targeted)


# ---------------------------------------------------------------------------
# 3. Full HCOOH-DME End-to-End Closed-Loop: GP -> Planner -> Overrides -> Exporter
# ---------------------------------------------------------------------------


def test_hcooh_dme_end_to_end_gp_planner_overrides_and_surf_export(tmp_path: Path) -> None:
    """Run authentic HCOOH-DME GP residuals -> planner -> HITL overrides -> SURF TOML/manifest export."""
    # Step 1: Locked HCOOH-DME transitions with realistic systematic (O - C) residuals
    locked_qns: List[Tuple[int, int, int, int, int, int]] = [
        (4, 0, 4, 3, 0, 3),
        (4, 1, 4, 3, 1, 3),
        (5, 0, 5, 4, 0, 4),
        (5, 1, 5, 4, 1, 4),
        (6, 0, 6, 5, 0, 5),
        (6, 1, 6, 5, 1, 5),
        (4, 1, 3, 3, 0, 3),
        (5, 1, 4, 4, 0, 4),
    ]
    locked_assignments: List[Tuple[int, int, int, int, int, int, float]] = []
    locked_residuals: List[float] = []
    for qn in locked_qns:
        nu_calc, _ = _compute_hcooh_dme_transition_and_jacobian(qn, HCOOH_DME_CONSTANTS)
        # Authentic smooth J/Ka-dependent unmodeled tunneling/quartic shift (~0.04-0.35 MHz)
        o_minus_c = 0.035 * qn[0] + 0.12 * qn[1] - 0.08
        locked_assignments.append((*qn, round(nu_calc + o_minus_c, 6)))
        locked_residuals.append(o_minus_c)

    # Create immutable FitState snapshot
    fit_state = FitState(
        locked_assignments=tuple(locked_assignments),
        physical_constants=dict(HCOOH_DME_CONSTANTS),
        global_rmsd=42.5,
        global_chi2=1.18,
    )

    # Train real QuantumFeatureExtractor + SpycFitGPRegressor
    extractor = QuantumFeatureExtractor()
    x_train = extractor.fit_transform(locked_assignments, HCOOH_DME_DIPOLES)
    gp = SpycFitGPRegressor(n_restarts_optimizer=1, random_state=42)
    gp.train(x_train, np.array(locked_residuals, dtype=np.float64))
    assert gp.is_trained

    # Step 2: Candidate unassigned HCOOH-DME transitions (including out-of-band lines to test clamping)
    candidate_qns: List[Tuple[int, int, int, int, int, int]] = [
        (3, 0, 3, 2, 0, 2),  # ~6865 MHz (below 8000 MHz SURF lower bound)
        (4, 2, 3, 3, 1, 2),  # b-type Ka=2 <- 1 (strong A sensitivity, inside 8-18 GHz)
        (5, 2, 4, 4, 1, 3),  # b-type Ka=2 <- 1 (inside 8-18 GHz)
        (6, 2, 5, 5, 1, 4),  # b-type Ka=2 <- 1 (inside 8-18 GHz)
        (7, 0, 7, 6, 0, 6),  # a-type J=7 <- 6 (~16018 MHz, inside 8-18 GHz)
        (7, 1, 6, 6, 0, 6),  # b-type (high frequency near 18000 MHz edge)
        (8, 0, 8, 7, 0, 7),  # ~18306 MHz (above 18000 MHz SURF upper bound)
    ]
    cand_calc_freqs: List[float] = []
    cand_jac_rows: List[np.ndarray] = []
    cand_tuples_7: List[Tuple[int, int, int, int, int, int, float]] = []
    for qn in candidate_qns:
        nu_calc, jac_row = _compute_hcooh_dme_transition_and_jacobian(qn, HCOOH_DME_CONSTANTS)
        cand_calc_freqs.append(nu_calc)
        cand_jac_rows.append(jac_row)
        cand_tuples_7.append((*qn, nu_calc))

    x_cand = extractor.transform(extractor.extract_raw_features(cand_tuples_7, HCOOH_DME_DIPOLES))
    gp_preds = gp.predict(x_cand, cand_calc_freqs, candidate_qns)
    assert all(p.provenance == ProvenanceTag.ML for p in gp_preds)

    # Compute D-optimal Information Gain via JAX engine
    cov_matrix = np.diag([25.0, 0.8, 0.6, 1e-4, 5e-4])
    jac_matrix = np.vstack(cand_jac_rows)
    ig_jax = calculate_information_gain(
        covariance_matrix=cov_matrix,
        jacobian=jac_matrix,
        measurement_variance=0.01,
        criterion=InformationCriterion.D_OPTIMAL,
    )
    ig_np = np.asarray(ig_jax, dtype=np.float64)

    # SmartScanNavigator identifies A as top target constant
    param_names = ["A", "B", "C", "DJ", "DJK"]
    nav = SmartScanNavigator(instrument_profile="grulab_surf")
    targets = nav.identify_target_constants(cov_matrix, param_names, n_targets=2)
    assert targets[0][0] == "A"

    # Optional cavity map near the first in-band targeted proposal
    in_band_first_nu = gp_preds[1].nu_proposed_mhz
    cmap_dir = tmp_path / "cavity_runs"
    cmap_dir.mkdir()
    cmap_csv = cmap_dir / f"{in_band_first_nu:.1f}MHz.csv"
    cmap_csv.write_bytes(
        (
            "Zaber Position (mm),Intensity (V),Frequency (MHz),Peaks,Intensity\n"
            f"18.250,0.512,{in_band_first_nu:.3f},18.250,0.512\n"
        ).encode("utf-8")
    )
    cmap_sha = hashlib.sha256(cmap_csv.read_bytes()).hexdigest()
    cavity_map = SurfCavityMap.from_cavity_runs_dir(cmap_dir)

    # Step 3: Plan targeted and survey windows via SurfScanPlanner
    planner = SurfScanPlanner(
        instrument_profile="grulab_surf",
        cavity_map=cavity_map,
        molecule_name="HCOOH_DME",
        output_location=str(tmp_path / "surf_runs"),
        scan_date="2026-11-24",
    )
    gp_params = gp.get_trained_kernel_params()
    targeted_proposals = planner.plan_targeted_windows(
        predictions=gp_preds,
        information_gains=ig_np,
        target_constants=[name for name, _ in targets],
        step_size_mhz=0.250,
        top_n=4,
        gp_hyperparameters=gp_params,
    )
    assert len(targeted_proposals) >= 2
    # Strictly ranked by IG / T_min descending
    for left, right in zip(targeted_proposals, targeted_proposals[1:]):
        assert left.information_gain_per_minute >= right.information_gain_per_minute
    # All proposals strictly clamped to [8000.0, 18000.0] MHz
    for prop in targeted_proposals:
        assert 8000.0 <= prop.start_freq_mhz < prop.end_freq_mhz <= 18000.0
        assert prop.estimated_runtime_minutes == pytest.approx(
            estimate_surf_scan_minutes(
                prop.start_freq_mhz,
                prop.end_freq_mhz,
                prop.step_size_mhz,
                prop.scan_settings.scan_parameters.acq_num,
                prop.scan_settings.timing_settings.rep_rate,
            )
        )

    # Verify cavity map pre-populated zaber_pos and SHA-256 on the matched region
    matched_props = [p for p in targeted_proposals if p.cavity_map_zaber_pos_mm is not None]
    assert len(matched_props) >= 1
    assert matched_props[0].cavity_map_zaber_pos_mm == pytest.approx(18.250)
    assert matched_props[0].cavity_map_source_sha256 == cmap_sha
    assert matched_props[0].scan_settings.scan_parameters.zaber_pos == pytest.approx(18.250)

    # Survey windows ranked by IG / T_min
    survey_proposals = planner.plan_survey_windows(
        transition_frequencies_mhz=[p.nu_proposed_mhz for p in gp_preds],
        information_gains=ig_np,
        target_constants=[name for name, _ in targets],
        max_chunk_bandwidth_mhz=50.0,
        step_size_mhz=0.500,
        top_n=3,
        gp_hyperparameters=gp_params,
    )
    assert len(survey_proposals) >= 1
    for left, right in zip(survey_proposals, survey_proposals[1:]):
        assert left.information_gain_per_minute >= right.information_gain_per_minute

    # Step 4: HITL User Parameter Override Table (including zaber_pos=None omission test & rf_level soft warn)
    top_prop = targeted_proposals[0]
    override_table = planner.create_override_table(top_prop)
    assert len(override_table.field_specs) == len(SURF_OVERRIDE_FIELD_SPECS) == 24

    # Out-of-bounds overrides MUST raise ValidationError without mutating state
    snapshot_before = override_table.current_scan_settings.model_dump()
    with pytest.raises(ValidationError):
        override_table.set_override("scan_parameters.start_freq", 7999.0)
    with pytest.raises(ValidationError):
        override_table.set_override("scan_parameters.step_size", 0.0)
    with pytest.raises(ValidationError):
        override_table.set_override("scan_parameters.zaber_pos", 50.5)
    with pytest.raises(ValidationError):
        override_table.set_override("valon_controller.rf_level", 21)
    assert override_table.current_scan_settings.model_dump() == snapshot_before

    # Apply valid operator overrides including clearing zaber_pos -> None, finer RBW -> 20 kHz, rf_level -> 14 dBm
    override_table.apply_overrides(
        {
            "scan_parameters.acq_num": 250,
            "scan_parameters.zaber_pos": None,
            "digitizer_settings.resolution": 20,
            "timing_settings.rep_rate": 10,
            "valon_controller.rf_level": 14,
        },
        edited_by="dr_spectroscopist",
        edited_at_utc="2026-11-24T13:15:00Z",
    )
    assert override_table.resolvability_resolution_mhz == pytest.approx(0.020)
    assert override_table.build_resolvability_filter().instrumental_resolution_mhz == pytest.approx(0.020)
    assert len(override_table.warnings) == 1
    assert "+10 dBm" in override_table.warnings[0]

    # Round-trip JSON and TOML override files onto a fresh override table
    json_ov_path = tmp_path / "overrides.json"
    toml_ov_path = tmp_path / "overrides.toml"
    override_table.save_overrides_json(json_ov_path)
    override_table.save_overrides_toml(toml_ov_path)

    table_from_json = planner.create_override_table(top_prop)
    table_from_json.load_overrides_json(json_ov_path)
    assert table_from_json.current_scan_settings == override_table.current_scan_settings
    assert table_from_json.current_control_config == override_table.current_control_config

    table_from_toml = planner.create_override_table(top_prop)
    table_from_toml.load_overrides_toml(toml_ov_path, edited_by="dr_spectroscopist")
    assert table_from_toml.current_scan_settings == override_table.current_scan_settings
    assert table_from_toml.current_control_config == override_table.current_control_config

    # Step 5: Export deterministic artifacts via SurfExporter
    spcat_cat = tmp_path / "hcooh_dme.cat"
    spcat_cat.write_bytes(b" 11441.5000  0.0100 -4.2100 3   12.3450  9  303  5 0 5  4 0 4\n")
    expected_cat_sha = hashlib.sha256(spcat_cat.read_bytes()).hexdigest()

    export_dir = tmp_path / "surf_export"
    exporter = SurfExporter(instrument_profile="grulab_surf")
    result = exporter.export_region(
        output_dir=export_dir,
        override_table=override_table,
        region_stem="hcooh_dme_region_01",
        fit_state=fit_state,
        spcat_catalog_path=spcat_cat,
    )

    assert result.control_config_path == export_dir / "hcooh_dme_region_01_control_config.toml"
    assert result.scan_settings_path == export_dir / "hcooh_dme_region_01_scan_settings.toml"
    assert result.manifest_path == export_dir / "hcooh_dme_region_01_surf_manifest.json"
    assert result.control_config_path.is_file()
    assert result.scan_settings_path.is_file()
    assert result.manifest_path.is_file()

    # Verify UTF-8 LF on all three exported artifacts
    for artifact in (result.control_config_path, result.scan_settings_path, result.manifest_path):
        raw = artifact.read_bytes()
        assert b"\r" not in raw
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert raw.endswith(b"\n")

    # Parse both TOMLs with stdlib tomllib and cross-validate against upstream SURF rules
    parsed_cfg = tomllib.loads(result.control_config_path.read_text(encoding="utf-8"))
    parsed_scan = tomllib.loads(result.scan_settings_path.read_text(encoding="utf-8"))
    _assert_matches_upstream_surf_config_contract(parsed_cfg)
    _assert_matches_upstream_surf_scan_settings_contract(parsed_scan, expect_zaber_pos=False)

    # Exact LO equation: valon_controller.freq == start_freq - awg_freq
    expected_lo = round(
        float(parsed_scan["scan_parameters"]["start_freq"])
        - float(parsed_cfg["awg_controller"]["awg_freq"]),
        6,
    )
    assert parsed_cfg["valon_controller"]["freq"] == pytest.approx(expected_lo, abs=1e-9)
    assert type(parsed_cfg["valon_controller"]["rf_level"]) is int
    assert parsed_cfg["valon_controller"]["rf_level"] == 14
    assert type(parsed_cfg["awg_controller"]["awg_freq"]) is int
    assert parsed_cfg["awg_controller"]["awg_freq"] == 30
    assert parsed_cfg["oscilloscope_controller"]["math4"]["resolution"] == pytest.approx(20.0)
    assert parsed_cfg["delay_generator_controller"]["trigger_rate"] == pytest.approx(10.0)
    assert "zaber_pos" not in parsed_scan["scan_parameters"]

    # Round-trip reload via SpycFit schemas
    assert SurfConfig.from_surf_toml_file(result.control_config_path) == result.control_config
    assert SurfFrequencyScanSettings.from_surf_toml_file(result.scan_settings_path) == result.scan_settings

    # Verify manifest strict JSON, exact SHA-256 digests, FitState ID, SPCAT SHA-256, GP, overrides, warnings
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    manifest = SurfScanManifest.from_strict_json(manifest_text)
    raw_manifest_dict = json.loads(manifest_text)
    assert raw_manifest_dict["fit_state_id"] == fit_state.state_id
    assert raw_manifest_dict["spcat_catalog_sha256"] == expected_cat_sha
    assert manifest.control_config_sha256 == hashlib.sha256(result.control_config_path.read_bytes()).hexdigest()
    assert manifest.scan_settings_sha256 == hashlib.sha256(result.scan_settings_path.read_bytes()).hexdigest()
    assert manifest.estimated_runtime_minutes == pytest.approx(override_table.estimated_runtime_minutes)
    assert manifest.recommendation_rank == 1
    assert manifest.target_constants == ("A", "B")
    assert len(manifest.gp_hyperparameters) > 0
    assert any("+10 dBm" in w for w in manifest.warnings)

    override_by_field = {rec.field: rec for rec in manifest.overrides}
    assert override_by_field["scan_parameters.acq_num"].proposed_value == top_prop.scan_settings.scan_parameters.acq_num
    assert override_by_field["scan_parameters.acq_num"].user_value == 250
    assert override_by_field["scan_parameters.acq_num"].edited_by == "dr_spectroscopist"
    assert override_by_field["valon_controller.rf_level"].proposed_value == 10
    assert override_by_field["valon_controller.rf_level"].user_value == 14

    # Also verify export WITH zaber_pos populated (non-None) includes zaber_pos in TOML
    override_table.set_override("scan_parameters.zaber_pos", 24.375, edited_by="dr_spectroscopist")
    res_with_zaber = exporter.export_region(
        output_dir=export_dir,
        override_table=override_table,
        region_stem="hcooh_dme_region_02",
    )
    parsed_scan_z = tomllib.loads(res_with_zaber.scan_settings_path.read_text(encoding="utf-8"))
    _assert_matches_upstream_surf_scan_settings_contract(parsed_scan_z, expect_zaber_pos=True)
    assert parsed_scan_z["scan_parameters"]["zaber_pos"] == pytest.approx(24.375)


# ---------------------------------------------------------------------------
# 4. GUI (PySide6 Offscreen QTableWidget) & Interactive Dashboard Integration
# ---------------------------------------------------------------------------


def test_gui_profile_selector_override_table_and_export(tmp_path: Path) -> None:
    """Verify headless controller and Qt ``SpycFitTab`` profile selector, QTableWidget, and export."""
    # Headless controller works regardless of display server
    controller = SurfOverridePanelController(operator_id="lab_operator")
    assert controller.instrument_profile == "grulab_surf"
    assert controller.select_instrument_profile("generic_cp_ftmw").profile_id == "generic_cp_ftmw"
    assert controller.select_instrument_profile("grulab_surf").profile_id == "grulab_surf"

    ok_good, _ = controller.apply_cell_edit("scan_parameters.acq_num", "300")
    assert ok_good is True
    assert controller.override_table.current_scan_settings.scan_parameters.acq_num == 300

    ok_bad, msg_bad = controller.apply_cell_edit("valon_controller.rf_level", "35")
    assert ok_bad is False
    assert "Validation rejected" in msg_bad
    assert controller.override_table.current_control_config.valon_controller.rf_level == 10

    ctrl_export = controller.export_surf_files(tmp_path / "headless_export", region_stem="ctrl_region")
    assert ctrl_export.control_config_path.is_file()
    assert ctrl_export.scan_settings_path.is_file()
    assert ctrl_export.manifest_path.is_file()

    # Full PySide6 offscreen widget verification
    if HAS_PYSIDE6:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        assert app is not None

        tab = SpycFitTab(operator_id="qt_operator")
        assert tab.selected_instrument_profile() == "grulab_surf"
        assert tab.profile_selector.currentText() == INSTRUMENT_PROFILE_LABELS["grulab_surf"]
        assert "GruLab SURF" in tab.profile_selector.currentText()
        assert tab.surf_override_table.rowCount() == len(SURF_OVERRIDE_FIELD_SPECS)

        # Switch to secondary profiles and back to grulab_surf
        tab.set_instrument_profile("generic_cp_ftmw")
        assert tab.selected_instrument_profile() == "generic_cp_ftmw"
        tab.set_instrument_profile("generic_ir_raman")
        assert tab.selected_instrument_profile() == "generic_ir_raman"
        tab.set_instrument_profile("grulab_surf")
        assert tab.selected_instrument_profile() == "grulab_surf"

        # Interactive QTableWidget cell edit -> valid override
        acq_row = tab._override_row_by_field["scan_parameters.acq_num"]
        tab.surf_override_table.item(acq_row, 3).setText("400")
        assert tab.surf_override_model.current_scan_settings.scan_parameters.acq_num == 400
        assert tab.last_override_error is None

        # Interactive QTableWidget cell edit -> invalid out-of-bounds value reverts cleanly
        rf_row = tab._override_row_by_field["valon_controller.rf_level"]
        tab.surf_override_table.item(rf_row, 3).setText("25")
        assert tab.last_override_error is not None
        assert tab.surf_override_model.current_control_config.valon_controller.rf_level == 10
        assert tab.surf_override_table.item(rf_row, 3).text() == "10"

        # Export from SpycFitTab -> verified artifacts
        qt_export = tab.export_surf_files(tmp_path / "qt_export", region_stem="qt_region")
        assert qt_export is not None
        parsed_qt_scan = tomllib.loads(qt_export.scan_settings_path.read_text(encoding="utf-8"))
        assert parsed_qt_scan["scan_parameters"]["acq_num"] == 400
        tab.close()


def test_interactive_dashboard_profile_selector_and_surf_export(tmp_path: Path) -> None:
    """Verify ``SpycFitInteractiveDashboard`` defaults to ``grulab_surf`` and exports SURF files."""
    cfg = DashboardConfig()
    assert cfg.instrument_profile == "grulab_surf"
    assert get_instrument_profile(cfg.instrument_profile) == GRULAB_SURF_PROFILE

    exp_peaks = [
        SpectralPeak(peak_id="E1", frequency=11441.52, intensity=1.2, quantum_label="5_0,5 -> 4_0,4"),
    ]
    theo_peaks = [
        SpectralPeak(peak_id="T1", frequency=11441.50, intensity=1.15, quantum_label="5_0,5 -> 4_0,4"),
    ]
    state = WhatIfParameterState(A_mhz=3826.42, B_mhz=1295.81, C_mhz=992.54, temperature_k=4.0)
    manager = HITLAssignmentManager(exp_peaks=exp_peaks, theo_peaks=theo_peaks)
    engine = WhatIfSpectrumEngine(
        base_peaks=theo_peaks,
        base_state=state,
        grid_min_mhz=11400.0,
        grid_max_mhz=11500.0,
        grid_points=200,
    )

    dashboard = SpycFitInteractiveDashboard(manager=manager, spectrum_engine=engine, config=cfg)
    assert dashboard.profile_dropdown.value == "grulab_surf"
    assert dashboard.get_active_instrument_profile().profile_id == "grulab_surf"

    dashboard.set_instrument_profile("generic_cp_ftmw")
    assert dashboard.profile_dropdown.value == "generic_cp_ftmw"
    dashboard.set_instrument_profile("grulab_surf")

    dashboard.set_surf_override("scan_parameters.step_size", 0.25, edited_by="notebook_user")
    assert len(dashboard.get_surf_override_rows()) == len(SURF_OVERRIDE_FIELD_SPECS)

    dash_export = dashboard.export_surf_artifacts(tmp_path / "dash_export", region_stem="dash_region")
    parsed_dash_scan = tomllib.loads(dash_export.scan_settings_path.read_text(encoding="utf-8"))
    assert parsed_dash_scan["scan_parameters"]["step_size"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 5. Strict UTF-8 LF, Zero-Pass AST, and Anti-Spoof Compliance Across Owned Files
# ---------------------------------------------------------------------------


def test_spyc_surf_02_owned_files_utf8_lf_zero_pass_and_anti_spoof() -> None:
    """Verify strict UTF-8 LF, zero pass statements, zero mocks, and strict lint cleanliness."""
    owned_files = [
        REPO_ROOT / "src" / "cochem_spycfit" / "instruments" / "surf_planner.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "instruments" / "surf_exporter.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_navigator.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_resolvability.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "gui.py",
        REPO_ROOT / "src" / "cochem_spycfit" / "interfaces" / "cochem_spycfit_ui.py",
        Path(__file__).resolve(),
    ]

    for py_path in owned_files:
        raw = py_path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {py_path}"
        assert b"\r\n" not in raw, f"CRLF detected in {py_path}"
        assert b"\r" not in raw, f"CR detected in {py_path}"
        assert raw.endswith(b"\n"), f"Missing trailing LF in {py_path}"

        tree = ast.parse(raw.decode("utf-8"), filename=str(py_path))
        pass_nodes = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Pass)]
        assert pass_nodes == [], f"Forbidden pass statements at {pass_nodes} in {py_path}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower(), f"Prohibited mock import in {py_path}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower(), f"Prohibited mock import in {py_path}"

        # Enforce strict anti-spoof linter on all non-amnesty files
        if py_path.name != "cochem_spycfit_ui.py":
            violations = check_script(py_path, strict_mode=True)
            assert violations == [], f"Anti-spoof strict violations in {py_path}: {violations}"
