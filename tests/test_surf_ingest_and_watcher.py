"""Zero-mock physical integration tests for SPYC-SURF-03: SURF parser, watcher, Doppler picker, and viewport.

Covers:
1. Authentic SURF fixture run tree parsing (CavityFiles/*.csv, finalized compacted CSV,
   scan_settings.toml, config TOML, CavityRuns/<freq>MHz.csv, search_settings.toml).
2. Incremental SurfRunWatcher ingestion for both StepDirection.Up and StepDirection.Down,
   exact +/- step_size/2 FP64 stitching, Windows-safe stability gating, real OS exclusive
   file-lock PermissionError retry, and strict JSONL SHA-256 provenance ledger.
3. End-of-run finalized compacted CSV reconciliation, missing-step detection, and
   Zaber rail-dropout flagging (0.0 mm / 50.0 mm) plus bounds/NaN rejection.
4. CavityRuns cavity resonance lookup table extraction with exact SHA-256 provenance.
5. Coaxial Fabry-Perot Doppler doublet peak picker across Ne, Ar, He, N2 carrier gases:
   paired doublet midpoint + raw components, uncollapsed singlets and overlapping blends,
   and strict HITL AssignmentStatus.PROPOSED enforcement (never auto-APPROVED).
6. Auto-detection of SURF FFT CSVs in cochem_spyfy_ingest without regressing assigned CSVs.
7. Live SURF spectrum overlay & cavity-track health telemetry in SpectralViewport.
8. Strict UTF-8 LF, zero-pass, zero-mock, and fixture air-gap invariants.
"""

from __future__ import annotations

import ast
import ctypes
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

SPYCFIT_ROOT = Path(__file__).resolve().parent.parent
SPYCFIT_SRC = SPYCFIT_ROOT / "src"
if str(SPYCFIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_SRC))

from cochem_spyfy.instruments.surf_ingest import (
    CARRIER_GAS_BEAM_VELOCITIES_M_S,
    SPEED_OF_LIGHT_M_S,
    StepDirection,
    SurfBoundsError,
    SurfCandidateTransition,
    SurfCsvSchemaError,
    SurfFeatureKind,
    SurfOutputParser,
    compute_file_sha256,
    expected_doppler_splitting_mhz,
    pick_surf_doppler_peaks,
)
from cochem_spyfy.instruments.surf_schemas import (
    SurfCavitySearchSettings,
    SurfConfig,
    SurfFrequencyScanSettings,
)
from cochem_spyfy.instruments.surf_watcher import SurfRunWatcher
from cochem_spyfy.intake.cochem_spyfy_ingest import (
    FileFormat,
    is_surf_fft_csv,
    parse_csv_spectroscopy,
    parse_spectroscopy_data,
)
from cochem_spyfy.interfaces.cochem_spyfy_ui import AssignmentStatus
from cochem_spyfy.interfaces.cochem_spyfy_ui_viewport import (
    SpectralViewport,
    ViewportError,
)

FIXTURE_RUN_DIR = SPYCFIT_ROOT / "tests" / "fixtures" / "surf" / "sample_run_0"


# =============================================================================
# HELPER FUNCTIONS (REAL PHYSICAL SPECTRUM BUILDERS, ZERO MOCKS)
# =============================================================================


def _build_gaussian_spectrum(
    freqs_mhz: np.ndarray,
    peaks: List[Tuple[float, float]],
    *,
    baseline: float = 0.0035,
    noise_sigma: float = 0.0005,
    line_sigma_mhz: float = 0.005,
    seed: int = 2026,
) -> np.ndarray:
    """Build deterministic physical Gaussian emission trace on a frequency grid."""
    rng = np.random.default_rng(seed)
    signal = np.clip(baseline + rng.normal(0.0, noise_sigma, size=freqs_mhz.size), 0.0001, None)
    for center, amp in peaks:
        signal = signal + amp * np.exp(-0.5 * ((freqs_mhz - center) / line_sigma_mhz) ** 2)
    return np.round(signal, 8).astype(np.float64)


def _write_step_csv(
    path: Path,
    freqs_mhz: np.ndarray,
    intensities: np.ndarray,
    center_freq_mhz: float,
    cavity_pos_mm: float,
) -> None:
    """Write an authentic sparse row-0 metadata SURF step CSV with strict UTF-8 LF."""
    spec_df = pd.DataFrame({"Frequency (MHz)": freqs_mhz, "Intensity": intensities})
    meta_df = pd.DataFrame({"Center Freq": [center_freq_mhz], "Cavity Position": [cavity_pos_mm]})
    combined = pd.concat([spec_df, meta_df], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(combined.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _write_finalized_compacted_csv(
    path: Path,
    step_tuples: List[Tuple[float, float, np.ndarray, np.ndarray]],
    *,
    step_size_mhz: float,
    step_direction: StepDirection,
) -> None:
    """Write an authentic SURF finalized compacted CSV mirroring spectrometer.py:705-758."""
    chunks: List[pd.DataFrame] = []
    for center_freq, cavity_pos, freqs, ints in step_tuples:
        df_spec = pd.DataFrame({"Frequency (MHz)": freqs, "Intensity": ints})
        filt = df_spec.loc[
            (df_spec["Frequency (MHz)"] >= center_freq - step_size_mhz / 2.0)
            & (df_spec["Frequency (MHz)"] <= center_freq + step_size_mhz / 2.0)
        ]
        if step_direction == StepDirection.Down:
            filt = filt.iloc[::-1]
        filt = filt.reset_index(drop=True)
        meta = pd.DataFrame({"Center Freq": [center_freq], "Cavity Position": [cavity_pos]})
        chunks.append(pd.concat([filt, meta], axis=1))

    combined = pd.concat(chunks, ignore_index=True)
    for col in ["Center Freq", "Cavity Position"]:
        combined[col] = combined[col].dropna().reset_index(drop=True)
    path.write_bytes(combined.to_csv(index=False, lineterminator="\n").encode("utf-8"))


# =============================================================================
# 1. AUTHENTIC FIXTURE TREE, SCHEMA & AIR-GAP INTEGRITY TESTS
# =============================================================================


def test_fixture_tree_schemas_encoding_and_airgap() -> None:
    """Verify fixture files are strictly .toml/.csv, UTF-8 LF, and validate against SURF schemas."""
    assert FIXTURE_RUN_DIR.is_dir()
    allowed_suffixes = {".toml", ".csv"}
    expected_relative = {
        "scan_settings.toml",
        "sample_run_config.toml",
        "search_settings.toml",
        "sample_run.csv",
        "CavityFiles/10899.0.csv",
        "CavityFiles/10899.5.csv",
        "CavityRuns/10899.0MHz.csv",
    }

    observed_relative = set()
    for path in FIXTURE_RUN_DIR.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() in allowed_suffixes, f"Disallowed fixture extension: {path}"
            raw = path.read_bytes()
            assert b"\r\n" not in raw and b"\r" not in raw, f"Non-LF line ending in {path}"
            assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM found in {path}"
            observed_relative.add(path.relative_to(FIXTURE_RUN_DIR).as_posix())

    assert expected_relative.issubset(observed_relative)

    scan_settings = SurfFrequencyScanSettings.from_surf_toml_file(FIXTURE_RUN_DIR / "scan_settings.toml")
    surf_cfg = SurfConfig.from_surf_toml_file(FIXTURE_RUN_DIR / "sample_run_config.toml")
    search_settings = SurfCavitySearchSettings.from_surf_toml_file(FIXTURE_RUN_DIR / "search_settings.toml")

    assert scan_settings.experiment.gas_name == "Ne"
    assert scan_settings.scan_parameters.start_freq == 10899.0
    assert scan_settings.scan_parameters.end_freq == 10899.5
    assert surf_cfg.valon_controller.freq == 10869.0
    assert surf_cfg.awg_controller.awg_freq == 30
    assert search_settings.freq == 10899.0


# =============================================================================
# 2. STEP CSV PARSING, BOUNDS VALIDATION & RECONCILIATION TESTS
# =============================================================================


def test_parser_step_csv_and_finalized_reconciliation_on_fixture() -> None:
    """Verify exact sparse row-0 parsing, SHA-256 hashing, +/- step_size/2 slicing, and reconciliation."""
    parser = SurfOutputParser()
    step_path = FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv"
    step = parser.parse_step_csv(step_path, step_size_mhz=0.5, step_direction=StepDirection.Up)

    assert step.file_sha256 == compute_file_sha256(step_path)
    assert step.center_freq_mhz == 10899.0
    assert step.cavity_position_mm == pytest.approx(18.425, abs=1e-9)
    assert step.cavity_dropout is False
    assert step.full_window_freq_mhz.dtype == np.float64
    assert step.full_window_intensity.dtype == np.float64
    assert step.stitched_freq_mhz[0] == pytest.approx(10898.75, abs=1e-9)
    assert step.stitched_freq_mhz[-1] == pytest.approx(10899.25, abs=1e-9)
    assert step.stitched_freq_mhz.size == 101

    report = parser.reconcile_run_directory(FIXTURE_RUN_DIR)
    assert report.reconciled is True
    assert report.spectrum_matches is True
    assert report.finalized_is_compacted is True
    assert report.has_cavity_dropout is False
    assert report.row_count_match is True
    assert report.stitched_row_count == 202
    assert report.finalized_row_count == 202
    assert report.max_freq_abs_diff_mhz == 0.0
    assert report.max_intensity_abs_diff == 0.0
    assert report.missing_steps_mhz == ()
    assert report.extra_steps_mhz == ()
    assert report.cavity_dropout_steps_mhz == ()
    assert report.finalized_csv_sha256 == compute_file_sha256(FIXTURE_RUN_DIR / "sample_run.csv")
    assert report.scan_settings_toml_sha256 == compute_file_sha256(FIXTURE_RUN_DIR / "scan_settings.toml")
    assert report.config_toml_sha256 == compute_file_sha256(FIXTURE_RUN_DIR / "sample_run_config.toml")


def test_parser_rejects_out_of_bounds_zaber_and_nonfinite_values(tmp_path: Path) -> None:
    """Verify strict bounds [0, 50] mm and finite FP64 enforcement on step CSVs."""
    parser = SurfOutputParser()
    freqs = np.linspace(10898.5, 10899.5, 21)
    ints = np.full(21, 0.005, dtype=np.float64)

    oob_high = tmp_path / "10899.0.csv"
    _write_step_csv(oob_high, freqs, ints, 10899.0, 50.05)
    with pytest.raises(SurfBoundsError):
        parser.parse_step_csv(oob_high, step_size_mhz=0.5)

    oob_low = tmp_path / "10899.5.csv"
    _write_step_csv(oob_low, freqs + 0.5, ints, 10899.5, -0.01)
    with pytest.raises(SurfBoundsError):
        parser.parse_step_csv(oob_low, step_size_mhz=0.5)

    nan_csv = tmp_path / "10900.0.csv"
    nan_csv.write_text(
        "Frequency (MHz),Intensity,Center Freq,Cavity Position\n"
        "10900.0,nan,10900.0,15.0\n",
        encoding="utf-8",
    )
    with pytest.raises(SurfCsvSchemaError):
        parser.parse_step_csv(nan_csv, step_size_mhz=0.5)

    bad_stem = tmp_path / "10901.0.csv"
    _write_step_csv(bad_stem, freqs, ints, 10899.0, 18.0)
    with pytest.raises(SurfCsvSchemaError):
        parser.parse_step_csv(bad_stem, step_size_mhz=0.5, verify_filename_stem=True)


def test_reconciliation_flags_missing_steps_and_cavity_rail_dropouts(tmp_path: Path) -> None:
    """Verify reconciliation flags missing CavityFiles steps and 0.0/50.0 mm rail dropouts."""
    run_dir = tmp_path / "dropout_run_0"
    shutil.copytree(FIXTURE_RUN_DIR, run_dir)

    # 1. Remove second step CSV to verify missing step detection
    (run_dir / "CavityFiles" / "10899.5.csv").unlink()
    parser = SurfOutputParser()
    missing_report = parser.reconcile_run_directory(run_dir)
    assert missing_report.reconciled is False
    assert 10899.5 in missing_report.missing_steps_mhz
    assert missing_report.row_count_match is False
    assert len(missing_report.discrepancies) >= 1

    # 2. Rebuild a 2-step run where step 1 hits 0.0 mm and step 2 hits 50.0 mm rail
    rail_dir = tmp_path / "rail_run_0"
    (rail_dir / "CavityFiles").mkdir(parents=True)
    shutil.copy2(FIXTURE_RUN_DIR / "scan_settings.toml", rail_dir / "scan_settings.toml")
    shutil.copy2(FIXTURE_RUN_DIR / "sample_run_config.toml", rail_dir / "sample_run_config.toml")

    f1 = np.round(np.linspace(10898.5, 10899.5, 201), 6)
    i1 = _build_gaussian_spectrum(f1, [(10899.0, 0.03)], seed=101)
    f2 = np.round(np.linspace(10899.0, 10900.0, 201), 6)
    i2 = _build_gaussian_spectrum(f2, [(10899.5, 0.03)], seed=102)

    _write_step_csv(rail_dir / "CavityFiles" / "10899.0.csv", f1, i1, 10899.0, 0.0)
    _write_step_csv(rail_dir / "CavityFiles" / "10899.5.csv", f2, i2, 10899.5, 50.0)
    _write_finalized_compacted_csv(
        rail_dir / "sample_run.csv",
        [(10899.0, 0.0, f1, i1), (10899.5, 50.0, f2, i2)],
        step_size_mhz=0.5,
        step_direction=StepDirection.Up,
    )

    rail_report = parser.reconcile_run_directory(rail_dir)
    assert rail_report.reconciled is True
    assert rail_report.has_cavity_dropout is True
    assert rail_report.cavity_dropout_steps_mhz == (10899.0, 10899.5)
    assert any("rail drop-out" in msg.lower() for msg in rail_report.warnings)


# =============================================================================
# 3. CAVITY SEARCH (CavityRuns/<freq>MHz.csv) LOOKUP EXTRACTION TESTS
# =============================================================================


def test_cavity_runs_lookup_table_and_trace_argmax_fallback(tmp_path: Path) -> None:
    """Verify CavityRuns/<freq>MHz.csv + search_settings.toml parsing into resonant Zaber map."""
    parser = SurfOutputParser()
    table = parser.parse_cavity_search_dir(FIXTURE_RUN_DIR)

    assert table.search_settings is not None
    assert table.search_settings.freq == 10899.0
    assert table.search_settings_sha256 == compute_file_sha256(FIXTURE_RUN_DIR / "search_settings.toml")

    pos_map = table.to_position_map()
    expected_sha = compute_file_sha256(FIXTURE_RUN_DIR / "CavityRuns" / "10899.0MHz.csv")
    assert 10899.0 in pos_map
    assert pos_map[10899.0] == (pytest.approx(18.425, abs=1e-6), expected_sha)

    entry = table.lookup(10899.2, max_delta_mhz=1.0)
    assert entry is not None
    assert entry.frequency_mhz == 10899.0
    assert entry.peak_source == "surf_peaks"
    assert entry.cavity_type == "continuous"
    assert len(entry.all_peaks) == 4
    assert table.lookup(12000.0, max_delta_mhz=5.0) is None

    # Fallback when SURF Peaks column is empty
    empty_peaks_csv = tmp_path / "10905.0MHz.csv"
    empty_peaks_csv.write_text(
        "Zaber Position (mm),Intensity (V),Frequency (MHz),Peaks,Intensity\n"
        "10.0,0.002,10905.0,,\n"
        "22.5,0.007,10905.0,,\n"
        "35.0,0.003,10905.0,,\n",
        encoding="utf-8",
    )
    fallback_entry = parser.parse_cavity_run_csv(empty_peaks_csv)
    assert fallback_entry.peak_source == "trace_argmax"
    assert fallback_entry.resonant_zaber_pos_mm == pytest.approx(22.5)
    assert fallback_entry.source_sha256 == compute_file_sha256(empty_peaks_csv)


# =============================================================================
# 4. LIVE WATCHER: INCREMENTAL UP & DOWN STITCHING, STABILITY GATE & JSONL LEDGER
# =============================================================================


def test_watcher_incremental_up_scan_stability_and_jsonl_ledger(tmp_path: Path) -> None:
    """Verify incremental SurfRunWatcher Up-scan stitching, stability gating, JSONL ledger, and reconciliation."""
    now_holder = [100.0]

    def advancing_clock() -> float:
        return now_holder[0]

    run_dir = tmp_path / "sample_run_0"
    (run_dir / "CavityFiles").mkdir(parents=True)
    shutil.copy2(FIXTURE_RUN_DIR / "scan_settings.toml", run_dir / "scan_settings.toml")
    shutil.copy2(FIXTURE_RUN_DIR / "sample_run_config.toml", run_dir / "sample_run_config.toml")

    watcher = SurfRunWatcher(
        tmp_path,
        run_prefix="sample_run",
        stability_polls=2,
        min_stable_seconds=0.25,
        clock=advancing_clock,
    )

    # Poll 1: registers scan_settings.toml only after 2 stable polls >= 0.25 s
    assert watcher.poll_once() == []
    now_holder[0] += 0.30
    assert watcher.poll_once() == []
    assert run_dir in watcher.discovered_run_dirs

    # Step 1 lands
    step1_dst = run_dir / "CavityFiles" / "10899.0.csv"
    shutil.copy2(FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv", step1_dst)

    # Stability gate MUST hold on poll 1 of step 1
    assert watcher.poll_once() == []
    now_holder[0] += 0.10
    # Still < 0.25 s elapsed -> MUST hold
    assert watcher.poll_once() == []
    now_holder[0] += 0.20

    # Now stat-stable across consecutive polls & >= 0.25 s -> ingests step 1!
    updates_1 = watcher.poll_once()
    assert len(updates_1) == 1
    upd1 = updates_1[0]
    assert upd1.step_index == 1
    assert upd1.step_data.center_freq_mhz == 10899.0
    assert upd1.step_data.file_sha256 == compute_file_sha256(step1_dst)
    assert upd1.cumulative_freq_mhz.size == 101
    assert upd1.cavity_health.status == "healthy"
    assert upd1.cavity_health.step_count == 1
    assert upd1.reconciliation is None
    assert len(upd1.proposed_candidates) >= 2
    assert all(c.status is AssignmentStatus.PROPOSED for c in upd1.proposed_candidates)
    assert all(c.source_step_sha256 == upd1.step_data.file_sha256 for c in upd1.proposed_candidates)

    # Step 2 lands
    step2_dst = run_dir / "CavityFiles" / "10899.5.csv"
    shutil.copy2(FIXTURE_RUN_DIR / "CavityFiles" / "10899.5.csv", step2_dst)
    assert watcher.poll_once() == []
    now_holder[0] += 0.30
    updates_2 = watcher.poll_once()
    assert len(updates_2) == 1
    upd2 = updates_2[0]
    assert upd2.step_index == 2
    assert upd2.cumulative_freq_mhz.size == 202

    # Finalized compacted sample_run.csv lands -> triggers automatic end-of-run reconciliation
    shutil.copy2(FIXTURE_RUN_DIR / "sample_run.csv", run_dir / "sample_run.csv")
    assert watcher.poll_once() == []
    now_holder[0] += 0.30
    assert watcher.poll_once() == []

    reconcile_report = watcher.get_reconciliation(run_dir)
    assert reconcile_report is not None
    assert reconcile_report.reconciled is True
    assert reconcile_report.max_freq_abs_diff_mhz == 0.0
    assert reconcile_report.max_intensity_abs_diff == 0.0

    # Verify stitched spectrum matches finalized CSV bit-for-bit in FP64
    cum_f, cum_i = watcher.get_stitched_spectrum(run_dir)
    finalized = SurfOutputParser().parse_finalized_csv(run_dir / "sample_run.csv")
    assert np.array_equal(cum_f, finalized.freq_mhz)
    assert np.array_equal(cum_i, finalized.intensity)

    # Verify strict JSONL provenance ledger entries (no NaN, no duplicates)
    ledger_path = run_dir / "surf_ingest_provenance.jsonl"
    assert ledger_path.is_file()
    raw_ledger = ledger_path.read_bytes()
    assert b"\r\n" not in raw_ledger
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = [rec["event_type"] for rec in records]
    assert event_types == ["run_discovered", "step_ingested", "step_ingested", "run_reconciled"]
    assert records[1]["sha256"] == compute_file_sha256(step1_dst)
    assert records[2]["sha256"] == compute_file_sha256(step2_dst)
    assert records[3]["sha256"] == compute_file_sha256(run_dir / "sample_run.csv")


def test_watcher_down_direction_exact_reversed_stitching(tmp_path: Path) -> None:
    """Verify StepDirection.Down reverses per-step slices and orders steps descending to match SURF."""
    now_holder = [50.0]

    run_dir = tmp_path / "down_scan_0"
    (run_dir / "CavityFiles").mkdir(parents=True)

    # Write scan_settings.toml with start_freq > end_freq (StepDirection.Down)
    base_settings = SurfFrequencyScanSettings.from_surf_toml_file(FIXTURE_RUN_DIR / "scan_settings.toml")
    down_settings = base_settings.model_copy(
        update={
            "scan_parameters": base_settings.scan_parameters.model_copy(
                update={"start_freq": 10899.5, "end_freq": 10899.0, "step_size": 0.5}
            ),
            "output_settings": base_settings.output_settings.model_copy(
                update={"filename": "down_scan", "location": str(tmp_path)}
            ),
        }
    )
    (run_dir / "scan_settings.toml").write_bytes(down_settings.to_surf_toml().encode("utf-8"))

    # Copy both step CSVs and write authentic SURF Down-direction finalized CSV
    parser = SurfOutputParser()
    s_high = parser.parse_step_csv(FIXTURE_RUN_DIR / "CavityFiles" / "10899.5.csv")
    s_low = parser.parse_step_csv(FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv")

    shutil.copy2(FIXTURE_RUN_DIR / "CavityFiles" / "10899.5.csv", run_dir / "CavityFiles" / "10899.5.csv")
    shutil.copy2(FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv", run_dir / "CavityFiles" / "10899.0.csv")

    _write_finalized_compacted_csv(
        run_dir / "down_scan.csv",
        [
            (10899.5, s_high.cavity_position_mm, s_high.full_window_freq_mhz, s_high.full_window_intensity),
            (10899.0, s_low.cavity_position_mm, s_low.full_window_freq_mhz, s_low.full_window_intensity),
        ],
        step_size_mhz=0.5,
        step_direction=StepDirection.Down,
    )

    watcher = SurfRunWatcher(
        tmp_path,
        run_prefix="down_scan",
        stability_polls=2,
        min_stable_seconds=0.1,
        clock=lambda: now_holder[0],
    )
    watcher.poll_once()
    now_holder[0] += 0.2
    watcher.poll_once()
    now_holder[0] += 0.2
    updates = watcher.poll_once()
    assert len(updates) == 2
    assert updates[0].step_data.center_freq_mhz == 10899.5
    assert updates[1].step_data.center_freq_mhz == 10899.0
    assert updates[0].step_data.step_direction == StepDirection.Down

    # Verify descending frequency slice ordering matches SURF spectrometer.py:725-726
    assert updates[0].step_data.stitched_freq_mhz[0] == pytest.approx(10899.75)
    assert updates[0].step_data.stitched_freq_mhz[-1] == pytest.approx(10899.25)

    now_holder[0] += 0.2
    watcher.poll_once()
    report = watcher.reconcile_run(run_dir)
    assert report.reconciled is True
    assert report.max_freq_abs_diff_mhz == 0.0
    assert report.max_intensity_abs_diff == 0.0

    # Cumulative peak picker still succeeds regardless of descending storage order
    peak_result = watcher.pick_run_peaks(run_dir)
    doublets = [c for c in peak_result.candidates if c.feature_kind == SurfFeatureKind.DOPPLER_DOUBLET]
    assert len(doublets) == 2
    assert all(c.status is AssignmentStatus.PROPOSED for c in peak_result.candidates)


def test_watcher_windows_permission_error_graceful_retry(tmp_path: Path) -> None:
    """Verify graceful retry on genuine Windows exclusive file lock PermissionError (zero mocks)."""
    now_holder = [10.0]
    run_dir = tmp_path / "lock_run_0"
    (run_dir / "CavityFiles").mkdir(parents=True)
    shutil.copy2(FIXTURE_RUN_DIR / "scan_settings.toml", run_dir / "scan_settings.toml")

    watcher = SurfRunWatcher(
        run_dir,
        stability_polls=2,
        min_stable_seconds=0.05,
        clock=lambda: now_holder[0],
    )
    watcher.poll_once()
    now_holder[0] += 0.1
    watcher.poll_once()

    locked_csv = run_dir / "CavityFiles" / "10899.0.csv"
    shutil.copy2(FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv", locked_csv)

    # First poll establishes stat signature
    assert watcher.poll_once() == []
    now_holder[0] += 0.1

    if os.name == "nt":
        # Acquire real OS exclusive file handle via Win32 CreateFileW (dwShareMode=0)
        generic_read_write = 0xC0000000
        open_existing = 3
        invalid_handle = ctypes.c_void_p(-1).value
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateFileW.restype = ctypes.c_void_p
        handle = kernel32.CreateFileW(
            str(locked_csv),
            generic_read_write,
            0,  # zero sharing -> genuine WinError 32 PermissionError for any other reader
            None,
            open_existing,
            0x80,
            None,
        )
        assert handle != invalid_handle
        try:
            # Watcher MUST catch PermissionError cleanly without crashing
            blocked_updates = watcher.poll_once()
            assert blocked_updates == []
            assert any(issue.error_type == "PermissionError" for issue in watcher.transient_issues)
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))

    # Once lock is released, next poll ingests cleanly with full SHA-256 provenance
    now_holder[0] += 0.1
    recovered = watcher.poll_once()
    assert len(recovered) == 1
    assert recovered[0].step_data.file_sha256 == compute_file_sha256(locked_csv)


# =============================================================================
# 5. COAXIAL FABRY-PEROT DOPPLER DOUBLET PEAK PICKER TESTS
# =============================================================================


def test_doppler_picker_on_fixture_step_doublet_and_singlet() -> None:
    """Verify fixture step 10899.0 yields one paired Ne doublet and one uncollapsed singlet, all PROPOSED."""
    parser = SurfOutputParser()
    step_path = FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv"
    step = parser.parse_step_csv(step_path, step_size_mhz=0.5, step_direction=StepDirection.Up)

    result = pick_surf_doppler_peaks(
        step.stitched_freq_mhz,
        step.stitched_intensity,
        gas_name="Ne",
        snr_threshold=5.0,
        source_step_sha256=step.file_sha256,
        source_file=step_path.name,
    )

    assert len(result.raw_peaks) == 3
    assert len(result.candidates) == 2

    doublets = [c for c in result.candidates if c.feature_kind == SurfFeatureKind.DOPPLER_DOUBLET]
    singlets = [c for c in result.candidates if c.feature_kind == SurfFeatureKind.SINGLET]
    assert len(doublets) == 1
    assert len(singlets) == 1

    dbl = doublets[0]
    assert dbl.rest_frequency_mhz == pytest.approx(10899.020, abs=0.002)
    assert len(dbl.raw_components) == 2
    assert dbl.observed_splitting_mhz == pytest.approx(
        expected_doppler_splitting_mhz(10899.020, "Ne"),
        rel=0.08,
    )
    assert dbl.uncertainty_mhz > 0.0
    assert dbl.source_step_sha256 == step.file_sha256
    assert dbl.status is AssignmentStatus.PROPOSED

    sgl = singlets[0]
    assert sgl.rest_frequency_mhz == pytest.approx(10898.840, abs=0.002)
    assert len(sgl.raw_components) == 1
    assert sgl.observed_splitting_mhz is None
    assert sgl.source_step_sha256 == step.file_sha256
    assert sgl.status is AssignmentStatus.PROPOSED


@pytest.mark.parametrize("gas_name", ["Ne", "Ar", "He", "N2"])
def test_doppler_picker_carrier_gas_velocity_models(gas_name: str) -> None:
    """Verify delta_nu_d = 2 * nu * (v_beam / c) pairing across Ne, Ar, He, and N2."""
    nu_rest = 12500.0
    v_beam = CARRIER_GAS_BEAM_VELOCITIES_M_S[gas_name]
    expected_sep = 2.0 * nu_rest * (v_beam / SPEED_OF_LIGHT_M_S)

    grid = np.round(np.linspace(nu_rest - 0.5, nu_rest + 0.5, 501), 6)  # 2 kHz spacing
    intensities = _build_gaussian_spectrum(
        grid,
        [
            (nu_rest - 0.5 * expected_sep, 0.050),
            (nu_rest + 0.5 * expected_sep, 0.047),
        ],
        line_sigma_mhz=0.004,
        seed=400 + len(gas_name),
    )

    sha_hex = "a" * 64
    result = pick_surf_doppler_peaks(
        grid,
        intensities,
        gas_name=gas_name,
        snr_threshold=5.0,
        source_step_sha256=sha_hex,
    )
    assert result.gas_name == gas_name
    assert result.beam_velocity_m_s == v_beam
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.feature_kind == SurfFeatureKind.DOPPLER_DOUBLET
    assert cand.rest_frequency_mhz == pytest.approx(nu_rest, abs=0.002)
    assert cand.observed_splitting_mhz == pytest.approx(expected_sep, rel=0.05)
    assert cand.status is AssignmentStatus.PROPOSED
    assert cand.source_step_sha256 == sha_hex


def test_doppler_picker_preserves_blends_uncollapsed_and_forbids_auto_approval() -> None:
    """Verify overlapping triplet blends are never silently collapsed and APPROVED status is rejected."""
    nu_center = 10900.0
    delta_d = expected_doppler_splitting_mhz(nu_center, "Ne")
    grid = np.round(np.linspace(nu_center - 0.3, nu_center + 0.3, 301), 6)

    # Three equally-spaced peaks separated by delta_d -> ambiguous blend (A-B and B-C both match!)
    blend_ints = _build_gaussian_spectrum(
        grid,
        [
            (nu_center - delta_d, 0.045),
            (nu_center, 0.046),
            (nu_center + delta_d, 0.044),
        ],
        line_sigma_mhz=0.004,
        seed=777,
    )

    result = pick_surf_doppler_peaks(grid, blend_ints, gas_name="Ne", snr_threshold=5.0)
    assert len(result.raw_peaks) == 3
    # NONE of the 3 blended peaks may be silently collapsed into a doublet!
    assert len(result.candidates) == 3
    assert all(c.feature_kind == SurfFeatureKind.UNRESOLVED_BLEND for c in result.candidates)
    assert all(len(c.raw_components) == 1 for c in result.candidates)
    assert all(c.status is AssignmentStatus.PROPOSED for c in result.candidates)

    # Attempting to forge an APPROVED SurfCandidateTransition must raise ValidationError
    with pytest.raises(ValidationError):
        SurfCandidateTransition(
            candidate_id="FORGED_APPROVED",
            rest_frequency_mhz=10900.0,
            uncertainty_mhz=0.002,
            intensity=0.04,
            snr=25.0,
            feature_kind=SurfFeatureKind.SINGLET,
            status=AssignmentStatus.APPROVED,
            raw_components=(result.raw_peaks[0],),
        )


# =============================================================================
# 6. INTAKE AUTO-DETECTION & VIEWPORT LIVE STEP OVERLAY TESTS
# =============================================================================


def test_ingest_auto_detects_surf_csv_without_breaking_assigned_csv(tmp_path: Path) -> None:
    """Verify cochem_spyfy_ingest auto-dispatches SURF FFT CSVs while keeping assigned CSVs intact."""
    step_csv = FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv"
    finalized_csv = FIXTURE_RUN_DIR / "sample_run.csv"

    assert is_surf_fft_csv(step_csv) is True
    assert is_surf_fft_csv(finalized_csv) is True
    assert is_surf_fft_csv(io.StringIO(finalized_csv.read_text(encoding="utf-8"))) is True

    df_step = parse_csv_spectroscopy(step_csv)
    assert list(df_step.columns) == ["Frequency_MHz", "Intensity", "Center_Freq_MHz", "Cavity_Position_mm"]
    assert df_step["Frequency_MHz"].dtype == np.float64
    assert df_step["Intensity"].dtype == np.float64
    assert df_step.attrs["source_format"] == FileFormat.SURF_CSV.value
    assert df_step.attrs["surf_file_sha256"] == compute_file_sha256(step_csv)
    assert df_step.attrs["surf_center_freqs_mhz"] == (10899.0,)

    res_final = parse_spectroscopy_data(finalized_csv)
    assert res_final.success is True
    assert len(res_final.df) == 202
    assert res_final.df.attrs["surf_is_compacted"] is True
    assert res_final.df.attrs["surf_center_freqs_mhz"] == (10899.0, 10899.5)
    assert res_final.report.syntax_dropped_count == 0
    assert res_final.report.physics_dropped_count == 0

    # Existing assigned-transition CSV path MUST remain completely unchanged
    assigned_csv = tmp_path / "assigned_h2o.csv"
    assigned_csv.write_text(
        "Frequency_MHz,Uncertainty,Intensity,J_upper,Ka_upper,Kc_upper,J_lower,Ka_lower,Kc_lower\n"
        "22235.08,0.005,1.0,1,1,0,1,0,1\n",
        encoding="utf-8",
    )
    assert is_surf_fft_csv(assigned_csv) is False
    df_assigned = parse_csv_spectroscopy(assigned_csv)
    assert list(df_assigned.columns)[:4] == ["Frequency_MHz", "Uncertainty", "Intensity", "J_upper"]
    assert len(df_assigned) == 1


def test_viewport_live_surf_step_overlay_and_cavity_health(tmp_path: Path) -> None:
    """Verify SpectralViewport.update_surf_live_step updates overlay, health telemetry, and PROPOSED peaks."""
    seed_csv = tmp_path / "initial_spectrum.csv"
    seed_csv.write_text("frequency_mhz,intensity\n10898.5,0.001\n10899.5,0.002\n", encoding="utf-8")
    jac = np.array([[0.0, 1.0, -1.0]], dtype=np.float64)

    viewport = SpectralViewport(
        experimental_csv_path=seed_csv,
        cached_jacobian=jac,
        parameter_names=["A", "B", "C"],
    )
    viewport.build_figure([])

    run_dir = tmp_path / "live_vp_0"
    shutil.copytree(FIXTURE_RUN_DIR, run_dir)
    now_holder = [0.0]
    watcher = SurfRunWatcher(
        run_dir,
        stability_polls=1,
        min_stable_seconds=0.0,
        clock=lambda: now_holder[0],
    )
    updates = watcher.poll_once()
    assert len(updates) == 2

    health_1 = viewport.update_surf_live_step(updates[0])
    assert health_1["status"] == "healthy"
    assert health_1["step_count"] == 1
    assert health_1["last_center_freq_mhz"] == 10899.0
    assert health_1["last_cavity_position_mm"] == pytest.approx(18.425)
    assert viewport.experimental_freqs is not None
    assert viewport.experimental_freqs.size == 101

    health_2 = viewport.update_surf_live_step(updates[1])
    assert health_2["step_count"] == 2
    assert health_2["dropout_count"] == 0
    assert viewport.experimental_freqs.size == 202
    assert len(viewport.surf_proposed_candidates) >= 3
    assert all(c.status is AssignmentStatus.PROPOSED for c in viewport.surf_proposed_candidates)

    # Inject a rail-dropout step at 50.0 mm -> health flips to rail_dropout
    health_3 = viewport.update_surf_live_step(
        freqs_mhz=np.array([10899.8, 10899.9, 10900.0]),
        intensities=np.array([0.003, 0.004, 0.003]),
        center_freq_mhz=10900.0,
        cavity_position_mm=50.0,
    )
    assert health_3["status"] == "rail_dropout"
    assert health_3["dropout_count"] == 1
    assert health_3["dropout_freqs_mhz"] == (10900.0,)

    # Out-of-bounds Zaber position must raise ViewportError
    with pytest.raises(ViewportError):
        viewport.update_surf_live_step(
            freqs_mhz=np.array([10900.5]),
            intensities=np.array([0.002]),
            center_freq_mhz=10900.5,
            cavity_position_mm=55.0,
        )

    # Attempting to push a non-PROPOSED candidate into live viewport must raise ViewportError
    class _ApprovedRecord:
        status = AssignmentStatus.APPROVED
        rest_frequency_mhz = 10900.5
        intensity = 0.05
        feature_kind = SurfFeatureKind.SINGLET

    with pytest.raises(ViewportError):
        viewport.update_surf_live_step(
            freqs_mhz=np.array([10900.4, 10900.5]),
            intensities=np.array([0.002, 0.003]),
            center_freq_mhz=10900.5,
            cavity_position_mm=18.5,
            proposed_candidates=[_ApprovedRecord()],
        )


def test_watcher_resets_stability_gate_when_step_file_still_growing(tmp_path: Path) -> None:
    """Verify watcher refuses to ingest a growing step CSV until (size, mtime_ns) stabilizes."""
    now_holder = [200.0]
    run_dir = tmp_path / "growing_run_0"
    (run_dir / "CavityFiles").mkdir(parents=True)
    shutil.copy2(FIXTURE_RUN_DIR / "scan_settings.toml", run_dir / "scan_settings.toml")

    watcher = SurfRunWatcher(
        run_dir,
        stability_polls=2,
        min_stable_seconds=0.25,
        clock=lambda: now_holder[0],
    )
    watcher.poll_once()
    now_holder[0] += 0.30
    watcher.poll_once()

    growing_csv = run_dir / "CavityFiles" / "10899.0.csv"
    full_lines = (FIXTURE_RUN_DIR / "CavityFiles" / "10899.0.csv").read_text(encoding="utf-8").splitlines(keepends=True)

    # Write partial prefix first
    growing_csv.write_bytes("".join(full_lines[:50]).encode("utf-8"))
    assert watcher.poll_once() == []

    # File finishes writing before second poll -> size changes, so stability counter MUST reset
    now_holder[0] += 0.30
    growing_csv.write_bytes("".join(full_lines).encode("utf-8"))
    assert watcher.poll_once() == []

    # Now stable across consecutive polls with elapsed >= 0.25 s -> ingests complete file
    now_holder[0] += 0.30
    updates = watcher.poll_once()
    assert len(updates) == 1
    assert updates[0].step_data.row_count == 4001
    assert updates[0].step_data.file_sha256 == compute_file_sha256(growing_csv)


# =============================================================================
# 7. STRICT ENCODING, ZERO-PASS & AST INTEGRITY AUDIT
# =============================================================================


def test_spyc_surf_03_files_lf_zero_pass_ast_integrity() -> None:
    """Verify UTF-8 LF, zero pass statements, and zero banned test-double imports across owned SPYC-SURF-03 files."""
    owned_py_files = [
        SPYCFIT_SRC / "cochem_spyfy" / "instruments" / "surf_ingest.py",
        SPYCFIT_SRC / "cochem_spyfy" / "instruments" / "surf_watcher.py",
        SPYCFIT_SRC / "cochem_spyfy" / "intake" / "cochem_spyfy_ingest.py",
        SPYCFIT_SRC / "cochem_spyfy" / "interfaces" / "cochem_spyfy_ui_viewport.py",
        Path(__file__).resolve(),
    ]

    for py_path in owned_py_files:
        raw = py_path.read_bytes()
        assert b"\r\n" not in raw and b"\r" not in raw, f"Non-LF line endings in {py_path.name}"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM in {py_path.name}"

        tree = ast.parse(raw.decode("utf-8"), filename=str(py_path))
        passes = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert passes == [], f"Prohibited 'pass' statements in {py_path.name} at lines {passes}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower()
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower()
