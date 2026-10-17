"""
Unit tests for FAIR publication export and legacy format serialization.
Verifies LaTeX tables, methods paragraph, Parquet cryptographic provenance,
and legacy Pickett Fortran file generation.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import pyarrow.parquet as pq
import pytest

from cochem_spycfit.core_engine.cochem_spycfit_state import CommitManager, FitState
from cochem_spycfit.serialization.cochem_spycfit_export import FairExporter


def get_authentic_water_assignments() -> (
    Tuple[Tuple[int, int, int, int, int, int, float], ...]
):
    """Return authentic ground-state H2O rotational transitions (MHz)."""
    return (
        (1, 1, 0, 1, 0, 1, 556936.0020),
        (2, 1, 1, 2, 0, 2, 752033.1380),
        (2, 2, 0, 2, 1, 1, 987926.7590),
        (3, 1, 3, 2, 2, 0, 183310.0870),
        (6, 1, 6, 5, 2, 3, 22235.0800),
    )


def get_authentic_water_constants() -> Dict[str, float]:
    """Return authentic ground-state H2O rotational constants (MHz)."""
    return {
        "A": 835840.300000,
        "B": 435351.700000,
        "C": 278139.800000,
        "DJ": 37.600000,
        "DJK": -172.000000,
        "DK": 648.000000,
    }


def create_water_experimental_csv(csv_path: Path) -> Path:
    """Create experimental CSV spectrum file with authentic transition frequencies."""
    content = (
        "frequency_mhz,intensity\n"
        "22235.0800,1.250\n"
        "183310.0870,0.910\n"
        "556936.0020,1.000\n"
        "752033.1380,0.850\n"
        "987926.7590,0.720\n"
    )
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def create_water_landscape_h5(h5_path: Path, uuid_str: str) -> Path:
    """Create authentic HDF5 landscape file with metadata UUID."""
    with h5py.File(h5_path, "w") as h5_handle:
        h5_handle.attrs["uuid"] = uuid_str
        h5_handle.create_dataset(
            "rotational_energies_cm",
            data=[0.0, 23.79, 70.09, 134.9, 206.3],
        )
    return h5_path


class DirectSpfitValidator:
    """Interface stand-in for the run_spfit protocol.

    It is not SPFIT: its numbers are test inputs that let the export layout be checked.
    Genuine SPFIT runs are tested in tests/test_cochem_spycfit_pickett_output.py.
    """

    def __init__(self, uncertainties: Dict[str, float], rmsd_khz: float) -> None:
        self._uncertainties = uncertainties
        self._rmsd_khz = rmsd_khz

    def run_spfit(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
    ) -> Dict[str, Any]:
        return {
            "constants": physical_constants,
            "uncertainties": self._uncertainties,
            "rmsd_khz": self._rmsd_khz,
            "n_lines": len(assignments),
            "exit_code": 0,
        }


def test_fair_exporter_initialization_custom_dir(tmp_path: Path) -> None:
    """Verify initialization with custom export directory."""
    export_dir = tmp_path / "custom_exports"
    exporter = FairExporter(export_dir=export_dir)
    assert export_dir.exists()
    assert exporter._export_dir == export_dir


def test_fair_exporter_initialization_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify initialization resolving COCHEM_ARTIFACTS_ROOT."""
    artifacts_root = tmp_path / "env_artifacts"
    monkeypatch.setenv("COCHEM_ARTIFACTS_ROOT", str(artifacts_root))
    exporter = FairExporter()
    expected_dir = artifacts_root / "exports"
    assert exporter._export_dir == expected_dir
    assert expected_dir.exists()


def test_fair_exporter_finalize_full_pipeline(tmp_path: Path) -> None:
    """Verify the full finalize pipeline output without external binary."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    h5_path = tmp_path / "water_landscape.h5"

    create_water_experimental_csv(csv_path)
    expected_uuid = "cochem-water-h5-uuid-2026"
    create_water_landscape_h5(h5_path, expected_uuid)

    commit_manager = CommitManager(registry_path=registry_path)
    water_assignments = get_authentic_water_assignments()
    water_constants = get_authentic_water_constants()

    fit_state = commit_manager.commit(
        locked_assignments=water_assignments,
        physical_constants=water_constants,
        global_rmsd=4.25,
        global_chi2=1.02,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=h5_path,
    )

    # 1. Verify outputs dictionary contains all expected keys
    assert "constants_table_tex" in outputs
    assert "linelist_tex" in outputs
    assert "methods_tex" in outputs
    assert "parquet" in outputs
    assert "var" in outputs
    assert "lin" not in outputs
    assert "lin_absent" in outputs

    # 2. Verify all files exist
    for file_key, out_file in outputs.items():
        assert out_file.exists(), f"Output file for {file_key} does not exist: {out_file}"
        assert out_file.parent == export_dir, f"File leaked outside export dir: {out_file}"

    # 3. Verify constants table LaTeX
    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{lSSs}" in table_content
    assert r"\tablecaption{Spectroscopic Constants (SpycFit fit state)\label{tab:constants}}" in table_content
    assert r"\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Uncertainty} & \colhead{Unit}}" in table_content
    assert r"\startdata" in table_content
    # FitState holds no uncertainties, so they read unknown (SF-OUT-002); values keep
    # full precision and constants stored in MHz are labelled MHz.
    assert "A & 835840.3 & {unknown} & {MHz}" in table_content
    assert "B & 435351.7 & {unknown} & {MHz}" in table_content
    assert "C & 278139.8 & {unknown} & {MHz}" in table_content
    assert "DJ & 37.6 & {unknown} & {MHz}" in table_content
    assert "DJK & -172.0 & {unknown} & {MHz}" in table_content
    assert "DK & 648.0 & {unknown} & {MHz}" in table_content
    assert r"{kHz}" not in table_content
    assert r"\enddata" in table_content
    assert r"\end{deluxetable}" in table_content

    # 4. Verify linelist table LaTeX
    linelist_content = outputs["linelist_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{cSSS}" in linelist_content
    assert r"\tablecaption{Observed Transition Frequencies\label{tab:linelist}}" in linelist_content
    assert r"$1(1,0) \leftarrow 1(0,1)$ & 556936.0020 & {---} & {---}" in linelist_content
    assert r"$2(1,1) \leftarrow 2(0,2)$ & 752033.1380 & {---} & {---}" in linelist_content
    assert r"$2(2,0) \leftarrow 2(1,1)$ & 987926.7590 & {---} & {---}" in linelist_content
    assert r"$3(1,3) \leftarrow 2(2,0)$ & 183310.0870 & {---} & {---}" in linelist_content
    assert r"$6(1,6) \leftarrow 5(2,3)$ & 22235.0800 & {---} & {---}" in linelist_content
    assert r"\enddata" in linelist_content
    assert r"\end{deluxetable}" in linelist_content

    # 5. Verify methods paragraph
    methods_content = outputs["methods_tex"].read_text(encoding="utf-8")
    # The fit state records no reduction or representation, so none is asserted.
    assert "Watson's $A$-reduction" not in methods_content
    assert "reduction and representation are not recorded" in methods_content
    assert "A total of 5 transitions were included in the fit" in methods_content
    assert "$J = 6$" in methods_content
    assert "$K_a = 2$" in methods_content
    assert "6 spectroscopic constants" in methods_content
    assert "-1 degrees of freedom" in methods_content or "degrees of freedom" in methods_content
    assert r"\SI{4.25}{kHz}" in methods_content

    # 6. Verify Parquet serialization and provenance metadata
    parquet_table = pq.read_table(outputs["parquet"])
    assert parquet_table.num_rows == 5
    assert "J_upper" in parquet_table.column_names
    assert "Ka_upper" in parquet_table.column_names
    assert "Kc_upper" in parquet_table.column_names
    assert "J_lower" in parquet_table.column_names
    assert "Ka_lower" in parquet_table.column_names
    assert "Kc_lower" in parquet_table.column_names
    assert "nu_obs_mhz" in parquet_table.column_names

    schema_metadata = parquet_table.schema.metadata
    assert b"cochem_git_hash" in schema_metadata
    assert b"experimental_csv_sha256" in schema_metadata
    assert b"landscape_h5_uuid" in schema_metadata
    assert b"export_timestamp" in schema_metadata
    assert b"state_id" in schema_metadata

    assert schema_metadata[b"landscape_h5_uuid"].decode("utf-8") == expected_uuid
    assert schema_metadata[b"state_id"].decode("utf-8") == fit_state.state_id

    # Verify CSV sha256 matches
    csv_bytes = csv_path.read_bytes()
    expected_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    assert schema_metadata[b"experimental_csv_sha256"].decode("utf-8") == expected_csv_sha

    # 7. Verify legacy .var file formatting
    var_content = outputs["var"].read_text(encoding="ascii")
    var_lines = [l for l in var_content.splitlines() if l.strip()]
    assert len(var_lines) == 6
    for line in var_lines:
        param_name = line[:10].strip()
        val_str = line[10:].strip()
        assert param_name in water_constants
        assert float(val_str) == pytest.approx(water_constants[param_name])

    # 8. No legacy .lin without measurement uncertainties; the reason is recorded.
    # At 7e42507 every line was written with an invented 0.0050 MHz uncertainty.
    assert not (export_dir / "cochem_export.lin").exists()
    absent_text = outputs["lin_absent"].read_text(encoding="utf-8")
    assert "carry no measurement uncertainties" in absent_text
    assert "5 locked assignments" in absent_text


def test_fair_exporter_with_spfit_wrapper(tmp_path: Path) -> None:
    """Verify finalize with active SPFIT validator providing uncertainties."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    water_assignments = get_authentic_water_assignments()
    water_constants = get_authentic_water_constants()

    fit_state = commit_manager.commit(
        locked_assignments=water_assignments,
        physical_constants=water_constants,
        global_rmsd=3.82,
    )

    validator_uncertainties = {
        "A": 0.051200,
        "B": 0.032100,
        "C": 0.021000,
        "DJ": 0.001200,
        "DJK": 0.003400,
        "DK": 0.005600,
    }
    validator = DirectSpfitValidator(
        uncertainties=validator_uncertainties,
        rmsd_khz=3.82,
    )

    exporter = FairExporter(export_dir=export_dir, spfit_wrapper=validator)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    # Regression: at 7e42507 the table paired fit-state values with SPFIT uncertainties
    # (amendment §5). Each result now has its own table (SF-OUT-001).
    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert "A & 835840.3 & {unknown} & {MHz}" in table_content
    assert "0.0512" not in table_content

    spfit_table = outputs["spfit_constants_table_tex"].read_text(encoding="utf-8")
    assert "SPFIT" in spfit_table
    assert "A & 835840.3 & 0.0512 & {MHz}" in spfit_table
    assert "B & 435351.7 & 0.0321 & {MHz}" in spfit_table
    assert "C & 278139.8 & 0.021 & {MHz}" in spfit_table


def test_fair_exporter_missing_optional_files(tmp_path: Path) -> None:
    """Verify exporter handles missing landscape.h5 and non-git directory gracefully."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    fit_state = commit_manager.commit(
        locked_assignments=get_authentic_water_assignments(),
        physical_constants=get_authentic_water_constants(),
        global_rmsd=4.10,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    parquet_table = pq.read_table(outputs["parquet"])
    schema_metadata = parquet_table.schema.metadata
    assert schema_metadata[b"landscape_h5_uuid"] == b""


def test_fair_exporter_empty_state(tmp_path: Path) -> None:
    """Verify exporter functions correctly with empty state."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    fit_state = commit_manager.commit(
        locked_assignments=(),
        physical_constants={},
        global_rmsd=0.0,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{lSSs}" in table_content
    assert r"\end{deluxetable}" in table_content

    methods_content = outputs["methods_tex"].read_text(encoding="utf-8")
    assert "A total of 0 transitions were included in the fit" in methods_content
    assert "$J = 0$" in methods_content
    assert "$K_a = 0$" in methods_content


def test_fair_exporter_sha256_and_git_hash(tmp_path: Path) -> None:
    """Verify SHA-256 computation and Git hash retrieval helper."""
    test_file = tmp_path / "integrity_test.txt"
    test_content = b"CoChem Spectroscopic Data Payload 2026"
    test_file.write_bytes(test_content)

    expected_hash = hashlib.sha256(test_content).hexdigest()
    computed_hash = FairExporter._compute_file_sha256(test_file)
    assert computed_hash == expected_hash

    git_hash = FairExporter._get_git_commit_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0


def test_fair_exporter_various_distortion_units(tmp_path: Path) -> None:
    """Verify unit mapping for various centrifugal distortion constants."""
    export_dir = tmp_path / "exports"
    exporter = FairExporter(export_dir=export_dir)

    registry_path = tmp_path / "cochem_fit_registry.json"
    commit_manager = CommitManager(registry_path=registry_path)
    distortion_constants = {
        "A": 835840.3,
        "B": 435351.7,
        "C": 278139.8,
        "Delta_J": 0.0376,
        "Delta_JK": -0.1720,
        "Delta_K": 0.6480,
        "delta_J": 0.0050,
        "delta_K": 0.0820,
    }
    fit_state = commit_manager.commit(
        locked_assignments=get_authentic_water_assignments(),
        physical_constants=distortion_constants,
        global_rmsd=4.25,
    )

    table_path = exporter._generate_constants_table(fit_state, spfit_result=None)
    table_content = table_path.read_text(encoding="utf-8")
    assert "Delta_J" in table_content
    assert "Delta_JK" in table_content
    assert "Delta_K" in table_content
    assert "delta_J" in table_content
    assert "delta_K" in table_content
    # Regression: these MHz values were labelled kHz without conversion (amendment §5).
    assert r"{kHz}" not in table_content
    assert "Delta_J & 0.0376 & {unknown} & {MHz}" in table_content

