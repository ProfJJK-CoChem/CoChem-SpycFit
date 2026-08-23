"""Comprehensive physical unit test suite for SpycFit Snapshot & Publication Export.

Verifies:
- FitProvenancePayload schema validation, field validators, serialization & deserialization.
- hash_dataset_iteratively with physical disk operations and chunked reading.
- generate_provenance_lock with physical HDF5 and binary datasets, metadata extraction,
  and serialization to disk.
- generate_aastex_longtable and generate_aastex_longtables with molecular rotational parameters,
  transitions, siunitx formatting, frozen parameter overrides, and top 200 truncation.
- get_required_dois and format_citations_to_bib with structured CrossRef metadata records.
- evaluate_compression_strategy sizing thresholds.
- package_fit_artifacts archiving (.zip and .tar.zst) and seal_artifact_read_only permissions.
- export_spycfit_snapshot end-to-end pipeline execution and artifact sealing.
- Workspace directory resolution with environment variable and platformdirs fallback.
- Dynamic IUPAC atomic weight and isotopic mass resolution via Mendeleev.
"""

from __future__ import annotations

import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zstandard
from pydantic import ValidationError

from cochem_spycfit.interfaces.cochem_vibspyc_snap import (
    FitProvenancePayload,
    evaluate_compression_strategy,
    export_spycfit_snapshot,
    format_citations_to_bib,
    generate_aastex_longtable,
    generate_aastex_longtables,
    generate_provenance_lock,
    get_element_atomic_weight,
    get_isotope_mass,
    get_required_dois,
    get_spycfit_processed_dir,
    hash_dataset_iteratively,
    package_fit_artifacts,
    resolve_processed_workspace_dir,
    seal_artifact_read_only,
)


@pytest.fixture
def clean_workspace(tmp_path: Path):
    """Provides an isolated temporary workspace and resets file permissions upon teardown."""
    yield tmp_path
    for root, _dirs, files in os.walk(tmp_path):
        for fname in files:
            target = os.path.join(root, fname)
            try:
                os.chmod(target, stat.S_IWRITE)
            except OSError as err:
                _ = err


@pytest.fixture
def authoritative_payload() -> FitProvenancePayload:
    """Constructs a valid, physically realistic FitProvenancePayload instance."""
    return FitProvenancePayload(
        session_id="SESSION_METHANOL_CH3OH_FIT_2026_01",
        timestamp="2026-08-23T14:30:00+00:00",
        dataset_hashes={
            "methanol_spectrum_chirped_pulse.dat": "a3f5b7c891024d3e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e",
            "assigned_transitions_methanol.cat": "8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d",
        },
        chi_squared=1.0428,
        huber_loss_delta=1.345,
        rms_mhz=0.0184,
        rms_cm_inv=0.00061376,
        jacobian_condition_number=142.85,
        sobol_parameter_audit={
            "A": True,
            "B": True,
            "C": True,
            "D_J": True,
            "D_JK": True,
            "D_K": False,
            "V_3": True,
        },
        semantic_git_history=[
            "git-commit-8f2a1b4: Initial Hamiltonian configuration for internal rotation",
            "git-commit-c9d3e5f: Refined centrifugal distortion parameters via Levenberg-Marquardt",
            "git-commit-4a7b9c0: Converged global spectroscopic fit with Huber robust weighting",
        ],
        active_models=[
            "watson_a_reduction",
            "iam_internal_rotation",
            "spcat",
            "jax_backend",
        ],
        additional_metadata={
            "spectrometer": "CP-FTMW 2-8 GHz",
            "carrier_gas": "Neon",
            "backing_pressure_bar": 2.5,
            "pulse_duration_us": 1.0,
            "temperature_k": 2.0,
        },
    )


@pytest.fixture
def molecular_parameters() -> list[dict[str, Any]]:
    """Realistic rotational and distortion parameters for methanol (CH3OH)."""
    return [
        {
            "name": "A",
            "latex_name": "A",
            "value": 127634.508,
            "uncertainty": 0.045,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal a-axis",
        },
        {
            "name": "B",
            "latex_name": "B",
            "value": 24684.382,
            "uncertainty": 0.012,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal b-axis",
        },
        {
            "name": "C",
            "latex_name": "C",
            "value": 23758.129,
            "uncertainty": 0.011,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal c-axis",
        },
        {
            "name": "D_J",
            "latex_name": "\\Delta_J",
            "value": 0.04021,
            "uncertainty": 0.00035,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_JK",
            "latex_name": "\\Delta_{JK}",
            "value": 0.12480,
            "uncertainty": 0.00080,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_K",
            "latex_name": "\\Delta_K",
            "value": 0.89200,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": False,
            "status": "fixed",
            "description": "Frozen quartic centrifugal distortion parameter from ab initio",
        },
        {
            "name": "V_3",
            "latex_name": "V_3",
            "value": 373.800,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": True,
            "status": "set",
            "description": "Three-fold torsional potential barrier height in cm^-1",
        },
    ]


@pytest.fixture
def molecular_transitions() -> list[dict[str, Any]]:
    """Calculated test list of 250 assigned transitions for methanol."""
    records: list[dict[str, Any]] = []
    for i in range(1, 251):
        j_upper = i % 10 + 1
        j_lower = i % 10
        ka_upper = min(j_upper, (i % 3))
        kc_upper = j_upper - ka_upper
        ka_lower = min(j_lower, (i % 3))
        kc_lower = j_lower - ka_lower

        obs_freq = 12000.0 + (i * 25.35)
        calc_freq = obs_freq + (((i % 7) - 3) * 0.003)
        res = obs_freq - calc_freq
        intensity = 1000.0 / (i + 1.0)

        records.append(
            {
                "upper_state": f"{j_upper}_{{{ka_upper},{kc_upper}}}",
                "lower_state": f"{j_lower}_{{{ka_lower},{kc_lower}}}",
                "observed_mhz": obs_freq,
                "calculated_mhz": calc_freq,
                "residual_mhz": res,
                "uncertainty_mhz": 0.010,
                "intensity": intensity,
            }
        )
    return records


# =====================================================================
# 1. FitProvenancePayload Validation and Serialization Tests
# =====================================================================


def test_fit_provenance_payload_validation(authoritative_payload: FitProvenancePayload):
    """Verifies valid instantiation and field access on FitProvenancePayload."""
    assert authoritative_payload.session_id == "SESSION_METHANOL_CH3OH_FIT_2026_01"
    assert authoritative_payload.chi_squared == 1.0428
    assert authoritative_payload.rms_mhz == 0.0184
    assert authoritative_payload.rms_cm_inv == 0.00061376
    assert authoritative_payload.jacobian_condition_number == 142.85
    assert len(authoritative_payload.dataset_hashes) == 2
    assert authoritative_payload.sobol_parameter_audit["V_3"] is True
    assert len(authoritative_payload.semantic_git_history) == 3
    assert "watson_a_reduction" in authoritative_payload.active_models


def test_fit_provenance_payload_invalid_timestamp():
    """Verifies that non-ISO formatted timestamps are rejected."""
    with pytest.raises(ValidationError) as excinfo:
        FitProvenancePayload(
            session_id="INVALID_TIMESTAMP_SESSION",
            timestamp="23-08-2026 14:30",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )
    assert "Timestamp" in str(excinfo.value) or "ISO" in str(excinfo.value)


def test_fit_provenance_payload_negative_metrics():
    """Verifies that negative error metrics trigger validation errors."""
    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_CHI_SQ_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=-0.5,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_RMS_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=-0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_JACOBIAN_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=-5.0,
        )


def test_fit_provenance_disk_roundtrip(
    authoritative_payload: FitProvenancePayload, clean_workspace: Path
):
    """Verifies physical disk serialization and deserialization roundtrips."""
    target_json = clean_workspace / "provenance" / "fit_provenance.json"
    written_path = authoritative_payload.to_json_file(target_json)

    assert written_path.exists()
    assert written_path.is_file()

    loaded_payload = FitProvenancePayload.from_json_file(written_path)
    assert loaded_payload.session_id == authoritative_payload.session_id
    assert loaded_payload.timestamp == authoritative_payload.timestamp
    assert loaded_payload.chi_squared == authoritative_payload.chi_squared
    assert loaded_payload.rms_mhz == authoritative_payload.rms_mhz
    assert loaded_payload.dataset_hashes == authoritative_payload.dataset_hashes
    assert (
        loaded_payload.sobol_parameter_audit
        == authoritative_payload.sobol_parameter_audit
    )
    assert (
        loaded_payload.semantic_git_history
        == authoritative_payload.semantic_git_history
    )
    assert (
        loaded_payload.additional_metadata == authoritative_payload.additional_metadata
    )


def test_fit_provenance_from_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to deserialize from a non-existent file."""
    non_existent = clean_workspace / "non_existent_provenance.json"
    with pytest.raises(FileNotFoundError):
        FitProvenancePayload.from_json_file(non_existent)


# =====================================================================
# 2. Iterative Hashing Tests
# =====================================================================


def test_hash_dataset_iteratively_small_file(clean_workspace: Path):
    """Verifies iterative SHA-256 calculation for a physical file on disk."""
    data_file = clean_workspace / "spectrum_input.dat"
    content = b"CH3OH Spectroscopic Data Stream v1.0\n12450.320 0.85\n12480.110 0.92\n"
    data_file.write_bytes(content)

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=16)
    expected_hash = "bca9b04d490184c37b9ae03358c4370f955c050a1953e3b54674122459ecf77c"

    import hashlib

    direct_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == expected_hash == direct_hash


def test_hash_dataset_iteratively_large_file(clean_workspace: Path):
    """Verifies iterative hashing across multi-chunk files exceeding chunk boundaries."""
    data_file = clean_workspace / "large_spectrum.bin"
    payload_chunk = b"RotationalTransitionRecord_1234567890_abcdefghij_" * 32
    total_chunks = 320
    with open(data_file, "wb") as f:
        f.writelines(payload_chunk for _ in range(total_chunks))

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=8192)
    assert isinstance(computed_hash, str)
    assert len(computed_hash) == 64


def test_hash_dataset_iteratively_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError on non-existent dataset path."""
    missing_file = clean_workspace / "non_existent_dataset.bin"
    with pytest.raises(FileNotFoundError):
        hash_dataset_iteratively(missing_file)


# =====================================================================
# 3. Provenance Lock Generation Tests (Physical HDF5 / Binary)
# =====================================================================


def test_generate_provenance_lock_hdf5(clean_workspace: Path):
    """Verifies cryptographic provenance lock creation on a physical HDF5 landscape file."""
    h5_file = clean_workspace / "landscape.h5"
    with h5py.File(h5_file, "w") as f:
        grp = f.create_group("spectroscopy")
        grp.create_dataset("frequencies", data=np.array([12450.32, 12480.11, 12510.95]))
        grp.create_dataset("intensities", data=np.array([0.85, 0.92, 0.45]))
        grp.attrs["molecule"] = "Methanol-CH3OH"
        grp.attrs["fit_converged"] = True

    metrics = {
        "session_id": "SESSION_H5_LOCK_TEST_01",
        "timestamp": "2026-08-23T15:00:00+00:00",
        "chi_squared": 1.025,
        "rms_mhz": 0.0142,
        "jacobian_condition_number": 88.5,
        "active_parameters": ["A", "B", "C", "D_J", "D_JK"],
        "frozen_parameters": ["D_K", "V_3"],
        "semantic_git_history": ["commit 1: Fitted ground state"],
        "active_models": ["watson_a", "spcat"],
        "metadata": {"spectrometer": "CP-FTMW"},
    }

    result = generate_provenance_lock(h5_file, metrics)

    assert result["session_id"] == "SESSION_H5_LOCK_TEST_01"
    assert result["chi_squared"] == 1.025
    assert result["rms_mhz"] == 0.0142
    assert "landscape.h5" in result["dataset_hashes"]
    expected_hash = hash_dataset_iteratively(h5_file)
    assert result["dataset_hashes"]["landscape.h5"] == expected_hash
    assert result["sobol_parameter_audit"]["A"] is True
    assert result["sobol_parameter_audit"]["D_K"] is False

    default_json = clean_workspace / "fit_provenance.json"
    assert default_json.exists()
    loaded_payload = FitProvenancePayload.from_json_file(default_json)
    assert loaded_payload.session_id == "SESSION_H5_LOCK_TEST_01"
    assert loaded_payload.dataset_hashes["landscape.h5"] == expected_hash


def test_generate_provenance_lock_with_payload_object(
    clean_workspace: Path, authoritative_payload: FitProvenancePayload
):
    """Verifies generate_provenance_lock when supplied with a pre-built FitProvenancePayload."""
    h5_file = clean_workspace / "state.h5"
    h5_file.write_bytes(b"HDF5_STRUCTURED_BINARY_STREAM_SIMULATION")

    custom_out = clean_workspace / "custom_dir" / "custom_provenance.json"
    result = generate_provenance_lock(
        h5_path=h5_file,
        metrics=authoritative_payload,
        output_path=custom_out,
    )

    assert custom_out.exists()
    assert result["session_id"] == authoritative_payload.session_id
    assert "state.h5" in result["dataset_hashes"]
    assert result["dataset_hashes"]["state.h5"] == hash_dataset_iteratively(h5_file)


def test_generate_provenance_lock_missing_h5_file(clean_workspace: Path):
    """Verifies FileNotFoundError when dataset file is missing."""
    missing_h5 = clean_workspace / "missing_landscape.h5"
    with pytest.raises(FileNotFoundError):
        generate_provenance_lock(missing_h5, {"chi_squared": 1.0, "rms_mhz": 0.01})


def test_generate_provenance_lock_invalid_metrics_type(clean_workspace: Path):
    """Verifies TypeError when invalid metrics argument is passed."""
    data_file = clean_workspace / "dataset.bin"
    data_file.write_bytes(b"DATA")
    with pytest.raises(TypeError, match="Expected metrics to be dict or FitProvenancePayload"):
        generate_provenance_lock(data_file, "invalid_string_metrics")  # type: ignore[arg-type]


# =====================================================================
# 4. AASTeX / LaTeX Table Generation Tests
# =====================================================================


def test_generate_aastex_longtable_with_dict():
    """Verifies generate_aastex_longtable with simple dictionary input."""
    constants = {
        "A": 127634.508,
        "B": 24684.382,
        "C": 23758.129,
        "D_J": 0.04021,
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Simple Fit")

    assert "\\begin{table}[htbp]" in rendered
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in rendered
    assert "CH3OH Simple Fit" in rendered
    assert "127634.508000" in rendered
    assert "24684.382000" in rendered
    assert "\\Delta_J" in rendered
    assert "\\bottomrule" in rendered


def test_generate_aastex_longtable_with_tuples_and_frozen():
    """Verifies generate_aastex_longtable with tuple inputs and frozen status."""
    constants = {
        "A": (127634.508, 0.045),
        "B": (24684.382, 0.012),
        "D_K": (0.892, None, True),
        "V_3": {"value": 373.8, "status": "set", "is_frozen": True},
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Tuple Test")

    assert "0.045000" in rendered
    assert "0.012000" in rendered
    assert "Fixed" in rendered
    assert "Set" in rendered
    assert "V_3" in rendered


def test_generate_aastex_longtable_with_transitions(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies generate_aastex_longtable with transitions producing full combined document."""
    rendered = generate_aastex_longtable(
        constants_dict=molecular_parameters,
        title_prefix="Methanol Combined Publication",
        transitions=molecular_transitions,
    )

    assert "\\documentclass[twocolumn]{aastex631}" in rendered
    assert "\\begin{longtable}" in rendered
    assert "\\begin{table}[htbp]" in rendered
    assert "Methanol Combined Publication" in rendered
    assert "\\end{document}" in rendered


def test_generate_aastex_longtable_invalid_input_type():
    """Verifies TypeError when constants_dict is not dict or Sequence."""
    with pytest.raises(TypeError, match="Expected constants_dict to be dict or Sequence"):
        generate_aastex_longtable(12345)


def test_generate_aastex_longtables_structure(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies structure and LaTeX formatting of generated tables and combined documents."""
    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="Methanol CH3OH Ground State",
    )

    assert "parameters_table" in result
    assert "transitions_table" in result
    assert "combined_document" in result

    params_tex = result["parameters_table"]
    trans_tex = result["transitions_table"]
    doc_tex = result["combined_document"]

    assert "\\begin{table}[htbp]" in params_tex
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in params_tex
    assert (
        "\\caption{Optimized Spectroscopic Parameters for Methanol CH3OH Ground State}"
        in params_tex
    )
    assert "\\toprule" in params_tex
    assert "\\midrule" in params_tex
    assert "\\bottomrule" in params_tex
    assert "Rotational constant around principal a-axis" in params_tex

    assert "127634.508000" in params_tex
    assert "0.045000" in params_tex
    assert "Fixed" in params_tex
    assert "Set" in params_tex

    assert "\\begin{longtable}{c c S[table-format=6.4]" in trans_tex
    assert (
        "\\caption{Observed and Calculated Transitions for Methanol CH3OH Ground State (Truncated to Top 200)}"
        in trans_tex
    )
    assert "\\endfirsthead" in trans_tex
    assert "\\endhead" in trans_tex
    assert "\\endfoot" in trans_tex
    assert "\\endlastfoot" in trans_tex

    assert "\\documentclass[twocolumn]{aastex631}" in doc_tex
    assert "\\usepackage{siunitx}" in doc_tex
    assert "\\usepackage{booktabs}" in doc_tex
    assert "\\usepackage{longtable}" in doc_tex
    assert (
        "\\title{High-Precision Rotational Spectroscopic Fit and Transition Assignments: Methanol CH3OH Ground State}"
        in doc_tex
    )


def test_generate_aastex_longtables_truncation(
    molecular_parameters: list[dict[str, Any]],
):
    """Verifies that transitions list is truncated to the top 200 by intensity."""
    many_transitions = [
        {
            "upper_state": f"State_{i}_up",
            "lower_state": f"State_{i}_low",
            "observed_mhz": 10000.0 + i,
            "calculated_mhz": 10000.0 + i,
            "residual_mhz": 0.0,
            "uncertainty_mhz": 0.01,
            "intensity": float(i),
        }
        for i in range(1, 301)
    ]

    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=many_transitions,
        title_prefix="Truncation Verification",
    )

    trans_table = result["transitions_table"]
    assert "(Truncated to Top 200)" in trans_table
    assert "State_300_up" in trans_table
    assert "State_101_up" in trans_table
    assert "State_1_up" not in trans_table


def test_generate_aastex_longtables_unparseable_uncertainty():
    """Verifies graceful handling of non-numeric string uncertainties."""
    params = [
        {
            "name": "CustomParam",
            "value": 12.3456,
            "uncertainty": "est. 0.05",
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
        }
    ]
    res = generate_aastex_longtables(params, [])
    assert "est. 0.05" in res["parameters_table"]


# =====================================================================
# 5. Mendeleev Dynamic Atomic Mass Resolution Tests
# =====================================================================


def test_mendeleev_element_atomic_weights():
    """Verifies dynamic IUPAC standard atomic weight retrieval via Mendeleev."""
    carbon_weight = get_element_atomic_weight("C")
    hydrogen_weight = get_element_atomic_weight("H")
    oxygen_weight = get_element_atomic_weight("O")
    nitrogen_weight = get_element_atomic_weight("N")

    assert 12.010 <= carbon_weight <= 12.012
    assert 1.007 <= hydrogen_weight <= 1.009
    assert 15.998 <= oxygen_weight <= 16.000
    assert 14.006 <= nitrogen_weight <= 14.008


def test_mendeleev_isotope_masses():
    """Verifies dynamic isotopic mass retrieval in atomic mass units via Mendeleev."""
    c12_mass = get_isotope_mass("C", 12)
    c13_mass = get_isotope_mass("C", 13)
    h1_mass = get_isotope_mass("H", 1)
    h2_mass = get_isotope_mass("H", 2)
    o16_mass = get_isotope_mass("O", 16)

    assert 11.999 <= c12_mass <= 12.001
    assert 13.002 <= c13_mass <= 13.004
    assert 1.007 <= h1_mass <= 1.009
    assert 2.013 <= h2_mass <= 2.015
    assert 15.994 <= o16_mass <= 15.996


def test_mendeleev_missing_isotope_error():
    """Verifies ValueError when requesting a non-existent isotope number."""
    with pytest.raises(ValueError, match="not found in Mendeleev database"):
        get_isotope_mass("C", 999)


# =====================================================================
# 6. DOI Resolution and BibTeX Citation Tests
# =====================================================================


def test_get_required_dois():
    """Verifies mapping of active physical models and quantum engines to authoritative DOIs."""
    active_models = [
        "watson_a",
        "spcat",
        "orca_v6",
        "jax",
        "pyarrow",
        "unknown_model_xyz",
    ]
    resolved = get_required_dois(active_models)

    assert "10.1016/0022-2852(77)90184-7" in resolved
    assert "10.1016/0022-2852(91)90393-O" in resolved
    assert "10.1063/5.0004608" in resolved
    assert "10.5281/zenodo.4766323" in resolved
    assert "10.5281/zenodo.4006056" in resolved
    assert len(resolved) == 5


def test_format_citations_to_bib():
    """Verifies parsing and deduplicating CrossRef JSON metadata into clean BibTeX format."""
    crossref_data = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": [
                "The determination of centrifugal distortion constants of asymmetric-top molecules"
            ],
            "author": [{"family": "Watson", "given": "James K. G."}],
            "container-title": ["Journal of Molecular Spectroscopy"],
            "volume": "65",
            "issue": "1",
            "page": "123-133",
            "issued": {"date-parts": [[1977, 4, 1]]},
        },
        {
            "message": {
                "DOI": "10.1016/0022-2852(91)90393-O",
                "title": [
                    "The fitting of sub-Doppler rotational spectra of molecules with internal rotor"
                ],
                "author": [{"family": "Pickett", "given": "Herbert M."}],
                "container-title": "Journal of Molecular Spectroscopy",
                "volume": "148",
                "number": "2",
                "page": "371-377",
                "issued": {"date-parts": [[1991, 8, 1]]},
            }
        },
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Duplicate Watson citation"],
        },
    ]

    bib_text = format_citations_to_bib(crossref_data)

    assert "@article{" in bib_text
    assert "author = {Watson, James K. G.}" in bib_text
    assert (
        "title = {{The determination of centrifugal distortion constants of asymmetric-top molecules}}"
        in bib_text
    )
    assert "journal = {Journal of Molecular Spectroscopy}" in bib_text
    assert "volume = {65}" in bib_text
    assert "number = {1}" in bib_text
    assert "pages = {123-133}" in bib_text
    assert "year = {1977}" in bib_text
    assert "doi = {10.1016/0022-2852(77)90184-7}" in bib_text

    assert "author = {Pickett, Herbert M.}" in bib_text
    assert "doi = {10.1016/0022-2852(91)90393-O}" in bib_text
    assert "year = {1991}" in bib_text

    assert bib_text.count("10.1016/0022-2852(77)90184-7") == 1


def test_format_citations_to_bib_extended_schemas():
    """Verifies parsing of various author schemas, missing DOIs, and year fallbacks."""
    records = [
        {"title": "Missing DOI Entry", "author": "No DOI Author"},
        {
            "DOI": "10.1000/str_author",
            "title": ["Single String Author Paper"],
            "author": ["Jane Doe"],
            "year": "2024",
        },
        {
            "DOI": "10.1000/family_only",
            "title": "Family Only Paper",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
        {
            "DOI": "10.1000/heisenberg_2",
            "title": "Family Only Paper 2",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
    ]

    bib_text = format_citations_to_bib(records)
    assert "Missing DOI Entry" not in bib_text
    assert "author = {Jane Doe}" in bib_text
    assert "author = {Heisenberg}" in bib_text
    assert "year = {2024}" in bib_text
    assert "year = {1927}" in bib_text
    assert "Heisenberg1927Family" in bib_text
    assert "Heisenberg1927Family_1" in bib_text


# =====================================================================
# 7. Compression Strategy and Packaging Tests
# =====================================================================


def test_evaluate_compression_strategy():
    """Verifies mathematical threshold evaluations for storage packaging."""
    small_size = 50 * 1024 * 1024
    assert evaluate_compression_strategy(small_size) == "ZIP"

    exact_threshold = 104857600
    assert evaluate_compression_strategy(exact_threshold) == "ZSTD"

    large_size = 500 * 1024 * 1024
    assert evaluate_compression_strategy(large_size) == "ZSTD"


def test_package_fit_artifacts_zip(clean_workspace: Path):
    """Verifies packaging workspace directory into a standard .zip archive."""
    src_dir = clean_workspace / "fit_export_src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "fit.log").write_text(
        "Fit iteration log output: Converged in 14 iterations.", encoding="utf-8"
    )
    (src_dir / "tables").mkdir()
    (src_dir / "tables" / "table1.tex").write_text(
        "\\begin{table}...\\end{table}", encoding="utf-8"
    )

    archive_base = clean_workspace / "export_archive"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZIP")

    assert archive_path.exists()
    assert archive_path.name == "export_archive.zip"
    assert zipfile.is_zipfile(archive_path)

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        assert "fit.log" in namelist
        assert "tables/table1.tex" in namelist


def test_package_fit_artifacts_zstd(clean_workspace: Path):
    """Verifies packaging workspace directory into a tar.zst archive."""
    src_dir = clean_workspace / "fit_export_src_zstd"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "data.bin").write_bytes(b"SPECTROSCOPIC_RAW_STREAM_DATA" * 500)
    (src_dir / "info.json").write_text('{"status": "READY"}', encoding="utf-8")

    archive_base = clean_workspace / "export_archive_zstd"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZSTD")

    assert archive_path.exists()
    assert archive_path.name == "export_archive_zstd.tar.zst"

    compressed_bytes = archive_path.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    decompressed_tar = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_tar), mode="r") as tf:
        names = tf.getnames()
        assert "data.bin" in names
        assert "info.json" in names


def test_package_fit_artifacts_edge_cases(clean_workspace: Path):
    """Verifies automatic strategy evaluation, extension normalization, and error handling."""
    src = clean_workspace / "edge_case_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "dataset.txt").write_text("Adversarial audit payload", encoding="utf-8")

    auto_archive = package_fit_artifacts(src, clean_workspace / "auto_archive")
    assert auto_archive.exists()
    assert auto_archive.suffix == ".zip"

    tar_zst_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar.zst", strategy="ZSTD"
    )
    assert tar_zst_archive.name == "explicit.tar.zst"
    assert tar_zst_archive.exists()

    tar_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar", strategy="ZSTD"
    )
    assert tar_archive.name == "explicit.tar.zst"
    assert tar_archive.exists()

    with pytest.raises(ValueError, match="Unsupported packaging strategy"):
        package_fit_artifacts(src, clean_workspace / "bad_strat", strategy="SEVEN_ZIP")

    with pytest.raises(NotADirectoryError):
        package_fit_artifacts(
            clean_workspace / "non_existent_folder", clean_workspace / "out"
        )


# =====================================================================
# 8. Read-Only Sealing Tests
# =====================================================================


def test_seal_artifact_read_only(clean_workspace: Path):
    """Verifies cross-platform read-only permissions are enforced on target artifacts."""
    target_file = clean_workspace / "sealed_artifact.json"
    target_file.write_text('{"immutable": true}', encoding="utf-8")

    assert seal_artifact_read_only(target_file) is True

    mode = target_file.stat().st_mode
    assert not (mode & stat.S_IWUSR)

    if os.name != "nt":
        with pytest.raises(PermissionError), open(target_file, "a") as f:
            f.write("mutation")


def test_seal_artifact_read_only_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to seal non-existent file."""
    missing = clean_workspace / "phantom_artifact.bin"
    with pytest.raises(FileNotFoundError):
        seal_artifact_read_only(missing)


# =====================================================================
# 9. End-to-End export_spycfit_snapshot Orchestration Tests
# =====================================================================


def test_export_spycfit_snapshot_pipeline(
    authoritative_payload: FitProvenancePayload,
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
    clean_workspace: Path,
):
    """Verifies full orchestration pipeline from input files to sealed archive."""
    source_dir = clean_workspace / "source_workspace"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "raw_spectrum.dat").write_text(
        "12000.0 0.5\n12050.0 0.9\n", encoding="utf-8"
    )

    processed_dir = clean_workspace / "processed_workspace"

    crossref_records = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Centrifugal Distortion in Asymmetric Tops"],
            "author": [{"family": "Watson", "given": "J. K. G."}],
            "journal": "J. Mol. Spectrosc.",
            "year": 1977,
        }
    ]

    result = export_spycfit_snapshot(
        payload=authoritative_payload,
        source_dir=source_dir,
        output_dir=processed_dir,
        active_models=authoritative_payload.active_models,
        crossref_records=crossref_records,
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="CH3OH Production Fit",
        strategy="ZIP",
    )

    assert result["status"] == "SUCCESS"
    assert result["session_id"] == authoritative_payload.session_id

    prov_path = Path(result["provenance_path"])
    params_tex_path = Path(result["parameters_table_path"])
    trans_tex_path = Path(result["transitions_table_path"])
    doc_path = Path(result["document_path"])
    bib_path = Path(result["citations_bib_path"])
    archive_path = Path(result["archive_path"])

    assert prov_path.exists()
    assert params_tex_path.exists()
    assert trans_tex_path.exists()
    assert doc_path.exists()
    assert bib_path.exists()
    assert archive_path.exists()

    loaded_prov = FitProvenancePayload.from_json_file(prov_path)
    assert loaded_prov.session_id == authoritative_payload.session_id

    assert not (prov_path.stat().st_mode & stat.S_IWUSR)
    assert not (archive_path.stat().st_mode & stat.S_IWUSR)


# =====================================================================
# 10. Workspace Directory Resolution Tests
# =====================================================================


def test_resolve_processed_workspace_dir(
    monkeypatch: pytest.MonkeyPatch, clean_workspace: Path
):
    """Verifies directory resolution with custom paths, environment variable override, and defaults."""
    custom_target = clean_workspace / "custom_processed"
    assert resolve_processed_workspace_dir(custom_target) == custom_target.resolve()

    env_dir = clean_workspace / "env_state_dir"
    monkeypatch.setenv("COCHEM_STATE_DIR", str(env_dir))
    expected_env_target = env_dir / "SpycFit_Workspace" / "Processed"
    assert resolve_processed_workspace_dir() == expected_env_target.resolve()

    monkeypatch.delenv("COCHEM_STATE_DIR", raising=False)
    resolved_default = get_spycfit_processed_dir()
    assert "SpycFit_Workspace" in str(resolved_default)
    assert "Processed" in str(resolved_default)
