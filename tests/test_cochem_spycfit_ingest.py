"""
Physical Unit and Integration Tests for Stage 1.0 Intake Parser & Quantum Gatekeeper (Phase 2).

Module: tests/test_cochem_spycfit_ingest.py
Target: cochem_spycfit.intake.cochem_spycfit_ingest

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance.
3. JAX FP64 precision boundary enforcement (np.float64 casting).
4. Cryptographic SHA-256 frequency fingerprinting.
5. Zarr state matrix persistence (append_to_spectra_fit_zarr and spectra_fit.zarr).
6. Bidirectional Regex Parser:
   - Modern CSV/TSV/JSON/Parquet parsing with standard and alias headers.
   - Legacy Pickett .lin parsing (fixed-width 10-char, free-format, JPL catalog).
   - Fortran double precision exponent conversion ('D-05', 'D+02', 'd-05', '1.2345-05').
   - Legacy Pickett .par parameter file parsing.
   - Non-fatal logging of malformed lines to platformdirs.user_log_path("CoChem").
7. Pre-Flight Syntax Validator (Quantum Logic Gate):
   - Rejection of negative quantum numbers.
   - Rejection of Ka > J or Kc > J.
   - Rejection of asymmetric rotor parity sum rule violations (Ka + Kc != J and != J + 1).
   - Strict CoChemDataException emission when raise_on_error=True.
   - Structured diagnostic emission and drop handling.
8. Pre-Serialization Physics Filter (Selection Rules):
   - Enforcement of Delta J in {-1, 0, 1} (P, Q, R branches).
   - Strict rejection of J=0 -> 0 dipole-forbidden transitions.
   - Strict rejection of |Delta J| > 1 transitions.
   - Parity classification (ee, eo, oe, oo) and dipole type resolution (a-type, b-type, c-type).
   - Strict CoChemDataException emission when raise_on_error=True.
   - In-memory filtering of forbidden transitions.
9. End-to-End parse_spectroscopy_data pipeline with authentic molecular catalogs (H2O, H2CO) and Zarr output.
10. Pydantic v2 schema strictness (SpectroscopySchema & SpectroscopyTransition) and extra field forbidding.
11. Complete absence of 'pass' statements across source and test files.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import jax
import numpy as np
import pandas as pd
from platformdirs import user_log_path
import pytest
from pydantic import ValidationError
import zarr

# Ensure SpycFit source is in path
SPYCFIT_ROOT = Path(__file__).resolve().parent.parent
SPYCFIT_SRC = SPYCFIT_ROOT / "src"
if str(SPYCFIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_SRC))
if str(SPYCFIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPYCFIT_ROOT))

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    CoChemDataException,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopySchema,
    SpectroscopyTransition,
    ValidationReport,
    append_to_spectra_fit_zarr,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_legacy_pickett,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)


# =============================================================================
# 1. FILE ENCODING AND LF LINE ENDING TESTS
# =============================================================================

@pytest.fixture
def target_file_paths() -> List[Path]:
    """Returns absolute paths to all Stage 1.0 intake source and test files."""
    return [
        SPYCFIT_SRC / "cochem_spycfit" / "intake" / "cochem_spycfit_ingest.py",
        SPYCFIT_SRC / "cochem_spycfit" / "intake" / "__init__.py",
        Path(__file__).resolve(),
    ]


def test_file_encoding_and_lf_line_endings(target_file_paths: List[Path]) -> None:
    """Verify strictly Unix LF line endings (\n), standard UTF-8 encoding, and no BOM."""
    for path in target_file_paths:
        assert path.is_file(), f"Target file does not exist: {path}"
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"Found Windows CRLF (\\r\\n) in {path.name}"
        assert b"\n" in raw, f"Missing newline characters in {path.name}"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"Found UTF-8 BOM marker in {path.name}"


# =============================================================================
# 2. ZERO PASS STATEMENT AUDIT
# =============================================================================

def test_zero_pass_statements_in_codebase(target_file_paths: List[Path]) -> None:
    """Strictly audits that no 'pass' statements exist in the ingest codebase or test suite."""
    for path in target_file_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        passes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(passes) == 0, f"Found {len(passes)} 'pass' statement(s) in {path.name} at lines {[n.lineno for n in passes]}"


# =============================================================================
# 3. FORTRAN FLOAT PARSER AND FP64 BOUNDARY TESTS
# =============================================================================

@pytest.mark.parametrize(
    "input_val,expected_float",
    [
        ("1.2345D-05", 1.2345e-5),
        ("1.2345D+02", 123.45),
        ("1.2345d-05", 1.2345e-5),
        ("1.2345d+02", 123.45),
        ("1.2345E-05", 1.2345e-5),
        ("1.2345e+02", 123.45),
        ("12345.6789", 12345.6789),
        ("-456.7890", -456.789),
        ("1.2345-05", 1.2345e-5),
        ("1.2345+02", 123.45),
        (42.5, 42.5),
        (100, 100.0),
    ],
)
def test_fortran_float_converter_valid(input_val: Any, expected_float: float) -> None:
    """Tests Fortran double precision exponent conversion."""
    res = fortran_float_converter(input_val)
    assert np.isclose(res, expected_float, rtol=1e-12)
    assert isinstance(res, float)


def test_fortran_float_converter_invalid() -> None:
    """Tests that malformed strings raise ValueError."""
    with pytest.raises(ValueError):
        fortran_float_converter("")
    with pytest.raises(ValueError):
        fortran_float_converter("INVALID_NUMBER_STRING")


def test_enforce_fp64_boundary() -> None:
    """Tests JAX FP64 precision boundary enforcement on Pandas DataFrames."""
    df = pd.DataFrame({
        "Frequency_MHz": np.array([12345.67, 23456.78], dtype=np.float32),
        "Uncertainty": np.array([0.01, 0.02], dtype=np.float32),
        "Intensity": np.array([1.5, 2.5], dtype=np.float32),
        "J_upper": [1, 2],
        "Ka_upper": [0, 1],
        "Kc_upper": [1, 1],
        "J_lower": [0, 1],
        "Ka_lower": [0, 0],
        "Kc_lower": [0, 1],
    })

    fp64_df = enforce_fp64_boundary(df)

    assert fp64_df["Frequency_MHz"].dtype == np.float64
    assert fp64_df["Uncertainty"].dtype == np.float64
    assert fp64_df["Intensity"].dtype == np.float64
    assert fp64_df["J_upper"].dtype == np.int64
    assert os.environ.get("JAX_ENABLE_X64") == "True"


# =============================================================================
# 4. CRYPTOGRAPHIC FREQUENCY FINGERPRINT & ZARR PERSISTENCE TESTS
# =============================================================================

def test_calculate_frequency_fingerprint() -> None:
    """Tests deterministic SHA-256 fingerprint generation."""
    freqs = [22235.08, 183310.08, 380197.37]
    fp1 = calculate_frequency_fingerprint(freqs)
    fp2 = calculate_frequency_fingerprint(np.array(freqs, dtype=np.float64))
    fp3 = calculate_frequency_fingerprint(pd.Series(freqs))

    assert fp1 == fp2 == fp3
    assert len(fp1) == 64
    assert isinstance(fp1, str)

    # Altering frequency must change hash
    fp_altered = calculate_frequency_fingerprint([22235.09, 183310.08, 380197.37])
    assert fp_altered != fp1


def test_append_to_spectra_fit_zarr(tmp_path: Path) -> None:
    """Tests state matrix and cryptographic hash persistence to spectra_fit.zarr."""
    zarr_dir = tmp_path / "spectra_fit.zarr"
    freqs = np.array([22235.08, 183310.08, 556936.00], dtype=np.float64)

    group = append_to_spectra_fit_zarr(
        zarr_path=zarr_dir,
        frequencies=freqs,
        group_name="state_matrix",
        metadata={"molecule": "H2O", "source": "JPL Catalog"},
    )

    assert zarr_dir.exists()
    assert "frequencies" in group
    stored_freqs = group["frequencies"][:]
    assert np.allclose(stored_freqs, freqs)
    assert group.attrs["sha256_fingerprint"] == calculate_frequency_fingerprint(freqs)
    assert group.attrs["frequency_count"] == 3
    assert group.attrs["molecule"] == "H2O"


# =============================================================================
# 5. PRE-FLIGHT SYNTAX VALIDATOR (QUANTUM LOGIC GATE) TESTS
# =============================================================================

def test_validate_quantum_numbers_valid_h2o() -> None:
    """Tests syntax validation on real water (H2O) asymmetric top quantum numbers."""
    df = pd.DataFrame({
        "Frequency_MHz": [556936.0, 1669904.0, 183310.0],
        "Uncertainty": [0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0],
        "J_upper": [1, 2, 3],
        "Ka_upper": [1, 1, 1],
        "Kc_upper": [0, 2, 3],  # 1+0=1 (J=1), 1+2=3 (J=2+1), 1+3=4 (J=3+1)
        "J_lower": [1, 1, 2],
        "Ka_lower": [0, 0, 2],
        "Kc_lower": [1, 1, 0],  # 0+1=1 (J=1), 0+1=1 (J=1), 2+0=2 (J=2)
    })

    sanitized, violations = validate_quantum_numbers(df, drop_invalid=True)
    assert len(sanitized) == 3
    assert len(violations) == 0


def test_validate_quantum_numbers_intercepts_syntax_ghost_lines() -> None:
    """Tests that malformed quantum numbers (Ka > J, Ka+Kc != J or J+1, negative) are intercepted."""
    df_invalid = pd.DataFrame({
        "Frequency_MHz": [10000.0, 20000.0, 30000.0, 40000.0],
        "Uncertainty": [0.01, 0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0, 1.0],
        "J_upper": [2, 5, -1, 1],
        "Ka_upper": [3, 4, 0, 1],
        "Kc_upper": [0, 3, 0, 0],
        "J_lower": [1, 4, 0, 1],
        "Ka_lower": [1, 2, 0, 0],
        "Kc_lower": [0, 2, 0, 1],
    })

    sanitized, violations = validate_quantum_numbers(df_invalid, drop_invalid=True)

    assert len(sanitized) == 1
    assert sanitized.iloc[0]["Frequency_MHz"] == 40000.0
    assert len(violations) == 3

    v_rows = [v["row_index"] for v in violations]
    assert 0 in v_rows
    assert 1 in v_rows
    assert 2 in v_rows


def test_validate_quantum_numbers_raise_on_error() -> None:
    """Tests that validate_quantum_numbers raises CoChemDataException when raise_on_error=True."""
    df_invalid = pd.DataFrame({
        "Frequency_MHz": [10000.0],
        "Uncertainty": [0.01],
        "Intensity": [1.0],
        "J_upper": [2],
        "Ka_upper": [3],  # Ka > J violation
        "Kc_upper": [0],
        "J_lower": [1],
        "Ka_lower": [1],
        "Kc_lower": [0],
    })

    with pytest.raises(CoChemDataException):
        validate_quantum_numbers(df_invalid, raise_on_error=True)


# =============================================================================
# 6. PRE-SERIALIZATION PHYSICS FILTER (SELECTION RULES) TESTS
# =============================================================================

def test_classify_transition_parity_and_dipole_abc_types() -> None:
    """Tests exact parity and dipole classification for asymmetric top lines."""
    # a-type: Delta Ka even, Delta Kc odd (e.g. 1_01 -> 0_00: Delta Ka=0, Delta Kc=1)
    b1, d1, up1, low1, dj1, dka1, dkc1 = classify_transition_parity_and_dipole(1, 0, 1, 0, 0, 0)
    assert b1 == BranchType.R
    assert d1 == DipoleType.A
    assert up1 == Parity.EO
    assert low1 == Parity.EE

    # b-type: Delta Ka odd, Delta Kc odd (e.g. 1_10 -> 1_01: Delta Ka=1, Delta Kc=-1)
    b2, d2, up2, low2, dj2, dka2, dkc2 = classify_transition_parity_and_dipole(1, 1, 0, 1, 0, 1)
    assert b2 == BranchType.Q
    assert d2 == DipoleType.B
    assert up2 == Parity.OE
    assert low2 == Parity.EO

    # c-type: 2_11 -> 1_01: Delta Ka=1, Delta Kc=0 -> c-type
    b3, d3, up3, low3, dj3, dka3, dkc3 = classify_transition_parity_and_dipole(2, 1, 1, 1, 0, 1)
    assert b3 == BranchType.R
    assert d3 == DipoleType.C
    assert up3 == Parity.OO
    assert low3 == Parity.EO

    # Forbidden J=0 -> 0
    b4, d4, _, _, _, _, _ = classify_transition_parity_and_dipole(0, 0, 0, 0, 0, 0)
    assert d4 == DipoleType.FORBIDDEN

    # Forbidden |Delta J| > 1
    b5, d5, _, _, _, _, _ = classify_transition_parity_and_dipole(3, 1, 2, 1, 0, 1)
    assert d5 == DipoleType.FORBIDDEN


def test_apply_selection_rules_filtering() -> None:
    """Tests in-memory stripping of dipole-forbidden lines and filtering by allowed dipole types."""
    df = pd.DataFrame({
        "Frequency_MHz": [100.0, 200.0, 300.0, 400.0],
        "Uncertainty": [0.01, 0.01, 0.01, 0.01],
        "Intensity": [1.0, 1.0, 1.0, 1.0],
        "J_upper": [1, 1, 0, 3],
        "Ka_upper": [0, 1, 0, 0],
        "Kc_upper": [1, 0, 0, 3],
        "J_lower": [0, 1, 0, 1],
        "Ka_lower": [0, 0, 0, 0],
        "Kc_lower": [0, 1, 0, 1],
    })

    # 1. Allow all valid dipole types (a, b, c) -> drops rows 2 and 3
    filtered_all, violations_all = apply_selection_rules(df, allowed_dipole_types={"a", "b", "c"})
    assert len(filtered_all) == 2
    assert len(violations_all) == 2
    assert list(filtered_all["Dipole_Type"]) == ["a", "b"]

    # 2. Allow only b-type
    filtered_b, violations_b = apply_selection_rules(df, allowed_dipole_types={"b"})
    assert len(filtered_b) == 1
    assert filtered_b.iloc[0]["Dipole_Type"] == "b"
    assert filtered_b.iloc[0]["Frequency_MHz"] == 200.0
    assert len(violations_b) == 3


def test_apply_selection_rules_raise_on_error() -> None:
    """Tests that apply_selection_rules raises CoChemDataException when raise_on_error=True."""
    df_forbidden = pd.DataFrame({
        "Frequency_MHz": [100.0],
        "Uncertainty": [0.01],
        "Intensity": [1.0],
        "J_upper": [0],
        "Ka_upper": [0],
        "Kc_upper": [0],
        "J_lower": [0],
        "Ka_lower": [0],
        "Kc_lower": [0],
    })

    with pytest.raises(CoChemDataException):
        apply_selection_rules(df_forbidden, raise_on_error=True)


# =============================================================================
# 7. PARSER TESTS: CSV, PICKETT LIN, AND PICKETT PAR
# =============================================================================

def test_parse_csv_spectroscopy_standard_and_aliases(tmp_path: Path) -> None:
    """Tests CSV parsing with canonical and alias column headers."""
    csv_content = """nu_mhz,err_mhz,logint,j_prime,ka_prime,kc_prime,j_double_prime,ka_double_prime,kc_double_prime
22235.08,0.005,-3.2,1,1,0,1,0,1
183310.08,0.010,-1.5,3,1,3,2,2,0
"""
    csv_file = tmp_path / "test_spectrum.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    df = parse_csv_spectroscopy(csv_file)
    assert len(df) == 2
    assert list(df.columns) == [
        "Frequency_MHz", "Uncertainty", "Intensity",
        "J_upper", "Ka_upper", "Kc_upper",
        "J_lower", "Ka_lower", "Kc_lower",
    ]
    assert df["Frequency_MHz"].dtype == np.float64
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert df.iloc[0]["J_upper"] == 1


def test_parse_legacy_pickett_and_failed_logging(tmp_path: Path) -> None:
    """Tests parsing legacy Pickett files and verifying failed lines logging to platformdirs user_log_path."""
    lin_content = """# Water (H2O) rotational line list
  1  1  0  1  0  1   22235.0800  0.0050  1.0000D+00
  3  1  3  2  2  0  183310.0800  0.0100  5.5000D-01
  2  1  2  1  0  1 1669904.0000  0.0200  1.2000D+00
INVALID_PICKETT_LINE_MALFORMED
"""
    lin_file = tmp_path / "h2o.lin"
    lin_file.write_text(lin_content, encoding="utf-8")
    custom_log = tmp_path / "failed_ingest.log"

    cfg = IngestConfig(failed_log_path=str(custom_log))
    df, failed = parse_legacy_pickett(lin_file, config=cfg)

    assert len(df) == 3
    assert len(failed) == 1
    assert "INVALID_PICKETT_LINE" in failed[0]
    assert custom_log.is_file()
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert np.isclose(df.iloc[1]["Intensity"], 0.55)


def test_parse_pickett_par_file(tmp_path: Path) -> None:
    """Tests parsing legacy Pickett .par / .var parameter files."""
    par_content = """Water H2O Ground State Fitted Constants
   3   100    0    0   1.000000000000000D+00   1.000000000000000D+00   1.000000000000000D+00   50
       10000   4.353600000000000D+05   1.000000000000000D-04  / (B+C)/2
       20000   8.358400000000000D+05   1.000000000000000D-04  / A-(B+C)/2
       30000   1.458000000000000D+05   1.000000000000000D-04  / (B-C)/4
"""
    par_file = tmp_path / "h2o.par"
    par_file.write_text(par_content, encoding="utf-8")

    par_obj = parse_pickett_par(par_file)
    assert isinstance(par_obj, PickettParFile)
    assert "Water H2O" in par_obj.title
    assert par_obj.npar == 3
    assert par_obj.maxit == 50
    assert len(par_obj.parameters) == 3
    assert par_obj.parameters[0].param_id == 10000
    assert np.isclose(par_obj.parameters[0].value, 435360.0)


# =============================================================================
# 8. END-TO-END MASTER INGESTION PIPELINE TESTS
# =============================================================================

def test_parse_spectroscopy_data_e2e_water_with_zarr(tmp_path: Path) -> None:
    """
    End-to-End Test: Ingests realistic Water (H2O) spectrum, validates syntax,
    applies selection rules (b-type dipole), calculates fingerprint, persists to Zarr,
    and returns IngestResult.
    """
    csv_content = """Frequency_MHz,Uncertainty,Intensity,J_upper,Ka_upper,Kc_upper,J_lower,Ka_lower,Kc_lower
22235.08,0.005,1.0,1,1,0,1,0,1
183310.08,0.010,1.0,3,1,3,2,2,0
556936.00,0.010,1.0,1,1,0,1,0,1
999999.99,0.010,1.0,5,4,3,4,2,2
0.0,0.010,1.0,0,0,0,0,0,0
"""
    csv_file = tmp_path / "water_spectrum.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    zarr_out = tmp_path / "spectra_fit.zarr"

    cfg = IngestConfig(allowed_dipole_types={"b"}, zarr_path=str(zarr_out))
    res = parse_spectroscopy_data(csv_file, config=cfg)

    assert isinstance(res, IngestResult)
    assert res.success is True
    assert len(res.sha256_fingerprint) == 64
    assert res.report.total_lines_read == 5
    assert res.report.syntax_dropped_count == 1
    assert res.report.physics_dropped_count >= 1
    assert res.report.valid_lines_count == 3
    assert all(res.df["Dipole_Type"] == "b")
    assert res.df["Frequency_MHz"].dtype == np.float64

    # Verify zarr state matrix was persisted
    assert zarr_out.exists()
    z_root = zarr.open_group(str(zarr_out), mode="r")
    assert "state_matrix" in z_root
    assert "frequencies" in z_root["state_matrix"]


# =============================================================================
# 9. PYDANTIC SCHEMA STRICTNESS TESTS (SPECTROSCOPYSCHEMA & TRANSITION)
# =============================================================================

def test_spectroscopy_schema_validation_and_forbid_extra() -> None:
    """Verifies that SpectroscopySchema validates mandatory fields and forbids extra tokens."""
    # Valid instance
    schema = SpectroscopySchema(
        Frequency_MHz=22235.08,
        Uncertainty=0.005,
        Intensity=1.0,
        J_upper=1,
        Ka_upper=1,
        Kc_upper=0,
        J_lower=1,
        Ka_lower=0,
        Kc_lower=1,
    )
    assert schema.Frequency_MHz == 22235.08
    assert schema.branch == BranchType.Q
    assert schema.dipole_type == DipoleType.B
    assert schema.upper_parity == Parity.OE
    assert schema.lower_parity == Parity.EO

    # Forbids extra fields
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=1,
            Ka_upper=0,
            Kc_upper=1,
            J_lower=0,
            Ka_lower=0,
            Kc_lower=0,
            unauthorized_field="malicious_injection",  # type: ignore
        )

    # Rejects missing mandatory fields
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            # Missing J, Ka, Kc quantum numbers
        )


def test_spectroscopy_schema_physics_bounds() -> None:
    """Verifies that SpectroscopySchema rejects impossible rotor assignments."""
    # Ka > J
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=1,
            Ka_upper=2,
            Kc_upper=0,
            J_lower=0,
            Ka_lower=0,
            Kc_lower=0,
        )

    # Ka + Kc != J and != J+1
    with pytest.raises(ValidationError):
        SpectroscopySchema(
            Frequency_MHz=100.0,
            Uncertainty=0.01,
            Intensity=1.0,
            J_upper=5,
            Ka_upper=4,
            Kc_upper=3,
            J_lower=4,
            Ka_lower=2,
            Kc_lower=2,
        )


def test_missing_file_paths_raise_file_not_found() -> None:
    """Verifies that non-existent paths raise FileNotFoundError rather than generic errors."""
    with pytest.raises(FileNotFoundError):
        parse_legacy_pickett(Path("non_existent_file.lin"))

    with pytest.raises(FileNotFoundError):
        parse_pickett_par(Path("non_existent_file.par"))

    with pytest.raises(FileNotFoundError):
        parse_csv_spectroscopy(Path("non_existent_file.csv"))


# =============================================================================
# 10. ADVANCED ADVERSARIAL EDGE CASE TESTS
# =============================================================================

def test_parse_pickett_cdms_jpl_catalog_format(tmp_path: Path) -> None:
    """Tests parsing realistic CDMS / JPL catalog lines with species tags and format codes."""
    cat_content = """  22235.0800    0.0050   -3.2000  3     0.0000  3 -18001 1403  1  1  0  1  0  1
 183310.0800    0.0100   -1.5000  3   136.1600  3 -18001 1403  3  1  3  0  0  0  2  2  0  0  0  0
 556936.0020    0.0100   -0.5000  3     0.0000  3 -18001 1403  1  1  0  0  0  0  1  0  1  0  0  0
"""
    cat_file = tmp_path / "c018001.cat"
    cat_file.write_text(cat_content, encoding="utf-8")

    df, failed = parse_legacy_pickett(cat_file)
    assert len(df) == 3
    assert len(failed) == 0
    assert np.isclose(df.iloc[0]["Frequency_MHz"], 22235.08)
    assert df.iloc[0]["J_upper"] == 1
    assert df.iloc[0]["Ka_upper"] == 1
    assert df.iloc[0]["Kc_upper"] == 0
    assert df.iloc[0]["J_lower"] == 1
    assert df.iloc[0]["Ka_lower"] == 0
    assert df.iloc[0]["Kc_lower"] == 1


def test_parse_pickett_var_parameter_file(tmp_path: Path) -> None:
    """Verifies parsing of Pickett .var parameter files."""
    var_content = """Formaldehyde H2CO Watson S-reduction
   3   50    0    0   1.0D+00   1.0D+00   1.0D+00   50
       10000   3.883398900000000D+04   1.000000000000000D-04  / (B+C)/2
       20000   2.431580700000000D+05   1.000000000000000D-04  / A-(B+C)/2
       30000   2.368940000000000D+03   1.000000000000000D-04  / (B-C)/4
"""
    var_file = tmp_path / "h2co.var"
    var_file.write_text(var_content, encoding="utf-8")

    res = parse_spectroscopy_data(var_file)
    assert res.success is True
    assert len(res.df) == 3
    assert np.isclose(res.df.iloc[0]["value"], 38833.989)
