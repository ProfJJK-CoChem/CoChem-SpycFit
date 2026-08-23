"""
Physical Unit and Integration Tests for Stage 1.0 Physical & Isotopic Resonance Filter.

Module: tests/test_cochem_spycfit_triage.py
Target: cochem_spycfit.intake.cochem_spycfit_triage

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance via zero-trust static inspection.
3. Mendeleev SQLite database local resolution and air-gap verification.
4. Natural abundance filtering, isotopic properties, and isotopologue triage.
5. Method Matrix v4 Mass Matrix Transformation (M^-1/2 F_cart M^-1/2) in FP64 precision.
6. Harmonic vibrational frequency computation, unit conversion, and normal mode analysis.
7. Resonance & Polyad Trap:
   - Fermi Type I resonance scan (omega_i ~ 2 * omega_j)
   - Fermi Type II resonance scan (omega_i ~ omega_j + omega_k)
   - Coriolis resonance scan (omega_i ~ omega_j)
   - Darling-Dennison resonance scan (2 * omega_i ~ 2 * omega_j)
   - Polyad block clustering and effective Hamiltonian recommendation
8. Unified triage pipeline (triage_spectroscopic_system) with deterministic SHA-256 provenance.
9. Pydantic v2 strict schema validation and extra field forbidding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from pydantic import ValidationError
import pytest
import h5py
from pathlib import Path

from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)

# =============================================================================
# 1. FILE ENCODING AND LF LINE ENDING TESTS
# =============================================================================


@pytest.fixture
def target_triage_file_paths() -> List[Path]:
    """Returns absolute paths to all Stage 1.0 triage source and test files."""
    root = Path(__file__).resolve().parent.parent
    return [
        root / "src" / "cochem_spycfit" / "intake" / "cochem_spycfit_triage.py",
        Path(__file__).resolve(),
    ]


def test_file_encoding_and_lf_line_endings(
    target_triage_file_paths: List[Path],
) -> None:
    """Verify strictly Unix LF line endings (\\n), standard UTF-8 encoding, and no BOM."""
    for path in target_triage_file_paths:
        assert path.is_file(), f"Target file does not exist: {path}"
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"Found Windows CRLF (\\r\\n) in {path.name}"
        assert b"\n" in raw, f"Missing newline characters in {path.name}"
        assert not raw.startswith(b"\xef\xbb\xbf"), (
            f"Found UTF-8 BOM marker in {path.name}"
        )


# =============================================================================
# 2. CODE INTEGRITY AND STATIC AUDIT
# =============================================================================


def test_anti_spoofing_and_integrity_compliance(
    target_triage_file_paths: List[Path],
) -> None:
    """Performs static inspection ensuring target files comply with anti-spoofing standards."""
    banned_keywords = ["mock", "dummy", "placeholder", "fake", "stub"]
    for path in target_triage_file_paths:
        content = path.read_text(encoding="utf-8").lower()
        for kw in banned_keywords:
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#") or "banned_keywords" in clean_line:
                    continue
                assert f"def {kw}" not in clean_line, (
                    f"Line {line_idx} in {path.name} contains banned function def: {kw}"
                )
                assert f"class {kw}" not in clean_line, (
                    f"Line {line_idx} in {path.name} contains banned class def: {kw}"
                )
                assert f"import unittest.{kw}" not in clean_line, (
                    f"Line {line_idx} in {path.name} imports {kw}"
                )


# =============================================================================
# 3. MENDELEEV DATABASE RESOLUTION AND AIR-GAP VERIFICATION
# =============================================================================


def test_resolve_mendeleev_database_default() -> None:
    """Tests that local Mendeleev SQLite database is resolved and verified without network access."""
    db_path = resolve_mendeleev_database()
    assert db_path.exists(), f"Database file does not exist at {db_path}"
    assert db_path.is_file(), f"Database path is not a file: {db_path}"
    assert db_path.stat().st_size > 1000000, "Database file is unexpectedly small"

    with open(db_path, "rb") as f:
        header = f.read(16)
    assert header.startswith(b"SQLite format 3\x00"), (
        "Database does not have valid SQLite header"
    )


def test_resolve_mendeleev_database_invalid_path() -> None:
    """Tests that non-existent database paths raise FileNotFoundError."""
    invalid_path = Path("/nonexistent/path/to/elements.db")
    with pytest.raises(FileNotFoundError):
        resolve_mendeleev_database(invalid_path)


def test_get_isotope_properties_carbon() -> None:
    """Tests querying isotopic properties for Carbon-12 and Carbon-13."""
    c12 = get_isotope_properties("C", 12)
    assert isinstance(c12, IsotopeSpec)
    assert c12.symbol == "C"
    assert c12.mass_number == 12
    assert c12.atomic_number == 6
    assert np.isclose(c12.mass_amu, 12.0, atol=1e-6)
    assert c12.abundance_percent > 98.0

    c13 = get_isotope_properties("C", 13)
    assert c13.symbol == "C"
    assert c13.mass_number == 13
    assert c13.atomic_number == 6
    assert np.isclose(c13.mass_amu, 13.003355, atol=1e-5)
    assert np.isclose(c13.abundance_percent, 1.06, atol=0.2)
    assert c13.nuclear_spin == "1/2"


def test_get_isotope_properties_hydrogen_and_oxygen() -> None:
    """Tests querying isotopic properties for Hydrogen and Oxygen isotopes."""
    h1 = get_isotope_properties("H", 1)
    assert h1.symbol == "H"
    assert h1.mass_number == 1
    assert np.isclose(h1.mass_amu, 1.007825, atol=1e-5)
    assert h1.abundance_percent > 99.9

    h2 = get_isotope_properties("H", 2)
    assert h2.mass_number == 2
    assert np.isclose(h2.mass_amu, 2.014102, atol=1e-5)
    assert h2.abundance_percent < 0.03

    o16 = get_isotope_properties("O", 16)
    assert o16.mass_number == 16
    assert np.isclose(o16.mass_amu, 15.994915, atol=1e-5)
    assert o16.abundance_percent > 99.7

    o18 = get_isotope_properties("O", 18)
    assert o18.mass_number == 18
    assert np.isclose(o18.mass_amu, 17.999160, atol=1e-5)
    assert np.isclose(o18.abundance_percent, 0.205, atol=0.05)


def test_get_isotope_properties_default_most_abundant() -> None:
    """Tests querying element without mass number returns the most abundant isotope."""
    c_default = get_isotope_properties("C")
    assert c_default.mass_number == 12
    assert c_default.abundance_percent > 98.0

    n_default = get_isotope_properties("N")
    assert n_default.mass_number == 14
    assert n_default.abundance_percent > 99.0


# =============================================================================
# 4. NATURAL ABUNDANCE FILTERING AND ISOTOPIC TRIAGE
# =============================================================================


def test_evaluate_isotopologue_abundance_parent_water() -> None:
    """Tests evaluation of parent water molecule (1H2 16O)."""
    h1 = get_isotope_properties("H", 1)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h1, h1, o16], threshold_percent=0.01)

    assert isinstance(result, IsotopeTriageResult)
    assert result.is_observable is True
    assert result.product_class == ProductClass.PRODUCT_A
    assert result.substitution_count == 0
    assert result.abundance_percent > 99.0
    assert np.isclose(result.total_mass_amu, 18.010565, atol=1e-4)


def test_evaluate_isotopologue_abundance_singly_substituted_water() -> None:
    """Tests evaluation of singly-substituted HDO (1H 2H 16O)."""
    h1 = get_isotope_properties("H", 1)
    h2 = get_isotope_properties("H", 2)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h1, h2, o16], threshold_percent=0.001)

    assert result.is_observable is True
    assert result.product_class in [ProductClass.PRODUCT_B, ProductClass.PRODUCT_C]
    assert result.substitution_count == 1
    assert result.abundance_percent > 0.001
    assert np.isclose(result.total_mass_amu, 19.016842, atol=1e-4)


def test_evaluate_isotopologue_abundance_rare_trace_d2o() -> None:
    """Tests evaluation of doubly-deuterated D2O (2H2 16O) falling below standard threshold."""
    h2 = get_isotope_properties("H", 2)
    o16 = get_isotope_properties("O", 16)
    result = evaluate_isotopologue_abundance([h2, h2, o16], threshold_percent=0.01)

    assert result.is_observable is False
    assert result.product_class == ProductClass.TRACE
    assert result.substitution_count == 2
    assert result.abundance_percent < 0.0001
    assert result.rejection_reason is not None


def test_check_isotope_abundance_string_input() -> None:
    """Tests check_isotope_abundance helper with tuple and string representations."""
    res_co2 = check_isotope_abundance([("C", 12), ("O", 16), ("O", 16)])
    assert res_co2.product_class == ProductClass.PRODUCT_A
    assert res_co2.is_observable is True

    res_13c_co2 = check_isotope_abundance([("C", 13), ("O", 16), ("O", 16)])
    assert res_13c_co2.product_class in [ProductClass.PRODUCT_B, ProductClass.PRODUCT_C]
    assert res_13c_co2.is_observable is True
    assert res_13c_co2.substitution_count == 1


def test_parse_isotopologue_input_formats() -> None:
    """Tests parse_isotopologue_input on string formulas, hyphenated tokens, and IsotopologueData models."""
    # 1. Hyphenated string token
    specs_hyphen = parse_isotopologue_input("13C-1H-35Cl")
    assert len(specs_hyphen) == 3
    assert specs_hyphen[0].symbol == "C" and specs_hyphen[0].mass_number == 13
    assert specs_hyphen[1].symbol == "H" and specs_hyphen[1].mass_number == 1
    assert specs_hyphen[2].symbol == "Cl" and specs_hyphen[2].mass_number == 35

    # 2. Parenthesized and spaced isotope formulas
    specs_paren = parse_isotopologue_input("(12C)(16O)2")
    assert len(specs_paren) == 3
    assert specs_paren[0].symbol == "C" and specs_paren[0].mass_number == 12
    assert specs_paren[1].symbol == "O" and specs_paren[1].mass_number == 16
    assert specs_paren[2].symbol == "O" and specs_paren[2].mass_number == 16

    specs_spaced = parse_isotopologue_input("12C 16O2")
    assert len(specs_spaced) == 3
    assert specs_spaced[0].symbol == "C" and specs_spaced[0].mass_number == 12
    assert specs_spaced[1].symbol == "O" and specs_spaced[1].mass_number == 16
    assert specs_spaced[2].symbol == "O" and specs_spaced[2].mass_number == 16

    specs_std = parse_isotopologue_input("H2O")
    assert len(specs_std) == 3
    assert specs_std[0].symbol == "H" and specs_std[0].mass_number == 1
    assert specs_std[1].symbol == "H" and specs_std[1].mass_number == 1
    assert specs_std[2].symbol == "O" and specs_std[2].mass_number == 16

    # 3. IsotopologueData Pydantic model
    iso_data = IsotopologueData(
        formula="H2O",
        name="Water",
        isotopes=[
            get_isotope_properties("H", 1),
            get_isotope_properties("H", 1),
            get_isotope_properties("O", 16),
        ],
    )
    specs_model = parse_isotopologue_input(iso_data)
    assert len(specs_model) == 3
    assert specs_model[0].symbol == "H"


def test_calculate_triage_provenance_hash_deterministic() -> None:
    """Tests that calculate_triage_provenance_hash generates reproducible SHA-256 digests."""
    payload1 = {"system": "H2O", "modes": [1595.0, 3657.0, 3756.0]}
    payload2 = {
        "modes": [1595.0, 3657.0, 3756.0],
        "system": "H2O",
    }  # Key order difference
    hash1 = calculate_triage_provenance_hash(payload1)
    hash2 = calculate_triage_provenance_hash(payload2)
    assert len(hash1) == 64
    assert hash1 == hash2


# =============================================================================
# 5. METHOD MATRIX V4 MASS MATRIX TRANSFORMATION (FP64 PRECISION)
# =============================================================================


def test_mass_matrix_transformation_1d_diatomic() -> None:
    """
    Tests mass matrix transformation on a 1D diatomic harmonic oscillator:
    F_cart = [[k, -k], [-k, k]]
    masses = [m1, m2]
    H_mw = [[k/m1, -k/sqrt(m1*m2)], [-k/sqrt(m1*m2), k/m2]]
    """
    k = 500.0  # N/m
    m1 = 12.0  # amu
    m2 = 16.0  # amu

    f_cart = np.array([[k, -k], [-k, k]], dtype=np.float64)
    masses = [m1, m2]

    h_mw = mass_matrix_transformation(f_cart, masses)

    assert h_mw.shape == (2, 2)
    assert h_mw.dtype == np.float64
    assert np.isclose(h_mw[0, 0], k / m1, rtol=1e-14)
    assert np.isclose(h_mw[1, 1], k / m2, rtol=1e-14)
    assert np.isclose(h_mw[0, 1], -k / math.sqrt(m1 * m2), rtol=1e-14)
    assert np.isclose(h_mw[1, 0], -k / math.sqrt(m1 * m2), rtol=1e-14)


def test_mass_matrix_transformation_3n_expansion() -> None:
    """Tests that atom masses (length N) are correctly broadcasted to 3N Cartesian coordinates."""
    n_atoms = 2
    masses = [1.008, 15.999]  # OH radical
    dim = 3 * n_atoms

    with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
        f_cart = f_h5['H_H2_TS'][:]

    h_mw = mass_matrix_transformation(f_cart, masses)

    assert h_mw.shape == (dim, dim)
    assert h_mw.dtype == np.float64

    coord_masses = np.repeat(masses, 3)
    for i in range(dim):
        for j in range(dim):
            expected = f_cart[i, j] / math.sqrt(coord_masses[i] * coord_masses[j])
            assert np.isclose(h_mw[i, j], expected, rtol=1e-14)


def test_mass_matrix_transformation_dimension_mismatch() -> None:
    """Tests that mismatched dimensions raise ValueError."""
    f_cart = np.eye(6, dtype=np.float64)
    with pytest.raises(ValueError):
        mass_matrix_transformation(f_cart, [12.0, 16.0, 1.0, 1.0])


# =============================================================================
# 6. HARMONIC VIBRATIONAL FREQUENCY COMPUTATION
# =============================================================================


def test_compute_harmonic_vibrational_frequencies_diatomic_co() -> None:
    """
    Tests harmonic frequency calculation for Carbon Monoxide (12C 16O).
    Force constant k = 1.20 Hartree / bohr^2 (realistic quantum chemistry magnitude).
    Analytical frequency: omega = sqrt(k / mu) * AU_TO_WAVENUMBER_CONVERSION
    """
    m_c = 12.0
    m_o = 15.994915
    mu = (m_c * m_o) / (m_c + m_o)

    k = 1.20  # Hartree / bohr^2
    with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
        f_cart = f_h5['f_cart_zero'][:]
    f_cart[2, 2] = k
    f_cart[5, 5] = k
    f_cart[2, 5] = -k
    f_cart[5, 2] = -k

    result = compute_harmonic_vibrational_frequencies(
        F_cart=f_cart,
        masses=[m_c, m_o],
        units="hartree_bohr2",
        zero_threshold_cm1=30.0,
    )

    assert isinstance(result, HarmonicVibrationalResult)
    assert len(result.frequencies_cm1) == 6
    assert result.imaginary_frequencies_count == 0
    assert result.zero_modes_count == 5

    expected_omega = math.sqrt(k / mu) * AU_TO_WAVENUMBER_CONVERSION
    assert len(result.vibrational_frequencies_cm1) == 1
    assert np.isclose(result.vibrational_frequencies_cm1[0], expected_omega, rtol=1e-5)


def test_compute_harmonic_vibrational_frequencies_imaginary_mode() -> None:
    """Tests that saddle-point negative Hessian eigenvalues produce negative/imaginary frequencies."""
    f_cart = np.array([[-0.5]], dtype=np.float64)
    masses = [1.0]

    result = compute_harmonic_vibrational_frequencies(
        F_cart=f_cart,
        masses=masses,
        units="hartree_bohr2",
        zero_threshold_cm1=10.0,
    )

    assert result.imaginary_frequencies_count == 1
    assert result.frequencies_cm1[0] < 0.0
    expected_imag = -math.sqrt(0.5 / 1.0) * AU_TO_WAVENUMBER_CONVERSION
    assert np.isclose(result.frequencies_cm1[0], expected_imag, rtol=1e-5)


# =============================================================================
# 7. RESONANCE & POLYAD TRAP TESTS
# =============================================================================


def test_scan_fermi_resonances_type1_co2() -> None:
    """
    Tests Fermi Type I resonance detection on Carbon Dioxide:
    nu_1 (symmetric stretch) = 1333.0 cm^-1
    nu_2 (bend) = 667.0 cm^-1
    2 * nu_2 = 1334.0 cm^-1 -> Delta E = |1333.0 - 1334.0| = 1.0 cm^-1 (Strong Fermi resonance)
    """
    freqs = [667.0, 667.0, 1333.0, 2349.0]
    resonances = scan_fermi_resonances(freqs, type1_tol=30.0, type2_tol=30.0)

    type1_resonances = [
        r for r in resonances if r.resonance_type == ResonanceType.FERMI_TYPE_I
    ]
    assert len(type1_resonances) >= 1

    matched = False
    for r in type1_resonances:
        if r.mode_indices == [2, 0] or r.mode_indices == [2, 1]:
            assert np.isclose(r.energy_difference_cm1, 1.0, atol=1e-4)
            assert r.severity == "STRONG"
            matched = True
    assert matched, "Fermi Type I resonance between nu1 and 2*nu2 not detected"


def test_scan_fermi_resonances_type2_triad() -> None:
    """
    Tests Fermi Type II resonance detection:
    omega_i ~ omega_j + omega_k
    nu_1 = 3000.0 cm^-1, nu_2 = 1400.0 cm^-1, nu_3 = 1605.0 cm^-1
    nu_2 + nu_3 = 3005.0 cm^-1 -> Delta E = |3000 - 3005| = 5.0 cm^-1 (Strong Fermi Type II)
    """
    freqs = [1400.0, 1605.0, 3000.0]
    resonances = scan_fermi_resonances(freqs, type1_tol=30.0, type2_tol=30.0)

    type2_resonances = [
        r for r in resonances if r.resonance_type == ResonanceType.FERMI_TYPE_II
    ]
    assert len(type2_resonances) == 1
    assert type2_resonances[0].mode_indices == [2, 0, 1]
    assert np.isclose(type2_resonances[0].energy_difference_cm1, 5.0, atol=1e-4)
    assert type2_resonances[0].severity == "STRONG"


def test_scan_coriolis_resonances_near_degenerate() -> None:
    """
    Tests Coriolis resonance detection on nearly degenerate modes:
    omega_0 = 1000.0 cm^-1, omega_1 = 1003.5 cm^-1 -> Delta E = 3.5 cm^-1
    """
    freqs = [1000.0, 1003.5, 2500.0]
    resonances = scan_coriolis_resonances(freqs, coriolis_tol=20.0)

    assert len(resonances) == 1
    assert resonances[0].resonance_type == ResonanceType.CORIOLIS
    assert resonances[0].mode_indices == [0, 1]
    assert np.isclose(resonances[0].energy_difference_cm1, 3.5, atol=1e-4)
    assert resonances[0].severity == "STRONG"


def test_scan_darling_dennison_resonances() -> None:
    """
    Tests Darling-Dennison resonance between overtone pairs:
    2*omega_1 ~ 2*omega_2
    omega_1 = 3000.0 cm^-1, omega_2 = 3010.0 cm^-1 -> 2*omega_1 = 6000, 2*omega_2 = 6020 -> Delta E = 20 cm^-1
    """
    freqs = [3000.0, 3010.0]
    resonances = scan_darling_dennison_resonances(freqs, dd_tol=30.0)

    assert len(resonances) == 1
    assert resonances[0].resonance_type == ResonanceType.DARLING_DENNISON
    assert resonances[0].mode_indices == [0, 1]
    assert np.isclose(resonances[0].energy_difference_cm1, 20.0, atol=1e-4)


def test_identify_polyad_blocks_clustering() -> None:
    """Tests grouping interacting resonant modes into coupled Polyad blocks."""
    freqs = [667.0, 1333.0, 2000.0, 3000.0]
    res1 = ResonanceCondition(
        resonance_type=ResonanceType.FERMI_TYPE_I,
        mode_indices=[1, 0],
        frequencies_cm1=[1333.0, 667.0],
        energy_difference_cm1=1.0,
        description="Fermi resonance between mode 1 and overtone of 0",
        severity="STRONG",
    )

    blocks = identify_polyad_blocks([res1], freqs)

    assert len(blocks) == 1
    polyad = blocks[0]
    assert isinstance(polyad, PolyadBlock)
    assert set(polyad.mode_indices) == {0, 1}
    assert polyad.dimension >= 2
    assert "Polyad" in polyad.recommended_treatment


def test_trap_resonances_full_report() -> None:
    """Tests trap_resonances high-level entry point returning structured ResonanceReport."""
    freqs = [667.0, 667.0, 1333.0, 2349.0]
    report = trap_resonances(freqs, fermi_tol=30.0, coriolis_tol=20.0)

    assert isinstance(report, ResonanceReport)
    assert report.has_strong_resonances is True
    assert report.total_resonances_count > 0
    assert len(report.fermi_resonances) > 0
    assert "Polyad" in report.recommended_hamiltonian


# =============================================================================
# 8. UNIFIED TRIAGE PIPELINE (triage_spectroscopic_system)
# =============================================================================


def test_triage_spectroscopic_system_parent_water() -> None:
    """Tests full triage pipeline on parent H2O molecule."""
    m_h = 1.007825
    m_o = 15.994915
    masses = [m_h, m_h, m_o]

    f_diag = np.diag(
        [
            m_h * (1595.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
            m_h * (3657.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
            m_o * (3756.0 / AU_TO_WAVENUMBER_CONVERSION) ** 2,
        ]
    )

    report = triage_spectroscopic_system(
        isotopologue=[("H", 1), ("H", 1), ("O", 16)],
        F_cart=f_diag,
        masses=masses,
        config=TriageConfig(abundance_threshold_percent=0.01),
    )

    assert isinstance(report, TriageReport)
    assert report.passed is True
    assert report.product_class == ProductClass.PRODUCT_A
    assert report.status in [TriageStatus.PASSED, TriageStatus.WARNING_RESONANCE]
    assert report.isotopologue_triage is not None
    assert report.isotopologue_triage.is_observable is True
    assert report.vibrational_triage is not None
    assert report.resonance_triage is not None
    assert len(report.sha256_provenance) == 64


def test_triage_spectroscopic_system_rare_rejection() -> None:
    """Tests that an ultra-rare isotopologue triggers sub-threshold triage warning/rejection."""
    report = triage_spectroscopic_system(
        isotopologue=[("H", 2), ("H", 2), ("O", 18)],
        config=TriageConfig(abundance_threshold_percent=0.01),
    )

    assert report.passed is False
    assert report.status == TriageStatus.REJECTED_RARE_ISOTOPE
    assert report.product_class == ProductClass.TRACE
    assert report.isotopologue_triage is not None
    assert report.isotopologue_triage.is_observable is False


# =============================================================================
# 9. PYDANTIC V2 SCHEMA STRICTNESS AND EXTRA FIELD FORBIDDING
# =============================================================================


def test_pydantic_schema_extra_fields_forbidden() -> None:
    """Ensures Pydantic v2 schemas reject unexpected/extra fields."""
    with pytest.raises(ValidationError):
        IsotopeSpec(
            symbol="C",
            mass_number=12,
            mass_amu=12.0,
            abundance_percent=98.9,
            unauthorized_field="illegal_payload",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        TriageConfig(
            abundance_threshold_percent=0.01,
            rogue_parameter=True,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        IsotopologueData(
            formula="H2O",
            isotopes=[get_isotope_properties("H", 1)],
            forbidden_attribute=123,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        ResonanceCondition(
            resonance_type=ResonanceType.FERMI_TYPE_I,
            mode_indices=[0, 1],
            frequencies_cm1=[1000.0, 500.0],
            energy_difference_cm1=0.0,
            description="Test",
            severity="STRONG",
            unwanted_extra=True,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        PolyadBlock(
            block_id=0,
            mode_indices=[0, 1],
            frequencies_cm1=[1000.0, 500.0],
            resonance_types=[ResonanceType.FERMI_TYPE_I],
            dimension=2,
            frequency_span_cm1=500.0,
            recommended_treatment="Test",
            bad_field="fail",  # type: ignore[call-arg]
        )


def test_parse_isotopologue_dict_input() -> None:
    """Tests parse_isotopologue_input and check_isotope_abundance with single and nested dict formats."""
    # Single isotope dict
    specs1 = parse_isotopologue_input({"symbol": "C", "mass_number": 13})
    assert len(specs1) == 1
    assert specs1[0].symbol == "C" and specs1[0].mass_number == 13

    # Element-key dict
    specs2 = parse_isotopologue_input({"element": "O", "a": 18})
    assert len(specs2) == 1
    assert specs2[0].symbol == "O" and specs2[0].mass_number == 18

    # Nested isotopes dict
    specs3 = parse_isotopologue_input(
        {
            "isotopes": [
                {"symbol": "H", "mass_number": 1},
                {"symbol": "F", "mass_number": 19},
            ]
        }
    )
    assert len(specs3) == 2
    assert specs3[0].symbol == "H" and specs3[1].symbol == "F"

    # check_isotope_abundance with single dict
    res = check_isotope_abundance({"symbol": "C", "mass_number": 12})
    assert res.is_observable is True
    assert res.product_class == ProductClass.PRODUCT_A


def test_compute_harmonic_vibrational_frequencies_reduced_mass_exact() -> None:
    """
    Tests that computed mode reduced mass mu matches the standard Gaussian/Wilson FG definition:
    mu = sum_i(m_i * q_cart_norm_i^2) = 1 / sum_i(L_ik^2 / m_i).
    For a 1D diatomic with m1=12.0 amu, m2=16.0 amu:
    Gaussian reduced mass = 1 / ( (L_1^2 / m1) + (L_2^2 / m2) ) = 13.44 amu.
    """
    m1 = 12.0
    m2 = 16.0
    k = 1.0  # arbitrary force constant

    f_cart = np.array([[k, -k], [-k, k]], dtype=np.float64)
    res = compute_harmonic_vibrational_frequencies(f_cart, [m1, m2])

    assert res.reduced_masses_amu is not None
    # Mode 0 is translation (zero mode), Mode 1 is genuine vibration
    expected_gaussian_mu = 1.0 / (
        (12.0 * 16.0 / 28.0) / (12.0**2) + (12.0 * 16.0 / 28.0) / (16.0**2)
    )
    # 13.44 amu
    assert np.isclose(expected_gaussian_mu, 13.44, rtol=1e-12)
    assert np.isclose(res.reduced_masses_amu[1], 13.44, rtol=1e-10)


def test_polyad_blocks_multi_component_partitioning() -> None:
    """Tests partitioning when multiple disjoint resonance networks exist simultaneously."""
    freqs = [500.0, 1000.0, 1500.0, 3000.0]
    # Resonance 1: mode 1 (1000) ~ 2 * mode 0 (500)
    res1 = ResonanceCondition(
        resonance_type=ResonanceType.FERMI_TYPE_I,
        mode_indices=[1, 0],
        frequencies_cm1=[1000.0, 500.0],
        energy_difference_cm1=0.0,
        description="Fermi pair 1",
        severity="STRONG",
    )
    # Resonance 2: mode 3 (3000) ~ 2 * mode 2 (1500)
    res2 = ResonanceCondition(
        resonance_type=ResonanceType.FERMI_TYPE_I,
        mode_indices=[3, 2],
        frequencies_cm1=[3000.0, 1500.0],
        energy_difference_cm1=0.0,
        description="Fermi pair 2",
        severity="STRONG",
    )

    blocks = identify_polyad_blocks([res1, res2], freqs)
    assert len(blocks) == 2
    assert set(blocks[0].mode_indices) == {0, 1}
    assert set(blocks[1].mode_indices) == {2, 3}
