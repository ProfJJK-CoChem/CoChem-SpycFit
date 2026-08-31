"""
Unit tests for CoChem Core Elements, Mendeleev Mass Resolver, and Spin-Parity Validator.

Provenance:
- Method Matrix [M]: Quantum spin-parity invariants ((N_e + M) mod 2 == 1), multiplicity bounds (M - 1 <= N_e).
- Derived Relations [D]: IUPAC CIAAW standard atomic weights and isotopic masses dynamically retrieved via mendeleev.
- Empirical [E]: Pre-flight oxidation state heuristics and valence boundaries.

Strict Zero-Mock Mandate Compliance:
- All atomic numbers, standard atomic weights, and isotopic masses are loaded dynamically from mendeleev.
- Zero synthetic doubles, zero dummy arrays, zero hardcoded mass tables.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import List

import mendeleev
import pytest

from cochem.core.cochem_elements import (
    get_available_isotopes,
    get_ciaaw_atomic_mass,
    get_ciaaw_isotope_mass,
    get_element_atomic_number,
    get_element_electronegativity,
    get_element_name,
    get_element_symbol,
    validate_charge_and_multiplicity,
)
from cochem.qc.exceptions import CoChemQCException, IsotopeNotFoundError, SpinParityViolationError
from cochem.qc.models import SpinChargeValidationResult


# =============================================================================
# 1. ELEMENT LOOKUP AND ATOMIC NUMBER RESOLUTION
# =============================================================================


class TestElementAtomicNumber:
    """Test dynamic atomic number resolution and symbol normalization."""

    def test_standard_elements(self) -> None:
        """Verify atomic numbers for standard chemical elements."""
        test_cases = [
            ("H", 1),
            ("He", 2),
            ("Li", 3),
            ("B", 5),
            ("C", 6),
            ("N", 7),
            ("O", 8),
            ("F", 9),
            ("Ne", 10),
            ("Na", 11),
            ("P", 15),
            ("S", 16),
            ("Cl", 17),
            ("Ar", 18),
            ("Fe", 26),
            ("Br", 35),
            ("I", 53),
            ("Au", 79),
            ("U", 92),
            ("Og", 118),
        ]
        for sym, expected_z in test_cases:
            assert get_element_atomic_number(sym) == expected_z
            assert get_element_atomic_number(sym) == mendeleev.element(sym).atomic_number

    def test_case_insensitivity_and_whitespace(self) -> None:
        """Verify symbols are normalized across casing and surrounding whitespace."""
        assert get_element_atomic_number("c") == 6
        assert get_element_atomic_number("  C  ") == 6
        assert get_element_atomic_number("cl") == 17
        assert get_element_atomic_number("CL") == 17
        assert get_element_atomic_number("cL") == 17
        assert get_element_atomic_number("  fe  ") == 26
        assert get_element_atomic_number("HE") == 2

    def test_invalid_element_symbols(self) -> None:
        """Verify invalid symbols raise ValueError."""
        invalid_symbols = ["Xx", "Zz", "123", "", "   ", "Carbon", "H2O", "None"]
        for inv in invalid_symbols:
            with pytest.raises((ValueError, KeyError)):
                get_element_atomic_number(inv)


# =============================================================================
# 2. DYNAMIC MENDELEEV CIAAW MASS RESOLUTION
# =============================================================================


class TestMendeleevMassResolution:
    """Test dynamic retrieval of IUPAC CIAAW atomic and isotopic masses."""

    def test_ciaaw_atomic_masses(self) -> None:
        """Verify standard atomic weights match dynamic mendeleev values."""
        symbols = ["H", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "Fe", "Br", "I"]
        for sym in symbols:
            mass = get_ciaaw_atomic_mass(sym)
            elem = mendeleev.element(sym)
            expected_mass = float(elem.mass if elem.mass is not None else elem.atomic_weight)
            assert isinstance(mass, float)
            assert mass > 0.0
            assert abs(mass - expected_mass) < 1e-7

    def test_ciaaw_isotope_masses_valid(self) -> None:
        """Verify isotopic masses for genuine isotopes match CIAAW records."""
        valid_isotopes = [
            ("H", 1, 1.007825),
            ("H", 2, 2.014101),
            ("C", 12, 12.000000),
            ("C", 13, 13.003354),
            ("N", 14, 14.003074),
            ("N", 15, 15.000108),
            ("O", 16, 15.994914),
            ("O", 18, 17.999160),
            ("Cl", 35, 34.968852),
            ("Cl", 37, 36.965902),
        ]
        for sym, mass_num, approx_val in valid_isotopes:
            iso_mass = get_ciaaw_isotope_mass(sym, mass_num)
            assert isinstance(iso_mass, float)
            assert abs(iso_mass - approx_val) < 1e-3

    def test_ciaaw_isotope_mass_nonexistent(self) -> None:
        """Verify requesting a non-existent isotope raises IsotopeNotFoundError with diagnostic details."""
        with pytest.raises(IsotopeNotFoundError) as exc_info:
            get_ciaaw_isotope_mass("C", 99)
        assert exc_info.value.symbol == "C"
        assert exc_info.value.mass_number == 99
        assert exc_info.value.available_mass_numbers is not None
        assert 12 in exc_info.value.available_mass_numbers
        assert 13 in exc_info.value.available_mass_numbers

        with pytest.raises(IsotopeNotFoundError):
            get_ciaaw_isotope_mass("H", 100)

    def test_get_available_isotopes(self) -> None:
        """Verify retrieval of available isotope mass numbers."""
        h_isotopes = get_available_isotopes("H")
        assert 1 in h_isotopes
        assert 2 in h_isotopes

        c_isotopes = get_available_isotopes("C")
        assert 12 in c_isotopes
        assert 13 in c_isotopes
        assert 14 in c_isotopes


# =============================================================================
# 3. EXTENDED ELEMENT PROPERTIES
# =============================================================================


class TestElementProperties:
    """Test helper lookups for symbols, names, and electronegativity."""

    def test_get_element_symbol_and_name(self) -> None:
        """Verify atomic number to symbol and name mappings."""
        assert get_element_symbol(1) == "H"
        assert get_element_symbol(6) == "C"
        assert get_element_symbol(8) == "O"
        assert get_element_symbol(26) == "Fe"

        assert get_element_name("H") == "Hydrogen"
        assert get_element_name("C") == "Carbon"
        assert get_element_name("fe") == "Iron"

    def test_get_element_electronegativity(self) -> None:
        """Verify Pauling electronegativity values."""
        en_f = get_element_electronegativity("F")
        en_o = get_element_electronegativity("O")
        en_c = get_element_electronegativity("C")
        en_h = get_element_electronegativity("H")
        en_na = get_element_electronegativity("Na")

        assert en_f is not None and en_f > 3.9
        assert en_o is not None and 3.4 < en_o < 3.6
        assert en_c is not None and 2.5 < en_c < 2.6
        assert en_h is not None and 2.1 < en_h < 2.3
        assert en_na is not None and en_na < 1.0


# =============================================================================
# 4. SPIN-PARITY AND VALENCE VALIDATION GATE (REQ-QC-081)
# =============================================================================


class TestSpinParityValidator:
    """
    Test pre-flight spin-parity and charge validation according to REQ-QC-081.

    Mathematical Invariants:
    1. Total nuclear charge Z_total = sum(Z_i).
    2. Total electron count N_e = Z_total - q >= 1.
    3. Parity condition: (N_e + M) mod 2 == 1 where M = 2S + 1.
    4. Multiplicity ceiling: M - 1 <= N_e.
    """

    def test_neutral_closed_shell_water(self) -> None:
        """Test neutral water H2O (N_e=10, M=1 -> Pass, M=2 -> Fail)."""
        symbols = ["H", "H", "O"]
        res = validate_charge_and_multiplicity(symbols, charge=0, multiplicity=1)
        assert isinstance(res, SpinChargeValidationResult)
        assert res.is_valid is True
        assert res.total_nuclear_charge == 10
        assert res.total_net_charge == 0
        assert res.total_electrons == 10
        assert res.multiplicity == 1
        assert res.parity_valid is True
        assert res.max_unpaired_electrons == 0

        # Parity violation: even N_e (10) with even M (2) gives (10+2)%2 = 0 != 1
        with pytest.raises(SpinParityViolationError) as exc_info:
            validate_charge_and_multiplicity(symbols, charge=0, multiplicity=2)
        assert exc_info.value.n_electrons == 10
        assert exc_info.value.multiplicity == 2

    def test_neutral_radical_oh(self) -> None:
        """Test hydroxyl radical OH (N_e=9, M=2 -> Pass, M=1 -> Fail)."""
        symbols = ["O", "H"]
        res = validate_charge_and_multiplicity(symbols, charge=0, multiplicity=2)
        assert res.is_valid is True
        assert res.total_nuclear_charge == 9
        assert res.total_net_charge == 0
        assert res.total_electrons == 9
        assert res.multiplicity == 2
        assert res.parity_valid is True
        assert res.max_unpaired_electrons == 1

        # Parity violation: odd N_e (9) with odd M (1) gives (9+1)%2 = 0 != 1
        with pytest.raises(SpinParityViolationError) as exc_info:
            validate_charge_and_multiplicity(symbols, charge=0, multiplicity=1)
        assert exc_info.value.n_electrons == 9
        assert exc_info.value.multiplicity == 1

    def test_triplet_and_singlet_oxygen(self) -> None:
        """Test ground-state triplet O2 (N_e=16, M=3) and excited singlet O2 (N_e=16, M=1)."""
        symbols = ["O", "O"]
        # Triplet O2
        res_trip = validate_charge_and_multiplicity(symbols, charge=0, multiplicity=3)
        assert res_trip.is_valid is True
        assert res_trip.total_electrons == 16
        assert res_trip.multiplicity == 3
        assert res_trip.max_unpaired_electrons == 2

        # Singlet O2
        res_sing = validate_charge_and_multiplicity(symbols, charge=0, multiplicity=1)
        assert res_sing.is_valid is True
        assert res_sing.total_electrons == 16
        assert res_sing.multiplicity == 1
        assert res_sing.max_unpaired_electrons == 0

        # Quintet O2
        res_quin = validate_charge_and_multiplicity(symbols, charge=0, multiplicity=5)
        assert res_quin.is_valid is True
        assert res_quin.max_unpaired_electrons == 4

        # Doublet O2 (unphysical for neutral O2)
        with pytest.raises(SpinParityViolationError):
            validate_charge_and_multiplicity(symbols, charge=0, multiplicity=2)

    def test_ionic_and_polyatomic_systems(self) -> None:
        """Test various polyatomic cations and anions."""
        # Hydronium H3O+ (Z=11, q=+1, N_e=10, M=1)
        res_h3o = validate_charge_and_multiplicity(["H", "H", "H", "O"], charge=1, multiplicity=1)
        assert res_h3o.is_valid is True
        assert res_h3o.total_electrons == 10

        # Hydroxide OH- (Z=9, q=-1, N_e=10, M=1)
        res_oh_minus = validate_charge_and_multiplicity(["O", "H"], charge=-1, multiplicity=1)
        assert res_oh_minus.is_valid is True
        assert res_oh_minus.total_electrons == 10

        # Superoxide O2^- (Z=16, q=-1, N_e=17, M=2)
        res_superoxide = validate_charge_and_multiplicity(["O", "O"], charge=-1, multiplicity=2)
        assert res_superoxide.is_valid is True
        assert res_superoxide.total_electrons == 17
        assert res_superoxide.multiplicity == 2

        # Carbonate CO3^2- (Z=30, q=-2, N_e=32, M=1)
        res_co3 = validate_charge_and_multiplicity(["C", "O", "O", "O"], charge=-2, multiplicity=1)
        assert res_co3.is_valid is True
        assert res_co3.total_electrons == 32

        # Ammonium NH4+ (Z=11, q=+1, N_e=10, M=1)
        res_nh4 = validate_charge_and_multiplicity(["N", "H", "H", "H", "H"], charge=1, multiplicity=1)
        assert res_nh4.is_valid is True
        assert res_nh4.total_electrons == 10

    def test_multiplicity_ceiling_and_unphysical_bounds(self) -> None:
        """Verify multiplicity exceeding total electrons or unphysical bounds raises SpinParityViolationError."""
        # H atom (Z=1, q=0, N_e=1): multiplicity=3 requires 2 unpaired electrons, but N_e=1
        with pytest.raises(SpinParityViolationError):
            validate_charge_and_multiplicity(["H"], charge=0, multiplicity=3)

        # Multiplicity < 1
        with pytest.raises(SpinParityViolationError):
            validate_charge_and_multiplicity(["H"], charge=0, multiplicity=0)

        # Total electron count < 1 (e.g. bare proton with q=+2 -> N_e = -1)
        with pytest.raises(SpinParityViolationError):
            validate_charge_and_multiplicity(["H"], charge=2, multiplicity=1)

        # Empty symbols list
        with pytest.raises(ValueError):
            validate_charge_and_multiplicity([], charge=0, multiplicity=1)

    def test_unphysical_valence_warnings(self) -> None:
        """Verify heuristic warnings are emitted for highly unphysical oxidation states."""
        # Single carbon with extreme charge +4 is C4+ (valid 2 electrons, singlet), but charge +5 gives C5+ (1 electron, doublet)
        res = validate_charge_and_multiplicity(["C"], charge=5, multiplicity=2)
        assert res.is_valid is True
        assert res.total_electrons == 1
        # Carbon in +5 state is an extreme bare-core ion, expect diagnostic warning
        assert len(res.unphysical_valence_warnings) >= 1
        assert any("Carbon" in w or "oxidation state" in w or "valence" in w for w in res.unphysical_valence_warnings)


# =============================================================================
# 5. MENDELEEV INTEGRITY AND AST COMPLIANCE AUDIT
# =============================================================================


class TestMendeleevIntegrityAndASTCompliance:
    """Verify implementation complies with Zero-Mock & Mendeleev mass retrieval directives."""

    def test_mendeleev_library_is_actively_imported(self) -> None:
        """Verify that src/cochem/core/cochem_elements.py imports mendeleev."""
        import cochem.core.cochem_elements as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        has_mendeleev_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if "mendeleev" in name.name:
                        has_mendeleev_import = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and "mendeleev" in node.module:
                    has_mendeleev_import = True

        assert has_mendeleev_import, "Module must dynamically import mendeleev for atomic masses."

    def test_no_hardcoded_mass_dictionaries(self) -> None:
        """Verify no static mass lookup tables exist in source."""
        import cochem.core.cochem_elements as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        if k.value in {"H", "C", "N", "O", "Cl", "Fe"}:
                            if isinstance(v, ast.Constant) and isinstance(v.value, float):
                                pytest.fail(f"Detected hardcoded atomic mass dictionary for '{k.value}': {v.value}")

    def test_no_dead_end_logic(self) -> None:
        """Verify no bare pass or NotImplementedError in implementation."""
        import cochem.core.cochem_elements as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for stmt in node.body:
                    if isinstance(stmt, ast.Pass):
                        pytest.fail(f"Found bare pass statement in function '{node.name}'")
                    if isinstance(stmt, ast.Raise):
                        if isinstance(stmt.exc, ast.Name) and stmt.exc.id == "NotImplementedError":
                            pytest.fail(f"Found NotImplementedError in function '{node.name}'")
                        if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                            pytest.fail(f"Found NotImplementedError in function '{node.name}'")

