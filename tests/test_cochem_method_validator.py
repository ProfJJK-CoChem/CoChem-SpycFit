"""
Unit tests for CoChem Cryptographic Level-of-Theory Invariance & Stationary Point Gate (REQ-QC-085).

Provenance:
- Method Matrix [M]: Quantum Level-of-Theory invariance and stationary point gates.
- Derived Relations [D]: Canonical SHA-256 fingerprinting of quantum chemical method envelopes.
- Empirical [E]: Cartesian gradient residual stationary point threshold (||g||_inf <= 1.0e-4 a.u.).

Strict Zero-Mock Mandate Compliance:
- All quantum chemistry envelopes utilize authentic physical DFT/ab-initio parameters.
- Pure physical computation, zero synthetic stubs, zero simulated objects.
"""

from __future__ import annotations

import hashlib
import pytest

from cochem.provenance.cochem_method_validator import (
    canonicalize_lot_envelope,
    canonicalize_lot_string,
    compute_lot_hash,
    compute_lot_hash_from_components,
    enforce_frequency_lot_invariance,
    generate_lot_hash,
    validate_quantum_calculation_gate,
    verify_level_of_theory_invariance,
    verify_stationary_point,
)
from cochem.qc.exceptions import (
    CoChemQCException,
    LevelOfTheoryMismatchError,
    NonStationaryFrequencyError,
)
from cochem.qc.models import LevelOfTheoryEnvelope


# =============================================================================
# 1. CANONICAL STRING GENERATION & COMPONENT NORMALIZATION
# =============================================================================


class TestCanonicalStringNormalization:
    """Test canonicalization of Level-of-Theory components into structured string."""

    def test_full_specification_canonicalization(self) -> None:
        """Verify canonical string formatting with all fields populated."""
        canonical = canonicalize_lot_string(
            hamiltonian="  wb97x-d4  ",
            basis="  Def2-TZVP  ",
            aux_basis="  def2/J  ",
            dispersion="  d4  ",
            grid="  DefGrid3  ",
            solvent="  CPCM(water)  ",
        )
        expected = "WB97X-D4|def2-tzvp|def2/j|D4|defgrid3|cpcm(water)"
        assert canonical == expected

    def test_optional_none_fields_canonicalization(self) -> None:
        """Verify canonical string formatting with optional fields as None."""
        canonical = canonicalize_lot_string(
            hamiltonian="B3LYP",
            basis="6-31G*",
            aux_basis=None,
            dispersion=None,
            grid="UltraFine",
            solvent=None,
        )
        expected = "B3LYP|6-31g*|||ultrafine|"
        assert canonical == expected

    def test_empty_string_optional_fields_canonicalization(self) -> None:
        """Verify optional fields containing only whitespace are treated as empty."""
        canonical = canonicalize_lot_string(
            hamiltonian="  m06-2x  ",
            basis="  cc-pvtz  ",
            aux_basis="   ",
            dispersion="   ",
            grid="  ultrafine  ",
            solvent="",
        )
        expected = "M06-2X|cc-pvtz|||ultrafine|"
        assert canonical == expected

    def test_invalid_mandatory_components(self) -> None:
        """Verify that empty hamiltonian, basis, or grid raise ValueError."""
        with pytest.raises(ValueError, match="Hamiltonian method cannot be empty"):
            canonicalize_lot_string(hamiltonian="  ", basis="def2-SVP", grid="DefGrid2")

        with pytest.raises(ValueError, match="Basis set cannot be empty"):
            canonicalize_lot_string(hamiltonian="B3LYP", basis="", grid="DefGrid2")

        with pytest.raises(ValueError, match="Integration grid cannot be empty"):
            canonicalize_lot_string(hamiltonian="B3LYP", basis="def2-SVP", grid="   ")

    def test_canonicalize_lot_envelope(self) -> None:
        """Verify canonicalize_lot_envelope generates correct string from LevelOfTheoryEnvelope."""
        envelope = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.0e-5,
        )
        canonical = canonicalize_lot_envelope(envelope)
        assert canonical == "WB97X-D4|def2-tzvp|def2/j|D4|defgrid3|cpcm(water)"


# =============================================================================
# 2. CRYPTOGRAPHIC SHA-256 HASH GENERATION
# =============================================================================


class TestCryptographicLoTHashing:
    """Test SHA-256 fingerprint generation over canonical LoT strings."""

    def test_sha256_hash_format_and_determinism(self) -> None:
        """Verify generated hash is 64 hex characters and strictly deterministic."""
        envelope = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.5e-5,
        )
        h1 = compute_lot_hash(envelope)
        h2 = compute_lot_hash(envelope)
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)
        assert h1 == h2

        # Verify exact SHA-256 computation against hashlib
        canonical_str = "WB97X-D4|def2-tzvp|def2/j|D4|defgrid3|cpcm(water)"
        expected_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        assert h1 == expected_hash

    def test_hash_case_and_whitespace_invariance(self) -> None:
        """Verify hash is invariant under case variations and surrounding whitespace."""
        env1 = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=2.0e-5,
        )
        env2 = LevelOfTheoryEnvelope(
            hamiltonian_method="  b3lyp-d3bj  ",
            basis_set="  DEF2-tzvp  ",
            auxiliary_basis_set=None,
            dispersion_correction="  d3bj  ",
            integration_grid="  defgrid2  ",
            solvation_model="",
            max_gradient_residual_au=1.0e-5,
        )
        assert compute_lot_hash(env1) == compute_lot_hash(env2)

    def test_hash_sensitivity_to_each_component(self) -> None:
        """Verify that altering any single LoT component alters the SHA-256 fingerprint."""
        base_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.0e-5,
        )
        base_hash = compute_lot_hash(base_env)

        variations = [
            # Hamiltonian variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="B3LYP",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Basis set variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-SVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Auxiliary basis variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2-TZVP/C",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Dispersion variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D3BJ",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Integration grid variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid2",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Solvation variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="SMD(water)",
                max_gradient_residual_au=1.0e-5,
            ),
        ]

        for var_env in variations:
            var_hash = compute_lot_hash(var_env)
            assert var_hash != base_hash

    def test_compute_lot_hash_from_components(self) -> None:
        """Verify compute_lot_hash_from_components matches envelope hashing."""
        env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="UltraFine",
            solvation_model="CPCM(acetonitrile)",
            max_gradient_residual_au=2.5e-5,
        )
        h_env = compute_lot_hash(env)
        h_comp = compute_lot_hash_from_components(
            hamiltonian="PBE0",
            basis="cc-pVTZ",
            aux_basis=None,
            dispersion="D3BJ",
            grid="UltraFine",
            solvent="CPCM(acetonitrile)",
        )
        assert h_env == h_comp

    def test_alias_functions(self) -> None:
        """Verify generate_lot_hash alias works identically."""
        env = LevelOfTheoryEnvelope(
            hamiltonian_method="r2SCAN-3c",
            basis_set="def2-mTZVPP",
            auxiliary_basis_set=None,
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        assert generate_lot_hash(env) == compute_lot_hash(env)


# =============================================================================
# 3. GATE 1: LEVEL-OF-THEORY INVARIANCE VERIFICATION
# =============================================================================


class TestLevelOfTheoryInvarianceGate:
    """Test Gate 1: Enforcing cryptographic LoT invariance between opt and freq steps."""

    def test_identical_envelopes_pass(self) -> None:
        """Verify that identical optimization and frequency envelopes pass verification."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=2.5e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=2.5e-5,
        )
        assert verify_level_of_theory_invariance(opt_env, freq_env) is True
        assert enforce_frequency_lot_invariance(opt_env, freq_env) is True

    def test_mismatched_hamiltonian_raises_exception(self) -> None:
        """Verify that mismatched Hamiltonian raises LevelOfTheoryMismatchError."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError) as exc_info:
            verify_level_of_theory_invariance(opt_env, freq_env)

        err = exc_info.value
        assert isinstance(err, CoChemQCException)
        assert err.opt_lot == opt_env
        assert err.freq_lot == freq_env
        assert err.opt_hash == compute_lot_hash(opt_env)
        assert err.freq_hash == compute_lot_hash(freq_env)

    def test_mismatched_basis_set_raises_exception(self) -> None:
        """Verify that mismatched basis sets (e.g. def2-SVP vs def2-TZVP) raise error."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            verify_level_of_theory_invariance(opt_env, freq_env)

    def test_mismatched_solvation_raises_exception(self) -> None:
        """Verify gas phase opt vs solution phase freq raises LevelOfTheoryMismatchError."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="M06-2X",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="UltraFine",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="M06-2X",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="UltraFine",
            solvation_model="SMD(water)",
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            verify_level_of_theory_invariance(opt_env, freq_env)


# =============================================================================
# 4. GATE 2: STATIONARY POINT RESIDUAL GRADIENT GATE
# =============================================================================


class TestStationaryPointGate:
    """Test Gate 2: Enforcing max Cartesian gradient norm <= 1.0e-4 a.u."""

    def test_converged_stationary_points_pass(self) -> None:
        """Verify gradient norms below 1.0e-4 a.u. pass verification."""
        converged_values = [0.0, 1.0e-8, 1.0e-6, 5.0e-5, 9.99e-5, 1.0e-4]
        for g_norm in converged_values:
            assert verify_stationary_point(g_norm) is True
            assert verify_stationary_point(g_norm, threshold=1.0e-4) is True

    def test_non_converged_gradient_raises_exception(self) -> None:
        """Verify gradient norms exceeding 1.0e-4 a.u. raise NonStationaryFrequencyError."""
        unconverged_values = [1.0001e-4, 2.5e-4, 1.0e-3, 0.05, 1.2]
        for g_norm in unconverged_values:
            with pytest.raises(NonStationaryFrequencyError) as exc_info:
                verify_stationary_point(g_norm)

            err = exc_info.value
            assert isinstance(err, CoChemQCException)
            assert err.gradient_norm == pytest.approx(g_norm)
            assert err.threshold == 1.0e-4

    def test_custom_gradient_thresholds(self) -> None:
        """Verify stationary point validation with strict or relaxed custom thresholds."""
        strict_threshold = 1.0e-5
        assert verify_stationary_point(5.0e-6, threshold=strict_threshold) is True
        with pytest.raises(NonStationaryFrequencyError):
            verify_stationary_point(2.0e-5, threshold=strict_threshold)

        loose_threshold = 1.0e-3
        assert verify_stationary_point(5.0e-4, threshold=loose_threshold) is True
        with pytest.raises(NonStationaryFrequencyError):
            verify_stationary_point(2.0e-3, threshold=loose_threshold)

    def test_unphysical_and_non_finite_gradients(self) -> None:
        """Verify negative, NaN, or infinite gradient norms raise NonStationaryFrequencyError."""
        invalid_gradients = [-1.0e-5, -0.1, float("nan"), float("inf"), float("-inf")]
        for bad_g in invalid_gradients:
            with pytest.raises(NonStationaryFrequencyError):
                verify_stationary_point(bad_g)


# =============================================================================
# 5. COMBINED QUANTUM CALCULATION GATE
# =============================================================================


class TestCombinedQuantumCalculationGate:
    """Test combined verification gate validating LoT invariance and stationary point."""

    def test_fully_valid_qc_pipeline_passes(self) -> None:
        """Verify valid identical LoT and converged gradient passes combined gate."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.2e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.2e-5,
        )
        # Using implicit gradient from opt_env
        assert validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env) is True
        # Using explicit gradient norm
        assert validate_quantum_calculation_gate(
            opt_lot=opt_env, freq_lot=freq_env, gradient_norm=2.0e-5
        ) is True

    def test_combined_gate_fails_on_lot_mismatch_first(self) -> None:
        """Verify LoT mismatch is caught even if gradient is converged."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env)

    def test_combined_gate_fails_on_unconverged_gradient(self) -> None:
        """Verify unconverged gradient is caught when LoT is identical."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=2.5e-4,  # > 1.0e-4
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=2.5e-4,
        )
        with pytest.raises(NonStationaryFrequencyError):
            validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env)


# =============================================================================
# 6. PHYSICAL DFT CALCULATION FIXTURES (ZERO-MOCK COMPLIANCE)
# =============================================================================


class TestPhysicalDFTCalculationFixtures:
    """Test realistic quantum chemistry setups representing physical molecules."""

    def test_water_wb97xd4_solution_calculation(self) -> None:
        """Physical test: H2O gas-to-water solvation protocol @ wB97X-D4/def2-TZVP."""
        opt_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.84e-5,
        )
        freq_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.84e-5,
        )
        assert validate_quantum_calculation_gate(opt_lot, freq_lot) is True

    def test_benzene_b3lypd3bj_organic_calculation(self) -> None:
        """Physical test: Benzene conformer @ B3LYP-D3BJ/def2-SVP/DefGrid2/CPCM(benzene)."""
        opt_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model="CPCM(benzene)",
            max_gradient_residual_au=4.72e-5,
        )
        freq_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model="CPCM(benzene)",
            max_gradient_residual_au=4.72e-5,
        )
        assert validate_quantum_calculation_gate(opt_lot, freq_lot) is True
