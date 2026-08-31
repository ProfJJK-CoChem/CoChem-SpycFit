"""
Unit tests for CoChem Quantum Chemistry Pydantic v2 Core Data Models.

Strictly adhering to:
- Zero-Mock Mandate (zero mocks, zero stubs, zero synthetic arrays, authentic physical data).
- Provenance tracking [M], [D], [E].
"""

import pytest
from pydantic import ValidationError

import cochem.qc as cochem_qc
from cochem.qc.models import (
    BondCriticalPointDescriptor,
    IRCTrajectoryResult,
    IsotopeSubstitutionPayload,
    LevelOfTheoryEnvelope,
    MarcusReorganizationResult,
    ProjectedFrequencyResult,
    SOCMEResult,
    SpinChargeValidationResult,
    StandardStateCorrectionPayload,
    WavefunctionStabilityReport,
)


class TestCoChemQCModels:
    """Test suite for CoChem QC frozen Pydantic v2 data models."""

    def test_module_reexport_integrity(self) -> None:
        """[D] Verify cochem.qc re-exports all 10 model classes in __all__."""
        expected_models = [
            "SpinChargeValidationResult",
            "MarcusReorganizationResult",
            "IRCTrajectoryResult",
            "StandardStateCorrectionPayload",
            "LevelOfTheoryEnvelope",
            "IsotopeSubstitutionPayload",
            "WavefunctionStabilityReport",
            "BondCriticalPointDescriptor",
            "SOCMEResult",
            "ProjectedFrequencyResult",
        ]
        for name in expected_models:
            assert hasattr(cochem_qc, name), f"cochem.qc missing export {name}"
            assert name in cochem_qc.__all__, f"{name} not in cochem.qc.__all__"

    def test_spin_charge_validation_result_valid(self) -> None:
        """[M] Test valid SpinChargeValidationResult instantiation and frozen immutability."""
        result = SpinChargeValidationResult(
            is_valid=True,
            total_nuclear_charge=10,
            total_net_charge=0,
            total_electrons=10,
            multiplicity=1,
            parity_valid=True,
            max_unpaired_electrons=0,
            unphysical_valence_warnings=[],
        )
        assert result.is_valid is True
        assert result.total_nuclear_charge == 10
        assert result.total_net_charge == 0
        assert result.total_electrons == 10
        assert result.multiplicity == 1
        assert result.parity_valid is True
        assert result.max_unpaired_electrons == 0
        assert result.unphysical_valence_warnings == []

        # Immutability check
        with pytest.raises(ValidationError):
            result.is_valid = False  # type: ignore[misc]

    def test_spin_charge_validation_result_bounds(self) -> None:
        """[M] Test SpinChargeValidationResult field bounds."""
        # total_nuclear_charge < 1
        with pytest.raises(ValidationError):
            SpinChargeValidationResult(
                is_valid=False,
                total_nuclear_charge=0,
                total_net_charge=0,
                total_electrons=1,
                multiplicity=1,
                parity_valid=True,
                max_unpaired_electrons=0,
            )

        # total_electrons < 1
        with pytest.raises(ValidationError):
            SpinChargeValidationResult(
                is_valid=False,
                total_nuclear_charge=1,
                total_net_charge=1,
                total_electrons=0,
                multiplicity=1,
                parity_valid=True,
                max_unpaired_electrons=0,
            )

        # multiplicity < 1
        with pytest.raises(ValidationError):
            SpinChargeValidationResult(
                is_valid=False,
                total_nuclear_charge=1,
                total_net_charge=0,
                total_electrons=1,
                multiplicity=0,
                parity_valid=False,
                max_unpaired_electrons=0,
            )

    def test_marcus_reorganization_result_valid(self) -> None:
        """[M][D] Test valid MarcusReorganizationResult instantiation and serialization."""
        marcus = MarcusReorganizationResult(
            e_neutral_at_neutral_geom_hartree=-76.4000,
            e_charged_at_neutral_geom_hartree=-76.0000,
            e_charged_at_charged_geom_hartree=-76.0500,
            e_neutral_at_charged_geom_hartree=-76.3500,
            lambda_1_ev=1.36057,
            lambda_2_ev=1.36057,
            lambda_inner_ev=2.72114,
            lambda_inner_kcal_mol=62.7509,
            lambda_outer_ev=0.2500,
            total_reorganization_energy_ev=2.97114,
            gibbs_free_energy_delta_ev=-0.5000,
            marcus_activation_barrier_ev=0.5140,
            electron_transfer_rate_s_inv=1.25e8,
        )
        assert marcus.lambda_1_ev == 1.36057
        assert marcus.lambda_inner_ev == 2.72114
        assert marcus.lambda_outer_ev == 0.2500
        assert marcus.electron_transfer_rate_s_inv == 1.25e8

        data = marcus.model_dump()
        assert data["total_reorganization_energy_ev"] == 2.97114
        deserialized = MarcusReorganizationResult.model_validate(data)
        assert deserialized == marcus

    def test_marcus_reorganization_result_bounds(self) -> None:
        """[M] Test MarcusReorganizationResult negative reorganization energy rejection."""
        with pytest.raises(ValidationError):
            MarcusReorganizationResult(
                e_neutral_at_neutral_geom_hartree=-76.4,
                e_charged_at_neutral_geom_hartree=-76.0,
                e_charged_at_charged_geom_hartree=-76.05,
                e_neutral_at_charged_geom_hartree=-76.35,
                lambda_1_ev=-0.1,  # Must be >= 0.0
                lambda_2_ev=1.0,
                lambda_inner_ev=0.9,
                lambda_inner_kcal_mol=20.0,
                total_reorganization_energy_ev=0.9,
                gibbs_free_energy_delta_ev=0.0,
                marcus_activation_barrier_ev=0.225,
            )

    def test_irc_trajectory_result_valid(self) -> None:
        """[M][E] Test valid IRCTrajectoryResult."""
        coords = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.1], [0.0, 0.0, 1.1]],
        ]
        energies = [-100.5000, -100.5020]
        irc = IRCTrajectoryResult(
            ts_energy_hartree=-100.4500,
            ts_imaginary_frequency_cm1=-450.2,
            forward_steps=25,
            reverse_steps=25,
            forward_endpoint_energy_hartree=-100.5200,
            reverse_endpoint_energy_hartree=-100.5150,
            forward_endpoint_rmsd_to_reactant_angstrom=0.042,
            reverse_endpoint_rmsd_to_product_angstrom=0.058,
            connectivity_verified=True,
            path_coordinates_bohr=coords,
            path_energies_hartree=energies,
        )
        assert irc.ts_imaginary_frequency_cm1 == -450.2
        assert irc.forward_steps == 25
        assert irc.connectivity_verified is True
        assert len(irc.path_coordinates_bohr) == 2

    def test_irc_trajectory_result_bounds(self) -> None:
        """[M] Test IRCTrajectoryResult imaginary frequency (must be < 0) and steps (>= 1)."""
        # Positive imaginary frequency must fail
        with pytest.raises(ValidationError):
            IRCTrajectoryResult(
                ts_energy_hartree=-100.0,
                ts_imaginary_frequency_cm1=100.0,  # Invalid: must be lt=0.0
                forward_steps=10,
                reverse_steps=10,
                forward_endpoint_energy_hartree=-100.1,
                reverse_endpoint_energy_hartree=-100.1,
                connectivity_verified=True,
                path_coordinates_bohr=[],
                path_energies_hartree=[],
            )

        # 0 steps must fail
        with pytest.raises(ValidationError):
            IRCTrajectoryResult(
                ts_energy_hartree=-100.0,
                ts_imaginary_frequency_cm1=-200.0,
                forward_steps=0,  # Invalid: ge=1
                reverse_steps=10,
                forward_endpoint_energy_hartree=-100.1,
                reverse_endpoint_energy_hartree=-100.1,
                connectivity_verified=True,
                path_coordinates_bohr=[],
                path_energies_hartree=[],
            )

    def test_standard_state_correction_payload(self) -> None:
        """[D] Test StandardStateCorrectionPayload."""
        payload = StandardStateCorrectionPayload(
            temperature_kelvin=298.15,
            pressure_atm=1.0,
            gas_molar_volume_liters=24.4654,
            gas_standard_concentration_m=0.040874,
            solution_standard_concentration_m=1.0,
            delta_g_1atm_to_1m_kcal_mol=1.8943,
            delta_g_1atm_to_1m_kj_mol=7.9258,
            g_gas_1atm_hartree=-76.4250,
            g_solv_1m_hartree=-76.4320,
            solvent_name="water",
            is_neat_solvent_corrected=False,
        )
        assert payload.temperature_kelvin == 298.15
        assert payload.delta_g_1atm_to_1m_kcal_mol == 1.8943
        assert payload.solvent_name == "water"

        # T <= 0 must fail
        with pytest.raises(ValidationError):
            StandardStateCorrectionPayload(
                temperature_kelvin=-5.0,
                pressure_atm=1.0,
                gas_molar_volume_liters=24.0,
                gas_standard_concentration_m=0.04,
                delta_g_1atm_to_1m_kcal_mol=1.89,
                delta_g_1atm_to_1m_kj_mol=7.92,
                g_gas_1atm_hartree=-76.0,
                g_solv_1m_hartree=-76.0,
                solvent_name="water",
            )

    def test_level_of_theory_envelope(self) -> None:
        """[D] Test LevelOfTheoryEnvelope and hashing functionality."""
        envelope = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="SMD(water)",
            max_gradient_residual_au=1.5e-5,
        )
        assert envelope.hamiltonian_method == "wB97X-D4"
        assert envelope.max_gradient_residual_au == 1.5e-5

        # Immutability
        with pytest.raises(ValidationError):
            envelope.basis_set = "def2-SVP"  # type: ignore[misc]

        # Negative gradient residual must fail
        with pytest.raises(ValidationError):
            LevelOfTheoryEnvelope(
                hamiltonian_method="B3LYP",
                basis_set="6-31G*",
                integration_grid="UltraFine",
                max_gradient_residual_au=-1.0e-5,
            )

    def test_isotope_substitution_payload(self) -> None:
        """[D] Test IsotopeSubstitutionPayload."""
        payload = IsotopeSubstitutionPayload(
            substituted_atoms={0: 2},
            mendeleev_masses_amu={0: 2.0141017778},
            reference_zpe_kcal_mol=13.25,
            substituted_zpe_kcal_mol=11.50,
            reference_frequencies_cm1=[1595.0, 3657.0, 3756.0],
            substituted_frequencies_cm1=[1178.0, 2671.0, 3707.0],
            kie_bigeleisen_mayer=1.385,
            mmi_factor=1.025,
            exc_factor=1.002,
            zpe_factor=1.348,
        )
        assert payload.substituted_atoms[0] == 2
        assert payload.kie_bigeleisen_mayer == 1.385
        assert len(payload.reference_frequencies_cm1) == 3

    def test_wavefunction_stability_report(self) -> None:
        """[M] Test WavefunctionStabilityReport literal enum and stability metrics."""
        report = WavefunctionStabilityReport(
            initial_stability="UNSTABLE_INTERNAL_RHF_UHF",
            lowest_orbital_hessian_eigenvalue=-0.035,
            is_broken_symmetry_required=True,
            s2_expected=0.0,
            s2_calculated=0.985,
            spin_contamination_percentage=98.5,
            yamaguchi_corrected_energy_hartree=-150.8250,
        )
        assert report.initial_stability == "UNSTABLE_INTERNAL_RHF_UHF"
        assert report.is_broken_symmetry_required is True
        assert report.yamaguchi_corrected_energy_hartree == -150.8250

        # Invalid stability literal
        with pytest.raises(ValidationError):
            WavefunctionStabilityReport(
                initial_stability="UNKNOWN_STATUS",  # type: ignore[arg-type]
                lowest_orbital_hessian_eigenvalue=0.01,
                is_broken_symmetry_required=False,
                s2_expected=0.0,
                s2_calculated=0.0,
                spin_contamination_percentage=0.0,
            )

    def test_bond_critical_point_descriptor(self) -> None:
        """[M] Test BondCriticalPointDescriptor topological properties."""
        bcp = BondCriticalPointDescriptor(
            bcp_index=0,
            atom_1_index=0,
            atom_2_index=1,
            coordinates_angstrom=(0.0, 0.0, 0.48),
            electron_density_au=0.285,
            laplacian_au=-0.850,
            kinetic_energy_density_g_au=0.120,
            potential_energy_density_v_au=-0.3325,
            total_energy_density_h_au=-0.2125,
            ellipticity=0.015,
            interaction_classification="COVALENT_SHARED",
        )
        assert bcp.bcp_index == 0
        assert bcp.electron_density_au == 0.285
        assert bcp.interaction_classification == "COVALENT_SHARED"

        # Invalid classification literal
        with pytest.raises(ValidationError):
            BondCriticalPointDescriptor(
                bcp_index=0,
                atom_1_index=0,
                atom_2_index=1,
                coordinates_angstrom=(0.0, 0.0, 0.0),
                electron_density_au=0.1,
                laplacian_au=0.1,
                kinetic_energy_density_g_au=0.1,
                potential_energy_density_v_au=-0.1,
                total_energy_density_h_au=0.0,
                ellipticity=0.0,
                interaction_classification="UNKNOWN_BOND",  # type: ignore[arg-type]
            )

        # Non-positive electron density must fail
        with pytest.raises(ValidationError):
            BondCriticalPointDescriptor(
                bcp_index=0,
                atom_1_index=0,
                atom_2_index=1,
                coordinates_angstrom=(0.0, 0.0, 0.0),
                electron_density_au=0.0,  # Invalid: gt=0.0
                laplacian_au=0.0,
                kinetic_energy_density_g_au=0.0,
                potential_energy_density_v_au=0.0,
                total_energy_density_h_au=0.0,
                ellipticity=0.0,
                interaction_classification="COVALENT_SHARED",
            )

    def test_socme_result(self) -> None:
        """[M][D] Test SOCMEResult photophysics parameters."""
        socme = SOCMEResult(
            initial_state="S1",
            final_state="T1",
            delta_e_singlet_triplet_ev=0.35,
            socme_magnitude_cm1=1.85,
            socme_components_cm1=(0.55, 1.20, 1.29),
            fcwd_density_states=0.045,
            k_isc_s_inv=3.2e7,
            phosphorescence_lifetime_seconds=0.0012,
        )
        assert socme.initial_state == "S1"
        assert socme.socme_magnitude_cm1 == 1.85
        assert socme.k_isc_s_inv == 3.2e7
        assert socme.phosphorescence_lifetime_seconds == 0.0012

    def test_projected_frequency_result(self) -> None:
        """[M] Test ProjectedFrequencyResult Sayvetz-Eckart properties."""
        proj = ProjectedFrequencyResult(
            unprojected_frequencies_cm1=[12.5, -4.2, 0.8, 1.1, -0.5, 3.4, 1595.0, 3657.0, 3756.0],
            projected_frequencies_cm1=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1595.0, 3657.0, 3756.0],
            rot_trans_residual_max_cm1=1.2e-5,
            num_projected_zero_modes=6,
            is_projection_clean=True,
        )
        assert len(proj.unprojected_frequencies_cm1) == 9
        assert len(proj.projected_frequencies_cm1) == 9
        assert proj.rot_trans_residual_max_cm1 == 1.2e-5
        assert proj.num_projected_zero_modes == 6
        assert proj.is_projection_clean is True
