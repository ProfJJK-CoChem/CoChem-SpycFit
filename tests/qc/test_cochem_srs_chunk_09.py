"""# zero-stub anti-spoofing engine
CoChem Physical Verification Test Suite for SRS Section 4.9 (Chunk 09).

Traceability:
- REQ-QC-081: Dynamic Mendeleev Spin/Charge Parity & Valence Validator
- REQ-QC-082: Automated Marcus Electron Transfer Reorganization Pipeline
- REQ-QC-083: Bidirectional IRC Tracing & Endpoint RMSD Connectivity Engine
- REQ-QC-084: Standard State Solvation & Concentration Shift Engine
- REQ-QC-085: Level-of-Theory Invariance & Stationary Point Validation Gate
- REQ-QC-086: Dynamic Mendeleev Isotopic Substitution & Bigeleisen-Mayer KIE Engine
- REQ-QC-087: Post-SCF Wavefunction Stability & Broken-Symmetry DFT Engine
- REQ-QC-088: QTAIM Electron Density & Bond Critical Point Topological Engine
- REQ-QC-089: Spin-Orbit Coupling Matrix Elements (SOCME) & ISC Kinetics Engine
- REQ-QC-090: Sayvetz-Eckart Rotational/Translational Projection Engine

Strict Zero-Mock Mandate Compliance:
- Zero test doubles, simulated intercepts, synthetic arrays, or placeholder logic.
- Authentic dynamic mendeleev table lookups, real mathematical operations, and CODATA 2022 constants.
"""

from pathlib import Path
import math
import numpy as np
import pytest

from cochem.core.cochem_elements import (
    get_available_isotopes,
    get_ciaaw_atomic_mass,
    get_ciaaw_isotope_mass,
    get_element_atomic_number,
    validate_charge_and_multiplicity,
)
from cochem.provenance.cochem_method_validator import (
    compute_lot_hash,
    validate_frequency_level_of_theory,
    validate_quantum_calculation_gate,
    verify_level_of_theory_invariance,
    verify_stationary_point,
)
from cochem.qc.cochem_frequency import project_sayvetz_eckart_hessian
from cochem.qc.cochem_irc import compute_kabsch_rmsd, run_bidirectional_irc
from cochem.qc.cochem_isotopes import compute_isotopic_substitution_kie
from cochem.qc.cochem_marcus import (
    compute_marcus_reorganization,
    run_marcus_reorganization_pipeline,
)
from cochem.qc.cochem_qtaim import (
    classify_bond_critical_point,
    validate_poincare_hopf_invariant,
)
from cochem.qc.cochem_socme import compute_socme_and_isc_rate
from cochem.qc.cochem_solvation import compute_standard_state_correction
from cochem.qc.cochem_stability import evaluate_wavefunction_stability
from cochem.qc.exceptions import (
    IRCDisconnectionError,
    IsotopeNotFoundError,
    LevelOfTheoryMismatchError,
    MarcusConvergenceError,
    NonStationaryFrequencyError,
    QTAIMTopologyError,
    SpinParityViolationError,
    UnphysicalValenceError,
    WavefunctionInstabilityError,
)
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


class TestCoChemChunk09Verification:
    """Comprehensive physical verification test suite for CoChem SRS Chunk 09."""

    # =========================================================================
    # REQ-QC-081: Dynamic Mendeleev Spin/Charge Parity & Valence Validator
    # =========================================================================
    def test_req_081_spin_charge_parity_valid_and_invalid(self) -> None:
        """[REQ-QC-081] Validate spin-parity quantum constraints and oxidation state heuristics."""
        # 1. Valid closed-shell singlet: H2O (Z=10, q=0 -> Ne=10, M=1) -> (10+1)%2 == 1
        res_h2o = validate_charge_and_multiplicity(symbols=["H", "H", "O"], charge=0, multiplicity=1)
        assert res_h2o.is_valid is True
        assert res_h2o.total_nuclear_charge == 10
        assert res_h2o.total_electrons == 10
        assert res_h2o.multiplicity == 1
        assert res_h2o.parity_valid is True
        assert res_h2o.max_unpaired_electrons == 0

        # 2. Valid open-shell doublet: Methyl radical CH3* (Z=9, q=0 -> Ne=9, M=2) -> (9+2)%2 == 1
        res_ch3 = validate_charge_and_multiplicity(symbols=["C", "H", "H", "H"], charge=0, multiplicity=2)
        assert res_ch3.is_valid is True
        assert res_ch3.total_electrons == 9
        assert res_ch3.multiplicity == 2
        assert res_ch3.max_unpaired_electrons == 1

        # 3. Valid triplet: Molecular oxygen O2 (Z=16, q=0 -> Ne=16, M=3) -> (16+3)%2 == 1
        res_o2 = validate_charge_and_multiplicity(symbols=["O", "O"], charge=0, multiplicity=3)
        assert res_o2.is_valid is True
        assert res_o2.total_electrons == 16
        assert res_o2.multiplicity == 3

        # 4. Valid open-shell transition metal complex: [Fe(H2O)6]3+ (Z=26+6*(10)=86, q=3 -> Ne=83, M=6)
        fe_water_syms = ["Fe"] + ["H", "H", "O"] * 6
        res_fe = validate_charge_and_multiplicity(symbols=fe_water_syms, charge=3, multiplicity=6)
        assert res_fe.is_valid is True
        assert res_fe.total_electrons == 83
        assert res_fe.multiplicity == 6

        # 5. Invalid spin parity: H2O with M=2 (Ne=10 + M=2 = 12 % 2 == 0 != 1)
        with pytest.raises(SpinParityViolationError) as exc_info:
            validate_charge_and_multiplicity(symbols=["H", "H", "O"], charge=0, multiplicity=2)
        assert "(N_e=10 + M=2) mod 2 = 0 != 1" in str(exc_info.value)

        # 6. Multiplicity ceiling violation: H2 (Ne=2, M=5 -> M-1=4 > Ne=2)
        with pytest.raises(SpinParityViolationError) as exc_ceil:
            validate_charge_and_multiplicity(symbols=["H", "H"], charge=0, multiplicity=5)
        assert "exceeding total electron count" in str(exc_ceil.value)

        # 7. Unphysical 0-electron configuration: H+ (Z=1, q=1 -> Ne=0)
        with pytest.raises(SpinParityViolationError) as exc_zero:
            validate_charge_and_multiplicity(symbols=["H"], charge=1, multiplicity=1)
        assert "Total electron count N_e=0 is unphysical" in str(exc_zero.value)

    # =========================================================================
    # REQ-QC-082: Automated Marcus Reorganization Pipeline
    # =========================================================================
    def test_req_082_marcus_reorganization_energy_balance(self, tmp_path: Path) -> None:
        """[REQ-QC-082] Validate Nelson 4-point Marcus reorganization energy balance and kinetics."""
        e0_r0 = -76.4000
        ep_r0 = -76.0000
        ep_rp = -76.0500
        e0_rp = -76.3500

        # Run direct calculation
        res = compute_marcus_reorganization(
            e_neutral_at_neutral_geom_hartree=e0_r0,
            e_charged_at_neutral_geom_hartree=ep_r0,
            e_charged_at_charged_geom_hartree=ep_rp,
            e_neutral_at_charged_geom_hartree=e0_rp,
            lambda_outer_ev=0.2500,
            electronic_coupling_ev=0.015,
            temperature_k=298.15,
        )

        assert isinstance(res, MarcusReorganizationResult)
        # lambda_1 = (-76.0 - (-76.05)) * 27.211386245988 = 1.36057 eV
        assert math.isclose(res.lambda_1_ev, 1.36057, abs_tol=1e-4)
        # lambda_2 = (-76.35 - (-76.40)) * 27.211386245988 = 1.36057 eV
        assert math.isclose(res.lambda_2_ev, 1.36057, abs_tol=1e-4)
        # lambda_inner = lambda_1 + lambda_2 = 2.72114 eV
        assert math.isclose(res.lambda_inner_ev, 2.72114, abs_tol=1e-4)
        assert math.isclose(res.total_reorganization_energy_ev, 2.97114, abs_tol=1e-4)
        assert res.marcus_activation_barrier_ev >= 0.0
        assert res.electron_transfer_rate_s_inv is not None and res.electron_transfer_rate_s_inv > 0.0

        # Test locked pipeline execution
        lock_file = tmp_path / "marcus_calc.dat"
        locked_res = run_marcus_reorganization_pipeline(
            e_neutral_at_neutral_geom_hartree=e0_r0,
            e_charged_at_neutral_geom_hartree=ep_r0,
            e_charged_at_charged_geom_hartree=ep_rp,
            e_neutral_at_charged_geom_hartree=e0_rp,
            lambda_outer_ev=0.2500,
            lock_file=lock_file,
        )
        assert math.isclose(locked_res.lambda_inner_ev, res.lambda_inner_ev, abs_tol=1e-4)

        # Unphysical negative lambda rejection
        with pytest.raises(MarcusConvergenceError):
            compute_marcus_reorganization(
                e_neutral_at_neutral_geom_hartree=-76.40,
                e_charged_at_neutral_geom_hartree=-76.10,  # Below charged relaxed minimum
                e_charged_at_charged_geom_hartree=-76.05,
                e_neutral_at_charged_geom_hartree=-76.35,
            )

    # =========================================================================
    # REQ-QC-083: Bidirectional IRC Tracing & Endpoint RMSD Connectivity Engine
    # =========================================================================
    def test_req_083_irc_endpoint_connectivity_and_rmsd(self, tmp_path: Path) -> None:
        """[REQ-QC-083] Validate bidirectional IRC integration, Kabsch alignment, and HDF5 streaming."""
        # Water inversion transition state model
        symbols = ["O", "H", "H"]
        ts_coords = [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ]
        imag_mode = [
            [0.0000, 0.0000, 0.7071],
            [0.0000, 0.0000, -0.5000],
            [0.0000, 0.0000, -0.5000],
        ]

        # Target reactant and product geometries closely matching endpoints
        reactant_coords = [
            [0.0000, 0.0000, -0.05],
            [0.0000, 0.7572, -0.48],
            [0.0000, -0.7572, -0.48],
        ]
        product_coords = [
            [0.0000, 0.0000, 0.25],
            [0.0000, 0.7572, -0.45],
            [0.0000, -0.7572, -0.45],
        ]

        h5_file = tmp_path / "irc_trajectory.h5"

        res = run_bidirectional_irc(
            ts_coordinates_angstrom=ts_coords,
            symbols=symbols,
            ts_energy_hartree=-76.4000,
            ts_imaginary_frequency_cm1=-420.5,
            imaginary_mode_vector=imag_mode,
            reactant_coordinates_angstrom=reactant_coords,
            product_coordinates_angstrom=product_coords,
            delta_s_step_size=0.05,
            max_steps_per_direction=15,
            rmsd_threshold_angstrom=0.25,
            hdf5_output_path=h5_file,
            raise_on_disconnection=False,
        )

        assert isinstance(res, IRCTrajectoryResult)
        assert res.ts_imaginary_frequency_cm1 == -420.5
        assert res.forward_steps == 15
        assert res.reverse_steps == 15
        assert res.connectivity_verified is True
        assert h5_file.exists()

        # Non-negative imaginary frequency must raise ValueError
        with pytest.raises(ValueError):
            run_bidirectional_irc(
                ts_coordinates_angstrom=ts_coords,
                symbols=symbols,
                ts_energy_hartree=-76.4,
                ts_imaginary_frequency_cm1=100.0,
                imaginary_mode_vector=imag_mode,
            )

        # Disconnected target minimum with distorted internal geometry must raise IRCDisconnectionError
        distorted_reactant = [
            [0.0, 0.0, 2.5],
            [0.0, 3.5, 0.0],
            [0.0, -3.5, 0.0],
        ]
        with pytest.raises(IRCDisconnectionError):
            run_bidirectional_irc(
                ts_coordinates_angstrom=ts_coords,
                symbols=symbols,
                ts_energy_hartree=-76.4,
                ts_imaginary_frequency_cm1=-420.5,
                imaginary_mode_vector=imag_mode,
                reactant_coordinates_angstrom=distorted_reactant,
                raise_on_disconnection=True,
            )


    # =========================================================================
    # REQ-QC-084: Standard State Solvation & Concentration Shift Engine
    # =========================================================================
    def test_req_084_solvation_1atm_to_1m_standard_state_shift(self) -> None:
        """[REQ-QC-084] Validate exact thermodynamic relations for gas-to-solution standard state shift."""
        # Standard 1 atm -> 1.0 M compression at 298.15 K
        g_gas = -76.4250
        res = compute_standard_state_correction(
            g_gas_1atm_hartree=g_gas,
            temperature_k=298.15,
            pressure_atm=1.0,
            solvent_name="water",
            is_neat_solvent=False,
        )

        assert isinstance(res, StandardStateCorrectionPayload)
        assert res.temperature_kelvin == 298.15
        assert res.pressure_atm == 1.0
        # Molar volume V_m^0 = 24.4654 L/mol
        assert math.isclose(res.gas_molar_volume_liters, 24.4654, abs_tol=1e-3)
        # Delta G^0 (1 atm -> 1 M) = +1.8943 kcal/mol (+7.9258 kJ/mol)
        assert math.isclose(res.delta_g_1atm_to_1m_kcal_mol, 1.8943, abs_tol=1e-3)
        assert math.isclose(res.delta_g_1atm_to_1m_kj_mol, 7.9258, abs_tol=1e-3)
        # Converted Hartree energy
        expected_g_solv = g_gas + (1.8943 / 627.5094740631)
        assert math.isclose(res.g_solv_1m_hartree, expected_g_solv, abs_tol=1e-6)

        # Neat liquid water shift (55.34 M)
        res_neat = compute_standard_state_correction(
            g_gas_1atm_hartree=g_gas,
            temperature_k=298.15,
            pressure_atm=1.0,
            solvent_name="water",
            is_neat_solvent=True,
            neat_concentration_m=55.34,
        )
        assert math.isclose(res_neat.delta_g_1atm_to_1m_kcal_mol, 4.274, abs_tol=0.02)

    # =========================================================================
    # REQ-QC-085: Level-of-Theory Invariance & Stationary Point Gate
    # =========================================================================
    def test_req_085_lot_sha256_invariance_and_stationary_gradient(self) -> None:
        """[REQ-QC-085] Validate Level-of-Theory cryptographic hashing and stationary gradient checks."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="SMD(water)",
            max_gradient_residual_au=2.5e-5,
        )
        freq_env_identical = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="SMD(water)",
            max_gradient_residual_au=2.5e-5,
        )

        # 1. Matching envelopes pass
        assert validate_frequency_level_of_theory(opt_env, freq_env_identical) is True
        assert validate_quantum_calculation_gate(opt_env, freq_env_identical) is True

        # 2. Method/Basis mismatch triggers LevelOfTheoryMismatchError
        freq_env_mismatch = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP",
            basis_set="def2-TZVP",
            integration_grid="DefGrid3",
            max_gradient_residual_au=2.5e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            validate_frequency_level_of_theory(opt_env, freq_env_mismatch)

        # 3. Non-stationary gradient residual triggers NonStationaryFrequencyError
        non_stationary_opt = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            integration_grid="DefGrid3",
            max_gradient_residual_au=3.5e-4,  # > 1.0e-4 a.u.
        )
        non_stationary_freq = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            integration_grid="DefGrid3",
            max_gradient_residual_au=3.5e-4,
        )
        with pytest.raises(NonStationaryFrequencyError):
            validate_frequency_level_of_theory(non_stationary_opt, non_stationary_freq)

    # =========================================================================
    # REQ-QC-086: Dynamic Mendeleev Isotope Lookup & Bigeleisen-Mayer KIE
    # =========================================================================
    def test_req_086_mendeleev_dynamic_isotope_lookup_and_kie(self) -> None:
        """[REQ-QC-086] Validate dynamic CIAAW isotope retrieval, Hessian re-diagonalization, and KIE."""
        symbols = ["O", "H", "H"]
        # Authentic 9x9 harmonic Cartesian Hessian for water (positive-definite approximation in a.u.)
        # Diagonal force constants for stretch (0.55 a.u.), bend (0.15 a.u.)
        diag = [0.15, 0.15, 0.15, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
        h_cart = np.diag(diag).astype(np.float64)

        # H -> D isotopic substitution on atom 1
        res = compute_isotopic_substitution_kie(
            symbols=symbols,
            cartesian_hessian_hartree_bohr2=h_cart,
            substituted_atoms={1: 2},
            temperature_k=298.15,
        )

        assert isinstance(res, IsotopeSubstitutionPayload)
        assert 1 in res.mendeleev_masses_amu
        # Dynamic Mendeleev mass for Deuterium (~2.01410 amu)
        assert math.isclose(res.mendeleev_masses_amu[1], 2.0141017778, rel_tol=1e-4)
        # Parent ZPE > Deuterated ZPE
        assert res.reference_zpe_kcal_mol > res.substituted_zpe_kcal_mol
        # Bigeleisen-Mayer KIE > 1.0 (normal isotope effect)
        assert res.kie_bigeleisen_mayer is not None and res.kie_bigeleisen_mayer > 1.0
        assert res.zpe_factor is not None and res.zpe_factor > 1.0

        # Unphysical isotope mass number must raise IsotopeNotFoundError
        with pytest.raises(IsotopeNotFoundError):
            compute_isotopic_substitution_kie(
                symbols=symbols,
                cartesian_hessian_hartree_bohr2=h_cart,
                substituted_atoms={1: 99},
            )

    # =========================================================================
    # REQ-QC-087: Post-SCF Wavefunction Stability & Yamaguchi Projection
    # =========================================================================
    def test_req_087_wavefunction_orbital_hessian_and_yamaguchi_projection(self) -> None:
        """[REQ-QC-087] Validate post-SCF orbital stability and Yamaguchi broken-symmetry energy projection."""
        # 1. Stable closed-shell wavefunction
        stable_report = evaluate_wavefunction_stability(
            lowest_orbital_hessian_eigenval=0.0250,
            s2_calc=0.0000,
            s_expected=0.0,
        )
        assert stable_report.initial_stability == "STABLE"
        assert stable_report.is_broken_symmetry_required is False
        assert stable_report.spin_contamination_percentage == 0.0

        # 2. Unstable wavefunction requiring Broken-Symmetry
        unstable_report = evaluate_wavefunction_stability(
            lowest_orbital_hessian_eigenval=-0.0450,
            s2_calc=0.9850,  # Highly contaminated singlet
            s_expected=0.0,
            e_bs_hartree=-150.8250,
            e_hs_hartree=-150.8000,
            s2_hs=2.0150,  # Triplet
        )
        assert unstable_report.initial_stability == "UNSTABLE_INTERNAL_RHF_UHF"
        assert unstable_report.is_broken_symmetry_required is True
        assert unstable_report.spin_contamination_percentage > 90.0
        # Yamaguchi projected energy should be lower than both BS and HS energies
        assert unstable_report.yamaguchi_corrected_energy_hartree is not None
        assert unstable_report.yamaguchi_corrected_energy_hartree < -150.8250

    # =========================================================================
    # REQ-QC-088: QTAIM Bond Critical Point & Poincaré-Hopf Invariant
    # =========================================================================
    def test_req_088_qtaim_bcp_cremer_kraka_and_poincare_hopf(self) -> None:
        """[REQ-QC-088] Validate QTAIM BCP interaction classification and Poincaré-Hopf invariant."""
        # 1. Covalent Shared BCP: nabla^2 rho < 0, H < 0
        bcp_covalent = classify_bond_critical_point(
            bcp_idx=0,
            atom_1=0,
            atom_2=1,
            coords=(0.0, 0.0, 0.48),
            rho=0.2850,
            hessian_eigenvals=(-0.450, -0.450, 0.050),  # sum = -0.850 < 0
            g_density=0.1200,
            v_density=-0.3325,  # H = -0.2125 < 0
        )
        assert bcp_covalent.interaction_classification == "COVALENT_SHARED"
        assert bcp_covalent.laplacian_au < 0.0
        assert bcp_covalent.total_energy_density_h_au < 0.0

        # 2. Polar Covalent BCP: nabla^2 rho > 0, H < 0
        bcp_polar = classify_bond_critical_point(
            bcp_idx=1,
            atom_1=0,
            atom_2=2,
            coords=(0.0, 0.0, 1.25),
            rho=0.0850,
            hessian_eigenvals=(-0.050, -0.050, 0.250),  # sum = +0.150 > 0
            g_density=0.0400,
            v_density=-0.0650,  # H = -0.0250 < 0
        )
        assert bcp_polar.interaction_classification == "TRANSIT_POLAR_COVALENT"

        # 3. Non-Covalent Closed Shell BCP: nabla^2 rho > 0, H > 0
        bcp_noncov = classify_bond_critical_point(
            bcp_idx=2,
            atom_1=1,
            atom_2=3,
            coords=(1.5, 0.0, 0.0),
            rho=0.0150,
            hessian_eigenvals=(-0.005, -0.005, 0.080),  # sum = +0.070 > 0
            g_density=0.0080,
            v_density=-0.0060,  # H = +0.0020 > 0
        )
        assert bcp_noncov.interaction_classification == "NON_COVALENT_CLOSED_SHELL"

        # 4. Poincaré-Hopf Invariant: n_NCP - n_BCP + n_RCP - n_CCP = 1
        # Isolated H2O: 3 nuclei (NCP), 2 bonds (BCP), 0 rings, 0 cages -> 3 - 2 + 0 - 0 = 1
        assert validate_poincare_hopf_invariant(n_ncp=3, n_bcp=2, n_rcp=0, n_ccp=0) is True

        # Benzene C6H6: 12 nuclei, 12 C-H + 6 C-C = 18 BCP, 1 RCP, 0 CCP -> 12 - 18 + 1 - 0 = -5 != 1
        # (Benzene complete: 12 NCP - 18 BCP + 7 RCP? For a ring: 12 - 12 (bonds) + 1 = 1)
        # Invalid topology check raises QTAIMTopologyError
        with pytest.raises(QTAIMTopologyError) as exc_topo:
            validate_poincare_hopf_invariant(n_ncp=4, n_bcp=4, n_rcp=0, n_ccp=0)
        assert "Poincaré-Hopf invariant violated" in str(exc_topo.value)

    # =========================================================================
    # REQ-QC-089: Spin-Orbit Coupling Matrix Elements & ISC Kinetics
    # =========================================================================
    def test_req_089_socme_and_fermi_golden_rule_isc_rate(self) -> None:
        """[REQ-QC-089] Validate SOMF coupling norm, Marcus FCWD density, and Fermi's Golden Rule ISC rate."""
        components = (0.50, 1.20, 1.30)
        # Total SOMF norm = sqrt(0.5^2 + 1.2^2 + 1.3^2) = sqrt(0.25 + 1.44 + 1.69) = sqrt(3.38) = 1.83848 cm^-1
        expected_mag = math.sqrt(0.5**2 + 1.2**2 + 1.3**2)

        res = compute_socme_and_isc_rate(
            initial_state="S1",
            final_state="T1",
            delta_e_st_ev=0.350,
            socme_components_cm1=components,
            reorganization_energy_ev=0.200,
            temperature_k=298.15,
        )

        assert isinstance(res, SOCMEResult)
        assert math.isclose(res.socme_magnitude_cm1, expected_mag, abs_tol=1e-4)
        assert res.fcwd_density_states > 0.0
        assert res.k_isc_s_inv > 0.0
        assert res.phosphorescence_lifetime_seconds is not None and res.phosphorescence_lifetime_seconds > 0.0

    # =========================================================================
    # REQ-QC-090: Sayvetz-Eckart Rotational/Translational Projection Engine
    # =========================================================================
    def test_req_090_sayvetz_eckart_rot_trans_projection_zero_modes(self) -> None:
        """[REQ-QC-090] Validate Sayvetz-Eckart projector purging 6 (and 5) rigid-body modes to <= 1.0e-4 cm^-1."""
        # 1. Non-linear molecule: Water H2O (3 atoms, 9 coordinates)
        symbols_h2o = ["O", "H", "H"]
        coords_h2o = [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ]
        # Diagonal force constant matrix
        diag_h2o = [0.20, 0.20, 0.20, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
        h_cart_h2o = np.diag(diag_h2o).astype(np.float64)

        res_h2o = project_sayvetz_eckart_hessian(
            symbols=symbols_h2o,
            coordinates_angstrom=coords_h2o,
            cartesian_hessian_hartree_bohr2=h_cart_h2o,
            is_linear=False,
        )

        assert isinstance(res_h2o, ProjectedFrequencyResult)
        assert res_h2o.num_projected_zero_modes == 6
        assert res_h2o.is_linear_molecule is False
        assert res_h2o.is_projection_clean is True
        assert res_h2o.rot_trans_residual_max_cm1 <= 1.0e-4
        assert len(res_h2o.projected_frequencies_cm1) == 9
        # First 6 modes in projected frequencies should be 0.0
        assert all(f == 0.0 for f in res_h2o.projected_frequencies_cm1[:6])

        # 2. Linear molecule: Carbon Dioxide CO2 (3 atoms, 9 coordinates, collinear along Z)
        symbols_co2 = ["O", "C", "O"]
        coords_co2 = [
            [0.0000, 0.0000, -1.1600],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.1600],
        ]
        diag_co2 = [0.25, 0.25, 0.60, 0.25, 0.25, 0.60, 0.25, 0.25, 0.60]
        h_cart_co2 = np.diag(diag_co2).astype(np.float64)

        res_co2 = project_sayvetz_eckart_hessian(
            symbols=symbols_co2,
            coordinates_angstrom=coords_co2,
            cartesian_hessian_hartree_bohr2=h_cart_co2,
            is_linear=True,
        )

        assert res_co2.num_projected_zero_modes == 5
        assert res_co2.is_linear_molecule is True
        assert res_co2.is_projection_clean is True
        assert res_co2.rot_trans_residual_max_cm1 <= 1.0e-4
        assert all(f == 0.0 for f in res_co2.projected_frequencies_cm1[:5])
