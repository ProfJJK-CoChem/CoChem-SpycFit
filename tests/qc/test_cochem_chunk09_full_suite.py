"""
CoChem Comprehensive Unit Test Suite for SRS Section 4.9 (Chunk 09, Prompts 01 to 22).

Zero-Mock Mandate Compliance:
- All atomic/isotopic masses are dynamically resolved via mendeleev (IUPAC CIAAW).
- Physical constants conform to CODATA 2022.
- All algorithms run physically against real mathematical & physical constraints.
"""

import math
import numpy as np
import pytest

from cochem.core.cochem_elements import (
    get_atomic_number,
    get_ciaaw_atomic_mass,
    get_ciaaw_isotope_mass,
    validate_charge_and_multiplicity,
    SpinParityError,
)
from cochem.qc.cochem_method_registry import (
    classify_functional,
    compute_canonical_lot_hash,
    compute_lot_fingerprint,
    FunctionalType,
    LevelOfTheory,
)
from cochem.provenance.cochem_method_validator import (
    verify_level_of_theory_invariance,
    verify_lot_invariance,
    verify_stationary_point,
    NonStationaryFrequencyError,
)
from cochem.core.cochem_orca_builder import (
    generate_orca_stability_deck,
    generate_orca_irc_deck,
    generate_orca_socme_deck,
    generate_orca_marcus_decks,
)
from cochem.core.cochem_gaussian_builder import (
    generate_gaussian_deck,
    GaussianRouteCard,
)
from cochem.qc.cochem_solvation import (
    compute_standard_state_correction,
    partition_solvation_free_energy,
)
from cochem.qc.cochem_frequency import (
    project_sayvetz_eckart_hessian,
)
from cochem.qc.cochem_thermo import (
    compute_statistical_thermodynamics,
    compute_translational_partition_function,
    compute_rotational_partition_function,
    compute_vibrational_thermodynamics,
)
from cochem.qc.cochem_isotopes import (
    compute_isotopic_substitution_kie,
    compute_eie_equilibrium_isotope_effect,
)
from cochem.qc.cochem_stability import (
    evaluate_wavefunction_stability,
    compute_yamaguchi_projection,
)
from cochem.qc.cochem_marcus import (
    compute_marcus_reorganization,
    compute_outer_sphere_reorganization,
)
from cochem.qc.cochem_irc import (
    compute_kabsch_rmsd,
    run_bidirectional_irc,
    IRCConnectivityError,
)
from cochem.qc.cochem_socme import (
    compute_socme_and_isc_rate,
)
from cochem.qc.cochem_qtaim import (
    classify_bond_critical_point,
    validate_poincare_hopf_invariant,
    PoincareHopfViolationError,
)
from cochem.topos.cochem_topos_crusher import (
    cluster_conformers_by_qtaim_topology,
    ToposConformerGraph,
)
from cochem.qc.cochem_qc_pipeline import (
    execute_master_qc_pipeline,
    QCPipelineTask,
)
from cochem.reporting.cochem_warning_engine import (
    format_spin_contamination_warning,
    format_wavefunction_instability_warning,
    format_gradient_threshold_warning,
    generate_warning_summary,
)
from cochem.reporting.cochem_report_generator import (
    AcademicReportSpec,
    generate_academic_report,
    generate_marcus_table,
    generate_kie_si_table,
    generate_qtaim_table,
)
from cochem.gui.cochem_gui import (
    CoChemMasterQCController,
)
from cochem.gui.widgets.irc_path_viewer import (
    IRCPathViewerWidget,
)
from cochem.gui.widgets.isotope_labeler import (
    IsotopeLabelerWidget,
)
from cochem.gui.widgets.qtaim_graph_viewer import (
    QTAIMGraphViewerWidget,
)
from cochem.qc.models import LevelOfTheoryEnvelope, BondCriticalPointDescriptor


# ==============================================================================
# Prompt 01: Elements & Spin-Parity Verification (REQ-QC-081, 086, 090)
# ==============================================================================
def test_prompt_01_dynamic_mendeleev_and_spin_parity():
    # Dynamic mass lookup
    h_mass = get_ciaaw_atomic_mass("H")
    d_mass = get_ciaaw_isotope_mass("H", 2)
    c_mass = get_ciaaw_atomic_mass("C")
    c13_mass = get_ciaaw_isotope_mass("C", 13)

    assert 1.007 < h_mass < 1.009
    assert 2.013 < d_mass < 2.015
    assert 12.010 < c_mass < 12.012
    assert 13.003 < c13_mass < 13.004

    # Spin-parity: H2O neutral singlet (10 e-, M=1) -> valid
    water_res = validate_charge_and_multiplicity(symbols=["O", "H", "H"], charge=0, multiplicity=1)
    assert water_res.is_valid is True
    assert water_res.total_electrons == 10

    # Radical OH neutral doublet (9 e-, M=2) -> valid
    oh_res = validate_charge_and_multiplicity(symbols=["O", "H"], charge=0, multiplicity=2)
    assert oh_res.is_valid is True
    assert oh_res.total_electrons == 9

    # Parity violation: H2O neutral doublet (10 e-, M=2) -> must raise SpinParityError
    with pytest.raises(SpinParityError):
        validate_charge_and_multiplicity(symbols=["O", "H", "H"], charge=0, multiplicity=2)


# ==============================================================================
# Prompt 02: Level of Theory Registry & Canonical Hashing (REQ-QC-085)
# ==============================================================================
def test_prompt_02_lot_classification_and_canonical_fingerprint():
    # Functional classification
    assert classify_functional("wB97X-D4") == FunctionalType.RANGE_SEPARATED_HYBRID
    assert classify_functional("B3LYP") == FunctionalType.GLOBAL_HYBRID
    assert classify_functional("DLPNO-CCSD(T)") == FunctionalType.POST_HF

    # Fingerprinting invariance
    lot1 = LevelOfTheory(hamiltonian="wB97X-D4", basis_set="def2-TZVP", dispersion="D4", grid="defgrid3")
    lot2 = LevelOfTheory(hamiltonian="wb97x-d4", basis_set="DEF2-TZVP", dispersion="d4", grid="defgrid3")

    fp1 = compute_lot_fingerprint(lot1)
    fp2 = compute_lot_fingerprint(lot2)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256


# ==============================================================================
# Prompt 03: Method Validator & Stationary Point Gradient Gate (REQ-QC-085)
# ==============================================================================
def test_prompt_03_lot_invariance_and_gradient_gate():
    env_opt = LevelOfTheoryEnvelope(
        hamiltonian_method="wB97X-D4",
        basis_set="def2-TZVP",
        max_gradient_residual_au=2.5e-5,
    )
    env_freq_match = LevelOfTheoryEnvelope(
        hamiltonian_method="wb97x-d4",
        basis_set="DEF2-TZVP",
        max_gradient_residual_au=2.5e-5,
    )
    assert verify_lot_invariance(env_opt, env_freq_match) is True

    # Gradient gate with scalar
    assert verify_stationary_point(5.0e-5, threshold=1.0e-4) is True

    # Gradient gate with Cartesian array
    grad_vector = np.array([[1.0e-5, -2.0e-5, 3.0e-5], [0.0, 1.2e-5, -4.5e-5]])
    assert verify_stationary_point(grad_vector, threshold=1.0e-4) is True

    # Violation: gradient exceeding threshold
    with pytest.raises(NonStationaryFrequencyError):
        verify_stationary_point(1.5e-4, threshold=1.0e-4)


# ==============================================================================
# Prompt 04: ORCA Deck Synthesizer (REQ-QC-082, 083, 087, 089)
# ==============================================================================
def test_prompt_04_orca_builder_specialized_decks():
    symbols = ["C", "H", "H", "H", "H"]
    coords = [(0.0, 0.0, 0.0), (0.6, 0.6, 0.6), (-0.6, -0.6, 0.6), (-0.6, 0.6, -0.6), (0.6, -0.6, -0.6)]

    # 1. Stability deck
    stab_deck = generate_orca_stability_deck(symbols, coords, method="wB97X-D4", broken_symmetry=(1, 1))
    assert "Stability Opt" in stab_deck
    assert "Brokensym 1, 1" in stab_deck

    # 2. IRC deck
    irc_deck = generate_orca_irc_deck(symbols, coords, method="wB97X-D4", direction="Both", max_steps=40)
    assert "%irc" in irc_deck
    assert "Direction Both" in irc_deck
    assert "MaxSteps 40" in irc_deck

    # 3. SOCME deck
    soc_deck = generate_orca_socme_deck(symbols, coords, method="B3LYP", soc_flags=(1, 2, 3))
    assert "%tddft" in soc_deck
    assert "SOCFlags 1, 2, 3" in soc_deck

    # 4. Marcus decks
    m_decks = generate_orca_marcus_decks(symbols, coords, coords, method="wB97X-D4")
    assert set(m_decks.keys()) == {"e0_r0", "ep_r0", "ep_rp", "e0_rp"}
    assert "0 1" in m_decks["e0_r0"]
    assert "1 2" in m_decks["ep_r0"]


# ==============================================================================
# Prompt 05: Gaussian Deck Synthesizer (REQ-QC-085, REQ-QC-087)
# ==============================================================================
def test_prompt_05_gaussian_builder():
    symbols = ["C", "H", "O"]
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.2, 0.0)]

    deck = generate_gaussian_deck(
        symbols=symbols,
        coordinates_angstrom=coords,
        method="B3LYP",
        basis="def2-TZVP",
        charge=-1,
        multiplicity=1,
        stability="Stable=Opt",
        solvation_model="SMD",
        solvent_name="water",
        read_isotopes=True,
        isotopic_substitutions={1: 2}, # H -> Deuterium
    )

    assert "#p B3LYP/def2-TZVP" in deck
    assert "Stable=Opt" in deck
    assert "Integral=(Grid=UltraFine)" in deck
    assert "SCRF=(SMD,Solvent=water)" in deck
    assert "-1 1" in deck
    assert "298.15 1.00 1.0000" in deck
    # Check that deuterium mass (~2.014) is written
    assert "2.014102" in deck


# ==============================================================================
# Prompt 06: Solvation Concentration Shift (REQ-QC-084)
# ==============================================================================
def test_prompt_06_solvation_standard_state_correction():
    # Test exact 1 atm -> 1.0 M standard state compression shift at 298.15 K
    g_gas_hartree = -76.400000
    corr = compute_standard_state_correction(
        g_gas_1atm_hartree=g_gas_hartree,
        temperature_k=298.15,
        pressure_atm=1.0,
        solvent_name="water",
    )

    # Known analytical value at 298.15 K: +1.8943 kcal/mol (+7.9258 kJ/mol)
    assert math.isclose(corr.delta_g_1atm_to_1m_kcal_mol, 1.8943, rel_tol=1e-3)
    assert math.isclose(corr.delta_g_1atm_to_1m_kj_mol, 7.9258, rel_tol=1e-3)
    assert corr.g_solv_1m_hartree > g_gas_hartree

    # Solvation energy partitioning
    part = partition_solvation_free_energy(
        delta_g_solv_kcal_mol=-8.5,
        delta_g_electrostatic_kcal_mol=-6.2,
        solvent_name="water",
    )
    assert math.isclose(part.delta_g_cds_kcal_mol, -2.3, abs_tol=1e-4)


# ==============================================================================
# Prompt 07: Sayvetz-Eckart Hessian Projection (REQ-QC-090)
# ==============================================================================
def test_prompt_07_sayvetz_eckart_projection():
    symbols = ["O", "H", "H"]
    coords = [
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ]
    # Simple synthetic 9x9 harmonic Cartesian Hessian for water
    h_cart = np.diag([0.05, 0.20, 0.35, 0.10, 0.25, 0.08, 0.10, 0.25, 0.08])

    res = project_sayvetz_eckart_hessian(
        symbols=symbols,
        coordinates_angstrom=coords,
        cartesian_hessian_hartree_bohr2=h_cart,
        is_linear=False,
    )

    assert len(res.projected_frequencies_cm1) == 9
    assert res.num_projected_zero_modes == 6
    # Lowest 6 projected modes should be cleanly zeroed
    assert all(math.isclose(f, 0.0, abs_tol=1e-4) for f in res.projected_frequencies_cm1[:6])
    assert all(f > 0.0 for f in res.projected_frequencies_cm1[6:])


# ==============================================================================
# Prompt 08: Statistical Thermodynamics & Quasi-RRHO (REQ-QC-084, REQ-QC-090)
# ==============================================================================
def test_prompt_08_statistical_thermodynamics():
    symbols = ["O", "H", "H"]
    freqs = [1600.0, 3650.0, 3750.0]
    moments = [0.61, 1.15, 1.76] # amu * Å^2
    e_elec = -76.400000

    thermo = compute_statistical_thermodynamics(
        symbols=symbols,
        frequencies_cm1=freqs,
        moments_of_inertia_amu_ang2=moments,
        electronic_energy_hartree=e_elec,
        temperature_k=298.15,
        pressure_atm=1.0,
        symmetry_number=2,
    )

    assert thermo.zpe_hartree > 0.0
    assert thermo.thermal_energy_u_hartree > thermo.electronic_energy_hartree
    assert thermo.enthalpy_h_hartree > thermo.thermal_energy_u_hartree
    assert thermo.entropy_total_cal_mol_k > 30.0 # Standard gas entropy ~ 45 cal/(mol K)
    assert thermo.gibbs_free_energy_gas_hartree < thermo.enthalpy_h_hartree
    assert thermo.partition_functions.q_total > 1.0


# ==============================================================================
# Prompt 09: Dynamic Mendeleev KIE / EIE Engine (REQ-QC-086)
# ==============================================================================
def test_prompt_09_bigeleisen_mayer_kie():
    symbols = ["C", "H", "H", "H", "H"]
    # 15x15 diagonal test Hessian
    h_cart = np.diag([0.15] * 15)

    # 1H -> 2H (Deuterium) substitution on atom 1
    kie_res = compute_isotopic_substitution_kie(
        symbols=symbols,
        cartesian_hessian_hartree_bohr2=h_cart,
        substituted_atoms={1: 2},
        temperature_k=298.15,
    )

    assert kie_res.reference_zpe_kcal_mol > kie_res.substituted_zpe_kcal_mol
    assert kie_res.kie_bigeleisen_mayer > 1.0
    assert 2 in kie_res.substituted_atoms.values()

    # EIE check
    eie_res = compute_eie_equilibrium_isotope_effect(kie_res, kie_res, temperature_k=298.15)
    assert math.isclose(eie_res.eie_total, 1.0, abs_tol=1e-4)


# ==============================================================================
# Prompt 10: Wavefunction Stability & Yamaguchi Projection (REQ-QC-087)
# ==============================================================================
def test_prompt_10_stability_and_yamaguchi():
    # Stable closed-shell singlet
    rep_stable = evaluate_wavefunction_stability(
        lowest_orbital_hessian_eigenval=0.045,
        s2_calc=0.000,
        s_expected=0.0,
    )
    assert rep_stable.initial_stability == "STABLE"
    assert rep_stable.is_broken_symmetry_required is False

    # Unstable open-shell broken symmetry singlet
    rep_unstable = evaluate_wavefunction_stability(
        lowest_orbital_hessian_eigenval=-0.012,
        s2_calc=0.985,
        s_expected=0.0,
        e_bs_hartree=-150.250000,
        e_hs_hartree=-150.240000,
        s2_hs=2.015,
    )
    assert rep_unstable.initial_stability == "UNSTABLE_INTERNAL_RHF_UHF"
    assert rep_unstable.is_broken_symmetry_required is True
    assert rep_unstable.yamaguchi_corrected_energy_hartree is not None
    assert rep_unstable.yamaguchi_corrected_energy_hartree < -150.250000


# ==============================================================================
# Prompt 11: Marcus Reorganization & Kinetics (REQ-QC-082)
# ==============================================================================
def test_prompt_11_marcus_reorganization():
    e0_r0 = -100.500000
    ep_r0 = -100.200000
    ep_rp = -100.220000
    e0_rp = -100.470000

    marcus = compute_marcus_reorganization(
        e_neutral_at_neutral_geom_hartree=e0_r0,
        e_charged_at_neutral_geom_hartree=ep_r0,
        e_charged_at_charged_geom_hartree=ep_rp,
        e_neutral_at_charged_geom_hartree=e0_rp,
        lambda_outer_ev=0.15,
        electronic_coupling_ev=0.01,
        temperature_k=298.15,
    )

    assert marcus.lambda_1_ev > 0.0
    assert marcus.lambda_2_ev > 0.0
    assert marcus.lambda_inner_ev > 0.0
    assert marcus.total_reorganization_energy_ev > marcus.lambda_inner_ev
    assert marcus.marcus_activation_barrier_ev > 0.0
    assert marcus.electron_transfer_rate_s_inv is not None
    assert marcus.electron_transfer_rate_s_inv > 0.0


# ==============================================================================
# Prompt 12: Bidirectional IRC Tracing & Kabsch RMSD (REQ-QC-083)
# ==============================================================================
def test_prompt_12_irc_and_kabsch_rmsd():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = p + 0.05 # small translation
    rmsd = compute_kabsch_rmsd(p, q)
    # Pure translation is eliminated by Kabsch alignment
    assert math.isclose(rmsd, 0.0, abs_tol=1e-6)

    # Test full IRC integration
    symbols = ["H", "H", "H"]
    ts_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    mode = [[0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1]]

    irc_res = run_bidirectional_irc(
        ts_coordinates_angstrom=ts_coords,
        symbols=symbols,
        ts_energy_hartree=-1.500000,
        ts_imaginary_frequency_cm1=-650.0,
        imaginary_mode_vector=mode,
        max_steps_per_direction=5,
    )

    assert irc_res.forward_steps == 5
    assert irc_res.reverse_steps == 5
    assert len(irc_res.path_energies_hartree) == 11
    assert irc_res.connectivity_verified is True


# ==============================================================================
# Prompt 13: SOCME & ISC Kinetics (REQ-QC-089)
# ==============================================================================
def test_prompt_13_socme_and_isc_rate():
    soc_res = compute_socme_and_isc_rate(
        initial_state="S1",
        final_state="T1",
        delta_e_st_ev=0.25,
        socme_components_cm1=(1.5, -2.0, 3.2),
        reorganization_energy_ev=0.20,
        temperature_k=298.15,
    )

    assert math.isclose(soc_res.socme_magnitude_cm1, math.sqrt(1.5**2 + (-2.0)**2 + 3.2**2), abs_tol=1e-4)
    assert soc_res.k_isc_s_inv > 0.0
    assert soc_res.phosphorescence_lifetime_seconds is not None


# ==============================================================================
# Prompt 14: QTAIM Critical Points & Poincaré-Hopf (REQ-QC-088)
# ==============================================================================
def test_prompt_14_qtaim_classification_and_poincare_hopf():
    # Covalent C-C bond critical point
    bcp = classify_bond_critical_point(
        bcp_idx=1,
        atom_1=0,
        atom_2=1,
        coords=(0.0, 0.0, 0.0),
        rho=0.285,
        hessian_eigenvals=(-0.65, -0.62, 0.85),
        g_density=0.085,
        v_density=-0.415,
    )
    assert bcp.laplacian_au < 0.0
    assert bcp.total_energy_density_h_au < 0.0
    assert bcp.interaction_classification == "COVALENT_SHARED"

    # Poincaré-Hopf check: Water (3 NCP, 2 BCP, 0 RCP, 0 CCP -> 3 - 2 + 0 - 0 = 1)
    assert validate_poincare_hopf_invariant(n_ncp=3, n_bcp=2, n_rcp=0, n_ccp=0) is True

    # Benzene (12 NCP, 12 BCP, 1 RCP, 0 CCP -> 12 - 12 + 1 - 0 = 1)
    assert validate_poincare_hopf_invariant(n_ncp=12, n_bcp=12, n_rcp=1, n_ccp=0) is True

    # Invariant failure
    with pytest.raises(PoincareHopfViolationError):
        validate_poincare_hopf_invariant(n_ncp=3, n_bcp=1, n_rcp=0, n_ccp=0)


# ==============================================================================
# Prompt 15: Conformer Crusher & QTAIM Graph Clustering (REQ-QC-088)
# ==============================================================================
def test_prompt_15_conformer_qtaim_clustering():
    g1 = ToposConformerGraph(
        conformer_id="conf_01",
        energy_hartree=-150.10,
        symbols=["C", "C"],
        coordinates_angstrom=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        ncp_count=2,
        bcp_count=1,
        bcp_bonds=[(0, 1)],
        graph_hash="hash_linear",
    )
    g2 = ToposConformerGraph(
        conformer_id="conf_02",
        energy_hartree=-150.08,
        symbols=["C", "C"],
        coordinates_angstrom=[(0.0, 0.0, 0.0), (1.52, 0.0, 0.0)],
        ncp_count=2,
        bcp_count=1,
        bcp_bonds=[(0, 1)],
        graph_hash="hash_linear",
    )
    g3 = ToposConformerGraph(
        conformer_id="conf_03",
        energy_hartree=-149.95,
        symbols=["C", "C"],
        coordinates_angstrom=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)],
        ncp_count=2,
        bcp_count=2,
        bcp_bonds=[(0, 1), (1, 0)],
        graph_hash="hash_cyclic",
    )

    clusters = cluster_conformers_by_qtaim_topology([g1, g2, g3])
    assert len(clusters) == 2
    assert clusters[0].representative_conformer_id == "conf_01"


# ==============================================================================
# Prompt 16: Master QC Pipeline Execution (REQ-QC-081..085)
# ==============================================================================
def test_prompt_16_master_qc_pipeline_execution():
    task = QCPipelineTask(
        task_id="task_001",
        workflow_type="MARCUS",
        symbols=["C", "H", "H", "H", "H"],
        coordinates_angstrom=[(0.0, 0.0, 0.0), (0.6, 0.6, 0.6), (-0.6, -0.6, 0.6), (-0.6, 0.6, -0.6), (0.6, -0.6, -0.6)],
        charge=0,
        multiplicity=1,
        gradient_residual_au=3.5e-5,
        orbital_hessian_lowest_eigenvalue=0.035,
        marcus_params={
            "e0_r0": -40.500000,
            "ep_r0": -40.200000,
            "ep_rp": -40.220000,
            "e0_rp": -40.480000,
            "lambda_outer": 0.10,
        },
    )

    report = execute_master_qc_pipeline(task)
    assert report.status == "SUCCESS"
    assert report.spin_parity_passed is True
    assert report.stationary_gradient_passed is True
    assert report.stability_verified is True
    assert "marcus" in report.workflow_results


# ==============================================================================
# Prompt 17: Diagnostic Warning Engine (REQ-QC-081, REQ-QC-087)
# ==============================================================================
def test_prompt_17_warning_engine():
    # Spin contamination
    w_spin = format_spin_contamination_warning(s2_calc=0.85, s_expected=0.0)
    assert w_spin is not None
    assert w_spin.category == "SPIN_CONTAMINATION"
    assert "Severe Spin Contamination" in w_spin.markdown_banner

    # Wavefunction instability
    w_stab = format_wavefunction_instability_warning(lowest_eigenvalue=-0.005)
    assert w_stab is not None
    assert w_stab.severity == "CRITICAL"

    # Borderline gradient
    w_grad = format_gradient_threshold_warning(max_gradient_au=8.5e-5)
    assert w_grad is not None
    assert w_grad.category == "GRADIENT_BORDERLINE"

    summary = generate_warning_summary([w_spin, w_stab, w_grad])
    assert summary.total_warnings == 3
    assert summary.critical_count >= 1


# ==============================================================================
# Prompt 18: Academic Report Generator (REQ-QC-082..089)
# ==============================================================================
def test_prompt_18_report_generator():
    marcus_res = compute_marcus_reorganization(
        e_neutral_at_neutral_geom_hartree=-100.5,
        e_charged_at_neutral_geom_hartree=-100.2,
        e_charged_at_charged_geom_hartree=-100.22,
        e_neutral_at_charged_geom_hartree=-100.47,
        lambda_outer_ev=0.10,
    )

    spec = AcademicReportSpec(
        title="Comprehensive Evaluation",
        authors="CoChem Team",
        system_name="Substituted Benzene",
        include_marcus=True,
        include_solvation_notes=True,
    )

    report = generate_academic_report(spec=spec, marcus_data=marcus_res)
    assert "Comprehensive Evaluation" in report.markdown_content
    assert "Marcus Electron Transfer" in report.markdown_content
    assert "\\begin{table}" in report.latex_content


# ==============================================================================
# Prompt 19: Main QC GUI Controller (REQ-QC-081)
# ==============================================================================
def test_prompt_19_gui_controller():
    gui = CoChemMasterQCController(symbols=["O", "H", "H"], coordinates=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    res = gui.update_spin_and_charge(charge=0, multiplicity=1)
    assert res.is_valid is True
    assert gui.get_state().is_spin_parity_valid is True

    # Parity violation
    with pytest.raises(SpinParityError):
        gui.update_spin_and_charge(charge=0, multiplicity=2)
    assert gui.get_state().is_spin_parity_valid is False


# ==============================================================================
# Prompt 20: IRC Path Viewer Widget (REQ-QC-083)
# ==============================================================================
def test_prompt_20_irc_path_viewer_widget():
    symbols = ["H", "H", "H"]
    ts_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    mode = [[0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1]]

    irc_res = run_bidirectional_irc(
        ts_coordinates_angstrom=ts_coords,
        symbols=symbols,
        ts_energy_hartree=-1.500000,
        ts_imaginary_frequency_cm1=-650.0,
        imaginary_mode_vector=mode,
        max_steps_per_direction=4,
    )

    widget = IRCPathViewerWidget(irc_res)
    state = widget.get_state()
    assert state.total_frames == 9
    assert state.status_badge == "CONNECTED"

    new_state = widget.set_frame(2)
    assert new_state.current_frame_index == 2


# ==============================================================================
# Prompt 21: Isotope Labeler Widget (REQ-QC-086)
# ==============================================================================
def test_prompt_21_isotope_labeler_widget():
    symbols = ["C", "H", "H", "H", "H"]
    h_cart = np.diag([0.15] * 15)

    widget = IsotopeLabelerWidget(symbols=symbols, cartesian_hessian=h_cart)
    # Set atom 1 to Deuterium (2H)
    state = widget.set_isotope(1, 2)
    assert 1 in state.substituted_atoms
    assert state.predicted_kie > 1.0

    # Clear isotope
    state_cleared = widget.clear_all()
    assert len(state_cleared.substituted_atoms) == 0


# ==============================================================================
# Prompt 22: QTAIM Graph Viewer Widget (REQ-QC-088)
# ==============================================================================
def test_prompt_22_qtaim_graph_viewer_widget():
    symbols = ["C", "C"]
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
    bcp = classify_bond_critical_point(
        bcp_idx=1,
        atom_1=0,
        atom_2=1,
        coords=(0.75, 0.0, 0.0),
        rho=0.28,
        hessian_eigenvals=(-0.6, -0.6, 0.8),
        g_density=0.08,
        v_density=-0.40,
    )

    widget = QTAIMGraphViewerWidget(symbols=symbols, coordinates=coords, bcps=[bcp])
    state = widget.get_state()
    assert state.n_ncp == 2
    assert state.n_bcp == 1
    assert state.is_poincare_hopf_valid is True
    assert len(state.critical_points_summary) == 3 # 2 NCP + 1 BCP

    sel_state = widget.select_bcp(1)
    assert sel_state.selected_bcp_details is not None
    assert sel_state.selected_bcp_details.bcp_index == 1


# ==============================================================================
# Adversarial Audit Invariant Tests
# ==============================================================================
def test_transition_state_sayvetz_eckart_imaginary_mode_preservation():
    """
    [REQ-QC-090] Verify that a Transition State (with lambda_TS < 0) correctly preserves
    the unique imaginary frequency and does NOT conflate it with rotational/translational zero modes.
    """
    symbols = ["H", "H", "H"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    # Construct a synthetic TS Hessian where mode 0 is strongly imaginary (-0.05 a.u.)
    # and modes 1..6 are rigid-body rotational/translational modes (< 1e-6 a.u.)
    h_ts = np.diag([-0.05, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])

    proj_res = project_sayvetz_eckart_hessian(
        symbols=symbols,
        coordinates_angstrom=coords,
        cartesian_hessian_hartree_bohr2=h_ts,
        is_linear=True,
    )

    # 1. Verify that the imaginary frequency is retained as negative
    imag_freqs = [f for f in proj_res.projected_frequencies_cm1 if f < -10.0]
    assert len(imag_freqs) >= 1, "Transition state imaginary frequency must NOT be zeroed out!"
    assert proj_res.is_projection_clean is True
    assert proj_res.rot_trans_residual_max_cm1 <= 1.0e-4


def test_gui_controller_uninitialized_and_load_flow():
    """
    [REQ-QC-081] Verify GUI Master Controller properly handles uninitialized empty symbols
    and loads molecules dynamically with load_molecule().
    """
    from cochem.gui.cochem_gui import CoChemMasterQCController

    ctrl = CoChemMasterQCController()
    state = ctrl.get_state()
    assert state.active_molecule_symbols == []

    # Updating spin/charge on empty symbols raises clear error and records invalid state
    with pytest.raises(ValueError, match="No molecular structure loaded"):
        ctrl.update_spin_and_charge(charge=0, multiplicity=1)

    # Load molecule and verify successful electronic state validation
    loaded_state = ctrl.load_molecule(symbols=["H", "H", "O"])
    assert loaded_state.is_spin_parity_valid is True
    assert len(loaded_state.active_molecule_symbols) == 3

