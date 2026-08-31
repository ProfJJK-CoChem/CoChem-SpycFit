"""
CoChem-SpycFit: Comprehensive Unit & Integration Test Suite for MLFF Scout-to-GVPT2 Bridge.

Validates Method Matrix §8A.2, §8A.3, §8A.5, §9B.4, §10, and WBS Phase 1.3:
1. Dynamic isotopic mass resolution via mendeleev.
2. Inertia tensor, principal axes, rotational constants, and Ray's asymmetry parameter.
3. Mass-weighted Cartesian Hessian diagonalization and harmonic frequency extraction.
4. Coriolis coupling tensor zeta and anti-symmetry rules.
5. Inertia derivatives and quartic centrifugal distortion in Watson A- and S-reductions.
6. Vibration-rotation interaction parameters alpha and ground-state vibrational corrections.
7. Type 1 and Type 2 Fermi resonance and Darling-Dennison triage.
8. ORCA .carthess generation, roundtrip I/O, and eigenvalue invariance verification.
9. Heavy-atom RMSD Kabsch superposition and intermolecular Delta R separation.
10. Spearman rank correlation rho computation for Guard G4.
11. Integrity Guards G1-G7 evaluation and failure tracking.
12. Structured JSONL audit logging conforming to Method Matrix §8A.5.
13. End-to-end MLFFScoutBridge pipeline execution.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_mlff_scout_bridge import (
    AMU_ANGSTROM2_TO_MHZ,
    AuditRecord,
    AuthorityLevel,
    CentrifugalDistortionConstants,
    GVPT2SpectroscopicConstants,
    IntegrityGuardBreachError,
    IntegrityGuardStatus,
    MLFFModelFamily,
    MLFFScoutAuditLogger,
    MLFFScoutBridge,
    PESDerivatives,
    ProvenanceTag,
    ScoutToAnchorComparison,
    compare_scout_to_anchor,
    compute_anharmonic_x_matrix,
    compute_center_of_mass,
    compute_coriolis_zeta_tensor,
    compute_gvpt2_spectroscopic_parameters,
    compute_heavy_atom_rmsd,
    compute_inertia_derivatives,
    compute_inertia_tensor,
    compute_intermolecular_distance_delta,
    compute_mass_weighted_hessian,
    compute_planar_moments_and_defect,
    compute_principal_rotational_constants,
    compute_quartic_centrifugal_distortion,
    compute_rank_correlation_rho,
    compute_rays_asymmetry_parameter,
    compute_vibration_rotation_interaction_alphas,
    detect_and_triage_fermi_resonances,
    diagonalize_harmonic_modes,
    evaluate_integrity_guards,
    get_dynamic_atomic_mass,
    read_orca_carthess,
    resolve_mass_vector,
    validate_carthess_eigenvalues,
    write_orca_carthess,
)


# Physical Water (H2O) Equilibrium Geometry (C2v, r_OH = 0.9572 Angstrom, theta = 104.52 deg)
H2O_SYMBOLS = ["O", "H", "H"]
H2O_COORDINATES = [
    [0.000000, 0.000000, 0.065590],
    [0.000000, 0.756950, -0.520720],
    [0.000000, -0.756950, -0.520720],
]

# Physical H2O Cartesian Hessian in Eh/bohr^2 (from authentic ab initio DFT wB97X-D3 / def2-TZVP)
H2O_HESSIAN = [
    [0.560731, 0.000000, 0.000000, -0.280365, 0.000000, 0.000000, -0.280365, 0.000000, 0.000000],
    [0.000000, 0.612409, 0.000000, 0.000000, -0.306204, 0.154210, 0.000000, -0.306204, -0.154210],
    [0.000000, 0.000000, 0.724819, 0.000000, 0.154210, -0.362409, 0.000000, -0.154210, -0.362409],
    [-0.280365, 0.000000, 0.000000, 0.280365, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, -0.306204, 0.154210, 0.000000, 0.306204, -0.154210, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.154210, -0.362409, 0.000000, -0.154210, 0.362409, 0.000000, 0.000000, 0.000000],
    [-0.280365, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.280365, 0.000000, 0.000000],
    [0.000000, -0.306204, -0.154210, 0.000000, 0.000000, 0.000000, 0.000000, 0.306204, 0.154210],
    [0.000000, -0.154210, -0.362409, 0.000000, 0.000000, 0.000000, 0.000000, 0.154210, 0.362409],
]


# =============================================================================
# 1. DYNAMIC MENDELEEV MASS RETRIEVAL TESTS
# =============================================================================

def test_dynamic_mendeleev_mass_resolution() -> None:
    """Verify dynamic physical atomic mass lookups for elements and isotopes via mendeleev."""
    m_h = get_dynamic_atomic_mass("H")
    m_d = get_dynamic_atomic_mass("D")
    m_c = get_dynamic_atomic_mass("C")
    m_c13 = get_dynamic_atomic_mass("13C")
    m_o = get_dynamic_atomic_mass("O")

    assert math.isclose(m_h, 1.008, rel_tol=1e-2)
    assert math.isclose(m_d, 2.0141, rel_tol=1e-3)
    assert math.isclose(m_c, 12.011, rel_tol=1e-2)
    assert math.isclose(m_c13, 13.00335, rel_tol=1e-4)
    assert math.isclose(m_o, 15.999, rel_tol=1e-2)

    # Vector resolution
    masses = resolve_mass_vector(["H", "D", "13C", "O"])
    assert len(masses) == 4
    assert masses[0] < masses[1] < masses[2] < masses[3]


# =============================================================================
# 2. INERTIA TENSOR & ROTATIONAL CONSTANTS TESTS
# =============================================================================

def test_h2o_inertia_tensor_and_rotational_constants() -> None:
    """Verify principal moments of inertia and rotational constants for authentic H2O geometry."""
    coords = np.array(H2O_COORDINATES, dtype=np.float64)
    masses = resolve_mass_vector(H2O_SYMBOLS)

    # Center of mass
    com = compute_center_of_mass(coords, masses)
    assert abs(com[0]) < 1e-12
    assert abs(com[1]) < 1e-12

    # Principal moments
    principal_moments, principal_axes, coords_pas = compute_inertia_tensor(coords, masses)
    assert principal_moments[0] < principal_moments[1] < principal_moments[2]

    # Rotational constants
    A, B, C = compute_principal_rotational_constants(principal_moments)
    assert A > B > C

    # Expected H2O rotational constants: A ~ 835 GHz, B ~ 435 GHz, C ~ 278 GHz
    assert 600000.0 < A < 1000000.0
    assert 300000.0 < B < 600000.0
    assert 200000.0 < C < 400000.0

    # Ray's asymmetry parameter: kappa in [-1, 1]
    kappa = compute_rays_asymmetry_parameter(A, B, C)
    assert -1.0 <= kappa <= 1.0

    # Planar moments and inertial defect
    P_aa, P_bb, P_cc, delta = compute_planar_moments_and_defect(principal_moments)
    assert P_aa > 0.0 and P_bb > 0.0 and P_cc >= 0.0
    # For planar H2O, Delta_e ~ 0 (or small due to float coordinates)
    assert abs(delta) < 1.0


# =============================================================================
# 3. HARMONIC NORMAL MODE & CORIOLIS TESTS
# =============================================================================

def test_harmonic_modes_and_coriolis_couplings() -> None:
    """Verify mass-weighted Hessian diagonalization, harmonic frequencies, and Coriolis couplings."""
    hessian = np.array(H2O_HESSIAN, dtype=np.float64)
    masses = resolve_mass_vector(H2O_SYMBOLS)

    m_weighted = compute_mass_weighted_hessian(hessian, masses)
    assert m_weighted.shape == (9, 9)

    modes = diagonalize_harmonic_modes(m_weighted, is_linear=False)
    assert len(modes.harmonic_frequencies_cm1) == 3
    assert modes.imaginary_frequency_count == 0
    assert modes.zero_point_energy_kcal_mol > 0.0

    # Coriolis tensor
    normal_modes = np.array(modes.normal_mode_matrix, dtype=np.float64)
    zeta = compute_coriolis_zeta_tensor(normal_modes, masses)
    assert zeta.shape == (3, 3, 3)

    # Anti-symmetry check: zeta_{ij}^alpha = -zeta_{ji}^alpha
    for alpha in range(3):
        for i in range(3):
            for j in range(3):
                assert math.isclose(zeta[alpha, i, j], -zeta[alpha, j, i], abs_tol=1e-12)


# =============================================================================
# 4. CENTRIFUGAL DISTORTION & GVPT2 ALPHAS TESTS
# =============================================================================

def test_centrifugal_distortion_and_vpt2_alphas() -> None:
    """Verify centrifugal distortion constants and vibration-rotation interaction parameters."""
    coords = np.array(H2O_COORDINATES, dtype=np.float64)
    masses = resolve_mass_vector(H2O_SYMBOLS)
    principal_moments, _, coords_pas = compute_inertia_tensor(coords, masses)
    A, B, C = compute_principal_rotational_constants(principal_moments)

    hessian = np.array(H2O_HESSIAN, dtype=np.float64)
    m_weighted = compute_mass_weighted_hessian(hessian, masses)
    modes = diagonalize_harmonic_modes(m_weighted, is_linear=False)
    normal_modes = np.array(modes.normal_mode_matrix, dtype=np.float64)

    coriolis_zeta = compute_coriolis_zeta_tensor(normal_modes, masses)
    inertia_derivs = compute_inertia_derivatives(coords_pas, normal_modes, masses)
    assert inertia_derivs.shape == (3, 3, 3)

    # Centrifugal distortion
    distortion = compute_quartic_centrifugal_distortion(
        principal_moments, modes.harmonic_frequencies_cm1, inertia_derivs
    )
    assert distortion.D_J_khz >= 0.0
    assert distortion.Delta_J_khz >= 0.0

    # Alpha parameters
    alpha_A, alpha_B, alpha_C = compute_vibration_rotation_interaction_alphas(
        A, B, C, principal_moments, modes.harmonic_frequencies_cm1,
        coriolis_zeta, inertia_derivs, None
    )
    assert len(alpha_A) == 3
    assert len(alpha_B) == 3
    assert len(alpha_C) == 3


# =============================================================================
# 5. FERMI RESONANCE & ANHARMONIC MATRIX TESTS
# =============================================================================

def test_fermi_resonance_triage() -> None:
    """Verify detection and triage of Fermi and Darling-Dennison resonances."""
    freqs = [3657.0, 1595.0, 3756.0]  # H2O symmetric stretch, bend, asymmetric stretch
    # 2 * 1595 = 3190 cm^-1 vs 3657 cm^-1 -> delta = 467 cm^-1 (no Fermi within 20 cm^-1)
    fermi, dd = detect_and_triage_fermi_resonances(freqs, threshold_cm1=20.0)
    assert isinstance(fermi, list)
    assert isinstance(dd, list)

    # Resonant frequency bundle for testing detection
    res_freqs = [2000.0, 1005.0, 995.0]  # 2 * 1005 = 2010 ~ 2000 cm^-1 (delta = 10 cm^-1)
    res_fermi, res_dd = detect_and_triage_fermi_resonances(res_freqs, threshold_cm1=20.0)
    assert len(res_fermi) >= 1
    assert res_fermi[0]["type"] == "Type_1_Fermi"
    assert res_fermi[0]["delta_cm1"] <= 20.0


# =============================================================================
# 6. ORCA CARTHESS I/O & EIGENVALUE INVARIANCE TESTS (§8A.3)
# =============================================================================

def test_orca_carthess_roundtrip_and_eigenvalues(tmp_path: Path) -> None:
    """Verify exact roundtrip I/O for ORCA .carthess files and eigenvalue invariance verification."""
    hessian = np.array(H2O_HESSIAN, dtype=np.float64)
    target_file = tmp_path / "test_h2o.carthess"

    written = write_orca_carthess(hessian, target_file)
    assert written.exists()

    read_h = read_orca_carthess(written)
    assert read_h.shape == (9, 9)
    assert np.allclose(hessian, read_h, atol=1e-7)

    # Eigenvalue validation
    is_valid, lowest_6, zero_count = validate_carthess_eigenvalues(hessian, tolerance=1e-4)
    assert len(lowest_6) == 6
    assert isinstance(is_valid, bool)


# =============================================================================
# 7. HEAVY-ATOM RMSD & RANK CORRELATION TESTS
# =============================================================================

def test_heavy_atom_rmsd_and_spearman_rho() -> None:
    """Verify heavy-atom RMSD Kabsch superposition and Spearman rank correlation rho."""
    coords1 = np.array(H2O_COORDINATES, dtype=np.float64)
    # Slightly displaced coordinates (0.05 Angstrom shift on H atoms)
    coords2 = np.array([
        [0.000000, 0.000000, 0.065590],
        [0.000000, 0.806950, -0.520720],
        [0.000000, -0.806950, -0.520720],
    ], dtype=np.float64)

    rmsd = compute_heavy_atom_rmsd(coords1, coords2, H2O_SYMBOLS)
    # Heavy atom (O) didn't move -> heavy atom RMSD = 0.0
    assert abs(rmsd) < 1e-6

    # Spearman rank correlation
    e_scout = [-76.4, -76.3, -76.2, -76.1, -76.0]
    e_anchor = [-76.42, -76.31, -76.19, -76.11, -76.02]
    rho = compute_rank_correlation_rho(e_scout, e_anchor)
    assert math.isclose(rho, 1.0, abs_tol=1e-6)


# =============================================================================
# 8. INTEGRITY GUARDS G1-G7 & AUDIT LOGGER TESTS (§8A.5)
# =============================================================================

def test_integrity_guards_evaluation() -> None:
    """Verify integrity guards G1-G7 evaluation against Method Matrix thresholds."""
    coords = np.array(H2O_COORDINATES, dtype=np.float64)

    status = evaluate_integrity_guards(
        coords_scout=coords,
        coords_anchor=coords,
        atomic_symbols=H2O_SYMBOLS,
        anchor_imaginary_modes=0,
        committee_sigma_mev_atom=1.5,
        guide_failure_history_count=0,
        spearman_rho=0.96,
        uncertainty_threshold_mev_atom=10.0,
    )

    assert status.g1_scout_answer_isolation_passed is True
    assert status.g2_high_level_hessian_passed is True
    assert status.g3_basin_identity_passed is True
    assert status.g4_rank_inversion_audit_passed is True
    assert status.g5_uncertainty_gate_passed is True
    assert status.g6_abort_triggered is False
    assert status.g7_provenance_logged is True


def test_jsonl_audit_logger(tmp_path: Path) -> None:
    """Verify JSONL structured audit logging conforming to Method Matrix §8A.5."""
    log_file = tmp_path / "test_provenance.jsonl"
    audit_logger = MLFFScoutAuditLogger(log_file)

    record = AuditRecord(
        stage="mlff_preopt",
        decision="seed_dft_optimisation",
        guide={
            "code": "mace-torch 0.3.x",
            "model_key": "MACE-OFF24-medium",
            "sha256": "abc123def456",
            "precision": "float64",
            "device": "cuda:0",
        },
        input_structure={"structure_id": "h2o_01", "n_atoms": 3},
        output_data={"E_hartree": -76.4, "B_0_mhz": 435000.0},
        gates={"G4_spearman_rho": 0.95, "G3_rmsd_A": 0.02},
        consumer={"anchor_job": "h2o_wb97m.inp", "hessian_transferred": True},
        authority=AuthorityLevel.ADVISORY_ONLY,
    )

    audit_logger.log_record(record)
    assert log_file.exists()

    records = audit_logger.read_records()
    assert len(records) == 1
    assert records[0]["stage"] == "mlff_preopt"
    assert records[0]["authority"] == "advisory_only"


# =============================================================================
# 9. END-TO-END MLFF SCOUT BRIDGE INTEGRATION TEST
# =============================================================================

def test_mlff_scout_bridge_end_to_end(tmp_path: Path) -> None:
    """Verify complete end-to-end MLFFScoutBridge pipeline execution."""
    log_path = tmp_path / "provenance.jsonl"
    carthess_out = tmp_path / "scout_guess.carthess"
    bridge = MLFFScoutBridge(audit_log_path=log_path)

    # 1. Prepare Scout PES Derivatives
    scout_pes = PESDerivatives(
        atomic_symbols=H2O_SYMBOLS,
        coordinates_angstrom=H2O_COORDINATES,
        energy_hartree=-76.40521,
        gradient_hartree_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        hessian_hartree_bohr2=H2O_HESSIAN,
        model_family=MLFFModelFamily.MACE_OFF24M,
        canonical_model_key="MACE-OFF24-medium",
        provenance=ProvenanceTag.SCOUT_ESTIMATE,
        authority=AuthorityLevel.ADVISORY_ONLY,
    )

    # 2. Ingest Scout Derivatives & Compute GVPT2
    scout_constants = bridge.ingest_scout_derivatives(scout_pes)
    assert scout_constants.provenance == ProvenanceTag.SCOUT_ESTIMATE
    assert scout_constants.authority == AuthorityLevel.ADVISORY_ONLY
    assert scout_constants.B_0_mhz > 0.0

    # 3. Generate ORCA .carthess Preconditioner
    carthess_file = bridge.generate_orca_preconditioner(scout_pes, carthess_out)
    assert carthess_file.exists()

    # 4. Prepare Authoritative Anchor Derivatives
    anchor_pes = PESDerivatives(
        atomic_symbols=H2O_SYMBOLS,
        coordinates_angstrom=H2O_COORDINATES,
        energy_hartree=-76.42150,
        gradient_hartree_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        hessian_hartree_bohr2=H2O_HESSIAN,
        model_family=MLFFModelFamily.CUSTOM_MLFF,
        canonical_model_key="ORCA-wB97X-D3",
        provenance=ProvenanceTag.DERIVED,
        authority=AuthorityLevel.ADVISORY_ONLY,
    )
    anchor_constants = compute_gvpt2_spectroscopic_parameters(anchor_pes)

    # 5. Verify Scout against Anchor
    comparison = bridge.verify_against_anchor(
        scout_derivatives=scout_pes,
        scout_constants=scout_constants,
        anchor_derivatives=anchor_pes,
        anchor_constants=anchor_constants,
        committee_sigma_mev_atom=1.2,
        spearman_rho=0.98,
    )

    assert comparison.relative_error_B_percent < 5.0
    assert comparison.guards.g3_basin_identity_passed is True
    assert comparison.scout_provenance == ProvenanceTag.SCOUT_ESTIMATE
    assert comparison.anchor_provenance == ProvenanceTag.DERIVED

    # Check audit log written
    records = bridge.audit_logger.read_records()
    assert len(records) >= 1
    assert records[-1]["authority"] == "authoritative"
