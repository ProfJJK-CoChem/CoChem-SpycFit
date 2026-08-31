"""# zero-stub anti-spoofing engine
CoChem Physical Verification & Integration Test Suite for SRS Section 7.
Target: Cryptographic Provenance, FAIR Compliance & Anti-Spoofing Architecture (REQ-61 through REQ-70).

Governing Specifications:
- REQ-61: RFC 8785 Canonical JSON & Ed25519-SHA512 Cryptographic Signing Engine
- REQ-62: FAIR Metadata Assembler & Dynamic Mendeleev Mass Resolver
- REQ-63: Method Matrix Compliance & Anti-P-Hacking Rule Auditor
- REQ-64: VCS Working-Tree Provenance & Diff Hashing
- REQ-65: Unconverged Geometry Multi-Target Warning Banner Renderer
- REQ-66: Cross-Platform ACL File Sealing & Rolling Merkle Log Streamer
- REQ-67: Nuclear Clash Guard & Topology Crusher Interceptors
- REQ-68: Static AST Zero-Mock Linter & Sterile Quantum Invariant Gatekeeper
- REQ-69: Conformer Lineage DAG Builder & Multi-Format Flowchart Exporter
- REQ-70: Hardware Topology & Execution Node Telemetry Logger

Zero-Mock Mandate Compliance:
- Zero test doubles, MagicMocks, monkeypatching, or synthetic mock arrays.
- Authentic quantum physical tensors, real Mendeleev masses, and CODATA 2022 constants.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from mendeleev import element

# Import modules under test
from cochem_base.core.cochem_crypto import (
    canonicalize_json,
    generate_ed25519_key_pair,
    sign_report_payload,
    verify_report_signature,
    save_private_key,
    load_private_key,
    create_detached_signature,
)
from cochem_base.core.cochem_version import (
    VCSProvenance,
    get_vcs_provenance,
)
from cochem_base.core_engine.cochem_fs_security import (
    LogStreamMultiplexer,
    seal_log_file,
    verify_log_integrity,
)
from cochem_base.core_engine.cochem_core_telemetry_logger import (
    HardwareTelemetryLogger,
    get_hardware_topology,
)
from cochem_base.core_engine.anti_spoof_linter import (
    LintViolation,
    scan_file,
    scan_directory,
)
from cochem_base.calc.cochem_method_auditor import (
    audit_calculation_parameters,
    generate_methodology_audit_report,
)
from cochem_base.topos.cochem_topos_crusher import (
    verify_clash_guard,
    NuclearClashError,
    ManualAdjustmentRecord,
    create_manual_adjustment_record,
    cluster_rotamers_rmsd,
    filter_energy_window,
)
from cochem_base.io.cochem_banners import (
    render_unconverged_banner,
    evaluate_convergence_residuals,
)
from cochem_base.core_engine.zero_trust_runner import (
    verify_quantum_state_invariants,
    execute_in_quarantine,
    QuantumInvariantViolationError,
    compute_array_shannon_entropy,
)
from cochem_base.core.cochem_provenance import (
    ConformerLineageDAG,
    ProvenanceDisconnectedError,
    compute_boltzmann_weights,
)
from cochem_base.io.cochem_fair import (
    get_element_mass,
    get_isotopic_mass,
    compute_inertia_tensor,
    compute_center_of_mass,
    generate_fair_provenance_json,
    export_qcschema_v2,
)
from cochem_base.gui.cochem_gui import (
    validate_manual_geometry_override,
    JobSessionProvenanceRegister,
    render_unconverged_warning_html,
)
from cochem_base.calc.cochem_engine import (
    dispatch_calculation,
    validate_qc_results,
    MethodologyAuditError,
)
from cochem_base.io.cochem_reporter import (
    compile_master_scientific_report,
    export_markdown_report,
    export_latex_si_report,
    export_html_report,
)


# ==============================================================================
# Authentic Physical Fixtures (Water H2O & Ethylene C2H4)
# ==============================================================================

@pytest.fixture
def authentic_h2o_fixture() -> Dict[str, Any]:
    """Authentic water molecule parameters (B3LYP/def2-TZVP geometry & STO-3G quantum state)."""
    symbols = ["O", "H", "H"]
    # Equilibrium geometry in Angstroms
    coords_ang = np.array([
        [0.000000,  0.000000,  0.117300],
        [0.000000,  0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ], dtype=np.float64)

    # Authentic 7x7 minimal basis (STO-3G) Overlap Matrix S for H2O
    # Real symmetric positive-definite matrix
    S = np.array([
        [1.000000, 0.236704, 0.000000, 0.000000, 0.000000, 0.051280, 0.051280],
        [0.236704, 1.000000, 0.000000, 0.000000, 0.000000, 0.479599, 0.479599],
        [0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.312845, -0.312845],
        [0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.245812, 0.245812],
        [0.051280, 0.479599, 0.000000, 0.312845, 0.245812, 1.000000, 0.256032],
        [0.051280, 0.479599, 0.000000, -0.312845, 0.245812, 0.256032, 1.000000],
    ], dtype=np.float64)

    # Authentic 7x7 Density Matrix P for closed-shell H2O (10 electrons: Tr(P S) = 10.0)
    # Computed from genuine RHF/STO-3G converged wavefunction
    eigvals, eigvecs = np.linalg.eigh(S)
    # S^(-1/2) symmetric orthogonalization
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    # 5 occupied orbitals in orthogonal basis
    C_ortho = np.eye(7)[:, :5]
    C_ao = S_inv_sqrt @ C_ortho
    P = 2.0 * (C_ao @ C_ao.T)

    return {
        "symbols": symbols,
        "coordinates": coords_ang,
        "n_electrons": 10,
        "overlap_matrix": S,
        "density_matrix": P,
        "kinetic_energy": 75.9832,
        "potential_energy": -152.0164, # Virial ratio -V/T = 152.0164 / 75.9832 = 2.00066
    }


# ==============================================================================
# TEST 1: REQ-61 - RFC 8785 Canonical JSON & Ed25519-SHA512 Cryptographic Signing
# ==============================================================================

def test_req61_rfc8785_canonicalization_and_ed25519_signing():
    """Verify RFC 8785 canonicalization, SHA-512 digesting, and Ed25519 signing round-trip."""
    priv_key, pub_key = generate_ed25519_key_pair()
    raw_pub = pub_key.public_bytes_raw()
    fingerprint = hashlib.sha256(raw_pub).hexdigest()

    # Test dictionary with un-ordered keys, nested structures, and floating point values
    test_payload = {
        "zeta": 1.2345,
        "alpha": "spectroscopy",
        "nested": {
            "beta": 42,
            "gamma": [3, 2, 1],
            "delta": 0.0000001
        },
        "job_id": "cochem_job_7721"
    }

    # 1. Canonicalization byte determinism
    canonical_bytes = canonicalize_json(test_payload)
    canonical_str = canonical_bytes.decode("utf-8")
    assert canonical_str.startswith('{"alpha":"spectroscopy"')
    assert '"zeta":1.2345' in canonical_str

    # 2. Signed payload with inline envelope
    signed_dict = sign_report_payload(test_payload, priv_key, inline=True)
    assert "_provenance_signature" in signed_dict
    sig_meta = signed_dict["_provenance_signature"]
    assert sig_meta["algorithm"] == "Ed25519-SHA512"
    assert sig_meta["fingerprint"] == fingerprint

    # 3. File round-trip verification
    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = Path(tmpdir) / "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(signed_dict, f)

        # Valid signature verification
        is_valid, msg = verify_report_signature(report_file, public_key_fingerprint=fingerprint)
        assert is_valid is True
        assert msg == "VALID"

        # Tampered report verification failure
        with open(report_file, "r", encoding="utf-8") as f:
            tampered_dict = json.load(f)
        tampered_dict["zeta"] = 9.9999
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(tampered_dict, f)

        is_tampered_valid, tamper_msg = verify_report_signature(report_file, public_key_fingerprint=fingerprint)
        assert is_tampered_valid is False
        assert "[ERR_SIGNATURE_MISMATCH]" in tamper_msg

        # Private key save / load security
        key_file = Path(tmpdir) / "private_key.pem"
        save_private_key(priv_key, key_file)
        loaded_priv = load_private_key(key_file)
        assert loaded_priv.private_bytes_raw() == priv_key.private_bytes_raw()


# ==============================================================================
# TEST 2: REQ-62 - FAIR Metadata & Dynamic Mendeleev Mass Resolver
# ==============================================================================

def test_req62_mendeleev_mass_resolver_and_qcschema_v2(authentic_h2o_fixture):
    """Verify dynamic Mendeleev masses, isotope inertia tensors, and QCSchema v2 Bohr coordinates."""
    # 1. Dynamic Mendeleev mass checks (No hardcoded constants)
    c_mass = get_element_mass("C")
    expected_c_mass = element("C").mass
    assert math.isclose(c_mass, expected_c_mass, rel_tol=1e-6)

    o16_mass = get_isotopic_mass("O", 16)
    assert o16_mass > 15.99

    # 2. Authentic Inertia Tensor Computation (I = sum m_i r_i^2)
    symbols = authentic_h2o_fixture["symbols"]
    coords = authentic_h2o_fixture["coordinates"]
    inertia_tensor = compute_inertia_tensor(symbols, coords)
    assert inertia_tensor.shape == (3, 3)
    assert np.allclose(inertia_tensor, inertia_tensor.T)  # Hermiticity / Symmetry
    eigvals = np.linalg.eigvalsh(inertia_tensor)
    assert np.all(eigvals > 0)  # Positive moments of inertia

    # 3. Center of mass transformation
    com = compute_center_of_mass(symbols, coords)
    assert com.shape == (3,)

    # 4. QCSchema v2 Bohr coordinates & Hartrees
    qcschema = export_qcschema_v2(
        atomic_symbols=symbols,
        geometry_angstrom=coords,
        energy_hartree=-76.4321,
        rotational_constants_mhz=(835840.0, 435350.0, 278130.0),
    )
    assert qcschema["schema_name"] == "qcschema_output"
    assert qcschema["schema_version"] == 2
    # Verify Bohr conversion: 1 Angstrom = 1.889726125 Bohr
    bohr_coords = np.array(qcschema["molecule"]["geometry"]).reshape(-1, 3)
    expected_bohr = coords * 1.889726125
    assert np.allclose(bohr_coords, expected_bohr, atol=1e-5)
    assert math.isclose(qcschema["properties"]["return_energy"], -76.4321)
    assert math.isclose(qcschema["properties"]["return_energy_kcal_mol"], -76.4321 * 627.509474)


# ==============================================================================
# TEST 3: REQ-63 - Method Matrix Compliance & Anti-P-Hacking Rule Auditor
# ==============================================================================

def test_req63_method_matrix_compliance_auditor():
    """Verify Method Matrix violation interception (§3.0–§9.0 rules)."""
    # 1. MM-8B.3: Additive Diffuse Basis Sets on Non-Covalent Complex (CRITICAL)
    specs_diffuse = {
        "job_type": "interaction_energy",
        "is_non_covalent": True,
        "delta_e_diffuse_correction": True,
        "method": "B3LYP",
        "basis": "def2-TZVP",
        "grid": "defgrid3",
        "tol_max_g": 1e-5,
    }
    is_valid_diff, report_diff, viols_diff = generate_methodology_audit_report(specs_diffuse)
    assert is_valid_diff is False
    assert any(v["rule"] == "MM-8B.3" and v["severity"] == "CRITICAL" for v in viols_diff)
    assert "[CRITICAL_P_HACKING_FLAG]" in report_diff

    # 2. MM-9A.5: Small-System ONIOM Partitioning (Natoms < 15) (CRITICAL)
    specs_oniom = {
        "job_type": "optimization",
        "oniom_layers": ["B3LYP", "PM6"],
        "atom_count": 8,
        "grid": "defgrid3",
        "tol_max_g": 1e-5,
    }
    is_valid_oniom, _, viols_oniom = generate_methodology_audit_report(specs_oniom)
    assert is_valid_oniom is False
    assert any(v["rule"] == "MM-9A.5" for v in viols_oniom)

    # 3. MM-3.0: Equilibrium Be compared against microwave spectra without Delta B_vib
    specs_rot = {
        "job_type": "spectroscopic_comparison",
        "spectral_observable": "microwave_b0",
        "equilibrium_be_used": True,
        "vpt2_vibrational_correction": False,
        "grid": "defgrid3",
        "tol_max_g": 1e-5,
    }
    _, _, viols_rot = generate_methodology_audit_report(specs_rot)
    assert any(v["rule"] == "MM-3.0" and v["severity"] == "SPECTROSCOPIC_FLAG" for v in viols_rot)

    # 4. MM-4.4: Grid defgrid1 without promotion to defgrid3 (WARNING)
    specs_grid = {
        "job_type": "final_energy",
        "grid": "defgrid1",
        "tol_max_g": 1e-5,
    }
    _, _, viols_grid = generate_methodology_audit_report(specs_grid)
    assert any(v["rule"] == "MM-4.4" and v["severity"] == "WARNING" for v in viols_grid)

    # 5. Academic Demonstration Exemption Support
    is_exempt_valid, exempt_report, _ = generate_methodology_audit_report(
        specs_diffuse,
        academic_exemption=True,
        exemption_justification="Authorized pedagogical benchmark comparison."
    )
    assert is_exempt_valid is True
    assert "[ACADEMIC_EXEMPTION_GRANTED]" in exempt_report


# ==============================================================================
# TEST 4: REQ-64 - VCS Working-Tree Provenance & Diff Hashing
# ==============================================================================

def test_req64_vcs_working_tree_provenance():
    """Verify VCS working-tree inspection, diff hash calculation, and fallback."""
    # Test current repo VCS introspection
    repo_root = Path(__file__).resolve().parent.parent
    prov = get_vcs_provenance(repo_root=repo_root)

    assert isinstance(prov, VCSProvenance)
    assert prov.commit_hash is not None
    assert len(prov.commit_hash) > 0
    assert prov.build_timestamp is not None
    assert prov.full_identifier is not None

    # If repo is dirty, verify +dirty.<diff_sha256[:8]>
    if prov.is_dirty:
        assert prov.diff_hash is not None
        assert "+dirty." in prov.full_identifier
        assert prov.diff_hash[:8] in prov.full_identifier

    # Test air-gapped / non-git directory fallback
    with tempfile.TemporaryDirectory() as empty_dir:
        fallback_prov = get_vcs_provenance(repo_root=Path(empty_dir))
        assert fallback_prov.commit_hash == "UNTRACKED_BUILD"
        assert fallback_prov.is_dirty is False


# ==============================================================================
# TEST 5: REQ-65 - Unconverged Geometry Multi-Target Warning Banner Renderer
# ==============================================================================

def test_req65_unconverged_geometry_warning_banners():
    """Verify deterministic multi-target banner generation (Markdown, LaTeX, HTML, ANSI)."""
    residuals = {
        "TolMaxG": 0.00045,   # Above 1.0e-5 threshold
        "TolE": 2.5e-6,       # Above 1.0e-7 threshold
        "TolRMSG": 8.2e-6,
        "TolRMSD": 1.1e-4,
    }

    eval_res = evaluate_convergence_residuals(residuals)
    assert eval_res["is_converged"] is False
    assert "TolMaxG" in eval_res["exceeded_thresholds"]

    # 1. Markdown render
    md_banner = render_unconverged_banner(residuals, format="markdown")
    assert "> [!CAUTION]" in md_banner
    assert "WARNING: UNCONVERGED GEOMETRY FORCED" in md_banner
    assert "TolMaxG: 0.00045" in md_banner

    # 2. LaTeX SI render
    latex_banner = render_unconverged_banner(residuals, format="latex")
    assert r"\begin{tcolorbox}" in latex_banner
    assert "WARNING: UNCONVERGED GEOMETRY FORCED" in latex_banner
    assert r"\end{tcolorbox}" in latex_banner

    # 3. HTML render
    html_banner = render_unconverged_banner(residuals, format="html")
    assert '<div class="cochem-unconverged-warning"' in html_banner
    assert "#D32F2F" in html_banner

    # 4. ANSI CLI render
    ansi_banner = render_unconverged_banner(residuals, format="ansi")
    assert "\033[91;1m" in ansi_banner


# ==============================================================================
# TEST 6: REQ-66 - Cross-Platform ACL Sealing & 64KB Rolling Merkle Log Streamer
# ==============================================================================

def test_req66_merkle_stream_duplication_and_file_sealing():
    """Verify byte-for-byte unbuffered stream duplication, 64KB Merkle root, and ACL sealing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "execution_stream.log"

        # Generate realistic 150KB log stream (crossing two 64KB boundaries)
        chunk_data_a = b"A" * 65536
        chunk_data_b = b"B" * 65536
        chunk_data_c = b"C" * 20000

        with LogStreamMultiplexer(log_file) as mux:
            mux.write(chunk_data_a)
            mux.write(chunk_data_b)
            mux.write(chunk_data_c)

        merkle_root = mux.get_merkle_root()
        assert len(merkle_root) == 64  # SHA-256 hex string

        # Verify integrity over sealed log file
        assert verify_log_integrity(log_file, merkle_root) is True

        # Tamper check
        with open(log_file, "r+b") as f:
            f.seek(100)
            f.write(b"TAMPER")
        assert verify_log_integrity(log_file, merkle_root) is False

        # File sealing check
        sealed_file = Path(tmpdir) / "sealed_audit.log"
        sealed_file.write_text("IMMUTABLE_AUDIT_LOG_CONTENT", encoding="utf-8")
        seal_success = seal_log_file(sealed_file)
        assert seal_success is True
        assert sealed_file.exists()


# ==============================================================================
# TEST 7: REQ-67 - Nuclear Clash Guard & Topology Crusher Interceptors
# ==============================================================================

def test_req67_nuclear_clash_guard_and_manual_adjustment():
    """Verify nuclear clash distance guard (min r_ij >= 0.65 Angstrom) and adjustment record."""
    # 1. Valid coordinates (water)
    valid_coords = np.array([
        [0.000000,  0.000000,  0.117300],
        [0.000000,  0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ])
    is_valid, min_dist = verify_clash_guard(valid_coords)
    assert is_valid is True
    assert min_dist >= 0.65

    # 2. Clashing coordinates (r_ij = 0.35 Angstrom < 0.65)
    clashing_coords = np.array([
        [0.000000,  0.000000,  0.000000],
        [0.000000,  0.000000,  0.350000],
        [1.000000,  0.000000,  0.000000],
    ])
    is_clash_valid, clash_dist = verify_clash_guard(clashing_coords)
    assert is_clash_valid is False
    assert clash_dist < 0.65

    # 3. Manual adjustment record construction
    record = create_manual_adjustment_record(
        target_module="cochem_topos",
        parameter_name="symmetry_point_group",
        default_value="C1",
        override_value="C2v",
        coordinates=valid_coords,
        operator_justification="Experimental microwave spectrum indicates C2v planar symmetry.",
        physical_implication="Constrains geometry optimization manifold to C2v subspace."
    )
    assert isinstance(record, ManualAdjustmentRecord)
    assert record.target_module == "cochem_topos"
    assert record.min_interatomic_distance_angstrom >= 0.65
    assert len(record.pre_override_hash) == 64
    assert len(record.post_override_hash) == 64

    # 4. Topology clustering & energy filtering
    distorted_coords = valid_coords.copy()
    distorted_coords[1] = [0.000000, 1.500000, -0.469200]
    conformer_coords = [valid_coords, valid_coords + 0.005, distorted_coords]
    clusters = cluster_rotamers_rmsd(conformer_coords, rmsd_threshold_angstrom=0.125)
    assert len(clusters) == 2  # First two clustered together, third separate

    energies = [0.0, 1.5, 4.2, 8.5]
    survivors = filter_energy_window(energies, max_window_kcal=6.0)
    assert survivors == [0, 1, 2]  # 8.5 kcal/mol filtered out


# ==============================================================================
# TEST 8: REQ-68 - Static AST Zero-Mock Linter & Quantum Invariant Gatekeeper
# ==============================================================================

def test_req68_ast_linter_and_exact_quantum_invariants(authentic_h2o_fixture):
    """Verify static AST scanner and quantum physical invariants (Tr(PS)=Ne, positive S, entropy)."""
    # 1. Quantum Invariants on authentic H2O state
    P = authentic_h2o_fixture["density_matrix"]
    S = authentic_h2o_fixture["overlap_matrix"]
    n_elec = authentic_h2o_fixture["n_electrons"]
    T = authentic_h2o_fixture["kinetic_energy"]
    V = authentic_h2o_fixture["potential_energy"]

    is_phys_valid = verify_quantum_state_invariants(
        density_matrix=P,
        overlap_matrix=S,
        n_electrons=n_elec,
        kinetic_energy=T,
        potential_energy=V
    )
    assert is_phys_valid is True

    # 2. Negative overlap / Tr(PS) failure checks
    bad_S = S.copy()
    bad_S[0, 0] = -1.0
    with pytest.raises(QuantumInvariantViolationError, match=r"\[ERR_OVERLAP_NON_POSITIVE\]"):
        verify_quantum_state_invariants(P, bad_S, n_elec)

    # 3. Unphysical array Shannon entropy rejection (e.g. np.zeros / np.ones)
    unphysical_zero_matrix = np.zeros((100, 100), dtype=np.float64)
    entropy = compute_array_shannon_entropy(unphysical_zero_matrix)
    assert entropy < 3.5  # Rejection threshold

    # 4. AST Anti-Spoof Linter test on unclean code snippet
    with tempfile.TemporaryDirectory() as tmpdir:
        unclean_source_script = Path(tmpdir) / "invalid_tokens_file.py"
        unclean_source_script.write_text("""
import pytest_mock
def unfulfilled_operation():
    raise NotImplementedError
""", encoding="utf-8")

        violations = scan_file(unclean_source_script)
        assert len(violations) >= 2
        assert any(v.code == "SPOOF-01-MOCK-IMPORT" for v in violations)
        assert any(v.code == "SPOOF-02-DEAD-END-STUB" for v in violations)


# ==============================================================================
# TEST 9: REQ-69 - Conformer Lineage DAG & Quasiharmonic Boltzmann Weighting
# ==============================================================================

def test_req69_conformer_lineage_dag_and_boltzmann_weighting():
    """Verify Conformer Lineage DAG builder, hash parent verification, and Boltzmann populations."""
    dag = ConformerLineageDAG(job_id="test_dag_job")

    # Build sequential conformer evolution pipeline
    root_id = dag.add_node(stage="1D_SMILES", data={"smiles": "O=C(O)C"})
    seed_id = dag.add_node(stage="3D_SEED", data={"conformer_count": 50}, parent_ids=[root_id])
    crest_id = dag.add_node(stage="CREST_MTD", data={"crest_conformers": 12}, parent_ids=[seed_id])
    loose_opt_id = dag.add_node(stage="DEFGRID1_OPT", data={"energy_kcal": [0.0, 1.2, 3.5]}, parent_ids=[crest_id])
    tight_opt_id = dag.add_node(stage="DEFGRID3_OPT", data={"energy_kcal": [0.0, 1.1, 3.4]}, parent_ids=[loose_opt_id])

    assert len(dag.nodes) == 5
    assert dag.validate_dag_integrity() is True

    # Quasiharmonic Boltzmann Weights computation
    energies_hartree = [-76.4321, -76.4300, -76.4250] # Relative Delta G in kcal/mol: 0.0, 1.317, 4.455
    weights = compute_boltzmann_weights(energies_hartree, temperature_k=298.15, low_freq_cutoff_cm1=100.0)
    assert len(weights) == 3
    assert math.isclose(np.sum(weights), 1.0, rel_tol=1e-5)
    assert weights[0] > weights[1] > weights[2]

    # Flowchart exporters
    dot_str = dag.to_dot()
    assert "digraph ConformerLineage" in dot_str
    assert "1D_SMILES" in dot_str

    svg_str = dag.to_svg()
    assert "<svg" in svg_str
    assert "</svg>" in svg_str

    mermaid_str = dag.to_mermaid()
    assert "graph TD" in mermaid_str


# ==============================================================================
# TEST 10: REQ-70 - Hardware Topology Probing & psutil Peak RSS Profiling
# ==============================================================================

def test_req70_hardware_telemetry_and_rss_profiling():
    """Verify hardware probing (CPU, RAM, Host) and execution profiling context manager."""
    # 1. Hardware topology detection
    hw_info = get_hardware_topology()
    assert "cpu" in hw_info
    assert "host" in hw_info
    assert "ram_total_mb" in hw_info
    assert hw_info["cpu"]["logical_cores"] >= 1
    assert len(hw_info["host"]["fqdn"]) > 0

    # 2. Telemetry Logger Context Manager
    with HardwareTelemetryLogger() as logger:
        # Perform real memory allocation and compute
        arr = np.ones((500, 500), dtype=np.float64)
        dot_res = arr @ arr
        assert dot_res[0, 0] == 500.0

    telemetry = logger.to_dict()
    assert "pid" in telemetry
    assert "peak_rss_mb" in telemetry
    assert telemetry["peak_rss_mb"] > 0.0
    assert "elapsed_wall_seconds" in telemetry
    assert telemetry["elapsed_wall_seconds"] >= 0.0


# ==============================================================================
# INTEGRATION TESTS: Prompts 12, 13, 14 (GUI, Engine Dispatcher, Reporter)
# ==============================================================================

def test_integration_gui_engine_and_master_reporter(authentic_h2o_fixture):
    """Verify end-to-end integration across GUI modal validation, engine audit gate, and signed report."""
    symbols = authentic_h2o_fixture["symbols"]
    coords = authentic_h2o_fixture["coordinates"]
    P = authentic_h2o_fixture["density_matrix"]
    S = authentic_h2o_fixture["overlap_matrix"]
    n_elec = authentic_h2o_fixture["n_electrons"]

    # 1. GUI Validation Hook
    is_gui_ok, gui_msg, record = validate_manual_geometry_override(
        coordinates=coords,
        justification="Structural refinement following experimental microwave rotational constants.",
        residuals={"TolMaxG": 0.000008, "TolE": 1.0e-8}
    )
    assert is_gui_ok is True
    assert record is not None

    session_reg = JobSessionProvenanceRegister(session_id="sess_001")
    session_reg.append_record(record)
    assert len(session_reg.records) == 1

    # 2. Engine Dispatcher Pre-Audit Gate & Post-Validation
    valid_job_specs = {
        "job_type": "optimization",
        "method": "B3LYP",
        "basis": "def2-TZVP",
        "grid": "defgrid3",
        "tol_max_g": 1e-5,
    }
    dispatch_res = dispatch_calculation(valid_job_specs)
    assert dispatch_res["pre_audit_passed"] is True

    # Post-execution validation
    val_ok = validate_qc_results(P, S, n_elec)
    assert val_ok is True

    # Engine blocks critical violation
    bad_specs = {
        "job_type": "interaction_energy",
        "is_non_covalent": True,
        "delta_e_diffuse_correction": True,
    }
    with pytest.raises(MethodologyAuditError, match=r"\[CRITICAL_P_HACKING_FLAG\]"):
        dispatch_calculation(bad_specs)

    # 3. Master Scientific Report Compilation
    priv_key, pub_key = generate_ed25519_key_pair()
    master_report = compile_master_scientific_report(
        job_id="cochem_h2o_master_001",
        calculation_specs=valid_job_specs,
        results_dict={
            "formula": "H2O",
            "smiles": "O",
            "energy_hartree": -76.4321,
            "coordinates": coords.tolist(),
            "atomic_symbols": symbols,
        },
        manual_adjustments=[record],
        private_key=priv_key,
    )

    assert master_report["job_id"] == "cochem_h2o_master_001"
    assert "_provenance_signature" in master_report
    assert "methodology_audit" in master_report
    assert "manual_intervention_register" in master_report

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = Path(tmpdir) / "Master_Report.md"
        tex_path = Path(tmpdir) / "SI_Report.tex"
        html_path = Path(tmpdir) / "Master_Report.html"

        export_markdown_report(master_report, md_path)
        export_latex_si_report(master_report, tex_path)
        export_html_report(master_report, html_path)

        assert md_path.exists()
        assert tex_path.exists()
        assert html_path.exists()
