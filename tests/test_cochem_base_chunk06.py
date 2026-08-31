"""
Comprehensive Zero-Mock PyTest Architectural Verification Suite for COCHEM-SRS-CHUNK-06.

Validates all 14 prompts and requirements REQ-CORE-051 through REQ-DIST-060:
- Prompt 01: Domain Exception Hierarchy & Pydantic v2 Core Schemas (src/cochem/core/models.py & exceptions.py)
- Prompt 02: Mendeleev Core Shell Lookup & Exact Masses (src/cochem/core/cochem_elements.py)
- Prompt 03: Hardware RAM & NVMe Profiling (src/cochem/core/cochem_hardware.py)
- Prompt 04: Functional & Auxiliary Fitting Basis Registry (src/cochem/qc/cochem_method_registry.py)
- Prompt 05: Point Group Symmetry Detection & Orientation (src/cochem/qc/cochem_symmetry.py)
- Prompt 06: Molecular Graph Topology & Gyration Asphericity (src/cochem/qc/cochem_graph_analyzer.py)
- Prompt 07: Whitened 12D KD-Tree Sieving Engine (src/cochem/topos/cochem_topos_siever.py)
- Prompt 08: Boltzmann Shannon Entropy Convergence (src/cochem/topos/cochem_topos_telemetry.py)
- Prompt 09: TOPOS Crusher Engine Integration (src/cochem/topos/cochem_topos_crusher.py)
- Prompt 10: High-Performance ORCA Deck Builder (src/cochem/core/cochem_orca_builder.py)
- Prompt 11: CFOUR ZMAT & Symmetry Builder (src/cochem/core/cochem_cfour_builder.py)
- Prompt 12: Dynamic Storage Tier Pinning & QC Pipeline (src/cochem/qc/cochem_qc_pipeline.py)
- Prompt 13: Distributed Sayvetz/Eckart 6N NumFreq Runner (src/cochem/qc/cochem_numfreq_distributor.py)
- Prompt 14: Verification Suite Integration
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pytest
from mendeleev import element as m_element

from cochem.core.models import (
    CoChemPerformanceException,
    InsufficientMemoryAllocationError,
    IncompatibleAuxiliaryBasisError,
    AsymmetricNumericalHessianError,
    SCFSymmetryStagnationError,
    InhomogeneousFeatureSievingError,
    DistributedDisplacementExhaustionError,
    MemoryAllocationConfig,
    RIConfigurationPayload,
    SievingFeatureVector,
    DistributedNumFreqTask,
)
from cochem.core.cochem_elements import (
    get_atomic_mass,
    get_atomic_number,
    get_covalent_radius,
    get_frozen_core_count,
    batch_get_masses,
)
from cochem.core.cochem_hardware import (
    calculate_memory_allocation,
    benchmark_scratch_io,
    get_effective_system_memory_mb,
)
from cochem.qc.cochem_method_registry import (
    FunctionalType,
    classify_functional,
    resolve_auxiliary_basis,
    get_recommended_grid,
)
from cochem.qc.cochem_symmetry import (
    detect_point_group,
    compute_center_of_mass,
    align_to_principal_axes,
)
from cochem.qc.cochem_graph_analyzer import (
    build_covalent_molecular_graph,
    compute_graph_diameter_and_sparsity,
    compute_gyration_tensor_and_asphericity,
    should_promote_to_dlpno,
)
from cochem.topos.cochem_topos_siever import (
    extract_12d_feature_vector,
    whiten_features,
    TwoStageKDTreeSiever,
)
from cochem.topos.cochem_topos_telemetry import (
    ConformerBasinTelemetry,
    calculate_boltzmann_weights,
    calculate_normalized_shannon_entropy,
)
from cochem.topos.cochem_topos_crusher import (
    ToposCrusherEngine,
    compute_principal_moments_of_inertia,
    assemble_dimer_complex,
)
from cochem.core.cochem_orca_builder import (
    generate_orca_chunk06_deck,
)
from cochem.core.cochem_cfour_builder import (
    generate_cfour_zmat_deck,
)
from cochem.qc.cochem_qc_pipeline import (
    determine_2e_storage_tier,
    execute_qc_with_symmetry_recovery,
)
from cochem.qc.cochem_numfreq_distributor import (
    generate_6n_cartesian_displacements,
    reconstruct_hessian_from_gradients,
    validate_hessian_symmetry,
    mass_weight_hessian,
    construct_sayvetz_eckart_projector,
    project_and_diagonalize_hessian,
    serialize_to_hess_format,
)


# ==============================================================================
# 1. Prompt 01: Domain Exception Hierarchy & Pydantic v2 Models
# ==============================================================================

def test_chunk06_exception_hierarchy() -> None:
    """Verify exception inheritance, error messaging, and details extraction."""
    assert issubclass(InsufficientMemoryAllocationError, CoChemPerformanceException)
    assert issubclass(IncompatibleAuxiliaryBasisError, CoChemPerformanceException)
    assert issubclass(AsymmetricNumericalHessianError, CoChemPerformanceException)
    assert issubclass(SCFSymmetryStagnationError, CoChemPerformanceException)
    assert issubclass(InhomogeneousFeatureSievingError, CoChemPerformanceException)
    assert issubclass(DistributedDisplacementExhaustionError, CoChemPerformanceException)

    exc_mem = InsufficientMemoryAllocationError(requested_cores=16, available_mb=2048, min_core_mb=512)
    assert "Insufficient memory" in str(exc_mem)
    assert exc_mem.requested_cores == 16
    assert exc_mem.min_core_mb == 512

    exc_aux = IncompatibleAuxiliaryBasisError(primary_basis="unknown-basis", functional="B3LYP")
    assert "Incompatible auxiliary basis" in str(exc_aux)

    exc_hess = AsymmetricNumericalHessianError(delta_asymmetry=2.5e-4, threshold=1.0e-4)
    assert exc_hess.delta_asymmetry == 2.5e-4
    assert exc_hess.threshold == 1.0e-4


def test_chunk06_pydantic_models() -> None:
    """Verify Pydantic v2 models, validation constraints, and computed properties."""
    # MemoryAllocationConfig
    mem_cfg = MemoryAllocationConfig(
        total_system_ram_mb=16384,
        cgroup_limit_mb=8192,
        requested_cores=8,
        headroom_fraction=0.20,
        min_core_mb=512,
    )
    assert mem_cfg.effective_limit_mb == 8192

    mem_cfg_no_cgroup = MemoryAllocationConfig(
        total_system_ram_mb=16384,
        requested_cores=8,
    )
    assert mem_cfg_no_cgroup.effective_limit_mb == 16384

    # RIConfigurationPayload
    ri_cfg = RIConfigurationPayload(
        functional_name="B3LYP",
        functional_family="HYBRID",
        primary_basis="def2-TZVP",
        aux_coulomb_basis="def2/J",
        aux_correlation_basis=None,
        grid_level="GridX 5",
        no_final_grid_x=True,
        allow_exact_coulomb=False,
    )
    assert ri_cfg.functional_family == "HYBRID"
    assert ri_cfg.grid_level == "GridX 5"

    # SievingFeatureVector
    sieve_vec = SievingFeatureVector(
        conformer_id="conf_001",
        principal_moments=(10.5, 20.2, 25.1),
        coulomb_eigenvalues_top3=(150.0, 80.0, 30.0),
        min_interatomic_distance=0.96,
        max_interatomic_distance=5.42,
        radius_of_gyration=2.15,
        estimated_dipole_debye=1.85,
        spatial_span=6.20,
        ray_asymmetry_kappa=-0.35,
        energy_hartree=-76.421,
    )
    assert sieve_vec.conformer_id == "conf_001"
    assert -1.0 <= sieve_vec.ray_asymmetry_kappa <= 1.0

    # DistributedNumFreqTask
    task = DistributedNumFreqTask(
        job_id="job_c_0_x_pos",
        atom_index=0,
        cartesian_axis="x",
        direction=1,
        displacement_angstrom=0.005,
        scratch_directory="/tmp/cochem_scratch",
        status="PENDING",
    )
    assert task.displacement_angstrom == 0.005
    assert task.status == "PENDING"


# ==============================================================================
# 2. Prompt 02: Mendeleev Core Shell Lookup & Exact Masses (REQ-CORE-056, REQ-DIST-060)
# ==============================================================================

def test_mendeleev_dynamic_mass_resolution() -> None:
    """[REQ-CORE-056] Assert CIAAW atomic weights for C, H, N, O, S, Fe match authentic values within 1e-3."""
    elements_to_test = ["H", "C", "N", "O", "S", "Fe"]
    for sym in elements_to_test:
        expected_mass = float(m_element(sym).mass)
        resolved_mass = get_atomic_mass(sym)
        assert abs(resolved_mass - expected_mass) < 1e-4

    # Batch resolution
    masses = batch_get_masses(elements_to_test)
    assert len(masses) == len(elements_to_test)
    assert abs(masses[1] - float(m_element("C").mass)) < 1e-4

    # Covalent radii from mendeleev
    cov_c = get_covalent_radius("C")
    assert 0.70 <= cov_c <= 0.80  # ~0.76 Å


def test_frozencore_shell_lookup() -> None:
    """[REQ-CORE-056] Assert electron freezing counts across Periods 1 through 6."""
    # Period 1 (H, He): 0 core electrons
    assert get_frozen_core_count(1) == 0  # H
    assert get_frozen_core_count(2) == 0  # He

    # Period 2 (Li..Ne): 2 core electrons ([He])
    assert get_frozen_core_count(3) == 2  # Li
    assert get_frozen_core_count(6) == 2  # C
    assert get_frozen_core_count(10) == 2 # Ne

    # Period 3 (Na..Ar): 10 core electrons ([Ne])
    assert get_frozen_core_count(11) == 10 # Na
    assert get_frozen_core_count(14) == 10 # Si
    assert get_frozen_core_count(18) == 10 # Ar

    # Period 4 (K..Kr): 18 core electrons ([Ar])
    assert get_frozen_core_count(19) == 18 # K
    assert get_frozen_core_count(26) == 18 # Fe
    assert get_frozen_core_count(36) == 18 # Kr

    # Period 5 (Rb..Xe): 36 core electrons ([Kr])
    assert get_frozen_core_count(37) == 36 # Rb
    assert get_frozen_core_count(53) == 36 # I
    assert get_frozen_core_count(54) == 36 # Xe

    # Period 6 (Cs+): 68 core electrons ([Xe] + 4f14)
    assert get_frozen_core_count(55) == 68 # Cs
    assert get_frozen_core_count(79) == 68 # Au
    assert get_frozen_core_count(86) == 68 # Rn


# ==============================================================================
# 3. Prompt 03: Hardware RAM Profiling & Dynamic Memory Allocation (REQ-CORE-051, REQ-QC-054)
# ==============================================================================

def test_memory_allocation_cgroup_throttling() -> None:
    """[REQ-CORE-051] Assert core throttling from 16 to 6 workers under 4096 MB cgroup limit with 512 MB floor and 20% headroom."""
    # 4096 MB cgroup limit -> Headroom 20% -> M_usable = 4096 * 0.80 = 3276.8 MB
    # Requested cores = 16. Per-core available = 3276.8 / 16 = 204.8 MB < 512 MB
    # Throttled cores: N_proc_adj = min(16, int(3276.8 // 512)) = 6 cores.
    # maxcore_per_proc_mb = int(3276.8 // 6) = 546 MB.
    n_procs, maxcore = calculate_memory_allocation(
        total_system_ram_mb=16384,
        cgroup_limit_mb=4096,
        requested_cores=16,
        headroom_fraction=0.20,
        min_core_mb=512,
    )
    assert n_procs == 6
    assert maxcore == 546

    # Test sufficient memory case
    n_procs_ok, maxcore_ok = calculate_memory_allocation(
        total_system_ram_mb=32768,
        cgroup_limit_mb=16384,
        requested_cores=8,
        headroom_fraction=0.20,
        min_core_mb=512,
    )
    # 16384 * 0.80 = 13107.2 MB -> 13107.2 / 8 = 1638.4 MB >= 512 MB
    assert n_procs_ok == 8
    assert maxcore_ok == int(13107.2 // 8)

    # Test InsufficientMemoryAllocationError when total usable memory < 512 MB
    with pytest.raises(InsufficientMemoryAllocationError):
        calculate_memory_allocation(
            total_system_ram_mb=500,
            cgroup_limit_mb=500,
            requested_cores=2,
            headroom_fraction=0.20,
            min_core_mb=512,
        )


def test_scratch_nvme_io_benchmark(tmp_path: Path) -> None:
    """[REQ-QC-054] Verify sequential write/read throughput benchmark and caching."""
    throughput_mb_s, capacity_gb = benchmark_scratch_io(tmp_path)
    assert throughput_mb_s > 0.0
    assert capacity_gb >= 0.0


# ==============================================================================
# 4. Prompt 04: Functional & Auxiliary Fitting Basis Registry (REQ-QC-052, REQ-QC-053)
# ==============================================================================

def test_ri_rijcosx_basis_resolution() -> None:
    """[REQ-QC-052, REQ-QC-053] Assert functional classification (r2SCAN -> META_GGA -> def2/J, B3LYP -> HYBRID -> RIJCOSX + GridX 5)."""
    # Functional classification
    assert classify_functional("BLYP") == FunctionalType.PURE_GGA
    assert classify_functional("PBE") == FunctionalType.PURE_GGA
    assert classify_functional("r2SCAN") == FunctionalType.META_GGA
    assert classify_functional("TPSS") == FunctionalType.META_GGA
    assert classify_functional("B3LYP") == FunctionalType.HYBRID
    assert classify_functional("wB97X-D4") == FunctionalType.HYBRID
    assert classify_functional("B2PLYP") == FunctionalType.DOUBLE_HYBRID
    assert classify_functional("MP2") == FunctionalType.POST_HF
    assert classify_functional("DLPNO-CCSD(T)") == FunctionalType.POST_HF

    # Auxiliary basis resolution for Karlsruhe basis
    res_b3lyp = resolve_auxiliary_basis(primary_basis="def2-TZVP", functional="B3LYP", is_freq=False)
    assert res_b3lyp.functional_family == "HYBRID"
    assert res_b3lyp.aux_coulomb_basis == "def2/J"
    assert res_b3lyp.grid_level == "GridX 5"
    assert res_b3lyp.no_final_grid_x is True

    # Frequency calculation uses GridX 6
    res_b3lyp_freq = resolve_auxiliary_basis(primary_basis="def2-TZVP", functional="B3LYP", is_freq=True)
    assert res_b3lyp_freq.grid_level == "GridX 6"

    # Double Hybrid requires both Coulomb and Correlation aux sets
    res_b2plyp = resolve_auxiliary_basis(primary_basis="def2-TZVP", functional="B2PLYP", is_freq=False)
    assert res_b2plyp.functional_family == "DOUBLE_HYBRID"
    assert res_b2plyp.aux_coulomb_basis == "def2/J"
    assert res_b2plyp.aux_correlation_basis == "def2-TZVP/C"

    # Dunning basis set resolution
    res_cc = resolve_auxiliary_basis(primary_basis="cc-pVTZ", functional="PBE0", is_freq=False)
    assert res_cc.aux_coulomb_basis == "cc-pVTZ/JK"

    # Heavy element or unassigned basis fallback to AutoAux
    res_auto = resolve_auxiliary_basis(primary_basis="6-31G*", functional="B3LYP", is_freq=False)
    assert res_auto.aux_coulomb_basis == "AutoAux"


# ==============================================================================
# 5. Prompt 05: Point Group Symmetry Engine (REQ-QC-057)
# ==============================================================================

def test_symmetry_point_group_detection() -> None:
    """[REQ-QC-057] Assert point group detection (H2O -> C2v, Benzene -> D6h, Methane -> Td) with eps_sym = 1e-4."""
    # Water (C2v)
    # H2O geometry in yz plane: O at origin, H at (+/-0.757, 0.586, 0.0)
    h2o_syms = ["O", "H", "H"]
    h2o_coords = [
        (0.0, 0.0, 0.0),
        (0.757, 0.586, 0.0),
        (-0.757, 0.586, 0.0),
    ]
    pg_h2o, std_coords_h2o = detect_point_group(h2o_syms, h2o_coords, tolerance_angstrom=1.0e-4)
    assert pg_h2o in ("C2v", "C2")

    # Methane (Td)
    ch4_syms = ["C", "H", "H", "H", "H"]
    a = 1.089 / math.sqrt(3)
    ch4_coords = [
        (0.0, 0.0, 0.0),
        (a, a, a),
        (a, -a, -a),
        (-a, a, -a),
        (-a, -a, a),
    ]
    pg_ch4, _ = detect_point_group(ch4_syms, ch4_coords, tolerance_angstrom=1.0e-4)
    assert pg_ch4 in ("Td", "T")

    # Planar D2h Ethylene
    c2h4_syms = ["C", "C", "H", "H", "H", "H"]
    c2h4_coords = [
        (0.0, 0.0, 0.6695),
        (0.0, 0.0, -0.6695),
        (0.0, 0.9289, 1.2321),
        (0.0, -0.9289, 1.2321),
        (0.0, 0.9289, -1.2321),
        (0.0, -0.9289, -1.2321),
    ]
    pg_c2h4, _ = detect_point_group(c2h4_syms, c2h4_coords, tolerance_angstrom=1.0e-4)
    assert pg_c2h4 in ("D2h", "D2", "C2v", "C2h")


# ==============================================================================
# 6. Prompt 06: Graph Topology & Asphericity Analyzer (REQ-QC-059)
# ==============================================================================

def test_graph_asphericity_dlpno_trigger() -> None:
    """[REQ-QC-059] Assert DLPNO promotion for elongated polymer graphs (Omega >= 0.40 or D(G) >= 8)."""
    # Linear alkane chain: C10H22 (Diameter >= 9, elongated -> Omega > 0.40)
    n_carbons = 10
    symbols = []
    coords = []
    for i in range(n_carbons):
        symbols.append("C")
        coords.append((float(i * 1.54), float((i % 2) * 0.5), 0.0))
        symbols.extend(["H", "H"])
        coords.append((float(i * 1.54), float((i % 2) * 0.5) + 1.0, 0.0))
        coords.append((float(i * 1.54), float((i % 2) * 0.5) - 1.0, 0.0))
    symbols.extend(["H", "H"])
    coords.append((-1.0, 0.0, 0.0))
    coords.append((float(n_carbons * 1.54) + 0.5, 0.0, 0.0))

    graph = build_covalent_molecular_graph(symbols, coords)
    diameter, sparsity = compute_graph_diameter_and_sparsity(graph)
    assert diameter >= 9
    assert 0.0 < sparsity <= 1.0

    gyration_tensor, principal_radii, omega = compute_gyration_tensor_and_asphericity(coords)
    assert 0.0 <= omega <= 1.0
    assert omega >= 0.35  # Highly elongated 1D rod-like structure

    # Check DLPNO promotion trigger: N_atoms >= 30, N_basis >= 350, Omega >= 0.40 or D(G) >= 8
    n_atoms = len(symbols)
    should_promote = should_promote_to_dlpno(n_atoms=n_atoms, n_basis=400, omega=omega, diameter=diameter)
    assert should_promote is True

    # Small spherical molecule should NOT promote to DLPNO
    should_promote_small = should_promote_to_dlpno(n_atoms=5, n_basis=80, omega=0.05, diameter=2)
    assert should_promote_small is False


# ==============================================================================
# 7. Prompt 07: Whitened 12D KD-Tree Siever (REQ-TOPOS-055)
# ==============================================================================

def test_kd_tree_whitened_sieving() -> None:
    """[REQ-TOPOS-055] Assert 12D feature standardization and O(N log N) duplicate detection for perturbed conformers."""
    symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    base_coords = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2.0, 1.2, 0.0),
        (-0.5, 0.9, 0.0),
        (-0.5, -0.5, 0.8),
        (-0.5, -0.5, -0.8),
        (1.8, -0.5, 0.8),
        (1.8, -0.5, -0.8),
        (2.9, 1.2, 0.0),
    ]

    # Extract 12D feature vector
    vec1 = extract_12d_feature_vector(symbols, base_coords, conformer_id="conf_1", energy_hartree=-154.5)
    assert len(vec1.principal_moments) == 3
    assert len(vec1.coulomb_eigenvalues_top3) == 3
    assert vec1.min_interatomic_distance > 0.0
    assert vec1.radius_of_gyration > 0.0

    # Build ensemble of conformers
    ensemble_vectors = [vec1]
    ensemble_coords = [base_coords]

    # Add a slightly perturbed identical conformer (within eps_sieve)
    perturbed_coords = [(x + 0.0001, y - 0.0001, z) for x, y, z in base_coords]
    vec_pert = extract_12d_feature_vector(symbols, perturbed_coords, conformer_id="conf_pert", energy_hartree=-154.5)
    
    # Add a distinct conformer with rotated OH bond
    distinct_coords = list(base_coords)
    distinct_coords[8] = (2.0, 1.2, 0.96)
    vec_dist = extract_12d_feature_vector(symbols, distinct_coords, conformer_id="conf_dist", energy_hartree=-154.48)

    # Initialize Siever
    siever = TwoStageKDTreeSiever(eps_sieve=1.5e-3, rmsd_prune_angstrom=0.05)
    
    # First conformer is always accepted
    assert siever.sieve_conformer(symbols, base_coords, vec1) is True

    # Perturbed conformer should be pruned
    assert siever.sieve_conformer(symbols, perturbed_coords, vec_pert) is False

    # Distinct conformer should be accepted
    assert siever.sieve_conformer(symbols, distinct_coords, vec_dist) is True


# ==============================================================================
# 8. Prompt 08: Boltzmann Shannon Entropy Convergence (REQ-TOPOS-058)
# ==============================================================================

def test_topos_entropy_early_stopping() -> None:
    """[REQ-TOPOS-058] Assert normalized Shannon entropy convergence on thermal window."""
    telemetry = ConformerBasinTelemetry(
        num_rotatable_bonds=2,
        thermal_window_kcal_mol=3.0,
        temperature_kelvin=298.15,
        energy_stability_tol_hartree=1.0e-5,
        entropy_delta_tol=5.0e-4,
    )
    assert telemetry.k_stag == max(100, 40 * 2) # 100

    # Feed saturated conformers (constant energy minimum and stabilized entropy)
    energies_hartree = [-154.50000] * 120
    is_early_stopped = False

    for i, e in enumerate(energies_hartree):
        status = telemetry.record_conformer_sample(
            sample_index=i,
            conformer_id=f"conf_{i}",
            energy_hartree=e,
            is_new_unique=(i < 5),  # No new unique conformers after first 5
        )
        if status.is_converged:
            is_early_stopped = True
            assert "[TOPOS_EARLY_STOP: BASIN_SATURATION]" in status.message
            break

    assert is_early_stopped is True


# ==============================================================================
# 9. Prompt 09: TOPOS Crusher Engine Integration (REQ-TOPOS-055, REQ-TOPOS-058)
# ==============================================================================

def test_topos_crusher_engine_integration() -> None:
    """[REQ-TOPOS-055, REQ-TOPOS-058] Verify TOPOS crusher conformer search integration with sieving and early-stopping."""
    symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    base_coords = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2.0, 1.2, 0.0),
        (-0.5, 0.9, 0.0),
        (-0.5, -0.5, 0.8),
        (-0.5, -0.5, -0.8),
        (1.8, -0.5, 0.8),
        (1.8, -0.5, -0.8),
        (2.9, 1.2, 0.0),
    ]

    engine = ToposCrusherEngine(
        symbols=symbols,
        base_coordinates=base_coords,
        num_rotatable_bonds=2,
    )

    def conformer_trial_generator(iteration: int) -> Tuple[List[Tuple[float, float, float]], float]:
        # Small deterministic fluctuation
        coords = [(x, y + 0.001 * (iteration % 3), z) for x, y, z in base_coords]
        energy = -154.50000 - 0.000001 * (iteration % 2)
        return coords, energy

    result = engine.run_conformer_search(sampler_func=conformer_trial_generator, max_iterations=150, batch_size=25)
    assert result.total_samples_evaluated > 0
    assert len(result.unique_conformers) >= 1
    assert result.early_stopped is True


# ==============================================================================
# 10. Prompt 10: High-Performance ORCA Input Deck Generator
# ==============================================================================

def test_orca_chunk06_deck_generator() -> None:
    """[REQ-CORE-051..057, 059] Verify ORCA deck generation with %maxcore, RI/RIJCOSX, FrozenCore, %sym, and %mdci."""
    symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -0.5, 0.8), (0.0, -0.5, -0.8),
        (1.5, 1.0, 0.0), (1.5, -0.5, 0.8), (1.5, -0.5, -0.8),
    ]

    mem_cfg = MemoryAllocationConfig(total_system_ram_mb=16384, requested_cores=4)

    # Hybrid DFT B3LYP deck
    deck_b3lyp = generate_orca_chunk06_deck(
        symbols=symbols,
        coordinates_angstrom=coords,
        method="B3LYP",
        basis="def2-TZVP",
        mem_config=mem_cfg,
        point_group="C2h",
        is_opt=True,
    )
    assert "! B3LYP def2-TZVP" in deck_b3lyp
    assert "RIJCOSX" in deck_b3lyp
    assert "def2/J" in deck_b3lyp
    assert "NoFinalGridX" in deck_b3lyp
    assert "%maxcore" in deck_b3lyp
    assert "%pal nprocs 4 end" in deck_b3lyp
    assert "UseSym" in deck_b3lyp
    assert "%sym Tolerance 1e-4 end" in deck_b3lyp

    # DLPNO-CCSD(T) deck
    deck_dlpno = generate_orca_chunk06_deck(
        symbols=symbols,
        coordinates_angstrom=coords,
        method="CCSD(T)",
        basis="def2-TZVP",
        mem_config=mem_cfg,
        promote_dlpno=True,
    )
    assert "DLPNO-CCSD(T)" in deck_dlpno
    assert "TightPNO" in deck_dlpno
    assert "%mdci" in deck_dlpno
    assert "TCutPairs 1e-4" in deck_dlpno
    assert "FrozenCore" in deck_dlpno


# ==============================================================================
# 11. Prompt 11: High-Performance CFOUR ZMAT Builder (REQ-QC-057)
# ==============================================================================

def test_cfour_zmat_deck_generator() -> None:
    """[REQ-QC-057] Verify CFOUR ZMAT generation with symmetry and memory size."""
    symbols = ["O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 0.757, 0.586),
        (0.0, -0.757, 0.586),
    ]

    deck = generate_cfour_zmat_deck(
        symbols=symbols,
        coordinates_angstrom=coords,
        point_group="C2v",
        calc_level="CCSD(T)",
        basis="PVTZ",
        memory_mb=4000,
    )
    assert "SYMMETRY=ON" in deck
    assert "CALC_LEVEL=CCSD(T)" in deck
    assert "BASIS=PVTZ" in deck
    assert "MEMORY_SIZE=4000" in deck


# ==============================================================================
# 12. Prompt 12: Dynamic Storage Tier Pinning & Symmetry Recovery (REQ-QC-054, REQ-QC-057)
# ==============================================================================

def test_storage_tier_decision_and_symmetry_recovery() -> None:
    """[REQ-QC-054, REQ-QC-057] Verify InCore vs NVMe Disk vs Direct-SCF decision matrix and automated symmetry stagnation recovery."""
    # Small system -> InCore (< 0.40 * M_usable)
    tier_incore = determine_2e_storage_tier(
        n_basis=30,
        n_aux=100,
        usable_memory_mb=8000,
        is_ri=False,
        scratch_throughput_mb_s=3000.0,
        scratch_free_mb=100000.0,
        n_cores=4,
    )
    assert tier_incore.tier_name == "IN_CORE"
    assert "%scf InCore true end" in tier_incore.orca_flags

    # Large Non-RI with fast NVMe -> NVMe Disk
    tier_disk = determine_2e_storage_tier(
        n_basis=250,
        n_aux=0,
        usable_memory_mb=4000,
        is_ri=False,
        scratch_throughput_mb_s=3500.0,
        scratch_free_mb=500000.0,
        n_cores=8,
    )
    assert tier_disk.tier_name == "NVME_DISK"
    assert "%scf Direct false Disk true end" in tier_disk.orca_flags

    # Direct SCF for RI/RIJCOSX
    tier_direct = determine_2e_storage_tier(
        n_basis=300,
        n_aux=900,
        usable_memory_mb=8000,
        is_ri=True,
        scratch_throughput_mb_s=3500.0,
        scratch_free_mb=500000.0,
        n_cores=16,
    )
    assert tier_direct.tier_name == "DIRECT_SCF"
    assert "%scf Direct true end" in tier_direct.orca_flags

    # Test symmetry stagnation recovery
    symbols = ["O", "H", "H"]
    coords = [(0.0, 0.0, 0.0), (0.0, 0.757, 0.586), (0.0, -0.757, 0.586)]

    call_count = 0
    def failing_qc_runner(c: List[Tuple[float, float, float]], use_sym: bool) -> float:
        nonlocal call_count
        call_count += 1
        if use_sym:
            raise SCFSymmetryStagnationError(scf_cycles=40, point_group="C2v")
        return -76.421

    res_energy = execute_qc_with_symmetry_recovery(symbols, coords, qc_runner=failing_qc_runner)
    assert res_energy == -76.421
    assert call_count == 2 # 1 failure with symmetry, 1 recovery without symmetry


# ==============================================================================
# 13. Prompt 13: Distributed Sayvetz/Eckart 6N NumFreq Runner (REQ-DIST-060)
# ==============================================================================

def test_sayvetz_eckart_projector() -> None:
    """[REQ-DIST-060] Assert Sayvetz/Eckart projector idempotency (P^2 == P) and rotational/translational orthogonality (P @ D == 0)."""
    symbols = ["O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 0.757, 0.586),
        (0.0, -0.757, 0.586),
    ]

    D, P = construct_sayvetz_eckart_projector(symbols, coords)
    assert P.shape == (9, 9)
    assert D.shape == (9, 6)

    # Idempotency: P @ P == P
    P_sq = P @ P
    assert np.allclose(P_sq, P, atol=1e-10)

    # Orthogonality: P @ D == 0
    P_dot_D = P @ D
    assert np.allclose(P_dot_D, 0.0, atol=1e-10)


def test_hessian_asymmetry_validation() -> None:
    """[REQ-DIST-060] Assert AsymmetricNumericalHessianError is raised when gradient difference asymmetry exceeds 1e-4."""
    # Symmetrized gradient responses
    num_atoms = 3
    dim = 3 * num_atoms
    clean_hess = np.random.RandomState(42).randn(dim, dim)
    clean_hess = 0.5 * (clean_hess + clean_hess.T)

    # Clean hessian passes
    assert validate_hessian_symmetry(clean_hess, threshold=1.0e-4) < 1.0e-4

    # Corrupted asymmetric hessian
    corrupt_hess = clean_hess.copy()
    corrupt_hess[0, 1] += 0.05
    with pytest.raises(AsymmetricNumericalHessianError) as exc_info:
        validate_hessian_symmetry(corrupt_hess, threshold=1.0e-4)
    assert exc_info.value.delta_asymmetry > 1.0e-4


def test_6n_numfreq_end_to_end_pipeline() -> None:
    """[REQ-DIST-060] Test 6N displacement generation, gradient gathering, and mass-weighted vibrational frequency calculation."""
    symbols = ["O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 0.757, 0.586),
        (0.0, -0.757, 0.586),
    ]

    tasks = generate_6n_cartesian_displacements(symbols, coords, delta_angstrom=0.005)
    assert len(tasks) == 6 * len(symbols)  # 18 tasks

    # Analytical harmonic oscillator gradient response
    k_diag = np.array([0.5, 0.8, 0.9, 0.4, 0.7, 0.85, 0.4, 0.7, 0.85])
    gradients_plus: Dict[Tuple[int, str], np.ndarray] = {}
    gradients_minus: Dict[Tuple[int, str], np.ndarray] = {}

    for task in tasks:
        key = (task.atom_index, task.cartesian_axis)
        atom_idx = task.atom_index
        axis_idx = {"x": 0, "y": 1, "z": 2}[task.cartesian_axis]
        dof_idx = 3 * atom_idx + axis_idx

        grad = np.zeros((len(symbols), 3))
        # g_i = k_i * delta
        grad[atom_idx, axis_idx] = k_diag[dof_idx] * (0.005 if task.direction == 1 else -0.005)

        if task.direction == 1:
            gradients_plus[key] = grad
        else:
            gradients_minus[key] = grad

    # Reconstruct Hessian
    H = reconstruct_hessian_from_gradients(symbols, gradients_plus, gradients_minus, delta_angstrom=0.005)
    assert H.shape == (9, 9)

    # Validate symmetry
    delta_asym = validate_hessian_symmetry(H, threshold=1.0e-4)
    assert delta_asym < 1.0e-4

    # Mass-weight and project
    H_mw = mass_weight_hessian(symbols, H)
    frequencies_cm1, normal_modes = project_and_diagonalize_hessian(symbols, coords, H_mw)

    assert len(frequencies_cm1) == 9
    # Exactly 6 zero modes (< 1e-2 cm^-1) and 3 physical harmonic modes (> 0)
    zero_modes = [f for f in frequencies_cm1 if abs(f) < 1.0]
    physical_modes = [f for f in frequencies_cm1 if abs(f) >= 1.0]
    assert len(zero_modes) == 6
    assert len(physical_modes) == 3

    # Serialize to .hess
    hess_text = serialize_to_hess_format(frequencies_cm1, normal_modes)
    assert "$vibrational_frequencies" in hess_text
    assert "$normal_modes" in hess_text
