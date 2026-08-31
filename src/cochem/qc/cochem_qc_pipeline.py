"""
CoChem Dynamic Storage Tier Pinning & QC Pipeline Execution Orchestrator.

Provenance:
- Method Matrix [M]: 2e-ERI storage tier decision matrix (InCore vs NVMe Disk vs Direct-SCF) (REQ-QC-054),
  automated symmetry stagnation recovery (REQ-QC-057), and mass-weighted saddle-point escape (REQ-QC-042).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_elements import batch_get_masses
from cochem.core.cochem_exceptions import CudaKernelExecutionError, SaddlePointEscapeError
from cochem.core.models import SCFSymmetryStagnationError


class StorageTierDecision(BaseModel):
    """Data contract representing 2e-ERI integral storage tier selection."""

    tier_name: Literal["IN_CORE", "NVME_DISK", "DIRECT_SCF"]
    m_4center_mb: float
    m_3center_mb: float
    orca_flags: List[str]
    description: str


class VibrationalStateModel(BaseModel):
    """Data contract representing the vibrational state of a molecular stationary point."""

    is_linear: bool = Field(default=False)
    num_atoms: int = Field(..., ge=1)
    expected_vibrational_dof: int = Field(..., ge=0)
    frequencies_cm1: List[float] = Field(default_factory=list)
    lowest_internal_frequency: float
    is_minimum: bool = Field(default=True)
    pivot_iteration: int = Field(default=0, ge=0)


def determine_2e_storage_tier(
    n_basis: int,
    n_aux: int,
    usable_memory_mb: float,
    is_ri: bool = False,
    scratch_throughput_mb_s: float = 2500.0,
    scratch_free_mb: float = 50000.0,
    n_cores: int = 1,
) -> StorageTierDecision:
    """
    [REQ-QC-054] Determine optimal 2e-ERI caching and integral storage tier.

    Tensor Estimates:
        M_4center_MB = (N_basis^4 * 8) / (8 * 1024^2) = N_basis^4 / 1024^2
        M_3center_MB = (N_basis^2 * N_aux * 8) / (1024^2)

    Decision Rules:
        - Branch 1 (InCore): M_4center_MB <= 0.40 * M_usable -> %scf InCore true end
        - Branch 2 (NVMe Disk): Non-RI and M_4center > 0.40 * M_usable and D_free >= 3 * M_4center and R_seq >= 2500 MB/s and N_cores <= 8
          -> %scf Direct false Disk true end
        - Branch 3 (Direct-SCF): RI or R_seq < 2500 MB/s or N_cores >= 16 -> %scf Direct true end

    Args:
        n_basis: Number of contracted primary basis functions.
        n_aux: Number of auxiliary fitting basis functions.
        usable_memory_mb: Effective system memory available in MB.
        is_ri: Whether density fitting / RI approximation is enabled.
        scratch_throughput_mb_s: Measured sequential I/O speed in MB/s.
        scratch_free_mb: Available free scratch space in MB.
        n_cores: Allocated CPU core count.

    Returns:
        StorageTierDecision instance.
    """
    m_4center_mb = float(n_basis ** 4) / float(1024 ** 2)
    m_3center_mb = float(n_basis ** 2 * n_aux * 8) / float(1024 ** 2)

    # Branch 1: InCore
    if m_4center_mb <= 0.40 * usable_memory_mb:
        return StorageTierDecision(
            tier_name="IN_CORE",
            m_4center_mb=round(m_4center_mb, 2),
            m_3center_mb=round(m_3center_mb, 2),
            orca_flags=["%scf InCore true end"],
            description="Four-center 2e-ERIs fit entirely within RAM cache (< 40% usable memory).",
        )

    # Branch 2: Fast NVMe Disk (Non-RI only)
    if (
        not is_ri
        and (m_4center_mb > 0.40 * usable_memory_mb)
        and (scratch_free_mb >= 3.0 * m_4center_mb)
        and (scratch_throughput_mb_s >= 2500.0)
        and (n_cores <= 8)
    ):
        return StorageTierDecision(
            tier_name="NVME_DISK",
            m_4center_mb=round(m_4center_mb, 2),
            m_3center_mb=round(m_3center_mb, 2),
            orca_flags=["%scf Direct false Disk true end"],
            description="Caching four-center integrals to ultra-fast NVMe scratch (> 2500 MB/s).",
        )

    # Branch 3: Direct-SCF
    return StorageTierDecision(
        tier_name="DIRECT_SCF",
        m_4center_mb=round(m_4center_mb, 2),
        m_3center_mb=round(m_3center_mb, 2),
        orca_flags=["%scf Direct true end"],
        description="Evaluating 2e-ERIs on the fly (Direct-SCF / RIJCOSX).",
    )


def execute_qc_with_symmetry_recovery(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    qc_runner: Callable[[List[Tuple[float, float, float]], bool], float],
    perturbation_angstrom: float = 1.0e-3,
) -> float:
    """
    [REQ-QC-057] Execute quantum chemistry calculation with automated symmetry stagnation fallback.

    If SCFSymmetryStagnationError occurs:
        1. Perturb geometry by delta = 1e-3 Å along lowest non-symmetric mode.
        2. Set use_sym=False.
        3. Re-execute calculation.

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        qc_runner: Callable taking (coordinates, use_sym) -> energy in Hartree.
        perturbation_angstrom: Coordinate perturbation magnitude in Å (default 1e-3 Å).

    Returns:
        Calculated electronic energy in Hartree.
    """
    coords = [tuple(row) for row in coordinates_angstrom]

    try:
        return qc_runner(coords, True)
    except SCFSymmetryStagnationError:
        # Automatic recovery: perturb coordinates slightly and disable symmetry
        perturbed = []
        for idx, (x, y, z) in enumerate(coords):
            # Asymmetric perturbation vector
            dx = perturbation_angstrom * math.sin(idx + 1)
            dy = perturbation_angstrom * math.cos(idx + 1)
            dz = perturbation_angstrom * (1.0 if idx % 2 == 0 else -1.0)
            perturbed.append((x + dx, y + dy, z + dz))

        return qc_runner(perturbed, False)


def compute_mass_weighted_displacement(
    symbols: Sequence[str],
    normal_mode_q1: np.ndarray,
    step_delta_angstrom: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REQ-QC-042] Compute mass-weighted Cartesian unit displacement vector and +/- delta trial displacements.
    """
    masses = batch_get_masses(list(symbols))
    num_atoms = len(symbols)

    q1_reshaped = np.asarray(normal_mode_q1, dtype=np.float64).reshape((num_atoms, 3))
    sqrt_masses = np.sqrt(masses)[:, np.newaxis]
    delta_r = q1_reshaped / sqrt_masses

    norm_delta_r = np.linalg.norm(delta_r)
    if norm_delta_r < 1e-12:
        u_hat = np.zeros_like(delta_r)
    else:
        u_hat = delta_r / norm_delta_r

    delta_plus = step_delta_angstrom * u_hat
    delta_minus = -step_delta_angstrom * u_hat

    return delta_plus, delta_minus


def escape_saddle_point_pipeline(
    symbols: Sequence[str],
    initial_coordinates: Sequence[Tuple[float, float, float]],
    initial_frequencies: Sequence[float],
    initial_normal_mode_q1: np.ndarray,
    energy_eval_func: Callable[[Sequence[Tuple[float, float, float]]], float],
    opt_func: Callable[[Sequence[Tuple[float, float, float]]], Tuple[Sequence[Tuple[float, float, float]], Sequence[float], np.ndarray]],
    max_pivot_cycles: int = 3,
    delta_angstrom: float = 0.08,
) -> Tuple[List[Tuple[float, float, float]], VibrationalStateModel]:
    """
    [REQ-QC-042] Execute automated mass-weighted saddle-point escape loop.
    """
    current_coords = np.array(initial_coordinates, dtype=np.float64)
    current_freqs = list(initial_frequencies)
    current_q1 = initial_normal_mode_q1
    num_atoms = len(symbols)

    lowest_freq = min(current_freqs) if current_freqs else 0.0
    iteration = 0

    while lowest_freq < -15.0 and iteration < max_pivot_cycles:
        iteration += 1

        delta_plus, delta_minus = compute_mass_weighted_displacement(
            symbols=symbols,
            normal_mode_q1=current_q1,
            step_delta_angstrom=delta_angstrom,
        )

        r_plus = current_coords + delta_plus
        r_minus = current_coords + delta_minus

        e_plus = energy_eval_func([tuple(row) for row in r_plus])
        e_minus = energy_eval_func([tuple(row) for row in r_minus])

        chosen_trial = r_plus if e_plus <= e_minus else r_minus

        opt_coords_seq, new_freqs_seq, new_q1 = opt_func([tuple(row) for row in chosen_trial])
        current_coords = np.array(opt_coords_seq, dtype=np.float64)
        current_freqs = list(new_freqs_seq)
        current_q1 = new_q1
        lowest_freq = min(current_freqs) if current_freqs else 0.0

    is_min = lowest_freq >= -15.0
    if not is_min:
        raise SaddlePointEscapeError(
            pivot_cycles=iteration,
            lowest_frequency=lowest_freq,
        )

    expected_dof = max(0, 3 * num_atoms - 6)
    state = VibrationalStateModel(
        is_linear=False,
        num_atoms=num_atoms,
        expected_vibrational_dof=expected_dof,
        frequencies_cm1=current_freqs,
        lowest_internal_frequency=lowest_freq,
        is_minimum=is_min,
        pivot_iteration=iteration,
    )

    final_coords = [tuple(row) for row in current_coords]
    return final_coords, state
