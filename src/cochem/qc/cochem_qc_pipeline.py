"""
CoChem Mass-Weighted Saddle-Point Escaping & Execution Pipeline.

Provenance:
- Method Matrix [M]: Bifurcated displacement delta = 0.08 Å along imaginary normal modes,
  MAX_PIVOT_CYCLES = 3, dynamic Mendeleev mass-weighting (REQ-QC-042).
- Derived Relations [D]: Delta R_{i, alpha} = Q_{1, i, alpha} / sqrt(m_i) Cartesian projection.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_elements import batch_get_masses
from cochem.core.cochem_exceptions import CudaKernelExecutionError, SaddlePointEscapeError
from cochem.qc.cochem_qc_parser import ParsedQcOutput


class VibrationalStateModel(BaseModel):
    """Data contract representing the vibrational state of a molecular stationary point."""

    is_linear: bool = Field(default=False)
    num_atoms: int = Field(..., ge=1)
    expected_vibrational_dof: int = Field(..., ge=0)
    frequencies_cm1: List[float] = Field(default_factory=list)
    lowest_internal_frequency: float
    is_minimum: bool = Field(default=True)
    pivot_iteration: int = Field(default=0, ge=0)


def compute_mass_weighted_displacement(
    symbols: Sequence[str],
    normal_mode_q1: np.ndarray,
    step_delta_angstrom: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REQ-QC-042] Compute mass-weighted Cartesian unit displacement vector and +/- delta trial displacements.

    Formula:
        Delta R_{i, alpha} = Q_{1, i, alpha} / sqrt(m_i)
        U_hat = Delta R / ||Delta R||_2
        Delta R_plus = +delta * U_hat
        Delta R_minus = -delta * U_hat

    Args:
        symbols: Sequence of chemical element symbols.
        normal_mode_q1: 1D array of 3N components or (N, 3) matrix representing Q_1 eigenvector.
        step_delta_angstrom: Displacement magnitude delta in Ångströms (default: 0.08 Å [M]).

    Returns:
        Tuple of (delta_R_plus, delta_R_minus) as (N_atoms, 3) arrays in Ångströms.
    """
    masses = batch_get_masses(list(symbols))  # (N_atoms,) in amu
    num_atoms = len(symbols)

    q1_reshaped = np.asarray(normal_mode_q1, dtype=np.float64).reshape((num_atoms, 3))

    # Apply mass weighting: Delta R_{i, alpha} = Q_{1, i, alpha} / sqrt(m_i)
    sqrt_masses = np.sqrt(masses)[:, np.newaxis]  # (N, 1)
    delta_r = q1_reshaped / sqrt_masses

    # Compute Euclidean norm over all 3N components
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

    Args:
        symbols: Sequence of chemical element symbols.
        initial_coordinates: Starting Cartesian geometry in Å.
        initial_frequencies: Vibrational harmonic modes in cm^-1.
        initial_normal_mode_q1: Imaginary mode eigenvector Q1.
        energy_eval_func: Callable evaluating single-point energy E(R).
        opt_func: Callable executing re-optimization, returning (coords, freqs, Q1).
        max_pivot_cycles: Maximum allowable pivot cycles (default: 3).
        delta_angstrom: Normal mode displacement magnitude (default: 0.08 Å).

    Returns:
        Tuple of (optimized_coordinates, VibrationalStateModel).

    Raises:
        SaddlePointEscapeError: If optimization fails to escape saddle point after max_pivot_cycles.
    """
    current_coords = np.array(initial_coordinates, dtype=np.float64)
    current_freqs = list(initial_frequencies)
    current_q1 = initial_normal_mode_q1
    num_atoms = len(symbols)

    lowest_freq = min(current_freqs) if current_freqs else 0.0
    iteration = 0

    while lowest_freq < -15.0 and iteration < max_pivot_cycles:
        iteration += 1

        # Compute mass-weighted +/- bifurcations
        delta_plus, delta_minus = compute_mass_weighted_displacement(
            symbols=symbols,
            normal_mode_q1=current_q1,
            step_delta_angstrom=delta_angstrom,
        )

        r_plus = current_coords + delta_plus
        r_minus = current_coords + delta_minus

        # Evaluate energy along both trial branches
        e_plus = energy_eval_func([tuple(row) for row in r_plus])
        e_minus = energy_eval_func([tuple(row) for row in r_minus])

        # Select branch of lowest single point energy
        chosen_trial = r_plus if e_plus <= e_minus else r_minus

        # Dispatch re-optimization
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
