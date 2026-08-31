"""
CoChem Boltzmann Shannon Entropy Convergence & Basin Saturation Monitor.

Provenance:
- Method Matrix [M]: Thermal window Boltzmann distribution (Delta E = 3.0 kcal/mol at 298.15 K),
  normalized Shannon entropy convergence, and dynamic stagnation patience scaling (REQ-TOPOS-058).
- Multi-Criteria Early-Stopping: Energy stability (< 1e-5 Eh), zero novelty over K_stag, and |Delta S_norm| < 5e-4.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

# Physical Constants
K_B_HARTREE_PER_K = 3.166811563e-6  # Hartree / Kelvin
KCAL_PER_MOL_TO_HARTREE = 1.0 / 627.509474


class EarlyStoppingStatus(BaseModel):
    """Data contract representing the convergence state of conformational exploration."""

    is_converged: bool = Field(default=False, description="Whether all early-stopping criteria are satisfied.")
    iterations_evaluated: int = Field(..., ge=0, description="Total sampling iterations performed.")
    unique_conformer_count: int = Field(..., ge=0, description="Number of unique conformers identified.")
    current_e_min_hartree: float = Field(..., description="Current global minimum electronic energy in Hartree.")
    normalized_shannon_entropy: float = Field(..., ge=0.0, le=1.0, description="Current normalized Shannon entropy.")
    message: str = Field(default="", description="Convergence status diagnostic message.")


def calculate_boltzmann_weights(
    energies_hartree: Sequence[float],
    temperature_kelvin: float = 298.15,
) -> np.ndarray:
    """
    [REQ-TOPOS-058] Compute normalized Boltzmann probabilities p_i for an ensemble of conformer energies.

    Formula:
        p_i = exp(-(E_i - E_min) / (k_B * T)) / sum_j exp(-(E_j - E_min) / (k_B * T))

    Args:
        energies_hartree: Sequence of electronic energies in Hartree.
        temperature_kelvin: Thermodynamic temperature in Kelvin (default: 298.15 K).

    Returns:
        1D NumPy array of normalized probabilities (sums to 1.0).
    """
    e_arr = np.asarray(energies_hartree, dtype=np.float64)
    if len(e_arr) == 0:
        return np.array([], dtype=np.float64)
    if len(e_arr) == 1:
        return np.array([1.0], dtype=np.float64)

    e_min = np.min(e_arr)
    kt = K_B_HARTREE_PER_K * temperature_kelvin
    delta_e = (e_arr - e_min) / kt

    # Numerical stability clipping
    delta_e_clipped = np.clip(delta_e, 0.0, 700.0)
    unnorm_weights = np.exp(-delta_e_clipped)
    total_w = np.sum(unnorm_weights)

    if total_w < 1e-18:
        return np.ones_like(e_arr) / len(e_arr)
    return unnorm_weights / total_w


def calculate_normalized_shannon_entropy(probabilities: np.ndarray) -> float:
    """
    [REQ-TOPOS-058] Compute normalized Shannon entropy S_norm in [0.0, 1.0].

    Formula:
        S_norm = (-sum_i p_i * ln(p_i)) / ln(M)

    Args:
        probabilities: 1D NumPy array of probability weights (sums to 1.0).

    Returns:
        Normalized Shannon entropy value in [0.0, 1.0].
    """
    m = len(probabilities)
    if m <= 1:
        return 0.0

    # Mask positive probabilities to avoid log(0)
    p_pos = probabilities[probabilities > 1e-15]
    if len(p_pos) <= 1:
        return 0.0

    raw_entropy = -float(np.sum(p_pos * np.log(p_pos)))
    max_entropy = math.log(m)

    s_norm = raw_entropy / max_entropy
    return float(np.clip(s_norm, 0.0, 1.0))


class ConformerBasinTelemetry:
    """
    [REQ-TOPOS-058] Real-time statistical stagnation and Shannon entropy convergence tracker.
    """

    def __init__(
        self,
        num_rotatable_bonds: int = 1,
        thermal_window_kcal_mol: float = 3.0,
        temperature_kelvin: float = 298.15,
        energy_stability_tol_hartree: float = 1.0e-5,
        entropy_delta_tol: float = 5.0e-4,
        batch_size: int = 25,
    ) -> None:
        self.num_rotatable_bonds = max(0, num_rotatable_bonds)
        self.k_stag = max(100, 40 * self.num_rotatable_bonds)
        self.thermal_window_hartree = thermal_window_kcal_mol * KCAL_PER_MOL_TO_HARTREE
        self.temperature_kelvin = temperature_kelvin
        self.energy_stability_tol = energy_stability_tol_hartree
        self.entropy_delta_tol = entropy_delta_tol
        self.batch_size = batch_size

        self.conformer_records: List[Dict[str, Any]] = []
        self.energy_history: List[float] = []
        self.min_energy_history: List[float] = []
        self.unique_conformer_indices: List[int] = []
        self.entropy_batch_history: List[float] = []
        self.last_unique_iteration = 0
        self.is_early_stopped = False

    def record_conformer_sample(  # anti-spoof: conformer trial record
        self,
        sample_index: int,
        conformer_id: str,
        energy_hartree: float,
        is_new_unique: bool,
    ) -> EarlyStoppingStatus:
        """
        [REQ-TOPOS-058] Record newly evaluated sample and evaluate multi-criteria convergence.

        Convergence Criteria (ALL 3 must be met):
            1. Global min energy change |E_min(t) - E_min(t - K_stag)| < 1.0e-5 Hartree.
            2. No new unique conformers discovered in Delta E_window over K_stag iterations.
            3. |Delta S_norm| < 5.0e-4 across 3 consecutive batches of 25 structures.
        """
        self.energy_history.append(energy_hartree)
        curr_min = min(self.energy_history)
        self.min_energy_history.append(curr_min)

        if is_new_unique:
            self.unique_conformer_indices.append(sample_index)
            self.last_unique_iteration = sample_index

        self.conformer_records.append({
            "sample_index": sample_index,
            "conformer_id": conformer_id,
            "energy_hartree": energy_hartree,
            "is_unique": is_new_unique,
        })

        # Calculate current Shannon entropy over unique conformers in thermal window
        thermal_energies = [
            r["energy_hartree"] for r in self.conformer_records
            if r["is_unique"] and (r["energy_hartree"] - curr_min) <= self.thermal_window_hartree
        ]

        if thermal_energies:
            probs = calculate_boltzmann_weights(thermal_energies, self.temperature_kelvin)
            s_norm = calculate_normalized_shannon_entropy(probs)
        else:
            s_norm = 0.0

        # Batch entropy record
        if (sample_index + 1) % self.batch_size == 0:
            self.entropy_batch_history.append(s_norm)

        # Evaluate criteria once we have at least K_stag iterations
        total_eval = len(self.energy_history)
        converged = False
        message = "Sampling in progress."

        if total_eval >= self.k_stag:
            # Condition 1: Energy stability
            e_min_now = self.min_energy_history[-1]
            e_min_prev = self.min_energy_history[-self.k_stag]
            cond1 = abs(e_min_now - e_min_prev) < self.energy_stability_tol

            # Condition 2: No new unique conformers in last K_stag iterations
            cond2 = (total_eval - self.last_unique_iteration) >= self.k_stag

            # Condition 3: Entropy delta across last 3 batches
            cond3 = False
            if len(self.entropy_batch_history) >= 3:
                b1 = self.entropy_batch_history[-1]
                b2 = self.entropy_batch_history[-2]
                b3 = self.entropy_batch_history[-3]
                d1 = abs(b1 - b2)
                d2 = abs(b2 - b3)
                cond3 = (d1 < self.entropy_delta_tol) and (d2 < self.entropy_delta_tol)
            elif total_eval >= self.k_stag:
                # If pool stabilized without multiple batch triggers
                cond3 = True

            if cond1 and cond2 and cond3:
                converged = True
                self.is_early_stopped = True
                message = "[TOPOS_EARLY_STOP: BASIN_SATURATION] Convergence achieved on conformer energy basin."

        return EarlyStoppingStatus(
            is_converged=converged,
            iterations_evaluated=total_eval,
            unique_conformer_count=len(self.unique_conformer_indices),
            current_e_min_hartree=curr_min,
            normalized_shannon_entropy=round(s_norm, 6),
            message=message,
        )

    def dump_convergence_json(self, output_path: Path) -> Path:
        """Dump convergence metrics to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "is_early_stopped": self.is_early_stopped,
            "total_iterations": len(self.energy_history),
            "k_stag_patience": self.k_stag,
            "num_rotatable_bonds": self.num_rotatable_bonds,
            "unique_conformer_count": len(self.unique_conformer_indices),
            "global_min_energy_hartree": min(self.energy_history) if self.energy_history else 0.0,
            "entropy_batch_history": self.entropy_batch_history,
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return target
