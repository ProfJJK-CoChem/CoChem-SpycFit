"""
CoChem Automated Marcus Electron Transfer Reorganization Pipeline (REQ-QC-082).

Provenance:
- Method Matrix [M]: Nelson 4-point single-point and geometry optimization state machine for inner-sphere (lambda_inner)
  and outer-sphere (lambda_outer) reorganization energies, Marcus activation barrier (Delta G^dagger), and rate constant (k_ET).
- Derived Relations [D]: CODATA 2022 constants (1 Hartree = 27.211386245988 eV = 627.5094740631 kcal/mol).
- Empirical [E]: Classical parabolic Marcus equation and dielectric continuum outer-sphere corrections.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from cochem.concurrency.atomic_file_lock import AtomicFileLock
from cochem.qc.exceptions import MarcusConvergenceError
from cochem.qc.models import MarcusReorganizationResult

# CODATA 2022 Unit Conversions & Physical Constants
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_KCAL_MOL: float = 627.5094740631
EV_TO_KCAL_MOL: float = 23.060548867
K_BOLTZMANN_EV_K: float = 8.617333262e-5
H_BAR_EV_S: float = 6.582119569e-16
CM1_TO_EV: float = 1.239841984e-4
VACUUM_PERMITTIVITY_EV_ANG: float = 14.399645 # e^2 / (4 * pi * epsilon_0) in eV * Å


class MarcusStatePoints(BaseModel):
    """Data contract representing Nelson 4-point electronic energies."""
    model_config = ConfigDict(frozen=True)

    e0_r0_hartree: float
    ep_r0_hartree: float
    ep_rp_hartree: float
    e0_rp_hartree: float


def compute_outer_sphere_reorganization(
    radius_donor_angstrom: float,
    radius_acceptor_angstrom: float,
    distance_da_angstrom: float,
    eps_optical: float = 1.78,
    eps_static: float = 78.4,
) -> float:
    """
    [M][D] Compute Marcus dielectric continuum outer-sphere reorganization energy in eV.
    lambda_outer = (e^2 / 4 pi eps_0) * (1/(2 r_D) + 1/(2 r_A) - 1/R_DA) * (1/eps_opt - 1/eps_stat)
    """
    if radius_donor_angstrom <= 0 or radius_acceptor_angstrom <= 0 or distance_da_angstrom <= 0:
        raise ValueError("Molecular radii and donor-acceptor distance must be positive.")

    geom_factor = (1.0 / (2.0 * radius_donor_angstrom)) + (1.0 / (2.0 * radius_acceptor_angstrom)) - (1.0 / distance_da_angstrom)
    dielectric_factor = (1.0 / eps_optical) - (1.0 / eps_static)
    lambda_outer = VACUUM_PERMITTIVITY_EV_ANG * geom_factor * dielectric_factor
    return max(0.0, float(lambda_outer))


def compute_marcus_reorganization(
    e_neutral_at_neutral_geom_hartree: float,
    e_charged_at_neutral_geom_hartree: float,
    e_charged_at_charged_geom_hartree: float,
    e_neutral_at_charged_geom_hartree: float,
    lambda_outer_ev: Optional[float] = None,
    delta_g_0_ev: Optional[float] = None,
    electronic_coupling_ev: Optional[float] = None,
    temperature_k: float = 298.15,
) -> MarcusReorganizationResult:
    """
    [M][D] Compute Nelson 4-point Marcus electron-transfer reorganization energy and kinetics (REQ-QC-082).
    """
    # 1. Evaluate individual energy differences in Hartree
    d_e1_hartree = e_charged_at_neutral_geom_hartree - e_charged_at_charged_geom_hartree
    d_e2_hartree = e_neutral_at_charged_geom_hartree - e_neutral_at_neutral_geom_hartree

    if d_e1_hartree < -1e-6:
        raise MarcusConvergenceError(
            step_name="lambda_1_evaluation",
            reason=(
                f"Negative reorganization energy lambda_1 = {d_e1_hartree * HARTREE_TO_EV:.4f} eV. "
                f"E_charged(R_0)={e_charged_at_neutral_geom_hartree:.6f} < "
                f"E_charged(R_charged)={e_charged_at_charged_geom_hartree:.6f} violating variational minimum."
            ),
        )

    if d_e2_hartree < -1e-6:
        raise MarcusConvergenceError(
            step_name="lambda_2_evaluation",
            reason=(
                f"Negative reorganization energy lambda_2 = {d_e2_hartree * HARTREE_TO_EV:.4f} eV. "
                f"E_neutral(R_charged)={e_neutral_at_charged_geom_hartree:.6f} < "
                f"E_neutral(R_0)={e_neutral_at_neutral_geom_hartree:.6f} violating variational minimum."
            ),
        )

    lambda_1_ev = max(0.0, d_e1_hartree * HARTREE_TO_EV)
    lambda_2_ev = max(0.0, d_e2_hartree * HARTREE_TO_EV)
    lambda_inner_ev = lambda_1_ev + lambda_2_ev
    lambda_inner_kcal = lambda_inner_ev * EV_TO_KCAL_MOL

    l_outer = max(0.0, float(lambda_outer_ev)) if lambda_outer_ev is not None else 0.0
    total_reorg_ev = lambda_inner_ev + l_outer

    if total_reorg_ev <= 1e-9:
        raise MarcusConvergenceError(
            step_name="total_reorganization",
            reason="Total reorganization energy lambda_total <= 0 is unphysical.",
        )

    # 2. Driving force Delta G^0
    if delta_g_0_ev is not None:
        dg_ev = float(delta_g_0_ev)
    else:
        dg_ev = (e_charged_at_charged_geom_hartree - e_neutral_at_neutral_geom_hartree) * HARTREE_TO_EV

    # 3. Marcus activation barrier Delta G^dagger = (Delta G^0 + lambda)^2 / (4 * lambda)
    barrier_numerator = dg_ev + total_reorg_ev
    marcus_barrier_ev = (barrier_numerator ** 2) / (4.0 * total_reorg_ev)

    # 4. Semiclassical Marcus electron transfer rate constant k_ET
    rate_s_inv: Optional[float] = None
    if electronic_coupling_ev is not None and electronic_coupling_ev > 0.0:
        k_b_t = K_BOLTZMANN_EV_K * temperature_k
        prefactor = (2.0 * math.pi / H_BAR_EV_S) * (electronic_coupling_ev ** 2)
        fc_factor = 1.0 / math.sqrt(4.0 * math.pi * total_reorg_ev * k_b_t)
        exp_factor = math.exp(-marcus_barrier_ev / k_b_t)
        rate_s_inv = prefactor * fc_factor * exp_factor

    return MarcusReorganizationResult(
        e_neutral_at_neutral_geom_hartree=e_neutral_at_neutral_geom_hartree,
        e_charged_at_neutral_geom_hartree=e_charged_at_neutral_geom_hartree,
        e_charged_at_charged_geom_hartree=e_charged_at_charged_geom_hartree,
        e_neutral_at_charged_geom_hartree=e_neutral_at_charged_geom_hartree,
        lambda_1_ev=round(lambda_1_ev, 5),
        lambda_2_ev=round(lambda_2_ev, 5),
        lambda_inner_ev=round(lambda_inner_ev, 5),
        lambda_inner_kcal_mol=round(lambda_inner_kcal, 4),
        lambda_outer_ev=round(l_outer, 5) if lambda_outer_ev is not None else None,
        total_reorganization_energy_ev=round(total_reorg_ev, 5),
        gibbs_free_energy_delta_ev=round(dg_ev, 5),
        marcus_activation_barrier_ev=round(marcus_barrier_ev, 5),
        electron_transfer_rate_s_inv=rate_s_inv,
    )


def run_marcus_reorganization_pipeline(
    e_neutral_at_neutral_geom_hartree: float,
    e_charged_at_neutral_geom_hartree: float,
    e_charged_at_charged_geom_hartree: float,
    e_neutral_at_charged_geom_hartree: float,
    lambda_outer_ev: Optional[float] = None,
    delta_g_0_ev: Optional[float] = None,
    electronic_coupling_ev: Optional[float] = None,
    temperature_k: float = 298.15,
    lock_file: Optional[Union[Path, str]] = None,
) -> MarcusReorganizationResult:
    """
    [M][E] Execute thread-safe Nelson 4-point Marcus reorganization calculation with atomic file lock.
    """
    target_lock = lock_file if lock_file is not None else Path("marcus_pipeline.lock")

    with AtomicFileLock(target_lock, timeout=30.0):
        return compute_marcus_reorganization(
            e_neutral_at_neutral_geom_hartree=e_neutral_at_neutral_geom_hartree,
            e_charged_at_neutral_geom_hartree=e_charged_at_neutral_geom_hartree,
            e_charged_at_charged_geom_hartree=e_charged_at_charged_geom_hartree,
            e_neutral_at_charged_geom_hartree=e_neutral_at_charged_geom_hartree,
            lambda_outer_ev=lambda_outer_ev,
            delta_g_0_ev=delta_g_0_ev,
            electronic_coupling_ev=electronic_coupling_ev,
            temperature_k=temperature_k,
        )
