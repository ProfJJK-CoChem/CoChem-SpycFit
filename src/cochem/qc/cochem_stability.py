"""
CoChem Post-SCF Wavefunction Stability and Broken-Symmetry DFT Engine (REQ-QC-087).

Provenance:
- Method Matrix [M]: Electronic orbital Hessian lowest eigenvalue stability evaluation, Yamaguchi approximate spin projection.
- Derived Relations [D]: Spin contamination metric and High-Spin (HS) / Broken-Symmetry (BS) state mixing coefficient.
- Empirical [E]: Post-SCF electronic structure classification (internal RHF->UHF, real->complex, external instabilities).
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.models import WavefunctionStabilityReport


class StabilityAnalysisResult(BaseModel):
    """Data contract representing electronic orbital stability analysis."""
    model_config = ConfigDict(frozen=True)

    is_stable: bool
    lowest_eigenvalue: float
    instability_mode: str
    requires_broken_symmetry: bool


class BrokenSymmetryState(BaseModel):
    """Data contract representing a Broken-Symmetry open-shell singlet and High-Spin state pair."""
    model_config = ConfigDict(frozen=True)

    e_bs_hartree: float
    s2_bs: float
    e_hs_hartree: float
    s2_hs: float
    yamaguchi_energy_hartree: float


class UnstableWavefunctionError(RuntimeError):
    """Raised when an electronic wavefunction exhibits unphysical or unresolved stability eigenvalues."""
    pass


def compute_yamaguchi_projection(
    e_bs_hartree: float,
    s2_bs: float,
    e_hs_hartree: float,
    s2_hs: float,
    s_expected: float = 0.0,
) -> float:
    """
    [M][D] Calculate the Yamaguchi approximate spin-projected singlet electronic energy.
    """
    s2_ls = s_expected * (s_expected + 1.0)
    denom_hs = s2_hs - s2_ls
    if abs(denom_hs) < 1e-6:
        raise ValueError("High-spin <S^2> denominator is too close to low-spin expectation value.")

    c_hs = (s2_bs - s2_ls) / denom_hs
    denom_proj = 1.0 - c_hs
    if abs(denom_proj) < 1e-6:
        raise ValueError("Yamaguchi denominator 1 - c_HS is zero.")

    e_proj = (e_bs_hartree - c_hs * e_hs_hartree) / denom_proj
    return float(e_proj)


def evaluate_wavefunction_stability(
    lowest_orbital_hessian_eigenval: float,
    s2_calc: float,
    s_expected: float,
    e_bs_hartree: Optional[float] = None,
    e_hs_hartree: Optional[float] = None,
    s2_hs: Optional[float] = None,
    instability_type: Optional[Literal["UNSTABLE_INTERNAL_RHF_UHF", "UNSTABLE_REAL_COMPLEX", "UNSTABLE_EXTERNAL"]] = None,
) -> WavefunctionStabilityReport:
    """
    [M][D] Evaluate post-SCF orbital Hessian stability and compute Yamaguchi spin-projected energies (REQ-QC-087).
    """
    if s_expected < 0.0:
        raise ValueError(f"Expected spin quantum number S must be >= 0, got S={s_expected}.")
    if s2_calc < 0.0:
        raise ValueError(f"Calculated <S^2> must be >= 0, got <S^2>={s2_calc}.")

    # 1. Stability Classification
    is_stable = lowest_orbital_hessian_eigenval >= 0.0
    if is_stable:
        status: Literal["STABLE", "UNSTABLE_INTERNAL_RHF_UHF", "UNSTABLE_REAL_COMPLEX", "UNSTABLE_EXTERNAL"] = "STABLE"
        broken_sym_req = False
    else:
        status = instability_type if instability_type is not None else "UNSTABLE_INTERNAL_RHF_UHF"
        broken_sym_req = True

    # 2. Spin Contamination Metric
    s2_exp = s_expected * (s_expected + 1.0)
    spin_diff = abs(s2_calc - s2_exp)
    spin_contamination_pct = (spin_diff / (s2_exp + 1.0)) * 100.0

    # 3. Yamaguchi Spin Projection for Broken-Symmetry state
    yamaguchi_energy: Optional[float] = None
    if (
        e_bs_hartree is not None
        and e_hs_hartree is not None
        and s2_hs is not None
    ):
        yamaguchi_energy = compute_yamaguchi_projection(
            e_bs_hartree=e_bs_hartree,
            s2_bs=s2_calc,
            e_hs_hartree=e_hs_hartree,
            s2_hs=s2_hs,
            s_expected=s_expected,
        )

    return WavefunctionStabilityReport(
        initial_stability=status,
        lowest_orbital_hessian_eigenvalue=lowest_orbital_hessian_eigenval,
        is_broken_symmetry_required=broken_sym_req,
        s2_expected=round(s2_exp, 4),
        s2_calculated=round(s2_calc, 4),
        spin_contamination_percentage=round(spin_contamination_pct, 4),
        yamaguchi_corrected_energy_hartree=round(yamaguchi_energy, 6) if yamaguchi_energy is not None else None,
    )
