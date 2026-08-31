"""
CoChem Post-SCF Wavefunction Stability and Broken-Symmetry DFT Engine (REQ-QC-087).

Provenance:
- Method Matrix [M]: Electronic orbital Hessian lowest eigenvalue stability evaluation, Yamaguchi approximate spin projection.
- Derived Relations [D]: Spin contamination metric and High-Spin (HS) / Broken-Symmetry (BS) state mixing coefficient.
- Empirical [E]: Post-SCF electronic structure classification (internal RHF->UHF, real->complex, external instabilities).
"""

from __future__ import annotations

from typing import Literal, Optional

from cochem.qc.models import WavefunctionStabilityReport


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

    Physical Principles:
        1. Orbital Stability:
           - lambda_orb >= 0: Stable electronic ground state.
           - lambda_orb < 0: Unstable solution. Requires broken-symmetry unrestricted SCF.
        2. Spin Contamination:
           - S_exp(S_exp + 1) vs calculated <S^2>.
           - Contamination % = (|<S^2>_calc - S_exp(S_exp + 1)|) / (S_exp(S_exp + 1) + 1) * 100%.
        3. Yamaguchi Spin Projection (Broken-Symmetry):
           - c_HS = (<S^2>_BS - S_LS(S_LS + 1)) / (<S^2>_HS - S_LS(S_LS + 1))
           - E_proj = (E_BS - c_HS * E_HS) / (1 - c_HS)

    Args:
        lowest_orbital_hessian_eigenval: Lowest eigenvalue lambda_orb of the electronic orbital Hessian.
        s2_calc: Post-SCF calculated <S^2> total spin operator expectation value.
        s_expected: Expected total spin quantum number S (e.g. 0 for singlet, 0.5 for doublet, 1.0 for triplet).
        e_bs_hartree: Optional Broken-Symmetry electronic energy in Hartree.
        e_hs_hartree: Optional High-Spin electronic energy in Hartree.
        s2_hs: Optional High-Spin state <S^2> expectation value.
        instability_type: Optional explicit instability mode if lambda_orb < 0.

    Returns:
        WavefunctionStabilityReport containing stability enum, spin contamination, and projected energy.

    Raises:
        ValueError: If s_expected < 0 or s2_calc < 0.
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
        s2_ls = s2_exp
        denom_hs = s2_hs - s2_ls
        if abs(denom_hs) > 1e-6:
            c_hs = (s2_calc - s2_ls) / denom_hs
            denom_proj = 1.0 - c_hs
            if abs(denom_proj) > 1e-6:
                yamaguchi_energy = (e_bs_hartree - c_hs * e_hs_hartree) / denom_proj

    return WavefunctionStabilityReport(
        initial_stability=status,
        lowest_orbital_hessian_eigenvalue=lowest_orbital_hessian_eigenval,
        is_broken_symmetry_required=broken_sym_req,
        s2_expected=round(s2_exp, 4),
        s2_calculated=round(s2_calc, 4),
        spin_contamination_percentage=round(spin_contamination_pct, 4),
        yamaguchi_corrected_energy_hartree=round(yamaguchi_energy, 6) if yamaguchi_energy is not None else None,
    )
