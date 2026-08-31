"""
CoChem Dynamic Mendeleev Isotopic Substitution and Bigeleisen-Mayer KIE/EIE Engine (REQ-QC-086).

Provenance:
- Method Matrix [M]: Mass-weighted Cartesian Hessian re-diagonalization, Bigeleisen-Mayer KIE/EIE formulation.
- Derived Relations [D]: Dynamic IUPAC CIAAW isotopic masses via mendeleev, CODATA 2022 constants.
- Empirical [E]: High-accuracy Zero-Point Energy (ZPE) evaluations and Harmonic Oscillator partition functions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import get_ciaaw_atomic_mass, get_ciaaw_isotope_mass
from cochem.qc.exceptions import IsotopeNotFoundError
from cochem.qc.models import IsotopeSubstitutionPayload

# CODATA 2022 Physical Constants
AMU_TO_AU: float = 1822.888486209
AU_TO_CM1: float = 219474.63136320
CM1_TO_KCAL_MOL: float = 2.85914316e-3
R_GAS_KCAL_MOL_K: float = 1.98720425864083e-3
H_C_OVER_K_B: float = 1.438776877  # hc/k_B in cm * K


class IsotopeSubstitutionSpec(BaseModel):
    """Specification of an isotopic substitution pattern."""
    model_config = ConfigDict(frozen=True)

    substituted_atoms: Dict[int, int] = Field(description="0-based atom index -> isotope mass number")
    description: Optional[str] = None


class KIEResult(BaseModel):
    """Result of Bigeleisen-Mayer Kinetic Isotope Effect calculation."""
    model_config = ConfigDict(frozen=True)

    kie_total: float
    mmi_factor: float
    exc_factor: float
    zpe_factor: float
    tun_factor: float = 1.0
    temperature_k: float


class EIEResult(BaseModel):
    """Result of Equilibrium Isotope Effect calculation."""
    model_config = ConfigDict(frozen=True)

    eie_total: float
    reactant_zpe_shift_kcal_mol: float
    product_zpe_shift_kcal_mol: float
    temperature_k: float


def _diagonalize_hessian_to_frequencies(
    cartesian_hessian: np.ndarray,
    masses_amu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize mass-weighted Cartesian Hessian and return harmonic frequencies in cm^-1.
    """
    n_atoms = len(masses_amu)
    m_diag = np.repeat(masses_amu * AMU_TO_AU, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_diag)

    # Form mass-weighted Hessian: H_MW = M^(-1/2) * H_Cart * M^(-1/2)
    h_mw = cartesian_hessian * np.outer(inv_sqrt_m, inv_sqrt_m)

    # Diagonalize symmetric matrix
    eigenvals, normal_modes = np.linalg.eigh(h_mw)

    # Frequencies in cm^-1: nu = sign(lambda) * sqrt(|lambda|) * 219474.63
    frequencies = np.sign(eigenvals) * np.sqrt(np.abs(eigenvals)) * AU_TO_CM1
    return frequencies, normal_modes


def compute_isotopic_substitution_kie(
    symbols: Sequence[str],
    cartesian_hessian_hartree_bohr2: Union[np.ndarray, Sequence[Sequence[float]]],
    substituted_atoms: Dict[int, int],
    temperature_k: float = 298.15,
) -> IsotopeSubstitutionPayload:
    """
    [M][D] Compute dynamic isotopic substitution frequencies, Zero-Point Energies, and Bigeleisen-Mayer KIE (REQ-QC-086).
    """
    n_atoms = len(symbols)
    h_cart = np.asarray(cartesian_hessian_hartree_bohr2, dtype=np.float64)

    if h_cart.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(
            f"Hessian dimensions {h_cart.shape} mismatch expected ({3 * n_atoms}, {3 * n_atoms}) for {n_atoms} atoms."
        )

    # 1. Retrieve dynamic reference masses (IUPAC CIAAW standard atomic weight)
    ref_masses = np.array([get_ciaaw_atomic_mass(s) for s in symbols], dtype=np.float64)
    iso_masses = ref_masses.copy()
    mendeleev_masses_dict: Dict[int, float] = {}

    # 2. Retrieve dynamic isotopic masses for substituted atoms
    for atom_idx, mass_num in substituted_atoms.items():
        if atom_idx < 0 or atom_idx >= n_atoms:
            raise ValueError(f"Substituted atom index {atom_idx} out of range [0, {n_atoms - 1}].")
        sym = symbols[atom_idx]
        mass_val = get_ciaaw_isotope_mass(sym, mass_num)
        iso_masses[atom_idx] = mass_val
        mendeleev_masses_dict[atom_idx] = mass_val

    # 3. Diagonalize reference and isotopologue Hessians
    ref_freqs, _ = _diagonalize_hessian_to_frequencies(h_cart, ref_masses)
    iso_freqs, _ = _diagonalize_hessian_to_frequencies(h_cart, iso_masses)

    # Filter genuine vibrational modes (> 10 cm^-1)
    pos_ref_mask = ref_freqs > 10.0
    pos_iso_mask = iso_freqs > 10.0

    pos_ref_freqs = ref_freqs[pos_ref_mask]
    pos_iso_freqs = iso_freqs[pos_iso_mask]

    # 4. Compute Zero-Point Vibrational Energies in kcal/mol
    ref_zpe = float(np.sum(0.5 * pos_ref_freqs * CM1_TO_KCAL_MOL))
    iso_zpe = float(np.sum(0.5 * pos_iso_freqs * CM1_TO_KCAL_MOL))

    # 5. Bigeleisen-Mayer KIE terms (MMI, EXC, ZPE)
    # ZPE exponential factor: exp((ZPE_ref - ZPE_iso) / (R * T))
    delta_zpe = ref_zpe - iso_zpe
    zpe_factor = math.exp(delta_zpe / (R_GAS_KCAL_MOL_K * temperature_k))

    # MMI and EXC product terms
    mmi_terms: List[float] = []
    exc_terms: List[float] = []

    min_len = min(len(pos_ref_freqs), len(pos_iso_freqs))
    for i in range(min_len):
        nu_ref = pos_ref_freqs[i]
        nu_iso = pos_iso_freqs[i]
        if nu_iso > 1e-3 and nu_ref > 1e-3:
            mmi_terms.append(nu_ref / nu_iso)
            u_ref = (H_C_OVER_K_B * nu_ref) / temperature_k
            u_iso = (H_C_OVER_K_B * nu_iso) / temperature_k

            exp_u_ref = math.exp(-u_ref) if u_ref < 50 else 0.0
            exp_u_iso = math.exp(-u_iso) if u_iso < 50 else 0.0
            exc_terms.append((1.0 - exp_u_ref) / (1.0 - exp_u_iso) if (1.0 - exp_u_iso) > 1e-9 else 1.0)

    mmi_factor = float(np.prod(mmi_terms)) if mmi_terms else 1.0
    exc_factor = float(np.prod(exc_terms)) if exc_terms else 1.0
    kie_total = mmi_factor * exc_factor * zpe_factor

    return IsotopeSubstitutionPayload(
        substituted_atoms=substituted_atoms,
        mendeleev_masses_amu=mendeleev_masses_dict,
        reference_zpe_kcal_mol=round(ref_zpe, 4),
        substituted_zpe_kcal_mol=round(iso_zpe, 4),
        reference_frequencies_cm1=[round(float(f), 2) for f in ref_freqs],
        substituted_frequencies_cm1=[round(float(f), 2) for f in iso_freqs],
        kie_bigeleisen_mayer=round(kie_total, 4),
        mmi_factor=round(mmi_factor, 4),
        exc_factor=round(exc_factor, 4),
        zpe_factor=round(zpe_factor, 4),
    )


def compute_eie_equilibrium_isotope_effect(
    reactants_payload: IsotopeSubstitutionPayload,
    products_payload: IsotopeSubstitutionPayload,
    temperature_k: float = 298.15,
) -> EIEResult:
    """
    [REQ-QC-086] Compute Equilibrium Isotope Effect (EIE) between reactant and product isotopologues.
    """
    delta_zpe_react = reactants_payload.reference_zpe_kcal_mol - reactants_payload.substituted_zpe_kcal_mol
    delta_zpe_prod = products_payload.reference_zpe_kcal_mol - products_payload.substituted_zpe_kcal_mol

    delta_delta_zpe = delta_zpe_react - delta_zpe_prod
    eie_val = math.exp(delta_delta_zpe / (R_GAS_KCAL_MOL_K * temperature_k))

    return EIEResult(
        eie_total=round(eie_val, 4),
        reactant_zpe_shift_kcal_mol=round(delta_zpe_react, 4),
        product_zpe_shift_kcal_mol=round(delta_zpe_prod, 4),
        temperature_k=temperature_k,
    )
