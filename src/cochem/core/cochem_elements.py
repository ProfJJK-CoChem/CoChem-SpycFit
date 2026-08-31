"""
CoChem Core Chemical Elements and Dynamic Mendeleev Mass Resolver.

Provenance:
- Method Matrix [M]: Dynamic atomic masses and Bondi-Mantina van der Waals radii dynamically
  retrieved via Mendeleev. Quantum spin-parity invariants ((N_e + M) mod 2 == 1), multiplicity
  upper bounds (M - 1 <= N_e), and pre-flight charge/multiplicity gating (REQ-QC-081).
- Derived Relations [D]: IUPAC CIAAW standard atomic weights and isotopic masses dynamically retrieved via mendeleev.
- Empirical [E]: Pre-flight oxidation state heuristics and valence boundaries.
- Chunk 06 Mandates: Frozen-core electron configurations (REQ-CORE-056) and Cordero/Pyykkö covalent radii (REQ-QC-059).
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mendeleev import element as _mendeleev_element
from mendeleev.models import Element
import numpy as np

from cochem.qc.exceptions import (
    CoChemQCException,
    IsotopeNotFoundError,
    SpinParityViolationError,
    UnphysicalValenceError,
)
from cochem.qc.models import SpinChargeValidationResult


@functools.lru_cache(maxsize=128)
def _get_element(symbol_or_z: Union[str, int]) -> Element:
    """
    Retrieve mendeleev Element object with cached lookups and symbol normalization.

    Args:
        symbol_or_z: Chemical element symbol (e.g. 'H', 'He', 'cl') or atomic number (e.g. 1, 6).

    Returns:
        mendeleev.models.Element object.

    Raises:
        ValueError: If element symbol or atomic number is invalid or cannot be resolved.
    """
    if isinstance(symbol_or_z, int):
        if symbol_or_z < 1 or symbol_or_z > 118:
            raise ValueError(f"Invalid atomic number Z={symbol_or_z}. Must be between 1 and 118.")
        try:
            return _mendeleev_element(symbol_or_z)
        except Exception as err:
            raise ValueError(f"Could not resolve element for atomic number {symbol_or_z}: {err}") from err

    raw = str(symbol_or_z).strip()
    if not raw:
        raise ValueError(f"Invalid chemical element symbol '{symbol_or_z}'.")

    if raw.isdigit():
        z_val = int(raw)
        if z_val < 1 or z_val > 118:
            raise ValueError(f"Invalid atomic number Z={z_val}. Must be between 1 and 118.")
        return _mendeleev_element(z_val)

    # Normalize symbol capitalization (e.g. 'c' -> 'C', 'he' -> 'He', 'CL' -> 'Cl')
    if len(raw) == 1:
        normalized = raw.upper()
    elif len(raw) <= 3:
        normalized = raw[0].upper() + raw[1:].lower()
    else:
        raise ValueError(f"Invalid chemical element symbol '{symbol_or_z}'. Must be a 1-3 letter element symbol.")

    try:
        elem = _mendeleev_element(normalized)
        if elem is None or elem.symbol.upper() != normalized.upper():
            raise ValueError(f"Element symbol '{symbol_or_z}' not found in Mendeleev database.")
        return elem
    except Exception as err:
        raise ValueError(f"Could not resolve chemical element symbol '{symbol_or_z}': {err}") from err


@functools.lru_cache(maxsize=128)
def get_atomic_number(symbol: str) -> int:
    """
    [M][D] Retrieve the atomic number Z for a chemical element symbol.

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'Fe').

    Returns:
        Atomic number Z (integer >= 1).
    """
    elem = _get_element(symbol)
    return int(elem.atomic_number)


def get_element_atomic_number(symbol: str) -> int:
    """Alias for get_atomic_number for backwards compatibility."""
    return get_atomic_number(symbol)


@functools.lru_cache(maxsize=128)
def get_atomic_mass(symbol_or_z: Union[str, int], isotope: Optional[int] = None) -> float:
    """
    [M][D] Dynamically retrieve the atomic mass or isotopic mass in amu from Mendeleev (REQ-CORE-056).

    Args:
        symbol_or_z: Chemical element symbol (e.g. 'H', 'C', 'Cl') or atomic number Z.
        isotope: Optional integer isotope mass number A (e.g. 13 for C-13).

    Returns:
        Atomic or isotopic mass in amu (float > 0).

    Raises:
        IsotopeNotFoundError: If specified isotope mass number is absent in Mendeleev database.
    """
    elem = _get_element(symbol_or_z)
    if isotope is not None:
        isotopes = elem.isotopes or []
        for iso in isotopes:
            if iso.mass_number == isotope and iso.mass is not None:
                return float(iso.mass)
        available = [iso.mass_number for iso in isotopes if iso.mass_number is not None]
        raise IsotopeNotFoundError(
            symbol=elem.symbol,
            mass_number=isotope,
            available_mass_numbers=available,
        )

    mass_val = elem.mass if elem.mass is not None else elem.atomic_weight
    if mass_val is None:
        raise ValueError(f"Standard atomic mass not available for element '{symbol_or_z}'.")
    return float(mass_val)


def get_ciaaw_atomic_mass(symbol: str) -> float:
    """Alias for get_atomic_mass(symbol) for backwards compatibility."""
    return get_atomic_mass(symbol)


def get_ciaaw_isotope_mass(symbol: str, mass_number: int) -> float:
    """Alias for get_atomic_mass(symbol, isotope=mass_number) for backwards compatibility."""
    return get_atomic_mass(symbol, isotope=mass_number)


@functools.lru_cache(maxsize=128)
def get_frozen_core_count(atomic_number: int) -> int:
    """
    [REQ-CORE-056] Dynamic noble gas frozen-core electron configuration mapping.

    Shell boundaries:
    - Z <= 2 (H, He): 0 core electrons.
    - Z 3..10 (Li..Ne): 2 core electrons ([He] 1s2).
    - Z 11..18 (Na..Ar): 10 core electrons ([Ne] 1s2 2s2 2p6).
    - Z 19..36 (K..Kr): 18 core electrons ([Ar] [Ne] 3s2 3p6).
    - Z 37..54 (Rb..Xe): 36 core electrons ([Kr] [Ar] 3d10 4s2 4p6).
    - Z >= 55 (Cs+): 68 core electrons ([Xe] + 4f14).

    Args:
        atomic_number: Nuclear charge Z (1 <= Z <= 118).

    Returns:
        Number of frozen core electrons.
    """
    if atomic_number < 1 or atomic_number > 118:
        raise ValueError(f"Invalid atomic number Z={atomic_number}. Must be between 1 and 118.")

    if atomic_number <= 2:
        return 0
    elif 3 <= atomic_number <= 10:
        return 2
    elif 11 <= atomic_number <= 18:
        return 10
    elif 19 <= atomic_number <= 36:
        return 18
    elif 37 <= atomic_number <= 54:
        return 36
    else:  # Z >= 55
        return 68


@functools.lru_cache(maxsize=128)
def get_covalent_radius(symbol_or_z: Union[str, int]) -> float:
    """
    [REQ-QC-059] Retrieve Cordero/Pyykkö covalent radius in Ångströms (converted from pm / 100.0).

    Args:
        symbol_or_z: Chemical element symbol or atomic number.

    Returns:
        Covalent radius in Ångströms.
    """
    elem = _get_element(symbol_or_z)
    radius_pm = getattr(elem, "covalent_radius_cordero", None)
    if radius_pm is None:
        radius_pm = getattr(elem, "covalent_radius_pyykko", None)
    if radius_pm is None:
        radius_pm = getattr(elem, "covalent_radius", None)
    if radius_pm is None:
        radius_pm = 75.0
    return float(radius_pm) / 100.0


@functools.lru_cache(maxsize=128)
def get_vdw_radius(symbol: str) -> float:
    """
    [M][D] Retrieve Bondi-Mantina van der Waals radius in Ångströms (converted from pm / 100.0).

    Args:
        symbol: Chemical element symbol.

    Returns:
        vdW radius in Ångströms.
    """
    elem = _get_element(symbol)
    radius_pm = elem.vdw_radius
    if radius_pm is None:
        radius_pm = getattr(elem, "vdw_radius_alvarez", None)
    if radius_pm is None:
        # Fallback default if not in table
        radius_pm = 170.0
    return float(radius_pm) / 100.0


@functools.lru_cache(maxsize=128)
def get_element_symbol(atomic_number: int) -> str:
    """
    [D] Retrieve canonical chemical symbol for a given atomic number Z.

    Args:
        atomic_number: Nuclear charge Z (1 <= Z <= 118).

    Returns:
        Canonical element symbol string.
    """
    elem = _get_element(atomic_number)
    return str(elem.symbol)


def batch_get_masses(symbols: List[str]) -> np.ndarray:
    """
    [M][D] Return a 1D NumPy array of atomic masses in amu for a list of element symbols.

    Args:
        symbols: List of element symbols.

    Returns:
        1D float64 NumPy array of atomic masses.
    """
    return np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)


@functools.lru_cache(maxsize=128)
def is_heavy_element(symbol: str, threshold_z: int = 36) -> bool:
    """
    [M] Determine if element is a heavy element (Z > threshold_z). Default threshold Z=36 (Krypton).

    Args:
        symbol: Chemical element symbol.
        threshold_z: Atomic number threshold (default 36).

    Returns:
        True if Z > threshold_z, False otherwise.
    """
    z = get_atomic_number(symbol)
    return z > threshold_z


def get_available_isotopes(symbol: str) -> List[int]:
    """
    [D] Retrieve list of all available IUPAC CIAAW isotope mass numbers for an element.

    Args:
        symbol: Chemical element symbol.

    Returns:
        Sorted list of integer mass numbers.
    """
    elem = _get_element(symbol)
    isotopes = elem.isotopes or []
    return sorted([iso.mass_number for iso in isotopes if iso.mass_number is not None])


def get_element_name(symbol: str) -> str:
    """
    [D] Retrieve full IUPAC English name for a chemical element.

    Args:
        symbol: Chemical element symbol.

    Returns:
        Element name (e.g. 'Hydrogen', 'Carbon').
    """
    elem = _get_element(symbol)
    return str(elem.name)


def get_element_electronegativity(symbol: str, scale: str = "pauling") -> Optional[float]:
    """
    [D] Retrieve electronegativity value for an element.

    Args:
        symbol: Chemical element symbol.
        scale: Electronegativity scale (default: 'pauling').

    Returns:
        Electronegativity float value, or None if undefined (e.g. Noble Gases).
    """
    elem = _get_element(symbol)
    try:
        val = elem.electronegativity(scale.lower())
        return float(val) if val is not None else None
    except Exception:
        return None


def validate_charge_and_multiplicity(
    symbols: Sequence[str],
    charge: int,
    multiplicity: int,
    coordinates_angstrom: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> SpinChargeValidationResult:
    """
    [M][E] Pre-flight quantum chemistry spin-parity and oxidation state validator (REQ-QC-081).

    Enforces exact physical quantum mechanical invariants:
    1. Total nuclear charge: Z_total = sum(Z_i).
    2. Total electron count: N_e = Z_total - q >= 1.
    3. Spin-Parity Invariant: (N_e + M) mod 2 == 1 where M = 2S + 1.
       - Even N_e requires odd multiplicity M in {1, 3, 5, ...}.
       - Odd N_e requires even multiplicity M in {2, 4, 6, ...}.
    4. Multiplicity Ceiling: M - 1 <= N_e (unpaired electrons cannot exceed total electrons).
    5. Pre-flight heuristic checks on formal oxidation states and valences.

    Args:
        symbols: Sequence of chemical element symbols in the molecular system.
        charge: Net total molecular charge q (integer).
        multiplicity: Total spin multiplicity M = 2S + 1 (integer >= 1).
        coordinates_angstrom: Optional Cartesian coordinates (x, y, z) per atom in Å.

    Returns:
        SpinChargeValidationResult containing validation status, electron tallies, and warnings.

    Raises:
        ValueError: If symbols sequence is empty.
        SpinParityViolationError: If spin-parity condition, multiplicity bound, or electron count is violated.
    """
    if not symbols:
        raise ValueError("Molecular symbols sequence cannot be empty.")

    total_nuclear_charge = sum(get_atomic_number(s) for s in symbols)
    total_electrons = total_nuclear_charge - int(charge)

    # 1. Total electron count floor
    if total_electrons < 1:
        raise SpinParityViolationError(
            message=(
                f"Spin-parity violation: Total electron count N_e={total_electrons} is unphysical. "
                f"A molecular structure requires at least 1 electron (Z_total={total_nuclear_charge}, charge={charge:+d})."
            ),
            n_electrons=total_electrons,
            multiplicity=multiplicity,
            charge=charge,
            total_nuclear_charge=total_nuclear_charge,
        )

    # 2. Multiplicity lower bound
    if multiplicity < 1:
        raise SpinParityViolationError(
            message=f"Spin-parity violation: Spin multiplicity M={multiplicity} is unphysical (must be >= 1).",
            n_electrons=total_electrons,
            multiplicity=multiplicity,
            charge=charge,
            total_nuclear_charge=total_nuclear_charge,
        )

    # 3. Spin-Parity parity condition: (N_e + M) mod 2 == 1
    rem = (total_electrons + multiplicity) % 2
    if rem != 1:
        raise SpinParityViolationError(
            n_electrons=total_electrons,
            multiplicity=multiplicity,
            charge=charge,
            total_nuclear_charge=total_nuclear_charge,
        )

    # 4. Multiplicity ceiling: M - 1 <= N_e
    unpaired_electrons = multiplicity - 1
    if unpaired_electrons > total_electrons:
        raise SpinParityViolationError(
            n_electrons=total_electrons,
            multiplicity=multiplicity,
            charge=charge,
            total_nuclear_charge=total_nuclear_charge,
        )

    # 5. Diagnostic valence and oxidation state heuristic checks
    warnings: List[str] = []
    formal_ox: Dict[int, float] = {}

    # Single atom ionic checks
    if len(symbols) == 1:
        formal_ox[0] = float(charge)
        elem = _get_element(symbols[0])
        oxistates = elem.oxistates or []
        if oxistates:
            max_oxi = max(oxistates)
            min_oxi = min(oxistates)
            if charge > max_oxi or charge < min_oxi:
                warnings.append(
                    f"Element '{elem.name}' ({elem.symbol}) has formal oxidation state {charge:+d}, "
                    f"which exceeds standard physical oxidation state range [{min_oxi:+d}, {max_oxi:+d}]."
                )
    else:
        # Polyatomic formal oxidation states using electronegativity weighting
        n_atoms = len(symbols)
        avg_charge = charge / n_atoms
        electronegativities = [get_element_electronegativity(s) for s in symbols]

        # Calculate electronegativity-weighted oxidation states
        valid_ens = [en if en is not None else 2.2 for en in electronegativities]
        mean_en = sum(valid_ens) / n_atoms
        en_diffs = [mean_en - en for en in valid_ens]
        diff_sum = sum(abs(d) for d in en_diffs)

        if diff_sum > 1e-6:
            for idx, d in enumerate(en_diffs):
                formal_ox[idx] = round(avg_charge + (d / diff_sum) * (abs(charge) if charge != 0 else 1.0), 3)
        else:
            for idx in range(n_atoms):
                formal_ox[idx] = round(avg_charge, 3)

        if abs(charge) > 2 * n_atoms:
            warnings.append(
                f"High net charge {charge:+d} across {n_atoms} atoms yields an unphysical "
                f"average oxidation state ({avg_charge:+.2f} per atom)."
            )

        # Check for highly positive charge on purely electronegative atoms
        if (
            all(en is not None and en > 3.0 for en in electronegativities)
            and charge >= n_atoms
        ):
            warnings.append(
                f"Highly positive charge {charge:+d} on electronegative cluster "
                f"({''.join(symbols)}) may indicate an unphysical valence state."
            )

    return SpinChargeValidationResult(
        is_valid=True,
        total_nuclear_charge=total_nuclear_charge,
        total_net_charge=charge,
        total_electrons=total_electrons,
        multiplicity=multiplicity,
        parity_valid=True,
        max_unpaired_electrons=unpaired_electrons,
        formal_oxidation_states=formal_ox,
        unphysical_valence_warnings=warnings,
    )
