"""
CoChem Core Chemical Elements and Dynamic Mendeleev Mass Resolver.

Provenance:
- Method Matrix [M]: Quantum spin-parity invariants ((N_e + M) mod 2 == 1), multiplicity upper bounds (M - 1 <= N_e),
  and pre-flight charge/multiplicity gating (REQ-QC-081).
- Derived Relations [D]: IUPAC CIAAW standard atomic weights and isotopic masses dynamically retrieved via mendeleev.
- Empirical [E]: Pre-flight oxidation state heuristics and valence boundaries.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Union

from mendeleev import element as _mendeleev_element
from mendeleev.models import Element

from cochem.qc.exceptions import (
    CoChemQCException,
    IsotopeNotFoundError,
    SpinParityViolationError,
    UnphysicalValenceError,
)
from cochem.qc.models import SpinChargeValidationResult


@lru_cache(maxsize=256)
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
    if not raw or raw.isdigit():
        raise ValueError(f"Invalid chemical element symbol '{symbol_or_z}'.")

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



def get_element_atomic_number(symbol: str) -> int:
    """
    [M][D] Retrieve the nuclear charge / atomic number Z for a chemical element symbol.

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'Fe').

    Returns:
        Atomic number Z (integer >= 1).
    """
    elem = _get_element(symbol)
    return int(elem.atomic_number)


def get_ciaaw_atomic_mass(symbol: str) -> float:
    """
    [D] Dynamically retrieve the standard IUPAC CIAAW atomic weight in atomic mass units (amu).

    Strictly satisfies the dynamic Mendeleev mandate without hardcoded constants.

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'Fe').

    Returns:
        Standard atomic mass in amu (float > 0).
    """
    elem = _get_element(symbol)
    mass_val = elem.mass if elem.mass is not None else elem.atomic_weight
    if mass_val is None:
        raise ValueError(f"Standard atomic mass not available for element '{symbol}'.")
    return float(mass_val)


def get_ciaaw_isotope_mass(symbol: str, mass_number: int) -> float:
    """
    [D] Dynamically retrieve the IUPAC CIAAW isotopic mass in atomic mass units (amu).

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'Cl').
        mass_number: Isotope mass number A (e.g. 1, 2 for H; 12, 13 for C).

    Returns:
        Isotopic mass in amu (float > 0).

    Raises:
        IsotopeNotFoundError: If isotope mass number is absent in IUPAC CIAAW records.
    """
    elem = _get_element(symbol)
    isotopes = elem.isotopes or []
    available_mass_numbers = [iso.mass_number for iso in isotopes if iso.mass_number is not None]

    for iso in isotopes:
        if iso.mass_number == mass_number:
            if iso.mass is not None:
                return float(iso.mass)

    raise IsotopeNotFoundError(
        symbol=elem.symbol,
        mass_number=mass_number,
        available_mass_numbers=available_mass_numbers,
    )


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

    Returns:
        SpinChargeValidationResult containing validation status, electron tallies, and warnings.

    Raises:
        ValueError: If symbols sequence is empty.
        SpinParityViolationError: If spin-parity condition, multiplicity bound, or electron count is violated.
    """
    if not symbols:
        raise ValueError("Molecular symbols sequence cannot be empty.")

    total_nuclear_charge = sum(get_element_atomic_number(s) for s in symbols)
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

    # Single atom ionic checks
    if len(symbols) == 1:
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
        # Polyatomic average charge / oxidation heuristics
        n_atoms = len(symbols)
        avg_charge = charge / n_atoms
        if abs(charge) > 2 * n_atoms:
            warnings.append(
                f"High net charge {charge:+d} across {n_atoms} atoms yields an unphysical "
                f"average oxidation state ({avg_charge:+.2f} per atom)."
            )

        # Check for highly positive charge on purely electronegative atoms
        electronegativities = [get_element_electronegativity(s) for s in symbols]
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
        unphysical_valence_warnings=warnings,
    )
