"""
CoChem Provenance & Method Validator Engine (REQ-QC-085).

Provenance:
- Method Matrix [M]: Quantum Level-of-Theory invariance and stationary point gates.
- Derived Relations [D]: Canonical SHA-256 fingerprinting over Hamiltonian, basis set,
  auxiliary basis, dispersion scheme, numerical quadrature grid, and solvation model.
- Empirical [E]: Cartesian gradient residual stationary point threshold (||g||_inf <= 1.0e-4 a.u.).
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from cochem.qc.exceptions import (
    LevelOfTheoryMismatchError,
    NonStationaryFrequencyError,
)
from cochem.qc.models import LevelOfTheoryEnvelope


DEFAULT_STATIONARY_GRADIENT_THRESHOLD: float = 1.0e-4


def canonicalize_lot_string(
    hamiltonian: str,
    basis: str,
    aux_basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    grid: Optional[str] = "DefGrid2",
    solvent: Optional[str] = None,
) -> str:
    """
    [D] Normalize and construct canonical Level-of-Theory string representation.

    Normalization Rules:
    - Hamiltonian method: stripped, converted to uppercase (e.g. 'WB97X-D4', 'B3LYP').
    - Basis set: stripped, converted to lowercase (e.g. 'def2-tzvp', 'cc-pvtz').
    - Auxiliary basis: stripped, converted to lowercase if present, else empty string.
    - Dispersion correction: stripped, converted to uppercase if present, else empty string.
    - Integration grid: stripped, converted to lowercase if present, else empty string.
    - Solvation model: stripped, converted to lowercase if present, else empty string.

    Format:
        '{hamiltonian}|{basis}|{aux_basis}|{dispersion}|{grid}|{solvent}'

    Parameters
    ----------
    hamiltonian : str
        Electronic structure Hamiltonian method.
    basis : str
        Primary orbital basis set.
    aux_basis : Optional[str], optional
        Auxiliary fitting / Coulomb basis set.
    dispersion : Optional[str], optional
        Empirical or semi-empirical dispersion correction scheme.
    grid : Optional[str], optional
        Numerical integration quadrature grid.
    solvent : Optional[str], optional
        Implicit solvent or continuum dielectric model.

    Returns
    -------
    str
        Canonical pipe-delimited normalized Level-of-Theory string.

    Raises
    ------
    ValueError
        If mandatory components (hamiltonian, basis, or grid) are empty.
    """
    if not hamiltonian or not hamiltonian.strip():
        raise ValueError("Hamiltonian method cannot be empty.")
    if not basis or not basis.strip():
        raise ValueError("Basis set cannot be empty.")
    if not grid or not grid.strip():
        raise ValueError("Integration grid cannot be empty.")

    ham_norm = hamiltonian.strip().upper()
    basis_norm = basis.strip().lower()
    aux_norm = aux_basis.strip().lower() if aux_basis and aux_basis.strip() else ""
    disp_norm = dispersion.strip().upper() if dispersion and dispersion.strip() else ""
    grid_norm = grid.strip().lower() if grid and grid.strip() else ""
    solv_norm = solvent.strip().lower() if solvent and solvent.strip() else ""

    return f"{ham_norm}|{basis_norm}|{aux_norm}|{disp_norm}|{grid_norm}|{solv_norm}"


def canonicalize_lot_envelope(envelope: LevelOfTheoryEnvelope) -> str:
    """
    [D] Normalize and construct canonical Level-of-Theory string from a LevelOfTheoryEnvelope.

    Parameters
    ----------
    envelope : LevelOfTheoryEnvelope
        The Level-of-Theory metadata envelope.

    Returns
    -------
    str
        Canonical pipe-delimited normalized Level-of-Theory string.
    """
    return canonicalize_lot_string(
        hamiltonian=envelope.hamiltonian_method,
        basis=envelope.basis_set,
        aux_basis=envelope.auxiliary_basis_set,
        dispersion=envelope.dispersion_correction,
        grid=envelope.integration_grid,
        solvent=envelope.solvation_model,
    )


def compute_lot_hash_from_components(
    hamiltonian: str,
    basis: str,
    aux_basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    grid: Optional[str] = "DefGrid2",
    solvent: Optional[str] = None,
) -> str:
    """
    [D] Compute the canonical SHA-256 cryptographic fingerprint from raw LoT components.

    Parameters
    ----------
    hamiltonian : str
        Electronic structure Hamiltonian method.
    basis : str
        Primary orbital basis set.
    aux_basis : Optional[str], optional
        Auxiliary fitting / Coulomb basis set.
    dispersion : Optional[str], optional
        Empirical dispersion correction scheme.
    grid : Optional[str], optional
        Numerical integration quadrature grid.
    solvent : Optional[str], optional
        Implicit solvent model.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical_str = canonicalize_lot_string(
        hamiltonian=hamiltonian,
        basis=basis,
        aux_basis=aux_basis,
        dispersion=dispersion,
        grid=grid,
        solvent=solvent,
    )
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_lot_hash(envelope: LevelOfTheoryEnvelope) -> str:
    """
    [D] Compute the canonical SHA-256 cryptographic fingerprint from a LevelOfTheoryEnvelope.

    Parameters
    ----------
    envelope : LevelOfTheoryEnvelope
        The Level-of-Theory metadata envelope.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    return compute_lot_hash_from_components(
        hamiltonian=envelope.hamiltonian_method,
        basis=envelope.basis_set,
        aux_basis=envelope.auxiliary_basis_set,
        dispersion=envelope.dispersion_correction,
        grid=envelope.integration_grid,
        solvent=envelope.solvation_model,
    )


# Alias for backwards compatibility
generate_lot_hash = compute_lot_hash


def verify_level_of_theory_invariance(
    opt_envelope: LevelOfTheoryEnvelope,
    freq_envelope: LevelOfTheoryEnvelope,
) -> bool:
    """
    [M][D] Gate 1: Cryptographic Level-of-Theory Invariance Gate (REQ-QC-085).

    Compares the canonical SHA-256 fingerprints of geometry optimization and subsequent
    frequency/thermodynamic/property evaluation envelopes.

    Parameters
    ----------
    opt_envelope : LevelOfTheoryEnvelope
        Optimization geometry Level-of-Theory envelope.
    freq_envelope : LevelOfTheoryEnvelope
        Frequency / property Level-of-Theory envelope.

    Returns
    -------
    bool
        True if fingerprints match identically.

    Raises
    ------
    LevelOfTheoryMismatchError
        If SHA-256 fingerprints diverge between optimization and frequency steps.
    """
    opt_hash = compute_lot_hash(opt_envelope)
    freq_hash = compute_lot_hash(freq_envelope)

    if opt_hash != freq_hash:
        raise LevelOfTheoryMismatchError(
            opt_lot=opt_envelope,
            freq_lot=freq_envelope,
            opt_hash=opt_hash,
            freq_hash=freq_hash,
        )

    return True


# Alias for explicit enforcement terminology
enforce_frequency_lot_invariance = verify_level_of_theory_invariance


def verify_stationary_point(
    gradient_norm: float,
    threshold: float = DEFAULT_STATIONARY_GRADIENT_THRESHOLD,
) -> bool:
    """
    [M][E] Gate 2: Stationary Point Residual Gradient Gate (REQ-QC-085).

    Validates that the optimization maximum gradient residual ||g||_inf <= threshold (1.0e-4 a.u.).

    Parameters
    ----------
    gradient_norm : float
        Maximum Cartesian gradient component residual in atomic units (Hartree/Bohr).
    threshold : float, optional
        Stationary point convergence threshold in a.u. Default is 1.0e-4 a.u.

    Returns
    -------
    bool
        True if gradient norm satisfies stationary convergence criteria.

    Raises
    ------
    NonStationaryFrequencyError
        If gradient_norm > threshold, gradient_norm < 0, or is NaN / infinite.
    """
    if (
        math.isnan(gradient_norm)
        or math.isinf(gradient_norm)
        or gradient_norm < 0.0
        or gradient_norm > threshold
    ):
        raise NonStationaryFrequencyError(
            gradient_norm=gradient_norm,
            threshold=threshold,
        )

    return True


def validate_quantum_calculation_gate(
    opt_lot: LevelOfTheoryEnvelope,
    freq_lot: LevelOfTheoryEnvelope,
    gradient_norm: Optional[float] = None,
    gradient_threshold: float = DEFAULT_STATIONARY_GRADIENT_THRESHOLD,
) -> bool:
    """
    [M][D][E] Combined QC Gate: Validates both Level-of-Theory invariance and stationary point conditions.

    Parameters
    ----------
    opt_lot : LevelOfTheoryEnvelope
        Optimization geometry Level-of-Theory envelope.
    freq_lot : LevelOfTheoryEnvelope
        Frequency / property Level-of-Theory envelope.
    gradient_norm : Optional[float], optional
        Cartesian gradient residual. If None, retrieved from opt_lot.max_gradient_residual_au.
    gradient_threshold : float, optional
        Stationary point convergence threshold in a.u. Default is 1.0e-4 a.u.

    Returns
    -------
    bool
        True if both Level-of-Theory invariance and stationary point gates pass.

    Raises
    ------
    LevelOfTheoryMismatchError
        If optimization and frequency Level-of-Theory envelopes diverge.
    NonStationaryFrequencyError
        If Cartesian gradient residual exceeds convergence threshold.
    """
    # 1. Gate 1: Cryptographic Level-of-Theory Invariance Gate
    verify_level_of_theory_invariance(opt_envelope=opt_lot, freq_envelope=freq_lot)

    # 2. Gate 2: Stationary Point Gradient Gate
    res_grad = gradient_norm if gradient_norm is not None else opt_lot.max_gradient_residual_au
    verify_stationary_point(gradient_norm=res_grad, threshold=gradient_threshold)

    return True
