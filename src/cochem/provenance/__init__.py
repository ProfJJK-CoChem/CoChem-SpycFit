"""
CoChem Provenance & Method Verification Subpackage.

Provenance:
- Method Matrix [M]: Cryptographic Level-of-Theory invariance and stationary point gates.
- Derived Relations [D]: Canonical SHA-256 Level-of-Theory fingerprinting.
- Empirical [E]: Gradient residual convergence thresholds (1.0e-4 a.u.).
"""

from __future__ import annotations

from cochem.provenance.cochem_method_validator import (
    DEFAULT_STATIONARY_GRADIENT_THRESHOLD,
    canonicalize_lot_envelope,
    canonicalize_lot_string,
    compute_lot_hash,
    compute_lot_hash_from_components,
    enforce_frequency_lot_invariance,
    generate_lot_hash,
    validate_quantum_calculation_gate,
    verify_level_of_theory_invariance,
    verify_stationary_point,
)

__all__ = [
    "DEFAULT_STATIONARY_GRADIENT_THRESHOLD",
    "canonicalize_lot_envelope",
    "canonicalize_lot_string",
    "compute_lot_hash",
    "compute_lot_hash_from_components",
    "enforce_frequency_lot_invariance",
    "generate_lot_hash",
    "validate_quantum_calculation_gate",
    "verify_level_of_theory_invariance",
    "verify_stationary_point",
]
