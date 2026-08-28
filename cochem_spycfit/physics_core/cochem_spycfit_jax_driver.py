"""
CoChem-SpycFit: JAX Hamiltonian Autodiff Driver Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (
        JaxHamiltonianDriver,
        STANDARD_WATSON_A_KEYS,
        get_asymmetric_level_index,
        _detect_device,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (  # type: ignore[no-redef]
        JaxHamiltonianDriver,
        STANDARD_WATSON_A_KEYS,
        get_asymmetric_level_index,
        _detect_device,
    )

__all__ = [
    "JaxHamiltonianDriver",
    "STANDARD_WATSON_A_KEYS",
    "get_asymmetric_level_index",
    "_detect_device",
]
