"""
CoChem-SPYFY: JAX Hamiltonian Autodiff Driver Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spyfy.physics_core.cochem_spyfy_jax_driver import (
        JaxHamiltonianDriver,
        STANDARD_WATSON_A_KEYS,
        get_asymmetric_level_index,
        _detect_device,
    )
except ImportError:
    from cochem_spyfy.physics_core.cochem_spyfy_jax_driver import (  # type: ignore[no-redef]
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
