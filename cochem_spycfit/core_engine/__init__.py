"""
CoChem-SpycFit Core Engine Re-export proxy.
"""

from src.cochem_spycfit.core_engine.cochem_spycfit_init import *  # noqa: F401, F403
from src.cochem_spycfit.core_engine.cochem_spycfit_init import (
    CANONICAL_BINARY_HASHES,
    DEFAULT_CHUNK_SIZE,
    MIN_GPU_VRAM_GB,
    TIER_BINARY_MAP,
    BinaryRouteResult,
    BinaryVerificationResult,
    HardwareGateResult,
    JAXPrecisionConfig,
    OSTier,
    SpycfitGatekeeperState,
    XLACacheConfig,
    calculate_binary_sha256,
    configure_xla_cache,
    enforce_jax_precision,
    evaluate_hardware_fallback_gate,
    initialize_spycfit_gatekeeper,
    query_physical_gpu_vram,
    resolve_cochem_system_config_path,
    route_6tier_fortran_binary,
    verify_binary_checksum,
)

__all__ = [
    "CANONICAL_BINARY_HASHES",
    "DEFAULT_CHUNK_SIZE",
    "MIN_GPU_VRAM_GB",
    "TIER_BINARY_MAP",
    "BinaryRouteResult",
    "BinaryVerificationResult",
    "HardwareGateResult",
    "JAXPrecisionConfig",
    "OSTier",
    "SpycfitGatekeeperState",
    "XLACacheConfig",
    "calculate_binary_sha256",
    "configure_xla_cache",
    "enforce_jax_precision",
    "evaluate_hardware_fallback_gate",
    "initialize_spycfit_gatekeeper",
    "query_physical_gpu_vram",
    "resolve_cochem_system_config_path",
    "route_6tier_fortran_binary",
    "verify_binary_checksum",
]
