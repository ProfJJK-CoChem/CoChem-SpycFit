"""
CoChem-SpycFit Core Engine: Stage 0 Hardware & Precision Gatekeeper and Fit Registry Manager (Root Proxy).
"""

from __future__ import annotations

import logging

_proxy_logger = logging.getLogger("CoChem-SpycFit.CoreEngineProxy")

try:
    from src.cochem_spycfit.core_engine import *  # noqa: F401, F403
    from src.cochem_spycfit.core_engine import __all__ as _src_all
    __all__ = _src_all
except ImportError:
    from cochem_spycfit.core_engine.cochem_fit_registry_manager import *  # noqa: F401, F403
    try:
        from cochem_spycfit.core_engine.cochem_spycfit_init import *  # noqa: F401, F403
    except ImportError as _err:
        _proxy_logger.debug("Root core_engine proxy init import resolution completed: %s", _err)
    try:
        from cochem_spycfit.core_engine.cochem_spycfit_state import (  # noqa: F401
            CommitManager,
            FitState,
        )
    except ImportError as _err:
        _proxy_logger.debug("Root core_engine proxy state import resolution completed: %s", _err)
    try:
        from cochem_spycfit.core_engine.cochem_spycfit_hdf5_cache import (  # noqa: F401
            TensorCache,
        )
    except ImportError as _err:
        _proxy_logger.debug("Root core_engine proxy hdf5_cache import resolution completed: %s", _err)
    try:
        from cochem_spycfit.core_engine.cochem_os_adapter import *  # noqa: F401, F403
    except ImportError as _err:
        _proxy_logger.debug("Root core_engine proxy os_adapter import resolution completed: %s", _err)

    __all__ = [
        "AIRGAP_TIER_A_CORE",
        "AIRGAP_TIER_B_SCRATCH",
        "AIRGAP_TIER_C_SEALED",
        "AirGapBoundaryError",
        "AirGapValidationRecord",
        "CANONICAL_BINARY_HASHES",
        "CHUNK_SIZE_BYTES",
        "CoChemOSError",
        "CommitManager",
        "CryptographicSealingError",
        "DEFAULT_CHUNK_SIZE",
        "DEFAULT_LEGACY_HASHES",
        "DEFAULT_LOCAL_LOCK_TIMEOUT",
        "DEFAULT_LUSTRE_LOCK_TIMEOUT",
        "DEFAULT_STALE_LOCK_TIMEOUT",
        "FALLBACK_REQUIRED",
        "FileLockRecord",
        "FitParameterRecord",
        "FitRegistryCorruptionError",
        "FitRegistryError",
        "FitRegistryManager",
        "FitRegistryMissingError",
        "FitRegistrySchemaError",
        "FitRegistryState",
        "FitState",
        "HDF5TensorError",
        "HardwareDetectionError",
        "HardwareGateResult",
        "HardwareWarning",
        "ImmutableLockError",
        "InstrumentLimits",
        "JAXPrecisionConfig",
        "LUSTRE_SUPER_MAGIC",
        "LockState",
        "LustreLockError",
        "LustreMkdirLock",
        "MIN_GPU_VRAM_GB",
        "MLFeedbackState",
        "OSTier",
        "OSType",
        "POSIX_READONLY_DIR_MODE",
        "POSIX_READONLY_FILE_MODE",
        "POSIX_STRICT_PRIVATE_DIR_MODE",
        "POSIX_WRITABLE_DIR_MODE",
        "POSIX_WRITABLE_FILE_MODE",
        "PermissionSealingError",
        "PermissionUnsealingError",
        "RegistryLockError",
        "RegistryLockTimeoutError",
        "SEQUENTIAL_MINIBATCHING",
        "SUBPROCESS_TIMEOUT_S",
        "SnapshotMetadata",
        "SpycfitGatekeeperState",
        "StandardFileLock",
        "SyscallExecutionError",
        "TIER_BINARY_MAP",
        "TensorCache",
        "WINDOWS_SID_AUTHENTICATED_USERS",
        "WINDOWS_SID_EVERYONE",
        "XLACacheConfig",
        "apply_posix_chmod_readonly",
        "apply_posix_chmod_writable",
        "apply_windows_icacls_readonly",
        "apply_windows_icacls_writable",
        "assert_airgap_containment",
        "calculate_binary_sha256",
        "check_vram_and_set_flags",
        "classify_airgap_tier",
        "compute_file_sha256",
        "compute_path_sha256",
        "configure_xla_cache",
        "create_atomic_snapshot",
        "detect_os_type",
        "enforce_jax_precision",
        "evaluate_hardware_fallback_gate",
        "get_atomic_lock",
        "get_atomic_mass",
        "get_mendeleev_element_mass",
        "immutable_seal_scope",
        "initialize_spycfit_gatekeeper",
        "is_fallback_required",
        "is_file_readonly",
        "is_lustre_filesystem",
        "is_path_read_only",
        "is_posix_os",
        "is_sequential_minibatching",
        "is_windows_os",
        "lock_directory_strict_private",
        "open_state_tensor",
        "query_physical_gpu_vram",
        "resolve_cochem_system_config_path",
        "resolve_lock_timeout",
        "route_6tier_fortran_binary",
        "seal_artifact_immutable",
        "seal_directory_readonly",
        "temporary_unseal_scope",
        "unseal_artifact_mutable",
        "validate_airgap_isolation",
        "validate_legacy_hashes",
        "verify_binary_checksum",
        "verify_sealed_immutability",
    ]
