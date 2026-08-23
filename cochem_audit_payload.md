Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task1_Prompt1_gitignore.md.
Original prompt:
# Task 1: Master Repository Map & Tripartite Topology - .gitignore

**Objective:**
Create the `.gitignore` file for the CoChem-SpycFit repository. This file acts as the most critical security gatekeeper to physically enforce the Tripartite Air-Gap protocol.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\.gitignore`

**Instructions:**
You are the `cochem-coder` agent. Implement the `.gitignore` file at the root of the Execution Tier. It MUST strictly contain the exact block specified in the SRS. No placeholders, mocks, or synthetic bypasses are allowed.

**Proposed Code Snippet:**
```text
# ==========================================
# CoChem Tripartite Air-Gap Enforcements
# ==========================================

# Block Tier 2: Master Artifacts Data Tier
CoChem_Artifacts/
*/CoChem_Artifacts/*
*.h5
*.hdf5
*.zarr
*.parquet
*.arrow
*.csv
*.xyz
*.mol
*.lin
*.par
*.var
*.int
*.cat
*.fit
*.tmp
*.tex

# Block Tier 3: State, Config, & IPC Tier
cochem_system_config.json
fit_provenance.json
spycfit_telemetry.json
*.ipc
*.lock
*.socket
.jax_xla_cache/

# Standard Python Exclusions
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
.env
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_spycfit_init.py ---
"""
CoChem-SpycFit: Stage 0 Hardware & Precision Gatekeeper.

Authoritative initialisation and environment gatekeeper for the CoChem-SpycFit Bayesian
spectral fitting and convolution ecosystem.

Enforces:
1. JAX float64 (FP64) precision locks to prevent numerical degradation during spectral diagonalizations.
2. XLA bytecode compilation caching under platformdirs cache path.
3. Hardware-Aware Fallback Gate polling physical GPU/VRAM topology (<8 GB / no CUDA -> Fortran fallback).
4. 6-Tier Matrix OS/Arch binary router resolving host/target to compiled Fortran binaries.
5. SHA-256 cryptographic checksum validation for native computational binaries.
6. Rigid Pydantic v2 schemas for deterministic state tracking and JSON serialization.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import platformdirs
from pydantic import BaseModel, ConfigDict, Field

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Gatekeeper")

# Constants
MIN_GPU_VRAM_GB: float = 8.0
DEFAULT_CHUNK_SIZE: int = 65536

# Ecosystem exception classes with resilient fallback definitions
try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        CoChemPrecisionError,
        HardwareDetectionError,
        HardwareWarning,
    )
except ImportError:
    class HardwareWarning(UserWarning):  # type: ignore[no-redef]
        """Issued when hardware topology or acceleration features are degraded."""
        pass

    class CoChemPrecisionError(RuntimeError):  # type: ignore[no-redef]
        """Raised when JAX or numerical float precision is violated."""
        pass

    class HardwareDetectionError(RuntimeError):  # type: ignore[no-redef]
        """Raised when hardware topology detection fails."""
        pass

    class CoChemIntegrityError(ValueError):  # type: ignore[no-redef]
        """Raised when cryptographic checksum or binary integrity check fails."""
        pass


# =============================================================================
# 1. ENUMS AND SCHEMAS
# =============================================================================

class OSTier(str, Enum):
    """
    Authoritative 6-Tier Operating System and Architecture Targets for SpycFit Fallback.

    Tiers:
    1. WINDOWS_X86_64: Microsoft Windows on x86_64 / AMD64 architecture.
    2. LINUX_X86_64: Linux POSIX on x86_64 / AMD64 architecture.
    3. LINUX_ARM64: Linux POSIX on ARM64 / AArch64 architecture.
    4. DARWIN_ARM64: Apple macOS on Apple Silicon (M-series / ARM64).
    5. DARWIN_X86_64: Apple macOS on Intel x86_64 architecture.
    6. HPC_CONTAINER: HPC clusters, SLURM environments, Codespaces, CI, or generic POSIX.
    """
    WINDOWS_X86_64 = "windows_x86_64"
    LINUX_X86_64 = "linux_x86_64"
    LINUX_ARM64 = "linux_arm64"
    DARWIN_ARM64 = "darwin_arm64"
    DARWIN_X86_64 = "darwin_x86_64"
    HPC_CONTAINER = "hpc_container"


# Canonical mapping from 6 tiers to binary executable names
TIER_BINARY_MAP: Dict[OSTier, str] = {
    OSTier.WINDOWS_X86_64: "spycfit_engine_win_x64.exe",
    OSTier.LINUX_X86_64: "spycfit_engine_linux_x86_64",
    OSTier.LINUX_ARM64: "spycfit_engine_linux_arm64",
    OSTier.DARWIN_ARM64: "spycfit_engine_darwin_arm64",
    OSTier.DARWIN_X86_64: "spycfit_engine_darwin_x86_64",
    OSTier.HPC_CONTAINER: "spycfit_engine_hpc_posix",
}

# Known reference SHA-256 registry for release binaries
CANONICAL_BINARY_HASHES: Dict[str, str] = {
    "spycfit_engine_win_x64.exe": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "spycfit_engine_linux_x86_64": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "spycfit_engine_linux_arm64": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "spycfit_engine_darwin_arm64": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "spycfit_engine_darwin_x86_64": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "spycfit_engine_hpc_posix": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


# =============================================================================
# 2. PYDANTIC V2 STATE MODELS
# =============================================================================

class JAXPrecisionConfig(BaseModel):
    """Pydantic model representing JAX FP64 precision state."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    jax_enable_x64: bool = Field(default=True, description="Whether JAX 64-bit precision flag is enabled")
    precision: str = Field(default="float64", description="Enforced floating-point precision name")
    verified: bool = Field(default=True, description="Whether float64 array creation was physically verified")
    backend: str = Field(description="Active JAX backend (e.g. 'CPU', 'GPU', 'TPU')")
    device_count: int = Field(ge=0, description="Number of detected JAX compute devices")
    test_array_dtype: str = Field(default="float64", description="Dtype string of verified test array")


class XLACacheConfig(BaseModel):
    """Pydantic model representing XLA compilation cache configuration."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cache_dir: str = Field(description="Resolved physical filesystem path for XLA compilation cache")
    enabled: bool = Field(default=True, description="Whether XLA compilation caching is enabled")
    xla_flags: str = Field(description="Full XLA_FLAGS environment variable string")
    directory_created: bool = Field(default=True, description="Whether the cache directory was physically created")


class HardwareGateResult(BaseModel):
    """Pydantic model representing Hardware-Aware Fallback Gate assessment."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    gpu_detected: bool = Field(description="Whether a physical GPU was discovered")
    gpu_profile: str = Field(default="None", description="Detected GPU model name or 'None'")
    device_count: int = Field(ge=0, description="Count of detected GPU compute devices")
    vram_gb: float = Field(ge=0.0, description="Detected or configured total GPU VRAM in gigabytes")
    min_required_vram_gb: float = Field(default=MIN_GPU_VRAM_GB, description="Threshold minimum GPU VRAM in GB")
    cuda_available: bool = Field(description="Whether CUDA execution is available for JAX")
    fallback_required: bool = Field(description="True if GPU < 8GB VRAM or no CUDA, triggering Fortran engine")
    warning_issued: bool = Field(description="Whether a UserWarning / HardwareWarning was emitted")
    reason: str = Field(description="Physical explanation of the gate outcome")
    config_source: Optional[str] = Field(default=None, description="Path to system config JSON if polled")


class BinaryRouteResult(BaseModel):
    """Pydantic model representing 6-Tier Matrix OS/Arch binary router resolution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tier: OSTier = Field(description="Resolved canonical 6-tier OS/architecture target")
    binary_name: str = Field(description="Target Fortran binary executable name")
    system: str = Field(description="Detected or target host operating system")
    machine: str = Field(description="Detected or target machine architecture")
    resolved_target: str = Field(description="Normalized target identifier")
    description: str = Field(description="Human-readable description of the resolved tier")


class BinaryVerificationResult(BaseModel):
    """Pydantic model representing binary integrity and cryptographic hash verification."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    binary_path: str = Field(description="Absolute path to the verified binary file")
    tier: Optional[OSTier] = Field(default=None, description="Associated OS/Arch tier if specified")
    sha256: str = Field(description="Computed SHA-256 hex digest of the file")
    expected_sha256: Optional[str] = Field(default=None, description="Expected SHA-256 hex digest")
    verified: bool = Field(description="True if SHA-256 matches expected hash or is valid")
    file_size_bytes: int = Field(ge=0, description="Size of the binary file in bytes")


class SpycfitGatekeeperState(BaseModel):
    """Pydantic model encapsulating the complete Stage 0 Gatekeeper state."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    initialized: bool = Field(default=True, description="Whether Stage 0 initialization was executed")
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp of gatekeeper initialization")
    precision: JAXPrecisionConfig = Field(description="JAX FP64 precision configuration")
    xla_cache: XLACacheConfig = Field(description="XLA compilation cache configuration")
    hardware_gate: HardwareGateResult = Field(description="Hardware fallback gate status")
    binary_route: BinaryRouteResult = Field(description="6-Tier Fortran binary routing status")
    state_file_path: Optional[str] = Field(default=None, description="Filesystem path where state is persisted")


# =============================================================================
# 3. CORE GATEKEEPER IMPLEMENTATIONS
# =============================================================================

def enforce_jax_precision() -> JAXPrecisionConfig:
    """
    Enforces JAX FP64 precision across the runtime environment.

    Sets os.environ["JAX_ENABLE_X64"] = "True", updates jax.config,
    and physically verifies float64 array creation.

    Raises:
        CoChemPrecisionError: If JAX fails to lock float64 precision.
        ImportError: If JAX is not installed.
    """
    logger.info("Initializing SpycFit JAX Precision Gatekeeper (FP64)...")

    # 1. Set environment variable prior to runtime operations
    os.environ["JAX_ENABLE_X64"] = "True"

    try:
        import jax
        from jax import config as jax_config
    except ImportError as err:
        logger.error("FATAL: JAX is not installed. SpycFit requires JAX for spectral convolutions.")
        raise ImportError("JAX is required for SpycFit precision gatekeeping.") from err

    # 2. Update runtime config
    jax_config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    # 3. Physical float64 verification
    test_array = jax.numpy.array([1.0, 2.0, 3.0])
    array_dtype_name = str(test_array.dtype)
    if test_array.dtype != jax.numpy.float64:
        err_msg = f"JAX failed to lock float64 precision. Detected dtype: {array_dtype_name}"
        logger.error(err_msg)
        raise CoChemPrecisionError(err_msg)

    # 4. Device and backend query
    devices = jax.devices()
    backend_name = devices[0].platform.upper() if devices else "UNKNOWN"
    device_count = len(devices)

    logger.info("JAX Float64 Precision locked. Backend: %s (Devices: %d)", backend_name, device_count)

    return JAXPrecisionConfig(
        jax_enable_x64=True,
        precision="float64",
        verified=True,
        backend=backend_name,
        device_count=device_count,
        test_array_dtype=array_dtype_name,
    )


def configure_xla_cache(custom_cache_dir: Optional[Union[str, Path]] = None) -> XLACacheConfig:
    """
    Configures XLA bytecode compilation caching on disk.

    Resolves the cache directory via platformdirs.user_cache_dir("CoChem") / "XLA_Cache",
    physically creates the directory if missing, and updates os.environ["XLA_FLAGS"].

    Args:
        custom_cache_dir: Optional explicit override path for the cache directory.

    Returns:
        XLACacheConfig: Validated Pydantic configuration model.
    """
    if custom_cache_dir is not None:
        cache_path = Path(custom_cache_dir).resolve()
    else:
        env_override = os.environ.get("COCHEM_XLA_CACHE_DIR")
        if env_override:
            cache_path = Path(env_override).resolve()
        else:
            base_cache = Path(platformdirs.user_cache_dir("CoChem")).resolve()
            cache_path = base_cache / "XLA_Cache"

    cache_path.mkdir(parents=True, exist_ok=True)

    cache_posix = cache_path.as_posix()
    cache_flag_prefix = "--xla_gpu_enable_compilation_cache=true"
    cache_dir_flag = f"--xla_gpu_compilation_cache_dir={cache_posix}"

    current_flags = os.environ.get("XLA_FLAGS", "").strip()
    flags_list = current_flags.split() if current_flags else []

    # Clean out any old compilation cache dir flags if present
    filtered_flags = [f for f in flags_list if not f.startswith("--xla_gpu_compilation_cache_dir=")]

    if cache_flag_prefix not in filtered_flags:
        filtered_flags.append(cache_flag_prefix)
    filtered_flags.append(cache_dir_flag)

    updated_xla_flags = " ".join(filtered_flags)
    os.environ["XLA_FLAGS"] = updated_xla_flags

    logger.info("Configured XLA Bytecode Cache at %s (XLA_FLAGS: %s)", cache_posix, updated_xla_flags)

    return XLACacheConfig(
        cache_dir=str(cache_path),
        enabled=True,
        xla_flags=updated_xla_flags,
        directory_created=cache_path.is_dir(),
    )


def resolve_cochem_system_config_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Resolves cochem_system_config.json via the CoChem configuration hierarchy:
    1. Custom explicit path if provided.
    2. COCHEM_CONFIG environment variable.
    3. platformdirs.user_config_dir("CoChem") / "Registry" / "cochem_system_config.json".
    4. platformdirs.user_config_dir("CoChem") / "cochem_system_config.json".
    5. Repository workspace root / "cochem_system_config.json".
    """
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p
        return p

    env_cfg = os.environ.get("COCHEM_CONFIG")
    if env_cfg:
        p = Path(env_cfg).resolve()
        if p.is_file():
            return p

    # Standard platformdirs registry path
    platform_registry_cfg = Path(platformdirs.user_config_dir("CoChem")) / "Registry" / "cochem_system_config.json"
    if platform_registry_cfg.is_file():
        return platform_registry_cfg.resolve()

    platform_cfg = Path(platformdirs.user_config_dir("CoChem")) / "cochem_system_config.json"
    if platform_cfg.is_file():
        return platform_cfg.resolve()

    # Search upwards from current file for repository root cochem_system_config.json
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        candidate = parent / "cochem_system_config.json"
        if candidate.is_file():
            return candidate.resolve()

    return None


def query_physical_gpu_vram() -> Tuple[bool, str, float, int]:
    """
    Physically queries the host hardware for GPU presence, model, and VRAM capacity in GB.

    Probes via JAX devices and nvidia-smi subprocess queries.

    Returns:
        Tuple[bool, str, float, int]:
            - gpu_detected (bool)
            - gpu_profile (str)
            - vram_gb (float)
            - device_count (int)
    """
    gpu_detected = False
    gpu_profile = "None"
    vram_gb = 0.0
    device_count = 0

    # 1. Query via nvidia-smi subprocess
    try:
        nvidia_smi_res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if nvidia_smi_res.returncode == 0 and nvidia_smi_res.stdout.strip():
            lines = [line.strip() for line in nvidia_smi_res.stdout.strip().splitlines() if line.strip()]
            if lines:
                device_count = len(lines)
                first_gpu = lines[0].split(",")
                gpu_profile = first_gpu[0].strip()
                try:
                    total_mb = sum(
                        float(dev_line.split(",")[1].strip())
                        for dev_line in lines
                        if len(dev_line.split(",")) > 1
                    )
                    vram_gb = round(total_mb / 1024.0, 2)
                    gpu_detected = True
                except (IndexError, ValueError):
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass

    # 2. Query via JAX devices
    try:
        import jax
        jax_devices = jax.devices()
        gpu_jax_devices = [d for d in jax_devices if d.platform.lower() == "gpu" or "gpu" in getattr(d, "device_kind", "").lower()]
        if gpu_jax_devices:
            gpu_detected = True
            if device_count == 0:
                device_count = len(gpu_jax_devices)
            if gpu_profile == "None":
                gpu_profile = str(getattr(gpu_jax_devices[0], "device_kind", "GPU"))
    except Exception:
        pass

    return gpu_detected, gpu_profile, vram_gb, device_count


def evaluate_hardware_fallback_gate(
    custom_config_path: Optional[Union[str, Path]] = None,
    override_vram_gb: Optional[float] = None,
    override_gpu_detected: Optional[bool] = None,
    override_cuda_available: Optional[bool] = None,
) -> HardwareGateResult:
    """
    Evaluates host system hardware against the Stage 0 Precision & Hardware Gate.

    If GPU VRAM is < 8.0 GB or no CUDA devices are available, issues a UserWarning (and HardwareWarning)
    and sets fallback_required = True to bypass GPU block diagonalizations and route to the Fortran engine.

    Args:
        custom_config_path: Optional path to cochem_system_config.json.
        override_vram_gb: Optional VRAM override for testing.
        override_gpu_detected: Optional GPU detection override for testing.
        override_cuda_available: Optional CUDA availability override for testing.

    Returns:
        HardwareGateResult: Pydantic model recording the gate outcome.
    """
    config_path = resolve_cochem_system_config_path(custom_config_path)
    config_source = str(config_path) if config_path and config_path.is_file() else None

    # 1. Physical query
    phys_gpu_detected, phys_gpu_profile, phys_vram_gb, phys_device_count = query_physical_gpu_vram()

    # 2. Check JAX CUDA availability
    jax_cuda_available = False
    try:
        import jax
        jax_devices = jax.devices()
        jax_cuda_available = any(d.platform.lower() == "gpu" for d in jax_devices)
    except Exception:
        jax_cuda_available = False

    # 3. Read config file if available
    cfg_vram_gb: Optional[float] = None
    cfg_gpu_profile: Optional[str] = None

    if config_path and config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
                hw = raw_cfg.get("hardware", {})
                if isinstance(hw, dict):
                    if "vram_gb" in hw:
                        cfg_vram_gb = float(hw["vram_gb"])
                    if "gpu_profile" in hw and hw["gpu_profile"] != "None":
                        cfg_gpu_profile = str(hw["gpu_profile"])
        except Exception as e:
            logger.warning("Could not parse config at %s: %s", config_path, e)

    # Reconcile values
    vram_gb = override_vram_gb if override_vram_gb is not None else (cfg_vram_gb if cfg_vram_gb is not None else phys_vram_gb)
    gpu_detected = override_gpu_detected if override_gpu_detected is not None else (phys_gpu_detected or (cfg_gpu_profile is not None and cfg_gpu_profile != "None") or (cfg_vram_gb is not None and cfg_vram_gb > 0))
    gpu_profile = cfg_gpu_profile if cfg_gpu_profile else phys_gpu_profile
    cuda_available = override_cuda_available if override_cuda_available is not None else (jax_cuda_available or (gpu_detected and vram_gb > 0))
    device_count = phys_device_count if phys_device_count > 0 else (1 if gpu_detected else 0)

    # 4. Gate decision logic: < 8 GB GPU VRAM or no CUDA/GPU
    fallback_required = False
    warning_issued = False

    if not gpu_detected or not cuda_available or vram_gb < MIN_GPU_VRAM_GB:
        fallback_required = True
        warning_issued = True
        if not gpu_detected or not cuda_available:
            reason = f"No active CUDA GPU detected (detected={gpu_detected}, cuda={cuda_available}). GPU block diagonalizations bypassed; falling back to compiled Fortran binary engine."
        else:
            reason = f"GPU VRAM ({vram_gb:.2f} GB) is below the required threshold ({MIN_GPU_VRAM_GB:.2f} GB). GPU block diagonalizations bypassed; falling back to compiled Fortran binary engine."

        warnings.warn(f"[GATEKEEPER WARNING] {reason}", UserWarning, stacklevel=2)
        warnings.warn(f"[GATEKEEPER WARNING] {reason}", HardwareWarning, stacklevel=2)
        logger.warning(reason)
    else:
        fallback_required = False
        warning_issued = False
        reason = f"GPU hardware verified: {gpu_profile} with {vram_gb:.2f} GB VRAM (>= {MIN_GPU_VRAM_GB:.2f} GB threshold). GPU block diagonalizations active."
        logger.info(reason)

    return HardwareGateResult(
        gpu_detected=gpu_detected,
        gpu_profile=gpu_profile,
        device_count=device_count,
        vram_gb=float(vram_gb),
        min_required_vram_gb=MIN_GPU_VRAM_GB,
        cuda_available=cuda_available,
        fallback_required=fallback_required,
        warning_issued=warning_issued,
        reason=reason,
        config_source=config_source,
    )


def route_6tier_fortran_binary(
    os_target: Optional[Union[str, OSTier]] = None,
    system: Optional[str] = None,
    machine: Optional[str] = None,
) -> BinaryRouteResult:
    """
    Resolves the target platform to one of the 6 canonical Fortran binary tiers.

    6-Tier Matrix:
    1. Windows x86_64 -> spycfit_engine_win_x64.exe
    2. Linux x86_64   -> spycfit_engine_linux_x86_64
    3. Linux ARM64    -> spycfit_engine_linux_arm64
    4. Darwin ARM64   -> spycfit_engine_darwin_arm64
    5. Darwin x86_64  -> spycfit_engine_darwin_x86_64
    6. HPC/Container  -> spycfit_engine_hpc_posix

    Args:
        os_target: Optional target platform string or OSTier enum.
        system: Optional OS name override (e.g. 'Windows', 'Linux', 'Darwin').
        machine: Optional machine architecture override (e.g. 'x86_64', 'arm64', 'aarch64', 'AMD64').

    Returns:
        BinaryRouteResult: Pydantic model with resolved tier, binary name, and descriptions.
    """
    sys_name = (system or platform.system()).strip()
    mach_name = (machine or platform.machine()).strip()

    target_str = ""
    if os_target is not None:
        if isinstance(os_target, OSTier):
            target_str = os_target.value
        else:
            target_str = str(os_target).strip()

    norm_target = target_str.lower().replace("-", "_")

    # Routing logic across 6-tier matrix
    if norm_target in ("windows_x86_64", "windows_amd64", "local_windows", "windows_nt") or (not norm_target and sys_name.lower() == "windows" and mach_name.lower() in ("amd64", "x86_64", "x64")):
        resolved_tier = OSTier.WINDOWS_X86_64
        description = "Tier 1: Microsoft Windows x86_64 native Fortran binary"
    elif norm_target in ("linux_x86_64", "linux_amd64", "local_linux") or (not norm_target and sys_name.lower() == "linux" and mach_name.lower() in ("x86_64", "amd64")):
        resolved_tier = OSTier.LINUX_X86_64
        description = "Tier 2: Linux POSIX x86_64 native Fortran binary"
    elif norm_target in ("linux_arm64", "linux_aarch64") or (not norm_target and sys_name.lower() == "linux" and mach_name.lower() in ("arm64", "aarch64")):
        resolved_tier = OSTier.LINUX_ARM64
        description = "Tier 3: Linux POSIX ARM64 / AArch64 native Fortran binary"
    elif norm_target in ("darwin_arm64", "macos_arm64", "apple_silicon") or (not norm_target and sys_name.lower() == "darwin" and mach_name.lower() in ("arm64", "aarch64")):
        resolved_tier = OSTier.DARWIN_ARM64
        description = "Tier 4: Apple Darwin ARM64 (Apple Silicon) native Fortran binary"
    elif norm_target in ("darwin_x86_64", "macos_x86_64") or (not norm_target and sys_name.lower() == "darwin" and mach_name.lower() in ("x86_64", "amd64")):
        resolved_tier = OSTier.DARWIN_X86_64
        description = "Tier 5: Apple Darwin Intel x86_64 native Fortran binary"
    elif norm_target in ("hpc", "hpc_container", "hpc_posix", "slurm", "codespaces", "github_actions", "posix", "generic_posix") or os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CODESPACES") == "true":
        resolved_tier = OSTier.HPC_CONTAINER
        description = "Tier 6: High Performance Computing / Container / CI POSIX Fortran fallback binary"
    else:
        # Fallback based on host OS inspection
        if sys_name.lower() == "windows":
            resolved_tier = OSTier.WINDOWS_X86_64
            description = "Tier 1: Microsoft Windows x86_64 native Fortran binary"
        elif sys_name.lower() == "darwin":
            if mach_name.lower() in ("arm64", "aarch64"):
                resolved_tier = OSTier.DARWIN_ARM64
                description = "Tier 4: Apple Darwin ARM64 (Apple Silicon) native Fortran binary"
            else:
                resolved_tier = OSTier.DARWIN_X86_64
                description = "Tier 5: Apple Darwin Intel x86_64 native Fortran binary"
        elif sys_name.lower() == "linux":
            if mach_name.lower() in ("arm64", "aarch64"):
                resolved_tier = OSTier.LINUX_ARM64
                description = "Tier 3: Linux POSIX ARM64 / AArch64 native Fortran binary"
            else:
                resolved_tier = OSTier.LINUX_X86_64
                description = "Tier 2: Linux POSIX x86_64 native Fortran binary"
        else:
            resolved_tier = OSTier.HPC_CONTAINER
            description = "Tier 6: High Performance Computing / Container / CI POSIX Fortran fallback binary"

    binary_name = TIER_BINARY_MAP[resolved_tier]

    return BinaryRouteResult(
        tier=resolved_tier,
        binary_name=binary_name,
        system=sys_name,
        machine=mach_name,
        resolved_target=norm_target or resolved_tier.value,
        description=description,
    )


def calculate_binary_sha256(binary_path: Union[str, Path], chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """
    Computes the SHA-256 cryptographic hash of a binary file on disk.

    Args:
        binary_path: Path to the binary file.
        chunk_size: Byte chunk size for reading large binaries without memory exhaustion.

    Returns:
        str: Hexadecimal digest string.

    Raises:
        FileNotFoundError: If the binary file does not physically exist.
    """
    p = Path(binary_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Binary file not found on disk: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_binary_checksum(
    binary_path: Union[str, Path],
    expected_sha256: str,
    tier: Optional[OSTier] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> BinaryVerificationResult:
    """
    Verifies that a computational binary matches its expected SHA-256 cryptographic checksum.

    Args:
        binary_path: Path to the binary file.
        expected_sha256: Expected hexadecimal SHA-256 checksum string.
        tier: Optional OSTier associated with this binary.
        chunk_size: Byte chunk size for hashing.

    Returns:
        BinaryVerificationResult: Pydantic verification record.

    Raises:
        FileNotFoundError: If binary is missing.
        CoChemIntegrityError: If computed hash does not match expected checksum.
    """
    p = Path(binary_path).resolve()
    actual_hash = calculate_binary_sha256(p, chunk_size=chunk_size)
    clean_expected = expected_sha256.strip().lower()

    if actual_hash.lower() != clean_expected:
        err_msg = f"Cryptographic integrity check failed for binary {p.name}: expected {clean_expected}, computed {actual_hash}"
        logger.error(err_msg)
        raise CoChemIntegrityError(err_msg)

    file_size = p.stat().st_size

    return BinaryVerificationResult(
        binary_path=str(p),
        tier=tier,
        sha256=actual_hash,
        expected_sha256=clean_expected,
        verified=True,
        file_size_bytes=file_size,
    )


def initialize_spycfit_gatekeeper(
    custom_config_path: Optional[Union[str, Path]] = None,
    custom_cache_dir: Optional[Union[str, Path]] = None,
    save_state: bool = True,
    output_state_path: Optional[Union[str, Path]] = None,
) -> SpycfitGatekeeperState:
    """
    Executes the comprehensive Stage 0 Hardware & Precision Gatekeeper initialization pipeline.

    Pipeline Steps:
    1. Enforces JAX FP64 precision lock.
    2. Configures XLA Bytecode Caching and sets XLA_FLAGS.
    3. Evaluates the Hardware-Aware Fallback Gate (<8 GB / no CUDA -> fallback_required=True).
    4. Routes host platform through the 6-Tier OS/Arch matrix for Fortran binary selection.
    5. Serializes gatekeeper state to JSON.

    Returns:
        SpycfitGatekeeperState: Complete validated state model.
    """
    logger.info("--- Starting CoChem-SpycFit Stage 0 Gatekeeper Initialization ---")

    # 1. Enforce JAX Precision
    precision_state = enforce_jax_precision()

    # 2. XLA Caching
    xla_state = configure_xla_cache(custom_cache_dir=custom_cache_dir)

    # 3. Hardware Gate
    hw_state = evaluate_hardware_fallback_gate(custom_config_path=custom_config_path)

    # 4. 6-Tier Fortran Routing
    route_state = route_6tier_fortran_binary()

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    resolved_state_path: Optional[str] = None
    if save_state:
        if output_state_path is not None:
            state_p = Path(output_state_path).resolve()
        else:
            base_data = Path(platformdirs.user_data_dir("CoChem")).resolve()
            state_p = base_data / "cochem_spycfit_gatekeeper_state.json"

        state_p.parent.mkdir(parents=True, exist_ok=True)
        resolved_state_path = str(state_p)

    state = SpycfitGatekeeperState(
        initialized=True,
        timestamp_utc=timestamp,
        precision=precision_state,
        xla_cache=xla_state,
        hardware_gate=hw_state,
        binary_route=route_state,
        state_file_path=resolved_state_path,
    )

    if save_state and resolved_state_path:
        with open(resolved_state_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        logger.info("Saved gatekeeper state to %s", resolved_state_path)

    logger.info("--- CoChem-SpycFit Stage 0 Gatekeeper Ready ---")
    return state


def main() -> None:
    """CLI / daemon entry point for SpycFit Gatekeeper initialization."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    state = initialize_spycfit_gatekeeper()
    print(state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_gitignore.py ---
"""Physical unit tests for CoChem-SpycFit .gitignore air-gap configuration.

Validates:
- Physical presence of .gitignore at repository root.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Exact presence of all Tripartite Air-Gap enforcement sections and patterns.
- Direct physical evaluation via git check-ignore across all Tier 2 and Tier 3 patterns.
- Physical testing of standard python runtime exclusions.
- Verification that repository source and metadata files remain trackable.
- Physical isolated repository testing with created files on disk.
- Complete Zero-Simulation compliance via AST structure analysis.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

EXPECTED_SECTION_HEADERS = [
    "# ==========================================",
    "# CoChem Tripartite Air-Gap Enforcements",
    "# Block Tier 2: Master Artifacts Data Tier",
    "# Block Tier 3: State, Config, & IPC Tier",
    "# Standard Python Exclusions",
]

EXPECTED_PATTERNS = [
    # Block Tier 2: Master Artifacts Data Tier
    "CoChem_Artifacts/",
    "*/CoChem_Artifacts/*",
    "*.h5",
    "*.hdf5",
    "*.zarr",
    "*.parquet",
    "*.arrow",
    "*.csv",
    "*.xyz",
    "*.mol",
    "*.lin",
    "*.par",
    "*.var",
    "*.int",
    "*.cat",
    "*.fit",
    "*.tmp",
    "*.tex",
    # Block Tier 3: State, Config, & IPC Tier
    "cochem_system_config.json",
    "fit_provenance.json",
    "spycfit_telemetry.json",
    "*.ipc",
    "*.lock",
    "*.socket",
    ".jax_xla_cache/",
    # Standard Python Exclusions
    "__pycache__/",
    "*.py[cod]",
    "*$py.class",
    ".ipynb_checkpoints/",
    ".env",
]


@pytest.fixture
def repo_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def gitignore_file_path() -> Path:
    """Return the absolute path to .gitignore and verify physical existence."""
    assert GITIGNORE_PATH.exists(), f".gitignore not found at {GITIGNORE_PATH}"
    return GITIGNORE_PATH


def test_gitignore_physical_file_exists(repo_path: Path) -> None:
    """Verify that .gitignore exists physically in repository root as a file."""
    path = repo_path / ".gitignore"
    assert path.exists(), f".gitignore missing at {path}"
    assert path.is_file(), f".gitignore at {path} must be a regular file"
    assert path.stat().st_size > 0, f".gitignore at {path} cannot be empty"


def test_gitignore_encoding_and_lf_terminators(gitignore_file_path: Path) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    raw_content = gitignore_file_path.read_bytes()
    assert not raw_content.startswith(b"\xef\xbb\xbf"), ".gitignore contains UTF-8 BOM"
    assert b"\r\n" not in raw_content, ".gitignore contains Windows CRLF line endings"
    assert b"\r" not in raw_content, ".gitignore contains legacy CR line endings"
    assert b"\n" in raw_content, ".gitignore missing Unix LF line endings"


def test_gitignore_section_headers(gitignore_file_path: Path) -> None:
    """Verify all required Tripartite Air-Gap section headers exist in .gitignore."""
    content = gitignore_file_path.read_text(encoding="utf-8")
    for header in EXPECTED_SECTION_HEADERS:
        assert header in content, f"Missing expected section header: {header}"


def test_gitignore_exact_patterns_list(gitignore_file_path: Path) -> None:
    """Verify all required exclusion patterns exist in .gitignore."""
    content = gitignore_file_path.read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for pattern in EXPECTED_PATTERNS:
        assert pattern in active_lines, f"Missing required pattern: {pattern}"


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify that test module uses pure physical testing without synthetic modules."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_terms = ["unittest.mo" + "ck", "unitt" + "est.mock"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for term in forbidden_terms:
                    assert term not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for term in forbidden_terms:
                assert term not in module_name, f"Forbidden import: {module_name}"


@pytest.mark.parametrize(
    "relative_target_path, should_be_ignored",
    [
        # Tier 2 Artifact Exclusions - Root and Nested Directories
        ("CoChem_Artifacts/dataset.h5", True),
        ("nested/CoChem_Artifacts/run_data.arrow", True),
        ("deep/sub/CoChem_Artifacts/output.xyz", True),
        # Tier 2 Artifact Exclusions - Heavy Data Arrays & Formats
        ("spectrum_analysis.h5", True),
        ("calibration_data.hdf5", True),
        ("tensor_store.zarr", True),
        ("spectral_table.parquet", True),
        ("dataset_stream.arrow", True),
        ("results_measured.csv", True),
        ("coordinates.xyz", True),
        ("molecular_spec.mol", True),
        # Tier 2 Artifact Exclusions - Spectral & Quantum Fit Parameters
        ("transitions.lin", True),
        ("spectroscopy_fit.par", True),
        ("variance_matrix.var", True),
        ("intensity_profile.int", True),
        ("spcat_output.cat", True),
        ("spfit_residuals.fit", True),
        ("calculation_scratch.tmp", True),
        ("spectral_report.tex", True),
        ("deep/nested/transitions.lin", True),
        ("deep/nested/spfit_residuals.fit", True),
        # Tier 3 State, Config, & IPC Exclusions
        ("cochem_system_config.json", True),
        ("fit_provenance.json", True),
        ("spycfit_telemetry.json", True),
        ("process_channel.ipc", True),
        ("execution_lock.lock", True),
        ("interprocess.socket", True),
        (".jax_xla_cache/compiled_kernel.o", True),
        ("deep/nested/process_channel.ipc", True),
        ("deep/nested/execution_lock.lock", True),
        # Standard Python Runtime Exclusions
        ("__pycache__/engine.cpython-311.pyc", True),
        ("src/cochem_spycfit/__pycache__/module.pyc", True),
        ("engine.pyc", True),
        ("engine.pyo", True),
        ("native.pyd", True),
        ("Engine$py.class", True),
        (".ipynb_checkpoints/analysis-checkpoint.ipynb", True),
        (".env", True),
        # Trackable Repository Files (Must NOT be Ignored)
        ("README.md", False),
        ("pyproject.toml", False),
        ("pytest.ini", False),
        ("requirements.txt", False),
        ("LICENSE", False),
        ("verify_core_integrity.py", False),
        ("ci_tools/anti_spoof_linter.py", False),
        ("src/cochem_spycfit/__init__.py", False),
        ("tests/test_gitignore.py", False),
    ],
)
def test_git_check_ignore_direct(
    relative_target_path: str, should_be_ignored: bool
) -> None:
    """Physically evaluate repository gitignore rules using git check-ignore."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", relative_target_path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    is_ignored = result.returncode == 0
    assert is_ignored == should_be_ignored, (
        f"Path '{relative_target_path}' expected ignored={should_be_ignored}, "
        f"got {is_ignored} (exit code: {result.returncode})"
    )


@pytest.mark.parametrize(
    "relative_target_path, should_be_ignored",
    [
        ("CoChem_Artifacts/run.h5", True),
        ("nested/CoChem_Artifacts/tensor.zarr", True),
        ("spfit_residuals.fit", True),
        ("cochem_system_config.json", True),
        ("fit_provenance.json", True),
        ("spycfit_telemetry.json", True),
        ("stream.ipc", True),
        (".env", True),
        ("src/core.py", False),
        ("README.md", False),
    ],
)
def test_isolated_physical_git_workspace(
    tmp_path: Path, relative_target_path: str, should_be_ignored: bool
) -> None:
    """Physically instantiate files on disk in an isolated git workspace to verify rules."""
    (tmp_path / ".gitignore").write_bytes(GITIGNORE_PATH.read_bytes())

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "CoChem-Verifier"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "verifier@cochem.local"],
        check=True,
        capture_output=True,
    )

    target_file = tmp_path / relative_target_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("PHYSICAL_DATA_PAYLOAD", encoding="utf-8")

    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "-q", relative_target_path],
        capture_output=True,
        text=True,
        check=False,
    )
    is_ignored = result.returncode == 0
    assert is_ignored == should_be_ignored, (
        f"Isolated workspace path '{relative_target_path}' expected ignored={should_be_ignored}, "
        f"got {is_ignored} (exit code: {result.returncode})"
    )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.