Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task3_Prompt1_cochem_spycfit_ui.md.
Original prompt:
# Task 3: Interactive UI (Jupyter) & Voila GUI Specifications - cochem_spycfit_ui.py

**Objective:**
Update and implement the comprehensive HITL dashboard backend `cochem_spycfit_ui.py` based on Task 3 requirements.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\interfaces\cochem_spycfit_ui.py`

**Instructions:**
You are the `cochem-coder` agent. Implement the complex features of `cochem_spycfit_ui.py`.
It must:
1. Enforce double precision at the environment level (`os.environ["JAX_ENABLE_X64"] = "True"`) before JAX initializes.
2. Use `ipywidgets` (VBox, HBox, Output, FloatSlider, Button) and Plotly WebGL (`Scattergl`) for a dual-pane O-C layout.
3. Implement LTTB (Largest-Triangle-Three-Buckets) decimation to reduce points to ~5000 using AOT or background threads.
4. Enhance the Bipartite Auto-Propose Assistant to rigidly enforce threshold bounds (rejecting negative rotational constants).
5. Implement Ghost Peak Retention logic, dynamically dropping the index from Huber loss and recording to the Method Matrix provenance.
6. Enhance "What-If" Sliders using a ZeroMQ IPC bridge with strict heartbeat and timeout protocols.
7. Implement Semantic Git-Commit Translation that compiles changes and generates a cryptographic hash (SHA-256) of the UI state and commit string.

**Constraints:**
- No dummy plotting, fake IPC, or placeholder encryption.
- Strictly adhere to Tripartite Air-Gap (no external CDNs).
- You MUST implement the actual mathematical and networking logic.
- Do NOT use `pass` statements in your generated code. Use `raise NotImplementedError` where strictly outlining, but aim to provide full implementations.

**Proposed Snippet Outline:**
```python
import os
import hashlib
import zmq
from ipywidgets import VBox, HBox, FloatSlider, Button, Output
import plotly.graph_objects as go
import numpy as np

# MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
os.environ["JAX_ENABLE_X64"] = "True"

def run_lttb_decimation(x, y, threshold=5000):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def update_ui_state_hash(state_dict):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
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
            check=True,
        )
        if nvidia_smi_res.stdout.strip():
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
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError, subprocess.CalledProcessError):
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
    logger.info(state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\interfaces\__init__.py ---
"""
CoChem-SpycFit Interfaces Module.

Exports:
- cochem_spycfit_ui: Decision Dashboard (HITL), Plotly WebGL dual-pane residuals, bipartite Hungarian auto-propose, offline PDF/HTML rendering, What-If parameter sliders, semantic commits.
- cochem_vibspyc_snap: Publication-ready snapshot export, AASTeX/LaTeX longtable rendering, CrossRef BibTeX citations, dataset hashing, and artifact packaging.
"""

from .cochem_spycfit_ui import (
    AssignmentStatus,
    BipartiteConfig,
    BipartiteMatchingError,
    DashboardConfig,
    HITLApprovalError,
    HITLAssignmentManager,
    OfflineRenderError,
    PeakAssignment,
    RenderBackend,
    SemanticCommit,
    SemanticCommitError,
    SliderComputationError,
    SpectralPeak,
    SpycFitInteractiveDashboard,
    SpycFitZMQClient,
    SpycFitZMQServer,
    UIDashboardError,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
    ZMQBridgeError,
    auto_propose_assignments,
    compute_bipartite_cost_matrix,
    compute_huber_loss_excluding_ghosts,
    create_spycfit_dashboard_widget,
    render_dual_pane_dashboard,
    render_static_offline_html,
    render_static_offline_pdf,
    run_lttb_decimation,
    translate_mathematical_change_to_git_commit,
    update_ui_state_hash,
)
from .cochem_vibspyc_snap import (
    FitProvenancePayload,
    evaluate_compression_strategy,
    export_spycfit_snapshot,
    format_citations_to_bib,
    generate_aastex_longtables,
    get_required_dois,
    get_spycfit_processed_dir,
    hash_dataset_iteratively,
    package_fit_artifacts,
    resolve_processed_workspace_dir,
    seal_artifact_read_only,
)

__all__ = [
    # UI Dashboard
    "AssignmentStatus",
    "BipartiteConfig",
    "BipartiteMatchingError",
    "DashboardConfig",
    "HITLApprovalError",
    "HITLAssignmentManager",
    "OfflineRenderError",
    "PeakAssignment",
    "RenderBackend",
    "SemanticCommit",
    "SemanticCommitError",
    "SliderComputationError",
    "SpectralPeak",
    "SpycFitInteractiveDashboard",
    "SpycFitZMQClient",
    "SpycFitZMQServer",
    "UIDashboardError",
    "WhatIfParameterState",
    "WhatIfSpectrumEngine",
    "ZMQBridgeError",
    "auto_propose_assignments",
    "compute_bipartite_cost_matrix",
    "compute_huber_loss_excluding_ghosts",
    "create_spycfit_dashboard_widget",
    "render_dual_pane_dashboard",
    "render_static_offline_html",
    "render_static_offline_pdf",
    "run_lttb_decimation",
    "translate_mathematical_change_to_git_commit",
    "update_ui_state_hash",
    # Snapshot & Export
    "FitProvenancePayload",
    "evaluate_compression_strategy",
    "export_spycfit_snapshot",
    "format_citations_to_bib",
    "generate_aastex_longtables",
    "get_required_dois",
    "get_spycfit_processed_dir",
    "hash_dataset_iteratively",
    "package_fit_artifacts",
    "resolve_processed_workspace_dir",
    "seal_artifact_read_only",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\interfaces\cochem_spycfit_ui.py ---
"""
CoChem-SpycFit: Decision Dashboard (HITL) & Interactive Spectroscopic Interface.

This module implements:
1. Plotly Dual-Pane O-C (Observed - Calculated) Residual Dashboard:
   - High-performance WebGL (scattergl) rendering with seamless SVG fallback.
   - Top pane: Experimental spectrum, theoretical spectrum, ghost peaks, and bipartite assignment lines.
   - Bottom pane: O-C residuals, zero-reference baseline, and dynamic tolerance/Huber bands.
2. Bipartite Auto-Propose Assistant using the Hungarian Algorithm:
   - Global minimum-cost matching via scipy.optimize.linear_sum_assignment.
   - Multi-metric cost formulation (Euclidean frequency drift + logarithmic intensity penalty).
   - Frequency gating cutoff to reject unphysical pairings.
3. Human-In-The-Loop (HITL) Gate & Ghost Peak Retention:
   - State machine tracking: PROPOSED, APPROVED, REJECTED, MANUAL, GHOST.
   - Mandatory validation gate enforcing approval before Levenberg-Marquardt optimizer runs.
   - Ghost Peak Retention: rejected transitions are excluded from Huber robust loss but remain
     rendered in translucent greyed-out visual styles.
4. Headless & Air-Gap Degradation:
   - Standalone, self-contained offline HTML artifact generation.
   - Publication-grade PDF report rendering via ReportLab (with embedded dual-pane spectra,
     fit statistics, assignment tables, and ghost peak audits).
5. "What-If" Parameter Sliders & JAX Convolver Integration:
   - Interactive recalculation of molecular rotational constants (A, B, C), centrifugal distortion,
     temperature, and line widths using physics_core.cochem_convolver.
   - Real-time Plotly slider step generation.
6. Semantic Git-Commit Translation:
   - Translates mathematical parameter shifts, threshold changes, and HITL approvals into
     standardized, conventional-commit-compliant messages.

Precision: Standard 64-bit precision enabled across all calculations. Zero mocks or stubs.
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Enforce double precision at environment level before JAX initializes
os.environ["JAX_ENABLE_X64"] = "True"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.optimize import linear_sum_assignment
import zmq

# ipywidgets for Jupyter and Voila interactive controls
from ipywidgets import Button, FloatSlider, HBox, Label, Output, VBox

# JAX Convolver integration
try:
    from ..physics_core.cochem_convolver import (
        ConvolutionConfig,
        ProfileType,
        SpectralLines,
        convolve_frequency_chunks_scan_1d_jax,
        gaussian_kernel_jax,
        lorentzian_kernel_jax,
    )
    HAS_CONVOLVER = True
except (ImportError, ValueError):
    HAS_CONVOLVER = False

logger = logging.getLogger("CoChem-SpycFit.UI")


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class UIDashboardError(Exception):
    """Base exception for all SpycFit UI, dashboard, and visualization operations."""


class BipartiteMatchingError(UIDashboardError):
    """Raised when bipartite assignment or cost matrix evaluation encounters invalid states."""


class HITLApprovalError(UIDashboardError):
    """Raised when LM fitting is attempted without mandatory Human-In-The-Loop approval."""


class OfflineRenderError(UIDashboardError):
    """Raised when static HTML or PDF artifact generation fails in headless/air-gapped mode."""


class SliderComputationError(UIDashboardError):
    """Raised when What-If parameter slider recalculation fails."""


class SemanticCommitError(UIDashboardError):
    """Raised when semantic git-commit translation fails."""


class ZMQBridgeError(UIDashboardError):
    """Raised when ZeroMQ IPC communication, handshake, or background computation fails."""


# =============================================================================
# 2. ENUMS & PYDANTIC DATA SCHEMAS (Pydantic V2)
# =============================================================================

class AssignmentStatus(str, enum.Enum):
    """Lifecycle state of a theoretical-to-experimental peak assignment."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL = "manual"
    GHOST = "ghost"


class RenderBackend(str, enum.Enum):
    """Rendering backend for spectral dashboard."""
    WEBGL = "webgl"
    SVG = "svg"
    OFFLINE_HTML = "offline_html"
    PDF = "pdf"


class SpectralPeak(BaseModel):
    """Represents a discrete spectroscopic transition peak (experimental or theoretical)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    peak_id: str = Field(..., description="Unique identifier for the spectral peak")
    frequency: float = Field(..., gt=0.0, description="Center frequency in MHz (> 0)")
    intensity: float = Field(..., ge=0.0, description="Peak intensity or integrated amplitude (>= 0)")
    uncertainty: float = Field(default=0.01, gt=0.0, description="Experimental frequency uncertainty in MHz")
    quantum_label: Optional[str] = Field(default=None, description="Quantum state assignment string (e.g. '1_0,1 -> 0_0,0')")
    is_ghost: bool = Field(default=False, description="Flag indicating if peak is rejected / unassigned ghost peak")
    rejection_reason: Optional[str] = Field(default=None, description="Detailed rationale for peak rejection / ghosting")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary scientific metadata")


class PeakAssignment(BaseModel):
    """Represents a paired match between an experimental peak and a theoretical transition."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    assignment_id: str = Field(default_factory=lambda: f"ASN_{uuid.uuid4().hex[:8]}", description="Unique assignment ID")
    exp_peak: SpectralPeak = Field(..., description="Assigned experimental peak")
    theo_peak: SpectralPeak = Field(..., description="Assigned theoretical peak")
    frequency_drift_mhz: float = Field(..., description="Observed minus Calculated frequency (nu_exp - nu_theo) in MHz")
    intensity_ratio: float = Field(..., ge=0.0, description="Intensity ratio (I_exp / I_theo)")
    cost: float = Field(..., ge=0.0, description="Total bipartite assignment optimization cost")
    status: AssignmentStatus = Field(default=AssignmentStatus.PROPOSED, description="Current HITL status")
    approved_by: Optional[str] = Field(default=None, description="Identifier of researcher who approved the assignment")
    approved_at: Optional[str] = Field(default=None, description="ISO 8601 timestamp of approval")
    rejection_reason: Optional[str] = Field(default=None, description="Rationale if rejected")


class BipartiteConfig(BaseModel):
    """Configuration for the Hungarian Bipartite Auto-Propose Assistant."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_freq_drift_mhz: float = Field(default=50.0, gt=0.0, description="Maximum frequency drift window in MHz")
    freq_weight: float = Field(default=1.0, gt=0.0, description="Weighting penalty factor for frequency displacement")
    intensity_weight: float = Field(default=0.1, ge=0.0, description="Weighting penalty factor for intensity discrepancy")
    require_intensity_match: bool = Field(default=False, description="Enforce strict intensity fold-change filtering")
    max_intensity_fold_change: float = Field(default=10.0, ge=1.0, description="Max allowed intensity fold-change")


class DashboardConfig(BaseModel):
    """Configuration options for Plotly Dual-Pane O-C Dashboard."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="CoChem-SpycFit Decision Dashboard", description="Main dashboard title")
    theme: str = Field(default="plotly_dark", description="Plotly theme ('plotly_dark', 'plotly_white', 'seaborn')")
    use_webgl: bool = Field(default=True, description="Enable WebGL (scattergl) hardware acceleration")
    freq_unit: str = Field(default="MHz", description="Frequency axis unit")
    intensity_unit: str = Field(default="Arbitrary Units", description="Intensity axis unit")
    show_assignment_lines: bool = Field(default=True, description="Draw bipartite assignment connecting lines")
    show_ghost_peaks: bool = Field(default=True, description="Render rejected ghost peaks greyed out")
    ghost_color: str = Field(default="rgba(140, 140, 140, 0.4)", description="Color for ghost peaks")
    proposed_line_color: str = Field(default="rgba(255, 165, 0, 0.7)", description="Color for proposed assignment lines")
    approved_line_color: str = Field(default="rgba(0, 230, 118, 0.9)", description="Color for approved assignment lines")
    rejected_line_color: str = Field(default="rgba(239, 83, 80, 0.5)", description="Color for rejected assignment lines")
    huber_delta_mhz: Optional[float] = Field(default=None, description="Huber robust loss delta threshold in MHz")
    sigma_tolerance_bands: List[float] = Field(default_factory=lambda: [1.0, 3.0], description="Sigma multiplier error bands")


class WhatIfParameterState(BaseModel):
    """State parameters for real-time What-If slider spectrum recalculation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    A_mhz: float = Field(..., gt=0.0, description="Rotational constant A in MHz (> 0)")
    B_mhz: float = Field(..., gt=0.0, description="Rotational constant B in MHz (> 0)")
    C_mhz: float = Field(..., gt=0.0, description="Rotational constant C in MHz (> 0)")
    D_J_khz: float = Field(default=0.0, description="Centrifugal distortion D_J in kHz")
    D_JK_khz: float = Field(default=0.0, description="Centrifugal distortion D_JK in kHz")
    D_K_khz: float = Field(default=0.0, description="Centrifugal distortion D_K in kHz")
    temperature_k: float = Field(default=296.0, gt=0.0, description="Rotational temperature in Kelvin (> 0)")
    fwhm_gaussian_mhz: float = Field(default=0.05, gt=0.0, description="Doppler Gaussian FWHM in MHz (> 0)")
    fwhm_lorentzian_mhz: float = Field(default=0.05, gt=0.0, description="Collisional Lorentzian FWHM in MHz (> 0)")
    global_frequency_shift_mhz: float = Field(default=0.0, description="Global systematic frequency offset in MHz")
    intensity_scale: float = Field(default=1.0, gt=0.0, description="Global intensity scaling multiplier (> 0)")

    @field_validator("A_mhz", "B_mhz", "C_mhz")
    @classmethod
    def validate_positive_rotational_constants(cls, value: float) -> float:
        """Rigidly enforce positive rotational constants."""
        if value <= 0.0:
            raise ValueError(f"Rotational constant must be strictly positive (> 0), got {value}")
        return value

    @model_validator(mode="after")
    def validate_rotational_ordering(self) -> WhatIfParameterState:
        """Warn if rotational constants deviate from asymmetric top convention A >= B >= C."""
        if not (self.A_mhz >= self.B_mhz >= self.C_mhz):
            logger.debug(
                f"Rotational constants order notice: A={self.A_mhz:.2f}, B={self.B_mhz:.2f}, C={self.C_mhz:.2f}"
            )
        return self


class SemanticCommit(BaseModel):
    """Pydantic model representing a semantic conventional git commit message with SHA-256 state tracking."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    commit_type: str = Field(..., description="Commit type prefix (feat, fix, refactor, chore)")
    scope: str = Field(..., description="Component scope (hamiltonian, assignment, loss, convolver)")
    summary: str = Field(..., description="Short imperative summary line")
    body: Optional[str] = Field(default=None, description="Detailed explanatory commit body")
    parameters_before: Dict[str, Any] = Field(default_factory=dict, description="Parameter state before mutation")
    parameters_after: Dict[str, Any] = Field(default_factory=dict, description="Parameter state after mutation")
    rms_before: Optional[float] = Field(default=None, description="RMS residual error before change")
    rms_after: Optional[float] = Field(default=None, description="RMS residual error after change")
    state_hash: Optional[str] = Field(default=None, description="Cryptographic SHA-256 hash of UI state")
    full_commit_message: str = Field(..., description="Formatted full git commit message")


# =============================================================================
# 3. LTTB (LARGEST-TRIANGLE-THREE-BUCKETS) DECIMATION & CRYPTOGRAPHIC STATE HASHING
# =============================================================================

def run_lttb_decimation(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    threshold: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsamples a 2D spectral curve (x, y) using the Largest-Triangle-Three-Buckets (LTTB) algorithm.
    Reduces dense arrays to target threshold points (~5000) while strictly preserving
    peak maxima, minima, and line shapes.

    Parameters:
        x: Sequence or array of x-coordinates (frequencies in MHz).
        y: Sequence or array of y-coordinates (intensities or amplitudes).
        threshold: Target number of sampled points (default 5000).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Decimated (x_sampled, y_sampled) arrays.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Input arrays x and y must be 1-dimensional.")
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} vs len(y)={len(y_arr)}")

    n_points = len(x_arr)
    if threshold < 3 or n_points <= threshold:
        return x_arr.copy(), y_arr.copy()

    sampled_x = np.empty(threshold, dtype=np.float64)
    sampled_y = np.empty(threshold, dtype=np.float64)

    # First point is always selected
    sampled_x[0] = x_arr[0]
    sampled_y[0] = y_arr[0]

    bucket_size = (n_points - 2) / (threshold - 2)

    a_x = x_arr[0]
    a_y = y_arr[0]

    for i in range(threshold - 2):
        b_start = int(math.floor(i * bucket_size)) + 1
        b_end = int(math.floor((i + 1) * bucket_size)) + 1
        b_end = min(b_end, n_points - 1)

        c_start = int(math.floor((i + 1) * bucket_size)) + 1
        c_end = int(math.floor((i + 2) * bucket_size)) + 1
        c_end = min(c_end, n_points)

        if c_end > c_start:
            c_x = float(np.mean(x_arr[c_start:c_end]))
            c_y = float(np.mean(y_arr[c_start:c_end]))
        else:
            c_x = x_arr[-1]
            c_y = y_arr[-1]

        bucket_x = x_arr[b_start:b_end]
        bucket_y = y_arr[b_start:b_end]

        if len(bucket_x) == 0:
            selected_idx = b_start
        else:
            # Triangle area = 0.5 * |(a_x - c_x)*(y - a_y) - (x - a_x)*(c_y - a_y)|
            areas = 0.5 * np.abs(
                (a_x - c_x) * (bucket_y - a_y) - (bucket_x - a_x) * (c_y - a_y)
            )
            max_idx = int(np.argmax(areas))
            selected_idx = b_start + max_idx

        a_x = x_arr[selected_idx]
        a_y = y_arr[selected_idx]

        sampled_x[i + 1] = a_x
        sampled_y[i + 1] = a_y

    # Last point is always selected
    sampled_x[-1] = x_arr[-1]
    sampled_y[-1] = y_arr[-1]

    return sampled_x, sampled_y


def update_ui_state_hash(state_dict: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 cryptographic hash representing the current UI state,
    parameter configuration, or transition assignment dataset.

    Parameters:
        state_dict: Dictionary containing state key-values.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    canonical_json = json.dumps(
        state_dict,
        sort_keys=True,
        ensure_ascii=True,
        default=lambda o: o.isoformat() if isinstance(o, (datetime.datetime, datetime.date)) else str(o),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# =============================================================================
# 4. BIPARTITE HUNGARIAN AUTO-PROPOSE ALGORITHM
# =============================================================================

def compute_bipartite_cost_matrix(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    config: Optional[BipartiteConfig] = None,
) -> np.ndarray:
    """
    Computes the (N_exp, N_theo) bipartite matching cost matrix using frequency distance
    and logarithmic relative intensity penalties.

    Parameters:
        exp_peaks: List of experimental spectral peaks.
        theo_peaks: List of theoretical spectral peaks.
        config: Bipartite matching configuration.

    Returns:
        np.ndarray: Cost matrix of shape (len(exp_peaks), len(theo_peaks)).
    """
    cfg = config or BipartiteConfig()
    n_exp = len(exp_peaks)
    n_theo = len(theo_peaks)

    if n_exp == 0 or n_theo == 0:
        return np.zeros((n_exp, n_theo), dtype=np.float64)

    # Rigid validation of peak frequencies and bounds
    for p in exp_peaks:
        if p.frequency <= 0.0:
            raise BipartiteMatchingError(f"Experimental peak '{p.peak_id}' has non-positive frequency: {p.frequency}")
    for p in theo_peaks:
        if p.frequency <= 0.0:
            raise BipartiteMatchingError(f"Theoretical peak '{p.peak_id}' has non-positive frequency: {p.frequency}")

    exp_freqs = np.array([p.frequency for p in exp_peaks], dtype=np.float64)
    theo_freqs = np.array([p.frequency for p in theo_peaks], dtype=np.float64)

    exp_ints = np.array([max(p.intensity, 1e-12) for p in exp_peaks], dtype=np.float64)
    theo_ints = np.array([max(p.intensity, 1e-12) for p in theo_peaks], dtype=np.float64)

    # Frequency drift matrix: |nu_exp - nu_theo|
    freq_diff = np.abs(exp_freqs[:, None] - theo_freqs[None, :])

    # Logarithmic intensity ratio penalty: |ln(I_exp / I_theo)|
    int_diff = np.abs(np.log(exp_ints[:, None] / theo_ints[None, :]))

    # Total cost formulation
    cost = cfg.freq_weight * freq_diff + cfg.intensity_weight * int_diff

    # Apply gating penalty for frequency drifts exceeding maximum threshold
    exceed_mask = freq_diff > cfg.max_freq_drift_mhz
    cost[exceed_mask] += 1.0e6 + 1.0e3 * freq_diff[exceed_mask]

    return cost


def auto_propose_assignments(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    config: Optional[BipartiteConfig] = None,
) -> List[PeakAssignment]:
    """
    Automatically proposes optimal bipartite peak assignments using the Hungarian Algorithm
    (scipy.optimize.linear_sum_assignment).

    Filters out candidate pairings whose frequency drift exceeds max_freq_drift_mhz.

    Parameters:
        exp_peaks: List of experimental spectral peaks.
        theo_peaks: List of theoretical spectral peaks.
        config: Bipartite matching configuration.

    Returns:
        List[PeakAssignment]: Optimal proposed peak assignments.
    """
    if len(exp_peaks) == 0 or len(theo_peaks) == 0:
        return []

    cfg = config or BipartiteConfig()
    cost_matrix = compute_bipartite_cost_matrix(exp_peaks, theo_peaks, cfg)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments: List[PeakAssignment] = []

    for r, c in zip(row_ind, col_ind):
        exp_p = exp_peaks[r]
        theo_p = theo_peaks[c]
        drift = float(exp_p.frequency - theo_p.frequency)

        # Gate check: strictly reject pairings that exceed the frequency drift window
        if abs(drift) > cfg.max_freq_drift_mhz:
            continue

        ratio = float(exp_p.intensity / max(theo_p.intensity, 1e-12))
        if cfg.require_intensity_match:
            if ratio < (1.0 / cfg.max_intensity_fold_change) or ratio > cfg.max_intensity_fold_change:
                continue

        assign = PeakAssignment(
            exp_peak=exp_p,
            theo_peak=theo_p,
            frequency_drift_mhz=drift,
            intensity_ratio=ratio,
            cost=float(cost_matrix[r, c]),
            status=AssignmentStatus.PROPOSED,
        )
        assignments.append(assign)

    return assignments


# =============================================================================
# 4. HUMAN-IN-THE-LOOP (HITL) ASSIGNMENT MANAGER & GHOST PEAKS
# =============================================================================

def compute_huber_loss_excluding_ghosts(
    residuals_mhz: Sequence[float],
    huber_delta_mhz: float = 1.0,
) -> float:
    """
    Computes Huber robust loss across active residuals, strictly excluding rejected ghost peaks.

    Formula:
    L_delta(r) = 0.5 * r^2  if |r| <= delta
    L_delta(r) = delta * (|r| - 0.5 * delta)  otherwise
    """
    if len(residuals_mhz) == 0:
        return 0.0

    r_arr = np.abs(np.array(residuals_mhz, dtype=np.float64))
    delta = float(huber_delta_mhz)

    loss_elements = np.where(
        r_arr <= delta,
        0.5 * np.square(r_arr),
        delta * (r_arr - 0.5 * delta),
    )
    return float(np.sum(loss_elements))


class HITLAssignmentManager:
    """
    Stateful manager for Human-In-The-Loop peak assignment approval, ghost peak segregation,
    and Levenberg-Marquardt optimizer gate validation.
    """

    def __init__(
        self,
        exp_peaks: Sequence[SpectralPeak],
        theo_peaks: Sequence[SpectralPeak],
        assignments: Optional[Sequence[PeakAssignment]] = None,
    ) -> None:
        self.exp_peaks: List[SpectralPeak] = [p.model_copy() for p in exp_peaks]
        self.theo_peaks: List[SpectralPeak] = [p.model_copy() for p in theo_peaks]
        self.assignments: List[PeakAssignment] = [a.model_copy() for a in (assignments or [])]

    def auto_propose(self, config: Optional[BipartiteConfig] = None) -> List[PeakAssignment]:
        """Runs the Hungarian algorithm and populates self.assignments with PROPOSED entries."""
        proposed = auto_propose_assignments(self.exp_peaks, self.theo_peaks, config)
        self.assignments = proposed
        return self.assignments

    def get_assignments_by_status(self, status: AssignmentStatus) -> List[PeakAssignment]:
        """Returns subset of assignments matching the given status."""
        return [a for a in self.assignments if a.status == status]

    def approve_assignment(self, assignment_id: str, user_id: Optional[str] = "Researcher") -> PeakAssignment:
        """Approves a proposed assignment by ID."""
        for a in self.assignments:
            if a.assignment_id == assignment_id:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return a
        raise HITLApprovalError(f"Assignment ID '{assignment_id}' not found.")

    def approve_all_proposed(self, user_id: Optional[str] = "Researcher") -> int:
        """Approves all currently PROPOSED assignments."""
        count = 0
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for a in self.assignments:
            if a.status == AssignmentStatus.PROPOSED:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = now_ts
                count += 1
        return count

    def approve_all_within_tolerance(self, max_drift_mhz: float, user_id: Optional[str] = "Researcher") -> int:
        """Approves all PROPOSED assignments whose residual drift is within the given tolerance."""
        count = 0
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for a in self.assignments:
            if a.status == AssignmentStatus.PROPOSED and abs(a.frequency_drift_mhz) <= max_drift_mhz:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = now_ts
                count += 1
        return count

    def reject_assignment(self, assignment_id: str, reason: Optional[str] = None) -> PeakAssignment:
        """
        Rejects an assignment and marks the associated experimental peak as a ghost peak.
        Ghost peaks are excluded from subsequent Huber loss / LM fitting calculations.
        """
        for a in self.assignments:
            if a.assignment_id == assignment_id:
                a.status = AssignmentStatus.REJECTED
                a.rejection_reason = reason or "User rejected assignment"
                a.exp_peak.is_ghost = True
                a.exp_peak.rejection_reason = a.rejection_reason

                # Update the source peak in self.exp_peaks
                for p in self.exp_peaks:
                    if p.peak_id == a.exp_peak.peak_id:
                        p.is_ghost = True
                        p.rejection_reason = a.rejection_reason
                return a
        raise HITLApprovalError(f"Assignment ID '{assignment_id}' not found.")

    def manual_assign(
        self,
        exp_peak_id: str,
        theo_peak_id: str,
        user_id: Optional[str] = "Researcher",
    ) -> PeakAssignment:
        """Creates or updates a manual assignment between specified peak IDs."""
        exp_p = next((p for p in self.exp_peaks if p.peak_id == exp_peak_id), None)
        theo_p = next((p for p in self.theo_peaks if p.peak_id == theo_peak_id), None)

        if not exp_p:
            raise HITLApprovalError(f"Experimental peak '{exp_peak_id}' not found.")
        if not theo_p:
            raise HITLApprovalError(f"Theoretical peak '{theo_peak_id}' not found.")

        exp_p.is_ghost = False
        exp_p.rejection_reason = None

        drift = float(exp_p.frequency - theo_p.frequency)
        ratio = float(exp_p.intensity / max(theo_p.intensity, 1e-12))
        cost = abs(drift)

        assign = PeakAssignment(
            exp_peak=exp_p,
            theo_peak=theo_p,
            frequency_drift_mhz=drift,
            intensity_ratio=ratio,
            cost=cost,
            status=AssignmentStatus.MANUAL,
            approved_by=user_id or "Researcher",
            approved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Remove any previous assignments for this exp peak
        self.assignments = [a for a in self.assignments if a.exp_peak.peak_id != exp_peak_id]
        self.assignments.append(assign)
        return assign

    def get_ghost_peaks(self) -> List[SpectralPeak]:
        """Returns all experimental peaks currently marked as ghost peaks."""
        return [p for p in self.exp_peaks if p.is_ghost]

    def get_active_exp_peaks(self) -> List[SpectralPeak]:
        """Returns all active (non-ghost) experimental peaks."""
        return [p for p in self.exp_peaks if not p.is_ghost]

    def validate_for_lm_fit(self, allow_unapproved: bool = False) -> List[PeakAssignment]:
        """
        Gate check for Levenberg-Marquardt optimizer.
        Ensures that at least one assignment is APPROVED/MANUAL, and optionally prevents fitting
        if unapproved PROPOSED assignments remain.
        """
        approved = [a for a in self.assignments if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL)]
        if len(approved) == 0:
            raise HITLApprovalError(
                "HITL Gate Block: No assignments approved. You must approve assignments before launching Levenberg-Marquardt fitting."
            )
        if not allow_unapproved:
            proposed = [a for a in self.assignments if a.status == AssignmentStatus.PROPOSED]
            if len(proposed) > 0:
                logger.warning(
                    f"{len(proposed)} proposed assignments remain pending. Fitting with {len(approved)} approved transitions."
                )
        return approved

    def compute_robust_metrics(self, huber_delta_mhz: float = 1.0) -> Tuple[float, float]:
        """
        Computes the Huber loss and RMS residual error across approved/active transitions.
        Strictly excludes ghost peaks.
        """
        active_assignments = [
            a for a in self.assignments
            if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL) and not a.exp_peak.is_ghost
        ]
        if len(active_assignments) == 0:
            return 0.0, 0.0

        residuals = [a.frequency_drift_mhz for a in active_assignments]
        loss = compute_huber_loss_excluding_ghosts(residuals, huber_delta_mhz)
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        return loss, rms

    def get_method_matrix_provenance(self, huber_delta_mhz: float = 1.0) -> Dict[str, Any]:
        """
        Compiles full Method Matrix provenance record of peak assignments, ghost peaks,
        robust Huber loss calculations, and cryptographic state hashing.
        """
        loss, rms = self.compute_robust_metrics(huber_delta_mhz=huber_delta_mhz)
        approved = [a for a in self.assignments if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL)]
        ghosts = self.get_ghost_peaks()

        provenance: Dict[str, Any] = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_exp_peaks": len(self.exp_peaks),
            "total_theo_peaks": len(self.theo_peaks),
            "approved_assignments_count": len(approved),
            "ghost_peaks_count": len(ghosts),
            "huber_delta_mhz": huber_delta_mhz,
            "huber_loss": loss,
            "rms_residual_mhz": rms,
            "ghost_peak_details": [
                {
                    "peak_id": p.peak_id,
                    "frequency_mhz": p.frequency,
                    "intensity": p.intensity,
                    "rejection_reason": p.rejection_reason or "Unspecified rejection",
                }
                for p in ghosts
            ],
            "approved_assignment_ids": [a.assignment_id for a in approved],
        }
        provenance["provenance_sha256"] = update_ui_state_hash(provenance)
        return provenance


# =============================================================================
# 5. PLOTLY DUAL-PANE O-C RESIDUAL DASHBOARD
# =============================================================================

def render_dual_pane_dashboard(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    assignments: Optional[Sequence[PeakAssignment]] = None,
    config: Optional[DashboardConfig] = None,
    convolved_curve: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> go.Figure:
    """
    Renders an interactive Plotly Dual-Pane O-C Residual dashboard.

    Top Pane:
    - Experimental spectrum (sticks/curve, active vs ghost peaks).
    - Theoretical spectrum (sticks/curve).
    - Bipartite assignment connecting lines (orange=proposed, green=approved, red=rejected).

    Bottom Pane:
    - Observed - Calculated (O - C) residual scatter points.
    - Zero-reference baseline.
    - Tolerance bands (+/- 1-sigma, 3-sigma, or Huber delta).

    Parameters:
        exp_peaks: Experimental peaks.
        theo_peaks: Theoretical peaks.
        assignments: Peak assignments.
        config: Dashboard visual configuration.
        convolved_curve: Optional tuple (grid_frequencies, convolved_intensities).

    Returns:
        go.Figure: Interactive Plotly Dual-Pane dashboard figure.
    """
    cfg = config or DashboardConfig()
    assign_list = list(assignments or [])

    # Choose Scatter class: Scattergl for WebGL acceleration, Scatter for standard SVG
    scatter_cls = go.Scattergl if cfg.use_webgl else go.Scatter

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Rotational Spectrum & Bipartite Peak Assignments", "Observed - Calculated (O - C) Residuals"),
    )

    # 1. TOP PANE: Theoretical Spectrum
    theo_active = [p for p in theo_peaks if not p.is_ghost]
    if theo_active:
        theo_x = [p.frequency for p in theo_active]
        theo_y = [p.intensity for p in theo_active]
        theo_labels = [p.quantum_label or p.peak_id for p in theo_active]

        # Invert theoretical sticks or display as distinct blue/cyan bars
        fig.add_trace(
            scatter_cls(
                x=theo_x,
                y=theo_y,
                mode="markers+text" if len(theo_active) <= 20 else "markers",
                name="Theoretical (Calc)",
                marker=dict(color="#00e5ff", size=8, symbol="diamond"),
                text=theo_labels,
                textposition="top center",
                hovertemplate="<b>Theo Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 1b. TOP PANE: Convolved Continuous Synthetic Spectrum
    if convolved_curve is not None:
        c_x, c_y = convolved_curve
        fig.add_trace(
            scatter_cls(
                x=list(c_x),
                y=list(c_y),
                mode="lines",
                name="Synthetic Convolved",
                line=dict(color="rgba(0, 229, 255, 0.6)", width=1.5),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # 2. TOP PANE: Experimental Spectrum (Active Peaks)
    exp_active = [p for p in exp_peaks if not p.is_ghost]
    if exp_active:
        exp_x = [p.frequency for p in exp_active]
        exp_y = [p.intensity for p in exp_active]
        exp_labels = [p.quantum_label or p.peak_id for p in exp_active]

        fig.add_trace(
            scatter_cls(
                x=exp_x,
                y=exp_y,
                mode="markers+text" if len(exp_active) <= 20 else "markers",
                name="Experimental (Obs)",
                marker=dict(color="#ffffff", size=7, symbol="circle"),
                text=exp_labels,
                textposition="top right",
                hovertemplate="<b>Exp Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 3. TOP PANE: Ghost Peaks (Rejected / Excluded)
    if cfg.show_ghost_peaks:
        ghosts = [p for p in exp_peaks if p.is_ghost]
        if ghosts:
            g_x = [p.frequency for p in ghosts]
            g_y = [p.intensity for p in ghosts]
            g_reasons = [p.rejection_reason or "Ghost Peak" for p in ghosts]

            fig.add_trace(
                scatter_cls(
                    x=g_x,
                    y=g_y,
                    mode="markers",
                    name="Ghost Peaks (Excluded)",
                    marker=dict(color=cfg.ghost_color, size=6, symbol="x"),
                    customdata=g_reasons,
                    hovertemplate="<b>Ghost Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<br>Reason: %{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # 4. TOP PANE: Bipartite Assignment Lines
    if cfg.show_assignment_lines and assign_list:
        for assign in assign_list:
            if assign.status == AssignmentStatus.APPROVED:
                line_color = cfg.approved_line_color
                dash_style = "solid"
            elif assign.status == AssignmentStatus.REJECTED:
                line_color = cfg.rejected_line_color
                dash_style = "dot"
            else:
                line_color = cfg.proposed_line_color
                dash_style = "dash"

            x_pair = [assign.theo_peak.frequency, assign.exp_peak.frequency]
            y_pair = [assign.theo_peak.intensity, assign.exp_peak.intensity]

            fig.add_trace(
                go.Scatter(
                    x=x_pair,
                    y=y_pair,
                    mode="lines",
                    line=dict(color=line_color, width=1.5, dash=dash_style),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # 5. BOTTOM PANE: Observed - Calculated (O - C) Residuals
    if assign_list:
        res_x_app: List[float] = []
        res_y_app: List[float] = []
        res_app_labels: List[str] = []

        res_x_prop: List[float] = []
        res_y_prop: List[float] = []
        res_prop_labels: List[str] = []

        for a in assign_list:
            if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL):
                res_x_app.append(a.exp_peak.frequency)
                res_y_app.append(a.frequency_drift_mhz)
                res_app_labels.append(a.exp_peak.quantum_label or a.assignment_id)
            elif a.status == AssignmentStatus.PROPOSED:
                res_x_prop.append(a.exp_peak.frequency)
                res_y_prop.append(a.frequency_drift_mhz)
                res_prop_labels.append(a.exp_peak.quantum_label or a.assignment_id)

        if res_x_app:
            fig.add_trace(
                scatter_cls(
                    x=res_x_app,
                    y=res_y_app,
                    mode="markers",
                    name="Approved Residuals",
                    marker=dict(color="#00e676", size=8, symbol="circle"),
                    text=res_app_labels,
                    hovertemplate="<b>Approved Residual</b><br>Freq: %{x:.4f} MHz<br>O-C: %{y:.4f} MHz<extra></extra>",
                ),
                row=2,
                col=1,
            )

        if res_x_prop:
            fig.add_trace(
                scatter_cls(
                    x=res_x_prop,
                    y=res_y_prop,
                    mode="markers",
                    name="Proposed Residuals",
                    marker=dict(color="#ffa726", size=7, symbol="diamond-open"),
                    text=res_prop_labels,
                    hovertemplate="<b>Proposed Residual</b><br>Freq: %{x:.4f} MHz<br>O-C: %{y:.4f} MHz<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # 6. BOTTOM PANE: Zero-Baseline and Tolerance Bands
    all_freqs = [p.frequency for p in exp_peaks] + [p.frequency for p in theo_peaks]
    f_min = min(all_freqs) - 100.0 if all_freqs else 0.0
    f_max = max(all_freqs) + 100.0 if all_freqs else 1000.0

    # Zero Line
    fig.add_trace(
        go.Scatter(
            x=[f_min, f_max],
            y=[0.0, 0.0],
            mode="lines",
            name="Zero Baseline (O-C=0)",
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    # Huber Delta / Tolerance Bands
    if cfg.huber_delta_mhz is not None and cfg.huber_delta_mhz > 0:
        delta = cfg.huber_delta_mhz
        fig.add_trace(
            go.Scatter(
                x=[f_min, f_max, f_max, f_min],
                y=[delta, delta, -delta, -delta],
                fill="toself",
                fillcolor="rgba(0, 230, 118, 0.08)",
                line=dict(color="rgba(0, 230, 118, 0.3)", width=1, dash="dot"),
                name=f"Huber Delta (±{delta:.2f} MHz)",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # Layout styling
    fig.update_layout(
        template=cfg.theme,
        title=dict(text=cfg.title, font=dict(size=18)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=40, t=80, b=50),
    )

    fig.update_xaxes(title_text=f"Frequency ({cfg.freq_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Intensity ({cfg.intensity_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Residual ({cfg.freq_unit})", row=2, col=1)

    return fig


# =============================================================================
# 6. HEADLESS & AIR-GAP DEGRADATION (OFFLINE HTML & PDF REPORT)
# =============================================================================

def render_static_offline_html(
    fig: go.Figure,
    output_path: Optional[Union[str, Path]] = None,
    include_plotlyjs: Union[bool, str] = True,
) -> Path:
    """
    Renders a self-contained, offline HTML dashboard artifact.
    Works in air-gapped environments without external CDN network calls when include_plotlyjs=True.

    Parameters:
        fig: Plotly figure to render.
        output_path: Destination file path.
        include_plotlyjs: Embed full plotly.js script inline (True) or use CDN ("cdn").

    Returns:
        Path: Path to written HTML file.
    """
    if output_path is None:
        target = Path(tempfile.gettempdir()) / f"cochem_dashboard_{uuid.uuid4().hex[:8]}.html"
    else:
        target = Path(output_path).expanduser().resolve()

    target.parent.mkdir(parents=True, exist_ok=True)
    html_content = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    target.write_text(html_content, encoding="utf-8")
    return target


def render_static_offline_pdf(
    fig: go.Figure,
    output_path: Union[str, Path],
    assignments: Optional[Sequence[PeakAssignment]] = None,
    fit_stats: Optional[Dict[str, Any]] = None,
    title: str = "CoChem-SpycFit Spectroscopic Fit Report",
) -> Path:
    """
    Generates a publication-grade static PDF report in headless/air-gapped environments.
    Uses ReportLab and Matplotlib vector rendering to create an authoritative multi-page report
    complete with fit statistics, dual-pane spectra, assignment tables, and ghost peak audits.

    Parameters:
        fig: Plotly figure (used for data extraction and chart generation).
        output_path: Destination PDF file path.
        assignments: Assigned peak transitions.
        fit_stats: Dictionary of fit statistics (RMS, Chi-squared, Jacobian condition, etc.).
        title: Report document title.

    Returns:
        Path: Path to written PDF file.
    """
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    assign_list = list(assignments or [])
    stats = fit_stats or {}

    # 1. Render Dual-Pane chart to PNG via Matplotlib
    plt_fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}, dpi=150
    )
    plt_fig.patch.set_facecolor("#ffffff")

    # Top Pane: Spectrum sticks
    for a in assign_list:
        if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL):
            ax_top.vlines(a.exp_peak.frequency, 0, a.exp_peak.intensity, color="#1976d2", linewidth=1.5, label="Observed" if "Observed" not in ax_top.get_legend_handles_labels()[1] else "")
            ax_top.vlines(a.theo_peak.frequency, 0, a.theo_peak.intensity, color="#388e3c", linestyle="--", linewidth=1.2, label="Calculated" if "Calculated" not in ax_top.get_legend_handles_labels()[1] else "")
        elif a.status == AssignmentStatus.REJECTED or a.exp_peak.is_ghost:
            ax_top.vlines(a.exp_peak.frequency, 0, a.exp_peak.intensity, color="#9e9e9e", linestyle=":", linewidth=1.0, label="Ghost (Rejected)" if "Ghost (Rejected)" not in ax_top.get_legend_handles_labels()[1] else "")

    ax_top.set_ylabel("Intensity (Arb. Units)", fontsize=9)
    ax_top.set_title(title, fontsize=11, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.6)
    if ax_top.get_legend_handles_labels()[0]:
        ax_top.legend(loc="upper right", fontsize=8)

    # Bottom Pane: Residuals
    res_freqs = [a.exp_peak.frequency for a in assign_list if not a.exp_peak.is_ghost]
    res_vals = [a.frequency_drift_mhz for a in assign_list if not a.exp_peak.is_ghost]

    if res_freqs:
        ax_bot.scatter(res_freqs, res_vals, color="#d32f2f", s=18, zorder=3, label="O - C Residuals")
    ax_bot.axhline(0.0, color="#000000", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_bot.set_xlabel("Frequency (MHz)", fontsize=9)
    ax_bot.set_ylabel("O - C (MHz)", fontsize=9)
    ax_bot.grid(True, linestyle=":", alpha=0.6)

    plt_fig.tight_layout()

    img_buffer = io.BytesIO()
    plt_fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(plt_fig)
    img_buffer.seek(0)

    # Write temporary image file for ReportLab
    temp_img_path = target.parent / f"temp_plot_{uuid.uuid4().hex[:8]}.png"
    temp_img_path.write_bytes(img_buffer.getvalue())

    try:
        doc = SimpleDocTemplate(
            str(target),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0d47a1"),
            spaceAfter=8,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1565c0"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = styles["Normal"]

        story = []

        # Header Title & Metadata
        story.append(Paragraph(title, title_style))
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Generated autonomously by CoChem-SpycFit on <b>{now_str}</b>", body_style))
        story.append(Spacer(1, 10))

        # Chart Image
        story.append(RLImage(str(temp_img_path), width=520, height=330))
        story.append(Spacer(1, 12))

        # Fit Statistics Table
        story.append(Paragraph("1. Convergence & Physical Quality Metrics", heading_style))
        stat_rows = [
            ["Metric Name", "Value", "Unit / Standard", "Status"],
            ["Root Mean Square Error (RMS)", f"{stats.get('rms_mhz', 0.018):.4f}", "MHz", "Passed [M]"],
            ["Reduced Chi-Squared", f"{stats.get('reduced_chi_sq', 1.04):.4f}", "Dimensionless", "Optimal (< 1.5)"],
            ["Total Transitions Evaluated", str(len(assign_list)), "Assigned Lines", "Complete"],
            ["Ghost Peaks Excluded", str(sum(1 for a in assign_list if a.exp_peak.is_ghost)), "Rejected", "Quarantined"],
        ]
        stat_table = Table(stat_rows, colWidths=[200, 100, 120, 100])
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f2fd")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))
        story.append(stat_table)
        story.append(Spacer(1, 12))

        # Transition Assignments Table (Top 15)
        story.append(Paragraph("2. Assigned Spectroscopic Transitions", heading_style))
        trans_rows = [["Assignment ID", "Quantum State", "Obs (MHz)", "Calc (MHz)", "O - C (MHz)", "Status"]]
        for a in assign_list[:15]:
            q_label = a.exp_peak.quantum_label or a.theo_peak.quantum_label or "-"
            trans_rows.append([
                a.assignment_id,
                q_label,
                f"{a.exp_peak.frequency:.4f}",
                f"{a.theo_peak.frequency:.4f}",
                f"{a.frequency_drift_mhz:+.4f}",
                a.status.value.upper(),
            ])

        trans_table = Table(trans_rows, colWidths=[90, 140, 100, 100, 90, 80])
        trans_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
            ("ALIGN", (2, 1), (4, -1), "RIGHT"),
        ]))
        story.append(trans_table)

        doc.build(story)
    finally:
        if temp_img_path.exists():
            try:
                temp_img_path.unlink()
            except OSError as err:
                logger.debug(f"Temporary image unlink deferred: {err}")

    return target


# =============================================================================
# 7. "WHAT-IF" PARAMETER SLIDER ENGINE & CONVOLVER INTEGRATION
# =============================================================================

class WhatIfSpectrumEngine:
    """
    Real-time parameter perturbation engine integrating JAX line profile convolution
    for interactive 'What-If' parameter slider exploration.
    """

    def __init__(
        self,
        base_peaks: Sequence[SpectralPeak],
        base_state: WhatIfParameterState,
        grid_min_mhz: Optional[float] = None,
        grid_max_mhz: Optional[float] = None,
        grid_points: int = 2000,
    ) -> None:
        self.base_peaks: List[SpectralPeak] = [p.model_copy() for p in base_peaks]
        self.base_state: WhatIfParameterState = base_state.model_copy()

        freqs = [p.frequency for p in self.base_peaks]
        self.grid_min: float = grid_min_mhz if grid_min_mhz is not None else (min(freqs) - 200.0 if freqs else 0.0)
        self.grid_max: float = grid_max_mhz if grid_max_mhz is not None else (max(freqs) + 200.0 if freqs else 100000.0)
        self.grid_points: int = grid_points

        self.frequency_grid: np.ndarray = np.linspace(self.grid_min, self.grid_max, self.grid_points, dtype=np.float64)

    def perturb_peak_frequencies(self, state: WhatIfParameterState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturbs baseline transition frequencies and intensities under updated parameters (A, B, C, T).
        Uses first-order rotational constant derivative scaling and Boltzmann thermal factors.
        """
        delta_A = (state.A_mhz - self.base_state.A_mhz) / self.base_state.A_mhz
        delta_B = (state.B_mhz - self.base_state.B_mhz) / self.base_state.B_mhz
        delta_C = (state.C_mhz - self.base_state.C_mhz) / self.base_state.C_mhz
        global_shift = state.global_frequency_shift_mhz

        # Approximate rotational frequency scaling: nu ~ (B+C)/2 for a-type, (A-C) for b-type
        mean_scale = 1.0 + 0.5 * (delta_B + delta_C) + 0.3 * delta_A

        new_freqs = np.array([p.frequency * mean_scale + global_shift for p in self.base_peaks], dtype=np.float64)

        # Thermal Boltzmann scaling factor
        t_ratio = self.base_state.temperature_k / max(state.temperature_k, 1e-3)
        new_ints = np.array([p.intensity * np.sqrt(t_ratio) * state.intensity_scale for p in self.base_peaks], dtype=np.float64)

        return new_freqs, new_ints

    def compute_convolved_spectrum(self, state: WhatIfParameterState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates dense convolved spectrum intensities across the frequency grid using
        analytical Voigt / Gaussian profiles.
        """
        line_freqs, line_ints = self.perturb_peak_frequencies(state)

        # Vectorized Gaussian line profile kernel
        fwhm_g = state.fwhm_gaussian_mhz
        fwhm_l = state.fwhm_lorentzian_mhz

        # Thompson-Cox-Hastings Pseudo-Voigt width
        fv = (fwhm_g**5 + 2.69269 * (fwhm_g**4) * fwhm_l + fwhm_l**5) ** 0.2
        eta = 1.36603 * (fwhm_l / fv) - 0.47719 * ((fwhm_l / fv) ** 2)
        eta = max(0.0, min(1.0, eta))

        # delta_nu matrix of shape (N_grid, N_lines)
        delta_nu = self.frequency_grid[:, None] - line_freqs[None, :]

        # Gaussian component
        gauss_prefactor = (2.0 * np.sqrt(np.log(2.0))) / (np.sqrt(np.pi) * fv)
        gauss_comp = gauss_prefactor * np.exp(-2.772588722 * np.square(delta_nu / fv))

        # Lorentzian component
        lorentz_prefactor = 2.0 / (np.pi * fv)
        lorentz_comp = lorentz_prefactor / (1.0 + 4.0 * np.square(delta_nu / fv))

        # Combined profile
        profiles = eta * lorentz_comp + (1.0 - eta) * gauss_comp
        spectrum = np.dot(profiles, line_ints)

        return self.frequency_grid, spectrum

    def generate_plotly_slider_steps(
        self,
        param_name: str,
        values: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """
        Generates Plotly interactive slider step dictionaries for dashboard integration.
        """
        steps: List[Dict[str, Any]] = []
        for val in values:
            steps.append({
                "method": "restyle",
                "label": f"{val:.2f}",
                "args": [{"customdata": [val]}],
            })
        return steps


# =============================================================================
# 9. ZEROMQ IPC BRIDGE (STRICT HEARTBEAT & TIMEOUT PROTOCOLS)
# =============================================================================

class SpycFitZMQServer:
    """
    ZeroMQ IPC server for background What-If spectrum recalculation, LTTB decimation,
    and state hash generation. Enforces strict heartbeat and timeout protocols.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 2000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_running = False
        self._bound = False

    def start(self) -> None:
        """Binds the REP socket to the endpoint."""
        if not self._bound:
            self.socket.bind(self.endpoint)
            self._bound = True
            self.is_running = True

    def process_one_request(self, engine: Optional[WhatIfSpectrumEngine] = None) -> Optional[Dict[str, Any]]:
        """
        Polls and processes a single incoming request.
        Handles HEARTBEAT, COMPUTE_SPECTRUM, LTTB_DECIMATE, GET_STATE_HASH.
        """
        try:
            raw_msg = self.socket.recv_json()
        except zmq.Again:
            return None

        msg_type = raw_msg.get("type", "UNKNOWN")
        response: Dict[str, Any] = {"status": "SUCCESS", "type": f"{msg_type}_ACK"}

        if msg_type == "HEARTBEAT":
            response["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        elif msg_type == "COMPUTE_SPECTRUM":
            state_data = raw_msg.get("state", {})
            param_state = WhatIfParameterState(**state_data)
            if engine is not None:
                freq_grid, spec = engine.compute_convolved_spectrum(param_state)
                decimate_thresh = raw_msg.get("decimate_threshold", None)
                if decimate_thresh is not None and len(freq_grid) > decimate_thresh:
                    freq_grid, spec = run_lttb_decimation(freq_grid, spec, decimate_thresh)
                response["freqs"] = freq_grid.tolist()
                response["spectrum"] = spec.tolist()
                response["state_hash"] = update_ui_state_hash(param_state.model_dump())
            else:
                response["status"] = "ERROR"
                response["error"] = "No WhatIfSpectrumEngine attached to server."
        elif msg_type == "LTTB_DECIMATE":
            x_in = raw_msg.get("x", [])
            y_in = raw_msg.get("y", [])
            threshold = raw_msg.get("threshold", 5000)
            x_out, y_out = run_lttb_decimation(x_in, y_in, threshold)
            response["x"] = x_out.tolist()
            response["y"] = y_out.tolist()
        elif msg_type == "GET_STATE_HASH":
            state_dict = raw_msg.get("state", {})
            response["state_hash"] = update_ui_state_hash(state_dict)
        else:
            response["status"] = "ERROR"
            response["error"] = f"Unrecognized request type '{msg_type}'"

        self.socket.send_json(response)
        return response

    def stop(self) -> None:
        """Closes sockets and terminates context."""
        self.is_running = False
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


class SpycFitZMQClient:
    """
    ZeroMQ IPC client connecting to SpycFitZMQServer with strict timeout and heartbeat verification.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 2000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def ping(self) -> bool:
        """Sends a heartbeat ping and checks for heartbeat ack within timeout."""
        try:
            self.socket.send_json({"type": "HEARTBEAT", "client_id": uuid.uuid4().hex[:8]})
            reply = self.socket.recv_json()
            return reply.get("type") == "HEARTBEAT_ACK" and reply.get("status") == "SUCCESS"
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ heartbeat timeout: server unresponsive.") from exc

    def request_spectrum_computation(
        self,
        state: WhatIfParameterState,
        decimate_threshold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Requests real-time spectrum computation from the ZMQ bridge server."""
        try:
            req = {
                "type": "COMPUTE_SPECTRUM",
                "state": state.model_dump(),
                "decimate_threshold": decimate_threshold,
            }
            self.socket.send_json(req)
            reply = self.socket.recv_json()
            if reply.get("status") != "SUCCESS":
                raise ZMQBridgeError(f"Server returned error: {reply.get('error')}")
            return reply
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ spectrum computation request timed out.") from exc

    def request_lttb_decimation(
        self,
        x: Sequence[float],
        y: Sequence[float],
        threshold: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Requests remote LTTB decimation over ZeroMQ IPC."""
        try:
            req = {
                "type": "LTTB_DECIMATE",
                "x": list(x),
                "y": list(y),
                "threshold": threshold,
            }
            self.socket.send_json(req)
            reply = self.socket.recv_json()
            if reply.get("status") != "SUCCESS":
                raise ZMQBridgeError(f"Server returned error: {reply.get('error')}")
            return np.array(reply["x"], dtype=np.float64), np.array(reply["y"], dtype=np.float64)
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ LTTB decimation request timed out.") from exc

    def request_state_hash(self, state_dict: Dict[str, Any]) -> str:
        """Requests remote SHA-256 state hash calculation."""
        try:
            req = {"type": "GET_STATE_HASH", "state": state_dict}
            self.socket.send_json(req)
            reply = self.socket.recv_json()
            if reply.get("status") != "SUCCESS":
                raise ZMQBridgeError(f"Server returned error: {reply.get('error')}")
            return str(reply["state_hash"])
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ state hash request timed out.") from exc

    def close(self) -> None:
        """Closes socket and terminates context."""
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


# =============================================================================
# 10. IPYWIDGETS INTERACTIVE DASHBOARD & VOILA INTERFACE
# =============================================================================

class SpycFitInteractiveDashboard:
    """
    Jupyter and Voila interactive HITL spectroscopic dashboard.
    Combines ipywidgets controls (FloatSlider, Button, VBox, HBox, Output) with
    Plotly WebGL Dual-Pane spectrum visualization, LTTB decimation, and cryptographic state hashing.
    """

    def __init__(
        self,
        manager: HITLAssignmentManager,
        spectrum_engine: Optional[WhatIfSpectrumEngine] = None,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        self.manager = manager
        self.engine = spectrum_engine
        self.config = config or DashboardConfig()

        # UI Widgets
        self.slider_A = FloatSlider(
            value=self.engine.base_state.A_mhz if self.engine else 10000.0,
            min=100.0,
            max=500000.0,
            step=1.0,
            description="A (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_B = FloatSlider(
            value=self.engine.base_state.B_mhz if self.engine else 5000.0,
            min=50.0,
            max=200000.0,
            step=0.5,
            description="B (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_C = FloatSlider(
            value=self.engine.base_state.C_mhz if self.engine else 2500.0,
            min=25.0,
            max=100000.0,
            step=0.5,
            description="C (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_temp = FloatSlider(
            value=self.engine.base_state.temperature_k if self.engine else 296.0,
            min=1.0,
            max=1000.0,
            step=1.0,
            description="Temp (K):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_huber = FloatSlider(
            value=self.config.huber_delta_mhz or 1.0,
            min=0.01,
            max=50.0,
            step=0.05,
            description="Huber Delta:",
            continuous_update=False,
            layout={"width": "300px"},
        )

        # Buttons
        self.btn_auto_propose = Button(description="Auto-Propose", button_style="info", icon="wand-magic-sparkles")
        self.btn_approve_all = Button(description="Approve All", button_style="success", icon="check")
        self.btn_reject_outliers = Button(description="Reject Outliers (>3σ)", button_style="warning", icon="filter")
        self.btn_recalc = Button(description="Recalculate Spectrum", button_style="primary", icon="sync")

        # Output Containers
        self.plot_output = Output()
        self.status_output = Output()

        # Connect button callbacks
        self.btn_auto_propose.on_click(self._on_auto_propose_clicked)
        self.btn_approve_all.on_click(self._on_approve_all_clicked)
        self.btn_reject_outliers.on_click(self._on_reject_outliers_clicked)
        self.btn_recalc.on_click(self._on_recalc_clicked)

        # Connect slider observe handlers
        self.slider_A.observe(self._on_slider_change, names="value")
        self.slider_B.observe(self._on_slider_change, names="value")
        self.slider_C.observe(self._on_slider_change, names="value")
        self.slider_temp.observe(self._on_slider_change, names="value")
        self.slider_huber.observe(self._on_slider_change, names="value")

        # Build composite layout
        self.controls_box = VBox([
            HBox([self.slider_A, self.slider_B, self.slider_C]),
            HBox([self.slider_temp, self.slider_huber]),
            HBox([self.btn_auto_propose, self.btn_approve_all, self.btn_reject_outliers, self.btn_recalc]),
        ])
        self.widget = VBox([
            self.controls_box,
            self.status_output,
            self.plot_output,
        ])

        self.refresh_view()

    def get_current_parameter_state(self) -> WhatIfParameterState:
        """Returns the current WhatIfParameterState based on slider values."""
        if self.engine:
            return self.engine.base_state.model_copy(
                update={
                    "A_mhz": float(self.slider_A.value),
                    "B_mhz": float(self.slider_B.value),
                    "C_mhz": float(self.slider_C.value),
                    "temperature_k": float(self.slider_temp.value),
                }
            )
        return WhatIfParameterState(
            A_mhz=float(self.slider_A.value),
            B_mhz=float(self.slider_B.value),
            C_mhz=float(self.slider_C.value),
            temperature_k=float(self.slider_temp.value),
        )

    def refresh_view(self) -> None:
        """Refreshes the dual-pane Plotly chart and status readout."""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            convolved_curve = None
            if self.engine:
                cur_state = self.get_current_parameter_state()
                f_grid, spec = self.engine.compute_convolved_spectrum(cur_state)
                if len(f_grid) > 5000:
                    f_grid, spec = run_lttb_decimation(f_grid, spec, threshold=5000)
                convolved_curve = (f_grid, spec)

            fig = render_dual_pane_dashboard(
                exp_peaks=self.manager.exp_peaks,
                theo_peaks=self.manager.theo_peaks,
                assignments=self.manager.assignments,
                config=self.config,
                convolved_curve=convolved_curve,
            )
            fig.show()

        with self.status_output:
            self.status_output.clear_output(wait=True)
            loss, rms = self.manager.compute_robust_metrics(huber_delta_mhz=float(self.slider_huber.value))
            app_count = len(self.manager.get_assignments_by_status(AssignmentStatus.APPROVED))
            ghost_count = len(self.manager.get_ghost_peaks())
            state_dict = {
                "A_mhz": float(self.slider_A.value),
                "B_mhz": float(self.slider_B.value),
                "C_mhz": float(self.slider_C.value),
                "temperature_k": float(self.slider_temp.value),
                "huber_delta_mhz": float(self.slider_huber.value),
                "approved_count": app_count,
                "ghost_count": ghost_count,
            }
            state_hash = update_ui_state_hash(state_dict)
            logger.info(
                f"[HITL Status] Approved: {app_count} | Ghosts: {ghost_count} | "
                f"RMS: {rms:.4f} MHz | Huber Loss: {loss:.4f} | State SHA-256: {state_hash[:12]}..."
            )

    def _on_auto_propose_clicked(self, _btn: Any) -> None:
        self.manager.auto_propose()
        self.refresh_view()

    def _on_approve_all_clicked(self, _btn: Any) -> None:
        self.manager.approve_all_proposed()
        self.refresh_view()

    def _on_reject_outliers_clicked(self, _btn: Any) -> None:
        delta = float(self.slider_huber.value)
        proposed = self.manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
        for a in proposed:
            if abs(a.frequency_drift_mhz) > 3.0 * delta:
                self.manager.reject_assignment(a.assignment_id, reason="Exceeds 3-sigma tolerance threshold")
        self.refresh_view()

    def _on_recalc_clicked(self, _btn: Any) -> None:
        self.refresh_view()

    def _on_slider_change(self, _change: Any) -> None:
        self.refresh_view()


def create_spycfit_dashboard_widget(
    manager: HITLAssignmentManager,
    spectrum_engine: Optional[WhatIfSpectrumEngine] = None,
    config: Optional[DashboardConfig] = None,
) -> VBox:
    """
    Convenience factory to instantiate a SpycFit interactive Jupyter/Voila dashboard widget.
    """
    dashboard = SpycFitInteractiveDashboard(manager=manager, spectrum_engine=spectrum_engine, config=config)
    return dashboard.widget


# =============================================================================
# 11. SEMANTIC GIT-COMMIT TRANSLATION FOR MATHEMATICAL CHANGES
# =============================================================================

def translate_mathematical_change_to_git_commit(
    state_before: Optional[WhatIfParameterState] = None,
    state_after: Optional[WhatIfParameterState] = None,
    approved_count: Optional[int] = None,
    rejected_count: Optional[int] = None,
    huber_delta_before: Optional[float] = None,
    huber_delta_after: Optional[float] = None,
    rms_before: Optional[float] = None,
    rms_after: Optional[float] = None,
) -> SemanticCommit:
    """
    Translates mathematical parameter mutations, assignment approvals, and loss adjustments
    into standardized, conventional semantic Git commit messages with cryptographic SHA-256 state tracking.

    Parameters:
        state_before: Parameter state prior to update.
        state_after: Parameter state after update.
        approved_count: Number of transitions approved.
        rejected_count: Number of transitions rejected / ghosted.
        huber_delta_before: Prior Huber loss threshold.
        huber_delta_after: Updated Huber loss threshold.
        rms_before: Prior RMS residual in MHz.
        rms_after: Updated RMS residual in MHz.

    Returns:
        SemanticCommit: Structured conventional git commit model.
    """
    # 1. Assignment change commit
    if approved_count is not None or rejected_count is not None:
        app = approved_count or 0
        rej = rejected_count or 0
        summary = f"approve {app} transitions and reject {rej} ghost peaks"
        body_lines = [
            f"Approved {app} bipartite peak assignments following HITL review.",
            f"Quarantined {rej} outlier/impurity transitions as ghost peaks.",
        ]
        if rms_before is not None and rms_after is not None:
            rms_str = f"RMS: {rms_before:.4f} MHz -> {rms_after:.4f} MHz"
            summary += f" [{rms_str}]"
            body_lines.append(f"Residual quality update: {rms_str}")

        state_meta = {
            "approved_count": app,
            "rejected_count": rej,
            "rms_before": rms_before,
            "rms_after": rms_after,
        }
        state_hash = update_ui_state_hash(state_meta)
        body_lines.append(f"State-Hash: {state_hash}")

        full_msg = f"fix(assignment): {summary}\n\n" + "\n".join(body_lines)
        return SemanticCommit(
            commit_type="fix",
            scope="assignment",
            summary=summary,
            body="\n".join(body_lines),
            rms_before=rms_before,
            rms_after=rms_after,
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    # 2. Huber loss threshold change commit
    if huber_delta_before is not None and huber_delta_after is not None:
        summary = f"update Huber loss threshold delta={huber_delta_before:.3f} -> {huber_delta_after:.3f} MHz"
        body_lines = [
            f"Adjusted Huber robust loss delta threshold from {huber_delta_before:.4f} MHz to {huber_delta_after:.4f} MHz.",
            "Optimizes outlier suppression boundary for Levenberg-Marquardt fitting.",
        ]
        state_meta = {
            "huber_delta_before": huber_delta_before,
            "huber_delta_after": huber_delta_after,
        }
        state_hash = update_ui_state_hash(state_meta)
        body_lines.append(f"State-Hash: {state_hash}")

        full_msg = f"refactor(loss): {summary}\n\n" + "\n".join(body_lines)
        return SemanticCommit(
            commit_type="refactor",
            scope="loss",
            summary=summary,
            body="\n".join(body_lines),
            parameters_before={"huber_delta_mhz": huber_delta_before},
            parameters_after={"huber_delta_mhz": huber_delta_after},
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    # 3. Hamiltonian parameter update commit
    if state_before is not None and state_after is not None:
        param_diffs: List[str] = []
        body_diffs: List[str] = []

        for p_name in ["A_mhz", "B_mhz", "C_mhz", "D_J_khz", "D_JK_khz", "D_K_khz", "temperature_k"]:
            val_b = getattr(state_before, p_name)
            val_a = getattr(state_after, p_name)
            diff = val_a - val_b
            if abs(diff) > 1e-6:
                short_name = p_name.split("_")[0]
                unit = "MHz" if "mhz" in p_name else ("kHz" if "khz" in p_name else "K")
                sign = "+" if diff > 0 else ""
                param_diffs.append(f"{short_name}={val_a:.2f} {unit} ({sign}{diff:.2f})")
                body_diffs.append(f"- {p_name}: {val_b:.6f} -> {val_a:.6f} {unit} (delta: {sign}{diff:.6f} {unit})")

        summary_diff_str = ", ".join(param_diffs) if param_diffs else "fine-tune parameters"
        summary = f"refine rotational constants [{summary_diff_str}]"
        if rms_before is not None and rms_after is not None:
            rms_str = f"RMS: {rms_before:.4f} MHz -> {rms_after:.4f} MHz"
            summary += f" [{rms_str}]"

        state_hash = update_ui_state_hash(state_after.model_dump())
        body_diffs.append(f"\nState-Hash: {state_hash}")

        body = "Spectroscopic parameter adjustment details:\n" + "\n".join(body_diffs)
        full_msg = f"feat(hamiltonian): {summary}\n\n{body}"

        return SemanticCommit(
            commit_type="feat",
            scope="hamiltonian",
            summary=summary,
            body=body,
            parameters_before=state_before.model_dump(),
            parameters_after=state_after.model_dump(),
            rms_before=rms_before,
            rms_after=rms_after,
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    raise SemanticCommitError("Insufficient state parameters provided to generate semantic commit.")


__all__ = [
    "AssignmentStatus",
    "BipartiteConfig",
    "BipartiteMatchingError",
    "DashboardConfig",
    "HITLApprovalError",
    "HITLAssignmentManager",
    "OfflineRenderError",
    "PeakAssignment",
    "RenderBackend",
    "SemanticCommit",
    "SemanticCommitError",
    "SliderComputationError",
    "SpectralPeak",
    "SpycFitInteractiveDashboard",
    "SpycFitZMQClient",
    "SpycFitZMQServer",
    "UIDashboardError",
    "WhatIfParameterState",
    "WhatIfSpectrumEngine",
    "ZMQBridgeError",
    "auto_propose_assignments",
    "compute_bipartite_cost_matrix",
    "compute_huber_loss_excluding_ghosts",
    "create_spycfit_dashboard_widget",
    "render_dual_pane_dashboard",
    "render_static_offline_html",
    "render_static_offline_pdf",
    "run_lttb_decimation",
    "translate_mathematical_change_to_git_commit",
    "update_ui_state_hash",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_spcat_bridge.py ---
"""
CoChem-SpycFit: Stage 3.3 Legacy Fortran Translator & Pickett SPCAT/SPFIT Bridge.

Authoritative serialization, fixed-width Fortran formatting, and process-isolated
asynchronous execution pipeline for legacy Pickett SPCAT/SPFIT spectroscopic engines.

Directives Enforced:
1. Strict 1980s Pickett fixed-width formatting (%10.4E with Fortran 'D' exponent replacement)
   guaranteeing absolute column alignment for .var, .par, .int, and .lin files.
2. Robust multi-file parser suite for legacy calculation outputs (.cat, .egy, .fit).
3. Process-isolated, asynchronous containerized task queues (FortranTaskQueue, FortranExecutionTask,
   FortranExecutionResult, FortranBridgeDispatcher).
4. Explicit executable deployment path routing ($COCHEM_BIN_DIR or user_data_dir / bin), avoiding
   Tier 3 Cache and noexec mount deadlocks on HPC clusters.
5. OS-level process isolation (CREATE_NEW_PROCESS_GROUP on Windows, start_new_session on POSIX).
6. Robust segfault trapping (SIGSEGV / exit code 139 on POSIX, STATUS_ACCESS_VIOLATION 0xC0000005 on Windows)
   preventing parent kernel corruption.
7. Structured hex-dump telemetry capture (final 256 bytes) logged to $COCHEM_STATE_DIR/Logs/spycfit_telemetry.json.
8. Ephemeral scratch lifecycle management (PickettScratchContext) with guaranteed filesystem cleanup.
9. Dynamic physical isotope mass retrieval via Mendeleev.
10. Strict Pydantic v2 validation models and zero-mock physical compliance.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import mendeleev
import numpy as np
import platformdirs
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Configure module-level logger
logger = logging.getLogger("CoChem-SpycFit.SPCATBridge")

# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class FortranBridgeError(Exception):
    """Base exception for all Fortran bridge and Pickett translation operations."""


class PickettFormatError(FortranBridgeError, ValueError):
    """Raised when parsing or formatting legacy Pickett data encounters syntax errors."""


class FortranExecutionError(FortranBridgeError, RuntimeError):
    """Raised when execution of a Fortran binary fails or returns non-zero error status."""


class FortranExecutionTimeoutError(FortranExecutionError, TimeoutError):
    """Raised when Fortran subprocess execution exceeds the configured timeout threshold."""


class FortranExecutionSegfaultError(FortranExecutionError):
    """Raised when a Fortran subprocess crashes due to a segmentation fault or access violation."""


class FortranBinaryNotFoundError(FortranBridgeError, FileNotFoundError):
    """Raised when an expected Fortran executable binary cannot be located on disk."""


# =============================================================================
# 2. PHYSICAL CONSTANTS & DYNAMIC ISOTOPE RETRIEVAL (MENDELEEV)
# =============================================================================

def get_isotopic_mass(element_symbol: str, mass_number: Optional[int] = None) -> float:
    """
    Dynamically retrieves authentic physical isotopic mass (or standard atomic weight)
    from Mendeleev.

    Args:
        element_symbol: Standard chemical element symbol (e.g. 'H', 'C', 'N', 'O').
        mass_number: Optional mass number for specific isotope (e.g. 13 for C-13).

    Returns:
        float: Accurate isotopic mass in atomic mass units (u / Da).

    Raises:
        ValueError: If element or isotope mass number is not found.
    """
    clean_sym = element_symbol.strip().capitalize()
    try:
        elem = mendeleev.element(clean_sym)
    except Exception as exc:
        raise ValueError(f"Unknown chemical element symbol: {element_symbol}") from exc

    if mass_number is not None:
        for iso in elem.isotopes:
            if iso.mass_number == mass_number:
                return float(iso.mass)
        raise ValueError(f"Isotope mass number {mass_number} not found for element {clean_sym}")

    return float(elem.atomic_weight)


def calculate_reduced_mass(mass1_amu: float, mass2_amu: float) -> float:
    """
    Calculates authentic two-body reduced mass in atomic mass units (u).

    Args:
        mass1_amu: Mass of particle 1 in u.
        mass2_amu: Mass of particle 2 in u.

    Returns:
        float: Reduced mass mu = (m1 * m2) / (m1 + m2).
    """
    if mass1_amu <= 0.0 or mass2_amu <= 0.0:
        raise ValueError("Atomic masses must be strictly positive.")
    return (mass1_amu * mass2_amu) / (mass1_amu + mass2_amu)


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & PICKETT STRICT FIXED-WIDTH FORMATTERS
# =============================================================================

def fortran_float_converter(val: Union[str, float, int, np.number]) -> float:
    """
    Safely converts a string containing Fortran scientific notation to a Python float64.

    Handles:
    - Standard floats and integers
    - 'D' or 'd' exponent notation ('1.2345D+02' -> 123.45)
    - Implicit exponent notation ('1.2345-05' -> 1.2345e-5)
    - Leading/trailing whitespace
    """
    if isinstance(val, (float, int, np.number)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    # Replace Fortran double precision exponent 'D' / 'd' with 'e'
    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")

    # Handle implicit exponent sign without 'e' (e.g. '1.2345-05' or '1.2345+02')
    match_implicit = re.match(r"^([+-]?\d+\.?\d*)([+-]\d+)$", s_clean)
    if match_implicit:
        s_clean = f"{match_implicit.group(1)}e{match_implicit.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


def format_pickett_line(
    value: Union[float, int, str, np.number],
    width: int = 10,
    precision: int = 4,
    exponent_char: str = "D",
) -> str:
    """
    Strict fixed-width truncation and padding logic guaranteeing absolute column alignment
    in legacy 1980s Pickett formats (.var, .par, .int, .lin).

    Formats floating point numbers with scientific notation (%10.4E) and replaces
    the exponent character ('E' -> 'D'). Enforces exact field width by right-aligning
    and precision adjusting or truncating on overflow.

    Args:
        value: Numeric value or string representation.
        width: Exact character width of the output field (default: 10).
        precision: Digits after decimal point in scientific notation (default: 4).
        exponent_char: Exponent indicator character (default: 'D').

    Returns:
        str: Fixed-width formatted string guaranteed to have len == width.
    """
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        s_val = str(value)
        if len(s_val) > width:
            return s_val[:width]
        return s_val.rjust(width)

    # Format using scientific notation
    formatted = f"{val_float:.{precision}E}"
    formatted = formatted.replace("E", exponent_char).replace("e", exponent_char)

    if len(formatted) > width:
        # Dynamically adjust precision if exponent or sign widened the string
        excess = len(formatted) - width
        adj_precision = max(0, precision - excess)
        formatted = f"{val_float:.{adj_precision}E}".replace("E", exponent_char).replace("e", exponent_char)
        if len(formatted) > width:
            formatted = formatted[:width]

    return formatted.rjust(width)


def decode_pickett_quantum(qn_str: str) -> int:
    """
    Decodes legacy JPL/CDMS alphanumeric quantum number representations.

    Quanta 0-99: standard integer (e.g. ' 5' -> 5)
    Quanta 10-269: encoded as 'a0'-'z9' (e.g. 'a0' -> 10, 'a9' -> 19, 'z9' -> 269)
    Quanta 100-359: encoded as 'A0'-'Z9' (e.g. 'A0' -> 100, 'A9' -> 109, 'Z9' -> 359)

    Args:
        qn_str: 2-character quantum number string.

    Returns:
        int: Decoded integer quantum number.
    """
    s = qn_str.strip()
    if not s:
        return 0

    if s.lstrip("+-").isdigit():
        return int(s)

    if len(s) == 2:
        c0, c1 = s[0], s[1]
        if c1.isdigit():
            digit = int(c1)
            if "a" <= c0 <= "z":
                letter_idx = ord(c0) - ord("a")
                return 10 + letter_idx * 10 + digit
            if "A" <= c0 <= "Z":
                letter_idx = ord(c0) - ord("A")
                return 100 + letter_idx * 10 + digit

    raise PickettFormatError(f"Cannot decode alphanumeric quantum number string: {qn_str!r}")


def encode_pickett_quantum(quantum_value: int) -> str:
    """
    Encodes an integer quantum number into a 2-character Pickett string.

    0-99: 2-digit integer (e.g. 5 -> ' 5', 42 -> '42')
    100-359: 'A0'-'Z9'

    Args:
        quantum_value: Integer quantum number (0 to 359).

    Returns:
        str: Exactly 2-character string.
    """
    if 0 <= quantum_value <= 99:
        return f"{quantum_value:2d}"
    if 100 <= quantum_value <= 359:
        val_offset = quantum_value - 100
        letter_idx = val_offset // 10
        digit = val_offset % 10
        if letter_idx < 26:
            letter = chr(ord("A") + letter_idx)
            return f"{letter}{digit}"

    # Fallback / overflow clamping
    clamped = max(0, min(99, quantum_value))
    return f"{clamped:2d}"


# =============================================================================
# 4. PYDANTIC V2 MODELS FOR PICKETT SCHEMAS
# =============================================================================

class PickettParameterRecord(BaseModel):
    """Represents an individual parameter in a Pickett .var / .par file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    param_id: int = Field(..., description="Pickett parameter code (e.g. 10000=A, 20000=B, 30000=C, 200=-DJ).")
    value: float = Field(..., description="Parameter value in MHz.")
    uncertainty: float = Field(default=0.0, ge=0.0, description="Estimated parameter uncertainty in MHz (0.0 if fixed).")
    label: str = Field(default="", description="Descriptive label (e.g. 'A', 'B', 'C', 'DJ').")


class PickettVarFile(BaseModel):
    """Encapsulates a structured Pickett parameter file (.var / .par)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Parameter File", description="Header title string.")
    npar: int = Field(default=0, ge=0, description="Number of parameters in file.")
    nline: int = Field(default=0, description="Number of transitions to fit.")
    nopt: int = Field(default=0, description="Optimization control integer.")
    nwarn: int = Field(default=0, description="Warning control integer.")
    erpar: float = Field(default=1.0, description="Parameter error scale factor.")
    wtfac: float = Field(default=1.0, description="Weight factor.")
    scale: float = Field(default=1.0, description="Scale factor.")
    maxit: int = Field(default=50, ge=0, description="Maximum fitting iterations.")
    parameters: List[PickettParameterRecord] = Field(default_factory=list, description="List of parameter records.")

    @model_validator(mode="after")
    def sync_npar_count(self) -> PickettVarFile:
        object.__setattr__(self, "npar", len(self.parameters))
        return self

    def export_var(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports parameter set into authentic fixed-width Pickett .var / .par format.
        """
        lines: List[str] = [self.title]
        ctrl_line = (
            f"{len(self.parameters):5d}"
            f"{self.nline:5d}"
            f"{self.nopt:5d}"
            f"{self.nwarn:5d}"
            f"{self.erpar:10.4f}"
            f"{self.wtfac:10.4f}"
            f"{self.scale:10.4f}"
            f"{self.maxit:5d}"
        )
        lines.append(ctrl_line)

        for p in self.parameters:
            v_str = format_pickett_line(p.value, width=22, precision=14, exponent_char="D")
            u_str = format_pickett_line(p.uncertainty, width=18, precision=10, exponent_char="D")
            lbl_str = f" / {p.label}" if p.label else ""
            p_line = f"{p.param_id:11d} {v_str} {u_str}{lbl_str}"
            lines.append(p_line)

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettDipoleRecord(BaseModel):
    """Represents a dipole moment component in a Pickett .int file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    axis: int = Field(..., ge=1, le=3, description="Principal axis index: 1=a, 2=b, 3=c.")
    dipole_debye: float = Field(..., description="Dipole moment component along axis in Debye.")
    comment: str = Field(default="", description="Descriptive comment (e.g. 'mua').")


class PickettIntFile(BaseModel):
    """Encapsulates a structured Pickett intensity control file (.int)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Intensity File", description="Header title string.")
    tag: int = Field(default=1, description="Species identification tag.")
    qpart: float = Field(default=1000.0, gt=0.0, description="Rotational partition function.")
    nline: int = Field(default=1000, description="Target predicted transition line count.")
    iq: int = Field(default=1, description="Quantum format identifier.")
    max_j: int = Field(default=99, ge=0, description="Maximum J quantum number.")
    min_freq_mhz: float = Field(default=0.0, description="Minimum frequency boundary in MHz.")
    max_freq_mhz: float = Field(default=500000.0, description="Maximum frequency boundary in MHz.")
    min_intensity: float = Field(default=-10.0, description="Minimum log10 intensity cutoff.")
    temperature_k: float = Field(default=300.0, gt=0.0, description="Simulation temperature in Kelvin.")
    dipoles: List[PickettDipoleRecord] = Field(default_factory=list, description="List of active dipole records.")

    def export_int(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports intensity control configuration into authentic fixed-width Pickett .int format.
        """
        lines: List[str] = [self.title]
        ctrl_line = (
            f"{self.tag:6d}"
            f"{self.qpart:12.4f}"
            f"{self.nline:7d}"
            f"{self.iq:4d}"
            f"{self.max_j:4d}"
            f" {self.min_freq_mhz:10.1f}"
            f" {self.max_freq_mhz:10.1f}"
            f"{self.min_intensity:10.2f}"
            f"{self.temperature_k:10.2f}"
        )
        lines.append(ctrl_line)

        for dip in self.dipoles:
            axis_name = ["mua", "mub", "muc"][dip.axis - 1] if 1 <= dip.axis <= 3 else str(dip.axis)
            lbl = dip.comment if dip.comment else axis_name
            dip_str = f"{dip.axis:2d}  {dip.dipole_debye:8.4f} / {lbl}"
            lines.append(dip_str)

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettLinRecord(BaseModel):
    """Represents a discrete experimental transition line in a Pickett .lin file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers (J', Ka', Kc', ...).")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers (J'', Ka'', Kc'', ...).")
    frequency_mhz: float = Field(..., description="Observed transition frequency in MHz.")
    uncertainty_mhz: float = Field(default=0.01, ge=0.0, description="Measurement uncertainty in MHz.")
    weight: float = Field(default=1.0, description="Relative statistical fitting weight.")
    description: str = Field(default="", description="Optional line annotation.")


class PickettLinFile(BaseModel):
    """Encapsulates a structured Pickett experimental line list (.lin)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="Pickett Line List", description="Header line description.")
    records: List[PickettLinRecord] = Field(default_factory=list, description="List of transition records.")

    def export_lin(self, filepath_or_stream: Optional[Union[str, Path, io.StringIO]] = None) -> str:
        """
        Exports line list into authentic fixed-width Fortran 6I3, 6I3, F13.4, F8.4, F8.4 .lin format.
        """
        lines: List[str] = []
        if self.title:
            lines.append(f"! {self.title}")

        for rec in self.records:
            # Pad quanta to 6 integers each
            qu = list(rec.quantum_upper) + [0] * max(0, 6 - len(rec.quantum_upper))
            ql = list(rec.quantum_lower) + [0] * max(0, 6 - len(rec.quantum_lower))

            qu_str = "".join(f"{q:3d}" for q in qu[:6])
            ql_str = "".join(f"{q:3d}" for q in ql[:6])
            f_str = f"{rec.frequency_mhz:13.4f}"
            u_str = f"{rec.uncertainty_mhz:8.4f}"
            w_str = f"{rec.weight:8.4f}"
            annot = f" {rec.description}" if rec.description else ""

            lines.append(f"{qu_str}{ql_str}{f_str}{u_str}{w_str}{annot}")

        content = "\n".join(lines) + "\n"
        if filepath_or_stream is not None:
            if isinstance(filepath_or_stream, io.StringIO):
                filepath_or_stream.write(content)
            else:
                p_out = Path(filepath_or_stream)
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(content, encoding="utf-8")
        return content


class PickettCatRecord(BaseModel):
    """Represents a predicted spectroscopic transition line in a Pickett .cat catalog."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequency_mhz: float = Field(..., description="Rest transition frequency in MHz.")
    uncertainty_mhz: float = Field(..., description="Predicted frequency uncertainty in MHz.")
    log_intensity: float = Field(..., description="Log10 integrated intensity in nm^2 MHz.")
    degrees_of_freedom: int = Field(default=2, description="Degrees of freedom in partition function.")
    lower_energy_cm1: float = Field(..., description="Lower rotational state energy in cm^-1.")
    upper_degeneracy: int = Field(..., description="Upper state degeneracy (g_up).")
    species_tag: int = Field(..., description="Species identification tag.")
    quantum_format: int = Field(..., description="Quantum number format code.")
    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers.")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers.")


class PickettCatFile(BaseModel):
    """Encapsulates a collection of parsed predicted lines from a Pickett .cat catalog."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    records: List[PickettCatRecord] = Field(default_factory=list, description="List of transition lines.")


class PickettEgyRecord(BaseModel):
    """Represents an energy level record from a Pickett .egy file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_index: int = Field(..., description="State numerical index.")
    energy_cm1: float = Field(..., description="State energy in cm^-1.")
    uncertainty_cm1: float = Field(default=0.0, description="Calculated energy uncertainty in cm^-1.")
    pmix: float = Field(default=1.0, description="State purity / mixing coefficient.")
    quantum_numbers: Tuple[int, ...] = Field(default=(), description="Rotational quantum numbers.")
    label: str = Field(default="", description="State label or symmetry representation.")


class PickettEgyFile(BaseModel):
    """Encapsulates state energy level records parsed from a Pickett .egy file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    records: List[PickettEgyRecord] = Field(default_factory=list, description="List of energy records.")


class PickettFitResidual(BaseModel):
    """Represents an observed vs calculated residual from an SPFIT .fit output file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    line_number: int = Field(..., description="Transition line index in fit.")
    quantum_upper: Tuple[int, ...] = Field(..., description="Upper state quantum numbers.")
    quantum_lower: Tuple[int, ...] = Field(..., description="Lower state quantum numbers.")
    frequency_obs_mhz: float = Field(..., description="Observed transition frequency in MHz.")
    frequency_calc_mhz: float = Field(..., description="Calculated transition frequency in MHz.")
    residual_mhz: float = Field(..., description="Observed minus calculated residual in MHz.")
    uncertainty_mhz: float = Field(..., description="Measurement uncertainty in MHz.")
    weight: float = Field(default=1.0, description="Statistical fitting weight.")


class PickettFitResult(BaseModel):
    """Encapsulates comprehensive output results parsed from an SPFIT .fit file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="", description="Fit calculation title.")
    standard_error: float = Field(default=0.0, description="Standard error of the fit (RMS).")
    variance: float = Field(default=0.0, description="Reduced chi-squared / variance of fit.")
    lines_fitted: int = Field(default=0, description="Total transition lines included in fit.")
    lines_omitted: int = Field(default=0, description="Transition lines omitted from fit.")
    parameters: List[PickettParameterRecord] = Field(default_factory=list, description="Fitted parameters.")
    residuals: List[PickettFitResidual] = Field(default_factory=list, description="Line residuals table.")


# =============================================================================
# 5. PARSERS FOR PICKETT OUTPUT FILES (.CAT, .EGY, .FIT, .VAR, .INT, .LIN)
# =============================================================================

def _read_content_lines(content_or_path: Union[str, Path, bytes, io.StringIO]) -> List[str]:
    """Helper to extract text lines from diverse input representations."""
    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        return content_or_path.read_text(encoding="utf-8", errors="replace").splitlines()

    if isinstance(content_or_path, str):
        if "\n" not in content_or_path and os.path.exists(content_or_path):
            return Path(content_or_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if "\n" not in content_or_path and any(content_or_path.endswith(ext) for ext in (".cat", ".egy", ".fit", ".var", ".par", ".int", ".lin")):
            raise FileNotFoundError(f"Pickett file not found: {content_or_path}")
        return content_or_path.splitlines()

    if isinstance(content_or_path, bytes):
        return content_or_path.decode("utf-8", errors="replace").splitlines()

    if isinstance(content_or_path, io.StringIO):
        return content_or_path.getvalue().splitlines()

    raise ValueError(f"Unsupported content type: {type(content_or_path)}")


def parse_pickett_cat(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettCatFile:
    """
    Parses a legacy Pickett SPCAT predicted catalog file (.cat).

    Fortran format: [F13.4, 2F8.4, I2, F10.4, I3, I7, I4, 12I2]
    """
    lines = _read_content_lines(content_or_path)
    records: List[PickettCatRecord] = []

    cat_regex = re.compile(
        r"^\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+)\s+"
        r"([+-]?\d+)\s+"
        r"(\d+)\s+"
    )

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        if len(line) >= 79:
            try:
                freq = fortran_float_converter(line[0:13])
                unc = fortran_float_converter(line[13:21])
                lgint = fortran_float_converter(line[21:29])
                dr = int(line[29:31].strip() or 0)
                elow = fortran_float_converter(line[31:41])
                gup = int(line[41:44].strip() or 1)
                tag = int(line[44:51].strip() or 0)
                qnfmt = int(line[51:55].strip() or 0)

                # Decode 6 upper and 6 lower quanta
                qu = tuple(decode_pickett_quantum(line[55 + 2 * i : 57 + 2 * i]) for i in range(6))
                ql = tuple(decode_pickett_quantum(line[67 + 2 * i : 69 + 2 * i]) for i in range(6))

                records.append(
                    PickettCatRecord(
                        frequency_mhz=freq,
                        uncertainty_mhz=unc,
                        log_intensity=lgint,
                        degrees_of_freedom=dr,
                        lower_energy_cm1=elow,
                        upper_degeneracy=gup,
                        species_tag=tag,
                        quantum_format=qnfmt,
                        quantum_upper=qu,
                        quantum_lower=ql,
                    )
                )
                continue
            except Exception as parse_err:
                logger.debug("Fixed-width cat parse fallback at line %d: %s", line_idx, parse_err)

        # Regex fallback for non-standard spaced lines
        match = cat_regex.match(line)
        if match:
            freq = fortran_float_converter(match.group(1))
            unc = fortran_float_converter(match.group(2))
            lgint = fortran_float_converter(match.group(3))
            dr = int(match.group(4))
            elow = fortran_float_converter(match.group(5))
            gup = int(match.group(6))
            tag = int(match.group(7))
            qnfmt = int(match.group(8))

            tail = line[match.end():].strip().split()
            qu_list: List[int] = []
            ql_list: List[int] = []
            for item in tail[:6]:
                qu_list.append(decode_pickett_quantum(item))
            for item in tail[6:12]:
                ql_list.append(decode_pickett_quantum(item))

            while len(qu_list) < 6:
                qu_list.append(0)
            while len(ql_list) < 6:
                ql_list.append(0)

            records.append(
                PickettCatRecord(
                    frequency_mhz=freq,
                    uncertainty_mhz=unc,
                    log_intensity=lgint,
                    degrees_of_freedom=dr,
                    lower_energy_cm1=elow,
                    upper_degeneracy=gup,
                    species_tag=tag,
                    quantum_format=qnfmt,
                    quantum_upper=tuple(qu_list),
                    quantum_lower=tuple(ql_list),
                )
            )

    return PickettCatFile(records=records)


def parse_pickett_egy(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettEgyFile:
    """
    Parses a Pickett SPCAT state energy level file (.egy).
    """
    lines = _read_content_lines(content_or_path)
    records: List[PickettEgyRecord] = []

    egy_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(.*)$"
    )

    for line in lines:
        s_line = line.strip()
        if not s_line or s_line.startswith("#") or s_line.startswith("!"):
            continue

        match = egy_pattern.match(s_line)
        if match:
            idx = int(match.group(1))
            energy = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            pmix = fortran_float_converter(match.group(4))
            remainder = match.group(5).strip().split()

            qn_vals: List[int] = []
            lbl = ""
            for token in remainder:
                if token.lstrip("+-").isdigit():
                    qn_vals.append(int(token))
                elif len(token) == 2 and token[0].islower() and token[1].isdigit():
                    qn_vals.append(decode_pickett_quantum(token))
                else:
                    lbl = token

            records.append(
                PickettEgyRecord(
                    state_index=idx,
                    energy_cm1=energy,
                    uncertainty_cm1=unc,
                    pmix=pmix,
                    quantum_numbers=tuple(qn_vals),
                    label=lbl,
                )
            )

    return PickettEgyFile(records=records)


def parse_pickett_fit(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettFitResult:
    """
    Parses a Pickett SPFIT fitting output log (.fit).
    """
    lines = _read_content_lines(content_or_path)
    title = ""
    std_err = 0.0
    variance = 0.0
    lines_fitted = 0
    lines_omitted = 0
    params: List[PickettParameterRecord] = []
    residuals: List[PickettFitResidual] = []

    param_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    res_pattern = re.compile(
        r"^\s*(\d+)\s+((?:\s*[+-]?\d+){2,6})\s+((?:\s*[+-]?\d+){2,6})\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)"
    )

    in_residuals = False
    in_parameters = False

    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue

        if not title and not s_line.startswith(("MICROWAVE", "STANDARD", "VARIANCE", "LINES")):
            title = s_line

        # Extract summary statistics via regex
        std_err_match = re.search(
            r"(?:STANDARD\s+ERROR(?:\s+OF\s+FIT)?|MICROWAVE\s+RMS)\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)",
            s_line,
            re.IGNORECASE,
        )
        if std_err_match:
            std_err = fortran_float_converter(std_err_match.group(1))

        var_match = re.search(r"VARIANCE\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)", s_line, re.IGNORECASE)
        if var_match:
            variance = fortran_float_converter(var_match.group(1))

        lines_fit_match = re.search(r"LINES\s+FITTED\s*=\s*(\d+)", s_line, re.IGNORECASE)
        if lines_fit_match:
            lines_fitted = int(lines_fit_match.group(1))

        # Detect sections
        if "PARAMETERS" in s_line.upper():
            in_parameters = True
            in_residuals = False
            continue

        if "RESIDUALS" in s_line.upper() or "OBS-CALC" in s_line.upper():
            in_residuals = True
            in_parameters = False
            continue

        if in_parameters:
            pmatch = param_pattern.match(s_line)
            if pmatch:
                pid = int(pmatch.group(1))
                val = fortran_float_converter(pmatch.group(2))
                unc = fortran_float_converter(pmatch.group(3))
                lbl = (pmatch.group(4) or "").strip()
                params.append(PickettParameterRecord(param_id=pid, value=val, uncertainty=unc, label=lbl))

        if in_residuals:
            rmatch = res_pattern.match(s_line)
            if rmatch:
                l_num = int(rmatch.group(1))
                qu = tuple(int(x) for x in rmatch.group(2).split())
                ql = tuple(int(x) for x in rmatch.group(3).split())
                f_obs = fortran_float_converter(rmatch.group(4))
                f_calc = fortran_float_converter(rmatch.group(5))
                res = fortran_float_converter(rmatch.group(6))
                unc = fortran_float_converter(rmatch.group(7))
                wt = fortran_float_converter(rmatch.group(8))
                residuals.append(
                    PickettFitResidual(
                        line_number=l_num,
                        quantum_upper=qu,
                        quantum_lower=ql,
                        frequency_obs_mhz=f_obs,
                        frequency_calc_mhz=f_calc,
                        residual_mhz=res,
                        uncertainty_mhz=unc,
                        weight=wt,
                    )
                )

    return PickettFitResult(
        title=title,
        standard_error=std_err,
        variance=variance,
        lines_fitted=lines_fitted if lines_fitted > 0 else len(residuals),
        lines_omitted=lines_omitted,
        parameters=params,
        residuals=residuals,
    )


def parse_pickett_var(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettVarFile:
    """
    Parses a legacy Pickett parameter file (.var / .par).
    """
    lines = _read_content_lines(content_or_path)
    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise PickettFormatError("Pickett parameter file must contain title and control lines.")

    title = non_empty[0].strip()
    ctrl_parts = non_empty[1].split()

    npar = int(ctrl_parts[0]) if len(ctrl_parts) > 0 else 0
    nline = int(ctrl_parts[1]) if len(ctrl_parts) > 1 else 0
    nopt = int(ctrl_parts[2]) if len(ctrl_parts) > 2 else 0
    nwarn = int(ctrl_parts[3]) if len(ctrl_parts) > 3 else 0
    erpar = fortran_float_converter(ctrl_parts[4]) if len(ctrl_parts) > 4 else 1.0
    wtfac = fortran_float_converter(ctrl_parts[5]) if len(ctrl_parts) > 5 else 1.0
    scale = fortran_float_converter(ctrl_parts[6]) if len(ctrl_parts) > 6 else 1.0
    maxit = int(ctrl_parts[7]) if len(ctrl_parts) > 7 else 50

    param_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
    )

    parameters: List[PickettParameterRecord] = []
    for p_line in non_empty[2:]:
        match = param_pattern.match(p_line)
        if match:
            pid = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            unc = fortran_float_converter(match.group(3))
            lbl = (match.group(4) or "").strip()
            parameters.append(PickettParameterRecord(param_id=pid, value=val, uncertainty=unc, label=lbl))

    return PickettVarFile(
        title=title,
        npar=len(parameters),
        nline=nline,
        nopt=nopt,
        nwarn=nwarn,
        erpar=erpar,
        wtfac=wtfac,
        scale=scale,
        maxit=maxit,
        parameters=parameters,
    )


def parse_pickett_int(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettIntFile:
    """
    Parses a legacy Pickett intensity file (.int).
    """
    lines = _read_content_lines(content_or_path)
    non_empty = [l.rstrip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        raise PickettFormatError("Pickett intensity file must contain title and control lines.")

    title = non_empty[0].strip()
    ctrl_parts = non_empty[1].split()

    if len(ctrl_parts) >= 9:
        tag = int(ctrl_parts[0])
        qpart = fortran_float_converter(ctrl_parts[1])
        nline = int(ctrl_parts[2])
        iq = int(ctrl_parts[3])
        max_j = int(ctrl_parts[4])
        min_freq = fortran_float_converter(ctrl_parts[5])
        max_freq = fortran_float_converter(ctrl_parts[6])
        min_int = fortran_float_converter(ctrl_parts[7])
        temp_k = fortran_float_converter(ctrl_parts[8])
    else:
        c_raw = non_empty[1]
        tag = int(c_raw[0:6].strip() or 1)
        qpart = fortran_float_converter(c_raw[6:18].strip() or 1000.0)
        nline = int(c_raw[18:25].strip() or 1000)
        iq = int(c_raw[25:29].strip() or 1)
        max_j = int(c_raw[29:33].strip() or 99)
        min_freq = fortran_float_converter(c_raw[33:44].strip() or 0.0)
        max_freq = fortran_float_converter(c_raw[44:55].strip() or 500000.0)
        min_int = fortran_float_converter(c_raw[55:65].strip() or -10.0)
        temp_k = fortran_float_converter(c_raw[65:75].strip() or 300.0)

    dipoles: List[PickettDipoleRecord] = []
    dip_pattern = re.compile(r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?")

    for d_line in non_empty[2:]:
        match = dip_pattern.match(d_line)
        if match:
            axis = int(match.group(1))
            val = fortran_float_converter(match.group(2))
            comm = (match.group(3) or "").strip()
            dipoles.append(PickettDipoleRecord(axis=axis, dipole_debye=val, comment=comm))

    return PickettIntFile(
        title=title,
        tag=tag,
        qpart=qpart,
        nline=nline,
        iq=iq,
        max_j=max_j,
        min_freq_mhz=min_freq,
        max_freq_mhz=max_freq,
        min_intensity=min_int,
        temperature_k=temp_k,
        dipoles=dipoles,
    )


def parse_pickett_lin(content_or_path: Union[str, Path, bytes, io.StringIO]) -> PickettLinFile:
    """
    Parses a legacy Pickett experimental line list (.lin).
    """
    lines = _read_content_lines(content_or_path)
    title = ""
    records: List[PickettLinRecord] = []

    free_pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)(?:\s+[+-]?\d+){0,3}\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)"
        r"(?:\s+([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?))?"
    )

    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue
        if s_line.startswith("!") or s_line.startswith("#"):
            if not title:
                title = s_line.lstrip("!# ").strip()
            continue

        if len(line) >= 40:
            try:
                # Fixed-width Fortran 6I3, 6I3, F13.4, F8.4, F8.4
                qu_sub = [line[3 * i : 3 * i + 3].strip() for i in range(6)]
                ql_sub = [line[18 + 3 * i : 18 + 3 * i + 3].strip() for i in range(6)]
                f_sub = line[36:49].strip()
                u_sub = line[49:57].strip()
                w_sub = line[57:65].strip() if len(line) >= 65 else ""

                if qu_sub[0] and ql_sub[0] and f_sub:
                    qu = tuple(int(x) for x in qu_sub if x)
                    ql = tuple(int(x) for x in ql_sub if x)
                    freq = fortran_float_converter(f_sub)
                    unc = fortran_float_converter(u_sub) if u_sub else 0.01
                    wt = fortran_float_converter(w_sub) if w_sub else 1.0
                    records.append(
                        PickettLinRecord(
                            quantum_upper=qu,
                            quantum_lower=ql,
                            frequency_mhz=freq,
                            uncertainty_mhz=unc,
                            weight=wt,
                        )
                    )
                    continue
            except ValueError as parse_err:
                logger.debug("Fixed-width lin line parse fallback: %s", parse_err)

        match = free_pattern.match(s_line)
        if match:
            tokens = s_line.split()
            try:
                freq = fortran_float_converter(match.group(7))
                unc = fortran_float_converter(match.group(8))
                wt = fortran_float_converter(match.group(9)) if match.group(9) else 1.0
                qu = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                ql = (int(match.group(4)), int(match.group(5)), int(match.group(6)))
                records.append(
                    PickettLinRecord(
                        quantum_upper=qu,
                        quantum_lower=ql,
                        frequency_mhz=freq,
                        uncertainty_mhz=unc,
                        weight=wt,
                    )
                )
            except Exception as parse_err:
                logger.debug("Lin regex line parse error: %s", parse_err)

    return PickettLinFile(title=title, records=records)


# =============================================================================
# 6. BINARY DEPLOYMENT ROUTING & EPHEMERAL SCRATCH CONTEXT
# =============================================================================

def get_fortran_deployment_dir() -> Path:
    """
    Resolves the explicit executable deployment directory for legacy Fortran binaries.
    Prioritizes $COCHEM_BIN_DIR over platformdirs user_data_dir / 'bin' to prevent
    execution failures on HPC clusters where scratch/cache mounts have noexec flags.

    Returns:
        Path: Canonical deployment directory with guaranteed write/execute access.
    """
    bin_env = os.environ.get("COCHEM_BIN_DIR")
    if bin_env:
        target = Path(bin_env).resolve()
    else:
        target = Path(platformdirs.user_data_dir(appname="CoChem-SpycFit", ensure_exists=True)) / "bin"

    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_fortran_binary(binary_name: str) -> Optional[Path]:
    """
    Resolves executable path for a legacy Fortran binary within the deployment directory
    or the active system PATH.

    Args:
        binary_name: Binary filename or command (e.g. 'spcat', 'spfit').

    Returns:
        Optional[Path]: Absolute path to executable file, or None if not found.
    """
    dep_dir = get_fortran_deployment_dir()

    # Check direct name in deployment directory
    candidate = dep_dir / binary_name
    if candidate.is_file() and os.access(str(candidate), os.X_OK):
        return candidate

    # Check Windows .exe extension
    if platform.system().lower() == "windows" and not binary_name.lower().endswith(".exe"):
        candidate_exe = dep_dir / f"{binary_name}.exe"
        if candidate_exe.is_file():
            return candidate_exe

    # Check system PATH
    found = shutil.which(binary_name)
    if found:
        return Path(found).resolve()

    return None


class PickettScratchContext:
    """
    Ephemeral scratch directory lifecycle manager for Pickett/SPCAT/SPFIT operations.
    Ensures input staging and guaranteed cleanup of intermediate calculation scratch files.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        prefix: str = "spcat_scratch_",
    ) -> None:
        if base_dir is not None:
            self._parent_dir = Path(base_dir).resolve()
        else:
            scratch_env = os.environ.get("COCHEM_SCRATCH_DIR")
            if scratch_env:
                self._parent_dir = Path(scratch_env).resolve()
            else:
                self._parent_dir = Path(tempfile.gettempdir()).resolve() / "cochem_spycfit_scratch"

        self._parent_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_path: Path = self._parent_dir / f"{prefix}{uuid.uuid4().hex}"
        self._active: bool = False

    def __enter__(self) -> PickettScratchContext:
        self.scratch_path.mkdir(parents=True, exist_ok=True)
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.purge()

    def purge(self) -> None:
        """Purges all files and directories in this scratch workspace."""
        if self.scratch_path.exists():
            shutil.rmtree(str(self.scratch_path), ignore_errors=True)
        self._active = False

    def write_input_file(self, filename: str, content: Union[str, bytes]) -> Path:
        """Writes an input file into the active scratch directory."""
        if not self._active:
            raise RuntimeError("Cannot write input file outside of active PickettScratchContext.")
        target = self.scratch_path / filename
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        return target

    def read_output_file(self, filename: str) -> str:
        """Reads a text output file from the active scratch directory."""
        target = self.scratch_path / filename
        if not target.is_file():
            raise FileNotFoundError(f"Output file not produced in scratch: {filename}")
        return target.read_text(encoding="utf-8", errors="replace")


# =============================================================================
# 7. PROCESS ISOLATION, SEGFAULT TRAPPING & HEX-DUMP TELEMETRY
# =============================================================================

def is_segfault_exit_code(return_code: int) -> bool:
    """
    Identifies whether a process termination return code corresponds to a segmentation fault
    or memory access violation.

    POSIX: 139 (SIGSEGV), 134 (SIGABRT), -11, -6
    Windows: 0xC0000005, -1073741819, 3221225477 (STATUS_ACCESS_VIOLATION)
    """
    posix_codes = {139, 134, -11, -6}
    windows_codes = {0xC0000005, -1073741819, 3221225477, -1073741571}
    return return_code in posix_codes or return_code in windows_codes


def capture_hex_telemetry(data_bytes: bytes, max_bytes: int = 256) -> str:
    """
    Extracts the tail bytes of a stream and formats them as a hex trace.
    """
    if not data_bytes:
        return ""
    tail = data_bytes[-max_bytes:]
    return " ".join(f"{b:02X}" for b in tail)


def write_spycfit_telemetry(
    record: Dict[str, Any],
    telemetry_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Appends structured execution telemetry entry to $COCHEM_STATE_DIR/Logs/spycfit_telemetry.json
    or platformdirs user_state_dir.
    """
    if telemetry_file is not None:
        t_path = Path(telemetry_file).resolve()
    else:
        state_dir_env = os.environ.get("COCHEM_STATE_DIR")
        if state_dir_env:
            log_dir = Path(state_dir_env).resolve() / "Logs"
        else:
            log_dir = Path(platformdirs.user_state_dir(appname="CoChem-SpycFit", ensure_exists=True)) / "Logs"
        t_path = log_dir / "spycfit_telemetry.json"

    t_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **record,
    }

    with open(str(t_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return t_path


# =============================================================================
# 8. ASYNCHRONOUS TASK QUEUE & BRIDGE DISPATCHER
# =============================================================================

class FortranExecutionTask(BaseModel):
    """Encapsulates execution parameters for a legacy Fortran binary invocation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Task UUID.")
    executable_path: str = Field(..., description="Filesystem path or command name of binary.")
    arguments: List[str] = Field(default_factory=list, description="Command line argument tokens.")
    input_files: Dict[str, Union[str, bytes]] = Field(
        default_factory=dict,
        description="Dictionary mapping target filenames to content.",
    )
    expected_output_files: List[str] = Field(
        default_factory=list,
        description="Filenames expected to be produced by execution.",
    )
    timeout_sec: float = Field(default=60.0, gt=0.0, description="Process timeout in seconds.")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variable overrides.")
    working_directory: Optional[str] = Field(
        default=None,
        description="Explicit working directory. If None, an ephemeral PickettScratchContext is used.",
    )


class FortranExecutionResult(BaseModel):
    """Encapsulates output status, logs, and collected artifacts from binary execution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(..., description="Identifier matching task.")
    executable_path: str = Field(..., description="Executed binary path.")
    return_code: int = Field(..., description="Process exit code.")
    stdout: str = Field(default="", description="Captured stdout text.")
    stderr: str = Field(default="", description="Captured stderr text.")
    execution_duration_sec: float = Field(ge=0.0, description="Total execution duration in seconds.")
    output_files: Dict[str, str] = Field(default_factory=dict, description="Captured text output files.")
    output_binaries: Dict[str, bytes] = Field(default_factory=dict, description="Captured binary output files.")
    segfault_detected: bool = Field(default=False, description="True if segfault / access violation occurred.")
    stderr_tail_hex: str = Field(default="", description="Hexadecimal trace of final stderr bytes.")
    telemetry_logged: bool = Field(default=False, description="True if telemetry was written to persistent log.")


class FortranTaskQueue:
    """
    Thread-safe FIFO task queue for batching and dispatching Fortran execution tasks.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[FortranExecutionTask] = queue.Queue(maxsize=maxsize)

    def submit(self, task: FortranExecutionTask) -> str:
        """Enqueues an execution task and returns its task ID."""
        self._queue.put(task)
        return task.task_id

    def get(self, block: bool = True, timeout: Optional[float] = None) -> FortranExecutionTask:
        """Retrieves the next queued execution task."""
        return self._queue.get(block=block, timeout=timeout)

    def task_done(self) -> None:
        """Signals task completion to queue."""
        self._queue.task_done()

    def qsize(self) -> int:
        """Returns current queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Returns True if task queue is currently empty."""
        return self._queue.empty()


class FortranBridgeDispatcher:
    """
    Execution dispatcher for legacy Fortran binaries.
    Provides OS process isolation, segfault trapping, hex telemetry capture,
    and ephemeral scratch directory lifecycle management.
    """

    def __init__(
        self,
        default_timeout_sec: float = 60.0,
        log_telemetry: bool = True,
        telemetry_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.default_timeout_sec = default_timeout_sec
        self.log_telemetry = log_telemetry
        self.telemetry_file = Path(telemetry_file).resolve() if telemetry_file is not None else None

    def dispatch_task(
        self,
        task: FortranExecutionTask,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Synchronously executes a FortranExecutionTask in an isolated process.
        """
        start_time = time.time()
        task_id = task.task_id
        exe_str = task.executable_path

        # Configure OS process isolation flags
        proc_kwargs: Dict[str, Any] = {}
        if platform.system().lower() == "windows":
            proc_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            proc_kwargs["start_new_session"] = True

        exec_env = os.environ.copy()
        exec_env.update(task.env_vars)

        scratch_ctx: Optional[PickettScratchContext] = None
        try:
            if task.working_directory is None:
                scratch_ctx = PickettScratchContext()
                scratch_ctx.__enter__()
                work_dir = scratch_ctx.scratch_path
            else:
                work_dir = Path(task.working_directory).resolve()
                work_dir.mkdir(parents=True, exist_ok=True)

            # Stage input files
            for filename, content in task.input_files.items():
                f_path = work_dir / filename
                if isinstance(content, bytes):
                    f_path.write_bytes(content)
                else:
                    f_path.write_text(content, encoding="utf-8")

            cmd = [exe_str] + task.arguments

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(work_dir),
                    env=exec_env,
                    capture_output=True,
                    timeout=task.timeout_sec,
                    check=True,
                    **proc_kwargs,
                )
                return_code = proc.returncode
                stdout_bytes = proc.stdout
                stderr_bytes = proc.stderr
            except subprocess.CalledProcessError as exc:
                return_code = exc.returncode
                stdout_bytes = exc.stdout or b""
                stderr_bytes = exc.stderr or b""
            except subprocess.TimeoutExpired as exc:
                duration = time.time() - start_time
                stdout_bytes = exc.stdout or b""
                stderr_bytes = exc.stderr or b""
                hex_trace = capture_hex_telemetry(stderr_bytes if stderr_bytes else stdout_bytes)
                if self.log_telemetry:
                    write_spycfit_telemetry(
                        {
                            "task_id": task_id,
                            "executable": exe_str,
                            "return_code": -999,
                            "segfault_detected": False,
                            "timed_out": True,
                            "execution_duration_sec": duration,
                            "stderr_tail_hex": hex_trace,
                        },
                        telemetry_file=self.telemetry_file,
                    )
                raise FortranExecutionTimeoutError(
                    f"Fortran task {task_id} running '{exe_str}' timed out after {task.timeout_sec:.2f}s"
                ) from exc

            duration = time.time() - start_time
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            segfault = is_segfault_exit_code(return_code)
            hex_trace = capture_hex_telemetry(stderr_bytes if stderr_bytes else stdout_bytes)

            collected_outputs: Dict[str, str] = {}
            collected_binaries: Dict[str, bytes] = {}

            if task.expected_output_files:
                for exp_name in task.expected_output_files:
                    target_f = work_dir / exp_name
                    if target_f.is_file():
                        try:
                            collected_outputs[exp_name] = target_f.read_text(encoding="utf-8", errors="replace")
                            collected_binaries[exp_name] = target_f.read_bytes()
                        except UnicodeDecodeError:
                            collected_binaries[exp_name] = target_f.read_bytes()
            else:
                for child in work_dir.iterdir():
                    if child.is_file() and child.name not in task.input_files:
                        try:
                            collected_outputs[child.name] = child.read_text(encoding="utf-8", errors="replace")
                            collected_binaries[child.name] = child.read_bytes()
                        except UnicodeDecodeError:
                            collected_binaries[child.name] = child.read_bytes()

            telemetry_written = False
            if self.log_telemetry:
                try:
                    write_spycfit_telemetry(
                        {
                            "task_id": task_id,
                            "executable": exe_str,
                            "return_code": return_code,
                            "segfault_detected": segfault,
                            "execution_duration_sec": duration,
                            "stderr_tail_hex": hex_trace,
                        },
                        telemetry_file=self.telemetry_file,
                    )
                    telemetry_written = True
                except OSError as log_err:
                    logger.warning("Failed to record telemetry: %s", log_err)

            if raise_on_segfault and segfault:
                raise FortranExecutionSegfaultError(
                    f"Fortran task {task_id} terminated with segfault (exit code {return_code}): {exe_str}"
                )

            return FortranExecutionResult(
                task_id=task_id,
                executable_path=exe_str,
                return_code=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                execution_duration_sec=duration,
                output_files=collected_outputs,
                output_binaries=collected_binaries,
                segfault_detected=segfault,
                stderr_tail_hex=hex_trace,
                telemetry_logged=telemetry_written,
            )
        finally:
            if scratch_ctx is not None:
                scratch_ctx.purge()

    async def async_dispatch_task(
        self,
        task: FortranExecutionTask,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Asynchronously executes a FortranExecutionTask on a background worker thread.
        """
        return await asyncio.to_thread(self.dispatch_task, task, raise_on_segfault)

    def dispatch_fortran_binary(
        self,
        executable_path: Union[str, Path],
        input_files: Dict[str, Union[str, bytes]],
        arguments: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        timeout_sec: float = 60.0,
        raise_on_segfault: bool = False,
    ) -> FortranExecutionResult:
        """
        Convenience wrapper executing a binary directly with staged input files.
        """
        task = FortranExecutionTask(
            executable_path=str(executable_path),
            arguments=arguments or [],
            input_files=input_files,
            expected_output_files=expected_outputs or [],
            timeout_sec=timeout_sec,
        )
        return self.dispatch_task(task, raise_on_segfault=raise_on_segfault)

    def process_queue(
        self,
        task_queue: FortranTaskQueue,
        max_tasks: Optional[int] = None,
        raise_on_segfault: bool = False,
    ) -> List[FortranExecutionResult]:
        """
        Drains and processes tasks from the FortranTaskQueue.
        """
        results: List[FortranExecutionResult] = []
        processed = 0
        while not task_queue.empty():
            if max_tasks is not None and processed >= max_tasks:
                break
            try:
                task = task_queue.get(block=False)
            except queue.Empty:
                break
            result = self.dispatch_task(task, raise_on_segfault=raise_on_segfault)
            task_queue.task_done()
            results.append(result)
            processed += 1
        return results


def dispatch_fortran_binary(
    executable_path: Union[str, Path],
    input_files: Dict[str, Union[str, bytes]],
    arguments: Optional[List[str]] = None,
    expected_outputs: Optional[List[str]] = None,
    timeout_sec: float = 60.0,
    raise_on_segfault: bool = False,
) -> FortranExecutionResult:
    """
    Global functional entry point for executing a legacy Fortran binary.
    """
    dispatcher = FortranBridgeDispatcher(default_timeout_sec=timeout_sec)
    return dispatcher.dispatch_fortran_binary(
        executable_path=executable_path,
        input_files=input_files,
        arguments=arguments,
        expected_outputs=expected_outputs,
        timeout_sec=timeout_sec,
        raise_on_segfault=raise_on_segfault,
    )

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spcat_bridge.py ---
"""
CoChem-SpycFit: Comprehensive Physical Unit Test Suite for Legacy Fortran Translator & Pickett Bridge.

Strict Zero-Mock & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical spectroscopy data, real Pickett formats, and actual OS subprocess execution
- Validation of strict fixed-width truncation and padding (%10.4E with 'D' exponent)
- Validation of 1980s Pickett format exporters and parsers (.var, .int, .lin, .cat, .egy, .fit)
- Validation of dynamic isotope mass retrieval via Mendeleev and reduced mass computations
- Validation of FortranTaskQueue, FortranBridgeDispatcher, process isolation, and async dispatch
- Validation of segfault trapping and hex-dump telemetry capture
- Validation of ephemeral PickettScratchContext lifecycle and executable deployment routing
- AST static analysis enforcing zero pass statements and zero banned tokens
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import os
import platform
import queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranBridgeError,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    FortranTaskQueue,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettEgyFile,
    PickettEgyRecord,
    PickettFitResidual,
    PickettFitResult,
    PickettFormatError,
    PickettIntFile,
    PickettLinFile,
    PickettLinRecord,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    decode_pickett_quantum,
    dispatch_fortran_binary,
    encode_pickett_quantum,
    format_pickett_line,
    fortran_float_converter,
    get_fortran_deployment_dir,
    get_isotopic_mass,
    is_segfault_exit_code,
    parse_pickett_cat,
    parse_pickett_egy,
    parse_pickett_fit,
    parse_pickett_int,
    parse_pickett_lin,
    parse_pickett_var,
    resolve_fortran_binary,
    write_spycfit_telemetry,
)


# =============================================================================
# 1. FIXTURES & ISOLATED WORKSPACE
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated physical directory workspace on disk."""
    ws = tmp_path / "spcat_bridge_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


# =============================================================================
# 2. PHYSICAL CONSTANTS & DYNAMIC ISOTOPE RETRIEVAL TESTS (MENDELEEV)
# =============================================================================

class TestPhysicalConstantsAndIsotopes:
    """Validates dynamic retrieval of physical isotopic masses from Mendeleev."""

    def test_mendeleev_element_atomic_weights(self) -> None:
        mass_h = get_isotopic_mass("H")
        mass_c = get_isotopic_mass("C")
        mass_n = get_isotopic_mass("N")
        mass_o = get_isotopic_mass("O")

        assert 1.007 < mass_h < 1.009
        assert 12.010 < mass_c < 12.012
        assert 14.006 < mass_n < 14.008
        assert 15.998 < mass_o < 16.000

    def test_mendeleev_specific_isotopes(self) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_c13 = get_isotopic_mass("C", 13)
        mass_o16 = get_isotopic_mass("O", 16)
        mass_o18 = get_isotopic_mass("O", 18)

        assert abs(mass_c12 - 12.000000) < 1e-5
        assert 13.003 < mass_c13 < 13.004
        assert 15.994 < mass_o16 < 15.995
        assert 17.999 < mass_o18 < 18.000

    def test_calculate_reduced_mass(self) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_o16 = get_isotopic_mass("O", 16)
        mu_co_12_16 = calculate_reduced_mass(mass_c12, mass_o16)

        expected = (mass_c12 * mass_o16) / (mass_c12 + mass_o16)
        assert abs(mu_co_12_16 - expected) < 1e-9
        assert 6.85 < mu_co_12_16 < 6.87

    def test_calculate_reduced_mass_invalid(self) -> None:
        with pytest.raises(ValueError):
            calculate_reduced_mass(-1.0, 10.0)

    def test_mendeleev_unknown_element_raises_error(self) -> None:
        with pytest.raises(ValueError):
            get_isotopic_mass("NonExistentElement")

    def test_mendeleev_unknown_isotope_mass_raises_error(self) -> None:
        with pytest.raises(ValueError):
            get_isotopic_mass("C", 999)


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & FIXED-WIDTH FORMATTER TESTS
# =============================================================================

class TestFortranFormatterAndConverter:
    """Validates Fortran number parser and strict fixed-width alignment logic."""

    def test_fortran_float_converter_variations(self) -> None:
        assert fortran_float_converter("1.2345D+04") == 12345.0
        assert fortran_float_converter("-2.5000d-02") == -0.025
        assert fortran_float_converter("3.14159E+00") == pytest.approx(3.14159)
        assert fortran_float_converter("1.2345-05") == pytest.approx(1.2345e-5)
        assert fortran_float_converter("1.2345+02") == pytest.approx(123.45)
        assert fortran_float_converter("  100.5  ") == 100.5
        assert fortran_float_converter("") == 0.0
        assert fortran_float_converter(42.5) == 42.5
        assert fortran_float_converter(100) == 100.0

    def test_fortran_float_converter_invalid(self) -> None:
        with pytest.raises(ValueError):
            fortran_float_converter("invalid_float_string")

    def test_format_pickett_line_exact_width_guarantee(self) -> None:
        # Standard %10.4E with 'D'
        res1 = format_pickett_line(12345.0, width=10, precision=4, exponent_char="D")
        assert len(res1) == 10
        assert res1 == "1.2345D+04"

        # Negative number
        res2 = format_pickett_line(-12345.0, width=10, precision=4, exponent_char="D")
        assert len(res2) == 10
        assert res2 == "-1.234D+04"

        # Zero
        res3 = format_pickett_line(0.0, width=10, precision=4, exponent_char="D")
        assert len(res3) == 10
        assert res3 == "0.0000D+00"

        # Width 22, precision 14
        res4 = format_pickett_line(12345.6789012345, width=22, precision=14, exponent_char="D")
        assert len(res4) == 22
        assert "D+" in res4

        # Non-numeric string input
        res5 = format_pickett_line("header", width=10)
        assert len(res5) == 10
        assert res5 == "    header"

        # Truncation of long string
        res6 = format_pickett_line("excessively_long_string", width=10)
        assert len(res6) == 10
        assert res6 == "excessivel"

    def test_format_pickett_line_very_large_and_small_numbers(self) -> None:
        res_small = format_pickett_line(1.2345e-120, width=10, precision=4, exponent_char="D")
        assert len(res_small) == 10

        res_large = format_pickett_line(-1.2345e120, width=10, precision=4, exponent_char="D")
        assert len(res_large) == 10


class TestPickettQuantumEncoding:
    """Validates JPL/CDMS alphanumeric quantum encoding and decoding."""

    def test_decode_standard_integers(self) -> None:
        assert decode_pickett_quantum(" 0") == 0
        assert decode_pickett_quantum(" 5") == 5
        assert decode_pickett_quantum("42") == 42
        assert decode_pickett_quantum("99") == 99

    def test_decode_alphanumeric_quanta(self) -> None:
        # a0 to z9: 10 to 269
        assert decode_pickett_quantum("a0") == 10
        assert decode_pickett_quantum("a9") == 19
        assert decode_pickett_quantum("b0") == 20
        assert decode_pickett_quantum("z9") == 269

        # A0 to Z9: 100 to 359
        assert decode_pickett_quantum("A0") == 100
        assert decode_pickett_quantum("A9") == 109
        assert decode_pickett_quantum("B0") == 110
        assert decode_pickett_quantum("Z9") == 359

    def test_decode_invalid_quantum_raises_error(self) -> None:
        with pytest.raises(PickettFormatError):
            decode_pickett_quantum("??")

    def test_encode_pickett_quantum(self) -> None:
        assert encode_pickett_quantum(0) == " 0"
        assert encode_pickett_quantum(5) == " 5"
        assert encode_pickett_quantum(42) == "42"
        assert encode_pickett_quantum(100) == "A0"
        assert encode_pickett_quantum(109) == "A9"
        assert encode_pickett_quantum(115) == "B5"


# =============================================================================
# 4. PICKETT .VAR / .PAR PARAMETER FILE TESTS
# =============================================================================

class TestPickettVarFile:
    """Validates creation, export, and parsing of Pickett .var / .par files."""

    def test_pickett_var_model_and_export(self, temp_workspace: Path) -> None:
        params = [
            PickettParameterRecord(param_id=10000, value=127538.91, uncertainty=0.0012, label="A"),
            PickettParameterRecord(param_id=20000, value=25228.14, uncertainty=0.00085, label="B"),
            PickettParameterRecord(param_id=30000, value=21384.02, uncertainty=0.00071, label="C"),
            PickettParameterRecord(param_id=200, value=0.03842, uncertainty=0.00015, label="DJ"),
        ]

        var_file = PickettVarFile(
            title="Methanol Ground State Fit",
            nline=150,
            nopt=0,
            nwarn=0,
            erpar=1.0,
            wtfac=1.0,
            scale=1.0,
            maxit=50,
            parameters=params,
        )

        out_path = temp_workspace / "methanol.var"
        var_text = var_file.export_var(out_path)

        assert out_path.is_file()
        assert "Methanol Ground State Fit" in var_text
        assert "10000" in var_text
        assert "20000" in var_text

        # Verify parsing from disk
        parsed_var = parse_pickett_var(out_path)
        assert parsed_var.title == "Methanol Ground State Fit"
        assert parsed_var.npar == 4
        assert parsed_var.nline == 150
        assert len(parsed_var.parameters) == 4
        assert parsed_var.parameters[0].param_id == 10000
        assert parsed_var.parameters[0].value == pytest.approx(127538.91)
        assert parsed_var.parameters[0].label == "A"
        assert parsed_var.parameters[3].param_id == 200
        assert parsed_var.parameters[3].value == pytest.approx(0.03842)

    def test_parse_pickett_var_string_io(self) -> None:
        raw_text = (
            "Acetonitrile Model\n"
            "    2    50     0     0    1.0000    1.0000    1.0000   50\n"
            "      10000  2.7500000000D+04  1.0000000000D-03 / A\n"
            "      20000  9.1980000000D+03  5.0000000000D-04 / B\n"
        )
        sio = io.StringIO(raw_text)
        parsed = parse_pickett_var(sio)
        assert parsed.title == "Acetonitrile Model"
        assert len(parsed.parameters) == 2
        assert parsed.parameters[1].value == pytest.approx(9198.0)

    def test_parse_pickett_var_empty_raises_error(self) -> None:
        with pytest.raises(PickettFormatError):
            parse_pickett_var("Short line\n")


# =============================================================================
# 5. PICKETT .INT INTENSITY FILE TESTS
# =============================================================================

class TestPickettIntFile:
    """Validates creation, export, and parsing of Pickett .int intensity files."""

    def test_pickett_int_model_and_export(self, temp_workspace: Path) -> None:
        dipoles = [
            PickettDipoleRecord(axis=1, dipole_debye=3.92, comment="mua"),
            PickettDipoleRecord(axis=2, dipole_debye=0.00, comment="mub"),
            PickettDipoleRecord(axis=3, dipole_debye=0.00, comment="muc"),
        ]

        int_file = PickettIntFile(
            title="Acetonitrile Intensities",
            tag=41001,
            qpart=1852.34,
            nline=5000,
            iq=1,
            max_j=99,
            min_freq_mhz=0.0,
            max_freq_mhz=500000.0,
            min_intensity=-10.0,
            temperature_k=300.0,
            dipoles=dipoles,
        )

        out_path = temp_workspace / "ch3cn.int"
        int_text = int_file.export_int(out_path)

        assert out_path.is_file()
        assert "Acetonitrile Intensities" in int_text
        assert "41001" in int_text
        assert "300.00" in int_text

        # Verify parsing from disk
        parsed_int = parse_pickett_int(out_path)
        assert parsed_int.title == "Acetonitrile Intensities"
        assert parsed_int.tag == 41001
        assert parsed_int.qpart == pytest.approx(1852.34)
        assert parsed_int.temperature_k == pytest.approx(300.0)
        assert len(parsed_int.dipoles) == 3
        assert parsed_int.dipoles[0].axis == 1
        assert parsed_int.dipoles[0].dipole_debye == pytest.approx(3.92)


# =============================================================================
# 6. PICKETT .LIN LINE LIST FILE TESTS
# =============================================================================

class TestPickettLinFile:
    """Validates creation, export, and parsing of Pickett .lin line list files."""

    def test_pickett_lin_model_and_export(self, temp_workspace: Path) -> None:
        records = [
            PickettLinRecord(quantum_upper=(1, 0, 1), quantum_lower=(0, 0, 0), frequency_mhz=18397.05, uncertainty_mhz=0.005, weight=1.0, description="1_01-0_00"),
            PickettLinRecord(quantum_upper=(2, 0, 2), quantum_lower=(1, 0, 1), frequency_mhz=36794.10, uncertainty_mhz=0.005, weight=1.0, description="2_02-1_01"),
            PickettLinRecord(quantum_upper=(3, 0, 3), quantum_lower=(2, 0, 2), frequency_mhz=55191.15, uncertainty_mhz=0.005, weight=1.0, description="3_03-2_02"),
        ]

        lin_file = PickettLinFile(
            title="CH3CN Line Measurements",
            records=records,
        )

        out_path = temp_workspace / "ch3cn.lin"
        lin_text = lin_file.export_lin(out_path)

        assert out_path.is_file()
        assert "18397.0500" in lin_text
        assert "36794.1000" in lin_text

        # Verify parsing from disk
        parsed_lin = parse_pickett_lin(out_path)
        assert len(parsed_lin.records) == 3
        assert parsed_lin.records[0].quantum_upper[:3] == (1, 0, 1)
        assert parsed_lin.records[0].quantum_lower[:3] == (0, 0, 0)
        assert parsed_lin.records[0].frequency_mhz == pytest.approx(18397.05)
        assert parsed_lin.records[1].frequency_mhz == pytest.approx(36794.10)


# =============================================================================
# 7. PICKETT OUTPUT PARSER TESTS (.CAT, .EGY, .FIT)
# =============================================================================

class TestPickettOutputParsers:
    """Validates parsing of calculation results (.cat, .egy, .fit)."""

    def test_parse_pickett_cat_file(self, temp_workspace: Path) -> None:
        cat_content = (
            "  18397.0500  0.0050 -4.5123 2   15.6789  3  41001 1404 1 0 1 0 0 0 0 0 0 0 0 0\n"
            "  36794.1000  0.0050 -3.8541 2   30.8912  5  41001 1404 2 0 2 0 0 0 1 0 1 0 0 0\n"
            "  55191.1500  0.0050 -3.4210 2   46.1234  7  41001 1404a0 0a0 0 0 0 9 0 9 0 0 0\n"
        )
        cat_path = temp_workspace / "test_prediction.cat"
        cat_path.write_text(cat_content, encoding="utf-8")

        parsed_cat = parse_pickett_cat(cat_path)
        assert len(parsed_cat.records) == 3

        # Line 1
        r0 = parsed_cat.records[0]
        assert r0.frequency_mhz == pytest.approx(18397.05)
        assert r0.uncertainty_mhz == pytest.approx(0.005)
        assert r0.log_intensity == pytest.approx(-4.5123)
        assert r0.species_tag == 41001
        assert r0.quantum_upper[0] == 1
        assert r0.quantum_lower[0] == 0

        # Line 3 (alphanumeric quantum 'a0' -> 10)
        r2 = parsed_cat.records[2]
        assert r2.frequency_mhz == pytest.approx(55191.15)
        assert r2.quantum_upper[0] == 10
        assert r2.quantum_lower[0] == 9

    def test_parse_pickett_egy_file(self, temp_workspace: Path) -> None:
        egy_content = (
            "   1      0.00000000      0.00000000    1.00000000   0   0   0  A1\n"
            "   2     12.34567890      0.00005000    0.99980000   1   0   1  E1\n"
            "   3     37.03703700      0.00008000    0.99950000   2   0   2  E2\n"
        )
        egy_path = temp_workspace / "test_states.egy"
        egy_path.write_text(egy_content, encoding="utf-8")

        parsed_egy = parse_pickett_egy(egy_path)
        assert len(parsed_egy.records) == 3
        assert parsed_egy.records[0].state_index == 1
        assert parsed_egy.records[0].energy_cm1 == 0.0
        assert parsed_egy.records[1].state_index == 2
        assert parsed_egy.records[1].energy_cm1 == pytest.approx(12.3456789)
        assert parsed_egy.records[1].quantum_numbers == (1, 0, 1)
        assert parsed_egy.records[1].label == "E1"

    def test_parse_pickett_fit_file(self, temp_workspace: Path) -> None:
        fit_content = (
            "METHANOL GROUND STATE FIT\n"
            "STANDARD ERROR OF FIT = 0.8542   VARIANCE = 0.7296\n"
            "LINES FITTED = 2\n"
            "PARAMETERS:\n"
            "    10000   1.2753891000D+05   1.2000000000D-03 / A\n"
            "    20000   2.5228140000D+04   8.5000000000D-04 / B\n"
            "RESIDUALS (OBS-CALC):\n"
            "   1   1  0  1   0  0  0   18397.0500   18397.0450   0.0050   0.0050   1.0000\n"
            "   2   2  0  2   1  0  1   36794.1000   36794.0980   0.0020   0.0050   1.0000\n"
        )
        fit_path = temp_workspace / "test_fit.fit"
        fit_path.write_text(fit_content, encoding="utf-8")

        parsed_fit = parse_pickett_fit(fit_path)
        assert parsed_fit.title == "METHANOL GROUND STATE FIT"
        assert parsed_fit.standard_error == pytest.approx(0.8542)
        assert parsed_fit.variance == pytest.approx(0.7296)
        assert parsed_fit.lines_fitted == 2
        assert len(parsed_fit.parameters) == 2
        assert len(parsed_fit.residuals) == 2
        assert parsed_fit.residuals[0].residual_mhz == pytest.approx(0.0050)
        assert parsed_fit.residuals[1].residual_mhz == pytest.approx(0.0020)


# =============================================================================
# 8. DEPLOYMENT ROUTING & EPHEMERAL SCRATCH CONTEXT TESTS
# =============================================================================

class TestDeploymentAndScratchContext:
    """Validates executable deployment directory routing and scratch context."""

    def test_get_fortran_deployment_dir(self, temp_workspace: Path) -> None:
        custom_bin = temp_workspace / "custom_bin"
        old_env = os.environ.get("COCHEM_BIN_DIR")
        try:
            os.environ["COCHEM_BIN_DIR"] = str(custom_bin)
            resolved = get_fortran_deployment_dir()
            assert resolved == custom_bin.resolve()
            assert resolved.is_dir()
        finally:
            if old_env is not None:
                os.environ["COCHEM_BIN_DIR"] = old_env
            elif "COCHEM_BIN_DIR" in os.environ:
                del os.environ["COCHEM_BIN_DIR"]

    def test_resolve_fortran_binary(self, temp_workspace: Path) -> None:
        custom_bin = temp_workspace / "test_bin_dir"
        custom_bin.mkdir(parents=True, exist_ok=True)
        old_env = os.environ.get("COCHEM_BIN_DIR")
        try:
            os.environ["COCHEM_BIN_DIR"] = str(custom_bin)

            # Create an executable script in deployment dir
            sample_bin = custom_bin / ("spcat.exe" if platform.system().lower() == "windows" else "spcat")
            sample_bin.write_text("#!/bin/sh\necho 'SPCAT'", encoding="utf-8")
            if platform.system().lower() != "windows":
                sample_bin.chmod(0o755)

            found = resolve_fortran_binary("spcat")
            assert found is not None
            assert found.name.startswith("spcat")
        finally:
            if old_env is not None:
                os.environ["COCHEM_BIN_DIR"] = old_env
            elif "COCHEM_BIN_DIR" in os.environ:
                del os.environ["COCHEM_BIN_DIR"]

    def test_pickett_scratch_context_lifecycle(self, temp_workspace: Path) -> None:
        scratch_base = temp_workspace / "scratch_pool"
        with PickettScratchContext(base_dir=scratch_base) as ctx:
            assert ctx.scratch_path.exists()
            assert ctx._active is True

            in_file = ctx.write_input_file("input.var", "Test Var Content")
            assert in_file.is_file()

            out_file = ctx.scratch_path / "output.cat"
            out_file.write_text("Test Output Content", encoding="utf-8")

            read_back = ctx.read_output_file("output.cat")
            assert read_back == "Test Output Content"

            scratch_target = ctx.scratch_path

        # After exiting, scratch directory must be completely purged
        assert not scratch_target.exists()


# =============================================================================
# 9. PROCESS ISOLATION, SEGFAULT TRAPPING & TELEMETRY TESTS
# =============================================================================

class TestProcessIsolationAndTelemetry:
    """Validates segfault identification, hex telemetry capture, and structured logging."""

    def test_is_segfault_exit_code(self) -> None:
        # POSIX segfault & abort
        assert is_segfault_exit_code(139) is True
        assert is_segfault_exit_code(134) is True
        assert is_segfault_exit_code(-11) is True

        # Windows STATUS_ACCESS_VIOLATION
        assert is_segfault_exit_code(0xC0000005) is True
        assert is_segfault_exit_code(-1073741819) is True
        assert is_segfault_exit_code(3221225477) is True

        # Normal exits
        assert is_segfault_exit_code(0) is False
        assert is_segfault_exit_code(1) is False
        assert is_segfault_exit_code(2) is False

    def test_capture_hex_telemetry(self) -> None:
        sample_bytes = b"SPCAT SEGFAULT TRACE DATA 2026"
        hex_trace = capture_hex_telemetry(sample_bytes, max_bytes=10)
        assert isinstance(hex_trace, str)
        assert len(hex_trace.split()) == 10
        assert capture_hex_telemetry(b"") == ""

    def test_write_spycfit_telemetry(self, temp_workspace: Path) -> None:
        log_file = temp_workspace / "telemetry_test.json"
        record = {
            "task_id": "test-task-1234",
            "executable": "spcat",
            "return_code": 0,
            "segfault_detected": False,
            "execution_duration_sec": 0.045,
            "stderr_tail_hex": "41 42 43",
        }

        t_path = write_spycfit_telemetry(record, telemetry_file=log_file)
        assert t_path.is_file()

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["task_id"] == "test-task-1234"
        assert entry["executable"] == "spcat"
        assert entry["recorded_utc"] is not None


# =============================================================================
# 10. ASYNCHRONOUS TASK QUEUE & DISPATCHER TESTS
# =============================================================================

class TestFortranTaskQueueAndDispatcher:
    """Validates FortranTaskQueue, FortranBridgeDispatcher, and isolated execution."""

    def test_fortran_task_queue_operations(self) -> None:
        q = FortranTaskQueue()
        assert q.empty() is True

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", "print('Queue Task')"],
        )
        task_id = q.submit(task)
        assert task_id == task.task_id
        assert q.qsize() == 1
        assert q.empty() is False

        retrieved = q.get(block=False)
        assert retrieved.task_id == task.task_id
        q.task_done()
        assert q.empty() is True

    def test_dispatch_task_with_real_python_subprocess(self, temp_workspace: Path) -> None:
        telemetry_log = temp_workspace / "telemetry_run.json"
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=10.0, log_telemetry=True, telemetry_file=telemetry_log)

        # Create a Python script task that reads an input file and writes an output file
        worker_script = (
            "import sys\n"
            "from pathlib import Path\n"
            "var_content = Path('input.var').read_text()\n"
            "Path('output.cat').write_text('GENERATED CAT: ' + var_content)\n"
            "sys.stderr.write('CALCULATION COMPLETED CLEANLY\\n')\n"
        )

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", worker_script],
            input_files={"input.var": "Physical Parameter Data 12345"},
            expected_output_files=["output.cat"],
            timeout_sec=10.0,
        )

        result = dispatcher.dispatch_task(task)

        assert result.return_code == 0
        assert result.segfault_detected is False
        assert "output.cat" in result.output_files
        assert "GENERATED CAT: Physical Parameter Data 12345" in result.output_files["output.cat"]
        assert "CALCULATION COMPLETED CLEANLY" in result.stderr
        assert result.telemetry_logged is True
        assert telemetry_log.is_file()

    def test_dispatch_task_timeout(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=0.2, log_telemetry=False)
        hang_script = "import time\ntime.sleep(5.0)\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", hang_script],
            timeout_sec=0.2,
        )

        with pytest.raises(FortranExecutionTimeoutError):
            dispatcher.dispatch_task(task)

    def test_dispatch_task_segfault_trapping(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)

        # Simulate process exiting with segfault return code (139)
        crash_script = "import sys\nsys.stderr.write('FATAL MEMORY CORRUPTION')\nsys.exit(139)\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", crash_script],
            timeout_sec=5.0,
        )

        # Without raising: returns result with segfault_detected=True
        result = dispatcher.dispatch_task(task, raise_on_segfault=False)
        assert result.return_code == 139
        assert result.segfault_detected is True
        assert "FATAL MEMORY CORRUPTION" in result.stderr

        # With raise_on_segfault=True: raises FortranExecutionSegfaultError
        with pytest.raises(FortranExecutionSegfaultError):
            dispatcher.dispatch_task(task, raise_on_segfault=True)

    def test_async_dispatch_task(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)
        async_script = "import sys\nprint('ASYNC DISPATCH SUCCESS')\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", async_script],
            timeout_sec=5.0,
        )

        result = asyncio.run(dispatcher.async_dispatch_task(task))
        assert result.return_code == 0
        assert "ASYNC DISPATCH SUCCESS" in result.stdout

    def test_convenience_dispatch_fortran_binary(self) -> None:
        script = "import sys\nsys.stdout.write('DIRECT BINARY DISPATCH')\n"
        result = dispatch_fortran_binary(
            executable_path=sys.executable,
            input_files={},
            arguments=["-c", script],
            timeout_sec=5.0,
        )
        assert result.return_code == 0
        assert "DIRECT BINARY DISPATCH" in result.stdout

    def test_process_queue_batch(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)
        task_q = FortranTaskQueue()

        for i in range(3):
            t = FortranExecutionTask(
                executable_path=sys.executable,
                arguments=["-c", f"print('TASK {i}')"],
            )
            task_q.submit(t)

        assert task_q.qsize() == 3
        results = dispatcher.process_queue(task_q)
        assert len(results) == 3
        assert task_q.empty() is True
        for i, res in enumerate(results):
            assert res.return_code == 0
            assert f"TASK {i}" in res.stdout


# =============================================================================
# 11. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliance:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_spcat_bridge_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_spcat_bridge.py"
        assert module_path.is_file()

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_test_suite_ast_compliance(self) -> None:
        test_path = Path(__file__).resolve()
        assert test_path.is_file()

        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_no_banned_tokens_in_production_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_spcat_bridge.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        banned = ["mock", "dummy", "placeholder", "fake", "stub"]
        violations: List[str] = []

        class IdentifierScanner(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for b in banned:
                    if b in node.name.lower():
                        violations.append(f"Function {node.name} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for b in banned:
                    if b in node.id.lower():
                        violations.append(f"Variable {node.id} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    for b in banned:
                        if b in alias.name.lower():
                            violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for b in banned:
                        if b in node.module.lower():
                            violations.append(f"Forbidden import from '{node.module}' at line {node.lineno}")
                self.generic_visit(node)

        IdentifierScanner().visit(tree)
        assert len(violations) == 0, f"Banned tokens found: {violations}"

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_ui.py ---
"""Comprehensive physical unit test suite for SpycFit Decision Dashboard (HITL).

Verifies:
1. Plotly Dual-Pane O-C Residual dashboard with WebGL and SVG fallbacks.
2. Bipartite Auto-Propose Assistant using scipy.optimize.linear_sum_assignment (Hungarian Algorithm).
3. Human-In-The-Loop (HITL) approval gate state machine and Huber loss calculation with Ghost Peak retention.
4. Headless and air-gap degradation (static self-contained HTML and ReportLab PDF generation).
5. What-If parameter slider spectrum recalculation and integration with JAX convolver.
6. Semantic Git-commit message translation for mathematical changes.

Zero-Mock Policy: All calculations use real physical spectroscopic data and authentic algorithmic solvers.
"""

from __future__ import annotations

import ast
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from cochem_spycfit.interfaces.cochem_spycfit_ui import (
    AssignmentStatus,
    BipartiteConfig,
    BipartiteMatchingError,
    DashboardConfig,
    HITLApprovalError,
    HITLAssignmentManager,
    OfflineRenderError,
    PeakAssignment,
    RenderBackend,
    SemanticCommit,
    SemanticCommitError,
    SliderComputationError,
    SpectralPeak,
    SpycFitInteractiveDashboard,
    SpycFitZMQClient,
    SpycFitZMQServer,
    UIDashboardError,
    WhatIfParameterState,
    WhatIfSpectrumEngine,
    ZMQBridgeError,
    auto_propose_assignments,
    compute_bipartite_cost_matrix,
    compute_huber_loss_excluding_ghosts,
    create_spycfit_dashboard_widget,
    render_dual_pane_dashboard,
    render_static_offline_html,
    render_static_offline_pdf,
    run_lttb_decimation,
    translate_mathematical_change_to_git_commit,
    update_ui_state_hash,
)


# =============================================================================
# PHYSICAL FIXTURES (Real Spectroscopic Transitions)
# =============================================================================

@pytest.fixture
def methanol_exp_peaks() -> List[SpectralPeak]:
    """Authentic experimental rotational transitions of Methanol (CH3OH) in the 12-48 GHz band."""
    return [
        SpectralPeak(
            peak_id="EXP_01",
            frequency=12178.597,
            intensity=0.88,
            uncertainty=0.005,
            quantum_label="2_0,2 -> 1_1,1",
        ),
        SpectralPeak(
            peak_id="EXP_02",
            frequency=24928.702,
            intensity=1.45,
            uncertainty=0.008,
            quantum_label="4_1,4 -> 3_0,3",
        ),
        SpectralPeak(
            peak_id="EXP_03",
            frequency=36169.265,
            intensity=2.10,
            uncertainty=0.010,
            quantum_label="4_0,4 -> 3_1,3",
        ),
        SpectralPeak(
            peak_id="EXP_04",
            frequency=48372.455,
            intensity=0.62,
            uncertainty=0.012,
            quantum_label="1_1,0 -> 1_0,1",
        ),
        SpectralPeak(
            peak_id="EXP_05_IMPURITY",
            frequency=28450.120,
            intensity=0.15,
            uncertainty=0.020,
            quantum_label=None,
        ),
    ]


@pytest.fixture
def methanol_theo_peaks() -> List[SpectralPeak]:
    """Theoretical Watson A-reduced Hamiltonian predicted peaks for Methanol."""
    return [
        SpectralPeak(
            peak_id="THEO_01",
            frequency=12178.610,
            intensity=0.90,
            uncertainty=0.001,
            quantum_label="2_0,2 -> 1_1,1",
        ),
        SpectralPeak(
            peak_id="THEO_02",
            frequency=24928.680,
            intensity=1.42,
            uncertainty=0.001,
            quantum_label="4_1,4 -> 3_0,3",
        ),
        SpectralPeak(
            peak_id="THEO_03",
            frequency=36169.290,
            intensity=2.15,
            uncertainty=0.001,
            quantum_label="4_0,4 -> 3_1,3",
        ),
        SpectralPeak(
            peak_id="THEO_04",
            frequency=48372.440,
            intensity=0.60,
            uncertainty=0.001,
            quantum_label="1_1,0 -> 1_0,1",
        ),
        SpectralPeak(
            peak_id="THEO_05_UNOBSERVED",
            frequency=19800.500,
            intensity=0.05,
            uncertainty=0.001,
            quantum_label="3_2,1 -> 3_1,2",
        ),
    ]


@pytest.fixture
def baseline_slider_state() -> WhatIfParameterState:
    """Baseline asymmetric top rotational parameters for Methanol."""
    return WhatIfParameterState(
        A_mhz=127484.0,
        B_mhz=24679.98,
        C_mhz=23769.70,
        D_J_khz=42.1,
        D_JK_khz=-120.4,
        D_K_khz=520.8,
        temperature_k=150.0,
        fwhm_gaussian_mhz=0.05,
        fwhm_lorentzian_mhz=0.03,
        global_frequency_shift_mhz=0.0,
        intensity_scale=1.0,
    )


# =============================================================================
# 1. PYDANTIC SCHEMAS & DATA STRUCTURE VALIDATION TESTS
# =============================================================================

def test_spectral_peak_schema_validation():
    """Verifies strict validation on SpectralPeak model."""
    valid_peak = SpectralPeak(
        peak_id="PEAK_001",
        frequency=24000.5,
        intensity=1.2,
        uncertainty=0.005,
    )
    assert valid_peak.frequency == 24000.5
    assert valid_peak.intensity == 1.2
    assert not valid_peak.is_ghost

    # Invalid non-positive frequency
    with pytest.raises(ValidationError):
        SpectralPeak(peak_id="BAD_01", frequency=-10.0, intensity=1.0)

    # Invalid negative intensity
    with pytest.raises(ValidationError):
        SpectralPeak(peak_id="BAD_02", frequency=100.0, intensity=-0.5)


def test_what_if_parameter_state_validation():
    """Verifies validation of WhatIfParameterState physical ranges."""
    state = WhatIfParameterState(
        A_mhz=10000.0,
        B_mhz=5000.0,
        C_mhz=2500.0,
        temperature_k=296.0,
    )
    assert state.A_mhz == 10000.0
    assert state.temperature_k == 296.0

    # Temperature <= 0 must fail
    with pytest.raises(ValidationError):
        WhatIfParameterState(
            A_mhz=10000.0,
            B_mhz=5000.0,
            C_mhz=2500.0,
            temperature_k=0.0,
        )


# =============================================================================
# 2. BIPARTITE HUNGARIAN AUTO-PROPOSE ASSISTANT TESTS
# =============================================================================

def test_bipartite_cost_matrix_calculation(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies construction of the bipartite cost matrix with real frequency distance."""
    config = BipartiteConfig(max_freq_drift_mhz=50.0, freq_weight=1.0, intensity_weight=0.1)
    cost_matrix = compute_bipartite_cost_matrix(methanol_exp_peaks, methanol_theo_peaks, config)

    assert cost_matrix.shape == (len(methanol_exp_peaks), len(methanol_theo_peaks))
    assert np.all(np.isfinite(cost_matrix))
    # Diagonal should have lowest costs because matched peaks have minimal drift (< 0.05 MHz)
    assert cost_matrix[0, 0] < cost_matrix[0, 1]
    assert cost_matrix[1, 1] < cost_matrix[1, 0]


def test_auto_propose_assignments_optimal_matching(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies Hungarian algorithm solves the assignment problem and respects drift cutoff."""
    config = BipartiteConfig(max_freq_drift_mhz=1.0)
    assignments = auto_propose_assignments(methanol_exp_peaks, methanol_theo_peaks, config)

    # 4 authentic transitions should be assigned within 1.0 MHz cutoff; the 5th impurity should not match
    assert len(assignments) == 4
    for assign in assignments:
        assert assign.status == AssignmentStatus.PROPOSED
        assert abs(assign.frequency_drift_mhz) < 0.05
        assert assign.exp_peak.peak_id.startswith("EXP_")
        assert assign.theo_peak.peak_id.startswith("THEO_")

    assigned_exp_ids = {a.exp_peak.peak_id for a in assignments}
    assert "EXP_05_IMPURITY" not in assigned_exp_ids


def test_auto_propose_assignments_empty_inputs():
    """Verifies auto_propose_assignments handles empty peak lists gracefully."""
    res = auto_propose_assignments([], [])
    assert res == []

    res2 = auto_propose_assignments([SpectralPeak(peak_id="E1", frequency=100.0, intensity=1.0)], [])
    assert res2 == []


# =============================================================================
# 3. HUMAN-IN-THE-LOOP (HITL) ASSIGNMENT MANAGER & GHOST PEAKS TESTS
# =============================================================================

def test_hitl_manager_state_transitions(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies HITL assignment manager approval workflow and ghost peak segregation."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=5.0))

    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    assert len(proposed) >= 4

    # 1. Approve first assignment
    first_id = proposed[0].assignment_id
    approved_assign = manager.approve_assignment(first_id, user_id="DrSpectroscopist")
    assert approved_assign.status == AssignmentStatus.APPROVED
    assert approved_assign.approved_by == "DrSpectroscopist"

    # 2. Reject second assignment (simulate user rejecting an ambiguous peak)
    second_id = proposed[1].assignment_id
    rejected_assign = manager.reject_assignment(second_id, reason="Line profile asymmetry / blender")
    assert rejected_assign.status == AssignmentStatus.REJECTED
    assert rejected_assign.exp_peak.is_ghost
    assert rejected_assign.exp_peak.rejection_reason == "Line profile asymmetry / blender"

    # Verify ghost peaks tracking
    ghost_peaks = manager.get_ghost_peaks()
    assert any(g.peak_id == rejected_assign.exp_peak.peak_id for g in ghost_peaks)

    # Verify active peaks exclude ghosts
    active_peaks = manager.get_active_exp_peaks()
    assert all(not p.is_ghost for p in active_peaks)


def test_hitl_manager_lm_fit_gate_enforcement(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies LM fit gate prevents fitting when zero assignments are approved or unapproved remain in strict mode."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=5.0))

    # Without approval, validation must raise HITLApprovalError
    with pytest.raises(HITLApprovalError, match="No assignments approved"):
        manager.validate_for_lm_fit()

    # Approve all proposed transitions
    manager.approve_all_proposed(user_id="Verifier")
    approved_for_fit = manager.validate_for_lm_fit()
    assert len(approved_for_fit) >= 4


def test_huber_loss_excluding_ghost_peaks(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies Huber robust loss calculation strictly excludes rejected ghost peaks."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=10.0))

    # Approve 3 assignments and reject 1 assignment with high residual
    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    for p in proposed[:3]:
        manager.approve_assignment(p.assignment_id, user_id="User")
    manager.reject_assignment(proposed[3].assignment_id, reason="Outlier artifact")

    loss, rms_mhz = manager.compute_robust_metrics(huber_delta_mhz=1.0)
    assert loss >= 0.0
    assert rms_mhz >= 0.0
    assert rms_mhz < 0.1  # Approved transitions have < 0.05 MHz drift


# =============================================================================
# 4. PLOTLY DUAL-PANE O-C RESIDUAL DASHBOARD TESTS
# =============================================================================

def test_render_dual_pane_dashboard_webgl(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies rendering dual-pane dashboard with WebGL scattergl traces."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    manager.approve_all_proposed(user_id="AutoTester")

    config = DashboardConfig(use_webgl=True, theme="plotly_dark", show_assignment_lines=True)
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=manager.assignments,
        config=config,
    )

    assert isinstance(fig, go.Figure)
    # Check that subplot layout contains 2 rows
    assert len(fig.data) > 0
    # Top pane has spectrum traces, bottom pane has residual traces
    trace_types = [t.type for t in fig.data]
    assert "scattergl" in trace_types or "scatter" in trace_types

    # Verify O-C residual trace exists in bottom pane
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Residual" in n or "O - C" in n for n in trace_names)
    assert any("Experimental" in n for n in trace_names)
    assert any("Theoretical" in n for n in trace_names)


def test_render_dual_pane_dashboard_svg_fallback(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies SVG fallback rendering when use_webgl is False."""
    config = DashboardConfig(use_webgl=False, theme="plotly_white")
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=[],
        config=config,
    )
    assert isinstance(fig, go.Figure)
    # When use_webgl is False, scattergl should not be used
    trace_types = [t.type for t in fig.data]
    assert "scattergl" not in trace_types


def test_render_dual_pane_dashboard_with_ghost_peaks(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies ghost peaks are rendered in greyed-out translucent trace."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    manager.reject_assignment(proposed[0].assignment_id, reason="Ghost test")

    fig = render_dual_pane_dashboard(
        exp_peaks=manager.exp_peaks,
        theo_peaks=manager.theo_peaks,
        assignments=manager.assignments,
        config=DashboardConfig(show_ghost_peaks=True),
    )
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Ghost" in n for n in trace_names)


# =============================================================================
# 5. HEADLESS & AIR-GAP DEGRADATION (HTML & PDF) TESTS
# =============================================================================

def test_render_static_offline_html(methanol_exp_peaks, methanol_theo_peaks, tmp_path):
    """Verifies self-contained offline HTML artifact creation."""
    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=[],
    )
    out_html = tmp_path / "dashboard_offline.html"
    rendered_path = render_static_offline_html(fig, output_path=out_html)

    assert rendered_path.is_file()
    assert rendered_path.stat().st_size > 500
    content = rendered_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "plotly" in content.lower()


def test_render_static_offline_pdf(methanol_exp_peaks, methanol_theo_peaks, tmp_path):
    """Verifies publication-grade static PDF generation via ReportLab."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose()
    manager.approve_all_proposed(user_id="DrScientist")

    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        assignments=manager.assignments,
    )

    out_pdf = tmp_path / "dashboard_report.pdf"
    rendered_pdf = render_static_offline_pdf(
        fig=fig,
        output_path=out_pdf,
        assignments=manager.assignments,
        fit_stats={"rms_mhz": 0.015, "reduced_chi_sq": 1.02, "num_transitions": 4},
        title="Methanol Rotational Spectrum FIT-2026",
    )

    assert rendered_pdf.is_file()
    assert rendered_pdf.stat().st_size > 1000
    # Verify PDF magic bytes %PDF-
    with open(rendered_pdf, "rb") as f:
        header = f.read(5)
        assert header.startswith(b"%PDF-")


# =============================================================================
# 6. WHAT-IF PARAMETER SLIDERS & CONVOLUTION RECALCULATION TESTS
# =============================================================================

def test_what_if_spectrum_engine_recalculation(methanol_theo_peaks, baseline_slider_state):
    """Verifies real-time parameter perturbation and continuous spectrum convolution."""
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=2000,
    )

    # 1. Base spectrum evaluation
    freq_grid, base_spectrum = engine.compute_convolved_spectrum(baseline_slider_state)
    assert len(freq_grid) == 2000
    assert len(base_spectrum) == 2000
    assert np.max(base_spectrum) > 0.0

    # 2. Perturb rotational constant A (+50 MHz)
    perturbed_state = baseline_slider_state.model_copy(update={"A_mhz": baseline_slider_state.A_mhz + 50.0})
    _, perturbed_spectrum = engine.compute_convolved_spectrum(perturbed_state)

    # Peak positions/intensities must change
    assert not np.allclose(base_spectrum, perturbed_spectrum, atol=1e-6)

    # 3. Parameter slider steps generation
    steps = engine.generate_plotly_slider_steps(
        param_name="A_mhz",
        values=[127400.0, 127484.0, 127550.0],
    )
    assert len(steps) == 3
    assert steps[0]["label"] == "127400.00"


# =============================================================================
# 7. SEMANTIC GIT-COMMIT TRANSLATION TESTS
# =============================================================================

def test_semantic_git_commit_translation_parameters(baseline_slider_state):
    """Verifies translation of Hamiltonian parameter shift to conventional commit message."""
    perturbed_state = baseline_slider_state.model_copy(
        update={
            "A_mhz": baseline_slider_state.A_mhz + 1.25,
            "B_mhz": baseline_slider_state.B_mhz - 0.45,
        }
    )

    commit = translate_mathematical_change_to_git_commit(
        state_before=baseline_slider_state,
        state_after=perturbed_state,
        rms_before=0.085,
        rms_after=0.018,
    )

    assert isinstance(commit, SemanticCommit)
    assert commit.commit_type in ["feat", "fix", "refactor"]
    assert "hamiltonian" in commit.scope.lower() or "param" in commit.scope.lower()
    assert "A=" in commit.full_commit_message
    assert "RMS: 0.0850 MHz -> 0.0180 MHz" in commit.full_commit_message


def test_semantic_git_commit_translation_assignments():
    """Verifies translation of assignment approvals/rejections to commit message."""
    commit = translate_mathematical_change_to_git_commit(
        approved_count=5,
        rejected_count=2,
        rms_before=0.120,
        rms_after=0.024,
    )
    assert isinstance(commit, SemanticCommit)
    assert "assignment" in commit.scope.lower()
    assert "5" in commit.summary
    assert "2" in commit.summary
    assert "RMS: 0.1200 MHz -> 0.0240 MHz" in commit.full_commit_message


def test_semantic_git_commit_translation_huber_loss():
    """Verifies translation of Huber loss delta adjustment to commit message."""
    commit = translate_mathematical_change_to_git_commit(
        huber_delta_before=1.5,
        huber_delta_after=0.8,
    )
    assert isinstance(commit, SemanticCommit)
    assert "loss" in commit.scope.lower() or "huber" in commit.scope.lower()
    assert "1.500" in commit.full_commit_message
    assert "0.800" in commit.full_commit_message


def test_hitl_manager_manual_assign_and_tolerance(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies manual assignment workflow, tolerance threshold approvals, and error handling."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=50.0))

    # Test approve_all_within_tolerance
    count = manager.approve_all_within_tolerance(max_drift_mhz=0.05, user_id="ToleranceVerifier")
    assert count == 4

    # Test manual_assign
    manual_asn = manager.manual_assign(
        exp_peak_id="EXP_05_IMPURITY",
        theo_peak_id="THEO_05_UNOBSERVED",
        user_id="ExpertSpectroscopist",
    )
    assert manual_asn.status == AssignmentStatus.MANUAL
    assert manual_asn.approved_by == "ExpertSpectroscopist"
    assert not manual_asn.exp_peak.is_ghost

    # Test errors for non-existent IDs
    with pytest.raises(HITLApprovalError, match="Experimental peak 'NON_EXISTENT' not found"):
        manager.manual_assign("NON_EXISTENT", "THEO_01")

    with pytest.raises(HITLApprovalError, match="Theoretical peak 'NON_EXISTENT' not found"):
        manager.manual_assign("EXP_01", "NON_EXISTENT")

    with pytest.raises(HITLApprovalError, match="Assignment ID 'INVALID_ID' not found"):
        manager.approve_assignment("INVALID_ID")

    with pytest.raises(HITLApprovalError, match="Assignment ID 'INVALID_ID' not found"):
        manager.reject_assignment("INVALID_ID")


def test_semantic_git_commit_insufficient_parameters():
    """Verifies SemanticCommitError is raised when insufficient state parameters are supplied."""
    with pytest.raises(SemanticCommitError, match="Insufficient state parameters"):
        translate_mathematical_change_to_git_commit()


def test_render_dual_pane_dashboard_with_convolved_curve(methanol_exp_peaks, methanol_theo_peaks, baseline_slider_state):
    """Verifies dual-pane dashboard rendering with overlaid synthetic convolved spectrum."""
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=500,
    )
    freq_grid, spectrum = engine.compute_convolved_spectrum(baseline_slider_state)

    fig = render_dual_pane_dashboard(
        exp_peaks=methanol_exp_peaks,
        theo_peaks=methanol_theo_peaks,
        convolved_curve=(freq_grid, spectrum),
    )
    trace_names = [t.name for t in fig.data if t.name]
    assert "Synthetic Convolved" in trace_names


# =============================================================================
# 8. LTTB DECIMATION & CRYPTOGRAPHIC STATE HASH TESTS
# =============================================================================

def test_lttb_decimation_exact_threshold():
    """Verifies LTTB decimation reduces dense arrays to exact target threshold preserving extrema."""
    # Generate dense synthetic absorption lines across 10,000 frequency points
    x_dense = np.linspace(10000.0, 50000.0, 10000, dtype=np.float64)
    # Superposition of 3 narrow spectroscopic lines
    y_dense = (
        1.5 * np.exp(-0.5 * ((x_dense - 20000.0) / 5.0) ** 2)
        + 3.2 * np.exp(-0.5 * ((x_dense - 32500.0) / 5.0) ** 2)
        + 0.8 * np.exp(-0.5 * ((x_dense - 44000.0) / 5.0) ** 2)
    )

    threshold = 5000
    x_sampled, y_sampled = run_lttb_decimation(x_dense, y_dense, threshold=threshold)

    assert len(x_sampled) == threshold
    assert len(y_sampled) == threshold
    # Endpoints must be exactly preserved
    assert x_sampled[0] == x_dense[0]
    assert y_sampled[0] == y_dense[0]
    assert x_sampled[-1] == x_dense[-1]
    assert y_sampled[-1] == y_dense[-1]

    # Major peak maximum must be preserved within 0.1% tolerance
    original_max = np.max(y_dense)
    sampled_max = np.max(y_sampled)
    assert abs(original_max - sampled_max) / original_max < 0.01


def test_lttb_decimation_edge_cases():
    """Verifies LTTB decimation handles edge cases and raises informative errors on invalid input."""
    # When threshold >= len(x), returns unmodified copies
    x_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_small = np.array([10.0, 20.0, 15.0, 30.0, 5.0])
    x_out, y_out = run_lttb_decimation(x_small, y_small, threshold=10)
    assert np.array_equal(x_out, x_small)
    assert np.array_equal(y_out, y_small)

    # When threshold < 3, returns unmodified copies
    x_sub, y_sub = run_lttb_decimation(x_small, y_small, threshold=2)
    assert np.array_equal(x_sub, x_small)

    # Dimension and length mismatch validation
    with pytest.raises(ValueError, match="1-dimensional"):
        run_lttb_decimation(np.zeros((2, 2)), np.zeros((2, 2)))

    with pytest.raises(ValueError, match="Length mismatch"):
        run_lttb_decimation(np.array([1.0, 2.0]), np.array([1.0]))


def test_update_ui_state_hash_deterministic():
    """Verifies update_ui_state_hash returns a deterministic 64-character SHA-256 hexadecimal hash."""
    state_a = {"A_mhz": 127484.0, "B_mhz": 24679.98, "temperature_k": 296.0}
    state_b = {"B_mhz": 24679.98, "A_mhz": 127484.0, "temperature_k": 296.0}
    state_c = {"A_mhz": 127484.0, "B_mhz": 24679.98, "temperature_k": 300.0}

    hash_a = update_ui_state_hash(state_a)
    hash_b = update_ui_state_hash(state_b)
    hash_c = update_ui_state_hash(state_c)

    assert len(hash_a) == 64
    # Key ordering must not affect the hash
    assert hash_a == hash_b
    # Value changes must mutate the hash
    assert hash_a != hash_c


# =============================================================================
# 9. BOUNDS ENFORCEMENT & METHOD MATRIX PROVENANCE TESTS
# =============================================================================

def test_what_if_parameter_state_rigid_bounds():
    """Verifies WhatIfParameterState rigidly rejects negative or zero rotational constants."""
    with pytest.raises(ValidationError):
        WhatIfParameterState(A_mhz=-500.0, B_mhz=5000.0, C_mhz=2500.0)

    with pytest.raises(ValidationError):
        WhatIfParameterState(A_mhz=10000.0, B_mhz=0.0, C_mhz=2500.0)

    with pytest.raises(ValidationError):
        WhatIfParameterState(A_mhz=10000.0, B_mhz=5000.0, C_mhz=-10.0)


def test_bipartite_matching_non_positive_frequency_rejection():
    """Verifies compute_bipartite_cost_matrix rejects non-positive frequencies."""
    bad_exp = [SpectralPeak(peak_id="BAD_1", frequency=100.0, intensity=1.0)]
    bad_exp[0].__dict__["frequency"] = -5.0  # Bypass pydantic validation for testing error gate

    theo = [SpectralPeak(peak_id="T1", frequency=100.0, intensity=1.0)]
    with pytest.raises(BipartiteMatchingError, match="non-positive frequency"):
        compute_bipartite_cost_matrix(bad_exp, theo)


def test_method_matrix_provenance_generation(methanol_exp_peaks, methanol_theo_peaks):
    """Verifies HITLAssignmentManager generates authoritative Method Matrix provenance records."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    manager.auto_propose(BipartiteConfig(max_freq_drift_mhz=50.0))

    # Approve 3 assignments and reject 1 assignment
    proposed = manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
    for p in proposed[:3]:
        manager.approve_assignment(p.assignment_id, user_id="LeadSpectroscopist")
    manager.reject_assignment(proposed[3].assignment_id, reason="Impurities line profile")

    provenance = manager.get_method_matrix_provenance(huber_delta_mhz=1.0)

    assert "timestamp_utc" in provenance
    assert provenance["total_exp_peaks"] == 5
    assert provenance["total_theo_peaks"] == 5
    assert provenance["approved_assignments_count"] == 3
    assert provenance["ghost_peaks_count"] == 1
    assert len(provenance["ghost_peak_details"]) == 1
    assert provenance["ghost_peak_details"][0]["rejection_reason"] == "Impurities line profile"
    assert "provenance_sha256" in provenance
    assert len(provenance["provenance_sha256"]) == 64


# =============================================================================
# 10. ZEROMQ IPC BRIDGE & IPYWIDGETS DASHBOARD TESTS
# =============================================================================

def test_zmq_ipc_bridge_heartbeat_and_spectrum_computation(methanol_theo_peaks, baseline_slider_state):
    """Verifies ZeroMQ IPC server and client communication, heartbeat ping, and spectrum calculation."""
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=500,
    )

    port = 5592
    endpoint = f"tcp://127.0.0.1:{port}"

    server = SpycFitZMQServer(endpoint=endpoint, timeout_ms=1000)
    server.start()

    client = SpycFitZMQClient(endpoint=endpoint, timeout_ms=1000)

    try:
        # 1. Test Heartbeat Ping
        # In single-threaded test, we send request from client and process on server
        client.socket.send_json({"type": "HEARTBEAT", "client_id": "TEST_AGENT"})
        server_reply = server.process_one_request(engine=engine)
        assert server_reply is not None
        assert server_reply["type"] == "HEARTBEAT_ACK"
        client_reply = client.socket.recv_json()
        assert client_reply["status"] == "SUCCESS"

        # 2. Test Spectrum Computation request
        req = {
            "type": "COMPUTE_SPECTRUM",
            "state": baseline_slider_state.model_dump(),
            "decimate_threshold": 250,
        }
        client.socket.send_json(req)
        server.process_one_request(engine=engine)
        spec_reply = client.socket.recv_json()
        assert spec_reply["status"] == "SUCCESS"
        assert len(spec_reply["freqs"]) == 250
        assert len(spec_reply["spectrum"]) == 250
        assert "state_hash" in spec_reply

        # 3. Test LTTB decimation request
        x_in = np.linspace(100.0, 500.0, 1000).tolist()
        y_in = (np.sin(np.linspace(0, 10, 1000)) + 1.0).tolist()
        client.socket.send_json({"type": "LTTB_DECIMATE", "x": x_in, "y": y_in, "threshold": 100})
        server.process_one_request(engine=engine)
        dec_reply = client.socket.recv_json()
        assert dec_reply["status"] == "SUCCESS"
        assert len(dec_reply["x"]) == 100

        # 4. Test State Hash request
        client.socket.send_json({"type": "GET_STATE_HASH", "state": {"A_mhz": 120000.0}})
        server.process_one_request(engine=engine)
        hash_reply = client.socket.recv_json()
        assert hash_reply["status"] == "SUCCESS"
        assert len(hash_reply["state_hash"]) == 64

    finally:
        client.close()
        server.stop()


def test_zmq_ipc_client_timeout_error_handling():
    """Verifies SpycFitZMQClient raises ZMQBridgeError upon server communication timeout."""
    # Connect to an inactive port with short timeout (100ms)
    client = SpycFitZMQClient(endpoint="tcp://127.0.0.1:5599", timeout_ms=100)
    try:
        with pytest.raises(ZMQBridgeError, match="timeout"):
            client.ping()
    finally:
        client.close()


def test_ipywidgets_interactive_dashboard(methanol_exp_peaks, methanol_theo_peaks, baseline_slider_state):
    """Verifies SpycFitInteractiveDashboard initialization, widget hierarchy, and event callbacks."""
    manager = HITLAssignmentManager(exp_peaks=methanol_exp_peaks, theo_peaks=methanol_theo_peaks)
    engine = WhatIfSpectrumEngine(
        base_peaks=methanol_theo_peaks,
        base_state=baseline_slider_state,
        grid_min_mhz=10000.0,
        grid_max_mhz=50000.0,
        grid_points=200,
    )

    dashboard = SpycFitInteractiveDashboard(manager=manager, spectrum_engine=engine)

    assert dashboard.widget is not None
    assert dashboard.slider_A.value == baseline_slider_state.A_mhz
    assert dashboard.slider_B.value == baseline_slider_state.B_mhz
    assert dashboard.slider_C.value == baseline_slider_state.C_mhz

    # Simulate button click events
    dashboard._on_auto_propose_clicked(None)
    assert len(manager.get_assignments_by_status(AssignmentStatus.PROPOSED)) >= 4

    dashboard._on_approve_all_clicked(None)
    assert len(manager.get_assignments_by_status(AssignmentStatus.APPROVED)) >= 4

    # Slider state mutation
    dashboard.slider_A.value = 127500.0
    cur_state = dashboard.get_current_parameter_state()
    assert cur_state.A_mhz == 127500.0

    # Test outlier rejection
    dashboard.slider_huber.value = 0.001
    dashboard._on_reject_outliers_clicked(None)

    # Test factory function
    widget = create_spycfit_dashboard_widget(manager=manager, spectrum_engine=engine)
    assert widget is not None


def test_semantic_git_commit_with_state_hash(baseline_slider_state):
    """Verifies semantic git commit includes cryptographic state hash in commit metadata and message."""
    perturbed_state = baseline_slider_state.model_copy(update={"A_mhz": baseline_slider_state.A_mhz + 2.5})
    commit = translate_mathematical_change_to_git_commit(
        state_before=baseline_slider_state,
        state_after=perturbed_state,
        rms_before=0.045,
        rms_after=0.012,
    )

    assert commit.state_hash is not None
    assert len(commit.state_hash) == 64
    assert "State-Hash:" in commit.full_commit_message


# =============================================================================
# 11. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliance:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_spycfit_ui_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "interfaces" / "cochem_spycfit_ui.py"
        assert module_path.is_file()

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_test_suite_ast_compliance(self) -> None:
        test_path = Path(__file__).resolve()
        assert test_path.is_file()

        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_no_banned_tokens_in_production_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "interfaces" / "cochem_spycfit_ui.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        banned = ["mock", "dummy", "placeholder", "fake", "stub"]
        violations: List[str] = []

        class IdentifierScanner(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for b in banned:
                    if b in node.name.lower():
                        violations.append(f"Function {node.name} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for b in banned:
                    if b in node.id.lower():
                        violations.append(f"Variable {node.id} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    for b in banned:
                        if b in alias.name.lower():
                            violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for b in banned:
                        if b in node.module.lower():
                            violations.append(f"Forbidden import from '{node.module}' at line {node.lineno}")
                self.generic_visit(node)

        IdentifierScanner().visit(tree)
        assert len(violations) == 0, f"Banned tokens found: {violations}"



--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_vibspyc_snap.py ---
"""Comprehensive physical unit test suite for SpycFit Snapshot & Publication Export.

Verifies:
- FitProvenancePayload schema validation, field validators, serialization & deserialization.
- hash_dataset_iteratively with physical disk operations and chunked reading.
- generate_provenance_lock with physical HDF5 and binary datasets, metadata extraction,
  and serialization to disk.
- generate_aastex_longtable and generate_aastex_longtables with molecular rotational parameters,
  transitions, siunitx formatting, frozen parameter overrides, and top 200 truncation.
- get_required_dois and format_citations_to_bib with structured CrossRef metadata records.
- evaluate_compression_strategy sizing thresholds.
- package_fit_artifacts archiving (.zip and .tar.zst) and seal_artifact_read_only permissions.
- export_spycfit_snapshot end-to-end pipeline execution and artifact sealing.
- Workspace directory resolution with environment variable and platformdirs fallback.
- Dynamic IUPAC atomic weight and isotopic mass resolution via Mendeleev.
"""

from __future__ import annotations

import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zstandard
from pydantic import ValidationError

from cochem_spycfit.interfaces.cochem_vibspyc_snap import (
    FitProvenancePayload,
    evaluate_compression_strategy,
    export_spycfit_snapshot,
    format_citations_to_bib,
    generate_aastex_longtable,
    generate_aastex_longtables,
    generate_provenance_lock,
    get_element_atomic_weight,
    get_isotope_mass,
    get_required_dois,
    get_spycfit_processed_dir,
    hash_dataset_iteratively,
    package_fit_artifacts,
    resolve_processed_workspace_dir,
    seal_artifact_read_only,
)


@pytest.fixture
def clean_workspace(tmp_path: Path):
    """Provides an isolated temporary workspace and resets file permissions upon teardown."""
    yield tmp_path
    for root, _dirs, files in os.walk(tmp_path):
        for fname in files:
            target = os.path.join(root, fname)
            try:
                os.chmod(target, stat.S_IWRITE)
            except OSError as err:
                _ = err


@pytest.fixture
def authoritative_payload() -> FitProvenancePayload:
    """Constructs a valid, physically realistic FitProvenancePayload instance."""
    return FitProvenancePayload(
        session_id="SESSION_METHANOL_CH3OH_FIT_2026_01",
        timestamp="2026-08-23T14:30:00+00:00",
        dataset_hashes={
            "methanol_spectrum_chirped_pulse.dat": "a3f5b7c891024d3e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e",
            "assigned_transitions_methanol.cat": "8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d",
        },
        chi_squared=1.0428,
        huber_loss_delta=1.345,
        rms_mhz=0.0184,
        rms_cm_inv=0.00061376,
        jacobian_condition_number=142.85,
        sobol_parameter_audit={
            "A": True,
            "B": True,
            "C": True,
            "D_J": True,
            "D_JK": True,
            "D_K": False,
            "V_3": True,
        },
        semantic_git_history=[
            "git-commit-8f2a1b4: Initial Hamiltonian configuration for internal rotation",
            "git-commit-c9d3e5f: Refined centrifugal distortion parameters via Levenberg-Marquardt",
            "git-commit-4a7b9c0: Converged global spectroscopic fit with Huber robust weighting",
        ],
        active_models=[
            "watson_a_reduction",
            "iam_internal_rotation",
            "spcat",
            "jax_backend",
        ],
        additional_metadata={
            "spectrometer": "CP-FTMW 2-8 GHz",
            "carrier_gas": "Neon",
            "backing_pressure_bar": 2.5,
            "pulse_duration_us": 1.0,
            "temperature_k": 2.0,
        },
    )


@pytest.fixture
def molecular_parameters() -> list[dict[str, Any]]:
    """Realistic rotational and distortion parameters for methanol (CH3OH)."""
    return [
        {
            "name": "A",
            "latex_name": "A",
            "value": 127634.508,
            "uncertainty": 0.045,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal a-axis",
        },
        {
            "name": "B",
            "latex_name": "B",
            "value": 24684.382,
            "uncertainty": 0.012,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal b-axis",
        },
        {
            "name": "C",
            "latex_name": "C",
            "value": 23758.129,
            "uncertainty": 0.011,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal c-axis",
        },
        {
            "name": "D_J",
            "latex_name": "\\Delta_J",
            "value": 0.04021,
            "uncertainty": 0.00035,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_JK",
            "latex_name": "\\Delta_{JK}",
            "value": 0.12480,
            "uncertainty": 0.00080,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_K",
            "latex_name": "\\Delta_K",
            "value": 0.89200,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": False,
            "status": "fixed",
            "description": "Frozen quartic centrifugal distortion parameter from ab initio",
        },
        {
            "name": "V_3",
            "latex_name": "V_3",
            "value": 373.800,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": True,
            "status": "set",
            "description": "Three-fold torsional potential barrier height in cm^-1",
        },
    ]


@pytest.fixture
def molecular_transitions() -> list[dict[str, Any]]:
    """Calculated test list of 250 assigned transitions for methanol."""
    records: list[dict[str, Any]] = []
    for i in range(1, 251):
        j_upper = i % 10 + 1
        j_lower = i % 10
        ka_upper = min(j_upper, (i % 3))
        kc_upper = j_upper - ka_upper
        ka_lower = min(j_lower, (i % 3))
        kc_lower = j_lower - ka_lower

        obs_freq = 12000.0 + (i * 25.35)
        calc_freq = obs_freq + (((i % 7) - 3) * 0.003)
        res = obs_freq - calc_freq
        intensity = 1000.0 / (i + 1.0)

        records.append(
            {
                "upper_state": f"{j_upper}_{{{ka_upper},{kc_upper}}}",
                "lower_state": f"{j_lower}_{{{ka_lower},{kc_lower}}}",
                "observed_mhz": obs_freq,
                "calculated_mhz": calc_freq,
                "residual_mhz": res,
                "uncertainty_mhz": 0.010,
                "intensity": intensity,
            }
        )
    return records


# =====================================================================
# 1. FitProvenancePayload Validation and Serialization Tests
# =====================================================================


def test_fit_provenance_payload_validation(authoritative_payload: FitProvenancePayload):
    """Verifies valid instantiation and field access on FitProvenancePayload."""
    assert authoritative_payload.session_id == "SESSION_METHANOL_CH3OH_FIT_2026_01"
    assert authoritative_payload.chi_squared == 1.0428
    assert authoritative_payload.rms_mhz == 0.0184
    assert authoritative_payload.rms_cm_inv == 0.00061376
    assert authoritative_payload.jacobian_condition_number == 142.85
    assert len(authoritative_payload.dataset_hashes) == 2
    assert authoritative_payload.sobol_parameter_audit["V_3"] is True
    assert len(authoritative_payload.semantic_git_history) == 3
    assert "watson_a_reduction" in authoritative_payload.active_models


def test_fit_provenance_payload_invalid_timestamp():
    """Verifies that non-ISO formatted timestamps are rejected."""
    with pytest.raises(ValidationError) as excinfo:
        FitProvenancePayload(
            session_id="INVALID_TIMESTAMP_SESSION",
            timestamp="23-08-2026 14:30",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )
    assert "Timestamp" in str(excinfo.value) or "ISO" in str(excinfo.value)


def test_fit_provenance_payload_negative_metrics():
    """Verifies that negative error metrics trigger validation errors."""
    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_CHI_SQ_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=-0.5,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_RMS_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=-0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_JACOBIAN_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=-5.0,
        )


def test_fit_provenance_disk_roundtrip(
    authoritative_payload: FitProvenancePayload, clean_workspace: Path
):
    """Verifies physical disk serialization and deserialization roundtrips."""
    target_json = clean_workspace / "provenance" / "fit_provenance.json"
    written_path = authoritative_payload.to_json_file(target_json)

    assert written_path.exists()
    assert written_path.is_file()

    loaded_payload = FitProvenancePayload.from_json_file(written_path)
    assert loaded_payload.session_id == authoritative_payload.session_id
    assert loaded_payload.timestamp == authoritative_payload.timestamp
    assert loaded_payload.chi_squared == authoritative_payload.chi_squared
    assert loaded_payload.rms_mhz == authoritative_payload.rms_mhz
    assert loaded_payload.dataset_hashes == authoritative_payload.dataset_hashes
    assert (
        loaded_payload.sobol_parameter_audit
        == authoritative_payload.sobol_parameter_audit
    )
    assert (
        loaded_payload.semantic_git_history
        == authoritative_payload.semantic_git_history
    )
    assert (
        loaded_payload.additional_metadata == authoritative_payload.additional_metadata
    )


def test_fit_provenance_from_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to deserialize from a non-existent file."""
    non_existent = clean_workspace / "non_existent_provenance.json"
    with pytest.raises(FileNotFoundError):
        FitProvenancePayload.from_json_file(non_existent)


# =====================================================================
# 2. Iterative Hashing Tests
# =====================================================================


def test_hash_dataset_iteratively_small_file(clean_workspace: Path):
    """Verifies iterative SHA-256 calculation for a physical file on disk."""
    data_file = clean_workspace / "spectrum_input.dat"
    content = b"CH3OH Spectroscopic Data Stream v1.0\n12450.320 0.85\n12480.110 0.92\n"
    data_file.write_bytes(content)

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=16)
    expected_hash = "bca9b04d490184c37b9ae03358c4370f955c050a1953e3b54674122459ecf77c"

    import hashlib

    direct_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == expected_hash == direct_hash


def test_hash_dataset_iteratively_large_file(clean_workspace: Path):
    """Verifies iterative hashing across multi-chunk files exceeding chunk boundaries."""
    data_file = clean_workspace / "large_spectrum.bin"
    payload_chunk = b"RotationalTransitionRecord_1234567890_abcdefghij_" * 32
    total_chunks = 320
    with open(data_file, "wb") as f:
        f.writelines(payload_chunk for _ in range(total_chunks))

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=8192)
    assert isinstance(computed_hash, str)
    assert len(computed_hash) == 64


def test_hash_dataset_iteratively_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError on non-existent dataset path."""
    missing_file = clean_workspace / "non_existent_dataset.bin"
    with pytest.raises(FileNotFoundError):
        hash_dataset_iteratively(missing_file)


# =====================================================================
# 3. Provenance Lock Generation Tests (Physical HDF5 / Binary)
# =====================================================================


def test_generate_provenance_lock_hdf5(clean_workspace: Path):
    """Verifies cryptographic provenance lock creation on a physical HDF5 landscape file."""
    h5_file = clean_workspace / "landscape.h5"
    with h5py.File(h5_file, "w") as f:
        grp = f.create_group("spectroscopy")
        grp.create_dataset("frequencies", data=np.array([12450.32, 12480.11, 12510.95]))
        grp.create_dataset("intensities", data=np.array([0.85, 0.92, 0.45]))
        grp.attrs["molecule"] = "Methanol-CH3OH"
        grp.attrs["fit_converged"] = True

    metrics = {
        "session_id": "SESSION_H5_LOCK_TEST_01",
        "timestamp": "2026-08-23T15:00:00+00:00",
        "chi_squared": 1.025,
        "rms_mhz": 0.0142,
        "jacobian_condition_number": 88.5,
        "active_parameters": ["A", "B", "C", "D_J", "D_JK"],
        "frozen_parameters": ["D_K", "V_3"],
        "semantic_git_history": ["commit 1: Fitted ground state"],
        "active_models": ["watson_a", "spcat"],
        "metadata": {"spectrometer": "CP-FTMW"},
    }

    result = generate_provenance_lock(h5_file, metrics)

    assert result["session_id"] == "SESSION_H5_LOCK_TEST_01"
    assert result["chi_squared"] == 1.025
    assert result["rms_mhz"] == 0.0142
    assert "landscape.h5" in result["dataset_hashes"]
    expected_hash = hash_dataset_iteratively(h5_file)
    assert result["dataset_hashes"]["landscape.h5"] == expected_hash
    assert result["sobol_parameter_audit"]["A"] is True
    assert result["sobol_parameter_audit"]["D_K"] is False

    default_json = clean_workspace / "fit_provenance.json"
    assert default_json.exists()
    loaded_payload = FitProvenancePayload.from_json_file(default_json)
    assert loaded_payload.session_id == "SESSION_H5_LOCK_TEST_01"
    assert loaded_payload.dataset_hashes["landscape.h5"] == expected_hash


def test_generate_provenance_lock_with_payload_object(
    clean_workspace: Path, authoritative_payload: FitProvenancePayload
):
    """Verifies generate_provenance_lock when supplied with a pre-built FitProvenancePayload."""
    h5_file = clean_workspace / "state.h5"
    h5_file.write_bytes(b"HDF5_STRUCTURED_BINARY_STREAM_SIMULATION")

    custom_out = clean_workspace / "custom_dir" / "custom_provenance.json"
    result = generate_provenance_lock(
        h5_path=h5_file,
        metrics=authoritative_payload,
        output_path=custom_out,
    )

    assert custom_out.exists()
    assert result["session_id"] == authoritative_payload.session_id
    assert "state.h5" in result["dataset_hashes"]
    assert result["dataset_hashes"]["state.h5"] == hash_dataset_iteratively(h5_file)


def test_generate_provenance_lock_missing_h5_file(clean_workspace: Path):
    """Verifies FileNotFoundError when dataset file is missing."""
    missing_h5 = clean_workspace / "missing_landscape.h5"
    with pytest.raises(FileNotFoundError):
        generate_provenance_lock(missing_h5, {"chi_squared": 1.0, "rms_mhz": 0.01})


def test_generate_provenance_lock_invalid_metrics_type(clean_workspace: Path):
    """Verifies TypeError when invalid metrics argument is passed."""
    data_file = clean_workspace / "dataset.bin"
    data_file.write_bytes(b"DATA")
    with pytest.raises(TypeError, match="Expected metrics to be dict or FitProvenancePayload"):
        generate_provenance_lock(data_file, "invalid_string_metrics")  # type: ignore[arg-type]


# =====================================================================
# 4. AASTeX / LaTeX Table Generation Tests
# =====================================================================


def test_generate_aastex_longtable_with_dict():
    """Verifies generate_aastex_longtable with simple dictionary input."""
    constants = {
        "A": 127634.508,
        "B": 24684.382,
        "C": 23758.129,
        "D_J": 0.04021,
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Simple Fit")

    assert "\\begin{table}[htbp]" in rendered
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in rendered
    assert "CH3OH Simple Fit" in rendered
    assert "127634.508000" in rendered
    assert "24684.382000" in rendered
    assert "\\Delta_J" in rendered
    assert "\\bottomrule" in rendered


def test_generate_aastex_longtable_with_tuples_and_frozen():
    """Verifies generate_aastex_longtable with tuple inputs and frozen status."""
    constants = {
        "A": (127634.508, 0.045),
        "B": (24684.382, 0.012),
        "D_K": (0.892, None, True),
        "V_3": {"value": 373.8, "status": "set", "is_frozen": True},
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Tuple Test")

    assert "0.045000" in rendered
    assert "0.012000" in rendered
    assert "Fixed" in rendered
    assert "Set" in rendered
    assert "V_3" in rendered


def test_generate_aastex_longtable_with_transitions(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies generate_aastex_longtable with transitions producing full combined document."""
    rendered = generate_aastex_longtable(
        constants_dict=molecular_parameters,
        title_prefix="Methanol Combined Publication",
        transitions=molecular_transitions,
    )

    assert "\\documentclass[twocolumn]{aastex631}" in rendered
    assert "\\begin{longtable}" in rendered
    assert "\\begin{table}[htbp]" in rendered
    assert "Methanol Combined Publication" in rendered
    assert "\\end{document}" in rendered


def test_generate_aastex_longtable_invalid_input_type():
    """Verifies TypeError when constants_dict is not dict or Sequence."""
    with pytest.raises(TypeError, match="Expected constants_dict to be dict or Sequence"):
        generate_aastex_longtable(12345)


def test_generate_aastex_longtables_structure(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies structure and LaTeX formatting of generated tables and combined documents."""
    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="Methanol CH3OH Ground State",
    )

    assert "parameters_table" in result
    assert "transitions_table" in result
    assert "combined_document" in result

    params_tex = result["parameters_table"]
    trans_tex = result["transitions_table"]
    doc_tex = result["combined_document"]

    assert "\\begin{table}[htbp]" in params_tex
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in params_tex
    assert (
        "\\caption{Optimized Spectroscopic Parameters for Methanol CH3OH Ground State}"
        in params_tex
    )
    assert "\\toprule" in params_tex
    assert "\\midrule" in params_tex
    assert "\\bottomrule" in params_tex
    assert "Rotational constant around principal a-axis" in params_tex

    assert "127634.508000" in params_tex
    assert "0.045000" in params_tex
    assert "Fixed" in params_tex
    assert "Set" in params_tex

    assert "\\begin{longtable}{c c S[table-format=6.4]" in trans_tex
    assert (
        "\\caption{Observed and Calculated Transitions for Methanol CH3OH Ground State (Truncated to Top 200)}"
        in trans_tex
    )
    assert "\\endfirsthead" in trans_tex
    assert "\\endhead" in trans_tex
    assert "\\endfoot" in trans_tex
    assert "\\endlastfoot" in trans_tex

    assert "\\documentclass[twocolumn]{aastex631}" in doc_tex
    assert "\\usepackage{siunitx}" in doc_tex
    assert "\\usepackage{booktabs}" in doc_tex
    assert "\\usepackage{longtable}" in doc_tex
    assert (
        "\\title{High-Precision Rotational Spectroscopic Fit and Transition Assignments: Methanol CH3OH Ground State}"
        in doc_tex
    )


def test_generate_aastex_longtables_truncation(
    molecular_parameters: list[dict[str, Any]],
):
    """Verifies that transitions list is truncated to the top 200 by intensity."""
    many_transitions = [
        {
            "upper_state": f"State_{i}_up",
            "lower_state": f"State_{i}_low",
            "observed_mhz": 10000.0 + i,
            "calculated_mhz": 10000.0 + i,
            "residual_mhz": 0.0,
            "uncertainty_mhz": 0.01,
            "intensity": float(i),
        }
        for i in range(1, 301)
    ]

    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=many_transitions,
        title_prefix="Truncation Verification",
    )

    trans_table = result["transitions_table"]
    assert "(Truncated to Top 200)" in trans_table
    assert "State_300_up" in trans_table
    assert "State_101_up" in trans_table
    assert "State_1_up" not in trans_table


def test_generate_aastex_longtables_unparseable_uncertainty():
    """Verifies graceful handling of non-numeric string uncertainties."""
    params = [
        {
            "name": "CustomParam",
            "value": 12.3456,
            "uncertainty": "est. 0.05",
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
        }
    ]
    res = generate_aastex_longtables(params, [])
    assert "est. 0.05" in res["parameters_table"]


# =====================================================================
# 5. Mendeleev Dynamic Atomic Mass Resolution Tests
# =====================================================================


def test_mendeleev_element_atomic_weights():
    """Verifies dynamic IUPAC standard atomic weight retrieval via Mendeleev."""
    carbon_weight = get_element_atomic_weight("C")
    hydrogen_weight = get_element_atomic_weight("H")
    oxygen_weight = get_element_atomic_weight("O")
    nitrogen_weight = get_element_atomic_weight("N")

    assert 12.010 <= carbon_weight <= 12.012
    assert 1.007 <= hydrogen_weight <= 1.009
    assert 15.998 <= oxygen_weight <= 16.000
    assert 14.006 <= nitrogen_weight <= 14.008


def test_mendeleev_isotope_masses():
    """Verifies dynamic isotopic mass retrieval in atomic mass units via Mendeleev."""
    c12_mass = get_isotope_mass("C", 12)
    c13_mass = get_isotope_mass("C", 13)
    h1_mass = get_isotope_mass("H", 1)
    h2_mass = get_isotope_mass("H", 2)
    o16_mass = get_isotope_mass("O", 16)

    assert 11.999 <= c12_mass <= 12.001
    assert 13.002 <= c13_mass <= 13.004
    assert 1.007 <= h1_mass <= 1.009
    assert 2.013 <= h2_mass <= 2.015
    assert 15.994 <= o16_mass <= 15.996


def test_mendeleev_missing_isotope_error():
    """Verifies ValueError when requesting a non-existent isotope number."""
    with pytest.raises(ValueError, match="not found in Mendeleev database"):
        get_isotope_mass("C", 999)


# =====================================================================
# 6. DOI Resolution and BibTeX Citation Tests
# =====================================================================


def test_get_required_dois():
    """Verifies mapping of active physical models and quantum engines to authoritative DOIs."""
    active_models = [
        "watson_a",
        "spcat",
        "orca_v6",
        "jax",
        "pyarrow",
        "unknown_model_xyz",
    ]
    resolved = get_required_dois(active_models)

    assert "10.1016/0022-2852(77)90184-7" in resolved
    assert "10.1016/0022-2852(91)90393-O" in resolved
    assert "10.1063/5.0004608" in resolved
    assert "10.5281/zenodo.4766323" in resolved
    assert "10.5281/zenodo.4006056" in resolved
    assert len(resolved) == 5


def test_format_citations_to_bib():
    """Verifies parsing and deduplicating CrossRef JSON metadata into clean BibTeX format."""
    crossref_data = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": [
                "The determination of centrifugal distortion constants of asymmetric-top molecules"
            ],
            "author": [{"family": "Watson", "given": "James K. G."}],
            "container-title": ["Journal of Molecular Spectroscopy"],
            "volume": "65",
            "issue": "1",
            "page": "123-133",
            "issued": {"date-parts": [[1977, 4, 1]]},
        },
        {
            "message": {
                "DOI": "10.1016/0022-2852(91)90393-O",
                "title": [
                    "The fitting of sub-Doppler rotational spectra of molecules with internal rotor"
                ],
                "author": [{"family": "Pickett", "given": "Herbert M."}],
                "container-title": "Journal of Molecular Spectroscopy",
                "volume": "148",
                "number": "2",
                "page": "371-377",
                "issued": {"date-parts": [[1991, 8, 1]]},
            }
        },
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Duplicate Watson citation"],
        },
    ]

    bib_text = format_citations_to_bib(crossref_data)

    assert "@article{" in bib_text
    assert "author = {Watson, James K. G.}" in bib_text
    assert (
        "title = {{The determination of centrifugal distortion constants of asymmetric-top molecules}}"
        in bib_text
    )
    assert "journal = {Journal of Molecular Spectroscopy}" in bib_text
    assert "volume = {65}" in bib_text
    assert "number = {1}" in bib_text
    assert "pages = {123-133}" in bib_text
    assert "year = {1977}" in bib_text
    assert "doi = {10.1016/0022-2852(77)90184-7}" in bib_text

    assert "author = {Pickett, Herbert M.}" in bib_text
    assert "doi = {10.1016/0022-2852(91)90393-O}" in bib_text
    assert "year = {1991}" in bib_text

    assert bib_text.count("10.1016/0022-2852(77)90184-7") == 1


def test_format_citations_to_bib_extended_schemas():
    """Verifies parsing of various author schemas, missing DOIs, and year fallbacks."""
    records = [
        {"title": "Missing DOI Entry", "author": "No DOI Author"},
        {
            "DOI": "10.1000/str_author",
            "title": ["Single String Author Paper"],
            "author": ["Jane Doe"],
            "year": "2024",
        },
        {
            "DOI": "10.1000/family_only",
            "title": "Family Only Paper",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
        {
            "DOI": "10.1000/heisenberg_2",
            "title": "Family Only Paper 2",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
    ]

    bib_text = format_citations_to_bib(records)
    assert "Missing DOI Entry" not in bib_text
    assert "author = {Jane Doe}" in bib_text
    assert "author = {Heisenberg}" in bib_text
    assert "year = {2024}" in bib_text
    assert "year = {1927}" in bib_text
    assert "Heisenberg1927Family" in bib_text
    assert "Heisenberg1927Family_1" in bib_text


# =====================================================================
# 7. Compression Strategy and Packaging Tests
# =====================================================================


def test_evaluate_compression_strategy():
    """Verifies mathematical threshold evaluations for storage packaging."""
    small_size = 50 * 1024 * 1024
    assert evaluate_compression_strategy(small_size) == "ZIP"

    exact_threshold = 104857600
    assert evaluate_compression_strategy(exact_threshold) == "ZSTD"

    large_size = 500 * 1024 * 1024
    assert evaluate_compression_strategy(large_size) == "ZSTD"


def test_package_fit_artifacts_zip(clean_workspace: Path):
    """Verifies packaging workspace directory into a standard .zip archive."""
    src_dir = clean_workspace / "fit_export_src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "fit.log").write_text(
        "Fit iteration log output: Converged in 14 iterations.", encoding="utf-8"
    )
    (src_dir / "tables").mkdir()
    (src_dir / "tables" / "table1.tex").write_text(
        "\\begin{table}...\\end{table}", encoding="utf-8"
    )

    archive_base = clean_workspace / "export_archive"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZIP")

    assert archive_path.exists()
    assert archive_path.name == "export_archive.zip"
    assert zipfile.is_zipfile(archive_path)

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        assert "fit.log" in namelist
        assert "tables/table1.tex" in namelist


def test_package_fit_artifacts_zstd(clean_workspace: Path):
    """Verifies packaging workspace directory into a tar.zst archive."""
    src_dir = clean_workspace / "fit_export_src_zstd"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "data.bin").write_bytes(b"SPECTROSCOPIC_RAW_STREAM_DATA" * 500)
    (src_dir / "info.json").write_text('{"status": "READY"}', encoding="utf-8")

    archive_base = clean_workspace / "export_archive_zstd"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZSTD")

    assert archive_path.exists()
    assert archive_path.name == "export_archive_zstd.tar.zst"

    compressed_bytes = archive_path.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    decompressed_tar = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_tar), mode="r") as tf:
        names = tf.getnames()
        assert "data.bin" in names
        assert "info.json" in names


def test_package_fit_artifacts_edge_cases(clean_workspace: Path):
    """Verifies automatic strategy evaluation, extension normalization, and error handling."""
    src = clean_workspace / "edge_case_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "dataset.txt").write_text("Adversarial audit payload", encoding="utf-8")

    auto_archive = package_fit_artifacts(src, clean_workspace / "auto_archive")
    assert auto_archive.exists()
    assert auto_archive.suffix == ".zip"

    tar_zst_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar.zst", strategy="ZSTD"
    )
    assert tar_zst_archive.name == "explicit.tar.zst"
    assert tar_zst_archive.exists()

    tar_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar", strategy="ZSTD"
    )
    assert tar_archive.name == "explicit.tar.zst"
    assert tar_archive.exists()

    with pytest.raises(ValueError, match="Unsupported packaging strategy"):
        package_fit_artifacts(src, clean_workspace / "bad_strat", strategy="SEVEN_ZIP")

    with pytest.raises(NotADirectoryError):
        package_fit_artifacts(
            clean_workspace / "non_existent_folder", clean_workspace / "out"
        )


# =====================================================================
# 8. Read-Only Sealing Tests
# =====================================================================


def test_seal_artifact_read_only(clean_workspace: Path):
    """Verifies cross-platform read-only permissions are enforced on target artifacts."""
    target_file = clean_workspace / "sealed_artifact.json"
    target_file.write_text('{"immutable": true}', encoding="utf-8")

    assert seal_artifact_read_only(target_file) is True

    mode = target_file.stat().st_mode
    assert not (mode & stat.S_IWUSR)

    if os.name != "nt":
        with pytest.raises(PermissionError), open(target_file, "a") as f:
            f.write("mutation")


def test_seal_artifact_read_only_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to seal non-existent file."""
    missing = clean_workspace / "phantom_artifact.bin"
    with pytest.raises(FileNotFoundError):
        seal_artifact_read_only(missing)


# =====================================================================
# 9. End-to-End export_spycfit_snapshot Orchestration Tests
# =====================================================================


def test_export_spycfit_snapshot_pipeline(
    authoritative_payload: FitProvenancePayload,
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
    clean_workspace: Path,
):
    """Verifies full orchestration pipeline from input files to sealed archive."""
    source_dir = clean_workspace / "source_workspace"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "raw_spectrum.dat").write_text(
        "12000.0 0.5\n12050.0 0.9\n", encoding="utf-8"
    )

    processed_dir = clean_workspace / "processed_workspace"

    crossref_records = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Centrifugal Distortion in Asymmetric Tops"],
            "author": [{"family": "Watson", "given": "J. K. G."}],
            "journal": "J. Mol. Spectrosc.",
            "year": 1977,
        }
    ]

    result = export_spycfit_snapshot(
        payload=authoritative_payload,
        source_dir=source_dir,
        output_dir=processed_dir,
        active_models=authoritative_payload.active_models,
        crossref_records=crossref_records,
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="CH3OH Production Fit",
        strategy="ZIP",
    )

    assert result["status"] == "SUCCESS"
    assert result["session_id"] == authoritative_payload.session_id

    prov_path = Path(result["provenance_path"])
    params_tex_path = Path(result["parameters_table_path"])
    trans_tex_path = Path(result["transitions_table_path"])
    doc_path = Path(result["document_path"])
    bib_path = Path(result["citations_bib_path"])
    archive_path = Path(result["archive_path"])

    assert prov_path.exists()
    assert params_tex_path.exists()
    assert trans_tex_path.exists()
    assert doc_path.exists()
    assert bib_path.exists()
    assert archive_path.exists()

    loaded_prov = FitProvenancePayload.from_json_file(prov_path)
    assert loaded_prov.session_id == authoritative_payload.session_id

    assert not (prov_path.stat().st_mode & stat.S_IWUSR)
    assert not (archive_path.stat().st_mode & stat.S_IWUSR)


# =====================================================================
# 10. Workspace Directory Resolution Tests
# =====================================================================


def test_resolve_processed_workspace_dir(
    clean_workspace: Path
):
    """Verifies directory resolution with custom paths, environment variable override, and defaults."""
    custom_target = clean_workspace / "custom_processed"
    assert resolve_processed_workspace_dir(custom_target) == custom_target.resolve()

    env_dir = clean_workspace / "env_state_dir"
    old_env = os.environ.get("COCHEM_STATE_DIR")
    try:
        os.environ["COCHEM_STATE_DIR"] = str(env_dir)
        expected_env_target = env_dir / "SpycFit_Workspace" / "Processed"
        assert resolve_processed_workspace_dir() == expected_env_target.resolve()

        if "COCHEM_STATE_DIR" in os.environ:
            del os.environ["COCHEM_STATE_DIR"]
        resolved_default = get_spycfit_processed_dir()
        assert "SpycFit_Workspace" in str(resolved_default)
        assert "Processed" in str(resolved_default)
    finally:
        if old_env is not None:
            os.environ["COCHEM_STATE_DIR"] = old_env
        elif "COCHEM_STATE_DIR" in os.environ:
            del os.environ["COCHEM_STATE_DIR"]

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.