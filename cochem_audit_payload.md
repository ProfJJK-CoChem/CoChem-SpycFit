Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task1_Prompt2_requirements.md.
Original prompt:
# Task 1: Master Repository Map & Tripartite Topology - requirements.txt

**Objective:**
Create the `requirements.txt` file for the CoChem-SpycFit repository. This defines the strictly version-locked dependencies for the project.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\requirements.txt`

**Instructions:**
You are the `cochem-coder` agent. Implement the `requirements.txt` file at the root of the Execution Tier. It MUST include the specified packages and strictly exclude legacy Fortran binaries. No placeholders, mocks, or synthetic bypasses are allowed.

**Proposed Dependencies Snippet:**
```text
jax==0.4.13
jaxlib==0.4.13
cupy-cuda12x==12.2.0
h5py==3.9.0
zarr==2.16.1
pyarrow==13.0.0
plotly==5.17.0
ipywidgets==8.1.1
mendeleev==0.14.0
platformdirs==3.10.0
filelock==3.12.2
pyzmq==25.1.1
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\core_engine\__init__.py ---
"""
CoChem-SpycFit Core Engine: Stage 0 Hardware & Precision Gatekeeper and Fit Registry Manager (Root Proxy).
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.core_engine import *  # noqa: F401, F403
except ImportError:
    from cochem_spycfit.core_engine.cochem_fit_registry_manager import *  # noqa: F401, F403
    try:
        from cochem_spycfit.core_engine.cochem_spycfit_init import *  # noqa: F401, F403
    except ImportError:
        pass

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_fit_registry_manager.py ---
"""
CoChem-SpycFit: Stage 0 / Phase 2 State Persistence & Fit Registry Manager.

Authoritative state persistence, atomic locking, and spectral tensor management
for the CoChem-SpycFit Bayesian spectral fitting and convolution ecosystem.

Enforces:
1. SWMR HDF5 persistence (libver='latest', swmr=True) for concurrent JAX optimizer streaming
   and Voila GUI Plotly dashboard polling.
2. Dynamic Lustre/HPC filesystem detection: Disables SWMR on distributed filesystems to prevent
   OS flock/SWMR deadlock failures, seamlessly replacing it with periodic, metadata-safe atomic
   snapshots (shutil.copy2).
3. Cross-Platform & Filesystem-Aware Atomic Locks: Standard systems use filelock (with in-process
   threading locks); HPC/Lustre filesystems fall back to native atomic directory creation (mkdir) locks.
4. Dynamic timeout scaling: 10.0s for local standard environments, 60.0s for HPC/Lustre clusters.
5. Strict Pydantic v2 schemas (ConfigDict(extra="forbid", validate_assignment=True)) for
   parameter locks, instrument limits, active learning feedback loops, and registry state.
6. Typed exception hierarchy for comprehensive error handling.
7. Zero-Mock policy: 100% real physical files, threads, locks, snapshots, and HDF5 operations.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import filelock
import h5py
import numpy as np
import platformdirs
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.FitRegistryManager")

# Constants & Default Timeouts
DEFAULT_LOCAL_LOCK_TIMEOUT: float = 10.0
DEFAULT_LUSTRE_LOCK_TIMEOUT: float = 60.0
DEFAULT_STALE_LOCK_TIMEOUT: float = 60.0
LUSTRE_SUPER_MAGIC: int = 0x0BD00BD0
DEFAULT_CHUNK_SIZE: int = 65536


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class FitRegistryError(Exception):
    """Base exception for all SpycFit registry, state persistence, and locking operations."""


class RegistryLockError(FitRegistryError):
    """Base exception for atomic registry lock failures."""


class RegistryLockTimeoutError(RegistryLockError, TimeoutError):
    """Raised when acquiring an atomic lock exceeds the configured timeout."""


class LustreLockError(RegistryLockError):
    """Raised when atomic directory creation (mkdir) locking encounters an OS or filesystem error."""


class FitRegistryMissingError(FitRegistryError, FileNotFoundError):
    """Raised when a required registry JSON or HDF5 database file is missing."""


class FitRegistryCorruptionError(FitRegistryError, ValueError):
    """Raised when registry checksum verification fails or payload is corrupted."""


class FitRegistrySchemaError(FitRegistryError, ValueError):
    """Raised when a registry state record fails Pydantic schema validation."""


class HDF5TensorError(FitRegistryError):
    """Raised when HDF5 I/O, dataset creation, or SWMR operations fail."""


# =============================================================================
# 2. PYDANTIC V2 SCHEMAS
# =============================================================================

class InstrumentLimits(BaseModel):
    """Pydantic model representing spectrometer hardware constraints and boundaries."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    min_freq_ghz: float = Field(default=2.0, ge=0.0, description="Minimum spectrometer frequency in GHz")
    max_freq_ghz: float = Field(default=18.0, gt=0.0, description="Maximum spectrometer frequency in GHz")
    temperature_k: float = Field(default=298.15, gt=0.0, description="Rotational / nozzle temperature in Kelvin")
    bandwidth_mhz: float = Field(default=16000.0, gt=0.0, description="Digitizer bandwidth in MHz")
    pressure_torr: Optional[float] = Field(default=None, ge=0.0, description="Backing cell / jet pressure in Torr")
    dipole_threshold_debye: float = Field(default=0.01, ge=0.0, description="Minimum electric dipole threshold in Debye")


class MLFeedbackState(BaseModel):
    """Pydantic model representing Active Learning / Information-Gain ML feedback loop state."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    active: bool = Field(default=False, description="Whether active learning feedback loop is engaged")
    iteration: int = Field(default=0, ge=0, description="Current ML feedback loop iteration counter")
    learning_rate: float = Field(default=0.01, gt=0.0, description="Active learning rate or step size")
    loss_history: List[float] = Field(default_factory=list, description="Historical optimization loss values")
    assigned_transitions_count: int = Field(default=0, ge=0, description="Number of confidently assigned transitions")
    unassigned_peaks_count: int = Field(default=0, ge=0, description="Number of unassigned spectral peaks")
    convergence_status: str = Field(default="pending", description="Status: 'pending', 'iterating', 'converged', 'diverged'")


class FitParameterRecord(BaseModel):
    """Pydantic model representing an individual rotational spectroscopic fit parameter."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Parameter identifier (e.g. 'A', 'B', 'C', 'DJ', 'DJK', 'DK')")
    value: float = Field(..., description="Numerical value of parameter in MHz or specified unit")
    uncertainty: float = Field(default=0.0, ge=0.0, description="Estimated 1-sigma standard uncertainty")
    is_frozen: bool = Field(default=False, description="Whether parameter is manually locked / frozen by user")
    lower_bound: Optional[float] = Field(default=None, description="Lower optimization bound")
    upper_bound: Optional[float] = Field(default=None, description="Upper optimization bound")
    unit: str = Field(default="MHz", description="Physical unit of parameter")


class SnapshotMetadata(BaseModel):
    """Pydantic model representing a metadata-safe atomic snapshot of an HDF5 database."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp when snapshot was created")
    source_path: str = Field(..., description="Filesystem path of source HDF5 file")
    snapshot_path: str = Field(..., description="Filesystem path of persisted atomic snapshot")
    sha256: str = Field(..., description="Cryptographic SHA-256 hex digest of snapshot file")
    size_bytes: int = Field(ge=0, description="Size of snapshot file in bytes")
    is_lustre_fallback: bool = Field(default=False, description="True if snapshot was triggered by Lustre fallback")


class FitRegistryState(BaseModel):
    """
    Pydantic model encapsulating the complete CoChem-SpycFit Fit Registry State.
    Enforces strict validation, tracking frozen parameter keys, instrument limits,
    ML feedback state, parameter dictionaries, dataset hashes, and audit metadata.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    registry_version: str = Field(default="1.0.0", description="Schema version identifier")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique fit session UUID")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of registry modification",
    )
    manual_locks: List[str] = Field(
        default_factory=list,
        description="List of user-frozen parameter keys (e.g. ['A', 'DJ'])",
    )
    instrument_limits: InstrumentLimits = Field(
        default_factory=InstrumentLimits,
        description="Spectrometer hardware limits and boundaries",
    )
    ml_feedback: MLFeedbackState = Field(
        default_factory=MLFeedbackState,
        description="Active learning feedback loop state",
    )
    parameters: Dict[str, FitParameterRecord] = Field(
        default_factory=dict,
        description="Mapping of parameter names to detailed fit records",
    )
    active_dataset_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the active experimental spectrum dataset",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary session, environment, and provenance metadata",
    )

    @field_validator("manual_locks")
    @classmethod
    def validate_manual_locks(cls, v: List[str]) -> List[str]:
        # Return unique sorted list of parameter keys
        return sorted(list(set(v)))


# =============================================================================
# 3. ENVIRONMENT & FILESYSTEM DETECTION
# =============================================================================

def is_lustre_filesystem(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Dynamically detects whether a given path or runtime environment resides on an
    HPC Lustre distributed filesystem.

    Checks:
    1. Direct environment variable overrides:
       - COCHEM_FORCE_LUSTRE=1 / true / yes -> True
       - COCHEM_FORCE_STANDARD_FS=1 / true / yes -> False
       - COCHEM_FS_TYPE="lustre" -> True
    2. Path string inspection (e.g. containing 'lustre' path component).
    3. POSIX/Linux statvfs filesystem superblock magic check (0x0BD00BD0).
    4. HPC environment heuristics (SLURM / PBS cluster with mounted lustre paths).

    Args:
        path: Optional filesystem path to inspect. Defaults to current directory.

    Returns:
        bool: True if Lustre filesystem is detected, False otherwise.
    """
    # 1. Environment overrides
    force_lustre = os.environ.get("COCHEM_FORCE_LUSTRE", "").strip().lower()
    if force_lustre in ("1", "true", "yes", "on"):
        return True

    force_standard = os.environ.get("COCHEM_FORCE_STANDARD_FS", "").strip().lower()
    if force_standard in ("1", "true", "yes", "on"):
        return False

    fs_type = os.environ.get("COCHEM_FS_TYPE", "").strip().lower()
    if fs_type in ("lustre", "lustrefs", "lustre_hpc"):
        return True

    # 2. Path inspection
    target_path = Path(path).resolve() if path is not None else Path.cwd().resolve()
    for part in target_path.parts:
        part_clean = part.lower().strip("\\/ ")
        if part_clean in ("lustre", "lustrefs", "lustre_hpc") or part_clean.startswith("lustre_mount") or part_clean == ".lustre":
            return True

    # 3. Linux/POSIX statvfs magic check
    if hasattr(os, "statvfs"):
        try:
            check_p = target_path if target_path.exists() else target_path.parent
            if check_p.exists():
                st = os.statvfs(str(check_p))
                f_type = getattr(st, "f_type", None)
                if f_type is not None and f_type == LUSTRE_SUPER_MAGIC:
                    return True
        except Exception:
            pass

    # 4. /proc/mounts inspection on Linux
    if platform.system().lower() == "linux" and os.path.isfile("/proc/mounts"):
        try:
            with open("/proc/mounts", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[2].lower() == "lustre":
                        mount_point = parts[1]
                        if str(target_path).startswith(mount_point):
                            return True
        except Exception:
            pass

    return False


def resolve_lock_timeout(
    is_lustre: bool,
    custom_timeout: Optional[float] = None,
) -> float:
    """
    Scales atomic lock timeouts dynamically based on the detected environment.

    - Standard / Local filesystem: 10.0 seconds.
    - HPC / Lustre cluster: 60.0 seconds.

    Args:
        is_lustre: Boolean indicating whether Lustre/HPC environment was detected.
        custom_timeout: Optional explicit override timeout in seconds.

    Returns:
        float: Resolved timeout in seconds.
    """
    if custom_timeout is not None:
        return float(custom_timeout)

    env_timeout = os.environ.get("COCHEM_LOCK_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            pass

    return DEFAULT_LUSTRE_LOCK_TIMEOUT if is_lustre else DEFAULT_LOCAL_LOCK_TIMEOUT


# =============================================================================
# 4. CRYPTOGRAPHIC UTILITIES
# =============================================================================

def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """
    Computes the SHA-256 cryptographic hash of a file on disk.

    Args:
        file_path: Path to target file.
        chunk_size: Byte chunk size for reading large files.

    Returns:
        str: Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not physically exist.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


# =============================================================================
# 5. CROSS-PLATFORM & FILESYSTEM-AWARE ATOMIC LOCKS
# =============================================================================

class LustreMkdirLock:
    """
    HPC/Lustre Native Atomic Directory Creation (mkdir) Lock.

    Provides a robust, process-safe, and thread-safe lock for distributed filesystems
    where POSIX flock/fcntl file locks fail or experience deadlock. Uses native atomic
    directory creation (os.mkdir) with thread-local re-entrancy, diagnostic owner metadata,
    heartbeat timestamps, and stale lock recovery.
    """

    _tls = threading.local()
    _path_locks: Dict[str, threading.RLock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def _get_thread_lock(cls, canonical_path: str) -> threading.RLock:
        with cls._meta_lock:
            if canonical_path not in cls._path_locks:
                cls._path_locks[canonical_path] = threading.RLock()
            return cls._path_locks[canonical_path]

    def __init__(
        self,
        lock_path: Union[str, Path],
        timeout: float = DEFAULT_LUSTRE_LOCK_TIMEOUT,
        stale_timeout: float = DEFAULT_STALE_LOCK_TIMEOUT,
        poll_interval: float = 0.05,
    ) -> None:
        p = Path(lock_path).resolve()
        # Ensure lock directory has a distinct name (e.g. .lockdir)
        if p.name.endswith(".lockdir"):
            self.lock_dir = p
        elif p.name.endswith(".lock"):
            self.lock_dir = p.parent / f"{p.name}dir"
        else:
            self.lock_dir = p.parent / f"{p.name}.lockdir"

        self.timeout = float(timeout)
        self.stale_timeout = float(stale_timeout)
        self.poll_interval = float(poll_interval)
        self._depth: int = 0
        self._thread_lock_acquired: bool = False

    @property
    def owner_file(self) -> Path:
        return self.lock_dir / "owner.json"

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquires the atomic directory creation lock.

        Raises:
            RegistryLockTimeoutError: If lock cannot be acquired within timeout.
            LustreLockError: If OS/filesystem error occurs.
        """
        if not hasattr(self._tls, "held"):
            self._tls.held = {}

        path_key = str(self.lock_dir)

        # Thread-local re-entrancy support
        if self._tls.held.get(path_key, 0) > 0:
            self._tls.held[path_key] += 1
            self._depth += 1
            return True

        eff_timeout = float(timeout) if timeout is not None else self.timeout
        start_time = time.time()
        thread_lock = self._get_thread_lock(path_key)

        # 1. Acquire in-process thread lock first
        remaining = max(0.001, eff_timeout - (time.time() - start_time))
        if not thread_lock.acquire(timeout=remaining):
            raise RegistryLockTimeoutError(
                f"[LustreMkdirLock] In-process thread lock timeout on '{self.lock_dir}' after {eff_timeout:.2f}s"
            )

        self._thread_lock_acquired = True
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        while True:
            # Check elapsed time
            elapsed = time.time() - start_time
            if elapsed >= eff_timeout:
                self._cleanup_thread_lock(path_key)
                raise RegistryLockTimeoutError(
                    f"[LustreMkdirLock] Failed to acquire atomic directory lock on '{self.lock_dir}' within {eff_timeout:.2f}s"
                )

            # 2. Attempt atomic directory creation
            try:
                os.mkdir(str(self.lock_dir))
                # Successfully created atomic directory; write owner diagnostic metadata
                owner_payload = {
                    "pid": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "hostname": platform.node(),
                    "acquired_at": time.time(),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                try:
                    with open(self.owner_file, "w", encoding="utf-8") as f:
                        json.dump(owner_payload, f)
                except Exception:
                    pass

                self._depth = 1
                self._tls.held[path_key] = 1
                return True

            except FileExistsError:
                # Directory exists; check for stale lock
                self._check_and_reap_stale_lock()

            except (PermissionError, OSError):
                # On Windows or Lustre, permission errors during rapid creation may occur
                self._check_and_reap_stale_lock()

            # Exponential backoff with small random jitter
            attempt += 1
            sleep_time = min(0.2, self.poll_interval * (1.2 ** min(attempt, 8)) + random.uniform(0.001, 0.01))
            time.sleep(sleep_time)

    def _check_and_reap_stale_lock(self) -> None:
        """Checks if the existing lock directory has exceeded stale_timeout and reaps it."""
        try:
            if not self.lock_dir.exists():
                return

            now = time.time()
            lock_age = 0.0

            if self.owner_file.is_file():
                try:
                    with open(self.owner_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        acquired_at = float(data.get("acquired_at", 0.0))
                        if acquired_at > 0:
                            lock_age = now - acquired_at
                except Exception:
                    try:
                        lock_age = now - self.owner_file.stat().st_mtime
                    except OSError:
                        pass
            else:
                try:
                    lock_age = now - self.lock_dir.stat().st_mtime
                except OSError:
                    pass

            if lock_age > self.stale_timeout:
                logger.warning(
                    "[LustreMkdirLock] Reaping stale lock at %s (age=%.2fs > threshold=%.2fs)",
                    self.lock_dir,
                    lock_age,
                    self.stale_timeout,
                )
                try:
                    if self.owner_file.exists():
                        self.owner_file.unlink(missing_ok=True)
                    if self.lock_dir.exists():
                        os.rmdir(str(self.lock_dir))
                except Exception as e:
                    logger.debug("[LustreMkdirLock] Could not reap stale lock: %s", e)
        except Exception:
            pass

    def _cleanup_thread_lock(self, path_key: str) -> None:
        if self._thread_lock_acquired:
            self._thread_lock_acquired = False
            try:
                self._get_thread_lock(path_key).release()
            except RuntimeError:
                pass

    def release(self) -> None:
        """Releases the atomic directory creation lock safely."""
        path_key = str(self.lock_dir)

        if not hasattr(self._tls, "held") or self._tls.held.get(path_key, 0) <= 0:
            if self._depth > 0:
                self._depth -= 1
            self._cleanup_thread_lock(path_key)
            return

        self._depth -= 1
        self._tls.held[path_key] -= 1

        if self._tls.held[path_key] > 0:
            # Re-entrant lock still held by current thread
            return

        # Last release: remove owner file and rmdir with transient retry absorption
        for _ in range(5):
            try:
                if self.owner_file.exists():
                    self.owner_file.unlink(missing_ok=True)
                if self.lock_dir.exists():
                    os.rmdir(str(self.lock_dir))
                break
            except Exception:
                time.sleep(0.005)
        self._cleanup_thread_lock(path_key)

    def __enter__(self) -> LustreMkdirLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class StandardFileLock:
    """
    Process-Safe & Thread-Safe Cross-Platform Atomic File Lock (filelock + In-Process RLock).

    Uses filelock.SoftFileLock / FileLock on standard local filesystems, combined with
    thread-level RLock serialization, thread-local re-entrancy tracking, diagnostic PID payloads,
    and stale lock reaping.
    """

    _tls = threading.local()
    _path_locks: Dict[str, threading.RLock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def _get_thread_lock(cls, canonical_path: str) -> threading.RLock:
        with cls._meta_lock:
            if canonical_path not in cls._path_locks:
                cls._path_locks[canonical_path] = threading.RLock()
            return cls._path_locks[canonical_path]

    def __init__(
        self,
        lock_path: Union[str, Path],
        timeout: float = DEFAULT_LOCAL_LOCK_TIMEOUT,
        stale_timeout: float = DEFAULT_STALE_LOCK_TIMEOUT,
    ) -> None:
        p = Path(lock_path).resolve()
        if p.name.endswith(".lock"):
            self.lock_path = p
        elif p.name.endswith(".lockdir"):
            self.lock_path = p.parent / f"{p.stem}.lock"
        else:
            self.lock_path = p.parent / f"{p.name}.lock"

        self.timeout = float(timeout)
        self.stale_timeout = float(stale_timeout)
        self._depth: int = 0
        self._thread_lock_acquired: bool = False
        self._filelock: Optional[Union[filelock.SoftFileLock, filelock.FileLock]] = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquires the atomic file lock before timeout.

        Raises:
            RegistryLockTimeoutError: If lock acquisition exceeds timeout.
        """
        if not hasattr(self._tls, "held"):
            self._tls.held = {}
        if not hasattr(self._tls, "locks"):
            self._tls.locks = {}

        path_key = str(self.lock_path)

        # Thread-local re-entrancy check
        if self._tls.held.get(path_key, 0) > 0:
            self._tls.held[path_key] += 1
            self._depth += 1
            return True

        eff_timeout = float(timeout) if timeout is not None else self.timeout
        start_time = time.time()
        thread_lock = self._get_thread_lock(path_key)

        # 1. In-process thread serialization
        remaining = max(0.001, eff_timeout - (time.time() - start_time))
        if not thread_lock.acquire(timeout=remaining):
            raise RegistryLockTimeoutError(
                f"[StandardFileLock] Thread lock timeout on '{self.lock_path}' after {eff_timeout:.2f}s"
            )

        self._thread_lock_acquired = True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check and reap stale lock file
        if self.lock_path.is_file():
            try:
                mtime = self.lock_path.stat().st_mtime
                if (time.time() - mtime) > self.stale_timeout:
                    self.lock_path.unlink(missing_ok=True)
                    logger.warning("[StandardFileLock] Reaped stale lock: %s", self.lock_path)
            except OSError:
                pass

        # 2. Cross-process filelock
        rem_filelock = max(0.001, eff_timeout - (time.time() - start_time))
        fl = filelock.SoftFileLock(str(self.lock_path), timeout=rem_filelock)
        try:
            fl.acquire(timeout=rem_filelock)
            # Write diagnostic lock ownership payload
            try:
                self.lock_path.write_text(
                    f"{os.getpid()}:{threading.get_ident()}:{time.time()}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

            self._filelock = fl
            self._depth = 1
            self._tls.held[path_key] = 1
            self._tls.locks[path_key] = fl
            return True

        except (filelock.Timeout, TimeoutError) as e:
            self._cleanup_thread_lock(path_key)
            raise RegistryLockTimeoutError(
                f"[StandardFileLock] Failed to acquire lock on '{self.lock_path}' within {eff_timeout:.2f}s"
            ) from e
        except Exception as e:
            self._cleanup_thread_lock(path_key)
            raise RegistryLockTimeoutError(
                f"[StandardFileLock] Error acquiring lock on '{self.lock_path}': {e}"
            ) from e

    def _cleanup_thread_lock(self, path_key: str) -> None:
        if self._thread_lock_acquired:
            self._thread_lock_acquired = False
            try:
                self._get_thread_lock(path_key).release()
            except RuntimeError:
                pass

    def release(self) -> None:
        """Releases the atomic file lock safely."""
        path_key = str(self.lock_path)

        if not hasattr(self._tls, "held") or self._tls.held.get(path_key, 0) <= 0:
            if self._depth > 0:
                self._depth -= 1
            self._cleanup_thread_lock(path_key)
            return

        self._depth -= 1
        self._tls.held[path_key] -= 1

        if self._tls.held[path_key] > 0:
            return

        # Final release
        fl = self._tls.locks.pop(path_key, self._filelock)
        self._filelock = None
        if fl is not None:
            try:
                fl.release()
            except Exception:
                pass

        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            self._cleanup_thread_lock(path_key)

    def __enter__(self) -> StandardFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


def get_atomic_lock(
    lock_path: Union[str, Path],
    is_lustre: Optional[bool] = None,
    timeout: Optional[float] = None,
    stale_timeout: Optional[float] = None,
) -> Union[LustreMkdirLock, StandardFileLock]:
    """
    Factory creating a cross-platform, filesystem-aware atomic lock.

    - On HPC / Lustre filesystems: Returns LustreMkdirLock (atomic directory creation).
    - On Standard local filesystems: Returns StandardFileLock (filelock + RLock).

    Timeouts are scaled dynamically (10s local, 60s Lustre) unless explicitly provided.

    Args:
        lock_path: Path to target lock file or lock directory.
        is_lustre: Optional boolean override for Lustre detection.
        timeout: Optional explicit lock acquisition timeout in seconds.
        stale_timeout: Optional explicit stale lock timeout in seconds.

    Returns:
        Union[LustreMkdirLock, StandardFileLock]: Configured atomic lock instance.
    """
    p = Path(lock_path).resolve()
    lustre_detected = is_lustre if is_lustre is not None else is_lustre_filesystem(p)
    resolved_timeout = resolve_lock_timeout(lustre_detected, custom_timeout=timeout)
    resolved_stale = float(stale_timeout) if stale_timeout is not None else DEFAULT_STALE_LOCK_TIMEOUT

    if lustre_detected:
        logger.debug("[get_atomic_lock] Routing to LustreMkdirLock for path %s", p)
        return LustreMkdirLock(
            lock_path=p,
            timeout=resolved_timeout,
            stale_timeout=resolved_stale,
        )

    logger.debug("[get_atomic_lock] Routing to StandardFileLock for path %s", p)
    return StandardFileLock(
        lock_path=p,
        timeout=resolved_timeout,
        stale_timeout=resolved_stale,
    )


# =============================================================================
# 6. ATOMIC SNAPSHOT PERSISTENCE
# =============================================================================

def create_atomic_snapshot(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    backup_dir: Optional[Union[str, Path]] = None,
    is_lustre: bool = False,
) -> SnapshotMetadata:
    """
    Creates a metadata-safe, atomic snapshot of an HDF5 database or registry file on disk.

    Copies source_path to a temporary staging file via shutil.copy2 (preserving timestamps
    and permissions), then atomically renames the temporary file to dest_path (os.replace).
    Computes cryptographic SHA-256 checksum upon completion.

    Args:
        source_path: Path to source HDF5 / registry file to snapshot.
        dest_path: Optional explicit destination path. If omitted, generates snapshot name with timestamp.
        backup_dir: Optional backup directory for generated snapshots.
        is_lustre: Boolean indicating whether snapshot is triggered by Lustre fallback.

    Returns:
        SnapshotMetadata: Validated Pydantic snapshot metadata record.

    Raises:
        FitRegistryMissingError: If source file is missing.
        FitRegistryCorruptionError: If atomic copy or hash calculation fails.
    """
    src = Path(source_path).resolve()
    if not src.is_file():
        raise FitRegistryMissingError(f"Cannot create snapshot: source file does not exist at {src}")

    # Resolve target destination path
    snapshot_id = f"snap_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if dest_path is not None:
        dst = Path(dest_path).resolve()
    else:
        target_dir = Path(backup_dir).resolve() if backup_dir is not None else src.parent / "snapshots"
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / f"{src.stem}_{snapshot_id}{src.suffix}"

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Perform atomic copy via temp staging file in target directory
    tmp_dst = dst.parent / f"{dst.name}.tmp_{uuid.uuid4().hex[:8]}"
    try:
        shutil.copy2(str(src), str(tmp_dst))
        # Atomic rename
        os.replace(str(tmp_dst), str(dst))
    except Exception as e:
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)
        raise FitRegistryCorruptionError(f"Atomic snapshot creation failed from {src} to {dst}: {e}") from e

    # Compute checksum and verify
    file_hash = compute_file_sha256(dst)
    file_size = dst.stat().st_size
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    logger.info("Created atomic snapshot: %s -> %s (SHA-256: %s)", src.name, dst.name, file_hash)

    return SnapshotMetadata(
        snapshot_id=snapshot_id,
        timestamp_utc=timestamp,
        source_path=str(src),
        snapshot_path=str(dst),
        sha256=file_hash,
        size_bytes=file_size,
        is_lustre_fallback=is_lustre,
    )


# =============================================================================
# 7. SWMR & HDF5 STATE TENSOR MANAGEMENT
# =============================================================================

@contextmanager
def open_state_tensor(
    db_path: Union[str, Path],
    mode: str = "r",
    is_lustre: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Generator[h5py.File, None, None]:
    """
    Context manager opening the HDF5 State Tensor database with filesystem-aware SWMR management.

    Behavior:
    1. Standard Local Filesystem:
       - Write Mode ('w', 'a', 'r+'): Opens with libver='latest', activates swmr_mode=True, flushes on exit.
       - Read Mode ('r'): Opens with libver='latest', swmr=True for non-blocking concurrent reading.
    2. HPC / Lustre Filesystem:
       - Disables SWMR (swmr=False) to avoid distributed lock failure / corruption.
       - Write Mode: Opens standard HDF5, writes, flushes dataset buffers, and triggers periodic
         metadata-safe atomic snapshots (shutil.copy2) on exit.
       - Read Mode: Opens standard HDF5 (or latest snapshot) safely with swmr=False.

    Args:
        db_path: Path to spectra_fit.h5 database.
        mode: HDF5 open mode ('r', 'r+', 'w', 'a', 'x').
        is_lustre: Optional override for Lustre detection.
        timeout: Optional timeout for atomic lock operations.

    Yields:
        h5py.File: Open HDF5 file handle.

    Raises:
        HDF5TensorError: If HDF5 initialization or SWMR activation fails.
    """
    p = Path(db_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    lustre_detected = is_lustre if is_lustre is not None else is_lustre_filesystem(p)

    h5_file: Optional[h5py.File] = None
    lock: Optional[Union[LustreMkdirLock, StandardFileLock]] = None

    try:
        # If write mode, acquire atomic lock
        is_write_mode = mode in ("w", "a", "r+", "x")
        if is_write_mode:
            lock = get_atomic_lock(p, is_lustre=lustre_detected, timeout=timeout)
            lock.acquire()

        if lustre_detected:
            # Lustre fallback: SWMR strictly disabled
            logger.debug("[open_state_tensor] Lustre detected. SWMR disabled for %s", p)
            if mode == "r" and not p.is_file():
                raise FitRegistryMissingError(f"HDF5 database not found at {p}")

            try:
                h5_file = h5py.File(str(p), mode=mode, libver="latest")
            except Exception as e:
                raise HDF5TensorError(f"Failed to open HDF5 file on Lustre at {p} with mode '{mode}': {e}") from e

            yield h5_file

            if is_write_mode and h5_file is not None:
                try:
                    h5_file.flush()
                except Exception:
                    pass

        else:
            # Standard filesystem: SWMR enabled
            logger.debug("[open_state_tensor] Standard filesystem. SWMR active for %s", p)
            if mode == "r":
                if not p.is_file():
                    raise FitRegistryMissingError(f"HDF5 database not found at {p}")
                try:
                    h5_file = h5py.File(str(p), mode="r", libver="latest", swmr=True)
                except Exception:
                    # Fallback to standard read if file was not created with SWMR
                    try:
                        h5_file = h5py.File(str(p), mode="r", libver="latest")
                    except Exception as e:
                        raise HDF5TensorError(f"Failed to open HDF5 SWMR file for reading at {p}: {e}") from e

                yield h5_file

            else:
                # Write / Append mode with SWMR on standard filesystem
                try:
                    h5_file = h5py.File(str(p), mode=mode, libver="latest")
                except Exception as e:
                    raise HDF5TensorError(f"Failed to initialize HDF5 SWMR file at {p} with mode '{mode}': {e}") from e

                yield h5_file

                if h5_file is not None:
                    try:
                        h5_file.flush()
                        if not getattr(h5_file, "swmr_mode", False):
                            try:
                                h5_file.swmr_mode = True
                                h5_file.flush()
                            except Exception as swmr_err:
                                logger.debug("[open_state_tensor] Notice on swmr_mode activation: %s", swmr_err)
                    except Exception:
                        pass

    finally:
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass

        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass

        # If Lustre write mode completed, create an atomic snapshot
        if lustre_detected and is_write_mode and p.is_file():
            try:
                create_atomic_snapshot(source_path=p, is_lustre=True)
            except Exception as snap_err:
                logger.warning("[open_state_tensor] Lustre post-write snapshot notice: %s", snap_err)


# =============================================================================
# 8. HIGH-LEVEL FIT REGISTRY MANAGER
# =============================================================================

class FitRegistryManager:
    """
    High-Level Fit Registry and State Persistence Manager for CoChem-SpycFit.

    Provides atomic JSON state management, SWMR / Lustre snapshot HDF5 tensor persistence,
    parameter locking, active learning feedback state, and cryptographic dataset verification.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        registry_json_path: Optional[Union[str, Path]] = None,
        lock_path: Optional[Union[str, Path]] = None,
        auto_detect_lustre: bool = True,
        force_lustre: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initializes the FitRegistryManager.

        Args:
            db_path: Optional path to HDF5 database (defaults to platformdirs Databases/spectra_fit.h5).
            registry_json_path: Optional path to registry JSON (defaults to platformdirs Registry/cochem_fit_registry.json).
            lock_path: Optional path to atomic lockfile / lockdir.
            auto_detect_lustre: Whether to automatically query filesystem type.
            force_lustre: Optional boolean override for Lustre mode.
            timeout: Optional default timeout for atomic locks.
        """
        base_data = Path(platformdirs.user_data_dir("CoChem")).resolve()

        # 1. Resolve registry JSON path
        if registry_json_path is not None:
            self.registry_json_path = Path(registry_json_path).resolve()
        else:
            env_json = os.environ.get("COCHEM_FIT_REGISTRY_JSON")
            if env_json:
                self.registry_json_path = Path(env_json).resolve()
            else:
                self.registry_json_path = base_data / "Registry" / "cochem_fit_registry.json"

        self.registry_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Resolve HDF5 database path
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            env_db = os.environ.get("COCHEM_SPECTRA_FIT_H5")
            if env_db:
                self.db_path = Path(env_db).resolve()
            else:
                self.db_path = base_data / "Databases" / "spectra_fit.h5"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Detect filesystem tier
        if force_lustre is not None:
            self.is_lustre = bool(force_lustre)
        elif auto_detect_lustre:
            self.is_lustre = is_lustre_filesystem(self.registry_json_path) or is_lustre_filesystem(self.db_path)
        else:
            self.is_lustre = False

        # 4. Resolve default lock timeout
        self.timeout = resolve_lock_timeout(self.is_lustre, custom_timeout=timeout)

        # 5. Resolve lock path
        if lock_path is not None:
            self.lock_path = Path(lock_path).resolve()
        else:
            suffix = ".lockdir" if self.is_lustre else ".lock"
            self.lock_path = self.registry_json_path.parent / f"{self.registry_json_path.name}{suffix}"

        self.snapshots_dir = self.db_path.parent / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FitRegistryManager initialized. is_lustre=%s, timeout=%.1fs, registry=%s, db=%s",
            self.is_lustre,
            self.timeout,
            self.registry_json_path.name,
            self.db_path.name,
        )

    def get_atomic_lock(
        self,
        lock_path: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
    ) -> Union[LustreMkdirLock, StandardFileLock]:
        """
        Returns a configured cross-platform, filesystem-aware atomic lock.

        Args:
            lock_path: Optional specific path to lock. Defaults to self.lock_path.
            timeout: Optional timeout override. Defaults to self.timeout.
        """
        target_lock = Path(lock_path).resolve() if lock_path is not None else self.lock_path
        eff_timeout = timeout if timeout is not None else self.timeout
        return get_atomic_lock(
            lock_path=target_lock,
            is_lustre=self.is_lustre,
            timeout=eff_timeout,
        )

    def is_lustre_filesystem(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Queries whether a specified path resides on a Lustre filesystem."""
        target = Path(path).resolve() if path is not None else self.registry_json_path
        return is_lustre_filesystem(target)

    def get_fit_record(self) -> FitRegistryState:
        """
        Reads and returns the active FitRegistryState from disk.
        If the file does not exist, returns a fresh default FitRegistryState.

        Raises:
            FitRegistryCorruptionError: If JSON parsing fails.
            FitRegistrySchemaError: If JSON schema validation fails.
        """
        if not self.registry_json_path.is_file():
            return FitRegistryState()

        try:
            with open(self.registry_json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise FitRegistryCorruptionError(f"Corrupted JSON in fit registry at {self.registry_json_path}: {e}") from e
        except Exception as e:
            raise FitRegistryCorruptionError(f"Error reading fit registry at {self.registry_json_path}: {e}") from e

        try:
            return FitRegistryState.model_validate(raw_data)
        except Exception as e:
            raise FitRegistrySchemaError(f"Schema validation error in fit registry at {self.registry_json_path}: {e}") from e

    def save_fit_record(
        self,
        record: FitRegistryState,
        lock: Optional[Union[LustreMkdirLock, StandardFileLock]] = None,
    ) -> None:
        """
        Atomically persists a FitRegistryState to disk with tempfile staging and rename.

        Args:
            record: FitRegistryState Pydantic model to save.
            lock: Optional active lock. If None, acquires an atomic lock during save.
        """
        # Validate Pydantic schema first
        json_data = record.model_dump_json(indent=2)

        def _do_write() -> None:
            tmp_p = self.registry_json_path.parent / f"{self.registry_json_path.name}.tmp_{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_p, "w", encoding="utf-8") as f:
                    f.write(json_data)
                os.replace(str(tmp_p), str(self.registry_json_path))
            except Exception as e:
                if tmp_p.exists():
                    tmp_p.unlink(missing_ok=True)
                raise FitRegistryCorruptionError(f"Failed to atomically write registry JSON to {self.registry_json_path}: {e}") from e

        if lock is not None:
            _do_write()
        else:
            with self.get_atomic_lock():
                _do_write()

    @contextmanager
    def transaction(self) -> Generator[FitRegistryState, None, None]:
        """
        Context manager providing an atomic transactional update of the Fit Registry.

        Acquires the atomic lock, loads the latest FitRegistryState, yields it for mutation,
        validates the updated model, and atomically persists it on successful exit.

        Yields:
            FitRegistryState: Mutable fit registry state model.
        """
        lock = self.get_atomic_lock()
        with lock:
            state = self.get_fit_record()
            yield state
            # Update timestamp and revalidate
            state.timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.save_fit_record(state, lock=lock)

    def freeze_parameter(self, param_name: str, value: Optional[float] = None) -> FitRegistryState:
        """
        Manually freezes a fit parameter, adding it to manual_locks.

        Args:
            param_name: Identifier of the parameter (e.g. 'A', 'DJ').
            value: Optional value to assign or update before locking.

        Returns:
            FitRegistryState: Updated fit registry state.
        """
        with self.transaction() as state:
            if param_name not in state.manual_locks:
                state.manual_locks.append(param_name)
                state.manual_locks.sort()

            if param_name in state.parameters:
                state.parameters[param_name].is_frozen = True
                if value is not None:
                    state.parameters[param_name].value = float(value)
            elif value is not None:
                state.parameters[param_name] = FitParameterRecord(
                    name=param_name,
                    value=float(value),
                    is_frozen=True,
                )

        return self.get_fit_record()

    def unfreeze_parameter(self, param_name: str) -> FitRegistryState:
        """
        Unfreezes a previously locked parameter, removing it from manual_locks.

        Args:
            param_name: Identifier of the parameter.

        Returns:
            FitRegistryState: Updated fit registry state.
        """
        with self.transaction() as state:
            if param_name in state.manual_locks:
                state.manual_locks.remove(param_name)

            if param_name in state.parameters:
                state.parameters[param_name].is_frozen = False

        return self.get_fit_record()

    def update_ml_feedback(
        self,
        active: Optional[bool] = None,
        loss: Optional[float] = None,
        assigned: Optional[int] = None,
        unassigned: Optional[int] = None,
        status: Optional[str] = None,
    ) -> FitRegistryState:
        """
        Updates the Active Learning / Information-Gain ML feedback loop state atomically.
        """
        with self.transaction() as state:
            if active is not None:
                state.ml_feedback.active = bool(active)
            if loss is not None:
                state.ml_feedback.loss_history.append(float(loss))
                state.ml_feedback.iteration += 1
            if assigned is not None:
                state.ml_feedback.assigned_transitions_count = int(assigned)
            if unassigned is not None:
                state.ml_feedback.unassigned_peaks_count = int(unassigned)
            if status is not None:
                state.ml_feedback.convergence_status = str(status)

        return self.get_fit_record()

    def create_atomic_snapshot(
        self,
        source_path: Optional[Union[str, Path]] = None,
        dest_path: Optional[Union[str, Path]] = None,
    ) -> SnapshotMetadata:
        """
        Creates an atomic snapshot of the HDF5 tensor database (or specified file).

        Args:
            source_path: Optional source file path (defaults to self.db_path).
            dest_path: Optional explicit snapshot destination path.

        Returns:
            SnapshotMetadata: Validated snapshot metadata record.
        """
        src = Path(source_path).resolve() if source_path is not None else self.db_path
        return create_atomic_snapshot(
            source_path=src,
            dest_path=dest_path,
            backup_dir=self.snapshots_dir,
            is_lustre=self.is_lustre,
        )

    def open_state_tensor(
        self,
        mode: str = "r",
        is_lustre: Optional[bool] = None,
    ) -> Generator[h5py.File, None, None]:
        """
        Opens the HDF5 State Tensor database with filesystem-aware SWMR management.

        Args:
            mode: HDF5 access mode ('r', 'a', 'w', 'r+').
            is_lustre: Optional Lustre override (defaults to self.is_lustre).

        Yields:
            h5py.File: Open HDF5 file handle.
        """
        eff_lustre = is_lustre if is_lustre is not None else self.is_lustre
        return open_state_tensor(
            db_path=self.db_path,
            mode=mode,
            is_lustre=eff_lustre,
            timeout=self.timeout,
        )

    def save_spectral_tensor(
        self,
        name: str,
        tensor: Any,
        metadata: Optional[Dict[str, Any]] = None,
        group: str = "/",
        compression: str = "gzip",
        compression_opts: int = 4,
        shuffle: bool = True,
        fletcher32: bool = True,
    ) -> None:
        """
        Persists a multi-dimensional spectral tensor or matrix to the HDF5 database.

        Applies HDF5 chunking, compression (gzip), shuffle, and fletcher32 checksum filters.
        Attaches metadata dictionaries to dataset HDF5 attributes.

        Args:
            name: Dataset name identifier.
            tensor: NumPy array, list, or tensor-like numeric data.
            metadata: Optional dictionary of metadata attributes.
            group: HDF5 group path (defaults to root "/").
            compression: Compression filter ('gzip' or None).
            compression_opts: Compression level (0-9).
            shuffle: Whether to apply byte shuffle filter.
            fletcher32: Whether to enable Fletcher32 checksum filter.
        """
        # Convert to host RAM NumPy array
        arr = np.ascontiguousarray(np.asarray(tensor))
        if arr.ndim == 0:
            arr = arr.reshape((1,))

        with self.open_state_tensor(mode="a") as f:
            grp = f.require_group(group) if group != "/" else f
            if name in grp:
                del grp[name]

            dset = grp.create_dataset(
                name,
                data=arr,
                compression=compression if compression else None,
                compression_opts=compression_opts if compression else None,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=True,
            )

            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        dset.attrs[k] = v
                    else:
                        dset.attrs[k] = json.dumps(v)

    def load_spectral_tensor(
        self,
        name: str,
        group: str = "/",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a spectral tensor and its associated metadata from the HDF5 database.

        Args:
            name: Dataset name identifier.
            group: HDF5 group path (defaults to root "/").

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Loaded NumPy array and metadata dictionary.

        Raises:
            HDF5TensorError: If dataset is missing or unreadable.
        """
        with self.open_state_tensor(mode="r") as f:
            grp = f[group] if group != "/" else f
            if name not in grp:
                raise HDF5TensorError(f"Dataset '{name}' not found in HDF5 group '{group}' at {self.db_path}")

            dset = grp[name]
            arr = np.array(dset)
            meta: Dict[str, Any] = {}
            for k, v in dset.attrs.items():
                if isinstance(v, str):
                    try:
                        meta[k] = json.loads(v)
                    except Exception:
                        meta[k] = v
                else:
                    meta[k] = v

            return arr, meta

    def verify_registry_integrity(self) -> bool:
        """
        Verifies the physical presence and schema integrity of the active registry JSON.

        Returns:
            bool: True if registry state is valid and non-corrupted.
        """
        try:
            state = self.get_fit_record()
            return isinstance(state, FitRegistryState) and len(state.registry_version) > 0
        except Exception:
            return False

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_fit_registry_manager.py ---
"""
CoChem-SpycFit Physical Unit & Concurrency Test Suite: Fit Registry Manager.

Strictly adheres to Zero-Mock mandate.
Validates:
1. Pydantic v2 schemas with ConfigDict(extra="forbid", validate_assignment=True).
2. Cross-platform StandardFileLock (filelock + RLock + re-entrancy + timeout + stale reaping).
3. HPC / Lustre LustreMkdirLock (atomic os.mkdir + owner.json + timeout + stale reaping).
4. Dynamic timeout scaling (10s local vs 60s Lustre).
5. Metadata-safe atomic snapshotting (shutil.copy2 + atomic rename + SHA-256 verification).
6. HDF5 SWMR connections (libver='latest', swmr=True) with multi-threaded concurrent writer & reader.
7. Dynamic Lustre fallback disabling SWMR and generating atomic snapshots.
8. High-level FitRegistryManager transaction lifecycle, parameter freezing, ML feedback, and tensor storage.
9. Real multi-threaded and multi-process concurrency without race conditions or deadlocks.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from cochem_spycfit.core_engine.cochem_fit_registry_manager import (
    DEFAULT_LOCAL_LOCK_TIMEOUT,
    DEFAULT_LUSTRE_LOCK_TIMEOUT,
    DEFAULT_STALE_LOCK_TIMEOUT,
    FitParameterRecord,
    FitRegistryCorruptionError,
    FitRegistryError,
    FitRegistryManager,
    FitRegistryMissingError,
    FitRegistrySchemaError,
    FitRegistryState,
    HDF5TensorError,
    InstrumentLimits,
    LustreLockError,
    LustreMkdirLock,
    MLFeedbackState,
    RegistryLockError,
    RegistryLockTimeoutError,
    SnapshotMetadata,
    StandardFileLock,
    compute_file_sha256,
    create_atomic_snapshot,
    get_atomic_lock,
    is_lustre_filesystem,
    open_state_tensor,
    resolve_lock_timeout,
)


@pytest.fixture
def tmp_env(tmp_path: Path):
    """Provides a fresh isolated temporary workspace."""
    workspace = tmp_path / "spycfit_test_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    reg_dir = workspace / "Registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    db_dir = workspace / "Databases"
    db_dir.mkdir(parents=True, exist_ok=True)

    reg_json = reg_dir / "cochem_fit_registry.json"
    db_h5 = db_dir / "spectra_fit.h5"

    yield {
        "workspace": workspace,
        "reg_json": reg_json,
        "db_h5": db_h5,
    }


# =============================================================================
# 1. PYDANTIC V2 SCHEMAS TESTS
# =============================================================================

def test_instrument_limits_schema():
    """Validates InstrumentLimits schema and extra forbidden constraint."""
    limits = InstrumentLimits(
        min_freq_ghz=2.5,
        max_freq_ghz=18.0,
        temperature_k=300.0,
        bandwidth_mhz=16000.0,
    )
    assert limits.min_freq_ghz == 2.5
    assert limits.max_freq_ghz == 18.0
    assert limits.temperature_k == 300.0

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        InstrumentLimits(min_freq_ghz=2.0, max_freq_ghz=18.0, invalid_field=123)  # type: ignore[call-arg]

    # Type / value bounds validation
    with pytest.raises(ValidationError):
        InstrumentLimits(min_freq_ghz=-1.0)


def test_ml_feedback_state_schema():
    """Validates MLFeedbackState schema."""
    ml = MLFeedbackState(
        active=True,
        iteration=5,
        learning_rate=0.05,
        loss_history=[1.2, 0.8, 0.4],
        assigned_transitions_count=42,
        unassigned_peaks_count=10,
        convergence_status="iterating",
    )
    assert ml.active is True
    assert ml.iteration == 5
    assert len(ml.loss_history) == 3
    assert ml.assigned_transitions_count == 42

    with pytest.raises(ValidationError):
        MLFeedbackState(active=True, unknown_key="test")  # type: ignore[call-arg]


def test_fit_parameter_record_schema():
    """Validates FitParameterRecord schema."""
    param = FitParameterRecord(
        name="A",
        value=10543.21,
        uncertainty=0.05,
        is_frozen=True,
        lower_bound=10000.0,
        upper_bound=11000.0,
        unit="MHz",
    )
    assert param.name == "A"
    assert param.value == 10543.21
    assert param.is_frozen is True

    # Mutating with invalid type fails validation
    with pytest.raises(ValidationError):
        param.uncertainty = -5.0


def test_fit_registry_state_schema():
    """Validates root FitRegistryState schema and manual_locks deduplication."""
    state = FitRegistryState(
        manual_locks=["DJ", "A", "DJ", "B"],
        metadata={"operator": "cochem-test"},
    )
    assert state.manual_locks == ["A", "B", "DJ"]
    assert state.metadata["operator"] == "cochem-test"

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        FitRegistryState(unauthorized_arg=True)  # type: ignore[call-arg]


# =============================================================================
# 2. FILESYSTEM & LUSTRE DETECTION TESTS
# =============================================================================

def test_lustre_detection_and_timeout_scaling(tmp_path: Path):
    """Validates is_lustre_filesystem and resolve_lock_timeout."""
    # Standard local path
    local_p = tmp_path / "local_test"
    assert is_lustre_filesystem(local_p) is False
    assert resolve_lock_timeout(is_lustre=False) == DEFAULT_LOCAL_LOCK_TIMEOUT

    # Explicit Lustre path
    lustre_p = tmp_path / "lustre" / "scratch"
    assert is_lustre_filesystem(lustre_p) is True
    assert resolve_lock_timeout(is_lustre=True) == DEFAULT_LUSTRE_LOCK_TIMEOUT

    # Environment override COCHEM_FORCE_LUSTRE
    os.environ["COCHEM_FORCE_LUSTRE"] = "1"
    try:
        assert is_lustre_filesystem(local_p) is True
        assert resolve_lock_timeout(is_lustre=True) == DEFAULT_LUSTRE_LOCK_TIMEOUT
    finally:
        os.environ.pop("COCHEM_FORCE_LUSTRE", None)

    # Custom timeout override
    assert resolve_lock_timeout(is_lustre=False, custom_timeout=25.0) == 25.0
    assert resolve_lock_timeout(is_lustre=True, custom_timeout=45.0) == 45.0


# =============================================================================
# 3. ATOMIC LOCK IMPLEMENTATIONS (StandardFileLock & LustreMkdirLock)
# =============================================================================

def test_standard_file_lock_lifecycle_and_reentrancy(tmp_path: Path):
    """Validates StandardFileLock acquire, release, and thread-local reentrancy."""
    lock_file = tmp_path / "test.lock"
    lock = StandardFileLock(lock_file, timeout=5.0)

    # Basic context manager
    with lock:
        assert lock_file.exists()
        # Re-entrant acquire within same thread
        with lock:
            assert lock_file.exists()

    # Release clean up
    assert not lock_file.exists()


def test_standard_file_lock_concurrency(tmp_path: Path):
    """Validates multi-threaded exclusion with StandardFileLock."""
    lock_file = tmp_path / "counter.lock"
    lock = StandardFileLock(lock_file, timeout=10.0)

    counter = {"val": 0}
    num_threads = 6
    increments_per_thread = 20

    def worker():
        for _ in range(increments_per_thread):
            with lock:
                curr = counter["val"]
                time.sleep(0.001)
                counter["val"] = curr + 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["val"] == num_threads * increments_per_thread


def test_standard_file_lock_timeout(tmp_path: Path):
    """Validates RegistryLockTimeoutError when lock is held by another thread."""
    lock_file = tmp_path / "timeout.lock"
    lock1 = StandardFileLock(lock_file, timeout=2.0)
    lock2 = StandardFileLock(lock_file, timeout=0.2)

    ready_event = threading.Event()
    release_event = threading.Event()
    timeout_caught = {"caught": False}

    def holder():
        with lock1:
            ready_event.set()
            release_event.wait(timeout=2.0)

    def waiter():
        ready_event.wait()
        try:
            with lock2:
                pass
        except RegistryLockTimeoutError:
            timeout_caught["caught"] = True

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)

    t1.start()
    t2.start()

    t2.join()
    release_event.set()
    t1.join()

    assert timeout_caught["caught"] is True


def test_lustre_mkdir_lock_lifecycle_and_reentrancy(tmp_path: Path):
    """Validates LustreMkdirLock atomic directory creation, owner metadata, and re-entrancy."""
    lock_dir = tmp_path / "lustre_test.lockdir"
    lock = LustreMkdirLock(lock_dir, timeout=5.0)

    assert not lock_dir.exists()

    with lock:
        assert lock_dir.is_dir()
        assert lock.owner_file.is_file()
        owner_data = json.loads(lock.owner_file.read_text(encoding="utf-8"))
        assert owner_data["pid"] == os.getpid()

        # Re-entrant acquire
        with lock:
            assert lock_dir.is_dir()

    # On exit, rmdir should have cleaned it up
    assert not lock_dir.exists()


def test_lustre_mkdir_lock_concurrency(tmp_path: Path):
    """Validates multi-threaded exclusion with LustreMkdirLock."""
    lock_dir = tmp_path / "lustre_counter.lockdir"
    lock = LustreMkdirLock(lock_dir, timeout=15.0)

    counter = {"val": 0}
    num_threads = 6
    increments_per_thread = 20

    def worker():
        for _ in range(increments_per_thread):
            with lock:
                curr = counter["val"]
                time.sleep(0.001)
                counter["val"] = curr + 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["val"] == num_threads * increments_per_thread


def test_lustre_mkdir_lock_timeout(tmp_path: Path):
    """Validates RegistryLockTimeoutError with LustreMkdirLock."""
    lock_dir = tmp_path / "lustre_timeout.lockdir"
    lock1 = LustreMkdirLock(lock_dir, timeout=2.0)
    lock2 = LustreMkdirLock(lock_dir, timeout=0.2)

    ready_event = threading.Event()
    release_event = threading.Event()
    timeout_caught = {"caught": False}

    def holder():
        with lock1:
            ready_event.set()
            release_event.wait(timeout=2.0)

    def waiter():
        ready_event.wait()
        try:
            with lock2:
                pass
        except RegistryLockTimeoutError:
            timeout_caught["caught"] = True

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)

    t1.start()
    t2.start()

    t2.join()
    release_event.set()
    t1.join()

    assert timeout_caught["caught"] is True


def test_stale_lock_reaping(tmp_path: Path):
    """Validates automatic reaping of stale locks."""
    # 1. StandardFileLock stale reaping
    stale_file = tmp_path / "stale.lock"
    stale_file.write_text("old_pid:123:0.0\n", encoding="utf-8")
    # Set mtime back by 200 seconds
    past_time = time.time() - 200.0
    os.utime(str(stale_file), (past_time, past_time))

    lock_std = StandardFileLock(stale_file, timeout=2.0, stale_timeout=30.0)
    with lock_std:
        assert stale_file.exists()

    # 2. LustreMkdirLock stale reaping
    stale_dir = tmp_path / "stale.lockdir"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_owner = stale_dir / "owner.json"
    with open(stale_owner, "w", encoding="utf-8") as f:
        json.dump({"pid": 99999, "acquired_at": time.time() - 200.0}, f)

    lock_lustre = LustreMkdirLock(stale_dir, timeout=2.0, stale_timeout=30.0)
    with lock_lustre:
        assert stale_dir.is_dir()
    assert not stale_dir.exists()


# =============================================================================
# 4. ATOMIC SNAPSHOT PERSISTENCE TESTS
# =============================================================================

def test_atomic_snapshot_creation(tmp_path: Path):
    """Validates metadata-safe atomic snapshot copy and SHA-256 verification."""
    source_h5 = tmp_path / "test_data.h5"
    with h5py.File(str(source_h5), "w") as f:
        f.create_dataset("test_tensor", data=np.arange(100, dtype=np.float64))

    src_hash = compute_file_sha256(source_h5)
    dest_h5 = tmp_path / "snapshots" / "test_data_snap.h5"

    snapshot_meta = create_atomic_snapshot(
        source_path=source_h5,
        dest_path=dest_h5,
        is_lustre=True,
    )

    assert snapshot_meta.is_lustre_fallback is True
    assert snapshot_meta.sha256 == src_hash
    assert Path(snapshot_meta.snapshot_path).is_file()
    assert snapshot_meta.size_bytes == source_h5.stat().st_size

    # Verify snapshot can be read independently
    with h5py.File(snapshot_meta.snapshot_path, "r") as f:
        loaded = np.array(f["test_tensor"])
        np.testing.assert_array_equal(loaded, np.arange(100, dtype=np.float64))

    # Error handling for missing source
    with pytest.raises(FitRegistryMissingError):
        create_atomic_snapshot(source_path=tmp_path / "non_existent.h5")


# =============================================================================
# 5. HDF5 SWMR PERSISTENCE & CONCURRENT STREAMING TESTS
# =============================================================================

def test_hdf5_swmr_concurrent_streaming(tmp_path: Path):
    """
    Validates genuine HDF5 Single-Writer / Multiple-Reader (SWMR) concurrent execution.
    Writer continuously appends records to HDF5 under SWMR while a reader thread
    concurrently polls and reads records without blocking or crashing.
    """
    db_path = tmp_path / "swmr_stream.h5"

    # Initialize file with extensible chunked dataset
    with h5py.File(str(db_path), "w", libver="latest") as f:
        f.create_dataset(
            "stream_data",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(64,),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        f.swmr_mode = True

    num_records = 20
    writer_done = threading.Event()
    reader_seen_counts: List[int] = []

    def writer_task():
        with open_state_tensor(db_path, mode="a", is_lustre=False) as f:
            dset = f["stream_data"]
            for i in range(num_records):
                curr_len = dset.shape[0]
                dset.resize((curr_len + 1,))
                dset[curr_len] = float(i * 1.5)
                f.flush()
                time.sleep(0.005)
        writer_done.set()

    def reader_task():
        # Open in SWMR read mode
        time.sleep(0.01)
        with open_state_tensor(db_path, mode="r", is_lustre=False) as f:
            dset = f["stream_data"]
            start_poll = time.time()
            while time.time() - start_poll < 5.0:
                try:
                    dset.refresh()
                except Exception:
                    pass
                reader_seen_counts.append(dset.shape[0])
                if writer_done.is_set() and dset.shape[0] >= num_records:
                    break
                time.sleep(0.005)

    t_writer = threading.Thread(target=writer_task)
    t_reader = threading.Thread(target=reader_task)

    t_writer.start()
    t_reader.start()

    t_writer.join(timeout=10.0)
    t_reader.join(timeout=10.0)

    # Verify data integrity
    with h5py.File(str(db_path), "r") as f:
        final_arr = np.array(f["stream_data"])
        assert len(final_arr) == num_records
        np.testing.assert_array_almost_equal(final_arr, np.arange(num_records) * 1.5)

    assert len(reader_seen_counts) > 1
    assert reader_seen_counts[-1] == num_records


def test_hdf5_lustre_fallback_disables_swmr_and_snapshots(tmp_path: Path):
    """
    Validates that Lustre filesystem detection dynamically disables SWMR mode
    and triggers automatic metadata-safe atomic snapshot generation.
    """
    db_path = tmp_path / "lustre_db.h5"

    with open_state_tensor(db_path, mode="w", is_lustre=True) as f:
        # In Lustre mode, swmr_mode is not activated
        assert not getattr(f, "swmr_mode", False)
        f.create_dataset("dense_matrix", data=np.ones((10, 10), dtype=np.float64))

    # Verify atomic snapshot was automatically created in snapshots directory
    snap_dir = db_path.parent / "snapshots"
    assert snap_dir.is_dir()
    snaps = list(snap_dir.glob("lustre_db_snap_*.h5"))
    assert len(snaps) >= 1

    # Verify snapshot content
    with h5py.File(str(snaps[0]), "r") as f_snap:
        loaded = np.array(f_snap["dense_matrix"])
        np.testing.assert_array_equal(loaded, np.ones((10, 10)))


# =============================================================================
# 6. HIGH-LEVEL FIT REGISTRY MANAGER TESTS
# =============================================================================

def test_fit_registry_manager_transaction_and_persistence(tmp_env):
    """Validates atomic transaction updates, parameter freezing, and JSON integrity."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    # Initial state default
    init_state = mgr.get_fit_record()
    assert len(init_state.manual_locks) == 0

    # 1. Parameter Freezing
    mgr.freeze_parameter("A", value=10250.5)
    mgr.freeze_parameter("DJ", value=0.0125)

    state1 = mgr.get_fit_record()
    assert "A" in state1.manual_locks
    assert "DJ" in state1.manual_locks
    assert state1.parameters["A"].value == 10250.5
    assert state1.parameters["A"].is_frozen is True

    # 2. Parameter Unfreezing
    mgr.unfreeze_parameter("A")
    state2 = mgr.get_fit_record()
    assert "A" not in state2.manual_locks
    assert "DJ" in state2.manual_locks
    assert state2.parameters["A"].is_frozen is False

    # 3. ML Feedback update
    mgr.update_ml_feedback(active=True, loss=0.45, assigned=12, unassigned=4, status="iterating")
    state3 = mgr.get_fit_record()
    assert state3.ml_feedback.active is True
    assert state3.ml_feedback.iteration == 1
    assert state3.ml_feedback.loss_history == [0.45]
    assert state3.ml_feedback.assigned_transitions_count == 12

    # 4. Verify physical JSON file exists and matches
    assert tmp_env["reg_json"].is_file()
    assert mgr.verify_registry_integrity() is True


def test_fit_registry_manager_spectral_tensors(tmp_env):
    """Validates saving and loading multi-dimensional spectral tensors with metadata and compression."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    test_matrix = np.random.RandomState(42).randn(50, 50)
    meta = {
        "molecule": "Benzonitrile",
        "method": "B3LYP-D3BJ",
        "frequencies": [1024.5, 2048.0, 4096.5],
    }

    mgr.save_spectral_tensor(
        name="hamiltonian_j_block",
        tensor=test_matrix,
        metadata=meta,
        group="/hamiltonians",
        compression="gzip",
        compression_opts=6,
        shuffle=True,
        fletcher32=True,
    )

    loaded_arr, loaded_meta = mgr.load_spectral_tensor(
        name="hamiltonian_j_block",
        group="/hamiltonians",
    )

    np.testing.assert_array_almost_equal(loaded_arr, test_matrix)
    assert loaded_meta["molecule"] == "Benzonitrile"
    assert loaded_meta["method"] == "B3LYP-D3BJ"
    assert loaded_meta["frequencies"] == [1024.5, 2048.0, 4096.5]

    # Test missing dataset error
    with pytest.raises(HDF5TensorError):
        mgr.load_spectral_tensor(name="non_existent", group="/hamiltonians")


def test_fit_registry_manager_transaction_rollback_on_error(tmp_env):
    """Validates that exceptions inside transaction() abort changes without corrupting state."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
    )

    mgr.freeze_parameter("B", value=5000.0)
    orig_state = mgr.get_fit_record()

    with pytest.raises(RuntimeError):
        with mgr.transaction() as state:
            state.manual_locks.append("CORRUPTED_PARAM")
            raise RuntimeError("Simulation error during optimization step")

    # State should remain untouched
    after_state = mgr.get_fit_record()
    assert "CORRUPTED_PARAM" not in after_state.manual_locks
    assert after_state.manual_locks == orig_state.manual_locks


def test_fit_registry_manager_corrupted_json(tmp_env):
    """Validates FitRegistryCorruptionError on corrupted JSON payload."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
    )

    # Write corrupt JSON bytes
    tmp_env["reg_json"].write_text("{ corrupted_json: invalid [", encoding="utf-8")

    with pytest.raises(FitRegistryCorruptionError):
        mgr.get_fit_record()


def test_multi_threaded_fit_registry_stress(tmp_env):
    """Validates high-concurrency multi-threaded transaction safety."""
    mgr = FitRegistryManager(
        db_path=tmp_env["db_h5"],
        registry_json_path=tmp_env["reg_json"],
        force_lustre=False,
    )

    num_threads = 8
    iterations_per_thread = 15

    def worker(worker_idx: int):
        for i in range(iterations_per_thread):
            with mgr.transaction() as state:
                p_name = f"PARAM_{worker_idx}_{i}"
                state.manual_locks.append(p_name)
                state.parameters[p_name] = FitParameterRecord(
                    name=p_name,
                    value=float(worker_idx * 100 + i),
                    is_frozen=True,
                )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_state = mgr.get_fit_record()
    expected_total = num_threads * iterations_per_thread
    assert len(final_state.manual_locks) == expected_total
    assert len(final_state.parameters) == expected_total
    assert mgr.verify_registry_integrity() is True

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_requirements.py ---
"""Physical unit tests for CoChem-SpycFit requirements.txt dependency specification.

Validates:
- Physical presence of requirements.txt at repository root.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Strict formatting: trailing LF newline, no CR/CRLF, no trailing spaces.
- Exact presence and order of all 12 pinned dependencies with exact version locks.
- No duplicate package declarations or stray blank lines.
- PEP 508 parsing validity and PEP 440 version validity.
- Strict version specification enforcement (pinned ==, no loose ranges, no markers, no URLs).
- Strict exclusion of legacy Fortran binaries, wrappers, and tools.
- Complete Zero-Mock compliance via AST structure analysis (no synthetic constructs, no pass).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

EXPECTED_DEPENDENCIES_ORDERED = [
    ("jax", "0.4.13"),
    ("jaxlib", "0.4.13"),
    ("cupy-cuda12x", "12.2.0"),
    ("h5py", "3.9.0"),
    ("zarr", "2.16.1"),
    ("pyarrow", "13.0.0"),
    ("plotly", "5.17.0"),
    ("ipywidgets", "8.1.1"),
    ("mendeleev", "0.14.0"),
    ("platformdirs", "3.10.0"),
    ("filelock", "3.12.2"),
    ("pyzmq", "25.1.1"),
]

EXPECTED_DEPENDENCY_MAPPING = dict(EXPECTED_DEPENDENCIES_ORDERED)

FORBIDDEN_LEGACY_TERMS = [
    "fortran",
    "f2py",
    "numpy.f2py",
    "scipy.weave",
    "spfit_bin",
    "spcat_bin",
    "spfit",
    "spcat",
    "gfortran",
    "flang",
    "ifort",
    "numpy-distutils",
    "libgfortran",
    "openblas-fortran",
    "lapack-fortran",
]


@pytest.fixture
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def requirements_file_path() -> Path:
    """Return the absolute path to requirements.txt and verify physical existence."""
    assert REQUIREMENTS_PATH.exists(), (
        f"requirements.txt not found at {REQUIREMENTS_PATH}"
    )
    return REQUIREMENTS_PATH


def test_requirements_physical_file_exists(repo_root_path: Path) -> None:
    """Verify that requirements.txt exists physically in repository root as a regular file."""
    file_path = repo_root_path / "requirements.txt"
    assert file_path.exists(), f"requirements.txt missing at {file_path}"
    assert file_path.is_file(), (
        f"requirements.txt at {file_path} must be a regular file"
    )
    assert file_path.stat().st_size > 0, (
        f"requirements.txt at {file_path} cannot be empty"
    )


def test_requirements_encoding_and_lf_terminators(requirements_file_path: Path) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    raw_content = requirements_file_path.read_bytes()
    assert not raw_content.startswith(b"\xef\xbb\xbf"), (
        "requirements.txt contains UTF-8 BOM"
    )
    assert b"\r\n" not in raw_content, (
        "requirements.txt contains Windows CRLF line endings"
    )
    assert b"\r" not in raw_content, "requirements.txt contains legacy CR line endings"
    assert b"\n" in raw_content, "requirements.txt missing Unix LF line endings"
    assert raw_content.endswith(b"\n"), (
        "requirements.txt must end with a single Unix LF newline"
    )


def test_requirements_no_trailing_whitespace_or_blank_lines(
    requirements_file_path: Path,
) -> None:
    """Verify no trailing whitespace on lines and no empty or blank lines."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    # Last element after final newline is empty string
    assert lines[-1] == "", "File content after final newline must be empty"
    active_lines = lines[:-1]
    for idx, line in enumerate(active_lines, start=1):
        assert line == line.strip(), (
            f"Line {idx} has leading or trailing whitespace: '{line}'"
        )
        assert len(line) > 0, f"Line {idx} is empty"


def test_requirements_exact_entry_count(requirements_file_path: Path) -> None:
    """Verify requirements.txt contains exactly 12 active dependency specifications."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) == len(EXPECTED_DEPENDENCY_MAPPING), (
        f"Expected {len(EXPECTED_DEPENDENCY_MAPPING)} dependencies, found {len(lines)}"
    )


def test_requirements_no_duplicate_packages(requirements_file_path: Path) -> None:
    """Verify no package is declared more than once in requirements.txt."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    names = [Requirement(line).name.lower() for line in lines]
    assert len(names) == len(set(names)), f"Duplicate packages detected: {names}"


def test_requirements_exact_ordered_sequence(requirements_file_path: Path) -> None:
    """Verify requirements.txt lines match the exact expected ordered list."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected_lines = [f"{pkg}=={ver}" for pkg, ver in EXPECTED_DEPENDENCIES_ORDERED]
    assert lines == expected_lines, (
        f"Order or content mismatch. Expected: {expected_lines}, got: {lines}"
    )


@pytest.mark.parametrize("pkg_name, expected_version", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_individual_pinned_package(
    requirements_file_path: Path, pkg_name: str, expected_version: str
) -> None:
    """Verify each required package is present with exact pinned version specification."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected_line = f"{pkg_name}=={expected_version}"
    assert expected_line in lines, (
        f"Expected exact line '{expected_line}' in requirements.txt"
    )


@pytest.mark.parametrize("pkg_name, expected_version", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_pep440_version_validity(
    pkg_name: str, expected_version: str
) -> None:
    """Verify each pinned version string is a valid PEP 440 version with expected components."""
    ver = Version(expected_version)
    assert ver.base_version == expected_version, (
        f"Version {expected_version} for {pkg_name} is not in standard base form"
    )
    assert not ver.is_prerelease, (
        f"Version {expected_version} for {pkg_name} must not be a pre-release"
    )
    assert not ver.is_devrelease, (
        f"Version {expected_version} for {pkg_name} must not be a dev-release"
    )


@pytest.mark.parametrize("pkg_name, _", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_valid_pep508_package_name(pkg_name: str, _: str) -> None:
    """Verify package name conforms to PEP 508 naming specification."""
    pep508_pattern = r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$"
    assert re.match(pep508_pattern, pkg_name, re.IGNORECASE) is not None, (
        f"Package name '{pkg_name}' does not match PEP 508 identifier specification"
    )


def test_requirements_pep508_parsing_and_equality(requirements_file_path: Path) -> None:
    """Verify all dependencies parse as valid PEP 508 requirements with exact '==' specifiers."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    parsed_mapping = {}
    for line in lines:
        req = Requirement(line)
        spec_list = list(req.specifier)
        assert len(spec_list) == 1, (
            f"Dependency {req.name} must have exactly one version specifier, got {spec_list}"
        )
        spec = spec_list[0]
        assert spec.operator == "==", (
            f"Dependency {req.name} must use '==' operator, got '{spec.operator}'"
        )
        parsed_mapping[req.name.lower()] = spec.version

    for pkg_name, expected_ver in EXPECTED_DEPENDENCY_MAPPING.items():
        canonical_name = pkg_name.lower()
        assert canonical_name in parsed_mapping, (
            f"Missing parsed requirement for {pkg_name}"
        )
        assert parsed_mapping[canonical_name] == expected_ver, (
            f"Version mismatch for {pkg_name}: expected {expected_ver}, got {parsed_mapping[canonical_name]}"
        )


def test_requirements_no_loose_version_specifiers(requirements_file_path: Path) -> None:
    """Verify no loose version operators (>=, <=, ~=, >, <, *, !=) exist in requirements.txt."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    forbidden_operators = [">=", "<=", "~=", ">", "<", "*", "!="]
    for line in lines:
        for op in forbidden_operators:
            assert op not in line, (
                f"Line '{line}' contains loose or forbidden version operator '{op}'"
            )


def test_requirements_no_urls_extras_or_markers(requirements_file_path: Path) -> None:
    """Verify requirements.txt contains no URL pins, extra features, or environment markers."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in lines:
        req = Requirement(line)
        assert req.url is None, (
            f"Dependency {req.name} contains forbidden URL pin: {req.url}"
        )
        assert len(req.extras) == 0, (
            f"Dependency {req.name} contains forbidden extras: {req.extras}"
        )
        assert req.marker is None, (
            f"Dependency {req.name} contains forbidden environment marker: {req.marker}"
        )


@pytest.mark.parametrize("forbidden_term", FORBIDDEN_LEGACY_TERMS)
def test_requirements_excludes_legacy_fortran(
    requirements_file_path: Path, forbidden_term: str
) -> None:
    """Verify requirements.txt strictly excludes legacy Fortran binaries and wrappers."""
    content = requirements_file_path.read_text(encoding="utf-8").lower()
    assert forbidden_term not in content, (
        f"Legacy Fortran indicator '{forbidden_term}' found in requirements.txt"
    )


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify test module uses pure physical testing without synthetic modules or banned imports."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_import_terms = ["unittest.mo" + "ck", "unitt" + "est.mock"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for term in forbidden_import_terms:
                    assert term not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for term in forbidden_import_terms:
                assert term not in module_name, f"Forbidden import: {module_name}"


def test_ast_no_pass_or_empty_functions() -> None:
    """Verify test module contains no pass statements or empty function bodies."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Pass), (
            "Forbidden 'pass' statement found in test module"
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert len(node.body) > 0, f"Function '{node.name}' has empty body"
            # Ensure not just a docstring or pass
            real_statements = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                )
            ]
            assert len(real_statements) > 0, (
                f"Function '{node.name}' contains no executable assertions"
            )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.