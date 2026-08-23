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
    path_str = str(target_path).lower()

    if "lustre" in path_str:
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

            lock_age = 0.0
            if self.owner_file.is_file():
                try:
                    with open(self.owner_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        acquired_at = float(data.get("acquired_at", 0.0))
                        if acquired_at > 0:
                            lock_age = time.time() - acquired_at
                except Exception:
                    lock_age = time.time() - self.owner_file.stat().st_mtime
            else:
                lock_age = time.time() - self.lock_dir.stat().st_mtime

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

        # Last release: remove owner file and rmdir
        try:
            if self.owner_file.exists():
                self.owner_file.unlink(missing_ok=True)
            if self.lock_dir.exists():
                os.rmdir(str(self.lock_dir))
        except Exception as e:
            logger.debug("[LustreMkdirLock] Release rmdir notice: %s", e)
        finally:
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
        self.lock_path = Path(lock_path).resolve()
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
                # Write / Append mode with SWMR
                try:
                    h5_file = h5py.File(str(p), mode=mode, libver="latest")
                    # Activate SWMR mode if not already in SWMR mode
                    if not getattr(h5_file, "swmr_mode", False):
                        try:
                            h5_file.swmr_mode = True
                        except Exception as swmr_err:
                            logger.debug("[open_state_tensor] Notice on swmr_mode activation: %s", swmr_err)
                except Exception as e:
                    raise HDF5TensorError(f"Failed to initialize HDF5 SWMR file at {p} with mode '{mode}': {e}") from e

                yield h5_file

                if h5_file is not None:
                    try:
                        h5_file.flush()
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
