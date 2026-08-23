"""
CoChem-SpycFit: State Persistence, Fit Registry Manager, and Zarr Persistence.

Authoritative state persistence, atomic locking, cryptographic dataset hashing,
FAIR compliance, SWMR HDF5 tensors, and Zarr tensor persistence for the
CoChem-SpycFit Bayesian spectral fitting and convolution ecosystem.

Enforces:
1. Lightweight JSON registry (cochem_fit_registry.json) tracking Manual-Locks,
   Hardware Instrument Limits, and Active Learning / Information-Gain ML Feedback Loops.
2. Cross-platform atomic locking using standard filelock (with in-process RLock) and
   HPC/Lustre atomic directory creation (os.mkdir) locks.
3. Cryptographic SHA-256 hash calculation and verification for dataset integrity (FAIR compliance).
4. Zarr Tensor Persistence for massive intermediate spectral arrays at
   platformdirs.user_data_path("CoChem_Artifacts") / "SpycFit_Workspace" / "Scratch" / "spectra_fit.zarr".
5. SWMR HDF5 persistence (libver='latest', swmr=True) for concurrent JAX optimizer streaming
   and Voila GUI dashboard polling, with Lustre fallback to metadata-safe atomic snapshots.
6. Strict Pydantic v2 schemas (ConfigDict(extra="forbid", validate_assignment=True)) for
   parameter locks, instrument limits, active learning feedback loops, and registry state.
7. Typed exception hierarchy for comprehensive error handling.
8. Zero-Mock policy: 100% real physical files, threads, locks, snapshots, HDF5, and Zarr operations.
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
from typing import Any, ContextManager, Dict, Generator, List, Optional, Tuple, Union

import filelock
import h5py
import numpy as np
import platformdirs
import zarr
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.FitRegistryManager")

# Constants & Default Timeouts
DEFAULT_LOCAL_LOCK_TIMEOUT: float = 10.0
DEFAULT_LUSTRE_LOCK_TIMEOUT: float = 60.0
DEFAULT_STALE_LOCK_TIMEOUT: float = 60.0
LUSTRE_SUPER_MAGIC: int = 0x0BD00BD0
DEFAULT_CHUNK_SIZE: int = 65536

DEFAULT_REGISTRY_FILENAME: str = "cochem_fit_registry.json"
DEFAULT_ZARR_STORENAME: str = "spectra_fit.zarr"
DEFAULT_H5_DBNAME: str = "spectra_fit.h5"


# =============================================================================
# DEFAULT PATH RESOLUTION HELPERS
# =============================================================================

def get_default_registry_path() -> Path:
    """Returns the default filesystem path for the lightweight JSON fit registry."""
    env_json = os.environ.get("COCHEM_FIT_REGISTRY_JSON")
    if env_json:
        return Path(env_json).resolve()
    base = Path(platformdirs.user_data_path("CoChem_Artifacts"))
    return base / DEFAULT_REGISTRY_FILENAME


def get_default_zarr_path() -> Path:
    """Returns the default filesystem path for the scratch Zarr tensor persistence store."""
    env_zarr = os.environ.get("COCHEM_SPECTRA_FIT_ZARR")
    if env_zarr:
        return Path(env_zarr).resolve()
    base = Path(platformdirs.user_data_path("CoChem_Artifacts"))
    return base / "SpycFit_Workspace" / "Scratch" / DEFAULT_ZARR_STORENAME


def get_default_db_path() -> Path:
    """Returns the default filesystem path for the HDF5 state tensor database."""
    env_db = os.environ.get("COCHEM_SPECTRA_FIT_H5")
    if env_db:
        return Path(env_db).resolve()
    base = Path(platformdirs.user_data_path("CoChem_Artifacts"))
    return base / "Databases" / DEFAULT_H5_DBNAME


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
    """Raised when a required registry JSON, Zarr store, or HDF5 database file is missing."""


class FitRegistryCorruptionError(FitRegistryError, ValueError):
    """Raised when registry checksum verification fails or payload is corrupted."""


class FitRegistrySchemaError(FitRegistryError, ValueError):
    """Raised when a registry state record fails Pydantic schema validation."""


class HDF5TensorError(FitRegistryError):
    """Raised when HDF5 I/O, dataset creation, or SWMR operations fail."""


class ZarrTensorError(FitRegistryError):
    """Raised when Zarr array creation, persistence, or retrieval operations fail."""


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
        return sorted(list(set(v)))


# =============================================================================
# 3. ENVIRONMENT & FILESYSTEM DETECTION
# =============================================================================

def is_lustre_filesystem(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Dynamically detects whether a given path or runtime environment resides on an
    HPC Lustre distributed filesystem.
    """
    force_lustre = os.environ.get("COCHEM_FORCE_LUSTRE", "").strip().lower()
    if force_lustre in ("1", "true", "yes", "on"):
        return True

    force_standard = os.environ.get("COCHEM_FORCE_STANDARD_FS", "").strip().lower()
    if force_standard in ("1", "true", "yes", "on"):
        return False

    fs_type = os.environ.get("COCHEM_FS_TYPE", "").strip().lower()
    if fs_type in ("lustre", "lustrefs", "lustre_hpc"):
        return True

    target_path = Path(path).resolve() if path is not None else Path.cwd().resolve()
    for part in target_path.parts:
        part_clean = part.lower().strip("\\/ ")
        if part_clean in ("lustre", "lustrefs", "lustre_hpc") or part_clean.startswith("lustre_mount") or part_clean == ".lustre":
            return True

    if hasattr(os, "statvfs"):
        try:
            check_p = target_path if target_path.exists() else target_path.parent
            if check_p.exists():
                st = os.statvfs(str(check_p))
                f_type = getattr(st, "f_type", None)
                if f_type is not None and f_type == LUSTRE_SUPER_MAGIC:
                    return True
        except Exception as stat_err:
            logger.debug("statvfs check notice: %s", stat_err)

    if platform.system().lower() == "linux" and os.path.isfile("/proc/mounts"):
        try:
            with open("/proc/mounts", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[2].lower() == "lustre":
                        mount_point = parts[1]
                        if str(target_path).startswith(mount_point):
                            return True
        except Exception as proc_err:
            logger.debug("/proc/mounts check notice: %s", proc_err)

    return False


def resolve_lock_timeout(
    is_lustre: bool,
    custom_timeout: Optional[float] = None,
) -> float:
    """Scales atomic lock timeouts dynamically based on the detected environment."""
    if custom_timeout is not None:
        return float(custom_timeout)

    env_timeout = os.environ.get("COCHEM_LOCK_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError as parse_err:
            logger.debug("Invalid COCHEM_LOCK_TIMEOUT value '%s': %s", env_timeout, parse_err)

    return DEFAULT_LUSTRE_LOCK_TIMEOUT if is_lustre else DEFAULT_LOCAL_LOCK_TIMEOUT


# =============================================================================
# 4. CRYPTOGRAPHIC UTILITIES & DATASET HASH LOGGING (FAIR COMPLIANCE)
# =============================================================================

def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Computes the SHA-256 cryptographic hash of a file on disk."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_dataset(dataset_path: Union[str, Path], chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """
    Computes a cryptographic SHA-256 hash for a dataset file or directory store (FAIR compliance).
    Supports individual dataset files as well as multi-file directory stores (.zarr).
    """
    p = Path(dataset_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found for hash calculation: {p}")

    if p.is_file():
        return compute_file_sha256(p, chunk_size=chunk_size)

    # Deterministic directory hashing (e.g. for Zarr directory stores)
    hasher = hashlib.sha256()
    for child in sorted(p.rglob("*")):
        if child.is_file():
            rel_path = child.relative_to(p).as_posix().encode("utf-8")
            hasher.update(rel_path)
            with open(child, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)

    return hasher.hexdigest()


def verify_dataset_hash(
    dataset_path: Union[str, Path],
    expected_hash: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Verifies that a dataset file or Zarr store matches its expected cryptographic SHA-256 hash."""
    try:
        actual_hash = hash_dataset(dataset_path, chunk_size=chunk_size)
        return actual_hash.lower() == expected_hash.strip().lower()
    except (FileNotFoundError, OSError) as e:
        logger.debug("Dataset hash verification notice on %s: %s", dataset_path, e)
        return False


# =============================================================================
# 5. CROSS-PLATFORM & FILESYSTEM-AWARE ATOMIC LOCKS
# =============================================================================

class LustreMkdirLock:
    """HPC/Lustre Native Atomic Directory Creation (mkdir) Lock."""

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
        if not hasattr(self._tls, "held"):
            self._tls.held = {}

        path_key = str(self.lock_dir)

        if self._tls.held.get(path_key, 0) > 0:
            self._tls.held[path_key] += 1
            self._depth += 1
            return True

        eff_timeout = float(timeout) if timeout is not None else self.timeout
        start_time = time.time()
        thread_lock = self._get_thread_lock(path_key)

        remaining = max(0.001, eff_timeout - (time.time() - start_time))
        if not thread_lock.acquire(timeout=remaining):
            raise RegistryLockTimeoutError(
                f"[LustreMkdirLock] In-process thread lock timeout on '{self.lock_dir}' after {eff_timeout:.2f}s"
            )

        self._thread_lock_acquired = True
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed >= eff_timeout:
                self._cleanup_thread_lock(path_key)
                raise RegistryLockTimeoutError(
                    f"[LustreMkdirLock] Failed to acquire atomic directory lock on '{self.lock_dir}' within {eff_timeout:.2f}s"
                )

            try:
                os.mkdir(str(self.lock_dir))
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
                except Exception as meta_err:
                    logger.debug("[LustreMkdirLock] Owner metadata write notice: %s", meta_err)

                self._depth = 1
                self._tls.held[path_key] = 1
                return True

            except (FileExistsError, PermissionError, OSError):
                self._check_and_reap_stale_lock()

            attempt += 1
            sleep_time = min(0.2, self.poll_interval * (1.2 ** min(attempt, 8)) + random.uniform(0.001, 0.01))
            time.sleep(sleep_time)

    def _check_and_reap_stale_lock(self) -> None:
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
                except Exception as read_err:
                    logger.debug("[LustreMkdirLock] Error reading owner file: %s", read_err)
                    try:
                        lock_age = now - self.owner_file.stat().st_mtime
                    except OSError as stat_err:
                        logger.debug("[LustreMkdirLock] Stat error on owner file: %s", stat_err)
            else:
                try:
                    lock_age = now - self.lock_dir.stat().st_mtime
                except OSError as stat_err:
                    logger.debug("[LustreMkdirLock] Stat error on lock directory: %s", stat_err)

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
        except Exception as reap_err:
            logger.debug("[LustreMkdirLock] Stale lock check notice: %s", reap_err)

    def _cleanup_thread_lock(self, path_key: str) -> None:
        if self._thread_lock_acquired:
            self._thread_lock_acquired = False
            try:
                self._get_thread_lock(path_key).release()
            except RuntimeError as r_err:
                logger.debug("[LustreMkdirLock] Thread lock release notice: %s", r_err)

    def release(self) -> None:
        path_key = str(self.lock_dir)

        if not hasattr(self._tls, "held") or self._tls.held.get(path_key, 0) <= 0:
            if self._depth > 0:
                self._depth -= 1
            self._cleanup_thread_lock(path_key)
            return

        self._depth -= 1
        self._tls.held[path_key] -= 1

        if self._tls.held[path_key] > 0:
            return

        for _ in range(5):
            try:
                if self.owner_file.exists():
                    self.owner_file.unlink(missing_ok=True)
                if self.lock_dir.exists():
                    os.rmdir(str(self.lock_dir))
                break
            except Exception as rm_err:
                logger.debug("[LustreMkdirLock] Rmdir transient retry notice: %s", rm_err)
                time.sleep(0.005)
        self._cleanup_thread_lock(path_key)

    def __enter__(self) -> LustreMkdirLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class StandardFileLock:
    """Process-Safe & Thread-Safe Cross-Platform Atomic File Lock (filelock + In-Process RLock)."""

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
        if not hasattr(self._tls, "held"):
            self._tls.held = {}
        if not hasattr(self._tls, "locks"):
            self._tls.locks = {}

        path_key = str(self.lock_path)

        if self._tls.held.get(path_key, 0) > 0:
            self._tls.held[path_key] += 1
            self._depth += 1
            return True

        eff_timeout = float(timeout) if timeout is not None else self.timeout
        start_time = time.time()
        thread_lock = self._get_thread_lock(path_key)

        remaining = max(0.001, eff_timeout - (time.time() - start_time))
        if not thread_lock.acquire(timeout=remaining):
            raise RegistryLockTimeoutError(
                f"[StandardFileLock] Thread lock timeout on '{self.lock_path}' after {eff_timeout:.2f}s"
            )

        self._thread_lock_acquired = True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        if self.lock_path.is_file():
            try:
                mtime = self.lock_path.stat().st_mtime
                if (time.time() - mtime) > self.stale_timeout:
                    self.lock_path.unlink(missing_ok=True)
                    logger.warning("[StandardFileLock] Reaped stale lock: %s", self.lock_path)
            except OSError as stat_err:
                logger.debug("[StandardFileLock] Stale stat check notice: %s", stat_err)

        rem_filelock = max(0.001, eff_timeout - (time.time() - start_time))
        fl = filelock.SoftFileLock(str(self.lock_path), timeout=rem_filelock)
        try:
            fl.acquire(timeout=rem_filelock)
            try:
                self.lock_path.write_text(
                    f"{os.getpid()}:{threading.get_ident()}:{time.time()}\n",
                    encoding="utf-8",
                )
            except Exception as write_diag_err:
                logger.debug("[StandardFileLock] Diagnostic write notice: %s", write_diag_err)

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
            except RuntimeError as r_err:
                logger.debug("[StandardFileLock] Thread lock cleanup notice: %s", r_err)

    def release(self) -> None:
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

        fl = self._tls.locks.pop(path_key, self._filelock)
        self._filelock = None
        if fl is not None:
            try:
                fl.release()
            except Exception as rel_err:
                logger.debug("[StandardFileLock] FileLock release notice: %s", rel_err)

        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError as unlink_err:
            logger.debug("[StandardFileLock] Lock file unlink notice: %s", unlink_err)
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
    """Factory creating a cross-platform, filesystem-aware atomic lock."""
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
    """Creates a metadata-safe, atomic snapshot of an HDF5 database or registry file on disk."""
    src = Path(source_path).resolve()
    if not src.is_file():
        raise FitRegistryMissingError(f"Cannot create snapshot: source file does not exist at {src}")

    snapshot_id = f"snap_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if dest_path is not None:
        dst = Path(dest_path).resolve()
    else:
        target_dir = Path(backup_dir).resolve() if backup_dir is not None else src.parent / "snapshots"
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / f"{src.stem}_{snapshot_id}{src.suffix}"

    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_dst = dst.parent / f"{dst.name}.tmp_{uuid.uuid4().hex[:8]}"
    try:
        shutil.copy2(str(src), str(tmp_dst))
        os.replace(str(tmp_dst), str(dst))
    except Exception as e:
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)
        raise FitRegistryCorruptionError(f"Atomic snapshot creation failed from {src} to {dst}: {e}") from e

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
    """Context manager opening the HDF5 State Tensor database with filesystem-aware SWMR management."""
    p = Path(db_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    lustre_detected = is_lustre if is_lustre is not None else is_lustre_filesystem(p)

    h5_file: Optional[h5py.File] = None
    lock: Optional[Union[LustreMkdirLock, StandardFileLock]] = None

    try:
        is_write_mode = mode in ("w", "a", "r+", "x")
        if is_write_mode:
            lock = get_atomic_lock(p, is_lustre=lustre_detected, timeout=timeout)
            lock.acquire()

        if lustre_detected:
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
                except Exception as flush_err:
                    logger.debug("[open_state_tensor] HDF5 flush notice on Lustre: %s", flush_err)

        else:
            logger.debug("[open_state_tensor] Standard filesystem. SWMR active for %s", p)
            if mode == "r":
                if not p.is_file():
                    raise FitRegistryMissingError(f"HDF5 database not found at {p}")
                try:
                    h5_file = h5py.File(str(p), mode="r", libver="latest", swmr=True)
                except Exception as swmr_read_err:
                    logger.debug("[open_state_tensor] SWMR read fallback notice: %s", swmr_read_err)
                    try:
                        h5_file = h5py.File(str(p), mode="r", libver="latest")
                    except Exception as e:
                        raise HDF5TensorError(f"Failed to open HDF5 SWMR file for reading at {p}: {e}") from e

                yield h5_file

            else:
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
                    except Exception as post_write_err:
                        logger.debug("[open_state_tensor] Post-write flush notice: %s", post_write_err)

    finally:
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception as close_err:
                logger.debug("[open_state_tensor] HDF5 file close notice: %s", close_err)

        if lock is not None:
            try:
                lock.release()
            except Exception as lock_rel_err:
                logger.debug("[open_state_tensor] Atomic lock release notice: %s", lock_rel_err)

        if lustre_detected and is_write_mode and p.is_file():
            try:
                create_atomic_snapshot(source_path=p, is_lustre=True)
            except Exception as snap_err:
                logger.warning("[open_state_tensor] Lustre post-write snapshot notice: %s", snap_err)


# =============================================================================
# 8. STANDALONE REGISTRY FUNCTIONS
# =============================================================================

def read_registry(registry_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Reads the lightweight JSON registry from disk wrapped in an atomic cross-platform lock."""
    target = Path(registry_path).resolve() if registry_path is not None else get_default_registry_path()
    if not target.is_file():
        return FitRegistryState().model_dump()

    lock = get_atomic_lock(target)
    with lock:
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise FitRegistryCorruptionError(f"Corrupted JSON in fit registry at {target}: {e}") from e
        except Exception as e:
            raise FitRegistryCorruptionError(f"Error reading fit registry at {target}: {e}") from e

        try:
            validated = FitRegistryState.model_validate(data)
            return validated.model_dump()
        except Exception as e:
            raise FitRegistrySchemaError(f"Schema validation error in fit registry at {target}: {e}") from e


def write_registry(
    data: Union[Dict[str, Any], FitRegistryState],
    registry_path: Optional[Union[str, Path]] = None,
) -> None:
    """Atomically persists data into the lightweight JSON registry under an atomic lock."""
    target = Path(registry_path).resolve() if registry_path is not None else get_default_registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, FitRegistryState):
        record = data
    elif isinstance(data, dict):
        try:
            record = FitRegistryState.model_validate(data)
        except Exception as e:
            raise FitRegistrySchemaError(f"Cannot write invalid registry data to {target}: {e}") from e
    else:
        raise FitRegistrySchemaError(f"Expected dict or FitRegistryState, got {type(data)}")

    json_data = record.model_dump_json(indent=2)
    lock = get_atomic_lock(target)
    with lock:
        tmp_p = target.parent / f"{target.name}.tmp_{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_p, "w", encoding="utf-8") as f:
                f.write(json_data)
            os.replace(str(tmp_p), str(target))
        except Exception as e:
            if tmp_p.exists():
                tmp_p.unlink(missing_ok=True)
            raise FitRegistryCorruptionError(f"Failed to atomically write registry JSON to {target}: {e}") from e


# =============================================================================
# 9. ZARR TENSOR PERSISTENCE (MASSIVE INTERMEDIATE SPECTRAL ARRAYS)
# =============================================================================

def _zarr_create_array(
    grp: Any,
    name: str,
    shape: Tuple[int, ...],
    dtype: Any,
    chunks: Union[bool, Tuple[int, ...], None] = True,
    overwrite: bool = True,
    data: Optional[np.ndarray] = None,
) -> Any:
    """Creates a dataset/array in a Zarr group compatible with both Zarr 2.x and 3.x."""
    if name in grp and overwrite:
        del grp[name]

    if hasattr(grp, "create_array"):
        # Zarr 3.x
        if chunks is True:
            chunks_spec = tuple(min(dim, 1000) for dim in shape) if len(shape) > 0 else (1,)
        elif chunks is None or chunks is False:
            chunks_spec = None
        else:
            chunks_spec = chunks
        dset = grp.create_array(name=name, shape=shape, dtype=dtype, chunks=chunks_spec)
        if data is not None:
            dset[...] = data
        return dset
    elif hasattr(grp, "create_dataset"):
        # Zarr 2.x
        return grp.create_dataset(
            name=name,
            data=data,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            overwrite=overwrite,
        )
    else:
        dset = grp.require_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
        if data is not None:
            dset[...] = data
        return dset


def _zarr_require_array(
    grp: Any,
    name: str,
    shape: Tuple[int, ...],
    dtype: Any,
    chunks: Union[bool, Tuple[int, ...], None] = True,
) -> Any:
    """Requires a dataset/array in a Zarr group compatible with both Zarr 2.x and 3.x."""
    if hasattr(grp, "require_array"):
        # Zarr 3.x
        if chunks is True:
            chunks_spec = tuple(min(dim, 1000) for dim in shape) if len(shape) > 0 else (1,)
        elif chunks is None or chunks is False:
            chunks_spec = None
        else:
            chunks_spec = chunks
        return grp.require_array(name=name, shape=shape, dtype=dtype, chunks=chunks_spec)
    elif hasattr(grp, "require_dataset"):
        # Zarr 2.x
        return grp.require_dataset(name=name, shape=shape, dtype=dtype, chunks=chunks)
    else:
        return _zarr_create_array(grp, name=name, shape=shape, dtype=dtype, chunks=chunks)


def init_zarr_persistence(
    zarr_path: Optional[Union[str, Path]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Any = float,
    chunks: Optional[Union[bool, Tuple[int, ...]]] = True,
    mode: str = "a",
    dataset_name: Optional[str] = None,
) -> Union[zarr.Group, zarr.Array]:
    """Initializes Zarr persistence store or array for massive intermediate spectral arrays."""
    target = Path(zarr_path).resolve() if zarr_path is not None else get_default_zarr_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        root = zarr.open_group(str(target), mode=mode)

        if dataset_name is not None and shape is not None:
            return _zarr_require_array(
                grp=root,
                name=dataset_name,
                shape=shape,
                dtype=dtype,
                chunks=chunks if chunks is not None else True,
            )
        elif dataset_name is not None:
            if dataset_name in root:
                return root[dataset_name]
            raise ZarrTensorError(
                f"Dataset '{dataset_name}' not found in Zarr store at {target} and no shape was provided."
            )

        return root
    except Exception as e:
        if isinstance(e, ZarrTensorError):
            raise
        raise ZarrTensorError(f"Failed to initialize Zarr persistence at {target}: {e}") from e


def save_zarr_tensor(
    name: str,
    tensor: Any,
    zarr_path: Optional[Union[str, Path]] = None,
    group: str = "/",
    metadata: Optional[Dict[str, Any]] = None,
    chunks: Union[bool, Tuple[int, ...]] = True,
    overwrite: bool = True,
) -> zarr.Array:
    """Persists a multi-dimensional array/tensor into a Zarr directory store."""
    target = Path(zarr_path).resolve() if zarr_path is not None else get_default_zarr_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
    else:
        arr = np.ascontiguousarray(np.array(tensor))
    if arr.ndim == 0:
        arr = arr.reshape((1,))

    lock = get_atomic_lock(target)
    with lock:
        try:
            root = zarr.open_group(str(target), mode="a")
            clean_grp = group.strip("/")
            grp = root.require_group(clean_grp) if clean_grp else root

            dset = _zarr_create_array(
                grp=grp,
                name=name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=chunks,
                overwrite=overwrite,
                data=arr,
            )

            if metadata:
                for k, v in metadata.items():
                    dset.attrs[k] = v

            return dset
        except Exception as e:
            raise ZarrTensorError(f"Failed to save Zarr tensor '{name}' to {target}: {e}") from e


def load_zarr_tensor(
    name: str,
    zarr_path: Optional[Union[str, Path]] = None,
    group: str = "/",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a tensor and its metadata attributes from a Zarr store."""
    target = Path(zarr_path).resolve() if zarr_path is not None else get_default_zarr_path()
    if not target.exists():
        raise FitRegistryMissingError(f"Zarr store does not exist at {target}")

    try:
        root = zarr.open_group(str(target), mode="r")
        clean_grp = group.strip("/")
        grp = root[clean_grp] if clean_grp else root

        if name not in grp:
            raise ZarrTensorError(f"Dataset '{name}' not found in Zarr group '{group}' at {target}")

        dset = grp[name]
        arr = np.array(dset[:])
        meta = dict(dset.attrs)
        return arr, meta
    except Exception as e:
        if isinstance(e, (FitRegistryMissingError, ZarrTensorError)):
            raise
        raise ZarrTensorError(f"Failed to load Zarr tensor '{name}' from {target}: {e}") from e


# =============================================================================
# 10. HIGH-LEVEL FIT REGISTRY MANAGER
# =============================================================================

class FitRegistryManager:
    """
    High-Level Fit Registry and State Persistence Manager for CoChem-SpycFit.

    Provides atomic JSON state management, SWMR / Lustre snapshot HDF5 tensor persistence,
    Zarr intermediate array persistence, parameter locking, active learning feedback state,
    and cryptographic dataset verification.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        registry_json_path: Optional[Union[str, Path]] = None,
        zarr_path: Optional[Union[str, Path]] = None,
        lock_path: Optional[Union[str, Path]] = None,
        auto_detect_lustre: bool = True,
        force_lustre: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if registry_json_path is not None:
            self.registry_json_path = Path(registry_json_path).resolve()
        else:
            self.registry_json_path = get_default_registry_path()

        self.registry_json_path.parent.mkdir(parents=True, exist_ok=True)

        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = get_default_db_path()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if zarr_path is not None:
            self.zarr_path = Path(zarr_path).resolve()
        else:
            self.zarr_path = get_default_zarr_path()

        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)

        if force_lustre is not None:
            self.is_lustre = bool(force_lustre)
        elif auto_detect_lustre:
            self.is_lustre = (
                is_lustre_filesystem(self.registry_json_path)
                or is_lustre_filesystem(self.db_path)
                or is_lustre_filesystem(self.zarr_path)
            )
        else:
            self.is_lustre = False

        self.timeout = resolve_lock_timeout(self.is_lustre, custom_timeout=timeout)

        if lock_path is not None:
            self.lock_path = Path(lock_path).resolve()
        else:
            suffix = ".lockdir" if self.is_lustre else ".lock"
            self.lock_path = self.registry_json_path.parent / f"{self.registry_json_path.name}{suffix}"

        self.snapshots_dir = self.db_path.parent / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FitRegistryManager initialized. is_lustre=%s, timeout=%.1fs, registry=%s, db=%s, zarr=%s",
            self.is_lustre,
            self.timeout,
            self.registry_json_path.name,
            self.db_path.name,
            self.zarr_path.name,
        )

    def get_atomic_lock(
        self,
        lock_path: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
    ) -> Union[LustreMkdirLock, StandardFileLock]:
        target_lock = Path(lock_path).resolve() if lock_path is not None else self.lock_path
        eff_timeout = timeout if timeout is not None else self.timeout
        return get_atomic_lock(
            lock_path=target_lock,
            is_lustre=self.is_lustre,
            timeout=eff_timeout,
        )

    def is_lustre_filesystem(self, path: Optional[Union[str, Path]] = None) -> bool:
        target = Path(path).resolve() if path is not None else self.registry_json_path
        return is_lustre_filesystem(target)

    def get_fit_record(self) -> FitRegistryState:
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
                raise FitRegistryCorruptionError(
                    f"Failed to atomically write registry JSON to {self.registry_json_path}: {e}"
                ) from e

        if lock is not None:
            _do_write()
        else:
            with self.get_atomic_lock():
                _do_write()

    @contextmanager
    def transaction(self) -> Generator[FitRegistryState, None, None]:
        lock = self.get_atomic_lock()
        with lock:
            state = self.get_fit_record()
            yield state
            state.timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.save_fit_record(state, lock=lock)

    def read_registry(self) -> Dict[str, Any]:
        return read_registry(self.registry_json_path)

    def write_registry(self, data: Union[Dict[str, Any], FitRegistryState]) -> None:
        write_registry(data, self.registry_json_path)

    def freeze_parameter(self, param_name: str, value: Optional[float] = None) -> FitRegistryState:
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
    ) -> ContextManager[h5py.File]:
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
        if isinstance(tensor, np.ndarray):
            arr = np.ascontiguousarray(tensor)
        else:
            arr = np.ascontiguousarray(np.array(tensor))
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
                    except Exception as load_err:
                        logger.debug("Attribute json parse notice: %s", load_err)
                        meta[k] = v
                else:
                    meta[k] = v

            return arr, meta

    def init_zarr_persistence(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Any = float,
        chunks: Optional[Union[bool, Tuple[int, ...]]] = True,
        mode: str = "a",
        dataset_name: Optional[str] = None,
        zarr_path: Optional[Union[str, Path]] = None,
    ) -> Union[zarr.Group, zarr.Array]:
        target = zarr_path if zarr_path is not None else self.zarr_path
        return init_zarr_persistence(
            zarr_path=target,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            mode=mode,
            dataset_name=dataset_name,
        )

    def save_zarr_tensor(
        self,
        name: str,
        tensor: Any,
        group: str = "/",
        metadata: Optional[Dict[str, Any]] = None,
        chunks: Union[bool, Tuple[int, ...]] = True,
        overwrite: bool = True,
        zarr_path: Optional[Union[str, Path]] = None,
    ) -> zarr.Array:
        target = zarr_path if zarr_path is not None else self.zarr_path
        return save_zarr_tensor(
            name=name,
            tensor=tensor,
            zarr_path=target,
            group=group,
            metadata=metadata,
            chunks=chunks,
            overwrite=overwrite,
        )

    def load_zarr_tensor(
        self,
        name: str,
        group: str = "/",
        zarr_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        target = zarr_path if zarr_path is not None else self.zarr_path
        return load_zarr_tensor(
            name=name,
            zarr_path=target,
            group=group,
        )

    def hash_dataset(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> str:
        target = dataset_path if dataset_path is not None else self.db_path
        return hash_dataset(target, chunk_size=chunk_size)

    def verify_dataset_hash(
        self,
        expected_hash: str,
        dataset_path: Optional[Union[str, Path]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> bool:
        target = dataset_path if dataset_path is not None else self.db_path
        return verify_dataset_hash(target, expected_hash, chunk_size=chunk_size)

    def set_active_dataset_hash(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        custom_hash: Optional[str] = None,
    ) -> FitRegistryState:
        target_hash = custom_hash if custom_hash is not None else self.hash_dataset(dataset_path)
        with self.transaction() as state:
            state.active_dataset_hash = target_hash
        return self.get_fit_record()

    def verify_registry_integrity(self) -> bool:
        """
        Verifies the physical presence and schema integrity of the active registry JSON.

        Returns:
            bool: True if registry state is valid and non-corrupted.
        """
        try:
            state = self.get_fit_record()
            return isinstance(state, FitRegistryState) and len(state.registry_version) > 0
        except Exception as ver_err:
            logger.debug("Registry integrity verification notice: %s", ver_err)
            return False

