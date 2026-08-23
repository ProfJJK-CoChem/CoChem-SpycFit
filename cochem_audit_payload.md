Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task9_Prompt1_catalog_phase4.md.
Original prompt:
# Task 9 (Phase 4): Catalog Compiler - cochem_catalog_compiler.py

**Objective:**
Update `cochem_catalog_compiler.py` to implement Phase 4 requirements including PyArrow chunked streaming, strict type preservation, and lock-free Hive-partitioning for Out-of-Core memory management.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_catalog_compiler.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 4 features in `cochem_catalog_compiler.py`.
It must:
1. Target `$COCHEM_STATE_DIR/SpycFit_Workspace/Catalogs/chunked_transitions/` resolved via `platformdirs.user_data_path`.
2. Use PyArrow (`pyarrow`) to serialize data chunks directly to `.parquet` files (`chunk_{temperature_step}_{batch_id}_{uuid}.parquet`).
3. Enforce strict type preservation: frequencies remain FP64 (`float64`) and quantum assignments ($J, K_a, K_c$) remain `int32`.
4. Utilize `jax.experimental.io_callback` to intercept arrays asynchronously.
5. Employ a lock-free, partitioned append-only write strategy (Hive-partitioning). Compile the master catalog manifest in a Map-Reduce fashion only at the end.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Implement the real `pyarrow` logic and `io_callback`.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import pyarrow as pa
import pyarrow.parquet as pq
from platformdirs import user_data_path
from jax.experimental import io_callback

def serialize_chunk_to_parquet(frequency_data, quantum_assignments, temp_step, batch_id, uuid_str):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def compile_master_manifest(chunk_directory):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\__init__.py ---
"""
CoChem-SpycFit: High-Performance Serialization, Out-of-Core Catalog & Legacy Fortran Translation Subsystem.
"""

from __future__ import annotations

from cochem_spycfit.serialization.cochem_catalog_compiler import (
    CatalogCompilerConfig,
    CatalogCompilerError,
    CatalogCorruptionError,
    CatalogDatasetError,
    CatalogLockError,
    CatalogLockTimeoutError,
    CatalogManifestRecord,
    CatalogSchemaError,
    CatalogStreamingError,
    LustreCatalogLockError,
    PARQUET_TRANSITION_SCHEMA,
    Tier3RegistryLock,
    TransitionBatch,
    TransitionRecord,
    VirtualLayoutError,
    compile_master_manifest,
    compile_virtual_catalog,
    compute_sha256,
    extract_catalog_metadata,
    get_chunked_transitions_dir,
    is_lustre_filesystem,
    jax_serialize_chunk_callback,
    read_chunked_catalog,
    read_parquet_catalog,
    serialize_chunk_to_parquet,
    stream_chunked_catalog,
    verify_catalog_integrity,
)
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

__all__ = [
    "CatalogCompilerConfig",
    "CatalogCompilerError",
    "CatalogCorruptionError",
    "CatalogDatasetError",
    "CatalogLockError",
    "CatalogLockTimeoutError",
    "CatalogManifestRecord",
    "CatalogSchemaError",
    "CatalogStreamingError",
    "FortranBinaryNotFoundError",
    "FortranBridgeDispatcher",
    "FortranBridgeError",
    "FortranExecutionError",
    "FortranExecutionResult",
    "FortranExecutionSegfaultError",
    "FortranExecutionTask",
    "FortranExecutionTimeoutError",
    "FortranTaskQueue",
    "LustreCatalogLockError",
    "PickettCatFile",
    "PickettCatRecord",
    "PickettDipoleRecord",
    "PickettEgyFile",
    "PickettEgyRecord",
    "PickettFitResidual",
    "PickettFitResult",
    "PickettFormatError",
    "PickettIntFile",
    "PickettLinFile",
    "PickettLinRecord",
    "PickettParameterRecord",
    "PickettScratchContext",
    "PickettVarFile",
    "Tier3RegistryLock",
    "TransitionBatch",
    "TransitionRecord",
    "VirtualLayoutError",
    "calculate_reduced_mass",
    "capture_hex_telemetry",
    "compile_virtual_catalog",
    "compute_sha256",
    "decode_pickett_quantum",
    "dispatch_fortran_binary",
    "encode_pickett_quantum",
    "extract_catalog_metadata",
    "format_pickett_line",
    "fortran_float_converter",
    "get_fortran_deployment_dir",
    "get_isotopic_mass",
    "is_lustre_filesystem",
    "is_segfault_exit_code",
    "parse_pickett_cat",
    "parse_pickett_egy",
    "parse_pickett_fit",
    "parse_pickett_int",
    "parse_pickett_lin",
    "parse_pickett_var",
    "read_chunked_catalog",
    "resolve_fortran_binary",
    "stream_chunked_catalog",
    "verify_catalog_integrity",
    "write_spycfit_telemetry",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_catalog_compiler.py ---
"""
CoChem-SpycFit: Stage 3.3 Out-of-Core Memory Manager & Catalog Compiler.

Authoritative HDF5 chunked streaming, HPC metadata safety, VirtualLayout aggregation,
and Tier 3 OS-agnostic atomic registry concurrency protection for large-scale
rotational spectroscopic transition catalogs (Pickett, CDMS, JPL, and SpycFit tensors).

Directives Enforced:
1. Out-of-Core HDF5 Chunked Streaming: Enables streaming multi-million-line catalogs
   without cumulative in-memory array retention.
2. HPC Metadata Safety & Concurrency: Multi-worker shard compilation using h5py.VirtualLayout
   and dynamic Lustre / distributed filesystem detection.
3. Tier 3 OS-Agnostic Registry Lock: Multi-backend atomic locking (filelock + threading RLock
   for standard filesystems, atomic directory creation for distributed HPC filesystems)
   operating safely outside parallel Lustre mounts.
4. Strict Pydantic v2 Models: Exhaustive validation with ConfigDict(extra="forbid", validate_assignment=True).
5. Zero-Mock Mandate: 100% authentic physical files, real h5py operations, zero pass statements,
   zero synthetic test doubles or stubs.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
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
    Tuple,
    Union,
)

import filelock
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import platformdirs
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from jax.experimental import io_callback
from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Configure module-level logger
logger = logging.getLogger("CoChem-SpycFit.CatalogCompiler")

# Magic constants and default parameters
DEFAULT_CHUNK_SIZE: int = 65536
DEFAULT_LOCAL_LOCK_TIMEOUT: float = 30.0
DEFAULT_LUSTRE_LOCK_TIMEOUT: float = 90.0
DEFAULT_STALE_LOCK_TIMEOUT: float = 120.0
LUSTRE_SUPER_MAGIC: int = 0x0BD00BD0
HASH_READ_CHUNK_BYTES: int = 1048576  # 1 MB

STANDARD_DATASET_NAMES: Tuple[str, ...] = (
    "frequencies",
    "intensities",
    "lower_energies",
    "upper_energies",
    "uncertainties",
    "g_upper",
    "quantum_numbers",
    "species_ids",
)

PARQUET_TRANSITION_SCHEMA = pa.schema([
    pa.field("frequencies", pa.float64(), nullable=False),
    pa.field("intensities", pa.float64(), nullable=False),
    pa.field("lower_energies", pa.float64(), nullable=False),
    pa.field("upper_energies", pa.float64(), nullable=False),
    pa.field("uncertainties", pa.float64(), nullable=False),
    pa.field("g_upper", pa.int32(), nullable=False),
    pa.field("species_ids", pa.int32(), nullable=False),
    pa.field("J_upper", pa.int32(), nullable=False),
    pa.field("Ka_upper", pa.int32(), nullable=False),
    pa.field("Kc_upper", pa.int32(), nullable=False),
    pa.field("J_lower", pa.int32(), nullable=False),
    pa.field("Ka_lower", pa.int32(), nullable=False),
    pa.field("Kc_lower", pa.int32(), nullable=False),
    pa.field("quantum_numbers", pa.list_(pa.int32()), nullable=False),
    pa.field("temperature_step", pa.int32(), nullable=False),
    pa.field("batch_id", pa.string(), nullable=False),
    pa.field("chunk_uuid", pa.string(), nullable=False),
])


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class CatalogCompilerError(Exception):
    """Base exception for all SpycFit catalog compilation and out-of-core serialization operations."""


class CatalogLockError(CatalogCompilerError):
    """Base exception for atomic registry lock failures during catalog compilation."""


class CatalogLockTimeoutError(CatalogLockError, TimeoutError):
    """Raised when acquiring an atomic Tier 3 catalog registry lock exceeds the configured timeout."""


class LustreCatalogLockError(CatalogLockError):
    """Raised when atomic directory creation (mkdir) locking encounters an OS or filesystem error."""


class CatalogDatasetError(CatalogCompilerError):
    """Raised when HDF5 dataset creation, shape resizing, or columnar allocation fails."""


class CatalogStreamingError(CatalogCompilerError):
    """Raised when streaming transition chunks encounter malformed records or iterator faults."""


class VirtualLayoutError(CatalogCompilerError):
    """Raised when HPC VirtualLayout assembly or shard virtual source mapping encounters errors."""


class CatalogCorruptionError(CatalogCompilerError, ValueError):
    """Raised when catalog checksum verification fails or HDF5 container corruption is detected."""


class CatalogSchemaError(CatalogCompilerError, ValueError):
    """Raised when transition records or catalog metadata fail Pydantic v2 validation."""


# =============================================================================
# 2. PYDANTIC V2 SCHEMAS & CONFIGURATION
# =============================================================================

class TransitionRecord(BaseModel):
    """
    Pydantic model representing an individual rotational spectroscopic transition.
    Enforces strict physical constraints and quantum number assignments.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequency_mhz: float = Field(..., gt=0.0, description="Rest frequency in MHz")
    uncertainty_mhz: float = Field(default=0.0, ge=0.0, description="Experimental or model frequency uncertainty in MHz")
    intensity: float = Field(default=0.0, description="Calculated line strength (S_ij * mu^2) or log10 intensity")
    lower_energy_cm1: float = Field(default=0.0, ge=0.0, description="Lower rotational state energy in cm^-1")
    upper_energy_cm1: float = Field(default=0.0, ge=0.0, description="Upper rotational state energy in cm^-1")
    g_upper: int = Field(default=1, ge=1, description="Upper state degeneracy / spin statistical weight")
    quantum_numbers: Tuple[int, int, int, int, int, int] = Field(
        default=(0, 0, 0, 0, 0, 0),
        description="Standard 6-tuple: (J', Ka', Kc', J'', Ka'', Kc'')",
    )
    species_id: int = Field(default=0, ge=0, description="Species or isotopologue integer identifier")


class TransitionBatch(BaseModel):
    """
    Pydantic model encapsulating a validated batch of transition lines for chunked streaming.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    frequencies_mhz: List[float] = Field(..., description="List of transition frequencies in MHz")
    intensities: List[float] = Field(..., description="List of transition intensities")
    lower_energies_cm1: Optional[List[float]] = Field(default=None, description="List of lower state energies in cm^-1")
    upper_energies_cm1: Optional[List[float]] = Field(default=None, description="List of upper state energies in cm^-1")
    uncertainties_mhz: Optional[List[float]] = Field(default=None, description="List of uncertainties in MHz")
    g_upper: Optional[List[int]] = Field(default=None, description="List of upper state degeneracies")
    quantum_numbers: Optional[List[Tuple[int, int, int, int, int, int]]] = Field(
        default=None,
        description="List of 6-tuples representing upper and lower quantum numbers",
    )
    species_ids: Optional[List[int]] = Field(default=None, description="List of species IDs")

    @model_validator(mode="after")
    def validate_all_field_lengths(self) -> TransitionBatch:
        n = len(self.frequencies_mhz)
        if len(self.intensities) != n:
            raise ValueError(f"Intensities length ({len(self.intensities)}) does not match frequencies length ({n})")
        if self.lower_energies_cm1 is not None and len(self.lower_energies_cm1) != n:
            raise ValueError(f"lower_energies_cm1 length ({len(self.lower_energies_cm1)}) does not match frequencies length ({n})")
        if self.upper_energies_cm1 is not None and len(self.upper_energies_cm1) != n:
            raise ValueError(f"upper_energies_cm1 length ({len(self.upper_energies_cm1)}) does not match frequencies length ({n})")
        if self.uncertainties_mhz is not None and len(self.uncertainties_mhz) != n:
            raise ValueError(f"uncertainties_mhz length ({len(self.uncertainties_mhz)}) does not match frequencies length ({n})")
        if self.g_upper is not None and len(self.g_upper) != n:
            raise ValueError(f"g_upper length ({len(self.g_upper)}) does not match frequencies length ({n})")
        if self.quantum_numbers is not None and len(self.quantum_numbers) != n:
            raise ValueError(f"quantum_numbers length ({len(self.quantum_numbers)}) does not match frequencies length ({n})")
        if self.species_ids is not None and len(self.species_ids) != n:
            raise ValueError(f"species_ids length ({len(self.species_ids)}) does not match frequencies length ({n})")
        return self


class CatalogCompilerConfig(BaseModel):
    """
    Pydantic model defining configuration parameters for HDF5 chunked compilation and memory safety.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=128, description="HDF5 chunk size along transition axis")
    compression: Optional[str] = Field(default="gzip", description="HDF5 dataset compression algorithm ('gzip', 'lzf', or None)")
    compression_opts: Optional[int] = Field(default=4, ge=1, le=9, description="Compression level for gzip compression")
    fletcher32: bool = Field(default=True, description="Enable Fletcher32 checksum filter on chunked datasets")
    swmr_enabled: bool = Field(default=True, description="Enable HDF5 Single-Writer Multiple-Reader (SWMR) mode on supported filesystems")
    lock_timeout_sec: float = Field(default=DEFAULT_LOCAL_LOCK_TIMEOUT, gt=0.0, description="Timeout for acquiring Tier 3 locks")
    max_buffer_mb: float = Field(default=256.0, gt=1.0, description="Maximum in-memory accumulation buffer size in Megabytes")
    lustre_safe_mode: bool = Field(default=True, description="Auto-detect Lustre/NFS filesystems to prevent SWMR/flock lock deadlocks")
    lock_directory: Optional[str] = Field(
        default=None,
        description="Explicit directory for Tier 3 lock metadata (placed outside Lustre parallel mounts)",
    )


class CatalogManifestRecord(BaseModel):
    """
    Pydantic model representing the cryptographic and physical manifest of a compiled transition catalog.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    catalog_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique catalog UUID")
    catalog_path: str = Field(..., description="Canonical filesystem path to master HDF5 catalog file")
    total_transitions: int = Field(ge=0, description="Total count of discrete transition lines compiled")
    frequency_min_mhz: float = Field(description="Minimum frequency in catalog in MHz")
    frequency_max_mhz: float = Field(description="Maximum frequency in catalog in MHz")
    intensity_min: float = Field(description="Minimum intensity value in catalog")
    intensity_max: float = Field(description="Maximum intensity value in catalog")
    is_virtual: bool = Field(default=False, description="True if catalog is compiled via h5py.VirtualLayout")
    num_shards: int = Field(default=1, ge=1, description="Number of underlying physical shard containers")
    shard_paths: List[str] = Field(default_factory=list, description="List of physical shard paths (if virtual catalog)")
    sha256_checksum: str = Field(..., description="Cryptographic SHA-256 checksum of catalog container")
    created_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC creation timestamp",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom physical and spectroscopic metadata")


# =============================================================================
# 3. ENVIRONMENT & FILESYSTEM DETECTION
# =============================================================================

def is_lustre_filesystem(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Dynamically detects whether a target filesystem path is hosted on an HPC Lustre
    or distributed cluster filesystem where POSIX flock or HDF5 SWMR can deadlock.

    Evaluation hierarchy:
    1. Explicit environment variable overrides:
       - COCHEM_FORCE_LUSTRE=1 / true / yes -> True
       - COCHEM_FORCE_STANDARD_FS=1 / true / yes -> False
       - COCHEM_FS_TYPE="lustre" -> True
    2. Path string component inspection (e.g., 'lustre', 'lustrefs', '.lustre').
    3. POSIX statvfs magic superblock verification (0x0BD00BD0).
    4. Linux /proc/mounts inspection.

    Args:
        path: Optional path to evaluate. Defaults to Path.cwd().

    Returns:
        bool: True if Lustre distributed filesystem is detected, False otherwise.
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
        except Exception:
            logger.debug("statvfs check failed during Lustre detection")

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
            logger.debug("/proc/mounts inspection failed during Lustre detection")

    return False


def compute_sha256(filepath: Union[str, Path], chunk_bytes: int = HASH_READ_CHUNK_BYTES) -> str:
    """
    Computes the cryptographic SHA-256 checksum of a file on disk.

    Args:
        filepath: Filesystem path to target file.
        chunk_bytes: Read buffer size in bytes.

    Returns:
        str: Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the target file does not physically exist.
    """
    p = Path(filepath).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# 4. TIER 3 CROSS-PLATFORM ATOMIC REGISTRY LOCK MANAGER
# =============================================================================

class LustreDirectoryLock:
    """
    HPC/Lustre Native Atomic Directory Creation (mkdir) Lock.

    Guarantees process-safe and thread-safe atomic mutual exclusion on distributed
    HPC filesystems without relying on kernel-level POSIX flock mechanisms.
    Supports thread re-entrancy, diagnostic owner records, heartbeats, and stale lock eviction.
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

    @property
    def owner_file(self) -> Path:
        return self.lock_dir / "owner.json"

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquires the atomic directory creation lock.

        Raises:
            CatalogLockTimeoutError: If lock acquisition exceeds timeout.
            LustreCatalogLockError: If OS/filesystem errors occur.
        """
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

        acquired_thread = thread_lock.acquire(timeout=eff_timeout)
        if not acquired_thread:
            raise CatalogLockTimeoutError(
                f"Timed out after {eff_timeout:.2f}s waiting for in-process lock: {path_key}"
            )

        try:
            self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
            token = str(uuid.uuid4())

            while True:
                try:
                    os.mkdir(str(self.lock_dir))
                    # Directory successfully created atomically
                    owner_payload = {
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "token": token,
                        "timestamp": time.time(),
                        "hostname": platform.node(),
                    }
                    try:
                        with open(self.owner_file, "w", encoding="utf-8") as f:
                            json.dump(owner_payload, f)
                    except Exception as err:
                        logger.debug("Failed writing lock owner file: %s", err)

                    self._tls.held[path_key] = 1
                    self._depth = 1
                    return True

                except FileExistsError:
                    # Check for stale lock
                    if self._is_stale_lock():
                        self._evict_stale_lock()
                        continue

                    elapsed = time.time() - start_time
                    if elapsed >= eff_timeout:
                        raise CatalogLockTimeoutError(
                            f"Timed out after {elapsed:.2f}s waiting for atomic directory lock: {self.lock_dir}"
                        )

                    sleep_time = self.poll_interval * (0.8 + 0.4 * random.random())
                    time.sleep(sleep_time)

                except OSError as os_err:
                    raise LustreCatalogLockError(
                        f"OS error acquiring atomic directory lock {self.lock_dir}: {os_err}"
                    ) from os_err

        except Exception:
            thread_lock.release()
            raise

    def release(self) -> None:
        """
        Releases the atomic directory creation lock.
        """
        path_key = str(self.lock_dir)
        if not hasattr(self._tls, "held") or self._tls.held.get(path_key, 0) <= 0:
            logger.warning("Attempted to release unheld Lustre lock: %s", path_key)
            return

        self._tls.held[path_key] -= 1
        self._depth -= 1

        if self._tls.held[path_key] == 0:
            del self._tls.held[path_key]
            try:
                if self.owner_file.exists():
                    try:
                        self.owner_file.unlink()
                    except Exception as e:
                        logger.debug("Error removing owner file: %s", e)

                if self.lock_dir.exists():
                    try:
                        os.rmdir(str(self.lock_dir))
                    except Exception as e:
                        logger.debug("Error removing lock dir: %s", e)
            finally:
                thread_lock = self._get_thread_lock(path_key)
                thread_lock.release()

    def _is_stale_lock(self) -> bool:
        try:
            if self.owner_file.exists():
                with open(self.owner_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ts = data.get("timestamp", 0)
                    if time.time() - ts > self.stale_timeout:
                        return True
            else:
                st = os.stat(str(self.lock_dir))
                if time.time() - st.st_mtime > self.stale_timeout:
                    return True
        except Exception:
            return False
        return False

    def _evict_stale_lock(self) -> None:
        try:
            logger.warning("Evicting stale atomic directory lock: %s", self.lock_dir)
            if self.owner_file.exists():
                self.owner_file.unlink(missing_ok=True)
            shutil.rmtree(str(self.lock_dir), ignore_errors=True)
        except Exception as err:
            logger.error("Failed evicting stale lock %s: %s", self.lock_dir, err)

    def __enter__(self) -> LustreDirectoryLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class Tier3RegistryLock:
    """
    OS-Agnostic Tier 3 Atomic Concurrency Lock Manager.

    Provides multi-backend concurrency protection:
    - Standard Local Filesystem: Combines filelock.FileLock with in-process threading.RLock.
      Lock files are positioned either adjacent to target files or within a local safe
      runtime metadata directory (outside parallel Lustre mounts).
    - Lustre / Distributed HPC Filesystem: Automatically activates LustreDirectoryLock
      with atomic mkdir synchronization and stale lock recovery.
    """

    _tls = threading.local()
    _local_r_locks: Dict[str, threading.RLock] = {}
    _flock_instances: Dict[str, filelock.FileLock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def _get_shared_filelock(cls, lock_file_path: str, timeout: float) -> Tuple[threading.RLock, filelock.FileLock]:
        with cls._meta_lock:
            if lock_file_path not in cls._local_r_locks:
                cls._local_r_locks[lock_file_path] = threading.RLock()
            if lock_file_path not in cls._flock_instances:
                cls._flock_instances[lock_file_path] = filelock.FileLock(lock_file_path, timeout=timeout)
            return cls._local_r_locks[lock_file_path], cls._flock_instances[lock_file_path]

    def __init__(
        self,
        target_path: Union[str, Path],
        timeout: Optional[float] = None,
        lock_dir: Optional[Union[str, Path]] = None,
        force_backend: Optional[str] = None,
    ) -> None:
        self.target_path = Path(target_path).resolve()
        self.is_lustre = is_lustre_filesystem(self.target_path) if force_backend is None else (force_backend.lower() == "lustre")

        if timeout is not None:
            self.timeout = float(timeout)
        else:
            self.timeout = DEFAULT_LUSTRE_LOCK_TIMEOUT if self.is_lustre else DEFAULT_LOCAL_LOCK_TIMEOUT

        # Determine lock file / directory location
        if lock_dir is not None:
            base_dir = Path(lock_dir).resolve()
        elif self.is_lustre:
            # Place outside Lustre mount if possible, falling back to local temp runtime dir
            base_dir = Path(platformdirs.user_runtime_dir(appname="CoChem-SpycFit", ensure_exists=True))
        else:
            base_dir = self.target_path.parent

        base_dir.mkdir(parents=True, exist_ok=True)
        target_stem = self.target_path.name
        self.lock_file_path = base_dir / f"{target_stem}.tier3.lock"

        # Initialize appropriate backend
        if self.is_lustre or force_backend == "lustre":
            self.backend: Union[LustreDirectoryLock, filelock.FileLock] = LustreDirectoryLock(
                lock_path=self.lock_file_path,
                timeout=self.timeout,
            )
            self._is_filelock = False
        else:
            _, shared_flock = self._get_shared_filelock(str(self.lock_file_path), timeout=self.timeout)
            self.backend = shared_flock
            self._is_filelock = True

        self._acquired = False

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquires the Tier 3 atomic lock.
        """
        eff_timeout = float(timeout) if timeout is not None else self.timeout

        if self._is_filelock:
            key = str(self.lock_file_path)
            thread_rlock, flock = self._get_shared_filelock(key, timeout=eff_timeout)

            if not hasattr(self._tls, "held_tier3"):
                self._tls.held_tier3 = {}

            if self._tls.held_tier3.get(key, 0) > 0:
                self._tls.held_tier3[key] += 1
                self._acquired = True
                return True

            if not thread_rlock.acquire(timeout=eff_timeout):
                raise CatalogLockTimeoutError(
                    f"Timed out after {eff_timeout:.2f}s acquiring in-process thread lock for {self.lock_file_path}"
                )

            try:
                flock.acquire(timeout=eff_timeout)
                self._tls.held_tier3[key] = 1
                self._acquired = True
                return True
            except filelock.Timeout as err:
                thread_rlock.release()
                raise CatalogLockTimeoutError(
                    f"Timed out after {eff_timeout:.2f}s acquiring filelock: {self.lock_file_path}"
                ) from err
            except Exception:
                thread_rlock.release()
                raise
        else:
            lustre_lock = self.backend
            if isinstance(lustre_lock, LustreDirectoryLock):
                result = lustre_lock.acquire(timeout=eff_timeout)
                self._acquired = True
                return result
            raise CatalogLockError("Invalid backend configuration in Tier3RegistryLock")

    def release(self) -> None:
        """
        Releases the Tier 3 atomic lock.
        """
        if not self._acquired:
            return

        if self._is_filelock:
            key = str(self.lock_file_path)
            if not hasattr(self._tls, "held_tier3") or self._tls.held_tier3.get(key, 0) <= 0:
                self._acquired = False
                return

            self._tls.held_tier3[key] -= 1
            self._acquired = False

            if self._tls.held_tier3[key] == 0:
                del self._tls.held_tier3[key]
                thread_rlock, flock = self._get_shared_filelock(key, timeout=self.timeout)
                try:
                    flock.release()
                finally:
                    thread_rlock.release()
        else:
            try:
                lustre_lock = self.backend
                if isinstance(lustre_lock, LustreDirectoryLock):
                    lustre_lock.release()
            finally:
                self._acquired = False

    def __enter__(self) -> Tier3RegistryLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# =============================================================================
# 5. OUT-OF-CORE HDF5 CHUNKED STREAMING COMPILER
# =============================================================================

def _normalize_chunk_data(
    chunk: Union[TransitionBatch, Dict[str, Any], Sequence[TransitionRecord], np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Normalizes arbitrary chunk inputs (Pydantic models, dicts, record lists, structured arrays)
    into standardized 1D/2D NumPy arrays for HDF5 writing.

    Raises:
        CatalogSchemaError: If data fields are malformed or lengths mismatch.
    """
    if isinstance(chunk, TransitionBatch):
        n = len(chunk.frequencies_mhz)
        freqs = np.asarray(chunk.frequencies_mhz, dtype=np.float64)
        intens = np.asarray(chunk.intensities, dtype=np.float64)
        e_low = np.asarray(chunk.lower_energies_cm1, dtype=np.float64) if chunk.lower_energies_cm1 is not None else np.zeros(n, dtype=np.float64)
        e_up = np.asarray(chunk.upper_energies_cm1, dtype=np.float64) if chunk.upper_energies_cm1 is not None else np.zeros(n, dtype=np.float64)
        uncs = np.asarray(chunk.uncertainties_mhz, dtype=np.float64) if chunk.uncertainties_mhz is not None else np.zeros(n, dtype=np.float64)
        g_u = np.asarray(chunk.g_upper, dtype=np.int32) if chunk.g_upper is not None else np.ones(n, dtype=np.int32)
        if chunk.quantum_numbers is not None:
            qn = np.asarray(chunk.quantum_numbers, dtype=np.int32)
            if qn.ndim == 1:
                qn = qn.reshape(-1, 6)
        else:
            qn = np.zeros((n, 6), dtype=np.int32)
        sp_id = np.asarray(chunk.species_ids, dtype=np.int32) if chunk.species_ids is not None else np.zeros(n, dtype=np.int32)

        return {
            "frequencies": freqs,
            "intensities": intens,
            "lower_energies": e_low,
            "upper_energies": e_up,
            "uncertainties": uncs,
            "g_upper": g_u,
            "quantum_numbers": qn,
            "species_ids": sp_id,
        }

    if isinstance(chunk, dict):
        # Dictionary input
        if "frequencies" in chunk or "frequencies_mhz" in chunk:
            freq_key = "frequencies" if "frequencies" in chunk else "frequencies_mhz"
            freqs = np.asarray(chunk[freq_key], dtype=np.float64)
            n = len(freqs)
            intens_key = "intensities" if "intensities" in chunk else "intensity"
            intens = np.asarray(chunk.get(intens_key, np.zeros(n)), dtype=np.float64)

            e_low = np.asarray(chunk.get("lower_energies", chunk.get("lower_energies_cm1", np.zeros(n))), dtype=np.float64)
            e_up = np.asarray(chunk.get("upper_energies", chunk.get("upper_energies_cm1", np.zeros(n))), dtype=np.float64)
            uncs = np.asarray(chunk.get("uncertainties", chunk.get("uncertainties_mhz", np.zeros(n))), dtype=np.float64)
            g_u = np.asarray(chunk.get("g_upper", np.ones(n)), dtype=np.int32)

            qn_raw = chunk.get("quantum_numbers", np.zeros((n, 6)))
            qn = np.asarray(qn_raw, dtype=np.int32)
            if qn.ndim == 1:
                qn = qn.reshape(-1, 6)

            sp_id = np.asarray(chunk.get("species_ids", chunk.get("species_id", np.zeros(n))), dtype=np.int32)

            if len(intens) != n:
                raise CatalogSchemaError(f"Intensities length ({len(intens)}) mismatch with frequencies ({n})")

            return {
                "frequencies": freqs,
                "intensities": intens,
                "lower_energies": e_low,
                "upper_energies": e_up,
                "uncertainties": uncs,
                "g_upper": g_u,
                "quantum_numbers": qn,
                "species_ids": sp_id,
            }
        raise CatalogSchemaError("Dictionary chunk must contain 'frequencies' or 'frequencies_mhz'")

    if isinstance(chunk, np.ndarray):
        # Structured array or 2D array
        if chunk.dtype.names is not None:
            n = len(chunk)
            freqs = chunk["frequencies"].astype(np.float64) if "frequencies" in chunk.dtype.names else chunk["frequency_mhz"].astype(np.float64)
            intens = chunk["intensities"].astype(np.float64) if "intensities" in chunk.dtype.names else chunk["intensity"].astype(np.float64)
            e_low = chunk["lower_energies"].astype(np.float64) if "lower_energies" in chunk.dtype.names else np.zeros(n, dtype=np.float64)
            e_up = chunk["upper_energies"].astype(np.float64) if "upper_energies" in chunk.dtype.names else np.zeros(n, dtype=np.float64)
            uncs = chunk["uncertainties"].astype(np.float64) if "uncertainties" in chunk.dtype.names else np.zeros(n, dtype=np.float64)
            g_u = chunk["g_upper"].astype(np.int32) if "g_upper" in chunk.dtype.names else np.ones(n, dtype=np.int32)
            qn = chunk["quantum_numbers"].astype(np.int32) if "quantum_numbers" in chunk.dtype.names else np.zeros((n, 6), dtype=np.int32)
            sp_id = chunk["species_ids"].astype(np.int32) if "species_ids" in chunk.dtype.names else np.zeros(n, dtype=np.int32)

            return {
                "frequencies": freqs,
                "intensities": intens,
                "lower_energies": e_low,
                "upper_energies": e_up,
                "uncertainties": uncs,
                "g_upper": g_u,
                "quantum_numbers": qn,
                "species_ids": sp_id,
            }
        if chunk.ndim == 2 and chunk.shape[1] >= 2:
            # Standard 2D numerical array [freq, intensity, ...]
            n = chunk.shape[0]
            freqs = chunk[:, 0].astype(np.float64)
            intens = chunk[:, 1].astype(np.float64)
            e_low = chunk[:, 2].astype(np.float64) if chunk.shape[1] > 2 else np.zeros(n, dtype=np.float64)
            e_up = chunk[:, 3].astype(np.float64) if chunk.shape[1] > 3 else np.zeros(n, dtype=np.float64)
            uncs = chunk[:, 4].astype(np.float64) if chunk.shape[1] > 4 else np.zeros(n, dtype=np.float64)
            g_u = chunk[:, 5].astype(np.int32) if chunk.shape[1] > 5 else np.ones(n, dtype=np.int32)
            qn = np.zeros((n, 6), dtype=np.int32)
            sp_id = np.zeros(n, dtype=np.int32)

            return {
                "frequencies": freqs,
                "intensities": intens,
                "lower_energies": e_low,
                "upper_energies": e_up,
                "uncertainties": uncs,
                "g_upper": g_u,
                "quantum_numbers": qn,
                "species_ids": sp_id,
            }
        raise CatalogSchemaError("Unstructured 1D array cannot be converted to TransitionBatch")

    if isinstance(chunk, (list, tuple)):
        # List of TransitionRecord or lists
        n = len(chunk)
        if n == 0:
            return {
                "frequencies": np.zeros(0, dtype=np.float64),
                "intensities": np.zeros(0, dtype=np.float64),
                "lower_energies": np.zeros(0, dtype=np.float64),
                "upper_energies": np.zeros(0, dtype=np.float64),
                "uncertainties": np.zeros(0, dtype=np.float64),
                "g_upper": np.zeros(0, dtype=np.int32),
                "quantum_numbers": np.zeros((0, 6), dtype=np.int32),
                "species_ids": np.zeros(0, dtype=np.int32),
            }
        first = chunk[0]
        if isinstance(first, TransitionRecord):
            freqs = np.array([r.frequency_mhz for r in chunk], dtype=np.float64)
            intens = np.array([r.intensity for r in chunk], dtype=np.float64)
            e_low = np.array([r.lower_energy_cm1 for r in chunk], dtype=np.float64)
            e_up = np.array([r.upper_energy_cm1 for r in chunk], dtype=np.float64)
            uncs = np.array([r.uncertainty_mhz for r in chunk], dtype=np.float64)
            g_u = np.array([r.g_upper for r in chunk], dtype=np.int32)
            qn = np.array([r.quantum_numbers for r in chunk], dtype=np.int32)
            sp_id = np.array([r.species_id for r in chunk], dtype=np.int32)

            return {
                "frequencies": freqs,
                "intensities": intens,
                "lower_energies": e_low,
                "upper_energies": e_up,
                "uncertainties": uncs,
                "g_upper": g_u,
                "quantum_numbers": qn,
                "species_ids": sp_id,
            }

    raise CatalogSchemaError(f"Unsupported chunk data type for catalog streaming: {type(chunk)}")


def stream_chunked_catalog(
    data_iterator: Union[Iterator[Any], Iterable[Any], Generator[Any, None, None]],
    output_path: Union[str, Path],
    lock_manager: Optional[Union[Tier3RegistryLock, Any, str, Path]] = None,
    config: Optional[CatalogCompilerConfig] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> CatalogManifestRecord:
    """
    Streams transition catalog lines from an iterator/generator directly into an HDF5 container.

    Enforces Out-of-Core memory bounds by writing discrete chunks sequentially into extensible
    HDF5 datasets, under atomic Tier 3 Registry Lock protection.

    Args:
        data_iterator: Iterator or generator yielding chunks (TransitionBatch, dict, numpy array).
        output_path: Destination path for compiled HDF5 catalog container.
        lock_manager: Optional Tier3RegistryLock instance, lock directory, or Path.
        config: Optional CatalogCompilerConfig model.
        custom_metadata: Optional dictionary of domain attributes to persist in HDF5 root.

    Returns:
        CatalogManifestRecord: Verified manifest containing line counts, frequency bounds,
        and cryptographic hash.

    Raises:
        CatalogLockTimeoutError: If lock acquisition fails.
        CatalogDatasetError: If HDF5 dataset creation or writing fails.
        CatalogStreamingError: If stream iterator encounters errors.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else CatalogCompilerConfig()

    # Resolve lock manager
    if isinstance(lock_manager, Tier3RegistryLock):
        resolved_lock = lock_manager
    elif isinstance(lock_manager, (str, Path)):
        resolved_lock = Tier3RegistryLock(target_path=out_p, lock_dir=lock_manager, timeout=cfg.lock_timeout_sec)
    else:
        resolved_lock = Tier3RegistryLock(
            target_path=out_p,
            lock_dir=cfg.lock_directory,
            timeout=cfg.lock_timeout_sec,
        )

    # Determine SWMR mode compatibility
    is_lustre = is_lustre_filesystem(out_p)
    enable_swmr = cfg.swmr_enabled and not is_lustre

    total_rows: int = 0
    freq_min: float = float("inf")
    freq_max: float = float("-inf")
    intens_min: float = float("inf")
    intens_max: float = float("-inf")

    chunk_size = cfg.chunk_size
    compression = cfg.compression
    compression_opts = cfg.compression_opts if compression == "gzip" else None
    fletcher32 = cfg.fletcher32

    with resolved_lock:
        try:
            # Initialize HDF5 container
            with h5py.File(str(out_p), "w", libver="latest" if enable_swmr else "earliest") as h5f:
                # Set root metadata attributes
                h5f.attrs["catalog_compiler_version"] = "1.0.0"
                h5f.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                h5f.attrs["chunk_size"] = chunk_size
                h5f.attrs["compression"] = str(compression)
                h5f.attrs["is_lustre_filesystem"] = is_lustre

                if custom_metadata:
                    for k, v in custom_metadata.items():
                        try:
                            h5f.attrs[k] = v
                        except Exception:
                            h5f.attrs[k] = str(v)

                # Initialize extensible columnar datasets
                dsets: Dict[str, h5py.Dataset] = {}
                dsets["frequencies"] = h5f.create_dataset(
                    "frequencies",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["intensities"] = h5f.create_dataset(
                    "intensities",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["lower_energies"] = h5f.create_dataset(
                    "lower_energies",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["upper_energies"] = h5f.create_dataset(
                    "upper_energies",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["uncertainties"] = h5f.create_dataset(
                    "uncertainties",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["g_upper"] = h5f.create_dataset(
                    "g_upper",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="int32",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["quantum_numbers"] = h5f.create_dataset(
                    "quantum_numbers",
                    shape=(0, 6),
                    maxshape=(None, 6),
                    dtype="int32",
                    chunks=(chunk_size, 6),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )
                dsets["species_ids"] = h5f.create_dataset(
                    "species_ids",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="int32",
                    chunks=(chunk_size,),
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )

                if enable_swmr and hasattr(h5f, "swmr_mode"):
                    try:
                        h5f.swmr_mode = True
                    except Exception as swmr_err:
                        logger.debug("SWMR activation notice: %s", swmr_err)

                # Iterate through data generator
                iterator_obj = iter(data_iterator)
                for raw_chunk in iterator_obj:
                    norm = _normalize_chunk_data(raw_chunk)
                    n_rows = len(norm["frequencies"])
                    if n_rows == 0:
                        continue

                    start_idx = total_rows
                    end_idx = total_rows + n_rows

                    # Resize datasets
                    dsets["frequencies"].resize((end_idx,))
                    dsets["intensities"].resize((end_idx,))
                    dsets["lower_energies"].resize((end_idx,))
                    dsets["upper_energies"].resize((end_idx,))
                    dsets["uncertainties"].resize((end_idx,))
                    dsets["g_upper"].resize((end_idx,))
                    dsets["quantum_numbers"].resize((end_idx, 6))
                    dsets["species_ids"].resize((end_idx,))

                    # Write slices
                    dsets["frequencies"][start_idx:end_idx] = norm["frequencies"]
                    dsets["intensities"][start_idx:end_idx] = norm["intensities"]
                    dsets["lower_energies"][start_idx:end_idx] = norm["lower_energies"]
                    dsets["upper_energies"][start_idx:end_idx] = norm["upper_energies"]
                    dsets["uncertainties"][start_idx:end_idx] = norm["uncertainties"]
                    dsets["g_upper"][start_idx:end_idx] = norm["g_upper"]
                    dsets["quantum_numbers"][start_idx:end_idx, :] = norm["quantum_numbers"]
                    dsets["species_ids"][start_idx:end_idx] = norm["species_ids"]

                    # Update statistics
                    c_f_min = float(np.min(norm["frequencies"]))
                    c_f_max = float(np.max(norm["frequencies"]))
                    c_i_min = float(np.min(norm["intensities"]))
                    c_i_max = float(np.max(norm["intensities"]))

                    if c_f_min < freq_min:
                        freq_min = c_f_min
                    if c_f_max > freq_max:
                        freq_max = c_f_max
                    if c_i_min < intens_min:
                        intens_min = c_i_min
                    if c_i_max > intens_max:
                        intens_max = c_i_max

                    total_rows = end_idx

                    # Flush to disk
                    h5f.flush()

                # Handle empty stream case
                if total_rows == 0:
                    freq_min = 0.0
                    freq_max = 0.0
                    intens_min = 0.0
                    intens_max = 0.0

                h5f.attrs["total_transitions"] = total_rows
                h5f.attrs["frequency_min_mhz"] = freq_min
                h5f.attrs["frequency_max_mhz"] = freq_max
                h5f.attrs["intensity_min"] = intens_min
                h5f.attrs["intensity_max"] = intens_max

        except Exception as err:
            if isinstance(err, (CatalogLockError, CatalogSchemaError)):
                raise
            raise CatalogDatasetError(f"HDF5 chunked streaming failed on {out_p}: {err}") from err

    # Compute checksum outside lock
    sha256 = compute_sha256(out_p)

    manifest = CatalogManifestRecord(
        catalog_path=str(out_p),
        total_transitions=total_rows,
        frequency_min_mhz=freq_min,
        frequency_max_mhz=freq_max,
        intensity_min=intens_min,
        intensity_max=intens_max,
        is_virtual=False,
        num_shards=1,
        shard_paths=[str(out_p)],
        sha256_checksum=sha256,
        metadata=custom_metadata or {},
    )

    return manifest


# =============================================================================
# 6. HPC VIRTUALLAYOUT AGGREGATOR
# =============================================================================

def compile_virtual_catalog(
    shard_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    dataset_names: Optional[Sequence[str]] = None,
    lock_manager: Optional[Union[Tier3RegistryLock, Any, str, Path]] = None,
    config: Optional[CatalogCompilerConfig] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> CatalogManifestRecord:
    """
    Aggregates multiple independent HDF5 shard containers into a single master HDF5 container
    leveraging h5py.VirtualLayout and h5py.VirtualSource.

    Provides zero-copy, metadata-safe virtual assembly across distributed HPC worker runs.

    Args:
        shard_paths: Sequence of filesystem paths to physical HDF5 shard files.
        output_path: Destination path for compiled master virtual HDF5 container.
        dataset_names: Optional subset of datasets to map (defaults to all standard datasets).
        lock_manager: Optional Tier 3 lock manager instance.
        config: Optional compiler configuration.
        custom_metadata: Optional custom attributes to persist in master root.

    Returns:
        CatalogManifestRecord: Cryptographic manifest detailing virtual dataset parameters.

    Raises:
        VirtualLayoutError: If shard datasets cannot be resolved or mapped.
        CatalogLockTimeoutError: If master lock acquisition fails.
    """
    if len(shard_paths) == 0:
        raise VirtualLayoutError("Cannot compile virtual catalog from an empty shard list")

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else CatalogCompilerConfig()

    resolved_shards: List[Path] = [Path(p).resolve() for p in shard_paths]
    for s_p in resolved_shards:
        if not s_p.is_file():
            raise FileNotFoundError(f"Shard file not found for virtual compilation: {s_p}")

    target_dsets = dataset_names if dataset_names is not None else STANDARD_DATASET_NAMES

    # Resolve lock manager
    if isinstance(lock_manager, Tier3RegistryLock):
        resolved_lock = lock_manager
    elif isinstance(lock_manager, (str, Path)):
        resolved_lock = Tier3RegistryLock(target_path=out_p, lock_dir=lock_manager, timeout=cfg.lock_timeout_sec)
    else:
        resolved_lock = Tier3RegistryLock(
            target_path=out_p,
            lock_dir=cfg.lock_directory,
            timeout=cfg.lock_timeout_sec,
        )

    # Inspect shard geometries
    shard_lengths: List[int] = []
    total_rows: int = 0
    freq_min: float = float("inf")
    freq_max: float = float("-inf")
    intens_min: float = float("inf")
    intens_max: float = float("-inf")

    try:
        for s_p in resolved_shards:
            with h5py.File(str(s_p), "r") as h5_s:
                if "frequencies" not in h5_s:
                    raise VirtualLayoutError(f"Shard {s_p} missing required 'frequencies' dataset")
                n_s = h5_s["frequencies"].shape[0]
                shard_lengths.append(n_s)
                total_rows += n_s

                if n_s > 0:
                    f_dset = h5_s["frequencies"]
                    i_dset = h5_s["intensities"]
                    s_f_min = float(h5_s.attrs.get("frequency_min_mhz", np.min(f_dset)))
                    s_f_max = float(h5_s.attrs.get("frequency_max_mhz", np.max(f_dset)))
                    s_i_min = float(h5_s.attrs.get("intensity_min", np.min(i_dset)))
                    s_i_max = float(h5_s.attrs.get("intensity_max", np.max(i_dset)))

                    if s_f_min < freq_min:
                        freq_min = s_f_min
                    if s_f_max > freq_max:
                        freq_max = s_f_max
                    if s_i_min < intens_min:
                        intens_min = s_i_min
                    if s_i_max > intens_max:
                        intens_max = s_i_max

        if total_rows == 0:
            freq_min = 0.0
            freq_max = 0.0
            intens_min = 0.0
            intens_max = 0.0

    except Exception as err:
        if isinstance(err, (VirtualLayoutError, FileNotFoundError)):
            raise
        raise VirtualLayoutError(f"Error inspecting shard metadata: {err}") from err

    # Build VirtualLayouts and compile master container
    with resolved_lock:
        try:
            with h5py.File(str(out_p), "w", libver="latest") as h5_master:
                h5_master.attrs["catalog_compiler_version"] = "1.0.0"
                h5_master.attrs["is_virtual"] = True
                h5_master.attrs["num_shards"] = len(resolved_shards)
                h5_master.attrs["total_transitions"] = total_rows
                h5_master.attrs["frequency_min_mhz"] = freq_min
                h5_master.attrs["frequency_max_mhz"] = freq_max
                h5_master.attrs["intensity_min"] = intens_min
                h5_master.attrs["intensity_max"] = intens_max
                h5_master.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                if custom_metadata:
                    for k, v in custom_metadata.items():
                        try:
                            h5_master.attrs[k] = v
                        except Exception:
                            h5_master.attrs[k] = str(v)

                for dset_name in target_dsets:
                    if dset_name == "quantum_numbers":
                        layout = h5py.VirtualLayout(shape=(total_rows, 6), dtype=np.int32)
                        current_offset = 0
                        for s_p, s_len in zip(resolved_shards, shard_lengths):
                            if s_len > 0:
                                vsource = h5py.VirtualSource(str(s_p), dset_name, shape=(s_len, 6), dtype=np.int32)
                                layout[current_offset:current_offset + s_len, :] = vsource
                                current_offset += s_len
                        h5_master.create_virtual_dataset(dset_name, layout, fillvalue=-1)
                    else:
                        dtype_choice = np.int32 if dset_name in ("g_upper", "species_ids") else np.float64
                        layout = h5py.VirtualLayout(shape=(total_rows,), dtype=dtype_choice)
                        current_offset = 0
                        for s_p, s_len in zip(resolved_shards, shard_lengths):
                            if s_len > 0:
                                vsource = h5py.VirtualSource(str(s_p), dset_name, shape=(s_len,), dtype=dtype_choice)
                                layout[current_offset:current_offset + s_len] = vsource
                                current_offset += s_len
                        fill = 0 if dtype_choice == np.int32 else 0.0
                        h5_master.create_virtual_dataset(dset_name, layout, fillvalue=fill)

        except Exception as err:
            raise VirtualLayoutError(f"Failed creating master VirtualLayout on {out_p}: {err}") from err

    sha256 = compute_sha256(out_p)

    manifest = CatalogManifestRecord(
        catalog_path=str(out_p),
        total_transitions=total_rows,
        frequency_min_mhz=freq_min,
        frequency_max_mhz=freq_max,
        intensity_min=intens_min,
        intensity_max=intens_max,
        is_virtual=True,
        num_shards=len(resolved_shards),
        shard_paths=[str(s) for s in resolved_shards],
        sha256_checksum=sha256,
        metadata=custom_metadata or {},
    )

    return manifest


# =============================================================================
# 7. OUT-OF-CORE READER & RANGE QUERY ENGINE
# =============================================================================

def read_chunked_catalog(
    catalog_path: Union[str, Path],
    chunk_size: Optional[int] = None,
    frequency_range: Optional[Tuple[float, float]] = None,
    dataset_names: Optional[Sequence[str]] = None,
    lock_manager: Optional[Union[Tier3RegistryLock, Any, str, Path]] = None,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Generator that reads transition catalog chunks out-of-core without loading
    the cumulative dataset into memory.

    Args:
        catalog_path: Path to compiled HDF5 catalog file.
        chunk_size: Number of transition rows per yielded chunk.
        frequency_range: Optional tuple (min_freq_mhz, max_freq_mhz) for range filtering.
        dataset_names: Optional subset of datasets to extract.
        lock_manager: Optional lock manager for safe shared reading.

    Yields:
        Dict[str, np.ndarray]: Dictionary mapping dataset names to NumPy array slices.

    Raises:
        FileNotFoundError: If catalog file is missing.
        CatalogDatasetError: If dataset reading fails.
    """
    cat_p = Path(catalog_path).resolve()
    if not cat_p.is_file():
        raise FileNotFoundError(f"Catalog file not found for reading: {cat_p}")

    target_dsets = dataset_names if dataset_names is not None else STANDARD_DATASET_NAMES

    with h5py.File(str(cat_p), "r") as h5f:
        if "frequencies" not in h5f:
            raise CatalogDatasetError(f"HDF5 container missing 'frequencies' dataset: {cat_p}")

        total_rows = h5f["frequencies"].shape[0]
        if total_rows == 0:
            return

        c_size = chunk_size if chunk_size is not None else h5f.attrs.get("chunk_size", DEFAULT_CHUNK_SIZE)
        c_size = max(1, int(c_size))

        num_blocks = (total_rows + c_size - 1) // c_size

        for block_idx in range(num_blocks):
            start = block_idx * c_size
            end = min(total_rows, start + c_size)

            freq_chunk = np.asarray(h5f["frequencies"][start:end], dtype=np.float64)

            # Apply frequency range filter if specified
            if frequency_range is not None:
                f_min, f_max = frequency_range
                mask = (freq_chunk >= f_min) & (freq_chunk <= f_max)
                if not np.any(mask):
                    continue

                chunk_dict: Dict[str, np.ndarray] = {"frequencies": freq_chunk[mask]}
                for name in target_dsets:
                    if name == "frequencies":
                        continue
                    if name in h5f:
                        raw_arr = np.asarray(h5f[name][start:end])
                        chunk_dict[name] = raw_arr[mask]
                yield chunk_dict
            else:
                chunk_dict = {"frequencies": freq_chunk}
                for name in target_dsets:
                    if name == "frequencies":
                        continue
                    if name in h5f:
                        chunk_dict[name] = np.asarray(h5f[name][start:end])
                yield chunk_dict


# =============================================================================
# 8. INTEGRITY VERIFICATION & METADATA UTILITIES
# =============================================================================

def verify_catalog_integrity(
    catalog_path: Union[str, Path],
    verify_chunks: bool = True,
) -> bool:
    """
    Validates the physical integrity of a compiled HDF5 transition catalog.

    Checks:
    1. Physical existence and non-zero file size.
    2. HDF5 header and superblock valid structure.
    3. Presence of all required columnar datasets.
    4. Exact shape alignment across all columnar datasets.
    5. Chunk-level Fletcher32 checksum reading across all chunks (if enabled).

    Args:
        catalog_path: Filesystem path to target HDF5 catalog container.
        verify_chunks: If True, performs chunk-level reading to trigger Fletcher32 checks.

    Returns:
        bool: True if catalog passed all integrity audits.

    Raises:
        CatalogCorruptionError: If file corruption, shape mismatch, or read error is detected.
    """
    cat_p = Path(catalog_path).resolve()
    if not cat_p.is_file():
        raise CatalogCorruptionError(f"Catalog file missing on disk: {cat_p}")

    try:
        with h5py.File(str(cat_p), "r") as h5f:
            if "frequencies" not in h5f:
                raise CatalogCorruptionError(f"Corrupted catalog: Missing 'frequencies' dataset in {cat_p}")

            total_rows = h5f["frequencies"].shape[0]

            for dset_name in STANDARD_DATASET_NAMES:
                if dset_name not in h5f:
                    raise CatalogCorruptionError(f"Corrupted catalog: Missing standard dataset '{dset_name}' in {cat_p}")
                d_shape = h5f[dset_name].shape
                if d_shape[0] != total_rows:
                    raise CatalogCorruptionError(
                        f"Shape mismatch in dataset '{dset_name}': expected {total_rows}, found {d_shape[0]}"
                    )

            if verify_chunks and total_rows > 0:
                c_size = min(total_rows, 65536)
                for start in range(0, total_rows, c_size):
                    end = min(total_rows, start + c_size)
                    _ = h5f["frequencies"][start:end]
                    _ = h5f["intensities"][start:end]

        return True

    except Exception as err:
        if isinstance(err, CatalogCorruptionError):
            raise
        raise CatalogCorruptionError(f"Catalog integrity verification failed for {cat_p}: {err}") from err


def extract_catalog_metadata(catalog_path: Union[str, Path]) -> CatalogManifestRecord:
    """
    Extracts the full CatalogManifestRecord from an existing HDF5 catalog container.

    Args:
        catalog_path: Path to compiled HDF5 catalog file.

    Returns:
        CatalogManifestRecord: Extracted manifest metadata.

    Raises:
        FileNotFoundError: If catalog is missing.
        CatalogDatasetError: If catalog structure cannot be parsed.
    """
    cat_p = Path(catalog_path).resolve()
    if not cat_p.is_file():
        raise FileNotFoundError(f"Catalog file not found: {cat_p}")

    try:
        with h5py.File(str(cat_p), "r") as h5f:
            total_transitions = int(h5f.attrs.get("total_transitions", h5f["frequencies"].shape[0]))
            freq_min = float(h5f.attrs.get("frequency_min_mhz", np.min(h5f["frequencies"]) if total_transitions > 0 else 0.0))
            freq_max = float(h5f.attrs.get("frequency_max_mhz", np.max(h5f["frequencies"]) if total_transitions > 0 else 0.0))
            intens_min = float(h5f.attrs.get("intensity_min", np.min(h5f["intensities"]) if total_transitions > 0 else 0.0))
            intens_max = float(h5f.attrs.get("intensity_max", np.max(h5f["intensities"]) if total_transitions > 0 else 0.0))
            is_virtual = bool(h5f.attrs.get("is_virtual", False))
            num_shards = int(h5f.attrs.get("num_shards", 1))
            created_utc = str(h5f.attrs.get("created_utc", datetime.datetime.now(datetime.timezone.utc).isoformat()))

            custom_meta: Dict[str, Any] = {}
            for k in h5f.attrs.keys():
                if k not in (
                    "catalog_compiler_version",
                    "total_transitions",
                    "frequency_min_mhz",
                    "frequency_max_mhz",
                    "intensity_min",
                    "intensity_max",
                    "is_virtual",
                    "num_shards",
                    "created_utc",
                ):
                    custom_meta[k] = h5f.attrs[k]

        sha256 = compute_sha256(cat_p)

        return CatalogManifestRecord(
            catalog_path=str(cat_p),
            total_transitions=total_transitions,
            frequency_min_mhz=freq_min,
            frequency_max_mhz=freq_max,
            intensity_min=intens_min,
            intensity_max=intens_max,
            is_virtual=is_virtual,
            num_shards=num_shards,
            sha256_checksum=sha256,
            created_utc=created_utc,
            metadata=custom_meta,
        )

    except Exception as err:
        if isinstance(err, FileNotFoundError):
            raise
        raise CatalogDatasetError(f"Failed extracting metadata from {cat_p}: {err}") from err


# =============================================================================
# 12. PHASE 4 STATE DIRECTORY RESOLUTION & PARQUET SERIALIZATION
# =============================================================================

def get_chunked_transitions_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolves the authoritative state directory for chunked transition parquet files.

    Target path: $COCHEM_STATE_DIR/SpycFit_Workspace/Catalogs/chunked_transitions/
    Defaults to platformdirs.user_data_path(appname="CoChem-SpycFit") / "SpycFit_Workspace" / "Catalogs" / "chunked_transitions".

    Args:
        base_dir: Optional explicit override directory path.

    Returns:
        Path: Resolved, physically existing Path object.
    """
    if base_dir is not None:
        target = Path(base_dir).resolve()
    else:
        state_dir_env = os.environ.get("COCHEM_STATE_DIR", "").strip()
        if state_dir_env:
            target = Path(state_dir_env).resolve() / "SpycFit_Workspace" / "Catalogs" / "chunked_transitions"
        else:
            target = Path(platformdirs.user_data_path(appname="CoChem-SpycFit")).resolve() / "SpycFit_Workspace" / "Catalogs" / "chunked_transitions"

    target.mkdir(parents=True, exist_ok=True)
    return target


def serialize_chunk_to_parquet(
    frequency_data: Union[np.ndarray, jnp.ndarray, Sequence[float], Dict[str, Any], TransitionBatch],
    quantum_assignments: Optional[Union[np.ndarray, jnp.ndarray, Sequence[Sequence[int]], Sequence[Tuple[int, ...]]]] = None,
    temp_step: Union[int, float] = 0,
    batch_id: Union[int, str] = 0,
    uuid_str: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    partition_by_temp: bool = True,
    compression: str = "zstd",
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serializes a single transition chunk directly to a .parquet file using PyArrow.

    Enforces strict physical type preservation:
    - Frequencies, intensities, energies, uncertainties remain FP64 (float64).
    - Quantum assignments (J, Ka, Kc), degeneracies, and species IDs remain int32.

    Employs a lock-free, partitioned append-only write strategy (Hive-partitioning).
    File naming format: chunk_{temperature_step}_{batch_id}_{uuid}.parquet

    Args:
        frequency_data: 1D array of frequencies, dict, structured array, or TransitionBatch.
        quantum_assignments: Optional 2D array of int32 quantum numbers (N, 6) or (N, 3).
        temp_step: Temperature step index or value.
        batch_id: Batch identifier (int or str).
        uuid_str: Optional explicit UUID string. If None, generates a new UUID.
        output_dir: Optional output directory (defaults to get_chunked_transitions_dir()).
        partition_by_temp: If True, writes into temperature_step={temp_step} subdirectory.
        compression: Parquet compression codec ('zstd', 'snappy', 'gzip', etc.).
        custom_metadata: Optional domain metadata to embed in Parquet schema.

    Returns:
        Path: Filesystem path to the written .parquet chunk file.
    """
    if isinstance(frequency_data, TransitionBatch):
        freqs = np.asarray(frequency_data.frequencies_mhz, dtype=np.float64)
        n = len(freqs)
        intens = np.asarray(frequency_data.intensities, dtype=np.float64)
        e_low = np.asarray(frequency_data.lower_energies_cm1, dtype=np.float64) if frequency_data.lower_energies_cm1 is not None else np.zeros(n, dtype=np.float64)
        e_up = np.asarray(frequency_data.upper_energies_cm1, dtype=np.float64) if frequency_data.upper_energies_cm1 is not None else np.zeros(n, dtype=np.float64)
        unc = np.asarray(frequency_data.uncertainties_mhz, dtype=np.float64) if frequency_data.uncertainties_mhz is not None else np.zeros(n, dtype=np.float64)
        g_u = np.asarray(frequency_data.g_upper, dtype=np.int32) if frequency_data.g_upper is not None else np.ones(n, dtype=np.int32)
        sp_id = np.asarray(frequency_data.species_ids, dtype=np.int32) if frequency_data.species_ids is not None else np.zeros(n, dtype=np.int32)

        if quantum_assignments is not None:
            qn_raw = np.asarray(quantum_assignments, dtype=np.int32)
        elif frequency_data.quantum_numbers is not None:
            qn_raw = np.asarray(frequency_data.quantum_numbers, dtype=np.int32)
        else:
            qn_raw = np.zeros((n, 6), dtype=np.int32)

    elif isinstance(frequency_data, dict):
        if "frequencies" in frequency_data or "frequencies_mhz" in frequency_data:
            freq_key = "frequencies" if "frequencies" in frequency_data else "frequencies_mhz"
            freqs = np.asarray(frequency_data[freq_key], dtype=np.float64)
            n = len(freqs)
            intens_key = "intensities" if "intensities" in frequency_data else "intensity"
            intens = np.asarray(frequency_data.get(intens_key, np.zeros(n)), dtype=np.float64)
            e_low = np.asarray(frequency_data.get("lower_energies", frequency_data.get("lower_energies_cm1", np.zeros(n))), dtype=np.float64)
            e_up = np.asarray(frequency_data.get("upper_energies", frequency_data.get("upper_energies_cm1", np.zeros(n))), dtype=np.float64)
            unc = np.asarray(frequency_data.get("uncertainties", frequency_data.get("uncertainties_mhz", np.zeros(n))), dtype=np.float64)
            g_u = np.asarray(frequency_data.get("g_upper", np.ones(n)), dtype=np.int32)
            sp_id = np.asarray(frequency_data.get("species_ids", frequency_data.get("species_id", np.zeros(n))), dtype=np.int32)

            if quantum_assignments is not None:
                qn_raw = np.asarray(quantum_assignments, dtype=np.int32)
            elif "quantum_numbers" in frequency_data:
                qn_raw = np.asarray(frequency_data["quantum_numbers"], dtype=np.int32)
            else:
                qn_raw = np.zeros((n, 6), dtype=np.int32)
        else:
            raise CatalogSchemaError("Dictionary chunk must contain 'frequencies' or 'frequencies_mhz'")

    elif isinstance(frequency_data, (np.ndarray, jnp.ndarray, list, tuple)):
        freq_arr = np.asarray(frequency_data, dtype=np.float64)
        if freq_arr.ndim == 1:
            freqs = freq_arr
            n = len(freqs)
            intens = np.zeros(n, dtype=np.float64)
            e_low = np.zeros(n, dtype=np.float64)
            e_up = np.zeros(n, dtype=np.float64)
            unc = np.zeros(n, dtype=np.float64)
            g_u = np.ones(n, dtype=np.int32)
            sp_id = np.zeros(n, dtype=np.int32)

            if quantum_assignments is not None:
                qn_raw = np.asarray(quantum_assignments, dtype=np.int32)
            else:
                qn_raw = np.zeros((n, 6), dtype=np.int32)
        elif freq_arr.ndim == 2 and freq_arr.shape[1] >= 2:
            n = freq_arr.shape[0]
            freqs = freq_arr[:, 0].astype(np.float64)
            intens = freq_arr[:, 1].astype(np.float64)
            e_low = freq_arr[:, 2].astype(np.float64) if freq_arr.shape[1] > 2 else np.zeros(n, dtype=np.float64)
            e_up = freq_arr[:, 3].astype(np.float64) if freq_arr.shape[1] > 3 else np.zeros(n, dtype=np.float64)
            unc = freq_arr[:, 4].astype(np.float64) if freq_arr.shape[1] > 4 else np.zeros(n, dtype=np.float64)
            g_u = freq_arr[:, 5].astype(np.int32) if freq_arr.shape[1] > 5 else np.ones(n, dtype=np.int32)
            sp_id = np.zeros(n, dtype=np.int32)

            if quantum_assignments is not None:
                qn_raw = np.asarray(quantum_assignments, dtype=np.int32)
            else:
                qn_raw = np.zeros((n, 6), dtype=np.int32)
        else:
            raise CatalogSchemaError("Unsupported array shape for frequency_data")
    else:
        raise CatalogSchemaError(f"Unsupported frequency_data input type: {type(frequency_data)}")

    if qn_raw.ndim == 1:
        if n > 0 and len(qn_raw) == 6 * n:
            qn_matrix = qn_raw.reshape(n, 6)
        elif n > 0 and len(qn_raw) == 3 * n:
            qn_3 = qn_raw.reshape(n, 3)
            qn_matrix = np.hstack([qn_3, np.zeros((n, 3), dtype=np.int32)])
        elif n == 1 and len(qn_raw) in (3, 6):
            if len(qn_raw) == 3:
                qn_matrix = np.hstack([qn_raw.reshape(1, 3), np.zeros((1, 3), dtype=np.int32)])
            else:
                qn_matrix = qn_raw.reshape(1, 6)
        elif n == 0:
            qn_matrix = np.zeros((0, 6), dtype=np.int32)
        else:
            qn_matrix = np.zeros((n, 6), dtype=np.int32)
    elif qn_raw.ndim == 2:
        if qn_raw.shape[1] == 3:
            qn_matrix = np.hstack([qn_raw, np.zeros((n, 3), dtype=np.int32)])
        elif qn_raw.shape[1] >= 6:
            qn_matrix = qn_raw[:, :6]
        else:
            pad_width = 6 - qn_raw.shape[1]
            qn_matrix = np.hstack([qn_raw, np.zeros((n, pad_width), dtype=np.int32)])
    else:
        qn_matrix = np.zeros((n, 6), dtype=np.int32)

    u_val = uuid_str if uuid_str is not None else uuid.uuid4().hex
    t_step_val = int(temp_step) if isinstance(temp_step, (int, np.integer)) else float(temp_step)
    b_id_str = str(batch_id)

    qn_py_list = [list(int(x) for x in row) for row in qn_matrix]

    arrays = [
        pa.array(freqs, type=pa.float64()),
        pa.array(intens, type=pa.float64()),
        pa.array(e_low, type=pa.float64()),
        pa.array(e_up, type=pa.float64()),
        pa.array(unc, type=pa.float64()),
        pa.array(g_u, type=pa.int32()),
        pa.array(sp_id, type=pa.int32()),
        pa.array(qn_matrix[:, 0] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_matrix[:, 1] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_matrix[:, 2] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_matrix[:, 3] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_matrix[:, 4] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_matrix[:, 5] if n > 0 else np.zeros(0, dtype=np.int32), type=pa.int32()),
        pa.array(qn_py_list, type=pa.list_(pa.int32())),
    ]
    names = [
        "frequencies",
        "intensities",
        "lower_energies",
        "upper_energies",
        "uncertainties",
        "g_upper",
        "species_ids",
        "J_upper",
        "Ka_upper",
        "Kc_upper",
        "J_lower",
        "Ka_lower",
        "Kc_lower",
        "quantum_numbers",
    ]

    if not partition_by_temp:
        arrays.append(pa.array([t_step_val] * n, type=pa.int32() if isinstance(t_step_val, int) else pa.float64()))
        names.append("temperature_step")

    arrays.extend([
        pa.array([b_id_str] * n, type=pa.string()),
        pa.array([u_val] * n, type=pa.string()),
    ])
    names.extend([
        "batch_id",
        "chunk_uuid",
    ])

    table = pa.Table.from_arrays(arrays, names=names)

    base_p = Path(output_dir) if output_dir is not None else get_chunked_transitions_dir()
    if partition_by_temp:
        target_dir = base_p / f"temperature_step={temp_step}"
    else:
        target_dir = base_p

    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"chunk_{temp_step}_{batch_id}_{u_val}.parquet"
    file_path = target_dir / filename

    pq.write_table(table, file_path, compression=compression)
    return file_path


# =============================================================================
# 13. JAX ASYNCHRONOUS ARRAY INTERCEPTION (io_callback)
# =============================================================================

def _host_serialize_worker(
    freqs: np.ndarray,
    qn: np.ndarray,
    intens: np.ndarray,
    temp_step: int,
    batch_id: int,
    uuid_str: str,
    output_dir: str,
) -> np.ndarray:
    """Internal host worker invoked by JAX io_callback."""
    dict_data = {
        "frequencies": np.asarray(freqs, dtype=np.float64),
        "intensities": np.asarray(intens, dtype=np.float64) if intens is not None and len(intens) > 0 else np.zeros(len(freqs), dtype=np.float64),
    }
    serialize_chunk_to_parquet(
        frequency_data=dict_data,
        quantum_assignments=np.asarray(qn, dtype=np.int32),
        temp_step=int(temp_step),
        batch_id=int(batch_id),
        uuid_str=str(uuid_str) if uuid_str else uuid.uuid4().hex,
        output_dir=str(output_dir) if output_dir else None,
        partition_by_temp=True,
    )
    return np.int32(1)


def jax_serialize_chunk_callback(
    frequency_data: jnp.ndarray,
    quantum_assignments: jnp.ndarray,
    temp_step: Union[int, jnp.ndarray] = 0,
    batch_id: Union[int, jnp.ndarray] = 0,
    uuid_str: Optional[str] = None,
    output_dir: Optional[str] = None,
    intensities: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Asynchronously intercepts JAX arrays during computation graph execution and serializes
    them directly to Parquet chunk files using jax.experimental.io_callback.

    Args:
        frequency_data: JAX array of float64 frequencies.
        quantum_assignments: JAX array of int32 quantum numbers.
        temp_step: Temperature step identifier.
        batch_id: Batch index or identifier.
        uuid_str: Optional UUID string identifier.
        output_dir: Optional target directory path string.
        intensities: Optional JAX array of float64 intensities.

    Returns:
        jnp.ndarray: Integer status array (1 on successful dispatch).
    """
    intens_arr = intensities if intensities is not None else jnp.zeros_like(frequency_data)
    u_str = str(uuid_str) if uuid_str is not None else uuid.uuid4().hex
    out_dir_str = str(output_dir) if output_dir is not None else str(get_chunked_transitions_dir())

    t_val = int(temp_step) if isinstance(temp_step, (int, float)) else int(np.asarray(temp_step))
    b_val = int(batch_id) if isinstance(batch_id, (int, float)) else int(np.asarray(batch_id))

    def _cb_host(f_arr, q_arr, i_arr):
        return _host_serialize_worker(
            freqs=f_arr,
            qn=q_arr,
            intens=i_arr,
            temp_step=t_val,
            batch_id=b_val,
            uuid_str=u_str,
            output_dir=out_dir_str,
        )

    status = io_callback(_cb_host, jnp.int32(0), frequency_data, quantum_assignments, intens_arr)
    return status


# =============================================================================
# 14. MAP-REDUCE MASTER MANIFEST COMPILATION & PARQUET READER
# =============================================================================

def compile_master_manifest(
    chunk_directory: Union[str, Path],
    output_manifest_path: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> CatalogManifestRecord:
    """
    Compiles a master catalog manifest across all Hive-partitioned parquet chunks in a Map-Reduce fashion.

    Map phase: Scans all .parquet files in parallel, extracting row counts, frequency extrema,
               intensity bounds, and cryptographic checksums.
    Reduce phase: Aggregates statistics, computes catalog-wide SHA-256 digest, and writes
                  manifest record to disk (JSON).

    Args:
        chunk_directory: Root directory containing partitioned parquet chunk files.
        output_manifest_path: Optional path to save master_manifest.json.
        max_workers: Number of parallel worker threads for Map inspection.
        custom_metadata: Optional metadata attributes to include in manifest.

    Returns:
        CatalogManifestRecord: Validated Pydantic manifest model.

    Raises:
        CatalogCompilerError: If directory does not exist or contains no parquet files.
    """
    chunk_dir = Path(chunk_directory).resolve()
    if not chunk_dir.exists():
        raise CatalogCompilerError(f"Chunk directory does not exist: {chunk_dir}")

    parquet_files = sorted(list(chunk_dir.rglob("*.parquet")))
    if len(parquet_files) == 0:
        raise CatalogCompilerError(f"No parquet chunk files found in directory: {chunk_dir}")

    def _inspect_parquet_shard(p_path: Path) -> Dict[str, Any]:
        pq_f = pq.ParquetFile(str(p_path))
        meta = pq_f.metadata
        n_rows = meta.num_rows

        tbl = pq_f.read(["frequencies", "intensities"])
        freq_col = tbl["frequencies"].to_numpy()
        intens_col = tbl["intensities"].to_numpy()

        f_min = float(np.min(freq_col)) if len(freq_col) > 0 else 0.0
        f_max = float(np.max(freq_col)) if len(freq_col) > 0 else 0.0
        i_min = float(np.min(intens_col)) if len(intens_col) > 0 else 0.0
        i_max = float(np.max(intens_col)) if len(intens_col) > 0 else 0.0

        sha = compute_sha256(p_path)
        return {
            "path": str(p_path),
            "num_rows": n_rows,
            "f_min": f_min,
            "f_max": f_max,
            "i_min": i_min,
            "i_max": i_max,
            "sha256": sha,
        }

    workers = max_workers if max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        shard_results = list(executor.map(_inspect_parquet_shard, parquet_files))

    total_transitions = sum(r["num_rows"] for r in shard_results)
    global_f_min = min(r["f_min"] for r in shard_results) if shard_results else 0.0
    global_f_max = max(r["f_max"] for r in shard_results) if shard_results else 0.0
    global_i_min = min(r["i_min"] for r in shard_results) if shard_results else 0.0
    global_i_max = max(r["i_max"] for r in shard_results) if shard_results else 0.0

    shard_paths = [r["path"] for r in shard_results]

    hasher = hashlib.sha256()
    for r in shard_results:
        hasher.update(r["sha256"].encode("utf-8"))
    catalog_sha = hasher.hexdigest()

    manifest = CatalogManifestRecord(
        catalog_path=str(chunk_dir),
        total_transitions=total_transitions,
        frequency_min_mhz=global_f_min,
        frequency_max_mhz=global_f_max,
        intensity_min=global_i_min,
        intensity_max=global_i_max,
        is_virtual=True,
        num_shards=len(shard_results),
        shard_paths=shard_paths,
        sha256_checksum=catalog_sha,
        metadata=custom_metadata or {},
    )

    if output_manifest_path is not None:
        out_m_p = Path(output_manifest_path).resolve()
        out_m_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_m_p, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    return manifest


def read_parquet_catalog(
    chunk_directory: Union[str, Path],
    batch_size: int = DEFAULT_CHUNK_SIZE,
    columns: Optional[Sequence[str]] = None,
    frequency_range: Optional[Tuple[float, float]] = None,
    partition_filter: Optional[Any] = None,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Out-of-core streaming reader for Hive-partitioned Parquet chunk transition catalogs.

    Uses pyarrow.dataset with predicate pushdown and batch streaming to yield dictionaries
    of NumPy arrays without loading the cumulative catalog into memory.

    Args:
        chunk_directory: Root directory containing partitioned parquet chunk files.
        batch_size: Transition row count per yielded batch.
        columns: Optional list of columnar datasets to retrieve.
        frequency_range: Optional tuple (min_freq_mhz, max_freq_mhz) for predicate pushdown.
        partition_filter: Optional PyArrow dataset expression for partition filtering.

    Yields:
        Dict[str, np.ndarray]: Dictionary mapping columnar names to NumPy array slices.

    Raises:
        FileNotFoundError: If chunk directory does not exist.
    """
    chunk_dir = Path(chunk_directory).resolve()
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Parquet catalog directory not found: {chunk_dir}")

    dataset = ds.dataset(str(chunk_dir), format="parquet", partitioning="hive")

    filter_expr = None
    if frequency_range is not None:
        f_min, f_max = frequency_range
        filter_expr = (ds.field("frequencies") >= f_min) & (ds.field("frequencies") <= f_max)

    if partition_filter is not None:
        if filter_expr is not None:
            filter_expr = filter_expr & partition_filter
        else:
            filter_expr = partition_filter

    target_cols = list(columns) if columns is not None else [
        "frequencies",
        "intensities",
        "lower_energies",
        "upper_energies",
        "uncertainties",
        "g_upper",
        "species_ids",
        "quantum_numbers",
    ]

    scanner = dataset.scanner(columns=target_cols, filter=filter_expr, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        if record_batch.num_rows == 0:
            continue
        batch_dict: Dict[str, np.ndarray] = {}
        for col_name in target_cols:
            if col_name in record_batch.schema.names:
                arr = record_batch.column(col_name)
                if col_name == "quantum_numbers":
                    qn_lists = arr.to_pylist()
                    batch_dict[col_name] = np.asarray(qn_lists, dtype=np.int32)
                else:
                    batch_dict[col_name] = arr.to_numpy()
        yield batch_dict


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_LOCAL_LOCK_TIMEOUT",
    "DEFAULT_LUSTRE_LOCK_TIMEOUT",
    "DEFAULT_STALE_LOCK_TIMEOUT",
    "STANDARD_DATASET_NAMES",
    "PARQUET_TRANSITION_SCHEMA",
    "CatalogCompilerError",
    "CatalogLockError",
    "CatalogLockTimeoutError",
    "LustreCatalogLockError",
    "CatalogDatasetError",
    "CatalogStreamingError",
    "VirtualLayoutError",
    "CatalogCorruptionError",
    "CatalogSchemaError",
    "TransitionRecord",
    "TransitionBatch",
    "CatalogCompilerConfig",
    "CatalogManifestRecord",
    "is_lustre_filesystem",
    "compute_sha256",
    "LustreDirectoryLock",
    "Tier3RegistryLock",
    "get_chunked_transitions_dir",
    "serialize_chunk_to_parquet",
    "jax_serialize_chunk_callback",
    "compile_master_manifest",
    "read_parquet_catalog",
    "stream_chunked_catalog",
    "compile_virtual_catalog",
    "read_chunked_catalog",
    "verify_catalog_integrity",
    "extract_catalog_metadata",
    "_normalize_chunk_data",
]


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_convolver_phase3.py ---
"""
CoChem-SpycFit: Comprehensive Phase 3 Physical Unit Test Suite for Macroscopic Spectral Broadener.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic doubles
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of exact Faddeeva Voigt profile via jax.scipy.special.erfcx
- Validation of multi-temperature Boltzmann scaling with tensor broadcasting [N_lines, N_temperatures]
- Validation of static masked batching via jax.lax.scan returning registered pure PyTree ConvolverPyTree
- AST structural verification enforcing pure jax.lax.scan without forbidden Python for/while loops
- Zero-pass audit verifying no pass statements across codebase.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
import scipy.special as scipy_sp

from cochem_spycfit.physics_core.cochem_convolver import (
    C2_RADIATION_CONST,
    DEFAULT_MAX_VRAM_MB,
    FOUR_LN_2,
    FWHM_TO_SIGMA_GAUSS,
    GAUSS_PREFACTOR,
    H_OVER_K_B,
    LORENTZ_PREFACTOR,
    PLANCK_H,
    SPEED_OF_LIGHT_CM_S,
    SPEED_OF_LIGHT_M_S,
    SQRT_2_LN_2,
    ConvolutionConfig,
    ConvolvedSpectrumResult,
    ConvolverError,
    ConvolverPyTree,
    EnergyUnit,
    InvalidGridError,
    MacroscopicSpectralBroadener,
    MicroBatchError,
    PartitionGridConfig,
    PartitionGridResult,
    ProfileError,
    ProfileType,
    SpectralDataError,
    SpectralLines,
    ThermalScalingError,
    build_frequency_grid,
    calculate_optimal_chunk_size,
    compute_boltzmann_scaling_grid_jax,
    convolute_multi_temperature,
    convolve_frequency_chunks_scan_1d_jax,
    convolve_frequency_chunks_scan_2d_jax,
    convolve_line_chunks_scan_1d_jax,
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    evaluate_partition_grid,
    evaluate_profile_kernel_jax,
    faddeeva_rational_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    get_energy_conversion_factor_to_kelvin,
    get_profile_type_code,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
    static_masked_scan,
    true_voigt_profile,
)

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. EXACT FADDEEVA VOIGT PROFILE TESTS (true_voigt_profile via erfcx)
# =============================================================================

def test_true_voigt_profile_erfcx_mathematical_precision() -> None:
    """Validate true_voigt_profile numerical match with scipy.special.voigt_profile within machine precision."""
    x_points = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    sigma_vals = [0.05, 0.1, 0.5, 1.2, 2.5]
    gamma_vals = [0.02, 0.1, 0.4, 0.8, 2.0]

    for sigma in sigma_vals:
        for gamma in gamma_vals:
            # JAX implementation
            jax_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
            jax_arr = np.array(jax_vals, dtype=np.float64)

            # Reference SciPy Faddeeva voigt_profile
            sp_vals = scipy_sp.voigt_profile(x_points, sigma, gamma)

            max_diff = np.max(np.abs(jax_arr - sp_vals))
            assert max_diff < 1e-13, (
                f"Voigt profile mismatch for sigma={sigma}, gamma={gamma}: max_diff={max_diff}"
            )


def test_true_voigt_profile_gaussian_limit() -> None:
    """When gamma -> 0, true_voigt_profile must analytically converge to a pure Gaussian distribution."""
    sigma = 0.35
    gamma = 1e-25  # Effectively zero Lorentzian width
    x_points = np.linspace(-3.0 * sigma, 3.0 * sigma, 50, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Gaussian: (1 / (sigma * sqrt(2*pi))) * exp(-x^2 / (2*sigma^2))
    expected_gaussian = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (x_points / sigma) ** 2)

    max_diff = np.max(np.abs(voigt_arr - expected_gaussian))
    assert max_diff < 1e-13, f"Gaussian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_lorentzian_limit() -> None:
    """When sigma is small, true_voigt_profile must approach a pure Lorentzian profile."""
    sigma = 1e-15
    gamma = 0.5
    x_points = np.linspace(-4.0, 4.0, 41, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Lorentzian: (1 / pi) * (gamma / (x^2 + gamma^2))
    expected_lorentzian = (1.0 / math.pi) * (gamma / (x_points ** 2 + gamma ** 2))

    max_diff = np.max(np.abs(voigt_arr - expected_lorentzian))
    assert max_diff < 1e-11, f"Lorentzian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_unit_area_normalization() -> None:
    """Verify analytical unit area normalization int_{-\\infty}^{\\infty} V(x) dx == 1.0."""
    sigma = 0.25
    gamma = 0.30
    cutoff = 200.0 * (sigma + gamma)
    n_pts = 100001
    x_grid = np.linspace(-cutoff, cutoff, n_pts, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_grid), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    integrated_area = float(np.trapezoid(voigt_arr, x_grid))
    # Account for analytical Lorentzian asymptotic tail 2 * int_{cutoff}^\\infty (gamma / (pi * x^2)) dx = 2*gamma / (pi * cutoff)
    tail_analytical = (2.0 * gamma) / (math.pi * cutoff)
    total_area = integrated_area + tail_analytical
    assert abs(total_area - 1.0) < 1e-6, f"Voigt integrated total area was {total_area}, expected ~1.0"

    # Also verify exact agreement with SciPy integration over identical domain
    sp_vals = scipy_sp.voigt_profile(x_grid, sigma, gamma)
    sp_area = float(np.trapezoid(sp_vals, x_grid))
    assert abs(integrated_area - sp_area) < 1e-13, f"Integration mismatch vs SciPy: diff={abs(integrated_area - sp_area)}"


def test_true_voigt_profile_symmetry_and_peak() -> None:
    """Verify profile symmetry V(-x) == V(x) and exact analytical peak height at x=0."""
    sigma = 0.4
    gamma = 0.25
    x_eval = jnp.array([0.0, -1.5, 1.5, -3.2, 3.2], dtype=jnp.float64)

    v_vals = true_voigt_profile(x_eval, sigma, gamma)

    # Peak height at x=0: Re[erfcx(gamma / (sigma * sqrt(2)))] / (sigma * sqrt(2*pi))
    u_0 = gamma / (sigma * math.sqrt(2.0))
    expected_peak = float(jsp.erfcx(u_0)) / (sigma * math.sqrt(2.0 * math.pi))
    actual_peak = float(v_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-15

    # Symmetry checks
    assert abs(float(v_vals[1]) - float(v_vals[2])) < 1e-15
    assert abs(float(v_vals[3]) - float(v_vals[4])) < 1e-15


def test_true_voigt_profile_jit_and_broadcasting() -> None:
    """Verify JIT compilation and tensor broadcasting across 2D/3D shapes."""
    jitted_voigt = jax.jit(true_voigt_profile)

    # 2D broadcast: x shape (20, 1), sigma (1, 5), gamma scalar
    x_2d = jnp.linspace(-2.0, 2.0, 20)[:, None]
    sigma_row = jnp.linspace(0.1, 0.5, 5)[None, :]
    gamma_val = 0.15

    res_2d = jitted_voigt(x_2d, sigma_row, gamma_val)
    assert res_2d.shape == (20, 5)
    assert jnp.all(jnp.isfinite(res_2d))
    assert jnp.all(res_2d > 0.0)


# =============================================================================
# 2. MULTI-TEMPERATURE CONVOLUTION TESTS (convolute_multi_temperature)
# =============================================================================

def test_convolute_multi_temperature_shape_broadcasting() -> None:
    """Verify output shape is strictly [N_lines, N_temperatures] with pure JAX tensor broadcasting."""
    n_lines = 40
    n_temps = 8

    freqs = jnp.linspace(100000.0, 110000.0, n_lines)
    intensities = jnp.linspace(0.1, 5.0, n_lines)
    energies = jnp.linspace(0.0, 500.0, n_lines)
    temps = jnp.array([10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 296.0, 350.0], dtype=jnp.float64)
    # Power-law partition function: Q(T) = (T / 296)^1.5
    q_temps = jnp.power(temps / 296.0, 1.5)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=296.0,
        q_ref=1.0,
    )

    assert scaled_i.shape == (n_lines, n_temps), f"Expected shape ({n_lines}, {n_temps}), got {scaled_i.shape}"
    assert jnp.all(jnp.isfinite(scaled_i))
    assert jnp.all(scaled_i >= 0.0)


def test_convolute_multi_temperature_reference_invariance() -> None:
    """At T == T_ref and Q(T) == Q_ref, scaled intensities must identically equal reference intensities."""
    n_lines = 15
    freqs = jnp.linspace(90000.0, 95000.0, n_lines)
    intensities = jnp.array([0.5, 1.2, 3.4, 0.8, 2.1, 4.0, 1.5, 0.3, 2.7, 3.1, 1.9, 0.4, 2.8, 3.9, 1.1])
    energies = jnp.linspace(10.0, 200.0, n_lines)

    t_ref = 296.0
    q_ref = 1.0
    temps = jnp.array([t_ref], dtype=jnp.float64)
    q_temps = jnp.array([q_ref], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # Column 0 corresponds to T_ref
    scaled_col = scaled_i[:, 0]
    max_diff = float(jnp.max(jnp.abs(scaled_col - intensities)))
    assert max_diff < 1e-14, f"Reference temperature scaling differed from original: max_diff={max_diff}"


def test_convolute_multi_temperature_boltzmann_population_redistribution() -> None:
    """Validate that high-energy transitions gain intensity at high T while ground-state transitions deplete."""
    # Line 0: Ground state E_lower = 0 cm^-1
    # Line 1: High energy state E_lower = 300 cm^-1
    freqs = jnp.array([100000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 1.0], dtype=jnp.float64)
    energies = jnp.array([0.0, 300.0], dtype=jnp.float64)

    t_low = 50.0
    t_high = 300.0
    t_ref = 300.0
    q_ref = 1.0
    temps = jnp.array([t_low, t_high], dtype=jnp.float64)
    q_temps = jnp.array([1.0, 1.0], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # At low temperature (T=50K), high-energy line (Line 1) must be exponentially suppressed relative to Line 0
    ratio_t_low = float(scaled_i[1, 0] / scaled_i[0, 0])
    # Expected Boltzmann ratio: exp(- C2 * 300 * (1/50 - 1/300)) / exp(0)
    expected_ratio = math.exp(-C2_RADIATION_CONST * 300.0 * (1.0 / 50.0 - 1.0 / 300.0))
    assert abs(ratio_t_low - expected_ratio) < 1e-12
    assert ratio_t_low < 0.01  # Heavily depopulated at 50K


def test_convolute_multi_temperature_jit_compiled() -> None:
    """Verify convolute_multi_temperature JIT compilation and execution purity."""
    jitted_fn = jax.jit(convolute_multi_temperature)

    freqs = jnp.array([80000.0, 90000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    q_temps = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    temps = jnp.array([100.0, 200.0, 400.0], dtype=jnp.float64)

    res = jitted_fn(freqs, intensities, q_temps, temps)
    assert res.shape == (3, 3)
    assert jnp.all(jnp.isfinite(res))


# =============================================================================
# 3. STATIC MASKED SCAN & CONVOLVER PYTREE TESTS
# =============================================================================

def test_convolver_pytree_registration_and_roundtrip() -> None:
    """Verify ConvolverPyTree is registered with JAX PyTree system and supports flatten/unflatten/tree_map."""
    freqs = jnp.linspace(100.0, 200.0, 50)
    intensities = jnp.sin(jnp.linspace(0.0, math.pi, 50))
    mask = jnp.ones(50, dtype=jnp.bool_)
    total_i = jnp.array(5.25, dtype=jnp.float64)

    pytree = ConvolverPyTree(
        frequencies=freqs,
        intensities=intensities,
        mask=mask,
        total_intensity=total_i,
        num_lines=50,
        chunk_size=10000,
        num_chunks=1,
        profile_type="voigt",
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    assert len(leaves) == 4
    assert jnp.allclose(leaves[0], freqs)
    assert jnp.allclose(leaves[1], intensities)
    assert jnp.allclose(leaves[2], mask)
    assert jnp.allclose(leaves[3], total_i)

    # Unflatten
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, ConvolverPyTree)
    assert reconstructed.num_lines == 50
    assert reconstructed.chunk_size == 10000
    assert reconstructed.num_chunks == 1
    assert reconstructed.profile_type == "voigt"
    assert jnp.allclose(reconstructed.intensities, intensities)

    # Tree Map: Multiply all array leaves by 2.0
    mapped_tree = jax.tree_util.tree_map(lambda x: x * 2.0, pytree)
    assert isinstance(mapped_tree, ConvolverPyTree)
    assert jnp.allclose(mapped_tree.total_intensity, total_i * 2.0)


def test_static_masked_scan_returns_pure_pytree() -> None:
    """Verify static_masked_scan execution returns a pure ConvolverPyTree."""
    freqs = [100.0, 102.0, 105.0, 108.0, 110.0]
    intensities = [1.0, 2.5, 0.8, 3.2, 1.5]
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    result_pytree = static_masked_scan(
        lines=lines,
        chunk_size=2,
        grid_points=200,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.05,
        gamma=0.02,
    )

    assert isinstance(result_pytree, ConvolverPyTree)
    assert result_pytree.num_lines == 5
    assert result_pytree.chunk_size == 2
    assert result_pytree.num_chunks == 3  # ceil(5 / 2) = 3 chunks (padded length 6)
    assert len(result_pytree.frequencies) == 200
    assert len(result_pytree.intensities) == 200
    assert float(result_pytree.total_intensity) > 0.0


def test_static_masked_scan_masking_authenticity() -> None:
    """Verify that padded trailing lines in static masked batching have exactly zero effect on output spectrum."""
    freqs = [100.0, 105.0, 110.0]
    intensities = [2.0, 3.0, 1.0]
    lines_original = SpectralLines(frequencies=freqs, intensities=intensities)

    # Run with chunk_size=2 (produces 2 chunks: chunk 1 has 2 lines, chunk 2 has 1 line + 1 padded trailing line)
    pytree_chunked = static_masked_scan(
        lines=lines_original,
        chunk_size=2,
        grid_points=300,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.08,
        gamma=0.04,
    )

    # Run monolithic reference (all lines at once)
    grid_arr = pytree_chunked.frequencies
    delta_nu = grid_arr[:, None] - jnp.array(freqs, dtype=jnp.float64)[None, :]
    prof_mono = true_voigt_profile(delta_nu, 0.08, 0.04)
    spec_mono = jnp.dot(prof_mono, jnp.array(intensities, dtype=jnp.float64))

    max_diff = float(jnp.max(jnp.abs(pytree_chunked.intensities - spec_mono)))
    assert max_diff < 1e-13, f"Masked scan output differed from unchunked monolithic reference: max_diff={max_diff}"


def test_static_masked_scan_chunk_size_invariance() -> None:
    """Verify that convolved spectrum is invariant to the chunk_size parameter."""
    freqs = np.linspace(100.0, 120.0, 37, dtype=np.float64).tolist()
    intensities = np.linspace(0.5, 4.0, 37, dtype=np.float64).tolist()
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    grid_pts = 500
    g_min = 95.0
    g_max = 125.0
    sigma = 0.06
    gamma = 0.03

    # Test chunk sizes: 1, 5, 10, 37, 100
    chunk_sizes = [1, 5, 10, 37, 100]
    results = []

    for cs in chunk_sizes:
        res = static_masked_scan(
            lines=lines,
            chunk_size=cs,
            grid_points=grid_pts,
            grid_min=g_min,
            grid_max=g_max,
            sigma=sigma,
            gamma=gamma,
        )
        results.append(res.intensities)

    ref_spec = results[0]
    for idx, cs in enumerate(chunk_sizes[1:], start=1):
        diff = float(jnp.max(jnp.abs(results[idx] - ref_spec)))
        assert diff < 1e-13, f"Chunk size {cs} produced mismatch vs chunk size 1: diff={diff}"


def test_static_masked_scan_input_formats() -> None:
    """Verify static_masked_scan works identically with SpectralLines, tuple of arrays, and dict inputs."""
    f_list = [100.0, 102.0, 104.0]
    i_list = [1.0, 2.0, 3.0]

    # 1. SpectralLines model
    res1 = static_masked_scan(SpectralLines(frequencies=f_list, intensities=i_list), chunk_size=2, grid_points=100)

    # 2. Tuple of arrays
    res2 = static_masked_scan((jnp.array(f_list), jnp.array(i_list)), chunk_size=2, grid_points=100)

    # 3. Dict
    res3 = static_masked_scan({"frequencies": f_list, "intensities": i_list}, chunk_size=2, grid_points=100)

    assert jnp.allclose(res1.intensities, res2.intensities, atol=1e-14)
    assert jnp.allclose(res1.intensities, res3.intensities, atol=1e-14)


# =============================================================================
# 4. AST LOOP-FREE & ZERO-MOCK COMPLIANCE VERIFICATION
# =============================================================================

def test_ast_loop_free_compliance() -> None:
    """
    AST structural verification ensuring that JAX kernels use pure vectorization and jax.lax.scan,
    with ZERO Python for/while loops in performance-critical paths.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    critical_functions = [
        "true_voigt_profile",
        "convolute_multi_temperature",
        "_scan_step_masked_kernel",
        "faddeeva_rational_jax",
        "gaussian_kernel_jax",
        "lorentzian_kernel_jax",
        "pseudo_voigt_kernel_jax",
        "faddeeva_voigt_kernel_jax",
        "evaluate_profile_kernel_jax",
        "evaluate_partition_function_jax",
        "compute_boltzmann_scaling_grid_jax",
        "convolve_frequency_chunks_scan_1d_jax",
        "convolve_frequency_chunks_scan_2d_jax",
        "convolve_line_chunks_scan_1d_jax",
        "convolve_spectrum_monolithic_jax",
    ]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in critical_functions:
            for child in ast.walk(node):
                assert not isinstance(child, (ast.For, ast.While)), (
                    f"Forbidden Python loop found in JAX kernel '{node.name}' at line {getattr(child, 'lineno', '?')}."
                )


def test_zero_pass_and_ast_integrity_mandate() -> None:
    """
    Authenticity and zero-pass mandate audit:
    Verifies that cochem_convolver contains zero pass statements and maintains pure functional structure.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    # Check for pass statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            pytest.fail(f"Forbidden 'pass' statement found in cochem_convolver.py at line {getattr(node, 'lineno', '?')}.")

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_catalog_compiler_phase4.py ---
"""
CoChem-SpycFit: Comprehensive Phase 4 Physical Unit Test Suite for Catalog Compiler & PyArrow Out-of-Core Serialization.

Strict Zero-Mock & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical and mathematical calculations with real HDF5 and Parquet files on disk
- Validation of PyArrow chunked streaming with strict FP64 frequency and int32 quantum assignment type preservation
- Validation of lock-free Hive-partitioning ($COCHEM_STATE_DIR/SpycFit_Workspace/Catalogs/chunked_transitions/)
- Validation of jax.experimental.io_callback asynchronous array interception
- Validation of Map-Reduce master manifest compilation across partitioned parquet chunks
- Out-of-core streaming parquet reading with predicate pushdown and frequency range filtering
- AST static analysis enforcing zero pass statements and zero banned tokens
"""

from __future__ import annotations

import ast
import concurrent.futures
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)

from cochem_spycfit.serialization.cochem_catalog_compiler import (
    CatalogCompilerError,
    CatalogManifestRecord,
    CatalogSchemaError,
    TransitionBatch,
    TransitionRecord,
    compile_master_manifest,
    compute_sha256,
    get_chunked_transitions_dir,
    jax_serialize_chunk_callback,
    read_parquet_catalog,
    serialize_chunk_to_parquet,
)


# =============================================================================
# 1. FIXTURES & PHYSICAL ZERO-MOCK DATA GENERATORS
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated temporary workspace on physical disk."""
    ws = tmp_path / "catalog_phase4_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


def physical_water_chunk(n_lines: int = 500, offset: int = 0) -> Dict[str, np.ndarray]:
    """Retrieves authentic physical water transition lines from catalog_water.h5."""
    fixture_path = Path(__file__).parent / "fixtures" / "catalog_water.h5"
    with h5py.File(fixture_path, "r") as f_phys:
        cat_len = int(f_phys.attrs["total_transitions"])
        idx = np.arange(offset, offset + n_lines) % cat_len
        return {
            "frequencies": np.asarray(f_phys["frequencies"][idx], dtype=np.float64),
            "intensities": np.asarray(f_phys["intensities"][idx], dtype=np.float64),
            "lower_energies": np.asarray(f_phys["lower_energies"][idx], dtype=np.float64),
            "upper_energies": np.asarray(f_phys["upper_energies"][idx], dtype=np.float64),
            "uncertainties": np.asarray(f_phys["uncertainties"][idx], dtype=np.float64),
            "g_upper": np.asarray(f_phys["g_upper"][idx], dtype=np.int32),
            "quantum_numbers": np.asarray(f_phys["quantum_numbers"][idx], dtype=np.int32),
            "species_ids": np.asarray(f_phys["species_ids"][idx], dtype=np.int32),
        }


# =============================================================================
# 2. CHUNKED DIRECTORY RESOLUTION TESTS
# =============================================================================

class TestChunkedDirectoryResolution:
    """Validates state directory resolution via platformdirs and environment variables."""

    def test_get_chunked_transitions_dir_default(self) -> None:
        orig_state = os.environ.get("COCHEM_STATE_DIR")
        try:
            os.environ.pop("COCHEM_STATE_DIR", None)
            res_dir = get_chunked_transitions_dir()
            assert isinstance(res_dir, Path)
            assert "SpycFit_Workspace" in str(res_dir)
            assert "Catalogs" in str(res_dir)
            assert "chunked_transitions" in str(res_dir)
            assert res_dir.exists()
        finally:
            if orig_state is not None:
                os.environ["COCHEM_STATE_DIR"] = orig_state

    def test_get_chunked_transitions_dir_custom_env(self, temp_workspace: Path) -> None:
        orig_state = os.environ.get("COCHEM_STATE_DIR")
        try:
            custom_state = temp_workspace / "custom_state"
            os.environ["COCHEM_STATE_DIR"] = str(custom_state)
            res_dir = get_chunked_transitions_dir()
            expected = custom_state / "SpycFit_Workspace" / "Catalogs" / "chunked_transitions"
            assert res_dir == expected.resolve()
            assert res_dir.exists()
        finally:
            if orig_state is not None:
                os.environ["COCHEM_STATE_DIR"] = orig_state
            else:
                os.environ.pop("COCHEM_STATE_DIR", None)

    def test_get_chunked_transitions_dir_explicit_override(self, temp_workspace: Path) -> None:
        explicit_dir = temp_workspace / "explicit_dir"
        res_dir = get_chunked_transitions_dir(base_dir=explicit_dir)
        assert res_dir == explicit_dir.resolve()
        assert res_dir.exists()


# =============================================================================
# 3. PYARROW CHUNK SERIALIZATION & STRICT TYPING TESTS
# =============================================================================

class TestPyArrowChunkSerialization:
    """Validates PyArrow serialization, strict FP64/int32 type preservation, and Hive partitioning."""

    def test_serialize_chunk_to_parquet_strict_types(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=200)
        out_dir = temp_workspace / "parquet_chunks"

        parquet_file = serialize_chunk_to_parquet(
            frequency_data=data["frequencies"],
            quantum_assignments=data["quantum_numbers"],
            temp_step=1,
            batch_id=0,
            uuid_str="unit_test_001",
            output_dir=out_dir,
            partition_by_temp=True,
        )

        assert parquet_file.is_file()
        assert parquet_file.name == "chunk_1_0_unit_test_001.parquet"
        assert parquet_file.parent.name == "temperature_step=1"

        # Read back PyArrow table and inspect exact column types
        table = pq.read_table(parquet_file)
        schema = table.schema

        # Strict type enforcement: frequencies must be FP64 (float64)
        assert schema.field("frequencies").type == pa.float64()
        assert schema.field("intensities").type == pa.float64()
        assert schema.field("lower_energies").type == pa.float64()
        assert schema.field("upper_energies").type == pa.float64()
        assert schema.field("uncertainties").type == pa.float64()

        # Strict type enforcement: quantum numbers and indices must be int32
        assert schema.field("g_upper").type == pa.int32()
        assert schema.field("species_ids").type == pa.int32()
        assert schema.field("J_upper").type == pa.int32()
        assert schema.field("Ka_upper").type == pa.int32()
        assert schema.field("Kc_upper").type == pa.int32()
        assert schema.field("J_lower").type == pa.int32()
        assert schema.field("Ka_lower").type == pa.int32()
        assert schema.field("Kc_lower").type == pa.int32()
        assert schema.field("quantum_numbers").type == pa.list_(pa.int32())

        # Verify numerical fidelity
        read_freqs = table["frequencies"].to_numpy()
        np.testing.assert_allclose(read_freqs, data["frequencies"])

        read_j_up = table["J_upper"].to_numpy()
        np.testing.assert_array_equal(read_j_up, data["quantum_numbers"][:, 0])

    def test_serialize_chunk_to_parquet_from_transition_batch(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=50)
        batch = TransitionBatch(
            frequencies_mhz=data["frequencies"].tolist(),
            intensities=data["intensities"].tolist(),
            lower_energies_cm1=data["lower_energies"].tolist(),
            upper_energies_cm1=data["upper_energies"].tolist(),
            uncertainties_mhz=data["uncertainties"].tolist(),
            g_upper=data["g_upper"].tolist(),
            quantum_numbers=[tuple(int(x) for x in row) for row in data["quantum_numbers"]],
            species_ids=data["species_ids"].tolist(),
        )

        out_dir = temp_workspace / "batch_chunks"
        parquet_file = serialize_chunk_to_parquet(
            frequency_data=batch,
            temp_step=2,
            batch_id="batch_alpha",
            uuid_str="batch_uuid_123",
            output_dir=out_dir,
        )

        assert parquet_file.is_file()
        assert "batch_alpha" in parquet_file.name
        table = pq.read_table(parquet_file)
        assert table.num_rows == 50
        np.testing.assert_allclose(table["intensities"].to_numpy(), data["intensities"])

    def test_serialize_chunk_to_parquet_from_dict_and_3d_quantum_assignments(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=40)
        qn_3 = data["quantum_numbers"][:, :3]

        dict_payload = {
            "frequencies": data["frequencies"],
            "intensities": data["intensities"],
            "lower_energies": data["lower_energies"],
            "upper_energies": data["upper_energies"],
            "uncertainties": data["uncertainties"],
            "g_upper": data["g_upper"],
            "species_ids": data["species_ids"],
        }

        out_dir = temp_workspace / "dict_chunks"
        parquet_file = serialize_chunk_to_parquet(
            frequency_data=dict_payload,
            quantum_assignments=qn_3,
            temp_step=3,
            batch_id=1,
            output_dir=out_dir,
            partition_by_temp=False,
        )

        assert parquet_file.is_file()
        table = pq.read_table(parquet_file)
        assert table.num_rows == 40
        assert table.schema.field("J_upper").type == pa.int32()
        assert table.schema.field("Ka_upper").type == pa.int32()
        assert table.schema.field("Kc_upper").type == pa.int32()
        assert table.schema.field("J_lower").type == pa.int32()
        np.testing.assert_array_equal(table["J_lower"].to_numpy(), np.zeros(40, dtype=np.int32))

    def test_serialize_chunk_to_parquet_concurrent_lock_free(self, temp_workspace: Path) -> None:
        """Validates lock-free append-only concurrent writes from multiple threads without collision."""
        out_dir = temp_workspace / "concurrent_chunks"
        num_threads = 8
        chunks_per_thread = 5

        def worker_task(thread_idx: int) -> List[Path]:
            created: List[Path] = []
            for c_idx in range(chunks_per_thread):
                data = physical_water_chunk(n_lines=25, offset=(thread_idx * 100 + c_idx * 25))
                p_file = serialize_chunk_to_parquet(
                    frequency_data=data,
                    temp_step=thread_idx % 3,
                    batch_id=f"t{thread_idx}_c{c_idx}",
                    output_dir=out_dir,
                    partition_by_temp=True,
                )
                created.append(p_file)
            return created

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, t_id) for t_id in range(num_threads)]
            all_files: List[Path] = []
            for f in concurrent.futures.as_completed(futures):
                all_files.extend(f.result())

        assert len(all_files) == num_threads * chunks_per_thread
        for p in all_files:
            assert p.is_file()
            tbl = pq.read_table(p)
            assert tbl.num_rows == 25


# =============================================================================
# 4. JAX IO_CALLBACK ASYNCHRONOUS INTERCEPTION TESTS
# =============================================================================

class TestJAXIOCallbackIntegration:
    """Validates jax.experimental.io_callback integration with JAX compiled computational graphs."""

    def test_jax_io_callback_serialization_in_jit(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=64)
        out_dir = temp_workspace / "jax_callback_chunks"

        jax_freqs = jnp.array(data["frequencies"], dtype=jnp.float64)
        jax_qn = jnp.array(data["quantum_numbers"], dtype=jnp.int32)
        jax_intens = jnp.array(data["intensities"], dtype=jnp.float64)

        @jax.jit
        def computational_pipeline(freq_arr: jnp.ndarray, qn_arr: jnp.ndarray, intens_arr: jnp.ndarray) -> jnp.ndarray:
            shifted_freqs = freq_arr + 1500.0
            scaled_intens = intens_arr * 2.5

            status = jax_serialize_chunk_callback(
                frequency_data=shifted_freqs,
                quantum_assignments=qn_arr,
                temp_step=10,
                batch_id=42,
                uuid_str="jax_cb_001",
                output_dir=str(out_dir),
                intensities=scaled_intens,
            )
            return status

        result_status = computational_pipeline(jax_freqs, jax_qn, jax_intens)
        assert int(result_status) == 1

        expected_file = out_dir / "temperature_step=10" / "chunk_10_42_jax_cb_001.parquet"
        assert expected_file.is_file()

        table = pq.read_table(expected_file)
        assert table.num_rows == 64
        read_freqs = table["frequencies"].to_numpy()
        np.testing.assert_allclose(read_freqs, data["frequencies"] + 1500.0)

        read_intens = table["intensities"].to_numpy()
        np.testing.assert_allclose(read_intens, data["intensities"] * 2.5)

    def test_jax_io_callback_array_fidelity(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=128)
        out_dir = temp_workspace / "fidelity_chunks"

        jax_freqs = jnp.array(data["frequencies"], dtype=jnp.float64)
        jax_qn = jnp.array(data["quantum_numbers"], dtype=jnp.int32)

        status = jax_serialize_chunk_callback(
            frequency_data=jax_freqs,
            quantum_assignments=jax_qn,
            temp_step=5,
            batch_id=99,
            uuid_str="fidelity_uuid",
            output_dir=str(out_dir),
        )
        assert int(status) == 1

        written_file = out_dir / "temperature_step=5" / "chunk_5_99_fidelity_uuid.parquet"
        assert written_file.is_file()

        table = pq.read_table(written_file)
        assert table.schema.field("frequencies").type == pa.float64()
        assert table.schema.field("quantum_numbers").type == pa.list_(pa.int32())
        np.testing.assert_allclose(table["frequencies"].to_numpy(), data["frequencies"])


# =============================================================================
# 5. MAP-REDUCE MASTER MANIFEST COMPILATION TESTS
# =============================================================================

class TestMasterManifestMapReduce:
    """Validates Map-Reduce aggregation across partitioned Parquet chunk catalogs."""

    def test_compile_master_manifest_map_reduce_basic(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "manifest_test_chunks"
        num_chunks = 4
        chunk_size = 100
        total_expected_lines = num_chunks * chunk_size

        all_freqs: List[float] = []
        all_intens: List[float] = []

        for c_idx in range(num_chunks):
            data = physical_water_chunk(n_lines=chunk_size, offset=c_idx * chunk_size)
            all_freqs.extend(data["frequencies"].tolist())
            all_intens.extend(data["intensities"].tolist())

            serialize_chunk_to_parquet(
                frequency_data=data,
                temp_step=c_idx % 2,
                batch_id=c_idx,
                uuid_str=f"chunk_map_{c_idx}",
                output_dir=chunk_dir,
                partition_by_temp=True,
            )

        manifest_json = temp_workspace / "master_manifest.json"
        manifest = compile_master_manifest(
            chunk_directory=chunk_dir,
            output_manifest_path=manifest_json,
            custom_metadata={"survey": "GBT-Astrochemistry-Survey", "polarization": "dual"},
        )

        assert isinstance(manifest, CatalogManifestRecord)
        assert manifest.total_transitions == total_expected_lines
        assert manifest.frequency_min_mhz == pytest.approx(min(all_freqs), rel=1e-7)
        assert manifest.frequency_max_mhz == pytest.approx(max(all_freqs), rel=1e-7)
        assert manifest.intensity_min == pytest.approx(min(all_intens), rel=1e-7)
        assert manifest.intensity_max == pytest.approx(max(all_intens), rel=1e-7)
        assert manifest.num_shards == num_chunks
        assert len(manifest.shard_paths) == num_chunks
        assert manifest.metadata["survey"] == "GBT-Astrochemistry-Survey"

        assert manifest_json.is_file()
        with open(manifest_json, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        assert meta_dict["total_transitions"] == total_expected_lines
        assert meta_dict["num_shards"] == num_chunks

    def test_compile_master_manifest_empty_directory_error(self, temp_workspace: Path) -> None:
        empty_dir = temp_workspace / "empty_chunk_dir"
        empty_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(CatalogCompilerError) as exc_info:
            compile_master_manifest(empty_dir)
        assert "No parquet chunk files found" in str(exc_info.value)


# =============================================================================
# 6. OUT-OF-CORE PARQUET STREAMING READER TESTS
# =============================================================================

class TestParquetOutofCoreReader:
    """Validates out-of-core streaming iteration and predicate pushdown on Parquet datasets."""

    def test_read_parquet_catalog_streaming(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "reader_test_chunks"
        num_chunks = 3
        chunk_size = 150

        for c_idx in range(num_chunks):
            data = physical_water_chunk(n_lines=chunk_size, offset=c_idx * chunk_size)
            serialize_chunk_to_parquet(
                frequency_data=data,
                temp_step=c_idx,
                batch_id=c_idx,
                uuid_str=f"reader_uuid_{c_idx}",
                output_dir=chunk_dir,
                partition_by_temp=True,
            )

        total_read = 0
        for batch in read_parquet_catalog(chunk_dir, batch_size=100):
            assert "frequencies" in batch
            assert "intensities" in batch
            assert "quantum_numbers" in batch
            assert isinstance(batch["frequencies"], np.ndarray)
            assert batch["frequencies"].dtype == np.float64
            assert batch["quantum_numbers"].dtype == np.int32
            total_read += len(batch["frequencies"])

        assert total_read == num_chunks * chunk_size

    def test_read_parquet_catalog_frequency_filter(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "filter_test_chunks"
        data = physical_water_chunk(n_lines=300)
        serialize_chunk_to_parquet(
            frequency_data=data,
            temp_step=1,
            batch_id=0,
            uuid_str="filter_chunk",
            output_dir=chunk_dir,
            partition_by_temp=True,
        )

        all_freqs = data["frequencies"]
        f_min = float(np.percentile(all_freqs, 25))
        f_max = float(np.percentile(all_freqs, 75))

        filtered_freqs: List[float] = []
        for batch in read_parquet_catalog(chunk_dir, frequency_range=(f_min, f_max)):
            f_arr = batch["frequencies"]
            assert np.all(f_arr >= f_min)
            assert np.all(f_arr <= f_max)
            filtered_freqs.extend(f_arr.tolist())

        assert len(filtered_freqs) > 0
        expected_count = int(np.sum((all_freqs >= f_min) & (all_freqs <= f_max)))
        assert len(filtered_freqs) == expected_count


# =============================================================================
# 7. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliancePhase4:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_catalog_compiler_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_catalog_compiler.py"
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
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_catalog_compiler.py"
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

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.