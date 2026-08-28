Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\01_P1_T02_hdf5_swmr_tensor_cache.md.
Original prompt:
# Prompt 02 — Phase 1, Task Group 02: Thread-Safe SWMR HDF5 Tensor Cache

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 1 — HDF5 DAG State Manager & Persistence
> **Phase:** 1 (Tasks 1–20) — State Management & RAM Protection
> **Dependency:** Prompt 01 (`cochem_spycfit_state.py` must exist — imports `FitState`)

---

## Objective

Implement the out-of-core tensor caching layer that persists heavy matrices (JAX covariance matrices, Hamiltonian eigenvectors, Gaussian Process weights) to disk via HDF5, while keeping lightweight metadata in RAM for <50 ms UI responsiveness. This is the memory protection backbone that prevents Jupyter kernel OOM crashes when handling dense vdW spectra.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\core_engine\cochem_spycfit_hdf5_cache.py
```

> **Scope:** This prompt targets ONLY the file above. The `__init__.py` export update is handled in Prompt 01b.

---

## Detailed Implementation Instructions

### 1. `TensorCache` — Thread-Safe SWMR HDF5 Manager

```python
import contextlib
import threading
import os
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
from filelock import FileLock

class TensorCache:
    """Thread-safe, cross-process SWMR HDF5 tensor cache for FitState heavy data."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        lustre_bypass: bool = False,
    ):
        """
        Parameters
        ----------
        cache_path : Path, optional
            Absolute path to the HDF5 cache file. Defaults to
            Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        lustre_bypass : bool
            If True, disables advisory file locks for Lustre/NFS parallel
            filesystems where POSIX locks are unreliable.
        """
        self._cache_path: Path = self._resolve_cache_path(cache_path)
        self._lustre_bypass: bool = lustre_bypass
        self._thread_lock: threading.Lock = threading.Lock()
        lock_path = self._cache_path.with_suffix(".lock")
        self._file_lock: FileLock = FileLock(str(lock_path), timeout=30)
```

### 2. Concurrency Guard Pattern (MANDATORY)

Every HDF5 read/write operation must use the following triple-guard pattern:

```python
def _write_dataset(self, state_id: str, key: str, data: np.ndarray) -> None:
    """Write a tensor dataset to the HDF5 cache under a state_id group."""
    # Guard 1: In-process thread lock
    with self._thread_lock:
        # Guard 2: Cross-process file lock (skip on Lustre/NFS)
        lock_ctx = self._file_lock if not self._lustre_bypass else contextlib.nullcontext()
        with lock_ctx:
            # Guard 3: SWMR-mode HDF5 write
            with h5py.File(self._cache_path, "a", libver="latest") as f:
                f.swmr_mode = True
                group = f.require_group(state_id)
                if key in group:
                    del group[key]
                group.create_dataset(
                    key,
                    data=data,
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                )
```

### 3. Core Methods to Implement

| Method | Signature | Purpose |
|:---|:---|:---|
| `store_state_tensors` | `(state_id: str, covariance: np.ndarray, hamiltonian: Optional[np.ndarray], gp_weights: Optional[np.ndarray]) -> None` | Chunk and write heavy tensors to HDF5, indexed by `state_id` |
| `restore_state_tensors` | `(state_id: str) -> dict[str, np.ndarray]` | Retrieve all tensors for a given `state_id` from HDF5 into RAM. Target: <100 ms for typical asymmetric-top datasets |
| `delete_state_tensors` | `(state_id: str) -> None` | Remove a state's tensor group from the cache (used during DAG branch purging on "Save & Commit") |
| `list_cached_states` | `() -> list[str]` | Return all `state_id` keys currently in the HDF5 file |
| `get_cache_size_bytes` | `() -> int` | Return the file size for memory budgeting |

### 4. Path Resolution Logic

```python
def _resolve_cache_path(self, user_path: Optional[Path]) -> Path:
    """Resolve the HDF5 cache file path using the Tier 2 hierarchy."""
    if user_path is not None:
        return Path(user_path)

    artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
    if artifacts_root:
        return Path(artifacts_root) / "spycfit_tensor_cache.h5"

    # Try loading from cochem_system_config.json
    config_path = Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
    if config_path.exists():
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        if "tensor_cache_path" in config:
            return Path(config["tensor_cache_path"])

    # Default fallback
    default_path = Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path
```

### 5. Integration Point with CommitManager

When `CommitManager.commit()` is called (from Prompt 01), it should invoke `TensorCache.store_state_tensors()` to offload heavy matrices. The `FitState.ml_weights_pointer` field stores the HDF5 dataset key (e.g., `"{state_id}/gp_weights"`).

---

## Validation Test Specification

1. Create a `TensorCache` instance pointing to a temporary HDF5 file inside a `tempfile.TemporaryDirectory()` (e.g., `Path(tmpdir) / "test_cache.h5"`)
2. Generate 50 sequential state entries, each storing:
   - A covariance matrix of authentic dimensions for a 6-parameter asymmetric-top fit (6×6 symmetric positive-definite matrix with physically realistic diagonal variances derived from water rotational constant uncertainties)
   - A GP weights vector of length 20 (representing 20 locked assignments with authentic O−C residual magnitudes)
3. Call `restore_state_tensors(state_id_25)` and assert the retrieved covariance matrix matches the stored matrix to float64 precision (`np.allclose`)
4. Verify `list_cached_states()` returns exactly 50 entries
5. Call `delete_state_tensors(state_id_50)` and verify it's removed
6. Measure that `restore_state_tensors` completes in <100 ms (use `time.perf_counter()`)

> **ZERO-MOCK MANDATE:** Covariance matrices must have physically realistic eigenvalue structure. Use water (H₂O) rotational constant uncertainties as the diagonal seed.

---

## Acceptance Criteria

- [ ] `TensorCache` implements the triple-guard concurrency pattern (thread lock + file lock + SWMR)
- [ ] `lustre_bypass` flag disables advisory locks for HPC environments
- [ ] All path resolution honors `COCHEM_ARTIFACTS_ROOT` and `cochem_system_config.json`
- [ ] `restore_state_tensors` retrieves data in <100 ms for typical datasets
- [ ] HDF5 datasets use chunking and gzip compression
- [ ] Validation test uses physically meaningful covariance matrices

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\core_engine\cochem_spycfit_hdf5_cache.py ---
"""
Thread-safe SWMR HDF5 TensorCache for out-of-core spectroscopic matrix persistence.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, List, Optional, Union

from filelock import FileLock
import h5py
import numpy as np

_logger = logging.getLogger("CoChem-SpycFit.TensorCache")


class TensorCache:
    """
    Thread-safe, cross-process SWMR HDF5 tensor cache for FitState heavy data.

    Persists heavy float64 covariance matrices, Hamiltonian eigenvectors,
    and Gaussian Process weight vectors indexed by FitState state_id hashes.
    """

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        lustre_bypass: bool = False,
    ) -> None:
        """
        Initialize the TensorCache manager.

        Parameters
        ----------
        cache_path : str or Path, optional
            Absolute path to the HDF5 cache file. Defaults to
            Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        lustre_bypass : bool, default False
            If True, disables advisory file locks for Lustre/NFS parallel
            filesystems where POSIX locks are unreliable.
        """
        self._cache_path: Path = self._resolve_cache_path(cache_path)
        self._lustre_bypass: bool = bool(lustre_bypass)
        self._thread_lock: threading.RLock = threading.RLock()
        lock_path = self._cache_path.with_suffix(".lock")
        self._file_lock: FileLock = FileLock(str(lock_path), timeout=30)

    @property
    def cache_path(self) -> Path:
        """Return the resolved filesystem Path to the HDF5 cache file."""
        return self._cache_path

    @property
    def lustre_bypass(self) -> bool:
        """Return True if Lustre advisory lock bypass is active."""
        return self._lustre_bypass

    def store_state_tensors(
        self,
        state_id: str,
        covariance: Optional[np.ndarray] = None,
        hamiltonian: Optional[np.ndarray] = None,
        gp_weights: Optional[np.ndarray] = None,
        **extra_tensors: np.ndarray,
    ) -> None:
        """
        Chunk and write heavy spectroscopic tensors to HDF5, indexed by state_id.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit.
        covariance : np.ndarray, optional
            Fitted parameter covariance matrix (typically 6x6 float64).
        hamiltonian : np.ndarray, optional
            Asymmetric rotor Hamiltonian matrix or block.
        gp_weights : np.ndarray, optional
            Gaussian Process regression weight vector (length-20 float64).
        **extra_tensors : np.ndarray
            Additional named tensor arrays to persist under the state group.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Guard 3: SWMR-mode HDF5 write
                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    group = f.require_group(state_id)

                    tensors_to_write: Dict[str, np.ndarray] = {}
                    if covariance is not None:
                        tensors_to_write["covariance"] = np.asarray(covariance, dtype=np.float64)
                    if hamiltonian is not None:
                        tensors_to_write["hamiltonian"] = np.asarray(hamiltonian, dtype=np.float64)
                    if gp_weights is not None:
                        tensors_to_write["gp_weights"] = np.asarray(gp_weights, dtype=np.float64)

                    for extra_key, extra_val in extra_tensors.items():
                        if extra_val is not None:
                            tensors_to_write[extra_key] = np.asarray(extra_val, dtype=np.float64)

                    for key, data in tensors_to_write.items():
                        if key in group:
                            del group[key]
                        group.create_dataset(
                            key,
                            data=data,
                            chunks=True,
                            compression="gzip",
                            compression_opts=4,
                        )
                    f.flush()

    def restore_state_tensors(self, state_id: str) -> Dict[str, np.ndarray]:
        """
        Retrieve all tensors for a given state_id from HDF5 into RAM in <100 ms.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping tensor keys to high-precision float64 numpy arrays.

        Raises
        ------
        KeyError
            If the cache file does not exist or the state_id is not found in cache.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    raise KeyError(
                        f"Tensor cache file '{self._cache_path}' does not exist."
                    )

                # Guard 3: SWMR-mode HDF5 read
                try:
                    f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
                except (ValueError, OSError, RuntimeError):
                    f = h5py.File(str(self._cache_path), "r", libver="latest")

                with f:
                    if state_id not in f:
                        raise KeyError(
                            f"State ID '{state_id}' not found in tensor cache."
                        )

                    group = f[state_id]
                    restored: Dict[str, np.ndarray] = {}
                    for key in group.keys():
                        dataset = group[key]
                        restored[key] = np.asarray(dataset, dtype=np.float64)

                    return restored

    def delete_state_tensors(self, state_id: str) -> None:
        """
        Remove a state's tensor group from the cache (used during DAG branch purging).

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit to remove.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    return

                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    if state_id in f:
                        del f[state_id]
                        f.flush()

    def list_cached_states(self) -> List[str]:
        """
        Return all state_id keys currently in the HDF5 file.

        Returns
        -------
        list[str]
            List of state_id string keys.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    return []

                try:
                    f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
                except (ValueError, OSError, RuntimeError):
                    f = h5py.File(str(self._cache_path), "r", libver="latest")

                with f:
                    return list(f.keys())

    def _write_dataset(self, state_id: str, key: str, data: np.ndarray) -> None:
        """
        Write a tensor dataset to the HDF5 cache under a state_id group.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash.
        key : str
            Dataset name key.
        data : np.ndarray
            Numpy array to persist.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Guard 3: SWMR-mode HDF5 write
                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    group = f.require_group(state_id)
                    if key in group:
                        del group[key]
                    group.create_dataset(
                        key,
                        data=np.asarray(data, dtype=np.float64),
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                    )
                    f.flush()

    def get_cache_size_bytes(self) -> int:
        """
        Return the on-disk size of the HDF5 cache file in bytes.

        Returns
        -------
        int
            Size in bytes, or 0 if the cache file does not exist.
        """
        with self._thread_lock:
            if self._cache_path.exists() and self._cache_path.is_file():
                return self._cache_path.stat().st_size
            return 0

    @staticmethod
    def _resolve_cache_path(user_path: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the HDF5 cache file path using the Tier 2 hierarchy:
        1. Explicit user_path argument.
        2. COCHEM_ARTIFACTS_ROOT environment variable.
        3. cochem_system_config.json configuration file.
        4. Default fallback: Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".

        Parameters
        ----------
        user_path : str or Path, optional
            User specified path.

        Returns
        -------
        Path
            Fully resolved absolute Path.
        """
        if user_path is not None:
            return Path(user_path).resolve()

        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return (Path(artifacts_root) / "spycfit_tensor_cache.h5").resolve()

        default_dir = Path.home() / "CoChem_Artifacts"
        config_path = default_dir / "Registry" / "cochem_system_config.json"
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if isinstance(config, dict) and "tensor_cache_path" in config:
                    return Path(config["tensor_cache_path"]).resolve()
            except Exception as _err:
                _logger.debug("Failed reading system config: %s", _err)

        default_dir.mkdir(parents=True, exist_ok=True)
        return (default_dir / "spycfit_tensor_cache.h5").resolve()

    def __len__(self) -> int:
        """Return the number of cached states in the HDF5 container."""
        return len(self.list_cached_states())

    def __contains__(self, state_id: str) -> bool:
        """Check if a state_id exists in the cache container."""
        with self._thread_lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                return False
            try:
                f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
            except (ValueError, OSError, RuntimeError):
                f = h5py.File(str(self._cache_path), "r", libver="latest")
            with f:
                return state_id in f


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_spycfit_hdf5_cache.py ---
"""
Thread-safe SWMR HDF5 TensorCache for out-of-core spectroscopic matrix persistence.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, List, Optional, Union

from filelock import FileLock
import h5py
import numpy as np

_logger = logging.getLogger("CoChem-SpycFit.TensorCache")


class TensorCache:
    """
    Thread-safe, cross-process SWMR HDF5 tensor cache for FitState heavy data.

    Persists heavy float64 covariance matrices, Hamiltonian eigenvectors,
    and Gaussian Process weight vectors indexed by FitState state_id hashes.
    """

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        lustre_bypass: bool = False,
    ) -> None:
        """
        Initialize the TensorCache manager.

        Parameters
        ----------
        cache_path : str or Path, optional
            Absolute path to the HDF5 cache file. Defaults to
            Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        lustre_bypass : bool, default False
            If True, disables advisory file locks for Lustre/NFS parallel
            filesystems where POSIX locks are unreliable.
        """
        self._cache_path: Path = self._resolve_cache_path(cache_path)
        self._lustre_bypass: bool = bool(lustre_bypass)
        self._thread_lock: threading.RLock = threading.RLock()
        lock_path = self._cache_path.with_suffix(".lock")
        self._file_lock: FileLock = FileLock(str(lock_path), timeout=30)

    @property
    def cache_path(self) -> Path:
        """Return the resolved filesystem Path to the HDF5 cache file."""
        return self._cache_path

    @property
    def lustre_bypass(self) -> bool:
        """Return True if Lustre advisory lock bypass is active."""
        return self._lustre_bypass

    def store_state_tensors(
        self,
        state_id: str,
        covariance: Optional[np.ndarray] = None,
        hamiltonian: Optional[np.ndarray] = None,
        gp_weights: Optional[np.ndarray] = None,
        **extra_tensors: np.ndarray,
    ) -> None:
        """
        Chunk and write heavy spectroscopic tensors to HDF5, indexed by state_id.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit.
        covariance : np.ndarray, optional
            Fitted parameter covariance matrix (typically 6x6 float64).
        hamiltonian : np.ndarray, optional
            Asymmetric rotor Hamiltonian matrix or block.
        gp_weights : np.ndarray, optional
            Gaussian Process regression weight vector (length-20 float64).
        **extra_tensors : np.ndarray
            Additional named tensor arrays to persist under the state group.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Guard 3: SWMR-mode HDF5 write
                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    group = f.require_group(state_id)

                    tensors_to_write: Dict[str, np.ndarray] = {}
                    if covariance is not None:
                        tensors_to_write["covariance"] = np.asarray(covariance, dtype=np.float64)
                    if hamiltonian is not None:
                        tensors_to_write["hamiltonian"] = np.asarray(hamiltonian, dtype=np.float64)
                    if gp_weights is not None:
                        tensors_to_write["gp_weights"] = np.asarray(gp_weights, dtype=np.float64)

                    for extra_key, extra_val in extra_tensors.items():
                        if extra_val is not None:
                            tensors_to_write[extra_key] = np.asarray(extra_val, dtype=np.float64)

                    for key, data in tensors_to_write.items():
                        if key in group:
                            del group[key]
                        group.create_dataset(
                            key,
                            data=data,
                            chunks=True,
                            compression="gzip",
                            compression_opts=4,
                        )
                    f.flush()

    def restore_state_tensors(self, state_id: str) -> Dict[str, np.ndarray]:
        """
        Retrieve all tensors for a given state_id from HDF5 into RAM in <100 ms.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping tensor keys to high-precision float64 numpy arrays.

        Raises
        ------
        KeyError
            If the cache file does not exist or the state_id is not found in cache.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    raise KeyError(
                        f"Tensor cache file '{self._cache_path}' does not exist."
                    )

                # Guard 3: SWMR-mode HDF5 read
                try:
                    f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
                except (ValueError, OSError, RuntimeError):
                    f = h5py.File(str(self._cache_path), "r", libver="latest")

                with f:
                    if state_id not in f:
                        raise KeyError(
                            f"State ID '{state_id}' not found in tensor cache."
                        )

                    group = f[state_id]
                    restored: Dict[str, np.ndarray] = {}
                    for key in group.keys():
                        dataset = group[key]
                        restored[key] = np.asarray(dataset, dtype=np.float64)

                    return restored

    def delete_state_tensors(self, state_id: str) -> None:
        """
        Remove a state's tensor group from the cache (used during DAG branch purging).

        Parameters
        ----------
        state_id : str
            Unique state identifier hash for the FitState commit to remove.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    return

                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    if state_id in f:
                        del f[state_id]
                        f.flush()

    def list_cached_states(self) -> List[str]:
        """
        Return all state_id keys currently in the HDF5 file.

        Returns
        -------
        list[str]
            List of state_id string keys.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                if not self._cache_path.exists() or not self._cache_path.is_file():
                    return []

                try:
                    f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
                except (ValueError, OSError, RuntimeError):
                    f = h5py.File(str(self._cache_path), "r", libver="latest")

                with f:
                    return list(f.keys())

    def _write_dataset(self, state_id: str, key: str, data: np.ndarray) -> None:
        """
        Write a tensor dataset to the HDF5 cache under a state_id group.

        Parameters
        ----------
        state_id : str
            Unique state identifier hash.
        key : str
            Dataset name key.
        data : np.ndarray
            Numpy array to persist.
        """
        # Guard 1: In-process thread lock
        with self._thread_lock:
            # Guard 2: Cross-process file lock (skip on Lustre/NFS)
            lock_ctx = (
                contextlib.nullcontext()
                if self._lustre_bypass
                else self._file_lock
            )
            with lock_ctx:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Guard 3: SWMR-mode HDF5 write
                with h5py.File(str(self._cache_path), "a", libver="latest") as f:
                    try:
                        f.swmr_mode = True
                    except (ValueError, OSError, RuntimeError):
                        pass

                    group = f.require_group(state_id)
                    if key in group:
                        del group[key]
                    group.create_dataset(
                        key,
                        data=np.asarray(data, dtype=np.float64),
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                    )
                    f.flush()

    def get_cache_size_bytes(self) -> int:
        """
        Return the on-disk size of the HDF5 cache file in bytes.

        Returns
        -------
        int
            Size in bytes, or 0 if the cache file does not exist.
        """
        with self._thread_lock:
            if self._cache_path.exists() and self._cache_path.is_file():
                return self._cache_path.stat().st_size
            return 0

    @staticmethod
    def _resolve_cache_path(user_path: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the HDF5 cache file path using the Tier 2 hierarchy:
        1. Explicit user_path argument.
        2. COCHEM_ARTIFACTS_ROOT environment variable.
        3. cochem_system_config.json configuration file.
        4. Default fallback: Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".

        Parameters
        ----------
        user_path : str or Path, optional
            User specified path.

        Returns
        -------
        Path
            Fully resolved absolute Path.
        """
        if user_path is not None:
            return Path(user_path).resolve()

        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return (Path(artifacts_root) / "spycfit_tensor_cache.h5").resolve()

        default_dir = Path.home() / "CoChem_Artifacts"
        config_path = default_dir / "Registry" / "cochem_system_config.json"
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if isinstance(config, dict) and "tensor_cache_path" in config:
                    return Path(config["tensor_cache_path"]).resolve()
            except Exception as _err:
                _logger.debug("Failed reading system config: %s", _err)

        default_dir.mkdir(parents=True, exist_ok=True)
        return (default_dir / "spycfit_tensor_cache.h5").resolve()

    def __len__(self) -> int:
        """Return the number of cached states in the HDF5 container."""
        return len(self.list_cached_states())

    def __contains__(self, state_id: str) -> bool:
        """Check if a state_id exists in the cache container."""
        with self._thread_lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                return False
            try:
                f = h5py.File(str(self._cache_path), "r", libver="latest", swmr=True)
            except (ValueError, OSError, RuntimeError):
                f = h5py.File(str(self._cache_path), "r", libver="latest")
            with f:
                return state_id in f

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.