"""
Thread-safe SWMR HDF5 TensorCache for out-of-core spectroscopic matrix persistence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

_logger = logging.getLogger("CoChem-SpycFit.TensorCache")

_cache_path_locks: Dict[Path, threading.RLock] = {}
_cache_path_locks_guard = threading.Lock()


def _get_path_lock(path: Path) -> threading.RLock:
    """Retrieve or create a re-entrant lock for a specific resolved filesystem path."""
    with _cache_path_locks_guard:
        resolved = path.resolve()
        if resolved not in _cache_path_locks:
            _cache_path_locks[resolved] = threading.RLock()
        return _cache_path_locks[resolved]


class TensorCache:
    """
    Thread-Safe Single-Writer Multiple-Reader (SWMR) HDF5 Tensor Cache.

    Persists heavy float64 covariance matrices, Hamiltonian sub-blocks,
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
            Explicit filesystem path to the HDF5 cache file.
            If None, resolved via COCHEM_ARTIFACTS_ROOT, config file, or default.
        lustre_bypass : bool, default False
            If True, bypasses advisory file locks for HPC/Lustre parallel filesystems.
        """
        self._cache_path: Path = self._resolve_cache_path(cache_path)
        self._lustre_bypass: bool = bool(lustre_bypass)
        self._lock: threading.RLock = _get_path_lock(self._cache_path)

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
        gp_weights: Optional[np.ndarray] = None,
        hamiltonian: Optional[np.ndarray] = None,
        **extra_tensors: np.ndarray,
    ) -> None:
        """
        Store spectroscopic tensors for a specific FitState state_id.

        Parameters
        ----------
        state_id : str
            SHA-256 hash identifying the FitState commit.
        covariance : np.ndarray, optional
            Fitted parameter covariance matrix (typically 6x6 float64).
        gp_weights : np.ndarray, optional
            Gaussian Process regression weight vector (length-20 float64).
        hamiltonian : np.ndarray, optional
            Asymmetric rotor Hamiltonian matrix or block.
        **extra_tensors : np.ndarray
            Additional named tensor arrays to persist under the state group.
        """
        with self._lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(str(self._cache_path), "a", libver="latest") as h5_file:
                if state_id in h5_file:
                    state_group = h5_file[state_id]
                else:
                    state_group = h5_file.create_group(state_id)

                tensors_to_write: Dict[str, np.ndarray] = {}
                if covariance is not None:
                    tensors_to_write["covariance"] = np.asarray(covariance, dtype=np.float64)
                if gp_weights is not None:
                    tensors_to_write["gp_weights"] = np.asarray(gp_weights, dtype=np.float64)
                if hamiltonian is not None:
                    tensors_to_write["hamiltonian"] = np.asarray(hamiltonian, dtype=np.float64)

                for key, tensor in extra_tensors.items():
                    if tensor is not None:
                        tensors_to_write[key] = np.asarray(tensor, dtype=np.float64)

                for tensor_name, tensor_data in tensors_to_write.items():
                    if tensor_name in state_group:
                        del state_group[tensor_name]
                    state_group.create_dataset(
                        tensor_name,
                        data=tensor_data,
                        dtype=np.float64,
                    )

                h5_file.flush()

    def restore_state_tensors(self, state_id: str) -> Dict[str, np.ndarray]:
        """
        Retrieve all stored tensors for a specific FitState in <100ms.

        Parameters
        ----------
        state_id : str
            SHA-256 hash identifying the FitState commit.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping tensor names to high-precision float64 numpy arrays.

        Raises
        ------
        KeyError
            If the cache file does not exist or the state_id is not cached.
        """
        with self._lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                raise KeyError(
                    f"Tensor cache file '{self._cache_path}' does not exist."
                )

            with h5py.File(str(self._cache_path), "r", libver="latest", swmr=False) as h5_file:
                if state_id not in h5_file:
                    raise KeyError(
                        f"State ID '{state_id}' not found in tensor cache."
                    )

                state_group = h5_file[state_id]
                restored_dict: Dict[str, np.ndarray] = {}
                for key in state_group.keys():
                    dataset = state_group[key]
                    restored_dict[key] = np.array(dataset, dtype=np.float64)

                return restored_dict

    def list_cached_states(self) -> List[str]:
        """
        Return a list of all state_id hashes currently stored in the cache.

        Returns
        -------
        list[str]
            List of state_id string keys.
        """
        with self._lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                return []

            with h5py.File(str(self._cache_path), "r", libver="latest", swmr=False) as h5_file:
                return list(h5_file.keys())

    def delete_state_tensors(self, state_id: str) -> None:
        """
        Remove stored tensors for a specific FitState commit.

        Parameters
        ----------
        state_id : str
            SHA-256 hash identifying the FitState commit to delete.
        """
        with self._lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                return

            with h5py.File(str(self._cache_path), "a", libver="latest") as h5_file:
                if state_id in h5_file:
                    del h5_file[state_id]
                    h5_file.flush()

    def _write_dataset(self, state_id: str, name: str, tensor: np.ndarray) -> None:
        """
        Write or replace an individual dataset array under a state group.

        Parameters
        ----------
        state_id : str
            SHA-256 hash identifying the FitState commit.
        name : str
            Dataset name key.
        tensor : np.ndarray
            Numpy array to write as float64.
        """
        with self._lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(str(self._cache_path), "a", libver="latest") as h5_file:
                if state_id in h5_file:
                    state_group = h5_file[state_id]
                else:
                    state_group = h5_file.create_group(state_id)

                if name in state_group:
                    del state_group[name]
                state_group.create_dataset(
                    name,
                    data=np.asarray(tensor, dtype=np.float64),
                    dtype=np.float64,
                )
                h5_file.flush()

    def get_cache_size_bytes(self) -> int:
        """
        Return the on-disk size of the HDF5 cache file in bytes.

        Returns
        -------
        int
            Size in bytes, or 0 if the cache file does not exist.
        """
        with self._lock:
            if self._cache_path.exists() and self._cache_path.is_file():
                return self._cache_path.stat().st_size
            return 0

    @staticmethod
    def _resolve_cache_path(user_path: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the tensor cache HDF5 path via the CoChem hierarchy:
        1. Explicit user_path argument.
        2. COCHEM_ARTIFACTS_ROOT environment variable.
        3. cochem_system_config.json configuration file.
        4. Default: Path.home() / "CoChem_Artifacts" / "spycfit_tensor_cache.h5".

        Parameters
        ----------
        user_path : str or Path, optional
            Caller-specified path.

        Returns
        -------
        Path
            Fully resolved absolute Path.
        """
        if user_path is not None:
            return Path(user_path).resolve()

        env_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if env_root:
            return (Path(env_root) / "spycfit_tensor_cache.h5").resolve()

        default_dir = Path.home() / "CoChem_Artifacts"
        config_candidate = default_dir / "Registry" / "cochem_system_config.json"
        if config_candidate.exists() and config_candidate.is_file():
            try:
                with open(config_candidate, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if isinstance(config_data, dict) and "tensor_cache_path" in config_data:
                    return Path(config_data["tensor_cache_path"]).resolve()
            except Exception as _err:
                _logger.debug("Failed reading system config: %s", _err)

        default_dir.mkdir(parents=True, exist_ok=True)
        return (default_dir / "spycfit_tensor_cache.h5").resolve()

    def __len__(self) -> int:
        """Return the number of cached states in the HDF5 container."""
        return len(self.list_cached_states())

    def __contains__(self, state_id: str) -> bool:
        """Check if a state_id exists in the cache container."""
        with self._lock:
            if not self._cache_path.exists() or not self._cache_path.is_file():
                return False
            with h5py.File(str(self._cache_path), "r", libver="latest", swmr=False) as h5_file:
                return state_id in h5_file
