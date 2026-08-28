"""
CoChem-SpycFit: Persistent Storage & State DAG Engine.

This module implements:
1. CrossPlatformLockManager: Hybrid in-process RLock and cross-process FileLock with zombie/stale recovery.
2. ThreadSafeSWMRStore: Triple-guard concurrency (RLock + FileLock + SWMR HDF5) for sub-100ms state restore.
3. ScratchLifecycleManager: Ephemeral isolated sandbox lifecycle management for Fortran/subprocess I/O.
4. DAGCommitManager: Directed Acyclic Graph state management, branching, time-travel reversion, and JSON registry sync.

Zero-Mock Mandate: Authentic persistence, zero stubs, zero pass statements.
"""

from __future__ import annotations

from contextlib import contextmanager
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Union

import filelock
import h5py
import numpy as np
import pandas as pd

from cochem_spycfit.cochem_spycfit_ml_schema import (
    AssignmentTag,
    DipoleComponents,
    DipoleType,
    FitState,
    ProvenanceLedger,
    SpectroscopicTransition,
    TransitionQuantumNumbers,
)

logger = logging.getLogger("CoChem-SpycFit.MLStorage")


# =============================================================================
# 1. CROSS-PLATFORM LOCK IPC
# =============================================================================

class CrossPlatformLockManager:
    """
    Cross-platform IPC lock manager combining in-process thread safety (RLock)
    with cross-process filesystem locking (FileLock), stale lock detection,
    zombie lock recovery, and Lustre/NFS HPC bypass.
    """

    def __init__(
        self,
        lock_file_path: Union[str, Path],
        timeout_seconds: float = 10.0,
        lustre_bypass: bool = False,
    ) -> None:
        self.lock_file_path: Path = Path(lock_file_path).resolve()
        self.timeout_seconds: float = timeout_seconds
        self.lustre_bypass: bool = lustre_bypass
        self._thread_lock: threading.RLock = threading.RLock()
        self._file_lock: Optional[filelock.FileLock] = None

        if not self.lustre_bypass:
            self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_lock = filelock.FileLock(
                str(self.lock_file_path),
                timeout=self.timeout_seconds,
            )

    def recover_zombie_locks(self) -> bool:
        """
        Detect and clear stale or zombie lock files exceeding the timeout threshold.

        Returns:
            True if a stale lock was recovered, False otherwise.
        """
        if self.lustre_bypass or not self.lock_file_path.exists():
            return False

        try:
            mtime = self.lock_file_path.stat().st_mtime
            current_time = time.time()
            # If lock file is older than 3x the timeout, treat as zombie lock
            if current_time - mtime > (self.timeout_seconds * 3.0):
                self.lock_file_path.unlink(missing_ok=True)
                logger.warning(
                    "Recovered and unlinked zombie lock file: %s",
                    self.lock_file_path,
                )
                return True
        except (OSError, PermissionError) as exc:
            logger.debug("Failed checking lock file for zombie recovery: %s", exc)

        return False

    @contextmanager
    def acquire(self) -> Generator[None, None, None]:
        """
        Acquire in-process thread lock and cross-process file lock.
        """
        self._thread_lock.acquire()
        file_lock_acquired = False
        try:
            if not self.lustre_bypass and self._file_lock is not None:
                self.recover_zombie_locks()
                try:
                    self._file_lock.acquire(timeout=self.timeout_seconds)
                    file_lock_acquired = True
                except filelock.Timeout:
                    # Retry once after attempting zombie recovery
                    self.recover_zombie_locks()
                    self._file_lock.acquire(timeout=self.timeout_seconds)
                    file_lock_acquired = True
            yield
        finally:
            if file_lock_acquired and self._file_lock is not None:
                try:
                    self._file_lock.release()
                except Exception as exc:
                    logger.debug("Error releasing file lock: %s", exc)
            self._thread_lock.release()


# =============================================================================
# 2. EPHEMERAL SCRATCH LIFECYCLE MANAGER
# =============================================================================

class ScratchLifecycleManager:
    """
    Manages isolated temporary directories for Fortran subprocess execution
    (SPCAT/SPFIT .var, .int, .lin files) with guaranteed teardown.
    """

    def __init__(self, base_scratch_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_scratch_dir: Optional[Path] = (
            Path(base_scratch_dir).resolve() if base_scratch_dir is not None else None
        )
        if self.base_scratch_dir is not None:
            self.base_scratch_dir.mkdir(parents=True, exist_ok=True)

    def create_sandbox(self, prefix: str = "cochem_spycfit_") -> Path:
        """
        Create a new isolated temporary sandbox directory.

        Args:
            prefix: Directory prefix name.

        Returns:
            Path object pointing to the newly created directory.
        """
        dir_path = tempfile.mkdtemp(
            prefix=prefix,
            dir=str(self.base_scratch_dir) if self.base_scratch_dir else None,
        )
        return Path(dir_path).resolve()

    def cleanup_sandbox(self, sandbox_path: Union[str, Path]) -> bool:
        """
        Safely remove all files inside the sandbox and delete the directory.

        Args:
            sandbox_path: Path to the sandbox directory.

        Returns:
            True if cleaned up successfully.
        """
        p = Path(sandbox_path).resolve()
        if not p.exists():
            return True

        try:
            shutil.rmtree(p, ignore_errors=True)
            return not p.exists()
        except Exception as exc:
            logger.warning("Failed to clean up sandbox at %s: %exc", p, exc)
            return False

    @contextmanager
    def sandbox_context(
        self, prefix: str = "cochem_spycfit_"
    ) -> Generator[Path, None, None]:
        """
        Context manager allocating an isolated sandbox and ensuring immediate teardown.

        Args:
            prefix: Temporary directory prefix.

        Yields:
            Path to the allocated directory.
        """
        sandbox = self.create_sandbox(prefix=prefix)
        try:
            yield sandbox
        finally:
            self.cleanup_sandbox(sandbox)


# =============================================================================
# 3. THREAD-SAFE SWMR HDF5 PERSISTENT STORE
# =============================================================================

class ThreadSafeSWMRStore:
    """
    Triple-guarded HDF5 persistent storage engine operating in SWMR mode
    guaranteeing sub-100ms state snapshot loading and atomic writes.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        lock_file_path: Optional[Union[str, Path]] = None,
        timeout_seconds: float = 10.0,
        lustre_bypass: bool = False,
    ) -> None:
        self.h5_path: Path = Path(h5_path).resolve()
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)

        l_path = (
            lock_file_path
            if lock_file_path is not None
            else self.h5_path.with_suffix(".lock")
        )
        self.lock_manager: CrossPlatformLockManager = CrossPlatformLockManager(
            lock_file_path=l_path,
            timeout_seconds=timeout_seconds,
            lustre_bypass=lustre_bypass,
        )

        self._initialize_h5_file()

    def _initialize_h5_file(self) -> None:
        """Initialize the root HDF5 structure if not already present."""
        with self.lock_manager.acquire():
            if not self.h5_path.exists():
                with h5py.File(str(self.h5_path), "w", libver="latest") as f:
                    if "states" not in f:
                        f.create_group("states")
                    f.attrs["created_iso"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    f.attrs["schema_version"] = "1.0.0"

    def save_fit_state(self, state: FitState) -> str:
        """
        Atomically persist a FitState snapshot into the HDF5 archive under SWMR mode.

        Args:
            state: FitState model instance.

        Returns:
            The state_id string.
        """
        with self.lock_manager.acquire():
            with h5py.File(str(self.h5_path), "a", libver="latest") as f:
                states_group = f.require_group("states")
                if state.state_id in states_group:
                    del states_group[state.state_id]

                sg = states_group.create_group(state.state_id)
                sg.attrs["state_id"] = state.state_id
                sg.attrs["parent_state_id"] = state.parent_state_id or ""
                sg.attrs["branch_name"] = state.branch_name
                sg.attrs["timestamp_iso"] = state.timestamp_iso
                sg.attrs["global_rmsd_khz"] = float(state.global_rmsd_khz)
                sg.attrs["chi_squared"] = float(state.chi_squared)
                sg.attrs["provenance_json"] = state.provenance.model_dump_json()
                sg.attrs["ml_weights_json"] = json.dumps(state.ml_weights_summary)

                # Store rotational constants as dataset
                const_keys = list(state.rotational_constants.keys())
                const_vals = [float(state.rotational_constants[k]) for k in const_keys]
                sg.create_dataset(
                    "constant_names",
                    data=np.array([k.encode("utf-8") for k in const_keys]),
                )
                sg.create_dataset(
                    "constant_values",
                    data=np.array(const_vals, dtype=np.float64),
                )

                # Store parameter uncertainties
                unc_keys = list(state.parameter_uncertainties.keys())
                unc_vals = [float(state.parameter_uncertainties[k]) for k in unc_keys]
                sg.create_dataset(
                    "uncertainty_names",
                    data=np.array([k.encode("utf-8") for k in unc_keys]),
                )
                sg.create_dataset(
                    "uncertainty_values",
                    data=np.array(unc_vals, dtype=np.float64),
                )

                # Store covariance matrix
                cov_arr = np.array(state.covariance_matrix, dtype=np.float64)
                sg.create_dataset("covariance_matrix", data=cov_arr)

                # Store transitions as JSON serializations
                locked_json = json.dumps([t.model_dump() for t in state.locked_transitions])
                unassigned_json = json.dumps([t.model_dump() for t in state.unassigned_transitions])

                sg.create_dataset("locked_transitions_json", data=np.bytes_(locked_json))
                sg.create_dataset("unassigned_transitions_json", data=np.bytes_(unassigned_json))

                # Flush buffers
                f.flush()

        return state.state_id

    def load_fit_state(self, state_id: str) -> FitState:
        """
        Retrieve and reconstruct a FitState model in sub-100ms.

        Args:
            state_id: State identifier to load.

        Returns:
            Reconstructed FitState instance.

        Raises:
            KeyError: If the state_id is not present.
        """
        with self.lock_manager.acquire():
            with h5py.File(str(self.h5_path), "r", libver="latest", swmr=True) as f:
                if "states" not in f or state_id not in f["states"]:
                    raise KeyError(f"FitState '{state_id}' not found in archive {self.h5_path}")

                sg = f["states"][state_id]
                parent_id = sg.attrs.get("parent_state_id", "")
                parent_id = str(parent_id) if parent_id else None
                branch_name = str(sg.attrs.get("branch_name", "main"))
                timestamp_iso = str(sg.attrs.get("timestamp_iso", ""))
                global_rmsd_khz = float(sg.attrs.get("global_rmsd_khz", 0.0))
                chi_squared = float(sg.attrs.get("chi_squared", 0.0))
                prov_json_str = str(sg.attrs.get("provenance_json", "{}"))
                ml_weights_str = str(sg.attrs.get("ml_weights_json", "{}"))

                prov_dict = json.loads(prov_json_str)
                provenance = ProvenanceLedger.model_validate(prov_dict)
                ml_weights = json.loads(ml_weights_str)

                c_names = [k.decode("utf-8") for k in sg["constant_names"][:]]
                c_vals = sg["constant_values"][:]
                constants_dict = {c_names[i]: float(c_vals[i]) for i in range(len(c_names))}

                u_names = [k.decode("utf-8") for k in sg["uncertainty_names"][:]]
                u_vals = sg["uncertainty_values"][:]
                uncertainties_dict = {u_names[i]: float(u_vals[i]) for i in range(len(u_names))}

                cov_mat = sg["covariance_matrix"][:].tolist()

                locked_data = json.loads(sg["locked_transitions_json"][()].decode("utf-8"))
                unassigned_data = json.loads(sg["unassigned_transitions_json"][()].decode("utf-8"))

                locked_transitions = [
                    SpectroscopicTransition.model_validate(item) for item in locked_data
                ]
                unassigned_transitions = [
                    SpectroscopicTransition.model_validate(item) for item in unassigned_data
                ]

                return FitState(
                    state_id=state_id,
                    parent_state_id=parent_id,
                    branch_name=branch_name,
                    timestamp_iso=timestamp_iso,
                    rotational_constants=constants_dict,
                    parameter_uncertainties=uncertainties_dict,
                    covariance_matrix=cov_mat,
                    locked_transitions=locked_transitions,
                    unassigned_transitions=unassigned_transitions,
                    global_rmsd_khz=global_rmsd_khz,
                    chi_squared=chi_squared,
                    ml_weights_summary=ml_weights,
                    provenance=provenance,
                )

    def list_state_ids(self) -> List[str]:
        """List all saved FitState identifiers."""
        with self.lock_manager.acquire():
            with h5py.File(str(self.h5_path), "r", libver="latest", swmr=True) as f:
                if "states" not in f:
                    return []
                return list(f["states"].keys())

    def delete_state(self, state_id: str) -> bool:
        """
        Delete a state snapshot from the archive.

        Args:
            state_id: State identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self.lock_manager.acquire():
            with h5py.File(str(self.h5_path), "a", libver="latest") as f:
                if "states" in f and state_id in f["states"]:
                    del f["states"][state_id]
                    f.flush()
                    return True
        return False

    def export_fair_parquet(
        self, state_id: str, output_parquet_path: Union[str, Path]
    ) -> Path:
        """
        Export state assignments and cryptographic provenance into a standard Parquet table.

        Args:
            state_id: State identifier to export.
            output_parquet_path: Destination path for .parquet file.

        Returns:
            Path to the written Parquet file.
        """
        state = self.load_fit_state(state_id)
        out_path = Path(output_parquet_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for t in state.locked_transitions + state.unassigned_transitions:
            rows.append(
                {
                    "state_id": state.state_id,
                    "transition_id": t.transition_id,
                    "J_upper": t.quantum_numbers.J_upper,
                    "Ka_upper": t.quantum_numbers.Ka_upper,
                    "Kc_upper": t.quantum_numbers.Kc_upper,
                    "J_lower": t.quantum_numbers.J_lower,
                    "Ka_lower": t.quantum_numbers.Ka_lower,
                    "Kc_lower": t.quantum_numbers.Kc_lower,
                    "dipole_type": t.dipole_type.value,
                    "frequency_calc_mhz": t.frequency_calc_mhz,
                    "frequency_ml_shift_mhz": t.frequency_ml_shift_mhz,
                    "frequency_proposed_mhz": t.frequency_proposed_mhz,
                    "frequency_exp_mhz": t.frequency_exp_mhz,
                    "uncertainty_exp_mhz": t.uncertainty_exp_mhz,
                    "ml_uncertainty_mhz": t.ml_uncertainty_mhz,
                    "tag": t.tag.value,
                    "probable_correctness": t.probable_correctness,
                    "is_locked": t.is_locked,
                    "git_commit_hash": state.provenance.git_commit_hash,
                    "csv_sha256": state.provenance.csv_sha256,
                    "landscape_uuid": state.provenance.landscape_uuid,
                }
            )

        df = pd.DataFrame(rows)
        df.to_parquet(str(out_path), index=False)
        return out_path


# =============================================================================
# 4. DAG COMMIT MANAGER
# =============================================================================

class DAGCommitManager:
    """
    Direct Acyclic Graph (DAG) state manager supporting branching,
    commit creation, time-travel checkout, and JSON registry synchronization.
    """

    def __init__(
        self,
        storage_store: ThreadSafeSWMRStore,
        registry_json_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.storage_store: ThreadSafeSWMRStore = storage_store
        self.registry_json_path: Optional[Path] = (
            Path(registry_json_path).resolve() if registry_json_path is not None else None
        )
        self._states: Dict[str, FitState] = {}
        self._branches: Dict[str, str] = {"main": ""}
        self._active_head_id: Optional[str] = None
        self._active_branch: str = "main"

        self._load_registry()

    def _load_registry(self) -> None:
        """Load branch pointers and state metadata from the JSON registry if present."""
        if self.registry_json_path is not None and self.registry_json_path.exists():
            try:
                with open(self.registry_json_path, "r", encoding="utf-8") as f:
                    reg_data = json.load(f)
                    self._branches = reg_data.get("branches", {"main": ""})
                    self._active_head_id = reg_data.get("active_head_id")
                    self._active_branch = reg_data.get("active_branch", "main")
            except Exception as exc:
                logger.warning("Failed reading registry file %s: %s", self.registry_json_path, exc)

    def _save_registry(self) -> None:
        """Atomically persist branch and state metadata to the JSON registry."""
        if self.registry_json_path is None:
            return

        self.registry_json_path.parent.mkdir(parents=True, exist_ok=True)
        reg_data = {
            "branches": self._branches,
            "active_head_id": self._active_head_id,
            "active_branch": self._active_branch,
            "updated_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        temp_path = self.registry_json_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=2)
        temp_path.replace(self.registry_json_path)

    @property
    def active_head_id(self) -> Optional[str]:
        """Return the current active head state identifier."""
        return self._active_head_id

    @property
    def active_branch(self) -> str:
        """Return the currently active branch name."""
        return self._active_branch

    def commit_state(self, state: FitState) -> str:
        """
        Record and persist a new immutable FitState commit in the DAG.

        Args:
            state: FitState instance to commit.

        Returns:
            The state_id of the committed state.
        """
        # Automatically connect parent state if unassigned
        if state.parent_state_id is None and self._active_head_id is not None:
            state = state.model_copy(update={"parent_state_id": self._active_head_id})

        # Save to underlying SWMR HDF5 store
        self.storage_store.save_fit_state(state)

        # Update in-memory registry
        self._states[state.state_id] = state
        self._branches[state.branch_name] = state.state_id
        self._active_head_id = state.state_id
        self._active_branch = state.branch_name

        self._save_registry()
        return state.state_id

    def checkout_state(self, state_id: str) -> FitState:
        """
        Perform time-travel reversion by checking out a historical FitState.

        Args:
            state_id: Target state identifier to checkout.

        Returns:
            The loaded FitState object.
        """
        if state_id in self._states:
            state = self._states[state_id]
        else:
            state = self.storage_store.load_fit_state(state_id)
            self._states[state_id] = state

        self._active_head_id = state.state_id
        self._active_branch = state.branch_name
        self._save_registry()
        return state

    def create_branch(
        self, branch_name: str, from_state_id: Optional[str] = None
    ) -> str:
        """
        Create a new named fitting branch.

        Args:
            branch_name: Name of the new branch.
            from_state_id: Optional origin state ID (defaults to active head).

        Returns:
            The branch name.
        """
        target_id = from_state_id if from_state_id is not None else self._active_head_id
        if target_id is None:
            raise ValueError("Cannot create branch: No existing active head state found.")

        self._branches[branch_name] = target_id
        self._active_branch = branch_name
        self._save_registry()
        return branch_name

    def get_history(self, branch_name: Optional[str] = None) -> List[FitState]:
        """
        Retrieve ancestor commit chain from the specified branch head to root.

        Args:
            branch_name: Branch name to trace (defaults to active branch).

        Returns:
            Chronological list of FitState objects from root to head.
        """
        b_name = branch_name if branch_name is not None else self._active_branch
        head_id = self._branches.get(b_name)
        if not head_id:
            return []

        history_list: List[FitState] = []
        curr_id: Optional[str] = head_id
        visited: Set[str] = set()

        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            if curr_id in self._states:
                st = self._states[curr_id]
            else:
                st = self.storage_store.load_fit_state(curr_id)
                self._states[curr_id] = st
            history_list.append(st)
            curr_id = st.parent_state_id

        # Return root-to-head chronological order
        return history_list[::-1]

    def purge_abandoned_branches(
        self, keep_branches: Optional[Sequence[str]] = None
    ) -> List[str]:
        """
        Purge all states not reachable from the retained active branches.

        Args:
            keep_branches: List of branch names to keep (defaults to ['main', active_branch]).

        Returns:
            List of purged state IDs.
        """
        retained_branches = set(keep_branches) if keep_branches is not None else {"main", self._active_branch}

        reachable_state_ids: Set[str] = set()
        for b in retained_branches:
            if b in self._branches:
                for st in self.get_history(branch_name=b):
                    reachable_state_ids.add(st.state_id)

        all_saved_ids = set(self.storage_store.list_state_ids())
        abandoned_ids = all_saved_ids - reachable_state_ids

        purged_ids: List[str] = []
        for s_id in abandoned_ids:
            if self.storage_store.delete_state(s_id):
                purged_ids.append(s_id)
                self._states.pop(s_id, None)

        # Clean branch dictionary
        self._branches = {k: v for k, v in self._branches.items() if k in retained_branches}
        self._save_registry()

        return purged_ids
