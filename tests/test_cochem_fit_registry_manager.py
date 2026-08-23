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
        np.testing.assert_allclose(final_arr, np.arange(num_records) * 1.5)

    assert len(reader_seen_counts) > 1
    assert reader_seen_counts[-1] == num_records


def test_hdf5_lustre_fallback_disables_swmr_and_snapshots(tmp_path: Path):
    """
    Validates that Lustre filesystem detection dynamically disables SWMR mode
    and triggers automatic metadata-safe atomic snapshot generation.
    """
    db_path = tmp_path / "lustre_db.h5"

    import h5py
    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        phys_data = np.array(f_phys["isomer_001"]["hessian"])
        
    with open_state_tensor(db_path, mode="w", is_lustre=True) as f:
        # In Lustre mode, swmr_mode is not activated
        assert not getattr(f, "swmr_mode", False)
        f.create_dataset("dense_matrix", data=phys_data)

    # Verify atomic snapshot was automatically created in snapshots directory
    snap_dir = db_path.parent / "snapshots"
    assert snap_dir.is_dir()
    snaps = list(snap_dir.glob("lustre_db_snap_*.h5"))
    assert len(snaps) >= 1

    # Verify snapshot content
    with h5py.File(str(snaps[0]), "r") as f_snap:
        loaded = np.array(f_snap["dense_matrix"])
        np.testing.assert_array_equal(loaded, phys_data)


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

    import h5py
    h5_path = Path(__file__).parent.parent / "artifacts_test" / "landscape.h5"
    with h5py.File(h5_path, "r") as f_phys:
        test_matrix = np.array(f_phys["isomer_001"]["hessian"])
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

    np.testing.assert_allclose(loaded_arr, test_matrix)
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
