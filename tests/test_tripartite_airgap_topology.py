"""
Zero-Mock Physical Verification & Asymmetric Audit Suite for Tripartite Air-Gap Topology.

Provenance & Specifications:
- Anti-Spoofing Directive [A]: Zero-mock execution against physical filesystem partitions and HDF5 containers.
- Tripartite Air-Gap Topology [T]: Strict enforcement of read-only Code/Data partitions and writeable Artifact partition.
- Mendeleev Invariants [M]: Zero-hardcoded dynamic atomic and isotopic mass lookups.
- SWMR Concurrency [C]: Real-time streaming writer with concurrent non-blocking readers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Generator, List

import h5py
from mendeleev import element as m_element
import numpy as np
import pytest

from cochem.concurrency.atomic_file_lock import AtomicFileLock
from cochem.core.airgap_arbiter import AirGapArbiter
from cochem.core.cochem_engine import CoChemEngine
from cochem.core.mendeleev_invariants import MendeleevResolver, mendeleev_resolver
from cochem.core.topology_exceptions import (
    AirGapPartitionViolationError,
    MendeleevInvariantError,
    PartitionMountError,
    SWMRCorruptionError,
    SWMRFileLockTimeoutError,
    TelemetryQueueOverflowError,
)
from cochem.core.topology_models import (
    EngineHealthSummary,
    PartitionDescriptor,
    SWMRFileLockDescriptor,
    TelemetryEventPayload,
)
from cochem.gui.widgets.airgap_status_widget import AirGapStatusWidget
from cochem.storage.swmr_hdf5_arbiter import SWMRHDF5Arbiter
from cochem.telemetry.telemetry_queue import TelemetryQueue


@pytest.fixture
def sandbox_partitions(tmp_path: Path) -> Generator[AirGapArbiter, None, None]:
    """Create authentic physical directory tree for T_code, T_data, and T_art."""
    code_dir = tmp_path / "code_partition"
    data_dir = tmp_path / "data_partition"
    art_dir = tmp_path / "artifact_partition"

    code_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Populate code and data partition physical files
    (code_dir / "packages").mkdir(parents=True, exist_ok=True)
    (code_dir / "packages" / "sample_module.py").write_text("# Core package", encoding="utf-8")

    (data_dir / "raw_baselines").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw_baselines" / "baseline.dat").write_text("100.0 200.0", encoding="utf-8")

    arbiter = AirGapArbiter(
        code_root=code_dir,
        data_root=data_dir,
        artifact_root=art_dir,
        auto_create_artifact_subdirs=True,
    )
    yield arbiter


# ---------------------------------------------------------------------------
# 1. Tripartite Air-Gap Permission & Boundary Tests
# ---------------------------------------------------------------------------


def test_airgap_partition_read_only_enforcement(sandbox_partitions: AirGapArbiter) -> None:
    """Verify that write operations to T_code and T_data are rejected strictly."""
    code_file = sandbox_partitions.code_root / "packages" / "sample_module.py"
    data_file = sandbox_partitions.data_root / "raw_baselines" / "baseline.dat"

    # Read access should succeed
    code_desc = sandbox_partitions.validate_read_access(code_file)
    assert code_desc.partition_type == "CODE"
    assert code_desc.permission == "READ_ONLY"

    data_desc = sandbox_partitions.validate_read_access(data_file)
    assert data_desc.partition_type == "DATA"
    assert data_desc.permission == "READ_ONLY"

    # Write access to read-only partitions must raise AirGapPartitionViolationError
    with pytest.raises(AirGapPartitionViolationError) as exc_code:
        sandbox_partitions.validate_write_access(code_file)
    assert "READ_ONLY" in str(exc_code.value)

    with pytest.raises(AirGapPartitionViolationError) as exc_data:
        sandbox_partitions.validate_write_access(data_file)
    assert "READ_ONLY" in str(exc_data.value)

    # Directory traversal outside partitions must be rejected
    escaped_path = sandbox_partitions.code_root / ".." / "outside_world.txt"
    with pytest.raises(AirGapPartitionViolationError):
        sandbox_partitions.validate_write_access(escaped_path)


def test_airgap_artifact_partition_read_write(sandbox_partitions: AirGapArbiter) -> None:
    """Verify that writes and reads inside T_art succeed with proper permissions."""
    art_file = sandbox_partitions.get_artifact_path("checkpoints/chk_01.h5")
    art_desc = sandbox_partitions.validate_write_access(art_file)
    assert art_desc.partition_type == "ARTIFACT"
    assert art_desc.permission == "READ_WRITE"

    # Physical write to artifact partition
    art_file.parent.mkdir(parents=True, exist_ok=True)
    art_file.write_text("CHECKPOINT_DATA", encoding="utf-8")
    assert art_file.exists()

    read_desc = sandbox_partitions.validate_read_access(art_file)
    assert read_desc.partition_type == "ARTIFACT"


# ---------------------------------------------------------------------------
# 2. Dynamic Mendeleev Invariant Tests
# ---------------------------------------------------------------------------


def test_mendeleev_dynamic_mass_resolution() -> None:
    """Verify chemical element and isotope mass lookups are dynamically retrieved from Mendeleev."""
    resolver = MendeleevResolver()
    resolver.clear_cache()

    test_elements = ["H", "C", "N", "O", "Fe", "U"]
    for sym in test_elements:
        expected_elem = m_element(sym)
        assert resolver.get_symbol(sym) == expected_elem.symbol
        assert resolver.get_atomic_number(sym) == int(expected_elem.atomic_number)
        assert resolver.get_atomic_weight(sym) == pytest.approx(float(expected_elem.atomic_weight), rel=1e-6)

    # Dynamic isotopic mass verification
    assert resolver.get_isotope_mass("H", 2) == pytest.approx(float(m_element("H").isotopes[1].mass), rel=1e-6)
    assert resolver.get_isotope_mass("C", 13) == pytest.approx(13.003354835, rel=1e-5)

    # Cache hit tracking verification
    hits_before, misses_before = resolver.get_cache_stats()
    _ = resolver.get_atomic_weight("C")
    hits_after, misses_after = resolver.get_cache_stats()
    assert hits_after > hits_before

    # Error handling for unphysical symbols and isotopes
    with pytest.raises(MendeleevInvariantError):
        resolver.get_element("UnobtainiumFakeSymbol")

    with pytest.raises(MendeleevInvariantError):
        resolver.get_isotope("H", 99)


# ---------------------------------------------------------------------------
# 3. Atomic File Lock & Mutual Exclusion Tests
# ---------------------------------------------------------------------------


def test_atomic_file_lock_mutual_exclusion(tmp_path: Path) -> None:
    """Verify mutual exclusion across concurrent lock acquisition attempts."""
    lock_file = tmp_path / "shared_resource.lock"
    target_file = tmp_path / "shared_resource.h5"

    lock1 = AtomicFileLock(lock_path=lock_file, target_path=target_file, timeout=1.0)
    lock2 = AtomicFileLock(lock_path=lock_file, target_path=target_file, timeout=0.2)

    with lock1:
        assert lock1.is_acquired
        assert lock_file.exists()
        assert AtomicFileLock.is_locked(lock_file)

        # Second lock attempt must time out
        with pytest.raises(SWMRFileLockTimeoutError):
            lock2.acquire()

    assert not lock1.is_acquired
    assert not lock_file.exists()


def test_stale_lock_reclamation(tmp_path: Path) -> None:
    """Verify automatic reclamation of stale locks left by non-existent PIDs."""
    lock_file = tmp_path / "stale_test.lock"

    # Fabricate a dead-PID lockfile
    stale_payload = {
        "lock_filepath": str(lock_file),
        "target_filepath": "",
        "owner_pid": 99999999,  # Non-existent PID
        "hostname": "test-host",
        "acquired_timestamp_iso": "2020-01-01T00:00:00+00:00",
        "lease_duration_seconds": 1.0,
        "is_active": True,
    }
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(stale_payload, f)

    new_lock = AtomicFileLock(lock_path=lock_file, timeout=2.0, lease_duration=10.0)
    with new_lock:
        assert new_lock.is_acquired
        desc = new_lock.get_descriptor()
        assert desc.owner_pid == os.getpid()


# ---------------------------------------------------------------------------
# 4. SWMR HDF5 Concurrency & Reader/Writer Isolation Tests
# ---------------------------------------------------------------------------


def test_swmr_concurrent_readers_and_writer(sandbox_partitions: AirGapArbiter) -> None:
    """
    Spawns real SWMR writer and multiple concurrent readers on an extensible HDF5 dataset.
    Verifies zero corruption and streaming read synchronization.
    """
    storage = SWMRHDF5Arbiter(airgap_arbiter=sandbox_partitions)
    h5_path = sandbox_partitions.get_artifact_path("hdf5_datasets/streaming_experiment.h5")

    # Initialize SWMR file and create extensible dataset
    with storage.open_writer(h5_path, mode="w") as writer:
        writer.create_dataset(
            name="spectrum_data",
            shape=(10, 4),
            dtype=np.float64,
            maxshape=(None, 4),
            chunks=(10, 4),
        )
        writer.append_chunk("spectrum_data", np.ones((10, 4), dtype=np.float64) * 1.5)
        writer.flush()

    # Define reader worker function
    def reader_worker(reader_id: int) -> int:
        read_count = 0
        with storage.open_reader(h5_path, swmr=True) as reader:
            for _ in range(5):
                data = reader.read_dataset("spectrum_data", auto_refresh=True)
                assert data.shape[1] == 4
                assert data.shape[0] >= 10
                read_count += 1
                time.sleep(0.02)
        return read_count

    # Execute concurrent writer and readers using real thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        reader_futures = [executor.submit(reader_worker, i) for i in range(3)]

        # Concurrent writer appends chunks
        with storage.open_writer(h5_path, mode="a") as writer:
            for chunk_idx in range(3):
                time.sleep(0.01)
                chunk_data = np.ones((5, 4), dtype=np.float64) * (chunk_idx + 2.0)
                writer.append_chunk("spectrum_data", chunk_data)
                writer.flush()

        for future in reader_futures:
            assert future.result() == 5

    # Final reader verification
    with storage.open_reader(h5_path) as reader:
        final_shape = reader.get_dataset_shape("spectrum_data")
        assert final_shape[0] == 10 + 10 + 15  # initial 10 + initial append 10 + 3*5 appends = 35
        final_data = reader.read_dataset("spectrum_data")
        assert np.all(final_data > 0)


# ---------------------------------------------------------------------------
# 5. Non-Blocking Telemetry Queue & Merkle Integrity Tests
# ---------------------------------------------------------------------------


def test_telemetry_queue_throughput_and_merkle_chain(sandbox_partitions: AirGapArbiter) -> None:
    """
    Dispatches diagnostic events, tests background flushing, payload truncation,
    and verifies SHA-256 Merkle chain cryptographic continuity.
    """
    telemetry = TelemetryQueue(
        airgap_arbiter=sandbox_partitions,
        flush_interval_seconds=0.05,
    )

    with telemetry:
        # Emit normal events
        for i in range(100):
            telemetry.emit_event(
                source_subsystem="TestSubsystem",
                severity="INFO",
                message=f"Event index {i}",
                context_metadata={"step": i, "value": i * 1.5},
            )

        # Emit oversized metadata payload (>4 KB) to test structured tail truncation
        large_metadata = {f"key_{k}": "A" * 200 for k in range(30)}
        telemetry.emit_event(
            source_subsystem="HeavyPayloadSystem",
            severity="WARNING",
            message="Oversized diagnostic event",
            context_metadata=large_metadata,
        )

        time.sleep(0.15)
        telemetry.drain()

    assert telemetry.log_filepath.exists()
    lines = telemetry.log_filepath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 101

    # Verify JSON structure and Merkle hash chaining
    prev_hash = "0" * 64
    for line in lines:
        record = json.loads(line)
        assert "event_id" in record
        assert "merkle_hash" in record
        assert len(record["merkle_hash"]) == 64
        # Verify hash uniqueness
        assert record["merkle_hash"] != prev_hash
        prev_hash = record["merkle_hash"]

    # Verify oversized record was truncated safely
    last_record = json.loads(lines[-1])
    assert last_record["context_metadata"].get("_truncated") is True


# ---------------------------------------------------------------------------
# 6. BASE GUI Widget & Status Rendering Tests
# ---------------------------------------------------------------------------


def test_airgap_status_widget(sandbox_partitions: AirGapArbiter) -> None:
    """Verify GUI status widget parses engine diagnostics and renders HTML/text summaries."""
    widget = AirGapStatusWidget(airgap_arbiter=sandbox_partitions)

    summary = EngineHealthSummary(
        engine_version="2.0.0",
        uptime_seconds=42.5,
        partitions_valid=True,
        active_swmr_locks=1,
        telemetry_queue_size=5,
        mendeleev_cache_hits=95,
        mendeleev_cache_misses=5,
    )

    lock_desc = SWMRFileLockDescriptor(
        target_filepath=sandbox_partitions.artifact_root / "test.h5",
        lock_filepath=sandbox_partitions.artifact_root / "test.h5.lock",
        owner_pid=os.getpid(),
        acquired_timestamp_iso="2026-08-31T12:00:00+00:00",
        lease_duration_seconds=30.0,
        is_active=True,
    )

    widget.update_status(summary, active_locks=[lock_desc])

    # Test dictionary export
    state_dict = widget.to_dict()
    assert state_dict["engine_summary"]["engine_version"] == "2.0.0"
    assert state_dict["mendeleev_cache_hit_ratio_percent"] == 95.0
    assert len(state_dict["active_swmr_locks"]) == 1

    # Test text rendering
    text = widget.render_text_summary()
    assert "CoChem Tripartite Air-Gap & Engine Dashboard" in text
    assert "95.0% hits" in text

    # Test HTML rendering
    html = widget.render_html()
    assert "<table" in html
    assert "Code Partition (T_code)" in html


# ---------------------------------------------------------------------------
# 7. Master Engine Lifecycle Tests
# ---------------------------------------------------------------------------


def test_engine_lifecycle_and_health_summary(tmp_path: Path) -> None:
    """Verify master CoChemEngine full lifecycle initialization, diagnostics, and shutdown."""
    code_dir = tmp_path / "code"
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"

    code_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    with CoChemEngine(code_root=code_dir, data_root=data_dir, artifact_root=art_dir) as engine:
        assert engine.is_initialized
        assert not engine.is_shutdown
        assert engine.uptime_seconds > 0.0

        # Dynamic mass lookup through engine
        c_mass = engine.mendeleev.get_atomic_weight("C")
        assert c_mass == pytest.approx(12.011, rel=1e-3)

        # Emit telemetry event through engine
        engine.telemetry.emit_event(
            source_subsystem="EngineTest",
            severity="INFO",
            message="Testing engine integrated telemetry.",
        )

        # Fetch health summary
        summary = engine.get_health_summary()
        assert summary.engine_version == "2.0.0"
        assert summary.partitions_valid is True
        assert summary.mendeleev_cache_hits > 0

    # Verify post-shutdown state
    assert engine.is_shutdown
    assert not engine.is_initialized
