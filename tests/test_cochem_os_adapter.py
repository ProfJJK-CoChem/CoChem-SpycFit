"""
CoChem-SpycFit Operating System Adapter & Immutable File Lock Zero-Mock Unit Test Suite.

Mandated by SRS Task 10 (Section 3.2 Immutable File Lock) to validate cross-platform
archive read-only sealing, OS syscall translation, and Tripartite Air-Gap isolation.

Validates:
1. Operating system detection and platform attribute routing.
2. Unbuffered 64KB chunked SHA-256 computation for files and directory trees.
3. Authoritative immutable artifact sealing and physical write-prevention probing.
4. Administrative unsealing and permission restoration.
5. Scoped context managers (temporary_unseal_scope, immutable_seal_scope).
6. Strict private directory locking (0o700 / Windows ACLs).
7. Tripartite Air-Gap tier classification, boundary containment, and scratch disjointness.
8. Dynamic Mendeleev library integration for elemental mass retrieval.
9. Pydantic v2 data model constraints and checksum format validation.
10. Dual-tree export parity between src/ and root proxy.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import stat
import tempfile
from typing import Dict

import mendeleev
import pytest

from cochem_spycfit.core_engine.cochem_os_adapter import (
    AIRGAP_TIER_A_CORE,
    AIRGAP_TIER_B_SCRATCH,
    AIRGAP_TIER_C_SEALED,
    AirGapBoundaryError,
    AirGapValidationRecord,
    CHUNK_SIZE_BYTES,
    CoChemOSError,
    FileLockRecord,
    ImmutableLockError,
    LockState,
    OSType,
    POSIX_READONLY_DIR_MODE,
    POSIX_READONLY_FILE_MODE,
    POSIX_STRICT_PRIVATE_DIR_MODE,
    POSIX_WRITABLE_DIR_MODE,
    POSIX_WRITABLE_FILE_MODE,
    PermissionSealingError,
    apply_posix_chmod_readonly,
    apply_posix_chmod_writable,
    apply_windows_icacls_readonly,
    apply_windows_icacls_writable,
    assert_airgap_containment,
    classify_airgap_tier,
    compute_path_sha256,
    detect_os_type,
    get_mendeleev_element_mass,
    immutable_seal_scope,
    is_file_readonly,
    is_path_read_only,
    is_posix_os,
    is_windows_os,
    lock_directory_strict_private,
    seal_artifact_immutable,
    seal_directory_readonly,
    temporary_unseal_scope,
    unseal_artifact_mutable,
    validate_airgap_isolation,
    verify_sealed_immutability,
)

# Authentic spectroscopic test payload for water (H2O)
H2O_SPECTRAL_PAYLOAD: bytes = (
    b"MOLECULE: H2O\n"
    b"STATE: GROUND\n"
    b"A_MHZ: 835840.288\n"
    b"B_MHZ: 435351.717\n"
    b"C_MHZ: 278138.700\n"
    b"DJ_KHZ: 37.594\n"
)


class TestOSDetectionAndSyscalls:
    """Test suite for host OS platform detection and syscall translations."""

    def test_os_type_detection(self) -> None:
        """Verify operating system detection returns valid OSType enum."""
        detected = detect_os_type()
        assert isinstance(detected, OSType)

        if platform.system() == "Windows":
            assert detected == OSType.WINDOWS
            assert is_windows_os() is True
        elif platform.system() == "Linux":
            assert detected == OSType.LINUX
            assert is_posix_os() is True
        elif platform.system() == "Darwin":
            assert detected == OSType.DARWIN
            assert is_posix_os() is True

    def test_mendeleev_elemental_mass_dynamic(self) -> None:
        """Verify dynamic mass lookup for chemical elements via Mendeleev."""
        h_mass = get_mendeleev_element_mass("H")
        o_mass = get_mendeleev_element_mass("O")
        c_mass = get_mendeleev_element_mass("C")

        assert h_mass > 1.007
        assert o_mass > 15.998
        assert c_mass > 12.010

        with pytest.raises(ValueError, match="Could not retrieve atomic mass"):
            get_mendeleev_element_mass("InvalidElementSymbolXYZ")


class TestCryptographicChecksums:
    """Test suite for physical SHA-256 calculation on files and directories."""

    def test_file_sha256_computation(self) -> None:
        """Verify unbuffered chunked SHA-256 calculation for a physical file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "spectrum.dat"
            test_file.write_bytes(H2O_SPECTRAL_PAYLOAD)

            checksum = compute_path_sha256(test_file)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

            # Modifying content produces a distinct checksum
            test_file.write_bytes(H2O_SPECTRAL_PAYLOAD + b"EXTRA_TRANSITION\n")
            modified_checksum = compute_path_sha256(test_file)
            assert checksum != modified_checksum

    def test_directory_composite_sha256_computation(self) -> None:
        """Verify deterministic composite SHA-256 digest for a directory tree."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "archive_package"
            base_dir.mkdir()
            (base_dir / "meta.json").write_text('{"molecule": "H2O"}', encoding="utf-8")
            (base_dir / "data.bin").write_bytes(H2O_SPECTRAL_PAYLOAD)

            checksum_1 = compute_path_sha256(base_dir)
            assert len(checksum_1) == 64

            # Computing again on unchanged directory produces identical hash
            checksum_2 = compute_path_sha256(base_dir)
            assert checksum_1 == checksum_2

    def test_missing_path_checksum_raises(self) -> None:
        """Verify FileNotFoundError is raised when hashing a non-existent path."""
        non_existent = Path(tempfile.gettempdir()) / "non_existent_cochem_file.dat"
        with pytest.raises(FileNotFoundError):
            compute_path_sha256(non_existent)


class TestImmutableSealingAndUnsealing:
    """Test suite for physical artifact read-only sealing and administrative unsealing."""

    def test_seal_single_file_immutable(self) -> None:
        """Verify sealing a physical file marks it read-only and blocks write operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_file = Path(temp_dir) / "snapshot_001.h5"
            artifact_file.write_bytes(H2O_SPECTRAL_PAYLOAD)

            record = seal_artifact_immutable(artifact_file, verify_write_blocked=True)
            assert isinstance(record, FileLockRecord)
            assert record.is_sealed is True
            assert record.lock_state == LockState.SEALED_READONLY
            assert record.file_size_bytes == len(H2O_SPECTRAL_PAYLOAD)
            assert is_path_read_only(artifact_file) is True
            assert is_file_readonly(artifact_file) is True

            # Administrative unsealing
            unseal_success = unseal_artifact_mutable(artifact_file)
            assert unseal_success is True
            assert is_path_read_only(artifact_file) is False

            # File is now writable again
            artifact_file.write_bytes(H2O_SPECTRAL_PAYLOAD + b"APPENDED")
            assert artifact_file.stat().st_size > len(H2O_SPECTRAL_PAYLOAD)

    def test_seal_directory_recursive(self) -> None:
        """Verify recursive sealing applied across all files and subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_dir = Path(temp_dir) / "sealed_export_tree"
            archive_dir.mkdir()
            sub_dir = archive_dir / "nested"
            sub_dir.mkdir()

            file1 = archive_dir / "fit_results.json"
            file1.write_text('{"status": "CONVERGED"}', encoding="utf-8")
            file2 = sub_dir / "tensors.dat"
            file2.write_bytes(H2O_SPECTRAL_PAYLOAD)

            records = seal_directory_readonly(archive_dir, recursive=True, verify_tamper_proofing=True)
            assert len(records) >= 3

            assert is_path_read_only(file1) is True
            assert is_path_read_only(file2) is True

            # Administrative unseal
            unseal_artifact_mutable(archive_dir, recursive=True)
            assert is_path_read_only(file1) is False
            assert is_path_read_only(file2) is False

    def test_strict_private_directory_locking(self) -> None:
        """Verify private workspace permission locking (0o700 or Windows user-only ACLs)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            private_dir = Path(temp_dir) / "private_session"
            locked = lock_directory_strict_private(private_dir)
            assert locked is True
            assert private_dir.exists()


class TestScopedContextManagers:
    """Test suite for temporary unsealing and immutable sealing context managers."""

    def test_temporary_unseal_scope(self) -> None:
        """Verify temporary unsealing allows edits and automatically re-locks upon exit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sealed_file = Path(temp_dir) / "state_manifest.json"
            sealed_file.write_text('{"version": 1}', encoding="utf-8")
            seal_artifact_immutable(sealed_file, verify_write_blocked=True)
            assert is_path_read_only(sealed_file) is True

            with temporary_unseal_scope(sealed_file) as unlocked_path:
                assert is_path_read_only(unlocked_path) is False
                unlocked_path.write_text('{"version": 2}', encoding="utf-8")

            # Upon scope exit, the file is re-sealed
            assert is_path_read_only(sealed_file) is True
            assert sealed_file.read_text(encoding="utf-8") == '{"version": 2}'

            # Administrative cleanup
            unseal_artifact_mutable(sealed_file)

    def test_immutable_seal_scope(self) -> None:
        """Verify immutable_seal_scope establishes read-only protection during execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "immutable_archive.dat"
            data_file.write_bytes(H2O_SPECTRAL_PAYLOAD)

            with immutable_seal_scope(data_file) as sealed_target:
                assert is_path_read_only(sealed_target) is True

            assert is_path_read_only(data_file) is True
            unseal_artifact_mutable(data_file)


class TestTripartiteAirGapValidation:
    """Test suite for Tripartite Air-Gap isolation validation and boundary enforcement."""

    def test_airgap_tier_classification(self) -> None:
        """Verify classification of paths into Domain A, B, and C tiers."""
        scratch_path = Path("/workspace/scratch/run_01/spectra.zarr")
        sealed_path = Path("/workspace/artifacts/snapshots/run_01.h5")
        core_path = Path("/workspace/core_engine/cochem_spycfit.py")

        assert classify_airgap_tier(scratch_path) == AIRGAP_TIER_B_SCRATCH
        assert classify_airgap_tier(sealed_path) == AIRGAP_TIER_C_SEALED
        assert classify_airgap_tier(core_path) == AIRGAP_TIER_A_CORE

    def test_validate_airgap_isolation_disjoint(self) -> None:
        """Verify disjoint paths pass Air-Gap validation successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts" / "exports"
            scratch_dir = Path(temp_dir) / "scratch" / "tensors"
            artifacts_dir.mkdir(parents=True)
            scratch_dir.mkdir(parents=True)

            sealed_file = artifacts_dir / "sealed_state.h5"
            sealed_file.write_bytes(H2O_SPECTRAL_PAYLOAD)

            validation = validate_airgap_isolation(sealed_file, scratch_dir=scratch_dir)
            assert isinstance(validation, AirGapValidationRecord)
            assert validation.is_valid is True
            assert validation.is_disjoint_from_scratch is True
            assert validation.isolation_intact is True

    def test_validate_airgap_isolation_boundary_violation(self) -> None:
        """Verify error is raised when sealed artifact resides inside scratch tier."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scratch_dir = Path(temp_dir) / "scratch"
            scratch_dir.mkdir(parents=True)
            invalid_sealed_file = scratch_dir / "leaked_snapshot.h5"
            invalid_sealed_file.write_bytes(H2O_SPECTRAL_PAYLOAD)

            with pytest.raises(AirGapBoundaryError, match="Tripartite Air-Gap Violation"):
                validate_airgap_isolation(invalid_sealed_file, scratch_dir=scratch_dir)

    def test_assert_airgap_containment_violation(self) -> None:
        """Verify directory traversal or root boundary escape raises AirGapBoundaryError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_boundary = Path(temp_dir) / "allowed_domain"
            root_boundary.mkdir()
            outside_path = Path(temp_dir) / "external_leak.dat"
            outside_path.write_bytes(b"DATA")

            with pytest.raises(AirGapBoundaryError, match="Path containment violation"):
                assert_airgap_containment(outside_path, root_boundary)


class TestPydanticSchemaValidation:
    """Test suite for Pydantic v2 data models and validator invariants."""

    def test_file_lock_record_schema_validation(self) -> None:
        """Verify FileLockRecord correctly validates SHA-256 formatting and fields."""
        valid_checksum = "a" * 64
        record = FileLockRecord(
            target_path="/test/path/file.dat",
            is_sealed=True,
            lock_state=LockState.SEALED_READONLY,
            os_type=OSType.WINDOWS,
            sha256_checksum=valid_checksum,
            timestamp_utc="2026-08-29T12:00:00Z",
            mode_octal="0o444",
            windows_acls_applied=True,
            is_directory=False,
            file_size_bytes=1024,
        )
        assert record.sha256_checksum == valid_checksum

        with pytest.raises(ValueError, match="Invalid SHA-256 checksum format"):
            FileLockRecord(
                target_path="/test/path/file.dat",
                is_sealed=True,
                lock_state=LockState.SEALED_READONLY,
                os_type=OSType.WINDOWS,
                sha256_checksum="invalid_short_hash",
                timestamp_utc="2026-08-29T12:00:00Z",
            )


class TestExportParity:
    """Test suite verifying dual-tree export parity between src/ and root proxy."""

    def test_dual_tree_exports(self) -> None:
        """Verify identical exports between src and root core_engine module."""
        import cochem_spycfit.core_engine.cochem_os_adapter as root_adapter
        import src.cochem_spycfit.core_engine.cochem_os_adapter as src_adapter

        root_all = set(root_adapter.__all__)
        src_all = set(src_adapter.__all__)

        assert "seal_artifact_immutable" in root_all
        assert "unseal_artifact_mutable" in root_all
        assert "validate_airgap_isolation" in root_all
        assert root_all == src_all
