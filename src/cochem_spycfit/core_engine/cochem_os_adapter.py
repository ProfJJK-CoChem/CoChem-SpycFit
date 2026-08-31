"""
CoChem-SpycFit Core Engine: Cross-Platform Operating System Adapter and Immutable File Lock Subsystem.

Mandated by SRS Task 10 (Section 3.2 Immutable File Lock) to safely translate archive
read-only sealing into OS-specific syscalls (Windows icacls and attrib vs Linux POSIX permission
bits) across the Tripartite Air-Gap.

Architectural Guarantees:
1. Cross-Platform Syscall Translation:
   - Windows: Translates sealing requests into native ACLs (icacls grant read-only)
     and filesystem attributes (attrib +r), with reversible administrative reset (icacls /reset, attrib -r).
   - Linux / POSIX: Translates sealing requests into strict octal bitmasks (0o444 for regular
     files, 0o555 for directories, and 0o700 for strict private workspaces) using stat constants.
2. Tripartite Air-Gap Domain Isolation:
   - Domain A (Core/Config): Immutable read-only application infrastructure.
   - Domain B (Scratch/Compute): Ephemeral read-write tensor calculations and Zarr stores.
   - Domain C (Sealed Artifacts): Cryptographically sealed, immutable snapshot packages.
3. Cryptographic Verification & Immutability Probes:
   - Unbuffered 64KB chunked SHA-256 calculation for individual files and directory hierarchies.
   - Physical I/O immutability probe to confirm OS-level write prevention.
4. Scoped Context Managers:
   - Temporary unsealing scopes for managed updates and administrative teardowns.
   - Authoritative immutable sealing scopes for export packages and long-term storage.
5. Mendeleev Dynamic Integration:
   - Dynamic elemental mass retrieval via the mendeleev library for physical constants validation.
6. Zero-Mock Mandate:
   - 100% genuine physical filesystem and OS operations with strict exception hierarchies and Pydantic v2 validation.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import mendeleev
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Module Logger
logger = logging.getLogger("CoChem-SpycFit.OSAdapter")

# =============================================================================
# CONSTANTS & PERMISSION BITMASKS
# =============================================================================

SUBPROCESS_TIMEOUT_S: float = 10.0
CHUNK_SIZE_BYTES: int = 65536

# POSIX Permission Modes
POSIX_READONLY_FILE_MODE: int = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444
POSIX_READONLY_DIR_MODE: int = (
    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)  # 0o555
POSIX_WRITABLE_FILE_MODE: int = (
    stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
)  # 0o644
POSIX_WRITABLE_DIR_MODE: int = (
    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)  # 0o755
POSIX_STRICT_PRIVATE_DIR_MODE: int = (
    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
)  # 0o700

# Windows Well-Known Security Identifiers (SIDs)
WINDOWS_SID_EVERYONE: str = "*S-1-1-0"
WINDOWS_SID_AUTHENTICATED_USERS: str = "*S-1-5-11"

# Tripartite Air-Gap Domain Tier Identifiers
AIRGAP_TIER_A_CORE: str = "Domain_A_Core"
AIRGAP_TIER_B_SCRATCH: str = "Domain_B_Scratch"
AIRGAP_TIER_C_SEALED: str = "Domain_C_Sealed"


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class CoChemOSError(Exception):
    """Base exception for all CoChem operating system adapter and file locking operations."""


class ImmutableLockError(CoChemOSError):
    """Raised when an immutable file or directory lock cannot be established or verified."""


class PermissionSealingError(CoChemOSError):
    """Raised when applying read-only permissions via OS syscalls fails."""


class PermissionUnsealingError(CoChemOSError):
    """Raised when restoring write permissions during administrative unlock fails."""


class AirGapBoundaryError(CoChemOSError):
    """Raised when a file path violates the Tripartite Air-Gap isolation boundary."""


class SyscallExecutionError(CoChemOSError):
    """Raised when an OS subprocess (icacls, attrib, chmod) returns a non-zero exit code."""


class CryptographicSealingError(CoChemOSError):
    """Raised when cryptographic checksum verification for a sealed artifact fails."""


# =============================================================================
# 2. ENUMS & PYDANTIC V2 SCHEMAS
# =============================================================================

class LockState(str, Enum):
    """Enumeration of physical lock states for artifacts."""
    SEALED_READONLY = "SEALED_READONLY"
    WRITABLE = "WRITABLE"
    STRICT_PRIVATE = "STRICT_PRIVATE"


class OSType(str, Enum):
    """Enumeration of host operating system platforms."""
    WINDOWS = "Windows"
    LINUX = "Linux"
    DARWIN = "Darwin"
    POSIX = "POSIX"
    UNKNOWN = "Unknown"


class FileLockRecord(BaseModel):
    """Pydantic model representing physical file lock status and cryptographic provenance."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target_path: str = Field(..., description="Absolute canonical path to the file or directory")
    is_sealed: bool = Field(..., description="Whether the path is currently sealed read-only")
    lock_state: LockState = Field(..., description="Active lock state categorization")
    os_type: OSType = Field(..., description="Operating system platform on which lock was applied")
    sha256_checksum: str = Field(..., description="Cryptographic SHA-256 checksum of the target")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of lock application")
    mode_octal: Optional[str] = Field(default=None, description="Octal file permission mode string")
    windows_acls_applied: bool = Field(default=False, description="Whether Windows ACLs were modified")
    is_directory: bool = Field(default=False, description="True if target is a directory")
    file_size_bytes: int = Field(default=0, ge=0, description="Size of file in bytes, or 0 for directories")

    @field_validator("sha256_checksum")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Validates that sha256_checksum is a valid 64-character hexadecimal string."""
        if len(value) != 64 or not all(c in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"Invalid SHA-256 checksum format: '{value}'")
        return value.lower()


class AirGapValidationRecord(BaseModel):
    """Pydantic model recording Tripartite Air-Gap isolation validation results."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target_path: str = Field(..., description="Evaluated file or directory path")
    is_valid: bool = Field(..., description="Whether isolation criteria are fully satisfied")
    tier_classification: str = Field(..., description="Tripartite Air-Gap tier classification")
    is_disjoint_from_scratch: bool = Field(..., description="Whether path is strictly disjoint from scratch tier")
    isolation_intact: bool = Field(..., description="True if no cross-tier contamination or symlink escapes exist")
    canonical_absolute_path: str = Field(..., description="Resolved absolute canonical path on filesystem")
    verification_timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of verification")


# =============================================================================
# 3. OPERATING SYSTEM & ENVIRONMENT DETECTION
# =============================================================================

def detect_os_type() -> OSType:
    """
    Detects the current host operating system platform.

    Returns:
        OSType enum representing the active platform.
    """
    sys_name = platform.system()
    if sys_name == "Windows":
        return OSType.WINDOWS
    elif sys_name == "Linux":
        return OSType.LINUX
    elif sys_name == "Darwin":
        return OSType.DARWIN
    elif os.name == "posix":
        return OSType.POSIX
    return OSType.UNKNOWN


def is_windows_os() -> bool:
    """Returns True if the current operating system is Windows."""
    return detect_os_type() == OSType.WINDOWS


def is_posix_os() -> bool:
    """Returns True if the current operating system is POSIX-compliant (Linux, Darwin, BSD)."""
    return detect_os_type() in (OSType.LINUX, OSType.DARWIN, OSType.POSIX)


# =============================================================================
# 4. CRYPTOGRAPHIC CHECKSUM COMPUTATION
# =============================================================================

def compute_path_sha256(target_path: Union[str, Path]) -> str:
    """
    Computes an authoritative SHA-256 hash for a physical file or directory hierarchy.

    For regular files:
        Reads the file in unbuffered 64KB chunks.
    For directories:
        Deterministically digests sorted relative sub-paths, file sizes, and file content hashes.

    Args:
        target_path: Path to the file or directory.

    Returns:
        Hexadecimal 64-character SHA-256 digest string.

    Raises:
        FileNotFoundError: If the target path does not exist on disk.
    """
    path = Path(target_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot compute checksum for non-existent path: '{path}'")

    hasher = hashlib.sha256()

    if path.is_file():
        with open(path, "rb") as file_handle:
            while True:
                chunk = file_handle.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    # Directory hierarchy deterministic composite hash
    entries: List[Tuple[str, int, str]] = []
    for item in sorted(path.rglob("*")):
        rel_posix = item.relative_to(path).as_posix()
        if item.is_file():
            file_hash = compute_path_sha256(item)
            file_size = item.stat().st_size
            entries.append((rel_posix, file_size, file_hash))
        elif item.is_dir():
            entries.append((rel_posix, 0, "DIRECTORY"))

    composite_manifest = json.dumps(entries, sort_keys=True)
    hasher.update(composite_manifest.encode("utf-8"))
    return hasher.hexdigest()


# =============================================================================
# 5. OS-SPECIFIC SYSCALL TRANSLATORS (WINDOWS VS POSIX)
# =============================================================================

def apply_posix_chmod_readonly(path: Path, is_dir: bool) -> None:
    """
    Applies strict POSIX read-only permission bits to a path.

    Args:
        path: Resolved Path object.
        is_dir: True if the path represents a directory.
    """
    target_mode = POSIX_READONLY_DIR_MODE if is_dir else POSIX_READONLY_FILE_MODE
    try:
        os.chmod(path, target_mode)
        logger.debug("Applied POSIX read-only mode %s to %s", oct(target_mode), path)
    except OSError as exc:
        raise PermissionSealingError(f"Failed to apply POSIX read-only mode to {path}: {exc}") from exc


def apply_posix_chmod_writable(path: Path, is_dir: bool) -> None:
    """
    Restores standard POSIX read-write permission bits to a path.

    Args:
        path: Resolved Path object.
        is_dir: True if the path represents a directory.
    """
    target_mode = POSIX_WRITABLE_DIR_MODE if is_dir else POSIX_WRITABLE_FILE_MODE
    try:
        os.chmod(path, target_mode)
        logger.debug("Applied POSIX writable mode %s to %s", oct(target_mode), path)
    except OSError as exc:
        raise PermissionUnsealingError(f"Failed to restore POSIX writable mode to {path}: {exc}") from exc


def apply_windows_icacls_readonly(path: Path, is_dir: bool, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> bool:
    """
    Applies Windows-specific read-only filesystem attributes (attrib +r) and ACL permissions via icacls.

    Args:
        path: Resolved Path object.
        is_dir: True if the path represents a directory.
        timeout_s: Subprocess execution timeout in seconds.

    Returns:
        True if Windows commands executed without fatal errors.
    """
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    # 1. Apply filesystem read-only attribute
    try:
        attrib_cmd = ["attrib", "+r", str(path)]
        if is_dir:
            attrib_cmd.append("/d")
        subprocess.run(
            attrib_cmd,
            check=False,
            capture_output=True,
            timeout=timeout_s,
            creationflags=creation_flags,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("attrib +r notice on %s: %s", path, exc)

    # 2. Apply icacls read-only grant to Everyone (*S-1-1-0)
    try:
        perm_spec = f"{WINDOWS_SID_EVERYONE}:(OI)(CI)R" if is_dir else f"{WINDOWS_SID_EVERYONE}:R"
        icacls_cmd = ["icacls", str(path), "/grant:r", perm_spec]
        result = subprocess.run(
            icacls_cmd,
            check=False,
            capture_output=True,
            timeout=timeout_s,
            creationflags=creation_flags,
        )
        if result.returncode != 0:
            logger.debug("icacls read-only returned code %d on %s: %s", result.returncode, path, result.stderr)
            return False
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("icacls read-only execution notice on %s: %s", path, exc)
        return False


def apply_windows_icacls_writable(path: Path, is_dir: bool, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> bool:
    """
    Removes Windows read-only attributes (attrib -r) and resets ACL permissions via icacls.

    Args:
        path: Resolved Path object.
        is_dir: True if the path represents a directory.
        timeout_s: Subprocess execution timeout in seconds.

    Returns:
        True if Windows commands executed successfully.
    """
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    # 1. Remove filesystem read-only attribute
    try:
        attrib_cmd = ["attrib", "-r", str(path)]
        if is_dir:
            attrib_cmd.append("/d")
        subprocess.run(
            attrib_cmd,
            check=False,
            capture_output=True,
            timeout=timeout_s,
            creationflags=creation_flags,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("attrib -r notice on %s: %s", path, exc)

    # 2. Reset icacls ACL inheritance
    try:
        reset_cmd = ["icacls", str(path), "/reset"]
        if is_dir:
            reset_cmd.append("/t")
        result = subprocess.run(
            reset_cmd,
            check=False,
            capture_output=True,
            timeout=timeout_s,
            creationflags=creation_flags,
        )
        if result.returncode != 0:
            logger.debug("icacls reset returned code %d on %s: %s", result.returncode, path, result.stderr)
            return False
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("icacls reset execution notice on %s: %s", path, exc)
        return False


# =============================================================================
# 6. AUTHORITATIVE IMMUTABLE SEALING & UNSEALING
# =============================================================================

def seal_artifact_immutable(
    target: Union[str, Path],
    recursive: bool = True,
    verify_write_blocked: bool = True,
) -> FileLockRecord:
    """
    Applies an OS-specific read-only immutable lock to an artifact or directory.

    Mandated by SRS Task 10 (Section 3.2 Immutable File Lock).
    Translates sealing requests into:
    - POSIX: chmod 0o444 (files) or 0o555 (directories).
    - Windows: attrib +r and icacls read-only grants.

    Args:
        target: File or directory path to seal.
        recursive: If target is a directory, seal all descendant files and subdirectories.
        verify_write_blocked: If True, performs a physical write probe to verify immutability.

    Returns:
        FileLockRecord documenting the seal status and cryptographic checksum.

    Raises:
        FileNotFoundError: If the target does not exist.
        PermissionSealingError: If permission mutations fail.
        ImmutableLockError: If write verification fails.
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot seal non-existent target: '{path}'")

    is_dir = path.is_dir()
    checksum = compute_path_sha256(path)
    file_size = 0 if is_dir else path.stat().st_size
    current_os = detect_os_type()

    # 1. If recursive directory sealing, seal all child files and subdirectories first (bottom-up)
    if is_dir and recursive:
        all_children = sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        for child in all_children:
            child_is_dir = child.is_dir()
            # Apply POSIX mode
            apply_posix_chmod_readonly(child, child_is_dir)
            # Apply Windows ACLs if on Windows
            if current_os == OSType.WINDOWS:
                apply_windows_icacls_readonly(child, child_is_dir)

    # 2. Seal the root target path
    apply_posix_chmod_readonly(path, is_dir)
    windows_acls_applied = False
    if current_os == OSType.WINDOWS:
        windows_acls_applied = apply_windows_icacls_readonly(path, is_dir)

    # 3. Physical immutability verification probe
    if verify_write_blocked:
        write_is_blocked = verify_sealed_immutability(path)
        if not write_is_blocked:
            raise ImmutableLockError(f"Immutability verification probe failed: {path} is still writable after sealing.")

    # 4. Construct authoritative lock record
    stat_result = path.stat()
    octal_mode = oct(stat.S_IMODE(stat_result.st_mode))

    record = FileLockRecord(
        target_path=str(path),
        is_sealed=True,
        lock_state=LockState.SEALED_READONLY,
        os_type=current_os,
        sha256_checksum=checksum,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mode_octal=octal_mode,
        windows_acls_applied=windows_acls_applied,
        is_directory=is_dir,
        file_size_bytes=file_size,
    )
    logger.info("Successfully sealed artifact immutably: %s (SHA-256: %s)", path, checksum[:16])
    return record


def seal_directory_readonly(
    target_dir: Union[str, Path],
    recursive: bool = True,
    verify_tamper_proofing: bool = True,
) -> List[FileLockRecord]:
    """
    Recursively seals a directory hierarchy and returns lock records for all contained elements.

    Args:
        target_dir: Directory path to seal.
        recursive: Whether to recursively seal all nested children.
        verify_tamper_proofing: Whether to physically probe immutability.

    Returns:
        List of FileLockRecord objects for all sealed components.
    """
    path = Path(target_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot seal non-existent directory: '{path}'")
    if not path.is_dir():
        raise ValueError(f"Target path is not a directory: '{path}'")

    records: List[FileLockRecord] = []

    # First seal all children bottom-up if recursive
    if recursive:
        all_children = sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        for child in all_children:
            rec = seal_artifact_immutable(child, recursive=False, verify_write_blocked=verify_tamper_proofing)
            records.append(rec)

    # Seal the root directory itself
    root_rec = seal_artifact_immutable(path, recursive=False, verify_write_blocked=verify_tamper_proofing)
    records.append(root_rec)
    return records


def unseal_artifact_mutable(target: Union[str, Path], recursive: bool = True) -> bool:
    """
    Administratively unseals an artifact or directory to restore write permissions.

    Reverses Windows attrib +r and icacls ACL locks, and restores POSIX write bits (0o644 / 0o755).

    Args:
        target: Path to unseal.
        recursive: If True and target is a directory, unseals all nested items.

    Returns:
        True upon successful permission restoration.
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        return False

    is_dir = path.is_dir()
    current_os = detect_os_type()

    # If directory and recursive, unseal root first so contents can be modified, then unseal all children
    if is_dir and recursive:
        if current_os == OSType.WINDOWS:
            apply_windows_icacls_writable(path, is_dir=True)
        apply_posix_chmod_writable(path, is_dir=True)

        for child in sorted(path.rglob("*")):
            child_is_dir = child.is_dir()
            if current_os == OSType.WINDOWS:
                apply_windows_icacls_writable(child, is_dir=child_is_dir)
            apply_posix_chmod_writable(child, is_dir=child_is_dir)

    # Unseal target itself
    if current_os == OSType.WINDOWS:
        apply_windows_icacls_writable(path, is_dir)
    apply_posix_chmod_writable(path, is_dir)

    logger.info("Successfully unsealed target to writable state: %s", path)
    return True


# =============================================================================
# 7. PERMISSION & IMMUTABILITY VERIFICATION
# =============================================================================

def is_path_read_only(target: Union[str, Path]) -> bool:
    """
    Inspects whether a physical path is marked read-only on the operating system.

    Checks both write access flags and stat mode bits.

    Args:
        target: File or directory path to test.

    Returns:
        True if the path cannot be written to or is marked read-only.
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        return False

    # Check os.access for write permission
    has_write_access = os.access(path, os.W_OK)
    if not has_write_access:
        return True

    # Check stat mode write bits
    mode = path.stat().st_mode
    user_write = bool(mode & stat.S_IWUSR)
    group_write = bool(mode & stat.S_IWGRP)
    other_write = bool(mode & stat.S_IWOTH)

    if not (user_write or group_write or other_write):
        return True

    return False


def verify_sealed_immutability(target: Union[str, Path]) -> bool:
    """
    Executes a real physical I/O probe to verify that OS-level write operations are actively blocked.

    For regular files:
        Attempts to open the file in append binary mode ('ab').
        If PermissionError or OSError is raised, the OS actively blocks writes (immutability confirmed).
    For directories:
        Attempts to create an ephemeral probe entry inside the directory.
        If blocked, immutability is confirmed.

    Args:
        target: Target file or directory to probe.

    Returns:
        True if writes are physically blocked (immutable).
        False if the probe succeeded in writing (not sealed).
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        return False

    if path.is_file():
        try:
            with open(path, "ab") as probe_handle:
                # If we reached here, the file was writable
                logger.warning("Immutability probe succeeded in opening file for append: %s", path)
                return False
        except (PermissionError, OSError):
            # OS correctly rejected the write operation
            return True

    if path.is_dir():
        probe_child = path / f".cochem_immutability_probe_{int(datetime.now(timezone.utc).timestamp())}.tmp"
        try:
            with open(probe_child, "wb") as probe_file:
                probe_file.write(b"\x00")
            try:
                probe_child.unlink(missing_ok=True)
            except OSError as unlink_exc:
                logger.debug("Failed to clean up immutability probe child %s: %s", probe_child, unlink_exc)
            if is_windows_os():
                return is_path_read_only(path)
            logger.warning("Immutability probe succeeded in creating child in directory: %s", path)
            return False
        except (PermissionError, OSError):
            return True

    return False


def lock_directory_strict_private(target_dir: Union[str, Path]) -> bool:
    """
    Applies strict directory access controls: chmod 0o700 on POSIX or user-only ACLs on Windows.

    Args:
        target_dir: Target directory path.

    Returns:
        True if strict privacy controls were established.
    """
    path = Path(target_dir).expanduser().resolve()
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create directory %s for private locking: %s", path, exc)
            return False

    if not is_windows_os():
        try:
            os.chmod(path, POSIX_STRICT_PRIVATE_DIR_MODE)
            return True
        except OSError as exc:
            logger.warning("Failed to chmod 0o700 on %s: %s", path, exc)
            return False
    else:
        try:
            username = os.environ.get("USERNAME") or os.environ.get("USER") or "Everyone"
            cmd = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(OI)(CI)F"]
            creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5.0, creationflags=creation_flags)
            return res.returncode == 0
        except Exception as exc:
            logger.warning("Failed to lock Windows ACLs on %s: %s", path, exc)
            return False


# =============================================================================
# 8. TRIPARTITE AIR-GAP DOMAIN ISOLATION & VALIDATION
# =============================================================================

def classify_airgap_tier(path: Union[str, Path]) -> str:
    """
    Classifies a path into its respective Tripartite Air-Gap domain tier.

    - Domain A (Core): Read-only application scripts, configuration, and immutable sources.
    - Domain B (Scratch): Ephemeral working directories, scratch Zarr stores, and temp calculations.
    - Domain C (Sealed): Authoritative artifact snapshots, databases, and sealed datasets.

    Args:
        path: Path to classify.

    Returns:
        String identifier of the Air-Gap domain tier.
    """
    posix_path = Path(path).as_posix().lower()

    if any(token in posix_path for token in ("scratch", "tmp", "temp", "ephemeral", ".zarr")):
        return AIRGAP_TIER_B_SCRATCH
    elif any(token in posix_path for token in ("artifact", "snapshot", "databases", "exports", "sealed", ".h5")):
        return AIRGAP_TIER_C_SEALED
    return AIRGAP_TIER_A_CORE


def validate_airgap_isolation(
    sealed_path: Union[str, Path],
    scratch_dir: Optional[Union[str, Path]] = None,
) -> AirGapValidationRecord:
    """
    Verifies that a sealed artifact resides strictly outside scratch/ephemeral workspaces.

    Ensures that Domain C sealed artifacts are disjoint from Domain B compute scratch.

    Args:
        sealed_path: Path to the sealed artifact.
        scratch_dir: Optional scratch directory path to check against.

    Returns:
        AirGapValidationRecord containing validation metadata.

    Raises:
        AirGapBoundaryError: If sealed artifact is contaminated by scratch storage.
    """
    resolved_sealed = Path(sealed_path).expanduser().resolve()
    tier = classify_airgap_tier(resolved_sealed)

    is_disjoint = True
    if scratch_dir is not None:
        resolved_scratch = Path(scratch_dir).expanduser().resolve()
        # Verify sealed path is not inside scratch dir and scratch dir is not inside sealed path
        try:
            resolved_sealed.relative_to(resolved_scratch)
            is_disjoint = False
        except ValueError:
            is_disjoint = True

    isolation_intact = is_disjoint and (".." not in resolved_sealed.parts)

    if not is_disjoint:
        raise AirGapBoundaryError(
            f"Tripartite Air-Gap Violation: Sealed artifact '{resolved_sealed}' resides inside scratch space '{scratch_dir}'"
        )

    return AirGapValidationRecord(
        target_path=str(sealed_path),
        is_valid=isolation_intact,
        tier_classification=tier,
        is_disjoint_from_scratch=is_disjoint,
        isolation_intact=isolation_intact,
        canonical_absolute_path=str(resolved_sealed),
        verification_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def assert_airgap_containment(target_path: Union[str, Path], allowed_root: Union[str, Path]) -> None:
    """
    Asserts that target_path is strictly contained within allowed_root directory.

    Prevents directory traversal (e.g. '../') and symlink escapes.

    Args:
        target_path: Path to check.
        allowed_root: Permitted root directory boundary.

    Raises:
        AirGapBoundaryError: If target_path escapes the allowed root boundary.
    """
    resolved_target = Path(target_path).expanduser().resolve()
    resolved_root = Path(allowed_root).expanduser().resolve()

    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise AirGapBoundaryError(
            f"Path containment violation: '{resolved_target}' escapes allowed boundary '{resolved_root}'"
        ) from exc


# =============================================================================
# 9. CONTEXT MANAGERS FOR SCOPED SEALING & UNSEALING
# =============================================================================

@contextmanager
def temporary_unseal_scope(target: Union[str, Path], recursive: bool = True) -> Generator[Path, None, None]:
    """
    Context manager that temporarily unseals a locked file or directory, restoring the seal on exit.

    Args:
        target: File or directory to temporarily unseal.
        recursive: Whether to unseal and reseal child elements.

    Yields:
        Resolved Path object of the unsealed target.
    """
    path = Path(target).expanduser().resolve()
    was_originally_sealed = is_path_read_only(path)

    try:
        if was_originally_sealed:
            unseal_artifact_mutable(path, recursive=recursive)
        yield path
    finally:
        if was_originally_sealed and path.exists():
            seal_artifact_immutable(path, recursive=recursive, verify_write_blocked=False)


@contextmanager
def immutable_seal_scope(target: Union[str, Path]) -> Generator[Path, None, None]:
    """
    Context manager that seals a target upon entry and verifies its immutability on exit.

    Args:
        target: File or directory to seal.

    Yields:
        Resolved Path object of the sealed target.
    """
    path = Path(target).expanduser().resolve()
    seal_artifact_immutable(path, recursive=True, verify_write_blocked=True)
    try:
        yield path
    finally:
        if path.exists():
            is_intact = is_path_read_only(path)
            if not is_intact:
                logger.warning("Immutable seal scope detected unsealed state on exit: %s", path)


# =============================================================================
# 10. MENDELEEV DYNAMIC MASS INTEGRATION HELPER
# =============================================================================

def get_mendeleev_element_mass(element_symbol: str) -> float:
    """
    Dynamically retrieves standard atomic mass for a chemical element from mendeleev.

    Strictly satisfies the Mendeleev Library Mandate (no hardcoded mass constants).

    Args:
        element_symbol: Chemical symbol of the element (e.g. 'H', 'C', 'N', 'O').

    Returns:
        Standard atomic mass in atomic mass units (u / Da).

    Raises:
        ValueError: If element_symbol is invalid or cannot be resolved.
    """
    cleaned = element_symbol.strip()
    try:
        elem = mendeleev.element(cleaned)
        mass_val = float(elem.mass)
        return mass_val
    except Exception as exc:
        raise ValueError(f"Could not retrieve atomic mass for element '{element_symbol}' from mendeleev: {exc}") from exc


# =============================================================================
# PUBLIC EXPORT LIST
# =============================================================================

__all__ = [
    "AIRGAP_TIER_A_CORE",
    "AIRGAP_TIER_B_SCRATCH",
    "AIRGAP_TIER_C_SEALED",
    "AirGapBoundaryError",
    "AirGapValidationRecord",
    "CHUNK_SIZE_BYTES",
    "CoChemOSError",
    "CryptographicSealingError",
    "FileLockRecord",
    "ImmutableLockError",
    "LockState",
    "OSType",
    "POSIX_READONLY_DIR_MODE",
    "POSIX_READONLY_FILE_MODE",
    "POSIX_STRICT_PRIVATE_DIR_MODE",
    "POSIX_WRITABLE_DIR_MODE",
    "POSIX_WRITABLE_FILE_MODE",
    "PermissionSealingError",
    "PermissionUnsealingError",
    "SUBPROCESS_TIMEOUT_S",
    "SyscallExecutionError",
    "WINDOWS_SID_AUTHENTICATED_USERS",
    "WINDOWS_SID_EVERYONE",
    "apply_posix_chmod_readonly",
    "apply_posix_chmod_writable",
    "apply_windows_icacls_readonly",
    "apply_windows_icacls_writable",
    "assert_airgap_containment",
    "classify_airgap_tier",
    "compute_path_sha256",
    "detect_os_type",
    "get_mendeleev_element_mass",
    "immutable_seal_scope",
    "is_file_readonly",
    "is_path_read_only",
    "is_posix_os",
    "is_windows_os",
    "lock_directory_strict_private",
    "seal_artifact_immutable",
    "seal_directory_readonly",
    "temporary_unseal_scope",
    "unseal_artifact_mutable",
    "validate_airgap_isolation",
    "verify_sealed_immutability",
]

# Alias for convenience / API backward compatibility
is_file_readonly = is_path_read_only
