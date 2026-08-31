"""
CoChem Tripartite Air-Gap Topology & Core Engine Domain Exceptions.

Provenance & Specifications:
- Method Matrix [M]: Tripartite Air-Gap Topology (T_code, T_data, T_art) boundary invariants.
- Concurrency Protocol [C]: OS-level Single-Writer Multiple-Reader (SWMR) file locking.
- Telemetry Integrity [T]: Non-blocking ring-buffered cryptographic event logging.
- Chemical Invariants [D]: Dynamic Mendeleev library element and isotope mass verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


class CoChemTopologyException(Exception):
    """Base domain exception for all CoChem topology, storage, and engine errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class AirGapPartitionViolationError(CoChemTopologyException):
    """
    Raised when an operation violates air-gap boundary rules.

    Examples:
    - Attempting write or modification on read-only partitions (T_code, T_data).
    - Attempting path traversal (e.g. '../') outside assigned partition roots.
    - Attempting unauthorized cross-partition filesystem modifications.
    """

    def __init__(
        self,
        message: str,
        attempted_path: Optional[Union[Path, str]] = None,
        partition_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if attempted_path is not None:
            merged_details["attempted_path"] = str(attempted_path)
        if partition_type is not None:
            merged_details["partition_type"] = partition_type

        super().__init__(message=message, details=merged_details)
        self.attempted_path = Path(attempted_path) if attempted_path is not None else None
        self.partition_type = partition_type


class PartitionMountError(CoChemTopologyException):
    """
    Raised when a required partition directory cannot be mounted, is missing,
    or has invalid filesystem access permissions.
    """

    def __init__(
        self,
        message: str,
        partition_type: Optional[str] = None,
        path: Optional[Union[Path, str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if partition_type is not None:
            merged_details["partition_type"] = partition_type
        if path is not None:
            merged_details["path"] = str(path)

        super().__init__(message=message, details=merged_details)
        self.partition_type = partition_type
        self.path = Path(path) if path is not None else None


class SWMRFileLockTimeoutError(CoChemTopologyException):
    """
    Raised when a Single-Writer atomic lock cannot be acquired within
    the designated timeout threshold (tau_lock).
    """

    def __init__(
        self,
        message: str,
        lock_filepath: Optional[Union[Path, str]] = None,
        target_filepath: Optional[Union[Path, str]] = None,
        timeout_seconds: Optional[float] = None,
        lock_owner_pid: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if lock_filepath is not None:
            merged_details["lock_filepath"] = str(lock_filepath)
        if target_filepath is not None:
            merged_details["target_filepath"] = str(target_filepath)
        if timeout_seconds is not None:
            merged_details["timeout_seconds"] = timeout_seconds
        if lock_owner_pid is not None:
            merged_details["lock_owner_pid"] = lock_owner_pid

        super().__init__(message=message, details=merged_details)
        self.lock_filepath = Path(lock_filepath) if lock_filepath is not None else None
        self.target_filepath = Path(target_filepath) if target_filepath is not None else None
        self.timeout_seconds = timeout_seconds
        self.lock_owner_pid = lock_owner_pid


class SWMRCorruptionError(CoChemTopologyException):
    """
    Raised when an HDF5 dataset or container in T_art violates SWMR consistency headers,
    contains broken Merkle integrity blocks, or experiences unrecoverable read/write errors.
    """

    def __init__(
        self,
        message: str,
        filepath: Optional[Union[Path, str]] = None,
        dataset_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if filepath is not None:
            merged_details["filepath"] = str(filepath)
        if dataset_name is not None:
            merged_details["dataset_name"] = dataset_name

        super().__init__(message=message, details=merged_details)
        self.filepath = Path(filepath) if filepath is not None else None
        self.dataset_name = dataset_name


class TelemetryQueueOverflowError(CoChemTopologyException):
    """
    Raised when the non-blocking ring-buffer capacity is exceeded and overflow
    mitigation fails during critical telemetry dispatch.
    """

    def __init__(
        self,
        message: str,
        queue_size: Optional[int] = None,
        dropped_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if queue_size is not None:
            merged_details["queue_size"] = queue_size
        if dropped_count is not None:
            merged_details["dropped_count"] = dropped_count

        super().__init__(message=message, details=merged_details)
        self.queue_size = queue_size
        self.dropped_count = dropped_count


class MendeleevInvariantError(CoChemTopologyException):
    """
    Raised when dynamic chemical element or isotope resolution fails,
    or when unphysical masses or invalid symbols are encountered.
    """

    def __init__(
        self,
        message: str,
        symbol_or_query: Optional[Union[str, int]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if symbol_or_query is not None:
            merged_details["symbol_or_query"] = symbol_or_query

        super().__init__(message=message, details=merged_details)
        self.symbol_or_query = symbol_or_query
