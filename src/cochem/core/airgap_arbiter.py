"""
Tripartite Air-Gap Path & Permission Arbiter.

Provenance & Specifications:
- Method Matrix [M]: Tripartite Partition Air-Gap Topology (T_code, T_data, T_art).
- Partition Invariants: T_code and T_data are strictly READ_ONLY; T_art is strictly READ_WRITE.
- Security Policy: Canonical path resolution to prevent directory traversal and cross-partition pollution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from cochem.core.topology_exceptions import (
    AirGapPartitionViolationError,
    PartitionMountError,
)
from cochem.core.topology_models import (
    DEFAULT_ALLOWED_SUBDIRECTORIES,
    PartitionDescriptor,
    PartitionPermission,
    PartitionType,
)


class AirGapArbiter:
    """
    Arbiter for enforcing tripartite air-gap boundaries, access control,
    and sandbox containment across T_code, T_data, and T_art partitions.
    """

    def __init__(
        self,
        code_root: Union[Path, str],
        data_root: Union[Path, str],
        artifact_root: Union[Path, str],
        auto_create_artifact_subdirs: bool = True,
    ) -> None:
        self.code_root = Path(code_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()

        self._descriptors: Dict[PartitionType, PartitionDescriptor] = {}
        self._init_descriptors()

        if auto_create_artifact_subdirs:
            self.create_artifact_subdirectories()

    def _init_descriptors(self) -> None:
        """Initialize frozen partition descriptors."""
        self._descriptors["CODE"] = PartitionDescriptor(
            partition_type="CODE",
            root_path=self.code_root,
            permission="READ_ONLY",
            allowed_subdirectories=list(DEFAULT_ALLOWED_SUBDIRECTORIES["CODE"]),
            is_air_gapped=True,
        )
        self._descriptors["DATA"] = PartitionDescriptor(
            partition_type="DATA",
            root_path=self.data_root,
            permission="READ_ONLY",
            allowed_subdirectories=list(DEFAULT_ALLOWED_SUBDIRECTORIES["DATA"]),
            is_air_gapped=True,
        )
        self._descriptors["ARTIFACT"] = PartitionDescriptor(
            partition_type="ARTIFACT",
            root_path=self.artifact_root,
            permission="READ_WRITE",
            allowed_subdirectories=list(DEFAULT_ALLOWED_SUBDIRECTORIES["ARTIFACT"]),
            is_air_gapped=True,
        )

    @property
    def code_descriptor(self) -> PartitionDescriptor:
        """Descriptor for T_code partition."""
        return self._descriptors["CODE"]

    @property
    def data_descriptor(self) -> PartitionDescriptor:
        """Descriptor for T_data partition."""
        return self._descriptors["DATA"]

    @property
    def artifact_descriptor(self) -> PartitionDescriptor:
        """Descriptor for T_art partition."""
        return self._descriptors["ARTIFACT"]

    def get_descriptor(self, partition_type: Union[str, PartitionType]) -> PartitionDescriptor:
        """Retrieve partition descriptor by type."""
        ptype = str(partition_type).upper()
        if ptype not in self._descriptors:
            raise PartitionMountError(
                f"Unknown partition type '{partition_type}'. Must be one of CODE, DATA, ARTIFACT.",
                partition_type=ptype,
            )
        return self._descriptors[ptype]  # type: ignore[index]

    def create_artifact_subdirectories(self) -> None:
        """Create required standard subdirectories in artifact root."""
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        for subdir in self.artifact_descriptor.allowed_subdirectories:
            (self.artifact_root / subdir).mkdir(parents=True, exist_ok=True)

    def verify_partitions(self) -> bool:
        """
        Verify physical existence and permissions of all mounted partitions.

        Returns:
            True if all partitions exist and are accessible.

        Raises:
            PartitionMountError: If any partition is missing or inaccessible.
        """
        for ptype, desc in self._descriptors.items():
            if not desc.root_path.exists():
                raise PartitionMountError(
                    f"Partition root for '{ptype}' does not exist: '{desc.root_path}'",
                    partition_type=ptype,
                    path=desc.root_path,
                )
            if not desc.root_path.is_dir():
                raise PartitionMountError(
                    f"Partition root for '{ptype}' is not a directory: '{desc.root_path}'",
                    partition_type=ptype,
                    path=desc.root_path,
                )
        return True

    def is_path_in_partition(self, path: Union[Path, str], partition_type: PartitionType) -> bool:
        """Check whether a path resides within the specified partition."""
        resolved = Path(path).resolve()
        root = self._descriptors[partition_type].root_path
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            return False

    def identify_partition(self, path: Union[Path, str]) -> Optional[PartitionDescriptor]:
        """Identify which partition contains the given path, if any."""
        resolved = Path(path).resolve()
        for desc in self._descriptors.values():
            try:
                resolved.relative_to(desc.root_path)
                return desc
            except ValueError:
                continue
        return None

    def validate_read_access(self, path: Union[Path, str]) -> PartitionDescriptor:
        """
        Validate read access for a path across any air-gapped partition.

        Returns:
            PartitionDescriptor of the containing partition.

        Raises:
            AirGapPartitionViolationError: If path is outside all partitions.
        """
        resolved = Path(path).resolve()
        desc = self.identify_partition(resolved)
        if desc is None:
            raise AirGapPartitionViolationError(
                f"Read access denied: path '{path}' resides outside registered air-gap partitions "
                f"(Code: {self.code_root}, Data: {self.data_root}, Artifact: {self.artifact_root}).",
                attempted_path=resolved,
            )
        return desc

    def validate_write_access(self, path: Union[Path, str]) -> PartitionDescriptor:
        """
        Validate write access for a path.
        Enforces that write operations are strictly restricted to T_art.

        Returns:
            PartitionDescriptor for ARTIFACT.

        Raises:
            AirGapPartitionViolationError: If targeting T_code, T_data, or outside partitions.
        """
        resolved = Path(path).resolve()
        desc = self.identify_partition(resolved)

        if desc is None:
            raise AirGapPartitionViolationError(
                f"Write access denied: path '{path}' resides outside registered air-gap partitions.",
                attempted_path=resolved,
            )

        if desc.permission == "READ_ONLY":
            raise AirGapPartitionViolationError(
                f"Write access violation: attempted mutation on READ_ONLY partition '{desc.partition_type}' "
                f"at path '{resolved}'. Writes are strictly restricted to ARTIFACT (T_art).",
                attempted_path=resolved,
                partition_type=desc.partition_type,
            )

        return desc

    def get_artifact_path(self, relative_subpath: Union[Path, str]) -> Path:
        """
        Resolve a relative subpath within T_art sandbox.
        Prevents directory traversal attacks ('../') escaping T_art.

        Returns:
            Resolved absolute Path inside T_art.

        Raises:
            AirGapPartitionViolationError: If traversal escapes T_art.
        """
        rel = Path(relative_subpath)
        if rel.is_absolute():
            resolved = rel.resolve()
        else:
            resolved = (self.artifact_root / rel).resolve()

        try:
            resolved.relative_to(self.artifact_root)
        except ValueError:
            raise AirGapPartitionViolationError(
                f"Path traversal violation: '{relative_subpath}' attempts to escape "
                f"artifact root '{self.artifact_root}'.",
                attempted_path=resolved,
                partition_type="ARTIFACT",
            )

        return resolved
