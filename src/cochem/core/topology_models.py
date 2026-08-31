"""
Frozen Pydantic v2 Core Topology & State Models.

Provenance & Specifications:
- Method Matrix [M]: Tripartite Air-Gap Topology (T_code, T_data, T_art) schemas.
- Concurrency Protocol [C]: OS-level Single-Writer Multiple-Reader (SWMR) lock descriptors.
- Telemetry [T]: Bounded cryptographic Merkle telemetry payloads.
- Lifecycle [L]: Engine runtime health summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PartitionType = Literal["CODE", "DATA", "ARTIFACT"]
PartitionPermission = Literal["READ_ONLY", "READ_WRITE"]
TelemetrySeverity = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_ALLOWED_SUBDIRECTORIES: Dict[PartitionType, List[str]] = {
    "CODE": ["packages", "assets_js", "binaries"],
    "DATA": ["raw_baselines", "spectral_libs", "fixtures"],
    "ARTIFACT": ["checkpoints", "hdf5_datasets", "logs"],
}


class PartitionDescriptor(BaseModel):
    """Immutable descriptor for an air-gapped filesystem partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_type: PartitionType
    root_path: Path
    permission: PartitionPermission
    allowed_subdirectories: List[str] = Field(default_factory=list)
    is_air_gapped: bool = True

    @model_validator(mode="before")
    @classmethod
    def populate_default_subdirectories(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ptype = data.get("partition_type")
            subdirs = data.get("allowed_subdirectories")
            if (subdirs is None or len(subdirs) == 0) and ptype in DEFAULT_ALLOWED_SUBDIRECTORIES:
                data["allowed_subdirectories"] = list(DEFAULT_ALLOWED_SUBDIRECTORIES[ptype])
        return data

    @model_validator(mode="after")
    def validate_partition_permissions(self) -> Self:
        if self.partition_type in ("CODE", "DATA") and self.permission != "READ_ONLY":
            raise ValueError(
                f"Partition type '{self.partition_type}' strictly mandates permission='READ_ONLY', "
                f"got '{self.permission}'."
            )
        if self.partition_type == "ARTIFACT" and self.permission != "READ_WRITE":
            raise ValueError(
                f"Partition type 'ARTIFACT' strictly mandates permission='READ_WRITE', "
                f"got '{self.permission}'."
            )
        return self


class SWMRFileLockDescriptor(BaseModel):
    """Immutable record of an active SWMR file lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_filepath: Path
    lock_filepath: Path
    owner_pid: int
    acquired_timestamp_iso: str
    lease_duration_seconds: float = 30.0
    is_active: bool = True


class TelemetryEventPayload(BaseModel):
    """Immutable cryptographic telemetry event payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    timestamp_iso: str
    source_subsystem: str
    severity: TelemetrySeverity
    message: str
    merkle_hash: str
    context_metadata: Dict[str, Any] = Field(default_factory=dict)


class EngineHealthSummary(BaseModel):
    """Immutable snapshot of live CoChem Core Engine health diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_version: str
    uptime_seconds: float
    partitions_valid: bool
    active_swmr_locks: int
    telemetry_queue_size: int
    mendeleev_cache_hits: int
    mendeleev_cache_misses: int
