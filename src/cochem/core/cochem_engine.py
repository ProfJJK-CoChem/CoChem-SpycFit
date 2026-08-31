"""
Master CoChem Runtime Engine Lifecycle Orchestrator.

Provenance & Specifications:
- Architecture [A]: Tripartite Air-Gap Topology (T_code, T_data, T_art) Master Lifecycle.
- Concurrency & SWMR [C]: Coordinated atomic file locking and SWMR HDF5 dataset management.
- Chemical Verification [M]: Zero-hardcoded dynamic Mendeleev element and isotope invariants.
- Telemetry [T]: Non-blocking ring-buffered cryptographic event logging with Merkle hash integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from cochem.core.airgap_arbiter import AirGapArbiter
from cochem.core.mendeleev_invariants import MendeleevResolver, mendeleev_resolver
from cochem.core.topology_exceptions import (
    AirGapPartitionViolationError,
    CoChemTopologyException,
    PartitionMountError,
)
from cochem.core.topology_models import EngineHealthSummary, SWMRFileLockDescriptor
from cochem.gui.widgets.airgap_status_widget import AirGapStatusWidget
from cochem.storage.swmr_hdf5_arbiter import SWMRHDF5Arbiter
from cochem.telemetry.telemetry_queue import TelemetryQueue

ENGINE_VERSION = "2.0.0"


class CoChemEngine:
    """
    Master CoChem runtime orchestrator. Coordinates partition air-gaps,
    dynamic chemical element resolution, SWMR HDF5 storage, and telemetry logging.
    """

    def __init__(
        self,
        code_root: Union[Path, str],
        data_root: Union[Path, str],
        artifact_root: Union[Path, str],
        resolver: Optional[MendeleevResolver] = None,
        auto_start_telemetry: bool = True,
    ) -> None:
        self.code_root = Path(code_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.auto_start_telemetry = auto_start_telemetry

        self._start_time: float = time.monotonic()
        self._is_initialized: bool = False
        self._is_shutdown: bool = False

        # Initialize sub-systems
        self.mendeleev: MendeleevResolver = resolver or mendeleev_resolver
        self.airgap = AirGapArbiter(
            code_root=self.code_root,
            data_root=self.data_root,
            artifact_root=self.artifact_root,
            auto_create_artifact_subdirs=True,
        )
        self.swmr = SWMRHDF5Arbiter(
            airgap_arbiter=self.airgap,
        )
        self.telemetry = TelemetryQueue(
            airgap_arbiter=self.airgap,
        )
        self.status_widget = AirGapStatusWidget(
            airgap_arbiter=self.airgap,
        )

        self.initialize()

    @property
    def is_initialized(self) -> bool:
        """Return True if engine is successfully initialized."""
        return self._is_initialized

    @property
    def is_shutdown(self) -> bool:
        """Return True if engine has been shut down."""
        return self._is_shutdown

    @property
    def uptime_seconds(self) -> float:
        """Elapsed runtime since engine instantiation in seconds."""
        return time.monotonic() - self._start_time

    def initialize(self) -> None:
        """
        Validate partition directories, verify air-gaps, and launch telemetry daemon.
        """
        if self._is_initialized:
            return

        # Verify physical existence and permissions of partitions
        self.airgap.verify_partitions()

        # Start telemetry daemon if configured
        if self.auto_start_telemetry:
            self.telemetry.start()
            self.telemetry.emit_event(
                source_subsystem="CoChemEngine",
                severity="INFO",
                message="CoChemEngine initialized with Tripartite Air-Gap Topology.",
                context_metadata={
                    "code_root": str(self.code_root),
                    "data_root": str(self.data_root),
                    "artifact_root": str(self.artifact_root),
                    "engine_version": ENGINE_VERSION,
                },
            )

        self._is_initialized = True
        self._is_shutdown = False

    def shutdown(self) -> None:
        """
        Gracefully shut down all sub-systems, flush telemetry, and release resources.
        """
        if self._is_shutdown:
            return

        try:
            self.telemetry.emit_event(
                source_subsystem="CoChemEngine",
                severity="INFO",
                message="CoChemEngine shutting down.",
                context_metadata={"uptime_seconds": self.uptime_seconds},
            )
        except Exception:
            pass

        # Stop and drain telemetry daemon
        self.telemetry.stop()

        self._is_shutdown = True
        self._is_initialized = False

    def get_health_summary(self) -> EngineHealthSummary:
        """
        Collect and return live engine diagnostic health snapshot.
        """
        partitions_valid = False
        try:
            partitions_valid = self.airgap.verify_partitions()
        except Exception:
            partitions_valid = False

        hits, misses = self.mendeleev.get_cache_stats()

        summary = EngineHealthSummary(
            engine_version=ENGINE_VERSION,
            uptime_seconds=self.uptime_seconds,
            partitions_valid=partitions_valid,
            active_swmr_locks=0,
            telemetry_queue_size=self.telemetry.queue_size,
            mendeleev_cache_hits=hits,
            mendeleev_cache_misses=misses,
        )

        # Update widget model snapshot
        self.status_widget.update_status(summary)
        return summary

    def __enter__(self) -> CoChemEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
