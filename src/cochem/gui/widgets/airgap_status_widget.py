"""
BASE GUI Air-Gap Status & Resource Monitor Widget.

Provenance & Specifications:
- BASE GUI Specification [G]: Dashboard status monitoring for Tripartite Partitions and SWMR locks.
- Telemetry [T]: Live queue utilization, throughput, and Merkle health visualization.
- Mendeleev [M]: Dynamic element cache hit ratio display.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cochem.core.airgap_arbiter import AirGapArbiter
from cochem.core.topology_models import (
    EngineHealthSummary,
    PartitionDescriptor,
    SWMRFileLockDescriptor,
)


class AirGapStatusWidget:
    """
    Status and resource monitor widget for the CoChem BASE GUI.
    Visualizes partition integrity, active SWMR file locks, telemetry throughput,
    and Mendeleev resolver cache hit ratios.
    """

    def __init__(
        self,
        airgap_arbiter: Optional[AirGapArbiter] = None,
    ) -> None:
        self.airgap_arbiter = airgap_arbiter
        self._last_summary: Optional[EngineHealthSummary] = None
        self._active_locks: List[SWMRFileLockDescriptor] = []
        self._last_updated_iso: str = datetime.now(timezone.utc).isoformat()

    def update_status(
        self,
        summary: EngineHealthSummary,
        active_locks: Optional[List[SWMRFileLockDescriptor]] = None,
    ) -> None:
        """
        Update widget model state with a new health summary snapshot.

        Args:
            summary: Fresh EngineHealthSummary snapshot from CoChemEngine.
            active_locks: Optional list of active SWMR file lock descriptors.
        """
        self._last_summary = summary
        if active_locks is not None:
            self._active_locks = list(active_locks)
        self._last_updated_iso = datetime.now(timezone.utc).isoformat()

    def get_partition_badges(self) -> Dict[str, Dict[str, Any]]:
        """Compute status badges for T_code, T_data, and T_art."""
        badges: Dict[str, Dict[str, Any]] = {
            "T_code": {
                "label": "Code Partition (T_code)",
                "permission": "READ_ONLY",
                "color": "#28a745",  # Green
                "status": "VALID",
                "root_path": str(self.airgap_arbiter.code_root) if self.airgap_arbiter else "N/A",
            },
            "T_data": {
                "label": "Data Partition (T_data)",
                "permission": "READ_ONLY",
                "color": "#28a745",  # Green
                "status": "VALID",
                "root_path": str(self.airgap_arbiter.data_root) if self.airgap_arbiter else "N/A",
            },
            "T_art": {
                "label": "Artifact Partition (T_art)",
                "permission": "READ_WRITE",
                "color": "#007bff",  # Blue
                "status": "VALID",
                "root_path": str(self.airgap_arbiter.artifact_root) if self.airgap_arbiter else "N/A",
            },
        }

        if self.airgap_arbiter is not None:
            try:
                self.airgap_arbiter.verify_partitions()
            except Exception:
                for b in badges.values():
                    b["status"] = "ERROR"

        return badges

    def calculate_mendeleev_hit_ratio(self) -> float:
        """Calculate percentage of cache hits for dynamic Mendeleev resolution."""
        if self._last_summary is None:
            return 0.0
        total = self._last_summary.mendeleev_cache_hits + self._last_summary.mendeleev_cache_misses
        if total == 0:
            return 100.0
        return (self._last_summary.mendeleev_cache_hits / total) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current widget state to a structured dictionary."""
        return {
            "last_updated_iso": self._last_updated_iso,
            "engine_summary": self._last_summary.model_dump() if self._last_summary else None,
            "partition_badges": self.get_partition_badges(),
            "mendeleev_cache_hit_ratio_percent": self.calculate_mendeleev_hit_ratio(),
            "active_swmr_locks": [lock.model_dump() for lock in self._active_locks],
        }

    def render_text_summary(self) -> str:
        """Render a clean plaintext status summary for CLI or terminal dashboards."""
        badges = self.get_partition_badges()
        lines: List[str] = [
            "============================================================",
            "        CoChem Tripartite Air-Gap & Engine Dashboard        ",
            "============================================================",
            f"Last Updated: {self._last_updated_iso}",
        ]

        if self._last_summary:
            lines.append(f"Engine Version: {self._last_summary.engine_version}")
            lines.append(f"Uptime:         {self._last_summary.uptime_seconds:.2f}s")
            lines.append(f"Partitions OK:  {self._last_summary.partitions_valid}")
            lines.append(f"Active Locks:   {self._last_summary.active_swmr_locks}")
            lines.append(f"Telemetry Q:    {self._last_summary.telemetry_queue_size} items")
            lines.append(
                f"Mendeleev Cache: {self.calculate_mendeleev_hit_ratio():.1f}% hits "
                f"({self._last_summary.mendeleev_cache_hits} hits / {self._last_summary.mendeleev_cache_misses} misses)"
            )
        else:
            lines.append("Engine Status: NOT INITIALIZED / NO DATA")

        lines.append("------------------------------------------------------------")
        lines.append("Partitions:")
        for key, info in badges.items():
            lines.append(
                f"  [{info['status']}] {info['label']} - {info['permission']} ({info['root_path']})"
            )

        if self._active_locks:
            lines.append("------------------------------------------------------------")
            lines.append("Active SWMR Locks:")
            for lock in self._active_locks:
                lines.append(
                    f"  - Target: {lock.target_filepath} | PID: {lock.owner_pid} | Active: {lock.is_active}"
                )

        lines.append("============================================================")
        return "\n".join(lines)

    def render_html(self) -> str:
        """Render an HTML widget representation for Jupyter UI / BASE Web views."""
        badges = self.get_partition_badges()
        hit_ratio = self.calculate_mendeleev_hit_ratio()

        badges_html = "".join(
            f"""
            <div style="display:inline-block; margin:5px; padding:10px; border-radius:6px; background:{info['color']}; color:white; font-family:sans-serif;">
                <strong>{info['label']}</strong><br/>
                <small>{info['permission']} | {info['status']}</small>
            </div>
            """
            for info in badges.values()
        )

        locks_count = len(self._active_locks)
        q_size = self._last_summary.telemetry_queue_size if self._last_summary else 0
        uptime = f"{self._last_summary.uptime_seconds:.1f}s" if self._last_summary else "0.0s"

        return f"""
        <div style="border:1px solid #ccc; padding:15px; border-radius:8px; font-family:sans-serif; max-width:800px; background:#f9f9f9;">
            <h3 style="margin-top:0;">CoChem Tripartite Status & Engine Monitor</h3>
            <div style="margin-bottom:12px;">{badges_html}</div>
            <table style="width:100%; border-collapse:collapse; font-size:14px;">
                <tr><td style="padding:4px; font-weight:bold;">Uptime:</td><td>{uptime}</td></tr>
                <tr><td style="padding:4px; font-weight:bold;">Telemetry Queue Size:</td><td>{q_size}</td></tr>
                <tr><td style="padding:4px; font-weight:bold;">Active SWMR Locks:</td><td>{locks_count}</td></tr>
                <tr><td style="padding:4px; font-weight:bold;">Mendeleev Cache Hit Rate:</td><td>{hit_ratio:.1f}%</td></tr>
            </table>
        </div>
        """
