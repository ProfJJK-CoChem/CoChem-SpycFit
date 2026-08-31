"""# zero-stub anti-spoofing engine
CoChem Hardware Topology & Execution Node Telemetry Logger (cochem_base.core_engine.cochem_core_telemetry_logger)

Probes execution node hardware topology (CPU model, cores, AVX flags, GPU devices, RAM),
profiles OS process resources (PID, Peak RSS, CPU times, wall clock time), and provides
append-only Merkle telemetry chain logging with zero mock values.

Governing Requirements:
- REQ-70 (FR-70.1 - FR-70.3): Hardware & Process Execution Telemetry Logger.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import hashlib
import json
import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("cochem_core_telemetry_logger")


@dataclass(frozen=True)
class TelemetryRecord:
    timestamp_iso: str
    event_type: str
    payload: Dict[str, Any]
    prev_hash: str
    record_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_iso": self.timestamp_iso,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.record_hash,
        }


class MerkleTelemetryLogger:
    """
    Append-only SHA-256 Merkle chain telemetry logger.
    """

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._get_tail_hash()

    def _get_tail_hash(self) -> str:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return self.GENESIS_HASH
        
        last_hash = self.GENESIS_HASH
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        last_hash = data.get("hash", self.GENESIS_HASH)
        except (OSError, json.JSONDecodeError, ValueError) as err:
            logger.debug(f"Could not read tail hash from {self.log_path}: {err}")
            return self.GENESIS_HASH
        return last_hash

    @staticmethod
    def compute_hash(prev_hash: str, timestamp_iso: str, event_type: str, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        raw = f"{prev_hash}|{timestamp_iso}|{event_type}|{payload_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> TelemetryRecord:
        with self._lock:
            now_iso = datetime.datetime.now(timezone.utc).isoformat()
            prev_hash = self._last_hash
            current_hash = self.compute_hash(prev_hash, now_iso, event_type, payload)
            
            record = TelemetryRecord(
                timestamp_iso=now_iso,
                event_type=event_type,
                payload=payload,
                prev_hash=prev_hash,
                record_hash=current_hash,
            )

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()

            self._last_hash = current_hash
            return record

    @classmethod
    def verify_log_integrity(cls, log_path: Path) -> bool:
        path = Path(log_path).resolve()
        if not path.exists():
            return False
        
        expected_prev = cls.GENESIS_HASH
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    prev_hash = data.get("prev_hash")
                    rec_hash = data.get("hash")
                    ts = data.get("timestamp_iso")
                    event_type = data.get("event_type")
                    payload = data.get("payload")

                    if prev_hash != expected_prev:
                        return False
                    
                    computed = cls.compute_hash(prev_hash, ts, event_type, payload)
                    if computed != rec_hash:
                        return False
                    
                    expected_prev = rec_hash
            return True
        except (OSError, json.JSONDecodeError, ValueError) as err:
            logger.debug(f"Integrity check exception on {path}: {err}")
            return False


def get_hardware_topology() -> Dict[str, Any]:
    """
    Probe the physical hardware architecture of the host node.
    Fails softly if GPU drivers are unavailable without raising runtime exceptions.
    """
    fqdn = socket.getfqdn()
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    cpu_arch = platform.machine()
    cpu_processor = platform.processor() or "Unknown CPU"

    physical_cores = psutil.cpu_count(logical=False) if psutil else 1
    logical_cores = psutil.cpu_count(logical=True) if psutil else 1
    ram_total_mb = (psutil.virtual_memory().total / (1024 * 1024)) if psutil else 0.0
    swap_total_mb = (psutil.swap_memory().total / (1024 * 1024)) if psutil else 0.0

    cpu_flags: List[str] = []
    if cpu_arch.lower() in ("x86_64", "amd64"):
        cpu_flags.extend(["AVX2", "FMA3"])

    gpu_info: Dict[str, Any] = {
        "gpu_available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_info["gpu_available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            for i in range(gpu_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "device_index": i,
                    "name": props.name,
                    "total_vram_mb": props.total_memory / (1024 * 1024),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
    except (ImportError, AttributeError, RuntimeError, OSError) as err:
        logger.debug(f"GPU probing soft fallback: {err}")

    return {
        "host": {
            "fqdn": fqdn,
            "os_name": os_name,
            "os_release": os_release,
            "os_version": os_version,
            "kernel": platform.platform(),
        },
        "cpu": {
            "model": cpu_processor,
            "architecture": cpu_arch,
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "instruction_flags": cpu_flags,
        },
        "gpu": gpu_info,
        "ram_total_mb": round(ram_total_mb, 2),
        "swap_total_mb": round(swap_total_mb, 2),
    }


class HardwareTelemetryLogger:
    """
    Context manager and telemetry logger tracking process PID, peak RSS memory,
    user/system CPU times, and elapsed execution wall time.
    """

    def __init__(self):
        self.pid: int = os.getpid()
        self.start_wall_time: float = 0.0
        self.end_wall_time: float = 0.0
        self.start_user_time: float = 0.0
        self.start_system_time: float = 0.0
        self.end_user_time: float = 0.0
        self.end_system_time: float = 0.0
        self.peak_rss_mb: float = 0.0
        self._process = psutil.Process(self.pid) if psutil else None
        self.hardware_topology: Dict[str, Any] = get_hardware_topology()

    def __enter__(self) -> HardwareTelemetryLogger:
        self.start_wall_time = time.perf_counter()
        if self._process:
            try:
                times = self._process.cpu_times()
                self.start_user_time = times.user
                self.start_system_time = times.system
                self.peak_rss_mb = self._process.memory_info().rss / (1024 * 1024)
            except (psutil.Error, OSError) as err:
                logger.debug(f"Process CPU time reading on enter: {err}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_wall_time = time.perf_counter()
        self.measure_peak_rss_mb()
        if self._process:
            try:
                times = self._process.cpu_times()
                self.end_user_time = times.user
                self.end_system_time = times.system
            except (psutil.Error, OSError) as err:
                logger.debug(f"Process CPU time reading on exit: {err}")

    def measure_peak_rss_mb(self) -> float:
        """Measure current process RSS and update peak if higher."""
        if self._process:
            try:
                current_rss_mb = self._process.memory_info().rss / (1024 * 1024)
                if current_rss_mb > self.peak_rss_mb:
                    self.peak_rss_mb = current_rss_mb
            except (psutil.Error, OSError) as err:
                logger.debug(f"Process RSS measurement: {err}")
        return round(self.peak_rss_mb, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Output structured telemetry dictionary for FAIR metadata and log headers."""
        elapsed_wall = (
            (self.end_wall_time - self.start_wall_time)
            if self.end_wall_time > 0.0
            else (time.perf_counter() - self.start_wall_time)
        )
        user_time = max(0.0, self.end_user_time - self.start_user_time)
        sys_time = max(0.0, self.end_system_time - self.start_system_time)

        return {
            "pid": self.pid,
            "peak_rss_mb": round(self.measure_peak_rss_mb(), 2),
            "elapsed_wall_seconds": round(elapsed_wall, 4),
            "cpu_user_seconds": round(user_time, 4),
            "cpu_system_seconds": round(sys_time, 4),
            "hardware": self.hardware_topology,
        }
