"""
Non-Blocking Ring-Buffered Telemetry Queue.

Provenance & Specifications:
- Telemetry Integrity [T]: SHA-256 Merkle hash chained cryptographic event logs.
- Resource Safeguards [R]: Bounded memory queue, log tail truncation (>4 KB), and non-blocking emission.
- Air-Gap Invariant [A]: Event records stream strictly into T_art/logs/ directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from cochem.core.airgap_arbiter import AirGapArbiter
from cochem.core.topology_exceptions import (
    AirGapPartitionViolationError,
    TelemetryQueueOverflowError,
)
from cochem.core.topology_models import TelemetryEventPayload, TelemetrySeverity

# Maximum un-truncated payload size in bytes before structured truncation is applied
MAX_PAYLOAD_BYTES = 4096

# Genesis block Merkle hash seed
GENESIS_MERKLE_HASH = "0" * 64


def _truncate_payload_metadata(metadata: Dict[str, Any], max_bytes: int = MAX_PAYLOAD_BYTES) -> Dict[str, Any]:
    """
    Truncate large metadata payloads with a structured summary to prevent token/memory overflow.
    """
    try:
        raw_json = json.dumps(metadata, default=str)
        if len(raw_json.encode("utf-8")) <= max_bytes:
            return metadata

        # Structured truncation summary
        truncated: Dict[str, Any] = {
            "_truncated": True,
            "_original_byte_length": len(raw_json.encode("utf-8")),
            "_key_count": len(metadata),
        }
        for k, v in list(metadata.items())[:10]:
            v_str = str(v)
            truncated[k] = v_str[:128] + ("..." if len(v_str) > 128 else "")
        return truncated
    except Exception:
        return {"_truncated": True, "_error": "Failed to serialize metadata"}


class TelemetryQueue:
    """
    Thread-safe, non-blocking telemetry event queue and background daemon logger.
    Buffers events in memory, computes running SHA-256 Merkle hash chaining,
    and flushes batches to T_art/logs/telemetry.jsonl.
    """

    def __init__(
        self,
        log_dir: Optional[Union[Path, str]] = None,
        airgap_arbiter: Optional[AirGapArbiter] = None,
        maxsize: int = 10000,
        flush_interval_seconds: float = 0.25,
        log_filename: str = "telemetry.jsonl",
    ) -> None:
        self.airgap_arbiter = airgap_arbiter
        self.maxsize = int(maxsize)
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.log_filename = log_filename

        # Determine log directory
        if log_dir is not None:
            self.log_dir = Path(log_dir).resolve()
        elif airgap_arbiter is not None:
            self.log_dir = airgap_arbiter.get_artifact_path("logs")
        else:
            self.log_dir = Path("logs").resolve()

        self.log_filepath = self.log_dir / self.log_filename

        # Internal queue and state synchronization
        self._queue: queue.Queue[TelemetryEventPayload] = queue.Queue(maxsize=self.maxsize)
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry metrics and Merkle state
        self._total_emitted: int = 0
        self._total_flushed: int = 0
        self._dropped_count: int = 0
        self._current_merkle_hash: str = GENESIS_MERKLE_HASH

    @property
    def queue_size(self) -> int:
        """Current number of buffered events in the in-memory queue."""
        return self._queue.qsize()

    @property
    def total_emitted(self) -> int:
        """Total events submitted via emit_event."""
        return self._total_emitted

    @property
    def total_flushed(self) -> int:
        """Total events successfully written and flushed to disk."""
        return self._total_flushed

    @property
    def dropped_count(self) -> int:
        """Total events dropped due to queue saturation."""
        return self._dropped_count

    @property
    def current_merkle_hash(self) -> str:
        """Current leaf hash of the running Merkle chain."""
        return self._current_merkle_hash

    def start(self) -> None:
        """Start the background daemon logging worker thread."""
        with self._lock:
            if self._running:
                return

            # Air-gap write permission verification
            if self.airgap_arbiter is not None:
                self.airgap_arbiter.validate_write_access(self.log_filepath)

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._daemon_worker_loop,
                name="CoChemTelemetryWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background worker and flush all remaining queued telemetry events."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        # Final drain to ensure zero lost events
        self.drain()

    def emit_event(
        self,
        source_subsystem: str,
        severity: TelemetrySeverity,
        message: str,
        context_metadata: Optional[Dict[str, Any]] = None,
        raise_on_overflow: bool = False,
    ) -> Optional[TelemetryEventPayload]:
        """
        Emit a telemetry event non-blockingly.

        Args:
            source_subsystem: Identifier for the originating subsystem.
            severity: Event severity level.
            message: Informational or diagnostic description.
            context_metadata: Arbitrary metadata dictionary (subject to auto-truncation).
            raise_on_overflow: Whether to raise TelemetryQueueOverflowError on saturation.

        Returns:
            TelemetryEventPayload instance or None if dropped.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        event_id = str(uuid.uuid4())
        meta = _truncate_payload_metadata(context_metadata or {})

        # Compute Merkle hash over (previous_hash + event_id + timestamp + message)
        with self._lock:
            prev_hash = self._current_merkle_hash
            raw_signature = f"{prev_hash}|{event_id}|{now_iso}|{source_subsystem}|{severity}|{message}"
            new_hash = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()
            self._current_merkle_hash = new_hash
            self._total_emitted += 1

        payload = TelemetryEventPayload(
            event_id=event_id,
            timestamp_iso=now_iso,
            source_subsystem=source_subsystem,
            severity=severity,
            message=message,
            merkle_hash=new_hash,
            context_metadata=meta,
        )

        try:
            self._queue.put_nowait(payload)
            return payload
        except queue.Full:
            with self._lock:
                self._dropped_count += 1

            if raise_on_overflow or severity in ("ERROR", "CRITICAL"):
                raise TelemetryQueueOverflowError(
                    f"Telemetry queue capacity ({self.maxsize}) exceeded during dispatch of {severity} event.",
                    queue_size=self.queue_size,
                    dropped_count=self._dropped_count,
                )
            return None

    def drain(self) -> int:
        """
        Drain all currently buffered events synchronously to the physical log file.

        Returns:
            Number of flushed records.
        """
        records: List[TelemetryEventPayload] = []
        while True:
            try:
                item = self._queue.get_nowait()
                records.append(item)
            except queue.Empty:
                break

        if not records:
            return 0

        self._write_records_to_disk(records)
        return len(records)

    def flush(self) -> int:
        """Alias for drain."""
        return self.drain()

    def _write_records_to_disk(self, records: List[TelemetryEventPayload]) -> None:
        """Atomically append records to the telemetry JSONL log."""
        if not records:
            return

        if self.airgap_arbiter is not None:
            self.airgap_arbiter.validate_write_access(self.log_filepath)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(rec.model_dump()) + "\n" for rec in records]

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()

        with self._lock:
            self._total_flushed += len(records)

    def _daemon_worker_loop(self) -> None:
        """Internal background daemon polling loop."""
        while self._running:
            try:
                self.drain()
            except Exception:
                pass
            time.sleep(self.flush_interval_seconds)

    def __enter__(self) -> TelemetryQueue:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
