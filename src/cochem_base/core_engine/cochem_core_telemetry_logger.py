"""
Append-Only Merkle Telemetry Logger with Cryptographic Chain Verification.
"""

from __future__ import annotations

import json
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


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
        except Exception:
            return self.GENESIS_HASH
        return last_hash

    @staticmethod
    def compute_hash(prev_hash: str, timestamp_iso: str, event_type: str, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        raw = f"{prev_hash}|{timestamp_iso}|{event_type}|{payload_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> TelemetryRecord:
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
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
        except Exception:
            return False
