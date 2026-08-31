"""
Stateful Job Manager with Atomic Cross-Platform FileLock Coordination.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import filelock

from cochem_base.core.models import CalculationJobPayload, QCResultsRecord


class JobManager:
    """
    Manages asynchronous calculation job queue using filelock concurrency and atomic disk swaps.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = Path(state_file).resolve()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.state_file.with_suffix(".lock")
        self._lock = filelock.FileLock(str(self.lock_file), timeout=30.0)

        with self._lock:
            if not self.state_file.exists():
                self._write_state({"jobs": {}, "workers": {}})

    def _read_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"jobs": {}, "workers": {}}

    def _write_state(self, state: Dict[str, Any]) -> None:
        temp_file = self.state_file.with_suffix(f".tmp_{uuid.uuid4().hex}")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, self.state_file)

    def submit_job(self, payload: CalculationJobPayload) -> str:
        with self._lock:
            state = self._read_state()
            job_dict = payload.model_dump()
            job_dict["status"] = "QUEUED"
            job_dict["results"] = None
            job_dict["error"] = None
            job_dict["worker_id"] = None
            state["jobs"][payload.job_id] = job_dict
            self._write_state(state)
            return payload.job_id

    def acquire_job(self, worker_id: str) -> Optional[CalculationJobPayload]:
        with self._lock:
            state = self._read_state()
            for job_id, job in state["jobs"].items():
                if job["status"] == "QUEUED":
                    job["status"] = "RUNNING"
                    job["worker_id"] = worker_id
                    state["workers"][worker_id] = job_id
                    self._write_state(state)
                    return CalculationJobPayload.model_validate(job)
            return None

    def complete_job(self, job_id: str, results: QCResultsRecord) -> None:
        with self._lock:
            state = self._read_state()
            if job_id in state["jobs"]:
                state["jobs"][job_id]["status"] = "COMPLETED"
                state["jobs"][job_id]["results"] = results.model_dump()
                worker_id = state["jobs"][job_id].get("worker_id")
                if worker_id and worker_id in state["workers"]:
                    del state["workers"][worker_id]
                self._write_state(state)

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._lock:
            state = self._read_state()
            if job_id in state["jobs"]:
                state["jobs"][job_id]["status"] = "FAILED"
                state["jobs"][job_id]["error"] = error_message
                worker_id = state["jobs"][job_id].get("worker_id")
                if worker_id and worker_id in state["workers"]:
                    del state["workers"][worker_id]
                self._write_state(state)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._read_state()
            return state["jobs"].get(job_id)
