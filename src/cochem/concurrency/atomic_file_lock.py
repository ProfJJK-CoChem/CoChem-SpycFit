"""
OS-Level Atomic File Lock Arbiter.

Provenance & Specifications:
- Concurrency Protocol [C]: OS-level physical atomic file locking (O_CREAT | O_EXCL).
- Stale Lock Reclamation: Dynamic PID liveness verification (psutil / os.kill) and lease timeouts.
- SWMR Invariance: Prevents multiple writers on HDF5 containers and ensures atomic exclusion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Dict, Optional, Union

import psutil

from cochem.core.topology_exceptions import SWMRFileLockTimeoutError
from cochem.core.topology_models import SWMRFileLockDescriptor


def _is_pid_alive(pid: int) -> bool:
    """Check whether a given PID is currently active on the host OS."""
    if pid <= 0:
        return False
    try:
        if not psutil.pid_exists(pid):
            return False
    except Exception:
        pass

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to another user
        return True
    except OSError:
        return False


class AtomicFileLock:
    """
    Cross-platform OS-level atomic file locking context manager.
    Uses physical atomic file creation (O_CREAT | O_EXCL) with lease expirations,
    PID stamping, and stale lock reclamation.
    """

    def __init__(
        self,
        lock_path: Union[Path, str],
        target_path: Optional[Union[Path, str]] = None,
        timeout: float = 10.0,
        lease_duration: float = 30.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.target_path = Path(target_path).resolve() if target_path else None
        self.timeout = float(timeout)
        self.lease_duration = float(lease_duration)
        self.poll_interval = float(poll_interval)

        self._is_acquired: bool = False
        self._acquired_iso: Optional[str] = None
        self._owner_pid: Optional[int] = None
        self._reentrancy_count: int = 0

    @property
    def is_acquired(self) -> bool:
        """Return True if this instance holds the active lock."""
        return self._is_acquired

    def acquire(self) -> SWMRFileLockDescriptor:
        """
        Acquire the atomic file lock within the timeout window.

        Returns:
            SWMRFileLockDescriptor describing the acquired lock.

        Raises:
            SWMRFileLockTimeoutError: If lock cannot be acquired within timeout.
        """
        current_pid = os.getpid()

        # Handle reentrancy in the same process/instance
        if self._is_acquired:
            self._reentrancy_count += 1
            return self.get_descriptor()

        # Ensure parent directory exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.monotonic()
        hostname = socket.gethostname()

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        last_known_owner_pid: Optional[int] = None

        while True:
            now_iso = datetime.now(timezone.utc).isoformat()
            payload = {
                "lock_filepath": str(self.lock_path),
                "target_filepath": str(self.target_path) if self.target_path else "",
                "owner_pid": current_pid,
                "hostname": hostname,
                "acquired_timestamp_iso": now_iso,
                "lease_duration_seconds": self.lease_duration,
                "is_active": True,
            }

            try:
                fd = os.open(str(self.lock_path), flags, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

                self._is_acquired = True
                self._acquired_iso = now_iso
                self._owner_pid = current_pid
                self._reentrancy_count = 1
                return self.get_descriptor()

            except FileExistsError:
                # Lock file already exists; inspect for staleness
                last_known_owner_pid = self._attempt_stale_reclamation(current_pid)

            elapsed = time.monotonic() - start_time
            if elapsed >= self.timeout:
                raise SWMRFileLockTimeoutError(
                    f"Failed to acquire atomic file lock on '{self.lock_path}' "
                    f"within {self.timeout:.2f}s timeout.",
                    lock_filepath=self.lock_path,
                    target_filepath=self.target_path,
                    timeout_seconds=self.timeout,
                    lock_owner_pid=last_known_owner_pid,
                )

            time.sleep(self.poll_interval)

    def _attempt_stale_reclamation(self, current_pid: int) -> Optional[int]:
        """Inspect lock file and remove if stale. Returns last known owner PID."""
        try:
            if not self.lock_path.exists():
                return None

            with open(self.lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            owner_pid = data.get("owner_pid")
            acquired_str = data.get("acquired_timestamp_iso")
            lease_sec = float(data.get("lease_duration_seconds", self.lease_duration))

            if owner_pid is None:
                # Corrupt lockfile -> reclaim
                self._force_unlink()
                return None

            # If current PID is owner but instance was reset, reclaim
            if owner_pid == current_pid and not self._is_acquired:
                self._force_unlink()
                return current_pid

            # Check PID liveness
            pid_alive = _is_pid_alive(int(owner_pid))
            if not pid_alive:
                # Owner process died -> stale lock reclamation
                self._force_unlink()
                return int(owner_pid)

            # Check lease expiration
            if acquired_str:
                try:
                    acquired_dt = datetime.fromisoformat(acquired_str)
                    now_dt = datetime.now(timezone.utc)
                    age_seconds = (now_dt - acquired_dt).total_seconds()
                    if age_seconds > lease_sec and not pid_alive:
                        self._force_unlink()
                        return int(owner_pid)
                except Exception:
                    pass

            return int(owner_pid)

        except (json.JSONDecodeError, OSError, ValueError):
            # If reading fails or file is empty during concurrent write, don't immediately delete
            return None

    def _force_unlink(self) -> None:
        """Safely delete the lock file."""
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except OSError:
            pass

    def release(self) -> None:
        """Release the lock and remove physical lock file."""
        if not self._is_acquired:
            return

        self._reentrancy_count -= 1
        if self._reentrancy_count > 0:
            return

        try:
            # Verify we are the owner before unlinking
            if self.lock_path.exists():
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("owner_pid") == os.getpid():
                        self.lock_path.unlink()
                except Exception:
                    self._force_unlink()
        finally:
            self._is_acquired = False
            self._acquired_iso = None
            self._owner_pid = None
            self._reentrancy_count = 0

    def get_descriptor(self) -> SWMRFileLockDescriptor:
        """Return the immutable lock descriptor."""
        if not self._is_acquired or self._acquired_iso is None or self._owner_pid is None:
            return SWMRFileLockDescriptor(
                target_filepath=self.target_path or self.lock_path,
                lock_filepath=self.lock_path,
                owner_pid=os.getpid(),
                acquired_timestamp_iso="",
                lease_duration_seconds=self.lease_duration,
                is_active=False,
            )

        return SWMRFileLockDescriptor(
            target_filepath=self.target_path or self.lock_path,
            lock_filepath=self.lock_path,
            owner_pid=self._owner_pid,
            acquired_timestamp_iso=self._acquired_iso,
            lease_duration_seconds=self.lease_duration,
            is_active=True,
        )

    def __enter__(self) -> AtomicFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    @staticmethod
    def is_locked(lock_path: Union[Path, str]) -> bool:
        """Check whether a lock file exists and is held by an active process."""
        p = Path(lock_path)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            owner_pid = data.get("owner_pid")
            if owner_pid is None:
                return False
            return _is_pid_alive(int(owner_pid))
        except Exception:
            return False

    @staticmethod
    def inspect_lock(lock_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
        """Inspect and parse lock metadata from disk."""
        p = Path(lock_path)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
