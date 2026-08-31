"""# zero-stub anti-spoofing engine
CoChem Cross-Platform ACL File Sealing & Rolling Merkle Log Streamer (cochem_base.core_engine.cochem_fs_security)

Provides unbuffered byte-for-byte stream duplication, rolling 64KB Merkle hash chaining,
cross-platform write-protection / ACL file sealing, and log tamper verification.

Governing Requirements:
- REQ-66 (FR-66.1 - FR-66.3): Filesystem Security, Merkle Hash Streams, & ACL Sealing.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("cochem_fs_security")

CHUNK_SIZE: int = 65536  # 64 KB chunk size


class LogStreamMultiplexer:
    """
    Unbuffered stream duplicator that intercepts stdout/stderr byte-for-byte,
    writing simultaneously to console and disk while maintaining a rolling Merkle hash chain.
    """

    def __init__(
        self,
        log_file_path: Union[str, Path],
        console_stream: Optional[io.TextIOBase] = None,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_stream = console_stream or sys.stdout
        self.chunk_size = chunk_size

        self._file_handle: Optional[io.BufferedWriter] = None
        self._buffer = bytearray()
        self._current_hash: str = "0" * 64
        self._total_bytes_written: int = 0
        self._is_closed: bool = False

    def __enter__(self) -> LogStreamMultiplexer:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Open the target log file for unbuffered binary appending."""
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.log_file_path, "wb", buffering=0)

    def write(self, data: Union[str, bytes]) -> int:
        """Write data simultaneously to console, disk, and Merkle hash buffer."""
        if self._is_closed:
            raise ValueError("Cannot write to closed LogStreamMultiplexer")
        if self._file_handle is None:
            self.open()

        raw_bytes = data.encode("utf-8") if isinstance(data, str) else data
        if not raw_bytes:
            return 0

        # 1. Write to console stream safely
        try:
            if isinstance(data, str):
                self.console_stream.write(data)
            else:
                self.console_stream.write(raw_bytes.decode("utf-8", errors="replace"))
            self.console_stream.flush()
        except (IOError, OSError) as err:
            logger.debug(f"Console stream write issue: {err}")

        # 2. Write to unbuffered log file
        assert self._file_handle is not None
        self._file_handle.write(raw_bytes)
        self._total_bytes_written += len(raw_bytes)

        # 3. Rolling Merkle Hash Chaining over 64KB blocks
        self._buffer.extend(raw_bytes)
        while len(self._buffer) >= self.chunk_size:
            chunk = bytes(self._buffer[: self.chunk_size])
            self._buffer = self._buffer[self.chunk_size :]
            self._current_hash = hashlib.sha256(
                self._current_hash.encode("utf-8") + chunk
            ).hexdigest()

        return len(raw_bytes)

    def flush(self) -> None:
        """Flush file and console streams."""
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.flush()
        if self.console_stream:
            try:
                self.console_stream.flush()
            except (IOError, OSError) as err:
                logger.debug(f"Console stream flush issue: {err}")

    def close(self) -> None:
        """Finalize the Merkle hash chain over remaining buffer bytes and close file."""
        if self._is_closed:
            return

        if len(self._buffer) > 0:
            chunk = bytes(self._buffer)
            self._buffer.clear()
            self._current_hash = hashlib.sha256(
                self._current_hash.encode("utf-8") + chunk
            ).hexdigest()

        if self._file_handle and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()

        self._is_closed = True

    def get_merkle_root(self) -> str:
        """Get the final 64-character SHA-256 Merkle root hash."""
        if not self._is_closed:
            self.close()
        return self._current_hash

    @property
    def root_hash(self) -> str:
        return self.get_merkle_root()


def seal_log_file(file_path: Union[str, Path]) -> bool:
    """
    Apply strict immutable write-protection / ACL file sealing on POSIX and Windows.
    
    POSIX: Sets permissions 0440 and applies immutable flag where permitted.
    Windows: Applies FILE_ATTRIBUTE_READONLY via Win32 API and removes write access.
    """
    p = Path(file_path).resolve()
    if not p.exists():
        logger.error(f"Cannot seal non-existent file: {p.as_posix()}")
        return False

    try:
        if sys.platform == "win32":
            # Set Windows Read-Only file attribute
            FILE_ATTRIBUTE_READONLY = 0x00000001
            try:
                ctypes.windll.kernel32.SetFileAttributesW(str(p), FILE_ATTRIBUTE_READONLY)
            except (AttributeError, OSError) as err:
                logger.debug(f"Win32 SetFileAttributesW not applicable: {err}")
            # Also apply os.chmod read-only
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        else:
            # POSIX chmod 0440
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP)
            # Try applying immutable chattr flag if root
            try:
                subprocess.run(["chattr", "+i", str(p)], capture_output=True, timeout=2)
            except (subprocess.SubprocessError, OSError) as err:
                logger.debug(f"POSIX chattr +i not permitted or unavailable: {err}")
        return True
    except Exception as e:
        logger.warning(f"Failed to seal log file {p.as_posix()}: {e}")
        return False


def verify_log_integrity(file_path: Union[str, Path], expected_merkle_root: str) -> bool:
    """
    Re-compute the 64KB rolling Merkle hash chain over a sealed log file
    and verify whether it matches the expected Merkle root.
    """
    p = Path(file_path).resolve()
    if not p.exists():
        return False

    current_hash = "0" * 64
    try:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                current_hash = hashlib.sha256(
                    current_hash.encode("utf-8") + chunk
                ).hexdigest()
        return current_hash == expected_merkle_root
    except Exception as e:
        logger.error(f"Integrity check failed with error: {e}")
        return False
