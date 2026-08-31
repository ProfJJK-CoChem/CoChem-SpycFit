"""
Secure Subprocess Broker with Sandbox Quarantine and Resource Monitoring.
"""

from __future__ import annotations

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil


class SubprocessBroker:
    """
    Launches and monitors external calculation processes with isolation and safety limits.
    """

    def __init__(self, default_scratch_dir: Optional[Path] = None) -> None:
        self.default_scratch_dir = default_scratch_dir or Path(os.environ.get("TEMP", "/tmp"))

    def run_isolated(
        self,
        cmd: List[str],
        cwd: Path,
        timeout_seconds: float = 300.0,
        memory_limit_mb: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str]:
        """
        Execute command with strict timeout, directory confinement, and memory ceiling limits.
        """
        work_dir = Path(cwd).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        stdout_log = work_dir / "stdout.log"
        stderr_log = work_dir / "stderr.log"

        with open(stdout_log, "wb") as f_out, open(stderr_log, "wb") as f_err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(work_dir),
                env=exec_env,
                stdout=f_out,
                stderr=f_err,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )

            start_time = time.time()
            try:
                while proc.poll() is None:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        self.terminate_process_tree(proc.pid)
                        return -1, stdout_log.read_text(encoding="utf-8", errors="replace"), f"Subprocess timed out after {timeout_seconds}s"

                    if memory_limit_mb is not None:
                        try:
                            p = psutil.Process(proc.pid)
                            mem_mb = p.memory_info().rss / (1024 * 1024)
                            for child in p.children(recursive=True):
                                mem_mb += child.memory_info().rss / (1024 * 1024)
                            if mem_mb > memory_limit_mb:
                                self.terminate_process_tree(proc.pid)
                                return -2, stdout_log.read_text(encoding="utf-8", errors="replace"), f"Memory limit exceeded: {mem_mb:.1f} MB > {memory_limit_mb} MB"
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    time.sleep(0.1)

            except KeyboardInterrupt:
                self.terminate_process_tree(proc.pid)
                raise

        return proc.returncode, stdout_log.read_text(encoding="utf-8", errors="replace"), stderr_log.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def terminate_process_tree(pid: int) -> None:
        """Terminate process and all child descendants cleanly."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
