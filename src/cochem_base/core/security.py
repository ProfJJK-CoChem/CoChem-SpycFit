"""
Filesystem Access Control Lists (ACLs) and Tripartite Air-Gap Sandbox Isolation.
"""

from __future__ import annotations

import os
import sys
import stat
import ctypes
from pathlib import Path
from typing import Dict


def enforce_readonly_acl(path: Path) -> None:
    """
    Apply operating-system level read-only lock to target file or directory.
    
    Uses Win32 SetFileAttributesW on Windows and POSIX chmod 0444 on Linux/macOS.
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist for read-only locking: {path_obj}")

    if sys.platform == "win32":
        FILE_ATTRIBUTE_READONLY = 0x01
        res = ctypes.windll.kernel32.SetFileAttributesW(str(path_obj), FILE_ATTRIBUTE_READONLY)
        if res == 0:
            raise OSError(f"Failed to set Windows read-only attribute on {path_obj}")
    else:
        if path_obj.is_dir():
            os.chmod(path_obj, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        else:
            os.chmod(path_obj, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def remove_readonly_acl(path: Path) -> None:
    """
    Remove read-only lock from target file or directory.
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        return

    if sys.platform == "win32":
        FILE_ATTRIBUTE_NORMAL = 0x80
        res = ctypes.windll.kernel32.SetFileAttributesW(str(path_obj), FILE_ATTRIBUTE_NORMAL)
        if res == 0:
            raise OSError(f"Failed to clear Windows read-only attribute on {path_obj}")
    else:
        if path_obj.is_dir():
            os.chmod(path_obj, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        else:
            os.chmod(path_obj, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def create_tripartite_sandbox(base_dir: Path) -> Dict[str, Path]:
    """
    Instantiate tripartite air-gap workspace structure with code/, data/, and artifact/ directories.
    """
    base = Path(base_dir).resolve()
    code_dir = base / "code"
    data_dir = base / "data"
    artifact_dir = base / "artifact"

    for d in [code_dir, data_dir, artifact_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "code": code_dir,
        "data": data_dir,
        "artifact": artifact_dir,
    }
