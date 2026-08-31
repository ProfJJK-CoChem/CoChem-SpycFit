"""
CoChem-BASE: Scientific Computational Chemistry & Spectroscopy Ecosystem.

Package entrypoint providing runtime platform discovery, versioning semantics,
and core namespace initialization adhering strictly to Zero-Mock invariants.
"""

from __future__ import annotations

import sys
import platform
import os
from pathlib import Path
from typing import Dict, Any

__version__ = "4.0.0-PROD"
__author__ = "CoChem Swarm"
__license__ = "Apache-2.0"


def check_runtime_environment() -> Dict[str, Any]:
    """
    Introspect host runtime platform, Python version, and execution tier.
    
    Verifies Python >= 3.10 and categorizes host execution environment
    (Windows, Linux, Darwin, HPC, Codespaces).
    """
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"CoChem-BASE requires Python 3.10 or higher. Found {sys.version}"
        )

    system_name = platform.system()
    tier = "LOCAL_WORKSTATION"
    
    if os.environ.get("CODESPACES") == "true" or os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"):
        tier = "CODESPACES"
    elif "SLURM_JOB_ID" in os.environ or "PBS_JOBID" in os.environ or "LSB_JOBID" in os.environ:
        tier = "HPC_CLUSTER"
    elif os.environ.get("CI") == "true":
        tier = "CI_PIPELINE"

    return {
        "version": __version__,
        "python_version": platform.python_version(),
        "platform": system_name,
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "tier": tier,
        "is_64bit": sys.maxsize > 2**32,
    }


__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "check_runtime_environment",
]
