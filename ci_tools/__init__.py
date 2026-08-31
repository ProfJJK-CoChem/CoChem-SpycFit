"""CoChem-SpycFit CI Tools Package (ci_tools/__init__.py).

Continuous integration, air-gap enforcement, anti-spoofing verification,
and zero-trust quarantine runner utilities for CoChem-SpycFit pipelines
across Windows, Linux, and macOS platforms.

Complies with Method Matrix v4 and CoChem Anti-Spoofing Protocol v2.
"""

from __future__ import annotations

from .anti_spoof_linter import (
    check_script,
    run_linter,
    load_amnesty_list,
    is_file_exempt,
)
from .verify_core_integrity import (
    HashringModel,
    calculate_hash,
    generate_hashring,
    get_core_dirs,
    get_hashring_file,
    get_root_dir,
    verify_hashring,
)
from .zero_trust_runner import (
    QuarantineEnvironment,
    QuarantineResult,
    run_in_quarantine,
    sweep_zombie_processes,
)

__all__ = [
    "check_script",
    "run_linter",
    "load_amnesty_list",
    "is_file_exempt",
    "HashringModel",
    "QuarantineEnvironment",
    "QuarantineResult",
    "calculate_hash",
    "generate_hashring",
    "get_core_dirs",
    "get_hashring_file",
    "get_root_dir",
    "run_in_quarantine",
    "sweep_zombie_processes",
    "verify_hashring",
]

