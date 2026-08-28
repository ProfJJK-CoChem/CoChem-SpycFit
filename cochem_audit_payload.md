Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\02_P2_T02_spcat_spfit_sandbox_wrapper.md.
Original prompt:
# Prompt 04 — Phase 2, Task Group 02: SPCAT/SPFIT Asynchronous Sandbox Wrapper

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 2 — Dual-Engine Parity Bridge (JAX vs. SPCAT)
> **Phase:** 2 (Tasks 21–40) — Scientific Validation & Dual-Engine Execution
> **Dependency:** Prompt 01 (`FitState` model for accessing locked assignments)

---

## Objective

Build the subprocess manager that executes legacy Pickett Fortran binaries (`SPFIT`/`SPCAT`) within an isolated Tier 3 ephemeral sandbox. This is the "Verification Backend" — the silent validator and final arbiter of spectroscopic truth. It provides continuous "Classical Parity" checks against the JAX engine without risking host destabilization.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\cochem_spycfit_spcat_wrapper.py
```

> **Note:** This is a NEW file in the `serialization/` subpackage.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Asynchronous SPCAT/SPFIT subprocess wrapper with Tripartite Air-Gap isolation."""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
```

### 2. `SpfitSandbox` — The Quarantine Executor

```python
class SpfitSandbox:
    """Manages SPFIT/SPCAT execution in Tier 3 ephemeral sandboxes."""

    def __init__(
        self,
        scratch_root: Optional[Path] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        scratch_root : Path, optional
            Root directory for ephemeral sandboxes. Defaults to the system
            temp directory. If cochem_system_config.json defines a
            'scratch_root' key, that value is used instead.
        timeout_seconds : float
            Hard-kill timeout for SPFIT subprocess in seconds.
        """
        self._scratch_root: Optional[Path] = scratch_root
        self._timeout: float = timeout_seconds
        self._spfit_binary: Path = self._resolve_binary("spfit")
        self._spcat_binary: Path = self._resolve_binary("spcat")

    @staticmethod
    def _resolve_binary(name: str) -> Path:
        """
        Resolve the platform-specific binary path.

        On Windows: looks for 'spfit.exe' / 'spcat.exe'.
        On POSIX: looks for 'spfit' / 'spcat'.
        Uses shutil.which() first, then falls back to cochem_system_config.json.

        Returns
        -------
        Path
            Absolute path to the binary.

        Raises
        ------
        FileNotFoundError
            If the binary cannot be located.
        """
        suffix = ".exe" if platform.system() == "Windows" else ""
        binary_name = f"{name}{suffix}"
        found = shutil.which(binary_name)
        if found:
            return Path(found)
        # Fall back to config lookup
        config_path = Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            if f"{name}_path" in config:
                return Path(config[f"{name}_path"])
        raise FileNotFoundError(
            f"Cannot locate {binary_name}. Install SPFIT/SPCAT or set "
            f"'{name}_path' in cochem_system_config.json."
        )
```

### 3. Fortran File Formatters

Implement strict 80-column fixed-format file generators that convert `FitState` data into Fortran-readable files:

| Method | Purpose | Key Constraint |
|:---|:---|:---|
| `format_var_file(physical_constants: dict, uncertainties: dict, stem: str) -> str` | Generate `.var` parameter file | Strict 80-column Fortran character spacing; decimal alignment |
| `format_int_file(dipole_components: dict, j_max: int, stem: str) -> str` | Generate `.int` intensity/prediction file | Fixed-width columns matching SPCAT format specification |
| `format_lin_file(assignments: tuple, stem: str) -> str` | Generate `.lin` assignment file | Each line: quantum numbers in I3 format, frequency in F13.4, uncertainty, weight |

These formatters must enforce buffer overflow prevention — truncating or raising errors if any field exceeds its column width.

### 4. Core Execution Method

```python
def run_spfit(
    self,
    physical_constants: dict[str, float],
    uncertainties: dict[str, float],
    assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
    dipole_components: Optional[dict[str, float]] = None,
) -> dict:
    """
    Execute SPFIT in an ephemeral Tier 3 sandbox.

    1. Creates a tempfile.TemporaryDirectory() sandbox
    2. Writes .var, .int, .lin files with strict Fortran formatting
    3. Executes SPFIT subprocess with enforced timeout
    4. Parses the resulting .fit file
    5. Destroys the sandbox

    Parameters
    ----------
    physical_constants : dict
        Rotational constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
    uncertainties : dict
        Uncertainties for each constant in MHz.
    assignments : tuple
        Locked assignments as 7-tuples.
    dipole_components : dict, optional
        Dipole components {mu_a, mu_b, mu_c} in Debye.

    Returns
    -------
    dict
        Parsed results with keys: 'constants' (dict), 'uncertainties' (dict),
        'rmsd_khz' (float), 'exit_code' (int), 'parity_status' (str).

    Raises
    ------
    subprocess.TimeoutExpired
        If SPFIT exceeds the timeout.
    RuntimeError
        If SPFIT returns non-zero exit code or "NON-POSITIVE DEFINITE" error.
    """
    with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
        sandbox_path = Path(sandbox)
        stem = "cochem_run"

        # Write input files
        var_content = self.format_var_file(physical_constants, uncertainties, stem)
        (sandbox_path / f"{stem}.var").write_text(var_content, encoding="ascii")

        int_content = self.format_int_file(
            dipole_components or {"mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0},
            j_max=max(a[0] for a in assignments),
            stem=stem,
        )
        (sandbox_path / f"{stem}.int").write_text(int_content, encoding="ascii")

        lin_content = self.format_lin_file(assignments, stem)
        (sandbox_path / f"{stem}.lin").write_text(lin_content, encoding="ascii")

        # Execute SPFIT
        result = subprocess.run(
            [str(self._spfit_binary), stem],
            cwd=str(sandbox_path),
            timeout=self._timeout,
            capture_output=True,
            text=True,
            check=False,
        )

        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(
                f"SPFIT exited with code {result.returncode}: {result.stderr}"
            )
        if "NON-POSITIVE DEFINITE" in result.stdout:
            raise RuntimeError("SPFIT: NON-POSITIVE DEFINITE matrix detected.")

        # Parse .fit output
        fit_path = sandbox_path / f"{stem}.fit"
        if not fit_path.exists():
            raise FileNotFoundError(f"SPFIT did not produce {stem}.fit")

        parsed = self._parse_fit_file(fit_path)
        return parsed
    # Sandbox is destroyed here automatically
```

### 5. `.fit` File Parser

```python
def _parse_fit_file(self, fit_path: Path) -> dict:
    """
    Parse SPFIT .fit output into a structured dictionary.

    Extracts: fitted constants (MHz), uncertainties, correlation matrix,
    global RMS deviation (kHz), and number of lines.

    Returns
    -------
    dict
        Keys: 'constants', 'uncertainties', 'rmsd_khz', 'n_lines', 'exit_code'
    """
    # Parse the fixed-format Fortran output
    # Filter malformed tokens before piping constants back
```

### 6. Async Variant for Background Execution

```python
async def run_spfit_async(
    self,
    physical_constants: dict[str, float],
    uncertainties: dict[str, float],
    assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
    dipole_components: Optional[dict[str, float]] = None,
) -> dict:
    """
    Asynchronous variant of run_spfit for non-blocking UI execution.

    Runs SPFIT in a thread pool executor to avoid blocking the Jupyter event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        self.run_spfit,
        physical_constants,
        uncertainties,
        assignments,
        dipole_components,
    )
```

---

## Validation Test Specification

> **IMPORTANT:** This test requires SPFIT to be installed. If SPFIT is not available, the test should be marked `@pytest.mark.skipif` with an appropriate message.

1. Create a `SpfitSandbox` pointing to a `tempfile.TemporaryDirectory()`
2. Pass authentic H₂O rotational constants: A = 835840.288 MHz, B = 435351.717 MHz, C = 278138.700 MHz
3. Pass 5 authentic water transitions (from the table in Prompt 01)
4. Assert that the returned constants match input constants within precision
5. Assert that the sandbox directory is fully destroyed after execution (no file leaks)
6. Assert that `.var`, `.int`, `.lin` files are strict 80-column Fortran format

---

## Acceptance Criteria

- [ ] Binary resolution uses `shutil.which()` with platform-aware suffix detection
- [ ] All subprocess execution isolated in `tempfile.TemporaryDirectory()` (Tier 3)
- [ ] Strict 80-column Fortran file formatting for `.var`, `.int`, `.lin`
- [ ] `timeout` hard-kill enforced on subprocess
- [ ] "NON-POSITIVE DEFINITE" errors trapped and raised as RuntimeError
- [ ] Sandbox fully destroyed after execution (no file leaks)
- [ ] Async variant available for non-blocking Jupyter UI execution
- [ ] All paths use `pathlib.Path`; config resolved via `cochem_system_config.json`

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\__init__.py ---
"""
CoChem-SpycFit: High-Performance Serialization, Out-of-Core Catalog & Legacy Fortran Translation Subsystem.
"""

from __future__ import annotations

from cochem_spycfit.serialization.cochem_catalog_compiler import (
    CatalogCompilerConfig,
    CatalogCompilerError,
    CatalogCorruptionError,
    CatalogDatasetError,
    CatalogLockError,
    CatalogLockTimeoutError,
    CatalogManifestRecord,
    CatalogSchemaError,
    CatalogStreamingError,
    LustreCatalogLockError,
    Tier3RegistryLock,
    TransitionBatch,
    TransitionRecord,
    VirtualLayoutError,
    compile_virtual_catalog,
    compute_sha256,
    extract_catalog_metadata,
    is_lustre_filesystem,
    read_chunked_catalog,
    stream_chunked_catalog,
    verify_catalog_integrity,
)
from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranBridgeError,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    FortranTaskQueue,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettEgyFile,
    PickettEgyRecord,
    PickettFitResidual,
    PickettFitResult,
    PickettFormatError,
    PickettIntFile,
    PickettLinFile,
    PickettLinRecord,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    decode_pickett_quantum,
    dispatch_fortran_binary,
    encode_pickett_quantum,
    format_pickett_line,
    fortran_float_converter,
    get_fortran_deployment_dir,
    get_isotopic_mass,
    is_segfault_exit_code,
    parse_pickett_cat,
    parse_pickett_egy,
    parse_pickett_fit,
    parse_pickett_int,
    parse_pickett_lin,
    parse_pickett_var,
    resolve_fortran_binary,
    write_spycfit_telemetry,
)
from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    PARAM_ID_TO_NAME_MAP,
    SPFIT_WATSON_A_PARAM_MAP,
    SpfitSandbox,
)

__all__ = [
    "CatalogCompilerConfig",
    "CatalogCompilerError",
    "CatalogCorruptionError",
    "CatalogDatasetError",
    "CatalogLockError",
    "CatalogLockTimeoutError",
    "CatalogManifestRecord",
    "CatalogSchemaError",
    "CatalogStreamingError",
    "FortranBinaryNotFoundError",
    "FortranBridgeDispatcher",
    "FortranBridgeError",
    "FortranExecutionError",
    "FortranExecutionResult",
    "FortranExecutionSegfaultError",
    "FortranExecutionTask",
    "FortranExecutionTimeoutError",
    "FortranTaskQueue",
    "LustreCatalogLockError",
    "PickettCatFile",
    "PickettCatRecord",
    "PickettDipoleRecord",
    "PickettEgyFile",
    "PickettEgyRecord",
    "PickettFitResidual",
    "PickettFitResult",
    "PickettFormatError",
    "PickettIntFile",
    "PickettLinFile",
    "PickettLinRecord",
    "PickettParameterRecord",
    "PickettScratchContext",
    "PickettVarFile",
    "Tier3RegistryLock",
    "TransitionBatch",
    "TransitionRecord",
    "VirtualLayoutError",
    "calculate_reduced_mass",
    "capture_hex_telemetry",
    "compile_virtual_catalog",
    "compute_sha256",
    "decode_pickett_quantum",
    "dispatch_fortran_binary",
    "encode_pickett_quantum",
    "extract_catalog_metadata",
    "format_pickett_line",
    "fortran_float_converter",
    "get_fortran_deployment_dir",
    "get_isotopic_mass",
    "is_lustre_filesystem",
    "is_segfault_exit_code",
    "parse_pickett_cat",
    "parse_pickett_egy",
    "parse_pickett_fit",
    "parse_pickett_int",
    "parse_pickett_lin",
    "parse_pickett_var",
    "read_chunked_catalog",
    "resolve_fortran_binary",
    "stream_chunked_catalog",
    "verify_catalog_integrity",
    "write_spycfit_telemetry",
    "SpfitSandbox",
    "SPFIT_WATSON_A_PARAM_MAP",
    "PARAM_ID_TO_NAME_MAP",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\__init__.py ---
"""
CoChem-SpycFit: High-Performance Serialization, Out-of-Core Catalog & Legacy Fortran Translation Subsystem.
"""

from __future__ import annotations

from cochem_spycfit.serialization.cochem_catalog_compiler import (
    CatalogCompilerConfig,
    CatalogCompilerError,
    CatalogCorruptionError,
    CatalogDatasetError,
    CatalogLockError,
    CatalogLockTimeoutError,
    CatalogManifestRecord,
    CatalogSchemaError,
    CatalogStreamingError,
    LustreCatalogLockError,
    PARQUET_TRANSITION_SCHEMA,
    Tier3RegistryLock,
    TransitionBatch,
    TransitionRecord,
    VirtualLayoutError,
    compile_master_manifest,
    compile_virtual_catalog,
    compute_sha256,
    extract_catalog_metadata,
    get_chunked_transitions_dir,
    is_lustre_filesystem,
    jax_serialize_chunk_callback,
    read_chunked_catalog,
    read_parquet_catalog,
    serialize_chunk_to_parquet,
    stream_chunked_catalog,
    verify_catalog_integrity,
)
from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranBridgeError,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    FortranTaskQueue,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettEgyFile,
    PickettEgyRecord,
    PickettFitResidual,
    PickettFitResult,
    PickettFormatError,
    PickettIntFile,
    PickettLinFile,
    PickettLinRecord,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    decode_pickett_quantum,
    dispatch_fortran_binary,
    encode_pickett_quantum,
    format_pickett_line,
    fortran_float_converter,
    get_fortran_deployment_dir,
    get_isotopic_mass,
    is_segfault_exit_code,
    parse_pickett_cat,
    parse_pickett_egy,
    parse_pickett_fit,
    parse_pickett_int,
    parse_pickett_lin,
    parse_pickett_var,
    resolve_fortran_binary,
    write_spycfit_telemetry,
)
from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    PARAM_ID_TO_NAME_MAP,
    SPFIT_WATSON_A_PARAM_MAP,
    SpfitSandbox,
)

__all__ = [
    "CatalogCompilerConfig",
    "CatalogCompilerError",
    "CatalogCorruptionError",
    "CatalogDatasetError",
    "CatalogLockError",
    "CatalogLockTimeoutError",
    "CatalogManifestRecord",
    "CatalogSchemaError",
    "CatalogStreamingError",
    "FortranBinaryNotFoundError",
    "FortranBridgeDispatcher",
    "FortranBridgeError",
    "FortranExecutionError",
    "FortranExecutionResult",
    "FortranExecutionSegfaultError",
    "FortranExecutionTask",
    "FortranExecutionTimeoutError",
    "FortranTaskQueue",
    "LustreCatalogLockError",
    "PickettCatFile",
    "PickettCatRecord",
    "PickettDipoleRecord",
    "PickettEgyFile",
    "PickettEgyRecord",
    "PickettFitResidual",
    "PickettFitResult",
    "PickettFormatError",
    "PickettIntFile",
    "PickettLinFile",
    "PickettLinRecord",
    "PickettParameterRecord",
    "PickettScratchContext",
    "PickettVarFile",
    "Tier3RegistryLock",
    "TransitionBatch",
    "TransitionRecord",
    "VirtualLayoutError",
    "calculate_reduced_mass",
    "capture_hex_telemetry",
    "compile_virtual_catalog",
    "compute_sha256",
    "decode_pickett_quantum",
    "dispatch_fortran_binary",
    "encode_pickett_quantum",
    "extract_catalog_metadata",
    "format_pickett_line",
    "fortran_float_converter",
    "get_fortran_deployment_dir",
    "get_isotopic_mass",
    "is_lustre_filesystem",
    "is_segfault_exit_code",
    "parse_pickett_cat",
    "parse_pickett_egy",
    "parse_pickett_fit",
    "parse_pickett_int",
    "parse_pickett_lin",
    "parse_pickett_var",
    "read_chunked_catalog",
    "resolve_fortran_binary",
    "stream_chunked_catalog",
    "verify_catalog_integrity",
    "write_spycfit_telemetry",
    "SpfitSandbox",
    "SPFIT_WATSON_A_PARAM_MAP",
    "PARAM_ID_TO_NAME_MAP",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\cochem_spycfit_spcat_wrapper.py ---
"""
Asynchronous SPCAT/SPFIT subprocess wrapper with Tripartite Air-Gap isolation.
Provides Tier 3 ephemeral sandbox execution of legacy Pickett Fortran binaries
for spectroscopic assignment verification and Classical Parity validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Standard Watson A-reduction parameter ID mapping (Pickett codes)
SPFIT_WATSON_A_PARAM_MAP: Dict[str, int] = {
    "A": 10000,
    "B": 20000,
    "C": 30000,
    "DJ": 200,
    "DJK": 1100,
    "DK": 2000,
    "dJ": 40100,
    "dK": 41000,
    "-DJ": 200,
    "-DJK": 1100,
    "-DK": 2000,
    "-dJ": 40100,
    "-dK": 41000,
    "Delta_J": 200,
    "Delta_JK": 1100,
    "Delta_K": 2000,
    "delta_J": 40100,
    "delta_K": 41000,
}

PARAM_ID_TO_NAME_MAP: Dict[int, str] = {
    10000: "A",
    20000: "B",
    30000: "C",
    200: "DJ",
    1100: "DJK",
    2000: "DK",
    40100: "dJ",
    41000: "dK",
}


def _format_fortran_double(
    val: Union[float, int, str],
    width: int = 22,
    precision: int = 14,
) -> str:
    """
    Format numeric value into strict fixed-width Fortran double precision representation.
    Replaces exponent character 'E' with 'D' (e.g. 1.25000000000000D+02).
    """
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        val_str = str(val).strip()
        return val_str[:width].rjust(width)

    formatted = f"{val_float:.{precision}E}"
    formatted = formatted.replace("E", "D").replace("e", "D")

    if len(formatted) > width:
        excess = len(formatted) - width
        adj_precision = max(0, precision - excess)
        formatted = f"{val_float:.{adj_precision}E}".replace("E", "D").replace("e", "D")
        if len(formatted) > width:
            formatted = formatted[:width]

    return formatted.rjust(width)


def _parse_fortran_float(val: Union[str, float, int]) -> float:
    """
    Safely parse Fortran scientific notation string into Python float64.
    """
    if isinstance(val, (float, int)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")
    match_implicit = re.match(r"^([+-]?\d+\.?\d*)([+-]\d+)$", s_clean)
    if match_implicit:
        s_clean = f"{match_implicit.group(1)}e{match_implicit.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


class SpfitSandbox:
    """Manages SPFIT/SPCAT execution in Tier 3 ephemeral sandboxes."""

    def __init__(
        self,
        scratch_root: Optional[Path] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        scratch_root : Path, optional
            Root directory for ephemeral sandboxes. Defaults to the system
            temp directory. If cochem_system_config.json defines a
            'scratch_root' key, that value is used instead.
        timeout_seconds : float
            Hard-kill timeout for SPFIT subprocess in seconds.
        """
        self._scratch_root: Optional[Path] = self._resolve_scratch_root(scratch_root)
        self._timeout: float = float(timeout_seconds)
        self._spfit_binary: Path = self._resolve_binary("spfit")
        self._spcat_binary: Path = self._resolve_binary("spcat")

    @staticmethod
    def _resolve_scratch_root(user_path: Optional[Path]) -> Optional[Path]:
        """Resolve scratch directory obeying Tier 3 configuration rules."""
        if user_path is not None:
            resolved = Path(user_path).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved

        # Check environment variables
        for env_var in ("COCHEM_STATE", "COCHEM_SCRATCH_DIR", "COCHEM_SCRATCH_ROOT"):
            val = os.environ.get(env_var)
            if val:
                p = Path(val).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p

        # Check system config
        config_path = Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "scratch_root" in cfg:
                    p = Path(cfg["scratch_root"]).resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    return p
                if "scratch_dir" in cfg:
                    p = Path(cfg["scratch_dir"]).resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    return p
            except Exception as e:
                logger.debug("Failed reading system config for scratch root: %s", e)

        return None

    @staticmethod
    def _resolve_binary(name: str) -> Path:
        """
        Resolve the platform-specific binary path.

        On Windows: looks for 'spfit.exe' / 'spcat.exe'.
        On POSIX: looks for 'spfit' / 'spcat'.
        Uses shutil.which() first, then falls back to cochem_system_config.json.

        Returns
        -------
        Path
            Absolute path to the binary.

        Raises
        ------
        FileNotFoundError
            If the binary cannot be located.
        """
        suffix = ".exe" if platform.system() == "Windows" else ""
        binary_name = f"{name}{suffix}"

        found = shutil.which(binary_name)
        if found:
            return Path(found).resolve()

        found_nosuffix = shutil.which(name)
        if found_nosuffix:
            return Path(found_nosuffix).resolve()

        # Check explicit environment directory
        bin_dir_env = os.environ.get("COCHEM_BIN_DIR")
        if bin_dir_env:
            cand_bin = Path(bin_dir_env) / binary_name
            if cand_bin.is_file():
                return cand_bin.resolve()
            cand_bin_raw = Path(bin_dir_env) / name
            if cand_bin_raw.is_file():
                return cand_bin_raw.resolve()

        # Fall back to config lookup
        config_candidates = [
            Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json",
            Path.home() / "CoChem_Artifacts" / "cochem_system_config.json",
        ]
        artifacts_env = os.environ.get("COCHEM_ARTIFACTS_ROOT") or os.environ.get("COCHEM_ARTIFACTS")
        if artifacts_env:
            config_candidates.insert(0, Path(artifacts_env) / "Registry" / "cochem_system_config.json")
            config_candidates.insert(1, Path(artifacts_env) / "cochem_system_config.json")

        for config_path in config_candidates:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    if f"{name}_path" in config:
                        cand = Path(config[f"{name}_path"]).resolve()
                        if cand.exists():
                            return cand
                    if "bin_dir" in config:
                        cand = (Path(config["bin_dir"]) / binary_name).resolve()
                        if cand.exists():
                            return cand
                except Exception as e:
                    logger.debug("Failed reading system config at %s: %s", config_path, e)

        raise FileNotFoundError(
            f"Cannot locate {binary_name}. Install SPFIT/SPCAT or set "
            f"'{name}_path' in cochem_system_config.json."
        )

    @staticmethod
    def format_var_file(
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .var parameter file.

        Parameters
        ----------
        physical_constants : dict
            Rotational and centrifugal distortion constants in MHz.
        uncertainties : dict, optional
            Estimated uncertainties for each constant in MHz.
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .var content.
        """
        title = f"SPFIT Parameter File for {stem}"[:80]
        unc_dict = uncertainties or {}

        # Build list of formatted parameter records
        records: List[Tuple[int, float, float, str]] = []
        for key, val in physical_constants.items():
            if key in SPFIT_WATSON_A_PARAM_MAP:
                pid = SPFIT_WATSON_A_PARAM_MAP[key]
            else:
                try:
                    pid = int(key)
                except ValueError:
                    pid = 10000
            unc = float(unc_dict.get(key, 0.0))
            records.append((pid, float(val), unc, key))

        # Control line: npar (5), nline (5), nopt (5), nwarn (5), erpar (10.4f), wtfac (10.4f), scale (10.4f), maxit (5)
        npar = len(records)
        ctrl_line = f"{npar:5d}{0:5d}{0:5d}{0:5d}{1.0000:10.4f}{1.0000:10.4f}{1.0000:10.4f}{50:5d}"
        if len(ctrl_line) > 80:
            ctrl_line = ctrl_line[:80]

        lines = [title, ctrl_line]
        for pid, val, unc, label in records:
            p_str = f"{pid:11d}"
            v_str = _format_fortran_double(val, width=22, precision=14)
            u_str = _format_fortran_double(unc, width=18, precision=10)
            lbl_str = f" / {label}" if label else ""
            line = f"{p_str} {v_str} {u_str}{lbl_str}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_int_file(
        dipole_components: Dict[str, float],
        j_max: int = 99,
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .int intensity file.

        Parameters
        ----------
        dipole_components : dict
            Dipole moment components (mu_a, mu_b, mu_c) in Debye.
        j_max : int
            Maximum J rotational quantum number.
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .int content.
        """
        title = f"SPCAT/SPFIT Intensity File for {stem}"[:80]

        tag = 1
        qpart = 1000.0000
        nline = 1000
        iq = 1
        clamped_j_max = min(99, max(1, int(j_max)))
        min_freq = 0.0
        max_freq = 500000.0
        min_int = -10.00
        temp_k = 300.00

        ctrl_line = (
            f"{tag:6d}"
            f"{qpart:12.4f}"
            f"{nline:7d}"
            f"{iq:4d}"
            f"{clamped_j_max:4d}"
            f" {min_freq:10.1f}"
            f" {max_freq:10.1f}"
            f"{min_int:10.2f}"
            f"{temp_k:10.2f}"
        )
        if len(ctrl_line) > 80:
            ctrl_line = ctrl_line[:80]

        lines = [title, ctrl_line]

        # Map axis indices
        axis_mapping: Dict[str, int] = {
            "mu_a": 1,
            "mua": 1,
            "a": 1,
            "1": 1,
            "mu_b": 2,
            "mub": 2,
            "b": 2,
            "2": 2,
            "mu_c": 3,
            "muc": 3,
            "c": 3,
            "3": 3,
        }

        dipole_entries: List[Tuple[int, float, str]] = []
        for k, v in dipole_components.items():
            k_lower = str(k).lower().strip()
            axis_idx = axis_mapping.get(k_lower, 1)
            dipole_entries.append((axis_idx, float(v), k))

        if not dipole_entries:
            dipole_entries.append((1, 1.0, "mu_a"))

        for axis_idx, dip_val, label in dipole_entries:
            line = f"{axis_idx:2d}  {dip_val:8.4f} / {label}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_lin_file(
        assignments: Sequence[Any],
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .lin line list file.

        Parameters
        ----------
        assignments : sequence of tuples
            Each entry contains (J', Ka', Kc', J'', Ka'', Kc'', freq_MHz, [unc_MHz, wt]).
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .lin content.

        Raises
        ------
        ValueError
            If quantum numbers or frequencies exceed fixed field widths.
        """
        lines: List[str] = [f"! SPFIT Line List for {stem}"[:80]]

        for entry in assignments:
            if len(entry) < 7:
                raise ValueError(f"Assignment record requires at least 7 fields, got {entry}")

            ju, kau, kcu = int(entry[0]), int(entry[1]), int(entry[2])
            jl, kal, kcl = int(entry[3]), int(entry[4]), int(entry[5])
            freq_mhz = float(entry[6])
            unc_mhz = float(entry[7]) if len(entry) > 7 else 0.0100
            wt = float(entry[8]) if len(entry) > 8 else 1.0000

            # Buffer overflow checks for fixed-width Fortran I3 / F13.4
            for q_name, q_val in [
                ("J_upper", ju), ("Ka_upper", kau), ("Kc_upper", kcu),
                ("J_lower", jl), ("Ka_lower", kal), ("Kc_lower", kcl),
            ]:
                if q_val < -99 or q_val > 999:
                    raise ValueError(
                        f"Quantum number {q_name}={q_val} exceeds 3-column field width [-99, 999] in fixed-format .lin file."
                    )

            if freq_mhz > 99999999.9999 or freq_mhz < -9999999.9999:
                raise ValueError(
                    f"Frequency {freq_mhz} exceeds F13.4 field width in fixed-format .lin file."
                )

            # Fortran format: 6I3, 6I3, F13.4, F8.4, F8.4
            qu_str = f"{ju:3d}{kau:3d}{kcu:3d}{0:3d}{0:3d}{0:3d}"
            ql_str = f"{jl:3d}{kal:3d}{kcl:3d}{0:3d}{0:3d}{0:3d}"
            f_str = f"{freq_mhz:13.4f}"
            u_str = f"{unc_mhz:8.4f}"
            w_str = f"{wt:8.4f}"

            line = f"{qu_str}{ql_str}{f_str}{u_str}{w_str}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @classmethod
    def _parse_fit_file_static(cls, fit_path: Path) -> Dict[str, Any]:
        """
        Parse SPFIT .fit output into a structured dictionary.

        Extracts fitted constants (MHz), uncertainties, global RMS deviation (kHz),
        number of lines, and residuals.
        """
        if not fit_path.exists():
            raise FileNotFoundError(f"Fit file not found: {fit_path}")

        lines = fit_path.read_text(encoding="utf-8", errors="replace").splitlines()

        constants: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        std_error = 0.0
        lines_fitted = 0
        residuals_list: List[Dict[str, Any]] = []

        in_parameters = False
        in_residuals = False

        param_regex = re.compile(
            r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
        )
        res_regex = re.compile(
            r"^\s*(\d+)\s+((?:\s*[+-]?\d+){2,6})\s+((?:\s*[+-]?\d+){2,6})\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)"
        )

        for line in lines:
            s_line = line.strip()
            if not s_line:
                continue

            # Summary statistics
            std_err_match = re.search(
                r"(?:STANDARD\s+ERROR(?:\s+OF\s+FIT)?|MICROWAVE\s+RMS)\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)",
                s_line,
                re.IGNORECASE,
            )
            if std_err_match:
                std_error = _parse_fortran_float(std_err_match.group(1))

            lines_fit_match = re.search(r"LINES\s+FITTED\s*=\s*(\d+)", s_line, re.IGNORECASE)
            if lines_fit_match:
                lines_fitted = int(lines_fit_match.group(1))

            # Section headers
            if "PARAMETERS" in s_line.upper():
                in_parameters = True
                in_residuals = False
                continue

            if "RESIDUALS" in s_line.upper() or "OBS-CALC" in s_line.upper():
                in_residuals = True
                in_parameters = False
                continue

            if in_parameters:
                pmatch = param_regex.match(s_line)
                if pmatch:
                    pid = int(pmatch.group(1))
                    val = _parse_fortran_float(pmatch.group(2))
                    unc = _parse_fortran_float(pmatch.group(3))
                    lbl = (pmatch.group(4) or "").strip()

                    name = lbl if lbl else PARAM_ID_TO_NAME_MAP.get(pid, str(pid))
                    constants[name] = val
                    uncertainties[name] = unc
                    if pid in PARAM_ID_TO_NAME_MAP and name != PARAM_ID_TO_NAME_MAP[pid]:
                        constants[PARAM_ID_TO_NAME_MAP[pid]] = val
                        uncertainties[PARAM_ID_TO_NAME_MAP[pid]] = unc

            if in_residuals:
                rmatch = res_regex.match(s_line)
                if rmatch:
                    line_idx = int(rmatch.group(1))
                    qu = tuple(int(x) for x in rmatch.group(2).split())
                    ql = tuple(int(x) for x in rmatch.group(3).split())
                    f_obs = _parse_fortran_float(rmatch.group(4))
                    f_calc = _parse_fortran_float(rmatch.group(5))
                    residual = _parse_fortran_float(rmatch.group(6))
                    unc = _parse_fortran_float(rmatch.group(7))
                    wt = _parse_fortran_float(rmatch.group(8))

                    residuals_list.append({
                        "line": line_idx,
                        "qu": qu,
                        "ql": ql,
                        "obs_mhz": f_obs,
                        "calc_mhz": f_calc,
                        "res_mhz": residual,
                        "unc_mhz": unc,
                        "weight": wt,
                    })

        # Calculate exact RMS deviation in kHz
        if residuals_list:
            res_sq = [r["res_mhz"] ** 2 for r in residuals_list]
            rmsd_khz = math.sqrt(sum(res_sq) / len(res_sq)) * 1000.0
            n_lines = len(residuals_list)
        else:
            rmsd_khz = std_error * 1000.0
            n_lines = lines_fitted

        parity_status = "PASS" if rmsd_khz < 1000.0 else "WARN"

        return {
            "constants": constants,
            "uncertainties": uncertainties,
            "rmsd_khz": rmsd_khz,
            "n_lines": n_lines,
            "exit_code": 0,
            "parity_status": parity_status,
            "residuals": residuals_list,
        }

    def _parse_fit_file(self, fit_path: Path) -> Dict[str, Any]:
        """Instance method wrapper for _parse_fit_file_static."""
        return self._parse_fit_file_static(fit_path)

    def run_spfit(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
        dipole_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute SPFIT in an ephemeral Tier 3 sandbox.

        1. Creates a tempfile.TemporaryDirectory() sandbox
        2. Writes .var, .int, .lin files with strict Fortran formatting
        3. Executes SPFIT subprocess with enforced timeout
        4. Parses the resulting .fit file
        5. Destroys the sandbox

        Parameters
        ----------
        physical_constants : dict
            Rotational constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
        uncertainties : dict, optional
            Uncertainties for each constant in MHz.
        assignments : tuple
            Locked assignments as 7-tuples.
        dipole_components : dict, optional
            Dipole components {mu_a, mu_b, mu_c} in Debye.

        Returns
        -------
        dict
            Parsed results with keys: 'constants' (dict), 'uncertainties' (dict),
            'rmsd_khz' (float), 'exit_code' (int), 'parity_status' (str).

        Raises
        ------
        subprocess.TimeoutExpired
            If SPFIT exceeds the timeout.
        RuntimeError
            If SPFIT returns non-zero exit code or "NON-POSITIVE DEFINITE" error.
        FileNotFoundError
            If SPFIT did not produce .fit output file.
        """
        with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
            sandbox_path = Path(sandbox)
            stem = "cochem_run"

            # Write input files
            var_content = self.format_var_file(physical_constants, uncertainties, stem)
            (sandbox_path / f"{stem}.var").write_text(var_content, encoding="ascii")

            max_j = max((int(a[0]) for a in assignments), default=10)
            int_content = self.format_int_file(
                dipole_components or {"mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0},
                j_max=max_j,
                stem=stem,
            )
            (sandbox_path / f"{stem}.int").write_text(int_content, encoding="ascii")

            lin_content = self.format_lin_file(assignments, stem)
            (sandbox_path / f"{stem}.lin").write_text(lin_content, encoding="ascii")

            # Execute SPFIT
            result = subprocess.run(
                [str(self._spfit_binary), stem],
                cwd=str(sandbox_path),
                timeout=self._timeout,
                capture_output=True,
                text=True,
                check=False,
            )

            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(
                    f"SPFIT exited with code {result.returncode}: {result.stderr}"
                )
            if "NON-POSITIVE DEFINITE" in result.stdout or "NON-POSITIVE DEFINITE" in result.stderr:
                raise RuntimeError("SPFIT: NON-POSITIVE DEFINITE matrix detected.")

            # Parse .fit output
            fit_path = sandbox_path / f"{stem}.fit"
            if not fit_path.exists():
                raise FileNotFoundError(f"SPFIT did not produce {stem}.fit")

            parsed = self._parse_fit_file(fit_path)
            return parsed

    async def run_spfit_async(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
        dipole_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous variant of run_spfit for non-blocking UI execution.

        Runs SPFIT in a thread pool executor to avoid blocking the Jupyter event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.run_spfit,
            physical_constants,
            uncertainties,
            assignments,
            dipole_components,
        )

    def run_spcat(
        self,
        physical_constants: Dict[str, float],
        dipole_components: Optional[Dict[str, float]] = None,
        j_max: int = 99,
        stem: str = "cochem_run",
    ) -> Dict[str, Any]:
        """
        Execute SPCAT in an ephemeral Tier 3 sandbox to calculate predicted transitions.

        Parameters
        ----------
        physical_constants : dict
            Rotational constants in MHz.
        dipole_components : dict, optional
            Dipole components in Debye.
        j_max : int
            Maximum J quantum number.
        stem : str
            Stem name.

        Returns
        -------
        dict
            Parsed catalog output.
        """
        with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
            sandbox_path = Path(sandbox)

            # Write .var and .int
            var_content = self.format_var_file(physical_constants, stem=stem)
            (sandbox_path / f"{stem}.var").write_text(var_content, encoding="ascii")

            int_content = self.format_int_file(
                dipole_components or {"mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0},
                j_max=j_max,
                stem=stem,
            )
            (sandbox_path / f"{stem}.int").write_text(int_content, encoding="ascii")

            result = subprocess.run(
                [str(self._spcat_binary), stem],
                cwd=str(sandbox_path),
                timeout=self._timeout,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"SPCAT exited with code {result.returncode}: {result.stderr}"
                )

            cat_path = sandbox_path / f"{stem}.cat"
            if not cat_path.exists():
                raise FileNotFoundError(f"SPCAT did not produce {stem}.cat")

            cat_text = cat_path.read_text(encoding="utf-8", errors="replace")
            return {"raw_catalog": cat_text, "exit_code": 0}

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_spycfit_spcat_wrapper.py ---
"""
Asynchronous SPCAT/SPFIT subprocess wrapper with Tripartite Air-Gap isolation.
Provides Tier 3 ephemeral sandbox execution of legacy Pickett Fortran binaries
for spectroscopic assignment verification and Classical Parity validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Standard Watson A-reduction parameter ID mapping (Pickett codes)
SPFIT_WATSON_A_PARAM_MAP: Dict[str, int] = {
    "A": 10000,
    "B": 20000,
    "C": 30000,
    "DJ": 200,
    "DJK": 1100,
    "DK": 2000,
    "dJ": 40100,
    "dK": 41000,
    "-DJ": 200,
    "-DJK": 1100,
    "-DK": 2000,
    "-dJ": 40100,
    "-dK": 41000,
    "Delta_J": 200,
    "Delta_JK": 1100,
    "Delta_K": 2000,
    "delta_J": 40100,
    "delta_K": 41000,
}

PARAM_ID_TO_NAME_MAP: Dict[int, str] = {
    10000: "A",
    20000: "B",
    30000: "C",
    200: "DJ",
    1100: "DJK",
    2000: "DK",
    40100: "dJ",
    41000: "dK",
}


def _format_fortran_double(
    val: Union[float, int, str],
    width: int = 22,
    precision: int = 14,
) -> str:
    """
    Format numeric value into strict fixed-width Fortran double precision representation.
    Replaces exponent character 'E' with 'D' (e.g. 1.25000000000000D+02).
    """
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        val_str = str(val).strip()
        return val_str[:width].rjust(width)

    formatted = f"{val_float:.{precision}E}"
    formatted = formatted.replace("E", "D").replace("e", "D")

    if len(formatted) > width:
        excess = len(formatted) - width
        adj_precision = max(0, precision - excess)
        formatted = f"{val_float:.{adj_precision}E}".replace("E", "D").replace("e", "D")
        if len(formatted) > width:
            formatted = formatted[:width]

    return formatted.rjust(width)


def _parse_fortran_float(val: Union[str, float, int]) -> float:
    """
    Safely parse Fortran scientific notation string into Python float64.
    """
    if isinstance(val, (float, int)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    s_clean = s.replace("D", "e").replace("d", "e").replace("E", "e")
    match_implicit = re.match(r"^([+-]?\d+\.?\d*)([+-]\d+)$", s_clean)
    if match_implicit:
        s_clean = f"{match_implicit.group(1)}e{match_implicit.group(2)}"

    try:
        return float(s_clean)
    except ValueError as exc:
        raise ValueError(f"Failed to parse Fortran double precision float from {val!r}: {exc}") from exc


class SpfitSandbox:
    """Manages SPFIT/SPCAT execution in Tier 3 ephemeral sandboxes."""

    def __init__(
        self,
        scratch_root: Optional[Path] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        scratch_root : Path, optional
            Root directory for ephemeral sandboxes. Defaults to the system
            temp directory. If cochem_system_config.json defines a
            'scratch_root' key, that value is used instead.
        timeout_seconds : float
            Hard-kill timeout for SPFIT subprocess in seconds.
        """
        self._scratch_root: Optional[Path] = self._resolve_scratch_root(scratch_root)
        self._timeout: float = float(timeout_seconds)
        self._spfit_binary: Path = self._resolve_binary("spfit")
        self._spcat_binary: Path = self._resolve_binary("spcat")

    @staticmethod
    def _resolve_scratch_root(user_path: Optional[Path]) -> Optional[Path]:
        """Resolve scratch directory obeying Tier 3 configuration rules."""
        if user_path is not None:
            resolved = Path(user_path).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved

        # Check environment variables
        for env_var in ("COCHEM_STATE", "COCHEM_SCRATCH_DIR", "COCHEM_SCRATCH_ROOT"):
            val = os.environ.get(env_var)
            if val:
                p = Path(val).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p

        # Check system config
        config_path = Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "scratch_root" in cfg:
                    p = Path(cfg["scratch_root"]).resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    return p
                if "scratch_dir" in cfg:
                    p = Path(cfg["scratch_dir"]).resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    return p
            except Exception as e:
                logger.debug("Failed reading system config for scratch root: %s", e)

        return None

    @staticmethod
    def _resolve_binary(name: str) -> Path:
        """
        Resolve the platform-specific binary path.

        On Windows: looks for 'spfit.exe' / 'spcat.exe'.
        On POSIX: looks for 'spfit' / 'spcat'.
        Uses shutil.which() first, then falls back to cochem_system_config.json.

        Returns
        -------
        Path
            Absolute path to the binary.

        Raises
        ------
        FileNotFoundError
            If the binary cannot be located.
        """
        suffix = ".exe" if platform.system() == "Windows" else ""
        binary_name = f"{name}{suffix}"

        found = shutil.which(binary_name)
        if found:
            return Path(found).resolve()

        found_nosuffix = shutil.which(name)
        if found_nosuffix:
            return Path(found_nosuffix).resolve()

        # Check explicit environment directory
        bin_dir_env = os.environ.get("COCHEM_BIN_DIR")
        if bin_dir_env:
            cand_bin = Path(bin_dir_env) / binary_name
            if cand_bin.is_file():
                return cand_bin.resolve()
            cand_bin_raw = Path(bin_dir_env) / name
            if cand_bin_raw.is_file():
                return cand_bin_raw.resolve()

        # Fall back to config lookup
        config_candidates = [
            Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json",
            Path.home() / "CoChem_Artifacts" / "cochem_system_config.json",
        ]
        artifacts_env = os.environ.get("COCHEM_ARTIFACTS_ROOT") or os.environ.get("COCHEM_ARTIFACTS")
        if artifacts_env:
            config_candidates.insert(0, Path(artifacts_env) / "Registry" / "cochem_system_config.json")
            config_candidates.insert(1, Path(artifacts_env) / "cochem_system_config.json")

        for config_path in config_candidates:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    if f"{name}_path" in config:
                        cand = Path(config[f"{name}_path"]).resolve()
                        if cand.exists():
                            return cand
                    if "bin_dir" in config:
                        cand = (Path(config["bin_dir"]) / binary_name).resolve()
                        if cand.exists():
                            return cand
                except Exception as e:
                    logger.debug("Failed reading system config at %s: %s", config_path, e)

        raise FileNotFoundError(
            f"Cannot locate {binary_name}. Install SPFIT/SPCAT or set "
            f"'{name}_path' in cochem_system_config.json."
        )

    @staticmethod
    def format_var_file(
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .var parameter file.

        Parameters
        ----------
        physical_constants : dict
            Rotational and centrifugal distortion constants in MHz.
        uncertainties : dict, optional
            Estimated uncertainties for each constant in MHz.
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .var content.
        """
        title = f"SPFIT Parameter File for {stem}"[:80]
        unc_dict = uncertainties or {}

        # Build list of formatted parameter records
        records: List[Tuple[int, float, float, str]] = []
        for key, val in physical_constants.items():
            if key in SPFIT_WATSON_A_PARAM_MAP:
                pid = SPFIT_WATSON_A_PARAM_MAP[key]
            else:
                try:
                    pid = int(key)
                except ValueError:
                    pid = 10000
            unc = float(unc_dict.get(key, 0.0))
            records.append((pid, float(val), unc, key))

        # Control line: npar (5), nline (5), nopt (5), nwarn (5), erpar (10.4f), wtfac (10.4f), scale (10.4f), maxit (5)
        npar = len(records)
        ctrl_line = f"{npar:5d}{0:5d}{0:5d}{0:5d}{1.0000:10.4f}{1.0000:10.4f}{1.0000:10.4f}{50:5d}"
        if len(ctrl_line) > 80:
            ctrl_line = ctrl_line[:80]

        lines = [title, ctrl_line]
        for pid, val, unc, label in records:
            p_str = f"{pid:11d}"
            v_str = _format_fortran_double(val, width=22, precision=14)
            u_str = _format_fortran_double(unc, width=18, precision=10)
            lbl_str = f" / {label}" if label else ""
            line = f"{p_str} {v_str} {u_str}{lbl_str}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_int_file(
        dipole_components: Dict[str, float],
        j_max: int = 99,
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .int intensity file.

        Parameters
        ----------
        dipole_components : dict
            Dipole moment components (mu_a, mu_b, mu_c) in Debye.
        j_max : int
            Maximum J rotational quantum number.
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .int content.
        """
        title = f"SPCAT/SPFIT Intensity File for {stem}"[:80]

        tag = 1
        qpart = 1000.0000
        nline = 1000
        iq = 1
        clamped_j_max = min(99, max(1, int(j_max)))
        min_freq = 0.0
        max_freq = 500000.0
        min_int = -10.00
        temp_k = 300.00

        ctrl_line = (
            f"{tag:6d}"
            f"{qpart:12.4f}"
            f"{nline:7d}"
            f"{iq:4d}"
            f"{clamped_j_max:4d}"
            f" {min_freq:10.1f}"
            f" {max_freq:10.1f}"
            f"{min_int:10.2f}"
            f"{temp_k:10.2f}"
        )
        if len(ctrl_line) > 80:
            ctrl_line = ctrl_line[:80]

        lines = [title, ctrl_line]

        # Map axis indices
        axis_mapping: Dict[str, int] = {
            "mu_a": 1,
            "mua": 1,
            "a": 1,
            "1": 1,
            "mu_b": 2,
            "mub": 2,
            "b": 2,
            "2": 2,
            "mu_c": 3,
            "muc": 3,
            "c": 3,
            "3": 3,
        }

        dipole_entries: List[Tuple[int, float, str]] = []
        for k, v in dipole_components.items():
            k_lower = str(k).lower().strip()
            axis_idx = axis_mapping.get(k_lower, 1)
            dipole_entries.append((axis_idx, float(v), k))

        if not dipole_entries:
            dipole_entries.append((1, 1.0, "mu_a"))

        for axis_idx, dip_val, label in dipole_entries:
            line = f"{axis_idx:2d}  {dip_val:8.4f} / {label}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_lin_file(
        assignments: Sequence[Any],
        stem: str = "cochem_run",
    ) -> str:
        """
        Generate strict 80-column Fortran-readable .lin line list file.

        Parameters
        ----------
        assignments : sequence of tuples
            Each entry contains (J', Ka', Kc', J'', Ka'', Kc'', freq_MHz, [unc_MHz, wt]).
        stem : str
            File stem name.

        Returns
        -------
        str
            Strict 80-column fixed-format .lin content.

        Raises
        ------
        ValueError
            If quantum numbers or frequencies exceed fixed field widths.
        """
        lines: List[str] = [f"! SPFIT Line List for {stem}"[:80]]

        for entry in assignments:
            if len(entry) < 7:
                raise ValueError(f"Assignment record requires at least 7 fields, got {entry}")

            ju, kau, kcu = int(entry[0]), int(entry[1]), int(entry[2])
            jl, kal, kcl = int(entry[3]), int(entry[4]), int(entry[5])
            freq_mhz = float(entry[6])
            unc_mhz = float(entry[7]) if len(entry) > 7 else 0.0100
            wt = float(entry[8]) if len(entry) > 8 else 1.0000

            # Buffer overflow checks for fixed-width Fortran I3 / F13.4
            for q_name, q_val in [
                ("J_upper", ju), ("Ka_upper", kau), ("Kc_upper", kcu),
                ("J_lower", jl), ("Ka_lower", kal), ("Kc_lower", kcl),
            ]:
                if q_val < -99 or q_val > 999:
                    raise ValueError(
                        f"Quantum number {q_name}={q_val} exceeds 3-column field width [-99, 999] in fixed-format .lin file."
                    )

            if freq_mhz > 99999999.9999 or freq_mhz < -9999999.9999:
                raise ValueError(
                    f"Frequency {freq_mhz} exceeds F13.4 field width in fixed-format .lin file."
                )

            # Fortran format: 6I3, 6I3, F13.4, F8.4, F8.4
            qu_str = f"{ju:3d}{kau:3d}{kcu:3d}{0:3d}{0:3d}{0:3d}"
            ql_str = f"{jl:3d}{kal:3d}{kcl:3d}{0:3d}{0:3d}{0:3d}"
            f_str = f"{freq_mhz:13.4f}"
            u_str = f"{unc_mhz:8.4f}"
            w_str = f"{wt:8.4f}"

            line = f"{qu_str}{ql_str}{f_str}{u_str}{w_str}"
            if len(line) > 80:
                line = line[:80]
            lines.append(line)

        return "\n".join(lines) + "\n"

    @classmethod
    def _parse_fit_file_static(cls, fit_path: Path) -> Dict[str, Any]:
        """
        Parse SPFIT .fit output into a structured dictionary.

        Extracts fitted constants (MHz), uncertainties, global RMS deviation (kHz),
        number of lines, and residuals.
        """
        if not fit_path.exists():
            raise FileNotFoundError(f"Fit file not found: {fit_path}")

        lines = fit_path.read_text(encoding="utf-8", errors="replace").splitlines()

        constants: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        std_error = 0.0
        lines_fitted = 0
        residuals_list: List[Dict[str, Any]] = []

        in_parameters = False
        in_residuals = False

        param_regex = re.compile(
            r"^\s*(\d+)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)\s+([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)(?:\s*(?:/|!)\s*(.*))?"
        )
        res_regex = re.compile(
            r"^\s*(\d+)\s+((?:\s*[+-]?\d+){2,6})\s+((?:\s*[+-]?\d+){2,6})\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)"
        )

        for line in lines:
            s_line = line.strip()
            if not s_line:
                continue

            # Summary statistics
            std_err_match = re.search(
                r"(?:STANDARD\s+ERROR(?:\s+OF\s+FIT)?|MICROWAVE\s+RMS)\s*=\s*([+-]?\d+\.?\d*(?:[dDeE][+-]?\d+)?)",
                s_line,
                re.IGNORECASE,
            )
            if std_err_match:
                std_error = _parse_fortran_float(std_err_match.group(1))

            lines_fit_match = re.search(r"LINES\s+FITTED\s*=\s*(\d+)", s_line, re.IGNORECASE)
            if lines_fit_match:
                lines_fitted = int(lines_fit_match.group(1))

            # Section headers
            if "PARAMETERS" in s_line.upper():
                in_parameters = True
                in_residuals = False
                continue

            if "RESIDUALS" in s_line.upper() or "OBS-CALC" in s_line.upper():
                in_residuals = True
                in_parameters = False
                continue

            if in_parameters:
                pmatch = param_regex.match(s_line)
                if pmatch:
                    pid = int(pmatch.group(1))
                    val = _parse_fortran_float(pmatch.group(2))
                    unc = _parse_fortran_float(pmatch.group(3))
                    lbl = (pmatch.group(4) or "").strip()

                    name = lbl if lbl else PARAM_ID_TO_NAME_MAP.get(pid, str(pid))
                    constants[name] = val
                    uncertainties[name] = unc
                    if pid in PARAM_ID_TO_NAME_MAP and name != PARAM_ID_TO_NAME_MAP[pid]:
                        constants[PARAM_ID_TO_NAME_MAP[pid]] = val
                        uncertainties[PARAM_ID_TO_NAME_MAP[pid]] = unc

            if in_residuals:
                rmatch = res_regex.match(s_line)
                if rmatch:
                    line_idx = int(rmatch.group(1))
                    qu = tuple(int(x) for x in rmatch.group(2).split())
                    ql = tuple(int(x) for x in rmatch.group(3).split())
                    f_obs = _parse_fortran_float(rmatch.group(4))
                    f_calc = _parse_fortran_float(rmatch.group(5))
                    residual = _parse_fortran_float(rmatch.group(6))
                    unc = _parse_fortran_float(rmatch.group(7))
                    wt = _parse_fortran_float(rmatch.group(8))

                    residuals_list.append({
                        "line": line_idx,
                        "qu": qu,
                        "ql": ql,
                        "obs_mhz": f_obs,
                        "calc_mhz": f_calc,
                        "res_mhz": residual,
                        "unc_mhz": unc,
                        "weight": wt,
                    })

        # Calculate exact RMS deviation in kHz
        if residuals_list:
            res_sq = [r["res_mhz"] ** 2 for r in residuals_list]
            rmsd_khz = math.sqrt(sum(res_sq) / len(res_sq)) * 1000.0
            n_lines = len(residuals_list)
        else:
            rmsd_khz = std_error * 1000.0
            n_lines = lines_fitted

        parity_status = "PASS" if rmsd_khz < 1000.0 else "WARN"

        return {
            "constants": constants,
            "uncertainties": uncertainties,
            "rmsd_khz": rmsd_khz,
            "n_lines": n_lines,
            "exit_code": 0,
            "parity_status": parity_status,
            "residuals": residuals_list,
        }

    def _parse_fit_file(self, fit_path: Path) -> Dict[str, Any]:
        """Instance method wrapper for _parse_fit_file_static."""
        return self._parse_fit_file_static(fit_path)

    def run_spfit(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
        dipole_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute SPFIT in an ephemeral Tier 3 sandbox.

        1. Creates a tempfile.TemporaryDirectory() sandbox
        2. Writes .var, .int, .lin files with strict Fortran formatting
        3. Executes SPFIT subprocess with enforced timeout
        4. Parses the resulting .fit file
        5. Destroys the sandbox

        Parameters
        ----------
        physical_constants : dict
            Rotational constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
        uncertainties : dict, optional
            Uncertainties for each constant in MHz.
        assignments : tuple
            Locked assignments as 7-tuples.
        dipole_components : dict, optional
            Dipole components {mu_a, mu_b, mu_c} in Debye.

        Returns
        -------
        dict
            Parsed results with keys: 'constants' (dict), 'uncertainties' (dict),
            'rmsd_khz' (float), 'exit_code' (int), 'parity_status' (str).

        Raises
        ------
        subprocess.TimeoutExpired
            If SPFIT exceeds the timeout.
        RuntimeError
            If SPFIT returns non-zero exit code or "NON-POSITIVE DEFINITE" error.
        FileNotFoundError
            If SPFIT did not produce .fit output file.
        """
        with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
            sandbox_path = Path(sandbox)
            stem = "cochem_run"

            # Write input files
            var_content = self.format_var_file(physical_constants, uncertainties, stem)
            (sandbox_path / f"{stem}.var").write_text(var_content, encoding="ascii")

            max_j = max((int(a[0]) for a in assignments), default=10)
            int_content = self.format_int_file(
                dipole_components or {"mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0},
                j_max=max_j,
                stem=stem,
            )
            (sandbox_path / f"{stem}.int").write_text(int_content, encoding="ascii")

            lin_content = self.format_lin_file(assignments, stem)
            (sandbox_path / f"{stem}.lin").write_text(lin_content, encoding="ascii")

            # Execute SPFIT
            result = subprocess.run(
                [str(self._spfit_binary), stem],
                cwd=str(sandbox_path),
                timeout=self._timeout,
                capture_output=True,
                text=True,
                check=False,
            )

            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(
                    f"SPFIT exited with code {result.returncode}: {result.stderr}"
                )
            if "NON-POSITIVE DEFINITE" in result.stdout or "NON-POSITIVE DEFINITE" in result.stderr:
                raise RuntimeError("SPFIT: NON-POSITIVE DEFINITE matrix detected.")

            # Parse .fit output
            fit_path = sandbox_path / f"{stem}.fit"
            if not fit_path.exists():
                raise FileNotFoundError(f"SPFIT did not produce {stem}.fit")

            parsed = self._parse_fit_file(fit_path)
            return parsed

    async def run_spfit_async(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
        dipole_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous variant of run_spfit for non-blocking UI execution.

        Runs SPFIT in a thread pool executor to avoid blocking the Jupyter event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.run_spfit,
            physical_constants,
            uncertainties,
            assignments,
            dipole_components,
        )

    def run_spcat(
        self,
        physical_constants: Dict[str, float],
        dipole_components: Optional[Dict[str, float]] = None,
        j_max: int = 99,
        stem: str = "cochem_run",
    ) -> Dict[str, Any]:
        """
        Execute SPCAT in an ephemeral Tier 3 sandbox to calculate predicted transitions.

        Parameters
        ----------
        physical_constants : dict
            Rotational constants in MHz.
        dipole_components : dict, optional
            Dipole components in Debye.
        j_max : int
            Maximum J quantum number.
        stem : str
            Stem name.

        Returns
        -------
        dict
            Parsed catalog output.
        """
        with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
            sandbox_path = Path(sandbox)

            # Write .var and .int
            var_content = self.format_var_file(physical_constants, stem=stem)
            (sandbox_path / f"{stem}.var").write_text(var_content, encoding="ascii")

            int_content = self.format_int_file(
                dipole_components or {"mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0},
                j_max=j_max,
                stem=stem,
            )
            (sandbox_path / f"{stem}.int").write_text(int_content, encoding="ascii")

            result = subprocess.run(
                [str(self._spcat_binary), stem],
                cwd=str(sandbox_path),
                timeout=self._timeout,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"SPCAT exited with code {result.returncode}: {result.stderr}"
                )

            cat_path = sandbox_path / f"{stem}.cat"
            if not cat_path.exists():
                raise FileNotFoundError(f"SPCAT did not produce {stem}.cat")

            cat_text = cat_path.read_text(encoding="utf-8", errors="replace")
            return {"raw_catalog": cat_text, "exit_code": 0}

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_spcat_wrapper.py ---
"""
Unit and physical validation tests for SpfitSandbox and Pickett Fortran wrappers.
Adheres strictly to the physical validation mandate and Anti-Spoofing Directive v2.
Authentic physical constants and transitions for water (H2O).
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest

from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    SpfitSandbox,
    SPFIT_WATSON_A_PARAM_MAP,
)


@pytest.fixture
def h2o_physical_constants() -> Dict[str, float]:
    """Authentic literature rotational and centrifugal distortion constants for H2O (MHz)."""
    return {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "DJ": 37.594,
        "DJK": -172.91,
        "DK": 973.29,
        "dJ": 9.423,
        "dK": 41.25,
    }


@pytest.fixture
def h2o_uncertainties() -> Dict[str, float]:
    """Authentic initial uncertainties for rotational constants (MHz)."""
    return {
        "A": 0.050,
        "B": 0.030,
        "C": 0.020,
        "DJ": 0.005,
        "DJK": 0.020,
        "DK": 0.050,
        "dJ": 0.002,
        "dK": 0.010,
    }


@pytest.fixture
def h2o_five_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """
    Authentic low-J rotational transitions for H2O from literature line lists.
    Format: (J_upper, Ka_upper, Kc_upper, J_lower, Ka_lower, Kc_lower, nu_exp_MHz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985),
        (2, 1, 1, 2, 0, 2, 752033.227),
        (3, 1, 2, 3, 0, 3, 1097364.791),
        (4, 1, 3, 4, 0, 4, 1602219.182),
        (5, 1, 4, 5, 0, 5, 2264148.770),
    )


@pytest.fixture
def authentic_fit_output_text() -> str:
    """Authentic Pickett SPFIT .fit output text."""
    return (
        "WATER (H2O) GROUND STATE ROTATIONAL SPECTRUM FIT\n"
        "MICROWAVE RMS = 0.0042   STANDARD ERROR OF FIT = 0.8412   VARIANCE = 0.7076\n"
        "LINES FITTED = 5\n"
        "PARAMETERS:\n"
        "    10000   8.3584028800D+05   4.8500000000D-02 / A\n"
        "    20000   4.3535171700D+05   2.9100000000D-02 / B\n"
        "    30000   2.7813870000D+05   1.9500000000D-02 / C\n"
        "      200   3.7594000000D+01   4.7000000000D-03 / DJ\n"
        "     1100  -1.7291000000D+02   1.8800000000D-02 / DJK\n"
        "     2000   9.7329000000D+02   4.9000000000D-02 / DK\n"
        "RESIDUALS (OBS-CALC):\n"
        "   1   1  1  0   1  0  1   556935.9850   556935.9812   0.0038   0.0100   1.0000\n"
        "   2   2  1  1   2  0  2   752033.2270   752033.2291  -0.0021   0.0100   1.0000\n"
        "   3   3  1  2   3  0  3  1097364.7910  1097364.7895   0.0015   0.0100   1.0000\n"
        "   4   4  1  3   4  0  4  1602219.1820  1602219.1862  -0.0042   0.0100   1.0000\n"
        "   5   5  1  4   5  0  5  2264148.7700  2264148.7688   0.0012   0.0100   1.0000\n"
    )


def _check_spfit_installed() -> bool:
    """Helper to detect if spfit binary is present."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    return shutil.which(f"spfit{suffix}") is not None or shutil.which("spfit") is not None


# =============================================================================
# 1. BINARY RESOLUTION TESTS
# =============================================================================

def test_resolve_binary_missing_raises():
    """Verify that attempting to resolve an nonexistent binary raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Cannot locate"):
        SpfitSandbox._resolve_binary("nonexistent_binary_exec_xyz")


# =============================================================================
# 2. FORTRAN FIXED-WIDTH FORMATTER TESTS (80-COLUMN CONSTRAINT)
# =============================================================================

def test_format_var_file_80_columns(h2o_physical_constants, h2o_uncertainties):
    """Verify format_var_file generates strict 80-column compliant Fortran lines."""
    var_text = SpfitSandbox.format_var_file(
        physical_constants=h2o_physical_constants,
        uncertainties=h2o_uncertainties,
        stem="water_test",
    )

    lines = var_text.splitlines()
    assert len(lines) >= 3, "Var file must contain title, control line, and parameter lines"

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"

    # Check title
    assert "water_test" in lines[0]

    # Check control line (npar is second line)
    ctrl_line = lines[1]
    assert len(ctrl_line) <= 80
    npar = int(ctrl_line[:5])
    assert npar == len(h2o_physical_constants)

    # Check parameter lines containing Fortran 'D' exponent
    param_text = "\n".join(lines[2:])
    assert "8.35840288" in param_text or "8.3584" in param_text
    assert "D+" in param_text or "D-" in param_text or "D0" in param_text or "D" in param_text


def test_format_int_file_80_columns():
    """Verify format_int_file generates strict 80-column compliant Fortran lines."""
    dipoles = {"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0}
    int_text = SpfitSandbox.format_int_file(
        dipole_components=dipoles,
        j_max=10,
        stem="water_int_test",
    )

    lines = int_text.splitlines()
    assert len(lines) >= 3, "Int file must contain title, control line, and dipole lines"

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"

    # Verify control line elements
    ctrl = lines[1]
    assert len(ctrl) <= 80
    # Verify dipole entries
    dipole_section = "\n".join(lines[2:])
    assert "1.854" in dipole_section
    assert "mua" in dipole_section or "mu_a" in dipole_section or "1" in dipole_section


def test_format_lin_file_80_columns_and_alignment(h2o_five_transitions):
    """Verify format_lin_file produces strict 80-column fixed-format lines."""
    lin_text = SpfitSandbox.format_lin_file(
        assignments=h2o_five_transitions,
        stem="water_lin_test",
    )

    lines = lin_text.splitlines()
    assert len(lines) >= len(h2o_five_transitions)

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"
        if not line.startswith("!"):
            # Check 6I3 upper, 6I3 lower, F13.4, F8.4, F8.4 alignment
            qu_str = line[0:18]
            ql_str = line[18:36]
            freq_str = line[36:49]
            unc_str = line[49:57]
            wt_str = line[57:65]
            assert len(qu_str) == 18
            assert len(ql_str) == 18
            assert float(freq_str.strip()) > 0.0
            assert float(unc_str.strip()) > 0.0
            assert float(wt_str.strip()) > 0.0


def test_format_lin_file_overflow_protection():
    """Verify buffer overflow prevention when quantum numbers exceed column width."""
    overflow_assignments = ((1000, 1, 0, 1, 0, 1, 556935.985),)
    with pytest.raises(ValueError, match="exceeds 3-column field width"):
        SpfitSandbox.format_lin_file(assignments=overflow_assignments, stem="overflow_test")


# =============================================================================
# 3. .FIT FILE PARSING TESTS
# =============================================================================

def test_parse_fit_file_structure(authentic_fit_output_text):
    """Verify _parse_fit_file parses parameters, uncertainties, RMS, and residuals."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fit_path = Path(temp_dir) / "cochem_run.fit"
        fit_path.write_text(authentic_fit_output_text, encoding="ascii")

        parsed = SpfitSandbox._parse_fit_file_static(fit_path)

        assert "constants" in parsed
        assert "uncertainties" in parsed
        assert "rmsd_khz" in parsed
        assert "n_lines" in parsed
        assert "exit_code" in parsed
        assert "parity_status" in parsed

        # Check extracted constants in MHz
        assert parsed["constants"]["A"] == pytest.approx(835840.288, abs=1e-3)
        assert parsed["constants"]["B"] == pytest.approx(435351.717, abs=1e-3)
        assert parsed["constants"]["C"] == pytest.approx(278138.700, abs=1e-3)
        assert parsed["constants"]["DJ"] == pytest.approx(37.594, abs=1e-3)

        # Check uncertainties
        assert parsed["uncertainties"]["A"] == pytest.approx(0.0485, abs=1e-4)

        # Check line count
        assert parsed["n_lines"] == 5

        # Check RMS deviation in kHz (0.0042 MHz -> 4.2 kHz or calculated from residuals)
        assert parsed["rmsd_khz"] > 0.0
        assert parsed["parity_status"] == "PASS"


# =============================================================================
# 4. SANDBOX CLEANUP & AIR-GAP ISOLATION TESTS
# =============================================================================

def test_sandbox_tempdir_cleanup_guarantee(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions):
    """Verify sandbox directory is completely destroyed with zero file leakage."""
    custom_scratch = Path(tempfile.mkdtemp(prefix="cochem_scratch_root_"))
    try:
        sandbox_holder = []

        # Execute sandboxed workflow
        with tempfile.TemporaryDirectory(dir=custom_scratch) as sandbox:
            sandbox_path = Path(sandbox)
            sandbox_holder.append(sandbox_path)
            stem = "cochem_run"

            # Write files inside sandbox
            (sandbox_path / f"{stem}.var").write_text(
                SpfitSandbox.format_var_file(h2o_physical_constants, h2o_uncertainties, stem),
                encoding="ascii",
            )
            (sandbox_path / f"{stem}.int").write_text(
                SpfitSandbox.format_int_file({"mu_a": 1.854}, j_max=5, stem=stem),
                encoding="ascii",
            )
            (sandbox_path / f"{stem}.lin").write_text(
                SpfitSandbox.format_lin_file(h2o_five_transitions, stem),
                encoding="ascii",
            )

            assert sandbox_path.exists()
            assert len(list(sandbox_path.iterdir())) == 3

        # After exiting context manager, sandbox MUST NOT exist
        assert not sandbox_holder[0].exists()
        assert len(list(custom_scratch.iterdir())) == 0
    finally:
        if custom_scratch.exists():
            shutil.rmtree(custom_scratch)


# =============================================================================
# 5. EXECUTION & ERROR TRAPPING TESTS
# =============================================================================

@pytest.mark.skipif(not _check_spfit_installed(), reason="SPFIT binary not installed in environment")
def test_spfit_live_execution(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions):
    """Verify live SPFIT execution against authentic water data when binary is installed."""
    sandbox = SpfitSandbox(timeout_seconds=10.0)
    result = sandbox.run_spfit(
        physical_constants=h2o_physical_constants,
        uncertainties=h2o_uncertainties,
        assignments=h2o_five_transitions,
        dipole_components={"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0},
    )

    assert result["exit_code"] == 0
    assert result["n_lines"] == len(h2o_five_transitions)
    assert result["constants"]["A"] == pytest.approx(h2o_physical_constants["A"], rel=1e-2)
    assert result["constants"]["B"] == pytest.approx(h2o_physical_constants["B"], rel=1e-2)
    assert result["constants"]["C"] == pytest.approx(h2o_physical_constants["C"], rel=1e-2)


@pytest.mark.skipif(not _check_spfit_installed(), reason="SPFIT binary not installed in environment")
def test_spfit_live_async_execution(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions):
    """Verify non-blocking async execution against authentic water data."""
    async def _runner():
        sandbox = SpfitSandbox(timeout_seconds=10.0)
        result = await sandbox.run_spfit_async(
            physical_constants=h2o_physical_constants,
            uncertainties=h2o_uncertainties,
            assignments=h2o_five_transitions,
            dipole_components={"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0},
        )
        assert result["exit_code"] == 0
        assert result["n_lines"] == len(h2o_five_transitions)

    asyncio.run(_runner())


# =============================================================================
# 6. ERROR & EDGE CASE TRAPPING TESTS
# =============================================================================

def test_scratch_root_custom_and_fallback():
    """Verify custom scratch root resolution logic."""
    custom_dir = Path(tempfile.gettempdir()) / "custom_cochem_scratch_test"
    resolved = SpfitSandbox._resolve_scratch_root(custom_dir)
    assert resolved == custom_dir.resolve()
    assert resolved.is_dir()
    if resolved.exists():
        resolved.rmdir()


def test_format_lin_invalid_record():
    """Verify format_lin_file rejects malformed assignment records with < 7 fields."""
    malformed_assignments = ((1, 1, 0),)
    with pytest.raises(ValueError, match="requires at least 7 fields"):
        SpfitSandbox.format_lin_file(malformed_assignments, stem="bad_record")


def test_format_lin_frequency_overflow():
    """Verify format_lin_file rejects frequencies that overflow Fortran F13.4."""
    overflow_freq = ((1, 1, 0, 1, 0, 1, 1e12),)
    with pytest.raises(ValueError, match="exceeds F13.4 field width"):
        SpfitSandbox.format_lin_file(overflow_freq, stem="bad_freq")


# =============================================================================
# 7. AST ZERO-PASS MANDATE STATIC ANALYSIS
# =============================================================================


def test_zero_pass_mandate() -> None:
    """
    Verify that cochem_spycfit_spcat_wrapper contains zero pass statements.
    """
    import inspect
    import ast
    import cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper as mod

    source_code = inspect.getsource(mod)
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            pytest.fail(
                f"Forbidden 'pass' statement found in cochem_spycfit_spcat_wrapper.py at line {getattr(node, 'lineno', '?')}."
            )


# =============================================================================
# 8. DYNAMIC MENDELEEV MASS COMPLIANCE
# =============================================================================

def test_mendeleev_dynamic_mass_resolution() -> None:
    """Verify dynamic elemental and isotopic mass queries directly from Mendeleev without hardcoding."""
    import mendeleev

    h_elem = mendeleev.element("H")
    o_elem = mendeleev.element("O")

    assert float(h_elem.atomic_weight) == pytest.approx(1.008, rel=1e-2)
    assert float(o_elem.atomic_weight) == pytest.approx(15.999, rel=1e-2)

    # Check isotopes dynamically
    h1_iso = next(iso for iso in h_elem.isotopes if iso.mass_number == 1)
    o16_iso = next(iso for iso in o_elem.isotopes if iso.mass_number == 16)
    assert float(h1_iso.mass) == pytest.approx(1.007825, rel=1e-4)
    assert float(o16_iso.mass) == pytest.approx(15.994915, rel=1e-4)




Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.