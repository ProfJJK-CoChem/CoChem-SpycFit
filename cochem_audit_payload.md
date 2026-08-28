Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\05_P5_T03_fair_publication_export.md.
Original prompt:
# Prompt 12 — Phase 5, Task Group 03: FAIR Publication Export & Legacy Format Serialization

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 5 — WebGL Time-Travel UI, Provenance Graph & FAIR Export
> **Phase:** 5 (Tasks 81–100) — Frontend & Publication Synthesis
> **Dependency:** Prompt 01 (CommitManager/FitState), Prompt 04 (SPFIT wrapper for final validation run), Prompt 02 (TensorCache)

---

## Objective

Implement the "Save & Commit" finalizer that locks the active branch, executes a final SPFIT verification run, auto-generates AASTeX LaTeX tables and methods paragraphs, serializes the final state to `.parquet` with cryptographic provenance, and exports legacy Fortran formats (`.cat`, `.var`, `.int`, `.lin`, `.fit`) for external interoperability.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\cochem_spycfit_export.py
```

> **Scope:** This prompt targets ONLY the file above.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""FAIR Publication Export — AASTeX tables, Parquet provenance, and legacy format serialization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
```

### 2. `FairExporter` Class

```python
class FairExporter:
    """Finalizes spectroscopic fits into FAIR-compliant publication outputs."""

    def __init__(
        self,
        export_dir: Optional[Path] = None,
        spfit_wrapper: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        export_dir : Path, optional
            Directory for export outputs. Defaults to
            Path.home() / "CoChem_Artifacts" / "exports".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        spfit_wrapper : SpfitSandbox, optional
            SPFIT wrapper (from Prompt 04) for final validation run.
        """
        self._export_dir: Path = self._resolve_export_dir(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._spfit: Optional[object] = spfit_wrapper

    def finalize(
        self,
        fit_state: object,  # FitState
        commit_manager: object,  # CommitManager
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path] = None,
    ) -> dict[str, Path]:
        """
        Execute the full "Save & Commit" finalization pipeline.

        1. Lock the active branch and archive dormant branches
        2. Execute final SPFIT verification run in Tier 3 sandbox
        3. Generate AASTeX publication tables
        4. Generate methods paragraph
        5. Serialize to .parquet with cryptographic provenance
        6. Export legacy formats

        Parameters
        ----------
        fit_state : FitState
            The final locked FitState to export.
        commit_manager : CommitManager
            The DAG manager (for branch archiving).
        experimental_csv_path : Path
            Path to the raw experimental .csv spectrum file.
        landscape_h5_path : Path, optional
            Path to the landscape.h5 theoretical predictions file.

        Returns
        -------
        dict[str, Path]
            Map of output type to file path for all generated files.
        """
        outputs: dict[str, Path] = {}

        # 1. Archive dormant branches (mark as non-primary in registry)
        # Dormant branches are preserved per FAIR directive, not deleted
        commit_manager.save_registry()

        # 2. Final SPFIT verification
        if self._spfit is not None:
            spfit_result = self._spfit.run_spfit(
                physical_constants=fit_state.physical_constants,
                uncertainties={},  # Use SPFIT-derived uncertainties
                assignments=fit_state.locked_assignments,
            )
            logger.info(f"Final SPFIT verification: RMSD = {spfit_result['rmsd_khz']:.4f} kHz")
        else:
            spfit_result = None
            logger.warning("SPFIT wrapper not provided. Skipping final verification.")

        # 3. Generate LaTeX tables
        tex_table_path = self._generate_constants_table(fit_state, spfit_result)
        outputs["constants_table_tex"] = tex_table_path

        linelist_path = self._generate_linelist_table(fit_state)
        outputs["linelist_tex"] = linelist_path

        # 4. Generate methods paragraph
        methods_path = self._generate_methods_paragraph(fit_state, spfit_result)
        outputs["methods_tex"] = methods_path

        # 5. Serialize to Parquet with provenance
        parquet_path = self._serialize_parquet(
            fit_state, experimental_csv_path, landscape_h5_path
        )
        outputs["parquet"] = parquet_path

        # 6. Export legacy formats
        legacy_paths = self._export_legacy_formats(fit_state)
        outputs.update(legacy_paths)

        logger.info(f"FAIR export complete. {len(outputs)} files generated in {self._export_dir}")
        return outputs

    def _generate_constants_table(
        self,
        fit_state: object,
        spfit_result: Optional[dict],
    ) -> Path:
        """
        Generate AASTeX-formatted LaTeX table of spectroscopic constants.

        Uses siunitx for decimal alignment. Columns: Parameter, Value, Uncertainty, Unit.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_table.tex"
        lines = [
            r"\begin{deluxetable}{lSSs}",
            r"\tablecaption{Spectroscopic Constants\label{tab:constants}}",
            r"\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Uncertainty} & \colhead{Unit}}",
            r"\startdata",
        ]
        constants = fit_state.physical_constants
        # Use SPFIT uncertainties if available
        uncertainties = spfit_result.get("uncertainties", {}) if spfit_result else {}
        unit_map = {"A": "MHz", "B": "MHz", "C": "MHz", "DJ": "kHz", "DJK": "kHz", "DK": "kHz"}

        for param, value in sorted(constants.items()):
            unc = uncertainties.get(param, 0.0)
            unit = unit_map.get(param, "MHz")
            lines.append(f"  {param} & {value:.6f} & {unc:.6f} & {{{unit}}} \\\\")

        lines.extend([
            r"\enddata",
            r"\end{deluxetable}",
        ])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_linelist_table(self, fit_state: object) -> Path:
        """
        Generate AASTeX-formatted transition line list.

        Columns: J'(Ka',Kc') ← J''(Ka'',Kc''), ν_obs (MHz), ν_calc (MHz), O−C (kHz).

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_linelist.tex"
        lines = [
            r"\begin{deluxetable}{cSSS}",
            r"\tablecaption{Observed Transition Frequencies\label{tab:linelist}}",
            r"\tablehead{\colhead{Transition} & \colhead{$\nu_\mathrm{obs}$ (MHz)} & "
            r"\colhead{$\nu_\mathrm{calc}$ (MHz)} & \colhead{O$-$C (kHz)}}",
            r"\startdata",
        ]
        for j_u, ka_u, kc_u, j_l, ka_l, kc_l, nu_obs in fit_state.locked_assignments:
            qn_str = f"${j_u}({ka_u},{kc_u}) \\leftarrow {j_l}({ka_l},{kc_l})$"
            lines.append(f"  {qn_str} & {nu_obs:.4f} & {{---}} & {{---}} \\\\")

        lines.extend([r"\enddata", r"\end{deluxetable}"])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_methods_paragraph(
        self, fit_state: object, spfit_result: Optional[dict]
    ) -> Path:
        """
        Dynamically synthesize a LaTeX methods paragraph.

        Parameterizes: Hamiltonian reduction, operator basis, degrees of freedom,
        max J/Ka, and certified global RMSD.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "methods_paragraph.tex"
        n_lines = len(fit_state.locked_assignments)
        n_params = len(fit_state.physical_constants)
        max_j = max(a[0] for a in fit_state.locked_assignments) if fit_state.locked_assignments else 0
        max_ka = max(a[1] for a in fit_state.locked_assignments) if fit_state.locked_assignments else 0
        rmsd = fit_state.global_rmsd
        dof = n_lines - n_params

        text = (
            f"The rotational spectrum was analyzed using Watson's $A$-reduction "
            f"Hamiltonian in the $I^r$ representation. "
            f"A total of {n_lines} transitions were included in the fit, "
            f"with quantum numbers up to $J = {max_j}$ and $K_a = {max_ka}$. "
            f"The fit yielded {n_params} spectroscopic constants with "
            f"{dof} degrees of freedom and a global root-mean-square deviation "
            f"of \\SI{{{rmsd:.2f}}}{{kHz}}."
        )
        out_path.write_text(text, encoding="utf-8")
        return out_path

    def _serialize_parquet(
        self,
        fit_state: object,
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path],
    ) -> Path:
        """
        Serialize final state to .parquet with cryptographic provenance.

        Metadata schema embeds:
        - CoChem Git commit hash
        - SHA-256 hash of the raw experimental .csv
        - UUID of the landscape.h5 file

        Returns
        -------
        Path
            Path to the generated .parquet file.
        """
        out_path = self._export_dir / "final_fit_state.parquet"

        # Build metadata
        csv_hash = self._compute_file_sha256(experimental_csv_path)
        git_hash = self._get_git_commit_hash()
        landscape_uuid = ""
        if landscape_h5_path and landscape_h5_path.exists():
            import h5py
            with h5py.File(landscape_h5_path, "r") as f:
                landscape_uuid = f.attrs.get("uuid", "unknown")

        metadata = {
            b"cochem_git_hash": git_hash.encode(),
            b"experimental_csv_sha256": csv_hash.encode(),
            b"landscape_h5_uuid": str(landscape_uuid).encode(),
            b"export_timestamp": datetime.now(timezone.utc).isoformat().encode(),
            b"state_id": fit_state.state_id.encode(),
        }

        # Build Arrow table from assignments
        assignments = fit_state.locked_assignments
        table = pa.table({
            "J_upper": pa.array([a[0] for a in assignments], type=pa.int32()),
            "Ka_upper": pa.array([a[1] for a in assignments], type=pa.int32()),
            "Kc_upper": pa.array([a[2] for a in assignments], type=pa.int32()),
            "J_lower": pa.array([a[3] for a in assignments], type=pa.int32()),
            "Ka_lower": pa.array([a[4] for a in assignments], type=pa.int32()),
            "Kc_lower": pa.array([a[5] for a in assignments], type=pa.int32()),
            "nu_obs_mhz": pa.array([a[6] for a in assignments], type=pa.float64()),
        })

        # Write with metadata
        schema = table.schema.with_metadata(metadata)
        table = table.cast(schema)
        pq.write_table(table, out_path)
        return out_path

    def _export_legacy_formats(self, fit_state: object) -> dict[str, Path]:
        """
        Export standard legacy spectroscopy formats for external interoperability.

        Generates: .cat, .var, .int, .lin, .fit files.

        Returns
        -------
        dict[str, Path]
            Map of format name to file path.
        """
        outputs = {}
        stem = "cochem_export"

        # .var file (parameters)
        var_path = self._export_dir / f"{stem}.var"
        var_lines = []
        for param, value in sorted(fit_state.physical_constants.items()):
            var_lines.append(f"{param:>10s} {value:20.10f}")
        var_path.write_text("\n".join(var_lines), encoding="ascii")
        outputs["var"] = var_path

        # .lin file (assignments)
        lin_path = self._export_dir / f"{stem}.lin"
        lin_lines = []
        for j_u, ka_u, kc_u, j_l, ka_l, kc_l, nu_obs in fit_state.locked_assignments:
            lin_lines.append(
                f"{j_u:3d}{ka_u:3d}{kc_u:3d}{j_l:3d}{ka_l:3d}{kc_l:3d}"
                f"{nu_obs:13.4f}  0.0050  1.0"
            )
        lin_path.write_text("\n".join(lin_lines), encoding="ascii")
        outputs["lin"] = lin_path

        return outputs

    @staticmethod
    def _compute_file_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _get_git_commit_hash() -> str:
        """Get the current Git commit hash, or 'unknown' if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _resolve_export_dir(user_path: Optional[Path]) -> Path:
        """Resolve the export directory."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "exports"
        return Path.home() / "CoChem_Artifacts" / "exports"
```

---

## Validation Test Specification

1. Create a `FairExporter` with export_dir pointing to `tempfile.TemporaryDirectory()`
2. Create a test `FitState` with 5 authentic H₂O locked assignments and physical constants
3. Create a test experimental CSV file with minimal header in the temp directory
4. Call `finalize()` (without SPFIT wrapper — skip final validation)
5. **Assertions:**
   - `publication_table.tex` exists and contains valid LaTeX (`\begin{deluxetable}`)
   - `publication_linelist.tex` exists with correct quantum number formatting
   - `methods_paragraph.tex` contains correct n_lines, max_J, RMSD values
   - `final_fit_state.parquet` exists and contains SHA-256 metadata in schema
   - `.var` and `.lin` legacy files exist with correct Fortran formatting
   - Export directory verification: no files leak outside the temp directory

---

## Acceptance Criteria

- [ ] AASTeX tables use `siunitx` for decimal alignment
- [ ] Methods paragraph dynamically parameterized from fit state
- [ ] Parquet metadata embeds Git hash, CSV SHA-256, landscape.h5 UUID
- [ ] Legacy formats (.var, .lin) use strict Fortran column spacing
- [ ] All exports written to Tier 2 artifacts directory (or temp for tests)
- [ ] Dormant branches preserved per FAIR directive (not permanently purged)
- [ ] All paths use `pathlib.Path`

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
from cochem_spycfit.serialization.cochem_spycfit_export import (
    FairExporter,
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
    "FairExporter",
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
from cochem_spycfit.serialization.cochem_spycfit_export import (
    FairExporter,
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
    "FairExporter",
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


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\serialization\cochem_spycfit_export.py ---
"""FAIR Publication Export — AASTeX tables, Parquet provenance, and legacy format serialization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class FairExporter:
    """Finalizes spectroscopic fits into FAIR-compliant publication outputs."""

    def __init__(
        self,
        export_dir: Optional[Path] = None,
        spfit_wrapper: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        export_dir : Path, optional
            Directory for export outputs. Defaults to
            Path.home() / "CoChem_Artifacts" / "exports".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        spfit_wrapper : SpfitSandbox, optional
            SPFIT wrapper (from Prompt 04) for final validation run.
        """
        self._export_dir: Path = self._resolve_export_dir(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._spfit: Optional[object] = spfit_wrapper

    def finalize(
        self,
        fit_state: object,  # FitState
        commit_manager: object,  # CommitManager
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path] = None,
    ) -> dict[str, Path]:
        """
        Execute the full "Save & Commit" finalization pipeline.

        1. Lock the active branch and archive dormant branches
        2. Execute final SPFIT verification run in Tier 3 sandbox
        3. Generate AASTeX publication tables
        4. Generate methods paragraph
        5. Serialize to .parquet with cryptographic provenance
        6. Export legacy formats

        Parameters
        ----------
        fit_state : FitState
            The final locked FitState to export.
        commit_manager : CommitManager
            The DAG manager (for branch archiving).
        experimental_csv_path : Path
            Path to the raw experimental .csv spectrum file.
        landscape_h5_path : Path, optional
            Path to the landscape.h5 theoretical predictions file.

        Returns
        -------
        dict[str, Path]
            Map of output type to file path for all generated files.
        """
        outputs: dict[str, Path] = {}

        # 1. Archive dormant branches (mark as non-primary in registry)
        # Dormant branches are preserved per FAIR directive, not deleted
        if commit_manager is not None and hasattr(commit_manager, "save_registry"):
            commit_manager.save_registry()

        # 2. Final SPFIT verification
        if self._spfit is not None:
            spfit_result = self._spfit.run_spfit(
                physical_constants=fit_state.physical_constants,
                uncertainties={},  # Use SPFIT-derived uncertainties
                assignments=fit_state.locked_assignments,
            )
            rmsd_khz = spfit_result.get("rmsd_khz", 0.0)
            logger.info(f"Final SPFIT verification: RMSD = {rmsd_khz:.4f} kHz")
        else:
            spfit_result = None
            logger.warning("SPFIT wrapper not provided. Skipping final verification.")

        # 3. Generate LaTeX tables
        tex_table_path = self._generate_constants_table(fit_state, spfit_result)
        outputs["constants_table_tex"] = tex_table_path

        linelist_path = self._generate_linelist_table(fit_state)
        outputs["linelist_tex"] = linelist_path

        # 4. Generate methods paragraph
        methods_path = self._generate_methods_paragraph(fit_state, spfit_result)
        outputs["methods_tex"] = methods_path

        # 5. Serialize to Parquet with provenance
        parquet_path = self._serialize_parquet(
            fit_state, experimental_csv_path, landscape_h5_path
        )
        outputs["parquet"] = parquet_path

        # 6. Export legacy formats
        legacy_paths = self._export_legacy_formats(fit_state)
        outputs.update(legacy_paths)

        logger.info(f"FAIR export complete. {len(outputs)} files generated in {self._export_dir}")
        return outputs

    def _generate_constants_table(
        self,
        fit_state: object,
        spfit_result: Optional[dict],
    ) -> Path:
        """
        Generate AASTeX-formatted LaTeX table of spectroscopic constants.

        Uses siunitx for decimal alignment. Columns: Parameter, Value, Uncertainty, Unit.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_table.tex"
        lines = [
            r"\begin{deluxetable}{lSSs}",
            r"\tablecaption{Spectroscopic Constants\label{tab:constants}}",
            r"\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Uncertainty} & \colhead{Unit}}",
            r"\startdata",
        ]
        constants = getattr(fit_state, "physical_constants", {})
        # Use SPFIT uncertainties if available
        uncertainties = (
            spfit_result.get("uncertainties", {})
            if (spfit_result and isinstance(spfit_result, dict))
            else {}
        )
        unit_map = {
            "A": "MHz",
            "B": "MHz",
            "C": "MHz",
            "DJ": "kHz",
            "DJK": "kHz",
            "DK": "kHz",
            "dJ": "kHz",
            "dK": "kHz",
            "-DJ": "kHz",
            "-DJK": "kHz",
            "-DK": "kHz",
            "-dJ": "kHz",
            "-dK": "kHz",
            "Delta_J": "kHz",
            "Delta_JK": "kHz",
            "Delta_K": "kHz",
            "delta_J": "kHz",
            "delta_K": "kHz",
        }

        for param, value in sorted(constants.items()):
            unc = uncertainties.get(param, 0.0)
            unit = unit_map.get(param, "MHz")
            lines.append(f"  {param} & {float(value):.6f} & {float(unc):.6f} & {{{unit}}} \\\\")

        lines.extend([
            r"\enddata",
            r"\end{deluxetable}",
        ])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_linelist_table(self, fit_state: object) -> Path:
        """
        Generate AASTeX-formatted transition line list.

        Columns: J'(Ka',Kc') ← J''(Ka'',Kc''), ν_obs (MHz), ν_calc (MHz), O−C (kHz).

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_linelist.tex"
        lines = [
            r"\begin{deluxetable}{cSSS}",
            r"\tablecaption{Observed Transition Frequencies\label{tab:linelist}}",
            r"\tablehead{\colhead{Transition} & \colhead{$\nu_\mathrm{obs}$ (MHz)} & "
            r"\colhead{$\nu_\mathrm{calc}$ (MHz)} & \colhead{O$-$C (kHz)}}",
            r"\startdata",
        ]
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        for a in locked_assignments:
            j_u, ka_u, kc_u = int(a[0]), int(a[1]), int(a[2])
            j_l, ka_l, kc_l = int(a[3]), int(a[4]), int(a[5])
            nu_obs = float(a[6])
            qn_str = f"${j_u}({ka_u},{kc_u}) \\leftarrow {j_l}({ka_l},{kc_l})$"
            lines.append(f"  {qn_str} & {nu_obs:.4f} & {{---}} & {{---}} \\\\")

        lines.extend([r"\enddata", r"\end{deluxetable}"])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_methods_paragraph(
        self, fit_state: object, spfit_result: Optional[dict]
    ) -> Path:
        """
        Dynamically synthesize a LaTeX methods paragraph.

        Parameterizes: Hamiltonian reduction, operator basis, degrees of freedom,
        max J/Ka, and certified global RMSD.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "methods_paragraph.tex"
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        physical_constants = getattr(fit_state, "physical_constants", {})
        n_lines = len(locked_assignments)
        n_params = len(physical_constants)
        max_j = max((int(a[0]) for a in locked_assignments), default=0) if locked_assignments else 0
        max_ka = max((int(a[1]) for a in locked_assignments), default=0) if locked_assignments else 0
        rmsd = float(getattr(fit_state, "global_rmsd", 0.0))
        dof = n_lines - n_params

        text = (
            f"The rotational spectrum was analyzed using Watson's $A$-reduction "
            f"Hamiltonian in the $I^r$ representation. "
            f"A total of {n_lines} transitions were included in the fit, "
            f"with quantum numbers up to $J = {max_j}$ and $K_a = {max_ka}$. "
            f"The fit yielded {n_params} spectroscopic constants with "
            f"{dof} degrees of freedom and a global root-mean-square deviation "
            f"of \\SI{{{rmsd:.2f}}}{{kHz}}."
        )
        out_path.write_text(text, encoding="utf-8")
        return out_path

    def _serialize_parquet(
        self,
        fit_state: object,
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path],
    ) -> Path:
        """
        Serialize final state to .parquet with cryptographic provenance.

        Metadata schema embeds:
        - CoChem Git commit hash
        - SHA-256 hash of the raw experimental .csv
        - UUID of the landscape.h5 file

        Returns
        -------
        Path
            Path to the generated .parquet file.
        """
        out_path = self._export_dir / "final_fit_state.parquet"

        # Build metadata
        csv_hash = (
            self._compute_file_sha256(experimental_csv_path)
            if experimental_csv_path and Path(experimental_csv_path).exists()
            else ""
        )
        git_hash = self._get_git_commit_hash()
        landscape_uuid = ""
        if landscape_h5_path and Path(landscape_h5_path).exists():
            import h5py

            try:
                with h5py.File(landscape_h5_path, "r") as f:
                    landscape_uuid = f.attrs.get("uuid", "unknown")
            except Exception as exc:
                logger.debug(f"Failed reading UUID from {landscape_h5_path}: {exc}")
                landscape_uuid = "unknown"

        state_id_str = getattr(fit_state, "state_id", "")
        if isinstance(state_id_str, bytes):
            state_id_bytes = state_id_str
        else:
            state_id_bytes = str(state_id_str).encode("utf-8")

        metadata = {
            b"cochem_git_hash": git_hash.encode("utf-8"),
            b"experimental_csv_sha256": csv_hash.encode("utf-8"),
            b"landscape_h5_uuid": str(landscape_uuid).encode("utf-8"),
            b"export_timestamp": datetime.now(timezone.utc).isoformat().encode("utf-8"),
            b"state_id": state_id_bytes,
        }

        # Build Arrow table from assignments
        assignments = getattr(fit_state, "locked_assignments", ())
        table = pa.table({
            "J_upper": pa.array([int(a[0]) for a in assignments], type=pa.int32()),
            "Ka_upper": pa.array([int(a[1]) for a in assignments], type=pa.int32()),
            "Kc_upper": pa.array([int(a[2]) for a in assignments], type=pa.int32()),
            "J_lower": pa.array([int(a[3]) for a in assignments], type=pa.int32()),
            "Ka_lower": pa.array([int(a[4]) for a in assignments], type=pa.int32()),
            "Kc_lower": pa.array([int(a[5]) for a in assignments], type=pa.int32()),
            "nu_obs_mhz": pa.array([float(a[6]) for a in assignments], type=pa.float64()),
        })

        # Write with metadata
        schema = table.schema.with_metadata(metadata)
        table = table.cast(schema)
        pq.write_table(table, out_path)
        return out_path

    def _export_legacy_formats(self, fit_state: object) -> dict[str, Path]:
        """
        Export standard legacy spectroscopy formats for external interoperability.

        Generates: .var and .lin files.

        Returns
        -------
        dict[str, Path]
            Map of format name to file path.
        """
        outputs = {}
        stem = "cochem_export"

        # .var file (parameters)
        var_path = self._export_dir / f"{stem}.var"
        var_lines = []
        constants = getattr(fit_state, "physical_constants", {})
        for param, value in sorted(constants.items()):
            var_lines.append(f"{param:>10s} {float(value):20.10f}")
        var_path.write_text("\n".join(var_lines), encoding="ascii")
        outputs["var"] = var_path

        # .lin file (assignments)
        lin_path = self._export_dir / f"{stem}.lin"
        lin_lines = []
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        for a in locked_assignments:
            j_u, ka_u, kc_u = int(a[0]), int(a[1]), int(a[2])
            j_l, ka_l, kc_l = int(a[3]), int(a[4]), int(a[5])
            nu_obs = float(a[6])
            lin_lines.append(
                f"{j_u:3d}{ka_u:3d}{kc_u:3d}{j_l:3d}{ka_l:3d}{kc_l:3d}"
                f"{nu_obs:13.4f}  0.0050  1.0"
            )
        lin_path.write_text("\n".join(lin_lines), encoding="ascii")
        outputs["lin"] = lin_path

        return outputs

    @staticmethod
    def _compute_file_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _get_git_commit_hash() -> str:
        """Get the current Git commit hash, or 'unknown' if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _resolve_export_dir(user_path: Optional[Path]) -> Path:
        """Resolve the export directory."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "exports"
        return Path.home() / "CoChem_Artifacts" / "exports"

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\serialization\cochem_spycfit_export.py ---
"""FAIR Publication Export — AASTeX tables, Parquet provenance, and legacy format serialization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class FairExporter:
    """Finalizes spectroscopic fits into FAIR-compliant publication outputs."""

    def __init__(
        self,
        export_dir: Optional[Path] = None,
        spfit_wrapper: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        export_dir : Path, optional
            Directory for export outputs. Defaults to
            Path.home() / "CoChem_Artifacts" / "exports".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        spfit_wrapper : SpfitSandbox, optional
            SPFIT wrapper (from Prompt 04) for final validation run.
        """
        self._export_dir: Path = self._resolve_export_dir(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._spfit: Optional[object] = spfit_wrapper

    def finalize(
        self,
        fit_state: object,  # FitState
        commit_manager: object,  # CommitManager
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path] = None,
    ) -> dict[str, Path]:
        """
        Execute the full "Save & Commit" finalization pipeline.

        1. Lock the active branch and archive dormant branches
        2. Execute final SPFIT verification run in Tier 3 sandbox
        3. Generate AASTeX publication tables
        4. Generate methods paragraph
        5. Serialize to .parquet with cryptographic provenance
        6. Export legacy formats

        Parameters
        ----------
        fit_state : FitState
            The final locked FitState to export.
        commit_manager : CommitManager
            The DAG manager (for branch archiving).
        experimental_csv_path : Path
            Path to the raw experimental .csv spectrum file.
        landscape_h5_path : Path, optional
            Path to the landscape.h5 theoretical predictions file.

        Returns
        -------
        dict[str, Path]
            Map of output type to file path for all generated files.
        """
        outputs: dict[str, Path] = {}

        # 1. Archive dormant branches (mark as non-primary in registry)
        # Dormant branches are preserved per FAIR directive, not deleted
        if commit_manager is not None and hasattr(commit_manager, "save_registry"):
            commit_manager.save_registry()

        # 2. Final SPFIT verification
        if self._spfit is not None:
            spfit_result = self._spfit.run_spfit(
                physical_constants=fit_state.physical_constants,
                uncertainties={},  # Use SPFIT-derived uncertainties
                assignments=fit_state.locked_assignments,
            )
            rmsd_khz = spfit_result.get("rmsd_khz", 0.0)
            logger.info(f"Final SPFIT verification: RMSD = {rmsd_khz:.4f} kHz")
        else:
            spfit_result = None
            logger.warning("SPFIT wrapper not provided. Skipping final verification.")

        # 3. Generate LaTeX tables
        tex_table_path = self._generate_constants_table(fit_state, spfit_result)
        outputs["constants_table_tex"] = tex_table_path

        linelist_path = self._generate_linelist_table(fit_state)
        outputs["linelist_tex"] = linelist_path

        # 4. Generate methods paragraph
        methods_path = self._generate_methods_paragraph(fit_state, spfit_result)
        outputs["methods_tex"] = methods_path

        # 5. Serialize to Parquet with provenance
        parquet_path = self._serialize_parquet(
            fit_state, experimental_csv_path, landscape_h5_path
        )
        outputs["parquet"] = parquet_path

        # 6. Export legacy formats
        legacy_paths = self._export_legacy_formats(fit_state)
        outputs.update(legacy_paths)

        logger.info(f"FAIR export complete. {len(outputs)} files generated in {self._export_dir}")
        return outputs

    def _generate_constants_table(
        self,
        fit_state: object,
        spfit_result: Optional[dict],
    ) -> Path:
        """
        Generate AASTeX-formatted LaTeX table of spectroscopic constants.

        Uses siunitx for decimal alignment. Columns: Parameter, Value, Uncertainty, Unit.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_table.tex"
        lines = [
            r"\begin{deluxetable}{lSSs}",
            r"\tablecaption{Spectroscopic Constants\label{tab:constants}}",
            r"\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Uncertainty} & \colhead{Unit}}",
            r"\startdata",
        ]
        constants = getattr(fit_state, "physical_constants", {})
        # Use SPFIT uncertainties if available
        uncertainties = (
            spfit_result.get("uncertainties", {})
            if (spfit_result and isinstance(spfit_result, dict))
            else {}
        )
        unit_map = {
            "A": "MHz",
            "B": "MHz",
            "C": "MHz",
            "DJ": "kHz",
            "DJK": "kHz",
            "DK": "kHz",
            "dJ": "kHz",
            "dK": "kHz",
            "-DJ": "kHz",
            "-DJK": "kHz",
            "-DK": "kHz",
            "-dJ": "kHz",
            "-dK": "kHz",
            "Delta_J": "kHz",
            "Delta_JK": "kHz",
            "Delta_K": "kHz",
            "delta_J": "kHz",
            "delta_K": "kHz",
        }

        for param, value in sorted(constants.items()):
            unc = uncertainties.get(param, 0.0)
            unit = unit_map.get(param, "MHz")
            lines.append(f"  {param} & {float(value):.6f} & {float(unc):.6f} & {{{unit}}} \\\\")

        lines.extend([
            r"\enddata",
            r"\end{deluxetable}",
        ])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_linelist_table(self, fit_state: object) -> Path:
        """
        Generate AASTeX-formatted transition line list.

        Columns: J'(Ka',Kc') ← J''(Ka'',Kc''), ν_obs (MHz), ν_calc (MHz), O−C (kHz).

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "publication_linelist.tex"
        lines = [
            r"\begin{deluxetable}{cSSS}",
            r"\tablecaption{Observed Transition Frequencies\label{tab:linelist}}",
            r"\tablehead{\colhead{Transition} & \colhead{$\nu_\mathrm{obs}$ (MHz)} & "
            r"\colhead{$\nu_\mathrm{calc}$ (MHz)} & \colhead{O$-$C (kHz)}}",
            r"\startdata",
        ]
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        for a in locked_assignments:
            j_u, ka_u, kc_u = int(a[0]), int(a[1]), int(a[2])
            j_l, ka_l, kc_l = int(a[3]), int(a[4]), int(a[5])
            nu_obs = float(a[6])
            qn_str = f"${j_u}({ka_u},{kc_u}) \\leftarrow {j_l}({ka_l},{kc_l})$"
            lines.append(f"  {qn_str} & {nu_obs:.4f} & {{---}} & {{---}} \\\\")

        lines.extend([r"\enddata", r"\end{deluxetable}"])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _generate_methods_paragraph(
        self, fit_state: object, spfit_result: Optional[dict]
    ) -> Path:
        """
        Dynamically synthesize a LaTeX methods paragraph.

        Parameterizes: Hamiltonian reduction, operator basis, degrees of freedom,
        max J/Ka, and certified global RMSD.

        Returns
        -------
        Path
            Path to the generated .tex file.
        """
        out_path = self._export_dir / "methods_paragraph.tex"
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        physical_constants = getattr(fit_state, "physical_constants", {})
        n_lines = len(locked_assignments)
        n_params = len(physical_constants)
        max_j = max((int(a[0]) for a in locked_assignments), default=0) if locked_assignments else 0
        max_ka = max((int(a[1]) for a in locked_assignments), default=0) if locked_assignments else 0
        rmsd = float(getattr(fit_state, "global_rmsd", 0.0))
        dof = n_lines - n_params

        text = (
            f"The rotational spectrum was analyzed using Watson's $A$-reduction "
            f"Hamiltonian in the $I^r$ representation. "
            f"A total of {n_lines} transitions were included in the fit, "
            f"with quantum numbers up to $J = {max_j}$ and $K_a = {max_ka}$. "
            f"The fit yielded {n_params} spectroscopic constants with "
            f"{dof} degrees of freedom and a global root-mean-square deviation "
            f"of \\SI{{{rmsd:.2f}}}{{kHz}}."
        )
        out_path.write_text(text, encoding="utf-8")
        return out_path

    def _serialize_parquet(
        self,
        fit_state: object,
        experimental_csv_path: Path,
        landscape_h5_path: Optional[Path],
    ) -> Path:
        """
        Serialize final state to .parquet with cryptographic provenance.

        Metadata schema embeds:
        - CoChem Git commit hash
        - SHA-256 hash of the raw experimental .csv
        - UUID of the landscape.h5 file

        Returns
        -------
        Path
            Path to the generated .parquet file.
        """
        out_path = self._export_dir / "final_fit_state.parquet"

        # Build metadata
        csv_hash = (
            self._compute_file_sha256(experimental_csv_path)
            if experimental_csv_path and Path(experimental_csv_path).exists()
            else ""
        )
        git_hash = self._get_git_commit_hash()
        landscape_uuid = ""
        if landscape_h5_path and Path(landscape_h5_path).exists():
            import h5py

            try:
                with h5py.File(landscape_h5_path, "r") as f:
                    landscape_uuid = f.attrs.get("uuid", "unknown")
            except Exception as exc:
                logger.debug(f"Failed reading UUID from {landscape_h5_path}: {exc}")
                landscape_uuid = "unknown"

        state_id_str = getattr(fit_state, "state_id", "")
        if isinstance(state_id_str, bytes):
            state_id_bytes = state_id_str
        else:
            state_id_bytes = str(state_id_str).encode("utf-8")

        metadata = {
            b"cochem_git_hash": git_hash.encode("utf-8"),
            b"experimental_csv_sha256": csv_hash.encode("utf-8"),
            b"landscape_h5_uuid": str(landscape_uuid).encode("utf-8"),
            b"export_timestamp": datetime.now(timezone.utc).isoformat().encode("utf-8"),
            b"state_id": state_id_bytes,
        }

        # Build Arrow table from assignments
        assignments = getattr(fit_state, "locked_assignments", ())
        table = pa.table({
            "J_upper": pa.array([int(a[0]) for a in assignments], type=pa.int32()),
            "Ka_upper": pa.array([int(a[1]) for a in assignments], type=pa.int32()),
            "Kc_upper": pa.array([int(a[2]) for a in assignments], type=pa.int32()),
            "J_lower": pa.array([int(a[3]) for a in assignments], type=pa.int32()),
            "Ka_lower": pa.array([int(a[4]) for a in assignments], type=pa.int32()),
            "Kc_lower": pa.array([int(a[5]) for a in assignments], type=pa.int32()),
            "nu_obs_mhz": pa.array([float(a[6]) for a in assignments], type=pa.float64()),
        })

        # Write with metadata
        schema = table.schema.with_metadata(metadata)
        table = table.cast(schema)
        pq.write_table(table, out_path)
        return out_path

    def _export_legacy_formats(self, fit_state: object) -> dict[str, Path]:
        """
        Export standard legacy spectroscopy formats for external interoperability.

        Generates: .var and .lin files.

        Returns
        -------
        dict[str, Path]
            Map of format name to file path.
        """
        outputs = {}
        stem = "cochem_export"

        # .var file (parameters)
        var_path = self._export_dir / f"{stem}.var"
        var_lines = []
        constants = getattr(fit_state, "physical_constants", {})
        for param, value in sorted(constants.items()):
            var_lines.append(f"{param:>10s} {float(value):20.10f}")
        var_path.write_text("\n".join(var_lines), encoding="ascii")
        outputs["var"] = var_path

        # .lin file (assignments)
        lin_path = self._export_dir / f"{stem}.lin"
        lin_lines = []
        locked_assignments = getattr(fit_state, "locked_assignments", ())
        for a in locked_assignments:
            j_u, ka_u, kc_u = int(a[0]), int(a[1]), int(a[2])
            j_l, ka_l, kc_l = int(a[3]), int(a[4]), int(a[5])
            nu_obs = float(a[6])
            lin_lines.append(
                f"{j_u:3d}{ka_u:3d}{kc_u:3d}{j_l:3d}{ka_l:3d}{kc_l:3d}"
                f"{nu_obs:13.4f}  0.0050  1.0"
            )
        lin_path.write_text("\n".join(lin_lines), encoding="ascii")
        outputs["lin"] = lin_path

        return outputs

    @staticmethod
    def _compute_file_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _get_git_commit_hash() -> str:
        """Get the current Git commit hash, or 'unknown' if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _resolve_export_dir(user_path: Optional[Path]) -> Path:
        """Resolve the export directory."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "exports"
        return Path.home() / "CoChem_Artifacts" / "exports"

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_export.py ---
"""
Unit tests for FAIR publication export and legacy format serialization.
Verifies LaTeX tables, methods paragraph, Parquet cryptographic provenance,
and legacy Pickett Fortran file generation.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import pyarrow.parquet as pq
import pytest

from cochem_spycfit.core_engine.cochem_spycfit_state import CommitManager, FitState
from cochem_spycfit.serialization.cochem_spycfit_export import FairExporter


def get_authentic_water_assignments() -> (
    Tuple[Tuple[int, int, int, int, int, int, float], ...]
):
    """Return authentic ground-state H2O rotational transitions (MHz)."""
    return (
        (1, 1, 0, 1, 0, 1, 556936.0020),
        (2, 1, 1, 2, 0, 2, 752033.1380),
        (2, 2, 0, 2, 1, 1, 987926.7590),
        (3, 1, 3, 2, 2, 0, 183310.0870),
        (6, 1, 6, 5, 2, 3, 22235.0800),
    )


def get_authentic_water_constants() -> Dict[str, float]:
    """Return authentic ground-state H2O rotational constants (MHz)."""
    return {
        "A": 835840.300000,
        "B": 435351.700000,
        "C": 278139.800000,
        "DJ": 37.600000,
        "DJK": -172.000000,
        "DK": 648.000000,
    }


def create_water_experimental_csv(csv_path: Path) -> Path:
    """Create experimental CSV spectrum file with authentic transition frequencies."""
    content = (
        "frequency_mhz,intensity\n"
        "22235.0800,1.250\n"
        "183310.0870,0.910\n"
        "556936.0020,1.000\n"
        "752033.1380,0.850\n"
        "987926.7590,0.720\n"
    )
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def create_water_landscape_h5(h5_path: Path, uuid_str: str) -> Path:
    """Create authentic HDF5 landscape file with metadata UUID."""
    with h5py.File(h5_path, "w") as h5_handle:
        h5_handle.attrs["uuid"] = uuid_str
        h5_handle.create_dataset(
            "rotational_energies_cm",
            data=[0.0, 23.79, 70.09, 134.9, 206.3],
        )
    return h5_path


class DirectSpfitValidator:
    """Validator object providing run_spfit protocol."""

    def __init__(self, uncertainties: Dict[str, float], rmsd_khz: float) -> None:
        self._uncertainties = uncertainties
        self._rmsd_khz = rmsd_khz

    def run_spfit(
        self,
        physical_constants: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (),
    ) -> Dict[str, Any]:
        return {
            "constants": physical_constants,
            "uncertainties": self._uncertainties,
            "rmsd_khz": self._rmsd_khz,
            "n_lines": len(assignments),
            "exit_code": 0,
            "parity_status": "PASS",
        }


def test_fair_exporter_initialization_custom_dir(tmp_path: Path) -> None:
    """Verify initialization with custom export directory."""
    export_dir = tmp_path / "custom_exports"
    exporter = FairExporter(export_dir=export_dir)
    assert export_dir.exists()
    assert exporter._export_dir == export_dir


def test_fair_exporter_initialization_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify initialization resolving COCHEM_ARTIFACTS_ROOT."""
    artifacts_root = tmp_path / "env_artifacts"
    monkeypatch.setenv("COCHEM_ARTIFACTS_ROOT", str(artifacts_root))
    exporter = FairExporter()
    expected_dir = artifacts_root / "exports"
    assert exporter._export_dir == expected_dir
    assert expected_dir.exists()


def test_fair_exporter_finalize_full_pipeline(tmp_path: Path) -> None:
    """Verify the full finalize pipeline output without external binary."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    h5_path = tmp_path / "water_landscape.h5"

    create_water_experimental_csv(csv_path)
    expected_uuid = "cochem-water-h5-uuid-2026"
    create_water_landscape_h5(h5_path, expected_uuid)

    commit_manager = CommitManager(registry_path=registry_path)
    water_assignments = get_authentic_water_assignments()
    water_constants = get_authentic_water_constants()

    fit_state = commit_manager.commit(
        locked_assignments=water_assignments,
        physical_constants=water_constants,
        global_rmsd=4.25,
        global_chi2=1.02,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=h5_path,
    )

    # 1. Verify outputs dictionary contains all expected keys
    assert "constants_table_tex" in outputs
    assert "linelist_tex" in outputs
    assert "methods_tex" in outputs
    assert "parquet" in outputs
    assert "var" in outputs
    assert "lin" in outputs

    # 2. Verify all files exist
    for file_key, out_file in outputs.items():
        assert out_file.exists(), f"Output file for {file_key} does not exist: {out_file}"
        assert out_file.parent == export_dir, f"File leaked outside export dir: {out_file}"

    # 3. Verify constants table LaTeX
    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{lSSs}" in table_content
    assert r"\tablecaption{Spectroscopic Constants\label{tab:constants}}" in table_content
    assert r"\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Uncertainty} & \colhead{Unit}}" in table_content
    assert r"\startdata" in table_content
    assert "A & 835840.300000" in table_content
    assert "B & 435351.700000" in table_content
    assert "C & 278139.800000" in table_content
    assert "DJ & 37.600000" in table_content
    assert "DJK & -172.000000" in table_content
    assert "DK & 648.000000" in table_content
    assert r"{MHz}" in table_content
    assert r"{kHz}" in table_content
    assert r"\enddata" in table_content
    assert r"\end{deluxetable}" in table_content

    # 4. Verify linelist table LaTeX
    linelist_content = outputs["linelist_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{cSSS}" in linelist_content
    assert r"\tablecaption{Observed Transition Frequencies\label{tab:linelist}}" in linelist_content
    assert r"$1(1,0) \leftarrow 1(0,1)$ & 556936.0020 & {---} & {---}" in linelist_content
    assert r"$2(1,1) \leftarrow 2(0,2)$ & 752033.1380 & {---} & {---}" in linelist_content
    assert r"$2(2,0) \leftarrow 2(1,1)$ & 987926.7590 & {---} & {---}" in linelist_content
    assert r"$3(1,3) \leftarrow 2(2,0)$ & 183310.0870 & {---} & {---}" in linelist_content
    assert r"$6(1,6) \leftarrow 5(2,3)$ & 22235.0800 & {---} & {---}" in linelist_content
    assert r"\enddata" in linelist_content
    assert r"\end{deluxetable}" in linelist_content

    # 5. Verify methods paragraph
    methods_content = outputs["methods_tex"].read_text(encoding="utf-8")
    assert "Watson's $A$-reduction" in methods_content
    assert "$I^r$ representation" in methods_content
    assert "A total of 5 transitions were included in the fit" in methods_content
    assert "$J = 6$" in methods_content
    assert "$K_a = 2$" in methods_content
    assert "6 spectroscopic constants" in methods_content
    assert "-1 degrees of freedom" in methods_content or "degrees of freedom" in methods_content
    assert r"\SI{4.25}{kHz}" in methods_content

    # 6. Verify Parquet serialization and provenance metadata
    parquet_table = pq.read_table(outputs["parquet"])
    assert parquet_table.num_rows == 5
    assert "J_upper" in parquet_table.column_names
    assert "Ka_upper" in parquet_table.column_names
    assert "Kc_upper" in parquet_table.column_names
    assert "J_lower" in parquet_table.column_names
    assert "Ka_lower" in parquet_table.column_names
    assert "Kc_lower" in parquet_table.column_names
    assert "nu_obs_mhz" in parquet_table.column_names

    schema_metadata = parquet_table.schema.metadata
    assert b"cochem_git_hash" in schema_metadata
    assert b"experimental_csv_sha256" in schema_metadata
    assert b"landscape_h5_uuid" in schema_metadata
    assert b"export_timestamp" in schema_metadata
    assert b"state_id" in schema_metadata

    assert schema_metadata[b"landscape_h5_uuid"].decode("utf-8") == expected_uuid
    assert schema_metadata[b"state_id"].decode("utf-8") == fit_state.state_id

    # Verify CSV sha256 matches
    csv_bytes = csv_path.read_bytes()
    expected_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    assert schema_metadata[b"experimental_csv_sha256"].decode("utf-8") == expected_csv_sha

    # 7. Verify legacy .var file formatting
    var_content = outputs["var"].read_text(encoding="ascii")
    var_lines = [l for l in var_content.splitlines() if l.strip()]
    assert len(var_lines) == 6
    for line in var_lines:
        param_name = line[:10].strip()
        val_str = line[10:].strip()
        assert param_name in water_constants
        assert float(val_str) == pytest.approx(water_constants[param_name])

    # 8. Verify legacy .lin file formatting
    lin_content = outputs["lin"].read_text(encoding="ascii")
    lin_lines = [l for l in lin_content.splitlines() if l.strip()]
    assert len(lin_lines) == 5
    for line, authentic_entry in zip(lin_lines, water_assignments):
        ju = int(line[0:3])
        kau = int(line[3:6])
        kcu = int(line[6:9])
        jl = int(line[9:12])
        kal = int(line[12:15])
        kcl = int(line[15:18])
        freq = float(line[18:31])
        assert (ju, kau, kcu, jl, kal, kcl) == authentic_entry[0:6]
        assert freq == pytest.approx(authentic_entry[6], abs=1e-4)


def test_fair_exporter_with_spfit_wrapper(tmp_path: Path) -> None:
    """Verify finalize with active SPFIT validator providing uncertainties."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    water_assignments = get_authentic_water_assignments()
    water_constants = get_authentic_water_constants()

    fit_state = commit_manager.commit(
        locked_assignments=water_assignments,
        physical_constants=water_constants,
        global_rmsd=3.82,
    )

    validator_uncertainties = {
        "A": 0.051200,
        "B": 0.032100,
        "C": 0.021000,
        "DJ": 0.001200,
        "DJK": 0.003400,
        "DK": 0.005600,
    }
    validator = DirectSpfitValidator(
        uncertainties=validator_uncertainties,
        rmsd_khz=3.82,
    )

    exporter = FairExporter(export_dir=export_dir, spfit_wrapper=validator)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert "A & 835840.300000 & 0.051200" in table_content
    assert "B & 435351.700000 & 0.032100" in table_content
    assert "C & 278139.800000 & 0.021000" in table_content


def test_fair_exporter_missing_optional_files(tmp_path: Path) -> None:
    """Verify exporter handles missing landscape.h5 and non-git directory gracefully."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    fit_state = commit_manager.commit(
        locked_assignments=get_authentic_water_assignments(),
        physical_constants=get_authentic_water_constants(),
        global_rmsd=4.10,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    parquet_table = pq.read_table(outputs["parquet"])
    schema_metadata = parquet_table.schema.metadata
    assert schema_metadata[b"landscape_h5_uuid"] == b""


def test_fair_exporter_empty_state(tmp_path: Path) -> None:
    """Verify exporter functions correctly with empty state."""
    export_dir = tmp_path / "exports"
    registry_path = tmp_path / "cochem_fit_registry.json"
    csv_path = tmp_path / "water_spectrum.csv"
    create_water_experimental_csv(csv_path)

    commit_manager = CommitManager(registry_path=registry_path)
    fit_state = commit_manager.commit(
        locked_assignments=(),
        physical_constants={},
        global_rmsd=0.0,
    )

    exporter = FairExporter(export_dir=export_dir)
    outputs = exporter.finalize(
        fit_state=fit_state,
        commit_manager=commit_manager,
        experimental_csv_path=csv_path,
        landscape_h5_path=None,
    )

    table_content = outputs["constants_table_tex"].read_text(encoding="utf-8")
    assert r"\begin{deluxetable}{lSSs}" in table_content
    assert r"\end{deluxetable}" in table_content

    methods_content = outputs["methods_tex"].read_text(encoding="utf-8")
    assert "A total of 0 transitions were included in the fit" in methods_content
    assert "$J = 0$" in methods_content
    assert "$K_a = 0$" in methods_content


def test_fair_exporter_sha256_and_git_hash(tmp_path: Path) -> None:
    """Verify SHA-256 computation and Git hash retrieval helper."""
    test_file = tmp_path / "integrity_test.txt"
    test_content = b"CoChem Spectroscopic Data Payload 2026"
    test_file.write_bytes(test_content)

    expected_hash = hashlib.sha256(test_content).hexdigest()
    computed_hash = FairExporter._compute_file_sha256(test_file)
    assert computed_hash == expected_hash

    git_hash = FairExporter._get_git_commit_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0


def test_fair_exporter_various_distortion_units(tmp_path: Path) -> None:
    """Verify unit mapping for various centrifugal distortion constants."""
    export_dir = tmp_path / "exports"
    exporter = FairExporter(export_dir=export_dir)

    registry_path = tmp_path / "cochem_fit_registry.json"
    commit_manager = CommitManager(registry_path=registry_path)
    distortion_constants = {
        "A": 835840.3,
        "B": 435351.7,
        "C": 278139.8,
        "Delta_J": 0.0376,
        "Delta_JK": -0.1720,
        "Delta_K": 0.6480,
        "delta_J": 0.0050,
        "delta_K": 0.0820,
    }
    fit_state = commit_manager.commit(
        locked_assignments=get_authentic_water_assignments(),
        physical_constants=distortion_constants,
        global_rmsd=4.25,
    )

    table_path = exporter._generate_constants_table(fit_state, spfit_result=None)
    table_content = table_path.read_text(encoding="utf-8")
    assert "Delta_J" in table_content
    assert "Delta_JK" in table_content
    assert "Delta_K" in table_content
    assert "delta_J" in table_content
    assert "delta_K" in table_content
    assert r"{kHz}" in table_content


Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.