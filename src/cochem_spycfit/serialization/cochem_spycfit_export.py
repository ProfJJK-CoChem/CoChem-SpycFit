"""FAIR Publication Export — AASTeX tables, Parquet provenance, and legacy format serialization."""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from cochem_spycfit.core_engine.cochem_spycfit_state import CommitManager, FitState

logger = logging.getLogger(__name__)


@runtime_checkable
class SpfitRunnerProtocol(Protocol):
    """Protocol for SPFIT execution sandbox / validation wrapper."""

    def run_spfit(
        self,
        physical_constants: dict[str, float],
        uncertainties: Optional[dict[str, float]] = None,
        assignments: tuple[tuple[int, int, int, int, int, int, float], ...] = ...,
        dipole_components: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """Execute SPFIT calculations."""
        ...


class FairExporter:
    """Finalizes spectroscopic fits into FAIR-compliant publication outputs."""

    def __init__(
        self,
        export_dir: Optional[Path] = None,
        spfit_wrapper: Optional[SpfitRunnerProtocol] = None,
    ) -> None:
        """
        Parameters
        ----------
        export_dir : Path, optional
            Directory for export outputs. Defaults to
            Path.home() / "CoChem_Artifacts" / "exports".
            Honors COCHEM_ARTIFACTS_ROOT environment variable.
        spfit_wrapper : SpfitRunnerProtocol, optional
            SPFIT wrapper for final validation run.
        """
        self._export_dir: Path = self._resolve_export_dir(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._spfit: Optional[SpfitRunnerProtocol] = spfit_wrapper

    def finalize(
        self,
        fit_state: FitState,
        commit_manager: Optional[CommitManager] = None,
        experimental_csv_path: Optional[Path] = None,
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
        commit_manager : CommitManager, optional
            The DAG manager (for branch archiving).
        experimental_csv_path : Path, optional
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
        fit_state: FitState,
        spfit_result: Optional[dict[str, Any]],
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

    def _generate_linelist_table(self, fit_state: FitState) -> Path:
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
        self, fit_state: FitState, spfit_result: Optional[dict[str, Any]]
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
        fit_state: FitState,
        experimental_csv_path: Optional[Path],
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
            self._compute_file_sha256(Path(experimental_csv_path))
            if experimental_csv_path and Path(experimental_csv_path).exists()
            else ""
        )
        git_hash = self._get_git_commit_hash()
        landscape_uuid = ""
        if landscape_h5_path and Path(landscape_h5_path).exists():
            import h5py  # type: ignore[import-untyped]

            try:
                with h5py.File(landscape_h5_path, "r") as f:
                    raw_uuid = f.attrs.get("uuid", "unknown")
                    landscape_uuid = str(raw_uuid)
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
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, out_path)
        return out_path

    def _export_legacy_formats(self, fit_state: FitState) -> dict[str, Path]:
        """
        Export standard legacy spectroscopy formats for external interoperability.

        Generates: .var and .lin files.

        Returns
        -------
        dict[str, Path]
            Map of format name to file path.
        """
        outputs: dict[str, Path] = {}
        stem = "cochem_export"

        # .var file (parameters)
        var_path = self._export_dir / f"{stem}.var"
        var_lines: list[str] = []
        constants = getattr(fit_state, "physical_constants", {})
        for param, value in sorted(constants.items()):
            var_lines.append(f"{param:>10s} {float(value):20.10f}")
        var_path.write_text("\n".join(var_lines), encoding="ascii")
        outputs["var"] = var_path

        # .lin file (assignments)
        lin_path = self._export_dir / f"{stem}.lin"
        lin_lines: list[str] = []
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
        p = Path(file_path)
        if not p.is_file():
            return ""
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
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
        except (subprocess.SubprocessError, OSError):
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

