"""
SPFIT/SPCAT subprocess wrapper.

Runs Pickett's SPFIT and SPCAT on scratch copies of their input files and keeps a record
of each run (executable and its SHA-256, command, exit code, stdout, stderr, input and
output files) as SF-CMP-011 requires. A missing program, a non-zero exit or a timeout
raises a typed error that names the program; nothing substitutes for the program.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cochem_spycfit_pickett_output import (
    PickettOutputError,
    PickettParameter,
    SpfitFitReport,
    parse_pickett_parameters,
    parse_spfit_fit,
)

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

# Names for Pickett parameter codes whose meaning does not depend on the reduction.
# SPFIT stores the negatives of the quartic distortion constants: code 200 holds -DJ,
# 1100 holds -DJK and 2000 holds -DK in both Watson reductions, and 41000 holds -dK
# (A reduction). Code 40100 is -dJ in the A reduction but d1 in the S reduction, so it
# keeps its numeric name.
PARAM_ID_TO_NAME_MAP: Dict[int, str] = {
    10000: "A",
    20000: "B",
    30000: "C",
    200: "-DJ",
    1100: "-DJK",
    2000: "-DK",
    41000: "-dK",
}

# Watson-sign names in SPFIT_WATSON_A_PARAM_MAP; their values are negated when written
# into the Pickett code that stores the negative constant.
_WATSON_SIGN_NAMES = frozenset({
    "DJ", "DJK", "DK", "dJ", "dK",
    "Delta_J", "Delta_JK", "Delta_K", "delta_J", "delta_K",
})


class PickettProgramError(RuntimeError):
    """SPFIT or SPCAT could not be found, failed, or timed out.

    Attributes:
        program: "spfit" or "spcat".
        record: The run record, when the program was started.
    """

    def __init__(self, program: str, message: str, record: Optional["PickettRunRecord"] = None) -> None:
        super().__init__(f"{program}: {message}")
        self.program = program
        self.record = record


class PickettInputError(ValueError):
    """Inputs cannot be turned into a complete, truthful Pickett input file."""


@dataclass(frozen=True)
class PickettRunRecord:
    """Evidence of one SPFIT or SPCAT execution (SF-CMP-011)."""

    program: str
    executable: str
    executable_sha256: str
    argv: Tuple[str, ...]
    exit_code: Optional[int]
    timed_out: bool
    stdout: str
    stderr: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SpfitRun:
    """A genuine SPFIT run with its parsed outputs."""

    record: PickettRunRecord
    report: SpfitFitReport
    parameters: Tuple[PickettParameter, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("Cannot parse an empty field as a Fortran float")

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
        spfit_binary: Optional[Path] = None,
        spcat_binary: Optional[Path] = None,
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
        spfit_binary, spcat_binary : Path, optional
            Explicit executables. When omitted, each is located by ``_resolve_binary``.
        """
        self._scratch_root: Optional[Path] = self._resolve_scratch_root(scratch_root)
        self._timeout: float = float(timeout_seconds)
        self._spfit_binary: Path = self._explicit_binary("spfit", spfit_binary) or self._resolve_binary("spfit")
        self._spcat_binary: Path = self._explicit_binary("spcat", spcat_binary) or self._resolve_binary("spcat")

    @staticmethod
    def _explicit_binary(name: str, path: Optional[Path]) -> Optional[Path]:
        """Return an explicitly configured executable, or None if none was given."""
        if path is None:
            return None
        candidate = Path(path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Configured {name} executable not found: {candidate}")
        return candidate

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

        Raises
        ------
        PickettInputError
            If a constant has no Pickett code or no a priori error. In SPFIT the a priori
            error decides whether a parameter is free or constrained, so it is never
            filled in.

        Notes
        -----
        Watson-sign names (DJ, DJK, DK, dJ, dK and the Delta_/delta_ spellings) are written
        negated, because SPFIT stores -DJ, -DJK, -DK, -dJ and -dK (A reduction). This
        output has no option line, so it is not a complete SPFIT/SPCAT input; a complete
        input is supplied to ``run_spfit_files``/``run_spcat_files``.
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
                except ValueError as exc:
                    raise PickettInputError(
                        f"Constant {key!r} has no Pickett parameter code; pass a known name or a numeric code"
                    ) from exc
            if key not in unc_dict or unc_dict[key] is None:
                raise PickettInputError(
                    f"No a priori error given for {key!r}; SPFIT reads it to decide whether the parameter is free"
                )
            unc = float(unc_dict[key])
            value = -float(val) if key in _WATSON_SIGN_NAMES else float(val)
            records.append((pid, value, unc, key))

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
            if k_lower not in axis_mapping:
                raise PickettInputError(f"Unknown dipole component {k!r}; expected mu_a, mu_b or mu_c")
            dipole_entries.append((axis_mapping[k_lower], float(v), k))

        if not dipole_entries:
            raise PickettInputError("No dipole components given; SPCAT intensities need them")

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
            unc_mhz = float(entry[7]) if len(entry) > 7 and entry[7] is not None else None
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
            if unc_mhz is None:
                raise PickettInputError(
                    f"Assignment {entry[:6]} has no measurement uncertainty; SPFIT weights lines by "
                    "1/ERR^2, so an uncertainty is never filled in"
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
        Parse an SPFIT ``.fit`` file, and the ``.var`` beside it when present.

        Returns a dictionary with:

        * ``constants`` / ``uncertainties``: final parameter values and errors keyed by
          ``PARAM_ID_TO_NAME_MAP`` names (Pickett storage sign, e.g. ``-DJ``) or by the
          numeric code. Full precision comes from the ``.var``; without it, from the
          ``.fit`` value(error) text, which keeps at most three error digits.
        * ``parameter_source``: ``"var"`` or ``"fit"``.
        * ``rmsd_khz``: SPFIT's MICROWAVE RMS of the final iteration in kHz, or None.
        * ``rms_error_old`` / ``rms_error_new``: SPFIT's dimensionless weighted RMS before
          and after the final iteration's update.
        * ``n_lines``: residual rows of the final iteration used in the fit.
        * ``residuals``, ``bad_lines``, ``error_scaling_notes`` and ``missing``.

        This is a fit report, not a comparison with another engine: it carries no parity
        or agreement status (SF-CMP-008, SF-CMP-012).

        Raises
        ------
        FileNotFoundError
            If the .fit file does not exist.
        PickettOutputError
            If the file is not a completed SPFIT output.
        """
        fit_path = Path(fit_path)
        if not fit_path.exists():
            raise FileNotFoundError(f"Fit file not found: {fit_path}")

        report = parse_spfit_fit(fit_path.read_text(encoding="utf-8", errors="replace"))
        final = report.final

        var_path = fit_path.with_suffix(".var")
        if var_path.exists():
            params = [(p.code, p.value, p.error) for p in parse_pickett_parameters(
                var_path.read_text(encoding="utf-8", errors="replace"))]
            source = "var"
        else:
            params = [(p.code, p.value, p.est_error) for p in final.parameters]
            source = "fit"

        constants: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        for code, value, error in params:
            name = PARAM_ID_TO_NAME_MAP.get(code, str(code))
            constants[name] = value
            uncertainties[name] = error

        used = [r for r in final.residuals if r.used_in_fit and not r.is_blend_width_record]
        residuals_list = [
            {
                "line": r.line,
                "qn": r.quantum_numbers,
                "obs_mhz": r.obs,
                "calc_mhz": r.calc,
                "res_mhz": r.obs_minus_calc,
                "unc_mhz": r.exp_err,
                "est_err_mhz": r.est_err,
                "used_in_fit": r.used_in_fit,
                "blend_weight": r.blend_weight,
            }
            for r in final.residuals
        ]

        return {
            "constants": constants,
            "uncertainties": uncertainties,
            "parameter_source": source,
            "rmsd_khz": None if final.microwave_rms_mhz is None else final.microwave_rms_mhz * 1000.0,
            "rms_error_old": final.rms_error_old,
            "rms_error_new": final.rms_error_new,
            "n_lines": len(used) if final.residuals else None,
            "iterations": len(report.iterations),
            "residuals": residuals_list,
            "bad_lines": list(report.bad_lines),
            "error_scaling_notes": list(report.error_scaling_notes),
            "missing": list(report.missing),
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
        Not available: a fit cannot be run from a constants dictionary alone.

        SPFIT needs a ``.par`` file with the model's option line(s), a control line that
        sets the line count, iterations and rejection threshold, and a ``.lin`` file with
        measured uncertainties. None of these can be derived from the arguments without
        inventing them. At commit 7e42507 this method wrote a ``.var`` file (SPFIT reads
        ``.par``) without an option line, and genuine SPFIT exited with code 1.

        Use ``run_spfit_files`` with complete ``.par`` and ``.lin`` text.

        Raises
        ------
        PickettInputError
            Always.
        """
        raise PickettInputError(
            "run_spfit cannot build a complete SPFIT input from a constants dictionary "
            "(no option line, control settings or measured uncertainties); use run_spfit_files"
        )

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
        Not available: a catalog cannot be computed from a constants dictionary alone.

        SPCAT needs a ``.var`` with the model's option line(s) and an ``.int`` file with
        the partition function, frequency and intensity limits and dipole components. At
        commit 7e42507 this method wrote neither correctly, and genuine SPCAT waited on
        standard input until the timeout.

        Use ``run_spcat_files`` with complete ``.var`` and ``.int`` text.

        Raises
        ------
        PickettInputError
            Always.
        """
        raise PickettInputError(
            "run_spcat cannot build a complete SPCAT input from a constants dictionary "
            "(no option line or .int settings); use run_spcat_files"
        )

    def run_pickett(
        self,
        program: str,
        files: Mapping[str, str],
        stem: str,
    ) -> PickettRunRecord:
        """
        Run genuine SPFIT or SPCAT on scratch copies of ``files`` and return its record.

        Parameters
        ----------
        program : str
            ``"spfit"`` or ``"spcat"``.
        files : mapping of file name to text
            Input files written into a fresh scratch directory (for example
            ``{"mol.par": ..., "mol.lin": ...}``). The caller's files are never touched.
        stem : str
            File stem passed to the program.

        Raises
        ------
        PickettProgramError
            On a non-zero exit code or a timeout. The run record is attached.
        """
        if program not in ("spfit", "spcat"):
            raise ValueError(f"Unknown Pickett program {program!r}")
        executable = self._spfit_binary if program == "spfit" else self._spcat_binary
        exe_sha256 = _sha256_file(executable)
        argv = (str(executable), stem)

        with tempfile.TemporaryDirectory(dir=self._scratch_root) as sandbox:
            sandbox_path = Path(sandbox)
            for name, text in files.items():
                if Path(name).name != name:
                    raise ValueError(f"Input file name must not contain a path: {name!r}")
                (sandbox_path / name).write_text(text, encoding="ascii")

            timed_out = False
            try:
                completed = subprocess.run(
                    list(argv),
                    cwd=str(sandbox_path),
                    stdin=subprocess.DEVNULL,
                    timeout=self._timeout,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                exit_code: Optional[int] = completed.returncode
                stdout, stderr = completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = None
                stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")

            outputs = {
                f.name: f.read_text(encoding="utf-8", errors="replace")
                for f in sorted(sandbox_path.iterdir())
                if f.is_file()
            }

        record = PickettRunRecord(
            program=program,
            executable=str(executable),
            executable_sha256=exe_sha256,
            argv=argv,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            inputs=dict(files),
            outputs=outputs,
        )
        if timed_out:
            raise PickettProgramError(program, f"timed out after {self._timeout} s", record)
        if exit_code != 0:
            raise PickettProgramError(program, f"exited with code {exit_code}: {stderr.strip()}", record)
        return record

    def run_spfit_files(self, par_text: str, lin_text: str, stem: str = "cochem_run") -> SpfitRun:
        """
        Run genuine SPFIT on complete ``.par`` and ``.lin`` text.

        SPFIT rewrites the ``.par`` and writes ``.bak``, ``.var`` and ``.fit``; all are kept
        in the record's ``outputs``. The returned parameters come from the ``.var``.

        Raises
        ------
        PickettProgramError
            If SPFIT fails or does not write a ``.fit`` and ``.var``.
        PickettOutputError
            If its output cannot be parsed.
        """
        record = self.run_pickett("spfit", {f"{stem}.par": par_text, f"{stem}.lin": lin_text}, stem)
        for ext in ("fit", "var"):
            if f"{stem}.{ext}" not in record.outputs:
                raise PickettProgramError("spfit", f"did not write {stem}.{ext}", record)
        report = parse_spfit_fit(record.outputs[f"{stem}.fit"])
        parameters = parse_pickett_parameters(record.outputs[f"{stem}.var"])
        return SpfitRun(record=record, report=report, parameters=parameters)

    def run_spcat_files(self, var_text: str, int_text: str, stem: str = "cochem_run") -> PickettRunRecord:
        """
        Run genuine SPCAT on complete ``.var`` and ``.int`` text.

        The catalog is ``record.outputs[f"{stem}.cat"]``.

        Raises
        ------
        PickettProgramError
            If SPCAT fails or does not write a ``.cat``.
        """
        record = self.run_pickett("spcat", {f"{stem}.var": var_text, f"{stem}.int": int_text}, stem)
        if f"{stem}.cat" not in record.outputs:
            raise PickettProgramError("spcat", f"did not write {stem}.cat", record)
        return record
