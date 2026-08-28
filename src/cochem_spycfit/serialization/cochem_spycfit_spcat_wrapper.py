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
