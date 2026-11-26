"""
CoChem QTAIM Electron Density and Bond Critical Point Topological Engine (REQ-QC-088).

Provenance:
- Method Matrix [M]: Quantum Theory of Atoms in Molecules (QTAIM) gradient vector field topology,
  Poincaré-Hopf relationship, Cremer-Kraka / Bader bond interaction classifications, and
  Non-Covalent Interaction (NCI / RDG) real-space analysis via authentic Multiwfn_noGUI.
- Derived Relations [D]: Local energy densities (G(r) kinetic, V(r) potential, H(r) total) and bond ellipticity.
- Empirical [E]: Bond Critical Point (3,-1) eigenvalues (lambda_1 <= lambda_2 < 0 < lambda_3).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.exceptions import QTAIMTopologyError
from cochem.qc.models import BondCriticalPointDescriptor

_MULTIWFN_SUPPORTED_EXTENSIONS = (
    ".molden.input",
    ".molden",
    ".fchk",
    ".wfx",
    ".wfn",
    ".gbw",
)
_MULTIWFN_MAX_PATH_LEN = 79
_QTAIM_SCHEMA = "cochem.spycfit.multiwfn-qtaim/1"
_NCI_SCHEMA = "cochem.spycfit.multiwfn-nci/1"


class CriticalPoint(BaseModel):
    """Data contract representing a general QTAIM critical point (NCP, BCP, RCP, CCP)."""

    model_config = ConfigDict(frozen=True)

    cp_type: Literal["(3,-3)", "(3,-1)", "(3,+1)", "(3,+3)"]
    cp_name: Literal["NCP", "BCP", "RCP", "CCP"]
    coordinates_angstrom: Tuple[float, float, float]
    density_au: float
    laplacian_au: float
    eigenvalues: Tuple[float, float, float]


class BCPDescriptor(BondCriticalPointDescriptor):
    """Alias for BondCriticalPointDescriptor."""


class QTAIMTopologyGraph(BaseModel):
    """Data contract representing the topological molecular graph formed by nuclei and BCP bond paths."""

    model_config = ConfigDict(frozen=True)

    n_ncp: int = Field(ge=1, description="Number of Nuclear Critical Points")
    n_bcp: int = Field(ge=0, description="Number of Bond Critical Points")
    n_rcp: int = Field(default=0, ge=0, description="Number of Ring Critical Points")
    n_ccp: int = Field(default=0, ge=0, description="Number of Cage Critical Points")
    is_poincare_hopf_satisfied: bool = True
    bcps: List[BondCriticalPointDescriptor] = Field(default_factory=list)


class PoincareHopfViolationError(QTAIMTopologyError, RuntimeError):
    """Raised when Poincaré-Hopf invariant n_NCP - n_BCP + n_RCP - n_CCP != 1 is violated."""


class MultiwfnQTAIMResult(BaseModel):
    """Authenticated Multiwfn QTAIM topology analysis result with SHA-256 provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["cochem.spycfit.multiwfn-qtaim/1"] = _QTAIM_SCHEMA
    engine: Literal["Multiwfn"] = "Multiwfn"
    multiwfn_version: str
    multiwfn_binary: str
    multiwfn_binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wavefunction_path: str
    wavefunction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    converted_molden_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    multiwfn_stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpprop_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_ncp: int = Field(ge=0)
    n_bcp: int = Field(ge=0)
    n_rcp: int = Field(ge=0)
    n_ccp: int = Field(ge=0)
    poincare_hopf_chi: int
    is_poincare_hopf_satisfied: bool
    topology_graph: QTAIMTopologyGraph
    critical_points: List[CriticalPoint]
    bcp_atom_pairs: List[Tuple[int, int]]
    bcp_symbol_pairs: List[Tuple[str, str]]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MultiwfnNCIResult(BaseModel):
    """Authenticated Multiwfn NCI / RDG weak-interaction analysis result with SHA-256 provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["cochem.spycfit.multiwfn-nci/1"] = _NCI_SCHEMA
    engine: Literal["Multiwfn"] = "Multiwfn"
    multiwfn_version: str
    multiwfn_binary: str
    multiwfn_binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wavefunction_path: str
    wavefunction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    converted_molden_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    multiwfn_stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nci_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grid_quality: int = Field(ge=1, le=3)
    rdg_cutoff: float = Field(gt=0.0)
    rho_cutoff: float = Field(gt=0.0)
    vdw_threshold: float = Field(gt=0.0)
    n_grid_points: int = Field(ge=1)
    n_low_rdg_points: int = Field(ge=0)
    n_nci_points: int = Field(ge=0)
    n_attractive_points: int = Field(ge=0)
    n_vdw_points: int = Field(ge=0)
    n_steric_points: int = Field(ge=0)
    attractive_fraction: float = Field(ge=0.0, le=1.0)
    vdw_fraction: float = Field(ge=0.0, le=1.0)
    steric_fraction: float = Field(ge=0.0, le=1.0)
    min_sign_lambda2_rho: float
    max_sign_lambda2_rho: float
    min_rdg: float = Field(ge=0.0)
    mean_nci_sign_lambda2_rho: Optional[float] = None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def classify_bond_critical_point(
    bcp_idx: int,
    atom_1: int,
    atom_2: int,
    coords: Tuple[float, float, float],
    rho: float,
    hessian_eigenvals: Tuple[float, float, float],
    g_density: float,
    v_density: float,
) -> BondCriticalPointDescriptor:
    """
    [M][D] Classify a (3,-1) Bond Critical Point using Bader and Cremer-Kraka criteria (REQ-QC-088).
    """
    if rho <= 0.0:
        raise ValueError(f"Electron density at BCP must be positive, got rho={rho} a.u.")
    if g_density < 0.0:
        raise ValueError(f"Kinetic energy density G must be >= 0, got G={g_density} a.u.")
    if v_density > 0.0:
        raise ValueError(f"Potential energy density V must be <= 0, got V={v_density} a.u.")

    # 1. Sort Hessian eigenvalues: lambda_1 <= lambda_2 <= lambda_3
    sorted_lambdas = sorted(hessian_eigenvals)
    l1, l2, l3 = sorted_lambdas[0], sorted_lambdas[1], sorted_lambdas[2]

    # 2. Laplacian nabla^2 rho = l1 + l2 + l3
    laplacian = l1 + l2 + l3

    # 3. Total energy density H = G + V
    h_density = g_density + v_density

    # 4. Bond Ellipticity epsilon = (lambda_1 / lambda_2) - 1
    if abs(l2) > 1e-9:
        ellipticity = max(0.0, (l1 / l2) - 1.0)
    else:
        ellipticity = 0.0

    # 5. Cremer-Kraka & Bader interaction classification
    classification: Literal[
        "COVALENT_SHARED", "TRANSIT_POLAR_COVALENT", "NON_COVALENT_CLOSED_SHELL", "HYDROGEN_BOND", "METALLIC"
    ]
    if laplacian < 0.0 and h_density < 0.0:
        classification = "COVALENT_SHARED"
    elif laplacian >= 0.0 and h_density < 0.0:
        classification = "TRANSIT_POLAR_COVALENT"
    else:
        classification = "NON_COVALENT_CLOSED_SHELL"

    return BondCriticalPointDescriptor(
        bcp_index=bcp_idx,
        atom_1_index=atom_1,
        atom_2_index=atom_2,
        coordinates_angstrom=coords,
        electron_density_au=round(rho, 6),
        laplacian_au=round(laplacian, 6),
        kinetic_energy_density_g_au=round(g_density, 6),
        potential_energy_density_v_au=round(v_density, 6),
        total_energy_density_h_au=round(h_density, 6),
        ellipticity=round(ellipticity, 6),
        interaction_classification=classification,
    )


def validate_poincare_hopf_invariant(
    n_ncp: int,
    n_bcp: int,
    n_rcp: int = 0,
    n_ccp: int = 0,
) -> bool:
    """
    [M] Enforce the global Poincaré-Hopf topological invariant on QTAIM critical points (REQ-QC-088).
    """
    euler_char = n_ncp - n_bcp + n_rcp - n_ccp
    if euler_char != 1:
        raise PoincareHopfViolationError(
            n_ncp=n_ncp,
            n_bcp=n_bcp,
            n_rcp=n_rcp,
            n_ccp=n_ccp,
            euler_char=euler_char,
        )
    return True


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _select_short_mw_dir(candidate: Path, environ: Mapping[str, str]) -> Tuple[Path, Path]:
    """Preserve short symlink paths (e.g. .../engines/mw) so Multiwfnpath stays < 80 chars."""
    abs_exe = Path(os.path.abspath(str(candidate)))
    mw_dir = abs_exe.parent
    if len(str(mw_dir)) <= _MULTIWFN_MAX_PATH_LEN:
        return abs_exe, mw_dir
    env_mw_path = environ.get("Multiwfnpath", "").strip()
    if env_mw_path:
        short_dir = Path(os.path.abspath(os.path.expanduser(env_mw_path)))
        if (
            len(str(short_dir)) <= _MULTIWFN_MAX_PATH_LEN
            and short_dir.is_dir()
            and os.path.samefile(short_dir, mw_dir)
        ):
            short_exe = short_dir / abs_exe.name
            if _is_executable_file(short_exe):
                return short_exe, short_dir
    raise RuntimeError(
        f"Multiwfnpath exceeds {_MULTIWFN_MAX_PATH_LEN}-character Fortran limit "
        f"({len(str(mw_dir))} chars): {mw_dir}"
    )


def resolve_multiwfn_binary(
    multiwfn_binary: Optional[str | Path] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    search_defaults: bool = True,
) -> Tuple[Path, Path]:
    """Resolve an authentic installed ``Multiwfn_noGUI`` binary and its directory.

    Fails closed with ``FileNotFoundError`` when no executable binary is present.
    Enforces Multiwfn's Fortran ``Multiwfnpath`` <80-character limit without
    dereferencing short directory symlinks such as ``.../engines/mw``.
    """
    environ = os.environ if env is None else env
    if multiwfn_binary is not None:
        explicit = Path(multiwfn_binary).expanduser()
        if not _is_executable_file(explicit):
            raise FileNotFoundError(
                f"Explicit Multiwfn binary is absent or not executable: {explicit}"
            )
        return _select_short_mw_dir(explicit, environ)

    candidates: List[Path] = []
    for var_name in ("COCHEM_MULTIWFN_BIN", "MULTIWFN_BIN"):
        val = environ.get(var_name, "").strip()
        if val:
            candidates.append(Path(val).expanduser())
    mw_path_env = environ.get("Multiwfnpath", "").strip()
    if mw_path_env:
        candidates.append(Path(mw_path_env).expanduser() / "Multiwfn_noGUI")

    path_str = environ.get("PATH", "")
    for name in ("Multiwfn_noGUI", "Multiwfn"):
        found = shutil.which(name, path=path_str)
        if found:
            candidates.append(Path(found))

    if search_defaults:
        candidates.append(
            Path.home() / "CoChem_Workstation_Direct" / "engines" / "mw" / "Multiwfn_noGUI"
        )

    for candidate in candidates:
        if _is_executable_file(candidate):
            return _select_short_mw_dir(candidate, environ)

    raise FileNotFoundError(
        "Multiwfn_noGUI binary not found. Set COCHEM_MULTIWFN_BIN / MULTIWFN_BIN / Multiwfnpath "
        "or install Multiwfn_noGUI in ~/CoChem_Workstation_Direct/engines/mw."
    )


def resolve_orca_2mkl_binary(
    orca_2mkl_binary: Optional[str | Path] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    search_defaults: bool = True,
) -> Path:
    """Resolve authentic ``orca_2mkl`` for converting ``.gbw`` wavefunctions to ``.molden.input``."""
    environ = os.environ if env is None else env
    if orca_2mkl_binary is not None:
        explicit = Path(orca_2mkl_binary).expanduser()
        if not _is_executable_file(explicit):
            raise FileNotFoundError(
                f"Explicit orca_2mkl binary is absent or not executable: {explicit}"
            )
        return explicit.resolve()

    candidates: List[Path] = []
    orca_dir = environ.get("ORCA_DIR", "").strip()
    if orca_dir:
        candidates.append(Path(orca_dir).expanduser() / "orca_2mkl")
    for var_name in ("COCHEM_ORCA_BIN", "ORCA_CMD", "ORCA_BIN"):
        val = environ.get(var_name, "").strip()
        if val:
            candidates.append(Path(val).expanduser().parent / "orca_2mkl")
    found = shutil.which("orca_2mkl", path=environ.get("PATH", ""))
    if found:
        candidates.append(Path(found))
    if search_defaults:
        candidates.append(
            Path.home()
            / ".local"
            / "share"
            / "cochem-topos"
            / "workstation-20261008"
            / "engines"
            / "orca-native-v1"
            / "orca_6_1_1_linux_x86-64_shared_openmpi418"
            / "orca_2mkl"
        )
    for candidate in candidates:
        if _is_executable_file(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "orca_2mkl binary not found; required to convert ORCA .gbw wavefunctions for Multiwfn."
    )


def probe_multiwfn(
    *,
    multiwfn_binary: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
    search_defaults: bool = True,
) -> Dict[str, Any]:
    """Observe whether authentic ``Multiwfn_noGUI`` is installed and usable on this host."""
    try:
        exe, mw_dir = resolve_multiwfn_binary(
            multiwfn_binary, env=env, search_defaults=search_defaults
        )
    except (FileNotFoundError, RuntimeError) as exc:
        return {"available": False, "engine": "Multiwfn", "reason": str(exc)}
    return {
        "available": True,
        "engine": "Multiwfn",
        "executable_path": str(exe),
        "multiwfnpath": str(mw_dir),
        "sha256": _sha256_file(exe),
    }


def _match_wavefunction_extension(path: Path) -> str:
    lower_name = path.name.lower()
    for ext in _MULTIWFN_SUPPORTED_EXTENSIONS:
        if lower_name.endswith(ext):
            return ext
    raise ValueError(
        f"Unsupported wavefunction file format {path.name!r}; supported extensions: "
        f"{', '.join(_MULTIWFN_SUPPORTED_EXTENSIONS)}."
    )


def _prepare_wavefunction_input(
    wavefunction_path: Path,
    run_dir: Path,
    *,
    orca_2mkl_binary: Optional[str | Path] = None,
    timeout_seconds: float = 120.0,
) -> Tuple[Path, str, Optional[str]]:
    """Stage wavefunction file into ``run_dir``; convert ``.gbw`` via authentic ``orca_2mkl``."""
    src = wavefunction_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Wavefunction file not found: {src}")
    ext = _match_wavefunction_extension(src)
    orig_sha = _sha256_file(src)

    if ext == ".gbw":
        mkl_exe = resolve_orca_2mkl_binary(orca_2mkl_binary)
        staged_gbw = run_dir / "wfn.gbw"
        shutil.copy2(src, staged_gbw)
        run_env = os.environ.copy()
        ld_parts: List[str] = []
        if run_env.get("COCHEM_ORCA_LD_LIBRARY_PATH"):
            ld_parts.append(run_env["COCHEM_ORCA_LD_LIBRARY_PATH"])
        ld_parts.append(str(mkl_exe.parent))
        if run_env.get("LD_LIBRARY_PATH"):
            ld_parts.append(run_env["LD_LIBRARY_PATH"])
        run_env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)
        proc = subprocess.run(
            [str(mkl_exe), "wfn", "-molden"],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            env=run_env,
            timeout=timeout_seconds,
            check=False,
        )
        converted = run_dir / "wfn.molden.input"
        if proc.returncode != 0 or not converted.is_file() or converted.stat().st_size == 0:
            raise RuntimeError(
                f"orca_2mkl failed converting {src.name} (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-400:]}"
            )
        return converted, orig_sha, _sha256_file(converted)

    staged = run_dir / f"wfn{ext}"
    shutil.copy2(src, staged)
    return staged, orig_sha, None


def _verify_multiwfn_banner(stdout_text: str) -> str:
    """Verify authentic Multiwfn banner in stdout and return parsed version string."""
    if "Multiwfn -- A Multifunctional Wavefunction Analyzer" not in stdout_text:
        raise RuntimeError(
            "Authentic Multiwfn banner not found in Multiwfn_noGUI stdout."
        )
    ver_match = re.search(r"Version\s+([0-9][0-9A-Za-z._-]+)", stdout_text)
    if ver_match is None:
        raise RuntimeError("Multiwfn version line not found in Multiwfn_noGUI stdout.")
    return ver_match.group(1)


def _execute_multiwfn(
    mw_exe: Path,
    mw_dir: Path,
    staged_wfn: Path,
    batch_stdin: str,
    run_dir: Path,
    *,
    n_threads: int = 2,
    timeout_seconds: float = 180.0,
) -> Tuple[str, str]:
    """Run ``Multiwfn_noGUI`` in ``run_dir`` and return ``(stdout_text, version)``."""
    run_env = os.environ.copy()
    run_env["Multiwfnpath"] = str(mw_dir)
    run_env["OMP_NUM_THREADS"] = str(max(1, int(n_threads)))
    run_env.setdefault("OMP_STACKSIZE", "200M")
    proc = subprocess.run(
        [str(mw_exe), staged_wfn.name],
        input=batch_stdin,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        env=run_env,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Multiwfn_noGUI exited with non-zero status {proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[-500:]}"
        )
    version_str = _verify_multiwfn_banner(proc.stdout)
    return proc.stdout, version_str


def _parse_stdout_bcp_connections(
    stdout_text: str,
) -> Dict[int, Tuple[Tuple[int, int], Tuple[str, str]]]:
    """Parse fallback BCP -> connected-atom pairs from Multiwfn stdout summary."""
    mapping: Dict[int, Tuple[Tuple[int, int], Tuple[str, str]]] = {}
    pattern = re.compile(
        r"^\s*(\d+)\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+"
        r"\(3,-1\)\s+(\d+)\(\s*([A-Za-z]{1,2})\s*\)\s*--\s*(\d+)\(\s*([A-Za-z]{1,2})\s*\)"
    )
    for line in stdout_text.splitlines():
        match = pattern.match(line)
        if match is not None:
            cp_idx = int(match.group(1))
            a1, s1, a2, s2 = (
                int(match.group(2)),
                match.group(3),
                int(match.group(4)),
                match.group(5),
            )
            mapping[cp_idx] = ((a1 - 1, a2 - 1), (s1, s2))
    return mapping


def _parse_float_after_colon(block: str, label: str) -> float:
    match = re.search(
        re.escape(label) + r"\s*([-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?)",
        block,
    )
    if match is None:
        raise RuntimeError(f"Failed to parse {label!r} from Multiwfn CPprop.txt.")
    val = float(match.group(1).replace("D", "E").replace("d", "e"))
    if not math.isfinite(val):
        raise RuntimeError(f"Non-finite value for {label!r} in Multiwfn CPprop.txt.")
    return val


def _parse_triple_after_colon(block: str, label: str) -> Tuple[float, float, float]:
    num = r"([-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?)"
    match = re.search(
        re.escape(label) + rf"\s*{num}\s+{num}\s+{num}",
        block,
    )
    if match is None:
        raise RuntimeError(f"Failed to parse 3-vector {label!r} from Multiwfn CPprop.txt.")
    vals = tuple(
        float(g.replace("D", "E").replace("d", "e")) for g in match.groups()
    )
    if not all(math.isfinite(v) for v in vals):
        raise RuntimeError(f"Non-finite vector for {label!r} in Multiwfn CPprop.txt.")
    return (vals[0], vals[1], vals[2])


def parse_multiwfn_cpprop(
    cpprop_text: str,
    stdout_text: str = "",
) -> Tuple[
    List[CriticalPoint],
    List[BondCriticalPointDescriptor],
    List[Tuple[int, int]],
    List[Tuple[str, str]],
    Dict[str, int],
]:
    """Parse Multiwfn ``CPprop.txt`` into CriticalPoints and classified BCP descriptors."""
    header_re = re.compile(
        r"^\s*-+\s+CP\s+(\d+),\s+Type\s+(\(3,[-+][13]\))\s+-+\s*$",
        re.MULTILINE,
    )
    matches = list(header_re.finditer(cpprop_text))
    if not matches:
        raise RuntimeError("No critical point blocks found in Multiwfn CPprop.txt.")

    stdout_conns = _parse_stdout_bcp_connections(stdout_text)
    cp_type_to_name: Dict[str, Literal["NCP", "BCP", "RCP", "CCP"]] = {
        "(3,-3)": "NCP",
        "(3,-1)": "BCP",
        "(3,+1)": "RCP",
        "(3,+3)": "CCP",
    }
    counts = {"NCP": 0, "BCP": 0, "RCP": 0, "CCP": 0}
    critical_points: List[CriticalPoint] = []
    bcps: List[BondCriticalPointDescriptor] = []
    bcp_atom_pairs: List[Tuple[int, int]] = []
    bcp_symbol_pairs: List[Tuple[str, str]] = []

    conn_re = re.compile(
        r"Connected atoms:\s*(\d+)\(\s*([A-Za-z]{1,2})\s*\)\s*--\s*(\d+)\(\s*([A-Za-z]{1,2})\s*\)"
    )

    for idx, match in enumerate(matches):
        cp_num = int(match.group(1))
        cp_type_raw = match.group(2)
        if cp_type_raw not in cp_type_to_name:
            continue
        cp_type: Literal["(3,-3)", "(3,-1)", "(3,+1)", "(3,+3)"] = cp_type_raw  # type: ignore[assignment]
        cp_name = cp_type_to_name[cp_type]
        block_start = match.end()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cpprop_text)
        block = cpprop_text[block_start:block_end]

        coords_ang = _parse_triple_after_colon(block, "Position (Angstrom):")
        rho = _parse_float_after_colon(block, "Density of all electrons:")
        laplacian = _parse_float_after_colon(block, "Laplacian of electron density:")
        hess_eigs = _parse_triple_after_colon(block, "Eigenvalues of Hessian:")

        counts[cp_name] += 1
        critical_points.append(
            CriticalPoint(
                cp_type=cp_type,
                cp_name=cp_name,
                coordinates_angstrom=coords_ang,
                density_au=rho,
                laplacian_au=laplacian,
                eigenvalues=hess_eigs,
            )
        )

        if cp_name == "BCP":
            g_val = _parse_float_after_colon(block, "Lagrangian kinetic energy G(r):")
            v_val = _parse_float_after_colon(block, "Potential energy density V(r):")
            conn_match = conn_re.search(block)
            if conn_match is not None:
                a1 = int(conn_match.group(1)) - 1
                s1 = conn_match.group(2)
                a2 = int(conn_match.group(3)) - 1
                s2 = conn_match.group(4)
            elif cp_num in stdout_conns:
                (a1, a2), (s1, s2) = stdout_conns[cp_num]
            else:
                raise RuntimeError(
                    f"Connected atom pair missing for BCP {cp_num} in Multiwfn output."
                )

            bcp_desc = classify_bond_critical_point(
                bcp_idx=len(bcps),
                atom_1=a1,
                atom_2=a2,
                coords=coords_ang,
                rho=rho,
                hessian_eigenvals=hess_eigs,
                g_density=g_val,
                v_density=v_val,
            )
            bcps.append(bcp_desc)
            bcp_atom_pairs.append((a1, a2))
            bcp_symbol_pairs.append((s1, s2))

    return critical_points, bcps, bcp_atom_pairs, bcp_symbol_pairs, counts


def run_multiwfn_qtaim(
    wavefunction_path: str | Path,
    *,
    multiwfn_binary: Optional[str | Path] = None,
    orca_2mkl_binary: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    enforce_poincare_hopf: bool = True,
    sphere_points: int = 200,
    timeout_seconds: float = 180.0,
    n_threads: int = 2,
) -> MultiwfnQTAIMResult:
    """Execute authentic ``Multiwfn_noGUI`` QTAIM critical-point analysis on a wavefunction.

    Supports ``.molden``, ``.molden.input``, ``.fchk``, ``.wfx``, ``.wfn``, and
    ORCA ``.gbw`` (automatically converted via authentic ``orca_2mkl``).
    Fails closed with ``FileNotFoundError`` / ``RuntimeError`` if Multiwfn is missing
    or execution fails.
    """
    mw_exe, mw_dir = resolve_multiwfn_binary(multiwfn_binary)
    wfn_path = Path(wavefunction_path).expanduser().resolve()

    if sphere_points > 0:
        batch_stdin = (
            f"2\n2\n3\n4\n5\n6\n11\n{int(sphere_points)}\n-1\n-9\n8\n7\n-1\n0\n-10\nq\n"
        )
    else:
        batch_stdin = "2\n2\n3\n4\n5\n8\n7\n-1\n0\n-10\nq\n"

    def _run_in_dir(run_root: Path) -> MultiwfnQTAIMResult:
        staged_wfn, wfn_sha, molden_sha = _prepare_wavefunction_input(
            wfn_path,
            run_root,
            orca_2mkl_binary=orca_2mkl_binary,
            timeout_seconds=timeout_seconds,
        )
        stdout_text, mw_version = _execute_multiwfn(
            mw_exe,
            mw_dir,
            staged_wfn,
            batch_stdin,
            run_root,
            n_threads=n_threads,
            timeout_seconds=timeout_seconds,
        )
        stdout_file = run_root / "multiwfn_qtaim.out"
        stdout_file.write_text(stdout_text, encoding="utf-8")

        cpprop_path = run_root / "CPprop.txt"
        if not cpprop_path.is_file() or cpprop_path.stat().st_size == 0:
            if "You must at least locate one (3,-1) CP" in stdout_text:
                raise RuntimeError(
                    "Multiwfn_noGUI located zero (3,-1) bond critical points "
                    "(wavefunction basis or geometry may lack a bond path)."
                )
            raise RuntimeError(
                "Multiwfn_noGUI completed without producing a non-empty CPprop.txt."
            )
        cpprop_bytes = cpprop_path.read_bytes()
        cpprop_text = cpprop_bytes.decode("utf-8", errors="replace")

        cps, bcps, atom_pairs, sym_pairs, counts = parse_multiwfn_cpprop(
            cpprop_text, stdout_text
        )
        n_ncp = counts["NCP"]
        n_bcp = counts["BCP"]
        n_rcp = counts["RCP"]
        n_ccp = counts["CCP"]
        chi = n_ncp - n_bcp + n_rcp - n_ccp
        ph_ok = chi == 1
        if enforce_poincare_hopf:
            validate_poincare_hopf_invariant(
                n_ncp=n_ncp, n_bcp=n_bcp, n_rcp=n_rcp, n_ccp=n_ccp
            )

        graph = QTAIMTopologyGraph(
            n_ncp=n_ncp,
            n_bcp=n_bcp,
            n_rcp=n_rcp,
            n_ccp=n_ccp,
            is_poincare_hopf_satisfied=ph_ok,
            bcps=bcps,
        )
        mw_bin_sha = _sha256_file(mw_exe)
        stdout_sha = _sha256_bytes(stdout_text.encode("utf-8"))
        cpprop_sha = _sha256_bytes(cpprop_bytes)
        core_payload = {
            "schema_version": _QTAIM_SCHEMA,
            "multiwfn_version": mw_version,
            "multiwfn_binary_sha256": mw_bin_sha,
            "wavefunction_sha256": wfn_sha,
            "converted_molden_sha256": molden_sha,
            "multiwfn_stdout_sha256": stdout_sha,
            "cpprop_sha256": cpprop_sha,
            "n_ncp": n_ncp,
            "n_bcp": n_bcp,
            "n_rcp": n_rcp,
            "n_ccp": n_ccp,
            "poincare_hopf_chi": chi,
            "bcps": [b.model_dump(mode="json") for b in bcps],
        }
        return MultiwfnQTAIMResult(
            multiwfn_version=mw_version,
            multiwfn_binary=str(mw_exe),
            multiwfn_binary_sha256=mw_bin_sha,
            wavefunction_path=str(wfn_path),
            wavefunction_sha256=wfn_sha,
            converted_molden_sha256=molden_sha,
            multiwfn_stdout_sha256=stdout_sha,
            cpprop_sha256=cpprop_sha,
            n_ncp=n_ncp,
            n_bcp=n_bcp,
            n_rcp=n_rcp,
            n_ccp=n_ccp,
            poincare_hopf_chi=chi,
            is_poincare_hopf_satisfied=ph_ok,
            topology_graph=graph,
            critical_points=cps,
            bcp_atom_pairs=atom_pairs,
            bcp_symbol_pairs=sym_pairs,
            result_sha256=_sha256_bytes(_canonical_json_bytes(core_payload)),
        )

    if work_dir is not None:
        run_root = Path(work_dir).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        return _run_in_dir(run_root)

    with tempfile.TemporaryDirectory(prefix="spyc_mw_qtaim_") as tmp_dir:
        return _run_in_dir(Path(tmp_dir))


def run_multiwfn_nci(
    wavefunction_path: str | Path,
    *,
    multiwfn_binary: Optional[str | Path] = None,
    orca_2mkl_binary: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    grid_quality: Literal[1, 2, 3] = 1,
    rdg_cutoff: float = 0.5,
    rho_cutoff: float = 0.05,
    vdw_threshold: float = 0.005,
    timeout_seconds: float = 180.0,
    n_threads: int = 2,
) -> MultiwfnNCIResult:
    """Execute authentic ``Multiwfn_noGUI`` NCI (Reduced Density Gradient) analysis.

    Parses Multiwfn's ``output.txt`` (columns: X, Y, Z, sign(lambda2)*rho, RDG),
    partitions points within ``RDG <= rdg_cutoff`` and ``|sign(lambda2)*rho| <= rho_cutoff``
    into attractive (``< -vdw_threshold``), van der Waals (``[-vdw_threshold, +vdw_threshold]``),
    and steric (``> +vdw_threshold``) interaction regimes, and emits SHA-256 provenance.
    """
    if grid_quality not in (1, 2, 3):
        raise ValueError(f"grid_quality must be 1, 2, or 3; got {grid_quality!r}.")
    if rdg_cutoff <= 0.0 or rho_cutoff <= 0.0 or vdw_threshold <= 0.0:
        raise ValueError("rdg_cutoff, rho_cutoff, and vdw_threshold must be positive.")

    mw_exe, mw_dir = resolve_multiwfn_binary(multiwfn_binary)
    wfn_path = Path(wavefunction_path).expanduser().resolve()
    batch_stdin = f"20\n1\n{int(grid_quality)}\n2\n0\n0\nq\n"

    def _run_in_dir(run_root: Path) -> MultiwfnNCIResult:
        staged_wfn, wfn_sha, molden_sha = _prepare_wavefunction_input(
            wfn_path,
            run_root,
            orca_2mkl_binary=orca_2mkl_binary,
            timeout_seconds=timeout_seconds,
        )
        stdout_text, mw_version = _execute_multiwfn(
            mw_exe,
            mw_dir,
            staged_wfn,
            batch_stdin,
            run_root,
            n_threads=n_threads,
            timeout_seconds=timeout_seconds,
        )
        stdout_file = run_root / "multiwfn_nci.out"
        stdout_file.write_text(stdout_text, encoding="utf-8")

        nci_out_path = run_root / "output.txt"
        if not nci_out_path.is_file() or nci_out_path.stat().st_size == 0:
            raise RuntimeError(
                "Multiwfn_noGUI NCI completed without producing a non-empty output.txt."
            )
        nci_sha = _sha256_file(nci_out_path)
        data = np.loadtxt(nci_out_path, usecols=(3, 4), dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, 2)
        if data.shape[1] != 2 or len(data) == 0 or not np.isfinite(data).all():
            raise RuntimeError("Invalid or non-finite NCI scatter table in output.txt.")

        sign_l2_rho = data[:, 0]
        rdg = data[:, 1]
        n_total = int(len(data))
        low_rdg_mask = rdg <= rdg_cutoff
        n_low_rdg = int(np.count_nonzero(low_rdg_mask))
        nci_mask = low_rdg_mask & (np.abs(sign_l2_rho) <= rho_cutoff)
        nci_vals = sign_l2_rho[nci_mask]
        n_nci = int(len(nci_vals))

        n_attr = int(np.count_nonzero(nci_vals < -vdw_threshold))
        n_vdw = int(np.count_nonzero((nci_vals >= -vdw_threshold) & (nci_vals <= vdw_threshold)))
        n_steric = int(np.count_nonzero(nci_vals > vdw_threshold))

        if n_nci > 0:
            attr_frac = n_attr / n_nci
            vdw_frac = n_vdw / n_nci
            steric_frac = n_steric / n_nci
            mean_nci = float(np.mean(nci_vals))
        else:
            attr_frac = 0.0
            vdw_frac = 0.0
            steric_frac = 0.0
            mean_nci = None

        mw_bin_sha = _sha256_file(mw_exe)
        stdout_sha = _sha256_bytes(stdout_text.encode("utf-8"))
        min_sl2r = float(np.min(sign_l2_rho))
        max_sl2r = float(np.max(sign_l2_rho))
        min_rdg_val = max(0.0, float(np.min(rdg)))

        core_payload = {
            "schema_version": _NCI_SCHEMA,
            "multiwfn_version": mw_version,
            "multiwfn_binary_sha256": mw_bin_sha,
            "wavefunction_sha256": wfn_sha,
            "converted_molden_sha256": molden_sha,
            "multiwfn_stdout_sha256": stdout_sha,
            "nci_output_sha256": nci_sha,
            "grid_quality": int(grid_quality),
            "n_grid_points": n_total,
            "n_nci_points": n_nci,
            "n_attractive_points": n_attr,
            "n_vdw_points": n_vdw,
            "n_steric_points": n_steric,
        }
        return MultiwfnNCIResult(
            multiwfn_version=mw_version,
            multiwfn_binary=str(mw_exe),
            multiwfn_binary_sha256=mw_bin_sha,
            wavefunction_path=str(wfn_path),
            wavefunction_sha256=wfn_sha,
            converted_molden_sha256=molden_sha,
            multiwfn_stdout_sha256=stdout_sha,
            nci_output_sha256=nci_sha,
            grid_quality=int(grid_quality),
            rdg_cutoff=float(rdg_cutoff),
            rho_cutoff=float(rho_cutoff),
            vdw_threshold=float(vdw_threshold),
            n_grid_points=n_total,
            n_low_rdg_points=n_low_rdg,
            n_nci_points=n_nci,
            n_attractive_points=n_attr,
            n_vdw_points=n_vdw,
            n_steric_points=n_steric,
            attractive_fraction=attr_frac,
            vdw_fraction=vdw_frac,
            steric_fraction=steric_frac,
            min_sign_lambda2_rho=min_sl2r,
            max_sign_lambda2_rho=max_sl2r,
            min_rdg=min_rdg_val,
            mean_nci_sign_lambda2_rho=mean_nci,
            result_sha256=_sha256_bytes(_canonical_json_bytes(core_payload)),
        )

    if work_dir is not None:
        run_root = Path(work_dir).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        return _run_in_dir(run_root)

    with tempfile.TemporaryDirectory(prefix="spyc_mw_nci_") as tmp_dir:
        return _run_in_dir(Path(tmp_dir))


__all__ = [
    "BCPDescriptor",
    "CriticalPoint",
    "MultiwfnNCIResult",
    "MultiwfnQTAIMResult",
    "PoincareHopfViolationError",
    "QTAIMTopologyGraph",
    "classify_bond_critical_point",
    "parse_multiwfn_cpprop",
    "probe_multiwfn",
    "resolve_multiwfn_binary",
    "resolve_orca_2mkl_binary",
    "run_multiwfn_nci",
    "run_multiwfn_qtaim",
    "validate_poincare_hopf_invariant",
]

