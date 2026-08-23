import os
import sys
import json
import logging
import subprocess
import shutil
import tempfile
import hashlib
import psutil
import atexit
import re
from pathlib import Path
import numpy as np
import h5py
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any

from cochem_spycfit.quantum_extractor import QuantumExtractor, apply_mass_weighting, MASS_DICT

artifact_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", str(Path.home() / ".cochem" / "artifacts"))).resolve()
artifact_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(artifact_dir / 'master.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSION_FACTOR_MHZ = 505379.008  # MHz * amu * Å^2

def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def safe_delete(filepath: Path) -> None:
    trash_dir = Path(os.getenv('COCHEM_TRASH_DIR', str(artifact_dir / '.trash')))
    trash_dir.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        shutil.move(str(filepath), str(trash_dir / filepath.name))

def cleanup_zombies():
    for proc in psutil.process_iter(['pid', 'status']):
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                proc.wait(timeout=1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass

atexit.register(cleanup_zombies)

def check_spin_contamination(stdout: str) -> None:
    for line in stdout.split('\n'):
        if "Expectation value of <S**2>" in line:
            parts = line.split(':')
            if len(parts) == 2:
                s2_val = float(parts[1].strip())
                ideal = round((np.sqrt(1 + 4 * s2_val) - 1) / 2 * 2) / 2
                ideal_s2 = ideal * (ideal + 1)
                if ideal_s2 == 0:
                    if s2_val > 0.10:
                        raise ValueError(f"[HARD_ABORT: PHYSICS WALL] Spin contamination exceeds 0.10 for singlet. <S**2> = {s2_val}")
                else:
                    contamination = abs(s2_val - ideal_s2) / ideal_s2
                    if contamination > 0.10:
                        raise ValueError(f"[HARD_ABORT: PHYSICS WALL] Spin contamination exceeds 10%. <S**2> = {s2_val}, Ideal = {ideal_s2}")

def find_monomers_and_weak_bond(coords: list) -> tuple[list[int], list[int], tuple[int, int]]:
    radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02}
    n = len(coords)
    adj = np.zeros((n, n), dtype=bool)
    pos = np.array([[x, y, z] for sym, x, y, z in coords])
    syms = [sym for sym, x, y, z in coords]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            cov_dist = radii.get(syms[i], 0.7) + radii.get(syms[j], 0.7)
            if dist < cov_dist * 1.3:
                adj[i, j] = adj[j, i] = True
                
    visited = set()
    monomers = []
    for i in range(n):
        if i not in visited:
            comp = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    for j in range(n):
                        if adj[node, j] and j not in visited:
                            stack.append(j)
            monomers.append(comp)
            
    if len(monomers) != 2:
        raise ValueError(f"[HARD_ABORT: PHYSICS WALL] Expected exactly 2 monomers for LAM scan, found {len(monomers)}.")

    min_dist = float('inf')
    weak_bond = (monomers[0][-1], monomers[1][0])
    for i in monomers[0]:
        for j in monomers[1]:
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < min_dist:
                min_dist = dist
                weak_bond = (i, j)
                
    return monomers[0], monomers[1], weak_bond

class RotationalConstants(BaseModel):
    A: float
    B: float
    C: float

class LAMResults(BaseModel):
    r_optimal_angstrom: float
    min_energy_hartree: float
    distances: List[float]
    energies: List[float]
    rotational_constants_Be_MHz: RotationalConstants
    delta_B_vib_MHz: RotationalConstants
    rotational_constants_B0_MHz: RotationalConstants
    rotational_constants_MHz: RotationalConstants
    jensen_bias_MHz: RotationalConstants
    dipole_moments_debye: Dict[str, float]

class IsoHessianResult(BaseModel):
    isotopologue_masses_amu: List[float]
    mass_weighted_hessian: List[List[float]]
    frequencies_cm1: List[float]
    eigenvalues: List[float]
    cartesian_modes: List[List[float]]
    zpe_cm1: float
    zero_el_cost: bool
    has_imaginary_frequencies: bool
    imaginary_frequencies_count: int
    imaginary_frequencies: List[float]

class OrcaPesInputConfig(BaseModel):
    r_min: float = Field(..., ge=0.5, description="Minimum distance in Angstroms")
    r_max: float = Field(..., description="Maximum distance in Angstroms")
    n_steps: int = Field(..., ge=1, description="Number of steps in the PES scan")
    tier: str = Field("wB97M-V def2-QZVPP", description="Method Matrix v4 Tier selection")
    nprocs: int = Field(7, description="Number of processors (Method Matrix v4 P-core cap)")
    maxcore: int = Field(3400, description="Max core per proc in MB")
    charge: int = Field(0, description="Molecular charge")
    mult: int = Field(1, description="Spin multiplicity")
    
    @model_validator(mode='after')
    def check_config(self):
        if self.r_min >= self.r_max:
            raise ValueError("r_min must be strictly less than r_max")
        dispersion_keys = ['D3', 'D4', '-V', '-D']
        if not any(k in self.tier.upper() for k in dispersion_keys):
            raise ValueError("[HARD_ABORT: PHYSICS WALL] Dispersion correction (D3/D4 or -V/-D functional) is mandatory for weak complexes.")
        return self

def compute_vibrationally_averaged_constants(inertia_tensors_q: np.ndarray, weights_q: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights_q, dtype=float)
    if np.sum(weights) == 0:
        raise ValueError("[HARD_ABORT: PHYSICS WALL] Sum of vibrational weights is zero. Invalid probability distribution.")
    weights = weights / np.sum(weights)

    tensors = np.asarray(inertia_tensors_q, dtype=float)
    if tensors.ndim == 3 and tensors.shape[1:] == (3, 3):
        inv_inertia_q = np.linalg.inv(tensors)
        avg_inv_inertia = np.sum(inv_inertia_q * weights[:, None, None], axis=0)
        eigvals = np.linalg.eigvalsh(avg_inv_inertia)
        rotational_constants_mhz = CONVERSION_FACTOR_MHZ * eigvals[::-1]
    elif tensors.ndim == 2 and tensors.shape[1] == 3:
        inv_inertia_q = 1.0 / tensors
        avg_inv_inertia = np.sum(inv_inertia_q * weights[:, None], axis=0)
        rotational_constants_mhz = CONVERSION_FACTOR_MHZ * np.sort(avg_inv_inertia)[::-1]
    else:
        raise ValueError("inertia_tensors_q must have shape (N_q, 3, 3) or (N_q, 3).")

    return rotational_constants_mhz

def reweight_isotopologue_hessian(cartesian_hessian : np.ndarray, 
                                  symbols: list, 
                                  mass_overrides: Optional[dict] = None, 
                                  new_symbols: Optional[list] = None) -> dict:
    mass_map = dict(MASS_DICT)
    mass_map.update({
        'D': 2.01410178, 'H2': 2.01410178,
        '2H': 2.01410178, '1H': 1.007825, '2D': 2.01410178, '3D': 3.016049,
        '12C': 12.000000, '13C': 13.003355, '14C': 14.003242,
        '14N': 14.003074, '15N': 15.000109,
        '16O': 15.994915, '17O': 16.999131, '18O': 17.999160,
        '35Cl': 34.968853, '37Cl': 36.965903,
        '79Br': 78.918337, '81Br': 80.916290
    })
    
    def resolve_symbol_mass(sym: str) -> float:
        s = str(sym).strip()
        if s in mass_map:
            return float(mass_map[s])
        match = re.match(r'^(\d+)([A-Za-z]+)$', s)
        if match:
            return float(match.group(1))
        return float(mass_map.get(s, 12.0))

    n_atoms = len(symbols)
    masses = np.array([resolve_symbol_mass(sym) for sym in symbols], dtype=float)
    
    if new_symbols is not None and len(new_symbols) == n_atoms:
        masses = np.array([resolve_symbol_mass(sym) for sym in new_symbols], dtype=float)
        
    if mass_overrides:
        for idx, m_val in mass_overrides.items():
            if 0 <= idx < n_atoms:
                masses[idx] = float(m_val)
                
    m_3n = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)
    weight_matrix = np.outer(inv_sqrt_m, inv_sqrt_m)
    
    mw_hessian = cartesian_hessian * weight_matrix
    mw_hessian = 0.5 * (mw_hessian + mw_hessian.T)
    
    eigvals, eigvecs = np.linalg.eigh(mw_hessian)
    
    HARTREE_BOHR2_AMU_TO_CM1 = 5140.48714
    freqs_cm1 = np.where(eigvals >= 0.0, np.sqrt(np.maximum(eigvals, 0.0)), -np.sqrt(np.maximum(-eigvals, 0.0))) * HARTREE_BOHR2_AMU_TO_CM1
    modes_cartesian = eigvecs * inv_sqrt_m[:, np.newaxis]
    zpe_cm1 = 0.5 * float(np.sum(freqs_cm1[freqs_cm1 > 10.0]))
    
    has_imaginary_freqs = bool(np.any(eigvals < 0.0))
    imag_count = int(np.sum(eigvals < 0.0))
    imag_freqs = freqs_cm1[eigvals < 0.0].tolist() if has_imaginary_freqs else []
    
    if has_imaginary_freqs:
        logging.warning(f"Imaginary frequencies detected: {imag_count} mode(s) with negative eigenvalues.")
    
    res = IsoHessianResult(
        isotopologue_masses_amu=masses.tolist(),
        mass_weighted_hessian=mw_hessian.tolist(),
        frequencies_cm1=freqs_cm1.tolist(),
        eigenvalues=eigvals.tolist(),
        cartesian_modes=modes_cartesian.tolist(),
        zpe_cm1=zpe_cm1,
        zero_el_cost=True,
        has_imaginary_frequencies=has_imaginary_freqs,
        imaginary_frequencies_count=imag_count,
        imaginary_frequencies=imag_freqs
    )
    return res.model_dump()

class SpycFitMaster:
    def __init__(self, h5_path: Optional[str] = None) -> None:
        if h5_path is None:
            h5_path = os.environ.get("COCHEM_TARGET_H5", str(artifact_dir / "landscape.h5"))
        self.h5_path = Path(h5_path)
        self.extractor = QuantumExtractor()

    def check_lam_trigger(self) -> bool:
        if self.h5_path.exists():
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    for grp_name in f.keys():
                        grp = f[grp_name]
                        if isinstance(grp, h5py.Group) and grp.attrs.get("LAM_TRIGGER_REQUIRED", False):
                            return True
            except Exception as e:
                raise RuntimeError(f"Error checking HDF5 for LAM trigger: {e}") from e

        buffer_path = artifact_dir / "spycfit_tensor_buffer.npz"
        if buffer_path.exists():
            try:
                with np.load(buffer_path, allow_pickle=True) as data:
                    if 'lam_trigger' in data and bool(data['lam_trigger']):
                        return True
            except Exception as e:
                raise RuntimeError(f"Error loading buffer for LAM trigger: {e}") from e
        return False

    def get_system_coords(self) -> list:
        coords = []
        if self.h5_path.exists():
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    for grp_name in f.keys():
                        grp = f[grp_name]
                        if isinstance(grp, h5py.Group) and grp.attrs.get("LAM_TRIGGER_REQUIRED", False):
                            if "symbols" in grp and "geometry" in grp:
                                symbols = grp["symbols"][:]
                                geometry = grp["geometry"][:]
                                for sym, pos in zip(symbols, geometry):
                                    if isinstance(sym, bytes):
                                        sym = sym.decode('utf-8')
                                    coords.append((sym, float(pos[0]), float(pos[1]), float(pos[2])))
                                return coords
            except Exception as e:
                raise RuntimeError(f"Error extracting coordinates from HDF5: {e}") from e
                
        buffer_path = artifact_dir / "spycfit_tensor_buffer.npz"
        if buffer_path.exists():
            try:
                with np.load(buffer_path, allow_pickle=True) as data:
                    if 'lam_trigger' in data and bool(data['lam_trigger']):
                        if 'symbols' in data and 'geometry' in data:
                            symbols = data['symbols']
                            geometry = data['geometry']
                            for sym, pos in zip(symbols, geometry):
                                if isinstance(sym, bytes):
                                    sym = sym.decode('utf-8')
                                coords.append((sym, float(pos[0]), float(pos[1]), float(pos[2])))
                            return coords
            except Exception as e:
                raise RuntimeError(f"Error extracting coordinates from npz: {e}") from e
                
        if not coords:
            raise ValueError("[MISSING DATA] Could not extract real physical coordinates for LAM scan. Fake data is prohibited.")
        return coords

    def generate_orca_1d_pes_input(self, coords: list, config: OrcaPesInputConfig) -> str:
        method_header = f"! {config.tier} def2/J RIJCOSX TightOPT TightSCF DEFGRID3"
        
        monomer1, monomer2, weak_bond = find_monomers_and_weak_bond(coords)
            
        inp = [
            method_header,
            f"%pal nprocs {config.nprocs} end",
            f"%maxcore {config.maxcore}",
            "%geom",
            "  InHess XTB2",
            "  TolE     1e-7",
            "  TolRMSG  3e-6",
            "  TolMaxG  1e-5",
            "  TolRMSD  5e-5",
            "  TolMaxD  1e-4",
            "  Scan",
            f"    B {weak_bond[0]} {weak_bond[1]} = {config.r_min:.2f}, {config.r_max:.2f}, {config.n_steps}",
            "  end",
            "  Constraints"
        ]
        
        pos = np.array([[x,y,z] for s,x,y,z in coords])
        for m in (monomer1, monomer2):
            for i in range(len(m)):
                for j in range(i+1, len(m)):
                    dist = np.linalg.norm(pos[m[i]] - pos[m[j]])
                    if dist < 2.0:
                        inp.append(f"    {{ B {m[i]} {m[j]} C }}")
                        
        inp.extend([
            "  end",
            "end",
            "",
            f"* xyz {config.charge} {config.mult}"
        ])
        for sym, x, y, z in coords:
            inp.append(f"  {sym:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        inp.append("*")
        return "\n".join(inp)

    def execute_lam_correction(self, coords: list, config: OrcaPesInputConfig) -> dict:
        orca_bin = shutil.which("orca")
        if not orca_bin:
            logger.error("ORCA binary not found in PATH.")
            return {"lam_active": True, "error": "orca executable not found", "status": "failed"}
            
        logger.info(f"⚡ [SpycFitMaster] Launching genuine ORCA 1D PES scan ({config.r_min:.1f} - {config.r_max:.1f} Å)...")
        logger.warning("[AUDIT] BSSE and Frozen-Core bias must be explicitly quantified per MMv4.")

        inp_str = self.generate_orca_1d_pes_input(coords, config)
        
        scan_dir = artifact_dir / "cochem_lam_scan"
        scan_dir.mkdir(parents=True, exist_ok=True)
        inp_path = scan_dir / "scan.inp"
        inp_path.write_text(inp_str, encoding="utf-8")
        
        try:
            result = subprocess.run([orca_bin, inp_path.name], cwd=str(scan_dir), check=True, capture_output=True, text=True, timeout=3600)
            logger.info("✅ [SpycFitMaster] ORCA execution completed normally.")
            out_file = scan_dir / "scan.out"
            out_file.write_text(result.stdout, encoding="utf-8")
            gbw_file = scan_dir / "scan.gbw"
            
            check_spin_contamination(result.stdout)
            
            out_hash = calculate_artifact_sha256(out_file) if out_file.exists() else None
            gbw_hash = calculate_artifact_sha256(gbw_file) if gbw_file.exists() else None
            logger.info(f"Provenance hashes: .out={out_hash}, .gbw={gbw_hash}")
            return {"lam_active": True, "artifact_path": str(out_file), "status": "completed", "out_hash": out_hash, "gbw_hash": gbw_hash}
        except subprocess.TimeoutExpired:
            logger.error("ORCA execution timed out after 3600s")
            cleanup_zombies()
            return {"lam_active": True, "error": "execution timed out after 3600s", "status": "failed"}
        except subprocess.CalledProcessError as e:
            logger.error(f"ORCA execution failed: {e.stderr}")
            return {"lam_active": True, "error": f"execution failed with code {e.returncode}", "status": "failed"}

    def run(self) -> dict:
        lam_needed = self.check_lam_trigger()
        if lam_needed:
            coords = self.get_system_coords()
            charge = int(os.environ.get("COCHEM_CHARGE", 0))
            mult = int(os.environ.get("COCHEM_MULT", 1))
            config = OrcaPesInputConfig(r_min=1.5, r_max=3.0, n_steps=15, charge=charge, mult=mult)
            return self.execute_lam_correction(coords, config)
        else:
            logger.info("ℹ️ [SpycFitMaster] Standard rigid-rotor pathway (LAM not required).")
            return {"lam_active": False}

def execute_lam_correction(coords: list, config: OrcaPesInputConfig) -> dict:
    master = SpycFitMaster()
    return master.execute_lam_correction(coords, config)

if __name__ == "__main__":
    master = SpycFitMaster()
    res = master.run()
    logger.info(f"Master Execution Result: {res}")