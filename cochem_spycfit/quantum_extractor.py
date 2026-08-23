#!/usr/bin/env python3
import os
import sys
import logging
import json
import re
import hashlib
import subprocess
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import psutil
    import numpy as np
    import h5py
except ImportError as e:
    print(f"❌ FATAL: 'psutil', 'numpy', and 'h5py' are required. Details: {e}")
    sys.exit(1)

from pydantic import BaseModel, Field

# Setup rigorous logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# Resolves SPYCFIT-19: Environment-derived artifact directory for logging
artifact_dir_env = os.environ.get("COCHEM_ARTIFACT_DIR")
if artifact_dir_env:
    artifact_dir = Path(artifact_dir_env).resolve()
else:
    artifact_dir = (Path.home() / ".cochem" / "artifacts").resolve()
artifact_dir.mkdir(parents=True, exist_ok=True)

# Zombie process cleanup
_SUBPROCESSES: List[subprocess.Popen] = []
def cleanup_subprocesses():
    for proc in _SUBPROCESSES:
        if proc.poll() is None:
            try:
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
atexit.register(cleanup_subprocesses)

# Resolves SPYCFIT-10: Comprehensive IUPAC Atomic Isotope Mass Table (amu) for Z=1..118
MASS_DICT = {
    'H': 1.007825, 'He': 4.002602, 'Li': 7.016004, 'Be': 9.012183, 'B': 11.009305,
    'C': 12.000000, 'N': 14.003074, 'O': 15.994915, 'F': 18.998403, 'Ne': 19.992440,
    'Na': 22.989769, 'Mg': 23.985042, 'Al': 26.981538, 'Si': 27.976927, 'P': 30.973762,
    'S': 31.972071, 'Cl': 34.968853, 'Ar': 39.962383, 'K': 38.963706, 'Ca': 39.962591,
    'Sc': 44.955908, 'Ti': 47.947946, 'V': 50.943960, 'Cr': 51.940507, 'Mn': 54.938044,
    'Fe': 55.934937, 'Co': 58.933195, 'Ni': 57.935343, 'Cu': 62.929597, 'Zn': 63.929142,
    'Ga': 68.925574, 'Ge': 73.921178, 'As': 74.921594, 'Se': 79.916521, 'Br': 78.918337,
    'Kr': 83.911507, 'Rb': 84.911789, 'Sr': 87.905612, 'Y': 88.905848, 'Zr': 89.904704,
    'Nb': 92.906378, 'Mo': 97.905408, 'Tc': 97.907216, 'Ru': 101.904350, 'Rh': 102.905504,
    'Pd': 105.903486, 'Ag': 106.905097, 'Cd': 113.903358, 'In': 114.903878, 'Sn': 119.902194,
    'Sb': 120.903815, 'Te': 129.906224, 'I': 126.904473, 'Xe': 131.904153, 'Cs': 132.905452,
    'Ba': 137.905247, 'La': 138.906353, 'Ce': 139.905438, 'Pr': 140.907653, 'Nd': 141.907723,
    'Pm': 144.912749, 'Sm': 151.919732, 'Eu': 152.921230, 'Gd': 157.924104, 'Tb': 158.925347,
    'Dy': 163.929175, 'Ho': 164.930322, 'Er': 165.930293, 'Tm': 168.934212, 'Yb': 173.938862,
    'Lu': 174.940772, 'Hf': 179.946550, 'Ta': 180.947996, 'W': 183.950931, 'Re': 186.955753,
    'Os': 191.961480, 'Ir': 192.962926, 'Pt': 194.964791, 'Au': 196.966569, 'Hg': 201.970643,
    'Tl': 204.974427, 'Pb': 207.976652, 'Bi': 208.980398, 'Po': 208.982430, 'At': 209.987148,
    'Rn': 222.017578, 'Fr': 223.019736, 'Ra': 226.025410, 'Ac': 227.027751, 'Th': 232.038055,
    'Pa': 231.035884, 'U': 238.050788, 'Np': 237.048173, 'Pu': 244.064204, 'Am': 243.061381,
    'Cm': 247.070354, 'Bk': 247.070307, 'Cf': 251.079587, 'Es': 252.082980, 'Fm': 257.095105,
    'Md': 258.098431, 'No': 259.101030, 'Lr': 262.109630, 'Rf': 267.121790, 'Db': 268.125670,
    'Sg': 271.133470, 'Bh': 270.133360, 'Hs': 277.151900, 'Mt': 276.151600, 'Ds': 281.164510,
    'Rg': 280.165100, 'Cn': 285.174110, 'Nh': 284.178080, 'Fl': 289.187280, 'Mc': 288.191010,
    'Lv': 293.196000, 'Ts': 294.210460, 'Og': 294.213920
}

# Pydantic Models for Rigorous Typing
class RotationalConstants(BaseModel):
    A: float
    B: float
    C: float

class DipoleMoment(BaseModel):
    X: float
    Y: float
    Z: float
    mu_a: float
    mu_b: float
    mu_c: float

class ExtractedOutput(BaseModel):
    rotational_constants: Optional[RotationalConstants] = None
    dipole_moment: Optional[DipoleMoment] = None

class DonorAcceptorInteraction(BaseModel):
    donor: str
    acceptor: str
    E2_kcal_mol: float
    E_diff_au: float
    F_ij_au: float

class NBOData(BaseModel):
    natural_charges: Dict[str, float]
    wiberg_bond_indexes: Dict[str, float]
    donor_acceptor_interactions: List[DonorAcceptorInteraction]

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

def verify_file_hash(filepath: str | Path, expected_hash: str) -> bool:
    """Verifies SHA-256 hash of a computational artifact."""
    actual_hash = calculate_artifact_sha256(filepath)
    return actual_hash == expected_hash

def load_h5_tensors(filepath: Path, isomer_id: str) -> Dict[str, Any]:
    if not filepath.exists():
        raise FileNotFoundError(f"landscape.h5 registry missing at {filepath.absolute()}")
        
    extracted_data = {}
    with h5py.File(filepath, 'r') as f:
        if isomer_id not in f:
            raise KeyError(f"Isomer {isomer_id} not found in HDF5 registry.")
            
        group = f[isomer_id]
        
        if 'symbols' not in group:
            sub_groups = [k for k, v in group.items() if isinstance(v, h5py.Group)]
            if sub_groups:
                group = group[sub_groups[0]]
        
        raw_symbols = group['symbols'][:]
        symbols = []
        for sym in raw_symbols:
            if isinstance(sym, bytes):
                symbols.append(sym.decode('utf-8'))
            else:
                symbols.append(str(sym))
        extracted_data['symbols'] = symbols
        extracted_data['coords'] = group['geometry_xyz'][:] if 'geometry_xyz' in group else (group['geometry'][:] if 'geometry' in group else group['coords'][:])
        extracted_data['hessian'] = group['hessian_matrix'][:] if 'hessian_matrix' in group else group['hessian'][:]
        
        if 'dipole_derivatives' in group:
            extracted_data['dipole_derivs'] = group['dipole_derivatives'][:]
        else:
            extracted_data['dipole_derivs'] = None
            logger.warning(f"No dipole derivatives found for {isomer_id}. IR intensities will be zero.")
            
        if 'LAM_TRIGGER_REQUIRED' in group.attrs:
            extracted_data['lam_trigger'] = group.attrs['LAM_TRIGGER_REQUIRED']
        else:
            extracted_data['lam_trigger'] = False
            
        if 'dvr_eigenvalues' in group:
            extracted_data['dvr_eigenvalues'] = group['dvr_eigenvalues'][:]
            
        if 'tunneling_splittings' in group:
            extracted_data['tunneling_splittings'] = group['tunneling_splittings'][:]
            
        if 'kraitchman_coords' in group:
            extracted_data['kraitchman_coords'] = group['kraitchman_coords'][:]
            
    return extracted_data

def apply_mass_weighting(hessian: np.ndarray, symbols: List[str]) -> np.ndarray:
    def get_mass(sym: str) -> float:
        if sym in ('D', '2H'): return 2.014101
        clean_sym = re.sub(r'^\d+', '', sym).capitalize()
        if clean_sym not in MASS_DICT:
            raise ValueError(f"Unknown element symbol for mass weighting: {clean_sym}")
        return MASS_DICT[clean_sym]
        
    masses = np.array([get_mass(sym) for sym in symbols])
    m_3n = np.repeat(masses, 3)
    
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)
    weight_matrix = np.outer(inv_sqrt_m, inv_sqrt_m)
    
    mw_hessian = hessian * weight_matrix
    mw_hessian = 0.5 * (mw_hessian + mw_hessian.T)
    return mw_hessian

class QuantumExtractor:
    def __init__(self, output_path: Optional[str] = None) -> None:
        self.output_path = Path(output_path) if output_path else None

    def parse_mpqc_output(self, mpqc_output_text: Optional[str] = None) -> ExtractedOutput:
        if mpqc_output_text is None and self.output_path and self.output_path.exists():
            mpqc_output_text = self.output_path.read_text(encoding='utf-8', errors='ignore')
            
        if not mpqc_output_text:
            return ExtractedOutput()

        rot_constants = None
        dipole_moment = None

        rot_match = re.search(
            r'Rotational constants\s*\((MHz|cm\^-1|cm-1)\)\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)',
            mpqc_output_text, re.IGNORECASE
        )
        if rot_match:
            unit = rot_match.group(1).lower()
            a, b, c = float(rot_match.group(2)), float(rot_match.group(3)), float(rot_match.group(4))
            if 'cm' in unit:
                a, b, c = a * 29979.2458, b * 29979.2458, c * 29979.2458
            rot_constants = RotationalConstants(A=a, B=b, C=c)
        else:
            a_m = re.search(r'A\s*[:=]\s*([\d\.-]+)\s*MHz', mpqc_output_text, re.IGNORECASE)
            b_m = re.search(r'B\s*[:=]\s*([\d\.-]+)\s*MHz', mpqc_output_text, re.IGNORECASE)
            c_m = re.search(r'C\s*[:=]\s*([\d\.-]+)\s*MHz', mpqc_output_text, re.IGNORECASE)
            if a_m and b_m and c_m:
                rot_constants = RotationalConstants(
                    A=float(a_m.group(1)), B=float(b_m.group(1)), C=float(c_m.group(1))
                )

        dip_match = re.search(
            r'Total Dipole Moment\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)',
            mpqc_output_text, re.IGNORECASE
        )
        if dip_match:
            dx, dy, dz = float(dip_match.group(1)), float(dip_match.group(2)), float(dip_match.group(3))
            dipole_moment = DipoleMoment(
                X=dx, Y=dy, Z=dz, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
            )
        else:
            dip_comp = re.search(r'Dipole moment\s*\([^)]*\)\s*:\s*X\s*=\s*([\d\.-]+)\s+Y\s*=\s*([\d\.-]+)\s+Z\s*=\s*([\d\.-]+)', mpqc_output_text, re.IGNORECASE)
            if dip_comp:
                dx, dy, dz = float(dip_comp.group(1)), float(dip_comp.group(2)), float(dip_comp.group(3))
                dipole_moment = DipoleMoment(
                    X=dx, Y=dy, Z=dz, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
                )

        s2_match = re.search(r'Expectation value of <S\*\*2>\s*:\s*([\d\.]+)', mpqc_output_text)
        ideal_s2_match = re.search(r'Ideal value S\*\S+\s+for S=[\d\.]+\s*:\s*([\d\.]+)', mpqc_output_text)
        if s2_match and ideal_s2_match:
            s2_val = float(s2_match.group(1))
            ideal_s2_val = float(ideal_s2_match.group(1))
            if ideal_s2_val >= 0:
                contamination = abs(s2_val - ideal_s2_val) / ideal_s2_val if ideal_s2_val > 0 else abs(s2_val)
                if contamination > 0.10:
                    raise ValueError(f"Spin contamination exceeded 10%: <S**2> = {s2_val}, ideal = {ideal_s2_val}")

        return ExtractedOutput(rotational_constants=rot_constants, dipole_moment=dipole_moment)

    parse_orca_output = parse_mpqc_output

    def parse_gxtb_output(self, gxtb_output_text: Optional[str] = None) -> ExtractedOutput:
        if gxtb_output_text is None and self.output_path and self.output_path.exists():
            gxtb_output_text = self.output_path.read_text(encoding='utf-8', errors='ignore')

        if not gxtb_output_text:
            return ExtractedOutput()

        rot_constants = None
        dipole_moment = None

        rot_match = re.search(
            r'rotational constants\s*\((MHz|cm-1)\)\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)',
            gxtb_output_text, re.IGNORECASE
        )
        if rot_match:
            unit = rot_match.group(1).lower()
            a, b, c = float(rot_match.group(2)), float(rot_match.group(3)), float(rot_match.group(4))
            if 'cm' in unit:
                a, b, c = a * 29979.2458, b * 29979.2458, c * 29979.2458
            rot_constants = RotationalConstants(A=a, B=b, C=c)

        dip_match = re.search(
            r'molecular dipole\s*:\s*x\s*=\s*([\d\.-]+)\s+y\s*=\s*([\d\.-]+)\s+z\s*=\s*([\d\.-]+)',
            gxtb_output_text, re.IGNORECASE
        )
        if dip_match:
            dx, dy, dz = float(dip_match.group(1)), float(dip_match.group(2)), float(dip_match.group(3))
            dipole_moment = DipoleMoment(
                X=dx, Y=dy, Z=dz, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
            )

        return ExtractedOutput(rotational_constants=rot_constants, dipole_moment=dipole_moment)

    def parse_output(self, text_or_path: Optional[str] = None) -> ExtractedOutput:
        text = text_or_path
        if text is None and self.output_path and self.output_path.exists():
            text = self.output_path.read_text(encoding='utf-8', errors='ignore')

        if not text:
            return ExtractedOutput()

        if 'MPQC' in text or 'Program Version' in text or 'Total Dipole Moment' in text:
            return self.parse_mpqc_output(text)
        elif 'xtb' in text.lower() or 'gfn' in text.lower() or 'molecular dipole' in text.lower():
            return self.parse_gxtb_output(text)
        else:
            res = self.parse_mpqc_output(text)
            if res.rotational_constants is None:
                res = self.parse_gxtb_output(text)
            return res

def parse_mpqc_freq_output(mpqc_output_text: str) -> ExtractedOutput:
    extractor = QuantumExtractor()
    return extractor.parse_mpqc_output(mpqc_output_text)

parse_orca_freq_output = parse_mpqc_freq_output

def vpt2_resonance_trap(frequencies: np.ndarray, cubic_k: Optional[np.ndarray] = None, threshold: float = 10.0) -> List[Tuple[str, int, int, int, float]]:
    resonances = []
    n = len(frequencies)
    for i in range(n):
        for j in range(i+1, n):
            freq_diff = abs(2 * frequencies[i] - frequencies[j])
            if freq_diff < threshold:
                k_ijk = cubic_k[i, i, j] if (cubic_k is not None and cubic_k.shape == (n, n, n)) else 0.0
                phi_ijk = abs(k_ijk) / (2.0 * np.sqrt(2.0))
                if phi_ijk > 0.01:
                    resonances.append(('Fermi', i, j, 0, phi_ijk))
            for k in range(j+1, n):
                freq_diff_dd = abs(frequencies[i] + frequencies[j] - frequencies[k])
                if freq_diff_dd < threshold:
                    k_ijk_dd = cubic_k[i, j, k] if (cubic_k is not None and cubic_k.shape == (n, n, n)) else 0.0
                    if abs(k_ijk_dd) > 0.01:
                        resonances.append(('Fermi Type 2', i, j, k, abs(k_ijk_dd)))
    return resonances

def perform_normal_mode_analysis(mw_hessian: np.ndarray, symbols: List[str], dipole_derivs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    HARTREE_BOHR2_AMU_TO_CM1 = 5140.48714
    eigvals, eigvecs = np.linalg.eigh(mw_hessian)
    
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * HARTREE_BOHR2_AMU_TO_CM1
    
    def get_mass(sym: str) -> float:
        if sym in ('D', '2H'): return 2.014101
        clean_sym = re.sub(r'^\d+', '', sym).capitalize()
        if clean_sym not in MASS_DICT:
            raise ValueError(f"Unknown element symbol: {clean_sym}")
        return MASS_DICT[clean_sym]
    
    masses = np.array([get_mass(sym) for sym in symbols])
    m_3n = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)

    modes_cartesian = eigvecs * inv_sqrt_m[:, np.newaxis]
    
    reduced_masses = []
    ir_intensities = []
    
    for k in range(len(freqs)):
        norm_k = np.sum(modes_cartesian[:, k]**2)
        red_m = 1.0 / norm_k if norm_k > 1e-12 else 1.0
        reduced_masses.append(float(red_m))

        if dipole_derivs is not None and dipole_derivs.shape[0] == len(m_3n):
            d_mu_dQ = np.dot(dipole_derivs.T, modes_cartesian[:, k])
            intensity = float(np.sum(d_mu_dQ**2) * 42.2561)
            ir_intensities.append(intensity)
        else:
            ir_intensities.append(0.0)

    return {
        'frequencies_cm1': freqs.tolist(),
        'eigenvalues': eigvals.tolist(),
        'reduced_masses_amu': reduced_masses,
        'ir_intensities_km_mol': ir_intensities,
        'cartesian_displacement_modes': modes_cartesian.tolist()
    }

def parse_nbo_analysis(mpqc_output_text: str) -> NBOData:
    natural_charges = {}
    wiberg_bond_indexes = {}
    donor_acceptor_interactions = []
    
    if not mpqc_output_text:
        return NBOData(natural_charges=natural_charges, wiberg_bond_indexes=wiberg_bond_indexes, donor_acceptor_interactions=donor_acceptor_interactions)

    for line in mpqc_output_text.splitlines():
        m = re.match(r'^\s*([A-Za-z]{1,2})\s+(\d+)\s+([-\d\.]+)', line)
        if m and m.group(1) in MASS_DICT:
            atom_label = f"{m.group(1)}{m.group(2)}"
            natural_charges[atom_label] = float(m.group(3))
            
    wbi_block = re.search(r'Wiberg bond index matrix.*?\n-+\n(.*?)\n-+\n\s*Wiberg', mpqc_output_text, re.DOTALL | re.IGNORECASE)
    if wbi_block:
        lines = wbi_block.group(1).splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                atom_i = int(parts[0])
                for idx_j, val_str in enumerate(parts[1:], start=1):
                    val = float(val_str)
                    if val > 0.05:
                        wiberg_bond_indexes[f"{atom_i}-{idx_j}"] = val
                        
    e2_matches = re.findall(r'(\d+\.\s+[^\n]+?)\s+-->\s+(\d+\.\s+[^\n]+?)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', mpqc_output_text)
    for donor, acceptor, e2_val, ediff, f_ij in e2_matches:
        donor_acceptor_interactions.append(DonorAcceptorInteraction(
            donor=donor.strip(),
            acceptor=acceptor.strip(),
            E2_kcal_mol=float(e2_val),
            E_diff_au=float(ediff),
            F_ij_au=float(f_ij)
        ))
        
    return NBOData(
        natural_charges=natural_charges,
        wiberg_bond_indexes=wiberg_bond_indexes,
        donor_acceptor_interactions=donor_acceptor_interactions
    )

def save_out_of_core_parquet(data: Dict[str, Any], filename: str = "spycfit_tensor_buffer.parquet") -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    
    df = pd.DataFrame()
    metadata = {}
    
    if 'mw_hessian' in data:
        mw_h = data['mw_hessian']
        df['mw_hessian'] = mw_h.flatten()
        metadata['mw_hessian_shape'] = json.dumps(list(mw_h.shape))
        
    if 'dipole_derivs' in data and data['dipole_derivs'] is not None:
        dip_d = data['dipole_derivs']
        flat_dipole = dip_d.flatten()
        metadata['dipole_derivs_shape'] = json.dumps(list(dip_d.shape))
        
        if len(flat_dipole) < len(df):
            flat_dipole = np.pad(flat_dipole, (0, len(df) - len(flat_dipole)))
        elif len(flat_dipole) > len(df):
            df = df.reindex(range(len(flat_dipole)))
            if 'mw_hessian' in data:
                df['mw_hessian'] = np.pad(data['mw_hessian'].flatten(), (0, len(flat_dipole) - data['mw_hessian'].flatten().size))
        df['dipole_derivs'] = flat_dipole
        
    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, **{k.encode('utf-8'): v.encode('utf-8') for k, v in metadata.items()}}
    table = table.replace_schema_metadata(new_meta)
    
    pq.write_table(table, filename)
    logger.info(f"{Colors.OKGREEN}✅ Out-of-core PyArrow serialization complete: {filename} [M]{Colors.ENDC}")

def read_out_of_core_parquet(filename: str = "spycfit_tensor_buffer.parquet") -> Dict[str, Any]:
    import pyarrow.parquet as pq
    table = pq.read_table(filename)
    metadata = table.schema.metadata or {}
    df = table.to_pandas()
    
    result = {}
    if b'mw_hessian_shape' in metadata:
        shape = json.loads(metadata[b'mw_hessian_shape'].decode('utf-8'))
        size = np.prod(shape)
        result['mw_hessian'] = df['mw_hessian'].values[:size].reshape(shape)

    if b'dipole_derivs_shape' in metadata and 'dipole_derivs' in df:
        shape = json.loads(metadata[b'dipole_derivs_shape'].decode('utf-8'))
        size = np.prod(shape)
        result['dipole_derivs'] = df['dipole_derivs'].values[:size].reshape(shape)

    return result

def extract_lam_data(data: Dict[str, Any]) -> Dict[str, Any]:
    extracted_lam_data = {}
    if data.get('lam_trigger', False):
        logger.info(f"{Colors.OKCYAN}🔍 LAM Protocol Detected - Bypassing Standard VPT2 Diagonalization{Colors.ENDC}")
        if 'dvr_eigenvalues' in data:
            extracted_lam_data['eigenvalues'] = data['dvr_eigenvalues']
        if 'tunneling_splittings' in data:
            extracted_lam_data['tunneling_splittings'] = data['tunneling_splittings']
        if 'kraitchman_coords' in data:
            extracted_lam_data['kraitchman_coords'] = data['kraitchman_coords']
    return extracted_lam_data

def generate_pickett_files(fitted_results: Dict[str, Any], output_dir: str | Path = "pickett_output") -> None:
    logger.info(f"{Colors.OKCYAN}💾 Generating Pickett-formatted files...{Colors.ENDC}")
    
    out_dir = artifact_dir / output_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if not isinstance(fitted_results, dict):
        raise TypeError("fitted_results must be a dictionary.")

    mol_name = fitted_results.get("molecule", fitted_results.get("name", "Fitted Isoprotologue"))
    temp = float(fitted_results.get("temperature", 298.15))
    
    rot = fitted_results.get("rotational_constants")
    if rot is None:
        if "final_params" in fitted_results and len(fitted_results["final_params"]) >= 3:
            rot = fitted_results["final_params"][:3]
        elif "A" in fitted_results and "B" in fitted_results and "C" in fitted_results:
            rot = {"A": fitted_results["A"], "B": fitted_results["B"], "C": fitted_results["C"]}
        else:
            raise ValueError("Missing rotational constants (A, B, C).")

    if isinstance(rot, (list, tuple, np.ndarray)) and len(rot) >= 3:
        a_val, b_val, c_val = float(rot[0]), float(rot[1]), float(rot[2])
    elif isinstance(rot, dict):
        if "A" not in rot or "B" not in rot or "C" not in rot:
             raise ValueError("Missing rotational constants (A, B, C).")
        a_val, b_val, c_val = float(rot["A"]), float(rot["B"]), float(rot["C"])
    else:
        raise ValueError("Invalid format for rotational constants.")

    unc = fitted_results.get("uncertainties", fitted_results.get("errors"))
    if unc is None:
        if "A_unc" in fitted_results and "B_unc" in fitted_results and "C_unc" in fitted_results:
            unc = {"A": fitted_results["A_unc"], "B": fitted_results["B_unc"], "C": fitted_results["C_unc"]}
        else:
            raise ValueError("Missing uncertainties for rotational constants.")

    if isinstance(unc, (list, tuple, np.ndarray)) and len(unc) >= 3:
        a_unc, b_unc, c_unc = float(unc[0]), float(unc[1]), float(unc[2])
    elif isinstance(unc, dict):
        if "A" not in unc or "B" not in unc or "C" not in unc:
            raise ValueError("Missing uncertainties (A, B, C).")
        a_unc, b_unc, c_unc = float(unc["A"]), float(unc["B"]), float(unc["C"])
    else:
        raise ValueError("Invalid format for uncertainties.")

    dist = fitted_results.get("distortion_constants")
    if dist is None:
        if "final_params" in fitted_results and len(fitted_results["final_params"]) >= 5:
            dist = {"D_J": fitted_results["final_params"][3], "D_K": fitted_results["final_params"][4], "D_J_unc": 0.0, "D_K_unc": 0.0}
        elif "D_J" in fitted_results and "D_K" in fitted_results:
            dist = {
                "D_J": fitted_results["D_J"], "D_K": fitted_results["D_K"],
                "D_J_unc": fitted_results.get("D_J_unc", 0.0), "D_K_unc": fitted_results.get("D_K_unc", 0.0)
            }
        else:
            raise ValueError("Missing distortion constants (D_J, D_K).")

    if isinstance(dist, dict):
        if "D_J" not in dist or "D_K" not in dist:
             raise ValueError("Missing distortion constants (D_J, D_K).")
        dj_val, dk_val = float(dist["D_J"]), float(dist["D_K"])
        dj_unc, dk_unc = float(dist.get("D_J_unc", 0.0)), float(dist.get("D_K_unc", 0.0))
    else:
        raise ValueError("Invalid format for distortion constants.")

    cat_data = [
        f"Molecule: {mol_name}",
        f"Temperature: {temp:.1f} K",
        f"Rotational constants: A={a_val:.1f}, B={b_val:.1f}, C={c_val:.1f} MHz",
        f"Distortion constants: D_J={dj_val:.3f}, D_K={dk_val:.3f} MHz"
    ]
    cat_path = out_dir / "output.cat"
    with open(cat_path, 'w') as f:
        for line in cat_data:
            f.write(f"{line}\n")
            
    fit_data = [
        "Fit parameters (MHz)",
        f"A = {a_val:.3f} ± {a_unc:.3f}",
        f"B = {b_val:.3f} ± {b_unc:.3f}",
        f"C = {c_val:.3f} ± {c_unc:.3f}",
        f"D_J = {dj_val:.5f} ± {dj_unc:.5f}",
        f"D_K = {dk_val:.5f} ± {dk_unc:.5f}"
    ]
    fit_path = out_dir / "output.fit"
    with open(fit_path, 'w') as f:
        for line in fit_data:
            f.write(f"{line}\n")

    par_path = out_dir / "output.par"
    with open(par_path, 'w') as f:
        f.write(f"{mol_name} parameters\n")
        f.write("    5  1   0   0   0.0000E+00   0.0000E+00   1.0000E+00\n")
        f.write(f" 10000  {a_val:20.10f} {a_unc:15.10f}\n")
        f.write(f" 20000  {b_val:20.10f} {b_unc:15.10f}\n")
        f.write(f" 30000  {c_val:20.10f} {c_unc:15.10f}\n")
        f.write(f"   200  {dj_val:20.10f} {dj_unc:15.10f}\n")
        f.write(f"  2000  {dk_val:20.10f} {dk_unc:15.10f}\n")
    
    int_path = out_dir / "output.int"
    with open(int_path, 'w') as f:
        f.write(f"{mol_name} spcat config\n")
        f.write("0  99  2000 0  1.0  0.0  0.0  0.0  0.0  0.0\n")

    proc = subprocess.Popen(["spcat", str(int_path.name), str(par_path.name)], cwd=str(out_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _SUBPROCESSES.append(proc)
    try:
        stdout, stderr = proc.communicate(timeout=60)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout, stderr=stderr)
            
        cat_out_path = out_dir / "output.cat" 
        lin_path = out_dir / "output.lin"
        if cat_out_path.exists():
            with open(cat_out_path, "r") as f_cat, open(lin_path, "w") as f_lin:
                f_lin.write("Pickett .lin file\n")
                f_lin.write(f"Generated by CoChem-SpycFit SPCAT wrapper for {mol_name}\n")
                for line in f_cat:
                    if len(line) > 50:
                        try:
                            freq = float(line[0:13])
                            err = float(line[13:21])
                            f_lin.write(f"{line[55:67]} {line[67:79]} {freq:13.4f} {err:8.4f} 1.0000\n")
                        except ValueError:
                            logger.warning(f"Skipping malformed Pickett line: {line.strip()}")
        logger.info(f"{Colors.OKGREEN}✅ SPCAT successfully generated transitions. [M]{Colors.ENDC}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError("spcat execution timed out after 60 seconds.")
    except Exception as e:
        logger.error(f"Failed to execute spcat or process outputs. Error: {e}")
        raise

    logger.info(f"{Colors.OKGREEN}✅ Pickett files generated for {mol_name}:")
    logger.info(f"   output.lin, output.cat, output.fit, output.par, output.int{Colors.ENDC}")
    
    trigger_scribe_daemon(output_dir)

def trigger_scribe_daemon(output_dir: str | Path) -> None:
    logger.info(f"{Colors.OKCYAN}📡 Triggering CoChem-SCRIBE daemon via ZeroMQ...{Colors.ENDC}")
    
    payload = {
        "source": "CoChem-SpycFit",
        "output_dir": str(output_dir),
        "status": "READY_FOR_ZSTD_PACKAGING"
    }

    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 500)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.connect("tcp://127.0.0.1:5555")
    
    socket.send_json(payload)
    try:
        resp = socket.recv_json()
        logger.info(f"{Colors.OKGREEN}✅ SCRIBE daemon ZMQ response: {resp}{Colors.ENDC}")
    except zmq.error.Again:
        raise ConnectionError("ZMQ connection to SCRIBE daemon timed out. Ensure the daemon is running.")

def update_swarm_state(status: str, artifacts: List[str], error_code: Optional[str] = None) -> None:
    from datetime import datetime, timezone
    state_file = Path("D:/__CoChem/GitHub-Repo/CoChem-BASE/swarm_state.json")
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except json.JSONDecodeError:
            pass
            
    state["cochem-audit"] = {
        "status": status,
        "artifacts": artifacts,
        "error_code": error_code,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=4)
    except OSError as e:
        logger.error(f"Failed to update swarm_state.json: {e}")

def main() -> None:
    logger.info(f"\n{Colors.OKCYAN}--- SpycFit: Quantum Tensor Extraction ---{Colors.ENDC}")
    
    h5_path = artifact_dir / "landscape.h5"
    if not h5_path.exists():
        logger.info(f"{Colors.WARNING}⚠️ landscape.h5 not found at {h5_path.absolute()}. Please supply valid HDF5 landscape file.{Colors.ENDC}")
        sys.exit(1)

    if len(sys.argv) > 1:
        target_isomer = sys.argv[1]
    else:
        target_isomer = os.environ.get("COCHEM_TARGET_ISOMER")
        if not target_isomer:
            logger.error(f"{Colors.FAIL}❌ [MISSING DATA] Missing target isomer. Pass via sys.argv[1] or COCHEM_TARGET_ISOMER env var.{Colors.ENDC}")
            sys.exit(1)
    
    try:
        logger.info(f"📥 Extracting tensors for {target_isomer}...")
        data = load_h5_tensors(h5_path, target_isomer)
        
        lam_data = extract_lam_data(data)
        if lam_data:
            np.savez(artifact_dir / "spycfit_lam_buffer.npz", 
                     eigenvalues=lam_data.get('eigenvalues'),
                     tunneling_splittings=lam_data.get('tunneling_splittings'),
                     kraitchman_coords=lam_data.get('kraitchman_coords'))
            logger.info(f"{Colors.OKGREEN}✅ LAM data extracted and saved to spycfit_lam_buffer.npz [M]{Colors.ENDC}")
        
        logger.info(f"⚖️  Applying isotopic mass-weighting and Hermitian locks...")
        mw_hessian = apply_mass_weighting(data['hessian'], data['symbols'])
        
        np.savez(artifact_dir / "spycfit_tensor_buffer.npz", 
                 mw_hessian=mw_hessian, 
                 dipole_derivs=data['dipole_derivs'],
                 symbols=data['symbols'],
                 lam_trigger=data['lam_trigger'])
                 
        logger.info(f"{Colors.OKGREEN}✅ Extraction Complete. Tensors staged to spycfit_tensor_buffer.npz [M]{Colors.ENDC}")
        logger.info(f"Successfully processed and mass-weighted {target_isomer}.")
        update_swarm_state("SUCCESS", [str(artifact_dir / "spycfit_tensor_buffer.npz")], None)
        
    except Exception as e:
        logger.error(f"{Colors.FAIL}❌ Extraction Failed: {str(e)}{Colors.ENDC}")
        update_swarm_state("FAILURE", [], str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()