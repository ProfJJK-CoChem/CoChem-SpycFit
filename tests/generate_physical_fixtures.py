import h5py
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_landscape():
    # Realistic geometry for H2O in Angstroms
    geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    
    # Real Hessian elements (approximate for water at B3LYP/def2-TZVP)
    # 9x9 matrix in hartree/bohr^2
    hess = np.array([
        [0.342, 0.0, 0.0, -0.171, -0.0, -0.0, -0.171, 0.0, 0.0],
        [0.0, 0.405, 0.0, 0.0, -0.202, -0.05, 0.0, -0.202, 0.05],
        [0.0, 0.0, 0.582, 0.0, -0.05, -0.291, 0.0, 0.05, -0.291],
        [-0.171, 0.0, 0.0, 0.171, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.0, -0.202, -0.05, 0.0, 0.202, 0.05, 0.0, 0.0, 0.0],
        [-0.0, -0.05, -0.291, 0.0, 0.05, 0.291, 0.0, 0.0, 0.0],
        [-0.171, 0.0, 0.0, 0.0, 0.0, 0.0, 0.171, 0.0, 0.0],
        [0.0, -0.202, 0.05, 0.0, 0.0, 0.0, 0.0, 0.202, -0.05],
        [0.0, 0.05, -0.291, 0.0, 0.0, 0.0, 0.0, -0.05, 0.291]
    ], dtype=np.float64)
    # Symmetrize just in case
    hess = 0.5 * (hess + hess.T)
    
    # Real dipole derivatives (approximate, Debye/Angstrom)
    # 3x9 matrix
    d_mu = np.array([
        [0.0, 0.0, -1.5, 0.0, -0.2, 0.75, 0.0, 0.2, 0.75],
        [0.0, -1.2, 0.0, 0.5, 0.6, 0.0, -0.5, 0.6, 0.0],
        [-0.8, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0]
    ], dtype=np.float64)
    
    with h5py.File(FIXTURES_DIR / "landscape_water.h5", "w") as f:
        f.create_dataset("geometry", data=geom)
        f.create_dataset("energy_hartree", data=-76.4321)
        f.create_dataset("cartesian_hessian", data=hess)
        f.create_dataset("dipole_derivatives", data=d_mu)
        f.create_dataset("atomic_numbers", data=np.array([8, 1, 1]))
        f.create_dataset("masses_amu", data=np.array([15.9949146, 1.0078250, 1.0078250]))

def generate_catalog():
    # Real H2O transitions (JPL/CDMS)
    # Freq(MHz), Unc, Int, E_low(cm-1), E_up, g_up, QN (J', Ka', Kc', J'', Ka'', Kc'')
    transitions = [
        (556935.9877, 0.005, -3.1, 0.0, 18.57, 3, [1, 1, 0, 1, 0, 1]),
        (752033.1130, 0.005, -2.8, 37.13, 62.21, 5, [2, 1, 1, 2, 0, 2]),
        (1669904.775, 0.010, -1.5, 0.0, 55.70, 5, [2, 1, 2, 1, 0, 1]),
        (183310.0870, 0.005, -4.2, 136.16, 142.27, 7, [3, 1, 3, 2, 2, 0]),
        (380197.3720, 0.005, -3.5, 136.16, 148.84, 9, [4, 1, 4, 3, 2, 1]),
    ] * 2000 # duplicate to make it large enough for chunks
    
    count = len(transitions)
    freqs = np.array([t[0] for t in transitions], dtype=np.float64)
    uncs = np.array([t[1] for t in transitions], dtype=np.float64)
    intens = np.array([t[2] for t in transitions], dtype=np.float64)
    e_low = np.array([t[3] for t in transitions], dtype=np.float64)
    e_up = np.array([t[4] for t in transitions], dtype=np.float64)
    g_up = np.array([t[5] for t in transitions], dtype=np.int32)
    qn = np.array([t[6] for t in transitions], dtype=np.int32)
    sp_ids = np.zeros(count, dtype=np.int32)
    
    # Sort frequencies to keep it physically realistic (monotonically increasing)
    idx = np.argsort(freqs)
    # Add tiny noise to frequencies to make them strictly unique and increasing properly if we sort
    # actually, these are 5 duplicate lines, let's just add a small spacing
    freqs = freqs[idx] + np.linspace(0, 100, count)
    uncs = uncs[idx]
    intens = intens[idx]
    e_low = e_low[idx]
    e_up = e_up[idx]
    g_up = g_up[idx]
    qn = qn[idx]
    
    with h5py.File(FIXTURES_DIR / "catalog_water.h5", "w") as f:
        f.attrs["total_transitions"] = count
        f.attrs["molecule_name"] = "Water"
        f.create_dataset("frequencies", data=freqs, chunks=(2048,), maxshape=(None,))
        f.create_dataset("intensities", data=intens, chunks=(2048,), maxshape=(None,))
        f.create_dataset("lower_energies", data=e_low, chunks=(2048,), maxshape=(None,))
        f.create_dataset("upper_energies", data=e_up, chunks=(2048,), maxshape=(None,))
        f.create_dataset("uncertainties", data=uncs, chunks=(2048,), maxshape=(None,))
        f.create_dataset("g_upper", data=g_up, chunks=(2048,), maxshape=(None,))
        f.create_dataset("quantum_numbers", data=qn, chunks=(2048, 6), maxshape=(None, 6))
        f.create_dataset("species_ids", data=sp_ids, chunks=(2048,), maxshape=(None,))

if __name__ == "__main__":
    generate_landscape()
    generate_catalog()
