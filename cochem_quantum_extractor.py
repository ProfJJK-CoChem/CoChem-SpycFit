#!/usr/bin/env python3
"""
CoChem-SpycFit: Quantum Tensor Extractor
Extracts Raw Hessians and Dipole Derivatives from the central landscape.h5.
Applies Isotopic Mass Weighting and enforces strict Hermitian symmetry.
"""

import os
import sys
import logging
import json
from pathlib import Path

try:
    import numpy as np
    import h5py
except ImportError:
    print("❌ FATAL: 'numpy' and 'h5py' are required for quantum extraction.")
    sys.exit(1)

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

logging.basicConfig(filename='cochem_spycfit_extractor.log', level=logging.INFO)

# Standard isotope masses (amu) - Extensible via dictionary updates
MASS_DICT = {
    'H': 1.007825, 'C': 12.000000, 'N': 14.003074, 
    'O': 15.994915, 'F': 18.998403, 'S': 31.972071, 'Cl': 34.968853
}

def load_h5_tensors(filepath: Path, isomer_id: str) -> dict:
    """Extracts raw geometry and tensor arrays from the CoChem HDF5 landscape."""
    if not filepath.exists():
        raise FileNotFoundError(f"landscape.h5 registry missing at {filepath.absolute()}")
        
    extracted_data = {}
    with h5py.File(filepath, 'r') as f:
        if isomer_id not in f:
            raise KeyError(f"Isomer {isomer_id} not found in HDF5 registry.")
            
        group = f[isomer_id]
        
        # Extract datasets, converting byte strings to regular strings for symbols
        extracted_data['symbols'] = [s.decode('utf-8') for s in group['symbols'][:]]
        extracted_data['coords'] = group['coords'][:]
        extracted_data['hessian'] = group['hessian'][:]
        
        if 'dipole_derivatives' in group:
            extracted_data['dipole_derivs'] = group['dipole_derivatives'][:]
        else:
            extracted_data['dipole_derivs'] = None
            logging.warning(f"No dipole derivatives found for {isomer_id}. IR intensities will be zero.")
            
    return extracted_data

def apply_mass_weighting(hessian: np.ndarray, symbols: list) -> np.ndarray:
    """
    Applies M^(-1/2) * H * M^(-1/2)
    Constructs the mass-weighted Hessian required for vibrational frequency analysis.
    Enforces H = (H + H^T)/2 to purge numerical artifacts.
    """
    n_atoms = len(symbols)
    masses = np.array([MASS_DICT.get(sym, 12.0) for sym in symbols])
    
    # Repeat masses for X, Y, Z degrees of freedom (3N array)
    m_3n = np.repeat(masses, 3)
    
    # Create the 1 / sqrt(m_i * m_j) weighting matrix
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)
    weight_matrix = np.outer(inv_sqrt_m, inv_sqrt_m)
    
    # Apply weighting
    mw_hessian = hessian * weight_matrix
    
    # Strict Hermitian Enforcement (Average out numerical noise from DFT grid grids)
    mw_hessian = 0.5 * (mw_hessian + mw_hessian.T)
    
    return mw_hessian

def main():
    print(f"\n{Colors.OKCYAN}--- SpycFit: Quantum Tensor Extraction ---{Colors.ENDC}")
    
    h5_path = Path("landscape.h5")
    
    # Mocking a landscape.h5 structure for diagnostic testing if missing
    if not h5_path.exists():
        print(f"{Colors.WARNING}⚠️ landscape.h5 not found. Generating a mock dataset for validation...{Colors.ENDC}")
        with h5py.File(h5_path, 'w') as f:
            grp = f.create_group("isomer_001")
            grp.create_dataset("symbols", data=np.array([b'O', b'H', b'H']))
            grp.create_dataset("coords", data=np.zeros((3, 3)))
            # Mock 9x9 symmetric Hessian
            mock_h = np.random.rand(9, 9)
            mock_h = 0.5 * (mock_h + mock_h.T)
            grp.create_dataset("hessian", data=mock_h)

    target_isomer = "isomer_001"
    
    try:
        print(f"📥 Extracting tensors for {target_isomer}...")
        data = load_h5_tensors(h5_path, target_isomer)
        
        print(f"⚖️  Applying isotopic mass-weighting and Hermitian locks...")
        mw_hessian = apply_mass_weighting(data['hessian'], data['symbols'])
        
        # Save extracted tensor buffer for the JAX Convolver (Stage 4.1)
        np.savez("spycfit_tensor_buffer.npz", 
                 mw_hessian=mw_hessian, 
                 dipole_derivs=data['dipole_derivs'],
                 symbols=data['symbols'])
                 
        print(f"{Colors.OKGREEN}✅ Extraction Complete. Tensors staged to spycfit_tensor_buffer.npz{Colors.ENDC}")
        logging.info(f"Successfully processed and mass-weighted {target_isomer}.")
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Extraction Failed: {str(e)}{Colors.ENDC}")
        logging.error(f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    main()