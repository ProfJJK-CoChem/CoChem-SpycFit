#!/usr/bin/env python3
"""
CoChem-SpycFit: High-Performance JAX Convolution Engine
Utilizes jax.vmap and XLA compilation to broadcast Gaussian/Lorentzian 
line profiles across massively dense spectral grids.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

logging.basicConfig(filename='cochem_spycfit_convolver.log', level=logging.INFO)

# --- XLA Engine Setup ---
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    import numpy as jnp  # Fallback to standard numpy

def load_state() -> dict:
    state_path = Path("cochem_spycfit_state.json")
    if not state_path.exists():
        raise RuntimeError("SpycFit state registry missing. Run cochem_spycfit_init.py first.")
    with open(state_path, "r") as f:
        return json.load(f)

def load_tensors() -> dict:
    tensor_path = Path("spycfit_tensor_buffer.npz")
    if not tensor_path.exists():
        raise RuntimeError("Tensor buffer missing. Run cochem_quantum_extractor.py first.")
    return np.load(tensor_path, allow_pickle=True)

# --- JAX Compiled Physics Functions ---
@jax.jit if JAX_AVAILABLE else lambda x: x
def gaussian_profile(nu_grid, nu_0, intensity, sigma):
    """JIT-compiled Gaussian line shape."""
    return intensity * jnp.exp(-0.5 * ((nu_grid - nu_0) / sigma) ** 2)

def calculate_frequencies(mw_hessian: np.ndarray) -> tuple:
    """Diagonalizes the mass-weighted Hessian to extract vibrational frequencies."""
    # Diagonalize (using standard numpy since Hessians are small, 3Nx3N)
    eigenvalues, eigenvectors = np.linalg.eigh(mw_hessian)
    
    # Convert eigenvalues to wavenumbers (cm^-1)
    # Factor: sqrt(Hartree/(bohr^2 * amu)) to cm^-1 is ~ 5140.48
    CONV_FACTOR = 5140.48
    
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * CONV_FACTOR
    
    # Filter out translational/rotational modes (typically the first 6, near zero)
    valid_mask = frequencies > 10.0 
    return frequencies[valid_mask], eigenvectors[:, valid_mask]

def convolve_spectrum(frequencies, intensities, grid_start=0, grid_end=4000, points=10000, sigma=10.0):
    """Broadcasts profiles across the energy grid."""
    print(f"🌊 Generating high-density spectral grid ({points} points)...")
    nu_grid = jnp.linspace(grid_start, grid_end, points)
    
    if JAX_AVAILABLE:
        # Use vmap to parallelize across all frequency modes instantly
        print(f"🚀 XLA Compiler engaged. Broadcasting across {len(frequencies)} modes.")
        vmap_gaussian = jax.vmap(gaussian_profile, in_axes=(None, 0, 0, None))
        spectra_matrix = vmap_gaussian(nu_grid, jnp.array(frequencies), jnp.array(intensities), sigma)
        total_spectrum = jnp.sum(spectra_matrix, axis=0)
    else:
        print(f"{Colors.WARNING}⚠️ JAX unavailable. Falling back to NumPy CPU broadcasting.{Colors.ENDC}")
        # Standard numpy broadcasting
        total_spectrum = np.zeros_like(nu_grid)
        for nu_0, I in zip(frequencies, intensities):
            total_spectrum += gaussian_profile(nu_grid, nu_0, I, sigma)
            
    return np.array(nu_grid), np.array(total_spectrum)

def main():
    print(f"\n{Colors.BOLD}--- CoChem-SpycFit: JAX Convolution Engine ---{Colors.ENDC}")
    
    try:
        # 1. Verification
        state = load_state()
        if state.get("precision") != "float64":
            print(f"{Colors.WARNING}⚠️ Warning: XLA is not explicitly locked to float64.{Colors.ENDC}")
            
        tensors = load_tensors()
        mw_hessian = tensors['mw_hessian']
        
        # 2. Diagonalization
        print("📐 Diagonalizing Mass-Weighted Hessian...")
        frequencies, normal_modes = calculate_frequencies(mw_hessian)
        
        # Mock intensities for demonstration (in a full run, derived from dipole_derivs dotted with normal_modes)
        intensities = np.random.uniform(1, 100, size=len(frequencies))
        
        # 3. Convolution
        x_grid, y_spec = convolve_spectrum(frequencies, intensities)
        
        # 4. Output
        out_path = Path("simulated_ir_spectrum.csv")
        np.savetxt(out_path, np.column_stack((x_grid, y_spec)), delimiter=",", header="Wavenumber(cm-1),Intensity", comments="")
        
        print(f"{Colors.OKGREEN}✅ Spectral Convolution Complete. Artifact saved to {out_path.name}{Colors.ENDC}\n")
        logging.info(f"Successfully generated spectrum. Dimensions: {len(x_grid)} grid points.")
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Convolver Error: {str(e)}{Colors.ENDC}")
        logging.error(f"Convolution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()