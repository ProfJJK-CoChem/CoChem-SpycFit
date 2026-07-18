#!/usr/bin/env python3
"""
CoChem-SpycFit: Initialization & JAX Gatekeeper
Locks the XLA compiler to float64 precision to prevent catastrophic 
numerical degradation during massively parallel spectral diagonalizations.
"""

import os
import sys
import json
import logging
from pathlib import Path

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

logging.basicConfig(
    filename='cochem_spycfit_init.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def enforce_jax_precision() -> dict:
    """Hooks into the JAX backend to override float32 ML defaults."""
    print(f"{Colors.OKCYAN}▶ Initializing SpycFit XLA Compiler Environment...{Colors.ENDC}")
    
    try:
        import jax
        from jax import config
        
        # CRITICAL: Force 64-bit precision for Quantum Mechanics
        config.update("jax_enable_x64", True)
        
        # Profile Backend Accelerators
        devices = jax.devices()
        backend_type = devices[0].platform.upper()
        device_count = len(devices)
        
        # Verify 64-bit actually engaged
        test_array = jax.numpy.array([1.0])
        if test_array.dtype != jax.numpy.float64:
            raise RuntimeError("JAX failed to lock float64 precision.")
            
        print(f"{Colors.OKGREEN}✅ JAX Float64 Precision Locked.{Colors.ENDC}")
        print(f"🖥️  XLA Backend: {backend_type} (Devices: {device_count})")
        
        logging.info(f"JAX Initialized. Backend: {backend_type}, Devices: {device_count}, Float64: Active.")
        
        return {
            "status": "ready",
            "backend": backend_type,
            "devices": device_count,
            "precision": "float64"
        }
        
    except ImportError:
        error_msg = "FATAL: 'jax' or 'jaxlib' is not installed. SpycFit requires JAX for GPU convolutions."
        print(f"{Colors.FAIL}❌ {error_msg}{Colors.ENDC}")
        logging.error(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"FATAL: JAX Environment failure: {str(e)}"
        print(f"{Colors.FAIL}❌ {error_msg}{Colors.ENDC}")
        logging.error(error_msg)
        sys.exit(1)

def main():
    print(f"\n{Colors.BOLD}--- CoChem-SpycFit: Environment Pre-Flight ---{Colors.ENDC}")
    
    env_state = enforce_jax_precision()
    
    # Export state to the local registry for downstream convolvers
    state_path = Path("cochem_spycfit_state.json")
    with open(state_path, "w") as f:
        json.dump(env_state, f, indent=4)
        
    print(f"{Colors.OKGREEN}✅ SpycFit Initialization Complete. Ready for tensor extraction.{Colors.ENDC}\n")

if __name__ == "__main__":
    main()