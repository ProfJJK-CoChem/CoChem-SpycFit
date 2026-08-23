import os
import sys
import h5py
import shutil
import numpy as np
import pytest
import subprocess
import logging
import psutil
import atexit
from pathlib import Path

# 1. Registry Consistency & Air-Gap Enforcement
# Replaced __file__ static lookup with environment variables / Path.home() dynamic lookup
MODULE_DIR: Path = Path(os.environ.get("COCHEM_SPYCFIT_DIR", Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.home() / ".cochem" / "CoChem-SpycFit"))
EXTRACTOR_SCRIPT: Path = MODULE_DIR / "src" / "cochem_spycfit" / "quantum_extractor.py"
REAL_ARTIFACTS_DIR: Path = Path(os.environ.get("COCHEM_ARTIFACTS_DIR", MODULE_DIR / "artifacts_test"))

# 3. Graceful Failure & Subprocess Safety (Zombie Process Sweeping)
def sweep_zombie_processes() -> None:
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.Error, OSError) as e:
        logging.error(f"Failed to sweep zombie processes: {e}")
        raise RuntimeError("Zombie process sweeping failed") from e

atexit.register(sweep_zombie_processes)

def test_quantum_extractor_execution(tmp_path: Path) -> None:
    """
    Executes the quantum extractor against a REAL physical landscape.h5.
    Mocks and stubs are strictly forbidden. Exception deflections are tightened.
    """
    ephemeral_artifact_dir: Path = tmp_path / "artifacts"
    ephemeral_artifact_dir.mkdir(exist_ok=True)
    
    # Locate real physical HDF5 registry
    real_h5_path: Path = REAL_ARTIFACTS_DIR / "landscape.h5"
    if not real_h5_path.exists():
        real_h5_path = REAL_ARTIFACTS_DIR / "artifacts_test" / "landscape.h5"
    if not real_h5_path.exists():
        # Strict failure on missing physical data (No exception deflection)
        raise FileNotFoundError(f"[MISSING DATA] Real physical landscape.h5 not found at {real_h5_path}. Synthetic generation is forbidden.")
        
    # Copy to ephemeral dir to prevent mutating the source
    ephemeral_h5_path: Path = ephemeral_artifact_dir / "landscape.h5"
    shutil.copy(real_h5_path, ephemeral_h5_path)
    
    # 2. Verify isomer exists in real data
    isomer_id: str = "isomer_001"
    with h5py.File(ephemeral_h5_path, "r") as f:
        if isomer_id not in f:
            raise KeyError(f"[MISSING DATA] Isomer {isomer_id} not found in physical HDF5 registry.")
        n_dim: int = len(f[isomer_id]["symbols"][:]) * 3

    # 3. Native invocation of the extractor (Strict Subprocess Safety)
    env = os.environ.copy()
    env["COCHEM_ARTIFACT_DIR"] = str(ephemeral_artifact_dir)
    env["COCHEM_TARGET_ISOMER"] = isomer_id
    
    cmd = [sys.executable, str(EXTRACTOR_SCRIPT)]
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=60)
        logging.info("Extractor execution succeeded.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Extractor failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        raise RuntimeError(f"Physical execution of extractor failed with return code {e.returncode}") from e
    except subprocess.TimeoutExpired as e:
        logging.error(f"Extractor timed out after {e.timeout}s")
        raise RuntimeError(f"Extractor timed out after {e.timeout}s") from e

    # 4. Domain-specific correctness checks
    tensor_buffer: Path = ephemeral_artifact_dir / "spycfit_tensor_buffer.npz"
    if not tensor_buffer.exists():
        raise FileNotFoundError(f"[MISSING DATA] spycfit_tensor_buffer.npz was not produced by the physical run in {ephemeral_artifact_dir}.")
    
    npz_data = np.load(tensor_buffer, allow_pickle=True)
    assert "mw_hessian" in npz_data.files, "[MISSING DATA] Mass-weighted Hessian missing from output"
    
    # Mass weighted Hessian must maintain dimensions
    mw_hessian = npz_data["mw_hessian"]
    assert mw_hessian.shape == (n_dim, n_dim), f"Invalid Hessian shape: {mw_hessian.shape} != ({n_dim}, {n_dim})"
    
    # Ensure Hermitian symmetry is strict
    np.testing.assert_allclose(mw_hessian, mw_hessian.T, rtol=1e-5, err_msg="Hessian Hermitian symmetry violation")
