import os
import sys
import h5py
import numpy as np
import subprocess
import json
import logging
import psutil
import atexit
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgentState(BaseModel):
    status: str
    artifacts: List[str] = Field(default_factory=list)
    timestamp: str

def cleanup_zombies() -> None:
    """Sweep and terminate any zombie processes."""
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        for child in children:
            try:
                if child.status() == psutil.STATUS_ZOMBIE:
                    logging.info(f"Reaping zombie process PID: {child.pid}")
                    child.terminate()
                    child.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as e:
                logging.error(f"Failed to terminate child {child.pid}: {e}")
    except psutil.Error as e:
        logging.error(f"Cleanup failed: {e}")

atexit.register(cleanup_zombies)

def generate_sha256(file_path: Path) -> str:
    """Generate SHA-256 hash for a given file [E]."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def update_swarm_state(state_file: Path, agent_name: str, status: str, artifacts: List[str]) -> None:
    """Update the global swarm_state.json."""
    state_dict: Dict[str, Any] = {}
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state_dict = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Error reading state file {state_file}: {e}")
            logging.error("Aborting to prevent overwriting with corrupted state.")
            sys.exit(1)
            
    # Validate existing state
    validated_state: Dict[str, Any] = {}
    for k, v in state_dict.items():
        try:
            # We preserve existing non-conforming data but try to parse if possible
            if isinstance(v, dict) and "status" in v:
                validated_state[k] = AgentState(**v)
            else:
                validated_state[k] = v
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to validate state for agent {k}: {e}. Retaining raw.")
            validated_state[k] = v
            
    # Update for current agent
    validated_state[agent_name] = AgentState(
        status=status,
        artifacts=artifacts,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    # Save back
    out_dict = {}
    for k, v in validated_state.items():
        if hasattr(v, 'model_dump'):
            out_dict[k] = v.model_dump()
        elif hasattr(v, 'dict'):
            out_dict[k] = v.dict()
        else:
            out_dict[k] = v
            
    try:
        with open(state_file, "w") as f:
            json.dump(out_dict, f, indent=2)
        logging.info(f"Updated {state_file}")
    except OSError as e:
        logging.error(f"Failed to write to {state_file}: {e}")

def main() -> None:
    logging.info("Starting Quantum Tensor Extractor E2E physical execution...")
    
    # 1. Prepare artifact directory
    script_dir: Path = Path(__file__).resolve().parent
    repo_root: Path = script_dir.parent
    artifact_dir: Path = script_dir / "artifacts_test"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    h5_path: Path = artifact_dir / "landscape.h5"
    
    # 2. Copy real physical landscape.h5
    source_h5: Path = repo_root / "CoChem-TOPOS" / "artifacts" / "interactive_session" / "landscape.h5"
    import shutil
    try:
        shutil.copy2(source_h5, h5_path)
        logging.info(f"Copied real physical landscape.h5 from {source_h5} to {h5_path}")
    except FileNotFoundError:
        logging.error(f"[MISSING DATA] Source h5 file not found at {source_h5}")
        sys.exit(1)
    
    # 3. Execute the quantum extractor
    extractor_script: Path = script_dir / "quantum_extractor.py"
    if not extractor_script.exists():
         logging.error(f"[MISSING DATA] Extractor script not found at {extractor_script}")
         sys.exit(1)

    env: Dict[str, str] = os.environ.copy()
    env["COCHEM_ARTIFACT_DIR"] = str(artifact_dir)
    env["COCHEM_TARGET_ISOMER"] = "isomer_001"
    
    cmd: List[str] = [sys.executable, str(extractor_script)]
    
    logging.info(f"Executing: {' '.join(cmd)}")
    success: bool = False
    expected_output: Path = artifact_dir / "spycfit_tensor_buffer.npz"

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, check=True)
        logging.info("STDOUT:\n" + result.stdout)
        if result.stderr:
            logging.warning("STDERR:\n" + result.stderr)
        
        # Verification
        if expected_output.exists():
            success = True
            logging.info("Execution SUCCESS. Tensor buffer generated.")
            
            sha256_val = generate_sha256(expected_output)
            logging.info(f"[E] Provenance Hash for {expected_output.name}: {sha256_val}")
            
            npz_data = np.load(expected_output, allow_pickle=True)
            logging.info(f"Keys in npz: {npz_data.files}")
            
            if "mw_hessian" in npz_data:
                mw_hessian = npz_data["mw_hessian"]
                logging.info(f"MW Hessian shape: {mw_hessian.shape}")
                
                assert np.all(np.isfinite(mw_hessian)), "Hessian contains non-finite values."
                assert mw_hessian.shape == (9, 9), f"Expected shape (9, 9), got {mw_hessian.shape}"
                assert np.allclose(mw_hessian, mw_hessian.T), "Hessian is not strictly Hermitian/symmetric."
            else:
                logging.error("[MISSING DATA] mw_hessian not found in npz output.")
                success = False
        else:
            logging.error(f"[MISSING DATA] Expected output not found: {expected_output}")
            
    except subprocess.TimeoutExpired as e:
        logging.error(f"Execution TIMEOUT: {e}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Execution FAILED with return code {e.returncode}")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)

    # Update swarm_state.json
    state_file: Path = repo_root.parent / "swarm_state.json"
    if not state_file.exists():
        state_file = repo_root / "CoChem-BASE" / "swarm_state.json"
    if not state_file.exists():
        state_file = repo_root / "swarm_state.json"
        
    update_swarm_state(
        state_file=state_file,
        agent_name="cochem-tester",
        status="SUCCESS" if success else "FAILURE",
        artifacts=[str(expected_output)] if success else []
    )

if __name__ == "__main__":
    main()
