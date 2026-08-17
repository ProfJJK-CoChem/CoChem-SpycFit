import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="CoChem-SpycFit - Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    target_procs: list[str] = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name_info: str | None = proc.info.get('name')
            if name_info is not None:
                name: str = name_info.lower()
                if any(target in name for target in target_procs):
                    logging.warning(f"Terminating zombie process: {name}")
                    proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-SpycFit Control Panel")
st.markdown("Execute the CoChem-SpycFit pipeline.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles: str = st.text_input("Target SMILES", "CCO")
    run_mode: str = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Executing pipeline for {target_smiles}..."):
        st.info("Initiating Pipeline Execution...")
        
        module_dir: Path = Path(__file__).resolve().parent
        master_script: Path = module_dir / "cochem_spycfit_master.py"
        
        artifact_dir: Path = Path(os.environ.get("COCHEM_ARTIFACT_DIR", Path.home() / "cochem_artifacts"))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env: dict[str, str] = os.environ.copy()
        env["COCHEM_TARGET_H5"] = str(artifact_dir / "landscape.h5")
        env["COCHEM_TARGET_SMILES"] = target_smiles
        env["COCHEM_RUN_MODE"] = run_mode
        
        try:
            cmd: list[str] = [sys.executable, str(master_script)]
            logging.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=3600, 
                cwd=str(module_dir),
                env=env
            )
            
            st.code(result.stdout[-3000:], language="text")
            st.success("✅ Execution Completed.")
                
        except subprocess.TimeoutExpired:
            logging.error("Execution timed out. Purging zombies.")
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            err_msg: str = e.stderr[-2000:] if e.stderr else (e.stdout[-2000:] if e.stdout else "No output")
            logging.error(f"Execution failed with code {e.returncode}: {err_msg}")
            st.error(f"Execution finished with non-zero exit code: {e.returncode}\n{err_msg}")
            kill_zombie_processes()
        except Exception as e:
            logging.exception("Pipeline crashed during execution.")
            st.error(f"Pipeline crashed during execution: {str(e)}")
            kill_zombie_processes()
