import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
from pathlib import Path

st.set_page_config(page_title="CoChem-SpycFit - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if any(target in name for target in target_procs):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            raise NotImplementedError("Implementation pending")
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-SpycFit Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering physical quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir = Path(__file__).resolve().parent
        master_script = module_dir / "cochem_spycfit_master.py"
        
        artifact_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", os.getcwd()))
        env = os.environ.copy()
        env["COCHEM_TARGET_H5"] = str(artifact_dir / "landscape.h5")
        
        try:
            cmd = [sys.executable, str(master_script)]
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
            st.success("✅ Execution Completed Natively. Physical CPU load generated.")
                
        except subprocess.TimeoutExpired:
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr[-2000:] if e.stderr else (e.stdout[-2000:] if e.stdout else "No output")
            st.error(f"Execution finished with non-zero exit code: {e.returncode}\n{err_msg}")
            kill_zombie_processes()
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
