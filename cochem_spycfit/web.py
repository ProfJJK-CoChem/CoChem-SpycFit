import streamlit as st
import subprocess
import os
import sys
import psutil
import logging
import time
import atexit
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cleanup_zombies():
    for proc in psutil.process_iter(['pid', 'status']):
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                proc.wait(timeout=1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass

atexit.register(cleanup_zombies)

st.set_page_config(page_title="CoChem-SpycFit - Pipeline UI", layout="wide")

st.title("🔬 CoChem-SpycFit Control Panel")
st.markdown("Execute the CoChem-SpycFit pipeline.")

with st.sidebar:
    st.header("Pipeline Configuration")
    uploaded_file = st.file_uploader("Upload landscape.h5 or spycfit_tensor_buffer.npz", type=["h5", "npz"])

if st.button("🚀 Execute Default Pipeline"):
    if not uploaded_file:
        st.error("[MISSING DATA] Please upload the required input file (.h5 or .npz) before executing.")
    else:
        st.info("Initiating Pipeline Execution...")
        
        artifact_dir_str = os.environ.get("COCHEM_ARTIFACT_DIR", str(Path.home() / ".cochem" / "artifacts"))
        artifact_dir = Path(artifact_dir_str).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        module_dir = Path(__file__).resolve().parent
        master_script = module_dir / "master.py"
        
        if not master_script.exists():
            st.error(f"[MISSING DATA] Master script not found at {master_script}")
            st.stop()
            
        log_file = artifact_dir / "master.log"
        
        # Save uploaded file to artifact_dir (working directory)
        save_path = artifact_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        env: dict[str, str] = os.environ.copy()
        env["COCHEM_ARTIFACT_DIR"] = str(artifact_dir)
        if uploaded_file.name.endswith(".h5"):
            env["COCHEM_TARGET_H5"] = str(save_path)
        
        log_container = st.empty()
        
        try:
            # Clear log file and initialize
            with open(log_file, 'w') as f:
                f.write("Starting execution...\n")
            
            cmd: list[str] = [sys.executable, str(master_script)]
            logging.info(f"Running command: {' '.join(cmd)}")
            
            with open(log_file, "a") as f_out:
                proc = subprocess.Popen(
                    cmd, 
                    cwd=str(artifact_dir),
                    env=env,
                    stdout=f_out,
                    stderr=subprocess.STDOUT
                )
                
                log_content = ""
                start_time = time.time()
                timeout = 3600  # 1 hour maximum
                
                with open(log_file, "r") as f:
                    while proc.poll() is None:
                        if time.time() - start_time > timeout:
                            logging.error("Execution timed out.")
                            st.error(f"Execution timed out after {timeout}s.")
                            proc.terminate()
                            break
                            
                        new_data = f.read()
                        if new_data:
                            log_content += new_data
                            log_container.code(log_content[-3000:], language="text")
                        time.sleep(0.5)
                    
                    # Read any remaining output after process finishes
                    new_data = f.read()
                    if new_data:
                        log_content += new_data
                        log_container.code(log_content[-3000:], language="text")
            
            if proc.returncode == 0:
                st.success("✅ Execution Completed.")
            else:
                st.error(f"Execution finished with non-zero exit code: {proc.returncode}")
                
        except Exception as e:
            logging.exception("Pipeline crashed during execution.")
            st.error(f"Pipeline crashed during execution: {str(e)}")
        finally:
            if 'proc' in locals() and proc.poll() is None:
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                            child.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except psutil.NoSuchProcess:
                    pass
            cleanup_zombies()
