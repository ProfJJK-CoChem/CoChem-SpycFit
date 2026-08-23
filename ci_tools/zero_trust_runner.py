import subprocess
import tempfile
import uuid
import sys
import shutil
import os
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_in_quarantine(command: List[str]) -> int:
    quarantine_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / f"cochem_exec_{quarantine_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Running in quarantine environment: {temp_dir}")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        result = subprocess.run(
            command,
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True
        )
        
        logging.info("--- STDOUT ---")
        logging.info(result.stdout)
        if result.stderr:
            logging.info("--- STDERR ---")
            logging.info(result.stderr)
            
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Process failed with return code {e.returncode}")
        logging.error(f"--- STDOUT ---\n{e.stdout}")
        logging.error(f"--- STDERR ---\n{e.stderr}")
        return e.returncode
    except subprocess.TimeoutExpired as e:
        logging.error(f"Process timed out after {e.timeout}s")
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: zero_trust_runner.py <command...>")
        sys.exit(1)
    
    sys.exit(run_in_quarantine(sys.argv[1:]))
