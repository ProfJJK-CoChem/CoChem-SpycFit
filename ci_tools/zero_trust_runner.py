import subprocess
import tempfile
import uuid
import sys
import shutil
import os
from pathlib import Path

def run_in_quarantine(command):
    quarantine_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / f"cochem_exec_{quarantine_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running in quarantine environment: {temp_dir}")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        result = subprocess.run(
            command,
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        print("--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
            
        return result.returncode
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: zero_trust_runner.py <command...>")
        sys.exit(1)
    
    sys.exit(run_in_quarantine(sys.argv[1:]))
