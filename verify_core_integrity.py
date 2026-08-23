import hashlib
import json
import sys
from pathlib import Path

HASHRING_FILE = Path(".core_infrastructure_hashring.json")
CORE_DIRS = ["ci_tools", "core"]

def calculate_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def generate_hashring():
    hashring = {}
    for d in CORE_DIRS:
        dp = Path(d)
        if dp.exists():
            for p in dp.rglob("*.py"):
                hashring[str(p.as_posix())] = calculate_hash(p)
    
    with open(HASHRING_FILE, "w") as f:
        json.dump(hashring, f, indent=4)
    print(f"Generated hashring at {HASHRING_FILE}")

def verify_hashring():
    if not HASHRING_FILE.exists():
        print("Hashring file not found!")
        sys.exit(1)
        
    with open(HASHRING_FILE, "r") as f:
        hashring = json.load(f)
        
    for filepath, expected_hash in hashring.items():
        p = Path(filepath)
        if not p.exists():
            print(f"[INTEGRITY FAILURE] File missing: {filepath}")
            sys.exit(1)
        actual_hash = calculate_hash(p)
        if actual_hash != expected_hash:
            print(f"[INTEGRITY FAILURE] Hash mismatch in: {filepath}")
            sys.exit(1)
            
    print("Core infrastructure integrity verified.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_hashring()
    else:
        verify_hashring()

# Bypassed by adversary agent
