import ast
import json
import sys
from pathlib import Path
import os

AMNESTY_FILE = Path(".anti_spoof_amnesty.json")
BANNED_WORDS = ["mock", "dummy", "placeholder", "fake", "stub"]

def load_amnesty():
    if AMNESTY_FILE.exists():
        with open(AMNESTY_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_amnesty(files):
    with open(AMNESTY_FILE, "w") as f:
        json.dump(list(files), f, indent=4)

def check_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return []
        
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    violations = []
    
    class SpoofVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                if "unittest.mock" in alias.name:
                    if "tests/api_mocks" not in str(filepath.as_posix()):
                        violations.append(f"Line {node.lineno}: Banned import 'unittest.mock' outside api_mocks")
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            if node.module and "unittest.mock" in node.module:
                if "tests/api_mocks" not in str(filepath.as_posix()):
                    violations.append(f"Line {node.lineno}: Banned import 'unittest.mock' outside api_mocks")
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            for word in BANNED_WORDS:
                if word in node.name.lower():
                    violations.append(f"Line {node.lineno}: Banned word '{word}' in function name '{node.name}'")
            self.generic_visit(node)
            
        def visit_Name(self, node):
            for word in BANNED_WORDS:
                if word in node.id.lower():
                    violations.append(f"Line {node.lineno}: Banned word '{word}' in variable name '{node.id}'")
            self.generic_visit(node)
            
    SpoofVisitor().visit(tree)
    return violations

def run_linter(strict_mode):
    amnesty_files = load_amnesty()
    failed = False
    all_violations = {}
    
    for py_file in Path(".").rglob("*.py"):
        if py_file.as_posix() in amnesty_files and strict_mode:
            continue
            
        v = check_file(py_file)
        if v:
            all_violations[py_file.as_posix()] = v
            failed = True
            
    if not strict_mode:
        save_amnesty(all_violations.keys())
        print(f"Recorded {len(all_violations)} files in amnesty baseline.")
        return 0
        
    if failed:
        print("[SPOOFING DETECTED] Violations found:")
        for f, v in all_violations.items():
            print(f"File: {f}")
            for line in v:
                print(f"  {line}")
        return 1
        
    print("Anti-spoofing check passed.")
    return 0

if __name__ == "__main__":
    strict = os.environ.get("SPOOF_STRICT_MODE", "false").lower() == "true"
    sys.exit(run_linter(strict))
