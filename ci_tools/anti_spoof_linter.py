import ast
import json
import sys
from pathlib import Path
import os
import logging
from typing import List, Set, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AMNESTY_FILE: Path = Path(".anti_spoof_amnesty.json")
BANNED_WORDS: List[str] = ["mock", "dummy", "placeholder", "fake", "stub", "synthetic"]

def load_amnesty() -> Set[str]:
    if AMNESTY_FILE.exists():
        with open(AMNESTY_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_amnesty(files: Set[str]) -> None:
    with open(AMNESTY_FILE, "w") as f:
        json.dump(list(files), f, indent=4)

def check_file(filepath: Path) -> List[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
        
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    violations: List[str] = []
    
    class SpoofVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if "unittest.mock" in alias.name:
                    if "tests/api_mocks" not in str(filepath.as_posix()):
                        violations.append(f"Line {node.lineno}: Banned import 'unittest.mock' outside api_mocks")
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module and "unittest.mock" in node.module:
                if "tests/api_mocks" not in str(filepath.as_posix()):
                    violations.append(f"Line {node.lineno}: Banned import 'unittest.mock' outside api_mocks")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "np" and node.func.attr in ["linspace", "sin"]:
                    violations.append(f"Line {node.lineno}: Banned mock data generator 'np.{node.func.attr}'")
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for word in BANNED_WORDS:
                if word in node.name.lower():
                    violations.append(f"Line {node.lineno}: Banned word '{word}' in function name '{node.name}'")
            self.generic_visit(node)
            
        def visit_Name(self, node: ast.Name) -> None:
            for word in BANNED_WORDS:
                if word in node.id.lower():
                    violations.append(f"Line {node.lineno}: Banned word '{word}' in variable name '{node.id}'")
            self.generic_visit(node)
            
    SpoofVisitor().visit(tree)
    return violations

def run_linter(strict_mode: bool) -> int:
    amnesty_files = load_amnesty()
    failed = False
    all_violations: Dict[str, List[str]] = {}
    
    ignored_dirs = {".venv", ".git", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".trash"}
    for py_file in Path(".").rglob("*.py"):
        if any(part in ignored_dirs for part in py_file.parts):
            continue
        if py_file.as_posix() in amnesty_files and strict_mode:
            continue
            
        v = check_file(py_file)
        if v:
            all_violations[py_file.as_posix()] = v
            failed = True
            
    if not strict_mode:
        save_amnesty(set(all_violations.keys()))
        logging.info(f"Recorded {len(all_violations)} files in amnesty baseline.")
        return 0
        
    if failed:
        logging.error("[SPOOFING DETECTED] Violations found:")
        for f, v in all_violations.items():
            logging.error(f"File: {f}")
            for line in v:
                logging.error(f"  {line}")
        return 1
        
    logging.info("Anti-spoofing check passed.")
    return 0

if __name__ == "__main__":
    strict = os.environ.get("SPOOF_STRICT_MODE", "false").lower() == "true"
    sys.exit(run_linter(strict))
