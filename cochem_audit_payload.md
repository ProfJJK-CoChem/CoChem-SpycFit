Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\01_P1_T01_fitstate_model_dag_commit_manager.md.
Original prompt:
# Prompt 01 — Phase 1, Task Group 01: FitState Pydantic Model & DAG CommitManager

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 1 — HDF5 DAG State Manager & Persistence
> **Phase:** 1 (Tasks 1–20) — State Management & RAM Protection
> **Dependency:** None (first prompt in the pipeline)

---

## Objective

Create the immutable version-control data model and branching DAG logic for spectroscopic assignments. This module defines how user decisions are tracked, hashed, and serialized as a Directed Acyclic Graph (DAG) of `FitState` commits, enabling non-destructive "time-travel" reversion.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\core_engine\cochem_spycfit_state.py
```

> **Scope:** This prompt targets ONLY the file above. The `__init__.py` export update is handled separately in Prompt 01b.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Immutable FitState model and DAG CommitManager for spectroscopic assignments."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
```

### 2. `FitState` — Frozen Pydantic v2 Model

Define an immutable Pydantic v2 `BaseModel` representing a single snapshot of the fitting state:

```python
class FitState(BaseModel):
    """Immutable snapshot of a spectroscopic fitting state."""
    model_config = ConfigDict(frozen=True)

    state_id: str = Field(
        default="",
        description="SHA-256 hash uniquely identifying this state. Auto-computed if empty."
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the parent state. None for root."
    )
    locked_assignments: tuple[tuple[int, int, int, int, int, int, float], ...] = Field(
        ...,
        description=(
            "Tuple of locked assignments. Each entry is "
            "(J_upper, Ka_upper, Kc_upper, J_lower, Ka_lower, Kc_lower, nu_exp_MHz)."
        )
    )
    ml_weights_pointer: Optional[str] = Field(
        default=None,
        description="HDF5 dataset key pointing to serialized GP weights in Tier 2 tensor cache."
    )
    physical_constants: dict[str, float] = Field(
        default_factory=dict,
        description="Fitted rotational constants (A, B, C, DJ, DJK, DK, etc.) in MHz."
    )
    global_rmsd: float = Field(
        ...,
        description="Root-mean-square deviation of the fit in kHz."
    )
    global_chi2: Optional[float] = Field(
        default=None,
        description="Dimensionless chi-squared statistic."
    )
    timestamp: str = Field(
        default="",
        description="ISO 8601 timestamp of commit creation. Auto-generated if empty."
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_compute_fields(cls, values: dict) -> dict:
        """Auto-compute state_id and timestamp if not provided."""
        if not values.get("timestamp"):
            values["timestamp"] = datetime.now(timezone.utc).isoformat()
        if not values.get("state_id"):
            values["state_id"] = cls._compute_state_id(
                parent_id=values.get("parent_id"),
                locked_assignments=values.get("locked_assignments", ()),
                physical_constants=values.get("physical_constants", {}),
            )
        return values

    @staticmethod
    def _compute_state_id(
        parent_id: Optional[str],
        locked_assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
        physical_constants: dict[str, float],
    ) -> str:
        """Compute a deterministic SHA-256 hash from the state's content."""
        payload = json.dumps({
            "parent_id": parent_id,
            "locked_assignments": locked_assignments,
            "physical_constants": physical_constants,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

### 3. `CommitManager` — DAG Branching Logic

```python
class CommitManager:
    """Directed Acyclic Graph manager for FitState version control."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """
        Initialize the CommitManager.

        Parameters
        ----------
        registry_path : Path, optional
            Absolute path to the JSON registry file. Defaults to
            Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json".
            Honors the COCHEM_ARTIFACTS_ROOT environment variable.
        """
        self._registry_path: Path = self._resolve_registry_path(registry_path)
        self._states: dict[str, FitState] = {}
        self._head: Optional[str] = None
        self.load_registry()

    def commit(
        self,
        locked_assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
        physical_constants: dict[str, float],
        global_rmsd: float,
        global_chi2: Optional[float] = None,
        ml_weights_pointer: Optional[str] = None,
    ) -> FitState:
        """
        Create a new FitState child of the current HEAD and append to the DAG.

        Parameters
        ----------
        locked_assignments : tuple of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        physical_constants : dict
            Fitted constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
        global_rmsd : float
            RMSD of the fit in kHz.
        global_chi2 : float, optional
            Dimensionless chi-squared.
        ml_weights_pointer : str, optional
            HDF5 key for GP weights.

        Returns
        -------
        FitState
            The newly created immutable state.
        """
        # Implementation: create FitState with parent_id=self._head,
        # add to self._states, update self._head

    def branch_from(self, state_id: str) -> None:
        """
        Set HEAD to a historical state_id for branching.

        Parameters
        ----------
        state_id : str
            SHA-256 hash of the historical state to branch from.

        Raises
        ------
        KeyError
            If state_id does not exist in the DAG.
        """
        # Implementation: validate state_id exists, set self._head = state_id

    def get_history(self, state_id: Optional[str] = None) -> list[FitState]:
        """
        Return the ordered ancestor chain from state_id back to root.

        Parameters
        ----------
        state_id : str, optional
            Starting state. Defaults to current HEAD.

        Returns
        -------
        list[FitState]
            Ordered list from root to the specified state.
        """
        # Implementation: follow parent_id links to root, reverse the list

    def get_branches(self) -> list[str]:
        """
        Return all leaf state_ids (branch tips with no children).

        Returns
        -------
        list[str]
            List of state_id hashes for DAG leaf nodes.
        """
        # Implementation: find states whose state_id is not any other state's parent_id

    def save_registry(self) -> None:
        """
        Atomically serialize the DAG to the Tier 2 JSON registry.

        Uses write-to-tmp + os.replace() to prevent corruption on crash.
        """
        # Implementation: serialize self._states to JSON, write to .tmp file,
        # then os.replace() to the final path

    def load_registry(self) -> None:
        """
        Load the DAG from the Tier 2 JSON registry if it exists.
        """
        # Implementation: read and deserialize from self._registry_path

    @staticmethod
    def _resolve_registry_path(user_path: Optional[Path]) -> Path:
        """Resolve the registry path using the Tier 2 hierarchy."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "cochem_fit_registry.json"
        default = Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        return default
```

---

## Validation Test Specification

> **IMPORTANT:** All test artifacts must be written to a Tier 3 ephemeral scratch directory created via `tempfile.TemporaryDirectory()`. The test must NOT write to Tier 2 persistent storage.

Write a validation test that:

1. Creates a `CommitManager` with `registry_path` pointing to a temp directory
2. Generates 50 sequential commits using the following authentic H₂O spectroscopic data:

   **Rotational constants:** A = 835840.288 MHz, B = 435351.717 MHz, C = 278138.700 MHz

   **Authentic low-J water transition frequencies (MHz):**
   | J'(Ka',Kc') ← J''(Ka'',Kc'') | ν_obs (MHz) |
   |:---|:---|
   | 1(1,0) ← 1(0,1) | 556935.985 |
   | 2(1,1) ← 2(0,2) | 752033.227 |
   | 3(1,2) ← 3(0,3) | 1097364.791 |
   | 4(1,3) ← 4(0,4) | 1602219.182 |
   | 5(1,4) ← 5(0,5) | 2264148.770 |
   | 1(1,1) ← 0(0,0) | 1113342.964 |
   | 2(0,2) ← 1(1,1) | 987926.764 |
   | 2(1,2) ← 1(0,1) | 1669904.775 |
   | 3(0,3) ← 2(1,2) | 1716769.318 |
   | 3(1,3) ← 2(0,2) | 2196345.756 |

   Use subsets of these transitions across the 50 commits, incrementally adding 1-2 lines per commit.

3. Branches off commit #25, perturbing DJ from 37.594 MHz to 40.0 MHz
4. Adds 10 more commits on the new branch
5. **Assertions:**
   - `get_history(commit_25_id)` returns exactly 26 states (root + 25 commits)
   - `get_branches()` returns exactly 2 leaf nodes
   - After `save_registry()` + `load_registry()`, all states are recovered identically
   - All `state_id` hashes are deterministic (same inputs → same hash)

---

## Acceptance Criteria

- [ ] `FitState` is a frozen Pydantic v2 model with deterministic SHA-256 `state_id`
- [ ] `FitState` auto-computes `state_id` and `timestamp` via `@model_validator`
- [ ] `CommitManager` supports linear commits, historical branching, and ancestry traversal
- [ ] All method signatures include full type annotations, docstrings, and parameter descriptions
- [ ] Registry serializes atomically to Tier 2 JSON via `os.replace()`
- [ ] All paths use `pathlib.Path`; `COCHEM_ARTIFACTS_ROOT` env var is honored
- [ ] Validation test uses authentic H₂O spectroscopic data (table provided above)
- [ ] Test registry outputs quarantined in Tier 3 `tempfile.TemporaryDirectory()`

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\ci_tools\anti_spoof_linter.py ---
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\core_engine\cochem_spycfit_state.py ---
"""
Immutable FitState model and DAG CommitManager for spectroscopic assignments.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FitState(BaseModel):
    """Immutable snapshot of a spectroscopic fitting state."""

    model_config = ConfigDict(frozen=True)

    state_id: str = Field(
        default="",
        description=(
            "SHA-256 hash uniquely identifying this state. "
            "Auto-computed if empty."
        ),
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the parent state. None for root.",
    )
    locked_assignments: (
        tuple[tuple[int, int, int, int, int, int, float], ...]
    ) = Field(
        ...,
        description=(
            "Tuple of locked assignments. Each entry is "
            "(J_upper, Ka_upper, Kc_upper, J_lower, "
            "Ka_lower, Kc_lower, nu_exp_MHz)."
        ),
    )
    ml_weights_pointer: Optional[str] = Field(
        default=None,
        description=(
            "HDF5 dataset key pointing to serialized GP weights "
            "in Tier 2 tensor cache."
        ),
    )
    physical_constants: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fitted rotational constants (A, B, C, DJ, DJK, DK, etc.) "
            "in MHz."
        ),
    )
    global_rmsd: float = Field(
        ...,
        description="Root-mean-square deviation of the fit in kHz.",
    )
    global_chi2: Optional[float] = Field(
        default=None,
        description="Dimensionless chi-squared statistic.",
    )
    timestamp: str = Field(
        default="",
        description=(
            "ISO 8601 timestamp of commit creation. "
            "Auto-generated if empty."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_compute_fields(cls, values: Any) -> Any:
        """Auto-compute state_id and timestamp if not provided."""
        if isinstance(values, dict):
            values = dict(values)
            if not values.get("timestamp"):
                values["timestamp"] = datetime.now(timezone.utc).isoformat()
            if not values.get("state_id"):
                values["state_id"] = cls._compute_state_id(
                    parent_id=values.get("parent_id"),
                    locked_assignments=values.get("locked_assignments", ()),
                    physical_constants=values.get("physical_constants", {}),
                )
        return values

    @staticmethod
    def _compute_state_id(
        parent_id: Optional[str],
        locked_assignments: Union[tuple, list],
        physical_constants: dict[str, float],
    ) -> str:
        """Compute a deterministic SHA-256 hash from state content."""
        payload = json.dumps(
            {
                "parent_id": parent_id,
                "locked_assignments": locked_assignments,
                "physical_constants": physical_constants,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CommitManager:
    """Directed Acyclic Graph manager for FitState version control."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """
        Initialize the CommitManager.

        Parameters
        ----------
        registry_path : Path, optional
            Absolute path to the JSON registry file. Defaults to
            Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json".
            Honors the COCHEM_ARTIFACTS_ROOT environment variable.
        """
        self._registry_path: Path = self._resolve_registry_path(registry_path)
        self._states: dict[str, FitState] = {}
        self._head: Optional[str] = None
        self.load_registry()

    @property
    def head(self) -> Optional[str]:
        """Return the state_id of the current HEAD commit, or None."""
        return self._head

    @property
    def states(self) -> dict[str, FitState]:
        """Return a copy of all FitState objects in the DAG."""
        return dict(self._states)

    @property
    def registry_path(self) -> Path:
        """Return the resolved filesystem Path to the JSON registry."""
        return self._registry_path

    def commit(
        self,
        locked_assignments: (
            tuple[tuple[int, int, int, int, int, int, float], ...]
        ),
        physical_constants: dict[str, float],
        global_rmsd: float,
        global_chi2: Optional[float] = None,
        ml_weights_pointer: Optional[str] = None,
    ) -> FitState:
        """
        Create a new FitState child of the current HEAD and append to DAG.

        Parameters
        ----------
        locked_assignments : tuple of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        physical_constants : dict
            Fitted constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
        global_rmsd : float
            RMSD of the fit in kHz.
        global_chi2 : float, optional
            Dimensionless chi-squared.
        ml_weights_pointer : str, optional
            HDF5 key for GP weights.

        Returns
        -------
        FitState
            The newly created immutable state.
        """
        state = FitState(
            parent_id=self._head,
            locked_assignments=locked_assignments,
            physical_constants=physical_constants,
            global_rmsd=global_rmsd,
            global_chi2=global_chi2,
            ml_weights_pointer=ml_weights_pointer,
        )
        self._states[state.state_id] = state
        self._head = state.state_id
        return state

    def branch_from(self, state_id: str) -> None:
        """
        Set HEAD to a historical state_id for branching.

        Parameters
        ----------
        state_id : str
            SHA-256 hash of the historical state to branch from.

        Raises
        ------
        KeyError
            If state_id does not exist in the DAG.
        """
        if state_id not in self._states:
            raise KeyError(
                f"State ID '{state_id}' does not exist in the DAG."
            )
        self._head = state_id

    def get_history(self, state_id: Optional[str] = None) -> list[FitState]:
        """
        Return the ordered ancestor chain from root to state_id.

        Parameters
        ----------
        state_id : str, optional
            Starting state. Defaults to current HEAD.

        Returns
        -------
        list[FitState]
            Ordered list from root to the specified state.

        Raises
        ------
        KeyError
            If specified state_id does not exist in the DAG.
        """
        target_id = state_id if state_id is not None else self._head
        if target_id is None:
            return []
        if target_id not in self._states:
            raise KeyError(
                f"State ID '{target_id}' does not exist in the DAG."
            )

        history_chain: list[FitState] = []
        current_id: Optional[str] = target_id
        while current_id is not None:
            state = self._states[current_id]
            history_chain.append(state)
            current_id = state.parent_id

        history_chain.reverse()
        return history_chain

    def get_branches(self) -> list[str]:
        """
        Return all leaf state_ids (branch tips with no children).

        Returns
        -------
        list[str]
            List of state_id hashes for DAG leaf nodes.
        """
        if not self._states:
            return []
        parent_ids = {
            s.parent_id
            for s in self._states.values()
            if s.parent_id is not None
        }
        return [sid for sid in self._states if sid not in parent_ids]

    def get_state(self, state_id: str) -> FitState:
        """
        Retrieve a FitState by its unique state_id.

        Parameters
        ----------
        state_id : str
            SHA-256 state identifier.

        Returns
        -------
        FitState
            The requested state snapshot.

        Raises
        ------
        KeyError
            If state_id is not present in the DAG.
        """
        if state_id not in self._states:
            raise KeyError(
                f"State ID '{state_id}' does not exist in the DAG."
            )
        return self._states[state_id]

    def save_registry(self) -> None:
        """
        Atomically serialize the DAG to the Tier 2 JSON registry.

        Uses write-to-tmp + os.replace() to prevent corruption on crash.
        """
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_payload = {
            "head": self._head,
            "states": {
                sid: state.model_dump()
                for sid, state in self._states.items()
            },
        }
        parent_dir = self._registry_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=parent_dir,
            delete=False,
            encoding="utf-8",
            suffix=".tmp",
        ) as temp_file:
            json.dump(registry_payload, temp_file, indent=2)
            temp_file_path = Path(temp_file.name)

        os.replace(temp_file_path, self._registry_path)

    def load_registry(self) -> None:
        """
        Load the DAG from the Tier 2 JSON registry if it exists.
        """
        if (
            not self._registry_path.exists()
            or not self._registry_path.is_file()
        ):
            return

        with open(
            self._registry_path, "r", encoding="utf-8"
        ) as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, dict):
            return

        if "states" in payload and isinstance(payload["states"], dict):
            self._head = payload.get("head")
            self._states = {
                sid: FitState.model_validate(raw_state)
                for sid, raw_state in payload["states"].items()
            }
        else:
            self._states = {}
            for sid, raw_state in payload.items():
                if isinstance(raw_state, dict) and (
                    "locked_assignments" in raw_state
                    or "state_id" in raw_state
                ):
                    self._states[sid] = FitState.model_validate(raw_state)
            if self._states and self._head is None:
                self._head = list(self._states.keys())[-1]

    @staticmethod
    def _resolve_registry_path(user_path: Optional[Path]) -> Path:
        """
        Resolve the registry path using the Tier 2 hierarchy.

        Parameters
        ----------
        user_path : Path, optional
            Explicit path provided by the caller.

        Returns
        -------
        Path
            Resolved absolute Path to the registry JSON file.
        """
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "cochem_fit_registry.json"
        default = Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        return default

    def __len__(self) -> int:
        """Return the number of commits in the DAG."""
        return len(self._states)

    def __getitem__(self, state_id: str) -> FitState:
        """Retrieve a state snapshot by state_id index."""
        return self.get_state(state_id)

    def __contains__(self, state_id: str) -> bool:
        """Check if a state_id exists in the DAG."""
        return state_id in self._states

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_spycfit_state.py ---
"""
Immutable FitState model and DAG CommitManager for spectroscopic assignments.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FitState(BaseModel):
    """Immutable snapshot of a spectroscopic fitting state."""

    model_config = ConfigDict(frozen=True)

    state_id: str = Field(
        default="",
        description=(
            "SHA-256 hash uniquely identifying this state. "
            "Auto-computed if empty."
        ),
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the parent state. None for root.",
    )
    locked_assignments: (
        tuple[tuple[int, int, int, int, int, int, float], ...]
    ) = Field(
        ...,
        description=(
            "Tuple of locked assignments. Each entry is "
            "(J_upper, Ka_upper, Kc_upper, J_lower, "
            "Ka_lower, Kc_lower, nu_exp_MHz)."
        ),
    )
    ml_weights_pointer: Optional[str] = Field(
        default=None,
        description=(
            "HDF5 dataset key pointing to serialized GP weights "
            "in Tier 2 tensor cache."
        ),
    )
    physical_constants: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fitted rotational constants (A, B, C, DJ, DJK, DK, etc.) "
            "in MHz."
        ),
    )
    global_rmsd: float = Field(
        ...,
        description="Root-mean-square deviation of the fit in kHz.",
    )
    global_chi2: Optional[float] = Field(
        default=None,
        description="Dimensionless chi-squared statistic.",
    )
    timestamp: str = Field(
        default="",
        description=(
            "ISO 8601 timestamp of commit creation. "
            "Auto-generated if empty."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_compute_fields(cls, values: Any) -> Any:
        """Auto-compute state_id and timestamp if not provided."""
        if isinstance(values, dict):
            values = dict(values)
            if not values.get("timestamp"):
                values["timestamp"] = datetime.now(timezone.utc).isoformat()
            if not values.get("state_id"):
                values["state_id"] = cls._compute_state_id(
                    parent_id=values.get("parent_id"),
                    locked_assignments=values.get("locked_assignments", ()),
                    physical_constants=values.get("physical_constants", {}),
                )
        return values

    @staticmethod
    def _compute_state_id(
        parent_id: Optional[str],
        locked_assignments: Union[tuple, list],
        physical_constants: dict[str, float],
    ) -> str:
        """Compute a deterministic SHA-256 hash from state content."""
        payload = json.dumps(
            {
                "parent_id": parent_id,
                "locked_assignments": locked_assignments,
                "physical_constants": physical_constants,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CommitManager:
    """Directed Acyclic Graph manager for FitState version control."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """
        Initialize the CommitManager.

        Parameters
        ----------
        registry_path : Path, optional
            Absolute path to the JSON registry file. Defaults to
            Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json".
            Honors the COCHEM_ARTIFACTS_ROOT environment variable.
        """
        self._registry_path: Path = self._resolve_registry_path(registry_path)
        self._states: dict[str, FitState] = {}
        self._head: Optional[str] = None
        self.load_registry()

    @property
    def head(self) -> Optional[str]:
        """Return the state_id of the current HEAD commit, or None."""
        return self._head

    @property
    def states(self) -> dict[str, FitState]:
        """Return a copy of all FitState objects in the DAG."""
        return dict(self._states)

    @property
    def registry_path(self) -> Path:
        """Return the resolved filesystem Path to the JSON registry."""
        return self._registry_path

    def commit(
        self,
        locked_assignments: (
            tuple[tuple[int, int, int, int, int, int, float], ...]
        ),
        physical_constants: dict[str, float],
        global_rmsd: float,
        global_chi2: Optional[float] = None,
        ml_weights_pointer: Optional[str] = None,
    ) -> FitState:
        """
        Create a new FitState child of the current HEAD and append to DAG.

        Parameters
        ----------
        locked_assignments : tuple of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        physical_constants : dict
            Fitted constants (A, B, C, DJ, DJK, DK, etc.) in MHz.
        global_rmsd : float
            RMSD of the fit in kHz.
        global_chi2 : float, optional
            Dimensionless chi-squared.
        ml_weights_pointer : str, optional
            HDF5 key for GP weights.

        Returns
        -------
        FitState
            The newly created immutable state.
        """
        state = FitState(
            parent_id=self._head,
            locked_assignments=locked_assignments,
            physical_constants=physical_constants,
            global_rmsd=global_rmsd,
            global_chi2=global_chi2,
            ml_weights_pointer=ml_weights_pointer,
        )
        self._states[state.state_id] = state
        self._head = state.state_id
        return state

    def branch_from(self, state_id: str) -> None:
        """
        Set HEAD to a historical state_id for branching.

        Parameters
        ----------
        state_id : str
            SHA-256 hash of the historical state to branch from.

        Raises
        ------
        KeyError
            If state_id does not exist in the DAG.
        """
        if state_id not in self._states:
            raise KeyError(
                f"State ID '{state_id}' does not exist in the DAG."
            )
        self._head = state_id

    def get_history(self, state_id: Optional[str] = None) -> list[FitState]:
        """
        Return the ordered ancestor chain from root to state_id.

        Parameters
        ----------
        state_id : str, optional
            Starting state. Defaults to current HEAD.

        Returns
        -------
        list[FitState]
            Ordered list from root to the specified state.

        Raises
        ------
        KeyError
            If specified state_id does not exist in the DAG.
        """
        target_id = state_id if state_id is not None else self._head
        if target_id is None:
            return []
        if target_id not in self._states:
            raise KeyError(
                f"State ID '{target_id}' does not exist in the DAG."
            )

        history_chain: list[FitState] = []
        current_id: Optional[str] = target_id
        while current_id is not None:
            state = self._states[current_id]
            history_chain.append(state)
            current_id = state.parent_id

        history_chain.reverse()
        return history_chain

    def get_branches(self) -> list[str]:
        """
        Return all leaf state_ids (branch tips with no children).

        Returns
        -------
        list[str]
            List of state_id hashes for DAG leaf nodes.
        """
        if not self._states:
            return []
        parent_ids = {
            s.parent_id
            for s in self._states.values()
            if s.parent_id is not None
        }
        return [sid for sid in self._states if sid not in parent_ids]

    def get_state(self, state_id: str) -> FitState:
        """
        Retrieve a FitState by its unique state_id.

        Parameters
        ----------
        state_id : str
            SHA-256 state identifier.

        Returns
        -------
        FitState
            The requested state snapshot.

        Raises
        ------
        KeyError
            If state_id is not present in the DAG.
        """
        if state_id not in self._states:
            raise KeyError(
                f"State ID '{state_id}' does not exist in the DAG."
            )
        return self._states[state_id]

    def save_registry(self) -> None:
        """
        Atomically serialize the DAG to the Tier 2 JSON registry.

        Uses write-to-tmp + os.replace() to prevent corruption on crash.
        """
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_payload = {
            "head": self._head,
            "states": {
                sid: state.model_dump()
                for sid, state in self._states.items()
            },
        }
        parent_dir = self._registry_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=parent_dir,
            delete=False,
            encoding="utf-8",
            suffix=".tmp",
        ) as temp_file:
            json.dump(registry_payload, temp_file, indent=2)
            temp_file_path = Path(temp_file.name)

        os.replace(temp_file_path, self._registry_path)

    def load_registry(self) -> None:
        """
        Load the DAG from the Tier 2 JSON registry if it exists.
        """
        if (
            not self._registry_path.exists()
            or not self._registry_path.is_file()
        ):
            return

        with open(
            self._registry_path, "r", encoding="utf-8"
        ) as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, dict):
            return

        if "states" in payload and isinstance(payload["states"], dict):
            self._head = payload.get("head")
            self._states = {
                sid: FitState.model_validate(raw_state)
                for sid, raw_state in payload["states"].items()
            }
        else:
            self._states = {}
            for sid, raw_state in payload.items():
                if isinstance(raw_state, dict) and (
                    "locked_assignments" in raw_state
                    or "state_id" in raw_state
                ):
                    self._states[sid] = FitState.model_validate(raw_state)
            if self._states and self._head is None:
                self._head = list(self._states.keys())[-1]

    @staticmethod
    def _resolve_registry_path(user_path: Optional[Path]) -> Path:
        """
        Resolve the registry path using the Tier 2 hierarchy.

        Parameters
        ----------
        user_path : Path, optional
            Explicit path provided by the caller.

        Returns
        -------
        Path
            Resolved absolute Path to the registry JSON file.
        """
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "cochem_fit_registry.json"
        default = Path.home() / "CoChem_Artifacts" / "cochem_fit_registry.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        return default

    def __len__(self) -> int:
        """Return the number of commits in the DAG."""
        return len(self._states)

    def __getitem__(self, state_id: str) -> FitState:
        """Retrieve a state snapshot by state_id index."""
        return self.get_state(state_id)

    def __contains__(self, state_id: str) -> bool:
        """Check if a state_id exists in the DAG."""
        return state_id in self._states

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_state.py ---
"""
CoChem-SpycFit Physical Unit & DAG State Persistence Test Suite.

Strictly adheres to Zero-Mock mandate.
Validates:
1. FitState immutable Pydantic v2 data model with frozen ConfigDict.
2. Auto-computation of deterministic SHA-256 state_id and ISO timestamp.
3. CommitManager linear commit succession, historical DAG branching.
4. Leaf branch discovery across diverging fitting trajectories.
5. Atomic serialization and deserialization via temporary files.
6. Resolution of registry paths via explicit arguments and environment.
7. Authentic physical and spectroscopic invariants using H2O transitions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import mendeleev
import pytest
from pydantic import ValidationError

from cochem_spycfit.core_engine.cochem_spycfit_state import (
    CommitManager,
    FitState,
)

# Authentic experimental rotational constants for H2O in MHz
# (A, B, C ground-state constants and quartic distortion constants)
H2O_CONSTANTS: Dict[str, float] = {
    "A": 835840.288,
    "B": 435351.717,
    "C": 278138.700,
    "DJ": 37.594,
    "DJK": -172.5,
    "DK": 227.0,
}

# Authentic experimental microwave/sub-millimeter transitions for H2O (MHz)
# Format: (J_upper, Ka_upper, Kc_upper, J_lower, Ka_lower, Kc_lower, nu_exp)
H2O_TRANSITIONS: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (
    (1, 1, 0, 1, 0, 1, 556935.985),
    (2, 1, 1, 2, 0, 2, 752033.227),
    (3, 1, 2, 3, 0, 3, 1097364.791),
    (4, 1, 3, 4, 0, 4, 1602219.182),
    (5, 1, 4, 5, 0, 5, 2264148.770),
    (1, 1, 1, 0, 0, 0, 1113342.964),
    (2, 0, 2, 1, 1, 1, 987926.764),
    (2, 1, 2, 1, 0, 1, 1669904.775),
    (3, 0, 3, 2, 1, 2, 1716769.318),
    (3, 1, 3, 2, 0, 2, 2196345.756),
)


class TestFitStateModel:
    """Test suite for the immutable FitState Pydantic v2 data model."""

    def test_fit_state_creation_and_auto_computation(self) -> None:
        """Verify automatic generation of SHA-256 state_id and timestamp."""
        assignments = (H2O_TRANSITIONS[0], H2O_TRANSITIONS[1])
        state = FitState(
            locked_assignments=assignments,
            physical_constants=H2O_CONSTANTS,
            global_rmsd=0.0152,
            global_chi2=1.04,
            ml_weights_pointer="tensor_cache/gp_weights_h2o_001",
        )

        assert state.state_id != ""
        assert len(state.state_id) == 64
        assert state.timestamp != ""
        assert state.parent_id is None
        assert state.locked_assignments == assignments
        assert state.physical_constants == H2O_CONSTANTS
        assert state.global_rmsd == pytest.approx(0.0152)
        assert state.global_chi2 == pytest.approx(1.04)
        assert state.ml_weights_pointer == "tensor_cache/gp_weights_h2o_001"

    def test_fit_state_immutability(self) -> None:
        """Verify that FitState instances are immutable (frozen=True)."""
        state = FitState(
            locked_assignments=(H2O_TRANSITIONS[0],),
            physical_constants=H2O_CONSTANTS,
            global_rmsd=0.0120,
        )

        with pytest.raises(ValidationError):
            state.global_rmsd = 0.0500  # type: ignore[misc]

        with pytest.raises(ValidationError):
            state.state_id = "illegal_mutation"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            state.locked_assignments = ()  # type: ignore[misc]

    def test_fit_state_deterministic_hashing(self) -> None:
        """Verify that identical state content yields identical SHA-256."""
        assignments = (H2O_TRANSITIONS[0], H2O_TRANSITIONS[5])
        constants_1 = {
            "A": 835840.288,
            "B": 435351.717,
            "C": 278138.700,
        }
        constants_2 = {
            "C": 278138.700,
            "A": 835840.288,
            "B": 435351.717,
        }

        state_a = FitState(
            parent_id="parent_hash_123456",
            locked_assignments=assignments,
            physical_constants=constants_1,
            global_rmsd=0.018,
        )
        state_b = FitState(
            parent_id="parent_hash_123456",
            locked_assignments=assignments,
            physical_constants=constants_2,
            global_rmsd=0.045,
        )

        assert state_a.state_id == state_b.state_id

        # Different parent_id must change state_id
        state_c = FitState(
            parent_id="different_parent_hash",
            locked_assignments=assignments,
            physical_constants=constants_1,
            global_rmsd=0.018,
        )
        assert state_c.state_id != state_a.state_id

        # Different assignments must change state_id
        state_d = FitState(
            parent_id="parent_hash_123456",
            locked_assignments=(H2O_TRANSITIONS[0],),
            physical_constants=constants_1,
            global_rmsd=0.018,
        )
        assert state_d.state_id != state_a.state_id


class TestCommitManagerDAG:
    """Test suite for the Directed Acyclic Graph CommitManager."""

    def test_dag_50_commits_branching_and_reversion(self) -> None:
        """
        Execute full DAG workflow:
        1. Initialize CommitManager in ephemeral Tier 3 directory.
        2. Create root state + 50 sequential commits using H2O transitions.
        3. Branch off commit #25 with perturbed DJ parameter.
        4. Add 10 commits to the branch.
        5. Validate history length, branch leaves, and atomic round-trip.
        """
        with tempfile.TemporaryDirectory() as ephemeral_dir:
            registry_file = Path(ephemeral_dir) / "test_h2o_registry.json"
            manager = CommitManager(registry_path=registry_file)

            assert manager.registry_path == registry_file
            assert len(manager) == 0
            assert manager.head is None
            assert manager.get_branches() == []

            # Step 1: Root commit (initial baseline)
            root_state = manager.commit(
                locked_assignments=(H2O_TRANSITIONS[0],),
                physical_constants=H2O_CONSTANTS,
                global_rmsd=0.0100,
                global_chi2=1.00,
                ml_weights_pointer="gp_weights/root",
            )
            assert root_state.parent_id is None
            assert manager.head == root_state.state_id
            assert root_state.state_id in manager
            assert manager[root_state.state_id] == root_state

            # Step 2: 50 sequential commits on main trunk
            main_trunk_states: list[FitState] = [root_state]
            for commit_idx in range(1, 51):
                subset_len = (commit_idx % len(H2O_TRANSITIONS)) + 1
                assigned_subset = H2O_TRANSITIONS[:subset_len]
                rmsd_measured = 0.0100 + 0.0005 * (commit_idx % 10)
                chi2_measured = 0.9500 + 0.0100 * (commit_idx % 5)

                state = manager.commit(
                    locked_assignments=assigned_subset,
                    physical_constants=H2O_CONSTANTS,
                    global_rmsd=rmsd_measured,
                    global_chi2=chi2_measured,
                )
                assert state.parent_id == main_trunk_states[-1].state_id
                main_trunk_states.append(state)

            assert len(manager) == 51
            commit_25 = main_trunk_states[25]
            commit_50 = main_trunk_states[50]

            # Step 3: Verify history at commit #25 contains 26 states
            history_at_25 = manager.get_history(commit_25.state_id)
            assert len(history_at_25) == 26
            assert history_at_25[0].state_id == root_state.state_id
            assert history_at_25[-1].state_id == commit_25.state_id

            # Step 4: Branch off commit #25
            manager.branch_from(commit_25.state_id)
            assert manager.head == commit_25.state_id

            # Perturb DJ constant from 37.594 MHz to 40.0 MHz
            perturbed_constants = dict(H2O_CONSTANTS)
            perturbed_constants["DJ"] = 40.0

            # Step 5: Add 10 commits on the side branch
            branch_states: list[FitState] = []
            for branch_idx in range(1, 11):
                subset_len = ((branch_idx + 3) % len(H2O_TRANSITIONS)) + 1
                assigned_subset = H2O_TRANSITIONS[:subset_len]
                rmsd_branch = 0.0250 + 0.0010 * branch_idx
                chi2_branch = 1.1500 + 0.0200 * branch_idx

                b_state = manager.commit(
                    locked_assignments=assigned_subset,
                    physical_constants=perturbed_constants,
                    global_rmsd=rmsd_branch,
                    global_chi2=chi2_branch,
                )
                if branch_idx == 1:
                    assert b_state.parent_id == commit_25.state_id
                else:
                    assert b_state.parent_id == branch_states[-1].state_id
                branch_states.append(b_state)

            assert len(manager) == 61
            branch_head = branch_states[-1]

            # Step 6: Verify branch leaves (commit_50 and branch_head)
            leaf_nodes = manager.get_branches()
            assert len(leaf_nodes) == 2
            assert commit_50.state_id in leaf_nodes
            assert branch_head.state_id in leaf_nodes

            # Step 7: Verify ancestry on new branch (26 + 10 = 36)
            branch_history = manager.get_history(branch_head.state_id)
            assert len(branch_history) == 36
            assert branch_history[0].state_id == root_state.state_id
            assert branch_history[25].state_id == commit_25.state_id
            assert branch_history[-1].state_id == branch_head.state_id

            # Step 8: Atomic serialization and deserialization
            manager.save_registry()
            assert registry_file.exists()

            restored_manager = CommitManager(registry_path=registry_file)
            assert len(restored_manager) == 61
            assert restored_manager.head == manager.head
            assert set(restored_manager.get_branches()) == set(leaf_nodes)
            assert (
                len(restored_manager.get_history(commit_25.state_id)) == 26
            )

            # Verify identical state reconstruction across all 61 commits
            for state_id, original_state in manager.states.items():
                loaded_state = restored_manager.get_state(state_id)
                assert loaded_state.state_id == original_state.state_id
                assert loaded_state.parent_id == original_state.parent_id
                assert (
                    loaded_state.locked_assignments
                    == original_state.locked_assignments
                )
                assert (
                    loaded_state.physical_constants
                    == original_state.physical_constants
                )
                assert loaded_state.global_rmsd == pytest.approx(
                    original_state.global_rmsd
                )
                assert (
                    loaded_state.ml_weights_pointer
                    == original_state.ml_weights_pointer
                )
                if original_state.global_chi2 is not None:
                    assert loaded_state.global_chi2 == pytest.approx(
                        original_state.global_chi2
                    )
                else:
                    assert loaded_state.global_chi2 is None

    def test_environment_variable_path_resolution(self) -> None:
        """Verify resolution of COCHEM_ARTIFACTS_ROOT env var."""
        with tempfile.TemporaryDirectory() as env_dir:
            env_path = Path(env_dir)
            old_val = os.environ.get("COCHEM_ARTIFACTS_ROOT")
            try:
                os.environ["COCHEM_ARTIFACTS_ROOT"] = str(env_path)
                resolved_path = CommitManager._resolve_registry_path(None)
                assert resolved_path == env_path / "cochem_fit_registry.json"

                # Explicit path overrides environment variable
                explicit = env_path / "custom" / "registry.json"
                assert (
                    CommitManager._resolve_registry_path(explicit) == explicit
                )

                # Test default fallback when environment variable is unset
                os.environ.pop("COCHEM_ARTIFACTS_ROOT", None)
                default_path = CommitManager._resolve_registry_path(None)
                expected_default = (
                    Path.home()
                    / "CoChem_Artifacts"
                    / "cochem_fit_registry.json"
                )
                assert default_path == expected_default
            finally:
                if old_val is not None:
                    os.environ["COCHEM_ARTIFACTS_ROOT"] = old_val
                else:
                    os.environ.pop("COCHEM_ARTIFACTS_ROOT", None)

    def test_error_handling_and_boundary_cases(self) -> None:
        """Verify KeyError handling on non-existent state IDs."""
        with tempfile.TemporaryDirectory() as ephemeral_dir:
            registry_file = Path(ephemeral_dir) / "empty_registry.json"
            manager = CommitManager(registry_path=registry_file)

            with pytest.raises(KeyError):
                manager.branch_from("non_existent_state_hash")

            with pytest.raises(KeyError):
                manager.get_history("non_existent_state_hash")

            with pytest.raises(KeyError):
                manager.get_state("non_existent_state_hash")

            with pytest.raises(KeyError):
                _ = manager["non_existent_state_hash"]

            assert "non_existent_state_hash" not in manager
            assert manager.get_history() == []

    def test_registry_legacy_format_and_invalid_data_handling(self) -> None:
        """Verify parsing of flat state-id mapping and non-dict JSON."""
        with tempfile.TemporaryDirectory() as ephemeral_dir:
            # 1. Non-dict JSON payload
            corrupt_file = Path(ephemeral_dir) / "corrupt.json"
            corrupt_file.write_text(
                '["item1", "item2"]', encoding="utf-8"
            )
            manager_corrupt = CommitManager(registry_path=corrupt_file)
            assert len(manager_corrupt) == 0

            # 2. Flat dict JSON payload (state_id -> state dict)
            sample_state = FitState(
                locked_assignments=(H2O_TRANSITIONS[0],),
                physical_constants=H2O_CONSTANTS,
                global_rmsd=0.014,
            )
            flat_file = Path(ephemeral_dir) / "flat.json"
            flat_file.write_text(
                json.dumps({sample_state.state_id: sample_state.model_dump()}),
                encoding="utf-8",
            )
            manager_flat = CommitManager(registry_path=flat_file)
            assert len(manager_flat) == 1
            assert manager_flat.head == sample_state.state_id
            assert sample_state.state_id in manager_flat

    def test_mendeleev_elemental_mass_and_spectroscopic_invariants(
        self,
    ) -> None:
        """Verify Mendeleev elemental retrieval and physical invariants."""
        h_mass = mendeleev.element("H").mass
        o_mass = mendeleev.element("O").mass

        assert h_mass > 1.007
        assert o_mass > 15.998

        molecular_weight_h2o = 2.0 * h_mass + o_mass
        assert molecular_weight_h2o == pytest.approx(18.01528, abs=0.01)

        # Invariant for asymmetric top (H2O: A > B > C > 0)
        assert (
            H2O_CONSTANTS["A"]
            > H2O_CONSTANTS["B"]
            > H2O_CONSTANTS["C"]
            > 0
        )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.