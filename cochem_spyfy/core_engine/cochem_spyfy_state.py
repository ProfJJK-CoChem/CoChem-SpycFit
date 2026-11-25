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
