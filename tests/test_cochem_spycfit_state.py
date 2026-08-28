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
