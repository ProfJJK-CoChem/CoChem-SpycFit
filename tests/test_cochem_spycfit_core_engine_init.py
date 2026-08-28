"""
CoChem-SpycFit Core Engine Package Init Update & Re-export Zero-Mock Unit Test Suite.

Strictly adheres to Zero-Mock mandate and Mendeleev mass mandate.
Validates:
1. Clean importability of FitState, CommitManager, and TensorCache from cochem_spycfit.core_engine.
2. Preservation of all legacy and existing exports in __all__.
3. Dual-tree parity between root package proxy and src/ distribution hierarchy.
4. No circular dependency loops during fresh module resolution.
5. Dynamic elemental mass retrieval from Mendeleev and authentic physical operations.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import mendeleev
import numpy as np
import pytest

# Direct root-level and package-level imports
from cochem_spycfit.core_engine import (
    CANONICAL_BINARY_HASHES,
    CommitManager,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOCAL_LOCK_TIMEOUT,
    FitParameterRecord,
    FitRegistryManager,
    FitState,
    HardwareGateResult,
    TensorCache,
)

# Authentic experimental rotational constants for H2O in MHz
H2O_CONSTANTS: Dict[str, float] = {
    "A": 835840.288,
    "B": 435351.717,
    "C": 278138.700,
    "DJ": 37.594,
    "DJK": -172.5,
    "DK": 227.0,
}

# Authentic experimental microwave transitions for H2O (MHz)
H2O_TRANSITIONS: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (
    (1, 1, 0, 1, 0, 1, 556935.985),
    (2, 1, 1, 2, 0, 2, 752033.227),
)


class TestCoreEngineInitExports:
    """Test suite for core_engine package exports, clean imports, and circularity."""

    def test_direct_imports_and_types(self) -> None:
        """Verify FitState, CommitManager, and TensorCache are directly importable."""
        import cochem_spycfit.core_engine as core_mod

        assert hasattr(core_mod, "FitState")
        assert hasattr(core_mod, "CommitManager")
        assert hasattr(core_mod, "TensorCache")

        assert issubclass(core_mod.FitState, object)
        assert issubclass(core_mod.CommitManager, object)
        assert issubclass(core_mod.TensorCache, object)

        # Check __all__ exports
        assert "FitState" in core_mod.__all__
        assert "CommitManager" in core_mod.__all__
        assert "TensorCache" in core_mod.__all__

    def test_existing_exports_preserved(self) -> None:
        """Verify existing exports remain accessible and intact in __all__."""
        import cochem_spycfit.core_engine as core_mod

        expected_legacy_exports = [
            "CANONICAL_BINARY_HASHES",
            "DEFAULT_CHUNK_SIZE",
            "DEFAULT_LOCAL_LOCK_TIMEOUT",
            "DEFAULT_LUSTRE_LOCK_TIMEOUT",
            "FitParameterRecord",
            "FitRegistryManager",
            "FitRegistryState",
            "HardwareGateResult",
            "InstrumentLimits",
            "StandardFileLock",
            "calculate_binary_sha256",
            "get_atomic_mass",
        ]

        for export_name in expected_legacy_exports:
            assert hasattr(core_mod, export_name), f"Missing export: {export_name}"
            assert export_name in core_mod.__all__, f"Export not listed in __all__: {export_name}"

    def test_functional_smoke_with_imported_classes(self) -> None:
        """Verify functional interaction among imported FitState, CommitManager, and TensorCache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "registry.json"
            cache_file = Path(temp_dir) / "tensor_cache.h5"

            manager = CommitManager(registry_path=registry_file)
            cache = TensorCache(cache_path=cache_file)

            # Create state
            state = manager.commit(
                locked_assignments=H2O_TRANSITIONS,
                physical_constants=H2O_CONSTANTS,
                global_rmsd=0.0125,
                global_chi2=1.02,
                ml_weights_pointer="weights/h2o_001",
            )
            assert isinstance(state, FitState)
            assert state.state_id in manager

            # Store in TensorCache
            cov_matrix = np.diag([0.050, 0.030, 0.020, 0.005, 0.010, 0.015]).astype(np.float64)
            gp_weights = np.array([0.012, 0.018, 0.009, 0.025], dtype=np.float64)

            cache.store_state_tensors(
                state_id=state.state_id,
                covariance=cov_matrix,
                gp_weights=gp_weights,
            )

            # Restore and verify round-trip
            restored = cache.restore_state_tensors(state.state_id)
            assert np.allclose(restored["covariance"], cov_matrix, rtol=1e-15, atol=1e-15)
            assert np.allclose(restored["gp_weights"], gp_weights, rtol=1e-15, atol=1e-15)

    def test_src_and_root_parity(self) -> None:
        """Verify exact export equivalence between src/ and root proxy core_engine."""
        import cochem_spycfit.core_engine as root_engine
        import src.cochem_spycfit.core_engine as src_engine

        root_all = set(root_engine.__all__)
        src_all = set(src_engine.__all__)

        assert "FitState" in root_all and "FitState" in src_all
        assert "CommitManager" in root_all and "CommitManager" in src_all
        assert "TensorCache" in root_all and "TensorCache" in src_all
        assert root_all == src_all

    def test_no_circular_imports_in_fresh_process(self) -> None:
        """Verify importing core_engine in an isolated Python process produces zero circular errors."""
        script = (
            "import sys; "
            "from cochem_spycfit.core_engine import FitState, CommitManager, TensorCache; "
            "assert FitState is not None; "
            "assert CommitManager is not None; "
            "assert TensorCache is not None; "
            "print('SUCCESS')"
        )
        repo_root = str(Path(__file__).resolve().parent.parent)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Subprocess import failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_mendeleev_mass_mandate(self) -> None:
        """Verify dynamic mass retrieval for H2O constituents via mendeleev."""
        h_mass = float(mendeleev.element("H").mass)
        o_mass = float(mendeleev.element("O").mass)

        assert h_mass > 1.007
        assert o_mass > 15.998
        assert abs(2.0 * h_mass + o_mass - 18.01528) < 0.01
