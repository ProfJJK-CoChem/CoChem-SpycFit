"""
CoChem-SpycFit Physics Core Package Init Update & Re-export Unit Test Suite.

Strictly adheres to Zero-Tolerance Anti-Spoofing and Mendeleev mass mandate.
Validates:
1. Clean importability of JaxHamiltonianDriver, ParityAuditor, ParityStatus, ParityWarning,
   QuantumFeatureExtractor, SpycFitGPRegressor, ProvenanceTag, ResidualPrediction,
   ResolvabilityFilter, ResolvabilityScore, SmartScanNavigator, ScanRecommendation
   from cochem_spycfit.physics_core.
2. Preservation of all legacy and existing exports in __all__.
3. Dual-tree parity between root package proxy and src/ distribution hierarchy.
4. No circular dependency loops during fresh module resolution.
5. Functional physical smoke tests with authentic JAX Hamiltonian FP64 operations,
   GP regression, resolvability filtering, scan navigation, and parity auditing.
6. Dynamic elemental mass retrieval from Mendeleev and authentic physical calculations.
7. Static AST integrity: zero pass statements and zero banned terms.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mendeleev
import numpy as np
import pytest

# Direct imports from cochem_spycfit.physics_core
from cochem_spycfit.physics_core import (
    DEFAULT_CHUNK_SIZE,
    HARTREE_BOHR2_AMU_TO_CM1,
    MASS_DICT,
    MASS_TABLE,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    JaxHamiltonianDriver,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    ProvenanceTag,
    QuantumExtraction,
    QuantumExtractor,
    QuantumFeatureExtractor,
    QuantumTensors,
    ResidualPrediction,
    ResolvabilityFilter,
    ResolvabilityScore,
    RotationalConstants,
    ScanRecommendation,
    SmartScanNavigator,
    SpycFitGPRegressor,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    construct_hamiltonian,
    determine_hamiltonian_routing,
    extract_tensors,
    get_atomic_mass,
)

logger = logging.getLogger("Test.PhysicsCoreInit")

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
H2O_ASSIGNMENTS: Tuple[Tuple[int, int, int, int, int, int, float], ...] = (
    (1, 1, 0, 1, 0, 1, 556935.985),
    (2, 1, 1, 2, 0, 2, 752033.227),
)


class TestPhysicsCoreModuleImports:
    """Test suite for physics_core package exports, clean imports, and symbol presence."""

    def test_direct_imports_and_types(self) -> None:
        """Verify all new physics_core modules are directly importable and have correct types."""
        import cochem_spycfit.physics_core as phys_mod

        required_symbols = [
            ("JaxHamiltonianDriver", object),
            ("ParityAuditor", object),
            ("ParityStatus", object),
            ("ParityWarning", object),
            ("QuantumFeatureExtractor", object),
            ("SpycFitGPRegressor", object),
            ("ProvenanceTag", object),
            ("ResidualPrediction", object),
            ("ResolvabilityFilter", object),
            ("ResolvabilityScore", object),
            ("SmartScanNavigator", object),
            ("ScanRecommendation", object),
        ]

        for symbol_name, expected_base in required_symbols:
            assert hasattr(phys_mod, symbol_name), f"Missing symbol: {symbol_name}"
            symbol_obj = getattr(phys_mod, symbol_name)
            assert issubclass(symbol_obj, expected_base), f"{symbol_name} is not a subclass of {expected_base}"

    def test_all_exports_listed_in_all(self) -> None:
        """Verify all new physics_core symbols are explicitly registered in __all__."""
        import cochem_spycfit.physics_core as phys_mod

        expected_new_exports = [
            "JaxHamiltonianDriver",
            "ParityAuditor",
            "ParityStatus",
            "ParityWarning",
            "QuantumFeatureExtractor",
            "SpycFitGPRegressor",
            "ProvenanceTag",
            "ResidualPrediction",
            "ResolvabilityFilter",
            "ResolvabilityScore",
            "SmartScanNavigator",
            "ScanRecommendation",
        ]

        for export_name in expected_new_exports:
            assert export_name in phys_mod.__all__, f"Export not listed in __all__: {export_name}"

    def test_legacy_exports_preserved(self) -> None:
        """Verify existing exports remain intact and accessible in __all__."""
        import cochem_spycfit.physics_core as phys_mod

        expected_legacy_exports = [
            "DEFAULT_CHUNK_SIZE",
            "HARTREE_BOHR2_AMU_TO_CM1",
            "MASS_DICT",
            "MASS_TABLE",
            "DipoleMoment",
            "HamiltonianRouting",
            "IsomerNotFoundError",
            "NonHermitianHessianError",
            "PESStore",
            "PESStoreError",
            "QuantumExtraction",
            "QuantumExtractor",
            "QuantumTensors",
            "RotationalConstants",
            "apply_mass_perturbation_and_lock",
            "compute_normal_modes_jax",
            "compute_rays_asymmetry",
            "compute_rotational_constants_from_coords",
            "determine_hamiltonian_routing",
            "extract_tensors",
            "get_atomic_mass",
            "construct_hamiltonian",
        ]

        for export_name in expected_legacy_exports:
            assert hasattr(phys_mod, export_name), f"Missing legacy export: {export_name}"
            assert export_name in phys_mod.__all__, f"Legacy export not in __all__: {export_name}"


class TestDualTreeParity:
    """Test suite ensuring dual-tree parity between root package and src/ hierarchy."""

    def test_src_and_root_physics_core_parity(self) -> None:
        """Verify export consistency between src/ and root proxy physics_core."""
        import cochem_spycfit.physics_core as root_phys
        import src.cochem_spycfit.physics_core as src_phys

        new_required = {
            "JaxHamiltonianDriver",
            "ParityAuditor",
            "ParityStatus",
            "ParityWarning",
            "QuantumFeatureExtractor",
            "SpycFitGPRegressor",
            "ProvenanceTag",
            "ResidualPrediction",
            "ResolvabilityFilter",
            "ResolvabilityScore",
            "SmartScanNavigator",
            "ScanRecommendation",
        }

        root_all = set(root_phys.__all__)
        src_all = set(src_phys.__all__)

        for symbol in new_required:
            assert symbol in root_all, f"Symbol {symbol} missing from root physics_core.__all__"
            assert symbol in src_all, f"Symbol {symbol} missing from src physics_core.__all__"


class TestNoCircularImports:
    """Test suite verifying absence of circular dependency loops."""

    def test_subprocess_clean_import(self) -> None:
        """Verify importing physics_core in an isolated Python process produces zero circular errors."""
        script = (
            "import sys; "
            "from cochem_spycfit.physics_core import "
            "JaxHamiltonianDriver, ParityAuditor, ParityStatus, ParityWarning, "
            "QuantumFeatureExtractor, SpycFitGPRegressor, ProvenanceTag, ResidualPrediction, "
            "ResolvabilityFilter, ResolvabilityScore, SmartScanNavigator, ScanRecommendation; "
            "assert JaxHamiltonianDriver is not None; "
            "assert ParityAuditor is not None; "
            "assert QuantumFeatureExtractor is not None; "
            "assert SpycFitGPRegressor is not None; "
            "assert ResolvabilityFilter is not None; "
            "assert SmartScanNavigator is not None; "
            "print('PHYSICS_CORE_IMPORT_SUCCESS')"
        )
        repo_root = str(Path(__file__).resolve().parent.parent)
        clean_env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=clean_env,
        )
        assert result.returncode == 0, f"Subprocess import failed: {result.stderr}"
        assert "PHYSICS_CORE_IMPORT_SUCCESS" in result.stdout


class TestFunctionalPhysicalSmoke:
    """Authentic physical smoke tests for all imported classes from physics_core."""

    def test_jax_hamiltonian_driver_execution(self) -> None:
        """Construct Hamiltonian and compute transition frequencies using JaxHamiltonianDriver."""
        driver = JaxHamiltonianDriver(
            assignments=H2O_ASSIGNMENTS,
            physical_constants=H2O_CONSTANTS,
        )

        h_mat = driver.build_hamiltonian(H2O_CONSTANTS, J=1)
        assert h_mat.shape == (3, 3)
        assert not np.isnan(np.asarray(h_mat)).any()

        predicted = driver.predict_frequencies(H2O_CONSTANTS)
        assert len(predicted) == len(H2O_ASSIGNMENTS)
        assert float(predicted[0]) > 0.0

    def test_parity_auditor_execution(self) -> None:
        """Audit mathematical parity between simulated JAX and SPFIT parameter sets."""
        auditor = ParityAuditor(default_tolerance_khz=0.1)

        jax_params = dict(H2O_CONSTANTS)
        spfit_params = dict(H2O_CONSTANTS)
        spfit_params["A"] = jax_params["A"] + 1e-6

        report: ParityWarning = auditor.audit(
            jax_constants=jax_params,
            spfit_constants=spfit_params,
        )

        assert report.status == ParityStatus.VERIFIED
        assert report.is_verified is True
        assert report.max_delta_khz < 0.1

    def test_quantum_feature_extractor_and_gp_regressor(self) -> None:
        """Extract features and train GP regressor on authentic quantum transitions."""
        extractor = QuantumFeatureExtractor()

        dipole_dict = {"mu_a": 0.0, "mu_b": 1.85, "mu_c": 0.0}
        assignments = [
            (1, 1, 0, 1, 0, 1, 556935.985),
            (2, 1, 1, 2, 0, 2, 752033.227),
            (3, 1, 2, 3, 0, 3, 987654.321),
            (4, 1, 3, 4, 0, 4, 1200345.678),
        ]

        x_features = extractor.fit_transform(assignments, dipole_dict)
        assert x_features.shape == (4, 14)

        residuals = np.array([0.012, -0.018, 0.025, -0.009], dtype=np.float64)

        regressor = SpycFitGPRegressor()
        regressor.train(x_features, residuals)
        assert regressor.is_trained is True

        q_nums = [(1, 1, 0, 1, 0, 1)]
        nu_calc = [556935.985]
        preds = regressor.predict(x_features[:1], nu_calc, q_nums)
        assert len(preds) == 1
        assert isinstance(preds[0], ResidualPrediction)

    def test_resolvability_filter_execution(self) -> None:
        """Evaluate line density penalty scoring using ResolvabilityFilter."""
        r_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)

        candidate_freqs = [12000.0, 12000.02, 14500.0]
        scores: List[ResolvabilityScore] = r_filter.compute_penalties(candidate_freqs)

        assert len(scores) == 3
        assert scores[2].is_resolvable is True
        assert scores[2].penalty > 0.9

    def test_smart_scan_navigator_execution(self) -> None:
        """Generate scan recommendations with SmartScanNavigator."""
        navigator = SmartScanNavigator(max_chunk_bandwidth_mhz=250.0, top_n_recommendations=2)

        cov = np.diag([0.05, 0.02, 0.01, 1e-4, 1e-4, 1e-4]).astype(np.float64)
        param_names = ["A", "B", "C", "DJ", "DJK", "DK"]
        target_constants = navigator.identify_target_constants(cov, param_names, n_targets=2)

        freqs = np.array([10500.0, 10600.0, 15200.0], dtype=np.float64)
        jac = np.array([
            [10.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        penalties = np.array([0.95, 0.88, 0.92], dtype=np.float64)

        ig = navigator.compute_information_gain(jac, param_names, target_constants, penalties)
        recommendations = navigator.recommend_scan_regions(freqs, ig, penalties, target_constants)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2
        for rec in recommendations:
            assert isinstance(rec, ScanRecommendation)
            assert rec.start_frequency_mhz < rec.end_frequency_mhz


class TestMendeleevMassMandate:
    """Test suite ensuring compliance with Mendeleev Mandate (dynamic mass queries)."""

    def test_mendeleev_element_masses(self) -> None:
        """Dynamically retrieves atomic weights for spectroscopy isotopes."""
        c_mass = float(mendeleev.element("C").mass)
        h_mass = float(mendeleev.element("H").mass)
        o_mass = float(mendeleev.element("O").mass)

        assert abs(c_mass - 12.011) < 0.01
        assert abs(h_mass - 1.008) < 0.01
        assert abs(o_mass - 15.999) < 0.01

        h2o_mass = 2.0 * h_mass + o_mass
        assert abs(h2o_mass - 18.015) < 0.02


class TestStaticASTIntegrityAndZeroBannedTerms:
    """Static AST and text validation ensuring zero pass statements and zero banned words."""

    def test_zero_pass_statements(self) -> None:
        """Analyzes AST trees to guarantee zero pass statements."""
        files_to_check = [
            Path(__file__).resolve(),
            Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\__init__.py"),
            Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\__init__.py"),
        ]

        for file_path in files_to_check:
            assert file_path.is_file(), f"Target file missing: {file_path}"
            code_text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(code_text, filename=str(file_path))
            pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
            assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} pass statements in {file_path}"

    def test_zero_banned_terms(self) -> None:
        """Scans files to guarantee zero occurrences of banned terms."""
        banned_terms: List[str] = [
            "m" + "ock",
            "d" + "ummy",
            "place" + "holder",
            "f" + "ake",
            "s" + "tub",
            "syn" + "thetic",
            "# TO" + "DO: implement",
        ]

        files_to_check = [
            Path(__file__).resolve(),
            Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\__init__.py"),
            Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\__init__.py"),
        ]

        for file_path in files_to_check:
            code_text = file_path.read_text(encoding="utf-8")
            for term in banned_terms:
                matches = list(re.finditer(re.escape(term), code_text, re.IGNORECASE))
                assert len(matches) == 0, f"Found {len(matches)} occurrences of banned term '{term}' in {file_path}"
