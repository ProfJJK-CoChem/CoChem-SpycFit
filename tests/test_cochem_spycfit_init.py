"""
Physical unit tests for CoChem-SpycFit Stage 0 Hardware & Precision Gatekeeper.

Validates:
1. JAX FP64 precision lock and physical float64 array allocations.
2. XLA compilation cache path routing and directory mutation.
3. Pre-Compute VRAM Sharding Gate with persistent fallback and sequential mini-batching flags.
4. Legacy SPFIT and SPCAT Fortran binary cryptographic hash validation.
5. 6-Tier Matrix OS/Arch binary routing.
6. Stage 0 initialization pipeline and state JSON serialization.
7. Dynamic atomic and isotopic mass resolution via mendeleev.
8. Static AST integrity: zero pass statements and zero banned terms.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import mendeleev
import platformdirs
import pytest

from cochem_spycfit.core_engine.cochem_spycfit_init import (
    CANONICAL_BINARY_HASHES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LEGACY_HASHES,
    FALLBACK_REQUIRED,
    MIN_GPU_VRAM_GB,
    SEQUENTIAL_MINIBATCHING,
    TIER_BINARY_MAP,
    BinaryRouteResult,
    BinaryVerificationResult,
    CoChemIntegrityError,
    CoChemPrecisionError,
    HardwareDetectionError,
    HardwareGateResult,
    HardwareWarning,
    JAXPrecisionConfig,
    OSTier,
    SpycfitGatekeeperState,
    XLACacheConfig,
    calculate_binary_sha256,
    check_vram_and_set_flags,
    configure_xla_cache,
    enforce_jax_precision,
    evaluate_hardware_fallback_gate,
    get_atomic_mass,
    initialize_spycfit_gatekeeper,
    is_fallback_required,
    is_sequential_minibatching,
    query_physical_gpu_vram,
    resolve_cochem_system_config_path,
    route_6tier_fortran_binary,
    validate_legacy_hashes,
    verify_binary_checksum,
)

logger = logging.getLogger("Test.Gatekeeper")


class TestJAXPrecisionLock:
    """Test suite for JAX FP64 precision locking and numerical verification."""

    def test_jax_fp64_environment_variable(self) -> None:
        """Verifies that JAX_ENABLE_X64 environment variable is active."""
        assert os.environ.get("JAX_ENABLE_X64") == "True"

    def test_enforce_jax_precision_execution(self) -> None:
        """Executes enforce_jax_precision and verifies the configuration model."""
        config: JAXPrecisionConfig = enforce_jax_precision()
        assert config.jax_enable_x64 is True
        assert config.precision == "float64"
        assert config.verified is True
        assert config.test_array_dtype == "float64"
        assert config.device_count >= 1
        assert len(config.backend) > 0

    def test_physical_float64_array_instantiation(self) -> None:
        """Allocates real JAX numerical arrays and checks float64 precision."""
        vector = jnp.array([1.234567890123456, 9.876543210987654])
        assert vector.dtype == jnp.float64

        matrix = jnp.array([[2.5, 1.1], [1.1, 3.7]])
        assert matrix.dtype == jnp.float64

        eigenvalues = jnp.linalg.eigvalsh(matrix)
        assert eigenvalues.dtype == jnp.float64
        assert eigenvalues.shape == (2,)


class TestXLACacheConfiguration:
    """Test suite for XLA bytecode compilation caching."""

    def test_default_xla_cache_routing(self) -> None:
        """Verifies default cache resolution to platformdirs user data path."""
        config: XLACacheConfig = configure_xla_cache()
        expected_base = platformdirs.user_data_path("CoChem_Artifacts")
        expected_path = (expected_base / "SpycFit_Workspace" / "Scratch" / ".jax_xla_cache").resolve()

        assert config.enabled is True
        assert config.directory_created is True
        assert Path(config.cache_dir).resolve() == expected_path
        assert Path(config.cache_dir).is_dir()
        assert "--xla_gpu_enable_compilation_cache=true" in os.environ.get("XLA_FLAGS", "")
        assert f"--xla_gpu_compilation_cache_dir={expected_path.as_posix()}" in os.environ.get("XLA_FLAGS", "")

    def test_custom_xla_cache_routing(self, tmp_path: Path) -> None:
        """Verifies custom cache directory parameter creation and environment mutation."""
        custom_dir = (tmp_path / "custom_xla_cache_dir").resolve()
        config: XLACacheConfig = configure_xla_cache(custom_cache_dir=custom_dir)

        assert config.enabled is True
        assert config.directory_created is True
        assert Path(config.cache_dir).resolve() == custom_dir
        assert custom_dir.is_dir()
        assert f"--xla_gpu_compilation_cache_dir={custom_dir.as_posix()}" in os.environ.get("XLA_FLAGS", "")

    def test_env_override_xla_cache_routing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies COCHEM_XLA_CACHE_DIR environment override."""
        env_dir = (tmp_path / "env_xla_cache_dir").resolve()
        monkeypatch.setenv("COCHEM_XLA_CACHE_DIR", str(env_dir))

        config: XLACacheConfig = configure_xla_cache()
        assert Path(config.cache_dir).resolve() == env_dir
        assert env_dir.is_dir()
        assert f"--xla_gpu_compilation_cache_dir={env_dir.as_posix()}" in os.environ.get("XLA_FLAGS", "")


class TestHardwareFallbackGateAndVRAM:
    """Test suite for Pre-Compute VRAM Sharding Gate and hardware fallback rules."""

    def test_physical_gpu_query(self) -> None:
        """Queries physical GPU topology on the executing machine."""
        gpu_detected, gpu_profile, vram_gb, device_count = query_physical_gpu_vram()
        assert isinstance(gpu_detected, bool)
        assert isinstance(gpu_profile, str)
        assert isinstance(vram_gb, float)
        assert isinstance(device_count, int)
        assert vram_gb >= 0.0
        assert device_count >= 0

    def test_gate_low_vram_triggers_warnings_and_fallback_flags(self) -> None:
        """Verifies that VRAM < 8.0 GB sets fallback and sequential mini-batching flags."""
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            gate_result: HardwareGateResult = check_vram_and_set_flags(
                override_vram_gb=4.0,
                override_gpu_detected=True,
                override_cuda_available=True,
            )

            assert gate_result.fallback_required is True
            assert gate_result.sequential_minibatching is True
            assert gate_result.warning_issued is True
            assert is_fallback_required() is True
            assert is_sequential_minibatching() is True

            warning_types = [w.category for w in recorded_warnings]
            assert UserWarning in warning_types or HardwareWarning in warning_types

    def test_gate_no_cuda_triggers_warnings_and_fallback_flags(self) -> None:
        """Verifies that lack of CUDA acceleration triggers fallback flags."""
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            gate_result: HardwareGateResult = check_vram_and_set_flags(
                override_vram_gb=12.0,
                override_gpu_detected=True,
                override_cuda_available=False,
            )

            assert gate_result.fallback_required is True
            assert gate_result.sequential_minibatching is True
            assert gate_result.warning_issued is True
            assert is_fallback_required() is True
            assert is_sequential_minibatching() is True
            assert len(recorded_warnings) >= 1

    def test_gate_sufficient_vram_and_cuda_active(self) -> None:
        """Verifies that VRAM >= 8.0 GB with CUDA clears fallback requirements."""
        gate_result: HardwareGateResult = check_vram_and_set_flags(
            override_vram_gb=16.0,
            override_gpu_detected=True,
            override_cuda_available=True,
        )

        assert gate_result.fallback_required is False
        assert gate_result.sequential_minibatching is False
        assert gate_result.warning_issued is False
        assert is_fallback_required() is False
        assert is_sequential_minibatching() is False

    def test_evaluate_hardware_fallback_gate_delegation(self) -> None:
        """Verifies evaluate_hardware_fallback_gate behaves identically."""
        gate_result: HardwareGateResult = evaluate_hardware_fallback_gate(
            override_vram_gb=2.0,
            override_gpu_detected=True,
            override_cuda_available=True,
        )
        assert gate_result.fallback_required is True
        assert gate_result.sequential_minibatching is True

    def test_hardware_gate_with_physical_config_file(self, tmp_path: Path) -> None:
        """Verifies hardware gate reading from a physical configuration JSON file."""
        cfg_file = tmp_path / "cochem_system_config.json"
        config_data = {
            "hardware": {
                "vram_gb": 24.0,
                "gpu_profile": "NVIDIA RTX 4090",
            }
        }
        cfg_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        gate_result: HardwareGateResult = check_vram_and_set_flags(
            custom_config_path=cfg_file,
            override_cuda_available=True,
        )

        assert gate_result.vram_gb == 24.0
        assert gate_result.gpu_profile == "NVIDIA RTX 4090"
        assert gate_result.fallback_required is False
        assert gate_result.config_source == str(cfg_file.resolve())


class TestLegacyBinaryValidation:
    """Test suite for SHA-256 cryptographic verification of native and legacy binaries."""

    def test_physical_sha256_calculation(self, tmp_path: Path) -> None:
        """Computes SHA-256 hash of a real file on disk."""
        test_file = tmp_path / "binary_test_payload.bin"
        payload_bytes = b"Physical Fortran binary byte stream content for checksum validation."
        test_file.write_bytes(payload_bytes)

        computed_sha = calculate_binary_sha256(test_file, chunk_size=32)
        expected_sha = hashlib.sha256(payload_bytes).hexdigest()
        assert computed_sha == expected_sha

    def test_verify_binary_checksum_success(self, tmp_path: Path) -> None:
        """Validates successful verification against matching SHA-256 checksum."""
        test_file = tmp_path / "valid_engine.exe"
        payload_bytes = b"Authoritative native binary compiled stream."
        test_file.write_bytes(payload_bytes)

        actual_sha = hashlib.sha256(payload_bytes).hexdigest()
        verification: BinaryVerificationResult = verify_binary_checksum(
            binary_path=test_file,
            expected_sha256=actual_sha,
            tier=OSTier.WINDOWS_X86_64,
        )

        assert verification.verified is True
        assert verification.sha256 == actual_sha
        assert verification.expected_sha256 == actual_sha
        assert verification.tier == OSTier.WINDOWS_X86_64
        assert verification.file_size_bytes == len(payload_bytes)

    def test_verify_binary_checksum_integrity_error(self, tmp_path: Path) -> None:
        """Verifies CoChemIntegrityError is raised when hash mismatches."""
        test_file = tmp_path / "corrupted_engine.exe"
        test_file.write_bytes(b"Corrupted binary stream.")

        mismatched_sha = "0" * 64
        with pytest.raises(CoChemIntegrityError):
            verify_binary_checksum(
                binary_path=test_file,
                expected_sha256=mismatched_sha,
            )

    def test_validate_legacy_hashes_success(self, tmp_path: Path) -> None:
        """Validates validate_legacy_hashes with physical files and matching safe hashes."""
        spfit_bin = tmp_path / "spfit"
        spcat_bin = tmp_path / "spcat"

        spfit_payload = b"SPFIT Fortran Core Subroutine V7.1"
        spcat_payload = b"SPCAT Fortran Core Subroutine V7.1"

        spfit_bin.write_bytes(spfit_payload)
        spcat_bin.write_bytes(spcat_payload)

        spfit_sha = hashlib.sha256(spfit_payload).hexdigest()
        spcat_sha = hashlib.sha256(spcat_payload).hexdigest()

        known_registry: Dict[str, str] = {
            "spfit": spfit_sha,
            "spcat": spcat_sha,
        }

        results = validate_legacy_hashes(
            spfit_path=spfit_bin,
            spcat_path=spcat_bin,
            known_hashes=known_registry,
            raise_on_error=True,
        )

        assert "spfit" in results
        assert "spcat" in results
        assert results["spfit"].verified is True
        assert results["spfit"].sha256 == spfit_sha
        assert results["spcat"].verified is True
        assert results["spcat"].sha256 == spcat_sha

    def test_validate_legacy_hashes_mismatch_raise(self, tmp_path: Path) -> None:
        """Validates CoChemIntegrityError is raised on checksum mismatch when raise_on_error=True."""
        spfit_bin = tmp_path / "spfit"
        spcat_bin = tmp_path / "spcat"

        spfit_bin.write_bytes(b"SPFIT Payload")
        spcat_bin.write_bytes(b"SPCAT Payload")

        known_registry: Dict[str, str] = {
            "spfit": "a" * 64,
            "spcat": "b" * 64,
        }

        with pytest.raises(CoChemIntegrityError):
            validate_legacy_hashes(
                spfit_path=spfit_bin,
                spcat_path=spcat_bin,
                known_hashes=known_registry,
                raise_on_error=True,
            )

    def test_validate_legacy_hashes_mismatch_no_raise(self, tmp_path: Path) -> None:
        """Validates results record failure without raising when raise_on_error=False."""
        spfit_bin = tmp_path / "spfit"
        spcat_bin = tmp_path / "spcat"

        spfit_bin.write_bytes(b"SPFIT Payload")
        spcat_bin.write_bytes(b"SPCAT Payload")

        known_registry: Dict[str, str] = {
            "spfit": "a" * 64,
            "spcat": "b" * 64,
        }

        results = validate_legacy_hashes(
            spfit_path=spfit_bin,
            spcat_path=spcat_bin,
            known_hashes=known_registry,
            raise_on_error=False,
        )

        assert results["spfit"].verified is False
        assert results["spcat"].verified is False

    def test_validate_legacy_hashes_missing_file_raise(self, tmp_path: Path) -> None:
        """Validates FileNotFoundError when binary files are absent and raise_on_error=True."""
        spfit_bin = tmp_path / "non_existent_spfit"
        spcat_bin = tmp_path / "non_existent_spcat"

        with pytest.raises(FileNotFoundError):
            validate_legacy_hashes(
                spfit_path=spfit_bin,
                spcat_path=spcat_bin,
                raise_on_error=True,
            )

    def test_validate_legacy_hashes_missing_file_no_raise(self, tmp_path: Path) -> None:
        """Validates results record failure for absent files when raise_on_error=False."""
        spfit_bin = tmp_path / "missing_spfit"
        spcat_bin = tmp_path / "missing_spcat"

        results = validate_legacy_hashes(
            spfit_path=spfit_bin,
            spcat_path=spcat_bin,
            raise_on_error=False,
        )

        assert results["spfit"].verified is False
        assert results["spcat"].verified is False


class TestSixTierBinaryRouter:
    """Test suite for the 6-Tier Operating System and Architecture Router."""

    def test_six_tier_explicit_targets(self) -> None:
        """Verifies explicit target strings route to the canonical 6-tier matrix."""
        tier1 = route_6tier_fortran_binary(os_target="windows_x86_64")
        assert tier1.tier == OSTier.WINDOWS_X86_64
        assert tier1.binary_name == "spycfit_engine_win_x64.exe"

        tier2 = route_6tier_fortran_binary(os_target="linux_x86_64")
        assert tier2.tier == OSTier.LINUX_X86_64
        assert tier2.binary_name == "spycfit_engine_linux_x86_64"

        tier3 = route_6tier_fortran_binary(os_target="linux_arm64")
        assert tier3.tier == OSTier.LINUX_ARM64
        assert tier3.binary_name == "spycfit_engine_linux_arm64"

        tier4 = route_6tier_fortran_binary(os_target="darwin_arm64")
        assert tier4.tier == OSTier.DARWIN_ARM64
        assert tier4.binary_name == "spycfit_engine_darwin_arm64"

        tier5 = route_6tier_fortran_binary(os_target="darwin_x86_64")
        assert tier5.tier == OSTier.DARWIN_X86_64
        assert tier5.binary_name == "spycfit_engine_darwin_x86_64"

        tier6 = route_6tier_fortran_binary(os_target="hpc_container")
        assert tier6.tier == OSTier.HPC_CONTAINER
        assert tier6.binary_name == "spycfit_engine_hpc_posix"

    def test_six_tier_platform_inference(self) -> None:
        """Verifies OS name and machine architecture inference across the matrix."""
        win_res = route_6tier_fortran_binary(system="Windows", machine="AMD64")
        assert win_res.tier == OSTier.WINDOWS_X86_64

        linux_x64_res = route_6tier_fortran_binary(system="Linux", machine="x86_64")
        assert linux_x64_res.tier == OSTier.LINUX_X86_64

        linux_arm_res = route_6tier_fortran_binary(system="Linux", machine="aarch64")
        assert linux_arm_res.tier == OSTier.LINUX_ARM64

        darwin_arm_res = route_6tier_fortran_binary(system="Darwin", machine="arm64")
        assert darwin_arm_res.tier == OSTier.DARWIN_ARM64

        darwin_x64_res = route_6tier_fortran_binary(system="Darwin", machine="x86_64")
        assert darwin_x64_res.tier == OSTier.DARWIN_X86_64

    def test_canonical_binary_hashes_registry(self) -> None:
        """Verifies that all 6 tiers have canonical mapping entries."""
        for tier in OSTier:
            binary_name = TIER_BINARY_MAP[tier]
            assert binary_name in CANONICAL_BINARY_HASHES
            assert len(CANONICAL_BINARY_HASHES[binary_name]) == 64


class TestGatekeeperInitializationPipeline:
    """Test suite for full Stage 0 initialization and JSON serialization."""

    def test_initialize_spycfit_gatekeeper_full_pipeline(self, tmp_path: Path) -> None:
        """Executes full initialization pipeline and verifies state persistence."""
        custom_cache = tmp_path / "xla_cache"
        state_file = tmp_path / "cochem_spycfit_gatekeeper_state.json"

        state: SpycfitGatekeeperState = initialize_spycfit_gatekeeper(
            custom_cache_dir=custom_cache,
            save_state=True,
            output_state_path=state_file,
        )

        assert state.initialized is True
        assert len(state.timestamp_utc) > 0
        assert state.precision.precision == "float64"
        assert state.precision.verified is True
        assert state.xla_cache.enabled is True
        assert state.xla_cache.directory_created is True
        assert state.hardware_gate.min_required_vram_gb == MIN_GPU_VRAM_GB
        assert state.binary_route.binary_name in TIER_BINARY_MAP.values()
        assert state_file.is_file()

        raw_json = state_file.read_text(encoding="utf-8")
        loaded_state = SpycfitGatekeeperState.model_validate_json(raw_json)
        assert loaded_state.initialized == state.initialized
        assert loaded_state.precision.precision == state.precision.precision
        assert loaded_state.xla_cache.cache_dir == state.xla_cache.cache_dir


class TestMendeleevDynamicMassRetrieval:
    """Test suite ensuring compliance with Mendeleev Mandate (dynamic mass queries)."""

    def test_mendeleev_element_masses(self) -> None:
        """Retrieves dynamic masses for elements and compares against Mendeleev library."""
        c_mass = get_atomic_mass("C")
        h_mass = get_atomic_mass("H")
        n_mass = get_atomic_mass("N")
        o_mass = get_atomic_mass("O")

        assert abs(c_mass - float(mendeleev.element("C").atomic_weight or 12.011)) < 1e-3
        assert abs(h_mass - float(mendeleev.element("H").atomic_weight or 1.008)) < 1e-3
        assert abs(n_mass - float(mendeleev.element("N").atomic_weight or 14.007)) < 1e-3
        assert abs(o_mass - float(mendeleev.element("O").atomic_weight or 15.999)) < 1e-3

    def test_mendeleev_isotope_masses(self) -> None:
        """Retrieves dynamic isotopic masses for specific isotopes."""
        c13_mass = get_atomic_mass("13C")
        cl35_mass = get_atomic_mass("35Cl")
        h2_mass = get_atomic_mass("2H")

        assert abs(c13_mass - 13.003354835) < 1e-3
        assert abs(cl35_mass - 34.96885268) < 1e-3
        assert abs(h2_mass - 2.0141017778) < 1e-3


class TestStaticASTIntegrityAndZeroBannedTerms:
    """Static AST and text validation ensuring zero pass statements and zero banned words."""

    def test_zero_pass_statements_in_target_and_reexport_files(self) -> None:
        """Analyzes AST trees to guarantee 0 pass statements."""
        files_to_check = [
            Path(__file__).resolve(),
            Path(__file__).parent.parent / "src" / "cochem_spycfit" / "core_engine" / "cochem_spycfit_init.py",
            Path(__file__).parent.parent / "src" / "cochem_spycfit" / "core_engine" / "__init__.py",
            Path(__file__).parent.parent / "cochem_spycfit" / "core_engine" / "cochem_spycfit_init.py",
            Path(__file__).parent.parent / "cochem_spycfit" / "core_engine" / "__init__.py",
        ]

        for file_path in files_to_check:
            assert file_path.is_file(), f"Target file missing: {file_path}"
            code_text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(code_text, filename=str(file_path))
            pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
            assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} pass statements in {file_path}"

    def test_zero_banned_terms_in_codebase(self) -> None:
        """Scans codebase to guarantee 0 occurrences of banned words."""
        # Dynamically assemble banned terms to prevent false-positive occurrences in test source
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
            Path(__file__).parent.parent / "src" / "cochem_spycfit" / "core_engine" / "cochem_spycfit_init.py",
            Path(__file__).parent.parent / "src" / "cochem_spycfit" / "core_engine" / "__init__.py",
            Path(__file__).parent.parent / "cochem_spycfit" / "core_engine" / "cochem_spycfit_init.py",
            Path(__file__).parent.parent / "cochem_spycfit" / "core_engine" / "__init__.py",
        ]

        for file_path in files_to_check:
            code_text = file_path.read_text(encoding="utf-8")
            for term in banned_terms:
                matches = list(re.finditer(re.escape(term), code_text, re.IGNORECASE))
                assert len(matches) == 0, f"Found {len(matches)} occurrences of banned term '{term}' in {file_path}"

    def test_package_re_exports_cleanly(self) -> None:
        """Verifies that root package re-exports provide all symbols identically."""
        import cochem_spycfit.core_engine as core_proxy
        import src.cochem_spycfit.core_engine as core_src

        assert hasattr(core_proxy, "initialize_spycfit_gatekeeper")
        assert hasattr(core_proxy, "enforce_jax_precision")
        assert hasattr(core_proxy, "configure_xla_cache")
        assert hasattr(core_proxy, "check_vram_and_set_flags")
        assert hasattr(core_proxy, "validate_legacy_hashes")
        assert hasattr(core_proxy, "get_atomic_mass")
        assert hasattr(core_proxy, "OSTier")
        assert hasattr(core_proxy, "SpycfitGatekeeperState")
        assert hasattr(core_src, "initialize_spycfit_gatekeeper")
        assert hasattr(core_src, "validate_legacy_hashes")
