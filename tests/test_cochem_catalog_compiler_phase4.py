"""
CoChem-SpycFit: Comprehensive Phase 4 Physical Unit Test Suite for Catalog Compiler & PyArrow Out-of-Core Serialization.

Strict Zero-Mock & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical and mathematical calculations with real HDF5 and Parquet files on disk
- Validation of PyArrow chunked streaming with strict FP64 frequency and int32 quantum assignment type preservation
- Validation of lock-free Hive-partitioning ($COCHEM_STATE_DIR/SpycFit_Workspace/Catalogs/chunked_transitions/)
- Validation of jax.experimental.io_callback asynchronous array interception
- Validation of Map-Reduce master manifest compilation across partitioned parquet chunks
- Out-of-core streaming parquet reading with predicate pushdown and frequency range filtering
- AST static analysis enforcing zero pass statements and zero banned tokens
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)

from cochem_spycfit.serialization.cochem_catalog_compiler import (
    CatalogCompilerError,
    CatalogManifestRecord,
    CatalogSchemaError,
    TransitionBatch,
    TransitionRecord,
    compile_master_manifest,
    compute_sha256,
    get_chunked_transitions_dir,
    jax_serialize_chunk_callback,
    read_parquet_catalog,
    serialize_chunk_to_parquet,
)


# =============================================================================
# 1. FIXTURES & PHYSICAL ZERO-MOCK DATA GENERATORS
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated temporary workspace on physical disk."""
    ws = tmp_path / "catalog_phase4_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


def physical_water_chunk(n_lines: int = 500, offset: int = 0) -> Dict[str, np.ndarray]:
    """Retrieves authentic physical water transition lines from catalog_water.h5."""
    fixture_path = Path(__file__).parent / "fixtures" / "catalog_water.h5"
    with h5py.File(fixture_path, "r") as f_phys:
        cat_len = int(f_phys.attrs["total_transitions"])
        idx = np.arange(offset, offset + n_lines) % cat_len
        return {
            "frequencies": np.asarray(f_phys["frequencies"][idx], dtype=np.float64),
            "intensities": np.asarray(f_phys["intensities"][idx], dtype=np.float64),
            "lower_energies": np.asarray(f_phys["lower_energies"][idx], dtype=np.float64),
            "upper_energies": np.asarray(f_phys["upper_energies"][idx], dtype=np.float64),
            "uncertainties": np.asarray(f_phys["uncertainties"][idx], dtype=np.float64),
            "g_upper": np.asarray(f_phys["g_upper"][idx], dtype=np.int32),
            "quantum_numbers": np.asarray(f_phys["quantum_numbers"][idx], dtype=np.int32),
            "species_ids": np.asarray(f_phys["species_ids"][idx], dtype=np.int32),
        }


# =============================================================================
# 2. CHUNKED DIRECTORY RESOLUTION TESTS
# =============================================================================

class TestChunkedDirectoryResolution:
    """Validates state directory resolution via platformdirs and environment variables."""

    def test_get_chunked_transitions_dir_default(self) -> None:
        orig_state = os.environ.get("COCHEM_STATE_DIR")
        try:
            os.environ.pop("COCHEM_STATE_DIR", None)
            res_dir = get_chunked_transitions_dir()
            assert isinstance(res_dir, Path)
            assert "SpycFit_Workspace" in str(res_dir)
            assert "Catalogs" in str(res_dir)
            assert "chunked_transitions" in str(res_dir)
            assert res_dir.exists()
        finally:
            if orig_state is not None:
                os.environ["COCHEM_STATE_DIR"] = orig_state

    def test_get_chunked_transitions_dir_custom_env(self, temp_workspace: Path) -> None:
        orig_state = os.environ.get("COCHEM_STATE_DIR")
        try:
            custom_state = temp_workspace / "custom_state"
            os.environ["COCHEM_STATE_DIR"] = str(custom_state)
            res_dir = get_chunked_transitions_dir()
            expected = custom_state / "SpycFit_Workspace" / "Catalogs" / "chunked_transitions"
            assert res_dir == expected.resolve()
            assert res_dir.exists()
        finally:
            if orig_state is not None:
                os.environ["COCHEM_STATE_DIR"] = orig_state
            else:
                os.environ.pop("COCHEM_STATE_DIR", None)

    def test_get_chunked_transitions_dir_explicit_override(self, temp_workspace: Path) -> None:
        explicit_dir = temp_workspace / "explicit_dir"
        res_dir = get_chunked_transitions_dir(base_dir=explicit_dir)
        assert res_dir == explicit_dir.resolve()
        assert res_dir.exists()


# =============================================================================
# 3. PYARROW CHUNK SERIALIZATION & STRICT TYPING TESTS
# =============================================================================

class TestPyArrowChunkSerialization:
    """Validates PyArrow serialization, strict FP64/int32 type preservation, and Hive partitioning."""

    def test_serialize_chunk_to_parquet_strict_types(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=200)
        out_dir = temp_workspace / "parquet_chunks"

        parquet_file = serialize_chunk_to_parquet(
            frequency_data=data["frequencies"],
            quantum_assignments=data["quantum_numbers"],
            temp_step=1,
            batch_id=0,
            uuid_str="unit_test_001",
            output_dir=out_dir,
            partition_by_temp=True,
        )

        assert parquet_file.is_file()
        assert parquet_file.name == "chunk_1_0_unit_test_001.parquet"
        assert parquet_file.parent.name == "temperature_step=1"

        # Read back PyArrow table and inspect exact column types
        table = pq.read_table(parquet_file)
        schema = table.schema

        # Strict type enforcement: frequencies must be FP64 (float64)
        assert schema.field("frequencies").type == pa.float64()
        assert schema.field("intensities").type == pa.float64()
        assert schema.field("lower_energies").type == pa.float64()
        assert schema.field("upper_energies").type == pa.float64()
        assert schema.field("uncertainties").type == pa.float64()

        # Strict type enforcement: quantum numbers and indices must be int32
        assert schema.field("g_upper").type == pa.int32()
        assert schema.field("species_ids").type == pa.int32()
        assert schema.field("J_upper").type == pa.int32()
        assert schema.field("Ka_upper").type == pa.int32()
        assert schema.field("Kc_upper").type == pa.int32()
        assert schema.field("J_lower").type == pa.int32()
        assert schema.field("Ka_lower").type == pa.int32()
        assert schema.field("Kc_lower").type == pa.int32()
        assert schema.field("quantum_numbers").type == pa.list_(pa.int32())

        # Verify numerical fidelity
        read_freqs = table["frequencies"].to_numpy()
        np.testing.assert_allclose(read_freqs, data["frequencies"])

        read_j_up = table["J_upper"].to_numpy()
        np.testing.assert_array_equal(read_j_up, data["quantum_numbers"][:, 0])

    def test_serialize_chunk_to_parquet_from_transition_batch(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=50)
        batch = TransitionBatch(
            frequencies_mhz=data["frequencies"].tolist(),
            intensities=data["intensities"].tolist(),
            lower_energies_cm1=data["lower_energies"].tolist(),
            upper_energies_cm1=data["upper_energies"].tolist(),
            uncertainties_mhz=data["uncertainties"].tolist(),
            g_upper=data["g_upper"].tolist(),
            quantum_numbers=[tuple(int(x) for x in row) for row in data["quantum_numbers"]],
            species_ids=data["species_ids"].tolist(),
        )

        out_dir = temp_workspace / "batch_chunks"
        parquet_file = serialize_chunk_to_parquet(
            frequency_data=batch,
            temp_step=2,
            batch_id="batch_alpha",
            uuid_str="batch_uuid_123",
            output_dir=out_dir,
        )

        assert parquet_file.is_file()
        assert "batch_alpha" in parquet_file.name
        table = pq.read_table(parquet_file)
        assert table.num_rows == 50
        np.testing.assert_allclose(table["intensities"].to_numpy(), data["intensities"])

    def test_serialize_chunk_to_parquet_from_dict_and_3d_quantum_assignments(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=40)
        qn_3 = data["quantum_numbers"][:, :3]

        dict_payload = {
            "frequencies": data["frequencies"],
            "intensities": data["intensities"],
            "lower_energies": data["lower_energies"],
            "upper_energies": data["upper_energies"],
            "uncertainties": data["uncertainties"],
            "g_upper": data["g_upper"],
            "species_ids": data["species_ids"],
        }

        out_dir = temp_workspace / "dict_chunks"
        parquet_file = serialize_chunk_to_parquet(
            frequency_data=dict_payload,
            quantum_assignments=qn_3,
            temp_step=3,
            batch_id=1,
            output_dir=out_dir,
            partition_by_temp=False,
        )

        assert parquet_file.is_file()
        table = pq.read_table(parquet_file)
        assert table.num_rows == 40
        assert table.schema.field("J_upper").type == pa.int32()
        assert table.schema.field("Ka_upper").type == pa.int32()
        assert table.schema.field("Kc_upper").type == pa.int32()
        assert table.schema.field("J_lower").type == pa.int32()
        np.testing.assert_array_equal(table["J_lower"].to_numpy(), np.zeros(40, dtype=np.int32))

    def test_serialize_chunk_to_parquet_concurrent_lock_free(self, temp_workspace: Path) -> None:
        """Validates lock-free append-only concurrent writes from multiple threads without collision."""
        out_dir = temp_workspace / "concurrent_chunks"
        num_threads = 8
        chunks_per_thread = 5

        def worker_task(thread_idx: int) -> List[Path]:
            created: List[Path] = []
            for c_idx in range(chunks_per_thread):
                data = physical_water_chunk(n_lines=25, offset=(thread_idx * 100 + c_idx * 25))
                p_file = serialize_chunk_to_parquet(
                    frequency_data=data,
                    temp_step=thread_idx % 3,
                    batch_id=f"t{thread_idx}_c{c_idx}",
                    output_dir=out_dir,
                    partition_by_temp=True,
                )
                created.append(p_file)
            return created

        threads = []
        results: Dict[int, List[Path]] = {}

        def thread_wrapper(t_id: int) -> None:
            results[t_id] = worker_task(t_id)

        for t_id in range(num_threads):
            t = threading.Thread(target=thread_wrapper, args=(t_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        all_files: List[Path] = []
        for t_id in range(num_threads):
            all_files.extend(results[t_id])

        assert len(all_files) == num_threads * chunks_per_thread
        for p in all_files:
            assert p.is_file()
            tbl = pq.read_table(p)
            assert tbl.num_rows == 25


# =============================================================================
# 4. JAX IO_CALLBACK ASYNCHRONOUS INTERCEPTION TESTS
# =============================================================================

class TestJAXIOCallbackIntegration:
    """Validates jax.experimental.io_callback integration with JAX compiled computational graphs."""

    def test_jax_io_callback_serialization_in_jit(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=64)
        out_dir = temp_workspace / "jax_callback_chunks"

        jax_freqs = jnp.array(data["frequencies"], dtype=jnp.float64)
        jax_qn = jnp.array(data["quantum_numbers"], dtype=jnp.int32)
        jax_intens = jnp.array(data["intensities"], dtype=jnp.float64)

        @jax.jit
        def computational_pipeline(freq_arr: jnp.ndarray, qn_arr: jnp.ndarray, intens_arr: jnp.ndarray) -> jnp.ndarray:
            shifted_freqs = freq_arr + 1500.0
            scaled_intens = intens_arr * 2.5

            status = jax_serialize_chunk_callback(
                frequency_data=shifted_freqs,
                quantum_assignments=qn_arr,
                temp_step=10,
                batch_id=42,
                uuid_str="jax_cb_001",
                output_dir=str(out_dir),
                intensities=scaled_intens,
            )
            return status

        result_status = computational_pipeline(jax_freqs, jax_qn, jax_intens)
        assert int(result_status) == 1

        expected_file = out_dir / "temperature_step=10" / "chunk_10_42_jax_cb_001.parquet"
        assert expected_file.is_file()

        table = pq.read_table(expected_file)
        assert table.num_rows == 64
        read_freqs = table["frequencies"].to_numpy()
        np.testing.assert_allclose(read_freqs, data["frequencies"] + 1500.0)

        read_intens = table["intensities"].to_numpy()
        np.testing.assert_allclose(read_intens, data["intensities"] * 2.5)

    def test_jax_io_callback_array_fidelity(self, temp_workspace: Path) -> None:
        data = physical_water_chunk(n_lines=128)
        out_dir = temp_workspace / "fidelity_chunks"

        jax_freqs = jnp.array(data["frequencies"], dtype=jnp.float64)
        jax_qn = jnp.array(data["quantum_numbers"], dtype=jnp.int32)

        status = jax_serialize_chunk_callback(
            frequency_data=jax_freqs,
            quantum_assignments=jax_qn,
            temp_step=5,
            batch_id=99,
            uuid_str="fidelity_uuid",
            output_dir=str(out_dir),
        )
        assert int(status) == 1

        written_file = out_dir / "temperature_step=5" / "chunk_5_99_fidelity_uuid.parquet"
        assert written_file.is_file()

        table = pq.read_table(written_file)
        assert table.schema.field("frequencies").type == pa.float64()
        assert table.schema.field("quantum_numbers").type == pa.list_(pa.int32())
        np.testing.assert_allclose(table["frequencies"].to_numpy(), data["frequencies"])


# =============================================================================
# 5. MAP-REDUCE MASTER MANIFEST COMPILATION TESTS
# =============================================================================

class TestMasterManifestMapReduce:
    """Validates Map-Reduce aggregation across partitioned Parquet chunk catalogs."""

    def test_compile_master_manifest_map_reduce_basic(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "manifest_test_chunks"
        num_chunks = 4
        chunk_size = 100
        total_expected_lines = num_chunks * chunk_size

        all_freqs: List[float] = []
        all_intens: List[float] = []

        for c_idx in range(num_chunks):
            data = physical_water_chunk(n_lines=chunk_size, offset=c_idx * chunk_size)
            all_freqs.extend(data["frequencies"].tolist())
            all_intens.extend(data["intensities"].tolist())

            serialize_chunk_to_parquet(
                frequency_data=data,
                temp_step=c_idx % 2,
                batch_id=c_idx,
                uuid_str=f"chunk_map_{c_idx}",
                output_dir=chunk_dir,
                partition_by_temp=True,
            )

        manifest_json = temp_workspace / "master_manifest.json"
        manifest = compile_master_manifest(
            chunk_directory=chunk_dir,
            output_manifest_path=manifest_json,
            custom_metadata={"survey": "GBT-Astrochemistry-Survey", "polarization": "dual"},
        )

        assert isinstance(manifest, CatalogManifestRecord)
        assert manifest.total_transitions == total_expected_lines
        assert manifest.frequency_min_mhz == pytest.approx(min(all_freqs), rel=1e-7)
        assert manifest.frequency_max_mhz == pytest.approx(max(all_freqs), rel=1e-7)
        assert manifest.intensity_min == pytest.approx(min(all_intens), rel=1e-7)
        assert manifest.intensity_max == pytest.approx(max(all_intens), rel=1e-7)
        assert manifest.num_shards == num_chunks
        assert len(manifest.shard_paths) == num_chunks
        assert manifest.metadata["survey"] == "GBT-Astrochemistry-Survey"

        assert manifest_json.is_file()
        with open(manifest_json, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        assert meta_dict["total_transitions"] == total_expected_lines
        assert meta_dict["num_shards"] == num_chunks

    def test_compile_master_manifest_empty_directory_error(self, temp_workspace: Path) -> None:
        empty_dir = temp_workspace / "empty_chunk_dir"
        empty_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(CatalogCompilerError) as exc_info:
            compile_master_manifest(empty_dir)
        assert "No parquet chunk files found" in str(exc_info.value)


# =============================================================================
# 6. OUT-OF-CORE PARQUET STREAMING READER TESTS
# =============================================================================

class TestParquetOutofCoreReader:
    """Validates out-of-core streaming iteration and predicate pushdown on Parquet datasets."""

    def test_read_parquet_catalog_streaming(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "reader_test_chunks"
        num_chunks = 3
        chunk_size = 150

        for c_idx in range(num_chunks):
            data = physical_water_chunk(n_lines=chunk_size, offset=c_idx * chunk_size)
            serialize_chunk_to_parquet(
                frequency_data=data,
                temp_step=c_idx,
                batch_id=c_idx,
                uuid_str=f"reader_uuid_{c_idx}",
                output_dir=chunk_dir,
                partition_by_temp=True,
            )

        total_read = 0
        for batch in read_parquet_catalog(chunk_dir, batch_size=100):
            assert "frequencies" in batch
            assert "intensities" in batch
            assert "quantum_numbers" in batch
            assert isinstance(batch["frequencies"], np.ndarray)
            assert batch["frequencies"].dtype == np.float64
            assert batch["quantum_numbers"].dtype == np.int32
            total_read += len(batch["frequencies"])

        assert total_read == num_chunks * chunk_size

    def test_read_parquet_catalog_frequency_filter(self, temp_workspace: Path) -> None:
        chunk_dir = temp_workspace / "filter_test_chunks"
        data = physical_water_chunk(n_lines=300)
        serialize_chunk_to_parquet(
            frequency_data=data,
            temp_step=1,
            batch_id=0,
            uuid_str="filter_chunk",
            output_dir=chunk_dir,
            partition_by_temp=True,
        )

        all_freqs = data["frequencies"]
        f_min = float(np.percentile(all_freqs, 25))
        f_max = float(np.percentile(all_freqs, 75))

        filtered_freqs: List[float] = []
        for batch in read_parquet_catalog(chunk_dir, frequency_range=(f_min, f_max)):
            f_arr = batch["frequencies"]
            assert np.all(f_arr >= f_min)
            assert np.all(f_arr <= f_max)
            filtered_freqs.extend(f_arr.tolist())

        assert len(filtered_freqs) > 0
        expected_count = int(np.sum((all_freqs >= f_min) & (all_freqs <= f_max)))
        assert len(filtered_freqs) == expected_count


# =============================================================================
# 7. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliancePhase4:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_catalog_compiler_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_catalog_compiler.py"
        assert module_path.is_file()

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_test_suite_ast_compliance(self) -> None:
        test_path = Path(__file__).resolve()
        assert test_path.is_file()

        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_no_banned_tokens_in_production_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_catalog_compiler.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        banned = ["mock", "dummy", "placeholder", "fake", "stub"]
        violations: List[str] = []

        class IdentifierScanner(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for b in banned:
                    if b in node.name.lower():
                        violations.append(f"Function {node.name} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for b in banned:
                    if b in node.id.lower():
                        violations.append(f"Variable {node.id} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    for b in banned:
                        if b in alias.name.lower():
                            violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for b in banned:
                        if b in node.module.lower():
                            violations.append(f"Forbidden import from '{node.module}' at line {node.lineno}")
                self.generic_visit(node)

        IdentifierScanner().visit(tree)
        assert len(violations) == 0, f"Banned tokens found: {violations}"
