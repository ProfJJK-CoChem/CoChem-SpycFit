Cycle 2: Implement code for prompt at D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_2_Prompt6_ui.md strictly adhering to Zero-Mock mandate. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit. Generate unit tests first. IMPORTANT: You MUST update/create `pytest.ini` to restrict `testpaths` to ONLY the tests you are writing for this prompt, otherwise the global 1500+ test suite will run and crash your context. 
Test Failures from previous run:
Output: ============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: D:\__CoChem\GitHub-Repo\CoChem-SpycFit
configfile: pytest.ini
plugins: anyio-4.10.0, typeguard-4.6.0
collected 91 items

tests\test_cochem_catalog_compiler.py ...................F...F.......... [ 37%]
..............                                                           [ 52%]
tests\test_cochem_spycfit_orchestrator.py .......................        [ 78%]
tests\test_cochem_spycfit_ui.py ....................                     [100%]

================================== FAILURES ===================================
___ TestHPCVirtualLayoutAggregator.test_compile_virtual_catalog_multi_shard ___

self = <test_cochem_catalog_compiler.TestHPCVirtualLayoutAggregator object at 0x00000267D0327890>
temp_workspace = WindowsPath('C:/Users/ansac/AppData/Local/Temp/pytest-of-ansac/pytest-8854/test_compile_virtual_catalog_m0/catalog_workspace')

    def test_compile_virtual_catalog_multi_shard(self, temp_workspace: Path) -> None:
        # Create 3 physical shard containers
        num_shards = 3
        shard_lines = [5000, 7500, 10000]
        shard_paths: List[Path] = []
    
        for i, count in enumerate(shard_lines):
            s_path = temp_workspace / f"worker_shard_{i}.h5"
            generator = chunk_generator(count, chunk_size=2048, base_freq=5000.0 + i * 2000.0)
            stream_chunked_catalog(
                data_iterator=generator,
                output_path=s_path,
                config=CatalogCompilerConfig(chunk_size=2048),
            )
            shard_paths.append(s_path)
    
        master_path = temp_workspace / "master_virtual_catalog.h5"
    
        manifest = compile_virtual_catalog(
            shard_paths=shard_paths,
            output_path=master_path,
            custom_metadata={"experiment": "High-Res Astrochemical Survey", "survey_band": "Ka"},
        )
    
        assert master_path.is_file()
        assert manifest.is_virtual is True
        assert manifest.num_shards == 3
        assert manifest.total_transitions == sum(shard_lines)
        assert len(manifest.shard_paths) == 3
        assert manifest.metadata["experiment"] == "High-Res Astrochemical Survey"
    
        # Verify VirtualLayout access across shard boundaries
        with h5py.File(str(master_path), "r") as h5_master:
            assert h5_master["frequencies"].is_virtual is True
            assert h5_master["frequencies"].shape == (sum(shard_lines),)
            assert h5_master["quantum_numbers"].shape == (sum(shard_lines), 6)
    
            # Read slice across first boundary (lines 4995 to 5005)
            boundary_slice = h5_master["frequencies"][4995:5005]
            assert len(boundary_slice) == 10
>           assert np.all(np.diff(boundary_slice) > 0.0)
E           assert np.False_
E            +  where np.False_ = <function all at 0x00000267C9BD9730>(array([ 1.00010001e-02,  1.00010000e-02,  1.00010001e-02,  1.00010001e-02,\n       -3.73675896e+05,  1.00010001e-02,  1.00010001e-02,  1.00010001e-02,\n        1.00010001e-02]) > 0.0)
E            +    where <function all at 0x00000267C9BD9730> = np.all
E            +    and   array([ 1.00010001e-02,  1.00010000e-02,  1.00010001e-02,  1.00010001e-02,\n       -3.73675896e+05,  1.00010001e-02,  1.00010001e-02,  1.00010001e-02,\n        1.00010001e-02]) = <function diff at 0x00000267C9F34CF0>(array([556985.9426955, 556985.9526965, 556985.9626975, 556985.9726985,\n       556985.9826995, 183310.087    , 183310.097001 , 183310.107002 ,\n       183310.117003 , 183310.127004 ]))
E            +      where <function diff at 0x00000267C9F34CF0> = np.diff

tests\test_cochem_catalog_compiler.py:530: AssertionError
_______ TestOutofCoreReader.test_read_chunked_catalog_frequency_filter ________

self = <test_cochem_catalog_compiler.TestOutofCoreReader object at 0x00000267D0327C50>
temp_workspace = WindowsPath('C:/Users/ansac/AppData/Local/Temp/pytest-of-ansac/pytest-8854/test_read_chunked_catalog_freq0/catalog_workspace')

    def test_read_chunked_catalog_frequency_filter(self, temp_workspace: Path) -> None:
        cat_file = temp_workspace / "catalog_filter.h5"
        total_transitions = 10000
        generator = chunk_generator(total_transitions, chunk_size=2048, base_freq=10000.0)
        stream_chunked_catalog(generator, cat_file, config=CatalogCompilerConfig(chunk_size=2048))
    
        # Query range 10100.0 MHz to 10200.0 MHz
        min_f, max_f = 10100.0, 10200.0
        filtered_freqs: List[float] = []
    
        for chunk in read_chunked_catalog(cat_file, chunk_size=2048, frequency_range=(min_f, max_f)):
            c_f = chunk["frequencies"]
            assert np.all(c_f >= min_f)
            assert np.all(c_f <= max_f)
            filtered_freqs.extend(c_f.tolist())
    
>       assert len(filtered_freqs) > 0
E       assert 0 > 0
E        +  where 0 = len([])

tests\test_cochem_catalog_compiler.py:584: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_cochem_catalog_compiler.py::TestHPCVirtualLayoutAggregator::test_compile_virtual_catalog_multi_shard
FAILED tests/test_cochem_catalog_compiler.py::TestOutofCoreReader::test_read_chunked_catalog_frequency_filter
======================== 2 failed, 89 passed in 5.36s =========================

Error: 