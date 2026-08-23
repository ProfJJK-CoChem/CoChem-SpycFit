Cycle 10: Implement code for prompt at D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_2_Prompt5_spcat_bridge.md strictly adhering to Zero-Mock mandate. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit. Generate unit tests first. IMPORTANT: You MUST update/create `pytest.ini` to restrict `testpaths` to ONLY the tests you are writing for this prompt, otherwise the global 1500+ test suite will run and crash your context. 
Test Failures from previous run:
Output: ============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: D:\__CoChem\GitHub-Repo\CoChem-SpycFit
configfile: pytest.ini
plugins: anyio-4.10.0, typeguard-4.6.0
collected 232 items

tests\test_cochem_catalog_compiler.py .................................. [ 14%]
..............                                                           [ 20%]
tests\test_cochem_convolver.py ................................          [ 34%]
tests\test_cochem_fit_registry_manager.py ....................           [ 43%]
tests\test_cochem_jax_fitter.py .....FF.F.......                         [ 50%]
tests\test_cochem_spycfit_orchestrator.py .......................        [ 59%]
tests\test_cochem_spycfit_triage.py ..............................       [ 72%]
tests\test_extractor_physical.py .                                       [ 73%]
tests\test_spycfit.py .......s................                           [ 83%]
tests\test_cochem_spcat_bridge.py ...................................... [100%]

================================== FAILURES ===================================
__________________ test_block_diagonalization_scan_accuracy ___________________

    def test_block_diagonalization_scan_accuracy() -> None:
        """Verifies block diagonalization eigenvalues and eigenvectors against scipy.linalg.eigh."""
        # Load physical Hessian to form physically grounded Hermitian blocks
        import h5py
        with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
>           blocks_np = f_h5['jax_fitter_blocks'][:]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_cochem_jax_fitter.py:245: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
C:\Users\ansac\anaconda3\Lib\site-packages\h5py\_hl\group.py:367: in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   KeyError: "Unable to synchronously open object (object 'jax_fitter_blocks' doesn't exist)"

h5py/h5o.pyx:255: KeyError
____________ test_block_diagonalization_chunking_and_vram_capping _____________

    def test_block_diagonalization_chunking_and_vram_capping() -> None:
        """Verifies chunked scan execution and VRAM budget capping produce identical results."""
        with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
>           blocks_np = f_h5['jax_fitter_blocks'][:]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_cochem_jax_fitter.py:273: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
C:\Users\ansac\anaconda3\Lib\site-packages\h5py\_hl\group.py:367: in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   KeyError: "Unable to synchronously open object (object 'jax_fitter_blocks' doesn't exist)"

h5py/h5o.pyx:255: KeyError
__________________ test_block_diagonalization_invalid_shape ___________________

    def test_block_diagonalization_invalid_shape() -> None:
        """Verifies that non-square and invalid dimensional tensors raise InvalidHamiltonianError."""
        # 2D tensor instead of 3D
        invalid_2d = np.eye(6)
        with pytest.raises(InvalidHamiltonianError, match="Expected 3D tensor"):
            block_diagonalization_scan(invalid_2d)
    
        # Non-square blocks (e.g. 5x4)
        with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
>           invalid_rect = f_h5['invalid_rect'][:]
                           ^^^^^^^^^^^^^^^^^^^^

tests\test_cochem_jax_fitter.py:323: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
C:\Users\ansac\anaconda3\Lib\site-packages\h5py\_hl\group.py:367: in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
    ???
h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   KeyError: "Unable to synchronously open object (object 'invalid_rect' doesn't exist)"

h5py/h5o.pyx:255: KeyError
=========================== short test summary info ===========================
FAILED tests/test_cochem_jax_fitter.py::test_block_diagonalization_scan_accuracy
FAILED tests/test_cochem_jax_fitter.py::test_block_diagonalization_chunking_and_vram_capping
FAILED tests/test_cochem_jax_fitter.py::test_block_diagonalization_invalid_shape
================== 3 failed, 228 passed, 1 skipped in 31.17s ==================

Error: 