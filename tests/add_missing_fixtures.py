import h5py
import numpy as np

count = 5

dt = np.dtype([
    ("frequencies", np.float64),
    ("intensities", np.float64),
    ("lower_state_energy", np.float64),
    ("g_upper", np.int32),
    ("species_ids", np.int32),
])

with h5py.File("D:/__CoChem/GitHub-Repo/CoChem-SpycFit/tests/fixtures/test_arrays.h5", "a") as f:
    # triage
    if "triage_a_mat" not in f:
        random_state = np.random.RandomState(42)
        a_mat = random_state.randn(9, 9)
        f.create_dataset("triage_a_mat", data=a_mat)

    # orchestrator
    if "invalid_d_mu" not in f:
        f.create_dataset("invalid_d_mu", data=np.ones((2, 6)))
    
    if "dipole_derivs_ones_3_9" not in f:
        f.create_dataset("dipole_derivs_ones_3_9", data=np.ones((3, 9)))
    
    # quantum extractor
    if "so2_dipole_derivs" not in f:
        f.create_dataset("so2_dipole_derivs", data=np.ones((9, 3), dtype=np.float64) * 0.12)
    
    # convolver
    if "convolver_multi_intensities" not in f:
        f.create_dataset("convolver_multi_intensities", data=np.ones((4, 20), dtype=np.float64))
        
    # fit registry manager
    if "registry_dense_matrix" not in f:
        f.create_dataset("registry_dense_matrix", data=np.ones((10, 10), dtype=np.float64))
    
    if "registry_test_matrix" not in f:
        f.create_dataset("registry_test_matrix", data=np.random.RandomState(42).randn(50, 50))
    
    # catalog compiler
    if "compiler_qn_zeros" not in f:
        f.create_dataset("compiler_qn_zeros", data=np.zeros((count, 6), dtype=np.int32))
    
    if "compiler_sp_ids_zeros" not in f:
        f.create_dataset("compiler_sp_ids_zeros", data=np.zeros(count, dtype=np.int32))
    
    if "compiler_arr_zeros" not in f:
        f.create_dataset("compiler_arr_zeros", data=np.zeros(5, dtype=dt))

    # jax fitter
    if "jax_fitter_blocks" not in f:
        rng = np.random.RandomState(42)
        raw_blocks = rng.randn(8, 4, 4)
        sym_blocks = 0.5 * (raw_blocks + np.transpose(raw_blocks, (0, 2, 1)))
        f.create_dataset("jax_fitter_blocks", data=sym_blocks)

    if "invalid_rect" not in f:
        f.create_dataset("invalid_rect", data=np.ones((2, 5, 4), dtype=np.float64))
