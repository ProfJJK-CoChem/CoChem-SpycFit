import os
import re
import h5py
import numpy as np
from pathlib import Path

tests_dir = Path(r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests")
fixtures_dir = tests_dir / "fixtures"
fixtures_dir.mkdir(exist_ok=True)

# 1. Create text fixtures
orca_text = """
    Rotational constants (MHz): 10234.56   5123.45   2561.78
    Total Dipole Moment :    1.2345   -0.5678   0.8912
"""
(fixtures_dir / "orca_freq.out").write_text(orca_text)

orca2_text = """
    Program Version 6.0.0
    Rotational constants (MHz) :  12500.4500   4800.2300   3600.1200
    Total Dipole Moment :    1.4500    0.8200    0.0000
"""
(fixtures_dir / "orca.out").write_text(orca2_text)

gxtb_text = """
    gfn2-xtb molecular dipole : x = 0.5000 y = 1.2000 z = 0.0000 (Debye)
    rotational constants (cm-1) : 0.41695 0.16011 0.12008
"""
(fixtures_dir / "gxtb.out").write_text(gxtb_text)

nbo_text = """
    NATURAL POPULATION ANALYSIS
    ---------------------------
    C 1  -0.345
    H 2   0.115
    H 3   0.115
    H 4   0.115
    ---------------------------
    ==========
"""
(fixtures_dir / "nbo.out").write_text(nbo_text)

# 2. Create HDF5 fixtures
with h5py.File(fixtures_dir / "test_arrays.h5", "w") as f:
    H_H2_TS = np.zeros((6, 6))
    H_H2_TS[2, 2] = -5.0
    H_H2_TS[5, 5] = -5.0
    H_H2_TS[2, 5] = 5.0
    H_H2_TS[5, 2] = 5.0
    f.create_dataset("H_H2_TS", data=H_H2_TS)
    
    cubic_k = np.zeros((3, 3, 3))
    cubic_k[0, 0, 1] = 0.5
    f.create_dataset("cubic_k", data=cubic_k)

    rm2_c = np.zeros(3)
    f.create_dataset("rm2_c", data=rm2_c)
    
    f_cart_zero = np.zeros((6, 6), dtype=np.float64)
    f.create_dataset("f_cart_zero", data=f_cart_zero)
