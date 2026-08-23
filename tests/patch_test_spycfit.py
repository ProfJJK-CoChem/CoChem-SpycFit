import os
import re

test_file = r"D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_spycfit.py"
with open(test_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Add h5py and fixture path
if "import h5py" not in content:
    content = content.replace("import pytest", "import pytest\nimport h5py")

# H_H2_TS
h_h2_ts_pattern = r"H_H2_TS: np\.ndarray = np\.zeros\(\(6, 6\)\)\n\s*H_H2_TS\[2, 2\] = -5\.0\n\s*H_H2_TS\[5, 5\] = -5\.0\n\s*H_H2_TS\[2, 5\] = 5\.0\n\s*H_H2_TS\[5, 2\] = 5\.0"
h_h2_ts_repl = """with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
    H_H2_TS: np.ndarray = f["H_H2_TS"][:]"""
content = re.sub(h_h2_ts_pattern, h_h2_ts_repl, content)

# cubic_k
cubic_k_pattern = r"cubic_k: np\.ndarray = np\.zeros\(\(3, 3, 3\)\)\n\s*cubic_k\[0, 0, 1\] = 0\.5"
cubic_k_repl = """with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
        cubic_k: np.ndarray = f["cubic_k"][:]"""
content = re.sub(cubic_k_pattern, cubic_k_repl, content)

# rm2_c
rm2_c_pattern = r"rm2: Dict\[str, Any\] = \{'c_params': np\.zeros\(3\), 'd_params': np\.array\(\[1\.0, 0\.5, 0\.25\]\)\}"
rm2_c_repl = """with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
        rm2: Dict[str, Any] = {'c_params': f["rm2_c"][:], 'd_params': np.array([1.0, 0.5, 0.25])}"""
content = re.sub(rm2_c_pattern, rm2_c_repl, content)

# orca freq
orca_freq_pattern = r"orca_output_text: str = \"\"\"\n\s*Rotational constants \(MHz\): 10234\.56   5123\.45   2561\.78\n\s*Total Dipole Moment :    1\.2345   -0\.5678   0\.8912\n\s*\"\"\""
orca_freq_repl = "orca_output_text: str = (Path(__file__).parent / \"fixtures\" / \"orca_freq.out\").read_text()"
content = re.sub(orca_freq_pattern, orca_freq_repl, content)

# orca normal
orca_pattern = r"orca_output_text: str = \"\"\"\n\s*Program Version 6\.0\.0\n\s*Rotational constants \(MHz\) :  12500\.4500   4800\.2300   3600\.1200\n\s*Total Dipole Moment :    1\.4500    0\.8200    0\.0000\n\s*\"\"\""
orca_repl = "orca_output_text: str = (Path(__file__).parent / \"fixtures\" / \"orca.out\").read_text()"
content = re.sub(orca_pattern, orca_repl, content)

# gxtb
gxtb_pattern = r"gxtb_output_text: str = \"\"\"\n\s*gfn2-xtb molecular dipole : x = 0\.5000 y = 1\.2000 z = 0\.0000 \(Debye\)\n\s*rotational constants \(cm-1\) : 0\.41695 0\.16011 0\.12008\n\s*\"\"\""
gxtb_repl = "gxtb_output_text: str = (Path(__file__).parent / \"fixtures\" / \"gxtb.out\").read_text()"
content = re.sub(gxtb_pattern, gxtb_repl, content)

# nbo
nbo_pattern = r"nbo_output_text: str = \"\"\"\n\s*NATURAL POPULATION ANALYSIS\n\s*---------------------------\n\s*C 1  -0\.345\n\s*H 2   0\.115\n\s*H 3   0\.115\n\s*H 4   0\.115\n\s*---------------------------\n\s*==========\n\s*\"\"\""
nbo_repl = "nbo_output_text: str = (Path(__file__).parent / \"fixtures\" / \"nbo.out\").read_text()"
content = re.sub(nbo_pattern, nbo_repl, content)

with open(test_file, 'w', encoding='utf-8') as f:
    f.write(content)
