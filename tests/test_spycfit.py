#!/usr/bin/env python3
"""
PyTest Suite for CoChem-SpycFit
Resolves SPYCFIT-17: Comprehensive automated unit test coverage.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import cochem_convolver
import cochem_quantum_extractor
import cochem_spycfit_init

def test_watson_s_reduced_model():
    jnp = cochem_convolver.jnp
    params = jnp.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])
    freqs = cochem_convolver.watson_s_reduced_model(params)
    assert freqs.shape == (8,)
    assert float(freqs[0]) > 0

def test_compute_frequency_matrix():
    hessian = np.diag([0.01, 0.04, 0.09])
    freqs = cochem_convolver.compute_frequency_matrix(hessian)
    assert len(freqs) == 3
    np.testing.assert_allclose(freqs[0], 0.1 * 5140.48714, rtol=1e-3)

def test_loio_cross_validation():
    test_data = np.array([12500.0, 11000.0, 9500.0, 15000.0, 14000.0])
    res = cochem_convolver.loio_cross_validation({}, test_data)
    assert res['n_isotopes'] == 5
    assert 'rmse' in res
    assert 0.0 <= res['r_squared'] <= 1.0

def test_penalty_function():
    params = np.array([10000.0, 5000.0, 2500.0])
    data = np.array([12500.0, 11000.0, 9500.0, 24000.0, 21500.0, 19000.0, 31000.0, 29500.0])
    pen = cochem_convolver.penalty_function(params, data)
    assert isinstance(pen, float)
    assert pen > 0

def test_mass_weighting_exact():
    symbols = ['H', 'O', 'H']
    hessian = np.eye(9)
    mw_h = cochem_quantum_extractor.apply_mass_weighting(hessian, symbols)
    assert mw_h.shape == (9, 9)
    expected_o_weight = 1.0 / 15.994915
    np.testing.assert_allclose(mw_h[3, 3], expected_o_weight, rtol=1e-4)

def test_parse_orca_freq_output():
    sample_orca = """
    Rotational constants (MHz): 10234.56   5123.45   2561.78
    Total Dipole Moment :    1.2345   -0.5678   0.8912
    """
    parsed = cochem_quantum_extractor.parse_orca_freq_output(sample_orca)
    assert parsed['rotational_constants']['A'] == 10234.56
    assert parsed['dipole_moment']['X'] == 1.2345

def test_vpt2_resonance_trap():
    freqs = np.array([100.0, 201.0, 500.0])
    cubic_k = np.zeros((3, 3, 3))
    cubic_k[0, 0, 1] = 0.5
    res = cochem_quantum_extractor.vpt2_resonance_trap(freqs, cubic_k, threshold=5.0)
    assert len(res) >= 1
    assert res[0][0] == 'Fermi'

def test_pickett_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cochem_quantum_extractor.generate_pickett_files({}, output_dir=tmpdir)
        assert (Path(tmpdir) / "output.lin").exists()
        assert (Path(tmpdir) / "output.cat").exists()
        assert (Path(tmpdir) / "output.fit").exists()

def test_spycfit_init_components():
    res = cochem_spycfit_init.initialize_spycfit_components()
    assert res["initialized"] is True
    assert "components" in res

def test_quantum_extractor_orca_and_gxtb():
    from cochem_quantum_extractor import QuantumExtractor
    extractor = QuantumExtractor()

    # ORCA test
    orca_sample = """
    Program Version 6.0.0
    Rotational constants (MHz) :  12500.4500   4800.2300   3600.1200
    Total Dipole Moment :    1.4500    0.8200    0.0000
    """
    res_orca = extractor.parse_orca_output(orca_sample)
    assert res_orca['rotational_constants']['A'] == 12500.45
    assert res_orca['dipole_moment']['mu_a'] == 1.45
    assert res_orca['dipole_moment']['mu_b'] == 0.82

    # g-xTB test
    gxtb_sample = """
    gfn2-xtb molecular dipole : x = 0.5000 y = 1.2000 z = 0.0000 (Debye)
    rotational constants (cm-1) : 0.41695 0.16011 0.12008
    """
    res_gxtb = extractor.parse_gxtb_output(gxtb_sample)
    assert abs(res_gxtb['rotational_constants']['A'] - 12500.0) < 10.0
    assert res_gxtb['dipole_moment']['mu_a'] == 0.5
    assert res_gxtb['dipole_moment']['mu_b'] == 1.2

def test_spycfit_master_lam_pes_scan():
    from cochem_spycfit_master import SpycFitMaster
    master = SpycFitMaster()
    res = master.execute_lam_correction(r_min=1.5, r_max=3.0, n_steps=15)
    assert "r_optimal_angstrom" in res
    assert 1.5 <= res["r_optimal_angstrom"] <= 3.0
    assert "rotational_constants_MHz" in res
    assert res["rotational_constants_MHz"]["A"] > 0
    assert Path("spycfit_lam_buffer.npz").exists()

def test_rm2_mass_scaling():
    rot = np.array([10000.0, 5000.0, 2500.0])
    symbols = ['H', 'O', 'H']
    res = cochem_convolver.apply_rm2_mass_scaling(rot, symbols)
    assert 'A_scaled_MHz' in res
    assert res['A_scaled_MHz'] > 0

def test_sobol_sensitivity():
    sobol = cochem_convolver.compute_sobol_sensitivity(n_samples=50)
    assert 'first_order_S' in sobol
    assert 'total_order_ST' in sobol
    assert 'A' in sobol['first_order_S']

def test_normal_mode_analysis():
    mw_h = np.eye(9) * 0.01
    symbols = ['H', 'O', 'H']
    nma = cochem_quantum_extractor.perform_normal_mode_analysis(mw_h, symbols)
    assert 'frequencies_cm1' in nma
    assert len(nma['frequencies_cm1']) == 9

def test_nbo_parsing():
    sample_nbo = """
    NATURAL POPULATION ANALYSIS
    ---------------------------
    C 1  -0.345
    H 2   0.115
    H 3   0.115
    H 4   0.115
    ---------------------------
    ==========
    """
    nbo = cochem_quantum_extractor.parse_nbo_analysis(sample_nbo)
    assert 'natural_charges' in nbo
    assert nbo['natural_charges']['C1'] == -0.345


