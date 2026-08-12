from typing import Any, Dict, List, Optional
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

def test_watson_s_reduced_model() -> None:
    jnp = cochem_convolver.jnp
    params = jnp.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])
    freqs = cochem_convolver.watson_s_reduced_model(params)
    assert freqs.shape == (8,)
    assert float(freqs[0]) > 0

def test_compute_frequency_matrix() -> None:
    hessian = np.diag([0.01, 0.04, 0.09])
    freqs = cochem_convolver.compute_frequency_matrix(hessian)
    assert len(freqs) == 3
    np.testing.assert_allclose(freqs[0], 0.1 * 5140.48714, rtol=1e-3)

def test_loio_cross_validation() -> None:
    test_data = np.array([12500.0, 11000.0, 9500.0, 15000.0, 14000.0])
    res = cochem_convolver.loio_cross_validation({}, test_data)
    assert res['n_isotopes'] == 5
    assert 'rmse' in res
    assert 0.0 <= res['r_squared'] <= 1.0

def test_penalty_function() -> None:
    params = np.array([10000.0, 5000.0, 2500.0])
    data = np.array([12500.0, 11000.0, 9500.0, 24000.0, 21500.0, 19000.0, 31000.0, 29500.0])
    pen = cochem_convolver.penalty_function(params, data)
    assert isinstance(pen, float)
    assert pen > 0

def test_mass_weighting_exact() -> None:
    symbols = ['H', 'O', 'H']
    hessian = np.eye(9)
    mw_h = cochem_quantum_extractor.apply_mass_weighting(hessian, symbols)
    assert mw_h.shape == (9, 9)
    expected_o_weight = 1.0 / 15.994915
    np.testing.assert_allclose(mw_h[3, 3], expected_o_weight, rtol=1e-4)

def test_parse_orca_freq_output() -> None:
    sample_orca = """
    Rotational constants (MHz): 10234.56   5123.45   2561.78
    Total Dipole Moment :    1.2345   -0.5678   0.8912
    """
    parsed = cochem_quantum_extractor.parse_orca_freq_output(sample_orca)
    assert parsed['rotational_constants']['A'] == 10234.56
    assert parsed['dipole_moment']['X'] == 1.2345

def test_vpt2_resonance_trap() -> None:
    freqs = np.array([100.0, 201.0, 500.0])
    cubic_k = np.zeros((3, 3, 3))
    cubic_k[0, 0, 1] = 0.5
    res = cochem_quantum_extractor.vpt2_resonance_trap(freqs, cubic_k, threshold=5.0)
    assert len(res) >= 1
    assert res[0][0] == 'Fermi'

def test_pickett_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cochem_quantum_extractor.generate_pickett_files({}, output_dir=tmpdir)
        assert (Path(tmpdir) / "output.lin").exists()
        assert (Path(tmpdir) / "output.cat").exists()
        assert (Path(tmpdir) / "output.fit").exists()

def test_spycfit_init_components() -> None:
    res = cochem_spycfit_init.initialize_spycfit_components()
    assert res["initialized"] is True
    assert "components" in res

def test_quantum_extractor_orca_and_gxtb() -> None:
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

def test_spycfit_master_lam_pes_scan() -> None:
    from cochem_spycfit_master import SpycFitMaster
    master = SpycFitMaster()
    res = master.execute_lam_correction(r_min=1.5, r_max=3.0, n_steps=15)
    assert "r_optimal_angstrom" in res
    assert 1.5 <= res["r_optimal_angstrom"] <= 3.0
    assert "rotational_constants_MHz" in res
    assert res["rotational_constants_MHz"]["A"] > 0
    assert Path("spycfit_lam_buffer.npz").exists()

def test_rm2_mass_scaling() -> None:
    rot = np.array([10000.0, 5000.0, 2500.0])
    symbols = ['H', 'O', 'H']
    res = cochem_convolver.apply_rm2_mass_scaling(rot, symbols)
    assert 'A_scaled_MHz' in res
    assert res['A_scaled_MHz'] > 0

def test_sobol_sensitivity() -> None:
    sobol = cochem_convolver.compute_sobol_sensitivity(n_samples=50)
    assert 'first_order_S' in sobol
    assert 'total_order_ST' in sobol
    assert 'A' in sobol['first_order_S']

def test_normal_mode_analysis() -> None:
    mw_h = np.eye(9) * 0.01
    symbols = ['H', 'O', 'H']
    nma = cochem_quantum_extractor.perform_normal_mode_analysis(mw_h, symbols)
    assert 'frequencies_cm1' in nma
    assert len(nma['frequencies_cm1']) == 9

def test_nbo_parsing() -> None:
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

def test_hybrid_fitting_voigt_and_solvation() -> None:
    target_freqs = np.array([12500.0, 11000.0, 9500.0, 15000.0, 14000.0, 13000.0, 16000.0, 15500.0])
    init_params = np.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])
    
    nm_res = cochem_convolver.nelder_mead_optimization(target_freqs, init_params, max_iter=20)
    assert 'final_params' in nm_res
    
    hyb_res = cochem_convolver.hybrid_spectral_fitting(target_freqs, init_params, max_iter_nm=10, max_iter_lm=10)
    assert 'final_params' in hyb_res

    freq_grid = np.linspace(10000, 15000, 100)
    line_centers = np.array([12000.0, 14000.0])
    intensities = np.array([1.0, 0.5])
    vp = cochem_convolver.voigt_profile(freq_grid, line_centers, intensities, sigma_doppler=5.0, gamma_lorentzian=5.0)
    assert len(vp) == 100
    assert np.max(vp) > 0.0

    rot_dict = {"A": 10000.0, "B": 5000.0, "C": 2500.0}
    dipole_dict = {"mu_a": 1.5, "mu_b": 0.5, "mu_c": 0.0}
    solv = cochem_convolver.apply_polar_solvation_dispersion(rot_dict, dipole_dict, dielectric_constant=78.39)
    assert "f_onsager" in solv
    assert solv["f_onsager"] > 0.0

def test_jensen_inequality_averaging() -> None:
    from cochem_spycfit_master import compute_vibrationally_averaged_constants
    I_sample = np.array([
        [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]],
        [[12.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 32.0]]
    ])
    weights = np.array([0.5, 0.5])
    CONVERSION_FACTOR = 505379.008

    # Correct <1/I> element-wise inverse inertia tensor expectation value
    b0_jensen = compute_vibrationally_averaged_constants(I_sample, weights)

    # Flawed 1/<I> (inverting after averaging)
    avg_I_flawed = np.sum(I_sample * weights[:, None, None], axis=0)
    inv_avg_flawed = np.linalg.inv(avg_I_flawed)
    b0_flawed = CONVERSION_FACTOR * np.sort(np.diag(inv_avg_flawed))[::-1]

    # Verify Jensen's inequality: <1/I> > 1/<I>
    for k in range(3):
        assert b0_jensen[k] > b0_flawed[k], f"Jensen inequality violated at axis {k}: {b0_jensen[k]} <= {b0_flawed[k]}"
    
    bias_mhz = b0_jensen - b0_flawed
    assert np.all(bias_mhz > 0.0)

def test_be_b0_separation() -> None:
    from cochem_spycfit_master import SpycFitMaster
    master = SpycFitMaster()
    res = master.execute_lam_correction(r_min=1.5, r_max=3.0, n_steps=15)
    
    assert "rotational_constants_Be_MHz" in res
    assert "rotational_constants_B0_MHz" in res
    assert "delta_B_vib_MHz" in res
    
    Be = res["rotational_constants_Be_MHz"]
    B0 = res["rotational_constants_B0_MHz"]
    dB = res["delta_B_vib_MHz"]
    
    # Assert B_0 = B_e + Delta B_vib
    np.testing.assert_allclose(B0["A_0"], Be["A_e"] + dB["dA_vib"], rtol=1e-5)
    np.testing.assert_allclose(B0["B_0"], Be["B_e"] + dB["dB_vib"], rtol=1e-5)
    np.testing.assert_allclose(B0["C_0"], Be["C_e"] + dB["dC_vib"], rtol=1e-5)
    
    # Ground-state B_0 > B_e due to inverse inertia vibrational averaging
    assert B0["A_0"] > Be["A_e"]

def test_reweight_isotopologue_hessian() -> None:
    from cochem_spycfit_master import reweight_isotopologue_hessian
    # 3-atom system (H-O-H model) 9x9 Hessian
    symbols = ['H', 'O', 'H']
    parent_hessian = np.eye(9) * 0.05
    
    # Re-analyze with Deuterium substitution on atom 0 (H -> D)
    iso_res = reweight_isotopologue_hessian(parent_hessian, symbols, mass_overrides={0: 2.01410178})
    parent_res = reweight_isotopologue_hessian(parent_hessian, symbols)
    
    assert iso_res["zero_el_cost"] is True
    assert len(iso_res["isotopologue_masses_amu"]) == 3
    assert abs(iso_res["isotopologue_masses_amu"][0] - 2.01410178) < 1e-4
    assert len(iso_res["frequencies_cm1"]) == 9
    
    # Heavy isotope substitution on atom 0 lowers ZPE
    assert iso_res["zpe_cm1"] < parent_res["zpe_cm1"]
    
    # D-O-D double substitution lowers all H mode frequencies
    dod_res = reweight_isotopologue_hessian(parent_hessian, symbols, mass_overrides={0: 2.01410178, 2: 2.01410178})
    assert max(dod_res["frequencies_cm1"]) < max(parent_res["frequencies_cm1"])

def test_orca_1d_pes_input_v4_tight_geom() -> None:
    from cochem_spycfit_master import SpycFitMaster
    master = SpycFitMaster()
    coords = [('H', 0.0, 0.0, 0.0), ('O', 0.0, 0.0, 0.96)]
    
    # Test v4 default tier (T3O-12h junChS)
    inp_junchs = master.generate_orca_1d_pes_input(coords, tier="T3O-12h junChS")
    assert "! junChS TightOPT TightSCF" in inp_junchs
    assert "TolE     1e-7" in inp_junchs
    assert "TolRMSG  3e-6" in inp_junchs
    assert "TolMaxG  1e-5" in inp_junchs
    assert "TolRMSD  5e-5" in inp_junchs
    assert "TolMaxD  1e-4" in inp_junchs
    
    # Test alternative tier (T1-30min r²SCAN-3c)
    inp_r2scan = master.generate_orca_1d_pes_input(coords, tier="T1-30min r²SCAN-3c")
    assert "! r2SCAN-3c TightOPT TightSCF" in inp_r2scan
    assert "TolMaxG  1e-5" in inp_r2scan

def test_pas_dipole_projections() -> None:
    from cochem_spycfit_init import compute_pas_dipole_projections
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.8, 0.0, -0.2]]
    symbols = ['O', 'H', 'H']
    dipole_vec = [1.2, 0.5, -0.8]
    
    pas_dip = compute_pas_dipole_projections(coords, symbols, dipole_vec)
    assert "mu_a" in pas_dip
    assert "mu_b" in pas_dip
    assert "mu_c" in pas_dip
    assert "total_mu" in pas_dip
    
    # Norm of projected components must match total dipole moment
    proj_norm = np.sqrt(pas_dip["mu_a"]**2 + pas_dip["mu_b"]**2 + pas_dip["mu_c"]**2)
    np.testing.assert_allclose(proj_norm, pas_dip["total_mu"], rtol=1e-4)

def test_2h_and_iupac_isotope_resolution() -> None:
    from cochem_spycfit_master import reweight_isotopologue_hessian
    parent_hessian = np.eye(9) * 0.05
    
    # Test '2H' symbol resolution specifically
    iso_2h = reweight_isotopologue_hessian(parent_hessian, symbols=['H', 'O', 'H'], new_symbols=['2H', '16O', '2H'])
    np.testing.assert_allclose(iso_2h["isotopologue_masses_amu"][0], 2.01410178, rtol=1e-6)
    np.testing.assert_allclose(iso_2h["isotopologue_masses_amu"][1], 15.994915, rtol=1e-6)
    
    # Test comprehensive set of IUPAC isotope strings
    iupac_symbols = ['1H', '2H', '2D', '3D', '12C', '13C', '14C', '14N', '15N', '16O', '17O', '18O', '35Cl', '37Cl', '79Br', '81Br']
    expected_masses = [1.007825, 2.01410178, 2.01410178, 3.016049, 12.000000, 13.003355, 14.003242, 14.003074, 15.000109, 15.994915, 16.999131, 17.999160, 34.968853, 36.965903, 78.918337, 80.916290]
    
    dummy_hessian = np.eye(3 * len(iupac_symbols)) * 0.05
    res = reweight_isotopologue_hessian(dummy_hessian, symbols=iupac_symbols)
    np.testing.assert_allclose(res["isotopologue_masses_amu"], expected_masses, rtol=1e-6)

def test_regex_parser_fallback_for_mass_prefix() -> None:
    from cochem_spycfit_master import reweight_isotopologue_hessian
    # Test symbols with mass prefixes not in explicit mass_map
    unmapped_symbols = ['34S', '100Hg', '200X']
    dummy_hessian = np.eye(3 * len(unmapped_symbols)) * 0.05
    res = reweight_isotopologue_hessian(dummy_hessian, symbols=unmapped_symbols)
    
    np.testing.assert_allclose(res["isotopologue_masses_amu"][0], 34.0, rtol=1e-6)
    np.testing.assert_allclose(res["isotopologue_masses_amu"][1], 100.0, rtol=1e-6)
    np.testing.assert_allclose(res["isotopologue_masses_amu"][2], 200.0, rtol=1e-6)

def test_imaginary_frequency_flagging() -> None:
    from cochem_spycfit_master import reweight_isotopologue_hessian
    # Create a Hessian with negative force constants to produce imaginary frequencies (eigvals < 0)
    hessian_imag = np.diag([-0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    res = reweight_isotopologue_hessian(hessian_imag, symbols=['H', 'H'])
    
    assert res["has_imaginary_frequencies"] is True
    assert res["imaginary_frequencies_count"] == 1
    assert len(res["imaginary_frequencies"]) == 1
    assert res["imaginary_frequencies"][0] < 0.0
    assert res["frequencies_cm1"][0] < 0.0

def test_right_handed_pas_frame_determinant_enforcement() -> None:
    from cochem_spycfit_init import compute_pas_dipole_projections
    
    # Test across multiple asymmetric coordinate geometries
    geom_list = [
        ([[0.0, 0.0, 0.0], [1.0, 0.2, -0.5], [-0.5, 1.5, 0.3], [0.2, -0.8, 1.2]], ['O', 'C', 'H', 'H']),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.8, 0.0, -0.2]], ['O', 'H', 'H']),
        ([[1.2, -0.3, 0.4], [-0.8, 1.1, -0.2], [0.5, -1.2, 0.9]], ['C', 'N', 'O']),
    ]
    dipole_vec = [1.5, -0.8, 2.1]
    
    for coords, symbols in geom_list:
        pas_res = compute_pas_dipole_projections(coords, symbols, dipole_vec)
        det = pas_res["pas_axes_det"]
        np.testing.assert_allclose(det, 1.0, atol=1e-6)
        assert det > 0.0, f"PAS axes determinant must be positive (+1), got {det}"
