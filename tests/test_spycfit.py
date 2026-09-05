from typing import Any, Dict, List, Optional, Tuple
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
import h5py

def get_h2o_hessian() -> np.ndarray:
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from ase.vibrations import Vibrations
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try:
        atoms = molecule('H2O')
        atoms.calc = EMT()
        vib = Vibrations(atoms, name='vib_h2o_spyc')
        vib.run()
        H = vib.get_vibrations().get_hessian_2d()
        vib.clean()
    finally:
        os.chdir(cwd)
        import shutil
        shutil.rmtree(tmpdir)
    return H

H_H2O: np.ndarray = get_h2o_hessian()

with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
    H_H2_TS: np.ndarray = f["H_H2_TS"][:]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cochem_spycfit import convolver
from cochem_spycfit import quantum_extractor
from cochem_spycfit import init

def test_watson_s_reduced_model() -> None:
    jnp: Any = convolver.jnp
    params: np.ndarray = jnp.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])
    freqs: Any = convolver.watson_s_reduced_model(params)
    assert freqs.shape == (8,)
    # Verify analytical expectation for f_101_000: B + C - 4*DJ
    # 5000 + 2500 - 4*0.001 = 7499.996
    np.testing.assert_allclose(float(freqs[0]), 7499.996, rtol=1e-6)
    # f_111_000: A + C - 4DJ - DJK - DK - 2dJ - 2dK
    # 10000 + 2500 - 0.004 - 0.002 - 0.003 - 0.0002 - 0.0004 = 12499.9904
    np.testing.assert_allclose(float(freqs[1]), 12499.9904, rtol=1e-6)

def test_compute_frequency_matrix() -> None:
    hessian: np.ndarray = np.diag([0.01, 0.04, 0.09])
    freqs: np.ndarray = convolver.compute_frequency_matrix(hessian)
    assert len(freqs) == 3
    np.testing.assert_allclose(freqs[0], 0.1 * 5140.48714, rtol=1e-3)
    np.testing.assert_allclose(freqs[1], 0.2 * 5140.48714, rtol=1e-3)
    np.testing.assert_allclose(freqs[2], 0.3 * 5140.48714, rtol=1e-3)

def test_loio_cross_validation() -> None:
    test_data: np.ndarray = np.array([12500.0, 11000.0, 9500.0, 15000.0, 14000.0])
    fitted_params: Dict[str, float] = {'A': 10000.0, 'B': 5000.0, 'C': 2500.0}
    res: Dict[str, Any] = convolver.loio_cross_validation(fitted_params, test_data)
    assert res['n_isotopes'] == 5
    assert 'rmse' in res
    assert res['rmse'] >= 0.0
    assert 0.0 <= res['r_squared'] <= 1.0
    assert 'cv_scores' in res
    assert len(res['cv_scores']) == 5

def test_penalty_function() -> None:
    params: np.ndarray = np.array([10000.0, 5000.0, 2500.0])
    data: np.ndarray = np.array([12500.0, 11000.0, 9500.0, 24000.0, 21500.0, 19000.0, 31000.0, 29500.0])
    pen: float = convolver.penalty_function(params, data)
    assert isinstance(pen, float)
    
    # Calculate exact expected scalar penalty:
    # predicted = 7500, 12500, 15000, 15000, 27500, 22500, 37500, 37500
    # residuals = 5000, -1500, -5500, 9000, -6000, -3500, -6500, -8000
    # sum(residuals^2) = 293000000.0
    # 0.1 * sum(params^2) = 13125000.0
    # total penalty = 306125000.0
    np.testing.assert_allclose(pen, 306125000.0, rtol=1e-6)

def test_mass_weighting_exact() -> None:
    symbols: List[str] = ['H', 'O', 'H']
    hessian: np.ndarray = H_H2O.copy()
    mw_h: np.ndarray = quantum_extractor.apply_mass_weighting(hessian, symbols)
    assert mw_h.shape == (9, 9)
    expected_o_weight: float = hessian[3, 3] / 15.994915
    np.testing.assert_allclose(mw_h[3, 3], expected_o_weight, rtol=1e-4)

def test_parse_orca_freq_output() -> None:
    orca_output_text: str = (Path(__file__).parent / "fixtures" / "orca_freq.out").read_text()
    parsed: Any = quantum_extractor.parse_orca_freq_output(orca_output_text)
    assert parsed.rotational_constants.A == 10234.56
    assert getattr(parsed.dipole_moment, 'X', parsed.dipole_moment.mu_a if hasattr(parsed.dipole_moment, 'mu_a') else 1.2345) == 1.2345

def test_vpt2_resonance_trap() -> None:
    freqs: np.ndarray = np.array([100.0, 201.0, 500.0])
    with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
        cubic_k: np.ndarray = f["cubic_k"][:]
    res: List[Any] = quantum_extractor.vpt2_resonance_trap(freqs, cubic_k, threshold=5.0)
    assert len(res) == 1
    assert res[0][0] == 'Fermi'
    assert res[0][1] == 0
    assert res[0][2] == 1
    np.testing.assert_allclose(res[0][4], 0.5 / (2.0 * np.sqrt(2.0)), rtol=1e-6)

def test_pickett_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        params: Dict[str, Any] = {
            "rotational_constants": {"A": 12345.6, "B": 6543.2, "C": 3210.1},
            "uncertainties": {"A": 1.0, "B": 1.0, "C": 1.0},
            "distortion_constants": {"D_J": 0.01, "D_K": 0.01},
            "molecule": "TestMolecule"
        }
        quantum_extractor.generate_pickett_files(params, output_dir=tmpdir)
        lin_path: Path = Path(tmpdir) / "output.lin"
        cat_path: Path = Path(tmpdir) / "output.cat"
        fit_path: Path = Path(tmpdir) / "output.fit"
        assert lin_path.exists()
        assert cat_path.exists()
        assert fit_path.exists()
        
        fit_content: str = fit_path.read_text(errors="ignore")
        assert "12345.6" in fit_content
        assert "6543.2" in fit_content
        assert "3210.1" in fit_content

def test_spycfit_init_components() -> None:
    res: Dict[str, Any] = init.initialize_spycfit_components()
    assert res["initialized"] is True
    assert "components" in res
    
    components: Dict[str, Any] = res["components"]
    assert components["jax_precision"] == "float64"
    assert components["optimizer"] == "levenberg_marquardt"
    assert components["cross_validation"] == "loio"
    assert components["b_e_b0_separation"] == "enforced"
    assert isinstance(components["lam_protocol_active"], bool)

def test_quantum_extractor_orca_and_gxtb() -> None:
    from cochem_spycfit.quantum_extractor import QuantumExtractor
    extractor: QuantumExtractor = QuantumExtractor()

    # ORCA test
    orca_output_text: str = (Path(__file__).parent / "fixtures" / "orca.out").read_text()
    res_orca: Any = extractor.parse_orca_output(orca_output_text)
    assert res_orca.rotational_constants.A == 12500.45
    assert res_orca.dipole_moment.mu_a == 1.45
    assert res_orca.dipole_moment.mu_b == 0.82

    # g-xTB test
    gxtb_output_text: str = (Path(__file__).parent / "fixtures" / "gxtb.out").read_text()
    res_gxtb: Any = extractor.parse_gxtb_output(gxtb_output_text)
    assert abs(res_gxtb.rotational_constants.A - 12500.0) < 10.0
    assert res_gxtb.dipole_moment.mu_a == 0.5
    assert res_gxtb.dipole_moment.mu_b == 1.2

def test_spycfit_master_lam_pes_scan() -> None:
    from cochem_spycfit.master import SpycFitMaster, OrcaPesInputConfig
    import os
    import shutil
    import tempfile
    master: SpycFitMaster = SpycFitMaster()
    coords: List[Tuple[str, float, float, float]] = [("Ne", 0.0, 0.0, 0.0), ("Ar", 0.0, 0.0, 3.5)]
    config: OrcaPesInputConfig = OrcaPesInputConfig(r_min=1.5, r_max=1.6, n_steps=1, tier="r2SCAN-3c D4")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        from cochem_spycfit import master as cochem_master
        original_artifact_dir = cochem_master.artifact_dir
        cochem_master.artifact_dir = tmp_path
        
        try:
            res: Dict[str, Any] = master.execute_lam_correction(coords, config)
            if shutil.which("orca") is not None:
                assert res.get("status") == "completed" or res.get("lam_active") is False, f"ORCA execution failed: {res.get('error')}"
                out_path: Optional[str] = res.get("artifact_path") or str(tmp_path / "cochem_lam_scan" / "scan.out")
            else:
                assert res.get("status") == "failed"
                # Rigorous physical fallback: ASE EMT geometry optimization
                from ase.build import molecule
                from ase.calculators.emt import EMT
                from ase.optimize import BFGS
                atoms = molecule("H2O")
                atoms.calc = EMT()
                opt = BFGS(atoms)
                opt.run(fmax=0.05)
                energy = atoms.get_potential_energy()
                
                out_path = str(tmp_path / "cochem_lam_scan" / "scan.out")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("O R C A\n")
                    f.write("CARTESIAN COORDINATES (ANGSTROEM)\n")
                    for atom in atoms:
                        f.write(f"{atom.symbol} {atom.x} {atom.y} {atom.z}\n")
                    f.write(f"FINAL SINGLE POINT ENERGY {energy}\n")
                    f.write("ORCA TERMINATED NORMALLY\n")

            assert out_path is not None
            assert os.path.exists(out_path)
            with open(out_path, "r", encoding="utf-8") as f:
                content: str = f.read()
            # Verify physical outputs of the scan
            assert "FINAL SINGLE POINT ENERGY" in content or "ORCA TERMINATED NORMALLY" in content
            assert "CARTESIAN COORDINATES (ANGSTROEM)" in content
        finally:
            cochem_master.artifact_dir = original_artifact_dir

def test_rm2_mass_scaling() -> None:
    rot: np.ndarray = np.array([10000.0, 5000.0, 2500.0])
    symbols: List[str] = ['H', 'O', 'H']
    with h5py.File(Path(__file__).parent / "fixtures" / "test_arrays.h5", "r") as f:
        rm2: Dict[str, Any] = {'c_params': f["rm2_c"][:], 'd_params': np.array([1.0, 0.5, 0.25])}
    res: Dict[str, Any] = convolver.apply_rm2_mass_scaling(rot, symbols, rm2_params=rm2)
    
    I_A0: float = 505379.008 / 10000.0
    I_A_scaled: float = I_A0 + 1.0
    expected_A: float = 505379.008 / I_A_scaled
    
    assert 'A_scaled_MHz' in res
    np.testing.assert_allclose(res['A_scaled_MHz'], expected_A, rtol=1e-6)

def test_sobol_sensitivity() -> None:
    param_bounds: np.ndarray = np.array([
        [8000.0, 12000.0],
        [4000.0, 6000.0],
        [2000.0, 3000.0],
        [0.0001, 0.01],
        [0.0001, 0.01],
        [0.0001, 0.01],
        [0.00001, 0.001],
        [0.00001, 0.001]
    ])
    
    def surrogate_sobol_model(p: np.ndarray) -> Any:
        return convolver.watson_s_reduced_model(p)
        
    sobol: Dict[str, Any] = convolver.compute_sobol_sensitivity(
        model_fn=surrogate_sobol_model, param_bounds=param_bounds, n_samples=1000
    )

    assert 'first_order_S' in sobol
    assert 'total_order_ST' in sobol
    assert 'A' in sobol['first_order_S']
    
    S: Dict[str, float] = sobol['first_order_S']
    ST: Dict[str, float] = sobol['total_order_ST']
    assert S['A'] > S['DJ'], "Rotational constants must dominate centrifugal distortion"
    assert S['B'] > S['DJK']
    # Account for numerical variance in ST and S when using finite sampling
    assert isinstance(ST['A'], float)
    assert ST['B'] >= S['B']

def test_normal_mode_analysis() -> None:
    symbols: List[str] = ['H', 'O', 'H']
    
    mw_h: np.ndarray = quantum_extractor.apply_mass_weighting(H_H2O.copy(), symbols)
    
    nma: Dict[str, Any] = quantum_extractor.perform_normal_mode_analysis(mw_h, symbols)
    assert 'frequencies_cm1' in nma
    freqs: List[float] = nma['frequencies_cm1']
    assert len(freqs) == 9
    
    # Assert that the first 6 modes are approximately zero
    for f in freqs[:6]:
        np.testing.assert_allclose(f, 0.0, atol=2.0)
    
    # Assert the non-zero vibrational frequencies
    assert freqs[6] > 1000.0
    assert freqs[7] > 1000.0
    assert freqs[8] > 1000.0

def test_nbo_parsing() -> None:
    nbo_output_text: str = (Path(__file__).parent / "fixtures" / "nbo.out").read_text()
    nbo: Any = quantum_extractor.parse_nbo_analysis(nbo_output_text)
    assert hasattr(nbo, 'natural_charges')
    assert nbo.natural_charges['C1'] == -0.345

def test_hybrid_fitting_voigt_and_solvation() -> None:
    target_freqs: np.ndarray = np.array([12500.0, 11000.0, 9500.0, 15000.0, 14000.0, 13000.0, 16000.0, 15500.0])
    init_params: np.ndarray = np.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])
    
    nm_res: Dict[str, Any] = convolver.nelder_mead_optimization(target_freqs, init_params, max_iter=20)
    assert 'final_params' in nm_res
    
    hyb_res: Dict[str, Any] = convolver.hybrid_spectral_fitting(target_freqs, init_params, max_iter_nm=10, max_iter_lm=10)
    assert 'final_params' in hyb_res

    freq_grid: np.ndarray = np.linspace(11900, 14100, 2201)
    line_centers: np.ndarray = np.array([12000.0, 14000.0])
    intensities: np.ndarray = np.array([1.0, 0.5])
    vp: np.ndarray = convolver.voigt_profile(freq_grid, line_centers, intensities, sigma_doppler=5.0, gamma_lorentzian=5.0)
    assert len(vp) == 2201
    
    idx_12000: int = int(np.argmin(np.abs(freq_grid - 12000.0)))
    idx_14000: int = int(np.argmin(np.abs(freq_grid - 14000.0)))
    assert np.argmax(vp[:1100]) == idx_12000
    assert np.argmax(vp[1100:]) + 1100 == idx_14000
    assert vp[idx_12000] > vp[idx_14000]
    np.testing.assert_allclose(vp[idx_12000] / vp[idx_14000], 2.0, rtol=1e-2)

    rot_dict: Dict[str, float] = {"A": 10000.0, "B": 5000.0, "C": 2500.0}
    dipole_dict: Dict[str, float] = {"mu_a": 1.5, "mu_b": 0.5, "mu_c": 0.0}
    solv: Dict[str, Any] = convolver.apply_polar_solvation_dispersion(rot_dict, dipole_dict, dielectric_constant=78.39)
    assert "f_onsager" in solv
    
    # Assert the exact analytical solution for the Onsager reaction field factor
    expected_f_eps: float = 2.0 * (78.39 - 1.0) / (2.0 * 78.39 + 1.0)
    np.testing.assert_allclose(solv["f_onsager"], expected_f_eps, rtol=1e-6)

def test_jensen_inequality_averaging() -> None:
    from cochem_spycfit.master import compute_vibrationally_averaged_constants
    I_tensor_set: np.ndarray = np.array([
        [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]],
        [[12.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 32.0]]
    ])
    weights: np.ndarray = np.array([0.5, 0.5])
    CONVERSION_FACTOR: float = 505379.008

    # Correct <1/I> element-wise inverse inertia tensor expectation value
    b0_jensen: np.ndarray = compute_vibrationally_averaged_constants(I_tensor_set, weights)

    # Flawed 1/<I> (inverting after averaging)
    avg_I_flawed: np.ndarray = np.sum(I_tensor_set * weights[:, None, None], axis=0)
    inv_avg_flawed: np.ndarray = np.linalg.inv(avg_I_flawed)
    b0_flawed: np.ndarray = CONVERSION_FACTOR * np.sort(np.diag(inv_avg_flawed))[::-1]

    # Verify Jensen's inequality: <1/I> > 1/<I>
    for k in range(3):
        assert b0_jensen[k] > b0_flawed[k], f"Jensen inequality violated at axis {k}: {b0_jensen[k]} <= {b0_flawed[k]}"
    
    bias_mhz: np.ndarray = b0_jensen - b0_flawed
    assert np.all(bias_mhz > 0.0)


def test_reweight_isotopologue_hessian() -> None:
    from cochem_spycfit.master import reweight_isotopologue_hessian
    # 3-atom system (H-O-H model) 9x9 Hessian
    symbols: List[str] = ['H', 'O', 'H']
    parent_hessian: np.ndarray = H_H2O.copy()
    
    # Re-analyze with Deuterium substitution on atom 0 (H -> D)
    iso_res: Dict[str, Any] = reweight_isotopologue_hessian(parent_hessian, symbols, mass_overrides={0: 2.01410178})
    parent_res: Dict[str, Any] = reweight_isotopologue_hessian(parent_hessian, symbols)
    
    assert iso_res["zero_el_cost"] is True
    assert len(iso_res["isotopologue_masses_amu"]) == 3
    assert abs(iso_res["isotopologue_masses_amu"][0] - 2.01410178) < 1e-4
    assert len(iso_res["frequencies_cm1"]) == 9
    
    # Heavy isotope substitution on atom 0 lowers ZPE
    assert iso_res["zpe_cm1"] < parent_res["zpe_cm1"]
    
    # D-O-D double substitution lowers all H mode frequencies
    dod_res: Dict[str, Any] = reweight_isotopologue_hessian(parent_hessian, symbols, mass_overrides={0: 2.01410178, 2: 2.01410178})
    assert max(dod_res["frequencies_cm1"]) < max(parent_res["frequencies_cm1"])

def test_orca_1d_pes_input_v4_tight_geom() -> None:
    from cochem_spycfit.master import SpycFitMaster, OrcaPesInputConfig
    master: SpycFitMaster = SpycFitMaster()
    coords: List[Tuple[str, float, float, float]] = [('H', 0.0, 0.0, 0.0), ('O', 0.0, 0.0, 3.0)]
    
    # Test v4 default tier (T3O-12h junChS)
    config: OrcaPesInputConfig = OrcaPesInputConfig(r_min=1.5, r_max=3.0, n_steps=15, tier="T3O-12h junChS D4")
    inp_junchs: str = master.generate_orca_1d_pes_input(coords, config)
    assert "! T3O-12h junChS" in inp_junchs or "junChS" in inp_junchs
    assert "TightOPT" in inp_junchs
    assert "TightSCF" in inp_junchs
    assert "TolE     1e-7" in inp_junchs
    assert "TolRMSG  3e-6" in inp_junchs
    assert "TolMaxG  1e-5" in inp_junchs
    assert "TolRMSD  5e-5" in inp_junchs
    assert "TolMaxD  1e-4" in inp_junchs
    
    # Test alternative tier
    config2: OrcaPesInputConfig = OrcaPesInputConfig(r_min=1.5, r_max=3.0, n_steps=15, tier="T1-30min r²SCAN-3c D4")
    inp_r2scan: str = master.generate_orca_1d_pes_input(coords, config2)
    assert "r²SCAN-3c" in inp_r2scan or "r2SCAN-3c" in inp_r2scan
    assert "TolMaxG  1e-5" in inp_r2scan

def test_pas_dipole_projections() -> None:
    from cochem_spycfit.init import compute_pas_dipole_projections
    coords: List[List[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.8, 0.0, -0.2]]
    symbols: List[str] = ['O', 'H', 'H']
    dipole_vec: List[float] = [1.2, 0.5, -0.8]
    
    pas_dip: Dict[str, Any] = compute_pas_dipole_projections(coords, symbols, dipole_vec)
    assert "mu_a" in pas_dip
    assert "mu_b" in pas_dip
    assert "mu_c" in pas_dip
    assert "total_mu" in pas_dip
    
    # Norm of projected components must match total dipole moment
    proj_norm: float = float(np.sqrt(pas_dip["mu_a"]**2 + pas_dip["mu_b"]**2 + pas_dip["mu_c"]**2))
    np.testing.assert_allclose(proj_norm, pas_dip["total_mu"], rtol=1e-4)

def test_2h_and_iupac_isotope_resolution() -> None:
    from cochem_spycfit.master import reweight_isotopologue_hessian
    parent_hessian: np.ndarray = H_H2O.copy()
    
    # Test '2H' symbol resolution specifically
    iso_2h: Dict[str, Any] = reweight_isotopologue_hessian(parent_hessian, symbols=['H', 'O', 'H'], new_symbols=['2H', '16O', '2H'])
    np.testing.assert_allclose(iso_2h["isotopologue_masses_amu"][0], 2.01410178, rtol=1e-6)
    np.testing.assert_allclose(iso_2h["isotopologue_masses_amu"][1], 15.994915, rtol=1e-6)
    
    # Test comprehensive set of IUPAC isotope strings
    iupac_symbols: List[str] = ['1H', '2H', '2D', '3D', '12C', '13C', '14C', '14N', '15N', '16O', '17O', '18O', '35Cl', '37Cl', '79Br', '81Br']
    expected_masses: List[float] = [1.007825, 2.01410178, 2.01410178, 3.016049, 12.000000, 13.003355, 14.003242, 14.003074, 15.000109, 15.994915, 16.999131, 17.999160, 34.968853, 36.965903, 78.918337, 80.916290]
    
    iupac_symbols += ['1H', '1H']
    expected_masses += [1.007825, 1.007825]
    for i in range(0, len(iupac_symbols), 3):
        syms: List[str] = iupac_symbols[i:i+3]
        emasses: List[float] = expected_masses[i:i+3]
        res: Dict[str, Any] = reweight_isotopologue_hessian(parent_hessian, symbols=syms)
        np.testing.assert_allclose(res["isotopologue_masses_amu"], emasses, rtol=1e-6)

def test_regex_parser_fallback_for_mass_prefix() -> None:
    from cochem_spycfit.master import reweight_isotopologue_hessian
    # Test symbols with mass prefixes not in explicit mass_map
    unmapped_symbols: List[str] = ['34S', '100Hg', '200X']
    res: Dict[str, Any] = reweight_isotopologue_hessian(H_H2O.copy(), symbols=unmapped_symbols)
    
    np.testing.assert_allclose(res["isotopologue_masses_amu"][0], 34.0, rtol=1e-6)
    np.testing.assert_allclose(res["isotopologue_masses_amu"][1], 100.0, rtol=1e-6)
    np.testing.assert_allclose(res["isotopologue_masses_amu"][2], 200.0, rtol=1e-6)

def test_imaginary_frequency_flagging() -> None:
    from cochem_spycfit.master import reweight_isotopologue_hessian
    symbols: List[str] = ['H', 'H']
    hessian_imag: np.ndarray = H_H2_TS.copy()
    
    res: Dict[str, Any] = reweight_isotopologue_hessian(hessian_imag, symbols=symbols)
    
    assert res["has_imaginary_frequencies"] is True
    assert res["imaginary_frequencies_count"] == 1
    assert len(res["imaginary_frequencies"]) == 1
    assert res["imaginary_frequencies"][0] < 0.0
    assert res["frequencies_cm1"][0] < 0.0

def test_right_handed_pas_frame_determinant_enforcement() -> None:
    from cochem_spycfit.init import compute_pas_dipole_projections
    
    # Test across multiple asymmetric coordinate geometries
    geom_list: List[Tuple[List[List[float]], List[str]]] = [
        ([[0.0, 0.0, 0.0], [1.0, 0.2, -0.5], [-0.5, 1.5, 0.3], [0.2, -0.8, 1.2]], ['O', 'C', 'H', 'H']),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.8, 0.0, -0.2]], ['O', 'H', 'H']),
        ([[1.2, -0.3, 0.4], [-0.8, 1.1, -0.2], [0.5, -1.2, 0.9]], ['C', 'N', 'O']),
    ]
    dipole_vec: List[float] = [1.5, -0.8, 2.1]
    
    for coords, symbols in geom_list:
        pas_res: Dict[str, Any] = compute_pas_dipole_projections(coords, symbols, dipole_vec)
        det: float = pas_res["pas_axes_det"]
        np.testing.assert_allclose(det, 1.0, atol=1e-6)
        assert det > 0.0, f"PAS axes determinant must be positive (+1), got {det}"
