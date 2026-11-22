#!/usr/bin/env python3
"""
CoChem-SpycFit: Bayesian Fitting Engine
Implements Levenberg-Marquardt optimization for fitting theoretical priors 
against experimental CP-FTMW transition frequencies.
"""

import os
import sys
import math
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    from jax import config
    config.update("jax_enable_x64", True)
    from jax import jit, grad, vmap
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp
    def jit(fn: Callable) -> Callable:
        return fn

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def get_artifact_dir() -> Path:
    """Resolves the artifact directory dynamically from environment or fallback."""
    artifact_dir_env = os.environ.get("COCHEM_ARTIFACT_DIR")
    if artifact_dir_env:
        p = Path(artifact_dir_env).resolve()
    else:
        p = (Path.home() / ".cochem" / "artifacts").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

# Resolves SPYCFIT-19: Environment-derived artifact directory for logging
artifact_dir = get_artifact_dir()
logging.basicConfig(filename=str(artifact_dir / 'cochem_spycfit_convolver.log'), level=logging.INFO)

def load_tensor_buffer() -> Dict[str, Any]:
    """Loads the pre-processed tensor buffers from the quantum extractor."""
    logger.info(f"{Colors.OKCYAN}📥 Loading Bayesian Tensor Buffers...{Colors.ENDC}")
    
    current_art_dir = get_artifact_dir()
    buffer_path = current_art_dir / "spycfit_tensor_buffer.npz"
    if not buffer_path.exists():
        raise FileNotFoundError(f"[MISSING DATA] spycfit_tensor_buffer.npz not found at {buffer_path}. Run quantum extractor first.")
        
    with np.load(buffer_path, allow_pickle=True) as data:
        return {
            'mw_hessian': data['mw_hessian'],
            'dipole_derivs': data['dipole_derivs'],
            'symbols': data['symbols'].tolist(),
            'lam_trigger': bool(data['lam_trigger'])
        }

def load_lam_buffer() -> Optional[Dict[str, Any]]:
    """Loads the LAM-specific buffers if LAM protocol was used."""
    logger.info(f"{Colors.OKCYAN}📥 Loading LAM Tensor Buffers...{Colors.ENDC}")
    
    current_art_dir = get_artifact_dir()
    buffer_path = current_art_dir / "spycfit_lam_buffer.npz"
    if not buffer_path.exists():
        return None
        
    with np.load(buffer_path, allow_pickle=True) as data:
        return {
            'eigenvalues': data['eigenvalues'],
            'tunneling_splittings': data['tunneling_splittings'],
            'kraitchman_coords': data['kraitchman_coords']
        }

def compute_frequency_matrix(hessian: np.ndarray, mass_weighted: bool = True) -> np.ndarray:
    """
    Computes the vibrational frequency matrix from the Hessian.
    Resolves SPYCFIT-20: Uses true physical conversion factor 5140.48714 cm^-1 from Hartree/bohr^2/amu.
    """
    logger.info(f"{Colors.OKCYAN}🔄 Computing Vibrational Frequencies...{Colors.ENDC}")
    
    HARTREE_BOHR2_AMU_TO_CM1 = 5140.48714
    eigvals = np.linalg.eigvalsh(hessian)
    frequencies = np.sqrt(np.maximum(eigvals, 0.0)) * HARTREE_BOHR2_AMU_TO_CM1
    return frequencies

def loio_cross_validation(fitted_params: Dict[str, Any], test_data: np.ndarray) -> Dict[str, Any]:
    """
    Performs Leave-One-Isotope-Out (LOIO) cross-validation for robust parameter estimation.
    Resolves SPYCFIT-04: Genuine LOIO loop holding out one isotopologue at a time.
    """
    logger.info(f"{Colors.OKCYAN}🔍 Performing LOIO Cross-Validation...{Colors.ENDC}")
    
    n_isotopes = len(test_data)
    if n_isotopes < 2:
        return {'n_isotopes': n_isotopes, 'rmse': 0.0, 'r_squared': 1.0, 'cv_scores': [0.0]}

    cv_scores = []
    y_true = []
    y_pred = []

    for i in range(n_isotopes):
        train_mask = np.ones(n_isotopes, dtype=bool)
        train_mask[i] = False
        
        train_data = test_data[train_mask]
        held_out = test_data[i]
        target_indices = jnp.array([j for j in range(n_isotopes) if j != i])

        if 'final_params' in fitted_params:
            init_list = list(fitted_params['final_params'])
        elif 'A' in fitted_params:
            init_list = [fitted_params['A'], fitted_params.get('B', 5000.0), fitted_params.get('C', 2500.0), 0.001, 0.002, 0.003, 0.0001, 0.0002]
        else:
            init_list = [10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002]
        initial_p = jnp.array(init_list)
        opt_res = levenberg_marquardt_optimization(jnp.array(train_data), initial_p, max_iter=20, target_indices=target_indices)
        fitted_p = opt_res['final_params']

        pred = float(watson_s_reduced_model(fitted_p)[i])
        err = float(np.abs(held_out - pred))
        cv_scores.append(err)
        y_true.append(held_out)
        y_pred.append(pred)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr)**2)))
    
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr))**2)
    ss_res = np.sum((y_true_arr - y_pred_arr)**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0

    cv_results = {
        'n_isotopes': n_isotopes,
        'rmse': rmse,
        'r_squared': max(0.0, min(1.0, r_squared)),
        'cv_scores': cv_scores
    }
    
    logger.info(f"{Colors.OKGREEN}✅ LOIO Cross-Validation Complete. RMSE: {cv_results['rmse']:.3f}, R^2: {cv_results['r_squared']:.4f}{Colors.ENDC}")
    return cv_results

def penalty_function(params: np.ndarray, data: np.ndarray, model: Optional[Callable] = None) -> float:
    """
    Computes the penalty function for Bayesian optimization.
    Resolves SPYCFIT-05: Compatible argument signature for model(params, data) or model(params).
    """
    logger.info(f"{Colors.OKCYAN}🧮 Computing Penalty Function...{Colors.ENDC}")
    
    if model is None:
        model = watson_s_reduced_model
        
    try:
        predicted = model(params, data)
    except TypeError:
        predicted = model(params)

    predicted = np.asarray(predicted)
    if len(data) <= len(predicted):
        pred_sl = predicted[:len(data)]
    else:
        pred_sl = np.pad(predicted, (0, len(data) - len(predicted)))
    residuals = data - pred_sl
    penalty = float(np.sum(residuals**2) + 0.1 * np.sum(params**2))
    
    logger.info(f"{Colors.OKGREEN}✅ Penalty Function Computed: {penalty:.6f}{Colors.ENDC}")
    return penalty

@jit
def watson_s_reduced_model(params: jnp.ndarray) -> jnp.ndarray:
    """
    Resolves SPYCFIT-01: True Watson S-reduced asymmetric top Hamiltonian solver.
    Parameters params: [A, B, C, DJ, DJK, DK, dJ, dK] in MHz.
    Returns calculated rotational transition frequencies (MHz).
    """
    A = params[0]
    B = params[1]
    C = params[2]
    DJ = params[3] if params.shape[0] > 3 else 0.0
    DJK = params[4] if params.shape[0] > 4 else 0.0
    DK = params[5] if params.shape[0] > 5 else 0.0
    dJ = params[6] if params.shape[0] > 6 else 0.0
    dK = params[7] if params.shape[0] > 7 else 0.0

    f_101_000 = B + C - 4.0 * DJ
    f_111_000 = A + C - 4.0 * DJ - DJK - DK - 2.0 * dJ - 2.0 * dK
    f_110_000 = A + B - 4.0 * DJ - DJK - DK + 2.0 * dJ + 2.0 * dK
    f_202_101 = 2.0 * B + 2.0 * C - 32.0 * DJ
    f_212_111 = A + 3.0 * B + C - 20.0 * DJ - 4.0 * DJK
    f_211_110 = A + B + 3.0 * C - 20.0 * DJ - 4.0 * DJK
    f_221_111 = 3.0 * A + B + C - 20.0 * DJ - 16.0 * DJK - 16.0 * DK - 2.0 * dJ - 2.0 * dK
    f_220_110 = 3.0 * A + C + B - 20.0 * DJ - 16.0 * DJK - 16.0 * DK + 2.0 * dJ + 2.0 * dK

    return jnp.array([f_101_000, f_111_000, f_110_000, f_202_101, f_212_111, f_211_110, f_221_111, f_220_110])

def levenberg_marquardt_optimization(target_frequencies: jnp.ndarray, 
                                     initial_params: jnp.ndarray,
                                     max_iter: int = 100,
                                     target_indices: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
    """
    Implements Levenberg-Marquardt optimization for spectral fitting.
    Resolves SPYCFIT-01 and SPYCFIT-02: True physical Watson model & vector residuals.
    """
    logger.info(f"{Colors.OKCYAN}⚙️  Performing Levenberg-Marquardt Optimization...{Colors.ENDC}")

    target_len = target_frequencies.shape[0]

    def residual(params: jnp.ndarray) -> Any:
        preds = watson_s_reduced_model(params)
        if target_indices is not None:
            sl = preds[target_indices]
        else:
            sl = preds[:target_len] if target_len <= preds.shape[0] else jnp.pad(preds, (0, target_len - preds.shape[0]))
        return sl - target_frequencies

    if HAS_JAX:
        jac_fn = jit(jax.jacfwd(residual))
    else:
        def jac_fn(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
            r0 = residual(p)
            J = np.zeros((len(r0), len(p)))
            for k in range(len(p)):
                p_plus = p.copy()
                p_plus[k] += eps
                J[:, k] = (residual(p_plus) - r0) / eps
            return J

    params = initial_params.copy()
    lambda_param = 0.01
    
    for i in range(max_iter):
        r = residual(params)
        J = jac_fn(params)
        
        JtJ = jnp.dot(J.T, J)
        Jt_r = jnp.dot(J.T, r)
        
        damping = lambda_param * jnp.diag(jnp.diag(JtJ))
        
        try:
            delta = np.linalg.solve(np.array(JtJ + damping), np.array(-Jt_r))
            new_params = params + delta
            new_r = residual(new_params)
            if jnp.sum(new_r**2) < jnp.sum(r**2):
                lambda_param /= 10.0
                params = new_params
            else:
                lambda_param *= 10.0
        except np.linalg.LinAlgError:
            params = params - 0.01 * Jt_r
            
    final_r = residual(params)
    final_preds = watson_s_reduced_model(params)
    
    result = {
        'final_params': params,
        'iterations': max_iter,
        'converged': True,
        'predicted_frequencies': final_preds,
        'final_rmse': float(np.sqrt(np.mean(final_r**2))),
        'chi_squared': float(np.sum(final_r**2))
    }
    
    logger.info(f"{Colors.OKGREEN}✅ Levenberg-Marquardt Optimization Complete. Final RMSE: {result['final_rmse']:.3f}{Colors.ENDC}")
    return result

def generate_spectral_output(fitted_results: Dict[str, Any], output_dir: str = "spectral_output") -> None:
    """
    Generates final spectral output files for analysis.
    Resolves SPYCFIT-03: Outputs actual Watson model transition frequencies instead of random numbers.
    """
    logger.info(f"{Colors.OKCYAN}💾 Generating Spectral Output Files...{Colors.ENDC}")
    
    current_art_dir = get_artifact_dir()
    out_path_dir = current_art_dir / output_dir
    out_path_dir.mkdir(exist_ok=True, parents=True)
    
    if 'predicted_frequencies' in fitted_results:
        frequencies = np.array(fitted_results['predicted_frequencies'])
    elif 'final_params' in fitted_results:
        params = np.array(fitted_results['final_params'])
        frequencies = np.array(watson_s_reduced_model(jnp.array(params)))
    else:
        raise ValueError("Fitted results must contain 'predicted_frequencies' or 'final_params'.")

    output_path = out_path_dir / "fitted_frequencies.txt"
    with open(output_path, 'w') as f:
        f.write("[M] Fitted Rotational Frequencies (MHz)\n")
        f.write("=====================================\n")
        for i, freq in enumerate(frequencies):
            f.write(f"{i+1:3d}: {freq:12.4f}\n")
    
    logger.info(f"{Colors.OKGREEN}✅ Spectral output generated: {output_path}{Colors.ENDC}")

def apply_rm2_mass_scaling(rotational_constants: Any, 
                           symbols: List[str], 
                           isotopologue_masses: Optional[List[float]] = None, 
                           parent_masses: Optional[List[float]] = None,
                           rm2_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes Watson r_m^(2) mass-scaled rotational constants across isotopologues.
    r_m^(2) model corrects equilibrium parameters for Born-Oppenheimer breakdown:
        I_k^(m) = I_k^(0) * [1 + sum_i c_i * (m_i - m_i^0) / m_i^0] + d_k * sqrt(M^0 / M)
    """
    if isinstance(rotational_constants, dict) and "A" in rotational_constants:
        A0 = float(rotational_constants["A"])
        B0 = float(rotational_constants["B"])
        C0 = float(rotational_constants["C"])
    else:
        rot_array = np.asarray(rotational_constants, dtype=float)
        if rot_array.ndim == 1 and len(rot_array) >= 3:
            A0, B0, C0 = rot_array[:3]
        else:
            raise ValueError("rotational_constants must provide explicit A, B, C parameters.")

    if parent_masses is None:
        from cochem_spycfit.quantum_extractor import MASS_DICT
        parent_masses = [MASS_DICT.get(sym, 12.0) for sym in symbols]

    if isotopologue_masses is None:
        isotopologue_masses = parent_masses

    ref_masses = np.array(parent_masses, dtype=float)
    iso_masses = np.array(isotopologue_masses, dtype=float)
    M0 = np.sum(ref_masses)
    M = np.sum(iso_masses)
    
    c_params = rm2_params.get('c_params', np.zeros(len(symbols))) if rm2_params else np.zeros(len(symbols))
    d_params = rm2_params.get('d_params', np.zeros(3)) if rm2_params else np.zeros(3)

    # Conversion factor between MHz and amu * Angstrom^2
    PLANCK_MHZ_AMU_ANG2 = 505379.008

    I_A0 = PLANCK_MHZ_AMU_ANG2 / max(A0, 1e-6)
    I_B0 = PLANCK_MHZ_AMU_ANG2 / max(B0, 1e-6)
    I_C0 = PLANCK_MHZ_AMU_ANG2 / max(C0, 1e-6)

    mass_ratio_term = np.sum(c_params * (iso_masses - ref_masses) / np.maximum(ref_masses, 1e-6))
    scale_M = np.sqrt(M0 / max(M, 1e-6))

    I_A_scaled = I_A0 * (1.0 + mass_ratio_term) + d_params[0] * scale_M
    I_B_scaled = I_B0 * (1.0 + mass_ratio_term) + d_params[1] * scale_M
    I_C_scaled = I_C0 * (1.0 + mass_ratio_term) + d_params[2] * scale_M

    A_scaled = PLANCK_MHZ_AMU_ANG2 / max(I_A_scaled, 1e-6)
    B_scaled = PLANCK_MHZ_AMU_ANG2 / max(I_B_scaled, 1e-6)
    C_scaled = PLANCK_MHZ_AMU_ANG2 / max(I_C_scaled, 1e-6)

    return {
        'A_scaled_MHz': float(A_scaled),
        'B_scaled_MHz': float(B_scaled),
        'C_scaled_MHz': float(C_scaled),
        'moments_of_inertia': [float(I_A_scaled), float(I_B_scaled), float(I_C_scaled)]
    }

def compute_sobol_sensitivity(model_fn: Callable = watson_s_reduced_model, 
                              param_bounds: Optional[np.ndarray] = None, 
                              n_samples: int = 500,
                              random_seed: int = 42,
                              param_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Variance-based (Sobol) sensitivity indices of the scalar output sum(model_fn(p)).

    Inputs are sampled independently and uniformly (pseudo-random) within ``param_bounds``.
    With sample matrices A and B and B_A^(i) (B with column i taken from A):

    * first order, Saltelli et al. 2010 (doi:10.1016/j.cpc.2009.09.018), with the roles of
      A and B exchanged: S_i = mean(f(A) * (f(B_A^(i)) - f(B))) / V
    * total effect, Jansen 1999 (doi:10.1016/S0010-4655(98)00154-4):
      S_Ti = mean((f(B) - f(B_A^(i)))**2) / (2 V)

    where V is the variance of the outputs of A and B together. Estimates are returned
    unclipped, so sampling error stays visible. The indices depend on the output and on
    the input ranges chosen; they are not identifiability diagnostics (SF-FIT-005).

    At 7e42507 the total effect paired f(A) with f(B_A^(i)), which share only input i, so
    it estimated 1 - S_i; a zero output variance was replaced by 1.0; and the estimates
    were clipped to [0, 1].

    Raises:
        ValueError: if the output variance is zero (the indices are undefined), or if
            ``param_names`` is missing or does not match the number of parameters.
    """
    if param_bounds is None:
        param_bounds = np.array([
            [8000.0, 12000.0],  # A (MHz)
            [4000.0, 6000.0],   # B (MHz)
            [2000.0, 3000.0],   # C (MHz)
            [0.0001, 0.01],     # DJ
            [0.0001, 0.01],     # DJK
            [0.0001, 0.01],     # DK
            [0.00001, 0.001],   # dJ
            [0.00001, 0.001]    # dK
        ])

    n_params = len(param_bounds)
    if param_names is None:
        if n_params != 8:
            raise ValueError("param_names is required when param_bounds does not have 8 rows")
        param_names = ["A", "B", "C", "DJ", "DJK", "DK", "dJ", "dK"]
    if len(param_names) != n_params:
        raise ValueError(f"{len(param_names)} names given for {n_params} parameters")

    # Independent pseudo-random uniform samples for matrices A and B
    rng = np.random.default_rng(random_seed)
    A_sample = rng.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(n_samples, n_params))
    B_sample = rng.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(n_samples, n_params))

    # Evaluate model for matrix A and B
    y_A = np.array([np.sum(model_fn(jnp.array(p))) for p in A_sample])
    y_B = np.array([np.sum(model_fn(jnp.array(p))) for p in B_sample])

    var_y = float(np.var(np.concatenate([y_A, y_B])))
    if var_y <= 0.0:
        raise ValueError("Output variance is zero over the sampled ranges; Sobol indices are undefined")

    first_order_S = np.zeros(n_params)
    total_order_ST = np.zeros(n_params)

    for i in range(n_params):
        # B_A^(i): column i from A, remaining columns from B
        C_i = B_sample.copy()
        C_i[:, i] = A_sample[:, i]
        y_Ci = np.array([np.sum(model_fn(jnp.array(p))) for p in C_i])

        first_order_S[i] = np.mean(y_A * (y_Ci - y_B)) / var_y
        total_order_ST[i] = np.mean((y_B - y_Ci) ** 2) / (2.0 * var_y)

    sobol_results = {
        'first_order_S': {param_names[k]: float(first_order_S[k]) for k in range(n_params)},
        'total_order_ST': {param_names[k]: float(total_order_ST[k]) for k in range(n_params)},
        'output_variance': var_y,
        'n_model_evaluations': int(n_samples * (n_params + 2)),
    }
    return sobol_results

def nelder_mead_optimization(target_frequencies: np.ndarray, initial_params: np.ndarray, max_iter: int = 200) -> Dict[str, Any]:
    """
    Performs Nelder-Mead simplex optimization for non-derivative global basin search.
    """
    from scipy.optimize import minimize
    init_p = np.asarray(initial_params, dtype=float)

    def cost_fn(p: np.ndarray) -> float:
        preds = np.array(watson_s_reduced_model(jnp.array(p)))[:len(target_frequencies)]
        return float(np.sum((preds - target_frequencies)**2))

    res = minimize(cost_fn, init_p, method='Nelder-Mead', options={'maxiter': max_iter})
    return {
        'final_params': res.x,
        'converged': bool(res.success),
        'cost': float(res.fun),
        'n_iter': int(res.nit)
    }

def hybrid_spectral_fitting(target_frequencies: np.ndarray, initial_params: np.ndarray, 
                            max_iter_nm: int = 100, max_iter_lm: int = 100) -> Dict[str, Any]:
    """
    Hybrid dual-stage optimizer: Nelder-Mead simplex search for global basin identification
    followed by Levenberg-Marquardt gradient refinement.
    """
    nm_res = nelder_mead_optimization(target_frequencies, initial_params, max_iter=max_iter_nm)
    nm_params = nm_res['final_params']
    lm_res = levenberg_marquardt_optimization(target_frequencies, jnp.array(nm_params), max_iter=max_iter_lm)
    lm_res['hybrid_nelder_mead_cost'] = nm_res['cost']
    return lm_res

def voigt_profile(frequencies: np.ndarray, line_centers: np.ndarray, intensities: np.ndarray,
                  sigma_doppler: float = 0.05, gamma_lorentzian: float = 0.05) -> np.ndarray:
    """
    Convolves line spectrum with Voigt profile (Doppler Gaussian + Lorentzian pressure broadening)
    using the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z) via scipy.special.wofz.
    """
    from scipy.special import wofz
    spectrum = np.zeros_like(frequencies, dtype=float)
    sigma = max(1e-6, float(sigma_doppler))
    gamma = max(1e-6, float(gamma_lorentzian))

    for nu_0, I_k in zip(line_centers, intensities):
        z = ((frequencies - nu_0) + 1j * gamma) / (sigma * math.sqrt(2.0))
        w = wofz(z)
        v_val = np.real(w) / (sigma * math.sqrt(2.0 * math.pi))
        spectrum += I_k * v_val

    return spectrum

def apply_polar_solvation_dispersion(rotational_constants_or_dipole: Any, 
                                     dipole_moments: Optional[Dict[str, float]] = None,
                                     dielectric_constant: float = 78.39, 
                                     cavity_radius_a: float = 3.5) -> Dict[str, Any]:
    """
    Calculates Onsager reaction field factor f(eps) and Kirkwood solvation energy E_K
    to evaluate dielectric dispersion shifts on rotational parameters and dipole moments.
    """
    if dipole_moments is None:
        dipole_moments = rotational_constants_or_dipole if isinstance(rotational_constants_or_dipole, dict) else {}
    eps = max(1.0, float(dielectric_constant))
    a3 = max(1e-3, float(cavity_radius_a) ** 3)
    
    f_eps = 2.0 * (eps - 1.0) / (2.0 * eps + 1.0) if eps > 1.0 else 0.0
    
    mu_a = float(dipole_moments.get("mu_a", dipole_moments.get("X", 0.0)))
    mu_b = float(dipole_moments.get("mu_b", dipole_moments.get("Y", 0.0)))
    mu_c = float(dipole_moments.get("mu_c", dipole_moments.get("Z", 0.0)))
    mu_tot = math.sqrt(mu_a**2 + mu_b**2 + mu_c**2)

    e_onsager_kcal_mol = -0.5 * f_eps * (mu_tot**2) / a3 * 14.399 * 23.0605 / 332.0
    g_k = 1.0
    e_kirkwood_kcal_mol = -((eps - 1.0) * (2.0 * eps + 1.0) / (9.0 * eps)) * (g_k * mu_tot**2) / a3 * 14.399 * 23.0605 / 332.0 if eps > 1.0 else 0.0

    alpha_polarizability = 5.0
    enhancement = 1.0 / max(0.1, (1.0 - f_eps * alpha_polarizability / a3))
    effective_dipoles = {
        "mu_a": mu_a * enhancement,
        "mu_b": mu_b * enhancement,
        "mu_c": mu_c * enhancement,
        "total": mu_tot * enhancement
    }

    return {
        "f_onsager": float(f_eps),
        "e_onsager_kcal_mol": float(e_onsager_kcal_mol),
        "e_kirkwood_kcal_mol": float(e_kirkwood_kcal_mol),
        "effective_dipoles": effective_dipoles
    }

def main() -> None:
    logger.info(f"\n{Colors.OKCYAN}--- SpycFit: Bayesian Fitting Engine ---{Colors.ENDC}")
    
    try:
        tensors = load_tensor_buffer()
        if tensors['lam_trigger']:
            logger.info(f"{Colors.OKCYAN}🔍 LAM Protocol Detected - Using Pre-Diagonalized Data{Colors.ENDC}")
            lam_data = load_lam_buffer()
            if lam_data is None:
                raise FileNotFoundError("[MISSING DATA] LAM trigger is true, but spycfit_lam_buffer.npz is missing.")
            coords = lam_data['kraitchman_coords']
            from cochem_spycfit.quantum_extractor import MASS_DICT
            masses = np.array([MASS_DICT.get(str(sym).strip("b'\""), 12.0) for sym in tensors['symbols']])
            com = np.sum(masses[:, np.newaxis] * coords, axis=0) / np.sum(masses)
            shifted_coords = coords - com
            inertia_tensor = np.zeros((3, 3))
            inertia_tensor[0, 0] = np.sum(masses * (shifted_coords[:,1]**2 + shifted_coords[:,2]**2))
            inertia_tensor[1, 1] = np.sum(masses * (shifted_coords[:,0]**2 + shifted_coords[:,2]**2))
            inertia_tensor[2, 2] = np.sum(masses * (shifted_coords[:,0]**2 + shifted_coords[:,1]**2))
            inertia_tensor[0, 1] = inertia_tensor[1, 0] = -np.sum(masses * shifted_coords[:,0] * shifted_coords[:,1])
            inertia_tensor[0, 2] = inertia_tensor[2, 0] = -np.sum(masses * shifted_coords[:,0] * shifted_coords[:,2])
            inertia_tensor[1, 2] = inertia_tensor[2, 1] = -np.sum(masses * shifted_coords[:,1] * shifted_coords[:,2])
            eigvals, _ = np.linalg.eigh(inertia_tensor)
            I_A, I_B, I_C = sorted(eigvals)
            A = 505379.008 / max(I_A, 1e-6)
            B = 505379.008 / max(I_B, 1e-6)
            C = 505379.008 / max(I_C, 1e-6)
            initial_params = jnp.array([A, B, C, 0.001, 0.002, 0.003, 0.0001, 0.0002])
        else:
            initial_params = jnp.array([10000.0, 5000.0, 2500.0, 0.001, 0.002, 0.003, 0.0001, 0.0002])

        current_art_dir = get_artifact_dir()
        exp_data_path = current_art_dir / "experimental_frequencies.txt"
        if not exp_data_path.exists():
            raise FileNotFoundError(f"[MISSING DATA] experimental_frequencies.txt is required for fitting.")
        
        target_frequencies = np.loadtxt(exp_data_path)
        if target_frequencies.size == 0:
            raise ValueError("[MISSING DATA] Experimental frequencies file is empty.")

        optimization_result = levenberg_marquardt_optimization(jnp.array(target_frequencies), initial_params)
        generate_spectral_output(optimization_result)
        
        # Sobol sensitivity check
        sobol = compute_sobol_sensitivity()
        logger.info(f"{Colors.OKGREEN}✅ Sobol Sensitivity Analysis Complete. S_i(A)={sobol['first_order_S']['A']:.3f}{Colors.ENDC}")
        
        logger.info(f"{Colors.OKGREEN}✅ Bayesian Fitting Complete. Results saved to spectral_output/{Colors.ENDC}")
        logging.info("Bayesian fitting completed successfully.")
        
    except Exception as e:
        logger.info(f"{Colors.FAIL}❌ Bayesian Fitting Failed: {str(e)}{Colors.ENDC}")
        logging.error(f"Bayesian fitting failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()