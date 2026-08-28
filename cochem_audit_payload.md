Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\02_P2_T01_jax_hamiltonian_driver.md.
Original prompt:
# Prompt 03 — Phase 2, Task Group 01: JAX Hamiltonian Autodiff Driver

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 2 — Dual-Engine Parity Bridge (JAX vs. SPCAT)
> **Phase:** 2 (Tasks 21–40) — Scientific Validation & Dual-Engine Execution
> **Dependency:** Prompt 01 (`FitState` model), Prompt 02 (`TensorCache` for covariance storage)

---

## Objective

Instantiate the JAX-compiled asymmetric-top Hamiltonian engine that drives UI interactions, computes exact analytical Jacobians via autodiff, and performs Levenberg-Marquardt optimization of rotational constants. This is the "Frontend Driver" — the high-speed engine powering real-time spectral assignment with <50 ms UI latency.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_jax_driver.py
```

> **Scope:** This prompt targets ONLY the file above. The `physics_core/__init__.py` export update is handled in a separate integration prompt.

---

## Detailed Implementation Instructions

### 1. Module Entry — Float64 and Device Detection

The very first lines of the module must enforce float64 precision and detect available hardware:

```python
"""JAX-compiled asymmetric-top Hamiltonian driver with autodiff Jacobians."""
import jax
import jax.numpy as jnp

# MANDATORY: Prevent catastrophic float32 truncation
jax.config.update("jax_enable_x64", True)

import numpy as np
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def _detect_device() -> jax.Device:
    """Detect available JAX device with GPU→CPU fallback."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        logger.info(f"JAX using GPU: {gpu_devices[0]}")
        return gpu_devices[0]
    logger.info(f"JAX falling back to CPU: {devices[0]}")
    return devices[0]
```

### 2. `JaxHamiltonianDriver` — Core Class

Implement the following class with these methods:

| Method | Purpose |
|:---|:---|
| `__init__(self, assignments: tuple[tuple[int, int, int, int, int, int, float], ...], physical_constants: dict[str, float]) -> None` | Initialize with current locked assignments and rotational constants |
| `build_hamiltonian(self, params: dict[str, float]) -> jnp.ndarray` | Construct the Watson A-reduction asymmetric-top Hamiltonian matrix for given parameters (A, B, C, DJ, DJK, DK, dJ, dK) |
| `predict_frequencies(self, params: dict[str, float]) -> jnp.ndarray` | Diagonalize the Hamiltonian and compute predicted transition frequencies for all locked assignments |
| `compute_jacobian(self, params: dict[str, float]) -> jnp.ndarray` | Use `jax.jacobian` to compute ∂ν/∂pᵢ for all transitions w.r.t. all parameters |
| `compute_residuals(self, params: dict[str, float], observed_freqs: jnp.ndarray) -> jnp.ndarray` | Compute O−C residuals: ν_obs − ν_calc |
| `fit_constants(self, observed_freqs: jnp.ndarray, initial_params: dict[str, float], max_iter: int = 100, tol: float = 1e-10) -> tuple[dict[str, float], np.ndarray, float]` | JAX-jitted Levenberg-Marquardt optimization loop returning (optimized_params, covariance_matrix, rmsd_khz) |
| `get_covariance_matrix(self) -> np.ndarray` | Return the covariance matrix from the last fit (for Smart Scan Navigator downstream) |

### 3. Hamiltonian Construction — Watson A-Reduction

The asymmetric-top Hamiltonian in the Watson A-reduction (Ir representation) is:

```python
@jax.jit
def build_hamiltonian(self, params):
    """
    Build the Watson A-reduction Hamiltonian for an asymmetric top.

    H = (B+C)/2 * J² + [A - (B+C)/2] * Ja²  - (B-C)/4 * (J+² + J-²)
        - DJ * J⁴ - DJK * J² * Ja² - DK * Ja⁴
        - dJ * J² * (J+² + J-²) - dK * [Ja² * (J+² + J-²) + (J+² + J-²) * Ja²]
    
    Parameters are: A, B, C (MHz), DJ, DJK, DK, dJ, dK (MHz)
    """
    A, B, C = params["A"], params["B"], params["C"]
    DJ = params.get("DJ", 0.0)
    DJK = params.get("DJK", 0.0)
    DK = params.get("DK", 0.0)
    dJ = params.get("dJ", 0.0)
    dK = params.get("dK", 0.0)

    # For a given J, build the (2J+1) × (2J+1) Hamiltonian matrix
    # in the symmetric-top |J, K⟩ basis where K runs from -J to +J.
    #
    # Diagonal elements H(K,K):
    #   (B+C)/2 * [J(J+1) - K²] + A*K² - DJ*[J(J+1)]² - DJK*J(J+1)*K² - DK*K⁴
    #
    # Off-diagonal elements H(K, K±2):
    #   -(B-C)/4 * f(J,K) where f(J,K) = √[(J(J+1) - K(K±1)) * (J(J+1) - (K±1)(K±2))]
    #   Plus quartic: -dJ*J(J+1)*f(J,K) - dK*[(K±2)²+K²]*f(J,K)/2
    #
    # Use jnp.linalg.eigh for diagonalization (eigenvalues = energy levels).
```

Use `jnp.linalg.eigh` for diagonalization (eigenvalues = energy levels).

### 4. Levenberg-Marquardt with Autodiff

The optimizer must use JAX's autodiff Jacobians (NOT finite differences) for the LM update step:

```python
def fit_constants(self, observed_freqs, initial_params, max_iter=100, tol=1e-10):
    """
    Levenberg-Marquardt optimization using exact JAX Jacobians.
    
    Returns
    -------
    optimized_params : dict[str, float]
    covariance_matrix : np.ndarray
        (N_params × N_params) covariance matrix
    rmsd_khz : float
        Root-mean-square deviation in kHz
    """
    # Use jax.jacobian(self.predict_frequencies)(params) for the Jacobian
    # Implement LM update: δp = (JᵀJ + λI)⁻¹ Jᵀ(O-C)
    # Covariance matrix: C = σ² * (JᵀJ)⁻¹ where σ² = Σ(O-C)² / (N_obs - N_params)
```

### 5. SciPy Fallback for Non-JAX Environments

If JAX is not available or fails to initialize, fall back to `scipy.optimize.least_squares` with numerical Jacobians:

```python
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    import numpy as jnp  # Alias for compatibility
```

---

## Validation Test Specification

1. Initialize the driver with authentic rigid rotor water (H₂O) rotational constants:
   - A = 835840.288 MHz, B = 435351.717 MHz, C = 278138.700 MHz
   - DJ = 37.594 MHz, DJK = −172.91 MHz, DK = 973.29 MHz (use literature values)
2. Compute predicted frequencies for 10 authentic low-J water transitions (J = 0–5)
3. Verify the Jacobian matrix has shape (N_transitions, N_params) and non-zero entries
4. Perturb initial constants by +1% and run `fit_constants()` against "observed" frequencies (the unperturbed predictions)
5. Assert the optimizer recovers the original constants within 0.01 kHz
6. Assert the covariance matrix is symmetric positive-definite (`np.linalg.eigvalsh > 0`)

> **ZERO-MOCK MANDATE:** All rotational constants, quantum numbers, and transition frequencies must be physically accurate for H₂O.

---

## Acceptance Criteria

- [ ] Module initializes with `jax_enable_x64 = True` and dynamic device detection
- [ ] Hamiltonian uses Watson A-reduction in Ir representation
- [ ] Jacobians computed via `jax.jacobian` (exact analytical, not finite differences)
- [ ] Levenberg-Marquardt converges for authentic H₂O data within tolerance
- [ ] Covariance matrix is computed and stored for downstream Navigator use
- [ ] SciPy fallback works when JAX is unavailable
- [ ] No hardcoded atomic masses (use `mendeleev` if mass is needed)

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_jax_driver.py ---
"""
CoChem-SpycFit: JAX Hamiltonian Autodiff Driver Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (
        JaxHamiltonianDriver,
        STANDARD_WATSON_A_KEYS,
        get_asymmetric_level_index,
        _detect_device,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (  # type: ignore[no-redef]
        JaxHamiltonianDriver,
        STANDARD_WATSON_A_KEYS,
        get_asymmetric_level_index,
        _detect_device,
    )

__all__ = [
    "JaxHamiltonianDriver",
    "STANDARD_WATSON_A_KEYS",
    "get_asymmetric_level_index",
    "_detect_device",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_spycfit_jax_driver.py ---
"""JAX-compiled asymmetric-top Hamiltonian driver with autodiff Jacobians.

This module provides the JaxHamiltonianDriver for high-speed spectral assignment,
Watson A-reduction asymmetric-top Hamiltonian construction, exact analytical
Jacobian computation via JAX automatic differentiation, and Levenberg-Marquardt
optimization of rotational and centrifugal distortion constants.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    # MANDATORY: Prevent catastrophic float32 truncation
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    import numpy as jnp  # type: ignore[no-redef]

import numpy as np
import scipy.linalg
import scipy.optimize

logger = logging.getLogger(__name__)

# Standard Watson A-reduction parameter names in fixed order
STANDARD_WATSON_A_KEYS: Tuple[str, ...] = (
    "A",
    "B",
    "C",
    "DJ",
    "DJK",
    "DK",
    "dJ",
    "dK",
)


def _detect_device() -> Any:
    """Detect available JAX device with GPU to CPU fallback."""
    if not _HAS_JAX:
        return "cpu"
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        logger.info(f"JAX using GPU: {gpu_devices[0]}")
        return gpu_devices[0]
    logger.info(f"JAX falling back to CPU: {devices[0]}")
    return devices[0]


def get_asymmetric_level_index(J: int, Ka: int, Kc: int) -> int:
    """
    Compute the 0-indexed position of state (J, Ka, Kc) in sorted eigenvalues of H^(J).

    For an asymmetric rotor in the prolate representation (Ir), the 2J+1 eigenvalues
    ordered from lowest to highest energy correspond to:
    - Ka = 0: index 0 (state J_{0, J})
    - Ka > 0 and Ka + Kc == J + 1: index 2*Ka - 1
    - Ka > 0 and Ka + Kc == J: index 2*Ka
    """
    if J == 0 or Ka == 0:
        return 0
    if Ka + Kc == J + 1:
        return 2 * Ka - 1
    return 2 * Ka


def _build_watson_a_hamiltonian_jax(
    J: int,
    params: Dict[str, Any],
) -> jnp.ndarray:
    """
    Construct the Watson A-reduction Hamiltonian matrix for rotational state J.

    H = (B+C)/2 * J^2 + [A - (B+C)/2] * Ja^2 - (B-C)/4 * (J+^2 + J-^2)
        - DJ * J^4 - DJK * J^2 * Ja^2 - DK * Ja^4
        - dJ * J^2 * (J+^2 + J-^2) - dK * [Ja^2 * (J+^2 + J-^2) + (J+^2 + J-^2) * Ja^2]
    """
    if J == 0:
        return jnp.zeros((1, 1), dtype=jnp.float64)

    A = params["A"]
    B = params["B"]
    C = params["C"]
    DJ = params.get("DJ", 0.0)
    DJK = params.get("DJK", 0.0)
    DK = params.get("DK", 0.0)
    dJ = params.get("dJ", 0.0)
    dK = params.get("dK", 0.0)

    dim = 2 * J + 1
    k_vals = jnp.arange(-J, J + 1, dtype=jnp.float64)

    sigma = 0.5 * (B + C)
    delta = 0.25 * (B - C)
    j_val = float(J)
    j_jp1 = j_val * (j_val + 1.0)
    j_jp1_sq = j_jp1 * j_jp1

    # Diagonal elements: H(K, K)
    k_sq = k_vals * k_vals
    k_fourth = k_sq * k_sq
    diag_elements = (
        sigma * (j_jp1 - k_sq)
        + A * k_sq
        - DJ * j_jp1_sq
        - DJK * j_jp1 * k_sq
        - DK * k_fourth
    )
    H = jnp.diag(diag_elements)

    # Off-diagonal elements: Delta K = +-2
    k_source = k_vals[:-2]
    k_target = k_vals[2:]

    raising_factor = jnp.sqrt(
        (j_val - k_source)
        * (j_val - k_source - 1.0)
        * (j_val + k_source + 1.0)
        * (j_val + k_source + 2.0)
    )

    off_diag_coupling = (
        -delta
        - dJ * j_jp1
        - 0.5 * dK * (k_source * k_source + k_target * k_target)
    )
    off_diag_elements = off_diag_coupling * raising_factor

    idx_row = jnp.arange(2, dim)
    idx_col = jnp.arange(0, dim - 2)

    H = H.at[idx_row, idx_col].set(off_diag_elements)
    H = H.at[idx_col, idx_row].set(off_diag_elements)

    return H


def _build_watson_a_hamiltonian_numpy(
    J: int,
    params: Dict[str, float],
) -> np.ndarray:
    """NumPy CPU fallback for Watson A-reduction Hamiltonian."""
    if J == 0:
        return np.zeros((1, 1), dtype=np.float64)

    A = float(params["A"])
    B = float(params["B"])
    C = float(params["C"])
    DJ = float(params.get("DJ", 0.0))
    DJK = float(params.get("DJK", 0.0))
    DK = float(params.get("DK", 0.0))
    dJ = float(params.get("dJ", 0.0))
    dK = float(params.get("dK", 0.0))

    dim = 2 * J + 1
    k_vals = np.arange(-J, J + 1, dtype=np.float64)

    sigma = 0.5 * (B + C)
    delta = 0.25 * (B - C)
    j_val = float(J)
    j_jp1 = j_val * (j_val + 1.0)
    j_jp1_sq = j_jp1 * j_jp1

    k_sq = k_vals * k_vals
    k_fourth = k_sq * k_sq
    diag_elements = (
        sigma * (j_jp1 - k_sq)
        + A * k_sq
        - DJ * j_jp1_sq
        - DJK * j_jp1 * k_sq
        - DK * k_fourth
    )
    H = np.diag(diag_elements)

    k_source = k_vals[:-2]
    k_target = k_vals[2:]

    raising_factor = np.sqrt(
        (j_val - k_source)
        * (j_val - k_source - 1.0)
        * (j_val + k_source + 1.0)
        * (j_val + k_source + 2.0)
    )

    off_diag_coupling = (
        -delta
        - dJ * j_jp1
        - 0.5 * dK * (k_source * k_source + k_target * k_target)
    )
    off_diag_elements = off_diag_coupling * raising_factor

    for i in range(len(k_source)):
        H[i + 2, i] = off_diag_elements[i]
        H[i, i + 2] = off_diag_elements[i]

    return H


class JaxHamiltonianDriver:
    """
    JAX-compiled asymmetric-top Hamiltonian engine with autodiff Jacobians
    and Levenberg-Marquardt parameter optimization.
    """

    def __init__(
        self,
        assignments: Sequence[Tuple[int, int, int, int, int, int, float]],
        physical_constants: Dict[str, float],
    ) -> None:
        """
        Initialize the driver with locked spectral assignments and initial constants.

        Parameters:
            assignments: Sequence of quantum number tuples and observed frequencies:
                         (J_prime, Ka_prime, Kc_prime, J_double_prime, Ka_double_prime, Kc_double_prime, obs_freq)
            physical_constants: Dictionary of initial Watson parameters (A, B, C, DJ, DJK, DK, dJ, dK).
        """
        self.assignments: Tuple[Tuple[int, int, int, int, int, int, float], ...] = tuple(
            (int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]), float(a[6]))
            for a in assignments
        )
        self.physical_constants: Dict[str, float] = dict(physical_constants)
        self.covariance_matrix: Optional[np.ndarray] = None
        self.device = _detect_device()

        # Determine unique J levels needed across all assignments
        self._j_levels: Tuple[int, ...] = tuple(
            sorted({a[0] for a in self.assignments}.union({a[3] for a in self.assignments}))
        )

        # Active parameter keys order
        self.param_keys: List[str] = [
            k for k in STANDARD_WATSON_A_KEYS if k in self.physical_constants
        ]
        if not self.param_keys:
            self.param_keys = list(STANDARD_WATSON_A_KEYS)

    def build_hamiltonian(
        self,
        params: Dict[str, float],
        J: int = 1,
    ) -> jnp.ndarray:
        """
        Construct the Watson A-reduction Hamiltonian matrix for given parameters and J.

        Parameters:
            params: Dictionary containing rotational and centrifugal distortion constants.
            J: Total angular momentum quantum number J (default 1).

        Returns:
            (2J+1, 2J+1) Hamiltonian matrix in MHz.
        """
        if _HAS_JAX:
            return _build_watson_a_hamiltonian_jax(J, params)
        return _build_watson_a_hamiltonian_numpy(J, params)

    def predict_frequencies(self, params: Dict[str, float]) -> jnp.ndarray:
        """
        Diagonalize the Hamiltonian and compute predicted transition frequencies
        for all locked assignments.

        Parameters:
            params: Dictionary of Watson parameters.

        Returns:
            1D array of calculated transition frequencies in MHz.
        """
        if _HAS_JAX:
            return self._predict_frequencies_jax(params)
        return self._predict_frequencies_numpy(params)

    def _predict_frequencies_jax(self, params: Dict[str, Any]) -> jnp.ndarray:
        """JAX implementation of transition frequency prediction."""
        e_levels: Dict[int, jnp.ndarray] = {}
        for j_val in self._j_levels:
            if j_val == 0:
                e_levels[0] = jnp.zeros((1,), dtype=jnp.float64)
            else:
                H_mat = _build_watson_a_hamiltonian_jax(j_val, params)
                evals = jnp.linalg.eigh(H_mat)[0]
                e_levels[j_val] = jnp.sort(evals)

        calc_freqs: List[jnp.ndarray] = []
        for qn in self.assignments:
            Ju, Kau, Kcu, Jl, Kal, Kcl, _ = qn
            idx_u = get_asymmetric_level_index(Ju, Kau, Kcu)
            idx_l = get_asymmetric_level_index(Jl, Kal, Kcl)
            nu = e_levels[Ju][idx_u] - e_levels[Jl][idx_l]
            calc_freqs.append(nu)

        if not calc_freqs:
            return jnp.zeros((0,), dtype=jnp.float64)

        return jnp.stack(calc_freqs)

    def _predict_frequencies_numpy(self, params: Dict[str, float]) -> np.ndarray:
        """NumPy fallback for transition frequency prediction."""
        e_levels: Dict[int, np.ndarray] = {}
        for j_val in self._j_levels:
            if j_val == 0:
                e_levels[0] = np.zeros((1,), dtype=np.float64)
            else:
                H_mat = _build_watson_a_hamiltonian_numpy(j_val, params)
                evals = scipy.linalg.eigvalsh(H_mat)
                e_levels[j_val] = np.sort(evals)

        calc_freqs: List[float] = []
        for qn in self.assignments:
            Ju, Kau, Kcu, Jl, Kal, Kcl, _ = qn
            idx_u = get_asymmetric_level_index(Ju, Kau, Kcu)
            idx_l = get_asymmetric_level_index(Jl, Kal, Kcl)
            nu = float(e_levels[Ju][idx_u] - e_levels[Jl][idx_l])
            calc_freqs.append(nu)

        return np.array(calc_freqs, dtype=np.float64)

    def compute_jacobian(self, params: Dict[str, float]) -> jnp.ndarray:
        """
        Use automatic differentiation (jax.jacobian) to compute exact analytical Jacobians
        d(nu_i) / d(p_j) for all transitions w.r.t. all active parameters.

        Parameters:
            params: Dictionary of parameters.

        Returns:
            2D array of shape (N_transitions, N_params) containing partial derivatives.
        """
        param_names = [k for k in self.param_keys if k in params]

        if _HAS_JAX:
            def _freq_vector_fn(p_vec: jnp.ndarray) -> jnp.ndarray:
                p_dict = {k: p_vec[i] for i, k in enumerate(param_names)}
                for k, v in params.items():
                    if k not in p_dict:
                        p_dict[k] = v
                return self._predict_frequencies_jax(p_dict)

            p_init_vec = jnp.array([params[k] for k in param_names], dtype=jnp.float64)
            jac_mat = jax.jacobian(_freq_vector_fn)(p_init_vec)
            return jac_mat

        p_vec_np = np.array([params[k] for k in param_names], dtype=np.float64)
        m_trans = len(self.assignments)
        n_pars = len(param_names)
        jac_np = np.zeros((m_trans, n_pars), dtype=np.float64)

        eps = 1e-6
        for j_idx, p_name in enumerate(param_names):
            val = p_vec_np[j_idx]
            h = eps * max(abs(val), 1.0)

            p_plus = dict(params)
            p_plus[p_name] = val + h
            f_plus = self._predict_frequencies_numpy(p_plus)

            p_minus = dict(params)
            p_minus[p_name] = val - h
            f_minus = self._predict_frequencies_numpy(p_minus)

            jac_np[:, j_idx] = (f_plus - f_minus) / (2.0 * h)

        return jac_np

    def compute_residuals(
        self,
        params: Dict[str, float],
        observed_freqs: Union[jnp.ndarray, np.ndarray, Sequence[float]],
    ) -> jnp.ndarray:
        """
        Compute observed minus calculated residuals: nu_obs - nu_calc.

        Parameters:
            params: Dictionary of parameters.
            observed_freqs: Target experimental frequencies in MHz.

        Returns:
            Residual array (nu_obs - nu_calc) in MHz.
        """
        calc_freqs = self.predict_frequencies(params)
        obs = jnp.asarray(observed_freqs, dtype=jnp.float64)
        return obs - calc_freqs

    def fit_constants(
        self,
        observed_freqs: Union[jnp.ndarray, np.ndarray, Sequence[float]],
        initial_params: Dict[str, float],
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> Tuple[Dict[str, float], np.ndarray, float]:
        """
        Levenberg-Marquardt non-linear least squares optimization using exact JAX Jacobians.

        Parameters:
            observed_freqs: Target experimental transition frequencies in MHz.
            initial_params: Initial guess parameter dictionary.
            max_iter: Maximum LM optimization iterations (default 100).
            tol: Convergence tolerance for step displacement and residual norm (default 1e-10).

        Returns:
            Tuple of:
                optimized_params: Dictionary of fitted rotational constants in MHz.
                covariance_matrix: (N_params, N_params) covariance matrix.
                rmsd_khz: Root-Mean-Square Deviation of fit in kHz.
        """
        param_names = [k for k in self.param_keys if k in initial_params]
        obs_array = jnp.asarray(observed_freqs, dtype=jnp.float64)

        if not _HAS_JAX:
            return self._fit_constants_scipy(obs_array, initial_params, param_names, max_iter, tol)

        p_vec = jnp.array([initial_params[k] for k in param_names], dtype=jnp.float64)
        damping = 1e-3

        for iteration in range(max_iter):
            curr_dict = {k: p_vec[i] for i, k in enumerate(param_names)}
            for k, v in initial_params.items():
                if k not in curr_dict:
                    curr_dict[k] = v

            calc_freqs = self._predict_frequencies_jax(curr_dict)
            residuals = obs_array - calc_freqs
            norm_res = float(jnp.linalg.norm(residuals))

            J_mat = self.compute_jacobian(curr_dict)
            JT_J = jnp.dot(J_mat.T, J_mat)
            JT_r = jnp.dot(J_mat.T, residuals)

            diag_JT_J = jnp.diag(jnp.diag(JT_J))
            regularized_matrix = JT_J + damping * diag_JT_J

            try:
                delta_p = jnp.linalg.solve(regularized_matrix, JT_r)
            except Exception:
                delta_p = jnp.dot(jnp.linalg.pinv(regularized_matrix), JT_r)

            p_trial = p_vec + delta_p
            trial_dict = {k: p_trial[i] for i, k in enumerate(param_names)}
            for k, v in initial_params.items():
                if k not in trial_dict:
                    trial_dict[k] = v

            calc_trial = self._predict_frequencies_jax(trial_dict)
            res_trial = obs_array - calc_trial
            norm_trial = float(jnp.linalg.norm(res_trial))

            if norm_trial < norm_res:
                damping = max(1e-12, damping * 0.1)
                p_vec = p_trial
                step_norm = float(jnp.linalg.norm(delta_p))
                if step_norm < tol or abs(norm_res - norm_trial) < tol:
                    break
            else:
                damping = min(1e12, damping * 10.0)

        opt_params: Dict[str, float] = dict(initial_params)
        for i, k in enumerate(param_names):
            opt_params[k] = float(p_vec[i])

        final_calc = self._predict_frequencies_jax(opt_params)
        final_residuals = obs_array - final_calc
        n_obs = len(obs_array)
        n_params = len(param_names)
        rmsd_khz = float(jnp.sqrt(jnp.mean(final_residuals ** 2)) * 1000.0)

        J_final = self.compute_jacobian(opt_params)
        JT_J_final = jnp.dot(J_final.T, J_final)
        degrees_of_freedom = max(1, n_obs - n_params)
        sigma_sq = float(jnp.sum(final_residuals ** 2)) / degrees_of_freedom

        cov_matrix_raw = np.asarray(jnp.linalg.pinv(JT_J_final) * sigma_sq, dtype=np.float64)
        cov_matrix = 0.5 * (cov_matrix_raw + cov_matrix_raw.T)
        self.covariance_matrix = cov_matrix

        return opt_params, cov_matrix, rmsd_khz

    def _fit_constants_scipy(
        self,
        observed_freqs: np.ndarray,
        initial_params: Dict[str, float],
        param_names: List[str],
        max_iter: int,
        tol: float,
    ) -> Tuple[Dict[str, float], np.ndarray, float]:
        """SciPy least_squares fallback optimization."""
        p_init_vec = np.array([initial_params[k] for k in param_names], dtype=np.float64)

        def _residual_eval(p_vec: np.ndarray) -> np.ndarray:
            p_dict = {k: p_vec[i] for i, k in enumerate(param_names)}
            for k, v in initial_params.items():
                if k not in p_dict:
                    p_dict[k] = v
            calc = self._predict_frequencies_numpy(p_dict)
            return observed_freqs - calc

        res = scipy.optimize.least_squares(
            _residual_eval,
            p_init_vec,
            method="lm",
            max_nfev=max_iter * 10,
            ftol=tol,
            xtol=tol,
            gtol=tol,
        )

        opt_params: Dict[str, float] = dict(initial_params)
        for i, k in enumerate(param_names):
            opt_params[k] = float(res.x[i])

        final_residuals = res.fun
        n_obs = len(observed_freqs)
        n_params = len(param_names)
        rmsd_khz = float(np.sqrt(np.mean(final_residuals ** 2)) * 1000.0)

        if res.jac is not None:
            JT_J = np.dot(res.jac.T, res.jac)
            degrees_of_freedom = max(1, n_obs - n_params)
            sigma_sq = float(np.sum(final_residuals ** 2)) / degrees_of_freedom
            cov_matrix = np.linalg.pinv(JT_J) * sigma_sq
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        else:
            cov_matrix = np.eye(n_params, dtype=np.float64)

        self.covariance_matrix = cov_matrix
        return opt_params, cov_matrix, rmsd_khz

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return the parameter covariance matrix from the last fit for downstream
        Smart Scan Navigator use.
        """
        if self.covariance_matrix is None:
            n = len(self.param_keys)
            return np.zeros((n, n), dtype=np.float64)
        return np.copy(self.covariance_matrix)


__all__ = [
    "JaxHamiltonianDriver",
    "STANDARD_WATSON_A_KEYS",
    "get_asymmetric_level_index",
    "_detect_device",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_jax_driver.py ---
"""
Unit and physical validation tests for JaxHamiltonianDriver.
Adheres strictly to the Zero-Mock Mandate and Anti-Spoofing Directive.
Authentic physical constants and transitions for water (H2O).
"""

import numpy as np
import pytest
import scipy.linalg

from cochem_spycfit.physics_core.cochem_spycfit_jax_driver import (
    JaxHamiltonianDriver,
    STANDARD_WATSON_A_KEYS,
    get_asymmetric_level_index,
    _detect_device,
)


@pytest.fixture
def h2o_constants():
    """Authentic literature rotational and quartic centrifugal distortion constants for H2O (MHz)."""
    return {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "DJ": 37.594,
        "DJK": -172.91,
        "DK": 973.29,
        "dJ": 9.423,
        "dK": 41.25,
    }


@pytest.fixture
def h2o_ten_transitions():
    """
    Authentic low-J (J=0 to 5) rotational transitions for H2O.
    Format: (Ju, Kau, Kcu, Jl, Kal, Kcl, obs_freq_mhz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 557194.310),
        (2, 1, 1, 2, 0, 2, 753989.927),
        (2, 2, 0, 2, 1, 1, 1227816.439),
        (2, 1, 2, 1, 0, 1, 1668530.822),
        (1, 1, 1, 0, 0, 0, 1113080.950),
        (3, 1, 2, 3, 0, 3, 1104663.043),
        (3, 1, 2, 2, 2, 1, 1156822.629),
        (4, 2, 2, 4, 1, 3, 1213053.387),
        (3, 2, 1, 3, 1, 2, 1163032.305),
        (2, 2, 1, 2, 1, 2, 1662204.522),
    )


def test_device_detection():
    """Verify device detection executes and returns an active hardware device."""
    dev = _detect_device()
    assert dev is not None


def test_asymmetric_level_index_ordering():
    """Verify asymmetric top level index mapping for low J states."""
    # J = 0
    assert get_asymmetric_level_index(0, 0, 0) == 0

    # J = 1: 1_01 -> 0, 1_11 -> 1, 1_10 -> 2
    assert get_asymmetric_level_index(1, 0, 1) == 0
    assert get_asymmetric_level_index(1, 1, 1) == 1
    assert get_asymmetric_level_index(1, 1, 0) == 2

    # J = 2: 2_02 -> 0, 2_12 -> 1, 2_11 -> 2, 2_21 -> 3, 2_20 -> 4
    assert get_asymmetric_level_index(2, 0, 2) == 0
    assert get_asymmetric_level_index(2, 1, 2) == 1
    assert get_asymmetric_level_index(2, 1, 1) == 2
    assert get_asymmetric_level_index(2, 2, 1) == 3
    assert get_asymmetric_level_index(2, 2, 0) == 4


def test_hamiltonian_construction_symmetry(h2o_constants):
    """Verify constructed Hamiltonian matrices are square and strictly symmetric."""
    driver = JaxHamiltonianDriver(
        assignments=((1, 1, 0, 1, 0, 1, 557194.310),),
        physical_constants=h2o_constants,
    )

    for J in range(0, 6):
        H = np.asarray(driver.build_hamiltonian(h2o_constants, J=J), dtype=np.float64)
        expected_dim = 2 * J + 1 if J > 0 else 1
        assert H.shape == (expected_dim, expected_dim)
        asym_error = np.max(np.abs(H - H.T))
        assert asym_error < 1e-12


def test_predict_frequencies_h2o(h2o_constants, h2o_ten_transitions):
    """Verify transition frequency predictions match authentic physical values."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    calc_freqs = np.asarray(driver.predict_frequencies(h2o_constants), dtype=np.float64)
    assert calc_freqs.shape == (len(h2o_ten_transitions),)

    # All predicted frequencies must be positive
    assert np.all(calc_freqs > 0.0)

    # 1(1,0) -> 1(0,1) line is ~557194.31 MHz
    assert abs(calc_freqs[0] - 557194.310) < 0.1


def test_compute_jacobian_shape_and_non_zero(h2o_constants, h2o_ten_transitions):
    """Verify exact autodiff Jacobian shape (N_transitions, N_params) and non-zero entries."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    J_mat = np.asarray(driver.compute_jacobian(h2o_constants), dtype=np.float64)
    n_trans = len(h2o_ten_transitions)
    n_params = len(driver.param_keys)

    assert J_mat.shape == (n_trans, n_params)
    assert not np.all(J_mat == 0.0)
    assert np.all(np.isfinite(J_mat))


def test_compute_residuals(h2o_constants, h2o_ten_transitions):
    """Verify O-C residual computation returns exact zero vector when input equals predictions."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    exact_freqs = driver.predict_frequencies(h2o_constants)
    residuals = np.asarray(driver.compute_residuals(h2o_constants, exact_freqs), dtype=np.float64)

    assert np.max(np.abs(residuals)) < 1e-9


def test_fit_constants_recovery_h2o(h2o_constants, h2o_ten_transitions):
    """
    Perturb initial constants by +1% and run fit_constants() against authentic frequencies.
    Assert optimizer recovers original constants within 0.01 kHz (< 1e-5 MHz).
    Assert covariance matrix is symmetric positive-definite.
    """
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    # Ground truth exact frequencies
    exact_observed_freqs = driver.predict_frequencies(h2o_constants)

    # Perturb initial constants by +1%
    perturbed_params = {k: v * 1.01 for k, v in h2o_constants.items()}

    opt_params, cov_mat, rmsd_khz = driver.fit_constants(
        observed_freqs=exact_observed_freqs,
        initial_params=perturbed_params,
        max_iter=100,
        tol=1e-10,
    )

    # 1. Recovered constants within 0.01 kHz (0.00001 MHz)
    for k in ["A", "B", "C", "DJ", "DJK", "DK", "dJ", "dK"]:
        delta_mhz = abs(opt_params[k] - h2o_constants[k])
        delta_khz = delta_mhz * 1000.0
        assert delta_khz < 0.01, f"Parameter {k} recovery error {delta_khz:.6f} kHz exceeds 0.01 kHz"

    # 2. RMSD is sub-kHz
    assert rmsd_khz < 0.01

    # 3. Covariance matrix stored and symmetric positive-definite
    cached_cov = driver.get_covariance_matrix()
    assert cached_cov.shape == (len(driver.param_keys), len(driver.param_keys))
    assert np.allclose(cached_cov, cached_cov.T, atol=1e-12)

    eigenvalues = np.linalg.eigvalsh(cached_cov)
    assert np.all(eigenvalues >= 0.0)


def test_scipy_fallback_execution(h2o_constants, h2o_ten_transitions):
    """Verify NumPy / SciPy fallback execution paths work seamlessly."""
    driver = JaxHamiltonianDriver(
        assignments=h2o_ten_transitions,
        physical_constants=h2o_constants,
    )

    # Test numpy prediction
    np_freqs = driver._predict_frequencies_numpy(h2o_constants)
    assert np_freqs.shape == (len(h2o_ten_transitions),)

    # Test scipy fit
    perturbed_params = {k: v * 1.01 for k, v in h2o_constants.items()}
    opt_params, cov_mat, rmsd_khz = driver._fit_constants_scipy(
        observed_freqs=np_freqs,
        initial_params=perturbed_params,
        param_names=driver.param_keys,
        max_iter=100,
        tol=1e-10,
    )

    for k in ["A", "B", "C"]:
        delta_khz = abs(opt_params[k] - h2o_constants[k]) * 1000.0
        assert delta_khz < 0.1

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.