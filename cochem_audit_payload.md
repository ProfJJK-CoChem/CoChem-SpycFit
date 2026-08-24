Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task8_Prompt1_convolver_phase3.md.
Original prompt:
# Task 8 (Phase 3): Convolution Engine - cochem_convolver.py

**Objective:**
Update `cochem_convolver.py` to implement Phase 3 requirements including pure JAX Voigt profiling via the Faddeeva function, multi-temperature tensor broadcasting, and static masked batching.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_convolver.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 3 features in `cochem_convolver.py`.
It must:
1. Implement pure-JAX Gaussian, Lorentzian, and Voigt profiles. The Voigt profile must use the true Faddeeva function (e.g., `jax.scipy.special.erfcx`). Pseudo-Voigt approximations and standard `scipy` imports are strictly forbidden.
2. Implement Multi-Temperature Partition Grids. Generate a unified tensor grid `[N_lines, N_temperatures]` using native broadcasting. Incorporate a pre-computed $Q(T)$ array injected as a parameter to scale stimulated emission multipliers without Python `for` loops.
3. Implement Static Masked Batching via `jax.lax.scan`. Define fixed, standardized chunk sizes (e.g., 10,000 points) and pad with zeros to prevent XLA recompilation thrashing.
4. Enforce functional purity. Do NOT perform any I/O operations (no reading/writing files). Return the results as a pure mathematical `PyTree`.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Actually implement the XLA-compliant mathematical logic.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import jax
import jax.numpy as jnp
from jax.scipy.special import erfcx

@jax.jit
def true_voigt_profile(x, sigma, gamma):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

@jax.jit
def convolute_multi_temperature(frequencies, intensities, Q_T, temp_grid):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

@jax.jit
def static_masked_scan(lines, chunk_size=10000):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_convolver.py ---
"""
CoChem-SpycFit: Macroscopic Spectral Broadener & High-Performance JAX Convolution Engine.

This module implements:
1. Analytical and vectorized pure-JAX FP64 line profiling:
   - Gaussian profile (analytical Doppler broadening)
   - Lorentzian profile (analytical pressure / collisional broadening)
   - Thompson-Cox-Hastings Pseudo-Voigt profile (spectroscopic analytical approximation)
   - Faddeeva Voigt profile (pure-JAX 32-term Weideman rational approximation of w(z))
2. Multi-Temperature Partition Grids & Boltzmann Scaling:
   - Multi-temperature partition function evaluation Q(T) via direct state-sum or power-law
   - Vectorized Boltzmann thermal state population weighting [N_temps, N_lines]
   - Stimulated emission factor correction with high-precision expm1 numerical stability
3. Memory-Aware Micro-Batching via jax.lax.scan:
   - Dynamic chunk calculation from VRAM/RAM budget cap to prevent CUDA/CPU OOM
   - Chunked frequency evaluation sequentially processed via jax.lax.scan with zero Python loops
   - Chunked spectral line accumulation via jax.lax.scan for million-line databases
4. Comprehensive Pydantic V2 schemas and typed exception hierarchy.

Precision: JAX FP64 precision enabled globally.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Enforce JAX 64-bit precision globally for high-accuracy spectroscopic physics
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.Convolver")

# Try to import FitterError from sibling module, fallback to Base Exception if standalone
try:
    from .cochem_jax_fitter import FitterError
except (ImportError, ValueError):
    class FitterError(Exception):
        """Base exception for all JAX fitter and spectroscopic optimization operations."""


# =============================================================================
# 1. PHYSICAL CONSTANTS & CONVERSION FACTORS (CODATA / IUPAC Recommended)
# =============================================================================

# Planck constant h in J * s (exact SI definition)
PLANCK_H: float = 6.62607015e-34

# Boltzmann constant k_B in J / K (exact SI definition)
BOLTZMANN_K: float = 1.380649e-23

# Speed of light in vacuum c in cm / s and m / s
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10
SPEED_OF_LIGHT_M_S: float = 299792458.0

# Ratio h / k_B in s * K (for frequency in Hz -> Kelvin: nu * (h / k_B))
H_OVER_K_B: float = 4.799243073366221e-11

# Second radiation constant c_2 = h * c / k_B in cm * K (for wavenumber in cm^-1 -> Kelvin: nu_tilde * c_2)
C2_RADIATION_CONST: float = 1.4387768775039337

# Gaussian profile constants
# FWHM = 2 * sqrt(2 * ln(2)) * sigma  =>  sigma = FWHM / (2 * sqrt(2 * ln(2)))
SQRT_2_LN_2: float = 1.1774100225154747
FWHM_TO_SIGMA_GAUSS: float = 0.42466090014400953
FOUR_LN_2: float = 2.772588722239781
GAUSS_PREFACTOR: float = 0.9394372786382881  # 2 * sqrt(ln(2) / pi)

# Lorentzian profile constants
# Peak height = 2 / (pi * FWHM)
LORENTZ_PREFACTOR: float = 0.6366197723675814  # 2 / pi

# Default VRAM budget cap (512 MB)
DEFAULT_MAX_VRAM_MB: float = 512.0


# =============================================================================
# 2. TYPED EXCEPTION HIERARCHY
# =============================================================================

class ConvolverError(FitterError):
    """Base exception for all macroscopic spectral broadener and convolution operations."""


class ProfileError(ConvolverError):
    """Raised when line profile parameters (FWHM, profile type, cutoff) are invalid or non-physical."""


class ThermalScalingError(ConvolverError):
    """Raised when temperature parameters (T <= 0, reference temp <= 0) or Boltzmann factors are invalid."""


class MicroBatchError(ConvolverError):
    """Raised when micro-batching calculations, chunk sizing, or scan allocations fail."""


class InvalidGridError(ConvolverError):
    """Raised when frequency grids are malformed, non-monotonic, empty, or degenerate."""


class SpectralDataError(ConvolverError):
    """Raised when spectral transition lines data (frequencies, intensities, energies) are invalid or mismatched."""


# =============================================================================
# 3. ENUMS & PYDANTIC CONFIGURATIONS
# =============================================================================

class ProfileType(str, enum.Enum):
    """Supported line profile line-shape types."""
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"


class EnergyUnit(str, enum.Enum):
    """Units of energy and frequency for spectroscopic transitions."""
    CM_1 = "cm-1"
    MHZ = "mhz"
    GHZ = "ghz"
    JOULE = "joule"
    KELVIN = "kelvin"


def get_energy_conversion_factor_to_kelvin(unit: Union[EnergyUnit, str]) -> float:
    """
    Return conversion factor to multiply a quantity by to obtain Kelvin (E / k_B).

    Parameters:
        unit: EnergyUnit enum or string representation.

    Returns:
        float: Multiplier factor converting given unit into Kelvin.
    """
    if isinstance(unit, EnergyUnit):
        unit_str = unit.value.lower().strip()
    else:
        unit_str = str(unit).lower().strip().replace("_", "-")

    if unit_str in ("cm-1", "cm_1", "wavenumber", "energyunit.cm-1", "energyunit.cm_1"):
        return C2_RADIATION_CONST
    elif unit_str in ("mhz", "megahertz", "energyunit.mhz"):
        return H_OVER_K_B * 1.0e6
    elif unit_str in ("ghz", "gigahertz", "energyunit.ghz"):
        return H_OVER_K_B * 1.0e9
    elif unit_str in ("joule", "j", "energyunit.joule"):
        return 1.0 / BOLTZMANN_K
    elif unit_str in ("kelvin", "k", "energyunit.kelvin"):
        return 1.0
    else:
        raise ThermalScalingError(f"Unsupported energy/frequency unit: '{unit}'")


class SpectralLines(BaseModel):
    """Container for molecular / atomic transition line catalog data."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequencies: List[float] = Field(..., description="Transition line center frequencies (positive)")
    intensities: List[float] = Field(..., description="Integrated line intensities at reference temperature")
    lower_state_energies: Optional[List[float]] = Field(
        default=None, description="Lower state energy levels E_lower"
    )
    quantum_numbers: Optional[List[str]] = Field(
        default=None, description="Quantum state assignment labels"
    )
    fwhm_gaussian_overrides: Optional[List[float]] = Field(
        default=None, description="Per-line Gaussian FWHM overrides"
    )
    fwhm_lorentzian_overrides: Optional[List[float]] = Field(
        default=None, description="Per-line Lorentzian FWHM overrides"
    )

    @field_validator("frequencies")
    @classmethod
    def validate_frequencies(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise SpectralDataError("Spectral lines frequency list cannot be empty.")
        for idx, freq in enumerate(v):
            if not math.isfinite(freq) or freq <= 0.0:
                raise SpectralDataError(f"Transition frequency at index {idx} ({freq}) must be finite and positive (> 0).")
        return v

    @field_validator("intensities")
    @classmethod
    def validate_intensities(cls, v: List[float]) -> List[float]:
        for idx, intensity in enumerate(v):
            if not math.isfinite(intensity) or intensity < 0.0:
                raise SpectralDataError(f"Transition intensity at index {idx} ({intensity}) must be finite and non-negative (>= 0).")
        return v

    @model_validator(mode="after")
    def validate_lengths_consistency(self) -> SpectralLines:
        n_lines = len(self.frequencies)
        if len(self.intensities) != n_lines:
            raise SpectralDataError(
                f"Mismatch: frequencies length ({n_lines}) does not match intensities length ({len(self.intensities)})."
            )
        if self.lower_state_energies is not None:
            if len(self.lower_state_energies) != n_lines:
                raise SpectralDataError(
                    f"Mismatch: lower_state_energies length ({len(self.lower_state_energies)}) does not match frequencies length ({n_lines})."
                )
            for idx, e in enumerate(self.lower_state_energies):
                if not math.isfinite(e) or e < 0.0:
                    raise SpectralDataError(f"lower_state_energies at index {idx} ({e}) must be finite and non-negative (>= 0).")
        if self.quantum_numbers is not None and len(self.quantum_numbers) != n_lines:
            raise SpectralDataError(
                f"Mismatch: quantum_numbers length ({len(self.quantum_numbers)}) does not match frequencies length ({n_lines})."
            )
        if self.fwhm_gaussian_overrides is not None:
            if len(self.fwhm_gaussian_overrides) != n_lines:
                raise SpectralDataError(
                    f"Mismatch: fwhm_gaussian_overrides length ({len(self.fwhm_gaussian_overrides)}) does not match frequencies ({n_lines})."
                )
            for idx, w in enumerate(self.fwhm_gaussian_overrides):
                if not math.isfinite(w) or w <= 0.0:
                    raise ProfileError(f"fwhm_gaussian_overrides at index {idx} ({w}) must be positive.")
        if self.fwhm_lorentzian_overrides is not None:
            if len(self.fwhm_lorentzian_overrides) != n_lines:
                raise SpectralDataError(
                    f"Mismatch: fwhm_lorentzian_overrides length ({len(self.fwhm_lorentzian_overrides)}) does not match frequencies ({n_lines})."
                )
            for idx, w in enumerate(self.fwhm_lorentzian_overrides):
                if not math.isfinite(w) or w <= 0.0:
                    raise ProfileError(f"fwhm_lorentzian_overrides at index {idx} ({w}) must be positive.")
        return self


class ConvolutionConfig(BaseModel):
    """Configuration for macroscopic spectral line broadening and JAX convolution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    profile_type: ProfileType = Field(default=ProfileType.GAUSSIAN, description="Line profile shape function")
    fwhm_gaussian: float = Field(default=0.01, gt=0.0, description="Global Gaussian FWHM (Doppler width)")
    fwhm_lorentzian: float = Field(default=0.01, gt=0.0, description="Global Lorentzian FWHM (Collisional width)")
    grid_resolution: Optional[float] = Field(default=None, gt=0.0, description="Dense frequency grid spacing")
    grid_points: Optional[int] = Field(default=None, ge=2, description="Explicit number of frequency grid points")
    grid_min: Optional[float] = Field(default=None, description="Lower bound of frequency domain")
    grid_max: Optional[float] = Field(default=None, description="Upper bound of frequency domain")
    cutoff_fwhm: float = Field(default=25.0, gt=0.0, description="Broadening domain margin in units of FWHM")
    chunk_size: Optional[int] = Field(default=None, ge=1, description="Explicit chunk size for micro-batching")
    max_vram_mb: float = Field(default=DEFAULT_MAX_VRAM_MB, gt=0.0, description="VRAM / RAM budget cap in Megabytes")
    t_ref: float = Field(default=298.15, gt=0.0, description="Reference temperature in Kelvin (default 298.15 K)")
    energy_unit: EnergyUnit = Field(default=EnergyUnit.CM_1, description="Unit of lower state energies")
    freq_unit: EnergyUnit = Field(default=EnergyUnit.MHZ, description="Unit of transition frequencies")
    normalize_integrated_area: bool = Field(default=True, description="Enforce analytical unit area normalization")

    @model_validator(mode="after")
    def validate_grid_bounds(self) -> ConvolutionConfig:
        if self.grid_min is not None and self.grid_max is not None:
            if self.grid_min >= self.grid_max:
                raise InvalidGridError(
                    f"grid_min ({self.grid_min}) must be strictly less than grid_max ({self.grid_max})."
                )
        return self


class PartitionGridConfig(BaseModel):
    """Configuration for multi-temperature partition function evaluation and Boltzmann scaling."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    temperatures: List[float] = Field(..., description="Target temperatures in Kelvin (all > 0)")
    t_ref: float = Field(default=298.15, gt=0.0, description="Reference temperature in Kelvin")
    power_law_index: Optional[float] = Field(
        default=None, description="Power law exponent n for Q(T) = Q(T_ref) * (T / T_ref)^n"
    )
    q_ref: float = Field(default=1.0, gt=0.0, description="Partition function value at reference temperature Q(T_ref)")
    partition_values: Optional[List[float]] = Field(
        default=None, description="Explicit tabulated partition function values Q(T) for each temperature"
    )
    energy_levels: Optional[List[float]] = Field(
        default=None, description="Energy levels for direct state-sum partition function calculation"
    )
    degeneracies: Optional[List[float]] = Field(
        default=None, description="State degeneracies g_k corresponding to energy_levels"
    )
    energy_unit: EnergyUnit = Field(default=EnergyUnit.CM_1, description="Unit of energy levels and lower state energies")

    @field_validator("temperatures")
    @classmethod
    def validate_temperatures(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ThermalScalingError("Temperature grid list cannot be empty.")
        for idx, temp in enumerate(v):
            if not math.isfinite(temp) or temp <= 0.0:
                raise ThermalScalingError(f"Temperature at index {idx} ({temp} K) must be finite and strictly positive (> 0).")
        return v

    @model_validator(mode="after")
    def validate_partition_options(self) -> PartitionGridConfig:
        n_temps = len(self.temperatures)
        if self.partition_values is not None:
            if len(self.partition_values) != n_temps:
                raise ThermalScalingError(
                    f"Mismatch: partition_values length ({len(self.partition_values)}) does not match temperatures length ({n_temps})."
                )
            for idx, q_val in enumerate(self.partition_values):
                if not math.isfinite(q_val) or q_val <= 0.0:
                    raise ThermalScalingError(f"Partition value at index {idx} ({q_val}) must be positive (> 0).")
        if self.energy_levels is not None:
            if self.degeneracies is None:
                raise ThermalScalingError("State degeneracies must be provided when energy_levels are supplied.")
            if len(self.energy_levels) != len(self.degeneracies):
                raise ThermalScalingError(
                    f"Mismatch: energy_levels length ({len(self.energy_levels)}) does not match degeneracies length ({len(self.degeneracies)})."
                )
            if len(self.energy_levels) == 0:
                raise ThermalScalingError("energy_levels list cannot be empty.")
            for idx, e in enumerate(self.energy_levels):
                if not math.isfinite(e) or e < 0.0:
                    raise ThermalScalingError(f"energy_levels at index {idx} ({e}) must be finite and non-negative (>= 0).")
            for idx, g in enumerate(self.degeneracies):
                if not math.isfinite(g) or g <= 0.0:
                    raise ThermalScalingError(f"degeneracies at index {idx} ({g}) must be finite and strictly positive (> 0).")
        return self


class PartitionGridResult(BaseModel):
    """Results of multi-temperature partition function and Boltzmann scaling evaluation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    temperatures: List[float] = Field(..., description="Evaluated temperatures in Kelvin")
    partition_function_values: List[float] = Field(..., description="Partition function values Q(T)")
    scaled_intensities: List[List[float]] = Field(
        ..., description="Temperature-scaled intensities matrix of shape [N_temps, N_lines]"
    )
    boltzmann_factors: List[List[float]] = Field(
        ..., description="Boltzmann state population weighting factors [N_temps, N_lines]"
    )
    stimulated_emission_factors: List[List[float]] = Field(
        ..., description="Stimulated emission correction factors [N_temps, N_lines]"
    )


class ConvolvedSpectrumResult(BaseModel):
    """Record of convolved macroscopic spectrum output."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequencies: List[float] = Field(..., description="Dense 1D frequency grid")
    intensities: List[float] = Field(..., description="1D convolved spectrum intensities at primary/reference temperature")
    temperatures: Optional[List[float]] = Field(default=None, description="Temperatures for multi-temperature grids")
    multi_temp_intensities: Optional[List[List[float]]] = Field(
        default=None, description="Multi-temperature convolved spectra matrix [N_temps, N_freqs]"
    )
    profile_type: ProfileType = Field(..., description="Line profile shape used for convolution")
    total_integrated_intensity: float = Field(..., description="Trapezoidal integrated total intensity across frequency grid")
    peak_frequency: float = Field(..., description="Frequency location of maximum spectral peak")
    peak_intensity: float = Field(..., description="Maximum spectral peak intensity")
    chunk_size_used: int = Field(..., ge=1, description="Micro-batch chunk size used for scan execution")
    num_chunks: int = Field(..., ge=1, description="Total number of micro-batch scan chunks")
    execution_time_seconds: float = Field(..., ge=0.0, description="Total computation wall-clock time in seconds")


# =============================================================================
# 4. PURE-JAX ANALYTICAL LINE PROFILE KERNELS (FP64, @jax.jit)
# =============================================================================

# Precalculated Weideman (1994) 32-term rational approximation constants for Faddeeva w(z)
_WEIDEMAN_N: int = 32
_WEIDEMAN_L: float = float(np.sqrt(_WEIDEMAN_N / np.sqrt(2.0)))
_WEIDEMAN_M: int = 2 * _WEIDEMAN_N
_WEIDEMAN_M2: int = 2 * _WEIDEMAN_M
_WEIDEMAN_K: np.ndarray = np.arange(-_WEIDEMAN_M + 1, _WEIDEMAN_M)
_WEIDEMAN_THETA: np.ndarray = _WEIDEMAN_K * np.pi / _WEIDEMAN_M
_WEIDEMAN_T: np.ndarray = _WEIDEMAN_L * np.tan(_WEIDEMAN_THETA / 2.0)
_WEIDEMAN_F: np.ndarray = np.exp(-_WEIDEMAN_T**2) * (_WEIDEMAN_L**2 + _WEIDEMAN_T**2)
_WEIDEMAN_F_PADDED: np.ndarray = np.concatenate([[0.0], _WEIDEMAN_F])
_WEIDEMAN_A: np.ndarray = np.real(np.fft.fft(np.fft.fftshift(_WEIDEMAN_F_PADDED))) / _WEIDEMAN_M2
_WEIDEMAN_COEFFS_JAX: jnp.ndarray = jnp.array(_WEIDEMAN_A[1:_WEIDEMAN_N + 1], dtype=jnp.float64)
_WEIDEMAN_COEFFS_REVERSED: jnp.ndarray = _WEIDEMAN_COEFFS_JAX[:-1][::-1]


@jax.jit
def faddeeva_rational_jax(z: jnp.ndarray) -> jnp.ndarray:
    """
    Pure-JAX Weideman 32-term rational approximation for the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z).

    Valid across the upper complex half-plane (Im(z) >= 0) with machine accuracy (~ 1e-14) [M].
    Uses jax.lax.scan for Horner polynomial evaluation to ensure pure XLA compilation without Python loops.

    Parameters:
        z: Complex JAX array with Im(z) >= 0.

    Returns:
        jnp.ndarray: Complex values of the Faddeeva function w(z).
    """
    capital_z = (_WEIDEMAN_L + 1j * z) / (_WEIDEMAN_L - 1j * z)

    def horner_step(carry_poly: jnp.ndarray, coeff: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        next_poly = carry_poly * capital_z + coeff
        return next_poly, None

    init_poly = jnp.full_like(z, _WEIDEMAN_COEFFS_JAX[_WEIDEMAN_N - 1], dtype=jnp.complex128)
    final_poly, _ = jax.lax.scan(horner_step, init_poly, _WEIDEMAN_COEFFS_REVERSED)

    denom = _WEIDEMAN_L - 1j * z
    w_val = 2.0 * final_poly / (denom * denom) + (1.0 / (jnp.sqrt(jnp.pi) * denom))
    return w_val


@jax.jit
def gaussian_kernel_jax(delta_nu: jnp.ndarray, fwhm_g: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical normalized Gaussian line profile g(nu - nu_0; FWHM_G).

    g(delta_nu) = (2 * sqrt(ln 2) / (sqrt(pi) * FWHM_G)) * exp(-4 * ln 2 * (delta_nu / FWHM_G)^2)

    Properties:
    - Peak at delta_nu = 0: height = 2 * sqrt(ln 2) / (sqrt(pi) * FWHM_G)
    - Half-maximum at delta_nu = +/- FWHM_G / 2: value = Peak / 2
    - Normalized area: int_{-\\infty}^{\\infty} g(delta_nu) d(delta_nu) = 1.0

    Parameters:
        delta_nu: Frequency displacement (nu - nu_0).
        fwhm_g: Full Width at Half Maximum of Gaussian component (> 0).

    Returns:
        jnp.ndarray: Normalized Gaussian profile values.
    """
    safe_fwhm = jnp.maximum(fwhm_g, 1e-18)
    prefactor = (2.0 * jnp.sqrt(jnp.log(2.0))) / (jnp.sqrt(jnp.pi) * safe_fwhm)
    exponent = -FOUR_LN_2 * jnp.square(delta_nu / safe_fwhm)
    return prefactor * jnp.exp(exponent)


@jax.jit
def lorentzian_kernel_jax(delta_nu: jnp.ndarray, fwhm_l: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical normalized Lorentzian line profile L(nu - nu_0; FWHM_L).

    L(delta_nu) = (2 / (pi * FWHM_L)) / (1 + 4 * (delta_nu / FWHM_L)^2)

    Properties:
    - Peak at delta_nu = 0: height = 2 / (pi * FWHM_L)
    - Half-maximum at delta_nu = +/- FWHM_L / 2: value = Peak / 2
    - Normalized area: int_{-\\infty}^{\\infty} L(delta_nu) d(delta_nu) = 1.0

    Parameters:
        delta_nu: Frequency displacement (nu - nu_0).
        fwhm_l: Full Width at Half Maximum of Lorentzian component (> 0).

    Returns:
        jnp.ndarray: Normalized Lorentzian profile values.
    """
    safe_fwhm = jnp.maximum(fwhm_l, 1e-18)
    prefactor = 2.0 / (jnp.pi * safe_fwhm)
    denominator = 1.0 + 4.0 * jnp.square(delta_nu / safe_fwhm)
    return prefactor / denominator


@jax.jit
def pseudo_voigt_kernel_jax(
    delta_nu: jnp.ndarray, fwhm_g: jnp.ndarray, fwhm_l: jnp.ndarray
) -> jnp.ndarray:
    """
    Thompson-Cox-Hastings (TCH) analytical Pseudo-Voigt line profile.

    Computes the Voigt FWHM Gamma_V and mixing parameter eta analytically:
    Gamma_V = (Gamma_G^5 + 2.69269*Gamma_G^4*Gamma_L + 2.42843*Gamma_G^3*Gamma_L^2
              + 4.47163*Gamma_G^2*Gamma_L^3 + 0.07842*Gamma_G*Gamma_L^4 + Gamma_L^5)^0.2
    eta = 1.36603 * (Gamma_L / Gamma_V) - 0.47719 * (Gamma_L / Gamma_V)^2 + 0.11116 * (Gamma_L / Gamma_V)^3
    pV(delta_nu) = eta * L(delta_nu; Gamma_V) + (1 - eta) * G(delta_nu; Gamma_V)

    Properties:
    - Area is strictly 1.0.
    - Smooth, differentiable, pure JAX, no non-XLA special functions.

    Parameters:
        delta_nu: Frequency displacement (nu - nu_0).
        fwhm_g: Gaussian component FWHM (> 0).
        fwhm_l: Lorentzian component FWHM (> 0).

    Returns:
        jnp.ndarray: Normalized Pseudo-Voigt profile values.
    """
    fg = jnp.maximum(fwhm_g, 1e-18)
    fl = jnp.maximum(fwhm_l, 1e-18)

    fg2 = fg * fg
    fg3 = fg2 * fg
    fg4 = fg3 * fg
    fg5 = fg4 * fg

    fl2 = fl * fl
    fl3 = fl2 * fl
    fl4 = fl3 * fl
    fl5 = fl4 * fl

    fv5 = fg5 + 2.69269 * fg4 * fl + 2.42843 * fg3 * fl2 + 4.47163 * fg2 * fl3 + 0.07842 * fg * fl4 + fl5
    fv = jnp.power(fv5, 0.2)

    rho = fl / fv
    eta = 1.36603 * rho - 0.47719 * (rho * rho) + 0.11116 * (rho * rho * rho)
    eta = jnp.clip(eta, 0.0, 1.0)

    g_comp = gaussian_kernel_jax(delta_nu, fv)
    l_comp = lorentzian_kernel_jax(delta_nu, fv)
    return eta * l_comp + (1.0 - eta) * g_comp


@jax.jit
def faddeeva_voigt_kernel_jax(
    delta_nu: jnp.ndarray, fwhm_g: jnp.ndarray, fwhm_l: jnp.ndarray
) -> jnp.ndarray:
    """
    Exact analytical Voigt line profile using pure-JAX Weideman Faddeeva evaluation.

    V(delta_nu; sigma, gamma) = Re[w(z)] / (sigma * sqrt(2 * pi))
    where sigma = FWHM_G / (2 * sqrt(2 * ln 2)), gamma = FWHM_L / 2, z = (delta_nu + i * gamma) / (sigma * sqrt(2)).

    Parameters:
        delta_nu: Frequency displacement (nu - nu_0).
        fwhm_g: Gaussian component FWHM (> 0).
        fwhm_l: Lorentzian component FWHM (> 0).

    Returns:
        jnp.ndarray: Normalized exact Voigt profile values.
    """
    fg = jnp.maximum(fwhm_g, 1e-18)
    fl = jnp.maximum(fwhm_l, 1e-18)

    sigma = fg * FWHM_TO_SIGMA_GAUSS
    gamma = 0.5 * fl

    z = (delta_nu + 1j * gamma) / (sigma * jnp.sqrt(2.0))
    w_val = faddeeva_rational_jax(z)
    voigt_val = jnp.real(w_val) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    return voigt_val


@jax.jit
def true_voigt_profile(
    x: Union[float, jnp.ndarray, np.ndarray],
    sigma: Union[float, jnp.ndarray, np.ndarray],
    gamma: Union[float, jnp.ndarray, np.ndarray],
) -> jnp.ndarray:
    """
    Analytical exact Voigt line profile using pure-JAX Faddeeva evaluation via jax.scipy.special.erfcx.

    Evaluates the exact Voigt profile V(x; sigma, gamma) [M]:
        V(x; sigma, gamma) = Re[erfcx((gamma - i*x) / (sigma * sqrt(2)))] / (sigma * sqrt(2*pi))

    Parameters:
        x: Frequency displacement (x = nu - nu_0) or array of displacements.
        sigma: Gaussian standard deviation (sigma > 0).
        gamma: Lorentzian HWHM (half-width at half-maximum, gamma >= 0).

    Returns:
        jnp.ndarray: Voigt profile values in FP64 precision.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)
    gamma_arr = jnp.asarray(gamma, dtype=jnp.float64)

    safe_sigma = jnp.maximum(sigma_arr, 1e-30)
    safe_gamma = jnp.maximum(gamma_arr, 0.0)

    u = (safe_gamma - 1j * x_arr) / (safe_sigma * jnp.sqrt(2.0))
    w_z = jsp.erfcx(u)
    voigt_val = jnp.real(w_z) / (safe_sigma * jnp.sqrt(2.0 * jnp.pi))
    return voigt_val


@jax.jit
def evaluate_profile_kernel_jax(
    delta_nu: Any,
    fwhm_g: Any,
    fwhm_l: Any,
    profile_code: Any,
) -> jnp.ndarray:
    """
    Unified JAX kernel evaluating the requested line profile shape.

    profile_code:
        0: GAUSSIAN
        1: LORENTZIAN
        2: PSEUDO_VOIGT
        3: VOIGT (Faddeeva)

    Parameters:
        delta_nu: Frequency displacement tensor.
        fwhm_g: Gaussian FWHM tensor or scalar.
        fwhm_l: Lorentzian FWHM tensor or scalar.
        profile_code: Integer opcode selecting line profile.

    Returns:
        jnp.ndarray: Evaluated profile tensor.
    """
    def eval_gaussian() -> jnp.ndarray:
        return gaussian_kernel_jax(delta_nu, fwhm_g)

    def eval_lorentzian() -> jnp.ndarray:
        return lorentzian_kernel_jax(delta_nu, fwhm_l)

    def eval_pseudo_voigt() -> jnp.ndarray:
        return pseudo_voigt_kernel_jax(delta_nu, fwhm_g, fwhm_l)

    def eval_faddeeva_voigt() -> jnp.ndarray:
        return faddeeva_voigt_kernel_jax(delta_nu, fwhm_g, fwhm_l)

    return jax.lax.switch(
        profile_code,
        [eval_gaussian, eval_lorentzian, eval_pseudo_voigt, eval_faddeeva_voigt],
    )


# =============================================================================
# 5. MULTI-TEMPERATURE PARTITION GRIDS & BOLTZMANN SCALING (PURE JAX)
# =============================================================================

@jax.jit
def evaluate_partition_function_jax(
    temperatures: Any,
    power_law_index: Optional[Any] = None,
    q_ref: Any = 1.0,
    t_ref: Any = 298.15,
    energy_levels: Optional[Any] = None,
    degeneracies: Optional[Any] = None,
    energy_to_k: Any = C2_RADIATION_CONST,
) -> jnp.ndarray:
    """
    Pure-JAX vectorized partition function Q(T) calculation across multi-temperature grids.

    Parameters:
        temperatures: Array of temperatures in Kelvin [N_temps].
        power_law_index: Power-law index n (if using Q(T) = Q(T_ref) * (T/T_ref)^n).
        q_ref: Reference partition function value Q(T_ref).
        t_ref: Reference temperature in Kelvin.
        energy_levels: Energy levels [N_states] for direct state-sum.
        degeneracies: Degeneracies [N_states] for direct state-sum.
        energy_to_k: Multiplier converting energy unit to Kelvin.

    Returns:
        jnp.ndarray: Evaluated partition functions Q(T) [N_temps].
    """
    n_temps = temperatures.shape[0]

    # If energy levels and degeneracies are supplied: Q(T) = sum_k g_k * exp(-c2 * E_k / T)
    if energy_levels is not None and degeneracies is not None:
        # T_grid: (N_temps, 1), E_grid: (1, N_states), G_grid: (1, N_states)
        t_col = temperatures[:, None]
        e_row = energy_levels[None, :] * energy_to_k
        g_row = degeneracies[None, :]
        boltzmann_states = jnp.exp(-e_row / t_col)
        q_temps = jnp.sum(g_row * boltzmann_states, axis=1)
        return q_temps

    # Else if power-law index is supplied: Q(T) = Q(T_ref) * (T / T_ref)^n
    if power_law_index is not None:
        q_temps = q_ref * jnp.power(temperatures / t_ref, power_law_index)
        return q_temps

    # Default constant Q(T) = Q(T_ref)
    return jnp.full((n_temps,), q_ref, dtype=jnp.float64)


@jax.jit
def compute_boltzmann_scaling_grid_jax(
    frequencies: jnp.ndarray,
    lower_energies: jnp.ndarray,
    reference_intensities: jnp.ndarray,
    temperatures: jnp.ndarray,
    q_temps: jnp.ndarray,
    t_ref: float,
    q_ref: float,
    freq_to_k: float,
    energy_to_k: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized multi-temperature Boltzmann scaling and stimulated emission corrections in pure JAX.

    Formula for each transition line i and temperature T_j:
    S_ij(T_j) = S_i(T_ref) * [Q(T_ref) / Q(T_j)] * exp(-c2 * E_lower_i * (1/T_j - 1/T_ref))
                * [1 - exp(-h * nu_i / (k_B * T_j))] / [1 - exp(-h * nu_i / (k_B * T_ref))]

    Utilizes -jnp.expm1(-x) for high numerical precision when h * nu / (k_B * T) << 1.

    Parameters:
        frequencies: Transition frequencies [N_lines].
        lower_energies: Lower state energies [N_lines].
        reference_intensities: Reference intensities S(T_ref) [N_lines].
        temperatures: Target temperatures [N_temps].
        q_temps: Partition function values Q(T) [N_temps].
        t_ref: Reference temperature in Kelvin.
        q_ref: Reference partition function Q(T_ref).
        freq_to_k: Conversion factor converting frequency to Kelvin (h * nu / k_B).
        energy_to_k: Conversion factor converting lower state energy to Kelvin (E / k_B).

    Returns:
        Tuple of:
        - scaled_intensities: [N_temps, N_lines]
        - boltzmann_factors: [N_temps, N_lines]
        - stimulated_emission_factors: [N_temps, N_lines]
    """
    # Shapes:
    # temperatures: (N_temps, 1)
    # q_temps: (N_temps, 1)
    # lower_energies: (1, N_lines)
    # frequencies: (1, N_lines)
    # reference_intensities: (1, N_lines)
    t_col = temperatures[:, None]
    q_col = q_temps[:, None]
    e_row = lower_energies[None, :]
    nu_row = frequencies[None, :]
    i_row = reference_intensities[None, :]

    # 1. Partition function ratio: Q(T_ref) / Q(T)
    partition_ratio = q_ref / q_col  # (N_temps, 1)

    # 2. Boltzmann state population factor: exp(- (E_lower / k_B) * (1/T - 1/T_ref))
    inv_t_diff = (1.0 / t_col) - (1.0 / t_ref)
    boltzmann_factors = jnp.exp(-energy_to_k * e_row * inv_t_diff)  # (N_temps, N_lines)

    # 3. Stimulated emission correction: [1 - exp(-h*nu / k_B*T)] / [1 - exp(-h*nu / k_B*T_ref)]
    # Use -expm1(-x) for robust stability across all frequency / temperature domains
    x_t = (nu_row * freq_to_k) / t_col
    x_ref = (nu_row * freq_to_k) / t_ref

    num_stim = -jnp.expm1(-x_t)
    den_stim = -jnp.expm1(-x_ref)
    safe_den_stim = jnp.where(jnp.abs(den_stim) < 1e-30, 1e-30, den_stim)
    stimulated_factors = num_stim / safe_den_stim  # (N_temps, N_lines)

    # Total scaled intensities [N_temps, N_lines]
    scaled_intensities = i_row * partition_ratio * boltzmann_factors * stimulated_factors
    return scaled_intensities, boltzmann_factors, stimulated_factors


@jax.jit
def convolute_multi_temperature(
    frequencies: jnp.ndarray,
    intensities: jnp.ndarray,
    Q_T: jnp.ndarray,
    temp_grid: jnp.ndarray,
    lower_energies: Optional[jnp.ndarray] = None,
    t_ref: float = 298.15,
    q_ref: float = 1.0,
    energy_to_k: float = C2_RADIATION_CONST,
    freq_to_k: float = H_OVER_K_B * 1.0e6,
) -> jnp.ndarray:
    """
    Vectorized multi-temperature Boltzmann scaling with tensor broadcasting [N_lines, N_temperatures].

    Computes the temperature-scaled spectral line intensities for all lines across all target temperatures:
        S_ij(T_j) = S_i(T_ref) * [Q(T_ref) / Q(T_j)] * exp(- (E_lower_i / k_B) * (1/T_j - 1/T_ref))
                    * [1 - exp(-h * nu_i / (k_B * T_j))] / [1 - exp(-h * nu_i / (k_B * T_ref))]

    Utilizes JAX tensor broadcasting between lines (N_lines, 1) and temperatures (1, N_temperatures)
    with numerically stable -jnp.expm1(-x) evaluation.

    Parameters:
        frequencies: Transition frequencies [N_lines].
        intensities: Reference intensities S(T_ref) [N_lines].
        Q_T: Partition function values Q(T) at target temperatures [N_temperatures].
        temp_grid: Target temperatures in Kelvin [N_temperatures].
        lower_energies: Lower state energies [N_lines]. If None, assumed 0.0.
        t_ref: Reference temperature in Kelvin (default 296 K).
        q_ref: Reference partition function value Q(T_ref) (default 1.0).
        energy_to_k: Multiplier converting energy unit to Kelvin (default C2_RADIATION_CONST for cm^-1).
        freq_to_k: Multiplier converting frequency unit to Kelvin (default MHz -> K).

    Returns:
        jnp.ndarray: Scaled intensities tensor of shape [N_lines, N_temperatures].
    """
    f_arr = jnp.asarray(frequencies, dtype=jnp.float64)
    i_arr = jnp.asarray(intensities, dtype=jnp.float64)
    q_arr = jnp.asarray(Q_T, dtype=jnp.float64)
    t_arr = jnp.asarray(temp_grid, dtype=jnp.float64)

    if lower_energies is not None:
        e_arr = jnp.asarray(lower_energies, dtype=jnp.float64)
    else:
        e_arr = jnp.zeros_like(f_arr)

    # Tensor broadcasting:
    # Lines: (N_lines, 1)
    # Temps: (1, N_temperatures)
    f_col = f_arr[:, None]
    i_col = i_arr[:, None]
    e_col = e_arr[:, None]
    q_row = q_arr[None, :]
    t_row = t_arr[None, :]

    # 1. Partition function ratio: Q(T_ref) / Q(T) -> shape (1, N_temperatures)
    q_ratio = q_ref / q_row

    # 2. Boltzmann state population factor -> shape (N_lines, N_temperatures)
    inv_t_diff = (1.0 / t_row) - (1.0 / t_ref)
    boltzmann = jnp.exp(-energy_to_k * e_col * inv_t_diff)

    # 3. Stimulated emission correction -> shape (N_lines, N_temperatures)
    x_t = (f_col * freq_to_k) / t_row
    x_ref = (f_col * freq_to_k) / t_ref

    num_stim = -jnp.expm1(-x_t)
    den_stim = -jnp.expm1(-x_ref)
    safe_den_stim = jnp.where(jnp.abs(den_stim) < 1e-30, 1e-30, den_stim)
    stimulated = num_stim / safe_den_stim

    # Scaled intensities: [N_lines, N_temperatures]
    scaled_intensities = i_col * q_ratio * boltzmann * stimulated
    return scaled_intensities


# =============================================================================
# 6. MEMORY-AWARE MICRO-BATCHING VIA JAX.LAX.SCAN
# =============================================================================

def calculate_optimal_chunk_size(
    n_freqs: int,
    n_lines: int,
    n_temps: int = 1,
    max_vram_mb: float = DEFAULT_MAX_VRAM_MB,
    explicit_chunk_size: Optional[int] = None,
) -> int:
    """
    Calculate optimal frequency micro-batch chunk size based on memory budget cap.

    Prevents VRAM/RAM exhaustion during dense grid convolutions against large line catalogs.

    Parameters:
        n_freqs: Number of frequency grid points.
        n_lines: Number of spectral transition lines.
        n_temps: Number of temperature points.
        max_vram_mb: Maximum memory allocation budget in Megabytes.
        explicit_chunk_size: Explicit chunk size override (if specified).

    Returns:
        int: Optimal micro-batch chunk size in grid points.
    """
    if explicit_chunk_size is not None:
        if explicit_chunk_size <= 0:
            raise MicroBatchError(f"Explicit chunk_size ({explicit_chunk_size}) must be strictly positive (> 0).")
        return min(explicit_chunk_size, n_freqs)

    if max_vram_mb <= 0.0:
        raise MicroBatchError(f"max_vram_mb ({max_vram_mb}) must be strictly positive (> 0).")

    # Memory per frequency point in float64 (8 bytes):
    # - Distance delta_nu: chunk_size * n_lines * 8
    # - Line profile tensor: chunk_size * n_lines * 8
    # - Output spectrum slice: chunk_size * n_temps * 8
    # Safety factor of 4.0 accounts for intermediate compiler temporary buffers
    bytes_budget = max_vram_mb * 1024.0 * 1024.0
    bytes_per_freq_point = (4.0 * n_lines + n_temps) * 8.0

    calculated_chunk = int(bytes_budget / bytes_per_freq_point)
    # Ensure chunk_size is at least 1 and capped at n_freqs
    optimal_chunk = max(1, min(n_freqs, calculated_chunk))
    return optimal_chunk


@jax.jit
def convolve_frequency_chunks_scan_1d_jax(
    freq_grid_padded: jnp.ndarray,
    line_freqs: jnp.ndarray,
    line_intensities: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_l: jnp.ndarray,
    profile_code: int,
) -> jnp.ndarray:
    """
    Micro-batched spectral convolution for 1D intensities using jax.lax.scan without Python loops.

    Padded frequency grid has shape [num_chunks, chunk_size].

    Parameters:
        freq_grid_padded: [num_chunks, chunk_size] frequency chunks.
        line_freqs: [N_lines] transition line centers.
        line_intensities: [N_lines] transition line intensities.
        fwhm_g: [N_lines] or scalar Gaussian FWHM.
        fwhm_l: [N_lines] or scalar Lorentzian FWHM.
        profile_code: Integer opcode for profile selection.

    Returns:
        jnp.ndarray: Convolved spectrum chunks of shape [num_chunks, chunk_size].
    """
    def scan_chunk_step(carry: None, freq_chunk: jnp.ndarray) -> Tuple[None, jnp.ndarray]:
        # freq_chunk: (chunk_size,)
        # line_freqs: (N_lines,)
        # delta_nu: (chunk_size, N_lines)
        delta_nu = freq_chunk[:, None] - line_freqs[None, :]

        # Evaluate vectorized line profile
        profiles = evaluate_profile_kernel_jax(delta_nu, fwhm_g, fwhm_l, profile_code)

        # Dot product with intensities: (chunk_size,)
        chunk_spectrum = jnp.dot(profiles, line_intensities)
        return carry, chunk_spectrum

    _, chunked_spectra = jax.lax.scan(scan_chunk_step, None, freq_grid_padded)
    return chunked_spectra


@jax.jit
def convolve_frequency_chunks_scan_2d_jax(
    freq_grid_padded: jnp.ndarray,
    line_freqs: jnp.ndarray,
    scaled_intensities: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_l: jnp.ndarray,
    profile_code: int,
) -> jnp.ndarray:
    """
    Micro-batched spectral convolution for 2D multi-temperature intensities using jax.lax.scan.

    Parameters:
        freq_grid_padded: [num_chunks, chunk_size] frequency chunks.
        line_freqs: [N_lines] transition line centers.
        scaled_intensities: [N_temps, N_lines] multi-temperature intensities.
        fwhm_g: [N_lines] or scalar Gaussian FWHM.
        fwhm_l: [N_lines] or scalar Lorentzian FWHM.
        profile_code: Integer opcode for profile selection.

    Returns:
        jnp.ndarray: Convolved spectrum chunks of shape [num_chunks, N_temps, chunk_size].
    """
    def scan_chunk_step_2d(carry: None, freq_chunk: jnp.ndarray) -> Tuple[None, jnp.ndarray]:
        # freq_chunk: (chunk_size,)
        # delta_nu: (chunk_size, N_lines)
        delta_nu = freq_chunk[:, None] - line_freqs[None, :]

        # profiles: (chunk_size, N_lines)
        profiles = evaluate_profile_kernel_jax(delta_nu, fwhm_g, fwhm_l, profile_code)

        # scaled_intensities: (N_temps, N_lines)
        # result: (N_temps, chunk_size) = matmul(scaled_intensities, profiles.T)
        chunk_spectrum_2d = jnp.matmul(scaled_intensities, profiles.T)
        return carry, chunk_spectrum_2d

    _, chunked_spectra_2d = jax.lax.scan(scan_chunk_step_2d, None, freq_grid_padded)
    return chunked_spectra_2d


@jax.jit
def convolve_line_chunks_scan_1d_jax(
    freq_grid: jnp.ndarray,
    line_freqs_padded: jnp.ndarray,
    line_intensities_padded: jnp.ndarray,
    fwhm_g_padded: jnp.ndarray,
    fwhm_l_padded: jnp.ndarray,
    profile_code: int,
) -> jnp.ndarray:
    """
    Micro-batched spectral convolution chunked across spectral lines using jax.lax.scan.

    Carries an accumulator frequency grid across line chunks [num_chunks, chunk_lines].

    Parameters:
        freq_grid: [N_freqs] full frequency grid.
        line_freqs_padded: [num_chunks, chunk_lines] line frequencies.
        line_intensities_padded: [num_chunks, chunk_lines] line intensities.
        fwhm_g_padded: [num_chunks, chunk_lines] Gaussian FWHMs.
        fwhm_l_padded: [num_chunks, chunk_lines] Lorentzian FWHMs.
        profile_code: Integer opcode for profile selection.

    Returns:
        jnp.ndarray: Accumulated convolved spectrum of shape [N_freqs].
    """
    chunk_data = (line_freqs_padded, line_intensities_padded, fwhm_g_padded, fwhm_l_padded)

    def scan_line_step(
        accumulated_spectrum: jnp.ndarray,
        chunk_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, None]:
        chunk_f, chunk_i, chunk_fg, chunk_fl = chunk_tuple
        # delta_nu: (N_freqs, chunk_lines)
        delta_nu = freq_grid[:, None] - chunk_f[None, :]
        profiles = evaluate_profile_kernel_jax(delta_nu, chunk_fg, chunk_fl, profile_code)
        chunk_contrib = jnp.dot(profiles, chunk_i)
        next_accum = accumulated_spectrum + chunk_contrib
        return next_accum, None

    init_spectrum = jnp.zeros_like(freq_grid, dtype=jnp.float64)
    final_spectrum, _ = jax.lax.scan(scan_line_step, init_spectrum, chunk_data)
    return final_spectrum


@jax.jit
def convolve_spectrum_monolithic_jax(
    freq_grid: jnp.ndarray,
    line_freqs: jnp.ndarray,
    line_intensities: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_l: jnp.ndarray,
    profile_code: int,
) -> jnp.ndarray:
    """
    Monolithic fully-vectorized convolution kernel in pure JAX FP64.

    Used for validation and benchmarking against micro-batched scan kernels.

    Parameters:
        freq_grid: [N_freqs] frequency grid.
        line_freqs: [N_lines] line centers.
        line_intensities: [N_lines] or [N_temps, N_lines] line intensities.
        fwhm_g: [N_lines] or scalar Gaussian FWHM.
        fwhm_l: [N_lines] or scalar Lorentzian FWHM.
        profile_code: Integer opcode for profile selection.

    Returns:
        jnp.ndarray: Monolithic convolved spectrum.
    """
    delta_nu = freq_grid[:, None] - line_freqs[None, :]
    profiles = evaluate_profile_kernel_jax(delta_nu, fwhm_g, fwhm_l, profile_code)

    if line_intensities.ndim == 1:
        return jnp.dot(profiles, line_intensities)
    else:
        # Multi-temperature (N_temps, N_lines)
        return jnp.matmul(line_intensities, profiles.T)


@jax.tree_util.register_pytree_node_class
class ConvolverPyTree:
    """
    Pure JAX PyTree encapsulating micro-batched convolved spectral results and line masks.
    Guarantees functional purity with zero I/O side effects and full compatibility with JAX transformations.
    """

    def __init__(
        self,
        frequencies: jnp.ndarray,
        intensities: jnp.ndarray,
        mask: jnp.ndarray,
        total_intensity: jnp.ndarray,
        num_lines: int,
        chunk_size: int,
        num_chunks: int,
        profile_type: str = "voigt",
    ) -> None:
        self.frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
        self.intensities = jnp.asarray(intensities, dtype=jnp.float64)
        self.mask = jnp.asarray(mask, dtype=jnp.bool_ if getattr(mask, "dtype", None) == jnp.bool_ else jnp.float64)
        self.total_intensity = jnp.asarray(total_intensity, dtype=jnp.float64)
        self.num_lines = int(num_lines)
        self.chunk_size = int(chunk_size)
        self.num_chunks = int(num_chunks)
        self.profile_type = str(profile_type)

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[Any, ...]]:
        children = (self.frequencies, self.intensities, self.mask, self.total_intensity)
        aux_data = (self.num_lines, self.chunk_size, self.num_chunks, self.profile_type)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jnp.ndarray, ...]) -> ConvolverPyTree:
        frequencies, intensities, mask, total_intensity = children
        num_lines, chunk_size, num_chunks, profile_type = aux_data
        return cls(
            frequencies=frequencies,
            intensities=intensities,
            mask=mask,
            total_intensity=total_intensity,
            num_lines=num_lines,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            profile_type=profile_type,
        )


@jax.jit
def _scan_step_masked_kernel(
    grid: jnp.ndarray,
    c_freqs: jnp.ndarray,
    c_intensities: jnp.ndarray,
    c_mask: jnp.ndarray,
    c_sigma: jnp.ndarray,
    c_gamma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Inner jax.lax.scan kernel executing static masked batching over frequency grid.

    Parameters:
        grid: Frequency domain grid array [N_freqs].
        c_freqs: [num_chunks, chunk_size] line centers.
        c_intensities: [num_chunks, chunk_size] line intensities.
        c_mask: [num_chunks, chunk_size] boolean/float validity mask.
        c_sigma: [num_chunks, chunk_size] Gaussian standard deviations.
        c_gamma: [num_chunks, chunk_size] Lorentzian HWHMs.

    Returns:
        jnp.ndarray: Accumulated convolved spectrum [N_freqs].
    """
    chunk_data = (c_freqs, c_intensities, c_mask, c_sigma, c_gamma)

    def scan_body(
        accumulated_spectrum: jnp.ndarray,
        chunk_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, None]:
        f_k, i_k, m_k, s_k, g_k = chunk_tuple
        # delta_nu: (N_grid, chunk_size)
        delta_nu = grid[:, None] - f_k[None, :]
        profiles = true_voigt_profile(delta_nu, s_k, g_k)
        effective_i = jnp.where(m_k, i_k, 0.0)
        chunk_contrib = jnp.dot(profiles, effective_i)
        next_accum = accumulated_spectrum + chunk_contrib
        return next_accum, None

    init_spectrum = jnp.zeros_like(grid, dtype=jnp.float64)
    final_spectrum, _ = jax.lax.scan(scan_body, init_spectrum, chunk_data)
    return final_spectrum


def static_masked_scan(
    lines: Union[SpectralLines, Tuple[Any, Any], jnp.ndarray, np.ndarray, Dict[str, Any]],
    chunk_size: int = 10000,
    freq_grid: Optional[Union[np.ndarray, jnp.ndarray, Sequence[float]]] = None,
    sigma: Union[float, jnp.ndarray, np.ndarray] = 0.01,
    gamma: Union[float, jnp.ndarray, np.ndarray] = 0.01,
    profile_type: Union[ProfileType, str] = ProfileType.VOIGT,
    cutoff_fwhm: float = 25.0,
    grid_resolution: Optional[float] = None,
    grid_points: Optional[int] = None,
    grid_min: Optional[float] = None,
    grid_max: Optional[float] = None,
) -> ConvolverPyTree:
    """
    Static masked batching convolution engine via jax.lax.scan returning a pure ConvolverPyTree.

    Pads spectral transition catalogs to fixed chunk multiples to ensure static XLA graph compilation,
    applying a boolean/float validity mask so padded trailing elements contribute exactly 0.0 to the spectrum.

    Parameters:
        lines: Transition line catalog (SpectralLines model, (frequencies, intensities) tuple, array, or dict).
        chunk_size: Static chunk size for micro-batching (default 10000).
        freq_grid: Optional explicit frequency grid array.
        sigma: Global or per-line Gaussian standard deviation (default 0.01).
        gamma: Global or per-line Lorentzian HWHM (default 0.01).
        profile_type: Profile shape function (default VOIGT).
        cutoff_fwhm: Boundary margin in units of FWHM for automatic grid construction.
        grid_resolution: Frequency grid spacing resolution.
        grid_points: Number of points for frequency domain.
        grid_min: Lower frequency bound.
        grid_max: Upper frequency bound.

    Returns:
        ConvolverPyTree: Pure JAX PyTree holding frequencies, intensities, mask, and metadata.
    """
    if chunk_size <= 0:
        raise MicroBatchError(f"chunk_size ({chunk_size}) must be strictly positive (> 0).")

    # Extract frequencies and intensities from various input formats
    if isinstance(lines, SpectralLines):
        freqs_list = lines.frequencies
        intensities_list = lines.intensities
        sigma_in = lines.fwhm_gaussian_overrides if lines.fwhm_gaussian_overrides is not None else sigma
        gamma_in = lines.fwhm_lorentzian_overrides if lines.fwhm_lorentzian_overrides is not None else gamma
    elif isinstance(lines, tuple) and len(lines) == 2:
        freqs_list, intensities_list = lines
        sigma_in = sigma
        gamma_in = gamma
    elif isinstance(lines, dict):
        freqs_list = lines["frequencies"]
        intensities_list = lines.get("intensities", [1.0] * len(freqs_list))
        sigma_in = lines.get("sigma", sigma)
        gamma_in = lines.get("gamma", gamma)
    else:
        freqs_list = lines
        intensities_list = jnp.ones_like(jnp.asarray(lines, dtype=jnp.float64))
        sigma_in = sigma
        gamma_in = gamma

    freqs_arr = jnp.asarray(freqs_list, dtype=jnp.float64)
    intensities_arr = jnp.asarray(intensities_list, dtype=jnp.float64)

    n_lines = freqs_arr.shape[0]
    if n_lines == 0:
        raise SpectralDataError("Spectral lines catalog cannot be empty for static_masked_scan.")

    # Validate finite positive frequencies
    if jnp.any(~jnp.isfinite(freqs_arr)) or jnp.any(freqs_arr <= 0.0):
        raise SpectralDataError("Transition frequencies must be finite and strictly positive (> 0).")

    sigma_arr = jnp.asarray(sigma_in, dtype=jnp.float64)
    gamma_arr = jnp.asarray(gamma_in, dtype=jnp.float64)

    # 1. Construct frequency grid if not supplied
    if freq_grid is not None:
        grid_jax = jnp.asarray(freq_grid, dtype=jnp.float64)
        if grid_jax.ndim != 1 or grid_jax.shape[0] < 2:
            raise InvalidGridError("User-supplied freq_grid must be 1D with at least 2 points.")
    else:
        min_f = float(jnp.min(freqs_arr))
        max_f = float(jnp.max(freqs_arr))
        eff_fwhm = float(max(float(jnp.max(sigma_arr)) * 2.354820045, float(jnp.max(gamma_arr)) * 2.0))
        margin = cutoff_fwhm * eff_fwhm
        g_min = grid_min if grid_min is not None else (min_f - margin)
        g_max = grid_max if grid_max is not None else (max_f + margin)
        if g_min >= g_max:
            raise InvalidGridError(f"Grid minimum ({g_min}) must be strictly less than grid maximum ({g_max}).")
        if grid_points is not None:
            n_pts = grid_points
        elif grid_resolution is not None:
            n_pts = max(2, int(math.ceil((g_max - g_min) / grid_resolution)) + 1)
        else:
            n_pts = max(100, int(math.ceil((g_max - g_min) / (eff_fwhm / 10.0))) + 1)
        grid_jax = jnp.linspace(g_min, g_max, n_pts, dtype=jnp.float64)

    # 2. Static padding & chunking
    num_chunks = int(math.ceil(n_lines / chunk_size))
    padded_len = num_chunks * chunk_size
    pad_len = padded_len - n_lines

    padded_freqs = jnp.pad(freqs_arr, (0, pad_len), constant_values=freqs_arr[-1])
    padded_intensities = jnp.pad(intensities_arr, (0, pad_len), constant_values=0.0)

    if sigma_arr.ndim == 1 and sigma_arr.shape[0] == n_lines:
        padded_sigma = jnp.pad(sigma_arr, (0, pad_len), constant_values=sigma_arr[-1])
    else:
        padded_sigma = jnp.full((padded_len,), sigma_arr, dtype=jnp.float64)

    if gamma_arr.ndim == 1 and gamma_arr.shape[0] == n_lines:
        padded_gamma = jnp.pad(gamma_arr, (0, pad_len), constant_values=gamma_arr[-1])
    else:
        padded_gamma = jnp.full((padded_len,), gamma_arr, dtype=jnp.float64)

    mask = jnp.arange(padded_len) < n_lines

    chunked_freqs = padded_freqs.reshape((num_chunks, chunk_size))
    chunked_intensities = padded_intensities.reshape((num_chunks, chunk_size))
    chunked_sigma = padded_sigma.reshape((num_chunks, chunk_size))
    chunked_gamma = padded_gamma.reshape((num_chunks, chunk_size))
    chunked_mask = mask.reshape((num_chunks, chunk_size))

    # 3. Execute static masked scan
    final_spectrum = _scan_step_masked_kernel(
        grid=grid_jax,
        c_freqs=chunked_freqs,
        c_intensities=chunked_intensities,
        c_mask=chunked_mask,
        c_sigma=chunked_sigma,
        c_gamma=chunked_gamma,
    )

    total_intensity = jnp.trapezoid(final_spectrum, grid_jax)

    p_str = str(profile_type).lower().replace("profiletype.", "")

    return ConvolverPyTree(
        frequencies=grid_jax,
        intensities=final_spectrum,
        mask=mask,
        total_intensity=total_intensity,
        num_lines=n_lines,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        profile_type=p_str,
    )


# =============================================================================
# 7. HIGH-LEVEL BROADENER API & FREQUENCY GRID UTILITIES
# =============================================================================

def build_frequency_grid(
    spectral_lines: SpectralLines,
    config: ConvolutionConfig,
) -> np.ndarray:
    """
    Construct a dense, uniform frequency grid spanning the spectral domain.

    Parameters:
        spectral_lines: SpectralLines catalog.
        config: ConvolutionConfig options.

    Returns:
        np.ndarray: 1D strictly monotonically increasing frequency grid.
    """
    freqs = spectral_lines.frequencies
    min_line_freq = min(freqs)
    max_line_freq = max(freqs)

    # Determine maximum effective line width
    max_fwhm = max(config.fwhm_gaussian, config.fwhm_lorentzian)
    if spectral_lines.fwhm_gaussian_overrides is not None:
        max_fwhm = max(max_fwhm, max(spectral_lines.fwhm_gaussian_overrides))
    if spectral_lines.fwhm_lorentzian_overrides is not None:
        max_fwhm = max(max_fwhm, max(spectral_lines.fwhm_lorentzian_overrides))

    margin = config.cutoff_fwhm * max_fwhm

    # Compute domain bounds
    grid_min = config.grid_min if config.grid_min is not None else (min_line_freq - margin)
    grid_max = config.grid_max if config.grid_max is not None else (max_line_freq + margin)

    if grid_min >= grid_max:
        raise InvalidGridError(f"Grid minimum ({grid_min}) must be strictly less than grid maximum ({grid_max}).")

    # Determine grid points
    if config.grid_points is not None:
        num_points = config.grid_points
    elif config.grid_resolution is not None:
        num_points = max(2, int(math.ceil((grid_max - grid_min) / config.grid_resolution)) + 1)
    else:
        # Default: at least 10 evaluation points per FWHM
        default_res = max_fwhm / 10.0
        num_points = max(100, int(math.ceil((grid_max - grid_min) / default_res)) + 1)

    grid = np.linspace(grid_min, grid_max, num_points, dtype=np.float64)
    return grid


def get_profile_type_code(profile_type: Union[ProfileType, str]) -> int:
    """Map ProfileType enum to integer opcode for JAX kernels."""
    p_str = str(profile_type).lower().replace("profiletype.", "")
    if p_str == "gaussian":
        return 0
    elif p_str == "lorentzian":
        return 1
    elif p_str == "pseudo_voigt":
        return 2
    elif p_str == "voigt":
        return 3
    else:
        raise ProfileError(f"Unsupported profile type: '{profile_type}'")


def evaluate_partition_grid(
    spectral_lines: SpectralLines,
    partition_config: PartitionGridConfig,
) -> PartitionGridResult:
    """
    Evaluate multi-temperature partition functions and Boltzmann thermal scaling.

    Parameters:
        spectral_lines: SpectralLines catalog.
        partition_config: PartitionGridConfig specifications.

    Returns:
        PartitionGridResult: Multi-temperature scaling matrices.
    """
    if spectral_lines.lower_state_energies is None:
        raise ThermalScalingError(
            "SpectralLines must provide lower_state_energies to compute multi-temperature Boltzmann scaling."
        )

    temps_jax = jnp.array(partition_config.temperatures, dtype=jnp.float64)
    freqs_jax = jnp.array(spectral_lines.frequencies, dtype=jnp.float64)
    energies_jax = jnp.array(spectral_lines.lower_state_energies, dtype=jnp.float64)
    intensities_jax = jnp.array(spectral_lines.intensities, dtype=jnp.float64)

    energy_to_k = get_energy_conversion_factor_to_kelvin(partition_config.energy_unit)
    freq_to_k = get_energy_conversion_factor_to_kelvin(EnergyUnit.MHZ)  # Default line frequency unit is MHz

    # Compute Q(T)
    if partition_config.partition_values is not None:
        q_temps_jax = jnp.array(partition_config.partition_values, dtype=jnp.float64)
    else:
        e_levels_jax = (
            jnp.array(partition_config.energy_levels, dtype=jnp.float64)
            if partition_config.energy_levels is not None
            else None
        )
        degen_jax = (
            jnp.array(partition_config.degeneracies, dtype=jnp.float64)
            if partition_config.degeneracies is not None
            else None
        )
        q_temps_jax = evaluate_partition_function_jax(
            temperatures=temps_jax,
            power_law_index=partition_config.power_law_index,
            q_ref=partition_config.q_ref,
            t_ref=partition_config.t_ref,
            energy_levels=e_levels_jax,
            degeneracies=degen_jax,
            energy_to_k=energy_to_k,
        )

    # Compute scaling grid
    scaled_i, boltz_f, stim_f = compute_boltzmann_scaling_grid_jax(
        frequencies=freqs_jax,
        lower_energies=energies_jax,
        reference_intensities=intensities_jax,
        temperatures=temps_jax,
        q_temps=q_temps_jax,
        t_ref=partition_config.t_ref,
        q_ref=partition_config.q_ref,
        freq_to_k=freq_to_k,
        energy_to_k=energy_to_k,
    )

    return PartitionGridResult(
        temperatures=[float(t) for t in partition_config.temperatures],
        partition_function_values=[float(q) for q in np.array(q_temps_jax)],
        scaled_intensities=np.array(scaled_i, dtype=np.float64).tolist(),
        boltzmann_factors=np.array(boltz_f, dtype=np.float64).tolist(),
        stimulated_emission_factors=np.array(stim_f, dtype=np.float64).tolist(),
    )


def convolve_spectrum(
    spectral_lines: SpectralLines,
    config: Optional[ConvolutionConfig] = None,
    freq_grid: Optional[Union[np.ndarray, jnp.ndarray, Sequence[float]]] = None,
    partition_config: Optional[PartitionGridConfig] = None,
) -> ConvolvedSpectrumResult:
    """
    Perform macroscopic spectral line broadening and convolution in pure JAX FP64.

    Processes dense frequency grids with memory-aware micro-batching via jax.lax.scan.

    Parameters:
        spectral_lines: SpectralLines catalog.
        config: ConvolutionConfig options (defaults applied if None).
        freq_grid: Optional user-supplied frequency grid array.
        partition_config: Optional multi-temperature partition grid configuration.

    Returns:
        ConvolvedSpectrumResult: Fully populated convolved spectrum results.
    """
    start_time = time.perf_counter()
    conv_config = config if config is not None else ConvolutionConfig()

    # 1. Validate & build frequency grid
    if freq_grid is not None:
        grid_arr = np.asarray(freq_grid, dtype=np.float64)
        if grid_arr.ndim != 1 or len(grid_arr) < 2:
            raise InvalidGridError("User-supplied freq_grid must be a 1D array with at least 2 points.")
        if np.any(np.diff(grid_arr) <= 0.0):
            raise InvalidGridError("User-supplied freq_grid must be strictly monotonically increasing.")
        if not np.all(np.isfinite(grid_arr)):
            raise InvalidGridError("User-supplied freq_grid contains non-finite (NaN or Inf) values.")
    else:
        grid_arr = build_frequency_grid(spectral_lines, conv_config)

    n_freqs = len(grid_arr)
    n_lines = len(spectral_lines.frequencies)
    profile_code = get_profile_type_code(conv_config.profile_type)

    # 2. Extract line parameters as JAX arrays
    line_freqs_jax = jnp.array(spectral_lines.frequencies, dtype=jnp.float64)

    if spectral_lines.fwhm_gaussian_overrides is not None:
        fwhm_g_jax = jnp.array(spectral_lines.fwhm_gaussian_overrides, dtype=jnp.float64)
    else:
        fwhm_g_jax = jnp.full((n_lines,), conv_config.fwhm_gaussian, dtype=jnp.float64)

    if spectral_lines.fwhm_lorentzian_overrides is not None:
        fwhm_l_jax = jnp.array(spectral_lines.fwhm_lorentzian_overrides, dtype=jnp.float64)
    else:
        fwhm_l_jax = jnp.full((n_lines,), conv_config.fwhm_lorentzian, dtype=jnp.float64)

    # 3. Handle multi-temperature scaling if partition_config is present
    multi_temp_intensities_list: Optional[List[List[float]]] = None
    temps_list: Optional[List[float]] = None
    is_multi_temp = partition_config is not None

    if is_multi_temp:
        assert partition_config is not None
        partition_result = evaluate_partition_grid(spectral_lines, partition_config)
        scaled_intensities_jax = jnp.array(partition_result.scaled_intensities, dtype=jnp.float64)
        n_temps = len(partition_config.temperatures)
        temps_list = [float(t) for t in partition_config.temperatures]
    else:
        n_temps = 1
        scaled_intensities_jax = jnp.array(spectral_lines.intensities, dtype=jnp.float64)

    # 4. Determine micro-batch chunk size and setup jax.lax.scan
    chunk_size = calculate_optimal_chunk_size(
        n_freqs=n_freqs,
        n_lines=n_lines,
        n_temps=n_temps,
        max_vram_mb=conv_config.max_vram_mb,
        explicit_chunk_size=conv_config.chunk_size,
    )

    num_chunks = int(math.ceil(n_freqs / chunk_size))
    padded_len = num_chunks * chunk_size
    pad_amount = padded_len - n_freqs

    # Pad frequency grid to exact multiple of chunk_size
    if pad_amount > 0:
        grid_padded = np.pad(grid_arr, (0, pad_amount), mode="edge")
    else:
        grid_padded = grid_arr

    grid_padded_jax = jnp.array(grid_padded, dtype=jnp.float64).reshape((num_chunks, chunk_size))

    # 5. Execute micro-batched scan kernel
    if not is_multi_temp:
        chunked_output = convolve_frequency_chunks_scan_1d_jax(
            freq_grid_padded=grid_padded_jax,
            line_freqs=line_freqs_jax,
            line_intensities=scaled_intensities_jax,
            fwhm_g=fwhm_g_jax,
            fwhm_l=fwhm_l_jax,
            profile_code=profile_code,
        )
        spectrum_1d_jax = chunked_output.reshape(-1)[:n_freqs]
        spectrum_1d = np.array(spectrum_1d_jax, dtype=np.float64)
    else:
        chunked_output_2d = convolve_frequency_chunks_scan_2d_jax(
            freq_grid_padded=grid_padded_jax,
            line_freqs=line_freqs_jax,
            scaled_intensities=scaled_intensities_jax,
            fwhm_g=fwhm_g_jax,
            fwhm_l=fwhm_l_jax,
            profile_code=profile_code,
        )
        # chunked_output_2d shape: (num_chunks, N_temps, chunk_size)
        # Swap axes to (N_temps, num_chunks, chunk_size) and reshape to (N_temps, padded_len)
        reshaped_2d = jnp.swapaxes(chunked_output_2d, 0, 1).reshape((n_temps, padded_len))[:, :n_freqs]
        spectrum_2d = np.array(reshaped_2d, dtype=np.float64)
        multi_temp_intensities_list = spectrum_2d.tolist()
        # Primary 1D spectrum is at first temperature
        spectrum_1d = spectrum_2d[0]

    # 6. Physical statistics and integration
    # Trapezoidal integration for total integrated intensity
    total_area = float(np.trapezoid(spectrum_1d, grid_arr))
    peak_idx = int(np.argmax(spectrum_1d))
    peak_freq = float(grid_arr[peak_idx])
    peak_val = float(spectrum_1d[peak_idx])

    execution_time = time.perf_counter() - start_time

    return ConvolvedSpectrumResult(
        frequencies=grid_arr.tolist(),
        intensities=spectrum_1d.tolist(),
        temperatures=temps_list,
        multi_temp_intensities=multi_temp_intensities_list,
        profile_type=conv_config.profile_type,
        total_integrated_intensity=total_area,
        peak_frequency=peak_freq,
        peak_intensity=peak_val,
        chunk_size_used=chunk_size,
        num_chunks=num_chunks,
        execution_time_seconds=execution_time,
    )


class MacroscopicSpectralBroadener:
    """
    High-Performance Spectroscopic Broadener & Convolver Engine.

    Provides end-to-end management of spectral line profiles, multi-temperature Boltzmann grids,
    and memory-aware micro-batched JAX convolution.
    """

    def __init__(self, config: Optional[ConvolutionConfig] = None) -> None:
        self.config = config if config is not None else ConvolutionConfig()

    def broaden(
        self,
        spectral_lines: SpectralLines,
        freq_grid: Optional[Union[np.ndarray, jnp.ndarray, Sequence[float]]] = None,
        partition_config: Optional[PartitionGridConfig] = None,
    ) -> ConvolvedSpectrumResult:
        """Execute macroscopic spectral broadening over spectral lines catalog."""
        return convolve_spectrum(
            spectral_lines=spectral_lines,
            config=self.config,
            freq_grid=freq_grid,
            partition_config=partition_config,
        )

    def evaluate_partition_grid(
        self,
        spectral_lines: SpectralLines,
        partition_config: PartitionGridConfig,
    ) -> PartitionGridResult:
        """Compute multi-temperature partition functions and Boltzmann thermal scaling factors."""
        return evaluate_partition_grid(
            spectral_lines=spectral_lines,
            partition_config=partition_config,
        )

    def build_grid(self, spectral_lines: SpectralLines) -> np.ndarray:
        """Construct the dense frequency domain grid."""
        return build_frequency_grid(spectral_lines, self.config)


__all__ = [
    # Physical Constants
    "PLANCK_H",
    "BOLTZMANN_K",
    "SPEED_OF_LIGHT_CM_S",
    "SPEED_OF_LIGHT_M_S",
    "H_OVER_K_B",
    "C2_RADIATION_CONST",
    "SQRT_2_LN_2",
    "FWHM_TO_SIGMA_GAUSS",
    "FOUR_LN_2",
    "GAUSS_PREFACTOR",
    "LORENTZ_PREFACTOR",
    "DEFAULT_MAX_VRAM_MB",
    # Exceptions
    "ConvolverError",
    "ProfileError",
    "ThermalScalingError",
    "MicroBatchError",
    "InvalidGridError",
    "SpectralDataError",
    # Enums & Models & PyTrees
    "ProfileType",
    "EnergyUnit",
    "SpectralLines",
    "ConvolutionConfig",
    "PartitionGridConfig",
    "PartitionGridResult",
    "ConvolvedSpectrumResult",
    "ConvolverPyTree",
    # JAX Kernels
    "faddeeva_rational_jax",
    "gaussian_kernel_jax",
    "lorentzian_kernel_jax",
    "pseudo_voigt_kernel_jax",
    "faddeeva_voigt_kernel_jax",
    "true_voigt_profile",
    "evaluate_profile_kernel_jax",
    "evaluate_partition_function_jax",
    "compute_boltzmann_scaling_grid_jax",
    "convolute_multi_temperature",
    "convolve_frequency_chunks_scan_1d_jax",
    "convolve_frequency_chunks_scan_2d_jax",
    "convolve_line_chunks_scan_1d_jax",
    "convolve_spectrum_monolithic_jax",
    "_scan_step_masked_kernel",
    "static_masked_scan",
    # High-level Utilities & Engine
    "get_energy_conversion_factor_to_kelvin",
    "get_profile_type_code",
    "calculate_optimal_chunk_size",
    "build_frequency_grid",
    "evaluate_partition_grid",
    "convolve_spectrum",
    "MacroscopicSpectralBroadener",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_convolver.py ---
"""
CoChem-SpycFit: Comprehensive Physical Unit Test Suite for Macroscopic Spectral Broadener.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of Gaussian, Lorentzian, Pseudo-Voigt, and Faddeeva-Voigt profiles
- Validation of multi-temperature partition functions Q(T) and Boltzmann thermal scaling
- Numerical verification of micro-batched jax.lax.scan vs monolithic vectorized output (diff < 1e-12)
- AST structural verification enforcing pure jax.lax.scan without forbidden Python for/while loops
- Negative physical tests for invalid inputs, non-physical temperatures, and dimension mismatches.
"""

from __future__ import annotations

import ast
import inspect
import math
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_convolver import (
    C2_RADIATION_CONST,
    DEFAULT_MAX_VRAM_MB,
    FOUR_LN_2,
    FWHM_TO_SIGMA_GAUSS,
    GAUSS_PREFACTOR,
    H_OVER_K_B,
    LORENTZ_PREFACTOR,
    PLANCK_H,
    SPEED_OF_LIGHT_CM_S,
    SPEED_OF_LIGHT_M_S,
    SQRT_2_LN_2,
    ConvolutionConfig,
    ConvolvedSpectrumResult,
    ConvolverError,
    EnergyUnit,
    InvalidGridError,
    MacroscopicSpectralBroadener,
    MicroBatchError,
    PartitionGridConfig,
    PartitionGridResult,
    ProfileError,
    ProfileType,
    SpectralDataError,
    SpectralLines,
    ThermalScalingError,
    build_frequency_grid,
    calculate_optimal_chunk_size,
    compute_boltzmann_scaling_grid_jax,
    convolve_frequency_chunks_scan_1d_jax,
    convolve_frequency_chunks_scan_2d_jax,
    convolve_line_chunks_scan_1d_jax,
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    evaluate_partition_grid,
    evaluate_profile_kernel_jax,
    faddeeva_rational_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    get_energy_conversion_factor_to_kelvin,
    get_profile_type_code,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
)

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. ANALYTICAL LINE PROFILE KERNEL PHYSICAL TESTS (FP64)
# =============================================================================

def test_gaussian_profile_peak_and_fwhm_geometry() -> None:
    """Validate Gaussian profile peak location, peak height, and half-maximum width."""
    fwhm = 0.4
    delta_nu = jnp.array([0.0, -0.2, 0.2, -0.4, 0.4], dtype=jnp.float64)
    g_vals = gaussian_kernel_jax(delta_nu, fwhm)

    # 1. Peak height at delta_nu = 0 matches analytical 2 * sqrt(ln 2) / (sqrt(pi) * FWHM)
    expected_peak = (2.0 * math.sqrt(math.log(2.0))) / (math.sqrt(math.pi) * fwhm)
    actual_peak = float(g_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-14

    # 2. At delta_nu = +/- FWHM / 2, profile value must be exactly 50% of the peak
    half_left = float(g_vals[1])
    half_right = float(g_vals[2])
    assert abs(half_left - 0.5 * expected_peak) < 1e-14
    assert abs(half_right - 0.5 * expected_peak) < 1e-14

    # 3. Symmetry check: g(-delta) == g(+delta)
    assert abs(float(g_vals[1]) - float(g_vals[2])) < 1e-15
    assert abs(float(g_vals[3]) - float(g_vals[4])) < 1e-15


def test_gaussian_profile_area_normalization() -> None:
    """Validate Gaussian profile area normalization over dense frequency grid."""
    fwhm = 0.25
    nu_grid = np.linspace(-10.0, 10.0, 200001, dtype=np.float64)
    g_vals = np.array(gaussian_kernel_jax(jnp.array(nu_grid), fwhm))

    integrated_area = float(np.trapezoid(g_vals, nu_grid))
    assert abs(integrated_area - 1.0) < 1e-11


def test_lorentzian_profile_peak_and_fwhm_geometry() -> None:
    """Validate Lorentzian profile peak location, peak height, and half-maximum width."""
    fwhm = 0.6
    delta_nu = jnp.array([0.0, -0.3, 0.3, -0.6, 0.6], dtype=jnp.float64)
    l_vals = lorentzian_kernel_jax(delta_nu, fwhm)

    # 1. Peak height at delta_nu = 0 matches analytical 2 / (pi * FWHM)
    expected_peak = 2.0 / (math.pi * fwhm)
    actual_peak = float(l_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-14

    # 2. At delta_nu = +/- FWHM / 2, profile value must be exactly 50% of the peak
    half_left = float(l_vals[1])
    half_right = float(l_vals[2])
    assert abs(half_left - 0.5 * expected_peak) < 1e-14
    assert abs(half_right - 0.5 * expected_peak) < 1e-14


def test_lorentzian_profile_analytical_integral_conservation() -> None:
    """Validate Lorentzian profile integral against analytical arctan Cauchy integral."""
    fwhm = 0.5
    half_span = 50.0
    nu_grid = np.linspace(-half_span, half_span, 500001, dtype=np.float64)
    l_vals = np.array(lorentzian_kernel_jax(jnp.array(nu_grid), fwhm))

    # Analytical integral of L(x; fwhm) from -L to +L is (2/pi) * arctan(2 * L / fwhm)
    expected_integral = (2.0 / math.pi) * math.atan((2.0 * half_span) / fwhm)
    numerical_integral = float(np.trapezoid(l_vals, nu_grid))

    assert abs(numerical_integral - expected_integral) < 1e-7


def test_pseudo_voigt_asymptotic_limits() -> None:
    """Validate Thompson-Cox-Hastings Pseudo-Voigt asymptotic convergence to pure Gaussian and Lorentzian."""
    nu_grid = jnp.linspace(-2.0, 2.0, 501, dtype=jnp.float64)
    fwhm_ref = 0.3

    # Limit 1: FWHM_L -> 0 (pure Gaussian)
    pv_gauss_limit = pseudo_voigt_kernel_jax(nu_grid, fwhm_ref, 1e-12)
    pure_gauss = gaussian_kernel_jax(nu_grid, fwhm_ref)
    diff_gauss = float(jnp.max(jnp.abs(pv_gauss_limit - pure_gauss)))
    assert diff_gauss < 1e-10

    # Limit 2: FWHM_G -> 0 (pure Lorentzian)
    pv_lorentz_limit = pseudo_voigt_kernel_jax(nu_grid, 1e-12, fwhm_ref)
    pure_lorentz = lorentzian_kernel_jax(nu_grid, fwhm_ref)
    diff_lorentz = float(jnp.max(jnp.abs(pv_lorentz_limit - pure_lorentz)))
    assert diff_lorentz < 1e-10


def test_faddeeva_rational_accuracy_vs_reference() -> None:
    """Validate pure-JAX Weideman Faddeeva evaluation against analytical identities."""
    # Test property: w(0) = 1.0 + 0.0j
    w_zero = faddeeva_rational_jax(jnp.array([0.0 + 0.0j], dtype=jnp.complex128))
    assert abs(complex(w_zero[0]) - 1.0) < 1e-13

    # Test property: w(i*y) = exp(y^2) * erfc(y) is strictly real for real y >= 0
    y_test = 1.5
    w_iy = faddeeva_rational_jax(jnp.array([0.0 + y_test * 1j], dtype=jnp.complex128))
    val = complex(w_iy[0])
    assert abs(val.imag) < 1e-14
    assert val.real > 0.0

    # Test exact Voigt profile peak and symmetry
    fg = 0.4
    fl = 0.2
    nu_test = jnp.array([0.0, -0.5, 0.5], dtype=jnp.float64)
    v_vals = faddeeva_voigt_kernel_jax(nu_test, fg, fl)

    assert float(v_vals[0]) > float(v_vals[1])
    assert abs(float(v_vals[1]) - float(v_vals[2])) < 1e-14


def test_unified_profile_kernel_dispatch() -> None:
    """Validate unified evaluate_profile_kernel_jax dispatch across all 4 profile codes."""
    delta = jnp.array([0.0], dtype=jnp.float64)
    fg = 0.2
    fl = 0.2

    for code in range(4):
        val = evaluate_profile_kernel_jax(delta, fg, fl, code)
        assert float(val[0]) > 0.0
        assert math.isfinite(float(val[0]))


# =============================================================================
# 2. MULTI-TEMPERATURE PARTITION GRIDS & BOLTZMANN SCALING TESTS
# =============================================================================

def test_partition_function_direct_state_sum() -> None:
    """Validate direct state-sum partition function calculation Q(T) = sum_k g_k * exp(-c2 * E_k / T)."""
    # 3-level quantum system: E = [0.0, 100.0, 300.0] cm^-1, g = [1.0, 3.0, 5.0]
    e_levels = jnp.array([0.0, 100.0, 300.0], dtype=jnp.float64)
    degeneracies = jnp.array([1.0, 3.0, 5.0], dtype=jnp.float64)
    temps = jnp.array([50.0, 150.0, 300.0], dtype=jnp.float64)
    c2 = C2_RADIATION_CONST

    q_jax = evaluate_partition_function_jax(
        temperatures=temps,
        power_law_index=None,
        q_ref=1.0,
        t_ref=298.15,
        energy_levels=e_levels,
        degeneracies=degeneracies,
        energy_to_k=c2,
    )

    # Analytical verification
    for idx, t in enumerate([50.0, 150.0, 300.0]):
        q_expected = (
            1.0 * math.exp(-c2 * 0.0 / t)
            + 3.0 * math.exp(-c2 * 100.0 / t)
            + 5.0 * math.exp(-c2 * 300.0 / t)
        )
        assert abs(float(q_jax[idx]) - q_expected) < 1e-12


def test_partition_function_power_law_scaling() -> None:
    """Validate power law temperature scaling Q(T) = Q(T_ref) * (T / T_ref)^n."""
    temps = jnp.array([100.0, 200.0, 298.15, 400.0], dtype=jnp.float64)
    t_ref = 298.15
    q_ref = 500.0
    n = 1.5  # Non-linear polyatomic asymmetric top exponent

    q_jax = evaluate_partition_function_jax(
        temperatures=temps,
        power_law_index=n,
        q_ref=q_ref,
        t_ref=t_ref,
        energy_levels=None,
        degeneracies=None,
        energy_to_k=C2_RADIATION_CONST,
    )

    # At T = T_ref, Q(T) must be exactly Q(T_ref)
    assert abs(float(q_jax[2]) - q_ref) < 1e-14
    # At T = 100 K, Q(T) = 500 * (100 / 296)^1.5
    expected_100 = q_ref * (100.0 / 298.15) ** 1.5
    assert abs(float(q_jax[0]) - expected_100) < 1e-12


def test_boltzmann_scaling_invariance_at_reference_temperature() -> None:
    """Validate that scaled intensities, Boltzmann factors, and stimulated emissions equal 1.0 at T = T_ref."""
    freqs = jnp.array([100000.0, 250000.0, 500000.0], dtype=jnp.float64)  # MHz
    energies = jnp.array([10.0, 80.0, 250.0], dtype=jnp.float64)  # cm^-1
    intensities = jnp.array([1.5, 0.8, 0.3], dtype=jnp.float64)

    t_ref = 298.15
    q_ref = 200.0
    temps = jnp.array([t_ref], dtype=jnp.float64)
    q_temps = jnp.array([q_ref], dtype=jnp.float64)

    freq_to_k = H_OVER_K_B * 1.0e6
    energy_to_k = C2_RADIATION_CONST

    scaled_i, boltz_f, stim_f = compute_boltzmann_scaling_grid_jax(
        frequencies=freqs,
        lower_energies=energies,
        reference_intensities=intensities,
        temperatures=temps,
        q_temps=q_temps,
        t_ref=t_ref,
        q_ref=q_ref,
        freq_to_k=freq_to_k,
        energy_to_k=energy_to_k,
    )

    # All factors at T_ref must be identically 1.0
    assert float(jnp.max(jnp.abs(boltz_f - 1.0))) < 1e-15
    assert float(jnp.max(jnp.abs(stim_f - 1.0))) < 1e-15
    assert float(jnp.max(jnp.abs(scaled_i[0] - intensities))) < 1e-15


def test_boltzmann_excited_state_thermal_activation() -> None:
    """Validate physical growth of high-energy transition line intensities with temperature."""
    # Line 1: ground state E_l = 0 cm^-1
    # Line 2: excited state E_l = 500 cm^-1
    freqs = jnp.array([100000.0, 100000.0], dtype=jnp.float64)
    energies = jnp.array([0.0, 500.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 1.0], dtype=jnp.float64)

    temps = jnp.array([50.0, 300.0], dtype=jnp.float64)
    t_ref = 300.0
    q_ref = 100.0
    q_temps = jnp.array([10.0, 100.0], dtype=jnp.float64)

    scaled_i, _, _ = compute_boltzmann_scaling_grid_jax(
        frequencies=freqs,
        lower_energies=energies,
        reference_intensities=intensities,
        temperatures=temps,
        q_temps=q_temps,
        t_ref=t_ref,
        q_ref=q_ref,
        freq_to_k=H_OVER_K_B * 1.0e6,
        energy_to_k=C2_RADIATION_CONST,
    )

    # At 50 K, excited state line (index 1) must be exponentially suppressed compared to 300 K
    i_excited_50k = float(scaled_i[0, 1])
    i_excited_300k = float(scaled_i[1, 1])
    assert i_excited_50k < 1e-3 * i_excited_300k


# =============================================================================
# 3. MICRO-BATCHED JAX.LAX.SCAN VS MONOLITHIC KERNEL TESTS
# =============================================================================

def test_frequency_chunked_scan_matches_monolithic_1d() -> None:
    """Verify that 1D micro-batched jax.lax.scan matches monolithic vectorized convolution (diff < 1e-12)."""
    n_freqs = 2048
    n_lines = 64
    freq_grid = jnp.linspace(80.0, 120.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(85.0, 115.0, n_lines, dtype=jnp.float64)
    line_intensities = jnp.linspace(0.1, 2.0, n_lines, dtype=jnp.float64)
    fwhm_g = jnp.full((n_lines,), 0.2, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.1, dtype=jnp.float64)

    # Test all profile types
    for profile_code in [0, 1, 2, 3]:
        # 1. Monolithic evaluation
        spec_mono = convolve_spectrum_monolithic_jax(
            freq_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
        )

        # 2. Micro-batched scan with small chunk size (e.g. 128 points = 16 chunks)
        chunk_size = 128
        num_chunks = (n_freqs + chunk_size - 1) // chunk_size
        padded_len = num_chunks * chunk_size
        padded_grid = jnp.pad(freq_grid, (0, padded_len - n_freqs), mode="edge")
        reshaped_grid = padded_grid.reshape((num_chunks, chunk_size))

        chunked_spec = convolve_frequency_chunks_scan_1d_jax(
            reshaped_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
        )
        spec_scan = chunked_spec.reshape(-1)[:n_freqs]

        diff = float(jnp.max(jnp.abs(spec_mono - spec_scan)))
        assert diff < 1e-12, f"Profile code {code} scan deviation {diff} exceeds 1e-12."


def test_frequency_chunked_scan_matches_monolithic_2d_multi_temp() -> None:
    """Verify that multi-temperature 2D jax.lax.scan matches monolithic evaluation (diff < 1e-12)."""
    n_freqs = 1500
    n_lines = 32
    n_temps = 4
    freq_grid = jnp.linspace(100.0, 200.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(110.0, 190.0, n_lines, dtype=jnp.float64)
    base_int = jnp.linspace(0.1, 1.0, n_lines, dtype=jnp.float64)
    temps = jnp.linspace(50.0, 300.0, n_temps, dtype=jnp.float64)
    multi_intensities = base_int[None, :] * (300.0 / temps[:, None])
    fwhm_g = jnp.full((n_lines,), 0.3, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.2, dtype=jnp.float64)
    profile_code = 2  # Pseudo-Voigt

    # Monolithic
    spec_mono = convolve_spectrum_monolithic_jax(
        freq_grid, line_freqs, multi_intensities, fwhm_g, fwhm_l, profile_code
    )

    # Micro-batched 2D scan
    chunk_size = 256
    num_chunks = (n_freqs + chunk_size - 1) // chunk_size
    padded_len = num_chunks * chunk_size
    padded_grid = jnp.pad(freq_grid, (0, padded_len - n_freqs), mode="edge")
    reshaped_grid = padded_grid.reshape((num_chunks, chunk_size))

    chunked_spec_2d = convolve_frequency_chunks_scan_2d_jax(
        reshaped_grid, line_freqs, multi_intensities, fwhm_g, fwhm_l, profile_code
    )
    spec_scan_2d = jnp.swapaxes(chunked_spec_2d, 0, 1).reshape((n_temps, padded_len))[:, :n_freqs]

    diff = float(jnp.max(jnp.abs(spec_mono - spec_scan_2d)))
    assert diff < 1e-12


def test_line_chunked_scan_accumulation_matches_monolithic() -> None:
    """Verify that spectral line chunking via jax.lax.scan accumulates identical spectrum to monolithic."""
    n_freqs = 500
    n_lines = 100
    chunk_lines = 25
    num_chunks = 4

    freq_grid = jnp.linspace(50.0, 150.0, n_freqs, dtype=jnp.float64)
    line_freqs = jnp.linspace(60.0, 140.0, n_lines, dtype=jnp.float64)
    line_intensities = jnp.linspace(0.5, 3.0, n_lines, dtype=jnp.float64)
    fwhm_g = jnp.full((n_lines,), 0.15, dtype=jnp.float64)
    fwhm_l = jnp.full((n_lines,), 0.10, dtype=jnp.float64)
    profile_code = 0  # Gaussian

    # Monolithic
    spec_mono = convolve_spectrum_monolithic_jax(
        freq_grid, line_freqs, line_intensities, fwhm_g, fwhm_l, profile_code
    )

    # Line chunked
    f_reshaped = line_freqs.reshape((num_chunks, chunk_lines))
    i_reshaped = line_intensities.reshape((num_chunks, chunk_lines))
    fg_reshaped = fwhm_g.reshape((num_chunks, chunk_lines))
    fl_reshaped = fwhm_l.reshape((num_chunks, chunk_lines))

    spec_line_scan = convolve_line_chunks_scan_1d_jax(
        freq_grid, f_reshaped, i_reshaped, fg_reshaped, fl_reshaped, profile_code
    )

    diff = float(jnp.max(jnp.abs(spec_mono - spec_line_scan)))
    assert diff < 1e-12


def test_optimal_chunk_size_calculation() -> None:
    """Validate dynamic micro-batch chunk calculation under memory cap."""
    n_freqs = 100000
    n_lines = 500
    max_mb = 10.0  # 10 MB budget

    chunk = calculate_optimal_chunk_size(
        n_freqs=n_freqs,
        n_lines=n_lines,
        n_temps=1,
        max_vram_mb=max_mb,
    )

    assert 1 <= chunk <= n_freqs
    # Memory occupied by chunk must be <= max_mb
    bytes_per_pt = (4.0 * n_lines + 1) * 8.0
    mem_mb = (chunk * bytes_per_pt) / (1024.0 * 1024.0)
    assert mem_mb <= max_mb + 1e-6


# =============================================================================
# 4. AST STRUCTURAL INSPECTION TESTS (FORBIDDEN PYTHON LOOPS)
# =============================================================================

def test_ast_scan_kernels_contain_no_forbidden_python_loops() -> None:
    """
    AST inspection test verifying that micro-batch scan kernels do NOT contain
    Python for/while loops and invoke jax.lax.scan directly.
    """
    import cochem_spycfit.physics_core.cochem_convolver as conv_module

    scan_functions = [
        conv_module.convolve_frequency_chunks_scan_1d_jax,
        conv_module.convolve_frequency_chunks_scan_2d_jax,
        conv_module.convolve_line_chunks_scan_1d_jax,
        conv_module.faddeeva_rational_jax,
    ]

    for fn in scan_functions:
        source = inspect.getsource(fn)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            assert not isinstance(node, (ast.For, ast.While, ast.AsyncFor)), (
                f"Forbidden Python loop {type(node).__name__} found in scan kernel {fn.__name__} at line {node.lineno}."
            )

        # Check for jax.lax.scan presence in the AST
        has_scan_call = any(
            isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "scan") or
                (isinstance(node.func, ast.Name) and "scan" in node.func.id)
            )
            for node in ast.walk(tree)
        )
        assert has_scan_call, f"Kernel {fn.__name__} must invoke jax.lax.scan."


# =============================================================================
# 5. END-TO-END MACROSCOPIC BROADENER & RESULT VALIDATION
# =============================================================================

def test_end_to_end_broadener_single_temperature() -> None:
    """Validate end-to-end broadening with MacroscopicSpectralBroadener."""
    lines = SpectralLines(
        frequencies=[100.0, 105.0, 110.0],
        intensities=[1.0, 2.5, 1.5],
        lower_state_energies=[10.0, 25.0, 40.0],
    )
    config = ConvolutionConfig(
        profile_type=ProfileType.GAUSSIAN,
        fwhm_gaussian=0.1,
        grid_resolution=0.01,
        cutoff_fwhm=20.0,
    )

    broadener = MacroscopicSpectralBroadener(config)
    result: ConvolvedSpectrumResult = broadener.broaden(lines)

    assert len(result.frequencies) > 100
    assert len(result.intensities) == len(result.frequencies)
    assert result.peak_frequency >= 90.0 and result.peak_frequency <= 120.0
    assert result.peak_intensity > 0.0
    assert result.execution_time_seconds > 0.0

    # Total area should approximately equal sum of input intensities (5.0)
    expected_total_intensity = sum(lines.intensities)
    assert abs(result.total_integrated_intensity - expected_total_intensity) < 1e-4


def test_end_to_end_broadener_multi_temperature() -> None:
    """Validate end-to-end multi-temperature broadening and result structures."""
    lines = SpectralLines(
        frequencies=[150.0, 160.0],
        intensities=[1.0, 1.0],
        lower_state_energies=[0.0, 200.0],
    )
    conv_config = ConvolutionConfig(
        profile_type=ProfileType.PSEUDO_VOIGT,
        fwhm_gaussian=0.05,
        fwhm_lorentzian=0.05,
        grid_resolution=0.01,
    )
    part_config = PartitionGridConfig(
        temperatures=[50.0, 150.0, 298.15],
        t_ref=298.15,
        power_law_index=1.5,
        q_ref=100.0,
    )

    broadener = MacroscopicSpectralBroadener(conv_config)
    result = broadener.broaden(lines, partition_config=part_config)

    assert result.temperatures == [50.0, 150.0, 298.15]
    assert result.multi_temp_intensities is not None
    assert len(result.multi_temp_intensities) == 3
    assert len(result.multi_temp_intensities[0]) == len(result.frequencies)


# =============================================================================
# 6. NEGATIVE TESTS & EXCEPTION HIERARCHY
# =============================================================================

def test_negative_temperature_raises_thermal_scaling_error() -> None:
    """Verify that non-physical temperatures T <= 0 raise ThermalScalingError."""
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[-10.0, 300.0])

    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[0.0, 300.0])

    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[])


def test_missing_energies_for_partition_grid_raises_thermal_scaling_error() -> None:
    """Verify that evaluating partition grid without lower state energies raises ThermalScalingError."""
    lines_no_energies = SpectralLines(
        frequencies=[100.0, 200.0],
        intensities=[1.0, 2.0],
        lower_state_energies=None,
    )
    part_config = PartitionGridConfig(temperatures=[100.0, 200.0])

    with pytest.raises(ThermalScalingError):
        evaluate_partition_grid(lines_no_energies, part_config)


def test_mismatched_spectral_lines_raises_spectral_data_error() -> None:
    """Verify that mismatched list lengths in SpectralLines raise SpectralDataError."""
    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, 200.0],
            intensities=[1.0],  # Mismatched length
        )

    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, -50.0],  # Negative frequency
            intensities=[1.0, 1.0],
        )

    with pytest.raises(SpectralDataError):
        SpectralLines(
            frequencies=[100.0, 200.0],
            intensities=[1.0, -0.5],  # Negative intensity
        )


def test_invalid_grid_raises_invalid_grid_error() -> None:
    """Verify that non-monotonic or inverted frequency grids raise InvalidGridError."""
    lines = SpectralLines(frequencies=[100.0], intensities=[1.0])

    # Inverted grid
    with pytest.raises(InvalidGridError):
        ConvolutionConfig(grid_min=200.0, grid_max=100.0)

    # Non-monotonic custom grid
    with pytest.raises(InvalidGridError):
        convolve_spectrum(lines, freq_grid=[100.0, 99.0, 101.0])


def test_invalid_vram_and_chunk_raises_micro_batch_error() -> None:
    """Verify that non-positive VRAM or chunk sizes raise MicroBatchError."""
    with pytest.raises(MicroBatchError):
        calculate_optimal_chunk_size(1000, 10, max_vram_mb=-5.0)

    with pytest.raises(MicroBatchError):
        calculate_optimal_chunk_size(1000, 10, explicit_chunk_size=0)


def test_unsupported_profile_type_raises_profile_error() -> None:
    """Verify that invalid profile type string raises ProfileError."""
    with pytest.raises(ProfileError):
        get_profile_type_code("unsupported_shape")


def test_energy_unit_conversions() -> None:
    """Verify conversion factors for all supported spectroscopic energy units."""
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.CM_1) - C2_RADIATION_CONST) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin("cm-1") - C2_RADIATION_CONST) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.MHZ) - H_OVER_K_B * 1.0e6) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.GHZ) - H_OVER_K_B * 1.0e9) < 1e-12
    assert abs(get_energy_conversion_factor_to_kelvin(EnergyUnit.KELVIN) - 1.0) < 1e-12
    assert get_energy_conversion_factor_to_kelvin(EnergyUnit.JOULE) > 0.0

    with pytest.raises(ThermalScalingError):
        get_energy_conversion_factor_to_kelvin("invalid_unit")


def test_per_line_fwhm_overrides_and_broadening() -> None:
    """Verify that per-line Gaussian and Lorentzian FWHM overrides are respected in broadening."""
    lines = SpectralLines(
        frequencies=[100.0, 110.0],
        intensities=[1.0, 2.0],
        fwhm_gaussian_overrides=[0.05, 0.20],
        fwhm_lorentzian_overrides=[0.02, 0.10],
        quantum_numbers=["J=1-0", "J=2-1"],
    )
    assert lines.quantum_numbers == ["J=1-0", "J=2-1"]

    config = ConvolutionConfig(profile_type=ProfileType.VOIGT)
    broadener = MacroscopicSpectralBroadener(config)
    result = broadener.broaden(lines)

    assert result.profile_type == ProfileType.VOIGT
    assert len(result.frequencies) > 0
    assert result.peak_intensity > 0.0


def test_spectral_lines_validation_errors() -> None:
    """Verify validation errors for malformed SpectralLines inputs."""
    # Empty frequencies
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[], intensities=[])

    # Mismatched lower state energies length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], lower_state_energies=[10.0])

    # Mismatched quantum numbers length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], quantum_numbers=["J=1-0"])

    # Mismatched Gaussian overrides length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_gaussian_overrides=[0.1])

    # Negative Gaussian override
    with pytest.raises(ProfileError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_gaussian_overrides=[0.1, -0.05])

    # Mismatched Lorentzian overrides length
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_lorentzian_overrides=[0.1])

    # Negative Lorentzian override
    with pytest.raises(ProfileError):
        SpectralLines(frequencies=[100.0, 200.0], intensities=[1.0, 1.0], fwhm_lorentzian_overrides=[-0.1, 0.05])


def test_partition_grid_config_validation_errors() -> None:
    """Verify validation errors for malformed PartitionGridConfig inputs."""
    # Tabulated partition values length mismatch
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0, 200.0], partition_values=[50.0])

    # Non-positive tabulated partition value
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0, 200.0], partition_values=[50.0, -10.0])

    # Energy levels without degeneracies
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0, 50.0])

    # Degeneracies length mismatch with energy levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0, 50.0], degeneracies=[1.0])

    # Empty energy levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[], degeneracies=[])


def test_tabulated_partition_values_evaluation() -> None:
    """Verify evaluate_partition_grid with explicit tabulated partition function values."""
    lines = SpectralLines(
        frequencies=[100.0, 200.0],
        intensities=[1.0, 2.0],
        lower_state_energies=[10.0, 50.0],
    )
    part_config = PartitionGridConfig(
        temperatures=[100.0, 200.0],
        partition_values=[25.0, 60.0],
        q_ref=25.0,
        t_ref=100.0,
    )
    result = evaluate_partition_grid(lines, part_config)
    assert result.partition_function_values == [25.0, 60.0]
    assert len(result.scaled_intensities) == 2


def test_broadener_helper_methods_and_custom_grids() -> None:
    """Verify MacroscopicSpectralBroadener helper methods: build_grid and custom grid checks."""
    lines = SpectralLines(frequencies=[100.0, 110.0], intensities=[1.0, 1.0])

    # Explicit grid_points
    cfg_points = ConvolutionConfig(grid_points=500)
    broadener = MacroscopicSpectralBroadener(cfg_points)
    grid = broadener.build_grid(lines)
    assert len(grid) == 500

    # Auto resolution grid
    cfg_auto = ConvolutionConfig(fwhm_gaussian=0.5)
    broadener_auto = MacroscopicSpectralBroadener(cfg_auto)
    grid_auto = broadener_auto.build_grid(lines)
    assert len(grid_auto) >= 100

    # Custom grid with single element raises InvalidGridError
    with pytest.raises(InvalidGridError):
        broadener.broaden(lines, freq_grid=[100.0])

    # Custom grid with NaN raises InvalidGridError
    with pytest.raises(InvalidGridError):
        broadener.broaden(lines, freq_grid=[100.0, float("nan"), 110.0])


def test_profile_type_code_mapping_and_padding() -> None:
    """Verify get_profile_type_code and chunk padding logic in convolve_spectrum."""
    assert get_profile_type_code(ProfileType.GAUSSIAN) == 0
    assert get_profile_type_code(ProfileType.LORENTZIAN) == 1
    assert get_profile_type_code(ProfileType.PSEUDO_VOIGT) == 2
    assert get_profile_type_code(ProfileType.VOIGT) == 3

    # Force padding by specifying chunk_size that does not divide n_freqs
    lines = SpectralLines(frequencies=[100.0], intensities=[1.0])
    cfg = ConvolutionConfig(chunk_size=17, grid_points=100)
    result = convolve_spectrum(lines, config=cfg)
    assert len(result.frequencies) == 100
    assert result.num_chunks == 6  # ceil(100 / 17) = 6


def test_broadener_evaluate_partition_grid_and_constant_q() -> None:
    """Verify broadener.evaluate_partition_grid and default constant Q(T) partition function."""
    lines = SpectralLines(
        frequencies=[100.0],
        intensities=[1.0],
        lower_state_energies=[10.0],
    )
    # Neither power_law_index nor energy_levels provided -> uses constant Q(T) = Q_ref
    part_cfg = PartitionGridConfig(temperatures=[100.0, 200.0], q_ref=42.0)
    broadener = MacroscopicSpectralBroadener()
    res = broadener.evaluate_partition_grid(lines, part_cfg)
    assert res.partition_function_values == [42.0, 42.0]



--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_convolver_phase3.py ---
"""
CoChem-SpycFit: Comprehensive Phase 3 Physical Unit Test Suite for Macroscopic Spectral Broadener.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic doubles
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of exact Faddeeva Voigt profile via jax.scipy.special.erfcx
- Validation of multi-temperature Boltzmann scaling with tensor broadcasting [N_lines, N_temperatures]
- Validation of static masked batching via jax.lax.scan returning registered pure PyTree ConvolverPyTree
- AST structural verification enforcing pure jax.lax.scan without forbidden Python for/while loops
- Zero-pass audit verifying no pass statements across codebase.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
import scipy.special as scipy_sp

from cochem_spycfit.physics_core.cochem_convolver import (
    C2_RADIATION_CONST,
    DEFAULT_MAX_VRAM_MB,
    FOUR_LN_2,
    FWHM_TO_SIGMA_GAUSS,
    GAUSS_PREFACTOR,
    H_OVER_K_B,
    LORENTZ_PREFACTOR,
    PLANCK_H,
    SPEED_OF_LIGHT_CM_S,
    SPEED_OF_LIGHT_M_S,
    SQRT_2_LN_2,
    ConvolutionConfig,
    ConvolvedSpectrumResult,
    ConvolverError,
    ConvolverPyTree,
    EnergyUnit,
    InvalidGridError,
    MacroscopicSpectralBroadener,
    MicroBatchError,
    PartitionGridConfig,
    PartitionGridResult,
    ProfileError,
    ProfileType,
    SpectralDataError,
    SpectralLines,
    ThermalScalingError,
    build_frequency_grid,
    calculate_optimal_chunk_size,
    compute_boltzmann_scaling_grid_jax,
    convolute_multi_temperature,
    convolve_frequency_chunks_scan_1d_jax,
    convolve_frequency_chunks_scan_2d_jax,
    convolve_line_chunks_scan_1d_jax,
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    evaluate_partition_grid,
    evaluate_profile_kernel_jax,
    faddeeva_rational_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    get_energy_conversion_factor_to_kelvin,
    get_profile_type_code,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
    static_masked_scan,
    true_voigt_profile,
)

# Enforce JAX FP64 precision globally
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. EXACT FADDEEVA VOIGT PROFILE TESTS (true_voigt_profile via erfcx)
# =============================================================================

def test_true_voigt_profile_erfcx_mathematical_precision() -> None:
    """Validate true_voigt_profile numerical match with scipy.special.voigt_profile within machine precision."""
    x_points = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    sigma_vals = [0.05, 0.1, 0.5, 1.2, 2.5]
    gamma_vals = [0.02, 0.1, 0.4, 0.8, 2.0]

    for sigma in sigma_vals:
        for gamma in gamma_vals:
            # JAX implementation
            jax_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
            jax_arr = np.array(jax_vals, dtype=np.float64)

            # Reference SciPy Faddeeva voigt_profile
            sp_vals = scipy_sp.voigt_profile(x_points, sigma, gamma)

            max_diff = np.max(np.abs(jax_arr - sp_vals))
            assert max_diff < 1e-13, (
                f"Voigt profile mismatch for sigma={sigma}, gamma={gamma}: max_diff={max_diff}"
            )


def test_true_voigt_profile_gaussian_limit() -> None:
    """When gamma -> 0, true_voigt_profile must analytically converge to a pure Gaussian distribution."""
    sigma = 0.35
    gamma = 1e-25  # Effectively zero Lorentzian width
    x_points = np.linspace(-3.0 * sigma, 3.0 * sigma, 50, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Gaussian: (1 / (sigma * sqrt(2*pi))) * exp(-x^2 / (2*sigma^2))
    expected_gaussian = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (x_points / sigma) ** 2)

    max_diff = np.max(np.abs(voigt_arr - expected_gaussian))
    assert max_diff < 1e-13, f"Gaussian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_lorentzian_limit() -> None:
    """When sigma is small, true_voigt_profile must approach a pure Lorentzian profile."""
    sigma = 1e-15
    gamma = 0.5
    x_points = np.linspace(-4.0, 4.0, 41, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_points), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    # Pure Lorentzian: (1 / pi) * (gamma / (x^2 + gamma^2))
    expected_lorentzian = (1.0 / math.pi) * (gamma / (x_points ** 2 + gamma ** 2))

    max_diff = np.max(np.abs(voigt_arr - expected_lorentzian))
    assert max_diff < 1e-11, f"Lorentzian limit mismatch: max_diff={max_diff}"


def test_true_voigt_profile_unit_area_normalization() -> None:
    """Verify analytical unit area normalization int_{-\\infty}^{\\infty} V(x) dx == 1.0."""
    sigma = 0.25
    gamma = 0.30
    cutoff = 200.0 * (sigma + gamma)
    n_pts = 100001
    x_grid = np.linspace(-cutoff, cutoff, n_pts, dtype=np.float64)

    voigt_vals = true_voigt_profile(jnp.array(x_grid), sigma, gamma)
    voigt_arr = np.array(voigt_vals, dtype=np.float64)

    integrated_area = float(np.trapezoid(voigt_arr, x_grid))
    # Account for analytical Lorentzian asymptotic tail 2 * int_{cutoff}^\\infty (gamma / (pi * x^2)) dx = 2*gamma / (pi * cutoff)
    tail_analytical = (2.0 * gamma) / (math.pi * cutoff)
    total_area = integrated_area + tail_analytical
    assert abs(total_area - 1.0) < 1e-6, f"Voigt integrated total area was {total_area}, expected ~1.0"

    # Also verify exact agreement with SciPy integration over identical domain
    sp_vals = scipy_sp.voigt_profile(x_grid, sigma, gamma)
    sp_area = float(np.trapezoid(sp_vals, x_grid))
    assert abs(integrated_area - sp_area) < 1e-13, f"Integration mismatch vs SciPy: diff={abs(integrated_area - sp_area)}"


def test_true_voigt_profile_symmetry_and_peak() -> None:
    """Verify profile symmetry V(-x) == V(x) and exact analytical peak height at x=0."""
    sigma = 0.4
    gamma = 0.25
    x_eval = jnp.array([0.0, -1.5, 1.5, -3.2, 3.2], dtype=jnp.float64)

    v_vals = true_voigt_profile(x_eval, sigma, gamma)

    # Peak height at x=0: Re[erfcx(gamma / (sigma * sqrt(2)))] / (sigma * sqrt(2*pi))
    u_0 = gamma / (sigma * math.sqrt(2.0))
    expected_peak = float(jsp.erfcx(u_0)) / (sigma * math.sqrt(2.0 * math.pi))
    actual_peak = float(v_vals[0])
    assert abs(actual_peak - expected_peak) < 1e-15

    # Symmetry checks
    assert abs(float(v_vals[1]) - float(v_vals[2])) < 1e-15
    assert abs(float(v_vals[3]) - float(v_vals[4])) < 1e-15


def test_true_voigt_profile_jit_and_broadcasting() -> None:
    """Verify JIT compilation and tensor broadcasting across 2D/3D shapes."""
    jitted_voigt = jax.jit(true_voigt_profile)

    # 2D broadcast: x shape (20, 1), sigma (1, 5), gamma scalar
    x_2d = jnp.linspace(-2.0, 2.0, 20)[:, None]
    sigma_row = jnp.linspace(0.1, 0.5, 5)[None, :]
    gamma_val = 0.15

    res_2d = jitted_voigt(x_2d, sigma_row, gamma_val)
    assert res_2d.shape == (20, 5)
    assert jnp.all(jnp.isfinite(res_2d))
    assert jnp.all(res_2d > 0.0)


# =============================================================================
# 2. MULTI-TEMPERATURE CONVOLUTION TESTS (convolute_multi_temperature)
# =============================================================================

def test_convolute_multi_temperature_shape_broadcasting() -> None:
    """Verify output shape is strictly [N_lines, N_temperatures] with pure JAX tensor broadcasting."""
    n_lines = 40
    n_temps = 8

    freqs = jnp.linspace(100000.0, 110000.0, n_lines)
    intensities = jnp.linspace(0.1, 5.0, n_lines)
    energies = jnp.linspace(0.0, 500.0, n_lines)
    temps = jnp.array([10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 296.0, 350.0], dtype=jnp.float64)
    # Power-law partition function: Q(T) = (T / 296)^1.5
    q_temps = jnp.power(temps / 296.0, 1.5)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=296.0,
        q_ref=1.0,
    )

    assert scaled_i.shape == (n_lines, n_temps), f"Expected shape ({n_lines}, {n_temps}), got {scaled_i.shape}"
    assert jnp.all(jnp.isfinite(scaled_i))
    assert jnp.all(scaled_i >= 0.0)


def test_convolute_multi_temperature_reference_invariance() -> None:
    """At T == T_ref and Q(T) == Q_ref, scaled intensities must identically equal reference intensities."""
    n_lines = 15
    freqs = jnp.linspace(90000.0, 95000.0, n_lines)
    intensities = jnp.array([0.5, 1.2, 3.4, 0.8, 2.1, 4.0, 1.5, 0.3, 2.7, 3.1, 1.9, 0.4, 2.8, 3.9, 1.1])
    energies = jnp.linspace(10.0, 200.0, n_lines)

    t_ref = 296.0
    q_ref = 1.0
    temps = jnp.array([t_ref], dtype=jnp.float64)
    q_temps = jnp.array([q_ref], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # Column 0 corresponds to T_ref
    scaled_col = scaled_i[:, 0]
    max_diff = float(jnp.max(jnp.abs(scaled_col - intensities)))
    assert max_diff < 1e-14, f"Reference temperature scaling differed from original: max_diff={max_diff}"


def test_convolute_multi_temperature_boltzmann_population_redistribution() -> None:
    """Validate that high-energy transitions gain intensity at high T while ground-state transitions deplete."""
    # Line 0: Ground state E_lower = 0 cm^-1
    # Line 1: High energy state E_lower = 300 cm^-1
    freqs = jnp.array([100000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 1.0], dtype=jnp.float64)
    energies = jnp.array([0.0, 300.0], dtype=jnp.float64)

    t_low = 50.0
    t_high = 300.0
    t_ref = 300.0
    q_ref = 1.0
    temps = jnp.array([t_low, t_high], dtype=jnp.float64)
    q_temps = jnp.array([1.0, 1.0], dtype=jnp.float64)

    scaled_i = convolute_multi_temperature(
        frequencies=freqs,
        intensities=intensities,
        Q_T=q_temps,
        temp_grid=temps,
        lower_energies=energies,
        t_ref=t_ref,
        q_ref=q_ref,
    )

    # At low temperature (T=50K), high-energy line (Line 1) must be exponentially suppressed relative to Line 0
    ratio_t_low = float(scaled_i[1, 0] / scaled_i[0, 0])
    # Expected Boltzmann ratio: exp(- C2 * 300 * (1/50 - 1/300)) / exp(0)
    expected_ratio = math.exp(-C2_RADIATION_CONST * 300.0 * (1.0 / 50.0 - 1.0 / 300.0))
    assert abs(ratio_t_low - expected_ratio) < 1e-12
    assert ratio_t_low < 0.01  # Heavily depopulated at 50K


def test_convolute_multi_temperature_jit_compiled() -> None:
    """Verify convolute_multi_temperature JIT compilation and execution purity."""
    jitted_fn = jax.jit(convolute_multi_temperature)

    freqs = jnp.array([80000.0, 90000.0, 100000.0], dtype=jnp.float64)
    intensities = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    q_temps = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    temps = jnp.array([100.0, 200.0, 400.0], dtype=jnp.float64)

    res = jitted_fn(freqs, intensities, q_temps, temps)
    assert res.shape == (3, 3)
    assert jnp.all(jnp.isfinite(res))


# =============================================================================
# 3. STATIC MASKED SCAN & CONVOLVER PYTREE TESTS
# =============================================================================

def test_convolver_pytree_registration_and_roundtrip() -> None:
    """Verify ConvolverPyTree is registered with JAX PyTree system and supports flatten/unflatten/tree_map."""
    freqs = jnp.linspace(100.0, 200.0, 50)
    intensities = jnp.sin(jnp.linspace(0.0, math.pi, 50))
    mask = jnp.ones(50, dtype=jnp.bool_)
    total_i = jnp.array(5.25, dtype=jnp.float64)

    pytree = ConvolverPyTree(
        frequencies=freqs,
        intensities=intensities,
        mask=mask,
        total_intensity=total_i,
        num_lines=50,
        chunk_size=10000,
        num_chunks=1,
        profile_type="voigt",
    )

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    assert len(leaves) == 4
    assert jnp.allclose(leaves[0], freqs)
    assert jnp.allclose(leaves[1], intensities)
    assert jnp.allclose(leaves[2], mask)
    assert jnp.allclose(leaves[3], total_i)

    # Unflatten
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, ConvolverPyTree)
    assert reconstructed.num_lines == 50
    assert reconstructed.chunk_size == 10000
    assert reconstructed.num_chunks == 1
    assert reconstructed.profile_type == "voigt"
    assert jnp.allclose(reconstructed.intensities, intensities)

    # Tree Map: Multiply all array leaves by 2.0
    mapped_tree = jax.tree_util.tree_map(lambda x: x * 2.0, pytree)
    assert isinstance(mapped_tree, ConvolverPyTree)
    assert jnp.allclose(mapped_tree.total_intensity, total_i * 2.0)


def test_static_masked_scan_returns_pure_pytree() -> None:
    """Verify static_masked_scan execution returns a pure ConvolverPyTree."""
    freqs = [100.0, 102.0, 105.0, 108.0, 110.0]
    intensities = [1.0, 2.5, 0.8, 3.2, 1.5]
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    result_pytree = static_masked_scan(
        lines=lines,
        chunk_size=2,
        grid_points=200,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.05,
        gamma=0.02,
    )

    assert isinstance(result_pytree, ConvolverPyTree)
    assert result_pytree.num_lines == 5
    assert result_pytree.chunk_size == 2
    assert result_pytree.num_chunks == 3  # ceil(5 / 2) = 3 chunks (padded length 6)
    assert len(result_pytree.frequencies) == 200
    assert len(result_pytree.intensities) == 200
    assert float(result_pytree.total_intensity) > 0.0


def test_static_masked_scan_masking_authenticity() -> None:
    """Verify that padded trailing lines in static masked batching have exactly zero effect on output spectrum."""
    freqs = [100.0, 105.0, 110.0]
    intensities = [2.0, 3.0, 1.0]
    lines_original = SpectralLines(frequencies=freqs, intensities=intensities)

    # Run with chunk_size=2 (produces 2 chunks: chunk 1 has 2 lines, chunk 2 has 1 line + 1 padded trailing line)
    pytree_chunked = static_masked_scan(
        lines=lines_original,
        chunk_size=2,
        grid_points=300,
        grid_min=95.0,
        grid_max=115.0,
        sigma=0.08,
        gamma=0.04,
    )

    # Run monolithic reference (all lines at once)
    grid_arr = pytree_chunked.frequencies
    delta_nu = grid_arr[:, None] - jnp.array(freqs, dtype=jnp.float64)[None, :]
    prof_mono = true_voigt_profile(delta_nu, 0.08, 0.04)
    spec_mono = jnp.dot(prof_mono, jnp.array(intensities, dtype=jnp.float64))

    max_diff = float(jnp.max(jnp.abs(pytree_chunked.intensities - spec_mono)))
    assert max_diff < 1e-13, f"Masked scan output differed from unchunked monolithic reference: max_diff={max_diff}"


def test_static_masked_scan_chunk_size_invariance() -> None:
    """Verify that convolved spectrum is invariant to the chunk_size parameter."""
    freqs = np.linspace(100.0, 120.0, 37, dtype=np.float64).tolist()
    intensities = np.linspace(0.5, 4.0, 37, dtype=np.float64).tolist()
    lines = SpectralLines(frequencies=freqs, intensities=intensities)

    grid_pts = 500
    g_min = 95.0
    g_max = 125.0
    sigma = 0.06
    gamma = 0.03

    # Test chunk sizes: 1, 5, 10, 37, 100
    chunk_sizes = [1, 5, 10, 37, 100]
    results = []

    for cs in chunk_sizes:
        res = static_masked_scan(
            lines=lines,
            chunk_size=cs,
            grid_points=grid_pts,
            grid_min=g_min,
            grid_max=g_max,
            sigma=sigma,
            gamma=gamma,
        )
        results.append(res.intensities)

    ref_spec = results[0]
    for idx, cs in enumerate(chunk_sizes[1:], start=1):
        diff = float(jnp.max(jnp.abs(results[idx] - ref_spec)))
        assert diff < 1e-13, f"Chunk size {cs} produced mismatch vs chunk size 1: diff={diff}"


def test_static_masked_scan_input_formats() -> None:
    """Verify static_masked_scan works identically with SpectralLines, tuple of arrays, and dict inputs."""
    f_list = [100.0, 102.0, 104.0]
    i_list = [1.0, 2.0, 3.0]

    # 1. SpectralLines model
    res1 = static_masked_scan(SpectralLines(frequencies=f_list, intensities=i_list), chunk_size=2, grid_points=100)

    # 2. Tuple of arrays
    res2 = static_masked_scan((jnp.array(f_list), jnp.array(i_list)), chunk_size=2, grid_points=100)

    # 3. Dict
    res3 = static_masked_scan({"frequencies": f_list, "intensities": i_list}, chunk_size=2, grid_points=100)

    assert jnp.allclose(res1.intensities, res2.intensities, atol=1e-14)
    assert jnp.allclose(res1.intensities, res3.intensities, atol=1e-14)


# =============================================================================
# 4. AST LOOP-FREE & ZERO-MOCK COMPLIANCE VERIFICATION
# =============================================================================

def test_ast_loop_free_compliance() -> None:
    """
    AST structural verification ensuring that JAX kernels use pure vectorization and jax.lax.scan,
    with ZERO Python for/while loops in performance-critical paths.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    critical_functions = [
        "true_voigt_profile",
        "convolute_multi_temperature",
        "_scan_step_masked_kernel",
        "faddeeva_rational_jax",
        "gaussian_kernel_jax",
        "lorentzian_kernel_jax",
        "pseudo_voigt_kernel_jax",
        "faddeeva_voigt_kernel_jax",
        "evaluate_profile_kernel_jax",
        "evaluate_partition_function_jax",
        "compute_boltzmann_scaling_grid_jax",
        "convolve_frequency_chunks_scan_1d_jax",
        "convolve_frequency_chunks_scan_2d_jax",
        "convolve_line_chunks_scan_1d_jax",
        "convolve_spectrum_monolithic_jax",
    ]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in critical_functions:
            for child in ast.walk(node):
                assert not isinstance(child, (ast.For, ast.While)), (
                    f"Forbidden Python loop found in JAX kernel '{node.name}' at line {getattr(child, 'lineno', '?')}."
                )


def test_zero_pass_and_ast_integrity_mandate() -> None:
    """
    Authenticity and zero-pass mandate audit:
    Verifies that cochem_convolver contains zero pass statements and maintains pure functional structure.
    """
    from cochem_spycfit.physics_core import cochem_convolver

    source_code = inspect.getsource(cochem_convolver)
    tree = ast.parse(source_code)

    # Check for pass statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            pytest.fail(f"Forbidden 'pass' statement found in cochem_convolver.py at line {getattr(node, 'lineno', '?')}.")


def test_robust_nan_and_negative_input_validation() -> None:
    """Validate that NaN/Inf and unphysical values in energies and degeneracies are rejected."""
    # NaN in lower_state_energies
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0], intensities=[1.0], lower_state_energies=[float("nan")])

    # Negative lower_state_energies
    with pytest.raises(SpectralDataError):
        SpectralLines(frequencies=[100.0], intensities=[1.0], lower_state_energies=[-5.0])

    # NaN in energy_levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[float("nan")], degeneracies=[1.0])

    # Negative energy_levels
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[-10.0], degeneracies=[1.0])

    # Zero or negative degeneracies
    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0], degeneracies=[0.0])

    with pytest.raises(ThermalScalingError):
        PartitionGridConfig(temperatures=[100.0], energy_levels=[0.0], degeneracies=[-1.0])


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_extractor_physical.py ---
import os
import sys
import h5py
import shutil
import numpy as np
import pytest
import subprocess
import logging
import psutil
import atexit
from pathlib import Path

# 1. Registry Consistency & Air-Gap Enforcement
# Replaced __file__ static lookup with environment variables / Path.home() dynamic lookup
MODULE_DIR: Path = Path(os.environ.get("COCHEM_SPYCFIT_DIR", Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.home() / ".cochem" / "CoChem-SpycFit"))
EXTRACTOR_SCRIPT: Path = MODULE_DIR / "src" / "cochem_spycfit" / "quantum_extractor.py"
REAL_ARTIFACTS_DIR: Path = Path(os.environ.get("COCHEM_ARTIFACTS_DIR", MODULE_DIR / "artifacts_test"))

# 3. Graceful Failure & Subprocess Safety (Zombie Process Sweeping)
def sweep_zombie_processes() -> None:
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.Error, OSError) as e:
        logging.error(f"Failed to sweep zombie processes: {e}")
        raise RuntimeError("Zombie process sweeping failed") from e

atexit.register(sweep_zombie_processes)

def test_quantum_extractor_execution(tmp_path: Path) -> None:
    """
    Executes the quantum extractor against a REAL physical landscape.h5.
    Mocks and stubs are strictly forbidden. Exception deflections are tightened.
    """
    ephemeral_artifact_dir: Path = tmp_path / "artifacts"
    ephemeral_artifact_dir.mkdir(exist_ok=True)
    
    # Locate real physical HDF5 registry
    real_h5_path: Path = REAL_ARTIFACTS_DIR / "artifacts_test" / "landscape.h5"
    if not real_h5_path.exists():
        real_h5_path = REAL_ARTIFACTS_DIR / "landscape.h5"
    if not real_h5_path.exists():
        # Strict failure on missing physical data (No exception deflection)
        raise FileNotFoundError(f"[MISSING DATA] Real physical landscape.h5 not found at {real_h5_path}. Synthetic generation is forbidden.")
        
    # Copy to ephemeral dir to prevent mutating the source
    ephemeral_h5_path: Path = ephemeral_artifact_dir / "landscape.h5"
    shutil.copy(real_h5_path, ephemeral_h5_path)
    
    # 2. Verify isomer exists in real data
    isomer_id: str = "isomer_001"
    with h5py.File(ephemeral_h5_path, "r") as f:
        if isomer_id not in f:
            raise KeyError(f"[MISSING DATA] Isomer {isomer_id} not found in physical HDF5 registry.")
        n_dim: int = len(f[isomer_id]["symbols"][:]) * 3

    # 3. Native invocation of the extractor (Strict Subprocess Safety)
    env = os.environ.copy()
    env["COCHEM_ARTIFACT_DIR"] = str(ephemeral_artifact_dir)
    env["COCHEM_TARGET_ISOMER"] = isomer_id
    
    cmd = [sys.executable, str(EXTRACTOR_SCRIPT)]
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=60)
        logging.info("Extractor execution succeeded.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Extractor failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        raise RuntimeError(f"Physical execution of extractor failed with return code {e.returncode}") from e
    except subprocess.TimeoutExpired as e:
        logging.error(f"Extractor timed out after {e.timeout}s")
        raise RuntimeError(f"Extractor timed out after {e.timeout}s") from e

    # 4. Domain-specific correctness checks
    tensor_buffer: Path = ephemeral_artifact_dir / "spycfit_tensor_buffer.npz"
    if not tensor_buffer.exists():
        raise FileNotFoundError(f"[MISSING DATA] spycfit_tensor_buffer.npz was not produced by the physical run in {ephemeral_artifact_dir}.")
    
    npz_data = np.load(tensor_buffer, allow_pickle=True)
    assert "mw_hessian" in npz_data.files, "[MISSING DATA] Mass-weighted Hessian missing from output"
    
    # Mass weighted Hessian must maintain dimensions
    mw_hessian = npz_data["mw_hessian"]
    assert mw_hessian.shape == (n_dim, n_dim), f"Invalid Hessian shape: {mw_hessian.shape} != ({n_dim}, {n_dim})"
    
    # Ensure Hermitian symmetry is strict
    np.testing.assert_allclose(mw_hessian, mw_hessian.T, rtol=1e-5, err_msg="Hessian Hermitian symmetry violation")

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.