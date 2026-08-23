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
        if self.lower_state_energies is not None and len(self.lower_state_energies) != n_lines:
            raise SpectralDataError(
                f"Mismatch: lower_state_energies length ({len(self.lower_state_energies)}) does not match frequencies length ({n_lines})."
            )
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
    t_ref: float = Field(default=296.0, gt=0.0, description="Reference temperature in Kelvin (default 296 K)")
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
    t_ref: float = Field(default=296.0, gt=0.0, description="Reference temperature in Kelvin")
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

    Valid across the upper complex half-plane (Im(z) >= 0) with machine accuracy (~ 1e-14).
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
def evaluate_profile_kernel_jax(
    delta_nu: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_l: jnp.ndarray,
    profile_code: int,
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
    temperatures: jnp.ndarray,
    power_law_index: Optional[float],
    q_ref: float,
    t_ref: float,
    energy_levels: Optional[jnp.ndarray],
    degeneracies: Optional[jnp.ndarray],
    energy_to_k: float,
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
        # Default: at least 10 sample points per FWHM
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
    # Enums & Models
    "ProfileType",
    "EnergyUnit",
    "SpectralLines",
    "ConvolutionConfig",
    "PartitionGridConfig",
    "PartitionGridResult",
    "ConvolvedSpectrumResult",
    # JAX Kernels
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
    # High-level Utilities & Engine
    "get_energy_conversion_factor_to_kelvin",
    "get_profile_type_code",
    "calculate_optimal_chunk_size",
    "build_frequency_grid",
    "evaluate_partition_grid",
    "convolve_spectrum",
    "MacroscopicSpectralBroadener",
]
