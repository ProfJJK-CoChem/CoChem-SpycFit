"""Versioned Instrument Profile Registry for CoChem-SPYFY.

Establishes ``grulab_surf`` (GruLabRotSpec/SURF cavity Fabry-Perot FTMW
spectrometer, verified at commit ``0486b877d2a38f5bbe50be3cb333d34e27c7c22d``)
as the default out-of-the-gate instrument profile while keeping secondary
broadband CP-FTMW and vibrational IR/Raman profiles first-class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_INSTRUMENT_PROFILE: str = "grulab_surf"

SURF_UPSTREAM_REPO: str = "https://github.com/GruLabRotSpec/SURF"
SURF_UPSTREAM_REPO_SLUG: str = "GruLabRotSpec/SURF"
SURF_VERIFIED_COMMIT_SHA: str = "0486b877d2a38f5bbe50be3cb333d34e27c7c22d"
SURF_SUPPORTED_VERSION_SPEC: str = ">=0.1.0,<0.2.0"


class InstrumentProfile(BaseModel):
    """Frozen hardware envelope and compatibility specification for an instrument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    upstream_repo: Optional[str] = None
    verified_commit_sha: Optional[str] = None
    supported_surf_version: Optional[str] = None
    technique: str = Field(min_length=1)
    spectral_observable: str = Field(min_length=1)
    spectral_unit: str = Field(default="MHz", min_length=1)
    doppler_doublet_expected: bool = False

    freq_min_mhz: float = Field(gt=0.0)
    freq_max_mhz: float = Field(gt=0.0)
    valon_lo_min_mhz: Optional[float] = Field(default=None, gt=0.0)
    valon_lo_max_mhz: Optional[float] = Field(default=None, gt=0.0)
    default_if_offset_mhz: float = Field(default=0.0, ge=0.0)

    step_size_min_mhz: float = Field(gt=0.0)
    step_size_max_mhz: float = Field(gt=0.0)
    default_step_size_mhz: float = Field(gt=0.0)
    default_fft_rbw_khz: float = Field(gt=0.0)
    default_scope_fft_span_mhz: Optional[float] = Field(default=None, gt=0.0)

    zaber_pos_min_mm: Optional[float] = Field(default=None, ge=0.0)
    zaber_pos_max_mm: Optional[float] = Field(default=None, gt=0.0)
    zaber_speed_min_mm_s: Optional[float] = Field(default=None, gt=0.0)
    zaber_speed_max_mm_s: Optional[float] = Field(default=None, gt=0.0)
    default_zaber_speed_mm_s: Optional[float] = Field(default=None, gt=0.0)

    rf_level_min_dbm: Optional[int] = None
    rf_level_max_dbm: Optional[int] = None
    rf_level_soft_warn_dbm: Optional[int] = None

    cavity_modes: Tuple[str, ...] = ()
    scan_apodization_windows: Tuple[str, ...] = ()
    control_apodization_windows: Tuple[str, ...] = ()

    recommended_targeted_window_min_mhz: float = Field(gt=0.0)
    recommended_targeted_window_max_mhz: float = Field(gt=0.0)
    recommended_survey_window_max_mhz: float = Field(gt=0.0)
    zaber_retune_overhead_s: float = Field(default=0.0, ge=0.0)

    @property
    def instrumental_resolution_mhz(self) -> float:
        """Return default instrumental FFT resolution bandwidth in MHz."""
        return self.default_fft_rbw_khz / 1000.0

    @property
    def recommended_targeted_window_mhz(self) -> Tuple[float, float]:
        """Return the recommended (min, max) targeted window bandwidth in MHz."""
        return (
            self.recommended_targeted_window_min_mhz,
            self.recommended_targeted_window_max_mhz,
        )

    @model_validator(mode="after")
    def _validate_envelope_consistency(self) -> InstrumentProfile:
        if self.freq_min_mhz >= self.freq_max_mhz:
            raise ValueError("freq_min_mhz must be strictly less than freq_max_mhz.")
        if self.step_size_min_mhz > self.step_size_max_mhz:
            raise ValueError("step_size_min_mhz must be <= step_size_max_mhz.")
        if not (self.step_size_min_mhz <= self.default_step_size_mhz <= self.step_size_max_mhz):
            raise ValueError("default_step_size_mhz must lie within [step_size_min_mhz, step_size_max_mhz].")
        if self.valon_lo_min_mhz is not None and self.valon_lo_max_mhz is not None:
            if self.valon_lo_min_mhz >= self.valon_lo_max_mhz:
                raise ValueError("valon_lo_min_mhz must be strictly less than valon_lo_max_mhz.")
        if self.zaber_pos_min_mm is not None and self.zaber_pos_max_mm is not None:
            if self.zaber_pos_min_mm >= self.zaber_pos_max_mm:
                raise ValueError("zaber_pos_min_mm must be strictly less than zaber_pos_max_mm.")
        if self.zaber_speed_min_mm_s is not None and self.zaber_speed_max_mm_s is not None:
            if self.zaber_speed_min_mm_s > self.zaber_speed_max_mm_s:
                raise ValueError("zaber_speed_min_mm_s must be <= zaber_speed_max_mm_s.")
            if self.default_zaber_speed_mm_s is not None and not (
                self.zaber_speed_min_mm_s <= self.default_zaber_speed_mm_s <= self.zaber_speed_max_mm_s
            ):
                raise ValueError(
                    "default_zaber_speed_mm_s must lie within [zaber_speed_min_mm_s, zaber_speed_max_mm_s]."
                )
        if self.rf_level_min_dbm is not None and self.rf_level_max_dbm is not None:
            if self.rf_level_min_dbm > self.rf_level_max_dbm:
                raise ValueError("rf_level_min_dbm must be <= rf_level_max_dbm.")
            if self.rf_level_soft_warn_dbm is not None and not (
                self.rf_level_min_dbm <= self.rf_level_soft_warn_dbm <= self.rf_level_max_dbm
            ):
                raise ValueError("rf_level_soft_warn_dbm must lie within [rf_level_min_dbm, rf_level_max_dbm].")
        if self.recommended_targeted_window_min_mhz > self.recommended_targeted_window_max_mhz:
            raise ValueError(
                "recommended_targeted_window_min_mhz must be <= recommended_targeted_window_max_mhz."
            )
        if self.recommended_survey_window_max_mhz < self.recommended_targeted_window_max_mhz:
            raise ValueError(
                "recommended_survey_window_max_mhz must be >= recommended_targeted_window_max_mhz."
            )
        return self


GRULAB_SURF_PROFILE = InstrumentProfile(
    profile_id="grulab_surf",
    display_name="GruLab SURF Cavity FTMW (Tennessee Tech Univ.)",
    upstream_repo=SURF_UPSTREAM_REPO,
    verified_commit_sha=SURF_VERIFIED_COMMIT_SHA,
    supported_surf_version=SURF_SUPPORTED_VERSION_SPEC,
    technique="cavity_ftmw_fabry_perot",
    spectral_observable="averaged_fft_magnitude_volts",
    spectral_unit="MHz",
    doppler_doublet_expected=True,
    freq_min_mhz=8000.0,
    freq_max_mhz=18000.0,
    valon_lo_min_mhz=7970.0,
    valon_lo_max_mhz=18000.0,
    default_if_offset_mhz=30.0,
    step_size_min_mhz=0.001,
    step_size_max_mhz=1.000,
    default_step_size_mhz=0.500,
    default_fft_rbw_khz=100.0,
    default_scope_fft_span_mhz=20.0,
    zaber_pos_min_mm=0.0,
    zaber_pos_max_mm=50.0,
    zaber_speed_min_mm_s=0.001,
    zaber_speed_max_mm_s=2.000,
    default_zaber_speed_mm_s=0.003,
    rf_level_min_dbm=-20,
    rf_level_max_dbm=20,
    rf_level_soft_warn_dbm=10,
    cavity_modes=("Continuous", "Pulsed"),
    scan_apodization_windows=("Hanning", "Hamming", "Blackman"),
    control_apodization_windows=("Rectangular", "Hamming", "Hanning", "Blackman"),
    recommended_targeted_window_min_mhz=2.0,
    recommended_targeted_window_max_mhz=20.0,
    recommended_survey_window_max_mhz=500.0,
    zaber_retune_overhead_s=14.0,
)

GENERIC_CP_FTMW_PROFILE = InstrumentProfile(
    profile_id="generic_cp_ftmw",
    display_name="Generic Chirped-Pulse FTMW (2-18 GHz Broadband)",
    upstream_repo=None,
    verified_commit_sha=None,
    supported_surf_version=None,
    technique="chirped_pulse_ftmw",
    spectral_observable="fft_magnitude_arbitrary",
    spectral_unit="MHz",
    doppler_doublet_expected=False,
    freq_min_mhz=2000.0,
    freq_max_mhz=18000.0,
    valon_lo_min_mhz=None,
    valon_lo_max_mhz=None,
    default_if_offset_mhz=0.0,
    step_size_min_mhz=0.005,
    step_size_max_mhz=10.0,
    default_step_size_mhz=0.025,
    default_fft_rbw_khz=25.0,
    default_scope_fft_span_mhz=None,
    zaber_pos_min_mm=None,
    zaber_pos_max_mm=None,
    zaber_speed_min_mm_s=None,
    zaber_speed_max_mm_s=None,
    default_zaber_speed_mm_s=None,
    rf_level_min_dbm=None,
    rf_level_max_dbm=None,
    rf_level_soft_warn_dbm=None,
    cavity_modes=(),
    scan_apodization_windows=("Kaiser-Bessel", "Hanning", "Blackman"),
    control_apodization_windows=(),
    recommended_targeted_window_min_mhz=50.0,
    recommended_targeted_window_max_mhz=2000.0,
    recommended_survey_window_max_mhz=10000.0,
    zaber_retune_overhead_s=0.0,
)

GENERIC_IR_RAMAN_PROFILE = InstrumentProfile(
    profile_id="generic_ir_raman",
    display_name="Generic Vibrational IR / Raman Spectrometer",
    upstream_repo=None,
    verified_commit_sha=None,
    supported_surf_version=None,
    technique="vibrational_ir_raman",
    spectral_observable="absorbance_or_raman_intensity",
    spectral_unit="cm-1",
    doppler_doublet_expected=False,
    freq_min_mhz=300000.0,
    freq_max_mhz=150000000.0,
    valon_lo_min_mhz=None,
    valon_lo_max_mhz=None,
    default_if_offset_mhz=0.0,
    step_size_min_mhz=1500.0,
    step_size_max_mhz=120000.0,
    default_step_size_mhz=15000.0,
    default_fft_rbw_khz=15000000.0,
    default_scope_fft_span_mhz=None,
    zaber_pos_min_mm=None,
    zaber_pos_max_mm=None,
    zaber_speed_min_mm_s=None,
    zaber_speed_max_mm_s=None,
    default_zaber_speed_mm_s=None,
    rf_level_min_dbm=None,
    rf_level_max_dbm=None,
    rf_level_soft_warn_dbm=None,
    cavity_modes=(),
    scan_apodization_windows=("Happ-Genzel", "Blackman-Harris", "Boxcar"),
    control_apodization_windows=(),
    recommended_targeted_window_min_mhz=300000.0,
    recommended_targeted_window_max_mhz=3000000.0,
    recommended_survey_window_max_mhz=120000000.0,
    zaber_retune_overhead_s=0.0,
)

BUILTIN_INSTRUMENT_PROFILES: Dict[str, InstrumentProfile] = {
    GRULAB_SURF_PROFILE.profile_id: GRULAB_SURF_PROFILE,
    GENERIC_CP_FTMW_PROFILE.profile_id: GENERIC_CP_FTMW_PROFILE,
    GENERIC_IR_RAMAN_PROFILE.profile_id: GENERIC_IR_RAMAN_PROFILE,
}

_INSTRUMENT_PROFILE_REGISTRY: Dict[str, InstrumentProfile] = dict(BUILTIN_INSTRUMENT_PROFILES)


def get_instrument_profile(profile_id: Optional[str] = None) -> InstrumentProfile:
    """Return a registered instrument profile, defaulting to ``grulab_surf``."""
    target = DEFAULT_INSTRUMENT_PROFILE if profile_id is None else profile_id
    if target not in _INSTRUMENT_PROFILE_REGISTRY:
        valid = ", ".join(_INSTRUMENT_PROFILE_REGISTRY.keys())
        raise KeyError(f"Unknown instrument profile {target!r}. Registered profiles: {valid}")
    return _INSTRUMENT_PROFILE_REGISTRY[target]


def list_instrument_profiles() -> List[str]:
    """Return registered instrument profile identifiers in registration order."""
    return list(_INSTRUMENT_PROFILE_REGISTRY.keys())


def register_instrument_profile(
    profile: InstrumentProfile,
    *,
    overwrite: bool = False,
) -> InstrumentProfile:
    """Register an additional (or replacement) frozen ``InstrumentProfile``."""
    if not isinstance(profile, InstrumentProfile):
        raise TypeError("profile must be an InstrumentProfile instance.")
    if profile.profile_id in _INSTRUMENT_PROFILE_REGISTRY and not overwrite:
        raise ValueError(
            f"Instrument profile {profile.profile_id!r} is already registered; "
            "pass overwrite=True to replace it."
        )
    _INSTRUMENT_PROFILE_REGISTRY[profile.profile_id] = profile
    return profile


def surf_compatibility_metadata() -> Dict[str, str]:
    """Return canonical upstream SURF compatibility metadata for provider discovery."""
    return {
        "repo": SURF_UPSTREAM_REPO_SLUG,
        "verified_commit": SURF_VERIFIED_COMMIT_SHA,
        "version_spec": SURF_SUPPORTED_VERSION_SPEC,
    }


__all__ = [
    "BUILTIN_INSTRUMENT_PROFILES",
    "DEFAULT_INSTRUMENT_PROFILE",
    "GENERIC_CP_FTMW_PROFILE",
    "GENERIC_IR_RAMAN_PROFILE",
    "GRULAB_SURF_PROFILE",
    "InstrumentProfile",
    "SURF_SUPPORTED_VERSION_SPEC",
    "SURF_UPSTREAM_REPO",
    "SURF_UPSTREAM_REPO_SLUG",
    "SURF_VERIFIED_COMMIT_SHA",
    "get_instrument_profile",
    "list_instrument_profiles",
    "register_instrument_profile",
    "surf_compatibility_metadata",
]
