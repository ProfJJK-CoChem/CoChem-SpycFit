"""Zero-mock verification suite for SPYC-SURF-01 instrument profiles and SURF schemas.

Verifies:
1. ``spyfy_provider()`` publishes ``default_instrument_profile == "grulab_surf"``,
   all registered ``instrument_profiles``, and verified ``surf_compatibility`` metadata.
2. Frozen ``InstrumentProfile`` registry defaults, secondary profiles, and custom registration.
3. Exact round-trip parse -> model -> ``.to_surf_toml()`` -> ``tomllib.loads`` of authentic
   ``GruLabRotSpec/SURF@0486b877`` ``default_config.toml`` and ``default_settings.toml``
   against upstream ``Config`` / ``Settings`` field contracts.
4. Strict hardware safety bounds (frequency envelope, equal start/end, step size, Zaber
   travel/speed, Valon LO minimum, strict int RF power + soft warning, path traversal).
5. ``SurfFrequencyScanSettings.to_surf_toml()`` omits ``zaber_pos`` when ``None`` and
   emits it when set, plus ``SurfCavitySearchSettings``, ``SurfOverrideRecord``, and
   strict RFC-8259 JSON ``SurfScanManifest``.
6. Strict UTF-8 LF line endings and zero-pass AST compliance on all SPYC-SURF-01 files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from ci_tools.anti_spoof_linter import check_script
from cochem_spyfy.instruments import (
    DEFAULT_INSTRUMENT_PROFILE,
    GENERIC_CP_FTMW_PROFILE,
    GENERIC_IR_RAMAN_PROFILE,
    GRULAB_SURF_PROFILE,
    SURF_SUPPORTED_VERSION_SPEC,
    SURF_UPSTREAM_REPO,
    SURF_UPSTREAM_REPO_SLUG,
    SURF_VERIFIED_COMMIT_SHA,
    AwgControllerConfig,
    DelayGeneratorControllerConfig,
    DigitizerSettings,
    ExperimentSettings,
    InstrumentProfile,
    OscilloscopeControllerConfig,
    OscilloscopeMathConfig,
    OutputSettings,
    ScanParameters,
    SurfAnalysisSettings,
    SurfCavitySearchSettings,
    SurfCavitySearchType,
    SurfConfig,
    SurfFrequencyScanSettings,
    SurfLoggingConfig,
    SurfOutputConfig,
    SurfOverrideRecord,
    SurfScanManifest,
    SurfScopePreset,
    SurfSettings,
    TimingSettings,
    ValonControllerConfig,
    ZaberControllerConfig,
    check_rf_level_soft_warning,
    dumps_surf_toml,
    get_instrument_profile,
    list_instrument_profiles,
    loads_surf_toml,
    register_instrument_profile,
)
from cochem_spyfy.instruments.profiles import _INSTRUMENT_PROFILE_REGISTRY
from cochem_spyfy.module_provider import spyfy_provider

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "surf"
DEFAULT_CONFIG_FIXTURE = FIXTURES_DIR / "default_config.toml"
DEFAULT_SETTINGS_FIXTURE = FIXTURES_DIR / "default_settings.toml"


def _build_valid_scan_settings(
    *,
    zaber_pos: float | None = None,
    start_freq: float = 10897.0,
    end_freq: float = 10902.0,
    step_size: float = 0.5,
    filename: str = "HCOOH_DME_10897.000_10902.000MHz_0.500MHz_100acqs_2026-11-24",
    location: str = "C:\\SURF_Data\\HCOOH_DME",
) -> SurfFrequencyScanSettings:
    """Construct a valid HCOOH-DME SURF frequency scan settings instance."""
    return SurfFrequencyScanSettings(
        experiment=ExperimentSettings(
            sample_name="HCOOH_DME",  # upstream SURF schema field (anti-spoof: authentic hardware key)
            sample_temp=22.0,  # upstream SURF schema field (anti-spoof: authentic hardware key)
            gas_name="Ne",
            gas_width=600.0,
            backing_pressure=15.0,
            chamber_pressure="1.0e-6 torr",
            mw_width=1.0,
        ),
        scan_parameters=ScanParameters(
            start_freq=start_freq,
            end_freq=end_freq,
            step_size=step_size,
            zaber_speed=0.003,
            zaber_pos=zaber_pos,
            acq_num=100,
            cavity_type="Continuous",
        ),
        digitizer_settings=DigitizerSettings(
            resolution=100,
            acq_window=14,
            apodization="Hanning",
        ),
        timing_settings=TimingSettings(
            rep_rate=5,
            valve_mw_delay=1300,
            spdt_width=10.0,
            acq_delay=22.5,
        ),
        output_settings=OutputSettings(
            filename=filename,
            location=location,
        ),
    )


def test_spyfy_provider_publishes_grulab_surf_metadata() -> None:
    """spyfy_provider() publishes grulab_surf default, profile list, and SURF compatibility."""
    meta = spyfy_provider()
    encoded = json.dumps(meta, allow_nan=False)
    assert isinstance(encoded, str) and len(encoded) > 0

    assert meta["default_instrument_profile"] == "grulab_surf"
    assert meta["instrument_profiles"] == [
        "grulab_surf",
        "generic_cp_ftmw",
        "generic_ir_raman",
    ]
    assert meta["surf_compatibility"] == {
        "repo": "GruLabRotSpec/SURF",
        "verified_commit": "0486b877d2a38f5bbe50be3cb333d34e27c7c22d",
        "version_spec": ">=0.1.0,<0.2.0",
    }
    # Preserve existing cochem.module-provider/1 operations contract verbatim
    assert meta["operations"] == ["predict_spectrum", "fit_pickett", "parse_pickett", "audit_fit"]


def test_instrument_profile_registry_and_grulab_surf_envelope() -> None:
    """Verify frozen grulab_surf defaults, secondary profiles, and custom registration."""
    assert DEFAULT_INSTRUMENT_PROFILE == "grulab_surf"
    assert list_instrument_profiles() == [
        "grulab_surf",
        "generic_cp_ftmw",
        "generic_ir_raman",
    ]

    surf = get_instrument_profile()
    assert surf == GRULAB_SURF_PROFILE
    assert surf.profile_id == "grulab_surf"
    assert surf.upstream_repo == SURF_UPSTREAM_REPO
    assert surf.verified_commit_sha == SURF_VERIFIED_COMMIT_SHA
    assert surf.supported_surf_version == SURF_SUPPORTED_VERSION_SPEC
    assert SURF_UPSTREAM_REPO_SLUG == "GruLabRotSpec/SURF"
    assert surf.technique == "cavity_ftmw_fabry_perot"
    assert surf.spectral_observable == "averaged_fft_magnitude_volts"
    assert surf.doppler_doublet_expected is True
    assert surf.freq_min_mhz == 8000.0
    assert surf.freq_max_mhz == 18000.0
    assert surf.valon_lo_min_mhz == 7970.0
    assert surf.valon_lo_max_mhz == 18000.0
    assert surf.default_if_offset_mhz == 30.0
    assert surf.step_size_min_mhz == 0.001
    assert surf.step_size_max_mhz == 1.000
    assert surf.default_step_size_mhz == 0.500
    assert surf.default_fft_rbw_khz == 100.0
    assert surf.instrumental_resolution_mhz == pytest.approx(0.100)
    assert surf.default_scope_fft_span_mhz == 20.0
    assert surf.zaber_pos_min_mm == 0.0
    assert surf.zaber_pos_max_mm == 50.0
    assert surf.zaber_speed_min_mm_s == 0.001
    assert surf.zaber_speed_max_mm_s == 2.000
    assert surf.default_zaber_speed_mm_s == 0.003
    assert surf.rf_level_min_dbm == -20
    assert surf.rf_level_max_dbm == 20
    assert surf.rf_level_soft_warn_dbm == 10
    assert surf.cavity_modes == ("Continuous", "Pulsed")
    assert surf.scan_apodization_windows == ("Hanning", "Hamming", "Blackman")
    assert surf.control_apodization_windows == ("Rectangular", "Hamming", "Hanning", "Blackman")
    assert surf.recommended_targeted_window_mhz == (2.0, 20.0)
    assert surf.recommended_survey_window_max_mhz == 500.0
    assert surf.zaber_retune_overhead_s == 14.0

    # Frozen immutability and extra="forbid"
    with pytest.raises(ValidationError):
        surf.freq_max_mhz = 20000.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        InstrumentProfile(**{**surf.model_dump(), "unknown_extra_key": 1})

    # Secondary profiles
    assert get_instrument_profile("generic_cp_ftmw") == GENERIC_CP_FTMW_PROFILE
    assert get_instrument_profile("generic_ir_raman") == GENERIC_IR_RAMAN_PROFILE
    with pytest.raises(KeyError, match="nonexistent_instrument"):
        get_instrument_profile("nonexistent_instrument")

    # Custom registration + duplicate rejection + cleanup
    custom = InstrumentProfile(
        profile_id="lab_k_band_cavity",
        display_name="K-Band Cavity FTMW (18-26.5 GHz)",
        technique="cavity_ftmw_fabry_perot",
        spectral_observable="averaged_fft_magnitude_volts",
        doppler_doublet_expected=True,
        freq_min_mhz=18000.0,
        freq_max_mhz=26500.0,
        step_size_min_mhz=0.001,
        step_size_max_mhz=1.0,
        default_step_size_mhz=0.5,
        default_fft_rbw_khz=100.0,
        recommended_targeted_window_min_mhz=2.0,
        recommended_targeted_window_max_mhz=20.0,
        recommended_survey_window_max_mhz=500.0,
    )
    try:
        registered = register_instrument_profile(custom)
        assert registered == custom
        assert get_instrument_profile("lab_k_band_cavity") == custom
        with pytest.raises(ValueError, match="already registered"):
            register_instrument_profile(custom)
        register_instrument_profile(custom, overwrite=True)
    finally:
        _INSTRUMENT_PROFILE_REGISTRY.pop("lab_k_band_cavity", None)


def test_authentic_surf_default_config_roundtrip_and_contract() -> None:
    """Round-trip authentic SURF default_config.toml and verify upstream Config contract."""
    raw_bytes = DEFAULT_CONFIG_FIXTURE.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")

    cfg = SurfConfig.from_surf_toml_file(DEFAULT_CONFIG_FIXTURE)
    assert type(cfg.valon_controller.rf_level) is int
    assert cfg.valon_controller.rf_level == 10
    assert cfg.valon_controller.rf_level_warnings() == []
    assert cfg.valon_controller.freq == pytest.approx(8000.0)
    assert type(cfg.awg_controller.awg_freq) is int
    assert cfg.awg_controller.awg_freq == 30
    assert cfg.oscilloscope_controller.math4.resolution == pytest.approx(100.0)
    assert cfg.oscilloscope_controller.math4.gate_position == pytest.approx(22.5)
    assert cfg.delay_generator_controller.trigger_rate == pytest.approx(15.0)
    assert cfg.delay_generator_controller.trigger_state == "INT"

    dumped = cfg.to_surf_toml()
    assert "\r" not in dumped
    assert dumped.endswith("\n")
    assert "rf_level = 10\n" in dumped
    assert "awg_freq = 30\n" in dumped

    parsed_dict = tomllib.loads(dumped)
    # Verify exact upstream SURF Config top-level and nested key contract (config.py:8-61)
    assert set(parsed_dict.keys()) == {
        "valon_controller",
        "zaber_controller",
        "awg_controller",
        "oscilloscope_controller",
        "delay_generator_controller",
    }
    assert set(parsed_dict["valon_controller"].keys()) == {
        "rf_output",
        "rf_level",
        "valon_port",
        "synth_power",
        "ref_source",
        "ref_freq",
        "freq",
    }
    assert type(parsed_dict["valon_controller"]["rf_level"]) is int
    assert set(parsed_dict["zaber_controller"].keys()) == {
        "zaber_scanning_speed",
        "zaber_moving_speed",
        "zaber_step_size",
        "zaber_port",
    }
    assert set(parsed_dict["awg_controller"].keys()) == {
        "awg_status",
        "awg_freq",
        "awg_run_mode",
        "awg_ch_1_output",
        "awg_ch_2_output",
    }
    assert type(parsed_dict["awg_controller"]["awg_freq"]) is int
    assert set(parsed_dict["oscilloscope_controller"].keys()) == {
        "channel",
        "acq_rate",
        "sample_rate",
        "math_averages",
        "visa_address",
        "math3",
        "math4",
        "math3_cont",
    }
    for math_key in ("math3", "math4", "math3_cont"):
        assert set(parsed_dict["oscilloscope_controller"][math_key].keys()) == {
            "window",
            "resolution",
            "gate_position",
            "scale",
        }
    assert set(parsed_dict["delay_generator_controller"].keys()) == {
        "trigger_rate",
        "trigger_state",
    }

    reloaded = SurfConfig.from_surf_toml(dumped)
    assert reloaded == cfg


def test_authentic_surf_default_settings_roundtrip_and_contract() -> None:
    """Round-trip authentic SURF default_settings.toml and verify upstream Settings contract."""
    raw_bytes = DEFAULT_SETTINGS_FIXTURE.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")

    settings = SurfSettings.from_surf_toml_file(DEFAULT_SETTINGS_FIXTURE)
    assert settings.theme == "auto"
    assert settings.output.filename == "my_scan"
    assert settings.logging.enabled is True
    assert set(settings.scope_preset.presets.keys()) == {"cavity", "scan_peak", "pulse"}
    assert settings.scope_preset.presets["scan_peak"].path == "2cavity_scan_peak001.set"
    assert settings.analysis.show_points is True
    assert settings.analysis.show_grid is True
    assert settings.analysis.show_crosshair is True

    dumped = settings.to_surf_toml()
    assert "\r" not in dumped
    assert dumped.endswith("\n")

    parsed_dict = tomllib.loads(dumped)
    assert set(parsed_dict.keys()) == {
        "output",
        "logging",
        "scope_preset",
        "theme",
        "analysis",
    }
    assert set(parsed_dict["output"].keys()) == {"location", "filename"}
    assert set(parsed_dict["logging"].keys()) == {"enabled", "location"}
    assert set(parsed_dict["scope_preset"].keys()) == {"root_path", "presets"}
    assert set(parsed_dict["scope_preset"]["presets"].keys()) == {"cavity", "scan_peak", "pulse"}
    assert set(parsed_dict["analysis"].keys()) == {"show_points", "show_grid", "show_crosshair"}

    reloaded = SurfSettings.from_surf_toml(dumped)
    assert reloaded == settings

    # Empty presets dict also round-trips cleanly
    empty_presets = SurfSettings(
        output=SurfOutputConfig(location="C:\\SURF_Data", filename="run_0"),
        logging=SurfLoggingConfig(enabled=False, location="./logs/"),
        scope_preset=SurfScopePreset(root_path="C:\\Setups", presets={}),
        theme="dark",
        analysis=SurfAnalysisSettings(),
    )
    assert SurfSettings.from_surf_toml(empty_presets.to_surf_toml()) == empty_presets


def test_hardware_bounds_reject_invalid_scan_and_config_parameters() -> None:
    """Verify strict hardware safety bounds raise ValidationError on every violation."""
    base_cfg = SurfConfig.from_surf_toml_file(DEFAULT_CONFIG_FIXTURE)

    # Frequency bounds: 8000.0 <= start_freq, end_freq <= 18000.0 and start_freq != end_freq
    for bad_start, bad_end in [
        (7999.9, 10000.0),
        (18000.1, 10000.0),
        (10000.0, 7999.9),
        (10000.0, 18000.1),
        (10899.0, 10899.0),
    ]:
        with pytest.raises(ValidationError):
            _build_valid_scan_settings(start_freq=bad_start, end_freq=bad_end)

    # Step size bounds: 0.001 <= step_size <= 1.000
    for bad_step in (0.0, -0.25, 0.0005, 1.001):
        with pytest.raises(ValidationError):
            _build_valid_scan_settings(step_size=bad_step)

    # Zaber position bounds: None or 0.0 <= zaber_pos <= 50.0
    for bad_pos in (-0.001, 50.001, float("nan")):
        with pytest.raises(ValidationError):
            _build_valid_scan_settings(zaber_pos=bad_pos)

    # Zaber speed, acq_num, cavity_type
    with pytest.raises(ValidationError):
        ScanParameters(
            start_freq=10000.0,
            end_freq=10010.0,
            step_size=0.5,
            zaber_speed=2.001,
            zaber_pos=None,
            acq_num=100,
            cavity_type="Continuous",
        )
    with pytest.raises(ValidationError):
        ScanParameters(
            start_freq=10000.0,
            end_freq=10010.0,
            step_size=0.5,
            zaber_speed=0.003,
            zaber_pos=None,
            acq_num=0,
            cavity_type="Continuous",
        )
    with pytest.raises(ValidationError):
        ScanParameters(
            start_freq=10000.0,
            end_freq=10010.0,
            step_size=0.5,
            zaber_speed=0.003,
            zaber_pos=None,
            acq_num=100,
            cavity_type="InvalidMode",  # type: ignore[arg-type]
        )

    # Digitizer and timing bounds
    with pytest.raises(ValidationError):
        DigitizerSettings(resolution=16385, acq_window=14, apodization="Hanning")
    with pytest.raises(ValidationError):
        DigitizerSettings(resolution=100, acq_window=14, apodization="Rectangular")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TimingSettings(rep_rate=1001, valve_mw_delay=1300, spdt_width=10.0, acq_delay=22.5)
    with pytest.raises(ValidationError):
        TimingSettings(rep_rate=5, valve_mw_delay=10001, spdt_width=10.0, acq_delay=22.5)
    with pytest.raises(ValidationError):
        TimingSettings(rep_rate=5, valve_mw_delay=1300, spdt_width=0.5, acq_delay=22.5)
    with pytest.raises(ValidationError):
        TimingSettings(rep_rate=5, valve_mw_delay=1300, spdt_width=10.0, acq_delay=10000.1)

    # Valon rf_level bounds [-20, 20], strict int rejection, and soft warning > 10 dBm
    valon_base = base_cfg.valon_controller.model_dump()
    for bad_rf in (21, -21, 10.5, "10", True):
        with pytest.raises(ValidationError):
            ValonControllerConfig(**{**valon_base, "rf_level": bad_rf})

    high_rf_valon = ValonControllerConfig(**{**valon_base, "rf_level": 15})
    assert high_rf_valon.rf_level == 15
    warnings = high_rf_valon.rf_level_warnings()
    assert len(warnings) == 1
    assert "+10 dBm" in warnings[0]
    assert check_rf_level_soft_warning(10) is None
    assert check_rf_level_soft_warning(12) is not None

    # Valon LO lower/upper bounds (< 7970.0 MHz or > 18000.0 MHz)
    for bad_lo in (7969.9, 18000.1, float("inf")):
        with pytest.raises(ValidationError):
            ValonControllerConfig(**{**valon_base, "freq": bad_lo})

    # Valon LO helper from scan_freq - awg_freq
    assert base_cfg.validate_lo_for_scan_freq(8000.0) == pytest.approx(7970.0)
    with pytest.raises(ValueError, match="7970.0"):
        base_cfg.validate_lo_for_scan_freq(7999.9)

    # AWG strict int check
    awg_base = base_cfg.awg_controller.model_dump()
    for bad_awg_freq in (30.5, 0, True, "30"):
        with pytest.raises(ValidationError):
            AwgControllerConfig(**{**awg_base, "awg_freq": bad_awg_freq})

    # Path traversal guards on output filename and location
    for bad_filename in (
        "../escape",
        "..\\escape",
        "subdir/run_0",
        "subdir\\run_0",
        "..",
        ".",
        "C:evil_scan",
        "bad\x00name",
    ):
        with pytest.raises(ValidationError):
            _build_valid_scan_settings(filename=bad_filename)
    with pytest.raises(ValidationError):
        _build_valid_scan_settings(location="C:\\SURF_Data\\..\\Windows\\System32")


def test_frequency_scan_settings_toml_zaber_pos_omission_and_inclusion() -> None:
    """to_surf_toml() omits zaber_pos when None and serializes it when explicitly set."""
    scan_none = _build_valid_scan_settings(zaber_pos=None)
    toml_none = scan_none.to_surf_toml()
    assert "zaber_pos" not in toml_none
    parsed_none = tomllib.loads(toml_none)
    assert "zaber_pos" not in parsed_none["scan_parameters"]
    assert SurfFrequencyScanSettings.from_surf_toml(toml_none) == scan_none

    scan_set = _build_valid_scan_settings(zaber_pos=24.375)
    toml_set = scan_set.to_surf_toml()
    assert "zaber_pos = 24.375\n" in toml_set
    parsed_set = tomllib.loads(toml_set)
    assert parsed_set["scan_parameters"]["zaber_pos"] == pytest.approx(24.375)
    assert SurfFrequencyScanSettings.from_surf_toml(toml_set) == scan_set

    # Runtime estimate matches SURF/spectrometer.py:168-181 exactly:
    # iterations = ceil(|10902 - 10897| / 0.5) + 1 = 11
    # time_delay = 100 / 5 = 20.0 s -> 11 * (20 + 14) / 60 = 374 / 60 min
    expected_minutes = (11 * (20.0 + 14.0)) / 60.0
    assert scan_none.estimated_runtime_minutes() == pytest.approx(expected_minutes)


def test_cavity_search_override_and_manifest_strict_json() -> None:
    """Verify SurfCavitySearchSettings, SurfOverrideRecord, and strict JSON SurfScanManifest."""
    search = SurfCavitySearchSettings(
        cavity_type=SurfCavitySearchType.CONTINUOUS,
        freq=10899.240,
        step_size=1.0,
        zaber_speed=0.01,
    )
    search_toml = search.to_surf_toml()
    parsed_search = loads_surf_toml(search_toml)
    assert parsed_search["cavity_type"] == "continuous"
    assert SurfCavitySearchSettings.from_surf_toml(search_toml) == search

    with pytest.raises(ValidationError):
        SurfCavitySearchSettings(
            cavity_type=SurfCavitySearchType.PULSED,
            freq=7999.0,
            step_size=1.0,
            zaber_speed=0.01,
        )

    override = SurfOverrideRecord(
        field="scan_parameters.acq_num",
        proposed_value=100,
        user_value=250,
        edited_by="spectroscopist",
        edited_at_utc="2026-11-24T12:00:00Z",
    )
    with pytest.raises(ValidationError):
        SurfOverrideRecord(
            field="scan_parameters.step_size",
            proposed_value=math.nan,
            user_value=0.5,
            edited_by="spectroscopist",
            edited_at_utc="2026-11-24T12:00:00Z",
        )

    cfg_toml = SurfConfig.from_surf_toml_file(DEFAULT_CONFIG_FIXTURE).to_surf_toml()
    scan_toml = _build_valid_scan_settings().to_surf_toml()
    manifest = SurfScanManifest(
        created_at_utc="2026-11-24T12:05:00Z",
        fit_state_id="fit-state-001",
        spcat_catalog_sha256=hashlib.sha256(b"spcat-catalog").hexdigest(),
        control_config_filename="region_01_control_config.toml",
        control_config_sha256=hashlib.sha256(cfg_toml.encode("utf-8")).hexdigest(),
        scan_settings_filename="region_01_scan_settings.toml",
        scan_settings_sha256=hashlib.sha256(scan_toml.encode("utf-8")).hexdigest(),
        estimated_runtime_minutes=6.2333,
        recommendation_rank=1,
        recommendation_score=4.82,
        target_constants=("A", "B", "C"),
        gp_hyperparameters={"length_scale": 1.25, "noise_level": 1e-4},
        overrides=(override,),
        warnings=(),
    )
    strict_json = manifest.to_strict_json()
    assert SurfScanManifest.from_strict_json(strict_json) == manifest

    # Strict JSON parser rejects NaN literals
    bad_json = strict_json.replace("4.82", "NaN")
    with pytest.raises(ValueError, match="Non-finite JSON constant"):
        SurfScanManifest.from_strict_json(bad_json)

    # Pure-Python TOML writer rejects None and NaN
    with pytest.raises(ValueError, match="None"):
        dumps_surf_toml({"section": {"missing": None}})
    with pytest.raises(ValueError, match="Non-finite float"):
        dumps_surf_toml({"section": {"val": float("nan")}})


def test_owned_files_utf8_lf_zero_pass_and_anti_spoof_clean() -> None:
    """Verify strict UTF-8 LF, zero pass statements, and strict anti-spoof lint on all owned files."""
    owned_python_files = [
        REPO_ROOT / "src" / "cochem_spyfy" / "instruments" / "__init__.py",
        REPO_ROOT / "src" / "cochem_spyfy" / "instruments" / "profiles.py",
        REPO_ROOT / "src" / "cochem_spyfy" / "instruments" / "surf_schemas.py",
        REPO_ROOT / "src" / "cochem_spyfy" / "module_provider.py",
        Path(__file__).resolve(),
    ]
    owned_toml_files = [
        DEFAULT_CONFIG_FIXTURE,
        DEFAULT_SETTINGS_FIXTURE,
    ]

    for path in [*owned_python_files, *owned_toml_files]:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM found in {path}"
        assert b"\r\n" not in raw, f"CRLF found in {path}"
        assert b"\r" not in raw, f"CR found in {path}"
        assert raw.endswith(b"\n"), f"Missing trailing LF in {path}"

    for py_path in owned_python_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert pass_nodes == [], f"Found pass statements in {py_path}"
        violations = check_script(py_path, strict_mode=True)
        assert violations == [], f"Anti-spoof strict violations in {py_path}: {violations}"
