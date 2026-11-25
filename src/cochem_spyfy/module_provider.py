"""CoChem-BASE compatible module provider for CoChem-SPYFY.

Publishes the ``cochem.modules`` entry point ``spyfy`` conforming to
``cochem.module-provider/1`` and ``cochem.module-handoff/1``. Discovery is
side-effect free and observes actual runtime availability of ``spfit`` and
``spcat`` executables without fabricating results or falling back to
surrogate models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_OPERATIONS = (
    "predict_spectrum",
    "fit_pickett",
    "parse_pickett",
    "audit_fit",
)
_SURF_OPERATIONS = (
    "plan_surf_scan",
    "ingest_surf_run",
)
_HANDOFF_OPERATIONS = _OPERATIONS + _SURF_OPERATIONS


def _load_submodule(dotted_name: str, relative_path: str) -> Any:
    """Import a lightweight submodule without forcing optional heavy package deps."""
    try:
        return __import__(dotted_name, fromlist=["*"])
    except ModuleNotFoundError:
        existing = sys.modules.get(dotted_name)
        if existing is not None:
            return existing
        target = _MODULE_ROOT / relative_path
        if not target.is_file():
            raise
        spec = importlib.util.spec_from_file_location(dotted_name, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to build module spec for {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted_name] = module
        spec.loader.exec_module(module)
        return module


def _load_pickett_output() -> Any:
    return _load_submodule(
        "cochem_spyfy.serialization.cochem_spyfy_pickett_output",
        "serialization/cochem_spyfy_pickett_output.py",
    )


def _load_spcat_wrapper() -> Any:
    # Ensure sibling parser submodule is registered before wrapper relative import
    _load_pickett_output()
    return _load_submodule(
        "cochem_spyfy.serialization.cochem_spyfy_spcat_wrapper",
        "serialization/cochem_spyfy_spcat_wrapper.py",
    )


def _load_auditor() -> Any:
    return _load_submodule(
        "cochem_spyfy.physics_core.cochem_spyfy_auditor",
        "physics_core/cochem_spyfy_auditor.py",
    )


def _load_instrument_profiles() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.profiles",
        "instruments/profiles.py",
    )


def _load_surf_planner() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.surf_planner",
        "instruments/surf_planner.py",
    )


def _load_surf_exporter() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.surf_exporter",
        "instruments/surf_exporter.py",
    )


def _load_surf_ingest() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.surf_ingest",
        "instruments/surf_ingest.py",
    )


def _load_surf_watcher() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.surf_watcher",
        "instruments/surf_watcher.py",
    )


def _load_surf_schemas() -> Any:
    return _load_submodule(
        "cochem_spyfy.instruments.surf_schemas",
        "instruments/surf_schemas.py",
    )


def _load_ml_core() -> Any:
    return _load_submodule(
        "cochem_spyfy.physics_core.cochem_spyfy_ml_core",
        "physics_core/cochem_spyfy_ml_core.py",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest() -> str:
    return _sha256_path(Path(__file__).resolve())


def _package_version() -> str:
    try:
        return importlib.metadata.version("CoChem-SPYFY")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _probe_pickett_binary(name: str) -> Dict[str, Any]:
    """Observe whether a real Pickett executable (spfit/spcat) is installed."""
    wrapper = _load_spcat_wrapper()
    try:
        exe = wrapper.SpfitSandbox._resolve_binary(name)
    except FileNotFoundError as exc:
        return {"available": False, "program": name, "reason": str(exc)}
    resolved = exe.resolve(strict=True)
    return {
        "available": True,
        "program": name,
        "version": "calpgm",
        "executable_path": str(resolved),
        "sha256": _sha256_path(resolved),
    }


def predict_spectrum(
    var_text: str,
    int_text: str,
    *,
    stem: str = "cochem_spcat",
    spcat_binary: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Run genuine Pickett SPCAT on complete ``.var`` and ``.int`` decks and parse ``.cat``."""
    wrapper = _load_spcat_wrapper()
    parser = _load_pickett_output()
    exe = Path(spcat_binary) if spcat_binary is not None else wrapper.SpfitSandbox._resolve_binary("spcat")
    sandbox = wrapper.SpfitSandbox(
        scratch_root=scratch_root,
        timeout_seconds=timeout_seconds,
        spfit_binary=exe,
        spcat_binary=exe,
    )
    record = sandbox.run_spcat_files(var_text=var_text, int_text=int_text, stem=stem)
    cat_text = record.outputs[f"{stem}.cat"]
    transitions = parser.parse_spcat_cat(cat_text)
    return {
        "program": record.program,
        "executable": record.executable,
        "executable_sha256": record.executable_sha256,
        "exit_code": record.exit_code,
        "stem": stem,
        "n_transitions": len(transitions),
        "transitions": [
            {
                "frequency_mhz": t.frequency_mhz,
                "uncertainty_mhz": t.uncertainty_mhz,
                "log_intensity": t.log_intensity,
                "degrees_of_freedom": t.degrees_of_freedom,
                "lower_energy_cm1": t.lower_energy_cm1,
                "upper_degeneracy": t.upper_degeneracy,
                "species_tag": t.species_tag,
                "quantum_format": t.quantum_format,
                "upper_quanta": list(t.upper_quanta),
                "lower_quanta": list(t.lower_quanta),
            }
            for t in transitions
        ],
        "cat_text": cat_text,
    }


def fit_pickett(
    par_text: str,
    lin_text: str,
    *,
    stem: str = "cochem_spfit",
    spfit_binary: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Run genuine Pickett SPFIT on complete ``.par`` and ``.lin`` decks and parse ``.fit``/``.var``."""
    wrapper = _load_spcat_wrapper()
    exe = Path(spfit_binary) if spfit_binary is not None else wrapper.SpfitSandbox._resolve_binary("spfit")
    sandbox = wrapper.SpfitSandbox(
        scratch_root=scratch_root,
        timeout_seconds=timeout_seconds,
        spfit_binary=exe,
        spcat_binary=exe,
    )
    run = sandbox.run_spfit_files(par_text=par_text, lin_text=lin_text, stem=stem)
    final = run.report.final
    constants: Dict[str, float] = {}
    uncertainties: Dict[str, float] = {}
    for param in run.parameters:
        key = wrapper.PARAM_ID_TO_NAME_MAP.get(param.code, str(param.code))
        constants[key] = param.value
        uncertainties[key] = param.error
    return {
        "program": run.record.program,
        "executable": run.record.executable,
        "executable_sha256": run.record.executable_sha256,
        "exit_code": run.record.exit_code,
        "stem": stem,
        "title": run.report.title,
        "iterations": len(run.report.iterations),
        "microwave_avg_mhz": final.microwave_avg_mhz,
        "microwave_rms_mhz": final.microwave_rms_mhz,
        "rms_error_old": final.rms_error_old,
        "rms_error_new": final.rms_error_new,
        "n_residuals": len(final.residuals),
        "bad_lines": list(run.report.bad_lines),
        "constants": constants,
        "uncertainties": uncertainties,
        "parameters": [
            {
                "code": p.code,
                "name": wrapper.PARAM_ID_TO_NAME_MAP.get(p.code, str(p.code)),
                "value": p.value,
                "error": p.error,
                "label": p.label,
            }
            for p in run.parameters
        ],
        "var_text": run.record.outputs[f"{stem}.var"],
        "fit_text": run.record.outputs[f"{stem}.fit"],
    }


def parse_pickett(kind: str, text: str) -> Dict[str, Any]:
    """Parse genuine Pickett ``.fit``, ``.par``/``.var``, or ``.cat`` text."""
    parser = _load_pickett_output()
    wrapper = _load_spcat_wrapper()
    normalized = kind.strip().lower().lstrip(".")
    if normalized in ("par", "var"):
        params = parser.parse_pickett_parameters(text)
        return {
            "kind": normalized,
            "n_parameters": len(params),
            "parameters": [
                {
                    "code": p.code,
                    "name": wrapper.PARAM_ID_TO_NAME_MAP.get(p.code, str(p.code)),
                    "value": p.value,
                    "error": p.error,
                    "label": p.label,
                }
                for p in params
            ],
        }
    if normalized == "fit":
        report = parser.parse_spfit_fit(text)
        final = report.final
        return {
            "kind": "fit",
            "title": report.title,
            "iterations": len(report.iterations),
            "microwave_avg_mhz": final.microwave_avg_mhz,
            "microwave_rms_mhz": final.microwave_rms_mhz,
            "rms_error_old": final.rms_error_old,
            "rms_error_new": final.rms_error_new,
            "n_residuals": len(final.residuals),
            "bad_lines": list(report.bad_lines),
            "missing": list(report.missing),
            "parameters": [
                {
                    "index": p.index,
                    "code": p.code,
                    "name": wrapper.PARAM_ID_TO_NAME_MAP.get(p.code, str(p.code)),
                    "label": p.label,
                    "value": p.value,
                    "est_error": p.est_error,
                    "change": p.change,
                }
                for p in final.parameters
            ],
        }
    if normalized == "cat":
        transitions = parser.parse_spcat_cat(text)
        return {
            "kind": "cat",
            "n_transitions": len(transitions),
            "transitions": [
                {
                    "frequency_mhz": t.frequency_mhz,
                    "uncertainty_mhz": t.uncertainty_mhz,
                    "log_intensity": t.log_intensity,
                    "degrees_of_freedom": t.degrees_of_freedom,
                    "lower_energy_cm1": t.lower_energy_cm1,
                    "upper_degeneracy": t.upper_degeneracy,
                    "species_tag": t.species_tag,
                    "quantum_format": t.quantum_format,
                    "upper_quanta": list(t.upper_quanta),
                    "lower_quanta": list(t.lower_quanta),
                }
                for t in transitions
            ],
        }
    raise ValueError(f"Unsupported Pickett artifact kind {kind!r}; expected 'fit', 'par', 'var', or 'cat'.")


def audit_fit(
    native_constants: Mapping[str, float],
    spfit_constants: Optional[Mapping[str, float]],
    *,
    tolerance_khz: Any,
    native_uncertainties: Optional[Mapping[str, float]] = None,
    spfit_uncertainties: Optional[Mapping[str, float]] = None,
    required_constants: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare native SpycFit constants against SPFIT constants under declared tolerances."""
    auditor_mod = _load_auditor()
    auditor = auditor_mod.ParityAuditor()
    report = auditor.audit(
        jax_constants=native_constants,
        spfit_constants=spfit_constants,
        jax_uncertainties=native_uncertainties,
        spfit_uncertainties=spfit_uncertainties,
        tolerance_khz=tolerance_khz,
        required_constants=required_constants,
    )
    return {
        "status": report.status.value,
        "is_within_tolerance": report.is_within_tolerance,
        "is_complete": report.is_complete,
        "has_divergence": report.has_divergence,
        "max_delta_khz": report.max_delta_khz,
        "divergent_constants": list(report.divergent_constants),
        "compared_constants": list(report.compared_constants),
        "missing_from_native": list(report.missing_from_native),
        "missing_from_spfit": list(report.missing_from_spfit),
        "tolerance_khz": dict(report.tolerance_khz),
        "contrast_table": dict(report.contrast_table),
    }


def _coerce_residual_predictions(raw_predictions: Sequence[Any]) -> list[Any]:
    """Convert mapping dicts or ``ResidualPrediction`` objects into ``ResidualPrediction`` instances."""
    ml_core = _load_ml_core()
    pred_cls = ml_core.ResidualPrediction
    prov_cls = ml_core.ProvenanceTag
    coerced: list[Any] = []
    for item in raw_predictions:
        if isinstance(item, pred_cls):
            coerced.append(item)
        elif isinstance(item, Mapping):
            raw_qn = item["quantum_numbers"]
            qn_tuple = tuple(int(q) for q in raw_qn)
            if len(qn_tuple) != 6:
                raise ValueError(f"Expected 6 quantum numbers per prediction, got {len(qn_tuple)}.")
            raw_prov = item.get("provenance", "TORQ")
            prov_tag = raw_prov if isinstance(raw_prov, prov_cls) else prov_cls(str(raw_prov))
            nu_calc = float(item["nu_calc_physical_mhz"])
            delta_ml = float(item.get("delta_nu_ml_shift_mhz", 0.0))
            nu_prop = float(item.get("nu_proposed_mhz", nu_calc + delta_ml))
            coerced.append(
                pred_cls(
                    quantum_numbers=qn_tuple,
                    nu_calc_physical_mhz=nu_calc,
                    delta_nu_ml_shift_mhz=delta_ml,
                    nu_proposed_mhz=nu_prop,
                    sigma_predict_mhz=float(item.get("sigma_predict_mhz", 0.1)),
                    probable_correctness=float(item.get("probable_correctness", 0.5)),
                    provenance=prov_tag,
                )
            )
        else:
            raise TypeError(
                f"Expected ResidualPrediction or mapping in predictions, got {type(item).__name__}."
            )
    return coerced


def plan_surf_scan(
    *,
    output_dir: str | Path,
    predictions: Optional[Sequence[Any]] = None,
    transition_frequencies_mhz: Optional[Sequence[float]] = None,
    information_gains: Optional[Sequence[float]] = None,
    start_freq_mhz: Optional[float] = None,
    end_freq_mhz: Optional[float] = None,
    mode: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    edited_by: str = "operator",
    instrument_profile: str = "grulab_surf",
    molecule_name: str = "HCOOH_DME",
    specimen_temp_c: float = 22.0,
    gas_name: str = "Ne",
    step_size_mhz: Optional[float] = None,
    acq_num: Optional[int] = None,
    rep_rate_hz: int = 5,
    max_chunk_bandwidth_mhz: float = 50.0,
    top_n: int = 3,
    proposal_index: int = 0,
    target_constants: Sequence[str] = (),
    resolvability_penalties: Optional[Sequence[float]] = None,
    line_strengths: Optional[Sequence[float]] = None,
    ml_uncertainties_mhz: Optional[Sequence[float]] = None,
    cavity_map_dir: Optional[str | Path] = None,
    cavity_map_csv: Optional[str | Path] = None,
    fit_state_id: Optional[str] = None,
    spcat_catalog_path: Optional[str | Path] = None,
    spcat_catalog_sha256: Optional[str] = None,
    gp_hyperparameters: Optional[Mapping[str, Any]] = None,
    region_stem: Optional[str] = None,
    scan_date: Optional[str] = None,
    surf_output_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan a GruLab SURF scan region, apply optional HITL overrides, and export validated TOMLs."""
    planner_mod = _load_surf_planner()
    exporter_mod = _load_surf_exporter()

    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    resolved_location = surf_output_location if surf_output_location is not None else str(out_path)

    cavity_map = None
    if cavity_map_dir is not None:
        cavity_map = planner_mod.SurfCavityMap.from_cavity_runs_dir(cavity_map_dir)
    elif cavity_map_csv is not None:
        cavity_map = planner_mod.SurfCavityMap.from_cavity_run_csv(cavity_map_csv)

    planner = planner_mod.SurfScanPlanner(
        instrument_profile=instrument_profile,
        cavity_map=cavity_map,
        molecule_name=molecule_name,
        specimen_temp_c=specimen_temp_c,
        gas_name=gas_name,
        output_location=resolved_location,
        default_rep_rate_hz=rep_rate_hz,
        scan_date=scan_date,
    )

    resolved_mode = mode
    if resolved_mode is None:
        if predictions is not None:
            resolved_mode = "targeted"
        elif transition_frequencies_mhz is not None:
            resolved_mode = "survey"
        elif start_freq_mhz is not None and end_freq_mhz is not None:
            resolved_mode = "explicit"
        else:
            raise ValueError(
                "plan_surf_scan requires predictions, transition_frequencies_mhz, "
                "or explicit start_freq_mhz and end_freq_mhz."
            )

    proposals: list[Any] = []
    selected_proposal = None

    if resolved_mode == "targeted":
        if predictions is None or information_gains is None:
            raise ValueError("Targeted SURF scan planning requires both predictions and information_gains.")
        coerced_preds = _coerce_residual_predictions(predictions)
        proposals = planner.plan_targeted_windows(
            coerced_preds,
            information_gains,
            resolvability_penalties=resolvability_penalties,
            line_strengths=line_strengths,
            target_constants=target_constants,
            step_size_mhz=0.250 if step_size_mhz is None else float(step_size_mhz),
            acq_num=acq_num,
            rep_rate_hz=rep_rate_hz,
            top_n=top_n,
            gp_hyperparameters=gp_hyperparameters,
        )
        if not proposals:
            raise ValueError("SurfScanPlanner produced zero valid in-band targeted scan proposals.")
        if not (0 <= proposal_index < len(proposals)):
            raise IndexError(f"proposal_index={proposal_index} out of range for {len(proposals)} proposals.")
        selected_proposal = proposals[proposal_index]
        override_table = planner.create_override_table(selected_proposal)
    elif resolved_mode == "survey":
        if transition_frequencies_mhz is None or information_gains is None:
            raise ValueError("Survey SURF scan planning requires transition_frequencies_mhz and information_gains.")
        proposals = planner.plan_survey_windows(
            transition_frequencies_mhz,
            information_gains,
            resolvability_penalties=resolvability_penalties,
            ml_uncertainties_mhz=ml_uncertainties_mhz,
            target_constants=target_constants,
            max_chunk_bandwidth_mhz=max_chunk_bandwidth_mhz,
            step_size_mhz=0.500 if step_size_mhz is None else float(step_size_mhz),
            acq_num=100 if acq_num is None else int(acq_num),
            rep_rate_hz=rep_rate_hz,
            top_n=top_n,
            gp_hyperparameters=gp_hyperparameters,
        )
        if not proposals:
            raise ValueError("SurfScanPlanner produced zero valid in-band survey scan proposals.")
        if not (0 <= proposal_index < len(proposals)):
            raise IndexError(f"proposal_index={proposal_index} out of range for {len(proposals)} proposals.")
        selected_proposal = proposals[proposal_index]
        override_table = planner.create_override_table(selected_proposal)
    elif resolved_mode == "explicit":
        if start_freq_mhz is None or end_freq_mhz is None:
            raise ValueError("Explicit SURF scan mode requires both start_freq_mhz and end_freq_mhz.")
        override_table = planner.create_default_override_table(
            start_freq_mhz=float(start_freq_mhz),
            end_freq_mhz=float(end_freq_mhz),
            step_size_mhz=0.500 if step_size_mhz is None else float(step_size_mhz),
            acq_num=100 if acq_num is None else int(acq_num),
            rep_rate_hz=rep_rate_hz,
        )
    else:
        raise ValueError(f"Unsupported plan_surf_scan mode {resolved_mode!r}; expected 'targeted', 'survey', or 'explicit'.")

    if overrides:
        override_table.apply_overrides(overrides, edited_by=edited_by)

    exporter = exporter_mod.SurfExporter(instrument_profile=instrument_profile)
    export_result = exporter.export_region(
        output_dir=out_path,
        override_table=override_table,
        region_stem=region_stem,
        fit_state_id=fit_state_id,
        spcat_catalog_path=spcat_catalog_path,
        spcat_catalog_sha256=spcat_catalog_sha256,
        gp_hyperparameters=gp_hyperparameters,
        recommendation=selected_proposal,
        target_constants=target_constants if target_constants else None,
    )
    duration_est = override_table.estimated_duration

    proposal_summary = None
    if selected_proposal is not None:
        proposal_summary = {
            "rank": selected_proposal.rank,
            "window_mode": selected_proposal.window_mode,
            "start_freq_mhz": selected_proposal.start_freq_mhz,
            "end_freq_mhz": selected_proposal.end_freq_mhz,
            "information_gain": selected_proposal.information_gain,
            "information_gain_per_minute": selected_proposal.information_gain_per_minute,
            "estimated_runtime_minutes": selected_proposal.estimated_runtime_minutes,
            "n_clean_lines": selected_proposal.n_clean_lines,
            "targeted_constants": list(selected_proposal.targeted_constants),
            "cavity_map_zaber_pos_mm": selected_proposal.cavity_map_zaber_pos_mm,
            "cavity_map_source_sha256": selected_proposal.cavity_map_source_sha256,
            "description": selected_proposal.description,
        }

    result: Dict[str, Any] = {
        "instrument_profile": exporter.instrument_profile,
        "mode": resolved_mode,
        "region_stem": export_result.region_stem,
        "output_dir": str(export_result.output_dir.resolve()),
        "control_config_path": str(export_result.control_config_path.resolve()),
        "scan_settings_path": str(export_result.scan_settings_path.resolve()),
        "manifest_path": str(export_result.manifest_path.resolve()),
        "control_config_sha256": export_result.control_config_sha256,
        "scan_settings_sha256": export_result.scan_settings_sha256,
        "manifest_sha256": export_result.manifest_sha256,
        "estimated_runtime_minutes": float(export_result.manifest.estimated_runtime_minutes),
        "estimated_iterations": int(duration_est.iterations),
        "start_freq_mhz": float(export_result.scan_settings.scan_parameters.start_freq),
        "end_freq_mhz": float(export_result.scan_settings.scan_parameters.end_freq),
        "step_size_mhz": float(export_result.scan_settings.scan_parameters.step_size),
        "valon_lo_mhz": float(export_result.control_config.valon_controller.freq),
        "awg_freq_mhz": int(export_result.control_config.awg_controller.awg_freq),
        "rf_level_dbm": int(export_result.control_config.valon_controller.rf_level),
        "overrides_applied": [rec.model_dump(mode="json") for rec in export_result.manifest.overrides],
        "warnings": list(export_result.warnings),
        "proposal_count": len(proposals),
        "selected_proposal": proposal_summary,
        "manifest": export_result.manifest.model_dump(mode="json"),
    }
    json.dumps(result, allow_nan=False)
    return result


def ingest_surf_run(
    run_dir: str | Path,
    *,
    snr_threshold: float = 5.0,
    gas_name: Optional[str] = None,
    beam_velocity_m_s: Optional[float] = None,
    reconcile: bool = True,
    write_provenance_ledger: bool = True,
    fp64_atol: float = 1e-9,
) -> Dict[str, Any]:
    """Ingest a GruLab SURF run directory and pick coaxial Doppler doublet PROPOSED candidates."""
    ingest_mod = _load_surf_ingest()
    watcher_mod = _load_surf_watcher()
    schemas_mod = _load_surf_schemas()

    resolved_dir = Path(run_dir).expanduser().resolve(strict=True)
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"SURF run path is not a directory: {resolved_dir}")

    settings_path = resolved_dir / "scan_settings.toml"
    cavity_dir = resolved_dir / "CavityFiles"
    if not settings_path.is_file():
        raise FileNotFoundError(f"scan_settings.toml not found in SURF run directory: {resolved_dir}")
    if not cavity_dir.is_dir():
        raise FileNotFoundError(f"CavityFiles/ directory not found in SURF run directory: {resolved_dir}")

    scan_settings = schemas_mod.SurfFrequencyScanSettings.from_surf_toml_file(settings_path)
    settings_sha = _sha256_path(settings_path)
    stem = scan_settings.output_settings.filename
    config_path = resolved_dir / f"{stem}_config.toml"
    config_sha = _sha256_path(config_path) if config_path.is_file() else None
    step_direction = ingest_mod.StepDirection.from_bounds(
        scan_settings.scan_parameters.start_freq,
        scan_settings.scan_parameters.end_freq,
    )
    step_size = float(scan_settings.scan_parameters.step_size)
    finalized_path = resolved_dir / f"{stem}.csv"

    parser = ingest_mod.SurfOutputParser()
    ledger_path: Optional[Path] = None
    ledger_sha: Optional[str] = None
    reconciliation_report = None

    if write_provenance_ledger:
        watcher = watcher_mod.SurfRunWatcher(
            resolved_dir,
            stability_polls=1,
            min_stable_seconds=0.0,
            snr_threshold=snr_threshold,
            parser=parser,
            fp64_atol=fp64_atol,
        )
        updates = watcher.poll_once()
        if not updates:
            raise ValueError(f"No valid CavityFiles/*.csv step files ingested from {resolved_dir}")
        ordered_steps = [upd.step_data for upd in updates]
        stitched_freq, stitched_int, ordered_steps = parser.stitch_steps(
            ordered_steps,
            step_direction=step_direction,
        )
        cavity_health = watcher.get_cavity_health(resolved_dir)
        if reconcile and finalized_path.is_file():
            reconciliation_report = watcher.reconcile_run(resolved_dir)
        else:
            reconciliation_report = watcher.get_reconciliation(resolved_dir)
        candidate_ledger = resolved_dir / "surf_ingest_provenance.jsonl"
        if candidate_ledger.is_file():
            ledger_path = candidate_ledger
            ledger_sha = _sha256_path(candidate_ledger)
    else:
        step_paths = sorted(cavity_dir.glob("*.csv"))
        if not step_paths:
            raise ValueError(f"No CavityFiles/*.csv step files found in {resolved_dir}")
        parsed_steps = [
            parser.parse_step_csv(
                step_path,
                step_size_mhz=step_size,
                step_direction=step_direction,
                verify_filename_stem=True,
            )
            for step_path in step_paths
        ]
        stitched_freq, stitched_int, ordered_steps = parser.stitch_steps(
            parsed_steps,
            step_direction=step_direction,
        )
        dropout_freqs = tuple(step.center_freq_mhz for step in ordered_steps if step.cavity_dropout)
        latest = ordered_steps[-1]
        cavity_health = watcher_mod.SurfCavityTrackHealth(
            status="rail_dropout" if dropout_freqs else "healthy",
            step_count=len(ordered_steps),
            last_center_freq_mhz=latest.center_freq_mhz,
            last_cavity_position_mm=latest.cavity_position_mm,
            last_step_dropout=latest.cavity_dropout,
            dropout_count=len(dropout_freqs),
            center_freqs_mhz=tuple(step.center_freq_mhz for step in ordered_steps),
            cavity_positions_mm=tuple(step.cavity_position_mm for step in ordered_steps),
            dropout_flags=tuple(step.cavity_dropout for step in ordered_steps),
            dropout_freqs_mhz=dropout_freqs,
        )
        if reconcile and finalized_path.is_file():
            reconciliation_report = parser.reconcile_run(
                finalized_csv_path=finalized_path,
                step_files=ordered_steps,
                scan_settings=scan_settings,
                config_toml_path=config_path if config_path.is_file() else None,
                scan_settings_toml_path=settings_path,
                fp64_atol=fp64_atol,
            )

    spans = [
        (
            float(step.stitched_freq_mhz.min()),
            float(step.stitched_freq_mhz.max()),
            step.file_sha256,
        )
        for step in ordered_steps
        if step.stitched_freq_mhz.size > 0
    ]
    effective_gas = scan_settings.experiment.gas_name if gas_name is None else gas_name
    peak_result = ingest_mod.pick_surf_doppler_peaks(
        stitched_freq,
        stitched_int,
        gas_name=effective_gas,
        snr_threshold=snr_threshold,
        beam_velocity_m_s=beam_velocity_m_s,
        step_provenance_spans=spans,
        source_file=f"{stem}.csv",
    )

    step_sha_map: Dict[str, str] = {}
    for step in ordered_steps:
        key = step.file_path.name if step.file_path is not None else f"{step.center_freq_mhz}.csv"
        step_sha_map[key] = step.file_sha256

    serialized_candidates: list[Dict[str, Any]] = []
    for cand in peak_result.candidates:
        status_str = cand.status.value if hasattr(cand.status, "value") else str(cand.status)
        if status_str != "proposed":
            raise ValueError(
                f"SURF ingest candidate {cand.candidate_id!r} violated HITL gate with status {status_str!r}."
            )
        serialized_candidates.append(
            {
                "candidate_id": cand.candidate_id,
                "rest_frequency_mhz": cand.rest_frequency_mhz,
                "uncertainty_mhz": cand.uncertainty_mhz,
                "intensity": cand.intensity,
                "snr": cand.snr,
                "feature_kind": cand.feature_kind.value,
                "status": status_str,
                "observed_splitting_mhz": cand.observed_splitting_mhz,
                "expected_splitting_mhz": cand.expected_splitting_mhz,
                "intensity_asymmetry": cand.intensity_asymmetry,
                "gas_name": cand.gas_name,
                "beam_velocity_m_s": cand.beam_velocity_m_s,
                "source_step_sha256": cand.source_step_sha256,
                "source_step_sha256s": list(cand.source_step_sha256s),
                "source_file": cand.source_file,
                "raw_components": [comp.model_dump(mode="json") for comp in cand.raw_components],
            }
        )

    result: Dict[str, Any] = {
        "run_dir": str(resolved_dir),
        "run_id": resolved_dir.name,
        "scan_settings_sha256": settings_sha,
        "config_toml_sha256": config_sha,
        "provenance_ledger_path": None if ledger_path is None else str(ledger_path),
        "provenance_ledger_sha256": ledger_sha,
        "step_direction": step_direction.value,
        "step_size_mhz": step_size,
        "gas_name": peak_result.gas_name,
        "beam_velocity_m_s": peak_result.beam_velocity_m_s,
        "step_count": len(ordered_steps),
        "step_file_sha256s": step_sha_map,
        "stitched_spectrum": {
            "n_points": int(stitched_freq.size),
            "dtype": str(stitched_freq.dtype),
            "freq_min_mhz": float(stitched_freq.min()) if stitched_freq.size > 0 else None,
            "freq_max_mhz": float(stitched_freq.max()) if stitched_freq.size > 0 else None,
            "intensity_min": float(stitched_int.min()) if stitched_int.size > 0 else None,
            "intensity_max": float(stitched_int.max()) if stitched_int.size > 0 else None,
            "baseline_intensity": float(peak_result.baseline_intensity),
            "mad_noise_sigma": float(peak_result.mad_noise_sigma),
        },
        "cavity_health": cavity_health.model_dump(mode="json"),
        "reconciliation": (
            None if reconciliation_report is None else reconciliation_report.model_dump(mode="json")
        ),
        "candidate_count": len(serialized_candidates),
        "proposed_candidates": serialized_candidates,
        "warnings": list(peak_result.warnings),
    }
    json.dumps(result, allow_nan=False)
    return result


def _publish_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish a strict JSON receipt atomically without overwriting existing files."""
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".spyfy-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execute_handoff(
    *,
    handoff_path: str | Path,
    output_directory: str | Path,
    registry: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute a verified BASE ``cochem.module-handoff/1`` spectroscopy request."""
    manifest = Path(handoff_path).expanduser().resolve(strict=True)
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("The handoff manifest must be a bounded regular JSON file.")

    def _unique_keys(items: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is prohibited.")
            result[key] = value
        return result

    record = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    json.dumps(record, allow_nan=False)
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != "cochem.module-handoff/1"
        or record.get("module_id") != "spyfy"
        or record.get("operation") not in _HANDOFF_OPERATIONS
        or record.get("scientific_execution_performed") is not False
    ):
        raise ValueError("Unsupported or invalid CoChem-SPYFY BASE handoff manifest.")

    options = record.get("options")
    if not isinstance(options, dict):
        raise ValueError("Handoff options must be an explicit JSON object.")

    artifact_ref = record.get("artifact")
    if not isinstance(artifact_ref, dict):
        raise ValueError("Handoff requires an explicit artifact reference.")
    filename = artifact_ref.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Artifact filename must stay within the handoff directory.")
    artifact = (manifest.parent / filename).resolve(strict=True)
    if artifact.parent != manifest.parent or artifact.is_symlink() or not artifact.is_file():
        raise ValueError("Artifact path escapes the handoff directory or is not a regular file.")
    raw_bytes = artifact.read_bytes()
    if len(raw_bytes) > _MAX_ARTIFACT_BYTES or len(raw_bytes) != artifact_ref.get("size_bytes"):
        raise ValueError("Artifact size disagrees with handoff manifest.")
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if input_sha256 != artifact_ref.get("sha256"):
        raise ValueError("Artifact SHA-256 disagrees with handoff manifest.")
    input_text = raw_bytes.decode("utf-8")

    operation = record["operation"]
    out_dir = Path(output_directory).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = registry  # Retained for BASE execution boundary signature parity.

    if operation == "parse_pickett":
        kind = str(options.get("kind") or artifact_ref.get("kind", "")).removeprefix("pickett_")
        op_result = parse_pickett(kind, input_text)
        scope = "Parse genuine Pickett .fit, .par/.var, or .cat text without fabricating missing quantities."
    elif operation == "audit_fit":
        payload_data = json.loads(input_text)
        op_result = audit_fit(
            native_constants=payload_data["native_constants"],
            spfit_constants=payload_data.get("spfit_constants"),
            tolerance_khz=options["tolerance_khz"],
            native_uncertainties=payload_data.get("native_uncertainties"),
            spfit_uncertainties=payload_data.get("spfit_uncertainties"),
            required_constants=options.get("required_constants"),
        )
        scope = "Compare native SpycFit constants with SPFIT constants under declared tolerances (SF-CMP-008/012)."
    elif operation == "fit_pickett":
        deck = json.loads(input_text)
        op_result = fit_pickett(
            par_text=deck["par_text"],
            lin_text=deck["lin_text"],
            stem=str(options.get("stem", "cochem_spfit")),
        )
        scope = "Run genuine Pickett SPFIT on complete .par and .lin decks and parse outputs."
    elif operation == "predict_spectrum":
        deck = json.loads(input_text)
        op_result = predict_spectrum(
            var_text=deck["var_text"],
            int_text=deck["int_text"],
            stem=str(options.get("stem", "cochem_spcat")),
        )
        scope = "Run genuine Pickett SPCAT on complete .var and .int decks and parse catalog transitions."
    elif operation == "plan_surf_scan":
        payload_data = json.loads(input_text)
        export_dir = options.get("output_dir") or payload_data.get("output_dir") or str(out_dir / "surf_export")
        op_result = plan_surf_scan(
            output_dir=export_dir,
            predictions=payload_data.get("predictions"),
            transition_frequencies_mhz=payload_data.get("transition_frequencies_mhz"),
            information_gains=payload_data.get("information_gains"),
            start_freq_mhz=payload_data.get("start_freq_mhz"),
            end_freq_mhz=payload_data.get("end_freq_mhz"),
            mode=options.get("mode", payload_data.get("mode")),
            overrides=options.get("overrides", payload_data.get("overrides")),
            edited_by=str(options.get("edited_by", payload_data.get("edited_by", "operator"))),
            instrument_profile=str(
                options.get("instrument_profile", payload_data.get("instrument_profile", "grulab_surf"))
            ),
            molecule_name=str(options.get("molecule_name", payload_data.get("molecule_name", "HCOOH_DME"))),
            specimen_temp_c=float(options.get("specimen_temp_c", payload_data.get("specimen_temp_c", 22.0))),
            gas_name=str(options.get("gas_name", payload_data.get("gas_name", "Ne"))),
            step_size_mhz=options.get("step_size_mhz", payload_data.get("step_size_mhz")),
            acq_num=options.get("acq_num", payload_data.get("acq_num")),
            rep_rate_hz=int(options.get("rep_rate_hz", payload_data.get("rep_rate_hz", 5))),
            max_chunk_bandwidth_mhz=float(
                options.get("max_chunk_bandwidth_mhz", payload_data.get("max_chunk_bandwidth_mhz", 50.0))
            ),
            top_n=int(options.get("top_n", payload_data.get("top_n", 3))),
            proposal_index=int(options.get("proposal_index", payload_data.get("proposal_index", 0))),
            target_constants=payload_data.get("target_constants", ()),
            resolvability_penalties=payload_data.get("resolvability_penalties"),
            line_strengths=payload_data.get("line_strengths"),
            ml_uncertainties_mhz=payload_data.get("ml_uncertainties_mhz"),
            cavity_map_dir=options.get("cavity_map_dir", payload_data.get("cavity_map_dir")),
            cavity_map_csv=options.get("cavity_map_csv", payload_data.get("cavity_map_csv")),
            fit_state_id=options.get("fit_state_id", payload_data.get("fit_state_id")),
            spcat_catalog_path=options.get("spcat_catalog_path", payload_data.get("spcat_catalog_path")),
            spcat_catalog_sha256=options.get("spcat_catalog_sha256", payload_data.get("spcat_catalog_sha256")),
            gp_hyperparameters=payload_data.get("gp_hyperparameters"),
            region_stem=options.get("region_stem", payload_data.get("region_stem")),
            scan_date=options.get("scan_date", payload_data.get("scan_date")),
            surf_output_location=options.get("surf_output_location", payload_data.get("surf_output_location")),
        )
        scope = (
            "Plan GruLab SURF cavity scan region via SurfScanPlanner, apply optional HITL overrides, "
            "and export validated SURF TOMLs with SHA-256 manifest."
        )
    elif operation == "ingest_surf_run":
        payload_data = json.loads(input_text)
        target_run_dir = options.get("run_dir") or payload_data.get("run_dir")
        if not target_run_dir:
            raise ValueError("ingest_surf_run handoff requires 'run_dir' in artifact payload or options.")
        op_result = ingest_surf_run(
            run_dir=target_run_dir,
            snr_threshold=float(options.get("snr_threshold", payload_data.get("snr_threshold", 5.0))),
            gas_name=options.get("gas_name", payload_data.get("gas_name")),
            beam_velocity_m_s=options.get("beam_velocity_m_s", payload_data.get("beam_velocity_m_s")),
            reconcile=bool(options.get("reconcile", payload_data.get("reconcile", True))),
            write_provenance_ledger=bool(
                options.get("write_provenance_ledger", payload_data.get("write_provenance_ledger", True))
            ),
            fp64_atol=float(options.get("fp64_atol", payload_data.get("fp64_atol", 1e-9))),
        )
        scope = (
            "Ingest GruLab SURF run directory via SurfRunWatcher/SurfOutputParser and coaxial "
            "Doppler doublet peak picker, emitting strictly PROPOSED candidates with SHA-256 provenance."
        )
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    if _sha256_path(artifact) != input_sha256:
        raise ValueError("Handoff artifact bytes changed during execution.")

    receipt: Dict[str, Any] = {
        "schema_version": "cochem.module-operation/1",
        "module_id": "spyfy",
        "operation": operation,
        "status": "completed",
        "input_sha256": input_sha256,
        "provider_file": str(Path(__file__).resolve()),
        "provider_sha256": _source_digest(),
        "provider_distribution": "CoChem-SPYFY",
        "provider_version": _package_version(),
        "scope": scope,
        "result": op_result,
    }
    json.dumps(receipt, allow_nan=False)
    _publish_json(out_dir / f"{operation}_receipt.json", receipt)
    return receipt


def spyfy_provider() -> Dict[str, Any]:
    """Zero-argument ``cochem.modules`` entry point for CoChem-SPYFY."""
    spfit_probe = _probe_pickett_binary("spfit")
    spcat_probe = _probe_pickett_binary("spcat")

    predict_def: Dict[str, Any] = {
        "available": bool(spcat_probe["available"]),
        "required_engines": ["spcat"],
        "supported_engines": ["spcat"],
        "handler": "cochem_spyfy.module_provider:predict_spectrum",
        "scope": (
            "Predict rotational transition frequencies and intensities using genuine "
            "Pickett SPCAT from complete .var and .int decks."
        ),
    }
    if spcat_probe["available"]:
        predict_def["runtime_receipts"] = {"spcat": spcat_probe}
    else:
        predict_def["reason"] = spcat_probe.get("reason", "spcat binary is unavailable.")

    fit_def: Dict[str, Any] = {
        "available": bool(spfit_probe["available"]),
        "required_engines": ["spfit"],
        "supported_engines": ["spfit"],
        "handler": "cochem_spyfy.module_provider:fit_pickett",
        "scope": (
            "Fit rotational and centrifugal-distortion parameters using genuine "
            "Pickett SPFIT from complete .par and .lin decks."
        ),
    }
    if spfit_probe["available"]:
        fit_def["runtime_receipts"] = {"spfit": spfit_probe}
    else:
        fit_def["reason"] = spfit_probe.get("reason", "spfit binary is unavailable.")

    parse_def: Dict[str, Any] = {
        "available": True,
        "required_engines": [],
        "supported_engines": [],
        "handler": "cochem_spyfy.module_provider:parse_pickett",
        "scope": (
            "Parse genuine Pickett .fit, .par/.var, and .cat files into structured "
            "records without fabricating absent fields."
        ),
    }

    audit_def: Dict[str, Any] = {
        "available": True,
        "required_engines": [],
        "supported_engines": [],
        "handler": "cochem_spyfy.module_provider:audit_fit",
        "scope": (
            "Compare native SpycFit constants against SPFIT constants under explicitly "
            "declared per-constant kHz tolerances (SF-CMP-008/012)."
        ),
    }

    plan_surf_def: Dict[str, Any] = {
        "available": True,
        "required_engines": [],
        "supported_engines": [],
        "handler": "cochem_spyfy.module_provider:plan_surf_scan",
        "scope": (
            "Plan GruLab SURF cavity scan regions with SurfScanPlanner, apply optional "
            "HITL operator overrides, and export validated SURF TOMLs and SHA-256 manifest."
        ),
    }

    ingest_surf_def: Dict[str, Any] = {
        "available": True,
        "required_engines": [],
        "supported_engines": [],
        "handler": "cochem_spyfy.module_provider:ingest_surf_run",
        "scope": (
            "Ingest a GruLab SURF run directory via SurfRunWatcher/SurfOutputParser and "
            "coaxial Doppler doublet peak picker, returning strictly PROPOSED candidates "
            "with per-file SHA-256 provenance."
        ),
    }

    profiles_mod = _load_instrument_profiles()
    metadata: Dict[str, Any] = {
        "module_id": "spyfy",
        "distribution": "CoChem-SPYFY",
        "version": _package_version(),
        "integration_contract": "cochem.module-handoff/1",
        "scientific_api_version": "cochem.module-provider/1",
        "execute_handoff": "cochem_spyfy.module_provider:execute_handoff",
        "default_instrument_profile": profiles_mod.DEFAULT_INSTRUMENT_PROFILE,
        "instrument_profiles": profiles_mod.list_instrument_profiles(),
        "surf_compatibility": profiles_mod.surf_compatibility_metadata(),
        "operations": list(_OPERATIONS),
        "surf_operations": list(_SURF_OPERATIONS),
        "scientific_apis": {
            "predict_spectrum": predict_def,
            "fit_pickett": fit_def,
            "parse_pickett": parse_def,
            "audit_fit": audit_def,
            "plan_surf_scan": plan_surf_def,
            "ingest_surf_run": ingest_surf_def,
        },
    }
    json.dumps(metadata, allow_nan=False)
    return metadata


module_provider = spyfy_provider

__all__ = [
    "audit_fit",
    "execute_handoff",
    "fit_pickett",
    "ingest_surf_run",
    "module_provider",
    "parse_pickett",
    "plan_surf_scan",
    "predict_spectrum",
    "spyfy_provider",
]
