"""CoChem-BASE compatible module provider for CoChem-SpycFit.

Publishes the ``cochem.modules`` entry point ``spycfit`` conforming to
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
        "cochem_spycfit.serialization.cochem_spycfit_pickett_output",
        "serialization/cochem_spycfit_pickett_output.py",
    )


def _load_spcat_wrapper() -> Any:
    # Ensure sibling parser submodule is registered before wrapper relative import
    _load_pickett_output()
    return _load_submodule(
        "cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper",
        "serialization/cochem_spycfit_spcat_wrapper.py",
    )


def _load_auditor() -> Any:
    return _load_submodule(
        "cochem_spycfit.physics_core.cochem_spycfit_auditor",
        "physics_core/cochem_spycfit_auditor.py",
    )


def _load_instrument_profiles() -> Any:
    return _load_submodule(
        "cochem_spycfit.instruments.profiles",
        "instruments/profiles.py",
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
        return importlib.metadata.version("CoChem-SpycFit")
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


def _publish_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish a strict JSON receipt atomically without overwriting existing files."""
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".spycfit-", delete=False
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
        or record.get("module_id") != "spycfit"
        or record.get("operation") not in _OPERATIONS
        or record.get("scientific_execution_performed") is not False
    ):
        raise ValueError("Unsupported or invalid SpycFit BASE handoff manifest.")

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
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    if _sha256_path(artifact) != input_sha256:
        raise ValueError("Handoff artifact bytes changed during execution.")

    receipt: Dict[str, Any] = {
        "schema_version": "cochem.module-operation/1",
        "module_id": "spycfit",
        "operation": operation,
        "status": "completed",
        "input_sha256": input_sha256,
        "provider_file": str(Path(__file__).resolve()),
        "provider_sha256": _source_digest(),
        "provider_distribution": "CoChem-SpycFit",
        "provider_version": _package_version(),
        "scope": scope,
        "result": op_result,
    }
    json.dumps(receipt, allow_nan=False)
    _publish_json(out_dir / f"{operation}_receipt.json", receipt)
    return receipt


def spycfit_provider() -> Dict[str, Any]:
    """Zero-argument ``cochem.modules`` entry point for CoChem-SpycFit."""
    spfit_probe = _probe_pickett_binary("spfit")
    spcat_probe = _probe_pickett_binary("spcat")

    predict_def: Dict[str, Any] = {
        "available": bool(spcat_probe["available"]),
        "required_engines": ["spcat"],
        "supported_engines": ["spcat"],
        "handler": "cochem_spycfit.module_provider:predict_spectrum",
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
        "handler": "cochem_spycfit.module_provider:fit_pickett",
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
        "handler": "cochem_spycfit.module_provider:parse_pickett",
        "scope": (
            "Parse genuine Pickett .fit, .par/.var, and .cat files into structured "
            "records without fabricating absent fields."
        ),
    }

    audit_def: Dict[str, Any] = {
        "available": True,
        "required_engines": [],
        "supported_engines": [],
        "handler": "cochem_spycfit.module_provider:audit_fit",
        "scope": (
            "Compare native SpycFit constants against SPFIT constants under explicitly "
            "declared per-constant kHz tolerances (SF-CMP-008/012)."
        ),
    }

    profiles_mod = _load_instrument_profiles()
    metadata: Dict[str, Any] = {
        "module_id": "spycfit",
        "distribution": "CoChem-SpycFit",
        "version": _package_version(),
        "integration_contract": "cochem.module-handoff/1",
        "scientific_api_version": "cochem.module-provider/1",
        "execute_handoff": "cochem_spycfit.module_provider:execute_handoff",
        "default_instrument_profile": profiles_mod.DEFAULT_INSTRUMENT_PROFILE,
        "instrument_profiles": profiles_mod.list_instrument_profiles(),
        "surf_compatibility": profiles_mod.surf_compatibility_metadata(),
        "operations": list(_OPERATIONS),
        "scientific_apis": {
            "predict_spectrum": predict_def,
            "fit_pickett": fit_def,
            "parse_pickett": parse_def,
            "audit_fit": audit_def,
        },
    }
    json.dumps(metadata, allow_nan=False)
    return metadata


module_provider = spycfit_provider

__all__ = [
    "audit_fit",
    "execute_handoff",
    "fit_pickett",
    "module_provider",
    "parse_pickett",
    "predict_spectrum",
    "spycfit_provider",
]
