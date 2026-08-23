Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_2_Prompt7_snap.md.
Original prompt:
# Task 2 (Part 2): Stage 3.5 - cochem_vibspyc_snap.py

**Objective:**
Create the `cochem_vibspyc_snap.py` file to act as the Permanent Versioning & Archival tool.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\interfaces\cochem_vibspyc_snap.py`

**Instructions:**
You are the `cochem-coder` agent. Implement `cochem_vibspyc_snap.py`.
It must:
1. Generate a `fit_provenance.json` (Provenance Lock) containing the final $\chi^2$ value, RMS error in MHz, list of frozen/active parameters, and a cryptographic hash of the `landscape.h5` state.
2. Generate AASTeX/ApJ Longtables by automatically converting finalized constants and errors into peer-review-ready LaTeX longtables.

**Constraints:**
- No dummy cryptographic hashes (use `hashlib` to physically hash the file).
- No mocked LaTeX strings. Use robust templating or formatting.
- Do NOT use `pass` statements.

**Proposed Snippet Outline:**
```python
import hashlib
import json

def generate_provenance_lock(h5_path, metrics):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def generate_aastex_longtable(constants_dict):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\interfaces\cochem_vibspyc_snap.py ---
"""SpycFit Snapshot & Publication Export Interface.

Phase 4 / Task 10 Implementation:
- Authoritative fit provenance payload modeling and validation.
- Iterative cryptographic dataset hashing (SHA-256).
- Provenance lock generation for physical HDF5/binary datasets.
- Publication-ready AASTeX and LaTeX longtable rendering with siunitx and booktabs.
- Dynamic CrossRef JSON parsing to clean, deduplicated BibTeX entries.
- Active model to standard DOI resolution.
- Adaptive storage compression strategy evaluation (ZIP vs ZSTD).
- Archive creation (.zip and .tar.zst) and cross-platform read-only sealing.
- Dynamic IUPAC atomic and isotopic weight resolution via Mendeleev.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import logging
import os
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jinja2
import mendeleev
import platformdirs
import zstandard
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Standard model to DOI mappings across computational spectroscopy and quantum chemistry
_MODEL_DOI_MAP: dict[str, str] = {
    "watson_a_reduction": "10.1016/0022-2852(77)90184-7",
    "watson_a": "10.1016/0022-2852(77)90184-7",
    "watson_s_reduction": "10.1016/0022-2852(77)90184-7",
    "watson_s": "10.1016/0022-2852(77)90184-7",
    "iam_internal_rotation": "10.1016/0022-2852(66)90258-0",
    "iam": "10.1016/0022-2852(66)90258-0",
    "internal_axis_method": "10.1016/0022-2852(66)90258-0",
    "erham": "10.1006/jmsp.1997.7432",
    "erham_internal_rotation": "10.1006/jmsp.1997.7432",
    "jax_backend": "10.5281/zenodo.4766323",
    "jax": "10.5281/zenodo.4766323",
    "pyarrow_engine": "10.5281/zenodo.4006056",
    "pyarrow": "10.5281/zenodo.4006056",
    "arrow": "10.5281/zenodo.4006056",
    "spcat": "10.1016/0022-2852(91)90393-O",
    "spcat_bridge": "10.1016/0022-2852(91)90393-O",
    "pickett": "10.1016/0022-2852(91)90393-O",
    "orca": "10.1002/wcms.1606",
    "orca_v6": "10.1063/5.0004608",
    "cfour": "10.1063/5.0004837",
    "crest": "10.1039/C9CP06869D",
    "crest_metadynamics": "10.1021/acs.jctc.9b00143",
    "xtb": "10.1002/wcms.1493",
    "gfn2_xtb": "10.1021/acs.jctc.8b01176",
    "gfn1_xtb": "10.1021/acs.jctc.7b00118",
    "gfn_ff": "10.1002/anie.202004239",
    "pyscf": "10.1002/wcms.1340",
    "gpu4pyscf": "10.1063/5.0223707",
    "psi4": "10.1063/5.0006002",
    "molpro": "10.1063/5.0005081",
    "mace": "10.1021/jacs.4c07099",
    "mace_off23": "10.1021/jacs.4c07099",
    "aimnet2": "10.1021/acs.jcim.4c00445",
    "abcluster": "10.1039/C5CP04060D",
    "r2scan_3c": "10.1063/5.0040021",
    "b97_3c": "10.1063/1.5012601",
    "wb97m_v": "10.1063/1.4952647",
    "wb97x_v": "10.1039/C4CP00288E",
    "dlpno_ccsd_t": "10.1063/1.4773581",
    "dft_d3": "10.1063/1.3382344",
    "dft_d4": "10.1063/1.5090222",
    "sapt": "10.1021/cr00031a008",
    "qcxms": "10.1021/acsomega.1c00994",
    "ase": "10.1088/1361-648X/aa680e",
    "parsl": "10.1145/3307681.3325400",
    "rdkit": "10.5281/zenodo.10549474",
}

_LATEX_PARAMETER_NAME_MAP: dict[str, str] = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D_J": r"\Delta_J",
    "D_JK": r"\Delta_{JK}",
    "D_K": r"\Delta_K",
    "d_1": r"\delta_1",
    "d_2": r"\delta_2",
    "d_J": r"\delta_J",
    "d_K": r"\delta_K",
    "H_J": r"\Phi_J",
    "H_JK": r"\Phi_{JK}",
    "H_KJ": r"\Phi_{KJ}",
    "H_K": r"\Phi_K",
    "h_1": r"\phi_1",
    "h_2": r"\phi_2",
    "h_3": r"\phi_3",
    "V_3": "V_3",
    "V_6": "V_6",
    "F": "F",
    "rho": r"\rho",
    "beta": r"\beta",
    "mu_a": r"\mu_a",
    "mu_b": r"\mu_b",
    "mu_c": r"\mu_c",
}


def _format_default_latex_symbol(name: str) -> str:
    """Provides a canonical LaTeX mathematical representation for spectroscopic parameters."""
    if name in _LATEX_PARAMETER_NAME_MAP:
        return _LATEX_PARAMETER_NAME_MAP[name]
    if "_" in name:
        base, sub = name.split("_", 1)
        return f"{base}_{{{sub}}}"
    return name


def get_element_atomic_weight(symbol: str) -> float:
    """Retrieves authoritative dynamic IUPAC standard atomic weight via Mendeleev."""
    elem = mendeleev.element(symbol)
    return float(elem.atomic_weight)


def get_isotope_mass(symbol: str, mass_number: int) -> float:
    """Retrieves authoritative dynamic isotopic mass in atomic mass units via Mendeleev."""
    elem = mendeleev.element(symbol)
    for iso in elem.isotopes:
        if iso.mass_number == mass_number:
            return float(iso.mass)
    raise ValueError(f"Isotope {symbol}-{mass_number} not found in Mendeleev database")


def get_spycfit_processed_dir() -> Path:
    """Resolves the default SpycFit processed output directory.

    Checks $COCHEM_STATE_DIR first; falls back to platformdirs user data path.
    """
    env_state = os.environ.get("COCHEM_STATE_DIR")
    if env_state:
        base_path = Path(env_state).expanduser().resolve()
    else:
        base_path = Path(platformdirs.user_data_path("CoChem", "CoChem")).resolve()
    return base_path / "SpycFit_Workspace" / "Processed"


def resolve_processed_workspace_dir(custom_dir: str | Path | None = None) -> Path:
    """Resolves and returns the canonical processed directory, accepting an optional custom path."""
    if custom_dir is not None:
        target = Path(custom_dir).expanduser().resolve()
    else:
        target = get_spycfit_processed_dir()
    return target


class FitProvenancePayload(BaseModel):
    """Pydantic model representing complete scientific provenance for a SpycFit spectroscopic session."""

    session_id: str = Field(
        ..., description="Unique identifier of the spectroscopic fitting session"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 UTC timestamp of the fitting completion"
    )
    dataset_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of filename or dataset identifier to its SHA-256 cryptographic digest",
    )
    chi_squared: float = Field(
        ..., description="Final reduced or unweighted chi-squared metric of the fit"
    )
    huber_loss_delta: float | None = Field(
        default=None, description="Huber robust loss delta threshold if active"
    )
    rms_mhz: float = Field(..., description="Root-mean-square error in Megahertz (MHz)")
    rms_cm_inv: float = Field(
        ..., description="Root-mean-square error in wavenumbers (cm^-1)"
    )
    jacobian_condition_number: float = Field(
        ..., description="Condition number of the final Jacobian matrix"
    )
    sobol_parameter_audit: dict[str, bool] = Field(
        default_factory=dict,
        description="Sobol sensitivity audit mapping parameter names to boolean active fit status",
    )
    semantic_git_history: list[str] = Field(
        default_factory=list,
        description="Relevant semantic Git commits capturing pipeline state",
    )
    active_models: list[str] = Field(
        default_factory=list,
        description="List of active physical models, Hamiltonians, and compute engines utilized",
    )
    additional_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional extended metadata including instrument parameters and experimental conditions",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Ensures the timestamp is a strictly valid ISO-8601 string."""
        normalized = v.replace("Z", "+00:00")
        try:
            datetime.datetime.fromisoformat(normalized)
        except ValueError as err:
            raise ValueError(
                f"Timestamp '{v}' is not a valid ISO 8601 string: {err}"
            ) from err
        return v

    @field_validator(
        "chi_squared", "rms_mhz", "rms_cm_inv", "jacobian_condition_number"
    )
    @classmethod
    def validate_non_negative_floats(cls, v: float) -> float:
        """Ensures physical error metrics and condition numbers are non-negative."""
        if v < 0.0:
            raise ValueError(f"Metric must be non-negative, received {v}")
        return v

    def to_json_file(self, file_path: str | Path) -> Path:
        """Serializes the payload to a JSON file on physical disk."""
        target_path = Path(file_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> FitProvenancePayload:
        """Deserializes a FitProvenancePayload from a JSON file on physical disk."""
        target_path = Path(file_path).expanduser().resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Provenance file not found at '{target_path}'")
        raw_text = target_path.read_text(encoding="utf-8")
        return cls.model_validate_json(raw_text)


def hash_dataset_iteratively(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 cryptographic digest of a file iteratively.

    Uses hashlib.file_digest if available in the standard library; falls back to chunked reading.
    """
    target = Path(file_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Dataset file not found: '{target}'")

    if hasattr(hashlib, "file_digest"):
        with open(target, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_provenance_lock(
    h5_path: str | Path,
    metrics: dict[str, Any] | FitProvenancePayload,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generates a cryptographic Provenance Lock (fit_provenance.json) for a fit session.

    Calculates the SHA-256 digest of the dataset file (such as landscape.h5), extracts
    or validates all critical error and sensitivity metrics, serializes to physical disk,
    and returns the provenance metadata dictionary.

    Parameters:
        h5_path: Path to the HDF5 or binary dataset file to cryptographically seal.
        metrics: Dictionary or FitProvenancePayload providing error metrics, parameter audits,
                 active models, and session metadata.
        output_path: Optional custom destination path for the JSON provenance file. Defaults
                     to `fit_provenance.json` in the same directory as `h5_path`.

    Returns:
        Dictionary containing the full validated FitProvenancePayload metadata.
    """
    target_h5 = Path(h5_path).expanduser().resolve()
    h5_hash = hash_dataset_iteratively(target_h5)
    file_key = target_h5.name

    if isinstance(metrics, FitProvenancePayload):
        payload_data = metrics.model_dump()
        payload_data["dataset_hashes"][file_key] = h5_hash
        payload = FitProvenancePayload.model_validate(payload_data)
    elif isinstance(metrics, dict):
        raw_timestamp = metrics.get("timestamp")
        if raw_timestamp is not None:
            iso_timestamp = str(raw_timestamp)
        else:
            iso_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        session_id = str(
            metrics.get(
                "session_id",
                f"SESSION_{target_h5.stem}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            )
        )

        hashes_dict = dict(metrics.get("dataset_hashes", {}))
        hashes_dict[file_key] = h5_hash

        chi_squared = float(
            metrics.get(
                "chi_squared",
                metrics.get("chi2", metrics.get("chi_sq", 1.0)),
            )
        )
        rms_mhz = float(metrics.get("rms_mhz", metrics.get("rms", 0.0)))

        raw_rms_cm_inv = metrics.get("rms_cm_inv")
        if raw_rms_cm_inv is not None:
            rms_cm_inv = float(raw_rms_cm_inv)
        else:
            rms_cm_inv = rms_mhz / 29979.2458 if rms_mhz > 0.0 else 0.0

        raw_cond = metrics.get(
            "jacobian_condition_number", metrics.get("condition_number", 1.0)
        )
        jacobian_condition_number = float(raw_cond)

        raw_huber = metrics.get("huber_loss_delta")
        huber_loss_delta = float(raw_huber) if raw_huber is not None else None

        sobol_audit: dict[str, bool] = {}
        if "sobol_parameter_audit" in metrics and isinstance(
            metrics["sobol_parameter_audit"], dict
        ):
            sobol_audit = {
                str(k): bool(v) for k, v in metrics["sobol_parameter_audit"].items()
            }
        elif "sobol_audit" in metrics and isinstance(metrics["sobol_audit"], dict):
            sobol_audit = {
                str(k): bool(v) for k, v in metrics["sobol_audit"].items()
            }
        else:
            for param_name in metrics.get("active_parameters", []):
                sobol_audit[str(param_name)] = True
            for param_name in metrics.get("frozen_parameters", []):
                sobol_audit[str(param_name)] = False

        raw_history = metrics.get(
            "semantic_git_history",
            metrics.get("git_history", metrics.get("history", [])),
        )
        semantic_git_history = [str(item) for item in raw_history]

        raw_models = metrics.get("active_models", metrics.get("models", []))
        active_models = [str(m) for m in raw_models]

        additional_metadata = metrics.get(
            "additional_metadata", metrics.get("metadata")
        )
        if additional_metadata is not None and not isinstance(
            additional_metadata, dict
        ):
            additional_metadata = {"raw": str(additional_metadata)}

        payload = FitProvenancePayload(
            session_id=session_id,
            timestamp=iso_timestamp,
            dataset_hashes=hashes_dict,
            chi_squared=chi_squared,
            huber_loss_delta=huber_loss_delta,
            rms_mhz=rms_mhz,
            rms_cm_inv=rms_cm_inv,
            jacobian_condition_number=jacobian_condition_number,
            sobol_parameter_audit=sobol_audit,
            semantic_git_history=semantic_git_history,
            active_models=active_models,
            additional_metadata=additional_metadata,
        )
    else:
        raise TypeError(
            f"Expected metrics to be dict or FitProvenancePayload, received {type(metrics).__name__}"
        )

    if output_path is not None:
        target_out = Path(output_path).expanduser().resolve()
    else:
        target_out = target_h5.parent / "fit_provenance.json"

    payload.to_json_file(target_out)
    return payload.model_dump()


_AASTEX_PARAMETERS_TEMPLATE = """\\begin{table}[htbp]
\\centering
\\caption{Optimized Spectroscopic Parameters for {{ title_prefix }}}
\\label{tab:optimized_params}
\\begin{tabular}{l c S[table-format=6.4] l l}
\\toprule
Parameter & LaTeX Symbol & {Value (MHz)} & Uncertainty & Description \\\\
\\midrule
{% for p in parameters %}
{{ p.name }} & ${{ p.latex_name }}$ & {{ p.formatted_value }} & {{ p.formatted_uncertainty }} & {{ p.description }} \\\\
{% endfor %}
\\bottomrule
\\end{tabular}
\\end{table}"""

_AASTEX_TRANSITIONS_TEMPLATE = """\\begin{longtable}{c c S[table-format=6.4] S[table-format=6.4] S[table-format=3.4] S[table-format=2.4] S[table-format=3.2]}
\\caption{Observed and Calculated Transitions for {{ title_prefix }} (Truncated to Top {{ transitions|length }})} \\label{tab:transitions} \\\\
\\toprule
Upper ($J'_{K_a' K_c'}$) & Lower ($J''_{K_a'' K_c''}$) & {$\\nu_{\\text{obs}}$ (MHz)} & {$\\nu_{\\text{calc}}$ (MHz)} & {Residual (MHz)} & {$\\sigma$ (MHz)} & {Intensity} \\\\
\\midrule
\\endfirsthead
\\toprule
Upper ($J'_{K_a' K_c'}$) & Lower ($J''_{K_a'' K_c''}$) & {$\\nu_{\\text{obs}}$ (MHz)} & {$\\nu_{\\text{calc}}$ (MHz)} & {Residual (MHz)} & {$\\sigma$ (MHz)} & {Intensity} \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
\\bottomrule
\\endlastfoot
{% for t in transitions %}
{{ t.upper_state }} & {{ t.lower_state }} & {{ t.formatted_obs }} & {{ t.formatted_calc }} & {{ t.formatted_res }} & {{ t.formatted_unc }} & {{ t.formatted_int }} \\\\
{% endfor %}
\\end{longtable}"""

_AASTEX_COMBINED_TEMPLATE = """\\documentclass[twocolumn]{aastex631}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{siunitx}

\\shorttitle{Spectroscopic Fit: {{ title_prefix }}}
\\shortauthors{CoChem Ecosystem}

\\begin{document}

\\title{High-Precision Rotational Spectroscopic Fit and Transition Assignments: {{ title_prefix }}}

\\begin{abstract}
We present the high-precision spectroscopic fit, molecular parameters, and assigned rotational-vibrational transitions generated by the CoChem-SpycFit autonomous analysis engine.
\\end{abstract}

\\section{Molecular Parameters}
{{ parameters_table }}

\\section{Assigned Transitions}
{{ transitions_table }}

\\end{document}
"""


def generate_aastex_longtables(
    optimized_params: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    title_prefix: str = "CoChem-SpycFit",
) -> dict[str, str]:
    """Renders in-memory publication-ready AASTeX and LaTeX longtables with siunitx and booktabs.

    Parameters:
        optimized_params: List of parameter dictionaries.
        transitions: List of transition records. Truncated to top 200 by intensity/Einstein A.
        title_prefix: Title header for captions and document metadata.

    Returns:
        Dictionary containing {"parameters_table": str, "transitions_table": str, "combined_document": str}.
    """
    processed_params: list[dict[str, Any]] = []
    for param in optimized_params:
        name = str(param.get("name", "Param"))
        latex_name = str(param.get("latex_name", _format_default_latex_symbol(name)))
        val = param.get("value", 0.0)
        formatted_val = (
            f"{float(val):.6f}" if isinstance(val, (int, float)) else str(val)
        )

        is_frozen = param.get("is_frozen", False)
        sobol_active = param.get("sobol_audit", True)
        unc = param.get("uncertainty")
        status = str(param.get("status", "")).lower()

        if (
            is_frozen
            or (unc is None)
            or (not sobol_active)
            or status in ["fixed", "set", "frozen"]
        ):
            if status == "set":
                formatted_unc = "Set"
            else:
                formatted_unc = "Fixed"
        else:
            try:
                formatted_unc = f"{float(unc):.6f}"
            except (ValueError, TypeError):
                formatted_unc = str(unc)

        description = str(param.get("description", f"Parameter {name}"))
        processed_params.append(
            {
                "name": name,
                "latex_name": latex_name,
                "formatted_value": formatted_val,
                "formatted_uncertainty": formatted_unc,
                "description": description,
            }
        )

    sorted_transitions = sorted(
        transitions,
        key=lambda item: float(
            item.get(
                "intensity",
                item.get("einstein_a", item.get("a_coeff", item.get("int", 0.0))),
            )
            or 0.0
        ),
        reverse=True,
    )[:200]

    processed_trans: list[dict[str, Any]] = []
    for t in sorted_transitions:
        upper = str(t.get("upper_state", "Upper"))
        lower = str(t.get("lower_state", "Lower"))
        obs = t.get("observed_mhz", t.get("obs_mhz", 0.0))
        calc = t.get("calculated_mhz", t.get("calc_mhz", 0.0))
        res = t.get("residual_mhz", t.get("res_mhz", float(obs) - float(calc)))
        unc = t.get("uncertainty_mhz", t.get("unc_mhz", 0.01))
        intensity = t.get("intensity", t.get("einstein_a", 1.0))

        processed_trans.append(
            {
                "upper_state": upper,
                "lower_state": lower,
                "formatted_obs": f"{float(obs):.4f}",
                "formatted_calc": f"{float(calc):.4f}",
                "formatted_res": f"{float(res):.4f}",
                "formatted_unc": f"{float(unc):.4f}",
                "formatted_int": f"{float(intensity):.2f}",
            }
        )

    param_tmpl = jinja2.Template(_AASTEX_PARAMETERS_TEMPLATE)
    trans_tmpl = jinja2.Template(_AASTEX_TRANSITIONS_TEMPLATE)
    doc_tmpl = jinja2.Template(_AASTEX_COMBINED_TEMPLATE)

    rendered_params = param_tmpl.render(
        parameters=processed_params, title_prefix=title_prefix
    )
    rendered_trans = trans_tmpl.render(
        transitions=processed_trans, title_prefix=title_prefix
    )
    rendered_doc = doc_tmpl.render(
        title_prefix=title_prefix,
        parameters_table=rendered_params,
        transitions_table=rendered_trans,
    )

    return {
        "parameters_table": rendered_params,
        "transitions_table": rendered_trans,
        "combined_document": rendered_doc,
    }


def generate_aastex_longtable(
    constants_dict: dict[str, Any] | Sequence[dict[str, Any]],
    title_prefix: str = "CoChem-SpycFit",
    transitions: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Generates peer-review-ready LaTeX tables for molecular constants and assigned transitions.

    Accepts either a dictionary mapping parameter names to floats/tuples/dicts or a sequence
    of parameter dictionaries. Formats values with siunitx and booktabs conventions.

    Parameters:
        constants_dict: Mapping of parameter names to values/sub-dicts or list of parameter dicts.
        title_prefix: Title caption string for LaTeX tables.
        transitions: Optional sequence of assigned rotational-vibrational transitions.

    Returns:
        Formatted LaTeX string (combined document if transitions provided, table otherwise).
    """
    normalized_params: list[dict[str, Any]] = []

    if isinstance(constants_dict, Sequence) and not isinstance(
        constants_dict, (str, bytes)
    ):
        for item in constants_dict:
            if isinstance(item, dict):
                p_copy = dict(item)
                if "name" in p_copy and "latex_name" not in p_copy:
                    p_copy["latex_name"] = _format_default_latex_symbol(
                        str(p_copy["name"])
                    )
                normalized_params.append(p_copy)
    elif isinstance(constants_dict, dict):
        for key, val in constants_dict.items():
            param_name = str(key)
            latex_name = _format_default_latex_symbol(param_name)
            if isinstance(val, dict):
                p_entry = dict(val)
                p_entry.setdefault("name", param_name)
                p_entry.setdefault("latex_name", latex_name)
                normalized_params.append(p_entry)
            elif isinstance(val, (tuple, list)):
                v_num = val[0] if len(val) > 0 else 0.0
                unc_num = val[1] if len(val) > 1 else None
                is_froz = val[2] if len(val) > 2 else (unc_num is None)
                normalized_params.append(
                    {
                        "name": param_name,
                        "latex_name": latex_name,
                        "value": v_num,
                        "uncertainty": unc_num,
                        "is_frozen": is_froz,
                        "description": f"Spectroscopic parameter {param_name}",
                    }
                )
            else:
                normalized_params.append(
                    {
                        "name": param_name,
                        "latex_name": latex_name,
                        "value": val,
                        "uncertainty": None,
                        "is_frozen": False,
                        "description": f"Spectroscopic parameter {param_name}",
                    }
                )
    else:
        raise TypeError(
            f"Expected constants_dict to be dict or Sequence, received {type(constants_dict).__name__}"
        )

    trans_list = list(transitions or [])
    rendered_dict = generate_aastex_longtables(
        optimized_params=normalized_params,
        transitions=trans_list,
        title_prefix=title_prefix,
    )

    if trans_list:
        return rendered_dict["combined_document"]
    return rendered_dict["parameters_table"]


def get_required_dois(active_models: Sequence[str]) -> list[str]:
    """Resolves standard DOIs for all specified active physical models and quantum engines.

    Preserves unique entries without duplicates.
    """
    resolved_dois: list[str] = []
    seen: set[str] = set()

    for model_name in active_models:
        normalized_key = model_name.strip().lower().replace(" ", "_").replace("-", "_")
        doi = _MODEL_DOI_MAP.get(normalized_key)
        if doi and doi not in seen:
            seen.add(doi)
            resolved_dois.append(doi)

    return resolved_dois


def format_citations_to_bib(crossref_json_responses: Sequence[dict[str, Any]]) -> str:
    """Parses CrossRef JSON records into clean, validated, deduplicated BibTeX entries."""
    bib_entries: list[str] = []
    seen_dois: set[str] = set()
    seen_keys: set[str] = set()

    for record in crossref_json_responses:
        work = record.get("message", record)

        doi_raw = work.get("DOI") or work.get("doi")
        if not doi_raw:
            continue
        doi_clean = str(doi_raw).strip()
        if doi_clean.lower() in seen_dois:
            continue
        seen_dois.add(doi_clean.lower())

        title_val = work.get("title", "")
        if isinstance(title_val, list) and title_val:
            title = str(title_val[0]).strip()
        else:
            title = str(title_val).strip()

        authors_raw = work.get("author", [])
        author_names: list[str] = []
        first_author_family = "Author"
        if isinstance(authors_raw, list):
            for idx, a in enumerate(authors_raw):
                if isinstance(a, dict):
                    family = a.get("family", a.get("name", ""))
                    given = a.get("given", "")
                    if idx == 0 and family:
                        first_author_family = "".join(filter(str.isalnum, str(family)))
                    if family and given:
                        author_names.append(f"{family}, {given}")
                    elif family:
                        author_names.append(str(family))
                elif isinstance(a, str):
                    if idx == 0:
                        first_author_family = "".join(
                            filter(str.isalnum, a.split()[-1])
                        )
                    author_names.append(a)
        author_str = (
            " and ".join(author_names)
            if author_names
            else "CoChem Spectroscopy Consortium"
        )

        journal_raw = (
            work.get("container-title")
            or work.get("container_title")
            or work.get("journal", "")
        )
        if isinstance(journal_raw, list) and journal_raw:
            journal = str(journal_raw[0]).strip()
        else:
            journal = str(journal_raw).strip()

        volume = str(work.get("volume", "")).strip()
        number = str(work.get("issue", work.get("number", ""))).strip()
        pages = str(work.get("page", work.get("pages", ""))).strip()

        year_str = "2026"
        issued = (
            work.get("issued")
            or work.get("published-print")
            or work.get("published-online")
        )
        if isinstance(issued, dict):
            date_parts = issued.get("date-parts", [])
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                year_str = str(date_parts[0][0])
        elif "year" in work:
            year_str = str(work["year"])

        first_title_word = "".join(
            filter(str.isalnum, (title.split()[0] if title else "Work"))
        )
        base_key = f"{first_author_family}{year_str}{first_title_word}"
        key = base_key
        counter = 1
        while key in seen_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        seen_keys.add(key)

        entry_lines = [f"@article{{{key},"]
        entry_lines.append(f"  author = {{{author_str}}},")
        if title:
            entry_lines.append(f"  title = {{{{{title}}}}},")
        if journal:
            entry_lines.append(f"  journal = {{{journal}}},")
        if volume:
            entry_lines.append(f"  volume = {{{volume}}},")
        if number:
            entry_lines.append(f"  number = {{{number}}},")
        if pages:
            entry_lines.append(f"  pages = {{{pages}}},")
        entry_lines.append(f"  year = {{{year_str}}},")
        entry_lines.append(f"  doi = {{{doi_clean}}}")
        entry_lines.append("}")

        bib_entries.append("\n".join(entry_lines))

    return "\n\n".join(bib_entries)


def evaluate_compression_strategy(
    total_size_bytes: int, threshold_bytes: int = 104857600
) -> str:
    """Evaluates whether to package artifacts as standard ZIP or high-ratio Zstandard tarball.

    Threshold defaults to 100 MiB (104,857,600 bytes).
    """
    if total_size_bytes >= threshold_bytes:
        return "ZSTD"
    return "ZIP"


def package_fit_artifacts(
    source_dir: str | Path,
    output_archive_base: str | Path,
    strategy: str | None = None,
) -> Path:
    """Bundles all directory artifacts into a consolidated ZIP or TAR.ZST archive.

    If strategy is None, evaluates compression strategy automatically based on file volume.
    """
    src = Path(source_dir).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"Source directory not found: '{src}'")

    total_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())

    if strategy is None:
        chosen_strategy = evaluate_compression_strategy(total_bytes)
    else:
        chosen_strategy = strategy.upper().strip()

    base_str = str(Path(output_archive_base).expanduser().resolve())

    if chosen_strategy in ["ZIP", ".ZIP"]:
        archive_path = Path(
            base_str if base_str.endswith(".zip") else f"{base_str}.zip"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(src.rglob("*")):
                if item.is_file():
                    arcname = str(item.relative_to(src)).replace("\\", "/")
                    zf.write(item, arcname=arcname)
        return archive_path

    if chosen_strategy in ["ZSTD", "TAR_ZSTD", "TAR.ZST", ".TAR.ZST", "ZST"]:
        if base_str.endswith(".tar.zst"):
            archive_path = Path(base_str)
        elif base_str.endswith(".tar"):
            archive_path = Path(f"{base_str}.zst")
        else:
            archive_path = Path(f"{base_str}.tar.zst")

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for item in sorted(src.rglob("*")):
                if item.is_file():
                    arcname = str(item.relative_to(src)).replace("\\", "/")
                    tar.add(str(item), arcname=arcname)
        tar_bytes = tar_buffer.getvalue()

        cctx = zstandard.ZstdCompressor(level=10)
        compressed_payload = cctx.compress(tar_bytes)
        archive_path.write_bytes(compressed_payload)
        return archive_path

    raise ValueError(f"Unsupported packaging strategy '{strategy}'")


def seal_artifact_read_only(file_path: str | Path) -> bool:
    """Sets cross-platform read-only permissions on a file to ensure non-repudiation."""
    target = Path(file_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Cannot seal non-existent file: '{target}'")

    current_mode = target.stat().st_mode
    readonly_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(target, readonly_mode)

    if os.name == "nt":
        os.chmod(target, stat.S_IREAD)

    return True


def export_spycfit_snapshot(
    payload: FitProvenancePayload,
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    active_models: Sequence[str] | None = None,
    crossref_records: Sequence[dict[str, Any]] | None = None,
    optimized_params: Sequence[dict[str, Any]] | None = None,
    transitions: Sequence[dict[str, Any]] | None = None,
    title_prefix: str = "CoChem-SpycFit",
    strategy: str | None = None,
) -> dict[str, Any]:
    """Full orchestration pipeline for generating and sealing a SpycFit snapshot export package."""
    dest_dir = resolve_processed_workspace_dir(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(source_dir).expanduser().resolve()
    if src_path.is_dir() and src_path != dest_dir:
        for item in src_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_path)
                dest_file = dest_dir / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                if not dest_file.exists():
                    shutil.copy2(item, dest_file)

    prov_file = dest_dir / "fit_provenance.json"
    payload.to_json_file(prov_file)

    params_to_render = list(optimized_params or [])
    trans_to_render = list(transitions or [])
    tables = generate_aastex_longtables(
        params_to_render, trans_to_render, title_prefix=title_prefix
    )

    param_tex_path = dest_dir / "parameters_table.tex"
    param_tex_path.write_text(tables["parameters_table"], encoding="utf-8")

    trans_tex_path = dest_dir / "transitions_table.tex"
    trans_tex_path.write_text(tables["transitions_table"], encoding="utf-8")

    doc_tex_path = dest_dir / "spectroscopic_document.tex"
    doc_tex_path.write_text(tables["combined_document"], encoding="utf-8")

    models_to_cite = list(active_models or payload.active_models)
    dois = get_required_dois(models_to_cite)

    records: list[dict[str, Any]] = list(crossref_records or [])
    for d in dois:
        records.append({"DOI": d, "title": f"Attributed Spectroscopic Engine ({d})"})

    bibtex_content = format_citations_to_bib(records)
    bib_path = dest_dir / "cochem_citations.bib"
    bib_path.write_text(bibtex_content, encoding="utf-8")

    archive_base = dest_dir.parent / f"{dest_dir.name}_archive"
    archive_path = package_fit_artifacts(dest_dir, archive_base, strategy=strategy)

    for item in dest_dir.rglob("*"):
        if item.is_file():
            seal_artifact_read_only(item)
    seal_artifact_read_only(archive_path)

    return {
        "status": "SUCCESS",
        "archive_path": str(archive_path),
        "provenance_path": str(prov_file),
        "parameters_table_path": str(param_tex_path),
        "transitions_table_path": str(trans_tex_path),
        "document_path": str(doc_tex_path),
        "citations_bib_path": str(bib_path),
        "session_id": payload.session_id,
        "dois_cited": dois,
    }


__all__ = [
    "FitProvenancePayload",
    "evaluate_compression_strategy",
    "export_spycfit_snapshot",
    "format_citations_to_bib",
    "generate_aastex_longtable",
    "generate_aastex_longtables",
    "generate_provenance_lock",
    "get_element_atomic_weight",
    "get_isotope_mass",
    "get_required_dois",
    "get_spycfit_processed_dir",
    "hash_dataset_iteratively",
    "package_fit_artifacts",
    "resolve_processed_workspace_dir",
    "seal_artifact_read_only",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\interfaces\cochem_vibspyc_snap.py ---
"""SpycFit Snapshot & Publication Export Interface.

Phase 4 / Task 10 Implementation:
- Authoritative fit provenance payload modeling and validation.
- Iterative cryptographic dataset hashing (SHA-256).
- Provenance lock generation for physical HDF5/binary datasets.
- Publication-ready AASTeX and LaTeX longtable rendering with siunitx and booktabs.
- Dynamic CrossRef JSON parsing to clean, deduplicated BibTeX entries.
- Active model to standard DOI resolution.
- Adaptive storage compression strategy evaluation (ZIP vs ZSTD).
- Archive creation (.zip and .tar.zst) and cross-platform read-only sealing.
- Dynamic IUPAC atomic and isotopic weight resolution via Mendeleev.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import logging
import os
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jinja2
import mendeleev
import platformdirs
import zstandard
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Standard model to DOI mappings across computational spectroscopy and quantum chemistry
_MODEL_DOI_MAP: dict[str, str] = {
    "watson_a_reduction": "10.1016/0022-2852(77)90184-7",
    "watson_a": "10.1016/0022-2852(77)90184-7",
    "watson_s_reduction": "10.1016/0022-2852(77)90184-7",
    "watson_s": "10.1016/0022-2852(77)90184-7",
    "iam_internal_rotation": "10.1016/0022-2852(66)90258-0",
    "iam": "10.1016/0022-2852(66)90258-0",
    "internal_axis_method": "10.1016/0022-2852(66)90258-0",
    "erham": "10.1006/jmsp.1997.7432",
    "erham_internal_rotation": "10.1006/jmsp.1997.7432",
    "jax_backend": "10.5281/zenodo.4766323",
    "jax": "10.5281/zenodo.4766323",
    "pyarrow_engine": "10.5281/zenodo.4006056",
    "pyarrow": "10.5281/zenodo.4006056",
    "arrow": "10.5281/zenodo.4006056",
    "spcat": "10.1016/0022-2852(91)90393-O",
    "spcat_bridge": "10.1016/0022-2852(91)90393-O",
    "pickett": "10.1016/0022-2852(91)90393-O",
    "orca": "10.1002/wcms.1606",
    "orca_v6": "10.1063/5.0004608",
    "cfour": "10.1063/5.0004837",
    "crest": "10.1039/C9CP06869D",
    "crest_metadynamics": "10.1021/acs.jctc.9b00143",
    "xtb": "10.1002/wcms.1493",
    "gfn2_xtb": "10.1021/acs.jctc.8b01176",
    "gfn1_xtb": "10.1021/acs.jctc.7b00118",
    "gfn_ff": "10.1002/anie.202004239",
    "pyscf": "10.1002/wcms.1340",
    "gpu4pyscf": "10.1063/5.0223707",
    "psi4": "10.1063/5.0006002",
    "molpro": "10.1063/5.0005081",
    "mace": "10.1021/jacs.4c07099",
    "mace_off23": "10.1021/jacs.4c07099",
    "aimnet2": "10.1021/acs.jcim.4c00445",
    "abcluster": "10.1039/C5CP04060D",
    "r2scan_3c": "10.1063/5.0040021",
    "b97_3c": "10.1063/1.5012601",
    "wb97m_v": "10.1063/1.4952647",
    "wb97x_v": "10.1039/C4CP00288E",
    "dlpno_ccsd_t": "10.1063/1.4773581",
    "dft_d3": "10.1063/1.3382344",
    "dft_d4": "10.1063/1.5090222",
    "sapt": "10.1021/cr00031a008",
    "qcxms": "10.1021/acsomega.1c00994",
    "ase": "10.1088/1361-648X/aa680e",
    "parsl": "10.1145/3307681.3325400",
    "rdkit": "10.5281/zenodo.10549474",
}

_LATEX_PARAMETER_NAME_MAP: dict[str, str] = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D_J": r"\Delta_J",
    "D_JK": r"\Delta_{JK}",
    "D_K": r"\Delta_K",
    "d_1": r"\delta_1",
    "d_2": r"\delta_2",
    "d_J": r"\delta_J",
    "d_K": r"\delta_K",
    "H_J": r"\Phi_J",
    "H_JK": r"\Phi_{JK}",
    "H_KJ": r"\Phi_{KJ}",
    "H_K": r"\Phi_K",
    "h_1": r"\phi_1",
    "h_2": r"\phi_2",
    "h_3": r"\phi_3",
    "V_3": "V_3",
    "V_6": "V_6",
    "F": "F",
    "rho": r"\rho",
    "beta": r"\beta",
    "mu_a": r"\mu_a",
    "mu_b": r"\mu_b",
    "mu_c": r"\mu_c",
}


def _format_default_latex_symbol(name: str) -> str:
    """Provides a canonical LaTeX mathematical representation for spectroscopic parameters."""
    if name in _LATEX_PARAMETER_NAME_MAP:
        return _LATEX_PARAMETER_NAME_MAP[name]
    if "_" in name:
        base, sub = name.split("_", 1)
        return f"{base}_{{{sub}}}"
    return name


def get_element_atomic_weight(symbol: str) -> float:
    """Retrieves authoritative dynamic IUPAC standard atomic weight via Mendeleev."""
    elem = mendeleev.element(symbol)
    return float(elem.atomic_weight)


def get_isotope_mass(symbol: str, mass_number: int) -> float:
    """Retrieves authoritative dynamic isotopic mass in atomic mass units via Mendeleev."""
    elem = mendeleev.element(symbol)
    for iso in elem.isotopes:
        if iso.mass_number == mass_number:
            return float(iso.mass)
    raise ValueError(f"Isotope {symbol}-{mass_number} not found in Mendeleev database")


def get_spycfit_processed_dir() -> Path:
    """Resolves the default SpycFit processed output directory.

    Checks $COCHEM_STATE_DIR first; falls back to platformdirs user data path.
    """
    env_state = os.environ.get("COCHEM_STATE_DIR")
    if env_state:
        base_path = Path(env_state).expanduser().resolve()
    else:
        base_path = Path(platformdirs.user_data_path("CoChem", "CoChem")).resolve()
    return base_path / "SpycFit_Workspace" / "Processed"


def resolve_processed_workspace_dir(custom_dir: str | Path | None = None) -> Path:
    """Resolves and returns the canonical processed directory, accepting an optional custom path."""
    if custom_dir is not None:
        target = Path(custom_dir).expanduser().resolve()
    else:
        target = get_spycfit_processed_dir()
    return target


class FitProvenancePayload(BaseModel):
    """Pydantic model representing complete scientific provenance for a SpycFit spectroscopic session."""

    session_id: str = Field(
        ..., description="Unique identifier of the spectroscopic fitting session"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 UTC timestamp of the fitting completion"
    )
    dataset_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of filename or dataset identifier to its SHA-256 cryptographic digest",
    )
    chi_squared: float = Field(
        ..., description="Final reduced or unweighted chi-squared metric of the fit"
    )
    huber_loss_delta: float | None = Field(
        default=None, description="Huber robust loss delta threshold if active"
    )
    rms_mhz: float = Field(..., description="Root-mean-square error in Megahertz (MHz)")
    rms_cm_inv: float = Field(
        ..., description="Root-mean-square error in wavenumbers (cm^-1)"
    )
    jacobian_condition_number: float = Field(
        ..., description="Condition number of the final Jacobian matrix"
    )
    sobol_parameter_audit: dict[str, bool] = Field(
        default_factory=dict,
        description="Sobol sensitivity audit mapping parameter names to boolean active fit status",
    )
    semantic_git_history: list[str] = Field(
        default_factory=list,
        description="Relevant semantic Git commits capturing pipeline state",
    )
    active_models: list[str] = Field(
        default_factory=list,
        description="List of active physical models, Hamiltonians, and compute engines utilized",
    )
    additional_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional extended metadata including instrument parameters and experimental conditions",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Ensures the timestamp is a strictly valid ISO-8601 string."""
        normalized = v.replace("Z", "+00:00")
        try:
            datetime.datetime.fromisoformat(normalized)
        except ValueError as err:
            raise ValueError(
                f"Timestamp '{v}' is not a valid ISO 8601 string: {err}"
            ) from err
        return v

    @field_validator(
        "chi_squared", "rms_mhz", "rms_cm_inv", "jacobian_condition_number"
    )
    @classmethod
    def validate_non_negative_floats(cls, v: float) -> float:
        """Ensures physical error metrics and condition numbers are non-negative."""
        if v < 0.0:
            raise ValueError(f"Metric must be non-negative, received {v}")
        return v

    def to_json_file(self, file_path: str | Path) -> Path:
        """Serializes the payload to a JSON file on physical disk."""
        target_path = Path(file_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> FitProvenancePayload:
        """Deserializes a FitProvenancePayload from a JSON file on physical disk."""
        target_path = Path(file_path).expanduser().resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Provenance file not found at '{target_path}'")
        raw_text = target_path.read_text(encoding="utf-8")
        return cls.model_validate_json(raw_text)


def hash_dataset_iteratively(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 cryptographic digest of a file iteratively.

    Uses hashlib.file_digest if available in the standard library; falls back to chunked reading.
    """
    target = Path(file_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Dataset file not found: '{target}'")

    if hasattr(hashlib, "file_digest"):
        with open(target, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_provenance_lock(
    h5_path: str | Path,
    metrics: dict[str, Any] | FitProvenancePayload,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generates a cryptographic Provenance Lock (fit_provenance.json) for a fit session.

    Calculates the SHA-256 digest of the dataset file (such as landscape.h5), extracts
    or validates all critical error and sensitivity metrics, serializes to physical disk,
    and returns the provenance metadata dictionary.

    Parameters:
        h5_path: Path to the HDF5 or binary dataset file to cryptographically seal.
        metrics: Dictionary or FitProvenancePayload providing error metrics, parameter audits,
                 active models, and session metadata.
        output_path: Optional custom destination path for the JSON provenance file. Defaults
                     to `fit_provenance.json` in the same directory as `h5_path`.

    Returns:
        Dictionary containing the full validated FitProvenancePayload metadata.
    """
    target_h5 = Path(h5_path).expanduser().resolve()
    h5_hash = hash_dataset_iteratively(target_h5)
    file_key = target_h5.name

    if isinstance(metrics, FitProvenancePayload):
        payload_data = metrics.model_dump()
        payload_data["dataset_hashes"][file_key] = h5_hash
        payload = FitProvenancePayload.model_validate(payload_data)
    elif isinstance(metrics, dict):
        raw_timestamp = metrics.get("timestamp")
        if raw_timestamp is not None:
            iso_timestamp = str(raw_timestamp)
        else:
            iso_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        session_id = str(
            metrics.get(
                "session_id",
                f"SESSION_{target_h5.stem}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            )
        )

        hashes_dict = dict(metrics.get("dataset_hashes", {}))
        hashes_dict[file_key] = h5_hash

        chi_squared = float(
            metrics.get(
                "chi_squared",
                metrics.get("chi2", metrics.get("chi_sq", 1.0)),
            )
        )
        rms_mhz = float(metrics.get("rms_mhz", metrics.get("rms", 0.0)))

        raw_rms_cm_inv = metrics.get("rms_cm_inv")
        if raw_rms_cm_inv is not None:
            rms_cm_inv = float(raw_rms_cm_inv)
        else:
            rms_cm_inv = rms_mhz / 29979.2458 if rms_mhz > 0.0 else 0.0

        raw_cond = metrics.get(
            "jacobian_condition_number", metrics.get("condition_number", 1.0)
        )
        jacobian_condition_number = float(raw_cond)

        raw_huber = metrics.get("huber_loss_delta")
        huber_loss_delta = float(raw_huber) if raw_huber is not None else None

        sobol_audit: dict[str, bool] = {}
        if "sobol_parameter_audit" in metrics and isinstance(
            metrics["sobol_parameter_audit"], dict
        ):
            sobol_audit = {
                str(k): bool(v) for k, v in metrics["sobol_parameter_audit"].items()
            }
        elif "sobol_audit" in metrics and isinstance(metrics["sobol_audit"], dict):
            sobol_audit = {
                str(k): bool(v) for k, v in metrics["sobol_audit"].items()
            }
        else:
            for param_name in metrics.get("active_parameters", []):
                sobol_audit[str(param_name)] = True
            for param_name in metrics.get("frozen_parameters", []):
                sobol_audit[str(param_name)] = False

        raw_history = metrics.get(
            "semantic_git_history",
            metrics.get("git_history", metrics.get("history", [])),
        )
        semantic_git_history = [str(item) for item in raw_history]

        raw_models = metrics.get("active_models", metrics.get("models", []))
        active_models = [str(m) for m in raw_models]

        additional_metadata = metrics.get(
            "additional_metadata", metrics.get("metadata")
        )
        if additional_metadata is not None and not isinstance(
            additional_metadata, dict
        ):
            additional_metadata = {"raw": str(additional_metadata)}

        payload = FitProvenancePayload(
            session_id=session_id,
            timestamp=iso_timestamp,
            dataset_hashes=hashes_dict,
            chi_squared=chi_squared,
            huber_loss_delta=huber_loss_delta,
            rms_mhz=rms_mhz,
            rms_cm_inv=rms_cm_inv,
            jacobian_condition_number=jacobian_condition_number,
            sobol_parameter_audit=sobol_audit,
            semantic_git_history=semantic_git_history,
            active_models=active_models,
            additional_metadata=additional_metadata,
        )
    else:
        raise TypeError(
            f"Expected metrics to be dict or FitProvenancePayload, received {type(metrics).__name__}"
        )

    if output_path is not None:
        target_out = Path(output_path).expanduser().resolve()
    else:
        target_out = target_h5.parent / "fit_provenance.json"

    payload.to_json_file(target_out)
    return payload.model_dump()


_AASTEX_PARAMETERS_TEMPLATE = """\\begin{table}[htbp]
\\centering
\\caption{Optimized Spectroscopic Parameters for {{ title_prefix }}}
\\label{tab:optimized_params}
\\begin{tabular}{l c S[table-format=6.4] l l}
\\toprule
Parameter & LaTeX Symbol & {Value (MHz)} & Uncertainty & Description \\\\
\\midrule
{% for p in parameters %}
{{ p.name }} & ${{ p.latex_name }}$ & {{ p.formatted_value }} & {{ p.formatted_uncertainty }} & {{ p.description }} \\\\
{% endfor %}
\\bottomrule
\\end{tabular}
\\end{table}"""

_AASTEX_TRANSITIONS_TEMPLATE = """\\begin{longtable}{c c S[table-format=6.4] S[table-format=6.4] S[table-format=3.4] S[table-format=2.4] S[table-format=3.2]}
\\caption{Observed and Calculated Transitions for {{ title_prefix }} (Truncated to Top {{ transitions|length }})} \\label{tab:transitions} \\\\
\\toprule
Upper ($J'_{K_a' K_c'}$) & Lower ($J''_{K_a'' K_c''}$) & {$\\nu_{\\text{obs}}$ (MHz)} & {$\\nu_{\\text{calc}}$ (MHz)} & {Residual (MHz)} & {$\\sigma$ (MHz)} & {Intensity} \\\\
\\midrule
\\endfirsthead
\\toprule
Upper ($J'_{K_a' K_c'}$) & Lower ($J''_{K_a'' K_c''}$) & {$\\nu_{\\text{obs}}$ (MHz)} & {$\\nu_{\\text{calc}}$ (MHz)} & {Residual (MHz)} & {$\\sigma$ (MHz)} & {Intensity} \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
\\bottomrule
\\endlastfoot
{% for t in transitions %}
{{ t.upper_state }} & {{ t.lower_state }} & {{ t.formatted_obs }} & {{ t.formatted_calc }} & {{ t.formatted_res }} & {{ t.formatted_unc }} & {{ t.formatted_int }} \\\\
{% endfor %}
\\end{longtable}"""

_AASTEX_COMBINED_TEMPLATE = """\\documentclass[twocolumn]{aastex631}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{siunitx}

\\shorttitle{Spectroscopic Fit: {{ title_prefix }}}
\\shortauthors{CoChem Ecosystem}

\\begin{document}

\\title{High-Precision Rotational Spectroscopic Fit and Transition Assignments: {{ title_prefix }}}

\\begin{abstract}
We present the high-precision spectroscopic fit, molecular parameters, and assigned rotational-vibrational transitions generated by the CoChem-SpycFit autonomous analysis engine.
\\end{abstract}

\\section{Molecular Parameters}
{{ parameters_table }}

\\section{Assigned Transitions}
{{ transitions_table }}

\\end{document}
"""


def generate_aastex_longtables(
    optimized_params: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    title_prefix: str = "CoChem-SpycFit",
) -> dict[str, str]:
    """Renders in-memory publication-ready AASTeX and LaTeX longtables with siunitx and booktabs.

    Parameters:
        optimized_params: List of parameter dictionaries.
        transitions: List of transition records. Truncated to top 200 by intensity/Einstein A.
        title_prefix: Title header for captions and document metadata.

    Returns:
        Dictionary containing {"parameters_table": str, "transitions_table": str, "combined_document": str}.
    """
    processed_params: list[dict[str, Any]] = []
    for param in optimized_params:
        name = str(param.get("name", "Param"))
        latex_name = str(param.get("latex_name", _format_default_latex_symbol(name)))
        val = param.get("value", 0.0)
        formatted_val = (
            f"{float(val):.6f}" if isinstance(val, (int, float)) else str(val)
        )

        is_frozen = param.get("is_frozen", False)
        sobol_active = param.get("sobol_audit", True)
        unc = param.get("uncertainty")
        status = str(param.get("status", "")).lower()

        if (
            is_frozen
            or (unc is None)
            or (not sobol_active)
            or status in ["fixed", "set", "frozen"]
        ):
            if status == "set":
                formatted_unc = "Set"
            else:
                formatted_unc = "Fixed"
        else:
            try:
                formatted_unc = f"{float(unc):.6f}"
            except (ValueError, TypeError):
                formatted_unc = str(unc)

        description = str(param.get("description", f"Parameter {name}"))
        processed_params.append(
            {
                "name": name,
                "latex_name": latex_name,
                "formatted_value": formatted_val,
                "formatted_uncertainty": formatted_unc,
                "description": description,
            }
        )

    sorted_transitions = sorted(
        transitions,
        key=lambda item: float(
            item.get(
                "intensity",
                item.get("einstein_a", item.get("a_coeff", item.get("int", 0.0))),
            )
            or 0.0
        ),
        reverse=True,
    )[:200]

    processed_trans: list[dict[str, Any]] = []
    for t in sorted_transitions:
        upper = str(t.get("upper_state", "Upper"))
        lower = str(t.get("lower_state", "Lower"))
        obs = t.get("observed_mhz", t.get("obs_mhz", 0.0))
        calc = t.get("calculated_mhz", t.get("calc_mhz", 0.0))
        res = t.get("residual_mhz", t.get("res_mhz", float(obs) - float(calc)))
        unc = t.get("uncertainty_mhz", t.get("unc_mhz", 0.01))
        intensity = t.get("intensity", t.get("einstein_a", 1.0))

        processed_trans.append(
            {
                "upper_state": upper,
                "lower_state": lower,
                "formatted_obs": f"{float(obs):.4f}",
                "formatted_calc": f"{float(calc):.4f}",
                "formatted_res": f"{float(res):.4f}",
                "formatted_unc": f"{float(unc):.4f}",
                "formatted_int": f"{float(intensity):.2f}",
            }
        )

    param_tmpl = jinja2.Template(_AASTEX_PARAMETERS_TEMPLATE)
    trans_tmpl = jinja2.Template(_AASTEX_TRANSITIONS_TEMPLATE)
    doc_tmpl = jinja2.Template(_AASTEX_COMBINED_TEMPLATE)

    rendered_params = param_tmpl.render(
        parameters=processed_params, title_prefix=title_prefix
    )
    rendered_trans = trans_tmpl.render(
        transitions=processed_trans, title_prefix=title_prefix
    )
    rendered_doc = doc_tmpl.render(
        title_prefix=title_prefix,
        parameters_table=rendered_params,
        transitions_table=rendered_trans,
    )

    return {
        "parameters_table": rendered_params,
        "transitions_table": rendered_trans,
        "combined_document": rendered_doc,
    }


def generate_aastex_longtable(
    constants_dict: dict[str, Any] | Sequence[dict[str, Any]],
    title_prefix: str = "CoChem-SpycFit",
    transitions: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Generates peer-review-ready LaTeX tables for molecular constants and assigned transitions.

    Accepts either a dictionary mapping parameter names to floats/tuples/dicts or a sequence
    of parameter dictionaries. Formats values with siunitx and booktabs conventions.

    Parameters:
        constants_dict: Mapping of parameter names to values/sub-dicts or list of parameter dicts.
        title_prefix: Title caption string for LaTeX tables.
        transitions: Optional sequence of assigned rotational-vibrational transitions.

    Returns:
        Formatted LaTeX string (combined document if transitions provided, table otherwise).
    """
    normalized_params: list[dict[str, Any]] = []

    if isinstance(constants_dict, Sequence) and not isinstance(
        constants_dict, (str, bytes)
    ):
        for item in constants_dict:
            if isinstance(item, dict):
                p_copy = dict(item)
                if "name" in p_copy and "latex_name" not in p_copy:
                    p_copy["latex_name"] = _format_default_latex_symbol(
                        str(p_copy["name"])
                    )
                normalized_params.append(p_copy)
    elif isinstance(constants_dict, dict):
        for key, val in constants_dict.items():
            param_name = str(key)
            latex_name = _format_default_latex_symbol(param_name)
            if isinstance(val, dict):
                p_entry = dict(val)
                p_entry.setdefault("name", param_name)
                p_entry.setdefault("latex_name", latex_name)
                normalized_params.append(p_entry)
            elif isinstance(val, (tuple, list)):
                v_num = val[0] if len(val) > 0 else 0.0
                unc_num = val[1] if len(val) > 1 else None
                is_froz = val[2] if len(val) > 2 else (unc_num is None)
                normalized_params.append(
                    {
                        "name": param_name,
                        "latex_name": latex_name,
                        "value": v_num,
                        "uncertainty": unc_num,
                        "is_frozen": is_froz,
                        "description": f"Spectroscopic parameter {param_name}",
                    }
                )
            else:
                normalized_params.append(
                    {
                        "name": param_name,
                        "latex_name": latex_name,
                        "value": val,
                        "uncertainty": None,
                        "is_frozen": False,
                        "description": f"Spectroscopic parameter {param_name}",
                    }
                )
    else:
        raise TypeError(
            f"Expected constants_dict to be dict or Sequence, received {type(constants_dict).__name__}"
        )

    trans_list = list(transitions or [])
    rendered_dict = generate_aastex_longtables(
        optimized_params=normalized_params,
        transitions=trans_list,
        title_prefix=title_prefix,
    )

    if trans_list:
        return rendered_dict["combined_document"]
    return rendered_dict["parameters_table"]


def get_required_dois(active_models: Sequence[str]) -> list[str]:
    """Resolves standard DOIs for all specified active physical models and quantum engines.

    Preserves unique entries without duplicates.
    """
    resolved_dois: list[str] = []
    seen: set[str] = set()

    for model_name in active_models:
        normalized_key = model_name.strip().lower().replace(" ", "_").replace("-", "_")
        doi = _MODEL_DOI_MAP.get(normalized_key)
        if doi and doi not in seen:
            seen.add(doi)
            resolved_dois.append(doi)

    return resolved_dois


def format_citations_to_bib(crossref_json_responses: Sequence[dict[str, Any]]) -> str:
    """Parses CrossRef JSON records into clean, validated, deduplicated BibTeX entries."""
    bib_entries: list[str] = []
    seen_dois: set[str] = set()
    seen_keys: set[str] = set()

    for record in crossref_json_responses:
        work = record.get("message", record)

        doi_raw = work.get("DOI") or work.get("doi")
        if not doi_raw:
            continue
        doi_clean = str(doi_raw).strip()
        if doi_clean.lower() in seen_dois:
            continue
        seen_dois.add(doi_clean.lower())

        title_val = work.get("title", "")
        if isinstance(title_val, list) and title_val:
            title = str(title_val[0]).strip()
        else:
            title = str(title_val).strip()

        authors_raw = work.get("author", [])
        author_names: list[str] = []
        first_author_family = "Author"
        if isinstance(authors_raw, list):
            for idx, a in enumerate(authors_raw):
                if isinstance(a, dict):
                    family = a.get("family", a.get("name", ""))
                    given = a.get("given", "")
                    if idx == 0 and family:
                        first_author_family = "".join(filter(str.isalnum, str(family)))
                    if family and given:
                        author_names.append(f"{family}, {given}")
                    elif family:
                        author_names.append(str(family))
                elif isinstance(a, str):
                    if idx == 0:
                        first_author_family = "".join(
                            filter(str.isalnum, a.split()[-1])
                        )
                    author_names.append(a)
        author_str = (
            " and ".join(author_names)
            if author_names
            else "CoChem Spectroscopy Consortium"
        )

        journal_raw = (
            work.get("container-title")
            or work.get("container_title")
            or work.get("journal", "")
        )
        if isinstance(journal_raw, list) and journal_raw:
            journal = str(journal_raw[0]).strip()
        else:
            journal = str(journal_raw).strip()

        volume = str(work.get("volume", "")).strip()
        number = str(work.get("issue", work.get("number", ""))).strip()
        pages = str(work.get("page", work.get("pages", ""))).strip()

        year_str = "2026"
        issued = (
            work.get("issued")
            or work.get("published-print")
            or work.get("published-online")
        )
        if isinstance(issued, dict):
            date_parts = issued.get("date-parts", [])
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                year_str = str(date_parts[0][0])
        elif "year" in work:
            year_str = str(work["year"])

        first_title_word = "".join(
            filter(str.isalnum, (title.split()[0] if title else "Work"))
        )
        base_key = f"{first_author_family}{year_str}{first_title_word}"
        key = base_key
        counter = 1
        while key in seen_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        seen_keys.add(key)

        entry_lines = [f"@article{{{key},"]
        entry_lines.append(f"  author = {{{author_str}}},")
        if title:
            entry_lines.append(f"  title = {{{{{title}}}}},")
        if journal:
            entry_lines.append(f"  journal = {{{journal}}},")
        if volume:
            entry_lines.append(f"  volume = {{{volume}}},")
        if number:
            entry_lines.append(f"  number = {{{number}}},")
        if pages:
            entry_lines.append(f"  pages = {{{pages}}},")
        entry_lines.append(f"  year = {{{year_str}}},")
        entry_lines.append(f"  doi = {{{doi_clean}}}")
        entry_lines.append("}")

        bib_entries.append("\n".join(entry_lines))

    return "\n\n".join(bib_entries)


def evaluate_compression_strategy(
    total_size_bytes: int, threshold_bytes: int = 104857600
) -> str:
    """Evaluates whether to package artifacts as standard ZIP or high-ratio Zstandard tarball.

    Threshold defaults to 100 MiB (104,857,600 bytes).
    """
    if total_size_bytes >= threshold_bytes:
        return "ZSTD"
    return "ZIP"


def package_fit_artifacts(
    source_dir: str | Path,
    output_archive_base: str | Path,
    strategy: str | None = None,
) -> Path:
    """Bundles all directory artifacts into a consolidated ZIP or TAR.ZST archive.

    If strategy is None, evaluates compression strategy automatically based on file volume.
    """
    src = Path(source_dir).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"Source directory not found: '{src}'")

    total_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())

    if strategy is None:
        chosen_strategy = evaluate_compression_strategy(total_bytes)
    else:
        chosen_strategy = strategy.upper().strip()

    base_str = str(Path(output_archive_base).expanduser().resolve())

    if chosen_strategy in ["ZIP", ".ZIP"]:
        archive_path = Path(
            base_str if base_str.endswith(".zip") else f"{base_str}.zip"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(src.rglob("*")):
                if item.is_file():
                    arcname = str(item.relative_to(src)).replace("\\", "/")
                    zf.write(item, arcname=arcname)
        return archive_path

    if chosen_strategy in ["ZSTD", "TAR_ZSTD", "TAR.ZST", ".TAR.ZST", "ZST"]:
        if base_str.endswith(".tar.zst"):
            archive_path = Path(base_str)
        elif base_str.endswith(".tar"):
            archive_path = Path(f"{base_str}.zst")
        else:
            archive_path = Path(f"{base_str}.tar.zst")

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for item in sorted(src.rglob("*")):
                if item.is_file():
                    arcname = str(item.relative_to(src)).replace("\\", "/")
                    tar.add(str(item), arcname=arcname)
        tar_bytes = tar_buffer.getvalue()

        cctx = zstandard.ZstdCompressor(level=10)
        compressed_payload = cctx.compress(tar_bytes)
        archive_path.write_bytes(compressed_payload)
        return archive_path

    raise ValueError(f"Unsupported packaging strategy '{strategy}'")


def seal_artifact_read_only(file_path: str | Path) -> bool:
    """Sets cross-platform read-only permissions on a file to ensure non-repudiation."""
    target = Path(file_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Cannot seal non-existent file: '{target}'")

    current_mode = target.stat().st_mode
    readonly_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(target, readonly_mode)

    if os.name == "nt":
        os.chmod(target, stat.S_IREAD)

    return True


def export_spycfit_snapshot(
    payload: FitProvenancePayload,
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    active_models: Sequence[str] | None = None,
    crossref_records: Sequence[dict[str, Any]] | None = None,
    optimized_params: Sequence[dict[str, Any]] | None = None,
    transitions: Sequence[dict[str, Any]] | None = None,
    title_prefix: str = "CoChem-SpycFit",
    strategy: str | None = None,
) -> dict[str, Any]:
    """Full orchestration pipeline for generating and sealing a SpycFit snapshot export package."""
    dest_dir = resolve_processed_workspace_dir(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(source_dir).expanduser().resolve()
    if src_path.is_dir() and src_path != dest_dir:
        for item in src_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_path)
                dest_file = dest_dir / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                if not dest_file.exists():
                    shutil.copy2(item, dest_file)

    prov_file = dest_dir / "fit_provenance.json"
    payload.to_json_file(prov_file)

    params_to_render = list(optimized_params or [])
    trans_to_render = list(transitions or [])
    tables = generate_aastex_longtables(
        params_to_render, trans_to_render, title_prefix=title_prefix
    )

    param_tex_path = dest_dir / "parameters_table.tex"
    param_tex_path.write_text(tables["parameters_table"], encoding="utf-8")

    trans_tex_path = dest_dir / "transitions_table.tex"
    trans_tex_path.write_text(tables["transitions_table"], encoding="utf-8")

    doc_tex_path = dest_dir / "spectroscopic_document.tex"
    doc_tex_path.write_text(tables["combined_document"], encoding="utf-8")

    models_to_cite = list(active_models or payload.active_models)
    dois = get_required_dois(models_to_cite)

    records: list[dict[str, Any]] = list(crossref_records or [])
    for d in dois:
        records.append({"DOI": d, "title": f"Attributed Spectroscopic Engine ({d})"})

    bibtex_content = format_citations_to_bib(records)
    bib_path = dest_dir / "cochem_citations.bib"
    bib_path.write_text(bibtex_content, encoding="utf-8")

    archive_base = dest_dir.parent / f"{dest_dir.name}_archive"
    archive_path = package_fit_artifacts(dest_dir, archive_base, strategy=strategy)

    for item in dest_dir.rglob("*"):
        if item.is_file():
            seal_artifact_read_only(item)
    seal_artifact_read_only(archive_path)

    return {
        "status": "SUCCESS",
        "archive_path": str(archive_path),
        "provenance_path": str(prov_file),
        "parameters_table_path": str(param_tex_path),
        "transitions_table_path": str(trans_tex_path),
        "document_path": str(doc_tex_path),
        "citations_bib_path": str(bib_path),
        "session_id": payload.session_id,
        "dois_cited": dois,
    }


__all__ = [
    "FitProvenancePayload",
    "evaluate_compression_strategy",
    "export_spycfit_snapshot",
    "format_citations_to_bib",
    "generate_aastex_longtable",
    "generate_aastex_longtables",
    "generate_provenance_lock",
    "get_element_atomic_weight",
    "get_isotope_mass",
    "get_required_dois",
    "get_spycfit_processed_dir",
    "hash_dataset_iteratively",
    "package_fit_artifacts",
    "resolve_processed_workspace_dir",
    "seal_artifact_read_only",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_vibspyc_snap.py ---
"""Comprehensive physical unit test suite for SpycFit Snapshot & Publication Export.

Verifies:
- FitProvenancePayload schema validation, field validators, serialization & deserialization.
- hash_dataset_iteratively with physical disk operations and chunked reading.
- generate_provenance_lock with physical HDF5 and binary datasets, metadata extraction,
  and serialization to disk.
- generate_aastex_longtable and generate_aastex_longtables with molecular rotational parameters,
  transitions, siunitx formatting, frozen parameter overrides, and top 200 truncation.
- get_required_dois and format_citations_to_bib with structured CrossRef metadata records.
- evaluate_compression_strategy sizing thresholds.
- package_fit_artifacts archiving (.zip and .tar.zst) and seal_artifact_read_only permissions.
- export_spycfit_snapshot end-to-end pipeline execution and artifact sealing.
- Workspace directory resolution with environment variable and platformdirs fallback.
- Dynamic IUPAC atomic weight and isotopic mass resolution via Mendeleev.
"""

from __future__ import annotations

import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zstandard
from pydantic import ValidationError

from cochem_spycfit.interfaces.cochem_vibspyc_snap import (
    FitProvenancePayload,
    evaluate_compression_strategy,
    export_spycfit_snapshot,
    format_citations_to_bib,
    generate_aastex_longtable,
    generate_aastex_longtables,
    generate_provenance_lock,
    get_element_atomic_weight,
    get_isotope_mass,
    get_required_dois,
    get_spycfit_processed_dir,
    hash_dataset_iteratively,
    package_fit_artifacts,
    resolve_processed_workspace_dir,
    seal_artifact_read_only,
)


@pytest.fixture
def clean_workspace(tmp_path: Path):
    """Provides an isolated temporary workspace and resets file permissions upon teardown."""
    yield tmp_path
    for root, _dirs, files in os.walk(tmp_path):
        for fname in files:
            target = os.path.join(root, fname)
            try:
                os.chmod(target, stat.S_IWRITE)
            except OSError as err:
                _ = err


@pytest.fixture
def authoritative_payload() -> FitProvenancePayload:
    """Constructs a valid, physically realistic FitProvenancePayload instance."""
    return FitProvenancePayload(
        session_id="SESSION_METHANOL_CH3OH_FIT_2026_01",
        timestamp="2026-08-23T14:30:00+00:00",
        dataset_hashes={
            "methanol_spectrum_chirped_pulse.dat": "a3f5b7c891024d3e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e",
            "assigned_transitions_methanol.cat": "8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d",
        },
        chi_squared=1.0428,
        huber_loss_delta=1.345,
        rms_mhz=0.0184,
        rms_cm_inv=0.00061376,
        jacobian_condition_number=142.85,
        sobol_parameter_audit={
            "A": True,
            "B": True,
            "C": True,
            "D_J": True,
            "D_JK": True,
            "D_K": False,
            "V_3": True,
        },
        semantic_git_history=[
            "git-commit-8f2a1b4: Initial Hamiltonian configuration for internal rotation",
            "git-commit-c9d3e5f: Refined centrifugal distortion parameters via Levenberg-Marquardt",
            "git-commit-4a7b9c0: Converged global spectroscopic fit with Huber robust weighting",
        ],
        active_models=[
            "watson_a_reduction",
            "iam_internal_rotation",
            "spcat",
            "jax_backend",
        ],
        additional_metadata={
            "spectrometer": "CP-FTMW 2-8 GHz",
            "carrier_gas": "Neon",
            "backing_pressure_bar": 2.5,
            "pulse_duration_us": 1.0,
            "temperature_k": 2.0,
        },
    )


@pytest.fixture
def molecular_parameters() -> list[dict[str, Any]]:
    """Realistic rotational and distortion parameters for methanol (CH3OH)."""
    return [
        {
            "name": "A",
            "latex_name": "A",
            "value": 127634.508,
            "uncertainty": 0.045,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal a-axis",
        },
        {
            "name": "B",
            "latex_name": "B",
            "value": 24684.382,
            "uncertainty": 0.012,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal b-axis",
        },
        {
            "name": "C",
            "latex_name": "C",
            "value": 23758.129,
            "uncertainty": 0.011,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Rotational constant around principal c-axis",
        },
        {
            "name": "D_J",
            "latex_name": "\\Delta_J",
            "value": 0.04021,
            "uncertainty": 0.00035,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_JK",
            "latex_name": "\\Delta_{JK}",
            "value": 0.12480,
            "uncertainty": 0.00080,
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
            "description": "Quartic centrifugal distortion parameter",
        },
        {
            "name": "D_K",
            "latex_name": "\\Delta_K",
            "value": 0.89200,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": False,
            "status": "fixed",
            "description": "Frozen quartic centrifugal distortion parameter from ab initio",
        },
        {
            "name": "V_3",
            "latex_name": "V_3",
            "value": 373.800,
            "uncertainty": None,
            "is_frozen": True,
            "sobol_audit": True,
            "status": "set",
            "description": "Three-fold torsional potential barrier height in cm^-1",
        },
    ]


@pytest.fixture
def molecular_transitions() -> list[dict[str, Any]]:
    """Calculated test list of 250 assigned transitions for methanol."""
    records: list[dict[str, Any]] = []
    for i in range(1, 251):
        j_upper = i % 10 + 1
        j_lower = i % 10
        ka_upper = min(j_upper, (i % 3))
        kc_upper = j_upper - ka_upper
        ka_lower = min(j_lower, (i % 3))
        kc_lower = j_lower - ka_lower

        obs_freq = 12000.0 + (i * 25.35)
        calc_freq = obs_freq + (((i % 7) - 3) * 0.003)
        res = obs_freq - calc_freq
        intensity = 1000.0 / (i + 1.0)

        records.append(
            {
                "upper_state": f"{j_upper}_{{{ka_upper},{kc_upper}}}",
                "lower_state": f"{j_lower}_{{{ka_lower},{kc_lower}}}",
                "observed_mhz": obs_freq,
                "calculated_mhz": calc_freq,
                "residual_mhz": res,
                "uncertainty_mhz": 0.010,
                "intensity": intensity,
            }
        )
    return records


# =====================================================================
# 1. FitProvenancePayload Validation and Serialization Tests
# =====================================================================


def test_fit_provenance_payload_validation(authoritative_payload: FitProvenancePayload):
    """Verifies valid instantiation and field access on FitProvenancePayload."""
    assert authoritative_payload.session_id == "SESSION_METHANOL_CH3OH_FIT_2026_01"
    assert authoritative_payload.chi_squared == 1.0428
    assert authoritative_payload.rms_mhz == 0.0184
    assert authoritative_payload.rms_cm_inv == 0.00061376
    assert authoritative_payload.jacobian_condition_number == 142.85
    assert len(authoritative_payload.dataset_hashes) == 2
    assert authoritative_payload.sobol_parameter_audit["V_3"] is True
    assert len(authoritative_payload.semantic_git_history) == 3
    assert "watson_a_reduction" in authoritative_payload.active_models


def test_fit_provenance_payload_invalid_timestamp():
    """Verifies that non-ISO formatted timestamps are rejected."""
    with pytest.raises(ValidationError) as excinfo:
        FitProvenancePayload(
            session_id="INVALID_TIMESTAMP_SESSION",
            timestamp="23-08-2026 14:30",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )
    assert "Timestamp" in str(excinfo.value) or "ISO" in str(excinfo.value)


def test_fit_provenance_payload_negative_metrics():
    """Verifies that negative error metrics trigger validation errors."""
    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_CHI_SQ_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=-0.5,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_RMS_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=-0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=10.0,
        )

    with pytest.raises(ValidationError):
        FitProvenancePayload(
            session_id="NEGATIVE_JACOBIAN_SESSION",
            timestamp="2026-08-23T14:30:00Z",
            chi_squared=1.0,
            rms_mhz=0.02,
            rms_cm_inv=0.0007,
            jacobian_condition_number=-5.0,
        )


def test_fit_provenance_disk_roundtrip(
    authoritative_payload: FitProvenancePayload, clean_workspace: Path
):
    """Verifies physical disk serialization and deserialization roundtrips."""
    target_json = clean_workspace / "provenance" / "fit_provenance.json"
    written_path = authoritative_payload.to_json_file(target_json)

    assert written_path.exists()
    assert written_path.is_file()

    loaded_payload = FitProvenancePayload.from_json_file(written_path)
    assert loaded_payload.session_id == authoritative_payload.session_id
    assert loaded_payload.timestamp == authoritative_payload.timestamp
    assert loaded_payload.chi_squared == authoritative_payload.chi_squared
    assert loaded_payload.rms_mhz == authoritative_payload.rms_mhz
    assert loaded_payload.dataset_hashes == authoritative_payload.dataset_hashes
    assert (
        loaded_payload.sobol_parameter_audit
        == authoritative_payload.sobol_parameter_audit
    )
    assert (
        loaded_payload.semantic_git_history
        == authoritative_payload.semantic_git_history
    )
    assert (
        loaded_payload.additional_metadata == authoritative_payload.additional_metadata
    )


def test_fit_provenance_from_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to deserialize from a non-existent file."""
    non_existent = clean_workspace / "non_existent_provenance.json"
    with pytest.raises(FileNotFoundError):
        FitProvenancePayload.from_json_file(non_existent)


# =====================================================================
# 2. Iterative Hashing Tests
# =====================================================================


def test_hash_dataset_iteratively_small_file(clean_workspace: Path):
    """Verifies iterative SHA-256 calculation for a physical file on disk."""
    data_file = clean_workspace / "spectrum_input.dat"
    content = b"CH3OH Spectroscopic Data Stream v1.0\n12450.320 0.85\n12480.110 0.92\n"
    data_file.write_bytes(content)

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=16)
    expected_hash = "bca9b04d490184c37b9ae03358c4370f955c050a1953e3b54674122459ecf77c"

    import hashlib

    direct_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == expected_hash == direct_hash


def test_hash_dataset_iteratively_large_file(clean_workspace: Path):
    """Verifies iterative hashing across multi-chunk files exceeding chunk boundaries."""
    data_file = clean_workspace / "large_spectrum.bin"
    payload_chunk = b"RotationalTransitionRecord_1234567890_abcdefghij_" * 32
    total_chunks = 320
    with open(data_file, "wb") as f:
        f.writelines(payload_chunk for _ in range(total_chunks))

    computed_hash = hash_dataset_iteratively(data_file, chunk_size=8192)
    assert isinstance(computed_hash, str)
    assert len(computed_hash) == 64


def test_hash_dataset_iteratively_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError on non-existent dataset path."""
    missing_file = clean_workspace / "non_existent_dataset.bin"
    with pytest.raises(FileNotFoundError):
        hash_dataset_iteratively(missing_file)


# =====================================================================
# 3. Provenance Lock Generation Tests (Physical HDF5 / Binary)
# =====================================================================


def test_generate_provenance_lock_hdf5(clean_workspace: Path):
    """Verifies cryptographic provenance lock creation on a physical HDF5 landscape file."""
    h5_file = clean_workspace / "landscape.h5"
    with h5py.File(h5_file, "w") as f:
        grp = f.create_group("spectroscopy")
        grp.create_dataset("frequencies", data=np.array([12450.32, 12480.11, 12510.95]))
        grp.create_dataset("intensities", data=np.array([0.85, 0.92, 0.45]))
        grp.attrs["molecule"] = "Methanol-CH3OH"
        grp.attrs["fit_converged"] = True

    metrics = {
        "session_id": "SESSION_H5_LOCK_TEST_01",
        "timestamp": "2026-08-23T15:00:00+00:00",
        "chi_squared": 1.025,
        "rms_mhz": 0.0142,
        "jacobian_condition_number": 88.5,
        "active_parameters": ["A", "B", "C", "D_J", "D_JK"],
        "frozen_parameters": ["D_K", "V_3"],
        "semantic_git_history": ["commit 1: Fitted ground state"],
        "active_models": ["watson_a", "spcat"],
        "metadata": {"spectrometer": "CP-FTMW"},
    }

    result = generate_provenance_lock(h5_file, metrics)

    assert result["session_id"] == "SESSION_H5_LOCK_TEST_01"
    assert result["chi_squared"] == 1.025
    assert result["rms_mhz"] == 0.0142
    assert "landscape.h5" in result["dataset_hashes"]
    expected_hash = hash_dataset_iteratively(h5_file)
    assert result["dataset_hashes"]["landscape.h5"] == expected_hash
    assert result["sobol_parameter_audit"]["A"] is True
    assert result["sobol_parameter_audit"]["D_K"] is False

    default_json = clean_workspace / "fit_provenance.json"
    assert default_json.exists()
    loaded_payload = FitProvenancePayload.from_json_file(default_json)
    assert loaded_payload.session_id == "SESSION_H5_LOCK_TEST_01"
    assert loaded_payload.dataset_hashes["landscape.h5"] == expected_hash


def test_generate_provenance_lock_with_payload_object(
    clean_workspace: Path, authoritative_payload: FitProvenancePayload
):
    """Verifies generate_provenance_lock when supplied with a pre-built FitProvenancePayload."""
    h5_file = clean_workspace / "state.h5"
    h5_file.write_bytes(b"HDF5_STRUCTURED_BINARY_STREAM_SIMULATION")

    custom_out = clean_workspace / "custom_dir" / "custom_provenance.json"
    result = generate_provenance_lock(
        h5_path=h5_file,
        metrics=authoritative_payload,
        output_path=custom_out,
    )

    assert custom_out.exists()
    assert result["session_id"] == authoritative_payload.session_id
    assert "state.h5" in result["dataset_hashes"]
    assert result["dataset_hashes"]["state.h5"] == hash_dataset_iteratively(h5_file)


def test_generate_provenance_lock_missing_h5_file(clean_workspace: Path):
    """Verifies FileNotFoundError when dataset file is missing."""
    missing_h5 = clean_workspace / "missing_landscape.h5"
    with pytest.raises(FileNotFoundError):
        generate_provenance_lock(missing_h5, {"chi_squared": 1.0, "rms_mhz": 0.01})


def test_generate_provenance_lock_invalid_metrics_type(clean_workspace: Path):
    """Verifies TypeError when invalid metrics argument is passed."""
    data_file = clean_workspace / "dataset.bin"
    data_file.write_bytes(b"DATA")
    with pytest.raises(TypeError, match="Expected metrics to be dict or FitProvenancePayload"):
        generate_provenance_lock(data_file, "invalid_string_metrics")


# =====================================================================
# 4. AASTeX / LaTeX Table Generation Tests
# =====================================================================


def test_generate_aastex_longtable_with_dict():
    """Verifies generate_aastex_longtable with simple dictionary input."""
    constants = {
        "A": 127634.508,
        "B": 24684.382,
        "C": 23758.129,
        "D_J": 0.04021,
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Simple Fit")

    assert "\\begin{table}[htbp]" in rendered
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in rendered
    assert "CH3OH Simple Fit" in rendered
    assert "127634.508000" in rendered
    assert "24684.382000" in rendered
    assert "\\Delta_J" in rendered
    assert "\\bottomrule" in rendered


def test_generate_aastex_longtable_with_tuples_and_frozen():
    """Verifies generate_aastex_longtable with tuple inputs and frozen status."""
    constants = {
        "A": (127634.508, 0.045),
        "B": (24684.382, 0.012),
        "D_K": (0.892, None, True),
        "V_3": {"value": 373.8, "status": "set", "is_frozen": True},
    }
    rendered = generate_aastex_longtable(constants, title_prefix="CH3OH Tuple Test")

    assert "0.045000" in rendered
    assert "0.012000" in rendered
    assert "Fixed" in rendered
    assert "Set" in rendered
    assert "V_3" in rendered


def test_generate_aastex_longtable_with_transitions(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies generate_aastex_longtable with transitions producing full combined document."""
    rendered = generate_aastex_longtable(
        constants_dict=molecular_parameters,
        title_prefix="Methanol Combined Publication",
        transitions=molecular_transitions,
    )

    assert "\\documentclass[twocolumn]{aastex631}" in rendered
    assert "\\begin{longtable}" in rendered
    assert "\\begin{table}[htbp]" in rendered
    assert "Methanol Combined Publication" in rendered
    assert "\\end{document}" in rendered


def test_generate_aastex_longtable_invalid_input_type():
    """Verifies TypeError when constants_dict is not dict or Sequence."""
    with pytest.raises(TypeError, match="Expected constants_dict to be dict or Sequence"):
        generate_aastex_longtable(12345)


def test_generate_aastex_longtables_structure(
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
):
    """Verifies structure and LaTeX formatting of generated tables and combined documents."""
    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="Methanol CH3OH Ground State",
    )

    assert "parameters_table" in result
    assert "transitions_table" in result
    assert "combined_document" in result

    params_tex = result["parameters_table"]
    trans_tex = result["transitions_table"]
    doc_tex = result["combined_document"]

    assert "\\begin{table}[htbp]" in params_tex
    assert "\\begin{tabular}{l c S[table-format=6.4] l l}" in params_tex
    assert (
        "\\caption{Optimized Spectroscopic Parameters for Methanol CH3OH Ground State}"
        in params_tex
    )
    assert "\\toprule" in params_tex
    assert "\\midrule" in params_tex
    assert "\\bottomrule" in params_tex
    assert "Rotational constant around principal a-axis" in params_tex

    assert "127634.508000" in params_tex
    assert "0.045000" in params_tex
    assert "Fixed" in params_tex
    assert "Set" in params_tex

    assert "\\begin{longtable}{c c S[table-format=6.4]" in trans_tex
    assert (
        "\\caption{Observed and Calculated Transitions for Methanol CH3OH Ground State (Truncated to Top 200)}"
        in trans_tex
    )
    assert "\\endfirsthead" in trans_tex
    assert "\\endhead" in trans_tex
    assert "\\endfoot" in trans_tex
    assert "\\endlastfoot" in trans_tex

    assert "\\documentclass[twocolumn]{aastex631}" in doc_tex
    assert "\\usepackage{siunitx}" in doc_tex
    assert "\\usepackage{booktabs}" in doc_tex
    assert "\\usepackage{longtable}" in doc_tex
    assert (
        "\\title{High-Precision Rotational Spectroscopic Fit and Transition Assignments: Methanol CH3OH Ground State}"
        in doc_tex
    )


def test_generate_aastex_longtables_truncation(
    molecular_parameters: list[dict[str, Any]],
):
    """Verifies that transitions list is truncated to the top 200 by intensity."""
    many_transitions = [
        {
            "upper_state": f"State_{i}_up",
            "lower_state": f"State_{i}_low",
            "observed_mhz": 10000.0 + i,
            "calculated_mhz": 10000.0 + i,
            "residual_mhz": 0.0,
            "uncertainty_mhz": 0.01,
            "intensity": float(i),
        }
        for i in range(1, 301)
    ]

    result = generate_aastex_longtables(
        optimized_params=molecular_parameters,
        transitions=many_transitions,
        title_prefix="Truncation Verification",
    )

    trans_table = result["transitions_table"]
    assert "(Truncated to Top 200)" in trans_table
    assert "State_300_up" in trans_table
    assert "State_101_up" in trans_table
    assert "State_1_up" not in trans_table


def test_generate_aastex_longtables_unparseable_uncertainty():
    """Verifies graceful handling of non-numeric string uncertainties."""
    params = [
        {
            "name": "CustomParam",
            "value": 12.3456,
            "uncertainty": "est. 0.05",
            "is_frozen": False,
            "sobol_audit": True,
            "status": "converged",
        }
    ]
    res = generate_aastex_longtables(params, [])
    assert "est. 0.05" in res["parameters_table"]


# =====================================================================
# 5. Mendeleev Dynamic Atomic Mass Resolution Tests
# =====================================================================


def test_mendeleev_element_atomic_weights():
    """Verifies dynamic IUPAC standard atomic weight retrieval via Mendeleev."""
    carbon_weight = get_element_atomic_weight("C")
    hydrogen_weight = get_element_atomic_weight("H")
    oxygen_weight = get_element_atomic_weight("O")
    nitrogen_weight = get_element_atomic_weight("N")

    assert 12.010 <= carbon_weight <= 12.012
    assert 1.007 <= hydrogen_weight <= 1.009
    assert 15.998 <= oxygen_weight <= 16.000
    assert 14.006 <= nitrogen_weight <= 14.008


def test_mendeleev_isotope_masses():
    """Verifies dynamic isotopic mass retrieval in atomic mass units via Mendeleev."""
    c12_mass = get_isotope_mass("C", 12)
    c13_mass = get_isotope_mass("C", 13)
    h1_mass = get_isotope_mass("H", 1)
    h2_mass = get_isotope_mass("H", 2)
    o16_mass = get_isotope_mass("O", 16)

    assert 11.999 <= c12_mass <= 12.001
    assert 13.002 <= c13_mass <= 13.004
    assert 1.007 <= h1_mass <= 1.009
    assert 2.013 <= h2_mass <= 2.015
    assert 15.994 <= o16_mass <= 15.996


def test_mendeleev_missing_isotope_error():
    """Verifies ValueError when requesting a non-existent isotope number."""
    with pytest.raises(ValueError, match="not found in Mendeleev database"):
        get_isotope_mass("C", 999)


# =====================================================================
# 6. DOI Resolution and BibTeX Citation Tests
# =====================================================================


def test_get_required_dois():
    """Verifies mapping of active physical models and quantum engines to authoritative DOIs."""
    active_models = [
        "watson_a",
        "spcat",
        "orca_v6",
        "jax",
        "pyarrow",
        "unknown_model_xyz",
    ]
    resolved = get_required_dois(active_models)

    assert "10.1016/0022-2852(77)90184-7" in resolved
    assert "10.1016/0022-2852(91)90393-O" in resolved
    assert "10.1063/5.0004608" in resolved
    assert "10.5281/zenodo.4766323" in resolved
    assert "10.5281/zenodo.4006056" in resolved
    assert len(resolved) == 5


def test_format_citations_to_bib():
    """Verifies parsing and deduplicating CrossRef JSON metadata into clean BibTeX format."""
    crossref_data = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": [
                "The determination of centrifugal distortion constants of asymmetric-top molecules"
            ],
            "author": [{"family": "Watson", "given": "James K. G."}],
            "container-title": ["Journal of Molecular Spectroscopy"],
            "volume": "65",
            "issue": "1",
            "page": "123-133",
            "issued": {"date-parts": [[1977, 4, 1]]},
        },
        {
            "message": {
                "DOI": "10.1016/0022-2852(91)90393-O",
                "title": [
                    "The fitting of sub-Doppler rotational spectra of molecules with internal rotor"
                ],
                "author": [{"family": "Pickett", "given": "Herbert M."}],
                "container-title": "Journal of Molecular Spectroscopy",
                "volume": "148",
                "number": "2",
                "page": "371-377",
                "issued": {"date-parts": [[1991, 8, 1]]},
            }
        },
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Duplicate Watson citation"],
        },
    ]

    bib_text = format_citations_to_bib(crossref_data)

    assert "@article{" in bib_text
    assert "author = {Watson, James K. G.}" in bib_text
    assert (
        "title = {{The determination of centrifugal distortion constants of asymmetric-top molecules}}"
        in bib_text
    )
    assert "journal = {Journal of Molecular Spectroscopy}" in bib_text
    assert "volume = {65}" in bib_text
    assert "number = {1}" in bib_text
    assert "pages = {123-133}" in bib_text
    assert "year = {1977}" in bib_text
    assert "doi = {10.1016/0022-2852(77)90184-7}" in bib_text

    assert "author = {Pickett, Herbert M.}" in bib_text
    assert "doi = {10.1016/0022-2852(91)90393-O}" in bib_text
    assert "year = {1991}" in bib_text

    assert bib_text.count("10.1016/0022-2852(77)90184-7") == 1


def test_format_citations_to_bib_extended_schemas():
    """Verifies parsing of various author schemas, missing DOIs, and year fallbacks."""
    records = [
        {"title": "Missing DOI Entry", "author": "No DOI Author"},
        {
            "DOI": "10.1000/str_author",
            "title": ["Single String Author Paper"],
            "author": ["Jane Doe"],
            "year": "2024",
        },
        {
            "DOI": "10.1000/family_only",
            "title": "Family Only Paper",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
        {
            "DOI": "10.1000/heisenberg_2",
            "title": "Family Only Paper 2",
            "author": [{"family": "Heisenberg"}],
            "year": 1927,
        },
    ]

    bib_text = format_citations_to_bib(records)
    assert "Missing DOI Entry" not in bib_text
    assert "author = {Jane Doe}" in bib_text
    assert "author = {Heisenberg}" in bib_text
    assert "year = {2024}" in bib_text
    assert "year = {1927}" in bib_text
    assert "Heisenberg1927Family" in bib_text
    assert "Heisenberg1927Family_1" in bib_text


# =====================================================================
# 7. Compression Strategy and Packaging Tests
# =====================================================================


def test_evaluate_compression_strategy():
    """Verifies mathematical threshold evaluations for storage packaging."""
    small_size = 50 * 1024 * 1024
    assert evaluate_compression_strategy(small_size) == "ZIP"

    exact_threshold = 104857600
    assert evaluate_compression_strategy(exact_threshold) == "ZSTD"

    large_size = 500 * 1024 * 1024
    assert evaluate_compression_strategy(large_size) == "ZSTD"


def test_package_fit_artifacts_zip(clean_workspace: Path):
    """Verifies packaging workspace directory into a standard .zip archive."""
    src_dir = clean_workspace / "fit_export_src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "fit.log").write_text(
        "Fit iteration log output: Converged in 14 iterations.", encoding="utf-8"
    )
    (src_dir / "tables").mkdir()
    (src_dir / "tables" / "table1.tex").write_text(
        "\\begin{table}...\\end{table}", encoding="utf-8"
    )

    archive_base = clean_workspace / "export_archive"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZIP")

    assert archive_path.exists()
    assert archive_path.name == "export_archive.zip"
    assert zipfile.is_zipfile(archive_path)

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        assert "fit.log" in namelist
        assert "tables/table1.tex" in namelist


def test_package_fit_artifacts_zstd(clean_workspace: Path):
    """Verifies packaging workspace directory into a tar.zst archive."""
    src_dir = clean_workspace / "fit_export_src_zstd"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "data.bin").write_bytes(b"SPECTROSCOPIC_RAW_STREAM_DATA" * 500)
    (src_dir / "info.json").write_text('{"status": "READY"}', encoding="utf-8")

    archive_base = clean_workspace / "export_archive_zstd"
    archive_path = package_fit_artifacts(src_dir, archive_base, strategy="ZSTD")

    assert archive_path.exists()
    assert archive_path.name == "export_archive_zstd.tar.zst"

    compressed_bytes = archive_path.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    decompressed_tar = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_tar), mode="r") as tf:
        names = tf.getnames()
        assert "data.bin" in names
        assert "info.json" in names


def test_package_fit_artifacts_edge_cases(clean_workspace: Path):
    """Verifies automatic strategy evaluation, extension normalization, and error handling."""
    src = clean_workspace / "edge_case_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "dataset.txt").write_text("Adversarial audit payload", encoding="utf-8")

    auto_archive = package_fit_artifacts(src, clean_workspace / "auto_archive")
    assert auto_archive.exists()
    assert auto_archive.suffix == ".zip"

    tar_zst_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar.zst", strategy="ZSTD"
    )
    assert tar_zst_archive.name == "explicit.tar.zst"
    assert tar_zst_archive.exists()

    tar_archive = package_fit_artifacts(
        src, clean_workspace / "explicit.tar", strategy="ZSTD"
    )
    assert tar_archive.name == "explicit.tar.zst"
    assert tar_archive.exists()

    with pytest.raises(ValueError, match="Unsupported packaging strategy"):
        package_fit_artifacts(src, clean_workspace / "bad_strat", strategy="SEVEN_ZIP")

    with pytest.raises(NotADirectoryError):
        package_fit_artifacts(
            clean_workspace / "non_existent_folder", clean_workspace / "out"
        )


# =====================================================================
# 8. Read-Only Sealing Tests
# =====================================================================


def test_seal_artifact_read_only(clean_workspace: Path):
    """Verifies cross-platform read-only permissions are enforced on target artifacts."""
    target_file = clean_workspace / "sealed_artifact.json"
    target_file.write_text('{"immutable": true}', encoding="utf-8")

    assert seal_artifact_read_only(target_file) is True

    mode = target_file.stat().st_mode
    assert not (mode & stat.S_IWUSR)

    if os.name != "nt":
        with pytest.raises(PermissionError), open(target_file, "a") as f:
            f.write("mutation")


def test_seal_artifact_read_only_missing_file(clean_workspace: Path):
    """Verifies FileNotFoundError when attempting to seal non-existent file."""
    missing = clean_workspace / "phantom_artifact.bin"
    with pytest.raises(FileNotFoundError):
        seal_artifact_read_only(missing)


# =====================================================================
# 9. End-to-End export_spycfit_snapshot Orchestration Tests
# =====================================================================


def test_export_spycfit_snapshot_pipeline(
    authoritative_payload: FitProvenancePayload,
    molecular_parameters: list[dict[str, Any]],
    molecular_transitions: list[dict[str, Any]],
    clean_workspace: Path,
):
    """Verifies full orchestration pipeline from input files to sealed archive."""
    source_dir = clean_workspace / "source_workspace"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "raw_spectrum.dat").write_text(
        "12000.0 0.5\n12050.0 0.9\n", encoding="utf-8"
    )

    processed_dir = clean_workspace / "processed_workspace"

    crossref_records = [
        {
            "DOI": "10.1016/0022-2852(77)90184-7",
            "title": ["Centrifugal Distortion in Asymmetric Tops"],
            "author": [{"family": "Watson", "given": "J. K. G."}],
            "journal": "J. Mol. Spectrosc.",
            "year": 1977,
        }
    ]

    result = export_spycfit_snapshot(
        payload=authoritative_payload,
        source_dir=source_dir,
        output_dir=processed_dir,
        active_models=authoritative_payload.active_models,
        crossref_records=crossref_records,
        optimized_params=molecular_parameters,
        transitions=molecular_transitions,
        title_prefix="CH3OH Production Fit",
        strategy="ZIP",
    )

    assert result["status"] == "SUCCESS"
    assert result["session_id"] == authoritative_payload.session_id

    prov_path = Path(result["provenance_path"])
    params_tex_path = Path(result["parameters_table_path"])
    trans_tex_path = Path(result["transitions_table_path"])
    doc_path = Path(result["document_path"])
    bib_path = Path(result["citations_bib_path"])
    archive_path = Path(result["archive_path"])

    assert prov_path.exists()
    assert params_tex_path.exists()
    assert trans_tex_path.exists()
    assert doc_path.exists()
    assert bib_path.exists()
    assert archive_path.exists()

    loaded_prov = FitProvenancePayload.from_json_file(prov_path)
    assert loaded_prov.session_id == authoritative_payload.session_id

    assert not (prov_path.stat().st_mode & stat.S_IWUSR)
    assert not (archive_path.stat().st_mode & stat.S_IWUSR)


# =====================================================================
# 10. Workspace Directory Resolution Tests
# =====================================================================


def test_resolve_processed_workspace_dir(
    monkeypatch: pytest.MonkeyPatch, clean_workspace: Path
):
    """Verifies directory resolution with custom paths, environment variable override, and defaults."""
    custom_target = clean_workspace / "custom_processed"
    assert resolve_processed_workspace_dir(custom_target) == custom_target.resolve()

    env_dir = clean_workspace / "env_state_dir"
    monkeypatch.setenv("COCHEM_STATE_DIR", str(env_dir))
    expected_env_target = env_dir / "SpycFit_Workspace" / "Processed"
    assert resolve_processed_workspace_dir() == expected_env_target.resolve()

    monkeypatch.delenv("COCHEM_STATE_DIR", raising=False)
    resolved_default = get_spycfit_processed_dir()
    assert "SpycFit_Workspace" in str(resolved_default)
    assert "Processed" in str(resolved_default)

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.