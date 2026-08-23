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
