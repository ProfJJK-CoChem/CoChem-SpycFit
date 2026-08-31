"""
CoChem Peer-Reviewer Transparency & Non-Standard Deviation Auditor (REQ-QC-100).

Provenance:
- Method Matrix v4.2 Quantum Chemistry Invariants:
  - Tight convergence criteria (TightOPT) mandated for stationary points.
  - Integration grid floor (defgrid2/defgrid3) for DFT optimizations.
  - Dispersion correction mandate (D3BJ, D4, VV10) for polyatomic systems.
  - Strict ban on Calc_Hess true and additive diffuse corrections.
  - Spin contamination tolerance (<S^2> excursion < 10.0%).
  - Level-of-Theory Invariance between geometry optimization and harmonic frequencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, ConfigDict, Field


class DeviationRecord(BaseModel):
    """Record of a detected methodological deviation from Method Matrix v4.2."""
    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(description="Method Matrix rule identifier (e.g. 'MM-RULE-001')")
    rule_name: str = Field(description="Descriptive name of the audited rule")
    severity: Literal["BENIGN_OPTIMIZATION", "JUSTIFIED_COMPROMISE", "METHODOLOGICAL_WARNING"] = Field(
        description="Severity classification of the deviation"
    )
    affected_job: str = Field(description="Job ID or species identifier where deviation occurred")
    expected_parameter: str = Field(description="Method Matrix standard parameter value")
    actual_parameter: str = Field(description="Observed calculation parameter value")
    scientific_justification: str = Field(description="Scientific rationale or impact assessment")


class TransparencyAuditReport(BaseModel):
    """Audited report summarizing methodological compliance for peer reviewers."""
    model_config = ConfigDict(frozen=True)

    total_jobs_audited: int = Field(description="Number of calculation DAG jobs inspected")
    compliance_score_percent: float = Field(description="Methodological compliance score (0.0 to 100.0%)")
    audit_verdict: Literal["FULLY_COMPLIANT", "COMPLIANT_WITH_JUSTIFIED_DEVIATIONS", "NON_COMPLIANT_WARNING"] = Field(
        description="Overall compliance verdict"
    )
    deviations: List[DeviationRecord] = Field(default_factory=list, description="List of recorded deviations")
    markdown_report_path: str = Field(description="Path to generated Markdown report")
    latex_report_path: str = Field(description="Path to generated LaTeX report")


def audit_project_deviations(
    job_manifests: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    markdown_filename: str = "Reviewer_Transparency_Report.md",
    latex_filename: str = "Reviewer_Transparency_Report.tex",
) -> TransparencyAuditReport:
    """
    Perform an automated peer-reviewer transparency audit of calculation parameters.

    Args:
        job_manifests: List of job execution dictionaries containing calculation parameters.
        output_dir: Directory where reports will be saved.
        markdown_filename: Output markdown filename.
        latex_filename: Output LaTeX filename.

    Returns:
        TransparencyAuditReport: Comprehensive compliance audit result.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / markdown_filename
    tex_file = out_path / latex_filename

    deviations: List[DeviationRecord] = []
    penalty_sum = 0.0

    SEVERITY_WEIGHTS = {
        "BENIGN_OPTIMIZATION": 2.0,
        "JUSTIFIED_COMPROMISE": 10.0,
        "METHODOLOGICAL_WARNING": 25.0,
    }

    for job in job_manifests:
        job_id = str(job.get("job_id", job.get("label", "unknown_job")))
        calc_type = str(job.get("calc_type", "opt")).lower()
        method = str(job.get("method", "dft")).lower()
        functional = str(job.get("functional", "")).lower()
        basis = str(job.get("basis", "")).lower()
        grid = str(job.get("grid", "defgrid3")).lower()
        conv = str(job.get("convergence", "tightopt")).lower()
        disp = str(job.get("dispersion", "")).lower()
        atom_count = int(job.get("num_atoms", 10))
        s2_calc = job.get("s2_calc")
        s2_ideal = job.get("s2_ideal")
        opt_lot = job.get("opt_lot")
        freq_lot = job.get("freq_lot")
        calc_hess = bool(job.get("calc_hess_true", False))
        additive_diffuse = bool(job.get("additive_diffuse_correction", False))

        # 1. Ban on Calc_Hess true
        if calc_hess:
            dev = DeviationRecord(
                rule_id="MM-BAN-001",
                rule_name="Ban on Calc_Hess true in Optimizations",
                severity="METHODOLOGICAL_WARNING",
                affected_job=job_id,
                expected_parameter="InHess XTB2 / Lindh / Model Hessian",
                actual_parameter="Calc_Hess true",
                scientific_justification="Calculating exact initial Hessians causes excessive wall time overhead.",
            )
            deviations.append(dev)
            penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 2. Ban on Additive Diffuse Corrections
        if additive_diffuse:
            dev = DeviationRecord(
                rule_id="MM-BAN-002",
                rule_name="Ban on Additive Diffuse Corrections",
                severity="METHODOLOGICAL_WARNING",
                affected_job=job_id,
                expected_parameter="Full diffuse basis set calculation (e.g. def2-TZVPD)",
                actual_parameter="Additive diffuse energy delta",
                scientific_justification="Additive diffuse corrections degrade intermolecular energy MAE and distort geometries.",
            )
            deviations.append(dev)
            penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 3. Convergence Criteria
        if "opt" in calc_type:
            if "loose" in conv or "default" in conv:
                severity: Literal["BENIGN_OPTIMIZATION", "JUSTIFIED_COMPROMISE", "METHODOLOGICAL_WARNING"] = (
                    "BENIGN_OPTIMIZATION" if "default" in conv else "JUSTIFIED_COMPROMISE"
                )
                dev = DeviationRecord(
                    rule_id="MM-OPT-001",
                    rule_name="Geometry Optimization Convergence Threshold",
                    severity=severity,
                    affected_job=job_id,
                    expected_parameter="TightOPT (TolMaxG 1e-4, TolRMSG 3e-5)",
                    actual_parameter=conv.upper(),
                    scientific_justification="Loose convergence risks artificial low imaginary frequencies or incomplete stationary point relaxation.",
                )
                deviations.append(dev)
                penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 4. Integration Grids
        if "dft" in method or functional:
            if "defgrid1" in grid or "coarse" in grid:
                dev = DeviationRecord(
                    rule_id="MM-GRID-001",
                    rule_name="DFT Quadrature Integration Grid Level",
                    severity="JUSTIFIED_COMPROMISE",
                    affected_job=job_id,
                    expected_parameter="defgrid2 or defgrid3",
                    actual_parameter=grid,
                    scientific_justification="Coarse integration grids induce grid-dependent noise in vibrational force fields.",
                )
                deviations.append(dev)
                penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 5. Dispersion Corrections on DFT
        if ("dft" in method or functional) and atom_count > 6:
            has_disp = any(d in disp for d in ["d3", "d4", "vv10", "bj"]) or any(
                f in functional for f in ["-d", "-d4", "-v", "m06"]
            )
            if not has_disp:
                dev = DeviationRecord(
                    rule_id="MM-DISP-001",
                    rule_name="Dispersion Correction Mandate for DFT",
                    severity="METHODOLOGICAL_WARNING",
                    affected_job=job_id,
                    expected_parameter="D3(BJ), D4, or non-local VV10",
                    actual_parameter="None",
                    scientific_justification="Uncorrected pure/hybrid DFT severely underestimates non-covalent and London dispersion interactions.",
                )
                deviations.append(dev)
                penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 6. Spin Contamination Check
        if s2_calc is not None and s2_ideal is not None and s2_ideal > 0:
            delta_s2 = abs(s2_calc - s2_ideal)
            pct = (delta_s2 / s2_ideal) * 100.0
            if pct > 10.0:
                dev = DeviationRecord(
                    rule_id="MM-SPIN-001",
                    rule_name="Unrestricted Open-Shell Spin Contamination",
                    severity="METHODOLOGICAL_WARNING",
                    affected_job=job_id,
                    expected_parameter=f"<S^2> ideal = {s2_ideal:.4f} (deviation < 10.0%)",
                    actual_parameter=f"<S^2> calc = {s2_calc:.4f} ({pct:.1f}% excursion)",
                    scientific_justification="Severe spin contamination indicates radical multireference character requiring BS-DFT or CASSCF.",
                )
                deviations.append(dev)
                penalty_sum += SEVERITY_WEIGHTS[dev.severity]

        # 7. Level-of-Theory Invariance
        if opt_lot and freq_lot and opt_lot != freq_lot:
            dev = DeviationRecord(
                rule_id="MM-LOT-001",
                rule_name="Level-of-Theory Invariance for Vibrational Frequencies",
                severity="METHODOLOGICAL_WARNING",
                affected_job=job_id,
                expected_parameter=f"Opt LoT == Freq LoT ({opt_lot})",
                actual_parameter=f"Opt={opt_lot} vs Freq={freq_lot}",
                scientific_justification="Computing frequencies at a different LoT violates potential energy surface stationary gradient conditions.",
            )
            deviations.append(dev)
            penalty_sum += SEVERITY_WEIGHTS[dev.severity]

    compliance_score = max(0.0, 100.0 - penalty_sum)
    if compliance_score >= 95.0:
        verdict: Literal["FULLY_COMPLIANT", "COMPLIANT_WITH_JUSTIFIED_DEVIATIONS", "NON_COMPLIANT_WARNING"] = "FULLY_COMPLIANT"
    elif compliance_score >= 75.0:
        verdict = "COMPLIANT_WITH_JUSTIFIED_DEVIATIONS"
    else:
        verdict = "NON_COMPLIANT_WARNING"

    # Export Markdown Report
    md_lines = [
        "# CoChem Reviewer Transparency & Methodological Audit Report",
        "",
        f"**Total Calculations Audited:** `{len(job_manifests)}`  ",
        f"**Method Matrix Compliance Score:** `{compliance_score:.1f}%`  ",
        f"**Audit Verdict:** `{verdict}`  ",
        "",
        "---",
        "",
        "## Summary of Methodological Deviations",
        "",
    ]
    if not deviations:
        md_lines.append("> [!NOTE]\n> **Zero Methodological Deviations Detected.** All calculations strictly adhere to Method Matrix v4.2 standards.")
    else:
        md_lines.extend([
            "| Rule ID | Severity | Affected Job | Expected | Actual | Scientific Justification |",
            "| :--- | :---: | :--- | :--- | :--- | :--- |",
        ])
        for d in deviations:
            md_lines.append(
                f"| `{d.rule_id}` | `{d.severity}` | `{d.affected_job}` | `{d.expected_parameter}` | `{d.actual_parameter}` | {d.scientific_justification} |"
            )

    with md_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    # Export LaTeX Report
    verdict_tex = verdict.replace("_", r"\_")
    tex_lines = [
        r"\section{Reviewer Transparency and Methodological Audit}",
        f"A total of {len(job_manifests)} quantum chemical calculations were audited against the standards of the CoChem Method Matrix v4.2.",
        f"The overall methodological compliance score is \\textbf{{{compliance_score:.1f}\\%}} (Verdict: \\texttt{{{verdict_tex}}}).",
        r"\vspace{0.5em}",
    ]
    if deviations:
        tex_lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Audited Methodological Deviations and Scientific Rationales.}",
            r"\label{tab:transparency_audit}",
            r"\begin{tabular}{lp{3.5cm}p{3.5cm}p{5cm}}",
            r"\toprule",
            r"Rule ID & Severity & Parameter (Actual / Expected) & Scientific Justification \\",
            r"\midrule",
        ])
        for d in deviations:
            r_id = d.rule_id.replace("_", r"\_")
            sev = d.severity.replace("_", r"\_")
            act = d.actual_parameter.replace("_", r"\_")
            exp = d.expected_parameter.replace("_", r"\_")
            just = d.scientific_justification.replace("_", r"\_")
            tex_lines.append(
                f"\\texttt{{{r_id}}} & \\texttt{{{sev}}} & {act} / {exp} & {just} \\\\"
            )
        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
    else:
        tex_lines.append(r"\noindent \textbf{Compliance Declaration:} Zero methodological deviations were detected across all stationary points.")

    with tex_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines) + "\n")

    return TransparencyAuditReport(
        total_jobs_audited=len(job_manifests),
        compliance_score_percent=compliance_score,
        audit_verdict=verdict,
        deviations=deviations,
        markdown_report_path=str(md_file),
        latex_report_path=str(tex_file),
    )
