"""# zero-stub anti-spoofing engine
CoChem Master Scientific Report & Supporting Information Compiler (cochem_base.io.cochem_reporter)

Compiles publication-grade computational reports and Supporting Information (SI) in Markdown,
LaTeX SI (tcolorbox / tabular), and HTML formats, signed with RFC 8785 Ed25519-SHA512 envelopes.

Governing Requirements:
- REQ-61, REQ-63, REQ-65, REQ-67, REQ-69, REQ-70: Master Scientific & SI Compiler.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.hazmat.primitives.asymmetric import ed25519

from cochem_base.calc.cochem_method_auditor import generate_methodology_audit_report
from cochem_base.core.cochem_crypto import sign_report_payload
from cochem_base.core.cochem_version import VCSProvenance
from cochem_base.io.cochem_banners import render_unconverged_banner
from cochem_base.topos.cochem_topos_crusher import ManualAdjustmentRecord

logger = logging.getLogger("cochem_reporter")


def compile_master_scientific_report(
    job_id: str,
    calculation_specs: Dict[str, Any],
    results_dict: Dict[str, Any],
    vcs_provenance: Optional[VCSProvenance] = None,
    telemetry_dict: Optional[Dict[str, Any]] = None,
    manual_adjustments: Optional[List[ManualAdjustmentRecord]] = None,
    residuals: Optional[Dict[str, float]] = None,
    private_key: Optional[ed25519.Ed25519PrivateKey] = None,
) -> Dict[str, Any]:
    """
    Compile complete, publication-grade master scientific report payload with cryptographic binding.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Methodology Audit
    is_audit_ok, audit_md, violations = generate_methodology_audit_report(calculation_specs)

    # 2. Unconverged Warning Banner if applicable
    unconverged_banner_md = ""
    if residuals:
        unconverged_banner_md = render_unconverged_banner(residuals, format="markdown")

    # 3. Manual Intervention Register Table
    adjustment_list = [r.to_dict() for r in manual_adjustments] if manual_adjustments else []

    report_payload: Dict[str, Any] = {
        "job_id": job_id,
        "timestamp_utc": timestamp,
        "calculation_specs": calculation_specs,
        "results": results_dict,
        "vcs_provenance": vcs_provenance.to_dict() if vcs_provenance else None,
        "hardware_telemetry": telemetry_dict,
        "methodology_audit": {
            "passed": is_audit_ok,
            "violations": violations,
            "markdown_report": audit_md,
        },
        "unconverged_warning_banner": unconverged_banner_md if unconverged_banner_md else None,
        "manual_intervention_register": adjustment_list,
    }

    # 4. Cryptographic Envelope Signing
    if private_key:
        signed_report = sign_report_payload(report_payload, private_key, inline=True)
        return signed_report

    return report_payload


def export_markdown_report(report_data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export master scientific report to GitHub-flavored Markdown."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    job_id = report_data.get("job_id", "cochem_job")
    specs = report_data.get("calculation_specs", {})
    results = report_data.get("results", {})
    vcs = report_data.get("vcs_provenance", {})
    sig = report_data.get("_provenance_signature", {})

    lines: List[str] = [
        f"# CoChem Computational Chemistry Report: {job_id}",
        f"**Date:** {report_data.get('timestamp_utc', 'N/A')}  ",
        f"**Software Version:** `{vcs.get('full_identifier', '0.1.0') if vcs else '0.1.0'}`  ",
        "",
    ]

    # Warning banner if present
    if report_data.get("unconverged_warning_banner"):
        lines.append(report_data["unconverged_warning_banner"])
        lines.append("")

    # Executive Summary & Results
    lines.extend([
        "## Executive Summary & Results",
        f"- **Molecular Formula:** {results.get('formula', 'N/A')}",
        f"- **Canonical SMILES:** `{results.get('smiles', 'N/A')}`",
        f"- **Electronic Energy:** {results.get('energy_hartree', 'N/A')} Hartree",
        "",
        "## Method Matrix Compliance Assessment",
        report_data.get("methodology_audit", {}).get("markdown_report", "Method Matrix compliant."),
        "",
    ])

    # Manual Intervention Register Table
    adjustments = report_data.get("manual_intervention_register", [])
    if adjustments:
        lines.extend([
            "## Manual Intervention & Computational Deviation Register",
            "| Module | Parameter | Override Value | Min Dist (Å) | Operator Justification |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for adj in adjustments:
            lines.append(
                f"| `{adj.get('target_module')}` | `{adj.get('parameter_name')}` | `{adj.get('override_value')}` | {adj.get('min_interatomic_distance_angstrom')} | {adj.get('operator_justification')} |"
            )
        lines.append("")

    # Cryptographic Provenance Block
    if sig:
        lines.extend([
            "## Cryptographic Provenance Signature",
            f"- **Algorithm:** `{sig.get('algorithm')}`",
            f"- **Public Key Fingerprint:** `{sig.get('fingerprint')}`",
            f"- **Signature (B64):** `{sig.get('signature')}`",
            "",
        ])

    content = "\n".join(lines)
    p.write_text(content, encoding="utf-8")
    return p


def export_latex_si_report(report_data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export Supporting Information (SI) report in ACS-compliant LaTeX format."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    job_id = report_data.get("job_id", "cochem_job")
    results = report_data.get("results", {})
    adjustments = report_data.get("manual_intervention_register", [])

    lines = [
        r"\documentclass[journal=jacsat,manuscript=suppinfo]{achemso}",
        r"\usepackage{amsmath,amssymb,booktabs,tcolorbox}",
        r"\title{Supporting Information: " + str(job_id) + "}",
        r"\begin{document}",
        r"\section{Computational Details and Energetics}",
        f"Molecular Formula: {results.get('formula', 'N/A')}\\\\",
        f"Final Electronic Energy: {results.get('energy_hartree', 'N/A')} Hartrees\\\\",
        "",
    ]

    if adjustments:
        lines.extend([
            r"\section{Manual Intervention and Computational Deviation Register}",
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{llllp{5cm}}",
            r"\toprule",
            r"Module & Parameter & Override & $r_{\min}$ (\AA) & Justification \\",
            r"\midrule",
        ])
        for adj in adjustments:
            lines.append(
                f"{adj.get('target_module')} & {adj.get('parameter_name')} & {adj.get('override_value')} & {adj.get('min_interatomic_distance_angstrom')} & {adj.get('operator_justification')} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    lines.append(r"\end{document}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def export_html_report(report_data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export standalone HTML computational report."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    job_id = report_data.get("job_id", "cochem_job")
    results = report_data.get("results", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CoChem Report: {job_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #212121; }}
h1, h2 {{ color: #0d47a1; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #cfd8dc; padding: 8px 12px; text-align: left; }}
th {{ background-color: #eceff1; }}
</style>
</head>
<body>
<h1>CoChem Scientific Report: {job_id}</h1>
<p><strong>Generated UTC:</strong> {report_data.get('timestamp_utc', 'N/A')}</p>
<h2>Results</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Formula</td><td>{results.get('formula', 'N/A')}</td></tr>
<tr><td>SMILES</td><td><code>{results.get('smiles', 'N/A')}</code></td></tr>
<tr><td>Energy (Hartree)</td><td>{results.get('energy_hartree', 'N/A')}</td></tr>
</table>
</body>
</html>"""
    p.write_text(html, encoding="utf-8")
    return p
