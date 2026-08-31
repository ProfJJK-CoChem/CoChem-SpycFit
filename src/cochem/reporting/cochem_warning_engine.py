"""
CoChem Diagnostic Warning Engine & Stability Flag Formatter (REQ-QC-081, REQ-QC-087).

Provenance:
- Method Matrix [M]: High-visibility diagnostic warning banners for spin contamination,
  electronic orbital Hessian instabilities, gradient threshold alerts, and Level-of-Theory discrepancies.
- Derived Relations [D]: ANSI terminal color escapes and GitHub-flavored markdown alert boxes.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DiagnosticWarning(BaseModel):
    """Data contract representing a diagnostic warning."""
    model_config = ConfigDict(frozen=True)

    category: Literal["SPIN_CONTAMINATION", "WAVEFUNCTION_INSTABILITY", "GRADIENT_BORDERLINE", "LOT_MISMATCH", "UNPHYSICAL_VALENCE"]
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    message: str
    ansi_banner: str
    markdown_banner: str
    remediation_advice: str


class WarningSummary(BaseModel):
    """Data contract summarizing all warnings raised across a calculation."""
    model_config = ConfigDict(frozen=True)

    total_warnings: int
    critical_count: int
    warning_count: int
    info_count: int
    warnings: List[DiagnosticWarning] = Field(default_factory=list)


def format_spin_contamination_warning(
    s2_calc: float,
    s_expected: float,
    threshold_delta_s2: float = 0.10,
) -> Optional[DiagnosticWarning]:
    """
    [REQ-QC-081, REQ-QC-087] Format high-visibility alert if spin contamination Delta <S^2> > 0.10.
    """
    s2_ideal = s_expected * (s_expected + 1.0)
    delta_s2 = abs(s2_calc - s2_ideal)

    if delta_s2 <= threshold_delta_s2:
        return None

    pct = (delta_s2 / (s2_ideal + 1.0)) * 100.0
    msg = (
        f"Significant spin contamination detected: <S^2>_calc = {s2_calc:.4f} "
        f"(ideal = {s2_ideal:.4f}, Delta <S^2> = {delta_s2:.4f}, {pct:.1f}% contamination)."
    )
    remediation = (
        "Wavefunction exhibits radical or multireference character. "
        "Recommend Broken-Symmetry DFT with Yamaguchi projection, CASSCF, or unrestricted DLPNO treatments."
    )

    ansi = (
        "\033[93m"
        "╔════════════════════════════════════════════════════════════════════════════════════╗\n"
        "║                     [WARNING] SEVERE SPIN CONTAMINATION                           ║\n"
        f"║  Calculated <S^2>: {s2_calc:7.4f} | Ideal: {s2_ideal:7.4f} | Delta: {delta_s2:7.4f}                     ║\n"
        "╚════════════════════════════════════════════════════════════════════════════════════╝"
        "\033[0m"
    )

    md = (
        "> [!WARNING]\n"
        "> **Severe Spin Contamination Detected**\n"
        f"> - Calculated $\\langle S^2 \\rangle$: `{s2_calc:.4f}`\n"
        f"> - Ideal $\\langle S^2 \\rangle$: `{s2_ideal:.4f}` ($S={s_expected}$)\n"
        f"> - Deviation $\\Delta \\langle S^2 \\rangle$: `{delta_s2:.4f}`\n"
        f"> - *Remediation*: {remediation}\n"
    )

    return DiagnosticWarning(
        category="SPIN_CONTAMINATION",
        severity="WARNING" if delta_s2 < 0.5 else "CRITICAL",
        message=msg,
        ansi_banner=ansi,
        markdown_banner=md,
        remediation_advice=remediation,
    )


def format_wavefunction_instability_warning(
    lowest_eigenvalue: float,
    instability_type: str = "UNSTABLE_INTERNAL_RHF_UHF",
) -> Optional[DiagnosticWarning]:
    """
    [REQ-QC-087] Format diagnostic banner when orbital Hessian eigenvalue lambda_orb <= 0.
    """
    if lowest_eigenvalue > 0.0:
        return None

    msg = f"Wavefunction instability detected: lowest orbital Hessian eigenvalue lambda_orb = {lowest_eigenvalue:.4e} <= 0."
    remediation = (
        "Orbital optimization converged to a saddle point or unstable solution. "
        "Triggering unconstrained Broken-Symmetry Second-Order SCF (SOSCF) re-optimization."
    )

    ansi = (
        "\033[91m"
        "╔════════════════════════════════════════════════════════════════════════════════════╗\n"
        "║                   [CRITICAL] WAVEFUNCTION INSTABILITY DETECTED                    ║\n"
        f"║  Lowest Orbital Eigenvalue lambda_orb = {lowest_eigenvalue:11.4e} <= 0                          ║\n"
        f"║  Mode: {instability_type:<67} ║\n"
        "╚════════════════════════════════════════════════════════════════════════════════════╝"
        "\033[0m"
    )

    md = (
        "> [!CAUTION]\n"
        "> **Wavefunction Instability Detected**\n"
        f"> - Orbital Hessian lowest eigenvalue $\\lambda_{{\\text{{orb}}}}$ = `{lowest_eigenvalue:.4e}`\n"
        f"> - Classification: `{instability_type}`\n"
        f"> - *Action*: {remediation}\n"
    )

    return DiagnosticWarning(
        category="WAVEFUNCTION_INSTABILITY",
        severity="CRITICAL",
        message=msg,
        ansi_banner=ansi,
        markdown_banner=md,
        remediation_advice=remediation,
    )


def format_gradient_threshold_warning(
    max_gradient_au: float,
    border_threshold_au: float = 1.0e-5,
    ceiling_threshold_au: float = 1.0e-4,
) -> Optional[DiagnosticWarning]:
    """
    [REQ-QC-085] Format warning if stationary point gradient borders the acceptance ceiling (1e-5 < ||g|| <= 1e-4 a.u.).
    """
    if max_gradient_au <= border_threshold_au or max_gradient_au > ceiling_threshold_au:
        return None

    msg = (
        f"Stationary point gradient is borderline: max ||g|| = {max_gradient_au:.2e} a.u. "
        f"(ceiling = {ceiling_threshold_au:.2e} a.u.)."
    )
    remediation = "Consider applying tighter optimization convergence criteria (! TightOpt / opt=tight) before frequency analysis."

    ansi = f"\033[93m[BORDERLINE GRADIENT] Max gradient ||g|| = {max_gradient_au:.2e} a.u. near ceiling.\033[0m"
    md = (
        "> [!NOTE]\n"
        f"> **Borderline Stationary Point Residual**: Max $\\|\\mathbf{{g}}\\|_\\infty = {max_gradient_au:.2e}\\text{{ a.u.}}$\n"
    )

    return DiagnosticWarning(
        category="GRADIENT_BORDERLINE",
        severity="INFO",
        message=msg,
        ansi_banner=ansi,
        markdown_banner=md,
        remediation_advice=remediation,
    )


def generate_warning_summary(warnings: List[DiagnosticWarning]) -> WarningSummary:
    """
    Compile a complete structured warning summary.
    """
    crit = sum(1 for w in warnings if w.severity == "CRITICAL")
    warn = sum(1 for w in warnings if w.severity == "WARNING")
    info = sum(1 for w in warnings if w.severity == "INFO")

    return WarningSummary(
        total_warnings=len(warnings),
        critical_count=crit,
        warning_count=warn,
        info_count=info,
        warnings=warnings,
    )


# =========================================================================
# Chunk 10: Reviewer Transparency Warning Diagnostics Formatter (REQ-QC-100)
# =========================================================================

def format_terminal_warning_summary(report: Any) -> str:
    """
    Format colored ANSI/Unicode summary banner for terminal display.

    Args:
        report: TransparencyAuditReport instance from cochem_transparency_auditor.

    Returns:
        str: ANSI formatted terminal string.
    """
    verdict = getattr(report, "audit_verdict", "UNKNOWN")
    score = getattr(report, "compliance_score_percent", 0.0)
    deviations = getattr(report, "deviations", [])

    if verdict == "FULLY_COMPLIANT":
        color = "\033[92m"  # Green
        badge = "[✓ FULLY COMPLIANT]"
    elif verdict == "COMPLIANT_WITH_JUSTIFIED_DEVIATIONS":
        color = "\033[93m"  # Yellow/Amber
        badge = "[⚠ COMPLIANT WITH JUSTIFIED DEVIATIONS]"
    else:
        color = "\033[91m"  # Red
        badge = "[✗ NON-COMPLIANT WARNING]"

    reset = "\033[0m"

    lines = [
        f"{color}╔════════════════════════════════════════════════════════════════════════════════════╗",
        f"║  COCHEM METHODOLOGICAL TRANSPARENCY AUDIT SUMMARY                                   ║",
        f"║  Verdict: {badge:<40} Compliance Score: {score:5.1f}% ║",
        f"╚════════════════════════════════════════════════════════════════════════════════════╝{reset}",
    ]

    if not deviations:
        lines.append(f"{color}  ✔ All {getattr(report, 'total_jobs_audited', 0)} calculations adhere strictly to Method Matrix v4.2 standards.{reset}")
    else:
        lines.append(f"\033[1m  Recorded Deviations ({len(deviations)}):{reset}")
        for idx, dev in enumerate(deviations, 1):
            sev_color = "\033[91m" if dev.severity == "METHODOLOGICAL_WARNING" else "\033[93m"
            lines.append(
                f"  {idx}. [{dev.rule_id}] {sev_color}{dev.severity}{reset} on '{dev.affected_job}': "
                f"Expected '{dev.expected_parameter}', Got '{dev.actual_parameter}'."
            )
            lines.append(f"     Rationale: {dev.scientific_justification}")

    return "\n".join(lines)


def generate_latex_warning_callout_box(deviations: List[Any]) -> str:
    """
    Generate LaTeX tcolorbox markup suitable for Supporting Information.

    Args:
        deviations: List of DeviationRecord objects.

    Returns:
        str: LaTeX code snippet with tcolorbox.
    """
    if not deviations:
        return (
            r"\begin{tcolorbox}[colback=ForestGreen!5!white,colframe=ForestGreen!75!black,title=\textbf{Methodological Compliance Declaration}]"
            "\n"
            r"All quantum chemical calculations reported herein conform strictly to the standard operating procedures defined in Method Matrix v4.2 with zero unapproved deviations."
            "\n"
            r"\end{tcolorbox}"
        )

    has_warning = any(getattr(d, "severity", "") == "METHODOLOGICAL_WARNING" for d in deviations)
    colframe = "Crimson" if has_warning else "Amber!80!black"
    colback = "Crimson!5!white" if has_warning else "Amber!5!white"
    box_title = "Methodological Warning Notice" if has_warning else "Methodological Deviation Justifications"

    lines = [
        f"\\begin{{tcolorbox}}[colback={colback},colframe={colframe},title=\\textbf{{{box_title}}}]",
        r"\textbf{The following calculation deviations were audited and justified:}",
        r"\begin{itemize}",
    ]
    for d in deviations:
        lines.append(
            f"  \\item \\textbf{{{d.rule_id}}} (\\textit{{{d.severity}}}): {d.affected_job} -- Expected \\texttt{{{d.expected_parameter}}}, used \\texttt{{{d.actual_parameter}}}. {d.scientific_justification}"
        )
    lines.extend([
        r"\end{itemize}",
        r"\end{tcolorbox}",
    ])
    return "\n".join(lines)


def generate_html_warning_badge(verdict: str, score: float) -> str:
    """
    Render clean SVG/HTML status badge for UI and web dashboard display.

    Args:
        verdict: Audit verdict string.
        score: Compliance percentage.

    Returns:
        str: HTML/SVG badge snippet.
    """
    if verdict == "FULLY_COMPLIANT":
        badge_color = "#28a745"
        label = "COMPLIANT"
    elif verdict == "COMPLIANT_WITH_JUSTIFIED_DEVIATIONS":
        badge_color = "#ffc107"
        label = "JUSTIFIED DEVIATIONS"
    else:
        badge_color = "#dc3545"
        label = "NON-COMPLIANT"

    return (
        f"<span class='badge' style='background-color: {badge_color}; color: white; padding: 6px 12px; font-size: 13px; font-weight: bold; border-radius: 4px;'>"
        f"{label} ({score:.1f}%)"
        f"</span>"
    )

