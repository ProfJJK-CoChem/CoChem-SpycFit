"""
CoChem Diagnostic Warning Engine & Stability Flag Formatter (REQ-QC-081, REQ-QC-087).

Provenance:
- Method Matrix [M]: High-visibility diagnostic warning banners for spin contamination,
  electronic orbital Hessian instabilities, gradient threshold alerts, and Level-of-Theory discrepancies.
- Derived Relations [D]: ANSI terminal color escapes and GitHub-flavored markdown alert boxes.
"""

from __future__ import annotations

from typing import List, Literal, Optional
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
