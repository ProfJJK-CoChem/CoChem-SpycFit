"""# zero-stub anti-spoofing engine
CoChem Multi-Target Warning Banner Renderer (cochem_base.io.cochem_banners)

Evaluates geometry optimization convergence residuals against Method Matrix §4.4 limits
and renders high-contrast, deterministic warning banners across Markdown, LaTeX SI, HTML, and ANSI CLI.

Governing Requirements:
- REQ-65 (FR-65.1 - FR-65.3): Unconverged Geometry Multi-Target Warning Banners.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CONVERGENCE_THRESHOLDS = {
    "TolMaxG": 1.0e-5,   # Maximum Cartesian gradient norm (Hartree/Bohr)
    "TolE": 1.0e-7,      # Electronic energy change (Hartree)
    "TolRMSG": 3.0e-6,   # Root-mean-square gradient (Hartree/Bohr)
    "TolRMSD": 5.0e-5,   # Root-mean-square displacement (Bohr)
}

CRITICAL_OVERRIDE_GRADIENT_CEILING: float = 5.0e-4


def evaluate_convergence_residuals(residuals: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate gradient and energy residuals against Method Matrix §4.4 thresholds.
    """
    exceeded: List[str] = []
    for key, threshold in CONVERGENCE_THRESHOLDS.items():
        if key in residuals and residuals[key] > threshold:
            exceeded.append(key)

    is_converged = (len(exceeded) == 0)
    requires_token = False
    if "TolMaxG" in residuals and residuals["TolMaxG"] > CRITICAL_OVERRIDE_GRADIENT_CEILING:
        requires_token = True

    return {
        "is_converged": is_converged,
        "exceeded_thresholds": exceeded,
        "requires_override_token": requires_token,
        "residuals": residuals,
    }


def render_unconverged_banner(
    residuals: Dict[str, float],
    format: str = "markdown",
    override_token: Optional[str] = None,
) -> str:
    """
    Render a high-contrast warning banner in the specified target format:
    'markdown', 'latex', 'html', or 'ansi'.
    """
    eval_res = evaluate_convergence_residuals(residuals)
    fmt = format.lower().strip()

    res_summary_md = ", ".join(f"{k}: {v}" for k, v in residuals.items())
    token_str = f" [Override Token: {override_token}]" if override_token else ""

    if fmt == "markdown":
        lines = [
            "> [!CAUTION]",
            "> **WARNING: UNCONVERGED GEOMETRY FORCED**",
            f"> Calculation geometry optimization did not reach Method Matrix §4.4 convergence thresholds{token_str}.",
            f"> **Residual Metrics:** {res_summary_md}",
            "> *Spectroscopic Disclaimer:* Rotational constants derived from unconverged gradients carry significant geometric error.",
        ]
        return "\n".join(lines)

    elif fmt in ("latex", "tex"):
        lines = [
            r"\begin{tcolorbox}[colback=red!10,colframe=red!80!black,title=\textbf{WARNING: UNCONVERGED GEOMETRY FORCED}]",
            r"\textbf{Method Matrix §4.4 Violation:} Geometry optimization terminated prior to full gradient convergence." + (f" Token: \\texttt{{{override_token}}}" if override_token else ""),
            r"\begin{itemize}",
        ]
        for k, v in residuals.items():
            thresh = CONVERGENCE_THRESHOLDS.get(k, "N/A")
            lines.append(rf"  \item \texttt{{{k}}}: {v:.2e} (Threshold: {thresh})")
        lines.extend([
            r"\end{itemize}",
            r"\textit{Spectroscopic Caution: Rotational constants computed from unconverged stationary points must not be published as equilibrium structures without VPT2 correction.}",
            r"\end{tcolorbox}",
        ])
        return "\n".join(lines)

    elif fmt in ("html", "gui"):
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;'>{k}</td><td style='padding:4px 8px;font-family:monospace;'>{v:.2e}</td><td style='padding:4px 8px;font-family:monospace;'>{CONVERGENCE_THRESHOLDS.get(k, 'N/A')}</td></tr>"
            for k, v in residuals.items()
        )
        return (
            f'<div class="cochem-unconverged-warning" style="background-color: #ffebee; border: 2px solid #D32F2F; color: #b71c1c; padding: 12px; border-radius: 4px; margin: 10px 0;">'
            f'<h4 style="margin: 0 0 8px 0; color: #D32F2F; font-weight: bold;">&#9888; WARNING: UNCONVERGED GEOMETRY FORCED{token_str}</h4>'
            f'<p style="margin: 0 0 8px 0; font-size: 13px;">Geometry optimization residuals exceeded Method Matrix §4.4 limits. Data flagged for non-standard deviation.</p>'
            f'<table style="border-collapse: collapse; width: 100%; font-size: 12px; background: #ffffff; border: 1px solid #ffcdd2;">'
            f'<tr style="background:#ef9a9a; font-weight:bold;"><th style="padding:4px 8px; text-align:left;">Residual Parameter</th><th style="padding:4px 8px; text-align:left;">Value</th><th style="padding:4px 8px; text-align:left;">Standard Threshold</th></tr>'
            f'{rows}'
            f'</table>'
            f'</div>'
        )

    elif fmt in ("ansi", "cli", "terminal"):
        border = "═" * 78
        return (
            f"\033[91;1m╔{border}╗\n"
            f"║                     WARNING: UNCONVERGED GEOMETRY FORCED                     ║\n"
            f"║ Residuals: {res_summary_md:<64} ║\n"
            f"║ CAUTION: Optimization gradients exceeded Method Matrix §4.4 threshold!       ║\n"
            f"╚{border}╝\033[0m"
        )

    else:
        return f"[WARNING: UNCONVERGED GEOMETRY FORCED] {res_summary_md}"
