"""# zero-stub anti-spoofing engine
CoChem Method Matrix Compliance & Anti-P-Hacking Rule Auditor (cochem_base.calc.cochem_method_auditor)

Evaluates computational quantum chemistry specifications against Method Matrix invariants (§3.0–§9.0)
to prevent methodological p-hacking, uncorrected spectroscopic comparison, and improper basis corrections.

Governing Requirements:
- REQ-63 (FR-63.1 - FR-63.3): Method Matrix Compliance & Anti-P-Hacking Rule Auditor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cochem_method_auditor")


def audit_calculation_parameters(
    calculation_specs: Dict[str, Any],
    academic_exemption: bool = False,
    exemption_justification: Optional[str] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Audit quantum chemistry calculation specifications against Method Matrix rules.
    
    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations: List[Dict[str, Any]] = []

    # Rule MM-8B.3: Additive Diffuse Basis Set Corrections on Non-Covalent Complexes
    if calculation_specs.get("is_non_covalent", False) and calculation_specs.get("delta_e_diffuse_correction", False):
        violations.append({
            "rule": "MM-8B.3",
            "severity": "CRITICAL",
            "category": "METHODOLOGICAL_P_HACKING",
            "message": "Additive Delta E_diffuse corrections on non-covalent complexes are prohibited due to basis set superposition error amplification.",
            "remediation": "Use fully uncontracted/diffuse augmented basis sets directly (e.g., aug-cc-pVTZ or def2-TZVPPD).",
        })

    # Rule MM-9A.5: Small-System ONIOM Partitioning (Natoms < 15)
    atom_count = calculation_specs.get("atom_count")
    oniom_layers = calculation_specs.get("oniom_layers")
    if oniom_layers and atom_count is not None and atom_count < 15:
        violations.append({
            "rule": "MM-9A.5",
            "severity": "CRITICAL",
            "category": "UNPHYSICAL_PARTITIONING",
            "message": f"QM/QM2 ONIOM partitioning rejected for small system (Natoms = {atom_count} < 15). Boundary polarization errors dominate.",
            "remediation": "Treat the entire molecular system at the primary quantum chemical fidelity level without ONIOM partitioning.",
        })

    # Rule MM-3.0: Be vs B0 Conflation (Equilibrium vs Ground-State Rotational Constants)
    if calculation_specs.get("equilibrium_be_used", False):
        obs = calculation_specs.get("spectral_observable", "")
        if "microwave" in obs.lower() or "b0" in obs.lower():
            if not calculation_specs.get("vpt2_vibrational_correction", False):
                violations.append({
                    "rule": "MM-3.0",
                    "severity": "SPECTROSCOPIC_FLAG",
                    "category": "SPECTROSCOPIC_CONFLATION",
                    "message": "Equilibrium rotational constants (Be) compared directly against ground-state microwave observables (B0) without VPT2 Delta B_vib correction.",
                    "remediation": "Perform second-order vibrational perturbation theory (VPT2) or compute alpha vibration-rotation interaction tensors.",
                })

    # Rule MM-4.4: Numerical Grid Gating (defgrid1 minimization without defgrid3 promotion)
    grid = str(calculation_specs.get("grid", "")).lower()
    job_type = str(calculation_specs.get("job_type", "")).lower()
    if "defgrid1" in grid and ("final" in job_type or "sp" in job_type):
        violations.append({
            "rule": "MM-4.4",
            "severity": "WARNING",
            "category": "GRID_INACCURACY",
            "message": "Calculation terminating at energy minimum on defgrid1 without final promotion to defgrid3.",
            "remediation": "Promote final single-point energy and gradient evaluation to defgrid3 or defgrid4.",
        })

    # Rule MM-4.4: Hessian Optimization Guard
    if calculation_specs.get("calc_hess_during_opt", False):
        violations.append({
            "rule": "MM-4.4",
            "severity": "EFFICIENCY_WARNING",
            "category": "EXCESSIVE_COMPUTE",
            "message": "Exact analytic Hessian evaluated during every optimization step ('Calc_Hess true').",
            "remediation": "Employ model Hessian initialization (InHess XTB2 or Lindh) with BFGS / GDIIS updates.",
        })

    # Rule MM-QS-1: Convergence Thresholds
    tol_max_g = calculation_specs.get("tol_max_g")
    if tol_max_g is not None and tol_max_g > 1.0e-5:
        violations.append({
            "rule": "MM-QS-1",
            "severity": "CONVERGENCE_WARNING",
            "category": "LOOSE_CONVERGENCE",
            "message": f"Loose geometry convergence threshold detected (TolMaxG = {tol_max_g} > 1.0e-5).",
            "remediation": "Enforce tight geometry convergence criteria: TolMaxG <= 1.0e-5, TolE <= 1.0e-7.",
        })

    # Check for critical rejections
    has_critical = any(v["severity"] == "CRITICAL" for v in violations)

    if has_critical:
        if academic_exemption:
            if exemption_justification and len(exemption_justification.strip()) >= 10:
                logger.info(f"Academic demonstration exemption granted: {exemption_justification}")
                return True, violations
            else:
                return False, violations
        return False, violations

    return True, violations


def generate_methodology_audit_report(
    calculation_specs: Dict[str, Any],
    academic_exemption: bool = False,
    exemption_justification: Optional[str] = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Generate a formatted Markdown and structured methodology audit report.
    """
    is_valid, violations = audit_calculation_parameters(
        calculation_specs,
        academic_exemption=academic_exemption,
        exemption_justification=exemption_justification,
    )

    lines: List[str] = [
        "# CoChem Method Matrix Compliance Audit Report",
        f"**Audit Status:** {'PASSED' if is_valid else 'FAILED / CRITICAL REJECTION'}",
        f"**Job Type:** {calculation_specs.get('job_type', 'N/A')}",
        "",
    ]

    if academic_exemption and is_valid and any(v["severity"] == "CRITICAL" for v in violations):
        lines.append("> [!IMPORTANT]")
        lines.append("> **[ACADEMIC_EXEMPTION_GRANTED]**")
        lines.append(f"> Justification: {exemption_justification}")
        lines.append("")

    if not is_valid:
        lines.append("> [!CAUTION]")
        lines.append("> **[CRITICAL_P_HACKING_FLAG] Execution Blocked by Method Matrix Auditor**")
        lines.append("> The submitted calculation specification contains critical methodological violations.")
        lines.append("")

    if violations:
        lines.append("## Identified Methodological Findings")
        lines.append("| Rule | Severity | Category | Finding & Remediation |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for v in violations:
            lines.append(
                f"| `{v['rule']}` | **{v['severity']}** | {v['category']} | {v['message']}<br/>*Remediation:* {v['remediation']} |"
            )
        lines.append("")
    else:
        lines.append("All calculation parameters conform strictly to Method Matrix §3.0–§9.0 invariants.")

    report_markdown = "\n".join(lines)
    return is_valid, report_markdown, violations
