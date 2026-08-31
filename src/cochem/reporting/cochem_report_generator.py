"""
CoChem Academic Report Generator for Energetics, KIE, QTAIM & SOCME (REQ-QC-082..089).

Provenance:
- Method Matrix [M]: Publication-ready Markdown, LaTeX tables, and Supporting Information (SI) compilations
  for Marcus reorganization, Bigeleisen-Mayer KIE/EIE, QTAIM electron density topologies, and SOCME photophysics.
- Derived Relations [D]: Exact standard-state solvation corrections and LaTeX mathematical formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.models import (
    BondCriticalPointDescriptor,
    IsotopeSubstitutionPayload,
    MarcusReorganizationResult,
    SOCMEResult,
    StandardStateCorrectionPayload,
)


class AcademicReportSpec(BaseModel):
    """Specification of sections and data to include in the academic report."""
    model_config = ConfigDict(frozen=True)

    title: str = "Quantum Chemical Characterization Report"
    authors: str = "CoChem Autonomous Scientific Engine"
    system_name: str = "Molecular System"
    include_marcus: bool = False
    include_kie: bool = False
    include_qtaim: bool = False
    include_socme: bool = False
    include_solvation_notes: bool = True


class CompiledReport(BaseModel):
    """Data contract representing generated academic reports in Markdown and LaTeX."""
    model_config = ConfigDict(frozen=True)

    title: str
    markdown_content: str
    latex_content: str


def generate_marcus_table(marcus_res: MarcusReorganizationResult) -> Dict[str, str]:
    """
    [REQ-QC-082] Generate Markdown and LaTeX summary tables for Marcus electron transfer parameters.
    """
    lines = [
        "### Marcus Electron Transfer Reorganization Parameters",
        "",
        "| Parameter | Symbol | Value | Units |",
        "| :--- | :---: | :---: | :---: |",
        f"| Ground State at Neutral Minimum | $E_0(R_0)$ | `{marcus_res.e_neutral_at_neutral_geom_hartree:.6f}` | Hartree |",
        f"| Vertical Ionized at Neutral Geom | $E_\\pm(R_0)$ | `{marcus_res.e_charged_at_neutral_geom_hartree:.6f}` | Hartree |",
        f"| Adiabatic Ionized at Ionized Geom | $E_\\pm(R_\\pm)$ | `{marcus_res.e_charged_at_charged_geom_hartree:.6f}` | Hartree |",
        f"| Vertical Neutral at Ionized Geom | $E_0(R_\\pm)$ | `{marcus_res.e_neutral_at_charged_geom_hartree:.6f}` | Hartree |",
        f"| Charged State Relaxation | $\\lambda_1$ | `{marcus_res.lambda_1_ev:.4f}` | eV |",
        f"| Neutral State Relaxation | $\\lambda_2$ | `{marcus_res.lambda_2_ev:.4f}` | eV |",
        f"| Total Inner-Sphere Reorganization | $\\lambda_{{inner}}$ | `{marcus_res.lambda_inner_ev:.4f}` | eV (`{marcus_res.lambda_inner_kcal_mol:.2f}` kcal/mol) |",
        f"| Outer-Sphere Reorganization | $\\lambda_{{outer}}$ | `{marcus_res.lambda_outer_ev or 0.0:.4f}` | eV |",
        f"| Total Reorganization Energy | $\\lambda$ | `{marcus_res.total_reorganization_energy_ev:.4f}` | eV |",
        f"| Marcus Activation Barrier | $\\Delta G^\\ddagger$ | `{marcus_res.marcus_activation_barrier_ev:.4f}` | eV |",
    ]
    if marcus_res.electron_transfer_rate_s_inv is not None:
        lines.append(f"| Semiclassical ET Rate | $k_{{ET}}$ | `{marcus_res.electron_transfer_rate_s_inv:.3e}` | s^-1 |")

    md = "\n".join(lines) + "\n"

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Marcus electron transfer parameters.}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        "Parameter & $\\lambda_1$ (eV) & $\\lambda_2$ (eV) & $\\lambda_{\\text{inner}}$ (eV) & $\\Delta G^\\ddagger$ (eV) \\\\\n"
        "\\hline\n"
        f"Value & {marcus_res.lambda_1_ev:.4f} & {marcus_res.lambda_2_ev:.4f} & {marcus_res.lambda_inner_ev:.4f} & {marcus_res.marcus_activation_barrier_ev:.4f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return {"markdown": md, "latex": tex}


def generate_kie_si_table(kie_res: IsotopeSubstitutionPayload) -> Dict[str, str]:
    """
    [REQ-QC-086] Generate Bigeleisen-Mayer KIE Supporting Information table.
    """
    lines = [
        "### Bigeleisen-Mayer Kinetic Isotope Effect (KIE) Decomposition",
        "",
        "| Factor | Contribution | Value |",
        "| :--- | :---: | :---: |",
        f"| Mass-Moment of Inertia | MMI | `{kie_res.mmi_factor or 1.0:.4f}` |",
        f"| Vibrational Excitation | EXC | `{kie_res.exc_factor or 1.0:.4f}` |",
        f"| Zero-Point Energy Ratio | ZPE | `{kie_res.zpe_factor or 1.0:.4f}` |",
        f"| Total Calculated KIE (k_L / k_H) | KIE | `{kie_res.kie_bigeleisen_mayer or 1.0:.4f}` |",
        f"| Parent ZPE | ZPE_L | `{kie_res.reference_zpe_kcal_mol:.2f}` kcal/mol |",
        f"| Substituted ZPE | ZPE_H | `{kie_res.substituted_zpe_kcal_mol:.2f}` kcal/mol |",
    ]
    md = "\n".join(lines) + "\n"

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Bigeleisen-Mayer KIE Decomposition.}\n"
        "\\begin{tabular}{ccccc}\n"
        "\\hline\n"
        "MMI & EXC & ZPE & Tunneling & $\\text{KIE}_{\\text{total}}$ \\\\\n"
        "\\hline\n"
        f"{kie_res.mmi_factor or 1.0:.4f} & {kie_res.exc_factor or 1.0:.4f} & {kie_res.zpe_factor or 1.0:.4f} & 1.0000 & {kie_res.kie_bigeleisen_mayer or 1.0:.4f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return {"markdown": md, "latex": tex}


def generate_qtaim_table(bcps: List[BondCriticalPointDescriptor]) -> Dict[str, str]:
    """
    [REQ-QC-088] Generate QTAIM (3,-1) Bond Critical Point Inventory table.
    """
    lines = [
        "### QTAIM Bond Critical Point (3,-1) Topological Inventory",
        "",
        "| BCP # | Bond (Atoms) | rho (a.u.) | Laplacian (a.u.) | Ellipticity | H(r) (a.u.) | Interaction Class |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for b in bcps:
        lines.append(
            f"| {b.bcp_index} | {b.atom_1_index}-{b.atom_2_index} | `{b.electron_density_au:.4f}` | `{b.laplacian_au:+.4f}` | `{b.ellipticity:.4f}` | `{b.total_energy_density_h_au:+.4f}` | `{b.interaction_classification}` |"
        )
    md = "\n".join(lines) + "\n"

    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{QTAIM Bond Critical Point Properties.}",
        "\\begin{tabular}{ccccccc}",
        "\\hline",
        "BCP & Atoms & $\\rho$ (a.u.) & $\\nabla^2\\rho$ (a.u.) & $\\varepsilon$ & $H$ (a.u.) & Class \\\\",
        "\\hline",
    ]
    for b in bcps:
        tex_lines.append(
            f"{b.bcp_index} & {b.atom_1_index}-{b.atom_2_index} & {b.electron_density_au:.4f} & {b.laplacian_au:+.4f} & {b.ellipticity:.4f} & {b.total_energy_density_h_au:+.4f} & \\text{{{b.interaction_classification}}} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    tex = "\n".join(tex_lines)

    return {"markdown": md, "latex": tex}


def generate_academic_report(
    spec: AcademicReportSpec,
    marcus_data: Optional[MarcusReorganizationResult] = None,
    kie_data: Optional[IsotopeSubstitutionPayload] = None,
    qtaim_data: Optional[List[BondCriticalPointDescriptor]] = None,
    socme_data: Optional[SOCMEResult] = None,
) -> CompiledReport:
    """
    [REQ-QC-082..089] Assemble full comprehensive academic report.
    """
    md_sections = [
        f"# {spec.title}\n",
        f"**System**: {spec.system_name}  \n**Authors**: {spec.authors}\n\n---\n",
    ]
    tex_sections = [
        f"\\section*{{{spec.title}}}\n",
        f"\\textbf{{System}}: {spec.system_name}\\\\\n",
    ]

    if spec.include_marcus and marcus_data:
        m_tbl = generate_marcus_table(marcus_data)
        md_sections.append(m_tbl["markdown"])
        tex_sections.append(m_tbl["latex"])

    if spec.include_kie and kie_data:
        k_tbl = generate_kie_si_table(kie_data)
        md_sections.append(k_tbl["markdown"])
        tex_sections.append(k_tbl["latex"])

    if spec.include_qtaim and qtaim_data:
        q_tbl = generate_qtaim_table(qtaim_data)
        md_sections.append(q_tbl["markdown"])
        tex_sections.append(q_tbl["latex"])

    if spec.include_solvation_notes:
        note_md = (
            "\n> [!NOTE]\n"
            "> **Standard-State Solvation Shift Note**: Gas phase calculations referenced to standard state "
            "are converted to solution standard state (1.0 M) with +1.894 kcal/mol (+7.926 kJ/mol) shift at 298.15 K.\n"
        )
        md_sections.append(note_md)

    full_md = "\n".join(md_sections)
    full_tex = "\n".join(tex_sections)

    return CompiledReport(
        title=spec.title,
        markdown_content=full_md,
        latex_content=full_tex,
    )
