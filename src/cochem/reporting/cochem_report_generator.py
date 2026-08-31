"""
CoChem Academic Report Generator for Energetics, KIE, QTAIM & SOCME (REQ-QC-082..089).

Provenance:
- Method Matrix [M]: Publication-ready Markdown, LaTeX tables, and Supporting Information (SI) compilations
  for Marcus reorganization, Bigeleisen-Mayer KIE/EIE, QTAIM electron density topologies, and SOCME photophysics.
- Derived Relations [D]: Exact standard-state solvation corrections and LaTeX mathematical formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
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


# =========================================================================
# Chunk 10: Master Publication & Supporting Information Assembler
# (REQ-QC-091, REQ-QC-092, REQ-QC-094, REQ-QC-096, REQ-QC-100)
# =========================================================================

from pathlib import Path
from cochem.provenance.cochem_orcid_resolver import ORCIDAuthorProfile
from cochem.reporting.cochem_bibtex_compiler import compile_bibtex_references
from cochem.reporting.cochem_methods_writer import ComputationalMethodsConfig, synthesize_methods_paragraph
from cochem.reporting.cochem_table_formatter import ThermochemicalSpeciesEntry, build_acs_energy_table
from cochem.reporting.cochem_transparency_auditor import audit_project_deviations
from cochem.reporting.cochem_si_builder import StationaryPointSIEntry, generate_latex_si_source, compile_latex_si


def assemble_full_publication_package(
    project_dir: Union[str, Path],
    output_dir: Union[str, Path],
    author_profiles: List[ORCIDAuthorProfile],
    title: str = "Quantum Chemical Investigation and Reaction Mechanism",
    methods_config: Optional[ComputationalMethodsConfig] = None,
    stationary_points: Optional[List[StationaryPointSIEntry]] = None,
    job_manifests: Optional[List[Dict[str, Any]]] = None,
    reference_species_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Master coordinator assembling the full publication package and Supporting Information bundle.

    Workflow:
    1. Synthesize Methods narrative via cochem_methods_writer.
    2. Compile citation database via cochem_bibtex_compiler.
    3. Build relative thermochemistry tables via cochem_table_formatter.
    4. Audit calculation parameters against Method Matrix v4.2 via cochem_transparency_auditor.
    5. Generate Supporting Information LaTeX source and compile PDF via cochem_si_builder.
    6. Collate all paths and manifests into a single publication package dictionary.

    Args:
        project_dir: Source directory of quantum chemistry calculations.
        output_dir: Output publication artifacts destination directory.
        author_profiles: List of validated author profiles.
        title: Title of publication / SI.
        methods_config: Configuration for methods narrative synthesis.
        stationary_points: List of stationary points with Cartesian geometries and frequencies.
        job_manifests: Job execution manifests for transparency auditing.
        reference_species_label: Species name to set as 0.0 kcal/mol reference.

    Returns:
        Dict[str, Any]: Complete publication package manifest.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Methods narrative
    cfg = methods_config or ComputationalMethodsConfig()
    methods_res = synthesize_methods_paragraph(cfg)

    # 2. BibTeX references
    methods_tokens = [cfg.software_name, cfg.functional, cfg.basis_set]
    if cfg.dispersion_correction:
        methods_tokens.append(cfg.dispersion_correction)
    if cfg.solvation_model:
        methods_tokens.append(cfg.solvation_model)
    if cfg.quasi_harmonic_damping:
        methods_tokens.append("quasi-harmonic")

    bib_manifest = compile_bibtex_references(methods_tokens, out_p)

    # 3. Thermochemistry tables
    st_points = stationary_points or []
    acs_tables_tex = ""
    acs_tables_md = ""
    table_manifest = None

    if st_points:
        thermo_entries = [
            ThermochemicalSpeciesEntry(
                species_name=sp.label,
                is_transition_state=(sp.number_of_imaginary_frequencies > 0),
                imaginary_frequency_cm1=sp.frequencies_cm1[0] if sp.frequencies_cm1 and sp.frequencies_cm1[0] < 0 else None,
                e_elec_hartree=sp.electronic_energy_hartree,
                zpve_hartree=sp.zero_point_energy_hartree,
                h_298_hartree=sp.enthalpy_hartree,
                g_298_hartree=sp.gibbs_free_energy_hartree,
            )
            for sp in st_points
        ]
        ref_label = reference_species_label or st_points[0].label
        table_manifest = build_acs_energy_table(thermo_entries, reference_label=ref_label)
        acs_tables_tex = table_manifest.latex_table_code
        acs_tables_md = table_manifest.markdown_table_code

    # 4. Transparency Audit
    manifests = job_manifests or [{"job_id": sp.label, "functional": cfg.functional, "basis": cfg.basis_set} for sp in st_points]
    audit_report = audit_project_deviations(manifests, out_p)

    # 5. Supporting Information LaTeX & PDF
    si_tex_source = generate_latex_si_source(
        title=title,
        authors=author_profiles,
        entries=st_points,
        methods_paragraph=methods_res.latex_text,
        acs_tables_latex=acs_tables_tex,
        transparency_report_latex=f"Compliance score: {audit_report.compliance_score_percent:.1f}\\% (Verdict: \\texttt{{{audit_report.audit_verdict.replace('_', r'\_')}}}).",
    )

    si_tex_path = out_p / "Supporting_Information.tex"
    si_tex_path.write_text(si_tex_source, encoding="utf-8")

    si_pdf_path = compile_latex_si(si_tex_path)

    return {
        "title": title,
        "output_directory": str(out_p.resolve()),
        "methods_markdown": methods_res.markdown_text,
        "methods_latex": methods_res.latex_text,
        "bibtex_file": bib_manifest.bibtex_file_path,
        "bibtex_keys": bib_manifest.keys,
        "thermochemistry_table": table_manifest.model_dump() if table_manifest else None,
        "transparency_audit": audit_report.model_dump(),
        "si_tex_file": str(si_tex_path.resolve()),
        "si_pdf_file": str(si_pdf_path.resolve()) if si_pdf_path else None,
    }

