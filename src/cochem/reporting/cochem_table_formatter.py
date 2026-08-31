"""
CoChem ACS-Compliant Relative Thermochemistry Table Formatter Engine (REQ-QC-096).

Provenance:
- ACS Style Guide and JACS / JCTC Table Design Rules (booktabs, zero vertical rules).
- CODATA 2022 Recommended Values: 1 Hartree = 627.5094740631 kcal/mol = 2625.499639479 kJ/mol.
- Strict siunitx alignment syntax for multi-column academic publication.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_exceptions import ReferenceSpeciesNotFoundError

# CODATA 2022 Fundamental Physical Constants
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_KJ_MOL = 2625.499639479


class ThermochemicalSpeciesEntry(BaseModel):
    """Absolute thermochemical properties for a molecular stationary point."""
    model_config = ConfigDict(frozen=True)

    species_name: str = Field(description="Molecular identifier or stationary point label")
    is_transition_state: bool = Field(default=False, description="True if state is a first-order saddle point")
    imaginary_frequency_cm1: Optional[float] = Field(default=None, description="Imaginary vibrational mode in cm^-1")
    e_elec_hartree: float = Field(description="Electronic SCF energy in Hartree")
    zpve_hartree: float = Field(description="Harmonic zero-point vibrational energy in Hartree")
    h_298_hartree: float = Field(description="Total enthalpy at 298.15 K in Hartree")
    g_298_hartree: float = Field(description="Total Gibbs free energy at 298.15 K in Hartree")
    g_qh_298_hartree: Optional[float] = Field(default=None, description="Quasi-harmonic Gibbs free energy at 298.15 K")


class ACSRelativeEnergyRow(BaseModel):
    """Relative energy row formatted for academic publication tables."""
    model_config = ConfigDict(frozen=True)

    species_name: str = Field(description="Species name with formatting annotations")
    delta_e: float = Field(description="Relative electronic energy Delta E")
    delta_e0: float = Field(description="Relative zero-point corrected energy Delta E_0")
    delta_h: float = Field(description="Relative enthalpy Delta H_298")
    delta_g: float = Field(description="Relative Gibbs free energy Delta G_298")
    delta_g_qh: Optional[float] = Field(default=None, description="Relative quasi-harmonic Gibbs energy Delta G_qh")
    imaginary_frequency_str: str = Field(description="Formatted imaginary frequency with dagger symbol")


class ACSTableManifest(BaseModel):
    """Manifest containing structured rows and ready-to-publish LaTeX, Markdown, and HTML tables."""
    model_config = ConfigDict(frozen=True)

    table_title: str = Field(description="Title caption for the table")
    reference_species: str = Field(description="Species chosen as thermodynamic reference zero (0.00)")
    energy_units: str = Field(description="Energy units ('kcal/mol' or 'kJ/mol')")
    rows: List[ACSRelativeEnergyRow] = Field(description="Computed relative energy entries")
    latex_table_code: str = Field(description="Formatted LaTeX booktabs table code")
    markdown_table_code: str = Field(description="GitHub Flavored Markdown table")
    html_table_code: str = Field(description="Clean HTML table card snippet")


def build_acs_energy_table(
    entries: List[ThermochemicalSpeciesEntry],
    reference_label: str,
    energy_units: str = "kcal_mol",
    table_title: str = "Relative Thermochemical Parameters of Reaction Profile",
) -> ACSTableManifest:
    """
    Compute relative energies from absolute Hartree values and generate publication-grade tables.

    Args:
        entries: List of stationary points with absolute thermochemistry.
        reference_label: Species name to set as reference (0.0 kcal/mol).
        energy_units: 'kcal_mol' (default) or 'kj_mol'.
        table_title: Caption for LaTeX/Markdown tables.

    Returns:
        ACSTableManifest: Complete formatted table bundle.

    Raises:
        ReferenceSpeciesNotFoundError: If reference_label is not present in entries.
    """
    ref_entry: Optional[ThermochemicalSpeciesEntry] = None
    all_names: List[str] = []
    for e in entries:
        all_names.append(e.species_name)
        if e.species_name == reference_label:
            ref_entry = e

    if ref_entry is None:
        raise ReferenceSpeciesNotFoundError(
            reference_label=reference_label,
            available_species=all_names,
        )

    conv = HARTREE_TO_KCAL_MOL if energy_units == "kcal_mol" else HARTREE_TO_KJ_MOL
    unit_str = "kcal/mol" if energy_units == "kcal_mol" else "kJ/mol"

    ref_e = ref_entry.e_elec_hartree
    ref_e0 = ref_entry.e_elec_hartree + ref_entry.zpve_hartree
    ref_h = ref_entry.h_298_hartree
    ref_g = ref_entry.g_298_hartree
    ref_g_qh = ref_entry.g_qh_298_hartree if ref_entry.g_qh_298_hartree is not None else ref_g

    has_qh = any(e.g_qh_298_hartree is not None for e in entries)

    rows: List[ACSRelativeEnergyRow] = []

    for ent in entries:
        de = (ent.e_elec_hartree - ref_e) * conv
        de0 = ((ent.e_elec_hartree + ent.zpve_hartree) - ref_e0) * conv
        dh = (ent.h_298_hartree - ref_h) * conv
        dg = (ent.g_298_hartree - ref_g) * conv
        dg_qh = None
        if ent.g_qh_298_hartree is not None:
            dg_qh = (ent.g_qh_298_hartree - ref_g_qh) * conv

        imag_str = "--"
        if ent.is_transition_state and ent.imaginary_frequency_cm1 is not None:
            imag_str = f"{abs(ent.imaginary_frequency_cm1):.1f}i†"
        elif ent.is_transition_state:
            imag_str = "TS†"

        rows.append(
            ACSRelativeEnergyRow(
                species_name=ent.species_name,
                delta_e=de,
                delta_e0=de0,
                delta_h=dh,
                delta_g=dg,
                delta_g_qh=dg_qh,
                imaginary_frequency_str=imag_str,
            )
        )

    # 1. LaTeX Code Generation
    qh_tex_head = r" & {$\Delta G_{\text{qh}}$}" if has_qh else ""
    col_def = "l S[table-format=-2.2] S[table-format=-2.2] S[table-format=-2.2] S[table-format=-2.2]" + (" S[table-format=-2.2]" if has_qh else "") + " c"

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{table_title} (Energies in {unit_str} relative to {reference_label}).}}",
        r"\label{tab:thermochemistry}",
        f"\\begin{{tabular}}{{{col_def}}}",
        r"\toprule",
        f"{{Species}} & {{$\\Delta E$}} & {{$\\Delta E_0$}} & {{$\\Delta H_{{298}}$}} & {{$\\Delta G_{{298}}$}}{qh_tex_head} & {{$\\nu_{{\\text{{imag}}}}$}} \\\\",
        r"\midrule",
    ]

    for r, ent in zip(rows, entries):
        name_clean = r.species_name.replace("_", r"\_")
        name_tex = f"\\textbf{{{name_clean}}}" if ent.is_transition_state else name_clean
        qh_val = f" & {r.delta_g_qh:.2f}" if (has_qh and r.delta_g_qh is not None) else (" & --" if has_qh else "")
        imag_tex = f"${abs(ent.imaginary_frequency_cm1):.1f}i^\\dagger$" if (ent.is_transition_state and ent.imaginary_frequency_cm1) else ("TS$^\\dagger$" if ent.is_transition_state else "--")
        latex_lines.append(
            f"{name_tex} & {r.delta_e:.2f} & {r.delta_e0:.2f} & {r.delta_h:.2f} & {r.delta_g:.2f}{qh_val} & {imag_tex} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex_table_code = "\n".join(latex_lines)

    # 2. Markdown Code Generation
    qh_md_head = " | $\\Delta G_{\\text{qh}}$" if has_qh else ""
    qh_md_div = " | :---:" if has_qh else ""

    md_lines = [
        f"### {table_title}",
        f"*Energies in {unit_str} relative to reference species `{reference_label}`.*",
        "",
        f"| Species | $\\Delta E$ | $\\Delta E_0$ | $\\Delta H_{{298}}$ | $\\Delta G_{{298}}${qh_md_head} | $\\nu_{{\\text{{imag}}}}$ (cm$^{{-1}}$) |",
        f"| :--- | :---: | :---: | :---: | :---:{qh_md_div} | :---: |",
    ]

    for r, ent in zip(rows, entries):
        name_md = f"**{r.species_name}**" if ent.is_transition_state else r.species_name
        qh_val_md = f" | `{r.delta_g_qh:.2f}`" if (has_qh and r.delta_g_qh is not None) else (" | --" if has_qh else "")
        md_lines.append(
            f"| {name_md} | `{r.delta_e:.2f}` | `{r.delta_e0:.2f}` | `{r.delta_h:.2f}` | `{r.delta_g:.2f}`{qh_val_md} | `{r.imaginary_frequency_str}` |"
        )

    markdown_table_code = "\n".join(md_lines)

    # 3. HTML Table Generation
    html_lines = [
        f"<div class='cochem-table-card'>",
        f"  <h4>{table_title}</h4>",
        f"  <p class='text-muted'>Energies in {unit_str} relative to <code>{reference_label}</code>.</p>",
        f"  <table class='table table-sm table-striped table-hover'>",
        f"    <thead><tr><th>Species</th><th>&Delta;E</th><th>&Delta;E<sub>0</sub></th><th>&Delta;H<sub>298</sub></th><th>&Delta;G<sub>298</sub></th>"
        + ("<th>&Delta;G<sub>qh</sub></th>" if has_qh else "") + "<th>&nu;<sub>imag</sub></th></tr></thead>",
        f"    <tbody>",
    ]

    for r, ent in zip(rows, entries):
        bold_class = " class='fw-bold'" if ent.is_transition_state else ""
        qh_html = f"<td>{r.delta_g_qh:.2f}</td>" if (has_qh and r.delta_g_qh is not None) else ("<td>--</td>" if has_qh else "")
        html_lines.append(
            f"      <tr{bold_class}><td>{r.species_name}</td><td>{r.delta_e:.2f}</td><td>{r.delta_e0:.2f}</td>"
            f"<td>{r.delta_h:.2f}</td><td>{r.delta_g:.2f}</td>{qh_html}<td>{r.imaginary_frequency_str}</td></tr>"
        )

    html_lines.extend([
        f"    </tbody>",
        f"  </table>",
        f"</div>",
    ])
    html_table_code = "\n".join(html_lines)

    return ACSTableManifest(
        table_title=table_title,
        reference_species=reference_label,
        energy_units=unit_str,
        rows=rows,
        latex_table_code=latex_table_code,
        markdown_table_code=markdown_table_code,
        html_table_code=html_table_code,
    )
