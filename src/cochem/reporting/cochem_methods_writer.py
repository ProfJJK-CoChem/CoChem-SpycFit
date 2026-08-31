"""
CoChem Automated "Computational Methods" Narrative Synthesizer (REQ-QC-092).

Provenance:
- Method Matrix [M]: Standardized academic prose for quantum chemistry computational methods
  following ACS, Wiley, and Nature chemistry style guidelines.
- Accurate inclusion of standard-state corrections, quasi-harmonic vibrational damping,
  and embedded LaTeX \\cite{...} bibliographic keys.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.reporting.cochem_bibtex_compiler import CANONICAL_CITATION_DATABASE


class ComputationalMethodsConfig(BaseModel):
    """Configuration parameters for synthesizing computational methods narrative."""
    model_config = ConfigDict(frozen=True)

    software_name: str = Field(default="ORCA", description="Quantum chemistry package (e.g. ORCA, Gaussian)")
    software_version: str = Field(default="5.0.4", description="Release version of the QM engine")
    functional: str = Field(default="wB97X-D4", description="Density functional (e.g. B3LYP, wB97X-D4, M06-2X)")
    basis_set: str = Field(default="def2-TZVP", description="Primary orbital basis set (e.g. def2-TZVP)")
    auxiliary_basis: Optional[str] = Field(default="def2/J", description="Auxiliary Coulomb/JK fitting basis")
    dispersion_correction: Optional[str] = Field(default="D4", description="Empirical dispersion correction (e.g. D3BJ, D4)")
    grid_level: str = Field(default="defgrid3", description="Integration grid level (e.g. defgrid2, defgrid3)")
    solvation_model: Optional[str] = Field(default="SMD", description="Implicit continuum solvation model (e.g. SMD, CPCM)")
    solvent: Optional[str] = Field(default="water", description="Solvent name (e.g. water, dichloromethane)")
    quasi_harmonic_damping: bool = Field(default=True, description="Whether quasi-harmonic entropy corrections are applied")
    quasi_harmonic_cutoff_cm1: float = Field(default=100.0, description="Quasi-harmonic rotor frequency cutoff in cm^-1")
    temperature_kelvin: float = Field(default=298.15, description="Thermodynamic reference temperature in Kelvin")
    pressure_atm: float = Field(default=1.0, description="Thermodynamic standard state gas pressure in atm")
    include_standard_state_correction: bool = Field(default=True, description="Whether to include Delta G(1 atm -> 1 M) shift")


class SynthesizedMethodsParagraph(BaseModel):
    """Synthesized Computational Methods narrative in Markdown and LaTeX formats."""
    model_config = ConfigDict(frozen=True)

    markdown_text: str = Field(description="Methods narrative in GitHub-Flavored Markdown")
    latex_text: str = Field(description="Methods narrative in LaTeX with embedded \\cite{...} commands")
    referenced_bibtex_keys: List[str] = Field(description="List of cited BibTeX keys")


def _resolve_citation_key(token: str) -> Optional[str]:
    """Helper to find matching BibTeX key from the canonical citation database."""
    cleaned = token.strip().lower().replace("(", "").replace(")", "").replace(" ", "").replace("_", "-")
    if cleaned in CANONICAL_CITATION_DATABASE:
        return CANONICAL_CITATION_DATABASE[cleaned]["key"]
    return None


def synthesize_methods_paragraph(config: ComputationalMethodsConfig) -> SynthesizedMethodsParagraph:
    """
    Generate publication-grade 'Computational Methods' prose in synchronized Markdown and LaTeX.

    Args:
        config: Specifications of the quantum chemistry methodology.

    Returns:
        SynthesizedMethodsParagraph: Structured container with Markdown, LaTeX, and cited BibTeX keys.
    """
    cited_keys: List[str] = []

    def get_cite(token: str) -> str:
        key = _resolve_citation_key(token)
        if key:
            if key not in cited_keys:
                cited_keys.append(key)
            return f"\\cite{{{key}}}"
        return ""

    # Software citation
    sw_token = config.software_name.lower()
    sw_cite = get_cite(sw_token)
    sw_str_tex = f"{config.software_name} (version {config.software_version}){(' ' + sw_cite) if sw_cite else ''}"
    sw_str_md = f"{config.software_name} (version {config.software_version})"

    # Functional citation
    func_cite = get_cite(config.functional)
    func_tex = f"{config.functional}{(' ' + func_cite) if func_cite else ''}"

    # Basis set citation
    basis_cite = get_cite(config.basis_set)
    basis_tex = f"{config.basis_set}{(' ' + basis_cite) if basis_cite else ''}"

    # Dispersion citation
    disp_clause_tex = ""
    disp_clause_md = ""
    if config.dispersion_correction:
        disp_cite = get_cite(config.dispersion_correction)
        disp_clause_tex = f" London dispersion interactions were accounted for using the {config.dispersion_correction} correction{(' ' + disp_cite) if disp_cite else ''}."
        disp_clause_md = f" London dispersion interactions were accounted for using the {config.dispersion_correction} correction."

    # Solvation clause
    solv_clause_tex = ""
    solv_clause_md = ""
    if config.solvation_model and config.solvent:
        solv_cite = get_cite(config.solvation_model)
        solv_clause_tex = f" Implicit continuum solvation effects in {config.solvent} were modeled using the {config.solvation_model} model{(' ' + solv_cite) if solv_cite else ''}."
        solv_clause_md = f" Implicit continuum solvation effects in {config.solvent} were modeled using the {config.solvation_model} model."
        if config.include_standard_state_correction:
            solv_clause_tex += (
                " Solvation free energies include a standard-state volume compression correction of "
                r"$\Delta G^\circ_{1\,\text{atm} \rightarrow 1\,\text{M}} = +1.894\text{ kcal/mol}$ "
                r"($+7.926\text{ kJ/mol}$) at $T = " + f"{config.temperature_kelvin:.2f}" + r"\text{ K}$."
            )
            solv_clause_md += (
                f" Solvation free energies include a standard-state volume compression correction of "
                f"Delta G°(1 atm -> 1 M) = +1.894 kcal/mol (+7.926 kJ/mol) at T = {config.temperature_kelvin:.2f} K."
            )

    # Thermochemistry / Quasi-Harmonic
    thermo_clause_tex = ""
    thermo_clause_md = ""
    if config.quasi_harmonic_damping:
        qh_cite = get_cite("quasi-harmonic")
        thermo_clause_tex = (
            f" Thermal corrections to Enthalpy ($H_{{298}}$) and Gibbs free energy ($G_{{298}}$) were evaluated at "
            f"{config.temperature_kelvin:.2f} K and {config.pressure_atm:.1f} atm within the harmonic approximation. "
            f"Low-frequency vibrational modes were damped using the modified quasi-harmonic approximation of Grimme"
            f"{(' ' + qh_cite) if qh_cite else ''} with a frequency cutoff of {config.quasi_harmonic_cutoff_cm1:.1f} cm$^{{-1}}$."
        )
        thermo_clause_md = (
            f" Thermal corrections to Enthalpy (H_298) and Gibbs free energy (G_298) were evaluated at "
            f"{config.temperature_kelvin:.2f} K and {config.pressure_atm:.1f} atm within the harmonic approximation. "
            f"Low-frequency vibrational modes were damped using the modified quasi-harmonic approximation of Grimme "
            f"with a frequency cutoff of {config.quasi_harmonic_cutoff_cm1:.1f} cm^-1."
        )
    else:
        thermo_clause_tex = (
            f" Thermal corrections to Enthalpy ($H_{{298}}$) and Gibbs free energy ($G_{{298}}$) were evaluated at "
            f"{config.temperature_kelvin:.2f} K and {config.pressure_atm:.1f} atm under the standard rigid-rotor harmonic-oscillator (RRHO) model."
        )
        thermo_clause_md = (
            f" Thermal corrections to Enthalpy (H_298) and Gibbs free energy (G_298) were evaluated at "
            f"{config.temperature_kelvin:.2f} K and {config.pressure_atm:.1f} atm under the standard rigid-rotor harmonic-oscillator (RRHO) model."
        )

    # Auxiliary basis fitting
    aux_tex = f" Resolution-of-identity Coulomb fitting was accelerated using the {config.auxiliary_basis} auxiliary basis set." if config.auxiliary_basis else ""
    aux_md = f" Resolution-of-identity Coulomb fitting was accelerated using the {config.auxiliary_basis} auxiliary basis set." if config.auxiliary_basis else ""

    # Assemble Full Paragraphs
    latex_text = (
        f"All quantum chemical electronic structure calculations were performed using the {sw_str_tex} package. "
        f"Geometry optimizations and harmonic vibrational frequency analyses were executed using the {func_tex} "
        f"density functional paired with the {basis_tex} orbital basis set.{aux_tex}{disp_clause_tex} "
        f"Numerical integration was conducted on the {config.grid_level} quadrature grid with tightened convergence criteria (TightOPT). "
        f"Stationary points were verified as true local minima (zero imaginary frequencies, $N_{{\\text{{imag}}}} = 0$) "
        f"or first-order saddle points (exactly one imaginary frequency, $N_{{\\text{{imag}}}} = 1$) via full Hessian diagonalizations.{solv_clause_tex}{thermo_clause_tex}"
    )

    markdown_text = (
        f"All quantum chemical electronic structure calculations were performed using the {sw_str_md} package. "
        f"Geometry optimizations and harmonic vibrational frequency analyses were executed using the {config.functional} "
        f"density functional paired with the {config.basis_set} orbital basis set.{aux_md}{disp_clause_md} "
        f"Numerical integration was conducted on the {config.grid_level} quadrature grid with tightened convergence criteria (TightOPT). "
        f"Stationary points were verified as true local minima (zero imaginary frequencies, N_imag = 0) "
        f"or first-order saddle points (exactly one imaginary frequency, N_imag = 1) via full Hessian diagonalizations.{solv_clause_md}{thermo_clause_md}"
    )

    return SynthesizedMethodsParagraph(
        markdown_text=markdown_text,
        latex_text=latex_text,
        referenced_bibtex_keys=cited_keys,
    )
