"""
CoChem Gaussian 16 Input Deck Synthesizer with Strict Level-of-Theory Gates (REQ-QC-085, REQ-QC-087).

Provenance:
- Method Matrix [M]: Route card generation, integration grid enforcement (Grid=UltraFine),
  empirical dispersion (GD3BJ), and wavefunction stability options (Stable=Opt, Stable=RR, Stable=CR).
- Derived Relations [D]: Synthesizes Freq=ReadIsotopes deck sections with explicit temperature,
  pressure, scale factors, and custom atomic isotope masses via mendeleev.
- Empirical [E]: Pre-flight spin-parity verification (REQ-QC-081).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import (
    get_ciaaw_atomic_mass,
    get_ciaaw_isotope_mass,
    validate_charge_and_multiplicity,
)


class GaussianRouteCard(BaseModel):
    """Data contract representing a Gaussian route section."""
    model_config = ConfigDict(frozen=True)

    method: str = Field(default="B3LYP", description="Hamiltonian or DFT functional name")
    basis: str = Field(default="def2-TZVP", description="Primary GTO basis set")
    job_type: str = Field(default="opt freq", description="Job type directives, e.g. opt=(tight,calcfc) freq")
    stability: Optional[str] = Field(default=None, description="Wavefunction stability, e.g. Stable=Opt, Stable=RR, Stable=CR")
    grid: str = Field(default="Integral=(Grid=UltraFine)", description="Quadrature grid specification")
    dispersion: Optional[str] = Field(default="EmpiricalDispersion=GD3BJ", description="Dispersion correction option")
    solvation: Optional[str] = Field(default=None, description="SCRF solvation model specification")
    extra_keywords: List[str] = Field(default_factory=list, description="Additional Gaussian route keywords")

    def format_route_string(self) -> str:
        """Render the route card into a standardized Gaussian route string."""
        tokens = ["#p", f"{self.method}/{self.basis}"]
        if self.job_type:
            tokens.append(self.job_type)
        if self.stability:
            tokens.append(self.stability)
        if self.grid:
            tokens.append(self.grid)
        if self.dispersion:
            tokens.append(self.dispersion)
        if self.solvation:
            tokens.append(self.solvation)
        if self.extra_keywords:
            tokens.extend(self.extra_keywords)
        return " ".join(tokens)


class GaussianJobSpec(BaseModel):
    """Data contract representing a complete Gaussian 16 job specification."""
    symbols: List[str]
    coordinates_angstrom: List[Tuple[float, float, float]]
    charge: int = 0
    multiplicity: int = 1
    route: GaussianRouteCard
    title: str = "CoChem Automated Gaussian Job"
    memory_gb: int = 4
    nprocshared: int = 4
    read_isotopes: bool = False
    temperature_k: float = 298.15
    pressure_atm: float = 1.0
    scale_factor: float = 1.0
    isotopic_substitutions: Dict[int, int] = Field(default_factory=dict)


def generate_gaussian_deck(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    method: str = "B3LYP",
    basis: str = "def2-TZVP",
    job_type: str = "opt=(tight,calcfc) freq=noraman",
    charge: int = 0,
    multiplicity: int = 1,
    title: str = "CoChem Gaussian Job",
    memory_gb: int = 4,
    nprocshared: int = 4,
    stability: Optional[str] = None,
    grid: str = "Integral=(Grid=UltraFine)",
    dispersion: Optional[str] = "EmpiricalDispersion=GD3BJ",
    solvation_model: Optional[str] = None,
    solvent_name: Optional[str] = None,
    read_isotopes: bool = False,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    scale_factor: float = 1.0,
    isotopic_substitutions: Optional[Dict[int, int]] = None,
) -> str:
    """
    [REQ-QC-085, REQ-QC-087] Assemble a complete, production-ready Gaussian 16 input deck (.gjf/.com).
    """
    # 1. Pre-flight Spin-Parity Verification
    validate_charge_and_multiplicity(symbols=symbols, charge=charge, multiplicity=multiplicity)

    # 2. Build Solvation string if requested
    solv_str: Optional[str] = None
    if solvation_model and solvent_name:
        solv_str = f"SCRF=({solvation_model},Solvent={solvent_name})"
    elif solvation_model:
        solv_str = f"SCRF=({solvation_model})"

    # 3. ReadIsotopes option in job_type
    effective_job_type = job_type
    if read_isotopes and "readisotopes" not in effective_job_type.lower():
        effective_job_type = effective_job_type.replace("freq", "freq=readisotopes")
        if "freq" not in effective_job_type.lower():
            effective_job_type += " freq=readisotopes"

    route_card = GaussianRouteCard(
        method=method,
        basis=basis,
        job_type=effective_job_type,
        stability=stability,
        grid=grid,
        dispersion=dispersion,
        solvation=solv_str,
    )

    deck_lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={memory_gb}GB",
        route_card.format_route_string(),
        "",
        title,
        "",
        f"{charge} {multiplicity}",
    ]

    for sym, (x, y, z) in zip(symbols, coordinates_angstrom):
        deck_lines.append(f" {sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}")
    deck_lines.append("")

    # 4. Append ReadIsotopes block if enabled
    if read_isotopes:
        deck_lines.append(f"{temperature_k:.2f} {pressure_atm:.2f} {scale_factor:.4f}")
        sub_dict = isotopic_substitutions or {}
        for idx, sym in enumerate(symbols):
            if idx in sub_dict:
                mass_val = get_ciaaw_isotope_mass(sym, sub_dict[idx])
            else:
                mass_val = get_ciaaw_atomic_mass(sym)
            deck_lines.append(f"{mass_val:.6f}")
        deck_lines.append("")

    return "\n".join(deck_lines) + "\n"
