"""
CoChem Standard-State Solvation and Concentration Correction Engine (REQ-QC-084).

Provenance:
- Method Matrix [M]: Thermodynamic standard-state volume compression and concentration shift conversions.
- Derived Relations [D]: CODATA 2022 physical constants (R = 0.082057366081 L atm / (mol K), 1 Hartree = 627.5094740631 kcal/mol).
- Empirical [E]: Neat solvent densities and molarities (e.g., neat water = 55.34 M).
"""

from __future__ import annotations

import math
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.models import StandardStateCorrectionPayload

# CODATA 2022 Physical Constants
R_GAS_L_ATM_MOL_K: float = 0.082057366081
R_GAS_KCAL_MOL_K: float = 1.98720425864083e-3
R_GAS_KJ_MOL_K: float = 8.31446261815324e-3
HARTREE_TO_KCAL_MOL: float = 627.5094740631
HARTREE_TO_KJ_MOL: float = 2625.499639479
KCAL_TO_KJ: float = 4.184


class SolvationResult(BaseModel):
    """Data contract representing partitioned continuum solvation free energies."""
    model_config = ConfigDict(frozen=True)

    delta_g_solv_kcal_mol: float = Field(description="Total solvation free energy Delta G_solv in kcal/mol")
    delta_g_electrostatic_kcal_mol: float = Field(description="Electrostatic polarization free energy Delta G_el in kcal/mol")
    delta_g_cds_kcal_mol: float = Field(description="Cavitation, dispersion, and solvent reorganization term in kcal/mol")
    solvent_name: str = Field(description="Solvent identifier")
    dielectric_constant: Optional[float] = Field(default=None, description="Static dielectric constant")


class StandardStateCorrection(BaseModel):
    """Data contract representing standard state compression correction."""
    model_config = ConfigDict(frozen=True)

    temperature_k: float = Field(default=298.15)
    pressure_atm: float = Field(default=1.0)
    molar_volume_liters: float
    delta_g_shift_kcal_mol: float
    delta_g_shift_kj_mol: float
    g_solv_hartree: float


def compute_standard_state_correction(
    g_gas_1atm_hartree: float,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    solvent_name: str = "water",
    is_neat_solvent: bool = False,
    neat_concentration_m: float = 55.34,
) -> StandardStateCorrectionPayload:
    """
    [M][D] Compute rigorous standard-state volume and concentration correction (REQ-QC-084).

    Converts gas-phase Gibbs free energy at 1 atm standard state to 1.0 M solution standard state
    (or neat liquid standard state).

    Exact Thermodynamic Relations:
        - Gas molar volume V_m^0 = R * T / P (L/mol)
        - Ideal gas standard concentration c_gas^0 = 1 / V_m^0 (mol/L)
        - Solution standard state c_solv^0 = neat_concentration_m if is_neat_solvent else 1.0 M
        - Delta G^0 (1 atm -> c_solv) = R * T * ln(c_solv^0 / c_gas^0) = R * T * ln(c_solv^0 * V_m^0)
        - At 298.15 K, 1.0 atm -> 1.0 M shift is +1.8943 kcal/mol (+7.9258 kJ/mol).
        - G_solv = G_gas + Delta G^0 (converted to Hartree).

    Args:
        g_gas_1atm_hartree: Gas phase Gibbs free energy at 1 atm in Hartree.
        temperature_k: Thermodynamic temperature T in Kelvin (default: 298.15 K).
        pressure_atm: Gas reference pressure P in atm (default: 1.0 atm).
        solvent_name: Name or descriptor of the solvent.
        is_neat_solvent: Whether to apply neat liquid solvent molarity correction.
        neat_concentration_m: Molar concentration of neat solvent in mol/L (default: 55.34 M for H2O).

    Returns:
        StandardStateCorrectionPayload containing exact thermodynamic shifts and converted energies.

    Raises:
        ValueError: If temperature_k <= 0 or pressure_atm <= 0.
    """
    if temperature_k <= 0.0:
        raise ValueError(f"Temperature must be positive, got T={temperature_k} K.")
    if pressure_atm <= 0.0:
        raise ValueError(f"Pressure must be positive, got P={pressure_atm} atm.")

    # 1. Ideal gas molar volume V_m^0 = RT / P
    v_molar = (R_GAS_L_ATM_MOL_K * temperature_k) / pressure_atm
    c_gas = 1.0 / v_molar

    # 2. Target solution concentration
    c_solution = neat_concentration_m if is_neat_solvent else 1.0
    if c_solution <= 0.0:
        raise ValueError(f"Solution concentration must be positive, got {c_solution} M.")

    # 3. Thermodynamic free energy shift Delta G^0 = R * T * ln(c_solution / c_gas)
    ratio = c_solution * v_molar
    delta_g_kcal = R_GAS_KCAL_MOL_K * temperature_k * math.log(ratio)
    delta_g_kj = R_GAS_KJ_MOL_K * temperature_k * math.log(ratio)

    # 4. Convert Delta G to Hartree
    delta_g_hartree = delta_g_kcal / HARTREE_TO_KCAL_MOL
    g_solv_hartree = g_gas_1atm_hartree + delta_g_hartree

    return StandardStateCorrectionPayload(
        temperature_kelvin=temperature_k,
        pressure_atm=pressure_atm,
        gas_molar_volume_liters=round(v_molar, 4),
        gas_standard_concentration_m=round(c_gas, 6),
        solution_standard_concentration_m=c_solution,
        delta_g_1atm_to_1m_kcal_mol=round(delta_g_kcal, 4),
        delta_g_1atm_to_1m_kj_mol=round(delta_g_kj, 4),
        g_gas_1atm_hartree=g_gas_1atm_hartree,
        g_solv_1m_hartree=g_solv_hartree,
        solvent_name=solvent_name,
        is_neat_solvent_corrected=is_neat_solvent,
    )


def partition_solvation_free_energy(
    delta_g_solv_kcal_mol: float,
    delta_g_electrostatic_kcal_mol: float,
    delta_g_cds_kcal_mol: Optional[float] = None,
    solvent_name: str = "water",
    dielectric_constant: Optional[float] = None,
) -> SolvationResult:
    """
    [M][D] Partition continuum solvation free energy into electrostatic and non-electrostatic (CDS) contributions.
    """
    if delta_g_cds_kcal_mol is None:
        cds = delta_g_solv_kcal_mol - delta_g_electrostatic_kcal_mol
    else:
        cds = delta_g_cds_kcal_mol

    return SolvationResult(
        delta_g_solv_kcal_mol=round(delta_g_solv_kcal_mol, 4),
        delta_g_electrostatic_kcal_mol=round(delta_g_electrostatic_kcal_mol, 4),
        delta_g_cds_kcal_mol=round(cds, 4),
        solvent_name=solvent_name,
        dielectric_constant=dielectric_constant,
    )
