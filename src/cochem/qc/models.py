"""
CoChem Quantum Chemistry Pydantic v2 Core Data Models.

Provenance:
- Method Matrix [M]: Quantum spin-parity invariants, Poincaré-Hopf topological index,
  Sayvetz-Eckart orthogonal projection, and Marcus electron transfer conditions.
- Derived Relations [D]: IUPAC CIAAW isotopic data structures, CODATA 2022 constants,
  and Level-of-Theory canonical hashing.
- Empirical [E]: Gradient convergence norms (1.0e-4 a.u.) and Kabsch RMSD alignment thresholds (0.25 Å).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class SpinChargeValidationResult(BaseModel):
    """
    [M] Pre-flight spin-parity and charge validation payload (REQ-QC-081).

    Enforces spin-parity invariants:
    - Total electron count N_e = Z_total - q >= 1.
    - Spin parity: (N_e + M) mod 2 == 1 where M = 2S + 1.
    - Multiplicity ceiling: M - 1 <= N_e.
    """

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(description="Whether the molecular configuration passes spin-parity gates")
    total_nuclear_charge: int = Field(ge=1, description="Total nuclear charge Z_total = sum(Z_i)")
    total_net_charge: int = Field(description="Net molecular charge q")
    total_electrons: int = Field(ge=1, description="Total electron count N_e = Z_total - q")
    multiplicity: int = Field(ge=1, description="Spin multiplicity M = 2S + 1")
    parity_valid: bool = Field(description="Whether (N_e + M) mod 2 == 1")
    max_unpaired_electrons: int = Field(default=0, ge=0, description="Upper bound on unpaired electrons (M - 1)")
    formal_oxidation_states: Dict[int, float] = Field(
        default_factory=dict, description="Estimated formal oxidation states per atom index"
    )
    unphysical_valence_warnings: List[str] = Field(
        default_factory=list, description="Diagnostic warnings for unphysical valence or oxidation states"
    )


class MarcusReorganizationResult(BaseModel):
    """
    [M][D] Nelson 4-point Marcus electron-transfer reorganization energy payload (REQ-QC-082).
    """

    model_config = ConfigDict(frozen=True)

    e_neutral_at_neutral_geom_hartree: float = Field(description="E_0(R_0) in Hartree")
    e_charged_at_neutral_geom_hartree: float = Field(description="E_pm(R_0) in Hartree")
    e_charged_at_charged_geom_hartree: float = Field(description="E_pm(R_pm) in Hartree")
    e_neutral_at_charged_geom_hartree: float = Field(description="E_0(R_pm) in Hartree")
    lambda_1_ev: float = Field(ge=0.0, description="E_pm(R_0) - E_pm(R_pm) in eV")
    lambda_2_ev: float = Field(ge=0.0, description="E_0(R_pm) - E_0(R_0) in eV")
    lambda_inner_ev: float = Field(ge=0.0, description="Inner-sphere reorganization energy lambda_1 + lambda_2 in eV")
    lambda_inner_kcal_mol: float = Field(ge=0.0, description="Inner-sphere reorganization energy in kcal/mol")
    lambda_outer_ev: Optional[float] = Field(default=None, ge=0.0, description="Continuum dielectric outer-sphere reorganization in eV")
    total_reorganization_energy_ev: float = Field(ge=0.0, description="Total reorganization energy lambda_total in eV")
    gibbs_free_energy_delta_ev: float = Field(description="Driving force delta G^0 in eV")
    marcus_activation_barrier_ev: float = Field(ge=0.0, description="Classical Marcus activation barrier Delta G^dagger in eV")
    electron_transfer_rate_s_inv: Optional[float] = Field(default=None, ge=0.0, description="Electron transfer rate constant k_ET in s^-1")


class IRCTrajectoryResult(BaseModel):
    """
    [M][E] Bidirectional Intrinsic Reaction Coordinate (IRC) trajectory payload (REQ-QC-083).
    """

    model_config = ConfigDict(frozen=True)

    ts_energy_hartree: float = Field(description="Transition state energy in Hartree")
    ts_imaginary_frequency_cm1: float = Field(lt=0.0, description="Unique imaginary transition mode frequency in cm^-1")
    forward_steps: int = Field(ge=1, description="Number of forward IRC integration steps")
    reverse_steps: int = Field(ge=1, description="Number of reverse IRC integration steps")
    forward_endpoint_energy_hartree: float = Field(description="Energy of optimized forward minimum in Hartree")
    reverse_endpoint_energy_hartree: float = Field(description="Energy of optimized reverse minimum in Hartree")
    forward_endpoint_rmsd_to_reactant_angstrom: Optional[float] = Field(
        default=None, ge=0.0, description="Kabsch RMSD from forward endpoint to reactant minimum in Å"
    )
    reverse_endpoint_rmsd_to_product_angstrom: Optional[float] = Field(
        default=None, ge=0.0, description="Kabsch RMSD from reverse endpoint to product minimum in Å"
    )
    connectivity_verified: bool = Field(description="Whether endpoints match expected reactant/product within 0.25 Å")
    path_coordinates_bohr: List[List[List[float]]] = Field(description="Trajectory coordinates per step [step, atom, xyz] in Bohr")
    path_energies_hartree: List[float] = Field(description="Energy profile along reaction coordinate in Hartree")
    hdf5_trajectory_path: Optional[str] = Field(default=None, description="Path to HDF5 file storing the full trajectory")



class StandardStateCorrectionPayload(BaseModel):
    """
    [D] Standard-state solvation volume and concentration correction payload (REQ-QC-084).
    """

    model_config = ConfigDict(frozen=True)

    temperature_kelvin: float = Field(default=298.15, gt=0.0, description="Thermodynamic temperature T in Kelvin")
    pressure_atm: float = Field(default=1.0, gt=0.0, description="Gas reference pressure P in atm")
    gas_molar_volume_liters: float = Field(gt=0.0, description="Ideal gas molar volume V_m^0 = RT/P in L/mol")
    gas_standard_concentration_m: float = Field(gt=0.0, description="Gas concentration 1/V_m^0 in mol/L (M)")
    solution_standard_concentration_m: float = Field(default=1.0, gt=0.0, description="Solution standard state concentration in M")
    delta_g_1atm_to_1m_kcal_mol: float = Field(description="Free energy compression penalty RT*ln(V_m^0) in kcal/mol")
    delta_g_1atm_to_1m_kj_mol: float = Field(description="Free energy compression penalty in kJ/mol")
    g_gas_1atm_hartree: float = Field(description="Gas phase free energy at 1 atm in Hartree")
    g_solv_1m_hartree: float = Field(description="Solution free energy at 1 M standard state in Hartree")
    solvent_name: str = Field(description="Solvent descriptor or chemical name")
    is_neat_solvent_corrected: bool = Field(default=False, description="Whether neat solvent molarity correction (e.g. 55.34 M water) is applied")


class LevelOfTheoryEnvelope(BaseModel):
    """
    [D] Cryptographic Level-of-Theory specification and provenance envelope (REQ-QC-085).
    """

    model_config = ConfigDict(frozen=True)

    hamiltonian_method: str = Field(description="Electronic structure method (e.g., 'wB97X-D4', 'B3LYP')")
    basis_set: str = Field(description="Primary orbital basis set (e.g., 'def2-TZVP')")
    auxiliary_basis_set: Optional[str] = Field(default=None, description="Auxiliary/Coulomb fitting basis set (e.g., 'def2/J')")
    dispersion_correction: Optional[str] = Field(default=None, description="Dispersion correction scheme (e.g., 'D3BJ', 'D4')")
    integration_grid: str = Field(description="Numerical integration grid (e.g., 'DefGrid3', 'UltraFine')")
    solvation_model: Optional[str] = Field(default=None, description="Implicit solvation model (e.g., 'CPCM(water)', 'SMD(benzene)')")
    max_gradient_residual_au: float = Field(ge=0.0, description="Maximum Cartesian gradient component residual in a.u.")


class IsotopeSubstitutionPayload(BaseModel):
    """
    [D] Dynamic Mendeleev isotopic substitution and Bigeleisen-Mayer KIE payload (REQ-QC-086).
    """

    model_config = ConfigDict(frozen=True)

    substituted_atoms: Dict[int, int] = Field(description="Mapping of 0-based atom indices to target mass numbers")
    mendeleev_masses_amu: Dict[int, float] = Field(description="Dynamic CIAAW isotopic masses retrieved from mendeleev in amu")
    reference_zpe_kcal_mol: float = Field(ge=0.0, description="Zero-point vibrational energy of parent molecule in kcal/mol")
    substituted_zpe_kcal_mol: float = Field(ge=0.0, description="Zero-point vibrational energy of isotopologue in kcal/mol")
    reference_frequencies_cm1: List[float] = Field(description="Harmonic vibrational frequencies of parent molecule in cm^-1")
    substituted_frequencies_cm1: List[float] = Field(description="Harmonic vibrational frequencies of isotopologue in cm^-1")
    kie_bigeleisen_mayer: Optional[float] = Field(default=None, ge=0.0, description="Total Bigeleisen-Mayer Kinetic Isotope Effect")
    mmi_factor: Optional[float] = Field(default=None, ge=0.0, description="Mass-Moment of Inertia (MMI) contribution to KIE")
    exc_factor: Optional[float] = Field(default=None, ge=0.0, description="Vibrational excitation (EXC) contribution to KIE")
    zpe_factor: Optional[float] = Field(default=None, ge=0.0, description="Zero-point energy (ZPE) exponential contribution to KIE")


class WavefunctionStabilityReport(BaseModel):
    """
    [M] Post-SCF orbital Hessian stability analysis and Yamaguchi projection report (REQ-QC-087).
    """

    model_config = ConfigDict(frozen=True)

    initial_stability: Literal["STABLE", "UNSTABLE_INTERNAL_RHF_UHF", "UNSTABLE_REAL_COMPLEX", "UNSTABLE_EXTERNAL"] = Field(
        description="Wavefunction stability classification from orbital Hessian diagonalization"
    )
    lowest_orbital_hessian_eigenvalue: float = Field(description="Lowest electronic orbital Hessian eigenvalue lambda_orb")
    is_broken_symmetry_required: bool = Field(description="Whether a broken-symmetry UHF/U-DFT treatment is physically required")
    s2_expected: float = Field(ge=0.0, description="Expected <S^2> value for pure spin state S(S+1)")
    s2_calculated: float = Field(ge=0.0, description="Calculated post-SCF <S^2> value")
    spin_contamination_percentage: float = Field(ge=0.0, description="Spin contamination metric percentage")
    yamaguchi_corrected_energy_hartree: Optional[float] = Field(
        default=None, description="Yamaguchi spin-projected energy for broken-symmetry singlet in Hartree"
    )


class BondCriticalPointDescriptor(BaseModel):
    """
    [M] QTAIM Bond Critical Point (3,-1) scalar field and topological descriptor (REQ-QC-088).
    """

    model_config = ConfigDict(frozen=True)

    bcp_index: int = Field(ge=0, description="Unique index of the Bond Critical Point")
    atom_1_index: int = Field(ge=0, description="0-based index of first bonded atom")
    atom_2_index: int = Field(ge=0, description="0-based index of second bonded atom")
    coordinates_angstrom: Tuple[float, float, float] = Field(description="Cartesian coordinates (x, y, z) of the BCP in Å")
    electron_density_au: float = Field(gt=0.0, description="Electron density rho(r_BCP) in a.u. (e/Bohr^3)")
    laplacian_au: float = Field(description="Laplacian of electron density nabla^2 rho in a.u.")
    kinetic_energy_density_g_au: float = Field(ge=0.0, description="Lagrangian kinetic energy density G(r) in a.u.")
    potential_energy_density_v_au: float = Field(le=0.0, description="Virial potential energy density V(r) in a.u.")
    total_energy_density_h_au: float = Field(description="Total electronic energy density H(r) = G(r) + V(r) in a.u.")
    ellipticity: float = Field(ge=0.0, description="Bond ellipticity epsilon = (lambda_1 / lambda_2) - 1")
    interaction_classification: Literal[
        "COVALENT_SHARED", "TRANSIT_POLAR_COVALENT", "NON_COVALENT_CLOSED_SHELL", "HYDROGEN_BOND", "METALLIC"
    ] = Field(description="Bader/Cremer-Kraka interaction classification")


class SOCMEResult(BaseModel):
    """
    [M][D] Spin-Orbit Coupling Matrix Element (SOMF/SOCME) and photophysics kinetics payload (REQ-QC-089).
    """

    model_config = ConfigDict(frozen=True)

    initial_state: str = Field(description="Initial electronic state label (e.g., 'S1')")
    final_state: str = Field(description="Final electronic state label (e.g., 'T1')")
    delta_e_singlet_triplet_ev: float = Field(description="Adiabatic or vertical state energy gap Delta E_ST in eV")
    socme_magnitude_cm1: float = Field(ge=0.0, description="Effective SOMF spin-orbit coupling matrix element magnitude in cm^-1")
    socme_components_cm1: Tuple[float, float, float] = Field(description="Cartesian components (H_SO,x, H_SO,y, H_SO,z) in cm^-1")
    fcwd_density_states: float = Field(ge=0.0, description="Franck-Condon weighted density of states rho_FCWD in eV^-1")
    k_isc_s_inv: float = Field(ge=0.0, description="Intersystem crossing rate constant k_ISC via Fermi Golden Rule in s^-1")
    phosphorescence_lifetime_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Predicted radiative phosphorescence lifetime tau_phos in seconds"
    )


class ProjectedFrequencyResult(BaseModel):
    """
    [M] Sayvetz-Eckart rotational/translational projected vibrational frequency payload (REQ-QC-090).
    """

    model_config = ConfigDict(frozen=True)

    unprojected_frequencies_cm1: List[float] = Field(description="Raw Cartesian Hessian harmonic frequencies in cm^-1")
    projected_frequencies_cm1: List[float] = Field(
        description="Vibrational frequencies after Sayvetz-Eckart P * H_MW * P projection in cm^-1"
    )
    rot_trans_residual_max_cm1: float = Field(
        ge=0.0, description="Maximum absolute frequency residual of projected 6 rigid-body zero modes in cm^-1"
    )
    num_projected_zero_modes: int = Field(default=6, ge=0, description="Number of projected translational/rotational zero modes")
    is_linear_molecule: bool = Field(default=False, description="Whether the molecular geometry is linear (5 zero modes)")
    is_projection_clean: bool = Field(
        description="Whether all rigid-body modes have residual frequencies <= 1.0e-4 cm^-1"
    )

