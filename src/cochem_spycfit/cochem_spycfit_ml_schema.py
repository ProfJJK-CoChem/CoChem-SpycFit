"""
CoChem-SpycFit: ML-Augmented Assignment Schema & Data Models.

This module defines:
1. Pydantic v2 strict schemas for multi-tier resource limits.
2. Dynamic isotope mass retrieval and molecular mass computation via the Mendeleev library.
3. Cryptographic SHA-256 provenance ledger computation.
4. Spectroscopic transition quantum models (asymmetric top quantum numbers, dipole components, tags).
5. Directed Acyclic Graph (DAG) FitState models with deterministic SHA-256 state hashing.
6. Parity status, parity warning structures, and hardware-aware scan chunk models.

Zero-Mock Mandate: Authentic Mendeleev isotopic retrieval, zero stubs, zero pass statements.
"""

from __future__ import annotations

import datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import mendeleev
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class MLSchemaError(Exception):
    """Base exception for all ML schema and data modeling errors."""


class IsotopeRetrievalError(MLSchemaError):
    """Raised when dynamic isotope or atomic mass retrieval fails."""


class ProvenanceError(MLSchemaError):
    """Raised when cryptographic provenance verification or serialization fails."""


class QuantumNumberError(MLSchemaError):
    """Raised when spectroscopic quantum numbers violate physical selection rules."""


# =============================================================================
# 2. MULTI-TIER RESOURCE LIMITS
# =============================================================================

class MultiTierResourceLimits(BaseModel):
    """Resource configuration limits tailored to execution tiers (GPU, TPU, CPU, HPC)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tier_name: str = Field(
        default="CPU-Standard",
        description="Execution tier identifier (e.g., 'GPU-Accelerated', 'CPU-Standard', 'Slurm-HPC').",
    )
    max_ram_gb: float = Field(
        default=16.0,
        gt=0.0,
        description="Maximum allocated RAM in gigabytes.",
    )
    max_vram_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum allocated VRAM in gigabytes (0.0 for CPU).",
    )
    cpu_threads: int = Field(
        default=4,
        ge=1,
        description="Number of CPU worker threads allocated for execution.",
    )
    max_chunk_bandwidth_mhz: float = Field(
        default=250.0,
        gt=0.0,
        description="Maximum instantaneous FT-MW spectrometer scan chunk bandwidth in MHz.",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for vectorization and ML inference.",
    )
    fp64_enforced: bool = Field(
        default=True,
        description="Whether JAX float64 double-precision lock is strictly enforced.",
    )


# =============================================================================
# 3. DYNAMIC MENDELEEV ISOTOPE & MOLECULAR MASS RETRIEVAL
# =============================================================================

def get_dynamic_isotope_mass(symbol: str, mass_number: Optional[int] = None) -> float:
    """
    Dynamically retrieve the high-precision atomic or isotopic mass from Mendeleev.

    Args:
        symbol: Chemical element symbol (e.g., 'H', 'C', 'O', 'S', 'Cl').
        mass_number: Optional integer isotope mass number (e.g., 12, 13, 16, 18, 32).

    Returns:
        Exact isotopic or standard atomic mass in unified atomic mass units (u).

    Raises:
        IsotopeRetrievalError: If the element or requested isotope is not found.
    """
    clean_symbol = symbol.strip()
    try:
        element_obj = mendeleev.element(clean_symbol)
    except Exception as exc:
        raise IsotopeRetrievalError(
            f"Failed to retrieve Mendeleev element record for symbol '{clean_symbol}': {exc}"
        ) from exc

    if mass_number is not None:
        matching_isotopes = [
            iso for iso in element_obj.isotopes if iso.mass_number == mass_number
        ]
        if not matching_isotopes:
            raise IsotopeRetrievalError(
                f"No isotope record found for '{clean_symbol}' with mass number {mass_number}."
            )
        target_iso = matching_isotopes[0]
        if target_iso.mass is None:
            raise IsotopeRetrievalError(
                f"Isotope {clean_symbol}-{mass_number} has no defined mass in Mendeleev database."
            )
        return float(target_iso.mass)

    abundant_isotopes = [
        iso for iso in element_obj.isotopes
        if iso.abundance is not None and iso.mass is not None
    ]
    if abundant_isotopes:
        most_abundant = max(abundant_isotopes, key=lambda iso: float(iso.abundance or 0.0))
        return float(most_abundant.mass)

    if element_obj.atomic_weight is not None:
        return float(element_obj.atomic_weight)

    if element_obj.isotopes:
        valid_mass_isotopes = [iso for iso in element_obj.isotopes if iso.mass is not None]
        if valid_mass_isotopes:
            return float(valid_mass_isotopes[0].mass)

    raise IsotopeRetrievalError(
        f"Unable to resolve any valid atomic or isotopic mass for element '{clean_symbol}'."
    )


def compute_dynamic_molecular_mass(
    formula_or_composition: Union[Dict[str, int], str]
) -> float:
    """
    Compute total molecular mass by dynamically summing Mendeleev isotopic masses.

    Args:
        formula_or_composition: Either a dictionary mapping isotope/element tokens to counts
            (e.g., {'13C': 1, '1H': 4} or {'C': 1, 'H': 4, 'O': 1}) or a chemical formula
            string (e.g., 'H2O', 'CH4', '13C1H4', 'SO2').

    Returns:
        Total molecular mass in unified atomic mass units (u / Da).

    Raises:
        IsotopeRetrievalError: If parsing fails or an invalid element symbol is supplied.
    """
    if isinstance(formula_or_composition, dict):
        total_mass = 0.0
        for token, count in formula_or_composition.items():
            if count <= 0:
                continue
            token_match = re.match(r"^(\d+)?([A-Z][a-z]?)$", token.strip())
            if not token_match:
                raise IsotopeRetrievalError(
                    f"Invalid chemical composition token '{token}' in dict."
                )
            mass_num_str, sym = token_match.groups()
            mass_num = int(mass_num_str) if mass_num_str else None
            element_mass = get_dynamic_isotope_mass(sym, mass_num)
            total_mass += count * element_mass
        return float(total_mass)

    formula_str = formula_or_composition.strip()
    tokens = re.findall(r"(\d*[A-Z][a-z]?)(\d*)", formula_str)
    if not tokens:
        raise IsotopeRetrievalError(f"Unable to parse chemical formula '{formula_str}'.")

    total_mass = 0.0
    matched_chars = 0
    for element_token, count_str in tokens:
        if not element_token:
            continue
        matched_chars += len(element_token) + len(count_str)
        count = int(count_str) if count_str else 1
        iso_match = re.match(r"^(\d+)?([A-Z][a-z]?)$", element_token)
        if not iso_match:
            raise IsotopeRetrievalError(
                f"Invalid element/isotope substring '{element_token}' in formula '{formula_str}'."
            )
        mass_num_str, sym = iso_match.groups()
        mass_num = int(mass_num_str) if mass_num_str else None
        element_mass = get_dynamic_isotope_mass(sym, mass_num)
        total_mass += count * element_mass

    if matched_chars != len(formula_str):
        raise IsotopeRetrievalError(
            f"Formula '{formula_str}' contains unparsed residual characters."
        )

    return float(total_mass)


# =============================================================================
# 4. CRYPTOGRAPHIC PROVENANCE LEDGER
# =============================================================================

def compute_bytes_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hexadecimal digest for raw bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hexadecimal digest of a file on disk."""
    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ProvenanceError(f"Target file for SHA-256 calculation does not exist: {path_obj}")

    hasher = hashlib.sha256()
    with open(path_obj, "rb") as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_object_sha256(obj: Any) -> str:
    """Compute deterministic SHA-256 hexadecimal digest of an arbitrary serializable object."""
    canonical_json = json.dumps(
        obj,
        sort_keys=True,
        default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o),
    )
    return compute_bytes_sha256(canonical_json.encode("utf-8"))


class ProvenanceLedger(BaseModel):
    """Cryptographic provenance ledger capturing experimental and computational lineage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    git_commit_hash: str = Field(
        default="0000000000000000000000000000000000000000",
        description="CoChem repository Git commit SHA.",
    )
    csv_sha256: str = Field(
        default="",
        description="Cryptographic SHA-256 digest of the ingested experimental spectrum CSV.",
    )
    landscape_uuid: str = Field(
        default="",
        description="Unique identifier linking to the underlying HDF5 potential energy surface landscape.",
    )
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO 8601 timestamp of record creation.",
    )
    source_level_of_theory: str = Field(
        default="CCSD(T)/CBS",
        description="Underlying quantum mechanical level of theory.",
    )
    user_id: str = Field(
        default="cochem_operator",
        description="Identifier of the operator or automated service account.",
    )
    metadata_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary auxiliary provenance key-value pairs.",
    )

    def compute_ledger_hash(self) -> str:
        """Compute authoritative SHA-256 checksum across all provenance fields."""
        payload_dict = {
            "git_commit_hash": self.git_commit_hash,
            "csv_sha256": self.csv_sha256,
            "landscape_uuid": self.landscape_uuid,
            "timestamp_iso": self.timestamp_iso,
            "source_level_of_theory": self.source_level_of_theory,
            "user_id": self.user_id,
            "metadata_payload": self.metadata_payload,
        }
        return compute_object_sha256(payload_dict)


# =============================================================================
# 5. SPECTROSCOPIC TRANSITION MODELS
# =============================================================================

class DipoleType(str, Enum):
    """Transition dipole orientation component."""
    A_TYPE = "a"
    B_TYPE = "b"
    C_TYPE = "c"


class AssignmentTag(str, Enum):
    """Provenance and lifecycle status tag of a spectroscopic transition."""
    TORQ = "[TORQ]"
    ML = "[ML]"
    USER = "[USER]"
    SPFIT = "[SPFIT]"
    LOCKED = "[LOCKED]"
    IGNORED = "[IGNORED]"
    DELETED = "[DELETED]"


class DipoleComponents(BaseModel):
    """Permanent and transition dipole moment vector components (in Debye)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mu_a: float = Field(default=0.0, description="Dipole moment component along the a principal axis.")
    mu_b: float = Field(default=0.0, description="Dipole moment component along the b principal axis.")
    mu_c: float = Field(default=0.0, description="Dipole moment component along the c principal axis.")

    def magnitude(self) -> float:
        """Compute the total dipole magnitude in Debye."""
        return float(math.sqrt(self.mu_a**2 + self.mu_b**2 + self.mu_c**2))

    def dominant_type(self) -> DipoleType:
        """Determine the dominant dipole transition type based on magnitude."""
        components = {
            DipoleType.A_TYPE: abs(self.mu_a),
            DipoleType.B_TYPE: abs(self.mu_b),
            DipoleType.C_TYPE: abs(self.mu_c),
        }
        return max(components, key=lambda dt: components[dt])


class TransitionQuantumNumbers(BaseModel):
    """Asymmetric top rotational quantum numbers for upper and lower states."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    J_upper: int = Field(..., ge=0, description="Upper state total angular momentum quantum number J'.")
    Ka_upper: int = Field(..., ge=0, description="Upper state prolate projection quantum number Ka'.")
    Kc_upper: int = Field(..., ge=0, description="Upper state oblate projection quantum number Kc'.")
    J_lower: int = Field(..., ge=0, description="Lower state total angular momentum quantum number J''.")
    Ka_lower: int = Field(..., ge=0, description="Lower state prolate projection quantum number Ka''.")
    Kc_lower: int = Field(..., ge=0, description="Lower state oblate projection quantum number Kc''.")

    @model_validator(mode="after")
    def validate_quantum_constraints(self) -> TransitionQuantumNumbers:
        """Validate asymmetric top quantum number bounds and selection rules."""
        if self.Ka_upper > self.J_upper or self.Kc_upper > self.J_upper:
            raise QuantumNumberError(
                f"Upper state projections Ka={self.Ka_upper}, Kc={self.Kc_upper} exceed J={self.J_upper}."
            )
        if self.Ka_lower > self.J_lower or self.Kc_lower > self.J_lower:
            raise QuantumNumberError(
                f"Lower state projections Ka={self.Ka_lower}, Kc={self.Kc_lower} exceed J={self.J_lower}."
            )
        sum_upper = self.Ka_upper + self.Kc_upper
        if sum_upper not in (self.J_upper, self.J_upper + 1):
            raise QuantumNumberError(
                f"Upper state Ka+Kc ({sum_upper}) must equal J or J+1 for J={self.J_upper}."
            )
        sum_lower = self.Ka_lower + self.Kc_lower
        if sum_lower not in (self.J_lower, self.J_lower + 1):
            raise QuantumNumberError(
                f"Lower state Ka+Kc ({sum_lower}) must equal J or J+1 for J={self.J_lower}."
            )
        if abs(self.J_upper - self.J_lower) > 1:
            raise QuantumNumberError(
                f"Electric dipole transition selection rule breached: |delta J| = {abs(self.J_upper - self.J_lower)} > 1."
            )
        return self

    def delta_J(self) -> int:
        """Calculate transition Delta J = J' - J''."""
        return self.J_upper - self.J_lower

    def delta_Ka(self) -> int:
        """Calculate transition Delta Ka = Ka' - Ka''."""
        return self.Ka_upper - self.Ka_lower

    def delta_Kc(self) -> int:
        """Calculate transition Delta Kc = Kc' - Kc''."""
        return self.Kc_upper - self.Kc_lower

    def to_label(self) -> str:
        """Format quantum numbers as standard spectroscopic notation."""
        return f"{self.J_upper}({self.Ka_upper},{self.Kc_upper}) -> {self.J_lower}({self.Ka_lower},{self.Kc_lower})"


class SpectroscopicTransition(BaseModel):
    """Complete spectroscopic transition representation with physical and ML attributes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    transition_id: str = Field(..., description="Unique alphanumeric identifier for the transition.")
    quantum_numbers: TransitionQuantumNumbers = Field(..., description="Rotational quantum numbers.")
    dipole_type: DipoleType = Field(..., description="Dipole orientation type (a, b, c).")
    dipole_components: DipoleComponents = Field(
        default_factory=DipoleComponents,
        description="Dipole components vector.",
    )
    lower_state_energy_cm1: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower state energy in wavenumbers (cm^-1).",
    )
    frequency_calc_mhz: float = Field(
        ...,
        gt=0.0,
        description="Pure physical calculated ab initio frequency in MHz.",
    )
    frequency_ml_shift_mhz: float = Field(
        default=0.0,
        description="ML-predicted systematic frequency shift Delta nu_ML in MHz.",
    )
    frequency_proposed_mhz: float = Field(
        default=0.0,
        description="Proposed frequency nu_proposed = nu_calc + Delta nu_ML in MHz.",
    )
    frequency_exp_mhz: Optional[float] = Field(
        default=None,
        description="Assigned experimental frequency in MHz (None if unassigned).",
    )
    uncertainty_exp_mhz: float = Field(
        default=0.01,
        gt=0.0,
        description="Experimental frequency measurement uncertainty in MHz.",
    )
    ml_uncertainty_mhz: float = Field(
        default=0.0,
        ge=0.0,
        description="Predictive standard deviation sigma_predict from the Gaussian Process model in MHz.",
    )
    tag: AssignmentTag = Field(
        default=AssignmentTag.TORQ,
        description="Current assignment provenance tag.",
    )
    probable_correctness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probabilistic correctness score in [0.0, 1.0].",
    )
    is_locked: bool = Field(
        default=False,
        description="Whether the user has locked this assignment as ground truth.",
    )
    is_ignored: bool = Field(
        default=False,
        description="Whether this candidate line is ignored during optimization.",
    )
    is_deleted: bool = Field(
        default=False,
        description="Whether this line has been removed from consideration.",
    )

    @model_validator(mode="after")
    def sync_proposed_frequency(self) -> SpectroscopicTransition:
        """Ensure proposed frequency matches physical frequency plus ML shift."""
        if self.frequency_proposed_mhz == 0.0:
            object.__setattr__(
                self,
                "frequency_proposed_mhz",
                float(self.frequency_calc_mhz + self.frequency_ml_shift_mhz),
            )
        return self

    @property
    def obs_minus_calc_mhz(self) -> Optional[float]:
        """Compute Observed minus Physical Calculated residual in MHz."""
        if self.frequency_exp_mhz is not None:
            return float(self.frequency_exp_mhz - self.frequency_calc_mhz)
        return None

    @property
    def obs_minus_proposed_mhz(self) -> Optional[float]:
        """Compute Observed minus ML-Proposed residual in MHz."""
        if self.frequency_exp_mhz is not None:
            return float(self.frequency_exp_mhz - self.frequency_proposed_mhz)
        return None


# =============================================================================
# 6. FITSTATE DAG MODEL
# =============================================================================

class FitState(BaseModel):
    """Immutable state snapshot in the Directed Acyclic Graph (DAG) fit history."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_id: str = Field(..., description="Unique state identifier or cryptographic state hash.")
    parent_state_id: Optional[str] = Field(
        default=None,
        description="Parent state identifier in the history DAG.",
    )
    branch_name: str = Field(
        default="main",
        description="Git-like branch name for alternative fitting explorations.",
    )
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp.",
    )
    rotational_constants: Dict[str, float] = Field(
        ...,
        description="Physical rotational and centrifugal distortion constants in MHz (e.g. A, B, C, DJ, DJK, DK).",
    )
    parameter_uncertainties: Dict[str, float] = Field(
        default_factory=dict,
        description="Standard errors (1-sigma uncertainties) for each fitted parameter in MHz.",
    )
    covariance_matrix: List[List[float]] = Field(
        default_factory=list,
        description="Full parameter covariance matrix.",
    )
    locked_transitions: List[SpectroscopicTransition] = Field(
        default_factory=list,
        description="List of locked ground-truth experimental assignments.",
    )
    unassigned_transitions: List[SpectroscopicTransition] = Field(
        default_factory=list,
        description="List of candidate unassigned theoretical transitions.",
    )
    global_rmsd_khz: float = Field(
        default=0.0,
        ge=0.0,
        description="Global Root Mean Square Deviation of the fit in kHz.",
    )
    chi_squared: float = Field(
        default=0.0,
        ge=0.0,
        description="Reduced or total chi-squared goodness of fit metric.",
    )
    ml_weights_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary metadata describing the trained ML model weights and hyperparameters.",
    )
    provenance: ProvenanceLedger = Field(
        ...,
        description="Cryptographic provenance record for this fitting state.",
    )

    def compute_state_hash(self) -> str:
        """
        Compute deterministic SHA-256 hash identifying this specific state configuration.
        """
        summary_repr = {
            "parent_state_id": self.parent_state_id,
            "branch_name": self.branch_name,
            "rotational_constants": sorted(self.rotational_constants.items()),
            "locked_transition_ids": sorted([t.transition_id for t in self.locked_transitions]),
            "global_rmsd_khz": round(self.global_rmsd_khz, 6),
            "chi_squared": round(self.chi_squared, 6),
            "provenance_hash": self.provenance.compute_ledger_hash(),
        }
        return compute_object_sha256(summary_repr)


# =============================================================================
# 7. PARITY AND NAVIGATOR MODELS
# =============================================================================

class ParityStatus(str, Enum):
    """Convergence and comparison status between JAX autodiff and legacy SPFIT."""
    CONVERGED_PARITY = "CONVERGED_PARITY"
    PARITY_WARNING = "PARITY_WARNING"
    PARITY_DIVERGED = "PARITY_DIVERGED"


class ParityWarning(BaseModel):
    """Comparison report contrasting JAX and SPFIT rotational parameters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    is_warning: bool = Field(
        ...,
        description="Whether parameter discrepancy exceeds the authoritative threshold.",
    )
    parity_status: ParityStatus = Field(
        ...,
        description="Overall parity classification.",
    )
    max_delta_khz: float = Field(
        ...,
        ge=0.0,
        description="Maximum absolute difference between JAX and SPFIT constants in kHz.",
    )
    threshold_khz: float = Field(
        default=0.1,
        gt=0.0,
        description="Parity warning threshold in kHz (default 0.1 kHz).",
    )
    delta_constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter-by-parameter absolute delta values in kHz.",
    )
    jax_constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameters determined by the pure JAX engine in MHz.",
    )
    spfit_constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameters determined by legacy SPFIT in MHz.",
    )
    details: str = Field(
        default="",
        description="Descriptive summary of the comparison.",
    )


class HardwareScanChunk(BaseModel):
    """Hardware-aware FT-MW spectrometer scan recommendation chunk."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    chunk_id: int = Field(..., ge=0, description="Sequential index of the scan chunk.")
    freq_start_mhz: float = Field(..., gt=0.0, description="Starting frequency boundary in MHz.")
    freq_end_mhz: float = Field(..., gt=0.0, description="Ending frequency boundary in MHz.")
    bandwidth_mhz: float = Field(..., gt=0.0, description="Total chunk bandwidth in MHz.")
    total_information_gain: float = Field(
        ...,
        ge=0.0,
        description="Aggregated Jacobian information gain across resolvable transitions in chunk.",
    )
    expected_clean_lines_count: int = Field(
        ...,
        ge=0,
        description="Number of well-resolved candidate transitions within this chunk.",
    )
    targeted_constants: List[str] = Field(
        default_factory=list,
        description="Physical constants most strongly decoupled by transitions in this chunk.",
    )
    resolvability_scores: List[float] = Field(
        default_factory=list,
        description="Resolvability penalty scores for candidate lines in this chunk.",
    )
    transition_ids: List[str] = Field(
        default_factory=list,
        description="Identifiers of candidate transitions enclosed within this frequency window.",
    )
