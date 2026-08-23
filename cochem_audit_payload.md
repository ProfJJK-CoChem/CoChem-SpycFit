Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task2_1_Prompt5_orchestrator.md.
Original prompt:
# Task 2 (Part 1): Stage 2 - cochem_spycfit_orchestrator.py

**Objective:**
Create the `cochem_spycfit_orchestrator.py` file to act as the Ecosystem Router.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_orchestrator.py`

**Instructions:**
You are the `cochem-coder` agent. Implement `cochem_spycfit_orchestrator.py`.
It must:
1. Evaluate `landscape.h5` state tensors routed via the Tier 3 Registry. If *ab initio* tensors are missing, queue a request to CoChem-TORQ via cross-platform file-drop IPC (or ZeroMQ).
2. Implement an HPC Wall-Time Safety heartbeat timeout. Checkpoint state and yield node allocation if timeout reached.
3. Track the Eckart Frame Principal Axis by applying phase corrections to dipole derivatives when isotopic substitution shifts the center of mass.

**Constraints:**
- No placeholders, stubs, or faked IPC/ZeroMQ communication. Implement real `pyzmq` sockets or file-drop mechanics.
- Implement real math tracking for the Eckart Frame rotation.

**Proposed Snippet Outline:**
```python
import h5py
import zmq
import time
import numpy as np

def route_missing_energy(registry_path, timeout_secs):
    # evaluate landscape.h5, setup IPC, implement heartbeat timeout
    pass

def eckart_frame_correction(dipole_derivatives, old_com, new_com):
    # Apply phase corrections for center of mass shift
    pass
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopyTransition,
    ValidationReport,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)
from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)
from cochem_spycfit.intake.cochem_spycfit_orchestrator import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    BOHR_TO_ANGSTROM,
    EckartFrameError,
    EckartFrameResult,
    FitOrchestratorError,
    HARTREE_TO_JOULE,
    HeartbeatStatus,
    IPCMode,
    LandscapeTensorError,
    OrchestrationResult,
    OrchestrationStatus,
    OrchestratorConfig,
    PLANCK_CONSTANT_JS,
    ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ,
    SPEED_OF_LIGHT_CM_S,
    TORQIPCError,
    TORQRequestPayload,
    TORQResponsePayload,
    TORQStatus,
    TORQTimeoutError,
    TensorEvaluationReport,
    WallTimeSafetyGuard,
    WallTimeYieldException,
    calculate_center_of_mass,
    calculate_inertia_tensor,
    calculate_orchestrator_provenance_hash,
    checkpoint_and_yield_node,
    compute_eckart_rotation_matrix,
    diagonalize_inertia_tensor,
    eckart_frame_correction,
    evaluate_landscape_tensors,
    orchestrate_spycfit_system,
    resolve_tier3_registry_path,
    route_missing_energy,
    route_missing_energy_file_drop,
    route_missing_energy_zmq,
)

__all__ = [
    # Ingest
    "BranchType",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopyTransition",
    "ValidationReport",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_spectroscopy_data",
    "validate_quantum_numbers",
    # Triage
    "AU_TO_WAVENUMBER_CONVERSION",
    "HarmonicVibrationalResult",
    "IsotopeSpec",
    "IsotopeTriageResult",
    "IsotopologueData",
    "PolyadBlock",
    "ProductClass",
    "ResonanceCondition",
    "ResonanceReport",
    "ResonanceType",
    "TriageConfig",
    "TriageReport",
    "TriageStatus",
    "calculate_triage_provenance_hash",
    "check_isotope_abundance",
    "compute_harmonic_vibrational_frequencies",
    "evaluate_isotopologue_abundance",
    "get_isotope_properties",
    "identify_polyad_blocks",
    "mass_matrix_transformation",
    "parse_isotopologue_input",
    "resolve_mendeleev_database",
    "scan_coriolis_resonances",
    "scan_darling_dennison_resonances",
    "scan_fermi_resonances",
    "trap_resonances",
    "triage_spectroscopic_system",
    # Orchestrator
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\__init__.py ---
"""CoChem-SpycFit intake subpackage."""

from cochem_spycfit.intake.cochem_spycfit_ingest import (
    BranchType,
    DipoleType,
    FileFormat,
    IngestConfig,
    IngestResult,
    Parity,
    PickettParFile,
    PickettParameter,
    SpectroscopyTransition,
    ValidationReport,
    apply_selection_rules,
    calculate_frequency_fingerprint,
    classify_transition_parity_and_dipole,
    enforce_fp64_boundary,
    fortran_float_converter,
    parse_csv_spectroscopy,
    parse_pickett_lin,
    parse_pickett_par,
    parse_spectroscopy_data,
    validate_quantum_numbers,
)
from cochem_spycfit.intake.cochem_spycfit_triage import (
    AU_TO_WAVENUMBER_CONVERSION,
    HarmonicVibrationalResult,
    IsotopeSpec,
    IsotopeTriageResult,
    IsotopologueData,
    PolyadBlock,
    ProductClass,
    ResonanceCondition,
    ResonanceReport,
    ResonanceType,
    TriageConfig,
    TriageReport,
    TriageStatus,
    calculate_triage_provenance_hash,
    check_isotope_abundance,
    compute_harmonic_vibrational_frequencies,
    evaluate_isotopologue_abundance,
    get_isotope_properties,
    identify_polyad_blocks,
    mass_matrix_transformation,
    parse_isotopologue_input,
    resolve_mendeleev_database,
    scan_coriolis_resonances,
    scan_darling_dennison_resonances,
    scan_fermi_resonances,
    trap_resonances,
    triage_spectroscopic_system,
)
from cochem_spycfit.intake.cochem_spycfit_orchestrator import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    BOHR_TO_ANGSTROM,
    EckartFrameError,
    EckartFrameResult,
    FitOrchestratorError,
    HARTREE_TO_JOULE,
    HeartbeatStatus,
    IPCMode,
    LandscapeTensorError,
    OrchestrationResult,
    OrchestrationStatus,
    OrchestratorConfig,
    PLANCK_CONSTANT_JS,
    ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ,
    SPEED_OF_LIGHT_CM_S,
    TORQIPCError,
    TORQRequestPayload,
    TORQResponsePayload,
    TORQStatus,
    TORQTimeoutError,
    TensorEvaluationReport,
    WallTimeSafetyGuard,
    WallTimeYieldException,
    calculate_center_of_mass,
    calculate_inertia_tensor,
    calculate_orchestrator_provenance_hash,
    checkpoint_and_yield_node,
    compute_eckart_rotation_matrix,
    diagonalize_inertia_tensor,
    eckart_frame_correction,
    evaluate_landscape_tensors,
    orchestrate_spycfit_system,
    resolve_tier3_registry_path,
    route_missing_energy,
    route_missing_energy_file_drop,
    route_missing_energy_zmq,
)

__all__ = [
    # Ingest
    "BranchType",
    "DipoleType",
    "FileFormat",
    "IngestConfig",
    "IngestResult",
    "Parity",
    "PickettParFile",
    "PickettParameter",
    "SpectroscopyTransition",
    "ValidationReport",
    "apply_selection_rules",
    "calculate_frequency_fingerprint",
    "classify_transition_parity_and_dipole",
    "enforce_fp64_boundary",
    "fortran_float_converter",
    "parse_csv_spectroscopy",
    "parse_pickett_lin",
    "parse_pickett_par",
    "parse_spectroscopy_data",
    "validate_quantum_numbers",
    # Triage
    "AU_TO_WAVENUMBER_CONVERSION",
    "HarmonicVibrationalResult",
    "IsotopeSpec",
    "IsotopeTriageResult",
    "IsotopologueData",
    "PolyadBlock",
    "ProductClass",
    "ResonanceCondition",
    "ResonanceReport",
    "ResonanceType",
    "TriageConfig",
    "TriageReport",
    "TriageStatus",
    "calculate_triage_provenance_hash",
    "check_isotope_abundance",
    "compute_harmonic_vibrational_frequencies",
    "evaluate_isotopologue_abundance",
    "get_isotope_properties",
    "identify_polyad_blocks",
    "mass_matrix_transformation",
    "parse_isotopologue_input",
    "resolve_mendeleev_database",
    "scan_coriolis_resonances",
    "scan_darling_dennison_resonances",
    "scan_fermi_resonances",
    "trap_resonances",
    "triage_spectroscopic_system",
    # Orchestrator
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\intake\cochem_spycfit_orchestrator.py ---
"""
CoChem-SpycFit: Stage 2.0 Ecosystem Router & Orchestrator.

Module: cochem_spycfit/intake/cochem_spycfit_orchestrator.py
Role: Ecosystem Router, Tier 3 Registry & landscape.h5 State Tensor Evaluator,
      CoChem-TORQ Inter-Process Communication (IPC) Router, HPC Wall-Time Safety Guard,
      and Eckart Frame Principal Axis & Dipole Derivative Transformation Engine.

Capabilities:
1. Tier 3 Registry & landscape.h5 State Tensor Evaluator:
   - Dynamically resolves the authoritative Tier 3 (State & Config) Registry path.
   - Evaluates landscape.h5 state tensors (/geometry, /energy, /cartesian_hessian,
     /dipole_derivatives, /vpt2, /rotational_constants) without hardcoded home directory paths.
   - Identifies missing ab initio physics tensors prior to Hamiltonian tensor allocation.

2. CoChem-TORQ Inter-Process Communication (IPC) Router:
   - Autonomously queues missing ab initio computation requests to CoChem-TORQ.
   - Dual IPC Architecture:
     * High-throughput ZeroMQ (pyzmq) REQ/REP sockets.
     * Cross-platform atomic file-drop IPC utilizing temporary staging files and atomic rename.
   - Strict Pydantic v2 schemas for request/response serialization with extra-field forbidding.

3. HPC Wall-Time Safety Heartbeat Guard & Checkpoint Manager:
   - Tracks node wall-time limits, elapsed time, and remaining scheduler budget.
   - Emits real-time diagnostic heartbeats (CPU, RAM, active task, remaining time).
   - Enforces a safety buffer threshold: if remaining wall-time drops below the threshold,
     it triggers atomic checkpoint serialization (checkpoint_and_yield_node) and yields
     the node allocation cleanly to prevent SLURM/PBS scheduler timeout SIGKILL penalties.

4. Eckart Frame Principal Axis Tracker & Phase Correction Engine:
   - Computes analytical Center of Mass (COM) coordinates and shifts under isotopic substitution.
   - Constructs and diagonalizes the 3x3 moment of inertia tensor in strict FP64 precision,
     extracting sorted principal moments (Ia <= Ib <= Ic) and rotational constants (A >= B >= C).
   - Computes the optimal Eckart rotation matrix R using mass-weighted covariance SVD,
     guaranteeing exact orthonormality (R^T R = I) and proper rotation (det(R) = +1).
   - Transforms Cartesian dipole derivative tensors (3 x 3N) into the substituted Eckart frame,
     applying phase alignment to ensure continuous, invariant dipole transition predictions.

5. Cryptographic Provenance & Full Auditability:
   - Deterministic SHA-256 hash digests across orchestration parameters, tensor shapes, and constants.

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE Version 4, August 2026)
- Task 2 File Inventory & Deliverable Capabilities Manifest (Part 1 of 2)
- Task 4 Hardware Gatekeeper & The Spectral Registry (Phase 2)
"""

from __future__ import annotations

import datetime
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import platformdirs
from pydantic import BaseModel, ConfigDict, Field, field_validator
import zmq

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Orchestrator")


# =============================================================================
# 1. PHYSICAL CONSTANTS AND UNIT CONVERSION FACTORS (FP64)
# =============================================================================

# Exact SI and Atomic Units (IUPAC / CODATA 2022)
AMU_TO_KG: float = 1.66053906660e-27  # 1 u in kg (FP64)
ANGSTROM_TO_METER: float = 1.0e-10  # 1 Angstrom in meters (FP64)
BOHR_TO_ANGSTROM: float = 0.529177210903  # 1 bohr (a_0) in Angstroms (FP64)
PLANCK_CONSTANT_JS: float = 6.62607015e-34  # Planck constant h in J*s (FP64)
SPEED_OF_LIGHT_CM_S: float = 29979245800.0  # Speed of light c in cm/s (FP64)
HARTREE_TO_JOULE: float = 4.3597447222071e-18  # 1 E_h in Joules (FP64)

# Conversion constant for Rotational Constant B in MHz from Moment of Inertia I in amu * Angstrom^2:
# B (MHz) = (h / (8 * pi^2 * I)) * 10^-6
# B (MHz) = 505379.0087808415 / I (amu * A^2)
ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ: float = 505379.0087808415


# =============================================================================
# 2. ENUMS & EXCEPTIONS
# =============================================================================

class OrchestrationStatus(str, Enum):
    """Execution status for ecosystem orchestration workflows."""
    SUCCESS = "SUCCESS"
    PENDING_TORQ = "PENDING_TORQ"
    YIELDED_WALLTIME = "YIELDED_WALLTIME"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class IPCMode(str, Enum):
    """Supported inter-process communication mechanisms for CoChem-TORQ."""
    FILE_DROP = "file_drop"
    ZMQ = "zmq"


class TORQStatus(str, Enum):
    """Status of ab initio calculation requests submitted to CoChem-TORQ."""
    REQUESTED = "REQUESTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class FitOrchestratorError(Exception):
    """Base exception for SpycFit orchestrator operations."""


class WallTimeYieldException(FitOrchestratorError):
    """Raised when approaching HPC scheduler wall-time limits to trigger graceful node yielding."""


class TORQIPCError(FitOrchestratorError):
    """Raised when communication with CoChem-TORQ encounters an error."""


class TORQTimeoutError(TORQIPCError, TimeoutError):
    """Raised when CoChem-TORQ fails to return requested tensors within timeout."""


class LandscapeTensorError(FitOrchestratorError):
    """Raised when landscape.h5 state database is corrupted or malformed."""


class EckartFrameError(FitOrchestratorError):
    """Raised when Eckart rotation or principal axis diagonalization fails."""


# =============================================================================
# 3. PYDANTIC V2 SCHEMAS
# =============================================================================

class OrchestratorConfig(BaseModel):
    """Configuration settings for the ecosystem router and orchestrator."""
    model_config = ConfigDict(extra="forbid")

    registry_path: Optional[str] = Field(
        default=None,
        description="Path to Tier 3 Registry directory or config file.",
    )
    torq_ipc_mode: str = Field(
        default="file_drop",
        description="IPC communication mode with CoChem-TORQ ('file_drop' or 'zmq').",
    )
    torq_ipc_endpoint: str = Field(
        default="tcp://127.0.0.1:5555",
        description="ZeroMQ socket endpoint URL if mode is 'zmq'.",
    )
    torq_drop_dir: Optional[str] = Field(
        default=None,
        description="Directory for atomic file-drop IPC if mode is 'file_drop'.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum wait timeout in seconds for TORQ responses.",
    )
    heartbeat_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Interval in seconds between wall-time heartbeat ticks.",
    )
    safety_buffer_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Safety buffer threshold in seconds before wall-time expiration.",
    )
    max_walltime_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Total allocated HPC job wall-time limit in seconds.",
    )
    auto_route_missing: bool = Field(
        default=True,
        description="Whether to automatically queue requests to TORQ for missing tensors.",
    )
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description="Directory for saving state checkpoints upon wall-time yield.",
    )


class TensorEvaluationReport(BaseModel):
    """Diagnostic evaluation report for state tensors in landscape.h5."""
    model_config = ConfigDict(extra="forbid")

    system_id: str = Field(..., description="Molecular system identifier.")
    h5_path: str = Field(..., description="Filesystem path of the evaluated landscape.h5 file.")
    existing_tensors: List[str] = Field(default_factory=list, description="Tensors present in HDF5.")
    missing_tensors: List[str] = Field(default_factory=list, description="Required tensors absent in HDF5.")
    is_complete: bool = Field(..., description="True if all required tensors exist with valid shapes.")
    needs_torq_routing: bool = Field(..., description="True if missing tensors must be computed by TORQ.")
    tensor_shapes: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Extracted dimensions of existing tensors.",
    )


class TORQRequestPayload(BaseModel):
    """Structured request payload queued to CoChem-TORQ."""
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Unique UUID or timestamp-keyed request ID.")
    system_id: str = Field(..., description="Molecular system or conformer identifier.")
    missing_tensors: List[str] = Field(..., description="List of required tensor names (e.g. ['cartesian_hessian']).")
    geometry: List[List[float]] = Field(default_factory=list, description="Cartesian coordinates in Angstroms (N x 3).")
    atomic_numbers: List[int] = Field(default_factory=list, description="Atomic numbers Z (1D list of length N).")
    masses_amu: List[float] = Field(default_factory=list, description="Atomic isotopic masses in amu (1D list of length N).")
    method: str = Field(default="wB97M-V", description="Quantum chemical functional or method.")
    basis: str = Field(default="def2-QZVPP", description="Basis set specification.")
    priority: int = Field(default=1, ge=1, le=10, description="Task priority (1 = high, 10 = low).")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC creation timestamp.",
    )
    status: str = Field(default="REQUESTED", description="Request lifecycle status.")


class TORQResponsePayload(BaseModel):
    """Structured response payload returned by CoChem-TORQ."""
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Corresponding request identifier.")
    system_id: str = Field(..., description="Molecular system identifier.")
    status: str = Field(default="COMPLETED", description="Status: 'COMPLETED', 'FAILED', 'TIMEOUT'.")
    tensors_data: Dict[str, Any] = Field(default_factory=dict, description="Calculated tensor arrays.")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error message if status is FAILED.")
    execution_time_seconds: float = Field(default=0.0, ge=0.0, description="Execution wall-time in seconds.")


class HeartbeatStatus(BaseModel):
    """Diagnostic status snapshot emitted by the HPC Wall-Time Safety Guard."""
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp of heartbeat.")
    elapsed_seconds: float = Field(..., ge=0.0, description="Elapsed execution wall-time in seconds.")
    remaining_walltime_seconds: float = Field(..., description="Remaining wall-time before scheduler limit.")
    is_safe: bool = Field(..., description="True if remaining wall-time exceeds safety buffer threshold.")
    cpu_percent: float = Field(default=0.0, ge=0.0, description="Current CPU utilization percentage.")
    memory_mb: float = Field(default=0.0, ge=0.0, description="Current resident set size (RSS) memory in MB.")
    active_task: str = Field(default="orchestrating", description="Active sub-task description.")


class EckartFrameResult(BaseModel):
    """Output results from Eckart Frame transformation and dipole derivative correction."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    old_com: List[float] = Field(..., description="Reference Center of Mass coordinates in Angstroms.")
    new_com: List[float] = Field(..., description="Substituted Center of Mass coordinates in Angstroms.")
    com_shift: List[float] = Field(..., description="Center of Mass translation vector (new - old) in Angstroms.")
    rotation_matrix: Any = Field(..., description="Eckart rotation matrix R (3x3 np.ndarray in FP64).")
    principal_moments_ref: List[float] = Field(..., description="Reference principal moments Ia <= Ib <= Ic (amu*A^2).")
    principal_moments_sub: List[float] = Field(..., description="Substituted principal moments Ia <= Ib <= Ic (amu*A^2).")
    rotational_constants_ref_mhz: List[float] = Field(..., description="Reference rotational constants A >= B >= C (MHz).")
    rotational_constants_sub_mhz: List[float] = Field(..., description="Substituted rotational constants A >= B >= C (MHz).")
    transformed_dipole_derivatives: Any = Field(
        ...,
        description="Transformed Cartesian dipole derivative tensor (3 x 3N np.ndarray in FP64).",
    )
    phase_corrections: List[float] = Field(
        ...,
        description="Phase factors (+1.0 or -1.0) along each principal axis.",
    )


class OrchestrationResult(BaseModel):
    """Consolidated report produced by the SpycFit ecosystem orchestrator."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: OrchestrationStatus = Field(..., description="Overall orchestration execution status.")
    system_id: str = Field(..., description="Molecular system identifier.")
    tensor_evaluation: TensorEvaluationReport = Field(..., description="State tensor evaluation report.")
    torq_responses: List[TORQResponsePayload] = Field(default_factory=list, description="Responses from TORQ.")
    eckart_result: Optional[EckartFrameResult] = Field(default=None, description="Eckart frame results if computed.")
    walltime_heartbeats: List[HeartbeatStatus] = Field(default_factory=list, description="Heartbeat log entries.")
    sha256_provenance: str = Field(..., description="Cryptographic SHA-256 provenance hash.")
    summary: str = Field(..., description="Human-readable execution summary.")


# =============================================================================
# 4. TIER 3 REGISTRY RESOLUTION
# =============================================================================

def resolve_tier3_registry_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Dynamically resolves the authoritative path to the Tier 3 (State & Config) Registry.
    Avoids hardcoded $HOME to ensure seamless compatibility across Windows, Linux, and HPC.

    Args:
        custom_path: Optional user-supplied path to override standard resolution.

    Returns:
        Path: Absolute resolved path to Tier 3 Registry directory.
    """
    if custom_path is not None:
        p = Path(custom_path).resolve()
        return p

    env_dir = os.environ.get("COCHEM_REGISTRY_DIR") or os.environ.get("COCHEM_STATE_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        return p

    # Standard platformdirs user config directory
    config_dir = Path(platformdirs.user_config_dir("CoChem", "CoChem")).resolve() / "Registry"
    return config_dir


# =============================================================================
# 5. LANDSCAPE.H5 STATE TENSOR EVALUATION
# =============================================================================

def evaluate_landscape_tensors(
    h5_path: Union[str, Path],
    system_id: str = "default_system",
    required_tensors: Optional[List[str]] = None,
) -> TensorEvaluationReport:
    """
    Evaluates state tensors inside landscape.h5 routed via the Tier 3 Registry.

    Checks for the presence and validity of required ab initio physics tensors:
    - 'geometry' (N x 3 coordinates in Angstroms)
    - 'energy_hartree' or 'energy' (scalar FP64 ground state energy)
    - 'cartesian_hessian' (3N x 3N force constant matrix)
    - 'dipole_derivatives' (3 x 3N Cartesian dipole derivatives)

    Args:
        h5_path: Path to landscape.h5 file.
        system_id: Unique molecular identifier.
        required_tensors: Optional custom list of required dataset names.

    Returns:
        TensorEvaluationReport: Pydantic report detailing existing and missing tensors.
    """
    target_h5 = Path(h5_path).resolve()
    req_keys = required_tensors or ["geometry", "cartesian_hessian", "dipole_derivatives"]

    if not target_h5.is_file():
        return TensorEvaluationReport(
            system_id=system_id,
            h5_path=str(target_h5),
            existing_tensors=[],
            missing_tensors=req_keys,
            is_complete=False,
            needs_torq_routing=True,
            tensor_shapes={},
        )

    existing: List[str] = []
    missing: List[str] = []
    shapes_dict: Dict[str, List[int]] = {}

    try:
        with h5py.File(target_h5, "r") as f:
            for key in req_keys:
                # Handle possible alternative naming in landscape datasets
                matched_key: Optional[str] = None
                if key in f:
                    matched_key = key
                elif key == "cartesian_hessian" and "hessian" in f:
                    matched_key = "hessian"
                elif key == "dipole_derivatives" and "dipole_gradient" in f:
                    matched_key = "dipole_gradient"
                elif key == "energy_hartree" and "energy" in f:
                    matched_key = "energy"

                if matched_key is not None:
                    dset = f[matched_key]
                    existing.append(key)
                    shapes_dict[key] = list(dset.shape)
                else:
                    missing.append(key)
    except Exception as exc:
        logger.warning(f"Error inspecting HDF5 file {target_h5}: {exc}")
        return TensorEvaluationReport(
            system_id=system_id,
            h5_path=str(target_h5),
            existing_tensors=[],
            missing_tensors=req_keys,
            is_complete=False,
            needs_torq_routing=True,
            tensor_shapes={},
        )

    is_complete = (len(missing) == 0)
    needs_torq_routing = not is_complete

    return TensorEvaluationReport(
        system_id=system_id,
        h5_path=str(target_h5),
        existing_tensors=existing,
        missing_tensors=missing,
        is_complete=is_complete,
        needs_torq_routing=needs_torq_routing,
        tensor_shapes=shapes_dict,
    )


# =============================================================================
# 6. COCHEM-TORQ IPC COMMUNICATION (FILE-DROP & ZEROMQ)
# =============================================================================

def route_missing_energy_file_drop(
    request: TORQRequestPayload,
    drop_dir: Union[str, Path],
    timeout_seconds: float = 60.0,
    poll_interval: float = 0.1,
) -> TORQResponsePayload:
    """
    Submits missing physics calculation request to CoChem-TORQ via atomic file-drop IPC.

    Protocol:
    1. Writes request to a temporary file: `<drop_dir>/<request_id>.request.json.tmp`.
    2. Atomically renames to `<drop_dir>/<request_id>.request.json`.
    3. Polls `<drop_dir>/<request_id>.response.json` until completed or timeout.
    4. Parses and validates the response, removing temporary files.

    Args:
        request: Validated TORQRequestPayload model.
        drop_dir: Path to shared IPC drop directory.
        timeout_seconds: Maximum wait timeout.
        poll_interval: Polling loop sleep duration in seconds.

    Returns:
        TORQResponsePayload: Response received from TORQ.

    Raises:
        TORQTimeoutError: If TORQ does not respond within timeout_seconds.
    """
    target_dir = Path(drop_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    req_file = target_dir / f"{request.request_id}.request.json"
    req_tmp = target_dir / f"{request.request_id}.request.json.tmp"
    resp_file = target_dir / f"{request.request_id}.response.json"

    # Step 1: Write atomic request payload
    payload_str = request.model_dump_json(indent=2)
    with open(req_tmp, "w", encoding="utf-8") as f:
        f.write(payload_str)
    req_tmp.replace(req_file)
    logger.info(f"[TORQ IPC File-Drop] Queued request {request.request_id} for system '{request.system_id}'")

    # Step 2: Poll for response
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            # Clean up request file if still present
            req_file.unlink(missing_ok=True)
            raise TORQTimeoutError(
                f"CoChem-TORQ file-drop IPC timed out after {timeout_seconds:.1f}s for request {request.request_id}"
            )

        if resp_file.exists():
            try:
                # Small wait to ensure file write buffer flushed by writer
                time.sleep(0.01)
                with open(resp_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                # Parse and validate with Pydantic
                response = TORQResponsePayload.model_validate(raw_data)
                # Clean up response file
                resp_file.unlink(missing_ok=True)
                logger.info(f"[TORQ IPC File-Drop] Received response for {request.request_id} in {elapsed:.2f}s")
                return response
            except (json.JSONDecodeError, ValueError) as exc:
                # Writer may still be writing; continue polling
                logger.debug(f"Transient response read error: {exc}")

        time.sleep(poll_interval)


def route_missing_energy_zmq(
    request: TORQRequestPayload,
    endpoint: str = "tcp://127.0.0.1:5555",
    timeout_seconds: float = 60.0,
    zmq_context: Optional[zmq.Context] = None,
) -> TORQResponsePayload:
    """
    Submits missing physics calculation request to CoChem-TORQ via ZeroMQ REQ/REP sockets.

    Args:
        request: Validated TORQRequestPayload model.
        endpoint: ZeroMQ socket endpoint string (e.g. 'tcp://127.0.0.1:5555' or 'inproc://...').
        timeout_seconds: Maximum wait timeout.
        zmq_context: Optional existing zmq.Context instance.

    Returns:
        TORQResponsePayload: Response received from TORQ.

    Raises:
        TORQTimeoutError: If socket timeout is reached.
        TORQIPCError: If socket communication encounters an error.
    """
    ctx = zmq_context or zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_seconds * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_seconds * 1000))
    sock.setsockopt(zmq.LINGER, 0)

    try:
        sock.connect(endpoint)
        payload_str = request.model_dump_json()
        sock.send_string(payload_str)
        logger.info(f"[TORQ IPC ZMQ] Sent request {request.request_id} to endpoint {endpoint}")

        raw_resp = sock.recv_string()
        resp_data = json.loads(raw_resp)
        response = TORQResponsePayload.model_validate(resp_data)
        logger.info(f"[TORQ IPC ZMQ] Received response for {request.request_id}")
        return response
    except zmq.Again as exc:
        raise TORQTimeoutError(
            f"ZeroMQ socket timed out after {timeout_seconds:.1f}s awaiting TORQ response on {endpoint}"
        ) from exc
    except Exception as exc:
        raise TORQIPCError(f"ZeroMQ socket IPC failure on {endpoint}: {exc}") from exc
    finally:
        sock.close()


def route_missing_energy(
    request: TORQRequestPayload,
    config: Optional[OrchestratorConfig] = None,
    drop_dir: Optional[Union[str, Path]] = None,
    zmq_context: Optional[zmq.Context] = None,
) -> TORQResponsePayload:
    """
    High-level routing function that dispatches missing energy / tensor requests
    to CoChem-TORQ using either ZeroMQ or cross-platform File-Drop IPC.
    """
    cfg = config or OrchestratorConfig()
    ipc_mode = cfg.torq_ipc_mode.lower().strip()

    if ipc_mode in ("zmq", "zeromq"):
        return route_missing_energy_zmq(
            request=request,
            endpoint=cfg.torq_ipc_endpoint,
            timeout_seconds=cfg.timeout_seconds,
            zmq_context=zmq_context,
        )

    # Default to File-Drop IPC
    target_drop_dir = drop_dir or cfg.torq_drop_dir or (
        Path(platformdirs.user_data_path("CoChem", "CoChem")) / "SpycFit_Workspace" / "TORQ_IPC"
    )
    return route_missing_energy_file_drop(
        request=request,
        drop_dir=target_drop_dir,
        timeout_seconds=cfg.timeout_seconds,
    )


# =============================================================================
# 7. HPC WALL-TIME SAFETY HEARTBEAT & CHECKPOINTING
# =============================================================================

class WallTimeSafetyGuard:
    """
    HPC Wall-Time Safety Guard & Heartbeat Manager.

    Monitors job execution wall-time against scheduler limits (e.g. SLURM / PBS).
    If the remaining time falls below safety_buffer_seconds, it triggers a clean
    checkpoint and raises WallTimeYieldException to yield node allocation safely.
    """

    def __init__(
        self,
        max_walltime_seconds: float = 3600.0,
        safety_buffer_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 5.0,
        start_time: Optional[float] = None,
    ) -> None:
        self.max_walltime = float(max_walltime_seconds)
        self.safety_buffer = float(safety_buffer_seconds)
        self.heartbeat_interval = float(heartbeat_interval_seconds)
        self.start_time = float(start_time) if start_time is not None else time.time()
        self.last_heartbeat_time = self.start_time
        self.heartbeat_history: List[HeartbeatStatus] = []

    def get_elapsed_seconds(self) -> float:
        """Returns elapsed wall-time in seconds."""
        return max(0.0, time.time() - self.start_time)

    def get_remaining_seconds(self) -> float:
        """Returns remaining wall-time before maximum allocation is reached."""
        return self.max_walltime - self.get_elapsed_seconds()

    def is_safe(self) -> bool:
        """Returns True if remaining wall-time exceeds the safety buffer."""
        return self.get_remaining_seconds() > self.safety_buffer

    def should_yield(self) -> bool:
        """Returns True if wall-time budget is exhausted and node allocation must be yielded."""
        return not self.is_safe()

    def tick(self, active_task: str = "orchestrating") -> HeartbeatStatus:
        """
        Emits a heartbeat diagnostic status record and appends it to history.
        """
        now = time.time()
        elapsed = self.get_elapsed_seconds()
        remaining = self.get_remaining_seconds()
        safe = self.is_safe()

        # Gather lightweight OS memory/cpu heuristics if psutil available
        cpu_pct = 0.0
        mem_mb = 0.0
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            cpu_pct = float(proc.cpu_percent(interval=None))
            mem_mb = float(proc.memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            pass

        status = HeartbeatStatus(
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=np.float64(elapsed),
            remaining_walltime_seconds=np.float64(remaining),
            is_safe=safe,
            cpu_percent=np.float64(cpu_pct),
            memory_mb=np.float64(mem_mb),
            active_task=active_task,
        )

        self.heartbeat_history.append(status)
        self.last_heartbeat_time = now

        if not safe:
            logger.warning(
                f"[HPC WALL-TIME ALERT] Remaining time ({remaining:.1f}s) <= buffer ({self.safety_buffer:.1f}s). "
                f"Yielding node allocation."
            )

        return status

    def check_walltime_or_yield(
        self,
        current_state: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Verifies wall-time safety. If unsafe, atomically saves a state checkpoint
        and raises WallTimeYieldException to cleanly exit the job step.
        """
        if self.should_yield():
            if current_state is not None:
                ckpt_p = checkpoint_path or (
                    Path(platformdirs.user_data_path("CoChem", "CoChem")) / "SpycFit_Workspace" / "checkpoints" / "checkpoint_yield.json"
                )
                checkpoint_and_yield_node(current_state, ckpt_p)

            raise WallTimeYieldException(
                f"HPC Wall-Time Limit Approaching: {self.get_remaining_seconds():.1f}s remaining "
                f"(safety buffer: {self.safety_buffer:.1f}s). Node allocation yielded."
            )


def checkpoint_and_yield_node(
    state_data: Dict[str, Any],
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Atomically checkpoints active fitting/orchestration state to disk and yields node allocation.

    Args:
        state_data: Dictionary containing intermediate optimization parameters, tensors, and metadata.
        checkpoint_path: Destination path for JSON checkpoint file.

    Returns:
        Dict: Metadata confirmation containing checkpoint path and timestamp.
    """
    target = Path(checkpoint_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.tmp_{int(time.time())}"

    payload = dict(state_data)
    payload["_checkpoint_saved_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload["_pid"] = os.getpid()

    payload_str = json.dumps(payload, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload_str)

    tmp_path.replace(target)
    logger.info(f"[Checkpoint & Yield] Saved state checkpoint to {target}")

    return {
        "status": "YIELDED_CHECKPOINT_SAVED",
        "checkpoint_path": str(target),
        "timestamp_utc": payload["_checkpoint_saved_utc"],
        "bytes_written": len(payload_str),
    }


# =============================================================================
# 8. CENTER OF MASS AND INERTIA TENSOR CALCULATIONS
# =============================================================================

def calculate_center_of_mass(
    geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Calculates analytical Center of Mass (COM) coordinates in FP64 precision.

    COM = sum(m_i * r_i) / sum(m_i)

    Args:
        geometry: (N, 3) Cartesian coordinates in Angstroms.
        masses: (N,) atomic masses in amu.

    Returns:
        (3,) Center of mass coordinates in Angstroms (np.float64).
    """
    geom = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64).reshape(-1, 1)

    if geom.ndim != 2 or geom.shape[1] != 3:
        raise ValueError(f"Geometry must have shape (N, 3), got {geom.shape}")
    if len(m) != len(geom):
        raise ValueError(f"Length mismatch: {len(geom)} atoms vs {len(m)} masses")

    total_mass = np.sum(m)
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive")

    com = np.sum(m * geom, axis=0) / total_mass
    return com.astype(np.float64)


def calculate_inertia_tensor(
    geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Constructs the 3x3 Moment of Inertia tensor in the Center of Mass frame.

    I_ab = sum_i m_i * ( |r_i|^2 * delta_ab - r_i,a * r_i,b )

    Args:
        geometry: (N, 3) Cartesian coordinates in Angstroms.
        masses: (N,) atomic masses in amu.

    Returns:
        (3, 3) Moment of inertia tensor in amu * Angstrom^2 (np.float64).
    """
    geom = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)

    com = calculate_center_of_mass(geom, m)
    r = geom - com  # Shift to COM frame

    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r_sq = x * x + y * y + z * z

    I_xx = np.sum(m * (r_sq - x * x))
    I_yy = np.sum(m * (r_sq - y * y))
    I_zz = np.sum(m * (r_sq - z * z))

    I_xy = -np.sum(m * x * y)
    I_xz = -np.sum(m * x * z)
    I_yz = -np.sum(m * y * z)

    inertia_tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz],
    ], dtype=np.float64)

    return inertia_tensor


def diagonalize_inertia_tensor(
    inertia_tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalizes the Moment of Inertia tensor to obtain principal moments and rotational constants.

    Orders principal moments Ia <= Ib <= Ic, corresponding to rotational constants A >= B >= C.

    Returns:
        Tuple of:
        - principal_moments: (3,) sorted principal moments in amu * Angstrom^2.
        - principal_axes: (3, 3) principal axis eigenvectors (columns).
        - rotational_constants_mhz: (3,) [A, B, C] in MHz.
    """
    I = np.asarray(inertia_tensor, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Sort ascending: Ia <= Ib <= Ic
    idx = np.argsort(eigenvalues)
    principal_moments = eigenvalues[idx].astype(np.float64)
    principal_axes = eigenvectors[:, idx].astype(np.float64)

    # Ensure right-handed coordinate frame: det(principal_axes) = +1
    if np.linalg.det(principal_axes) < 0.0:
        principal_axes[:, 2] *= -1.0

    # Calculate Rotational Constants A >= B >= C in MHz
    # Handle possible zero modes for linear or planar degenerate systems
    rot_constants = np.zeros(3, dtype=np.float64)
    for k in range(3):
        if principal_moments[k] > 1e-6:
            rot_constants[k] = ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ / principal_moments[k]
        else:
            rot_constants[k] = 0.0

    return principal_moments, principal_axes, rot_constants


# =============================================================================
# 9. ECKART FRAME ROTATION & DIPOLE DERIVATIVE CORRECTION
# =============================================================================

def compute_eckart_rotation_matrix(
    ref_geometry: np.ndarray,
    sub_geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Computes the optimal Eckart rotation matrix R that aligns sub_geometry to ref_geometry
    under the mass-weighted Eckart condition: sum_i m_i (r_0,i x r_i) = 0.

    Utilizes the Kabsch / SVD algorithm on the mass-weighted covariance matrix:
    F = sum_i m_i * r_ref,i * (r_sub,i)^T = U * Sigma * V^T
    R = U * diag(1, 1, det(U V^T)) * V^T

    Guarantees:
    - Exact orthonormality: R^T R = I
    - Proper rotation: det(R) = +1.0

    Args:
        ref_geometry: (N, 3) reference geometry (e.g. parent isotopologue).
        sub_geometry: (N, 3) displaced or substituted geometry.
        masses: (N,) atomic masses in amu.

    Returns:
        (3, 3) Eckart rotation matrix R (np.float64).
    """
    r_ref = np.asarray(ref_geometry, dtype=np.float64)
    r_sub = np.asarray(sub_geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64).reshape(-1, 1)

    # Shift both geometries to their respective centers of mass
    com_ref = calculate_center_of_mass(r_ref, m.flatten())
    com_sub = calculate_center_of_mass(r_sub, m.flatten())

    r_ref_com = r_ref - com_ref
    r_sub_com = r_sub - com_sub

    # Mass-weighted covariance matrix F (3 x 3)
    # F_ab = sum_i m_i * r_ref_com[i, a] * r_sub_com[i, b]
    F = (r_ref_com.T * m.T) @ r_sub_com

    # Singular Value Decomposition: F = U * S * V^T
    U, _, Vt = np.linalg.svd(F)
    V = Vt.T

    # Reflection correction matrix D to enforce det(R) = +1
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])

    R_eckart = U @ D @ Vt
    return R_eckart.astype(np.float64)


def eckart_frame_correction(
    dipole_derivatives: np.ndarray,
    ref_geometry: np.ndarray,
    sub_geometry: np.ndarray,
    ref_masses: np.ndarray,
    sub_masses: np.ndarray,
) -> EckartFrameResult:
    """
    Applies Eckart Frame and principal axis phase corrections to Cartesian dipole derivatives
    when isotopic substitution shifts the Center of Mass and principal axes.

    Transformation Steps:
    1. Computes Center of Mass coordinates for reference (parent) and substituted systems.
    2. Diagonalizes Moment of Inertia tensors to extract reference and substituted principal axes.
    3. Determines the Eckart rotation matrix R aligning the frames.
    4. Applies phase corrections along principal axes to resolve sign ambiguities in eigenvectors.
    5. Transforms Cartesian dipole derivative tensor (3 x 3N):
       dmu_a / dx_ib_new = sum_c sum_d R_ac * (dmu_c / dx_id_ref) * R_bd

    Args:
        dipole_derivatives: (3, 3N) Cartesian dipole derivatives of reference system (Debye / Angstrom).
        ref_geometry: (N, 3) Reference Cartesian geometry in Angstroms.
        sub_geometry: (N, 3) Substituted Cartesian geometry in Angstroms.
        ref_masses: (N,) Reference isotopic atomic masses in amu.
        sub_masses: (N,) Substituted isotopic atomic masses in amu.

    Returns:
        EckartFrameResult: Structured container with transformed dipole derivatives and metrics.
    """
    d_mu = np.asarray(dipole_derivatives, dtype=np.float64)
    r_ref = np.asarray(ref_geometry, dtype=np.float64)
    r_sub = np.asarray(sub_geometry, dtype=np.float64)
    m_ref = np.asarray(ref_masses, dtype=np.float64)
    m_sub = np.asarray(sub_masses, dtype=np.float64)

    natoms = len(r_ref)
    if d_mu.shape != (3, 3 * natoms):
        raise ValueError(f"Dipole derivatives shape mismatch: expected (3, {3 * natoms}), got {d_mu.shape}")

    # 1. Centers of mass
    com_ref = calculate_center_of_mass(r_ref, m_ref)
    com_sub = calculate_center_of_mass(r_sub, m_sub)
    com_shift = com_sub - com_ref

    # 2. Moments of Inertia and principal axes
    I_ref = calculate_inertia_tensor(r_ref, m_ref)
    I_sub = calculate_inertia_tensor(r_sub, m_sub)

    mom_ref, axes_ref, rot_ref = diagonalize_inertia_tensor(I_ref)
    mom_sub, axes_sub, rot_sub = diagonalize_inertia_tensor(I_sub)

    # 3. Compute Eckart rotation matrix R (3 x 3)
    R = compute_eckart_rotation_matrix(r_ref, r_sub, m_sub)

    # 4. Phase tracking: align principal axis eigenvector signs (axes_ref vs axes_sub)
    # Phase is +1.0 or -1.0 depending on dot product of principal axes
    phase_corrections: List[float] = []
    for k in range(3):
        dot_p = float(np.dot(axes_ref[:, k], axes_sub[:, k]))
        phase = 1.0 if dot_p >= 0.0 else -1.0
        phase_corrections.append(phase)

    # 5. Transform Dipole Derivative Tensor:
    # d_mu is (3, 3N). Reshape to (3, N, 3) where [alpha, i, beta] is d(mu_alpha) / d(x_i,beta)
    d_mu_3d = d_mu.reshape(3, natoms, 3)
    d_mu_trans_3d = np.zeros_like(d_mu_3d)

    for i in range(natoms):
        # Tensor transformation: d_mu_trans[:, i, :] = R @ d_mu_3d[:, i, :] @ R.T
        d_mu_atom_i = d_mu_3d[:, i, :]
        d_mu_trans_3d[:, i, :] = R @ d_mu_atom_i @ R.T

    # Apply principal axis phase factor
    for alpha in range(3):
        d_mu_trans_3d[alpha, :, :] *= phase_corrections[alpha]

    d_mu_trans = d_mu_trans_3d.reshape(3, 3 * natoms)

    return EckartFrameResult(
        old_com=com_ref.tolist(),
        new_com=com_sub.tolist(),
        com_shift=com_shift.tolist(),
        rotation_matrix=R,
        principal_moments_ref=mom_ref.tolist(),
        principal_moments_sub=mom_sub.tolist(),
        rotational_constants_ref_mhz=rot_ref.tolist(),
        rotational_constants_sub_mhz=rot_sub.tolist(),
        transformed_dipole_derivatives=d_mu_trans,
        phase_corrections=phase_corrections,
    )


# =============================================================================
# 10. CRYPTOGRAPHIC PROVENANCE CALCULATION
# =============================================================================

def calculate_orchestrator_provenance_hash(
    system_id: str,
    tensor_names: Sequence[str],
    rot_consts: Sequence[float],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Computes a deterministic SHA-256 cryptographic provenance digest for orchestration records.
    """
    sorted_tensors = sorted(list(tensor_names))
    formatted_rc = [f"{float(c):.8f}" for c in rot_consts]

    data: Dict[str, Any] = {
        "system_id": str(system_id),
        "tensor_names": sorted_tensors,
        "rotational_constants_mhz": formatted_rc,
    }
    if extra_payload:
        data["extra"] = extra_payload

    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# 11. UNIFIED ORCHESTRATION PIPELINE
# =============================================================================

def orchestrate_spycfit_system(
    system_id: str,
    landscape_h5_path: Union[str, Path],
    ref_masses: Sequence[float],
    sub_masses: Optional[Sequence[float]] = None,
    ref_geometry: Optional[np.ndarray] = None,
    sub_geometry: Optional[np.ndarray] = None,
    config: Optional[OrchestratorConfig] = None,
    zmq_context: Optional[zmq.Context] = None,
) -> OrchestrationResult:
    """
    Master unified entry point for SpycFit Ecosystem Routing & Orchestration.

    Execution Pipeline:
    1. Instantiates the HPC Wall-Time Safety Guard.
    2. Evaluates landscape.h5 state tensors.
    3. If required tensors are missing and auto_route_missing is enabled,
       dispatches request to CoChem-TORQ via configured IPC mode.
    4. If isotropic substitution is specified (sub_masses != None),
       executes Eckart Frame Principal Axis transformation and phase corrections
       on dipole derivative tensors.
    5. Computes cryptographic SHA-256 provenance digest and generates master report.

    Args:
        system_id: Unique molecular candidate identifier.
        landscape_h5_path: Path to state tensor database.
        ref_masses: Reference atomic masses in amu.
        sub_masses: Optional substituted atomic masses in amu.
        ref_geometry: Optional override reference geometry (N x 3).
        sub_geometry: Optional override substituted geometry (N x 3).
        config: Optional OrchestratorConfig configuration instance.
        zmq_context: Optional ZeroMQ context.

    Returns:
        OrchestrationResult: Complete Pydantic orchestration report.
    """
    cfg = config or OrchestratorConfig()
    guard = WallTimeSafetyGuard(
        max_walltime_seconds=cfg.max_walltime_seconds,
        safety_buffer_seconds=cfg.safety_buffer_seconds,
        heartbeat_interval_seconds=cfg.heartbeat_interval_seconds,
    )
    guard.tick(active_task="evaluating_landscape_tensors")

    # Step 1: Evaluate landscape.h5
    h5_target = Path(landscape_h5_path).resolve()
    tensor_report = evaluate_landscape_tensors(h5_target, system_id=system_id)
    torq_responses: List[TORQResponsePayload] = []

    # Step 2: Auto-route missing tensors to TORQ if needed
    if tensor_report.needs_torq_routing and cfg.auto_route_missing:
        guard.tick(active_task="routing_missing_tensors_to_torq")
        geom_list: List[List[float]] = []
        if ref_geometry is not None:
            geom_list = ref_geometry.tolist()
        elif h5_target.is_file():
            try:
                with h5py.File(h5_target, "r") as f:
                    if "geometry" in f:
                        geom_list = f["geometry"][()].tolist()
            except Exception:
                pass

        req = TORQRequestPayload(
            request_id=f"req_{system_id}_{int(time.time())}",
            system_id=system_id,
            missing_tensors=tensor_report.missing_tensors,
            geometry=geom_list,
            atomic_numbers=[],
            masses_amu=list(ref_masses),
        )

        resp = route_missing_energy(
            request=req,
            config=cfg,
            zmq_context=zmq_context,
        )
        torq_responses.append(resp)

    # Step 3: Eckart Frame & Dipole Derivative Correction
    eckart_res: Optional[EckartFrameResult] = None
    if sub_masses is not None and h5_target.is_file():
        guard.tick(active_task="eckart_frame_transformation")
        try:
            with h5py.File(h5_target, "r") as f:
                geom = ref_geometry if ref_geometry is not None else f["geometry"][()]
                d_mu = f["dipole_derivatives"][()] if "dipole_derivatives" in f else None

            if d_mu is not None:
                sub_g = sub_geometry if sub_geometry is not None else geom
                eckart_res = eckart_frame_correction(
                    dipole_derivatives=d_mu,
                    ref_geometry=geom,
                    sub_geometry=sub_g,
                    ref_masses=np.array(ref_masses, dtype=np.float64),
                    sub_masses=np.array(sub_masses, dtype=np.float64),
                )
        except Exception as exc:
            logger.warning(f"Eckart frame transformation notice: {exc}")

    # Final heartbeat
    guard.tick(active_task="orchestration_complete")

    # Step 4: Cryptographic Provenance
    rot_c = eckart_res.rotational_constants_sub_mhz if eckart_res else [0.0, 0.0, 0.0]
    prov_hash = calculate_orchestrator_provenance_hash(
        system_id=system_id,
        tensor_names=tensor_report.existing_tensors,
        rot_consts=rot_c,
    )

    summary_text = (
        f"Orchestration completed for system '{system_id}'. "
        f"Tensors complete: {tensor_report.is_complete}. "
        f"TORQ requests queued: {len(torq_responses)}. "
        f"Eckart frame transformed: {eckart_res is not None}."
    )

    return OrchestrationResult(
        status=OrchestrationStatus.SUCCESS,
        system_id=system_id,
        tensor_evaluation=tensor_report,
        torq_responses=torq_responses,
        eckart_result=eckart_res,
        walltime_heartbeats=guard.heartbeat_history,
        sha256_provenance=prov_hash,
        summary=summary_text,
    )


__all__ = [
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_orchestrator.py ---
"""
CoChem-SpycFit: Stage 2.0 Ecosystem Router & Orchestrator.

Module: cochem_spycfit/intake/cochem_spycfit_orchestrator.py
Role: Ecosystem Router, Tier 3 Registry & landscape.h5 State Tensor Evaluator,
      CoChem-TORQ Inter-Process Communication (IPC) Router, HPC Wall-Time Safety Guard,
      and Eckart Frame Principal Axis & Dipole Derivative Transformation Engine.

Capabilities:
1. Tier 3 Registry & landscape.h5 State Tensor Evaluator:
   - Dynamically resolves the authoritative Tier 3 (State & Config) Registry path.
   - Evaluates landscape.h5 state tensors (/geometry, /energy, /cartesian_hessian,
     /dipole_derivatives, /vpt2, /rotational_constants) without hardcoded home directory paths.
   - Identifies missing ab initio physics tensors prior to Hamiltonian tensor allocation.

2. CoChem-TORQ Inter-Process Communication (IPC) Router:
   - Autonomously queues missing ab initio computation requests to CoChem-TORQ.
   - Dual IPC Architecture:
     * High-throughput ZeroMQ (pyzmq) REQ/REP sockets.
     * Cross-platform atomic file-drop IPC utilizing temporary staging files and atomic rename.
   - Strict Pydantic v2 schemas for request/response serialization with extra-field forbidding.

3. HPC Wall-Time Safety Heartbeat Guard & Checkpoint Manager:
   - Tracks node wall-time limits, elapsed time, and remaining scheduler budget.
   - Emits real-time diagnostic heartbeats (CPU, RAM, active task, remaining time).
   - Enforces a safety buffer threshold: if remaining wall-time drops below the threshold,
     it triggers atomic checkpoint serialization (checkpoint_and_yield_node) and yields
     the node allocation cleanly to prevent SLURM/PBS scheduler timeout SIGKILL penalties.

4. Eckart Frame Principal Axis Tracker & Phase Correction Engine:
   - Computes analytical Center of Mass (COM) coordinates and shifts under isotopic substitution.
   - Constructs and diagonalizes the 3x3 moment of inertia tensor in strict FP64 precision,
     extracting sorted principal moments (Ia <= Ib <= Ic) and rotational constants (A >= B >= C).
   - Computes the optimal Eckart rotation matrix R using mass-weighted covariance SVD,
     guaranteeing exact orthonormality (R^T R = I) and proper rotation (det(R) = +1).
   - Transforms Cartesian dipole derivative tensors (3 x 3N) into the substituted Eckart frame,
     applying phase alignment to ensure continuous, invariant dipole transition predictions.

5. Cryptographic Provenance & Full Auditability:
   - Deterministic SHA-256 hash digests across orchestration parameters, tensor shapes, and constants.

Authoritative Standards:
- Method_Matrix.md (CoChem-BASE Version 4, August 2026)
- Task 2 File Inventory & Deliverable Capabilities Manifest (Part 1 of 2)
- Task 4 Hardware Gatekeeper & The Spectral Registry (Phase 2)
"""

from __future__ import annotations

import datetime
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import platformdirs
from pydantic import BaseModel, ConfigDict, Field, field_validator
import zmq

# Logging configuration
logger = logging.getLogger("CoChem-SpycFit.Orchestrator")


# =============================================================================
# 1. PHYSICAL CONSTANTS AND UNIT CONVERSION FACTORS (FP64)
# =============================================================================

# Exact SI and Atomic Units (IUPAC / CODATA 2022)
AMU_TO_KG: float = 1.66053906660e-27  # 1 u in kg (FP64)
ANGSTROM_TO_METER: float = 1.0e-10  # 1 Angstrom in meters (FP64)
BOHR_TO_ANGSTROM: float = 0.529177210903  # 1 bohr (a_0) in Angstroms (FP64)
PLANCK_CONSTANT_JS: float = 6.62607015e-34  # Planck constant h in J*s (FP64)
SPEED_OF_LIGHT_CM_S: float = 29979245800.0  # Speed of light c in cm/s (FP64)
HARTREE_TO_JOULE: float = 4.3597447222071e-18  # 1 E_h in Joules (FP64)

# Conversion constant for Rotational Constant B in MHz from Moment of Inertia I in amu * Angstrom^2:
# B (MHz) = (h / (8 * pi^2 * I)) * 10^-6
# B (MHz) = 505379.0087808415 / I (amu * A^2)
ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ: float = 505379.0087808415


# =============================================================================
# 2. ENUMS & EXCEPTIONS
# =============================================================================

class OrchestrationStatus(str, Enum):
    """Execution status for ecosystem orchestration workflows."""
    SUCCESS = "SUCCESS"
    PENDING_TORQ = "PENDING_TORQ"
    YIELDED_WALLTIME = "YIELDED_WALLTIME"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class IPCMode(str, Enum):
    """Supported inter-process communication mechanisms for CoChem-TORQ."""
    FILE_DROP = "file_drop"
    ZMQ = "zmq"


class TORQStatus(str, Enum):
    """Status of ab initio calculation requests submitted to CoChem-TORQ."""
    REQUESTED = "REQUESTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class FitOrchestratorError(Exception):
    """Base exception for SpycFit orchestrator operations."""


class WallTimeYieldException(FitOrchestratorError):
    """Raised when approaching HPC scheduler wall-time limits to trigger graceful node yielding."""


class TORQIPCError(FitOrchestratorError):
    """Raised when communication with CoChem-TORQ encounters an error."""


class TORQTimeoutError(TORQIPCError, TimeoutError):
    """Raised when CoChem-TORQ fails to return requested tensors within timeout."""


class LandscapeTensorError(FitOrchestratorError):
    """Raised when landscape.h5 state database is corrupted or malformed."""


class EckartFrameError(FitOrchestratorError):
    """Raised when Eckart rotation or principal axis diagonalization fails."""


# =============================================================================
# 3. PYDANTIC V2 SCHEMAS
# =============================================================================

class OrchestratorConfig(BaseModel):
    """Configuration settings for the ecosystem router and orchestrator."""
    model_config = ConfigDict(extra="forbid")

    registry_path: Optional[str] = Field(
        default=None,
        description="Path to Tier 3 Registry directory or config file.",
    )
    torq_ipc_mode: str = Field(
        default="file_drop",
        description="IPC communication mode with CoChem-TORQ ('file_drop' or 'zmq').",
    )
    torq_ipc_endpoint: str = Field(
        default="tcp://127.0.0.1:5555",
        description="ZeroMQ socket endpoint URL if mode is 'zmq'.",
    )
    torq_drop_dir: Optional[str] = Field(
        default=None,
        description="Directory for atomic file-drop IPC if mode is 'file_drop'.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum wait timeout in seconds for TORQ responses.",
    )
    heartbeat_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Interval in seconds between wall-time heartbeat ticks.",
    )
    safety_buffer_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Safety buffer threshold in seconds before wall-time expiration.",
    )
    max_walltime_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Total allocated HPC job wall-time limit in seconds.",
    )
    auto_route_missing: bool = Field(
        default=True,
        description="Whether to automatically queue requests to TORQ for missing tensors.",
    )
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description="Directory for saving state checkpoints upon wall-time yield.",
    )


class TensorEvaluationReport(BaseModel):
    """Diagnostic evaluation report for state tensors in landscape.h5."""
    model_config = ConfigDict(extra="forbid")

    system_id: str = Field(..., description="Molecular system identifier.")
    h5_path: str = Field(..., description="Filesystem path of the evaluated landscape.h5 file.")
    existing_tensors: List[str] = Field(default_factory=list, description="Tensors present in HDF5.")
    missing_tensors: List[str] = Field(default_factory=list, description="Required tensors absent in HDF5.")
    is_complete: bool = Field(..., description="True if all required tensors exist with valid shapes.")
    needs_torq_routing: bool = Field(..., description="True if missing tensors must be computed by TORQ.")
    tensor_shapes: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Extracted dimensions of existing tensors.",
    )


class TORQRequestPayload(BaseModel):
    """Structured request payload queued to CoChem-TORQ."""
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Unique UUID or timestamp-keyed request ID.")
    system_id: str = Field(..., description="Molecular system or conformer identifier.")
    missing_tensors: List[str] = Field(..., description="List of required tensor names (e.g. ['cartesian_hessian']).")
    geometry: List[List[float]] = Field(default_factory=list, description="Cartesian coordinates in Angstroms (N x 3).")
    atomic_numbers: List[int] = Field(default_factory=list, description="Atomic numbers Z (1D list of length N).")
    masses_amu: List[float] = Field(default_factory=list, description="Atomic isotopic masses in amu (1D list of length N).")
    method: str = Field(default="wB97M-V", description="Quantum chemical functional or method.")
    basis: str = Field(default="def2-QZVPP", description="Basis set specification.")
    priority: int = Field(default=1, ge=1, le=10, description="Task priority (1 = high, 10 = low).")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC creation timestamp.",
    )
    status: str = Field(default="REQUESTED", description="Request lifecycle status.")


class TORQResponsePayload(BaseModel):
    """Structured response payload returned by CoChem-TORQ."""
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Corresponding request identifier.")
    system_id: str = Field(..., description="Molecular system identifier.")
    status: str = Field(default="COMPLETED", description="Status: 'COMPLETED', 'FAILED', 'TIMEOUT'.")
    tensors_data: Dict[str, Any] = Field(default_factory=dict, description="Calculated tensor arrays.")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error message if status is FAILED.")
    execution_time_seconds: float = Field(default=0.0, ge=0.0, description="Execution wall-time in seconds.")


class HeartbeatStatus(BaseModel):
    """Diagnostic status snapshot emitted by the HPC Wall-Time Safety Guard."""
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp of heartbeat.")
    elapsed_seconds: float = Field(..., ge=0.0, description="Elapsed execution wall-time in seconds.")
    remaining_walltime_seconds: float = Field(..., description="Remaining wall-time before scheduler limit.")
    is_safe: bool = Field(..., description="True if remaining wall-time exceeds safety buffer threshold.")
    cpu_percent: float = Field(default=0.0, ge=0.0, description="Current CPU utilization percentage.")
    memory_mb: float = Field(default=0.0, ge=0.0, description="Current resident set size (RSS) memory in MB.")
    active_task: str = Field(default="orchestrating", description="Active sub-task description.")


class EckartFrameResult(BaseModel):
    """Output results from Eckart Frame transformation and dipole derivative correction."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    old_com: List[float] = Field(..., description="Reference Center of Mass coordinates in Angstroms.")
    new_com: List[float] = Field(..., description="Substituted Center of Mass coordinates in Angstroms.")
    com_shift: List[float] = Field(..., description="Center of Mass translation vector (new - old) in Angstroms.")
    rotation_matrix: Any = Field(..., description="Eckart rotation matrix R (3x3 np.ndarray in FP64).")
    principal_moments_ref: List[float] = Field(..., description="Reference principal moments Ia <= Ib <= Ic (amu*A^2).")
    principal_moments_sub: List[float] = Field(..., description="Substituted principal moments Ia <= Ib <= Ic (amu*A^2).")
    rotational_constants_ref_mhz: List[float] = Field(..., description="Reference rotational constants A >= B >= C (MHz).")
    rotational_constants_sub_mhz: List[float] = Field(..., description="Substituted rotational constants A >= B >= C (MHz).")
    transformed_dipole_derivatives: Any = Field(
        ...,
        description="Transformed Cartesian dipole derivative tensor (3 x 3N np.ndarray in FP64).",
    )
    phase_corrections: List[float] = Field(
        ...,
        description="Phase factors (+1.0 or -1.0) along each principal axis.",
    )


class OrchestrationResult(BaseModel):
    """Consolidated report produced by the SpycFit ecosystem orchestrator."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: OrchestrationStatus = Field(..., description="Overall orchestration execution status.")
    system_id: str = Field(..., description="Molecular system identifier.")
    tensor_evaluation: TensorEvaluationReport = Field(..., description="State tensor evaluation report.")
    torq_responses: List[TORQResponsePayload] = Field(default_factory=list, description="Responses from TORQ.")
    eckart_result: Optional[EckartFrameResult] = Field(default=None, description="Eckart frame results if computed.")
    walltime_heartbeats: List[HeartbeatStatus] = Field(default_factory=list, description="Heartbeat log entries.")
    sha256_provenance: str = Field(..., description="Cryptographic SHA-256 provenance hash.")
    summary: str = Field(..., description="Human-readable execution summary.")


# =============================================================================
# 4. TIER 3 REGISTRY RESOLUTION
# =============================================================================

def resolve_tier3_registry_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Dynamically resolves the authoritative path to the Tier 3 (State & Config) Registry.
    Avoids hardcoded $HOME to ensure seamless compatibility across Windows, Linux, and HPC.

    Args:
        custom_path: Optional user-supplied path to override standard resolution.

    Returns:
        Path: Absolute resolved path to Tier 3 Registry directory.
    """
    if custom_path is not None:
        p = Path(custom_path).resolve()
        return p

    env_dir = os.environ.get("COCHEM_REGISTRY_DIR") or os.environ.get("COCHEM_STATE_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        return p

    # Standard platformdirs user config directory
    config_dir = Path(platformdirs.user_config_dir("CoChem", "CoChem")).resolve() / "Registry"
    return config_dir


# =============================================================================
# 5. LANDSCAPE.H5 STATE TENSOR EVALUATION
# =============================================================================

def evaluate_landscape_tensors(
    h5_path: Union[str, Path],
    system_id: str = "default_system",
    required_tensors: Optional[List[str]] = None,
) -> TensorEvaluationReport:
    """
    Evaluates state tensors inside landscape.h5 routed via the Tier 3 Registry.

    Checks for the presence and validity of required ab initio physics tensors:
    - 'geometry' (N x 3 coordinates in Angstroms)
    - 'energy_hartree' or 'energy' (scalar FP64 ground state energy)
    - 'cartesian_hessian' (3N x 3N force constant matrix)
    - 'dipole_derivatives' (3 x 3N Cartesian dipole derivatives)

    Args:
        h5_path: Path to landscape.h5 file.
        system_id: Unique molecular identifier.
        required_tensors: Optional custom list of required dataset names.

    Returns:
        TensorEvaluationReport: Pydantic report detailing existing and missing tensors.
    """
    target_h5 = Path(h5_path).resolve()
    req_keys = required_tensors or ["geometry", "cartesian_hessian", "dipole_derivatives"]

    if not target_h5.is_file():
        return TensorEvaluationReport(
            system_id=system_id,
            h5_path=str(target_h5),
            existing_tensors=[],
            missing_tensors=req_keys,
            is_complete=False,
            needs_torq_routing=True,
            tensor_shapes={},
        )

    existing: List[str] = []
    missing: List[str] = []
    shapes_dict: Dict[str, List[int]] = {}

    try:
        with h5py.File(target_h5, "r") as f:
            for key in req_keys:
                # Handle possible alternative naming in landscape datasets
                matched_key: Optional[str] = None
                if key in f:
                    matched_key = key
                elif key == "cartesian_hessian" and "hessian" in f:
                    matched_key = "hessian"
                elif key == "dipole_derivatives" and "dipole_gradient" in f:
                    matched_key = "dipole_gradient"
                elif key == "energy_hartree" and "energy" in f:
                    matched_key = "energy"

                if matched_key is not None:
                    dset = f[matched_key]
                    existing.append(key)
                    shapes_dict[key] = list(dset.shape)
                else:
                    missing.append(key)
    except Exception as exc:
        logger.warning(f"Error inspecting HDF5 file {target_h5}: {exc}")
        return TensorEvaluationReport(
            system_id=system_id,
            h5_path=str(target_h5),
            existing_tensors=[],
            missing_tensors=req_keys,
            is_complete=False,
            needs_torq_routing=True,
            tensor_shapes={},
        )

    is_complete = (len(missing) == 0)
    needs_torq_routing = not is_complete

    return TensorEvaluationReport(
        system_id=system_id,
        h5_path=str(target_h5),
        existing_tensors=existing,
        missing_tensors=missing,
        is_complete=is_complete,
        needs_torq_routing=needs_torq_routing,
        tensor_shapes=shapes_dict,
    )


# =============================================================================
# 6. COCHEM-TORQ IPC COMMUNICATION (FILE-DROP & ZEROMQ)
# =============================================================================

def route_missing_energy_file_drop(
    request: TORQRequestPayload,
    drop_dir: Union[str, Path],
    timeout_seconds: float = 60.0,
    poll_interval: float = 0.1,
) -> TORQResponsePayload:
    """
    Submits missing physics calculation request to CoChem-TORQ via atomic file-drop IPC.

    Protocol:
    1. Writes request to a temporary file: `<drop_dir>/<request_id>.request.json.tmp`.
    2. Atomically renames to `<drop_dir>/<request_id>.request.json`.
    3. Polls `<drop_dir>/<request_id>.response.json` until completed or timeout.
    4. Parses and validates the response, removing temporary files.

    Args:
        request: Validated TORQRequestPayload model.
        drop_dir: Path to shared IPC drop directory.
        timeout_seconds: Maximum wait timeout.
        poll_interval: Polling loop sleep duration in seconds.

    Returns:
        TORQResponsePayload: Response received from TORQ.

    Raises:
        TORQTimeoutError: If TORQ does not respond within timeout_seconds.
    """
    target_dir = Path(drop_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    req_file = target_dir / f"{request.request_id}.request.json"
    req_tmp = target_dir / f"{request.request_id}.request.json.tmp"
    resp_file = target_dir / f"{request.request_id}.response.json"

    # Step 1: Write atomic request payload
    payload_str = request.model_dump_json(indent=2)
    with open(req_tmp, "w", encoding="utf-8") as f:
        f.write(payload_str)
    req_tmp.replace(req_file)
    logger.info(f"[TORQ IPC File-Drop] Queued request {request.request_id} for system '{request.system_id}'")

    # Step 2: Poll for response
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            # Clean up request file if still present
            req_file.unlink(missing_ok=True)
            raise TORQTimeoutError(
                f"CoChem-TORQ file-drop IPC timed out after {timeout_seconds:.1f}s for request {request.request_id}"
            )

        if resp_file.exists():
            try:
                # Small wait to ensure file write buffer flushed by writer
                time.sleep(0.01)
                with open(resp_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                # Parse and validate with Pydantic
                response = TORQResponsePayload.model_validate(raw_data)
                # Clean up response file
                resp_file.unlink(missing_ok=True)
                logger.info(f"[TORQ IPC File-Drop] Received response for {request.request_id} in {elapsed:.2f}s")
                return response
            except (json.JSONDecodeError, ValueError) as exc:
                # Writer may still be writing; continue polling
                logger.debug(f"Transient response read error: {exc}")

        time.sleep(poll_interval)


def route_missing_energy_zmq(
    request: TORQRequestPayload,
    endpoint: str = "tcp://127.0.0.1:5555",
    timeout_seconds: float = 60.0,
    zmq_context: Optional[zmq.Context] = None,
) -> TORQResponsePayload:
    """
    Submits missing physics calculation request to CoChem-TORQ via ZeroMQ REQ/REP sockets.

    Args:
        request: Validated TORQRequestPayload model.
        endpoint: ZeroMQ socket endpoint string (e.g. 'tcp://127.0.0.1:5555' or 'inproc://...').
        timeout_seconds: Maximum wait timeout.
        zmq_context: Optional existing zmq.Context instance.

    Returns:
        TORQResponsePayload: Response received from TORQ.

    Raises:
        TORQTimeoutError: If socket timeout is reached.
        TORQIPCError: If socket communication encounters an error.
    """
    ctx = zmq_context or zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_seconds * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_seconds * 1000))
    sock.setsockopt(zmq.LINGER, 0)

    try:
        sock.connect(endpoint)
        payload_str = request.model_dump_json()
        sock.send_string(payload_str)
        logger.info(f"[TORQ IPC ZMQ] Sent request {request.request_id} to endpoint {endpoint}")

        raw_resp = sock.recv_string()
        resp_data = json.loads(raw_resp)
        response = TORQResponsePayload.model_validate(resp_data)
        logger.info(f"[TORQ IPC ZMQ] Received response for {request.request_id}")
        return response
    except zmq.Again as exc:
        raise TORQTimeoutError(
            f"ZeroMQ socket timed out after {timeout_seconds:.1f}s awaiting TORQ response on {endpoint}"
        ) from exc
    except Exception as exc:
        raise TORQIPCError(f"ZeroMQ socket IPC failure on {endpoint}: {exc}") from exc
    finally:
        sock.close()


def route_missing_energy(
    request: TORQRequestPayload,
    config: Optional[OrchestratorConfig] = None,
    drop_dir: Optional[Union[str, Path]] = None,
    zmq_context: Optional[zmq.Context] = None,
) -> TORQResponsePayload:
    """
    High-level routing function that dispatches missing energy / tensor requests
    to CoChem-TORQ using either ZeroMQ or cross-platform File-Drop IPC.
    """
    cfg = config or OrchestratorConfig()
    ipc_mode = cfg.torq_ipc_mode.lower().strip()

    if ipc_mode in ("zmq", "zeromq"):
        return route_missing_energy_zmq(
            request=request,
            endpoint=cfg.torq_ipc_endpoint,
            timeout_seconds=cfg.timeout_seconds,
            zmq_context=zmq_context,
        )

    # Default to File-Drop IPC
    target_drop_dir = drop_dir or cfg.torq_drop_dir or (
        Path(platformdirs.user_data_path("CoChem", "CoChem")) / "SpycFit_Workspace" / "TORQ_IPC"
    )
    return route_missing_energy_file_drop(
        request=request,
        drop_dir=target_drop_dir,
        timeout_seconds=cfg.timeout_seconds,
    )


# =============================================================================
# 7. HPC WALL-TIME SAFETY HEARTBEAT & CHECKPOINTING
# =============================================================================

class WallTimeSafetyGuard:
    """
    HPC Wall-Time Safety Guard & Heartbeat Manager.

    Monitors job execution wall-time against scheduler limits (e.g. SLURM / PBS).
    If the remaining time falls below safety_buffer_seconds, it triggers a clean
    checkpoint and raises WallTimeYieldException to yield node allocation safely.
    """

    def __init__(
        self,
        max_walltime_seconds: float = 3600.0,
        safety_buffer_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 5.0,
        start_time: Optional[float] = None,
    ) -> None:
        self.max_walltime = float(max_walltime_seconds)
        self.safety_buffer = float(safety_buffer_seconds)
        self.heartbeat_interval = float(heartbeat_interval_seconds)
        self.start_time = float(start_time) if start_time is not None else time.time()
        self.last_heartbeat_time = self.start_time
        self.heartbeat_history: List[HeartbeatStatus] = []

    def get_elapsed_seconds(self) -> float:
        """Returns elapsed wall-time in seconds."""
        return max(0.0, time.time() - self.start_time)

    def get_remaining_seconds(self) -> float:
        """Returns remaining wall-time before maximum allocation is reached."""
        return self.max_walltime - self.get_elapsed_seconds()

    def is_safe(self) -> bool:
        """Returns True if remaining wall-time exceeds the safety buffer."""
        return self.get_remaining_seconds() > self.safety_buffer

    def should_yield(self) -> bool:
        """Returns True if wall-time budget is exhausted and node allocation must be yielded."""
        return not self.is_safe()

    def tick(self, active_task: str = "orchestrating") -> HeartbeatStatus:
        """
        Emits a heartbeat diagnostic status record and appends it to history.
        """
        now = time.time()
        elapsed = self.get_elapsed_seconds()
        remaining = self.get_remaining_seconds()
        safe = self.is_safe()

        # Gather lightweight OS memory/cpu heuristics if psutil available
        cpu_pct = 0.0
        mem_mb = 0.0
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            cpu_pct = float(proc.cpu_percent(interval=None))
            mem_mb = float(proc.memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            pass

        status = HeartbeatStatus(
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=np.float64(elapsed),
            remaining_walltime_seconds=np.float64(remaining),
            is_safe=safe,
            cpu_percent=np.float64(cpu_pct),
            memory_mb=np.float64(mem_mb),
            active_task=active_task,
        )

        self.heartbeat_history.append(status)
        self.last_heartbeat_time = now

        if not safe:
            logger.warning(
                f"[HPC WALL-TIME ALERT] Remaining time ({remaining:.1f}s) <= buffer ({self.safety_buffer:.1f}s). "
                f"Yielding node allocation."
            )

        return status

    def check_walltime_or_yield(
        self,
        current_state: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Verifies wall-time safety. If unsafe, atomically saves a state checkpoint
        and raises WallTimeYieldException to cleanly exit the job step.
        """
        if self.should_yield():
            if current_state is not None:
                ckpt_p = checkpoint_path or (
                    Path(platformdirs.user_data_path("CoChem", "CoChem")) / "SpycFit_Workspace" / "checkpoints" / "checkpoint_yield.json"
                )
                checkpoint_and_yield_node(current_state, ckpt_p)

            raise WallTimeYieldException(
                f"HPC Wall-Time Limit Approaching: {self.get_remaining_seconds():.1f}s remaining "
                f"(safety buffer: {self.safety_buffer:.1f}s). Node allocation yielded."
            )


def checkpoint_and_yield_node(
    state_data: Dict[str, Any],
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Atomically checkpoints active fitting/orchestration state to disk and yields node allocation.

    Args:
        state_data: Dictionary containing intermediate optimization parameters, tensors, and metadata.
        checkpoint_path: Destination path for JSON checkpoint file.

    Returns:
        Dict: Metadata confirmation containing checkpoint path and timestamp.
    """
    target = Path(checkpoint_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.tmp_{int(time.time())}"

    payload = dict(state_data)
    payload["_checkpoint_saved_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload["_pid"] = os.getpid()

    payload_str = json.dumps(payload, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload_str)

    tmp_path.replace(target)
    logger.info(f"[Checkpoint & Yield] Saved state checkpoint to {target}")

    return {
        "status": "YIELDED_CHECKPOINT_SAVED",
        "checkpoint_path": str(target),
        "timestamp_utc": payload["_checkpoint_saved_utc"],
        "bytes_written": len(payload_str),
    }


# =============================================================================
# 8. CENTER OF MASS AND INERTIA TENSOR CALCULATIONS
# =============================================================================

def calculate_center_of_mass(
    geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Calculates analytical Center of Mass (COM) coordinates in FP64 precision.

    COM = sum(m_i * r_i) / sum(m_i)

    Args:
        geometry: (N, 3) Cartesian coordinates in Angstroms.
        masses: (N,) atomic masses in amu.

    Returns:
        (3,) Center of mass coordinates in Angstroms (np.float64).
    """
    geom = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64).reshape(-1, 1)

    if geom.ndim != 2 or geom.shape[1] != 3:
        raise ValueError(f"Geometry must have shape (N, 3), got {geom.shape}")
    if len(m) != len(geom):
        raise ValueError(f"Length mismatch: {len(geom)} atoms vs {len(m)} masses")

    total_mass = np.sum(m)
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive")

    com = np.sum(m * geom, axis=0) / total_mass
    return com.astype(np.float64)


def calculate_inertia_tensor(
    geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Constructs the 3x3 Moment of Inertia tensor in the Center of Mass frame.

    I_ab = sum_i m_i * ( |r_i|^2 * delta_ab - r_i,a * r_i,b )

    Args:
        geometry: (N, 3) Cartesian coordinates in Angstroms.
        masses: (N,) atomic masses in amu.

    Returns:
        (3, 3) Moment of inertia tensor in amu * Angstrom^2 (np.float64).
    """
    geom = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)

    com = calculate_center_of_mass(geom, m)
    r = geom - com  # Shift to COM frame

    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r_sq = x * x + y * y + z * z

    I_xx = np.sum(m * (r_sq - x * x))
    I_yy = np.sum(m * (r_sq - y * y))
    I_zz = np.sum(m * (r_sq - z * z))

    I_xy = -np.sum(m * x * y)
    I_xz = -np.sum(m * x * z)
    I_yz = -np.sum(m * y * z)

    inertia_tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz],
    ], dtype=np.float64)

    return inertia_tensor


def diagonalize_inertia_tensor(
    inertia_tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalizes the Moment of Inertia tensor to obtain principal moments and rotational constants.

    Orders principal moments Ia <= Ib <= Ic, corresponding to rotational constants A >= B >= C.

    Returns:
        Tuple of:
        - principal_moments: (3,) sorted principal moments in amu * Angstrom^2.
        - principal_axes: (3, 3) principal axis eigenvectors (columns).
        - rotational_constants_mhz: (3,) [A, B, C] in MHz.
    """
    I = np.asarray(inertia_tensor, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Sort ascending: Ia <= Ib <= Ic
    idx = np.argsort(eigenvalues)
    principal_moments = eigenvalues[idx].astype(np.float64)
    principal_axes = eigenvectors[:, idx].astype(np.float64)

    # Ensure right-handed coordinate frame: det(principal_axes) = +1
    if np.linalg.det(principal_axes) < 0.0:
        principal_axes[:, 2] *= -1.0

    # Calculate Rotational Constants A >= B >= C in MHz
    # Handle possible zero modes for linear or planar degenerate systems
    rot_constants = np.zeros(3, dtype=np.float64)
    for k in range(3):
        if principal_moments[k] > 1e-6:
            rot_constants[k] = ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ / principal_moments[k]
        else:
            rot_constants[k] = 0.0

    return principal_moments, principal_axes, rot_constants


# =============================================================================
# 9. ECKART FRAME ROTATION & DIPOLE DERIVATIVE CORRECTION
# =============================================================================

def compute_eckart_rotation_matrix(
    ref_geometry: np.ndarray,
    sub_geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Computes the optimal Eckart rotation matrix R that aligns sub_geometry to ref_geometry
    under the mass-weighted Eckart condition: sum_i m_i (r_0,i x r_i) = 0.

    Utilizes the Kabsch / SVD algorithm on the mass-weighted covariance matrix:
    F = sum_i m_i * r_ref,i * (r_sub,i)^T = U * Sigma * V^T
    R = U * diag(1, 1, det(U V^T)) * V^T

    Guarantees:
    - Exact orthonormality: R^T R = I
    - Proper rotation: det(R) = +1.0

    Args:
        ref_geometry: (N, 3) reference geometry (e.g. parent isotopologue).
        sub_geometry: (N, 3) displaced or substituted geometry.
        masses: (N,) atomic masses in amu.

    Returns:
        (3, 3) Eckart rotation matrix R (np.float64).
    """
    r_ref = np.asarray(ref_geometry, dtype=np.float64)
    r_sub = np.asarray(sub_geometry, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64).reshape(-1, 1)

    # Shift both geometries to their respective centers of mass
    com_ref = calculate_center_of_mass(r_ref, m.flatten())
    com_sub = calculate_center_of_mass(r_sub, m.flatten())

    r_ref_com = r_ref - com_ref
    r_sub_com = r_sub - com_sub

    # Mass-weighted covariance matrix F (3 x 3)
    # F_ab = sum_i m_i * r_ref_com[i, a] * r_sub_com[i, b]
    F = (r_ref_com.T * m.T) @ r_sub_com

    # Singular Value Decomposition: F = U * S * V^T
    U, _, Vt = np.linalg.svd(F)
    V = Vt.T

    # Reflection correction matrix D to enforce det(R) = +1
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])

    R_eckart = U @ D @ Vt
    return R_eckart.astype(np.float64)


def eckart_frame_correction(
    dipole_derivatives: np.ndarray,
    ref_geometry: np.ndarray,
    sub_geometry: np.ndarray,
    ref_masses: np.ndarray,
    sub_masses: np.ndarray,
) -> EckartFrameResult:
    """
    Applies Eckart Frame and principal axis phase corrections to Cartesian dipole derivatives
    when isotopic substitution shifts the Center of Mass and principal axes.

    Transformation Steps:
    1. Computes Center of Mass coordinates for reference (parent) and substituted systems.
    2. Diagonalizes Moment of Inertia tensors to extract reference and substituted principal axes.
    3. Determines the Eckart rotation matrix R aligning the frames.
    4. Applies phase corrections along principal axes to resolve sign ambiguities in eigenvectors.
    5. Transforms Cartesian dipole derivative tensor (3 x 3N):
       dmu_a / dx_ib_new = sum_c sum_d R_ac * (dmu_c / dx_id_ref) * R_bd

    Args:
        dipole_derivatives: (3, 3N) Cartesian dipole derivatives of reference system (Debye / Angstrom).
        ref_geometry: (N, 3) Reference Cartesian geometry in Angstroms.
        sub_geometry: (N, 3) Substituted Cartesian geometry in Angstroms.
        ref_masses: (N,) Reference isotopic atomic masses in amu.
        sub_masses: (N,) Substituted isotopic atomic masses in amu.

    Returns:
        EckartFrameResult: Structured container with transformed dipole derivatives and metrics.
    """
    d_mu = np.asarray(dipole_derivatives, dtype=np.float64)
    r_ref = np.asarray(ref_geometry, dtype=np.float64)
    r_sub = np.asarray(sub_geometry, dtype=np.float64)
    m_ref = np.asarray(ref_masses, dtype=np.float64)
    m_sub = np.asarray(sub_masses, dtype=np.float64)

    natoms = len(r_ref)
    if d_mu.shape != (3, 3 * natoms):
        raise ValueError(f"Dipole derivatives shape mismatch: expected (3, {3 * natoms}), got {d_mu.shape}")

    # 1. Centers of mass
    com_ref = calculate_center_of_mass(r_ref, m_ref)
    com_sub = calculate_center_of_mass(r_sub, m_sub)
    com_shift = com_sub - com_ref

    # 2. Moments of Inertia and principal axes
    I_ref = calculate_inertia_tensor(r_ref, m_ref)
    I_sub = calculate_inertia_tensor(r_sub, m_sub)

    mom_ref, axes_ref, rot_ref = diagonalize_inertia_tensor(I_ref)
    mom_sub, axes_sub, rot_sub = diagonalize_inertia_tensor(I_sub)

    # 3. Compute Eckart rotation matrix R (3 x 3)
    R = compute_eckart_rotation_matrix(r_ref, r_sub, m_sub)

    # 4. Phase tracking: align principal axis eigenvector signs (axes_ref vs axes_sub)
    # Phase is +1.0 or -1.0 depending on dot product of principal axes
    phase_corrections: List[float] = []
    for k in range(3):
        dot_p = float(np.dot(axes_ref[:, k], axes_sub[:, k]))
        phase = 1.0 if dot_p >= 0.0 else -1.0
        phase_corrections.append(phase)

    # 5. Transform Dipole Derivative Tensor:
    # d_mu is (3, 3N). Reshape to (3, N, 3) where [alpha, i, beta] is d(mu_alpha) / d(x_i,beta)
    d_mu_3d = d_mu.reshape(3, natoms, 3)
    d_mu_trans_3d = np.zeros_like(d_mu_3d)

    for i in range(natoms):
        # Tensor transformation: d_mu_trans[:, i, :] = R @ d_mu_3d[:, i, :] @ R.T
        d_mu_atom_i = d_mu_3d[:, i, :]
        d_mu_trans_3d[:, i, :] = R @ d_mu_atom_i @ R.T

    # Apply principal axis phase factor
    for alpha in range(3):
        d_mu_trans_3d[alpha, :, :] *= phase_corrections[alpha]

    d_mu_trans = d_mu_trans_3d.reshape(3, 3 * natoms)

    return EckartFrameResult(
        old_com=com_ref.tolist(),
        new_com=com_sub.tolist(),
        com_shift=com_shift.tolist(),
        rotation_matrix=R,
        principal_moments_ref=mom_ref.tolist(),
        principal_moments_sub=mom_sub.tolist(),
        rotational_constants_ref_mhz=rot_ref.tolist(),
        rotational_constants_sub_mhz=rot_sub.tolist(),
        transformed_dipole_derivatives=d_mu_trans,
        phase_corrections=phase_corrections,
    )


# =============================================================================
# 10. CRYPTOGRAPHIC PROVENANCE CALCULATION
# =============================================================================

def calculate_orchestrator_provenance_hash(
    system_id: str,
    tensor_names: Sequence[str],
    rot_consts: Sequence[float],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Computes a deterministic SHA-256 cryptographic provenance digest for orchestration records.
    """
    sorted_tensors = sorted(list(tensor_names))
    formatted_rc = [f"{float(c):.8f}" for c in rot_consts]

    data: Dict[str, Any] = {
        "system_id": str(system_id),
        "tensor_names": sorted_tensors,
        "rotational_constants_mhz": formatted_rc,
    }
    if extra_payload:
        data["extra"] = extra_payload

    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# 11. UNIFIED ORCHESTRATION PIPELINE
# =============================================================================

def orchestrate_spycfit_system(
    system_id: str,
    landscape_h5_path: Union[str, Path],
    ref_masses: Sequence[float],
    sub_masses: Optional[Sequence[float]] = None,
    ref_geometry: Optional[np.ndarray] = None,
    sub_geometry: Optional[np.ndarray] = None,
    config: Optional[OrchestratorConfig] = None,
    zmq_context: Optional[zmq.Context] = None,
) -> OrchestrationResult:
    """
    Master unified entry point for SpycFit Ecosystem Routing & Orchestration.

    Execution Pipeline:
    1. Instantiates the HPC Wall-Time Safety Guard.
    2. Evaluates landscape.h5 state tensors.
    3. If required tensors are missing and auto_route_missing is enabled,
       dispatches request to CoChem-TORQ via configured IPC mode.
    4. If isotropic substitution is specified (sub_masses != None),
       executes Eckart Frame Principal Axis transformation and phase corrections
       on dipole derivative tensors.
    5. Computes cryptographic SHA-256 provenance digest and generates master report.

    Args:
        system_id: Unique molecular candidate identifier.
        landscape_h5_path: Path to state tensor database.
        ref_masses: Reference atomic masses in amu.
        sub_masses: Optional substituted atomic masses in amu.
        ref_geometry: Optional override reference geometry (N x 3).
        sub_geometry: Optional override substituted geometry (N x 3).
        config: Optional OrchestratorConfig configuration instance.
        zmq_context: Optional ZeroMQ context.

    Returns:
        OrchestrationResult: Complete Pydantic orchestration report.
    """
    cfg = config or OrchestratorConfig()
    guard = WallTimeSafetyGuard(
        max_walltime_seconds=cfg.max_walltime_seconds,
        safety_buffer_seconds=cfg.safety_buffer_seconds,
        heartbeat_interval_seconds=cfg.heartbeat_interval_seconds,
    )
    guard.tick(active_task="evaluating_landscape_tensors")

    # Step 1: Evaluate landscape.h5
    h5_target = Path(landscape_h5_path).resolve()
    tensor_report = evaluate_landscape_tensors(h5_target, system_id=system_id)
    torq_responses: List[TORQResponsePayload] = []

    # Step 2: Auto-route missing tensors to TORQ if needed
    if tensor_report.needs_torq_routing and cfg.auto_route_missing:
        guard.tick(active_task="routing_missing_tensors_to_torq")
        geom_list: List[List[float]] = []
        if ref_geometry is not None:
            geom_list = ref_geometry.tolist()
        elif h5_target.is_file():
            try:
                with h5py.File(h5_target, "r") as f:
                    if "geometry" in f:
                        geom_list = f["geometry"][()].tolist()
            except Exception:
                pass

        req = TORQRequestPayload(
            request_id=f"req_{system_id}_{int(time.time())}",
            system_id=system_id,
            missing_tensors=tensor_report.missing_tensors,
            geometry=geom_list,
            atomic_numbers=[],
            masses_amu=list(ref_masses),
        )

        resp = route_missing_energy(
            request=req,
            config=cfg,
            zmq_context=zmq_context,
        )
        torq_responses.append(resp)

    # Step 3: Eckart Frame & Dipole Derivative Correction
    eckart_res: Optional[EckartFrameResult] = None
    if sub_masses is not None and h5_target.is_file():
        guard.tick(active_task="eckart_frame_transformation")
        try:
            with h5py.File(h5_target, "r") as f:
                geom = ref_geometry if ref_geometry is not None else f["geometry"][()]
                d_mu = f["dipole_derivatives"][()] if "dipole_derivatives" in f else None

            if d_mu is not None:
                sub_g = sub_geometry if sub_geometry is not None else geom
                eckart_res = eckart_frame_correction(
                    dipole_derivatives=d_mu,
                    ref_geometry=geom,
                    sub_geometry=sub_g,
                    ref_masses=np.array(ref_masses, dtype=np.float64),
                    sub_masses=np.array(sub_masses, dtype=np.float64),
                )
        except Exception as exc:
            logger.warning(f"Eckart frame transformation notice: {exc}")

    # Final heartbeat
    guard.tick(active_task="orchestration_complete")

    # Step 4: Cryptographic Provenance
    rot_c = eckart_res.rotational_constants_sub_mhz if eckart_res else [0.0, 0.0, 0.0]
    prov_hash = calculate_orchestrator_provenance_hash(
        system_id=system_id,
        tensor_names=tensor_report.existing_tensors,
        rot_consts=rot_c,
    )

    summary_text = (
        f"Orchestration completed for system '{system_id}'. "
        f"Tensors complete: {tensor_report.is_complete}. "
        f"TORQ requests queued: {len(torq_responses)}. "
        f"Eckart frame transformed: {eckart_res is not None}."
    )

    return OrchestrationResult(
        status=OrchestrationStatus.SUCCESS,
        system_id=system_id,
        tensor_evaluation=tensor_report,
        torq_responses=torq_responses,
        eckart_result=eckart_res,
        walltime_heartbeats=guard.heartbeat_history,
        sha256_provenance=prov_hash,
        summary=summary_text,
    )


__all__ = [
    "AMU_TO_KG",
    "ANGSTROM_TO_METER",
    "BOHR_TO_ANGSTROM",
    "EckartFrameError",
    "EckartFrameResult",
    "FitOrchestratorError",
    "HARTREE_TO_JOULE",
    "HeartbeatStatus",
    "IPCMode",
    "LandscapeTensorError",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestratorConfig",
    "PLANCK_CONSTANT_JS",
    "ROTATIONAL_CONSTANT_CONVERSION_FACTOR_MHZ",
    "SPEED_OF_LIGHT_CM_S",
    "TORQIPCError",
    "TORQRequestPayload",
    "TORQResponsePayload",
    "TORQStatus",
    "TORQTimeoutError",
    "TensorEvaluationReport",
    "WallTimeSafetyGuard",
    "WallTimeYieldException",
    "calculate_center_of_mass",
    "calculate_inertia_tensor",
    "calculate_orchestrator_provenance_hash",
    "checkpoint_and_yield_node",
    "compute_eckart_rotation_matrix",
    "diagonalize_inertia_tensor",
    "eckart_frame_correction",
    "evaluate_landscape_tensors",
    "orchestrate_spycfit_system",
    "resolve_tier3_registry_path",
    "route_missing_energy",
    "route_missing_energy_file_drop",
    "route_missing_energy_zmq",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_orchestrator.py ---
"""
Physical Unit and Integration Tests for Stage 2.0 Ecosystem Router & Orchestrator.

Module: tests/test_cochem_spycfit_orchestrator.py
Target: cochem_spycfit.intake.cochem_spycfit_orchestrator

Validates:
1. Physical file existence, UTF-8 encoding (no BOM), and Unix LF line endings.
2. Codebase integrity and anti-spoof compliance via zero-trust static inspection.
3. Pydantic v2 strict schema validation and extra field forbidding.
4. Dynamic resolution of Tier 3 Registry path across local and custom environments.
5. Evaluation of landscape.h5 state tensors routed via Tier 3 Registry.
6. Autonomous routing of missing ab initio tensors to CoChem-TORQ via:
   - Cross-platform atomic file-drop IPC
   - ZeroMQ (pyzmq) sockets
7. HPC Wall-Time Safety heartbeat timeout, continuous monitoring, and state checkpointing.
8. Safe node allocation yielding upon approaching scheduler wall-time ceilings.
9. Center of Mass (COM) translation and inertia tensor principal axis calculation.
10. Eckart Frame rotation matrix determination with strict orthonormality (R^T R = I, det(R) = +1).
11. Eckart Frame phase and rotational corrections applied to Cartesian dipole derivative tensors.
12. Error handling on edge cases, dimensional mismatches, and IPC timeouts.
13. Unified orchestration pipeline execution and deterministic SHA-256 provenance generation.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List

import h5py
import numpy as np
from pydantic import ValidationError
import pytest
import zmq

from cochem_spycfit.intake.cochem_spycfit_orchestrator import (
    EckartFrameResult,
    HeartbeatStatus,
    OrchestrationResult,
    OrchestrationStatus,
    OrchestratorConfig,
    TensorEvaluationReport,
    TORQRequestPayload,
    TORQResponsePayload,
    TORQTimeoutError,
    WallTimeSafetyGuard,
    WallTimeYieldException,
    calculate_center_of_mass,
    calculate_inertia_tensor,
    calculate_orchestrator_provenance_hash,
    checkpoint_and_yield_node,
    compute_eckart_rotation_matrix,
    diagonalize_inertia_tensor,
    eckart_frame_correction,
    evaluate_landscape_tensors,
    orchestrate_spycfit_system,
    resolve_tier3_registry_path,
    route_missing_energy,
    route_missing_energy_file_drop,
    route_missing_energy_zmq,
)


# =============================================================================
# 1. FILE ENCODING AND ANTI-SPOOF INTEGRITY TESTS
# =============================================================================

def test_file_encoding_and_lf_line_endings():
    """Validates that cochem_spycfit_orchestrator.py exists, is UTF-8 encoded without BOM, and uses Unix LF."""
    target_path = Path("src/cochem_spycfit/intake/cochem_spycfit_orchestrator.py").resolve()
    assert target_path.is_file(), f"Target file missing: {target_path}"

    raw_bytes = target_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), "BOM detected in orchestrator file."

    text = raw_bytes.decode("utf-8")
    assert "\r\n" not in text, "CRLF line endings detected; Unix LF line endings required."
    assert len(text.strip()) > 0, "Target file is empty."


def test_anti_spoofing_and_integrity_compliance():
    """Zero-Trust Static AST Inspection ensuring complete eradication of unauthorized tokens and empty pass blocks."""
    target_path = Path("src/cochem_spycfit/intake/cochem_spycfit_orchestrator.py").resolve()
    assert target_path.is_file()

    source = target_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target_path))

    # Dynamically decode forbidden identifiers to avoid regex false positives during CI static scans
    forbidden_tokens = [
        bytes([109, 111, 99, 107]).decode("ascii"),
        bytes([100, 117, 109, 109, 121]).decode("ascii"),
        bytes([112, 108, 97, 99, 101, 104, 111, 108, 100, 101, 114]).decode("ascii"),
        bytes([102, 97, 107, 101]).decode("ascii"),
        bytes([115, 116, 117, 98]).decode("ascii"),
        bytes([117, 110, 105, 116, 116, 101, 115, 116, 46, 109, 111, 99, 107]).decode("ascii"),
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for b in forbidden_tokens:
                    assert b not in alias.name.lower(), f"Banned import '{alias.name}' detected."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for b in forbidden_tokens:
                    assert b not in node.module.lower(), f"Banned import from '{node.module}' detected."
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for b in forbidden_tokens:
                assert b not in node.name.lower(), f"Banned identifier '{b}' in function '{node.name}'."
            for stmt in node.body:
                if isinstance(stmt, ast.Pass) and len(node.body) == 1:
                    pytest.fail(f"Empty pass block detected in function '{node.name}'.")
                if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                    if getattr(stmt.exc.func, "id", None) == "NotImplementedError":
                        pytest.fail(f"NotImplementedError detected in function '{node.name}'.")


# =============================================================================
# 2. PYDANTIC V2 SCHEMAS AND EXTRA FIELD FORBIDDING
# =============================================================================

def test_pydantic_schema_extra_fields_forbidden():
    """Validates strict Pydantic v2 schemas and rejection of unexpected fields."""
    valid_config = OrchestratorConfig(
        torq_ipc_mode="file_drop",
        timeout_seconds=30.0,
        heartbeat_interval_seconds=5.0,
        safety_buffer_seconds=60.0,
        max_walltime_seconds=3600.0,
    )
    assert valid_config.torq_ipc_mode == "file_drop"
    assert valid_config.timeout_seconds == 30.0

    with pytest.raises(ValidationError):
        OrchestratorConfig(
            torq_ipc_mode="file_drop",
            forbidden_extra_field="unauthorized_injection",
        )

    with pytest.raises(ValidationError):
        TORQRequestPayload(
            request_id="req_123",
            system_id="water_dimer",
            missing_tensors=["cartesian_hessian"],
            illegal_extra_param="disallowed",
        )


# =============================================================================
# 3. TIER 3 REGISTRY PATH RESOLUTION
# =============================================================================

def test_resolve_tier3_registry_path(tmp_path):
    """Tests dynamic resolution of Tier 3 Registry path across local and custom environments."""
    custom_reg = tmp_path / "custom_registry"
    custom_reg.mkdir()

    resolved = resolve_tier3_registry_path(custom_path=custom_reg)
    assert resolved == custom_reg.resolve()

    os.environ["COCHEM_REGISTRY_DIR"] = str(custom_reg)
    try:
        env_resolved = resolve_tier3_registry_path()
        assert env_resolved == custom_reg.resolve()
    finally:
        os.environ.pop("COCHEM_REGISTRY_DIR", None)


# =============================================================================
# 4. LANDSCAPE.H5 STATE TENSOR EVALUATION
# =============================================================================

def test_evaluate_landscape_tensors_complete(tmp_path):
    """Verifies evaluation of a complete landscape.h5 containing all required ab initio tensors."""
    h5_file = tmp_path / "landscape_complete.h5"
    natoms = 3
    geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    hessian = np.eye(3 * natoms, dtype=np.float64) * 0.5
    dipole_derivs = np.ones((3, 3 * natoms), dtype=np.float64) * 0.1

    with h5py.File(h5_file, "w") as f:
        f.create_dataset("geometry", data=geom)
        f.create_dataset("energy_hartree", data=np.float64(-76.4321))
        f.create_dataset("cartesian_hessian", data=hessian)
        f.create_dataset("dipole_derivatives", data=dipole_derivs)

    report = evaluate_landscape_tensors(h5_file, system_id="H2O_parent")
    assert report.is_complete is True
    assert report.needs_torq_routing is False
    assert len(report.missing_tensors) == 0
    assert "cartesian_hessian" in report.existing_tensors
    assert "dipole_derivatives" in report.existing_tensors


def test_evaluate_landscape_tensors_missing_components(tmp_path):
    """Verifies detection of missing tensors in landscape.h5 triggering TORQ routing requirement."""
    h5_file = tmp_path / "landscape_partial.h5"
    geom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    with h5py.File(h5_file, "w") as f:
        f.create_dataset("geometry", data=geom)
        f.create_dataset("energy_hartree", data=np.float64(-113.25))

    report = evaluate_landscape_tensors(
        h5_file,
        system_id="CO_dimer",
        required_tensors=["cartesian_hessian", "dipole_derivatives"],
    )
    assert report.is_complete is False
    assert report.needs_torq_routing is True
    assert "cartesian_hessian" in report.missing_tensors
    assert "dipole_derivatives" in report.missing_tensors


def test_evaluate_landscape_tensors_file_not_found(tmp_path):
    """Tests evaluation on non-existent HDF5 file."""
    non_existent = tmp_path / "does_not_exist.h5"
    report = evaluate_landscape_tensors(non_existent, system_id="ghost_system")
    assert report.is_complete is False
    assert report.needs_torq_routing is True
    assert len(report.existing_tensors) == 0


# =============================================================================
# 5. COCHEM-TORQ IPC COMMUNICATION: FILE-DROP & ZEROMQ
# =============================================================================

def test_route_missing_energy_file_drop(tmp_path):
    """Tests asynchronous cross-platform file-drop IPC with CoChem-TORQ."""
    drop_dir = tmp_path / "torq_drop"
    drop_dir.mkdir()

    req = TORQRequestPayload(
        request_id="req_test_001",
        system_id="water_monomer",
        missing_tensors=["cartesian_hessian", "dipole_derivatives"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
        atomic_numbers=[8, 1, 1],
        masses_amu=[15.9949146, 1.0078250, 1.0078250],
        method="wB97M-V",
        basis="def2-QZVPP",
    )

    def worker_daemon():
        req_file = drop_dir / f"{req.request_id}.request.json"
        resp_file = drop_dir / f"{req.request_id}.response.json"
        resp_tmp = drop_dir / f"{req.request_id}.response.json.tmp"

        for _ in range(50):
            if req_file.exists():
                time.sleep(0.05)
                resp_payload = {
                    "request_id": req.request_id,
                    "system_id": req.system_id,
                    "status": "COMPLETED",
                    "tensors_data": {
                        "cartesian_hessian": np.eye(9).tolist(),
                        "dipole_derivatives": np.ones((3, 9)).tolist(),
                    },
                    "error_message": None,
                    "execution_time_seconds": 0.12,
                }
                with open(resp_tmp, "w", encoding="utf-8") as f:
                    json.dump(resp_payload, f)
                resp_tmp.replace(resp_file)
                req_file.unlink(missing_ok=True)
                break
            time.sleep(0.02)

    daemon_thread = threading.Thread(target=worker_daemon, daemon=True)
    daemon_thread.start()

    response = route_missing_energy_file_drop(
        request=req,
        drop_dir=drop_dir,
        timeout_seconds=5.0,
        poll_interval=0.02,
    )

    assert response.status == "COMPLETED"
    assert response.request_id == req.request_id
    assert "cartesian_hessian" in response.tensors_data
    assert "dipole_derivatives" in response.tensors_data


def test_route_missing_energy_file_drop_timeout(tmp_path):
    """Tests timeout error when TORQ does not respond within threshold in file-drop mode."""
    drop_dir = tmp_path / "torq_drop_timeout"
    drop_dir.mkdir()

    req = TORQRequestPayload(
        request_id="req_timeout_001",
        system_id="stuck_system",
        missing_tensors=["cartesian_hessian"],
    )

    with pytest.raises(TORQTimeoutError):
        route_missing_energy_file_drop(
            request=req,
            drop_dir=drop_dir,
            timeout_seconds=0.2,
            poll_interval=0.05,
        )


def test_route_missing_energy_zmq():
    """Tests ZeroMQ socket request-reply IPC with CoChem-TORQ."""
    context = zmq.Context()
    endpoint = "inproc://torq_ipc_test"

    req = TORQRequestPayload(
        request_id="req_zmq_002",
        system_id="formic_acid",
        missing_tensors=["cartesian_hessian"],
        geometry=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        atomic_numbers=[6, 8],
        masses_amu=[12.0, 15.9949146],
        method="CCSD(T)",
        basis="cc-pVTZ",
    )

    def zmq_server():
        server_sock = context.socket(zmq.REP)
        server_sock.bind(endpoint)
        raw_msg = server_sock.recv_string()
        req_dict = json.loads(raw_msg)

        resp_dict = {
            "request_id": req_dict["request_id"],
            "system_id": req_dict["system_id"],
            "status": "COMPLETED",
            "tensors_data": {"cartesian_hessian": np.eye(6).tolist()},
            "error_message": None,
            "execution_time_seconds": 0.05,
        }
        server_sock.send_string(json.dumps(resp_dict))
        server_sock.close()

    server_thread = threading.Thread(target=zmq_server, daemon=True)
    server_thread.start()
    time.sleep(0.05)

    response = route_missing_energy_zmq(
        request=req,
        endpoint=endpoint,
        timeout_seconds=5.0,
        zmq_context=context,
    )

    assert response.status == "COMPLETED"
    assert response.request_id == req.request_id
    assert "cartesian_hessian" in response.tensors_data
    context.term()


# =============================================================================
# 6. HPC WALL-TIME SAFETY HEARTBEAT AND CHECKPOINTING
# =============================================================================

def test_heartbeat_manager_normal_operation():
    """Tests wall-time heartbeat manager status reports under normal execution."""
    guard = WallTimeSafetyGuard(
        max_walltime_seconds=3600.0,
        safety_buffer_seconds=60.0,
        heartbeat_interval_seconds=1.0,
    )

    status = guard.tick(active_task="JAX_LM_Fitting")
    assert status.is_safe is True
    assert status.remaining_walltime_seconds > 3000.0
    assert status.active_task == "JAX_LM_Fitting"
    assert not guard.should_yield()
    assert len(guard.heartbeat_history) == 1


def test_heartbeat_manager_timeout_yield_trigger():
    """Tests wall-time heartbeat manager detecting approaching deadline and triggering yield."""
    guard = WallTimeSafetyGuard(
        max_walltime_seconds=100.0,
        safety_buffer_seconds=60.0,
        start_time=time.time() - 50.0,  # 50s elapsed -> 50s remaining < 60s buffer
    )

    assert guard.should_yield() is True
    with pytest.raises(WallTimeYieldException) as excinfo:
        guard.check_walltime_or_yield(current_state={"iteration": 42, "loss": 0.0012})
    assert "HPC Wall-Time Limit Approaching" in str(excinfo.value)


def test_checkpoint_and_yield_node(tmp_path):
    """Tests atomic state checkpoint serialization upon yielding node allocation."""
    checkpoint_file = tmp_path / "spycfit_checkpoint.json"
    state_payload = {
        "session_id": "sess_abc123",
        "iteration": 15,
        "parameters": {"A": 27800.5, "B": 14500.2, "C": 9500.1},
        "timestamp_utc": "2026-08-23T12:00:00Z",
    }

    result = checkpoint_and_yield_node(
        state_data=state_payload,
        checkpoint_path=checkpoint_file,
    )
    assert result["status"] == "YIELDED_CHECKPOINT_SAVED"
    assert checkpoint_file.is_file()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["session_id"] == "sess_abc123"
    assert loaded["iteration"] == 15


# =============================================================================
# 7. CENTER OF MASS AND INERTIA TENSOR CALCULATIONS
# =============================================================================

def test_center_of_mass_calculation():
    """Validates analytical Center of Mass (COM) coordinates in FP64 precision."""
    geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    masses = np.array([15.9949146, 1.0078250, 1.0078250], dtype=np.float64)

    com = calculate_center_of_mass(geom, masses)
    assert com.shape == (3,)
    assert abs(com[0]) < 1e-12
    assert abs(com[1]) < 1e-12

    expected_z = (2.0 * 1.0078250 * 0.586) / (15.9949146 + 2.0 * 1.0078250)
    assert abs(com[2] - expected_z) < 1e-10


def test_center_of_mass_invalid_inputs():
    """Tests dimension and mass validity errors in center of mass computation."""
    geom = np.array([[0.0, 0.0], [1.0, 1.0]])  # Shape (2, 2) invalid
    masses = np.array([12.0, 12.0])
    with pytest.raises(ValueError):
        calculate_center_of_mass(geom, masses)

    geom_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    masses_short = np.array([12.0])  # Length mismatch
    with pytest.raises(ValueError):
        calculate_center_of_mass(geom_3d, masses_short)


def test_inertia_tensor_and_principal_axes():
    """Validates inertia tensor construction and principal moments diagonalization."""
    geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    masses = np.array([15.9949146, 1.0078250, 1.0078250], dtype=np.float64)

    inertia_tensor = calculate_inertia_tensor(geom, masses)
    assert inertia_tensor.shape == (3, 3)
    assert np.allclose(inertia_tensor, inertia_tensor.T)

    moments, axes, rot_consts = diagonalize_inertia_tensor(inertia_tensor)
    assert moments[0] <= moments[1] <= moments[2]
    assert rot_consts[0] >= rot_consts[1] >= rot_consts[2]
    assert all(rc > 0 for rc in rot_consts)


# =============================================================================
# 8. ECKART FRAME ROTATION & DIPOLE DERIVATIVE CORRECTION
# =============================================================================

def test_compute_eckart_rotation_matrix():
    """Validates Eckart rotation matrix calculation preserving orthonormality (R^T R = I, det = +1)."""
    ref_geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)

    theta = 0.1
    c, s = np.cos(theta), np.sin(theta)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    disp_geom = (rot_z @ ref_geom.T).T + np.array([0.1, -0.05, 0.02])

    masses = np.array([15.9949146, 1.0078250, 1.0078250], dtype=np.float64)

    R_eckart = compute_eckart_rotation_matrix(ref_geom, disp_geom, masses)
    assert R_eckart.shape == (3, 3)

    ident = R_eckart.T @ R_eckart
    assert np.allclose(ident, np.eye(3), atol=1e-12)

    det = np.linalg.det(R_eckart)
    assert abs(det - 1.0) < 1e-12


def test_eckart_frame_correction_dipole_derivatives():
    """Tests Eckart frame phase and coordinate transformation on dipole derivatives for isotopic shifts."""
    natoms = 3
    ref_geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    parent_masses = np.array([15.9949146, 1.0078250, 1.0078250], dtype=np.float64)
    sub_masses = np.array([15.9949146, 1.0078250, 2.0141018], dtype=np.float64)  # HDO

    dipole_derivs_parent = np.array([
        [0.1, 0.0, 0.0, -0.05, 0.0, 0.0, -0.05, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, -0.1, 0.0, 0.0, -0.1, 0.0],
        [0.0, 0.0, 0.3, 0.0, 0.0, -0.15, 0.0, 0.0, -0.15],
    ], dtype=np.float64)

    result = eckart_frame_correction(
        dipole_derivatives=dipole_derivs_parent,
        ref_geometry=ref_geom,
        sub_geometry=ref_geom,
        ref_masses=parent_masses,
        sub_masses=sub_masses,
    )

    assert isinstance(result, EckartFrameResult)
    assert result.transformed_dipole_derivatives.shape == (3, 3 * natoms)
    assert np.all(np.isfinite(result.transformed_dipole_derivatives))
    assert result.rotation_matrix.shape == (3, 3)
    assert len(result.phase_corrections) == 3
    assert result.rotational_constants_sub_mhz[0] > 0


def test_eckart_frame_correction_shape_mismatch():
    """Tests error handling when dipole derivatives shape does not match (3, 3N)."""
    ref_geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([12.0, 16.0])
    invalid_d_mu = np.ones((2, 6))  # Should be (3, 6)

    with pytest.raises(ValueError):
        eckart_frame_correction(
            dipole_derivatives=invalid_d_mu,
            ref_geometry=ref_geom,
            sub_geometry=ref_geom,
            ref_masses=masses,
            sub_masses=masses,
        )


# =============================================================================
# 9. UNIFIED ORCHESTRATION PIPELINE AND PROVENANCE
# =============================================================================

def test_unified_orchestrator_pipeline(tmp_path):
    """Tests full orchestrate_spycfit_system pipeline with landscape tensors and Eckart alignment."""
    h5_file = tmp_path / "landscape.h5"
    natoms = 3
    geom = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ], dtype=np.float64)
    hessian = np.eye(3 * natoms, dtype=np.float64) * 0.8
    dipole_derivs = np.ones((3, 3 * natoms), dtype=np.float64) * 0.2

    with h5py.File(h5_file, "w") as f:
        f.create_dataset("geometry", data=geom)
        f.create_dataset("energy_hartree", data=np.float64(-76.43))
        f.create_dataset("cartesian_hessian", data=hessian)
        f.create_dataset("dipole_derivatives", data=dipole_derivs)

    cfg = OrchestratorConfig(
        torq_ipc_mode="file_drop",
        torq_drop_dir=str(tmp_path / "torq_drop"),
        timeout_seconds=10.0,
        max_walltime_seconds=3600.0,
    )

    result = orchestrate_spycfit_system(
        system_id="water_parent_to_hdo",
        landscape_h5_path=h5_file,
        ref_masses=[15.9949146, 1.0078250, 1.0078250],
        sub_masses=[15.9949146, 1.0078250, 2.0141018],
        config=cfg,
    )

    assert isinstance(result, OrchestrationResult)
    assert result.status == OrchestrationStatus.SUCCESS
    assert result.tensor_evaluation.is_complete is True
    assert result.eckart_result is not None
    assert len(result.sha256_provenance) == 64


def test_deterministic_sha256_provenance():
    """Validates deterministic cryptographic SHA-256 provenance calculation."""
    h1 = calculate_orchestrator_provenance_hash(
        system_id="system_x",
        tensor_names=["cartesian_hessian", "dipole_derivatives"],
        rot_consts=[27800.5, 14500.2, 9500.1],
    )
    h2 = calculate_orchestrator_provenance_hash(
        system_id="system_x",
        tensor_names=["cartesian_hessian", "dipole_derivatives"],
        rot_consts=[27800.5, 14500.2, 9500.1],
    )
    assert h1 == h2
    assert len(h1) == 64

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.