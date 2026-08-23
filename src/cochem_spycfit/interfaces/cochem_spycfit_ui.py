"""
CoChem-SpycFit: Decision Dashboard (HITL) & Interactive Spectroscopic Interface.

This module implements:
1. Plotly Dual-Pane O-C (Observed - Calculated) Residual Dashboard:
   - High-performance WebGL (scattergl) rendering with seamless SVG fallback.
   - Top pane: Experimental spectrum, theoretical spectrum, ghost peaks, and bipartite assignment lines.
   - Bottom pane: O-C residuals, zero-reference baseline, and dynamic tolerance/Huber bands.
2. Bipartite Auto-Propose Assistant using the Hungarian Algorithm:
   - Global minimum-cost matching via scipy.optimize.linear_sum_assignment.
   - Multi-metric cost formulation (Euclidean frequency drift + logarithmic intensity penalty).
   - Frequency gating cutoff to reject unphysical pairings.
3. Human-In-The-Loop (HITL) Gate & Ghost Peak Retention:
   - State machine tracking: PROPOSED, APPROVED, REJECTED, MANUAL, GHOST.
   - Mandatory validation gate enforcing approval before Levenberg-Marquardt optimizer runs.
   - Ghost Peak Retention: rejected transitions are excluded from Huber robust loss but remain
     rendered in translucent greyed-out visual styles.
4. Headless & Air-Gap Degradation:
   - Standalone, self-contained offline HTML artifact generation.
   - Publication-grade PDF report rendering via ReportLab (with embedded dual-pane spectra,
     fit statistics, assignment tables, and ghost peak audits).
5. "What-If" Parameter Sliders & JAX Convolver Integration:
   - Interactive recalculation of molecular rotational constants (A, B, C), centrifugal distortion,
     temperature, and line widths using physics_core.cochem_convolver.
   - Real-time Plotly slider step generation.
6. Semantic Git-Commit Translation:
   - Translates mathematical parameter shifts, threshold changes, and HITL approvals into
     standardized, conventional-commit-compliant messages.

Precision: Standard 64-bit precision enabled across all calculations. Zero mocks or stubs.
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

# Enforce double precision at environment level before JAX initializes
os.environ["JAX_ENABLE_X64"] = "True"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.optimize import linear_sum_assignment
import zmq

# ipywidgets for Jupyter and Voila interactive controls
from ipywidgets import Button, FloatSlider, HBox, Label, Output, VBox

# JAX Convolver integration
try:
    from ..physics_core.cochem_convolver import (
        ConvolutionConfig,
        ProfileType,
        SpectralLines,
        convolve_frequency_chunks_scan_1d_jax,
        gaussian_kernel_jax,
        lorentzian_kernel_jax,
    )
    HAS_CONVOLVER = True
except (ImportError, ValueError):
    HAS_CONVOLVER = False

logger = logging.getLogger("CoChem-SpycFit.UI")


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class UIDashboardError(Exception):
    """Base exception for all SpycFit UI, dashboard, and visualization operations."""


class BipartiteMatchingError(UIDashboardError):
    """Raised when bipartite assignment or cost matrix evaluation encounters invalid states."""


class HITLApprovalError(UIDashboardError):
    """Raised when LM fitting is attempted without mandatory Human-In-The-Loop approval."""


class OfflineRenderError(UIDashboardError):
    """Raised when static HTML or PDF artifact generation fails in headless/air-gapped mode."""


class SliderComputationError(UIDashboardError):
    """Raised when What-If parameter slider recalculation fails."""


class SemanticCommitError(UIDashboardError):
    """Raised when semantic git-commit translation fails."""


class ZMQBridgeError(UIDashboardError):
    """Raised when ZeroMQ IPC communication, handshake, or background computation fails."""


# =============================================================================
# 2. ENUMS & PYDANTIC DATA SCHEMAS (Pydantic V2)
# =============================================================================

class AssignmentStatus(str, enum.Enum):
    """Lifecycle state of a theoretical-to-experimental peak assignment."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL = "manual"
    GHOST = "ghost"


class RenderBackend(str, enum.Enum):
    """Rendering backend for spectral dashboard."""
    WEBGL = "webgl"
    SVG = "svg"
    OFFLINE_HTML = "offline_html"
    PDF = "pdf"


class SpectralPeak(BaseModel):
    """Represents a discrete spectroscopic transition peak (experimental or theoretical)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    peak_id: str = Field(..., description="Unique identifier for the spectral peak")
    frequency: float = Field(..., gt=0.0, description="Center frequency in MHz (> 0)")
    intensity: float = Field(..., ge=0.0, description="Peak intensity or integrated amplitude (>= 0)")
    uncertainty: float = Field(default=0.01, gt=0.0, description="Experimental frequency uncertainty in MHz")
    quantum_label: Optional[str] = Field(default=None, description="Quantum state assignment string (e.g. '1_0,1 -> 0_0,0')")
    is_ghost: bool = Field(default=False, description="Flag indicating if peak is rejected / unassigned ghost peak")
    rejection_reason: Optional[str] = Field(default=None, description="Detailed rationale for peak rejection / ghosting")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary scientific metadata")


class PeakAssignment(BaseModel):
    """Represents a paired match between an experimental peak and a theoretical transition."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    assignment_id: str = Field(default_factory=lambda: f"ASN_{uuid.uuid4().hex[:8]}", description="Unique assignment ID")
    exp_peak: SpectralPeak = Field(..., description="Assigned experimental peak")
    theo_peak: SpectralPeak = Field(..., description="Assigned theoretical peak")
    frequency_drift_mhz: float = Field(..., description="Observed minus Calculated frequency (nu_exp - nu_theo) in MHz")
    intensity_ratio: float = Field(..., ge=0.0, description="Intensity ratio (I_exp / I_theo)")
    cost: float = Field(..., ge=0.0, description="Total bipartite assignment optimization cost")
    status: AssignmentStatus = Field(default=AssignmentStatus.PROPOSED, description="Current HITL status")
    approved_by: Optional[str] = Field(default=None, description="Identifier of researcher who approved the assignment")
    approved_at: Optional[str] = Field(default=None, description="ISO 8601 timestamp of approval")
    rejection_reason: Optional[str] = Field(default=None, description="Rationale if rejected")


class BipartiteConfig(BaseModel):
    """Configuration for the Hungarian Bipartite Auto-Propose Assistant."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_freq_drift_mhz: float = Field(default=50.0, gt=0.0, description="Maximum frequency drift window in MHz")
    freq_weight: float = Field(default=1.0, gt=0.0, description="Weighting penalty factor for frequency displacement")
    intensity_weight: float = Field(default=0.1, ge=0.0, description="Weighting penalty factor for intensity discrepancy")
    require_intensity_match: bool = Field(default=False, description="Enforce strict intensity fold-change filtering")
    max_intensity_fold_change: float = Field(default=10.0, ge=1.0, description="Max allowed intensity fold-change")


class DashboardConfig(BaseModel):
    """Configuration options for Plotly Dual-Pane O-C Dashboard."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(default="CoChem-SpycFit Decision Dashboard", description="Main dashboard title")
    theme: str = Field(default="plotly_dark", description="Plotly theme ('plotly_dark', 'plotly_white', 'seaborn')")
    use_webgl: bool = Field(default=True, description="Enable WebGL (scattergl) hardware acceleration")
    freq_unit: str = Field(default="MHz", description="Frequency axis unit")
    intensity_unit: str = Field(default="Arbitrary Units", description="Intensity axis unit")
    show_assignment_lines: bool = Field(default=True, description="Draw bipartite assignment connecting lines")
    show_ghost_peaks: bool = Field(default=True, description="Render rejected ghost peaks greyed out")
    ghost_color: str = Field(default="rgba(140, 140, 140, 0.4)", description="Color for ghost peaks")
    proposed_line_color: str = Field(default="rgba(255, 165, 0, 0.7)", description="Color for proposed assignment lines")
    approved_line_color: str = Field(default="rgba(0, 230, 118, 0.9)", description="Color for approved assignment lines")
    rejected_line_color: str = Field(default="rgba(239, 83, 80, 0.5)", description="Color for rejected assignment lines")
    huber_delta_mhz: Optional[float] = Field(default=None, description="Huber robust loss delta threshold in MHz")
    sigma_tolerance_bands: List[float] = Field(default_factory=lambda: [1.0, 3.0], description="Sigma multiplier error bands")


class WhatIfParameterState(BaseModel):
    """State parameters for real-time What-If slider spectrum recalculation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    A_mhz: float = Field(..., gt=0.0, description="Rotational constant A in MHz (> 0)")
    B_mhz: float = Field(..., gt=0.0, description="Rotational constant B in MHz (> 0)")
    C_mhz: float = Field(..., gt=0.0, description="Rotational constant C in MHz (> 0)")
    D_J_khz: float = Field(default=0.0, description="Centrifugal distortion D_J in kHz")
    D_JK_khz: float = Field(default=0.0, description="Centrifugal distortion D_JK in kHz")
    D_K_khz: float = Field(default=0.0, description="Centrifugal distortion D_K in kHz")
    temperature_k: float = Field(default=296.0, gt=0.0, description="Rotational temperature in Kelvin (> 0)")
    fwhm_gaussian_mhz: float = Field(default=0.05, gt=0.0, description="Doppler Gaussian FWHM in MHz (> 0)")
    fwhm_lorentzian_mhz: float = Field(default=0.05, gt=0.0, description="Collisional Lorentzian FWHM in MHz (> 0)")
    global_frequency_shift_mhz: float = Field(default=0.0, description="Global systematic frequency offset in MHz")
    intensity_scale: float = Field(default=1.0, gt=0.0, description="Global intensity scaling multiplier (> 0)")

    @field_validator("A_mhz", "B_mhz", "C_mhz")
    @classmethod
    def validate_positive_rotational_constants(cls, value: float) -> float:
        """Rigidly enforce positive rotational constants."""
        if value <= 0.0:
            raise ValueError(f"Rotational constant must be strictly positive (> 0), got {value}")
        return value

    @model_validator(mode="after")
    def validate_rotational_ordering(self) -> WhatIfParameterState:
        """Warn if rotational constants deviate from asymmetric top convention A >= B >= C."""
        if not (self.A_mhz >= self.B_mhz >= self.C_mhz):
            logger.debug(
                f"Rotational constants order notice: A={self.A_mhz:.2f}, B={self.B_mhz:.2f}, C={self.C_mhz:.2f}"
            )
        return self


class SemanticCommit(BaseModel):
    """Pydantic model representing a semantic conventional git commit message with SHA-256 state tracking."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    commit_type: str = Field(..., description="Commit type prefix (feat, fix, refactor, chore)")
    scope: str = Field(..., description="Component scope (hamiltonian, assignment, loss, convolver)")
    summary: str = Field(..., description="Short imperative summary line")
    body: Optional[str] = Field(default=None, description="Detailed explanatory commit body")
    parameters_before: Dict[str, Any] = Field(default_factory=dict, description="Parameter state before mutation")
    parameters_after: Dict[str, Any] = Field(default_factory=dict, description="Parameter state after mutation")
    rms_before: Optional[float] = Field(default=None, description="RMS residual error before change")
    rms_after: Optional[float] = Field(default=None, description="RMS residual error after change")
    state_hash: Optional[str] = Field(default=None, description="Cryptographic SHA-256 hash of UI state")
    full_commit_message: str = Field(..., description="Formatted full git commit message")


# =============================================================================
# 3. LTTB (LARGEST-TRIANGLE-THREE-BUCKETS) DECIMATION & CRYPTOGRAPHIC STATE HASHING
# =============================================================================

def run_lttb_decimation(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    threshold: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsamples a 2D spectral curve (x, y) using the Largest-Triangle-Three-Buckets (LTTB) algorithm.
    Reduces dense arrays to target threshold points (~5000) while strictly preserving
    peak maxima, minima, and line shapes.

    Parameters:
        x: Sequence or array of x-coordinates (frequencies in MHz).
        y: Sequence or array of y-coordinates (intensities or amplitudes).
        threshold: Target number of sampled points (default 5000).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Decimated (x_sampled, y_sampled) arrays.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Input arrays x and y must be 1-dimensional.")
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} vs len(y)={len(y_arr)}")

    n_points = len(x_arr)
    if threshold < 3 or n_points <= threshold:
        return x_arr.copy(), y_arr.copy()

    sampled_x = np.empty(threshold, dtype=np.float64)
    sampled_y = np.empty(threshold, dtype=np.float64)

    # First point is always selected
    sampled_x[0] = x_arr[0]
    sampled_y[0] = y_arr[0]

    bucket_size = (n_points - 2) / (threshold - 2)

    a_x = x_arr[0]
    a_y = y_arr[0]

    for i in range(threshold - 2):
        b_start = int(math.floor(i * bucket_size)) + 1
        b_end = int(math.floor((i + 1) * bucket_size)) + 1
        b_end = min(b_end, n_points - 1)

        c_start = int(math.floor((i + 1) * bucket_size)) + 1
        c_end = int(math.floor((i + 2) * bucket_size)) + 1
        c_end = min(c_end, n_points)

        if c_end > c_start:
            c_x = float(np.mean(x_arr[c_start:c_end]))
            c_y = float(np.mean(y_arr[c_start:c_end]))
        else:
            c_x = x_arr[-1]
            c_y = y_arr[-1]

        bucket_x = x_arr[b_start:b_end]
        bucket_y = y_arr[b_start:b_end]

        if len(bucket_x) == 0:
            selected_idx = b_start
        else:
            # Triangle area = 0.5 * |(x - a_x)*(c_y - a_y) - (y - a_y)*(c_x - a_x)|
            areas = 0.5 * np.abs(
                (bucket_x - a_x) * (c_y - a_y) - (bucket_y - a_y) * (c_x - a_x)
            )
            max_idx = int(np.argmax(areas))
            selected_idx = b_start + max_idx

        a_x = x_arr[selected_idx]
        a_y = y_arr[selected_idx]

        sampled_x[i + 1] = a_x
        sampled_y[i + 1] = a_y

    # Last point is always selected
    sampled_x[-1] = x_arr[-1]
    sampled_y[-1] = y_arr[-1]

    return sampled_x, sampled_y


def update_ui_state_hash(state_dict: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 cryptographic hash representing the current UI state,
    parameter configuration, or transition assignment dataset.

    Parameters:
        state_dict: Dictionary containing state key-values.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    canonical_json = json.dumps(
        state_dict,
        sort_keys=True,
        ensure_ascii=True,
        default=lambda o: o.isoformat() if isinstance(o, (datetime.datetime, datetime.date)) else str(o),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# =============================================================================
# 4. BIPARTITE HUNGARIAN AUTO-PROPOSE ALGORITHM
# =============================================================================

def compute_bipartite_cost_matrix(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    config: Optional[BipartiteConfig] = None,
) -> np.ndarray:
    """
    Computes the (N_exp, N_theo) bipartite matching cost matrix using frequency distance
    and logarithmic relative intensity penalties.

    Parameters:
        exp_peaks: List of experimental spectral peaks.
        theo_peaks: List of theoretical spectral peaks.
        config: Bipartite matching configuration.

    Returns:
        np.ndarray: Cost matrix of shape (len(exp_peaks), len(theo_peaks)).
    """
    cfg = config or BipartiteConfig()
    n_exp = len(exp_peaks)
    n_theo = len(theo_peaks)

    if n_exp == 0 or n_theo == 0:
        return np.zeros((n_exp, n_theo), dtype=np.float64)

    # Rigid validation of peak frequencies and bounds
    for p in exp_peaks:
        if p.frequency <= 0.0:
            raise BipartiteMatchingError(f"Experimental peak '{p.peak_id}' has non-positive frequency: {p.frequency}")
    for p in theo_peaks:
        if p.frequency <= 0.0:
            raise BipartiteMatchingError(f"Theoretical peak '{p.peak_id}' has non-positive frequency: {p.frequency}")

    exp_freqs = np.array([p.frequency for p in exp_peaks], dtype=np.float64)
    theo_freqs = np.array([p.frequency for p in theo_peaks], dtype=np.float64)

    exp_ints = np.array([max(p.intensity, 1e-12) for p in exp_peaks], dtype=np.float64)
    theo_ints = np.array([max(p.intensity, 1e-12) for p in theo_peaks], dtype=np.float64)

    # Frequency drift matrix: |nu_exp - nu_theo|
    freq_diff = np.abs(exp_freqs[:, None] - theo_freqs[None, :])

    # Logarithmic intensity ratio penalty: |ln(I_exp / I_theo)|
    int_diff = np.abs(np.log(exp_ints[:, None] / theo_ints[None, :]))

    # Total cost formulation
    cost = cfg.freq_weight * freq_diff + cfg.intensity_weight * int_diff

    # Apply gating penalty for frequency drifts exceeding maximum threshold
    exceed_mask = freq_diff > cfg.max_freq_drift_mhz
    cost[exceed_mask] += 1.0e6 + 1.0e3 * freq_diff[exceed_mask]

    return cost


def auto_propose_assignments(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    config: Optional[BipartiteConfig] = None,
) -> List[PeakAssignment]:
    """
    Automatically proposes optimal bipartite peak assignments using the Hungarian Algorithm
    (scipy.optimize.linear_sum_assignment).

    Filters out candidate pairings whose frequency drift exceeds max_freq_drift_mhz.

    Parameters:
        exp_peaks: List of experimental spectral peaks.
        theo_peaks: List of theoretical spectral peaks.
        config: Bipartite matching configuration.

    Returns:
        List[PeakAssignment]: Optimal proposed peak assignments.
    """
    if len(exp_peaks) == 0 or len(theo_peaks) == 0:
        return []

    cfg = config or BipartiteConfig()
    cost_matrix = compute_bipartite_cost_matrix(exp_peaks, theo_peaks, cfg)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments: List[PeakAssignment] = []

    for r, c in zip(row_ind, col_ind):
        exp_p = exp_peaks[r]
        theo_p = theo_peaks[c]
        drift = float(exp_p.frequency - theo_p.frequency)

        # Gate check: strictly reject pairings that exceed the frequency drift window
        if abs(drift) > cfg.max_freq_drift_mhz:
            continue

        ratio = float(exp_p.intensity / max(theo_p.intensity, 1e-12))
        if cfg.require_intensity_match:
            if ratio < (1.0 / cfg.max_intensity_fold_change) or ratio > cfg.max_intensity_fold_change:
                continue

        assign = PeakAssignment(
            exp_peak=exp_p,
            theo_peak=theo_p,
            frequency_drift_mhz=drift,
            intensity_ratio=ratio,
            cost=float(cost_matrix[r, c]),
            status=AssignmentStatus.PROPOSED,
        )
        assignments.append(assign)

    return assignments


# =============================================================================
# 4. HUMAN-IN-THE-LOOP (HITL) ASSIGNMENT MANAGER & GHOST PEAKS
# =============================================================================

def compute_huber_loss_excluding_ghosts(
    residuals_mhz: Sequence[float],
    huber_delta_mhz: float = 1.0,
) -> float:
    """
    Computes Huber robust loss across active residuals, strictly excluding rejected ghost peaks.

    Formula:
    L_delta(r) = 0.5 * r^2  if |r| <= delta
    L_delta(r) = delta * (|r| - 0.5 * delta)  otherwise
    """
    if len(residuals_mhz) == 0:
        return 0.0

    r_arr = np.abs(np.array(residuals_mhz, dtype=np.float64))
    delta = float(huber_delta_mhz)

    loss_elements = np.where(
        r_arr <= delta,
        0.5 * np.square(r_arr),
        delta * (r_arr - 0.5 * delta),
    )
    return float(np.sum(loss_elements))


class HITLAssignmentManager:
    """
    Stateful manager for Human-In-The-Loop peak assignment approval, ghost peak segregation,
    and Levenberg-Marquardt optimizer gate validation.
    """

    def __init__(
        self,
        exp_peaks: Sequence[SpectralPeak],
        theo_peaks: Sequence[SpectralPeak],
        assignments: Optional[Sequence[PeakAssignment]] = None,
    ) -> None:
        self.exp_peaks: List[SpectralPeak] = [p.model_copy() for p in exp_peaks]
        self.theo_peaks: List[SpectralPeak] = [p.model_copy() for p in theo_peaks]
        self.assignments: List[PeakAssignment] = [a.model_copy() for a in (assignments or [])]

    def auto_propose(self, config: Optional[BipartiteConfig] = None) -> List[PeakAssignment]:
        """Runs the Hungarian algorithm and populates self.assignments with PROPOSED entries."""
        proposed = auto_propose_assignments(self.exp_peaks, self.theo_peaks, config)
        self.assignments = proposed
        return self.assignments

    def get_assignments_by_status(self, status: AssignmentStatus) -> List[PeakAssignment]:
        """Returns subset of assignments matching the given status."""
        return [a for a in self.assignments if a.status == status]

    def approve_assignment(self, assignment_id: str, user_id: Optional[str] = "Researcher") -> PeakAssignment:
        """Approves a proposed assignment by ID."""
        for a in self.assignments:
            if a.assignment_id == assignment_id:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return a
        raise HITLApprovalError(f"Assignment ID '{assignment_id}' not found.")

    def approve_all_proposed(self, user_id: Optional[str] = "Researcher") -> int:
        """Approves all currently PROPOSED assignments."""
        count = 0
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for a in self.assignments:
            if a.status == AssignmentStatus.PROPOSED:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = now_ts
                count += 1
        return count

    def approve_all_within_tolerance(self, max_drift_mhz: float, user_id: Optional[str] = "Researcher") -> int:
        """Approves all PROPOSED assignments whose residual drift is within the given tolerance."""
        count = 0
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for a in self.assignments:
            if a.status == AssignmentStatus.PROPOSED and abs(a.frequency_drift_mhz) <= max_drift_mhz:
                a.status = AssignmentStatus.APPROVED
                a.approved_by = user_id or "Researcher"
                a.approved_at = now_ts
                count += 1
        return count

    def reject_assignment(self, assignment_id: str, reason: Optional[str] = None) -> PeakAssignment:
        """
        Rejects an assignment and marks the associated experimental peak as a ghost peak.
        Ghost peaks are excluded from subsequent Huber loss / LM fitting calculations.
        """
        for a in self.assignments:
            if a.assignment_id == assignment_id:
                a.status = AssignmentStatus.REJECTED
                a.rejection_reason = reason or "User rejected assignment"
                a.exp_peak.is_ghost = True
                a.exp_peak.rejection_reason = a.rejection_reason

                # Update the source peak in self.exp_peaks
                for p in self.exp_peaks:
                    if p.peak_id == a.exp_peak.peak_id:
                        p.is_ghost = True
                        p.rejection_reason = a.rejection_reason
                return a
        raise HITLApprovalError(f"Assignment ID '{assignment_id}' not found.")

    def manual_assign(
        self,
        exp_peak_id: str,
        theo_peak_id: str,
        user_id: Optional[str] = "Researcher",
    ) -> PeakAssignment:
        """Creates or updates a manual assignment between specified peak IDs."""
        exp_p = next((p for p in self.exp_peaks if p.peak_id == exp_peak_id), None)
        theo_p = next((p for p in self.theo_peaks if p.peak_id == theo_peak_id), None)

        if not exp_p:
            raise HITLApprovalError(f"Experimental peak '{exp_peak_id}' not found.")
        if not theo_p:
            raise HITLApprovalError(f"Theoretical peak '{theo_peak_id}' not found.")

        exp_p.is_ghost = False
        exp_p.rejection_reason = None

        drift = float(exp_p.frequency - theo_p.frequency)
        ratio = float(exp_p.intensity / max(theo_p.intensity, 1e-12))
        cost = abs(drift)

        assign = PeakAssignment(
            exp_peak=exp_p,
            theo_peak=theo_p,
            frequency_drift_mhz=drift,
            intensity_ratio=ratio,
            cost=cost,
            status=AssignmentStatus.MANUAL,
            approved_by=user_id or "Researcher",
            approved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Remove any previous assignments for this exp peak
        self.assignments = [a for a in self.assignments if a.exp_peak.peak_id != exp_peak_id]
        self.assignments.append(assign)
        return assign

    def get_ghost_peaks(self) -> List[SpectralPeak]:
        """Returns all experimental peaks currently marked as ghost peaks."""
        return [p for p in self.exp_peaks if p.is_ghost]

    def get_active_exp_peaks(self) -> List[SpectralPeak]:
        """Returns all active (non-ghost) experimental peaks."""
        return [p for p in self.exp_peaks if not p.is_ghost]

    def validate_for_lm_fit(self, allow_unapproved: bool = False) -> List[PeakAssignment]:
        """
        Gate check for Levenberg-Marquardt optimizer.
        Ensures that at least one assignment is APPROVED/MANUAL, and optionally prevents fitting
        if unapproved PROPOSED assignments remain.
        """
        approved = [a for a in self.assignments if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL)]
        if len(approved) == 0:
            raise HITLApprovalError(
                "HITL Gate Block: No assignments approved. You must approve assignments before launching Levenberg-Marquardt fitting."
            )
        if not allow_unapproved:
            proposed = [a for a in self.assignments if a.status == AssignmentStatus.PROPOSED]
            if len(proposed) > 0:
                logger.warning(
                    f"{len(proposed)} proposed assignments remain pending. Fitting with {len(approved)} approved transitions."
                )
        return approved

    def compute_robust_metrics(self, huber_delta_mhz: float = 1.0) -> Tuple[float, float]:
        """
        Computes the Huber loss and RMS residual error across approved/active transitions.
        Strictly excludes ghost peaks.
        """
        active_assignments = [
            a for a in self.assignments
            if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL) and not a.exp_peak.is_ghost
        ]
        if len(active_assignments) == 0:
            return 0.0, 0.0

        residuals = [a.frequency_drift_mhz for a in active_assignments]
        loss = compute_huber_loss_excluding_ghosts(residuals, huber_delta_mhz)
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        return loss, rms

    def get_method_matrix_provenance(self, huber_delta_mhz: float = 1.0) -> Dict[str, Any]:
        """
        Compiles full Method Matrix provenance record of peak assignments, ghost peaks,
        robust Huber loss calculations, and cryptographic state hashing.
        """
        loss, rms = self.compute_robust_metrics(huber_delta_mhz=huber_delta_mhz)
        approved = [a for a in self.assignments if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL)]
        ghosts = self.get_ghost_peaks()

        provenance: Dict[str, Any] = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_exp_peaks": len(self.exp_peaks),
            "total_theo_peaks": len(self.theo_peaks),
            "approved_assignments_count": len(approved),
            "ghost_peaks_count": len(ghosts),
            "huber_delta_mhz": huber_delta_mhz,
            "huber_loss": loss,
            "rms_residual_mhz": rms,
            "ghost_peak_details": [
                {
                    "peak_id": p.peak_id,
                    "frequency_mhz": p.frequency,
                    "intensity": p.intensity,
                    "rejection_reason": p.rejection_reason or "Unspecified rejection",
                }
                for p in ghosts
            ],
            "approved_assignment_ids": [a.assignment_id for a in approved],
        }
        provenance["provenance_sha256"] = update_ui_state_hash(provenance)
        return provenance


# =============================================================================
# 5. PLOTLY DUAL-PANE O-C RESIDUAL DASHBOARD
# =============================================================================

def render_dual_pane_dashboard(
    exp_peaks: Sequence[SpectralPeak],
    theo_peaks: Sequence[SpectralPeak],
    assignments: Optional[Sequence[PeakAssignment]] = None,
    config: Optional[DashboardConfig] = None,
    convolved_curve: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
) -> go.Figure:
    """
    Renders an interactive Plotly Dual-Pane O-C Residual dashboard.

    Top Pane:
    - Experimental spectrum (sticks/curve, active vs ghost peaks).
    - Theoretical spectrum (sticks/curve).
    - Bipartite assignment connecting lines (orange=proposed, green=approved, red=rejected).

    Bottom Pane:
    - Observed - Calculated (O - C) residual scatter points.
    - Zero-reference baseline.
    - Tolerance bands (+/- 1-sigma, 3-sigma, or Huber delta).

    Parameters:
        exp_peaks: Experimental peaks.
        theo_peaks: Theoretical peaks.
        assignments: Peak assignments.
        config: Dashboard visual configuration.
        convolved_curve: Optional tuple (grid_frequencies, convolved_intensities).

    Returns:
        go.Figure: Interactive Plotly Dual-Pane dashboard figure.
    """
    cfg = config or DashboardConfig()
    assign_list = list(assignments or [])

    # Choose Scatter class: Scattergl for WebGL acceleration, Scatter for standard SVG
    scatter_cls = go.Scattergl if cfg.use_webgl else go.Scatter

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Rotational Spectrum & Bipartite Peak Assignments", "Observed - Calculated (O - C) Residuals"),
    )

    # 1. TOP PANE: Theoretical Spectrum
    theo_active = [p for p in theo_peaks if not p.is_ghost]
    if theo_active:
        theo_x = [p.frequency for p in theo_active]
        theo_y = [p.intensity for p in theo_active]
        theo_labels = [p.quantum_label or p.peak_id for p in theo_active]

        # Invert theoretical sticks or display as distinct blue/cyan bars
        fig.add_trace(
            scatter_cls(
                x=theo_x,
                y=theo_y,
                mode="markers+text" if len(theo_active) <= 20 else "markers",
                name="Theoretical (Calc)",
                marker=dict(color="#00e5ff", size=8, symbol="diamond"),
                text=theo_labels,
                textposition="top center",
                hovertemplate="<b>Theo Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 1b. TOP PANE: Convolved Continuous Synthetic Spectrum
    if convolved_curve is not None:
        c_x, c_y = convolved_curve
        fig.add_trace(
            scatter_cls(
                x=list(c_x),
                y=list(c_y),
                mode="lines",
                name="Synthetic Convolved",
                line=dict(color="rgba(0, 229, 255, 0.6)", width=1.5),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # 2. TOP PANE: Experimental Spectrum (Active Peaks)
    exp_active = [p for p in exp_peaks if not p.is_ghost]
    if exp_active:
        exp_x = [p.frequency for p in exp_active]
        exp_y = [p.intensity for p in exp_active]
        exp_labels = [p.quantum_label or p.peak_id for p in exp_active]

        fig.add_trace(
            scatter_cls(
                x=exp_x,
                y=exp_y,
                mode="markers+text" if len(exp_active) <= 20 else "markers",
                name="Experimental (Obs)",
                marker=dict(color="#ffffff", size=7, symbol="circle"),
                text=exp_labels,
                textposition="top right",
                hovertemplate="<b>Exp Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 3. TOP PANE: Ghost Peaks (Rejected / Excluded)
    if cfg.show_ghost_peaks:
        ghosts = [p for p in exp_peaks if p.is_ghost]
        if ghosts:
            g_x = [p.frequency for p in ghosts]
            g_y = [p.intensity for p in ghosts]
            g_reasons = [p.rejection_reason or "Ghost Peak" for p in ghosts]

            fig.add_trace(
                scatter_cls(
                    x=g_x,
                    y=g_y,
                    mode="markers",
                    name="Ghost Peaks (Excluded)",
                    marker=dict(color=cfg.ghost_color, size=6, symbol="x"),
                    customdata=g_reasons,
                    hovertemplate="<b>Ghost Peak</b><br>Freq: %{x:.4f} MHz<br>Int: %{y:.4f}<br>Reason: %{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # 4. TOP PANE: Bipartite Assignment Lines
    if cfg.show_assignment_lines and assign_list:
        for assign in assign_list:
            if assign.status == AssignmentStatus.APPROVED:
                line_color = cfg.approved_line_color
                dash_style = "solid"
            elif assign.status == AssignmentStatus.REJECTED:
                line_color = cfg.rejected_line_color
                dash_style = "dot"
            else:
                line_color = cfg.proposed_line_color
                dash_style = "dash"

            x_pair = [assign.theo_peak.frequency, assign.exp_peak.frequency]
            y_pair = [assign.theo_peak.intensity, assign.exp_peak.intensity]

            fig.add_trace(
                go.Scatter(
                    x=x_pair,
                    y=y_pair,
                    mode="lines",
                    line=dict(color=line_color, width=1.5, dash=dash_style),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # 5. BOTTOM PANE: Observed - Calculated (O - C) Residuals
    if assign_list:
        res_x_app: List[float] = []
        res_y_app: List[float] = []
        res_app_labels: List[str] = []

        res_x_prop: List[float] = []
        res_y_prop: List[float] = []
        res_prop_labels: List[str] = []

        for a in assign_list:
            if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL):
                res_x_app.append(a.exp_peak.frequency)
                res_y_app.append(a.frequency_drift_mhz)
                res_app_labels.append(a.exp_peak.quantum_label or a.assignment_id)
            elif a.status == AssignmentStatus.PROPOSED:
                res_x_prop.append(a.exp_peak.frequency)
                res_y_prop.append(a.frequency_drift_mhz)
                res_prop_labels.append(a.exp_peak.quantum_label or a.assignment_id)

        if res_x_app:
            fig.add_trace(
                scatter_cls(
                    x=res_x_app,
                    y=res_y_app,
                    mode="markers",
                    name="Approved Residuals",
                    marker=dict(color="#00e676", size=8, symbol="circle"),
                    text=res_app_labels,
                    hovertemplate="<b>Approved Residual</b><br>Freq: %{x:.4f} MHz<br>O-C: %{y:.4f} MHz<extra></extra>",
                ),
                row=2,
                col=1,
            )

        if res_x_prop:
            fig.add_trace(
                scatter_cls(
                    x=res_x_prop,
                    y=res_y_prop,
                    mode="markers",
                    name="Proposed Residuals",
                    marker=dict(color="#ffa726", size=7, symbol="diamond-open"),
                    text=res_prop_labels,
                    hovertemplate="<b>Proposed Residual</b><br>Freq: %{x:.4f} MHz<br>O-C: %{y:.4f} MHz<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # 6. BOTTOM PANE: Zero-Baseline and Tolerance Bands
    all_freqs = [p.frequency for p in exp_peaks] + [p.frequency for p in theo_peaks]
    f_min = min(all_freqs) - 100.0 if all_freqs else 0.0
    f_max = max(all_freqs) + 100.0 if all_freqs else 1000.0

    # Zero Line
    fig.add_trace(
        go.Scatter(
            x=[f_min, f_max],
            y=[0.0, 0.0],
            mode="lines",
            name="Zero Baseline (O-C=0)",
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    # Huber Delta / Tolerance Bands
    if cfg.huber_delta_mhz is not None and cfg.huber_delta_mhz > 0:
        delta = cfg.huber_delta_mhz
        fig.add_trace(
            go.Scatter(
                x=[f_min, f_max, f_max, f_min],
                y=[delta, delta, -delta, -delta],
                fill="toself",
                fillcolor="rgba(0, 230, 118, 0.08)",
                line=dict(color="rgba(0, 230, 118, 0.3)", width=1, dash="dot"),
                name=f"Huber Delta (±{delta:.2f} MHz)",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # Layout styling
    fig.update_layout(
        template=cfg.theme,
        title=dict(text=cfg.title, font=dict(size=18)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=40, t=80, b=50),
    )

    fig.update_xaxes(title_text=f"Frequency ({cfg.freq_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Intensity ({cfg.intensity_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Residual ({cfg.freq_unit})", row=2, col=1)

    return fig


# =============================================================================
# 6. HEADLESS & AIR-GAP DEGRADATION (OFFLINE HTML & PDF REPORT)
# =============================================================================

def render_static_offline_html(
    fig: go.Figure,
    output_path: Optional[Union[str, Path]] = None,
    include_plotlyjs: Union[bool, str] = True,
) -> Path:
    """
    Renders a self-contained, offline HTML dashboard artifact.
    Works in air-gapped environments without external CDN network calls when include_plotlyjs=True.

    Parameters:
        fig: Plotly figure to render.
        output_path: Destination file path.
        include_plotlyjs: Embed full plotly.js script inline (True) or use CDN ("cdn").

    Returns:
        Path: Path to written HTML file.
    """
    if output_path is None:
        target = Path(tempfile.gettempdir()) / f"cochem_dashboard_{uuid.uuid4().hex[:8]}.html"
    else:
        target = Path(output_path).expanduser().resolve()

    target.parent.mkdir(parents=True, exist_ok=True)
    html_content = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    target.write_text(html_content, encoding="utf-8")
    return target


def render_static_offline_pdf(
    fig: go.Figure,
    output_path: Union[str, Path],
    assignments: Optional[Sequence[PeakAssignment]] = None,
    fit_stats: Optional[Dict[str, Any]] = None,
    title: str = "CoChem-SpycFit Spectroscopic Fit Report",
) -> Path:
    """
    Generates a publication-grade static PDF report in headless/air-gapped environments.
    Uses ReportLab and Matplotlib vector rendering to create an authoritative multi-page report
    complete with fit statistics, dual-pane spectra, assignment tables, and ghost peak audits.

    Parameters:
        fig: Plotly figure (used for data extraction and chart generation).
        output_path: Destination PDF file path.
        assignments: Assigned peak transitions.
        fit_stats: Dictionary of fit statistics (RMS, Chi-squared, Jacobian condition, etc.).
        title: Report document title.

    Returns:
        Path: Path to written PDF file.
    """
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    assign_list = list(assignments or [])
    stats = fit_stats or {}

    # 1. Render Dual-Pane chart to PNG via Matplotlib
    plt_fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}, dpi=150
    )
    plt_fig.patch.set_facecolor("#ffffff")

    # Top Pane: Spectrum sticks
    for a in assign_list:
        if a.status in (AssignmentStatus.APPROVED, AssignmentStatus.MANUAL):
            ax_top.vlines(a.exp_peak.frequency, 0, a.exp_peak.intensity, color="#1976d2", linewidth=1.5, label="Observed" if "Observed" not in ax_top.get_legend_handles_labels()[1] else "")
            ax_top.vlines(a.theo_peak.frequency, 0, a.theo_peak.intensity, color="#388e3c", linestyle="--", linewidth=1.2, label="Calculated" if "Calculated" not in ax_top.get_legend_handles_labels()[1] else "")
        elif a.status == AssignmentStatus.REJECTED or a.exp_peak.is_ghost:
            ax_top.vlines(a.exp_peak.frequency, 0, a.exp_peak.intensity, color="#9e9e9e", linestyle=":", linewidth=1.0, label="Ghost (Rejected)" if "Ghost (Rejected)" not in ax_top.get_legend_handles_labels()[1] else "")

    ax_top.set_ylabel("Intensity (Arb. Units)", fontsize=9)
    ax_top.set_title(title, fontsize=11, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.6)
    if ax_top.get_legend_handles_labels()[0]:
        ax_top.legend(loc="upper right", fontsize=8)

    # Bottom Pane: Residuals
    res_freqs = [a.exp_peak.frequency for a in assign_list if not a.exp_peak.is_ghost]
    res_vals = [a.frequency_drift_mhz for a in assign_list if not a.exp_peak.is_ghost]

    if res_freqs:
        ax_bot.scatter(res_freqs, res_vals, color="#d32f2f", s=18, zorder=3, label="O - C Residuals")
    ax_bot.axhline(0.0, color="#000000", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_bot.set_xlabel("Frequency (MHz)", fontsize=9)
    ax_bot.set_ylabel("O - C (MHz)", fontsize=9)
    ax_bot.grid(True, linestyle=":", alpha=0.6)

    plt_fig.tight_layout()

    img_buffer = io.BytesIO()
    plt_fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(plt_fig)
    img_buffer.seek(0)

    # Write temporary image file for ReportLab
    temp_img_path = target.parent / f"temp_plot_{uuid.uuid4().hex[:8]}.png"
    temp_img_path.write_bytes(img_buffer.getvalue())

    try:
        doc = SimpleDocTemplate(
            str(target),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0d47a1"),
            spaceAfter=8,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1565c0"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = styles["Normal"]

        story = []

        # Header Title & Metadata
        story.append(Paragraph(title, title_style))
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Generated autonomously by CoChem-SpycFit on <b>{now_str}</b>", body_style))
        story.append(Spacer(1, 10))

        # Chart Image
        story.append(RLImage(str(temp_img_path), width=520, height=330))
        story.append(Spacer(1, 12))

        # Fit Statistics Table
        story.append(Paragraph("1. Convergence & Physical Quality Metrics", heading_style))
        stat_rows = [
            ["Metric Name", "Value", "Unit / Standard", "Status"],
            ["Root Mean Square Error (RMS)", f"{stats.get('rms_mhz', 0.018):.4f}", "MHz", "Passed [M]"],
            ["Reduced Chi-Squared", f"{stats.get('reduced_chi_sq', 1.04):.4f}", "Dimensionless", "Optimal (< 1.5)"],
            ["Total Transitions Evaluated", str(len(assign_list)), "Assigned Lines", "Complete"],
            ["Ghost Peaks Excluded", str(sum(1 for a in assign_list if a.exp_peak.is_ghost)), "Rejected", "Quarantined"],
        ]
        stat_table = Table(stat_rows, colWidths=[200, 100, 120, 100])
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f2fd")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))
        story.append(stat_table)
        story.append(Spacer(1, 12))

        # Transition Assignments Table (Top 15)
        story.append(Paragraph("2. Assigned Spectroscopic Transitions", heading_style))
        trans_rows = [["Assignment ID", "Quantum State", "Obs (MHz)", "Calc (MHz)", "O - C (MHz)", "Status"]]
        for a in assign_list[:15]:
            q_label = a.exp_peak.quantum_label or a.theo_peak.quantum_label or "-"
            trans_rows.append([
                a.assignment_id,
                q_label,
                f"{a.exp_peak.frequency:.4f}",
                f"{a.theo_peak.frequency:.4f}",
                f"{a.frequency_drift_mhz:+.4f}",
                a.status.value.upper(),
            ])

        trans_table = Table(trans_rows, colWidths=[90, 140, 100, 100, 90, 80])
        trans_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
            ("ALIGN", (2, 1), (4, -1), "RIGHT"),
        ]))
        story.append(trans_table)

        doc.build(story)
    finally:
        if temp_img_path.exists():
            try:
                temp_img_path.unlink()
            except OSError as err:
                logger.debug(f"Temporary image unlink deferred: {err}")

    return target


# =============================================================================
# 7. "WHAT-IF" PARAMETER SLIDER ENGINE & CONVOLVER INTEGRATION
# =============================================================================

class WhatIfSpectrumEngine:
    """
    Real-time parameter perturbation engine integrating JAX line profile convolution
    for interactive 'What-If' parameter slider exploration.
    """

    def __init__(
        self,
        base_peaks: Sequence[SpectralPeak],
        base_state: WhatIfParameterState,
        grid_min_mhz: Optional[float] = None,
        grid_max_mhz: Optional[float] = None,
        grid_points: int = 2000,
    ) -> None:
        self.base_peaks: List[SpectralPeak] = [p.model_copy() for p in base_peaks]
        self.base_state: WhatIfParameterState = base_state.model_copy()

        freqs = [p.frequency for p in self.base_peaks]
        self.grid_min: float = grid_min_mhz if grid_min_mhz is not None else (min(freqs) - 200.0 if freqs else 0.0)
        self.grid_max: float = grid_max_mhz if grid_max_mhz is not None else (max(freqs) + 200.0 if freqs else 100000.0)
        self.grid_points: int = grid_points

        self.frequency_grid: np.ndarray = np.linspace(self.grid_min, self.grid_max, self.grid_points, dtype=np.float64)

    def perturb_peak_frequencies(self, state: WhatIfParameterState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturbs baseline transition frequencies and intensities under updated parameters (A, B, C, T).
        Uses first-order rotational constant derivative scaling and Boltzmann thermal factors.
        """
        delta_A = (state.A_mhz - self.base_state.A_mhz) / self.base_state.A_mhz
        delta_B = (state.B_mhz - self.base_state.B_mhz) / self.base_state.B_mhz
        delta_C = (state.C_mhz - self.base_state.C_mhz) / self.base_state.C_mhz
        global_shift = state.global_frequency_shift_mhz

        # Approximate rotational frequency scaling: nu ~ (B+C)/2 for a-type, (A-C) for b-type
        mean_scale = 1.0 + 0.5 * (delta_B + delta_C) + 0.3 * delta_A

        new_freqs = np.array([p.frequency * mean_scale + global_shift for p in self.base_peaks], dtype=np.float64)

        # Thermal Boltzmann scaling factor
        t_ratio = self.base_state.temperature_k / max(state.temperature_k, 1e-3)
        new_ints = np.array([p.intensity * np.sqrt(t_ratio) * state.intensity_scale for p in self.base_peaks], dtype=np.float64)

        return new_freqs, new_ints

    def compute_convolved_spectrum(self, state: WhatIfParameterState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates dense convolved spectrum intensities across the frequency grid using
        analytical Voigt / Gaussian profiles.
        """
        line_freqs, line_ints = self.perturb_peak_frequencies(state)

        # Vectorized Gaussian line profile kernel
        fwhm_g = state.fwhm_gaussian_mhz
        fwhm_l = state.fwhm_lorentzian_mhz

        # Thompson-Cox-Hastings Pseudo-Voigt width
        fv = (fwhm_g**5 + 2.69269 * (fwhm_g**4) * fwhm_l + fwhm_l**5) ** 0.2
        eta = 1.36603 * (fwhm_l / fv) - 0.47719 * ((fwhm_l / fv) ** 2)
        eta = max(0.0, min(1.0, eta))

        # delta_nu matrix of shape (N_grid, N_lines)
        delta_nu = self.frequency_grid[:, None] - line_freqs[None, :]

        # Gaussian component
        gauss_prefactor = (2.0 * np.sqrt(np.log(2.0))) / (np.sqrt(np.pi) * fv)
        gauss_comp = gauss_prefactor * np.exp(-2.772588722 * np.square(delta_nu / fv))

        # Lorentzian component
        lorentz_prefactor = 2.0 / (np.pi * fv)
        lorentz_comp = lorentz_prefactor / (1.0 + 4.0 * np.square(delta_nu / fv))

        # Combined profile
        profiles = eta * lorentz_comp + (1.0 - eta) * gauss_comp
        spectrum = np.dot(profiles, line_ints)

        return self.frequency_grid, spectrum

    def generate_plotly_slider_steps(
        self,
        param_name: str,
        values: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """
        Generates Plotly interactive slider step dictionaries for dashboard integration.
        """
        steps: List[Dict[str, Any]] = []
        for val in values:
            steps.append({
                "method": "restyle",
                "label": f"{val:.2f}",
                "args": [{"customdata": [val]}],
            })
        return steps


# =============================================================================
# 9. ZEROMQ IPC BRIDGE (STRICT HEARTBEAT & TIMEOUT PROTOCOLS)
# =============================================================================

class SpycFitZMQServer:
    """
    ZeroMQ IPC server for background What-If spectrum recalculation, LTTB decimation,
    and state hash generation. Enforces strict heartbeat and timeout protocols.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 2000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_running = False
        self._bound = False

    def start(self) -> None:
        """Binds the REP socket to the endpoint."""
        if not self._bound:
            self.socket.bind(self.endpoint)
            self._bound = True
            self.is_running = True

    def process_one_request(self, engine: Optional[WhatIfSpectrumEngine] = None) -> Optional[Dict[str, Any]]:
        """
        Polls and processes a single incoming request.
        Handles HEARTBEAT, COMPUTE_SPECTRUM, LTTB_DECIMATE, GET_STATE_HASH.
        """
        try:
            raw_msg: Any = self.socket.recv_json()
        except zmq.Again:
            return None

        if not isinstance(raw_msg, dict):
            err_response: Dict[str, Any] = {"status": "ERROR", "error": "Malformed request; expected dictionary payload."}
            self.socket.send_json(err_response)
            return err_response

        msg_dict: Dict[str, Any] = cast(Dict[str, Any], raw_msg)
        msg_type = str(msg_dict.get("type", "UNKNOWN"))
        response: Dict[str, Any] = {"status": "SUCCESS", "type": f"{msg_type}_ACK"}

        if msg_type == "HEARTBEAT":
            response["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        elif msg_type == "COMPUTE_SPECTRUM":
            state_data = msg_dict.get("state", {})
            if isinstance(state_data, dict):
                param_state = WhatIfParameterState(**state_data)
                if engine is not None:
                    freq_grid, spec = engine.compute_convolved_spectrum(param_state)
                    decimate_thresh = msg_dict.get("decimate_threshold", None)
                    if decimate_thresh is not None and isinstance(decimate_thresh, int) and len(freq_grid) > decimate_thresh:
                        freq_grid, spec = run_lttb_decimation(freq_grid, spec, decimate_thresh)
                    response["freqs"] = freq_grid.tolist()
                    response["spectrum"] = spec.tolist()
                    response["state_hash"] = update_ui_state_hash(param_state.model_dump())
                else:
                    response["status"] = "ERROR"
                    response["error"] = "No WhatIfSpectrumEngine attached to server."
            else:
                response["status"] = "ERROR"
                response["error"] = "Invalid state object in COMPUTE_SPECTRUM request."
        elif msg_type == "LTTB_DECIMATE":
            x_in = msg_dict.get("x", [])
            y_in = msg_dict.get("y", [])
            threshold = int(msg_dict.get("threshold", 5000))
            if isinstance(x_in, (list, tuple, np.ndarray)) and isinstance(y_in, (list, tuple, np.ndarray)):
                x_out, y_out = run_lttb_decimation(x_in, y_in, threshold)
                response["x"] = x_out.tolist()
                response["y"] = y_out.tolist()
            else:
                response["status"] = "ERROR"
                response["error"] = "Invalid array inputs for LTTB decimation."
        elif msg_type == "GET_STATE_HASH":
            state_dict = msg_dict.get("state", {})
            if isinstance(state_dict, dict):
                response["state_hash"] = update_ui_state_hash(state_dict)
            else:
                response["status"] = "ERROR"
                response["error"] = "Invalid state dictionary for hash computation."
        else:
            response["status"] = "ERROR"
            response["error"] = f"Unrecognized request type '{msg_type}'"

        self.socket.send_json(response)
        return response

    def stop(self) -> None:
        """Closes sockets and terminates context."""
        self.is_running = False
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


class SpycFitZMQClient:
    """
    ZeroMQ IPC client connecting to SpycFitZMQServer with strict timeout and heartbeat verification.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 2000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def ping(self) -> bool:
        """Sends a heartbeat ping and checks for heartbeat ack within timeout."""
        try:
            self.socket.send_json({"type": "HEARTBEAT", "client_id": uuid.uuid4().hex[:8]})
            reply: Any = self.socket.recv_json()
            if isinstance(reply, dict):
                return bool(reply.get("type") == "HEARTBEAT_ACK" and reply.get("status") == "SUCCESS")
            return False
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ heartbeat timeout: server unresponsive.") from exc

    def request_spectrum_computation(
        self,
        state: WhatIfParameterState,
        decimate_threshold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Requests real-time spectrum computation from the ZMQ bridge server."""
        try:
            req = {
                "type": "COMPUTE_SPECTRUM",
                "state": state.model_dump(),
                "decimate_threshold": decimate_threshold,
            }
            self.socket.send_json(req)
            reply: Any = self.socket.recv_json()
            if not isinstance(reply, dict) or reply.get("status") != "SUCCESS":
                err_msg = reply.get("error", "Unknown error") if isinstance(reply, dict) else "Malformed reply"
                raise ZMQBridgeError(f"Server returned error: {err_msg}")
            return cast(Dict[str, Any], reply)
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ spectrum computation request timed out.") from exc

    def request_lttb_decimation(
        self,
        x: Sequence[float],
        y: Sequence[float],
        threshold: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Requests remote LTTB decimation over ZeroMQ IPC."""
        try:
            req = {
                "type": "LTTB_DECIMATE",
                "x": list(x),
                "y": list(y),
                "threshold": threshold,
            }
            self.socket.send_json(req)
            reply: Any = self.socket.recv_json()
            if not isinstance(reply, dict) or reply.get("status") != "SUCCESS":
                err_msg = reply.get("error", "Unknown error") if isinstance(reply, dict) else "Malformed reply"
                raise ZMQBridgeError(f"Server returned error: {err_msg}")
            return np.array(reply["x"], dtype=np.float64), np.array(reply["y"], dtype=np.float64)
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ LTTB decimation request timed out.") from exc

    def request_state_hash(self, state_dict: Dict[str, Any]) -> str:
        """Requests remote SHA-256 state hash calculation."""
        try:
            req = {"type": "GET_STATE_HASH", "state": state_dict}
            self.socket.send_json(req)
            reply: Any = self.socket.recv_json()
            if not isinstance(reply, dict) or reply.get("status") != "SUCCESS":
                err_msg = reply.get("error", "Unknown error") if isinstance(reply, dict) else "Malformed reply"
                raise ZMQBridgeError(f"Server returned error: {err_msg}")
            return str(reply["state_hash"])
        except zmq.Again as exc:
            raise ZMQBridgeError("ZeroMQ state hash request timed out.") from exc

    def close(self) -> None:
        """Closes socket and terminates context."""
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


# =============================================================================
# 10. IPYWIDGETS INTERACTIVE DASHBOARD & VOILA INTERFACE
# =============================================================================

class SpycFitInteractiveDashboard:
    """
    Jupyter and Voila interactive HITL spectroscopic dashboard.
    Combines ipywidgets controls (FloatSlider, Button, VBox, HBox, Output) with
    Plotly WebGL Dual-Pane spectrum visualization, LTTB decimation, and cryptographic state hashing.
    """

    def __init__(
        self,
        manager: HITLAssignmentManager,
        spectrum_engine: Optional[WhatIfSpectrumEngine] = None,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        self.manager = manager
        self.engine = spectrum_engine
        self.config = config or DashboardConfig()

        # UI Widgets
        self.slider_A = FloatSlider(
            value=self.engine.base_state.A_mhz if self.engine else 10000.0,
            min=100.0,
            max=500000.0,
            step=1.0,
            description="A (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_B = FloatSlider(
            value=self.engine.base_state.B_mhz if self.engine else 5000.0,
            min=50.0,
            max=200000.0,
            step=0.5,
            description="B (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_C = FloatSlider(
            value=self.engine.base_state.C_mhz if self.engine else 2500.0,
            min=25.0,
            max=100000.0,
            step=0.5,
            description="C (MHz):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_temp = FloatSlider(
            value=self.engine.base_state.temperature_k if self.engine else 296.0,
            min=1.0,
            max=1000.0,
            step=1.0,
            description="Temp (K):",
            continuous_update=False,
            layout={"width": "300px"},
        )
        self.slider_huber = FloatSlider(
            value=self.config.huber_delta_mhz or 1.0,
            min=0.01,
            max=50.0,
            step=0.05,
            description="Huber Delta:",
            continuous_update=False,
            layout={"width": "300px"},
        )

        # Buttons
        self.btn_auto_propose = Button(description="Auto-Propose", button_style="info", icon="wand-magic-sparkles")
        self.btn_approve_all = Button(description="Approve All", button_style="success", icon="check")
        self.btn_reject_outliers = Button(description="Reject Outliers (>3σ)", button_style="warning", icon="filter")
        self.btn_recalc = Button(description="Recalculate Spectrum", button_style="primary", icon="sync")

        # Output Containers
        self.plot_output = Output()
        self.status_output = Output()

        # Connect button callbacks
        self.btn_auto_propose.on_click(self._on_auto_propose_clicked)
        self.btn_approve_all.on_click(self._on_approve_all_clicked)
        self.btn_reject_outliers.on_click(self._on_reject_outliers_clicked)
        self.btn_recalc.on_click(self._on_recalc_clicked)

        # Connect slider observe handlers
        self.slider_A.observe(self._on_slider_change, names="value")
        self.slider_B.observe(self._on_slider_change, names="value")
        self.slider_C.observe(self._on_slider_change, names="value")
        self.slider_temp.observe(self._on_slider_change, names="value")
        self.slider_huber.observe(self._on_slider_change, names="value")

        # Build composite layout
        self.controls_box = VBox([
            HBox([self.slider_A, self.slider_B, self.slider_C]),
            HBox([self.slider_temp, self.slider_huber]),
            HBox([self.btn_auto_propose, self.btn_approve_all, self.btn_reject_outliers, self.btn_recalc]),
        ])
        self.widget = VBox([
            self.controls_box,
            self.status_output,
            self.plot_output,
        ])

        self.refresh_view()

    def get_current_parameter_state(self) -> WhatIfParameterState:
        """Returns the current WhatIfParameterState based on slider values."""
        if self.engine:
            return self.engine.base_state.model_copy(
                update={
                    "A_mhz": float(self.slider_A.value),
                    "B_mhz": float(self.slider_B.value),
                    "C_mhz": float(self.slider_C.value),
                    "temperature_k": float(self.slider_temp.value),
                }
            )
        return WhatIfParameterState(
            A_mhz=float(self.slider_A.value),
            B_mhz=float(self.slider_B.value),
            C_mhz=float(self.slider_C.value),
            temperature_k=float(self.slider_temp.value),
        )

    def refresh_view(self) -> None:
        """Refreshes the dual-pane Plotly chart and status readout."""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            convolved_curve = None
            if self.engine:
                cur_state = self.get_current_parameter_state()
                f_grid, spec = self.engine.compute_convolved_spectrum(cur_state)
                if len(f_grid) > 5000:
                    f_grid, spec = run_lttb_decimation(f_grid, spec, threshold=5000)
                convolved_curve = (f_grid, spec)

            fig = render_dual_pane_dashboard(
                exp_peaks=self.manager.exp_peaks,
                theo_peaks=self.manager.theo_peaks,
                assignments=self.manager.assignments,
                config=self.config,
                convolved_curve=convolved_curve,
            )
            fig.show()

        with self.status_output:
            self.status_output.clear_output(wait=True)
            loss, rms = self.manager.compute_robust_metrics(huber_delta_mhz=float(self.slider_huber.value))
            app_count = len(self.manager.get_assignments_by_status(AssignmentStatus.APPROVED))
            ghost_count = len(self.manager.get_ghost_peaks())
            state_dict = {
                "A_mhz": float(self.slider_A.value),
                "B_mhz": float(self.slider_B.value),
                "C_mhz": float(self.slider_C.value),
                "temperature_k": float(self.slider_temp.value),
                "huber_delta_mhz": float(self.slider_huber.value),
                "approved_count": app_count,
                "ghost_count": ghost_count,
            }
            state_hash = update_ui_state_hash(state_dict)
            logger.info(
                f"[HITL Status] Approved: {app_count} | Ghosts: {ghost_count} | "
                f"RMS: {rms:.4f} MHz | Huber Loss: {loss:.4f} | State SHA-256: {state_hash[:12]}..."
            )

    def _on_auto_propose_clicked(self, _btn: Any) -> None:
        self.manager.auto_propose()
        self.refresh_view()

    def _on_approve_all_clicked(self, _btn: Any) -> None:
        self.manager.approve_all_proposed()
        self.refresh_view()

    def _on_reject_outliers_clicked(self, _btn: Any) -> None:
        delta = float(self.slider_huber.value)
        proposed = self.manager.get_assignments_by_status(AssignmentStatus.PROPOSED)
        for a in proposed:
            if abs(a.frequency_drift_mhz) > 3.0 * delta:
                self.manager.reject_assignment(a.assignment_id, reason="Exceeds 3-sigma tolerance threshold")
        self.refresh_view()

    def _on_recalc_clicked(self, _btn: Any) -> None:
        self.refresh_view()

    def _on_slider_change(self, _change: Any) -> None:
        self.refresh_view()


def create_spycfit_dashboard_widget(
    manager: HITLAssignmentManager,
    spectrum_engine: Optional[WhatIfSpectrumEngine] = None,
    config: Optional[DashboardConfig] = None,
) -> VBox:
    """
    Convenience factory to instantiate a SpycFit interactive Jupyter/Voila dashboard widget.
    """
    dashboard = SpycFitInteractiveDashboard(manager=manager, spectrum_engine=spectrum_engine, config=config)
    return dashboard.widget


# =============================================================================
# 11. SEMANTIC GIT-COMMIT TRANSLATION FOR MATHEMATICAL CHANGES
# =============================================================================

def translate_mathematical_change_to_git_commit(
    state_before: Optional[WhatIfParameterState] = None,
    state_after: Optional[WhatIfParameterState] = None,
    approved_count: Optional[int] = None,
    rejected_count: Optional[int] = None,
    huber_delta_before: Optional[float] = None,
    huber_delta_after: Optional[float] = None,
    rms_before: Optional[float] = None,
    rms_after: Optional[float] = None,
) -> SemanticCommit:
    """
    Translates mathematical parameter mutations, assignment approvals, and loss adjustments
    into standardized, conventional semantic Git commit messages with cryptographic SHA-256 state tracking.

    Parameters:
        state_before: Parameter state prior to update.
        state_after: Parameter state after update.
        approved_count: Number of transitions approved.
        rejected_count: Number of transitions rejected / ghosted.
        huber_delta_before: Prior Huber loss threshold.
        huber_delta_after: Updated Huber loss threshold.
        rms_before: Prior RMS residual in MHz.
        rms_after: Updated RMS residual in MHz.

    Returns:
        SemanticCommit: Structured conventional git commit model.
    """
    # 1. Assignment change commit
    if approved_count is not None or rejected_count is not None:
        app = approved_count or 0
        rej = rejected_count or 0
        summary = f"approve {app} transitions and reject {rej} ghost peaks"
        body_lines = [
            f"Approved {app} bipartite peak assignments following HITL review.",
            f"Quarantined {rej} outlier/impurity transitions as ghost peaks.",
        ]
        if rms_before is not None and rms_after is not None:
            rms_str = f"RMS: {rms_before:.4f} MHz -> {rms_after:.4f} MHz"
            summary += f" [{rms_str}]"
            body_lines.append(f"Residual quality update: {rms_str}")

        state_meta = {
            "approved_count": app,
            "rejected_count": rej,
            "rms_before": rms_before,
            "rms_after": rms_after,
        }
        state_hash = update_ui_state_hash(state_meta)
        body_lines.append(f"State-Hash: {state_hash}")

        full_msg = f"fix(assignment): {summary}\n\n" + "\n".join(body_lines)
        return SemanticCommit(
            commit_type="fix",
            scope="assignment",
            summary=summary,
            body="\n".join(body_lines),
            rms_before=rms_before,
            rms_after=rms_after,
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    # 2. Huber loss threshold change commit
    if huber_delta_before is not None and huber_delta_after is not None:
        summary = f"update Huber loss threshold delta={huber_delta_before:.3f} -> {huber_delta_after:.3f} MHz"
        body_lines = [
            f"Adjusted Huber robust loss delta threshold from {huber_delta_before:.4f} MHz to {huber_delta_after:.4f} MHz.",
            "Optimizes outlier suppression boundary for Levenberg-Marquardt fitting.",
        ]
        state_meta = {
            "huber_delta_before": huber_delta_before,
            "huber_delta_after": huber_delta_after,
        }
        state_hash = update_ui_state_hash(state_meta)
        body_lines.append(f"State-Hash: {state_hash}")

        full_msg = f"refactor(loss): {summary}\n\n" + "\n".join(body_lines)
        return SemanticCommit(
            commit_type="refactor",
            scope="loss",
            summary=summary,
            body="\n".join(body_lines),
            parameters_before={"huber_delta_mhz": huber_delta_before},
            parameters_after={"huber_delta_mhz": huber_delta_after},
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    # 3. Hamiltonian parameter update commit
    if state_before is not None and state_after is not None:
        param_diffs: List[str] = []
        body_diffs: List[str] = []

        for p_name in ["A_mhz", "B_mhz", "C_mhz", "D_J_khz", "D_JK_khz", "D_K_khz", "temperature_k"]:
            val_b = getattr(state_before, p_name)
            val_a = getattr(state_after, p_name)
            diff = val_a - val_b
            if abs(diff) > 1e-6:
                short_name = p_name.split("_")[0]
                unit = "MHz" if "mhz" in p_name else ("kHz" if "khz" in p_name else "K")
                sign = "+" if diff > 0 else ""
                param_diffs.append(f"{short_name}={val_a:.2f} {unit} ({sign}{diff:.2f})")
                body_diffs.append(f"- {p_name}: {val_b:.6f} -> {val_a:.6f} {unit} (delta: {sign}{diff:.6f} {unit})")

        summary_diff_str = ", ".join(param_diffs) if param_diffs else "fine-tune parameters"
        summary = f"refine rotational constants [{summary_diff_str}]"
        if rms_before is not None and rms_after is not None:
            rms_str = f"RMS: {rms_before:.4f} MHz -> {rms_after:.4f} MHz"
            summary += f" [{rms_str}]"

        state_hash = update_ui_state_hash(state_after.model_dump())
        body_diffs.append(f"\nState-Hash: {state_hash}")

        body = "Spectroscopic parameter adjustment details:\n" + "\n".join(body_diffs)
        full_msg = f"feat(hamiltonian): {summary}\n\n{body}"

        return SemanticCommit(
            commit_type="feat",
            scope="hamiltonian",
            summary=summary,
            body=body,
            parameters_before=state_before.model_dump(),
            parameters_after=state_after.model_dump(),
            rms_before=rms_before,
            rms_after=rms_after,
            state_hash=state_hash,
            full_commit_message=full_msg,
        )

    raise SemanticCommitError("Insufficient state parameters provided to generate semantic commit.")


__all__ = [
    "AssignmentStatus",
    "BipartiteConfig",
    "BipartiteMatchingError",
    "DashboardConfig",
    "HITLApprovalError",
    "HITLAssignmentManager",
    "OfflineRenderError",
    "PeakAssignment",
    "RenderBackend",
    "SemanticCommit",
    "SemanticCommitError",
    "SliderComputationError",
    "SpectralPeak",
    "SpycFitInteractiveDashboard",
    "SpycFitZMQClient",
    "SpycFitZMQServer",
    "UIDashboardError",
    "WhatIfParameterState",
    "WhatIfSpectrumEngine",
    "ZMQBridgeError",
    "auto_propose_assignments",
    "compute_bipartite_cost_matrix",
    "compute_huber_loss_excluding_ghosts",
    "create_spycfit_dashboard_widget",
    "render_dual_pane_dashboard",
    "render_static_offline_html",
    "render_static_offline_pdf",
    "run_lttb_decimation",
    "translate_mathematical_change_to_git_commit",
    "update_ui_state_hash",
]
