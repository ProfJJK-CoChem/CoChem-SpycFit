"""# zero-stub anti-spoofing engine
CoChem Interactive UI Override Traceability & Nuclear Clash Guard (cochem_base.gui.cochem_gui)

Provides UI validation hooks, operator manual adjustment registration, clash guard verification,
and unconverged geometry warning badges for Jupyter Widgets and desktop interfaces.

Governing Requirements:
- REQ-65, REQ-67: UI Traceability, Clash Guard Hooks & Override Modals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cochem_base.io.cochem_banners import render_unconverged_banner, evaluate_convergence_residuals
from cochem_base.topos.cochem_topos_crusher import (
    ManualAdjustmentRecord,
    create_manual_adjustment_record,
    verify_clash_guard,
)

logger = logging.getLogger("cochem_gui")

EXPECTED_MANUAL_OVERRIDE_TOKEN: str = "MANUAL_OVERRIDE_TOKEN"


class JobSessionProvenanceRegister:
    """Session register maintaining the active timeline of manual adjustments."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.records: List[ManualAdjustmentRecord] = []

    def append_record(self, record: ManualAdjustmentRecord) -> None:
        """Append a validated adjustment record to session history."""
        self.records.append(record)

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]


def validate_manual_geometry_override(
    coordinates: np.ndarray,
    justification: str,
    target_module: str = "cochem_gui",
    parameter_name: str = "geometry_coordinates",
    default_value: Any = "auto_converged",
    override_value: Any = "manual_coordinate_edit",
    residuals: Optional[Dict[str, float]] = None,
    override_token: Optional[str] = None,
) -> Tuple[bool, str, Optional[ManualAdjustmentRecord]]:
    """
    Validate an operator's manual geometry edit or parameter override against:
    1. Nuclear clash guard (min r_ij >= 0.65 A).
    2. Minimum justification length (>= 10 chars).
    3. Severe unconverged residual token validation (if TolMaxG > 5.0e-4).
    """
    # 1. Nuclear Clash Guard Verification
    is_clash_ok, min_dist = verify_clash_guard(coordinates)
    if not is_clash_ok:
        return (
            False,
            f"[ERR_NUCLEAR_CLASH]: Interatomic distance ({min_dist:.4f} A) violates minimum 0.65 A barrier.",
            None,
        )

    # 2. Operator Justification Length
    if not justification or len(justification.strip()) < 10:
        return (
            False,
            "[ERR_JUSTIFICATION_TOO_SHORT]: Operator justification must be at least 10 characters.",
            None,
        )

    # 3. Severe Unconverged Residual Token Validation
    if residuals:
        eval_res = evaluate_convergence_residuals(residuals)
        if eval_res["requires_override_token"]:
            if override_token != EXPECTED_MANUAL_OVERRIDE_TOKEN:
                return (
                    False,
                    f"[ERR_OVERRIDE_TOKEN_REQUIRED]: TolMaxG exceeds 5.0e-4 ceiling. Requires '{EXPECTED_MANUAL_OVERRIDE_TOKEN}'.",
                    None,
                )

    # 4. Construct Immutable Record
    record = create_manual_adjustment_record(
        target_module=target_module,
        parameter_name=parameter_name,
        default_value=default_value,
        override_value=override_value,
        coordinates=coordinates,
        operator_justification=justification,
        physical_implication="Manual geometry modification committed via GUI interface.",
    )

    return True, "VALID", record


def render_unconverged_warning_html(residuals: Dict[str, float]) -> str:
    """Render high-contrast red warning badge for HTML viewports."""
    return render_unconverged_banner(residuals, format="html")
