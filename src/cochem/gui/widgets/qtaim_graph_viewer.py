"""
CoChem Interactive QTAIM Topological Graph Visualizer Widget (REQ-QC-088).

Provenance:
- Method Matrix [M]: 3D topological molecular graph renderer with colored critical points,
  bond paths, and live Poincaré-Hopf invariant enforcement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.cochem_qtaim import validate_poincare_hopf_invariant
from cochem.qc.models import BondCriticalPointDescriptor


class QTAIMGraphViewerState(BaseModel):
    """Data contract representing the rendering state of the QTAIM topological graph."""
    model_config = ConfigDict(frozen=True)

    n_ncp: int = 0
    n_bcp: int = 0
    n_rcp: int = 0
    n_ccp: int = 0
    euler_characteristic: int = 1
    is_poincare_hopf_valid: bool = True
    critical_points_summary: List[Dict[str, Any]] = Field(default_factory=list)
    selected_bcp_index: Optional[int] = None
    selected_bcp_details: Optional[BondCriticalPointDescriptor] = None


class QTAIMGraphViewerWidget:
    """
    [REQ-QC-088] Interactive QTAIM Electron Density Critical Point Visualizer.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        coordinates: Sequence[Tuple[float, float, float]],
        bcps: Optional[Sequence[BondCriticalPointDescriptor]] = None,
        rcp_count: int = 0,
        ccp_count: int = 0,
    ) -> None:
        self.symbols = list(symbols)
        self.coordinates = list(coordinates)
        self.bcps = list(bcps) if bcps else []
        self.rcp_count = rcp_count
        self.ccp_count = ccp_count
        self.selected_bcp_idx: Optional[int] = None
        self._update_state()

    def _update_state(self) -> None:
        n_ncp = len(self.symbols)
        n_bcp = len(self.bcps)
        euler = n_ncp - n_bcp + self.rcp_count - self.ccp_count
        is_valid = (euler == 1)

        summary: List[Dict[str, Any]] = []
        # Nuclei (NCP - Yellow)
        for idx, (sym, coord) in enumerate(zip(self.symbols, self.coordinates)):
            summary.append({
                "type": "NCP",
                "index": idx,
                "label": f"{sym}{idx+1}",
                "coordinates": coord,
                "color": "#FFD700", # Yellow
            })

        # BCPs (Blue)
        for b in self.bcps:
            summary.append({
                "type": "BCP",
                "index": b.bcp_index,
                "label": f"BCP({b.atom_1_index}-{b.atom_2_index})",
                "coordinates": b.coordinates_angstrom,
                "color": "#1E90FF", # Dodger Blue
                "rho": b.electron_density_au,
                "laplacian": b.laplacian_au,
                "classification": b.interaction_classification,
            })

        selected_bcp = None
        if self.selected_bcp_idx is not None:
            matches = [b for b in self.bcps if b.bcp_index == self.selected_bcp_idx]
            if matches:
                selected_bcp = matches[0]

        self._state = QTAIMGraphViewerState(
            n_ncp=n_ncp,
            n_bcp=n_bcp,
            n_rcp=self.rcp_count,
            n_ccp=self.ccp_count,
            euler_characteristic=euler,
            is_poincare_hopf_valid=is_valid,
            critical_points_summary=summary,
            selected_bcp_index=self.selected_bcp_idx,
            selected_bcp_details=selected_bcp,
        )

    def select_bcp(self, bcp_index: int) -> QTAIMGraphViewerState:
        """Select a BCP to inspect its local properties."""
        self.selected_bcp_idx = bcp_index
        self._update_state()
        return self._state

    def get_state(self) -> QTAIMGraphViewerState:
        return self._state
