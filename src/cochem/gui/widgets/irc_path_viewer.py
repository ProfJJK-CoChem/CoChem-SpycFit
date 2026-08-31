"""
CoChem Interactive IRC Reaction Path Viewer Widget (REQ-QC-083).

Provenance:
- Method Matrix [M]: 2D Reaction coordinate profile E(s) vs s, 3D structure frame indexing,
  and Kabsch RMSD endpoint connectivity status badge.
- Derived Relations [D]: Energy conversions (Hartree -> kcal/mol) and reaction coordinate distances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from cochem.qc.cochem_irc import compute_kabsch_rmsd
from cochem.qc.models import IRCTrajectoryResult


class IRCPathViewerState(BaseModel):
    """Data contract representing the internal state of the IRC path visualizer."""
    model_config = ConfigDict(frozen=True)

    total_frames: int = 0
    current_frame_index: int = 0
    transition_state_frame_index: int = 0
    current_energy_hartree: float = 0.0
    relative_energy_kcal_mol: float = 0.0
    reaction_coordinate_s: float = 0.0
    is_connected: bool = True
    forward_rmsd_angstrom: Optional[float] = None
    reverse_rmsd_angstrom: Optional[float] = None
    status_badge: str = "NO_DATA"


class IRCPathViewerWidget:
    """
    [REQ-QC-083] Interactive IRC Trajectory and Energy Profile Viewer.
    """

    def __init__(self, trajectory: Optional[IRCTrajectoryResult] = None) -> None:
        self.trajectory = trajectory
        self.current_frame = 0
        if trajectory is not None:
            self._init_from_trajectory(trajectory)
        else:
            self._state = IRCPathViewerState()

    def _init_from_trajectory(self, traj: IRCTrajectoryResult) -> None:
        n_frames = len(traj.path_energies_hartree)
        ts_idx = traj.reverse_steps
        self.current_frame = ts_idx
        ts_e = traj.ts_energy_hartree

        cur_e = traj.path_energies_hartree[self.current_frame]
        rel_e_kcal = (cur_e - ts_e) * 627.509474

        badge = "CONNECTED" if traj.connectivity_verified else "DISCONNECTED"

        self._state = IRCPathViewerState(
            total_frames=n_frames,
            current_frame_index=self.current_frame,
            transition_state_frame_index=ts_idx,
            current_energy_hartree=round(cur_e, 6),
            relative_energy_kcal_mol=round(rel_e_kcal, 3),
            reaction_coordinate_s=0.0,
            is_connected=traj.connectivity_verified,
            forward_rmsd_angstrom=traj.forward_endpoint_rmsd_to_reactant_angstrom,
            reverse_rmsd_angstrom=traj.reverse_endpoint_rmsd_to_product_angstrom,
            status_badge=badge,
        )

    def load_trajectory(self, traj: IRCTrajectoryResult) -> IRCPathViewerState:
        """Load new IRC trajectory and update view state."""
        self.trajectory = traj
        self._init_from_trajectory(traj)
        return self._state

    def set_frame(self, frame_idx: int) -> IRCPathViewerState:
        """Seek to a specific frame along the IRC path."""
        if self.trajectory is None:
            return self._state

        max_frame = len(self.trajectory.path_energies_hartree) - 1
        clamped_idx = max(0, min(frame_idx, max_frame))
        self.current_frame = clamped_idx

        ts_e = self.trajectory.ts_energy_hartree
        cur_e = self.trajectory.path_energies_hartree[clamped_idx]
        rel_e_kcal = (cur_e - ts_e) * 627.509474

        s_coord = (clamped_idx - self.trajectory.reverse_steps) * 0.05

        self._state = IRCPathViewerState(
            total_frames=len(self.trajectory.path_energies_hartree),
            current_frame_index=clamped_idx,
            transition_state_frame_index=self.trajectory.reverse_steps,
            current_energy_hartree=round(cur_e, 6),
            relative_energy_kcal_mol=round(rel_e_kcal, 3),
            reaction_coordinate_s=round(s_coord, 4),
            is_connected=self.trajectory.connectivity_verified,
            forward_rmsd_angstrom=self.trajectory.forward_endpoint_rmsd_to_reactant_angstrom,
            reverse_rmsd_angstrom=self.trajectory.reverse_endpoint_rmsd_to_product_angstrom,
            status_badge="CONNECTED" if self.trajectory.connectivity_verified else "DISCONNECTED",
        )
        return self._state

    def get_state(self) -> IRCPathViewerState:
        return self._state
