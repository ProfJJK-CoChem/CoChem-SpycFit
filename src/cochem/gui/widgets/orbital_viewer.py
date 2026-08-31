"""
CoChem GUI High-Res Orbital Viewer & Renderer Configuration Widget (REQ-QC-094).

Provenance:
- Method Matrix [M]: UI controls for orbital selection, isovalue slider/spinbox, resolution presets,
  PyMOL/VMD backend detection, and asynchronous 300+ DPI render dispatch.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from cochem.visualization.cochem_orbital_renderer import (
    OrbitalFigureSet,
    OrbitalRenderJob,
    assemble_composite_orbital_figure,
    execute_headless_orbital_render,
)


class OrbitalViewerState(BaseModel):
    """Data contract representing the reactive state of the orbital viewer widget."""
    model_config = ConfigDict(frozen=True)

    selected_orbital_type: str = "HOMO"
    orbital_index: int = 1
    isovalue: float = 0.03
    resolution_dpi: int = 300
    backend_engine: str = "PyMOL"
    is_backend_available: bool = False
    cube_file_loaded: Optional[str] = None
    preview_image_path: Optional[str] = None
    status_message: str = "Ready"


class OrbitalViewerWidget:
    """
    [REQ-QC-094] GUI Orbital Viewer & Ray-Tracing Configuration Controller.
    """

    def __init__(self, cube_file_path: Optional[str] = None) -> None:
        self.cube_file_path = cube_file_path
        self.selected_orbital_type = "HOMO"
        self.orbital_index = 1
        self.isovalue = 0.03
        self.resolution_dpi = 300
        self.backend = "PyMOL"
        self._state = self._check_environment_and_update()

    def _check_environment_and_update(self) -> OrbitalViewerState:
        has_pymol = bool(shutil.which("pymol") or shutil.which("pymol.exe"))
        has_vmd = bool(shutil.which("vmd") or shutil.which("vmd.exe"))

        is_avail = has_pymol if self.backend == "PyMOL" else has_vmd
        status = "Engine ready" if is_avail else f"{self.backend} executable not found in PATH (PIL fallback enabled)"

        return OrbitalViewerState(
            selected_orbital_type=self.selected_orbital_type,
            orbital_index=self.orbital_index,
            isovalue=self.isovalue,
            resolution_dpi=self.resolution_dpi,
            backend_engine=self.backend,
            is_backend_available=is_avail,
            cube_file_loaded=self.cube_file_path,
            status_message=status,
        )

    def select_orbital(self, orbital_type: str, orbital_index: int) -> OrbitalViewerState:
        """Select orbital descriptor (HOMO, LUMO, HOMO-1, etc.) and index."""
        self.selected_orbital_type = orbital_type
        self.orbital_index = orbital_index
        self._state = self._check_environment_and_update()
        return self._state

    def set_isovalue(self, isovalue: float) -> OrbitalViewerState:
        """Adjust isosurface contour value (0.001 <= isovalue <= 0.20)."""
        if isovalue <= 0.0:
            raise ValueError("Isovalue must be strictly positive.")
        self.isovalue = float(isovalue)
        self._state = self._check_environment_and_update()
        return self._state

    def set_resolution_dpi(self, dpi: int) -> OrbitalViewerState:
        """Set resolution preset (e.g. 150 Draft, 300 Publication, 600 Cover Art)."""
        if dpi not in [150, 300, 600, 1200]:
            raise ValueError("Resolution must be one of [150, 300, 600, 1200] DPI.")
        self.resolution_dpi = dpi
        self._state = self._check_environment_and_update()
        return self._state

    def render_figure(
        self,
        output_png_path: str,
        energy_hartree: float = -0.25,
        energy_ev: float = -6.80,
    ) -> str:
        """
        Execute ray-traced rendering of the currently configured orbital.
        """
        if not self.cube_file_path:
            raise ValueError("Cannot render orbital figure: No volumetric .cube file loaded.")

        job = OrbitalRenderJob(
            orbital_index=self.orbital_index,
            orbital_type=self.selected_orbital_type,
            energy_hartree=energy_hartree,
            energy_ev=energy_ev,
            isovalue=self.isovalue,
            cube_file_path=self.cube_file_path,
            output_png_path=output_png_path,
            resolution_dpi=self.resolution_dpi,
        )

        out_path = execute_headless_orbital_render(job)
        self._state = OrbitalViewerState(
            selected_orbital_type=self.selected_orbital_type,
            orbital_index=self.orbital_index,
            isovalue=self.isovalue,
            resolution_dpi=self.resolution_dpi,
            backend_engine=self.backend,
            is_backend_available=self._state.is_backend_available,
            cube_file_loaded=self.cube_file_path,
            preview_image_path=str(out_path),
            status_message=f"Successfully rendered at {self.resolution_dpi} DPI to {out_path.name}",
        )
        return str(out_path)

    @property
    def state(self) -> OrbitalViewerState:
        return self._state
