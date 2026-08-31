"""
ACS Publication Standards Plotting API and Vector Rendering Engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt


class DockVisualsAPI:
    """
    Renders publication-grade 2D plots strictly compliant with American Chemical Society (ACS) guidelines.
    """

    @classmethod
    def apply_acs_style(cls, ax: plt.Axes) -> None:
        """Apply ACS formatting standards to matplotlib axes."""
        ax.tick_params(direction="in", length=4, width=1.0, top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")
        ax.grid(False)

    @classmethod
    def plot_pes_1d(
        cls,
        angles_deg: np.ndarray,
        energies_kcal: np.ndarray,
        title: str = "Torsional Potential Energy Surface",
        output_path: Optional[Path] = None,
        dpi: int = 600,
    ) -> plt.Figure:
        """Plot 1D torsional potential energy surface."""
        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=dpi)
        cls.apply_acs_style(ax)

        ax.plot(angles_deg, energies_kcal, color="#003366", lw=1.5, label="V(θ)")
        ax.scatter(angles_deg, energies_kcal, color="#CC0000", s=15, zorder=5)

        ax.set_xlabel("Dihedral Angle θ (deg)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 360)

        plt.tight_layout()

        if output_path:
            p = Path(output_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

        return fig

    @classmethod
    def plot_vibrational_spectrum(
        cls,
        frequencies_cm1: List[float],
        intensities: Optional[List[float]] = None,
        output_path: Optional[Path] = None,
        dpi: int = 600,
    ) -> plt.Figure:
        """Plot simulated harmonic vibrational spectrum."""
        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=dpi)
        cls.apply_acs_style(ax)

        freqs = np.array(frequencies_cm1, dtype=float)
        ints = np.array(intensities, dtype=float) if intensities else np.ones_like(freqs)

        ax.vlines(freqs, 0, ints, color="#8B0000", lw=1.2)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Intensity (km/mol)", fontsize=10, fontweight="bold")
        ax.set_title("Harmonic Vibrational Spectrum", fontsize=10, fontweight="bold")
        ax.set_xlim(max(4000, np.max(freqs) + 200), 0)  # Reversed IR convention

        plt.tight_layout()

        if output_path:
            p = Path(output_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

        return fig

    @classmethod
    def export_plot(cls, fig: plt.Figure, output_path: Path, dpi: int = 600) -> None:
        """Export matplotlib figure to file."""
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
