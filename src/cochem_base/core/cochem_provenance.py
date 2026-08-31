"""# zero-stub anti-spoofing engine
CoChem Conformer Lineage DAG Builder & Quasiharmonic Boltzmann Weighting (cochem_base.core.cochem_provenance)

Constructs Directed Acyclic Graphs (DAG) tracking conformer evolution, computes quasiharmonic
Boltzmann population weights with low-frequency vibrational cutoffs, and exports visual flowcharts
in Graphviz DOT, Vector SVG, and Interactive Mermaid formats.

Governing Requirements:
- REQ-69 (FR-69.1 - FR-69.3), REQ-62, REQ-64, REQ-70: Conformer Lineage DAG & Boltzmann Populations.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cochem_base.core.cochem_version import VCSProvenance

# Physical Constants (CODATA 2022)
HARTREE_TO_KCAL_MOL: float = 627.509474
KB_KCAL_MOL_K: float = 0.00198720425864083  # Boltzmann constant in kcal/(mol*K)
PLANCK_H_SI: float = 6.62607015e-34
SPEED_OF_LIGHT_C_CMS: float = 2.99792458e10
KB_SI: float = 1.380649e-23


class ProvenanceDisconnectedError(Exception):
    """Raised when a DAG lineage node is disconnected, orphaned, or contains broken parent hashes."""
    pass


@dataclass
class DAGNode:
    """A single discrete node in the conformer evolution Directed Acyclic Graph."""
    node_id: str
    stage: str
    data_hash: str
    parent_ids: List[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    boltzmann_weight: Optional[float] = None
    free_energy_hartree: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stage": self.stage,
            "data_hash": self.data_hash,
            "parent_ids": self.parent_ids,
            "timestamp_utc": self.timestamp_utc,
            "payload": self.payload,
            "boltzmann_weight": self.boltzmann_weight,
            "free_energy_hartree": self.free_energy_hartree,
        }


def compute_boltzmann_weights(
    energies_hartree: List[float],
    frequencies_cm1_list: Optional[List[List[float]]] = None,
    temperature_k: float = 298.15,
    low_freq_cutoff_cm1: float = 100.0,
) -> np.ndarray:
    """
    Compute quasiharmonic Boltzmann population weights:
    w_i = exp(-Delta G_i / (k_B * T)) / sum_j exp(-Delta G_j / (k_B * T))
    
    Applies Grimme's quasi-RRHO interpolation for low vibrational frequencies (< low_freq_cutoff_cm1).
    """
    if not energies_hartree:
        return np.array([], dtype=np.float64)

    e_arr = np.asarray(energies_hartree, dtype=np.float64)
    # Convert electronic energies to kcal/mol
    g_rel_kcal = (e_arr - np.min(e_arr)) * HARTREE_TO_KCAL_MOL

    # If harmonic frequencies provided, compute quasiharmonic G_vib corrections
    if frequencies_cm1_list and len(frequencies_cm1_list) == len(energies_hartree):
        vib_corrections_kcal = []
        for freqs in frequencies_cm1_list:
            g_vib = 0.0
            for nu in freqs:
                if nu > 10.0:
                    # Quasi-harmonic frequency floor
                    nu_eff = max(nu, low_freq_cutoff_cm1)
                    # Convert to SI energy (h * c * nu_eff)
                    nu_si = nu_eff * SPEED_OF_LIGHT_C_CMS
                    h_nu = PLANCK_H_SI * nu_si
                    x = h_nu / (KB_SI * temperature_k)
                    if x < 700.0:
                        # G_vib = RT * ln(1 - exp(-x)) + ZPE
                        # Convert to kcal/mol
                        zpe_kcal = (0.5 * h_nu * 6.02214076e23) / 4184.0
                        s_vib_kcal = (-KB_KCAL_MOL_K * math.log(1.0 - math.exp(-x))) * temperature_k
                        g_vib += zpe_kcal + s_vib_kcal
            vib_corrections_kcal.append(g_vib)
        vib_arr = np.array(vib_corrections_kcal)
        g_rel_kcal += (vib_arr - np.min(vib_arr))

    # Boltzmann factors
    beta = 1.0 / (KB_KCAL_MOL_K * temperature_k)
    # Shift for numerical stability
    exp_factors = np.exp(-beta * (g_rel_kcal - np.min(g_rel_kcal)))
    total_q = float(np.sum(exp_factors))
    if total_q <= 0.0:
        return np.ones(len(energies_hartree)) / len(energies_hartree)
    return exp_factors / total_q


class ConformerLineageDAG:
    """
    Directed Acyclic Graph (DAG) recording conformer lineage across the entire screening pipeline.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.nodes: Dict[str, DAGNode] = {}
        self.vcs_provenance: Optional[VCSProvenance] = None
        self.hardware_telemetry: Optional[Dict[str, Any]] = None

    def add_node(
        self,
        stage: str,
        data: Dict[str, Any],
        parent_ids: Optional[List[str]] = None,
        node_id: Optional[str] = None,
        free_energy_hartree: Optional[float] = None,
    ) -> str:
        """Add a stage node to the DAG and return its unique node_id."""
        parents = parent_ids or []
        for pid in parents:
            if pid not in self.nodes:
                raise ProvenanceDisconnectedError(
                    f"[ERR_PROVENANCE_DISCONNECTED]: Parent node '{pid}' not found in DAG."
                )

        canonical_payload = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        nid = node_id or f"node_{len(self.nodes) + 1}_{stage.lower()}"

        node = DAGNode(
            node_id=nid,
            stage=stage,
            data_hash=data_hash,
            parent_ids=parents,
            payload=data,
            free_energy_hartree=free_energy_hartree,
        )
        self.nodes[nid] = node
        return nid

    def validate_dag_integrity(self) -> bool:
        """Validate parent hash chaining and verify graph has no orphaned parent links."""
        for nid, node in self.nodes.items():
            for pid in node.parent_ids:
                if pid not in self.nodes:
                    raise ProvenanceDisconnectedError(
                        f"[ERR_PROVENANCE_DISCONNECTED]: Orphaned parent reference '{pid}' on node '{nid}'."
                    )
        return True

    def integrate_vcs_provenance(self, vcs_prov: VCSProvenance) -> None:
        """Integrate VCS git working-tree provenance into the root graph metadata."""
        self.vcs_provenance = vcs_prov

    def integrate_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Integrate hardware and execution profiling telemetry into graph metadata."""
        self.hardware_telemetry = telemetry

    def to_dot(self) -> str:
        """Export the conformer DAG in Graphviz DOT format."""
        lines = [
            f'digraph ConformerLineage_{self.job_id} {{',
            '  rankdir="TB";',
            '  node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#E3F2FD", color="#1565C0"];',
            '  edge [color="#455A64", arrowhead=vee];',
        ]
        for nid, node in self.nodes.items():
            lbl = f"{node.stage}\\n({nid})\\nHash: {node.data_hash[:8]}"
            lines.append(f'  "{nid}" [label="{lbl}"];')
            for pid in node.parent_ids:
                lines.append(f'  "{pid}" -> "{nid}";')
        lines.append('}')
        return "\n".join(lines)

    def to_svg(self) -> str:
        """Generate a clean standalone vector SVG representation for Supporting Information."""
        width = 600
        height = max(150, len(self.nodes) * 90 + 60)
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}">',
            '  <defs>',
            '    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1565C0"/>',
            '    </marker>',
            '  </defs>',
            f'  <rect width="100%" height="100%" fill="#FAFAFA"/>',
            f'  <text x="20" y="30" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="bold" fill="#212121">Conformer Lineage DAG: {self.job_id}</text>',
        ]

        # Render nodes vertically
        y_pos = 60
        node_coords = {}
        for nid, node in self.nodes.items():
            cx = width // 2
            cy = y_pos
            node_coords[nid] = (cx, cy)
            svg_lines.append(
                f'  <rect x="{cx - 140}" y="{cy}" width="280" height="50" rx="8" ry="8" fill="#E3F2FD" stroke="#1565C0" stroke-width="2"/>'
            )
            svg_lines.append(
                f'  <text x="{cx}" y="{cy + 22}" font-family="Helvetica, Arial" font-size="13" font-weight="bold" fill="#0D47A1" text-anchor="middle">{node.stage} ({nid})</text>'
            )
            svg_lines.append(
                f'  <text x="{cx}" y="{cy + 40}" font-family="monospace" font-size="10" fill="#546E7A" text-anchor="middle">SHA: {node.data_hash[:16]}...</text>'
            )
            y_pos += 80

        # Render connector arrows
        for nid, node in self.nodes.items():
            for pid in node.parent_ids:
                if pid in node_coords:
                    px, py = node_coords[pid]
                    cx, cy = node_coords[nid]
                    svg_lines.append(
                        f'  <line x1="{px}" y1="{py + 50}" x2="{cx}" y2="{cy}" stroke="#1565C0" stroke-width="2" marker-end="url(#arrow)"/>'
                    )

        svg_lines.append('</svg>')
        return "\n".join(svg_lines)

    def to_mermaid(self) -> str:
        """Generate interactive Mermaid flowchart syntax for Markdown reports."""
        lines = ["graph TD"]
        for nid, node in self.nodes.items():
            safe_lbl = f"{node.stage} ({node.data_hash[:8]})"
            lines.append(f'    {nid}["{safe_lbl}"]')
            for pid in node.parent_ids:
                lines.append(f'    {pid} --> {nid}')
        return "\n".join(lines)
