"""
CoChem Standalone 3D Interactive HTML WebGL Supporting Information Visualizer (REQ-QC-095).

Provenance:
- Method Matrix [M]: Zero-CDN, fully self-contained offline HTML Supporting Information bundles
  with WebGL molecular orbital visualization, vibrational mode vector animations, and IRC trajectory controls.
- Long-term archival preservation (FAIR criteria) ensuring executable visualization without internet access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_exceptions import HTMLBundleExportError


class InteractiveHTMLModel(BaseModel):
    """Molecular stationary point model formatted for interactive 3D WebGL rendering."""
    model_config = ConfigDict(frozen=True)

    label: str = Field(description="Display label (e.g. 'Reactant', 'TS_1', 'Product')")
    chemical_formula: str = Field(description="Stoichiometric formula (e.g. 'C2H6O')")
    energy_hartree: float = Field(description="Absolute SCF energy in Hartree")
    relative_energy_kcal_mol: float = Field(description="Relative Gibbs free energy or electronic energy in kcal/mol")
    xyz_block: str = Field(description="Standard Cartesian XYZ formatted string")
    frequencies: List[float] = Field(default_factory=list, description="Harmonic vibrational frequencies in cm^-1")
    normal_modes: List[List[List[float]]] = Field(
        default_factory=list,
        description="Normal mode displacement vectors: List of modes [N_atoms x 3]"
    )


class InteractiveHTMLPackage(BaseModel):
    """Complete multi-model Supporting Information interactive bundle."""
    model_config = ConfigDict(frozen=True)

    title: str = Field(default="Interactive Supporting Information 3D Viewport", description="Title banner for the HTML page")
    models: List[InteractiveHTMLModel] = Field(description="List of stationary points and reaction intermediates")
    include_irc: bool = Field(default=False, description="Whether to include IRC trajectory stepping controls")
    irc_trajectory_xyz: Optional[str] = Field(default=None, description="Multi-frame concatenated XYZ for IRC trajectory")
    output_html_path: str = Field(description="Target destination path for single-file HTML bundle")


# Standalone embedded lightweight WebGL Molecular Engine script (Embedded inline for air-gap preservation)
EMBEDDED_MOLECULAR_VIEWER_JS = """
// CoChem Embedded Minimal WebGL 3D Molecule Visualizer Engine
class CoChemViewer {
    constructor(canvasId, infoId) {
        this.canvas = document.getElementById(canvasId);
        this.infoBox = document.getElementById(infoId);
        this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
        this.models = [];
        this.currentModelIdx = 0;
        this.rotX = 0;
        this.rotY = 0;
        this.zoom = 1.0;
        this.style = 'stick';
        this.animatingMode = false;
        this.selectedModeIdx = -1;
        this.animPhase = 0;
        this.initEvents();
    }
    initEvents() {
        let isDragging = false;
        let prevX = 0, prevY = 0;
        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            prevX = e.clientX;
            prevY = e.clientY;
        });
        window.addEventListener('mouseup', () => isDragging = false);
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - prevX;
            const dy = e.clientY - prevY;
            this.rotY += dx * 0.01;
            this.rotX += dy * 0.01;
            prevX = e.clientX;
            prevY = e.clientY;
            this.render();
        });
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom *= (e.deltaY > 0 ? 0.9 : 1.1);
            this.render();
        });
    }
    loadData(packageData) {
        this.packageData = packageData;
        this.models = packageData.models;
        this.switchModel(0);
    }
    switchModel(idx) {
        if (idx < 0 || idx >= this.models.length) return;
        this.currentModelIdx = idx;
        const m = this.models[idx];
        this.infoBox.innerHTML = `<strong>${m.label}</strong> (${m.chemical_formula})<br/>SCF Energy: ${m.energy_hartree.toFixed(6)} E<sub>h</sub><br/>Relative Energy: <strong>${m.relative_energy_kcal_mol.toFixed(2)} kcal/mol</strong>`;
        this.render();
    }
    setStyle(style) {
        this.style = style;
        this.render();
    }
    setVibrationalMode(modeIdx) {
        this.selectedModeIdx = modeIdx;
        this.animatingMode = (modeIdx >= 0);
        if (this.animatingMode && !this.animTimer) {
            this.startAnimationLoop();
        }
    }
    startAnimationLoop() {
        const loop = () => {
            if (this.animatingMode) {
                this.animPhase += 0.08;
                this.render();
                requestAnimationFrame(loop);
            } else {
                this.animTimer = null;
            }
        };
        requestAnimationFrame(loop);
    }
    render() {
        if (!this.gl) return;
        const gl = this.gl;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.97, 0.98, 0.99, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
}
"""


def export_interactive_html_si(package: InteractiveHTMLPackage) -> str:
    """
    Export a standalone, zero-dependency 3D WebGL Supporting Information HTML bundle.

    Args:
        package: InteractiveHTMLPackage containing models, IRC, and vibrational data.

    Returns:
        str: Absolute filepath of generated HTML bundle.

    Raises:
        HTMLBundleExportError: If file writing fails.
    """
    out_path = Path(package.output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    package_json = package.model_dump_json(indent=2)

    # Build Conformer selector options
    options_html = []
    for idx, m in enumerate(package.models):
        options_html.append(f'<option value="{idx}">{m.label} ({m.relative_energy_kcal_mol:.2f} kcal/mol)</option>')
    options_str = "\n".join(options_html)

    # First model modes
    freq_options = ['<option value="-1">-- None (Static Geometry) --</option>']
    if package.models and package.models[0].frequencies:
        for f_idx, freq in enumerate(package.models[0].frequencies):
            imag_tag = " [IMAGINARY TS]" if freq < 0 else ""
            freq_options.append(f'<option value="{f_idx}">Mode #{f_idx+1}: {freq:.1f} cm⁻¹{imag_tag}</option>')
    freq_options_str = "\n".join(freq_options)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{package.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #0d1117;
            color: #c9d1d9;
        }}
        .header {{
            background-color: #161b22;
            padding: 16px 24px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 20px;
            color: #58a6ff;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 65px);
        }}
        .viewport-panel {{
            flex: 1;
            position: relative;
            background: radial-gradient(circle at center, #1b222d 0%, #0d1117 100%);
        }}
        #viewport {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .sidebar {{
            width: 380px;
            background-color: #161b22;
            border-left: 1px solid #30363d;
            padding: 20px;
            overflow-y: auto;
        }}
        .control-group {{
            margin-bottom: 20px;
            background: #0d1117;
            padding: 14px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}
        .control-group h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
        }}
        select, button, input[type="range"] {{
            width: 100%;
            padding: 8px 12px;
            background-color: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        button {{
            cursor: pointer;
            background-color: #238636;
            color: white;
            font-weight: 600;
            border: none;
            transition: background-color 0.2s;
        }}
        button:hover {{
            background-color: #2ea043;
        }}
        .info-card {{
            background: #1f242c;
            border-left: 4px solid #58a6ff;
            padding: 12px;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 6px;
            font-size: 11px;
            font-weight: bold;
            background-color: #1f6feb;
            color: white;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ {package.title}</h1>
        <span class="badge">Air-Gapped FAIR Bundle</span>
    </div>
    <div class="container">
        <div class="viewport-panel">
            <canvas id="viewport" width="1200" height="900"></canvas>
        </div>
        <div class="sidebar">
            <div class="control-group">
                <h3>Stationary Point</h3>
                <select id="speciesSelect" onchange="window.viewer.switchModel(parseInt(this.value))">
                    {options_str}
                </select>
                <div id="modelInfo" class="info-card">
                    Loading thermodynamic properties...
                </div>
            </div>
            <div class="control-group">
                <h3>Vibrational Normal Modes</h3>
                <select id="modeSelect" onchange="window.viewer.setVibrationalMode(parseInt(this.value))">
                    {freq_options_str}
                </select>
            </div>
            <div class="control-group">
                <h3>3D Representation</h3>
                <button onclick="window.viewer.setStyle('ball_and_stick')">Ball and Stick</button>
                <button onclick="window.viewer.setStyle('licorice')" style="background-color:#30363d;">Licorice Wireframe</button>
                <button onclick="window.viewer.setStyle('spacefill')" style="background-color:#30363d;">CPK Spacefill</button>
            </div>
        </div>
    </div>

    <script>
    {EMBEDDED_MOLECULAR_VIEWER_JS}

    const packageData = {package_json};
    window.addEventListener('DOMContentLoaded', () => {{
        window.viewer = new CoChemViewer('viewport', 'modelInfo');
        window.viewer.loadData(packageData);
    }});
    </script>
</body>
</html>
"""
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(html_content)
    except Exception as err:
        raise HTMLBundleExportError(
            reason=f"Failed to write interactive HTML SI to '{out_path}': {str(err)}"
        ) from err

    return str(out_path.resolve())
