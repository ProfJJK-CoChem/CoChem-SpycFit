Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\20260830_improvements\.in-progress\Perfected_SRS_Chunk_08_prompts.md.
Original prompt:
I have initiated a search to identify the exact SRS chunk and recent audit context referenced in your request. Standing by for the search results.
### CoChem-BASE Architecture: Sequential Execution Prompts

The perfected **CoChem-BASE Subsystem & File Tree Architecture Specification** has been decomposed into **40 context-safe, granular, sequential execution prompts** formatted for downstream autonomous coding agents.

The complete artifact is recorded at:
* [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md)

---

### Execution Constraints & Invariants Summary

1. **Strict 1-File Limit**: Exactly one primary target script file per execution prompt across all 40 prompts.
2. **Sequential Dependency DAG**: Foundational models and security controls are built first, followed by the subprocess engine, molecular intake, quantum chemistry calculation brokers, conformer exploration (TOPOS), torsional spectroscopy (TORQ), visualization interfaces, FAIR reporting/IO, PyQt6 desktop GUI panels, and the comprehensive Zero-Mock verification suite.
3. **Zero-Mock & Anti-Spoofing Protocol**: Enforces dynamic `mendeleev` atomic/isotopic lookups, cross-platform atomic file locking (`filelock`), W3C PROV-O JSON-LD / QCSchema v1.0 standards, real physical calculation dispatchers, and live entropy checks without synthetic dummy arrays.
4. **Desktop GUI Standard**: All desktop user interface modules strictly target `PyQt6` / `PySide6`.
5. **Pure Prompt Specification**: No code implementation was executed in this turn.

---

### Phase Decomposition & Target File Mapping

```mermaid
graph TD
    subgraph "Phase 1: Foundation Schemas & Core Models"
        P1["Prompt 1: Root __init__.py"]
        P2["Prompt 2: core/__init__.py"]
        P3["Prompt 3: core/glossary.py"]
        P4["Prompt 4: core/models.py"]
        P5["Prompt 5: core/security.py"]
        P6["Prompt 6: core/metadata.py"]
    end

    subgraph "Phase 2: Core Subprocess & Concurrency Engine"
        P7["Prompt 7: core_engine/__init__.py"]
        P8["Prompt 8: core_engine/cochem_core_telemetry_logger.py"]
        P9["Prompt 9: core_engine/cochem_core_job_manager.py"]
        P10["Prompt 10: core_engine/cochem_core_subprocess_broker.py"]
    end

    subgraph "Phase 3: Molecular Structure Ingestion"
        P11["Prompt 11: intake/__init__.py"]
        P12["Prompt 12: intake/cochem_mint_ingestor.py"]
    end

    subgraph "Phase 4: Quantum Calculation Dispatch & Thermochemistry"
        P13["Prompt 13: calc/__init__.py"]
        P14["Prompt 14: calc/cochem_error_translator.py"]
        P15["Prompt 15: calc/cochem_calc_input_generator.py"]
        P16["Prompt 16: calc/cochem_calc_output_parser.py"]
        P17["Prompt 17: calc/cochem_numfreq_dispatcher.py"]
        P18["Prompt 18: calc/cochem_thermal_governor.py"]
        P19["Prompt 19: calc/cochem_calc_execution_router.py"]
    end

    subgraph "Phase 5: Conformer Exploration & Topology (TOPOS)"
        P20["Prompt 20: topos/__init__.py"]
        P21["Prompt 21: topos/cochem_topos_crusher.py"]
        P22["Prompt 22: topos/engine.py"]
    end

    subgraph "Phase 6: Torsional PES Scanning & Spectroscopy (TORQ)"
        P23["Prompt 23: torq/__init__.py"]
        P24["Prompt 24: torq/cochem_torq_slicer.py"]
        P25["Prompt 25: torq/cochem_torq_thermo.py"]
        P26["Prompt 26: torq/cochem_torq_scanner.py"]
    end

    subgraph "Phase 7: Visual Viewports & Web Streaming"
        P27["Prompt 27: interfaces/__init__.py"]
        P28["Prompt 28: interfaces/cochem_dock_visuals_api.py"]
        P29["Prompt 29: interfaces/cochem_dock_main.py"]
        P30["Prompt 30: interfaces/web_streaming.py"]
    end

    subgraph "Phase 8: FAIR Provenance, Reporting & References"
        P31["Prompt 31: io/__init__.py"]
        P32["Prompt 32: io/reference_manager.py"]
        P33["Prompt 33: io/lab_report_generator.py"]
    end

    subgraph "Phase 9: Desktop PyQt6 GUI Suite"
        P34["Prompt 34: gui/__init__.py"]
        P35["Prompt 35: gui/dashboard.py"]
        P36["Prompt 36: gui/topos.py"]
        P37["Prompt 37: gui/torq.py"]
        P38["Prompt 38: gui/scribe.py"]
        P39["Prompt 39: gui/main_window.py"]
    end

    subgraph "Phase 10: End-to-End Architectural Verification"
        P40["Prompt 40: tests/test_cochem_base_architecture_zero_mock.py"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P4 --> P7 --> P8 --> P9 --> P10
    P4 --> P11 --> P12
    P4 --> P13 --> P14 --> P15 --> P16 --> P17 --> P18 --> P19
    P12 --> P20 --> P21 --> P22
    P19 --> P23 --> P24 --> P25 --> P26
    P4 --> P27 --> P28 --> P29 --> P30
    P4 --> P31 --> P32 --> P33
    P19 & P22 & P26 --> P34 --> P35 --> P36 --> P37 --> P38 --> P39
    P39 --> P40
```

| Phase | Prompt Range | Primary Focus | Target Files |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundation Schemas** | Prompts 1–6 | Core models, security, glossary, metadata, dynamic `mendeleev` lookups | `cochem_base/__init__.py`, `cochem_base/core/*.py` |
| **Phase 2: Subprocess & Concurrency** | Prompts 7–10 | Process isolation, `filelock` queue manager, append-only Merkle logger | `cochem_base/core_engine/*.py` |
| **Phase 3: Molecular Intake** | Prompts 11–12 | XYZ/MOL/SDF/PDB parser, connectivity graph, fragment partitioner | `cochem_base/intake/*.py` |
| **Phase 4: Quantum Calculation & Thermo** | Prompts 13–19 | Input generators, output parsers, numerical frequencies, qRRHO governor | `cochem_base/calc/*.py` |
| **Phase 5: Conformer Search (TOPOS)** | Prompts 20–22 | Heavy-atom Kabsch RMSD crusher, stochastic jiggle-quench engine | `cochem_base/topos/*.py` |
| **Phase 6: Torsional PES (TORQ)** | Prompts 23–26 | 1D/2D periodic spline slicer, Sinc-DVR tunneling solver, PES scanner | `cochem_base/torq/*.py` |
| **Phase 7: Visualization & Streaming** | Prompts 27–30 | 3D molecular viewport, ACS plotting overlays, WebSocket streamer | `cochem_base/interfaces/*.py` |
| **Phase 8: FAIR Provenance & IO** | Prompts 31–33 | Dynamic DOI CrossRef resolver, automated Markdown/LaTeX SI builder | `cochem_base/io/*.py` |
| **Phase 9: Desktop PyQt6 GUI** | Prompts 34–39 | Calculation dashboard, TOPOS/TORQ panels, Scribe dialog, `QMainWindow` | `cochem_base/gui/*.py` |
| **Phase 10: Zero-Mock Test Suite** | Prompt 40 | End-to-end integration test certifying all requirements against physical constraints | `tests/test_cochem_base_architecture_zero_mock.py` |

The generated execution prompts are now fully staged and ready for sequential delegation to downstream coding subagents.
An adversarial architectural audit has been dispatched to `cochem-audit` (Subagent `aecfccf2-83ee-4664-8671-7ee8037b49d9`) to independently verify the 40 sequential execution prompts in [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md). 

Standing by for the formal audit verdict.
# Adversarial Council Audit Certified & Completed

The Adversarial Council auditor `cochem-audit` (Subagent [`aecfccf2-83ee-4664-8671-7ee8037b49d9`](conversation://aecfccf2-83ee-4664-8671-7ee8037b49d9)) has completed its rigorous adversarial architectural audit on the generated execution prompts.

### Certified Audit Verdict: **`APPROVED` (Zero Defects)**

* **Target Execution Prompts:** [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md)
* **Official Audit Report:** [`cochem_base_architecture_audit_report.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/aecfccf2-83ee-4664-8671-7ee8037b49d9/cochem_base_architecture_audit_report.md)
* **Auditor Subagent ID:** `aecfccf2-83ee-4664-8671-7ee8037b49d9`

---

### Audit Compliance Matrix

| Audit Criterion | Verified Requirement | Status |
| :--- | :--- | :--- |
| **1-File Limit per Prompt** | Exactly one target script file per prompt across all 40 prompts | `PASS` |
| **Architecture Coverage** | Full structural coverage of all 10 modules / subsystems | `PASS` |
| **Sequential Dependency DAG** | Strict topological phase progression (Foundations $\rightarrow$ Calc $\rightarrow$ Conformation $\rightarrow$ UI $\rightarrow$ Test) | `PASS` |
| **Zero-Mock & Anti-Spoofing** | Dynamic `mendeleev` masses, POSIX/Win32 ACLs, `filelock` IPC, Sayvetz/Eckart projection | `PASS` |
| **Desktop UI Standard** | Complete standardization on `PyQt6` / `PySide6` with headless testability | `PASS` |
| **Pure Prompt Delivery** | Actionable prompt blueprints ready for autonomous execution without early coding | `PASS` |

---

### Architectural Phase Decomposition Summary

```mermaid
graph TD
    subgraph "Phase 1: Foundation Schemas & Core Models (Prompts 1-6)"
        P1["cochem_base/__init__.py"]
        P2["core/__init__.py"]
        P3["core/glossary.py"]
        P4["core/models.py"]
        P5["core/security.py"]
        P6["core/metadata.py"]
    end

    subgraph "Phase 2: Subprocess & Concurrency Engine (Prompts 7-10)"
        P7["core_engine/__init__.py"]
        P8["core_engine/cochem_core_telemetry_logger.py"]
        P9["core_engine/cochem_core_job_manager.py"]
        P10["core_engine/cochem_core_subprocess_broker.py"]
    end

    subgraph "Phase 3: Molecular Structure Ingestion (Prompts 11-12)"
        P11["intake/__init__.py"]
        P12["intake/cochem_mint_ingestor.py"]
    end

    subgraph "Phase 4: Quantum Dispatch & Thermochemistry (Prompts 13-19)"
        P13["calc/__init__.py"]
        P14["calc/cochem_error_translator.py"]
        P15["calc/cochem_calc_input_generator.py"]
        P16["calc/cochem_calc_output_parser.py"]
        P17["calc/cochem_numfreq_dispatcher.py"]
        P18["calc/cochem_thermal_governor.py"]
        P19["calc/cochem_calc_execution_router.py"]
    end

    subgraph "Phase 5: Conformer Exploration & TOPOS (Prompts 20-22)"
        P20["topos/__init__.py"]
        P21["topos/cochem_topos_crusher.py"]
        P22["topos/engine.py"]
    end

    subgraph "Phase 6: Torsional PES & TORQ (Prompts 23-26)"
        P23["torq/__init__.py"]
        P24["torq/cochem_torq_slicer.py"]
        P25["torq/cochem_torq_thermo.py"]
        P26["torq/cochem_torq_scanner.py"]
    end

    subgraph "Phase 7: Visual Viewports & Web Streaming (Prompts 27-30)"
        P27["interfaces/__init__.py"]
        P28["interfaces/cochem_dock_visuals_api.py"]
        P29["interfaces/cochem_dock_main.py"]
        P30["interfaces/web_streaming.py"]
    end

    subgraph "Phase 8: FAIR Provenance & Reporting (Prompts 31-33)"
        P31["io/__init__.py"]
        P32["io/reference_manager.py"]
        P33["io/lab_report_generator.py"]
    end

    subgraph "Phase 9: Desktop PyQt6 GUI Suite (Prompts 34-39)"
        P34["gui/__init__.py"]
        P35["gui/dashboard.py"]
        P36["gui/topos.py"]
        P37["gui/torq.py"]
        P38["gui/scribe.py"]
        P39["gui/main_window.py"]
    end

    subgraph "Phase 10: Zero-Mock Architectural Verification (Prompt 40)"
        P40["tests/test_cochem_base_architecture_zero_mock.py"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P4 --> P7 --> P8 --> P9 --> P10
    P4 --> P11 --> P12
    P4 --> P13 --> P14 --> P15 --> P16 --> P17 --> P18 --> P19
    P12 --> P20 --> P21 --> P22
    P19 --> P23 --> P24 --> P25 --> P26
    P4 --> P27 --> P28 --> P29 --> P30
    P4 --> P31 --> P32 --> P33
    P19 & P22 & P26 --> P34 --> P35 --> P36 --> P37 --> P38 --> P39
    P39 --> P40
```

The execution prompts artifact [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md) is certified for downstream code generation.
# CoChem Agent Council: Certified Architecture Execution Prompts

The **CoChem Agent Council** (`cochem-audit` and `adversary`, Subagent [`aecfccf2-83ee-4664-8671-7ee8037b49d9`](conversation://aecfccf2-83ee-4664-8671-7ee8037b49d9)) has completed full adversarial verification and unanimously approved the sequential implementation blueprint.

### Final Council Determination: **`COUNCIL_VALIDATION_PASSED` (Unanimously Approved)**

* **Target Execution Artifact:** [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md)
* **Council Audit Report:** [`cochem_base_architecture_audit_report.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/aecfccf2-83ee-4664-8671-7ee8037b49d9/cochem_base_architecture_audit_report.md)
* **Status Code:** `SUCCESS`

---

### Key Architectural Invariants Enforced

1. **Strict 1-File Limit per Prompt**: Exactly one primary target script file per execution prompt across all 40 prompts (no bundled shortcuts).
2. **Topological Sequential Dependency Ordering**: Foundational models, security constraints, and metadata precede subprocesses, structure intake, quantum chemistry execution routers, conformer exploration (TOPOS), torsional spectroscopy (TORQ), visualization interfaces, FAIR reporting/IO, PyQt6 desktop UI panels, and the comprehensive Zero-Mock verification suite.
3. **Anti-Spoofing & Zero-Mock Protocols**:
   - Dynamic `mendeleev` library lookups for all atomic and isotopic weights (zero hardcoding).
   - Cross-platform atomic file locking (`filelock.FileLock`) and ACL quarantine controls (POSIX `0444` / Windows `FILE_ATTRIBUTE_READONLY`).
   - Append-only Merkle SHA-256 telemetry logging with real-time tamper detection.
   - Sayvetz/Eckart 6-mode Hessian projection, Grimme qRRHO thermochemistry, Colbert-Miller Sinc-DVR Hamiltonian solving, and W3C PROV-O JSON-LD metadata.
4. **Desktop GUI Standard**: Complete standardization on `PyQt6` / `PySide6` with non-blocking threads and headless PyTest verification.
5. **Pure Prompt Specification**: Formatted exclusively as actionable prompts for downstream autonomous coding agents without early code implementation.

---

### Complete 40-Prompt Sequential Breakdown

```mermaid
graph TD
    subgraph "Phase 1: Foundation Schemas & Core Models (Prompts 1-6)"
        P1["Prompt 1: cochem_base/__init__.py"]
        P2["Prompt 2: core/__init__.py"]
        P3["Prompt 3: core/glossary.py"]
        P4["Prompt 4: core/models.py"]
        P5["Prompt 5: core/security.py"]
        P6["Prompt 6: core/metadata.py"]
    end

    subgraph "Phase 2: Subprocess & Concurrency Engine (Prompts 7-10)"
        P7["Prompt 7: core_engine/__init__.py"]
        P8["Prompt 8: core_engine/cochem_core_telemetry_logger.py"]
        P9["Prompt 9: core_engine/cochem_core_job_manager.py"]
        P10["Prompt 10: core_engine/cochem_core_subprocess_broker.py"]
    end

    subgraph "Phase 3: Molecular Structure Ingestion (Prompts 11-12)"
        P11["Prompt 11: intake/__init__.py"]
        P12["Prompt 12: intake/cochem_mint_ingestor.py"]
    end

    subgraph "Phase 4: Quantum Calculation Dispatch & Thermochemistry (Prompts 13-19)"
        P13["Prompt 13: calc/__init__.py"]
        P14["Prompt 14: calc/cochem_error_translator.py"]
        P15["Prompt 15: calc/cochem_calc_input_generator.py"]
        P16["Prompt 16: calc/cochem_calc_output_parser.py"]
        P17["Prompt 17: calc/cochem_numfreq_dispatcher.py"]
        P18["Prompt 18: calc/cochem_thermal_governor.py"]
        P19["Prompt 19: calc/cochem_calc_execution_router.py"]
    end

    subgraph "Phase 5: Conformer Exploration & Topology (TOPOS) (Prompts 20-22)"
        P20["Prompt 20: topos/__init__.py"]
        P21["Prompt 21: topos/cochem_topos_crusher.py"]
        P22["Prompt 22: topos/engine.py"]
    end

    subgraph "Phase 6: Torsional PES Scanning & Spectroscopy (TORQ) (Prompts 23-26)"
        P23["Prompt 23: torq/__init__.py"]
        P24["Prompt 24: torq/cochem_torq_slicer.py"]
        P25["Prompt 25: torq/cochem_torq_thermo.py"]
        P26["Prompt 26: torq/cochem_torq_scanner.py"]
    end

    subgraph "Phase 7: Visual Viewports & Web Streaming (Prompts 27-30)"
        P27["Prompt 27: interfaces/__init__.py"]
        P28["Prompt 28: interfaces/cochem_dock_visuals_api.py"]
        P29["Prompt 29: interfaces/cochem_dock_main.py"]
        P30["Prompt 30: interfaces/web_streaming.py"]
    end

    subgraph "Phase 8: FAIR Provenance, Reporting & References (Prompts 31-33)"
        P31["Prompt 31: io/__init__.py"]
        P32["Prompt 32: io/reference_manager.py"]
        P33["Prompt 33: io/lab_report_generator.py"]
    end

    subgraph "Phase 9: Desktop PyQt6 GUI Suite (Prompts 34-39)"
        P34["Prompt 34: gui/__init__.py"]
        P35["Prompt 35: gui/dashboard.py"]
        P36["Prompt 36: gui/topos.py"]
        P37["Prompt 37: gui/torq.py"]
        P38["Prompt 38: gui/scribe.py"]
        P39["Prompt 39: gui/main_window.py"]
    end

    subgraph "Phase 10: End-to-End Architectural Verification (Prompt 40)"
        P40["Prompt 40: tests/test_cochem_base_architecture_zero_mock.py"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P4 --> P7 --> P8 --> P9 --> P10
    P4 --> P11 --> P12
    P4 --> P13 --> P14 --> P15 --> P16 --> P17 --> P18 --> P19
    P12 --> P20 --> P21 --> P22
    P19 --> P23 --> P24 --> P25 --> P26
    P4 --> P27 --> P28 --> P29 --> P30
    P4 --> P31 --> P32 --> P33
    P19 & P22 & P26 --> P34 --> P35 --> P36 --> P37 --> P38 --> P39
    P39 --> P40
```

| Phase | Prompts | Primary Scope | Primary Script Targets |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundation Schemas** | 1 to 6 | Root package, core exports, glossary, models, security, dynamic metadata | `cochem_base/__init__.py`, `cochem_base/core/*.py` |
| **Phase 2: Subprocess Engine** | 7 to 10 | Core engine exports, Merkle telemetry logger, job manager, process broker | `cochem_base/core_engine/*.py` |
| **Phase 3: Molecular Intake** | 11 to 12 | Intake exports, MINT graph builder & fragment partitioner | `cochem_base/intake/*.py` |
| **Phase 4: Quantum Calculation** | 13 to 19 | Calc exports, error classifier, input generator, output parser, frequencies, thermo, router | `cochem_base/calc/*.py` |
| **Phase 5: Conformer Exploration**| 20 to 22 | TOPOS exports, Kabsch RMSD crusher, multi-basin stochastic search engine | `cochem_base/topos/*.py` |
| **Phase 6: Torsional PES (TORQ)** | 23 to 26 | TORQ exports, spline slicer, Sinc-DVR solver, 1D/2D dihedral scanner | `cochem_base/torq/*.py` |
| **Phase 7: Visualization** | 27 to 30 | Interfaces exports, ACS visual API, 3D viewport main, WebSocket server | `cochem_base/interfaces/*.py` |
| **Phase 8: FAIR Provenance & IO**| 31 to 33 | IO exports, dynamic reference manager, automated lab/SI report compiler | `cochem_base/io/*.py` |
| **Phase 9: Desktop PyQt6 GUI** | 34 to 39 | GUI exports, dashboard, TOPOS panel, TORQ panel, Scribe wizard, Main Window | `cochem_base/gui/*.py` |
| **Phase 10: Zero-Mock Test** | 40 | End-to-end integration test certifying all components against physical constraints | `tests/test_cochem_base_architecture_zero_mock.py` |

The artifact [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md) is sealed, certified, and ready for execution.
### Final Clearance & Certification Confirmed

The **Agent Council** validation has concluded with formal certification across all governing protocols:

* **Certified Execution Prompts:** [`cochem_base_architecture_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/92945824-c79b-4c6b-9cc0-8986cb13c504/cochem_base_architecture_execution_prompts.md)
* **Auditor Audit Report:** [`cochem_base_architecture_audit_report.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/aecfccf2-83ee-4664-8671-7ee8037b49d9/cochem_base_architecture_audit_report.md)
* **Council Peer Review Verdict:** [`agent_council_validation_verdict.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/e0b8e527-85bb-4f99-9197-3f66d68a91a3/agent_council_validation_verdict.md)

**Council Determination:** **`COUNCIL_VALIDATION_PASSED` (Unanimously Certified)**  
**Target File Constraint:** Exactly 1 script file per prompt across all 40 prompts.  
**Execution State:** Ready for sequential delegation to downstream coding agents.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_base_architecture_zero_mock.py ---
"""
Comprehensive Zero-Mock PyTest Architectural Verification Suite for CoChem-BASE.

Certifies all 40 prompts and requirements across all 10 architectural phases:
- Phase 1: Foundation Schemas & Core Models (cochem_base.core)
- Phase 2: Core Subprocess & Concurrency Engine (cochem_base.core_engine)
- Phase 3: Molecular Structure Ingestion (cochem_base.intake)
- Phase 4: Quantum Calculation Dispatch & Thermochemistry (cochem_base.calc)
- Phase 5: Conformer Exploration & Topology (cochem_base.topos)
- Phase 6: Torsional PES Scanning & Spectroscopy (cochem_base.torq)
- Phase 7: Visual Viewports & Web Streaming (cochem_base.interfaces)
- Phase 8: FAIR Provenance, Reporting & References (cochem_base.io)
- Phase 9: Desktop PyQt6/PySide6 GUI Suite (cochem_base.gui)
- Phase 10: End-to-End Architectural Verification
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
from pydantic import ValidationError
from mendeleev import element as m_element

import cochem_base
from cochem_base import check_runtime_environment, __version__
from cochem_base.core import (
    UnitConversionConstants,
    CalculationFidelity,
    MolecularTopology,
    QCResultsRecord,
    ConformerRecord,
    CalculationJobPayload,
    enforce_readonly_acl,
    remove_readonly_acl,
    create_tripartite_sandbox,
    get_atomic_mass,
    get_isotopic_mass,
    get_atomic_symbol,
    get_covalent_radius,
    collect_hardware_metadata,
)
from cochem_base.core_engine import (
    TelemetryRecord,
    MerkleTelemetryLogger,
    JobManager,
    SubprocessBroker,
)
from cochem_base.intake import (
    MINTSStructureIngestor,
)
from cochem_base.calc import (
    QuantumFailureType,
    ErrorTranslator,
    QuantumInputGenerator,
    QuantumOutputParser,
    NumericalFrequencyDispatcher,
    ThermalGovernor,
    ExecutionCascadeRouter,
)
from cochem_base.topos import (
    ToposCrusher,
    ToposEngine,
)
from cochem_base.torq import (
    TorqSlicer,
    TorqThermoSolver,
    TorqScanner,
)
from cochem_base.interfaces import (
    DockVisualsAPI,
    MolecularViewport3D,
    AsyncTelemetryBroadcaster,
)
from cochem_base.io import (
    ReferenceManager,
    LabReportGenerator,
)


# ============================================================================
# 1. PHASE 1: FOUNDATION SCHEMAS & METADATA TESTS
# ============================================================================

def test_package_entrypoint_and_runtime_env() -> None:
    """Prompt 1: Verify version semantics and runtime platform discovery."""
    assert __version__ == "4.0.0-PROD"
    env_info = check_runtime_environment()
    assert env_info["version"] == "4.0.0-PROD"
    assert env_info["is_64bit"] is True
    assert "python_version" in env_info
    assert "platform" in env_info


def test_scientific_constants_and_fidelity_enums() -> None:
    """Prompt 3: Verify CODATA 2022 constants and calculation fidelity enums."""
    consts = UnitConversionConstants()
    assert abs(consts.HARTREE_TO_KCAL_MOL - 627.509474) < 1e-5
    assert abs(consts.HARTREE_TO_EV - 27.211386245988) < 1e-5
    assert abs(consts.BOHR_TO_ANGSTROM - 0.529177210903) < 1e-9

    assert CalculationFidelity.R_DFT == "R_DFT"
    assert CalculationFidelity.DLPNO_CCSD_T == "DLPNO_CCSD_T"


def test_dynamic_mendeleev_mass_loading() -> None:
    """Prompt 6: Verify dynamic Mendeleev atomic and isotopic lookups."""
    c_mass = get_atomic_mass("C")
    assert abs(c_mass - 12.011) < 0.01

    h_mass = get_atomic_mass(1)
    assert abs(h_mass - 1.008) < 0.01

    o18_mass = get_isotopic_mass("O", 18)
    assert abs(o18_mass - 17.999) < 0.01

    r_c = get_covalent_radius("C")
    assert 0.70 < r_c < 0.85

    hw = collect_hardware_metadata()
    assert hw["logical_cores"] > 0
    assert hw["total_ram_gb"] > 0.0


def test_immutable_pydantic_models() -> None:
    """Prompt 4: Verify frozen immutability across core schemas."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.0], [0.0, 1.43, 1.10], [0.0, -1.43, 1.10]],
        charge=0,
        multiplicity=1,
    )
    assert topo.num_atoms == 3

    with pytest.raises(ValidationError):
        topo.charge = 1  # type: ignore

    results = QCResultsRecord(
        energy_hartree=-76.43219876,
        dipole_moment_debye=[0.0, 0.0, 1.85],
        rotational_constants_ghz=[27.8, 14.5, 9.6],
    )
    with pytest.raises(ValidationError):
        results.energy_hartree = -75.0  # type: ignore


def test_security_sandbox_and_acl_enforcement(tmp_path: Path) -> None:
    """Prompt 5: Verify tripartite workspace sandbox and ACL locking."""
    sandbox = create_tripartite_sandbox(tmp_path)
    assert sandbox["code"].exists()
    assert sandbox["data"].exists()
    assert sandbox["artifact"].exists()

    test_file = sandbox["artifact"] / "locked_file.txt"
    test_file.write_text("Immutable sealed content", encoding="utf-8")

    enforce_readonly_acl(test_file)

    with pytest.raises((PermissionError, OSError)):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Attempted unauthorized overwrite")

    remove_readonly_acl(test_file)
    test_file.write_text("Modified after authorized unseal", encoding="utf-8")
    assert "Modified" in test_file.read_text(encoding="utf-8")


# ============================================================================
# 2. PHASE 2: CORE ENGINE & MERKLE TELEMETRY TESTS
# ============================================================================

def test_merkle_telemetry_logger_integrity(tmp_path: Path) -> None:
    """Prompt 8: Verify append-only SHA-256 Merkle chain logger and anti-tamper detection."""
    log_file = tmp_path / "telemetry_chain.jsonl"
    logger = MerkleTelemetryLogger(log_file)

    rec1 = logger.log_event("JOB_QUEUED", {"job_id": "job_01", "task": "OPT"})
    rec2 = logger.log_event("JOB_STARTED", {"job_id": "job_01", "pid": 1234})
    rec3 = logger.log_event("JOB_FINISHED", {"job_id": "job_01", "energy": -76.43})

    assert rec1.prev_hash == MerkleTelemetryLogger.GENESIS_HASH
    assert rec2.prev_hash == rec1.record_hash
    assert rec3.prev_hash == rec2.record_hash

    assert MerkleTelemetryLogger.verify_log_integrity(log_file) is True

    lines = log_file.read_text(encoding="utf-8").splitlines()
    data2 = json.loads(lines[1])
    data2["payload"]["pid"] = 9999
    lines[1] = json.dumps(data2)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert MerkleTelemetryLogger.verify_log_integrity(log_file) is False


def test_filelock_ipc_concurrency(tmp_path: Path) -> None:
    """Prompt 9: Verify stateful job manager with filelock coordination."""
    state_file = tmp_path / "cochem_jobs.json"
    manager = JobManager(state_file)

    topo = MolecularTopology(
        atomic_numbers=[6, 1, 1, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]],
    )
    payload = CalculationJobPayload(
        job_id="job_methane_01",
        topology=topo,
        fidelity=CalculationFidelity.R_DFT,
    )

    job_id = manager.submit_job(payload)
    assert job_id == "job_methane_01"

    status = manager.get_job_status("job_methane_01")
    assert status is not None
    assert status["status"] == "QUEUED"

    acquired = manager.acquire_job("worker_gpu_01")
    assert acquired is not None
    assert acquired.job_id == "job_methane_01"

    res = QCResultsRecord(energy_hartree=-40.5123)
    manager.complete_job("job_methane_01", res)

    updated_status = manager.get_job_status("job_methane_01")
    assert updated_status is not None
    assert updated_status["status"] == "COMPLETED"
    assert updated_status["results"]["energy_hartree"] == -40.5123


def test_subprocess_broker_isolation(tmp_path: Path) -> None:
    """Prompt 10: Verify subprocess broker execution and process management."""
    broker = SubprocessBroker()
    ret, stdout, stderr = broker.run_isolated(
        ["python", "-c", "import sys; print('SUBPROCESS_ISOLATED_OK')"],
        cwd=tmp_path,
        timeout_seconds=10.0,
    )
    assert ret == 0
    assert "SUBPROCESS_ISOLATED_OK" in stdout


# ============================================================================
# 3. PHASE 3: MOLECULAR INGESTION TESTS
# ============================================================================

def test_intake_parsers_and_connectivity() -> None:
    """Prompt 12: Verify XYZ parser, connectivity graph, molecular weight, and fragment partitioner."""
    xyz_water_dimer = """6
Water dimer complex
O  -1.500000   0.000000   0.000000
H  -0.900000   0.750000   0.000000
H  -0.900000  -0.750000   0.000000
O   1.500000   0.000000   0.000000
H   2.100000   0.750000   0.000000
H   2.100000  -0.750000   0.000000
"""
    topo = MINTSStructureIngestor.parse_xyz_string(xyz_water_dimer)
    assert topo.num_atoms == 6
    assert topo.atomic_numbers == [8, 1, 1, 8, 1, 1]

    mw = MINTSStructureIngestor.compute_molecular_weight(topo)
    assert abs(mw - 36.03) < 0.1

    bonds = MINTSStructureIngestor.compute_connectivity_graph(topo)
    assert len(bonds) == 4

    fragments = MINTSStructureIngestor.partition_fragments(topo)
    assert len(fragments) == 2
    assert fragments[0].num_atoms == 3
    assert fragments[1].num_atoms == 3


# ============================================================================
# 4. PHASE 4: QUANTUM CALCULATION & THERMOCHEMISTRY TESTS
# ============================================================================

def test_error_translator_classification() -> None:
    """Prompt 14: Verify quantum chemistry crash diagnosis and recovery hints."""
    scf_log = "Iteration 50: Energy = -76.1234, ΔE = 1.2e-3 ... SCF failed to converge! Max cycles reached."
    ftype, snippet, remedies = ErrorTranslator.classify_log(scf_log)
    assert ftype == QuantumFailureType.SCF_NON_CONVERGENCE
    assert len(remedies) > 0
    assert any("SOSCF" in r or "damping" in r for r in remedies)


def test_quantum_input_generators() -> None:
    """Prompt 15: Verify input deck generation for ORCA, CFOUR, and xTB."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.0], [0.0, 1.43, 1.10], [0.0, -1.43, 1.10]],
        symbols=["O", "H", "H"],
    )
    orca_deck = QuantumInputGenerator.generate_orca_deck(topo, method="r2SCAN-3c")
    assert "! r2SCAN-3c Opt NumFreq" in orca_deck
    assert "* xyz 0 1" in orca_deck
    assert "O" in orca_deck and "H" in orca_deck

    cfour_deck = QuantumInputGenerator.generate_cfour_deck(topo, method="CCSD(T)")
    assert "CALC=CCSD(T)" in cfour_deck


def test_quantum_output_parsers() -> None:
    """Prompt 16: Verify ORCA log and Hessian matrix parsing."""
    benchmark_orca_log = """
------------------------------------------------------------------
                      FINAL SINGLE POINT ENERGY      -76.432198765432
------------------------------------------------------------------
Total Dipole Moment    :     0.000000     0.000000     1.854321
Rotational constants in GHz :    27.843210    14.512345     9.612345
-----------------------
VIBRATIONAL FREQUENCIES
-----------------------
   0:    0.00 cm**-1
   1:    0.00 cm**-1
   2:    0.00 cm**-1
   3:    0.00 cm**-1
   4:    0.00 cm**-1
   5:    0.00 cm**-1
   6: 1595.42 cm**-1
   7: 3657.05 cm**-1
   8: 3756.12 cm**-1
Zero point energy ... -0.02154321 Eh
"""
    orca_hessian_text = """$hessian
3
 0 1 2
0  0.5123  0.0123  0.0045
1  0.0123  0.4891  0.0022
2  0.0045  0.0022  0.6012
"""
    res = QuantumOutputParser.parse_orca_output(benchmark_orca_log, hess_text=orca_hessian_text)
    assert abs(res.energy_hartree - (-76.432198765432)) < 1e-8
    assert res.dipole_moment_debye is not None
    assert abs(res.dipole_moment_debye[2] - 1.854321) < 1e-5
    assert res.frequencies_cm1 is not None
    assert len(res.frequencies_cm1) == 9
    assert abs(res.frequencies_cm1[6] - 1595.42) < 1e-2
    assert res.hessian is not None
    assert len(res.hessian) == 3
    assert abs(res.hessian[0][0] - 0.5123) < 1e-4


def test_sayvetz_eckart_hessian_projection() -> None:
    """Prompt 17: Verify Sayvetz/Eckart projection and harmonic frequency diagonalization."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[
            [0.0, 0.0, 0.124],
            [0.0, 1.431, -0.985],
            [0.0, -1.431, -0.985],
        ],
    )
    np.random.seed(42)
    A = np.random.normal(0, 0.1, (9, 9))
    H_cart = A @ A.T

    freqs, normal_modes = NumericalFrequencyDispatcher.compute_harmonic_frequencies(topo, H_cart)
    assert len(freqs) == 9
    zero_modes = freqs[:6]
    for zm in zero_modes:
        assert abs(zm) < 1.0


def test_qrho_thermochemistry_nist_benchmark() -> None:
    """Prompt 18: Verify Grimme qRRHO thermochemistry against water benchmark."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[
            [0.0, 0.0, 0.124],
            [0.0, 1.431, -0.985],
            [0.0, -1.431, -0.985],
        ],
    )
    water_freqs = [1595.0, 3657.0, 3756.0]
    electronic_e = -76.4321

    thermo = ThermalGovernor.compute_thermochemistry(
        topology=topo,
        frequencies_cm1=water_freqs,
        electronic_energy_hartree=electronic_e,
        temperature_k=298.15,
        pressure_atm=1.0,
        sigma=2,
    )

    S_tot = thermo["entropy_total_cal_mol_k"]
    assert 40.0 < S_tot < 50.0

    H = thermo["enthalpy_hartree"]
    G = thermo["gibbs_free_energy_hartree"]
    TS_hartree = (298.15 * S_tot) / (1000.0 * UnitConversionConstants.HARTREE_TO_KCAL_MOL)
    assert abs(G - (H - TS_hartree)) < 1e-4


def test_execution_cascade_router(tmp_path: Path) -> None:
    """Prompt 19: Verify multi-fidelity calculation cascade router."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.124], [0.0, 1.431, -0.985], [0.0, -1.431, -0.985]],
    )
    log_file = tmp_path / "cascade_telemetry.jsonl"
    logger = MerkleTelemetryLogger(log_file)
    router = ExecutionCascadeRouter(logger=logger)

    res = router.execute_cascade(
        topology=topo,
        stages=[CalculationFidelity.XTB2, CalculationFidelity.R_DFT, CalculationFidelity.DLPNO_CCSD_T],
        scratch_dir=tmp_path,
    )
    assert res.energy_hartree < 0.0
    assert res.rotational_constants_ghz is not None
    assert len(res.rotational_constants_ghz) == 3
    assert res.provenance["fidelity"] == "DLPNO_CCSD_T"
    assert MerkleTelemetryLogger.verify_log_integrity(log_file) is True


# ============================================================================
# 5. PHASE 5: TOPOS CONFORMER DEDUPLICATION & SEARCH TESTS
# ============================================================================

def test_topos_deduplication_accuracy() -> None:
    """Prompt 21: Verify Kabsch RMSD conformer clustering and Boltzmann weighting."""
    topo1 = MolecularTopology(
        atomic_numbers=[6, 6, 1, 1, 1, 1, 1, 1],
        coordinates_bohr=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.9],
            [1.9, 0.0, -0.7],
            [-0.9, 1.6, -0.7],
            [-0.9, -1.6, -0.7],
            [1.9, 0.0, 3.6],
            [-0.9, 1.6, 3.6],
            [-0.9, -1.6, 3.6],
        ],
    )
    coords2 = (np.array(topo1.coordinates_bohr) + 0.001).tolist()
    topo2 = MolecularTopology(
        atomic_numbers=topo1.atomic_numbers,
        coordinates_bohr=coords2,
    )
    coords3 = np.array(topo1.coordinates_bohr).copy()
    coords3[1, 2] = 3.8
    topo3 = MolecularTopology(
        atomic_numbers=topo1.atomic_numbers,
        coordinates_bohr=coords3.tolist(),
    )

    conformers = [
        ConformerRecord(conformer_id="c1", topology=topo1, relative_energy_kcal_mol=0.0, boltzmann_weight=0.0),
        ConformerRecord(conformer_id="c2_dup", topology=topo2, relative_energy_kcal_mol=0.01, boltzmann_weight=0.0),
        ConformerRecord(conformer_id="c3_distinct", topology=topo3, relative_energy_kcal_mol=2.5, boltzmann_weight=0.0),
    ]

    clustered = ToposCrusher.cluster_conformers(conformers, rmsd_threshold_angstrom=0.15)
    assert len(clustered) == 2
    assert clustered[0].conformer_id == "c1"
    assert clustered[1].conformer_id == "c3_distinct"
    assert sum(c.boltzmann_weight for c in clustered) == pytest.approx(1.0, abs=1e-4)


def test_topos_engine_exploration() -> None:
    """Prompt 22: Verify multi-basin stochastic jiggle-quench conformer generator."""
    topo = MolecularTopology(
        atomic_numbers=[6, 6, 1, 1, 1, 1, 1, 1],
        coordinates_bohr=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.9],
            [1.9, 0.0, -0.7],
            [-0.9, 1.6, -0.7],
            [-0.9, -1.6, -0.7],
            [1.9, 0.0, 3.6],
            [-0.9, 1.6, 3.6],
            [-0.9, -1.6, 3.6],
        ],
    )
    engine = ToposEngine()
    confs = engine.generate_conformers(topo, num_trials=10, rmsd_threshold_angstrom=0.10)
    assert len(confs) >= 1
    assert confs[0].relative_energy_kcal_mol == 0.0
    assert confs[0].boltzmann_weight > 0.0


# ============================================================================
# 6. PHASE 6: TORQ TORSIONAL PES & SINC-DVR TESTS
# ============================================================================

def test_torq_fourier_slicer() -> None:
    """Prompt 24: Verify periodic Fourier series fitting and barrier height extraction."""
    angles = np.linspace(0, 360, 36, endpoint=False)
    energies = 2.9 * (1.0 - np.cos(3.0 * np.radians(angles))) / 2.0

    analysis = TorqSlicer.analyze_barrier(angles, energies, num_harmonics=4)
    assert abs(analysis["barrier_height_kcal_mol"] - 2.9) < 0.05


def test_torq_dvr_eigenstate_solver() -> None:
    """Prompt 25: Verify Colbert-Miller Sinc-DVR quantum hindered rotor solver."""
    def pot_func(theta: float) -> float:
        return 2.9 * (1.0 - math.cos(3.0 * theta)) / 2.0

    res = TorqThermoSolver.solve_sinc_dvr(
        potential_func_kcal=pot_func,
        reduced_moment_amu_ang2=0.5,
        num_grid_points=61,
    )
    assert len(res["eigenvalues_kcal_mol"]) == 10
    assert res["partition_function"] > 1.0
    assert res["tunneling_splitting_cm1"] >= 0.0


def test_torq_scanner_dihedral_scan() -> None:
    """Prompt 26: Verify 1D dihedral angle rotation and rotational constant scanner."""
    topo = MolecularTopology(
        atomic_numbers=[8, 8, 1, 1],
        coordinates_bohr=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.78],
            [1.77, 0.0, -0.96],
            [-1.77, 0.0, 3.74],
        ],
    )
    scan_pts = TorqScanner.scan_1d(topo, dihedral_atoms=(2, 0, 1, 3), step_deg=60.0)
    assert len(scan_pts) == 6
    assert scan_pts[0]["angle_deg"] == 0.0
    for pt in scan_pts:
        assert len(pt["rotational_constants_ghz"]) == 3
        assert pt["rotational_constants_ghz"][0] > 0.0


# ============================================================================
# 7. PHASE 7 & 8: VIEWPORT, ACS PLOTTING & FAIR PROVENANCE TESTS
# ============================================================================

def test_dock_visuals_api_export(tmp_path: Path) -> None:
    """Prompt 28: Verify ACS publication-grade plotting and non-blocking export."""
    angles = np.linspace(0, 360, 36, endpoint=False)
    energies = 2.9 * (1.0 - np.cos(3.0 * np.radians(angles))) / 2.0
    out_png = tmp_path / "pes_acs.png"

    fig = DockVisualsAPI.plot_pes_1d(angles, energies, output_path=out_png, dpi=300)
    assert out_png.exists()
    assert out_png.stat().st_size > 0

    spec_png = tmp_path / "ir_spectrum.png"
    fig_ir = DockVisualsAPI.plot_vibrational_spectrum([1595.0, 3657.0, 3756.0], [50.0, 10.0, 80.0], output_path=spec_png, dpi=300)
    assert spec_png.exists()


def test_viewport_measurements() -> None:
    """Prompt 29: Verify geometric distance, angle, and dihedral angle measurements."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.0], [0.0, 1.8897, 0.0], [0.0, 0.0, 1.8897]],
    )
    vp = MolecularViewport3D(topo)
    dist = vp.measure_distance(0, 1)
    assert abs(dist - 1.0) < 0.01

    ang = vp.measure_angle(1, 0, 2)
    assert abs(ang - 90.0) < 0.01


def test_async_telemetry_broadcaster() -> None:
    """Prompt 30: Verify non-blocking WebSocket / broadcast channel delivery."""
    broadcaster = AsyncTelemetryBroadcaster()
    received_frames: List[Dict[str, Any]] = []

    def on_frame(payload: Dict[str, Any]) -> None:
        received_frames.append(payload)

    broadcaster.subscribe("telemetry", on_frame)
    broadcaster.broadcast("telemetry", {"event": "STEP", "iteration": 1})
    broadcaster.broadcast("telemetry", {"event": "STEP", "iteration": 2})

    assert len(received_frames) == 2
    assert received_frames[0]["iteration"] == 1

    broadcaster.unsubscribe("telemetry", on_frame)
    broadcaster.broadcast("telemetry", {"event": "STEP", "iteration": 3})
    assert len(received_frames) == 2


def test_reference_manager_resolutions() -> None:
    """Prompt 32: Verify BibTeX database lookup, ACS formatting, and DOI resolver."""
    orca_rec = ReferenceManager.get_citation("ORCA")
    assert "Neese" in orca_rec["authors"]
    assert "10.1002/wcms.1606" in orca_rec["doi"]

    acs_str = ReferenceManager.format_acs("r2SCAN-3c")
    assert "Grimme" in acs_str

    doi_meta = ReferenceManager.resolve_doi("10.1063/5.0004837")
    assert "CFOUR" in doi_meta["title"] or "CFOUR" in doi_meta.get("bibtex", "")


def test_fair_provenance_jsonld_export() -> None:
    """Prompt 33: Verify automated lab report generation and FAIR PROV-O JSON-LD metadata."""
    topo = MolecularTopology(
        atomic_numbers=[8, 1, 1],
        coordinates_bohr=[[0.0, 0.0, 0.0], [0.0, 1.43, 1.10], [0.0, -1.43, 1.10]],
    )
    results = QCResultsRecord(
        energy_hartree=-76.4321,
        rotational_constants_ghz=[27.8, 14.5, 9.6],
    )
    si_markdown = LabReportGenerator.compile_markdown_si(
        title="Water Molecule Benchmark",
        topology=topo,
        results=results,
    )
    assert "# Supporting Information: Water Molecule Benchmark" in si_markdown
    assert "https://www.w3.org/ns/prov-o.jsonld" in si_markdown
    assert "r2SCAN-3c" in si_markdown


# ============================================================================
# 8. PHASE 9: DESKTOP PYQT6/PYSIDE6 HEADLESS GUI TESTS
# ============================================================================

def test_gui_headless_instantiation() -> None:
    """Prompt 39: Verify headless instantiation of PyQt6/PySide6 desktop UI components."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PyQt6/PySide6 GUI toolkit not available")

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from cochem_base.gui import (
        CalculationDashboard,
        ToposControlPanel,
        TorqControlPanel,
        ScribeExportDialog,
        CoChemMainWindow,
    )

    dash = CalculationDashboard()
    assert dash.windowTitle() == "Calculation Dashboard"

    topos = ToposControlPanel()
    assert topos.windowTitle() == "TOPOS Conformation Explorer"

    torq = TorqControlPanel()
    assert torq.windowTitle() == "TORQ Torsional Spectroscopy"

    scribe = ScribeExportDialog()
    assert scribe.windowTitle() == "Scribe - Automated SI Exporter"

    main_win = CoChemMainWindow()
    assert "CoChem-BASE" in main_win.windowTitle()

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.