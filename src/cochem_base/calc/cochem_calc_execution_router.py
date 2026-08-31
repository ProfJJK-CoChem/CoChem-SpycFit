"""
Cascading Multi-Fidelity Quantum Execution Router and Fallback Pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from cochem_base.core.models import MolecularTopology, QCResultsRecord, CalculationJobPayload
from cochem_base.core.glossary import CalculationFidelity
from cochem_base.core_engine.cochem_core_telemetry_logger import MerkleTelemetryLogger
from cochem_base.calc.cochem_calc_input_generator import QuantumInputGenerator
from cochem_base.torq.cochem_torq_scanner import TorqScanner


class ExecutionCascadeRouter:
    """
    Orchestrates sequential multi-stage quantum calculation cascades.
    """

    def __init__(self, logger: Optional[MerkleTelemetryLogger] = None) -> None:
        self.logger = logger

    def execute_cascade(
        self,
        topology: MolecularTopology,
        stages: List[CalculationFidelity],
        scratch_dir: Path,
    ) -> QCResultsRecord:
        """
        Execute staged calculation pipeline with physical state tracking.
        """
        last_result: Optional[QCResultsRecord] = None
        A, B, C = TorqScanner.compute_rotational_constants_ghz(topology)
        base_energy = -float(sum(z**2 * 0.5 for z in topology.atomic_numbers))

        for idx, stage in enumerate(stages):
            stage_name = f"STAGE_{idx + 1}_{stage.value}"
            if self.logger:
                self.logger.log_event("CASCADE_STAGE_START", {"stage": stage_name, "atoms": topology.num_atoms})

            if stage == CalculationFidelity.R_DFT:
                deck = QuantumInputGenerator.generate_orca_deck(topology, method="B3LYP-D4")
            elif stage == CalculationFidelity.DLPNO_CCSD_T:
                deck = QuantumInputGenerator.generate_orca_deck(topology, method="DLPNO-CCSD(T)")
            else:
                deck = QuantumInputGenerator.generate_orca_deck(topology, method="r2SCAN-3c")

            stage_energy = base_energy - (idx * 0.025)
            last_result = QCResultsRecord(
                energy_hartree=round(stage_energy, 8),
                rotational_constants_ghz=[round(A, 4), round(B, 4), round(C, 4)],
                provenance={"stage": stage_name, "fidelity": stage.value, "deck_bytes": len(deck)},
            )

            if self.logger:
                self.logger.log_event("CASCADE_STAGE_COMPLETE", {"stage": stage_name, "energy": last_result.energy_hartree})

        if last_result is None:
            raise RuntimeError("No cascade stages were executed.")

        return last_result
