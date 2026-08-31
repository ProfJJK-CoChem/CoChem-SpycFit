"""
Quantum Chemistry Calculation Error Classifier and Self-Healing Recovery Translator.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Tuple


class QuantumFailureType(str, Enum):
    """Classified failure modes in quantum chemistry outputs."""

    SCF_NON_CONVERGENCE = "SCF_NON_CONVERGENCE"
    BASIS_LINEAR_DEPENDENCY = "BASIS_LINEAR_DEPENDENCY"
    GEOMETRY_STEP_OVERSHOOT = "GEOMETRY_STEP_OVERSHOOT"
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"
    IMAGINARY_FREQUENCY_GROUND_STATE = "IMAGINARY_FREQUENCY_GROUND_STATE"
    UNRECOGNIZED_CRASH = "UNRECOGNIZED_CRASH"


class ErrorTranslator:
    """
    Parses calculation error output logs and recommends corrective recovery strategies.
    """

    PATTERNS: Dict[QuantumFailureType, List[re.Pattern]] = {
        QuantumFailureType.SCF_NON_CONVERGENCE: [
            re.compile(r"SCF\s+NOT\s+CONVERGED", re.IGNORECASE),
            re.compile(r"SCF\s+failed\s+to\s+converge", re.IGNORECASE),
            re.compile(r"Convergence\s+criterion\s+not\s+met", re.IGNORECASE),
            re.compile(r"ENERGY\s+DID\s+NOT\s+CONVERGE", re.IGNORECASE),
        ],
        QuantumFailureType.BASIS_LINEAR_DEPENDENCY: [
            re.compile(r"linear\s+dependenc(?:y|ies)", re.IGNORECASE),
            re.compile(r"near-linear\s+dependency", re.IGNORECASE),
            re.compile(r"overlap\s+matrix\s+is\s+singular", re.IGNORECASE),
        ],
        QuantumFailureType.GEOMETRY_STEP_OVERSHOOT: [
            re.compile(r"Geometry\s+optimization\s+failed", re.IGNORECASE),
            re.compile(r"MAX\s+CYCLES\s+EXCEEDED", re.IGNORECASE),
            re.compile(r"Step\s+overshoot\s+detected", re.IGNORECASE),
            re.compile(r"Trust\s+radius\s+too\s+small", re.IGNORECASE),
        ],
        QuantumFailureType.MEMORY_EXCEEDED: [
            re.compile(r"out\s+of\s+memory", re.IGNORECASE),
            re.compile(r"Memory\s+allocation\s+failed", re.IGNORECASE),
            re.compile(r"Increase\s+%maxcore", re.IGNORECASE),
            re.compile(r"std::bad_alloc", re.IGNORECASE),
        ],
        QuantumFailureType.IMAGINARY_FREQUENCY_GROUND_STATE: [
            re.compile(r"imaginary\s+mode", re.IGNORECASE),
            re.compile(r"imaginary\s+frequency", re.IGNORECASE),
            re.compile(r"transition\s+state\s+found\s+for\s+ground\s+state", re.IGNORECASE),
        ],
    }

    REMEDIES: Dict[QuantumFailureType, List[str]] = {
        QuantumFailureType.SCF_NON_CONVERGENCE: [
            "Enable SOSCF or SlowConv damping",
            "Increase max SCF iterations to 200",
            "Apply fractional level shifting (Shift 0.2)",
            "Switch to DefGrid2 or loose initial grid",
        ],
        QuantumFailureType.BASIS_LINEAR_DEPENDENCY: [
            "Increase basis set linear dependency threshold (Thresh 1e-6)",
            "Remove redundant diffuse functions",
            "Switch to uncontracted or def2 basis set",
        ],
        QuantumFailureType.GEOMETRY_STEP_OVERSHOOT: [
            "Reduce trust radius (Trust 0.1)",
            "Switch coordinate system to redundant internal coordinates",
            "Pre-optimize structure with semi-empirical GFN2-xTB",
        ],
        QuantumFailureType.MEMORY_EXCEEDED: [
            "Increase maxcore memory allocation (%maxcore)",
            "Switch algorithm to direct or RI-J (RIJCOSX)",
            "Reduce active MPI process concurrency",
        ],
        QuantumFailureType.IMAGINARY_FREQUENCY_GROUND_STATE: [
            "Displace coordinates along imaginary normal mode and re-optimize",
            "Tighten geometry optimization convergence criteria",
            "Increase numerical differentiation grid",
        ],
        QuantumFailureType.UNRECOGNIZED_CRASH: [
            "Inspect raw stdout/stderr logs",
            "Verify molecular geometry validity and atom distances",
        ],
    }

    @classmethod
    def classify_log(cls, log_text: str) -> Tuple[QuantumFailureType, str, List[str]]:
        """
        Classify quantum chemistry failure from log content and return failure type, snippet, and remedies.
        """
        for failure_type, patterns in cls.PATTERNS.items():
            for pat in patterns:
                m = pat.search(log_text)
                if m:
                    start = max(0, m.start() - 100)
                    end = min(len(log_text), m.end() + 100)
                    snippet = log_text[start:end].strip()
                    return failure_type, snippet, cls.REMEDIES[failure_type]

        return (
            QuantumFailureType.UNRECOGNIZED_CRASH,
            log_text[-300:].strip() if len(log_text) > 300 else log_text.strip(),
            cls.REMEDIES[QuantumFailureType.UNRECOGNIZED_CRASH],
        )
