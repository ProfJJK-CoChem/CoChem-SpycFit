"""# zero-stub anti-spoofing engine
CoChem Zero-Trust Quarantine Sandbox & Quantum Invariant Gatekeeper (cochem_base.core_engine.zero_trust_runner)

Enforces zero-mock execution invariants, sterile quarantine sandboxing, Shannon entropy array
validation, exact atomic orbital electron trace conservation (Tr(PS) = Ne), overlap positive-definiteness,
and virial ratio checks.

Governing Requirements:
- REQ-68 (FR-68.2 - FR-68.3), REQ-66: Zero-Trust Quarantine Sandbox & Quantum Gatekeeper.
"""

from __future__ import annotations

import atexit
import datetime
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from cochem_base.core_engine.cochem_fs_security import LogStreamMultiplexer, seal_log_file
from cochem_base.core_engine.anti_spoof_linter import scan_file

logger = logging.getLogger("zero_trust_runner")


class QuantumInvariantViolationError(Exception):
    """Raised when a quantum state, matrix invariant, or execution violates the zero-mock mandate."""
    pass


@dataclass
class QuarantineResult:
    """Result of a process execution within the sterile quarantine environment."""
    exit_code: int
    stdout: str
    stderr: str
    quarantine_dir: str
    duration_s: float
    passed: bool
    merkle_root: str
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "quarantine_dir": self.quarantine_dir,
            "duration_s": self.duration_s,
            "passed": self.passed,
            "merkle_root": self.merkle_root,
            "timed_out": self.timed_out,
        }


def compute_array_shannon_entropy(arr: np.ndarray) -> float:
    """
    Calculate the byte-level Shannon entropy of a raw numpy array.
    Used to reject synthetic mock arrays (e.g. np.zeros, np.ones, simple arithmetic progressions).
    Authentic quantum physical matrices have Shannon entropy > 3.5 bits/byte.
    """
    raw_bytes = np.asarray(arr).tobytes()
    if not raw_bytes:
        return 0.0

    counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(raw_bytes)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def verify_quantum_state_invariants(
    density_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    n_electrons: int,
    kinetic_energy: Optional[float] = None,
    potential_energy: Optional[float] = None,
    enforce_entropy_check: bool = True,
) -> bool:
    """
    Verify exact quantum mechanical invariants:
    1. Overlap matrix positive-definiteness: lambda_min(S) > 1.0e-6
    2. Density matrix Hermiticity: ||P - P^T||_inf < 1.0e-7
    3. AO Electron Count Invariant: |Tr(P S) - Ne| <= 1.0e-5
       (Direct check Tr(P) = Ne is strictly forbidden for non-orthogonal AO bases)
    4. Array Shannon entropy: H > 3.5 bits/byte (rejecting synthetic dummy arrays)
    5. Virial ratio check: -<V>/<T> in [1.98, 2.02] (if energies provided)
    """
    P = np.asarray(density_matrix, dtype=np.float64)
    S = np.asarray(overlap_matrix, dtype=np.float64)

    if P.ndim != 2 or S.ndim != 2 or P.shape != S.shape or P.shape[0] != P.shape[1]:
        raise QuantumInvariantViolationError(
            f"[ERR_QUANTUM_INVARIANT]: Dimension mismatch: P shape {P.shape} != S shape {S.shape}"
        )

    # 1. Overlap Positive-Definiteness
    s_eigvals = np.linalg.eigvalsh(S)
    min_s_eig = float(np.min(s_eigvals))
    if min_s_eig <= 1.0e-6:
        raise QuantumInvariantViolationError(
            f"[ERR_OVERLAP_NON_POSITIVE]: Overlap matrix minimum eigenvalue ({min_s_eig:.2e}) <= 1.0e-6 threshold."
        )

    # 2. Density Matrix Hermiticity / Symmetry
    herm_diff = float(np.max(np.abs(P - P.T)))
    if herm_diff >= 1.0e-7:
        raise QuantumInvariantViolationError(
            f"[ERR_DENSITY_HERMITICITY]: Density matrix asymmetry ({herm_diff:.2e}) exceeds 1.0e-7 threshold."
        )

    # 3. Exact AO Electron Count Invariant: Tr(P S) == Ne
    # Tr(P @ S) = sum_ij (P_ij * S_ji)
    computed_nelec = float(np.trace(P @ S))
    diff_nelec = abs(computed_nelec - float(n_electrons))
    if diff_nelec > 1.0e-5:
        raise QuantumInvariantViolationError(
            f"[ERR_ELECTRON_CONSERVATION]: Exact AO electron trace Tr(PS) = {computed_nelec:.6f} != {n_electrons} (diff = {diff_nelec:.2e} > 1.0e-5)."
        )

    # 4. Shannon Entropy Scoping over array bytes
    if enforce_entropy_check and P.size >= 16:
        entropy_p = compute_array_shannon_entropy(P)
        if entropy_p < 3.5:
            raise QuantumInvariantViolationError(
                f"[ERR_SYNTHETIC_MOCK_ARRAY]: Density matrix Shannon entropy ({entropy_p:.2f} bits/byte) < 3.5. Synthetic mock array rejected."
            )

    # 5. Virial Ratio Check (if kinetic & potential energies provided)
    if kinetic_energy is not None and potential_energy is not None:
        if abs(kinetic_energy) > 1.0e-6:
            virial_ratio = -potential_energy / kinetic_energy
            if not (1.98 <= virial_ratio <= 2.02):
                raise QuantumInvariantViolationError(
                    f"[ERR_VIRIAL_RATIO]: Virial theorem ratio -<V>/<T> = {virial_ratio:.4f} outside physical bound [1.98, 2.02]."
                )

    return True


def trigger_physics_autopsy(quarantine_dir: Path, failure_reason: str) -> Path:
    """Generate a Physics_Autopsy_Report.md upon zero-trust validation abort."""
    autopsy_path = quarantine_dir / "Physics_Autopsy_Report.md"
    content = f"""# [HARD_ABORT: ZERO-MOCK VIOLATION] Physics Autopsy Report

**Timestamp:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}  
**Quarantine Sandbox:** `{quarantine_dir.as_posix()}`  
**Failure Diagnosis:** `{failure_reason}`  

## Root Cause Analysis
The execution was aborted prior to publishing because quantum mechanical state invariants
or anti-spoofing requirements failed verification.

- All quarantine logs have been sealed with cross-platform read-only ACLs.
- Implementation branches must resolve the physical discrepancy without synthetic bypasses.
"""
    autopsy_path.write_text(content, encoding="utf-8")
    seal_log_file(autopsy_path)
    return autopsy_path


def execute_in_quarantine(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 300.0,
    scan_scripts: bool = True,
) -> QuarantineResult:
    """
    Execute a calculation or test binary within an isolated sterile quarantine sandbox.
    """
    quarantine_id = f"cochem_exec_{uuid.uuid4()}"
    base_temp = Path(tempfile.gettempdir())
    quarantine_dir = base_temp / quarantine_id
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    log_file = quarantine_dir / "quarantine_execution.log"
    start_time = time.perf_counter()

    try:
        # Pre-execution AST scan if Python script
        if scan_scripts and len(command) >= 2 and command[1].endswith(".py"):
            script_path = Path(command[1]).resolve()
            if script_path.exists():
                violations = scan_file(script_path)
                if violations:
                    trigger_physics_autopsy(quarantine_dir, f"AST Linter detected {len(violations)} mock violations.")
                    raise QuantumInvariantViolationError(f"[ERR_AST_MOCK_DETECTED]: Script contains forbidden mock tokens: {script_path.as_posix()}")

        with LogStreamMultiplexer(log_file) as mux:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd or quarantine_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )

            stdout_text, stderr_text = proc.communicate(timeout=timeout)
            if stdout_text:
                mux.write(stdout_text)
            if stderr_text:
                mux.write(stderr_text)

            merkle_root = mux.get_merkle_root()

        seal_log_file(log_file)
        duration = time.perf_counter() - start_time

        return QuarantineResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            quarantine_dir=str(quarantine_dir),
            duration_s=round(duration, 4),
            passed=(proc.returncode == 0),
            merkle_root=merkle_root,
        )

    except subprocess.TimeoutExpired:
        proc.kill()
        duration = time.perf_counter() - start_time
        trigger_physics_autopsy(quarantine_dir, f"Execution timed out after {timeout}s")
        return QuarantineResult(
            exit_code=-1,
            stdout="",
            stderr=f"Process timed out after {timeout} seconds.",
            quarantine_dir=str(quarantine_dir),
            duration_s=round(duration, 4),
            passed=False,
            merkle_root="",
            timed_out=True,
        )
    except Exception as e:
        trigger_physics_autopsy(quarantine_dir, str(e))
        raise
