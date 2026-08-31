"""# zero-stub anti-spoofing engine
Core Engine Package Forwarder.
"""
from cochem_base.core_engine.cochem_fs_security import LogStreamMultiplexer, seal_log_file, verify_log_integrity
from cochem_base.core_engine.cochem_core_telemetry_logger import HardwareTelemetryLogger, get_hardware_topology
from cochem_base.core_engine.anti_spoof_linter import LintViolation, scan_file, scan_directory
from cochem_base.core_engine.zero_trust_runner import (
    verify_quantum_state_invariants,
    execute_in_quarantine,
    QuantumInvariantViolationError,
    compute_array_shannon_entropy,
    QuarantineResult,
)
