"""
Desktop PyQt6 / PySide6 User Interface Suite for CoChem-BASE.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CalculationDashboard": ("cochem_base.gui.dashboard", "CalculationDashboard"),
    "ToposControlPanel": ("cochem_base.gui.topos", "ToposControlPanel"),
    "TorqControlPanel": ("cochem_base.gui.torq", "TorqControlPanel"),
    "ScribeExportDialog": ("cochem_base.gui.scribe", "ScribeExportDialog"),
    "CoChemMainWindow": ("cochem_base.gui.main_window", "CoChemMainWindow"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
