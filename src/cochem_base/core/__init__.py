"""
Core foundation subpackage for CoChem-BASE.

Exposes glossary, domain data models, security sandbox controls,
and dynamic isotopic metadata.
"""

from __future__ import annotations

from cochem_base.core.glossary import (
    UnitConversionConstants,
    CalculationFidelity,
)
from cochem_base.core.models import (
    MolecularTopology,
    QCResultsRecord,
    ConformerRecord,
    CalculationJobPayload,
)
from cochem_base.core.security import (
    enforce_readonly_acl,
    remove_readonly_acl,
    create_tripartite_sandbox,
)
from cochem_base.core.metadata import (
    get_atomic_mass,
    get_isotopic_mass,
    get_atomic_symbol,
    get_covalent_radius,
    collect_hardware_metadata,
)

__all__ = [
    "UnitConversionConstants",
    "CalculationFidelity",
    "MolecularTopology",
    "QCResultsRecord",
    "ConformerRecord",
    "CalculationJobPayload",
    "enforce_readonly_acl",
    "remove_readonly_acl",
    "create_tripartite_sandbox",
    "get_atomic_mass",
    "get_isotopic_mass",
    "get_atomic_symbol",
    "get_covalent_radius",
    "collect_hardware_metadata",
]
