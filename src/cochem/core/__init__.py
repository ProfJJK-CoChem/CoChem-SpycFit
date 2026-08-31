"""CoChem Core Chemical Data, Topology, and Invariant Modules."""

from .cochem_elements import (
    get_available_isotopes,
    get_ciaaw_atomic_mass,
    get_ciaaw_isotope_mass,
    get_element_atomic_number,
    get_element_electronegativity,
    get_element_name,
    get_element_symbol,
    validate_charge_and_multiplicity,
)
from .airgap_arbiter import AirGapArbiter
from .mendeleev_invariants import MendeleevResolver, mendeleev_resolver
from .topology_exceptions import (
    AirGapPartitionViolationError,
    CoChemTopologyException,
    MendeleevInvariantError,
    PartitionMountError,
    SWMRCorruptionError,
    SWMRFileLockTimeoutError,
    TelemetryQueueOverflowError,
)
from .topology_models import (
    EngineHealthSummary,
    PartitionDescriptor,
    PartitionPermission,
    PartitionType,
    SWMRFileLockDescriptor,
    TelemetryEventPayload,
    TelemetrySeverity,
)

__all__ = [
    # Element functions
    "get_element_atomic_number",
    "get_ciaaw_atomic_mass",
    "get_ciaaw_isotope_mass",
    "get_available_isotopes",
    "get_element_symbol",
    "get_element_name",
    "get_element_electronegativity",
    "validate_charge_and_multiplicity",
    # Topology
    "AirGapArbiter",
    "MendeleevResolver",
    "mendeleev_resolver",
    # Exceptions
    "CoChemTopologyException",
    "AirGapPartitionViolationError",
    "PartitionMountError",
    "SWMRFileLockTimeoutError",
    "SWMRCorruptionError",
    "TelemetryQueueOverflowError",
    "MendeleevInvariantError",
    # Models
    "PartitionType",
    "PartitionPermission",
    "TelemetrySeverity",
    "PartitionDescriptor",
    "SWMRFileLockDescriptor",
    "TelemetryEventPayload",
    "EngineHealthSummary",
]
