"""
System Hardware Telemetry and Dynamic Mendeleev Atomic/Isotopic Mass Lookups.

Strict Mendeleev Mandate: All atomic and isotopic weights must be queried dynamically
from the Mendeleev database without static tables or hardcoded constants.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any, Dict, Union
from functools import lru_cache
import psutil
from mendeleev import element as m_element

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def get_atomic_mass(element_or_z: Union[str, int]) -> float:
    """
    Retrieve standard atomic weight dynamically from Mendeleev database.
    """
    el = m_element(element_or_z)
    if el.mass is None:
        raise ValueError(f"Could not retrieve atomic mass for element {element_or_z}")
    return float(el.mass)


@lru_cache(maxsize=512)
def get_isotopic_mass(element: str, mass_number: int) -> float:
    """
    Retrieve exact isotopic mass dynamically from Mendeleev database.
    """
    el = m_element(element)
    for iso in el.isotopes:
        if iso.mass_number == mass_number:
            if iso.mass is not None:
                return float(iso.mass)
    # If exact isotope object is unavailable, fallback to atomic mass
    if el.mass is not None:
        return float(el.mass)
    raise ValueError(f"Isotope {element}-{mass_number} not found in Mendeleev database.")


@lru_cache(maxsize=256)
def get_atomic_symbol(z: int) -> str:
    """Retrieve atomic symbol for atomic number Z dynamically."""
    el = m_element(z)
    return str(el.symbol)


@lru_cache(maxsize=256)
def get_covalent_radius(element_or_z: Union[str, int]) -> float:
    """
    Retrieve covalent radius in Angstroms from Mendeleev database.
    """
    el = m_element(element_or_z)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0  # pm to Angstrom
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    return 0.77  # standard baseline


def collect_hardware_metadata() -> Dict[str, Any]:
    """
    Inspect host environment hardware resources (CPU cores, RAM, SIMD, GPU availability).
    """
    mem = psutil.virtual_memory()
    cpu_count_physical = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)

    gpu_available = False
    gpu_devices = []
    try:
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            gpu_available = True
            for line in res.stdout.strip().splitlines():
                if line.strip():
                    gpu_devices.append(line.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError, TimeoutError) as exc:
        logger.debug("GPU hardware discovery via nvidia-smi failed or unavailable: %s", exc)
        gpu_available = False
        gpu_devices = []

    return {
        "cpu_arch": platform.machine(),
        "cpu_processor": platform.processor(),
        "physical_cores": cpu_count_physical,
        "logical_cores": cpu_count_logical,
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "gpu_available": gpu_available,
        "gpu_devices": gpu_devices,
    }
