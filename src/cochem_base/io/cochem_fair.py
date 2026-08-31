"""# zero-stub anti-spoofing engine
CoChem FAIR Metadata Assembler & Dynamic Mendeleev Mass Resolver (cochem_base.io.cochem_fair)

Generates schema.org Dataset JSON-LD FAIR provenance metadata, exports QCSchema v2 structures
with CODATA 2022 constants, and resolves exact atomic and isotopic masses dynamically via Mendeleev.

Governing Requirements:
- REQ-62 (FR-62.1 - FR-62.4): FAIR Metadata Assembler & Dynamic Mendeleev Resolver.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mendeleev import element

from cochem_base.core.cochem_version import VCSProvenance

# CODATA 2022 Physical Conversion Constants
ANGSTROM_TO_BOHR: float = 1.889726125
HARTREE_TO_KCAL_MOL: float = 627.509474


def get_element_mass(symbol: str) -> float:
    """Dynamically retrieve IUPAC standard atomic weight from the Mendeleev library."""
    el = element(symbol.strip())
    if el.mass is None:
        raise ValueError(f"No standard atomic weight available for element '{symbol}'.")
    return float(el.mass)


def get_isotopic_mass(symbol: str, mass_number: int) -> float:
    """Dynamically retrieve exact isotopic mass from the Mendeleev library."""
    el = element(symbol.strip())
    for iso in el.isotopes:
        if iso.mass_number == mass_number:
            if iso.mass is not None:
                return float(iso.mass)
    # Fallback to standard atomic mass if specific isotope not listed
    return float(el.mass) if el.mass else float(mass_number)


def compute_center_of_mass(
    atomic_symbols: List[str],
    coordinates_angstrom: np.ndarray,
    isotopes: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute the center-of-mass vector in Angstroms using dynamic Mendeleev masses."""
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    masses = []
    for idx, sym in enumerate(atomic_symbols):
        if isotopes and idx < len(isotopes) and isotopes[idx] is not None:
            masses.append(get_isotopic_mass(sym, isotopes[idx]))
        else:
            masses.append(get_element_mass(sym))

    m_arr = np.asarray(masses, dtype=np.float64)
    total_mass = float(np.sum(m_arr))
    com = np.sum(coords * m_arr[:, np.newaxis], axis=0) / total_mass
    return com


def compute_inertia_tensor(
    atomic_symbols: List[str],
    coordinates_angstrom: np.ndarray,
    isotopes: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute the 3x3 moment of inertia tensor: I_g = sum m_i r_i^2 (amu * Angstrom^2)
    relative to the molecular center of mass.
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    com = compute_center_of_mass(atomic_symbols, coords, isotopes=isotopes)
    r_com = coords - com  # Center geometry

    masses = []
    for idx, sym in enumerate(atomic_symbols):
        if isotopes and idx < len(isotopes) and isotopes[idx] is not None:
            masses.append(get_isotopic_mass(sym, isotopes[idx]))
        else:
            masses.append(get_element_mass(sym))
    m = np.asarray(masses, dtype=np.float64)

    x, y, z = r_com[:, 0], r_com[:, 1], r_com[:, 2]

    I_xx = np.sum(m * (y**2 + z**2))
    I_yy = np.sum(m * (x**2 + z**2))
    I_zz = np.sum(m * (x**2 + y**2))
    I_xy = -np.sum(m * x * y)
    I_xz = -np.sum(m * x * z)
    I_yz = -np.sum(m * y * z)

    I = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ], dtype=np.float64)
    return I


def generate_fair_provenance_json(
    job_id: str,
    formula: str,
    smiles: str,
    inchi_key: str,
    vcs_provenance: Optional[VCSProvenance] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate FAIR4Chem schema.org Dataset JSON-LD provenance record.
    """
    record_uuid = str(uuid.uuid4())
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    doc: Dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "identifier": f"urn:uuid:{record_uuid}",
        "name": f"CoChem Computational Provenance Dataset: {job_id}",
        "description": f"FAIR-compliant computational chemistry results and conformer lineage for {formula}.",
        "dateCreated": timestamp,
        "chemicalStructure": {
            "formula": formula,
            "smiles": smiles,
            "inchiKey": inchi_key,
        },
        "publisher": {
            "@type": "Organization",
            "name": "CoChem Autonomous Scientific Swarm",
        },
    }

    if vcs_provenance:
        doc["softwareVersion"] = vcs_provenance.to_dict()

    if extra_metadata:
        doc["computationalMetadata"] = extra_metadata

    return doc


def export_qcschema_v2(
    atomic_symbols: List[str],
    geometry_angstrom: np.ndarray,
    energy_hartree: float,
    rotational_constants_mhz: Optional[Tuple[float, float, float]] = None,
    vcs_provenance: Optional[VCSProvenance] = None,
) -> Dict[str, Any]:
    """
    Export calculation results conforming to the QCSchema v2 specification.
    Coordinates are normalized to Bohr (a0) and energies to Hartrees / kcal/mol.
    """
    coords_ang = np.asarray(geometry_angstrom, dtype=np.float64)
    coords_bohr = (coords_ang * ANGSTROM_TO_BOHR).flatten().tolist()
    atomic_numbers = [int(element(sym).atomic_number) for sym in atomic_symbols]

    schema: Dict[str, Any] = {
        "schema_name": "qcschema_output",
        "schema_version": 2,
        "molecule": {
            "symbols": atomic_symbols,
            "atomic_numbers": atomic_numbers,
            "geometry": coords_bohr,
            "units": "bohr",
        },
        "properties": {
            "return_energy": float(energy_hartree),
            "return_energy_kcal_mol": round(float(energy_hartree) * HARTREE_TO_KCAL_MOL, 6),
        },
        "provenance": {
            "creator": "CoChem Swarm Engine",
            "version": vcs_provenance.full_identifier if vcs_provenance else "0.1.0",
            "routine": "qcschema_v2_export",
        },
    }

    if rotational_constants_mhz is not None:
        schema["properties"]["rotational_constants_mhz"] = list(rotational_constants_mhz)

    return schema
