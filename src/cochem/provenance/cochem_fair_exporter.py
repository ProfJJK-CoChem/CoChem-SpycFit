"""
CoChem QCSchema v1 & W3C PROV-O FAIR Metadata Exporter (REQ-QC-097, REQ-QC-098).

Provenance:
- MolSSI QCSchema Specification v1 for Quantum Chemistry Interchange.
- W3C PROV-O: The PROV Data Model (W3C Recommendation 30 April 2013).
- HDF5 Provenance Root-Attribute Header Standardization.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

import h5py

from cochem.provenance.cochem_orcid_resolver import ORCIDAuthorProfile


class QCSchemaProvenanceRecord(BaseModel):
    """QCSchema v1 Compliant Provenance Record."""
    model_config = ConfigDict(frozen=True)

    creator: str = Field(default="CoChem-BASE", description="Software generating the data")
    version: str = Field(default="1.0.0", description="Software release version")
    routine: str = Field(default="quantum_chemistry_pipeline", description="Execution entrypoint routine")
    git_commit: Optional[str] = Field(default=None, description="VCS commit SHA")
    authors: List[ORCIDAuthorProfile] = Field(default_factory=list, description="Validated contributor profiles")
    credit_roles: List[str] = Field(default_factory=list, description="Aggregated CRediT contribution roles")
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="Calculation completion timestamp in UTC",
    )


class PROV_O_Graph(BaseModel):
    """W3C PROV-O JSON-LD Representation."""
    model_config = ConfigDict(populate_by_name=True)

    context: Dict[str, Any] = Field(
        alias="@context",
        default_factory=lambda: {
            "prov": "http://www.w3.org/ns/prov#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "cochem": "https://cochem.science/provenance/",
            "orcid": "https://orcid.org/",
        },
    )
    graph: List[Dict[str, Any]] = Field(alias="@graph", default_factory=list)


def inject_provenance_into_qcschema(
    qcschema_payload: Dict[str, Any],
    authors: List[ORCIDAuthorProfile],
    calculation_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inject verified ORCID provenance and CRediT contributions into a QCSchema dictionary.

    Args:
        qcschema_payload: Target QCSchema v1 compliant calculation dictionary.
        authors: List of validated author profiles.
        calculation_metadata: Optional extra execution metadata (engine, functional, basis).

    Returns:
        Dict[str, Any]: Updated QCSchema payload with injected provenance block.
    """
    updated = dict(qcschema_payload)
    meta = calculation_metadata or {}

    prov_record = QCSchemaProvenanceRecord(
        creator=meta.get("software_name", "CoChem-BASE"),
        version=meta.get("software_version", "1.0.0"),
        routine=meta.get("routine", "cochem_qc_pipeline"),
        git_commit=meta.get("git_commit"),
        authors=authors,
        credit_roles=list({role for a in authors for role in a.credit_roles}),
    )

    authors_serialized = [
        {
            "name": a.credit_name or f"{a.given_names} {a.family_name}",
            "orcid": a.orcid_id,
            "institution": a.primary_institution,
            "credit_roles": a.credit_roles,
        }
        for a in authors
    ]

    provenance_dict = {
        "creator": prov_record.creator,
        "version": prov_record.version,
        "routine": prov_record.routine,
        "git_commit": prov_record.git_commit,
        "timestamp": prov_record.timestamp_iso,
        "authors": authors_serialized,
        "credit_roles": prov_record.credit_roles,
    }

    updated["provenance"] = provenance_dict
    return updated


def generate_prov_o_jsonld(
    job_manifest: Dict[str, Any],
    authors: List[ORCIDAuthorProfile],
    output_json_path: Union[str, Path],
) -> Path:
    """
    Build a W3C PROV-O JSON-LD graph linking calculations, agents, and resulting artifacts.

    Args:
        job_manifest: Manifest describing calculation inputs, engine parameters, and output files.
        authors: List of validated author profiles.
        output_json_path: Target filepath to save JSON-LD output.

    Returns:
        Path: Path to created JSON-LD graph file.
    """
    job_id = job_manifest.get("job_id", "cochem-job-001")
    activity_id = f"cochem:activity/{job_id}"
    entity_output_id = f"cochem:entity/wavefunction-{job_id}"

    graph_nodes: List[Dict[str, Any]] = [
        # Activity
        {
            "@id": activity_id,
            "@type": "prov:Activity",
            "rdfs:label": f"Quantum Chemical Calculation {job_id}",
            "prov:startedAtTime": job_manifest.get("start_time", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "prov:endedAtTime": job_manifest.get("end_time", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "cochem:method": job_manifest.get("method", "DFT"),
            "cochem:basis": job_manifest.get("basis", "def2-TZVP"),
        },
        # Generated Entity
        {
            "@id": entity_output_id,
            "@type": "prov:Entity",
            "rdfs:label": f"Wavefunction and Energetics {job_id}",
            "prov:wasGeneratedBy": {"@id": activity_id},
        },
    ]

    # Agents (Authors)
    for auth in authors:
        agent_id = f"https://orcid.org/{auth.orcid_id}"
        agent_node = {
            "@id": agent_id,
            "@type": ["prov:Agent", "prov:Person"],
            "rdfs:label": auth.credit_name or f"{auth.given_names} {auth.family_name}",
            "cochem:orcid": auth.orcid_id,
            "cochem:institution": auth.primary_institution,
        }
        graph_nodes.append(agent_node)

        # Association / Attribution
        graph_nodes.append({
            "@type": "prov:Association",
            "prov:activity": {"@id": activity_id},
            "prov:agent": {"@id": agent_id},
            "prov:hadRole": auth.credit_roles,
        })
        graph_nodes.append({
            "@type": "prov:Attribution",
            "prov:entity": {"@id": entity_output_id},
            "prov:agent": {"@id": agent_id},
        })

    prov_graph = PROV_O_Graph(
        graph=graph_nodes,
    )

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as fh:
        json.dump(prov_graph.model_dump(by_alias=True), fh, indent=2)

    return out_p


def inject_hdf5_provenance_headers(
    hdf5_path: Union[str, Path],
    authors: List[ORCIDAuthorProfile],
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Safely write root-level string attributes to an HDF5 data archive.

    Args:
        hdf5_path: Path to existing or new HDF5 file.
        authors: List of author profiles to serialize into headers.
        extra_attributes: Additional key-value metadata to attach.
    """
    p = Path(hdf5_path)
    if not p.exists():
        mode = "w"
    else:
        mode = "r+"

    orcids_str = ";".join(a.orcid_id for a in authors)
    names_str = ";".join(a.credit_name or f"{a.given_names} {a.family_name}" for a in authors)
    roles_str = ";".join(",".join(a.credit_roles) for a in authors)
    ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with h5py.File(str(p), mode) as h5f:
        h5f.attrs["creator_orcid"] = orcids_str
        h5f.attrs["creator_name"] = names_str
        h5f.attrs["credit_roles"] = roles_str
        h5f.attrs["provenance_timestamp"] = ts_now
        h5f.attrs["software_provenance"] = "CoChem-BASE (QCSchema & PROV-O FAIR Pipeline)"

        if extra_attributes:
            for k, v in extra_attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    h5f.attrs[k] = v
                else:
                    h5f.attrs[k] = json.dumps(v)
