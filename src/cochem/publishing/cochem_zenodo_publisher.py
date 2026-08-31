"""
CoChem One-Click Zenodo FAIR Data Archival & DOI Minting Engine (REQ-QC-097, REQ-QC-098).

Provenance:
- Zenodo REST API v2 Specifications and DataCite 4.4 Metadata Schema.
- FAIR Data Guiding Principles (Findable, Accessible, Interoperable, Reusable).
- Ed25519 Cryptographic Integrity Manifests and Streaming SHA-256 Hashing.
"""

from __future__ import annotations

import json
import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_exceptions import ZenodoAPIError
from cochem.provenance.cochem_signer import compute_file_checksums, sign_manifest


class ZenodoAuthorMetadata(BaseModel):
    """Author metadata formatted for Zenodo and DataCite schemas."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Author name in 'Family Name, Given Name' format")
    affiliation: Optional[str] = Field(default=None, description="Author institutional affiliation")
    orcid: Optional[str] = Field(default=None, description="Author verified ORCID identifier")


class ZenodoDepositMetadata(BaseModel):
    """Complete deposition metadata payload conforming to DataCite 4.4 and Zenodo requirements."""
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of dataset / calculation archive")
    description: str = Field(description="Detailed HTML or Markdown description of calculations")
    creators: List[ZenodoAuthorMetadata] = Field(description="List of dataset creators")
    upload_type: Literal["dataset", "software", "publication"] = Field(
        default="dataset", description="Deposition artifact type"
    )
    access_right: Literal["open", "embargoed", "restricted"] = Field(
        default="open", description="Access permission tier"
    )
    license: str = Field(default="CC-BY-4.0", description="Open science data license")
    keywords: List[str] = Field(
        default_factory=lambda: ["quantum-chemistry", "dft", "fair-data", "computational-chemistry"],
        description="Search indexing tags"
    )
    related_identifiers: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Cross-references to journal DOIs, GitHub repos, or parent datasets"
    )


class ZenodoPublishResult(BaseModel):
    """Result returned upon successful reservation or publication of a Zenodo deposition."""
    model_config = ConfigDict(frozen=True)

    deposition_id: int = Field(description="Unique Zenodo numeric deposition ID")
    reserved_doi: str = Field(description="Reserved DOI string (e.g. '10.5281/zenodo.1234567')")
    published_doi: Optional[str] = Field(default=None, description="Live minted DOI if published")
    is_published: bool = Field(default=False, description="True if published and immutable")
    record_url: str = Field(description="Web browser URL to access the deposition")
    sha256_checksum: str = Field(description="SHA-256 cryptographic digest of uploaded tar archive")


def package_calculation_archive(
    project_dir: Union[str, Path],
    output_tar_path: Union[str, Path],
) -> Tuple[Path, str]:
    """
    Package calculation inputs (.inp), outputs (.out), geometries (.xyz), and metadata into a tar.gz archive.

    Args:
        project_dir: Source folder containing calculation artifacts.
        output_tar_path: Output archive path (e.g. 'cochem_calculation_archive.tar.gz').

    Returns:
        Tuple[Path, str]: (Path to tar.gz, SHA-256 hex digest).
    """
    proj_p = Path(project_dir)
    out_p = Path(output_tar_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    target_extensions = {".inp", ".out", ".xyz", ".gbw", ".cube", ".h5", ".json", ".bib", ".tex", ".pdf"}

    with tarfile.open(str(out_p), "w:gz") as tar:
        if proj_p.is_dir():
            for root, _, files in os.walk(str(proj_p)):
                for file in files:
                    fp = Path(root) / file
                    if fp.suffix.lower() in target_extensions:
                        rel = fp.relative_to(proj_p)
                        tar.add(str(fp), arcname=str(rel))
        elif proj_p.is_file():
            tar.add(str(proj_p), arcname=proj_p.name)

    checksums = compute_file_checksums(out_p)
    return out_p, checksums["sha256"]


def build_datacite_metadata_payload(metadata: ZenodoDepositMetadata) -> Dict[str, Any]:
    """
    Construct the nested metadata dictionary required by the Zenodo REST API.

    Args:
        metadata: ZenodoDepositMetadata instance.

    Returns:
        Dict[str, Any]: JSON payload ready for transmission.
    """
    creators_list = []
    for c in metadata.creators:
        c_dict: Dict[str, Any] = {"name": c.name}
        if c.affiliation:
            c_dict["affiliation"] = c.affiliation
        if c.orcid:
            c_dict["orcid"] = c.orcid
        creators_list.append(c_dict)

    return {
        "metadata": {
            "title": metadata.title,
            "upload_type": metadata.upload_type,
            "description": metadata.description,
            "creators": creators_list,
            "access_right": metadata.access_right,
            "license": metadata.license,
            "keywords": metadata.keywords,
            "related_identifiers": metadata.related_identifiers,
        }
    }


class ZenodoClient:
    """Zenodo REST API v2 Connector with sandbox support."""

    def __init__(self, api_token: str, sandbox: bool = True, timeout_seconds: float = 30.0) -> None:
        self.api_token = api_token.strip()
        self.sandbox = sandbox
        self.timeout = timeout_seconds
        self.base_url = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"

    def _headers(self, content_type: str = "application/json") -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": content_type,
            "User-Agent": "CoChem-BASE/1.0 (Quantum Chemistry FAIR Archival Engine)",
        }

    def create_deposition(self) -> int:
        """Create a new empty deposition resource on Zenodo."""
        url = f"{self.base_url}/deposit/depositions"
        req = urllib.request.Request(url, data=b"{}", headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return int(data["id"])
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            raise ZenodoAPIError(status_code=err.code, response_body=body) from err
        except Exception as err:
            raise ZenodoAPIError(status_code=0, response_body=str(err)) from err

    def reserve_doi(self, deposition_id: int) -> str:
        """Reserve a DOI before publication."""
        url = f"{self.base_url}/deposit/depositions/{deposition_id}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                metadata = data.get("metadata", {})
                doi = metadata.get("prereserve_doi", {}).get("doi")
                if not doi:
                    doi = f"10.5281/zenodo.{deposition_id}"
                return doi
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            raise ZenodoAPIError(status_code=err.code, response_body=body) from err
        except Exception as err:
            raise ZenodoAPIError(status_code=0, response_body=str(err)) from err

    def upload_archive_bucket(self, deposition_id: int, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Upload archive file into deposition bucket."""
        fp = Path(file_path)
        if not fp.is_file():
            raise FileNotFoundError(f"Archive file not found at '{fp}'")

        # Zenodo API v2 uses bucket URLs or direct file endpoint
        url = f"{self.base_url}/deposit/depositions/{deposition_id}/files"

        # Read file binary
        file_bytes = fp.read_bytes()
        filename = fp.name

        # Multipart form data boundary
        boundary = "----CoChemZenodoBoundary123456789"
        body_parts = [
            f"--{boundary}".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8"),
            b"Content-Type: application/gzip",
            b"",
            file_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        payload = b"\r\n".join(body_parts)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            raise ZenodoAPIError(status_code=err.code, response_body=body) from err
        except Exception as err:
            raise ZenodoAPIError(status_code=0, response_body=str(err)) from err

    def update_metadata(self, deposition_id: int, metadata: ZenodoDepositMetadata) -> Dict[str, Any]:
        """Update deposition metadata."""
        url = f"{self.base_url}/deposit/depositions/{deposition_id}"
        payload = build_datacite_metadata_payload(metadata)
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data_bytes, headers=self._headers(), method="PUT")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            raise ZenodoAPIError(status_code=err.code, response_body=body) from err
        except Exception as err:
            raise ZenodoAPIError(status_code=0, response_body=str(err)) from err

    def publish_deposition(self, deposition_id: int) -> ZenodoPublishResult:
        """Publish deposition to make it permanent and immutable."""
        url = f"{self.base_url}/deposit/depositions/{deposition_id}/actions/publish"
        req = urllib.request.Request(url, data=b"{}", headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                doi = data.get("doi", f"10.5281/zenodo.{deposition_id}")
                html_url = data.get("links", {}).get("html", f"{self.base_url}/records/{deposition_id}")
                return ZenodoPublishResult(
                    deposition_id=deposition_id,
                    reserved_doi=doi,
                    published_doi=doi,
                    is_published=True,
                    record_url=html_url,
                    sha256_checksum=data.get("files", [{}])[0].get("checksum", ""),
                )
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            raise ZenodoAPIError(status_code=err.code, response_body=body) from err
        except Exception as err:
            raise ZenodoAPIError(status_code=0, response_body=str(err)) from err
