"""
CoChem ORCID Identifier Validator and Public API Resolution Engine (REQ-QC-098).

Provenance:
- ISO/IEC 7064:2003, MOD 11-2 Checksum algorithm for ORCID and ISNI identifier validation.
- ORCID Public API v3.0 Specification.
- CRediT (Contributor Roles Taxonomy) standardization for provenance tracking.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cochem.core.cochem_exceptions import ORCIDValidationError

# ORCID regex pattern: 16 digits formatted as 4 groups of 4 separated by hyphens, check digit [0-9X]
ORCID_REGEX = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def validate_orcid_checksum(orcid_id: str) -> bool:
    """
    Validate an ORCID identifier using the ISO/IEC 7064 Mod 11,2 check-digit algorithm.

    Algorithm:
        1. Strip hyphens -> 16 characters.
        2. Process the first 15 digits:
           Total = 0
           For each digit d:
               Total = (Total + d) * 2
        3. Remainder = Total % 11
        4. Result = (12 - Remainder) % 11
        5. Check digit = 'X' if Result == 10 else str(Result)
        6. Match computed check digit against the 16th character.

    Args:
        orcid_id: The ORCID string (e.g. '0000-0002-1825-0097').

    Returns:
        bool: True if format and checksum are valid, False otherwise.
    """
    if not isinstance(orcid_id, str):
        return False

    cleaned = orcid_id.strip()
    if not ORCID_REGEX.match(cleaned):
        return False

    digits_only = cleaned.replace("-", "")
    if len(digits_only) != 16:
        return False

    base_digits = digits_only[:15]
    expected_check_digit = digits_only[15]

    total = 0
    for char in base_digits:
        if not char.isdigit():
            return False
        total = (total + int(char)) * 2

    remainder = total % 11
    result = (12 - remainder) % 11
    calculated_check_digit = "X" if result == 10 else str(result)

    return calculated_check_digit == expected_check_digit


class ORCIDAuthorProfile(BaseModel):
    """
    Author Profile with verified ORCID identifier and CRediT contribution roles.
    """
    model_config = ConfigDict(frozen=True)

    orcid_id: str = Field(description="16-character hyphenated ORCID identifier")
    given_names: str = Field(description="Author given / first names")
    family_name: str = Field(description="Author family / surname")
    credit_name: Optional[str] = Field(default=None, description="Preferred published display name")
    primary_institution: Optional[str] = Field(default=None, description="Primary research institution")
    institution_ror_id: Optional[str] = Field(default=None, description="Research Organization Registry (ROR) ID")
    credit_roles: List[str] = Field(default_factory=list, description="CRediT author contribution roles")

    @field_validator("orcid_id")
    @classmethod
    def check_orcid_validity(cls, v: str) -> str:
        cleaned = v.strip()
        if not validate_orcid_checksum(cleaned):
            raise ValueError(f"Invalid ORCID format or ISO 7064 Mod 11,2 checksum failure for '{v}'")
        return cleaned


def resolve_orcid_public_api(orcid_id: str, timeout_seconds: float = 5.0) -> ORCIDAuthorProfile:
    """
    Query the ORCID Public API v3.0 to resolve author profile metadata.

    Args:
        orcid_id: Verified ORCID identifier (e.g. '0000-0002-1825-0097').
        timeout_seconds: Network request timeout in seconds.

    Returns:
        ORCIDAuthorProfile: Resolved profile with parsed names and affiliations.

    Raises:
        ORCIDValidationError: If checksum fails or network resolution fails.
    """
    cleaned_id = orcid_id.strip()
    if not validate_orcid_checksum(cleaned_id):
        raise ORCIDValidationError(
            orcid_id=cleaned_id,
            reason="Failed ISO 7064 Mod 11,2 checksum validation prior to API resolution.",
        )

    url = f"https://pub.orcid.org/v3.0/{cleaned_id}/record"
    headers = {
        "Accept": "application/json",
        "User-Agent": "CoChem-BASE/1.0 (Quantum Chemistry FAIR Archival Suite)",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ORCIDValidationError(
                    orcid_id=cleaned_id,
                    reason=f"ORCID Public API returned HTTP status {response.status}",
                )
            raw_data = response.read().decode("utf-8")
            data = json.loads(raw_data)

            # Parse Person Names
            person = data.get("person", {})
            name_obj = person.get("name", {}) or {}
            given_names = name_obj.get("given-names", {}).get("value", "") if name_obj.get("given-names") else ""
            family_name = name_obj.get("family-name", {}).get("value", "") if name_obj.get("family-name") else ""
            credit_name = name_obj.get("credit-name", {}).get("value") if name_obj.get("credit-name") else None

            # Parse Affiliations
            activities = data.get("activities-summary", {}) or {}
            employments = activities.get("employments", {}).get("affiliation-group", []) or []
            primary_institution = None
            institution_ror_id = None

            if employments and len(employments) > 0:
                first_group = employments[0]
                summaries = first_group.get("summaries", [])
                if summaries:
                    summary = summaries[0].get("employment-summary", {})
                    org = summary.get("organization", {})
                    primary_institution = org.get("name")
                    disambiguated = org.get("disambiguated-organization", {})
                    if disambiguated and disambiguated.get("disambiguation-source") == "ROR":
                        institution_ror_id = disambiguated.get("disambiguated-organization-identifier")

            return ORCIDAuthorProfile(
                orcid_id=cleaned_id,
                given_names=given_names or "Unknown",
                family_name=family_name or "Unknown",
                credit_name=credit_name,
                primary_institution=primary_institution,
                institution_ror_id=institution_ror_id,
                credit_roles=[],
            )

    except urllib.error.HTTPError as err:
        raise ORCIDValidationError(
            orcid_id=cleaned_id,
            reason=f"HTTP {err.code} error from ORCID API: {err.reason}",
        ) from err
    except urllib.error.URLError as err:
        raise ORCIDValidationError(
            orcid_id=cleaned_id,
            reason=f"Network error contacting ORCID API: {err.reason}",
        ) from err
    except Exception as err:
        raise ORCIDValidationError(
            orcid_id=cleaned_id,
            reason=f"Unexpected error resolving ORCID record: {str(err)}",
        ) from err
