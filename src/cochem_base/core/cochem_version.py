"""# zero-stub anti-spoofing engine
CoChem VCS Working-Tree Provenance & Diff Hashing Engine (cochem_base.core.cochem_version)

Provides comprehensive Git working-tree introspection, dirty tree SHA-256 diff hashing,
and deterministic fallback for air-gapped / containerized execution environments.

Governing Requirements:
- REQ-64 (FR-64.1 - FR-64.4): VCS Working-Tree Provenance & Diff Hashing.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional

logger = logging.getLogger("cochem_version")


@dataclass(frozen=True)
class VCSProvenance:
    """Immutable VCS Working-Tree Provenance Record."""
    commit_hash: str
    version_tag: Optional[str]
    is_dirty: bool
    diff_hash: Optional[str]
    branch: str
    build_timestamp: str
    full_identifier: str

    def to_dict(self) -> dict:
        return {
            "commit_hash": self.commit_hash,
            "version_tag": self.version_tag,
            "is_dirty": self.is_dirty,
            "diff_hash": self.diff_hash,
            "branch": self.branch,
            "build_timestamp": self.build_timestamp,
            "full_identifier": self.full_identifier,
        }


def _run_git_cmd(args: list[str], cwd: Path) -> Optional[str]:
    """Execute a git command safely, returning stripped stdout or None on failure."""
    try:
        res = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if res.returncode == 0 and res.stdout is not None:
            return res.stdout.strip()
        if res.stderr:
            logger.debug(f"Git command failed with code {res.returncode}: {res.stderr.strip()}")
    except (subprocess.SubprocessError, OSError) as err:
        logger.debug(f"Git execution error: {err}")
    return None


def get_vcs_provenance(repo_root: Optional[Path] = None) -> VCSProvenance:
    """
    Introspect the active VCS working tree, capture dirty-tree diff hashes,
    or fallback cleanly to build manifests in air-gapped environments.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if repo_root is None:
        # Default to repository root from this file location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
    else:
        repo_root = Path(repo_root).resolve()
        
    git_dir = repo_root / ".git"
    
    # Check for Git availability and repository presence
    if git_dir.exists():
        commit_hash = _run_git_cmd(["rev-parse", "HEAD"], repo_root)
        version_tag = _run_git_cmd(["describe", "--tags", "--always"], repo_root)
        branch = _run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], repo_root) or "detached"
        status_porcelain = _run_git_cmd(["status", "--porcelain"], repo_root)
        
        if commit_hash:
            is_dirty = bool(status_porcelain and len(status_porcelain.strip()) > 0)
            diff_hash: Optional[str] = None
            
            if is_dirty:
                raw_diff = _run_git_cmd(["diff", "HEAD"], repo_root) or ""
                # Also include untracked status if diff is empty
                diff_payload = (raw_diff + "\n" + (status_porcelain or "")).encode("utf-8")
                diff_hash = hashlib.sha256(diff_payload).hexdigest()
                
                base_ident = version_tag or commit_hash[:8]
                full_ident = f"{base_ident}+dirty.{diff_hash[:8]}"
            else:
                full_ident = version_tag or commit_hash[:8]
                
            return VCSProvenance(
                commit_hash=commit_hash,
                version_tag=version_tag,
                is_dirty=is_dirty,
                diff_hash=diff_hash,
                branch=branch,
                build_timestamp=timestamp,
                full_identifier=full_ident,
            )
            
    # Air-Gapped & Container Fallback Mechanism
    manifest_file = repo_root / ".build_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return VCSProvenance(
                commit_hash=manifest.get("commit_hash", "AIR_GAPPED_BUILD"),
                version_tag=manifest.get("version_tag", "v0.1.0"),
                is_dirty=manifest.get("is_dirty", False),
                diff_hash=manifest.get("diff_hash", None),
                branch=manifest.get("branch", "release"),
                build_timestamp=manifest.get("build_timestamp", timestamp),
                full_identifier=manifest.get("full_identifier", "v0.1.0-airgap"),
            )
        except Exception as e:
            logger.warning(f"Failed to read .build_manifest.json: {e}")
            
    # Final Untracked Fallback
    logger.warning("No VCS repository or build manifest found. Operating as UNTRACKED_BUILD.")
    return VCSProvenance(
        commit_hash="UNTRACKED_BUILD",
        version_tag=None,
        is_dirty=False,
        diff_hash=None,
        branch="unknown",
        build_timestamp=timestamp,
        full_identifier="UNTRACKED_BUILD",
    )
