"""
CoChem Async Monomer Worker Pool & Cross-Platform File Locking.

Provenance:
- Method Matrix [M]: Cross-platform filelock synchronization, thread explosion prevention (REQ-TOPOS-050).
- Derived Relations [D]: N_workers * N_threads_per_worker <= N_cores, SHA256 composite monomer cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import filelock
import psutil
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import ConcurrencyLockTimeoutError

logger = logging.getLogger("cochem_topos_async")


class MonomerPreprocessResult(BaseModel):
    """Data contract representing pre-computed monomer properties and cache state."""

    composite_cache_key: str
    isomeric_smiles: str
    charge: int = 0
    multiplicity: int = 1
    dipole_moment_debye: float
    rotational_constants_ghz: Tuple[float, float, float]
    moments_of_inertia_amu_ang2: Tuple[float, float, float]
    cache_hit: bool = False


def compute_monomer_cache_key(
    isomeric_smiles: str,
    charge: int = 0,
    multiplicity: int = 1,
    method_hash: str = "r2scan-3c",
) -> str:
    """
    [REQ-TOPOS-050] Generate SHA-256 composite cache key.

    Formula:
        Key = SHA256(isomeric_smiles || charge || multiplicity || method_hash)
    """
    payload = f"{isomeric_smiles.strip()}|{charge}|{multiplicity}|{method_hash.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_worker_threads(
    num_monomers: int,
    logical_cores: Optional[int] = None,
) -> Tuple[int, int]:
    """
    [REQ-TOPOS-050] Partition workers and threads enforcing N_workers * N_threads <= N_cores.

    Formula:
        N_cores = logical_cores or psutil.cpu_count(logical=False) or 1
        N_workers = min(num_monomers, N_cores)
        N_threads_per_worker = max(1, floor(N_cores / N_workers))
    """
    cores = logical_cores if logical_cores is not None else (psutil.cpu_count(logical=False) or 1)
    cores = max(1, cores)
    n_monomers = max(1, num_monomers)

    n_workers = min(n_monomers, cores)
    n_threads = max(1, cores // n_workers)

    return n_workers, n_threads


def process_monomer_cached(
    isomeric_smiles: str,
    compute_func: Callable[[str, int, int], Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
    method_hash: str = "r2scan-3c",
    cache_dir: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> MonomerPreprocessResult:
    """
    [REQ-TOPOS-050] Compute or retrieve monomer properties with cross-platform file locking.

    Args:
        isomeric_smiles: Standard isomeric SMILES string.
        compute_func: Computation callback returning (dipole, rot_consts, moments_of_inertia).
        charge: Net monomer charge.
        multiplicity: Spin multiplicity (2S+1).
        method_hash: Computation method signature.
        cache_dir: Directory to persist cache JSON files (default: .cache/monomers).
        timeout_seconds: FileLock timeout in seconds.

    Returns:
        MonomerPreprocessResult instance.

    Raises:
        ConcurrencyLockTimeoutError: If filelock acquisition exceeds timeout_seconds.
    """
    cache_key = compute_monomer_cache_key(
        isomeric_smiles=isomeric_smiles,
        charge=charge,
        multiplicity=multiplicity,
        method_hash=method_hash,
    )

    base_cache_path = Path(cache_dir) if cache_dir else (Path.home() / ".cochem_cache" / "monomers")
    base_cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = base_cache_path / f"{cache_key}.json"
    lock_file = base_cache_path / f"{cache_key}.json.lock"

    lock = filelock.FileLock(str(lock_file), timeout=timeout_seconds)

    try:
        with lock:
            if cache_file.is_file():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return MonomerPreprocessResult(
                        composite_cache_key=cache_key,
                        isomeric_smiles=isomeric_smiles,
                        charge=charge,
                        multiplicity=multiplicity,
                        dipole_moment_debye=data["dipole_moment_debye"],
                        rotational_constants_ghz=tuple(data["rotational_constants_ghz"]),
                        moments_of_inertia_amu_ang2=tuple(data["moments_of_inertia_amu_ang2"]),
                        cache_hit=True,
                    )
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as read_err:
                    logger.warning("Corrupted monomer cache file %s, recomputing: %s", cache_file, read_err)

            # Compute if not cached
            dipole, rot_consts, moments = compute_func(isomeric_smiles, charge, multiplicity)

            record = {
                "composite_cache_key": cache_key,
                "isomeric_smiles": isomeric_smiles,
                "charge": charge,
                "multiplicity": multiplicity,
                "dipole_moment_debye": float(dipole),
                "rotational_constants_ghz": [float(x) for x in rot_consts],
                "moments_of_inertia_amu_ang2": [float(x) for x in moments],
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            return MonomerPreprocessResult(
                composite_cache_key=cache_key,
                isomeric_smiles=isomeric_smiles,
                charge=charge,
                multiplicity=multiplicity,
                dipole_moment_debye=float(dipole),
                rotational_constants_ghz=rot_consts,
                moments_of_inertia_amu_ang2=moments,
                cache_hit=False,
            )

    except filelock.Timeout as exc:
        raise ConcurrencyLockTimeoutError(
            lock_file_path=str(lock_file),
            timeout_seconds=timeout_seconds,
        ) from exc
