"""
CoChem ORCA Input Generator & Relativistic ECP Handler.

Provenance:
- Method Matrix [M]: Karlsruhe def2-ECP / Stuttgart-Dresden relativistic pseudo-potentials (REQ-CORE-041),
  composite functional bypass (r2SCAN-3c, B97-3c, HF-3c, PBEh-3c).
- Derived Relations [D]: HPC hardware blocks (%pal, %maxcore) and GPU keywords.
- Chunk 06 Mandates: Dynamic maxcore injection, RI/RIJCOSX auto-activation, FrozenCore allocation,
  point group symmetry flags, and DLPNO-CCSD(T) parameters (REQ-CORE-051..057, 059, 060).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from cochem.core.cochem_elements import get_atomic_number, is_heavy_element
from cochem.core.cochem_exceptions import ECPConfigurationError
from cochem.core.cochem_hardware import CudaHardwareConfigModel, HpcResourceAllocationModel
from cochem.core.models import MemoryAllocationConfig
from cochem.qc.cochem_method_registry import (
    FunctionalType,
    classify_functional,
    resolve_auxiliary_basis,
)

COMPOSITE_METHODS = {
    "r2scan-3c",
    "b97-3c",
    "hf-3c",
    "pbeh-3c",
    "wb97x-3c",
    "b97-3c-d4",
}


class ECPAssignment(BaseModel):
    """Data contract for element-specific ECP and GTO basis assignment."""

    element_symbol: str = Field(..., description="Chemical element symbol.")
    atomic_number: int = Field(..., ge=1, le=118, description="Atomic number Z.")
    assigned_ecp: str = Field(..., description="Assigned relativistic pseudo-potential (ECP).")
    basis_gto: str = Field(..., description="Assigned GTO basis set for valence.")
    is_composite_bypassed: bool = Field(default=False, description="Whether ECP assignment was bypassed for composite functional.")


class OrcaInputResult(BaseModel):
    """Data contract representing the complete synthesized ORCA input deck."""

    route_line: str
    ecp_assignments: List[ECPAssignment] = Field(default_factory=list)
    basis_block: Optional[str] = None
    pal_block: Optional[str] = None
    maxcore_line: Optional[str] = None
    full_input_text: str


def assign_ecp_for_element(symbol: str, basis: str, method: str) -> ECPAssignment:
    """
    [REQ-CORE-041] Determine relativistic ECP and valence basis for an element.
    """
    z = get_atomic_number(symbol)
    method_lower = method.strip().lower()

    if method_lower in COMPOSITE_METHODS:
        return ECPAssignment(
            element_symbol=symbol,
            atomic_number=z,
            assigned_ecp="none",
            basis_gto=basis,
            is_composite_bypassed=True,
        )

    # Elements Z <= 36 (H through Kr) do not strictly require ECP under standard def2
    if z <= 36:
        return ECPAssignment(
            element_symbol=symbol,
            atomic_number=z,
            assigned_ecp="none",
            basis_gto=basis,
            is_composite_bypassed=False,
        )

    basis_lower = basis.strip().lower()

    # 37 <= Z <= 56 (Rb-Ba) and 72 <= Z <= 86 (Hf-Rn)
    if (37 <= z <= 56) or (72 <= z <= 86):
        if "def2" in basis_lower:
            assigned_ecp = "def2-ECP"
        elif "cc-pv" in basis_lower or "aug-cc-pv" in basis_lower:
            assigned_ecp = "cc-pVnZ-PP"
        else:
            assigned_ecp = "def2-ECP"

    # 57 <= Z <= 71 (Lanthanides: La-Lu)
    elif 57 <= z <= 71:
        assigned_ecp = "ECP28MWB"

    # 89 <= Z <= 103 (Actinides: Ac-Lr)
    elif 89 <= z <= 103:
        assigned_ecp = "ECP60MWB"

    else:
        raise ECPConfigurationError(
            element_symbol=symbol,
            atomic_number=z,
            basis=basis,
            reason=f"Superheavy transactinide element Z={z} exceeds standard relativistic ECP tables.",
        )

    return ECPAssignment(
        element_symbol=symbol,
        atomic_number=z,
        assigned_ecp=assigned_ecp,
        basis_gto=basis,
        is_composite_bypassed=False,
    )


def build_orca_basis_block(ecp_assignments: Sequence[ECPAssignment]) -> Optional[str]:
    """
    [REQ-CORE-041] Construct the ORCA %basis ... end block for heavy elements requiring ECPs.
    """
    active_assignments = [
        a for a in ecp_assignments
        if not a.is_composite_bypassed and a.assigned_ecp != "none"
    ]
    if not active_assignments:
        return None

    # Deduplicate by element symbol
    seen_symbols = set()
    unique_assignments = []
    for a in active_assignments:
        if a.element_symbol not in seen_symbols:
            seen_symbols.add(a.element_symbol)
            unique_assignments.append(a)

    lines = ["%basis"]
    for a in unique_assignments:
        lines.append(f'  NewGTO {a.element_symbol} "{a.basis_gto}" end')
        lines.append(f'  NewECP {a.element_symbol} "{a.assigned_ecp}" end')
    lines.append("end")

    return "\n".join(lines)


def generate_orca_input(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    method: str = "B3LYP",
    basis: str = "def2-TZVP",
    job_type: str = "Opt Freq",
    charge: int = 0,
    multiplicity: int = 1,
    hpc_alloc: Optional[HpcResourceAllocationModel] = None,
    cuda_config: Optional[CudaHardwareConfigModel] = None,
) -> OrcaInputResult:
    """
    [REQ-CORE-041, REQ-HW-046, REQ-HW-047] Assemble a complete, production-ready ORCA input file.
    """
    method_lower = method.strip().lower()
    is_composite = method_lower in COMPOSITE_METHODS

    # Construct main route line
    route_parts = ["!"]
    route_parts.append(method)
    if not is_composite:
        route_parts.append(basis)
    if job_type:
        route_parts.append(job_type)

    if cuda_config and cuda_config.cuda_available:
        for flag in cuda_config.gpu_flags:
            if flag.startswith("!"):
                flag_clean = flag.lstrip("!").strip()
                if flag_clean not in route_parts:
                    route_parts.append(flag_clean)

    route_line = " ".join(route_parts)

    # Process ECP assignments
    ecp_assignments = [assign_ecp_for_element(s, basis, method) for s in symbols]
    basis_block = build_orca_basis_block(ecp_assignments) if not is_composite else None

    # Hardware parallelization & memory blocks
    pal_block = None
    maxcore_line = None
    if hpc_alloc is not None and hpc_alloc.allocated_procs > 1:
        pal_block = f"%pal nprocs {hpc_alloc.allocated_procs} end"

    if cuda_config is not None:
        maxcore_line = f"%maxcore {cuda_config.maxcore_cpu_mb}"
    elif hpc_alloc is not None:
        maxcore_line = f"%maxcore {hpc_alloc.memory_per_core_mb}"

    # Assemble full input deck
    deck_lines = [route_line]
    if maxcore_line:
        deck_lines.append(maxcore_line)
    if pal_block:
        deck_lines.append(pal_block)
    if basis_block:
        deck_lines.append(basis_block)

    # CUDA scf block if applicable
    if cuda_config and cuda_config.cuda_available:
        for flag in cuda_config.gpu_flags:
            if flag.startswith("%"):
                deck_lines.append(flag)

    # Coordinate block
    deck_lines.append(f"* xyz {charge} {multiplicity}")
    for sym, (x, y, z) in zip(symbols, coordinates_angstrom):
        deck_lines.append(f"  {sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}")
    deck_lines.append("*")

    full_input_text = "\n".join(deck_lines) + "\n"

    return OrcaInputResult(
        route_line=route_line,
        ecp_assignments=ecp_assignments,
        basis_block=basis_block,
        pal_block=pal_block,
        maxcore_line=maxcore_line,
        full_input_text=full_input_text,
    )


def generate_orca_chunk06_deck(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    method: str = "B3LYP",
    basis: str = "def2-TZVP",
    job_type: str = "Opt",
    charge: int = 0,
    multiplicity: int = 1,
    mem_config: Optional[MemoryAllocationConfig] = None,
    point_group: str = "C1",
    promote_dlpno: bool = False,
    is_opt: bool = True,
) -> str:
    """
    [REQ-CORE-051..057, 059] High-performance ORCA deck synthesizer with Method Matrix v4 invariants.

    Invariants:
    1. Memory & Parallel: %maxcore <MB> and %pal nprocs <cores> end.
    2. RI / RIJCOSX: Pure/Meta-GGA -> ! RI def2/J; Hybrid -> ! RIJCOSX def2/J, %scf GridX 5 end, ! NoFinalGridX for Opt.
    3. Post-HF / DLPNO: ! FrozenCore; DLPNO promotion -> ! DLPNO-CCSD(T) TightPNO, %mdci.
    4. Symmetry: PointGroup != C1 -> ! UseSym and %sym Tolerance 1e-4 end.
    """
    family = classify_functional(method)
    ri_info = resolve_auxiliary_basis(primary_basis=basis, functional=method, is_freq=(not is_opt))

    route_tokens = ["!"]

    # 1. Method and Basis
    if promote_dlpno and family == FunctionalType.POST_HF:
        route_tokens.extend(["DLPNO-CCSD(T)", "TightPNO", basis])
    else:
        route_tokens.extend([method, basis])

    # 2. RI / RIJCOSX flags
    if family in (FunctionalType.PURE_GGA, FunctionalType.META_GGA):
        route_tokens.extend(["RI", ri_info.aux_coulomb_basis])
    elif family == FunctionalType.HYBRID:
        route_tokens.extend(["RIJCOSX", ri_info.aux_coulomb_basis])
        if is_opt:
            route_tokens.append("NoFinalGridX")
    elif family == FunctionalType.DOUBLE_HYBRID:
        route_tokens.extend(["RIJCOSX", ri_info.aux_coulomb_basis])
        if ri_info.aux_correlation_basis:
            route_tokens.append(ri_info.aux_correlation_basis)
        if is_opt:
            route_tokens.append("NoFinalGridX")
    elif family == FunctionalType.POST_HF:
        if "cv" not in basis.lower():
            route_tokens.append("FrozenCore")
        else:
            route_tokens.append("NoFrozenCore")

    # 3. Symmetry flag
    if point_group and point_group != "C1":
        route_tokens.append("UseSym")

    if job_type:
        route_tokens.append(job_type)

    deck_lines = [" ".join(route_tokens)]

    # 4. %maxcore and %pal blocks
    if mem_config is not None:
        usable_mb = float(mem_config.effective_limit_mb) * (1.0 - mem_config.headroom_fraction)
        cores = max(1, mem_config.requested_cores)
        maxcore_mb = max(mem_config.min_core_mb, int(usable_mb // cores))
        deck_lines.append(f"%maxcore {maxcore_mb}")
        deck_lines.append(f"%pal nprocs {cores} end")

    # 5. %sym block
    if point_group and point_group != "C1":
        deck_lines.append("%sym Tolerance 1e-4 end")

    # 6. %mdci DLPNO parameter block
    if promote_dlpno and family == FunctionalType.POST_HF:
        deck_lines.append("%mdci TCutPairs 1e-4 TCutPNO 3.33e-7 TCutMKN 1e-3 end")

    # 7. Coordinates block
    deck_lines.append(f"* xyz {charge} {multiplicity}")
    for sym, (x, y, z) in zip(symbols, coordinates_angstrom):
        deck_lines.append(f"  {sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}")
    deck_lines.append("*")

    return "\n".join(deck_lines) + "\n"
