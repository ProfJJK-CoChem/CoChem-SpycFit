"""
CoChem Hardware Discovery, CUDA Inspection & HPC Allocation Engine.

Provenance:
- Derived Relations [D]: MaxCore CPU allocation with 30% pinned host buffer, Workstation 2.5 GB/core ratio.
- Empirical [E]: SLURM/PBS/MPI environment parsing and CUDA sm_60 (Pascal+) capability thresholding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import psutil
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import HardwareAllocationFloorError


class CudaHardwareConfigModel(BaseModel):
    """Data contract representing discovered CUDA GPU hardware topology and ORCA flags."""

    cuda_available: bool = Field(..., description="Whether compatible CUDA GPU is available.")
    device_count: int = Field(default=0, ge=0, description="Total number of compatible GPU devices.")
    compute_capability: Tuple[int, int] = Field(default=(0, 0), description="Major, minor compute capability.")
    total_vram_mb: int = Field(default=0, ge=0, description="Total available GPU VRAM in megabytes.")
    maxcore_cpu_mb: int = Field(default=512, ge=512, description="Calculated MaxCore CPU memory in MB.")
    gpu_flags: List[str] = Field(default_factory=list, description="ORCA GPU execution keywords.")


class HpcResourceAllocationModel(BaseModel):
    """Data contract representing detected HPC scheduler or local workstation resources."""

    scheduler_type: Literal["slurm", "pbs", "openmpi", "workstation"] = Field(
        ..., description="Detected cluster scheduler or workstation environment."
    )
    allocated_procs: int = Field(..., gt=0, description="Allocated CPU processes / cores.")
    memory_per_core_mb: int = Field(..., gt=0, description="Allocated memory per core in MB.")
    is_oversubscribed: bool = Field(default=False, description="Whether core allocation exceeds physical cores.")


def discover_cuda_hardware(mpi_procs: int = 1) -> CudaHardwareConfigModel:
    """
    [REQ-HW-046] Discover NVIDIA CUDA hardware and compute capability sm_60+ thresholding.

    Calculates MaxCore CPU memory with 30% pinned buffer:
        MaxCore_CPU = max(512, floor((RAM_free_MB * 0.70) / N_MPI))

    Args:
        mpi_procs: Number of MPI processes sharing host memory (default: 1).

    Returns:
        CudaHardwareConfigModel with capability and ORCA GPU keywords.
    """
    mem = psutil.virtual_memory()
    ram_free_mb = int(mem.available / (1024 * 1024))
    n_mpi = max(1, mpi_procs)
    maxcore_cpu_mb = max(512, int((ram_free_mb * 0.70) // n_mpi))

    cuda_available = False
    device_count = 0
    compute_cap: Tuple[int, int] = (0, 0)
    total_vram_mb = 0
    gpu_flags: List[str] = []

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                major, minor = torch.cuda.get_device_capability(0)
                compute_cap = (major, minor)
                # sm_60 (Pascal) or newer
                if major >= 6:
                    cuda_available = True
                    props = torch.cuda.get_device_properties(0)
                    total_vram_mb = int(props.total_memory / (1024 * 1024))
                    gpu_flags = ["! RIJCOSX RI-J-GPU", "%scf Cuda true end"]
    except Exception:
        cuda_available = False

    return CudaHardwareConfigModel(
        cuda_available=cuda_available,
        device_count=device_count,
        compute_capability=compute_cap,
        total_vram_mb=total_vram_mb,
        maxcore_cpu_mb=maxcore_cpu_mb,
        gpu_flags=gpu_flags,
    )


def allocate_hpc_resources(
    env_override: Optional[Dict[str, str]] = None,
    override_free_ram_gb: Optional[float] = None,
    override_phys_cores: Optional[int] = None,
) -> HpcResourceAllocationModel:
    """
    [REQ-HW-047] Inspect SLURM, PBS, OpenMPI or workstation host resources and calculate core counts.

    Workstation Allocation:
        N_procs = max(1, min(N_phys, floor(RAM_free_GB / 2.5)))

    Args:
        env_override: Optional environment variable dictionary for testing.
        override_free_ram_gb: Optional free RAM override in GB.
        override_phys_cores: Optional physical cores count override.

    Returns:
        HpcResourceAllocationModel instance.

    Raises:
        HardwareAllocationFloorError: If calculated cores <= 0.
    """
    env = os.environ if env_override is None else env_override

    # Physical host discovery
    phys_cores = override_phys_cores if override_phys_cores is not None else (psutil.cpu_count(logical=False) or 1)
    if override_free_ram_gb is not None:
        free_ram_gb = override_free_ram_gb
        free_ram_mb = int(free_ram_gb * 1024)
    else:
        vmem = psutil.virtual_memory()
        free_ram_gb = vmem.available / (1024.0 ** 3)
        free_ram_mb = int(vmem.available / (1024 * 1024))

    scheduler_type: Literal["slurm", "pbs", "openmpi", "workstation"] = "workstation"
    allocated_procs = 0

    if "SLURM_NTASKS" in env:
        scheduler_type = "slurm"
        try:
            n_tasks = int(env["SLURM_NTASKS"])
            cpus_per_task = int(env.get("SLURM_CPUS_PER_TASK", "1"))
            allocated_procs = n_tasks * cpus_per_task
        except (ValueError, TypeError):
            allocated_procs = 0

    elif "PBS_NODEFILE" in env:
        scheduler_type = "pbs"
        nodefile_path = env["PBS_NODEFILE"]
        if Path(nodefile_path).is_file():
            try:
                with open(nodefile_path, "r", encoding="utf-8") as f:
                    allocated_procs = len([line.strip() for line in f if line.strip()])
            except Exception:
                allocated_procs = 0
        else:
            allocated_procs = 0

    elif "OMPI_COMM_WORLD_SIZE" in env:
        scheduler_type = "openmpi"
        try:
            allocated_procs = int(env["OMPI_COMM_WORLD_SIZE"])
        except (ValueError, TypeError):
            allocated_procs = 0

    else:
        scheduler_type = "workstation"
        # 2.5 GB / core floor guard
        ram_constrained_cores = int(free_ram_gb // 2.5)
        allocated_procs = max(1, min(phys_cores, ram_constrained_cores))

    if allocated_procs <= 0:
        raise HardwareAllocationFloorError(
            calculated_cores=allocated_procs,
            scheduler_type=scheduler_type,
            raw_metrics={
                "phys_cores": phys_cores,
                "free_ram_gb": round(free_ram_gb, 2),
                "env_keys": [k for k in ["SLURM_NTASKS", "PBS_NODEFILE", "OMPI_COMM_WORLD_SIZE"] if k in env],
            },
        )

    memory_per_core_mb = max(256, int(free_ram_mb // allocated_procs))
    is_oversubscribed = allocated_procs > phys_cores

    return HpcResourceAllocationModel(
        scheduler_type=scheduler_type,
        allocated_procs=allocated_procs,
        memory_per_core_mb=memory_per_core_mb,
        is_oversubscribed=is_oversubscribed,
    )
