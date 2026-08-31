"""
CoChem Hardware Discovery, CUDA Inspection, NVMe Profiling & Dynamic Memory Allocation Engine.

Provenance:
- Derived Relations [D]: Dynamic cgroup/SLURM memory limits, 20% safety headroom, dynamic core throttling (REQ-CORE-051).
- Empirical [E]: SLURM/PBS/MPI environment parsing and CUDA sm_60 (Pascal+) capability thresholding.
- I/O Benchmarking: Micro-benchmark of local NVMe scratch write/read throughput (REQ-QC-054).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Literal, Optional, Tuple

import psutil
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import HardwareAllocationFloorError
from cochem.core.models import (
    InsufficientMemoryAllocationError,
    MemoryAllocationConfig,
)


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


def get_effective_system_memory_mb(
    cgroup_path: Optional[Path] = None,
    env_override: Optional[Dict[str, str]] = None,
) -> int:
    """
    [REQ-CORE-051] Inspect physical RAM, Linux cgroups v1/v2, and SLURM allocations.

    Formula:
        M_cgroup = min(M_host_avail, M_slurm_alloc, M_cgroup_limit)

    Args:
        cgroup_path: Optional explicit path to cgroup memory file for testing.
        env_override: Optional environment variable mapping.

    Returns:
        Effective available system memory in megabytes (integer > 0).
    """
    env = os.environ if env_override is None else env_override

    # 1. Physical host available RAM
    vmem = psutil.virtual_memory()
    host_avail_mb = int(vmem.available / (1024 * 1024))
    candidates = [host_avail_mb]

    # 2. Linux cgroups v2 & v1 limits
    cgroup_limit_mb: Optional[int] = None
    if cgroup_path is not None and cgroup_path.is_file():
        try:
            content = cgroup_path.read_text(encoding="utf-8").strip()
            if content != "max" and content.isdigit():
                cgroup_limit_mb = int(int(content) / (1024 * 1024))
        except Exception:
            cgroup_limit_mb = None
    else:
        # Check standard Linux cgroup paths
        for path_str in [
            "/sys/fs/cgroup/memory.max",  # cgroups v2
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroups v1
        ]:
            p = Path(path_str)
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8").strip()
                    if content != "max" and content.isdigit():
                        val_mb = int(int(content) / (1024 * 1024))
                        # Ignore unconstrained default (~9e18 bytes)
                        if val_mb < host_avail_mb * 10:
                            cgroup_limit_mb = val_mb
                            break
                except Exception:
                    continue

    if cgroup_limit_mb is not None and cgroup_limit_mb > 0:
        candidates.append(cgroup_limit_mb)

    # 3. SLURM memory allocation variables
    if "SLURM_MEM_PER_NODE" in env:
        try:
            slurm_mb = int(env["SLURM_MEM_PER_NODE"])
            if slurm_mb > 0:
                candidates.append(slurm_mb)
        except (ValueError, TypeError):
            pass
    elif "SLURM_MEM_PER_CPU" in env:
        try:
            mem_per_cpu = int(env["SLURM_MEM_PER_CPU"])
            cpus = int(env.get("SLURM_CPUS_PER_TASK", "1"))
            total_slurm_mb = mem_per_cpu * cpus
            if total_slurm_mb > 0:
                candidates.append(total_slurm_mb)
        except (ValueError, TypeError):
            pass

    return max(256, min(candidates))


def calculate_memory_allocation(
    total_system_ram_mb: Optional[int] = None,
    cgroup_limit_mb: Optional[int] = None,
    requested_cores: int = 1,
    headroom_fraction: float = 0.20,
    min_core_mb: int = 512,
) -> Tuple[int, int]:
    """
    [REQ-CORE-051] Calculate usable memory, apply safety headroom, and throttle core counts if necessary.

    Mathematical Formulation:
        M_cgroup = min(total_system_ram_mb, cgroup_limit_mb)
        M_usable = M_cgroup * (1 - alpha)
        If (M_usable / N_proc_req) < min_core_mb:
            N_proc_adj = min(N_proc_req, int(M_usable // min_core_mb))
            maxcore_per_proc_mb = int(M_usable // N_proc_adj)
        Else:
            N_proc_adj = N_proc_req
            maxcore_per_proc_mb = int(M_usable // N_proc_req)

    Args:
        total_system_ram_mb: Total physical RAM in MB (defaults to effective system memory).
        cgroup_limit_mb: Optional cgroup limit in MB.
        requested_cores: Number of cores requested (integer >= 1).
        headroom_fraction: Safety fraction alpha in [0.05, 0.50] (default 0.20).
        min_core_mb: Floor memory per process in MB (default 512 MB).

    Returns:
        Tuple of (n_proc_adj, maxcore_per_proc_mb).

    Raises:
        InsufficientMemoryAllocationError: If usable memory cannot satisfy even 1 core at min_core_mb.
    """
    if total_system_ram_mb is None:
        total_ram = get_effective_system_memory_mb()
    else:
        total_ram = total_system_ram_mb

    if cgroup_limit_mb is not None:
        m_cgroup = min(total_ram, cgroup_limit_mb)
    else:
        m_cgroup = total_ram

    m_usable = float(m_cgroup) * (1.0 - headroom_fraction)

    if m_usable < float(min_core_mb):
        raise InsufficientMemoryAllocationError(
            requested_cores=requested_cores,
            available_mb=int(m_usable),
            min_core_mb=min_core_mb,
        )

    n_req = max(1, requested_cores)
    mem_per_req = m_usable / n_req

    if mem_per_req < float(min_core_mb):
        # Throttle process count
        n_proc_adj = max(1, min(n_req, int(m_usable // min_core_mb)))
    else:
        n_proc_adj = n_req

    maxcore_per_proc = int(m_usable // n_proc_adj)

    return n_proc_adj, maxcore_per_proc


def benchmark_scratch_io(scratch_dir: Path) -> Tuple[float, float]:
    """
    [REQ-QC-054] Micro-benchmark sequential write/read throughput (MB/s) and free capacity (GB) on scratch disk.

    Args:
        scratch_dir: Path to directory being benchmarked.

    Returns:
        Tuple of (throughput_mb_s, capacity_gb).
    """
    target_dir = Path(scratch_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check free disk space
    try:
        usage = psutil.disk_usage(str(target_dir))
        capacity_gb = float(usage.free) / (1024.0 ** 3)
    except Exception:
        capacity_gb = 10.0

    # Write & read 16 MB chunk to measure sequential throughput
    test_file = target_dir / f"_cochem_io_bench_{int(time.time() * 1000)}.tmp"
    chunk_size = 16 * 1024 * 1024  # 16 MB
    payload = b"\x00" * chunk_size

    try:
        t0 = time.perf_counter()
        with open(test_file, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        t1 = time.perf_counter()

        with open(test_file, "rb") as f:
            _ = f.read()
        t2 = time.perf_counter()

        write_time = max(1e-5, t1 - t0)
        read_time = max(1e-5, t2 - t1)
        avg_time = 0.5 * (write_time + read_time)
        throughput_mb_s = round(16.0 / avg_time, 2)
    except Exception:
        throughput_mb_s = 2500.0
    finally:
        if test_file.exists():
            try:
                test_file.unlink()
            except OSError:
                pass

    # Save to profile cache
    cache_dir = Path(os.environ.get("COCHEM_STATE_DIR", Path.home() / ".cochem"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "hardware_profile.json"
        profile_data = {
            "timestamp": time.time(),
            "scratch_dir": str(target_dir.resolve()),
            "throughput_mb_s": throughput_mb_s,
            "capacity_gb": capacity_gb,
        }
        cache_file.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
    except (OSError, IOError):
        pass

    return throughput_mb_s, capacity_gb


def discover_cuda_hardware(mpi_procs: int = 1) -> CudaHardwareConfigModel:
    """
    [REQ-HW-046] Discover NVIDIA CUDA hardware and compute capability sm_60+ thresholding.
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
    """
    env = os.environ if env_override is None else env_override

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
