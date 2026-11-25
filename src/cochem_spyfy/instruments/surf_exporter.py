"""Deterministic native SURF TOML and strict JSON manifest exporter.

For each human-approved scan region, ``SurfExporter`` emits three verified,
UTF-8 LF artifacts grounded byte-for-byte in ``GruLabRotSpec/SURF@0486b877``:
1. ``<region_stem>_control_config.toml`` — validated against ``SurfConfig`` with
   ``valon_controller.freq == start_freq - awg_freq``, strict ``int``
   ``rf_level`` and ``awg_freq``, and explicit soft warning if ``rf_level > 10``.
2. ``<region_stem>_scan_settings.toml`` — validated against
   ``SurfFrequencyScanSettings``, omitting ``zaber_pos`` when ``None``.
3. ``<region_stem>_surf_manifest.json`` — strict RFC-8259 JSON (``allow_nan=False``)
   recording ``fit_state_id``, ``spcat_catalog_sha256``, GP hyperparameters,
   recommendation metadata, exact user override audit diffs, estimated runtime
   ``T_min``, surfaced warnings, and exact SHA-256 digests of both TOMLs.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..core_engine.cochem_spyfy_state import FitState
from .profiles import (
    DEFAULT_INSTRUMENT_PROFILE,
    get_instrument_profile,
)
from .surf_planner import (
    SurfParameterOverrideTable,
    SurfScanProposal,
    build_default_surf_config,
    sanitize_gp_hyperparameters,
    sync_surf_control_config,
)
from .surf_schemas import (
    OutputSettings,
    SurfConfig,
    SurfFrequencyScanSettings,
    SurfOverrideRecord,
    SurfScanManifest,
)


def _utc_now_iso() -> str:
    """Return current UTC timestamp in RFC-3339 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_utf8_lf(target: Path, text: str) -> str:
    """Atomically write strict UTF-8 LF text (no BOM/CR) and return its SHA-256 hex digest."""
    if "\r" in text:
        raise ValueError(f"Refusing to write CR line endings to {target.name}.")
    if text.startswith("\ufeff"):
        raise ValueError(f"Refusing to write UTF-8 BOM to {target.name}.")

    raw_bytes = text.encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return digest


def _validate_region_stem(region_stem: str) -> str:
    """Validate that ``region_stem`` is a safe single filename stem with no traversal."""
    checked = OutputSettings(filename=region_stem, location=".")
    return checked.filename


@dataclass(frozen=True)
class SurfExportResult:
    """Immutable export bundle describing all three deterministic SURF artifacts."""

    region_stem: str
    output_dir: Path
    control_config_path: Path
    scan_settings_path: Path
    manifest_path: Path
    control_config_sha256: str
    scan_settings_sha256: str
    manifest_sha256: str
    control_config: SurfConfig
    scan_settings: SurfFrequencyScanSettings
    manifest: SurfScanManifest
    warnings: Tuple[str, ...]


class SurfExporter:
    """Deterministic exporter producing native SURF TOML files and provenance manifests."""

    def __init__(self, *, instrument_profile: str = DEFAULT_INSTRUMENT_PROFILE) -> None:
        self._profile = get_instrument_profile(instrument_profile)

    @property
    def instrument_profile(self) -> str:
        """Return active instrument profile identifier."""
        return self._profile.profile_id

    def export_region(
        self,
        output_dir: Path | str,
        *,
        override_table: Optional[SurfParameterOverrideTable] = None,
        scan_settings: Optional[SurfFrequencyScanSettings] = None,
        control_config: Optional[SurfConfig] = None,
        overrides: Optional[Sequence[SurfOverrideRecord]] = None,
        region_stem: Optional[str] = None,
        fit_state_id: Optional[str] = None,
        fit_state: Optional[FitState] = None,
        spcat_catalog_path: Optional[Path | str] = None,
        spcat_catalog_sha256: Optional[str] = None,
        gp_hyperparameters: Optional[Mapping[str, Any]] = None,
        recommendation: Optional[Any] = None,
        recommendation_rank: Optional[int] = None,
        recommendation_score: Optional[float] = None,
        target_constants: Optional[Sequence[str]] = None,
        extra_warnings: Optional[Sequence[str]] = None,
        sync_control_config: bool = True,
        created_at_utc: Optional[str] = None,
    ) -> SurfExportResult:
        """Validate and write ``control_config.toml``, ``scan_settings.toml``, and ``surf_manifest.json``."""
        source_proposal: Optional[SurfScanProposal] = None
        if override_table is not None:
            resolved_scan = override_table.current_scan_settings
            resolved_cfg = override_table.current_control_config
            resolved_overrides: Tuple[SurfOverrideRecord, ...] = tuple(override_table.get_override_records())
            source_proposal = override_table.proposal
        else:
            if scan_settings is None:
                raise ValueError("Either override_table or scan_settings must be provided.")
            resolved_scan = SurfFrequencyScanSettings.model_validate(scan_settings.model_dump())
            resolved_cfg = (
                SurfConfig.model_validate(control_config.model_dump())
                if control_config is not None
                else build_default_surf_config(
                    start_freq_mhz=resolved_scan.scan_parameters.start_freq,
                    zaber_scanning_speed=resolved_scan.scan_parameters.zaber_speed,
                    math4_resolution=float(resolved_scan.digitizer_settings.resolution),
                    math4_window=resolved_scan.digitizer_settings.apodization,
                    math4_gate_position=float(resolved_scan.timing_settings.acq_delay),
                    trigger_rate=float(resolved_scan.timing_settings.rep_rate),
                )
            )
            resolved_overrides = tuple(overrides) if overrides is not None else ()

        if overrides is not None and override_table is not None:
            resolved_overrides = tuple(overrides)

        # Synchronize or strictly verify hardware control config against scan settings
        expected_lo = round(
            float(resolved_scan.scan_parameters.start_freq)
            - float(resolved_cfg.awg_controller.awg_freq),
            6,
        )
        if sync_control_config:
            final_cfg = sync_surf_control_config(resolved_cfg, resolved_scan)
        else:
            resolved_cfg.validate_lo_for_scan_freq(resolved_scan.scan_parameters.start_freq)
            if not math.isclose(resolved_cfg.valon_controller.freq, expected_lo, abs_tol=1e-9):
                raise ValueError(
                    f"valon_controller.freq ({resolved_cfg.valon_controller.freq}) must equal "
                    f"start_freq - awg_freq ({expected_lo})."
                )
            final_cfg = SurfConfig.model_validate(resolved_cfg.model_dump())

        final_scan = SurfFrequencyScanSettings.model_validate(resolved_scan.model_dump())

        # Resolve stem and output filenames
        resolved_stem = _validate_region_stem(
            region_stem if region_stem is not None else final_scan.output_settings.filename
        )
        control_filename = f"{resolved_stem}_control_config.toml"
        scan_filename = f"{resolved_stem}_scan_settings.toml"
        manifest_filename = f"{resolved_stem}_surf_manifest.json"

        # Serialize deterministic TOMLs and verify immediately via stdlib tomllib
        control_toml = final_cfg.to_surf_toml()
        scan_toml = final_scan.to_surf_toml()

        parsed_cfg = tomllib.loads(control_toml)
        if type(parsed_cfg["valon_controller"]["rf_level"]) is not int:
            raise ValueError("Serialized valon_controller.rf_level is not a strict TOML integer.")
        if type(parsed_cfg["awg_controller"]["awg_freq"]) is not int:
            raise ValueError("Serialized awg_controller.awg_freq is not a strict TOML integer.")
        parsed_lo = float(parsed_cfg["valon_controller"]["freq"])
        parsed_expected_lo = round(
            float(final_scan.scan_parameters.start_freq)
            - float(parsed_cfg["awg_controller"]["awg_freq"]),
            6,
        )
        if not math.isclose(parsed_lo, parsed_expected_lo, abs_tol=1e-9):
            raise ValueError(
                f"Serialized valon_controller.freq ({parsed_lo}) does not match "
                f"start_freq - awg_freq ({parsed_expected_lo})."
            )

        parsed_scan = tomllib.loads(scan_toml)
        if final_scan.scan_parameters.zaber_pos is None and "zaber_pos" in parsed_scan["scan_parameters"]:
            raise ValueError("Serialized scan_settings.toml must omit zaber_pos when None.")

        # Resolve provenance metadata
        resolved_fit_state_id = fit_state_id
        if resolved_fit_state_id is None and fit_state is not None:
            resolved_fit_state_id = fit_state.state_id

        resolved_catalog_sha = spcat_catalog_sha256
        if resolved_catalog_sha is None and spcat_catalog_path is not None:
            catalog_bytes = Path(spcat_catalog_path).read_bytes()
            resolved_catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()

        rec_source = recommendation if recommendation is not None else source_proposal
        resolved_rank = recommendation_rank
        resolved_score = recommendation_score
        resolved_targets: Tuple[str, ...] = tuple(target_constants) if target_constants is not None else ()

        if rec_source is not None:
            if resolved_rank is None and hasattr(rec_source, "rank"):
                resolved_rank = int(getattr(rec_source, "rank"))
            if resolved_score is None:
                if hasattr(rec_source, "information_gain_per_minute"):
                    resolved_score = float(getattr(rec_source, "information_gain_per_minute"))
                elif hasattr(rec_source, "chunk_score"):
                    resolved_score = float(getattr(rec_source, "chunk_score"))
            if not resolved_targets and hasattr(rec_source, "targeted_constants"):
                resolved_targets = tuple(str(c) for c in getattr(rec_source, "targeted_constants"))

        gp_dict: Dict[str, Any] = {}
        if source_proposal is not None and source_proposal.gp_hyperparameters:
            gp_dict.update(sanitize_gp_hyperparameters(source_proposal.gp_hyperparameters))
        if gp_hyperparameters:
            gp_dict.update(sanitize_gp_hyperparameters(gp_hyperparameters))

        # Collect soft warnings (e.g. rf_level > 10 dBm)
        warnings_list: List[str] = list(final_cfg.rf_level_warnings())
        if extra_warnings:
            for warn in extra_warnings:
                warn_str = str(warn).strip()
                if warn_str and warn_str not in warnings_list:
                    warnings_list.append(warn_str)

        # Write both TOML files atomically and record exact byte SHA-256 digests
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        control_path = out_root / control_filename
        scan_path = out_root / scan_filename
        manifest_path = out_root / manifest_filename

        control_sha256 = _atomic_write_utf8_lf(control_path, control_toml)
        scan_sha256 = _atomic_write_utf8_lf(scan_path, scan_toml)

        estimated_minutes = final_scan.estimated_runtime_minutes(
            retune_overhead_s=self._profile.zaber_retune_overhead_s
        )

        manifest = SurfScanManifest(
            instrument_profile=self._profile.profile_id,
            created_at_utc=created_at_utc if created_at_utc is not None else _utc_now_iso(),
            fit_state_id=resolved_fit_state_id,
            spcat_catalog_sha256=resolved_catalog_sha,
            control_config_filename=control_filename,
            control_config_sha256=control_sha256,
            scan_settings_filename=scan_filename,
            scan_settings_sha256=scan_sha256,
            estimated_runtime_minutes=estimated_minutes,
            recommendation_rank=resolved_rank,
            recommendation_score=resolved_score,
            target_constants=resolved_targets,
            gp_hyperparameters=gp_dict,
            overrides=resolved_overrides,
            warnings=tuple(warnings_list),
        )
        manifest_json = manifest.to_strict_json(indent=2)
        # Verify strict RFC-8259 JSON round-trip before writing
        SurfScanManifest.from_strict_json(manifest_json)
        manifest_sha256 = _atomic_write_utf8_lf(manifest_path, manifest_json)

        return SurfExportResult(
            region_stem=resolved_stem,
            output_dir=out_root,
            control_config_path=control_path,
            scan_settings_path=scan_path,
            manifest_path=manifest_path,
            control_config_sha256=control_sha256,
            scan_settings_sha256=scan_sha256,
            manifest_sha256=manifest_sha256,
            control_config=final_cfg,
            scan_settings=final_scan,
            manifest=manifest,
            warnings=tuple(warnings_list),
        )

    def export_from_override_table(
        self,
        override_table: SurfParameterOverrideTable,
        output_dir: Path | str,
        *,
        region_stem: Optional[str] = None,
        fit_state_id: Optional[str] = None,
        fit_state: Optional[FitState] = None,
        spcat_catalog_path: Optional[Path | str] = None,
        spcat_catalog_sha256: Optional[str] = None,
        gp_hyperparameters: Optional[Mapping[str, Any]] = None,
        recommendation: Optional[Any] = None,
    ) -> SurfExportResult:
        """Convenience wrapper exporting directly from a HITL ``SurfParameterOverrideTable``."""
        return self.export_region(
            output_dir=output_dir,
            override_table=override_table,
            region_stem=region_stem,
            fit_state_id=fit_state_id,
            fit_state=fit_state,
            spcat_catalog_path=spcat_catalog_path,
            spcat_catalog_sha256=spcat_catalog_sha256,
            gp_hyperparameters=gp_hyperparameters,
            recommendation=recommendation,
        )

    def export_batch(
        self,
        override_tables: Sequence[SurfParameterOverrideTable],
        output_dir: Path | str,
        *,
        fit_state_id: Optional[str] = None,
        fit_state: Optional[FitState] = None,
        spcat_catalog_path: Optional[Path | str] = None,
        spcat_catalog_sha256: Optional[str] = None,
        gp_hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> List[SurfExportResult]:
        """Export a sequence of approved HITL override tables to deterministic artifact sets."""
        results: List[SurfExportResult] = []
        used_stems: set[str] = set()
        for idx, table in enumerate(override_tables, start=1):
            base_stem = table.current_scan_settings.output_settings.filename
            stem = base_stem if base_stem not in used_stems else f"{base_stem}_r{idx}"
            used_stems.add(stem)
            results.append(
                self.export_region(
                    output_dir=output_dir,
                    override_table=table,
                    region_stem=stem,
                    fit_state_id=fit_state_id,
                    fit_state=fit_state,
                    spcat_catalog_path=spcat_catalog_path,
                    spcat_catalog_sha256=spcat_catalog_sha256,
                    gp_hyperparameters=gp_hyperparameters,
                )
            )
        return results


__all__ = [
    "SurfExportResult",
    "SurfExporter",
]
