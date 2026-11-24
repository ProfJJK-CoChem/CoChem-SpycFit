"""Windows-safe live directory watcher and SHA-256 provenance ledger for SURF runs.

Implements ``SurfRunWatcher`` for Pillar 3 (`SPYC-SURF-03`) of the
CoChem-SpycFit Primary SURF Integration Plan:

- Discovers ``<filename>_<k>/`` run directories under ``<output.location>``
  containing ``scan_settings.toml`` and ``CavityFiles/`` (or watches a single
  run directory directly).
- Applies a Windows-safe stability gate (non-zero size, stable ``(size, mtime_ns)``
  across consecutive polls, exact 4-column SURF header, finite FP64 values, and
  graceful retry on transient ``PermissionError``).
- Appends strict RFC-8259 JSON Lines (``allow_nan=False``) SHA-256 provenance
  records to ``surf_ingest_provenance.jsonl``.
- Yields incremental stitched spectrum chunks, cavity-track health telemetry,
  and strictly ``AssignmentStatus.PROPOSED`` Doppler doublet peak candidates as
  each immutable ``CavityFiles/<total_frequency>.csv`` step lands.
- Performs end-of-run reconciliation against finalized compacted
  ``<filename>.csv`` when ready.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem_spycfit.instruments.surf_ingest import (
    StepDirection,
    SurfBoundsError,
    SurfCandidateTransition,
    SurfCsvSchemaError,
    SurfIngestError,
    SurfOutputParser,
    SurfPeakPickResult,
    SurfReconciliationReport,
    SurfStepData,
    compute_file_sha256,
    pick_surf_doppler_peaks,
)
from cochem_spycfit.instruments.surf_schemas import SurfConfig, SurfFrequencyScanSettings

logger = logging.getLogger("CoChem-SpycFit.SurfWatcher")

_RUN_DIR_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)$")


# =============================================================================
# 1. WATCHER TELEMETRY & LIVE UPDATE MODELS
# =============================================================================


class SurfCavityTrackHealth(BaseModel):
    """Live cavity-tracking health telemetry accumulated across ingested steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["idle", "healthy", "rail_dropout"]
    step_count: int = Field(ge=0)
    last_center_freq_mhz: Optional[float] = None
    last_cavity_position_mm: Optional[float] = None
    last_step_dropout: bool = False
    dropout_count: int = Field(ge=0)
    center_freqs_mhz: Tuple[float, ...] = ()
    cavity_positions_mm: Tuple[float, ...] = ()
    dropout_flags: Tuple[bool, ...] = ()
    dropout_freqs_mhz: Tuple[float, ...] = ()


class SurfLiveStepUpdate(BaseModel):
    """Incremental payload yielded when a new ``CavityFiles/<freq>.csv`` step lands."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    run_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    step_data: SurfStepData
    cumulative_freq_mhz: np.ndarray
    cumulative_intensity: np.ndarray
    cavity_health: SurfCavityTrackHealth
    proposed_candidates: Tuple[SurfCandidateTransition, ...] = ()
    provenance_record: Dict[str, Any]
    reconciliation: Optional[SurfReconciliationReport] = None


class SurfTransientIssue(BaseModel):
    """Diagnostic record for a retryable file access or incomplete-write event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: Path
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    recorded_at_utc: str = Field(min_length=1)


# =============================================================================
# 2. INTERNAL STABILITY & RUN STATE TRACKING
# =============================================================================


@dataclass
class _FileStabilityState:
    """Tracks consecutive ``(size, mtime_ns)`` observations for one candidate file."""

    size_bytes: int
    mtime_ns: int
    first_stable_monotonic: float
    consecutive_matches: int = 1
    permission_retries: int = 0
    schema_retries: int = 0


@dataclass
class _WatchedRunState:
    """Mutable ingestion ledger state for one discovered ``<filename>_<k>/`` run."""

    run_dir: Path
    run_id: str
    scan_settings: SurfFrequencyScanSettings
    scan_settings_sha256: str
    step_direction: StepDirection
    config_toml_path: Optional[Path]
    config_toml_sha256: Optional[str]
    provenance_path: Path
    ingested_steps: Dict[str, SurfStepData] = field(default_factory=dict)
    ingested_order: List[str] = field(default_factory=list)
    recorded_ledger_keys: set[Tuple[str, str]] = field(default_factory=set)
    reconciled_finalized_sha256: Optional[str] = None
    last_reconciliation: Optional[SurfReconciliationReport] = None


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _append_strict_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one strict RFC-8259 JSON line (UTF-8 LF, ``allow_nan=False``) to disk."""
    serialized = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()


def _load_existing_ledger_keys(path: Path) -> set[Tuple[str, str]]:
    """Load previously recorded ``(event_type, relative_path_or_sha)`` keys to avoid duplicates."""
    keys: set[Tuple[str, str]] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                event_type = str(entry.get("event_type", ""))
                rel_path = str(entry.get("relative_path", ""))
                sha_val = str(entry.get("sha256", ""))
                if event_type and (rel_path or sha_val):
                    keys.add((event_type, rel_path or sha_val))
    return keys


def _step_filename_sort_key(path: Path, *, reverse_freq: bool) -> Tuple[int, float, str]:
    """Sort numeric ``CavityFiles/<freq>.csv`` filenames deterministically by frequency."""
    try:
        numeric = float(path.stem)
        return (0, -numeric if reverse_freq else numeric, path.name)
    except ValueError:
        return (1, 0.0, path.name)


# =============================================================================
# 3. SURF RUN WATCHER
# =============================================================================


class SurfRunWatcher:
    """Monitors ``<output.location>`` for SURF run trees and incrementally ingests steps."""

    def __init__(
        self,
        watch_root: Union[str, Path],
        *,
        run_prefix: Optional[str] = None,
        stability_polls: int = 2,
        min_stable_seconds: float = 0.25,
        snr_threshold: float = 5.0,
        parser: Optional[SurfOutputParser] = None,
        provenance_filename: str = "surf_ingest_provenance.jsonl",
        max_permission_retries: int = 10,
        fp64_atol: float = 1e-9,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stability_polls < 1:
            raise SurfIngestError("stability_polls must be >= 1.")
        if min_stable_seconds < 0.0:
            raise SurfIngestError("min_stable_seconds must be >= 0.0.")
        if max_permission_retries < 1:
            raise SurfIngestError("max_permission_retries must be >= 1.")

        self._watch_root = Path(watch_root)
        self._run_prefix = run_prefix
        self._stability_polls = int(stability_polls)
        self._min_stable_seconds = float(min_stable_seconds)
        self._snr_threshold = float(snr_threshold)
        self._parser = parser or SurfOutputParser()
        self._provenance_filename = provenance_filename
        self._max_permission_retries = int(max_permission_retries)
        self._fp64_atol = float(fp64_atol)
        self._clock = clock

        self._runs: Dict[Path, _WatchedRunState] = {}
        self._stability: Dict[Path, _FileStabilityState] = {}
        self._transient_issues: List[SurfTransientIssue] = []

    @property
    def watch_root(self) -> Path:
        """Return configured root directory being watched."""
        return self._watch_root

    @property
    def transient_issues(self) -> Tuple[SurfTransientIssue, ...]:
        """Return recorded retryable file-access or partial-write diagnostics."""
        return tuple(self._transient_issues)

    @property
    def discovered_run_dirs(self) -> Tuple[Path, ...]:
        """Return paths of all currently tracked SURF run directories."""
        return tuple(sorted(self._runs.keys()))

    def discover_run_dirs(self) -> List[Path]:
        """Discover ``<filename>_<k>/`` directories containing ``scan_settings.toml`` and ``CavityFiles/``."""
        if not self._watch_root.exists() or not self._watch_root.is_dir():
            return []

        if (self._watch_root / "scan_settings.toml").is_file() and (self._watch_root / "CavityFiles").is_dir():
            return [self._watch_root]

        discovered: List[Tuple[str, int, Path]] = []
        try:
            children = list(self._watch_root.iterdir())
        except PermissionError as exc:
            self._record_transient_issue(self._watch_root, exc, attempt=1)
            return []

        for child in children:
            if not child.is_dir():
                continue
            match = _RUN_DIR_SUFFIX_RE.match(child.name)
            if match is None:
                continue
            stem = match.group("stem")
            index_num = int(match.group("index"))
            if self._run_prefix is not None and stem != self._run_prefix:
                continue
            if (child / "scan_settings.toml").is_file() and (child / "CavityFiles").is_dir():
                discovered.append((stem, index_num, child))

        discovered.sort(key=lambda item: (item[0], item[1], item[2].name))
        return [item[2] for item in discovered]

    def _record_transient_issue(self, file_path: Path, exc: Exception, *, attempt: int) -> None:
        issue = SurfTransientIssue(
            file_path=file_path,
            error_type=type(exc).__name__,
            message=str(exc),
            attempt=attempt,
            recorded_at_utc=_utc_now_iso(),
        )
        self._transient_issues.append(issue)
        logger.warning(
            "Transient SURF watcher issue on %s (attempt %d): %s: %s",
            file_path,
            attempt,
            issue.error_type,
            issue.message,
        )

    def _is_file_stat_stable(self, file_path: Path) -> bool:
        """Apply Windows-safe ``size > 0`` and consecutive ``(size, mtime_ns)`` stability gate."""
        try:
            stat_result = file_path.stat()
        except PermissionError as exc:
            prev = self._stability.get(file_path)
            attempt = (prev.permission_retries + 1) if prev is not None else 1
            if prev is not None:
                prev.permission_retries = attempt
            self._record_transient_issue(file_path, exc, attempt=attempt)
            return False
        except OSError:
            return False

        if stat_result.st_size <= 0:
            self._stability.pop(file_path, None)
            return False

        now = self._clock()
        state = self._stability.get(file_path)
        if state is None:
            self._stability[file_path] = _FileStabilityState(
                size_bytes=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                first_stable_monotonic=now,
                consecutive_matches=1,
            )
            return self._stability_polls <= 1 and self._min_stable_seconds <= 0.0

        if state.size_bytes != stat_result.st_size or state.mtime_ns != stat_result.st_mtime_ns:
            state.size_bytes = stat_result.st_size
            state.mtime_ns = stat_result.st_mtime_ns
            state.first_stable_monotonic = now
            state.consecutive_matches = 1
            return False

        state.consecutive_matches += 1
        elapsed = now - state.first_stable_monotonic
        return state.consecutive_matches >= self._stability_polls and elapsed >= self._min_stable_seconds

    def _ensure_run_registered(self, run_dir: Path) -> Optional[_WatchedRunState]:
        """Load ``scan_settings.toml`` and initialize provenance ledger for a newly discovered run."""
        if run_dir in self._runs:
            return self._runs[run_dir]

        settings_path = run_dir / "scan_settings.toml"
        if not self._is_file_stat_stable(settings_path):
            return None

        try:
            settings_bytes = settings_path.read_bytes()
        except PermissionError as exc:
            state = self._stability.get(settings_path)
            attempt = (state.permission_retries + 1) if state is not None else 1
            if state is not None:
                state.permission_retries = attempt
            self._record_transient_issue(settings_path, exc, attempt=attempt)
            return None

        scan_settings = SurfFrequencyScanSettings.from_surf_toml(settings_bytes)
        settings_sha = compute_file_sha256(settings_path)
        direction = StepDirection.from_bounds(
            scan_settings.scan_parameters.start_freq,
            scan_settings.scan_parameters.end_freq,
        )

        stem = scan_settings.output_settings.filename
        config_path = run_dir / f"{stem}_config.toml"
        config_sha: Optional[str] = None
        if config_path.is_file():
            try:
                SurfConfig.from_surf_toml_file(config_path)
                config_sha = compute_file_sha256(config_path)
            except PermissionError as exc:
                self._record_transient_issue(config_path, exc, attempt=1)
                return None

        provenance_path = run_dir / self._provenance_filename
        recorded_keys = _load_existing_ledger_keys(provenance_path)

        run_state = _WatchedRunState(
            run_dir=run_dir,
            run_id=run_dir.name,
            scan_settings=scan_settings,
            scan_settings_sha256=settings_sha,
            step_direction=direction,
            config_toml_path=config_path if config_path.is_file() else None,
            config_toml_sha256=config_sha,
            provenance_path=provenance_path,
            recorded_ledger_keys=recorded_keys,
        )

        discovery_key = ("run_discovered", "scan_settings.toml")
        if discovery_key not in run_state.recorded_ledger_keys:
            discovery_record: Dict[str, Any] = {
                "event_type": "run_discovered",
                "run_id": run_state.run_id,
                "relative_path": "scan_settings.toml",
                "sha256": settings_sha,
                "config_toml_sha256": config_sha,
                "start_freq_mhz": float(scan_settings.scan_parameters.start_freq),
                "end_freq_mhz": float(scan_settings.scan_parameters.end_freq),
                "step_size_mhz": float(scan_settings.scan_parameters.step_size),
                "step_direction": direction.value,
                "gas_name": scan_settings.experiment.gas_name,
                "recorded_at_utc": _utc_now_iso(),
            }
            _append_strict_jsonl(provenance_path, discovery_record)
            run_state.recorded_ledger_keys.add(discovery_key)

        self._runs[run_dir] = run_state
        return run_state

    @staticmethod
    def _build_cavity_health(run_state: _WatchedRunState) -> SurfCavityTrackHealth:
        """Assemble cumulative cavity-track health telemetry for a tracked run."""
        if not run_state.ingested_steps:
            return SurfCavityTrackHealth(
                status="idle",
                step_count=0,
                dropout_count=0,
            )

        ordered_steps = [run_state.ingested_steps[name] for name in run_state.ingested_order]
        center_freqs = tuple(step.center_freq_mhz for step in ordered_steps)
        cavity_positions = tuple(step.cavity_position_mm for step in ordered_steps)
        dropout_flags = tuple(step.cavity_dropout for step in ordered_steps)
        dropout_freqs = tuple(step.center_freq_mhz for step in ordered_steps if step.cavity_dropout)
        latest = ordered_steps[-1]

        return SurfCavityTrackHealth(
            status="rail_dropout" if dropout_freqs else "healthy",
            step_count=len(ordered_steps),
            last_center_freq_mhz=latest.center_freq_mhz,
            last_cavity_position_mm=latest.cavity_position_mm,
            last_step_dropout=latest.cavity_dropout,
            dropout_count=len(dropout_freqs),
            center_freqs_mhz=center_freqs,
            cavity_positions_mm=cavity_positions,
            dropout_flags=dropout_flags,
            dropout_freqs_mhz=dropout_freqs,
        )

    def _try_reconcile_run(
        self,
        run_state: _WatchedRunState,
        *,
        force: bool = False,
    ) -> Optional[SurfReconciliationReport]:
        """Reconcile finalized compacted ``<filename>.csv`` when stable and ready."""
        stem = run_state.scan_settings.output_settings.filename
        finalized_path = run_state.run_dir / f"{stem}.csv"
        if not finalized_path.is_file():
            return None
        if not run_state.ingested_steps and not force:
            return None

        if not force and not self._is_file_stat_stable(finalized_path):
            return None

        try:
            finalized_bytes = finalized_path.read_bytes()
        except PermissionError as exc:
            state = self._stability.get(finalized_path)
            attempt = (state.permission_retries + 1) if state is not None else 1
            if state is not None:
                state.permission_retries = attempt
            self._record_transient_issue(finalized_path, exc, attempt=attempt)
            return None

        try:
            parsed_finalized = self._parser.parse_finalized_csv_bytes(
                finalized_bytes,
                source_path=finalized_path,
            )
        except (SurfCsvSchemaError, SurfBoundsError) as exc:
            if not force:
                state = self._stability.get(finalized_path)
                attempt = (state.schema_retries + 1) if state is not None else 1
                if state is not None:
                    state.schema_retries = attempt
                self._record_transient_issue(finalized_path, exc, attempt=attempt)
                return None
            raise

        # Only trigger automatic end-of-run reconciliation once finalize_csv() has compacted metadata.
        # For multi-step scans, wait until >= 2 compacted metadata entries (or all expected steps) exist
        # so an unfinalized single-step append (metadata at row 0 only) does not prematurely reconcile.
        expected_step_count = len(self._parser.expected_step_frequencies(run_state.scan_settings))
        min_auto_steps = 2 if expected_step_count >= 2 else 1
        if not force and not (
            parsed_finalized.is_compacted
            and len(parsed_finalized.center_freqs_mhz) >= max(len(run_state.ingested_steps), min_auto_steps)
        ):
            return None

        if run_state.reconciled_finalized_sha256 == parsed_finalized.file_sha256 and not force:
            return None

        report = self._parser.reconcile_run(
            finalized_csv_path=finalized_path,
            step_files=list(run_state.ingested_steps.values()),
            scan_settings=run_state.scan_settings,
            config_toml_path=run_state.config_toml_path,
            scan_settings_toml_path=run_state.run_dir / "scan_settings.toml",
            fp64_atol=self._fp64_atol,
        )

        run_state.reconciled_finalized_sha256 = report.finalized_csv_sha256
        run_state.last_reconciliation = report

        reconcile_key = ("run_reconciled", report.finalized_csv_sha256)
        if reconcile_key not in run_state.recorded_ledger_keys:
            reconcile_record: Dict[str, Any] = {
                "event_type": "run_reconciled",
                "run_id": run_state.run_id,
                "relative_path": finalized_path.name,
                "sha256": report.finalized_csv_sha256,
                "reconciled": report.reconciled,
                "spectrum_matches": report.spectrum_matches,
                "has_cavity_dropout": report.has_cavity_dropout,
                "step_file_count": report.step_file_count,
                "finalized_metadata_count": report.finalized_metadata_count,
                "finalized_is_compacted": report.finalized_is_compacted,
                "max_freq_abs_diff_mhz": report.max_freq_abs_diff_mhz,
                "max_intensity_abs_diff": report.max_intensity_abs_diff,
                "missing_steps_mhz": list(report.missing_steps_mhz),
                "extra_steps_mhz": list(report.extra_steps_mhz),
                "cavity_dropout_steps_mhz": list(report.cavity_dropout_steps_mhz),
                "discrepancies": list(report.discrepancies),
                "warnings": list(report.warnings),
                "recorded_at_utc": _utc_now_iso(),
            }
            _append_strict_jsonl(run_state.provenance_path, reconcile_record)
            run_state.recorded_ledger_keys.add(reconcile_key)

        return report

    def poll_once(self) -> List[SurfLiveStepUpdate]:
        """Run one non-blocking poll cycle across discovered SURF runs and return newly landed steps."""
        updates: List[SurfLiveStepUpdate] = []
        run_dirs = self.discover_run_dirs()

        for run_dir in run_dirs:
            run_state = self._ensure_run_registered(run_dir)
            if run_state is None:
                continue

            cavity_dir = run_dir / "CavityFiles"
            try:
                candidate_files = [p for p in cavity_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
            except PermissionError as exc:
                self._record_transient_issue(cavity_dir, exc, attempt=1)
                continue

            candidate_files.sort(
                key=lambda item: _step_filename_sort_key(
                    item,
                    reverse_freq=(run_state.step_direction == StepDirection.Down),
                )
            )

            step_size = float(run_state.scan_settings.scan_parameters.step_size)
            gas_name = run_state.scan_settings.experiment.gas_name

            for step_path in candidate_files:
                if step_path.name in run_state.ingested_steps:
                    continue

                if not self._is_file_stat_stable(step_path):
                    continue

                try:
                    raw_bytes = step_path.read_bytes()
                except PermissionError as exc:
                    st = self._stability.get(step_path)
                    attempt = (st.permission_retries + 1) if st is not None else 1
                    if st is not None:
                        st.permission_retries = attempt
                    self._record_transient_issue(step_path, exc, attempt=attempt)
                    continue

                try:
                    step_data = self._parser.parse_step_csv_bytes(
                        raw_bytes,
                        source_path=step_path,
                        step_size_mhz=step_size,
                        step_direction=run_state.step_direction,
                        verify_filename_stem=True,
                    )
                except (SurfCsvSchemaError, SurfBoundsError) as exc:
                    st = self._stability.get(step_path)
                    attempt = (st.schema_retries + 1) if st is not None else 1
                    if st is not None:
                        st.schema_retries = attempt
                    self._record_transient_issue(step_path, exc, attempt=attempt)
                    continue

                run_state.ingested_steps[step_path.name] = step_data
                run_state.ingested_order.append(step_path.name)

                stitched_freq, stitched_int, _ = self._parser.stitch_steps(
                    list(run_state.ingested_steps.values()),
                    step_direction=run_state.step_direction,
                )
                cavity_health = self._build_cavity_health(run_state)

                if step_data.stitched_freq_mhz.size > 0:
                    pick_result = pick_surf_doppler_peaks(
                        step_data.stitched_freq_mhz,
                        step_data.stitched_intensity,
                        gas_name=gas_name,
                        snr_threshold=self._snr_threshold,
                        source_step_sha256=step_data.file_sha256,
                        source_file=step_path.name,
                    )
                    proposed = pick_result.candidates
                else:
                    proposed = ()

                rel_path = f"CavityFiles/{step_path.name}"
                provenance_record: Dict[str, Any] = {
                    "event_type": "step_ingested",
                    "run_id": run_state.run_id,
                    "relative_path": rel_path,
                    "file_path": str(step_path),
                    "sha256": step_data.file_sha256,
                    "center_freq_mhz": step_data.center_freq_mhz,
                    "cavity_position_mm": step_data.cavity_position_mm,
                    "cavity_dropout": step_data.cavity_dropout,
                    "step_size_mhz": step_size,
                    "step_direction": run_state.step_direction.value,
                    "full_window_point_count": int(step_data.full_window_freq_mhz.size),
                    "stitched_point_count": int(step_data.stitched_freq_mhz.size),
                    "proposed_peak_count": len(proposed),
                    "recorded_at_utc": _utc_now_iso(),
                }
                step_key = ("step_ingested", rel_path)
                if step_key not in run_state.recorded_ledger_keys:
                    _append_strict_jsonl(run_state.provenance_path, provenance_record)
                    run_state.recorded_ledger_keys.add(step_key)

                reconciliation_report = self._try_reconcile_run(run_state, force=False)

                updates.append(
                    SurfLiveStepUpdate(
                        run_dir=run_dir,
                        run_id=run_state.run_id,
                        step_index=len(run_state.ingested_order),
                        step_data=step_data,
                        cumulative_freq_mhz=stitched_freq,
                        cumulative_intensity=stitched_int,
                        cavity_health=cavity_health,
                        proposed_candidates=proposed,
                        provenance_record=provenance_record,
                        reconciliation=reconciliation_report,
                    )
                )

            # Also check for finalized CSV reconciliation even when no new step CSV arrived
            if run_state.ingested_steps and run_state.reconciled_finalized_sha256 is None:
                self._try_reconcile_run(run_state, force=False)

        return updates

    def iter_updates(
        self,
        *,
        poll_interval_s: float = 0.25,
        max_polls: Optional[int] = None,
        stop_when_reconciled: bool = False,
    ) -> Iterator[SurfLiveStepUpdate]:
        """Yield incremental ``SurfLiveStepUpdate`` payloads across polling rounds."""
        poll_count = 0
        while max_polls is None or poll_count < max_polls:
            poll_count += 1
            for update in self.poll_once():
                yield update
            if stop_when_reconciled and self._runs and all(
                state.last_reconciliation is not None for state in self._runs.values()
            ):
                break
            if max_polls is None or poll_count < max_polls:
                if poll_interval_s > 0.0:
                    time.sleep(poll_interval_s)

    def get_stitched_spectrum(self, run_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Return cumulative stitched ``(freq_mhz, intensity)`` arrays for a tracked run."""
        resolved = Path(run_dir)
        if resolved not in self._runs:
            raise KeyError(f"SURF run directory not tracked by watcher: {resolved}")
        run_state = self._runs[resolved]
        stitched_freq, stitched_int, _ = self._parser.stitch_steps(
            list(run_state.ingested_steps.values()),
            step_direction=run_state.step_direction,
        )
        return stitched_freq, stitched_int

    def get_cavity_health(self, run_dir: Union[str, Path]) -> SurfCavityTrackHealth:
        """Return current cavity-track health telemetry for a tracked run."""
        resolved = Path(run_dir)
        if resolved not in self._runs:
            raise KeyError(f"SURF run directory not tracked by watcher: {resolved}")
        return self._build_cavity_health(self._runs[resolved])

    def get_reconciliation(self, run_dir: Union[str, Path]) -> Optional[SurfReconciliationReport]:
        """Return latest end-of-run reconciliation report for a tracked run, if available."""
        resolved = Path(run_dir)
        if resolved not in self._runs:
            raise KeyError(f"SURF run directory not tracked by watcher: {resolved}")
        return self._runs[resolved].last_reconciliation

    def reconcile_run(self, run_dir: Union[str, Path]) -> SurfReconciliationReport:
        """Explicitly run end-of-run reconciliation for a tracked SURF run directory."""
        resolved = Path(run_dir)
        if resolved not in self._runs:
            raise KeyError(f"SURF run directory not tracked by watcher: {resolved}")
        report = self._try_reconcile_run(self._runs[resolved], force=True)
        if report is None:
            raise FileNotFoundError(f"Finalized CSV not found in SURF run directory: {resolved}")
        return report

    def pick_run_peaks(
        self,
        run_dir: Union[str, Path],
        *,
        snr_threshold: Optional[float] = None,
    ) -> SurfPeakPickResult:
        """Pick Doppler doublet & singlet peaks across the full cumulative stitched spectrum."""
        resolved = Path(run_dir)
        if resolved not in self._runs:
            raise KeyError(f"SURF run directory not tracked by watcher: {resolved}")
        run_state = self._runs[resolved]
        stitched_freq, stitched_int, ordered_steps = self._parser.stitch_steps(
            list(run_state.ingested_steps.values()),
            step_direction=run_state.step_direction,
        )
        spans: List[Tuple[float, float, str]] = []
        for step in ordered_steps:
            if step.stitched_freq_mhz.size > 0:
                spans.append(
                    (
                        float(np.min(step.stitched_freq_mhz)),
                        float(np.max(step.stitched_freq_mhz)),
                        step.file_sha256,
                    )
                )
        return pick_surf_doppler_peaks(
            stitched_freq,
            stitched_int,
            gas_name=run_state.scan_settings.experiment.gas_name,
            snr_threshold=self._snr_threshold if snr_threshold is None else snr_threshold,
            step_provenance_spans=spans,
            source_file=f"{run_state.scan_settings.output_settings.filename}.csv",
        )


__all__ = [
    "SurfCavityTrackHealth",
    "SurfLiveStepUpdate",
    "SurfRunWatcher",
    "SurfTransientIssue",
]
