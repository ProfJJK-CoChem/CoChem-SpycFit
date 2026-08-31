r"""
CoChem Necessity-First File Triage Protocol — Phase 3: Cleanup Executor
========================================================================
Reads triage manifests and executes file moves/deletes based on verdicts.

Usage:
    python cleanup_executor.py --manifests D:\__CoChem\.docs\triage --dry-run
    python cleanup_executor.py --manifests D:\__CoChem\.docs\triage --execute

Actions by verdict:
    PURGE  — Permanently delete (caches, node_modules, audit debris)
    TRASH  — Move to D:\__CoChem\.trash\<date>\ with undo manifest
    KEEP   — No action
    TRIAGE — No action (needs Phase 2 LLM review first)
    EXCLUDE — No action
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union


def load_manifests(manifest_path: Union[str, Path]) -> list[dict]:
    """Load all triage manifests from an output directory or a single manifest file."""
    p = Path(manifest_path)
    manifests: list[dict] = []
    if p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            manifests.append(json.load(fh))
    elif p.is_dir():
        for f in sorted(p.glob("*_triage_manifest.json")):
            with open(f, "r", encoding="utf-8") as fh:
                manifests.append(json.load(fh))
    return manifests


def plan_actions(manifests: list[dict]) -> dict:
    """Build an action plan from manifests."""
    plan: dict[str, Any] = {
        "purge_files": [],
        "purge_dirs": [],
        "trash_files": [],
        "keep_count": 0,
        "triage_count": 0,
        "exclude_count": 0,
    }

    for manifest in manifests:
        target_dir = Path(manifest.get("target_dir", "."))

        for filepath_str in manifest.get("files", {}).get("PURGE", []):
            p = Path(filepath_str)
            filepath = p if p.is_absolute() else target_dir / filepath_str
            if filepath.exists():
                plan["purge_files"].append(str(filepath))

        for filepath_str in manifest.get("files", {}).get("TRASH", []):
            p = Path(filepath_str)
            if p.is_absolute():
                filepath = p
                try:
                    rel_path = filepath.relative_to(target_dir).as_posix()
                except ValueError:
                    rel_path = filepath.name
            else:
                filepath = target_dir / filepath_str
                rel_path = Path(filepath_str).as_posix()

            if filepath.exists():
                plan["trash_files"].append({
                    "src": str(filepath),
                    "target_dir": str(target_dir),
                    "rel_path": rel_path,
                })

        plan["keep_count"] += manifest.get("counts", {}).get("KEEP", 0)
        plan["triage_count"] += manifest.get("counts", {}).get("TRIAGE", 0)
        plan["exclude_count"] += manifest.get("counts", {}).get("EXCLUDE", 0)

    # Optimize: identify entire directories that can be purged at once
    purge_dirs = _find_purgeable_dirs(plan["purge_files"])
    plan["purge_dirs"] = purge_dirs

    # Remove individual files that are inside purgeable dirs
    purge_dir_set = {Path(d).resolve() for d in purge_dirs}
    plan["purge_files"] = [
        f for f in plan["purge_files"]
        if not any(Path(f).resolve().is_relative_to(pd) for pd in purge_dir_set)
    ]

    return plan


def _find_purgeable_dirs(file_list: list[str], min_files: int = 5) -> list[str]:
    """Identify directories where ALL files are being purged."""
    from collections import Counter

    dir_counts: Dict[str, int] = Counter()
    for f in file_list:
        parent = str(Path(f).parent)
        dir_counts[parent] += 1

    file_set = {Path(f).resolve() for f in file_list}

    # A directory is purgeable if it has enough files and they're all being purged
    purgeable = []
    for dir_path, count in dir_counts.items():
        if count >= min_files:
            dp = Path(dir_path)
            if dp.exists() and dp.is_dir():
                actual_files = [f.resolve() for f in dp.rglob("*") if f.is_file()]
                if len(actual_files) >= min_files and all(f in file_set for f in actual_files):
                    purgeable.append(dir_path)

    # Deduplicate: if parent is purgeable, don't list children
    purgeable.sort(key=len)
    final: list[str] = []
    for p in purgeable:
        pp = Path(p)
        if not any(pp.is_relative_to(Path(f)) for f in final):
            final.append(p)

    return final


def execute_plan(plan: dict, trash_root: Path, dry_run: bool = True) -> dict:
    """Execute the cleanup plan."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_dest = Path(trash_root) / timestamp
    results: dict[str, Any] = {
        "timestamp": timestamp,
        "dry_run": dry_run,
        "purged_dirs": 0,
        "purged_files": 0,
        "trashed_files": 0,
        "errors": [],
        "bytes_recovered": 0,
    }
    trashed_records: list[dict[str, str]] = []

    # ── Purge directories ──
    for dir_path in plan.get("purge_dirs", []):
        dp = Path(dir_path)
        if dp.exists():
            size = _dir_size(dp)
            if dry_run:
                file_count = sum(1 for _ in dp.rglob("*") if _.is_file())
                print(f"  [DRY-RUN] PURGE DIR: {dir_path} ({file_count} files, {_fmt_size(size)})")
            else:
                try:
                    shutil.rmtree(dp)
                    print(f"  [PURGED] DIR: {dir_path} ({_fmt_size(size)})")
                except Exception as e:
                    results["errors"].append(f"PURGE DIR {dir_path}: {e}")
                    print(f"  [ERROR] PURGE DIR: {dir_path}: {e}")
                    continue
            results["purged_dirs"] += 1
            results["bytes_recovered"] += size

    # ── Purge individual files ──
    for filepath in plan.get("purge_files", []):
        fp = Path(filepath)
        if fp.exists():
            size = fp.stat().st_size
            if dry_run:
                print(f"  [DRY-RUN] PURGE: {filepath} ({_fmt_size(size)})")
            else:
                try:
                    fp.unlink()
                    print(f"  [PURGED] FILE: {filepath} ({_fmt_size(size)})")
                except Exception as e:
                    results["errors"].append(f"PURGE {filepath}: {e}")
                    print(f"  [ERROR] PURGE {filepath}: {e}")
                    continue
            results["purged_files"] += 1
            results["bytes_recovered"] += size

    # ── Trash files (safe move) ──
    trash_files = plan.get("trash_files", [])
    if trash_files:
        if not dry_run:
            trash_dest.mkdir(parents=True, exist_ok=True)

        for item in trash_files:
            if isinstance(item, dict):
                src_path = Path(item["src"])
                if not src_path.exists():
                    continue
                rel_path = Path(item.get("rel_path", src_path.name))
                dest = trash_dest / rel_path
            else:
                src_path = Path(item)
                if not src_path.exists():
                    continue
                target_dir = Path(plan.get("target_dir", src_path.parent))
                try:
                    rel_path = src_path.relative_to(target_dir)
                except ValueError:
                    rel_path = Path(src_path.name)
                dest = trash_dest / rel_path

            try:
                size = src_path.stat().st_size
            except OSError:
                size = 0

            if dry_run:
                print(f"  [DRY-RUN] TRASH: {src_path} -> {dest}")
                results["trashed_files"] += 1
                results["bytes_recovered"] += size
            else:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dest))
                    trashed_records.append({
                        "original": str(src_path),
                        "trashed_to": str(dest),
                    })
                    results["trashed_files"] += 1
                    results["bytes_recovered"] += size
                    print(f"  [TRASHED] {src_path} -> {dest}")
                except Exception as e:
                    results["errors"].append(f"TRASH {src_path}: {e}")
                    print(f"  [ERROR] TRASH {src_path}: {e}")
                    continue

    # ── Write undo manifest ──
    if not dry_run and trashed_records:
        undo_manifest = {
            "timestamp": timestamp,
            "trash_dest": str(trash_dest),
            "files": trashed_records,
        }
        undo_path = trash_dest / "_undo_manifest.json"
        with open(undo_path, "w", encoding="utf-8") as fh:
            json.dump(undo_manifest, fh, indent=2)

    return results


def _dir_size(path: Path) -> int:
    """Calculate total size of a directory."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _fmt_size(size_bytes: int | float) -> str:
    """Format bytes as human-readable."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_results(results: dict, plan: dict) -> None:
    """Print execution summary."""
    mode = "DRY-RUN" if results.get("dry_run", True) else "EXECUTED"
    print()
    print("=" * 60)
    print(f"  NFTP Phase 3: Cleanup Results ({mode})")
    print("=" * 60)
    print(f"  Directories purged:  {results.get('purged_dirs', 0)}")
    print(f"  Files purged:        {results.get('purged_files', 0)}")
    print(f"  Files trashed:       {results.get('trashed_files', 0)}")
    print(f"  Space recovered:     {_fmt_size(results.get('bytes_recovered', 0))}")
    print(f"  Errors:              {len(results.get('errors', []))}")
    print()
    print(f"  Files kept (KEEP):   {plan.get('keep_count', 0)}")
    print(f"  Files pending (TRIAGE): {plan.get('triage_count', 0)}")
    if results.get("errors"):
        print()
        print("  Errors:")
        for e in results["errors"][:10]:
            print(f"    - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFTP Phase 3: Cleanup Executor")
    parser.add_argument("--manifests", required=True, help="Directory or file containing triage manifest(s)")
    parser.add_argument("--trash", default=r"D:\__CoChem\.trash", help="Trash root directory")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", default=False, help="Preview only (default)")
    mode_group.add_argument("--execute", action="store_true", default=False, help="Actually move/delete files")
    args = parser.parse_args()

    manifest_path = Path(args.manifests).resolve()
    trash_root = Path(args.trash).resolve()
    dry_run = not args.execute

    if not manifest_path.exists():
        print(f"ERROR: Manifest path not found: {manifest_path}")
        sys.exit(1)

    manifests = load_manifests(manifest_path)
    if not manifests:
        print(f"ERROR: No manifests found in {manifest_path}")
        sys.exit(1)

    # Inject target_dir into plan for trash pathing
    plan = plan_actions(manifests)
    if manifests and "target_dir" not in plan:
        plan["target_dir"] = manifests[0].get("target_dir", "")

    print(f"  Loaded {len(manifests)} manifests")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'EXECUTE'}")
    print()

    results = execute_plan(plan, trash_root, dry_run=dry_run)
    print_results(results, plan)
