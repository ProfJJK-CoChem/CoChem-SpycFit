"""Command-line interface entrypoint for CoChem-SPYFY."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from cochem_spyfy import __version__
from cochem_spyfy.module_provider import spyfy_provider


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``cochem-spyfy`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cochem-spyfy",
        description="CoChem-SPYFY computational spectroscopy fitting and SURF integration CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CoChem-SPYFY {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "info",
        help="Print JSON provider metadata, runtime calpgm availability, and SURF compatibility.",
    )
    subparsers.add_parser(
        "profiles",
        help="List registered instrument profiles and the active default profile.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the ``cochem-spyfy`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        print(json.dumps(spyfy_provider(), indent=2, sort_keys=True))
        return 0

    if args.command == "profiles":
        meta = spyfy_provider()
        payload = {
            "default_instrument_profile": meta["default_instrument_profile"],
            "instrument_profiles": meta["instrument_profiles"],
            "surf_compatibility": meta["surf_compatibility"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
