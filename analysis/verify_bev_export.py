#!/usr/bin/env python3
"""Compile export-related modules to catch syntax errors quickly."""
from __future__ import annotations

import argparse
import compileall
from pathlib import Path
import sys


TARGETS = [
    Path("mmcv/models/detectors/uniad_e2e.py"),
    Path("analysis/export_bev_traj_pairs.py"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompile UniAD export helpers to surface syntax issues quickly. "
            "This command does not export data; use export_bev_traj_pairs.py "
            "for that."
        )
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Optional hint that prints an example export command using the "
            "provided directory."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    success = True
    for relative in TARGETS:
        target = (repo_root / relative).resolve()
        if not target.exists():
            print(f"Missing target: {target}", file=sys.stderr)
            success = False
            continue
        compiled = compileall.compile_file(str(target), force=True, quiet=1)
        if not compiled:
            print(f"Failed to compile: {target}", file=sys.stderr)
            success = False
    if not success:
        sys.exit(1)

    if args.out_dir:
        print(
            "To export BEV/token pairs run:\n"
            "  python analysis/export_bev_traj_pairs.py "
            "<config.py> <checkpoint.pth> --out-dir "
            f"{args.out_dir}"
        )

    print("Compiled export-related modules successfully.")
    print("No data was written; run export_bev_traj_pairs.py to dump files.")


if __name__ == "__main__":
    main()