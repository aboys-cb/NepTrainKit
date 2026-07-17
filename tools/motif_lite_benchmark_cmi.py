#!/usr/bin/env python
"""Benchmark motif-lite against the external CMI runner on generated fcc alloys."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sizes", default="2,3,4")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--skip-cmi", action="store_true")
    parser.add_argument("--mode", choices=["count", "pair"], default="count")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in [int(item) for item in args.sizes.split(",") if item.strip()]:
        case_dir = args.output_dir / f"fcc_size{size}"
        case_dir.mkdir(parents=True, exist_ok=True)
        dump_path = case_dir / "fcc.dump"
        xyz_path = case_dir / "fcc.xyz"

        _run(
            [
                sys.executable,
                str(ROOT / "tools/motif_lite_make_fcc_dump.py"),
                "--dump-output",
                str(dump_path),
                "--extxyz-output",
                str(xyz_path),
                "--size",
                str(size),
                "--seed",
                str(args.seed),
            ]
        )
        natoms = 4 * size**3
        motif_time = _run(
            [
                sys.executable,
                str(ROOT / "tools/motif_lite_analyze.py"),
                str(xyz_path),
                "--mode",
                args.mode,
                "--output",
                str(case_dir / "motif_lite.json"),
            ]
        )
        cmi_time = None
        if not args.skip_cmi:
            cmi_time = _run(
                [
                    sys.executable,
                    str(ROOT / "tools/cmi_run_official_example.py"),
                    "--frameworks",
                    str(args.frameworks),
                    "--dump-file",
                    str(dump_path),
                    "--root",
                    str(case_dir / "cmi_out"),
                    "--crystal-structure",
                    "fcc",
                ]
            )
        rows.append(
            {
                "size": size,
                "natoms": natoms,
                "motif_lite_seconds": motif_time,
                "cmi_seconds": cmi_time,
                "speedup_cmi_over_motif_lite": (cmi_time / motif_time) if cmi_time is not None else "",
            }
        )
        print(rows[-1])

    output_csv = args.output_csv or (args.output_dir / "benchmark.csv")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
