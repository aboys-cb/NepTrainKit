#!/usr/bin/env python
"""Evaluate motif-lite representativeness against CMI on generated fcc alloys."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_analyze import analyze_structures, structure_signatures


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def _js_divergence(left: Counter[str], right: Counter[str]) -> float:
    labels = set(left) | set(right)
    ltotal = sum(left.values())
    rtotal = sum(right.values())
    if not ltotal or not rtotal:
        return 0.0

    def kl(counter: Counter[str], total: int, middle: dict[str, float]) -> float:
        out = 0.0
        for label in labels:
            p = counter[label] / total
            if p:
                out += p * math.log(p / middle[label])
        return out

    middle = {label: 0.5 * (left[label] / ltotal + right[label] / rtotal) for label in labels}
    return 0.5 * kl(left, ltotal, middle) + 0.5 * kl(right, rtotal, middle)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    xmean = sum(x) / len(x)
    ymean = sum(y) / len(y)
    xdev = [value - xmean for value in x]
    ydev = [value - ymean for value in y]
    denom = math.sqrt(sum(v * v for v in xdev) * sum(v * v for v in ydev))
    return sum(a * b for a, b in zip(xdev, ydev)) / denom if denom else None


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def _run(cmd: list[str], quiet: bool) -> None:
    kwargs = {"check": True}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
    subprocess.run(cmd, **kwargs)


def _cmi_composition_labels(root: Path, elements: list[str]) -> list[str]:
    centers = np.load(root / "central_atoms.npy").astype(int)
    concentrations = np.load(root / "concentration_before_permutation.npy").astype(int)
    labels = []
    for center, counts in zip(centers, concentrations):
        by_symbol = {elements[i]: int(counts[i]) for i in range(len(elements))}
        count_text = " ".join(f"{symbol}{by_symbol[symbol]}" for symbol in sorted(by_symbol) if by_symbol[symbol])
        labels.append(f"{elements[int(center)]} | NN: {count_text or 'none'} | cn={int(np.sum(counts))}")
    return labels


def _case(args, seed: int) -> dict:
    elements = [item.strip() for item in args.elements.split(",") if item.strip()]
    case_dir = args.output_dir / f"seed_{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_path = case_dir / "fcc.dump"
    xyz_path = case_dir / "fcc.xyz"
    cmi_root = case_dir / "cmi_out"

    if not args.reuse or not (cmi_root / "shell_ids.npy").exists():
        _run(
            [
                sys.executable,
                str(ROOT / "tools/motif_lite_make_fcc_dump.py"),
                "--dump-output",
                str(dump_path),
                "--extxyz-output",
                str(xyz_path),
                "--size",
                str(args.size),
                "--seed",
                str(seed),
                "--elements",
                args.elements,
            ],
            quiet=True,
        )
        _run(
            [
                sys.executable,
                str(ROOT / "tools/cmi_run_official_example.py"),
                "--frameworks",
                str(args.frameworks),
                "--dump-file",
                str(dump_path),
                "--root",
                str(cmi_root),
                "--crystal-structure",
                "fcc",
            ],
            quiet=args.quiet,
        )

    atoms = read(xyz_path, index=0)
    count_labels = structure_signatures(atoms, mode="count")
    motif_labels = structure_signatures(atoms, mode=args.mode)
    motif_counts = Counter(motif_labels)
    cmi_composition_labels = _cmi_composition_labels(cmi_root, elements)
    cmi_composition_counts = Counter(cmi_composition_labels)
    shell_ids = np.load(cmi_root / "shell_ids.npy")
    centers = np.load(cmi_root / "central_atoms.npy").astype(int)
    cmi_full_labels = [f"{elements[center]}|{shell_id}" for center, shell_id in zip(centers, shell_ids)]
    cmi_full_counts = Counter(cmi_full_labels)

    split_by_motif = {}
    for motif, full in zip(motif_labels, cmi_full_labels):
        split_by_motif.setdefault(motif, set()).add(full)
    split_sizes = [len(values) for values in split_by_motif.values()]

    agreement = sum(a == b for a, b in zip(count_labels, cmi_composition_labels)) / len(count_labels)
    motif_report = analyze_structures([atoms], mode=args.mode)
    return {
        "seed": seed,
        "natoms": len(atoms),
        "motif_lite_unique": len(motif_counts),
        "cmi_composition_unique": len(cmi_composition_counts),
        "cmi_full_unique": len(cmi_full_counts),
        "composition_agreement": agreement,
        "js_count_vs_cmi_composition": _js_divergence(Counter(count_labels), cmi_composition_counts),
        "motif_entropy": motif_report["entropy"],
        "cmi_composition_entropy": _entropy(cmi_composition_counts),
        "cmi_full_entropy": _entropy(cmi_full_counts),
        "full_per_motif_lite_unique": len(cmi_full_counts) / len(motif_counts),
        "max_cmi_full_split_per_motif_lite": max(split_sizes) if split_sizes else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--elements", default="Ni,Co,Cr")
    parser.add_argument("--mode", choices=["count", "pair"], default="count")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_case(args, int(seed)) for seed in args.seeds.split(",") if seed.strip()]
    csv_path = args.output_dir / "representativeness.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": rows,
        "mean_composition_agreement": sum(row["composition_agreement"] for row in rows) / len(rows),
        "mean_js_count_vs_cmi_composition": sum(row["js_count_vs_cmi_composition"] for row in rows) / len(rows),
        "mean_full_per_motif_lite_unique": sum(row["full_per_motif_lite_unique"] for row in rows) / len(rows),
        "spearman_motif_vs_cmi_composition_entropy": _spearman(
            [row["motif_entropy"] for row in rows],
            [row["cmi_composition_entropy"] for row in rows],
        ),
        "spearman_motif_vs_cmi_full_entropy": _spearman(
            [row["motif_entropy"] for row in rows],
            [row["cmi_full_entropy"] for row in rows],
        ),
    }
    json_path = args.output_dir / "representativeness_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
