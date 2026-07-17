#!/usr/bin/env python
"""Compare local motif signatures across shell definitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from ase.io import iread, read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_analyze import structure_signatures


METHODS = {
    "natural-cutoff": {"shell_method": "natural-cutoff"},
    "knn12": {"shell_method": "knn", "shell_k": 12},
    "adaptive-gap": {"shell_method": "adaptive-gap", "adaptive_min_neighbors": 4, "adaptive_max_neighbors": 14},
}


def _cn(signature: str) -> int:
    return int(signature.rsplit("cn=", 1)[1].split(" | ", 1)[0])


def _summary(histogram: Counter[str], top: int) -> dict:
    total = sum(histogram.values())
    entropy = -sum((count / total) * math.log(count / total) for count in histogram.values()) if total else 0.0
    cn_counts: Counter[int] = Counter()
    for signature, count in histogram.items():
        cn_counts[_cn(signature)] += count
    return {
        "total_environments": total,
        "unique_signature_count": len(histogram),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(len(histogram)) if len(histogram) > 1 else 0.0,
        "cn_distribution": dict(sorted(cn_counts.items())),
        "top_signatures": [
            {"signature": signature, "count": count, "fraction": count / total if total else 0.0}
            for signature, count in histogram.most_common(top)
        ],
    }


def _top_overlap(histograms: dict[str, Counter[str]], top: int) -> dict:
    overlap = {}
    names = list(histograms)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_top = {signature for signature, _ in histograms[left].most_common(top)}
            right_top = {signature for signature, _ in histograms[right].most_common(top)}
            union = left_top | right_top
            overlap[f"{left}__{right}"] = {
                "shared_top_count": len(left_top & right_top),
                "jaccard": len(left_top & right_top) / len(union) if union else 0.0,
            }
    return overlap


def analyze_shell_sensitivity(atoms_list, top: int = 20, max_examples: int = 20) -> tuple[dict, list[dict]]:
    histograms = {name: Counter() for name in METHODS}
    frame_rows = []
    atom_total = 0
    stable_atoms = 0

    for index, atoms in enumerate(atoms_list):
        signatures = {name: structure_signatures(atoms, **kwargs) for name, kwargs in METHODS.items()}
        for name, local in signatures.items():
            histograms[name].update(local)

        changed = 0
        for atom_signatures in zip(*signatures.values()):
            if len(set(atom_signatures)) == 1:
                stable_atoms += 1
            else:
                changed += 1
        atom_total += len(atoms)
        frame_rows.append(
            {
                "index": index,
                "natoms": len(atoms),
                "config_type": str(atoms.info.get("Config_type", atoms.info.get("config_type", ""))),
                "changed_atoms": changed,
                "changed_fraction": changed / len(atoms) if atoms else 0.0,
                **{f"{name}_unique": len(set(local)) for name, local in signatures.items()},
            }
        )

    report = {
        "settings": {"methods": METHODS, "top": top},
        "total_structures": len(frame_rows),
        "total_atoms": atom_total,
        "stable_atom_fraction": stable_atoms / atom_total if atom_total else 0.0,
        "methods": {name: _summary(histogram, top) for name, histogram in histograms.items()},
        "top_overlap": _top_overlap(histograms, top),
        "unstable_structure_examples": sorted(
            frame_rows,
            key=lambda row: (-row["changed_fraction"], row["index"]),
        )[:max_examples],
    }
    return report, frame_rows


def _write_frame_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--index", default=":1000")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-csv", type=Path)
    args = parser.parse_args(argv)

    atoms_list = iread(args.input, index=args.index) if ":" in args.index else [read(args.input, index=args.index)]
    report, rows = analyze_shell_sensitivity(atoms_list, top=args.top, max_examples=args.max_examples)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.frame_csv:
        _write_frame_csv(args.frame_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
