#!/usr/bin/env python
"""Analyze local chemistry signatures in ASE-readable structure files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from ase.io import iread, read
from ase.neighborlist import natural_cutoffs


def _format_counts(counts: Counter[str]) -> str:
    return " ".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts)) or "none"


def _distance_matrix(atoms) -> np.ndarray:
    n_atoms = len(atoms)
    if n_atoms == 0:
        return np.empty((0, 0))
    scaled = atoms.get_scaled_positions(wrap=False)
    delta = scaled[None, :, :] - scaled[:, None, :]
    pbc = np.asarray(atoms.pbc, dtype=bool)
    axes = [(-1, 0, 1) if periodic else (0,) for periodic in pbc]
    shifts = list(product(*axes))
    cell = np.asarray(atoms.cell)
    distances_sq = np.full((n_atoms, n_atoms), np.inf)
    for shift in shifts:
        vectors = (delta + np.asarray(shift)) @ cell
        distances_sq = np.minimum(distances_sq, np.einsum("ijk,ijk->ij", vectors, vectors))
    return np.sqrt(distances_sq)


def _adaptive_shell(distances: np.ndarray, min_neighbors: int, max_neighbors: int) -> list[int]:
    local = np.flatnonzero(distances > 1e-12)
    if len(local) == 0:
        return []
    ordered = local[np.argsort(distances[local])]
    limit = min(max_neighbors, len(ordered))
    if limit <= min_neighbors:
        return ordered[:limit].astype(int).tolist()
    shell_distances = distances[ordered[: limit + 1]]
    gaps = shell_distances[min_neighbors:limit] - shell_distances[min_neighbors - 1 : limit - 1]
    return ordered[: min_neighbors + int(np.argmax(gaps))].astype(int).tolist()


def _neighbor_indices(
    atoms,
    cutoff: float | None,
    mult: float,
    shell_method: str = "natural-cutoff",
    shell_k: int = 12,
    adaptive_min_neighbors: int = 4,
    adaptive_max_neighbors: int = 14,
) -> tuple[list[list[int]], list[set[int]]]:
    n_atoms = len(atoms)
    if n_atoms == 0:
        return [], []
    distances = _distance_matrix(atoms)
    if cutoff is None:
        if shell_method == "natural-cutoff":
            radii = np.asarray(natural_cutoffs(atoms, mult=mult), dtype=float)
            thresholds = radii[:, None] + radii[None, :]
            mask = (distances > 1e-12) & (distances <= thresholds)
            neighbors = [np.flatnonzero(mask[i]).astype(int).tolist() for i in range(n_atoms)]
        elif shell_method == "knn":
            neighbors = [
                np.argsort(np.where(distances[i] > 1e-12, distances[i], np.inf))[: min(shell_k, n_atoms - 1)]
                .astype(int)
                .tolist()
                for i in range(n_atoms)
            ]
        elif shell_method == "adaptive-gap":
            neighbors = [
                _adaptive_shell(distances[i], adaptive_min_neighbors, adaptive_max_neighbors)
                for i in range(n_atoms)
            ]
        else:
            raise ValueError("shell_method must be 'natural-cutoff', 'knn', or 'adaptive-gap'")
    else:
        thresholds = np.full((n_atoms, n_atoms), float(cutoff))
        mask = (distances > 1e-12) & (distances <= thresholds)
        neighbors = [np.flatnonzero(mask[i]).astype(int).tolist() for i in range(n_atoms)]
    return neighbors, [set(local) for local in neighbors]


def _pair_counts(symbols: list[str], neighbor_sets: list[set[int]], local_neighbors: list[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for i, atom_i in enumerate(local_neighbors):
        for atom_j in local_neighbors[i + 1 :]:
            if atom_j not in neighbor_sets[atom_i] and atom_i not in neighbor_sets[atom_j]:
                continue
            left, right = sorted((symbols[atom_i], symbols[atom_j]))
            counts[f"{left}-{right}"] += 1
    return counts


def structure_signatures(
    atoms,
    cutoff: float | None = None,
    mult: float = 1.2,
    mode: str = "count",
    shell_method: str = "natural-cutoff",
    shell_k: int = 12,
    adaptive_min_neighbors: int = 4,
    adaptive_max_neighbors: int = 14,
) -> list[str]:
    if mode not in {"count", "pair"}:
        raise ValueError("mode must be 'count' or 'pair'")
    symbols = atoms.get_chemical_symbols()
    neighbors, neighbor_sets = _neighbor_indices(
        atoms,
        cutoff,
        mult,
        shell_method=shell_method,
        shell_k=shell_k,
        adaptive_min_neighbors=adaptive_min_neighbors,
        adaptive_max_neighbors=adaptive_max_neighbors,
    )
    signatures = []
    for i, local_neighbors in enumerate(neighbors):
        counts = Counter(symbols[j] for j in local_neighbors)
        signature = f"{symbols[i]} | NN: {_format_counts(counts)} | cn={sum(counts.values())}"
        if mode == "pair":
            pairs = _pair_counts(symbols, neighbor_sets, local_neighbors)
            signature += f" | NN-pairs: {_format_counts(pairs)}"
        signatures.append(signature)
    return signatures


def analyze_structures(
    atoms_list,
    cutoff: float | None = None,
    mult: float = 1.2,
    rare_max_count: int = 2,
    mode: str = "count",
    shell_method: str = "natural-cutoff",
    shell_k: int = 12,
    adaptive_min_neighbors: int = 4,
    adaptive_max_neighbors: int = 14,
) -> dict:
    per_structure = []
    histogram: Counter[str] = Counter()
    structure_counts = []

    for index, atoms in enumerate(atoms_list):
        counts = Counter(
            structure_signatures(
                atoms,
                cutoff=cutoff,
                mult=mult,
                mode=mode,
                shell_method=shell_method,
                shell_k=shell_k,
                adaptive_min_neighbors=adaptive_min_neighbors,
                adaptive_max_neighbors=adaptive_max_neighbors,
            )
        )
        structure_counts.append(counts)
        histogram.update(counts)
        per_structure.append(
            {
                "index": index,
                "natoms": len(atoms),
                "config_type": str(atoms.info.get("Config_type", atoms.info.get("config_type", ""))),
                "unique_signature_count": len(counts),
            }
        )

    total = sum(histogram.values())
    entropy = -sum((count / total) * math.log(count / total) for count in histogram.values()) if total else 0.0
    normalized_entropy = entropy / math.log(len(histogram)) if len(histogram) > 1 else 0.0

    for item, counts in zip(per_structure, structure_counts):
        rare = {key: count for key, count in counts.items() if histogram[key] <= rare_max_count}
        item["rare_environment_count"] = sum(rare.values())
        item["rare_signatures"] = dict(sorted(rare.items()))

    return {
        "settings": {
            "cutoff": cutoff,
            "natural_cutoff_mult": mult,
            "rare_max_count": rare_max_count,
            "mode": mode,
            "shell_method": shell_method,
            "shell_k": shell_k,
            "adaptive_min_neighbors": adaptive_min_neighbors,
            "adaptive_max_neighbors": adaptive_max_neighbors,
        },
        "total_structures": len(per_structure),
        "total_environments": total,
        "unique_signature_count": len(histogram),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "histogram": [
            {"signature": key, "count": count, "fraction": count / total if total else 0.0}
            for key, count in histogram.most_common()
        ],
        "structures": per_structure,
    }


def _write_histogram_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signature", "count", "fraction"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ASE-readable structure file, for example extxyz/xyz/vasp")
    parser.add_argument("--index", default=":", help='ASE index selector, default ":" for all frames')
    parser.add_argument("--cutoff", type=float, default=None, help="Fixed neighbor cutoff in Angstrom")
    parser.add_argument("--natural-cutoff-mult", type=float, default=1.2, help="ASE natural cutoff multiplier")
    parser.add_argument("--shell-method", choices=["natural-cutoff", "knn", "adaptive-gap"], default="natural-cutoff")
    parser.add_argument("--shell-k", type=int, default=12, help="Neighbor count for --shell-method knn")
    parser.add_argument("--adaptive-min-neighbors", type=int, default=4)
    parser.add_argument("--adaptive-max-neighbors", type=int, default=14)
    parser.add_argument("--rare-max-count", type=int, default=2, help="Global count threshold for rare signatures")
    parser.add_argument("--mode", choices=["count", "pair"], default="count")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--hist-csv", type=Path, help="Write histogram CSV to this path")
    args = parser.parse_args(argv)

    atoms_list = iread(args.input, index=args.index) if ":" in args.index else [read(args.input, index=args.index)]
    report = analyze_structures(
        atoms_list,
        cutoff=args.cutoff,
        mult=args.natural_cutoff_mult,
        rare_max_count=args.rare_max_count,
        mode=args.mode,
        shell_method=args.shell_method,
        shell_k=args.shell_k,
        adaptive_min_neighbors=args.adaptive_min_neighbors,
        adaptive_max_neighbors=args.adaptive_max_neighbors,
    )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    if args.hist_csv:
        _write_histogram_csv(args.hist_csv, report["histogram"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
