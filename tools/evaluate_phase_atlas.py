#!/usr/bin/env python
"""Evaluate PhaseSketch and PTM agreement on a real EXTXYZ dataset.

The result is deliberately called an agreement report rather than an accuracy
report: production datasets normally do not carry trusted phase labels.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms

from NepTrainKit.core.structure import Structure
from NepTrainKit.core.audit.phase_refinement import refine_l12, refine_laves
from tools.benchmark_phase_sketch import (
    Frame,
    _phase_label,
    _ptm_prediction,
    _sketch_frame,
    build_training_frames,
    fit_phase_sketch,
)


def _composition_key(symbols: np.ndarray) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(str(value) for value in symbols).items()))


def _composition_record(key: tuple[tuple[str, int], ...]) -> dict[str, object]:
    atom_count = sum(count for _, count in key)
    return {
        "counts": {symbol: count for symbol, count in key},
        "atomic_percent": {
            symbol: 100.0 * count / atom_count for symbol, count in key
        },
    }


def _fractions(values: list[str]) -> dict[str, float]:
    counts = Counter(values)
    return {
        label: count / len(values)
        for label, count in sorted(counts.items())
    }


def evaluate_dataset(
    path: Path,
    *,
    sample_per_composition: int = 10,
    seed: int = 20260716,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    populations: Counter[tuple[tuple[str, int], ...]] = Counter()
    selected: dict[tuple[tuple[str, int], ...], list[tuple[int, Atoms]]] = defaultdict(list)
    refinement_labels = {"l12": Counter(), "laves": Counter()}
    refinement_eligible = Counter()
    refinement_confirmed = {"l12": [], "laves": []}
    refinement_seconds = 0.0
    total_atoms = 0
    scan_started = time.perf_counter()
    for index, structure in enumerate(Structure.iter_read_multiple(str(path))):
        symbols = np.asarray(structure.atomic_properties["species"])
        key = _composition_key(symbols)
        populations[key] += 1
        total_atoms += len(symbols)
        atoms = Atoms(
            symbols=symbols.tolist(),
            positions=np.asarray(structure.atomic_properties["pos"]),
            cell=np.asarray(structure.lattice),
            pbc=True,
        )
        refinement_started = time.perf_counter()
        for candidate, refine in (("l12", refine_l12), ("laves", refine_laves)):
            result = refine(
                atoms.positions,
                atoms.cell.array,
                atoms.pbc,
                atoms.numbers,
            )
            if not result.a_types:
                continue
            refinement_eligible[candidate] += 1
            refinement_labels[candidate][result.label] += 1
            if result.confirmed:
                refinement_confirmed[candidate].append(index)
        refinement_seconds += time.perf_counter() - refinement_started
        reservoir = selected[key]
        if len(reservoir) < sample_per_composition:
            reservoir.append((index, atoms))
        else:
            replacement = int(rng.integers(populations[key]))
            if replacement < sample_per_composition:
                reservoir[replacement] = (index, atoms)
    scan_seconds = time.perf_counter() - scan_started

    rows = [
        (key, index, atoms)
        for key in sorted(selected)
        for index, atoms in sorted(selected[key])
    ]
    model, _ = fit_phase_sketch(build_training_frames())

    phase_started = time.perf_counter()
    sketches = [
        _sketch_frame(Frame(atoms, "unknown", "unknown", "real", str(index)))
        for _, index, atoms in rows
    ]
    phase_predictions = model.predict_many(sketches)
    phase_seconds = time.perf_counter() - phase_started

    ptm_started = time.perf_counter()
    ptm_predictions = [
        _ptm_prediction(Frame(atoms, "unknown", "unknown", "real", str(index)))
        for _, index, atoms in rows
    ]
    ptm_seconds = time.perf_counter() - ptm_started

    refinement_summary: dict[str, dict[str, object]] = {}
    for candidate in ("l12", "laves"):
        refinement_summary[candidate] = {
            "eligible_structures": refinement_eligible[candidate],
            "labels": dict(sorted(refinement_labels[candidate].items())),
            "confirmed_structures": len(refinement_confirmed[candidate]),
            "confirmed_indices": refinement_confirmed[candidate],
        }

    phase_geometry = [geometry for geometry, _ in phase_predictions]
    phase_labels = [_phase_label(*prediction) for prediction in phase_predictions]
    ptm_geometry = [geometry for geometry, _ in ptm_predictions]
    ptm_labels = [_phase_label(*prediction) for prediction in ptm_predictions]
    geometry_joint = Counter(zip(phase_geometry, ptm_geometry))

    composition_rows = []
    cursor = 0
    for key in sorted(selected):
        count = len(selected[key])
        stop = cursor + count
        composition_rows.append(
            {
                **_composition_record(key),
                "dataset_structures": populations[key],
                "sample_structures": count,
                "sample_indices": [index for index, _ in sorted(selected[key])],
                "phase_sketch": _fractions(phase_labels[cursor:stop]),
                "ptm": _fractions(ptm_labels[cursor:stop]),
                "structures": [
                    {
                        "index": index,
                        "phase_sketch": phase_labels[cursor + offset],
                        "ptm": ptm_labels[cursor + offset],
                    }
                    for offset, (index, _) in enumerate(sorted(selected[key]))
                ],
            }
        )
        cursor = stop

    elements = sorted(
        {symbol for key in populations for symbol, _ in key}
    )
    by_element_fraction: dict[str, list[dict[str, object]]] = {}
    for element in elements:
        grouped: dict[float, dict[str, object]] = {}
        for row in composition_rows:
            atomic_percent = float(row["atomic_percent"].get(element, 0.0))
            group = grouped.setdefault(
                atomic_percent,
                {
                    "atomic_percent": atomic_percent,
                    "dataset_structures": 0,
                    "phase_sketch_weighted": Counter(),
                    "ptm_weighted": Counter(),
                },
            )
            population = int(row["dataset_structures"])
            group["dataset_structures"] += population
            for label, fraction in row["phase_sketch"].items():
                group["phase_sketch_weighted"][label] += population * fraction
            for label, fraction in row["ptm"].items():
                group["ptm_weighted"][label] += population * fraction
        output_rows = []
        for atomic_percent, group in sorted(grouped.items()):
            population = int(group["dataset_structures"])
            output_rows.append(
                {
                    "atomic_percent": atomic_percent,
                    "dataset_structures": population,
                    "phase_sketch_estimated_fraction": {
                        label: value / population
                        for label, value in sorted(group["phase_sketch_weighted"].items())
                    },
                    "ptm_estimated_fraction": {
                        label: value / population
                        for label, value in sorted(group["ptm_weighted"].items())
                    },
                }
            )
        by_element_fraction[element] = output_rows

    return {
        "dataset": {
            "path": str(path),
            "structures": sum(populations.values()),
            "atoms": total_atoms,
            "composition_points": len(populations),
            "scan_seconds": scan_seconds,
        },
        "sample": {
            "strategy": "deterministic reservoir per exact composition",
            "seed": seed,
            "per_composition": sample_per_composition,
            "structures": len(rows),
        },
        "phase_sketch": {
            "geometry": dict(sorted(Counter(phase_geometry).items())),
            "phase": dict(sorted(Counter(phase_labels).items())),
            "seconds": phase_seconds,
        },
        "ptm": {
            "geometry": dict(sorted(Counter(ptm_geometry).items())),
            "phase": dict(sorted(Counter(ptm_labels).items())),
            "seconds": ptm_seconds,
        },
        "agreement": {
            "geometry": float(np.mean(np.asarray(phase_geometry) == np.asarray(ptm_geometry))),
            "phase": float(np.mean(np.asarray(phase_labels) == np.asarray(ptm_labels))),
            "geometry_joint": {
                f"PhaseSketch={left}|PTM={right}": count
                for (left, right), count in sorted(geometry_joint.items())
            },
        },
        "candidate_refinement": {
            **refinement_summary,
            "seconds": refinement_seconds,
            "note": "candidate confirmations are unlabelled findings, not accuracy",
        },
        "by_element_fraction": by_element_fraction,
        "by_composition": composition_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-per-composition", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_per_composition <= 0:
        parser.error("--sample-per-composition must be positive")
    result = evaluate_dataset(
        args.dataset,
        sample_per_composition=args.sample_per_composition,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
