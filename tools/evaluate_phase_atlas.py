#!/usr/bin/env python
"""Evaluate PhaseSketch, PTM, and CNA agreement on a real EXTXYZ dataset.

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


def _cna_prediction(frame: Frame) -> str:
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import CommonNeighborAnalysisModifier
    from ovito.pipeline import Pipeline, StaticSource

    modifier = CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
    )
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(frame.atoms)))
    pipeline.modifiers.append(modifier)
    output = pipeline.compute()
    structure_types = np.asarray(output.particles["Structure Type"], dtype=np.int32)
    mapping = {0: "unknown", 1: "fcc", 2: "hcp", 3: "bcc", 4: "ico"}
    counts = Counter(mapping.get(int(value), "unknown") for value in structure_types)
    label, count = counts.most_common(1)[0]
    return label if count / len(structure_types) >= 0.5 else "unknown"


def _evidence_record(value) -> dict[str, object]:
    local = None
    if value.local_evidence_evaluated:
        local = {
            "support_fraction": value.local_support_fraction,
            "unknown_fraction": value.local_unknown_fraction,
            "margin_median": value.local_margin_median,
            "distance_median": value.local_distance_median,
        }
    return {
        "label": value.label,
        "candidate": value.candidate,
        "confidence_state": value.confidence_state,
        "structure_distance_ratio": value.structure_distance_ratio,
        "forest_probability": value.forest_probability,
        "translational_order_score": value.translational_order_score,
        "translational_order_limit": value.translational_order_limit,
        "cna_phase_fractions": value.cna_phase_fractions,
        "local": local,
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
    refinement_by_composition: dict[
        str, dict[tuple[tuple[str, int], ...], dict[str, object]]
    ] = {"l12": {}, "laves": {}}
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
            bucket = refinement_by_composition[candidate].setdefault(
                key,
                {
                    "eligible_structures": 0,
                    "confirmed_structures": 0,
                    "eligible_atoms": 0,
                    "local_match_atoms": 0.0,
                    "labels": Counter(),
                },
            )
            bucket["eligible_structures"] += 1
            bucket["eligible_atoms"] += len(atoms)
            bucket["local_match_atoms"] += len(atoms) * result.joint_match_fraction
            bucket["labels"][result.label] += 1
            if result.confirmed:
                refinement_confirmed[candidate].append(index)
                bucket["confirmed_structures"] += 1
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
    phase_predictions, geometry_evidence = model.predict_many_with_evidence(sketches)
    phase_seconds = time.perf_counter() - phase_started

    ptm_started = time.perf_counter()
    ptm_cutoffs = (0.10, 0.12, 0.15, 0.20)
    ptm_predictions_by_cutoff = {
        cutoff: [
            _ptm_prediction(
                Frame(atoms, "unknown", "unknown", "real", str(index)),
                rmsd_cutoff=cutoff,
            )
            for _, index, atoms in rows
        ]
        for cutoff in ptm_cutoffs
    }
    ptm_seconds = time.perf_counter() - ptm_started

    cna_started = time.perf_counter()
    cna_geometry = [
        _cna_prediction(Frame(atoms, "unknown", "unknown", "real", str(index)))
        for _, index, atoms in rows
    ]
    cna_seconds = time.perf_counter() - cna_started

    refinement_summary: dict[str, dict[str, object]] = {}
    for candidate in ("l12", "laves"):
        refinement_summary[candidate] = {
            "eligible_structures": refinement_eligible[candidate],
            "labels": dict(sorted(refinement_labels[candidate].items())),
            "confirmed_structures": len(refinement_confirmed[candidate]),
            "confirmed_indices": refinement_confirmed[candidate],
        }

    phase_geometry = [geometry for geometry, _ in phase_predictions]
    candidate_geometry = [value.candidate for value in geometry_evidence]
    confidence_states = [value.confidence_state for value in geometry_evidence]
    phase_labels = [_phase_label(*prediction) for prediction in phase_predictions]
    ptm_predictions = ptm_predictions_by_cutoff[0.10]
    ptm_geometry = [geometry for geometry, _ in ptm_predictions]
    ptm_labels = [_phase_label(*prediction) for prediction in ptm_predictions]
    geometry_joint = Counter(zip(phase_geometry, ptm_geometry))

    default_double_reject = [
        index
        for index, (value, ptm, cna) in enumerate(
            zip(geometry_evidence, ptm_geometry, cna_geometry)
        )
        if value.confidence_state in {"matched", "low_local_support"}
        and ptm == "unknown"
        and cna == "unknown"
    ]
    tolerance_sweep: dict[str, dict[str, int]] = {}
    for cutoff, predictions in ptm_predictions_by_cutoff.items():
        relaxed_geometry = [geometry for geometry, _ in predictions]
        comparison = Counter()
        for row in default_double_reject:
            label = relaxed_geometry[row]
            candidate = candidate_geometry[row]
            if label == candidate:
                comparison["same_as_candidate"] += 1
            elif label == "unknown":
                comparison["unknown"] += 1
            else:
                comparison[f"different:{label}"] += 1
        tolerance_sweep[f"{cutoff:.2f}"] = dict(sorted(comparison.items()))

    composition_rows = []
    cursor = 0
    for key in sorted(selected):
        count = len(selected[key])
        stop = cursor + count
        candidate_local_phases = {}
        for candidate in ("l12", "laves"):
            bucket = refinement_by_composition[candidate].get(key)
            if bucket is None:
                continue
            eligible_atoms = int(bucket["eligible_atoms"])
            candidate_local_phases[candidate] = {
                "eligible_structures": int(bucket["eligible_structures"]),
                "confirmed_structures": int(bucket["confirmed_structures"]),
                "labels": dict(sorted(bucket["labels"].items())),
                "eligible_atoms": eligible_atoms,
                "local_match_fraction": (
                    float(bucket["local_match_atoms"]) / eligible_atoms
                ),
            }
        composition_rows.append(
            {
                **_composition_record(key),
                "dataset_structures": populations[key],
                "sample_structures": count,
                "sample_indices": [index for index, _ in sorted(selected[key])],
                "phase_sketch": _fractions(phase_labels[cursor:stop]),
                "ptm": _fractions(ptm_labels[cursor:stop]),
                "cna_local_fractions": {
                    label: float(
                        np.mean(
                            [
                                value.cna_phase_fractions[label]
                                for value in geometry_evidence[cursor:stop]
                            ]
                        )
                    )
                    for label in ("fcc", "hcp", "bcc", "other_or_unresolved")
                },
                "candidate_local_phases": candidate_local_phases,
                "structures": [
                    {
                        "index": index,
                        "phase_sketch": phase_labels[cursor + offset],
                        "geometry_evidence": _evidence_record(
                            geometry_evidence[cursor + offset]
                        ),
                        "ptm": ptm_labels[cursor + offset],
                        "cna": cna_geometry[cursor + offset],
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
                    "cna_local_weighted": Counter(),
                    "candidate_local_weighted": {
                        "l12": Counter(),
                        "laves": Counter(),
                    },
                    "estimated_atoms": 0,
                },
            )
            population = int(row["dataset_structures"])
            group["dataset_structures"] += population
            for label, fraction in row["phase_sketch"].items():
                group["phase_sketch_weighted"][label] += population * fraction
            for label, fraction in row["ptm"].items():
                group["ptm_weighted"][label] += population * fraction
            atom_count = sum(int(value) for value in row["counts"].values())
            estimated_atoms = population * atom_count
            group["estimated_atoms"] += estimated_atoms
            for label, fraction in row["cna_local_fractions"].items():
                group["cna_local_weighted"][label] += (
                    estimated_atoms * fraction
                )
            for candidate, values in row["candidate_local_phases"].items():
                candidate_bucket = group["candidate_local_weighted"][candidate]
                candidate_bucket["eligible_structures"] += values[
                    "eligible_structures"
                ]
                candidate_bucket["confirmed_structures"] += values[
                    "confirmed_structures"
                ]
                candidate_bucket["eligible_atoms"] += values["eligible_atoms"]
                candidate_bucket["local_match_atoms"] += (
                    values["eligible_atoms"] * values["local_match_fraction"]
                )
        output_rows = []
        for atomic_percent, group in sorted(grouped.items()):
            population = int(group["dataset_structures"])
            estimated_atoms = int(group["estimated_atoms"])
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
                    "cna_local_estimated_fraction": {
                        label: value / estimated_atoms
                        for label, value in sorted(
                            group["cna_local_weighted"].items()
                        )
                    },
                    "candidate_local_phase_estimates": {
                        candidate: {
                            "eligible_structures": int(values["eligible_structures"]),
                            "confirmed_structures": int(values["confirmed_structures"]),
                            "eligible_atoms": int(values["eligible_atoms"]),
                            "local_match_fraction": (
                                values["local_match_atoms"] / values["eligible_atoms"]
                            ),
                        }
                        for candidate, values in group[
                            "candidate_local_weighted"
                        ].items()
                        if values["eligible_atoms"]
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
            "candidate_geometry": dict(sorted(Counter(candidate_geometry).items())),
            "confidence_state": dict(sorted(Counter(confidence_states).items())),
            "phase": dict(sorted(Counter(phase_labels).items())),
            "seconds": phase_seconds,
        },
        "ptm": {
            "geometry": dict(sorted(Counter(ptm_geometry).items())),
            "phase": dict(sorted(Counter(ptm_labels).items())),
            "seconds": ptm_seconds,
            "rmsd_cutoff": 0.10,
            "tolerance_sweep_on_default_double_reject": tolerance_sweep,
        },
        "cna": {
            "geometry": dict(sorted(Counter(cna_geometry).items())),
            "seconds": cna_seconds,
        },
        "agreement": {
            "geometry": float(np.mean(np.asarray(phase_geometry) == np.asarray(ptm_geometry))),
            "phase": float(np.mean(np.asarray(phase_labels) == np.asarray(ptm_labels))),
            "geometry_joint": {
                f"PhaseSketch={left}|PTM={right}": count
                for (left, right), count in sorted(geometry_joint.items())
            },
            "default_ptm_and_cna_reject_candidate": {
                "structures": len(default_double_reject),
                "indices": [rows[row][1] for row in default_double_reject],
            },
            "low_local_support": {
                "structures": sum(
                    state == "low_local_support" for state in confidence_states
                ),
                "indices": [
                    rows[row][1]
                    for row, state in enumerate(confidence_states)
                    if state == "low_local_support"
                ],
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
