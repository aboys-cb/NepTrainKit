#!/usr/bin/env python
"""Falsification-oriented checks for L1_2 and Laves phase refinement.

Laves prototypes and parameters are taken from the AFLOW crystallographic
prototype encyclopedia entries LL0C (C14), 8YL7 (C15), and HV5V (C36).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict

import numpy as np
from ase import Atoms
from ase.build import bulk

from NepTrainKit.core.audit.phase_sketch import periodic_knn_vectors
from NepTrainKit.core.audit.phase_refinement import refine_l12, refine_laves
from tools.benchmark_phase_sketch import (
    Frame,
    _distort,
    _l12,
    _ptm_prediction,
    _sketch_frame,
    build_training_frames,
    fit_phase_sketch,
)


def _hexagonal_cell(a: float, c_over_a: float) -> np.ndarray:
    c = a * c_over_a
    return np.asarray(
        (
            (0.5 * a, -0.5 * np.sqrt(3.0) * a, 0.0),
            (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
            (0.0, 0.0, c),
        )
    )


def c14_mgzn2() -> Atoms:
    a, c_over_a, z2, x3 = 5.223, 1.64005, 0.06286, 0.830483
    basis = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5),
        (1 / 3, 2 / 3, z2),
        (2 / 3, 1 / 3, z2 + 0.5),
        (2 / 3, 1 / 3, -z2),
        (1 / 3, 2 / 3, 0.5 - z2),
        (x3, 2 * x3, 0.25),
        (-2 * x3, -x3, 0.25),
        (x3, -x3, 0.25),
        (-x3, -2 * x3, 0.75),
        (2 * x3, x3, 0.75),
        (-x3, x3, 0.75),
    )
    return Atoms(
        symbols=("Zn", "Zn") + ("Mg",) * 4 + ("Zn",) * 6,
        scaled_positions=np.mod(basis, 1.0),
        cell=_hexagonal_cell(a, c_over_a),
        pbc=True,
    )


def c15_mgcu2() -> Atoms:
    a = 7.02
    cell = 0.5 * a * np.asarray(((0, 1, 1), (1, 0, 1), (1, 1, 0)))
    basis = (
        (3 / 8, 3 / 8, 3 / 8),
        (5 / 8, 5 / 8, 5 / 8),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.5, 0.0),
        (0.5, 0.0, 0.0),
    )
    return Atoms(
        symbols=("Mg", "Mg") + ("Cu",) * 4,
        scaled_positions=basis,
        cell=cell,
        pbc=True,
    )


def c36_mgni2() -> Atoms:
    a, c_over_a = 4.824, 3.28068
    z1, z2, z3, x5 = 0.094, 0.84417, 0.12514, 0.16429
    basis = (
        (0.0, 0.0, z1),
        (0.0, 0.0, z1 + 0.5),
        (0.0, 0.0, -z1),
        (0.0, 0.0, 0.5 - z1),
        (1 / 3, 2 / 3, z2),
        (2 / 3, 1 / 3, z2 + 0.5),
        (2 / 3, 1 / 3, -z2),
        (1 / 3, 2 / 3, 0.5 - z2),
        (1 / 3, 2 / 3, z3),
        (2 / 3, 1 / 3, z3 + 0.5),
        (2 / 3, 1 / 3, -z3),
        (1 / 3, 2 / 3, 0.5 - z3),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (x5, 2 * x5, 0.25),
        (-2 * x5, -x5, 0.25),
        (x5, -x5, 0.25),
        (-x5, -2 * x5, 0.75),
        (2 * x5, x5, 0.75),
        (-x5, x5, 0.75),
    )
    return Atoms(
        symbols=("Mg",) * 8 + ("Ni",) * 16,
        scaled_positions=np.mod(basis, 1.0),
        cell=_hexagonal_cell(a, c_over_a),
        pbc=True,
    )


def _coordination_check(atoms: Atoms, large: str | None = None) -> dict[str, object]:
    vectors, indices, valid = periodic_knn_vectors(
        atoms.positions, atoms.cell.array, atoms.pbc, neighbors=20
    )
    symbols = np.asarray(atoms.get_chemical_symbols())
    composition = Counter(atoms.get_chemical_symbols())
    if large is None:
        if len(composition) != 2:
            raise ValueError("the prototype check requires exactly two species")
        large = min(composition, key=composition.get)
    mismatches = 0
    gap_ratios = []
    observed = Counter()
    normalized_b_csp = []
    for atom in range(len(atoms)):
        distances = np.linalg.norm(vectors[atom, valid[atom]], axis=1)
        order = np.argsort(distances, kind="stable")
        expected_count = 16 if symbols[atom] == large else 12
        neighbor_symbols = symbols[indices[atom, valid[atom]][order[:expected_count]]]
        signature = (int(np.sum(neighbor_symbols == large)), expected_count - int(np.sum(neighbor_symbols == large)))
        observed[(str(symbols[atom]), signature)] += 1
        expected_signature = (4, 12) if symbols[atom] == large else (6, 6)
        mismatches += signature != expected_signature
        gap_ratios.append(float(distances[order[expected_count]] / distances[order[expected_count - 1]]))
        if symbols[atom] != large:
            b_neighbor_mask = symbols[indices[atom, valid[atom]]] != large
            b_vectors = vectors[atom, valid[atom]][b_neighbor_mask]
            b_distances = np.linalg.norm(b_vectors, axis=1)
            b_vectors = b_vectors[np.argsort(b_distances, kind="stable")[:6]]
            normalized_b_csp.append(_normalized_csp(b_vectors))
    csp_values = np.asarray(normalized_b_csp)
    return {
        "atoms": len(atoms),
        "stoichiometry": dict(composition),
        "coordination_mismatches": mismatches,
        "coordination_match_fraction": 1.0 - mismatches / len(atoms),
        "minimum_shell_gap_ratio": min(gap_ratios),
        "b2_fraction": float(np.mean(csp_values > 0.8)),
        "normalized_b_csp_quantiles": {
            str(percentile): float(np.percentile(csp_values, percentile))
            for percentile in (0, 25, 50, 75, 100)
        },
        "observed_coordination": {
            f"{center}:{same_large}+{other}B": count
            for (center, (same_large, other)), count in sorted(observed.items())
        },
    }


def _random_ab2(atoms: Atoms, seed: int) -> Atoms:
    if len(atoms) % 3:
        raise ValueError("AB2 control atom count must be divisible by three")
    result = atoms.copy()
    symbols = np.full(len(result), "Cu", dtype=object)
    symbols[np.random.default_rng(seed).choice(len(result), len(result) // 3, replace=False)] = "Mg"
    result.set_chemical_symbols(symbols.tolist())
    return result


def _negative_controls(seed: int = 1700) -> dict[str, Atoms]:
    return {
        "fcc_random_ab2": _random_ab2(
            bulk("Cu", "fcc", a=3.62, cubic=True).repeat((3, 3, 3)), seed
        ),
        "bcc_random_ab2": _random_ab2(
            bulk("Fe", "bcc", a=2.88, cubic=True).repeat((3, 3, 3)), seed + 1
        ),
        "hcp_random_ab2": _random_ab2(
            bulk("Ti", "hcp", a=2.95, c=4.68, orthorhombic=True).repeat((3, 3, 3)),
            seed + 2,
        ),
        "diamond_random_ab2": _random_ab2(
            bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)), seed + 3
        ),
    }


def _l12_with_antisites(swap_fraction: float, seed: int) -> Atoms:
    atoms = _l12()
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    minority = np.flatnonzero(symbols == "Au")
    majority = np.flatnonzero(symbols == "Cu")
    pair_count = min(
        len(minority),
        len(majority),
        int(round(0.5 * swap_fraction * len(atoms))),
    )
    rng = np.random.default_rng(seed)
    if pair_count:
        minority_sites = rng.choice(minority, pair_count, replace=False)
        majority_sites = rng.choice(majority, pair_count, replace=False)
        symbols[minority_sites] = "Cu"
        symbols[majority_sites] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def _normalized_csp(vectors: np.ndarray) -> float:
    """Return the minimum disjoint-pair CSP normalized by mean bond length."""
    if len(vectors) % 2 or not len(vectors):
        raise ValueError("CSP requires a positive even number of vectors")

    def minimum_cost(remaining: tuple[int, ...]) -> float:
        if not remaining:
            return 0.0
        first = remaining[0]
        return min(
            float(np.dot(vectors[first] + vectors[other], vectors[first] + vectors[other]))
            + minimum_cost(remaining[1:index] + remaining[index + 1 :])
            for index, other in enumerate(remaining[1:], start=1)
        )

    mean_squared_distance = float(np.mean(np.sum(vectors * vectors, axis=1)))
    return minimum_cost(tuple(range(len(vectors)))) / mean_squared_distance


def run_validation() -> dict[str, object]:
    prototypes = {
        "laves_c14": c14_mgzn2().repeat((3, 3, 2)),
        "laves_c15": c15_mgcu2().repeat((3, 3, 3)),
        "laves_c36": c36_mgni2().repeat((3, 3, 2)),
    }
    model, _ = fit_phase_sketch(build_training_frames())
    result: dict[str, object] = {"prototypes": {}, "open_set": {}, "negative_controls": {}}
    for label, atoms in prototypes.items():
        result["prototypes"][label] = _coordination_check(atoms)
        predictions = []
        for replica, (strain, noise, shear) in enumerate(
            ((0.0, 0.0, 0.0), (0.03, 0.02, 0.03), (0.06, 0.05, 0.08), (0.09, 0.08, 0.12))
        ):
            distorted = _distort(
                atoms,
                8000 + replica,
                strain=strain,
                noise=noise,
                shear=shear,
            )
            frame = Frame(distorted, "unknown", "unknown", "validation", label)
            candidate_check = _coordination_check(distorted)
            predictions.append(
                {
                    "condition": replica,
                    "laves_candidate": {
                        "coordination_match_fraction": candidate_check[
                            "coordination_match_fraction"
                        ],
                        "minimum_shell_gap_ratio": candidate_check[
                            "minimum_shell_gap_ratio"
                        ],
                        "b2_fraction": candidate_check["b2_fraction"],
                    },
                    "refinement": asdict(
                        refine_laves(
                            distorted.positions,
                            distorted.cell.array,
                            distorted.pbc,
                            distorted.numbers,
                        )
                    ),
                    "phase_sketch": model.predict(_sketch_frame(frame)),
                    "ptm": _ptm_prediction(frame),
                }
            )
        result["open_set"][label] = predictions

    for label, atoms in _negative_controls().items():
        checks = []
        for replica, (strain, noise, shear) in enumerate(
            ((0.0, 0.0, 0.0), (0.03, 0.02, 0.03), (0.06, 0.05, 0.08))
        ):
            distorted = _distort(
                atoms,
                10000 + 17 * replica,
                strain=strain,
                noise=noise,
                shear=shear,
            )
            candidate_check = _coordination_check(distorted, large="Mg")
            checks.append(
                {
                    "condition": replica,
                    "coordination_match_fraction": candidate_check[
                        "coordination_match_fraction"
                    ],
                    "minimum_shell_gap_ratio": candidate_check["minimum_shell_gap_ratio"],
                    "b2_fraction": candidate_check["b2_fraction"],
                    "refinement": asdict(
                        refine_laves(
                            distorted.positions,
                            distorted.cell.array,
                            distorted.pbc,
                            distorted.numbers,
                        )
                    ),
                }
            )
        result["negative_controls"][label] = checks

    l12_results = []
    for replica, (strain, noise, shear) in enumerate(
        ((0.0, 0.0, 0.0), (0.03, 0.02, 0.03), (0.06, 0.05, 0.08), (0.09, 0.08, 0.12))
    ):
        atoms = _distort(_l12(), 9000 + replica, strain=strain, noise=noise, shear=shear)
        frame = Frame(atoms, "fcc", "l12", "validation", "l12")
        l12_results.append(
            {
                "condition": replica,
                "refinement": asdict(
                    refine_l12(atoms.positions, atoms.cell.array, atoms.pbc, atoms.numbers)
                ),
                "phase_sketch": model.predict(_sketch_frame(frame)),
                "ptm": _ptm_prediction(frame),
            }
        )
    result["l12"] = l12_results
    l12_antisites = []
    for swap_fraction in (0.0, 0.02, 0.06, 0.12, 0.24, 0.48):
        atoms = _l12_with_antisites(swap_fraction, 11000 + int(100 * swap_fraction))
        changed = int(np.sum(np.asarray(atoms.get_chemical_symbols()) != np.asarray(_l12().get_chemical_symbols())))
        frame = Frame(atoms, "fcc", "l12", "antisite", "l12")
        l12_antisites.append(
            {
                "requested_swap_fraction": swap_fraction,
                "actual_swap_fraction": changed / len(atoms),
                "refinement": asdict(
                    refine_l12(atoms.positions, atoms.cell.array, atoms.pbc, atoms.numbers)
                ),
                "phase_sketch": model.predict(_sketch_frame(frame)),
                "ptm": _ptm_prediction(frame),
            }
        )
    result["l12_antisites"] = l12_antisites
    return result


if __name__ == "__main__":
    print(json.dumps(run_validation(), ensure_ascii=False, indent=2))
