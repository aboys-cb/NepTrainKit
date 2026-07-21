#!/usr/bin/env python3
"""Compare global and composition-stratified FPS on a labelled XYZ dataset.

This validation does not require a trained NEP model.  It uses two transparent
structure proxies:

* pooled: composition, cell shape, all-pair radial histogram, and globally
  pooled spin statistics;
* local: species-pair radial histograms and per-species spin statistics in
  addition to the pooled channels.

All selectors receive the same budget.  Their selected labels train the same
inverse-distance k-nearest-neighbour regressor, which is evaluated on held-out
frames.  This measures whether the selected structures cover distinct labelled
behaviour; it is not a substitute for a full NEP retraining A/B.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io import iread
from scipy.spatial import cKDTree


Array = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]
ELEMENTS = ("V", "Co", "Ni")
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
PAIR_TYPES = tuple((left, right) for left in range(3) for right in range(left, 3))
RADIAL_EDGES = np.linspace(0.0, 7.5, 25)
TARGET_NAMES = (
    "energy_per_atom",
    "force_rms",
    "force_max",
    "mforce_rms",
    "mforce_max",
    "virial_norm_per_atom",
)


@dataclass(frozen=True)
class DatasetFeatures:
    """Features, labels, and exact-composition strata for every XYZ frame."""

    pooled: Array
    local: Array
    targets: Array
    compositions: IndexArray
    composition_keys: tuple[tuple[int, int, int], ...]
    dilute_mask: npt.NDArray[np.bool_]
    feature_source: str


@dataclass(frozen=True)
class Evaluation:
    """Metrics for one selector on one held-out split."""

    normalized_rmse: float
    dilute_normalized_rmse: float
    target_rmse: Array
    selected_strata: int
    local_radius_p95: float


def farthest_point_sampling(points: Array, n_samples: int) -> IndexArray:
    """Greedy FPS matching NepTrainKit's fixed-count, zero-threshold path."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0 or n_samples <= 0:
        return np.empty(0, dtype=np.int64)
    selected = [0]
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    selected_mask[0] = True
    min_distances = np.linalg.norm(points - points[0], axis=1)
    min_distances[0] = -np.inf
    while len(selected) < min(int(n_samples), points.shape[0]):
        current = int(np.argmax(min_distances))
        if not np.isfinite(min_distances[current]):
            break
        selected.append(current)
        selected_mask[current] = True
        min_distances = np.minimum(
            min_distances,
            np.linalg.norm(points - points[current], axis=1),
        )
        min_distances[selected_mask] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def normalized_histogram(values: Array, edges: Array) -> Array:
    """Return a unit-sum histogram, or zeros when a pair type is absent."""
    counts = np.histogram(values, bins=edges)[0].astype(float)
    total = float(np.sum(counts))
    return counts / total if total > 0.0 else counts


def spin_statistics(spins: Array) -> Array:
    """Return mean and standard deviation of spin xyz and spin magnitude."""
    if spins.shape[0] == 0:
        return np.zeros(8, dtype=float)
    augmented = np.column_stack((spins, np.linalg.norm(spins, axis=1)))
    return np.concatenate((np.mean(augmented, axis=0), np.std(augmented, axis=0)))


def frame_features(atoms: Atoms) -> tuple[Array, Array, Array, tuple[int, int, int], bool]:
    """Extract pooled/local structural proxies and labelled target summaries."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    counts = tuple(int(np.sum(symbols == element)) for element in ELEMENTS)
    if sum(counts) != len(atoms):
        unknown = sorted(set(symbols.tolist()) - set(ELEMENTS))
        raise ValueError(f"unsupported elements in XYZ: {unknown}")
    fractions = np.asarray(counts, dtype=float) / float(len(atoms))

    cell = np.asarray(atoms.cell.array, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    volume_per_atom = float(atoms.get_volume() / len(atoms))
    scale = max(volume_per_atom, 1.0e-12) ** (1.0 / 3.0)
    normalized_lengths = lengths / scale
    cosines = np.asarray(
        [
            np.dot(cell[1], cell[2]) / max(lengths[1] * lengths[2], 1.0e-12),
            np.dot(cell[0], cell[2]) / max(lengths[0] * lengths[2], 1.0e-12),
            np.dot(cell[0], cell[1]) / max(lengths[0] * lengths[1], 1.0e-12),
        ]
    )
    cell_features = np.concatenate(([volume_per_atom], normalized_lengths, cosines))

    upper = np.triu_indices(len(atoms), k=1)
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)[upper]
    left_symbols = symbols[upper[0]]
    right_symbols = symbols[upper[1]]
    global_radial = normalized_histogram(distances, RADIAL_EDGES)
    pair_radial: list[Array] = []
    for left_index, right_index in PAIR_TYPES:
        left = ELEMENTS[left_index]
        right = ELEMENTS[right_index]
        pair_mask = ((left_symbols == left) & (right_symbols == right)) | (
            (left_symbols == right) & (right_symbols == left)
        )
        pair_radial.append(normalized_histogram(distances[pair_mask], RADIAL_EDGES))

    spins = np.asarray(atoms.arrays["spin"], dtype=float)
    global_spin = spin_statistics(spins)
    species_spin = [spin_statistics(spins[symbols == element]) for element in ELEMENTS]

    pooled = np.concatenate((fractions, cell_features, global_radial, global_spin))
    local = np.concatenate((pooled, *pair_radial, *species_spin))

    forces = np.asarray(atoms.arrays["force"], dtype=float)
    magnetic_forces = np.asarray(atoms.arrays["force_mag"], dtype=float)
    force_norms = np.linalg.norm(forces, axis=1)
    magnetic_force_norms = np.linalg.norm(magnetic_forces, axis=1)
    virial = np.asarray(atoms.info["virial"], dtype=float)
    targets = np.asarray(
        [
            float(atoms.get_potential_energy() / len(atoms)),
            float(np.sqrt(np.mean(forces**2))),
            float(np.max(force_norms)),
            float(np.sqrt(np.mean(magnetic_forces**2))),
            float(np.max(magnetic_force_norms)),
            float(np.linalg.norm(virial) / len(atoms)),
        ]
    )
    nonzero_counts = [count for count in counts if count > 0]
    dilute = len(nonzero_counts) > 1 and min(nonzero_counts) <= 3
    return pooled, local, targets, counts, dilute


def load_xyz(path: Path) -> DatasetFeatures:
    """Stream an extended XYZ file and extract the validation arrays."""
    pooled_rows: list[Array] = []
    local_rows: list[Array] = []
    target_rows: list[Array] = []
    keys: list[tuple[int, int, int]] = []
    dilute: list[bool] = []
    for atoms in iread(path, index=":"):
        pooled, local, targets, key, is_dilute = frame_features(atoms)
        pooled_rows.append(pooled)
        local_rows.append(local)
        target_rows.append(targets)
        keys.append(key)
        dilute.append(is_dilute)
    if not pooled_rows:
        raise ValueError(f"XYZ contains no frames: {path}")
    unique_keys = tuple(sorted(set(keys)))
    key_to_index = {key: index for index, key in enumerate(unique_keys)}
    compositions = np.asarray([key_to_index[key] for key in keys], dtype=np.int64)
    return DatasetFeatures(
        pooled=np.vstack(pooled_rows),
        local=np.vstack(local_rows),
        targets=np.vstack(target_rows),
        compositions=compositions,
        composition_keys=unique_keys,
        dilute_mask=np.asarray(dilute, dtype=bool),
        feature_source="structural proxy",
    )


def load_nep_descriptor_features(
    xyz_path: Path,
    nep_path: Path,
    data: DatasetFeatures,
    pooled_descriptor_path: Path | None = None,
    chunk_frames: int = 512,
) -> DatasetFeatures:
    """Replace proxy features with actual per-atom NEP descriptors.

    The pooled representation exactly follows the current FPS card contract:
    arithmetic mean over every atom in a structure.  The local representation
    concatenates per-element descriptor means and standard deviations.
    """
    try:
        from nep_adapters import NEPCalculator
    except ImportError as error:
        raise RuntimeError(
            "--nep requires a Python environment with nep-adapters installed"
        ) from error

    pooled_rows: list[Array] = []
    local_rows: list[Array] = []
    chunk: list[Atoms] = []

    def process_chunk(calculator: NEPCalculator, structures: list[Atoms]) -> None:
        if not structures:
            return
        per_atom = np.asarray(calculator.predict_descriptors(structures), dtype=float)
        offset = 0
        for atoms in structures:
            block = per_atom[offset : offset + len(atoms)]
            offset += len(atoms)
            symbols = np.asarray(atoms.get_chemical_symbols())
            pooled_rows.append(np.mean(block, axis=0))
            species_blocks: list[Array] = []
            for element in ELEMENTS:
                selected = block[symbols == element]
                if selected.shape[0] == 0:
                    species_blocks.extend(
                        (np.zeros(block.shape[1], dtype=float), np.zeros(block.shape[1], dtype=float))
                    )
                else:
                    species_blocks.extend((np.mean(selected, axis=0), np.std(selected, axis=0)))
            local_rows.append(np.concatenate(species_blocks))

    with NEPCalculator(nep_path, backend="cpu") as calculator:
        for atoms in iread(xyz_path, index=":"):
            chunk.append(atoms)
            if len(chunk) >= chunk_frames:
                process_chunk(calculator, chunk)
                chunk.clear()
        process_chunk(calculator, chunk)

    if len(pooled_rows) != data.targets.shape[0]:
        raise RuntimeError("descriptor frame count does not match XYZ labels")
    calculated_pooled = np.vstack(pooled_rows)
    source_detail = f"calculated from {nep_path}"
    if pooled_descriptor_path is not None:
        provided_pooled = np.loadtxt(pooled_descriptor_path, dtype=float, ndmin=2)
        if provided_pooled.shape != calculated_pooled.shape:
            raise ValueError(
                f"provided descriptor shape {provided_pooled.shape} does not match "
                f"calculated shape {calculated_pooled.shape}"
            )
        max_abs_difference = float(np.max(np.abs(provided_pooled - calculated_pooled)))
        pooled = provided_pooled
        source_detail = (
            f"pooled={pooled_descriptor_path}, per-atom={nep_path}, "
            f"mean-check max_abs={max_abs_difference:.3g}"
        )
    else:
        pooled = calculated_pooled
    return DatasetFeatures(
        pooled=pooled,
        local=np.vstack(local_rows),
        targets=data.targets,
        compositions=data.compositions,
        composition_keys=data.composition_keys,
        dilute_mask=data.dilute_mask,
        feature_source=f"NEP descriptors ({source_detail})",
    )


def fit_feature_scaling(features: Array, indices: IndexArray) -> tuple[Array, Array]:
    """Fit safe standardization parameters on candidate structures only."""
    mean = np.mean(features[indices], axis=0)
    scale = np.std(features[indices], axis=0)
    scale[scale < 1.0e-12] = 1.0
    return mean, scale


def scale_feature_blocks(features: Array, mean: Array, scale: Array) -> Array:
    """Standardize features without allowing large blocks to dominate distance."""
    standardized = (features - mean) / scale
    result = standardized.copy()
    if result.shape[1] in {35, 210}:
        if result.shape[1] == 35:
            return result / np.sqrt(35)
        for cursor in range(0, result.shape[1], 35):
            result[:, cursor : cursor + 35] /= np.sqrt(35)
        return result
    # pooled blocks: composition(3), cell(7), radial(24), spin(8)
    boundaries = (0, 3, 10, 34, 42)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        result[:, start:stop] /= np.sqrt(stop - start)
    # local-only blocks: six pair histograms and three species-spin blocks
    cursor = 42
    while cursor < min(result.shape[1], 42 + 6 * 24):
        result[:, cursor : cursor + 24] /= np.sqrt(24)
        cursor += 24
    while cursor < result.shape[1]:
        result[:, cursor : cursor + 8] /= np.sqrt(8)
        cursor += 8
    return result


def stratified_split(
    compositions: IndexArray,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[IndexArray, IndexArray]:
    """Hold out frames inside every exact-composition stratum."""
    candidate: list[int] = []
    test: list[int] = []
    for group in np.unique(compositions):
        indices = np.flatnonzero(compositions == group)
        shuffled = rng.permutation(indices)
        n_test = max(3, int(round(test_fraction * indices.size)))
        n_test = min(n_test, indices.size - 3)
        test.extend(shuffled[:n_test].tolist())
        candidate.extend(shuffled[n_test:].tolist())
    return np.asarray(candidate, dtype=np.int64), np.asarray(test, dtype=np.int64)


def allocate_quotas(compositions: IndexArray, candidates: IndexArray, budget: int) -> dict[int, int]:
    """Give every composition a floor, then allocate the remainder by size."""
    groups = np.unique(compositions)
    counts = {int(group): int(np.sum(compositions[candidates] == group)) for group in groups}
    floor = min(4, min(counts.values()))
    if budget < floor * len(groups):
        raise ValueError(f"budget must be at least {floor * len(groups)} for stratified coverage")
    quotas = {group: floor for group in counts}
    remaining = budget - floor * len(groups)
    capacities = {group: counts[group] - floor for group in counts}
    total_capacity = sum(capacities.values())
    raw = {
        group: (remaining * capacities[group] / total_capacity if total_capacity else 0.0)
        for group in counts
    }
    for group in counts:
        addition = min(capacities[group], int(np.floor(raw[group])))
        quotas[group] += addition
    left = budget - sum(quotas.values())
    order = sorted(counts, key=lambda group: (raw[group] - np.floor(raw[group]), counts[group]), reverse=True)
    for group in order:
        if left == 0:
            break
        if quotas[group] < counts[group]:
            quotas[group] += 1
            left -= 1
    if sum(quotas.values()) != budget:
        raise RuntimeError("failed to allocate the complete sampling budget")
    return quotas


def stratified_select(
    features: Array,
    compositions: IndexArray,
    candidates: IndexArray,
    quotas: dict[int, int],
    *,
    random: bool,
    rng: np.random.Generator,
) -> IndexArray:
    """Select within every exact-composition stratum."""
    selected: list[int] = []
    for group, quota in quotas.items():
        group_indices = candidates[compositions[candidates] == group]
        if random:
            chosen = rng.choice(group_indices, quota, replace=False)
        else:
            local = farthest_point_sampling(features[group_indices], quota)
            chosen = group_indices[local]
        selected.extend(chosen.tolist())
    return np.asarray(selected, dtype=np.int64)


def select_methods(
    data: DatasetFeatures,
    candidates: IndexArray,
    pooled_raw: Array,
    pooled_scaled: Array,
    local: Array,
    budget: int,
    rng: np.random.Generator,
) -> dict[str, IndexArray]:
    """Run equal-budget global, stratified, and random selectors."""
    quotas = allocate_quotas(data.compositions, candidates, budget)
    return {
        "global raw mean FPS": candidates[farthest_point_sampling(pooled_raw[candidates], budget)],
        "global scaled mean FPS": candidates[farthest_point_sampling(pooled_scaled[candidates], budget)],
        "stratified scaled mean FPS": stratified_select(
            pooled_scaled, data.compositions, candidates, quotas, random=False, rng=rng
        ),
        "stratified random": stratified_select(
            local, data.compositions, candidates, quotas, random=True, rng=rng
        ),
        "stratified local FPS": stratified_select(
            local, data.compositions, candidates, quotas, random=False, rng=rng
        ),
    }


def knn_predict(train_x: Array, train_y: Array, test_x: Array, neighbors: int = 5) -> tuple[Array, Array]:
    """Predict labels with the same inverse-distance kNN model for every selector."""
    tree = cKDTree(train_x)
    distances, indices = tree.query(test_x, k=min(neighbors, train_x.shape[0]))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1.0e-12)
    weights /= np.sum(weights, axis=1, keepdims=True)
    prediction = np.sum(train_y[indices] * weights[:, :, None], axis=1)
    return prediction, distances[:, 0]


def evaluate(
    data: DatasetFeatures,
    features: Array,
    selected: IndexArray,
    test: IndexArray,
    target_scale: Array,
) -> Evaluation:
    """Evaluate downstream label interpolation and held-out local coverage."""
    prediction, nearest = knn_predict(features[selected], data.targets[selected], features[test])
    errors = prediction - data.targets[test]
    normalized_squared = (errors / target_scale) ** 2
    dilute = data.dilute_mask[test]
    dilute_rmse = float(np.sqrt(np.mean(normalized_squared[dilute]))) if np.any(dilute) else float("nan")
    return Evaluation(
        normalized_rmse=float(np.sqrt(np.mean(normalized_squared))),
        dilute_normalized_rmse=dilute_rmse,
        target_rmse=np.sqrt(np.mean(errors**2, axis=0)),
        selected_strata=int(np.unique(data.compositions[selected]).size),
        local_radius_p95=float(np.quantile(nearest, 0.95)),
    )


def run_split(data: DatasetFeatures, seed: int, budget: int) -> dict[str, Evaluation]:
    """Run one held-out split and all equal-budget selectors."""
    rng = np.random.default_rng(seed)
    candidates, test = stratified_split(data.compositions, 0.2, rng)
    pooled_mean, pooled_scale = fit_feature_scaling(data.pooled, candidates)
    local_mean, local_scale = fit_feature_scaling(data.local, candidates)
    pooled = scale_feature_blocks(data.pooled, pooled_mean, pooled_scale)
    local = scale_feature_blocks(data.local, local_mean, local_scale)
    selections = select_methods(data, candidates, data.pooled, pooled, local, budget, rng)
    target_scale = np.std(data.targets[candidates], axis=0)
    target_scale[target_scale < 1.0e-12] = 1.0
    return {
        name: evaluate(data, local, selected, test, target_scale)
        for name, selected in selections.items()
    }


def median_evaluation(results: list[dict[str, Evaluation]], method: str) -> Evaluation:
    """Aggregate a method over repeated held-out splits."""
    normalized = np.median([result[method].normalized_rmse for result in results])
    dilute = np.median([result[method].dilute_normalized_rmse for result in results])
    target_rmse = np.median([result[method].target_rmse for result in results], axis=0)
    strata = int(round(np.median([result[method].selected_strata for result in results])))
    radius = np.median([result[method].local_radius_p95 for result in results])
    return Evaluation(float(normalized), float(dilute), target_rmse, strata, float(radius))


def print_report(path: Path, data: DatasetFeatures, budget: int, results: list[dict[str, Evaluation]]) -> None:
    """Print dataset facts and median validation results."""
    composition_counts = Counter(data.compositions.tolist())
    print(f"XYZ: {path}")
    print(
        f"frames={data.targets.shape[0]}, exact_compositions={len(composition_counts)}, "
        f"dilute_frames={int(np.sum(data.dilute_mask))}, budget={budget}, repeats={len(results)}"
    )
    print(f"features={data.feature_source}")
    print("The same species-aware kNN learner and held-out frames are used for every method.\n")
    print(f"{'method':<25} {'norm RMSE':>10} {'dilute RMSE':>12} {'strata':>8} {'local r95':>11}")
    print("-" * 72)
    for method in results[0]:
        metric = median_evaluation(results, method)
        print(
            f"{method:<25} {metric.normalized_rmse:10.4f} {metric.dilute_normalized_rmse:12.4f} "
            f"{metric.selected_strata:8d} {metric.local_radius_p95:11.4f}"
        )
    print("\nPer-target physical RMSE (median over splits):")
    print(f"{'method':<25} " + " ".join(f"{name:>18}" for name in TARGET_NAMES))
    for method in results[0]:
        metric = median_evaluation(results, method)
        print(f"{method:<25} " + " ".join(f"{value:18.6g}" for value in metric.target_rmse))

    baseline = median_evaluation(results, "global raw mean FPS")
    stratified = median_evaluation(results, "stratified local FPS")
    print("\nReductions from global raw mean FPS to stratified local FPS (positive is better):")
    print(f"  normalized RMSE: {1.0 - stratified.normalized_rmse / baseline.normalized_rmse:+.1%}")
    print(
        "  dilute-composition normalized RMSE: "
        f"{1.0 - stratified.dilute_normalized_rmse / baseline.dilute_normalized_rmse:+.1%}"
    )
    print(f"  local coverage radius p95: {1.0 - stratified.local_radius_p95 / baseline.local_radius_p95:+.1%}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--nep", type=Path)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--budget", type=int, default=330)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    """Load the real dataset and run repeated held-out A/B comparisons."""
    args = parse_args()
    if args.budget <= 0 or args.repeats <= 0:
        raise ValueError("budget and repeats must be positive")
    data = load_xyz(args.xyz)
    if args.descriptor is not None and args.nep is None:
        raise ValueError("--descriptor requires --nep so per-atom descriptor statistics can be calculated")
    if args.nep is not None:
        data = load_nep_descriptor_features(args.xyz, args.nep, data, args.descriptor)
    results = [run_split(data, args.seed + repeat, args.budget) for repeat in range(args.repeats)]
    print_report(args.xyz, data, args.budget, results)


if __name__ == "__main__":
    main()
