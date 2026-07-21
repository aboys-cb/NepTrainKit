#!/usr/bin/env python3
"""Controlled demo of a global-FPS failure mode in complex compositions.

The synthetic structures contain a majority-host environment ``u`` and a
minority-species environment ``v``.  The minority species occupies 4% of the
atoms, so global mean pooling reduces its contribution from ``v`` to
``0.04 * v``.  The downstream target, however, is deliberately sensitive to
that minority environment.

This is a falsifiable counterexample, not a claim that stratified FPS wins on
every real dataset.  It compares four equal-budget selectors and evaluates all
of them with the same kernel-ridge learner on an independent balanced test set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]

COMPOSITION_COUNTS = (1800, 900, 450, 240, 120, 60)
N_COMPOSITIONS = len(COMPOSITION_COUNTS)
MINORITY_FRACTION = 0.04
DEFAULT_BUDGET = 36


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic composition and local-environment coordinates."""

    composition: IndexArray
    host_environment: Array
    minority_environment: Array


@dataclass(frozen=True)
class Metrics:
    """Downstream and coverage metrics for one selector."""

    rmse: float
    minority_extreme_rmse: float
    local_coverage_radius_p95: float


def farthest_point_sampling(points: Array, n_samples: int) -> IndexArray:
    """Pure-NumPy greedy FPS matching NepTrainKit's zero-threshold path."""
    points = np.asarray(points, dtype=float)
    n_points = int(points.shape[0])
    if n_points == 0 or int(n_samples) <= 0:
        return np.empty(0, dtype=np.int64)

    selected = [0]
    selected_mask = np.zeros(n_points, dtype=bool)
    selected_mask[0] = True
    min_distances = np.linalg.norm(points - points[0], axis=1)
    min_distances[selected_mask] = -np.inf

    while len(selected) < min(int(n_samples), n_points):
        current = int(np.argmax(min_distances))
        if not np.isfinite(min_distances[current]):
            break
        selected.append(current)
        selected_mask[current] = True
        new_distances = np.linalg.norm(points - points[current], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_mask] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def make_candidate_pool(rng: np.random.Generator) -> SyntheticData:
    """Create an imbalanced six-composition candidate pool."""
    composition = np.concatenate(
        [np.full(count, group, dtype=np.int64) for group, count in enumerate(COMPOSITION_COUNTS)]
    )
    host = np.concatenate([rng.uniform(-1.0, 1.0, count) for count in COMPOSITION_COUNTS])
    minority = np.concatenate([rng.uniform(-1.0, 1.0, count) for count in COMPOSITION_COUNTS])
    return SyntheticData(composition, host, minority)


def make_balanced_test_set(rng: np.random.Generator, per_composition: int = 500) -> SyntheticData:
    """Create an independent test set with equal weight per composition."""
    composition = np.repeat(np.arange(N_COMPOSITIONS, dtype=np.int64), per_composition)
    size = int(composition.size)
    return SyntheticData(
        composition,
        rng.uniform(-1.0, 1.0, size),
        rng.uniform(-1.0, 1.0, size),
    )


def global_mean_features(data: SyntheticData) -> Array:
    """Mimic one vector per structure after all-atom mean pooling."""
    composition_axis = 8.0 * data.composition.astype(float)
    pooled_environment = data.host_environment + MINORITY_FRACTION * data.minority_environment
    return np.column_stack((composition_axis, pooled_environment))


def species_aware_features(data: SyntheticData) -> Array:
    """Keep majority and minority local environments as separate channels."""
    return np.column_stack((data.host_environment, data.minority_environment))


def target(data: SyntheticData) -> Array:
    """Synthetic label with a force-like sensitivity to the minority environment."""
    c = data.composition.astype(float)
    u = data.host_environment
    v = data.minority_environment
    return 0.2 * c + np.sin(np.pi * u) + 3.5 * v + 1.2 * u * v


def stratified_fps(data: SyntheticData, features: Array, budget: int) -> IndexArray:
    """Allocate an equal minimum quota, then run FPS within each composition."""
    if budget % N_COMPOSITIONS != 0:
        raise ValueError(f"budget must be divisible by {N_COMPOSITIONS}")
    quota = budget // N_COMPOSITIONS
    selected: list[int] = []
    for group in range(N_COMPOSITIONS):
        group_indices = np.flatnonzero(data.composition == group)
        local_indices = farthest_point_sampling(features[group_indices], quota)
        selected.extend(group_indices[local_indices].tolist())
    return np.asarray(selected, dtype=np.int64)


def stratified_random(data: SyntheticData, budget: int, rng: np.random.Generator) -> IndexArray:
    """Strong simple baseline: equal composition quota plus random selection."""
    if budget % N_COMPOSITIONS != 0:
        raise ValueError(f"budget must be divisible by {N_COMPOSITIONS}")
    quota = budget // N_COMPOSITIONS
    selected: list[int] = []
    for group in range(N_COMPOSITIONS):
        group_indices = np.flatnonzero(data.composition == group)
        selected.extend(rng.choice(group_indices, quota, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def select_training_sets(
    candidates: SyntheticData,
    budget: int,
    rng: np.random.Generator,
) -> dict[str, IndexArray]:
    """Apply all equal-budget selection strategies."""
    mean_features = global_mean_features(candidates)
    local_features = species_aware_features(candidates)
    return {
        "global mean FPS": farthest_point_sampling(mean_features, budget),
        "stratified mean FPS": stratified_fps(candidates, mean_features, budget),
        "stratified random": stratified_random(candidates, budget, rng),
        "stratified local FPS": stratified_fps(candidates, local_features, budget),
    }


def rbf_kernel(left: Array, right: Array, gamma: float = 0.5) -> Array:
    """Return the radial-basis-function kernel matrix."""
    squared_distances = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * squared_distances)


def evaluate_selection(
    candidates: SyntheticData,
    test: SyntheticData,
    selected: IndexArray,
) -> Metrics:
    """Fit the same per-composition kernel ridge model and score a selection."""
    candidate_features = species_aware_features(candidates)
    test_features = species_aware_features(test)
    candidate_targets = target(candidates)
    test_targets = target(test)
    prediction = np.empty(test_targets.shape, dtype=float)
    nearest_distances = np.empty(test_targets.shape, dtype=float)

    for group in range(N_COMPOSITIONS):
        train_indices = selected[candidates.composition[selected] == group]
        if train_indices.size == 0:
            return Metrics(float("inf"), float("inf"), float("inf"))
        test_indices = np.flatnonzero(test.composition == group)
        train_x = candidate_features[train_indices]
        test_x = test_features[test_indices]
        kernel = rbf_kernel(train_x, train_x)
        coefficients = np.linalg.solve(
            kernel + 1.0e-5 * np.eye(train_indices.size),
            candidate_targets[train_indices],
        )
        prediction[test_indices] = rbf_kernel(test_x, train_x) @ coefficients
        distances = np.linalg.norm(test_x[:, None, :] - train_x[None, :, :], axis=2)
        nearest_distances[test_indices] = np.min(distances, axis=1)

    errors = prediction - test_targets
    minority_extreme = np.abs(test.minority_environment) >= 0.75
    return Metrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        minority_extreme_rmse=float(np.sqrt(np.mean(errors[minority_extreme] ** 2))),
        local_coverage_radius_p95=float(np.quantile(nearest_distances, 0.95)),
    )


def run_once(seed: int, budget: int = DEFAULT_BUDGET) -> tuple[dict[str, Metrics], dict[str, IndexArray]]:
    """Run one deterministic candidate-selection and train/test comparison."""
    rng = np.random.default_rng(seed)
    candidates = make_candidate_pool(rng)
    selections = select_training_sets(candidates, budget, rng)
    test = make_balanced_test_set(rng)
    metrics = {
        name: evaluate_selection(candidates, test, selected)
        for name, selected in selections.items()
    }
    return metrics, selections


def run_repeated(seed: int, repeats: int, budget: int = DEFAULT_BUDGET) -> list[dict[str, Metrics]]:
    """Repeat the complete experiment with consecutive deterministic seeds."""
    return [run_once(seed + repeat, budget)[0] for repeat in range(repeats)]


def median_metrics(results: list[dict[str, Metrics]], method: str) -> Metrics:
    """Aggregate one method over repeated experiments."""
    values = np.asarray(
        [
            [
                result[method].rmse,
                result[method].minority_extreme_rmse,
                result[method].local_coverage_radius_p95,
            ]
            for result in results
        ],
        dtype=float,
    )
    medians = np.median(values, axis=0)
    return Metrics(*medians.tolist())


def win_rate(results: list[dict[str, Metrics]], method: str, field: str) -> float:
    """Return the fraction of repeats beating global mean FPS."""
    baseline = np.asarray([getattr(result["global mean FPS"], field) for result in results])
    challenger = np.asarray([getattr(result[method], field) for result in results])
    return float(np.mean(challenger < baseline))


def print_report(seed: int, repeats: int, budget: int, results: list[dict[str, Metrics]]) -> None:
    """Print a compact, copyable validation report."""
    print("Controlled complex-composition FPS demo")
    print(f"seed={seed}, repeats={repeats}, budget={budget}, minority_fraction={MINORITY_FRACTION:.2f}")
    print("All methods use the same budget, test set, features, and kernel-ridge learner.\n")
    print(f"{'method':<24} {'RMSE':>9} {'rare RMSE':>11} {'local r95':>11} {'rare wins':>11}")
    print("-" * 72)
    for method in results[0]:
        metrics = median_metrics(results, method)
        wins = 0.0 if method == "global mean FPS" else win_rate(
            results, method, "minority_extreme_rmse"
        )
        print(
            f"{method:<24} {metrics.rmse:9.4f} {metrics.minority_extreme_rmse:11.4f} "
            f"{metrics.local_coverage_radius_p95:11.4f} {wins:10.1%}"
        )

    baseline = median_metrics(results, "global mean FPS")
    proposed = median_metrics(results, "stratified local FPS")
    overall_gain = 1.0 - proposed.rmse / baseline.rmse
    rare_gain = 1.0 - proposed.minority_extreme_rmse / baseline.minority_extreme_rmse
    coverage_gain = 1.0 - proposed.local_coverage_radius_p95 / baseline.local_coverage_radius_p95
    print("\nMedian improvement of stratified local FPS over global mean FPS:")
    print(f"  overall RMSE: {overall_gain:.1%}")
    print(f"  minority-extreme RMSE: {rare_gain:.1%}")
    print(f"  local coverage radius p95: {coverage_gain:.1%}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    return parser.parse_args()


def main() -> None:
    """Run the controlled validation demo."""
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.budget <= 0 or args.budget % N_COMPOSITIONS != 0:
        raise ValueError(f"budget must be positive and divisible by {N_COMPOSITIONS}")
    results = run_repeated(args.seed, args.repeats, args.budget)
    print_report(args.seed, args.repeats, args.budget, results)


if __name__ == "__main__":
    main()
