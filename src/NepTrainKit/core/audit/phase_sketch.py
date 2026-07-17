"""Experimental geometry/chemistry-factorized local phase fingerprints.

This module is intentionally UI-independent.  It is a research prototype for
the Training Set Audit phase-atlas direction, not yet a public product API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_legendre

try:
    from NepTrainKit._native import _phase as _native_phase
except ImportError:  # pragma: no cover - source checkout/reference fallback
    _native_phase = None


_EPS = 1.0e-12


@dataclass(frozen=True)
class PhaseSketch:
    """Per-atom geometry and chemical-decoration fingerprints."""

    geometry: np.ndarray
    chemistry: np.ndarray
    translational_order_score: float | None
    translational_order_limit: float | None
    cna_labels: np.ndarray


@dataclass(frozen=True)
class PrototypePrediction:
    labels: np.ndarray
    distances: np.ndarray
    margins: np.ndarray


def _periodic_images(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand positions that have already been wrapped into the primary cell."""
    atom_count = len(positions)
    source_indices = np.arange(atom_count, dtype=np.int32)
    if not np.any(pbc):
        return positions, source_indices, np.zeros((atom_count, 3), dtype=np.int32)

    inverse = np.linalg.inv(cell)
    face_distances = 1.0 / np.linalg.norm(inverse.T, axis=1)
    ranges = [
        range(-int(np.ceil(cutoff / face_distances[axis])) - 1,
              int(np.ceil(cutoff / face_distances[axis])) + 2)
        if pbc[axis]
        else (0,)
        for axis in range(3)
    ]
    shifts = np.asarray(tuple(product(*ranges)), dtype=np.int32)
    image_positions = (
        positions[np.newaxis, :, :] + (shifts @ cell)[:, np.newaxis, :]
    ).reshape(-1, 3)
    return (
        image_positions,
        np.tile(source_indices, len(shifts)),
        np.repeat(shifts, atom_count, axis=0),
    )


def periodic_knn_vectors(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    *,
    neighbors: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest periodic neighbor vectors for orthogonal or skewed cells.

    The output has shapes ``(N, K, 3)``, ``(N, K)``, and ``(N, K)`` for
    vectors, source atom indices, and a validity mask.  Periodic images of the
    same source atom are legitimate distinct neighbors; only the zero-shift
    central self is removed.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    box = np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3)
    periodic = np.asarray(pbc, dtype=bool).reshape(3)
    atom_count = len(pos)
    if atom_count == 0:
        return (
            np.empty((0, neighbors, 3), dtype=np.float64),
            np.empty((0, neighbors), dtype=np.int32),
            np.empty((0, neighbors), dtype=bool),
        )
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")

    volume = abs(float(np.linalg.det(box))) if np.any(periodic) else 0.0
    if np.any(periodic) and volume <= _EPS:
        raise ValueError("periodic cell must be invertible")
    if np.any(periodic):
        inverse = np.linalg.inv(box)
        fractional = pos @ inverse
        fractional[:, periodic] -= np.floor(fractional[:, periodic])
        pos = fractional @ box
        density = atom_count / volume
        radius = (3.0 * (neighbors + 2) / (4.0 * np.pi * density)) ** (1.0 / 3.0)
        cutoff = max(radius * 1.8, np.min(np.linalg.norm(box, axis=1)) * 0.55)
    else:
        extent = np.ptp(pos, axis=0)
        cutoff = max(float(np.linalg.norm(extent)), 1.0)

    query_count = min(neighbors + 8, max(neighbors + 1, atom_count * 2))
    for _ in range(8):
        image_pos, image_indices, image_shifts = _periodic_images(pos, box, periodic, cutoff)
        tree = cKDTree(image_pos)
        k_query = min(max(query_count, neighbors + 1), len(image_pos))
        distances, candidates = tree.query(pos, k=k_query)
        if k_query == 1:
            distances = distances[:, None]
            candidates = candidates[:, None]

        vectors = np.zeros((atom_count, neighbors, 3), dtype=np.float64)
        indices = np.full((atom_count, neighbors), -1, dtype=np.int32)
        valid = np.zeros((atom_count, neighbors), dtype=bool)
        kth_distances = np.zeros(atom_count, dtype=np.float64)
        complete = True
        for center in range(atom_count):
            candidate_ids = np.asarray(candidates[center], dtype=np.intp)
            candidate_distances = np.asarray(distances[center], dtype=np.float64)
            central_self = (
                (image_indices[candidate_ids] == center)
                & np.all(image_shifts[candidate_ids] == 0, axis=1)
            )
            keep = ~central_self & np.isfinite(candidate_distances)
            candidate_ids = candidate_ids[keep]
            candidate_distances = candidate_distances[keep]
            order = np.argsort(candidate_distances, kind="stable")
            candidate_ids = candidate_ids[order][:neighbors]
            candidate_distances = candidate_distances[order][:neighbors]
            count = len(candidate_ids)
            if count:
                vectors[center, :count] = image_pos[candidate_ids] - pos[center]
                indices[center, :count] = image_indices[candidate_ids]
                valid[center, :count] = True
                kth_distances[center] = candidate_distances[-1]
            if count < neighbors:
                complete = False
        if not np.any(periodic) or (complete and float(np.max(kth_distances)) < cutoff * 0.82):
            return vectors, indices, valid
        cutoff *= 1.7
        query_count = min(len(image_pos), max(query_count * 2, neighbors + 8))
    raise RuntimeError("failed to recover the requested periodic neighbors")


def accelerated_periodic_knn_vectors(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    *,
    neighbors: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the native cell-list KNN when available, otherwise the oracle."""
    if (
        _native_phase is not None
        and hasattr(_native_phase, "periodic_knn_vectors")
    ):
        return _native_phase.periodic_knn_vectors(
            positions,
            cell,
            pbc,
            neighbors,
        )
    return periodic_knn_vectors(
        positions,
        cell,
        pbc,
        neighbors=neighbors,
    )


def _translational_order_reference(
    positions: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, float]:
    fractional = np.asarray(positions, dtype=np.float64) @ np.linalg.inv(
        np.asarray(cell, dtype=np.float64)
    )
    fractional -= np.floor(fractional)
    atom_count = len(fractional)
    maximum_harmonic = min(64, int(np.ceil(4.0 * atom_count ** (1.0 / 3.0))))
    directions = np.asarray(
        (
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
        ),
        dtype=np.float64,
    )
    harmonics = np.arange(1, maximum_harmonic + 1, dtype=np.float64)
    score = 0.0
    for direction in directions:
        phases = (fractional @ direction)[:, None] * harmonics[None, :]
        amplitudes = np.abs(np.mean(np.exp(2j * np.pi * phases), axis=0)) ** 2
        score = max(score, float(np.max(amplitudes)))
    wave_count = len(directions) * maximum_harmonic
    random_limit = min(1.0, float(np.log(wave_count / 0.01) / atom_count))
    return score, random_limit


def translational_order_evidence(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
) -> tuple[float | None, float | None]:
    """Return a Bragg-order score and a finite-size random-position limit."""
    periodic = np.asarray(pbc, dtype=bool).reshape(3)
    if not np.all(periodic):
        return None, None
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    box = np.ascontiguousarray(cell, dtype=np.float32).reshape(3, 3)
    if not len(pos):
        raise ValueError("translational order requires at least one atom")
    if _native_phase is not None and hasattr(
        _native_phase, "translational_order_evidence"
    ):
        score, limit = _native_phase.translational_order_evidence(
            pos, box, periodic
        )
        return float(score), float(limit)
    return _translational_order_reference(pos, box)


def _longest_graph_path(
    adjacency: np.ndarray,
    vertex: int,
    visited: int,
) -> int:
    best = 0
    for neighbor in np.flatnonzero(adjacency[vertex]):
        bit = 1 << int(neighbor)
        if visited & bit:
            continue
        best = max(
            best,
            1 + _longest_graph_path(adjacency, int(neighbor), visited | bit),
        )
    return best


def _adaptive_cna_reference(
    vectors: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    labels = np.zeros(len(vectors), dtype=np.int8)
    for atom, row in enumerate(vectors):
        neighbors = row[valid[atom]]
        distances = np.linalg.norm(neighbors, axis=1)
        order = np.argsort(distances, kind="stable")
        neighbors = neighbors[order]
        distances = distances[order]
        if (
            len(neighbors) >= 13
            and distances[11] > _EPS
            and distances[12] / distances[11] >= 1.08
        ):
            cutoff = 0.5 * (distances[11] + distances[12])
            signatures: Counter[tuple[int, int, int]] = Counter()
            for bonded in range(12):
                common = [
                    candidate
                    for candidate in range(12)
                    if candidate != bonded
                    and np.linalg.norm(neighbors[candidate] - neighbors[bonded]) < cutoff
                ]
                if len(common) != 4:
                    break
                adjacency = np.zeros((4, 4), dtype=bool)
                for left in range(4):
                    for right in range(left + 1, 4):
                        if np.linalg.norm(
                            neighbors[common[left]] - neighbors[common[right]]
                        ) < cutoff:
                            adjacency[left, right] = True
                            adjacency[right, left] = True
                bond_count = int(np.sum(adjacency) // 2)
                longest_path = max(
                    _longest_graph_path(adjacency, vertex, 1 << vertex)
                    for vertex in range(4)
                )
                signatures[(len(common), bond_count, longest_path)] += 1
            else:
                if signatures == {(4, 2, 1): 12}:
                    labels[atom] = 1
                    continue
                if signatures == {(4, 2, 1): 6, (4, 2, 2): 6}:
                    labels[atom] = 2
                    continue
        if (
            len(neighbors) >= 15
            and distances[13] > _EPS
            and distances[14] / distances[13] >= 1.08
        ):
            cutoff = 0.5 * (distances[13] + distances[14])
            signatures = Counter()
            for bonded in range(14):
                common = [
                    candidate
                    for candidate in range(14)
                    if candidate != bonded
                    and np.linalg.norm(neighbors[candidate] - neighbors[bonded]) < cutoff
                ]
                adjacency = np.zeros((len(common), len(common)), dtype=bool)
                for left in range(len(common)):
                    for right in range(left + 1, len(common)):
                        if np.linalg.norm(
                            neighbors[common[left]] - neighbors[common[right]]
                        ) < cutoff:
                            adjacency[left, right] = True
                            adjacency[right, left] = True
                bond_count = int(np.sum(adjacency) // 2)
                longest_path = (
                    max(
                        _longest_graph_path(adjacency, vertex, 1 << vertex)
                        for vertex in range(len(common))
                    )
                    if common
                    else 0
                )
                signatures[(len(common), bond_count, longest_path)] += 1
            if signatures == {(4, 4, 3): 6, (6, 6, 5): 8}:
                labels[atom] = 3
    return labels


def adaptive_cna_labels(
    vectors: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Classify ideal-to-moderately-distorted FCC/HCP/BCC local topology."""
    if _native_phase is not None and hasattr(
        _native_phase, "adaptive_cna_labels"
    ):
        return np.asarray(
            _native_phase.adaptive_cna_labels(vectors, indices, valid),
            dtype=np.int8,
        )
    return _adaptive_cna_reference(vectors, valid)


def _bond_order_features(unit_vectors: np.ndarray, counts: Sequence[int]) -> list[float]:
    features: list[float] = []
    orders = (2, 4, 6, 8, 10, 12)
    for count in counts:
        current = unit_vectors[: min(count, len(unit_vectors))]
        if not len(current):
            features.extend(0.0 for _ in orders)
            continue
        cosine = np.clip(current @ current.T, -1.0, 1.0)
        for order in orders:
            q_squared = float(np.mean(eval_legendre(order, cosine)))
            features.append(float(np.sqrt(max(q_squared, 0.0))))
    return features


def _topology_barcode(vectors: np.ndarray, normalized_distances: np.ndarray) -> list[float]:
    features: list[float] = []
    thresholds = (1.08, 1.22, 1.42, 1.75, 2.10)
    edge_threshold = 1.24
    for threshold in thresholds:
        selected = vectors[normalized_distances <= threshold]
        count = len(selected)
        if count < 2:
            features.extend((count / 24.0, 0.0, 0.0, 0.0, 0.0))
            continue
        pair_distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=2)
        adjacency = (pair_distances <= edge_threshold) & ~np.eye(count, dtype=bool)
        degrees = np.sum(adjacency, axis=1, dtype=np.float64)
        edges = float(np.sum(degrees) / 2.0)
        triangles = float(np.trace(adjacency.astype(np.float64) @ adjacency @ adjacency) / 6.0)
        possible_edges = max(count * (count - 1) / 2.0, 1.0)
        possible_triangles = max(count * (count - 1) * (count - 2) / 6.0, 1.0)
        features.extend(
            (
                count / 24.0,
                edges / possible_edges,
                triangles / possible_triangles,
                float(np.mean(degrees)) / max(count - 1, 1),
                float(np.std(degrees)) / max(count - 1, 1),
            )
        )
    return features


def _chemistry_features(
    center_type: int,
    neighbor_types: np.ndarray,
    normalized_distances: np.ndarray,
) -> list[float]:
    """Return smooth, label-invariant radial chemistry correlations.

    Fixed neighbor counts are deliberately avoided: cutting an exactly
    degenerate coordination shell makes chemical order depend on arbitrary
    k-d-tree tie ordering.  Smooth radial windows retain the same information
    while remaining continuous under strain and thermal displacement.
    """
    features: list[float] = []
    species = np.unique(neighbor_types)
    shell_cutoff = 1.0 / (1.0 + np.exp((normalized_distances - 1.55) / 0.04))
    for center in (0.95, 1.18, 1.42):
        for width in (0.08, 0.16):
            weights = (
                np.exp(-0.5 * ((normalized_distances - center) / width) ** 2)
                * shell_cutoff
            )
            total = float(np.sum(weights))
            if total <= _EPS:
                features.extend((0.0,) * 7)
                continue
            fractions = np.asarray(
                [np.sum(weights[neighbor_types == value]) / total for value in species],
                dtype=np.float64,
            )
            ordered = np.sort(fractions)[::-1]
            padded = np.pad(ordered[:3], (0, max(0, 3 - len(ordered))))
            pair_equal = float(np.sum(fractions * fractions))
            entropy = float(-np.sum(fractions * np.log(fractions + _EPS)) / np.log(4.0))
            effective_species = min(1.0 / max(pair_equal, _EPS), 4.0) / 4.0
            same_type = float(np.sum(weights[neighbor_types == center_type]) / total)
            features.extend(
                (
                    same_type,
                    effective_species,
                    float(padded[0]),
                    float(padded[1]),
                    float(padded[2]),
                    pair_equal,
                    entropy,
                )
            )
    return features


def phase_sketch(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    atom_types: Sequence[int],
    *,
    neighbors: int = 24,
) -> PhaseSketch:
    """Compute factorized local fingerprints for all atoms in one structure."""
    types = np.asarray(atom_types, dtype=np.int32)
    translational_score, translational_limit = translational_order_evidence(
        positions, cell, pbc
    )
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        positions,
        cell,
        pbc,
        neighbors=neighbors,
    )
    if _native_phase is not None:
        geometry, chemistry = _native_phase.phase_features(
            vectors,
            indices,
            valid,
            types,
        )
        cna_labels = adaptive_cna_labels(vectors, indices, valid)
        return PhaseSketch(
            geometry=np.asarray(geometry, dtype=np.float32),
            chemistry=np.asarray(chemistry, dtype=np.float32),
            translational_order_score=translational_score,
            translational_order_limit=translational_limit,
            cna_labels=cna_labels,
        )
    cna_labels = _adaptive_cna_reference(vectors, valid)
    geometry_rows: list[list[float]] = []
    chemistry_rows: list[list[float]] = []
    for center in range(len(vectors)):
        current_vectors = vectors[center, valid[center]]
        current_indices = indices[center, valid[center]]
        distances = np.linalg.norm(current_vectors, axis=1)
        order = np.argsort(distances, kind="stable")
        current_vectors = current_vectors[order]
        current_indices = current_indices[order]
        distances = distances[order]
        if not len(distances):
            raise ValueError("each atom must have at least one valid neighbor")
        scale_count = min(6, len(distances))
        scale = float(np.mean(distances[:scale_count]))
        if scale <= _EPS:
            raise ValueError("overlapping atoms cannot be fingerprinted")
        normalized_vectors = current_vectors / scale
        normalized_distances = distances / scale
        unit_vectors = current_vectors / np.maximum(distances[:, None], _EPS)

        radial = np.full(neighbors, 3.0, dtype=np.float64)
        radial[: min(neighbors, len(normalized_distances))] = np.clip(
            normalized_distances[:neighbors], 0.0, 3.0
        )
        gaps = np.zeros(neighbors - 1, dtype=np.float64)
        gap_count = min(neighbors - 1, max(len(normalized_distances) - 1, 0))
        if gap_count:
            gaps[:gap_count] = np.log(
                np.maximum(normalized_distances[1 : gap_count + 1], _EPS)
                / np.maximum(normalized_distances[:gap_count], _EPS)
            )
        angular = _bond_order_features(unit_vectors, (4, 6, 8, 12, 14, 18, 24))
        cosine = np.clip(unit_vectors[: min(12, len(unit_vectors))] @ unit_vectors[: min(12, len(unit_vectors))].T, -1.0, 1.0)
        upper = cosine[np.triu_indices(len(cosine), k=1)]
        histogram = np.histogram(upper, bins=np.linspace(-1.0, 1.0, 9))[0].astype(np.float64)
        if histogram.sum():
            histogram /= histogram.sum()
        geometry_rows.append(
            radial.tolist()
            + gaps.tolist()
            + angular
            + _topology_barcode(normalized_vectors, normalized_distances)
            + histogram.tolist()
        )
        chemistry_rows.append(
            _chemistry_features(
                int(types[center]),
                types[current_indices],
                normalized_distances,
            )
        )
    return PhaseSketch(
        geometry=np.asarray(geometry_rows, dtype=np.float32),
        chemistry=np.asarray(chemistry_rows, dtype=np.float32),
        translational_order_score=translational_score,
        translational_order_limit=translational_limit,
        cna_labels=cna_labels,
    )


def summarize_phase_sketch(values: np.ndarray) -> np.ndarray:
    """Collapse an atom-level sketch into a robust structure-level signature."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("values must be a non-empty atom-by-feature matrix")
    return np.concatenate(
        (
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0),
            np.quantile(matrix, 0.10, axis=0),
            np.quantile(matrix, 0.50, axis=0),
            np.quantile(matrix, 0.90, axis=0),
        )
    ).astype(np.float32, copy=False)


def _lloyd_centers(values: np.ndarray, count: int, iterations: int = 30) -> np.ndarray:
    """Small deterministic farthest-point initialized k-means."""
    if count >= len(values):
        return values.copy()
    mean = np.mean(values, axis=0)
    first = int(np.argmin(np.sum((values - mean) ** 2, axis=1)))
    selected = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((values - values[index]) ** 2, axis=1))
    centers = values[np.asarray(selected)].copy()
    for _ in range(iterations):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        updated = centers.copy()
        for cluster in range(count):
            members = values[assignments == cluster]
            if len(members):
                updated[cluster] = np.mean(members, axis=0)
        if np.allclose(updated, centers, rtol=1.0e-6, atol=1.0e-7):
            break
        centers = updated
    return centers


class PrototypeBank:
    """Compact deterministic prototype-cloud classifier with open-set rejection."""

    def __init__(
        self,
        *,
        prototypes_per_class: int = 12,
        rejection_quantile: float = 0.995,
        rejection_scale: float = 1.35,
        minimum_margin: float = 1.02,
        samples_per_prototype: int = 24,
    ) -> None:
        self.prototypes_per_class = int(prototypes_per_class)
        self.rejection_quantile = float(rejection_quantile)
        self.rejection_scale = float(rejection_scale)
        self.minimum_margin = float(minimum_margin)
        self.samples_per_prototype = max(1, int(samples_per_prototype))
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.center_labels_: np.ndarray | None = None
        self.thresholds_: dict[str, float] = {}

    def fit(self, values: np.ndarray, labels: Iterable[str]) -> "PrototypeBank":
        matrix = np.asarray(values, dtype=np.float64)
        label_array = np.asarray(tuple(labels), dtype=object)
        if matrix.ndim != 2 or len(matrix) != len(label_array) or not len(matrix):
            raise ValueError("values and labels must contain a non-empty aligned matrix")
        self.mean_ = np.mean(matrix, axis=0)
        self.scale_ = np.std(matrix, axis=0)
        self.scale_[self.scale_ < 1.0e-6] = 1.0
        normalized = (matrix - self.mean_) / self.scale_

        centers: list[np.ndarray] = []
        center_labels: list[str] = []
        for label in sorted({str(value) for value in label_array}):
            rows = normalized[label_array == label]
            count = min(
                self.prototypes_per_class,
                max(1, len(rows) // self.samples_per_prototype),
                len(rows),
            )
            class_centers = _lloyd_centers(rows, count)
            centers.append(class_centers)
            center_labels.extend([label] * len(class_centers))
        self.centers_ = np.concatenate(centers, axis=0)
        self.center_labels_ = np.asarray(center_labels, dtype=object)

        distances = np.sum(
            (normalized[:, None, :] - self.centers_[None, :, :]) ** 2,
            axis=2,
        ) / normalized.shape[1]
        for label in sorted({str(value) for value in label_array}):
            class_columns = self.center_labels_ == label
            own_rows = label_array == label
            nearest = np.min(distances[own_rows][:, class_columns], axis=1)
            self.thresholds_[label] = max(
                float(np.quantile(nearest, self.rejection_quantile)) * self.rejection_scale,
                1.0e-7,
            )
        return self

    def predict(self, values: np.ndarray) -> PrototypePrediction:
        if self.mean_ is None or self.scale_ is None or self.centers_ is None or self.center_labels_ is None:
            raise RuntimeError("prototype bank must be fitted before prediction")
        matrix = np.asarray(values, dtype=np.float64)
        normalized = (matrix - self.mean_) / self.scale_
        distances = np.sum(
            (normalized[:, None, :] - self.centers_[None, :, :]) ** 2,
            axis=2,
        ) / normalized.shape[1]
        classes = tuple(sorted(self.thresholds_))
        class_distances = np.column_stack(
            [np.min(distances[:, self.center_labels_ == label], axis=1) for label in classes]
        )
        order = np.argsort(class_distances, axis=1)
        best_indices = order[:, 0]
        best = class_distances[np.arange(len(matrix)), best_indices]
        second = class_distances[np.arange(len(matrix)), order[:, 1]] if len(classes) > 1 else np.full(len(matrix), np.inf)
        margins = second / np.maximum(best, _EPS)
        labels = np.asarray([classes[index] for index in best_indices], dtype=object)
        rejected = np.asarray(
            [
                distance > self.thresholds_[str(label)] or margin < self.minimum_margin
                for label, distance, margin in zip(labels, best, margins)
            ],
            dtype=bool,
        )
        labels[rejected] = "unknown"
        return PrototypePrediction(labels=labels, distances=best, margins=margins)

    def distances_by_class(self, values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        """Return normalized distance to the nearest prototype of every class."""
        if self.mean_ is None or self.scale_ is None or self.centers_ is None or self.center_labels_ is None:
            raise RuntimeError("prototype bank must be fitted before prediction")
        matrix = np.asarray(values, dtype=np.float64)
        normalized = (matrix - self.mean_) / self.scale_
        distances = np.sum(
            (normalized[:, None, :] - self.centers_[None, :, :]) ** 2,
            axis=2,
        ) / normalized.shape[1]
        classes = tuple(sorted(self.thresholds_))
        return classes, np.column_stack(
            [np.min(distances[:, self.center_labels_ == label], axis=1) for label in classes]
        )

    def accepts_labels(self, values: np.ndarray, labels: Iterable[str]) -> np.ndarray:
        """Test externally predicted labels against this bank's open-set envelope."""
        requested = tuple(str(label) for label in labels)
        classes, distances = self.distances_by_class(values)
        class_indices = {label: index for index, label in enumerate(classes)}
        accepted = np.zeros(len(requested), dtype=bool)
        for row, label in enumerate(requested):
            index = class_indices.get(label)
            if index is None:
                continue
            accepted[row] = distances[row, index] <= self.thresholds_[label]
        return accepted
