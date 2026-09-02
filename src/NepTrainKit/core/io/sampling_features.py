"""Feature construction for representative whole-dataset sampling.

Hard physical partitions protect discrete coverage (element set, phase, and
magnetic order).  This module supplies the continuous geometry used by FPS
inside each partition.  Callers do not need to know how model, lattice, and
spin evidence are pooled or balanced.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from NepTrainKit.core.audit.magnetic_inventory import analyze_structure_magnetism
from NepTrainKit.core.geometry_cache import structure_cell_array, structure_pbc_flags

_SCALE_EPS = 1.0e-12


@dataclass(frozen=True)
class SamplingFeatureBlocks:
    """Named continuous feature blocks aligned one-to-one with structures."""

    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]

    @property
    def row_count(self) -> int:
        return 0 if not self.values else int(self.values[0].shape[0])

    def take(self, rows: Sequence[int]) -> SamplingFeatureBlocks:
        """Return aligned rows without exposing the block representation."""
        indices = np.asarray(rows, dtype=np.int64)
        return SamplingFeatureBlocks(
            self.names,
            tuple(np.ascontiguousarray(block[indices]) for block in self.values),
        )


def _validated_structure_descriptors(
    descriptors: np.ndarray,
    structure_count: int,
) -> np.ndarray:
    values = np.ascontiguousarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != int(structure_count):
        raise ValueError("structure descriptors must align with structures")
    if not np.isfinite(values).all():
        raise ValueError("structure descriptors must be finite")
    return values


def _atomic_descriptor_spread(
    per_atom_descriptors: np.ndarray,
    atom_counts: Sequence[int],
    descriptor_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(per_atom_descriptors, dtype=np.float64)
    counts = np.asarray(atom_counts, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != int(descriptor_dim):
        raise ValueError("atomic descriptors must match the model descriptor dimension")
    if np.any(counts <= 0) or int(np.sum(counts)) != values.shape[0]:
        raise ValueError("atomic descriptors must align with structure atom counts")
    if not np.isfinite(values).all():
        raise ValueError("atomic descriptors must be finite")

    std_rows: list[np.ndarray] = []
    tail_rows: list[np.ndarray] = []
    offset = 0
    for count in counts:
        block = values[offset : offset + int(count)]
        center = np.mean(block, axis=0)
        std_rows.append(np.std(block, axis=0))
        # The farthest local environment is retained instead of disappearing
        # in a structure mean.  Absolute deviation is sign independent.
        tail_rows.append(np.max(np.abs(block - center), axis=0))
        offset += int(count)
    return np.vstack(std_rows), np.vstack(tail_rows)


def _cell_feature_row(structure: Any) -> np.ndarray:
    cell = np.asarray(structure_cell_array(structure), dtype=np.float64).reshape(3, 3)
    pbc = np.asarray(structure_pbc_flags(structure), dtype=np.float64).reshape(3)
    atom_count = max(1, len(structure))
    volume = abs(float(np.linalg.det(cell)))
    volume_per_atom = np.log(max(volume / atom_count, _SCALE_EPS))

    lengths = np.linalg.norm(cell, axis=1)
    valid = lengths > _SCALE_EPS
    cosines = np.zeros(3, dtype=np.float64)
    for output, (left, right) in enumerate(((1, 2), (0, 2), (0, 1))):
        if valid[left] and valid[right]:
            cosines[output] = float(
                np.clip(
                    np.dot(cell[left], cell[right]) / (lengths[left] * lengths[right]),
                    -1.0,
                    1.0,
                )
            )

    singular = np.linalg.svd(cell, compute_uv=False)
    singular = np.maximum(singular, _SCALE_EPS)
    geometric_mean = float(np.exp(np.mean(np.log(singular))))
    shape = np.log(singular / geometric_mean)
    return np.concatenate(([volume_per_atom], shape, cosines, pbc))


def _magnetic_feature_row(structure: Any) -> np.ndarray:
    evidence = analyze_structure_magnetism(structure, source_index=0)
    if evidence is None:
        return np.zeros(17, dtype=np.float64)

    q_index = np.asarray(evidence.q_vector, dtype=np.float64)
    q_cart = np.zeros(3, dtype=np.float64)
    cell = np.asarray(structure_cell_array(structure), dtype=np.float64).reshape(3, 3)
    if np.linalg.matrix_rank(cell) == 3 and np.any(q_index):
        q_cart = 2.0 * np.pi * (q_index @ np.linalg.inv(cell).T)
    q_norm = float(np.linalg.norm(q_cart))
    direction = q_cart / q_norm if q_norm > _SCALE_EPS else np.zeros(3)
    direction_tensor = np.asarray(
        (
            direction[0] * direction[0],
            direction[1] * direction[1],
            direction[2] * direction[2],
            direction[0] * direction[1],
            direction[0] * direction[2],
            direction[1] * direction[2],
        ),
        dtype=np.float64,
    )
    q_peak = float(evidence.q_peak_strength)
    return np.asarray(
        (
            evidence.mean_moment,
            evidence.moment_std,
            evidence.net_moment_ratio,
            evidence.collinearity,
            evidence.coplanarity,
            evidence.neighbor_correlation,
            evidence.neighbor_abs_correlation,
            evidence.parallel_fraction,
            evidence.antiparallel_fraction,
            q_peak,
            np.log1p(q_norm),
            *(q_peak * direction_tensor),
        ),
        dtype=np.float64,
    )


def build_sampling_feature_blocks(
    structures: Sequence[Any],
    structure_descriptors: np.ndarray,
    *,
    per_atom_descriptors: np.ndarray | None = None,
    spin_model: bool,
) -> SamplingFeatureBlocks:
    """Build model, local-environment, lattice, and optional spin blocks."""
    structure_list = tuple(structures)
    mean = _validated_structure_descriptors(
        structure_descriptors,
        len(structure_list),
    )
    names: list[str] = ["descriptor_mean"]
    blocks: list[np.ndarray] = [mean]
    if per_atom_descriptors is not None:
        std, tail = _atomic_descriptor_spread(
            per_atom_descriptors,
            [len(structure) for structure in structure_list],
            mean.shape[1],
        )
        names.extend(("descriptor_std", "descriptor_tail"))
        blocks.extend((std, tail))
    lattice = np.vstack([_cell_feature_row(structure) for structure in structure_list])
    names.append("lattice")
    blocks.append(lattice)
    if spin_model:
        magnetic = np.vstack(
            [_magnetic_feature_row(structure) for structure in structure_list]
        )
        names.append("magnetism")
        blocks.append(magnetic)
    return SamplingFeatureBlocks(tuple(names), tuple(blocks))


def _normalize_block_sets(
    candidate: np.ndarray,
    existing: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    fit = candidate if existing is None else np.vstack((candidate, existing))
    center = np.median(fit, axis=0)
    q25, q75 = np.percentile(fit, (25.0, 75.0), axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.std(fit, axis=0)
    scale = np.where(scale > _SCALE_EPS, scale, standard)
    active = scale > _SCALE_EPS
    safe_scale = np.where(active, scale, 1.0)
    candidate_normalized = (candidate - center) / safe_scale
    candidate_normalized[:, ~active] = 0.0
    existing_normalized = None
    if existing is not None:
        existing_normalized = (existing - center) / safe_scale
        existing_normalized[:, ~active] = 0.0
    # A 200-dimensional model block must not overwhelm a 10-dimensional
    # lattice or magnetic block merely because it has more columns.
    block_weight = np.sqrt(max(1, int(np.count_nonzero(active))))
    candidate_normalized /= block_weight
    if existing_normalized is not None:
        existing_normalized /= block_weight
    return candidate_normalized, existing_normalized


def representative_sampling_features(
    candidate: SamplingFeatureBlocks,
    existing: SamplingFeatureBlocks | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize and balance blocks into the single FPS feature geometry."""
    if existing is not None and existing.names != candidate.names:
        raise ValueError("candidate and existing sampling feature blocks differ")
    candidate_parts: list[np.ndarray] = []
    existing_parts: list[np.ndarray] = []
    for block_index, candidate_block in enumerate(candidate.values):
        existing_block = None if existing is None else existing.values[block_index]
        normalized, normalized_existing = _normalize_block_sets(
            candidate_block,
            existing_block,
        )
        candidate_parts.append(normalized)
        if normalized_existing is not None:
            existing_parts.append(normalized_existing)
    candidate_values = np.ascontiguousarray(
        np.hstack(candidate_parts),
        dtype=np.float32,
    )
    existing_values = (
        None
        if existing is None
        else np.ascontiguousarray(np.hstack(existing_parts), dtype=np.float32)
    )
    return candidate_values, existing_values
