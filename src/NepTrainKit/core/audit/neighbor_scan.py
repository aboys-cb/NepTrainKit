"""Batched short-distance scanning with a native backend and exact fallback."""
from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    from NepTrainKit.core import _fastaudit as _native_scan
except ImportError:  # Source-only development and unsupported build platforms.
    _native_scan = None


_NATIVE_CELL_DETERMINANT_TOLERANCE = 1.0e-12
_CELL_VALID = np.uint8(1)
_NATIVE_NEIGHBOR_SUPPORTED = np.uint8(2)
_CELL_STATUS_AMBIGUOUS = np.uint8(4)


def _valid_periodic_cell_python(cell: np.ndarray, pbc: np.ndarray) -> bool:
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return False
    periodic_vectors = cell[pbc]
    if periodic_vectors.size == 0:
        return True
    if np.any(np.linalg.norm(periodic_vectors, axis=1) <= 1.0e-12):
        return False
    return int(np.linalg.matrix_rank(periodic_vectors, tol=1.0e-12)) == int(np.sum(pbc))


def periodic_cell_statuses(
    cells: Sequence[np.ndarray],
    pbc_flags: Sequence[np.ndarray],
) -> np.ndarray:
    """Return cell-valid and native-neighbor capability bits for each row."""
    if len(cells) != len(pbc_flags):
        raise ValueError("cells and pbc must contain the same number of structures")
    statuses = np.zeros(len(cells), dtype=np.uint8)
    normalized: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row, (cell, pbc) in enumerate(zip(cells, pbc_flags)):
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        pbc_array = np.ascontiguousarray(pbc, dtype=np.uint8)
        if cell_array.shape != (3, 3) or pbc_array.shape != (3,):
            continue
        normalized.append((row, cell_array, pbc_array))
    if not normalized:
        return statuses

    normalized_cells = np.stack([item[1] for item in normalized])
    normalized_pbc = np.stack([item[2] for item in normalized])
    if _native_scan is not None and hasattr(_native_scan, "cell_status_mask"):
        native_statuses = np.asarray(
            _native_scan.cell_status_mask(normalized_cells, normalized_pbc),
            dtype=np.uint8,
        )
        if native_statuses.shape != (len(normalized),):
            raise RuntimeError("native cell scanner returned an invalid result shape")
    else:
        native_statuses = np.full(len(normalized), _CELL_STATUS_AMBIGUOUS, dtype=np.uint8)

    for index, (row, cell, pbc) in enumerate(normalized):
        status = native_statuses[index]
        if status & _CELL_STATUS_AMBIGUOUS:
            flags = pbc.astype(bool)
            if not _valid_periodic_cell_python(cell, flags):
                status = np.uint8(0)
            else:
                supported = (
                    not np.any(flags)
                    or abs(float(np.linalg.det(cell))) > _NATIVE_CELL_DETERMINANT_TOLERANCE
                )
                status = np.uint8(_CELL_VALID | (_NATIVE_NEIGHBOR_SUPPORTED if supported else 0))
        statuses[row] = status
    return statuses


def _has_short_distance_python(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> bool:
    """Reference fallback matching the distinct-pair native contract."""
    atom_count = int(positions.shape[0])
    if atom_count < 2:
        return False
    if not np.any(pbc):
        return bool(cKDTree(positions).query_pairs(cutoff))

    off_diagonal = cell - np.diag(np.diag(cell))
    if np.all(pbc) and np.all(np.diag(cell) > 0.0) and np.allclose(
        off_diagonal, 0.0, atol=1.0e-12
    ):
        lengths = np.diag(cell)
        wrapped = np.mod(positions, lengths)
        return bool(cKDTree(wrapped, boxsize=lengths).query_pairs(cutoff))

    pinv = np.linalg.pinv(cell)
    fractional = positions @ pinv
    residual = positions - fractional @ cell
    wrapped_fractional = fractional.copy()
    wrapped_fractional[:, pbc] -= np.floor(wrapped_fractional[:, pbc])
    wrapped = wrapped_fractional @ cell + residual
    shift_axes = [(-1, 0, 1) if periodic else (0,) for periodic in pbc]
    translations = np.asarray(
        [np.asarray(shift, dtype=np.float64) @ cell for shift in product(*shift_axes)]
    )
    expanded = np.concatenate([wrapped + translation for translation in translations], axis=0)
    for first, second in cKDTree(expanded).query_pairs(cutoff):
        if first % atom_count != second % atom_count:
            return True
    return False


def _cutoff_neighbor_pairs_python(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference adapter returning directed periodic neighbors within ``cutoff``."""
    atom_count = len(positions)
    if atom_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.float64)

    wrapped = positions
    image_indices = np.arange(atom_count, dtype=np.int32)
    image_shifts = np.zeros((atom_count, 3), dtype=np.int32)
    if np.any(pbc):
        inverse = np.linalg.inv(cell)
        fractional = positions @ inverse
        fractional[:, pbc] %= 1.0
        wrapped = fractional @ cell
        face_distances = 1.0 / np.linalg.norm(inverse.T, axis=1)
        shift_ranges = [
            range(-int(np.ceil(cutoff / face_distances[axis])), int(np.ceil(cutoff / face_distances[axis])) + 1)
            if pbc[axis]
            else (0,)
            for axis in range(3)
        ]
        shifts = np.asarray(tuple(product(*shift_ranges)), dtype=np.int32)
        image_positions = (
            wrapped[np.newaxis, :, :] + (shifts @ cell)[:, np.newaxis, :]
        ).reshape(-1, 3)
        image_indices = np.tile(image_indices, len(shifts))
        image_shifts = np.repeat(shifts, atom_count, axis=0)
    else:
        image_positions = wrapped

    tree = cKDTree(image_positions)
    center_parts: list[np.ndarray] = []
    neighbor_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for center, candidates in enumerate(tree.query_ball_point(wrapped, cutoff)):
        candidate_images = np.asarray(candidates, dtype=np.intp)
        candidate_neighbors = image_indices[candidate_images]
        candidate_distances = np.linalg.norm(
            image_positions[candidate_images] - wrapped[center],
            axis=1,
        )
        central_self = (candidate_neighbors == center) & np.all(
            image_shifts[candidate_images] == 0,
            axis=1,
        )
        valid = (candidate_distances < cutoff) & ~central_self
        if np.any(valid):
            count = int(np.count_nonzero(valid))
            center_parts.append(np.full(count, center, dtype=np.int32))
            neighbor_parts.append(candidate_neighbors[valid].astype(np.int32, copy=False))
            distance_parts.append(candidate_distances[valid])
    if not center_parts:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.float64)
    return (
        np.concatenate(center_parts),
        np.concatenate(neighbor_parts),
        np.concatenate(distance_parts),
    )


def cutoff_neighbor_pairs_batch(
    positions_by_structure: Sequence[np.ndarray],
    cells: Sequence[np.ndarray],
    pbc_flags: Sequence[np.ndarray],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pair offsets and directed local neighbor pairs for one structure batch."""
    if not (len(positions_by_structure) == len(cells) == len(pbc_flags)):
        raise ValueError("positions, cells, and pbc must contain the same number of structures")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    normalized = [
        (
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(cell, dtype=np.float64),
            np.ascontiguousarray(pbc, dtype=np.uint8),
        )
        for positions, cell, pbc in zip(positions_by_structure, cells, pbc_flags)
    ]
    for positions, cell, pbc in normalized:
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("each positions array must have shape (N, 3)")
        if cell.shape != (3, 3) or pbc.shape != (3,):
            raise ValueError("each cell and pbc value must have shape (3, 3) and (3,)")

    frame_results: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(normalized)
    statuses = periodic_cell_statuses(
        [item[1] for item in normalized],
        [item[2] for item in normalized],
    )
    native_rows = [
        row
        for row, status in enumerate(statuses)
        if _native_scan is not None
        and hasattr(_native_scan, "cutoff_neighbor_pairs")
        and status & _NATIVE_NEIGHBOR_SUPPORTED
    ]
    if native_rows:
        native_positions = [normalized[row][0] for row in native_rows]
        atom_counts = np.asarray([len(value) for value in native_positions], dtype=np.int64)
        atom_offsets = np.empty(len(native_rows) + 1, dtype=np.int64)
        atom_offsets[0] = 0
        np.cumsum(atom_counts, out=atom_offsets[1:])
        pair_offsets, centers, neighbors, distances = _native_scan.cutoff_neighbor_pairs(
            np.concatenate(native_positions, axis=0),
            atom_offsets,
            np.stack([normalized[row][1] for row in native_rows]),
            np.stack([normalized[row][2] for row in native_rows]),
            float(cutoff),
        )
        pair_offsets = np.asarray(pair_offsets, dtype=np.int64)
        centers = np.asarray(centers, dtype=np.int32)
        neighbors = np.asarray(neighbors, dtype=np.int32)
        distances = np.asarray(distances, dtype=np.float64)
        for native_index, row in enumerate(native_rows):
            start = int(pair_offsets[native_index])
            stop = int(pair_offsets[native_index + 1])
            frame_results[row] = (centers[start:stop], neighbors[start:stop], distances[start:stop])

    for row, result in enumerate(frame_results):
        if result is None:
            positions, cell, pbc = normalized[row]
            frame_results[row] = _cutoff_neighbor_pairs_python(
                positions,
                cell,
                pbc.astype(bool),
                float(cutoff),
            )

    counts = np.asarray([len(result[0]) for result in frame_results if result is not None], dtype=np.int64)
    pair_offsets = np.empty(len(frame_results) + 1, dtype=np.int64)
    pair_offsets[0] = 0
    np.cumsum(counts, out=pair_offsets[1:])
    if pair_offsets[-1] == 0:
        return (
            pair_offsets,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
    completed = [result for result in frame_results if result is not None]
    return (
        pair_offsets,
        np.concatenate([result[0] for result in completed]),
        np.concatenate([result[1] for result in completed]),
        np.concatenate([result[2] for result in completed]),
    )


def typed_contact_summary(
    atom_offsets: np.ndarray,
    atom_types: np.ndarray,
    pair_offsets: np.ndarray,
    centers: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    cutoff_matrices: np.ndarray,
    detail_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return native descriptive contact aggregates, or ``None`` for fallback."""
    if _native_scan is None or not hasattr(_native_scan, "typed_contact_summary"):
        return None
    metrics, normalized_codes, normalized_values = _native_scan.typed_contact_summary(
        np.ascontiguousarray(atom_offsets, dtype=np.int64),
        np.ascontiguousarray(atom_types, dtype=np.int32),
        np.ascontiguousarray(pair_offsets, dtype=np.int64),
        np.ascontiguousarray(centers, dtype=np.int32),
        np.ascontiguousarray(neighbors, dtype=np.int32),
        np.ascontiguousarray(distances, dtype=np.float64),
        np.ascontiguousarray(cutoff_matrices, dtype=np.float64),
        np.ascontiguousarray(detail_mask, dtype=np.uint8),
    )
    return (
        np.asarray(metrics, dtype=np.float64),
        np.asarray(normalized_codes, dtype=np.int32),
        np.asarray(normalized_values, dtype=np.float64),
    )


def typed_neighbor_counts(
    atom_offsets: np.ndarray,
    atom_types: np.ndarray,
    pair_offsets: np.ndarray,
    centers: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    cutoff_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return native per-atom typed counts, or ``None`` for Python fallback."""
    if _native_scan is None or not hasattr(_native_scan, "typed_neighbor_counts"):
        return None
    counts, type_counts = _native_scan.typed_neighbor_counts(
        np.ascontiguousarray(atom_offsets, dtype=np.int64),
        np.ascontiguousarray(atom_types, dtype=np.int32),
        np.ascontiguousarray(pair_offsets, dtype=np.int64),
        np.ascontiguousarray(centers, dtype=np.int32),
        np.ascontiguousarray(neighbors, dtype=np.int32),
        np.ascontiguousarray(distances, dtype=np.float64),
        np.ascontiguousarray(cutoff_matrices, dtype=np.float64),
    )
    return (
        np.asarray(counts, dtype=np.int32),
        np.asarray(type_counts, dtype=np.int32),
    )


def find_short_distance_structure_rows(
    positions_by_structure: Sequence[np.ndarray],
    cells: Sequence[np.ndarray],
    pbc_flags: Sequence[np.ndarray],
    cutoff: float,
) -> tuple[int, ...]:
    """Return input-row indices containing a distinct atom pair within ``cutoff``.

    The native boundary is crossed once per batch. Periodic singular cells use
    the Python reference path because the native implementation intentionally
    accepts only invertible periodic cells.
    """
    if not (
        len(positions_by_structure) == len(cells) == len(pbc_flags)
    ):
        raise ValueError("positions, cells, and pbc must contain the same number of structures")
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be finite and non-negative")

    normalized: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for positions, cell, pbc in zip(positions_by_structure, cells, pbc_flags):
        position_array = np.ascontiguousarray(positions, dtype=np.float64)
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        pbc_array = np.ascontiguousarray(pbc, dtype=np.uint8)
        if position_array.ndim != 2 or position_array.shape[1:] != (3,):
            raise ValueError("each positions array must have shape (N, 3)")
        if cell_array.shape != (3, 3):
            raise ValueError("each cell must have shape (3, 3)")
        if pbc_array.shape != (3,):
            raise ValueError("each pbc value must have shape (3,)")
        normalized.append((position_array, cell_array, pbc_array))

    found: set[int] = set()
    native_rows: list[int] = []
    fallback_rows: list[int] = []
    cell_status = periodic_cell_statuses(
        [item[1] for item in normalized],
        [item[2] for item in normalized],
    )
    for row, status in enumerate(cell_status):
        if not status & _CELL_VALID:
            continue
        if _native_scan is not None and status & _NATIVE_NEIGHBOR_SUPPORTED:
            native_rows.append(row)
        else:
            fallback_rows.append(row)

    if native_rows:
        native_positions = [normalized[row][0] for row in native_rows]
        atom_counts = np.asarray([len(value) for value in native_positions], dtype=np.int64)
        offsets = np.empty(len(native_rows) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(atom_counts, out=offsets[1:])
        flat_positions = np.concatenate(native_positions, axis=0)
        native_cells = np.stack([normalized[row][1] for row in native_rows])
        native_pbc = np.stack([normalized[row][2] for row in native_rows])
        mask = np.asarray(
            _native_scan.short_distance_mask(
                flat_positions,
                offsets,
                native_cells,
                native_pbc,
                float(cutoff),
            ),
            dtype=bool,
        )
        if mask.shape != (len(native_rows),):
            raise RuntimeError("native short-distance scanner returned an invalid result shape")
        found.update(native_rows[index] for index in np.flatnonzero(mask))

    for row in fallback_rows:
        positions, cell, pbc = normalized[row]
        if _has_short_distance_python(positions, cell, pbc.astype(bool), float(cutoff)):
            found.add(row)

    return tuple(sorted(found))
