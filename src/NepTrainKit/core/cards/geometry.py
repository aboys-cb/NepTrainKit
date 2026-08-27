"""Small geometry helpers shared by Make Dataset card operations."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ase.geometry import minkowski_reduce


_MIC_NEIGHBOR_HKLS = np.asarray(
    list(itertools.product((-1, 0, 1), repeat=3)),
    dtype=float,
)


def _gauss_reduce_2d(basis: np.ndarray) -> np.ndarray:
    """Return a Gauss-reduced basis for a rank-2 Cartesian lattice."""
    first, second = np.asarray(basis, dtype=float).copy()
    while True:
        if np.dot(second, second) < np.dot(first, first):
            first, second = second, first
        coefficient = int(np.rint(np.dot(first, second) / np.dot(first, first)))
        if coefficient == 0:
            return np.asarray([first, second])
        second -= coefficient * first


@dataclass(frozen=True)
class MinimumImageContext:
    """Cached geometry needed for exact row-vector minimum images."""

    cell: np.ndarray
    pbc: np.ndarray
    inv_cell: np.ndarray
    orthogonal: bool
    reduced_cell: np.ndarray | None = None
    reduced_inv_cell: np.ndarray | None = None
    neighbor_vectors: np.ndarray | None = None
    periodic_basis: np.ndarray | None = None
    periodic_dual: np.ndarray | None = None


@lru_cache(maxsize=32)
def _minimum_image_context(cell_bytes: bytes, pbc_bytes: bytes) -> MinimumImageContext:
    cell = np.frombuffer(cell_bytes, dtype=np.float64).reshape(3, 3).copy()
    pbc = np.frombuffer(pbc_bytes, dtype=np.bool_).copy()
    metric = cell @ cell.T
    off_diagonal = metric - np.diag(np.diag(metric))
    orthogonal = bool(np.all(np.abs(off_diagonal) <= 1e-12))
    inv_cell = np.linalg.inv(cell)
    if not bool(np.all(pbc)):
        periodic_basis = cell[pbc]
        if len(periodic_basis) == 2:
            periodic_basis = _gauss_reduce_2d(periodic_basis)
        gram = periodic_basis @ periodic_basis.T
        periodic_dual = periodic_basis.T @ np.linalg.inv(gram)
        rank = len(periodic_basis)
        neighbor_hkls = np.asarray(
            list(itertools.product((-1, 0, 1), repeat=rank)), dtype=float
        )
        return MinimumImageContext(
            cell,
            pbc,
            inv_cell,
            orthogonal,
            neighbor_vectors=neighbor_hkls @ periodic_basis,
            periodic_basis=periodic_basis,
            periodic_dual=periodic_dual,
        )
    if orthogonal:
        return MinimumImageContext(cell, pbc, inv_cell, orthogonal)

    reduced_cell, _operation = minkowski_reduce(cell, pbc=True)
    reduced_cell = np.asarray(reduced_cell, dtype=float)
    return MinimumImageContext(
        cell,
        pbc,
        inv_cell,
        False,
        reduced_cell=reduced_cell,
        reduced_inv_cell=np.linalg.inv(reduced_cell),
        neighbor_vectors=_MIC_NEIGHBOR_HKLS @ reduced_cell,
    )


def minimum_image_context(
    cell: np.ndarray,
    pbc: np.ndarray | tuple[bool, bool, bool] | bool = True,
) -> MinimumImageContext:
    """Build or reuse an exact minimum-image context for a valid 3D cell."""
    normalized_cell = np.ascontiguousarray(cell, dtype=np.float64)
    normalized_pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=np.bool_)
    return _minimum_image_context(normalized_cell.tobytes(), normalized_pbc.tobytes())


def minimum_image_delta(
    delta: np.ndarray,
    context: MinimumImageContext,
) -> np.ndarray:
    """Return exact minimum images for scalar or batched Cartesian deltas."""
    delta_array = np.asarray(delta, dtype=float)
    original_shape = delta_array.shape
    flat_delta = delta_array.reshape(-1, 3)

    if context.orthogonal:
        fractional = flat_delta @ context.inv_cell
        fractional[:, context.pbc] -= np.round(fractional[:, context.pbc])
        result = fractional @ context.cell
    elif bool(np.all(context.pbc)):
        assert context.reduced_cell is not None
        assert context.reduced_inv_cell is not None
        assert context.neighbor_vectors is not None
        reduced_frac = flat_delta @ context.reduced_inv_cell
        reduced_frac -= np.floor(reduced_frac)
        wrapped = reduced_frac @ context.reduced_cell
        candidates = wrapped[:, None, :] + context.neighbor_vectors[None, :, :]
        squared_lengths = np.einsum("...i,...i->...", candidates, candidates)
        indices = np.argmin(squared_lengths, axis=1)
        result = candidates[np.arange(len(indices)), indices]
    else:
        assert context.periodic_basis is not None
        assert context.periodic_dual is not None
        assert context.neighbor_vectors is not None
        lattice_coordinates = flat_delta @ context.periodic_dual
        base = np.floor(lattice_coordinates) @ context.periodic_basis
        wrapped = flat_delta - base
        candidates = wrapped[:, None, :] - context.neighbor_vectors[None, :, :]
        squared_lengths = np.einsum("...i,...i->...", candidates, candidates)
        indices = np.argmin(squared_lengths, axis=1)
        result = candidates[np.arange(len(indices)), indices]
    return result.reshape(original_shape)


def scaled_positions(structure, positions: np.ndarray | None = None, *, wrap: bool = False) -> np.ndarray:
    """Return fractional coordinates without ASE's per-call linear solve overhead."""
    cart = np.asarray(structure.positions if positions is None else positions, dtype=float)
    cell = np.asarray(structure.cell.array, dtype=float)
    scaled = cart @ np.linalg.inv(cell)
    if wrap:
        scaled[:, np.asarray(structure.pbc, dtype=bool)] %= 1.0
    return scaled


def wrapped_positions(structure, positions: np.ndarray) -> np.ndarray:
    """Wrap Cartesian positions through fractional coordinates."""
    if not np.any(structure.pbc):
        return np.asarray(positions, dtype=float)
    return scaled_positions(structure, positions, wrap=True) @ np.asarray(structure.cell.array, dtype=float)
