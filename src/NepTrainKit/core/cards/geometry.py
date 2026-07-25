"""Small geometry helpers shared by Make Dataset card operations."""

from __future__ import annotations

import numpy as np


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
