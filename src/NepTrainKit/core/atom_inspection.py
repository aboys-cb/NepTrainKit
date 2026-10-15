"""Per-atom values shown when hovering an atom in the structure viewer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase.geometry import find_mic, minkowski_reduce

from NepTrainKit.core.geometry_cache import structure_cell_array, structure_pbc_flags

# Display-only vectors injected by Dataset Display for arrow overlays.
DFT_FORCE_KEY = "__ntk_dft_force"
ML_FORCE_KEY = "__ntk_ml_force"
DFT_MFORCE_KEY = "__ntk_dft_mforce"
ML_MFORCE_KEY = "__ntk_ml_mforce"
# Raw extxyz fields shown under a named row instead of "other".  ``force_mag``
# and ``mforce`` are the DFT magnetic-force labels read by the NEP/TACE loaders.
_DFT_FORCE_FIELDS = ("force", "forces")
_DFT_MFORCE_FIELDS = ("force_mag", "mforce")
_HANDLED_KEYS = {"species", "species_id", "pos", "spin", *_DFT_FORCE_FIELDS, *_DFT_MFORCE_FIELDS}
_INTERNAL_PREFIX = "__ntk_"


@dataclass(frozen=True)
class AtomInspection:
    """Values for one atom; vectors are ``None`` when the frame lacks them."""

    index: int
    element: str
    position: np.ndarray
    fractional: np.ndarray | None = None
    dft_force: np.ndarray | None = None
    ml_force: np.ndarray | None = None
    spin: np.ndarray | None = None
    dft_mforce: np.ndarray | None = None
    ml_mforce: np.ndarray | None = None
    nearest_distance: float | None = None
    nearest_index: int | None = None
    nearest_element: str | None = None
    # Remaining per-atom fields, e.g. ("charge", (0.31,)).
    extra: tuple[tuple[str, tuple[object, ...]], ...] = field(default_factory=tuple)

    @property
    def force_error(self) -> float | None:
        """Return |F_ML - F_DFT| when both forces are available."""
        if self.dft_force is None or self.ml_force is None:
            return None
        return float(np.linalg.norm(self.ml_force - self.dft_force))


def _vector(properties: dict, key: str, index: int, atom_count: int) -> np.ndarray | None:
    if key not in properties:
        return None
    try:
        values = np.asarray(properties[key], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape != (atom_count, 3):
        return None
    value = values[index]
    return value if np.isfinite(value).all() else None


def _nearest_neighbor(structure, index: int) -> tuple[float | None, int | None]:
    """Return the nearest other atom, using periodic images along PBC axes."""
    positions = np.asarray(structure.positions, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] < 1:
        return None, None
    delta = positions - positions[index]
    cell = structure_cell_array(structure, dtype=np.float64).reshape(3, 3)
    pbc = np.asarray(structure_pbc_flags(structure), dtype=bool).reshape(3)
    pbc &= np.any(cell != 0, axis=1)
    _, distances = find_mic(delta, cell, pbc=pbc)
    distances[index] = np.inf
    if pbc.any():
        # MIC removes the zero self image. A reduced basis also gives the
        # shortest nonzero self image, including highly skew periodic cells.
        reduced, _ = minkowski_reduce(cell, pbc=pbc)
        distances[index] = np.linalg.norm(reduced[pbc], axis=1).min()
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    if not np.isfinite(best_distance):
        return None, None
    return best_distance, best_index


def _fractional(structure, position: np.ndarray) -> np.ndarray | None:
    cell = structure_cell_array(structure, dtype=np.float64).reshape(3, 3)
    try:
        if abs(np.linalg.det(cell)) < 1e-12:
            return None
        return np.linalg.solve(cell.T, position)
    except np.linalg.LinAlgError:
        return None


def _extra_fields(properties: dict, index: int, atom_count: int):
    extra = []
    for key, values in properties.items():
        if key in _HANDLED_KEYS or str(key).startswith(_INTERNAL_PREFIX):
            continue
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != atom_count:
            continue
        row = np.atleast_1d(array[index])
        # Equal values do not imply equal physical meaning (e.g. zero force
        # and velocity); only the known field aliases above are consolidated.
        extra.append((str(key), tuple(row.tolist())))
    return tuple(extra)


def inspect_atom(structure, index: int) -> AtomInspection | None:
    """Collect the hover-card values for atom ``index`` of ``structure``."""
    atom_count = len(structure)
    if not 0 <= int(index) < atom_count:
        return None
    index = int(index)
    properties = getattr(structure, "atomic_properties", None) or {}
    position = np.asarray(structure.positions, dtype=np.float64).reshape(-1, 3)[index]
    raw_elements = getattr(structure, "elements", None)
    elements = [] if raw_elements is None else [str(item) for item in raw_elements]
    element = str(elements[index]) if index < len(elements) else ""
    dft_force = _vector(properties, DFT_FORCE_KEY, index, atom_count)
    for key in _DFT_FORCE_FIELDS:
        if dft_force is None:
            dft_force = _vector(properties, key, index, atom_count)
    dft_mforce = _vector(properties, DFT_MFORCE_KEY, index, atom_count)
    for key in _DFT_MFORCE_FIELDS:
        if dft_mforce is None:
            dft_mforce = _vector(properties, key, index, atom_count)
    ml_force = _vector(properties, ML_FORCE_KEY, index, atom_count)
    spin = _vector(properties, "spin", index, atom_count)
    ml_mforce = _vector(properties, ML_MFORCE_KEY, index, atom_count)
    nearest_distance, nearest_index = _nearest_neighbor(structure, index)
    return AtomInspection(
        index=index,
        element=element,
        position=position,
        fractional=_fractional(structure, position),
        dft_force=dft_force,
        ml_force=ml_force,
        spin=spin,
        dft_mforce=dft_mforce,
        ml_mforce=ml_mforce,
        nearest_distance=nearest_distance,
        nearest_index=nearest_index,
        nearest_element=(
            str(elements[nearest_index])
            if nearest_index is not None and nearest_index < len(elements)
            else None
        ),
        extra=_extra_fields(properties, index, atom_count),
    )


def pick_front_atom(
    screen: np.ndarray,
    depth: np.ndarray,
    radius: np.ndarray,
    cursor: tuple[float, float],
    *,
    min_radius: float = 4.0,
) -> int | None:
    """Return the nearest-to-camera atom whose projected disc covers ``cursor``.

    ``depth`` follows the OpenGL convention: smaller values are closer.
    """
    screen = np.asarray(screen, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    radius = np.maximum(np.asarray(radius, dtype=np.float64).reshape(-1), min_radius)
    if screen.shape[0] == 0:
        return None
    offset = screen - np.asarray(cursor, dtype=np.float64)[:2]
    inside = np.sum(offset * offset, axis=1) <= radius * radius
    inside &= np.isfinite(depth) & np.all(np.isfinite(screen), axis=1)
    if not inside.any():
        return None
    candidates = np.flatnonzero(inside)
    return int(candidates[np.argmin(depth[candidates])])


__all__ = ["AtomInspection", "inspect_atom", "pick_front_atom"]
