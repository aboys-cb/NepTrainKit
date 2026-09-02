"""Dataset-owned immutable geometry snapshots for repeated analysis."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence, TypeVar

import numpy as np
from ase.data import atomic_numbers


_T = TypeVar("_T")


@dataclass(frozen=True)
class GeometrySnapshot:
    """Contiguous geometry arrays aligned with ``source_indices``."""

    source_indices: np.ndarray
    positions: np.ndarray
    atom_offsets: np.ndarray
    cells: np.ndarray
    pbc: np.ndarray
    atomic_numbers: np.ndarray

    def __post_init__(self) -> None:
        for array in (
            self.source_indices,
            self.positions,
            self.atom_offsets,
            self.cells,
            self.pbc,
            self.atomic_numbers,
        ):
            array.setflags(write=False)

    @property
    def structure_count(self) -> int:
        return int(self.source_indices.size)

    @property
    def atom_count(self) -> int:
        return int(self.positions.shape[0])


def structure_cell_array(
    structure: Any,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Return contiguous cell data for ASE and plain-array structures."""
    cell = structure.cell
    storage = getattr(cell, "array", cell)
    return np.ascontiguousarray(storage, dtype=dtype)


def structure_pbc_flags(structure: Any) -> np.ndarray:
    fields = getattr(structure, "additional_fields", {}) or {}
    if "pbc" in fields:
        value = fields["pbc"]
    else:
        value = getattr(structure, "pbc", "T T T")
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if len(tokens) == 1:
            tokens *= 3
        if len(tokens) != 3:
            raise ValueError("A structure has invalid PBC metadata.")
        mapping = {
            "t": 1,
            "true": 1,
            "1": 1,
            "yes": 1,
            "f": 0,
            "false": 0,
            "0": 0,
            "no": 0,
        }
        try:
            return np.asarray([mapping[token.lower()] for token in tokens], dtype=np.uint8)
        except KeyError as exc:
            raise ValueError("A structure has invalid PBC metadata.") from exc
    flags = np.asarray(value, dtype=np.uint8).reshape(-1)
    if flags.size == 1:
        flags = np.repeat(flags, 3)
    if flags.size != 3:
        raise ValueError("A structure has invalid PBC metadata.")
    return np.ascontiguousarray(flags, dtype=np.uint8)


class StructureGeometryCache:
    """Cache one immutable full snapshot and the most recent projection.

    Mask changes only alter the requested ``source_indices``. The underlying
    geometry snapshot remains valid for the lifetime of its ``StructureData``.
    """

    def __init__(self, structures: Sequence[Any]) -> None:
        self._structures = tuple(structures)
        self._lock = threading.Lock()
        self._all_snapshot: GeometrySnapshot | None = None
        self._projected_key: tuple[int, ...] | None = None
        self._projected_snapshot: GeometrySnapshot | None = None
        self._analysis_lock = threading.Lock()
        self._analysis_results: dict[tuple[str, Hashable], Any] = {}

    def snapshot(self, source_indices: Sequence[int] | np.ndarray | None = None) -> GeometrySnapshot:
        """Return a contiguous snapshot for all or selected source rows."""
        with self._lock:
            full = self._all_snapshot
            if full is None:
                full = self._build_all()
                self._all_snapshot = full
            if source_indices is None:
                return full
            indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= len(self._structures)):
                raise IndexError("Geometry snapshot indices are outside the dataset.")
            if np.array_equal(indices, full.source_indices):
                return full
            key = tuple(int(index) for index in indices)
            if key == self._projected_key and self._projected_snapshot is not None:
                return self._projected_snapshot
            projected = self._project(full, indices)
            self._projected_key = key
            self._projected_snapshot = projected
            return projected

    def analysis_result(
        self,
        namespace: str,
        key: Hashable,
        build: Callable[[], _T],
    ) -> tuple[_T, bool]:
        """Return one dataset-owned derived result and whether it was cached."""
        cache_key = (str(namespace), key)
        with self._analysis_lock:
            if cache_key in self._analysis_results:
                return self._analysis_results[cache_key], True
            result = build()
            self._analysis_results[cache_key] = result
            return result, False

    def _build_all(self) -> GeometrySnapshot:
        positions: list[np.ndarray] = []
        cells: list[np.ndarray] = []
        pbc: list[np.ndarray] = []
        numbers: list[np.ndarray] = []
        atom_counts = np.empty(len(self._structures), dtype=np.int64)
        for row, structure in enumerate(self._structures):
            symbols = tuple(str(symbol) for symbol in structure.elements)
            frame_positions = np.ascontiguousarray(structure.positions, dtype=np.float32)
            frame_cell = structure_cell_array(structure, dtype=np.float32)
            if frame_positions.shape != (len(symbols), 3) or frame_cell.shape != (3, 3):
                raise ValueError("A structure has invalid positions or cell data.")
            try:
                frame_numbers = np.asarray(
                    [atomic_numbers[symbol] for symbol in symbols], dtype=np.int16
                )
            except KeyError as exc:
                raise ValueError("A structure contains an unknown element.") from exc
            positions.append(frame_positions)
            cells.append(frame_cell)
            pbc.append(structure_pbc_flags(structure))
            numbers.append(frame_numbers)
            atom_counts[row] = len(symbols)
        atom_offsets = np.empty(len(self._structures) + 1, dtype=np.int64)
        atom_offsets[0] = 0
        np.cumsum(atom_counts, out=atom_offsets[1:])
        return GeometrySnapshot(
            source_indices=np.arange(len(self._structures), dtype=np.int64),
            positions=(
                np.concatenate(positions, axis=0)
                if positions
                else np.empty((0, 3), dtype=np.float32)
            ),
            atom_offsets=atom_offsets,
            cells=np.stack(cells) if cells else np.empty((0, 3, 3), dtype=np.float32),
            pbc=np.stack(pbc) if pbc else np.empty((0, 3), dtype=np.uint8),
            atomic_numbers=(
                np.concatenate(numbers)
                if numbers
                else np.empty(0, dtype=np.int16)
            ),
        )

    @staticmethod
    def _project(full: GeometrySnapshot, indices: np.ndarray) -> GeometrySnapshot:
        counts = full.atom_offsets[indices + 1] - full.atom_offsets[indices]
        offsets = np.empty(indices.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        position_parts = [
            full.positions[int(full.atom_offsets[index]) : int(full.atom_offsets[index + 1])]
            for index in indices
        ]
        number_parts = [
            full.atomic_numbers[int(full.atom_offsets[index]) : int(full.atom_offsets[index + 1])]
            for index in indices
        ]
        return GeometrySnapshot(
            source_indices=np.ascontiguousarray(indices, dtype=np.int64),
            positions=(
                np.concatenate(position_parts, axis=0)
                if position_parts
                else np.empty((0, 3), dtype=np.float32)
            ),
            atom_offsets=offsets,
            cells=np.ascontiguousarray(full.cells[indices], dtype=np.float32),
            pbc=np.ascontiguousarray(full.pbc[indices], dtype=np.uint8),
            atomic_numbers=(
                np.concatenate(number_parts)
                if number_parts
                else np.empty(0, dtype=np.int16)
            ),
        )
