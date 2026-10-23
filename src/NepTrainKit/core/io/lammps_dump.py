#!/usr/bin/env python
"""LAMMPS dump reading: probing, column plans, frames, and element resolution.

A text dump has two very different halves:

* a tiny header (``ITEM: TIMESTEP`` ... ``ITEM: ATOMS``) whose cell layout and
  column names change between LAMMPS styles and versions; and
* a large table of whitespace-separated atom rows.

This module keeps the flexible half in Python (``_read_header`` and
``build_plan``) and reads the bulk half as vectorised row blocks.  A native
block parser can later replace :func:`_parse_rows` without knowing anything
about column names: every scientific decision (scaled -> Cartesian, spin
reconstruction, unwrapping, element mapping) stays in this file.
"""
from __future__ import annotations

import gzip
import json
import mmap
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from ase.data import atomic_masses, chemical_symbols
from loguru import logger

from NepTrainKit.core.precision import get_storage_float_dtype
from NepTrainKit.core.structure import Structure, atomic_numbers
from NepTrainKit.paths import PathLike, as_path

__all__ = [
    "ColumnOverrides",
    "ColumnPlan",
    "DumpFrame",
    "DumpImportCancelled",
    "DumpProbe",
    "ElementSuggestion",
    "LammpsDumpError",
    "build_plan",
    "coordinate_triplets",
    "discover_element_map",
    "iter_dump_frames",
    "iter_dump_structures",
    "load_dump_structures",
    "probe_dump",
    "remember_dump_settings",
    "remembered_dump_settings",
    "suggest_element_mapping",
]

_ITEM_TIMESTEP = "ITEM: TIMESTEP"
_ITEM_NUMBER = "ITEM: NUMBER OF ATOMS"
_ITEM_BOX = "ITEM: BOX BOUNDS"
_ITEM_ATOMS = "ITEM: ATOMS"
_ITEM_ATOMS_BYTES = _ITEM_ATOMS.encode("ascii")
_ITEM_TIMESTEP_BYTES = _ITEM_TIMESTEP.encode("ascii")

#: Coordinate triplets in decreasing order of preference.
_COORDINATE_SETS: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("scaled", ("xs", "ys", "zs")),
    ("wrapped", ("x", "y", "z")),
    ("unwrapped", ("xu", "yu", "zu")),
    ("scaled_unwrapped", ("xsu", "ysu", "zsu")),
)
_BOUNDARY_TOKEN = re.compile(r"^[pfsmPFSM]{2}$")
_TILT_TOKENS = ("xy", "xz", "yz")
#: Default ``custom`` columns that carry bookkeeping rather than physics.
_BOOKKEEPING_COLUMNS = frozenset({"id", "mol", "proc", "procp1", "type", "element", "typelabel"})
_SPIN_COMPUTE_IDS = frozenset({"spin", "sp"})
_VELOCITY_TOKENS = ("vx", "vy", "vz")
_CHUNK_ROWS = 65536
#: Relative tolerance used when a LAMMPS ``Masses`` entry is matched to an element.
_MASS_TOLERANCE = 1e-3
_DATA_FILE = re.compile(r"(?:^data[._-].*|.*\.(?:data|lmp|lammps))$", re.IGNORECASE)
_INPUT_FILE = re.compile(r"(?:^in[._-].*|.*\.in)$", re.IGNORECASE)


class LammpsDumpError(ValueError):
    """Raised when a dump frame violates the documented data contract."""


class DumpImportCancelled(RuntimeError):
    """Raised when an interactive element-mapping request is cancelled."""


class _Cancelled(RuntimeError):
    """Internal signal used to unwind the block reader on cancellation."""


@dataclass(frozen=True)
class RawColumn:
    """One whitespace token of an atom row that must be materialised."""

    name: str
    token: int
    dtype: str  # "f8" | "i4" | "str"


@dataclass(frozen=True)
class ColumnPlan:
    """Declarative mapping from dump columns to scientific roles.

    The plan is plain data on purpose: a future native block parser can consume
    ``raw``/``n_tokens`` without understanding any LAMMPS column semantics.
    """

    signature: tuple[str, ...]
    n_tokens: int
    raw: tuple[RawColumn, ...]
    coord_mode: str | None = None
    coord_tokens: tuple[int, int, int] | None = None
    force_tokens: tuple[int, int, int] | None = None
    spin_mode: str | None = None
    spin_tokens: tuple[int, ...] | None = None
    species_token: int | None = None
    species_name: str | None = None
    type_token: int | None = None
    image_tokens: tuple[int, int, int] | None = None
    extras: tuple[tuple[str, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class DumpFrame:
    """One parsed dump snapshot with resolved species and Cartesian positions."""

    timestep: int
    lattice: np.ndarray
    pbc: str
    species: np.ndarray
    positions: np.ndarray
    forces: np.ndarray | None = None
    spins: np.ndarray | None = None
    extras: Mapping[str, np.ndarray] | None = None
    frame_index: int = 0

    def to_structure(self, *, keep_extra: bool = True, dtype: Any = None) -> Structure:
        """Convert the frame into a :class:`~NepTrainKit.core.structure.Structure`."""
        dtype = dtype or get_storage_float_dtype()
        properties: list[dict[str, Any]] = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ]
        atomic_properties: dict[str, np.ndarray] = {
            "species": self.species,
            "pos": np.asarray(self.positions, dtype=dtype),
        }
        if self.forces is not None:
            properties.append({"name": "forces", "type": "R", "count": 3})
            atomic_properties["forces"] = np.asarray(self.forces, dtype=dtype)
        if self.spins is not None:
            properties.append({"name": "spin", "type": "R", "count": 3})
            atomic_properties["spin"] = np.asarray(self.spins, dtype=dtype)
        if keep_extra and self.extras:
            for name, values in self.extras.items():
                array = np.asarray(values, dtype=dtype)
                count = 1 if array.ndim == 1 else int(array.shape[1])
                properties.append({"name": name, "type": "R", "count": count})
                atomic_properties[name] = array
        return Structure(
            lattice=self.lattice,
            atomic_properties=atomic_properties,
            properties=properties,
            additional_fields={
                "Config_type": f"LAMMPS_{self.timestep}",
                "pbc": self.pbc,
            },
        )


@dataclass(frozen=True)
class DumpProbe:
    """Cheap metadata about a dump file, gathered without parsing every atom row."""

    path: Path
    frame_count: int
    declared_frame_count: int
    atom_counts: tuple[int, ...]
    timesteps: tuple[int, ...]
    signatures: tuple[tuple[str, ...], ...]
    columns: tuple[str, ...]
    types: tuple[int, ...]
    species_source: str | None
    cell_styles: tuple[str, ...]
    truncated: bool
    compressed: str | None
    lattice: tuple[float, ...] = ()
    pbc: str = "T T T"

    @property
    def needs_element_mapping(self) -> bool:
        """Return ``True`` when the dump only carries numeric atom types."""
        return self.species_source == "type" and bool(self.types)


@dataclass(frozen=True)
class ElementSuggestion:
    """Probed dump plus the element mapping discovered next to it."""

    probe: DumpProbe
    mapping: Mapping[int, str]
    missing: tuple[int, ...]


@dataclass(frozen=True)
class _Header:
    timestep: int
    n_atoms: int
    lattice: np.ndarray
    origin: np.ndarray
    pbc: str
    cell_style: str
    columns: tuple[str, ...]


def _is_cancelled(cancel_event: Any) -> bool:
    return bool(cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set())


def _warn(message: str, on_warning: Callable[[str], None] | None) -> None:
    if on_warning is not None:
        on_warning(message)
    else:
        logger.warning(message)


def _normalise_element_map(raw: Mapping[Any, Any] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for key, value in (raw or {}).items():
        try:
            type_id = int(key)
        except (TypeError, ValueError):
            continue
        symbol = str(value).strip()
        if type_id > 0 and symbol:
            mapping[type_id] = symbol
    return mapping


@dataclass(frozen=True)
class ColumnOverrides:
    """User overrides for how dump columns are interpreted.

    ``None`` means "detect automatically" and empty tuples mean "off". Names
    (not token indices) keep overrides valid when a later frame reorders or
    drops columns.
    """

    coordinate_names: tuple[str, str, str] | None = None
    force_names: tuple[str, str, str] | None = None
    spin_mode: str | None = None
    spin_names: tuple[str, ...] = ()
    ignored_extras: frozenset[str] = frozenset()
    extra_names: tuple[tuple[str, str], ...] = ()
    custom_fields: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the override for the config store."""
        payload: dict[str, Any] = {}
        if self.coordinate_names is not None:
            payload["coordinates"] = list(self.coordinate_names)
        if self.force_names is not None:
            payload["forces"] = list(self.force_names)
        if self.spin_mode is not None:
            payload["spin_mode"] = self.spin_mode
        if self.spin_names:
            payload["spin"] = list(self.spin_names)
        if self.ignored_extras:
            payload["ignored"] = sorted(self.ignored_extras)
        if self.extra_names:
            payload["extra_names"] = dict(self.extra_names)
        if self.custom_fields:
            payload["custom_fields"] = [
                {"name": name, "columns": list(columns)} for name, columns in self.custom_fields
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ColumnOverrides | None:
        """Rebuild an override previously produced by :meth:`to_dict`."""
        if not isinstance(payload, dict):
            return None

        def names(key: str) -> tuple[str, ...] | None:
            raw = payload.get(key)
            if raw is None:
                return None
            if not isinstance(raw, (list, tuple)):
                return None
            return tuple(str(value) for value in raw)

        spin_mode = payload.get("spin_mode")
        if spin_mode not in (None, "magnitude_direction", "vector", "off"):
            spin_mode = None
        ignored = payload.get("ignored")
        ignored_set = (
            frozenset(str(value) for value in ignored)
            if isinstance(ignored, (list, tuple))
            else frozenset()
        )
        raw_names = payload.get("extra_names")
        extra_names = (
            tuple((str(key), str(value)) for key, value in raw_names.items())
            if isinstance(raw_names, dict)
            else ()
        )
        raw_fields = payload.get("custom_fields")
        custom_fields: list[tuple[str, tuple[str, ...]]] = []
        if isinstance(raw_fields, list):
            for entry in raw_fields:
                if not isinstance(entry, dict):
                    continue
                columns = entry.get("columns")
                if not isinstance(columns, (list, tuple)):
                    continue
                custom_fields.append(
                    (str(entry.get("name", "")), tuple(str(column) for column in columns))
                )
        return cls(
            coordinate_names=names("coordinates"),  # type: ignore[arg-type]
            force_names=names("forces"),  # type: ignore[arg-type]
            spin_mode=spin_mode,
            spin_names=names("spin") or (),
            ignored_extras=ignored_set,
            extra_names=extra_names,
            custom_fields=tuple(custom_fields),
        )


def coordinate_triplets(columns: Sequence[str]) -> list[tuple[str, tuple[str, str, str]]]:
    """Return every complete coordinate triplet available in ``columns``."""
    names = {str(name) for name in columns}
    return [(mode, triplet) for mode, triplet in _COORDINATE_SETS if all(name in names for name in triplet)]


def _coordinate_mode(names: Sequence[str]) -> str:
    trio = tuple(names)
    for mode, candidate in _COORDINATE_SETS:
        if trio == candidate:
            return mode
    return "wrapped"


def build_plan(columns: Sequence[str], overrides: ColumnOverrides | None = None) -> ColumnPlan:
    """Map raw dump column names onto coordinates, forces, spin and extras."""
    signature = tuple(str(name) for name in columns)
    index = {name: position for position, name in enumerate(signature)}

    coord_mode: str | None = None
    coord_tokens: tuple[int, int, int] | None = None
    for mode, triplet in _COORDINATE_SETS:
        if all(name in index for name in triplet):
            coord_mode = mode
            coord_tokens = tuple(index[name] for name in triplet)  # type: ignore[assignment]
            break
    if overrides is not None and overrides.coordinate_names:
        names = tuple(overrides.coordinate_names)
        if len(names) == 3 and all(name in index for name in names):
            coord_mode = _coordinate_mode(names)
            coord_tokens = tuple(index[name] for name in names)  # type: ignore[assignment]
        else:
            logger.debug("LAMMPS coordinate override is absent from this frame; using automatic coordinates")

    force_tokens: tuple[int, int, int] | None = None
    if all(name in index for name in ("fx", "fy", "fz")):
        force_tokens = tuple(index[name] for name in ("fx", "fy", "fz"))  # type: ignore[assignment]
    if overrides is not None and overrides.force_names is not None:
        names = tuple(overrides.force_names)
        if not names:
            force_tokens = None
        elif len(names) == 3 and all(name in index for name in names):
            force_tokens = tuple(index[name] for name in names)  # type: ignore[assignment]
        else:
            logger.debug("LAMMPS force override is absent from this frame; using automatic forces")

    spin_mode, spin_tokens = _detect_spin(index)
    if overrides is not None and overrides.spin_mode is not None:
        if overrides.spin_mode == "off":
            spin_mode, spin_tokens = None, None
        else:
            names = tuple(overrides.spin_names)
            expected = 4 if overrides.spin_mode == "magnitude_direction" else 3
            if len(names) == expected and all(name in index for name in names):
                spin_mode = overrides.spin_mode
                spin_tokens = tuple(index[name] for name in names)
            else:
                logger.debug("LAMMPS spin override is absent from this frame; using automatic spin")

    species_name = "element" if "element" in index else ("typelabel" if "typelabel" in index else None)
    species_token = index.get(species_name) if species_name else None
    type_token = index.get("type")
    image_tokens = (
        tuple(index[name] for name in ("ix", "iy", "iz"))
        if all(name in index for name in ("ix", "iy", "iz"))
        else None
    )

    role_dtypes: dict[str, str] = {}
    for tokens, dtype in (
        (coord_tokens, "f8"),
        (force_tokens, "f8"),
        (spin_tokens, "f8"),
        (image_tokens, "i4"),
    ):
        for token in tokens or ():
            role_dtypes[signature[token]] = dtype
    if species_token is not None:
        role_dtypes[signature[species_token]] = "str"
    if type_token is not None:
        role_dtypes[signature[type_token]] = "i4"

    consumed = set(coord_tokens or ()) | set(force_tokens or ()) | set(spin_tokens or ())
    consumed |= set(image_tokens or ())
    if species_token is not None:
        consumed.add(species_token)
    if type_token is not None:
        consumed.add(type_token)
    skipped = {name for _, triplet in _COORDINATE_SETS for name in triplet}
    skipped |= {"fx", "fy", "fz", "ix", "iy", "iz"} | set(_BOOKKEEPING_COLUMNS)
    ignored = set(overrides.ignored_extras) if overrides is not None else set()
    renames = dict(overrides.extra_names) if overrides is not None else {}
    reserved_outputs = {"species", "pos", "forces", "spin"}

    def output_name(auto: str, used: set[str]) -> str:
        candidate = renames.get(auto, auto).strip()
        if not candidate or candidate in reserved_outputs or candidate in used:
            candidate = auto
        used.add(candidate)
        return candidate

    used_outputs: set[str] = set()
    custom_entries: list[tuple[str, tuple[int, ...]]] = []
    custom_tokens: set[int] = set()
    custom_names: set[str] = set()
    if overrides is not None:
        for field_name, column_names in overrides.custom_fields:
            name = str(field_name).strip()
            names = tuple(str(column) for column in column_names)
            tokens = tuple(index[column] for column in names if column in index)
            if not name or not tokens or len(tokens) != len(names):
                continue
            if name in reserved_outputs or name in custom_names:
                continue
            if any(token in consumed or token in custom_tokens for token in tokens):
                continue
            custom_names.add(name)
            custom_tokens.update(tokens)
            custom_entries.append((name, tokens))
            for token in tokens:
                role_dtypes[signature[token]] = "f8"
    consumed |= custom_tokens

    extras: list[tuple[str, tuple[int, ...]]] = list(custom_entries)
    remaining = [
        name for position, name in enumerate(signature) if position not in consumed and name not in skipped
    ]
    if all(name in remaining for name in _VELOCITY_TOKENS):
        if "vel" not in ignored:
            velocity_tokens = tuple(index[name] for name in _VELOCITY_TOKENS)
            extras.append((output_name("vel", used_outputs), velocity_tokens))
            for token in velocity_tokens:
                role_dtypes[signature[token]] = "f8"
        remaining = [name for name in remaining if name not in _VELOCITY_TOKENS]
    for name in remaining:
        output = f"dump.{name}"
        if output in ignored:
            continue
        extras.append((output_name(output, used_outputs), (index[name],)))
        role_dtypes[name] = "f8"

    raw = tuple(
        RawColumn(name, index[name], dtype)
        for name, dtype in sorted(role_dtypes.items(), key=lambda item: index[item[0]])
    )
    return ColumnPlan(
        signature=signature,
        n_tokens=len(signature),
        raw=raw,
        coord_mode=coord_mode,
        coord_tokens=coord_tokens,
        force_tokens=force_tokens,
        spin_mode=spin_mode,
        spin_tokens=spin_tokens,
        species_token=species_token,
        species_name=species_name,
        type_token=type_token,
        image_tokens=image_tokens,
        extras=tuple(extras),
    )


def _detect_spin(index: Mapping[str, int]) -> tuple[str | None, tuple[int, ...] | None]:
    """Recognise spin columns without relying on a single compute name."""
    if all(name in index for name in ("spx", "spy", "spz")):
        return "vector", tuple(index[name] for name in ("spx", "spy", "spz"))
    grouped: dict[str, dict[int, str]] = {}
    pattern = re.compile(r"^c_(?P<compute>.+?)\[(?P<slot>\d+)\]$")
    for name in index:
        match = pattern.match(name)
        if match and match.group("compute").lower() in _SPIN_COMPUTE_IDS:
            grouped.setdefault(match.group("compute"), {})[int(match.group("slot"))] = name
    for slots in grouped.values():
        if all(slot in slots for slot in (1, 2, 3, 4)):
            return "magnitude_direction", tuple(index[slots[slot]] for slot in (1, 2, 3, 4))
        if not slots.get(1) and all(slot in slots for slot in (2, 3, 4)):
            return "vector", tuple(index[slots[slot]] for slot in (2, 3, 4))
        if not slots.get(4) and all(slot in slots for slot in (1, 2, 3)):
            return "vector", tuple(index[slots[slot]] for slot in (1, 2, 3))
    return None, None


def _read_header(next_line: Callable[[], str], marker: str) -> _Header:
    """Parse one frame header starting from its ``ITEM: TIMESTEP`` line."""
    if not marker.strip().startswith(_ITEM_TIMESTEP):
        raise LammpsDumpError("Truncated LAMMPS dump: missing timestep value")
    ts_line = next_line()
    if not ts_line:
        raise LammpsDumpError("Truncated LAMMPS dump: missing timestep value")
    try:
        timestep = int(ts_line.split()[0])
    except (IndexError, ValueError) as exc:
        raise LammpsDumpError(f"Invalid LAMMPS timestep value: {ts_line.strip()!r}") from exc

    count_header = next_line()
    if not count_header or not count_header.strip().startswith(_ITEM_NUMBER):
        raise LammpsDumpError(f"LAMMPS frame {timestep} is missing 'ITEM: NUMBER OF ATOMS'")
    count_line = next_line()
    if not count_line:
        raise LammpsDumpError(f"LAMMPS frame {timestep} is missing the atom count")
    try:
        n_atoms = int(count_line.split()[0])
    except (IndexError, ValueError) as exc:
        raise LammpsDumpError(f"Invalid LAMMPS atom count in frame {timestep}: {count_line.strip()!r}") from exc
    if n_atoms <= 0:
        raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid atom count {n_atoms}")

    box_header = next_line()
    if not box_header or not box_header.strip().startswith(_ITEM_BOX):
        raise LammpsDumpError(f"LAMMPS frame {timestep} is missing 'ITEM: BOX BOUNDS'")
    lattice, origin, pbc, cell_style = _read_box_bounds(box_header, next_line, timestep)

    atoms_header = next_line()
    if not atoms_header or not atoms_header.strip().startswith(_ITEM_ATOMS):
        raise LammpsDumpError(f"LAMMPS frame {timestep} is missing 'ITEM: ATOMS'")
    columns = tuple(atoms_header.strip().split()[2:])
    return _Header(timestep, n_atoms, lattice, origin, pbc, cell_style, columns)


def _read_box_bounds(
    header: str,
    next_line: Callable[[], str],
    timestep: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    tokens = header.strip().split()
    general = "abc" in tokens and "origin" in tokens
    if general:
        # ``dump_modify triclinic/general``: three lines of ``a b c origin``.
        rows = [next_line().split() for _ in range(3)]
        if any(len(row) < 4 for row in rows):
            raise LammpsDumpError(f"LAMMPS frame {timestep} has incomplete general BOX BOUNDS")
        try:
            lattice = np.asarray([[float(value) for value in row[:3]] for row in rows], dtype=np.float64)
            origin = np.asarray([float(rows[0][3]), float(rows[1][3]), float(rows[2][3])], dtype=np.float64)
        except ValueError as exc:
            raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid BOX BOUNDS") from exc
        if not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(origin)):
            raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid BOX BOUNDS")
        if float(np.linalg.det(lattice)) <= 0.0:
            raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid BOX BOUNDS")
        return lattice, origin, "T T T", "general"

    boundary = [token for token in tokens[3:] if _BOUNDARY_TOKEN.match(token)]
    if len(boundary) != 3:
        raise LammpsDumpError(f"LAMMPS frame {timestep} does not declare three BOX BOUNDS boundary flags")
    pbc = " ".join("T" if token[0].lower() == "p" else "F" for token in boundary)
    tilts = [token for token in tokens if token in _TILT_TOKENS]
    rows = [next_line().split() for _ in range(3)]
    try:
        if tilts:
            if any(len(row) < 3 for row in rows):
                raise LammpsDumpError(f"LAMMPS frame {timestep} has incomplete triclinic BOX BOUNDS")
            xlo_bound, xhi_bound, xy = (float(value) for value in rows[0][:3])
            ylo_bound, yhi_bound, xz = (float(value) for value in rows[1][:3])
            zlo, zhi, yz = (float(value) for value in rows[2][:3])
            xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
            xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max(0.0, yz)
            lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
            lattice = np.asarray([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=np.float64)
            style = "triclinic"
            origin = np.asarray([xlo, ylo, zlo], dtype=np.float64)
        else:
            if any(len(row) < 2 for row in rows):
                raise LammpsDumpError(f"LAMMPS frame {timestep} has incomplete orthogonal BOX BOUNDS")
            xlo, xhi = (float(value) for value in rows[0][:2])
            ylo, yhi = (float(value) for value in rows[1][:2])
            zlo, zhi = (float(value) for value in rows[2][:2])
            lattice = np.asarray(
                [[xhi - xlo, 0.0, 0.0], [0.0, yhi - ylo, 0.0], [0.0, 0.0, zhi - zlo]],
                dtype=np.float64,
            )
            style = "ortho"
            origin = np.asarray([xlo, ylo, zlo], dtype=np.float64)
    except ValueError as exc:
        raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid BOX BOUNDS") from exc
    lengths = np.diag(lattice)
    if not np.all(np.isfinite(lattice)) or float(np.min(lengths)) <= 0.0:
        raise LammpsDumpError(f"LAMMPS frame {timestep} has invalid BOX BOUNDS")
    return lattice, origin, pbc, style


class _BlockResult(NamedTuple):
    """Parsed atom block plus the first structural problem, if any."""

    raw: dict[str, np.ndarray]
    rows: int
    problem_row: int
    problem_width: int
    bad_value_row: int
    problem_at_eof: bool
    invalid_symbols: tuple[str, ...] | None = None


_BATCH_MAX_FRAMES = 64
_BATCH_MAX_CELLS = 8_000_000


@dataclass
class _PendingFrame:
    """One scanned frame waiting for its atom block to be parsed."""

    header: _Header
    plan: ColumnPlan
    data_start: int
    block_end: int
    is_last: bool
    result: _BlockResult | None = None


def _allocate_arrays(n_atoms: int, plan: ColumnPlan) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in plan.raw:
        if column.dtype == "str":
            arrays[column.name] = np.empty(n_atoms, dtype=object)
        elif column.dtype == "i4":
            arrays[column.name] = np.empty(n_atoms, dtype=np.int32)
        else:
            arrays[column.name] = np.empty(n_atoms, dtype=np.float64)
    return arrays


def _fill_from_table(table: np.ndarray, plan: ColumnPlan, arrays: dict[str, np.ndarray]) -> None:
    count = int(table.shape[0])
    for column in plan.raw:
        target = arrays[column.name]
        if column.dtype == "i4":
            target[:count] = table[:count, column.token].astype(np.int64)
        else:
            target[:count] = table[:count, column.token]


def _fill_from_parts(parts: list[list[str]], plan: ColumnPlan, arrays: dict[str, np.ndarray]) -> int:
    """Fill arrays from split rows; return the first row with a bad numeric token."""
    if not parts:
        return -1
    count = len(parts)
    cells = np.asarray(parts, dtype=object)
    for column in plan.raw:
        values = cells[:, column.token]
        target = arrays[column.name]
        if column.dtype == "str":
            target[:count] = values
            continue
        try:
            if column.dtype == "i4":
                target[:count] = values.astype(np.int64)
            else:
                target[:count] = values.astype(np.float64)
        except (TypeError, ValueError):
            for index in range(count):
                try:
                    float(values[index])
                except (TypeError, ValueError):
                    return index
            raise
    return -1


_NATIVE_DUMP: Any = None
_NATIVE_DUMP_READY = False


def _native_dump() -> Any:
    """Return the optional native block parser, respecting the disable switch."""
    global _NATIVE_DUMP, _NATIVE_DUMP_READY
    if _NATIVE_DUMP_READY:
        return _NATIVE_DUMP
    _NATIVE_DUMP_READY = True
    if os.environ.get("NEPKIT_DISABLE_FASTDUMP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        logger.debug("NEPKIT_DISABLE_FASTDUMP is set; using the Python dump block parser")
        return None
    try:
        from NepTrainKit._native import _dump as native

        if hasattr(native, "parse_block"):
            _NATIVE_DUMP = native
    except Exception:
        _NATIVE_DUMP = None
    return _NATIVE_DUMP


def _decode_native_result(output: Any, plan: ColumnPlan) -> _BlockResult:
    columns, rows, problem_row, problem_width, bad_value_row, problem_at_eof = output
    raw: dict[str, np.ndarray] = {}
    invalid_symbols: tuple[str, ...] | None = None
    for column, value in zip(plan.raw, columns):
        if column.dtype == "str":
            ids, table = value
            if plan.species_name == column.name:
                invalid_symbols = tuple(label for label in table if label not in atomic_numbers)
            if rows:
                symbols = np.asarray(table, dtype=np.str_)[ids[:rows]]
            else:
                symbols = np.empty(0, dtype=np.str_)
            raw[column.name] = symbols
        else:
            raw[column.name] = value[:rows]
    return _BlockResult(
        raw,
        int(rows),
        int(problem_row),
        int(problem_width),
        int(bad_value_row),
        bool(problem_at_eof),
        invalid_symbols,
    )


def _parse_block_native(native: Any, view: memoryview, n_atoms: int, plan: ColumnPlan) -> _BlockResult:
    tokens = [column.token for column in plan.raw]
    kinds = [2 if column.dtype == "str" else 1 if column.dtype == "i4" else 0 for column in plan.raw]
    return _decode_native_result(
        native.parse_block(view, n_atoms, plan.n_tokens, tokens, kinds), plan
    )


def _dump_workers(requested: int) -> int:
    """Resolve the frame-parallel worker count, honouring the debug override."""
    raw = os.environ.get("NEPKIT_FASTDUMP_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return int(requested)


def _parse_pending(buffer: Any, pending: list[_PendingFrame], max_workers: int) -> None:
    """Parse a same-signature batch, in parallel when the native parser is present."""
    native = _native_dump()
    plan = pending[0].plan
    if native is not None and hasattr(native, "parse_blocks"):
        tokens = [column.token for column in plan.raw]
        kinds = [2 if column.dtype == "str" else 1 if column.dtype == "i4" else 0 for column in plan.raw]
        try:
            outputs = native.parse_blocks(
                buffer,
                [item.data_start for item in pending],
                [item.block_end for item in pending],
                [item.header.n_atoms for item in pending],
                plan.n_tokens,
                tokens,
                kinds,
                max_workers,
            )
        except (RuntimeError, ValueError) as exc:
            raise LammpsDumpError(str(exc)) from exc
        for item, output in zip(pending, outputs):
            item.result = _decode_native_result(output, item.plan)
        return
    for item in pending:
        view = memoryview(buffer)[item.data_start:item.block_end]
        try:
            item.result = _parse_block(view, item.header.n_atoms, item.plan)
        finally:
            view.release()


def _parse_block_text(view: memoryview, n_atoms: int, plan: ColumnPlan) -> _BlockResult:
    text = bytes(view).decode("utf8", "ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    problem_row = -1
    problem_width = 0
    extra_rows = len(lines) > n_atoms
    if extra_rows:
        problem_row = n_atoms
        problem_width = -1
        lines = lines[:n_atoms]
    arrays = _allocate_arrays(n_atoms, plan)
    if not lines:
        return _BlockResult(arrays, 0, problem_row, problem_width, -1, False)

    if not any(column.dtype == "str" for column in plan.raw):
        values = np.fromstring(text, sep=" ", dtype=np.float64)
        if values.size == len(lines) * plan.n_tokens:
            table = values.reshape(len(lines), plan.n_tokens)
            _fill_from_table(table, plan, arrays)
            return _BlockResult(arrays, len(lines), problem_row, problem_width, -1, False)

    parts: list[list[str]] = []
    for index, line in enumerate(lines):
        part = line.split()
        if len(part) != plan.n_tokens:
            problem_row, problem_width = index, len(part)
            break
        parts.append(part)
    bad_row = _fill_from_parts(parts, plan, arrays)
    at_eof = (
        not extra_rows
        and bad_row < 0
        and problem_row >= 0
        and all(not line.strip() for line in lines[problem_row + 1:])
    )
    return _BlockResult(arrays, len(parts), problem_row, problem_width, bad_row, at_eof)


def _parse_block(view: memoryview, n_atoms: int, plan: ColumnPlan) -> _BlockResult:
    native = _native_dump()
    if native is not None:
        try:
            return _parse_block_native(native, view, n_atoms, plan)
        except (RuntimeError, ValueError) as exc:
            raise LammpsDumpError(str(exc)) from exc
    return _parse_block_text(view, n_atoms, plan)


def _apply_unwrap(positions: np.ndarray, images: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return positions + images.astype(np.float64) @ lattice


def _assemble_frame(
    header: _Header,
    plan: ColumnPlan,
    raw: dict[str, np.ndarray],
    species: np.ndarray,
    *,
    unwrap: bool,
    dtype: Any = None,
) -> DumpFrame:
    dtype = dtype or get_storage_float_dtype()
    signature = plan.signature

    if plan.coord_tokens is None:
        raise LammpsDumpError(
            f"LAMMPS frame {header.timestep} must contain one complete coordinate triplet: "
            "xs/ys/zs, x/y/z, xu/yu/zu, or xsu/ysu/zsu"
        )
    vectors = [raw[signature[token]].astype(np.float64) for token in plan.coord_tokens]
    lattice64 = np.asarray(header.lattice, dtype=np.float64)
    if plan.coord_mode in {"scaled", "scaled_unwrapped"}:
        positions = (
            vectors[0][:, None] * lattice64[0]
            + vectors[1][:, None] * lattice64[1]
            + vectors[2][:, None] * lattice64[2]
        )
        origin = np.zeros(3, dtype=np.float64)
    else:
        positions = np.stack(vectors, axis=1)
        origin = np.asarray(header.origin, dtype=np.float64)
    positions = positions - origin
    if unwrap and plan.image_tokens is not None and plan.coord_mode == "wrapped":
        images = np.stack([raw[signature[token]] for token in plan.image_tokens], axis=1)
        positions = _apply_unwrap(positions, images, lattice64)

    forces = None
    if plan.force_tokens is not None:
        forces = np.stack([raw[signature[token]] for token in plan.force_tokens], axis=1).astype(dtype)

    spins = None
    if plan.spin_tokens is not None:
        components = [raw[signature[token]].astype(np.float64) for token in plan.spin_tokens]
        if plan.spin_mode == "magnitude_direction":
            direction = np.stack(components[1:], axis=1)
            spins = components[0][:, None] * direction
        else:
            spins = np.stack(components, axis=1)
        spins = np.asarray(spins, dtype=dtype)

    extras: dict[str, np.ndarray] = {}
    for name, tokens in plan.extras:
        if len(tokens) == 1:
            extras[name] = raw[signature[tokens[0]]].astype(dtype)
        else:
            extras[name] = np.stack([raw[signature[token]] for token in tokens], axis=1).astype(dtype)

    return DumpFrame(
        timestep=header.timestep,
        lattice=np.asarray(header.lattice, dtype=dtype),
        pbc=header.pbc,
        species=species,
        positions=np.asarray(positions, dtype=dtype),
        forces=forces,
        spins=spins,
        extras=extras or None,
    )


def _resolve_species(
    header: _Header,
    plan: ColumnPlan,
    raw: dict[str, np.ndarray],
    mapping: Mapping[int, str],
    invalid_symbols: tuple[str, ...] | None = None,
) -> np.ndarray:
    invalid: list[str]
    if plan.species_token is not None:
        values = raw[plan.signature[plan.species_token]]
        symbols = values if values.dtype.kind == "U" else values.astype(np.str_)
        if invalid_symbols is None:
            used = np.unique(symbols).tolist()
            invalid = sorted(symbol for symbol in used if symbol not in atomic_numbers)
        else:
            invalid = sorted(invalid_symbols)
    
    elif plan.type_token is not None:
        types = raw[plan.signature[plan.type_token]]
        present = np.unique(types)
        unresolved = [int(value) for value in present.tolist() if int(value) not in mapping]
        if unresolved:
            raise LammpsDumpError(
                "LAMMPS numeric atom types require an explicit element mapping; "
                f"unresolved types: {unresolved}"
            )
        used = sorted({mapping[int(value)] for value in present.tolist()})
        width = max((len(symbol) for symbol in used), default=1)
        lookup = np.full(int(present.max()) + 1, "", dtype=f"<U{max(width, 1)}")
        for type_id, symbol in mapping.items():
            if 0 <= type_id < lookup.size:
                lookup[type_id] = symbol
        symbols = lookup[types]
        invalid = sorted(symbol for symbol in used if symbol not in atomic_numbers)
    else:
        raise LammpsDumpError(f"LAMMPS frame {header.timestep} must contain an element or type column")

    if invalid:
        raise LammpsDumpError(
            f"LAMMPS frame {header.timestep} contains invalid element symbols: {invalid}"
        )
    return symbols


def iter_dump_frames(
    path: PathLike,
    *,
    element_map: Mapping[Any, Any] | None = None,
    element_resolver: Callable[[list[int], dict[str, Any]], Mapping[int, str]] | None = None,
    column_overrides: ColumnOverrides | None = None,
    unwrap: bool = False,
    strict: bool = False,
    search_neighbours: bool = True,
    cancel_event: Any = None,
    on_warning: Callable[[str], None] | None = None,
    max_workers: int = -1,
) -> Iterator[DumpFrame]:
    """Yield parsed frames from a LAMMPS dump file.

    Parameters
    ----------
    path : str or os.PathLike
        Dump file to read; ``.gz`` files are decompressed transparently.
    element_map : mapping, optional
        Explicit ``{atom type: element}`` mapping.
    element_resolver : callable, optional
        Fallback used when a numeric type is still unresolved; receives the
        missing types and a context dictionary.
    column_overrides : ColumnOverrides, optional
        Manual column roles; omitted fields stay on automatic detection.
    unwrap : bool, default False
        Add the ``ix/iy/iz`` image flags to wrapped coordinates.
    strict : bool, default False
        Raise on the first malformed or truncated frame instead of dropping a
        damaged trailing frame once at least one complete frame was read.
    search_neighbours : bool, default True
        Look for ``Atom Type Labels``/``Masses``/``labelmap`` hints in the
        dump directory before asking ``element_resolver``.
    cancel_event : threading.Event, optional
        Stop reading when the event is set.
    on_warning : callable, optional
        Receives non-fatal diagnostics instead of the module logger.
    max_workers : int, default -1
        Frame-parallel worker count for the native parser; ``-1`` uses all
        cores. Frames are still yielded in file order.

    Yields
    ------
    DumpFrame
        One parsed frame in file order.

    """
    candidate = as_path(path)
    mapping = _normalise_element_map(element_map)
    plans: dict[tuple[str, ...], ColumnPlan] = {}
    emitted = 0
    resolved_types: set[int] = set()
    discovered = False
    stop = False
    workers = _dump_workers(max_workers)
    storage_dtype = get_storage_float_dtype()
    pending: list[_PendingFrame] = []
    pending_cells = 0
    buffer, close = _open_read_buffer(candidate)

    def consume(batch: list[_PendingFrame]):
        nonlocal emitted, discovered, stop
        for item in batch:
            header = item.header
            plan = item.plan
            result = item.result
            if result is None:
                raise LammpsDumpError(f"LAMMPS frame {header.timestep} was not parsed")
            if result.bad_value_row >= 0:
                raise LammpsDumpError(
                    f"LAMMPS frame {header.timestep}, atom row {result.bad_value_row + 1} "
                    "contains a non-numeric value"
                )
            if result.problem_row >= 0 and not (item.is_last and result.problem_at_eof):
                if result.problem_width < 0:
                    raise LammpsDumpError(
                        f"LAMMPS frame {header.timestep} contains more atom rows than the "
                        f"declared {header.n_atoms}"
                    )
                raise LammpsDumpError(
                    f"LAMMPS frame {header.timestep}, atom row {result.problem_row + 1} has "
                    f"{result.problem_width} columns; expected {plan.n_tokens}"
                )
            if result.rows < header.n_atoms:
                message = (
                    f"LAMMPS frame {header.timestep} is truncated: expected "
                    f"{header.n_atoms} atom rows, found {result.rows}"
                )
                if strict or emitted == 0 or not item.is_last:
                    raise LammpsDumpError(message)
                _warn(f"{message}; dropping the incomplete trailing frame", on_warning)
                stop = True
                return
            raw = result.raw
            if plan.type_token is not None and plan.species_token is None:
                pending_types = {
                    int(value)
                    for value in raw[plan.signature[plan.type_token]].tolist()
                    if int(value) not in mapping
                }
                if pending_types and search_neighbours and not discovered:
                    discovered = True
                    hints = discover_element_map(
                        candidate.parent, natoms=header.n_atoms, exclude=candidate
                    )
                    for type_id, symbol in hints.items():
                        mapping.setdefault(type_id, symbol)
                    pending_types = {type_id for type_id in pending_types if type_id not in mapping}
                fresh = sorted(pending_types - resolved_types)
                if fresh and callable(element_resolver):
                    extra = element_resolver(
                        fresh,
                        {
                            "path": candidate,
                            "n_atoms": header.n_atoms,
                            "timestep": header.timestep,
                            "present_types": fresh,
                        },
                    )
                    mapping.update(_normalise_element_map(extra))
                    resolved_types.update(fresh)
            species = _resolve_species(header, plan, raw, mapping, result.invalid_symbols)
            yield _assemble_frame(header, plan, raw, species, unwrap=unwrap, dtype=storage_dtype)
            emitted += 1

    def flush():
        nonlocal pending, pending_cells
        if not pending:
            return
        batch = pending
        pending = []
        pending_cells = 0
        _parse_pending(buffer, batch, workers)
        yield from consume(batch)

    try:
        offset = _next_frame_offset(buffer, 0)
        if offset < 0:
            raise LammpsDumpError(f"{candidate.name} does not contain a LAMMPS 'ITEM: TIMESTEP' header")
        while offset >= 0 and not stop:
            if _is_cancelled(cancel_event):
                return
            next_offset = _next_frame_offset(buffer, offset + len(_ITEM_TIMESTEP_BYTES))
            is_last = next_offset < 0
            block_end = len(buffer) if is_last else next_offset
            try:
                header, data_start = _header_from_buffer(buffer, offset)
                plan = plans.get(header.columns)
                if plan is None:
                    plan = build_plan(header.columns, column_overrides)
                    plans[header.columns] = plan
                if plan.coord_tokens is None:
                    raise LammpsDumpError(
                        f"LAMMPS frame {header.timestep} must contain one complete coordinate triplet: "
                        "xs/ys/zs, x/y/z, xu/yu/zu, or xsu/ysu/zsu"
                    )
                if plan.species_token is None and plan.type_token is None:
                    raise LammpsDumpError(f"LAMMPS frame {header.timestep} must contain an element or type column")
            except LammpsDumpError as exc:
                yield from flush()
                if stop:
                    return
                if strict or emitted == 0:
                    raise
                _warn(f"{exc}; stopping at the damaged LAMMPS frame", on_warning)
                return
            cells = header.n_atoms * plan.n_tokens
            if pending and (
                plan.signature != pending[-1].plan.signature
                or len(pending) >= _BATCH_MAX_FRAMES
                or pending_cells + cells > _BATCH_MAX_CELLS
            ):
                yield from flush()
                if stop:
                    return
            pending.append(_PendingFrame(header, plan, data_start, block_end, is_last))
            pending_cells += cells
            offset = next_offset
        yield from flush()
    finally:
        close()


def iter_dump_structures(path: PathLike, **kwargs: Any) -> Iterator[Structure]:
    """Yield :class:`Structure` objects parsed from a dump file."""
    keep_extra = bool(kwargs.pop("keep_extra", True))
    dtype = get_storage_float_dtype()
    for frame in iter_dump_frames(path, **kwargs):
        yield frame.to_structure(keep_extra=keep_extra, dtype=dtype)


def load_dump_structures(path: PathLike, **kwargs: Any) -> list[Structure]:
    """Load every frame of a dump file as a list of structures."""
    return list(iter_dump_structures(path, **kwargs))


_WHOLE_FILE_LIMIT = 64 * 1024 * 1024


def _open_read_buffer(path: Path) -> tuple[Any, Callable[[], None]]:
    if path.name.lower().endswith(".gz"):
        return gzip.decompress(path.read_bytes()), lambda: None
    size = path.stat().st_size
    if size == 0:
        return b"", lambda: None
    if size <= _WHOLE_FILE_LIMIT:
        # bytes.find scans ~2.5x faster than mmap.find on CPython, and small
        # dumps fit comfortably in memory.
        return path.read_bytes(), lambda: None
    handle = path.open("rb")
    try:
        buffer = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        handle.close()
        raise

    def close() -> None:
        buffer.close()
        handle.close()

    return buffer, close


def _next_frame_offset(buffer: Any, start: int) -> int:
    """Return the next byte offset that starts an ``ITEM: TIMESTEP`` line."""
    offset = buffer.find(_ITEM_TIMESTEP_BYTES, start)
    while offset > 0 and bytes(buffer[offset - 1:offset]) != b"\n":
        offset = buffer.find(_ITEM_TIMESTEP_BYTES, offset + 1)
    return offset


def _frame_offsets(buffer: Any) -> list[int]:
    marker = _ITEM_TIMESTEP.encode("ascii")
    offsets: list[int] = []
    position = 0
    size = len(buffer)
    while position < size:
        found = buffer.find(marker, position)
        if found < 0:
            break
        if found == 0 or buffer[found - 1:found] == b"\n":
            offsets.append(found)
        position = found + len(marker)
    return offsets


def _header_from_buffer(buffer: Any, offset: int) -> tuple[_Header, int]:
    """Parse one header and return the byte offset of its first atom row."""
    atoms_at = buffer.find(_ITEM_ATOMS_BYTES, offset)
    if atoms_at >= 0:
        line_end = buffer.find(b"\n", atoms_at)
        if line_end >= 0:
            lines = bytes(buffer[offset:line_end]).decode("utf8", "replace").splitlines()
            if lines:
                iterator = iter(lines[1:])
                return _read_header(lambda: next(iterator, ""), lines[0]), line_end + 1
    window = bytes(buffer[offset:offset + 4096]).decode("utf8", "replace").splitlines()
    if not window:
        raise LammpsDumpError("Truncated LAMMPS dump: missing timestep value")
    iterator = iter(window[1:])
    return _read_header(lambda: next(iterator, ""), window[0]), offset


def _count_newlines(buffer: Any, start: int, end: int, chunk: int = 1 << 20) -> int:
    total = 0
    position = start
    while position < end:
        stop = min(position + chunk, end)
        total += bytes(buffer[position:stop]).count(b"\n")
        position = stop
    return total


def probe_dump(path: PathLike, *, type_sample_limit: int = 50000) -> DumpProbe:
    """Return frame/column/type metadata without parsing the whole file."""
    candidate = as_path(path)
    buffer, close = _open_read_buffer(candidate)
    try:
        offsets = _frame_offsets(buffer)
        if not offsets:
            raise LammpsDumpError(f"{candidate.name} does not contain a LAMMPS 'ITEM: TIMESTEP' header")
        headers: list[_Header] = []
        broken_tail = False
        for offset in offsets:
            try:
                headers.append(_header_from_buffer(buffer, offset)[0])
            except LammpsDumpError:
                broken_tail = True
                break
        if not headers:
            raise LammpsDumpError(f"{candidate.name} does not contain a readable LAMMPS frame header")
        truncated = broken_tail
        if not broken_tail:
            last = offsets[-1]
            last_header = headers[-1]
            # The atom block of the last frame starts after its 'ITEM: ATOMS' header.
            atoms_header_offset = buffer.find(_ITEM_ATOMS.encode("ascii"), last)
            tail_offset = buffer.find(b"\n", atoms_header_offset) if atoms_header_offset >= 0 else -1
            if tail_offset >= 0:
                available = _count_newlines(buffer, tail_offset + 1, len(buffer))
                truncated = available < last_header.n_atoms
        complete = len(headers) - (1 if truncated and not broken_tail else 0)

        signatures: list[tuple[str, ...]] = []
        for header in headers:
            if header.columns not in signatures:
                signatures.append(header.columns)
        columns: list[str] = []
        for signature in signatures:
            for name in signature:
                if name not in columns:
                    columns.append(name)
        first_columns = headers[0].columns
        species_source = next(
            (source for source in ("element", "typelabel", "type") if source in first_columns),
            None,
        )
        types = _sample_types(buffer, offsets[0], headers[0], type_sample_limit) if species_source == "type" else ()
        styles: list[str] = []
        for header in headers:
            if header.cell_style not in styles:
                styles.append(header.cell_style)
        return DumpProbe(
            path=candidate,
            frame_count=max(complete, 0),
            declared_frame_count=len(offsets),
            atom_counts=tuple(header.n_atoms for header in headers),
            timesteps=tuple(header.timestep for header in headers),
            signatures=tuple(signatures),
            columns=tuple(columns),
            types=types,
            species_source=species_source or None,
            cell_styles=tuple(styles),
            truncated=truncated,
            compressed="gz" if candidate.name.lower().endswith(".gz") else None,
            lattice=tuple(float(value) for value in headers[0].lattice.reshape(-1)),
            pbc=headers[0].pbc,
        )
    finally:
        close()


def _offset_after_rows(buffer: Any, start: int, rows: int, chunk: int = 1 << 20) -> int:
    """Return the byte offset just past ``rows`` newlines from ``start``."""
    remaining = rows
    position = start
    size = len(buffer)
    while remaining > 0 and position < size:
        stop = min(position + chunk, size)
        block = bytes(buffer[position:stop])
        count = block.count(b"\n")
        if count >= remaining:
            index = -1
            for _ in range(remaining):
                index = block.find(b"\n", index + 1)
            return position + index + 1
        remaining -= count
        position = stop
    return size


def _sample_types(buffer: Any, offset: int, header: _Header, limit: int) -> tuple[int, ...]:
    plan = build_plan(header.columns)
    if plan.type_token is None:
        return ()
    marker = buffer.find(b"\n", buffer.find(_ITEM_ATOMS_BYTES, offset))
    if marker < 0:
        return ()
    start = marker + 1
    rows = min(header.n_atoms, max(limit, 1))
    stop = _offset_after_rows(buffer, start, rows)
    mini = ColumnPlan(
        signature=plan.signature,
        n_tokens=plan.n_tokens,
        raw=(RawColumn("type", plan.type_token, "i4"),),
        type_token=plan.type_token,
    )
    view = memoryview(buffer)[start:stop]
    try:
        result = _parse_block(view, rows, mini)
    finally:
        view.release()
    found = np.unique(result.raw["type"][:result.rows])
    return tuple(int(type_id) for type_id in found if type_id > 0)


def _candidate_files(directory: Path, *, limit_files: int, exclude: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return candidates
    for child in children:
        if len(candidates) >= limit_files:
            break
        if not child.is_file() or child == exclude:
            continue
        name = child.name
        if not (_DATA_FILE.match(name) or _INPUT_FILE.match(name)):
            continue
        try:
            if child.stat().st_size > 20 * 1024 * 1024:
                continue
        except OSError:
            continue
        candidates.append(child)
    return candidates


def _read_small_text(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    try:
        with path.open("r", encoding="utf8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _section(body: str, name: str) -> list[str]:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == name.lower():
            rows: list[str] = []
            for following in lines[index + 1:]:
                stripped = following.strip()
                if not stripped:
                    if rows:
                        break
                    continue
                rows.append(stripped)
            return rows
    return []


def _parse_data_file(body: str) -> tuple[dict[int, str], dict[int, float], int | None, int | None]:
    labels: dict[int, str] = {}
    masses: dict[int, float] = {}
    for row in _section(body, "Atom Type Labels"):
        parts = row.split("#", 1)[0].split()
        if len(parts) >= 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
    for row in _section(body, "Masses"):
        parts = row.split("#", 1)[0].split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                masses[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    ntypes = None
    match = re.search(r"^\s*(\d+)\s+atom types\s*$", body, re.MULTILINE)
    if match:
        ntypes = int(match.group(1))
    natoms = None
    match = re.search(r"^\s*(\d+)\s+atoms\s*$", body, re.MULTILINE)
    if match:
        natoms = int(match.group(1))
    return labels, masses, ntypes, natoms


def _parse_input_script(body: str) -> tuple[dict[int, str], dict[int, float]]:
    labels: dict[int, str] = {}
    masses: dict[int, float] = {}
    for line in body.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        keyword = parts[0].lower()
        if keyword == "labelmap" and len(parts) >= 4 and parts[1].lower() == "atom":
            rest = parts[2:]
            # LAMMPS accepts both ``1 Fe 2 Si`` and ``1 2 Fe Si``.
            if len(rest) % 2 == 0 and all(token.isdigit() for token in rest[::2]):
                for type_token, label in zip(rest[::2], rest[1::2]):
                    labels[int(type_token)] = label
                continue
            split = next((index for index, token in enumerate(rest) if not token.isdigit()), None)
            if split is None or split == 0:
                continue
            type_tokens, label_tokens = rest[:split], rest[split:]
            if len(type_tokens) == len(label_tokens):
                for type_token, label in zip(type_tokens, label_tokens):
                    labels[int(type_token)] = label
        elif keyword == "mass" and len(parts) >= 3 and parts[1].isdigit():
            try:
                masses[int(parts[1])] = float(parts[2])
            except ValueError:
                continue
    return labels, masses


def _element_for_mass(mass: float) -> str | None:
    best: tuple[float, str] | None = None
    for number in range(1, len(atomic_masses)):
        reference = float(atomic_masses[number])
        if reference <= 0.0:
            continue
        relative = abs(mass - reference) / reference
        if relative <= _MASS_TOLERANCE and (best is None or relative < best[0]):
            best = (relative, chemical_symbols[number])
    return best[1] if best is not None else None


def discover_element_map(
    directory: Path,
    *,
    types: Sequence[int] = (),
    natoms: int | None = None,
    exclude: Path | None = None,
    limit_files: int = 24,
) -> dict[int, str]:
    """Read ``Atom Type Labels``/``Masses``/``mass``/``labelmap`` hints nearby.

    A data file is only trusted when its declared atom count matches the dump
    (or it shares the dump's filename stem); otherwise unrelated files in a
    shared folder - a Desktop, for example - would silently supply the wrong
    elements. Input scripts are trusted only when they mention the dump file
    or share its stem. Only hints that all trusted files agree on are kept.
    """
    wanted = {int(value) for value in types}
    agreed: dict[int, str] = {}
    conflicts: set[int] = set()
    for candidate in _candidate_files(directory, limit_files=limit_files, exclude=exclude):
        body = _read_small_text(candidate)
        if not body:
            continue
        same_stem = exclude is not None and candidate.stem == exclude.stem
        is_data = bool(_DATA_FILE.match(candidate.name))
        if is_data:
            labels, masses, ntypes, file_natoms = _parse_data_file(body)
            if wanted and ntypes is not None and max(wanted) > ntypes:
                continue
            if natoms is not None:
                if file_natoms is None and not same_stem:
                    continue
                if file_natoms is not None and file_natoms != natoms and not same_stem:
                    continue
        else:
            labels, masses = _parse_input_script(body)
            if not same_stem and (exclude is None or exclude.name not in body):
                continue
        found: dict[int, str] = {
            type_id: label for type_id, label in labels.items()
            if label in atomic_numbers and (not wanted or type_id in wanted)
        }
        for type_id, mass in masses.items():
            if type_id in found or (wanted and type_id not in wanted):
                continue
            symbol = _element_for_mass(mass)
            if symbol is not None:
                found[type_id] = symbol
        for type_id, symbol in found.items():
            if type_id in conflicts:
                continue
            previous = agreed.get(type_id)
            if previous is None:
                agreed[type_id] = symbol
            elif previous != symbol:
                conflicts.add(type_id)
    for type_id in conflicts:
        agreed.pop(type_id, None)
    return agreed


def suggest_element_mapping(
    path: PathLike,
    *,
    element_map: Mapping[Any, Any] | None = None,
    types: Sequence[int] | None = None,
) -> ElementSuggestion:
    """Probe a dump and combine explicit, discovered and probed type mappings."""
    candidate = as_path(path)
    probe = probe_dump(candidate)
    mapping = _normalise_element_map(element_map)
    relevant = tuple(types) if types is not None else probe.types
    if probe.needs_element_mapping:
        discovered = discover_element_map(
            candidate.parent,
            types=relevant,
            natoms=probe.atom_counts[0] if probe.atom_counts else None,
            exclude=candidate,
        )
        for type_id, symbol in discovered.items():
            mapping.setdefault(type_id, symbol)
    missing = tuple(type_id for type_id in probe.types if type_id not in mapping)
    return ElementSuggestion(probe=probe, mapping=mapping, missing=missing)


_MAPPING_SECTION = "lammps"
_MAPPING_OPTION = "element_maps"
_MAPPING_ENTRY_LIMIT = 32


def _mapping_store_key(directory: Path, types: Sequence[int], natoms: int | None) -> str:
    try:
        root = str(Path(directory).expanduser().resolve())
    except OSError:
        root = str(directory)
    width = "" if natoms is None else str(int(natoms))
    return f"{root}\x1f{width}\x1f{','.join(str(int(value)) for value in sorted(types))}"


def _read_mapping_store() -> dict[str, Any]:
    try:
        from NepTrainKit.config import Config

        raw = Config.get(_MAPPING_SECTION, _MAPPING_OPTION)
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_mapping_store(store: dict[str, Any]) -> None:
    try:
        from NepTrainKit.config import Config

        Config.set(
            _MAPPING_SECTION,
            _MAPPING_OPTION,
            json.dumps(store, ensure_ascii=False, sort_keys=True),
        )
    except Exception:
        logger.debug("could not persist the LAMMPS element mapping")


@dataclass(frozen=True)
class DumpSettings:
    """A confirmed element mapping plus optional manual column roles."""

    element_map: Mapping[int, str]
    overrides: ColumnOverrides | None = None


def remembered_dump_settings(
    directory: Path,
    types: Sequence[int],
    *,
    natoms: int | None = None,
    signature: Sequence[str] = (),
) -> DumpSettings | None:
    """Return a previously confirmed mapping for this folder and type set."""
    key = _mapping_store_key(Path(directory), types, natoms)
    entry = _read_mapping_store().get(key)
    if not isinstance(entry, dict):
        return None
    raw_map = entry.get("map")
    if not isinstance(raw_map, dict):
        return None
    mapping: dict[int, str] = {}
    for raw_type, symbol in raw_map.items():
        try:
            mapping[int(raw_type)] = str(symbol)
        except (TypeError, ValueError):
            return None
    if any(int(type_id) not in mapping for type_id in types):
        return None
    stored_signature = entry.get("signature")
    overrides = None
    if isinstance(stored_signature, list) and tuple(str(value) for value in stored_signature) == tuple(signature):
        overrides = ColumnOverrides.from_dict(entry.get("overrides"))
    return DumpSettings(element_map=mapping, overrides=overrides)


def remember_dump_settings(
    directory: Path,
    types: Sequence[int],
    mapping: Mapping[int, str],
    *,
    natoms: int | None = None,
    signature: Sequence[str] = (),
    overrides: ColumnOverrides | None = None,
) -> None:
    """Persist a confirmed mapping and column roles for later imports."""
    key = _mapping_store_key(Path(directory), types, natoms)
    store = _read_mapping_store()
    store[key] = {
        "types": [int(type_id) for type_id in sorted(types)],
        "map": {str(int(type_id)): str(mapping[int(type_id)]) for type_id in sorted(mapping)},
        "signature": [str(name) for name in signature],
        "overrides": overrides.to_dict() if overrides is not None else None,
    }
    while len(store) > _MAPPING_ENTRY_LIMIT:
        store.pop(next(iter(store)))
    _write_mapping_store(store)
