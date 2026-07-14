"""Extract compact records from NepTrainKit datasets for Training Set Audit."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from NepTrainKit.core.structure import Structure


@dataclass(frozen=True)
class StructureAuditRecord:
    index: int
    formula: str
    num_atoms: int
    composition: dict[str, float]
    config_type: str
    energy_per_atom: float | None
    max_force: float | None
    virial_norm: float | None


def _composition(elements: Sequence[str]) -> dict[str, float]:
    counts = Counter(str(element) for element in elements)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {element: count / total for element, count in sorted(counts.items())}


def _flag(structure: Structure, name: str) -> bool:
    value = getattr(structure, name, False)
    return bool(value() if callable(value) else value)


def _safe_energy_per_atom(structure: Structure) -> float | None:
    try:
        if _flag(structure, "has_energy") and structure.num_atoms:
            return float(structure.energy) / float(structure.num_atoms)
    except Exception:
        return None
    return None


def _safe_max_force(structure: Structure) -> float | None:
    try:
        if _flag(structure, "has_forces"):
            forces = np.asarray(structure.forces, dtype=np.float64).reshape(-1, 3)
            if forces.size:
                return float(np.linalg.norm(forces, axis=1).max())
    except Exception:
        return None
    return None


def _safe_virial_norm(structure: Structure) -> float | None:
    try:
        if _flag(structure, "has_virial"):
            virial = np.asarray(structure.virial, dtype=np.float64).reshape(-1)
            if virial.size:
                return float(np.linalg.norm(virial))
    except Exception:
        return None
    return None


def _record_from_structure(index: int, structure: Structure) -> StructureAuditRecord:
    elements = [str(element) for element in structure.elements]
    return StructureAuditRecord(
        index=int(index),
        formula=structure.formula,
        num_atoms=int(structure.num_atoms),
        composition=_composition(elements),
        config_type=str(getattr(structure, "tag", "") or ""),
        energy_per_atom=_safe_energy_per_atom(structure),
        max_force=_safe_max_force(structure),
        virial_norm=_safe_virial_norm(structure),
    )


def records_from_structures(structures: Sequence[Structure]) -> list[StructureAuditRecord]:
    return [_record_from_structure(index, structure) for index, structure in enumerate(structures)]


def records_from_indexed_structures(
    structures: Sequence[tuple[int, Structure]],
) -> list[StructureAuditRecord]:
    """Build compact records while preserving source-dataset indices."""
    return [_record_from_structure(index, structure) for index, structure in structures]


def indexed_structures_from_result_data(result_data: Any) -> list[tuple[int, Structure]]:
    """Return active structures with indices relative to the original dataset."""
    structure_data = getattr(result_data, "structure", result_data)
    if structure_data is None:
        return []

    all_data = getattr(structure_data, "all_data", None)
    now_indices = getattr(structure_data, "now_indices", None)
    now_data = getattr(structure_data, "now_data", None)

    if all_data is not None:
        if now_indices is None:
            now_indices = range(len(all_data))
        indexed: list[tuple[int, Structure]] = []
        for index in now_indices:
            idx = int(index)
            if idx < 0 or idx >= len(all_data):
                continue
            indexed.append((idx, all_data[idx]))
        return indexed

    if now_data is not None:
        return [(index, structure) for index, structure in enumerate(now_data)]

    if isinstance(structure_data, Sequence) and not isinstance(structure_data, (str, bytes)):
        return [(index, structure) for index, structure in enumerate(structure_data)]

    return []


def records_from_result_data(result_data: Any) -> list[StructureAuditRecord]:
    return records_from_indexed_structures(indexed_structures_from_result_data(result_data))
