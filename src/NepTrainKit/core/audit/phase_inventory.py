"""Production phase-evidence contract for Training Set Audit.

This module owns complete local-structure analysis, ordering refinements, and
dataset-level caching. Callers only consume the versioned result contract.
"""
from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from NepTrainKit.core.geometry_cache import GeometrySnapshot
from NepTrainKit.core.geometry_cache import structure_cell_array
from NepTrainKit.core.geometry_cache import structure_pbc_flags

from .phase_refinement import refine_l12, refine_laves
from .prototype_registry import match_common_prototype
from .result import (
    CompositionPoint,
    CompositionPhaseEvidence,
    DatasetInventory,
    PhaseEvidenceSummary,
    PhaseInventory,
    StructurePhaseEvidence,
)


PHASE_SCHEMA_VERSION = "phase-inventory-v2"
PHASE_METHOD_ID = "adaptive-cna-prototype-v2"
PHASE_REFERENCE_BANK_ID = "aflow-common-prototypes-v2"
PHASE_ANALYSIS_STRATEGY = "all-structures-v1"

_LOCAL_PHASES = ("fcc", "hcp", "bcc", "unresolved")
_CNA_CODES = {1: "fcc", 2: "hcp", 3: "bcc", 0: "unresolved"}
PHASE_PARTITION_LABELS = (
    "fcc",
    "bcc",
    "hcp",
    "diamond",
    "l10",
    "l12",
    "b1",
    "b2",
    "b3",
    "b4",
    "fluorite",
    "nias",
    "d03",
    "l21",
    "c1b",
    "d019",
    "c14",
    "c15",
    "mixed",
    "unresolved",
)


def phase_partition_label(structure: StructurePhaseEvidence) -> str:
    """Keep mixed local structures out of hard phase-share buckets."""
    if structure.confidence_state == "mixed":
        return "mixed"
    return structure.phase_label


def _frame_arrays(
    geometry: GeometrySnapshot,
    row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(geometry.atom_offsets[row])
    stop = int(geometry.atom_offsets[row + 1])
    return (
        geometry.positions[start:stop],
        geometry.cells[row],
        geometry.pbc[row].astype(bool, copy=False),
        geometry.atomic_numbers[start:stop],
    )


def _local_phase_counts(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    cna_labels: np.ndarray | None = None,
) -> Counter[str]:
    if cna_labels is None:
        phase_module = importlib.import_module("NepTrainKit.core.audit.phase_sketch")
        vectors, indices, valid = phase_module.accelerated_periodic_knn_vectors(
            positions,
            cell,
            pbc,
            neighbors=24,
        )
        codes = phase_module.adaptive_cna_labels(vectors, indices, valid)
    else:
        codes = np.asarray(cna_labels, dtype=np.int8)
    return Counter(_CNA_CODES.get(int(code), "unresolved") for code in codes)


def _confirmed_ordering(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    atom_types: np.ndarray,
    local_counts: Counter[str],
    *,
    neighbor_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> str | None:
    if not np.all(pbc):
        return None
    if np.unique(atom_types).size == 2:
        local_total = sum(local_counts.values())
        incompatible_l12_skeleton = (
            local_total > 0
            and max(
                local_counts.get("bcc", 0),
                local_counts.get("hcp", 0),
            )
            / local_total
            >= 0.50
        )
        if not incompatible_l12_skeleton:
            l12 = refine_l12(
                positions,
                cell,
                pbc,
                atom_types,
                _neighbor_data=neighbor_data,
            )
            if l12.confirmed:
                return "l12"
        laves = refine_laves(
            positions,
            cell,
            pbc,
            atom_types,
            _neighbor_data=neighbor_data,
        )
        if laves.confirmed:
            return laves.label
    common = match_common_prototype(
        positions,
        cell,
        pbc,
        atom_types,
        candidate_labels=_common_prototype_candidates(atom_types, local_counts),
        _neighbor_data=neighbor_data,
    )
    if common.confirmed:
        return common.label
    return None


def _common_prototype_candidates(
    atom_types: np.ndarray,
    local_counts: Counter[str],
) -> tuple[str, ...]:
    """Use composition and a-CNA skeleton evidence to narrow costly matching."""
    _present, counts = np.unique(atom_types, return_counts=True)
    fractions = np.sort(counts.astype(float) / float(np.sum(counts)))
    local_total = sum(local_counts.values())
    dominant = None
    if local_total:
        candidate, count = max(
            (
                (phase, local_counts.get(phase, 0))
                for phase in ("fcc", "hcp", "bcc")
            ),
            key=lambda item: item[1],
        )
        if count / local_total >= 0.50:
            dominant = candidate

    def composition_is(expected: tuple[float, ...]) -> bool:
        return len(fractions) == len(expected) and bool(
            np.all(np.abs(fractions - np.asarray(expected)) <= 0.035)
        )

    if composition_is((1.0,)):
        return ("diamond",) if dominant is None else ()
    if composition_is((0.5, 0.5)):
        if dominant == "fcc":
            return ("l10",)
        if dominant == "bcc":
            return ("b2",)
        if dominant == "hcp":
            return ("b4", "nias")
        return ("b1", "b3", "b4", "nias")
    if composition_is((0.25, 0.75)):
        if dominant == "bcc":
            return ("d03",)
        if dominant == "hcp":
            return ("d019",)
        return ("d03", "d019")
    if composition_is((1.0 / 3.0, 2.0 / 3.0)):
        return ("fluorite",)
    if composition_is((0.25, 0.25, 0.50)):
        return ("l21",)
    if composition_is((1.0 / 3.0,) * 3):
        return ("c1b",)
    return ()


def _phase_label_and_confidence(
    local_counts: Counter[str],
    confirmed_ordering: str | None,
) -> tuple[str, str]:
    if confirmed_ordering is not None:
        return confirmed_ordering, "strong"
    atom_count = sum(local_counts.values())
    if atom_count <= 0:
        return "unresolved", "unresolved"
    candidate, count = max(
        ((phase, local_counts.get(phase, 0)) for phase in ("fcc", "hcp", "bcc")),
        key=lambda item: item[1],
    )
    fraction = count / atom_count
    if fraction >= 0.80:
        return candidate, "strong"
    if fraction >= 0.50:
        return candidate, "mixed"
    return "unresolved", "unresolved"


def _classify_structure_arrays(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    atom_types: np.ndarray,
    *,
    source_index: int,
) -> tuple[StructurePhaseEvidence, Counter[str], str | None]:
    """Classify one structure while preserving local-topology evidence."""
    neighbor_data = None
    cna_labels = None
    try:
        phase_module = importlib.import_module(
            "NepTrainKit.core.audit.phase_sketch"
        )
        vectors, indices, valid, cna_labels = (
            phase_module.phase_partition_primitives(
                positions,
                cell,
                pbc,
                neighbors=32,
            )
        )
        neighbor_data = (vectors, indices, valid)
    except (RuntimeError, ValueError):
        neighbor_data = None
        cna_labels = None
    try:
        local_counts = _local_phase_counts(
            positions,
            cell,
            pbc,
            cna_labels=cna_labels,
        )
    except ValueError:
        local_counts = Counter({"unresolved": len(atom_types)})
    try:
        ordering = _confirmed_ordering(
            positions,
            cell,
            pbc,
            atom_types,
            local_counts,
            neighbor_data=neighbor_data,
        )
    except ValueError:
        ordering = None
    label, confidence = _phase_label_and_confidence(local_counts, ordering)
    atom_count = len(atom_types)
    local_total = sum(local_counts.values())
    evidence = StructurePhaseEvidence(
        source_index=int(source_index),
        atom_count=atom_count,
        phase_label=label,
        confidence_state=confidence,
        local_phase_fractions=tuple(
            (
                phase,
                0.0 if local_total <= 0 else local_counts.get(phase, 0) / local_total,
            )
            for phase in _LOCAL_PHASES
        ),
    )
    return evidence, local_counts, ordering


def analyze_structure_phase(
    structure: Any,
    *,
    source_index: int = 0,
) -> StructurePhaseEvidence:
    """Return conservative structural-phase evidence for one structure frame."""
    positions = np.ascontiguousarray(structure.positions, dtype=np.float32)
    cell = structure_cell_array(structure, dtype=np.float32)
    pbc = np.ascontiguousarray(structure_pbc_flags(structure), dtype=bool)
    atom_types = np.ascontiguousarray(structure.numbers, dtype=np.int16)
    if positions.shape != (len(atom_types), 3) or cell.shape != (3, 3):
        raise ValueError("A structure has invalid positions or cell data.")
    evidence, _local_counts, _ordering = _classify_structure_arrays(
        positions,
        cell,
        pbc,
        atom_types,
        source_index=source_index,
    )
    return evidence


def _analyze_composition_point(
    geometry: GeometrySnapshot,
    source_rows: dict[int, int],
    point: CompositionPoint,
    selected: tuple[int, ...],
) -> CompositionPhaseEvidence:
    local_counts: Counter[str] = Counter()
    structure_labels: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    confirmed_candidates: Counter[str] = Counter()
    structure_evidence: list[StructurePhaseEvidence] = []
    analyzed_atoms = 0
    for source_index in selected:
        row = source_rows.get(int(source_index))
        if row is None:
            continue
        positions, cell, pbc, atom_types = _frame_arrays(geometry, row)
        evidence, current_local_counts, ordering = _classify_structure_arrays(
            positions,
            cell,
            pbc,
            atom_types,
            source_index=int(source_index),
        )
        structure_evidence.append(evidence)
        label = evidence.phase_label
        confidence = evidence.confidence_state
        atom_count = evidence.atom_count
        local_counts.update(current_local_counts)
        structure_labels[label] += 1
        confidence_counts[confidence] += 1
        if ordering is not None:
            confirmed_candidates[ordering] += 1
        analyzed_atoms += atom_count
    analyzed_structures = len(structure_evidence)
    local_total = sum(local_counts.values())
    return CompositionPhaseEvidence(
        reduced_counts=point.reduced_counts,
        source_structure_count=point.structure_count,
        analyzed_structure_count=analyzed_structures,
        analyzed_atom_count=analyzed_atoms,
        local_phase_fractions=tuple(
            (
                phase,
                0.0 if local_total <= 0 else local_counts[phase] / local_total,
            )
            for phase in _LOCAL_PHASES
        ),
        structure_phase_fractions=tuple(
            (label, count / analyzed_structures)
            for label, count in sorted(
                structure_labels.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ) if analyzed_structures else (),
        confidence_counts=tuple(
            (label, confidence_counts[label])
            for label in ("strong", "mixed", "unresolved")
            if confidence_counts[label]
        ),
        confirmed_candidates=tuple(sorted(confirmed_candidates.items())),
        structures=tuple(structure_evidence),
    )


def _build_uncached(
    geometry: GeometrySnapshot,
    inventory: DatasetInventory,
    progress: Callable[[int, int], None] | None = None,
) -> PhaseInventory:
    source_rows = {
        int(source_index): row
        for row, source_index in enumerate(geometry.source_indices)
    }
    completed = 0
    total = inventory.structure_count
    points: list[CompositionPhaseEvidence] = []
    for point in inventory.composition_points:
        points.append(
            _analyze_composition_point(
                geometry,
                source_rows,
                point,
                point.structure_indices,
            )
        )
        completed += point.structure_count
        if progress is not None:
            progress(completed, total)
    return PhaseInventory(
        schema_version=PHASE_SCHEMA_VERSION,
        method_id=PHASE_METHOD_ID,
        reference_bank_id=PHASE_REFERENCE_BANK_ID,
        analysis_strategy=PHASE_ANALYSIS_STRATEGY,
        source_structure_count=inventory.structure_count,
        analyzed_structure_count=sum(point.analyzed_structure_count for point in points),
        analyzed_atom_count=sum(point.analyzed_atom_count for point in points),
        composition_points=tuple(points),
    )


def build_phase_inventory(
    geometry: GeometrySnapshot,
    inventory: DatasetInventory,
    *,
    cache_owner: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[PhaseInventory, bool]:
    """Build or reuse complete phase evidence for an immutable geometry scope."""
    cache_key = (
        PHASE_SCHEMA_VERSION,
        PHASE_METHOD_ID,
        PHASE_REFERENCE_BANK_ID,
        PHASE_ANALYSIS_STRATEGY,
        tuple(int(index) for index in geometry.source_indices),
    )
    build = lambda: _build_uncached(geometry, inventory, progress)
    cached_analysis = getattr(cache_owner, "cached_geometry_analysis", None)
    if callable(cached_analysis):
        result, cache_hit = cached_analysis(
            "training-set-audit-phase", cache_key, build
        )
    else:
        result, cache_hit = build(), False
    if cache_hit and progress is not None:
        progress(result.analyzed_structure_count, result.source_structure_count)
    return result, cache_hit


def summarize_phase_inventory(
    inventory: PhaseInventory,
    reduced_counts: set[tuple[int, ...]] | None = None,
) -> PhaseEvidenceSummary | None:
    """Aggregate selected points using all analyzed atoms and structures."""
    points = tuple(
        point
        for point in inventory.composition_points
        if reduced_counts is None or point.reduced_counts in reduced_counts
    )
    analyzed_structures = sum(point.analyzed_structure_count for point in points)
    analyzed_atoms = sum(point.analyzed_atom_count for point in points)
    if analyzed_structures <= 0 or analyzed_atoms <= 0:
        return None
    local_totals: Counter[str] = Counter()
    confidence_totals: Counter[str] = Counter()
    confirmed_totals: Counter[str] = Counter()
    for point in points:
        for label, fraction in point.local_phase_fractions:
            local_totals[label] += point.analyzed_atom_count * fraction
        confidence_totals.update(dict(point.confidence_counts))
        confirmed_totals.update(dict(point.confirmed_candidates))
    return PhaseEvidenceSummary(
        source_structure_count=sum(point.source_structure_count for point in points),
        analyzed_structure_count=analyzed_structures,
        analyzed_atom_count=analyzed_atoms,
        local_phase_fractions=tuple(
            (label, local_totals[label] / analyzed_atoms)
            for label in _LOCAL_PHASES
        ),
        confidence_counts=tuple(
            (label, confidence_totals[label])
            for label in ("strong", "mixed", "unresolved")
            if confidence_totals[label]
        ),
        confirmed_candidates=tuple(sorted(confirmed_totals.items())),
    )
