"""Complete magnetic-order evidence for Training Set Audit.

Only the canonical per-atom ``spin`` vector is interpreted as magnetic state.
Magnetic-force labels such as ``mforce`` and ``force_mag`` are deliberately
outside this module's input contract.
"""
from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from NepTrainKit.core.geometry_cache import GeometrySnapshot

from .result import (
    CompositionMagneticEvidence,
    CompositionPoint,
    DatasetInventory,
    ElementMagneticEvidence,
    ElementMagneticSummary,
    ElementPairMagneticEvidence,
    ElementPairMagneticSummary,
    MagneticEvidenceSummary,
    MagneticInventory,
    StructureMagneticEvidence,
)


MAGNETIC_SCHEMA_VERSION = "magnetic-inventory-v2"
MAGNETIC_METHOD_ID = "spin-order-sf-neighbor-element-v2"
MAGNETIC_ANALYSIS_STRATEGY = "all-spin-structures-v1"

_ORDER_LABELS = (
    "fm",
    "afm",
    "ferrimagnetic",
    "spin_spiral",
    "noncollinear",
    "collinear_mixed",
    "spin_disordered",
    "low_moment",
)
_ELEMENT_ORDER_LABELS = (
    "aligned",
    "compensated",
    "modulated",
    "noncollinear",
    "collinear_mixed",
    "disordered",
    "low_moment",
    "insufficient",
)
_COUPLING_LABELS = ("parallel", "antiparallel", "mixed")


def _ranked_fractions(
    counts: Counter[str], total: int, label_order: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    rank = {label: index for index, label in enumerate(label_order)}
    return tuple(
        (label, count / total)
        for label, count in sorted(
            ((label, counts[label]) for label in label_order if counts[label]),
            key=lambda item: (-item[1], rank[item[0]]),
        )
    )


def _native_module() -> Any:
    try:
        return importlib.import_module("NepTrainKit._native._magnetism")
    except ImportError as exc:
        raise RuntimeError("The native magnetic-order module is unavailable.") from exc


def _frame_geometry(
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


def _spin_array(structure: Any, atom_count: int) -> np.ndarray | None:
    properties = getattr(structure, "atomic_properties", {}) or {}
    if "spin" not in properties:
        return None
    spins = np.asarray(properties["spin"])
    if spins.shape != (atom_count, 3) or not np.issubdtype(spins.dtype, np.number):
        return None
    spins = np.ascontiguousarray(spins, dtype=np.float32)
    if not np.isfinite(spins).all():
        return None
    return spins


def _analyze_structure(
    native: Any,
    source_index: int,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    spins: np.ndarray,
    atomic_numbers: np.ndarray,
) -> StructureMagneticEvidence:
    values = native.magnetic_order_evidence(
        positions, cell, pbc, spins, 12, 3
    )
    element_rows, pair_rows = native.element_magnetic_evidence(
        positions,
        cell,
        pbc,
        spins,
        np.ascontiguousarray(atomic_numbers, dtype=np.int16),
        12,
        3,
    )
    element_evidence = tuple(
        ElementMagneticEvidence(
            element=chemical_symbols[int(row[0])],
            atom_count=int(row[1]),
            spin_atom_count=int(row[2]),
            mean_moment=float(row[3]),
            net_moment_ratio=float(row[4]),
            collinearity=float(row[5]),
            intra_element_correlation=float(row[6]),
            intra_element_pair_count=int(row[7]),
            q_peak_strength=float(row[8]),
            q_vector=(int(row[9]), int(row[10]), int(row[11])),
            order_label=str(row[12]),
        )
        for row in element_rows
    )
    pair_evidence = tuple(
        ElementPairMagneticEvidence(
            element_a=chemical_symbols[int(row[0])],
            element_b=chemical_symbols[int(row[1])],
            pair_count=int(row[2]),
            correlation=float(row[3]),
            coupling_label=str(row[4]),
        )
        for row in pair_rows
    )
    return StructureMagneticEvidence(
        source_index=int(source_index),
        atom_count=len(positions),
        spin_atom_count=int(values[0]),
        mean_moment=float(values[1]),
        moment_std=float(values[2]),
        net_moment_ratio=float(values[3]),
        collinearity=float(values[4]),
        coplanarity=float(values[5]),
        neighbor_correlation=float(values[6]),
        neighbor_abs_correlation=float(values[7]),
        parallel_fraction=float(values[8]),
        antiparallel_fraction=float(values[9]),
        q_peak_strength=float(values[10]),
        q_vector=(int(values[11]), int(values[12]), int(values[13])),
        order_label=str(values[14]),
        confidence_state=str(values[15]),
        element_evidence=element_evidence,
        element_pair_evidence=pair_evidence,
    )


def _summarize_elements(
    structures: Sequence[StructureMagneticEvidence],
) -> tuple[ElementMagneticSummary, ...]:
    grouped: dict[str, list[ElementMagneticEvidence]] = {}
    for structure in structures:
        for evidence in structure.element_evidence:
            grouped.setdefault(evidence.element, []).append(evidence)
    summaries: list[ElementMagneticSummary] = []
    for element in sorted(grouped):
        evidence = grouped[element]
        labels = Counter(item.order_label for item in evidence)
        total = len(evidence)
        mean = lambda name: float(np.mean([getattr(item, name) for item in evidence]))
        summaries.append(
            ElementMagneticSummary(
                element=element,
                structure_count=total,
                order_fractions=_ranked_fractions(
                    labels, total, _ELEMENT_ORDER_LABELS
                ),
                mean_moment=mean("mean_moment"),
                mean_net_moment_ratio=mean("net_moment_ratio"),
                mean_collinearity=mean("collinearity"),
                mean_intra_element_correlation=mean("intra_element_correlation"),
                mean_q_peak_strength=mean("q_peak_strength"),
            )
        )
    return tuple(summaries)


def _summarize_element_pairs(
    structures: Sequence[StructureMagneticEvidence],
) -> tuple[ElementPairMagneticSummary, ...]:
    grouped: dict[tuple[str, str], list[ElementPairMagneticEvidence]] = {}
    for structure in structures:
        for evidence in structure.element_pair_evidence:
            key = (evidence.element_a, evidence.element_b)
            grouped.setdefault(key, []).append(evidence)
    summaries: list[ElementPairMagneticSummary] = []
    for element_a, element_b in sorted(grouped):
        evidence = grouped[(element_a, element_b)]
        labels = Counter(item.coupling_label for item in evidence)
        total = len(evidence)
        summaries.append(
            ElementPairMagneticSummary(
                element_a=element_a,
                element_b=element_b,
                structure_count=total,
                coupling_fractions=_ranked_fractions(
                    labels, total, _COUPLING_LABELS
                ),
                mean_correlation=float(np.mean([item.correlation for item in evidence])),
            )
        )
    return tuple(summaries)


def _analyze_composition_point(
    native: Any,
    geometry: GeometrySnapshot,
    source_rows: dict[int, int],
    structures: Sequence[Any],
    point: CompositionPoint,
) -> CompositionMagneticEvidence:
    evidence: list[StructureMagneticEvidence] = []
    missing = 0
    labels: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    for source_index in point.structure_indices:
        row = source_rows.get(int(source_index))
        if row is None or source_index < 0 or source_index >= len(structures):
            missing += 1
            continue
        positions, cell, pbc, atomic_numbers = _frame_geometry(geometry, row)
        spins = _spin_array(structures[source_index], len(positions))
        if spins is None:
            missing += 1
            continue
        current = _analyze_structure(
            native, source_index, positions, cell, pbc, spins, atomic_numbers
        )
        evidence.append(current)
        labels[current.order_label] += 1
        confidence[current.confidence_state] += 1
    analyzed = len(evidence)
    mean = lambda name: (
        float(np.mean([getattr(item, name) for item in evidence])) if evidence else 0.0
    )
    return CompositionMagneticEvidence(
        reduced_counts=point.reduced_counts,
        source_structure_count=point.structure_count,
        analyzed_structure_count=analyzed,
        missing_spin_count=missing,
        order_fractions=_ranked_fractions(labels, analyzed, _ORDER_LABELS),
        confidence_counts=tuple(
            (label, confidence[label])
            for label in ("strong", "mixed", "unresolved")
            if confidence[label]
        ),
        mean_net_moment_ratio=mean("net_moment_ratio"),
        mean_collinearity=mean("collinearity"),
        mean_q_peak_strength=mean("q_peak_strength"),
        element_summaries=_summarize_elements(evidence),
        element_pair_summaries=_summarize_element_pairs(evidence),
        structures=tuple(evidence),
    )


def _build_uncached(
    geometry: GeometrySnapshot,
    inventory: DatasetInventory,
    structures: Sequence[Any],
    progress: Callable[[int, int], None] | None,
) -> MagneticInventory:
    native = _native_module()
    source_rows = {
        int(source_index): row
        for row, source_index in enumerate(geometry.source_indices)
    }
    completed = 0
    points: list[CompositionMagneticEvidence] = []
    for point in inventory.composition_points:
        points.append(
            _analyze_composition_point(
                native, geometry, source_rows, structures, point
            )
        )
        completed += point.structure_count
        if progress is not None:
            progress(completed, inventory.structure_count)
    return MagneticInventory(
        schema_version=MAGNETIC_SCHEMA_VERSION,
        method_id=MAGNETIC_METHOD_ID,
        analysis_strategy=MAGNETIC_ANALYSIS_STRATEGY,
        source_structure_count=inventory.structure_count,
        analyzed_structure_count=sum(point.analyzed_structure_count for point in points),
        missing_spin_count=sum(point.missing_spin_count for point in points),
        composition_points=tuple(points),
    )


def build_magnetic_inventory(
    geometry: GeometrySnapshot,
    inventory: DatasetInventory,
    structures: Sequence[Any],
    *,
    cache_owner: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[MagneticInventory, bool]:
    """Analyze every in-scope structure carrying a valid ``spin:R:3`` field."""
    cache_key = (
        MAGNETIC_SCHEMA_VERSION,
        MAGNETIC_METHOD_ID,
        MAGNETIC_ANALYSIS_STRATEGY,
        tuple(int(index) for index in geometry.source_indices),
    )
    build = lambda: _build_uncached(geometry, inventory, structures, progress)
    cached_analysis = getattr(cache_owner, "cached_geometry_analysis", None)
    if callable(cached_analysis):
        result, cache_hit = cached_analysis(
            "training-set-audit-magnetism", cache_key, build
        )
    else:
        result, cache_hit = build(), False
    if cache_hit and progress is not None:
        progress(result.source_structure_count, result.source_structure_count)
    return result, cache_hit


def summarize_magnetic_inventory(
    inventory: MagneticInventory,
    reduced_counts: set[tuple[int, ...]] | None = None,
) -> MagneticEvidenceSummary | None:
    """Aggregate magnetic-order evidence for selected composition points."""
    points = tuple(
        point
        for point in inventory.composition_points
        if reduced_counts is None or point.reduced_counts in reduced_counts
    )
    analyzed = sum(point.analyzed_structure_count for point in points)
    if analyzed <= 0:
        return None
    labels: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    for point in points:
        for label, fraction in point.order_fractions:
            labels[label] += point.analyzed_structure_count * fraction
        confidence.update(dict(point.confidence_counts))
    weighted = lambda name: sum(
        point.analyzed_structure_count * getattr(point, name) for point in points
    ) / analyzed
    return MagneticEvidenceSummary(
        source_structure_count=sum(point.source_structure_count for point in points),
        analyzed_structure_count=analyzed,
        missing_spin_count=sum(point.missing_spin_count for point in points),
        order_fractions=_ranked_fractions(labels, analyzed, _ORDER_LABELS),
        confidence_counts=tuple(
            (label, confidence[label])
            for label in ("strong", "mixed", "unresolved")
            if confidence[label]
        ),
        mean_net_moment_ratio=weighted("mean_net_moment_ratio"),
        mean_collinearity=weighted("mean_collinearity"),
        mean_q_peak_strength=weighted("mean_q_peak_strength"),
        element_summaries=_summarize_elements(
            tuple(structure for point in points for structure in point.structures)
        ),
        element_pair_summaries=_summarize_element_pairs(
            tuple(structure for point in points for structure in point.structures)
        ),
    )
