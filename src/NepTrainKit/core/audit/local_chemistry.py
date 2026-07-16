"""Local-chemistry audit based on the active NEP model cutoffs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

import numpy as np
from ase.data import chemical_symbols

from NepTrainKit.core.geometry_cache import GeometrySnapshot
from NepTrainKit.core.structure import Structure

from .nep_cutoff import NepCutoffProfile
from .neighbor_scan import local_chemistry_summary_batch
from .result import AuditBiasType, AuditDimension, AuditSeverity, AuditSlice, AuditStatus, SliceMetric


_SCOPE_TITLES = {"angular": "Angular core", "radial": "Radial context"}
_FRACTION_EDGES = tuple(index / 10.0 for index in range(11))
_FRACTION_LABELS = tuple(f"{10 * index}-{10 * (index + 1)}%" for index in range(10))
_NATIVE_BATCH_SIZE = 512


@dataclass
class _Histogram:
    counts: Counter[int] = field(default_factory=Counter)
    structure_indices: dict[int, dict[int, None]] = field(default_factory=lambda: defaultdict(dict))
    sample_count: int = 0

    def add(self, bin_index: int, structure_index: int) -> None:
        self.counts[bin_index] += 1
        self.structure_indices[bin_index][structure_index] = None
        self.sample_count += 1

    def add_many(self, bin_indices: np.ndarray, structure_indices: np.ndarray) -> None:
        """Merge a full-scope vectorized histogram payload."""
        bins = np.asarray(bin_indices, dtype=np.int64).reshape(-1)
        sources = np.asarray(structure_indices, dtype=np.int64).reshape(-1)
        if bins.size != sources.size:
            raise ValueError("Histogram bins and structure indices must have matching sizes.")
        if bins.size == 0:
            return
        if np.any(bins < 0):
            raise ValueError("Histogram bins must be non-negative.")
        counts = np.bincount(bins)
        unique_bins = np.flatnonzero(counts)
        self.counts.update(
            {int(bin_index): int(counts[bin_index]) for bin_index in unique_bins}
        )

        source_min = int(np.min(sources))
        source_span = int(np.max(sources)) - source_min + 1
        encoded = bins * source_span + (sources - source_min)
        unique_pairs, first_indices = np.unique(encoded, return_index=True)
        pair_bins = unique_pairs // source_span
        starts = np.flatnonzero(np.r_[True, pair_bins[1:] != pair_bins[:-1]])
        stops = np.r_[starts[1:], pair_bins.size]
        for start, stop in zip(starts, stops):
            encounter_order = np.sort(first_indices[start:stop])
            ordered_sources = dict.fromkeys(int(index) for index in sources[encounter_order])
            self.structure_indices[int(pair_bins[start])].update(ordered_sources)
        self.sample_count += int(bins.size)


def _pbc_flags(structure: Structure) -> tuple[bool, bool, bool]:
    value = getattr(structure, "additional_fields", {}).get("pbc", "T T T")
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if len(tokens) == 1:
            tokens *= 3
        if len(tokens) != 3:
            raise ValueError("A structure has invalid PBC metadata.")
        mapping = {
            "t": True,
            "true": True,
            "1": True,
            "yes": True,
            "f": False,
            "false": False,
            "0": False,
            "no": False,
        }
        try:
            return tuple(mapping[token.lower()] for token in tokens)  # type: ignore[return-value]
        except KeyError as exc:
            raise ValueError("A structure has invalid PBC metadata.") from exc

    flags = np.asarray(value, dtype=np.bool_).reshape(-1)
    if flags.size == 1:
        flags = np.repeat(flags, 3)
    if flags.size != 3:
        raise ValueError("A structure has invalid PBC metadata.")
    return tuple(bool(flag) for flag in flags)  # type: ignore[return-value]


def _geometry_arrays(
    structure: Structure,
    model_elements: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    symbols = tuple(str(symbol) for symbol in structure.elements)
    unknown = sorted(set(symbols) - model_elements)
    if unknown:
        raise ValueError("A structure contains elements not declared in the active NEP model.")

    positions = np.asarray(structure.positions, dtype=np.float64)
    cell = np.asarray(structure.cell, dtype=np.float64)
    if positions.shape != (len(symbols), 3) or cell.shape != (3, 3):
        raise ValueError("A structure has invalid positions or cell data.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise ValueError("A structure has non-finite positions or cell data.")
    return positions, cell, np.asarray(_pbc_flags(structure), dtype=np.bool_), symbols


def _iter_geometry_batches(
    indexed_structures: Sequence[tuple[int, Structure]],
    model_elements: set[str],
):
    for start in range(0, len(indexed_structures), _NATIVE_BATCH_SIZE):
        chunk = indexed_structures[start : start + _NATIVE_BATCH_SIZE]
        prepared = [
            (structure_index, *_geometry_arrays(structure, model_elements))
            for structure_index, structure in chunk
        ]
        atom_counts = np.asarray([len(item[1]) for item in prepared], dtype=np.int64)
        atom_offsets = np.empty(len(prepared) + 1, dtype=np.int64)
        atom_offsets[0] = 0
        np.cumsum(atom_counts, out=atom_offsets[1:])
        yield prepared, atom_offsets


def _iter_cached_geometry_batches(
    geometry: GeometrySnapshot,
    model_elements: set[str],
):
    for start in range(0, geometry.structure_count, _NATIVE_BATCH_SIZE):
        stop = min(start + _NATIVE_BATCH_SIZE, geometry.structure_count)
        atom_begin = int(geometry.atom_offsets[start])
        local_offsets = np.ascontiguousarray(
            geometry.atom_offsets[start : stop + 1] - atom_begin,
            dtype=np.int64,
        )
        prepared = []
        for row in range(start, stop):
            frame_begin = int(geometry.atom_offsets[row])
            frame_end = int(geometry.atom_offsets[row + 1])
            symbols = tuple(
                chemical_symbols[int(number)]
                for number in geometry.atomic_numbers[frame_begin:frame_end]
            )
            unknown = sorted(set(symbols) - model_elements)
            if unknown:
                raise ValueError("A structure contains elements not declared in the active NEP model.")
            prepared.append(
                (
                    int(geometry.source_indices[row]),
                    geometry.positions[frame_begin:frame_end],
                    geometry.cells[row],
                    geometry.pbc[row],
                    symbols,
                )
            )
        yield prepared, local_offsets


def _cutoff_matrix(profile: NepCutoffProfile, elements: tuple[str, ...], scope: str) -> np.ndarray:
    cutoffs = np.asarray(
        [profile.pair_cutoff(element, element, scope) for element in elements],
        dtype=np.float64,
    )
    return 0.5 * (cutoffs[:, np.newaxis] + cutoffs[np.newaxis, :])


def _severity(sample_fraction: float) -> AuditSeverity:
    if sample_fraction < 0.03:
        return AuditSeverity.HIGH
    if sample_fraction < 0.08:
        return AuditSeverity.MEDIUM
    return AuditSeverity.LOW


def _plot_and_slices(
    histogram: _Histogram,
    *,
    scope: str,
    center: str,
    metric: str,
    metric_label: str,
    fraction_metric: bool,
) -> tuple[dict[str, object], tuple[AuditSlice, ...]]:
    if fraction_metric:
        bin_keys = tuple(range(10))
        bin_edges = _FRACTION_EDGES
        bin_labels = _FRACTION_LABELS
    else:
        minimum = min(histogram.counts)
        maximum = max(histogram.counts)
        bin_keys = tuple(range(minimum, maximum + 1))
        bin_edges = tuple(float(value) - 0.5 for value in range(minimum, maximum + 2))
        bin_labels = tuple(str(value) for value in bin_keys)

    counts = tuple(histogram.counts.get(key, 0) for key in bin_keys)
    index_groups = tuple(tuple(histogram.structure_indices.get(key, ())) for key in bin_keys)
    sparse_keys = tuple(
        key
        for key in bin_keys
        if histogram.counts.get(key, 0) and histogram.counts[key] / histogram.sample_count < 0.10
    )
    sparse_positions = tuple(bin_keys.index(key) for key in sparse_keys)
    plot_id = f"local_chemistry:{scope}:{center}:{metric}"
    plot = {
        "kind": "histogram",
        "id": plot_id,
        "scope": scope,
        "center_element": center,
        "metric": metric,
        "title": f"{_SCOPE_TITLES[scope]}: {center} {metric_label}",
        "x_label": metric_label.capitalize(),
        "y_label": "Local environments",
        "series": (
            {
                "id": metric,
                "label": metric_label,
                "bin_edges": bin_edges,
                "bin_labels": bin_labels,
                "counts": counts,
                "highlighted_bins": sparse_positions,
                "structure_indices": index_groups,
            },
        ),
        "sample_count": histogram.sample_count,
    }

    slices = []
    for key in sparse_keys:
        position = bin_keys.index(key)
        count = histogram.counts[key]
        sample_fraction = count / histogram.sample_count
        label = bin_labels[position]
        slices.append(
            AuditSlice(
                id=f"{plot_id}:{position}",
                title=f"Sparse {_SCOPE_TITLES[scope].lower()} bin: {center} {metric_label} {label}",
                dimension_id="local_chemistry",
                severity=_severity(sample_fraction),
                bias_type=AuditBiasType.SPARSITY,
                structure_indices=tuple(histogram.structure_indices[key]),
                observed=(
                    f"{label} contains {count} of {histogram.sample_count} comparable {center} "
                    f"local environments ({sample_fraction:.1%})."
                ),
                interpretation="This populated local-chemistry bin is thin relative to comparable environments.",
                limit="Sparsity is only actionable when this environment matters for the intended model use.",
                metrics=(
                    SliceMetric("sample_count", count, "local environments", histogram.sample_count, "low"),
                    SliceMetric("sample_fraction", round(sample_fraction, 4), "", None, "low"),
                ),
            )
        )
    return plot, tuple(slices)


def audit_local_chemistry(
    indexed_structures: Sequence[tuple[int, Structure]],
    profile: NepCutoffProfile,
    *,
    pair_contact_collector=None,
    geometry: GeometrySnapshot | None = None,
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    """Audit per-center neighbor counts and chemical fractions at NEP cutoffs."""
    empty_overview = {"available_scopes": (), "center_element_count": 0, "sparse_bin_count": 0}
    if not indexed_structures:
        return (
            AuditDimension("local_chemistry", "Local chemistry", AuditStatus.UNAVAILABLE, "No structures are loaded."),
            (),
            empty_overview,
        )

    audit_started = perf_counter()
    timings_ms = {
        "batch_geometry_prepare": 0.0,
        "batch_type_prepare": 0.0,
        "neighbor_kernel": 0.0,
        "batch_result_collect": 0.0,
    }
    elements = profile.elements
    model_elements = set(elements)
    histograms: dict[tuple[str, str, str], _Histogram] = defaultdict(_Histogram)
    radial_cutoff_matrix = _cutoff_matrix(profile, elements, "radial")
    angular_cutoff_matrix = _cutoff_matrix(profile, elements, "angular")
    element_indices = {element: index for index, element in enumerate(elements)}
    present_elements: set[str] = set()
    cutoff_matrices = np.stack([angular_cutoff_matrix, radial_cutoff_matrix])
    source_atom_parts: list[np.ndarray] = []
    atom_type_parts: list[np.ndarray] = []
    neighbor_count_parts: list[np.ndarray] = []
    neighbor_type_count_parts: list[np.ndarray] = []
    batches = (
        _iter_cached_geometry_batches(geometry, model_elements)
        if geometry is not None
        else _iter_geometry_batches(indexed_structures, model_elements)
    )
    batch_iterator = iter(batches)
    while True:
        stage_started = perf_counter()
        try:
            prepared, atom_offsets = next(batch_iterator)
        except StopIteration:
            break
        timings_ms["batch_geometry_prepare"] += (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        atom_types = np.concatenate(
            [np.asarray([element_indices[symbol] for symbol in item[4]], dtype=np.int32) for item in prepared]
        )
        detail_mask = (
            pair_contact_collector.detail_mask()
            if pair_contact_collector is not None
            else np.zeros((2, len(elements) * (len(elements) + 1) // 2), dtype=np.uint8)
        )
        timings_ms["batch_type_prepare"] += (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        neighbor_counts, neighbor_type_counts, *contact_summary = local_chemistry_summary_batch(
            [item[1] for item in prepared],
            [item[2] for item in prepared],
            [item[3] for item in prepared],
            atom_types,
            cutoff_matrices,
            detail_mask,
        )
        timings_ms["neighbor_kernel"] += (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        source_atom_parts.append(
            np.repeat(
                np.asarray([item[0] for item in prepared], dtype=np.int64),
                np.diff(atom_offsets),
            )
        )
        atom_type_parts.append(atom_types)
        neighbor_count_parts.append(neighbor_counts)
        neighbor_type_count_parts.append(neighbor_type_counts)
        if pair_contact_collector is not None:
            pair_contact_collector.observe_batch(
                [item[0] for item in prepared],
                [item[4] for item in prepared],
                *contact_summary,
            )
        for _structure_index, _positions, _cell, _pbc, symbols in prepared:
            present_elements.update(symbols)
        timings_ms["batch_result_collect"] += (perf_counter() - stage_started) * 1000.0

    stage_started = perf_counter()
    if source_atom_parts:
        source_atoms = np.concatenate(source_atom_parts)
        all_atom_types = np.concatenate(atom_type_parts)
        all_neighbor_counts = np.concatenate(neighbor_count_parts, axis=1)
        all_neighbor_type_counts = np.concatenate(neighbor_type_count_parts, axis=1)
        for scope_index, scope in enumerate(("angular", "radial")):
            for center_index, center in enumerate(elements):
                center_mask = all_atom_types == center_index
                center_counts = all_neighbor_counts[scope_index, center_mask]
                center_sources = source_atoms[center_mask]
                histograms[(scope, center, "neighbor_count")].add_many(
                    center_counts,
                    center_sources,
                )
                for neighbor_index, neighbor_element in enumerate(elements):
                    fractions = np.divide(
                        all_neighbor_type_counts[scope_index, center_mask, neighbor_index],
                        center_counts,
                        out=np.zeros(center_counts.shape, dtype=np.float64),
                        where=center_counts != 0,
                    )
                    fraction_bins = np.minimum((fractions * 10.0).astype(np.int64), 9)
                    histograms[(scope, center, f"neighbor_fraction_{neighbor_element}")].add_many(
                        fraction_bins,
                        center_sources,
                    )
    timings_ms["histogram_aggregation"] = (perf_counter() - stage_started) * 1000.0

    if not present_elements:
        return (
            AuditDimension("local_chemistry", "Local chemistry", AuditStatus.UNAVAILABLE, "No atoms are loaded."),
            (),
            empty_overview,
        )
    elements = tuple(element for element in profile.elements if element in present_elements)

    stage_started = perf_counter()
    plots: list[dict[str, object]] = []
    slices: list[AuditSlice] = []
    for scope in ("angular", "radial"):
        for center in elements:
            plot, plot_slices = _plot_and_slices(
                histograms[(scope, center, "neighbor_count")],
                scope=scope,
                center=center,
                metric="neighbor_count",
                metric_label="neighbor count",
                fraction_metric=False,
            )
            plots.append(plot)
            slices.extend(plot_slices)
            for neighbor_element in elements:
                metric = f"neighbor_fraction_{neighbor_element}"
                plot, plot_slices = _plot_and_slices(
                    histograms[(scope, center, metric)],
                    scope=scope,
                    center=center,
                    metric=metric,
                    metric_label=f"{neighbor_element} neighbor fraction",
                    fraction_metric=True,
                )
                plots.append(plot)
                slices.extend(plot_slices)
    timings_ms["plot_assembly"] = (perf_counter() - stage_started) * 1000.0

    dimension = AuditDimension(
        "local_chemistry",
        "Local chemistry",
        AuditStatus.AVAILABLE,
        plots=tuple(plots),
    )
    overview = {
        "available_scopes": ("angular", "radial"),
        "center_element_count": len(elements),
        "sparse_bin_count": len(slices),
        "timings_ms": {
            "total": round((perf_counter() - audit_started) * 1000.0, 3),
            "stages": {key: round(value, 3) for key, value in timings_ms.items()},
        },
    }
    return dimension, tuple(slices), overview
