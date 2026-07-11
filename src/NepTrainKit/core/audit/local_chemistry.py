"""Local-chemistry audit based on the active NEP model cutoffs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

from NepTrainKit.core.structure import Structure

from .nep_cutoff import NepCutoffProfile
from .result import AuditBiasType, AuditDimension, AuditSeverity, AuditSlice, AuditStatus, SliceMetric


_SCOPE_TITLES = {"angular": "Angular core", "radial": "Radial context"}
_FRACTION_EDGES = tuple(index / 10.0 for index in range(11))
_FRACTION_LABELS = tuple(f"{10 * index}-{10 * (index + 1)}%" for index in range(10))


@dataclass
class _Histogram:
    counts: Counter[int] = field(default_factory=Counter)
    structure_indices: dict[int, dict[int, None]] = field(default_factory=lambda: defaultdict(dict))
    sample_count: int = 0

    def add(self, bin_index: int, structure_index: int) -> None:
        self.counts[bin_index] += 1
        self.structure_indices[bin_index][structure_index] = None
        self.sample_count += 1


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


def _as_atoms(structure: Structure, model_elements: set[str]) -> tuple[Atoms, tuple[str, ...]]:
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
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=_pbc_flags(structure)), symbols


def _pair_cutoffs(profile: NepCutoffProfile, elements: tuple[str, ...], scope: str) -> dict[tuple[int, int], float]:
    cutoffs: dict[tuple[int, int], float] = {}
    for first_index, first in enumerate(elements):
        for second in elements[first_index:]:
            cutoffs[(atomic_numbers[first], atomic_numbers[second])] = profile.pair_cutoff(first, second, scope)
    return cutoffs


def _cutoff_matrix(profile: NepCutoffProfile, elements: tuple[str, ...], scope: str) -> np.ndarray:
    cutoffs = np.asarray(
        [profile.pair_cutoff(element, element, scope) for element in elements],
        dtype=np.float64,
    )
    return 0.5 * (cutoffs[:, np.newaxis] + cutoffs[np.newaxis, :])


def _compiled_neighbor_pairs(
    atoms: Atoms,
    pair_cutoffs: dict[tuple[int, int], float],
    cutoff_matrix: np.ndarray,
    element_indices: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed NEP-cutoff neighbors through SciPy's compiled KD-tree."""
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    pbc = np.asarray(atoms.pbc, dtype=np.bool_)
    atom_count = len(atoms)
    if atom_count == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, np.empty(0, dtype=np.float64)

    image_indices = np.arange(atom_count, dtype=np.intp)
    image_shifts = np.zeros((atom_count, 3), dtype=np.intp)
    if np.any(pbc):
        try:
            inverse_cell = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            return neighbor_list("ijd", atoms, pair_cutoffs, self_interaction=False)

        scaled_positions = positions @ inverse_cell
        scaled_positions[:, pbc] %= 1.0
        positions = scaled_positions @ cell
        face_distances = 1.0 / np.linalg.norm(inverse_cell.T, axis=1)
        max_cutoff = float(np.max(cutoff_matrix))
        shift_ranges = []
        for axis in range(3):
            if pbc[axis]:
                image_count = int(np.ceil(max_cutoff / face_distances[axis]))
                shift_ranges.append(range(-image_count, image_count + 1))
            else:
                shift_ranges.append((0,))
        shifts = np.asarray(tuple(product(*shift_ranges)), dtype=np.intp)
        image_positions = (positions[np.newaxis, :, :] + (shifts @ cell)[:, np.newaxis, :]).reshape(-1, 3)
        image_indices = np.tile(image_indices, len(shifts))
        image_shifts = np.repeat(shifts, atom_count, axis=0)
    else:
        image_positions = positions

    atom_element_indices = np.asarray([element_indices[number] for number in atoms.numbers], dtype=np.intp)
    tree = cKDTree(image_positions)
    image_element_indices = atom_element_indices[image_indices]
    center_parts: list[np.ndarray] = []
    neighbor_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for center, candidates in enumerate(tree.query_ball_point(positions, float(np.max(cutoff_matrix)))):
        candidate_image_indices = np.asarray(candidates, dtype=np.intp)
        candidate_neighbors = image_indices[candidate_image_indices]
        candidate_distances = np.linalg.norm(image_positions[candidate_image_indices] - positions[center], axis=1)
        candidate_cutoffs = cutoff_matrix[atom_element_indices[center], image_element_indices[candidate_image_indices]]
        self_mask = (candidate_neighbors == center) & np.all(image_shifts[candidate_image_indices] == 0, axis=1)
        valid = (candidate_distances < candidate_cutoffs) & ~self_mask
        if np.any(valid):
            count = int(np.count_nonzero(valid))
            center_parts.append(np.full(count, center, dtype=np.intp))
            neighbor_parts.append(candidate_neighbors[valid])
            distance_parts.append(candidate_distances[valid])

    if not center_parts:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, np.empty(0, dtype=np.float64)
    return np.concatenate(center_parts), np.concatenate(neighbor_parts), np.concatenate(distance_parts)


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
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    """Audit per-center neighbor counts and chemical fractions at NEP cutoffs."""
    empty_overview = {"available_scopes": (), "center_element_count": 0, "sparse_bin_count": 0}
    if not indexed_structures:
        return (
            AuditDimension("local_chemistry", "Local chemistry", AuditStatus.UNAVAILABLE, "No structures are loaded."),
            (),
            empty_overview,
        )

    elements = profile.elements
    model_elements = set(elements)
    histograms: dict[tuple[str, str, str], _Histogram] = defaultdict(_Histogram)
    max_radial_cutoff = float(max(profile.radial_cutoffs))
    radial_pair_cutoffs = {
        (atomic_numbers[first], atomic_numbers[second]): max_radial_cutoff
        for first_index, first in enumerate(elements)
        for second in elements[first_index:]
    }
    candidate_cutoff_matrix = np.full((len(elements), len(elements)), max_radial_cutoff, dtype=np.float64)
    radial_cutoff_matrix = _cutoff_matrix(profile, elements, "radial")
    angular_cutoff_matrix = _cutoff_matrix(profile, elements, "angular")
    element_indices = {element: index for index, element in enumerate(elements)}
    element_number_indices = {atomic_numbers[element]: index for element, index in element_indices.items()}
    present_elements: set[str] = set()
    for structure_index, structure in indexed_structures:
        atoms, symbols = _as_atoms(structure, model_elements)
        present_elements.update(symbols)
        centers, neighbors, distances = _compiled_neighbor_pairs(
            atoms,
            radial_pair_cutoffs,
            candidate_cutoff_matrix,
            element_number_indices,
        )
        atom_element_indices = np.asarray([element_indices[symbol] for symbol in symbols], dtype=np.intp)
        neighbor_element_indices = atom_element_indices[neighbors]
        radial_mask = distances < radial_cutoff_matrix[
            atom_element_indices[centers], neighbor_element_indices
        ]
        angular_mask = distances < angular_cutoff_matrix[
            atom_element_indices[centers], neighbor_element_indices
        ]
        if pair_contact_collector is not None:
            pair_contact_collector.observe(
                structure_index,
                symbols,
                centers,
                neighbors,
                distances,
                {"angular": angular_cutoff_matrix, "radial": radial_cutoff_matrix},
            )
        for scope, scope_centers, scope_neighbor_element_indices in (
            ("angular", centers[angular_mask], neighbor_element_indices[angular_mask]),
            ("radial", centers[radial_mask], neighbor_element_indices[radial_mask]),
        ):
            neighbor_counts = np.bincount(scope_centers, minlength=len(symbols))
            neighbor_element_counts = {
                neighbor_element: np.bincount(
                    scope_centers[scope_neighbor_element_indices == neighbor_index],
                    minlength=len(symbols),
                )
                for neighbor_index, neighbor_element in enumerate(elements)
            }
            for atom_index, center in enumerate(symbols):
                count = int(neighbor_counts[atom_index])
                histograms[(scope, center, "neighbor_count")].add(count, structure_index)
                for neighbor_element in elements:
                    fraction = 0.0 if count == 0 else neighbor_element_counts[neighbor_element][atom_index] / count
                    fraction_bin = min(int(float(fraction) * 10.0), 9)
                    histograms[(scope, center, f"neighbor_fraction_{neighbor_element}")].add(
                        fraction_bin,
                        structure_index,
                    )

    if not present_elements:
        return (
            AuditDimension("local_chemistry", "Local chemistry", AuditStatus.UNAVAILABLE, "No atoms are loaded."),
            (),
            empty_overview,
        )
    elements = tuple(element for element in profile.elements if element in present_elements)

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
    }
    return dimension, tuple(slices), overview
