"""Element-pair contact support under the active NEP cutoff rules."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import sqrt
from typing import Iterable

import numpy as np

from .nep_cutoff import NepCutoffProfile
from .result import AuditBiasType, AuditDimension, AuditSeverity, AuditSlice, AuditStatus, SliceMetric


@dataclass
class _PairStats:
    co_occurring: dict[int, None] = field(default_factory=dict)
    contact_structures: dict[int, None] = field(default_factory=dict)
    contact_edges: int = 0
    expected_edges: float = 0.0
    opportunity_edges: int = 0
    a_centers: int = 0
    b_centers: int = 0
    a_exposed: int = 0
    b_exposed: int = 0
    normalized_distances: list[float] = field(default_factory=list)
    detail_saturated: bool = False


def _pair_probability(count_a: int, count_b: int, atom_count: int, *, same_element: bool) -> float:
    if atom_count < 2:
        return 0.0
    denominator = atom_count * (atom_count - 1)
    if same_element:
        return count_a * (count_a - 1) / denominator
    return 2.0 * count_a * count_b / denominator


def _quantile(values: Iterable[float], fraction: float) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return None
    return round(float(np.quantile(array, fraction)), 4)


class PairContactCollector:
    """Accumulate contact support while the local-chemistry scan owns neighbor recovery."""

    def __init__(self, profile: NepCutoffProfile) -> None:
        self._profile = profile
        self._elements = profile.elements
        self._element_indices = {element: index for index, element in enumerate(self._elements)}
        self._pairs = tuple(combinations_with_replacement(self._elements, 2))
        self._stats: dict[tuple[str, str, str], _PairStats] = defaultdict(_PairStats)
        self._present: set[str] = set()

    def detail_mask(self) -> np.ndarray:
        """Return which scope/pair groups still need distance-detail samples."""
        return np.asarray(
            [
                [
                    not self._stats[(scope, first, second)].detail_saturated
                    for first, second in self._pairs
                ]
                for scope in ("angular", "radial")
            ],
            dtype=np.uint8,
        )

    def observe(
        self,
        structure_index: int,
        symbols: tuple[str, ...],
        centers: np.ndarray,
        neighbors: np.ndarray,
        distances: np.ndarray,
        cutoff_matrices: dict[str, np.ndarray],
    ) -> None:
        self._present.update(symbols)
        atom_count = len(symbols)
        if atom_count == 0:
            return
        atom_types = np.asarray([self._element_indices[symbol] for symbol in symbols], dtype=np.intp)
        counts = np.bincount(atom_types, minlength=len(self._elements))
        same_parent = centers == neighbors

        for scope, cutoff_matrix in cutoff_matrices.items():
            for first, second in self._pairs:
                first_index = self._element_indices[first]
                second_index = self._element_indices[second]
                first_count = int(counts[first_index])
                second_count = int(counts[second_index])
                same_element = first == second
                co_occurring = first_count >= (2 if same_element else 1) and second_count >= 1
                stats = self._stats[(scope, first, second)]
                if not co_occurring:
                    continue
                stats.co_occurring[structure_index] = None

                cutoff = float(cutoff_matrix[first_index, second_index])
                geometric = distances < cutoff
                if not same_element:
                    geometric &= ~same_parent
                nonself_opportunities = int(np.count_nonzero(geometric & ~same_parent))
                self_opportunities = int(np.count_nonzero(geometric & same_parent))
                stats.opportunity_edges += nonself_opportunities + self_opportunities
                expected = nonself_opportunities * _pair_probability(
                    first_count, second_count, atom_count, same_element=same_element
                )
                if same_element and self_opportunities:
                    expected += self_opportunities * first_count / atom_count
                stats.expected_edges += expected

                center_types = atom_types[centers]
                neighbor_types = atom_types[neighbors]
                if same_element:
                    actual = geometric & (center_types == first_index) & (neighbor_types == first_index)
                    first_centers = atom_types == first_index
                    exposed_first = np.zeros(atom_count, dtype=np.bool_)
                    exposed_first[centers[actual]] = True
                    stats.a_centers += int(np.count_nonzero(first_centers))
                    stats.a_exposed += int(np.count_nonzero(exposed_first))
                else:
                    actual = geometric & (
                        ((center_types == first_index) & (neighbor_types == second_index))
                        | ((center_types == second_index) & (neighbor_types == first_index))
                    )
                    first_centers = atom_types == first_index
                    second_centers = atom_types == second_index
                    first_exposed = np.zeros(atom_count, dtype=np.bool_)
                    second_exposed = np.zeros(atom_count, dtype=np.bool_)
                    first_exposed[centers[geometric & (center_types == first_index) & (neighbor_types == second_index)]] = True
                    second_exposed[centers[geometric & (center_types == second_index) & (neighbor_types == first_index)]] = True
                    stats.a_centers += int(np.count_nonzero(first_centers))
                    stats.b_centers += int(np.count_nonzero(second_centers))
                    stats.a_exposed += int(np.count_nonzero(first_exposed))
                    stats.b_exposed += int(np.count_nonzero(second_exposed))

                observed = int(np.count_nonzero(actual))
                stats.contact_edges += observed
                if observed:
                    stats.contact_structures[structure_index] = None
                    if not stats.detail_saturated:
                        stats.normalized_distances.extend((distances[actual] / cutoff).tolist())
                if stats.contact_edges >= 20 and len(stats.contact_structures) >= 3:
                    stats.detail_saturated = True
                    stats.normalized_distances.clear()

    def observe_batch(
        self,
        structure_indices: list[int],
        symbols_by_structure: list[tuple[str, ...]],
        metrics: np.ndarray,
        normalized_codes: np.ndarray,
        normalized_values: np.ndarray,
    ) -> None:
        """Merge native descriptive contact aggregates without applying policy."""
        for symbols in symbols_by_structure:
            self._present.update(symbols)
        expected_shape = (len(structure_indices), 2, len(self._pairs), 8)
        if metrics.shape != expected_shape:
            raise ValueError(f"Contact summary shape {metrics.shape} does not match {expected_shape}.")
        source_indices = np.asarray(structure_indices, dtype=np.int64)
        for scope_index, scope in enumerate(("angular", "radial")):
            for pair_index, (first, second) in enumerate(self._pairs):
                values = metrics[:, scope_index, pair_index, :]
                co_occurring = values[:, 0] != 0.0
                if not np.any(co_occurring):
                    continue
                stats = self._stats[(scope, first, second)]
                for structure_index in source_indices[co_occurring]:
                    stats.co_occurring[int(structure_index)] = None
                stats.opportunity_edges += int(np.sum(values[:, 1], dtype=np.float64))
                stats.expected_edges += sum(values[:, 2].tolist())
                contacts = values[:, 3].astype(np.int64, copy=False)
                stats.contact_edges += int(np.sum(contacts, dtype=np.int64))
                stats.a_centers += int(np.sum(values[:, 4], dtype=np.float64))
                stats.b_centers += int(np.sum(values[:, 5], dtype=np.float64))
                stats.a_exposed += int(np.sum(values[:, 6], dtype=np.float64))
                stats.b_exposed += int(np.sum(values[:, 7], dtype=np.float64))
                for structure_index in source_indices[contacts > 0]:
                    stats.contact_structures[int(structure_index)] = None
                if stats.contact_edges >= 20 and len(stats.contact_structures) >= 3:
                    stats.detail_saturated = True
                    stats.normalized_distances.clear()
                elif not stats.detail_saturated:
                    code = scope_index * len(self._pairs) + pair_index
                    stats.normalized_distances.extend(
                        normalized_values[normalized_codes == code].tolist()
                    )

    def finalize(self) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
        elements = tuple(element for element in self._elements if element in self._present)
        if len(elements) < 1:
            return (
                AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, "No atoms are loaded."),
                (),
                {"pair_count": 0, "co_occurring_pair_count": 0, "zero_contact_pair_count": 0},
            )

        plots = []
        slices = []
        co_occurring_count = 0
        zero_contact_count = 0
        for scope in ("angular", "radial"):
            labels = []
            counts = []
            groups = []
            highlighted = []
            for first, second in combinations_with_replacement(elements, 2):
                stats = self._stats[(scope, first, second)]
                labels.append(f"{first}-{second}")
                counts.append(stats.contact_edges)
                groups.append(tuple(stats.contact_structures or stats.co_occurring))
                if stats.co_occurring and stats.contact_edges == 0:
                    highlighted.append(len(labels) - 1)
                if stats.co_occurring:
                    co_occurring_count += 1
                if stats.co_occurring and stats.contact_edges == 0:
                    zero_contact_count += 1

                ratio = None if stats.expected_edges <= 0.0 else stats.contact_edges / stats.expected_edges
                support_label = "not co-occurring" if not stats.co_occurring else (
                    "co-occurring; no local contact" if stats.contact_edges == 0 else
                    "observed; low support" if stats.contact_edges < 20 or len(stats.contact_structures) < 3 else
                    "observed; multi-structure support"
                )
                if not stats.co_occurring or stats.contact_edges == 0 or support_label == "observed; low support":
                    slices.append(
                        AuditSlice(
                            id=f"pair_contacts:{scope}:{first}:{second}",
                            title=f"{scope.title()} {first}-{second}: {support_label}",
                            dimension_id="pair_contacts",
                            severity=AuditSeverity.INFO,
                            bias_type=AuditBiasType.INFORMATIONAL,
                            structure_indices=tuple(stats.contact_structures or stats.co_occurring),
                            observed=(
                                f"{stats.contact_edges} directed NEP-cutoff contact edges across "
                                f"{len(stats.contact_structures)} of {len(stats.co_occurring)} co-occurring structures."
                            ),
                            interpretation="This is a dataset-support observation, not a sampling recommendation.",
                            limit="O/E is shown as a descriptive same-geometry, same-composition reference; low expected counts make the ratio unstable.",
                            metrics=(
                                SliceMetric("co_occurring_structures", len(stats.co_occurring), "structures"),
                                SliceMetric("contact_structures", len(stats.contact_structures), "structures"),
                                SliceMetric("contact_edges", stats.contact_edges, "directed edges"),
                                SliceMetric("geometric_opportunities", stats.opportunity_edges, "directed edges"),
                                SliceMetric("expected_edges", round(stats.expected_edges, 4), "directed edges"),
                                SliceMetric("observed_expected_ratio", None if ratio is None else round(ratio, 4)),
                                SliceMetric("first_center_exposure", round(stats.a_exposed / stats.a_centers, 4) if stats.a_centers else None),
                                SliceMetric("second_center_exposure", round(stats.b_exposed / stats.b_centers, 4) if stats.b_centers else None),
                                SliceMetric("normalized_distance_q50", _quantile(stats.normalized_distances, 0.5)),
                                SliceMetric("near_cutoff_fraction", round(sum(value > 0.95 for value in stats.normalized_distances) / len(stats.normalized_distances), 4) if stats.normalized_distances else None),
                            ),
                        )
                    )
            plots.append(
                {
                    "kind": "categorical_bars",
                    "id": f"pair_contacts:{scope}",
                    "title": f"{scope.title()} element-pair contact edges",
                    "x_label": "Directed NEP-cutoff contact edges",
                    "y_label": "Element pair",
                    "series": ({"id": scope, "label": scope, "labels": tuple(labels), "counts": tuple(counts), "highlighted_bins": tuple(highlighted), "structure_indices": tuple(groups)},),
                }
            )

        return (
            AuditDimension("pair_contacts", "Pair contacts", AuditStatus.AVAILABLE, plots=tuple(plots)),
            tuple(slices),
            {"pair_count": len(self._pairs), "co_occurring_pair_count": co_occurring_count, "zero_contact_pair_count": zero_contact_count},
        )
