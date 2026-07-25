"""Composition audit dimension."""
from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

from .extract import StructureAuditRecord
from .result import AuditBiasType, AuditDimension, AuditSeverity, AuditSlice, AuditStatus, SliceMetric


_COMPOSITION_EDGES = (0.0, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 1.0)
_COMPOSITION_BIN_LABELS = (
    "0-5%",
    "5-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-95%",
    "95-100%",
)


def _composition_bin_index(fraction: float) -> int:
    """Use left-closed fixed bins, with the final 100% endpoint included."""
    return min(
        max(bisect_right(_COMPOSITION_EDGES, fraction) - 1, 0),
        len(_COMPOSITION_BIN_LABELS) - 1,
    )


def _severity_for_fraction(fraction: float) -> AuditSeverity:
    if fraction < 0.03:
        return AuditSeverity.HIGH
    if fraction < 0.08:
        return AuditSeverity.MEDIUM
    return AuditSeverity.LOW


def audit_composition(
    records: Sequence[StructureAuditRecord],
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    if not records:
        return (
            AuditDimension("composition", "Composition", AuditStatus.UNAVAILABLE, "No structures are loaded."),
            (),
            {"element_count": 0, "structure_count": 0, "sparse_bin_count": 0},
        )

    elements = sorted({element for record in records for element in record.composition})
    if not elements:
        return (
            AuditDimension("composition", "Composition", AuditStatus.UNAVAILABLE, "No element information found."),
            (),
            {"element_count": 0, "structure_count": len(records), "sparse_bin_count": 0},
        )

    total = len(records)
    slices: list[AuditSlice] = []
    plots: list[dict[str, object]] = []
    for element in elements:
        index_groups: list[list[int]] = [[] for _ in range(len(_COMPOSITION_EDGES) - 1)]
        for record in records:
            fraction = float(record.composition.get(element, 0.0))
            bin_index = _composition_bin_index(fraction)
            index_groups[bin_index].append(record.index)

        counts = tuple(len(group) for group in index_groups)
        sparse_bin_indices = tuple(
            index for index, group in enumerate(index_groups) if group and len(group) / total < 0.10
        )
        plots.append(
            {
                "kind": "histogram",
                "id": f"composition:{element}",
                "title": f"{element} concentration distribution",
                "x_label": "Atomic fraction",
                "y_label": "Structures",
                "series": (
                    {
                        "id": element,
                        "label": element,
                        "bin_edges": _COMPOSITION_EDGES,
                        "bin_labels": _COMPOSITION_BIN_LABELS,
                        "counts": counts,
                        "highlighted_bins": sparse_bin_indices,
                        "structure_indices": tuple(tuple(group) for group in index_groups),
                    },
                ),
            }
        )

        for label, indices in zip(_COMPOSITION_BIN_LABELS, index_groups):
            if not indices:
                continue
            fraction = len(indices) / total
            if fraction >= 0.10:
                continue
            slices.append(
                AuditSlice(
                    id=f"composition:{element}:{label}",
                    title=f"Sparse composition bin: {element} {label}",
                    dimension_id="composition",
                    severity=_severity_for_fraction(fraction),
                    bias_type=AuditBiasType.SPARSITY,
                    structure_indices=tuple(indices),
                    observed=f"{element} {label} contains {len(indices)} of {total} structures ({fraction:.1%}).",
                    interpretation="This composition region is thin relative to the current dataset distribution.",
                    limit="A sparse bin is only actionable if it matters for the intended training or evaluation target.",
                    metrics=(
                        SliceMetric("structure_count", len(indices), "structures", total, "low"),
                        SliceMetric("dataset_fraction", round(fraction, 4), "", None, "low"),
                    ),
                )
            )

    dimension = AuditDimension("composition", "Composition", AuditStatus.AVAILABLE, plots=tuple(plots))
    overview = {
        "element_count": len(elements),
        "structure_count": total,
        "sparse_bin_count": len(slices),
    }
    return dimension, tuple(slices), overview
