"""Label-range audit dimension."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .extract import StructureAuditRecord
from .result import AuditBiasType, AuditDimension, AuditSeverity, AuditSlice, AuditStatus, SliceMetric


def _label_histogram_plot(
    metric_id: str,
    title: str,
    x_label: str,
    records: Sequence[StructureAuditRecord],
    values: Sequence[float | None],
) -> dict[str, object] | None:
    labeled = [(record.index, float(value)) for record, value in zip(records, values) if value is not None and np.isfinite(value)]
    if not labeled:
        return None

    numeric_values = np.asarray([value for _, value in labeled], dtype=np.float64)
    counts, bin_edges = np.histogram(numeric_values, bins=min(20, len(numeric_values)))
    index_groups: list[list[int]] = [[] for _ in range(len(counts))]
    for structure_index, value in labeled:
        bin_index = min(np.searchsorted(bin_edges, value, side="right") - 1, len(counts) - 1)
        index_groups[int(bin_index)].append(structure_index)

    return {
        "kind": "histogram",
        "id": f"label_ranges:{metric_id}",
        "title": title,
        "x_label": x_label,
        "y_label": "Structures",
        "series": (
            {
                "id": metric_id,
                "label": x_label,
                "bin_edges": tuple(float(edge) for edge in bin_edges),
                "counts": tuple(int(count) for count in counts),
                "structure_indices": tuple(tuple(group) for group in index_groups),
            },
        ),
        "labeled_count": len(labeled),
        "total_count": len(records),
    }


def _tail_selection(
    records: Sequence[StructureAuditRecord],
    values: list[float | None],
    quantile: float,
) -> tuple[list[int], float | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size < 4:
        return [], None
    threshold = float(np.quantile(finite, quantile))
    if not np.isfinite(threshold):
        return [], None
    return (
        [
            record.index
            for record, value in zip(records, values)
            if value is not None and np.isfinite(value) and value > threshold
        ],
        threshold,
    )


def audit_label_ranges(
    records: Sequence[StructureAuditRecord],
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    if not records:
        return (
            AuditDimension("label_ranges", "Label ranges", AuditStatus.UNAVAILABLE, "No structures are loaded."),
            (),
            {
                "has_energy": False,
                "has_force": False,
                "has_virial": False,
                "energy_labeled_count": 0,
                "force_labeled_count": 0,
                "virial_labeled_count": 0,
                "label_total_count": 0,
            },
        )

    slices: list[AuditSlice] = []
    total_count = len(records)

    force_values = [record.max_force for record in records]
    force_labeled_count = sum(value is not None and np.isfinite(value) for value in force_values)
    energy_values = [record.energy_per_atom for record in records]
    energy_labeled_count = sum(value is not None and np.isfinite(value) for value in energy_values)
    virial_values = [record.virial_norm for record in records]
    virial_labeled_count = sum(value is not None and np.isfinite(value) for value in virial_values)
    plots = tuple(
        plot
        for plot in (
            _label_histogram_plot("energy_per_atom", "Energy per atom distribution", "Energy per atom", records, energy_values),
            _label_histogram_plot("max_force", "Maximum force distribution", "Maximum force", records, force_values),
            _label_histogram_plot("virial_norm", "Virial norm distribution", "Virial norm", records, virial_values),
        )
        if plot is not None
    )
    high_force, high_force_threshold = _tail_selection(
        records, force_values, 0.90
    )
    if high_force:
        observed = f"{len(high_force)} structures are in the top 10% of maximum force values."
        if force_labeled_count < total_count:
            observed = (
                f"{len(high_force)} structures are in the top 10% of maximum force values within the "
                f"labeled subset ({force_labeled_count} of {total_count} structures)."
            )
        slices.append(
            AuditSlice(
                id="label_ranges:force_high_tail",
                title="High force tail",
                dimension_id="label_ranges",
                severity=AuditSeverity.HIGH if len(high_force) / force_labeled_count >= 0.20 else AuditSeverity.MEDIUM,
                bias_type=AuditBiasType.RISK_CONCENTRATION,
                structure_indices=tuple(high_force),
                observed=observed,
                interpretation="High-force structures concentrate training pressure and deserve inspection for bad geometries or difficult environments.",
                limit="High force can be physically intended; this finding is a review target, not a delete recommendation.",
                metrics=(
                    SliceMetric("tail_quantile", 0.90, "", None, "high"),
                    SliceMetric(
                        "threshold",
                        high_force_threshold,
                        "eV/angstrom",
                        None,
                        "high",
                    ),
                    SliceMetric("labeled_count", force_labeled_count),
                    SliceMetric("total_count", total_count),
                ),
            )
        )

    high_energy, high_energy_threshold = _tail_selection(
        records, energy_values, 0.95
    )
    if high_energy:
        observed = f"{len(high_energy)} structures are in the top 5% of energy per atom."
        if energy_labeled_count < total_count:
            observed = (
                f"{len(high_energy)} structures are in the top 5% of energy per atom within the "
                f"labeled subset ({energy_labeled_count} of {total_count} structures)."
            )
        slices.append(
            AuditSlice(
                id="label_ranges:energy_high_tail",
                title="High energy-per-atom tail",
                dimension_id="label_ranges",
                severity=AuditSeverity.MEDIUM,
                bias_type=AuditBiasType.RISK_CONCENTRATION,
                structure_indices=tuple(high_energy),
                observed=observed,
                interpretation="The energy tail may contain strained, defective, hot, or problematic structures.",
                limit="This does not say the structures are wrong; inspect source and geometry before acting.",
                metrics=(
                    SliceMetric("tail_quantile", 0.95, "", None, "high"),
                    SliceMetric(
                        "threshold",
                        high_energy_threshold,
                        "eV/atom",
                        None,
                        "high",
                    ),
                    SliceMetric("labeled_count", energy_labeled_count),
                    SliceMetric("total_count", total_count),
                ),
            )
        )

    overview = {
        "has_energy": bool(energy_labeled_count > 0),
        "has_force": bool(force_labeled_count > 0),
        "has_virial": bool(virial_labeled_count > 0),
        "energy_labeled_count": energy_labeled_count,
        "energy_total_count": total_count,
        "force_labeled_count": force_labeled_count,
        "force_total_count": total_count,
        "virial_labeled_count": virial_labeled_count,
        "virial_total_count": total_count,
        "label_total_count": total_count,
    }
    if not overview["has_energy"] and not overview["has_force"] and not overview["has_virial"]:
        dimension = AuditDimension("label_ranges", "Label ranges", AuditStatus.UNAVAILABLE, "No energy, force, or virial labels found.")
    else:
        fully_labeled = []
        partially_labeled = []
        for label_name, labeled_count in (
            ("energy", energy_labeled_count),
            ("force", force_labeled_count),
            ("virial", virial_labeled_count),
        ):
            if labeled_count == 0:
                continue
            if labeled_count == total_count:
                fully_labeled.append(label_name)
            else:
                partially_labeled.append(f"{label_name} ({labeled_count}/{total_count})")
        if partially_labeled:
            reason = "Available on labeled subsets only: " + ", ".join(partially_labeled) + "."
            dimension = AuditDimension("label_ranges", "Label ranges", AuditStatus.PARTIAL, reason, plots=plots)
        else:
            dimension = AuditDimension("label_ranges", "Label ranges", AuditStatus.AVAILABLE, plots=plots)
    return dimension, tuple(slices), overview
