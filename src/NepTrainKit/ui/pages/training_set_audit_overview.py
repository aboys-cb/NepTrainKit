#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Presentation model for the Training Set Audit composition overview."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p

from PySide6.QtGui import QColor
from qfluentwidgets import TableItemDelegate

from NepTrainKit.core.audit.result import DatasetInventory


OVERVIEW_JET_COLORS = (
    "#000080",
    "#0000ff",
    "#007fff",
    "#00dfff",
    "#40ff80",
    "#dfff20",
    "#ffbf00",
    "#ff4000",
    "#800000",
)


@dataclass(frozen=True)
class ElementSetSummary:
    """One exact set of present elements, aggregated across stoichiometries."""

    elements: tuple[str, ...]
    structure_count: int
    structure_indices: tuple[int, ...]


class MatrixItemDelegate(TableItemDelegate):
    """Keep cell selection without the row-oriented Fluent indicator."""

    def setSelectedRows(self, indexes) -> None:
        del indexes
        self.selectedRows.clear()


def overview_heat_color(value: int, maximum: int) -> QColor:
    """Map a positive count to the discrete overview color scale."""
    if value <= 0 or maximum <= 0:
        return QColor("#f8fafc")
    strength = log1p(value) / log1p(maximum)
    palette_index = min(
        len(OVERVIEW_JET_COLORS) - 1,
        round(strength * (len(OVERVIEW_JET_COLORS) - 1)),
    )
    return QColor(OVERVIEW_JET_COLORS[palette_index])


def element_set_summaries(
    inventory: DatasetInventory,
) -> tuple[ElementSetSummary, ...]:
    """Aggregate exact compositions into element-presence sets."""
    grouped_counts: dict[tuple[str, ...], int] = {}
    grouped_indices: dict[tuple[str, ...], set[int]] = {}
    for point in inventory.composition_points:
        element_set = tuple(
            element
            for element, count in zip(inventory.elements, point.reduced_counts)
            if int(count) > 0
        )
        if not element_set:
            continue
        grouped_counts[element_set] = (
            grouped_counts.get(element_set, 0) + int(point.structure_count)
        )
        grouped_indices.setdefault(element_set, set()).update(
            int(index) for index in point.structure_indices
        )
    return tuple(
        sorted(
            (
                ElementSetSummary(
                    elements=elements,
                    structure_count=grouped_counts[elements],
                    structure_indices=tuple(sorted(grouped_indices[elements])),
                )
                for elements in grouped_counts
            ),
            key=lambda item: (-item.structure_count, item.elements),
        )
    )
