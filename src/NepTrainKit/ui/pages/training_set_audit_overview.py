#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Presentation model for the Training Set Audit composition overview."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p

from PySide6.QtGui import QColor
from qfluentwidgets import TableItemDelegate

from NepTrainKit.core.audit.result import DatasetInventory


# Single-hue sequential scale (light → dark teal, the app accent).  A rainbow
# scale exaggerates arbitrary hue steps and makes small counts look alarming.
OVERVIEW_HEAT_COLORS = (
    "#e6f5f6",
    "#c7e9eb",
    "#a3dade",
    "#7cc9cf",
    "#52b5bd",
    "#2c9ea8",
    "#12848f",
    "#0b6a74",
    "#08525a",
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
        len(OVERVIEW_HEAT_COLORS) - 1,
        round(strength * (len(OVERVIEW_HEAT_COLORS) - 1)),
    )
    return QColor(OVERVIEW_HEAT_COLORS[palette_index])


def overview_count_text(value: int) -> str:
    """Return a compact cell label such as ``87``, ``1.2k`` or ``35k``."""
    if value < 1000:
        return str(value)
    if value < 10000:
        return f"{value / 1000:.1f}k"
    return f"{round(value / 1000)}k"


def overview_text_color(background: QColor) -> QColor:
    """Pick dark or light text that stays readable on ``background``."""
    luminance = (
        0.2126 * background.red()
        + 0.7152 * background.green()
        + 0.0722 * background.blue()
    )
    return QColor("#ffffff" if luminance < 145 else "#111827")


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
