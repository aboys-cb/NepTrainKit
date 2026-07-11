"""Deterministic QPainter charts for Training Set Audit plot payloads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite
from numbers import Integral
from typing import Any

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget


_BACKGROUND = QColor("#FFFFFF")
_GRID = QColor("#D7E0E2")
_TEXT = QColor("#374151")
_TEAL = QColor("#159A9C")
_ORANGE = QColor("#E8871E")


class AuditChartWidget(QWidget):
    """Render one normalized audit plot and emit clicked structure groups."""

    selectedGroupSignal = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot: dict[str, Any] | None = None
        self._bar_rects: list[tuple[QRectF, list[int] | None]] = []
        self._bar_tooltips: list[str] = []
        self.setMouseTracking(True)
        self.setMinimumHeight(220)

    @property
    def plot_id(self) -> str:
        return "" if self._plot is None else self._plot["id"]

    @property
    def has_data(self) -> bool:
        return self._plot is not None

    @property
    def empty_state_text(self) -> str:
        return self.tr("No numeric distribution available")

    def sizeHint(self) -> QSize:
        return QSize(640, 260)

    def set_plot(self, plot: Mapping[str, Any] | None) -> None:
        """Normalize and render one plot payload emitted by an audit dimension."""
        self._plot = self._normalize_plot(plot)
        self._bar_rects = []
        self._bar_tooltips = []
        self.update()

    def clear(self) -> None:
        """Discard the current plot and render the empty state."""
        self._plot = None
        self._bar_rects = []
        self._bar_tooltips = []
        self.update()

    @staticmethod
    def _normalize_plot(plot: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(plot, Mapping):
            return None

        kind = plot.get("kind")
        plot_id = plot.get("id")
        if kind not in {"histogram", "categorical_bars"} or not isinstance(plot_id, str) or not plot_id:
            return None

        series_items = plot.get("series")
        if not isinstance(series_items, Sequence) or isinstance(series_items, (str, bytes)) or not series_items:
            return None
        series = series_items[0]
        if not isinstance(series, Mapping):
            return None

        counts = AuditChartWidget._numeric_values(series.get("counts"))
        if not counts:
            return None
        index_groups = AuditChartWidget._structure_groups(series.get("structure_indices"), len(counts))
        if index_groups is None:
            return None
        highlighted_bins = AuditChartWidget._highlighted_bins(series.get("highlighted_bins"), len(counts))

        normalized: dict[str, Any] = {
            "kind": kind,
            "id": plot_id,
            "title": str(plot.get("title", "")),
            "x_label": str(plot.get("x_label", "")),
            "y_label": str(plot.get("y_label", "")),
            "counts": counts,
            "highlighted_bins": highlighted_bins,
            "structure_indices": index_groups,
        }
        if kind == "histogram":
            edges = AuditChartWidget._numeric_values(series.get("bin_edges"), nonnegative=False)
            if len(edges) != len(counts) + 1 or any(right <= left for left, right in zip(edges, edges[1:])):
                return None
            normalized["bin_edges"] = edges
            labels = series.get("bin_labels")
            if (
                isinstance(labels, Sequence)
                and not isinstance(labels, (str, bytes))
                and len(labels) == len(counts)
            ):
                normalized["bin_labels"] = tuple(str(label) for label in labels)
        else:
            labels = series.get("labels")
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) != len(counts):
                return None
            normalized["labels"] = tuple(str(label) for label in labels)
        return normalized

    @staticmethod
    def _numeric_values(values: Any, *, nonnegative: bool = True) -> tuple[float, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return ()
        result: list[float] = []
        for value in values:
            if isinstance(value, bool):
                return ()
            try:
                number = float(value)
            except (TypeError, ValueError):
                return ()
            if not isfinite(number) or (nonnegative and number < 0):
                return ()
            result.append(number)
        return tuple(result)

    @staticmethod
    def _structure_groups(values: Any, count: int) -> tuple[list[int] | None, ...] | None:
        if values is None:
            return tuple(None for _ in range(count))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != count:
            return None

        groups: list[list[int] | None] = []
        for group in values:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                return None
            if any(isinstance(index, bool) or not isinstance(index, Integral) for index in group):
                return None
            groups.append([int(index) for index in group])
        return tuple(groups)

    @staticmethod
    def _highlighted_bins(values: Any, count: int) -> frozenset[int]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return frozenset()
        highlighted: set[int] = set()
        for value in values:
            if isinstance(value, bool):
                continue
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < count:
                highlighted.add(index)
        return frozenset(highlighted)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt hook
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BACKGROUND)
        self._bar_rects = []
        self._bar_tooltips = []
        if self._plot is None:
            painter.setPen(_TEXT)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.empty_state_text,
            )
            return

        self._draw_title(painter)
        if self._plot["kind"] == "histogram":
            self._draw_histogram(painter)
        else:
            self._draw_categorical_bars(painter)

    def _draw_title(self, painter: QPainter) -> None:
        title_font = QFont(painter.font())
        title_font.setPointSize(max(9, title_font.pointSize()))
        painter.setFont(title_font)
        painter.setPen(_TEXT)
        painter.drawText(QRectF(8, 6, max(1, self.width() - 16), 22), Qt.AlignmentFlag.AlignLeft, self._plot["title"])
        if self._plot["highlighted_bins"]:
            legend = self.tr("Orange = low-frequency range")
            metrics = QFontMetrics(painter.font())
            width = metrics.horizontalAdvance(legend)
            painter.fillRect(QRectF(self.width() - width - 32, 12, 10, 10), _ORANGE)
            painter.drawText(
                QRectF(self.width() - width - 18, 5, width + 10, 22),
                Qt.AlignmentFlag.AlignVCenter,
                legend,
            )

    def _draw_histogram(self, painter: QPainter) -> None:
        chart = QRectF(58, 38, max(1, self.width() - 74), max(1, self.height() - 82))
        self._draw_grid(painter, chart, horizontal=True)
        counts = self._plot["counts"]
        max_count = max(counts) if max(counts) > 0 else 1.0
        slot_width = chart.width() / len(counts)
        bar_margin = min(8.0, slot_width * 0.15)
        for index, count in enumerate(counts):
            height = chart.height() * count / max_count
            rect = QRectF(
                chart.left() + index * slot_width + bar_margin,
                chart.bottom() - height,
                max(1.0, slot_width - 2 * bar_margin),
                height,
            )
            painter.fillRect(rect, _ORANGE if index in self._plot["highlighted_bins"] else _TEAL)
            self._bar_rects.append((rect, self._plot["structure_indices"][index]))
            if "bin_labels" in self._plot:
                label = self._plot["bin_labels"][index]
            else:
                label = f"{self._plot['bin_edges'][index]:g}–{self._plot['bin_edges'][index + 1]:g}"
            self._bar_tooltips.append(f"{label}: {count:g}")
            if rect.width() >= 24 and rect.height() >= 12:
                painter.setPen(_TEXT)
                painter.drawText(
                    QRectF(rect.left(), max(chart.top(), rect.top() - 17), rect.width(), 15),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{count:g}",
                )

        self._draw_axis_labels(painter, chart)
        metrics = QFontMetrics(painter.font())
        if "bin_labels" in self._plot:
            labels = self._plot["bin_labels"]
            step = max(1, ceil(len(labels) / max(1, self.width() // 110)))
            for index, label in enumerate(labels):
                if index % step and index != len(labels) - 1:
                    continue
                x = chart.left() + slot_width * (index + 0.5)
                width = min(slot_width * step, metrics.horizontalAdvance(label) + 4)
                painter.drawText(
                    QRectF(x - width / 2, chart.bottom() + 4, width, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
        else:
            edges = self._plot["bin_edges"]
            step = max(1, ceil((len(edges) - 1) / max(1, self.width() // 110)))
            for index, edge in enumerate(edges):
                if index % step and index != len(edges) - 1:
                    continue
                x = chart.left() + chart.width() * index / (len(edges) - 1)
                label = f"{edge:g}"
                width = metrics.horizontalAdvance(label)
                painter.drawText(
                    QRectF(x - width / 2, chart.bottom() + 4, width + 2, 16),
                    Qt.AlignmentFlag.AlignLeft,
                    label,
                )

    def _draw_categorical_bars(self, painter: QPainter) -> None:
        left_margin = min(170.0, max(90.0, self.width() * 0.28))
        chart = QRectF(left_margin, 38, max(1, self.width() - left_margin - 76), max(1, self.height() - 78))
        self._draw_grid(painter, chart, horizontal=False)
        counts = self._plot["counts"]
        max_count = max(counts) if max(counts) > 0 else 1.0
        row_height = chart.height() / len(counts)
        bar_height = max(1.0, row_height * 0.62)
        metrics = QFontMetrics(painter.font())
        for index, (label, count) in enumerate(zip(self._plot["labels"], counts)):
            y = chart.top() + index * row_height + (row_height - bar_height) / 2
            rect = QRectF(chart.left(), y, chart.width() * count / max_count, bar_height)
            painter.fillRect(rect, _ORANGE if index in self._plot["highlighted_bins"] else _TEAL)
            elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(left_margin - 12))
            painter.drawText(
                QRectF(6, y, left_margin - 12, bar_height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                elided,
            )
            self._bar_rects.append((rect, self._plot["structure_indices"][index]))
            self._bar_tooltips.append(f"{label}: {count:g}")
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(rect.right() + 4, y, 70, bar_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{count:g}",
            )

        self._draw_axis_labels(painter, chart)

    def _draw_grid(self, painter: QPainter, chart: QRectF, *, horizontal: bool) -> None:
        painter.setPen(QPen(_GRID, 1))
        max_count = max(self._plot["counts"]) if max(self._plot["counts"]) > 0 else 1.0
        metrics = QFontMetrics(painter.font())
        for index in range(5):
            fraction = index / 4
            if horizontal:
                y = chart.top() + chart.height() * fraction
                painter.drawLine(chart.left(), y, chart.right(), y)
                label = str(int(round(max_count * (1.0 - fraction))))
                painter.setPen(_TEXT)
                painter.drawText(
                    QRectF(2, y - 8, chart.left() - 8, 16),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
                painter.setPen(QPen(_GRID, 1))
            else:
                x = chart.left() + chart.width() * fraction
                painter.drawLine(x, chart.top(), x, chart.bottom())
                label = str(int(round(max_count * fraction)))
                width = metrics.horizontalAdvance(label)
                painter.setPen(_TEXT)
                painter.drawText(
                    QRectF(x - width / 2, chart.bottom() + 3, width + 2, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
                painter.setPen(QPen(_GRID, 1))
        painter.setPen(QPen(_TEXT, 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

    def _draw_axis_labels(self, painter: QPainter, chart: QRectF) -> None:
        axis_font = QFont(painter.font())
        axis_font.setPointSize(max(7, axis_font.pointSize() - 1))
        painter.setFont(axis_font)
        painter.setPen(_TEXT)
        painter.drawText(
            QRectF(chart.left(), self.height() - 21, chart.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self._plot["x_label"],
        )
        painter.drawText(
            QRectF(6, 24, max(1, chart.left() - 12), 14),
            Qt.AlignmentFlag.AlignLeft,
            self._plot["y_label"],
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt hook
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            for rect, structure_indices in self._bar_rects:
                if rect.contains(point):
                    if structure_indices is not None:
                        self.selectedGroupSignal.emit(list(structure_indices))
                    break
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt hook
        point = event.position()
        tooltip = ""
        for index, (rect, _) in enumerate(self._bar_rects):
            if rect.contains(point):
                tooltip = self._bar_tooltips[index]
                break
        self.setToolTip(tooltip)
        super().mouseMoveEvent(event)
