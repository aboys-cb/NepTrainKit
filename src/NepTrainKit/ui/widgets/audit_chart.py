"""Deterministic QPainter charts for Training Set Audit plot payloads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, expm1, isfinite, log1p
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
_PHASE_COLORS = {
    "fcc": QColor("#159A9C"),
    "bcc": QColor("#3B6FB6"),
    "hcp": QColor("#E8871E"),
    "l12": QColor("#775DA6"),
    "c14": QColor("#2E8B57"),
    "c15": QColor("#B44C6C"),
    "unresolved": QColor("#89969A"),
}
_EVIDENCE_COLORS = {
    "strong": QColor("#159A9C"),
    "mixed": QColor("#E8871E"),
    "unresolved": QColor("#89969A"),
}


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
        if kind not in {
            "histogram",
            "categorical_bars",
            "composition_stems",
            "composition_phase_stacks",
        } or not isinstance(plot_id, str) or not plot_id:
            return None

        if kind == "composition_phase_stacks":
            return AuditChartWidget._normalize_composition_phase_stacks(plot)

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
        elif kind == "categorical_bars":
            labels = series.get("labels")
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) != len(counts):
                return None
            normalized["labels"] = tuple(str(label) for label in labels)
            bar_ids = series.get("bar_ids")
            if (
                isinstance(bar_ids, Sequence)
                and not isinstance(bar_ids, (str, bytes))
                and len(bar_ids) == len(counts)
            ):
                normalized["bar_ids"] = tuple(str(bar_id) for bar_id in bar_ids)
        else:
            x_values = AuditChartWidget._numeric_values(
                series.get("x_values"), nonnegative=False
            )
            if len(x_values) != len(counts) or any(
                value < 0.0 or value > 1.0 for value in x_values
            ):
                return None
            labels = series.get("labels")
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) != len(counts):
                labels = tuple(f"{value:.4g}" for value in x_values)
            target_points = AuditChartWidget._numeric_values(
                plot.get("target_points", ()), nonnegative=False
            )
            try:
                x_min = float(plot.get("x_min", 0.0))
                x_max = float(plot.get("x_max", 1.0))
            except (TypeError, ValueError):
                return None
            if (
                not isfinite(x_min)
                or not isfinite(x_max)
                or x_min >= x_max
                or any(value < x_min or value > x_max for value in x_values)
            ):
                return None
            normalized["x_values"] = x_values
            normalized["x_min"] = x_min
            normalized["x_max"] = x_max
            normalized["labels"] = tuple(str(label) for label in labels)
            normalized["log_scale"] = bool(plot.get("log_scale", False))
            normalized["target_points"] = tuple(
                value for value in target_points if 0.0 <= value <= 1.0
            )
        return normalized

    @staticmethod
    def _normalize_composition_phase_stacks(
        plot: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        plot_id = plot.get("id")
        x_values = AuditChartWidget._numeric_values(
            plot.get("x_values"), nonnegative=False
        )
        if not isinstance(plot_id, str) or not plot_id or not x_values:
            return None
        if any(value < 0.0 or value > 1.0 for value in x_values):
            return None
        labels = plot.get("labels")
        if (
            not isinstance(labels, Sequence)
            or isinstance(labels, (str, bytes))
            or len(labels) != len(x_values)
        ):
            return None
        series_items = plot.get("series")
        if (
            not isinstance(series_items, Sequence)
            or isinstance(series_items, (str, bytes))
            or not series_items
        ):
            return None
        normalized_series = []
        totals = [0.0] * len(x_values)
        for item in series_items:
            if not isinstance(item, Mapping):
                return None
            counts = AuditChartWidget._numeric_values(item.get("counts"))
            if len(counts) != len(x_values):
                return None
            structure_indices = AuditChartWidget._structure_groups(
                item.get("structure_indices"), len(counts)
            )
            if structure_indices is None:
                return None
            for index, count in enumerate(counts):
                totals[index] += count
            normalized_series.append(
                {
                    "id": str(item.get("id", "")),
                    "label": str(item.get("label", item.get("id", ""))),
                    "counts": counts,
                    "structure_indices": structure_indices,
                }
            )
        try:
            x_min = float(plot.get("x_min", 0.0))
            x_max = float(plot.get("x_max", 1.0))
        except (TypeError, ValueError):
            return None
        if not isfinite(x_min) or not isfinite(x_max) or x_min >= x_max:
            return None
        return {
            "kind": "composition_phase_stacks",
            "id": plot_id,
            "title": str(plot.get("title", "")),
            "x_label": str(plot.get("x_label", "")),
            "y_label": str(plot.get("y_label", "")),
            "x_values": x_values,
            "x_min": x_min,
            "x_max": x_max,
            "labels": tuple(str(label) for label in labels),
            "counts": tuple(totals),
            "series": tuple(normalized_series),
            "target_points": tuple(
                value
                for value in AuditChartWidget._numeric_values(
                    plot.get("target_points", ()), nonnegative=False
                )
                if 0.0 <= value <= 1.0
            ),
            "highlighted_bins": frozenset(),
            "structure_indices": tuple(None for _ in x_values),
        }

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
        elif self._plot["kind"] == "categorical_bars":
            self._draw_categorical_bars(painter)
        elif self._plot["kind"] == "composition_phase_stacks":
            self._draw_composition_phase_stacks(painter)
        else:
            self._draw_composition_stems(painter)

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
            exact_width = chart.width() * count / max_count
            visible_width = max(2.0, exact_width) if count > 0 else 0.0
            rect = QRectF(chart.left(), y, visible_width, bar_height)
            if index in self._plot["highlighted_bins"]:
                color = _ORANGE
            else:
                bar_ids = self._plot.get("bar_ids", ())
                bar_id = bar_ids[index] if index < len(bar_ids) else ""
                color = _PHASE_COLORS.get(
                    bar_id,
                    _EVIDENCE_COLORS.get(bar_id, _TEAL),
                )
            painter.fillRect(rect, color)
            elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(left_margin - 12))
            painter.drawText(
                QRectF(6, y, left_margin - 12, bar_height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                elided,
            )
            hit_rect = QRectF(
                rect.left(),
                rect.top(),
                max(20.0, rect.width()),
                rect.height(),
            )
            self._bar_rects.append(
                (hit_rect, self._plot["structure_indices"][index])
            )
            self._bar_tooltips.append(f"{label}: {count:g}")
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(rect.right() + 4, y, 70, bar_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{count:g}",
            )

        self._draw_axis_labels(painter, chart)

    def _draw_composition_stems(self, painter: QPainter) -> None:
        chart = QRectF(58, 54, max(1, self.width() - 76), max(1, self.height() - 98))
        counts = self._plot["counts"]
        x_min = self._plot["x_min"]
        x_max = self._plot["x_max"]
        x_span = x_max - x_min

        def chart_x(value: float) -> float:
            return chart.left() + chart.width() * (value - x_min) / x_span

        log_scale = self._plot["log_scale"]
        display_counts = tuple(log1p(value) for value in counts) if log_scale else counts
        max_display = max(display_counts) if max(display_counts) > 0 else 1.0

        painter.setPen(QPen(_GRID, 1))
        metrics = QFontMetrics(painter.font())
        for index in range(5):
            fraction = index / 4
            y = chart.bottom() - chart.height() * fraction
            painter.drawLine(chart.left(), y, chart.right(), y)
            display_value = max_display * fraction
            value = expm1(display_value) if log_scale else display_value
            label = f"{int(round(value)):,}"
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(2, y - 8, chart.left() - 8, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(QPen(_GRID, 1))

        for target in self._plot["target_points"]:
            x = chart_x(target)
            painter.setPen(QPen(_ORANGE, 1, Qt.PenStyle.DashLine))
            painter.drawLine(x, chart.top(), x, chart.bottom())

        painter.setPen(QPen(_TEXT, 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

        point_width = max(7.0, min(16.0, chart.width() / max(1, len(counts)) * 0.35))
        labeled_indices = set(
            sorted(range(len(counts)), key=lambda index: counts[index], reverse=True)[:3]
        )
        labeled_indices.update(
            index
            for index, value in enumerate(self._plot["x_values"])
            if value <= 1.0e-8 or value >= 1.0 - 1.0e-8
        )
        labeled_indices.update(
            index
            for index, value in enumerate(self._plot["x_values"])
            if any(abs(value - target) <= 1.0e-8 for target in self._plot["target_points"])
        )
        occupied_labels: list[QRectF] = []
        for index, (x_value, count, display_count) in enumerate(
            zip(self._plot["x_values"], counts, display_counts)
        ):
            x = chart_x(x_value)
            height = chart.height() * display_count / max_display
            top = chart.bottom() - height
            color = _ORANGE if index in self._plot["highlighted_bins"] else _TEAL
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, chart.bottom(), x, top)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(x - 3.5, top - 3.5, 7, 7))
            hit_rect = QRectF(
                x - point_width / 2,
                max(chart.top(), top - 8),
                point_width,
                max(16.0, chart.bottom() - top + 8),
            )
            self._bar_rects.append((hit_rect, self._plot["structure_indices"][index]))
            self._bar_tooltips.append(
                f"{self._plot['labels'][index]}: {count:g}"
            )
            duplicates_top_left_tick = (
                x_value <= 0.02
                and abs(display_count - max_display) <= 1.0e-8
            )
            if index in labeled_indices and not duplicates_top_left_tick:
                label = f"{count:,.0f}"
                width = metrics.horizontalAdvance(label) + 8
                left = min(
                    max(chart.left(), x - width / 2),
                    chart.right() - width,
                )
                candidates = (
                    top - 22,
                    top - 40,
                    top + 5,
                    top - 58,
                )
                label_rect = next(
                    (
                        QRectF(left, max(chart.top(), candidate), width, 16)
                        for candidate in candidates
                        if not any(
                            existing.intersects(
                                QRectF(left, max(chart.top(), candidate), width, 16)
                            )
                            for existing in occupied_labels
                        )
                    ),
                    None,
                )
                if label_rect is not None:
                    occupied_labels.append(label_rect.adjusted(-2, -1, 2, 1))
                    painter.fillRect(label_rect.adjusted(-2, -1, 2, 1), _BACKGROUND)
                    painter.setPen(_TEXT)
                    painter.drawText(
                        label_rect,
                        Qt.AlignmentFlag.AlignCenter,
                        label,
                    )

        for index in range(5):
            fraction = index / 4
            x = chart_x(fraction)
            label = f"{fraction:.0%}"
            width = metrics.horizontalAdvance(label) + 4
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(x - width / 2, chart.bottom() + 4, width, 16),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
        self._draw_axis_labels(painter, chart, y_label_above=True)

    def _draw_composition_phase_stacks(self, painter: QPainter) -> None:
        legend_y = 32.0
        metrics = QFontMetrics(painter.font())
        legend_x = 10.0
        for series in self._plot["series"]:
            label = series["label"]
            width = metrics.horizontalAdvance(label) + 34
            if legend_x + width > self.width() - 8:
                legend_x = 10.0
                legend_y += 20.0
            color = _PHASE_COLORS.get(series["id"], _TEAL)
            painter.fillRect(QRectF(legend_x, legend_y + 3, 11, 11), color)
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(legend_x + 16, legend_y, width - 16, 17),
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            legend_x += width

        chart_top = legend_y + 28
        chart = QRectF(
            58,
            chart_top,
            max(1, self.width() - 76),
            max(1, self.height() - chart_top - 44),
        )
        counts = self._plot["counts"]
        max_count = max(counts) if max(counts) > 0 else 1.0
        x_min = self._plot["x_min"]
        x_span = self._plot["x_max"] - x_min

        def chart_x(value: float) -> float:
            return chart.left() + chart.width() * (value - x_min) / x_span

        painter.setPen(QPen(_GRID, 1))
        for index in range(5):
            fraction = index / 4
            y = chart.bottom() - chart.height() * fraction
            painter.drawLine(chart.left(), y, chart.right(), y)
            label = f"{int(round(max_count * fraction)):,}"
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(2, y - 8, chart.left() - 8, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(QPen(_GRID, 1))
        for target in self._plot["target_points"]:
            x = chart_x(target)
            painter.setPen(QPen(_ORANGE, 1, Qt.PenStyle.DashLine))
            painter.drawLine(x, chart.top(), x, chart.bottom())
        painter.setPen(QPen(_TEXT, 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

        bar_width = max(5.0, min(24.0, chart.width() / len(counts) * 0.58))
        labeled_indices = set(
            sorted(range(len(counts)), key=lambda index: counts[index], reverse=True)[:3]
        )
        labeled_indices.update(
            index
            for index, value in enumerate(self._plot["x_values"])
            if value <= 1.0e-8 or value >= 1.0 - 1.0e-8
        )
        for group_index, (x_value, total) in enumerate(
            zip(self._plot["x_values"], counts)
        ):
            x = chart_x(x_value)
            bottom = chart.bottom()
            for series in self._plot["series"]:
                count = series["counts"][group_index]
                if count <= 0:
                    continue
                height = chart.height() * count / max_count
                rect = QRectF(x - bar_width / 2, bottom - height, bar_width, height)
                painter.fillRect(rect, _PHASE_COLORS.get(series["id"], _TEAL))
                painter.setPen(QPen(_BACKGROUND, 0.7))
                painter.drawRect(rect)
                self._bar_rects.append(
                    (rect, series["structure_indices"][group_index])
                )
                share = 0.0 if total <= 0 else count / total
                self._bar_tooltips.append(
                    f"{self._plot['labels'][group_index]} · "
                    f"{series['label']}: {count:,.0f} ({share:.1%})"
                )
                bottom -= height
            if group_index in labeled_indices and total > 0:
                label = f"{total:,.0f}"
                width = metrics.horizontalAdvance(label) + 6
                painter.setPen(_TEXT)
                painter.drawText(
                    QRectF(x - width / 2, max(chart.top(), bottom - 18), width, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

        for index in range(5):
            fraction = index / 4
            x = chart_x(fraction)
            label = f"{fraction:.0%}"
            width = metrics.horizontalAdvance(label) + 4
            painter.setPen(_TEXT)
            painter.drawText(
                QRectF(x - width / 2, chart.bottom() + 4, width, 16),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
        self._draw_axis_labels(painter, chart, y_label_above=True)

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

    def _draw_axis_labels(
        self,
        painter: QPainter,
        chart: QRectF,
        *,
        y_label_above: bool = False,
    ) -> None:
        axis_font = QFont(painter.font())
        axis_font.setPointSize(max(7, axis_font.pointSize() - 1))
        painter.setFont(axis_font)
        painter.setPen(_TEXT)
        painter.drawText(
            QRectF(chart.left(), self.height() - 21, chart.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self._plot["x_label"],
        )
        if y_label_above:
            y_label_rect = QRectF(chart.left(), chart.top() - 18, chart.width(), 14)
        else:
            y_label_rect = QRectF(6, 24, max(1, chart.left() - 12), 14)
        painter.drawText(y_label_rect, Qt.AlignmentFlag.AlignLeft, self._plot["y_label"])

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
