#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pyqtgraph backend for distribution inspector plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QTransform
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from qfluentwidgets import CaptionLabel

from NepTrainKit.ui.canvas.base.distribution import DistributionPlotBase


class _RotatedLabel(QLabel):
    """A label rotated 90° counter-clockwise."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._angle = -90.0  # counter-clockwise

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        painter.rotate(self._angle)
        fm = painter.fontMetrics()
        text_rect = self.rect()
        # After rotating -90°, the new origin is at bottom-left.
        # Map the text rect accordingly.
        painter.drawText(
            -text_rect.height(), 0,
            text_rect.height(), text_rect.width(),
            int(Qt.AlignmentFlag.AlignCenter),
            self.text(),
        )
        painter.end()

    def sizeHint(self):
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(self.text()) + 4
        h = fm.height() + 2
        # Swap because rotated
        from PySide6.QtCore import QSize
        return QSize(h, w)

    def minimumSizeHint(self):
        return self.sizeHint()


class PyqtgraphDistributionPlot(DistributionPlotBase):
    """Pyqtgraph implementation for histogram + optional overlay curve."""

    ALL_SERIES_KEY = "__all__"

    _SERIES_COLORS = [
        (37, 99, 235),     # blue - reference
        (22, 163, 74),     # green - prediction
        (220, 38, 38),     # red - error
        (147, 51, 234),    # purple
        (234, 179, 8),     # amber
        (20, 184, 166),    # teal
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__()
        # Keep import local so pyqtgraph is loaded only when this backend is selected.
        import NepTrainKit.ui.canvas.pyqtgraph  # noqa: F401
        import pyqtgraph as pg

        self._pg = pg
        # Container: legend row + plot row (with external Y label)
        self._container = QWidget(parent)
        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Legend row (hidden when no series)
        self._legend_row = QWidget(self._container)
        self._legend_row.setStyleSheet("background: transparent;")
        self._legend_layout = QHBoxLayout(self._legend_row)
        self._legend_layout.setContentsMargins(8, 2, 8, 2)
        self._legend_layout.setSpacing(12)
        self._legend_layout.addStretch()
        self._legend_row.setVisible(False)
        container_layout.addWidget(self._legend_row)

        # Plot row: [Y label] [PlotWidget] [X label rotated]
        plot_row = QWidget(self._container)
        plot_row_layout = QHBoxLayout(plot_row)
        plot_row_layout.setContentsMargins(0, 0, 0, 0)
        plot_row_layout.setSpacing(2)

        self._ylabel_label = _RotatedLabel("", plot_row)
        self._ylabel_label.setStyleSheet("color: #333; font-size: 13px;")
        self._ylabel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_row_layout.addWidget(self._ylabel_label)

        self._plot = self._pg.PlotWidget(parent=plot_row)
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        plot_item = self._plot.getPlotItem()
        plot_item.setMenuEnabled(False)
        plot_item.showAxis("left", True)
        plot_item.showAxis("bottom", True)
        # Hide built-in axis labels; we use external labels
        plot_item.setLabel("bottom", "")
        plot_item.setLabel("left", "")
        plot_item.getAxis("left").setWidth(56)
        for axis_name in ("left", "bottom", "top"):
            axis = plot_item.getAxis(axis_name)
            axis.setStyle(autoExpandTextSpace=True)
        self._plot.setMinimumHeight(220)
        plot_row_layout.addWidget(self._plot, 1)

        # X label below the plot
        self._xlabel_label = CaptionLabel("", plot_row)
        self._xlabel_label.setStyleSheet("color: #333; font-size: 13px;")
        self._xlabel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._xlabel_label.setFixedHeight(20)
        x_below = QVBoxLayout()
        x_below.setContentsMargins(0, 0, 0, 0)
        x_below.setSpacing(0)
        x_below.addWidget(self._plot, 1)
        x_below.addWidget(self._xlabel_label)
        plot_row_layout.addLayout(x_below, 1)

        container_layout.addWidget(plot_row, 1)

        self._metric: dict[str, Any] | None = None
        self._series: dict[str, Any] | None = None
        self._bars = None
        self._curve = None
        self._legend_labels: list[CaptionLabel] = []
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def widget(self) -> QWidget:
        return self._container

    def clear(self) -> None:
        self._metric = None
        self._series = None
        self._bars = None
        self._curve = None
        self._clear_legend()
        plot_item = self._plot.getPlotItem()
        if getattr(plot_item, "legend", None) is not None:
            try:
                plot_item.legend.scene().removeItem(plot_item.legend)
            except Exception:
                pass
            plot_item.legend = None
        self._plot.clear()
        self._plot.setTitle("")

    def set_payload(self, metric: dict[str, Any] | None, series: dict[str, Any] | None) -> None:
        self.clear()
        self._metric = metric
        self._series = series
        if not metric or not series:
            return

        series_key = str(series.get("series_key", "") or "")
        if series_key == self.ALL_SERIES_KEY:
            self._render_all_series(metric)
            return

        hist = np.asarray(series.get("hist", []) or [], dtype=np.float64).reshape(-1)
        bins = int(metric.get("bins", hist.size) or hist.size or 0)
        lo = float(metric.get("hist_left", 0.0) or 0.0)
        hi = float(metric.get("hist_right", 0.0) or 0.0)
        if bins <= 0 or hist.size == 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        if hist.size != bins:
            hist = hist[:bins] if hist.size > bins else np.pad(hist, (0, bins - hist.size))

        width = float(hi - lo) / float(max(1, bins))
        centers = lo + (np.arange(bins, dtype=np.float64) + 0.5) * width
        self._bars = self._pg.BarGraphItem(
            x=centers,
            height=hist,
            width=max(1e-12, width * 0.9),
            brush=self._pg.mkBrush(37, 99, 235, 120),
            pen=self._pg.mkPen(color=(37, 99, 235), width=1),
        )
        self._plot.addItem(self._bars)

        curve_x = np.asarray(series.get("curve_x", []) or [], dtype=np.float64).reshape(-1)
        curve_y = np.asarray(series.get("curve_y", []) or [], dtype=np.float64).reshape(-1)
        if curve_x.size >= 2 and curve_x.size == curve_y.size:
            self._curve = self._plot.plot(curve_x, curve_y, pen=self._pg.mkPen(color=(220, 20, 60), width=2))

        field = str(metric.get("field_label", metric.get("field_key", "")) or "")
        component = str(metric.get("component", "") or "")
        series_name = str(series.get("name", series.get("series_key", "")) or "")
        total = int(series.get("total", 0) or 0)
        mean = float(series.get("mean", 0.0) or 0.0)
        std = float(series.get("std", 0.0) or 0.0)
        title = f"{series_name} | N={total}, mean={mean:.4g}, std={std:.4g}"
        self._plot.setTitle(title)

        unit = str(metric.get("unit", "unknown") or "unknown")
        xlabel = component if component else "value"
        if unit and unit != "unknown":
            xlabel = f"{xlabel} ({unit})"
        self._xlabel_label.setText(xlabel)
        self._ylabel_label.setText("Count")
        self._plot.getPlotItem().setXRange(lo, hi, padding=0.02)
        self._plot.getPlotItem().enableAutoRange(axis="y", enable=True)

    def _render_all_series(self, metric: dict[str, Any]) -> None:
        """Render all series as overlaid lines for cross-group comparison."""
        series_items = list(metric.get("series", []) or [])
        if not series_items:
            return
        bins = int(metric.get("bins", 0) or 0)
        lo = float(metric.get("hist_left", 0.0) or 0.0)
        hi = float(metric.get("hist_right", 0.0) or 0.0)
        if bins <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return

        plot_item = self._plot.getPlotItem()
        x_hist = lo + (np.arange(bins, dtype=np.float64) + 0.5) * ((hi - lo) / float(max(1, bins)))
        total_n = 0

        for i, item in enumerate(series_items):
            name = str(item.get("name", item.get("series_key", f"s{i + 1}")) or f"s{i + 1}")
            rgb = self._SERIES_COLORS[i % len(self._SERIES_COLORS)]
            hist = np.asarray(item.get("hist", []) or [], dtype=np.float64).reshape(-1)
            if hist.size != bins:
                hist = hist[:bins] if hist.size > bins else np.pad(hist, (0, bins - hist.size))
            total_n += int(item.get("total", 0) or 0)

            # Draw histogram as semi-transparent bars
            width = float(hi - lo) / float(max(1, bins))
            centers = lo + (np.arange(bins, dtype=np.float64) + 0.5) * width
            bar = self._pg.BarGraphItem(
                x=centers,
                height=hist,
                width=max(1e-12, width * 0.85),
                brush=self._pg.mkBrush(rgb[0], rgb[1], rgb[2], 50),
                pen=self._pg.mkPen(color=rgb, width=0.5),
            )
            self._plot.addItem(bar)

            # Draw curve or step line (no name -> no pyqtgraph legend)
            curve_x = np.asarray(item.get("curve_x", []) or [], dtype=np.float64).reshape(-1)
            curve_y = np.asarray(item.get("curve_y", []) or [], dtype=np.float64).reshape(-1)
            if curve_x.size >= 2 and curve_x.size == curve_y.size:
                self._plot.plot(curve_x, curve_y, pen=self._pg.mkPen(color=rgb, width=2))
            else:
                self._plot.plot(x_hist, hist, pen=self._pg.mkPen(color=rgb, width=1.5))

        field = str(metric.get("field_label", metric.get("field_key", "")) or "")
        component = str(metric.get("component", "") or "")
        n_groups = len(series_items)
        title_parts = []
        if field:
            title_parts.append(field)
        title_parts.append(f"{n_groups} group{'s' if n_groups > 1 else ''}")
        title_parts.append(f"N={total_n}")
        self._plot.setTitle("  |  ".join(title_parts), size="10pt")

        unit = str(metric.get("unit", "unknown") or "unknown")
        xlabel = component if component else "value"
        if unit and unit != "unknown":
            xlabel = f"{xlabel} ({unit})"
        self._xlabel_label.setText(xlabel)
        self._ylabel_label.setText("Count")
        plot_item.setXRange(lo, hi, padding=0.02)
        plot_item.enableAutoRange(axis="y", enable=True)

        # Build colored legend labels in the row above the plot
        self._build_colored_legend(series_items)

    def _clear_legend(self) -> None:
        for lbl in self._legend_labels:
            lbl.deleteLater()
        self._legend_labels.clear()
        self._legend_row.setVisible(False)

    def _build_colored_legend(self, series_items: list[dict[str, Any]]) -> None:
        """Create colored labels in a row above the plot (outside viewBox)."""
        self._clear_legend()
        for i, item in enumerate(series_items):
            name = str(item.get("name", item.get("series_key", f"s{i + 1}")) or f"s{i + 1}")
            rgb = self._SERIES_COLORS[i % len(self._SERIES_COLORS)]
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            lbl = CaptionLabel(f"■ {name}", self._legend_row)
            lbl.setStyleSheet(f"color: {hex_color}; font-size: 11px; font-weight: bold; background: transparent;")
            self._legend_labels.append(lbl)
            # Insert before the stretch (which is at index 0)
            self._legend_layout.insertWidget(self._legend_layout.count() - 1, lbl)
        if self._legend_labels:
            self._legend_row.setVisible(True)

    def _on_mouse_clicked(self, event: Any) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._metric is None or self._series is None:
            return
        scene_pos = event.scenePos()
        vb = self._plot.getPlotItem().vb
        if vb is None:
            return
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        point = vb.mapSceneToView(scene_pos)
        x = float(point.x())
        lo = float(self._metric.get("hist_left", 0.0) or 0.0)
        hi = float(self._metric.get("hist_right", 0.0) or 0.0)
        bins = int(self._metric.get("bins", 0) or 0)
        if bins <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        bin_w = float(hi - lo) / float(max(1, bins))
        if x < (lo - 0.5 * bin_w) or x > (hi + 0.5 * bin_w):
            return
        x_clamped = float(np.clip(x, lo, hi))
        edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
        bin_idx = int(np.searchsorted(edges, x_clamped, side="right") - 1)
        bin_idx = int(np.clip(bin_idx, 0, bins - 1))
        if self._bin_click_callback is not None:
            self._bin_click_callback(bin_idx)
