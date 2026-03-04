#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pyqtgraph backend for distribution inspector plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from NepTrainKit.ui.canvas.base.distribution import DistributionPlotBase


class PyqtgraphDistributionPlot(DistributionPlotBase):
    """Pyqtgraph implementation for histogram + optional overlay curve."""

    ALL_SERIES_KEY = "__all__"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__()
        # Keep import local so pyqtgraph is loaded only when this backend is selected.
        import NepTrainKit.ui.canvas.pyqtgraph  # noqa: F401
        import pyqtgraph as pg

        self._pg = pg
        self._plot = self._pg.PlotWidget(parent=parent)
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.getPlotItem().setMenuEnabled(False)
        self._plot.setMinimumHeight(220)
        self._metric: dict[str, Any] | None = None
        self._series: dict[str, Any] | None = None
        self._bars = None
        self._curve = None
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def widget(self) -> QWidget:
        return self._plot

    def clear(self) -> None:
        self._metric = None
        self._series = None
        self._bars = None
        self._curve = None
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
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", "Count")
        self._plot.getPlotItem().setXRange(lo, hi, padding=0.02)
        self._plot.getPlotItem().enableAutoRange(axis="y", enable=True)
        if field:
            self._plot.getPlotItem().setLabels(top=field)

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
        plot_item.addLegend(offset=(8, 8))
        x_hist = lo + (np.arange(bins, dtype=np.float64) + 0.5) * ((hi - lo) / float(max(1, bins)))
        total_n = 0

        for i, item in enumerate(series_items):
            name = str(item.get("name", item.get("series_key", f"s{i + 1}")) or f"s{i + 1}")
            color = self._pg.intColor(i, hues=max(6, len(series_items)), alpha=220)
            hist = np.asarray(item.get("hist", []) or [], dtype=np.float64).reshape(-1)
            if hist.size != bins:
                hist = hist[:bins] if hist.size > bins else np.pad(hist, (0, bins - hist.size))
            total_n += int(item.get("total", 0) or 0)
            self._plot.plot(x_hist, hist, pen=self._pg.mkPen(color=color, width=1), name=name)

            curve_x = np.asarray(item.get("curve_x", []) or [], dtype=np.float64).reshape(-1)
            curve_y = np.asarray(item.get("curve_y", []) or [], dtype=np.float64).reshape(-1)
            if curve_x.size >= 2 and curve_x.size == curve_y.size:
                self._plot.plot(curve_x, curve_y, pen=self._pg.mkPen(color=color, width=2))

        field = str(metric.get("field_label", metric.get("field_key", "")) or "")
        component = str(metric.get("component", "") or "")
        self._plot.setTitle(f"All groups overlay | groups={len(series_items)}, N={total_n}")
        unit = str(metric.get("unit", "unknown") or "unknown")
        xlabel = component if component else "value"
        if unit and unit != "unknown":
            xlabel = f"{xlabel} ({unit})"
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", "Count")
        plot_item.setXRange(lo, hi, padding=0.02)
        plot_item.enableAutoRange(axis="y", enable=True)
        if field:
            plot_item.setLabels(top=field)

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
