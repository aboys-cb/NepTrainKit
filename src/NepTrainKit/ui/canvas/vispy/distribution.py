#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vispy backend for distribution inspector plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QWidget

from NepTrainKit.ui.canvas.base.distribution import DistributionPlotBase


class VispyDistributionPlot(DistributionPlotBase):
    """Vispy implementation for histogram + optional overlay curve."""

    ALL_SERIES_KEY = "__all__"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__()
        # Ensure project-level vispy backend selection is applied.
        import NepTrainKit.ui.canvas.vispy  # noqa: F401
        from vispy import scene
        from vispy.color import get_colormap

        self._scene = scene
        self._get_colormap = get_colormap
        self._canvas = scene.SceneCanvas(keys=None, show=False, bgcolor="white")
        self._canvas.native.setParent(parent)
        self._canvas.native.setMinimumHeight(220)
        self._canvas.native.setStyleSheet("background: white;")

        grid = self._canvas.central_widget.add_grid(margin=6)
        self._title = scene.Label("", color="black", font_size=10)
        self._title.height_max = 28
        grid.add_widget(self._title, row=0, col=0, col_span=3)

        self._yaxis = scene.AxisWidget(
            orientation="left",
            axis_width=1,
            tick_label_margin=5,
            axis_color="black",
            text_color="black",
        )
        self._yaxis.width_max = 48
        grid.add_widget(self._yaxis, row=1, col=0)

        self._xaxis = scene.AxisWidget(
            orientation="bottom",
            axis_width=1,
            tick_label_margin=8,
            axis_color="black",
            text_color="black",
        )
        self._xaxis.height_max = 28
        grid.add_widget(self._xaxis, row=2, col=1)

        right_pad = grid.add_widget(row=1, col=2)
        right_pad.width_max = 4

        self._view = grid.add_view(row=1, col=1)
        self._view.camera = scene.cameras.PanZoomCamera()
        self._view.camera.interactive = False
        self._xaxis.link_view(self._view)
        self._yaxis.link_view(self._view)

        self._metric: dict[str, Any] | None = None
        self._series: dict[str, Any] | None = None
        self._visuals: list[Any] = []

        self._canvas.events.mouse_press.connect(self._on_mouse_press)

    def widget(self) -> QWidget:
        return self._canvas.native

    def clear(self) -> None:
        self._metric = None
        self._series = None
        self._title.text = ""
        for visual in self._visuals:
            try:
                visual.parent = None
            except Exception:
                pass
        self._visuals = []
        try:
            self._view.camera.set_range(x=(0, 1), y=(0, 1), z=(0, 0))
        except Exception:
            pass

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
        hist_line = self._make_hist_segments(centers, hist, color=(0.145, 0.388, 0.921, 0.85), width=2.0)
        self._visuals.append(hist_line)

        curve_x = np.asarray(series.get("curve_x", []) or [], dtype=np.float64).reshape(-1)
        curve_y = np.asarray(series.get("curve_y", []) or [], dtype=np.float64).reshape(-1)
        if curve_x.size >= 2 and curve_x.size == curve_y.size:
            curve_pos = np.column_stack([curve_x, curve_y, np.zeros_like(curve_x)])
            curve = self._scene.visuals.Line(
                pos=curve_pos, color=(0.86, 0.08, 0.24, 0.95), width=2.2, parent=self._view.scene
            )
            self._visuals.append(curve)

        field = str(metric.get("field_label", metric.get("field_key", "")) or "")
        series_name = str(series.get("name", series.get("series_key", "")) or "")
        total = int(series.get("total", 0) or 0)
        mean = float(series.get("mean", 0.0) or 0.0)
        std = float(series.get("std", 0.0) or 0.0)
        self._title.text = f"{field} | {series_name} | N={total}, mean={mean:.4g}, std={std:.4g}"

        unit = str(metric.get("unit", "unknown") or "unknown")
        component = str(metric.get("component", "") or "")
        xlabel = component if component else "value"
        if unit and unit != "unknown":
            xlabel = f"{xlabel} ({unit})"
        self._xaxis.axis.axis_label = xlabel
        self._yaxis.axis.axis_label = "Count"

        ymax = float(np.max(hist)) if hist.size else 1.0
        if curve_y.size:
            ymax = max(ymax, float(np.nanmax(curve_y)))
        ymax = max(1.0, ymax * 1.08)
        self._view.camera.set_range(x=(lo, hi), y=(0.0, ymax), z=(0, 0))

    def _render_all_series(self, metric: dict[str, Any]) -> None:
        series_items = list(metric.get("series", []) or [])
        if not series_items:
            return
        bins = int(metric.get("bins", 0) or 0)
        lo = float(metric.get("hist_left", 0.0) or 0.0)
        hi = float(metric.get("hist_right", 0.0) or 0.0)
        if bins <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return

        x_hist = lo + (np.arange(bins, dtype=np.float64) + 0.5) * ((hi - lo) / float(max(1, bins)))
        total_n = 0
        ymax = 1.0
        cmap = self._get_colormap("viridis")
        denom = float(max(1, len(series_items) - 1))

        for i, item in enumerate(series_items):
            t = float(np.clip(i / denom, 0.0, 1.0))
            color = tuple(float(v) for v in np.asarray(cmap.map(np.asarray([t], dtype=np.float32))[0]).reshape(-1))
            hist = np.asarray(item.get("hist", []) or [], dtype=np.float64).reshape(-1)
            if hist.size != bins:
                hist = hist[:bins] if hist.size > bins else np.pad(hist, (0, bins - hist.size))
            total_n += int(item.get("total", 0) or 0)
            ymax = max(ymax, float(np.max(hist)) if hist.size else 0.0)

            hist_pos = np.column_stack([x_hist, hist, np.zeros_like(x_hist)])
            hist_line = self._scene.visuals.Line(pos=hist_pos, color=color, width=1.4, parent=self._view.scene)
            self._visuals.append(hist_line)

            curve_x = np.asarray(item.get("curve_x", []) or [], dtype=np.float64).reshape(-1)
            curve_y = np.asarray(item.get("curve_y", []) or [], dtype=np.float64).reshape(-1)
            if curve_x.size >= 2 and curve_x.size == curve_y.size:
                ymax = max(ymax, float(np.nanmax(curve_y)))
                curve_pos = np.column_stack([curve_x, curve_y, np.zeros_like(curve_x)])
                curve = self._scene.visuals.Line(pos=curve_pos, color=color, width=2.0, parent=self._view.scene)
                self._visuals.append(curve)

        field = str(metric.get("field_label", metric.get("field_key", "")) or "")
        component = str(metric.get("component", "") or "")
        self._title.text = f"{field} | All groups overlay | groups={len(series_items)}, N={total_n}"
        unit = str(metric.get("unit", "unknown") or "unknown")
        xlabel = component if component else "value"
        if unit and unit != "unknown":
            xlabel = f"{xlabel} ({unit})"
        self._xaxis.axis.axis_label = xlabel
        self._yaxis.axis.axis_label = "Count"
        self._view.camera.set_range(x=(lo, hi), y=(0.0, max(1.0, ymax * 1.08)), z=(0, 0))

    def _make_hist_segments(self, centers: np.ndarray, heights: np.ndarray, color: tuple[float, ...], width: float) -> Any:
        n = int(centers.size)
        seg = np.zeros((max(0, n) * 2, 3), dtype=np.float32)
        if n:
            seg[0::2, 0] = centers
            seg[1::2, 0] = centers
            seg[1::2, 1] = heights
        return self._scene.visuals.Line(
            pos=seg, connect="segments", color=color, width=float(width), parent=self._view.scene
        )

    def _on_mouse_press(self, event: Any) -> None:
        if event is None:
            return
        button = int(getattr(event, "button", 0) or 0)
        if button != 1:
            return
        if self._metric is None or self._series is None:
            return

        lo = float(self._metric.get("hist_left", 0.0) or 0.0)
        hi = float(self._metric.get("hist_right", 0.0) or 0.0)
        bins = int(self._metric.get("bins", 0) or 0)
        if bins <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return

        try:
            tr = self._canvas.scene.node_transform(self._view.scene)
            x, _y, _z, _w = tr.map(event.pos)
            x = float(x)
        except Exception:
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
