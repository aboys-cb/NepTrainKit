#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base interface for distribution plot backends."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PySide6.QtWidgets import QWidget


class DistributionPlotBase:
    """Minimal plotting interface for distribution inspector backends."""

    def __init__(self) -> None:
        self._bin_click_callback: Callable[[int], None] | None = None

    def widget(self) -> QWidget:
        """Return the backend widget inserted into the dialog."""
        raise NotImplementedError

    def set_payload(self, metric: dict[str, Any] | None, series: dict[str, Any] | None) -> None:
        """Render the selected metric+series payload."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all rendered items."""
        raise NotImplementedError

    def set_bin_click_callback(self, callback: Callable[[int], None] | None) -> None:
        """Register callback for histogram-bin click events."""
        self._bin_click_callback = callback

    def _emit_bin_click(self, metric: dict[str, Any], x: float) -> None:
        if self._bin_click_callback is None:
            return
        lo = float(metric.get("hist_left", 0.0) or 0.0)
        hi = float(metric.get("hist_right", 0.0) or 0.0)
        bins = int(metric.get("bins", 0) or 0)
        if bins <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        half_bin = 0.5 * ((hi - lo) / bins)
        if x < lo - half_bin or x > hi + half_bin:
            return
        edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
        bin_index = np.searchsorted(edges, x, side="right") - 1
        self._bin_click_callback(int(np.clip(bin_index, 0, bins - 1)))
