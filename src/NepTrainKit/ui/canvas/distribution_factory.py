#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Factory for distribution plot backends."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from NepTrainKit.ui.canvas.base.distribution import DistributionPlotBase


def create_distribution_plot_adapter(canvas_type: str, parent: QWidget | None = None) -> tuple[DistributionPlotBase, bool]:
    """Create plot adapter for selected backend.

    Returns
    -------
    tuple[DistributionPlotBase, bool]
        Adapter instance and fallback flag (True means requested backend failed
        and adapter fell back to pyqtgraph).
    """
    text = str(canvas_type or "").strip().lower()
    if text in {"vispy", "canvasmode.vispy"}:
        try:
            from NepTrainKit.ui.canvas.vispy.distribution import VispyDistributionPlot
            return VispyDistributionPlot(parent), False
        except Exception:
            from NepTrainKit.ui.canvas.pyqtgraph.distribution import PyqtgraphDistributionPlot
            return PyqtgraphDistributionPlot(parent), True
    from NepTrainKit.ui.canvas.pyqtgraph.distribution import PyqtgraphDistributionPlot
    return PyqtgraphDistributionPlot(parent), False
