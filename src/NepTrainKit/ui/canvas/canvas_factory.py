#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Factory helpers for canvas and structure plotting backends."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from NepTrainKit.core.types import CanvasMode


def _normalize_canvas_type(canvas_type: object) -> str:
    """Normalize backend config values into canonical backend strings."""
    text = str(canvas_type or "").strip().lower()
    if text in {CanvasMode.VISPY.value, "canvasmode.vispy", CanvasMode.VISPY.name.lower()}:
        return CanvasMode.VISPY.value
    if text in {CanvasMode.PYQTGRAPH.value, "canvasmode.pyqtgraph", CanvasMode.PYQTGRAPH.name.lower()}:
        return CanvasMode.PYQTGRAPH.value
    return CanvasMode.PYQTGRAPH.value


def _create_pyqtgraph_result_canvas(parent: QWidget | None):
    from NepTrainKit.ui.canvas.pyqtgraph import PyqtgraphCanvas

    return PyqtgraphCanvas(parent)


def _create_vispy_result_canvas(parent: QWidget | None):
    from NepTrainKit.ui.canvas.vispy import VispyCanvas

    return VispyCanvas(parent=parent, bgcolor="white")


def _create_pyqtgraph_structure_plot(parent: QWidget | None):
    from NepTrainKit.ui.canvas.pyqtgraph import StructurePlotWidget

    return StructurePlotWidget(parent)


def _create_vispy_structure_plot(parent: QWidget | None):
    from NepTrainKit.ui.canvas.vispy import StructurePlotWidget

    return StructurePlotWidget(parent=parent)


def create_result_canvas(canvas_type: object, parent: QWidget | None = None) -> tuple[object, bool]:
    """Create result scatter canvas for selected backend.

    Returns
    -------
    tuple[object, bool]
        Canvas object and fallback flag (``True`` if vispy failed and
        pyqtgraph was used instead).
    """
    mode = _normalize_canvas_type(canvas_type)
    if mode == CanvasMode.VISPY.value:
        try:
            return _create_vispy_result_canvas(parent), False
        except Exception:
            return _create_pyqtgraph_result_canvas(parent), True
    return _create_pyqtgraph_result_canvas(parent), False


def create_structure_plot(canvas_type: object, parent: QWidget | None = None) -> tuple[object, bool]:
    """Create structure viewer widget for selected backend.

    Returns
    -------
    tuple[object, bool]
        Structure plot object and fallback flag (``True`` if vispy failed and
        pyqtgraph was used instead).
    """
    mode = _normalize_canvas_type(canvas_type)
    if mode == CanvasMode.VISPY.value:
        try:
            return _create_vispy_structure_plot(parent), False
        except Exception:
            return _create_pyqtgraph_structure_plot(parent), True
    return _create_pyqtgraph_structure_plot(parent), False


def resolve_canvas_host_widget(canvas_obj: object) -> QWidget:
    """Return the QWidget added to layouts for a backend canvas object."""
    native = getattr(canvas_obj, "native", None)
    if isinstance(native, QWidget):
        return native
    if isinstance(canvas_obj, QWidget):
        return canvas_obj
    if native is not None:
        return native
    return canvas_obj


def supports_structure_arrows(canvas_obj: object) -> bool:
    """Return True if a structure canvas supports arrow overlay APIs."""
    show_arrow = getattr(canvas_obj, "show_arrow", None)
    clear_arrow = getattr(canvas_obj, "clear_arrow", None)
    return callable(show_arrow) and callable(clear_arrow)
