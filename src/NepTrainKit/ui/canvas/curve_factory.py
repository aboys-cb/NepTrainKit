"""Select a point/line canvas without loading the unused rendering library."""
from functools import lru_cache
from math import ceil, floor, isfinite, log10

from PySide6.QtCore import Qt
from PySide6.QtGui import QOpenGLContext

from NepTrainKit.config import Config
from NepTrainKit.ui.canvas.canvas_factory import _normalize_canvas_type


def tick_spacing(span, pixels, gap):
    """Choose round values at a readable distance in logical pixels."""
    if not isfinite(span) or span <= 0:
        return 1.
    rough = span / max(2, min(10, pixels / gap))
    base = 10. ** floor(log10(rough))
    return next(n * base for n in (1, 2, 2.5, 5, 10) if n * base >= rough)


def category_ticks(categories, lo, hi, pixels, metrics):
    visible = [(x, str(label)) for x, label in categories if lo <= x <= hi]
    if not visible or pixels <= 0:
        return []
    gap = min(140, max(64, *(metrics.horizontalAdvance(label) + 16 for _, label in visible)))
    count = max(1, int(pixels // gap))
    selected = visible[::max(1, ceil(len(visible) / count))]
    width = max(24, int(pixels / len(selected)) - 12)
    return [(x, metrics.elidedText(label, Qt.TextElideMode.ElideRight, width)) for x, label in selected]


@lru_cache(maxsize=1)
def _has_gl_context():
    return QOpenGLContext().create()


def create_curve_plot(parent):
    from loguru import logger

    mode = _normalize_canvas_type(Config.get("widget", "canvas_type", "auto"))
    if mode == "vispy":
        try:
            if not _has_gl_context():
                raise RuntimeError("No OpenGL context available")
            from NepTrainKit.ui.canvas.vispy.curve import VispyCurvePlot
            return VispyCurvePlot(parent)
        except Exception as error:
            logger.warning("VisPy curve canvas unavailable, using PyQtGraph: {}", error)
    try:
        from NepTrainKit.ui.canvas.pyqtgraph.curve import PyqtgraphCurvePlot
    except ImportError:
        if mode == "vispy":
            raise
        from NepTrainKit.ui.canvas.vispy.curve import VispyCurvePlot
        return VispyCurvePlot(parent)
    return PyqtgraphCurvePlot(parent)
