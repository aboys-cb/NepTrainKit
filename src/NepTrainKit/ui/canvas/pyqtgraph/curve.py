"""PyQtGraph rendering for sampled curves."""
import pyqtgraph as pg
from PySide6.QtGui import QFontMetrics
from PySide6.QtCore import Qt

from NepTrainKit import palette
from NepTrainKit.ui.canvas.curve_factory import category_ticks, tick_spacing


class _CurveAxis(pg.AxisItem):
    categories = None

    def tickValues(self, minVal, maxVal, size):
        if self.categories is None:
            return super().tickValues(minVal, maxVal, size)
        metrics = QFontMetrics(self.style["tickFont"] or self.font())
        ticks = category_ticks(self.categories, minVal, maxVal, size, metrics)
        self._category_strings = dict(ticks)
        return [(1, [x for x, _ in ticks])]

    def tickStrings(self, values, scale, spacing):
        if self.categories is None:
            return super().tickStrings(values, scale, spacing)
        return [self._category_strings.get(value, "") for value in values]

    def tickSpacing(self, minVal, maxVal, size):
        gap = 55 if self.orientation == "left" else max(
            68, *(QFontMetrics(self.style["tickFont"] or self.font()).horizontalAdvance(f"{value:.6g}") + 24 for value in (maxVal, minVal)))
        return [(tick_spacing(abs(maxVal - minVal), size, gap), 0)]


class PyqtgraphCurvePlot(pg.PlotWidget):
    def __init__(self, parent):
        super().__init__(parent, background=palette.PLOT_BACKGROUND, axisItems={side: _CurveAxis(side) for side in ("left", "bottom")})

    def plot(self, *args, antialias=True, **kwargs):
        return self.plotItem.plot(*args, antialias=antialias, **kwargs)

    def configure(self, xlabel, ylabel, legend, zero_pen=None):
        self.getAxis("bottom").categories = None
        if legend and self.plotItem.legend is None:
            self.addLegend(offset=(8, 3), colCount=2, labelTextSize="11pt",
                           labelTextColor=palette.TEXT_SECONDARY)
        if self.plotItem.legend:
            self.plotItem.legend.setVisible(legend)
        self.showGrid(x=True, y=True, alpha=0.12)
        self.getViewBox().state["wheelScaleFactor"] = -0.03
        font = self.font()
        font.setPixelSize(15)
        for key, label in (("bottom", xlabel), ("left", ylabel)):
            axis = self.getAxis(key)
            axis.setTickFont(font)
            axis.setPen(pg.mkPen(palette.BORDER))
            axis.setTextPen(pg.mkPen(palette.TEXT_PRIMARY))
            axis.setStyle(maxTickLevel=0, maxTextLevel=0, tickTextOffset=3, tickLength=-4)
            self.setLabel(key, label, **{"font-size": "16px", "color": palette.TEXT_PRIMARY})
        if zero_pen is not None:
            self.addLine(y=0, pen=zero_pen)

    def finish(self):
        if self.plotItem.legend:
            for sample, _ in self.plotItem.legend.items:
                sample.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.fit()

    def fit(self):
        view = self.getViewBox()
        view.enableAutoRange(x=True, y=True)
        view.updateAutoRange()
        if hasattr(self, "comparison_y_range"):
            self.setYRange(*self.comparison_y_range, padding=.08)

    def add_zero_line(self, pen):
        self.addLine(y=0, pen=pen)

    def set_category_ticks(self, ticks):
        axis = self.getAxis("bottom")
        axis.categories = ticks
        axis.picture = None
        axis.update()

    def cursor_coordinates(self, pos):
        view = self.getViewBox()
        scene = self.mapToScene(pos)
        if view.sceneBoundingRect().contains(scene):
            point = view.mapSceneToView(scene)
            dx, dy = view.viewPixelSize()
            return point.x(), point.y(), abs(dx), abs(dy)
