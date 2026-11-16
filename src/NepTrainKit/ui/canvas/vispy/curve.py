"""VisPy point/line chart. All stored samples stay in data coordinates."""

import numpy as np
from PySide6.QtCore import QObject, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from vispy import scene
from vispy.gloo.context import forget_canvas

from NepTrainKit import palette
from NepTrainKit.ui.canvas.curve_factory import category_ticks, tick_spacing
from NepTrainKit.ui.canvas.vispy.distribution import _RotatedLabel


class _TickLabels(QWidget):
    """Qt uses the same logical pixels and fonts as the adjacent axis title."""
    def __init__(self, vertical, parent):
        super().__init__(parent)
        self.vertical = vertical
        self.bounds = (0., 1.)
        self.categories = None
        font = self.font()
        font.setPixelSize(15)
        self.setFont(font)
        if not vertical:
            self.setFixedHeight(self.fontMetrics().height() + 5)

    def set_bounds(self, lo, hi):
        self.bounds = float(lo), float(hi)
        if self.vertical:
            labels = self._ticks()
            self.setFixedWidth(max((self.fontMetrics().horizontalAdvance(text) for _, text in labels), default=24) + 8)
        self.update()

    def _ticks(self):
        lo, hi = self.bounds
        if hi <= lo:
            return []
        if self.categories is not None:
            return category_ticks(self.categories, lo, hi, self.width(), self.fontMetrics())
        fm = self.fontMetrics()
        gap = 55 if self.vertical else max(68, *(fm.horizontalAdvance(f"{v:.6g}") + 20 for v in self.bounds))
        length = self.height() if self.vertical else self.width()
        step = tick_spacing(hi - lo, length, gap)
        return [(n * step, f"{n * step:.6g}") for n in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1)]

    def paintEvent(self, event):
        lo, hi = self.bounds
        width, height = self.width(), self.height()
        length = height if self.vertical else width
        fm = self.fontMetrics()
        painter = QPainter(self)
        painter.setPen(QColor(palette.PLOT_AXIS))
        if self.vertical:
            painter.drawLine(width - 1, 0, width - 1, height)
        else:
            painter.drawLine(0, 0, width, 0)
        for value, text in self._ticks():
            pos = round((value - lo) / (hi - lo) * (length - 1))
            painter.setPen(QColor(palette.PLOT_AXIS))
            if self.vertical:
                y = height - 1 - pos
                painter.drawLine(width - 5, y, width - 1, y)
                rect = QRect(0, min(max(0, y - fm.height() // 2), height - fm.height()), width - 8, fm.height())
                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                painter.drawLine(pos, 0, pos, 4)
                tw = fm.horizontalAdvance(text) + 2
                rect = QRect(min(max(0, pos - tw // 2), width - tw), 4, tw, fm.height())
                align = Qt.AlignmentFlag.AlignCenter
            painter.setPen(QColor(palette.TEXT_PRIMARY))
            painter.drawText(rect, align, text)
        painter.end()


def _dashed_segments(points, units, rect, pattern):
    """Clip in both dimensions before allocating screen-space dash vertices."""
    low = np.array([rect.left, rect.bottom])
    high = np.array([rect.right, rect.top])
    offsets = np.r_[0., np.cumsum(pattern)]
    period = offsets[-1]
    segments = []
    phase = 0.
    for first, last in zip(points, points[1:]):
        delta = last - first
        length = np.linalg.norm(delta / units)
        if not length:
            continue
        moving = delta != 0
        if np.all(first[~moving] >= low[~moving]) and np.all(first[~moving] <= high[~moving]):
            a = (low[moving] - first[moving]) / delta[moving]
            b = (high[moving] - first[moving]) / delta[moving]
            begin = max(0., np.minimum(a, b).max()) * length
            end = min(1., np.maximum(a, b).min()) * length
            if begin < end:
                cycles = np.arange(begin - (phase + begin) % period, end, period)
                for left, right in zip(offsets[:-1:2], offsets[1::2]):
                    starts = np.maximum(cycles + left, begin)
                    stops = np.minimum(cycles + right, end)
                    visible = starts < stops
                    pair = np.column_stack((starts[visible], stops[visible])).reshape(-1, 1)
                    segments.append(first + (pair / length) * delta)
        phase = (phase + length) % period
    return np.concatenate(segments).astype(np.float32) if segments else np.empty((0, 2), np.float32)


class _Curve(QObject):
    def __init__(self, plot, xs, ys, pen, symbol, size, name):
        super().__init__(plot)
        self.chart = plot
        self.xData, self.yData = np.asarray(xs), np.asarray(ys)
        self.positions = np.column_stack((xs, ys)).astype(np.float32)
        self.opts = {"pen": pen, "symbol": symbol, "symbolPen": pen}
        self.label = name
        self.size = size
        self.line = scene.visuals.Line(parent=plot.view.scene, method="gl", antialias=True)
        self.points = scene.visuals.Markers(parent=plot.view.scene)
        self.polygon = scene.visuals.Mesh(parent=plot.view.scene)
        # Coplanar strokes and markers overlap in drawing order.
        for visual in (self.line, self.points, self.polygon):
            visual.update_gl_state(depth_test=False)
        self.visible = True
        self.refresh()

    def refresh(self):
        pen = self.opts["pen"] or self.opts["symbolPen"]
        color = pen.color().name()
        symbol = self.opts["symbol"]
        self.line.visible = self.visible and self.opts["pen"] is not None
        polygon = symbol in ("h", "p")
        self.points.visible = self.visible and bool(symbol) and not polygon
        self.polygon.visible = self.visible and polygon
        if not self.visible or not len(self.positions):
            return
        positions = self.positions
        # VisPy's public line API has no dash pattern. Split in screen space
        # so a dashed reference stays distinguishable at every zoom level.
        rect = self.chart.view.camera.rect
        size = np.maximum(self.chart.view.size, 1)
        units = np.maximum((rect.width, rect.height), 1e-15) / size
        pattern = pen.dashPattern()
        if pattern and len(positions) > 1:
            positions = _dashed_segments(positions, units, rect, np.asarray(pattern) * pen.widthF())
        self.line.set_data(pos=positions, color=color, width=pen.widthF(), connect="segments" if len(pattern) else "strip")
        if polygon:
            sides = 6 if symbol == "h" else 5
            angles = np.arange(sides) * (2 * np.pi / sides) + np.pi / 2
            ring = np.column_stack((np.cos(angles), np.sin(angles))) * units * self.size / 2
            verts = (self.positions[:, None, :] + ring).reshape(-1, 2)
            faces = np.array([[0, i, i + 1] for i in range(1, sides - 1)])
            faces = faces + (np.arange(len(self.positions)) * sides)[:, None, None]
            self.polygon.set_data(vertices=verts.astype(np.float32), faces=faces.reshape(-1, 3), color=color)
        elif symbol:
            name = {"t1": "triangle_up", "t": "triangle_down", "d": "diamond"}.get(symbol, symbol)
            self.points.set_data(self.positions, symbol=name, size=self.size, edge_width=0, face_color=color)

    def setPen(self, pen):
        self.opts["pen"] = pen
        self.refresh()
        self.chart.refresh_legend()

    def setSymbol(self, symbol):
        self.opts["symbol"] = symbol
        self.refresh()

    def setSymbolBrush(self, color):
        self.opts["symbolPen"] = QPen(QColor(color))
        self.refresh()
        self.chart.refresh_legend()

    def setSymbolPen(self, color):
        self.setSymbolBrush(color)

    def setVisible(self, visible):
        self.visible = visible
        self.refresh()
        self.chart.refresh_legend()

    def isVisible(self):
        return self.visible


class VispyCurvePlot(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        colors = self.palette()
        colors.setColor(QPalette.ColorRole.Window, QColor(palette.PLOT_BACKGROUND))
        self.setPalette(colors)
        self.setAutoFillBackground(True)
        self.canvas = scene.SceneCanvas(keys=None, show=False, size=(280, 160),
                                       bgcolor=palette.PLOT_BACKGROUND, config={"samples": 4})
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.legend = QLabel(self)
        self.legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self.canvas.native.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.y_label = _RotatedLabel("", self)
        grid = QGridLayout()
        layout.addLayout(grid, 1)
        grid.setSpacing(0)
        self.y_ticks = _TickLabels(True, self)
        self.x_ticks = _TickLabels(False, self)
        self.x_label = QLabel(self)
        self.x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.y_label, 0, 0)
        grid.addWidget(self.y_ticks, 0, 1)
        grid.addWidget(self.canvas.native, 0, 2)
        grid.addWidget(self.x_ticks, 1, 2)
        grid.addWidget(self.x_label, 2, 2)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        for label in (self.y_label, self.x_label):
            label.setStyleSheet(f"color: {palette.TEXT_PRIMARY}; font-size: 16px")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=None)
        self.curves = []
        self.zero = None
        self.view.scene.transform.changed.connect(self._view_changed)
        self.view.camera.transform.changed.connect(self._view_changed)
        # Qt owns the embedded GL widget. Canvas.close() pumps Qt events during destruction.
        canvas = self.canvas
        def release_canvas():
            canvas.events.close()
            forget_canvas(canvas)
        self.destroyed.connect(release_canvas)

    def viewport(self):
        return self.canvas.native

    def configure(self, xlabel, ylabel, legend, zero_pen=None):
        self.y_label.setText(ylabel)
        self.x_label.setText(xlabel)
        self.legend.setVisible(legend)
        if zero_pen is not None:
            self.add_zero_line(zero_pen)

    def add_zero_line(self, pen):
        self.zero = scene.visuals.InfiniteLine(pos=0, vertical=False, color=pen.color().getRgbF(),
                                             parent=self.view.scene)

    def set_category_ticks(self, ticks):
        self.x_ticks.categories = ticks
        self.x_ticks.update()

    def plot(self, xs, ys, *, pen, symbol, symbolSize, name, symbolPen=None, **_):
        marker_pen = pen or QPen(QColor(symbolPen or palette.PLOT_AXIS))
        curve = _Curve(self, xs, ys, marker_pen, symbol, symbolSize, name)
        if pen is None:
            curve.opts["pen"] = None
            curve.refresh()
        self.curves.append(curve)
        return curve

    def refresh_legend(self):
        self.legend.setText("　".join(f'<span style="color:{(c.opts["pen"] or c.opts["symbolPen"]).color().name()}">●</span> {c.label}'
                                     for c in self.curves if c.visible and c.label))

    def listDataItems(self):
        return self.curves

    def clear(self):
        self.x_ticks.categories = None
        for curve in self.curves:
            for visual in (curve.line, curve.points, curve.polygon):
                visual.parent = None
            curve.deleteLater()
        self.curves.clear()
        if self.zero is not None:
            self.zero.parent = None
            self.zero = None
        self.legend.clear()

    def _view_changed(self, *_):
        rect = self.view.camera.rect
        self.y_ticks.set_bounds(rect.bottom, rect.top)
        self.x_ticks.set_bounds(rect.left, rect.right)
        for curve in self.curves:
            pen = curve.opts["pen"]
            if (pen is not None and pen.dashPattern()) or curve.opts["symbol"] in ("p", "h"):
                curve.refresh()

    def finish(self):
        self.refresh_legend()
        self.fit()

    def fit(self):
        data = [c for c in self.curves if c.visible and len(c.xData)]
        if data:
            x = (min(c.xData.min() for c in data), max(c.xData.max() for c in data))
            y = (min(c.yData.min() for c in data), max(c.yData.max() for c in data))
            self.view.camera.set_range(x=x, y=y, z=(-1, 1), margin=.08)
            self._view_changed()

    def cursor_coordinates(self, pos):
        transform = self.canvas.scene.node_transform(self.view.scene)
        x, y = transform.map((pos.x(), pos.y()))[:2]
        rect = self.view.camera.rect
        if rect.left <= x <= rect.right and rect.bottom <= y <= rect.top:
            x1, y1 = transform.map((pos.x() + 1, pos.y() + 1))[:2]
            return x, y, abs(x1 - x), abs(y1 - y)
