"""Cursor readout for validation curves sampled on a sorted x axis."""

from bisect import bisect_left, bisect_right
from html import escape
from math import isclose, isfinite
from weakref import proxy

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel

from NepTrainKit.ui.widgets.validation_model_picker import MARKER_GLYPHS


class PlotPointHover(QLabel):
    def __init__(self, plot, x_label, unit):
        viewport = plot.viewport()
        super().__init__(viewport)
        self.plot = proxy(plot)
        self.x_label = x_label
        self.unit = unit
        self.setObjectName("validationPointHover")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "QLabel#validationPointHover { background: #263244; color: white; "
            "border-radius: 6px; padding: 7px 10px; font-size: 13px; }"
        )
        self.hide()
        viewport.setMouseTracking(True)
        viewport.installEventFilter(self)

    def eventFilter(self, viewport, event):
        kind = event.type()
        if kind == QEvent.Type.MouseMove and not event.buttons():
            self._follow(event.position().toPoint())
        elif kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._follow(event.position().toPoint(), pin=True)
            self.hide()
        elif kind in (QEvent.Type.Leave, QEvent.Type.Hide, QEvent.Type.MouseButtonPress,
                      QEvent.Type.Wheel, QEvent.Type.Resize):
            self.hide()
        return False

    def _follow(self, pos, pin=False):
        coords = self.plot.cursor_coordinates(pos)
        if coords is None:
            self.hide()
            return
        x, y, dx, dy = coords
        if not dx or not dy:
            self.hide()
            return
        curves = [c for c in self.plot.listDataItems() if c.isVisible()]
        radius = 14
        best_distance = radius ** 2
        sample = None
        for curve in curves:
            xs, ys = curve.xData, curve.yData
            if xs is None:
                continue
            start = bisect_left(xs, x - radius * dx)
            end = bisect_right(xs, x + radius * dx)
            for i in range(start, end):
                distance = ((xs[i] - x) / dx) ** 2 + ((ys[i] - y) / dy) ** 2
                if distance <= best_distance:
                    best_distance = distance
                    sample = xs[i]
                    hit = curve, i
        if sample is None:
            self.hide()
            return

        if pin:
            curve, index = hit
            if hasattr(curve, "read_point"):
                curve.read_point(curve, index)
            return

        # Match the physical coordinate, since each scan may have its own sampling grid.
        lines = [f"<b>{escape(self.x_label.format(sample))}</b>"]
        for curve in curves:
            xs = curve.xData
            if xs is None or not len(xs):
                continue
            i = bisect_left(xs, sample)
            index = min((j for j in (i - 1, i) if 0 <= j < len(xs)), key=lambda j: abs(xs[j] - sample))
            if not isclose(xs[index], sample, rel_tol=1e-9, abs_tol=1e-12):
                continue
            value = curve.yData[index]
            if not isfinite(value):
                continue
            marker = MARKER_GLYPHS.get(curve.opts["symbol"], "●")
            color = curve.opts["pen"].color().name()
            lines.append(f'<span style="color:{color}">{marker}</span>　<b>{value:.6g}</b> {self.unit}')

        content = "<br>".join(lines)
        if self.text() != content:
            self.setText(content)
            self.adjustSize()
        viewport = self.parentWidget()
        x, y = pos.x() + 14, pos.y() + 16
        if x + self.width() > viewport.width():
            x = pos.x() - self.width() - 14
        if y + self.height() > viewport.height():
            y = pos.y() - self.height() - 12
        self.move(max(0, x), max(0, y))
        self.show()
        self.raise_()
