"""Labels that render gradient progress backgrounds."""

from PySide6.QtWidgets import QWidget
from qfluentwidgets import BodyLabel

from PySide6.QtGui import QPainter, QLinearGradient, QColor
from PySide6.QtCore import Qt, QRectF


class ProcessLabel(BodyLabel):
    """Label that overlays text on top of a progress-colored background."""

    _progress: int = 0
    _colors: list[QColor] = [QColor("white"), QColor("white")]

    def set_progress(self, value):
        """Update the progress percentage and trigger a repaint.

        Parameters
        ----------
        value : int
            Progress value between 0 and 100.
        """
        self._progress = max(0, min(100, value))
        self.update()

    def set_colors(self, colors):
        """Apply gradient colors used for the progress fill.

        Parameters
        ----------
        colors : Sequence[QColor | str]
            Colors that define the gradient from left to right. When a single
            color is supplied it is duplicated for a flat fill.
        """
        if len(colors) == 1:
            colors = [colors[0]] * 2

        self._colors = [QColor(c) if isinstance(c, str) else c for c in colors]

        self.update()

    def paintEvent(self, event):
        """Paint the gradient background and centered label text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        progress_width = self.width() * (self._progress / 100)

        gradient = QLinearGradient(0, 0, self.width(), 0)
        for i, color in enumerate(self._colors):
            gradient.setColorAt(i / (len(self._colors) - 1), color)

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(0, 0, self.width(), self.height()))

        if self._progress < 100:
            mask_color = QColor("white")
            painter.setBrush(mask_color)
            painter.drawRect(QRectF(progress_width, 0, self.width() - progress_width, self.height()))

        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


