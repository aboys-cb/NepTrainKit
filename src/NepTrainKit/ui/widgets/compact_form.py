"""Small, reusable building blocks for the card header/body visual language.

These widgets exist so individual cards do not have to re-invent status
indicators, category labels, or dense form layouts. They are intentionally
independent of `MakeDataCard` internals so they can be adopted by any card
body incrementally.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, themeColor

# Mirrors the colors `MakeDataCard` already uses for `status_label.set_colors(...)`
# (see ui/widgets/card_widget.py) so the header dot and the footer status text
# never disagree about what state a card is in.
STATUS_DOT_COLORS = {
    "idle": "#9aa4ae",
    "running": "#59745A",
    "succeeded": "#a5d6a7",
    "failed": "#ff0000",
    "canceled": "#d49b26",
    "canceling": "#d49b26",
    "disabled": "#c3c9cf",
}


class StatusDot(QWidget):
    """Small colored dot summarizing a card's run state at a glance."""

    def __init__(self, parent=None, diameter: int = 8):
        """Create the dot at a fixed pixel diameter, starting idle.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        diameter : int, optional
            Fixed width/height of the dot in pixels.

        """
        super().__init__(parent)
        self._diameter = diameter
        self._state = "idle"
        self._color = QColor(STATUS_DOT_COLORS["idle"])
        self.setFixedSize(diameter, diameter)

    def state(self) -> str:
        """Return the last state passed to `set_state`."""
        return self._state

    def set_state(self, state: str) -> None:
        """Recolor the dot for a `MakeDataCard.run_outcome`-style state.

        Unknown states fall back to the idle color rather than raising, since
        this is a purely cosmetic indicator.
        """
        self._state = state
        self._color = QColor(STATUS_DOT_COLORS.get(state, STATUS_DOT_COLORS["idle"]))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(self.rect())


class CategoryTag(QWidget):
    """Small pill identifying a card's functional category (e.g. "Doping").

    `ShareCheckableHeaderCardWidget` populates this automatically from a
    card's existing `group` class attribute, so most cards get a tag for
    free without any per-card changes.
    """

    def __init__(self, text: str = "", parent=None):
        """Create the tag, hidden until `text` is non-empty.

        Parameters
        ----------
        text : str, optional
            Category text to display; an empty string keeps the tag hidden.
        parent : QWidget, optional
            Parent widget responsible for ownership.

        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)
        self.label = CaptionLabel(text, self)
        layout.addWidget(self.label)
        self.setFixedHeight(20)
        self._refresh_style()
        self.setVisible(bool(text))

    def _refresh_style(self) -> None:
        color = themeColor()
        rgb = f"{color.red()}, {color.green()}, {color.blue()}"
        self.setStyleSheet(
            f"CategoryTag{{background:rgba({rgb}, 28); "
            f"border:1px solid rgba({rgb}, 90); border-radius:10px;}}"
        )
        self.label.setStyleSheet(f"color: rgb({rgb}); font-weight:600;")

    def setText(self, text: str) -> None:
        """Update the tag text, hiding the tag entirely when empty."""
        self.label.setText(text or "")
        self.setVisible(bool(text))

    def text(self) -> str:
        return self.label.text()


class CompactField(QWidget):
    """Label-above-input pair for a dense, aligned two-column form grid."""

    def __init__(self, label: str, input_widget: QWidget, parent=None):
        """Stack a caption above the given input widget.

        Parameters
        ----------
        label : str
            Caption text shown above the input.
        input_widget : QWidget
            The actual input control (spin box, combo box, etc.).
        parent : QWidget, optional
            Parent widget responsible for ownership.

        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.caption = CaptionLabel(label, self)
        self.caption.setStyleSheet("color:#8a95a0;")
        layout.addWidget(self.caption)
        layout.addWidget(input_widget)
        self.input_widget = input_widget

    def set_label(self, text: str) -> None:
        self.caption.setText(text)


class SegmentedControl(QWidget):
    """Compact pill-shaped multi-way choice.

    A denser alternative to radio buttons for a small, fixed set of
    mutually exclusive options.
    """

    currentIndexChanged = Signal(int)

    def __init__(self, options: list[str] | None = None, parent=None):
        """Build the control, optionally pre-populated with `options`.

        Parameters
        ----------
        options : list[str], optional
            Initial set of choice labels; the first one starts selected.
        parent : QWidget, optional
            Parent widget responsible for ownership.

        """
        super().__init__(parent)
        self._buttons: list[QPushButton] = []
        self._current = -1
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setFixedHeight(28)
        if options:
            self.set_options(options)

    def set_options(self, options: list[str]) -> None:
        """Replace the available choices, selecting the first one."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._buttons = []

        for index, text in enumerate(options):
            button = QPushButton(text, self)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("first", index == 0)
            button.setProperty("last", index == len(options) - 1)
            button.clicked.connect(lambda _checked, i=index: self._select(i))
            self._layout.addWidget(button, 1)
            self._buttons.append(button)

        self._apply_style()
        if options:
            self._select(0, emit=False)

    def _apply_style(self) -> None:
        accent = themeColor().name()
        self.setStyleSheet(
            "QPushButton{border:1px solid #d7dbe0; border-right:none; background:transparent;"
            " color:#57636e; font-size:12.5px; padding:0 6px;}"
            "QPushButton[last=\"true\"]{border-right:1px solid #d7dbe0;"
            " border-top-right-radius:6px; border-bottom-right-radius:6px;}"
            "QPushButton[first=\"true\"]{border-top-left-radius:6px; border-bottom-left-radius:6px;}"
            f"QPushButton:checked{{background:{accent}; color:white; font-weight:600; border-color:{accent};}}"
            "QPushButton:hover:!checked{background:rgba(0, 0, 0, 15);}"
        )

    def _select(self, index: int, emit: bool = True) -> None:
        self._current = index
        for i, button in enumerate(self._buttons):
            button.setChecked(i == index)
        if emit:
            self.currentIndexChanged.emit(index)

    def currentIndex(self) -> int:
        return self._current

    def setCurrentIndex(self, index: int) -> None:
        self._select(index, emit=False)

    def currentText(self) -> str:
        if 0 <= self._current < len(self._buttons):
            return self._buttons[self._current].text()
        return ""
