"""Small, reusable building blocks for the card header/body visual language.

These widgets exist so individual cards do not have to re-invent status
indicators, category labels, or dense form layouts. They are intentionally
independent of `MakeDataCard` internals so they can be adopted by any card
body incrementally.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, StrongBodyLabel, isDarkTheme, themeColor

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

    stateChanged = Signal(str)

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

    def set_state(self, state: str, detail: str = "") -> None:
        """Recolor the dot for a `MakeDataCard.run_outcome`-style state.

        Unknown states fall back to the idle color rather than raising, since
        this is a purely cosmetic indicator.
        """
        self._state = state
        self._color = QColor(STATUS_DOT_COLORS.get(state, STATUS_DOT_COLORS["idle"]))
        self.update()
        self.stateChanged.emit(state)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(self.rect())


class StatusBadge(QFrame):
    """Compact text badge that communicates workflow state without colour alone."""

    _LIGHT_COLORS = {
        "idle": (88, 105, 113),
        "running": (16, 112, 166),
        "succeeded": (28, 137, 83),
        "failed": (190, 48, 48),
        "canceled": (166, 105, 16),
        "canceling": (166, 105, 16),
        "disabled": (100, 112, 118),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "idle"
        self.label = CaptionLabel(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.addWidget(self.label)
        self.setFixedHeight(22)
        self.set_state("idle")

    def state(self) -> str:
        return self._state

    def set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        labels = {
            "idle": self.tr("Ready"),
            "running": self.tr("Running"),
            "succeeded": self.tr("Done"),
            "failed": self.tr("Failed"),
            "canceled": self.tr("Stopped"),
            "canceling": self.tr("Stopping"),
            "disabled": self.tr("Skipped"),
        }
        role = state if state in labels else "idle"
        base_text = labels[role]
        detail = str(detail or "").strip()
        text = f"{base_text} · {detail}" if detail else base_text
        self.label.setText(text)
        # Runtime counts are part of the status contract; do not let the
        # header layout silently clip e.g. ``24→120`` into ``24→``.
        self.label.setFixedWidth(self.label.sizeHint().width())
        self.setAccessibleName(self.tr("Card status: {status}").format(status=text))
        red, green, blue = self._LIGHT_COLORS[role]
        text_color = (
            f"rgb({min(255, red + 80)}, {min(255, green + 80)}, {min(255, blue + 80)})"
            if isDarkTheme()
            else f"rgb({red}, {green}, {blue})"
        )
        self.setStyleSheet(
            "StatusBadge {"
            f"background: rgba({red}, {green}, {blue}, 24);"
            f"border: 1px solid rgba({red}, {green}, {blue}, 62);"
            "border-radius: 11px; }"
        )
        self.label.setStyleSheet(f"color: {text_color}; font-weight: 600;")


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
        layout.setContentsMargins(5, 0, 7, 0)
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
    """Label-above-input pair sized for the workflow inspector."""

    def __init__(
        self,
        label: str,
        input_widget: QWidget,
        parent=None,
        helper_text: str = "",
    ):
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
        layout.setSpacing(3)
        self.caption = CaptionLabel(label, self)
        self.caption.setStyleSheet("color:#8a95a0;")
        layout.addWidget(self.caption)
        layout.addWidget(input_widget)
        self.input_widget = input_widget
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        input_widget.setMinimumWidth(0)
        self.helper_label = CaptionLabel(helper_text, self)
        self.helper_label.setWordWrap(True)
        self.helper_label.setStyleSheet("color:#8a95a0;")
        self.helper_label.setVisible(bool(helper_text))
        layout.addWidget(self.helper_label)

    def set_label(self, text: str) -> None:
        self.caption.setText(text)

    def set_helper_text(self, text: str) -> None:
        self.helper_label.setText(text or "")
        self.helper_label.setVisible(bool(text))


class InspectorSection(QWidget):
    """Lightweight titled group for related inspector parameters."""

    def __init__(self, title: str, parent=None, description: str = ""):
        super().__init__(parent)
        self.setObjectName("inspectorSection")
        self.setStyleSheet(
            "QWidget#inspectorSection {"
            "background: rgba(100, 120, 128, 10);"
            "border: 1px solid rgba(100, 120, 128, 28);"
            "border-radius: 7px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 8)
        root.setSpacing(6)
        self.title_label = StrongBodyLabel(title, self)
        root.addWidget(self.title_label)
        self.description_label = CaptionLabel(description, self)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color:#8a95a0;")
        self.description_label.setVisible(bool(description))
        root.addWidget(self.description_label)
        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        root.addWidget(self.content_widget)

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt-style API
        self.content_layout.addWidget(widget)

    def addLayout(self, layout) -> None:  # noqa: N802 - Qt-style API
        self.content_layout.addLayout(layout)


class ResponsiveFormGrid(QWidget):
    """Reflow fields between one and two columns as inspector width changes."""

    def __init__(self, parent=None, two_column_threshold: int = 320):
        super().__init__(parent)
        self._threshold = two_column_threshold
        self._fields: list[tuple[QWidget, int]] = []
        self._column_count = 1
        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.timeout.connect(lambda: self._reflow(self.width()))
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(10)
        self._layout.setVerticalSpacing(6)

    def add_field(self, widget: QWidget, span: int = 1) -> None:
        widget.setMinimumWidth(0)
        self._fields.append((widget, max(1, span)))
        widget.installEventFilter(self)
        self._reflow(self.width())

    def column_count(self) -> int:
        return self._column_count

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._reflow(event.size().width())

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.Show, QEvent.Type.Hide) and any(
            watched is widget for widget, _span in self._fields
        ):
            self._reflow_timer.start(0)
        return super().eventFilter(watched, event)

    def _reflow(self, width: int) -> None:
        columns = 2 if width >= self._threshold else 1
        while self._layout.count():
            self._layout.takeAt(0)
        visible_fields = [
            (widget, span)
            for widget, span in self._fields
            if not widget.isHidden()
        ]
        for widget, _span in self._fields:
            widget.setProperty("responsiveGridRow", -1)
            widget.setProperty("responsiveGridColumn", -1)
            widget.setProperty("responsiveGridSpan", 0)
        row = 0
        column = 0
        for index, (widget, requested_span) in enumerate(visible_fields):
            span = min(requested_span, columns)
            if columns == 2 and span == 1 and column == 0:
                next_span = (
                    min(visible_fields[index + 1][1], columns)
                    if index + 1 < len(visible_fields)
                    else columns
                )
                if next_span == columns:
                    span = columns
            if span == columns or column + span > columns:
                if column:
                    row += 1
                column = 0
            self._layout.addWidget(widget, row, column, 1, span)
            widget.setProperty("responsiveGridRow", row)
            widget.setProperty("responsiveGridColumn", column)
            widget.setProperty("responsiveGridSpan", span)
            if span == columns:
                row += 1
                column = 0
            else:
                column += span
                if column >= columns:
                    row += 1
                    column = 0
        for index in range(columns):
            self._layout.setColumnStretch(index, 1)
        self._column_count = columns


class _LegacyCompactField(QWidget):
    """Rehouse one legacy label/control row without replacing its widgets."""

    def __init__(self, label: QLabel, entries: list[tuple], parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)
        label.setParent(self)
        label.setStyleSheet("color:#8a95a0;")
        label.setWordWrap(True)
        root.addWidget(label)

        content = QWidget(self)
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(6)
        content_layout.setVerticalSpacing(3)
        first_column = min(entry[1] for entry in entries)
        for _row, column, row_span, column_span, item in entries:
            _add_layout_item(
                content_layout,
                item,
                0,
                column - first_column,
                row_span,
                column_span,
            )
        for column in range(content_layout.columnCount()):
            content_layout.setColumnStretch(column, 1)
        root.addWidget(content)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._tracked_widgets = [label]
        self._tracked_widgets.extend(
            entry[4].widget()
            for entry in entries
            if entry[4].widget() is not None
        )
        for widget in self._tracked_widgets:
            widget.installEventFilter(self)
        self._visibility_timer = QTimer(self)
        self._visibility_timer.setSingleShot(True)
        self._visibility_timer.timeout.connect(self._sync_visibility)
        self._visibility_timer.start(0)

    def _sync_visibility(self) -> None:
        self.setVisible(any(not widget.isHidden() for widget in self._tracked_widgets))

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
            self._visibility_timer.start(0)
        return super().eventFilter(watched, event)


def _add_layout_item(layout: QGridLayout, item, row, column, row_span, column_span):
    widget = item.widget()
    if widget is not None:
        widget.setMinimumWidth(0)
        widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            widget.sizePolicy().verticalPolicy(),
        )
        layout.addWidget(widget, row, column, row_span, column_span)
        return
    child_layout = item.layout()
    if child_layout is not None:
        layout.addLayout(child_layout, row, column, row_span, column_span)
        return
    spacer = item.spacerItem()
    if spacer is not None:
        layout.addItem(spacer, row, column, row_span, column_span)


def _row_host(entries: list[tuple], parent: QWidget) -> QWidget:
    host = QWidget(parent)
    layout = QGridLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(6)
    layout.setVerticalSpacing(3)
    first_column = min(entry[1] for entry in entries)
    for _row, column, row_span, column_span, item in entries:
        _add_layout_item(
            layout,
            item,
            0,
            column - first_column,
            row_span,
            column_span,
        )
    for column in range(layout.columnCount()):
        layout.setColumnStretch(column, 1)
    host.setMinimumWidth(0)
    host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return host


def _button_grid_host(entries: list[tuple], parent: QWidget) -> QWidget:
    """Wrap a legacy row of choices instead of clipping its last label."""
    host = QWidget(parent)
    layout = QGridLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(4)
    for index, entry in enumerate(entries):
        _add_layout_item(layout, entry[4], index // 2, index % 2, 1, 1)
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 1)
    host.setMinimumWidth(0)
    host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return host


def _label_widget(entry: tuple) -> QLabel | None:
    widget = entry[4].widget()
    return widget if isinstance(widget, QLabel) else None


def adapt_legacy_inspector_form(editor: QWidget, layout: QLayout | None) -> bool:
    """Reflow an old label-left form without replacing its controls.

    Keeping the original widgets preserves card attributes, connections and
    serialization. Bespoke forms already using ``InspectorSection`` are left
    untouched.
    """
    if layout is None or editor.property("legacyInspectorFormAdapted"):
        return False
    if not isinstance(layout, QGridLayout):
        return False
    if any(
        isinstance(layout.itemAt(index).widget(), InspectorSection)
        for index in range(layout.count())
    ):
        editor.setProperty("legacyInspectorFormAdapted", True)
        return False

    entries: list[tuple] = []
    for index in range(layout.count()):
        row, column, row_span, column_span = layout.getItemPosition(index)
        entries.append((row, column, row_span, column_span, layout.itemAt(index)))
    if not entries:
        editor.setProperty("legacyInspectorFormAdapted", True)
        return False

    while layout.count():
        layout.takeAt(0)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    form = ResponsiveFormGrid(editor)
    rows: dict[int, list[tuple]] = {}
    for entry in entries:
        rows.setdefault(entry[0], []).append(entry)

    for row_entries in rows.values():
        labels = [entry for entry in row_entries if _label_widget(entry) is not None]
        if len(labels) == 1 and len(row_entries) > 1:
            label_entry = labels[0]
            controls = [entry for entry in row_entries if entry is not label_entry]
            needs_full_width = any(
                entry[4].widget() is not None
                and (
                    entry[4].widget().minimumSizeHint().width() > 300
                    or len(getattr(entry[4].widget(), "object_list", ())) > 1
                    or isinstance(entry[4].widget(), (QCheckBox, QRadioButton))
                )
                for entry in controls
            )
            form.add_field(
                _LegacyCompactField(_label_widget(label_entry), controls, form),
                span=2 if needs_full_width else 1,
            )
            continue

        # Alternate editors can deliberately occupy the same old grid cells.
        # Keep that visibility-driven overlay while stacking each caption.
        if len(labels) > 1:
            overlay = QWidget(form)
            overlay_layout = QGridLayout(overlay)
            overlay_layout.setContentsMargins(0, 0, 0, 0)
            for index, label_entry in enumerate(labels):
                start = row_entries.index(label_entry) + 1
                end = (
                    row_entries.index(labels[index + 1])
                    if index + 1 < len(labels)
                    else len(row_entries)
                )
                controls = row_entries[start:end]
                if controls:
                    overlay_layout.addWidget(
                        _LegacyCompactField(
                            _label_widget(label_entry), controls, overlay
                        ),
                        0,
                        0,
                    )
            form.add_field(overlay, span=2)
            continue

        button_only = all(
            isinstance(entry[4].widget(), QAbstractButton)
            for entry in row_entries
        )
        host = (
            _button_grid_host(row_entries, form)
            if button_only
            else _row_host(row_entries, form)
        )
        has_button = any(
            isinstance(entry[4].widget(), QAbstractButton)
            for entry in row_entries
        )
        form.add_field(
            host,
            span=2 if has_button or len(row_entries) != 2 else 1,
        )

    layout.addWidget(form, 0, 0, 1, 3)
    editor.setProperty("legacyInspectorFormAdapted", True)
    return True


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
        self._data: list[object] = []
        self._current = -1
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
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
        self._data = []

        for text in options:
            self._append_button(text, text)

        self._apply_style()
        if options:
            self._select(0, emit=False)

    def _append_button(self, text: str, data) -> None:
        index = len(self._buttons)
        button = QPushButton(text, self)
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked, i=index: self._select(i))
        self._layout.addWidget(button, 1)
        self._buttons.append(button)
        self._data.append(data)
        for i, item in enumerate(self._buttons):
            item.setProperty("first", i == 0)
            item.setProperty("last", i == len(self._buttons) - 1)
            item.style().unpolish(item)
            item.style().polish(item)

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
        if 0 <= index < len(self._buttons):
            self._select(index, emit=index != self._current)

    def currentText(self) -> str:
        if 0 <= self._current < len(self._buttons):
            return self._buttons[self._current].text()
        return ""

    def itemText(self, index: int) -> str:  # noqa: N802 - ComboBox-compatible API
        if 0 <= index < len(self._buttons):
            return self._buttons[index].text()
        return ""

    def itemData(self, index: int):  # noqa: N802 - ComboBox-compatible API
        if 0 <= index < len(self._data):
            return self._data[index]
        return None

    def addItem(self, text: str, userData=None) -> None:  # noqa: N802 - ComboBox-compatible API
        self._append_button(text, text if userData is None else userData)
        self._apply_style()
        if self._current < 0:
            self._select(0, emit=False)

    def count(self) -> int:
        return len(self._buttons)

    def currentData(self):  # noqa: N802 - ComboBox-compatible API
        if 0 <= self._current < len(self._data):
            return self._data[self._current]
        return None

    def findData(self, value) -> int:  # noqa: N802 - ComboBox-compatible API
        try:
            return self._data.index(value)
        except ValueError:
            return -1

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        index = next(
            (
                i
                for i, button in enumerate(self._buttons)
                if button.text() == text or self._data[i] == text
            ),
            -1,
        )
        if index >= 0:
            self.setCurrentIndex(index)

    def setText(self, text: str) -> None:  # noqa: N802 - EditableComboBox compatibility
        self.setCurrentText(text)
