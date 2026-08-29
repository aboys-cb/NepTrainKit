"""Searchable Fluent-style periodic-table element picker."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentStyleSheet,
    FluentTitleBar,
    PrimaryPushButton,
    PushButton,
    SearchLineEdit,
    StrongBodyLabel,
    isDarkTheme,
    qconfig,
    themeColor,
)

from NepTrainKit import module_path

if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM", "").split(":")[0].lower() == "offscreen":

    class FramelessDialog(QDialog):
        """Headless-safe stand-in for qframelesswindow's macOS dialog."""

        def setTitleBar(self, title_bar):  # noqa: N802 - qframelesswindow API
            self.titleBar = title_bar
            title_bar.setParent(self)

else:
    from qframelesswindow import FramelessDialog


_ELEMENT_FAMILY_COLORS = {
    "alkali": "#ef6c6c",
    "alkaline": "#e6a23c",
    "transition": "#4f8bd6",
    "post_transition": "#6f9f80",
    "metalloid": "#23a5a5",
    "nonmetal": "#7e8bd2",
    "halogen": "#9a72cf",
    "noble": "#5b9bd5",
    "lanthanide": "#d87aa6",
    "actinide": "#bb6f9a",
}


def _element_family(number: int, group: int, symbol: str) -> str:
    """Return a compact visual family used only by the periodic-table UI."""
    if 57 <= number <= 71:
        return "lanthanide"
    if 89 <= number <= 103:
        return "actinide"
    if group == 18:
        return "noble"
    if group == 17:
        return "halogen"
    if group == 1 and symbol != "H":
        return "alkali"
    if group == 2:
        return "alkaline"
    if 3 <= group <= 12:
        return "transition"
    if symbol in {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}:
        return "metalloid"
    if symbol in {"H", "C", "N", "O", "P", "S", "Se"}:
        return "nonmetal"
    return "post_transition"


class _PeriodicElementButton(QAbstractButton):
    """Compact, theme-aware element tile with visible keyboard focus."""

    doubleClicked = Signal()

    def __init__(self, number: int, info: dict[str, Any], parent=None):
        super().__init__(parent)
        self.atomic_number = int(number)
        self.symbol = str(info["symbol"])
        self.element_name = str(info.get("name", self.symbol))
        family = _element_family(self.atomic_number, int(info.get("group", 0)), self.symbol)
        self.family_color = QColor(_ELEMENT_FAMILY_COLORS[family])
        self.search_match = True
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(f"{self.atomic_number} · {self.symbol} · {self.element_name}")
        self.setAccessibleName(f"{self.element_name}, {self.symbol}, atomic number {self.atomic_number}")

    def set_search_match(self, matches: bool) -> None:
        self.search_match = bool(matches)
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt event API
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.search_match:
            painter.setOpacity(0.22)

        dark = isDarkTheme()
        surface = QColor(self.family_color)
        surface.setAlpha(42 if dark else 28)
        outline = QColor(self.family_color)
        outline.setAlpha(105 if dark else 78)
        accent = themeColor()
        if self.isChecked():
            surface = QColor(accent)
            surface.setAlpha(72 if dark else 42)
            outline = QColor(accent)
        elif self.underMouse():
            surface = QColor(self.family_color)
            surface.setAlpha(70 if dark else 48)
            outline = QColor(accent)
            outline.setAlpha(150)
        if self.hasFocus():
            outline = QColor(accent)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(surface)
        painter.setPen(QPen(outline, 1.6 if self.isChecked() or self.hasFocus() else 1.0))
        painter.drawRoundedRect(rect, 4, 4)

        text_color = QColor("#f5f5f5" if dark else "#17191c")
        painter.setPen(text_color)
        number_font = self.font()
        number_font.setPointSizeF(max(5.5, number_font.pointSizeF() - 3.5))
        painter.setFont(number_font)
        painter.drawText(
            self.rect().adjusted(3, 0, -2, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            str(self.atomic_number),
        )

        symbol_font = self.font()
        symbol_font.setPointSizeF(max(8.5, symbol_font.pointSizeF() - 0.5))
        symbol_font.setBold(True)
        painter.setFont(symbol_font)
        painter.drawText(
            self.rect().adjusted(0, 4, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            self.symbol,
        )


class PeriodicTableDialog(FramelessDialog):
    """Searchable Fluent-style single-element picker."""

    elementSelected = Signal(str)

    def __init__(self, parent=None):
        """Create the single-selection periodic-table dialog."""
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))
        self.setWindowTitle(self.tr("Periodic table"))
        self.setWindowIcon(QIcon(":/images/src/images/logo.png"))
        self.setMinimumSize(676, 456)
        self.resize(700, 486)

        with open(module_path / "Config/ptable.json", encoding="utf-8") as f:
            self.table_data = {int(k): v for k, v in json.load(f).items()}

        self.selected_symbol = ""
        self.element_buttons: dict[str, _PeriodicElementButton] = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        FluentStyleSheet.DIALOG.apply(self)
        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(18, 8, 18, 12)
        self.__layout.setSpacing(8)
        self.__layout.setMenuBar(self.titleBar)

        header = QHBoxLayout()
        header.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        self.heading_label = StrongBodyLabel(self.tr("Choose an element"), self)
        self.help_label = CaptionLabel(
            self.tr("Search or select an element from its periodic-table position."),
            self,
        )
        heading.addWidget(self.heading_label)
        heading.addWidget(self.help_label)
        header.addLayout(heading, 1)
        self.search_edit = SearchLineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("Symbol, name, or number"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(250)
        header.addWidget(self.search_edit)
        self.__layout.addLayout(header)

        self.table_frame = QFrame(self)
        self.table_frame.setObjectName("periodicTableSurface")
        table_layout = QGridLayout(self.table_frame)
        table_layout.setContentsMargins(10, 9, 10, 10)
        table_layout.setHorizontalSpacing(5)
        table_layout.setVerticalSpacing(5)
        table_layout.setRowMinimumHeight(8, 5)
        for group in range(1, 19):
            label = CaptionLabel(str(group), self.table_frame)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            table_layout.addWidget(label, 0, group - 1)

        self.lanthanide_label = CaptionLabel(self.tr("Lanthanides"), self.table_frame)
        self.actinide_label = CaptionLabel(self.tr("Actinides"), self.table_frame)
        self.lanthanide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.actinide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        table_layout.addWidget(self.lanthanide_label, 9, 0, 1, 2)
        table_layout.addWidget(self.actinide_label, 10, 0, 1, 2)

        for num in range(1, 119):
            info = self.table_data.get(num)
            if not info:
                continue
            group = info.get("group", 0)
            period = self._get_period(num)
            row, col = self._grid_position(num, group, period)
            layout_row = row + 1 if row < 7 else row + 2
            button = _PeriodicElementButton(num, info, self.table_frame)
            button.clicked.connect(lambda _checked=False, symbol=info["symbol"]: self._select_element(symbol))
            button.doubleClicked.connect(lambda symbol=info["symbol"]: self._commit_element(symbol))
            self._button_group.addButton(button, num)
            self.element_buttons[str(info["symbol"])] = button
            table_layout.addWidget(button, layout_row, col)
        self.__layout.addWidget(self.table_frame, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.selection_label = BodyLabel(self.tr("No element selected"), self)
        footer.addWidget(self.selection_label, 1)
        self.cancel_button = PushButton(self.tr("Cancel"), self)
        self.choose_button = PrimaryPushButton(self.tr("Choose"), self)
        self.choose_button.setEnabled(False)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.choose_button)
        self.__layout.addLayout(footer)

        self.search_edit.textChanged.connect(self._apply_search)
        self.search_edit.returnPressed.connect(self._select_first_match)
        self.cancel_button.clicked.connect(self.reject)
        self.choose_button.clicked.connect(self._commit_selection)
        qconfig.themeChangedFinished.connect(self._refresh_theme)
        self._refresh_theme()

    def _get_period(self, num: int) -> int:
        if num <= 2:
            return 1
        elif num <= 10:
            return 2
        elif num <= 18:
            return 3
        elif num <= 36:
            return 4
        elif num <= 54:
            return 5
        elif num <= 86:
            return 6
        else:
            return 7

    def _grid_position(self, num: int, group: int, period: int) -> tuple[int, int]:
        if group == 0:
            if 57 <= num <= 71:
                row = 8
                col = num - 54
            elif 89 <= num <= 103:
                row = 9
                col = num - 86
            else:
                row, col = period, 1
        else:
            row, col = period, group
        return row - 1, col - 1

    def _refresh_theme(self) -> None:
        if isDarkTheme():
            surface = "#252525"
            border = "#454545"
        else:
            surface = "#f7f9fb"
            border = "#dce2e9"
        self.table_frame.setStyleSheet(
            f"QFrame#periodicTableSurface {{background: {surface}; border: 1px solid {border}; border-radius: 8px;}}"
        )
        for button in self.element_buttons.values():
            button.update()

    def _select_element(self, symbol: str) -> None:
        button = self.element_buttons.get(symbol)
        if button is None:
            return
        self.selected_symbol = symbol
        button.setChecked(True)
        info = self.table_data[button.atomic_number]
        self.selection_label.setText(
            self.tr("{symbol} · {name} · atomic number {number}").format(
                symbol=symbol,
                name=info.get("name", symbol),
                number=button.atomic_number,
            )
        )
        self.choose_button.setText(self.tr("Choose {symbol}").format(symbol=symbol))
        self.choose_button.setEnabled(True)

    def set_selected_element(self, symbol: str) -> None:
        """Preselect an element without committing or closing the dialog."""
        self._select_element(str(symbol).strip())

    def set_prompt(self, heading: str, help_text: str = "") -> None:
        """Adapt the picker guidance for a specific element-input workflow."""
        self.heading_label.setText(str(heading))
        self.help_label.setText(str(help_text))
        self.help_label.setVisible(bool(str(help_text).strip()))

    def _commit_element(self, symbol: str) -> None:
        self._select_element(symbol)
        self._commit_selection()

    def _apply_search(self, text: str) -> None:
        query = str(text or "").strip().casefold()
        for button in self.element_buttons.values():
            searchable = f"{button.symbol} {button.element_name} {button.atomic_number}".casefold()
            button.set_search_match(not query or query in searchable)

    def _select_first_match(self) -> None:
        for button in self.element_buttons.values():
            if button.search_match:
                self._select_element(button.symbol)
                button.setFocus(Qt.FocusReason.ShortcutFocusReason)
                return

    def _commit_selection(self) -> None:
        if not self.selected_symbol:
            return
        self.elementSelected.emit(self.selected_symbol)
        self.accept()


__all__ = ["PeriodicTableDialog"]
