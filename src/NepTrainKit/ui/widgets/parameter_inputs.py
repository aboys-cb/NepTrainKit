"""Reusable user-facing editors for common scientific card parameters."""

from __future__ import annotations

import math
import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import CheckBox, ComboBox, LineEdit, PushButton

from .input import RangeTripletInputFrame, SpinBoxUnitInputFrame


def _format_number(value: float) -> str:
    value = 0.0 if abs(float(value)) < 1.0e-12 else float(value)
    return f"{value:.12g}"


class NumericScanInput(QWidget):
    """Minimum/maximum/step scan with an explicit custom-list escape hatch.

    The public value remains the comma-separated representation used by the
    existing core Params classes, so adopting this widget does not change the
    persisted card contract.
    """

    def __init__(
        self,
        parent=None,
        *,
        minimum: float,
        maximum: float,
        decimals: int = 3,
        suffix: str = "",
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(125)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.range_frame = RangeTripletInputFrame(self, suffix=suffix)
        self.range_frame.setRange(minimum, maximum)
        self.range_frame.setDecimals(decimals)
        self.custom_checkbox = CheckBox(self.tr("Use custom coordinate list"), self)
        self.custom_edit = LineEdit(self)
        self.custom_edit.setPlaceholderText(self.tr("Comma-separated values"))
        self.custom_edit.hide()
        self.custom_checkbox.toggled.connect(self.custom_edit.setVisible)
        self.custom_checkbox.toggled.connect(
            lambda checked: self.setMaximumHeight(160 if checked else 125)
        )
        layout.addWidget(self.range_frame)
        layout.addWidget(self.custom_checkbox)
        layout.addWidget(self.custom_edit)

    def set_range(self, minimum: float, maximum: float, step: float) -> None:
        self.custom_checkbox.setChecked(False)
        self.range_frame.set_input_value([minimum, maximum, step])

    def set_scan_text(self, text: str) -> None:
        raw = str(text or "").strip()
        try:
            values = [float(item.strip()) for item in raw.split(",") if item.strip()]
        except ValueError:
            values = []
        if len(values) >= 2:
            step = values[1] - values[0]
            if step > 0.0 and all(
                math.isclose(values[index] - values[index - 1], step, rel_tol=1.0e-9, abs_tol=1.0e-12)
                for index in range(2, len(values))
            ):
                self.set_range(values[0], values[-1], step)
                return
        self.custom_edit.setText(raw)
        self.custom_checkbox.setChecked(True)

    def values(self) -> list[float]:
        if self.custom_checkbox.isChecked():
            raw_values = [item.strip() for item in self.custom_edit.text().split(",") if item.strip()]
            if not raw_values:
                raise ValueError(self.tr("Custom coordinate list cannot be empty."))
            try:
                values = [float(item) for item in raw_values]
            except ValueError as exc:
                raise ValueError(self.tr("Custom coordinate list must contain only numbers.")) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(self.tr("Custom coordinate list must contain only finite numbers."))
            return values

        minimum, maximum, step = (float(value) for value in self.range_frame.get_input_value())
        if step <= 0.0:
            raise ValueError(self.tr("Scan step must be positive."))
        if maximum < minimum:
            raise ValueError(self.tr("Scan maximum must be greater than or equal to the minimum."))
        count = int(math.floor((maximum - minimum) / step + 1.0e-9)) + 1
        if count > 10000:
            raise ValueError(self.tr("Coordinate scan contains more than 10000 points."))
        values = [minimum + index * step for index in range(count)]
        if not math.isclose(values[-1], maximum, rel_tol=1.0e-9, abs_tol=1.0e-10):
            values.append(maximum)
        return values

    def scan_text(self) -> str:
        if self.custom_checkbox.isChecked():
            self.values()  # validate without rewriting the user's explicit coordinates
            return ",".join(
                item.strip() for item in self.custom_edit.text().split(",") if item.strip()
            )
        return ",".join(_format_number(value) for value in self.values())

    # LineEdit-compatible aliases keep existing card tests and external UI
    # automation working while the visible editor becomes structured.
    def text(self) -> str:
        return self.scan_text()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt compatibility
        self.set_scan_text(text)

    def count(self) -> int:
        return len(self.values())


class DirectionInput(QWidget):
    """Common Cartesian direction presets with a normalized custom vector."""

    _PRESETS = (
        ("x", "[100] / X", (1.0, 0.0, 0.0)),
        ("y", "[010] / Y", (0.0, 1.0, 0.0)),
        ("z", "[001] / Z", (0.0, 0.0, 1.0)),
        ("xy", "[110]", (1.0, 1.0, 0.0)),
        ("xyz", "[111]", (1.0, 1.0, 1.0)),
    )

    def __init__(self, parent=None, *, default=(0.0, 0.0, 1.0)):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(90)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.preset_combo = ComboBox(self)
        for key, label, _vector in self._PRESETS:
            self.preset_combo.addItem(self.tr(label), userData=key)
        self.preset_combo.addItem(self.tr("Custom Cartesian vector"), userData="custom")
        self.vector_frame = SpinBoxUnitInputFrame(self)
        self.vector_frame.set_input("", 3, "float")
        self.vector_frame.setRange(-100.0, 100.0)
        self.vector_frame.setDecimals(8)
        layout.addWidget(self.preset_combo)
        layout.addWidget(self.vector_frame)
        self.preset_combo.currentIndexChanged.connect(self._update_custom_visibility)
        self.set_vector(default)

    @staticmethod
    def _normalized(vector) -> tuple[float, float, float]:
        values = tuple(float(value) for value in vector)
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1.0e-12:
            raise ValueError("Direction vector must be nonzero.")
        return tuple(value / norm for value in values)

    def _update_custom_visibility(self, _index: int = -1) -> None:
        self.vector_frame.setVisible(self.preset_combo.currentData() == "custom")

    def vector(self) -> tuple[float, float, float]:
        key = self.preset_combo.currentData()
        if key == "custom":
            return self._normalized(self.vector_frame.get_input_value())
        for preset_key, _label, vector in self._PRESETS:
            if key == preset_key:
                return self._normalized(vector)
        raise ValueError("Unknown direction preset.")

    def set_vector(self, vector) -> None:
        normalized = self._normalized(vector)
        for index, (_key, _label, preset) in enumerate(self._PRESETS):
            if all(math.isclose(value, ref, abs_tol=1.0e-9) for value, ref in zip(normalized, self._normalized(preset))):
                self.preset_combo.setCurrentIndex(index)
                self.vector_frame.set_input_value(list(normalized))
                self._update_custom_visibility()
                return
        custom_index = self.preset_combo.findData("custom")
        self.preset_combo.setCurrentIndex(custom_index)
        self.vector_frame.set_input_value(list(normalized))
        self._update_custom_visibility()

    def get_input_value(self) -> list[float]:
        return list(self.vector())

    def set_input_value(self, values) -> None:
        self.set_vector(values)


class KeyValueTableInput(QWidget):
    """Small editable table that serializes to the existing ``key:value`` form."""

    editingFinished = Signal()

    def __init__(self, key_title: str, value_title: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(205)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels([key_title, value_title])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(96)
        self.table.setMaximumHeight(150)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.add_button = PushButton(self.tr("Add row"), self)
        self.remove_button = PushButton(self.tr("Remove selected"), self)
        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected)
        self.table.itemChanged.connect(lambda _item: self.editingFinished.emit())
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def add_row(self, key: str = "", value: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(key)))
        self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)

    def clear(self) -> None:
        self.table.setRowCount(0)

    def set_text(self, text: str, *, default_value: str = "1") -> None:
        self.clear()
        raw = str(text or "").strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                mapping = json.loads(raw)
            except json.JSONDecodeError:
                mapping = {}
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    self.add_row(str(key), str(value))
                return
        for token in (item.strip() for item in raw.replace(";", ",").split(",")):
            if not token:
                continue
            separator = ":" if ":" in token else "=" if "=" in token else None
            if separator:
                key, value = token.split(separator, 1)
            else:
                key, value = token, default_value
            self.add_row(key.strip(), value.strip())

    def text(self) -> str:
        entries: list[str] = []
        json_entries: dict[str, object] = {}
        has_structured_value = False
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = key_item.text().strip() if key_item else ""
            value = value_item.text().strip() if value_item else ""
            if key:
                value = value or "1"
                entries.append(f"{key}:{value}")
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                has_structured_value = has_structured_value or isinstance(parsed_value, (list, dict))
                json_entries[key] = parsed_value
        if has_structured_value:
            return json.dumps(json_entries, separators=(",", ":"))
        return ",".join(entries)

    def setText(self, text: str) -> None:  # noqa: N802 - compatibility
        self.set_text(text)


class CompositionPathTableInput(QWidget):
    """Element/start/end table for a one-dimensional composition path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(
            [self.tr("Element"), self.tr("Start fraction"), self.tr("End fraction")]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(105)
        self.table.setMaximumHeight(165)
        buttons = QHBoxLayout()
        self.add_button = PushButton(self.tr("Add element"), self)
        self.remove_button = PushButton(self.tr("Remove selected"), self)
        self.add_button.clicked.connect(lambda: self.add_row())
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    @staticmethod
    def _mapping(text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for token in str(text or "").replace(";", ",").split(","):
            if ":" in token:
                key, value = token.split(":", 1)
                if key.strip():
                    result[key.strip()] = value.strip()
        return result

    def add_row(self, element: str = "", start: str = "0", end: str = "0") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((element, start, end)):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)

    def set_values(self, elements: str, start: str, end: str) -> None:
        self.table.setRowCount(0)
        start_map = self._mapping(start)
        end_map = self._mapping(end)
        names = [item.strip() for item in str(elements or "").split(",") if item.strip()]
        for name in names:
            self.add_row(name, start_map.get(name, "0"), end_map.get(name, "0"))

    def values(self) -> tuple[str, str, str]:
        names: list[str] = []
        start: list[str] = []
        end: list[str] = []
        for row in range(self.table.rowCount()):
            items = [self.table.item(row, column) for column in range(3)]
            name = items[0].text().strip() if items[0] else ""
            if not name:
                continue
            start_value = items[1].text().strip() if items[1] else "0"
            end_value = items[2].text().strip() if items[2] else "0"
            names.append(name)
            start.append(f"{name}:{start_value or '0'}")
            end.append(f"{name}:{end_value or '0'}")
        return ",".join(names), ",".join(start), ",".join(end)
