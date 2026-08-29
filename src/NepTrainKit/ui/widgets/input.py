"""Input widgets that pair spin boxes with unit labels."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QSpinBox,
)
from qfluentwidgets import (
    CaptionLabel,
    CompactDoubleSpinBox,
    CompactSpinBox,
    DoubleSpinBox,
)


class AdaptiveCompactSpinBox(CompactSpinBox):
    """Keep integer text and its step control visible at the normal hint."""

    _TEXT_PADDING = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineEdit().textChanged.connect(self._update_symbol_visibility)

    def _update_symbol_visibility(self, *_args) -> None:
        button = getattr(self, "compactSpinButton", None)
        if button is None:
            return
        # Integer controls are used for short counts and indices. Keep their
        # primary step affordance available at every editable width; the line
        # edit can scroll a long value without hiding the button.
        show_symbol = not self.isReadOnly()
        if button.isHidden() == show_symbol:
            self.setSymbolVisible(show_symbol)
        if not show_symbol:
            self.lineEdit().setGeometry(
                6,
                1,
                max(0, self.width() - 12),
                max(0, self.height() - 2),
            )

    def readable_width_hint(self) -> int:
        """Return the width needed to show the value and step control."""
        return (
            self.fontMetrics().horizontalAdvance(self.text())
            + self._step_button_width()
            + self._TEXT_PADDING
        )

    def _step_button_width(self) -> int:
        """Return one stable reserve shared by layout hints and visibility."""
        button = getattr(self, "compactSpinButton", None)
        if button is None:
            return 0
        return max(30, button.sizeHint().width(), button.minimumSizeHint().width())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_symbol_visibility()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt override
        QSpinBox.focusInEvent(self, event)


class AdaptiveInlineDoubleSpinBox(DoubleSpinBox):
    """Inline double spin box that yields its two buttons before clipping text."""

    _TEXT_PADDING = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineEdit().textChanged.connect(self._update_symbol_visibility)

    def _update_symbol_visibility(self, *_args) -> None:
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        buttons_width = self.upButton.width() + self.downButton.width() + 5
        show_symbols = (
            not self.isReadOnly()
            and self.width() >= text_width + buttons_width + self._TEXT_PADDING
        )
        if self.upButton.isHidden() == show_symbols:
            self.setSymbolVisible(show_symbols)
        if not show_symbols:
            self.lineEdit().setGeometry(
                6,
                1,
                max(0, self.width() - 12),
                max(0, self.height() - 2),
            )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_symbol_visibility()


class AdaptiveCompactDoubleSpinBox(CompactDoubleSpinBox):
    """Keep numeric text and its step control visible at the normal hint."""

    _TEXT_PADDING = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shared_text_width = 0
        self.lineEdit().textChanged.connect(self._update_symbol_visibility)

    def set_shared_text_width(self, width: int) -> None:
        self._shared_text_width = max(0, int(width))
        self._update_symbol_visibility()

    def _update_symbol_visibility(self, *_args) -> None:
        button = getattr(self, "compactSpinButton", None)
        if button is None:
            return
        # Match the integer control: editable compact fields keep their
        # primary step affordance on every platform. Long values can scroll
        # inside the line edit without making the control undiscoverable.
        show_symbol = not self.isReadOnly()
        if button.isHidden() == show_symbol:
            self.setSymbolVisible(show_symbol)
        if not show_symbol:
            self.lineEdit().setGeometry(
                6,
                1,
                max(0, self.width() - 12),
                max(0, self.height() - 2),
            )

    def readable_width_hint(self) -> int:
        """Return the width needed for the widest value and step control."""
        text_width = max(
            self._shared_text_width,
            self.fontMetrics().horizontalAdvance(self.text()),
        )
        return text_width + self._step_button_width() + self._TEXT_PADDING

    def _step_button_width(self) -> int:
        """Return one stable reserve shared by layout hints and visibility."""
        button = getattr(self, "compactSpinButton", None)
        if button is None:
            return 0
        return max(30, button.sizeHint().width(), button.minimumSizeHint().width())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_symbol_visibility()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Opening a step flyout merely from keyboard focus is disruptive in a
        # dense form and can leave transient native windows alive during card
        # teardown. The compact button remains the explicit pointer affordance;
        # keyboard arrows and the wheel continue to adjust the value directly.
        QDoubleSpinBox.focusInEvent(self, event)


class SpinBoxUnitInputFrame(QFrame):
    """Composite input frame with spin boxes followed by unit labels.

    Uses the Fluent-styled `SpinBox`/`DoubleSpinBox` (rounded, inline
    up/down controls, focus ring) rather than bare `QSpinBox`/`QDoubleSpinBox`
    so every numeric field across the app shares one polished look, while
    keeping this class's public API (`set_input`, `get_input_value`,
    `set_input_value`, `setRange`, `setDecimals`, `setSingleStep`,
    `object_list`) unchanged for existing callers.
    """

    def __init__(self, parent=None):
        """Create the layout and track added input widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(SpinBoxUnitInputFrame, self).__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(4)
        self._layout.setVerticalSpacing(4)
        self.object_list: list[QSpinBox | QDoubleSpinBox] = []
        self._unit_labels: list[CaptionLabel] = []
        self._column_count = 1
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_input(self, unit_str, object_num, input_type="int"):
        """Populate the frame with spin boxes and unit labels.

        Parameters
        ----------
        unit_str : str or list[str]
            Unit string applied to each input or list of per-input units.
        object_num : int
            Number of inputs to create.
            input_type : str or list[str], optional
            Either a single type of "int"/"float" or a list specifying the
            type for each input in sequence.

        Raises
        ------
        TypeError
            Raised when `unit_str` or `input_type` is not a string or list.
        """
        if isinstance(unit_str, str):
            unit_str = [unit_str] * object_num
        elif isinstance(unit_str, list):
            unit_str = unit_str
        else:
            raise TypeError("unit_str must be str or list")

        if isinstance(input_type, str):
            input_type = [input_type] * object_num
        elif isinstance(input_type, list):
            input_type = input_type
        else:
            raise TypeError("input_type must be str or list")

        for i in range(object_num):
            if input_type[i % len(unit_str)] == "int":
                input_object = AdaptiveCompactSpinBox(self)
            elif input_type[i % len(unit_str)] == "float":
                input_object = AdaptiveCompactDoubleSpinBox(self)
                input_object.setDecimals(3)
            else:
                raise TypeError("input_type must be int or float")

            input_object.setFixedHeight(30)
            input_object.setMinimumWidth(0)
            input_object.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
            unit_label = CaptionLabel(unit_str[i % len(unit_str)], self)
            unit_label.setStyleSheet("color:#8a95a0;")
            unit_label.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            unit_label.setVisible(bool(unit_label.text()))
            self.object_list.append(input_object)
            self._unit_labels.append(unit_label)
        for input_object in self.object_list:
            input_object.lineEdit().textChanged.connect(
                self._sync_float_symbol_visibility
            )
        self._sync_float_symbol_visibility()
        self._reflow_inputs(self.width())

    def _sync_float_symbol_visibility(self, *_args) -> None:
        floats = [
            item
            for item in self.object_list
            if isinstance(item, AdaptiveCompactDoubleSpinBox)
        ]
        if floats:
            shared_width = max(
                item.fontMetrics().horizontalAdvance(item.text()) for item in floats
            )
            for item in floats:
                item.set_shared_text_width(shared_width)
        self._reflow_inputs(self.width())

    def _reflow_inputs(self, width: int) -> None:
        """Keep every value on one row and divide the available width evenly."""
        count = len(self.object_list)
        if not count:
            return
        if self._column_count == count and self._layout.count() == count * 2:
            return
        while self._layout.count():
            self._layout.takeAt(0)
        for column in range(count * 2):
            self._layout.setColumnStretch(column, 0)
            self._layout.setColumnMinimumWidth(column, 0)
        for index, (input_object, unit_label) in enumerate(
            zip(self.object_list, self._unit_labels)
        ):
            input_column = index * 2
            self._layout.addWidget(input_object, 0, input_column)
            self._layout.addWidget(unit_label, 0, input_column + 1)
            self._layout.setColumnStretch(input_column, 1)
        self._column_count = count
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._sync_float_symbol_visibility()

    def setRange(self, min_value, max_value):
        """Apply the same range constraints to every spin box.

        Parameters
        ----------
        min_value : int | float
            Minimum allowable value for the inputs.
        max_value : int | float
            Maximum allowable value for the inputs.
        """
        for input_object in self.object_list:
            input_object.setRange(min_value, max_value)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Prefer enough width for each value while still allowing compression."""
        if not self.object_list:
            return super().sizeHint()
        widths = [control.readable_width_hint() for control in self.object_list]
        widths.extend(
            label.sizeHint().width() for label in self._unit_labels if label.text()
        )
        visible_items = len(self.object_list) + sum(
            bool(label.text()) for label in self._unit_labels
        )
        spacing = self._layout.horizontalSpacing() * max(0, visible_items - 1)
        return QSize(sum(widths) + spacing, 30)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Let narrow inspectors compress the equal-width row without overflow."""
        if len(self.object_list) == 1:
            return QSize(self.sizeHint().width(), 30)
        return QSize(0, 30)

    def setDecimals(self, decimals: int):
        """Set the number of decimals for every double spin box.

        Parameters
        ----------
        decimals : int
            Number of decimal places.
        """
        for input_object in self.object_list:
            if isinstance(input_object, QDoubleSpinBox):
                input_object.setDecimals(decimals)

    def setSingleStep(self, step: int | float):
        """Set keyboard/wheel increment for every spin box."""
        for input_object in self.object_list:
            input_object.setSingleStep(step)

    def get_input_value(self) -> list[int | float]:
        """Return the numeric values from each input widget.

        Returns
        -------
        list[int | float]
            Values retrieved from the spin boxes in order.
        """
        return [input_object.value() for input_object in self.object_list]

    def set_input_value(self, value_list):
        """Populate the spin boxes with supplied values.

        Parameters
        ----------
        value_list : int or float or list[int | float]
            Single value or list of values applied to the inputs.
        """
        if not isinstance(value_list, list):
            value_list = [value_list] * len(self.object_list)

        for i, input_object in enumerate(self.object_list):
            input_object.setValue(value_list[i])


class RangeTripletInputFrame(QFrame):
    """Compact minimum/maximum/step editor that remains legible in a side panel."""

    def __init__(self, parent=None, suffix: str = "%"):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)
        self.object_list: list[QDoubleSpinBox] = []
        titles = (
            self.tr("Min (%)") if suffix == "%" else self.tr("Minimum"),
            self.tr("Max (%)") if suffix == "%" else self.tr("Maximum"),
            self.tr("Step (%)") if suffix == "%" else self.tr("Step"),
        )
        for column, title in enumerate(titles):
            label = CaptionLabel(title, self)
            label.setStyleSheet("color:#8a95a0;")
            # The ordinary Fluent double spin box reserves two 31 px buttons.
            # A range triplet cannot afford that; the compact variant keeps a
            # single 26 px step affordance and leaves the value readable.
            spin = AdaptiveCompactDoubleSpinBox(self)
            spin.setDecimals(3)
            spin.setFixedHeight(30)
            spin.setMinimumWidth(0)
            # Fluent spin boxes have a generous sizeHint intended for ordinary
            # forms. Ignore it horizontally so three values fit the inspector.
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            layout.addWidget(label, 0, column)
            layout.addWidget(spin, 1, column)
            layout.setColumnStretch(column, 1)
            self.object_list.append(spin)
            spin.lineEdit().textChanged.connect(self._sync_symbol_visibility)
        self._sync_symbol_visibility()

    def _sync_symbol_visibility(self, *_args) -> None:
        if not self.object_list:
            return
        shared_width = max(
            item.fontMetrics().horizontalAdvance(item.text())
            for item in self.object_list
        )
        for item in self.object_list:
            item.set_shared_text_width(shared_width)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._sync_symbol_visibility()

    def setRange(self, min_value, max_value):
        for input_object in self.object_list:
            input_object.setRange(min_value, max_value)

    def setDecimals(self, decimals: int):
        for input_object in self.object_list:
            input_object.setDecimals(decimals)

    def setSingleStep(self, step: int | float):
        for input_object in self.object_list:
            input_object.setSingleStep(step)

    def get_input_value(self) -> list[float]:
        return [float(input_object.value()) for input_object in self.object_list]

    def set_input_value(self, value_list):
        if not isinstance(value_list, list):
            value_list = [value_list] * len(self.object_list)
        for index, input_object in enumerate(self.object_list):
            input_object.setValue(value_list[index])
