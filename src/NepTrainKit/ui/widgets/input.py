"""Input widgets that pair spin boxes with unit labels."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QSpinBox, QDoubleSpinBox
from qfluentwidgets import BodyLabel


class SpinBoxUnitInputFrame(QFrame):
    """Composite input frame with spin boxes followed by unit labels."""

    def __init__(self, parent=None):
        """Create the layout and track added input widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(SpinBoxUnitInputFrame, self).__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.object_list: list[QSpinBox | QDoubleSpinBox] = []

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
                input_object = QSpinBox(self)
                input_object.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            elif input_type[i % len(unit_str)] == "float":
                input_object = QDoubleSpinBox(self)
                input_object.setDecimals(3)
                input_object.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            else:
                raise TypeError("input_type must be int or float")

            input_object.setFixedHeight(25)
            self._layout.addWidget(input_object)
            self._layout.addWidget(BodyLabel(unit_str[i % len(unit_str)], self))
            self.object_list.append(input_object)

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

