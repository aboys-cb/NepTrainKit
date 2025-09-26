"""Setting card widgets that expose common input controls."""

from typing import Union

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, Qt, QColor
from PySide6.QtWidgets import QPushButton, QColorDialog
from qfluentwidgets import OptionsConfigItem, FluentIconBase, ComboBox, SettingCard, DoubleSpinBox, LineEdit


class MyComboBoxSettingCard(SettingCard):
    """Setting card that binds a combo box to an options config item."""

    optionChanged = Signal(str)

    def __init__(
        self,
        configItem: OptionsConfigItem,
        icon: Union[str, QIcon, FluentIconBase],
        title,
        content,
        texts: list[str],
        default=None,
        parent=None,
    ):
        """Create a combo box linked to a configuration item.

        Parameters
        ----------
        configItem : OptionsConfigItem
            Configuration item whose value mirrors the combo box selection.
        icon : str or QIcon or FluentIconBase
            Icon displayed in the card header.
        title : str
            Title placed on the card.
        content : str
            Description displayed under the title.
        texts : list[str]
            Text labels shown for each available option.
        default : str, optional
            Text of the option selected by default.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.comboBox = ComboBox(self)
        self.hBoxLayout.addWidget(self.comboBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

        self.optionToText = {o: t for o, t in zip(configItem.options, texts)}
        for text, option in zip(texts, configItem.options):
            self.comboBox.addItem(text, userData=option)
        if default is not None:
            self.comboBox.setCurrentText(default)
        self.comboBox.currentTextChanged.connect(self.optionChanged)
        configItem.valueChanged.connect(self.setValue)

    def setValue(self, value):
        """Synchronize the combo box with the provided option value.

        Parameters
        ----------
        value : str
            Option key stored in the configuration item.
        """
        if value not in self.optionToText:
            return

        self.comboBox.setCurrentText(self.optionToText[value])


class DoubleSpinBoxSettingCard(SettingCard):
    """Setting card that exposes a numeric input via `DoubleSpinBox`."""

    valueChanged = Signal(float)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        """Create the double spin box and hook up change notifications.

        Parameters
        ----------
        icon : str or QIcon or FluentIconBase
            Icon displayed in the card header.
        title : str
            Title placed on the card.
        content : str, optional
            Description displayed under the title.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(icon, title, content, parent)
        self.doubleSpinBox = DoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setMinimumWidth(200)
        self.hBoxLayout.addWidget(self.doubleSpinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.doubleSpinBox.valueChanged.connect(self.valueChanged)

    def setValue(self, value):
        """Assign a double value to the spin box."""
        self.doubleSpinBox.setValue(value)

    def setRange(self, min_value, max_value):
        """Apply inclusive limits to the spin box.

        Parameters
        ----------
        min_value : float
            Minimum allowed value.
        max_value : float
            Maximum allowed value.
        """
        self.doubleSpinBox.setRange(min_value, max_value)


class ColorSettingCard(SettingCard):
    """Setting card that stores a color chosen from a dialog."""

    colorChanged = Signal(str)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title: str, content: str | None = None, parent=None):
        """Create the color button and bind change handlers.

        Parameters
        ----------
        icon : str or QIcon or FluentIconBase
            Icon displayed in the card header.
        title : str
            Title placed on the card.
        content : str, optional
            Description displayed under the title.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(icon, title, content, parent)
        self._color = QColor("#000000")
        self.button = QPushButton(self)
        self.button.setFixedSize(64, 24)
        self.button.clicked.connect(self._choose_color)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self._apply_button_style()

    def _apply_button_style(self):
        """Update the button stylesheet to reflect the current color."""
        self.button.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; border: 1px solid #999; border-radius: 4px; }}"
        )

    def _choose_color(self):
        """Open the color dialog and emit a change signal by hex code."""
        col = QColorDialog.getColor(self._color, self)
        if col.isValid():
            self._color = col
            self._apply_button_style()
            self.colorChanged.emit(self._color.name())

    def setValue(self, value: str):
        """Update the stored color and refresh the button background.

        Parameters
        ----------
        value : str
            Color expressed as a hex string.
        """
        try:
            col = QColor(value)
            if col.isValid():
                self._color = col
                self._apply_button_style()
        except Exception:
            pass


class LineEditSettingCard(SettingCard):
    """Setting card that exposes a single-line text field."""

    textChanged = Signal(str)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title: str, content: str | None = None, parent=None):
        """Create the line edit and bind change notifications.

        Parameters
        ----------
        icon : str or QIcon or FluentIconBase
            Icon displayed in the card header.
        title : str
            Title placed on the card.
        content : str, optional
            Description displayed under the title.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(icon, title, content, parent)
        self.lineEdit = LineEdit(self)
        self.lineEdit.setMinimumWidth(220)
        self.hBoxLayout.addWidget(self.lineEdit, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.lineEdit.textChanged.connect(self.textChanged)

    def setValue(self, text: str):
        """Set the line-edit content to the provided text."""
        self.lineEdit.setText(str(text) if text is not None else "")

    def value(self) -> str:
        """Return the current text stored in the line edit.

        Returns
        -------
        str
            Value currently displayed to the user.
        """
        return self.lineEdit.text()
