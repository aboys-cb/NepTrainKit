#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 12:57
# @Author  : 兵
# @email    : 1747193328@qq.com
from typing import Union

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, Qt
from qfluentwidgets import OptionsConfigItem, FluentIconBase, ComboBox, SettingCard, DoubleSpinBox, LineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog


class MyComboBoxSettingCard(SettingCard):
    """ Setting card with a combo box """
    optionChanged = Signal(str)

    def __init__(self, configItem: OptionsConfigItem,
                 icon: Union[str, QIcon, FluentIconBase],
                 title, content , texts:list[str], default=None,
                 parent=None):
        """
        Parameters
        ----------
        configItem: OptionsConfigItem
            configuration item operated by the card

        icon: str | QIcon | FluentIconBase
            the icon to be drawn

        title: str
            the title of card

        content: str
            the content of card

        texts: List[str]
            the text of items

        parent: QWidget
            parent widget
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
        if value not in self.optionToText:
            return

        self.comboBox.setCurrentText(self.optionToText[value])





class DoubleSpinBoxSettingCard(SettingCard):
    """ Setting card with a push button """

    valueChanged = Signal(float)

    def __init__(self,   icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        """
        Parameters
        ----------
        text: str
            the text of push button

        icon: str | QIcon | FluentIconBase
            the icon to be drawn

        title: str
            the title of card

        content: str
            the content of card

        parent: QWidget
            parent widget
        """
        super().__init__(icon, title, content, parent)
        self.doubleSpinBox = DoubleSpinBox(  self)
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setMinimumWidth(200)
        self.hBoxLayout.addWidget(self.doubleSpinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.doubleSpinBox.valueChanged.connect(self.valueChanged)
    def setValue(self, value):
        self.doubleSpinBox.setValue(value)
    def setRange(self, min, max):
        self.doubleSpinBox.setRange(min, max)


class ColorSettingCard(SettingCard):
    """Setting card with a color picker button"""

    colorChanged = Signal(str)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title: str, content: str | None = None,
                 parent=None):
        super().__init__(icon, title, content, parent)
        self._color = QColor("#000000")
        self.button = QPushButton(self)
        self.button.setFixedSize(64, 24)
        self.button.clicked.connect(self._choose_color)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self._apply_button_style()

    def _apply_button_style(self):
        # show current color as button background
        self.button.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; border: 1px solid #999; border-radius: 4px; }}"
        )

    def _choose_color(self):
        col = QColorDialog.getColor(self._color, self)
        if col.isValid():
            self._color = col
            self._apply_button_style()
            # emit as hex RGB string
            self.colorChanged.emit(self._color.name())

    def setValue(self, value: str):
        try:
            col = QColor(value)
            if col.isValid():
                self._color = col
                self._apply_button_style()
        except Exception:
            pass


class LineEditSettingCard(SettingCard):
    """Setting card with a single-line text input"""

    textChanged = Signal(str)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title: str, content: str | None = None,
                 parent=None):
        super().__init__(icon, title, content, parent)
        self.lineEdit = LineEdit(self)
        self.lineEdit.setMinimumWidth(220)
        self.hBoxLayout.addWidget(self.lineEdit, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.lineEdit.textChanged.connect(self.textChanged)

    def setValue(self, text: str):
        self.lineEdit.setText(str(text) if text is not None else "")

    def value(self) -> str:
        return self.lineEdit.text()
