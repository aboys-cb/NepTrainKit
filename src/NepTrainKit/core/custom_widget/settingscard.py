#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 12:57
# @Author  : 兵
# @email    : 1747193328@qq.com
from typing import Union

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, Qt
from qfluentwidgets import SettingCard, OptionsConfigItem, FluentIconBase, ComboBox


class MyComboBoxSettingCard(SettingCard):
    """ Setting card with a combo box """
    optionChanged = Signal(str)

    def __init__(self, configItem: OptionsConfigItem,
                 icon: Union[str, QIcon, FluentIconBase],
                 title, content=None, texts=None, default=None,
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
        self.hBoxLayout.addWidget(self.comboBox, 0, Qt.AlignRight)
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