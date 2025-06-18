#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/2 19:58
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QCompleter
from qfluentwidgets import SearchLineEdit
from qfluentwidgets.components.widgets.line_edit import CompleterMenu, LineEditButton

from .completer import CompleterModel, JoinDelegate


class ConfigTypeSearchLineEdit(SearchLineEdit):
    checkSignal=Signal(str)
    uncheckSignal=Signal(str)
    def __init__(self, parent):
        super().__init__(parent)
        self.init()

    def init(self):
        self.searchButton.setToolTip("Searching for structures based on Config_type")
        self.checkButton = LineEditButton(":/images/src/images/check.svg", self)
        self.checkButton.setToolTip("Mark structure according to Config_type")
        self.uncheckButton = LineEditButton(":/images/src/images/uncheck.svg", self)
        self.uncheckButton.setToolTip("Unmark structure according to Config_type")

        self.searchButton.setIconSize(QSize(16, 16))

        self.checkButton.setIconSize(QSize(16, 16))
        self.uncheckButton.setIconSize(QSize(16, 16))

        self.hBoxLayout.addWidget(self.checkButton, 0, Qt.AlignRight)
        self.hBoxLayout.addWidget(self.uncheckButton, 0, Qt.AlignRight)


        self.checkButton.clicked.connect(self._checked)
        self.uncheckButton.clicked.connect(self._unchecked)


        self.setObjectName("search_lineEdit")
        self.setPlaceholderText("Click to view Config_type")
        stands = []
        self.completer_model = CompleterModel(stands)


        completer = QCompleter( self.completer_model , self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        # completer.setMaxVisibleItems(10)
        completer.setFilterMode(Qt.MatchContains)
        self.setCompleter(completer)
        _completerMenu=CompleterMenu(self)
        self.setCompleterMenu(_completerMenu)
        self._delegate =JoinDelegate(self,{})
        _completerMenu.view.setItemDelegate(self._delegate)
        _completerMenu.view.setMaxVisibleItems(10)







    def _checked(self):
        self.checkSignal.emit(self.text())

    def _unchecked(self):
        self.uncheckSignal.emit(self.text())



    def mousePressEvent(self,event):

        self._completer.setCompletionPrefix(self.text())
        self._completerMenu.setCompletion(self._completer.completionModel())
        self._completerMenu.popup()
        super().mousePressEvent(event)


    def setCompleterKeyWord(self, new_words):
        if isinstance(new_words, list):
            new_words = self.completer_model.parser_list(new_words)
        self._delegate.data=new_words
        self.completer_model.set_data(new_words)


