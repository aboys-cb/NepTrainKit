#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/2 19:58
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QCompleter
from qfluentwidgets import SearchLineEdit, CheckBox, ToolTipFilter, ToolTipPosition
from qfluentwidgets.components.widgets.line_edit import CompleterMenu, LineEditButton

from .completer import CompleterModel, JoinDelegate
from ..core.types import SearchType


class ConfigTypeSearchLineEdit(SearchLineEdit):
    searchSignal = Signal(str,SearchType)

    checkSignal=Signal(str,SearchType)
    uncheckSignal=Signal(str,SearchType)
    typeChangeSignal=Signal(str )
    def __init__(self, parent):
        super().__init__(parent)
        self.init()
        self.search_type:SearchType = SearchType.TAG
    def init(self):



        self.searchButton.setToolTip("Searching for structures based on Config_type")
        self.searchButton.installEventFilter(ToolTipFilter(self.searchButton, 300, ToolTipPosition.TOP))

        self.checkButton = LineEditButton(":/images/src/images/check.svg", self)
        self.checkButton.installEventFilter(ToolTipFilter(self.checkButton, 300, ToolTipPosition.TOP))

        self.checkButton.setToolTip("Mark structure according to Config_type")
        self.uncheckButton = LineEditButton(":/images/src/images/uncheck.svg", self)
        self.uncheckButton.setToolTip("Unmark structure according to Config_type")
        self.uncheckButton.installEventFilter(ToolTipFilter(self.uncheckButton, 300, ToolTipPosition.TOP))

        self.searchButton.setIconSize(QSize(16, 16))

        self.checkButton.setIconSize(QSize(16, 16))
        self.uncheckButton.setIconSize(QSize(16, 16))

        self.hBoxLayout.addWidget(self.checkButton, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addWidget(self.uncheckButton, 0, Qt.AlignmentFlag.AlignRight)


        self.checkButton.clicked.connect(self._checked)
        self.uncheckButton.clicked.connect(self._unchecked)


        self.setObjectName("search_lineEdit")
        self.set_search_type(SearchType.TAG)
        stands = []
        self.completer_model = CompleterModel(stands)

        completer = QCompleter( self.completer_model , self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # completer.setMaxVisibleItems(10)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(completer)
        _completerMenu=CompleterMenu(self)
        self.setCompleterMenu(_completerMenu)
        self._delegate =JoinDelegate(self,{})
        _completerMenu.view.setItemDelegate(self._delegate)
        _completerMenu.view.setMaxVisibleItems(10)
    def search(self):
        """ emit search signal """
        text = self.text().strip()
        if text:
            self.searchSignal.emit(text, self.search_type)
        else:
            self.clearSignal.emit()


    def set_search_type(self, search_type:SearchType):
        self.search_type = search_type
        self.setPlaceholderText(f"Mark structure according to {search_type}")
        self.typeChangeSignal.emit(search_type)

    def _checked(self):
        self.checkSignal.emit(self.text(), self.search_type)

    def _unchecked(self):
        self.uncheckSignal.emit(self.text(), self.search_type)



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


