#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/1/7 23:23
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QIcon, QDrag, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout
from qfluentwidgets import HeaderCardWidget, CheckBox, TransparentToolButton

from NepTrainKit.core import MessageManager

from qfluentwidgets import FluentIcon as FIF
from NepTrainKit.core.custom_widget import ProcessLabel
from NepTrainKit import utils



class CheckableHeaderCardWidget(HeaderCardWidget):

    def __init__(self, parent=None):
        super(CheckableHeaderCardWidget, self).__init__(parent)
        self.state_checkbox=CheckBox()
        self.state_checkbox.setChecked(True)
        self.state_checkbox.stateChanged.connect(self.state_changed)


        self.headerLayout.insertWidget(0, self.state_checkbox, 0,Qt.AlignmentFlag.AlignLeft)


        self.headerLayout.setStretch(1, 3)
        self.headerLayout.setContentsMargins(10, 0, 3, 0)
        self.headerLayout.setSpacing(3)
        self.viewLayout.setContentsMargins(6, 0, 6, 0)
        self.headerLayout.setAlignment(self.headerLabel, Qt.AlignmentFlag.AlignLeft)
        self.check_state=True
    def state_changed(self, state):
        if state == 2:
            self.check_state = True
        else:
            self.check_state = False


class ShareCheckableHeaderCardWidget(CheckableHeaderCardWidget):
    exportSignal=Signal()
    def __init__(self, parent=None):
        super(ShareCheckableHeaderCardWidget, self).__init__(parent)
        self.export_button=TransparentToolButton(QIcon(":/images/src/images/export1.svg"),self)
        self.export_button.clicked.connect(self.exportSignal)

        self.close_button=TransparentToolButton(FIF.CLOSE,self)
        self.close_button.clicked.connect(self.close)


        self.headerLayout.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)

class MakeDataCardWidget(ShareCheckableHeaderCardWidget):
    windowStateChangedSignal=Signal( )
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.window_state="expand"
        self.collapse_button=TransparentToolButton(QIcon(":/images/src/images/collapse.svg"),self)
        self.collapse_button.clicked.connect(self.collapse)
        self.headerLayout.insertWidget(0, self.collapse_button, 0,Qt.AlignmentFlag.AlignLeft)
        self.windowStateChangedSignal.connect(self.update_window_state)

    def mouseMoveEvent(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        drag = QDrag(self)
        mime = QMimeData()
        drag.setMimeData(mime)

        # 显示拖拽时的控件预览
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())

        drag.exec(Qt.MoveAction)
    def collapse(self):

        if self.window_state == "collapse":
            self.window_state = "expand"
        else:

            self.window_state = "collapse"

        self.windowStateChangedSignal.emit( )
    def update_window_state(self):
        if self.window_state == "expand":
            self.collapse_button.setIcon(QIcon(":/images/src/images/collapse.svg"))
        else:
            self.collapse_button.setIcon(QIcon(":/images/src/images/expand.svg"))

    def from_dict(self, data):
        self.state_checkbox.setChecked(data['check_state'])

    def to_dict(self):

        return {
            'class': self.__class__.__name__,
            # 'name': self.card_name,
            'check_state': self.check_state,

        }

