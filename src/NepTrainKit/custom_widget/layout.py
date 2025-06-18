#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/5 22:17
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import QRect, Qt, QSize, QPoint
from PySide6.QtWidgets import QLayout



class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=10):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None
    def insertWidget(self, index, widget):
        self.addWidget(widget)
        self.moveItem(len(self.itemList)-1,index)

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.sizeHint())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                      margins.top() + margins.bottom())
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            nextX = x + item.sizeHint().width() + self.spacing()
            if nextX - self.spacing() > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + self.spacing()
                nextX = x + item.sizeHint().width() + self.spacing()
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

    def findItemAt(self, pos):
        for i, item in enumerate(self.itemList):
            if item.geometry().contains(pos):
                return i, item
        return -1, None

    def findWidgetAt(self, widget):
        for i, item in enumerate(self.itemList):
            if item.widget() is widget:
                return i, item
        return -1, None

    def moveItem(self, from_index, to_index):
        if 0 <= from_index < len(self.itemList) and 0 <= to_index < len(self.itemList):
            item = self.itemList.pop(from_index)
            self.itemList.insert(to_index, item)
            self.update()