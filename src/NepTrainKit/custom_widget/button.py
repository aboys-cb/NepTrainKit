from typing import Union

from PySide6.QtCore import QRectF, Qt, QSize, Signal
from PySide6.QtGui import QPainter, QIcon, QPainterPath, QColor
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy
from qfluentwidgets import (PushButton, TransparentPushButton, FluentIconBase,
                            FlowLayout, TransparentToolButton, FluentIcon, TransparentTogglePushButton)

from qfluentwidgets.common.overload import singledispatchmethod



class CloseWidgetBase(QWidget):
    """ Split widget base class """

    closeClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.isHover=False
        self.isPressed=False
        self.closeButton = TransparentToolButton( self)
        self.closeButton.setIcon(FluentIcon.CLOSE)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.closeButton)

        self.closeButton.clicked.connect(self.closeClicked)
        self.borderRadius=3
        self.backgroundColor="#FFFFFF"
        self.setBackgroundColor(self.backgroundColor)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


    def setWidget(self, widget: QWidget):
        """ set the widget on left side """
        self.hBoxLayout.insertWidget(0, widget, 1, Qt.AlignLeft)

    def setDropButton(self, button):
        """ set drop dow button """
        self.hBoxLayout.removeWidget(self.closeButton)
        self.closeButton.deleteLater()

        self.closeButton = button
        self.closeButton.clicked.connect(self.closeClicked)

        self.hBoxLayout.addWidget(button)
    def setBackgroundColor(self,color):
        self.backgroundColor= color
        self.update()
    def setWidget(self, widget: QWidget):
        """ set the widget on left side """
        self.hBoxLayout.insertWidget(0, widget, 1, Qt.AlignLeft)
 
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = self.borderRadius
        d = 2 * r

        isDark = False

        # draw top border
        path = QPainterPath()
        # path.moveTo(1, h - r)
        path.arcMoveTo(1, h - d - 1, d, d, 240)
        path.arcTo(1, h - d - 1, d, d, 225, -60)
        path.lineTo(1, r)
        path.arcTo(1, 1, d, d, -180, -90)
        path.lineTo(w - r, 1)
        path.arcTo(w - d - 1, 1, d, d, 90, -90)
        path.lineTo(w - 1, h - r)
        path.arcTo(w - d - 1, h - d - 1, d, d, 0, -60)

        topBorderColor = QColor(0, 0, 0, 20)
        if isDark:
            if self.isPressed:
                topBorderColor = QColor(255, 255, 255, 18)
            elif self.isHover:
                topBorderColor = QColor(255, 255, 255, 13)
        else:
            topBorderColor = QColor(0, 0, 0, 15)

        painter.strokePath(path, topBorderColor)

        # draw bottom border
        path = QPainterPath()
        path.arcMoveTo(1, h - d - 1, d, d, 240)
        path.arcTo(1, h - d - 1, d, d, 240, 30)
        path.lineTo(w - r - 1, h - 1)
        path.arcTo(w - d - 1, h - d - 1, d, d, 270, 30)

        bottomBorderColor = topBorderColor
        if not isDark and self.isHover and not self.isPressed:
            bottomBorderColor = QColor(0, 0, 0, 27)

        painter.strokePath(path, bottomBorderColor)

        # draw background
        painter.setPen(Qt.NoPen)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(self.backgroundColor)
        painter.drawRoundedRect(rect, r, r)
 

class TagPushButton(CloseWidgetBase):
    checkedSignal = Signal(str)
    @singledispatchmethod
    def __init__(self, parent: QWidget = None,checkable: bool = False):

        super(TagPushButton, self).__init__(parent)
        if checkable:

            self.button = TransparentTogglePushButton(FluentIcon.TAG,"", self)
            self.button.toggled.connect(lambda :self.checkedSignal.emit(self.button.text()))
        else:
            self.button = TransparentPushButton(FluentIcon.TAG, "", self)
        self.button.setObjectName('PushButton')

        self.setWidget(self.button)


    @__init__.register
    def _(self, text: str, parent: QWidget = None, icon: Union[QIcon, str, FluentIconBase] = None, checkable: bool = False):
        self.__init__(parent,checkable)
        self.setText(text)

        self.setIcon(icon)

    @__init__.register
    def _(self, icon: QIcon, text: str, checkable: bool = False, parent: QWidget = None):
        self.__init__(text, parent, icon,checkable)

    @__init__.register
    def _(self, icon: FluentIconBase, text: str, checkable: bool = False, parent: QWidget = None):
        self.__init__(text, parent, icon,checkable)
    def text(self):
        return self.button.text()

    def setText(self, text: str):
        self.button.setText(text)
        self.adjustSize()

    def icon(self):
        return self.button.icon()

    def setIcon(self, icon: Union[QIcon, FluentIconBase, str]):
        self.button.setIcon(icon)

    def setIconSize(self, size: QSize):
        self.button.setIconSize(size)

class TagGroup(QWidget):
    tagRemovedSignal = Signal(str)
    tagCheckedSignal = Signal(str)
    def __init__(self,tags=None, parent: QWidget = None):
        super(TagGroup, self).__init__(parent)
        self._layout = FlowLayout(self, needAni=True)
        self.tags={}

        if tags is not None:
            for tag in tags:
                self.add_tag(tag)

    def has_tag(self, tag):
        return tag in self.tags

    def add_tag(self, tag,color=None,icon=FluentIcon.TAG,checkable=False) ->TagPushButton:


        button = TagPushButton(icon,tag,checkable, self)
        button.checkedSignal.connect(self.tagCheckedSignal)
        button.closeClicked.connect(lambda _tag=tag:self.del_tag(_tag))
        if color is not None:
            button.setBackgroundColor(color)
        self._layout.addWidget(button)
        self.tags[tag]=button
        return button

    def del_tag(self, tag):
        button = self.tags[tag]
        self._layout.removeWidget(button)
        button.deleteLater()
        self.tagRemovedSignal.emit(tag)
        del self.tags[tag]
        self._layout.update()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    from PySide6.QtWidgets import QFrame

    frame = QWidget()
    frame.resize(800, 600)
    layout = QVBoxLayout()
    frame.setLayout(layout)
    pushButton = TagPushButton(FluentIcon.PROJECTOR,"dwasd",frame )
    layout.addWidget(pushButton)

    frame.show()
    app.exec_()