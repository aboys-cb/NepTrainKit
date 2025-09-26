"""Custom button widgets used for tagging and grouped controls."""

from typing import Union

from PySide6.QtCore import QRectF, Qt, QSize, Signal
from PySide6.QtGui import QPainter, QIcon, QPainterPath, QColor
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy
from qfluentwidgets import (
    PushButton,
    TransparentPushButton,
    FluentIconBase,
    FlowLayout,
    TransparentToolButton,
    FluentIcon,
    TransparentTogglePushButton,
)

from qfluentwidgets.common.overload import singledispatchmethod


class CloseWidgetBase(QWidget):
    """Widget container with an embedded close button for tag-like content.

    Signals
    -------
    closeClicked : Signal
        Emitted when the close button is activated.
    """

    closeClicked = Signal()

    def __init__(self, parent=None):
        """Build the container and place the close button.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(parent=parent)
        self.isHover = False
        self.isPressed = False
        self.closeButton = TransparentToolButton(self)
        self.closeButton.setIcon(FluentIcon.CLOSE)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.closeButton)

        self.closeButton.clicked.connect(self.closeClicked)
        self.borderRadius = 3
        self.backgroundColor = "#FFFFFF"
        self.setBackgroundColor(self.backgroundColor)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setWidget(self, widget: QWidget):
        """Add a widget to the left of the close button.

        Parameters
        ----------
        widget : QWidget
            Widget inserted before the close button.
        """
        self.hBoxLayout.insertWidget(0, widget, 1, Qt.AlignLeft)

    def setDropButton(self, button):
        """Replace the close button with a drop-down style control.

        Parameters
        ----------
        button : QWidget
            Button that replaces the default close button.
        """
        self.hBoxLayout.removeWidget(self.closeButton)
        self.closeButton.deleteLater()

        self.closeButton = button
        self.closeButton.clicked.connect(self.closeClicked)

        self.hBoxLayout.addWidget(button)

    def setBackgroundColor(self, color):
        """Update the background color for the container.

        Parameters
        ----------
        color : QColor or str
            Color applied to the widget background.
        """
        self.backgroundColor = color
        self.update()

    def paintEvent(self, e):
        """Render the border and background with rounded corners.

        Parameters
        ----------
        e : QPaintEvent
            Event carrying the region that requires repainting.
        """
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = self.borderRadius
        d = 2 * r

        isDark = False

        # draw top border
        path = QPainterPath()
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
    """Closable tag widget with optional toggle behavior."""

    checkedSignal = Signal(str)

    @singledispatchmethod
    def __init__(self, parent: QWidget = None, checkable: bool = False):
        """Create a tag button with optional toggle support.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        checkable : bool, optional
            Whether the button behaves as a toggle.
        """
        super(TagPushButton, self).__init__(parent)
        if checkable:
            self.button = TransparentTogglePushButton(FluentIcon.TAG, "", self)
            self.button.toggled.connect(lambda: self.checkedSignal.emit(self.button.text()))
        else:
            self.button = TransparentPushButton(FluentIcon.TAG, "", self)
        self.button.setObjectName("PushButton")

        self.setWidget(self.button)

    @__init__.register
    def _(self, text: str, parent: QWidget = None, icon: Union[QIcon, str, FluentIconBase] = None, checkable: bool = False):
        """Initialize the button with text before the icon variant.

        Parameters
        ----------
        text : str
            Tag label to display.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        icon : QIcon or str or FluentIconBase, optional
            Icon displayed alongside the text.
        checkable : bool, optional
            Whether the button behaves as a toggle.
        """
        self.__init__(parent, checkable)
        self.setText(text)
        self.setIcon(icon)

    @__init__.register
    def _(self, icon: QIcon, text: str, checkable: bool = False, parent: QWidget = None):
        """Initialize the button with a Qt icon instance.

        Parameters
        ----------
        icon : QIcon
            Icon displayed alongside the text.
        text : str
            Tag label to display.
        checkable : bool, optional
            Whether the button behaves as a toggle.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        self.__init__(text, parent, icon, checkable)

    @__init__.register
    def _(self, icon: FluentIconBase, text: str, checkable: bool = False, parent: QWidget = None):
        """Initialize the button with a Fluent icon descriptor.

        Parameters
        ----------
        icon : FluentIconBase
            Icon displayed alongside the text.
        text : str
            Tag label to display.
        checkable : bool, optional
            Whether the button behaves as a toggle.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        self.__init__(text, parent, icon, checkable)

    def text(self):
        """Return the current tag text.

        Returns
        -------
        str
            Label shown on the button.
        """
        return self.button.text()

    def setText(self, text: str):
        """Assign the label displayed by the tag.

        Parameters
        ----------
        text : str
            Label shown on the button.
        """
        self.button.setText(text)
        self.adjustSize()

    def icon(self):
        """Return the current button icon.

        Returns
        -------
        QIcon
            Icon displayed on the tag.
        """
        return self.button.icon()

    def setIcon(self, icon: Union[QIcon, FluentIconBase, str]):
        """Set the icon displayed alongside the tag text.

        Parameters
        ----------
        icon : QIcon or FluentIconBase or str
            Icon resource applied to the button.
        """
        self.button.setIcon(icon)

    def setIconSize(self, size: QSize):
        """Adjust the icon size for the button.

        Parameters
        ----------
        size : QSize
            Target size for the icon.
        """
        self.button.setIconSize(size)


class TagGroup(QWidget):
    """Layout that holds a collection of closable tag buttons."""

    tagRemovedSignal = Signal(str)
    tagCheckedSignal = Signal(str)

    def __init__(self, tags=None, parent: QWidget = None):
        """Create the tag group and optionally preload tags.

        Parameters
        ----------
        tags : Iterable[str], optional
            Initial set of tag labels to create.
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(TagGroup, self).__init__(parent)
        self._layout = FlowLayout(self, needAni=True)
        self.tags = {}

        if tags is not None:
            for tag in tags:
                self.add_tag(tag)

    def has_tag(self, tag):
        """Check whether the group already contains a tag.

        Parameters
        ----------
        tag : str
            Tag label to look up.

        Returns
        -------
        bool
            True if the tag is present, False otherwise.
        """
        return tag in self.tags

    def add_tag(self, tag, color=None, icon=FluentIcon.TAG, checkable=False) -> TagPushButton:
        """Add a tag button to the group.

        Parameters
        ----------
        tag : str
            Label displayed on the tag.
        color : QColor or str, optional
            Background color applied to the tag.
        icon : FluentIcon or QIcon, optional
            Icon displayed alongside the tag text.
        checkable : bool, optional
            Whether the tag behaves as a toggle.

        Returns
        -------
        TagPushButton
            The button instance created for the tag.
        """
        button = TagPushButton(icon, tag, checkable, self)
        button.checkedSignal.connect(self.tagCheckedSignal)
        button.closeClicked.connect(lambda _tag=tag: self.del_tag(_tag))
        if color is not None:
            button.setBackgroundColor(color)
        self._layout.addWidget(button)
        self.tags[tag] = button
        return button

    def del_tag(self, tag):
        """Remove a tag button from the group.

        Parameters
        ----------
        tag : str
            Label of the tag to remove.
        """
        button = self.tags[tag]
        self._layout.removeWidget(button)
        button.deleteLater()
        self.tagRemovedSignal.emit(tag)
        del self.tags[tag]
        self._layout.update()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QFrame

    app = QApplication(sys.argv)

    frame = QWidget()
    frame.resize(800, 600)
    layout = QVBoxLayout()
    frame.setLayout(layout)
    pushButton = TagPushButton(FluentIcon.PROJECTOR, "dwasd", frame)
    layout.addWidget(pushButton)

    frame.show()
    app.exec_()

