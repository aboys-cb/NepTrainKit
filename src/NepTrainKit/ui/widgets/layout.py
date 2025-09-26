"""Custom flow layout that arranges widgets in rows."""

from PySide6.QtCore import QRect, Qt, QSize, QPoint
from PySide6.QtWidgets import QLayout, QLayoutItem


class FlowLayout(QLayout):
    """Flow layout similar to `QtGui.QFlowLayout` with wrapping support."""

    def __init__(self, parent=None, margin=0, spacing=10):
        """Initialize the layout configuration.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget that will own the layout.
        margin : int, optional
            Padding applied to all sides of the layout.
        spacing : int, optional
            Spacing between items in pixels.
        """
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList: list[QLayoutItem] = []

    def addItem(self, item):
        """Add a layout item to the flow."""
        self.itemList.append(item)

    def count(self):
        """Return the number of managed items."""
        return len(self.itemList)

    def itemAt(self, index):
        """Return the item at the requested index or None.

        Parameters
        ----------
        index : int
            Position within the layout.

        Returns
        -------
        QLayoutItem or None
            Layout item at `index` if it exists.
        """
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        """Remove and return the item at the given index.

        Parameters
        ----------
        index : int
            Position within the layout.

        Returns
        -------
        QLayoutItem or None
            Removed layout item if the index was valid.
        """
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return super().takeAt(index)

    def insertWidget(self, index, widget):
        """Insert the widget at the provided position.

        Parameters
        ----------
        index : int
            Target position for the widget.
        widget : QWidget
            Widget to insert into the layout.
        """
        self.addWidget(widget)
        self.moveItem(len(self.itemList) - 1, index)

    def expandingDirections(self):
        """Return the layout expansion capabilities."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        """Indicate that height depends on width."""
        return True

    def heightForWidth(self, width):
        """Calculate the height required for a given width."""
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        """Apply geometry and lay out all items."""
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        """Return the recommended size of the layout."""
        return self.minimumSize()

    def minimumSize(self):
        """Compute the minimum size required to show all items."""
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.sizeHint())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    def doLayout(self, rect, testOnly):
        """Position items within `rect` and return the used height.

        Parameters
        ----------
        rect : QRect
            Bounding rectangle for layout calculations.
        testOnly : bool
            When True, only compute geometry without applying it.

        Returns
        -------
        int
            Vertical span consumed by the items.
        """
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
        """Return the item whose geometry contains the position.

        Parameters
        ----------
        pos : QPoint
            Position to query.

        Returns
        -------
        tuple[int, QLayoutItem | None]
            Index and layout item if found, otherwise (-1, None).
        """
        for i, item in enumerate(self.itemList):
            if item.geometry().contains(pos):
                return i, item
        return -1, None

    def findWidgetAt(self, widget):
        """Return the index and item that manage the given widget.

        Parameters
        ----------
        widget : QWidget
            Widget to search for.

        Returns
        -------
        tuple[int, QLayoutItem | None]
            Index and layout item if found, otherwise (-1, None).
        """
        for i, item in enumerate(self.itemList):
            if item.widget() is widget:
                return i, item
        return -1, None

    def moveItem(self, from_index, to_index):
        """Move a layout item to a new index and redraw.

        Parameters
        ----------
        from_index : int
            Original index of the item.
        to_index : int
            New index for the item.
        """
        if 0 <= from_index < len(self.itemList) and 0 <= to_index < len(self.itemList):
            item = self.itemList.pop(from_index)
            self.itemList.insert(to_index, item)
            self.update()
