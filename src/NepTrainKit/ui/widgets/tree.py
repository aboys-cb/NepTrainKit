"""Tree models and delegates supporting tagged dataset display."""

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QRect, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QIcon
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QWidget, QLineEdit, QCompleter


class TreeItem:
    """Node used by `TreeModel` to store hierarchical data."""

    __slots__ = ("parentItem", "childItems", "itemData", "_row", "icon")

    def __init__(self, data, parent=None):
        """Initialize the tree item with data and a parent node.

        Parameters
        ----------
        data : Iterable[Any]
            Column values stored for this node.
        parent : TreeItem, optional
            Parent node in the tree hierarchy.
        """
        self.parentItem = parent
        self.childItems = []
        self.icon: QIcon | None = None
        self.itemData = tuple(data) if not isinstance(data, tuple) else data
        self._row = -1

    def appendChild(self, child):
        """Append a child and update its bookkeeping."""
        child.parentItem = self
        child._row = len(self.childItems)
        self.childItems.append(child)
        return child

    def insertChild(self, row, child):
        """Insert a child at the specified index and re-index siblings."""
        if row < 0:
            row = 0
        elif row > len(self.childItems):
            row = len(self.childItems)
        child.parentItem = self
        child._row = row
        self.childItems.insert(row, child)
        for i in range(row + 1, len(self.childItems)):
            self.childItems[i]._row = i

    def removeChild(self, row):
        """Remove the child at `row` and update sibling indices."""
        if 0 <= row < len(self.childItems):
            self.childItems.pop(row)
            for i in range(row, len(self.childItems)):
                self.childItems[i]._row = i

    def clear(self):
        """Remove all child items."""
        self.childItems.clear()

    def child(self, row):
        """Return the child at `row` if it exists."""
        if 0 <= row < len(self.childItems):
            return self.childItems[row]
        return None

    def childCount(self):
        """Return the number of immediate children."""
        return len(self.childItems)

    def columnCount(self):
        """Return the number of columns stored in this node."""
        return len(self.itemData)

    def data(self, column):
        """Return the data for a specific column."""
        if 0 <= column < len(self.itemData):
            return self.itemData[column]
        return None

    def setRow(self, row: int):
        """Update the cached row index for this item."""
        self._row = row

    def row(self):
        """Return the cached row index relative to the parent."""
        return self._row

    def parent(self):
        """Return the parent item."""
        return self.parentItem


class TreeModel(QAbstractItemModel):
    """Hierarchical model with optional per-node icons and tag support."""

    def __init__(self, parent=None):
        """Create an empty model with a root item placeholder."""
        super(TreeModel, self).__init__(parent)
        self.rootItem = TreeItem(())
        self.count_column = None
        self._base_flags = super().flags(QModelIndex())

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """Return display text or decoration data for the index."""
        if not index.isValid():
            return None
        item = index.internalPointer()
        col = index.column()

        if role == Qt.ItemDataRole.DecorationRole and col == 0:
            return item.icon

        if role != Qt.DisplayRole:
            return None

        if col == self.count_column:
            return item.childCount()
        return item.data(col)

    def flags(self, index: QModelIndex):
        """Return item flags indicating enabled/selectable state."""
        if not index.isValid():
            return self._base_flags | Qt.ItemIsEnabled
        return self._base_flags | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        """Provide header labels for the view."""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)
        return None

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        """Construct an index for the requested row and column."""
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        childItem = parentItem.child(row)
        if childItem is None:
            return QModelIndex()
        return self.createIndex(row, column, childItem)

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Return the parent index for the given child index."""
        if not index.isValid():
            return QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem is None or parentItem == self.rootItem:
            return QModelIndex()
        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of children for the parent index."""
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        return parentItem.childCount()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns in the model."""
        return self.rootItem.columnCount()

    def clear(self):
        """Remove all children from the root item."""
        self.rootItem.clear()

    def setHeader(self, header):
        """Reset the model using the provided header tuple."""
        self.beginResetModel()
        self.rootItem = TreeItem(tuple(header))
        self.endResetModel()

    def rebuild(self, builder_fn):
        """Repopulate the tree using an external builder callback.

        Parameters
        ----------
        builder_fn : Callable[[TreeItem], None]
            Function that receives the root item and populates children.
        """
        self.beginResetModel()
        self.rootItem.clear()
        builder_fn(self.rootItem)
        self.endResetModel()

    def insertRows(self, row, count, parent=QModelIndex()):
        """Insert empty rows at the given index."""
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        if row < 0 or row > parentItem.childCount() or count <= 0:
            return False
        self.beginInsertRows(parent, row, row + count - 1)
        for i in range(count):
            parentItem.insertChild(row + i, TreeItem((), parentItem))
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QModelIndex()):
        """Remove rows starting at `row` for the given parent."""
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        if row < 0 or row + count > parentItem.childCount() or count <= 0:
            return False
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            parentItem.removeChild(row)
        self.endRemoveRows()
        return True

    def add_item(self, parent, site):
        """Convenience helper to append a new child item.

        Parameters
        ----------
        parent : int or TreeItem
            Parent identifier or item that will own the new node. When an
            integer is provided `-1` points to the root.
        site : Iterable[Any]
            Column values stored in the new tree item.

        Returns
        -------
        TreeItem
            Newly created tree item instance.
        """
        self.beginResetModel()
        if isinstance(parent, int):
            if parent == -1:
                parent = self.rootItem
            else:
                parent = self.rootItem.child(parent)
        elif isinstance(parent, TreeItem):
            pass
        else:
            raise ValueError("parent must be TreeItem or int")
        child = parent.appendChild(TreeItem(site, parent))
        self.endResetModel()
        return child


class TagDelegate(QStyledItemDelegate):
    """Delegate that draws colored tag chips inside tree view cells."""

    def __init__(self, parent=None, tag_list=None):
        """Store optional tag suggestions for the editor completer."""
        super().__init__(parent)
        self.tag_list = tag_list or []

    def sizeHint(self, option, index):
        """Return a size hint that fits all rendered tags."""
        tags = index.data(Qt.DisplayRole)
        if not tags:
            return super().sizeHint(option, index)

        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        font = option.font
        font.setPointSize(font.pointSize() - 1)
        metrics = option.fontMetrics if option.fontMetrics else option.widget.fontMetrics()

        spacing = 4
        padding = 6
        height = metrics.height() + 8

        total_width = 4
        for tag_info in tags:
            tag = tag_info["name"]
            tag_width = metrics.horizontalAdvance(tag) + padding * 2
            total_width += tag_width + spacing

        return QSize(total_width, height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        """Draw rounded rectangles for each tag with contrasting text."""
        tags = index.data(Qt.DisplayRole)
        if not tags:
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        rect = option.rect
        x, y = rect.x() + 4, rect.y() + 2
        spacing = 4
        padding = 6
        tag_height = rect.height() - 4

        font = option.font
        font.setPointSize(font.pointSize() - 1)
        painter.setFont(font)

        for tag_info in tags:
            tag = tag_info["name"]
            color = tag_info["color"]
            tag_width = painter.fontMetrics().horizontalAdvance(tag) + padding * 2
            r = QRect(x, y, tag_width, tag_height)

            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(r, 6, 6)

            painter.setPen(QPen(Qt.black))
            painter.drawText(r, Qt.AlignCenter, tag)

            x += tag_width + spacing
            if x > rect.right():
                break

        painter.restore()

    def createEditor(self, parent: QWidget, option, index):
        """Create a line edit that supports comma-separated tag entry."""
        editor = QLineEdit(parent)
        editor.setPlaceholderText("Comma-separated tags")
        if self.tag_list:
            completer = QCompleter(self.tag_list, editor)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            editor.setCompleter(completer)
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        """Populate the editor with the tag list from the model."""
        tags = index.data(Qt.EditRole)
        if isinstance(tags, list):
            editor.setText(", ".join(tags))
        elif isinstance(tags, str):
            editor.setText(tags)

    def setModelData(self, editor: QLineEdit, model, index):
        """Write the edited tag list back to the model."""
        text = editor.text().strip()
        tags = [t.strip() for t in text.split(",") if t.strip()]
        model.setData(index, tags, Qt.EditRole)


