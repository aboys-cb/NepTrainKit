"""Completer utilities for displaying suggestion frequency."""

from collections import defaultdict

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from PySide6.QtWidgets import QApplication, QCompleter, QStyleOptionViewItem, QStyledItemDelegate, QStyle

CountRole = Qt.ItemDataRole.UserRole + 1


class CompleterModel(QAbstractListModel):
    """List model that keeps track of completion items and their counts."""

    def __init__(self, data=None, parent=None):
        """Initialize the model from a mapping or list of values.

        Parameters
        ----------
        data : dict[str, int] or list[str], optional
            Either a mapping of values to occurrence counts or a raw list
            that will be aggregated into counts.
        parent : QObject, optional
            Parent object that manages the model lifetime.
        """
        super().__init__(parent)
        if isinstance(data, list):
            self.data_map = self.parser_list(data)
        else:
            self.data_map = data or {}

    def parser_list(self, data):
        """Convert a list of strings into a count mapping.

        Parameters
        ----------
        data : list[str]
            Input values whose frequency will be recorded.

        Returns
        -------
        dict[str, int]
            Mapping from string to number of occurrences.
        """
        counter = defaultdict(int)
        for row in data:
            counter[row] += 1
        return dict(counter)

    def rowCount(self, parent=QModelIndex()):
        """Return the number of unique completion items.

        Parameters
        ----------
        parent : QModelIndex, optional
            Required by Qt but unused because the model is flat.

        Returns
        -------
        int
            Count of distinct entries stored in the model.
        """
        return len(self.data_map.keys())

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole):
        """Return display data or metadata for the requested index.

        Parameters
        ----------
        index : QModelIndex
            Index of the requested item.
        role : int, optional
            Data role specifying the representation to return.

        Returns
        -------
        str or None
            Item text for display/edit roles, or the count string for
            `CountRole`. Returns None for invalid indexes.
        """
        if not index.isValid():
            return None

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, CountRole):
            word = list(self.data_map.keys())[index.row()]
            count = self.data_map[word]
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return word
            if role == CountRole:
                return str(count)
        return None

    def set_data(self, data):
        """Replace the underlying mapping and notify the view.

        Parameters
        ----------
        data : dict[str, int]
            New mapping of completion items to their counts.
        """
        self.beginResetModel()
        self.data_map = data
        self.endResetModel()


class JoinDelegate(QStyledItemDelegate):
    """Item delegate that renders the suggestion and its count."""

    def __init__(self, parent=None, data=None):
        """Configure the delegate with optional precomputed counts.

        Parameters
        ----------
        parent : QObject, optional
            Owner of the delegate.
        data : dict[str, int], optional
            Mapping from suggestion text to frequency.
        """
        super().__init__(parent)
        self.data = data or {}

    def paint(self, painter, option, index):
        """Render the suggestion text and count within the same row.

        Parameters
        ----------
        painter : QPainter
            Painter used for drawing.
        option : QStyleOptionViewItem
            Style information provided by the view.
        index : QModelIndex
            Index of the item to render.
        """
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        model = index.model()

        text1 = model.data(index, Qt.ItemDataRole.DisplayRole)
        text2 = str(self.data[text1]) if text1 in self.data else "unknown"

        opt.text = text1  # type: ignore[assignment]
        opt.displayAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter  # type: ignore[assignment]
        widget = option.widget  # type: ignore[attr-defined]
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        opt.text = text2
        rect2 = opt.rect  # type: ignore[assignment]
        rect2.setLeft(opt.rect.left() + opt.rect.width() // 2)  # type: ignore[attr-defined]
        opt.displayAlignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  # type: ignore[assignment]
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)


class ConfigCompleter(QCompleter):
    """Completer that filters configuration keys using substring matching."""

    def __init__(self, data, parent=None):
        """Initialize the completer with a pre-populated model.

        Parameters
        ----------
        data : dict[str, int] or list[str]
            Source data used to populate the completion model.
        parent : QObject, optional
            Owner widget for the completer.
        """
        QCompleter.__init__(self, parent)
        self._model = CompleterModel(data)
        self.setModel(self._model)
        self.setFilterMode(Qt.MatchFlag.MatchContains)
