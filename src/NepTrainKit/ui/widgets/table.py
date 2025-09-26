"""Table models used for ID/name lookups with streaming support."""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class IdNameTableModel(QAbstractTableModel):
    """Two-column table model that lazily loads `(id, name)` records."""

    __slots__ = ("_rows", "_headers", "_fetch_supplier", "_eof")

    def __init__(self, parent=None):
        """Initialize internal storage and default headers."""
        super().__init__(parent)
        self._rows = []
        self._headers = ("ID", "Name")
        self._fetch_supplier = None
        self._eof = True

    def rowCount(self, parent=QModelIndex()) -> int:
        """Return the number of populated rows."""
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return the fixed column count."""
        return 0 if parent.isValid() else 2

    def setHorizontalHeaderLabels(self, labels):
        """Set human-readable column headers."""
        self._headers = tuple(labels)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        """Provide header text for horizontal or vertical sections."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < 2:
                return self._headers[section]
            return None
        if orientation == Qt.Vertical:
            return section + 1
        return None

    def data(self, index, role=Qt.DisplayRole):
        """Return cell values for display roles."""
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        r, c = index.row(), index.column()
        if not (0 <= r < len(self._rows)) or not (0 <= c < 2):
            return None
        return self._rows[r][c]

    def flags(self, index):
        """Mark rows as selectable but not editable."""
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        """Sort rows in-place by the requested column."""
        if column not in (0, 1):
            return
        self.layoutAboutToBeChanged.emit()
        self._rows.sort(key=lambda x: x[column], reverse=(order == Qt.SortOrder.DescendingOrder))
        self.layoutChanged.emit()

    def set_header(self, headers=("ID", "Name")):
        """Convenience method to update both header labels."""
        self._headers = tuple(headers)

    def clear(self):
        """Reset the model, removing cached rows and suppliers."""
        self.beginResetModel()
        self._rows.clear()
        self._fetch_supplier = None
        self._eof = True
        self.endResetModel()

    def load_records_fast(self, records):
        """Load all records from memory into the model at once.

        Parameters
        ----------
        records : Iterable[tuple[int, str] | dict[str, Any]]
            Sequence of `(id, name)` tuples or dictionaries with `id`/`name` keys.
        """
        self.beginResetModel()
        rows = []
        for r in records:
            if isinstance(r, dict):
                rows.append((r.get("id"), r.get("name")))
            else:
                rows.append((r[0], r[1]))
        self._rows = rows
        self._fetch_supplier = None
        self._eof = True
        self.endResetModel()

    def load_records_stream(self, query, batch_size=5000):
        """Stream records from an iterator-like query in batches.

        Parameters
        ----------
        query : Iterable
            Iterator yielding objects or tuples containing id/name values.
        batch_size : int, optional
            Number of records to insert per batch.
        """
        self.beginResetModel()
        self._rows = []
        self._fetch_supplier = None
        self._eof = True
        self.endResetModel()

        buf = []

        def flush(buf_):
            if not buf_:
                return
            start = len(self._rows)
            end = start + len(buf_) - 1
            self.beginInsertRows(QModelIndex(), start, end)
            self._rows.extend(buf_)
            self.endInsertRows()
            buf_.clear()

        for row in query:
            if isinstance(row, (tuple, list)):
                rid, rname = row[0], row[1]
            else:
                rid = getattr(row, "id", None)
                rname = getattr(row, "name", None)
            buf.append((rid, rname))
            if len(buf) >= batch_size:
                flush(buf)
        flush(buf)

    def set_fetch_supplier(self, supplier_callable):
        """Register a callable that supplies additional rows on demand.

        Parameters
        ----------
        supplier_callable : Callable[[], list[tuple[int, str]] | None]
            Function invoked by `fetchMore` to retrieve extra rows. Returning
            `None` or an empty list marks the end of data.
        """
        self._fetch_supplier = supplier_callable
        self._eof = supplier_callable is None

    def canFetchMore(self, parent=QModelIndex()) -> bool:
        """Report whether additional data can be fetched."""
        if parent.isValid():
            return False
        return not self._eof and (self._fetch_supplier is not None)

    def fetchMore(self, parent=QModelIndex()):
        """Fetch additional rows using the registered supplier."""
        if parent.isValid() or self._fetch_supplier is None:
            return
        chunk = self._fetch_supplier()
        if not chunk:
            self._eof = True
            return
        start = len(self._rows)
        end = start + len(chunk) - 1
        self.beginInsertRows(QModelIndex(), start, end)
        self._rows.extend(chunk)
        self.endInsertRows()

    def get_row(self, row: int):
        """Return the record at `row` as a dictionary.

        Parameters
        ----------
        row : int
            Row index to retrieve.

        Returns
        -------
        dict[str, Any] or None
            Mapping containing `id` and `name`, or None if the index is invalid.
        """
        if 0 <= row < len(self._rows):
            rid, rname = self._rows[row]
            return {"id": rid, "name": rname}
        return None

