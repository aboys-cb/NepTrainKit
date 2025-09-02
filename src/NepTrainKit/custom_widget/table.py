from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

class IdNameTableModel(QAbstractTableModel):
    """
    高性能两列表模型：ID / Name
    - load_records_fast(records): 一次性灌入 [(id, name), ...] 或 [{'id':..,'name':..}, ...]
    - load_records_stream(query, batch_size): SQLAlchemy 流式插入（持续插行信号）
    - set_fetch_supplier(supplier): 配合 canFetchMore/fetchMore 的懒加载（滚动再拉）
    """

    __slots__ = ("_rows", "_headers", "_fetch_supplier", "_eof")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []                 # list[tuple(id, name)]
        self._headers = ("ID", "Name")  # 表头
        self._fetch_supplier = None     # 懒加载供应器: () -> list[tuple]|None
        self._eof = True                # 懒加载是否已到结尾

    # ---------- 基础接口 ----------
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 2

    def setHorizontalHeaderLabels(self, labels):
        self._headers = tuple(labels)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < 2:
                return self._headers[section]
            return None
        # 垂直表头可返回行号（按需）
        if orientation == Qt.Vertical:
            return section + 1
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        r, c = index.row(), index.column()
        if not (0 <= r < len(self._rows)) or not (0 <= c < 2):
            return None
        return self._rows[r][c]

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    # ---------- 可选：排序（在内存里按列排序；更建议 SQL 端排序） ----------
    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        if column not in (0, 1):
            return
        self.layoutAboutToBeChanged.emit()
        self._rows.sort(key=lambda x: x[column], reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()

    # ---------- 表头设置 ----------
    def set_header(self, headers=("ID", "Name")):
        self._headers = tuple(headers)

    # ---------- 快速清空 ----------
    def clear(self):
        self.beginResetModel()
        self._rows.clear()
        self._fetch_supplier = None
        self._eof = True
        self.endResetModel()

    # ---------- 方式 A：一次性灌入（内存允许时最快） ----------
    def load_records_fast(self, records):
        """
        records: Iterable of (id, name) 或 {'id':..., 'name':...}
        """
        self.beginResetModel()
        rows = []
        # 若是可预知长度，可预分配（Python list 无显式预分配，这里直接收集）
        for r in records:
            if isinstance(r, dict):
                rows.append((r.get("id"), r.get("name")))
            else:
                # 假定 (id, name)
                rows.append((r[0], r[1]))
        self._rows = rows
        self._fetch_supplier = None
        self._eof = True
        self.endResetModel()

    # ---------- 方式 B：SQLAlchemy 流式/分批灌入（省内存，UI 可见持续增长） ----------
    def load_records_stream(self, query, batch_size=5000):
        """
        query: 例如
          q = (session.query(User.id, User.name)
                 .execution_options(stream_results=True)
                 .yield_per(batch_size))
        """
        # 先重置
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

    # ---------- 方式 C：懒加载（视图滚动到底部再拉数据） ----------
    def set_fetch_supplier(self, supplier_callable):
        """
        supplier_callable: 无参可调用 -> list[(id, name)] 或 None/[]（表示已到底）
        可把 SQLAlchemy 的游标/生成器封装成每次返回 <= batch_size 的列表。
        """
        self._fetch_supplier = supplier_callable
        self._eof = supplier_callable is None

    def canFetchMore(self, parent=QModelIndex()) -> bool:
        if parent.isValid():
            return False
        return not self._eof and (self._fetch_supplier is not None)

    def fetchMore(self, parent=QModelIndex()):
        if parent.isValid() or self._fetch_supplier is None:
            return
        chunk = self._fetch_supplier()  # 期望返回 list[(id, name)] 或 None
        if not chunk:
            self._eof = True
            return
        start = len(self._rows)
        end = start + len(chunk) - 1
        self.beginInsertRows(QModelIndex(), start, end)
        self._rows.extend(chunk)
        self.endInsertRows()

    # ---------- 小工具：读取一行 ----------
    def get_row(self, row: int):
        if 0 <= row < len(self._rows):
            rid, rname = self._rows[row]
            return {"id": rid, "name": rname}
        return None
