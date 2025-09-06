#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 09:54
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QRect, QSize


from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush,QIcon
from PySide6.QtWidgets import (
    QStyledItemDelegate, QStyleOptionViewItem, QWidget,
    QLineEdit, QCompleter
)

class TreeItem:
    __slots__ = ("parentItem", "childItems", "itemData", "_row","icon")

    def __init__(self, data, parent=None):
        # data 建议用 tuple，一次性创建后不再改动，节省内存 & 更快
        self.parentItem = parent
        self.childItems = []
        self.icon:QIcon|None=None
        self.itemData = tuple(data) if not isinstance(data, tuple) else data
        self._row = -1  # 本节点在父节点中的行号（O(1) 获取）

    # ------- 结构维护：确保父子关系与 _row 始终一致 -------
    def appendChild(self, child):
        # ⚡ 必须维护 parent & row
        child.parentItem = self
        child._row = len(self.childItems)
        self.childItems.append(child)
        return child
    def insertChild(self, row, child):
        # 安全行号
        if row < 0:
            row = 0
        elif row > len(self.childItems):
            row = len(self.childItems)
        child.parentItem = self
        child._row = row
        self.childItems.insert(row, child)
        # ⚡ 更新后续兄弟的 _row（一次性顺序修正，仍是 O(n) 但仅在插入/删除发生时）
        for i in range(row + 1, len(self.childItems)):
            self.childItems[i]._row = i

    def removeChild(self, row):
        if 0 <= row < len(self.childItems):
            # ⚡ 不用 list.remove(child) —— 直接按行号删除（避免 O(n) 查找）
            self.childItems.pop(row)
            # ⚡ 批量修正行号
            for i in range(row, len(self.childItems)):
                self.childItems[i]._row = i

    def clear(self):
        # 大量删除时，更快
        self.childItems.clear()

    # ------- 快速访问 -------
    def child(self, row):
        # ⚡ 边界检查，安全且更快
        if 0 <= row < len(self.childItems):
            return self.childItems[row]
        return None

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return len(self.itemData)

    def data(self, column):

        # ⚡ 边界检查
        if 0 <= column < len(self.itemData):
            return self.itemData[column]
        return None

    def setRow(self, row: int):
        self._row = row

    def row(self):
        return self._row

    def parent(self):
        return self.parentItem


class TreeModel(QAbstractItemModel):
    def __init__(self, parent=None):
        super(TreeModel, self).__init__(parent)
        self.rootItem = TreeItem(())  # 根的 header 在 setHeader 设置
        # 预缓存：flags 常量
        self.count_column = None
        self._base_flags = super().flags(QModelIndex())

    # ------- 数据访问 -------
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        # ⚡ 快速路径：只处理显示文本，其他 role 直接返回 None
        item = index.internalPointer()
        col = index.column()

        if role==Qt.ItemDataRole.DecorationRole:

            if col==0:
                return item.icon


        if role != Qt.DisplayRole  :


            return None



        if col == self.count_column:
            return item.childCount()
        else:
            # 原实现为 item.data(0)
            return item.data(col)



    def flags(self, index: QModelIndex):
        if not index.isValid():
            return self._base_flags | Qt.ItemIsEnabled
        # ⚡ 避免 super().flags(index) 的重复调用（会创建临时对象）
        return self._base_flags | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    # ------- 表头 -------
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)
        return None

    # ------- 索引构造 -------
    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        childItem = parentItem.child(row)
        if childItem is None:
            return QModelIndex()
        # ⚡ 使用内部指针保存 item，row/column 也写入（Qt 会缓存一部分）
        return self.createIndex(row, column, childItem)

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem is None or parentItem == self.rootItem:
            return QModelIndex()
        # ⚡ 这里依赖子节点维护好的 _row，O(1)
        return self.createIndex(parentItem.row(), 0, parentItem)

    # ------- 维度 -------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        return parentItem.childCount()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        # ⚡ 表头长度固定 O(1)
        return self.rootItem.columnCount()
    def clear(self):
        self.rootItem.clear()

    # ------- 批量重建 / 表头 -------
    def setHeader(self, header):
        # ⚡ 一次性重置模型（大量变更时显著更快）
        self.beginResetModel()
        self.rootItem = TreeItem(tuple(header))
        self.endResetModel()

    # ------- 可选：批量构建接口（推荐用于大数据载入） -------
    def rebuild(self, builder_fn):
        """
        builder_fn(rootItem: TreeItem) -> None
        外部传入一个函数，在里面把整棵树构建到 rootItem 下。
        好处：一次 beginResetModel/endResetModel，大幅减少 UI 刷新和 QModelIndex 生成。
        """
        self.beginResetModel()
        # 清空旧树
        self.rootItem.clear()
        # 构建新树
        builder_fn(self.rootItem)
        self.endResetModel()

    # ------- 可选：插入/删除行（若你需要动态更新 UI） -------
    def insertRows(self, row, count, parent=QModelIndex()):
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        if row < 0 or row > parentItem.childCount() or count <= 0:
            return False
        self.beginInsertRows(parent, row, row + count - 1)
        for i in range(count):
            parentItem.insertChild(row + i, TreeItem((), parentItem))
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QModelIndex()):
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        if row < 0 or row + count > parentItem.childCount() or count <= 0:
            return False
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            parentItem.removeChild(row)
        self.endRemoveRows()
        return True

    def add_item(self,parent,site):

        self.beginResetModel()
        if   isinstance(parent,int):
            if parent==-1:
                parent = self.rootItem
            else:
                parent= self.rootItem.child(parent)
        elif isinstance(parent, TreeItem):
            pass
        else:
            raise ValueError("parent must be TreeItem or int")
        child = parent.appendChild(TreeItem(site, parent))

        self.endResetModel()
        return child




class TagDelegate(QStyledItemDelegate):
    """
    一个简单的 Tag 代理：
    - 在 TreeView 中以彩色小圆角矩形显示标签
    - 双击可编辑，输入逗号分隔的标签
    - 可选支持自动补全（传入 tag_list）
    """

    def __init__(self, parent=None, tag_list=None):
        super().__init__(parent)
        self.tag_list = tag_list or []   # 用于 QCompleter
    def sizeHint(self, option, index):
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

        # 计算总宽度
        total_width = 4  # 左侧内边距
        for tag_info in tags:
            tag=tag_info["name"]
            tag_width = metrics.horizontalAdvance(tag) + padding * 2
            total_width += tag_width + spacing

        return QSize(total_width, height)

    # --- 显示 ---
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        # 从 model 拿到数据，假设是 list[str]
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
            tag=tag_info["name"]
            color = tag_info["color"]
            tag_width = painter.fontMetrics().horizontalAdvance(tag) + padding * 2
            r = QRect(x, y, tag_width, tag_height)

            # 背景
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(r, 6, 6)

            # 文本
            painter.setPen(QPen(Qt.black))
            painter.drawText(r, Qt.AlignCenter, tag)

            x += tag_width + spacing
            if x > rect.right():
                break  # 超出就不画

        painter.restore()

    # --- 编辑器 ---
    def createEditor(self, parent: QWidget, option, index):
        editor = QLineEdit(parent)
        editor.setPlaceholderText("逗号分隔多个标签")
        if self.tag_list:
            completer = QCompleter(self.tag_list, editor)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            editor.setCompleter(completer)
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        tags = index.data(Qt.EditRole)
        if isinstance(tags, list):
            editor.setText(", ".join(tags))
        elif isinstance(tags, str):
            editor.setText(tags)

    def setModelData(self, editor: QLineEdit, model, index):
        text = editor.text().strip()
        tags = [t.strip() for t in text.split(",") if t.strip()]
        model.setData(index, tags, Qt.EditRole)


