#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/3 13:41
# @Author  : 兵
# @email    : 1747193328@qq.com
from collections import defaultdict

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from PySide6.QtWidgets import QApplication, QCompleter, QStyleOptionViewItem, QStyledItemDelegate, QStyle

CountRole = Qt.UserRole +1

class CompleterModel(QAbstractListModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        if isinstance(data, list):
            self.data_map=self.parser_list(data)
        else:

            self.data_map = data or {}

    def parser_list(self,data):
        _dict = defaultdict(int)  # 默认值为 0
        for row in data:
            _dict[row] += 1
        return dict(_dict)  # 转换回普通字典



    def rowCount(self, parent=QModelIndex()):
        return len(self.data_map.keys())

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):

        if not index.isValid():

            return None
        # print(role)
        # Qt.ItemDataRole.DisplayRole
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole or role == CountRole:
            # 获取词和对应的出现次数
            word = list(self.data_map.keys())[index.row()]
            count = self.data_map[word]
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return word
            elif role == CountRole:

                return str(count)




    def set_data(self, data):
        self.beginResetModel()
        self.data_map = data
        self.endResetModel()

class JoinDelegate(QStyledItemDelegate):
    def __init__(self, parent=None,data={}):
        super().__init__(parent)
        self.data = data
    def paint(self, painter, option, index):
        # 使用 initStyleOption 初始化选项
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        # 获取模型数据
        model = index.model()

        text1 = model.data(index , Qt.DisplayRole)  # 第一列数据
        if text1  in self.data:
            text2=str(self.data[text1])
        else:
            text2="unknown"


        # 修改绘制的文本
        opt.text = text1
        opt.displayAlignment = Qt.AlignLeft | Qt.AlignVCenter
        widget = option.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        # 绘制第二列，右对齐
        opt.text = text2
        rect2 = opt.rect
        rect2.setLeft(opt.rect.left() + opt.rect.width() // 2)  # 将第二列区域推到右边
        opt.displayAlignment = Qt.AlignRight | Qt.AlignVCenter
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)




class ConfigCompleter(QCompleter):
    def __init__(self, data, parent=None):
        QCompleter.__init__(self, parent)
        # columns: are the columns that are going to concatenate
        self._model = CompleterModel(data)
        self.setModel(self._model)
        self.setFilterMode(Qt.MatchContains)


