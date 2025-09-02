#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @Author  : 兵
# @email    : 1747193328@qq.com

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication, QSplitter

from NepTrainKit.views import ModelItemTableWidget
from NepTrainKit.views.project_view import ProjectWidget


class DataManagerWidget(QWidget):

    def __init__(self,parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("DataManagerWidget")
        self.setAcceptDrops(True)


        self.init_ui()


    def dragEnterEvent(self, event):
        # 检查拖拽的内容是否包含文件
        if event.mimeData().hasUrls():
            event.acceptProposedAction()  # 接受拖拽事件
        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):
        # 获取拖拽的文件路径
        # print("dropEvent",event)
        urls = event.mimeData().urls()

        if urls:

            for url in urls:
                pass
    def init_ui(self):

        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("DataManagerWidget_gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.project_widget = ProjectWidget(self)
        self.project_widget.setObjectName("project_widget")
        self.project_widget.setAutoFillBackground(True)

        self.data_item_widget = ModelItemTableWidget(self)
        self.data_info_widget = QWidget(self)
        self.data_info_widget.setAutoFillBackground(True)
        self.splitter.addWidget(self.project_widget)
        self.splitter.addWidget(self.data_item_widget)
        self.splitter.addWidget(self.data_info_widget)

        self.splitter.setSizes([160,800,200])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 8)
        self.splitter.setStretchFactor(2, 2)


        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        # self.gridLayout.addWidget(self.setting_group, 0, 0, 1, 2)
        # self.gridLayout.addWidget(self.workspace_card_widget, 1, 0, 1, 2)
        # self.gridLayout.addWidget(self.dataset_info_label, 2, 0, 1, 1)
        # self.gridLayout.addWidget(self.path_label, 2, 1, 1, 1,alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(self.gridLayout)
