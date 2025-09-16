#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication, QSplitter

from NepTrainKit import get_user_config_path
from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.services import ModelService, ProjectService, TagService

from NepTrainKit.views.dataset_widget import ModelItemWidget
from NepTrainKit.views.project_view import ProjectWidget


class DataManagerWidget(QWidget):

    def __init__(self,parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("DataManagerWidget")
        self.setAcceptDrops(True)
        user_path = get_user_config_path()
        self._db = Database(os.path.join(user_path, "mlpman.db"))
        self.model_service = ModelService(self._db)
        self.project_service = ProjectService(self._db)
        self.tag_service = TagService(self._db)
        self.init_ui()
        self.project_widget.gen_nep_data_git()
        # self.project_widget.gen_test()

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
        self.project_widget.db=self._db
        self.project_widget.project_service=self.project_service
        self.project_widget.model_service=self.model_service
        self.project_widget.tag_service=self.tag_service
        self.splitter.addWidget(self.project_widget)
        self.data_item_widget = ModelItemWidget(self)
        self.data_item_widget.db=self._db
        self.data_item_widget.project_service=self.project_service
        self.data_item_widget.model_service=self.model_service
        self.data_item_widget.tag_service=self.tag_service
        # self.data_info_widget = QWidget(self)
        # self.data_info_widget.setAutoFillBackground(True)


        self.project_widget.projectChangedSignal.connect(self.data_item_widget.load_models_by_project)

        self.splitter.addWidget(self.project_widget)
        self.splitter.addWidget(self.data_item_widget)
        # self.splitter.addWidget(self.data_info_widget)

        self.splitter.setSizes([170,800,200])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 8)
        # self.splitter.setStretchFactor(2, 2)


        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

        self.setLayout(self.gridLayout)
