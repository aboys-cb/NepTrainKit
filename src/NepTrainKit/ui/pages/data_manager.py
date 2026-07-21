#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @email    : 1747193328@qq.com
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication, QSplitter

from NepTrainKit.paths import get_user_config_path
from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.seed import seed_nep_data_git
from NepTrainKit.core.dataset.services import ModelService, ProjectService, TagService

from NepTrainKit.ui.views import ModelItemWidget, ProjectWidget


class DataManagerWidget(QWidget):
    """Coordinate project and model management widgets backed by the local database.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that holds this page.
    """

    def __init__(self,parent=None):
        """Initialise database services and connect child widgets.

        Parameters
        ----------
        parent : QWidget | None
            Optional owner widget that holds this page.
        """
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("DataManagerWidget")
        self.setAcceptDrops(True)
        user_path = get_user_config_path()
        self._db = Database(user_path / "mlpman.db")
        self.model_service = ModelService(self._db)
        self.project_service = ProjectService(self._db)
        self.tag_service = TagService(self._db)
        self.init_ui()
        seed_nep_data_git(self._db, self.project_service, self.model_service)

    def dragEnterEvent(self, event):
        """Accept one or more local model directories.

        Parameters
        ----------
        event : QDragEnterEvent
            Drag event forwarded by Qt.

        Returns
        -------
        None
            This handler updates the event acceptance state.
        """
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(url.isLocalFile() and os.path.isdir(url.toLocalFile()) for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Open model creation with the first dropped directory prefilled.

        Parameters
        ----------
        event : QDropEvent
            Drop event containing file URLs.

        Returns
        -------
        None
            This handler opens the existing model editor for the selected project.
        """
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile() and os.path.isdir(url.toLocalFile())
        ]
        if not paths:
            event.ignore()
            return
        if not hasattr(self.data_item_widget, "project"):
            MessageManager.send_info_message(
                self.tr("Select a project before dropping a model folder")
            )
            event.ignore()
            return
        self.data_item_widget.create_model(initial_path=paths[0])
        event.acceptProposedAction()

    def init_ui(self):
        """Build the project and model split view.

        Returns
        -------
        None
            The UI tree is created directly on the widget.
        """

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

        self.splitter.setSizes([260,740])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 8)
        # self.splitter.setStretchFactor(2, 2)


        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

        self.setLayout(self.gridLayout)
