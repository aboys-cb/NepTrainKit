#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @email    : 1747193328@qq.com
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QApplication, QSplitter, QStackedWidget
from qfluentwidgets import SegmentedWidget

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset.seed import seed_nep_data_git
from NepTrainKit.core.dataset.services import ModelService, ProjectService, TagService
from NepTrainKit.core.validation.library import get_library

from NepTrainKit.ui.views import ModelItemWidget, ProjectWidget
from NepTrainKit.ui.views.model_library import ModelLibraryWidget


class DataManagerWidget(QWidget):
    """Model library: validation scorecard plus the project and model registry.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that holds this page.
    """

    toolbarActionsChanged = Signal()

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
        # The data manager and model validation share the library database.
        self._db = get_library().database
        self.model_service = ModelService(self._db)
        self.project_service = ProjectService(self._db)
        self.tag_service = TagService(self._db)
        self.init_ui()
        seed_nep_data_git(self._db, self.project_service, self.model_service)

    def toolbar_actions(self):
        """Expose validation defaults only while its tab is selected."""
        if self.stack.currentWidget() is self.library_widget:
            return {"open": self.library_widget.import_action, "save": self.library_widget.save_action,
                    "save_menu": self.library_widget.save_plan_action if self.library_widget.save_plan_action.isEnabled() else self.library_widget.export_package_action}
        return {}

    def _sync_toolbar_actions(self, *_args):
        active = self.isVisible() and self.stack.currentWidget() is self.library_widget
        for menu_name, actions in (
            ("load_menu", (self.library_widget.diagnostic_action, self.library_widget.edit_diagnostic_action, self.library_widget.import_action, self.library_widget.import_package_action,)),
            ("save_menu", (self.library_widget.save_action, self.library_widget.star_action, self.library_widget.save_plan_action, self.library_widget.export_package_action)),
        ):
            menu = getattr(self._parent, menu_name, None)
            if menu is None:
                continue
            for action in actions:
                if action in menu.actions():
                    menu.removeAction(action)
                if active:
                    menu.addAction(action)
        self.toolbarActionsChanged.emit()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_toolbar_actions()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sync_toolbar_actions()

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
        self.show_projects()
        self.data_item_widget.create_model(initial_path=paths[0])
        event.acceptProposedAction()

    def show_projects(self):
        """Switch to the project registry tab."""
        database = get_library().database
        if database.path != self._db.path:
            self._db = database
            self.model_service = ModelService(database)
            self.project_service = ProjectService(database)
            self.tag_service = TagService(database)
            for widget in (self.project_widget, self.data_item_widget):
                widget.db = database
                widget.model_service = self.model_service
                widget.project_service = self.project_service
                widget.tag_service = self.tag_service
            self.data_item_widget.project = None
            self.data_item_widget._model.clear()
            self.data_item_widget._update_action_state()
        self.tabs.setCurrentItem("projects")
        self.stack.setCurrentWidget(self.splitter)
        self.project_widget.load_all_projects()
        project = getattr(self.data_item_widget, "project", None)
        if project is not None:
            self.data_item_widget.load_models_by_project(project)

    def open_scorecard(self, suite_id, model_id=None, *, draft=None):
        """Switch to the validation tab with ``suite_id`` and ``model_id`` selected."""
        self.tabs.setCurrentItem("validation")
        self.stack.setCurrentWidget(self.library_widget)
        self.library_widget.open_scorecard(suite_id, model_id, draft=draft)

    def init_ui(self):
        """Build the validation scorecard and the project/model split view.

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


        self.library_widget = ModelLibraryWidget(self)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.library_widget)
        self.stack.addWidget(self.splitter)
        self.tabs = SegmentedWidget(self)
        self.tabs.addItem("validation", "验证与诊断", lambda: self.stack.setCurrentWidget(self.library_widget))
        self.tabs.addItem("projects", self.tr("Projects"), self.show_projects)
        self.tabs.setCurrentItem("validation")

        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(8, 6, 8, 2)
        tab_row.addWidget(self.tabs)
        tab_row.addStretch(1)
        self.gridLayout.addLayout(tab_row, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.stack, 1, 0, 1, 1)

        self.setLayout(self.gridLayout)
        self.stack.currentChanged.connect(self._sync_toolbar_actions)
        for action in (self.library_widget.import_action, self.library_widget.save_action,
                       self.library_widget.star_action, self.library_widget.import_package_action,
                       self.library_widget.export_package_action, self.library_widget.diagnostic_action, self.library_widget.edit_diagnostic_action, self.library_widget.save_plan_action):
            action.changed.connect(self.toolbarActionsChanged)
