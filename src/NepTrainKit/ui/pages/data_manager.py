#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QApplication, QSplitter, QStackedWidget
from qfluentwidgets import SegmentedWidget

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset.seed import seed_nep_data_git
from NepTrainKit.core.dataset.services import ModelService, ProjectService, TagService
from NepTrainKit.core.validation.library import get_library
from NepTrainKit.core.validation.service import ValidationService, find_model_files

from NepTrainKit.ui.views import ModelItemWidget, ProjectWidget
from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
from NepTrainKit.ui.widgets.validation_dialogs import ImportModelDialog


def _dropped_models(mime_data):
    if not mime_data.hasUrls():
        return []
    return list(dict.fromkeys(
        model for url in mime_data.urls() if url.isLocalFile()
        for model in find_model_files(url.toLocalFile())
    ))


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
            library = self.library_widget
            editing = not library.editor_scroll.isHidden()
            temporary = library.current_suite_id() is not None and library.current_suite_id() < 0
            return {"open": library.import_action,
                    "save": library.save_plan_action if editing or temporary else library.save_action,
                    "save_menu": library.save_action if temporary else library.export_package_action}
        return {}

    def _sync_toolbar_actions(self, *_args):
        active = self.isVisible() and self.stack.currentWidget() is self.library_widget
        for menu_name, actions in (
            ("load_menu", (self.library_widget.diagnostic_action, self.library_widget.edit_diagnostic_action, self.library_widget.import_action, self.library_widget.import_files_action, self.library_widget.import_folder_action, self.library_widget.import_package_action,)),
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
        """Accept TXT files or folders containing recognisable NEP model headers."""
        if _dropped_models(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Review the detected file, then save its bytes as a model snapshot."""
        paths = _dropped_models(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._import_models(paths)

    def import_model_path(self, path):
        paths = find_model_files(path)
        if not paths:
            MessageManager.send_error_message(self.tr("No valid NEP model header was found in this path."))
            return
        self._import_models(paths)

    def _import_models(self, paths):
        self.show_projects()
        project = self.data_item_widget.project
        service = ValidationService(get_library())
        dialog = ImportModelDialog(self.window(), service, paths, project_name=project.name if project else "")
        try:
            if not dialog.exec():
                return
            record = dialog.save_model(dialog.draft)
        except (OSError, ValueError) as error:
            MessageManager.send_error_message(self.tr("Import failed: {error}").format(error=error))
            return
        finally:
            dialog.deleteLater()
        self.show_projects()
        project = self.project_service.get_project_by_id(record["project_id"])
        self.data_item_widget.load_models_by_project(project)
        self._on_registry_changed(record["id"])
        MessageManager.send_success_message(self.tr("Model imported: {name}").format(name=record["name"]))

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
        self.library_widget.open_scorecard(suite_id, model_id, draft=draft)
        self.tabs.setCurrentItem("validation")
        self.stack.setCurrentWidget(self.library_widget)

    def _on_registry_changed(self, model_id=None):
        """Keep the project counts and validation selectors in sync after edits."""
        self.project_widget.load_all_projects()
        current = self.data_item_widget.project
        if current is not None:
            updated = self.project_widget.project_item_dict.get(current.project_id)
            if updated is None:
                self.data_item_widget.load_models_by_project(None)
            else:
                self.data_item_widget.load_models_by_project(updated)
        if self.library_widget._run_task is None:
            self.library_widget.refresh()
            reason = getattr(self.library_widget, "_unavailable_models", {}).get(model_id)
            if reason:
                MessageManager.send_info_message("模型已登记，但暂不能用于验证：" + reason)

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
        self.data_item_widget.importRequested.connect(self.import_model_path)

        self.splitter.addWidget(self.project_widget)
        self.splitter.addWidget(self.data_item_widget)
        # self.splitter.addWidget(self.data_info_widget)

        self.splitter.setSizes([260,740])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 8)
        # self.splitter.setStretchFactor(2, 2)


        self.library_widget = ValidationWorkspace(self)
        self.data_item_widget.modelsChanged.connect(self._on_registry_changed)
        self.project_widget.projectsChanged.connect(self._on_registry_changed)
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
        self.library_widget.toolbarContextChanged.connect(self.toolbarActionsChanged)
        for action in (self.library_widget.import_action, self.library_widget.save_action,
                       self.library_widget.import_files_action, self.library_widget.import_folder_action,
                       self.library_widget.star_action, self.library_widget.import_package_action,
                       self.library_widget.export_package_action, self.library_widget.diagnostic_action, self.library_widget.edit_diagnostic_action, self.library_widget.save_plan_action):
            action.changed.connect(self.toolbarActionsChanged)
