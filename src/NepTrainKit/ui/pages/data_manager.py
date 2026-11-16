#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/27 17:18
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget
from sqlalchemy.exc import SQLAlchemyError

from NepTrainKit.config import Config
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
    return list(
        dict.fromkeys(
            model
            for url in mime_data.urls()
            if url.isLocalFile()
            for model in find_model_files(url.toLocalFile())
        )
    )


class _ModelDropPage(QWidget):
    modelsDropped = Signal(list)

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
        self.modelsDropped.emit(paths)


class ValidationPage(_ModelDropPage):
    toolbarActionsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("ValidationPage")
        self.setAcceptDrops(True)
        self.library_widget = ValidationWorkspace(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.library_widget)
        self.library_widget.toolbarContextChanged.connect(self.toolbarActionsChanged)
        for action in (
            self.library_widget.import_action,
            self.library_widget.save_action,
            self.library_widget.import_files_action,
            self.library_widget.import_folder_action,
            self.library_widget.star_action,
            self.library_widget.import_package_action,
            self.library_widget.export_package_action,
            self.library_widget.diagnostic_action,
            self.library_widget.edit_diagnostic_action,
            self.library_widget.save_plan_action,
        ):
            action.changed.connect(self.toolbarActionsChanged)

    def toolbar_actions(self):
        """Expose validation defaults only for the validation page."""
        library = self.library_widget
        editing = not library.editor_scroll.isHidden()
        temporary = library.current_suite_id() is not None and library.current_suite_id() < 0
        return {
            "open": library.import_action,
            "save": library.save_plan_action if editing or temporary else library.save_action,
            "save_menu": library.save_action if temporary else library.export_package_action,
        }

    def _sync_toolbar_actions(self, *_args):
        active = self.isVisible()
        for menu_name, actions in (
            (
                "load_menu",
                (
                    self.library_widget.diagnostic_action,
                    self.library_widget.edit_diagnostic_action,
                    self.library_widget.import_action,
                    self.library_widget.import_files_action,
                    self.library_widget.import_folder_action,
                    self.library_widget.import_package_action,
                ),
            ),
            (
                "save_menu",
                (
                    self.library_widget.save_action,
                    self.library_widget.star_action,
                    self.library_widget.save_plan_action,
                    self.library_widget.export_package_action,
                ),
            ),
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


class DataManagerWidget(_ModelDropPage):
    """Project registry and model handoff to the independent validation page."""

    navigateRequested = Signal(QWidget)

    def __init__(self, parent=None):
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
        self.validation_page = ValidationPage(parent or self)
        self.validation_page.hide()
        self.library_widget = self.validation_page.library_widget
        # Finish the native drop before entering the modal import dialog.
        self.modelsDropped.connect(self._import_models, Qt.ConnectionType.QueuedConnection)
        self.validation_page.modelsDropped.connect(self._import_models, Qt.ConnectionType.QueuedConnection)
        self.init_ui()
        seed_nep_data_git(self._db, self.project_service, self.model_service)

    def import_model_path(self, path):
        paths = find_model_files(path)
        if not paths:
            MessageManager.send_error_message(
                self.tr("No valid NEP model header was found in this path.")
            )
            return
        self._import_models(paths)

    def _import_models(self, paths):
        self.show_projects()
        project = self.data_item_widget.project
        service = ValidationService(get_library())
        dialog = ImportModelDialog(
            self.window(), service, paths, project_name=project.name if project else ""
        )
        dialog.finished.connect(lambda result: self._finish_model_import(dialog, result))
        dialog.open()

    def _finish_model_import(self, dialog, result):
        try:
            if result != QDialog.DialogCode.Accepted:
                return
            record = dialog.save_model(dialog.draft)
        except (OSError, ValueError, SQLAlchemyError) as error:
            MessageManager.send_error_message(self.tr("Import failed: {error}").format(error=error))
            return
        finally:
            dialog.deleteLater()
        self.data_item_widget.project = self.project_service.get_project_by_id(record["project_id"])
        self._on_registry_changed(record["id"])
        MessageManager.send_success_message(self.tr("Model imported: {name}").format(name=record["name"]))

    def show_projects(self):
        """Open the registry and refresh its project list."""
        if self.isVisible():
            self._refresh_projects()
        else:
            self.navigateRequested.emit(self)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_projects()

    def _refresh_projects(self):
        database = get_library().database
        library_changed = database.path != self._db.path
        if library_changed:
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
        view = self.data_item_widget
        self.project_widget.load_all_projects()
        projects = self.project_widget.project_item_dict
        view.project_names = {key: project.name for key, project in projects.items()}
        selected = view.project.project_id if view.project else None
        picker = view.project_filter
        picker.blockSignals(True)
        picker.clear()
        picker.addItem(self.tr("All projects"))
        for key, project in projects.items():
            picker.addItem(project.name, userData=key)
        picker.setCurrentIndex(max(0, picker.findData(selected)))
        picker.blockSignals(False)
        self._select_project(projects.get(selected))

        library = self.library_widget
        suites = library.service().suites()
        picker = view.suite_filter
        preferred = [picker.currentData()] if not library_changed else []
        if library._library_root == library.service().library.root:
            preferred.append(library.current_suite_id())
        if not library_changed:
            preferred.append(Config.getint("validation", "suite", 0))
        picker.blockSignals(True)
        picker.clear()
        for suite in suites:
            picker.addItem(suite["name"], userData=suite["id"])
        if not suites:
            picker.addItem(self.tr("No validation suite"))
        suite_ids = [suite["id"] for suite in suites]
        picker.setCurrentIndex(next((suite_ids.index(key) for key in preferred if key in suite_ids), 0))
        picker.blockSignals(False)
        self._update_suite_summary()

    def _select_project(self, project):
        view = self.data_item_widget
        key = project.project_id if project else None
        view.project_filter.blockSignals(True)
        search = view.project_filter.findData("search")
        if search >= 0:
            view.project_filter.removeItem(search)
        view.project_filter.setCurrentIndex(max(0, view.project_filter.findData(key)))
        view.project_filter.blockSignals(False)
        view.load_models_by_project(project)

    def _update_suite_summary(self, *_):
        view, library = self.data_item_widget, self.library_widget
        suite_id = view.suite_filter.currentData()
        active = (library._loaded and library.current_suite_id() == suite_id
                  and library._library_root == library.service().library.root)
        if active:
            rows, columns = library.model._rows, library.model._columns
        else:
            card = library.service().scorecard(suite_id)
            rows, columns = card["rows"], card["columns"]
        view.baseline_id = next((r["model_id"] for r in rows if r["is_baseline"]), None)
        view.current_id = library.model_combo.currentData() if active else None
        view.comparison_ids = set(library.comparison_picker.selected_ids()) if active else set()
        view.set_suite_results(rows, [dict(c, display_label=library.model.metric_label(c)) for c in columns])

    def _active_scorecard_changed(self):
        if self.isVisible() and self.library_widget.current_suite_id() == self.data_item_widget.suite_filter.currentData():
            self._update_suite_summary()

    def _open_registry_suite(self):
        library = self.library_widget
        suite = self.data_item_widget.suite_filter.currentData()
        if library.current_suite_id() != suite and library._run_task is not None:
            MessageManager.send_info_message(self.tr("Wait for the current validation run to finish."))
            return
        if not library._loaded or library.current_suite_id() != suite or library._library_root != library.service().library.root:
            library.refresh(suite)
        self.navigateRequested.emit(self.validation_page)

    def _set_registry_role(self, key, role):
        library = self.library_widget
        if library._run_task is not None:
            MessageManager.send_info_message(self.tr("Wait for the current validation run to finish."))
            return
        suite = self.data_item_widget.suite_filter.currentData()
        if (not library._loaded or library.current_suite_id() != suite
                or library._library_root != library.service().library.root or library.model.row_of_model(key) < 0):
            library.refresh(suite)
        if library.model.row_of_model(key) < 0:
            MessageManager.send_info_message(library._unavailable_models.get(key) or self.tr("This model is unavailable for this suite."))
            return
        if role == "open":
            self.navigateRequested.emit(self.validation_page)
        else:
            library._choose_model_role(key, role)
            self._update_suite_summary()
        return True

    def open_scorecard(self, suite_id, model_id=None, *, draft=None):
        """Open the validation page with ``suite_id`` and ``model_id`` selected."""
        self.library_widget.open_scorecard(suite_id, model_id, draft=draft)
        self.navigateRequested.emit(self.validation_page)

    def validate_registered_model(self, model_id: int) -> None:
        """Open the selected registered model directly as the Current Model in the validation workbench."""
        if self._set_registry_role(model_id, "current"):
            self.navigateRequested.emit(self.validation_page)

    def compare_registered_model(self, model_id: int) -> None:
        self._set_registry_role(model_id, "compare")

    def _on_registry_changed(self, model_id=None):
        """Keep the project counts and validation selectors in sync after edits."""
        if self.library_widget._run_task is None:
            self.library_widget.refresh()
            reason = getattr(self.library_widget, "_unavailable_models", {}).get(model_id)
            if reason:
                MessageManager.send_info_message("模型已登记，但暂不能用于验证：" + reason)
        self._refresh_projects()

    def init_ui(self):
        """Keep project organization available without taking width from the registry."""
        self.gridLayout = QVBoxLayout(self)
        self.gridLayout.setObjectName("DataManagerWidget_gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.project_dialog = QDialog(self)
        self.project_dialog.setWindowTitle(self.tr("Manage projects"))
        self.project_dialog.resize(520, 360)
        self.project_widget = ProjectWidget(self.project_dialog)
        dialog_layout = QVBoxLayout(self.project_dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(self.project_widget)
        self.project_widget.setObjectName("project_widget")
        self.project_widget.setAutoFillBackground(True)
        self.project_widget.db = self._db
        self.project_widget.project_service = self.project_service
        self.project_widget.model_service = self.model_service
        self.project_widget.tag_service = self.tag_service
        self.data_item_widget = ModelItemWidget(self)
        self.data_item_widget.db = self._db
        self.data_item_widget.project_service = self.project_service
        self.data_item_widget.model_service = self.model_service
        self.data_item_widget.tag_service = self.tag_service

        view = self.data_item_widget
        view.project_filter.currentIndexChanged.connect(lambda _: self._select_project(
            self.project_widget.project_item_dict.get(view.project_filter.currentData())))
        view.suite_filter.currentIndexChanged.connect(self._update_suite_summary)
        view.manageProjectsRequested.connect(self.project_dialog.show)
        view.open_validation.clicked.connect(self._open_registry_suite)
        view.roleRequested.connect(self._set_registry_role)
        self.library_widget.scorecardChanged.connect(self._active_scorecard_changed, Qt.ConnectionType.QueuedConnection)
        self.project_widget.projectChangedSignal.connect(self._select_project)
        self.data_item_widget.importRequested.connect(self.import_model_path)
        self.data_item_widget.validateModelRequested.connect(self.validate_registered_model)
        self.data_item_widget.compareModelRequested.connect(self.compare_registered_model)

        self.gridLayout.addWidget(view)
        self.data_item_widget.modelsChanged.connect(self._on_registry_changed)
        self.project_widget.projectsChanged.connect(self._on_registry_changed)
