"""Widgets for browsing and editing NEP model datasets."""

import os

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal, QPoint, QUrl
from PySide6.QtGui import QCursor, QColor, QIcon, QDesktopServices, QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QHeaderView
from qfluentwidgets import (
    Action,
    BodyLabel,
    MessageBox,
    PushButton,
    RoundMenu,
    TreeItemDelegate,
    TreeView,
)

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.dataset.services import ProjectItem, ModelItem
from NepTrainKit.core.types import ModelTypeIcon
from NepTrainKit.ui.widgets import TreeModel, TreeItem, TagDelegate
from NepTrainKit.ui.widgets import ModelInfoMessageBox, AdvancedModelSearchDialog, TagManageDialog


class ModelItemWidget(QWidget, DatasetManager):
    """Tree view widget that lists models grouped by project.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget used to anchor dialogs and menus.

    Attributes
    ----------
    model_item_dict : dict[int, ModelItem]
        Cache of loaded models for quick lookup by identifier.
    projectChangedSignal : Signal
        Emitted with a project identifier when the current selection changes.
    """

    model_item_dict: dict[int, ModelItem] = {}
    projectChangedSignal = Signal(int)

    def __init__(self, parent=None):
        """Configure the backing model, view, shortcuts, and context menu."""
        super().__init__(parent)
        self._parent = parent
        self.project: ProjectItem

        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self._model = TreeModel()
        self._view.setModel(self._model)
        self._model.setHeader([
            "ID",
            self.tr("Name"),
            self.tr("Size"),
            "E(meV/atom)",
            "F(meV/Å)",
            "V(meV/atom)",
            self.tr("Tags"),
            self.tr("Created at"),
        ])
        self._view.setItemDelegateForColumn(6, TagDelegate(self._model))
        header = self._view.header()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        for column in (0, 2, 3, 4, 5, 7):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for column in (1, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        self.create_menu()
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 8, 10, 8)
        toolbar.setSpacing(6)
        toolbar.addWidget(BodyLabel(self.tr("Models"), self))
        toolbar.addStretch(1)
        self.new_model_button = PushButton(self.tr("New model"), self)
        self.modify_model_button = PushButton(self.tr("Modify"), self)
        self.open_folder_button = PushButton(self.tr("Open folder"), self)
        self.delete_model_button = PushButton(self.tr("Delete"), self)
        self.manage_tags_button = PushButton(self.tr("Manage tags"), self)
        self.search_button = PushButton(self.tr("Search"), self)
        for button in (
            self.new_model_button,
            self.modify_model_button,
            self.open_folder_button,
            self.delete_model_button,
            self.manage_tags_button,
            self.search_button,
        ):
            toolbar.addWidget(button)
        self._layout.addLayout(toolbar)
        self._layout.addWidget(self._view)
        self.new_model_button.clicked.connect(
            lambda: self.create_model(modify=False)
        )
        self.modify_model_button.clicked.connect(
            lambda: self.create_model(modify=True)
        )
        self.open_folder_button.clicked.connect(self.open_folder)
        self.delete_model_button.clicked.connect(self.remove_model)
        self.manage_tags_button.clicked.connect(self.manage_tags)
        self.search_button.clicked.connect(self.show_search_dialog)
        self._view.selectionModel().currentChanged.connect(self._update_action_state)
        self._update_action_state()

        self.search_shortcut = QShortcut(
            QKeySequence("Ctrl+F"),
            self,
            context=Qt.ShortcutContext.WindowShortcut,
        )
        self.search_shortcut.activated.connect(self.show_search_dialog)

    def item_clicked(self, index: QModelIndex) -> None:
        """Emit the selected project's identifier when an item is clicked.

        Parameters
        ----------
        index : QModelIndex
            Index provided by the view for the triggered row.
        """
        item = index.internalPointer()
        self.projectChangedSignal.emit(item.data(1))

    def create_menu(self) -> None:
        """Create and wire up the context menu for the model tree."""
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        create_action = Action(self.tr("New"), self.menu)
        create_action.triggered.connect(lambda: self.create_model(modify=False))
        self.menu.addAction(create_action)

        modify_action = Action(self.tr("Modify"), self.menu)
        modify_action.triggered.connect(lambda: self.create_model(modify=True))
        self.menu.addAction(modify_action)

        open_action = Action(self.tr("Open folder"), self.menu)
        open_action.triggered.connect(self.open_folder)
        self.menu.addAction(open_action)

        delete_action = Action(self.tr("Delete"), self.menu)
        delete_action.triggered.connect(self.remove_model)
        self.menu.addAction(delete_action)

        tag_action = Action(self.tr("Manage tags"), self.menu)
        tag_action.triggered.connect(self.manage_tags)
        self.menu.addAction(tag_action)

        self._view.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, pos: QPoint) -> None:
        """Display the context menu at the requested location.

        Parameters
        ----------
        pos : QPoint
            Position in viewport coordinates where the menu is requested.
        """
        self._menu_pos = pos
        index = self._view.indexAt(pos)
        if index.isValid():
            self._view.setCurrentIndex(index)
        else:
            self._view.clearSelection()
            self._view.setCurrentIndex(QModelIndex())
        self._update_action_state()
        self.menu.exec_(self._view.viewport().mapToGlobal(pos))

    def _current_index(self):
        """Return the row selected for toolbar or context-menu actions."""
        return self._view.currentIndex()

    def _update_action_state(self, *_args) -> None:
        has_project = hasattr(self, "project")
        has_selection = self._current_index().isValid()
        self.new_model_button.setEnabled(has_project)
        for button in (
            self.modify_model_button,
            self.open_folder_button,
            self.delete_model_button,
        ):
            button.setEnabled(has_selection)

    def manage_tags(self) -> None:
        """Open the tag management dialog and refresh tag data on close."""
        dlg = TagManageDialog(self.tag_service, self._parent)
        dlg.exec_()

    def _build_tree(self, model: ModelItem, parent: TreeItem) -> TreeItem:
        """Convert a ModelItem into a TreeItem and attach it to the parent.

        Parameters
        ----------
        model : ModelItem
            Model to append to the tree.
        parent : TreeItem
            Parent tree node receiving the model entry.

        Returns
        -------
        TreeItem
            The tree node created for the provided model.
        """
        child = TreeItem(
            (
                model.model_id,
                model.name,
                model.data_size,
                model.energy,
                model.force,
                model.virial,
                [{"name": tag.name, "color": tag.color} for tag in model.tags],
                model.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        child.icon = QIcon(ModelTypeIcon.NEP)

        self.model_item_dict[model.model_id] = model
        parent.appendChild(child)
        for item in model.children:
            self._build_tree(item, child)
        return child

    def load_models_by_project(self, project: ProjectItem) -> None:
        """Refresh the tree with models that belong to the given project.

        Parameters
        ----------
        project : ProjectItem
            Project whose models will be displayed in the tree.
        """
        self._model.clear()
        self.project = project
        self._update_action_state()
        models = self.model_service.get_models_by_project_id(project.project_id)
        self.add_models_to_table(models)

    def add_models_to_table(self, models: list[ModelItem]) -> None:
        """Populate the tree model with the supplied dataset entries.

        Parameters
        ----------
        models : list of ModelItem
            Models that will be appended to the tree model.
        """
        self._model.beginResetModel()
        for model in models:
            self._build_tree(model, self._model.rootItem)
        self._model.endResetModel()

    def create_model(self, modify: bool = False, initial_path: str = "") -> None:
        """Create a new model or update the currently selected one.

        Parameters
        ----------
        modify : bool, default=False
            When ``True`` the selected model is updated; otherwise a new
            version entry is inserted.
        """
        if not hasattr(self, "project"):
            MessageManager.send_info_message(self.tr("Select a project first"))
            return
        box = ModelInfoMessageBox(self._parent)
        index = self._current_index()
        box.parent_combox.addItem(self.tr("Top model"), userData=None)
        for model in self.model_item_dict.values():
            box.parent_combox.addItem(
                f"{model.model_id}-{model.name}",
                userData=model.model_id,
            )

        if index.row() != -1:
            item = index.internalPointer()
            box.parent_combox.setCurrentText(f"{item.data(0)}-{item.data(1)}")
            model_id = item.data(0)
        else:
            box.parent_combox.setCurrentIndex(0)
            if modify:
                return
            model_id = None

        box.setWindowTitle(self.tr("Model info"))
        if initial_path and not modify:
            box.train_path_edit.setText(initial_path)
        if modify:
            current_model = self.model_item_dict[model_id]
            box.model_name_edit.setText(current_model.name)
            box.model_note_edit.setText(current_model.notes)
            box.train_path_edit.setText(current_model.model_path)
            box.model_type_combox.setText(current_model.model_type)
            box.energy_spinBox.setText(str(current_model.energy))
            box.force_spinBox.setText(str(current_model.force))
            box.virial_spinBox.setText(str(current_model.virial))
            for tag in current_model.tags:
                box.add_tag(tag.name)

            if current_model.parent_id is not None:
                parent_model = self.model_item_dict[current_model.parent_id]
                box.parent_combox.setCurrentText(
                    f"{parent_model.model_id}-{parent_model.name}"
                )
            else:
                box.parent_combox.setCurrentIndex(0)

        if not box.exec_():
            return

        data = box.get_dict()
        data["project_id"] = self.project.project_id

        if modify:
            self.model_service.modify_model(current_model.model_id, **data)
            self.load_models_by_project(self.project)
            MessageManager.send_success_message(self.tr("Model modified successfully"))
            return

        project = self.model_service.add_version(**data)
        if project is None:
            MessageManager.send_error_message(self.tr("Failed to create model"))
        else:
            MessageManager.send_success_message(self.tr("Model created successfully"))
            self.load_models_by_project(self.project)

    def remove_model(self) -> None:
        """Delete the currently highlighted model after confirmation."""
        index = self._current_index()

        if not index.isValid():
            MessageManager.send_info_message(self.tr("Select a model first"))
            return

        item = index.internalPointer()
        model_id = item.data(0)
        box = MessageBox(
            self.tr("Confirm"),
            self.tr("Do you want to delete this item?\nAll child items will also be deleted."),
            self._parent,
        )
        box.exec_()
        if box.result() == 0:
            return

        self.model_service.remove_model(model_id=model_id)
        MessageManager.send_success_message(self.tr("Model deleted successfully"))
        self.load_models_by_project(self.project)

    def open_folder(self) -> None:
        """Open the directory or URL associated with the selected model."""
        index = self._current_index()

        if not index.isValid():
            MessageManager.send_info_message(self.tr("Select a model first"))
            return

        item = index.internalPointer()
        model_id = item.data(0)
        model = self.model_item_dict[model_id]
        path = model.model_path
        if path.startswith("http"):
            QDesktopServices.openUrl(QUrl(path))
        else:
            if os.path.exists(path):
                QDesktopServices.openUrl(QUrl("file:///" + path))
            else:
                MessageManager.send_info_message(
                    self.tr("Model folder does not exist: {path}").format(path=path)
                )

    def on_search(self, params: dict) -> None:
        """Run an advanced search and refresh the table with the results.

        Parameters
        ----------
        params : dict
            Filters provided by the advanced search dialog.
        """
        models = self.model_service.search_models_advanced(**params)
        self._model.clear()
        self.add_models_to_table(models)

    def show_search_dialog(self) -> None:
        """Display the advanced model search dialog and register callbacks."""
        box = AdvancedModelSearchDialog(self._parent)
        box.searchRequested.connect(self.on_search)
        box.show()
