"""Project tree widgets for managing NEP training datasets."""

from PySide6.QtCore import QModelIndex, QTimer, Qt, QPoint, Signal

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from qfluentwidgets import (
    Action,
    FluentIcon,
    MessageBox,
    PushButton,
    RoundMenu,
    TreeView,
)

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.dataset.services import ProjectItem
from NepTrainKit.ui.widgets import TreeModel, TreeItem, ProjectInfoMessageBox

class ProjectWidget(QWidget, DatasetManager):
    """Tree view widget that displays projects and exposes project actions.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget used to anchor dialogs and context menus.

    Attributes
    ----------
    project_item_dict : dict[int, ProjectItem]
        Cache of project objects indexed by identifier.
    projectChangedSignal : Signal
        Emitted with the selected project instance whenever the selection changes.
    """

    project_item_dict: dict[int, ProjectItem] = {}
    projectChangedSignal = Signal(ProjectItem)

    def __init__(self, parent=None):
        """Configure the project tree view, model, menus, and load trigger."""
        super().__init__(parent)
        self._parent = parent

        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.header().setDefaultSectionSize(5)
        self._view.header().setStretchLastSection(True)

        self._model = TreeModel()
        self._view.setModel(self._model)
        self._model.setHeader([self.tr("(ID) Project name"), "ID", ""])
        self._view.setColumnHidden(1, True)
        self._view.setColumnWidth(0, 150)

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 8, 8, 8)
        toolbar.setSpacing(6)
        self.new_project_button = PushButton(self.tr("New project"), self)
        self.modify_project_button = PushButton(self.tr("Modify"), self)
        self.delete_project_button = PushButton(self.tr("Delete"), self)
        toolbar.addWidget(self.new_project_button)
        toolbar.addWidget(self.modify_project_button)
        toolbar.addWidget(self.delete_project_button)
        toolbar.addStretch(1)
        self._layout.addLayout(toolbar)
        self._layout.addWidget(self._view)

        self.create_menu()
        self.new_project_button.clicked.connect(
            lambda: self.create_project(modify=False)
        )
        self.modify_project_button.clicked.connect(
            lambda: self.create_project(modify=True)
        )
        self.delete_project_button.clicked.connect(self.remove_project)
        self._view.selectionModel().currentChanged.connect(self._update_action_state)
        self._update_action_state()
        QTimer.singleShot(1, self.load)

    def item_clicked(self, index):
        """Emit the selected project when a tree row is activated.

        Parameters
        ----------
        index : QModelIndex
            Model index representing the clicked tree node.
        """
        item = index.internalPointer()
        project = self.project_item_dict[item.data(1)]
        self.projectChangedSignal.emit(project)

    def create_menu(self) -> None:
        """Create the context menu for managing projects."""
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        create_action = Action(self.tr("New"), self.menu)
        create_action.triggered.connect(lambda: self.create_project(modify=False))
        self.menu.addAction(create_action)

        modify_action = Action(self.tr("Modify"), self.menu)
        modify_action.triggered.connect(lambda: self.create_project(modify=True))
        self.menu.addAction(modify_action)

        delete_action = Action(self.tr("Delete"), self.menu)
        delete_action.triggered.connect(self.remove_project)
        self.menu.addAction(delete_action)

        self._view.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, pos: QPoint) -> None:
        """Display the project context menu at the requested position.

        Parameters
        ----------
        pos : QPoint
            Location where the context menu should appear.
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
        has_selection = self._current_index().isValid()
        self.modify_project_button.setEnabled(has_selection)
        self.delete_project_button.setEnabled(has_selection)

    def create_project(self, modify: bool = False) -> None:
        """Create a new project or update the currently selected project.

        Parameters
        ----------
        modify : bool, default=False
            When ``True`` the selected project is updated instead of creating a new entry.
        """
        box = ProjectInfoMessageBox(self._parent)
        index = self._current_index()
        box.parent_combox.addItem(self.tr("Top project"), userData=None)
        for project in self.project_item_dict.values():
            box.parent_combox.addItem(project.name, userData=project.project_id)

        if index.row() != -1:
            item = index.internalPointer()
            project_id = item.data(1)
            project = self.project_item_dict[project_id]
            box.parent_combox.setCurrentText(project.name)
        else:
            box.parent_combox.setCurrentIndex(0)
            if modify:
                return
            project_id = None

        box.setWindowTitle(self.tr("Project info"))
        if modify:
            current_project = self.project_item_dict[project_id]
            box.project_name.setText(current_project.name)
            box.project_note.setText(current_project.notes)
            if current_project.parent_id is not None:
                parent_project = self.project_item_dict[current_project.parent_id]
                box.parent_combox.setCurrentText(parent_project.name)
            else:
                box.parent_combox.setCurrentIndex(0)

        if not box.exec_():
            return

        name = box.project_name.text().strip()
        note = box.project_note.toPlainText().strip()
        parent_id = box.parent_combox.currentData()

        if modify:
            self.project_service.modify_project(
                current_project.project_id,
                name=name,
                notes=note,
                parent_id=parent_id,
            )
            self.load_all_projects()
            MessageManager.send_success_message(self.tr("Project modified successfully"))
            return

        project = self.project_service.create_project(
            name=name,
            notes=note,
            parent_id=parent_id,
        )
        if project is None:
            MessageManager.send_error_message(self.tr("Failed to create project"))
        else:
            MessageManager.send_success_message(self.tr("Project created successfully"))
            self.load_all_projects()

    def remove_project(self) -> None:
        """Remove the selected project after user confirmation."""
        index = self._current_index()
        if not index.isValid():
            MessageManager.send_info_message(self.tr("Select a project first"))
            return

        item = index.internalPointer()
        project_id = item.data(1)
        box = MessageBox(
            self.tr("Confirm"),
            self.tr("Do you want to delete this item?\nAll child items will also be deleted."),
            self._parent,
        )
        box.exec_()
        if box.result() == 0:
            return

        self.project_service.remove_project(project_id=project_id)
        MessageManager.send_success_message(self.tr("Project deleted successfully"))
        self.load_all_projects()

    def load(self) -> None:
        """Defer initial loading until the widget is shown."""
        self.load_all_projects()

    def _build_tree(self, project: ProjectItem, parent: TreeItem) -> TreeItem:
        """Create a tree node for the provided project and attach it to the parent.

        Parameters
        ----------
        project : ProjectItem
            Project that will be represented in the tree.
        parent : TreeItem
            Parent tree node that receives the new child.

        Returns
        -------
        TreeItem
            Newly created tree node for the supplied project.
        """
        child = TreeItem((f"({project.project_id}){project.name}", project.project_id, project.model_num))
        child.icon = FluentIcon.FOLDER.icon()

        self.project_item_dict[project.project_id] = project
        parent.appendChild(child)
        for item in project.children:
            self._build_tree(item, child)
        return child

    def load_all_projects(self) -> None:
        """Reload the entire project tree from the data source."""
        self._model.clear()
        all_projects = self.project_service.search_projects(parent_id=None)
        self._model.beginResetModel()
        for project in all_projects:
            self._build_tree(project, self._model.rootItem)
        self._model.endResetModel()
