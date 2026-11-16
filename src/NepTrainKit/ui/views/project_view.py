"""Project tree widgets for managing NEP training datasets."""

from PySide6.QtCore import QModelIndex, QTimer, Qt, QPoint, QSize, Signal

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QHeaderView, QHBoxLayout, QLabel, QStyledItemDelegate, QVBoxLayout, QWidget
from qfluentwidgets import (
    Action,
    FluentIcon,
    MessageBox,
    PrimaryPushButton,
    TransparentToolButton,
    RoundMenu,
    TreeView,
)

from NepTrainKit import palette
from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.dataset.services import ProjectItem
from NepTrainKit.ui.messages import translate_runtime_error
from NepTrainKit.ui.widgets import TreeModel, TreeItem, ProjectInfoMessageBox


class _ProjectTreeModel(TreeModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.ToolTipRole:
            project = self.parent_widget.project_item_dict[index.internalPointer().data(1)]
            return "\n".join(filter(None, [f"{project.name}  ·  ID {project.project_id}", project.notes]))
        return super().data(index, role)

    def __init__(self, widget):
        super().__init__(widget)
        self.parent_widget = widget

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if section == 2 and orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return super().headerData(section, orientation, role)


class _ProjectRowDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.font.setWeight(QFont.Weight.DemiBold)
        if index.column() == 2:
            option.displayAlignment = Qt.AlignmentFlag.AlignCenter
            option.palette.setColor(option.palette.ColorRole.Text, QColor(palette.TEXT_SECONDARY))


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
    projectsChanged = Signal()

    def __init__(self, parent=None):
        """Configure the project tree view, model, menus, and load trigger."""
        super().__init__(parent)
        self._parent = parent
        self.project_item_dict = {}

        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._model = _ProjectTreeModel(self)
        self._view.setModel(self._model)
        self._model.setHeader([self.tr("Project name"), "ID", self.tr("Models")])
        self._view.setColumnHidden(1, True)
        self._view.setUniformRowHeights(True)
        self._view.setIconSize(QSize(18, 18))
        self._view.setItemDelegate(_ProjectRowDelegate(self._view))
        header = self._view.header()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 82)
        self._view.setObjectName("projectList")
        self._view.setStyleSheet(f"""
            QTreeView#projectList {{background: white; color: {palette.TEXT_PRIMARY};
                border: 1px solid {palette.BORDER}; border-radius: 8px; outline: none;}}
            QTreeView#projectList::item {{height: 42px; padding: 0 8px; border-radius: 0; border-bottom: 1px solid {palette.BORDER_SUBTLE};}}
            QTreeView#projectList::item:hover {{background: {palette.SURFACE_SUBTLE};}}
            QTreeView#projectList::item:selected {{background: {palette.ACCENT_SOFT}; color: {palette.ACCENT_PRESSED};}}
            QHeaderView::section {{background: {palette.SURFACE_SUBTLE}; color: {palette.TEXT_SECONDARY};
                border: none; border-bottom: 1px solid {palette.BORDER}; padding: 8px;}}
        """)

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(12)
        self._layout.setContentsMargins(16, 14, 16, 14)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        summary = QVBoxLayout()
        summary.setSpacing(1)
        self.project_count = QLabel(self)
        self.project_count.setStyleSheet(f"font-size: 18px; font-weight: 650; color: {palette.TEXT_PRIMARY};")
        self.model_count = QLabel(self)
        self.model_count.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
        summary.addWidget(self.project_count)
        summary.addWidget(self.model_count)
        toolbar.addLayout(summary)
        toolbar.addStretch(1)
        self.modify_project_button = TransparentToolButton(FluentIcon.EDIT, self)
        self.delete_project_button = TransparentToolButton(FluentIcon.DELETE, self)
        for button, caption in ((self.modify_project_button, self.tr("Modify")),
                                (self.delete_project_button, self.tr("Delete"))):
            button.setAccessibleName(caption)
            button.setToolTip(caption)
            toolbar.addWidget(button)
        toolbar.addSpacing(4)
        self.new_project_button = PrimaryPushButton(FluentIcon.ADD, self.tr("New project"), self)
        toolbar.addWidget(self.new_project_button)
        self._layout.addLayout(toolbar)
        self._layout.addWidget(self._view, 1)

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
        excluded = set()
        if modify and index.isValid():
            pending = [self.project_item_dict[index.internalPointer().data(1)]]
            while pending:
                project = pending.pop()
                if project.project_id in excluded:
                    continue
                excluded.add(project.project_id)
                pending.extend(project.children)
        box.parent_combox.addItem(self.tr("Top project"), userData=None)
        for project in self.project_item_dict.values():
            if project.project_id not in excluded:
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

        box.existing_names = {p.name for p in self.project_item_dict.values()
                              if not modify or p.project_id != project_id}
        if not box.exec_():
            return

        name = box.project_name.text().strip()
        note = box.project_note.toPlainText().strip()
        parent_id = box.parent_combox.currentData()

        if modify:
            try:
                self.project_service.modify_project(
                    current_project.project_id,
                    name=name,
                    notes=note,
                    parent_id=parent_id,
                )
            except ValueError as error:
                MessageManager.send_error_message(translate_runtime_error(error))
                return
            self.load_all_projects()
            self.projectsChanged.emit()
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
            self.projectsChanged.emit()

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
        self.projectsChanged.emit()

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
        child = TreeItem((project.name, project.project_id, project.model_num))
        child.icon = FluentIcon.FOLDER.icon(color=palette.ACCENT_PRESSED)

        self.project_item_dict[project.project_id] = project
        parent.appendChild(child)
        for item in project.children:
            self._build_tree(item, child)
        return child

    def load_all_projects(self) -> None:
        """Refresh counts and names without losing the selected or expanded projects."""
        def indexes(parent=QModelIndex()):
            for row in range(self._model.rowCount(parent)):
                index = self._model.index(row, 0, parent)
                yield index
                yield from indexes(index)

        current = self._view.currentIndex()
        selected_id = current.internalPointer().data(1) if current.isValid() else None
        expanded = {index.internalPointer().data(1) for index in indexes() if self._view.isExpanded(index)}
        scroll = self._view.verticalScrollBar().value()
        all_projects = self.project_service.search_projects(parent_id=None)
        self._model.beginResetModel()
        self._model.clear()
        self.project_item_dict.clear()
        for project in all_projects:
            self._build_tree(project, self._model.rootItem)
        self._model.endResetModel()
        count = len(self.project_item_dict)
        versions = sum(project.model_num for project in all_projects)
        self.project_count.setText(self.tr("1 project") if count == 1 else self.tr("{count} projects").format(count=count))
        self.model_count.setText(self.tr("1 model version") if versions == 1 else self.tr("{count} model versions").format(count=versions))
        for index in indexes():
            project_id = index.internalPointer().data(1)
            if project_id in expanded:
                self._view.setExpanded(index, True)
            if project_id == selected_id:
                self._view.setCurrentIndex(index)
        self._view.verticalScrollBar().setValue(scroll)
