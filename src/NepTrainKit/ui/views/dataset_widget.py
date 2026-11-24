"""Widgets for browsing and editing NEP model datasets."""

import os

from PySide6.QtCore import QModelIndex, QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QHeaderView, QMenu, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    Action,
    ComboBox,
    SearchLineEdit,
    StrongBodyLabel,
    TransparentToolButton,
    CaptionLabel,
    FluentIcon,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    RoundMenu,
    TreeView,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.dataset.services import ModelItem, ProjectItem
from NepTrainKit.core.validation.library import VIEWER_DEFAULT_MODEL_OPTION, VIEWER_MODEL_SECTION
from NepTrainKit.ui.widgets.project_registry import RMSE_COLUMNS, RegistryDelegate, RegistryHeader
from NepTrainKit.ui.widgets.validation_model_picker import model_role
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.messages import translate_runtime_error
from NepTrainKit.ui.widgets import (
    AdvancedModelSearchDialog,
    ModelInfoMessageBox,
    TagDelegate,
    TagManageDialog,
    TreeItem,
    TreeModel,
)


class RegistryModel(TreeModel):
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and section == 8 and self.registry.metric_index >= 0:
            col = self.registry.score_columns[self.registry.metric_index]
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                return col["display_label"] + "\n" + col["set_name"]
        if orientation == Qt.Orientation.Horizontal and section in (3, 4, 5):
            if role == Qt.ItemDataRole.ToolTipRole:
                return self.registry.tr("Recorded RMSE")
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignCenter
        return super().headerData(section, orientation, role)

    def __init__(self, registry):
        super().__init__(registry)
        self.registry = registry

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            page = self.registry
            key = index.internalPointer().data(0)
            model = page.model_item_dict[key]
            column = index.column()
            if column in (3, 4, 5):
                value = (model.energy, model.force, model.virial)[column - 3]
                if role == Qt.ItemDataRole.TextAlignmentRole:
                    return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                if role == Qt.ItemDataRole.ForegroundRole and value is None:
                    return QColor(palette.TEXT_MUTED)
                if role == Qt.ItemDataRole.FontRole and value is not None:
                    font = QFont(page._view.font())
                    font.setWeight(QFont.Weight.DemiBold)
                    return font
            if column == 7 and role == Qt.ItemDataRole.ForegroundRole:
                return QColor(palette.TEXT_SECONDARY)
            if role == Qt.ItemDataRole.ToolTipRole:
                if index.column() == 8:
                    return page.coverage_tooltip(key)
                if index.column() == 9:
                    return {"current": page.tr("Open validation"), "baseline": page.tr("Open validation"),
                            "compare": page.tr("Click to remove from comparison"), "hidden": page.tr("Add to comparison")}[page.role(key)]
                values = [("E", model.energy, "meV/atom"), ("F", model.force, "meV/Å"), ("V", model.virial, "meV/atom")]
                metrics = "  ·  ".join(f"{name} {v:.4g} {unit}" for name, v, unit in values if v is not None)
                source = model.accuracy_source
                if metrics and source:
                    group = source.get("group")
                    scope = {"train": page.tr("Training set"), "test": page.tr("Test set")}.get(group, group)
                    status = page.tr("Verified") if source["status"] == "verified" else page.tr("Unverified")
                    metrics += "\n" + " · ".join(filter(None, (scope, status)))
                elif metrics:
                    metrics += "\n" + page.tr("Source not recorded")
                return "\n".join(filter(None, [page.project_names.get(model.project_id), model.name,
                    metrics, model.notes, ", ".join(tag.name for tag in model.tags), str(model.created_at or "")]))
            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 8:
                    return " · ".join(filter(None, page.metric_text(key)))
                if index.column() == 9:
                    return page.role_caption(page.role(key))
        return super().data(index, role)


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
    modelsChanged = Signal(int)
    importRequested = Signal(str)
    validateModelRequested = Signal(int)
    compareModelRequested = Signal(int)
    roleRequested = Signal(int, str)
    manageProjectsRequested = Signal()

    def __init__(self, parent=None):
        """Configure the backing model, view, shortcuts, and context menu."""
        super().__init__(parent)
        self._parent = parent
        self.project: ProjectItem | None = None
        self.model_item_dict = {}

        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.project_names = {}
        self.score_rows = {}
        self.current_id = self.baseline_id = None
        self.comparison_ids = set()
        self._roots = []
        self.metric_index = -1
        self._model = RegistryModel(self)
        self.score_columns = []
        self._view.setHeader(RegistryHeader(Qt.Orientation.Horizontal, self._view))
        self._view.setModel(self._model)
        self._model.setHeader(["ID", self.tr("Model / project"), self.tr("Architecture"),
            *(f"{label} RMSE\n{unit}" for label, unit in RMSE_COLUMNS.values()), self.tr("Tags"), self.tr("Created at"),
            self.tr("Suite results"), self.tr("Comparison role")])
        self._view.setTreePosition(1)
        self._view.setUniformRowHeights(True)
        self._view.setIndentation(14)
        self._view.setColumnHidden(0, True)
        self._view.setItemDelegate(RegistryDelegate(self))
        self._view.setItemDelegateForColumn(6, TagDelegate(self._model))
        header = self._view.header()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(34)
        header.setSectionsClickable(True)
        header.setSortIndicatorClearable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(8, Qt.SortOrder.AscendingOrder)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.moveSection(header.visualIndex(8), header.visualIndex(6))
        for col, width in ((2, 122), (3, 99), (4, 93), (5, 99), (6, 92), (8, 192), (9, 130)):
            header.resizeSection(col, width)
        self._view.setObjectName("modelRegistry")
        self._view.setStyleSheet(f"""
            QTreeView#modelRegistry {{background:white; border:1px solid {palette.BORDER}; border-radius:9px;
                selection-background-color:{palette.ACCENT_SOFT}; selection-color:{palette.TEXT_PRIMARY};}}
            QTreeView#modelRegistry::item {{border-radius:0; border-bottom:1px solid {palette.BORDER_SUBTLE}; padding:0 8px;}}
            QHeaderView::section {{background:{palette.SURFACE_SUBTLE}; color:{palette.TEXT_SECONDARY};
                border:none; border-bottom:1px solid {palette.BORDER}; padding:7px 4px; font-weight:600;}}
        """)

        self.create_menu()
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(10)
        self._layout.setContentsMargins(12, 12, 12, 12)
        panel = QFrame(self)
        panel.setObjectName("registryControls")
        panel.setStyleSheet(f"#registryControls {{background:white; border:1px solid {palette.BORDER}; border-radius:9px;}}")
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(14, 10, 14, 10)
        controls.setSpacing(8)
        toolbar = QHBoxLayout()
        title = StrongBodyLabel(self.tr("Model registry"), panel)
        title.setStyleSheet(f"font-size:18px; font-weight:650; color:{palette.TEXT_PRIMARY};")
        toolbar.addWidget(title)
        self.count_label = CaptionLabel(panel)
        self.count_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        toolbar.addWidget(self.count_label)
        toolbar.addStretch()
        self.manage_projects = PushButton(FluentIcon.FOLDER, self.tr("Manage projects"), panel)
        self.manage_projects.clicked.connect(self.manageProjectsRequested)
        toolbar.addWidget(self.manage_projects)
        self.open_validation = PushButton(FluentIcon.PLAY, self.tr("Open validation"), panel)
        toolbar.addWidget(self.open_validation)
        self.new_model_button = PrimaryPushButton(FluentIcon.ADD, self.tr("New model"), panel)
        toolbar.addWidget(self.new_model_button)
        controls.addLayout(toolbar)
        filters = QHBoxLayout()
        self.project_filter = ComboBox(panel)
        self.project_filter.setAccessibleName(self.tr("Project"))
        self.project_filter.setMaximumWidth(220)
        self.quick_search = SearchLineEdit(panel)
        self.quick_search.setMinimumWidth(90)
        self.quick_search.setPlaceholderText(self.tr("Search project or model"))
        self.quick_search.setAccessibleName(self.quick_search.placeholderText())
        self.quick_search.textChanged.connect(self._filter_rows)
        self.suite_label = CaptionLabel(self.tr("Validation suite"), panel)
        self.suite_filter = ComboBox(panel)
        self.suite_filter.setAccessibleName(self.suite_label.text())
        self.suite_filter.setMaximumWidth(220)
        for combo in (self.suite_filter, self.project_filter):
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        filters.addWidget(self.project_filter, 2)
        filters.addWidget(self.quick_search, 4)
        filters.addWidget(self.suite_label)
        filters.addWidget(self.suite_filter, 2)
        self.metric_button = PushButton(self.tr("Choose metric"), panel)
        self.metric_button.clicked.connect(self._choose_metric)
        self.metric_button.setEnabled(False)
        filters.addWidget(self.metric_button)
        self.more = TransparentToolButton(FluentIcon.MORE, panel)
        self.more.setAccessibleName(self.tr("More actions"))
        self.more.setToolTip(self.more.accessibleName())
        self.more.clicked.connect(lambda: self.menu.exec(self.more.mapToGlobal(self.more.rect().bottomLeft())))
        filters.addWidget(self.more)
        controls.addLayout(filters)
        self._layout.addWidget(panel)
        self._layout.addWidget(self._view, 1)
        self.empty_hint = CaptionLabel(self._view.viewport())
        self.empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        hint_layout = QVBoxLayout(self._view.viewport())
        hint_layout.addWidget(self.empty_hint, 0, Qt.AlignmentFlag.AlignCenter)
        self._model.modelReset.connect(self._refresh_empty_hint)
        self.new_model_button.clicked.connect(lambda: self.create_model(modify=False))
        self._view.selectionModel().currentChanged.connect(self._update_action_state)
        self._view.setExpandsOnDoubleClick(False)
        self._view.doubleClicked.connect(self._open_version)
        self._update_action_state()
        self.search_shortcut = QShortcut(QKeySequence.StandardKey.Find, self, context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.search_shortcut.activated.connect(self.quick_search.setFocus)
        self.role_shortcut = QShortcut(QKeySequence("Space"), self._view)
        self.role_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self.role_shortcut.activated.connect(self._selected_role_menu)
        header.sortIndicatorChanged.connect(lambda *_: self.add_models_to_table(self._roots))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = event.size().width()
        for col, minimum in ((2, 905), (3, 720), (4, 606), (5, 790), (6, 1290), (7, 1120), (8, 480)):
            self._view.setColumnHidden(col, width < minimum)
        self.suite_label.setVisible(width >= 760)
        for button, name in ((self.manage_projects, self.tr("Manage projects")), (self.open_validation, self.tr("Open validation"))):
            button.setText(name if width >= 570 else "")
            button.setAccessibleName(name)
            button.setToolTip(name)

    def _filter_rows(self, *_):
        query = self.quick_search.text().strip().casefold()
        count = 0
        def visit(parent):
            nonlocal count
            visible = False
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                obj = self.model_item_dict[idx.internalPointer().data(0)]
                match = query in (self.project_names.get(obj.project_id, "") + " / " + obj.name).casefold()
                child_match = visit(idx)
                self._view.setRowHidden(row, parent, not (match or child_match))
                if child_match and query:
                    self._view.expand(idx)
                count += int(match)
                visible |= match or child_match
            return visible
        visit(QModelIndex())
        self.count_label.setText(self.tr("1 version") if count == 1 else self.tr("{count} versions").format(count=count))
        self._refresh_empty_hint()
        if query and not count:
            self.empty_hint.setText(self.tr("No matching models"))
            self.empty_hint.show()
        idx = self._current_index()
        if idx.isValid() and self._view.isRowHidden(idx.row(), idx.parent()):
            self._view.setCurrentIndex(QModelIndex())

    def role(self, key):
        return model_role(key, self.current_id, self.baseline_id, self.comparison_ids)

    def role_caption(self, role):
        return {"current": self.tr("Current model"), "baseline": self.tr("Baseline model"),
                "compare": self.tr("In comparison"), "hidden": self.tr("+ Compare")}[role]

    def set_suite_results(self, rows, columns):
        previous = self.score_columns[self.metric_index] if self.metric_index >= 0 else None
        self.score_rows = {r["model_id"]: r for r in rows}
        self.score_columns = columns
        selected = next((i for i, col in enumerate(columns) if previous and
                         (col["set_id"], col["metric"]) == (previous["set_id"], previous["metric"])), None)
        if selected is None:
            selected = max(range(len(columns)), default=-1,
                           key=lambda i: sum(r["cells"][i]["value"] is not None for r in rows
                                             if r["model_id"] in self.model_item_dict))
        self.metric_button.setEnabled(bool(columns))
        self._select_metric(selected)

    def _choose_metric(self):
        menu = QMenu(self)
        sections = {}
        columns = self.score_columns
        for i, col in enumerate(columns):
            if col["set_id"] not in sections:
                sections[col["set_id"]] = menu.addMenu(col["set_name"])
            action = sections[col["set_id"]].addAction(col["display_label"] + "  " + col["unit"])
            action.setCheckable(True)
            action.setChecked(i == self.metric_index)
            action.setData(i)
        def choose(action):
            column = columns[action.data()]
            if column in self.score_columns:
                self._select_metric(self.score_columns.index(column))
        menu.triggered.connect(choose)
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(self.metric_button.mapToGlobal(self.metric_button.rect().bottomLeft()))

    def _select_metric(self, index):
        self.metric_index = index
        self._model.headerDataChanged.emit(Qt.Orientation.Horizontal, 8, 8)
        column = self._view.header().sortIndicatorSection()
        if column in (8, 9):
            show_tree = column == 8 and index < 0
            displayed = [item.data(0) for item in self._model.rootItem.childItems]
            order_changed = not show_tree and displayed != self._sorted_ids(displayed, column)
            if self._view.rootIsDecorated() != show_tree or order_changed:
                self.add_models_to_table(self._roots)
        self._view.viewport().update()

    def metric_cell(self, key):
        row = self.score_rows.get(key)
        if row and self.metric_index >= 0:
            return row["cells"][self.metric_index]
        return {}

    def metric_text(self, key):
        if self.metric_index < 0:
            return "—", ""
        cell = self.metric_cell(key)
        value = cell.get("value")
        if value is None:
            status = cell.get("status")
            text = {"failed": self.tr("Failed"), "unsupported": self.tr("Unsupported"),
                    "succeeded": self.tr("No valid value"), "missing": self.tr("Not run")}
            return text.get(status, self.tr("No record")), ""
        unit = self.score_columns[self.metric_index]["unit"]
        detail = ""
        baseline = self.metric_cell(self.baseline_id).get("value")
        if key == self.baseline_id:
            detail = self.tr("Baseline model")
        elif baseline == value:
            detail = self.tr("Matches baseline")
        elif baseline is not None:
            detail = self.tr("vs baseline {delta}").format(delta=f"{value - baseline:+.4g}")
        return f"{value:.4g} {unit}".strip(), detail

    def coverage_text(self, key):
        row = self.score_rows.get(key)
        if not row or not row["total_sets"]:
            return "—", ""
        done, total = row["completed_sets"], row["total_sets"]
        text = self.tr("{done} / {total} computed").format(done=done, total=total)
        alerts = []
        if row["failed_sets"]:
            alerts.append(self.tr("{count} failed").format(count=row["failed_sets"]))
        if row["regressions"]:
            worse = {col["set_id"] for col, cell in zip(self.score_columns, row["cells"]) if cell.get("verdict") == "worse"}
            alerts.append(self.tr("1 regression") if len(worse) == 1 else self.tr("{count} regressions").format(count=len(worse)))
        if not done and not alerts and all(cell["status"] == "missing" for cell in row["cells"]):
            return self.tr("Not run"), ""
        return text, " · ".join(alerts)

    def coverage_tooltip(self, key):
        row = self.score_rows.get(key)
        if not row:
            return self.suite_filter.currentText()
        lines = [self.suite_filter.currentText(), *filter(None, self.coverage_text(key))]
        if self.metric_index >= 0:
            column = self.score_columns[self.metric_index]
            lines.extend([column["set_name"] + " · " + column["display_label"], *filter(None, self.metric_text(key))])
            if self.metric_cell(key).get("error"):
                lines.append(self.metric_cell(key)["error"])
        shown = set()
        for column, cell in zip(self.score_columns, row["cells"]):
            if len(shown) == 8:
                lines.append(self.tr("Open validation") + "…")
                break
            if column["set_id"] not in shown and cell.get("value") is not None:
                shown.add(column["set_id"])
                lines.append(f"{column['set_name']} · {column['label']}: {cell['value']:.4g} {column['unit']}")
        return "\n".join(lines)

    def activate_role(self, key):
        role = self.role(key)
        if role in ("current", "baseline"):
            self.roleRequested.emit(key, "open")
        else:
            self.roleRequested.emit(key, "hidden" if role == "compare" else "compare")

    def _selected_role_menu(self):
        index = self._current_index()
        if index.isValid():
            self.show_role_menu(self._selected_model_id(), self._view.viewport().mapToGlobal(self._view.visualRect(index).bottomLeft()))

    def show_role_menu(self, key, position):
        menu = QMenu(self)
        role = self.role(key)
        for value, label in (("current", self.tr("Set as current")), ("baseline", self.tr("Set as baseline")),
                             ("compare", self.tr("Add to comparison")), ("hidden", self.tr("Remove from comparison"))):
            action = menu.addAction(label)
            action.setEnabled(value != role and (value != "baseline" or self.suite_filter.currentData() is not None))
            action.triggered.connect(lambda checked=False, v=value: self.roleRequested.emit(key, v))
        menu.exec(position)
        menu.deleteLater()

    def _open_version(self, index):
        if index.column() != 9:
            key = index.internalPointer().data(0)
            if self.role(key) in ("current", "baseline", "compare"):
                self.roleRequested.emit(key, "open")
            else:
                self.validateModelRequested.emit(key)

    def _selected_model_id(self) -> int | None:
        index = self._current_index()
        if not index.isValid():
            return None
        item = index.internalPointer()
        return item.data(0) if item is not None else None

    def send_to_validation(self) -> None:
        model_id = self._selected_model_id()
        if model_id is not None:
            self.validateModelRequested.emit(model_id)

    def add_to_comparison(self) -> None:
        model_id = self._selected_model_id()
        if model_id is not None:
            self.compareModelRequested.emit(model_id)

    def item_clicked(self, index: QModelIndex) -> None:
        """Emit the selected project's identifier when an item is clicked."""
        if not index.isValid():
            return
        item = index.internalPointer()
        model = self.model_item_dict[item.data(0)]
        self.projectChangedSignal.emit(model.project_id)

    def create_menu(self) -> None:
        """Create and wire up the context menu for the model tree."""
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        validate_action = Action(FluentIcon.PLAY, "作为当前势打开验证", self.menu)
        validate_action.triggered.connect(self.send_to_validation)
        self.menu.addAction(validate_action)

        compare_action = Action(FluentIcon.ADD, "加入验证对比槽", self.menu)
        compare_action.triggered.connect(self.add_to_comparison)
        self.menu.addAction(compare_action)
        self.viewer_visibility_action = Action(FluentIcon.VIEW, self.tr("Show in Show NEP"), self.menu)
        self.viewer_visibility_action.triggered.connect(self._toggle_viewer_model_visibility)
        self.menu.addAction(self.viewer_visibility_action)
        self.menu.addSeparator()

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

        self.validate_model_button = validate_action
        self.compare_model_button = compare_action
        self.modify_model_button = modify_action
        self.delete_model_button = delete_action
        self.open_folder_button = open_action
        self.manage_tags_button = tag_action
        self.search_button = Action(self.tr("Search"), self.menu)
        self.search_button.triggered.connect(self.show_search_dialog)
        self.menu.addAction(self.search_button)
        self._view.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, pos: QPoint) -> None:
        """Display the context menu at the requested location."""
        self._menu_pos = pos
        index = self._view.indexAt(pos)
        if index.isValid():
            self._view.setCurrentIndex(index)
        else:
            self._view.clearSelection()
            self._view.setCurrentIndex(QModelIndex())
        self._update_action_state()
        digest = self._viewer_model_digest()
        is_default = bool(digest and digest == Config.get("nep", VIEWER_DEFAULT_MODEL_OPTION))
        self.viewer_visibility_action.setEnabled(bool(digest) and not is_default)
        shown = is_default or bool(digest and Config.getboolean(VIEWER_MODEL_SECTION, digest))
        label = self.tr("Hide from Show NEP") if shown else self.tr("Show in Show NEP")
        self.viewer_visibility_action.setText(label)
        self.menu.exec_(self._view.viewport().mapToGlobal(pos))

    def _viewer_model_digest(self) -> str | None:
        model = self.model_item_dict.get(self._selected_model_id())
        if model is None or any(kind in (model.model_kind or "") for kind in ("dipole", "polarizability")):
            return None
        return model.content_hash

    def _toggle_viewer_model_visibility(self) -> None:
        digest = self._viewer_model_digest()
        if digest is None or digest == Config.get("nep", VIEWER_DEFAULT_MODEL_OPTION):
            return
        if Config.getboolean(VIEWER_MODEL_SECTION, digest):
            Config.delete(VIEWER_MODEL_SECTION, digest)
        else:
            Config.set(VIEWER_MODEL_SECTION, digest, True)

    def _current_index(self):
        """Return the row selected for toolbar or context-menu actions."""
        return self._view.currentIndex()

    def _update_action_state(self, *_args) -> None:
        has_selection = self._current_index().isValid()
        for button in (
            self.validate_model_button,
            self.compare_model_button,
            self.modify_model_button,
            self.open_folder_button,
            self.delete_model_button,
        ):
            button.setEnabled(has_selection)

    def manage_tags(self) -> None:
        """Open the tag management dialog and refresh tag data on close."""
        dlg = TagManageDialog(self.tag_service, self._parent)
        dlg.exec_()
        self.load_models_by_project(self.project)

    def _build_tree(self, model: ModelItem, parent: TreeItem, flat: bool) -> TreeItem:
        """Convert a ModelItem into a TreeItem and attach it to the parent."""
        child = TreeItem(
            (
                model.model_id,
                model.name,
                model.data_size,
                *(f"{v:.4g}" if v is not None else "—" for v in (model.energy, model.force, model.virial)),
                [{"name": tag.name, "color": tag.color} for tag in model.tags],
                model.created_at.strftime("%Y-%m-%d") if model.created_at else "—",
                "", "",
            )
        )

        self.model_item_dict[model.model_id] = model
        parent.appendChild(child)
        for item in model.children:
            self._build_tree(item, parent if flat else child, flat)
        return child

    def _sort_value(self, key, column):
        model = self.model_item_dict[key]
        if column in RMSE_COLUMNS:
            return (model.energy, model.force, model.virial)[column - 3]
        if column == 8:
            return self.metric_cell(key).get("value")
        if column == 1:
            return model.name.casefold()
        if column == 2:
            return (model.model_kind or model.model_type).casefold()
        if column == 6:
            return tuple(tag.name.casefold() for tag in model.tags) or None
        if column == 7:
            return model.created_at
        return ("current", "baseline", "compare", "hidden").index(self.role(key))

    def _sorted_ids(self, ids, column):
        ranked, missing = [], []
        for key in ids:
            value = self._sort_value(key, column)
            if value is None:
                missing.append(key)
            else:
                ranked.append((value, key))
        descending = self._view.header().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
        ranked.sort(key=lambda pair: pair[0], reverse=descending)
        return [key for _, key in ranked] + missing

    def load_models_by_project(self, project: ProjectItem | None) -> None:
        """Refresh the tree with models that belong to the given project."""
        self.project = project
        models = self.model_service.get_models_by_project_id(project.project_id if project else None)
        self.add_models_to_table(models)
        self._update_action_state()

    def _refresh_empty_hint(self) -> None:
        """Show guidance while the model list is empty."""
        if self._model.rowCount() > 0:
            self.empty_hint.hide()
            return
        if getattr(self, "project", None) is not None:
            self.empty_hint.setText(
                self.tr(
                    "No models in this project yet. Use New model, "
                    "or drag a model folder here."
                )
            )
        else:
            self.empty_hint.setText(
                self.tr(
                    "Register a model file or drop a model folder here."
                )
            )
        self.empty_hint.show()

    def add_models_to_table(self, models: list[ModelItem]) -> None:
        """Populate the tree model with the supplied dataset entries."""
        selected = self._selected_model_id()
        self._roots = models
        header = self._view.header()
        column = header.sortIndicatorSection()
        flat = column > 0 and (column != 8 or self.metric_index >= 0)
        self._model.beginResetModel()
        self._model.clear()
        self.model_item_dict.clear()
        for model in models:
            self._build_tree(model, self._model.rootItem, flat)
        if flat:
            root = self._model.rootItem
            by_id = {item.data(0): item for item in root.childItems}
            root.clear()
            for key in self._sorted_ids(by_id, column):
                root.appendChild(by_id[key])
        self._model.endResetModel()
        self._view.setRootIsDecorated(not flat)
        self._view.expandAll()
        self._filter_rows()
        if selected in self.model_item_dict:
            matches = self._model.match(self._model.index(0, 0), Qt.ItemDataRole.DisplayRole, selected, 1,
                                        Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive)
            if matches and not self._view.isRowHidden(matches[0].row(), matches[0].parent()):
                self._view.setCurrentIndex(matches[0].siblingAtColumn(1))

    def create_model(self, modify: bool = False, initial_path: str = "") -> None:
        """Import weights through the shared dialog, or edit saved model metadata."""
        if not modify:
            path = initial_path or call_path_dialog(
                self, self.tr("Select NEP model file"), "select", file_filter="NEP (*.txt);;All files (*)"
            )
            if path:
                self.importRequested.emit(path)
            return
        index = self._current_index()
        if not index.isValid():
            return
        current_model = self.model_item_dict[index.internalPointer().data(0)]
        box = ModelInfoMessageBox(self._parent)
        candidates = {m.model_id: m for m in self.model_service.search_models_advanced(
            project_id=current_model.project_id, include_descendants=False)}
        excluded, pending = set(), [current_model]
        while pending:
            model = pending.pop()
            excluded.add(model.model_id)
            pending.extend(model.children)
        box.parent_combox.addItem(self.tr("Top model"), userData=None)
        for model in candidates.values():
            if model.project_id == current_model.project_id and model.model_id not in excluded:
                box.parent_combox.addItem(f"{model.model_id}-{model.name}", userData=model.model_id)
        box.setWindowTitle(self.tr("Model info"))
        box.model_name_edit.setText(current_model.name)
        box.model_note_edit.setText(current_model.notes)
        box.train_path_edit.setText(current_model.model_path)
        box.model_type_combox.setText(current_model.model_type)
        box.energy_spinBox.setText("" if current_model.energy is None else str(current_model.energy))
        box.force_spinBox.setText("" if current_model.force is None else str(current_model.force))
        box.virial_spinBox.setText("" if current_model.virial is None else str(current_model.virial))
        for tag in current_model.tags:
            box.add_tag(tag.name)
        if current_model.parent_id is not None:
            parent_model = candidates.get(current_model.parent_id)
            if parent_model is None:
                box.parent_combox.addItem(str(current_model.parent_id), userData=current_model.parent_id)
            box.parent_combox.setCurrentIndex(box.parent_combox.findData(current_model.parent_id))
        else:
            box.parent_combox.setCurrentIndex(0)
        if not box.exec_():
            return
        data = box.get_dict()
        data["project_id"] = current_model.project_id
        if data.get("data_size") is None:
            data["data_size"] = current_model.data_size
        try:
            saved = self.model_service.modify_model(current_model.model_id, **data)
        except (OSError, ValueError) as error:
            MessageManager.send_error_message(translate_runtime_error(error))
            return
        self.load_models_by_project(self.project)
        self.modelsChanged.emit(saved.id)
        matches = self._model.match(
            self._model.index(0, 0),
            Qt.ItemDataRole.DisplayRole,
            saved.id,
            1,
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
        )
        if matches:
            self._view.setCurrentIndex(matches[0])
            parent = matches[0].parent()
            while parent.isValid():
                self._view.expand(parent)
                parent = parent.parent()
            self._view.scrollTo(matches[0])
        if saved.id != current_model.model_id:
            MessageManager.send_success_message("模型内容已变化，已创建新版本；旧版本、基准势和历史结果保持不变。")
        else:
            MessageManager.send_success_message(self.tr("Model modified successfully"))

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
        self.modelsChanged.emit(model_id)

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
                folder = os.path.abspath(path)
                if os.path.isfile(folder):
                    folder = os.path.dirname(folder)
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
            else:
                MessageManager.send_info_message(
                    self.tr("Model folder does not exist: {path}").format(path=path)
                )

    def on_search(self, params: dict) -> None:
        """Run an advanced search and refresh the table with the results."""
        models = self.model_service.search_models_advanced(**params)
        matched = {m.model_id: m for m in models if m.project_id is not None}
        for model in matched.values():
            model.children.clear()
        roots = []
        for model in matched.values():
            parent = matched.get(model.parent_id)
            (parent.children if parent else roots).append(model)
        self.project = None
        self.project_filter.blockSignals(True)
        result = self.project_filter.findData("search")
        if result < 0:
            self.project_filter.addItem(self.tr("Search results"), userData="search")
            result = self.project_filter.count() - 1
        self.project_filter.setCurrentIndex(result)
        self.project_filter.blockSignals(False)
        self.add_models_to_table(roots)

    def show_search_dialog(self) -> None:
        """Display the advanced model search dialog and register callbacks."""
        box = AdvancedModelSearchDialog(self._parent)
        box.searchRequested.connect(self.on_search)
        box.show()
