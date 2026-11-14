"""Suite navigation and management around the existing validation workbench."""

from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QListWidgetItem,
    QStackedWidget,
    QTableWidgetItem,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CheckBox,
    FluentIcon,
    ListWidget,
    MessageBox,
    PushButton,
    RoundMenu,
    SearchLineEdit,
    TableWidget,
    TreeWidget,
)

from NepTrainKit.core import MessageManager
from NepTrainKit.core.validation.protocols import protocol_for_set
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views.model_library import ModelLibraryWidget, format_metric
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog


class ValidationWorkspace(ModelLibraryWidget):
    """Navigation never mutates the independent set of checked run targets."""

    def __init__(self, parent=None, service=None):
        self._workspace_ready = False
        super().__init__(parent, service)
        self.checked_ids = set()
        self.topic_id = "all"
        self._workspace_suite = None
        self._items = []
        self._topics = []
        self._focused_reference = None
        self._rendering = False
        self._target_models = []

        # Keep the result renderer and diagnostic editor as the detail page.
        root = self.layout()
        detail_page = QWidget(self)
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        while root.count() > 2:
            item = root.takeAt(2)
            if item.widget():
                detail_layout.addWidget(item.widget(), item.stretch() if hasattr(item, "stretch") else 0)
            else:
                detail_layout.addLayout(item.layout())
        detail_layout.setStretch(detail_layout.indexOf(self.splitter), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.editor_scroll), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.curves_placeholder), 1)

        self.sidebar = QWidget(self)
        self.sidebar.setMinimumWidth(170)
        self.sidebar.setMaximumWidth(235)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(10, 8, 6, 8)
        side.addWidget(BodyLabel(self.tr("Validation suites"), self.sidebar))
        fields = self.controls_panel.layout().itemAt(0).layout()
        suite_field = fields.takeAt(0).layout()
        caption = suite_field.takeAt(0).widget()
        caption.hide()
        caption.deleteLater()
        chooser = QHBoxLayout()
        chooser.addWidget(self.suite_combo, 1)
        chooser.addWidget(self.refresh_button)
        side.addLayout(chooser)
        suite_field.deleteLater()
        self.topic_tree = TreeWidget(self.sidebar)
        self.topic_tree.setHeaderHidden(True)
        self.topic_tree.setIndentation(16)
        self.topic_tree.currentItemChanged.connect(lambda node, _: self._browse_topic(node) if node else None)
        side.addWidget(self.topic_tree, 1)
        new_suite = PushButton(FluentIcon.ADD, self.tr("New suite"), self.sidebar)
        new_suite.clicked.connect(self.create_suite)
        side.addWidget(new_suite)
        self.new_suite_button = new_suite
        self.new_topic_button = PushButton(FluentIcon.FOLDER, self.tr("New topic"), self.sidebar)
        self.new_topic_button.clicked.connect(self.create_topic)
        side.addWidget(self.new_topic_button)
        self.existing_button = PushButton(self.tr("Add existing items…"), self.sidebar)
        self.existing_button.clicked.connect(self.add_existing_items)
        side.addWidget(self.existing_button)
        self.organize_button = PushButton(self.tr("Group by source folders"), self.sidebar)
        self.organize_button.clicked.connect(self.organize_sources)
        side.addWidget(self.organize_button)
        for label, callback in [("Rename topic…", self.rename_topic), ("Remove topic", self.remove_topic)]:
            action = Action(self.tr(label), self.topic_tree)
            action.triggered.connect(callback)
            self.topic_tree.addAction(action)
        self.topic_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        main = QWidget(self)
        body = QVBoxLayout(main)
        body.setContentsMargins(6, 4, 10, 8)
        self.breadcrumb = BodyLabel(self)
        body.addWidget(self.breadcrumb)
        navigation = QHBoxLayout()
        self.view_buttons = {}
        for label, key in [
            ("Overview", "overview"),
            ("Validation items", "items"),
            ("Run history", "history"),
            ("Item details", "details"),
        ]:
            button = PushButton(self.tr(label), self)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, page=key: self.show_page(page))
            navigation.addWidget(button)
            self.view_buttons[key] = button
        navigation.addStretch(1)
        body.addLayout(navigation)
        self.pages = QStackedWidget(main)
        body.addWidget(self.pages, 1)
        self.page_widgets = {}
        self.overview = TableWidget(self)
        self.overview.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.overview.cellDoubleClicked.connect(self._overview_open)
        self.overview.itemChanged.connect(self._overview_checked)
        self.item_table = TableWidget(self)
        self.item_table.setColumnCount(7)
        self.item_table.setHorizontalHeaderLabels(
            [
                self.tr(label)
                for label in (
                    "Validation item",
                    "Topic",
                    "Analysis",
                    "Role",
                    "Frames",
                    "Current model metric",
                    "Status",
                )
            ]
        )
        self.item_table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.item_table.setSelectionMode(TableWidget.SelectionMode.ExtendedSelection)
        self.item_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.item_table.itemChanged.connect(self._item_checked)
        self.item_table.itemDoubleClicked.connect(lambda item: self.open_item(item.data(Qt.ItemDataRole.UserRole)))
        self.item_table.itemSelectionChanged.connect(self._management_state)
        self.history_table = TableWidget(self)
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            [self.tr(label) for label in ("Validation item", "Model", "Status", "Recorded at", "Error")]
        )
        self.history_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        for key, widget in [
            ("overview", self.overview),
            ("items", self.item_table),
            ("history", self.history_table),
            ("details", detail_page),
        ]:
            self.page_widgets[key] = widget
            self.pages.addWidget(widget)
        for table in (self.overview, self.item_table, self.history_table):
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            table.verticalHeader().setDefaultSectionSize(36)
        self.context_hint = CaptionLabel(self)
        self.context_hint.setWordWrap(True)
        body.addWidget(self.context_hint)
        tools = QHBoxLayout()
        self.select_visible_button = PushButton(self.tr("Check visible items"), self)
        self.select_visible_button.clicked.connect(self.check_visible)
        self.clear_checked_button = PushButton(self.tr("Clear checks"), self)
        self.clear_checked_button.clicked.connect(self.clear_checks)
        self.manage_button = PushButton(FluentIcon.EDIT, self.tr("Manage selected items"), self)
        self.manage_menu = RoundMenu(parent=self)
        self.manage_actions = {}
        for label, method in [
            ("Open details", self.open_selected_item),
            ("Rename in this suite…", self.rename_item),
            ("Move to topic…", self.move_items),
            ("Edit analysis version…", self.edit_item),
            ("Remove from this suite", self.remove_items),
            ("Archive unshared items", self.archive_items),
        ]:
            action = Action(self.tr(label), self)
            action.triggered.connect(method)
            self.manage_menu.addAction(action)
            self.manage_actions[label] = action
            self.item_table.addAction(action)
        self.item_table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.manage_button.clicked.connect(
            lambda: self.manage_menu.exec(self.manage_button.mapToGlobal(self.manage_button.rect().bottomLeft()))
        )
        self.import_workspace_button = PushButton(FluentIcon.ADD, self.tr("Add reference data…"), self)
        import_menu = RoundMenu(parent=self)
        for action in (
            self.import_action,
            self.import_files_action,
            self.import_folder_action,
            self.import_package_action,
        ):
            import_menu.addAction(action)
        self.import_workspace_button.clicked.connect(
            lambda: import_menu.exec(
                self.import_workspace_button.mapToGlobal(self.import_workspace_button.rect().bottomLeft())
            )
        )
        for widget in (self.select_visible_button, self.clear_checked_button, self.manage_button):
            tools.addWidget(widget)
        tools.addStretch(1)
        tools.addWidget(self.import_workspace_button)
        body.addLayout(tools)
        split = QHBoxLayout()
        split.addWidget(self.sidebar)
        split.addWidget(main, 1)
        root.addLayout(split, 1)
        self.property_combo.hide()
        self.scope_combo.hide()
        self.curves_button.hide()
        self._workspace_ready = True
        self.show_page("overview")

    def tr(self, text, disambiguation=None, n=-1):
        # Inherited controls retain the workbench translations.
        translated = QCoreApplication.translate("ValidationWorkspace", text, disambiguation, n)
        return (
            translated
            if translated != text
            else QCoreApplication.translate("ModelLibraryWidget", text, disambiguation, n)
        )

    def _load_scorecard(self, model_id=None):
        super()._load_scorecard(model_id)
        if self._workspace_ready:
            self._reload_workspace()

    def _reload_workspace(self):
        suite_id = self.current_suite_id()
        context = (str(self.service().library.root), suite_id)
        if context != self._workspace_suite:
            self.checked_ids.clear()
            self.topic_id = "all"
            self._focused_reference = None
            self._workspace_suite = context
            self.show_page("overview")
        self._items = self.service().suite_items(suite_id) if suite_id is not None else []
        self.checked_ids.intersection_update(item["id"] for item in self._items)
        if self._focused_reference not in {item["id"] for item in self._items}:
            self._focused_reference = None
            if self.pages.currentWidget() is self.page_widgets["details"]:
                self.show_page("items")
        self._topics = self.service().suite_groups(suite_id) if suite_id is not None else []
        if self.topic_id not in ("all", None) and not any(t["id"] == self.topic_id for t in self._topics):
            self.topic_id = "all"
        self.topic_tree.blockSignals(True)
        self.topic_tree.clear()
        root = QTreeWidgetItem([self.suite_combo.currentText() + f" · {len(self._items)}"])
        root.setData(0, Qt.ItemDataRole.UserRole, "all")
        self.topic_tree.addTopLevelItem(root)
        entries = [(t["id"], t["name"]) for t in self._topics]
        if any(i["group_id"] is None for i in self._items) or not entries:
            entries.append((None, self.tr("Ungrouped")))
        for key, label in entries:
            count = sum(i["group_id"] == key for i in self._items)
            node = QTreeWidgetItem([f"{label} · {count}"])
            node.setData(0, Qt.ItemDataRole.UserRole, key)
            root.addChild(node)
            if key == self.topic_id:
                self.topic_tree.setCurrentItem(node)
        root.setExpanded(True)
        if self.topic_id == "all":
            self.topic_tree.setCurrentItem(root)
        self.topic_tree.blockSignals(False)
        self._render_workspace()
        self._update_run_scope()

    def _browse_topic(self, node, _column=0):
        self.topic_id = node.data(0, Qt.ItemDataRole.UserRole)
        self._render_workspace()
        self.show_page("overview" if self.topic_id == "all" else "items")

    def visible_items(self):
        return [i for i in self._items if self.topic_id == "all" or i["group_id"] == self.topic_id]

    def selected_benchmarks(self):
        if not self._workspace_ready:
            return super().selected_benchmarks()
        return [i for i in self._items if i["id"] in self.checked_ids]

    def _cell_status(self, model_id, set_id):
        row = self.model.row(self.model.row_of_model(model_id))
        cells = [cell for col, cell in zip(self.model._columns, row["cells"]) if col["set_id"] == set_id] if row else []
        first = cells[0] if cells else {"status": "missing", "error": ""}
        return dict(
            first,
            worse=any(c.get("verdict") == "worse" for c in cells),
            better=any(c.get("verdict") == "better" for c in cells),
        )

    def _render_workspace(self):
        if not self._workspace_ready:
            return
        self._rendering = True
        selected = self.selected_model_ids()
        self._target_models = list(selected)
        suite_id = self.current_suite_id()
        baseline = self.service().get_suite(suite_id).get("baseline_model_id") if suite_id is not None else None
        models = list(dict.fromkeys(selected + ([baseline] if baseline else [])))
        self.overview.blockSignals(True)
        self.overview.setColumnCount(2 + len(models))
        labels = [self.tr("Topic"), self.tr("Items")]
        for model_id in models:
            index = self.model.row_of_model(model_id)
            row = self.model.row(index)
            label = row["name"] if row else self.service().get_model(model_id)["name"]
            labels.append(label + (" · " + self.tr("Baseline") if model_id == baseline else ""))
        self.overview.setHorizontalHeaderLabels(labels)
        self.overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        topics = [(t["id"], t["name"]) for t in self._topics]
        if any(i["group_id"] is None for i in self._items) or not topics:
            topics.append((None, self.tr("Ungrouped")))
        if self.topic_id != "all":
            topics = [t for t in topics if t[0] == self.topic_id]
        self.overview.setRowCount(len(topics))
        for row, (key, label) in enumerate(topics):
            members = [i for i in self._items if i["group_id"] == key]
            title = QTableWidgetItem(label)
            title.setFlags(title.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            count = sum(i["id"] in self.checked_ids for i in members)
            title.setCheckState(
                Qt.CheckState.Checked
                if members and count == len(members)
                else Qt.CheckState.PartiallyChecked
                if count
                else Qt.CheckState.Unchecked
            )
            title.setData(Qt.ItemDataRole.UserRole, key)
            self.overview.setItem(row, 0, title)
            self.overview.setItem(row, 1, QTableWidgetItem(str(len(members))))
            for column, model_id in enumerate(models, 2):
                states = [self._cell_status(model_id, i["id"]) for i in members]
                done = sum(s["status"] == "succeeded" for s in states)
                failed = sum(s["status"] == "failed" for s in states)
                unsupported = sum(s["status"] == "unsupported" for s in states)
                worse = sum(s["worse"] for s in states)
                text = self.tr("Done {done}/{total}").format(done=done, total=len(members))
                if failed:
                    text += " · " + self.tr("{count} failed").format(count=failed)
                if unsupported:
                    text += " · " + self.tr("{count} unsupported").format(count=unsupported)
                if worse:
                    text += " · " + self.tr("{count} regressed").format(count=worse)
                cell = QTableWidgetItem(text)
                cell.setToolTip(
                    "\n".join(i["name"] + ": " + self._status_text(s["status"]) for i, s in zip(members, states))
                )
                self.overview.setItem(row, column, cell)
        self.overview.blockSignals(False)
        self.item_table.blockSignals(True)
        focused = {
            self.item_table.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
            for i in self.item_table.selectionModel().selectedRows()
        }
        self.item_table.clearSelection()
        items = self.visible_items()
        self.item_table.setRowCount(len(items))
        current = self.model_combo.currentData()
        roles = {
            "training": self.tr("Training regression"),
            "development": self.tr("Development validation"),
            "validation": self.tr("Validation (declared)"),
            "unknown": self.tr("Not declared"),
        }
        for row, item in enumerate(items):
            name = QTableWidgetItem(item["name"].rsplit(" / ", 1)[-1])
            name.setData(Qt.ItemDataRole.UserRole, item["id"])
            name.setToolTip(item["name"])
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if item["id"] in self.checked_ids else Qt.CheckState.Unchecked)
            self.item_table.setItem(row, 0, name)
            protocol = protocol_for_set(item)
            summary = item["summary"] or {}
            analysis = (summary.get("analysis") or {}).get("mode")
            mode_labels = {
                "scan": self.tr("Reference scan"),
                "difference": self.tr("State differences"),
                "four_state": self.tr("Four-state response"),
                "cross": self.tr("Cross response"),
                "combination": self.tr("Signed combination"),
                "slices": self.tr("Accuracy by group"),
            }
            role = (summary.get("analysis") or {}).get("role", item.get("provenance", {}).get("role", "unknown"))
            metric = protocol.headline_metrics()[0]
            column = next(
                (
                    n + 1
                    for n, c in enumerate(self.model._columns)
                    if c["set_id"] == item["id"] and c["metric"] == metric.key
                ),
                -1,
            )
            model_row = self.model.row(self.model.row_of_model(current))
            value = model_row["cells"][column - 1]["value"] if model_row and column > 0 else None
            display = (format_metric(value, metric.better) + " " + metric.unit) if value is not None else "—"
            values = [
                item.get("group_name") or self.tr("Ungrouped"),
                mode_labels.get(analysis, protocol.title),
                roles.get(role, role),
                str(summary.get("frames", len(summary.get("volumes", [])))),
                display,
                self._status_text(self._cell_status(current, item["id"])["status"]),
            ]
            for c, value in enumerate(values, 1):
                entry = QTableWidgetItem(str(value))
                entry.setData(Qt.ItemDataRole.UserRole, item["id"])
                entry.setToolTip(str(value))
                self.item_table.setItem(row, c, entry)
            if item["id"] in focused:
                from PySide6.QtCore import QItemSelectionModel

                self.item_table.selectionModel().select(
                    self.item_table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )
        self.item_table.blockSignals(False)
        self._rendering = False
        topic = next((t["name"] for t in self._topics if t["id"] == self.topic_id), self.tr("Ungrouped"))
        self.breadcrumb.setText(
            self.suite_combo.currentText() + " / " + (self.tr("All topics") if self.topic_id == "all" else topic)
        )
        self.context_hint.setText(
            self.tr(
                "{visible} visible · {checked} checked across topics. Viewing details does not change run targets."
            ).format(visible=len(items), checked=len(self.checked_ids))
        )
        self._management_state()
        if self.pages.currentWidget() is self.history_table:
            self._render_history()

    def _status_text(self, status):
        return {
            "missing": self.tr("Not run"),
            "succeeded": self.tr("Computed"),
            "failed": self.tr("Failed"),
            "unsupported": self.tr("Unsupported"),
        }.get(status, status)

    def _item_checked(self, item):
        if self._rendering or item.column() != 0:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.checked_ids.add(key)
        else:
            self.checked_ids.discard(key)
        self._render_workspace()
        self._update_run_scope()

    def _overview_checked(self, item):
        if self._rendering or item.column() != 0:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        for record in self._items:
            if record["group_id"] == key:
                if item.checkState() != Qt.CheckState.Unchecked:
                    self.checked_ids.add(record["id"])
                else:
                    self.checked_ids.discard(record["id"])
        self._render_workspace()
        self._update_run_scope()

    def _overview_open(self, row, _column):
        self.topic_id = self.overview.item(row, 0).data(Qt.ItemDataRole.UserRole)
        root = self.topic_tree.topLevelItem(0)
        for index in range(root.childCount()):
            node = root.child(index)
            if node.data(0, Qt.ItemDataRole.UserRole) == self.topic_id:
                self.topic_tree.setCurrentItem(node)
                break
        self._render_workspace()
        self.show_page("items")

    def check_visible(self):
        self.checked_ids.update(i["id"] for i in self.visible_items())
        self._render_workspace()
        self._update_run_scope()

    def clear_checks(self):
        self.checked_ids.clear()
        self._render_workspace()
        self._update_run_scope()

    def show_page(self, key):
        if not self._workspace_ready:
            return
        if key == "details" and self._focused_reference and not self.curves_button.isChecked():
            self.open_item(self._focused_reference)
            return
        self.view_buttons["details"].setEnabled(key == "details" or self._focused_reference is not None)
        self.pages.setCurrentWidget(self.page_widgets[key])
        for name, button in self.view_buttons.items():
            button.setChecked(name == key)
        if key == "history":
            self._render_history()

    def open_item(self, set_id):
        record = next((i for i in self._items if i["id"] == set_id), None)
        if record is None:
            return
        self._focused_reference = set_id
        self.property_combo.setCurrentIndex(self.property_combo.findData(record["protocol"]))
        self._load_groups(update_detail=False)
        row = next(i for i, r in enumerate(self.metrics_model.benchmarks) if r["id"] == set_id)
        self.table.selectRow(row)
        self._expand_curves()
        self._update_detail()
        self.show_page("details")
        self.breadcrumb.setText(
            self.suite_combo.currentText()
            + " / "
            + (record.get("group_name") or self.tr("Ungrouped"))
            + " / "
            + record["name"].rsplit(" / ", 1)[-1]
        )

    def open_scorecard(self, suite_id, model_id=None, *, draft=None):
        super().open_scorecard(suite_id, model_id, draft=draft, show_curves=False)
        self._reload_workspace()
        self.show_page("overview")

    def _on_model_changed(self, *_args):
        if not self._workspace_ready:
            return super()._on_model_changed(*_args)
        primary = self.model_combo.currentData()
        self._update_comparison_options([key for key in self._target_models if key != primary])
        self._load_groups(update_detail=False)
        self._on_selection_changed()

    def _on_selection_changed(self, *, update_detail=True):
        super()._on_selection_changed(update_detail=update_detail)
        if self._workspace_ready:
            self._render_workspace()

    def _update_run_scope(self, *args):
        super()._update_run_scope(*args)
        if not self._workspace_ready:
            return
        self.property_combo.hide()
        self.scope_combo.hide()
        self.curves_button.hide()
        if self._run_task is None:
            self.run_selected_button.setText(self.tr("Evaluate checked items"))
        # Changing the view/next selection cannot change the frozen active job list.
        self.model_combo.setEnabled(bool(self.model._rows))
        self.comparison_picker.setEnabled(bool(self.comparison_picker._options))
        idle = self._run_task is None
        saved = self.current_suite_id() is not None and self.current_suite_id() > 0
        self.new_suite_button.setEnabled(idle)
        for widget in (self.new_topic_button, self.existing_button, self.organize_button):
            widget.setEnabled(idle and saved)
        self.import_workspace_button.setEnabled(idle)
        self._management_state()

    def _management_ids(self):
        return [
            self.item_table.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
            for i in self.item_table.selectionModel().selectedRows()
        ]

    def _management_state(self):
        if not self._workspace_ready:
            return
        count = len(self._management_ids())
        saved = (self.current_suite_id() or 0) > 0
        idle = self._run_task is None
        node = self.topic_tree.currentItem()
        topic = node.data(0, Qt.ItemDataRole.UserRole) if node else None
        for action in self.topic_tree.actions():
            action.setEnabled(idle and saved and isinstance(topic, int))
        self.manage_button.setEnabled(bool(count) and idle)
        for name, action in self.manage_actions.items():
            action.setEnabled(
                idle
                and count > 0
                and (name == "Open details" or saved)
                and (
                    count == 1 if name in ("Open details", "Rename in this suite…", "Edit analysis version…") else True
                )
            )

    def _run_validation(self, *args, **kwargs):
        task = super()._run_validation(*args, **kwargs)
        if task is not None:
            self.run_summary.show()
        return task

    def _render_history(self):
        if not self._workspace_ready:
            return
        rows = self.service().reference_history([i["id"] for i in self.visible_items()])
        names = {i["id"]: i["name"] for i in self._items}
        models = {m["id"]: m["name"] for m in self.service().models()}
        self.history_table.setRowCount(len(rows))
        for r, run in enumerate(rows):
            values = [
                names[run["set_id"]],
                models.get(run["model_id"], str(run["model_id"])),
                self._status_text(run["status"]),
                run.get("created_at", self.tr("Session only")),
                translate_runtime_message(run.get("error", "")),
            ]
            for c, value in enumerate(values):
                self.history_table.setItem(r, c, QTableWidgetItem(str(value)))

    def _change(self, operation):
        try:
            operation()
        except (OSError, ValueError) as error:
            MessageManager.send_error_message(str(error))
            return False
        self.refresh()
        return True

    def show_diagnostic_editor(self):
        super().show_diagnostic_editor()
        if self._workspace_ready and not self.editor_scroll.isHidden():
            self.show_page("details")

    def _prepare_diagnostic_plan(self, name, experiments, temporary=False):
        super()._prepare_diagnostic_plan(name, experiments, temporary=temporary)
        if self._workspace_ready and self.editor_scroll.isHidden() and self._items:
            self._focused_reference = self._items[0]["id"]
            self.show_page("details")

    def create_suite(self):
        name, ok = QInputDialog.getText(self, self.tr("New suite"), self.tr("Suite name"))
        if ok and name.strip():
            try:
                record = self.service().create_suite(name)
            except Exception as error:
                MessageManager.send_error_message(str(error))
                return
            self.refresh(record["id"])
            self.show_page("overview")

    def create_topic(self):
        name, ok = QInputDialog.getText(self, self.tr("New topic"), self.tr("Topic name"))
        if ok and name.strip():
            self._change(lambda: self.service().create_group(self.current_suite_id(), name))

    def rename_topic(self):
        node = self.topic_tree.currentItem()
        key = node.data(0, Qt.ItemDataRole.UserRole) if node else None
        if not isinstance(key, int):
            return
        topic = next(t for t in self._topics if t["id"] == key)
        name, ok = QInputDialog.getText(self, self.tr("Rename topic"), self.tr("Topic name"), text=topic["name"])
        if ok:
            self._change(lambda: self.service().rename_group(self.current_suite_id(), key, name))

    def remove_topic(self):
        node = self.topic_tree.currentItem()
        key = node.data(0, Qt.ItemDataRole.UserRole) if node else None
        if isinstance(key, int):
            self._change(lambda: self.service().remove_group(self.current_suite_id(), key))

    def open_selected_item(self):
        ids = self._management_ids()
        if len(ids) == 1:
            self.open_item(ids[0])

    def rename_item(self):
        ids = self._management_ids()
        if len(ids) != 1:
            return
        item = next(i for i in self._items if i["id"] == ids[0])
        name, ok = QInputDialog.getText(self, self.tr("Rename in this suite"), self.tr("Item name"), text=item["name"])
        if ok:
            self._change(lambda: self.service().rename_suite_item(self.current_suite_id(), ids[0], name))

    def move_items(self):
        ids = self._management_ids()
        if not ids:
            return
        names = [self.tr("Ungrouped")] + [t["name"] for t in self._topics]
        name, ok = QInputDialog.getItem(self, self.tr("Move to topic"), self.tr("Topic"), names, editable=True)
        if not ok or not name.strip():
            return
        try:
            group = None if name == names[0] else self.service().create_group(self.current_suite_id(), name)["id"]
            self._change(lambda: self.service().assign_group(self.current_suite_id(), ids, group))
        except ValueError as error:
            MessageManager.send_error_message(str(error))

    def remove_items(self):
        ids = self._management_ids()
        if not ids:
            return
        box = MessageBox(
            self.tr("Remove from this suite"),
            self.tr(
                "Remove {count} items? Other suites, source files and historical evaluations are preserved."
            ).format(count=len(ids)),
            self.window(),
        )
        if box.exec():
            self._change(lambda: self.service().remove_from_suite(self.current_suite_id(), ids))

    def archive_items(self):
        ids = self._management_ids()
        if not ids:
            return
        box = MessageBox(
            self.tr("Archive references"),
            self.tr(
                "Archive {count} items and remove their current memberships? References shared by another suite cannot be archived. Snapshots and history are retained."
            ).format(count=len(ids)),
            self.window(),
        )
        if box.exec():
            self._change(lambda: self.service().archive_reference(self.current_suite_id(), ids))

    def organize_sources(self):
        if (self.current_suite_id() or 0) <= 0:
            return

        def organize():
            for item in self._items:
                if item["group_id"] is not None:
                    continue
                source = item.get("provenance", {}).get("original_source_path") or (item.get("source_ref") or {}).get(
                    "path", ""
                )
                name = Path(source).parent.name if source else ""
                if len(name) == 2 and all(c in "0123456789abcdef" for c in name.lower()):
                    name = ""
                if not name and " / " in item["name"]:
                    name = item["name"].split(" / ", 1)[0]
                if name:
                    group = self.service().create_group(self.current_suite_id(), name)
                    self.service().assign_group(self.current_suite_id(), [item["id"]], group["id"])

        self._change(organize)

    def add_existing_items(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Add existing validation items"))
        dialog.resize(700, 520)
        layout = QVBoxLayout(dialog)
        search = SearchLineEdit(dialog)
        search.setPlaceholderText(self.tr("Search names or original source paths"))
        layout.addWidget(search)
        listing = ListWidget(dialog)
        layout.addWidget(listing)
        controls = QHBoxLayout()
        select_all = PushButton(self.tr("Check visible items"), dialog)
        clear = PushButton(self.tr("Clear checks"), dialog)
        controls.addWidget(select_all)
        controls.addWidget(clear)
        controls.addStretch()
        layout.addLayout(controls)

        def filter_items(text):
            for row in range(listing.count()):
                item = listing.item(row)
                item.setHidden(text.casefold() not in item.toolTip().casefold())

        def check_all(checked):
            for row in range(listing.count()):
                item = listing.item(row)
                if not item.isHidden():
                    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

        search.textChanged.connect(filter_items)
        select_all.clicked.connect(lambda: check_all(True))
        clear.clicked.connect(lambda: check_all(False))
        present = {i["id"] for i in self._items}
        for record in self.service().sets(include_archived=True):
            if record["id"] in present:
                continue
            item = QListWidgetItem(record["name"] + (" · " + self.tr("Archived") if record["archived"] else ""))
            item.setToolTip(
                record["name"]
                + "\n"
                + str((record.get("source_ref") or {}).get("path", ""))
                + "\n"
                + str(record.get("provenance", {}).get("original_source_path", ""))
            )
            item.setData(Qt.ItemDataRole.UserRole, record["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            listing.addItem(item)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            ids = [
                listing.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(listing.count())
                if listing.item(i).checkState() == Qt.CheckState.Checked
            ]
            if ids:

                def add():
                    self.service().add_existing_references(
                        self.current_suite_id(), ids, group_id=self.topic_id if isinstance(self.topic_id, int) else None
                    )

                self._change(add)

    def edit_item(self):
        ids = self._management_ids()
        if len(ids) != 1:
            return
        record = next(i for i in self._items if i["id"] == ids[0])
        if record["protocol"] == "diagnostic":
            self.open_item(ids[0])
            self.edit_diagnostic_plan()
            return
        dialog = BenchmarkImportDialog(self.window(), self.service().suites(), self.suite_combo.currentText())
        dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData(record["protocol"]))
        dialog.load_file(self.service().library.open_object(record["object_hash"]))
        dialog.suite_combo.setEnabled(False)
        dialog.name_edit.setText(record["name"])
        dialog.notes_edit.setText(record.get("provenance", {}).get("notes", ""))
        if record["protocol"] == "response":
            dialog.analysis_editor.set_settings(record["source_ref"]["analysis"])
        if record["protocol"] == "eos":
            group = record["source_ref"].get("eos_group")
            if group:
                dialog.protocol_combo.setEnabled(False)
                dialog.protocol_combo.setToolTip(
                    self.tr(
                        "This item selects an EOS subgroup. Keep its protocol when editing; create a new reference item for a different analysis."
                    )
                )
            dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData(group["field"] if group else ""))
            if group:
                for row, data in enumerate(dialog.eos_groups):
                    dialog.group_table.item(row, 0).setCheckState(
                        Qt.CheckState.Checked if data["key"] == group["value"] else Qt.CheckState.Unchecked
                    )
        dialog.yesButton.setText(self.tr("Save analysis version"))
        dialog.provenance_extra = {k: v for k, v in record.get("provenance", {}).items() if k != "notes"} | {
            "derived_from_set": record["id"],
            "revision_id": uuid.uuid4().hex,
            "original_source_path": record.get("provenance", {}).get("original_source_path")
            or record["source_ref"].get("path", ""),
        }
        if not dialog.exec():
            return
        try:
            replacements = dialog.import_into(self.service(), self._confirm_large_copy)
            if not replacements:
                return
            new = replacements[0]
            # The importer adds to the suite; replace the old link in its original slot.
            if new["id"] != record["id"]:
                self.service().remove_from_suite(self.current_suite_id(), [r["id"] for r in replacements])
                self.service().replace_suite_item(self.current_suite_id(), record["id"], new["id"])
                self.service().rename_suite_item(self.current_suite_id(), new["id"], new["name"])
                self.service().add_to_suite(self.current_suite_id(), [r["id"] for r in replacements[1:]])
                if len(replacements) > 1:
                    self.service().assign_group(
                        self.current_suite_id(), [r["id"] for r in replacements[1:]], record["group_id"]
                    )
            self.refresh()
            self.show_page("items")
        except (OSError, ValueError) as error:
            MessageManager.send_error_message(str(error))
