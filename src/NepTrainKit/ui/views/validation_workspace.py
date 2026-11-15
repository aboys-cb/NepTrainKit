"""Suite navigation and management around the existing validation workbench."""

from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QScrollArea,
    QBoxLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QListWidgetItem,
    QStackedWidget,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    ComboBox,
    SegmentedWidget,
    CaptionLabel,
    FluentIcon,
    ListWidget,
    MessageBox,
    PushButton,
    RoundMenu,
    SearchLineEdit,
    TransparentPushButton,
    TableWidget,
    TransparentToolButton,
)

from NepTrainKit import palette
from NepTrainKit.core import MessageManager
from NepTrainKit.core.validation.protocols import protocol_for_set
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views.model_library import VERDICT_COLORS, ModelLibraryWidget, format_metric
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog


class ComparisonTable(TableWidget):
    """The run column has its own hit target, independent of row selection."""

    def _run_cell(self, event):
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.column() == self.columnCount() - 1:
            return item
        return None

    def mousePressEvent(self, event):
        if self._run_cell(event) is None:
            super().mousePressEvent(event)
        else:
            event.accept()

    def mouseReleaseEvent(self, event):
        item = self._run_cell(event)
        if item is None:
            super().mouseReleaseEvent(event)
        else:
            item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)
            event.accept()


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
        self._search_filter = ""

        # Enable adaptive plot sizing and the right-hand multi-model quantitative inspector in the workspace
        self.detail.set_adaptive_layout(True)

        # Reuse the existing plot and inspector below the comparison matrix.
        root = self.layout()
        detail_page = QFrame(self)
        detail_page.setObjectName("comparisonDetails")
        detail_page.setStyleSheet(f"#comparisonDetails {{background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px;}}")
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(12, 8, 12, 6)
        detail_layout.setSpacing(6)

        while root.count() > 2:
            item = root.takeAt(2)
            if item.widget():
                detail_layout.addWidget(item.widget(), item.stretch() if hasattr(item, "stretch") else 0)
            else:
                detail_layout.addLayout(item.layout())
        detail_layout.setStretch(detail_layout.indexOf(self.splitter), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.editor_scroll), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.curves_placeholder), 1)

        # Compact, structured left sidebar without stacked full-width bottom buttons
        self.sidebar = QFrame(self)
        self.sidebar.setObjectName("validationSidebar")
        self.sidebar.setStyleSheet(
            f"#validationSidebar {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}"
        )
        self.sidebar.setMinimumWidth(205)
        self.sidebar.setMaximumWidth(260)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(10, 8, 10, 8)
        side.setSpacing(6)

        side_header = QHBoxLayout()
        side_title = BodyLabel(self.tr("Validation suites"), self.sidebar)
        side_font = side_title.font()
        side_font.setBold(True)
        side_title.setFont(side_font)
        side_header.addWidget(side_title, 1)
        new_suite = TransparentToolButton(FluentIcon.ADD.colored(palette.ACCENT, palette.ACCENT), self.sidebar)
        new_suite.setToolTip(self.tr("New suite"))
        new_suite.setAccessibleName(self.tr("New suite"))
        new_suite.setFixedSize(26, 26)
        new_suite.clicked.connect(self.create_suite)
        self.new_suite_button = new_suite
        side_header.addWidget(new_suite)

        self.new_topic_button = TransparentToolButton(
            FluentIcon.FOLDER_ADD.colored(palette.ACCENT, palette.ACCENT), self.sidebar
        )
        self.new_topic_button.setToolTip(self.tr("New topic"))
        self.new_topic_button.setAccessibleName(self.tr("New topic"))
        self.new_topic_button.setFixedSize(26, 26)
        self.new_topic_button.clicked.connect(self.create_topic)
        side_header.addWidget(self.new_topic_button)
        side.addLayout(side_header)

        fields = self.controls_panel.layout().itemAt(0).layout()
        suite_field = fields.takeAt(0).layout()
        caption = suite_field.takeAt(0).widget()
        caption.hide()
        caption.deleteLater()
        chooser = QHBoxLayout()
        chooser.setSpacing(4)
        chooser.addWidget(self.suite_combo, 1)
        chooser.addWidget(self.refresh_button)
        side.addLayout(chooser)
        suite_field.deleteLater()

        self.sidebar_search = SearchLineEdit(self.sidebar)
        self.sidebar_search.setPlaceholderText("搜索主题或验证项…")
        self.sidebar_search.textChanged.connect(self._on_sidebar_search)
        side.addWidget(self.sidebar_search)

        self.topic_picker = ComboBox(self.sidebar)
        self.topic_picker.setAccessibleName(self.tr("Topic"))
        self.topic_picker.currentIndexChanged.connect(self._browse_topic)
        side.insertWidget(2, self.topic_picker)

        self.sidebar_item_list = ComparisonTable(self.sidebar)
        self.sidebar_item_list.setColumnCount(2)
        self.sidebar_item_list.setHorizontalHeaderLabels([self.tr("Validation item"), "✓"])
        self.sidebar_item_list.horizontalHeader().moveSection(1, 0)
        self.sidebar_item_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sidebar_item_list.setColumnWidth(1, 34)
        self.sidebar_item_list.verticalHeader().hide()
        self.sidebar_item_list.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.sidebar_item_list.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.sidebar_item_list.setSelectionMode(TableWidget.SelectionMode.SingleSelection)
        self.sidebar_item_list.currentCellChanged.connect(self._navigate_item)
        self.sidebar_item_list.itemChanged.connect(self._navigator_checked)
        side.addWidget(self.sidebar_item_list, 1)

        # Compact 2-button bottom row instead of 4 tall full-width buttons
        self.existing_button = PushButton(self.tr("Add existing items…"), self.sidebar)
        self.existing_button.setToolTip(self.tr("Add existing items…"))
        self.existing_button.clicked.connect(self.add_existing_items)
        self.organize_button = PushButton(self.tr("Group by source folders"), self.sidebar)
        self.organize_button.setToolTip(self.tr("Group by source folders"))
        self.organize_button.clicked.connect(self.organize_sources)
        self.existing_button.setText(self.tr("Add existing items…"))
        self.existing_button.hide()
        self.organize_button.hide()
        self.suite_tools = QMenu(self.sidebar)
        for button in (new_suite, self.new_topic_button, self.existing_button, self.organize_button):
            action = self.suite_tools.addAction(button.toolTip(), button.click)
            button.hide()
            self.suite_tools.aboutToShow.connect(lambda a=action, b=button: a.setEnabled(b.isEnabled()))
        self.topic_actions = []
        for label, callback in (("Rename topic…", self.rename_topic), ("Remove topic", self.remove_topic)):
            self.topic_actions.append(self.suite_tools.addAction(self.tr(label), callback))
        suite_tools_button = TransparentToolButton(FluentIcon.MORE, self.sidebar)
        suite_tools_button.setAccessibleName(self.tr("Actions"))
        suite_tools_button.clicked.connect(lambda: self.suite_tools.exec(
            suite_tools_button.mapToGlobal(suite_tools_button.rect().bottomLeft())))
        side_header.addWidget(suite_tools_button)

        main = QWidget(self)
        self.main_panel = main
        body = QVBoxLayout(main)
        body.setContentsMargins(6, 4, 12, 8)
        body.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self.view_picker = SegmentedWidget(main)
        for label, key in (("Results", "result"), ("Model comparison", "items"),
                           ("Overview", "overview"), ("Run history", "history")):
            self.view_picker.addItem(key, self.tr(label), lambda page=key: self.show_page(page))
        top_bar.addWidget(self.view_picker)
        top_bar.addStretch()
        self.pane_button = PushButton(self.tr("Physical metrics"), main)
        self.pane_button.setCheckable(True)
        self.pane_button.toggled.connect(lambda _: self._present_workspace())
        top_bar.addWidget(self.pane_button)
        body.addLayout(top_bar)

        self.pages = QStackedWidget(main)
        body.addWidget(self.pages, 1)
        self.page_widgets = {}
        self.overview = TableWidget(self)
        self.overview.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.overview.cellDoubleClicked.connect(self._overview_open)
        self.overview.itemChanged.connect(self._overview_checked)
        self.item_table = ComparisonTable(self)
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
        self.item_table.cellClicked.connect(self._compare_cell)
        self.item_table.cellActivated.connect(self._compare_cell)
        self.item_table.itemSelectionChanged.connect(self._management_state)
        self.history_table = TableWidget(self)
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            [self.tr(label) for label in ("Validation item", "Model", "Status", "Recorded at", "Error")]
        )
        self.history_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        matrix_card = QFrame(self)
        matrix_card.setObjectName("validationMatrix")
        matrix_card.setStyleSheet(f"#validationMatrix {{background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px;}}")
        matrix_layout = QVBoxLayout(matrix_card)
        matrix_layout.setContentsMargins(14, 9, 14, 6)
        self.matrix_title = BodyLabel(matrix_card)
        font = self.matrix_title.font()
        font.setPointSize(15)
        font.setBold(True)
        self.matrix_title.setFont(font)
        matrix_layout.addWidget(self.matrix_title)
        matrix_layout.addWidget(self.item_table)
        self.matrix_card = matrix_card
        self.detail_scroll = QScrollArea(self)
        self.detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setWidget(detail_page)
        for key, widget in [
            ("overview", self.overview),
            ("items", matrix_card),
            ("result", self.detail_scroll),
            ("history", self.history_table),
        ]:
            self.page_widgets[key] = widget
            self.pages.addWidget(widget)
        for table in (self.overview, self.item_table, self.history_table):
            table.setAlternatingRowColors(False)
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            table.verticalHeader().setDefaultSectionSize(36)

        self.footer_container = QWidget(self)
        footer_layout = QHBoxLayout(self.footer_container)
        footer_layout.setContentsMargins(0, 2, 0, 0)
        footer_layout.setSpacing(8)
        self.context_hint = CaptionLabel(self.footer_container)
        self.context_hint.setWordWrap(True)
        self.context_hint.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")

        self.select_visible_button = PushButton(self.tr("Check visible items"), self.footer_container)
        self.select_visible_button.clicked.connect(self.check_visible)
        self.clear_checked_button = PushButton(self.tr("Clear checks"), self.footer_container)
        self.clear_checked_button.clicked.connect(self.clear_checks)
        self.manage_button = PushButton(FluentIcon.EDIT, self.tr("Manage selected items"), self.footer_container)
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
        self.import_workspace_button = PushButton(
            FluentIcon.ADD, self.tr("Add reference data…"), self.footer_container
        )
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
            footer_layout.addWidget(widget)
        self.bulk_menu = QMenu(self)
        self.bulk_menu.addAction(self.select_visible_button.text(), self.check_visible)
        self.bulk_menu.addAction(self.clear_checked_button.text(), self.clear_checks)
        self.bulk_menu.addSeparator()
        self.bulk_menu.addActions(list(self.manage_actions.values()))
        self.bulk_button = PushButton(self.tr("Actions"), self.footer_container)
        self.bulk_button.clicked.connect(lambda: self.bulk_menu.exec(
            self.bulk_button.mapToGlobal(self.bulk_button.rect().bottomLeft())))
        footer_layout.addWidget(self.bulk_button)
        footer_layout.addWidget(self.context_hint, 1)
        footer_layout.addWidget(self.import_workspace_button)
        body.addWidget(self.footer_container)

        self.workspace_split = QSplitter(Qt.Orientation.Horizontal, self)
        self.workspace_split.setChildrenCollapsible(False)
        self.workspace_split.setHandleWidth(5)
        self.workspace_split.addWidget(self.sidebar)
        self.workspace_split.addWidget(main)
        self.workspace_split.setStretchFactor(1, 1)
        self.workspace_split.setSizes([218, 960])
        self.workspace_split.splitterMoved.connect(lambda *_: self._present_workspace())
        root.addWidget(self.workspace_split, 1)
        self.capsule_panel.hide()
        self.tolerance_label.hide()
        self.property_combo.hide()
        self.scope_combo.hide()
        self.curves_button.hide()
        self._workspace_ready = True
        self.baseline_label.hide()
        self.model_fields = fields
        self.model_options = QWidget(self.controls_panel)
        options = QHBoxLayout(self.model_options)
        options.setContentsMargins(0, 0, 0, 0)
        while fields.count():
            options.addLayout(fields.takeAt(0).layout(), 1)
        self.model_summary = TransparentPushButton(self.controls_panel)
        self.model_summary.setCheckable(True)
        self.model_summary.clicked.connect(lambda _: self._adapt_window())
        fields.addWidget(self.model_summary)
        fields.addWidget(self.model_options, 1)
        run_controls = QHBoxLayout()
        run_controls.addStretch()
        run_controls.addWidget(self.scope_hint)
        run_controls.addWidget(self.run_selected_button)
        fields.addLayout(run_controls)
        self.sidebar.setMinimumWidth(164)
        self.sidebar.setMaximumWidth(420)
        self.setMinimumWidth(500)
        self._compact = None
        self._adapt_window()
        self.controls_panel.layout().setSpacing(0)
        self.controls_panel.layout().setContentsMargins(14, 9, 14, 9)
        self.show_page("result")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_compact"):
            self._adapt_window()

    def _adapt_window(self):
        self._compact = self.width() < 1050
        self.model_summary.setVisible(self._compact)
        self.model_options.setVisible(not self._compact or self.model_summary.isChecked())
        self.model_fields.setDirection(QBoxLayout.Direction.TopToBottom if self._compact and self.model_summary.isChecked() else QBoxLayout.Direction.LeftToRight)
        self._present_workspace()

    def _present_workspace(self):
        if not hasattr(self, "_compact"):
            return
        result = self.pages.currentWidget() is self.detail_scroll
        editing = not self.editor_scroll.isHidden()
        focused = bool(self._focused_reference and self.curves_button.isChecked())
        self.sidebar_item_list.setVisible(result)
        self.sidebar.layout().setAlignment(Qt.AlignmentFlag.AlignTop if not result else Qt.AlignmentFlag(0))
        self.bulk_button.show()
        for button in (self.select_visible_button, self.clear_checked_button, self.manage_button):
            button.hide()
        self.footer_container.setVisible(not editing)
        self.model_summary.setText((self.model_combo.currentText() or self.tr("Current model")) + "  ▾")
        self.scope_hint.setVisible(self.width() >= 760)
        narrow = self.width() - self.sidebar.width() < 880
        self.pane_button.setVisible(result and focused and narrow and not editing)
        metrics = narrow and self.pane_button.isChecked()
        self.pane_button.setText(self.tr("Show curve") if metrics else self.tr("Physical metrics"))
        self.detail.plot_pane.setVisible(not metrics)
        self.detail.inspector_panel.setVisible(focused and (not narrow or metrics))

    def _navigate_item(self, row, column, *_):
        if self._rendering or row < 0 or column != 0:
            return
        item = self.sidebar_item_list.item(row, 0)
        if item is not None:
            self.open_item(item.data(Qt.ItemDataRole.UserRole))

    def _navigator_checked(self, item):
        if self._rendering or item.column() != 1:
            return
        self.item_table.item(item.row(), self.check_column).setCheckState(item.checkState())

    def tr(self, text, disambiguation=None, n=-1):
        # Inherited controls retain the workbench translations.
        translated = QCoreApplication.translate("ValidationWorkspace", text, disambiguation, n)
        return (
            translated
            if translated != text
            else QCoreApplication.translate("ModelLibraryWidget", text, disambiguation, n)
        )

    def _on_sidebar_search(self, text: str) -> None:
        self._search_filter = text.strip().casefold()
        self._render_workspace()
        self._focus_first_visible()

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
            self.pages.setCurrentWidget(self.detail_scroll)
            self.view_picker.setCurrentItem("result")
        self._items = self.service().suite_items(suite_id) if suite_id is not None else []
        self.checked_ids.intersection_update(item["id"] for item in self._items)
        if self._focused_reference not in {item["id"] for item in self._items}:
            self._focused_reference = None
        self._topics = self.service().suite_groups(suite_id) if suite_id is not None else []
        if self.topic_id not in ("all", None) and not any(t["id"] == self.topic_id for t in self._topics):
            self.topic_id = "all"
        self.topic_picker.blockSignals(True)
        self.topic_picker.clear()
        self.topic_picker.addItem(self.tr("All topics"), userData="all")
        for group in self._topics:
            self.topic_picker.addItem(group["name"], userData=group["id"])
        if any(item["group_id"] is None for item in self._items):
            self.topic_picker.addItem(self.tr("Ungrouped"), userData=None)
        self.topic_picker.setCurrentIndex(self.topic_picker.findData(self.topic_id))
        self.topic_picker.blockSignals(False)
        self._render_workspace()
        self._update_run_scope()
        self._focus_first_visible()

    def _focus_first_visible(self):
        if self.pages.currentWidget() is self.detail_scroll and not self._focused_reference and self.editor_scroll.isHidden():
            items = self.visible_items()
            if items:
                self.open_item(items[0]["id"])

    def _browse_topic(self, index):
        if index < 0:
            return
        self.topic_id = self.topic_picker.currentData()
        self._render_workspace()
        self._focus_first_visible()

    def visible_items(self):
        items = [i for i in self._items if self.topic_id == "all" or i["group_id"] == self.topic_id]
        if self._search_filter:
            items = [
                i
                for i in items
                if self._search_filter in i["name"].casefold()
                or self._search_filter in (i.get("group_name") or "").casefold()
                or self._search_filter in i.get("protocol", "").casefold()
            ]
        return items

    def selected_benchmarks(self):
        if not self._workspace_ready:
            return super().selected_benchmarks()
        return [i for i in self._items if i["id"] in self.checked_ids]

    def _cell_status(self, model_id, set_id):
        row = self.model.row(self.model.row_of_model(model_id))
        cells = (
            [cell for col, cell in zip(self.model._columns, row["cells"]) if col["set_id"] == set_id]
            if row
            else []
        )
        first = cells[0] if cells else {"status": "missing", "error": ""}
        return dict(
            first,
            worse=any(c.get("verdict") == "worse" for c in cells),
            better=any(c.get("verdict") == "better" for c in cells),
        )

    def _headline_display_for_model(self, model_id, item):
        """Return (formatted_text, cell_dict) for a model on a validation item."""
        protocol = protocol_for_set(item)
        metric = protocol.headline_metrics()[0]
        column = next(
            (
                n + 1
                for n, c in enumerate(self.model._columns)
                if c["set_id"] == item["id"] and c["metric"] == metric.key
            ),
            -1,
        )
        model_row = self.model.row(self.model.row_of_model(model_id))
        cell = model_row["cells"][column - 1] if model_row and column > 0 else None
        value = cell["value"] if cell and cell.get("status") == "succeeded" else None
        if value is not None:
            display = format_metric(value, metric.better) + (" " + metric.unit if metric.unit else "")
            if cell.get("verdict") == "better":
                display += " ↓改善"
            elif cell.get("verdict") == "worse":
                display += " ↑退化"
            return display, cell
        status = (cell or {}).get("status", "missing")
        return ("—" if status == "missing" else self._status_text(status)), cell

    def _render_workspace(self):
        if not self._workspace_ready or self._rendering:
            return
        self._rendering = True
        selected = self.selected_model_ids()
        self._target_models = list(selected)
        suite_id = self.current_suite_id()
        baseline = self.service().get_suite(suite_id).get("baseline_model_id") if suite_id is not None else None
        models = list(dict.fromkeys(selected + ([baseline] if baseline else [])))
        self.overview.blockSignals(True)
        self.overview.clearContents()
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
                better = sum(s["better"] for s in states)
                text = self.tr("Done {done}/{total}").format(done=done, total=len(members))
                if failed:
                    text += " · " + self.tr("{count} failed").format(count=failed)
                if unsupported:
                    text += " · " + self.tr("{count} unsupported").format(count=unsupported)
                if worse:
                    text += " · " + self.tr("{count} regressed").format(count=worse)
                elif better:
                    text += f" · {better} 改善"
                # Append headline metric preview when a single item or uniform metric exists in the topic
                if len(members) == 1 and done == 1:
                    m_text, _ = self._headline_display_for_model(model_id, members[0])
                    if m_text and m_text != "—":
                        text += f" ({m_text})"
                cell = QTableWidgetItem(text)
                if worse:
                    cell.setBackground(VERDICT_COLORS["worse"])
                elif better and done == len(members):
                    cell.setBackground(VERDICT_COLORS["better"])
                cell.setToolTip(
                    "\n".join(
                        i["name"] + ": " + self._status_text(s["status"]) for i, s in zip(members, states)
                    )
                )
                self.overview.setItem(row, column, cell)
        self.overview.blockSignals(False)

        # Multi-model item_table: keep the 7 base columns for contract compatibility,
        # and append columns for each comparison/baseline model so all models are compared side-by-side!
        current = self.model_combo.currentData()
        extra_models = [mid for mid in models if mid != current and mid is not None]
        base_headers = [
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
        extra_headers = []
        for mid in extra_models:
            idx = self.model.row_of_model(mid)
            m_row = self.model.row(idx)
            m_name = m_row["name"] if m_row else self.service().get_model(mid)["name"]
            if mid == baseline:
                m_name += " (基准)"
            extra_headers.append(m_name)

        self.item_table.blockSignals(True)
        focused = {
            self.item_table.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
            for i in self.item_table.selectionModel().selectedRows()
            if self.item_table.item(i.row(), 0) is not None
        }
        self.item_table.clearSelection()
        self.item_table.clearContents()
        self.check_column = 7 + len(extra_models)
        self.item_table.setColumnCount(self.check_column + 1)
        if current is not None:
            base_headers[5] = self.model_combo.currentText().split(" / ")[-1]
        self.item_table.setHorizontalHeaderLabels(base_headers + extra_headers + ["✓"])
        self.item_table.horizontalHeaderItem(self.check_column).setToolTip(self.tr("Evaluate checked items"))
        header = self.item_table.horizontalHeader()
        # Keep model values adjacent, with run checkboxes separate from navigation.
        for position, column in enumerate([self.check_column, 0, 2, 5, *range(7, self.check_column), 6]):
            header.moveSection(header.visualIndex(column), position)
        for column in (1, 3, 4):
            self.item_table.setColumnHidden(column, True)
        self.item_table.setColumnHidden(self.check_column, False)
        self.item_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.item_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        items = self.visible_items()
        if self._focused_reference and not any(i["id"] == self._focused_reference for i in items):
            self._focused_reference = None
            self.curves_button.setChecked(False)
        if not self.curves_button.isChecked():
            self.curves_hint.setText(self.tr("Click a cell to see its details"))
            self.overview_label.setText(self.tr("Item details"))
        self.item_table.setRowCount(len(items))
        roles = {
            "training": self.tr("Training regression"),
            "development": self.tr("Development validation"),
            "validation": self.tr("Validation (declared)"),
            "unknown": self.tr("Not declared"),
        }

        self.sidebar_item_list.blockSignals(True)
        self.sidebar_item_list.setRowCount(len(items))

        for row, item in enumerate(items):
            short_item_name = item["name"].rsplit(" / ", 1)[-1]
            name = QTableWidgetItem(short_item_name)
            name.setData(Qt.ItemDataRole.UserRole, item["id"])
            name.setToolTip(item["name"])
            check = QTableWidgetItem()
            check.setData(Qt.ItemDataRole.UserRole, item["id"])
            check.setToolTip(self.tr("Evaluate checked items"))
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if item["id"] in self.checked_ids else Qt.CheckState.Unchecked)
            self.item_table.setItem(row, self.check_column, check)
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
            display, cur_cell = self._headline_display_for_model(current, item)
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
                if c == 5 and cur_cell and cur_cell.get("verdict") in ("better", "worse"):
                    entry.setForeground(QColor(palette.PLOT_SHOW if cur_cell["verdict"] == "better" else palette.PLOT_REJECT))
                self.item_table.setItem(row, c, entry)

            for extra_idx, mid in enumerate(extra_models):
                extra_display, extra_cell = self._headline_display_for_model(mid, item)
                extra_item = QTableWidgetItem(str(extra_display))
                extra_item.setData(Qt.ItemDataRole.UserRole, item["id"])
                extra_item.setToolTip(str(extra_display))
                if extra_cell and extra_cell.get("verdict") in ("better", "worse"):
                    extra_item.setForeground(QColor(palette.PLOT_SHOW if extra_cell["verdict"] == "better" else palette.PLOT_REJECT))
                self.item_table.setItem(row, 7 + extra_idx, extra_item)

            if item["id"] in focused:
                from PySide6.QtCore import QItemSelectionModel

                self.item_table.selectionModel().select(
                    self.item_table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )

            nav_item = QTableWidgetItem(short_item_name if self.topic_id != "all" else item["name"])
            nav_item.setData(Qt.ItemDataRole.UserRole, item["id"])
            nav_item.setToolTip(item["name"])
            self.sidebar_item_list.setItem(row, 0, nav_item)
            self.sidebar_item_list.setItem(row, 1, check.clone())
            if item["id"] == self._focused_reference:
                self.sidebar_item_list.setCurrentCell(row, 0)

        self.sidebar_item_list.blockSignals(False)
        self.item_table.blockSignals(False)
        self._rendering = False
        self.matrix_title.setText(self.tr("Model comparison") + f"  ·  {len(items)}")
        self.context_hint.setText("")
        self._management_state()
        self._present_workspace()
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
        if self._rendering or item.column() != self.check_column:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.checked_ids.add(key)
        else:
            self.checked_ids.discard(key)
        nav = self.sidebar_item_list.item(item.row(), 1)
        if nav is not None:
            self.sidebar_item_list.blockSignals(True)
            nav.setCheckState(item.checkState())
            self.sidebar_item_list.blockSignals(False)
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
        topic = self.overview.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.topic_picker.setCurrentIndex(self.topic_picker.findData(topic))
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
        self.pages.setCurrentWidget(self.page_widgets[key])
        self.view_picker.setCurrentItem(key)
        if key == "overview":
            self._render_workspace()
        self._present_workspace()
        if key == "history":
            self._render_history()
        if key == "result":
            self._focus_first_visible()

    def _compare_cell(self, row, column):
        if column != self.check_column:
            item = self.item_table.item(row, 0)
            if item is not None:
                self.open_item(item.data(Qt.ItemDataRole.UserRole))

    def open_item(self, set_id):
        items = self.visible_items()
        index = next((i for i, item in enumerate(items) if item["id"] == set_id), None)
        if index is None:
            return
        self.editor_scroll.hide()
        self.diagnostic_editor.hide()
        self._focused_reference = set_id
        self.property_combo.blockSignals(True)
        self.property_combo.setCurrentIndex(self.property_combo.findData(items[index]["protocol"]))
        self.property_combo.blockSignals(False)
        self._load_groups(update_detail=False)
        self.sidebar_item_list.blockSignals(True)
        self.sidebar_item_list.setCurrentCell(index, 0)
        self.sidebar_item_list.blockSignals(False)
        if not self.item_table.selectionModel().isRowSelected(index):
            self.item_table.selectRow(index)
        self.item_table.scrollToItem(self.item_table.item(index, 0))
        self._expand_curves()
        self.show_page("result")
        self._update_detail()
        self._present_workspace()

    def _update_detail(self):
        if self._workspace_ready and self.curves_button.isChecked():
            row = next((i for i, item in enumerate(self.metrics_model.benchmarks)
                        if item["id"] == self._focused_reference), None)
            if row is None:
                self.detail.show_runs(None, [])
                return
            selection = self.table.selectionModel()
            selection.blockSignals(True)
            self.table.selectRow(row)
            selection.blockSignals(False)
        super()._update_detail()
        if self._workspace_ready:
            self.all_groups_button.hide()
            if self._focused_reference and self.detail._args[0]:
                self.detail.title_label.setText(self.detail._args[0]["name"])


    def open_scorecard(self, suite_id, model_id=None, *, draft=None):
        super().open_scorecard(suite_id, model_id, draft=draft, show_curves=False)
        self.show_page("result")

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
            self._present_workspace()

    def _update_run_scope(self, *args):
        super()._update_run_scope(*args)
        if not self._workspace_ready:
            return
        self.capsule_panel.hide()
        self.tolerance_label.hide()
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
            if self.item_table.item(i.row(), 0) is not None
        ]

    def _management_state(self):
        if not self._workspace_ready:
            return
        count = len(self._management_ids())
        saved = (self.current_suite_id() or 0) > 0
        idle = self._run_task is None
        topic = self.topic_id
        for action in self.topic_actions:
            action.setEnabled(idle and saved and isinstance(topic, int))
        self.manage_button.setEnabled(bool(count) and idle)
        for name, action in self.manage_actions.items():
            action.setEnabled(
                idle
                and count > 0
                and (name == "Open details" or saved)
                and (
                    count == 1
                    if name in ("Open details", "Rename in this suite…", "Edit analysis version…")
                    else True
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
            self.show_page("result")

    def _prepare_diagnostic_plan(self, name, experiments, temporary=False):
        super()._prepare_diagnostic_plan(name, experiments, temporary=temporary)
        if self._workspace_ready and self.editor_scroll.isHidden() and self._items:
            self._focused_reference = self._items[0]["id"]
            self.open_item(self._focused_reference)

    def create_suite(self):
        name, ok = QInputDialog.getText(self, self.tr("New suite"), self.tr("Suite name"))
        if ok and name.strip():
            try:
                record = self.service().create_suite(name)
            except Exception as error:
                MessageManager.send_error_message(str(error))
                return
            self.refresh(record["id"])
            self.show_page("items")

    def create_topic(self):
        name, ok = QInputDialog.getText(self, self.tr("New topic"), self.tr("Topic name"))
        if ok and name.strip():
            self._change(lambda: self.service().create_group(self.current_suite_id(), name))

    def rename_topic(self):
        key = self.topic_id
        if not isinstance(key, int):
            return
        topic = next(t for t in self._topics if t["id"] == key)
        name, ok = QInputDialog.getText(
            self, self.tr("Rename topic"), self.tr("Topic name"), text=topic["name"]
        )
        if ok:
            self._change(lambda: self.service().rename_group(self.current_suite_id(), key, name))

    def remove_topic(self):
        key = self.topic_id
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
        name, ok = QInputDialog.getText(
            self, self.tr("Rename in this suite"), self.tr("Item name"), text=item["name"]
        )
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
                source = item.get("provenance", {}).get("original_source_path") or (
                    item.get("source_ref") or {}
                ).get("path", "")
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
            item = QListWidgetItem(
                record["name"] + (" · " + self.tr("Archived") if record["archived"] else "")
            )
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
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog
        )
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
                        self.current_suite_id(),
                        ids,
                        group_id=self.topic_id if isinstance(self.topic_id, int) else None,
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
        dialog = BenchmarkImportDialog(
            self.window(), self.service().suites(), self.suite_combo.currentText()
        )
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
            dialog.group_field_combo.setCurrentIndex(
                dialog.group_field_combo.findData(group["field"] if group else "")
            )
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
                self.service().remove_from_suite(
                    self.current_suite_id(), [r["id"] for r in replacements]
                )
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