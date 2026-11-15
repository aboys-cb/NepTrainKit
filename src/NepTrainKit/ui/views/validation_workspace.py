"""Suite navigation and management around the existing validation workbench."""

from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QCoreApplication, QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsOpacityEffect,
    QScrollArea,
    QSizePolicy,
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
    AdaptiveFlowLayout,
    ListWidget,
    MessageBox,
    PushButton,
    SearchLineEdit,
    TransparentPushButton,
    TableWidget,
    TransparentToolButton,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.core.validation.protocols import protocol_for_set
from NepTrainKit.core.validation.service import ModelDraft, parse_model_header
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views.model_library import VERDICT_COLORS, ModelLibraryWidget, format_metric
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from NepTrainKit.ui.widgets.validation_model_picker import ModelColorButton, ValidationModelPicker, model_role


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


class _ModelCard(QFrame):
    def __init__(self, workspace, key):
        super().__init__(workspace.card_scroll.widget())
        self.setObjectName("comparisonModelCard")
        self.setStyleSheet(f"#comparisonModelCard {{ background: white; border: 1px solid {palette.BORDER_STRONG}; border-radius: 8px; }}")
        body = QVBoxLayout(self)
        body.setContentsMargins(6, 2, 6, 3)
        body.setSpacing(0)
        self.heading = QHBoxLayout()
        self.heading.setSpacing(3)
        body.addLayout(self.heading)
        self.color_button = ModelColorButton(self)
        self.color_button.colorChanged.connect(lambda color: workspace._set_model_color(key, color))
        self.heading.addWidget(self.color_button)
        self.name = BodyLabel(self)
        self.short_title = ""
        self.name.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {palette.TEXT_PRIMARY};")
        self.name.setTextFormat(Qt.TextFormat.PlainText)
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.heading.addWidget(self.name, 1)
        self.badge = CaptionLabel(self)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roles = (workspace.tr("Current model"), workspace.tr("Baseline model"), workspace.tr("Comparison model"))
        self.badge.setFixedWidth(max(self.badge.fontMetrics().horizontalAdvance(role) for role in roles) + 8)
        self.heading.addWidget(self.badge)
        self.metrics = CaptionLabel(self)
        self.metrics.setStyleSheet(f"color: {palette.TEXT_SECONDARY}; font-size: 11px;")
        self.metrics.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        body.addWidget(self.metrics)
        self.pin = self.action(FluentIcon.PIN, workspace.tr("Set baseline model"))
        self.normal_pin_style = self.pin.styleSheet()
        self.pin.clicked.connect(lambda: workspace._choose_model_role(key, "baseline", reveal=False))
        self.eye = self.action(FluentIcon.VIEW, workspace.tr("Show or hide model curves"))
        self.eye.setCheckable(True)
        self.eye.toggled.connect(lambda shown: workspace.detail.set_series_visible(key, shown))
        self.remove = self.action(FluentIcon.CLOSE, workspace.tr("Remove comparison model"))
        self.remove.clicked.connect(lambda: workspace.comparison_picker.remove_id(key))
        policy = self.remove.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.remove.setSizePolicy(policy)

    def set_contents(self, title, short_title, role, colors, metrics, color):
        self.short_title = short_title
        self.name.setToolTip(title)
        self.badge.setText(role)
        foreground, background = colors
        self.badge.setStyleSheet(f"background: {background}; color: {foreground}; padding: 1px 4px; border-radius: 3px;")
        self.metrics.setToolTip(metrics)
        self._fit_labels()
        self.color_button.set_color(color)
        self.color_button.setAccessibleName(f"{title} · {self.color_button.toolTip()}")
        self.setAccessibleName(f"{title} · {role}")

    def _fit_labels(self):
        font = self.name.fontMetrics()
        width = self.name.contentsRect().width()
        title = self.name.toolTip()
        if font.horizontalAdvance(title) > width:
            title = self.short_title
        self.name.setText(font.elidedText(title, Qt.TextElideMode.ElideMiddle, width))
        self.metrics.setText(self.metrics.fontMetrics().elidedText(
            self.metrics.toolTip(), Qt.TextElideMode.ElideRight, self.metrics.contentsRect().width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_labels()

    def action(self, icon, label):
        button = TransparentToolButton(icon, self)
        button.setFixedSize(22, 22)
        button.setIconSize(QSize(14, 14))
        button.setToolTip(label)
        button.setAccessibleName(label)
        self.heading.addWidget(button)
        return button

    def show_visibility(self, visible):
        self.eye.blockSignals(True)
        self.eye.setChecked(visible)
        self.eye.setIcon(FluentIcon.VIEW if visible else FluentIcon.HIDE)
        self.eye.blockSignals(False)
        if visible:
            self.setGraphicsEffect(None)
        elif self.graphicsEffect() is None:
            fade = QGraphicsOpacityEffect(self)
            fade.setOpacity(0.5)
            self.setGraphicsEffect(fade)


class _ModelStrip(QScrollArea):
    """Wrap cards in reading order, showing up to two rows at a time."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        content = QWidget()
        self.flow = AdaptiveFlowLayout(content)
        self.flow.setWidgetMaximumWidth(360)
        self.flow.setContentsMargins(0, 0, 0, 0)
        self.flow.setHorizontalSpacing(8)
        self.flow.setVerticalSpacing(4)
        self.setWidget(content)
        content.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)

    def fit_cards(self):
        if not self.flow.count():
            return
        cards = [self.flow.itemAt(i).widget() for i in range(self.flow.count())]
        # Keep the role, all actions and a recognizable part of the model name visible.
        readable = max(card.minimumSizeHint().width() + 10 * card.name.fontMetrics().averageCharWidth()
                       for card in cards)
        available = self.width()
        gap = self.flow.horizontalSpacing()
        slots = min(4, len(cards))
        self.flow.setWidgetMinimumWidth(max(readable, (available - (slots - 1) * gap) // slots))
        two_rows = 2 * max(card.sizeHint().height() for card in cards) + self.flow.verticalSpacing()
        if self.flow.heightForWidth(max(available, readable)) > two_rows:
            available -= self.verticalScrollBar().sizeHint().width()
            self.flow.setWidgetMinimumWidth(max(readable, (available - (slots - 1) * gap) // slots))
        width = max(available, readable)
        height = self.flow.heightForWidth(width)
        self.widget().resize(width, height)
        self.flow.setGeometry(self.widget().rect())
        scrollbar = self.horizontalScrollBar().sizeHint().height() if width > available else 0
        self.setFixedHeight(min(height, two_rows) + scrollbar)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.fit_cards()


class ValidationWorkspace(ModelLibraryWidget):
    """Navigation never mutates the independent set of checked run targets."""

    def __init__(self, parent=None, service=None):
        self._workspace_ready = False
        self._color_rows = None
        super().__init__(parent, service)
        self.checked_ids = set()
        self.topic_id = "all"
        self._workspace_suite = None
        self._items = []
        self._topics = []
        self._focused_reference = None
        self._rendering = False
        self._search_filter = ""

        # Enable adaptive plot sizing and the right-hand multi-model quantitative inspector in the workspace
        self.detail.set_adaptive_layout(True)
        self.detail.header_layout.insertWidget(1, self.detail.legend_panel)

        # Reuse the existing plot and inspector below the comparison matrix.
        root = self.layout()
        root.itemAt(0).layout().setContentsMargins(0, 0, 0, 0)
        self.controls_panel.setStyleSheet(
            f"#validationControls {{ background: {palette.SURFACE}; border: 0; border-bottom: 1px solid {palette.BORDER}; }}"
        )
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
        self.sidebar_search.setPlaceholderText(self.tr("Search topics or items…"))
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
        self.manage_menu = QMenu(self)
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
        # These actions also live in the global Fluent menu, which owns their item bindings.
        import_menu = QMenu(self)
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
        self.baseline_label.hide()
        # The shared workbench owns the selectors and their signals.
        selectors = QWidget(self.controls_panel)
        selector_layout = QHBoxLayout(selectors)
        while fields.count():
            selector_layout.addLayout(fields.takeAt(0).layout())
        selectors.hide()
        self.role_dialog = None
        self._picker_rows = None
        self.scope_hint.hide()
        self.cards = {}
        self._card_context = None
        self._card_rows = None
        self._card_state = None
        self.card_scroll = _ModelStrip(self.controls_panel)
        self.detail.seriesVisibilityChanged.connect(self._sync_card_visibility)
        self.edit_models = PushButton("+  " + self.tr("Add models"), self.controls_panel)
        self.edit_models.setAccessibleName(self.tr("Manage comparison models"))
        self.edit_models.setToolTip(self.tr("Manage comparison models"))
        self.edit_models.setMinimumHeight(30)
        self.edit_models.setStyleSheet(f"PushButton {{ color: {palette.ACCENT_PRESSED}; border: 1px dashed {palette.ACCENT}; border-radius: 6px; padding: 3px 9px; background: {palette.ACCENT_SOFT}; }}")
        self.edit_models.clicked.connect(self._open_role_dialog)
        fields.setSpacing(8)
        fields.addWidget(self.card_scroll, 1)
        fields.addWidget(self.edit_models)
        fields.addWidget(self.run_selected_button)
        self.controls_panel.layout().setContentsMargins(12, 5, 12, 5)
        self.controls_panel.layout().setSpacing(0)
        self.sidebar.setMinimumWidth(164)
        self.sidebar.setMaximumWidth(420)
        self.setMinimumWidth(500)
        self._workspace_ready = True
        self._present_workspace()
        self.run_summary.hide()
        self._update_header(None)
        self.show_page("result")

    def _update_header(self, baseline):
        suite_name = self.suite_combo.currentText() or self.tr("Model validation")
        current = self.current_model_row()
        self._update_cards(baseline)

        done, compared, regressed, failed = set(), set(), set(), set()
        for column, cell in zip(self.model._columns, current["cells"] if current else ()):
            key = column["set_id"]
            if cell.get("status") == "succeeded":
                done.add(key)
                if cell.get("verdict") in ("better", "same", "worse"):
                    compared.add(key)
                if cell.get("verdict") == "worse":
                    regressed.add(key)
            elif cell.get("status") == "failed":
                failed.add(key)
        parts = [self.tr("Current model evaluated on {done} of {total} items").format(
            done=len(done), total=len(self._items)) if current else self.tr("No current model selected")]
        if compared and baseline != self.model_combo.currentData():
            parts.append(self.tr("{regressed} of {compared} comparable items regressed").format(
                regressed=len(regressed), compared=len(compared)))
        if failed:
            parts.append(self.tr("{count} failed").format(count=len(failed)))
        summary = "   ·   ".join(parts) if self._items else self.tr("No validation items")
        self.suite_combo.setToolTip(f"{suite_name}\n{summary}")

    def comparison_curve_ids(self):
        current = self.model_combo.currentData()
        base = next((row["model_id"] for row in self.model._rows if row["is_baseline"]), None)
        active = list(dict.fromkeys(key for key in (current, base, *self.selected_model_ids()) if key is not None))
        if (self._library_root, self.current_suite_id()) != self._card_context:
            return active
        retained = [key for key in self.cards if key in active]
        return list(dict.fromkeys([*retained, *active]))

    def _update_cards(self, baseline):
        keys = self.comparison_curve_ids()
        primary = self.model_combo.currentData()
        state = (tuple(keys), primary, baseline, self.current_suite_id(), self._run_task is None)
        context = (self._library_root, self.current_suite_id())
        if self._card_context == context and self._card_rows is self.model._rows and self._card_state == state:
            return
        new_comparisons = [key for key in keys if key not in self.cards and key not in (primary, baseline)]
        for key in list(self.cards):
            if context == self._card_context and key in keys:
                continue
            card = self.cards.pop(key)
            self.card_scroll.flow.removeWidget(card)
            card.hide()
            card.deleteLater()
        self._card_context = context
        for key in keys:
            row = self.model.row(self.model.row_of_model(key))
            if row is None:
                continue
            is_current, is_base = key == primary, key == baseline
            if is_current:
                role = self.tr("Current model")
                colors = (palette.INFO, palette.INFO_SOFT)
            elif is_base:
                role = self.tr("Baseline model")
                colors = (palette.WARNING_STRONG, palette.WARNING_SOFT)
            else:
                role = self.tr("Comparison model")
                colors = (palette.TEXT_SECONDARY, palette.SURFACE_SUBTLE)
            title = " / ".join(filter(None, (row["project_name"], row["name"])))
            if key not in self.cards:
                card = _ModelCard(self, key)
                self.cards[key] = card
                self.card_scroll.flow.addWidget(card)
            card = self.cards[key]
            short_title = row["name"] if row["project_name"] == self.suite_combo.currentText() else title
            card.set_contents(title, short_title, role, colors, self._card_metrics(row), self.detail._series_pen(key).color())
            pin_color = QColor(Qt.GlobalColor.white) if is_base else palette.TEXT_SECONDARY
            card.pin.setIcon(FluentIcon.PIN.colored(pin_color, pin_color))
            card.pin.setStyleSheet(f"QToolButton {{ background: {palette.WARNING_STRONG}; border: 0; border-radius: 5px; }}" if is_base else card.normal_pin_style)
            label = self.tr("Baseline model") if is_base else self.tr("Set baseline model")
            card.pin.setAccessibleName(label)
            card.pin.setToolTip(self.tr("Save to a project to use as baseline") if row["temporary"] else label)
            card.pin.setEnabled(is_base or (not row["temporary"] and self.current_suite_id() is not None and self._run_task is None))
            card.show_visibility(self.detail.series_visible(key))
            card.remove.setVisible(not is_current and not is_base)
            card.show()
            card.layout().activate()
        self.card_scroll.setVisible(bool(self.cards))
        self.card_scroll.fit_cards()
        if new_comparisons:
            self.card_scroll.ensureWidgetVisible(self.cards[new_comparisons[-1]].remove, 5, 0)
        self._card_rows, self._card_state = self.model._rows, state

    def _open_role_dialog(self):
        if self.role_dialog is None:
            self.role_dialog = ValidationModelPicker(self.window())
            self.role_dialog.roleRequested.connect(self._choose_model_role)
            self.role_dialog.colorRequested.connect(self._set_model_color)
            self.role_dialog.filesDropped.connect(self._add_temporary_models)
        self._picker_rows = None
        self._sync_role_dialog()
        self.role_dialog.show_picker()

    def hideEvent(self, event):
        super().hideEvent(event)
        if getattr(self, "role_dialog", None) is not None:
            self.role_dialog.close()

    def _sync_role_dialog(self):
        if self.role_dialog is None:
            return
        rows = self.model._rows
        if self._picker_rows is not rows:
            versions = []
            for row in rows:
                key = row["model_id"]
                name = " / ".join(filter(None, (row["project_name"], row["name"])))
                kind = row.get("model_kind") or ""
                if key == 0 and self._draft:
                    kind = parse_model_header(self._draft.data)[0]
                details = [kind.upper()] if kind else []
                if row["temporary"]:
                    details.append(self.tr("Save to a project to use as baseline"))
                if row.get("data_size") is not None:
                    details.append(self.tr("{count} frames").format(count=row["data_size"]))
                details.append(self._card_metrics(row))
                versions.append((key, name, " · ".join(details), self.detail._series_pen(key).color().name()))
            order = (list(self.role_dialog.buttons) if self.role_dialog.isVisible()
                     else self.comparison_curve_ids())
            position = {key: index for index, key in enumerate(order)}
            versions.sort(key=lambda version: position.get(version[0], len(order)))
            self.role_dialog.set_models(versions)
            self._picker_rows = rows
        current = self.model_combo.currentData()
        baseline = next((r["model_id"] for r in rows if r["is_baseline"]), None)
        eligible = {r["model_id"] for r in rows if not r["temporary"]} if self.current_suite_id() is not None else set()
        self.role_dialog.set_roles(current, baseline, self.comparison_picker.selected_ids(), eligible,
                                   running=self._run_task is not None)
        self.role_dialog.setAcceptDrops(self._run_task is None)

    def _set_model_color(self, key, color):
        row = self.model.row(self.model.row_of_model(key))
        if row is None:
            return
        Config.set(f"validation_colors:{self._library_root}", f"{key}:{row['content_hash']}", color.name())
        self.detail.set_series_color(key, color)
        if key in self.cards:
            self.cards[key].color_button.set_color(color)
        if self.role_dialog is not None and key in self.role_dialog.color_buttons:
            self.role_dialog.color_buttons[key].set_color(color)

    def _choose_model_role(self, key, role, *, reveal=True):
        row = self.model.row(self.model.row_of_model(key))
        if row is None:
            return
        current = self.model_combo.currentData()
        base = next((r["model_id"] for r in self.model._rows if r["is_baseline"]), None)
        selected = self.selected_model_ids()
        if role == model_role(key, current, base, selected):
            return
        next_base = base
        if key == base and self._run_task is not None:
            self._sync_role_dialog()
            return
        if key == current and role != "current":
            current = None
        if role == "baseline":
            if self._run_task or self.current_suite_id() is None or row["temporary"]:
                self._sync_role_dialog()
                return
            next_base = key
            if base is not None and base not in selected:
                selected.append(base)
        elif role == "current":
            current = key
            if key == base:
                next_base = None
        elif key == base:
            next_base = None
        if role == "hidden":
            selected = [model for model in selected if model != key]
        elif key not in selected:
            selected.append(key)
        if next_base != base:
            try:
                self.service().set_baseline(self.current_suite_id(), next_base)
            except (OSError, ValueError) as error:
                MessageManager.send_error_message(str(error))
                self._sync_role_dialog()
                return
        if reveal and role != "hidden":
            self.detail.set_series_visible(key, True)
        if next_base != base:
            same_curves = role == "baseline" and key in self.cards
            if same_curves:
                self.detail.set_current_series(current)
            self._load_scorecard(current, comparison_ids=selected, update_detail=not same_curves)
        else:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(self.model.row_of_model(current))
            self.model_combo.blockSignals(False)
            self._update_comparison_options(selected)
            self._load_groups(update_detail=False)
            self._on_selection_changed()

    def _add_temporary_models(self, paths):
        if self._run_task is not None:
            return
        selected = self.selected_model_ids()
        current = self.model_combo.currentData()
        added = []
        for path in paths:
            try:
                draft = ModelDraft.from_file(path)
                elements = set(parse_model_header(draft.data)[1])
                if self._items and not any(set(item["elements"]) <= elements for item in self._items):
                    raise ValueError(self.tr("No compatible validation items for {name}.").format(name=path.name))
                key = self.service().capture_model(draft, temporary=True)["id"]
                added.append(key)
                self.detail.set_series_visible(key, True)
            except (OSError, ValueError) as error:
                MessageManager.send_error_message(str(error))
        if added:
            self._load_scorecard(current, comparison_ids=selected + added)

    def _card_metrics(self, row):
        values = []
        for field, symbol, unit in (("force", "F", "meV/Å"), ("energy", "E", "meV/atom")):
            value = row.get(field)
            if value is not None:
                values.append(f"{symbol} {value:.3g} {unit}")
        if not values:
            return self.tr("No recorded RMSE")
        source = self.service().imported_accuracy(row["model_id"])
        group = source.get("group")
        reported = source.get("metrics", {})
        matches = all(row.get(k) == reported.get(k + "_rmse") for k in ("force", "energy"))
        if matches and group in ("train", "test"):
            label = self.tr("Training") if group == "train" else self.tr("Test")
            if source.get("status") != "verified":
                label += " · " + self.tr("Unverified")
        else:
            label = self.tr("Historical RMSE")
        return label + " " + " · ".join(values)

    def _sync_card_visibility(self, key, shown):
        if key in self.cards:
            self.cards[key].show_visibility(shown)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "edit_models"):
            self._present_workspace()

    def _present_workspace(self):
        if not hasattr(self, "edit_models"):
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
        self.edit_models.setText("+" if self.width() < 700 else "+  " + self.tr("Add models"))
        narrow = self.main_panel.width() < 970
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

    def _scorecard_current_id(self, requested, comparison_ids):
        row = self.model.row(self.model.row_of_model(requested))
        if row is not None:
            return None if row["is_baseline"] else requested
        same_suite = self._displayed_suite_id == self.current_suite_id()
        was_empty = self.model_combo.count() > 0 and self.model_combo.currentIndex() < 0
        if requested is None and (comparison_ids is not None or (same_suite and was_empty)):
            return None
        return next((r["model_id"] for r in self.model._rows if not r["is_baseline"]), None)

    def _load_scorecard(self, model_id=None, *, comparison_ids=None, update_detail=True):
        super()._load_scorecard(model_id, comparison_ids=comparison_ids, update_detail=update_detail)
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
        if self.pages.currentWidget() is self.detail_scroll and self.editor_scroll.isHidden():
            if self._focused_reference and not self.curves_button.isChecked():
                self.open_item(self._focused_reference)
            elif not self._focused_reference and self.visible_items():
                self.open_item(self.visible_items()[0]["id"])

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
        models = self.comparison_curve_ids()
        suite_id = self.current_suite_id()
        baseline = self.service().get_suite(suite_id).get("baseline_model_id") if suite_id is not None else None
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
        model_columns = {key: column for column, key in enumerate(extra_models, 7)}
        model_columns[current] = 5
        ordered_columns = [model_columns[key] for key in models] or [5]
        for position, column in enumerate([self.check_column, 0, 2, *ordered_columns, 6]):
            header.moveSection(header.visualIndex(column), position)
        for column in (1, 3, 4):
            self.item_table.setColumnHidden(column, True)
        for column in (5, 6):
            self.item_table.setColumnHidden(column, current is None and bool(extra_models))
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
            self.sidebar_item_list.setItem(row, 1, QTableWidgetItem(check))
            if item["id"] == self._focused_reference:
                self.sidebar_item_list.setCurrentCell(row, 0)

        self.sidebar_item_list.blockSignals(False)
        self.item_table.blockSignals(False)
        self._rendering = False
        self.matrix_title.setText(self.tr("Model comparison") + f"  ·  {len(items)}")
        self._update_header(baseline)
        if self.role_dialog is not None and self.role_dialog.isVisible():
            self._sync_role_dialog()
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
        self.detail._current_key = self.model_combo.currentData()
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
        if primary is not None and self._card_state is not None:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(self.model.row_of_model(self._card_state[1]))
            self.model_combo.blockSignals(False)
            self._choose_model_role(primary, "current")

    def _on_selection_changed(self, *, update_detail=True):
        if self._color_rows is not self.model._rows:
            saved = Config.get_section(f"validation_colors:{self._library_root}")
            colors = {}
            for row in self.model._rows:
                key, digest = row["model_id"], row["content_hash"]
                color = saved.get(f"{key}:{digest}") or saved.get(f"0:{digest}")
                if color and QColor(color).isValid():
                    colors[key] = color
            self.detail.model_colors = colors
            self._color_rows = self.model._rows
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
        self.run_selected_button.setToolTip(self.scope_hint.text() + "\n" + self.scope_hint.toolTip())
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
