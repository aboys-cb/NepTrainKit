"""Suite navigation and management around the existing validation workbench."""

from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QCoreApplication, QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsOpacityEffect,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QSizePolicy,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QListWidgetItem,
    QStackedWidget,
    QSplitter,
    QTableWidgetItem,
    QToolButton,
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
    TableWidget,
    TransparentToolButton,
    TogglePushButton,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.core.validation.protocols import protocol_for_set
from NepTrainKit.core.validation.service import ModelDraft, parse_model_header
from NepTrainKit.ui.views.model_library import ModelLibraryWidget, RunDetailPanel, format_metric
from NepTrainKit.ui.widgets.layout import FlowLayout
from NepTrainKit.ui.widgets.validation_gallery import ValidationGallery
from NepTrainKit.ui.widgets.validation_presentation import presentation_for, uses_accuracy_cards, uses_curve_chart
from NepTrainKit.ui.widgets.reference_response_plot import is_magnetic_eos, response_branches
from NepTrainKit.ui.widgets.validation_group_comparison import ValidationGroupComparison
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from NepTrainKit.ui.widgets.validation_model_picker import ModelColorButton, ValidationModelPicker, group_marker_icon, model_role


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


class _SuiteItemDelegate(QStyledItemDelegate):
    """Paint a card across both columns while preserving native table hit targets."""

    def paint(self, painter, option, index):
        table = self.parent()
        rect = option.rect
        card = QRect(1, rect.y() + 3, table.viewport().width() - 3, rect.height() - 6)
        selected = table.currentRow() == index.row()
        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(palette.ACCENT if selected else palette.BORDER))
        painter.setBrush(QColor(palette.ACCENT_SOFT if selected else "white"))
        painter.drawRoundedRect(card, 8, 8)
        if index.column() == 1:
            check = QStyleOptionButton()
            check.initFrom(table)
            checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
            check.state |= QStyle.StateFlag.State_On if checked else QStyle.StateFlag.State_Off
            check.rect = QRect(0, 0, 16, 16)
            check.rect.moveCenter(rect.center())
            table.style().drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorCheckBox, check, painter, table)
        else:
            data = index.data(Qt.ItemDataRole.UserRole + 1)
            if data:
                subtitle, protocol, status, ink, fill = data
                left, right = rect.left() + 1, card.right() - 11
                font = option.font
                font.setPixelSize(11)
                painter.setFont(font)
                small = painter.fontMetrics()
                alignment = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft

                def badge(text, top, foreground, background):
                    width = small.horizontalAdvance(text) + 10
                    box = QRect(right - width, top, width, 16)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(background))
                    painter.drawRoundedRect(box, 4, 4)
                    painter.setPen(QColor(foreground))
                    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
                    return box.left() - 5

                end = badge(protocol, card.top() + 7, palette.TEXT_SECONDARY, palette.SURFACE_SUBTLE)
                meta_end = badge(status, card.bottom() - 20, ink, fill) if status else right
                painter.setPen(QColor(palette.TEXT_SECONDARY))
                painter.drawText(QRect(left, card.bottom() - 20, meta_end - left, 16), alignment,
                                 small.elidedText(subtitle, Qt.TextElideMode.ElideRight, meta_end - left))
                font.setPixelSize(13)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QColor(palette.TEXT_PRIMARY))
                marker = index.data(Qt.ItemDataRole.DecorationRole)
                if marker:
                    marker.paint(painter, QRect(left, card.top() + 8, 13, 13))
                    left += 17
                title = painter.fontMetrics().elidedText(index.data() or "", Qt.TextElideMode.ElideMiddle, end - left)
                painter.drawText(QRect(left, card.top() + 5, end - left, 20), alignment, title)
        painter.restore()


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
        self._detail_dirty = True
        self._curves_suspended = False
        self._curve_timer = QTimer(self)
        self._curve_timer.setSingleShot(True)
        self._curve_timer.timeout.connect(self._show_current_curve)
        self._rendering = False
        self._search_filter = ""

        # Enable adaptive plot sizing and the right-hand multi-model quantitative inspector in the workspace
        self.metrics_footer.hide()
        self.detail.set_adaptive_layout(True)
        self.detail.response_view_changed = self._update_detail
        self.detail.set_inspector_expanded(False)
        self.groups = ValidationGroupComparison(self)
        self.groups.currentRequested.connect(self._set_current_group)
        self.groups.changed.connect(self._update_detail)
        self.groups.markerChanged.connect(self._group_marker_changed)
        self.detail.markerRequested.connect(self.groups.set_marker)
        self.detail.layout().setContentsMargins(0, 0, 0, 0)
        self.chart_tools = QWidget(self.detail)
        tools = QHBoxLayout(self.chart_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(8)
        self.plot_switches = []
        for label, icon, check in (
            (self.detail.tr("Reference"), FluentIcon.VIEW, None),
            (self.detail.relative_checkbox.text(), None, self.detail.relative_checkbox),
            (self.detail.offset_checkbox.text(), FluentIcon.ALIGNMENT, self.detail.offset_checkbox),
            (self.detail.wheel_zoom_checkbox.text(), FluentIcon.ZOOM_IN, self.detail.wheel_zoom_checkbox),
        ):
            action = QAction(label, self.chart_tools)
            action.setCheckable(True)
            action.setToolTip(label if check is None else label + "\n" + check.toolTip())
            action.setChecked(self.detail.series_visible(None) if check is None else check.isChecked())
            if icon:
                action.setIcon(icon.icon())
            else:
                action.setIconText("E₀")
            if check is None:
                action.triggered.connect(lambda on: self.detail.set_series_visible(None, on))
                self.detail.seriesVisibilityChanged.connect(lambda key, on, ref=action: ref.setChecked(on) if key is None else None)
            else:
                action.toggled.connect(check.setChecked)
                check.toggled.connect(action.setChecked)
            button = QToolButton(self.chart_tools)
            button.setDefaultAction(action)
            button.setAccessibleName(label)
            button.setFixedSize(30, 30)
            button.setIconSize(QSize(16, 16))
            button.setStyleSheet(f"QToolButton {{ border: 1px solid transparent; border-radius: 5px; font-weight: 600; }}"
                                 f"QToolButton:checked {{ background: {palette.ACCENT_SOFT}; border-color: {palette.ACCENT}; }}"
                                 f"QToolButton:hover {{ background: {palette.SURFACE_HOVER}; border-color: {palette.BORDER_STRONG}; }}")
            tools.addWidget(button)
            self.plot_switches.append(button)
        self.eos_grid_mode = TogglePushButton(self.tr("Residual"), self.chart_tools)
        self.eos_grid_mode.setToolTip(self.tr("Show residuals instead of energy"))
        self.eos_grid_mode.toggled.connect(self.detail._change_eos_view)
        self.eos_grid_mode.toggled.connect(lambda _: self._present_workspace())
        tools.addWidget(self.eos_grid_mode)
        self.eos_layout = TogglePushButton(self.tr("Overlay groups"), self.chart_tools)
        self.eos_layout.setToolTip(self.tr("Overlay groups on shared axes"))
        self.eos_layout.toggled.connect(self.detail.set_eos_overlay)
        tools.addWidget(self.eos_layout)
        self.add_eos_groups = PushButton(FluentIcon.ADD, self.tr("Comparison groups"), self.chart_tools)
        self.add_eos_groups.clicked.connect(self.groups.open_picker)
        tools.addWidget(self.add_eos_groups)
        self.chart_options = TransparentToolButton(FluentIcon.MORE, self.chart_tools)
        self.chart_options.setToolTip(self.tr("Chart options"))
        self.chart_options.setAccessibleName(self.chart_options.toolTip())
        menu = QMenu(self.chart_options)
        self.chart_options.setMenu(menu)
        self.chart_options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_eos_groups.setToolTip(self.tr("Comparison groups"))
        self.add_eos_groups.setAccessibleName(self.add_eos_groups.toolTip())
        self.chart_options.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        menu.aboutToShow.connect(self._chart_options_menu)
        tools.addWidget(self.chart_options)
        self.chart_tools.hide()
        self.detail.groupRemoved.connect(lambda key: self.groups.choose_role(key, "hidden"))
        self.standard_options = QWidget(self.detail)
        options = QHBoxLayout(self.standard_options)
        options.setContentsMargins(0, 0, 0, 0)
        for widget in (self.detail.legend_panel, self.detail.wheel_zoom_checkbox,
                       self.detail.relative_checkbox, self.detail.offset_checkbox):
            options.addWidget(widget)
        self.detail.header_layout.addWidget(self.standard_options)

        # Keep the shared detail and editor controls together above the comparison matrix.
        root = self.layout()
        root.itemAt(0).layout().setContentsMargins(0, 0, 0, 0)
        self.controls_panel.setStyleSheet(
            f"#validationControls {{ background: {palette.SURFACE}; border: 0; border-bottom: 1px solid {palette.BORDER}; }}"
        )
        detail_page = QFrame(self)
        detail_page.setObjectName("comparisonDetails")
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)

        while root.count() > 2:
            item = root.takeAt(2)
            if item.widget():
                detail_layout.addWidget(item.widget(), item.stretch() if hasattr(item, "stretch") else 0)
            else:
                detail_layout.addLayout(item.layout())
        detail_layout.setStretch(detail_layout.indexOf(self.splitter), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.editor_scroll), 1)
        detail_layout.setStretch(detail_layout.indexOf(self.curves_placeholder), 1)

        self.sidebar = QFrame(self)
        self.sidebar.setObjectName("validationSidebar")
        self.sidebar.setStyleSheet(
            f"#validationSidebar {{ background: {palette.SURFACE}; border-right: 1px solid {palette.BORDER}; }}"
        )
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(10, 7, 10, 6)
        side.setSpacing(5)

        side_header = QHBoxLayout()
        side_header.setSpacing(2)
        side_title = CaptionLabel(self.tr("Validation suites"), self.sidebar)
        side_title.setStyleSheet(f"color: {palette.TEXT_SECONDARY}; font-weight: bold;")
        side_header.addWidget(side_title, 1)
        self.new_suite_button = QPushButton(FluentIcon.ADD.icon(), self.tr("New suite"), self.sidebar)
        self.new_suite_button.clicked.connect(self.create_suite)
        side_header.addWidget(self.new_suite_button)
        side.addLayout(side_header)

        fields = self.controls_panel.layout().itemAt(0).layout()
        suite_field = fields.takeAt(0).layout()
        caption = suite_field.takeAt(0).widget()
        caption.hide()
        caption.deleteLater()
        side.addWidget(self.suite_combo)
        suite_field.deleteLater()

        self.topic_picker = ComboBox(self.sidebar)
        self.topic_picker.setAccessibleName(self.tr("Topic"))
        self.topic_picker.currentIndexChanged.connect(self._browse_topic)
        side.addWidget(self.topic_picker)
        self.topic_chips = QWidget(self.sidebar)
        self.topic_chips.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.topic_flow = FlowLayout(self.topic_chips, spacing=4)
        self.topic_chips.setStyleSheet(
            f"QPushButton {{ background: {palette.SURFACE_SUBTLE}; color: {palette.TEXT_SECONDARY}; "
            f"border: 1px solid {palette.BORDER}; border-radius: 5px; padding: 3px 6px; font-size: 12px; }}"
            f"QPushButton:hover, QPushButton:focus {{ border-color: {palette.ACCENT}; }}"
            f"QPushButton:checked {{ background: {palette.ACCENT_PRESSED}; color: white; }}"
        )
        side.addWidget(self.topic_chips)
        self.sidebar_search = SearchLineEdit(self.sidebar)
        self.sidebar_search.setFixedHeight(29)
        self.sidebar_search.setPlaceholderText(self.tr("Search items, topics or protocols…"))
        self.sidebar_search.textChanged.connect(self._on_sidebar_search)
        side.addWidget(self.sidebar_search)

        self.sidebar_item_list = ComparisonTable(self.sidebar)
        self.sidebar_item_list.setColumnCount(2)
        self.sidebar_item_list.setHorizontalHeaderLabels([self.tr("Validation item"), "✓"])
        self.sidebar_item_list.horizontalHeader().moveSection(1, 0)
        self.sidebar_item_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sidebar_item_list.horizontalHeader().hide()
        self.sidebar_item_list.setAlternatingRowColors(False)
        self.sidebar_item_list.horizontalHeader().setMinimumSectionSize(26)
        self.sidebar_item_list.setColumnWidth(1, 33)
        self.sidebar_item_list.verticalHeader().setDefaultSectionSize(59)
        self.sidebar_item_list.setStyleSheet(f"QTableView {{ background: {palette.SURFACE}; border: none; }}")
        delegate = _SuiteItemDelegate(self.sidebar_item_list)
        for column in (0, 1):
            self.sidebar_item_list.setItemDelegateForColumn(column, delegate)
        self.sidebar_item_list.verticalHeader().hide()
        self.sidebar_item_list.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.sidebar_item_list.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.sidebar_item_list.setSelectionMode(TableWidget.SelectionMode.SingleSelection)
        self.sidebar_item_list.currentCellChanged.connect(self._navigate_item)
        self.sidebar_item_list.cellClicked.connect(
            lambda row, col: self._navigate_item(row, col) if self.pages.currentWidget() is not self.result_page else None)
        self.sidebar_item_list.cellDoubleClicked.connect(self._run_item_missing)
        self.sidebar_item_list.itemChanged.connect(self._navigator_checked)
        side.addWidget(self.sidebar_item_list, 1)

        self.empty_items = CaptionLabel(self.sidebar)
        self.empty_items.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.empty_items.setStyleSheet(f"color: {palette.TEXT_SECONDARY}; padding-top: 16px;")
        side.addWidget(self.empty_items, 1)

        self.new_topic_button = PushButton(self.tr("New topic"), self.sidebar)
        self.new_topic_button.clicked.connect(self.create_topic)
        self.existing_button = PushButton(self.tr("Add existing items…"), self.sidebar)
        self.existing_button.setToolTip(self.tr("Add existing items…"))
        self.existing_button.clicked.connect(self.add_existing_items)
        self.organize_button = PushButton(self.tr("Group by source folders"), self.sidebar)
        self.organize_button.setToolTip(self.tr("Group by source folders"))
        self.organize_button.clicked.connect(self.organize_sources)
        self.suite_tools = QMenu(self.sidebar)
        for button in (self.new_topic_button, self.existing_button, self.organize_button, self.refresh_button):
            action = self.suite_tools.addAction(button.text() or button.toolTip(), button.click)
            button.hide()
            self.suite_tools.aboutToShow.connect(lambda a=action, b=button: a.setEnabled(b.isEnabled()))
        self.topic_actions = []
        for label, callback in (("Rename topic…", self.rename_topic), ("Remove topic", self.remove_topic)):
            self.topic_actions.append(self.suite_tools.addAction(self.tr(label), callback))
        suite_tools_button = QPushButton(FluentIcon.FOLDER.icon(), self.tr("Organize"), self.sidebar)
        for button in (self.new_suite_button, suite_tools_button):
            button.setFixedHeight(24)
            button.setIconSize(QSize(12, 12))
            button.setStyleSheet(
                f"QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; "
                f"color: {palette.TEXT_SECONDARY}; padding: 1px 4px; font-size: 12px; }}"
                f"QPushButton:hover, QPushButton:focus {{ background: {palette.SURFACE_HOVER}; border-color: {palette.BORDER}; }}"
            )
        suite_tools_button.clicked.connect(lambda: self.suite_tools.exec(
            suite_tools_button.mapToGlobal(suite_tools_button.rect().bottomLeft())))
        side_header.addWidget(suite_tools_button)

        main = QWidget(self)
        self.main_panel = main
        body = QVBoxLayout(main)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.setContentsMargins(4, 2, 4, 2)
        self.view_picker = SegmentedWidget(main)
        self.compact_view_picker = ComboBox(main)
        self.compact_view_picker.setAccessibleName(self.tr("Results"))
        for label, key in (("Validation curve", "result"), ("Overview", "items"), ("Suite plots", "gallery")):
            self.view_picker.addItem(key, self.tr(label), lambda checked=False, page=key: self.show_page(page))
            self.compact_view_picker.addItem(self.tr(label), userData=key)
        self.compact_view_picker.activated.connect(lambda i: self.show_page(self.compact_view_picker.itemData(i)))
        top_bar.addWidget(self.view_picker)
        top_bar.addWidget(self.compact_view_picker)
        top_bar.addStretch()
        top_bar.addWidget(self.chart_tools)
        body.addLayout(top_bar)
        for button in self.view_picker.items.values():
            button.setFixedWidth(button.fontMetrics().horizontalAdvance(button.text()) + 26)
        self.view_picker.setFixedWidth(sum(b.width() for b in self.view_picker.items.values()) + 2)

        self.pages = QStackedWidget(main)
        body.addWidget(self.pages, 1)
        self.page_widgets = {}
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
        self.item_table.cellActivated.connect(self._compare_cell)
        self.item_table.itemSelectionChanged.connect(self._management_state)
        matrix_card = QFrame(self)
        matrix_card.setObjectName("validationMatrix")
        matrix_card.setStyleSheet(f"#validationMatrix {{background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px;}}")
        matrix_layout = QVBoxLayout(matrix_card)
        matrix_layout.setContentsMargins(14, 9, 14, 6)
        self.matrix_title = BodyLabel(matrix_card)
        font = self.matrix_title.font()
        font.setPixelSize(13)
        font.setBold(True)
        self.matrix_title.setFont(font)
        matrix_header = QHBoxLayout()
        matrix_header.addWidget(self.matrix_title, 1)
        self.table_metric = ComboBox(matrix_card)
        self.table_metric.setAccessibleName(self.tr("Comparison metric"))
        self.table_metric.currentIndexChanged.connect(self._render_workspace)
        matrix_header.addWidget(self.table_metric)
        matrix_layout.addLayout(matrix_header)
        matrix_layout.addWidget(self.item_table)
        self.matrix_card = matrix_card
        self.detail_scroll = QScrollArea(self)
        self.detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setWidget(detail_page)
        self.result_page = self.detail_scroll
        self.gallery = ValidationGallery(self._plot_thumbnail, main)
        self.gallery.itemOpened.connect(self.open_item)
        self._thumbnail_renderer = None
        for key, widget in (("result", self.result_page), ("items", matrix_card), ("gallery", self.gallery)):
            self.page_widgets[key] = widget
            self.pages.addWidget(widget)
        self.item_table.setAlternatingRowColors(False)
        self.item_table.verticalHeader().hide()
        self.item_table.verticalHeader().setDefaultSectionSize(36)

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
            self.manage_actions[label] = action
        for table in (self.sidebar_item_list, self.item_table):
            table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            table.customContextMenuRequested.connect(lambda pos, view=table: self._item_menu(view, pos))

        self.workspace_split = QSplitter(Qt.Orientation.Horizontal, self)
        self.workspace_split.setChildrenCollapsible(False)
        self.workspace_split.setHandleWidth(5)
        self.workspace_split.addWidget(self.sidebar)
        self.workspace_split.addWidget(main)
        self.workspace_split.setStretchFactor(1, 1)
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
        self.sidebar.setMinimumWidth(212)
        self.sidebar.setMaximumWidth(420)
        self.workspace_split.setSizes([304, 960])
        self.setMinimumWidth(500)
        self._workspace_ready = True
        self.main_panel.installEventFilter(self)
        self._present_workspace()
        self.run_summary.hide()
        self._update_header(None)
        self.show_page("items")

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

    def showEvent(self, event):
        self._curves_suspended = False
        super().showEvent(event)
        if self._workspace_ready and self.pages.currentWidget() is self.result_page and self._detail_dirty:
            self._curve_timer.start(16)

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._workspace_ready:
            self._curves_suspended = True
            self._curve_timer.stop()
        if hasattr(self, "groups"):
            self.groups.close_picker()
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
        self._reload_gallery()

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
        self._reload_gallery()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "edit_models"):
            self._present_workspace()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize and obj is getattr(self, "main_panel", None):
            self._present_workspace()
        return super().eventFilter(obj, event)

    def _present_workspace(self):
        if not hasattr(self, "edit_models"):
            return
        width = self.sidebar.contentsRect().width() - 20
        height = self.topic_flow.heightForWidth(width)
        chips_fit = height <= 72 and self.topic_flow.minimumSize().width() <= width
        self.topic_chips.setFixedHeight(height if chips_fit else 0)
        self.topic_chips.setVisible(chips_fit)
        self.topic_picker.setVisible(not chips_fit)
        has_items = self.sidebar_item_list.rowCount() > 0
        self.sidebar_item_list.setVisible(has_items)
        self.empty_items.setVisible(not has_items)
        self.edit_models.setText("+" if self.width() < 700 else "+  " + self.tr("Add models"))
        width = self.main_panel.width()
        eos = (self.pages.currentWidget() is self.result_page and self.detail._groups
               and uses_curve_chart(self.detail._groups[0][1]) and self.editor_scroll.isHidden())
        self.chart_tools.setVisible(bool(eos))
        self.add_eos_groups.setText(self.tr("Comparison groups") if width >= 530 else "")
        candidates = [b for i, b in enumerate(self.plot_switches) if self._plot_switch_available(i)] if eos else []
        gap = self.chart_tools.layout().spacing()
        basic = sum(w.sizeHint().width() + gap for w in (self.eos_grid_mode, self.eos_layout, self.add_eos_groups)
                    if not w.isHidden()) if eos else 0
        nav_width = self.view_picker.width()
        compact = width < nav_width + basic + self.chart_options.sizeHint().width() + 32
        self.view_picker.setVisible(not compact)
        self.compact_view_picker.setVisible(compact)
        if eos:
            nav_width = self.compact_view_picker.sizeHint().width() if compact else nav_width
            budget = width - nav_width - basic - 32
            step = self.plot_switches[0].width() + gap
            overflow = budget < len(candidates) * step
            slots = max(0, (budget - self.chart_options.sizeHint().width() - gap) // step) if overflow else len(candidates)
            for button in self.plot_switches:
                button.setVisible(button in candidates[:slots])
            self.chart_options.setVisible(overflow)

    def _reload_gallery(self):
        self.gallery.invalidate()
        self._show_gallery()

    def _show_gallery(self):
        if self.pages.currentWidget() is self.gallery:
            self.gallery.set_records(self.visible_items())

    def _plot_thumbnail(self, item, size):
        if self._thumbnail_renderer is None:
            self._thumbnail_renderer = RunDetailPanel(self.gallery)
            self._thumbnail_renderer.hide()
        renderer = self._thumbnail_renderer
        rows = [self.model.row(self.model.row_of_model(key)) for key in self.comparison_curve_ids()]
        entries = self._curve_entries(item, [row for row in rows if row])
        renderer.group_markers = self.groups.markers
        renderer.model_colors = self.detail.model_colors.copy()
        renderer._series_styles = self.detail._series_styles.copy()
        renderer._hidden_series = self.detail._hidden_series.copy()
        renderer.show_runs(item, entries)
        chart = renderer.eos_plots[0] if renderer._groups and renderer.eos_plots else renderer.plot
        plots = [] if renderer._groups and not renderer.eos_plots else chart.listDataItems()
        if not any(line.isVisible() and line.xData is not None and len(line.xData) for line in plots):
            empty = CaptionLabel(self.tr("No plotted data"))
            empty.resize(size)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return empty.grab(), "", ""
        if hasattr(chart, "x_label"):
            x_label, y_label = chart.x_label.text(), chart.y_label.text()
            chart.x_label.hide()
            chart.y_label.hide()
        else:
            axes = [chart.getAxis(side) for side in ("bottom", "left")]
            x_label, y_label = [" ".join(filter(None, (axis.labelText, axis.labelUnits))) for axis in axes]
            for axis in axes:
                axis.showLabel(False)
            chart.getPlotItem().layout.setContentsMargins(2, 7, 4, 2)
        chart.setFixedSize(size)
        if getattr(chart, "category_labels", None):
            # Thumbnails use state indices; full labels remain on the interactive curve.
            chart.set_category_ticks([(x, str(x)) for x in chart.category_labels])
            x_label = renderer.tr("Combination index")
        if hasattr(chart, "fit"):
            chart.fit()
        else:
            chart.getPlotItem().autoRange()
        return chart.grab(), x_label, y_label

    def _navigate_item(self, row, column, *_):
        if self._rendering or row < 0 or column != 0:
            return
        item = self.sidebar_item_list.item(row, 0)
        if item is not None:
            self.open_item(item.data(Qt.ItemDataRole.UserRole))

    def _run_item_missing(self, row, column):
        if column != 0 or self._run_task is not None:
            return
        item = self.sidebar_item_list.item(row, 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item else None
        benchmark = next((i for i in self._items if i["id"] == key), None)
        if benchmark:
            jobs = [job for job in self.run_jobs(model_ids=self.comparison_curve_ids(), benchmarks=[benchmark])
                    if self._cell_status(*job)["status"] in ("missing", "failed")]
            if jobs:
                self._run_validation(jobs=jobs)

    def _plot_switch_available(self, index):
        if self.detail._groups and index in (1, 2):
            item = self.detail._groups[0][1]
            if item["protocol"] == "response":
                spec = item["summary"]["analysis"]
                if spec["mode"] != "scan" or spec["quantity"] != "energy":
                    return False
        return not (self.detail._eos_residual and index < 2)

    def _chart_options_menu(self):
        menu = self.chart_options.menu()
        menu.clear()
        for index, button in enumerate(self.plot_switches):
            if button.isHidden() and self._plot_switch_available(index):
                menu.addAction(button.defaultAction())

    def apply_canvas_mode(self, _mode):
        if self._workspace_ready:
            self._update_detail()
            self._reload_gallery()

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
        self._refresh_filtered_items()

    def _refresh_filtered_items(self):
        self._render_workspace()
        if self._focused_reference and self.metrics_model.kind == "eos" and self.curves_button.isChecked():
            self._update_detail()
        else:
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
            self.groups.set_candidates(None, [])
            self.topic_id = "all"
            self._focused_reference = None
            self._workspace_suite = context
            self._detail_dirty = True
        self._items = self.service().suite_items(suite_id) if suite_id is not None else []
        families = {}
        for item in self._items:
            view = presentation_for(item["protocol"])
            if item["protocol"] == "response" and uses_curve_chart(item):
                families[item["id"]] = [(n, branch[0]) for n, branch in enumerate(response_branches(item))]
            elif view.source_group:
                group = view.group(item)
                family = view.family(item) or item["id"]
                families.setdefault(family, []).append((min(group.get("indices") or [0]), item["id"]))
        self.groups.load_markers(self.service().library.root, ([key for _, key in sorted(rows)] for rows in families.values()))
        self.detail.group_markers = self.groups.markers
        self.checked_ids.intersection_update(item["id"] for item in self._items)
        if self._focused_reference not in {item["id"] for item in self._items}:
            self._focused_reference = None
        self._topics = self.service().suite_groups(suite_id) if suite_id is not None else []
        if self.topic_id not in ("all", None) and not any(t["id"] == self.topic_id for t in self._topics):
            self.topic_id = "all"
        self.topic_picker.blockSignals(True)
        self.topic_picker.clear()
        self.topic_picker.addItem(self.tr("All ({count})").format(count=len(self._items)), userData="all")
        for group in self._topics:
            self.topic_picker.addItem(group["name"], userData=group["id"])
        if any(item["group_id"] is None for item in self._items):
            self.topic_picker.addItem(self.tr("Ungrouped"), userData=None)
        self.topic_picker.setCurrentIndex(self.topic_picker.findData(self.topic_id))
        self.topic_picker.blockSignals(False)
        while self.topic_flow.count():
            chip = self.topic_flow.takeAt(0).widget()
            chip.hide()
            chip.deleteLater()
        for index in range(self.topic_picker.count()):
            chip = QPushButton(self.topic_picker.itemText(index), self.topic_chips)
            chip.setCheckable(True)
            chip.setChecked(index == self.topic_picker.currentIndex())
            chip.clicked.connect(lambda _, i=index: self.topic_picker.setCurrentIndex(i))
            chip.setAutoExclusive(True)
            self.topic_flow.addWidget(chip)
            chip.show()
        self._render_workspace()
        self._update_run_scope()
        self._focus_first_visible()

    def _focus_first_visible(self):
        if self.pages.currentWidget() is self.result_page and self.editor_scroll.isHidden():
            if self._focused_reference and not self.curves_button.isChecked():
                self.open_item(self._focused_reference)
            elif not self._focused_reference and self.visible_items():
                self.open_item(self.visible_items()[0]["id"])

    def _browse_topic(self, index):
        if index < 0:
            return
        self.topic_id = self.topic_picker.currentData()
        self.topic_flow.itemAt(index).widget().setChecked(True)
        self._refresh_filtered_items()

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

    def _headline_display_for_model(self, model_id, item, metric):
        """Return (formatted_text, cell_dict) for a model on a validation item."""
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
        return ("—" if status in ("missing", "succeeded") else self._status_text(status)), cell

    def _render_workspace(self):
        if not self._workspace_ready or self._rendering:
            return
        self._rendering = True
        models = self.comparison_curve_ids()
        suite_id = self.current_suite_id()
        baseline = self.service().get_suite(suite_id).get("baseline_model_id") if suite_id is not None else None
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
        kinds = {item["protocol"] for item in items}
        choices = ()
        if len(kinds) == 1 and presentation_for(items[0]["protocol"]).choose_table_metric:
            choices = protocol_for_set(items[0]).headline_metrics()
        if [self.table_metric.itemData(i) for i in range(self.table_metric.count())] != [m.key for m in choices]:
            self.table_metric.blockSignals(True)
            self.table_metric.clear()
            for metric in choices:
                label = QCoreApplication.translate("ScorecardTableModel", metric.label)
                self.table_metric.addItem(f"{label} ({metric.unit})", userData=metric.key)
            self.table_metric.blockSignals(False)
        self.table_metric.setVisible(len(choices) > 1)
        if self._focused_reference and not any(i["id"] == self._focused_reference for i in items):
            self._focused_reference = None
            self.curves_button.setChecked(False)
            self._sync_comparison_groups(None)
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
            view = presentation_for(item["protocol"])
            name = QTableWidgetItem(view.item_title(item))
            marker = self.groups.markers.get(item["id"])
            if marker:
                name.setIcon(group_marker_icon(marker))
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
            metric = protocol.metric(self.table_metric.currentData()) if choices else protocol.headline_metrics()[0]
            display, cur_cell = self._headline_display_for_model(current, item, metric)
            state = self._cell_status(current, item["id"])
            status_label = self.tr("Update needed") if state.get("previous_version") else self._status_text(state["status"])
            values = [
                item.get("group_name") or self.tr("Ungrouped"),
                QCoreApplication.translate("ScorecardTableModel", metric.label) if view.choose_table_metric else mode_labels.get(analysis, protocol.title),
                roles.get(role, role),
                str(summary.get("frames", len(summary.get("volumes", [])))),
                display,
                status_label,
            ]
            for c, value in enumerate(values, 1):
                entry = QTableWidgetItem(str(value))
                entry.setData(Qt.ItemDataRole.UserRole, item["id"])
                entry.setToolTip(str(value))
                if c == 5 and value == "—" and cur_cell and cur_cell["status"] == "succeeded":
                    entry.setToolTip("\n".join(translate_runtime_message(w) for w in summary.get("warnings", [])) or str(value))
                if c == 5 and cur_cell and cur_cell.get("verdict") in ("better", "worse"):
                    entry.setForeground(QColor(palette.PLOT_SHOW if cur_cell["verdict"] == "better" else palette.PLOT_REJECT))
                self.item_table.setItem(row, c, entry)

            for extra_idx, mid in enumerate(extra_models):
                extra_display, extra_cell = self._headline_display_for_model(mid, item, metric)
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

            nav_item = QTableWidgetItem(name.text() if view.source_group else item["name"].removeprefix(values[0] + " / "))
            if marker:
                nav_item.setIcon(name.icon())
            status = status_label if current is not None else ""
            ink, fill = palette.TEXT_SECONDARY, palette.SURFACE_SUBTLE
            if state["status"] == "succeeded":
                ink, fill = palette.SUCCESS, palette.SUCCESS_SOFT
            elif state["status"] == "failed":
                ink, fill = palette.DANGER, palette.DANGER_SOFT
            elif state["status"] == "unsupported":
                ink, fill = palette.WARNING_STRONG, palette.WARNING_SOFT
            count_label = self.tr("{count} frames").format(count=values[3])
            if protocol.kind == "diagnostic":
                count_label = self.tr("{count} scan points").format(count=len(summary.get("coordinates", [])))
            subtitle = f"{values[0]} · {count_label}"
            badge = view.item_badge(item)
            nav_item.setData(Qt.ItemDataRole.UserRole + 1, (subtitle, badge, status, ink, fill))
            nav_item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, f"{subtitle} · {badge} · {status}")
            nav_item.setData(Qt.ItemDataRole.UserRole, item["id"])
            nav_item.setToolTip(f"{item['name']}\n{subtitle} · {self.tr('Current model')}: {status}" if status else item["name"])
            nav_item.setToolTip(nav_item.toolTip() + "\n" + self.tr("Double-click to evaluate missing or failed models for this item."))
            self.sidebar_item_list.setItem(row, 0, nav_item)
            # A native clone leaves a live PySide wrapper after Qt replaces the item.
            self.sidebar_item_list.setItem(row, 1, QTableWidgetItem(check))
            if item["id"] == self._focused_reference:
                self.sidebar_item_list.setCurrentCell(row, 0)

        self.empty_items.setText(self.tr("No matching items") if self._items else self.tr("No validation items"))
        self.sidebar_item_list.blockSignals(False)
        self.item_table.blockSignals(False)
        self._rendering = False
        self.matrix_title.setText(self.tr("Overview") + f"  ·  {len(items)}")
        self._update_header(baseline)
        if self.role_dialog is not None and self.role_dialog.isVisible():
            self._sync_role_dialog()
        self._management_state()
        self._present_workspace()
        self._show_gallery()

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

    def check_visible(self):
        self.checked_ids.update(i["id"] for i in self.visible_items())
        self._render_workspace()
        self._update_run_scope()

    def clear_checks(self):
        self.checked_ids.clear()
        self._render_workspace()
        self._update_run_scope()

    def show_page(self, key, *, defer_curves=True):
        if not self._workspace_ready:
            return
        self._curve_timer.stop()
        if key != "result":
            self.groups.close_picker()
        self.pages.setCurrentWidget(self.page_widgets[key])
        self.view_picker.setCurrentItem(key)
        self.compact_view_picker.setCurrentIndex(self.compact_view_picker.findData(key))
        self._present_workspace()
        if key == "gallery":
            self._show_gallery()
        if key == "result":
            if defer_curves:
                self._curve_timer.start(16)
            else:
                self._show_current_curve()

    def _show_current_curve(self):
        if self._curves_suspended or self.pages.currentWidget() is not self.result_page or not self.editor_scroll.isHidden():
            return
        if self._focused_reference and self.curves_button.isChecked():
            if self._detail_dirty:
                self._update_detail()
        else:
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
        self._detail_dirty = True
        if self.pages.currentWidget() is self.result_page:
            self._update_detail()
        else:
            self.show_page("result", defer_curves=False)
        self._present_workspace()

    def _group_marker_changed(self, key, symbol):
        self.detail.set_group_marker(key, symbol)
        for table in (self.sidebar_item_list, self.item_table):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == key:
                    item.setIcon(group_marker_icon(symbol))
                    break
        self._reload_gallery()

    def _set_current_group(self, key):
        if self.detail.magnetic_selection is not None:
            self.groups.browse(key)
            self._update_detail()
        else:
            self.open_item(key)

    def _focus_plot_group(self, tag):
        if self.detail.magnetic_selection is not None:
            self.groups.focus(tag)
        elif self.groups.comparisons:
            self.groups.focus(self.metrics_model.benchmarks[int(tag[1:]) - 1]["id"])
        else:
            super()._focus_plot_group(tag)

    def _select_all_groups(self):
        if hasattr(self, "groups") and self.groups.focused:
            self.groups.expand()
        else:
            super()._select_all_groups()

    def _sync_comparison_groups(self, benchmark):
        candidates = []
        view = presentation_for(benchmark["protocol"]) if benchmark else None
        family = view.family(benchmark) if view else None
        magnetic = benchmark and benchmark["protocol"] == "response" and uses_curve_chart(benchmark)
        current = self._focused_reference
        if magnetic:
            self.service().response_plot_series(benchmark)
            from NepTrainKit.core.validation.response_plot import exchange_plot_view

            display_summary = exchange_plot_view(benchmark["summary"], self.detail.response_kind)
            branches = response_branches(dict(benchmark, summary=display_summary))
            candidates = [(key, label, self.tr("{count} frames").format(count=len(indices)))
                          for key, label, indices in branches]
            keys = [key for key, _, _ in candidates]
            if self.groups.current in keys:
                current = self.groups.current
            else:
                current = keys[0]
                series = display_summary.get("plot_series") or []
                if presentation_for("response").item_badge(benchmark) == "MAE" and series:
                    self.groups.comparisons = {key for key, branch in zip(keys[1:], series[1:])
                                              if branch.get("plane") == series[0].get("plane")}
                elif series and benchmark["summary"]["analysis"]["mode"] == "four_state":
                    self.groups.comparisons = {key for key, branch in zip(keys[1:], series[1:])
                                              if branch.get("condition") == series[0].get("condition")}
                else:
                    self.groups.comparisons = set(keys[1:])
                self.groups.focused = False
        elif family:
            for item in self.visible_items():
                if item["protocol"] == benchmark["protocol"] and view.family(item) == family:
                    group = view.group(item)
                    caption = self.tr("{count} frames").format(count=len(group["indices"]))
                    candidates.append((item["id"], str(group["value"]), caption))
        self.groups.set_candidates(current, candidates)
        self.detail.magnetic_selection = None
        if magnetic:
            self.detail.magnetic_selection = {current} if self.groups.focused else self.groups.comparisons | {current}
        self.detail.magnetic_can_focus = bool(magnetic and self.groups.comparisons)
        self.detail.eos_current_id = current if len(candidates) > 1 else None
        self.add_eos_groups.setVisible(len(candidates) > 1)
        eos = bool(benchmark and uses_curve_chart(benchmark))
        accuracy = bool(benchmark and uses_accuracy_cards(benchmark))
        self.chart_tools.setVisible(eos)
        self.standard_options.setVisible(not eos and not accuracy)
        self.detail.title_label.setVisible(not eos and not accuracy)
        self.detail.info_label.setVisible(not eos and not accuracy)
        comparing = bool(self.groups.comparisons) and not self.groups.focused
        self.eos_layout.setVisible(comparing)

    def _update_detail(self):
        self.detail._current_key = self.model_combo.currentData()
        if self._workspace_ready:
            self._curve_timer.stop()
            if self._curves_suspended or self.pages.currentWidget() is not self.result_page:
                self._detail_dirty = True
                return
            self._detail_dirty = False
        group_rows = None
        group_total = None
        if self._workspace_ready and self.curves_button.isChecked():
            benchmarks = self.metrics_model.benchmarks
            row = next((i for i, item in enumerate(benchmarks) if item["id"] == self._focused_reference), None)
            if row is None:
                self.detail.show_runs(None, [])
                self._sync_comparison_groups(None)
                return
            selection = self.table.selectionModel()
            selection.blockSignals(True)
            self.table.selectRow(row)
            selection.blockSignals(False)
            self._sync_comparison_groups(benchmarks[row])
            if presentation_for(benchmarks[row]["protocol"]).source_group:
                keys = self.groups.comparisons | {self._focused_reference}
                group_rows = [row] if self.groups.focused else [i for i, item in enumerate(benchmarks) if item["id"] in keys]
                group_total = len(keys)
        super()._update_detail(group_rows=group_rows, group_total=group_total)
        if self._workspace_ready:
            self.all_groups_button.hide()
            if self._focused_reference and self.detail._args[0] and group_rows is None:
                self.detail.title_label.setText(self.detail._args[0]["name"])

    def open_scorecard(self, suite_id, model_id=None, *, draft=None):
        super().open_scorecard(suite_id, model_id, draft=draft, show_curves=False)
        self.show_page("result", defer_curves=False)

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
            self.gallery.invalidate()
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
        self._management_state()

    def _item_menu(self, table, pos):
        menu = QMenu(table)
        row = table.rowAt(pos.y())
        if row >= 0:
            if table is self.sidebar_item_list:
                table.setCurrentCell(row, 0)
                self.item_table.selectRow(row)
            elif not table.selectionModel().isRowSelected(row):
                table.selectRow(row)
            for name, action in self.manage_actions.items():
                if table is self.item_table or name != "Open details":
                    menu.addAction(action)
            menu.addSeparator()
        menu.addAction(self.tr("Check visible items"), self.check_visible).setEnabled(bool(table.rowCount()))
        menu.addAction(self.tr("Clear checks"), self.clear_checks).setEnabled(bool(self.checked_ids))
        menu.exec(table.viewport().mapToGlobal(pos))
        menu.deleteLater()

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
        kind = "response" if record["protocol"] == "efv" else record["protocol"]
        dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData(kind))
        dialog.load_file(self.service().library.open_object(record["object_hash"]))
        dialog.suite_combo.setEnabled(False)
        dialog.name_edit.setText(record["name"])
        dialog.notes_edit.setText(record.get("provenance", {}).get("notes", ""))
        if record["protocol"] == "response":
            dialog.analysis_editor.set_settings(record["source_ref"]["analysis"])
        elif record["protocol"] == "efv":
            dialog.analysis_editor.set_settings({"mode": "slices"})
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
