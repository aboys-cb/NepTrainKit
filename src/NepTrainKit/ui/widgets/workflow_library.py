"""Fluent workflow library sidebar for the Make Dataset workbench."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QDateTime, QLocale, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListWidgetItem,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
)
from qfluentwidgets import (
    Action,
    CaptionLabel,
    FluentIcon,
    ListItemDelegate,
    ListWidget,
    Pivot,
    PrimaryPushButton,
    RoundMenu,
    SearchLineEdit,
    StrongBodyLabel,
    TransparentToolButton,
    setFont,
)

from NepTrainKit.core.workflow_library import WorkflowEntry

_ENTRY_ROLE = int(Qt.ItemDataRole.UserRole)
_NAME_ROLE = _ENTRY_ROLE + 1
_META_ROLE = _ENTRY_ROLE + 2
_CURRENT_ROLE = _ENTRY_ROLE + 3


def localized_workflow_entry_name(entry: WorkflowEntry) -> str:
    """Return a translated display name for a stable built-in template ID."""
    if entry.origin != "builtin":
        return entry.name
    names = {
        "builtin-crystal-strain": QCoreApplication.translate(
            "BuiltinWorkflowTemplates", "Crystal strain"
        ),
        "builtin-supercell-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates", "Atom perturbation"
        ),
        "builtin-alloy-occupancy": QCoreApplication.translate(
            "BuiltinWorkflowTemplates", "Alloy occupancy"
        ),
        "builtin-vacancy-candidates": QCoreApplication.translate(
            "BuiltinWorkflowTemplates", "Vacancy sampling"
        ),
        "builtin-spin-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates", "Spin perturbation"
        ),
    }
    return names.get(entry.workflow_id, entry.name)


def localized_workflow_entry_description(entry: WorkflowEntry) -> str:
    """Return translated built-in guidance while preserving user text verbatim."""
    if entry.origin != "builtin":
        return entry.description
    descriptions = {
        "builtin-crystal-strain": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Build an elemental crystal prototype and sample independent uniaxial lattice strains.",
        ),
        "builtin-supercell-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Expand each input structure and generate randomly displaced atomic configurations.",
        ),
        "builtin-alloy-occupancy": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Expand a parent cell, plan target alloy compositions, and realize each target by random site occupancy.",
        ),
        "builtin-vacancy-candidates": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Expand each input structure and generate one single-vacancy candidate from every expanded cell.",
        ),
        "builtin-spin-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Normalize existing scalar or vector initial moments, then sample nearby spin directions and magnitudes.",
        ),
    }
    return descriptions.get(entry.workflow_id, entry.description)


def localized_workflow_input_requirement(entry: WorkflowEntry) -> str:
    """Return the localized pre-run requirement for a built-in template."""
    if entry.origin != "builtin":
        return entry.input_requirement
    requirements = {
        "builtin-crystal-strain": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "No input structure is required. Set the lattice, element, lattice "
            "constant, strain range, and output limit before running.",
        ),
        "builtin-supercell-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Load one or more periodic structures. Review the replication factors, "
            "atom limit, displacement amplitude, and outputs per input before running.",
        ),
        "builtin-alloy-occupancy": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Load a periodic parent structure. Replace the example Co, Cr, and Ni "
            "element set and check that the supercell has enough sites for the "
            "requested exact ratios.",
        ),
        "builtin-vacancy-candidates": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Load one or more periodic structures. Check the supercell size and "
            "switch the vacancy count or concentration mode when one vacancy per "
            "cell is not appropriate.",
        ),
        "builtin-spin-perturb": QCoreApplication.translate(
            "BuiltinWorkflowTemplates",
            "Load structures containing spin or ASE initial magnetic moments. The "
            "template does not invent missing moment magnitudes; verify the scalar "
            "lift axis and perturbation range.",
        ),
    }
    return requirements.get(entry.workflow_id, entry.input_requirement)


class WorkflowItemDelegate(ListItemDelegate):
    """Render a dense two-line workflow item in the app's Fluent language."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 56)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(2, 2, -2, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        accent = QColor(15, 143, 145)
        if selected:
            background = QColor(accent)
            background.setAlpha(25)
            painter.setBrush(background)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)
        elif hovered:
            background = option.palette.color(QPalette.ColorRole.AlternateBase)
            background.setAlpha(170)
            painter.setBrush(background)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)

        if bool(index.data(_CURRENT_ROLE)):
            painter.setBrush(accent)
            painter.drawRoundedRect(
                QRect(rect.left(), rect.top() + 9, 3, rect.height() - 18), 2, 2
            )

        icon_rect = QRect(rect.left() + 11, rect.top() + 12, 16, 16)
        FluentIcon.DOCUMENT.icon().paint(painter, icon_rect)
        text_left = icon_rect.right() + 10
        text_right = rect.right() - 25

        title_font = QFont(option.font)
        title_font.setPixelSize(13)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            QRect(text_left, rect.top() + 6, text_right - text_left, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(index.data(_NAME_ROLE) or ""),
        )

        meta_font = QFont(option.font)
        meta_font.setPixelSize(11)
        painter.setFont(meta_font)
        painter.setPen(option.palette.color(QPalette.ColorRole.PlaceholderText))
        painter.drawText(
            QRect(text_left, rect.top() + 28, text_right - text_left, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(index.data(_META_ROLE) or ""),
        )
        painter.drawText(
            QRect(rect.right() - 24, rect.top(), 20, rect.height()),
            Qt.AlignmentFlag.AlignCenter,
            "⋯",
        )
        painter.restore()


class WorkflowLibraryPanel(QFrame):
    """Browse and manage named workflows and reusable templates."""

    newRequested = Signal()
    copyRequested = Signal()
    pasteRequested = Signal()
    saveRequested = Signal()
    saveAsRequested = Signal(str)
    openRequested = Signal(str, str)
    renameRequested = Signal(str, str)
    duplicateRequested = Signal(str, str)
    deleteRequested = Signal(str, str)
    exportRequested = Signal(str, str)
    importRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_id: str | None = None
        self._can_start_new = False
        self.setObjectName("workflowLibraryPanel")
        # Responsive splitter sizing owns the visible width. Keeping this at
        # zero lets the main window cross the breakpoint that hides the panel.
        self.setMinimumWidth(0)
        self.setMaximumWidth(310)
        self.setStyleSheet(
            "QFrame#workflowLibraryPanel {"
            "border: 1px solid rgba(100,120,128,38); border-radius: 10px;"
            "background: rgba(255,255,255,232); }"
            "QFrame#currentWorkflowCard {"
            "border: 1px solid rgba(15,143,145,42); border-radius: 8px;"
            "background: rgba(15,143,145,10); }"
        )

        self.title_label = StrongBodyLabel(self.tr("Workflow library"), self)
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.title_label, 1)

        self.new_shortcut_action = QAction(self.tr("New blank workflow"), self)
        self.new_shortcut_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_shortcut_action.setShortcutContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.new_shortcut_action.setEnabled(False)
        self.new_shortcut_action.triggered.connect(self.newRequested)
        self.addAction(self.new_shortcut_action)

        self.current_card = QFrame(self)
        self.current_card.setObjectName("currentWorkflowCard")
        self.current_caption = CaptionLabel(
            self.tr("CURRENT"), self.current_card
        )
        self.current_caption.setStyleSheet("color: #087f81; font-weight: 600;")
        self.copy_button = TransparentToolButton(FluentIcon.COPY, self.current_card)
        self.copy_button.setFixedSize(28, 28)
        self.copy_button.setToolTip(self.tr("Copy workflow JSON"))
        self.copy_button.setAccessibleName(self.tr("Copy workflow JSON"))
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copyRequested)
        self.paste_button = TransparentToolButton(FluentIcon.PASTE, self.current_card)
        self.paste_button.setFixedSize(28, 28)
        self.paste_button.setToolTip(self.tr("Add cards from clipboard"))
        self.paste_button.setAccessibleName(self.tr("Add cards from clipboard"))
        self.paste_button.clicked.connect(self.pasteRequested)
        current_header = QHBoxLayout()
        current_header.setContentsMargins(0, 0, 0, 0)
        current_header.setSpacing(2)
        current_header.addWidget(self.current_caption, 1)
        current_header.addWidget(self.copy_button)
        current_header.addWidget(self.paste_button)
        self.current_label = StrongBodyLabel(
            self.tr("Unsaved workflow"), self.current_card
        )
        setFont(self.current_label, 13, QFont.Weight.DemiBold)
        self.dirty_label = CaptionLabel("", self.current_card)
        self.dirty_label.setStyleSheet("color: #b36b00;")
        current_layout = QVBoxLayout(self.current_card)
        current_layout.setContentsMargins(10, 7, 10, 7)
        current_layout.setSpacing(1)
        current_layout.addLayout(current_header)
        current_layout.addWidget(self.current_label)
        current_layout.addWidget(self.dirty_label)

        self.search_edit = SearchLineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("Search"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_lists)

        self.pivot = Pivot(self)
        self.stack = QStackedWidget(self)
        self.workflow_list = self._create_list()
        self.builtin_template_list = self._create_list()
        self.template_list = self._create_list()
        self.stack.addWidget(self.workflow_list)
        self.stack.addWidget(self.builtin_template_list)
        self.stack.addWidget(self.template_list)
        self.pivot.addItem(
            "workflows",
            self.tr("Saved"),
            lambda: self.stack.setCurrentWidget(self.workflow_list),
        )
        self.pivot.addItem(
            "builtins",
            self.tr("Built-in"),
            lambda: self.stack.setCurrentWidget(self.builtin_template_list),
        )
        self.pivot.addItem(
            "templates",
            self.tr("User"),
            lambda: self.stack.setCurrentWidget(self.template_list),
        )
        for item in self.pivot.items.values():
            setFont(item, 14)
            item.setFixedHeight(36)
        self.pivot.setCurrentItem("workflows")

        self.save_button = PrimaryPushButton(
            FluentIcon.SAVE,
            self.tr("Save workflow"),
            self,
        )
        self.save_button.clicked.connect(self.saveRequested)
        self.more_button = TransparentToolButton(FluentIcon.MORE, self)
        self.more_button.setToolTip(self.tr("More workflow actions"))
        self.more_button.setAccessibleName(self.tr("More workflow actions"))
        self.more_button.clicked.connect(self._show_more_menu)
        action_row = QHBoxLayout()
        action_row.setSpacing(5)
        action_row.addWidget(self.save_button, 1)
        action_row.addWidget(self.more_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 14, 13, 12)
        layout.setSpacing(9)
        layout.addLayout(header)
        layout.addWidget(self.current_card)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.pivot)
        layout.addWidget(self.stack, 1)
        layout.addLayout(action_row)

    def _create_list(self) -> ListWidget:
        widget = ListWidget(self.stack)
        widget.setItemDelegate(WorkflowItemDelegate(widget))
        widget.setSpacing(1)
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda point, selected=widget: self._show_item_menu(selected, point)
        )
        widget.itemActivated.connect(self._open_item)
        widget.setToolTip(self.tr("Double-click to open; right-click to manage."))
        return widget

    @staticmethod
    def _item_data(item: QListWidgetItem | None):
        return item.data(_ENTRY_ROLE) if item is not None else None

    def _open_item(self, item: QListWidgetItem) -> None:
        data = self._item_data(item)
        if data:
            self.openRequested.emit(data["id"], data["kind"])

    def _show_item_menu(self, widget: ListWidget, point) -> None:
        item = widget.itemAt(point)
        data = self._item_data(item)
        if not data:
            return
        menu = RoundMenu(parent=self)
        actions = [(FluentIcon.FOLDER, self.tr("Open"), self.openRequested)]
        if data.get("origin") != "builtin":
            actions.append((FluentIcon.EDIT, self.tr("Rename"), self.renameRequested))
        actions.extend(
            (
                (FluentIcon.COPY, self.tr("Duplicate"), self.duplicateRequested),
                (FluentIcon.SAVE, self.tr("Export"), self.exportRequested),
            )
        )
        for icon, text, signal in actions:
            action = Action(icon, text, self)
            action.triggered.connect(
                lambda _checked=False, selected=signal: selected.emit(
                    data["id"], data["kind"]
                )
            )
            menu.addAction(action)
        if data.get("origin") != "builtin":
            menu.addSeparator()
            delete_action = Action(FluentIcon.DELETE, self.tr("Delete"), self)
            delete_action.triggered.connect(
                lambda: self.deleteRequested.emit(data["id"], data["kind"])
            )
            menu.addAction(delete_action)
        menu.exec(widget.mapToGlobal(point))

    def _show_more_menu(self) -> None:
        menu = RoundMenu(parent=self)
        new_workflow = Action(
            FluentIcon.ADD, self.tr("New blank workflow"), self
        )
        new_workflow.setEnabled(self._can_start_new)
        new_workflow.triggered.connect(self.newRequested)
        menu.addAction(new_workflow)
        menu.addSeparator()
        save_workflow = Action(FluentIcon.SAVE, self.tr("Save as workflow"), self)
        save_workflow.triggered.connect(lambda: self.saveAsRequested.emit("workflow"))
        menu.addAction(save_workflow)
        save_template = Action(FluentIcon.COPY, self.tr("Save as template"), self)
        save_template.triggered.connect(lambda: self.saveAsRequested.emit("template"))
        menu.addAction(save_template)
        menu.addSeparator()
        import_workflow = Action(
            FluentIcon.FOLDER_ADD, self.tr("Import workflow"), self
        )
        import_workflow.triggered.connect(lambda: self.importRequested.emit("workflow"))
        menu.addAction(import_workflow)
        import_template = Action(
            FluentIcon.FOLDER_ADD, self.tr("Import template"), self
        )
        import_template.triggered.connect(lambda: self.importRequested.emit("template"))
        menu.addAction(import_template)
        menu.exec(self.more_button.mapToGlobal(self.more_button.rect().topRight()))

    @staticmethod
    def _format_updated(value: str) -> str:
        date_time = QDateTime.fromString(value, Qt.DateFormat.ISODate)
        if not date_time.isValid():
            return value
        return QLocale().toString(
            date_time.toLocalTime(), QLocale.FormatType.ShortFormat
        )

    def _populate(self, widget: ListWidget, entries: list[WorkflowEntry]) -> None:
        widget.clear()
        for entry in entries:
            name = localized_workflow_entry_name(entry)
            description = localized_workflow_entry_description(entry)
            input_requirement = localized_workflow_input_requirement(entry)
            if entry.origin == "builtin":
                category = QCoreApplication.translate("CardCatalog", entry.category)
                detail = self.tr("{category} · {count} cards").format(
                    category=category,
                    count=entry.card_count,
                )
            else:
                detail = self.tr("{count} cards · {updated}").format(
                    count=entry.card_count,
                    updated=self._format_updated(entry.updated_at),
                )
            item = QListWidgetItem()
            item.setData(
                _ENTRY_ROLE,
                {
                    "id": entry.workflow_id,
                    "kind": entry.kind,
                    "name": name,
                    "origin": entry.origin,
                    "search_text": " ".join((name, description, input_requirement)),
                },
            )
            item.setData(_NAME_ROLE, name)
            item.setData(_META_ROLE, detail)
            item.setData(_CURRENT_ROLE, entry.workflow_id == self._current_id)
            if entry.origin == "builtin":
                item.setToolTip("\n".join(part for part in (description, input_requirement) if part))
            else:
                item.setToolTip(
                    self.tr("Double-click to open; right-click to manage {name}.").format(
                        name=name
                    )
                )
            widget.addItem(item)

    def set_entries(
        self,
        workflows: list[WorkflowEntry],
        templates: list[WorkflowEntry],
    ) -> None:
        self._populate(self.workflow_list, workflows)
        self._populate(
            self.builtin_template_list,
            [entry for entry in templates if entry.origin == "builtin"],
        )
        self._populate(
            self.template_list,
            [entry for entry in templates if entry.origin == "user"],
        )
        self._filter_lists(self.search_edit.text())

    def set_current(
        self,
        name: str | None,
        *,
        dirty: bool = False,
        workflow_id: str | None = None,
        has_cards: bool = False,
        template_preview: bool = False,
    ) -> None:
        self._current_id = workflow_id
        self.current_caption.setText(
            self.tr("PREVIEW") if template_preview else self.tr("CURRENT")
        )
        self.current_label.setText(name or self.tr("Unsaved workflow"))
        if template_preview:
            status_text = self.tr("Not modified")
        else:
            status_text = (
                self.tr("Unsaved changes") if dirty else self.tr("All changes saved")
            )
        self.dirty_label.setText(status_text)
        self.dirty_label.setStyleSheet(
            "color: #b36b00;" if dirty else "color: #718087;"
        )
        self.save_button.setText(
            self.tr("Save workflow")
            if workflow_id is not None
            else self.tr("Save as workflow")
        )
        self.save_button.setIcon(
            FluentIcon.SAVE if workflow_id is not None else None
        )
        save_action_name = (
            self.tr("Save workflow")
            if workflow_id is not None
            else self.tr("Save as workflow")
        )
        self.save_button.setToolTip(save_action_name)
        self.save_button.setAccessibleName(save_action_name)
        self.copy_button.setEnabled(has_cards)
        self._can_start_new = bool(has_cards or workflow_id or dirty)
        self.new_shortcut_action.setEnabled(self._can_start_new)
        for widget in (
            self.workflow_list,
            self.builtin_template_list,
            self.template_list,
        ):
            for index in range(widget.count()):
                item = widget.item(index)
                data = self._item_data(item) or {}
                item.setData(_CURRENT_ROLE, data.get("id") == workflow_id)
        self.workflow_list.viewport().update()
        self.builtin_template_list.viewport().update()
        self.template_list.viewport().update()

    def _filter_lists(self, text: str) -> None:
        query = text.strip().casefold()
        for widget in (
            self.workflow_list,
            self.builtin_template_list,
            self.template_list,
        ):
            for index in range(widget.count()):
                item = widget.item(index)
                data = self._item_data(item) or {}
                item.setHidden(query not in str(data.get("search_text", "")).casefold())


__all__ = [
    "WorkflowLibraryPanel",
    "localized_workflow_entry_description",
    "localized_workflow_entry_name",
    "localized_workflow_input_requirement",
]
