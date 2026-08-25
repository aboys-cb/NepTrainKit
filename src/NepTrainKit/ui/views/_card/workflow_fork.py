"""Permanent workflow fork with independent linear branch pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase.io import write as ase_write
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, StrongBodyLabel

from NepTrainKit.core import CardManager
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.widgets.card_widget import MakeDataCardWidget


@dataclass
class WorkflowBranch:
    """A named linear card sequence owned by a permanent fork."""

    branch_id: str
    name: str
    enabled: bool = True


class ForkConnector(QWidget):
    """Paint a compact one-to-many connector without a graph scene."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._branch_count = 2
        self.setFixedHeight(56)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_branch_count(self, count: int) -> None:
        self._branch_count = max(1, int(count))
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        connector_color = QColor(157, 178, 184)
        accent_color = QColor(15, 143, 145)
        painter.setPen(
            QPen(
                connector_color,
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        center_x = self.width() // 2
        junction_y = 13
        bottom_y = self.height() - 5
        positions = [
            self.width() * (2 * index + 1) // (2 * self._branch_count)
            for index in range(self._branch_count)
        ]
        painter.drawLine(center_x, 0, center_x, junction_y)
        for x in positions:
            path = QPainterPath(QPointF(center_x, junction_y))
            path.cubicTo(
                QPointF(center_x, junction_y + 13),
                QPointF(x, bottom_y - 18),
                QPointF(x, bottom_y - 6),
            )
            painter.drawPath(path)
            painter.setBrush(connector_color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4, bottom_y - 7),
                        QPointF(x + 4, bottom_y - 7),
                        QPointF(x, bottom_y),
                    ]
                )
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_color)
        painter.drawEllipse(QPointF(center_x, junction_y), 4, 4)


class BranchCardsHost(QWidget):
    """Paint directional connectors between cards inside one branch lane."""

    def __init__(self, branch, parent=None):
        super().__init__(parent)
        self.branch = branch
        self._drop_index: int | None = None

    def set_drop_index(self, index: int | None) -> None:
        if self._drop_index == index:
            return
        self._drop_index = index
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        cards = [card for card in self.branch.cards if card.isVisible()]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        connector_color = QColor(157, 178, 184)
        painter.setPen(
            QPen(
                connector_color,
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.setBrush(connector_color)
        for upper, lower in zip(cards, cards[1:]):
            x = upper.geometry().center().x()
            start_y = upper.geometry().bottom() + 2
            tip_y = lower.geometry().top() - 2
            if tip_y - start_y < 8:
                continue
            painter.drawLine(x, start_y, x, tip_y - 5)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4, tip_y - 6),
                        QPointF(x + 4, tip_y - 6),
                        QPointF(x, tip_y),
                    ]
                )
            )
        if self._drop_index is None:
            return
        if not cards:
            indicator_y = 22
        elif self._drop_index <= 0:
            indicator_y = cards[0].geometry().top() - 8
        elif self._drop_index >= len(cards):
            indicator_y = cards[-1].geometry().bottom() + 8
        else:
            indicator_y = (
                cards[self._drop_index - 1].geometry().bottom()
                + cards[self._drop_index].geometry().top()
            ) // 2
        accent = QColor(15, 143, 145)
        center_x = self.width() // 2
        half_width = min(120, max(70, self.width() // 4))
        painter.setPen(QPen(accent, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(accent)
        painter.drawLine(center_x - half_width, indicator_y, center_x + half_width, indicator_y)
        painter.drawEllipse(QPointF(center_x - half_width, indicator_y), 3, 3)
        painter.drawEllipse(QPointF(center_x + half_width, indicator_y), 3, 3)


class WorkflowBranchWidget(QFrame):
    """Visual lane containing one independent linear card sequence."""

    cardDropRequested = Signal(object, object, int)
    cardSelected = Signal(object)
    branchSelected = Signal(object)

    def __init__(self, spec: WorkflowBranch, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.cards: list[MakeDataCardWidget] = []
        self._dragged_card: MakeDataCardWidget | None = None
        self._drag_placeholder: QFrame | None = None
        self._drag_slot_index: int | None = None
        self.setObjectName("workflowBranchLane")
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setStyleSheet(
            "QFrame#workflowBranchLane {"
            "border: 1px solid rgba(110, 130, 138, 45);"
            "border-radius: 10px; background: rgba(255,255,255,150); }"
            "QFrame#workflowBranchLane[workflowBranchSelected=\"true\"] {"
            "border: 1px solid rgba(79,127,189,190);"
            "background: rgba(79,127,189,12); }"
        )

        self.enabled_checkbox = QCheckBox(self)
        self.enabled_checkbox.setChecked(spec.enabled)
        self.enabled_checkbox.setAccessibleName(self.tr("Enable branch"))
        self.badge_label = QLabel(spec.branch_id, self)
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_label.setFixedSize(26, 24)
        self.badge_label.setStyleSheet(
            "border: 0; border-radius: 6px; color: #28709c;"
            "background: #eaf3fb; font-weight: 700;"
        )
        self.name_label = StrongBodyLabel(spec.name, self)
        self.output_label = CaptionLabel(self.tr("Not run"), self)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 2, 2, 8)
        header_layout.setSpacing(7)
        header_layout.addWidget(self.enabled_checkbox)
        header_layout.addWidget(self.badge_label)
        header_layout.addWidget(self.name_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.output_label)

        self.cards_host = BranchCardsHost(self, self)
        self.cards_layout = QVBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(22, 2, 0, 0)
        self.cards_layout.setSpacing(16)
        self.empty_label = CaptionLabel(
            self.tr("Drop cards here to build this branch."), self.cards_host
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.cards_layout.addWidget(self.empty_label)
        self.cards_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        layout.addLayout(header_layout)
        layout.addWidget(self.cards_host)

        self.enabled_checkbox.toggled.connect(self._sync_enabled)
        for widget in (
            self.enabled_checkbox,
            self.badge_label,
            self.name_label,
            self.output_label,
        ):
            widget.installEventFilter(self)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("workflowBranchSelected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _sync_enabled(self, enabled: bool) -> None:
        self.spec.enabled = bool(enabled)
        self.cards_host.setEnabled(enabled)

    def add_card(self, card: MakeDataCardWidget, index: int | None = None) -> None:
        if card in self.cards:
            old_index = self.cards.index(card)
            self.cards.pop(old_index)
            self.cards_layout.removeWidget(card)
        if index is None:
            index = len(self.cards)
        index = max(0, min(index, len(self.cards)))
        self.cards.insert(index, card)
        card.setParent(self.cards_host)
        card.set_compact_header(True)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.cards_layout.insertWidget(index, card)
        self.empty_label.hide()
        card.collapse_button.clicked.connect(
            lambda _checked=False, selected=card: self.cardSelected.emit(selected)
        )
        card.headerView.installEventFilter(self)
        card.headerLabel.installEventFilter(self)
        card.close_button.clicked.connect(
            lambda _checked=False, closed=card: self._forget_closed_card(closed)
        )
        card.dragStartedSignal.connect(self._on_drag_started)
        card.dragFinishedSignal.connect(self._on_drag_finished)
        card.show()
        self.cards_host.update()

    def _on_drag_started(self, card: MakeDataCardWidget) -> None:
        if card not in self.cards or self._dragged_card is not None:
            return
        self._dragged_card = card
        index = self.cards.index(card)
        self._drag_slot_index = index
        placeholder = QFrame(self.cards_host)
        placeholder.setObjectName("branchDragPlaceholder")
        placeholder.setFixedSize(card.width(), min(76, max(54, card.headerView.height() + 14)))
        placeholder.setStyleSheet(
            "QFrame#branchDragPlaceholder {"
            "border: 2px dashed rgba(15,143,145,150); border-radius: 8px;"
            "background: rgba(15,143,145,12); }"
        )
        placeholder_label = CaptionLabel(self.tr("Move card here"), placeholder)
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("color: #087f81; font-weight: 600;")
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_layout.setContentsMargins(6, 6, 6, 6)
        placeholder_layout.addWidget(placeholder_label)
        self.cards_layout.removeWidget(card)
        card.hide()
        self.cards_layout.insertWidget(index, placeholder)
        placeholder.show()
        self._drag_placeholder = placeholder
        self.cards_host.set_drop_index(None)

    def _on_drag_finished(self, card: MakeDataCardWidget, _moved: bool) -> None:
        if self._dragged_card is not card:
            return
        placeholder = self._drag_placeholder
        if placeholder is not None:
            self.cards_layout.removeWidget(placeholder)
            placeholder.deleteLater()
        if card in self.cards and self.cards_layout.indexOf(card) < 0:
            self.cards_layout.insertWidget(self.cards.index(card), card)
            card.show()
        self._dragged_card = None
        self._drag_placeholder = None
        self._drag_slot_index = None
        self.cards_host.set_drop_index(None)
        self.cards_host.update()

    def _drop_slot(self, point_y: int) -> int:
        cards = [card for card in self.cards if card is not self._dragged_card]
        if not cards:
            return 0
        spacing = self.cards_layout.spacing()
        slots = [cards[0].geometry().top() - spacing // 2]
        slots.extend(
            (upper.geometry().bottom() + lower.geometry().top()) // 2
            for upper, lower in zip(cards, cards[1:])
        )
        slots.append(cards[-1].geometry().bottom() + spacing // 2)
        distances = [abs(point_y - slot_y) for slot_y in slots]
        candidate = min(range(len(slots)), key=distances.__getitem__)
        current = self._drag_slot_index
        if (
            current is not None
            and 0 <= current < len(slots)
            and current != candidate
            and distances[current] <= distances[candidate] + 12
        ):
            return current
        return candidate

    def _preview_drop_slot(self, index: int) -> None:
        placeholder = self._drag_placeholder
        if placeholder is None:
            self.cards_host.set_drop_index(index)
            return
        cards = [card for card in self.cards if card is not self._dragged_card]
        index = max(0, min(index, len(cards)))
        self._drag_slot_index = index
        self.cards_layout.removeWidget(placeholder)
        self.cards_layout.insertWidget(index, placeholder)
        self.cards_host.set_drop_index(None)

    def _workflow_area(self):
        parent = self.parentWidget()
        while parent is not None and not hasattr(parent, "_update_drag_auto_scroll"):
            parent = parent.parentWidget()
        return parent

    def _forget_closed_card(self, card: MakeDataCardWidget) -> None:
        if card.isVisible():
            return
        if card not in self.cards:
            return
        self.cards.remove(card)
        self.cards_layout.removeWidget(card)
        self.empty_label.setVisible(not self.cards)
        self.cards_host.update()

    def remove_card(self, card: MakeDataCardWidget) -> bool:
        if card not in self.cards:
            return False
        self.cards.remove(card)
        self.cards_layout.removeWidget(card)
        card.setParent(None)
        self.empty_label.setVisible(not self.cards)
        self.cards_host.update()
        return True

    def set_output_count(self, count: int | None, outcome: str = "idle") -> None:
        if outcome == "failed":
            self.output_label.setText(self.tr("Failed"))
        elif outcome == "canceled":
            self.output_label.setText(self.tr("Stopped"))
        elif count is None:
            self.output_label.setText(self.tr("Not run"))
        else:
            self.output_label.setText(
                self.tr("{count} structures").format(count=count)
            )

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if watched in (
                self.enabled_checkbox,
                self.badge_label,
                self.name_label,
                self.output_label,
            ):
                self.branchSelected.emit(self)
            card = next(
                (
                    item
                    for item in self.cards
                    if watched in (item.headerView, item.headerLabel)
                ),
                None,
            )
            if card is not None:
                if card.__class__.__name__ == "CardGroup":
                    card.collapse()
                    self.cardSelected.emit(card)
                    return True
                self.cardSelected.emit(card)
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event) -> None:
        source = event.source()
        if isinstance(source, MakeDataCardWidget) and not isinstance(source, WorkflowFork):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        source = event.source()
        if not isinstance(source, MakeDataCardWidget) or isinstance(source, WorkflowFork):
            event.ignore()
            return
        point = self.cards_host.mapFrom(self, event.position().toPoint())
        index = self._drop_slot(point.y())
        self._preview_drop_slot(index)
        workflow_area = self._workflow_area()
        if workflow_area is not None:
            workflow_area.canvas.set_drop_index(None)
            workflow_area._drag_canvas_point = None
            viewport_point = workflow_area.scroll_area.viewport().mapFromGlobal(
                self.mapToGlobal(event.position().toPoint())
            )
            workflow_area._update_drag_auto_scroll(viewport_point.y())
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self.cards_host.set_drop_index(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self.cards_host.set_drop_index(None)
        source = event.source()
        if not isinstance(source, MakeDataCardWidget) or isinstance(source, WorkflowFork):
            event.ignore()
            return
        point = self.cards_host.mapFrom(self, event.position().toPoint())
        index = self._drop_slot(point.y())
        self.cardDropRequested.emit(source, self, index)
        event.acceptProposedAction()


@CardManager.register_card
class WorkflowFork(MakeDataCardWidget):
    """Create persistent branch pipelines that merge only when explicitly enabled."""

    group = "Container"
    separator = True
    card_name = "Permanent Fork"
    menu_icon = r":/images/src/images/group.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]
    description = (
        "Split one input into independent linear branch pipelines. Branch outputs stay "
        "separate unless an explicit merge is enabled."
    )
    runFinishedSignal = Signal(int)
    cardDropRequested = Signal(object, object, int)
    cardSelected = Signal(object)
    branchSelected = Signal(object)
    structureChanged = Signal()
    _MAX_BRANCHES = 3
    _STACK_BRANCHES_BREAKPOINT = 760

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Permanent Fork"))
        self.setAcceptDrops(True)
        self.dataset: Any = []
        self.result_dataset: list[Any] = []
        self.branch_results: dict[str, list[Any]] = {}
        self.branch_outcomes: dict[str, str] = {}
        self.run_outcome = "idle"
        self.index = 0
        self.merge_enabled = False
        self._running_branch_index = 0
        self._running_cards: list[MakeDataCardWidget] = []
        self._running_card_index = 0
        self._current_branch_data: list[Any] = []
        self._stopping = False
        self.branches: list[WorkflowBranchWidget] = []

        self.description_label = CaptionLabel(
            self.tr(
                "Each branch receives the same input and keeps its own linear data flow."
            ),
            self,
        )
        self.description_label.setWordWrap(True)
        self.connector = ForkConnector(self)
        self.branch_host = QWidget(self)
        self.branch_layout = QHBoxLayout(self.branch_host)
        self.branch_layout.setContentsMargins(0, 0, 0, 0)
        self.branch_layout.setSpacing(14)
        self.branch_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.merge_checkbox = QCheckBox(self.tr("Explicitly merge branch outputs"), self)
        self.merge_checkbox.setToolTip(
            self.tr("When enabled, concatenate successful branch outputs in branch order.")
        )
        self.merge_checkbox.toggled.connect(self._set_merge_enabled)
        self.merge_checkbox.hide()
        self.add_branch_button = QPushButton(self.tr("+ Add branch"), self)
        self.add_branch_button.clicked.connect(self.add_branch)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.add_branch_button)

        self.output_mode_frame = QFrame(self)
        self.output_mode_frame.setObjectName("forkOutputMode")
        self.output_mode_frame.setStyleSheet(
            "QFrame#forkOutputMode {"
            "border-top: 1px dashed rgba(110,130,138,70); background: transparent; }"
            "QFrame#forkOutputMode QPushButton {"
            "border: 1px solid rgba(110,130,138,55); padding: 0 13px;"
            "background: rgba(232,237,238,210); color: #627078; }"
            "QFrame#forkOutputMode QPushButton:checked {"
            "border-color: rgba(15,143,145,120); background: white;"
            "color: #176f70; font-weight: 600; }"
        )
        output_layout = QVBoxLayout(self.output_mode_frame)
        output_layout.setContentsMargins(0, 12, 0, 0)
        output_layout.setSpacing(8)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self.keep_separate_button = QPushButton(
            self.tr("Keep independent outputs"), self.output_mode_frame
        )
        self.merge_output_button = QPushButton(
            self.tr("Merge into one output"), self.output_mode_frame
        )
        for button in (self.keep_separate_button, self.merge_output_button):
            button.setCheckable(True)
            button.setFixedHeight(32)
        self.output_mode_group = QButtonGroup(self.output_mode_frame)
        self.output_mode_group.setExclusive(True)
        self.output_mode_group.addButton(self.keep_separate_button)
        self.output_mode_group.addButton(self.merge_output_button)
        self.keep_separate_button.setChecked(True)
        self.keep_separate_button.clicked.connect(
            lambda: self.merge_checkbox.setChecked(False)
        )
        self.merge_output_button.clicked.connect(
            lambda: self.merge_checkbox.setChecked(True)
        )
        mode_row.addStretch(1)
        mode_row.addWidget(self.keep_separate_button)
        mode_row.addWidget(self.merge_output_button)
        mode_row.addStretch(1)
        output_layout.addLayout(mode_row)

        self.output_terminal = QFrame(self.output_mode_frame)
        self.output_terminal.setObjectName("forkOutputTerminal")
        terminal_layout = QHBoxLayout(self.output_terminal)
        terminal_layout.setContentsMargins(10, 7, 10, 7)
        self.output_terminal_title = StrongBodyLabel(
            self.tr("Independent branch outputs"), self.output_terminal
        )
        self.output_terminal_detail = CaptionLabel(
            self.tr("Shared downstream cards are unavailable"),
            self.output_terminal,
        )
        terminal_layout.addStretch(1)
        terminal_layout.addWidget(self.output_terminal_title)
        terminal_layout.addWidget(self.output_terminal_detail)
        terminal_layout.addStretch(1)
        output_layout.addWidget(self.output_terminal)
        self._refresh_output_terminal()

        self.body_widget = QWidget(self)
        body_layout = QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)
        body_layout.addWidget(self.description_label)
        body_layout.addWidget(self.connector)
        body_layout.addWidget(self.branch_host)
        body_layout.addLayout(footer)
        body_layout.addWidget(self.output_mode_frame)
        self.viewLayout.addWidget(self.body_widget)
        self.collapsed_summary = CaptionLabel("", self)
        self.collapsed_summary.setWordWrap(True)
        self.collapsed_summary.setStyleSheet("color:#8a95a0; padding: 4px 8px;")
        self.collapsed_summary.hide()
        self.vBoxLayout.addWidget(self.collapsed_summary)
        self.windowStateChangedSignal.connect(self._show_body)
        self.exportSignal.connect(self.export_data)

        self.add_branch("A", self.tr("Branch A"))
        self.add_branch("B", self.tr("Branch B"))
        self.set_export_available(False)

    @property
    def requires_input_dataset(self) -> bool:
        return any(
            getattr(card, "requires_input_dataset", True)
            for branch in self.branches
            if branch.spec.enabled
            for card in branch.cards
            if card.check_state
        )

    def _show_body(self) -> None:
        self.body_widget.setVisible(self.window_state == "expand")
        self.collapsed_summary.setText(self.get_summary_text())
        self.collapsed_summary.setVisible(self.window_state != "expand")

    def _set_merge_enabled(self, enabled: bool) -> None:
        self.merge_enabled = bool(enabled)
        self._refresh_output_terminal()
        self.set_export_available(self.merge_enabled and bool(self.result_dataset))
        self.structureChanged.emit()

    def _refresh_output_terminal(self) -> None:
        if not hasattr(self, "output_terminal"):
            return
        self.keep_separate_button.setChecked(not self.merge_enabled)
        self.merge_output_button.setChecked(self.merge_enabled)
        if self.merge_enabled:
            self.output_terminal_title.setText(self.tr("Explicit merge"))
            self.output_terminal_detail.setText(
                self.tr("Shared downstream cards are available")
            )
            self.output_terminal.setStyleSheet(
                "QFrame#forkOutputTerminal {"
                "border: 1px solid rgba(46,150,96,90); border-radius: 8px;"
                "background: rgba(46,150,96,18); }"
            )
        else:
            self.output_terminal_title.setText(
                self.tr("Independent branch outputs")
            )
            self.output_terminal_detail.setText(
                self.tr("Shared downstream cards are unavailable")
            )
            self.output_terminal.setStyleSheet(
                "QFrame#forkOutputTerminal {"
                "border: 1px solid rgba(110,130,138,55); border-radius: 8px;"
                "background: rgba(248,250,251,220); }"
            )

    def get_summary_text(self) -> str:
        enabled = sum(branch.spec.enabled for branch in self.branches)
        mode = self.tr("explicit Merge") if self.merge_enabled else self.tr("independent outputs")
        return self.tr("{enabled}/{total} branches · {mode}").format(
            enabled=enabled,
            total=len(self.branches),
            mode=mode,
        )

    def get_guidance_text(self) -> str:
        if self.merge_enabled:
            return self.tr(
                "Branches keep independent linear pipelines until the explicit Merge, "
                "which concatenates successful outputs in branch order."
            )
        return self.tr(
            "Branches keep independent linear pipelines and final outputs. "
            "Add an explicit Merge before any shared downstream card."
        )

    def get_inspector_overview_text(self) -> str:
        rows = [
            self.tr("Flow structure"),
            self.tr("Common input → independent linear branch pipelines"),
            "",
            self.tr("Branches"),
        ]
        for branch in self.branches:
            rows.append(
                self.tr("{id} · {name} · {count} cards · {status}").format(
                    id=branch.spec.branch_id,
                    name=branch.spec.name,
                    count=len(branch.cards),
                    status=branch.output_label.text(),
                )
            )
        rows.extend(
            [
                "",
                self.tr("Output mode"),
                self.tr("Explicit merge")
                if self.merge_enabled
                else self.tr("Independent outputs"),
            ]
        )
        return "\n".join(rows)

    def add_branch(
        self,
        branch_id: str | None = None,
        name: str | None = None,
        enabled: bool = True,
    ) -> WorkflowBranchWidget:
        if len(self.branches) >= self._MAX_BRANCHES:
            raise ValueError(self.tr("A permanent fork supports up to three branches."))
        if branch_id is None:
            branch_id = chr(ord("A") + len(self.branches))
        if name is None:
            name = self.tr("Branch {branch}").format(branch=branch_id)
        branch = WorkflowBranchWidget(
            WorkflowBranch(str(branch_id), str(name), bool(enabled)), self.branch_host
        )
        branch.cardDropRequested.connect(self.cardDropRequested)
        branch.cardSelected.connect(self.cardSelected)
        branch.branchSelected.connect(self.branchSelected)
        branch.enabled_checkbox.toggled.connect(
            lambda _enabled: self.structureChanged.emit()
        )
        self.branches.append(branch)
        self.branch_layout.addWidget(branch, 1, Qt.AlignmentFlag.AlignTop)
        self.connector.set_branch_count(len(self.branches))
        self.add_branch_button.setEnabled(len(self.branches) < self._MAX_BRANCHES)
        self.structureChanged.emit()
        return branch

    def add_card(
        self,
        card: MakeDataCardWidget,
        branch: WorkflowBranchWidget | int = 0,
        index: int | None = None,
    ) -> bool:
        if isinstance(card, WorkflowFork):
            return False
        target = self.branches[branch] if isinstance(branch, int) else branch
        if target not in self.branches:
            return False
        self.remove_card(card)
        card.setMinimumWidth(0)
        card.setMaximumWidth(16777215)
        target.add_card(card, index)
        self.structureChanged.emit()
        return True

    def remove_card(self, card: MakeDataCardWidget) -> bool:
        for branch in self.branches:
            if branch.remove_card(card):
                self.structureChanged.emit()
                return True
        return False

    def terminal_cards(self) -> list[MakeDataCardWidget]:
        terminals = []
        for branch in self.branches:
            enabled = [card for card in branch.cards if card.check_state]
            if branch.spec.enabled and enabled:
                terminals.append(enabled[-1])
        return terminals

    def available_output_cards(self) -> list[MakeDataCardWidget]:
        if self.merge_enabled and self.result_dataset:
            return [self]
        return [
            card
            for card in self.terminal_cards()
            if getattr(card, "result_dataset", None)
        ]

    def write_result_dataset(self, file, **kwargs) -> None:
        if not self.merge_enabled:
            raise ValueError(self.tr("Merge branch outputs before exporting one combined file."))
        export_dataset = [
            prepare_magnetic_extxyz_export(atoms) for atoms in self.result_dataset
        ]
        ase_write(file, export_dataset, format="extxyz", **kwargs)

    def export_data(self) -> None:
        if not self.merge_enabled or not self.result_dataset:
            return
        path = call_path_dialog(
            self,
            self.tr("Choose a file save location"),
            "file",
            "export_merged_branches.xyz",
            file_filter="XYZ Files (*.xyz)",
        )
        if not path:
            return
        thread = BackgroundTask(self, show_tip=True, title=self.tr("Exporting data"))
        thread.start_work(self.write_result_dataset, path)

    def set_dataset(self, dataset) -> None:
        self.dataset = dataset or []
        self.result_dataset = []
        self.branch_results = {}
        self.branch_outcomes = {}
        self.run_outcome = "idle"
        self.set_output_available(False)
        for branch in self.branches:
            branch.set_output_count(None)
            for card in branch.cards:
                card.set_dataset([])

    def run(self) -> None:
        if not self.check_state:
            self.result_dataset = list(self.dataset)
            self.run_outcome = "succeeded"
            self.runFinishedSignal.emit(self.index)
            return
        self._stopping = False
        self.result_dataset = []
        self.branch_results = {}
        self.branch_outcomes = {}
        self.run_outcome = "running"
        self.status_dot.set_state("running")
        self._running_branch_index = 0
        self._start_next_branch()

    def _start_next_branch(self) -> None:
        if self._stopping:
            return
        while self._running_branch_index < len(self.branches):
            branch = self.branches[self._running_branch_index]
            if not branch.spec.enabled:
                self.branch_outcomes[branch.spec.branch_id] = "disabled"
                self.branch_results[branch.spec.branch_id] = []
                self._running_branch_index += 1
                continue
            self._running_cards = [card for card in branch.cards if card.check_state]
            self._running_card_index = 0
            self._current_branch_data = list(self.dataset)
            if not self._running_cards:
                self._finish_current_branch("failed")
                return
            self._start_current_card()
            return
        self._finish_fork()

    def _start_current_card(self) -> None:
        card = self._running_cards[self._running_card_index]
        card.index = self._running_card_index
        card.set_dataset(self._current_branch_data)
        card.runFinishedSignal.connect(self._on_child_finished)
        card.run()

    def _on_child_finished(self, _index: int) -> None:
        card = self._running_cards[self._running_card_index]
        try:
            card.runFinishedSignal.disconnect(self._on_child_finished)
        except (RuntimeError, TypeError):
            pass
        outcome = str(getattr(card, "run_outcome", "failed"))
        if outcome != "succeeded":
            self._finish_current_branch(outcome)
            return
        self._current_branch_data = list(card.result_dataset)
        self._running_card_index += 1
        if self._running_card_index < len(self._running_cards):
            self._start_current_card()
        else:
            self._finish_current_branch("succeeded")

    def _finish_current_branch(self, outcome: str) -> None:
        branch = self.branches[self._running_branch_index]
        result = list(self._current_branch_data) if outcome == "succeeded" else []
        self.branch_outcomes[branch.spec.branch_id] = outcome
        self.branch_results[branch.spec.branch_id] = result
        branch.set_output_count(len(result), outcome)
        self._running_branch_index += 1
        self._start_next_branch()

    def _finish_fork(self) -> None:
        enabled_ids = [
            branch.spec.branch_id for branch in self.branches if branch.spec.enabled
        ]
        failures = [
            branch_id
            for branch_id in enabled_ids
            if self.branch_outcomes.get(branch_id) != "succeeded"
        ]
        if not enabled_ids:
            self.result_dataset = []
            self.run_outcome = "failed"
            self.set_output_available(False)
            self.status_dot.set_state("failed")
            self.runFinishedSignal.emit(self.index)
            return
        if self.merge_enabled:
            if failures:
                self.result_dataset = []
                self.run_outcome = "failed"
            else:
                self.result_dataset = [
                    item
                    for branch_id in enabled_ids
                    for item in self.branch_results.get(branch_id, [])
                ]
                self.run_outcome = "succeeded"
        else:
            self.result_dataset = []
            self.run_outcome = "partial_failed" if failures else "succeeded"
        self.set_output_available(bool(self.available_output_cards()))
        self.set_export_available(self.merge_enabled and bool(self.result_dataset))
        self.status_dot.set_state(
            "failed" if self.run_outcome in ("failed", "partial_failed") else "succeeded"
        )
        self.runFinishedSignal.emit(self.index)

    def stop(self) -> None:
        import warnings

        self._stopping = True
        for branch in self.branches:
            for card in branch.cards:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        card.runFinishedSignal.disconnect(self._on_child_finished)
                except (RuntimeError, TypeError):
                    pass
                card.stop()
        self.run_outcome = "canceled"
        self.result_dataset = []
        self.set_output_available(False)
        self.status_dot.set_state("canceled")

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["merge"] = self.merge_enabled
        data["branches"] = [
            {
                "id": branch.spec.branch_id,
                "name": branch.spec.name,
                "enabled": branch.spec.enabled,
                "cards": [card.to_dict() for card in branch.cards],
            }
            for branch in self.branches
        ]
        return data

    def from_dict(self, data: dict[str, Any]) -> None:
        super().from_dict(data)
        for branch in self.branches:
            for card in list(branch.cards):
                branch.remove_card(card)
                card.close()
            branch.setParent(None)
            branch.deleteLater()
        self.branches.clear()

        branch_data = data.get("branches") or []
        if len(branch_data) > self._MAX_BRANCHES:
            raise ValueError(self.tr("A permanent fork supports up to three branches."))
        for branch_index, branch_config in enumerate(branch_data):
            branch_id = str(branch_config.get("id") or chr(ord("A") + branch_index))
            branch = self.add_branch(
                branch_id,
                str(branch_config.get("name") or self.tr("Branch {branch}").format(branch=branch_id)),
                bool(branch_config.get("enabled", True)),
            )
            for card_config in branch_config.get("cards", []):
                card_name = card_config.get("class")
                card_class = CardManager.card_info_dict.get(card_name)
                if card_class is None or card_class is WorkflowFork:
                    raise ValueError(self.tr("Invalid card in permanent branch: {name}").format(name=card_name))
                card = card_class(self)
                card.from_dict(card_config)
                self.add_card(card, branch)
        if not self.branches:
            self.add_branch("A", self.tr("Branch A"))
            self.add_branch("B", self.tr("Branch B"))
        self.merge_checkbox.setChecked(bool(data.get("merge", False)))

    def resizeEvent(self, event) -> None:
        stacked = event.size().width() < self._STACK_BRANCHES_BREAKPOINT
        direction = (
            QBoxLayout.Direction.TopToBottom
            if stacked
            else QBoxLayout.Direction.LeftToRight
        )
        if self.branch_layout.direction() != direction:
            self.branch_layout.setDirection(direction)
        self.connector.setVisible(not stacked)
        super().resizeEvent(event)

    def _nearest_branch(self, point) -> WorkflowBranchWidget | None:
        visible = [branch for branch in self.branches if branch.isVisible()]
        if not visible:
            return None
        host_point = self.branch_host.mapFrom(self, point)
        stacked = self.branch_layout.direction() == QBoxLayout.Direction.TopToBottom
        return min(
            visible,
            key=lambda branch: abs(
                (
                    branch.geometry().center().y() - host_point.y()
                    if stacked
                    else branch.geometry().center().x() - host_point.x()
                )
            ),
        )

    def dragEnterEvent(self, event) -> None:
        source = event.source()
        if (
            not isinstance(source, MakeDataCardWidget)
            or isinstance(source, WorkflowFork)
            or source is self
        ):
            event.ignore()
            return
        if self.window_state != "expand":
            self.window_state = "expand"
            self.windowStateChangedSignal.emit()
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        source = event.source()
        if isinstance(source, MakeDataCardWidget) and not isinstance(
            source, WorkflowFork
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        source = event.source()
        if not isinstance(source, MakeDataCardWidget) or isinstance(
            source, WorkflowFork
        ):
            event.ignore()
            return
        branch = self._nearest_branch(event.position().toPoint())
        if branch is None:
            event.ignore()
            return
        global_point = self.mapToGlobal(event.position().toPoint())
        branch_point = branch.cards_host.mapFromGlobal(global_point)
        self.cardDropRequested.emit(
            source,
            branch,
            branch._drop_slot(branch_point.y()),
        )
        event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        self.stop()
        for branch in self.branches:
            for card in list(branch.cards):
                card.close()
        super().closeEvent(event)
