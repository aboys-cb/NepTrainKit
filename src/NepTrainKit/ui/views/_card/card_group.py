"""Composite card that fans one input out and merges independent outputs."""

from typing import Any

from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, StrongBodyLabel
from shiboken6 import isValid

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.widgets import FilterDataCard, MakeDataCard, MakeDataCardWidget


class FanOutCardsHost(QWidget):
    """Paint the shared-input and automatic-merge paths behind group cards."""

    def __init__(self, group, parent=None):
        super().__init__(parent)
        self.group = group

    def paintEvent(self, event):  # noqa: N802 - Qt override
        super().paintEvent(event)
        cards = [card for card in self.group.card_list if card.isVisible()]
        if not cards:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(157, 178, 184)
        painter.setPen(
            QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        center_x = self.width() / 2
        split_y = 15
        merge_y = self.height() - 15
        painter.drawLine(QPointF(center_x, 0), QPointF(center_x, split_y))
        if self.group._grid_columns == 1 and len(cards) > 1:
            first_y = cards[0].geometry().center().y()
            last_y = cards[-1].geometry().center().y()
            left_rail = max(14, min(card.geometry().left() for card in cards) - 20)
            right_rail = min(
                self.width() - 14,
                max(card.geometry().right() for card in cards) + 20,
            )
            split = QPainterPath(QPointF(center_x, split_y))
            split.cubicTo(
                QPointF(center_x, split_y + 14),
                QPointF(left_rail, first_y - 18),
                QPointF(left_rail, first_y),
            )
            painter.drawPath(split)
            painter.drawLine(
                QPointF(left_rail, first_y), QPointF(left_rail, last_y)
            )
            painter.drawLine(
                QPointF(right_rail, first_y), QPointF(right_rail, last_y)
            )
            for card in cards:
                y = card.geometry().center().y()
                painter.drawLine(
                    QPointF(left_rail, y),
                    QPointF(card.geometry().left() - 2, y),
                )
                painter.drawLine(
                    QPointF(card.geometry().right() + 2, y),
                    QPointF(right_rail, y),
                )
            merge = QPainterPath(QPointF(right_rail, last_y))
            merge.cubicTo(
                QPointF(right_rail, last_y + 18),
                QPointF(center_x, merge_y - 14),
                QPointF(center_x, merge_y),
            )
            painter.drawPath(merge)
        else:
            for card in cards:
                x = card.geometry().center().x()
                top = card.geometry().top() - 2
                bottom = card.geometry().bottom() + 2
                split = QPainterPath(QPointF(center_x, split_y))
                split.cubicTo(
                    QPointF(center_x, split_y + 12),
                    QPointF(x, max(split_y + 12, top - 14)),
                    QPointF(x, top),
                )
                painter.drawPath(split)
                merge = QPainterPath(QPointF(x, bottom))
                merge.cubicTo(
                    QPointF(x, min(merge_y - 12, bottom + 14)),
                    QPointF(center_x, merge_y - 12),
                    QPointF(center_x, merge_y),
                )
                painter.drawPath(merge)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 143, 145))
        painter.drawEllipse(QPointF(center_x, split_y), 4, 4)
        painter.setBrush(color)
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(center_x - 4, merge_y - 1),
                    QPointF(center_x + 4, merge_y - 1),
                    QPointF(center_x, merge_y + 6),
                ]
            )
        )


@CardManager.register_card
class CardGroup(MakeDataCardWidget):
    """Run child cards from one shared input and concatenate their outputs.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the group card.
    
    Attributes
    ----------
    runFinishedSignal : Signal
        Emitted with the group index when execution finishes.
    filter_card : FilterDataCard or None
        Optional post-processing card applied to the aggregated dataset.
    """

    separator=True
    group = "Container"
    card_name= "Branch Merge"
    menu_icon=r":/images/src/images/group.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]
    runFinishedSignal=Signal(int)
    cardSelected = Signal(object)
    structureChanged = Signal()
    def __init__(self, parent=None):
        """Initialise layouts, drag-and-drop targets, and default execution state.
        """
        super().__init__(parent)
        self.setTitle(self.tr("Branch Merge"))
        self.setAcceptDrops(True)
        self.index=0
        self._cards: list[MakeDataCardWidget] = []
        self._grid_columns = 1
        self._active_card = None
        self._filter_signal_connected = False
        self.branch_widget = QWidget(self)
        self.branch_layout = QVBoxLayout(self.branch_widget)
        self.branch_layout.setContentsMargins(0, 0, 0, 0)
        self.branch_layout.setSpacing(0)
        self.branch_hint = CaptionLabel(
            self.tr("Common input · not loaded"),
            self,
        )
        self.branch_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.branch_hint.setFixedHeight(36)
        self.branch_hint.setStyleSheet(
            "padding: 0 12px; border: 1px solid rgba(110,130,138,55);"
            "border-radius: 8px; background: rgba(248,250,251,220);"
            "color: #53656c; font-weight: 600;"
        )
        input_row = QHBoxLayout()
        input_row.addStretch(1)
        input_row.addWidget(self.branch_hint)
        input_row.addStretch(1)
        self.branch_layout.addLayout(input_row)
        self.group_widget = FanOutCardsHost(self, self)
        self.group_layout = QGridLayout(self.group_widget)
        self.group_layout.setContentsMargins(0, 36, 0, 36)
        self.group_layout.setHorizontalSpacing(12)
        self.group_layout.setVerticalSpacing(10)
        self.group_empty_label = CaptionLabel(
            self.tr(
                "Select this group, then add or drop cards here. Each card receives the common input."
            ),
            self.group_widget,
        )
        self.group_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_empty_label.setWordWrap(True)
        self.group_layout.addWidget(self.group_empty_label, 0, 0, 1, 2)
        self.branch_layout.addWidget(self.group_widget)

        self.merge_frame = QFrame(self.branch_widget)
        self.merge_frame.setObjectName("fanOutMergeTerminal")
        self.merge_frame.setStyleSheet(
            "QFrame#fanOutMergeTerminal {"
            "border: 1px solid rgba(46,150,96,90); border-radius: 8px;"
            "background: rgba(46,150,96,18); }"
        )
        merge_layout = QHBoxLayout(self.merge_frame)
        merge_layout.setContentsMargins(10, 6, 10, 6)
        self.merge_title = StrongBodyLabel(self.tr("Automatic merge"), self.merge_frame)
        self.merge_count_label = CaptionLabel(
            self.tr("Waiting for branch outputs"), self.merge_frame
        )
        merge_layout.addWidget(self.merge_title)
        merge_layout.addWidget(self.merge_count_label)
        merge_row = QHBoxLayout()
        merge_row.addStretch(1)
        merge_row.addWidget(self.merge_frame)
        merge_row.addStretch(1)
        self.branch_layout.addLayout(merge_row)
        self.viewLayout.addWidget(self.branch_widget)
        self.exportSignal.connect(self.export_data)
        self.windowStateChangedSignal.connect(self.show_card_setting)
        self.filter_widget = QWidget(self)
        self.filter_hint = CaptionLabel(
            self.tr("Post-merge filter · Drop one filter card here (optional)."),
            self,
        )
        self.filter_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filter_hint.setWordWrap(True)
        self.filter_hint.setMinimumHeight(40)
        self.filter_hint.setStyleSheet(
            "padding: 6px 10px; border: 1px dashed rgba(110,130,138,85);"
            "border-radius: 8px; background: rgba(248,250,251,160); color: #5f7077;"
        )
        self.vBoxLayout.addWidget(self.filter_hint)
        self.filter_layout = QVBoxLayout(self.filter_widget)
        self.filter_layout.setContentsMargins(8, 4, 8, 8)
        self.vBoxLayout.addWidget(self.filter_widget)
        self.summary_label = CaptionLabel("", self)
        self.summary_label.setWordWrap(True)
        self.vBoxLayout.addWidget(self.summary_label)
        self.run_card_num:int
        self.filter_card=None
        self.dataset:Any=None
        self.result_dataset=[]
        self._merged_count: int | None = None
        self._post_filter_applied = False
        self.run_outcome = "idle"
        self.cards_to_run = []
        self.current_index = 0
        self.resize(820, 260)
        self._refresh_summary()

    def set_filter_card(self,card):
        """Attach a filter card that refines results after the grouped cards run.
        
        Parameters
        ----------
        card : QWidget
            Filter card widget to embed beneath the grouped cards.
        """
        if (
            self.filter_card is not None
            and isValid(self.filter_card)
            and self.filter_card is not card
        ):
            MessageManager.send_warning_message(
                self.tr(
                    "This group already has a post-filter. Close or move it before adding another."
                )
            )
            return False
        self.filter_card=card
        self.filter_layout.addWidget(card)
        card.set_compact_header(True)
        card.headerView.installEventFilter(self)
        card.headerLabel.installEventFilter(self)
        card.close_button.clicked.connect(
            lambda _checked=False, closed=card: self._forget_closed_filter(closed)
        )
        card.state_checkbox.stateChanged.connect(self._refresh_summary)
        if self.dataset is not None:
            card.set_dataset([])
        self._refresh_summary()
        return True

    def _forget_closed_filter(self, card):
        if card.isVisible():
            return
        if self.filter_card is not card:
            return
        self.filter_layout.removeWidget(card)
        self.filter_card = None
        self._refresh_summary()

    def state_changed(self, state):
        """Enable or bypass the group without changing child-card choices.
        
        Parameters
        ----------
        state : bool
            Toggle state propagated from the group header.
        """
        super().state_changed(state)
        self._refresh_summary()

    @property
    def card_list(self)->list["MakeDataCard"]:
        """List the child card widgets currently managed by the group.
        
        Returns
        -------
        list of MakeDataCard
            Ordered collection of child cards.
        """
        return list(self._cards)
    @property
    def requires_input_dataset(self):
        return any([getattr(card, "requires_input_dataset", True)  for card in self.card_list])

    def show_card_setting(self):
        """Collapse the composite as one workflow step without changing children."""
        expanded = self.window_state == "expand"
        self.branch_widget.setVisible(expanded)
        self.filter_hint.setVisible(expanded)
        self.filter_widget.setVisible(expanded)
        self.summary_label.setVisible(not expanded)
    def set_dataset(self,dataset):
        """Store the shared dataset reference and clear accumulated results.
        
        Parameters
        ----------
        dataset : Any
            Dataset that will be passed to each child card.
        """
        self.dataset =dataset
        self.result_dataset=[]
        self._merged_count = None
        self._post_filter_applied = False
        self.run_outcome = "idle"
        self.set_output_available(False)
        self._set_run_status("idle")
        for card in self.card_list:
            card.set_dataset(dataset)
        if self.filter_card and isValid(self.filter_card):
            self.filter_card.set_dataset([])
        self._refresh_summary()

    def add_card(self, card, *, preserve_legacy_branch: bool = False):
        """Insert a card widget into the group layout.
        
        Parameters
        ----------
        card : QWidget
            Card widget to append.
        """
        if isinstance(card, FilterDataCard) and not preserve_legacy_branch:
            return self.set_filter_card(card)
        if card is self or card in self.card_list:
            return False
        card.set_compact_header(True)
        card.set_group_tile_presentation(True)
        card.setMinimumWidth(0)
        card.setMaximumWidth(16777215)
        self._cards.append(card)
        card.setParent(self.group_widget)
        card.headerView.installEventFilter(self)
        card.headerLabel.installEventFilter(self)
        self._reflow_cards()
        card.close_button.clicked.connect(
            lambda _checked=False, closed=card: self._forget_closed_card(closed)
        )
        card.state_checkbox.stateChanged.connect(self._refresh_summary)
        if self.dataset is not None:
            card.set_dataset(self.dataset)
        self._refresh_summary()
        return True

    def _forget_closed_card(self, card):
        if card.isVisible():
            return
        if card not in self.card_list:
            return
        self._cards.remove(card)
        self.group_layout.removeWidget(card)
        self._reflow_cards()
        self._refresh_summary()

    def remove_card(self, card):
        """Remove a card widget from the group layout.
        
        Parameters
        ----------
        card : QWidget
            Card widget to detach.
        """
        self.group_layout.removeWidget(card)
        if card in self._cards:
            self._cards.remove(card)
        card.setParent(None)
        card.set_group_tile_presentation(False)
        card.setMinimumWidth(0)
        card.setMaximumWidth(16777215)
        self._reflow_cards()
        self._refresh_summary()

    def clear_cards(self):
        """Remove every child card from the layout.
        """
        for card in self.card_list:
            self.group_layout.removeWidget(card)
            card.close()
        self._cards.clear()
        self._reflow_cards()
        self._refresh_summary()

    def _reflow_cards(self) -> None:
        self._grid_columns = self._responsive_column_count(self.width())
        while self.group_layout.count():
            self.group_layout.takeAt(0)
        # QGridLayout keeps stretch factors for columns that are no longer used.
        # Clear both responsive columns before rebuilding, otherwise a wide →
        # narrow resize leaves an invisible second column consuming half the row.
        for column in range(3):
            self.group_layout.setColumnStretch(column, 0)
            self.group_layout.setColumnMinimumWidth(column, 0)
        if not self._cards:
            self.group_layout.addWidget(
                self.group_empty_label, 0, 0, 1, self._grid_columns
            )
            self.group_empty_label.show()
        else:
            self.group_empty_label.hide()
            available_width = max(0, self.width() - 24)
            card_width = max(
                0,
                (
                    available_width
                    - self.group_layout.horizontalSpacing()
                    * (self._grid_columns - 1)
                )
                // self._grid_columns,
            )
            for index, card in enumerate(self._cards):
                target_width = max(140, min(220, card_width or 220))
                card.setFixedWidth(target_width)
                row = index // self._grid_columns
                column = index % self._grid_columns
                if (
                    self._grid_columns == 2
                    and index == len(self._cards) - 1
                    and len(self._cards) % 2
                ):
                    self.group_layout.addWidget(
                        card,
                        row,
                        0,
                        1,
                        2,
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    )
                else:
                    self.group_layout.addWidget(
                        card,
                        row,
                        column,
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    )
            for column in range(self._grid_columns):
                self.group_layout.setColumnStretch(column, 1)
        self.group_widget.update()
        self.group_widget.updateGeometry()
        self.branch_widget.updateGeometry()
        self.updateGeometry()

    def _responsive_column_count(self, width: int) -> int:
        count = len(self._cards)
        if count <= 3:
            available_columns = 3 if width >= 450 else 1
        elif count == 4:
            available_columns = 4 if width >= 780 else 2 if width >= 450 else 1
        else:
            available_columns = 3 if width >= 620 else 2 if width >= 450 else 1
        return max(1, min(count or 1, available_columns))

    def _refresh_summary(self, *_args) -> None:
        if not hasattr(self, "summary_label"):
            return
        cards = self.card_list
        enabled = sum(bool(card.check_state) for card in cards)
        if self.dataset is None:
            input_text = self.tr("input not loaded")
        else:
            try:
                input_text = self.tr("{count} input structures").format(
                    count=len(self.dataset)
                )
            except TypeError:
                input_text = self.tr("input loaded")
        input_count = None
        if self.dataset is not None:
            try:
                input_count = len(self.dataset)
            except TypeError:
                pass
        has_filter = self.filter_card is not None and isValid(self.filter_card)
        filter_enabled = has_filter and bool(self.filter_card.check_state)
        filter_name = self.filter_card.getTitle() if has_filter else ""
        filter_text = (
            self.tr("post-filter: {name}").format(name=filter_name)
            if has_filter
            else self.tr("no post-filter")
        )
        if self.run_outcome == "succeeded":
            merged_count = self._merged_count
            if self._post_filter_applied and merged_count is not None:
                summary = self.tr(
                    "{input_count} input → {merged_count} merged → {result_count} kept"
                ).format(
                    input_count=input_count if input_count is not None else "—",
                    merged_count=merged_count,
                    result_count=len(self.result_dataset),
                )
            else:
                summary = self.tr("{input_count} input → {result_count} merged").format(
                    input_count=input_count if input_count is not None else "—",
                    result_count=len(self.result_dataset),
                )
        elif self.run_outcome == "failed":
            summary = self.tr("Run failed · no output")
        elif self.run_outcome == "canceled":
            summary = self.tr("Run canceled · no output")
        else:
            summary = self.tr(
                "{input} · {enabled}/{total} branch cards enabled · merged output · {filter}"
            ).format(
                input=input_text,
                enabled=enabled,
                total=len(cards),
                filter=filter_text,
            )
        self.summary_label.setText(summary)
        self.summary_label.setVisible(self.window_state != "expand")
        self.branch_hint.setText(
            self.tr("Common input · {count} structures").format(count=input_count)
            if input_count is not None
            else self.tr("Common input · not loaded")
        )
        if self.run_outcome == "succeeded" and self._post_filter_applied:
            self.merge_count_label.setText(
                self.tr("{merged} merged → {kept} kept").format(
                    merged=self._merged_count or 0,
                    kept=len(self.result_dataset),
                )
            )
        elif self.run_outcome == "succeeded":
            self.merge_count_label.setText(
                self.tr("{count} merged structures").format(
                    count=len(self.result_dataset)
                )
            )
        elif self.run_outcome == "running" and self._filter_signal_connected:
            self.merge_count_label.setText(self.tr("Applying post-merge filter"))
        elif self.run_outcome == "running":
            self.merge_count_label.setText(
                self.tr("Running branch {current}/{total}").format(
                    current=min(self.current_index + 1, len(self.cards_to_run)),
                    total=len(self.cards_to_run),
                )
            )
        elif self.run_outcome == "failed":
            self.merge_count_label.setText(self.tr("Run failed · no output"))
        elif self.run_outcome == "canceled":
            self.merge_count_label.setText(self.tr("Run canceled · no output"))
        else:
            self.merge_count_label.setText(
                self.tr("{enabled} enabled branches").format(enabled=enabled)
            )
        if has_filter:
            filter_state = self.tr("enabled") if filter_enabled else self.tr("disabled")
            self.filter_hint.setText(
                self.tr("Post-merge filter · {name} · {state}").format(
                    name=filter_name,
                    state=filter_state,
                )
            )
        else:
            self.filter_hint.setText(
                self.tr("Post-merge filter · Drop one filter card here (optional).")
            )
        self.structureChanged.emit()

    def _set_run_status(self, state: str, detail: str = "") -> None:
        """Keep the header dot and badge aligned with the container outcome."""
        self.status_dot.set_state(state)
        self.status_badge.set_state(state, detail)

    @staticmethod
    def _safe_count(value) -> int:
        try:
            return len(value)
        except TypeError:
            return 0

    def get_summary_text(self) -> str:
        enabled = sum(bool(card.check_state) for card in self.card_list)
        filter_text = (
            self.tr("post-filter enabled")
            if self.filter_card is not None and isValid(self.filter_card)
            else self.tr("no post-filter")
        )
        return self.tr(
            "{enabled}/{total} paths · automatic merge · {filter}"
        ).format(enabled=enabled, total=len(self.card_list), filter=filter_text)

    def get_guidance_text(self) -> str:
        return self.tr(
            "Every enabled child receives the same group input. Child outputs are "
            "concatenated immediately; use Permanent Fork when each path must continue independently."
        )

    def get_inspector_overview_text(self) -> str:
        rows = [
            self.tr("Flow structure"),
            self.tr("Common input → independent transforms → automatic merge"),
            "",
            self.tr("Paths"),
        ]
        rows.extend(
            self.tr("{name} · {state}").format(
                name=card.getTitle(),
                state=self.tr("enabled") if card.check_state else self.tr("disabled"),
            )
            for card in self.card_list
        )
        rows.extend(
            [
                "",
                self.tr("Post-merge filter"),
                self.filter_card.getTitle()
                if self.filter_card is not None and isValid(self.filter_card)
                else self.tr("None"),
            ]
        )
        return "\n".join(rows)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            candidates = list(self.card_list)
            if self.filter_card is not None and isValid(self.filter_card):
                candidates.append(self.filter_card)
            selected = next(
                (
                    card
                    for card in candidates
                    if watched in (card.headerView, card.headerLabel)
                ),
                None,
            )
            if selected is not None:
                self.cardSelected.emit(selected)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        if self._cards or self._grid_columns != 1:
            self._reflow_cards()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Close nested cards before destroying the group widget.
        
        Parameters
        ----------
        event : QCloseEvent
            Close event propagated from Qt.
        """
        for card in self.card_list:
            card.close()
        self.deleteLater()
        super().closeEvent(event)

    def dragEnterEvent(self, event):
        """Accept drag events from compatible card widgets.
        
        Parameters
        ----------
        event : QDragEnterEvent
            Drag event describing the incoming payload.
        """
        widget = event.source()

        if widget == self:
            return
        if (
            isinstance(widget, MakeDataCardWidget)
            and widget.__class__.__name__ not in ("CardGroup", "WorkflowFork")
        ):
            self._set_drop_highlight(True)
            workflow_area = self._workflow_area()
            if workflow_area is not None:
                workflow_area.canvas.set_drop_index(None)
                workflow_area._drag_canvas_point = None
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        widget = event.source()
        if (
            not isinstance(widget, MakeDataCardWidget)
            or widget.__class__.__name__ in ("CardGroup", "WorkflowFork")
        ):
            event.ignore()
            return
        workflow_area = self._workflow_area()
        if workflow_area is not None:
            viewport_point = workflow_area.scroll_area.viewport().mapFromGlobal(
                self.mapToGlobal(event.position().toPoint())
            )
            workflow_area._update_drag_auto_scroll(viewport_point.y())
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_drop_highlight(False)
        super().dragLeaveEvent(event)

    def _set_drop_highlight(self, active: bool) -> None:
        self.branch_hint.setStyleSheet(
            "padding: 0 12px; border-radius: 8px;"
            + (
                "color: #087f81; background: rgba(15,143,145,22);"
                "border: 1px dashed rgba(15,143,145,150);"
                if active
                else "color: #53656c; background: rgba(248,250,251,220);"
                "border: 1px solid rgba(110,130,138,55); font-weight: 600;"
            )
        )

    def _workflow_area(self):
        parent = self.parentWidget()
        while parent is not None and not hasattr(parent, "_update_drag_auto_scroll"):
            parent = parent.parentWidget()
        return parent

    def dropEvent(self, event):
        """Handle dropped cards by inserting them or assigning the filter card.
        
        Parameters
        ----------
        event : QDropEvent
            Drop event containing the dragged widget.
        """
        self._set_drop_highlight(False)
        widget = event.source()
        if widget == self:
            return
        workflow_area = self.parentWidget()
        while workflow_area is not None and not hasattr(
            workflow_area, "move_card_to_group"
        ):
            workflow_area = workflow_area.parentWidget()
        if workflow_area is not None and isinstance(widget, MakeDataCardWidget):
            accepted = bool(workflow_area.move_card_to_group(widget, self))
        elif isinstance(widget, FilterDataCard):
            accepted = self.set_filter_card(widget)
        elif isinstance(widget, (MakeDataCard, CardGroup)):
            accepted = self.add_card(widget)
        else:
            accepted = False
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

    def on_card_finished(self, index):
        """Collect results from the finished card and start the next queued card.
        
        Parameters
        ----------
        index : int
            Index of the card that finished processing.
        """
        card = self.cards_to_run[self.current_index]
        card.runFinishedSignal.disconnect(self.on_card_finished)
        self._active_card = None
        if getattr(card, "run_outcome", "succeeded") != "succeeded":
            self.result_dataset = []
            self.run_outcome = getattr(card, "run_outcome", "failed")
            self.set_output_available(False)
            self._set_run_status(
                "canceled" if self.run_outcome == "canceled" else "failed"
            )
            self._refresh_summary()
            self.runFinishedSignal.emit(self.index)
            return
        self.result_dataset.extend(card.result_dataset)
        self.current_index += 1
        self.run_card_num -= 1

        if self.current_index < len(self.cards_to_run):
            self.start_next_card()
        else:
            self._finish_branches()

    def _finish_branches(self):
        """Record the merged size, then run the optional post-merge filter."""
        self._merged_count = len(self.result_dataset)
        if self.filter_card and isValid(self.filter_card) and self.filter_card.check_state:
            self.filter_card.set_dataset(self.result_dataset)
            self.filter_card.runFinishedSignal.connect(self.on_filter_finished)
            self._filter_signal_connected = True
            self._refresh_summary()
            self.filter_card.run()
            return
        self.run_outcome = "succeeded"
        self.set_output_available(bool(self.result_dataset))
        self._set_run_status(
            "succeeded",
            f"{self._safe_count(self.dataset)}→{len(self.result_dataset)}",
        )
        self._refresh_summary()
        self.runFinishedSignal.emit(self.index)

    def on_filter_finished(self, _index):
        """Finish the group only after its optional post-filter has completed."""
        if self.filter_card is None or not isValid(self.filter_card):
            self.result_dataset = []
            self.run_outcome = "failed"
        else:
            self.filter_card.runFinishedSignal.disconnect(self.on_filter_finished)
            self._filter_signal_connected = False
            self.run_outcome = getattr(self.filter_card, "run_outcome", "succeeded")
            if self.run_outcome == "succeeded":
                self.result_dataset = list(self.filter_card.result_dataset)
                self._post_filter_applied = True
            else:
                self.result_dataset = []
        self.set_output_available(
            self.run_outcome == "succeeded" and bool(self.result_dataset)
        )
        if self.run_outcome == "succeeded":
            self._set_run_status(
                "succeeded",
                f"{self._safe_count(self.dataset)}→{len(self.result_dataset)}",
            )
        else:
            self._set_run_status(
                "canceled" if self.run_outcome == "canceled" else "failed"
            )
        self._refresh_summary()
        self.runFinishedSignal.emit(self.index)

    def stop(self):
        """Stop execution across child cards and the optional filter card.
        """
        active_card = self._active_card
        if active_card is not None and isValid(active_card):
            active_card.runFinishedSignal.disconnect(self.on_card_finished)
        self._active_card = None
        for card in self.card_list:
            card.stop()
        if self.filter_card:
            if self._filter_signal_connected:
                self.filter_card.runFinishedSignal.disconnect(self.on_filter_finished)
                self._filter_signal_connected = False
            self.filter_card.stop()
        self.result_dataset = []
        self._merged_count = None
        self._post_filter_applied = False
        self.run_outcome = "canceled"
        self.set_output_available(False)
        self._set_run_status("canceled")
        self._refresh_summary()

    def run(self):
        """Run independent child branches sequentially from the same input."""
        for card in self.card_list:
            card.set_dataset(self.dataset)
        if self.filter_card and isValid(self.filter_card):
            self.filter_card.set_dataset([])
        self.cards_to_run = [card for card in self.card_list if card.check_state]
        self.run_card_num = len(self.cards_to_run)
        self.current_index = 0
        self.run_outcome = "running"
        self.result_dataset = []
        self._merged_count = None
        self._post_filter_applied = False
        self.set_output_available(False)
        self._set_run_status("running")
        self._refresh_summary()

        if not self.check_state:
            self.result_dataset = self.dataset
            self._merged_count = (
                len(self.result_dataset) if self.result_dataset is not None else 0
            )
            self.run_outcome = "succeeded"
            self.set_output_available(bool(self.result_dataset))
            self._set_run_status("disabled")
            self._refresh_summary()
            self.runFinishedSignal.emit(self.index)
        elif self.run_card_num > 0:
            self.start_next_card()
        else:
            self.run_outcome = "failed"
            self._set_run_status("failed")
            MessageManager.send_error_message(
                self.tr("Branch Merge needs at least one enabled branch.")
            )
            self._refresh_summary()
            self.runFinishedSignal.emit(self.index)

    def start_next_card(self):
        if self.current_index < len(self.cards_to_run):
            card = self.cards_to_run[self.current_index]
            card.set_dataset(self.dataset)
            card.index = self.current_index
            card.runFinishedSignal.connect(self.on_card_finished)
            self._active_card = card
            self._refresh_summary()
            card.run()
        else:
            self._finish_branches()

    def write_result_dataset(self, file,**kwargs):
        if self.filter_card and isValid(self.filter_card) and  self.filter_card.check_state:
            self.filter_card.write_result_dataset(file,**kwargs)
            return

        for index,card in enumerate(self.card_list):
            if index==0:
                if "append" not in kwargs:
                    kwargs["append"] = False
            else:
                kwargs["append"] = True
            if card.check_state:
                card.write_result_dataset(file,**kwargs)

    def export_data(self):
        if self.dataset is not None:
            path = call_path_dialog(
                self,
                self.tr("Choose a file save location"),
                "file",
                f"export_{self.getTitle()}_structure.xyz",
            )
            if not path:
                return
            thread=BackgroundTask(
                self,
                show_tip=True,
                title=self.tr("Exporting data"),
            )
            thread.start_work(self.write_result_dataset, path)
    def to_dict(self):
        data_dict = super().to_dict()

        data_dict["card_list"]=[]

        for card in self.card_list:
            data_dict["card_list"].append(card.to_dict())
        if self.filter_card and isValid(self.filter_card)  :
            data_dict["filter_card"]=self.filter_card.to_dict()
        else:
            data_dict["filter_card"]=None

        return data_dict
    def from_dict(self,data_dict):
        self.state_checkbox.setChecked(data_dict['check_state'])
        for sub_card in data_dict.get("card_list",[]):
            card_name=sub_card["class"]
            card  = CardManager.card_info_dict[card_name](self)
            self.add_card(card, preserve_legacy_branch=True)
            card.from_dict(sub_card)

        if data_dict.get("filter_card"):
            card_name=data_dict["filter_card"]["class"]
            filter_card  = CardManager.card_info_dict[card_name](self)
            filter_card.from_dict(data_dict["filter_card"])
            self.set_filter_card(filter_card)
