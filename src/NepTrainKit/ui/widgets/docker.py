"""Vertical accordion workflow canvas with persistent branch lanes."""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLayout,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon,
    Flyout,
    FlyoutAnimationType,
    ScrollArea,
    ScrollBarHandleDisplayMode,
    StrongBodyLabel,
    TransparentToolButton,
)
from shiboken6 import isValid

from NepTrainKit.core import CardManager
from NepTrainKit.ui.messages import translate_runtime_message

from .card_metadata import (
    contributors_text,
    localized_card_description,
    localized_card_group,
)
from .card_widget import FilterDataCard, MakeDataCardWidget
from .compact_form import adapt_legacy_inspector_form
from .workflow_library import WorkflowLibraryPanel


class CenteredWorkflowLayout(QVBoxLayout):
    """Keep fixed-width workflow nodes on one visual centerline."""

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        for index in range(self.count()):
            widget = self.itemAt(index).widget()
            if widget is not None and not widget.isHidden():
                widget.move(max(0, (rect.width() - widget.width()) // 2), widget.y())


class WorkflowCanvas(QWidget):
    """Paint the quiet vertical trunk behind top-level cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drop_index: int | None = None

    def set_drop_index(self, index: int | None) -> None:
        if self._drop_index == index:
            return
        self._drop_index = index
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        layout = self.layout()
        if layout is None:
            return
        widgets = [
            layout.itemAt(index).widget()
            for index in range(layout.count())
            if layout.itemAt(index).widget() is not None
            and layout.itemAt(index).widget().isVisible()
        ]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        connector_color = QColor(157, 178, 184)
        painter.setPen(QPen(connector_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(connector_color)
        for upper, lower in zip(widgets, widgets[1:]):
            x = upper.geometry().center().x()
            start_y = upper.geometry().bottom() + 2
            end_y = lower.geometry().top() - 2
            if end_y - start_y < 8:
                continue
            arrow_tip = end_y
            painter.drawLine(x, start_y, x, arrow_tip - 5)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4, arrow_tip - 6),
                        QPointF(x + 4, arrow_tip - 6),
                        QPointF(x, arrow_tip),
                    ]
                )
            )
        if self._drop_index is None:
            return
        if not widgets:
            indicator_y = 24
            indicator_width = 300
        elif self._drop_index <= 0:
            indicator_y = widgets[0].geometry().top() - 11
            indicator_width = widgets[0].width()
        elif self._drop_index >= len(widgets):
            indicator_y = widgets[-1].geometry().bottom() + 11
            indicator_width = widgets[-1].width()
        else:
            indicator_y = (
                widgets[self._drop_index - 1].geometry().bottom()
                + widgets[self._drop_index].geometry().top()
            ) // 2
            indicator_width = min(
                widgets[self._drop_index - 1].width(),
                widgets[self._drop_index].width(),
            )
        accent = QColor(15, 143, 145)
        half_width = min(240, max(120, indicator_width // 3))
        center_x = self.width() // 2
        painter.setPen(QPen(accent, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(accent)
        painter.drawLine(center_x - half_width, indicator_y, center_x + half_width, indicator_y)
        painter.drawEllipse(QPointF(center_x - half_width, indicator_y), 3, 3)
        painter.drawEllipse(QPointF(center_x + half_width, indicator_y), 3, 3)


class WorkflowGuidancePanel(QFrame):
    """Inspector hosting card parameters, guidance, and output context."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._card: MakeDataCardWidget | None = None
        self._editor_widget: QWidget | None = None
        self._editor_size_policy: QSizePolicy | None = None
        self._editor_maximum_width: int | None = None
        self._editor_minimum_height: int | None = None
        self._editor_maximum_height: int | None = None
        self._editor_height_animation: QVariantAnimation | None = None
        self._editor_reflow_start_height: int | None = None
        self._editor_reflow_target_height: int | None = None
        self._editor_reflow_height_delta = 0
        self._adaptive_table_row_counts: dict[QWidget, int] = {}
        self._context_signal_connections: list[tuple[object, object]] = []
        self._editor_reflow_timer = QTimer(self)
        self._editor_reflow_timer.setSingleShot(True)
        self._editor_reflow_timer.timeout.connect(self._reflow_editor_height)
        self.setObjectName("workflowGuidancePanel")
        # The workbench assigns this pane a stable width while it is visible.
        # A hard minimum here prevents the main window from ever reaching the
        # breakpoint that hides the pane on narrow screens.
        self.setMinimumWidth(0)
        self.setMaximumWidth(460)
        self.setStyleSheet(
            "QFrame#workflowGuidancePanel {"
            "border: 1px solid rgba(100,120,128,45); border-radius: 10px;"
            "background: rgba(255,255,255,230); }"
        )

        self.eyebrow = CaptionLabel(self.tr("Card inspector"), self)
        self.eyebrow.setStyleSheet("color: #078b8d; font-weight: 600;")
        self.title_label = StrongBodyLabel(self.tr("Select a card"), self)
        self.copy_card_button = TransparentToolButton(FluentIcon.COPY, self)
        self.copy_card_button.setFixedSize(28, 28)
        self.copy_card_button.setIconSize(QSize(14, 14))
        self.copy_card_button.setToolTip(self.tr("Copy card JSON"))
        self.copy_card_button.setAccessibleName(self.tr("Copy card JSON"))
        self.copy_card_button.clicked.connect(self._copy_card_json)
        self.copy_card_button.setEnabled(False)
        self.docs_button = TransparentToolButton(FluentIcon.DOCUMENT, self)
        self.docs_button.setFixedSize(28, 28)
        self.docs_button.setIconSize(QSize(14, 14))
        self.docs_button.setToolTip(self.tr("Open full documentation"))
        self.docs_button.setAccessibleName(self.tr("Open full documentation"))
        self.docs_button.clicked.connect(self._open_docs)
        self.docs_button.setEnabled(False)
        self.info_button = TransparentToolButton(FluentIcon.INFO, self)
        self.info_button.setFixedSize(28, 28)
        self.info_button.setIconSize(QSize(14, 14))
        self.info_button.setToolTip(self.tr("Card information and contributors"))
        self.info_button.setAccessibleName(self.tr("Card information and contributors"))
        self.info_button.clicked.connect(self._show_card_info)
        self.info_button.setEnabled(False)
        self._card_description = ""
        self._about_text = ""
        self._citation = ""
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("QTabWidget::pane { border: 0; }")

        self.parameter_page = QWidget(self.tabs)
        parameter_page_layout = QVBoxLayout(self.parameter_page)
        parameter_page_layout.setContentsMargins(0, 4, 0, 0)
        self.parameter_scroll = ScrollArea(self.parameter_page)
        self.parameter_scroll.scrollDelagate.vScrollBar.setHandleDisplayMode(
            ScrollBarHandleDisplayMode.ALWAYS
        )
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.parameter_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameter_host = QWidget(self.parameter_scroll)
        self.parameter_host.setObjectName("workflowParameterHost")
        self.parameter_host.setMinimumWidth(0)
        self.parameter_host.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.parameter_scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QWidget#workflowParameterHost { background: transparent; }"
        )
        self.parameter_layout = QVBoxLayout(self.parameter_host)
        # Child size hints must not widen the inspector host on platforms whose
        # default fonts are slightly wider. Height still follows the layout's
        # size hint, while width is fitted explicitly to the scroll viewport.
        self.parameter_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.parameter_layout.setContentsMargins(4, 4, 4, 4)
        self.parameter_layout.setSpacing(8)

        self.context_widget = QWidget(self.parameter_host)
        context_layout = QVBoxLayout(self.context_widget)
        context_layout.setContentsMargins(0, 4, 0, 0)
        context_layout.setSpacing(7)

        self.current_context_caption = CaptionLabel(
            self.tr("Current configuration"),
            self.context_widget,
        )
        self.current_context_label = BodyLabel("", self.context_widget)
        self.current_context_label.setWordWrap(True)
        self.current_context_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.current_context_label.setStyleSheet(
            "padding: 8px; border-left: 3px solid #0f8f91;"
            "background: rgba(15,143,145,14);"
        )

        self.recommend_caption = CaptionLabel(self.tr("Recommended checks"), self.context_widget)
        self.recommend_label = BodyLabel("", self.context_widget)
        self.recommend_label.setWordWrap(True)
        self.recommend_label.setStyleSheet(
            "padding: 8px; border-left: 3px solid #d49b26;"
            "background: rgba(212,155,38,14);"
        )

        context_layout.addWidget(self.current_context_caption)
        context_layout.addWidget(self.current_context_label)
        context_layout.addWidget(self.recommend_caption)
        context_layout.addWidget(self.recommend_label)
        self.context_widget.hide()

        self.parameter_placeholder = BodyLabel(
            self.tr("Select a parameter card to edit it here."),
            self.parameter_host,
        )
        self.parameter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parameter_placeholder.setWordWrap(True)
        self.parameter_layout.addWidget(self.context_widget)
        self.parameter_layout.addWidget(self.parameter_placeholder)
        self.parameter_layout.addStretch(1)
        self.parameter_scroll.setWidget(self.parameter_host)
        parameter_page_layout.addWidget(self.parameter_scroll)

        self.tabs.addTab(self.parameter_page, self.tr("Parameters"))
        self.tabs.tabBar().hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(9)
        layout.addWidget(self.eyebrow)
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_layout.addWidget(self.title_label, 1)
        title_layout.addWidget(self.docs_button)
        title_layout.addWidget(self.info_button)
        title_layout.addWidget(self.copy_card_button)
        layout.addLayout(title_layout)
        layout.addWidget(self.tabs, 1)

    def set_card(self, card: MakeDataCardWidget | None) -> None:
        if card is not self._card:
            self._release_editor()
        self._card = card
        if card is None:
            self.title_label.setText(self.tr("Select a card"))
            self.current_context_label.clear()
            self.recommend_label.clear()
            self._card_description = ""
            self._about_text = ""
            self._citation = ""
            self.docs_button.setEnabled(False)
            self.info_button.setEnabled(False)
            self.copy_card_button.setEnabled(False)
            self.context_widget.hide()
            self.parameter_placeholder.setText(
                self.tr("Select a parameter card to edit it here.")
            )
            self.parameter_placeholder.setStyleSheet("")
            self.parameter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.parameter_placeholder.show()
            return

        if self._editor_widget is None:
            self._attach_editor(card)

        metadata = CardManager.get_card_metadata(card.__class__.__name__)
        description = localized_card_description(metadata) if metadata is not None else ""
        description = description or str(getattr(card, "description", "") or "")
        self.title_label.setText(card.getTitle())
        self._card_description = description
        self._set_about_metadata(card, metadata)
        url_getter = getattr(card, "get_online_doc_url", None)
        self.docs_button.setEnabled(bool(url_getter and url_getter()))
        self.info_button.setEnabled(True)
        self.copy_card_button.setEnabled(True)
        self.context_widget.show()
        self._refresh_context()

    def _show_card_info(self) -> None:
        if self._card is None:
            return
        sections = [text for text in (self._card_description, self._about_text) if text]
        if self._citation:
            sections.append(self.tr("Citation: {citation}").format(citation=self._citation))
        Flyout.create(
            title=self.tr("Card information"),
            content="\n\n".join(sections) or self.tr("No additional card information."),
            icon=FluentIcon.INFO,
            isClosable=True,
            target=self.info_button,
            parent=self,
            aniType=FlyoutAnimationType.DROP_DOWN,
        )

    def _refresh_context(self) -> None:
        card = self._card
        if card is None or not isValid(card):
            return
        summary_getter = getattr(card, "get_summary_text", None)
        guidance_getter = getattr(card, "get_guidance_text", None)
        try:
            summary = str(summary_getter() or "").strip() if callable(summary_getter) else ""
            guidance = (
                str(guidance_getter() or "").strip()
                if callable(guidance_getter)
                else ""
            )
        except ValueError as exc:
            self.current_context_label.clear()
            self.current_context_caption.setVisible(False)
            self.current_context_label.setVisible(False)
            self.recommend_caption.setText(self.tr("Parameter issue"))
            self.recommend_label.setText(translate_runtime_message(exc))
            self.recommend_caption.show()
            self.recommend_label.show()
            return

        self.current_context_label.setText(summary)
        self.current_context_caption.setVisible(bool(summary))
        self.current_context_label.setVisible(bool(summary))

        self.recommend_caption.setText(self.tr("Recommended checks"))
        self.recommend_label.setText(guidance)
        self.recommend_caption.setVisible(bool(guidance))
        self.recommend_label.setVisible(bool(guidance))
        if getattr(card, "setting_widget", None) is None:
            self._set_container_overview(card)

    def _set_container_overview(self, card: MakeDataCardWidget) -> None:
        overview_getter = getattr(card, "get_inspector_overview_text", None)
        overview = (
            str(overview_getter() or "").strip()
            if callable(overview_getter)
            else ""
        )
        self.parameter_placeholder.setText(
            overview
            or self.tr("This workflow container is edited directly on the canvas.")
        )
        self.parameter_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.parameter_placeholder.setStyleSheet(
            "padding: 10px; border: 1px solid rgba(110,130,138,45);"
            "border-radius: 8px; background: rgba(248,250,251,190);"
        )

    def _schedule_context_refresh(self, *_args) -> None:
        QTimer.singleShot(0, self._refresh_context)

    def _schedule_editor_reflow(self, row_count: int | None = None) -> None:
        viewport = self.parameter_scroll.viewport()
        if viewport.updatesEnabled():
            viewport.setUpdatesEnabled(False)
        if self._editor_reflow_start_height is None and self._editor_widget is not None:
            self._editor_reflow_start_height = self._editor_widget.height()
        sender = self.sender()
        if isinstance(sender, QWidget) and isinstance(row_count, int):
            previous_count = self._adaptive_table_row_counts.get(sender, row_count)
            table = getattr(sender, "table", None)
            row_height = (
                table.verticalHeader().defaultSectionSize()
                if table is not None
                else 30
            )
            self._editor_reflow_height_delta += (row_count - previous_count) * row_height
            self._adaptive_table_row_counts[sender] = row_count
        self._editor_reflow_timer.start(0)

    def _reflow_editor_height(self) -> None:
        viewport = self.parameter_scroll.viewport()
        editor = self._editor_widget
        try:
            if editor is None or not isValid(editor):
                return
            if self._editor_height_animation is not None:
                self._editor_height_animation.stop()
                self._editor_height_animation.deleteLater()
                self._editor_height_animation = None
            start_height = self._editor_reflow_start_height or editor.height()
            self._editor_reflow_start_height = None
            editor.setMinimumHeight(self._editor_minimum_height or 0)
            editor.setMaximumHeight(self._editor_maximum_height or 16777215)
            editor.adjustSize()
            target_height = (
                start_height + self._editor_reflow_height_delta
                if self._editor_reflow_height_delta
                else editor.sizeHint().height()
            )
            self._editor_reflow_height_delta = 0
            self._editor_reflow_target_height = target_height
            self._fit_editor_width()
            self.parameter_host.updateGeometry()
            if start_height != target_height:
                editor.setFixedHeight(start_height)
                animation = QVariantAnimation(self)
                animation.setStartValue(start_height)
                animation.setEndValue(target_height)
                animation.setDuration(120)
                animation.setEasingCurve(QEasingCurve.Type.OutCubic)
                animation.valueChanged.connect(self._set_animated_editor_height)
                animation.finished.connect(self._finish_editor_height_animation)
                self._editor_height_animation = animation
                # Headless Windows runners occasionally stop advancing a Qt
                # variant animation without emitting ``finished``. Keep the
                # real 120 ms transition, but guarantee that its final layout
                # state is committed even when the platform animation driver
                # stalls.
                watchdog = QTimer(animation)
                watchdog.setSingleShot(True)
                watchdog.timeout.connect(
                    lambda current=animation: self._complete_editor_height_animation(
                        current
                    )
                )
                watchdog.start(animation.duration() + 80)
            else:
                editor.resize(editor.width(), target_height)
                self._editor_reflow_target_height = None
        finally:
            viewport.setUpdatesEnabled(True)
            viewport.update()
        if self._editor_height_animation is not None:
            self._editor_height_animation.start()

    def _set_animated_editor_height(self, value) -> None:
        editor = self._editor_widget
        if editor is not None and isValid(editor):
            editor.setFixedHeight(int(round(float(value))))

    def _complete_editor_height_animation(self, animation: QVariantAnimation) -> None:
        """Commit a stalled platform animation without affecting newer reflows."""
        if self._editor_height_animation is not animation:
            return
        animation.setCurrentTime(animation.duration())
        if self._editor_height_animation is animation:
            self._finish_editor_height_animation()

    def _finish_editor_height_animation(self) -> None:
        editor = self._editor_widget
        animation = self._editor_height_animation
        target_height = self._editor_reflow_target_height
        self._editor_height_animation = None
        self._editor_reflow_target_height = None
        if editor is not None and isValid(editor):
            editor.setMinimumHeight(self._editor_minimum_height or 0)
            editor.setMaximumHeight(self._editor_maximum_height or 16777215)
            if target_height is not None:
                editor.resize(editor.width(), target_height)
            self._fit_editor_width()
        if animation is not None:
            animation.deleteLater()

    def _connect_context_signals(self, editor: QWidget) -> None:
        self._disconnect_context_signals()
        presentation_signal = getattr(self._card, "presentationChanged", None)
        if presentation_signal is not None and hasattr(presentation_signal, "connect"):
            presentation_signal.connect(self._schedule_context_refresh)
            self._context_signal_connections.append((presentation_signal, self._schedule_context_refresh))
        signal_names = ("valueChanged", "currentIndexChanged", "textChanged", "toggled", "stateChanged")
        for widget in [editor, *editor.findChildren(QWidget)]:
            row_count_signal = getattr(widget, "rowCountChanged", None)
            if row_count_signal is not None and hasattr(row_count_signal, "connect"):
                table = getattr(widget, "table", None)
                if table is not None:
                    self._adaptive_table_row_counts[widget] = table.rowCount()
                row_count_signal.connect(self._schedule_editor_reflow)
                self._context_signal_connections.append(
                    (row_count_signal, self._schedule_editor_reflow)
                )
            for name in signal_names:
                signal = getattr(widget, name, None)
                if signal is None or not hasattr(signal, "connect"):
                    continue
                try:
                    signal.connect(self._schedule_context_refresh)
                except (RuntimeError, TypeError):
                    continue
                self._context_signal_connections.append((signal, self._schedule_context_refresh))
                break

    def _disconnect_context_signals(self) -> None:
        for signal, slot in self._context_signal_connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._context_signal_connections.clear()
        self._adaptive_table_row_counts.clear()

    def _copy_card_json(self) -> None:
        """Copy the selected card without duplicating actions in its header."""
        if self._card is not None:
            self._card.copy_json_to_clipboard()

    def _set_about_metadata(self, card, metadata) -> None:
        """Render compact provenance in the inspector instead of a modal."""
        if metadata is None:
            card_type = str(getattr(card, "group", "") or self.tr("Not specified"))
            contributors = self.tr("Not specified")
            version = str(getattr(card, "card_version", "") or "")
            license_name = str(getattr(card, "license", "") or "")
            citation = str(getattr(card, "citation", "") or "")
        else:
            card_type = localized_card_group(metadata) or self.tr("Not specified")
            contributors = contributors_text(metadata)
            version = metadata.version
            license_name = metadata.license
            citation = metadata.citation

        lines = [
            self.tr("Type: {type}").format(type=card_type),
            self.tr("Contributors: {contributors}").format(
                contributors=contributors
            ),
        ]
        if version:
            lines.append(self.tr("Version: {version}").format(version=version))
        if license_name:
            lines.append(self.tr("License: {license}").format(license=license_name))
        self._about_text = "\n".join(lines)
        self._citation = citation

    def _attach_editor(self, card: MakeDataCardWidget) -> None:
        editor = getattr(card, "setting_widget", None)
        if editor is None:
            self._set_container_overview(card)
            self.parameter_placeholder.show()
            self.context_widget.show()
            return
        adapt_legacy_inspector_form(editor, getattr(card, "settingLayout", None))
        card.viewLayout.removeWidget(editor)
        editor.setParent(self.parameter_host)
        self._editor_size_policy = QSizePolicy(editor.sizePolicy())
        self._editor_maximum_width = editor.maximumWidth()
        self._editor_minimum_height = editor.minimumHeight()
        self._editor_maximum_height = editor.maximumHeight()
        editor.setMinimumWidth(0)
        editor.setMaximumWidth(16777215)
        editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.parameter_layout.insertWidget(
            0,
            editor,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        editor.setProperty("workflowInspectorHosted", True)
        editor.show()
        card.refresh_compact_presentation()
        self.parameter_placeholder.hide()
        self.parameter_placeholder.setStyleSheet("")
        self.parameter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor_widget = editor
        self._connect_context_signals(editor)
        QTimer.singleShot(0, self._fit_editor_width)

    def _fit_editor_width(self) -> None:
        """Keep editor size hints from widening the scroll area's viewport."""
        editor = self._editor_widget
        if editor is None or not isValid(editor):
            return
        viewport_width = max(0, self.parameter_scroll.viewport().width())
        self.parameter_host.setMaximumWidth(viewport_width or 16777215)
        if viewport_width:
            self.parameter_host.resize(viewport_width, self.parameter_host.height())
        margins = self.parameter_layout.contentsMargins()
        available = max(
            0,
            viewport_width - margins.left() - margins.right(),
        )
        editor.setMaximumWidth(available or 16777215)
        if available:
            editor.resize(available, editor.height())
            editor.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_editor_width)

    def _release_editor(self) -> None:
        if self._editor_reflow_timer.isActive():
            self._editor_reflow_timer.stop()
            self.parameter_scroll.viewport().setUpdatesEnabled(True)
        self._editor_reflow_start_height = None
        self._editor_reflow_target_height = None
        self._editor_reflow_height_delta = 0
        if self._editor_height_animation is not None:
            self._editor_height_animation.stop()
            self._editor_height_animation.deleteLater()
            self._editor_height_animation = None
        editor = self._editor_widget
        card = self._card
        self._editor_widget = None
        if editor is None:
            return
        self._disconnect_context_signals()
        self.parameter_layout.removeWidget(editor)
        editor.setProperty("workflowInspectorHosted", False)
        if self._editor_size_policy is not None:
            editor.setSizePolicy(self._editor_size_policy)
        self._editor_size_policy = None
        if self._editor_maximum_width is not None:
            editor.setMaximumWidth(self._editor_maximum_width)
        self._editor_maximum_width = None
        if self._editor_minimum_height is not None:
            editor.setMinimumHeight(self._editor_minimum_height)
        self._editor_minimum_height = None
        if self._editor_maximum_height is not None:
            editor.setMaximumHeight(self._editor_maximum_height)
        self._editor_maximum_height = None
        if card is not None and isValid(card):
            editor.setParent(card.view)
            card.viewLayout.insertWidget(0, editor)
            editor.setVisible(card.window_state == "expand")
            card.refresh_compact_presentation()
        else:
            editor.setParent(None)
        self.parameter_placeholder.show()

    def refresh(self) -> None:
        if self._card is not None and isValid(self._card):
            self.set_card(self._card)

    def _open_docs(self) -> None:
        if self._card is None:
            return
        url = self._card.get_online_doc_url()
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def closeEvent(self, event) -> None:
        self._release_editor()
        super().closeEvent(event)


class MakeWorkflowArea(QWidget):
    """Single-canvas vertical accordion with optional permanent fork nodes."""

    workflowChanged = Signal()

    _CARD_MAX_WIDTH = 520
    _FORK_MAX_WIDTH = 920
    _CARD_MIN_WIDTH = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._cards: list[MakeDataCardWidget] = []
        self._selected_card: MakeDataCardWidget | None = None
        self._accordion_updating = False
        self._dragged_root: MakeDataCardWidget | None = None
        self._drag_placeholder: QFrame | None = None
        self._drag_slot_index: int | None = None
        self._drag_scroll_speed = 0
        self._drag_canvas_point: QPoint | None = None
        self._explicit_branch_context = None
        self.setObjectName("MakeWorkflowArea")
        self.setAcceptDrops(True)
        self.setStyleSheet(
            'QWidget#headerView[workflowSelected="true"] {'
            "background: rgba(15,143,145,16);"
            "border-left: 3px solid #0f8f91;"
            "}"
        )

        self._drag_scroll_timer = QTimer(self)
        self._drag_scroll_timer.setInterval(24)
        self._drag_scroll_timer.timeout.connect(self._auto_scroll_drag)
        self._width_update_timer = QTimer(self)
        self._width_update_timer.setSingleShot(True)
        self._width_update_timer.timeout.connect(self._update_card_widths)

        self.canvas = WorkflowCanvas(self)
        self.canvas_layout = CenteredWorkflowLayout(self.canvas)
        self.canvas_layout.setContentsMargins(36, 12, 36, 28)
        self.canvas_layout.setSpacing(22)
        self.empty_label = CaptionLabel(
            self.tr("Add a card to start building the workflow."), self.canvas
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.canvas_layout.addWidget(self.empty_label)
        self.canvas_layout.addStretch(1)

        self.scroll_area = ScrollArea(self)
        self.scroll_area.scrollDelagate.vScrollBar.setHandleDisplayMode(
            ScrollBarHandleDisplayMode.ON_HOVER
        )
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.viewport().installEventFilter(self)

        self.guidance_panel = WorkflowGuidancePanel(self)
        self.library_panel = WorkflowLibraryPanel(self)
        for panel in (self.library_panel, self.guidance_panel):
            panel.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Expanding,
            )
        self.library_panel.setMinimumWidth(220)
        self.guidance_panel.setMinimumWidth(380)
        self.canvas_column = QWidget(self)
        self.canvas_column.setMinimumWidth(0)
        self.canvas_column.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.canvas_column_layout = QVBoxLayout(self.canvas_column)
        self.canvas_column_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_column_layout.setSpacing(8)
        self.canvas_column_layout.addWidget(self.scroll_area, 1)
        self._command_bar: QWidget | None = None
        self._status_bar: QWidget | None = None
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(3)
        self.splitter.addWidget(self.library_panel)
        self.splitter.addWidget(self.canvas_column)
        self.splitter.addWidget(self.guidance_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.handle(1).setEnabled(False)
        self.splitter.handle(2).setEnabled(True)
        self.splitter.handle(2).setCursor(Qt.CursorShape.SplitHCursor)
        self.splitter.setSizes([220, 800, 330])

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self.splitter, 1)

    def _apply_responsive_splitter_sizes(self) -> None:
        """Reset pane proportions after a responsive visibility transition."""
        width = max(0, self.splitter.width() - self.splitter.handleWidth() * 2)
        spacious = width >= 1320
        library_width = 250 if spacious else 220
        guidance_width = 410 if spacious else 380
        canvas_width = max(0, width - library_width - guidance_width)
        self.splitter.setSizes([library_width, canvas_width, guidance_width])
        self._update_card_widths()
        self._width_update_timer.start(0)

    def set_command_bar(self, widget: QWidget) -> None:
        """Place the workflow command bar inside the center workbench column."""
        if self._command_bar is widget:
            return
        if self._command_bar is not None:
            self.canvas_column_layout.removeWidget(self._command_bar)
            self._command_bar.setParent(None)
        self._command_bar = widget
        widget.setParent(self.canvas_column)
        self.canvas_column_layout.insertWidget(0, widget)

    def set_status_bar(self, widget: QWidget) -> None:
        """Place shared dataset and runtime context across the full workbench."""
        if self._status_bar is widget:
            return
        if self._status_bar is not None:
            self.main_layout.removeWidget(self._status_bar)
            self._status_bar.setParent(None)
        self._status_bar = widget
        widget.setParent(self)
        self.main_layout.addWidget(widget)

    @property
    def cards(self) -> list[MakeDataCardWidget]:
        return list(self._cards)

    @staticmethod
    def _is_card_group(card) -> bool:
        return card is not None and card.__class__.__name__ == "CardGroup"

    @staticmethod
    def _is_fork(card) -> bool:
        return card is not None and card.__class__.__name__ == "WorkflowFork"

    def _is_structural(self, card) -> bool:
        return self._is_fork(card) or self._is_card_group(card)

    def _configure_node_presentation(self, card: MakeDataCardWidget) -> None:
        """Keep operation cards compact; structural containers expand in place."""
        structural = self._is_structural(card)
        card.collapse_button.setVisible(structural)
        if not structural and card.window_state != "collapse":
            card.window_state = "collapse"
            card.windowStateChangedSignal.emit()

    def _update_selection_style(self, selected: MakeDataCardWidget) -> None:
        for card in self.all_cards():
            header = card.headerView
            is_selected = card is selected
            if bool(header.property("workflowSelected")) == is_selected:
                continue
            header.setProperty("workflowSelected", is_selected)
            card.set_workflow_selected(is_selected)
            header.style().unpolish(header)
            header.style().polish(header)
            header.update()

    def _iter_nested_cards(self, root: MakeDataCardWidget):
        yield root
        for child in getattr(root, "card_list", []):
            yield from self._iter_nested_cards(child)
        for branch in getattr(root, "branches", []):
            for child in getattr(branch, "cards", []):
                yield from self._iter_nested_cards(child)
        filter_card = getattr(root, "filter_card", None)
        if filter_card is not None:
            yield from self._iter_nested_cards(filter_card)

    def all_cards(self) -> list[MakeDataCardWidget]:
        return [item for root in self._cards for item in self._iter_nested_cards(root)]

    def add_card(self, card: MakeDataCardWidget) -> None:
        group = self._selected_group_context()
        if group is not None and not self._is_structural(card):
            if group.add_card(card):
                for nested in self._iter_nested_cards(card):
                    self._configure_node_presentation(nested)
                    self._connect_guidance_refresh(nested)
                    self._connect_close_cleanup(nested)
                    self._connect_drag_lifecycle(nested)
                self.select_card(card, expand=True, ensure_visible=False)
                self.workflowChanged.emit()
            return
        branch_context = self._selected_branch_context()
        if branch_context is not None and not self._is_fork(card):
            fork, branch, index = branch_context
            if fork.add_card(card, branch, index):
                for nested in self._iter_nested_cards(card):
                    self._configure_node_presentation(nested)
                    self._connect_guidance_refresh(nested)
                    self._connect_close_cleanup(nested)
                    self._connect_drag_lifecycle(nested)
                self.select_card(card, expand=True, ensure_visible=False)
                self.workflowChanged.emit()
                return
        index = len(self._cards)
        if self._selected_card is not None:
            for root_index, root in enumerate(self._cards):
                if self._selected_card in list(self._iter_nested_cards(root)):
                    index = root_index + 1
                    break
        self._insert_card(index, card)
        self.select_card(card, expand=True, ensure_visible=False)
        self.workflowChanged.emit()

    def add_root_card(self, card: MakeDataCardWidget) -> None:
        """Append a card at workflow root, ignoring interactive selection context."""
        self._insert_card(len(self._cards), card)
        self.select_card(card, expand=True, ensure_visible=False)
        self.workflowChanged.emit()

    def _selected_branch_context(self):
        explicit = self._explicit_branch_context
        if explicit is not None:
            fork, branch = explicit
            if fork in self._cards and branch in getattr(fork, "branches", []):
                return fork, branch, len(branch.cards)
            self._explicit_branch_context = None
        selected = self._selected_card
        if selected is None:
            return None
        for root in self._cards:
            if not self._is_fork(root):
                continue
            for branch in root.branches:
                for index, child in enumerate(branch.cards):
                    if selected in list(self._iter_nested_cards(child)):
                        return root, branch, index + 1
        return None

    def _selected_group_context(self):
        selected = self._selected_card
        if selected is None:
            return None
        for candidate in self.all_cards():
            if not self._is_card_group(candidate):
                continue
            if selected is candidate or selected in getattr(candidate, "card_list", []):
                return candidate
        return None

    def _insert_card(self, index: int, card: MakeDataCardWidget) -> None:
        if card in self._cards:
            return
        index = max(0, min(index, len(self._cards)))
        self._cards.insert(index, card)
        card.setParent(self.canvas)
        card.set_compact_header(False)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.canvas_layout.insertWidget(
            index,
            card,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )
        self.empty_label.hide()
        card.windowStateChangedSignal.connect(
            lambda selected=card: self._on_window_state_changed(selected)
        )
        for nested in self._iter_nested_cards(card):
            self._configure_node_presentation(nested)
            self._connect_guidance_refresh(nested)
            self._connect_close_cleanup(nested)
            self._connect_drag_lifecycle(nested)
        card.headerView.installEventFilter(self)
        card.headerLabel.installEventFilter(self)
        if self._is_fork(card):
            card.cardDropRequested.connect(self._move_card_to_branch)
            card.cardSelected.connect(lambda selected: self.select_card(selected))
            card.branchSelected.connect(
                lambda branch, fork=card: self._activate_branch(fork, branch)
            )
        if self._is_card_group(card):
            card.cardSelected.connect(
                lambda selected: self.select_card(selected, expand=True)
            )
        structure_changed = getattr(card, "structureChanged", None)
        if structure_changed is not None:
            structure_changed.connect(
                lambda selected=card: self._refresh_guidance_for(selected)
            )
        card.show()
        self._width_update_timer.start(0)
        self.canvas.update()

    def _connect_drag_lifecycle(self, card: MakeDataCardWidget) -> None:
        if bool(card.property("workflowDragLifecycleConnected")):
            return
        card.dragStartedSignal.connect(self._on_drag_started)
        card.dragFinishedSignal.connect(self._on_drag_finished)
        card.setProperty("workflowDragLifecycleConnected", True)

    def _on_drag_started(self, card: MakeDataCardWidget) -> None:
        if card not in self._cards or self._dragged_root is not None:
            return
        self._dragged_root = card
        index = self._cards.index(card)
        self._drag_slot_index = index
        placeholder = QFrame(self.canvas)
        placeholder.setObjectName("workflowDragPlaceholder")
        placeholder.setFixedSize(card.width(), min(88, max(60, card.headerView.height() + 18)))
        placeholder.setStyleSheet(
            "QFrame#workflowDragPlaceholder {"
            "border: 2px dashed rgba(15,143,145,150); border-radius: 9px;"
            "background: rgba(15,143,145,12); }"
        )
        placeholder_label = CaptionLabel(self.tr("Move card here"), placeholder)
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("color: #087f81; font-weight: 600;")
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_layout.setContentsMargins(8, 8, 8, 8)
        placeholder_layout.addWidget(placeholder_label)
        self.canvas_layout.removeWidget(card)
        card.hide()
        self.canvas_layout.insertWidget(
            index,
            placeholder,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )
        placeholder.show()
        self._drag_placeholder = placeholder
        self.canvas.set_drop_index(None)
        self.canvas.updateGeometry()

    def _on_drag_finished(self, card: MakeDataCardWidget, _moved: bool) -> None:
        self._stop_drag_auto_scroll()
        if self._dragged_root is not card:
            return
        self._restore_dragged_root(card)

    def _restore_dragged_root(self, card: MakeDataCardWidget) -> None:
        placeholder = self._drag_placeholder
        if placeholder is not None:
            self.canvas_layout.removeWidget(placeholder)
            placeholder.deleteLater()
        if card in self._cards and self.canvas_layout.indexOf(card) < 0:
            index = self._cards.index(card)
            self.canvas_layout.insertWidget(
                index,
                card,
                0,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            )
        card.show()
        self._dragged_root = None
        self._drag_placeholder = None
        self._drag_slot_index = None
        self.canvas.set_drop_index(None)
        self.canvas.update()

    def _preview_top_level_slot(self, index: int) -> None:
        placeholder = self._drag_placeholder
        if placeholder is None:
            self.canvas.set_drop_index(index)
            return
        candidates = [card for card in self._cards if card is not self._dragged_root]
        index = max(0, min(index, len(candidates)))
        self._drag_slot_index = index
        self.canvas_layout.removeWidget(placeholder)
        self.canvas_layout.insertWidget(
            index,
            placeholder,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )
        self.canvas.set_drop_index(None)

    def _update_drag_auto_scroll(self, y: int) -> None:
        margin = 64
        height = self.scroll_area.viewport().height()
        if y < margin:
            self._drag_scroll_speed = -max(5, (margin - y) // 3)
        elif y > height - margin:
            self._drag_scroll_speed = max(5, (y - (height - margin)) // 3)
        else:
            self._drag_scroll_speed = 0
        if self._drag_scroll_speed:
            if not self._drag_scroll_timer.isActive():
                self._drag_scroll_timer.start()
        else:
            self._drag_scroll_timer.stop()

    def _auto_scroll_drag(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        before = bar.value()
        bar.setValue(before + self._drag_scroll_speed)
        if bar.value() != before and self._drag_canvas_point is not None:
            self._preview_top_level_slot(
                self._top_level_drop_index(self._drag_canvas_point)
            )

    def _stop_drag_auto_scroll(self) -> None:
        self._drag_scroll_speed = 0
        self._drag_scroll_timer.stop()
        self._drag_canvas_point = None

    def _forget_closed_card(self, card: MakeDataCardWidget) -> None:
        """Remove a user-closed card from its current workflow owner immediately."""
        if card.isVisible():
            return
        if card in self._cards:
            descendants = list(self._iter_nested_cards(card))
            self._cards.remove(card)
            self.canvas_layout.removeWidget(card)
        else:
            descendants = [card]
            self._detach_card(card)
        if self._selected_card in descendants or self._selected_card is card:
            self._selected_card = None
            self.guidance_panel.set_card(None)
        self.empty_label.setVisible(not self._cards)
        self.canvas.set_drop_index(None)
        self.canvas.update()
        self.workflowChanged.emit()

    def _connect_close_cleanup(self, card: MakeDataCardWidget) -> None:
        if bool(card.property("workflowCloseCleanupConnected")):
            return
        card.close_button.clicked.connect(
            lambda _checked=False, closed=card: self._forget_closed_card(closed)
        )
        card.setProperty("workflowCloseCleanupConnected", True)

    def _update_card_widths(self) -> None:
        viewport_width = self.scroll_area.viewport().width()
        if viewport_width >= 360:
            side_margin = 36
        elif viewport_width >= 240:
            side_margin = 20
        else:
            side_margin = 8
        margins = self.canvas_layout.contentsMargins()
        self.canvas_layout.setContentsMargins(
            side_margin,
            margins.top(),
            side_margin,
            margins.bottom(),
        )
        available_width = max(
            self._CARD_MIN_WIDTH,
            viewport_width - side_margin * 2,
        )
        for card in self._cards:
            preferred_width = (
                self._FORK_MAX_WIDTH
                if self._is_structural(card)
                else self._CARD_MAX_WIDTH
            )
            card.setFixedWidth(min(preferred_width, available_width))
        self.canvas_layout.invalidate()
        self.canvas_layout.activate()

    def _connect_guidance_refresh(self, card: MakeDataCardWidget) -> None:
        if bool(card.property("workflowGuidanceConnected")):
            return
        card.runFinishedSignal.connect(
            lambda _index, selected=card: self._refresh_guidance_for(selected)
        )
        card.setProperty("workflowGuidanceConnected", True)

    def _refresh_guidance_for(self, card: MakeDataCardWidget) -> None:
        if card is self._selected_card:
            self.guidance_panel.refresh()

    def _on_window_state_changed(self, card: MakeDataCardWidget) -> None:
        if self._accordion_updating:
            return
        self.select_card(card, expand=card.window_state == "expand")

    def select_card(
        self,
        card: MakeDataCardWidget,
        *,
        expand: bool = False,
        ensure_visible: bool = True,
    ) -> None:
        if card not in self.all_cards():
            return
        self._explicit_branch_context = None
        for root in self._cards:
            if self._is_fork(root):
                for branch in root.branches:
                    branch.set_selected(False)
        self._configure_node_presentation(card)
        self._connect_guidance_refresh(card)
        self._selected_card = card
        self._update_selection_style(card)
        if not expand:
            self.guidance_panel.set_card(card)
            return
        expanded_path = {
            candidate
            for candidate in self.all_cards()
            if self._is_structural(candidate)
            and card in list(self._iter_nested_cards(candidate))
        }
        self._accordion_updating = True
        try:
            for root in self._cards:
                descendants = list(self._iter_nested_cards(root))
                for candidate in descendants:
                    target_state = (
                        "expand"
                        if candidate in expanded_path
                        else "collapse"
                    )
                    if candidate.window_state != target_state:
                        candidate.window_state = target_state
                        candidate.windowStateChangedSignal.emit()
        finally:
            self._accordion_updating = False
        self.guidance_panel.set_card(card)
        if ensure_visible:
            self.scroll_area.ensureWidgetVisible(card, 18, 18)

    def _activate_branch(self, fork, branch) -> None:
        if fork not in self._cards or branch not in getattr(fork, "branches", []):
            return
        self.select_card(fork, expand=True, ensure_visible=False)
        self._explicit_branch_context = (fork, branch)
        for candidate in fork.branches:
            candidate.set_selected(candidate is branch)
        self.guidance_panel.refresh()

    def move_card(self, card: MakeDataCardWidget, target_index: int) -> None:
        if card not in self._cards:
            return
        old_index = self._cards.index(card)
        target_index = max(0, min(target_index, len(self._cards) - 1))
        if old_index == target_index:
            return
        self._cards.pop(old_index)
        self._cards.insert(target_index, card)
        self.canvas_layout.removeWidget(card)
        self.canvas_layout.insertWidget(
            target_index,
            card,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )
        self.canvas.update()
        self.workflowChanged.emit()

    def remove_card(self, card: MakeDataCardWidget, *, close: bool = True) -> None:
        if card not in self._cards:
            return
        self._cards.remove(card)
        self.canvas_layout.removeWidget(card)
        if self._selected_card in list(self._iter_nested_cards(card)):
            self._selected_card = None
            self.guidance_panel.set_card(None)
        if close:
            card.close()
        else:
            card.setParent(None)
        self.empty_label.setVisible(not self._cards)
        self.canvas.update()
        self.workflowChanged.emit()

    def clear_cards(self) -> None:
        for card in list(self._cards):
            self.remove_card(card)
        self._selected_card = None
        self.guidance_panel.set_card(None)

    def _detach_card(self, card: MakeDataCardWidget) -> bool:
        if card in self._cards:
            self.remove_card(card, close=False)
            return True
        for root in self._cards:
            if self._detach_from_container(root, card):
                return True
        return False

    def _detach_from_container(self, root, card: MakeDataCardWidget) -> bool:
        if self._is_card_group(root):
            if getattr(root, "filter_card", None) is card:
                root.filter_layout.removeWidget(card)
                root.filter_card = None
                root._refresh_summary()
                return True
            if card in getattr(root, "card_list", []):
                root.remove_card(card)
                return True
            return any(
                self._detach_from_container(child, card)
                for child in list(getattr(root, "card_list", []))
            )
        if self._is_fork(root):
            if root.remove_card(card):
                return True
            return any(
                self._detach_from_container(child, card)
                for branch in root.branches
                for child in list(branch.cards)
            )
        return False

    def move_card_to_group(self, card: MakeDataCardWidget, group: MakeDataCardWidget) -> bool:
        if (
            card is group
            or not self._is_card_group(group)
            or self._is_fork(card)
            or self._is_card_group(card)
        ):
            return False
        if (
            isinstance(card, FilterDataCard)
            and getattr(group, "filter_card", None) is not None
            and group.filter_card is not card
        ):
            return False
        self._detach_card(card)
        accepted = bool(group.add_card(card))
        if accepted:
            self.select_card(card, expand=True)
            self.workflowChanged.emit()
        return accepted

    def _move_card_to_branch(self, card, branch, index: int) -> None:
        fork = next(
            (root for root in self._cards if self._is_fork(root) and branch in root.branches),
            None,
        )
        if fork is None or self._is_fork(card):
            return
        if card in branch.cards and getattr(branch, "_dragged_card", None) is not card:
            old_index = branch.cards.index(card)
            if index > old_index:
                index -= 1
        self._detach_card(card)
        fork.add_card(card, branch, index)
        self.select_card(card, expand=False)
        self.workflowChanged.emit()

    def _top_level_drop_index(self, point: QPoint) -> int:
        canvas_point = self.canvas.mapFrom(self.scroll_area.viewport(), point)
        cards = [card for card in self._cards if card is not self._dragged_root]
        if not cards:
            return 0
        slots = [cards[0].geometry().top() - self.canvas_layout.spacing() // 2]
        slots.extend(
            (upper.geometry().bottom() + lower.geometry().top()) // 2
            for upper, lower in zip(cards, cards[1:])
        )
        slots.append(cards[-1].geometry().bottom() + self.canvas_layout.spacing() // 2)
        distances = [abs(canvas_point.y() - slot_y) for slot_y in slots]
        candidate = min(range(len(slots)), key=distances.__getitem__)
        current = self._drag_slot_index
        if (
            current is not None
            and 0 <= current < len(slots)
            and current != candidate
            and distances[current] <= distances[candidate] + 14
        ):
            return current
        return candidate

    def _handle_top_level_drop(self, event, point: QPoint) -> None:
        self.canvas.set_drop_index(None)
        self._stop_drag_auto_scroll()
        card = event.source()
        if not isinstance(card, MakeDataCardWidget):
            event.ignore()
            return
        index = self._top_level_drop_index(point)
        if card in self._cards:
            if card is self._dragged_root:
                self._cards.remove(card)
                self._cards.insert(max(0, min(index, len(self._cards))), card)
                self._restore_dragged_root(card)
            else:
                old_index = self._cards.index(card)
                if index > old_index:
                    index -= 1
                self.move_card(card, min(index, len(self._cards) - 1))
        elif self._detach_card(card):
            self._insert_card(index, card)
            self.select_card(card, expand=False)
        else:
            event.ignore()
            return
        event.acceptProposedAction()

    def eventFilter(self, watched, event):
        if watched is self.scroll_area.viewport():
            if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                if isinstance(event.source(), MakeDataCardWidget):
                    point = event.position().toPoint()
                    self._drag_canvas_point = point
                    index = self._top_level_drop_index(point)
                    self._preview_top_level_slot(index)
                    self._update_drag_auto_scroll(point.y())
                    event.acceptProposedAction()
                    return True
            if event.type() == QEvent.Type.DragLeave:
                self.canvas.set_drop_index(None)
                self._drag_canvas_point = None
                self._stop_drag_auto_scroll()
                return True
            if event.type() == QEvent.Type.Drop:
                self._handle_top_level_drop(event, event.position().toPoint())
                return True
        if event.type() == QEvent.Type.MouseButtonPress:
            card = next(
                (
                    candidate
                    for candidate in self.all_cards()
                    if watched in (candidate.headerView, candidate.headerLabel)
                ),
                None,
            )
            if card is not None:
                if self._is_structural(card):
                    card.collapse()
                    return True
                else:
                    self.select_card(card)
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if isinstance(event.source(), MakeDataCardWidget):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if isinstance(event.source(), MakeDataCardWidget):
            point = self.scroll_area.viewport().mapFrom(
                self,
                event.position().toPoint(),
            )
            self._drag_canvas_point = point
            self._preview_top_level_slot(self._top_level_drop_index(point))
            self._update_drag_auto_scroll(point.y())
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.canvas.set_drop_index(None)
        self._drag_canvas_point = None
        self._stop_drag_auto_scroll()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        point = self.scroll_area.viewport().mapFrom(self, event.position().toPoint())
        self._handle_top_level_drop(event, point)

    def unmerged_fork_before(self, index: int) -> MakeDataCardWidget | None:
        for card in self._cards[:index]:
            if self._is_fork(card) and not bool(getattr(card, "merge_enabled", False)):
                return card
        return None

    def resizeEvent(self, event) -> None:
        self.guidance_panel.show()
        self.library_panel.show()
        super().resizeEvent(event)
        self._width_update_timer.start(0)
        self._apply_responsive_splitter_sizes()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MakeWorkflowArea()
    window.resize(1280, 760)
    window.show()
    sys.exit(app.exec())
