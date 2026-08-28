"""Card widgets supporting drag-and-drop workflows and dataset processing."""

import inspect
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from ase.io import write as ase_write
from PySide6.QtCore import Property, QCoreApplication, QEvent, QMimeData, QPoint, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDrag, QFont, QIcon
from PySide6.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    FluentIcon,
    FluentStyleSheet,
    ToolTipFilter,
    ToolTipPosition,
    TransparentToolButton,
    setFont,
)
from qfluentwidgets.components.widgets.card_widget import CardSeparator, SimpleCardWidget

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.card_manager import build_card_metadata
from NepTrainKit.core.cards.operation import DatasetOperation, GeneratorOperation, StructureOperation
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask, DataProcessingThread, FilterProcessingThread
from NepTrainKit.version import DOCS_BASE_URL

from .card_metadata import CardMetadataDialog, localized_card_description
from .compact_form import CategoryTag, StatusBadge, StatusDot
from .label import ProcessLabel


class HeaderCardWidget(SimpleCardWidget):
    """Card widget with a header and content area separated by a divider."""

    def __init__(self, parent=None):
        """Initialize header and body layouts.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(parent)
        self.headerView = QWidget(self)
        self.headerLabel = QLabel(self)
        self.headerLabel.setMinimumWidth(0)
        self.headerLabel.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.separator = CardSeparator(self)
        self.view = QWidget(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.headerViewLayout = QVBoxLayout(self.headerView)
        self.headerTopView = QWidget(self.headerView)
        self.headerInfoView = QWidget(self.headerView)
        self.headerLayout = QHBoxLayout(self.headerTopView)
        self.headerInfoLayout = QHBoxLayout(self.headerInfoView)
        self.viewLayout = QHBoxLayout(self.view)

        self.headerLayout.addWidget(self.headerLabel)
        self.headerLayout.setContentsMargins(24, 0, 16, 0)
        self.headerInfoLayout.setContentsMargins(10, 0, 3, 3)
        self.headerInfoLayout.setSpacing(4)
        self.headerViewLayout.setContentsMargins(0, 0, 0, 0)
        self.headerViewLayout.setSpacing(0)
        self.headerViewLayout.addWidget(self.headerTopView)
        self.headerViewLayout.addWidget(self.headerInfoView)
        self.headerTopView.setFixedHeight(48)
        self.headerInfoView.setFixedHeight(34)
        self.headerInfoView.hide()
        self.headerView.setFixedHeight(48)

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.headerView)
        self.vBoxLayout.addWidget(self.separator)
        self.vBoxLayout.addWidget(self.view)

        self.viewLayout.setContentsMargins(24, 24, 24, 24)
        setFont(self.headerLabel, 15, QFont.Weight.DemiBold)

        self.view.setObjectName("view")
        self.headerView.setObjectName("headerView")
        self.headerTopView.setObjectName("headerTopView")
        self.headerInfoView.setObjectName("headerInfoView")
        self.headerLabel.setObjectName("headerLabel")
        FluentStyleSheet.CARD_WIDGET.apply(self)

        self._postInit()

    def getTitle(self):
        """Return the title text displayed in the header.

        Returns
        -------
        str
            Current title text.
        """
        return self.headerLabel.text()

    def setTitle(self, title: str):
        """Update the title shown in the header.

        Parameters
        ----------
        title : str
            Text placed inside the header label.
        """
        self.headerLabel.setText(title)
        self.headerLabel.setToolTip(title)

    def _postInit(self):
        """Extension hook for subclasses to customize the layout."""
        pass

    title = Property(str, getTitle, setTitle)


class CheckableHeaderCardWidget(HeaderCardWidget):
    """Header card with a checkbox for toggling operational state."""

    def __init__(self, parent=None):
        """Create the card and add a leading checkbox.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(CheckableHeaderCardWidget, self).__init__(parent)
        self.state_checkbox = CheckBox()
        self.state_checkbox.setStyleSheet(
            self.state_checkbox.styleSheet()
            + "\nCheckBox { min-width: 24px; max-width: 24px; padding: 0; }"
        )
        self.state_checkbox.setChecked(True)
        self.state_checkbox.stateChanged.connect(self.state_changed)
        self.state_checkbox.setToolTip(self.tr("Enable or disable this card"))
        self.headerLayout.insertWidget(0, self.state_checkbox, 0, Qt.AlignmentFlag.AlignLeft)
        self.headerLayout.setStretch(1, 3)
        self.headerLayout.setContentsMargins(10, 0, 3, 0)
        self.headerLayout.setSpacing(2)
        self.viewLayout.setContentsMargins(6, 0, 6, 0)
        self.headerLayout.setAlignment(self.headerLabel, Qt.AlignmentFlag.AlignLeft)
        self.check_state = True

    def state_changed(self, state):
        """Update the enabled flag when the checkbox state switches.

        Parameters
        ----------
        state : int
            Checkbox state provided by Qt (0 unchecked, 2 checked).
        """
        if state == 2:
            self.check_state = True
        else:
            self.check_state = False


class ShareCheckableHeaderCardWidget(CheckableHeaderCardWidget):
    """Checkable card that provides export and close buttons in the header."""

    doc_page_path = ""
    doc_anchor = ""
    exportSignal = Signal()

    @staticmethod
    def _compact_header_action(button, icon_size: int = 14) -> None:
        """Use a quiet icon while retaining a practical desktop hit target."""
        button.setFixedSize(28, 28)
        button.setIconSize(QSize(icon_size, icon_size))

    def __init__(self, parent=None):
        """Create the card and attach export/close controls.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(ShareCheckableHeaderCardWidget, self).__init__(parent)

        # doc/info are presented by the workflow inspector. They stay real,
        # independently-parented
        # widgets -- not added to `headerLayout` -- purely so their
        # tooltip/accessibleName/visibility API stays available for existing
        # callers. Parenting them to a widget that is never shown keeps
        # `setVisible()` toggling their own `isHidden()` flag (for
        # compatibility) without ever painting them floating over the header
        # at (0, 0), which is what happens to an unlayouted child once
        # something calls `setVisible(True)` on it.
        self._legacy_action_holder = QWidget(self)
        self._legacy_action_holder.setVisible(False)

        self.doc_button = TransparentToolButton(FluentIcon.HELP, self._legacy_action_holder)
        self.doc_button.clicked.connect(self.open_online_doc)
        self.doc_button.setToolTip(self.tr("Open online documentation"))
        self.doc_button.setAccessibleName(self.tr("Open online documentation"))
        self.doc_button.installEventFilter(ToolTipFilter(self.doc_button, 300, ToolTipPosition.TOP))

        self.info_button = TransparentToolButton(FluentIcon.INFO, self._legacy_action_holder)
        self.info_button.clicked.connect(self.show_card_info)
        self.info_button.setToolTip(self.tr("Show card information and contributors"))
        self.info_button.setAccessibleName(
            self.tr("Show card information and contributors")
        )
        self.info_button.installEventFilter(ToolTipFilter(self.info_button, 300, ToolTipPosition.TOP))

        # Copy stays visible on top-level workflow cards; export remains in the
        # overflow menu until there is a result worth exporting.
        self.copy_json_button = TransparentToolButton(FluentIcon.COPY, self)
        self._compact_header_action(self.copy_json_button)
        self.copy_json_button.clicked.connect(self.copy_json_to_clipboard)
        self.copy_json_button.setToolTip(self.tr("Copy card JSON"))
        self.copy_json_button.setAccessibleName(self.tr("Copy card JSON"))
        self.copy_json_button.installEventFilter(ToolTipFilter(self.copy_json_button, 300, ToolTipPosition.TOP))
        self.copy_json_button.hide()

        self.export_button = TransparentToolButton(QIcon(":/images/src/images/export1.svg"), self)
        self._compact_header_action(self.export_button)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.exportSignal)
        self.export_button.setToolTip(self.tr("Export data"))
        self.export_button.setAccessibleName(self.tr("Export data"))
        self.export_button.installEventFilter(ToolTipFilter(self.export_button, 300, ToolTipPosition.TOP))

        self.close_button = TransparentToolButton(FluentIcon.CLOSE, self)
        self._compact_header_action(self.close_button, 11)
        self.close_button.clicked.connect(self.close)
        self.close_button.setToolTip(self.tr("Close card"))
        self.close_button.setAccessibleName(self.tr("Close card"))
        self.close_button.installEventFilter(ToolTipFilter(self.close_button, 300, ToolTipPosition.TOP))

        group = str(getattr(self, "group", "") or "")
        translated_group = QCoreApplication.translate("CardCatalog", group)
        self._nested_header = False
        self.category_tag = CategoryTag(translated_group, self)
        self.status_dot = StatusDot(self)
        self.status_dot.setToolTip(self.tr("Card status"))
        self.status_badge = StatusBadge(self)
        self.status_badge.hide()
        self.status_dot.stateChanged.connect(self.status_badge.set_state)
        self.headerLayout.insertWidget(1, self.category_tag, 0, Qt.AlignmentFlag.AlignLeft)
        self.headerLayout.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.status_badge, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)
        self.refresh_doc_button()

    def set_category_tag(self, text: str) -> None:
        """Update the small category pill shown next to the card title."""
        self.category_tag.setText(text)
        self._update_category_tag_visibility()

    def _update_category_tag_visibility(self) -> None:
        """Yield secondary category text before the primary title clips."""
        if getattr(self, "_group_tile_enabled", False):
            self.category_tag.setVisible(bool(self.category_tag.text().strip()))
            return
        if self._nested_header:
            self.category_tag.setVisible(False)
            return
        tag_text = self.category_tag.text().strip()
        if not tag_text:
            self.category_tag.hide()
            return

        margins = self.headerLayout.contentsMargins()
        required = margins.left() + margins.right()
        visible_widgets = []
        for index in range(self.headerLayout.count()):
            widget = self.headerLayout.itemAt(index).widget()
            if widget is None or widget is self.headerLabel:
                continue
            if widget is self.category_tag or not widget.isHidden():
                visible_widgets.append(widget)
                required += widget.sizeHint().width()
        required += self.headerLabel.fontMetrics().horizontalAdvance(
            self.headerLabel.text()
        )
        required += self.headerLayout.spacing() * len(visible_widgets)
        self.category_tag.setVisible(self.headerTopView.width() >= required)

    def set_compact_header(self, compact: bool) -> None:
        """Switch between a full top-level header and a narrow nested header."""
        self._nested_header = bool(compact)
        set_two_row = getattr(self, "set_two_row_header", None)
        if callable(set_two_row):
            set_two_row(not compact and hasattr(self, "setting_widget"))
        self._update_category_tag_visibility()
        # Card JSON is available from the right inspector; keeping the legacy
        # button hidden preserves the API without duplicating the action here.
        self.copy_json_button.setVisible(False)
        self.status_dot.setVisible(compact)
        self.status_badge.setVisible(not compact)
        self._refresh_result_action_group()
        update_close = getattr(self, "_update_close_affordance", None)
        if callable(update_close):
            update_close()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_category_tag_visibility()

    def _refresh_result_action_group(self) -> None:
        """Show available result actions as one connected visual unit."""
        group = getattr(self, "result_action_group", None)
        view_button = getattr(self, "view_output_button", None)
        if group is None or view_button is None:
            return
        top_level = not self._nested_header
        view_button.setVisible(top_level and view_button.isEnabled())
        self.export_button.setVisible(top_level and self.export_button.isEnabled())
        group.setVisible(
            top_level
            and (view_button.isEnabled() or self.export_button.isEnabled())
        )

    def set_export_available(self, available: bool) -> None:
        """Expose direct export only when this card owns a valid result."""
        available = bool(available)
        self.export_button.setEnabled(available)
        self._refresh_result_action_group()

    def _derive_builtin_doc_page_path(self) -> str:
        """Return the default docs page path for built-in Make Dataset cards."""
        configured = str(getattr(self, "doc_page_path", "") or "").strip()
        if configured:
            return configured

        try:
            module_file = Path(inspect.getfile(self.__class__)).resolve()
        except (TypeError, OSError):
            return ""

        if module_file.parent.name != "_card":
            return ""

        slug = module_file.stem.replace("_", "-")
        return f"module/make-dataset-cards/cards/{slug}.html"

    def get_online_doc_url(self) -> str:
        """Return the online documentation URL for this card, if available."""
        page_path = self._derive_builtin_doc_page_path()
        if not page_path:
            return ""

        if page_path.startswith(("http://", "https://")):
            url = page_path
        else:
            url = urljoin(DOCS_BASE_URL, page_path.lstrip("/"))

        anchor = str(getattr(self, "doc_anchor", "") or "").strip().lstrip("#")
        if anchor:
            return f"{url}#{anchor}"
        return url

    def refresh_doc_button(self) -> None:
        """Show the doc button only when an online documentation URL exists."""
        has_url = bool(self.get_online_doc_url())
        self.doc_button.setVisible(has_url)
        self.doc_button.setEnabled(has_url)

    def open_online_doc(self) -> None:
        """Open the online documentation page for the current card."""
        url = self.get_online_doc_url()
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def show_card_info(self) -> None:
        """Show contributor and provenance metadata for this card."""
        class_name = self.__class__.__name__
        metadata = CardManager.get_card_metadata(class_name) or build_card_metadata(self.__class__)
        dialog = CardMetadataDialog(metadata, self)
        dialog.exec()

    def copy_json_to_clipboard(self) -> None:
        """Copy this card's current configuration JSON to the system clipboard."""
        QApplication.clipboard().setText(self.to_json_text())
        MessageManager.send_success_message(self.tr("Card JSON copied to clipboard."))

    def to_json_text(self) -> str:
        """Return this card's current configuration as pretty JSON text."""
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)


class MakeDataCardWidget(ShareCheckableHeaderCardWidget):
    """Base widget for cards participating in the console workflow."""

    group = None
    description = ""
    card_version = ""
    contributors = ()
    maintainer = ""
    license = ""
    citation = ""
    docs_url = ""

    windowStateChangedSignal = Signal()
    viewOutputSignal = Signal(object)
    dragStartedSignal = Signal(object)
    dragFinishedSignal = Signal(object, bool)
    presentationChanged = Signal()

    def __init__(self, parent=None):
        """Configure collapse controls and state tracking.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(parent)
        self.setMouseTracking(True)
        self.window_state = "expand"
        self._drag_start_pos = None
        self._two_row_header = False
        self._group_tile_enabled = False
        self._group_tile_content: QWidget | None = None
        self._workflow_selected = False
        self._header_hovered = False
        self._close_icon = self.close_button.icon()
        self.view_output_button = TransparentToolButton(
            QIcon(":/images/src/images/show_nep.svg"),
            self,
        )
        self._compact_header_action(self.view_output_button)
        self.view_output_button.setEnabled(False)
        self.view_output_button.hide()
        self.view_output_button.setToolTip(self.tr("View this card output"))
        self.view_output_button.setAccessibleName(self.tr("View this card output"))
        self.view_output_button.installEventFilter(
            ToolTipFilter(self.view_output_button, 300, ToolTipPosition.TOP)
        )
        self.view_output_button.clicked.connect(self.request_view_output)
        self.headerLayout.removeWidget(self.export_button)
        self.result_action_group = QWidget(self.headerView)
        self.result_action_group.setFixedHeight(28)
        result_layout = QHBoxLayout(self.result_action_group)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(2)
        result_layout.addWidget(self.view_output_button)
        result_layout.addWidget(self.export_button)
        self.headerLayout.insertWidget(
            self.headerLayout.indexOf(self.close_button),
            self.result_action_group,
            0,
            Qt.AlignmentFlag.AlignRight,
        )
        self.result_action_group.hide()
        self.copy_json_button.hide()
        self.collapse_button = TransparentToolButton(QIcon(":/images/src/images/collapse.svg"), self)
        self._compact_header_action(self.collapse_button, 12)
        self.collapse_button.clicked.connect(self.collapse)
        self.collapse_button.setToolTip(self.tr("Collapse or expand card"))
        self.collapse_button.setAccessibleName(self.tr("Collapse or expand card"))
        self.collapse_button.installEventFilter(ToolTipFilter(self.collapse_button, 300, ToolTipPosition.TOP))

        self.headerLayout.insertWidget(0, self.collapse_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.drag_handle = QLabel("⠿", self)
        self.drag_handle.setFixedWidth(24)
        self.drag_handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self.drag_handle.setToolTip(self.tr("Drag to reorder card"))
        self.drag_handle.setAccessibleName(self.tr("Drag to reorder card"))
        self.drag_handle.setStyleSheet(
            "color: rgba(72, 96, 104, 175); font-size: 14px;"
        )
        self.drag_handle.installEventFilter(self)
        self.headerLayout.insertWidget(0, self.drag_handle, 0, Qt.AlignmentFlag.AlignLeft)
        self.windowStateChangedSignal.connect(self.update_window_state)
        self._update_close_affordance()

    def set_group_tile_presentation(self, enabled: bool) -> None:
        """Render a nested group child as a compact portrait tile."""
        enabled = bool(enabled)
        if enabled == self._group_tile_enabled:
            return
        self._group_tile_enabled = enabled
        summary_label = getattr(self, "summary_label", None)
        if enabled:
            self.set_compact_header(True)
            if self._group_tile_content is None:
                self._group_tile_content = QWidget(self.headerTopView)
                self._group_tile_content.setObjectName("groupCardTileContent")
                self._group_tile_layout = QVBoxLayout(self._group_tile_content)
                self._group_tile_layout.setContentsMargins(7, 6, 7, 7)
                self._group_tile_layout.setSpacing(3)
                self._group_tile_top = QWidget(self._group_tile_content)
                self._group_tile_top_layout = QHBoxLayout(self._group_tile_top)
                self._group_tile_top_layout.setContentsMargins(0, 0, 0, 0)
                self._group_tile_top_layout.setSpacing(2)
                self._group_tile_layout.addWidget(self._group_tile_top)
            self.headerLayout.addWidget(self._group_tile_content, 1)
            while self._group_tile_top_layout.count():
                self._group_tile_top_layout.takeAt(0)
            self._group_tile_top_layout.addWidget(self.drag_handle)
            self._group_tile_top_layout.addWidget(self.state_checkbox)
            self._group_tile_top_layout.addStretch(1)
            self._group_tile_top_layout.addWidget(self.status_dot)
            self._group_tile_top_layout.addWidget(self.close_button)
            self._group_tile_layout.addWidget(self.category_tag, 0, Qt.AlignmentFlag.AlignLeft)
            self._group_tile_layout.addWidget(self.headerLabel)
            if summary_label is not None:
                self.viewLayout.removeWidget(summary_label)
                self.headerInfoLayout.removeWidget(summary_label)
                self._group_tile_layout.addWidget(summary_label)
                summary_label.setWordWrap(True)
                summary_label.setFixedHeight(30)
                summary_label.show()
            self.drag_handle.setFixedWidth(18)
            self.close_button.setFixedSize(24, 24)
            self.close_button.setIconSize(QSize(10, 10))
            self.category_tag.show()
            self.headerLabel.setWordWrap(True)
            self.headerLabel.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            self.headerLabel.setFixedHeight(38)
            self.headerLayout.setContentsMargins(0, 0, 0, 0)
            self.headerTopView.setFixedHeight(140)
            self.headerView.setFixedHeight(140)
            self.view.hide()
            self.separator.hide()
            self._group_tile_content.show()
        else:
            tile_content = self._group_tile_content
            if tile_content is not None:
                self.headerLayout.removeWidget(tile_content)
                tile_content.hide()
            for widget in (
                self.drag_handle,
                self.collapse_button,
                self.state_checkbox,
                self.category_tag,
                self.headerLabel,
                self.status_dot,
                self.status_badge,
                self.result_action_group,
                self.close_button,
            ):
                self.headerLayout.removeWidget(widget)
            self.headerLayout.addWidget(self.drag_handle)
            self.headerLayout.addWidget(self.collapse_button)
            self.headerLayout.addWidget(self.state_checkbox)
            self.headerLayout.addWidget(self.category_tag)
            self.headerLayout.addWidget(self.headerLabel, 1)
            self.headerLayout.addWidget(self.status_dot)
            self.headerLayout.addWidget(self.status_badge)
            self.headerLayout.addWidget(self.result_action_group)
            self.headerLayout.addWidget(self.close_button)
            if summary_label is not None:
                self._group_tile_layout.removeWidget(summary_label)
                self.viewLayout.addWidget(summary_label)
                summary_label.setWordWrap(False)
                summary_label.setMinimumHeight(0)
                summary_label.setMaximumHeight(16777215)
            self.drag_handle.setFixedWidth(24)
            self.close_button.setFixedSize(28, 28)
            self.close_button.setIconSize(QSize(11, 11))
            self.headerLabel.setWordWrap(False)
            self.headerLabel.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.headerLabel.setMinimumHeight(0)
            self.headerLabel.setMaximumHeight(16777215)
            self.headerLayout.setContentsMargins(10, 0, 3, 0)
            self.headerTopView.setFixedHeight(48)
            self.headerView.setFixedHeight(48)
            self.view.show()
            self.separator.show()
            self.set_compact_header(True)
            self.refresh_compact_presentation()

    def set_two_row_header(self, enabled: bool) -> None:
        """Place runtime context on a dedicated row for root operation cards."""
        enabled = bool(enabled)
        if enabled == self._two_row_header:
            return
        self._two_row_header = enabled
        summary_label = getattr(self, "summary_label", None)
        if enabled:
            if summary_label is not None:
                self.viewLayout.removeWidget(summary_label)
                self.headerInfoLayout.insertWidget(0, summary_label, 1)
            self.headerLayout.removeWidget(self.status_badge)
            self.headerLayout.removeWidget(self.result_action_group)
            self.headerInfoLayout.removeWidget(self.result_action_group)
            self.headerInfoLayout.addWidget(
                self.status_badge, 0, Qt.AlignmentFlag.AlignRight
            )
            self.headerLayout.insertWidget(
                self.headerLayout.indexOf(self.close_button),
                self.result_action_group, 0, Qt.AlignmentFlag.AlignRight
            )
            self.headerTopView.setFixedHeight(40)
            self.headerInfoView.show()
            self.headerView.setFixedHeight(74)
        else:
            self.headerInfoLayout.removeWidget(self.status_badge)
            close_index = self.headerLayout.indexOf(self.close_button)
            self.headerLayout.insertWidget(
                close_index,
                self.status_badge,
                0,
                Qt.AlignmentFlag.AlignRight,
            )
            if self.headerLayout.indexOf(self.result_action_group) < 0:
                self.headerLayout.insertWidget(
                    self.headerLayout.indexOf(self.close_button),
                    self.result_action_group,
                    0,
                    Qt.AlignmentFlag.AlignRight,
                )
            if summary_label is not None:
                self.headerInfoLayout.removeWidget(summary_label)
                self.viewLayout.addWidget(summary_label)
            self.headerInfoView.hide()
            self.headerTopView.setFixedHeight(48)
            self.headerView.setFixedHeight(48)
        self._refresh_result_action_group()

    def request_view_output(self) -> None:
        """Request opening this card's current result dataset."""
        self.viewOutputSignal.emit(self)

    def set_output_available(self, available: bool) -> None:
        """Keep the card-level output action aligned with its result state."""
        available = bool(available)
        self.view_output_button.setEnabled(available)
        self.set_export_available(available)
        self._refresh_result_action_group()

    def set_workflow_selected(self, selected: bool) -> None:
        """Expose destructive close affordance for the active workflow node."""
        self._workflow_selected = bool(selected)
        self._update_close_affordance()

    def _update_close_affordance(self) -> None:
        show_icon = self._workflow_selected or self._header_hovered
        self.close_button.setEnabled(show_icon)
        self.close_button.setIcon(self._close_icon if show_icon else QIcon())

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._header_hovered = True
        self._update_close_affordance()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._header_hovered = False
        self._update_close_affordance()
        super().leaveEvent(event)

    def mousePressEvent(self, e):
        """Remember where a possible background drag started."""
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        """Enable drag-and-drop reordering for the card.

        Parameters
        ----------
        e : QMouseEvent
            Mouse move event emitted by Qt.
        """
        if e.buttons() != Qt.MouseButton.LeftButton:
            return
        if self._drag_start_pos is None:
            self._drag_start_pos = e.position().toPoint()
            return
        current_pos = e.position().toPoint()
        if (
            current_pos - self._drag_start_pos
        ).manhattanLength() < QApplication.startDragDistance():
            return
        self._start_drag(current_pos)

    def _start_drag(self, current_pos: QPoint) -> None:
        """Start a compact card drag preview from an explicit local position."""
        self.dragStartedSignal.emit(self)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-neptrainkit-card", b"card")
        drag.setMimeData(mime)

        pixmap = self.headerView.grab()
        if pixmap.width() > 460:
            pixmap = pixmap.scaledToWidth(
                460,
                Qt.TransformationMode.SmoothTransformation,
            )
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(min(28, pixmap.width() // 2), pixmap.height() // 2))

        result = drag.exec(Qt.DropAction.MoveAction)
        self._drag_start_pos = None
        self.drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self.dragFinishedSignal.emit(self, result == Qt.DropAction.MoveAction)

    def eventFilter(self, watched, event):
        drag_handle = getattr(self, "drag_handle", None)
        if drag_handle is not None and watched is drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_start_pos = self.drag_handle.mapTo(
                        self,
                        event.position().toPoint(),
                    )
                    self.drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    current_pos = self.drag_handle.mapTo(
                        self,
                        event.position().toPoint(),
                    )
                    if (
                        self._drag_start_pos is not None
                        and (current_pos - self._drag_start_pos).manhattanLength()
                        >= QApplication.startDragDistance()
                    ):
                        self._start_drag(current_pos)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_start_pos = None
                self.drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)

    def collapse(self):
        """Toggle between collapsed and expanded states."""
        if self.window_state == "collapse":
            self.window_state = "expand"
        else:
            self.window_state = "collapse"

        self.windowStateChangedSignal.emit()

    def update_window_state(self):
        """Refresh the collapse button icon to match the current state."""
        if self.window_state == "expand":
            self.collapse_button.setIcon(QIcon(":/images/src/images/collapse.svg"))
        else:
            self.collapse_button.setIcon(QIcon(":/images/src/images/expand.svg"))

    def from_dict(self, data_dict):
        """Restore persisted state values from a dictionary.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Serialized data previously generated by `to_dict`.
        """
        self.state_checkbox.setChecked(data_dict["check_state"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the card configuration for persistence.

        Returns
        -------
        dict[str, Any]
            Mapping that describes the card type and enabled state.
        """
        metadata = CardManager.get_card_metadata(self.__class__.__name__) or build_card_metadata(self.__class__)
        return {
            "class": self.__class__.__name__,
            "check_state": self.check_state,
            "metadata": {
                "card_name": metadata.card_name,
                "card_version": metadata.version,
                "contributors": [item.name for item in metadata.contributors],
            },
        }


class MakeDataCard(MakeDataCardWidget):
    """Workflow card that processes datasets in a background thread.

    Notes for card authors
    ----------------------
    - When adding provenance to ``atoms.info["Config_type"]``, do not manually
      concatenate strings. Use ``NepTrainKit.core.config_type.append_config_tag``.
    - Keep tags short, stable, and quote-free so they are safe to export via EXTXYZ.
    """

    separator = False
    card_name = "MakeDataCard"
    menu_icon = r":/images/src/images/logo.png"
    runFinishedSignal = Signal(int)

    def __init__(self, parent=None):
        """Prepare UI elements, state holders, and signals.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(parent)
        self.exportSignal.connect(self.export_data)
        self.dataset: Any = None
        self.result_dataset = []
        self._last_elapsed_seconds: float | None = None
        self.run_outcome = "idle"
        self._cancel_requested = False
        self.index = 0
        self.setting_widget = QWidget(self)
        self.viewLayout.setContentsMargins(3, 6, 3, 6)
        self.viewLayout.addWidget(self.setting_widget)
        self.settingLayout = QGridLayout(self.setting_widget)
        self.settingLayout.setContentsMargins(5, 0, 5, 0)
        self.settingLayout.setSpacing(3)
        self.summary_label = CaptionLabel("", self)
        self.summary_label.setWordWrap(False)
        self.summary_label.setMinimumWidth(0)
        self.summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.summary_label.setStyleSheet("color:#7b8990; padding: 0 6px;")
        self.summary_label.setVisible(False)
        self.viewLayout.addWidget(self.summary_label)
        self.status_label = ProcessLabel(self)
        self.vBoxLayout.addWidget(self.status_label)
        self.windowStateChangedSignal.connect(self.show_setting)

    def show_setting(self):
        """Show the configuration panel expanded, or a one-line summary collapsed."""
        if bool(self.setting_widget.property("workflowInspectorHosted")):
            self.setting_widget.show()
        else:
            self.setting_widget.setVisible(self.window_state == "expand")
        self._refresh_collapsed_summary()

    def _refresh_collapsed_summary(self) -> None:
        """Show one compact detail line when parameters live in the inspector."""
        inspector_hosted = bool(
            self.setting_widget.property("workflowInspectorHosted")
        )
        compact = inspector_hosted or self.window_state != "expand"
        summary = self._compact_detail_text() if compact else ""
        show_summary = bool(summary)
        if show_summary:
            self.summary_label.setText(summary)
            self.summary_label.setToolTip(summary)
        self.summary_label.setVisible(show_summary)
        self.status_label.setVisible(not compact)

    def _compact_detail_text(self) -> str:
        """Return the stable single-line settings summary for the canvas."""
        summary = str(self.get_summary_text() or "").strip()
        if not summary:
            summary = self.summary_label.text().strip()
        if summary:
            return summary

        metadata = CardManager.get_card_metadata(self.__class__.__name__)
        return (
            localized_card_description(metadata).strip()
            if metadata is not None
            else str(getattr(self, "description", "") or "").strip()
        )

    def refresh_compact_presentation(self) -> None:
        """Refresh the canvas summary after parameters or inspector state change."""
        self._refresh_collapsed_summary()
        self.presentationChanged.emit()

    @staticmethod
    def _safe_count(value) -> int:
        try:
            return len(value)
        except TypeError:
            return 0

    def _status_count_detail(self) -> str:
        return (
            f"{self._safe_count(self.dataset)}"
            f"→{self._safe_count(self.result_dataset)}"
        )

    def _set_card_status(
        self,
        state: str,
        detail: str = "",
        *,
        tooltip: str = "",
    ) -> None:
        """Synchronize the nested dot and top-level text badge."""
        self.status_dot.set_state(state)
        self.status_badge.set_state(state, detail)
        self.status_badge.setToolTip(tooltip or self.status_badge.label.text())

    def get_summary_text(self) -> str:
        """Return a one-line description of this card's current settings.

        Shown in place of the settings panel while the card is collapsed, so
        a long pipeline of collapsed cards stays scannable. Subclasses
        override this; the default is empty (no summary shown).
        """
        return ""

    def set_dataset(self, dataset):
        """Attach the dataset to be processed by the card.

        Parameters
        ----------
        dataset : Iterable[ase.Atoms]
            Collection of atomic structures to process.
        """
        self.dataset = dataset
        self.result_dataset = []
        self._last_elapsed_seconds = None
        self.run_outcome = "idle"
        self._cancel_requested = False
        self._set_card_status("idle")
        self.summary_label.clear()

        self.update_dataset_info()

    def write_result_dataset(self, file, **kwargs):
        """Write the processed dataset to disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Target file path for the export.
        **kwargs
            Additional keyword arguments forwarded to `ase.io.write`.
        """
        export_dataset = [
            prepare_magnetic_extxyz_export(atoms)
            for atoms in self.result_dataset
        ]
        ase_write(file, export_dataset, format="extxyz", **kwargs)

    def export_data(self):
        """Prompt the user for an export path and dump results if available."""
        if self.dataset is not None:
            path = call_path_dialog(
                self,
                self.tr("Choose a file save location"),
                "file",
                f"export_{self.card_name.replace(' ', '_')}_structure.xyz",
                file_filter="XYZ Files (*.xyz)",
            )
            if not path:
                return
            thread = BackgroundTask(self, show_tip=True, title=self.tr("Exporting data"))
            thread.start_work(self.write_result_dataset, path)

    def process_structure(self, structure):
        """Transform a single structure and return derived results.

        Parameters
        ----------
        structure : ase.Atoms
            Structure selected from the dataset.

        Returns
        -------
        list[ase.Atoms]
            Processed structures generated from the input.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method to provide logic.
        """
        raise NotImplementedError

    def get_params(self):
        """Return UI-independent operation parameters for migrated cards."""
        return None

    def set_params(self, params) -> None:
        """Apply UI-independent operation parameters to the card widgets."""

    def create_operation(self):
        """Return a UI-independent operation object for migrated cards."""
        return None

    def closeEvent(self, event):
        """Ensure worker threads are stopped before closing the card."""
        was_running, stopped_now = self._stop_worker_thread(discard_results=False)
        if was_running and stopped_now:
            self.runFinishedSignal.emit(self.index)
        if hasattr(self, "worker_thread") and self.worker_thread.isRunning():
            event.ignore()
            return
        self.deleteLater()
        super().closeEvent(event)

    def _stop_worker_thread(self, discard_results: bool = False) -> tuple[bool, bool]:
        """Request worker interruption before dropping its reference."""
        if not hasattr(self, "worker_thread"):
            return False, False

        thread = self.worker_thread
        was_running = thread.isRunning()
        if was_running:
            self._cancel_requested = True
            thread.requestInterruption()
            if not thread.wait(200):
                self.run_outcome = "canceling"
                self.set_output_available(False)
                self.status_label.set_colors(["#d49b26"])
                self.status_label.setText(self.tr("Stopping…"))
                self._set_card_status("canceling")
                return True, False

        if not discard_results:
            self.result_dataset = thread.result_dataset
        else:
            self.result_dataset = []
        self._last_elapsed_seconds = None
        del self.worker_thread
        if was_running:
            self._apply_canceled_state()
        return was_running, was_running

    def _wait_for_worker_thread(self):
        """Wait for a worker that just emitted completion before deleting it."""
        if not hasattr(self, "worker_thread"):
            return None
        thread = self.worker_thread
        if thread.isRunning():
            thread.wait()
        return thread

    def stop(self):
        """Stop any running processing thread and capture partial results."""
        was_running, stopped_now = self._stop_worker_thread(discard_results=False)
        if was_running and stopped_now:
            self.runFinishedSignal.emit(self.index)

    def run(self):
        """Launch processing in a background thread when enabled."""
        if self.check_state:
            if hasattr(self, "worker_thread") and self.worker_thread.isRunning():
                return
            self.run_outcome = "running"
            self._cancel_requested = False
            self.result_dataset = []
            self._last_elapsed_seconds = None
            self.set_output_available(False)
            self.status_label.setText(self.tr("Processing 0%"))
            self._set_card_status("running", "0%")
            self._refresh_collapsed_summary()
            operation = self.create_operation()
            params = self.get_params()
            if isinstance(operation, StructureOperation):
                self.worker_thread = DataProcessingThread(self.dataset, operation, params)
            elif isinstance(operation, DatasetOperation):
                self.worker_thread = FilterProcessingThread(
                    dataset=self.dataset,
                    operation=operation,
                    params=params,
                )
            elif isinstance(operation, GeneratorOperation):
                self.worker_thread = FilterProcessingThread(
                    dataset=self.dataset or [],
                    operation=operation,
                    params=params,
                )
            else:
                self.worker_thread = DataProcessingThread(
                    self.dataset,
                    self.process_structure,
                )
            self.status_label.set_colors(["#59745A"])

            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self._last_elapsed_seconds = 0.0
            self.run_outcome = "succeeded"
            self._set_card_status("disabled")
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)

    def update_progress(self, progress):
        """Reflect worker-thread progress on the status label.

        Parameters
        ----------
        progress : int
            Percentage reported by the background worker.
        """
        if self.run_outcome != "running":
            return
        self.status_label.setText(self.tr("Processing {progress}%").format(progress=progress))
        self.status_label.set_progress(progress)
        self._set_card_status("running", f"{progress}%")
        self._refresh_collapsed_summary()

    def on_processing_finished(self):
        """Handle a successful run and emit the completion signal."""
        worker_thread = self._wait_for_worker_thread()
        if worker_thread is None:
            return
        self.result_dataset = worker_thread.result_dataset
        self._last_elapsed_seconds = worker_thread.elapsed_seconds
        if self._cancel_requested or getattr(worker_thread, "outcome", "") == "canceled":
            del self.worker_thread
            self._apply_canceled_state()
            self.runFinishedSignal.emit(self.index)
            return
        self.run_outcome = "succeeded"
        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7"])
        self._set_card_status("succeeded", self._status_count_detail())
        self.runFinishedSignal.emit(self.index)
        del self.worker_thread

    def on_processing_error(self, error):
        """Handle runtime errors and notify the user.

        Parameters
        ----------
        error : Exception
            Exception raised by the processing thread.
        """
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red"])
        worker_thread = self._wait_for_worker_thread()
        if worker_thread is None:
            return
        self.result_dataset = []
        self._last_elapsed_seconds = getattr(worker_thread, "elapsed_seconds", None)
        del self.worker_thread
        self.run_outcome = "failed"
        self.set_output_available(False)
        translated_error = translate_runtime_message(error)
        failure_text = self.tr("Failed: {error}").format(error=translated_error)
        self.status_label.setText(failure_text)
        self.status_label.setToolTip(failure_text)
        self._set_card_status("failed", tooltip=failure_text)
        self._refresh_collapsed_summary()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(
            self.tr("Error occurred: {error}").format(error=translated_error)
        )

    def _apply_canceled_state(self) -> None:
        """Mark partial worker output as unavailable after cancellation."""
        self.run_outcome = "canceled"
        self.set_output_available(False)
        self.status_label.set_colors(["#d49b26"])
        self.status_label.setText(
            self.tr("Stopped | Partial output: {output_count}").format(
                output_count=len(self.result_dataset),
            )
        )
        self._set_card_status(
            "canceled",
            self.tr("{count} partial").format(count=len(self.result_dataset)),
        )
        self._refresh_collapsed_summary()

    def update_dataset_info(self):
        """Display dataset statistics in the status label."""
        self.set_output_available(bool(self.result_dataset))
        self.status_label.setText(self._format_dataset_info())
        self._refresh_collapsed_summary()

    def _format_dataset_info(self) -> str:
        """Return the compact input/output/time summary shown below the card."""
        text = self.tr("Input: {input_count} -> Output: {output_count}").format(
            input_count=self._safe_count(self.dataset),
            output_count=self._safe_count(self.result_dataset),
        )
        if self._last_elapsed_seconds is not None:
            text = self.tr("{summary} | Time: {seconds:.2f} s").format(
                summary=text,
                seconds=self._last_elapsed_seconds,
            )
        return text


class FilterDataCard(MakeDataCard):
    """Variant of `MakeDataCard` that filters structures instead of transforming them."""

    def __init__(self, parent=None):
        """Initialize the filter card and configure the title."""
        super().__init__(parent)
        self.setTitle(self.tr("Filter data"))

    def stop(self):
        """Terminate the worker thread and discard partial results."""
        was_running, stopped_now = self._stop_worker_thread(discard_results=True)
        if was_running and stopped_now:
            self.runFinishedSignal.emit(self.index)

    def update_progress(self, progress):
        """Display worker progress in the status label."""
        if self.run_outcome != "running":
            return
        self.status_label.setText(self.tr("Processing {progress}%").format(progress=progress))
        self.status_label.set_progress(progress)
        self._set_card_status("running", f"{progress}%")
        self._refresh_collapsed_summary()

    def on_processing_finished(self):
        """Refresh status once filtering completes."""
        super().on_processing_finished()

    def on_processing_error(self, error):
        """Handle errors raised during filtering.

        Parameters
        ----------
        error : Exception
            Exception raised by the worker thread.
        """
        super().on_processing_error(error)

    def update_dataset_info(self):
        """Display the number of structures kept by the filter."""
        super().update_dataset_info()
