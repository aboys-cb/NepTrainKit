"""Card widgets supporting drag-and-drop workflows and dataset processing."""

import inspect
import json
from pathlib import Path
from urllib.parse import urljoin

from typing import Any, Iterable

from PySide6.QtCore import Qt, Signal, QMimeData, Property, QUrl, QPoint
from PySide6.QtGui import QIcon, QDrag, QPixmap, QFont, QDesktopServices
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel

from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    TransparentToolButton,
    ToolTipFilter,
    ToolTipPosition,
    FluentStyleSheet,
    setFont,
    FluentIcon,
    RoundMenu,
    Action,
)

from qfluentwidgets.components.widgets.card_widget import CardSeparator, SimpleCardWidget

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export
from NepTrainKit.core.cards.operation import DatasetOperation, GeneratorOperation, StructureOperation
from NepTrainKit.core.card_manager import build_card_metadata
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import DataProcessingThread, FilterProcessingThread, BackgroundTask
from NepTrainKit.version import DOCS_BASE_URL
from .card_metadata import CardMetadataDialog
from .compact_form import StatusDot, CategoryTag
from .label import ProcessLabel
from ase.io import write as ase_write


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
        self.separator = CardSeparator(self)
        self.view = QWidget(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.headerLayout = QHBoxLayout(self.headerView)
        self.viewLayout = QHBoxLayout(self.view)

        self.headerLayout.addWidget(self.headerLabel)
        self.headerLayout.setContentsMargins(24, 0, 16, 0)
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
        self.state_checkbox.setChecked(True)
        self.state_checkbox.stateChanged.connect(self.state_changed)
        self.state_checkbox.setToolTip(self.tr("Enable or disable this card"))
        self.headerLayout.insertWidget(0, self.state_checkbox, 0, Qt.AlignmentFlag.AlignLeft)
        self.headerLayout.setStretch(1, 3)
        self.headerLayout.setContentsMargins(10, 0, 3, 0)
        self.headerLayout.setSpacing(3)
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

    def __init__(self, parent=None):
        """Create the card and attach export/close controls.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(ShareCheckableHeaderCardWidget, self).__init__(parent)

        # doc/info/copy-json/export used to be always-visible header icons;
        # they now only back the overflow menu (see `show_overflow_menu`).
        # They stay real, independently-parented widgets -- not added to
        # `headerLayout` -- purely so their tooltip/accessibleName/visibility
        # API stays available for existing callers. Parenting them to a
        # widget that is never shown keeps `setVisible()` toggling their own
        # `isHidden()` flag (for compatibility) without ever painting them
        # floating over the header at (0, 0), which is what happens to an
        # unlayouted child once something calls `setVisible(True)` on it.
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

        self.copy_json_button = TransparentToolButton(FluentIcon.COPY, self._legacy_action_holder)
        self.copy_json_button.clicked.connect(self.copy_json_to_clipboard)
        self.copy_json_button.setToolTip(self.tr("Copy card JSON"))
        self.copy_json_button.setAccessibleName(self.tr("Copy card JSON"))
        self.copy_json_button.installEventFilter(ToolTipFilter(self.copy_json_button, 300, ToolTipPosition.TOP))

        self.export_button = TransparentToolButton(QIcon(":/images/src/images/export1.svg"), self._legacy_action_holder)
        self.export_button.clicked.connect(self.exportSignal)
        self.export_button.setToolTip(self.tr("Export data"))
        self.export_button.setAccessibleName(self.tr("Export data"))
        self.export_button.installEventFilter(ToolTipFilter(self.export_button, 300, ToolTipPosition.TOP))

        self.close_button = TransparentToolButton(FluentIcon.CLOSE, self)
        self.close_button.clicked.connect(self.close)
        self.close_button.setToolTip(self.tr("Close card"))
        self.close_button.setAccessibleName(self.tr("Close card"))
        self.close_button.installEventFilter(ToolTipFilter(self.close_button, 300, ToolTipPosition.TOP))

        self.category_tag = CategoryTag(str(getattr(self, "group", "") or ""), self)
        self.status_dot = StatusDot(self)
        self.status_dot.setToolTip(self.tr("Card status"))
        self.overflow_button = TransparentToolButton(FluentIcon.MORE, self)
        self.overflow_button.setToolTip(self.tr("More actions"))
        self.overflow_button.setAccessibleName(self.tr("More actions"))
        self.overflow_button.clicked.connect(self.show_overflow_menu)
        self.overflow_button.installEventFilter(ToolTipFilter(self.overflow_button, 300, ToolTipPosition.TOP))

        self.headerLayout.insertWidget(1, self.category_tag, 0, Qt.AlignmentFlag.AlignLeft)
        self.headerLayout.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.overflow_button, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)
        self.refresh_doc_button()

    def set_category_tag(self, text: str) -> None:
        """Update the small category pill shown next to the card title."""
        self.category_tag.setText(text)

    def show_overflow_menu(self) -> None:
        """Open the menu holding the card's secondary actions.

        Documentation only appears when `get_online_doc_url()` resolves to
        something (mirrors `doc_button`'s own visibility).
        """
        menu = RoundMenu(parent=self)
        if not self.doc_button.isHidden():
            doc_action = Action(FluentIcon.HELP, self.tr("Open documentation"), self)
            doc_action.triggered.connect(self.open_online_doc)
            menu.addAction(doc_action)

        info_action = Action(FluentIcon.INFO, self.tr("Card info and contributors"), self)
        info_action.triggered.connect(self.show_card_info)
        menu.addAction(info_action)

        copy_action = Action(FluentIcon.COPY, self.tr("Copy card JSON"), self)
        copy_action.triggered.connect(self.copy_json_to_clipboard)
        menu.addAction(copy_action)

        export_action = Action(QIcon(":/images/src/images/export1.svg"), self.tr("Export data..."), self)
        export_action.triggered.connect(self.exportSignal)
        menu.addAction(export_action)

        pos = self.overflow_button.mapToGlobal(QPoint(0, self.overflow_button.height() + 4))
        menu.exec(pos)

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
        self.view_output_button = TransparentToolButton(
            QIcon(":/images/src/images/show_nep.svg"),
            self,
        )
        self.view_output_button.setEnabled(False)
        self.view_output_button.setToolTip(self.tr("View this card output"))
        self.view_output_button.setAccessibleName(self.tr("View this card output"))
        self.view_output_button.installEventFilter(
            ToolTipFilter(self.view_output_button, 300, ToolTipPosition.TOP)
        )
        self.view_output_button.clicked.connect(self.request_view_output)
        self.headerLayout.insertWidget(
            self.headerLayout.indexOf(self.overflow_button),
            self.view_output_button,
            0,
            Qt.AlignmentFlag.AlignRight,
        )
        self.collapse_button = TransparentToolButton(QIcon(":/images/src/images/collapse.svg"), self)
        self.collapse_button.clicked.connect(self.collapse)
        self.collapse_button.setToolTip(self.tr("Collapse or expand card"))
        self.collapse_button.setAccessibleName(self.tr("Collapse or expand card"))
        self.collapse_button.installEventFilter(ToolTipFilter(self.collapse_button, 300, ToolTipPosition.TOP))

        self.headerLayout.insertWidget(0, self.collapse_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.windowStateChangedSignal.connect(self.update_window_state)

    def request_view_output(self) -> None:
        """Request opening this card's current result dataset."""
        self.viewOutputSignal.emit(self)

    def set_output_available(self, available: bool) -> None:
        """Keep the card-level output action aligned with its result state."""
        self.view_output_button.setEnabled(bool(available))

    def mousePressEvent(self, e):
        """Remember where a possible card drag started."""
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
        if (current_pos - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mime = QMimeData()
        drag.setMimeData(mime)

        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(current_pos)

        drag.exec(Qt.DropAction.MoveAction)
        self._drag_start_pos = None

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
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color:#8a95a0; padding: 0 3px;")
        self.summary_label.setVisible(False)
        self.viewLayout.addWidget(self.summary_label)
        self.status_label = ProcessLabel(self)
        self.vBoxLayout.addWidget(self.status_label)
        self.windowStateChangedSignal.connect(self.show_setting)

    def show_setting(self):
        """Show the configuration panel expanded, or a one-line summary collapsed."""
        expanded = self.window_state == "expand"
        self.setting_widget.setVisible(expanded)
        summary = "" if expanded else self.get_summary_text()
        self.summary_label.setText(summary)
        self.summary_label.setVisible(bool(summary))

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
        self.status_dot.set_state("idle")

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
            self.status_dot.set_state("running")

            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self._last_elapsed_seconds = 0.0
            self.run_outcome = "succeeded"
            self.status_dot.set_state("disabled")
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
        self.status_dot.set_state("succeeded")
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
        self.status_dot.set_state("failed")
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
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(
            self.tr("Error occurred: {error}").format(error=translated_error)
        )

    def _apply_canceled_state(self) -> None:
        """Mark partial worker output as unavailable after cancellation."""
        self.run_outcome = "canceled"
        self.set_output_available(False)
        self.status_label.set_colors(["#d49b26"])
        self.status_dot.set_state("canceled")
        self.status_label.setText(
            self.tr("Stopped | Partial output: {output_count}").format(
                output_count=len(self.result_dataset),
            )
        )

    def update_dataset_info(self):
        """Display dataset statistics in the status label."""
        self.set_output_available(bool(self.result_dataset))
        self.status_label.setText(self._format_dataset_info())

    def _format_dataset_info(self) -> str:
        """Return the compact input/output/time summary shown below the card."""
        text = self.tr("Input: {input_count} -> Output: {output_count}").format(
            input_count=len(self.dataset),
            output_count=len(self.result_dataset),
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
        self.set_output_available(bool(self.result_dataset))
        self.status_label.setText(self._format_dataset_info())
