"""Card widgets supporting drag-and-drop workflows and dataset processing."""

from typing import Any, Iterable

from PySide6.QtCore import Qt, Signal, QMimeData, Property
from PySide6.QtGui import QIcon, QDrag, QPixmap, QFont
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel

from qfluentwidgets import (
    CheckBox,
    TransparentToolButton,
    ToolTipFilter,
    ToolTipPosition,
    FluentStyleSheet,
    setFont,
    FluentIcon,
)

from qfluentwidgets.components.widgets.card_widget import CardSeparator, SimpleCardWidget

from NepTrainKit import utils
from NepTrainKit.core import MessageManager
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
        self.state_checkbox.setToolTip("Enable or disable this card")
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

    exportSignal = Signal()

    def __init__(self, parent=None):
        """Create the card and attach export/close controls.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super(ShareCheckableHeaderCardWidget, self).__init__(parent)
        self.export_button = TransparentToolButton(QIcon(":/images/src/images/export1.svg"), self)
        self.export_button.clicked.connect(self.exportSignal)
        self.export_button.setToolTip("Export data")
        self.export_button.installEventFilter(ToolTipFilter(self.export_button, 300, ToolTipPosition.TOP))

        self.close_button = TransparentToolButton(FluentIcon.CLOSE, self)
        self.close_button.clicked.connect(self.close)
        self.close_button.setToolTip("Close card")
        self.close_button.installEventFilter(ToolTipFilter(self.close_button, 300, ToolTipPosition.TOP))

        self.headerLayout.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)


class MakeDataCardWidget(ShareCheckableHeaderCardWidget):
    """Base widget for cards participating in the console workflow."""

    group = None

    windowStateChangedSignal = Signal()

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
        self.collapse_button = TransparentToolButton(QIcon(":/images/src/images/collapse.svg"), self)
        self.collapse_button.clicked.connect(self.collapse)
        self.collapse_button.setToolTip("Collapse or expand card")
        self.collapse_button.installEventFilter(ToolTipFilter(self.collapse_button, 300, ToolTipPosition.TOP))

        self.headerLayout.insertWidget(0, self.collapse_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.windowStateChangedSignal.connect(self.update_window_state)

    def mouseMoveEvent(self, e):
        """Enable drag-and-drop reordering for the card.

        Parameters
        ----------
        e : QMouseEvent
            Mouse move event emitted by Qt.
        """
        if e.buttons() != Qt.MouseButton.LeftButton:
            return
        drag = QDrag(self)
        mime = QMimeData()
        drag.setMimeData(mime)

        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())

        drag.exec(Qt.DropAction.MoveAction)

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
        return {
            "class": self.__class__.__name__,
            "check_state": self.check_state,
        }


class MakeDataCard(MakeDataCardWidget):
    """Workflow card that processes datasets in a background thread."""

    separator = False
    card_name = "MakeDataCard"
    menu_icon = r":/images/src/images/logo.svg"
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
        self.index = 0
        self.setting_widget = QWidget(self)
        self.viewLayout.setContentsMargins(3, 6, 3, 6)
        self.viewLayout.addWidget(self.setting_widget)
        self.settingLayout = QGridLayout(self.setting_widget)
        self.settingLayout.setContentsMargins(5, 0, 5, 0)
        self.settingLayout.setSpacing(3)
        self.status_label = ProcessLabel(self)
        self.vBoxLayout.addWidget(self.status_label)
        self.windowStateChangedSignal.connect(self.show_setting)

    def show_setting(self):
        """Show or hide the configuration panel based on state."""
        if self.window_state == "expand":
            self.setting_widget.show()
        else:
            self.setting_widget.hide()

    def set_dataset(self, dataset):
        """Attach the dataset to be processed by the card.

        Parameters
        ----------
        dataset : Iterable[ase.Atoms]
            Collection of atomic structures to process.
        """
        self.dataset = dataset
        self.result_dataset = []

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
        ase_write(file, self.result_dataset, format="extxyz", **kwargs)

    def export_data(self):
        """Prompt the user for an export path and dump results if available."""
        if self.dataset is not None:
            path = utils.call_path_dialog(
                self,
                "Choose a file save location",
                "file",
                f"export_{self.card_name.replace(' ', '_')}_structure.xyz",
                file_filter="XYZ Files (*.xyz)",
            )
            if not path:
                return
            thread = utils.LoadingThread(self, show_tip=True, title="Exporting data")
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

    def closeEvent(self, event):
        """Ensure worker threads are stopped before closing the card."""
        if hasattr(self, "worker_thread"):
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.runFinishedSignal.emit(self.index)

        self.deleteLater()
        super().closeEvent(event)

    def stop(self):
        """Stop any running processing thread and capture partial results."""
        if hasattr(self, "worker_thread"):
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.result_dataset = self.worker_thread.result_dataset
                self.update_dataset_info()
                del self.worker_thread

    def run(self):
        """Launch processing in a background thread when enabled."""
        if self.check_state:
            self.worker_thread = utils.DataProcessingThread(
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
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)

    def update_progress(self, progress):
        """Reflect worker-thread progress on the status label.

        Parameters
        ----------
        progress : int
            Percentage reported by the background worker.
        """
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):
        """Handle a successful run and emit the completion signal."""
        self.result_dataset = self.worker_thread.result_dataset
        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7"])
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
        self.result_dataset = self.worker_thread.result_dataset
        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")

    def update_dataset_info(self):
        """Display dataset statistics in the status label."""
        text = f"Input structures: {len(self.dataset)} -> Output: {len(self.result_dataset)}"
        self.status_label.setText(text)


class FilterDataCard(MakeDataCard):
    """Variant of `MakeDataCard` that filters structures instead of transforming them."""

    def __init__(self, parent=None):
        """Initialize the filter card and configure the title."""
        super().__init__(parent)
        self.setTitle("Filter Data")

    def stop(self):
        """Terminate the worker thread and discard partial results."""
        if hasattr(self, "worker_thread"):
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.result_dataset = []
                self.update_dataset_info()
                del self.worker_thread

    def update_progress(self, progress):
        """Display worker progress in the status label."""
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):
        """Refresh status once filtering completes."""
        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7"])
        self.runFinishedSignal.emit(self.index)
        if hasattr(self, "worker_thread"):
            del self.worker_thread

    def on_processing_error(self, error):
        """Handle errors raised during filtering.

        Parameters
        ----------
        error : Exception
            Exception raised by the worker thread.
        """
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red"])

        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")

    def update_dataset_info(self):
        """Display the number of structures kept by the filter."""
        text = f"Input structures: {len(self.dataset)} -> Selected: {len(self.result_dataset)}"
        self.status_label.setText(text)



