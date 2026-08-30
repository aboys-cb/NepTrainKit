"""Console toolbar for creating and executing card instances."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QWidget
from qfluentwidgets import (
    PrimaryPushButton,
    PushButton,
    CommandBar,
    FluentIcon,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.paths import get_user_config_path, ensure_directory
from NepTrainKit.core import load_cards_from_directory
from NepTrainKit.ui.widgets.card_metadata import CardLibraryPopup

from ase.io import extxyz, cif, vasp  # noqa: F401
from NepTrainKit.ui.views._card import *  # noqa: F401, F403


card_path = ensure_directory(get_user_config_path() / "cards")
load_cards_from_directory(card_path)


class ConsoleWidget(QWidget):
    """Command bar for creating and executing card instances.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the console.

    Attributes
    ----------
    newCardSignal : Signal
        Emitted with the selected card class name when a menu entry is chosen.
    stopSignal : Signal
        Emitted when the stop action is triggered.
    runSignal : Signal
        Emitted when the run action is triggered.
    """

    newCardSignal = Signal(str)
    viewOutputSignal = Signal()
    stopSignal = Signal()
    runSignal = Signal()

    def __init__(self, parent=None):
        """Initialize the widget and populate the initial actions."""
        super().__init__(parent)
        self.setObjectName("ConsoleWidget")
        self.setFixedHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QWidget#ConsoleWidget {"
            "border: 1px solid rgba(100,120,128,38); border-radius: 10px;"
            "background: rgba(255,255,255,232); }"
        )
        self.init_ui()

    def init_ui(self):
        """Construct layouts, configure menus, and wire up actions."""
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("console_gridLayout")
        self.gridLayout.setContentsMargins(6, 3, 6, 3)
        self.setting_command = CommandBar(self)
        self.new_card_button = PrimaryPushButton(
            FluentIcon.ADD,
            self.tr("Add new card"),
            self,
        )
        self.new_card_button.setMaximumWidth(200)
        self.new_card_button.setObjectName("new_card_button")

        self.new_card_button.setToolTip(self.tr("Add a new card"))
        self.new_card_button.installEventFilter(
            ToolTipFilter(self.new_card_button, 300, ToolTipPosition.TOP)
        )
        self.new_card_button.clicked.connect(self.show_card_library)
        self.card_popup: CardLibraryPopup | None = None

        self.view_output_action = QAction(
            QIcon(r":/images/src/images/show_nep.svg"),
            self.tr("View selected outputs"),
            self,
        )
        self.view_output_action.setToolTip(
            self.tr("Open outputs from all checked cards in NEP Dataset Display")
        )
        self.view_output_action.setEnabled(False)
        self.view_output_action.triggered.connect(self.view_output)
        self.view_output_button = self.setting_command.addAction(
            self.view_output_action
        )
        self.view_output_button.setText(self.tr("View output"))
        self.view_output_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.view_output_button.adjustSize()
        self.view_output_button.setAccessibleName(self.tr("View selected outputs"))

        self.run_button = PushButton(
            QIcon(r":/images/src/images/run.svg"),
            self.tr("Run"),
            self,
        )
        self.run_button.setToolTip(self.tr("Run selected cards"))
        self.run_button.setAccessibleName(self.tr("Run selected cards"))
        self.run_button.installEventFilter(
            ToolTipFilter(self.run_button, 300, ToolTipPosition.TOP)
        )
        self.run_button.clicked.connect(self.run)

        self.stop_button = PushButton(
            QIcon(r":/images/src/images/stop.svg"),
            self.tr("Stop"),
            self,
        )
        self.stop_button.setToolTip(self.tr("Stop running cards"))
        self.stop_button.setAccessibleName(self.tr("Stop running cards"))
        self.stop_button.installEventFilter(
            ToolTipFilter(self.stop_button, 300, ToolTipPosition.TOP)
        )
        self.stop_button.clicked.connect(self.stop)

        self.gridLayout.addWidget(self.new_card_button, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.setting_command, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.run_button, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.stop_button, 0, 3, 1, 1)
        self.gridLayout.setColumnStretch(1, 1)

    def show_card_library(self):
        """Open the categorized card picker below the Add Card button."""
        if self.card_popup is None:
            self.card_popup = CardLibraryPopup(self)
            self.card_popup.cardRequested.connect(self.newCardSignal.emit)
        self.card_popup.show_for(self.new_card_button)

    def run(self, *args, **kwargs):
        """Emit the run signal to start card execution."""
        self.runSignal.emit()

    def view_output(self, *args, **kwargs):
        """Request opening outputs from all checked workflow cards."""
        self.viewOutputSignal.emit()

    def set_output_available(self, available: bool) -> None:
        """Keep the output handoff action aligned with workflow state."""
        self.view_output_action.setEnabled(bool(available))

    def stop(self, *args, **kwargs):
        """Emit the stop signal to abort card execution."""
        self.stopSignal.emit()
