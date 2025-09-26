"""Console toolbar for managing registered card widgets."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QWidget
from qfluentwidgets import (
    RoundMenu,
    PrimaryDropDownPushButton,
    CommandBar,
    Action,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.paths import get_user_config_path, ensure_directory
from NepTrainKit.core import load_cards_from_directory, CardManager
from NepTrainKit.config import Config

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
    stopSignal = Signal()
    runSignal = Signal()

    def __init__(self, parent=None):
        """Initialize the widget and populate the initial actions."""
        super().__init__(parent)
        self.setObjectName("ConsoleWidget")
        self.setMinimumHeight(50)
        self.init_ui()

    def init_ui(self):
        """Construct layouts, configure menus, and wire up actions."""
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("console_gridLayout")
        self.setting_command = CommandBar(self)
        self.new_card_button = PrimaryDropDownPushButton(
            QIcon(":/images/src/images/copy_figure.svg"),
            "Add new card",
            self,
        )
        self.new_card_button.setMaximumWidth(200)
        self.new_card_button.setObjectName("new_card_button")

        self.new_card_button.setToolTip("Add a new card")
        self.new_card_button.installEventFilter(
            ToolTipFilter(self.new_card_button, 300, ToolTipPosition.TOP)
        )

        self.menu = RoundMenu(parent=self)

        use_group_menu = Config.getboolean("widget", "use_group_menu", False)
        if use_group_menu:
            group_menus = {}
            for class_name, card_class in CardManager.card_info_dict.items():
                group = getattr(card_class, "group", None)
                target_menu = self.menu
                if group:
                    if group not in group_menus:
                        group_menu = RoundMenu(group, self.menu)
                        group_menus[group] = group_menu
                        self.menu.addMenu(group_menu)
                    target_menu = group_menus[group]
                if card_class.separator:
                    target_menu.addSeparator()
                action = QAction(QIcon(card_class.menu_icon), card_class.card_name)
                action.setObjectName(class_name)
                target_menu.addAction(action)
        else:
            for class_name, card_class in CardManager.card_info_dict.items():
                if card_class.separator:
                    self.menu.addSeparator()
                action = QAction(QIcon(card_class.menu_icon), card_class.card_name)
                action.setObjectName(class_name)
                self.menu.addAction(action)

        self.menu.triggered.connect(self.menu_clicked)
        self.new_card_button.setMenu(self.menu)
        self.setting_command.addWidget(self.new_card_button)

        self.setting_command.addSeparator()
        run_action = Action(
            QIcon(r":/images/src/images/run.svg"),
            'Run',
            triggered=self.run,
        )
        run_action.setToolTip('Run selected cards')
        run_action.installEventFilter(
            ToolTipFilter(run_action, 300, ToolTipPosition.TOP)
        )

        self.setting_command.addAction(run_action)
        stop_action = Action(
            QIcon(r":/images/src/images/stop.svg"),
            'Stop',
            triggered=self.stop,
        )
        stop_action.setToolTip('Stop running cards')
        stop_action.installEventFilter(
            ToolTipFilter(stop_action, 300, ToolTipPosition.TOP)
        )

        self.setting_command.addAction(stop_action)

        self.gridLayout.addWidget(self.setting_command, 0, 0, 1, 1)

    def menu_clicked(self, action):
        """Emit the card selection signal.

        Parameters
        ----------
        action : QAction
            Triggered menu action whose object name stores the card class.
        """
        self.newCardSignal.emit(action.objectName())

    def run(self, *args, **kwargs):
        """Emit the run signal to start card execution."""
        self.runSignal.emit()

    def stop(self, *args, **kwargs):
        """Emit the stop signal to abort card execution."""
        self.stopSignal.emit()

