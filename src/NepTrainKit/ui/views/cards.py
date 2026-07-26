"""Console toolbar for managing registered card widgets."""

from PySide6.QtCore import QPoint, Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QWidget
from qfluentwidgets import (
    RoundMenu,
    PrimaryDropDownPushButton,
    PushButton,
    CommandBar,
    Action,
    FluentIcon,
    MenuAnimationType,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.paths import get_user_config_path, ensure_directory
from NepTrainKit.core import load_cards_from_directory, CardManager
from NepTrainKit.config import Config
from NepTrainKit.ui.widgets.card_metadata import (
    CardLibraryDialog,
    card_tooltip,
    localized_card_group,
    localized_card_name,
)

from ase.io import extxyz, cif, vasp  # noqa: F401
from NepTrainKit.ui.views._card import *  # noqa: F401, F403


card_path = ensure_directory(get_user_config_path() / "cards")
_CARD_MENU_MAX_VISIBLE_ITEMS = 16

load_cards_from_directory(card_path)


class _ScreenSafeRoundMenu(RoundMenu):
    """Show the tall card menu without an off-screen start animation."""

    def exec(
        self,
        pos: QPoint,
        ani: bool = True,
        aniType: MenuAnimationType = MenuAnimationType.DROP_DOWN,
    ) -> None:
        super().exec(pos, ani=False, aniType=MenuAnimationType.NONE)


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
    pasteSignal : Signal
        Emitted when the user requests card creation from clipboard JSON.
    stopSignal : Signal
        Emitted when the stop action is triggered.
    runSignal : Signal
        Emitted when the run action is triggered.
    """

    newCardSignal = Signal(str)
    pasteSignal = Signal()
    copySignal = Signal()
    viewOutputSignal = Signal()
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

        self.menu = _ScreenSafeRoundMenu(parent=self)

        use_group_menu = Config.getboolean("widget", "use_group_menu", False)
        if use_group_menu:
            group_menus = {}
            for class_name, card_class in CardManager.card_info_dict.items():
                if not getattr(card_class, "discoverable", True):
                    continue
                group = getattr(card_class, "group", None)
                metadata = CardManager.get_card_metadata(class_name)
                target_menu = self.menu
                if group:
                    if group not in group_menus:
                        group_label = (
                            localized_card_group(metadata)
                            if metadata is not None
                            else group
                        )
                        group_menu = RoundMenu(group_label, self.menu)
                        group_menus[group] = group_menu
                        self.menu.addMenu(group_menu)
                    target_menu = group_menus[group]
                if card_class.separator:
                    target_menu.addSeparator()
                action_text = (
                    localized_card_name(metadata)
                    if metadata is not None
                    else card_class.card_name
                )
                action = QAction(QIcon(card_class.menu_icon), action_text)
                action.setObjectName(class_name)
                if metadata is not None:
                    action.setToolTip(card_tooltip(metadata))
                target_menu.addAction(action)
        else:
            for class_name, card_class in CardManager.card_info_dict.items():
                if not getattr(card_class, "discoverable", True):
                    continue
                if card_class.separator:
                    self.menu.addSeparator()
                metadata = CardManager.get_card_metadata(class_name)
                action_text = (
                    localized_card_name(metadata)
                    if metadata is not None
                    else card_class.card_name
                )
                action = QAction(QIcon(card_class.menu_icon), action_text)
                action.setObjectName(class_name)
                if metadata is not None:
                    action.setToolTip(card_tooltip(metadata))
                self.menu.addAction(action)

        self.menu.view.setMaxVisibleItems(_CARD_MENU_MAX_VISIBLE_ITEMS)
        self.menu.triggered.connect(self.menu_clicked)
        self.new_card_button.setMenu(self.menu)
        self.setting_command.addWidget(self.new_card_button)

        self.find_card_button = PushButton(
            FluentIcon.SEARCH,
            self.tr("Find card"),
            self,
        )
        self.find_card_button.setToolTip(
            self.tr("Search cards and add the selected card to the workspace")
        )
        self.find_card_button.installEventFilter(
            ToolTipFilter(self.find_card_button, 300, ToolTipPosition.TOP)
        )
        self.find_card_button.clicked.connect(self.show_card_library)
        self.setting_command.addWidget(self.find_card_button)

        paste_action = Action(
            FluentIcon.PASTE,
            self.tr("Paste JSON"),
            triggered=self.paste,
        )
        paste_action.setToolTip(self.tr("Create card(s) from clipboard JSON"))
        paste_action.installEventFilter(
            ToolTipFilter(paste_action, 300, ToolTipPosition.TOP)
        )
        self.setting_command.addAction(paste_action)

        copy_action = Action(
            FluentIcon.COPY,
            self.tr("Copy JSON"),
            triggered=self.copy,
        )
        copy_action.setToolTip(self.tr("Copy current workflow card JSON"))
        copy_action.installEventFilter(
            ToolTipFilter(copy_action, 300, ToolTipPosition.TOP)
        )
        self.setting_command.addAction(copy_action)

        self.view_output_button = PushButton(
            QIcon(r":/images/src/images/show_nep.svg"),
            self.tr("View selected outputs"),
            self,
        )
        self.view_output_button.setToolTip(
            self.tr(
                "Open outputs from all checked cards in NEP Dataset Display"
            )
        )
        self.view_output_button.setAccessibleName(self.tr("View selected outputs"))
        self.view_output_button.setEnabled(False)
        self.view_output_button.clicked.connect(self.view_output)
        self.setting_command.addWidget(self.view_output_button)

        self.setting_command.addSeparator()
        run_action = Action(
            QIcon(r":/images/src/images/run.svg"),
            self.tr("Run"),
            triggered=self.run,
        )
        run_action.setToolTip(self.tr("Run selected cards"))
        run_action.installEventFilter(
            ToolTipFilter(run_action, 300, ToolTipPosition.TOP)
        )

        self.setting_command.addAction(run_action)
        stop_action = Action(
            QIcon(r":/images/src/images/stop.svg"),
            self.tr("Stop"),
            triggered=self.stop,
        )
        stop_action.setToolTip(self.tr("Stop running cards"))
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

    def show_card_library(self):
        """Open the searchable card browser and forward add requests."""
        dialog = CardLibraryDialog(self)
        dialog.cardRequested.connect(self.newCardSignal.emit)
        dialog.exec()

    def run(self, *args, **kwargs):
        """Emit the run signal to start card execution."""
        self.runSignal.emit()

    def paste(self, *args, **kwargs):
        """Emit the paste signal to append cards from clipboard JSON."""
        self.pasteSignal.emit()

    def copy(self, *args, **kwargs):
        """Emit the copy signal to copy workflow JSON to the clipboard."""
        self.copySignal.emit()

    def view_output(self, *args, **kwargs):
        """Request opening outputs from all checked workflow cards."""
        self.viewOutputSignal.emit()

    def set_output_available(self, available: bool) -> None:
        """Keep the output handoff action aligned with workflow state."""
        self.view_output_button.setEnabled(bool(available))

    def stop(self, *args, **kwargs):
        """Emit the stop signal to abort card execution."""
        self.stopSignal.emit()
