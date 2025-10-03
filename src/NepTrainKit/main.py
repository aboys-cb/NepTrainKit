#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Application entry point for the NepTrainKit desktop client."""

import os
import sys
if sys.platform == "darwin":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
import traceback
from pathlib import Path
import warnings

from PySide6.QtCore import Qt, QFile
from PySide6.QtGui import QIcon, QFont, QPalette, QColor
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout
from qfluentwidgets import (
    setTheme,
    Theme,
    FluentWindow,
    NavigationItemPosition,
    SplitToolButton,
    RoundMenu,
    FluentIcon,
)
from loguru import logger

from NepTrainKit.core import MessageManager
from NepTrainKit.ui.pages import *
from NepTrainKit.utils import timeit
from NepTrainKit.ui.updater import unzip
from NepTrainKit.paths import as_path

warnings.filterwarnings("ignore")




class NepTrainKitMainWindow(FluentWindow):
    """Main application window providing navigation between NepTrainKit pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setMicaEffectEnabled(False)
        self.init_ui()

    @timeit
    def init_ui(self) -> None:
        """Initialise interface elements and navigation."""
        MessageManager._createInstance(self)
        self.init_menu()
        self.init_widget()
        self.init_navigation()
        self.initWindow()

    def init_menu(self) -> None:
        """Create the toolbar housing common open/save actions."""
        self.menu_widget = QWidget(self)
        self.menu_widget.setStyleSheet("ButtonView{background: rgb(240, 244, 249)}")
        self.menu_gridLayout = QGridLayout(self.menu_widget)
        self.menu_gridLayout.setContentsMargins(3, 0, 3, 0)
        self.menu_gridLayout.setSpacing(1)

        self.open_dir_button = SplitToolButton(QIcon(':/images/src/images/open.svg'), self.menu_widget)
        self.open_dir_button.clicked.connect(self.open_file_dialog)
        self.load_menu = RoundMenu(parent=self)
        self.open_dir_button.setFlyout(self.load_menu)

        self.save_dir_button = SplitToolButton(QIcon(':/images/src/images/save.svg'), self.menu_widget)
        self.save_dir_button.clicked.connect(self.export_file_dialog)

        self.save_menu = RoundMenu(parent=self)
        self.save_dir_button.setFlyout(self.save_menu)

        self.menu_gridLayout.addWidget(self.open_dir_button, 0, 0)
        self.menu_gridLayout.addWidget(self.save_dir_button, 0, 1)
        self.titleBar.hBoxLayout.insertWidget(
            2,
            self.menu_widget,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignCenter,
        )

    def init_navigation(self) -> None:
        """Register the navigation items and default pages."""
        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(200)
        self.navigationInterface.addSeparator()

        self.addSubInterface(
            self.show_nep_interface,
            QIcon(':/images/src/images/show_nep.svg'),
            'NEP Dataset Display',
        )
        self.addSubInterface(
            self.make_data_interface,
            QIcon(':/images/src/images/make.svg'),
            'Make Data',
        )
        self.addSubInterface(
            self.data_manager_interface,
            QIcon(':/images/src/images/dataset.svg'),
            'Data Management',
        )
        self.addSubInterface(
            self.setting_interface,
            FluentIcon.SETTING,
            'Settings',
            NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.activateWindow()

    def init_widget(self) -> None:
        """Instantiate the page widgets used by the navigation interface."""
        self.show_nep_interface = ShowNepWidget(self)
        self.make_data_interface = MakeDataWidget(self)
        self.setting_interface = SettingsWidget(self)
        self.data_manager_interface = DataManagerWidget(self)

    def initWindow(self) -> None:
        """Configure top-level window parameters such as size and title."""
        self.resize(1200, 700)
        self.setWindowIcon(QIcon(':/images/src/images/logo.svg'))
        self.setWindowTitle('NepTrainKit')
        desktop = QApplication.screens()[0].availableGeometry()
        width, height = desktop.width(), desktop.height()
        self.move(width // 2 - self.width() // 2, height // 2 - self.height() // 2)

    def open_file_dialog(self) -> None:
        """Delegate to the current widget's ``open_file`` handler when available."""
        widget = self.stackedWidget.currentWidget()
        if hasattr(widget, "open_file"):
            widget.open_file()  # pyright: ignore[attr-defined]

    def export_file_dialog(self) -> None:
        """Delegate to the current widget's ``export_file`` handler when available."""
        widget = self.stackedWidget.currentWidget()
        if hasattr(widget, "export_file"):
            widget.export_file()  # pyright: ignore[attr-defined]


def global_exception_handler(exc_type, exc_value, exc_traceback) -> None:
    """Log uncaught exceptions through ``loguru`` for post-mortem analysis."""
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(error_message)


def set_light_theme(app: QApplication) -> None:
    """Apply a light colour palette to ``app``."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)
    app.setStyle("Fusion")


def main() -> None:
    """Launch the NepTrainKit GUI application."""
    setTheme(Theme.LIGHT)
    sys.excepthook = global_exception_handler

    update_zip = Path("update.zip")
    update_tar = Path("update.tar.gz")
    if update_zip.exists() or update_tar.exists():
        unzip()

    app = QApplication(sys.argv)
    set_light_theme(app)
    font = QFont("Arial", 12)
    app.setFont(font)

    theme_file = QFile(":/theme/src/qss/theme.qss")
    if theme_file.open(QFile.OpenModeFlag.ReadOnly):
        theme = theme_file.readAll().data().decode("utf-8")  # pyright: ignore[reportArgumentType]
        theme_file.close()
        app.setStyleSheet(theme)

    window = NepTrainKitMainWindow()
    window.show()

    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        resolved = as_path(dir_path).resolve()
        window.show_nep_interface.set_work_path(str(resolved))

    app.exec()


if __name__ == "__main__":
    main()
