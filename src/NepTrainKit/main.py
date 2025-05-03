#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:03
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import sys
import traceback

from PySide6.QtCore import Qt, QFile
from PySide6.QtGui import QIcon, QFont, QPalette, QColor

from PySide6.QtWidgets import QApplication, QWidget, QGridLayout
from qfluentwidgets import (setTheme, Theme, FluentWindow, NavigationItemPosition,
                            SplitToolButton, RoundMenu)
from qfluentwidgets import FluentIcon as FIF
from loguru import logger
from NepTrainKit.core import MessageManager, Config
from NepTrainKit.core.pages import *
from NepTrainKit import utils
import warnings
warnings.filterwarnings("ignore")


class NepTrainKitMainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.setMicaEffectEnabled(False)
        self.init_ui()

    def init_ui(self):
        # create sub interface
        MessageManager._createInstance(self)
        Config()
        self.init_menu()
        self.init_widget()
        self.init_navigation()
        self.initWindow()

    def init_menu(self):
        self.menu_widget = QWidget(self)
        self.menu_widget.setStyleSheet("ButtonView{background: rgb(240, 244, 249)}")
        self.menu_gridLayout = QGridLayout(self.menu_widget)
        self.menu_gridLayout.setContentsMargins(3,0,3,0)
        self.menu_gridLayout.setSpacing(1)

        self.open_dir_button = SplitToolButton(QIcon(':/images/src/images/open.svg'),
                                               self.menu_widget)
        self.open_dir_button.clicked.connect(self.open_file_dialog)
        self.load_menu = RoundMenu(parent=self)
        self.open_dir_button.setFlyout(self.load_menu)

        self.save_dir_button = SplitToolButton(QIcon(':/images/src/images/save.svg'),
                                               self.menu_widget)
        self.save_dir_button.clicked.connect(self.export_file_dialog)

        self.save_menu = RoundMenu(parent=self)
        self.save_dir_button.setFlyout(self.save_menu)

        self.menu_gridLayout.addWidget(self.open_dir_button, 0, 0)
        self.menu_gridLayout.addWidget(self.save_dir_button, 0, 1)

        self.titleBar.hBoxLayout.insertWidget(2,
                                              self.menu_widget,
                                              0,
                                              Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignCenter)

    def init_navigation(self):
        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(200)
        self.navigationInterface.addSeparator()

        self.addSubInterface(self.show_nep_interface,
                             QIcon(':/images/src/images/show_nep.svg'),
                             'NEP Dataset Display')
        self.addSubInterface(self.make_data_widget,
                             QIcon(':/images/src/images/make.svg'),
                             'Make Data' )

        self.addSubInterface(self.setting_interface,
                             FIF.SETTING,
                             'Settings',
                             NavigationItemPosition.BOTTOM)



        self.navigationInterface.activateWindow()

    def init_widget(self):
        self.show_nep_interface = ShowNepWidget(self)
        self.make_data_widget = MakeDataWidget(self)

        self.setting_interface = SettingsWidget(self)

    def initWindow(self):
        self.resize(1200, 700)
        self.setWindowIcon(QIcon(':/images/src/images/logo.svg'))
        self.setWindowTitle(f'NepTrainKit')
        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)


    def open_file_dialog(self):


        widget = self.stackedWidget.currentWidget()
        if hasattr(widget,"open_file"):
            widget.open_file( )
    def export_file_dialog(self):
        widget = self.stackedWidget.currentWidget()
        if hasattr(widget,"export_file"):
            widget.export_file( )

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    全局异常处理函数
    """


    # 格式化异常信息
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # 写入日志
    logger.error(error_message)


def set_light_theme(app):
    # 创建一个自定义调色板
    palette = QPalette()

    # 设置亮色主题的颜色
    palette.setColor(QPalette.Window, QColor(240, 240, 240))  # 窗口背景色（浅灰）
    palette.setColor(QPalette.WindowText, Qt.black)  # 文本颜色
    palette.setColor(QPalette.Base, Qt.white)  # 输入框等背景色
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))  # 交替颜色
    palette.setColor(QPalette.Text, Qt.black)  # 输入框文字颜色
    palette.setColor(QPalette.Button, QColor(230, 230, 230))  # 按钮背景色
    palette.setColor(QPalette.ButtonText, Qt.black)  # 按钮文字颜色
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))  # 高亮颜色（选中状态）
    palette.setColor(QPalette.HighlightedText, Qt.white)  # 高亮文字颜色

    # 应用调色板
    app.setPalette(palette)

    # 禁用系统主题（可选，视平台而定）
    app.setStyle("Fusion")  # 使用 Fusion 样式，确保一致性
def main():
    setTheme(Theme.LIGHT)

    sys.excepthook = global_exception_handler
    if os.path.exists("update.zip") or os.path.exists("update.tar.gz"):
        utils.unzip()
    app = QApplication(sys.argv)

    set_light_theme(app)
    font = QFont("Arial", 12)  # 设置字体为 Arial，字号为 12
    app.setFont(font)
    theme_file = QFile(":/theme/src/qss/theme.qss")
    theme_file.open(QFile.ReadOnly )
    theme=theme_file.readAll().data().decode("utf-8")
    app.setStyleSheet(theme)
    w = NepTrainKitMainWindow()
    w.show()

    app.exec()


if __name__ == '__main__':

    main()
