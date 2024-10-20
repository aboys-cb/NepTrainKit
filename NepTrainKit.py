#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:03
# @Author  : 兵
# @email    : 1747193328@qq.com

import sys

import pyqtgraph as pg
from PySide6.QtCore import Qt

from PySide6.QtWidgets import QApplication, QMenuBar
from qfluentwidgets import (setTheme, Theme, FluentWindow, NavigationItemPosition)
from qfluentwidgets import FluentIcon as FIF

from core import MessageManager, Config
from core.widget import *

pg.setConfigOptions(antialias=False )

pg.setConfigOption('background', 'w')  # 设置背景为白色
pg.setConfigOption('foreground', 'k')  # 设置前景元素为黑色（如坐标轴）
import utils
@utils.loghandle
class NepTrainKitMainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.setMicaEffectEnabled(False)
        self.init_ui()


    def init_ui(self):
        # create sub interface
        MessageManager._createInstance(self)
        Config()

        self.init_widget()
        self.init_navigation()
        self.initWindow()
        self.init_menu()
    def init_menu(self):
        self.menu = QMenuBar(self)

        file_menu = self.menu.addMenu("文件")

        open_dir_action = file_menu.addAction(utils.image_to_qicon('open.svg'),"打开")
        open_dir_action.triggered.connect(self.open_file_dialog)
        export_action=file_menu.addAction(utils.image_to_qicon('save.svg'),"导出")

        export_action.triggered.connect(self.export_file_dialog)


        self.titleBar.hBoxLayout.insertWidget(2, self.menu,0,Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignCenter)

    def init_navigation(self):
        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(200)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.show_nep_interface, utils.image_to_qicon('show_nep.svg'), 'NEP数据集展示')

        self.addSubInterface(self.setting_interface, FIF.SETTING, '设置', NavigationItemPosition.BOTTOM)

        self.navigationInterface.activateWindow()

    def init_widget(self):
        self.show_nep_interface=ShowNepWidget(self)

        self.setting_interface=SettingsWidget(self)
    def initWindow(self):
        self.resize(1200, 700)
        self.setWindowIcon(utils.image_to_qicon('logo.svg'))
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

if __name__ == '__main__':
    setTheme(Theme.LIGHT)

    app = QApplication(sys.argv)


    with open("./src/qss/theme.qss", "r",encoding="utf8") as f:
        app.setStyleSheet(f.read())
    w = NepTrainKitMainWindow()
    w.show()

    app.exec()


