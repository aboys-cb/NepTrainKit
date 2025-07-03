#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QWidget
from qfluentwidgets import RoundMenu, PrimaryDropDownPushButton, CommandBar, Action, ToolTipFilter, ToolTipPosition

from NepTrainKit import get_user_config_path
from NepTrainKit.core import load_cards_from_directory, CardManager, Config
from ase.io import extxyz,cif,vasp
from NepTrainKit.views._card import *

user_config_path = get_user_config_path()
if os.path.exists(f"{user_config_path}/cards"):
    load_cards_from_directory(os.path.join(user_config_path, "cards"))
else:
    os.makedirs(f"{user_config_path}/cards")

class ConsoleWidget(QWidget):
    """
控制台"""
    newCardSignal = Signal(str)  # 定义一个信号，用于通知上层组件新增卡片
    stopSignal = Signal()
    runSignal = Signal( )
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleWidget")
        self.setMinimumHeight(50)
        self.init_ui()

    def init_ui(self):
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("console_gridLayout")
        self.setting_command =CommandBar(self)
        self.new_card_button = PrimaryDropDownPushButton(QIcon(":/images/src/images/copy_figure.svg"),
                                                         "Add new card",self)
        self.new_card_button.setMaximumWidth(200 )
        self.new_card_button.setObjectName("new_card_button")

        self.new_card_button.setToolTip("Add a new card")
        self.new_card_button.installEventFilter(ToolTipFilter(self.new_card_button, 300, ToolTipPosition.TOP))

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
        run_action = Action(QIcon(r":/images/src/images/run.svg"), 'Run', triggered=self.run)
        run_action.setToolTip('Run selected cards')
        run_action.installEventFilter(ToolTipFilter(run_action, 300, ToolTipPosition.TOP))

        self.setting_command.addAction(run_action)
        stop_action = Action(QIcon(r":/images/src/images/stop.svg"), 'Stop', triggered=self.stop)
        stop_action.setToolTip('Stop running cards')
        stop_action.installEventFilter(ToolTipFilter(stop_action, 300, ToolTipPosition.TOP))

        self.setting_command.addAction(stop_action)



        self.gridLayout.addWidget(self.setting_command, 0, 0, 1, 1)

    def menu_clicked(self,action):


        self.newCardSignal.emit(action.objectName())

    def run(self,*args,**kwargs):
        self.runSignal.emit()
    def stop(self,*args,**kwargs):
        self.stopSignal.emit()
