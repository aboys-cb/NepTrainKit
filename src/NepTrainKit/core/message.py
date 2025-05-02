#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:38
# @Author  : 兵
# @email    : 1747193328@qq.com

from PySide6.QtCore import QObject, Signal, Qt
from qfluentwidgets import   InfoBar, InfoBarIcon, InfoBarPosition,MessageBox

class MessageManager(QObject):
    """
    全局的消息弹窗 单例模式 直接导入调用即可
    比如
    from core import MessageManager
    MessageManager.send_info_message("这是一条info消息")
    MessageManager.send_info_message( "这是一条info消息","标题")

    """
    _instance = None
    showMessageSignal = Signal(InfoBarIcon, str, str)
    showBoxSignal= Signal(str, str)

    def __init__(self,parent=None):
        super().__init__()
        self._parent = parent
        self.showMessageSignal.connect(self._show_message)
        self.showBoxSignal.connect(self._show_box)

    @classmethod
    def _createInstance(cls,parent=None):
        # 创建实例
        if not cls._instance:
            cls._instance = MessageManager(parent)

    @classmethod
    def get_instance(cls):
        cls._createInstance()
        return cls._instance

    @classmethod
    def send_info_message(cls,message,title="Tip"):
        cls._createInstance()
        cls._instance.showMessageSignal.emit(InfoBarIcon.INFORMATION, message, title)

    @classmethod
    def send_success_message(cls,message,title="Success"):
        cls._createInstance()
        cls._instance.showMessageSignal.emit(InfoBarIcon.SUCCESS, message, title)

    @classmethod
    def send_warning_message(cls,message,title="Warning"):
        cls._createInstance()
        cls._instance.showMessageSignal.emit(InfoBarIcon.WARNING, message, title)

    @classmethod
    def send_error_message(cls, message,title="Error"):
        cls._createInstance()
        cls._instance.showMessageSignal.emit(InfoBarIcon.ERROR, message, title)

    @classmethod
    def send_message_box(cls,message,title="Tip"):
        cls._createInstance()
        cls._instance.showBoxSignal.emit(message, title)

    def _show_box(self,message,title ):
        w = MessageBox(title, message, self._parent)
        w.cancelButton.hide()
        w.exec_()

    def _show_message(self,msg_type,msg,title):

        InfoBar.new(msg_type,
            title=title,
            content=msg,
            orient=Qt.Orientation.Vertical,  # vertical layout
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self._parent
        )



