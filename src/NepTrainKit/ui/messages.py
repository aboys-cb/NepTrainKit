#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Global UI message center (Qt-based)

from PySide6.QtCore import QObject, Signal, Qt
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition, MessageBox
from loguru import logger


class MessageManager(QObject):
    """Qt message center singleton for showing InfoBars and message boxes.

    Typical usage:
        from NepTrainKit.ui.messages import MessageManager
        MessageManager.send_info_message("Hello")
    """

    _instance = None
    showMessageSignal = Signal(InfoBarIcon, str, str)
    showBoxSignal = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self._instance: MessageManager
        self.showMessageSignal.connect(self._show_message)
        self.showBoxSignal.connect(self._show_box)

    @classmethod
    def _createInstance(cls, parent=None):
        if not cls._instance:
            cls._instance = MessageManager(parent)

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def send_info_message(cls, message, title="Tip"):
        if cls._instance is None:
            logger.info(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.INFORMATION, message, title)

    @classmethod
    def send_success_message(cls, message, title="Success"):
        if cls._instance is None:
            logger.success(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.SUCCESS, message, title)

    @classmethod
    def send_warning_message(cls, message, title="Warning"):
        if cls._instance is None:
            logger.warning(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.WARNING, message, title)

    @classmethod
    def send_error_message(cls, message, title="Error"):
        if cls._instance is None:
            logger.error(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.ERROR, message, title)

    @classmethod
    def send_message_box(cls, message, title="Tip"):
        if cls._instance is None:
            logger.info(message)
        else:
            cls._instance.showBoxSignal.emit(message, title)

    def _show_box(self, message, title):
        w = MessageBox(title, message, self._parent)
        w.cancelButton.hide()
        w.exec_()

    def _show_message(self, msg_type, msg, title):
        if msg_type == InfoBarIcon.ERROR:
            duration = 10000
        elif msg_type == InfoBarIcon.WARNING:
            duration = 8000
        else:
            duration = 5000
        InfoBar.new(
            msg_type,
            title=title,
            content=msg,
            orient=Qt.Orientation.Vertical,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=duration,
            parent=self._parent,
        )

