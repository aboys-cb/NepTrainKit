#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""UI-agnostic messaging facade used throughout the toolkit.

This module forwards message invocations to the optional GUI layer when
available, and falls back to logging otherwise. It avoids a hard dependency on
Qt during headless or batch operations.

Examples
--------
>>> from NepTrainKit.core.message import MessageManager
>>> MessageManager.send_info_message('Hello')  # logs in headless mode
"""

from loguru import logger


class MessageManager:
    """UI-agnostic proxy for user-visible messages.

    Notes
    -----
    - Delegates to ``NepTrainKit.ui.messages.MessageManager`` when available.
    - Falls back to :mod:`loguru` logging when the UI layer is absent.

    Examples
    --------
    >>> MessageManager.send_warning_message('Check your input')
    """

    @staticmethod
    def _ui_manager():
        try:
            from NepTrainKit.ui.messages import MessageManager as UI
            return UI
        except Exception:
            return None

    @classmethod
    def _createInstance(cls, parent=None):
        """Initialise the UI message singleton if the UI layer is present."""
        ui = cls._ui_manager()
        if ui is not None:
            ui._createInstance(parent)

    @classmethod
    def get_instance(cls):
        """Return the UI message singleton instance or ``None`` in headless mode."""
        ui = cls._ui_manager()
        return None if ui is None else ui.get_instance()

    @classmethod
    def send_info_message(cls, message, title="Tip"):
        """Emit an informational message.

        Parameters
        ----------
        message : str
            Body text to display.
        title : str, default='Tip'
            Optional title for GUI message boxes.
        """
        ui = cls._ui_manager()
        if ui is None or ui.get_instance() is None:
            logger.info(message)
        else:
            ui.send_info_message(message, title)

    @classmethod
    def send_success_message(cls, message, title="Success"):
        """Emit a success/positive message.

        Parameters
        ----------
        message : str
            Body text to display.
        title : str, default='Success'
            Optional title for GUI message boxes.
        """
        ui = cls._ui_manager()
        if ui is None or ui.get_instance() is None:
            logger.success(message)
        else:
            ui.send_success_message(message, title)

    @classmethod
    def send_warning_message(cls, message, title="Warning"):
        """Emit a warning message.

        Parameters
        ----------
        message : str
            Body text to display.
        title : str, default='Warning'
            Optional title for GUI message boxes.
        """
        ui = cls._ui_manager()
        if ui is None or ui.get_instance() is None:
            logger.warning(message)
        else:
            ui.send_warning_message(message, title)

    @classmethod
    def send_error_message(cls, message, title="Error"):
        """Emit an error message.

        Parameters
        ----------
        message : str
            Body text to display.
        title : str, default='Error'
            Optional title for GUI message boxes.
        """
        ui = cls._ui_manager()
        if ui is None or ui.get_instance() is None:
            logger.error(message)
        else:
            ui.send_error_message(message, title)

    @classmethod
    def send_message_box(cls, message, title="Tip"):
        """Show a message box in GUI mode or log otherwise.

        Parameters
        ----------
        message : str
            Body text to display.
        title : str, default='Tip'
            Optional title for GUI message boxes.
        """
        ui = cls._ui_manager()
        if ui is None or ui.get_instance() is None:
            logger.info(message)
        else:
            ui.send_message_box(message, title)

    # No-op methods here; UI methods live in NepTrainKit.ui.messages