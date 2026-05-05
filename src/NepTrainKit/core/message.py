#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""UI-agnostic messaging facade used throughout the toolkit.

The core layer owns the public messaging API and writes to Loguru by default.
GUI code can register a Qt-backed sink at startup without making core import
the UI package.

Examples
--------
>>> from NepTrainKit.core.message import MessageManager
>>> MessageManager.send_info_message('Hello')  # logs in headless mode
"""

from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class MessageSink(Protocol):
    """Runtime contract for message sinks registered by the GUI layer."""

    @classmethod
    def send_info_message(cls, message, title="Tip"):
        ...

    @classmethod
    def send_success_message(cls, message, title="Success"):
        ...

    @classmethod
    def send_warning_message(cls, message, title="Warning"):
        ...

    @classmethod
    def send_error_message(cls, message, title="Error"):
        ...

    @classmethod
    def send_message_box(cls, message, title="Tip"):
        ...


class _LoggerMessageSink:
    """Headless sink that maps user-visible messages to Loguru."""

    @classmethod
    def send_info_message(cls, message, title="Tip"):
        logger.info(message)

    @classmethod
    def send_success_message(cls, message, title="Success"):
        logger.success(message)

    @classmethod
    def send_warning_message(cls, message, title="Warning"):
        logger.warning(message)

    @classmethod
    def send_error_message(cls, message, title="Error"):
        logger.error(message)

    @classmethod
    def send_message_box(cls, message, title="Tip"):
        logger.info(message)


class MessageManager:
    """UI-agnostic entry point for user-visible messages."""

    _sink: MessageSink = _LoggerMessageSink

    @classmethod
    def register_sink(cls, sink: MessageSink):
        """Install the active message sink."""
        cls._sink = sink

    @classmethod
    def reset_sink(cls):
        """Restore headless logging for tests and non-GUI use."""
        cls._sink = _LoggerMessageSink

    @classmethod
    def _createInstance(cls, parent=None):
        """Backward-compatible no-op; GUI startup registers its own sink."""
        return None

    @classmethod
    def get_instance(cls):
        """Return the active sink object."""
        return cls._sink

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
        cls._sink.send_info_message(message, title)

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
        cls._sink.send_success_message(message, title)

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
        cls._sink.send_warning_message(message, title)

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
        cls._sink.send_error_message(message, title)

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
        cls._sink.send_message_box(message, title)
