#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import traceback
from collections.abc import Iterable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot
from qfluentwidgets import StateToolTip
from ase.build.tools import sort as ase_sort
from loguru import logger


class LoadingThread(QThread):
    progressSignal = Signal(int)

    def __init__(self, parent=None, show_tip=True, title='running'):
        super(LoadingThread, self).__init__(parent)
        self.show_tip = show_tip
        self.title = title
        self._parent = parent
        self.tip: StateToolTip
        self._kwargs: Any
        self._args: Any
        self._func: Any

    def run(self):
        result = self._func(*self._args, **self._kwargs)
        if isinstance(result, Iterable):
            for i, _ in enumerate(result):
                self.progressSignal.emit(i)

    def start_work(self, func, *args, **kwargs):
        if self.show_tip:
            self.tip = StateToolTip(self.title, 'Please wait patiently~~', self._parent)
            self.tip.show()
            self.finished.connect(self.__finished_work)
            self.tip.closedSignal.connect(self.stop_work)
            time.sleep(0.0001)
        else:
            self.tip = None  # pyright:ignore
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self.start()

    def __finished_work(self):
        if self.tip:
            self.tip.setContent('success!')
            self.tip.setState(True)

    def stop_work(self):
        self.terminate()


class DataProcessingThread(QThread):

    progressSignal = Signal(int)
    finishSignal = Signal()
    errorSignal = Signal(str)

    def __init__(self, dataset, process_func):
        super().__init__()
        self.dataset = dataset
        self.process_func = process_func
        self.result_dataset = []
        self.setStackSize(8 * 1024 * 1024)

    def run(self):
        try:
            total = len(self.dataset)
            self.progressSignal.emit(0)
            from NepTrainKit.config import Config  # Lazy import to avoid cycles
            sort_atoms = Config.getboolean("widget", "sort_atoms", False)
            for index, structure in enumerate(self.dataset):
                processed = self.process_func(structure)
                if sort_atoms:
                    processed = [ase_sort(s) for s in processed]
                self.result_dataset.extend(processed)
                self.progressSignal.emit(int((index + 1) / total * 100))
            self.finishSignal.emit()
        except Exception as e:  # noqa: BLE001
            logger.debug(traceback.format_exc())
            self.errorSignal.emit(str(e))


class FilterProcessingThread(QThread):

    progressSignal = Signal(int)
    finishSignal = Signal()
    errorSignal = Signal(str)

    def __init__(self, process_func):
        super().__init__()
        self.process_func = process_func

    def run(self):
        try:
            self.progressSignal.emit(0)
            self.process_func()
            self.progressSignal.emit(100)
            self.finishSignal.emit()
        except Exception as e:  # noqa: BLE001
            logger.debug(traceback.format_exc())
            self.errorSignal.emit(str(e))


class FunctionWorker(QObject):
    """Run an arbitrary callable in a QThread and return its result via signals.

    Notes
    -----
    The callable must not touch Qt UI objects. It should only perform pure
    computation or IO and then return a Python object.
    """

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func, args=(), kwargs=None):
        super().__init__()
        self._func = func
        self._args = args or ()
        self._kwargs = kwargs or {}

    @Slot()
    def run(self) -> None:
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as e:  # noqa: BLE001
            logger.debug(traceback.format_exc())
            self.error.emit(str(e))
            return
        self.finished.emit(result)


class CallbackRelay(QObject):
    """Forward worker results back to the relay object's thread."""

    def __init__(self, on_finished=None, on_error=None, parent=None):
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_error = on_error

    @Slot(object)
    def handle_finished(self, result) -> None:
        if self._on_finished is not None:
            self._on_finished(result)

    @Slot(str)
    def handle_error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)


def run_in_thread(parent, func, *args, on_finished=None, on_error=None, **kwargs) -> QThread:
    """Convenience helper to run ``func`` in a background QThread.

    Returns
    -------
    QThread
        Started thread. Caller should keep a reference until finished.
    """
    thread = QThread(parent)
    worker = FunctionWorker(func, args=args, kwargs=kwargs)
    worker.moveToThread(thread)
    # Keep a strong Python reference so the worker is not GC'd before `thread.started`.
    # (If GC'd early, the thread event loop can keep running and callers may never
    # receive finished/error signals.)
    setattr(thread, "_ntk_worker", worker)
    relay = CallbackRelay(on_finished=on_finished, on_error=on_error, parent=parent)
    setattr(thread, "_ntk_callback_relay", relay)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    worker.finished.connect(relay.handle_finished)
    worker.error.connect(relay.handle_error)

    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    thread.finished.connect(lambda: setattr(thread, "_ntk_worker", None))
    thread.finished.connect(lambda: setattr(thread, "_ntk_callback_relay", None))
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread


__all__ = [
    'LoadingThread',
    'DataProcessingThread',
    'FilterProcessingThread',
    'FunctionWorker',
    'run_in_thread',
]

