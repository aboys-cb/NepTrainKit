#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import traceback
from collections.abc import Iterable
from typing import Any

from PySide6.QtCore import QThread, Signal
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


__all__ = [
    'LoadingThread',
    'DataProcessingThread',
    'FilterProcessingThread',
]

