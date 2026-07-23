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

from NepTrainKit.core.cards.operation import DatasetOperation, GeneratorOperation, StructureOperation


_NUMPY_WORKER_STACK_SIZE = 8 * 1024 * 1024


class LoadingThread(QThread):
    progressSignal = Signal(int)

    def __init__(self, parent=None, show_tip=True, title='running'):
        super(LoadingThread, self).__init__(parent)
        self.setStackSize(_NUMPY_WORKER_STACK_SIZE)
        self.show_tip = show_tip
        self.title = self.tr("Running") if title == "running" else title
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
            self.tip = StateToolTip(self.title, self.tr("Please wait patiently..."), self._parent)
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
            self.tip.setContent(self.tr("Success"))
            self.tip.setState(True)

    def stop_work(self):
        self.terminate()


class DataProcessingThread(QThread):

    progressSignal = Signal(int)
    finishSignal = Signal()
    errorSignal = Signal(str)

    def __init__(self, dataset, process_func, params=None):
        super().__init__()
        self.dataset = dataset
        self.process_func = process_func
        self.params = params
        self.result_dataset = []
        self.elapsed_seconds = 0.0
        self.setStackSize(_NUMPY_WORKER_STACK_SIZE)

    def run(self):
        start = time.perf_counter()
        try:
            total = len(self.dataset)
            self.progressSignal.emit(0)
            from NepTrainKit.config import Config  # Lazy import to avoid cycles
            sort_atoms = Config.getboolean("widget", "sort_atoms", False)
            for index, structure in enumerate(self.dataset):
                if self.isInterruptionRequested():
                    break
                if isinstance(self.process_func, StructureOperation):
                    processed = self.process_func.run_structure(structure, self.params)
                else:
                    processed = self.process_func(structure)
                if sort_atoms:
                    processed = [ase_sort(s) for s in processed]
                self.result_dataset.extend(processed)
                self.progressSignal.emit(int((index + 1) / total * 100))
                if self.isInterruptionRequested():
                    break
            self.elapsed_seconds = time.perf_counter() - start
            self.finishSignal.emit()
        except Exception as e:  # noqa: BLE001
            self.elapsed_seconds = time.perf_counter() - start
            logger.debug(traceback.format_exc())
            self.errorSignal.emit(str(e))


class FilterProcessingThread(QThread):

    progressSignal = Signal(int)
    finishSignal = Signal()
    errorSignal = Signal(str)

    def __init__(self, process_func=None, dataset=None, operation=None, params=None):
        super().__init__()
        self.process_func = process_func
        self.dataset = dataset
        self.operation = operation
        self.params = params
        self.result_dataset = []
        self.elapsed_seconds = 0.0
        self.setStackSize(_NUMPY_WORKER_STACK_SIZE)

    def run(self):
        start = time.perf_counter()
        try:
            self.progressSignal.emit(0)
            if isinstance(self.operation, DatasetOperation):
                self.result_dataset = self.operation.run_dataset(self.dataset, self.params)
            elif isinstance(self.operation, GeneratorOperation):
                self.result_dataset = self.operation.generate(self.params)
            else:
                result = self.process_func()
                if result is not None:
                    self.result_dataset = result
            self.progressSignal.emit(100)
            self.elapsed_seconds = time.perf_counter() - start
            self.finishSignal.emit()
        except Exception as e:  # noqa: BLE001
            self.elapsed_seconds = time.perf_counter() - start
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
    """Forward worker results and final cleanup to the parent's thread."""

    def __init__(self, on_finished=None, on_error=None, parent=None):
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_error = on_error
        self._thread: QThread | None = None
        self._worker: FunctionWorker | None = None
        self._outcome: tuple[str, object] | None = None
        self._thread_finished = False
        self._worker_destroyed = False

    def bind_job(self, thread: QThread, worker: FunctionWorker) -> None:
        """Retain the job objects until the worker thread has fully stopped."""
        self._thread = thread
        self._worker = worker

    @Slot(object)
    def capture_finished(self, result) -> None:
        self._outcome = ("finished", result)
        self._finalize_if_ready()

    @Slot(str)
    def capture_error(self, message: str) -> None:
        self._outcome = ("error", message)
        self._finalize_if_ready()

    @Slot()
    def handle_thread_finished(self) -> None:
        self._thread_finished = True
        self._finalize_if_ready()

    @Slot()
    def handle_worker_destroyed(self) -> None:
        self._worker_destroyed = True
        self._finalize_if_ready()

    def _finalize_if_ready(self) -> None:
        """Publish the result only after native thread and worker teardown."""
        if (
            self._outcome is None
            or not self._thread_finished
            or not self._worker_destroyed
        ):
            return

        thread = self._thread
        outcome, payload = self._outcome
        try:
            # ``QThread.finished`` is emitted before all thread-local cleanup is
            # complete.  Synchronize here before any owner can release QThread.
            if thread is not None:
                thread.wait()
            if outcome == "finished":
                if self._on_finished is not None:
                    self._on_finished(payload)
            elif self._on_error is not None:
                self._on_error(str(payload))
        finally:
            self._worker = None
            self._thread = None
            self._outcome = None
            self._on_finished = None
            self._on_error = None
            if thread is not None:
                thread.deleteLater()
            self.deleteLater()


def run_in_thread(parent, func, *args, on_finished=None, on_error=None, **kwargs) -> QThread:
    """Convenience helper to run ``func`` in a background QThread.

    Returns
    -------
    QThread
        Started thread. Caller should keep a reference until finished.
    """
    thread = QThread(parent)
    # Qt's default macOS worker stack can be as small as 544 KiB.  NumPy/SciPy
    # routines used by background dataset analysis can exceed that limit.
    thread.setStackSize(_NUMPY_WORKER_STACK_SIZE)
    worker = FunctionWorker(func, args=args, kwargs=kwargs)
    worker.moveToThread(thread)
    relay = CallbackRelay(on_finished=on_finished, on_error=on_error, parent=parent)
    # The main-thread relay owns the Python references.  Never clear these from
    # a context-less ``QThread.finished`` lambda: that signal is emitted by the
    # worker thread and can race Shiboken wrapper destruction.
    relay.bind_job(thread, worker)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    worker.finished.connect(relay.capture_finished)
    worker.error.connect(relay.capture_error)

    # QThread guarantees that deferred deletions posted from ``finished`` are
    # processed before the native thread is fully torn down.
    thread.finished.connect(worker.deleteLater)
    worker.destroyed.connect(relay.handle_worker_destroyed)
    thread.finished.connect(relay.handle_thread_finished)

    thread.start()
    return thread


__all__ = [
    'LoadingThread',
    'DataProcessingThread',
    'FilterProcessingThread',
    'FunctionWorker',
    'run_in_thread',
]
