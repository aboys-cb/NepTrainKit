import threading
import time

from PySide6.QtCore import QCoreApplication, QObject, QThread

from NepTrainKit.ui.threads import run_in_thread


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    app = QCoreApplication.instance() or QCoreApplication([])
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_run_in_thread_finished_callback_runs_on_parent_thread():
    app = QCoreApplication.instance() or QCoreApplication([])
    parent = QObject()
    main_thread = parent.thread()
    main_ident = threading.get_ident()
    state: dict[str, object] = {}

    def work():
        return threading.get_ident()

    def on_finished(worker_ident):
        state["worker_ident"] = worker_ident
        state["callback_ident"] = threading.get_ident()
        state["callback_thread"] = QThread.currentThread()

    thread = run_in_thread(parent, work, on_finished=on_finished)
    assert _wait_until(lambda: "callback_ident" in state)
    app.processEvents()

    assert state["worker_ident"] != main_ident
    assert state["callback_ident"] == main_ident
    assert state["callback_thread"] is main_thread
    assert thread.isFinished()


def test_run_in_thread_error_callback_runs_on_parent_thread():
    app = QCoreApplication.instance() or QCoreApplication([])
    parent = QObject()
    main_thread = parent.thread()
    main_ident = threading.get_ident()
    state: dict[str, object] = {}

    def work():
        raise ValueError("boom")

    def on_error(message):
        state["message"] = message
        state["callback_ident"] = threading.get_ident()
        state["callback_thread"] = QThread.currentThread()

    thread = run_in_thread(parent, work, on_error=on_error)
    assert _wait_until(lambda: "message" in state)
    app.processEvents()

    assert state["message"] == "boom"
    assert state["callback_ident"] == main_ident
    assert state["callback_thread"] is main_thread
    assert thread.isFinished()
