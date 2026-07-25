import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.io.base import ResultData
from NepTrainKit.ui.threads import BackgroundTask, LoadingThread, run_in_thread


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    app = QApplication.instance() or QApplication([])
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_loading_thread_uses_larger_stack_for_numpy_workers():
    thread = LoadingThread(show_tip=False)
    assert thread.stackSize() >= 8 * 1024 * 1024


def test_background_task_reports_failure_without_success():
    task = BackgroundTask(show_tip=False)
    outcomes: list[str] = []
    task.succeeded.connect(lambda _result: outcomes.append("succeeded"))
    task.failed.connect(lambda message: outcomes.append(f"failed:{message}"))

    task.start_work(lambda: (_ for _ in ()).throw(ValueError("boom")))

    assert _wait_until(task.isFinished)
    assert _wait_until(lambda: bool(outcomes))
    assert task.outcome == "failed"
    assert task.error_message == "boom"
    assert outcomes == ["failed:boom"]


def test_background_task_cancels_generator_cooperatively():
    task = BackgroundTask(show_tip=False)
    outcomes: list[str] = []
    task.canceled.connect(lambda: outcomes.append("canceled"))

    def work():
        for index in range(1000):
            time.sleep(0.001)
            yield index

    task.start_work(work)
    assert _wait_until(lambda: task.isRunning())
    task.stop_work()

    assert _wait_until(task.isFinished)
    assert _wait_until(lambda: bool(outcomes))
    assert task.outcome == "canceled"
    assert outcomes == ["canceled"]


def test_run_in_thread_finished_callback_runs_on_parent_thread():
    app = QApplication.instance() or QApplication([])
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
    assert thread.stackSize() >= 8 * 1024 * 1024
    assert _wait_until(lambda: "callback_ident" in state)
    app.processEvents()

    assert state["worker_ident"] != main_ident
    assert state["callback_ident"] == main_ident
    assert state["callback_thread"] is main_thread


def test_run_in_thread_error_callback_runs_on_parent_thread():
    app = QApplication.instance() or QApplication([])
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


def test_run_in_thread_repeated_cleanup_stays_on_parent_thread():
    QApplication.instance() or QApplication([])
    parent = QObject()
    main_ident = threading.get_ident()
    callback_threads: list[int] = []
    target = 100

    def launch_next() -> None:
        run_in_thread(
            parent,
            lambda: None,
            on_finished=lambda _result: handle_finished(),
        )

    def handle_finished() -> None:
        callback_threads.append(threading.get_ident())
        if len(callback_threads) < target:
            launch_next()

    launch_next()
    assert _wait_until(lambda: len(callback_threads) == target, timeout=30.0)
    assert callback_threads == [main_ident] * target


def test_result_data_returns_to_origin_thread_after_background_load():
    app = QApplication.instance() or QApplication([])
    data = ResultData(Path("nep.txt"), Path("train.xyz"), Path("descriptor.out"))
    origin = data.thread()
    loader = QThread()

    data.move_to_load_thread(loader)
    assert data.thread() is loader
    loader.started.connect(data._restore_load_thread_affinity)
    loader.started.connect(loader.quit)
    loader.start()

    assert _wait_until(loader.isFinished)
    app.processEvents()
    assert data.thread() is origin
    loader.deleteLater()
