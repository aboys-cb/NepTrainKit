"""Exercise destruction in subprocesses: a regression aborts the process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = r'''
import sys, threading, time
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
from PySide6.QtCore import QObject, QThread, QTimer, Slot, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import delete
from NepTrainKit.ui.threads import BackgroundTask, FilterProcessingThread, ThreadShutdownGuard, run_in_thread

kind, quit_mode, protected = sys.argv[1:]
# Windows Qt can send fatal diagnostics to the debugger instead of stderr.
# Capture the actual Qt message explicitly, before abort can discard buffers.
def qt_message(kind, context, message):
    print(message, file=sys.stderr, flush=True)
qInstallMessageHandler(qt_message)
app = QApplication([])
entered, completed = threading.Event(), threading.Event()
callbacks = []
class Window(QWidget):
    def closeEvent(self, event):
        if protected == 'yes' and not self.guard.request_close():
            event.ignore()
        else:
            event.accept()
window = Window()
if protected == 'yes':
    window.guard = ThreadShutdownGuard(window)
window.show()
def work():
    entered.set()
    time.sleep(.2)
    completed.set()
if kind == 'background':
    thread = BackgroundTask(window, show_tip=False)
    thread.start_work(work)
elif kind == 'unparented':
    thread = FilterProcessingThread(process_func=work)
    thread.start()
elif kind == 'function':
    thread = run_in_thread(window, work, on_finished=lambda result: callbacks.append(result))
else:
    class Loader(QObject):
        @Slot()
        def run(self):
            work()
    worker = Loader()
    thread = QThread(window)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.start()
heartbeat = []
timer = QTimer()
timer.setInterval(5)
def attempt():
    heartbeat.append(1)
    if entered.is_set():
        timer.timeout.disconnect(attempt)
        timer.timeout.connect(lambda: heartbeat.append(1))
        if quit_mode == 'close': window.close()
        else: app.quit()
timer.timeout.connect(attempt)
timer.start()
app.exec()
if protected == 'yes':
    assert completed.is_set(), 'application exited before native work returned'
    assert not callbacks, 'shutdown must not publish results or launch follow-up jobs'
    if quit_mode == 'close': assert len(heartbeat) > 2, 'GUI stopped processing events'
delete(window)
if kind == 'unparented': delete(thread)
print('clean exit', flush=True)
'''


def _run(tmp_path, kind, quit_mode, protected):
    script = tmp_path / 'exit_probe.py'
    script.write_text(SCRIPT)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    return subprocess.run([sys.executable, str(script), kind, quit_mode, protected],
                          env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize('kind', ['background', 'unparented', 'function', 'loader'])
@pytest.mark.parametrize('quit_mode', ['close', 'quit'])
def test_shutdown_waits_for_running_workers_without_destroying_them(tmp_path, kind, quit_mode):
    result = _run(tmp_path, kind, quit_mode, 'yes')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'clean exit' in result.stdout
    assert 'Destroyed while thread' not in result.stderr


def test_unprotected_owner_destruction_reproduces_reported_qthread_abort(tmp_path):
    result = _run(tmp_path, 'background', 'close', 'no')
    assert result.returncode != 0
    assert 'Destroyed while thread' in result.stderr


def test_shutdown_does_not_launch_deferred_audits_or_update_checks():
    from types import SimpleNamespace

    from NepTrainKit.main import NepTrainKitMainWindow
    from NepTrainKit.ui.update import AutoUpdateNotifier

    owner = SimpleNamespace(_thread_shutdown=SimpleNamespace(closing=True))
    # No other window/update attributes: a scheduled callback must return
    # before touching data or scheduling another worker during shutdown.
    NepTrainKitMainWindow.open_training_set_audit(owner)
    NepTrainKitMainWindow._request_training_set_structure_evidence(owner)
    NepTrainKitMainWindow._audit_on_page_entry(owner)
    AutoUpdateNotifier.start_if_due(SimpleNamespace(_parent=owner))


def test_application_and_fluent_widgets_use_installed_font_families(monkeypatch):
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import fontFamilies, setFontFamilies

    import NepTrainKit.main as main_module

    app = QApplication.instance()
    old_font, old_families = app.font(), fontFamilies()
    monkeypatch.setattr(main_module, 'set_light_theme', lambda app: None)
    monkeypatch.setattr(main_module, 'install_translator', lambda app: None)
    monkeypatch.setattr(main_module, '_set_macos_dock_icon', lambda *args: None)
    monkeypatch.setattr(app, 'setStyleSheet', lambda *args: None)
    try:
        main_module.configure_app(app)
        families = app.font().families()
        system = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
        assert families[0] == system
        assert fontFamilies() == families
        assert set(families[1:]).issubset(QFontDatabase.families())
    finally:
        app.setFont(old_font)
        setFontFamilies(old_families, save=False)
