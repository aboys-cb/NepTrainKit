from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QWidget

import NepTrainKit.ui.update as update_module
from NepTrainKit.runtime_package import (
    RuntimePackageIndexError,
    RuntimePackageInstall,
    RuntimePackageSpec,
    RuntimePackageUpdate,
    WheelArtifact,
)
from NepTrainKit.ui.update import RuntimePackageUpdateWorker


class _ImmediateTask:
    def __init__(self, *_args, **_kwargs):
        pass

    def isRunning(self) -> bool:
        return False

    def start_work(self, function) -> None:
        function()


class _AcceptedMessageBox:
    last_title = ""
    last_message = ""

    def __init__(self, title: str, message: str, _parent):
        type(self).last_title = title
        type(self).last_message = message
        self.yesButton = _Button()
        self.cancelButton = _Button()

    def exec(self) -> None:
        return None

    def result(self) -> int:
        return 1


class _Button:
    def setText(self, _text: str) -> None:
        return None


def test_runtime_update_worker_reuses_same_flow_for_another_package(
    tmp_path: Path,
) -> None:
    QApplication.instance() or QApplication([])
    parent = QWidget()
    spec = RuntimePackageSpec(
        distribution="requests",
        import_name="requests",
        version_constraint=">=2.31,<3",
    )
    update = RuntimePackageUpdate(
        current_version="2.31.0",
        latest_version="2.34.2",
        update_available=True,
        artifact=WheelArtifact(
            version="2.34.2",
            filename="requests-2.34.2-py3-none-any.whl",
            url="https://packages.invalid/requests.whl",
            sha256="0" * 64,
        ),
    )
    installed = RuntimePackageInstall(
        version="2.34.2",
        previous_version="2.31.0",
        package_path=tmp_path / "requests" / "versions" / "2.34.2",
    )
    finished: list[dict] = []

    with (
        patch.object(update_module, "BackgroundTask", _ImmediateTask),
        patch.object(update_module, "MessageBox", _AcceptedMessageBox),
        patch.object(
            update_module,
            "check_runtime_package_update",
            return_value=update,
        ) as check,
        patch.object(
            update_module,
            "install_runtime_package_update",
            return_value=installed,
        ) as install,
        patch.object(
            update_module.MessageManager,
            "send_info_message",
        ) as info,
        patch.object(update_module.MessageManager, "send_success_message"),
    ):
        worker = RuntimePackageUpdateWorker(
            parent,
            spec=spec,
            runtime_name="requests test runtime",
        )
        worker.check(on_finished=finished.append)

    check.assert_called_once_with(spec, update_module.managed_runtime_root)
    install.assert_called_once_with(spec, update_module.managed_runtime_root, update)
    info.assert_called_once()
    assert "requests" in _AcceptedMessageBox.last_message
    assert "2.31.0" in _AcceptedMessageBox.last_message
    assert "2.34.2" in _AcceptedMessageBox.last_message
    assert _AcceptedMessageBox.last_title == "requests test runtime Update"
    assert finished == [
        {
            "ok": True,
            "updated": True,
            "version": "2.34.2",
            "path": str(installed.package_path),
        }
    ]


def test_startup_runtime_update_does_not_interrupt_an_open_import_dialog():
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(640, 480)
    parent.show()
    dialog = QDialog(parent)
    layout = QVBoxLayout(dialog)
    confirm = QPushButton("Confirm import", dialog)
    layout.addWidget(confirm)
    confirm.clicked.connect(dialog.accept)
    update = RuntimePackageUpdate(
        current_version="1.0.9", latest_version="1.0.10", update_available=True, artifact=None,
    )
    loaded = []
    try:
        with (
            patch.object(update_module, "BackgroundTask", _ImmediateTask),
            patch.object(update_module, "MessageBox", wraps=_AcceptedMessageBox) as message_box,
            patch.object(update_module, "check_runtime_package_update", return_value=update),
        ):
            worker = RuntimePackageUpdateWorker(parent)
            with patch.object(worker, "_start_install") as install:
                def finish_import():
                    confirm.setFocus()
                    before = app.activeModalWidget()
                    worker.check(manual=False)
                    loaded.extend((before, app.activeModalWidget(), dialog.focusWidget()))
                    confirm.click()

                QTimer.singleShot(0, finish_import)
                assert dialog.exec() == QDialog.DialogCode.Accepted
                assert loaded == [dialog, dialog, confirm]
                notice = worker._update_notice
                assert notice is not None and notice.isVisible() and notice.window() is parent
                install.assert_not_called()
                notice.findChild(update_module.PushButton).click()
                install.assert_called_once_with(update)
                assert worker._update_notice is None
                message_box.assert_not_called()
    finally:
        parent.close()
        parent.deleteLater()


def test_startup_runtime_check_is_silent_when_no_update_is_available() -> None:
    QApplication.instance() or QApplication([])
    parent = QWidget()
    update = RuntimePackageUpdate(
        current_version="2.34.2",
        latest_version="2.34.2",
        update_available=False,
        artifact=None,
    )
    finished: list[dict] = []

    with (
        patch.object(update_module, "BackgroundTask", _ImmediateTask),
        patch.object(
            update_module,
            "check_runtime_package_update",
            return_value=update,
        ),
        patch.object(
            update_module.MessageManager,
            "send_info_message",
        ) as info,
        patch.object(
            update_module.MessageManager,
            "send_success_message",
        ) as success,
        patch.object(
            update_module.MessageManager,
            "send_warning_message",
        ) as warning,
        patch.object(
            update_module.MessageManager,
            "send_error_message",
        ) as error,
    ):
        worker = RuntimePackageUpdateWorker(parent)
        worker.check(on_finished=finished.append, manual=False)

    info.assert_not_called()
    success.assert_not_called()
    warning.assert_not_called()
    error.assert_not_called()
    assert finished == [
        {
            "ok": True,
            "updated": False,
            "current_version": "2.34.2",
        }
    ]


def test_startup_runtime_check_is_silent_when_network_is_unavailable() -> None:
    QApplication.instance() or QApplication([])
    parent = QWidget()
    finished: list[dict] = []

    with (
        patch.object(update_module, "BackgroundTask", _ImmediateTask),
        patch.object(
            update_module,
            "check_runtime_package_update",
            side_effect=RuntimePackageIndexError("network unavailable"),
        ),
        patch.object(update_module.logger, "warning") as log_warning,
        patch.object(update_module.logger, "error") as log_error,
        patch.object(
            update_module.MessageManager,
            "send_info_message",
        ) as info,
        patch.object(
            update_module.MessageManager,
            "send_success_message",
        ) as success,
        patch.object(
            update_module.MessageManager,
            "send_warning_message",
        ) as warning,
        patch.object(
            update_module.MessageManager,
            "send_error_message",
        ) as error,
    ):
        worker = RuntimePackageUpdateWorker(parent)
        worker.check(on_finished=finished.append, manual=False)

    info.assert_not_called()
    success.assert_not_called()
    warning.assert_not_called()
    error.assert_not_called()
    log_warning.assert_called_once()
    log_error.assert_not_called()
    assert finished == [{"ok": False, "error": "network unavailable"}]


def test_app_launch_always_starts_a_silent_runtime_check() -> None:
    QApplication.instance() or QApplication([])
    notifier = update_module.AutoUpdateNotifier()
    notifier.runtime_update_worker = Mock()

    with (
        patch.object(notifier, "_show_startup_pending_notice"),
        patch.object(
            update_module.Config,
            "getint",
            return_value=int(time.time()),
        ),
    ):
        notifier.start_if_due()

    notifier.runtime_update_worker.check.assert_called_once_with(manual=False)
