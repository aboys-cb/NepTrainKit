#!/usr/bin/env python
"""Exercise the managed updater against real Requests releases and a real dialog."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LOW_REQUESTS_VERSION = "2.31.0"
UPDATE_TIMEOUT_MS = 120_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="managed runtime root; defaults to an isolated temporary directory",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="optional PNG path for the actual update confirmation dialog",
    )
    parser.add_argument(
        "--language",
        choices=("en_US", "zh_CN"),
        default="zh_CN",
        help="language used by the confirmation dialog",
    )
    return parser.parse_args()


def _run(
    runtime_root: Path,
    screenshot: Path | None,
    language: str,
) -> dict[str, str]:
    os.environ["NEPTRAINKIT_RUNTIME_ROOT"] = str(runtime_root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QWidget
    from qfluentwidgets import MessageBox

    import NepTrainKit.ui.update as update_module
    from NepTrainKit.i18n import install_translator
    from NepTrainKit.runtime_package import (
        RuntimePackageSpec,
        seed_runtime_package,
    )

    spec = RuntimePackageSpec(
        distribution="requests",
        import_name="requests",
        version_constraint=">=2.31,<3",
    )
    low_target = (
        runtime_root
        / spec.key
        / "versions"
        / LOW_REQUESTS_VERSION
    )
    if low_target.exists():
        raise RuntimeError(f"Low-version seed directory already exists: {low_target}")
    low_target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(low_target),
            f"requests=={LOW_REQUESTS_VERSION}",
        ],
        check=True,
    )
    seed_runtime_package(spec, runtime_root, LOW_REQUESTS_VERSION)

    dialog: dict[str, str] = {}
    capture_error: list[str] = []

    class AutoAcceptMessageBox(MessageBox):
        def __init__(self, title: str, message: str, parent=None):
            super().__init__(title, message, parent)
            dialog["title"] = title
            dialog["message"] = message

        def exec(self):
            def capture_and_accept() -> None:
                try:
                    if screenshot is not None:
                        screenshot.parent.mkdir(parents=True, exist_ok=True)
                        self.ensurePolished()
                        self.show()
                        QApplication.processEvents()
                        self.repaint()
                        QApplication.processEvents()
                        pixmap = QPixmap(self.size())
                        pixmap.fill(Qt.GlobalColor.transparent)
                        painter = QPainter(pixmap)
                        try:
                            self.render(painter, QPoint())
                        finally:
                            painter.end()
                        if not pixmap.save(str(screenshot), "PNG"):
                            raise RuntimeError(
                                "Unable to save update-dialog screenshot: "
                                f"{screenshot}"
                            )
                except Exception as exc:  # noqa: BLE001 - report after closing
                    capture_error.append(str(exc))
                finally:
                    self.yesButton.click()

            QTimer.singleShot(750, capture_and_accept)
            return super().exec()

    original_message_box = update_module.MessageBox
    update_module.MessageBox = AutoAcceptMessageBox
    app = QApplication.instance() or QApplication([])
    install_translator(app, language)
    parent = QWidget()
    parent.resize(760, 420)
    parent.setWindowTitle("NepTrainKit managed package update probe")
    parent.show()
    completed: dict[str, object] = {}

    worker = update_module.RuntimePackageUpdateWorker(
        parent,
        spec=spec,
        runtime_name=(
            "Requests 测试运行时"
            if language == "zh_CN"
            else "Requests test runtime"
        ),
    )

    def finish(result: dict[str, object]) -> None:
        completed.update(result)
        QTimer.singleShot(250, app.quit)

    def timeout() -> None:
        if not completed:
            completed.update(
                {
                    "ok": False,
                    "error": "Timed out waiting for the runtime update flow.",
                }
            )
            app.quit()

    QTimer.singleShot(UPDATE_TIMEOUT_MS, timeout)
    QTimer.singleShot(
        0,
        lambda: worker.check(
            on_finished=finish,
            manual=False,
        ),
    )
    app.exec()
    worker.check_thread.wait(5_000)
    worker.install_thread.wait(5_000)
    update_module.MessageBox = original_message_box

    if not completed.get("ok") or not completed.get("updated"):
        raise RuntimeError(
            str(completed.get("error") or f"Unexpected result: {completed}")
        )
    if capture_error:
        raise RuntimeError(capture_error[0])
    installed_version = str(completed.get("version") or "")
    message = dialog.get("message", "")
    if LOW_REQUESTS_VERSION not in message or installed_version not in message:
        raise AssertionError(
            "The confirmation dialog did not show the current and target versions."
        )

    restart_probe = "\n".join(
        [
            "from pathlib import Path",
            "from NepTrainKit.runtime_package import (",
            "    RuntimePackageSpec, activate_runtime_package",
            ")",
            "spec = RuntimePackageSpec('requests', 'requests', '>=2.31,<3')",
            f"activation = activate_runtime_package(spec, Path({str(runtime_root)!r}))",
            "import requests",
            f"assert requests.__version__ == {installed_version!r}",
            "assert activation.version == requests.__version__",
            "print(requests.__version__)",
        ]
    )
    restarted = subprocess.run(
        [sys.executable, "-c", restart_probe],
        cwd=runtime_root,
        capture_output=True,
        text=True,
        check=True,
    )
    restarted_version = restarted.stdout.strip().splitlines()[-1]

    return {
        "dialog_message": message,
        "dialog_title": dialog.get("title", ""),
        "installed": installed_version,
        "language": language,
        "mode": "pip-installed-or-source",
        "restarted": restarted_version,
        "seeded": LOW_REQUESTS_VERSION,
    }


def main() -> int:
    args = _parse_args()
    if args.runtime_root is not None:
        runtime_root = args.runtime_root.resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        result = _run(runtime_root, args.screenshot, args.language)
    else:
        with tempfile.TemporaryDirectory(
            prefix="neptrainkit-requests-runtime-"
        ) as temporary:
            result = _run(Path(temporary), args.screenshot, args.language)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
