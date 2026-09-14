#!/usr/bin/env python
r"""Launch NepTrainKit with diagnostics for native crashes and Qt warnings.

Run it from a source checkout (Windows included) and it writes everything it sees
to stderr plus ``tmp/qt_diagnostics_<timestamp>.log``::

    python tools\\diagnose_qt_run.py                # normal run
    python tools\\diagnose_qt_run.py > out.txt 2>&1 # keep the console output too

A native crash (access violation) never reaches ``sys.excepthook``, and Qt prints
its own warnings outside ``loguru``, so the three diagnostics below cover the paths
that stay invisible otherwise:

* ``faulthandler`` prints the Python stacks of every thread for a fatal signal, so
  a crash that kills the process still names the Python call that died.
* A Qt message handler routes Qt messages into the log and attaches the current
  Python stack to every warning or error -- that includes the
  ``UpdateLayeredWindowIndirect failed ... 参数错误`` warning that a translucent
  window logs when a dirty region reaches outside it.
* An overlay guard reports the child graphics effects that paint outside a
  translucent top-level window (the dirty region Windows rejects), and traces the
  top-level windows and focus changes, which shows what was opening when it died.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loguru import logger  # noqa: E402 - the source tree must be importable first
from PySide6.QtCore import QEvent, QObject, Qt, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_QT_LEVELS = {
    QtMsgType.QtDebugMsg: "DEBUG",
    QtMsgType.QtInfoMsg: "INFO",
    QtMsgType.QtWarningMsg: "WARNING",
    QtMsgType.QtCriticalMsg: "ERROR",
    QtMsgType.QtFatalMsg: "CRITICAL",
}


def _log_file() -> Path:
    """Return the file that receives the diagnostic log."""
    directory = ROOT / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"qt_diagnostics_{datetime.now():%Y%m%d_%H%M%S}.log"


def _report_environment() -> None:
    """Log the versions and environment variables that change Qt's behaviour."""
    logger.info("Python {}", sys.version.replace("\n", " "))
    logger.info("PySide6 {}", __import__("PySide6").__version__)
    logger.info("platform {}", sys.platform)
    for name in (
        "QT_QPA_PLATFORM",
        "QT_LOGGING_RULES",
        "QT_FATAL_WARNINGS",
        "QT_DEBUG_PLUGINS",
        "QT_OPENGL",
        "QT_SCALE_FACTOR",
        "PYTHONFAULTHANDLER",
    ):
        if name in os.environ:
            logger.info("env {}={}", name, os.environ[name])


def _install_qt_message_handler() -> None:
    """Route Qt's own messages into the log, with a Python stack for warnings."""

    def handler(mode, context, message):  # noqa: ANN001 - Qt callback signature
        level = _QT_LEVELS.get(mode, "INFO")
        location = f" ({context.file}:{context.line})" if context.file else ""
        text = f"[Qt] {message}{location}"
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            frames = [frame for frame in traceback.extract_stack()[:-1] if Path(frame.filename) != Path(__file__)]
            text = f"{text}\n{''.join(traceback.format_list(frames[-4:]))}"
        logger.log(level, text)

    qInstallMessageHandler(handler)


class _OverlayGuard(QObject):
    """Trace the overlay windows of the app and the effects that leave them."""

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._reported: set[tuple[int, int, tuple[int, int, int, int]]] = set()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Report overlays on show/resize and trace the windows that appear."""
        event_type = event.type()
        if not isinstance(watched, QWidget):
            return False
        if event_type in (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.Close) and watched.isWindow():
            logger.debug(
                "[窗口] {} {} {}{} rect={}",
                event_type.name,
                type(watched).__name__,
                watched.objectName() or "-",
                " 半透明" if _is_translucent(watched) else "",
                watched.geometry().getRect(),
            )
        if event_type in (QEvent.Type.Show, QEvent.Type.Resize) and _is_translucent(watched):
            self._report_overshoots(watched)
        if event_type == QEvent.Type.FocusIn:
            logger.debug(
                "[焦点] {} {} 窗口={}",
                type(watched).__name__,
                watched.objectName() or "-",
                type(watched.window()).__name__,
            )
        return False

    def _report_overshoots(self, window: QWidget) -> None:
        """Log every effect that paints outside ``window`` (see module docstring)."""
        from PySide6.QtCore import QPoint, QRectF

        rect = window.rect()
        for widget in [window, *window.findChildren(QWidget)]:
            effect = widget.graphicsEffect()
            if effect is None or widget.window() is not window:
                continue
            offset = widget.mapTo(window, QPoint(0, 0))
            bounds = (
                effect.boundingRectFor(QRectF(widget.rect()))
                .translated(float(offset.x()), float(offset.y()))
                .toAlignedRect()
            )
            if rect.contains(bounds):
                continue
            key = (id(window), id(widget), bounds.getRect())
            if key in self._reported:
                continue
            self._reported.add(key)
            logger.warning(
                "[越界] {} {} 的 {} 绘制范围 {} 超出窗口 {} ({})",
                type(window).__name__,
                window.objectName() or "-",
                type(widget).__name__,
                bounds.getRect(),
                rect.getRect(),
                "已 mask 裁剪" if not window.mask().isEmpty() else "无 mask，Windows 上会合成失败",
            )


def _is_translucent(widget: QWidget) -> bool:
    """Return whether ``widget`` is a translucent top-level overlay window."""
    return widget.isWindow() and widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def main() -> int:
    """Install the diagnostics and launch the application."""
    log_file = _log_file()
    logger.add(str(log_file), level="DEBUG", encoding="utf-8")
    print(f"[诊断] 详细日志写入 {log_file}", flush=True)
    _report_environment()
    faulthandler.enable()
    _install_qt_message_handler()

    app = QApplication.instance() or QApplication(sys.argv)
    app.installEventFilter(_OverlayGuard(app))

    from NepTrainKit.main import main as run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
