from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

_TEST_STORAGE = tempfile.TemporaryDirectory(prefix="neptrainkit-tests-", ignore_cleanup_errors=True)
LOCALAPPDATA_PATH = Path(_TEST_STORAGE.name)
os.environ["LOCALAPPDATA"] = str(LOCALAPPDATA_PATH)
os.environ["NEPTRAINKIT_LIBRARY"] = str(LOCALAPPDATA_PATH / "library")
os.environ["NEPTRAINKIT_RUNTIME_ROOT"] = str(LOCALAPPDATA_PATH / "runtime")

# Patch before collecting modules: Config opens its database during import.
# Each xdist worker owns a different temporary directory.
import NepTrainKit.paths as _paths


def _test_config_path(*, create=True):
    path = LOCALAPPDATA_PATH / "config"
    return _paths.ensure_directory(path) if create else path


_paths.get_user_config_path = _test_config_path

# CI pins the headless plugin; a developer desktop does not. Without this the
# native platform clamps window sizes to the real screen, so a test that asks
# for a 1600 px canvas silently receives the display width and layout
# assertions fail depending on the machine. Keep local runs identical to CI
# unless a caller explicitly selects another platform (for example to debug
# rendering by hand).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ["QT_QPA_PLATFORM"] in {"offscreen", "minimal"}:
    from vispy import config as _vispy_config

    # Headless macOS reports a zero-sized screen to Vispy's physical DPI probe.
    _vispy_config["dpi"] = 96


_TEST_QAPPLICATION: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def _keep_qapplication_alive():
    """Own one process-wide QApplication for the full pytest session.

    PySide must release the native application before Python starts destroying
    module globals. On Windows the offscreen plugin can otherwise return exit
    code 1 after pytest has already printed a successful test summary.
    """
    global _TEST_QAPPLICATION
    _TEST_QAPPLICATION = QApplication.instance() or QApplication([])
    try:
        yield _TEST_QAPPLICATION
    finally:
        QApplication.closeAllWindows()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        _TEST_QAPPLICATION.processEvents()
        _TEST_QAPPLICATION.shutdown()
        _TEST_QAPPLICATION = None
