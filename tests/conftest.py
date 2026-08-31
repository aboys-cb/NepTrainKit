from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

LOCALAPPDATA_PATH = Path(__file__).resolve().parent / "_localappdata"
LOCALAPPDATA_PATH.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(LOCALAPPDATA_PATH)


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
