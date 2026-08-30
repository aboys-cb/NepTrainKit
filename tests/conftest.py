from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
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
    """Keep one process-wide QApplication alive for the full suite.

    A number of unittest-style Qt suites share this application. Destroying it
    between classes can make the Windows offscreen plugin exit non-zero after
    an otherwise successful test run.
    """
    global _TEST_QAPPLICATION
    _TEST_QAPPLICATION = QApplication.instance() or QApplication([])
    yield _TEST_QAPPLICATION
