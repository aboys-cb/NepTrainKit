from __future__ import annotations

from pathlib import Path


def test_unittest_suites_do_not_quit_the_shared_qapplication():
    """Prevent Windows Qt teardown failures after a green pytest summary."""
    tests_root = Path(__file__).resolve().parent
    offenders = []
    for path in tests_root.rglob("*.py"):
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "cls._app.quit()" in text:
            offenders.append(str(path.relative_to(tests_root)))

    assert offenders == []


def test_session_fixture_explicitly_shuts_down_the_native_qapplication():
    """Keep PySide teardown out of late interpreter finalization on Windows."""
    conftest = Path(__file__).with_name("conftest.py").read_text(encoding="utf-8")

    assert "QApplication.closeAllWindows()" in conftest
    assert "QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)" in conftest
    assert "_TEST_QAPPLICATION.shutdown()" in conftest
