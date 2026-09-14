"""Tests for the workaround that keeps Python-made ``QStyle`` objects alive."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QStyleFactory
from qfluentwidgets import LineEdit
from qfluentwidgets.components.widgets.line_edit import CompleterMenu

import NepTrainKit.main as main_module
from NepTrainKit.ui.widgets import style_lifetime
from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _run_script(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run ``script`` in its own interpreter, where a crash is an exit code."""
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT / "src")})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _assert_painted(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_created_styles_are_retained(qapp):
    keep_created_styles_alive()

    style = QStyleFactory.create("fusion")

    assert style is not None
    assert any(retained is style for retained in style_lifetime._retained_styles)


def test_each_call_answers_with_its_own_style(qapp):
    """Qt frees a style with the widget that uses it, so instances are never shared."""
    keep_created_styles_alive()

    first = QStyleFactory.create("fusion")
    second = QStyleFactory.create("fusion")

    assert first is not second


def test_installing_twice_keeps_the_same_factory(qapp):
    keep_created_styles_alive()
    installed = QStyleFactory.create

    keep_created_styles_alive()

    assert QStyleFactory.create is installed


def test_menu_style_is_retained_by_the_workaround(qapp, monkeypatch):
    """The style a menu sets is retained, so it outlives the statement that set it."""
    keep_created_styles_alive()
    asked = []
    retained_before = len(style_lifetime._retained_styles)
    create = QStyleFactory.create

    def record(name):
        asked.append(name)
        return create(name)

    monkeypatch.setattr(QStyleFactory, "create", staticmethod(record))

    menu = CompleterMenu(LineEdit())
    try:
        created = style_lifetime._retained_styles[retained_before:]
    finally:
        menu.close()
        menu.deleteLater()

    assert "fusion" in asked, "building a completer menu should ask the factory for its style"
    assert created, "the style the menu set should be retained"


def test_destroyed_styles_are_forgotten(qapp):
    """Styles Qt freed with their widget stop counting towards the retained set."""
    keep_created_styles_alive()
    doomed = QStyleFactory.create("fusion")
    alive = QStyleFactory.create("fusion")
    shiboken6.delete(doomed)

    style_lifetime._drop_destroyed_styles()

    assert all(retained is not doomed for retained in style_lifetime._retained_styles)
    assert any(retained is alive for retained in style_lifetime._retained_styles)


def test_dropped_style_survives_a_repaint(qapp):
    """The pattern qfluentwidgets uses: set a factory style, add a style sheet, drop it."""
    result = _run_script(
        """
        import gc

        from PySide6.QtWidgets import QApplication, QListWidget, QStyleFactory

        from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive

        app = QApplication([])
        keep_created_styles_alive()

        widget = QListWidget()
        widget.addItems(["surface", "bulk"])
        style = QStyleFactory.create("fusion")
        widget.setStyle(style)
        widget.setStyleSheet("QListWidget { background: #202020; }")
        del style
        gc.collect()

        for _ in range(3):
            widget.grab()

        print("OK")
        """
    )

    _assert_painted(result)


def test_completer_menu_paints_with_its_style_retained(qapp):
    """Regression: painting the suggestion popup crashed with an access violation."""
    result = _run_script(
        """
        from PySide6.QtWidgets import QApplication
        from qfluentwidgets import LineEdit
        from qfluentwidgets.components.widgets.line_edit import CompleterMenu

        from NepTrainKit.ui.widgets.completer import CompleterModel, JoinDelegate
        from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive

        app = QApplication([])
        keep_created_styles_alive()

        edit = LineEdit()
        menu = CompleterMenu(edit)
        menu.setCompletion(CompleterModel({"surface": 3, "bulk": 2}))
        menu.view.setItemDelegate(JoinDelegate(edit, {"surface": 3, "bulk": 2}))
        menu.resize(283, 71)
        menu.show()

        for _ in range(3):
            menu.grab()

        print("OK")
        """
    )

    _assert_painted(result)


def test_configure_app_keeps_created_styles_alive(qapp, monkeypatch):
    """The desktop entry point installs the workaround before the first menu."""
    monkeypatch.setattr(main_module, "set_light_theme", lambda app: None)
    monkeypatch.setattr(main_module, "install_translator", lambda app, language=None: "en")
    monkeypatch.setattr(main_module, "_set_macos_dock_icon", lambda app, icon: None)
    monkeypatch.setattr(qapp, "setStyleSheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(qapp, "setFont", lambda *args, **kwargs: None)

    main_module.configure_app(qapp)

    assert QStyleFactory.create is style_lifetime._keep_alive
