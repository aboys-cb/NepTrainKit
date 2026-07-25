from __future__ import annotations

import os
from pathlib import Path

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from PySide6.QtWidgets import QApplication

from NepTrainKit.config import Config
from NepTrainKit.logging_config import (
    DEFAULT_LOG_LEVEL,
    get_log_level,
    normalize_log_level,
    set_log_level,
)
from NepTrainKit.ui.pages.settings import SettingsWidget


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _restore_config(previous: object, previous_runtime: str) -> None:
    if previous is None:
        Config.delete("logging", "level")
    else:
        Config.set("logging", "level", previous)
    set_log_level(previous_runtime)


def test_log_level_normalization_defaults_to_info():
    assert normalize_log_level(None) == DEFAULT_LOG_LEVEL
    assert normalize_log_level("not-a-level") == DEFAULT_LOG_LEVEL
    assert normalize_log_level(" debug ") == "DEBUG"


def test_settings_widget_defaults_log_level_to_info():
    _app()
    previous = Config.get("logging", "level")
    previous_runtime = get_log_level()
    try:
        Config.delete("logging", "level")
        widget = SettingsWidget(None)
        assert widget.log_level_combo.currentData() == DEFAULT_LOG_LEVEL
    finally:
        _restore_config(previous, previous_runtime)


def test_settings_widget_loads_and_applies_configured_log_level():
    _app()
    previous = Config.get("logging", "level")
    previous_runtime = get_log_level()
    try:
        Config.set("logging", "level", "WARNING")
        set_log_level("WARNING")
        widget = SettingsWidget(None)
        assert widget.log_level_combo.currentData() == "WARNING"

        widget.log_level_combo.setCurrentIndex(
            widget.log_level_combo.findData("ERROR")
        )

        assert Config.get("logging", "level") == "ERROR"
        assert get_log_level() == "ERROR"
    finally:
        _restore_config(previous, previous_runtime)
