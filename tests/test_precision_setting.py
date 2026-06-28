#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from PySide6.QtWidgets import QApplication

from NepTrainKit.config import Config
from NepTrainKit.core.precision import get_storage_float_dtype, get_storage_precision
from NepTrainKit.core.types import DataPrecision, VispyThumbnailMode
from NepTrainKit.ui.pages.settings import SettingsWidget


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_storage_precision_defaults_to_float32():
    Config.delete("nep", "data_precision")

    assert get_storage_precision() == DataPrecision.FLOAT32
    assert get_storage_float_dtype() == np.dtype(np.float32)


def test_storage_precision_switches_to_float64():
    prev = Config.get("nep", "data_precision")
    try:
        Config.set("nep", "data_precision", DataPrecision.FLOAT64)
        assert get_storage_precision() == DataPrecision.FLOAT64
        assert get_storage_float_dtype() == np.dtype(np.float64)
    finally:
        if prev is None:
            Config.delete("nep", "data_precision")
        else:
            Config.set("nep", "data_precision", prev)


def test_settings_widget_shows_default_data_precision():
    _app()
    Config.delete("nep", "data_precision")

    widget = SettingsWidget(None)
    assert widget.data_precision_card.comboBox.currentText() == DataPrecision.FLOAT32.value


def test_settings_widget_shows_float64_when_configured():
    _app()
    prev = Config.get("nep", "data_precision")
    try:
        Config.set("nep", "data_precision", DataPrecision.FLOAT64)
        widget = SettingsWidget(None)
        assert widget.data_precision_card.comboBox.currentText() == DataPrecision.FLOAT64.value
    finally:
        if prev is None:
            Config.delete("nep", "data_precision")
        else:
            Config.set("nep", "data_precision", prev)


def test_settings_widget_shows_vispy_thumbnail_controls():
    _app()
    prev_mode = Config.get("widget", "vispy_thumbnail_mode")
    prev_limit = Config.get("widget", "vispy_thumbnail_point_limit")
    try:
        Config.set("widget", "vispy_thumbnail_mode", VispyThumbnailMode.SMART)
        Config.set("widget", "vispy_thumbnail_point_limit", 12345)
        widget = SettingsWidget(None)
        assert widget.vispy_thumbnail_mode_card.comboBox.currentText() == VispyThumbnailMode.SMART.value
        assert int(widget.vispy_thumbnail_limit_card.doubleSpinBox.value()) == 12345
    finally:
        if prev_mode is None:
            Config.delete("widget", "vispy_thumbnail_mode")
        else:
            Config.set("widget", "vispy_thumbnail_mode", prev_mode)
        if prev_limit is None:
            Config.delete("widget", "vispy_thumbnail_point_limit")
        else:
            Config.set("widget", "vispy_thumbnail_point_limit", prev_limit)
