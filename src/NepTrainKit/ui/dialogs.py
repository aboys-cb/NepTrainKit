#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog

from NepTrainKit.config import Config
from NepTrainKit.paths import check_path_type


def call_path_dialog(
    self,
    title: str,
    dialog_type: str = "file",
    default_filename: str = "",
    file_filter: str = "",
    selected_filter: str = "",
) -> Any:
    """Invoke a Qt file dialog and persist the chosen directory in config.

    Returns the selected path string or None.
    """
    base_path = Config.get_path()
    dialog_map = {
        "file": lambda: QFileDialog.getSaveFileName(
            self,
            title,
            str(base_path / default_filename) if default_filename else str(base_path),
            file_filter,
            selected_filter,
        ),
        "select": lambda: QFileDialog.getOpenFileName(self, title, str(base_path), file_filter),
        "selects": lambda: QFileDialog.getOpenFileNames(self, title, str(base_path), file_filter),
        "directory": lambda: QFileDialog.getExistingDirectory(self, title, str(base_path)),
    }

    dialog_func = dialog_map.get(dialog_type)
    if not dialog_func:
        return None

    select_path = dialog_func()

    if isinstance(select_path, tuple):
        select_path = select_path[0]
    elif isinstance(select_path, list):
        if not select_path:
            return None
        select_path = select_path[0]

    if not select_path:
        return None

    selected = Path(select_path) if isinstance(select_path, str) else  Path(select_path[0])
    last_dir = selected.parent if check_path_type(selected) == "file" else selected
    Config.set("setting", "last_path", str(last_dir))
    return select_path


__all__ = [
    'call_path_dialog',
]

