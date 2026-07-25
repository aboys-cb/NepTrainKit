"""Fail-closed startup checks that must run before importing the full GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Type

from sqlalchemy.exc import SQLAlchemyError

from NepTrainKit.paths import get_user_config_path


def _configuration_error_text(config_path: Path, error: Exception) -> str:
    details = f"{type(error).__name__}: {error}"
    return (
        "NepTrainKit 无法创建或打开配置目录：\n"
        f"{config_path}\n\n"
        "请确认其父目录存在、确实是目录且可写，然后重新启动 NepTrainKit。"
        "程序未使用临时配置继续运行。\n\n"
        "Configuration storage is unavailable:\n"
        f"{config_path}\n\n"
        "Check that the parent directory exists, is a directory, and is writable, "
        "then restart NepTrainKit. No temporary configuration was used.\n\n"
        f"Details: {details}"
    )


def _can_show_startup_dialog() -> bool:
    platform_name = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform_name in {"offscreen", "minimal"}:
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _report_configuration_error(error: Exception) -> None:
    config_path = get_user_config_path(create=False)
    message = _configuration_error_text(config_path, error)
    if sys.stderr is not None:
        print(message, file=sys.stderr)

    if not _can_show_startup_dialog():
        return

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        owns_app = app is None
        if app is None:
            app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "NepTrainKit 配置错误 / Configuration Error",
            message,
        )
        if owns_app:
            app.quit()
    except Exception as dialog_error:  # pragma: no cover - platform GUI failure
        if sys.stderr is not None:
            print(
                f"Unable to show the startup error dialog: "
                f"{type(dialog_error).__name__}: {dialog_error}",
                file=sys.stderr,
            )


def load_config_class() -> Type:
    """Load the persistent config class or terminate with an actionable error."""
    try:
        from NepTrainKit.config import Config
    except (OSError, SQLAlchemyError) as error:
        _report_configuration_error(error)
        raise SystemExit(2) from None
    return Config


__all__ = ["load_config_class"]
