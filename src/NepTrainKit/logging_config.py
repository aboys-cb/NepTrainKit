"""Central loguru configuration for console and packaged file logging."""
from __future__ import annotations

import sys
from typing import TextIO

from loguru import logger


DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_console_handler_id: int | None = None
_file_handler_id: int | None = None
_file_sink: str | None = None
_configured_level = DEFAULT_LOG_LEVEL
_initialized = False


def normalize_log_level(value: object) -> str:
    """Return one supported log level, falling back to ``INFO``."""
    level = str(value or "").strip().upper()
    return level if level in LOG_LEVELS else DEFAULT_LOG_LEVEL


def _remove_handler(handler_id: int | None) -> None:
    if handler_id is None:
        return
    try:
        logger.remove(handler_id)
    except ValueError:
        pass


def _install_handlers(level: str, console_sink: TextIO = sys.stderr) -> None:
    global _console_handler_id, _file_handler_id
    _remove_handler(_console_handler_id)
    _remove_handler(_file_handler_id)
    _console_handler_id = logger.add(console_sink, level=level)
    _file_handler_id = (
        logger.add(_file_sink, level=level) if _file_sink is not None else None
    )


def initialize_logging(
    level: object = DEFAULT_LOG_LEVEL,
    *,
    file_sink: str | None = None,
) -> str:
    """Replace loguru's default sink and initialize NepTrainKit-owned handlers."""
    global _file_sink, _configured_level, _initialized
    normalized = normalize_log_level(level)
    if not _initialized:
        try:
            logger.remove(0)
        except ValueError:
            pass
    _file_sink = file_sink
    _install_handlers(normalized)
    _configured_level = normalized
    _initialized = True
    return normalized


def set_log_level(level: object) -> str:
    """Apply a new minimum level immediately to all NepTrainKit log sinks."""
    global _configured_level
    normalized = normalize_log_level(level)
    if not _initialized:
        return initialize_logging(normalized)
    _install_handlers(normalized)
    _configured_level = normalized
    return normalized


def get_log_level() -> str:
    """Return the currently active NepTrainKit log level."""
    return _configured_level
