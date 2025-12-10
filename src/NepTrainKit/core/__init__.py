#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Core domain models and utilities for NEP workflows.

This package exposes lightweight, lazily-loaded entry points for the rest of
NepTrainKit to avoid heavy imports at startup and keep UI responsiveness high.

Notes
-----
- Public symbols are exported via ``__getattr__`` to defer imports.
- Modules in this package cover structures, messaging, dataset IO, and helpers.


"""
# Lightweight, lazy exports to avoid heavy imports at startup.
from __future__ import annotations

from typing import Any

__all__ = [
    'MessageManager',
    'CardManager', 'load_cards_from_directory',
]

from .card_manager import CardManager, load_cards_from_directory
from .message import MessageManager

# 可选：通过环境变量 NEP_NATIVE_STDIO 控制原生 stdout/stderr 重定向
try:
    import os as _os
    _mode = _os.environ.get("NEP_NATIVE_STDIO", "").strip()
    if _mode:
        from .cstdio_redirect import redirect_c_stdout_stderr as _redir
        if _mode.lower() == "inherit":
            _guard = None
        elif _mode.lower() == "silent":
            _guard = _redir(None)
            _guard.__enter__()
        else:
            _guard = _redir(_mode)
            _guard.__enter__()
        _NEP_NATIVE_STDIO_GUARD = _guard
except Exception:
    _NEP_NATIVE_STDIO_GUARD = None

