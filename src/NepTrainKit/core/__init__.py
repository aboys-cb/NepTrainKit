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

# Optional early native stdio redirection controlled by environment
# This helps silence C/C++ printf even during early imports on POSIX/WSL.
try:
    import os as _os
    _mode = _os.environ.get("NEP_NATIVE_STDIO", "").strip()
    if _mode:
        from .cstdio_redirect import redirect_c_stdout_stderr as _redir
        if _mode.lower() == "inherit":
            _guard = None  # no-op
        elif _mode.lower() == "silent":
            _guard = _redir(None)
            _guard.__enter__()
        else:
            # treat as filepath
            _guard = _redir(_mode)
            _guard.__enter__()
        # Keep a reference to avoid GC closing it
        _NEP_NATIVE_STDIO_GUARD = _guard
except Exception:
    # Fail-open: never block imports because of redirection issues
    _NEP_NATIVE_STDIO_GUARD = None

