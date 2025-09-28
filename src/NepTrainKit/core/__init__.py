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

