#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Lightweight, lazy exports to avoid heavy imports at startup.
from __future__ import annotations

from typing import Any

__all__ = [
    'MessageManager',
    'Structure', 'process_organic_clusters', 'get_clusters',
    'CardManager', 'load_cards_from_directory',
]


def __getattr__(name: str) -> Any:
    if name == 'MessageManager':
        from .message import MessageManager as _M
        return _M
    if name in ('Structure', 'process_organic_clusters', 'get_clusters'):
        from .structure import Structure as _S, process_organic_clusters as _P, get_clusters as _G
        return {'Structure': _S, 'process_organic_clusters': _P, 'get_clusters': _G}[name]
    if name in ('CardManager', 'load_cards_from_directory'):
        from .card_manager import CardManager as _C, load_cards_from_directory as _L
        return {'CardManager': _C, 'load_cards_from_directory': _L}[name]
    raise AttributeError(name)
