#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base interface for distribution plot backends."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QWidget


class DistributionPlotBase:
    """Minimal plotting interface for distribution inspector backends."""

    def __init__(self) -> None:
        self._bin_click_callback: Callable[[int], None] | None = None

    def widget(self) -> QWidget:
        """Return the backend widget inserted into the dialog."""
        raise NotImplementedError

    def set_payload(self, metric: dict[str, Any] | None, series: dict[str, Any] | None) -> None:
        """Render the selected metric+series payload."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all rendered items."""
        raise NotImplementedError

    def set_bin_click_callback(self, callback: Callable[[int], None] | None) -> None:
        """Register callback for histogram-bin click events."""
        self._bin_click_callback = callback
