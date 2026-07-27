"""Structured user-facing errors for Make Dataset operations."""

from __future__ import annotations

from typing import Any


class CardOperationError(ValueError):
    """Carry a stable error code, translatable template, and format values."""

    def __init__(self, code: str, template: str, **values: Any):
        self.code = str(code)
        self.template = str(template)
        self.values = dict(values)
        super().__init__(self.template.format(**self.values))
