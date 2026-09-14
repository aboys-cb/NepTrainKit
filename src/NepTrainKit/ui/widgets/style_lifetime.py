"""Keep the ``QStyle`` objects that Python creates alive for the whole process.

qfluentwidgets styles every menu with ``self.setStyle(QStyleFactory.create("fusion"))``
(``RoundMenu.__initWidgets``) and does the same with the window-vista style in
``updateDynamicStyle``.  PySide6 owns whatever ``QStyleFactory.create`` returns, so such
a temporary is destroyed as soon as its statement ends, while the widget keeps a raw
pointer to it -- and Qt builds its ``QStyleSheetStyle`` proxy on that same pointer as
soon as the fluent style sheet is applied.  Painting the widget then calls through the
freed vtable: the completer popup of the structure filter bar took the whole application
down with an access violation on Windows, where the freed block is reused at once, and
crashed only intermittently on macOS.

``keep_created_styles_alive`` retains every style a Python caller creates, so no widget
can hold a style that no longer exists.  The callers above create a handful of these
objects (one per menu, one per dynamic style update); they stay small and are released
when the process exits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QStyle, QStyleFactory

_retained_styles: list[QStyle] = []
_create_style = QStyleFactory.create


def keep_created_styles_alive() -> None:
    """Retain the styles that ``QStyleFactory.create`` hands to Python.

    Call this once, before the application builds its first menu; calling it again does
    nothing.  See the module docstring for the dangling style this prevents.
    """
    if QStyleFactory.create is _keep_alive:
        return
    QStyleFactory.create = staticmethod(_keep_alive)


def _keep_alive(name: str) -> QStyle:
    """Create a style and keep a reference to it for the rest of the process."""
    style = _create_style(name)
    if style is not None:
        _retained_styles.append(style)
    return style
