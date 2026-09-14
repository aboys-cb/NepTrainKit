"""Keep the ``QStyle`` objects that Python creates alive for their widget.

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
can hold a style that no longer exists.  One instance per call is deliberate: Qt destroys
a widget's style together with the widget, so a style shared between widgets would be
released while its other users are still painting.  Retaining is also what keeps the
workaround small in practice -- Qt frees the styles of the widgets that are gone, and
``_drop_destroyed_styles`` forgets them on the way.
"""

from __future__ import annotations

import shiboken6
from PySide6.QtWidgets import QStyle, QStyleFactory

_retained_styles: list[QStyle] = []
_factory_create = QStyleFactory.create
_PRUNE_AT = 256


def keep_created_styles_alive() -> None:
    """Retain the styles that ``QStyleFactory.create`` hands to Python.

    Call this once, before the application builds its first menu; calling it again does
    nothing.  See the module docstring for the dangling style this prevents.
    """
    if QStyleFactory.create is _keep_alive:
        return
    QStyleFactory.create = staticmethod(_keep_alive)


def _keep_alive(name: str) -> QStyle:
    """Create a style and retain it until Qt destroys it with the widget using it."""
    style = _factory_create(name)
    if style is not None:
        _retained_styles.append(style)
        if len(_retained_styles) >= _PRUNE_AT:
            _drop_destroyed_styles()
    return style


def _drop_destroyed_styles() -> None:
    """Forget the styles Qt already destroyed, so they do not pile up."""
    _retained_styles[:] = [style for style in _retained_styles if shiboken6.isValid(style)]
