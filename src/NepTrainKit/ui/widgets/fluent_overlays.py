"""Fluent overlays: translucent popups and tooltips.

Windows composites a translucent top-level window through
``UpdateLayeredWindowIndirect``, which rejects a dirty region reaching outside the
window: the call fails with ``ERROR_INVALID_PARAMETER`` ("参数错误") on every repaint
and the window stays black.  A drop shadow painted by a child of such a window reaches
outside by design, so on that platform the popups keep their translucent surface but
lose the shadow that outgrows them (``popup_shadows_allowed``, ``drop_popup_shadow``)
and are clipped to their own rect (``clip_popup_to_window``), which keeps the rounded
corners.

Tooltips always come from the fluent widget: ``install_fluent_tooltip`` fits a single
widget, and ``ensure_fluent_tooltips`` routes every remaining Qt tooltip -- plain
``setToolTip()`` widgets as well as item views -- through the same fluent tooltip, so
nothing falls back to the native tip, which ignores the app theme.  The same router
applies the policy above to the tooltips that ``ToolTipFilter`` builds for itself, so
no tip of the application reaches outside its own translucent window either.
"""

from __future__ import annotations

import sys

import shiboken6
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QAbstractItemView, QApplication, QTableView, QTreeView, QWidget
from qfluentwidgets import ToolTip, ToolTipFilter, ToolTipPosition
from qfluentwidgets.components.widgets.tool_tip import ItemViewToolTip, ItemViewToolTipType

_IS_WINDOWS = sys.platform == "win32"
_TOOLTIP_DURATION = 4000


def popup_shadows_allowed() -> bool:
    """Return whether a translucent popup may paint a drop shadow (see module docstring)."""
    return not _IS_WINDOWS


def drop_popup_shadow(owner: QWidget) -> None:
    """Remove the drop shadow of ``owner`` (see the module docstring)."""
    owner.setGraphicsEffect(None)


def clip_popup_to_window(window: QWidget) -> None:
    """Clip a popup to its own rect, so no child can paint outside of it.

    Only the overflowing part of a shadow is cut, which is what a layered window
    rejects on Windows; the translucent surface keeps its rounded corners.  Call
    this again whenever ``window`` is resized, because a Qt mask does not grow
    with the widget.
    """
    if not _IS_WINDOWS:
        return
    window.setMask(window.rect())


def keep_tooltip_inside_its_window(tooltip: ToolTip) -> None:
    """Apply the overlay policy above to one tooltip window (see module docstring).

    ``ToolTipFilter`` builds its own tooltip -- shadow included -- so every tip that
    this module did not create itself is routed through here as well.  The mask is
    refreshed on every resize, because a tooltip is resized to the text it shows.
    """
    if popup_shadows_allowed():
        return
    drop_popup_shadow(tooltip.container)
    clip_popup_to_window(tooltip)


def _new_tooltip(tooltip_type: type[ToolTip], text: str, parent: QWidget | None) -> ToolTip:
    tooltip = tooltip_type(text, parent)
    keep_tooltip_inside_its_window(tooltip)
    return tooltip


def create_fluent_tooltip(text: str, parent: QWidget | None = None) -> ToolTip:
    """Create a fluent tooltip, without the shadow that outgrows its window."""
    return _new_tooltip(ToolTip, text, parent)


class FluentToolTipFilter(ToolTipFilter):
    """Show one widget's tooltip with the fluent tooltip widget."""

    def _createToolTip(self) -> ToolTip:
        """Build the tooltip for the filtered widget."""
        parent = self.parent()
        return create_fluent_tooltip(parent.toolTip(), parent.window())


def install_fluent_tooltip(widget: QWidget, position: ToolTipPosition = ToolTipPosition.TOP) -> None:
    """Show ``widget``'s tooltip in the fluent style used by the rest of the app."""
    widget.installEventFilter(FluentToolTipFilter(widget, 300, position))


class _FluentToolTipRouter(QObject):
    """Show the tooltips of the whole application with the fluent widget.

    The router answers the widgets that own no fluent filter of their own and keeps
    every fluent tooltip -- whoever built it -- inside its own window.

    Creating the router installs it on ``app``.  Qt calls an event filter once per
    installation, so a router must never be installed twice.
    """

    def __init__(self, app: QApplication):
        super().__init__(app)
        self._tooltip: ToolTip | None = None
        self._owner: QWidget | None = None
        app.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Replace Qt's native tooltip by the fluent one."""
        event_type = event.type()
        if event_type == QEvent.Type.ToolTip:
            return self._show_tooltip(watched, event)
        if event_type in (QEvent.Type.Show, QEvent.Type.Resize):
            if isinstance(watched, ToolTip):
                keep_tooltip_inside_its_window(watched)
        elif event_type in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress) and watched is self._owner:
            self.hide_tooltip()
        return False

    def hide_tooltip(self) -> None:
        """Hide the tooltip this router is showing."""
        self._owner = None
        tooltip = self._tooltip
        if tooltip is not None and shiboken6.isValid(tooltip):
            tooltip.hide()

    def _show_tooltip(self, watched: QObject, event: QHelpEvent) -> bool:
        if not isinstance(watched, QWidget) or not watched.isEnabled():
            return False
        if self._owns_fluent_filter(watched):
            return False
        index = self._item_index(watched, event)
        text = self._tooltip_text(watched, index)
        if not text:
            return False
        if watched is not self._owner:
            self.hide_tooltip()
        tooltip = self._ensure_tooltip(watched, index is not None)
        tooltip.setText(text)
        if index is None:
            tooltip.adjustPos(watched, ToolTipPosition.TOP)
        else:
            view = watched.parent()
            tooltip.adjustPos(view, view.visualRect(index), self._item_tooltip_type(view))
        tooltip.show()
        self._owner = watched
        return True

    @staticmethod
    def _tooltip_text(widget: QWidget, index) -> str:
        text = widget.toolTip()
        if text:
            return text
        if index is not None:
            return str(index.data(Qt.ItemDataRole.ToolTipRole) or "")
        return ""

    def _ensure_tooltip(self, watched: QWidget, for_item: bool) -> ToolTip:
        """Return the tooltip to use, rebuilding it when its window is gone.

        The tooltip outlives nothing: it is a child window of the window it points
        at, so it is destroyed with that window and must not be reused afterwards.
        """
        window = watched.window()
        if self._tooltip is not None and shiboken6.isValid(self._tooltip):
            reusable = isinstance(self._tooltip, ItemViewToolTip) == for_item and self._tooltip.parent() is window
            if reusable:
                return self._tooltip
            self.hide_tooltip()
        self._tooltip = _new_tooltip(ItemViewToolTip if for_item else ToolTip, "", window)
        self._tooltip.setDuration(_TOOLTIP_DURATION)
        return self._tooltip

    @staticmethod
    def _owns_fluent_filter(widget: QWidget) -> bool:
        """Return whether ``widget`` already shows its tooltip itself."""
        return any(isinstance(child, ToolTipFilter) for child in widget.children())

    @staticmethod
    def _item_index(widget: QWidget, event: QHelpEvent):
        view = widget.parent()
        if isinstance(view, QAbstractItemView) and widget is view.viewport():
            index = view.indexAt(event.pos())
            return index if index.isValid() else None
        return None

    @staticmethod
    def _item_tooltip_type(view: QAbstractItemView) -> ItemViewToolTipType:
        if isinstance(view, (QTableView, QTreeView)):
            return ItemViewToolTipType.TABLE
        return ItemViewToolTipType.LIST


def ensure_fluent_tooltips(app: QApplication) -> _FluentToolTipRouter:
    """Install the application wide fluent tooltip router (idempotent)."""
    for child in app.children():
        if isinstance(child, _FluentToolTipRouter):
            return child
    return _FluentToolTipRouter(app)
