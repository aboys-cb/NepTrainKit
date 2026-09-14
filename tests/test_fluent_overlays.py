"""Tests for the fluent tooltip and translucent popup helpers."""

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QWidget
from qfluentwidgets import ToolTipFilter, ToolTipPosition
from qfluentwidgets.components.widgets.tool_tip import ItemViewToolTip

import NepTrainKit.main as main_module
from NepTrainKit.ui.widgets import fluent_overlays
from NepTrainKit.ui.widgets.fluent_overlays import (
    clip_popup_to_window,
    create_fluent_tooltip,
    ensure_fluent_tooltips,
    install_fluent_tooltip,
    popup_shadows_allowed,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def router(qapp):
    instance = ensure_fluent_tooltips(qapp)
    yield instance
    _remove_router(qapp)


def _remove_router(app):
    """Uninstall the application router so it reaches no later test."""
    for child in app.children():
        if isinstance(child, fluent_overlays._FluentToolTipRouter):
            child.hide_tooltip()
            app.removeEventFilter(child)
            child.setParent(None)
    app.processEvents()


def _hover(widget, position=None):
    point = position or QPoint(4, 4)
    QApplication.sendEvent(widget, QHelpEvent(QEvent.Type.ToolTip, point, widget.mapToGlobal(point)))
    QApplication.instance().processEvents()


def _hoverable_widget(tooltip: str) -> QWidget:
    widget = QWidget()
    widget.setToolTip(tooltip)
    widget.resize(40, 20)
    widget.show()
    return widget


def _drop(widget: QWidget) -> None:
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.instance().processEvents()


def test_windows_popups_drop_the_shadow_that_outgrows_the_window(monkeypatch):
    monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", True)
    assert not popup_shadows_allowed()
    assert create_fluent_tooltip("hint").container.graphicsEffect() is None
    monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", False)
    assert popup_shadows_allowed()
    assert create_fluent_tooltip("hint").container.graphicsEffect() is not None


def test_popup_is_clipped_to_its_window_on_windows_only(monkeypatch):
    widget = QWidget()
    try:
        monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", False)
        widget.resize(90, 40)
        clip_popup_to_window(widget)
        assert widget.mask().isEmpty()

        monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", True)
        clip_popup_to_window(widget)
        assert widget.mask().boundingRect() == widget.rect()
    finally:
        widget.deleteLater()


def test_router_answers_plain_widget_tooltips_with_the_fluent_tip(router, qapp):
    widget = _hoverable_widget("plain hint")
    try:
        _hover(widget)
        assert router._tooltip is not None
        assert router._tooltip.isVisible()
        assert router._tooltip.text() == "plain hint"
    finally:
        _drop(widget)


def test_router_leaves_widgets_with_their_own_fluent_filter_alone(router, qapp):
    widget = _hoverable_widget("filtered hint")
    install_fluent_tooltip(widget)
    try:
        _hover(widget)
        assert router._tooltip is None
    finally:
        _drop(widget)


def test_router_shows_item_view_tooltips(router, qapp):
    view = QListWidget()
    item = QListWidgetItem("row")
    item.setToolTip("row hint")
    view.addItem(item)
    view.resize(140, 60)
    view.show()
    try:
        position = view.visualItemRect(item).center()
        _hover(view.viewport(), position)
        assert isinstance(router._tooltip, ItemViewToolTip)
        assert router._tooltip.isVisible()
        assert router._tooltip.text() == "row hint"
    finally:
        _drop(view)


def test_router_ignores_tooltip_less_widgets(router, qapp):
    widget = QWidget()
    widget.resize(40, 20)
    widget.show()
    try:
        _hover(widget)
        assert router._tooltip is None
    finally:
        _drop(widget)


def test_router_rebuilds_its_tooltip_when_its_window_dies(router, qapp):
    """The tooltip is a child window of the window it points at."""
    dead_window = _hoverable_widget("first hint")
    _hover(dead_window)
    dead_tooltip = router._tooltip
    assert dead_tooltip is not None
    _drop(dead_window)
    assert not shiboken6.isValid(dead_tooltip)

    router.hide_tooltip()

    live_window = _hoverable_widget("second hint")
    try:
        _hover(live_window)
        assert router._tooltip is not None
        assert router._tooltip is not dead_tooltip
        assert router._tooltip.text() == "second hint"
    finally:
        _drop(live_window)


def test_router_keeps_a_foreign_fluent_tooltip_inside_its_window(router, qapp, monkeypatch):
    """A tooltip that ``ToolTipFilter`` built for itself must not overhang either."""
    monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", True)
    widget = _hoverable_widget("foreign hint")
    tooltip_filter = ToolTipFilter(widget, 0, ToolTipPosition.TOP)
    widget.installEventFilter(tooltip_filter)
    try:
        QApplication.sendEvent(widget, QEvent(QEvent.Type.Enter))
        qapp.processEvents()
        tooltip = tooltip_filter._tooltip
        assert tooltip is not None and tooltip.isVisible()
        assert tooltip.text() == "foreign hint"
        assert tooltip.container.graphicsEffect() is None
        assert tooltip.mask().boundingRect() == tooltip.rect()
    finally:
        tooltip_filter.hideToolTip()
        _drop(widget)


def test_router_is_installed_once_per_application(qapp):
    first = ensure_fluent_tooltips(qapp)
    second = ensure_fluent_tooltips(qapp)
    try:
        assert first is second
    finally:
        _remove_router(qapp)


def test_configure_app_answers_the_tooltips_of_the_whole_application(qapp, monkeypatch):
    """The desktop entry point installs the router, whatever created the widget."""
    # Do not re-theme the application shared by the rest of the test session.
    monkeypatch.setattr(main_module, "set_light_theme", lambda app: None)
    monkeypatch.setattr(main_module, "install_translator", lambda app, language=None: "en")
    monkeypatch.setattr(main_module, "_set_macos_dock_icon", lambda app, icon: None)
    monkeypatch.setattr(qapp, "setStyleSheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(qapp, "setFont", lambda *args, **kwargs: None)

    main_module.configure_app(qapp)

    widget = _hoverable_widget("entry point hint")
    try:
        _hover(widget)
        router = ensure_fluent_tooltips(qapp)
        assert router._tooltip is not None
        assert router._tooltip.isVisible()
        assert router._tooltip.text() == "entry point hint"
    finally:
        _drop(widget)
        _remove_router(qapp)
