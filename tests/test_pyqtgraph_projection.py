"""Projection contracts exercised against both pyqtgraph 0.13 and 0.14."""

import importlib
import inspect
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtGui import QVector3D
from pyqtgraph.opengl import GLViewWidget

from NepTrainKit.ui.canvas.pyqtgraph.structure import StructurePlotWidget
import NepTrainKit.ui.canvas.pyqtgraph.structure as structure_module


_EXPLICIT_VIEWPORT = "viewport" in inspect.signature(GLViewWidget.projectionMatrix).parameters


@pytest.fixture
def viewer():
    widget = StructurePlotWidget()
    widget.resize(640, 480)
    widget.opts.update(distance=10.0, fov=90.0)
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.mark.parametrize("ortho", [True, False])
def test_projection_switch_only_schedules_repaint(viewer, monkeypatch, ortho):
    projection = Mock(side_effect=AssertionError("GL state must only change during rendering"))
    update = Mock()
    monkeypatch.setattr(viewer, "setProjection", projection)
    monkeypatch.setattr(viewer, "update", update)
    viewer.set_projection(ortho)
    assert viewer.ortho is ortho
    projection.assert_not_called()
    update.assert_called_once_with()


@pytest.mark.parametrize("cropped", [False, True])
def test_perspective_uses_backend_projection(viewer, cropped):
    viewport = viewer.getViewport()
    region = (160, 120, 320, 240) if cropped else viewport
    args = (region, viewport) if _EXPLICIT_VIEWPORT else (region,)
    expected = GLViewWidget.projectionMatrix(viewer, *args)
    actual = viewer.projectionMatrix(*args)
    np.testing.assert_allclose(actual.data(), expected.data())


@pytest.mark.parametrize("cropped", [False, True])
def test_orthographic_projection_respects_viewport_and_crop(viewer, monkeypatch, cropped):
    viewer.ortho = True
    viewport = (20, 30, 640, 480)
    region = (180, 150, 320, 240) if cropped else viewport
    # In 0.14 the explicit viewport may differ from the widget, e.g. export.
    monkeypatch.setattr(viewer, "getViewport", lambda: (0, 0, 100, 100) if _EXPLICIT_VIEWPORT else viewport)
    args = (region, viewport) if _EXPLICIT_VIEWPORT else (region,)
    matrix = viewer.projectionMatrix(*args)
    half_width, half_height = (5.0, 3.75) if cropped else (10.0, 7.5)
    for sign in (-1, 1):
        point = matrix.map(QVector3D(sign * half_width, sign * half_height, -10))
        np.testing.assert_allclose([point.x(), point.y()], [sign, sign], atol=1e-6)


@pytest.mark.parametrize("ortho", [False, True])
def test_backend_paint_updates_its_projection_state(viewer, monkeypatch, ortho):
    backend = importlib.import_module("pyqtgraph.opengl.GLViewWidget")
    gl_api = backend.GL if _EXPLICIT_VIEWPORT else backend
    for name in ("glViewport", "glClearColor", "glClear", "glMatrixMode", "glLoadMatrixf"):
        call = Mock()
        monkeypatch.setattr(gl_api, name, call)
        if hasattr(structure_module, name):
            monkeypatch.setattr(structure_module, name, call)
    monkeypatch.setattr(viewer, "setModelview", Mock())
    draw = Mock()
    monkeypatch.setattr(viewer, "drawItemTree", draw)
    viewer.ortho = ortho
    # Exercise the installed backend's real dispatch, without a GPU context.
    viewer.paintGL()
    draw.assert_called_once_with(useItemNames=False)
    viewport = viewer.getViewport()
    if _EXPLICIT_VIEWPORT:
        assert len(viewer._projectionStack) == 1
        np.testing.assert_allclose(
            viewer.currentProjection().data(), viewer.projectionMatrix(viewport, viewport).data()
        )
    else:
        gl_api.glLoadMatrixf.assert_called_once()
        np.testing.assert_allclose(
            gl_api.glLoadMatrixf.call_args.args[0], viewer.projectionMatrix().data()
        )
