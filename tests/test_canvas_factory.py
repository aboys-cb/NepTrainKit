#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QWidget

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from NepTrainKit.core.types import CanvasMode
import NepTrainKit.ui.canvas.canvas_factory as canvas_factory


class _ArrowCapable:
    def show_arrow(self, *_args, **_kwargs):
        return None

    def clear_arrow(self):
        return None


class _ArrowMissing:
    def show_arrow(self, *_args, **_kwargs):
        return None


class TestCanvasFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_create_result_canvas_pyqtgraph_default(self):
        canvas, fallback = canvas_factory.create_result_canvas(CanvasMode.PYQTGRAPH, None)
        self.assertEqual(type(canvas).__name__, "PyqtgraphCanvas")
        self.assertFalse(fallback)

    def test_create_result_canvas_unknown_defaults_to_pyqtgraph(self):
        canvas, fallback = canvas_factory.create_result_canvas("unknown-backend", None)
        self.assertEqual(type(canvas).__name__, "PyqtgraphCanvas")
        self.assertFalse(fallback)

    def test_create_result_canvas_vispy_failure_falls_back_to_pyqtgraph(self):
        sentinel = object()
        with patch.object(canvas_factory, "_create_vispy_result_canvas", side_effect=RuntimeError("boom")), patch.object(
            canvas_factory, "_create_pyqtgraph_result_canvas", return_value=sentinel
        ):
            canvas, fallback = canvas_factory.create_result_canvas(CanvasMode.VISPY, None)
        self.assertIs(canvas, sentinel)
        self.assertTrue(fallback)

    def test_create_structure_plot_vispy_failure_falls_back_to_pyqtgraph(self):
        sentinel = object()
        with patch.object(canvas_factory, "_create_vispy_structure_plot", side_effect=RuntimeError("boom")), patch.object(
            canvas_factory, "_create_pyqtgraph_structure_plot", return_value=sentinel
        ):
            canvas, fallback = canvas_factory.create_structure_plot(CanvasMode.VISPY, None)
        self.assertIs(canvas, sentinel)
        self.assertTrue(fallback)

    def test_resolve_canvas_host_widget_prefers_native(self):
        plain = QWidget()
        self.assertIs(canvas_factory.resolve_canvas_host_widget(plain), plain)

        native = QWidget()
        wrapped = SimpleNamespace(native=native)
        self.assertIs(canvas_factory.resolve_canvas_host_widget(wrapped), native)

    def test_supports_structure_arrows_detects_runtime_methods(self):
        self.assertTrue(canvas_factory.supports_structure_arrows(_ArrowCapable()))
        self.assertFalse(canvas_factory.supports_structure_arrows(_ArrowMissing()))
