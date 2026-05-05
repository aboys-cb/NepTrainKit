#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from PySide6.QtWidgets import QApplication, QWidget

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from NepTrainKit.core.io.base import NepPlotData
from NepTrainKit.core.types import Brushes, CanvasMode, Pens
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

    def test_pyqtgraph_canvas_apply_overlay_groups_recolors_points(self):
        canvas, fallback = canvas_factory.create_result_canvas(CanvasMode.PYQTGRAPH, None)
        self.assertFalse(fallback)

        dataset = NepPlotData(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ],
                dtype=np.float32,
            ),
            index_list=np.array([0, 1, 2], dtype=np.int32),
            title="descriptor",
        )
        dataset.display_title = "Training Overlay"
        dataset.x_label = "PC1"
        dataset.y_label = "PC2"
        dataset.parity_mode = False
        dataset.show_rmse = False
        dataset.base_brush = Brushes.TrainingOverlay
        dataset.base_pen = Pens.TrainingOverlay
        result_data = SimpleNamespace(datasets=[dataset], select_index=set(), reject_index=set())

        canvas.init_axes(1)
        canvas.set_nep_result_data(result_data)
        canvas.plot_nep_result()
        canvas.apply_overlay_groups([1, 2], [2])

        plot = canvas.axes_list[0]
        brushes = plot._scatter.data["brush"]
        self.assertEqual(brushes[0].color().rgba(), Brushes.TrainingOverlay.color().rgba())
        self.assertEqual(brushes[1].color().rgba(), Brushes.LoadedOverlay.color().rgba())
        self.assertEqual(brushes[2].color().rgba(), Brushes.Selected.color().rgba())

        canvas.apply_overlay_groups([], [])
        brushes = plot._scatter.data["brush"]
        self.assertEqual(brushes[0].color().rgba(), Brushes.TrainingOverlay.color().rgba())
        self.assertEqual(brushes[1].color().rgba(), Brushes.TrainingOverlay.color().rgba())
        self.assertEqual(brushes[2].color().rgba(), Brushes.TrainingOverlay.color().rgba())

    def test_vispy_set_view_layout_single_axis_uses_nonzero_col_span(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        plot = SimpleNamespace(_stretch=None, rmse_size=None)
        canvas.axes_list = [plot]
        canvas.current_axes = plot
        canvas.grid = SimpleNamespace(remove_widget=MagicMock(), add_widget=MagicMock())

        canvas.set_view_layout()

        canvas.grid.add_widget.assert_called_once_with(plot, row=0, col=0, row_span=6, col_span=1)

    def test_pyqtgraph_set_view_layout_single_axis_does_not_reserve_subplot_space(self):
        canvas = canvas_factory._create_pyqtgraph_result_canvas(None)
        plot = SimpleNamespace(rmse_size=None)
        canvas.axes_list = [plot]
        canvas.current_axes = plot
        canvas.ci = SimpleNamespace(clear=MagicMock(), layout=SimpleNamespace(setRowStretchFactor=MagicMock()))
        canvas.addItem = MagicMock()

        canvas.set_view_layout()

        canvas.addItem.assert_called_once_with(plot, row=0, col=0, colspan=1)
        canvas.ci.layout.setRowStretchFactor.assert_called_once_with(0, 1)
