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

from NepTrainKit.config import Config
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

    def test_create_result_canvas_auto_defaults_to_pyqtgraph(self):
        canvas, fallback = canvas_factory.create_result_canvas(CanvasMode.AUTO, None)
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

    def test_vispy_init_axes_keeps_thumbnails_compact_until_selected(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)

        canvas.init_axes(3)

        self.assertIsNotNone(canvas.axes_list[0].xaxis)
        self.assertIsNotNone(canvas.axes_list[1].xaxis)
        self.assertIsNotNone(canvas.axes_list[2].xaxis)
        self.assertTrue(canvas.axes_list[1].xaxis.visible)
        self.assertFalse(canvas.axes_list[1].text.visible)

        canvas.set_current_axes(canvas.axes_list[2])
        canvas.set_view_layout()

        self.assertIsNotNone(canvas.axes_list[2].xaxis)
        self.assertTrue(canvas.axes_list[0].xaxis.visible)
        self.assertTrue(canvas.axes_list[0].text.visible)
        self.assertFalse(canvas.axes_list[2].text.visible)

    def test_vispy_layout_switch_keeps_numeric_grid_stretch(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)

        canvas.init_axes(3)
        canvas.grid._recreate_solver()
        canvas.set_current_axes(canvas.axes_list[1])
        canvas.grid._recreate_solver()

        for plot in canvas.axes_list:
            self.assertTrue(all(value is not None for value in plot.stretch))

    def test_vispy_compact_thumbnail_keeps_axes_and_diagonal(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)

        canvas.init_axes(2)
        main_plot = canvas.axes_list[0]
        thumbnail = canvas.axes_list[1]
        main_plot.title = "energy"
        main_plot.scatter(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
        )
        thumbnail.title = "energy"
        thumbnail.scatter(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
        )

        self.assertIsNotNone(main_plot.xaxis)
        self.assertIsNotNone(thumbnail.xaxis)
        self.assertTrue(thumbnail.xaxis.visible)
        self.assertFalse(thumbnail.text.visible)
        self.assertIsNotNone(thumbnail._diagonal)

        canvas.set_current_axes(thumbnail)
        canvas.set_view_layout()

        self.assertFalse(thumbnail.text.visible)
        self.assertTrue(main_plot.text.visible)

    def test_vispy_point_at_uses_cpu_pick_without_render(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        canvas.nep_result_data = object()
        plot = canvas.axes_list[0]
        plot.scatter(
            np.array([1.02, 3.0], dtype=np.float32),
            np.array([1.01, 3.0], dtype=np.float32),
            np.array([10, 20], dtype=np.int32),
        )

        def map_pos(_axes, pos):
            if pos == (16, 10):
                return 1.2, 1.0
            if pos == (10, 16):
                return 1.0, 1.2
            return 1.0, 1.0

        with patch.object(canvas, "_get_clicked_axes", return_value=plot), patch.object(
            canvas, "_canvas_to_data_pos", side_effect=map_pos
        ), patch.object(canvas, "render", side_effect=AssertionError("render should not be used")):
            self.assertEqual(canvas.point_at((10, 10)), 0)

    def test_vispy_structure_at_uses_full_dataset_not_rendered_sample(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        plot.scatter(
            np.array([1.02], dtype=np.float32),
            np.array([1.01], dtype=np.float32),
            np.array([10], dtype=np.int32),
        )
        dataset = SimpleNamespace(
            x=np.array([1.02, 2.02], dtype=np.float32),
            y=np.array([1.01, 2.01], dtype=np.float32),
            structure_index=np.array([10, 20], dtype=np.int32),
        )
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset])

        def map_pos(_axes, pos):
            if pos == (16, 10):
                return 2.2, 2.0
            if pos == (10, 16):
                return 2.0, 2.2
            return 2.0, 2.0

        with patch.object(canvas, "_get_clicked_axes", return_value=plot), patch.object(
            canvas, "_canvas_to_data_pos", side_effect=map_pos
        ), patch.object(canvas, "render", side_effect=AssertionError("render should not be used")):
            self.assertEqual(canvas.point_at((10, 10)), 1)
            self.assertEqual(canvas.structure_at((10, 10)), 20)

    def test_vispy_polygon_selection_uses_full_dataset_not_rendered_sample(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        canvas.current_axes = plot
        plot.scatter(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([10], dtype=np.int32),
        )
        dataset = SimpleNamespace(
            x=np.array([0.0, 2.0, 3.0], dtype=np.float32),
            y=np.array([0.0, 2.0, 3.0], dtype=np.float32),
            structure_index=np.array([10, 20, 30], dtype=np.int32),
        )
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset])
        canvas.select_index = MagicMock()

        canvas.select_point_from_polygon(
            np.array([[1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]], dtype=np.float32),
            False,
        )

        canvas.select_index.assert_called_once_with([20], False)

    def test_vispy_reuses_scatter_layer_when_dataset_version_is_unchanged(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            x=np.arange(10, dtype=np.float32),
            y=np.arange(10, dtype=np.float32),
            structure_index=np.arange(10, dtype=np.int32),
            all_data=np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)]),
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(
                version=0,
                num=10,
                all_data=np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)]),
                mask_array=np.ones(10, dtype=bool),
            ),
            group_array=SimpleNamespace(version=0, num=10, all_data=np.arange(10, dtype=np.int32)),
        )

        canvas._plot_dataset_on_axes(plot, dataset, True)
        scatter = plot._scatter_layers[canvas._dataset_layer_key(dataset)]

        with patch.object(scatter, "set_data", side_effect=AssertionError("scatter VBO should be reused")):
            canvas._plot_dataset_on_axes(plot, dataset, True)

    def test_vispy_reuse_path_does_not_recompute_plot_arrays(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            x=np.arange(10, dtype=np.float32),
            y=np.arange(10, dtype=np.float32),
            structure_index=np.arange(10, dtype=np.int32),
            all_data=np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)]),
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(
                version=0,
                num=10,
                all_data=np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)]),
                mask_array=np.ones(10, dtype=bool),
            ),
            group_array=SimpleNamespace(version=0, num=10, all_data=np.arange(10, dtype=np.int32)),
        )

        canvas._plot_dataset_on_axes(plot, dataset, True)

        with patch.object(canvas, "_full_plot_arrays", side_effect=AssertionError("plot arrays should be cached")):
            canvas._plot_dataset_on_axes(plot, dataset, True)

    def test_vispy_base_scatter_uses_fast_scatter(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        canvas.axes_list[0].scatter(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0], dtype=np.int32),
        )

        self.assertEqual(type(canvas.axes_list[0]._scatter).__name__, "FastScatter")

    def test_fast_scatter_keeps_edge_color_when_face_is_transparent(self):
        from NepTrainKit.ui.canvas.vispy.fast_scatter import FastScatterVisual

        visual = FastScatterVisual()
        visual.set_data(
            np.array([[0.0, 0.0]], dtype=np.float32),
            face_color=(1.0, 1.0, 1.0, 0.0),
            edge_color=(0.0, 0.25, 0.75, 1.0),
            edge_width=0.8,
        )

        self.assertEqual(visual._face_color, (1.0, 1.0, 1.0, 0.0))
        self.assertEqual(visual._edge_color, (0.0, 0.25, 0.75, 1.0))
        self.assertEqual(visual._edge_width, 0.8)

    def test_vispy_active_mask_updates_index_buffer_without_reuploading_positions(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.arange(6, dtype=np.float32), np.arange(6, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            x=np.arange(6, dtype=np.float32),
            y=np.arange(6, dtype=np.float32),
            structure_index=np.arange(6, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=6, all_data=all_data, mask_array=np.ones(6, dtype=bool)),
            group_array=SimpleNamespace(version=0, num=6, all_data=np.arange(6, dtype=np.int32)),
        )

        canvas._plot_dataset_on_axes(plot, dataset, True)
        scatter = plot._scatter_layers[canvas._dataset_layer_key(dataset)]
        dataset.data.mask_array = np.array([True, False, True, False, True, True], dtype=bool)
        dataset.data.version = 1
        dataset.data.num = 4
        dataset.group_array.version = 1
        dataset.group_array.num = 4

        with patch.object(scatter, "set_data", side_effect=AssertionError("positions should stay on GPU")), patch.object(
            scatter, "set_indices", wraps=scatter.set_indices
        ) as set_indices:
            canvas._plot_dataset_on_axes(plot, dataset, True)

        set_indices.assert_called_once()
        np.testing.assert_array_equal(set_indices.call_args.args[0], np.array([0, 2, 4, 5], dtype=np.uint32))

    def test_vispy_active_point_indices_repeat_force_components(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        all_data = np.zeros((4, 6), dtype=np.float32)
        dataset = SimpleNamespace(
            all_data=all_data,
            data=SimpleNamespace(mask_array=np.array([True, False, True, False], dtype=bool)),
        )

        np.testing.assert_array_equal(
            canvas._active_point_indices(dataset),
            np.array([0, 1, 2, 6, 7, 8], dtype=np.uint32),
        )

    def test_vispy_plot_coord_version_reuploads_positions(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            x=np.arange(4, dtype=np.float32),
            y=np.arange(4, dtype=np.float32),
            structure_index=np.arange(4, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=4, all_data=all_data, mask_array=np.ones(4, dtype=bool)),
            group_array=SimpleNamespace(version=0, num=4, all_data=np.arange(4, dtype=np.int32)),
            _plot_coord_version=0,
        )

        canvas._plot_dataset_on_axes(plot, dataset, True)
        scatter = plot._scatter_layers[canvas._dataset_layer_key(dataset)]
        dataset.all_data[0, 1] = 100.0
        dataset._plot_coord_version = 1

        with patch.object(scatter, "set_data", wraps=scatter.set_data) as set_data:
            canvas._plot_dataset_on_axes(plot, dataset, True)

        set_data.assert_called_once()

    def test_vispy_preview_uses_image_and_main_reuses_dataset_layer(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            x=np.arange(10, dtype=np.float32),
            y=np.arange(10, dtype=np.float32),
            structure_index=np.arange(10, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=10, all_data=all_data, mask_array=np.ones(10, dtype=bool)),
            group_array=SimpleNamespace(version=0, num=10, all_data=np.arange(10, dtype=np.int32)),
        )

        canvas._plot_dataset_on_axes(plot, dataset, True)
        scatter = plot._scatter_layers[canvas._dataset_layer_key(dataset)]
        canvas._plot_dataset_on_axes(plot, dataset, False)
        self.assertIsNotNone(plot._preview_image)
        self.assertIsNone(plot._scatter)

        with patch.object(scatter, "set_data", side_effect=AssertionError("positions should stay on GPU")):
            canvas._plot_dataset_on_axes(plot, dataset, True)

    def test_vispy_preview_uses_pen_color_when_face_is_transparent(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.linspace(0, 1, 10, dtype=np.float32), np.linspace(0, 1, 10, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            show_rmse=False,
            x=all_data[:, 1],
            y=all_data[:, 0],
            structure_index=np.arange(10, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=10, all_data=all_data, mask_array=np.ones(10, dtype=bool)),
            group_array=SimpleNamespace(
                version=0,
                num=10,
                all_data=np.arange(10, dtype=np.int32),
                now_data=np.arange(10, dtype=np.int32),
            ),
        )

        canvas._render_plot(plot, dataset, False)

        image = plot._preview_image._data
        visible = image[:, :, 3] > 0
        self.assertTrue(np.any(visible))
        self.assertGreater(np.max(image[:, :, 2][visible]), np.max(image[:, :, 0][visible]))

    def test_vispy_preview_reuses_cached_image_without_set_data(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.linspace(0, 1, 10, dtype=np.float32), np.linspace(0, 1, 10, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="energy",
            display_title="energy",
            parity_mode=True,
            show_rmse=False,
            x=all_data[:, 1],
            y=all_data[:, 0],
            structure_index=np.arange(10, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=10, all_data=all_data, mask_array=np.ones(10, dtype=bool)),
            group_array=SimpleNamespace(
                version=0,
                num=10,
                all_data=np.arange(10, dtype=np.int32),
                now_data=np.arange(10, dtype=np.int32),
            ),
        )

        canvas._render_plot(plot, dataset, False)

        with patch.object(plot._preview_image, "set_data", side_effect=AssertionError("cached preview image should be reused")):
            canvas._render_plot(plot, dataset, False)

    def test_vispy_preview_does_not_compute_rmse(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        all_data = np.column_stack([np.linspace(0, 1, 10, dtype=np.float32), np.linspace(0, 1, 10, dtype=np.float32)])
        dataset = SimpleNamespace(
            title="force",
            display_title="force",
            parity_mode=True,
            show_rmse=True,
            get_formart_rmse=MagicMock(side_effect=AssertionError("preview should not compute RMSE")),
            x=all_data[:, 1],
            y=all_data[:, 0],
            structure_index=np.arange(10, dtype=np.int32),
            all_data=all_data,
            x_cols=slice(1, None),
            y_cols=slice(None, 1),
            data=SimpleNamespace(version=0, num=10, all_data=all_data, mask_array=np.ones(10, dtype=bool)),
            group_array=SimpleNamespace(
                version=0,
                num=10,
                all_data=np.arange(10, dtype=np.int32),
                now_data=np.arange(10, dtype=np.int32),
            ),
        )

        canvas._render_plot(plot, dataset, False)

    def test_vispy_preview_current_marker_is_scaled_down(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(2)
        main_plot = canvas.axes_list[0]
        preview_plot = canvas.axes_list[1]

        main_plot.set_current_point(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
        preview_plot.set_current_point(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))

        self.assertLess(preview_plot.current_point._data["a_size"][0], main_plot.current_point._data["a_size"][0])

    def test_vispy_switch_view_box_swaps_dataset_mapping_without_layout_move(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(2)
        datasets = []
        for offset, title in enumerate(("energy", "force")):
            base = np.arange(4, dtype=np.float32) + offset * 100
            all_data = np.column_stack([base, base + 10])
            datasets.append(
                SimpleNamespace(
                    title=title,
                    display_title=title,
                    parity_mode=True,
                    show_rmse=False,
                    x=all_data[:, 1],
                    y=all_data[:, 0],
                    structure_index=np.arange(4, dtype=np.int32),
                    all_data=all_data,
                    x_cols=slice(1, None),
                    y_cols=slice(None, 1),
                    convert_index=lambda index: np.array([int(index)], dtype=np.int64),
                    is_visible=lambda array_index: np.asarray(array_index).size > 0,
                    data=SimpleNamespace(version=0, num=4, all_data=all_data, mask_array=np.ones(4, dtype=bool)),
                    group_array=SimpleNamespace(
                        version=0,
                        num=4,
                        all_data=np.arange(4, dtype=np.int32),
                        now_data=np.arange(4, dtype=np.int32),
                    ),
                )
            )
        canvas.nep_result_data = SimpleNamespace(datasets=datasets, select_index=set(), reject_index=set())
        canvas._ensure_plot_dataset_indices()
        canvas.plot_nep_result()
        canvas.plot_current_point(2)

        with patch.object(canvas, "_get_clicked_axes", return_value=canvas.axes_list[1]):
            canvas.switch_view_box(SimpleNamespace(pos=(10, 10)))

        self.assertIs(canvas.current_axes, canvas.axes_list[0])
        self.assertEqual(canvas._plot_dataset_indices, [1, 0])
        self.assertEqual(canvas.get_axes_dataset(canvas.axes_list[0]).title, "force")
        self.assertEqual(canvas.get_axes_dataset(canvas.axes_list[1]).title, "energy")
        self.assertIsNotNone(canvas.axes_list[1]._preview_image)
        np.testing.assert_allclose(canvas.axes_list[0].current_point._data["a_position"][:, :2], [[112.0, 102.0]])
        np.testing.assert_allclose(canvas.axes_list[1].current_point._data["a_position"][:, :2], [[12.0, 2.0]])

        canvas.plot_nep_result()

        np.testing.assert_allclose(canvas.axes_list[0].current_point._data["a_position"][:, :2], [[112.0, 102.0]])
        np.testing.assert_allclose(canvas.axes_list[1].current_point._data["a_position"][:, :2], [[12.0, 2.0]])

    def test_vispy_lasso_overlay_avoids_scene_line_redraw_in_off_mode(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        canvas.current_axes = canvas.axes_list[0]
        canvas.draw_mode = True

        class Transform:
            def map(self, pos):
                return float(pos[0]), float(pos[1]), 0.0, 1.0

        press = SimpleNamespace(button=1, pos=(10, 10))
        move_1 = SimpleNamespace(button=1, pos=(20, 20))
        move_2 = SimpleNamespace(button=1, pos=(30, 20))
        release = SimpleNamespace(button=1, pos=(30, 10))
        with patch.object(canvas.scene, "node_transform", return_value=Transform()), patch.object(
            canvas.current_axes.view, "add", side_effect=AssertionError("lasso should not enter VisPy scene")
        ), patch.object(canvas._lasso_overlay, "setGeometry", side_effect=AssertionError("lasso geometry should be stable")), patch.object(
            canvas._lasso_overlay, "show", side_effect=AssertionError("lasso overlay should stay mounted")
        ), patch.object(canvas._lasso_overlay, "hide", side_effect=AssertionError("lasso overlay should stay mounted")), patch.object(
            canvas._lasso_overlay, "raise_", side_effect=AssertionError("lasso overlay should not change stacking during drag")
        ), patch.object(canvas, "select_point_from_polygon") as select_polygon:
            canvas.on_mouse_press(press)
            canvas.on_mouse_move(move_1)
            canvas.on_mouse_move(move_2)
            canvas.on_mouse_release(release)

        select_polygon.assert_called_once()
        self.assertFalse(canvas._lasso_overlay.isHidden())
        self.assertEqual(canvas._lasso_overlay._points, [])

    def test_vispy_update_scatter_color_uses_overlay_position_cache(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        x = np.arange(6, dtype=np.float32)
        y = x + 1
        structure_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        dataset = SimpleNamespace(
            title="energy",
            x=x,
            y=y,
            structure_index=structure_index,
        )
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset], select_index=set())
        plot.scatter(x, y, structure_index)

        with patch("NepTrainKit.ui.canvas.vispy.canvas.np.isin", side_effect=AssertionError("np.isin should not be used")):
            canvas.update_scatter_color([1], Brushes.Selected)

        overlay = plot._overlays["selected"].positions
        np.testing.assert_array_equal(overlay[:, :2], np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float32))

    def test_vispy_reject_highlight_uses_overlay_position_cache(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        plot = canvas.axes_list[0]
        x = np.arange(6, dtype=np.float32)
        y = x + 1
        structure_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        dataset = SimpleNamespace(
            title="energy",
            x=x,
            y=y,
            structure_index=structure_index,
        )
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset], select_index=set())
        plot.scatter(x, y, structure_index)

        with patch("NepTrainKit.ui.canvas.vispy.canvas.np.isin", side_effect=AssertionError("np.isin should not be used")):
            canvas.set_reject_highlight([1], True)

        overlay = plot._overlays["reject"]._data["a_position"]
        np.testing.assert_array_equal(overlay[:, :2], np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float32))

    def test_vispy_plot_nep_result_prewarms_overlay_position_cache(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        dataset = NepPlotData(
            np.array(
                [
                    [0.0, 0.1],
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                ],
                dtype=np.float32,
            ),
            index_list=np.arange(4, dtype=np.int32),
            title="energy",
        )
        dataset.show_rmse = False
        dataset.x_label = "x"
        dataset.y_label = "y"
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset], select_index=set(), reject_index=set())

        canvas.plot_nep_result()

        signature = canvas._overlay_cache_signature(dataset)
        self.assertIn(signature, canvas._overlay_position_cache)
        self.assertIn("selected", canvas.axes_list[0]._overlays)
        self.assertFalse(canvas.axes_list[0]._overlays["selected"].visible)

    def test_vispy_overlay_cache_handles_atom_level_mforce_components(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        row_groups = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
        cols = 3
        x = np.arange(row_groups.size * cols, dtype=np.float32)
        y = x + 100
        dataset = SimpleNamespace(
            title="mforce",
            x=x,
            y=y,
            cols=cols,
            group_array=SimpleNamespace(now_data=row_groups, all_data=row_groups),
        )

        pos = canvas._overlay_positions_for_indices(dataset, {2})

        expected_x = np.arange(9, 18, dtype=np.float32)
        np.testing.assert_array_equal(pos[:, 0], expected_x)
        np.testing.assert_array_equal(pos[:, 1], expected_x + 100)

    def test_vispy_overlay_positions_dense_selection_excludes_missing_structures(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        row_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        x = np.arange(row_groups.size, dtype=np.float32)
        y = x + 100
        dataset = SimpleNamespace(
            title="energy",
            x=x,
            y=y,
            structure_index=row_groups,
        )

        pos = canvas._overlay_positions_for_indices(dataset, {0, 1, 3, 4})

        np.testing.assert_array_equal(pos[:, 0], np.array([0, 1, 2, 3, 6, 7, 8, 9], dtype=np.float32))
        np.testing.assert_array_equal(pos[:, 1], np.array([100, 101, 102, 103, 106, 107, 108, 109], dtype=np.float32))

    def test_vispy_preview_dense_overlay_uses_image_layer(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)

        class PreviewPlot:
            pass

        plot = PreviewPlot()
        plot._scatter = None
        plot._preview_image = object()
        plot._preview_image_source = np.zeros((10, 10, 4), dtype=np.uint8)
        plot._preview_image_source[..., 3] = 120
        plot._preview_image_range = ((0.0, 9.0), (0.0, 9.0))
        plot._full_detail = False
        plot.convert_color = MagicMock(return_value=(1.0, 0.0, 0.0, 1.0))
        plot.set_overlay_image = MagicMock()
        plot.set_overlay_positions = MagicMock()

        dataset = SimpleNamespace(
            title="force",
            x=np.arange(10, dtype=np.float32),
            y=np.arange(10, dtype=np.float32),
            structure_index=np.arange(10, dtype=np.int32),
        )
        canvas.axes_list = [plot]
        canvas.nep_result_data = SimpleNamespace(select_index=set(range(9)), reject_index=set())
        with patch.object(canvas, "get_axes_dataset", return_value=dataset):
            canvas.update_scatter_color(range(9))

        plot.set_overlay_image.assert_called_once()
        plot.set_overlay_positions.assert_not_called()
        image = plot.set_overlay_image.call_args.args[1]
        self.assertGreater(int(image[0, 0, 3]), 0)
        self.assertEqual(int(image[9, 9, 3]), 0)

    def test_vispy_overlay_cache_signature_reuses_nep_plotdata_properties(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        rows = np.arange(24, dtype=np.float32).reshape(4, 6)
        dataset = NepPlotData(rows, group_list=np.array([1, 2, 1], dtype=np.int32), title="force")

        signature_1 = canvas._overlay_cache_signature(dataset)
        signature_2 = canvas._overlay_cache_signature(dataset)
        lookup_1 = canvas._overlay_position_lookup(dataset)
        lookup_2 = canvas._overlay_position_lookup(dataset)

        self.assertEqual(signature_1, signature_2)
        self.assertIs(lookup_1, lookup_2)

    def test_vispy_polygon_selection_prefilters_candidates_by_bounds(self):
        canvas = canvas_factory._create_vispy_result_canvas(None)
        canvas.init_axes(1)
        canvas.current_axes = canvas.axes_list[0]
        x = np.array([0.0, 0.2, 0.8, 5.0, 6.0], dtype=np.float32)
        y = np.array([0.0, 0.2, 0.8, 5.0, 6.0], dtype=np.float32)
        structure_index = np.arange(5, dtype=np.int32)
        dataset = SimpleNamespace(x=x, y=y, structure_index=structure_index)
        canvas.nep_result_data = SimpleNamespace(datasets=[dataset], select_index=set())
        polygon = np.array([[-0.1, -0.1], [1.0, -0.1], [1.0, 1.0], [-0.1, 1.0]], dtype=np.float32)

        def inside(points, _polygon):
            self.assertEqual(points.shape[0], 3)
            return np.array([True, False, True])

        with patch.object(canvas, "is_point_in_polygon", side_effect=inside), patch.object(canvas, "select_index") as select_index:
            canvas.select_point_from_polygon(polygon, False)

        select_index.assert_called_once_with([0, 2], False)

    def test_vispy_structure_plot_reuses_static_visuals(self):
        plot = canvas_factory._create_vispy_structure_plot(None)
        structure_1 = SimpleNamespace(
            numbers=np.array([26, 26, 26], dtype=np.int32),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            cell=np.eye(3, dtype=np.float32) * 3.0,
            atomic_properties={},
            get_bad_bond_pairs=MagicMock(return_value=[]),
            get_bond_pairs=MagicMock(return_value=[]),
        )
        structure_2 = SimpleNamespace(
            numbers=np.array([26, 26, 26], dtype=np.int32),
            positions=np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]], dtype=np.float32),
            cell=np.eye(3, dtype=np.float32) * 3.0,
            atomic_properties={},
            get_bad_bond_pairs=MagicMock(return_value=[]),
            get_bond_pairs=MagicMock(return_value=[]),
        )

        plot.show_structure(structure_1)
        atom_mesh = plot._atom_mesh
        lattice_item = plot.lattice_item
        axes = plot.axes

        plot.show_structure(structure_2)

        self.assertIs(plot._atom_mesh, atom_mesh)
        self.assertIs(plot.lattice_item, lattice_item)
        self.assertIs(plot.axes, axes)

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
