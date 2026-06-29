"""VisPy canvas widgets for interactive NepTrain result exploration.
"""

import os
os.environ["VISPY_IGNORE_OLD_VERSION"] = "true"

# os.environ["VISPY_PYQT5_SHARE_CONTEXT"] = "true"

import numpy as np
import time

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget
from loguru import logger
from vispy import scene

from vispy.visuals.filters import MarkerPickingFilter
from vispy.visuals.transforms import STTransform
from NepTrainKit.utils import timeit
from NepTrainKit.config import Config
from NepTrainKit.ui.canvas.base.canvas import VispyCanvasLayoutBase
from NepTrainKit.ui.canvas.vispy.fast_scatter import FastScatter
from NepTrainKit.core.io import NepTrainResultData
from NepTrainKit.core.types import Brushes, Pens


VISPY_PREVIEW_WIDTH = 640
VISPY_PREVIEW_HEIGHT = 240
VISPY_PREVIEW_RASTER_OVERLAY_MIN_RATIO = 0.9


def _vispy_perf_enabled():
    return os.environ.get("NEPKIT_VISPY_PERF", "").strip().lower() in {"1", "true", "yes", "on"}


def _vispy_perf_log(message, **values):
    if not _vispy_perf_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    logger.debug(f"[vispy-perf] {message} {payload}".rstrip())


def _elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


class _LassoOverlay(QWidget):
    """Paint lasso feedback without asking VisPy to redraw point clouds."""

    def __init__(self, parent):
        super().__init__(parent)
        self._points = []
        self._pen = QPen(QColor(255, 0, 0), 1.5)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.hide()

    def clear_points(self):
        self._points = []
        self.update()

    def append_point(self, point):
        self._points.append(QPointF(float(point[0]), float(point[1])))
        self.update()

    def set_points(self, points):
        self._points = [QPointF(float(x), float(y)) for x, y in points]
        self.update()

    def paintEvent(self, _event):
        if len(self._points) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(self._pen)
        for i in range(1, len(self._points)):
            painter.drawLine(self._points[i - 1], self._points[i])


class DatasetGpuCache:
    """Cache full plot arrays and full-data preview rasters per dataset."""

    def __init__(self):
        self._arrays = {}
        self._previews = {}

    def clear(self):
        self._arrays.clear()
        self._previews.clear()

    def position_signature(self, dataset):
        data = getattr(dataset, "data", None)
        group_array = getattr(dataset, "group_array", None)
        return (
            id(dataset),
            int(getattr(dataset, "_plot_coord_version", getattr(dataset, "_content_version", 0)) or 0),
            id(getattr(data, "all_data", None)),
            id(getattr(group_array, "all_data", None)),
            np.shape(getattr(data, "all_data", ())),
            np.shape(getattr(group_array, "all_data", ())),
        )

    def dataset_version(self, dataset):
        data = getattr(dataset, "data", None)
        group_array = getattr(dataset, "group_array", None)
        return (
            getattr(data, "version", 0),
            getattr(group_array, "version", 0),
            getattr(data, "num", None),
            getattr(group_array, "num", None),
            np.shape(getattr(data, "all_data", ())),
            np.shape(getattr(group_array, "all_data", ())),
        )

    def arrays(self, dataset):
        signature = self.position_signature(dataset)
        cached = self._arrays.get(signature)
        if cached is not None:
            return cached

        data = np.asarray(dataset.all_data)
        if data.ndim < 2 or data.shape[1] < 2:
            x = data.reshape(-1)
            y = x
            structure_index = np.asarray(dataset.group_array.all_data, dtype=np.int32).reshape(-1)
        else:
            cols = data.shape[1] // 2
            x = data[:, dataset.x_cols].ravel()
            y = data[:, dataset.y_cols].ravel()
            structure_index = np.asarray(dataset.group_array.all_data, dtype=np.int32).repeat(cols)

        cached = (x, y, structure_index)
        dataset_id = id(dataset)
        for key in list(self._arrays):
            if key[0] == dataset_id and key != signature:
                self._arrays.pop(key, None)
        self._arrays[signature] = cached
        return cached

    def active_indices(self, dataset):
        data = getattr(dataset, "data", None)
        mask = getattr(data, "mask_array", None)
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size == 0 or bool(np.all(mask)):
            return None
        all_data = np.asarray(dataset.all_data)
        if all_data.ndim < 2 or all_data.shape[1] < 2:
            return np.nonzero(mask)[0].astype(np.uint32, copy=False)
        cols = all_data.shape[1] // 2
        rows = np.nonzero(mask)[0].astype(np.uint32, copy=False)
        offsets = np.arange(cols, dtype=np.uint32)
        return (rows[:, None] * np.uint32(cols) + offsets[None, :]).ravel()

    def data_range(self, x, y, parity_mode):
        mask = (x > -10000) & np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return [0, 1], [0, 1]
        x_range = [float(np.min(x[mask])), float(np.max(x[mask]))]
        y_range = [float(np.min(y[mask])), float(np.max(y[mask]))]
        if parity_mode:
            real_range = [min(x_range[0], y_range[0]), max(x_range[1], y_range[1])]
            return real_range, real_range
        return x_range, y_range

    def preview(self, dataset, color, parity_mode, active_indices=None):
        color_arr = np.asarray(color, dtype=np.float32).reshape(-1)
        color_key = tuple(float(v) for v in color_arr[:4])
        preview_signature = (
            self.position_signature(dataset),
            self.dataset_version(dataset),
            color_key,
            bool(parity_mode),
            VISPY_PREVIEW_WIDTH,
            VISPY_PREVIEW_HEIGHT,
        )
        cached = self._previews.get(preview_signature)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        x, y, _structure_index = self.arrays(dataset)
        if active_indices is not None:
            active_indices = np.asarray(active_indices, dtype=np.int64)
            x = x[active_indices]
            y = y[active_indices]

        x_range, y_range = self.data_range(x, y, parity_mode)
        image = self._rasterize_preview(x, y, x_range, y_range, color_arr)
        cached = (image, x_range, y_range)
        dataset_id = id(dataset)
        for key in list(self._previews):
            if key[0][0] == dataset_id and key != preview_signature:
                self._previews.pop(key, None)
        self._previews[preview_signature] = cached
        _vispy_perf_log(
            "preview.raster",
            title=getattr(dataset, "title", ""),
            points=int(np.asarray(x).size),
            width=VISPY_PREVIEW_WIDTH,
            height=VISPY_PREVIEW_HEIGHT,
            ms=f"{_elapsed_ms(t0):.3f}",
        )
        return cached

    def _rasterize_preview(self, x, y, x_range, y_range, color):
        width = VISPY_PREVIEW_WIDTH
        height = VISPY_PREVIEW_HEIGHT
        image = np.zeros((height, width, 4), dtype=np.uint8)
        x = np.asarray(x)
        y = np.asarray(y)
        mask = (x > -10000) & np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return image

        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0

        px = ((x[mask] - x_min) / (x_max - x_min) * (width - 1)).astype(np.int64)
        py = ((y[mask] - y_min) / (y_max - y_min) * (height - 1)).astype(np.int64)
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not np.any(valid):
            return image

        px = px[valid]
        py = py[valid]
        counts = np.zeros((height, width), dtype=np.uint16)
        np.add.at(counts, (py, px), 1)
        occupied = counts > 0
        rgb = np.clip(color[:3] * 255, 0, 255).astype(np.uint8)
        alpha_base = float(color[3]) if color.size >= 4 else 1.0
        alpha_base = max(alpha_base, 0.85)
        alpha = np.clip((80 + np.log1p(counts.astype(np.float32)) * 45) * alpha_base, 0, 255).astype(np.uint8)
        image[occupied, :3] = rgb
        image[occupied, 3] = alpha[occupied]
        image[1:, :, :] = np.maximum(image[1:, :, :], image[:-1, :, :])
        image[:, 1:, :] = np.maximum(image[:, 1:, :], image[:, :-1, :])
        return image


class MainPlotView:
    """Render the active plot as a real interactive scatter visual."""

    def __init__(self, cache: DatasetGpuCache):
        self.cache = cache

    def render(self, plot, dataset, brush, pen, marker_size, layer_key, cache_signature, index_signature):
        x, y, structure_index = self.cache.arrays(dataset)
        indices = self.cache.active_indices(dataset)
        plot.set_preview_visible(False)
        return plot.scatter(
            x,
            y,
            data=structure_index,
            brush=brush,
            pen=pen,
            symbol='o',
            size=marker_size,
            layer_key=layer_key,
            cache_signature=cache_signature,
            indices=indices,
            index_signature=index_signature,
        )


class PreviewPlotView:
    """Render inactive plots as cached full-data preview images plus overlays."""

    def __init__(self, cache: DatasetGpuCache):
        self.cache = cache

    def render(self, plot, dataset, brush, pen, marker_size):
        face_color = np.asarray(plot.convert_color(brush), dtype=np.float32).reshape(-1)
        if face_color.size < 4 or face_color[3] <= 0.1:
            color = plot.convert_color(pen)
        else:
            color = face_color
        active_indices = self.cache.active_indices(dataset)
        image, x_range, y_range = self.cache.preview(dataset, color, plot.parity_mode, active_indices=active_indices)
        plot.set_preview_image(image, x_range, y_range)
        return plot._preview_image


class ViewBoxWidget(scene.Widget):

    """Composite widget combining axes, scatter visuals, and overlays for a single subplot.
    """
    def __init__(self, title, full_detail: bool = False, *args, **kwargs):
        """Initialise the widget layout, axes, and default visuals.
        
        Parameters
        ----------
        title : str
            Title label rendered above the view.
        *args : tuple
            Positional arguments forwarded to :class:`vispy.scene.Widget`.
        **kwargs : dict
            Keyword arguments forwarded to :class:`vispy.scene.Widget`.
        """
        super(ViewBoxWidget, self).__init__(*args, **kwargs)

        self.unfreeze()
        self.grid = self.add_grid(margin=0)

        self.grid.spacing = 0
        self.title_label = scene.Label(title, color='black',font_size=8)
        self.title_label.height_max = 30
        self.grid.add_widget(self.title_label, row=0, col=0, col_span=3)

        right_padding = self.grid.add_widget(row=1, col=2, row_span=1)
        right_padding.width_max = 5
        self._view = self.grid.add_view(row=1, col=1,  )

        self._view.camera = scene.cameras.PanZoomCamera()
        self._view.camera.interactive = False

        self.xaxis = None
        self.yaxis = None
        self.text = None
        self._x_label = ""
        self._y_label = ""
        self._rmse_text = ""
        self._full_detail = False


        self.data=np.array([])

        # Configurable marker antialias and size
        try:
            self.marker_antialias = Config.getfloat("widget", "vispy_marker_antialias", 0.5)
        except Exception:
            self.marker_antialias = 0.5
        try:
            self.marker_size_default = Config.getint("widget", "vispy_marker_size", 6)
        except Exception:
            self.marker_size_default = 6

        self.picking_filter = MarkerPickingFilter()

        self._scatter=None
        self._scatter_layers = {}
        self._scatter_signatures = {}
        self._scatter_index_signatures = {}
        self._scatter_ranges = {}
        self._scatter_active_indices = {}
        self._active_scatter_layer = None
        self._scatter_active_signature = None
        self._preview_image = None
        self._preview_image_source = None
        self._preview_image_range = None
        # Overlay marker layers by name (e.g., 'selected', 'show')
        self._overlays = {}
        self._overlay_images = {}
        self.parity_mode = True

        self._diagonal=None
        self.current_point=None
        self._layout_attached = False
        self._layout_position = None
        self._plot_full_detail = None
        self.set_full_detail(full_detail)
        self.freeze()

    def set_full_detail(self, enabled: bool):
        """Toggle full plot controls for the active main plot."""
        if self.xaxis is None:
            self.xaxis = scene.AxisWidget(
                orientation='bottom',
                axis_width=1,
                tick_label_margin=10,
                axis_color="black",
                text_color="black",
            )
            self.xaxis.height_max = 30
            self.grid.add_widget(self.xaxis, row=2, col=1)

            self.yaxis = scene.AxisWidget(
                orientation='left',
                axis_width=1,
                tick_label_margin=5,
                axis_color="black",
                text_color="black",
            )
            self.yaxis.width_max = 50
            self.grid.add_widget(self.yaxis, row=1, col=0)

            self.xaxis.link_view(self._view)
            self.yaxis.link_view(self._view)

            self.text = scene.Text('', parent=self._view.scene, color='red', anchor_x="left", anchor_y="top")
            self.text.font_size = 8
            self.set_axis_labels(self._x_label, self._y_label)
            self.set_rmse_text(self._rmse_text)

        self._full_detail = bool(enabled)
        if self.xaxis is not None:
            self.xaxis.visible = True
            self.xaxis.height_max = 30 if enabled else 22
        if self.yaxis is not None:
            self.yaxis.visible = True
            self.yaxis.width_max = 50 if enabled else 38
        if self.text is not None:
            self.text.visible = bool(enabled)
        if enabled and self.parity_mode and self.title not in ("", "descriptor") and self._diagonal is None:
            self.add_diagonal(color="red", width=3, antialias=True, method='gl')

    def set_preview_visible(self, visible: bool):
        if self._preview_image is not None:
            self._preview_image.visible = bool(visible)
        if not visible:
            for image in self._overlay_images.values():
                image.visible = False
        for layer in self._scatter_layers.values():
            layer.visible = not visible and layer is self._scatter

    def set_preview_image(self, image, x_range, y_range):
        preview_range = (tuple(float(v) for v in x_range), tuple(float(v) for v in y_range))
        if self._preview_image is not None and self._preview_image_source is image and self._preview_image_range == preview_range:
            self._preview_image.visible = True
            for layer in self._scatter_layers.values():
                layer.visible = False
            self._scatter = None
            self._active_scatter_layer = None
            return

        if self._preview_image is None:
            self._preview_image = scene.visuals.Image(image, interpolation="nearest", parent=self._view.scene)
            self._preview_image.order = 1
        else:
            self._preview_image.set_data(image)
        self._preview_image_source = image
        self._preview_image_range = preview_range

        width = max(1, image.shape[1])
        height = max(1, image.shape[0])
        x_min, x_max = x_range
        y_min, y_max = y_range
        self._preview_image.transform = STTransform(
            scale=((x_max - x_min) / width, (y_max - y_min) / height),
            translate=(x_min, y_min),
        )
        self._preview_image.visible = True
        for layer in self._scatter_layers.values():
            layer.visible = False
        self._scatter = None
        self._active_scatter_layer = None
        self._apply_data_range(x_range, y_range)
        if self.parity_mode and self.title not in ("", "descriptor") and self._diagonal is None:
            self.add_diagonal(color="red", width=3, antialias=True, method='gl')
        self.update_diagonal()

    def set_axis_labels(self, x_label=None, y_label=None):
        """Store and apply axis labels when full controls exist."""
        self._x_label = str(x_label or "")
        self._y_label = str(y_label or "")
        if self.xaxis is not None:
            self.xaxis.axis.axis_label = self._x_label
        if self.yaxis is not None:
            self.yaxis.axis.axis_label = self._y_label

    def set_rmse_text(self, text: str):
        """Store and apply the RMSE annotation."""
        self._rmse_text = str(text or "")
        if self.text is not None:
            self.text.text = self._rmse_text



    def convert_color(self, obj):
        """Convert Qt colour objects to RGBA floats understood by VisPy.
        
        Parameters
        ----------
        obj : QPen or QBrush or QColor or Sequence[float]
            Colour-like object to convert.
        
        Returns
        -------
        list[float]
            Normalised RGBA components.
        """
        if isinstance(obj, (QPen, QBrush)):

            color = obj.color()
            edge_color = list(color.getRgbF())
        elif isinstance(obj, QColor):
            color = obj
            edge_color = list(color.getRgbF())

        else:
            edge_color = obj

        return edge_color

    def _range_from_arrays(self, x, y):
        mask = (x > -10000) & np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return [0, 1], [0, 1]
        return [float(np.min(x[mask])), float(np.max(x[mask]))], [float(np.min(y[mask])), float(np.max(y[mask]))]

    def _apply_data_range(self, x_range, y_range):
        if self.parity_mode:
            real_range=(min(x_range[0],y_range[0]),max(x_range[1],y_range[1]))
            # Provide z-range to avoid VisPy querying scene bounds for z (empty visuals would error)
            self._view.camera.set_range(x=real_range, y=real_range, z=(0, 0))
        else:
            self._view.camera.set_range(x=x_range, y=y_range, z=(0, 0))

    def _apply_scatter_range(self, layer_key):
        cached_range = self._scatter_ranges.get(layer_key)
        if cached_range is None:
            return False
        self._apply_data_range(*cached_range)
        return True

    def auto_range(self):
        """Auto-scale the pan/zoom camera to fit the scatter data.
        """
        if self._scatter is None:
            return

        if hasattr(self._scatter, "positions"):
            pos = self._scatter.positions
        else:
            pos = self._scatter._data["a_position"]
        if pos.size==0:
            return
        active_indices = self._scatter_active_indices.get(self._active_scatter_layer)
        if active_indices is not None:
            pos = pos[np.asarray(active_indices, dtype=np.int64)]
        x_range, y_range = self._range_from_arrays(pos[:, 0], pos[:, 1])
        if self._active_scatter_layer is not None:
            self._scatter_ranges[self._active_scatter_layer] = (x_range, y_range)
        self._apply_data_range(x_range, y_range)

    def set_current_point(self, x,y):

        """Display a highlighted marker for the active structure.
        
        Parameters
        ----------
        x : ndarray
            X coordinates of the marker points.
        y : ndarray
            Y coordinates of the marker points.
        """
        if np.array(x).size == 0:
            if self.current_point is not None:
                self.current_point.parent=None
                self.current_point=None

            return
        if self.current_point is None:
            # Create a top-most marker layer for current structure
            self.current_point = scene.visuals.Markers(antialias=1)
            # Ensure it renders above all other markers/overlays
            self.current_point.order = 100
            # Disable depth testing so nothing can occlude it
            self.current_point.update_gl_state(depth_test=False)
            self._view.add(self.current_point)

        z=np.full(x.shape,2)
        current_size = Config.getint("plot", "current_marker_size", 20) or 20
        if not self._full_detail:
            current_size = max(6, int(current_size * 0.45))
        self.current_point.set_data(
            np.vstack([x, y, z]).T,
            face_color=self.convert_color(Brushes.Current),
            edge_color=self.convert_color(Pens.Current),
            symbol='star',
            size=current_size,
        )

    def _ensure_scatter(self, layer_key):
        if layer_key in self._scatter_layers and self._scatter_layers[layer_key] is not None:
            return self._scatter_layers[layer_key]

        scatter = FastScatter()
        scatter.order=1
        self._view.add(scatter)
        scatter.visible = False
        self._scatter_layers[layer_key] = scatter
        return scatter

    def _set_scatter_indices(self, layer_key, indices=None, index_signature=None):
        scatter = self._scatter_layers.get(layer_key)
        if scatter is None or not hasattr(scatter, "set_indices"):
            return False
        if index_signature is not None and self._scatter_index_signatures.get(layer_key) == index_signature:
            return False
        scatter.set_indices(indices)
        self._scatter_active_indices[layer_key] = None if indices is None else np.asarray(indices, dtype=np.uint32)
        if indices is None:
            cached_range = self._scatter_ranges.get((layer_key, "all"))
            if cached_range is not None:
                self._scatter_ranges[layer_key] = cached_range
        if index_signature is not None:
            self._scatter_index_signatures[layer_key] = index_signature
        return True

    def activate_scatter_layer(self, layer_key, data=None, cache_signature=None, indices=None, index_signature=None):
        scatter = self._ensure_scatter(layer_key)
        if self._preview_image is not None:
            self._preview_image.visible = False
        layer_changed = self._active_scatter_layer != layer_key
        for key, layer in self._scatter_layers.items():
            layer.visible = key == layer_key
        self._scatter = scatter
        self._active_scatter_layer = layer_key
        if data is not None:
            self.data = data
        if cache_signature is not None and self._scatter_signatures.get(layer_key) == cache_signature:
            index_changed = self._set_scatter_indices(layer_key, indices, index_signature)
            active_signature = (cache_signature, index_signature)
            if layer_changed or index_changed or self._scatter_active_signature != active_signature:
                self._apply_scatter_range(layer_key)
                self.update_diagonal()
                self._scatter_active_signature = active_signature
            return True
        return False

    def scatter(
        self,
        x,
        y,
        data,
        brush=None,
        pen=None,
        *,
        layer_key="full",
        cache_signature=None,
        indices=None,
        index_signature=None,
        **kwargs
    ):
        """Update or create the primary scatter visual.
        
        Parameters
        ----------
        x : ndarray
            X coordinates of data points.
        y : ndarray
            Y coordinates of data points.
        data : array-like
            Metadata array stored alongside the scatter.
        brush : Any, optional
            Brush or colour specification applied to marker faces.
        pen : Any, optional
            Pen or colour specification applied to marker edges.
        **kwargs : dict
            Additional styling arguments forwarded to :class:`vispy.scene.visuals.Markers`.
        
        Returns
        -------
        vispy.scene.visuals.Markers
            Scatter visual used to render the data.
        """
        if brush is not None:

            kwargs["face_color"]=self.convert_color(brush)
        if pen is not None:

            kwargs["edge_color"]=self.convert_color(pen)
            if isinstance(pen, QPen):
                kwargs["edge_width"] = max(0.0, float(pen.widthF() or pen.width() or 1.0))
        signature = cache_signature
        if signature is None:
            signature = (
                id(x),
                id(y),
                id(data),
                tuple(np.shape(x)),
                tuple(np.shape(y)),
                tuple(np.shape(data)),
                kwargs.get("size"),
                kwargs.get("symbol"),
                str(kwargs.get("face_color")),
                str(kwargs.get("edge_color")),
                kwargs.get("edge_width"),
            )

        if self.activate_scatter_layer(
            layer_key,
            data=data,
            cache_signature=signature,
            indices=indices,
            index_signature=index_signature,
        ):
            return self._scatter
        scatter = self._scatter

        if x.size != 0:

            pos = np.column_stack([x, y]).astype(np.float32, copy=False)
            # Ensure a default size if caller didn't provide one
            if 'size' not in kwargs or kwargs.get('size') is None:
                kwargs['size'] = self.marker_size_default
            # self._scatter.update_gl_state(depth_test=False)
            scatter.set_data(pos, **kwargs)
            self._scatter_signatures[layer_key] = signature
            full_range = self._range_from_arrays(pos[:, 0], pos[:, 1])
            self._scatter_ranges[(layer_key, "all")] = full_range
            self._scatter_ranges[layer_key] = full_range
            self._set_scatter_indices(layer_key, indices, index_signature)
            self._apply_scatter_range(layer_key)
            self._scatter_active_signature = (signature, index_signature)
        else:
            scatter.set_data(np.empty((0, 3)), **kwargs)
            self._scatter_signatures[layer_key] = signature
            self._scatter_ranges[(layer_key, "all")] = ([0, 1], [0, 1])
            self._scatter_ranges[layer_key] = ([0, 1], [0, 1])
            self._set_scatter_indices(layer_key, None, index_signature)
            self._apply_scatter_range(layer_key)
            self._scatter_active_signature = (signature, index_signature)
        if self.parity_mode and self.title not in ("", "descriptor") and self._diagonal is None:
            self.add_diagonal(color="red", width=3, antialias=True, method='gl')
        self.update_diagonal()
        return self._scatter

    def line(self,x,y,**kwargs):
        """Draw a line visual within the view.
        
        Parameters
        ----------
        x : ndarray
            X coordinates.
        y : ndarray
            Y coordinates.
        **kwargs : dict
            Line styling arguments forwarded to :class:`vispy.scene.visuals.Line`.
        
        Returns
        -------
        vispy.scene.visuals.Line
            Line visual added to the view.
        """
        xy=np.vstack([x,y]).T

        line=scene.Line(xy , **kwargs)
        self.view.add(line)
        return line

    def add_diagonal(self,**kwargs):


        """Add a parity diagonal overlay using the current axis domain.
        
        Parameters
        ----------
        **kwargs : dict
            Styling arguments forwarded to :meth:`line`.
        """
        if self.xaxis is None:
            return
        x_domain = self.xaxis.axis.domain
        line_data = np.linspace(*x_domain,num=100)
        self._diagonal=self.line(line_data,line_data,**kwargs)

        self._diagonal.order=3
    def update_diagonal(self):
        """Update the parity diagonal to match the latest axis domain.
        """
        if self._diagonal is None:
            return None
        if self.xaxis is None:
            return None
        x_domain = self.xaxis.axis.domain

        line_data = np.linspace(*x_domain,num=100)
        xy = np.vstack([line_data, line_data]).T
        self._diagonal.set_data(xy)

    @property
    def view(self):
        """scene.widgets.ViewBox: Underlying view used to render data.
        """
        return self._view

    def _ensure_overlay(self, name:str, color, size:int=9, symbol:str='o'):
        """Create or retrieve a named overlay marker layer.
        
        Parameters
        ----------
        name : str
            Key used to cache the overlay.
        color : Any
            Colour specification for the overlay markers.
        size : int, optional
            Marker size in logical pixels.
        symbol : str, optional
            Marker symbol used for rendering.
        
        Returns
        -------
        vispy.scene.visuals.Markers
            Overlay visual ready to receive data.
        """
        if name in self._overlays and self._overlays[name] is not None:
            return self._overlays[name]
        if symbol == 'o':
            ov = FastScatter()
        else:
            ov = scene.visuals.Markers(antialias=1)
        order_map = {
            "loaded": 4,
            "show": 4,
            "reject": 5,
            "selected": 6,
        }
        ov.order = order_map.get(name, 4)  # above base scatter and diagonal
        # keep same scene/camera
        self._view.add(ov)
        if hasattr(ov, "update_gl_state"):
            ov.update_gl_state(depth_test=False)
        # initialize with empty data and hide from bounds
        ov.set_data(np.empty((0, 2), dtype=np.float32), face_color=self.convert_color(color), edge_width=0, symbol=symbol, size=size)
        ov.visible = False
        self._overlays[name] = ov
        return ov

    def set_overlay_positions(self, name:str, pos:np.ndarray, color=None, size:int=9, symbol:str='o'):
        """Replace the geometry of a named overlay layer.
        
        Parameters
        ----------
        name : str
            Overlay identifier created by :meth:`_ensure_overlay`.
        pos : ndarray
            Position array with shape ``(N, 2)`` in view coordinates.
        color : Any, optional
            Colour override applied to the overlay markers.
        size : int, optional
            Marker size in logical pixels.
        symbol : str, optional
            Marker symbol used when drawing the overlay.
        
        Returns
        -------
        vispy.scene.visuals.Markers
            Overlay visual that was updated.
        """
        if pos is None:
            pos = np.empty((0, 2), dtype=np.float32)
        ov = self._ensure_overlay(name, color=color if color is not None else Brushes.Selected, size=size, symbol=symbol)
        kwargs = {}
        if color is not None:
            kwargs['face_color'] = self.convert_color(color)
        # Use face fill for highlight; no edge for lower cost
        pos = np.asarray(pos, dtype=np.float32)
        ov.set_data(pos=pos, edge_width=0, symbol=symbol, size=size, **kwargs)
        ov.visible = bool(pos.size)
        image = self._overlay_images.get(name)
        if image is not None:
            image.visible = False
        return ov

    def set_overlay_image(self, name: str, image: np.ndarray | None, x_range, y_range):
        """Replace a preview overlay image layer."""
        marker = self._overlays.get(name)
        if marker is not None:
            marker.visible = False
        overlay = self._overlay_images.get(name)
        if image is None or not np.asarray(image).size:
            if overlay is not None:
                overlay.visible = False
            return overlay

        image = np.asarray(image, dtype=np.uint8)
        if overlay is None:
            overlay = scene.visuals.Image(image, interpolation="nearest", parent=self._view.scene)
            order_map = {
                "loaded": 4,
                "show": 4,
                "reject": 5,
                "selected": 6,
            }
            overlay.order = order_map.get(name, 4)
            self._overlay_images[name] = overlay
        else:
            overlay.set_data(image)

        width = max(1, image.shape[1])
        height = max(1, image.shape[0])
        x_min, x_max = x_range
        y_min, y_max = y_range
        overlay.transform = STTransform(
            scale=((x_max - x_min) / width, (y_max - y_min) / height),
            translate=(x_min, y_min),
        )
        overlay.visible = bool(np.any(image[..., 3]))
        return overlay

    def clear_overlays(self):
        """Hide and clear all overlay layers for this view.
        """
        empty = np.empty((0, 2), dtype=np.float32)
        for ov in self._overlays.values():
            ov.set_data(pos=empty, edge_width=0, symbol='o', size=9)
            ov.visible = False
        for image in self._overlay_images.values():
            image.visible = False

    @property
    def title(self):
        """Get the text displayed above the view.
        
        Returns
        -------
        str
            Title text.
        """
        return self.title_label._text_visual.text
    @property
    def rmse_size(self):
        """Get the font size used for the RMSE annotation.
        
        Returns
        -------
        int
            Font size in points.
        """
        if self.text is None:
            return 0
        return self.text.font_size
    @rmse_size.setter
    def rmse_size(self,size):

        """Set the font size used for the RMSE annotation.
        
        Parameters
        ----------
        size : int
            Font size in points.
        """
        if self.text is not None:
            self.text.font_size=size
    @title.setter
    def title(self, t):

        """Update the title label and refresh derived overlays.
        
        Parameters
        ----------
        t : str
            New title text.
        """
        if t==self.title:
            return
        self.title_label._text_visual.text = t
        if self.xaxis is not None and self.parity_mode and t != "descriptor" and self._diagonal is None:
            self.add_diagonal(color="red", width=3, antialias=True, method='gl')
class CombinedMeta(type(VispyCanvasLayoutBase), type(scene.SceneCanvas) ):
    """Metaclass bridging ``VispyCanvasLayoutBase`` with ``SceneCanvas`` inheritance.
    """
    pass


class VispyCanvas(VispyCanvasLayoutBase, scene.SceneCanvas, metaclass=CombinedMeta):

    """SceneCanvas-based implementation that arranges multiple ViewBoxWidget instances.
    """
    def __init__(self, *args, **kwargs):

        """Initialise the scene canvas and shared layout state.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`vispy.scene.SceneCanvas`.
        **kwargs : dict
            Keyword arguments forwarded to :class:`vispy.scene.SceneCanvas`.
        """
        VispyCanvasLayoutBase.__init__(self)

        scene.SceneCanvas.__init__(self, *args,    **kwargs)

        self.unfreeze()
        self.nep_result_data = None

        # Per-axes overlay state: track indices to render in overlays without touching base VBO
        self._selected_by_plot = {}
        self._show_by_plot = {}
        self._loaded_by_plot = {}
        self._reject_by_plot = {}
        self._overlay_position_cache = {}
        self._dataset_cache = DatasetGpuCache()
        self._main_plot_view = MainPlotView(self._dataset_cache)
        self._preview_plot_view = PreviewPlotView(self._dataset_cache)
        self._plot_dataset_indices = []


        self.grid = self.central_widget.add_grid(margin=0, spacing=0)
        self.grid.spacing = 0


        self.events.mouse_double_click.connect(self.switch_view_box)
        self._lasso_overlay = _LassoOverlay(self.native)
        self._sync_lasso_overlay_geometry()
        self._lasso_overlay.show()
        self._lasso_overlay.raise_()
        self._lasso_screen_path = []
        self._lasso_perf = None
        self.events.resize.connect(lambda _event: self._sync_lasso_overlay_geometry())
        # Use filters to affect the rendering of the mesh.

    def clear_axes(self):
        """Remove all ViewBox widgets from the grid and reset internal state.
        """
        for widget in self.axes_list:
            widget._stretch = (None, None)
            widget.parent=None
            if getattr(widget, "_layout_attached", False):
                self.grid.remove_widget(widget)
                widget._layout_attached = False
        self._selected_by_plot.clear()
        self._show_by_plot.clear()
        self._loaded_by_plot.clear()
        self._reject_by_plot.clear()
        self._overlay_position_cache.clear()
        self._dataset_cache.clear()
        self._plot_dataset_indices.clear()
        self.current_axes = None

        super().clear_axes()


    def set_nep_result_data(self,dataset):
        """Attach a NepTrain result dataset to the canvas.
        
        Parameters
        ----------
        dataset : NepTrainResultData
            Dataset used for plotting and interaction.
        """
        self.nep_result_data:NepTrainResultData=dataset
        self._ensure_plot_dataset_indices()

    def _ensure_plot_dataset_indices(self):
        count = len(getattr(self.nep_result_data, "datasets", []) or [])
        if len(self._plot_dataset_indices) != len(self.axes_list) or any(idx >= count for idx in self._plot_dataset_indices):
            self._plot_dataset_indices = list(range(min(len(self.axes_list), count)))

    def _dataset_index_for_plot(self, plot):
        if plot not in self.axes_list:
            return None
        self._ensure_plot_dataset_indices()
        plot_index = self.axes_list.index(plot)
        if plot_index >= len(self._plot_dataset_indices):
            return None
        return self._plot_dataset_indices[plot_index]

    def get_axes_dataset(self, axes):
        if axes is None or self.nep_result_data is None:
            return None
        dataset_index = self._dataset_index_for_plot(axes)
        if dataset_index is None:
            return None
        datasets = self.nep_result_data.datasets
        if dataset_index >= len(datasets):
            return None
        return datasets[dataset_index]


    def _canvas_to_data_pos(self, axes, pos):
        """Map a canvas pixel position into an axes data coordinate."""
        tr = self.scene.node_transform(axes.view.scene)
        x, y, _, _ = tr.map(pos)
        return float(x), float(y)

    def _interaction_arrays_for_axes(self, axes):
        """Return full data arrays for picking and polygon selection."""
        dataset = None
        try:
            dataset = self.get_axes_dataset(axes)
        except (AttributeError, IndexError, ValueError):
            dataset = None
        if dataset is not None:
            return np.asarray(dataset.x), np.asarray(dataset.y), np.asarray(dataset.structure_index)

        if axes is None or axes.data.size == 0 or axes._scatter is None:
            return None, None, None
        if hasattr(axes._scatter, "positions"):
            positions = axes._scatter.positions
        else:
            positions = axes._scatter._data["a_position"]
        if positions.size == 0:
            return None, None, None
        positions = positions[:, :2]
        size = min(positions.shape[0], axes.data.size)
        return positions[:size, 0], positions[:size, 1], np.asarray(axes.data[:size])

    def point_at(self, pos, current_axes=None):
        """Return the marker index under the given canvas position.
        
        Parameters
        ----------
        pos : tuple[float, float]
            Mouse position in canvas coordinates.
        
        Returns
        -------
        int or None
            Index of the nearest marker, or ``None`` if nothing was picked.
        """
        if self.nep_result_data is None:
            return None
        if current_axes is None:
            current_axes=self._get_clicked_axes(pos)
        if current_axes is None:
            return None
        x, y, structure_index = self._interaction_arrays_for_axes(current_axes)
        if x is None or y is None or structure_index is None or structure_index.size == 0:
            return None

        x0, y0 = self._canvas_to_data_pos(current_axes, pos)
        radius_px = float(max(6, getattr(current_axes, "marker_size_default", 6)))
        x1, y1 = self._canvas_to_data_pos(current_axes, (pos[0] + radius_px, pos[1]))
        x2, y2 = self._canvas_to_data_pos(current_axes, (pos[0], pos[1] + radius_px))
        dx = max(abs(x1 - x0), np.finfo(np.float32).eps)
        dy = max(abs(y2 - y0), np.finfo(np.float32).eps)

        mask = (x > -10000) & (np.abs(x - x0) <= dx) & (np.abs(y - y0) <= dy)
        if not np.any(mask):
            return None

        candidate_indices = np.nonzero(mask)[0]
        candidate_x = x[candidate_indices]
        candidate_y = y[candidate_indices]
        dist2 = ((candidate_x - x0) / dx) ** 2 + ((candidate_y - y0) / dy) ** 2
        nearest = int(np.argmin(dist2))
        if float(dist2[nearest]) > 1.0:
            return None
        return int(candidate_indices[nearest])

    def structure_at(self, pos, current_axes=None):
        """Return the structure index under the given canvas position."""
        index = self.point_at(pos, current_axes)
        if index is None:
            return None
        if current_axes is None:
            current_axes = self._get_clicked_axes(pos)
        _x, _y, structure_index = self._interaction_arrays_for_axes(current_axes)
        if structure_index is None or index >= structure_index.size:
            return None
        return int(structure_index[index])

    def _sync_lasso_overlay_geometry(self):
        if self._lasso_overlay is None:
            return
        self._lasso_overlay.setGeometry(self.native.rect())

    def _begin_lasso_overlay(self, point):
        if self._lasso_overlay is None:
            return
        self._lasso_overlay.clear_points()
        self._lasso_overlay.append_point(point)

    def _append_lasso_overlay_point(self, point):
        if self._lasso_overlay is None:
            return
        self._lasso_overlay.append_point(point)

    def _clear_lasso_overlay(self):
        if self._lasso_overlay is None:
            return
        self._lasso_overlay.clear_points()

    def on_mouse_press(self, event):
        """Handle mouse press events for either picking or polygon drawing.
        
        Parameters
        ----------
        event : vispy.app.MouseEvent
            Mouse press event.
        """

        if not self.draw_mode:

            current_axes = self._get_clicked_axes(event.pos)
            structure_index = self.structure_at(event.pos, current_axes)

            if structure_index is not None:
                self.structureIndexChanged.emit(structure_index)

            return False

        if event.button == 1 or event.button ==2:
            if self.draw_mode:

                event_t0 = time.perf_counter()
                transform_t0 = time.perf_counter()
                tr = self.scene.node_transform(self.current_axes.view.scene)
                x, y, _, _ = tr.map(event.pos)
                transform_ms = _elapsed_ms(transform_t0)
                self.mouse_path = [[x, y]]
                self._lasso_screen_path = [tuple(event.pos)]
                overlay_t0 = time.perf_counter()
                self._begin_lasso_overlay(event.pos)
                overlay_ms = _elapsed_ms(overlay_t0)
                self._lasso_perf = {
                    "start": event_t0,
                    "moves": 0,
                    "move_ms": 0.0,
                    "max_move_ms": 0.0,
                    "transform_ms": transform_ms,
                    "overlay_ms": overlay_ms,
                }
                _vispy_perf_log(
                    "lasso.press",
                    transform_ms=f"{transform_ms:.3f}",
                    overlay_ms=f"{overlay_ms:.3f}",
                    total_ms=f"{_elapsed_ms(event_t0):.3f}",
                )


    def on_mouse_move(self, event):
        """Update the polygon path while the user is drawing.
        
        Parameters
        ----------
        event : vispy.app.MouseEvent
            Mouse move event.
        """

        if not self.draw_mode:
            return
        if (event.button == 1 or event.button ==2) and len(self.mouse_path) > 0:
            event_t0 = time.perf_counter()
            transform_t0 = time.perf_counter()
            tr = self.scene.node_transform(self.current_axes.view.scene)
            x, y, _, _ = tr.map(event.pos)
            transform_ms = _elapsed_ms(transform_t0)

            self.mouse_path.append([x,y])
            self._lasso_screen_path.append(tuple(event.pos))
            overlay_t0 = time.perf_counter()
            self._append_lasso_overlay_point(event.pos)
            overlay_ms = _elapsed_ms(overlay_t0)
            total_ms = _elapsed_ms(event_t0)
            if self._lasso_perf is not None:
                self._lasso_perf["moves"] += 1
                self._lasso_perf["move_ms"] += total_ms
                self._lasso_perf["max_move_ms"] = max(self._lasso_perf["max_move_ms"], total_ms)
                self._lasso_perf["transform_ms"] += transform_ms
                self._lasso_perf["overlay_ms"] += overlay_ms
            if total_ms >= 8.0:
                _vispy_perf_log(
                    "lasso.move_slow",
                    points=len(self.mouse_path),
                    transform_ms=f"{transform_ms:.3f}",
                    overlay_ms=f"{overlay_ms:.3f}",
                    total_ms=f"{total_ms:.3f}",
                )

    def on_mouse_release(self, event):
        """Complete selection interactions when the mouse button is released.
        
        Parameters
        ----------
        event : vispy.app.MouseEvent
            Mouse release event.
        """
        if not self.draw_mode:
            return
        if event.button == 1 or event.button ==2:
            event_t0 = time.perf_counter()
            reverse=event.button == 2

            clear_t0 = time.perf_counter()
            self._clear_lasso_overlay()
            clear_ms = _elapsed_ms(clear_t0)
            select_ms = 0.0

            if len(self.mouse_path)>2:

                select_t0 = time.perf_counter()
                self.select_point_from_polygon(np.array(self.mouse_path),reverse)
                select_ms = _elapsed_ms(select_t0)
            else:
                select_t0 = time.perf_counter()
                structure_index = self.structure_at(event.pos, self.current_axes)
                if structure_index is not None:
                    self.select_index(structure_index,reverse)
                select_ms = _elapsed_ms(select_t0)

            perf = self._lasso_perf or {}
            moves = int(perf.get("moves", 0) or 0)
            move_ms = float(perf.get("move_ms", 0.0) or 0.0)
            _vispy_perf_log(
                "lasso.release",
                points=len(self.mouse_path),
                moves=moves,
                avg_move_ms=f"{(move_ms / moves) if moves else 0.0:.3f}",
                max_move_ms=f"{float(perf.get('max_move_ms', 0.0) or 0.0):.3f}",
                transform_ms=f"{float(perf.get('transform_ms', 0.0) or 0.0):.3f}",
                overlay_ms=f"{float(perf.get('overlay_ms', 0.0) or 0.0):.3f}",
                clear_ms=f"{clear_ms:.3f}",
                select_ms=f"{select_ms:.3f}",
                total_ms=f"{_elapsed_ms(event_t0):.3f}",
            )
            self.mouse_path = []
            self._lasso_screen_path = []
            self._lasso_perf = None

    def _get_clicked_axes(self,pos):
        """Return the ViewBoxWidget beneath the given canvas position.
        
        Parameters
        ----------
        pos : tuple[float, float]
            Mouse position in canvas coordinates.
        
        Returns
        -------
        ViewBoxWidget or None
            Widget that contains the point, if any.
        """
        view = self.visual_at(pos)

        while view is not None and not isinstance(view, scene.ViewBox):
            view = getattr(view, "parent", None)

        if isinstance(view, scene.ViewBox):
            for axes in self.axes_list:
                if axes.view == view:
                    return axes

        return None
    def switch_view_box(self,event ):
        """Focus the axes associated with a double-click event.
        
        Parameters
        ----------
        event : vispy.app.MouseEvent
            Mouse double-click event.
        """
        mouse_pos = event.pos
        axes=self._get_clicked_axes(mouse_pos)
        if axes is None:
            return
        if axes is self.axes_list[0]:
            return
        old_axes = self.axes_list[0]
        self.current_axes = old_axes
        old_index = self._dataset_index_for_plot(old_axes)
        new_index = self._dataset_index_for_plot(axes)
        if old_index is None or new_index is None:
            return
        main_slot = self.axes_list.index(old_axes)
        preview_slot = self.axes_list.index(axes)
        self._plot_dataset_indices[main_slot], self._plot_dataset_indices[preview_slot] = new_index, old_index
        self._render_plot(old_axes, self.nep_result_data.datasets[new_index], True)
        self._render_plot(axes, self.nep_result_data.datasets[old_index], False)
        for plot in (old_axes, axes):
            plot.clear_overlays()
            self._selected_by_plot.setdefault(plot, set()).clear()
            self._show_by_plot.setdefault(plot, set()).clear()
            self._loaded_by_plot.setdefault(plot, set()).clear()
            self._reject_by_plot.setdefault(plot, set()).clear()
        selected = sorted(getattr(self.nep_result_data, "select_index", set()) or [])
        if selected:
            self.update_scatter_color(selected, Brushes.Selected)
        reject = sorted(getattr(self.nep_result_data, "reject_index", set()) or [])
        if reject:
            self.set_reject_highlight(reject, True)
        self._refresh_current_axes_annotations()

    def init_axes(self,axes_num   ):
        """Create the requested number of axes widgets.
        
        Parameters
        ----------
        axes_num : int
            Number of subplots to allocate.
        """
        self.clear_axes()
        for r in range(axes_num):
            plot = ViewBoxWidget(title="", full_detail=(r == 0))
            self.axes_list.append(plot)
        if self.axes_list:
            self.current_axes = self.axes_list[0]
        self.set_view_layout()
        self.update()

    def set_view_layout(self):
        """Arrange axes so the active plot occupies the main area and others align below.
        """
        total_t0 = time.perf_counter()
        if len(self.axes_list)==0:
            return
        self.current_axes = self.axes_list[0]

        i = 0
        moved = 0
        unchanged = 0
        row_0_col_span = max(1, len(self.axes_list) - 1)
        for plot_index, widget in enumerate(self.axes_list):
            if plot_index == 0:
                full_detail = True
                rmse_size = 8
                layout = (0, 0, 6, row_0_col_span)
            else:
                full_detail = False
                rmse_size = 4
                layout = (6, i, 2, 1)
                i += 1

            if hasattr(widget, "set_full_detail"):
                widget.set_full_detail(full_detail)
            widget.rmse_size = rmse_size

            old_layout = getattr(widget, "_layout_position", None)
            attached = getattr(widget, "_layout_attached", False)
            if attached and old_layout != layout:
                self.grid.remove_widget(widget)
                widget._layout_attached = False
                attached = False

            if not attached:
                widget._stretch = (None, None)
                self.grid.add_widget(widget, row=layout[0], col=layout[1], row_span=layout[2], col_span=layout[3])
                widget._layout_attached = True
                widget._layout_position = layout
                moved += 1
            else:
                unchanged += 1
        _vispy_perf_log(
            "layout.switch",
            axes=len(self.axes_list),
            moved=moved,
            unchanged=unchanged,
            total_ms=f"{_elapsed_ms(total_t0):.3f}",
        )

    def _dataset_version(self, dataset):
        data = getattr(dataset, "data", None)
        group_array = getattr(dataset, "group_array", None)
        return (
            getattr(data, "version", 0),
            getattr(group_array, "version", 0),
            getattr(data, "num", None),
            getattr(group_array, "num", None),
            np.shape(getattr(data, "all_data", ())),
            np.shape(getattr(group_array, "all_data", ())),
        )

    def _dataset_position_version(self, dataset):
        return self._dataset_cache.position_signature(dataset)[1:]

    def _full_plot_arrays(self, dataset):
        return self._dataset_cache.arrays(dataset)

    def _active_point_indices(self, dataset):
        return self._dataset_cache.active_indices(dataset)

    def _dataset_layer_key(self, dataset):
        return f"dataset-{id(dataset)}"

    def _plot_cache_signature(self, plot, dataset, full_detail: bool):
        marker_size = Config.getint("widget", "vispy_marker_size", 6) or 6
        layer_key = self._dataset_layer_key(dataset)
        brush = getattr(dataset, "base_brush", Brushes.get(dataset.title.upper()))
        pen = getattr(dataset, "base_pen", Pens.get(dataset.title.upper()))
        version = self._dataset_position_version(dataset)
        index_signature = self._dataset_version(dataset)
        return (
            layer_key,
            brush,
            pen,
            marker_size,
            (
                id(dataset),
                version,
                layer_key,
                marker_size,
                dataset.title,
                getattr(dataset, "display_title", dataset.title),
                tuple(plot.convert_color(brush)) if brush is not None else None,
                tuple(plot.convert_color(pen)) if pen is not None else None,
            ),
            index_signature,
        )

    def _apply_plot_annotations(self, plot, dataset, full_detail):
        rmse_func = getattr(dataset, "get_formart_rmse", None)
        show_rmse = bool(full_detail) and bool(getattr(dataset, "show_rmse", dataset.title not in ["descriptor"])) and callable(rmse_func)
        rmse_text = f"rmse: {rmse_func()}" if show_rmse else ""
        plot.set_rmse_text(rmse_text if full_detail else "")
        if show_rmse and full_detail:
            pos = self.convert_pos(plot, (0.1, 0.8))
            if plot.text is not None:
                plot.text.pos = pos
        else:
            plot.set_rmse_text("")
        plot.set_axis_labels(getattr(dataset, "x_label", None), getattr(dataset, "y_label", None))

    def _render_plot(self, plot, dataset, full_detail):
        plot.parity_mode = bool(getattr(dataset, "parity_mode", dataset.title != "descriptor"))
        plot.title = dataset.title
        display_title = str(getattr(dataset, "display_title", dataset.title) or dataset.title)
        if display_title != plot.title:
            plot.title_label._text_visual.text = display_title

        layer_key, brush, pen, marker_size, cache_signature, index_signature = self._plot_cache_signature(plot, dataset, full_detail)
        if full_detail:
            plot.set_full_detail(True)
            self._main_plot_view.render(plot, dataset, brush, pen, marker_size, layer_key, cache_signature, index_signature)
        else:
            plot.set_full_detail(False)
            self._preview_plot_view.render(plot, dataset, brush, pen, marker_size)
            plot._scatter_signatures[layer_key] = cache_signature
            plot._scatter_index_signatures[layer_key] = index_signature
        plot._plot_full_detail = bool(full_detail)
        self._apply_plot_annotations(plot, dataset, full_detail)

    def _plot_dataset_on_axes(self, plot, dataset, full_detail: bool):
        total_t0 = time.perf_counter()
        if not full_detail:
            self._render_plot(plot, dataset, False)
            _vispy_perf_log(
                "plot.dataset",
                title=getattr(dataset, "title", ""),
                detail="preview",
                path="preview",
                total_ms=f"{_elapsed_ms(total_t0):.3f}",
            )
            return
        plot.parity_mode = bool(getattr(dataset, "parity_mode", dataset.title != "descriptor"))
        plot.title = dataset.title
        display_title = str(getattr(dataset, "display_title", dataset.title) or dataset.title)
        if display_title != plot.title:
            plot.title_label._text_visual.text = display_title

        signature_t0 = time.perf_counter()
        layer_key, brush, pen, marker_size, cache_signature, index_signature = self._plot_cache_signature(plot, dataset, full_detail)
        signature_ms = _elapsed_ms(signature_t0)
        if (
            full_detail
            and plot._scatter_signatures.get(layer_key) == cache_signature
            and plot._scatter_index_signatures.get(layer_key) == index_signature
            and plot.activate_scatter_layer(layer_key, cache_signature=cache_signature, index_signature=index_signature)
        ):
            plot._plot_full_detail = bool(full_detail)
            _vispy_perf_log(
                "plot.dataset",
                title=getattr(dataset, "title", ""),
                detail="full" if full_detail else "thumbnail",
                path="reuse_full",
                signature_ms=f"{signature_ms:.3f}",
                total_ms=f"{_elapsed_ms(total_t0):.3f}",
            )
            return

        index_t0 = time.perf_counter()
        indices = self._active_point_indices(dataset)
        index_ms = _elapsed_ms(index_t0)
        activate_t0 = time.perf_counter()
        if plot.activate_scatter_layer(
            layer_key,
            cache_signature=cache_signature,
            indices=indices,
            index_signature=index_signature,
        ):
            plot._plot_full_detail = bool(full_detail)
            _vispy_perf_log(
                "plot.dataset",
                title=getattr(dataset, "title", ""),
                detail="full" if full_detail else "thumbnail",
                path="reuse_layer",
                active_indices=0 if indices is None else int(indices.size),
                signature_ms=f"{signature_ms:.3f}",
                index_ms=f"{index_ms:.3f}",
                activate_ms=f"{_elapsed_ms(activate_t0):.3f}",
                total_ms=f"{_elapsed_ms(total_t0):.3f}",
            )
            return

        arrays_t0 = time.perf_counter()
        x, y, structure_index = self._full_plot_arrays(dataset)
        arrays_ms = _elapsed_ms(arrays_t0)
        scatter_t0 = time.perf_counter()
        plot.scatter(
            x,
            y,
            data=structure_index,
            brush=brush,
            pen=pen,
            symbol='o',
            size=marker_size,
            layer_key=layer_key,
            cache_signature=cache_signature,
            indices=indices,
            index_signature=index_signature,
        )
        scatter_ms = _elapsed_ms(scatter_t0)
        plot._plot_full_detail = bool(full_detail)
        _vispy_perf_log(
            "plot.dataset",
            title=getattr(dataset, "title", ""),
            detail="full" if full_detail else "thumbnail",
            path="upload",
            points=np.asarray(x).size,
            active_indices=0 if indices is None else int(indices.size),
            signature_ms=f"{signature_ms:.3f}",
            index_ms=f"{index_ms:.3f}",
            arrays_ms=f"{arrays_ms:.3f}",
            scatter_ms=f"{scatter_ms:.3f}",
            total_ms=f"{_elapsed_ms(total_t0):.3f}",
        )

    def _refresh_plot_detail(self, plot):
        if plot is None or self.nep_result_data is None or plot not in self.axes_list:
            return
        dataset = self.get_axes_dataset(plot)
        if dataset is None:
            return
        full_detail = plot is self.current_axes
        if plot._plot_full_detail == full_detail:
            return
        self._plot_dataset_on_axes(plot, dataset, full_detail)

    def auto_range(self):
        """Delegate auto-ranging to the currently active axes.
        """
        self.current_axes.auto_range()


    def pan(self ,checked):
        """Enable or disable panning mode on the active axes.
        
        Parameters
        ----------
        checked : bool
            Whether panning should be enabled.
        """
        self.current_axes.view.camera.interactive = checked



    def pen(self, checked):
        """Toggle polygon-drawing mode used for lasso selection.
        
        Parameters
        ----------
        checked : bool
            ``True`` to begin collecting polygon vertices, ``False`` to cancel.
        """
        if self.current_axes is None:
            return False

        if checked:
            self.draw_mode = True

        else:
            self.draw_mode = False
            pass

    @timeit

    def plot_nep_result(self):
        """Render all dataset scatter plots and refresh overlay layers.
        
        Notes
        -----
        Called after data mutations to keep the canvas in sync with the dataset.
        """
        self.nep_result_data.select_index.clear()
        self._ensure_plot_dataset_indices()
        # Clear all overlays so deleted selections do not persist visually
        for plot in self.axes_list:
            if hasattr(plot, 'clear_overlays'):
                plot.clear_overlays()
            if plot in self._selected_by_plot:
                self._selected_by_plot[plot].clear()
            if plot in self._show_by_plot:
                self._show_by_plot[plot].clear()
            if plot in self._loaded_by_plot:
                self._loaded_by_plot[plot].clear()
            if plot in self._reject_by_plot:
                self._reject_by_plot[plot].clear()

        for plot_index, dataset_index in enumerate(self._plot_dataset_indices):
            if plot_index >= len(self.axes_list) or dataset_index >= len(self.nep_result_data.datasets):
                continue
            _dataset = self.nep_result_data.datasets[dataset_index]
            plot=self.axes_list[plot_index]


            self._plot_dataset_on_axes(plot, _dataset, plot is self.current_axes)

            # continue
            if _dataset.group_array.num !=0:
                if self.structure_index not in _dataset.group_array.now_data:
                    self.structure_index=_dataset.group_array.now_data[0]
                    self.structureIndexChanged.emit(self.structure_index)

            else:
                plot.set_current_point([], [])

            is_current_plot = plot is self.current_axes
            show_rmse = is_current_plot and bool(getattr(_dataset, "show_rmse", _dataset.title not in ["descriptor"]))
            rmse_text = f"rmse: {_dataset.get_formart_rmse()}" if show_rmse else ""
            plot.set_rmse_text(rmse_text if is_current_plot else "")
            if show_rmse and is_current_plot:
                pos=self.convert_pos(plot,(0.1 ,0.8))
                if plot.text is not None:
                    plot.text.pos=pos
            else:
                plot.set_rmse_text("")

            x_label = getattr(_dataset, "x_label", None)
            y_label = getattr(_dataset, "y_label", None)
            plot.set_axis_labels(x_label, y_label)

        # Restore reject highlights after a full replot.
        reject = getattr(self.nep_result_data, "reject_index", None)
        if reject:
            self.set_reject_highlight(list(reject), True)
        self.prewarm_overlay_position_cache()

    def _refresh_current_axes_annotations(self):
        """Refresh labels and RMSE text after promoting a thumbnail to the main plot."""
        plot = self.current_axes
        dataset = self.get_axes_dataset(plot)
        if plot is None or dataset is None:
            return

        plot.set_axis_labels(getattr(dataset, "x_label", None), getattr(dataset, "y_label", None))
        if bool(getattr(dataset, "show_rmse", dataset.title not in ["descriptor"])):
            plot.set_rmse_text(f"rmse: {dataset.get_formart_rmse()}")
            if plot.text is not None:
                plot.text.pos = self.convert_pos(plot, (0.1, 0.8))
        else:
            plot.set_rmse_text("")

    def convert_pos(self,plot,pos):
        """Convert a relative position tuple to view coordinates.
        
        Parameters
        ----------
        plot : ViewBoxWidget
            Plot providing axis domains.
        pos : Tuple[float, float]
            Relative x/y positions in the range ``[0, 1]``.
        
        Returns
        -------
        tuple[float, float]
            Absolute coordinate in plot space.
        """
        x_range = plot.xaxis.axis.domain
        y_range = plot.yaxis.axis.domain

        x_percent = pos[0]
        y_percent =  pos[1]

        x_pos = x_range[0] + x_percent * (x_range[1] - x_range[0])
        y_pos = y_range[0] + y_percent * (y_range[1] - y_range[0])
        return x_pos,y_pos
    def plot_current_point(self,structure_index):
        """Highlight the selected structure across all axes.
        
        Parameters
        ----------
        structure_index : int
            Structure index to highlight.
        """
        self.structure_index=structure_index
        for plot in  self.axes_list :
            dataset=self.get_axes_dataset(plot)
            array_index=dataset.convert_index(structure_index)
            if dataset.is_visible(array_index) :

                data=dataset.all_data[array_index,: ]
                plot.set_current_point(data[:,dataset.x_cols].flatten(),
                                       data[:, dataset.y_cols].flatten(),
                                       )
            else:
                plot.set_current_point([], [])

    def _overlay_cache_signature(self, dataset):
        data = getattr(dataset, "data", None)
        group_array = getattr(dataset, "group_array", None)
        raw_x = getattr(dataset, "__dict__", {}).get("x")
        raw_y = getattr(dataset, "__dict__", {}).get("y")
        data_array = getattr(data, "all_data", None)
        group_array_data = getattr(group_array, "all_data", None)
        x_source = data_array if data_array is not None else raw_x
        y_source = data_array if data_array is not None else raw_y
        return (
            id(dataset),
            id(x_source),
            id(y_source),
            id(group_array_data),
            getattr(data, "version", 0),
            getattr(group_array, "version", 0),
            np.shape(x_source if x_source is not None else ()),
            np.shape(y_source if y_source is not None else ()),
            np.shape(group_array_data if group_array_data is not None else ()),
            int(getattr(dataset, "cols", 0) or 0),
            int(getattr(dataset, "_plot_coord_version", getattr(dataset, "_content_version", 0)) or 0),
        )

    def _overlay_position_lookup(self, dataset):
        signature = self._overlay_cache_signature(dataset)
        cached = self._overlay_position_cache.get(signature)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        if hasattr(dataset, "now_data") and hasattr(dataset, "x_cols") and hasattr(dataset, "y_cols"):
            rows = np.asarray(dataset.now_data)
            cols = int(getattr(dataset, "cols", 0) or 0)
            if cols == 0:
                x = rows.reshape(-1)
                y = x
            else:
                x = rows[:, dataset.x_cols].ravel()
                y = rows[:, dataset.y_cols].ravel()
        else:
            x = np.asarray(dataset.x)
            y = np.asarray(dataset.y)
        group_array = getattr(dataset, "group_array", None)
        row_groups = getattr(group_array, "now_data", None)
        cols = int(getattr(dataset, "cols", 0) or 0)
        if row_groups is not None and cols > 0:
            row_groups = np.asarray(row_groups, dtype=np.int64).reshape(-1)
            count = min(x.size, y.size, row_groups.size * cols)
            rows = count // cols
            row_groups = row_groups[:rows]
            sidx = row_groups
            scale = cols
            x = x[: rows * cols]
            y = y[: rows * cols]
        else:
            sidx = np.asarray(dataset.structure_index, dtype=np.int64)
            count = min(x.size, y.size, sidx.size)
            x = x[:count]
            y = y[:count]
            sidx = sidx[:count]
            scale = 1

        if sidx.size == 0:
            lookup = {
                "x": x,
                "y": y,
                "unique": np.empty(0, dtype=np.int64),
                "starts": np.empty(0, dtype=np.int64),
                "counts": np.empty(0, dtype=np.int64),
                "scale": scale,
                "order": None,
            }
        else:
            is_sorted = bool(np.all(sidx[:-1] <= sidx[1:])) if sidx.size > 1 else True
            if is_sorted:
                unique, starts, counts = np.unique(sidx, return_index=True, return_counts=True)
                order = None
            else:
                order = np.argsort(sidx, kind="stable")
                sorted_sidx = sidx[order]
                unique, starts, counts = np.unique(sorted_sidx, return_index=True, return_counts=True)
            lookup = {
                "x": x,
                "y": y,
                "unique": unique,
                "starts": starts.astype(np.int64, copy=False),
                "counts": counts.astype(np.int64, copy=False),
                "scale": scale,
                "order": order,
            }

        dataset_id = id(dataset)
        for key in list(self._overlay_position_cache):
            if key[0] == dataset_id and key != signature:
                self._overlay_position_cache.pop(key, None)
        self._overlay_position_cache[signature] = lookup
        _vispy_perf_log(
            "overlay.cache_build",
            points=count,
            structures=lookup["unique"].size,
            scale=scale,
            sorted=lookup["order"] is None,
            ms=f"{_elapsed_ms(t0):.3f}",
        )
        return lookup

    def prewarm_overlay_position_cache(self):
        if self.nep_result_data is None:
            return

        total_t0 = time.perf_counter()
        warmed = 0
        overlay_size = Config.getint("widget", "vispy_marker_size", 6) or 6
        empty = np.empty((0, 2), dtype=np.float32)
        for plot in self.axes_list:
            dataset = self.get_axes_dataset(plot)
            if dataset is None:
                continue
            t0 = time.perf_counter()
            self._overlay_position_lookup(dataset)
            if getattr(plot, "_scatter", None):
                plot.set_overlay_positions("loaded", empty, color=Brushes.LoadedOverlay, size=overlay_size)
                plot.set_overlay_positions("show", empty, color=Brushes.Show, size=overlay_size)
                plot.set_overlay_positions("selected", empty, color=Brushes.Selected, size=overlay_size)
            warmed += 1
            _vispy_perf_log(
                "overlay.cache_prewarm_plot",
                title=getattr(dataset, "title", ""),
                ms=f"{_elapsed_ms(t0):.3f}",
            )
        _vispy_perf_log(
            "overlay.cache_prewarm",
            plots=warmed,
            total_ms=f"{_elapsed_ms(total_t0):.3f}",
        )

    def _overlay_positions_for_indices(self, dataset, indices:set[int]):
        if not indices:
            return np.empty((0, 2), dtype=np.float32)

        t0 = time.perf_counter()
        lookup = self._overlay_position_lookup(dataset)
        unique = lookup["unique"]
        if unique.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        indices_arr = np.fromiter(indices, dtype=np.int64)
        if lookup["order"] is None and indices_arr.size > unique.size * 0.75:
            selected_mask = np.zeros(unique.size, dtype=bool)
            positions = np.searchsorted(unique, indices_arr)
            in_bounds = positions < unique.size
            positions = positions[in_bounds]
            valid_indices = indices_arr[in_bounds]
            valid = unique[positions] == valid_indices
            selected_mask[positions[valid]] = True

            starts = lookup["starts"]
            counts = lookup["counts"]
            scale = int(lookup.get("scale", 1) or 1)
            point_counts = counts * scale
            total_points = int(np.sum(point_counts))
            point_mask = np.ones(total_points, dtype=bool)
            missing = np.nonzero(~selected_mask)[0]
            if missing.size:
                point_starts = starts[missing] * scale
                for start, count in zip(point_starts, point_counts[missing]):
                    point_mask[int(start):int(start + count)] = False
            pos = np.column_stack([lookup["x"][point_mask], lookup["y"][point_mask]]).astype(np.float32, copy=False)
            _vispy_perf_log(
                "overlay.positions",
                requested=indices_arr.size,
                matched=int(np.count_nonzero(selected_mask)),
                points=pos.shape[0],
                mode="dense",
                ms=f"{_elapsed_ms(t0):.3f}",
            )
            return pos

        positions = np.searchsorted(unique, indices_arr)
        in_bounds = positions < unique.size
        valid = np.zeros(indices_arr.shape, dtype=bool)
        valid[in_bounds] = unique[positions[in_bounds]] == indices_arr[in_bounds]
        positions = positions[valid]
        if positions.size == 0:
            _vispy_perf_log(
                "overlay.positions",
                requested=indices_arr.size,
                matched=0,
                points=0,
                ms=f"{_elapsed_ms(t0):.3f}",
            )
            return np.empty((0, 2), dtype=np.float32)

        starts = lookup["starts"][positions]
        counts = lookup["counts"][positions]
        scale = int(lookup.get("scale", 1) or 1)
        total = int(np.sum(counts) * scale)
        order = lookup["order"]
        if order is None:
            point_starts = starts * scale
            point_counts = counts * scale
            if point_counts.size > 0 and bool(np.all(point_counts == point_counts[0])):
                offsets = np.arange(int(point_counts[0]), dtype=np.int64)
                result_indices = (point_starts[:, None] + offsets[None, :]).ravel()
            else:
                group_offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(point_counts) - point_counts, point_counts)
                result_indices = np.repeat(point_starts, point_counts) + group_offsets
        else:
            if counts.size > 0 and bool(np.all(counts == counts[0])):
                row_offsets = np.arange(int(counts[0]), dtype=np.int64)
                row_indices = (starts[:, None] + row_offsets[None, :]).ravel()
            else:
                row_total = int(np.sum(counts))
                row_offsets = np.arange(row_total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
                row_indices = np.repeat(starts, counts) + row_offsets
            row_indices = order[row_indices]
            if scale > 1:
                comp_offsets = np.arange(scale, dtype=np.int64)
                result_indices = (row_indices[:, None] * scale + comp_offsets[None, :]).ravel()
            else:
                result_indices = row_indices

        pos = np.column_stack([lookup["x"][result_indices], lookup["y"][result_indices]]).astype(np.float32, copy=False)
        _vispy_perf_log(
            "overlay.positions",
            requested=indices_arr.size,
            matched=positions.size,
            points=pos.shape[0],
            ms=f"{_elapsed_ms(t0):.3f}",
        )
        return pos

    def _preview_overlay_image_for_indices(self, plot, dataset, indices: set[int], brush):
        if getattr(plot, "_full_detail", False):
            return None
        base_image = getattr(plot, "_preview_image_source", None)
        preview_range = getattr(plot, "_preview_image_range", None)
        if base_image is None or preview_range is None:
            return None
        if not indices:
            return np.zeros_like(np.asarray(base_image, dtype=np.uint8))

        t0 = time.perf_counter()
        lookup = self._overlay_position_lookup(dataset)
        unique = lookup["unique"]
        if unique.size == 0:
            return np.zeros_like(np.asarray(base_image, dtype=np.uint8))

        indices_arr = np.fromiter(indices, dtype=np.int64)
        positions = np.searchsorted(unique, indices_arr)
        in_bounds = positions < unique.size
        positions = positions[in_bounds]
        valid_indices = indices_arr[in_bounds]
        valid = unique[positions] == valid_indices
        selected_mask = np.zeros(unique.size, dtype=bool)
        selected_mask[positions[valid]] = True
        matched = int(np.count_nonzero(selected_mask))
        ratio = matched / float(unique.size)
        if ratio < VISPY_PREVIEW_RASTER_OVERLAY_MIN_RATIO:
            return None

        base = np.asarray(base_image, dtype=np.uint8)
        image = np.zeros_like(base)
        color = np.asarray(plot.convert_color(brush), dtype=np.float32).reshape(-1)
        rgb = np.clip(color[:3] * 255, 0, 255).astype(np.uint8)
        alpha_factor = float(color[3]) if color.size >= 4 else 1.0
        image[..., :3] = rgb
        image[..., 3] = np.clip(base[..., 3].astype(np.float32) * max(0.35, alpha_factor), 0, 255).astype(np.uint8)

        missing = unique[~selected_mask]
        if missing.size:
            pos = self._overlay_positions_for_indices(dataset, set(int(v) for v in missing.tolist()))
            if pos.size:
                (x_range, y_range) = preview_range
                x_min, x_max = x_range
                y_min, y_max = y_range
                if x_max != x_min and y_max != y_min:
                    height, width = image.shape[:2]
                    x = pos[:, 0]
                    y = pos[:, 1]
                    mask = (x > -10000) & np.isfinite(x) & np.isfinite(y)
                    px = ((x[mask] - x_min) / (x_max - x_min) * (width - 1)).astype(np.int64)
                    py = ((y[mask] - y_min) / (y_max - y_min) * (height - 1)).astype(np.int64)
                    valid_px = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                    px = px[valid_px]
                    py = py[valid_px]
                    for dy in (0, 1):
                        yy = py + dy
                        y_ok = yy < height
                        if not np.any(y_ok):
                            continue
                        for dx in (0, 1):
                            xx = px[y_ok] + dx
                            x_ok = xx < width
                            if np.any(x_ok):
                                image[yy[y_ok][x_ok], xx[x_ok], 3] = 0

        _vispy_perf_log(
            "overlay.preview_image",
            title=getattr(dataset, "title", ""),
            requested=indices_arr.size,
            matched=matched,
            missing=int(missing.size),
            ratio=f"{ratio:.3f}",
            ms=f"{_elapsed_ms(t0):.3f}",
        )
        return image

    @timeit
    def update_scatter_color(self,structure_index,color=Brushes.Selected):
        # Switch to overlay layers so we don't reupload the entire base VBO
        """Update overlay colours to reflect the latest selection state.
        
        Parameters
        ----------
        structure_index : Sequence[int]
            Indices whose colours should be refreshed.
        color : Any, optional
            Brush applied to the selected points.
        """
        total_t0 = time.perf_counter()
        idx = np.atleast_1d(np.asarray(structure_index)).astype(np.int64)
        if idx.size == 0:
            return

        selected_global = set(getattr(self.nep_result_data, "select_index", set())) if self.nep_result_data is not None else set()
        idx_list = idx.tolist()
        idx_set = set(idx_list)
        total_positions_ms = 0.0
        total_overlay_ms = 0.0
        for plot in self.axes_list:
            plot_t0 = time.perf_counter()
            if not plot._scatter and getattr(plot, "_preview_image", None) is None:
                continue
            # init overlay sets for this plot
            if plot not in self._selected_by_plot:
                self._selected_by_plot[plot] = set()
            if plot not in self._show_by_plot:
                self._show_by_plot[plot] = set()
            if plot not in self._loaded_by_plot:
                self._loaded_by_plot[plot] = set()

            sets_t0 = time.perf_counter()
            dirty_layers = set()
            if color is Brushes.Default:
                # remove from both overlays
                self._selected_by_plot[plot].difference_update(idx_list)
                self._show_by_plot[plot].difference_update(idx_list)
                self._loaded_by_plot[plot].difference_update(idx_list)
                dirty_layers.update(("selected", "show", "loaded"))
            elif color is Brushes.Selected:
                # add to selected, remove from show to avoid duplicates
                show_dirty = bool(self._show_by_plot[plot].intersection(idx_set))
                loaded_dirty = bool(self._loaded_by_plot[plot].intersection(idx_set))
                self._selected_by_plot[plot].update(idx_list)
                self._show_by_plot[plot].difference_update(idx_list)
                self._loaded_by_plot[plot].difference_update(idx_list)
                dirty_layers.add("selected")
                if show_dirty:
                    dirty_layers.add("show")
                if loaded_dirty:
                    dirty_layers.add("loaded")
            elif color is Brushes.Show:
                loaded_dirty = bool(self._loaded_by_plot[plot].intersection(idx_set))
                self._show_by_plot[plot].update(idx_list)
                self._loaded_by_plot[plot].difference_update(idx_list)
                dirty_layers.add("show")
                if loaded_dirty:
                    dirty_layers.add("loaded")
            elif color is Brushes.LoadedOverlay:
                show_dirty = bool(self._show_by_plot[plot].intersection(idx_set))
                self._loaded_by_plot[plot].update(idx_list)
                self._show_by_plot[plot].difference_update(idx_list)
                dirty_layers.add("loaded")
                if show_dirty:
                    dirty_layers.add("show")
            else:
                # Fallback: treat as selected
                self._selected_by_plot[plot].update(idx_list)
                dirty_layers.add("selected")
            sets_ms = _elapsed_ms(sets_t0)

            dataset = self.get_axes_dataset(plot)
            if dataset is None:
                continue

            # Update overlays (filled squares, no edges for perf)
            overlay_size = Config.getint("widget", "vispy_marker_size", 6) or 6
            if not getattr(plot, "_full_detail", False):
                overlay_size = max(3, int(overlay_size * 0.65))
            position_ms = 0.0
            overlay_t0 = time.perf_counter()

            def _refresh_overlay(name, indices, brush, symbol='o'):
                nonlocal position_ms
                pos_t0 = time.perf_counter()
                if symbol == 'o' and not getattr(plot, "_full_detail", False):
                    preview_range = getattr(plot, "_preview_image_range", None)
                    if not indices and preview_range is not None:
                        position_ms += _elapsed_ms(pos_t0)
                        plot.set_overlay_image(name, None, *preview_range)
                        return
                    image = self._preview_overlay_image_for_indices(plot, dataset, indices, brush)
                    if image is not None:
                        position_ms += _elapsed_ms(pos_t0)
                        plot.set_overlay_image(name, image, *preview_range)
                        return
                pos = self._overlay_positions_for_indices(dataset, indices)
                position_ms += _elapsed_ms(pos_t0)
                if pos.size:
                    plot.set_overlay_positions(name, pos, color=brush, size=overlay_size, symbol=symbol)
                else:
                    plot.set_overlay_positions(name, np.empty((0, 2), dtype=np.float32), color=brush, size=overlay_size, symbol=symbol)

            if "selected" in dirty_layers:
                _refresh_overlay("selected", self._selected_by_plot[plot], Brushes.Selected)
            if "show" in dirty_layers:
                _refresh_overlay("show", self._show_by_plot[plot], Brushes.Show)
            if "loaded" in dirty_layers:
                _refresh_overlay("loaded", self._loaded_by_plot[plot], Brushes.LoadedOverlay)

            # Keep reject overlay on top of show but below selected.
            reject_set = self._reject_by_plot.get(plot, set())
            display_reject = reject_set - selected_global if reject_set else set()
            if reject_set and bool(reject_set.intersection(idx_set)):
                _refresh_overlay("reject", display_reject, Brushes.Reject, symbol='x')
            overlay_ms = _elapsed_ms(overlay_t0)
            total_positions_ms += position_ms
            total_overlay_ms += overlay_ms
            _vispy_perf_log(
                "overlay.update_plot",
                title=getattr(dataset, "title", ""),
                changed=idx.size,
                dirty=",".join(sorted(dirty_layers)) or "-",
                selected=len(self._selected_by_plot[plot]),
                show=len(self._show_by_plot[plot]),
                loaded=len(self._loaded_by_plot[plot]),
                reject=len(display_reject),
                sets_ms=f"{sets_ms:.3f}",
                positions_ms=f"{position_ms:.3f}",
                overlay_ms=f"{overlay_ms:.3f}",
                total_ms=f"{_elapsed_ms(plot_t0):.3f}",
            )
        _vispy_perf_log(
            "overlay.update_total",
            changed=idx.size,
            plots=len(self.axes_list),
            positions_ms=f"{total_positions_ms:.3f}",
            overlay_ms=f"{total_overlay_ms:.3f}",
            total_ms=f"{_elapsed_ms(total_t0):.3f}",
        )

    def set_reject_highlight(self, structure_indices, enabled: bool) -> None:
        """Toggle the reject highlight overlay for the provided structure indices.

        Notes
        -----
        Reject highlighting is independent from selection. Selected points are
        removed from the reject overlay to keep selection visually dominant.
        """
        if self.nep_result_data is None:
            return

        idx = np.atleast_1d(np.asarray(structure_indices, dtype=np.int64)).ravel()
        if idx.size == 0:
            return

        selected = set(getattr(self.nep_result_data, "select_index", set()))
        indices = idx.tolist()

        for plot in self.axes_list:
            if not getattr(plot, "_scatter", None) and getattr(plot, "_preview_image", None) is None:
                continue
            if plot not in self._reject_by_plot:
                self._reject_by_plot[plot] = set()

            if enabled:
                self._reject_by_plot[plot].update(indices)
            else:
                self._reject_by_plot[plot].difference_update(indices)

            dataset = self.get_axes_dataset(plot)
            if dataset is None:
                continue

            overlay_size = Config.getint("widget", "vispy_marker_size", 6) or 6
            if not getattr(plot, "_full_detail", False):
                overlay_size = max(3, int(overlay_size * 0.65))

            display_reject = self._reject_by_plot[plot] - selected if selected else set(self._reject_by_plot[plot])
            if not display_reject:
                plot.set_overlay_positions(
                    "reject",
                    np.empty((0, 2), dtype=np.float32),
                    color=Brushes.Reject,
                    size=overlay_size,
                    symbol="x",
                )
                continue

            pos = self._overlay_positions_for_indices(dataset, display_reject)
            plot.set_overlay_positions(
                "reject",
                pos,
                color=Brushes.Reject,
                size=overlay_size,
                symbol="x",
            )

    def apply_overlay_groups(self, loaded_index, selected_index) -> None:
        """Apply read-only overlay coloring for a synthetic single-plot dataset."""
        if self.nep_result_data is None:
            return

        loaded_ids = {int(v) for v in np.atleast_1d(np.asarray(loaded_index, dtype=np.int64)).tolist()} if loaded_index is not None else set()
        selected_ids = {int(v) for v in np.atleast_1d(np.asarray(selected_index, dtype=np.int64)).tolist()} if selected_index is not None else set()

        for plot in self.axes_list:
            self._selected_by_plot.setdefault(plot, set()).clear()
            self._show_by_plot.setdefault(plot, set()).clear()
            self._loaded_by_plot.setdefault(plot, set()).clear()
            self._reject_by_plot.setdefault(plot, set())

        if loaded_ids:
            self.update_scatter_color(sorted(loaded_ids), Brushes.LoadedOverlay)
        if selected_ids:
            self.update_scatter_color(sorted(selected_ids), Brushes.Selected)


    def select_point_from_polygon(self,polygon_xy,reverse ):
        """Select points enclosed by the polygon drawn by the user.
        
        Parameters
        ----------
        polygon_xy : ndarray
            Polygon vertices expressed in view coordinates.
        reverse : bool
            When ``True`` remove the enclosed points from the selection.
        """
        total_t0 = time.perf_counter()
        arrays_t0 = time.perf_counter()
        x, y, structure_index = self._interaction_arrays_for_axes(self.current_axes)
        arrays_ms = _elapsed_ms(arrays_t0)
        if x is None or y is None or structure_index is None or structure_index.size == 0:
            _vispy_perf_log(
                "polygon.select",
                points=0,
                selected=0,
                arrays_ms=f"{arrays_ms:.3f}",
                total_ms=f"{_elapsed_ms(total_t0):.3f}",
            )
            return
        mask_t0 = time.perf_counter()
        polygon_xy = np.asarray(polygon_xy)
        px = polygon_xy[:, 0]
        py = polygon_xy[:, 1]
        xmin = np.min(px)
        xmax = np.max(px)
        ymin = np.min(py)
        ymax = np.max(py)
        mask = (x > -10000) & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        mask_ms = _elapsed_ms(mask_t0)
        if not np.any(mask):
            apply_t0 = time.perf_counter()
            self.select_index([], reverse)
            apply_ms = _elapsed_ms(apply_t0)
            _vispy_perf_log(
                "polygon.select",
                points=int(x.size),
                candidates=0,
                vertices=int(polygon_xy.shape[0]),
                selected=0,
                reverse=bool(reverse),
                arrays_ms=f"{arrays_ms:.3f}",
                mask_ms=f"{mask_ms:.3f}",
                stack_ms="0.000",
                pip_ms="0.000",
                unique_ms="0.000",
                apply_ms=f"{apply_ms:.3f}",
                total_ms=f"{_elapsed_ms(total_t0):.3f}",
            )
            return
        stack_t0 = time.perf_counter()
        points = np.column_stack([x[mask], y[mask]])
        stack_ms = _elapsed_ms(stack_t0)
        pip_t0 = time.perf_counter()
        selected = self.is_point_in_polygon(points, polygon_xy)
        pip_ms = _elapsed_ms(pip_t0)
        unique_t0 = time.perf_counter()
        select_index=np.unique(structure_index[mask][selected]).astype(np.int64).tolist()
        unique_ms = _elapsed_ms(unique_t0)
        apply_t0 = time.perf_counter()
        self.select_index(select_index,reverse)
        apply_ms = _elapsed_ms(apply_t0)
        _vispy_perf_log(
            "polygon.select",
            points=int(x.size),
            candidates=int(points.shape[0]),
            vertices=int(polygon_xy.shape[0]),
            selected=len(select_index),
            reverse=bool(reverse),
            arrays_ms=f"{arrays_ms:.3f}",
            mask_ms=f"{mask_ms:.3f}",
            stack_ms=f"{stack_ms:.3f}",
            pip_ms=f"{pip_ms:.3f}",
            unique_ms=f"{unique_ms:.3f}",
            apply_ms=f"{apply_ms:.3f}",
            total_ms=f"{_elapsed_ms(total_t0):.3f}",
        )
