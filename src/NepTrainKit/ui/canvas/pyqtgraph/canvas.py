"""PyQtGraph-based canvas widgets for interactive structure exploration.
"""

from functools import partial

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from pyqtgraph import GraphicsLayoutWidget, ScatterPlotItem, PlotItem, ViewBox, TextItem

from NepTrainKit.utils import timeit
from NepTrainKit.core.types import Brushes, Pens
from NepTrainKit.config import Config
from ..base.canvas import CanvasLayoutBase
from NepTrainKit.core.io import NepTrainResultData


class MyPlotItem(PlotItem):
    """Plot item that bundles scatter data, annotations, and current-point markers.
    """

    def __init__(self, **kwargs):
        """Initialise the plot item with scatter, annotation, and current-point handles.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to :class:`pyqtgraph.PlotItem`.
        """
        super().__init__(**kwargs)
        self.disableAutoRange()

        self._scatter = ScatterPlotItem()
        self.addItem(self._scatter)

        self.text = TextItem(color=(231, 63, 50))

        self.addItem(self.text)

        self.current_point = ScatterPlotItem()
        self.current_point.setZValue(100)
        if "title" in kwargs:
            self.setTitle(kwargs["title"])

    def scatter(self, *args, **kargs):
        """Update the underlying scatter plot data.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`pyqtgraph.ScatterPlotItem.setData`.
        **kargs : dict
            Keyword arguments forwarded to :meth:`pyqtgraph.ScatterPlotItem.setData`.
        """
        self._scatter.setData(*args, **kargs)

    def set_current_point(self, x, y):
        """Display a highlighted marker for the active structure.
        
        Parameters
        ----------
        x : array-like
            X coordinates of the marker.
        y : array-like
            Y coordinates of the marker.
        """
        current_size = Config.getint("plot", "current_marker_size", 20) or 20
        self.current_point.setData(x, y, brush=Brushes.Current, pen=Pens.Current,
                                   symbol='star', size=current_size)
        if self.current_point not in self.items:
            self.addItem(self.current_point)

    def add_diagonal(self):
        """Draw a unit diagonal line used for parity-style plots.
        """
        self.addLine(angle=45, pos=(0.5, 0.5), pen=Pens.Line)

    def item_clicked(self, scatter_item, items, event):
        """Emit the selected structure index when a scatter point is clicked.
        
        Parameters
        ----------
        scatter_item : ScatterPlotItem
            Item that produced the signal.
        items : Sequence
            Collection of clicked point handles.
        event : Any
            Mouse event provided by PyQtGraph.
        """
        if items.any():
            item = items[0]

            self.structureIndexChanged.emit(item.data())

    @property
    def title(self):
        """Get the current axes title.
        
        Returns
        -------
        str
            Active title text.
        """
        return self.titleLabel.text

    @title.setter
    def title(self, t):
        """Set the axes title and keep auxiliary overlays in sync.
        
        Parameters
        ----------
        t : str
            New title text.
        """
        if t == self.title:
            return
        self.setTitle(t)
        if t != "descriptor":
            self.add_diagonal()

    @property
    def rmse_size(self):
        """Get the font size used for RMSE annotations.
        
        Returns
        -------
        int
            Point size of the RMSE text label.
        """
        return self.text.textItem.font().pointSize()
    @rmse_size.setter
    def rmse_size(self,size):
        """Set the font size used for RMSE annotations.
        
        Parameters
        ----------
        size : int
            Point size applied to the RMSE label.
        """
        self.text.setFont(QFont("Arial",size))
class CombinedMeta(type(CanvasLayoutBase), type(GraphicsLayoutWidget)):
    """Metaclass bridging ``CanvasLayoutBase`` with ``GraphicsLayoutWidget``.
    """
    pass


class PyqtgraphCanvas(CanvasLayoutBase, GraphicsLayoutWidget, metaclass=CombinedMeta):
    """Canvas implementation backed by :class:`pyqtgraph.GraphicsLayoutWidget`.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the layout widget and shared canvas state.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`pyqtgraph.GraphicsLayoutWidget`.
        **kwargs : dict
            Keyword arguments forwarded to :class:`pyqtgraph.GraphicsLayoutWidget`.
        """
        GraphicsLayoutWidget.__init__(self, *args, **kwargs)

        CanvasLayoutBase.__init__(self)
        self.nep_result_data = None

    def set_nep_result_data(self, dataset):
        """Attach a NepTrain result dataset to the canvas.
        
        Parameters
        ----------
        dataset : NepTrainResultData
            Dataset used for plotting and interaction.
        """
        self.nep_result_data: NepTrainResultData = dataset

    def clear_axes(self):
        """Remove all axes from the layout and the cached axis list.
        """
        self.clear()

        super().clear_axes()

    def init_axes(self, axes_num):
        """Create the requested number of axes.
        
        Parameters
        ----------
        axes_num : int
            Number of subplots to allocate.
        """
        self.clear_axes()

        for r in range(axes_num):
            plot = MyPlotItem(title="")
            self.addItem(plot)
            plot.getViewBox().mouseDoubleClickEvent = partial(self.view_on_double_clicked, plot=plot)
            plot.getViewBox().setMouseEnabled(False, False)
            self.axes_list.append(plot)

            plot._scatter.sigClicked.connect(self.item_clicked)

        self.set_view_layout()

    def view_on_double_clicked(self, event, plot):
        """Focus an axes when the user double-clicks it.
        
        Parameters
        ----------
        event : QMouseEvent
            Double-click event provided by PyQtGraph.
        plot : MyPlotItem
            Plot item that should become active.
        """
        self.set_current_axes(plot)

    def set_view_layout(self):
        """Arrange axes so the active plot spans the first row and others share the second.
        """
        if len(self.axes_list) == 0:
            return
        if self.current_axes not in self.axes_list:
            self.set_current_axes(self.axes_list[0])
            return

        self.ci.clear()
        self.addItem(self.current_axes, row=0, col=0, colspan=4)
        self.current_axes.rmse_size = 12
        # Place the remaining plots on the second row.
        other_plots = [p for p in self.axes_list if p != self.current_axes]
        for i, other_plot in enumerate(other_plots):
            self.addItem(other_plot, row=1, col=i)
            other_plot.rmse_size = 6

        for col, factor in enumerate([3, 1]):
            self.ci.layout.setRowStretchFactor(col, factor)

    @timeit
    def plot_nep_result(self):
        """Render all dataset scatter plots and their annotations.
        
        Notes
        -----
        Invoked after data mutations (delete, undo, reload) to refresh the canvas.
        """
        self.nep_result_data.select_index.clear()

        for index, _dataset in enumerate(self.nep_result_data.datasets):
            plot = self.axes_list[index]
            plot.title = _dataset.title
            pg_size = Config.getint("widget", "pg_marker_size", 7) or 7
            plot.scatter(
                _dataset.x,
                _dataset.y,
                data=_dataset.structure_index,
                brush=Brushes.get(_dataset.title.upper()),
                pen=Pens.get(_dataset.title.upper()),
                symbol='o',
                size=pg_size,
            )

            # Update the view box range after plotting.
            self.auto_range(plot)
            if _dataset.group_array.num != 0:
                # Update the structure selection.
                if self.structure_index not in _dataset.group_array.now_data:
                    self.structure_index = _dataset.group_array.now_data[0]
                    self.structureIndexChanged.emit(self.structure_index)
            else:
                plot.set_current_point([], [])

            if _dataset.title not in ["descriptor"]:
                # Update RMSE overlays for non-descriptor plots.
                pos = self.convert_pos(plot, (0, 1))
                text = f"rmse: {_dataset.get_formart_rmse()}"
                plot.text.setText(text)
                plot.text.setPos(*pos)

    def plot_current_point(self, structure_index):
        """Highlight the selected structure across all axes.
        
        Parameters
        ----------
        structure_index : int
            Structure index to highlight.
        """
        self.structure_index = structure_index

        for plot in self.axes_list:
            dataset = self.get_axes_dataset(plot)
            array_index = dataset.convert_index(structure_index)
            if dataset.is_visible(array_index) :

                data=dataset.all_data[array_index,: ]
                plot.set_current_point(data[:,dataset.x_cols].flatten(),
                                       data[:, dataset.y_cols].flatten(),
                                       )
            else:
                plot.set_current_point([], [])
    def item_clicked(self, scatter_item, items, event):
        """Handle scatter click events and propagate the selected structure index.
        
        Parameters
        ----------
        scatter_item : ScatterPlotItem
            Item emitting the signal.
        items : Sequence
            Collection of clicked point handles.
        event : Any
            Mouse event provided by PyQtGraph.
        """
        if items.any():
            item = items[0]

            self.structureIndexChanged.emit(item.data())

    def select_point_from_polygon(self, polygon_xy, reverse):
        """Select points enclosed by a user-drawn polygon.
        
        Parameters
        ----------
        polygon_xy : ndarray
            Polygon vertices expressed in plot coordinates.
        reverse : bool
            When ``True`` remove the enclosed points from the selection.
        """
        index = self.is_point_in_polygon(
            np.column_stack([self.current_axes._scatter.data["x"], self.current_axes._scatter.data["y"]]), polygon_xy)
        index = np.where(index)[0]
        select_index = self.current_axes._scatter.data[index]["data"].tolist()
        self.select_index(select_index, reverse)

    def select_point(self, pos, reverse):
        """Select a single point at the provided scene position.
        
        Parameters
        ----------
        pos : QPointF
            Scene position to map back to data coordinates.
        reverse : bool
            When ``True`` toggle the current selection state off.
        """
        items = self.current_axes._scatter.pointsAt(pos)
        if len(items):
            item = items[0]
            index = item.index()
            structure_index = item.data()
            self.select_index(structure_index, reverse)
    @timeit
    def update_scatter_color(self, structure_index, color=Brushes.Selected):
        """Update scatter colours to reflect the latest selection state.
        
        Parameters
        ----------
        structure_index : Sequence[int]
            Indices whose colours should be refreshed.
        color : Any
            Brush applied to the selected points.
        """

        for i, plot in enumerate(self.axes_list):

            if not plot._scatter:
                continue
            structure_index_set = set(structure_index)
            index_list = [i for i, val in enumerate(plot._scatter.data["data"]) if val in structure_index_set]

            plot._scatter.data["brush"][index_list] = color
            plot._scatter.data['sourceRect'][index_list] = (0, 0, 0, 0)

            plot._scatter.updateSpots()

    def convert_pos(self, plot, pos):
        """Convert a relative position tuple to plot coordinates.
        
        Parameters
        ----------
        plot : MyPlotItem
            Plot providing the view range.
        pos : Tuple[float, float]
            Relative x/y positions in the range ``[0, 1]``.
        
        Returns
        -------
        tuple of float
            Converted coordinate in plot space.
        """
        view_range = plot.viewRange()
        x_range = view_range[0]  # x-axis span [xmin, xmax]
        y_range = view_range[1]  # y-axis span [ymin, ymax]
        y_range = view_range[1]
        # Convert the percentage position into data coordinates.
        x_percent = pos[0]  # 50% corresponds to the midpoint of the x-axis.
        y_percent = pos[1]  # Relative y position expressed as a percentage.
        y_percent = pos[1]
        x_pos = x_range[0] + x_percent * (x_range[1] - x_range[0])  # Compute the absolute x coordinate.
        y_pos = y_range[0] + y_percent * (y_range[1] - y_range[0])  # Compute the absolute y coordinate.
        y_pos = y_range[0] + y_percent * (y_range[1] - y_range[0])
        return x_pos, y_pos

    def auto_range(self, plot=None):
        """Auto-scale the specified plot or the currently active axes.
        
        Parameters
        ----------
        plot : MyPlotItem, optional
            Plot to rescale. Defaults to the active axes when ``None``.
        """
        if plot is None:
            plot = self.current_axes
        if plot:

            view = plot.getViewBox()

            x_range = [10000, -10000]
            y_range = [10000, -10000]
            for item in view.addedItems:
                if isinstance(item, ScatterPlotItem):

                    x = item.data["x"]
                    y = item.data["y"]

                    x = x[x > -10000]
                    y = y[y > -10000]
                    if x.size == 0:
                        x_range = [0, 1]
                        y_range = [0, 1]
                        continue
                    x_min = np.min(x)
                    x_max = np.max(x)
                    y_min = np.min(y)
                    y_max = np.max(y)
                    if x_min < x_range[0]:
                        x_range[0] = x_min
                    if x_max > x_range[1]:
                        x_range[1] = x_max
                    if y_min < y_range[0]:
                        y_range[0] = y_min
                    if y_max > y_range[1]:
                        y_range[1] = y_max
            if plot.title != "descriptor":

                real_range = (min(x_range[0], y_range[0]), max(x_range[1], y_range[1]))
                view.setRange(xRange=real_range, yRange=real_range)
            else:
                view.setRange(xRange=x_range, yRange=y_range)

    def pan(self, checked):
        """Enable or disable panning mode on the active axes.
        
        Parameters
        ----------
        checked : bool
            Whether panning should be enabled.
        """
        if self.current_axes:
            self.current_axes.setMouseEnabled(checked, checked)
            self.current_axes.getViewBox().setMouseMode(ViewBox.PanMode)

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
            # Reset mouse state and path buffers.
            self.is_drawing = False
            self.x_data = []
            self.y_data = []
        else:
            self.draw_mode = False

    def mousePressEvent(self, event):
        """Handle mouse press events for polygon drawing and default interaction.

        Parameters
        ----------
        event : QMouseEvent
            Mouse press event forwarded by Qt.
        """
        if not self.draw_mode:
            return super().mousePressEvent(event)

        if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
            self.is_drawing = True
            self.x_data.clear()
            self.y_data.clear()
            self.curve = self.current_axes.plot([], [], pen='r')
            self.curve.setData([], [])

    def mouseReleaseEvent(self, event):
        """Complete polygon selection or defer to the default handler.

        Parameters
        ----------
        event : QMouseEvent
            Mouse release event forwarded by Qt.
        """
        if not self.draw_mode:
            return super().mouseReleaseEvent(event)
        if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
            self.is_drawing = False
            reverse = event.button() == Qt.MouseButton.RightButton
            self.current_axes.removeItem(self.curve)
            if len(self.x_data) > 2:
                # Build a polygon from the captured mouse path.
                points = np.column_stack((self.x_data, self.y_data))
                self.select_point_from_polygon(points, reverse)
            else:
                # On right click, treat the interaction as a single-point selection.
                mouse_point = self.current_axes.getViewBox().mapSceneToView(event.pos())
                self.select_point(mouse_point, reverse)
            return

    def mouseMoveEvent(self, event):
        """Track the polygon path while the user is drawing.

        Parameters
        ----------
        event : QMouseEvent
            Mouse move event forwarded by Qt.
        """
        if not self.draw_mode:
            return super().mouseMoveEvent(event)

        if self.is_drawing:
            pos = event.pos()
            if self.current_axes.sceneBoundingRect().contains(pos):
                mouse_point = self.current_axes.getViewBox().mapSceneToView(pos)
                x, y = mouse_point.x(), mouse_point.y()
                self.x_data.append(x)
                self.y_data.append(y)
                self.curve.setData(self.x_data, self.y_data)
