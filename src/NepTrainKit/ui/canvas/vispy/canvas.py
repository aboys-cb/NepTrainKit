"""VisPy canvas widgets for interactive NepTrain result exploration.
"""

import os
os.environ["VISPY_IGNORE_OLD_VERSION"] = "true"

# os.environ["VISPY_PYQT5_SHARE_CONTEXT"] = "true"

import numpy as np

from PySide6.QtGui import QBrush, QColor, QPen
from vispy import scene

from vispy.visuals.filters import MarkerPickingFilter
from NepTrainKit.utils import timeit
from NepTrainKit.config import Config
from NepTrainKit.ui.canvas.base.canvas import VispyCanvasLayoutBase
from NepTrainKit.core.io import NepTrainResultData
from NepTrainKit.core.types import Brushes, Pens



class ViewBoxWidget(scene.Widget):

    """Composite widget combining axes, scatter visuals, and overlays for a single subplot.
    """
    def __init__(self, title, *args, **kwargs):
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

        self.yaxis = scene.AxisWidget(orientation='left',
                                 axis_width=1,
                                 # axis_label='Y Axis',
                                 # axis_font_size=12,
                                 # axis_label_margin=10,
                                 tick_label_margin=5,
                                 axis_color="black",
                                 text_color="black"
                                 )
        self.yaxis.width_max = 50
        self.grid.add_widget(self.yaxis, row=1, col=0)

        self.xaxis = scene.AxisWidget(orientation='bottom',
                                 axis_width=1,
                                 tick_label_margin=10,
                                 axis_color="black",
                                 text_color="black"

                                 )

        self.xaxis.height_max = 30
        self.grid.add_widget(self.xaxis, row=2, col=1)

        right_padding = self.grid.add_widget(row=1, col=2, row_span=1)
        right_padding.width_max = 5
        self._view = self.grid.add_view(row=1, col=1,  )

        self._view.camera = scene.cameras.PanZoomCamera()
        self._view.camera.interactive = False

        self.xaxis.link_view(self._view)
        self.yaxis.link_view(self._view)

        self.text=  scene.Text('', parent=self._view.scene, color='red',anchor_x="left", anchor_y="top" )
        self.text.font_size = 8


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
        # Overlay marker layers by name (e.g., 'selected', 'show')
        self._overlays = {}

        self._diagonal=None
        self.current_point=None
        self.freeze()



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

    def auto_range(self):
        """Auto-scale the pan/zoom camera to fit the scatter data.
        """
        if self._scatter is None:
            return

        pos = self._scatter._data["a_position"]
        if pos.size==0:
            return
        x_range = [10000, -10000]
        y_range = [10000, -10000]


        x = pos[:,0]
        y = pos[:,1]

        x = x[x > -10000]
        y = y[y > -10000]
        if x.size == 0:
            x_range =[0,1]
            y_range =[0,1]

        else:

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
        # self._view.camera.set_range( )
        #
        if "descriptor"!=self.title:
            real_range=(min(x_range[0],y_range[0]),max(x_range[1],y_range[1]))
            # Provide z-range to avoid VisPy querying scene bounds for z (empty visuals would error)
            self._view.camera.set_range(x=real_range, y=real_range, z=(0, 0))
        else:
            self._view.camera.set_range(x=x_range, y=y_range, z=(0, 0))

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
        self.current_point.set_data(
            np.vstack([x, y, z]).T,
            face_color=self.convert_color(Brushes.Current),
            edge_color=self.convert_color(Pens.Current),
            symbol='star',
            size=current_size,
        )

    def scatter(self,x,y,data,brush=None,pen=None ,**kwargs):
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
        if self._scatter is None:
            # Use configurable antialias level (default 0.5)
            self._scatter = scene.visuals.Markers(antialias=self.marker_antialias)
            self._scatter.order=1
            self._scatter.attach(self.picking_filter)

            self._view.add(self._scatter)

        self.data=data

        if brush is not None:

            kwargs["face_color"]=self.convert_color(brush)
        if pen is not None:

            kwargs["edge_color"]=self.convert_color(pen)
        if x.size != 0:

            pos = np.vstack([x, y], dtype=np.float32).T
            # Ensure a default size if caller didn't provide one
            if 'size' not in kwargs or kwargs.get('size') is None:
                kwargs['size'] = self.marker_size_default
            # self._scatter.update_gl_state(depth_test=False)
            self._scatter.set_data(pos, **kwargs)
            self.auto_range()
        else:
            self._scatter.set_data(np.empty((0, 3)), **kwargs)
            self.auto_range()
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
        x_domain = self.xaxis.axis.domain
        line_data = np.linspace(*x_domain,num=100)
        self._diagonal=self.line(line_data,line_data,**kwargs)

        self._diagonal.order=3
    def update_diagonal(self):
        """Update the parity diagonal to match the latest axis domain.
        """
        if self._diagonal is None:
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
        ov = scene.visuals.Markers(antialias=1)
        ov.order = 4  # above base scatter and diagonal
        # keep same scene/camera
        self._view.add(ov)
        # overlays are 2D; no need for depth test
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
        return ov

    def clear_overlays(self):
        """Hide and clear all overlay layers for this view.
        """
        empty = np.empty((0, 2), dtype=np.float32)
        for ov in self._overlays.values():
            ov.set_data(pos=empty, edge_width=0, symbol='o', size=9)
            ov.visible = False

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
        return self.text.font_size
    @rmse_size.setter
    def rmse_size(self,size):

        """Set the font size used for the RMSE annotation.
        
        Parameters
        ----------
        size : int
            Font size in points.
        """
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
        if t != "descriptor":
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


        self.grid = self.central_widget.add_grid(margin=0, spacing=0)
        self.grid.spacing = 0


        self.events.mouse_double_click.connect(self.switch_view_box)
        self.path_line = scene.visuals.Line(color='red', method='gl', antialias=False)
        self._path_update_step = 3
        # Use filters to affect the rendering of the mesh.

    def clear_axes(self):
        """Remove all ViewBox widgets from the grid and reset internal state.
        """
        for widget in self.axes_list:
            widget._stretch = (None, None)
            widget.parent=None
            self.grid.remove_widget(widget)

        super().clear_axes()


    def set_nep_result_data(self,dataset):
        """Attach a NepTrain result dataset to the canvas.
        
        Parameters
        ----------
        dataset : NepTrainResultData
            Dataset used for plotting and interaction.
        """
        self.nep_result_data:NepTrainResultData=dataset


    def point_at(self,pos):
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
        current_axes=self._get_clicked_axes(pos)
        if current_axes is None:
            return None
        # adjust the event position for hidpi screens
        render_size = tuple(int(round(d * self.pixel_scale)) for d in self.size)
        x_pos = int(round(pos[0] * self.pixel_scale))
        y_pos = int(round(render_size[1] - (pos[1] * self.pixel_scale)))
        # print(canvas.pixel_scale)
        # render a small patch around the mouse cursor
        restore_state = not current_axes.picking_filter.enabled
        current_axes.picking_filter.enabled = True
        # Temporarily hide overlays and current-point so picking only sees base scatter
        hidden = []
        if hasattr(current_axes, '_overlays'):
            for ov in current_axes._overlays.values():
                if ov is not None and ov.visible:
                    hidden.append(ov)
                    ov.visible = False
        if getattr(current_axes, 'current_point', None) is not None and current_axes.current_point.visible:
            hidden.append(current_axes.current_point)
            current_axes.current_point.visible = False
        # Also hide drawing helpers to avoid contaminating the pick render
        path_was_visible = False
        if getattr(self, 'path_line', None) is not None and self.path_line.parent is not None:
            path_was_visible = getattr(self.path_line, 'visible', True)
            self.path_line.visible = False
        diag_was_visible = False
        if getattr(current_axes, '_diagonal', None) is not None:
            diag_was_visible = getattr(current_axes._diagonal, 'visible', False)
            current_axes._diagonal.visible = False
        current_axes._scatter.update_gl_state(blend=False)
        picking_render = self.render(
            crop=(x_pos - 3, y_pos - 3, 7, 7),
            bgcolor=(0, 0, 0, 0),
            alpha=True,
        )
        # Restore previously hidden visuals
        for v in hidden:
            v.visible = True
        if getattr(self, 'path_line', None) is not None and self.path_line.parent is not None:
            self.path_line.visible = path_was_visible
        if getattr(current_axes, '_diagonal', None) is not None:
            current_axes._diagonal.visible = diag_was_visible
        if restore_state:
            current_axes.picking_filter.enabled = False
        current_axes._scatter.update_gl_state(blend=not current_axes.picking_filter.enabled)

        # unpack indices in patch and pick the nearest valid marker index
        patch = (picking_render.view(np.uint32) - 1)[:, :, 0]
        if current_axes.data.size == 0:
            return None
        # valid indices are in [0, data_size)
        valid_mask = patch < int(current_axes.data.size)
        if not np.any(valid_mask):
            return None
        # compute distance to center for valid pixels and choose nearest
        h, w = patch.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        # set invalid distances to large
        dist2 = np.where(valid_mask, dist2, 1e9)
        iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
        marker_idx = int(patch[iy, ix])
        return marker_idx
    def on_mouse_press(self, event):
        """Handle mouse press events for either picking or polygon drawing.
        
        Parameters
        ----------
        event : vispy.app.MouseEvent
            Mouse press event.
        """

        if not self.draw_mode:

            index = self.point_at(event.pos)

            current_axes = self._get_clicked_axes(event.pos)

            if index is not None:
                structure_index=current_axes.data[index]


                self.structureIndexChanged.emit(structure_index)

            return False

        if event.button == 1 or event.button ==2:
            if self.draw_mode:

                tr = self.scene.node_transform(self.current_axes.view.scene)
                x, y, _, _ = tr.map(event.pos)
                self.mouse_path = [[x, y]]
                self.path_line.set_data(pos=np.array(self.mouse_path))
                self.current_axes.view.add(self.path_line)


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
            tr = self.scene.node_transform(self.current_axes.view.scene)
            x, y, _, _ = tr.map(event.pos)

            self.mouse_path.append([x,y])

            if (len(self.mouse_path) % getattr(self, '_path_update_step', 3)) == 0:
                self.path_line.set_data(pos=np.array(self.mouse_path))

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
            reverse=event.button == 2

            self.path_line.parent=None

            if len(self.mouse_path)>2:

                self.select_point_from_polygon(np.array(self.mouse_path),reverse)
            else:
                index = self.point_at(event.pos)
                if index is not None:
                    structure_index = self.current_axes.data[index]

                    self.select_index(structure_index,reverse)

            self.mouse_path = []

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

        if isinstance(view, scene.ViewBox)  :
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
        if self.current_axes != axes:
            self.set_current_axes(axes)
            self.set_view_layout()

    def init_axes(self,axes_num   ):
        """Create the requested number of axes widgets.
        
        Parameters
        ----------
        axes_num : int
            Number of subplots to allocate.
        """
        self.clear_axes()
        for r in range(axes_num):
            plot = ViewBoxWidget(title="")
            self.axes_list.append(plot)
        self.set_view_layout()
        self.update()

    def set_view_layout(self):
        """Arrange axes so the active plot occupies the main area and others align below.
        """
        if len(self.axes_list)==0:
            return
        if self.current_axes not in self.axes_list:
            self.set_current_axes(self.axes_list[0])
            return

        i = 0
        row_0_col_span=len(self.axes_list)-1
        for widget in self.axes_list:
            widget._stretch = (None, None)
            self.grid.remove_widget(widget)

            if widget == self.current_axes:
                widget.rmse_size=8
                self.grid.add_widget(widget, row=0, col=0, row_span=6, col_span=row_0_col_span)
            else:
                widget.rmse_size=4

                self.grid.add_widget(widget, row=6, col=i, row_span=2, col_span=1)

                i += 1

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
        # Clear all overlays so deleted selections do not persist visually
        for plot in self.axes_list:
            if hasattr(plot, 'clear_overlays'):
                plot.clear_overlays()
            if plot in self._selected_by_plot:
                self._selected_by_plot[plot].clear()
            if plot in self._show_by_plot:
                self._show_by_plot[plot].clear()

        for index,_dataset in enumerate(self.nep_result_data.datasets):

            plot=self.axes_list[index]


            plot.title= _dataset.title

            marker_size = Config.getint("widget", "vispy_marker_size", 6) or 6
            plot.scatter(_dataset.x,
                         _dataset.y,
                         data=_dataset.structure_index,
                         brush=Brushes.get(_dataset.title.upper()) ,
                         pen=Pens.get(_dataset.title.upper()),
                         symbol='o',
                         size=marker_size,

                                      )

            # continue
            if _dataset.group_array.num !=0:
                if self.structure_index not in _dataset.group_array.now_data:
                    self.structure_index=_dataset.group_array.now_data[0]
                    self.structureIndexChanged.emit(self.structure_index)

            else:
                plot.set_current_point([], [])

            if _dataset.title not in ["descriptor"]:
            #
                pos=self.convert_pos(plot,(0.1 ,0.8))
                text=f"rmse: {_dataset.get_formart_rmse()}"
                plot.text.text=text
                plot.text.pos=pos

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
        idx = np.atleast_1d(np.asarray(structure_index)).astype(np.int64)
        if idx.size == 0:
            return

        for plot in self.axes_list:
            if not plot._scatter:
                continue
            # init overlay sets for this plot
            if plot not in self._selected_by_plot:
                self._selected_by_plot[plot] = set()
            if plot not in self._show_by_plot:
                self._show_by_plot[plot] = set()

            if color is Brushes.Default:
                # remove from both overlays
                self._selected_by_plot[plot].difference_update(idx.tolist())
                self._show_by_plot[plot].difference_update(idx.tolist())
            elif color is Brushes.Selected:
                # add to selected, remove from show to avoid duplicates
                self._selected_by_plot[plot].update(idx.tolist())
                self._show_by_plot[plot].difference_update(idx.tolist())
            elif color is Brushes.Show:
                self._show_by_plot[plot].update(idx.tolist())
            else:
                # Fallback: treat as selected
                self._selected_by_plot[plot].update(idx.tolist())

            # Recompute overlay positions lazily from dataset (handles decimation of base)
            dataset = self.get_axes_dataset(plot)
            if dataset is None:
                continue

            def _indices_to_positions(indices:set[int]):
                """Return scatter positions for the given structure indices.

                Parameters
                ----------
                indices : set[int]
                    Indices to extract.

                Returns
                -------
                numpy.ndarray
                    Positions as an array with shape ``(N, 2)``.
                """
                if not indices:
                    return np.empty((0, 2), dtype=np.float32)
                indices_arr = np.fromiter(indices, dtype=np.int64)
                # Only consider currently visible (active) structures
                # The scatter uses flattened x/y; structure_index aligns with those
                try:
                    sidx = dataset.structure_index
                    mask = np.isin(sidx, indices_arr)
                    if not np.any(mask):
                        return np.empty((0, 2), dtype=np.float32)
                    x = dataset.x[mask]
                    y = dataset.y[mask]
                    return np.vstack([x, y], dtype=np.float32).T
                except Exception:
                    return np.empty((0, 2), dtype=np.float32)

            sel_pos = _indices_to_positions(self._selected_by_plot[plot])
            show_pos = _indices_to_positions(self._show_by_plot[plot])

            # Update overlays (filled squares, no edges for perf)
            overlay_size = Config.getint("widget", "vispy_marker_size", 6) or 6
            if sel_pos.size:
                plot.set_overlay_positions('selected', sel_pos, color=Brushes.Selected, size=overlay_size, symbol='o')
            else:
                plot.set_overlay_positions('selected', np.empty((0, 2), dtype=np.float32), color=Brushes.Selected, size=overlay_size, symbol='o')

            if show_pos.size:
                plot.set_overlay_positions('show', show_pos, color=Brushes.Show, size=overlay_size, symbol='o')
            else:
                plot.set_overlay_positions('show', np.empty((0, 2), dtype=np.float32), color=Brushes.Show, size=overlay_size, symbol='o')


    def select_point_from_polygon(self,polygon_xy,reverse ):
        """Select points enclosed by the polygon drawn by the user.
        
        Parameters
        ----------
        polygon_xy : ndarray
            Polygon vertices expressed in view coordinates.
        reverse : bool
            When ``True`` remove the enclosed points from the selection.
        """
        index=self.is_point_in_polygon(np.column_stack([self.current_axes._scatter._data["a_position"][:,0],self.current_axes._scatter._data["a_position"][:,1]]),polygon_xy)
        index = np.where(index)[0]
        select_index=self.current_axes.data[index].tolist()
        self.select_index(select_index,reverse)
