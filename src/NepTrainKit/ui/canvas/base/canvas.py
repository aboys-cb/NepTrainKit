"""Shared abstractions for canvas widgets used by multiple backends."""

from abc import ABC, abstractmethod

import numpy as np
from PySide6.QtCore import QObject, Signal

from NepTrainKit.core import MessageManager
from NepTrainKit.core.types import Brushes
from NepTrainKit.ui.views import KitToolBarBase


class CanvasBase(ABC):
    """Abstract base class for canvas backends.

    Notes
    -----
    Subclasses must implement user interaction hooks such as panning and data removal.
    """

    def __init__(self):
        """Initialise shared canvas state.

        Attributes
        ----------
        current_axes : Any
            Reference to the axes currently receiving interactions.
        tool_bar : KitToolBarBase
            Toolbar instance mirroring the canvas state.
        """

        self.current_axes = None
        self.tool_bar: KitToolBarBase

    @abstractmethod
    def pan(self, *args, **kwargs):
        """Enable panning on the active axes.

        Parameters
        ----------
        *args : tuple
            Backend-specific positional arguments.
        **kwargs : dict
            Backend-specific keyword arguments.
        """

    def pen(self, *args, **kwargs):
        """Toggle drawing mode for polygon selection tools.

        Parameters
        ----------
        *args : tuple
            Backend-specific positional arguments.
        **kwargs : dict
            Backend-specific keyword arguments.
        """

    @abstractmethod
    def auto_range(self):
        """Resize the axes to fit the visible data.

        Notes
        -----
        Implementations should discard sentinel values before computing limits.
        """

    @abstractmethod
    def delete(self, *args, **kwargs):
        """Remove structures from the underlying dataset.

        Parameters
        ----------
        *args : tuple
            Backend-specific positional arguments.
        **kwargs : dict
            Backend-specific keyword arguments.
        """

    def select_point_from_polygon(self, *args, **kwargs):
        """Select structures enclosed by a user-drawn polygon.

        Parameters
        ----------
        *args : tuple
            Backend-specific positional arguments for the selection routine.
        **kwargs : dict
            Backend-specific keyword arguments for the selection routine.
        """

    @staticmethod
    def is_point_in_polygon(points, polygon):
        """Vectorised point-in-polygon test.

        Parameters
        ----------
        points : ndarray
            Array of shape (N, 2) with point coordinates.
        polygon : ndarray
            Array of shape (M, 2) with polygon vertex coordinates.

        Returns
        -------
        ndarray
            Boolean mask indicating whether each point lies inside the polygon.
        """

        n = len(polygon)
        inside = np.zeros(len(points), dtype=bool)

        px, py = points[:, 0], points[:, 1]
        p1x, p1y = polygon[0]

        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            mask = ((py > np.minimum(p1y, p2y)) &
                    (py <= np.maximum(p1y, p2y)) &
                    (px <= np.maximum(p1x, p2x)) &
                    (p1y != p2y))
            xinters = (py[mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            inside[mask] ^= (px[mask] <= xinters)
            p1x, p1y = p2x, p2y

        return inside


class CombinedMeta(type(CanvasBase), type(QObject)):
    """Metaclass that merges ``CanvasBase`` and ``QObject`` inheritance hierarchies."""

    pass


class CanvasLayoutBase(CanvasBase):
    """Mixin that manages axes layout and selection state."""

    CurrentAxesChanged = Signal()
    structureIndexChanged = Signal(int)

    def __init__(self):
        """Initialise layout-specific bookkeeping."""

        CanvasBase.__init__(self)
        self.draw_mode = False
        self.structure_index = 0
        self.axes_list = []
        self.CurrentAxesChanged.connect(self.set_view_layout)

    @abstractmethod
    def set_view_layout(self):
        """Arrange axes widgets in the backing canvas."""

    @abstractmethod
    def init_axes(self, *args, **kwargs):
        """Create axes widgets for the current layout.

        Parameters
        ----------
        *args : tuple
            Backend-specific positional arguments.
        **kwargs : dict
            Backend-specific keyword arguments.
        """

    def set_current_axes(self, axes):
        """Update the axes receiving focus and emit change events.

        Parameters
        ----------
        axes : Any
            Axes instance that should become active.

        Returns
        -------
        bool
            ``True`` when the focus changed, otherwise ``False``.
        """

        if self.current_axes != axes:
            self.current_axes = axes
            if self.tool_bar is not None:
                self.tool_bar.reset()
            self.CurrentAxesChanged.emit()
            return True
        return False

    def get_axes_dataset(self, axes):
        """Return the dataset bound to a given axes.

        Parameters
        ----------
        axes : Any
            Axes instance present in ``axes_list``.

        Returns
        -------
        NepTrainResultData or None
            Dataset corresponding to the axes, or ``None`` when unavailable.
        """

        if axes is None or self.nep_result_data is None:
            return None
        axes_index = self.axes_list.index(axes)
        return self.nep_result_data.datasets[axes_index]

    def clear_axes(self):
        """Remove all tracked axes without touching backend widgets."""

        self.axes_list.clear()

    def delete(self):
        """Delete selected structures and refresh the canvas."""

        if self.nep_result_data is not None and self.nep_result_data.select_index:
            self.nep_result_data.delete_selected()
            self.plot_nep_result()

    def revoke(self):
        """Undo the most recent deletion when possible."""

        if self.nep_result_data and self.nep_result_data.is_revoke:
            self.nep_result_data.revoke()
            self.plot_nep_result()
        else:
            MessageManager.send_info_message("No undoable deletion!")

    def select_index(self, structure_index, reverse):
        """Toggle selection state for one or more structures.

        Parameters
        ----------
        structure_index : int or Sequence[int]
            Structure indices to update.
        reverse : bool
            When ``True`` remove the indices from the selection.
        """

        if isinstance(structure_index, (int, np.number)):
            structure_index = [structure_index]
        elif isinstance(structure_index, np.ndarray):
            structure_index = structure_index.tolist()

        if not structure_index:
            return
        if reverse:
            self.nep_result_data.uncheck(structure_index)
            self.update_scatter_color(structure_index, Brushes.Default)
        else:
            self.nep_result_data.select(structure_index)
            self.update_scatter_color(structure_index, Brushes.Selected)

    def inverse_select(self):
        """Flip the selection state across all active structures."""

        if self.nep_result_data is None:
            return

        active_indices = set(self.nep_result_data.structure.now_indices.tolist())
        selected = set(self.nep_result_data.select_index)

        self.select_index(list(selected), True)
        self.select_index(list(active_indices - selected), False)


class VispyCanvasLayoutBase(CanvasLayoutBase, QObject, metaclass=CombinedMeta):
    """Base class for VisPy-backed canvases with Qt integration."""

    def __init__(self, *args, **kwargs):
        """Initialise the QObject and layout base classes."""

        QObject.__init__(self)
        CanvasLayoutBase.__init__(self)
