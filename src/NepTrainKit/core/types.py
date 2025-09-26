#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Common types and UI styling helpers used across the core package.

This module defines enums describing backends and modes, plus lightweight
Qt-graphics helpers for pens/brushes initialised from the config.

Examples
--------
>>> from NepTrainKit.core.types import NepBackend
>>> NepBackend.AUTO.value
'auto'
"""
import sys
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QIcon
from NepTrainKit.config import Config

if sys.version_info >= (3, 11):
    from enum import StrEnum          # 3.11+
else:
    from enum import Enum
    class StrEnum(str, Enum):         # Fallback for Python 3.10-
        pass

def mkPen(*args, **kwargs):
    """Construct a ``QPen`` from flexible arguments.

    Parameters
    ----------
    color : Any, optional
        Any value accepted by :class:`QColor`, e.g. name, hex string, RGB.
    width : float, default=1
        Line width in device-independent pixels.
    style : Qt.PenStyle, optional
        Dash/line style.
    dash : Sequence[float], optional
        Custom dash pattern.
    cosmetic : bool, default=True
        If ``True``, the pen width is independent of view transforms.

    Returns
    -------
    QPen
        A configured pen instance. For widths > 4.0 the cap style is set to
        ``RoundCap`` to avoid visual artifacts for many short segments.

    Examples
    --------
    >>> isinstance(mkPen('#f00', width=2), QPen)  # doctest: +SKIP
    True
    """
    color = kwargs.get('color', None)
    width = kwargs.get('width', 1)
    style = kwargs.get('style', None)
    dash = kwargs.get('dash', None)
    cosmetic = kwargs.get('cosmetic', True)
    hsv = kwargs.get('hsv', None)

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            return mkPen(**arg)
        if isinstance(arg, QPen):
            return QPen(arg)  # return a copy of this pen
        elif arg is None:
            style = Qt.PenStyle.NoPen
        else:
            color = arg
    if len(args) > 1:
        color = args

    color = QColor(color)

    pen = QPen(QBrush(color), width)
    pen.setCosmetic(cosmetic)
    if style is not None:
        pen.setStyle(style)
    if dash is not None:
        pen.setDashPattern(dash)

    if width > 4.0:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    return pen

class ForcesMode(StrEnum):
    """How to visualise forces in plots."""
    Raw = "Raw"
    Norm = "Norm"

class CanvasMode(StrEnum):
    """Preferred canvas backend for visualisation."""
    VISPY = "vispy"
    PYQTGRAPH = "pyqtgraph"

class SearchType(StrEnum):
    """Structure search attribute family."""
    TAG = "Config_type"
    FORMULA = "formula"

class NepBackend(StrEnum):
    """NEP calculator backend preference."""
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"

class Base:
    """Mixin providing a ``get`` helper that falls back to ``Default``."""
    @classmethod
    def get(cls, name):
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            return getattr(cls, "Default")

def _get_color(section: str, option: str, default_hex: str) -> QColor:
    """Read a color from config with a safe fallback to ``default_hex``."""
    val = Config.get(section, option, default_hex)
    try:
        c = QColor(val)
        if c.isValid():
            return c
        return QColor(default_hex)
    except Exception:
        return QColor(default_hex)

class Pens(Base):
    """Convenience accessors for pens configured via the config file."""
    @classmethod
    def update_from_config(cls):
        edge = _get_color("plot", "marker_edge_color", "#07519C")
        current = _get_color("plot", "current_color", "#FF0000")
        line = _get_color("plot", "line_color", "#FF0000")

        cls.Default = mkPen(color=edge, width=0.8)
        cls.Energy = cls.Default
        cls.Force = cls.Default
        cls.Virial = cls.Default
        cls.Stress = cls.Default
        cls.Descriptor = cls.Default
        cls.Current = mkPen(color=current, width=1)
        cls.Line = mkPen(color=line, width=2)

    def __getattr__(self, item):
        return getattr(self.Default, item)

class Brushes(Base):
    """Convenience accessors for brushes configured via the config file."""
    @classmethod
    def update_from_config(cls):
        face = _get_color("plot", "marker_face_color", "#FFFFFF")
        alpha = Config.getint("plot", "marker_face_alpha", 0) or 0
        face.setAlpha(int(max(0, min(255, alpha))))

        show = _get_color("plot", "show_color", "#00FF00")
        selected = _get_color("plot", "selected_color", "#FF0000")
        current = _get_color("plot", "current_color", "#FF0000")

        cls.BlueBrush = QBrush(QColor(0, 0, 255))
        cls.YellowBrush = QBrush(QColor(255, 255, 0))
        cls.Default = QBrush(face)
        cls.Energy = cls.Default
        cls.Force = cls.Default
        cls.Virial = cls.Default
        cls.Stress = cls.Default
        cls.Descriptor = cls.Default
        cls.Show = QBrush(show)
        cls.Selected = QBrush(selected)
        cls.Current = QBrush(current)

    def __getattr__(self, item):
        return getattr(self.Default, item)

class ModelTypeIcon(Base):
    """Static resource paths used for model type icons."""
    NEP=':/images/src/images/gpumd_new.png'

# Initialize pens/brushes on import
Pens.update_from_config()
Brushes.update_from_config()