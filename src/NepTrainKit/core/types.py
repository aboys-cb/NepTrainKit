#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/2 20:02
# @Author  : 兵
# @email    : 1747193328@qq.com
import sys
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QIcon
from NepTrainKit.config import Config

if sys.version_info >= (3, 11):
    from enum import StrEnum          # 3.11+
else:
    from enum import Enum
    class StrEnum(str, Enum):         # 3.10- 的回退
        pass
#pyqtgragh导入很慢  所以先拷贝过来 后面看要不要优化掉
def mkPen(*args, **kwargs):
    """
    Convenience function for constructing QPen.

    Examples::

        mkPen(color)
        mkPen(color, width=2)
        mkPen(cosmetic=False, width=4.5, color='r')
        mkPen({'color': "#FF0", width: 2})
        mkPen(None)   # (no pen)

    In these examples, *color* may be replaced with any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`    """
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
            return QPen(arg)  ## return a copy of this pen
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

    # for width > 1.0, we are drawing many short segments to emulate a
    # single polyline. the default SquareCap style causes artifacts.
    # these artifacts can be avoided by using RoundCap.
    # this does have a performance penalty, so enable it only
    # for thicker line widths where the artifacts are visible.
    if width > 4.0:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    return pen

class ForcesMode(StrEnum):
    Raw="Raw"
    Norm="Norm"

class CanvasMode(StrEnum):
    VISPY= "vispy"
    PYQTGRAPH = "pyqtgraph"

class SearchType(StrEnum):
    TAG="Config_type"
    FORMULA="formula"

class NepBackend(StrEnum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"

class Base:
    @classmethod
    def get(cls,name):
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            return getattr(cls,"Default")

def _get_color(section: str, option: str, default_hex: str) -> QColor:
    val = Config.get(section, option, default_hex)
    try:
        c = QColor(val)
        if c.isValid():
            return c
        return QColor(default_hex)
    except Exception:
        return QColor(default_hex)


class Pens(Base):
    # Initialize from config with sensible defaults
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
    # Initialize from config
    @classmethod
    def update_from_config(cls):
        face = _get_color("plot", "marker_face_color", "#FFFFFF")
        # optional alpha channel from separate setting
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

    NEP=':/images/src/images/gpumd_new.png'

# Initialize pens/brushes on import
Pens.update_from_config()
Brushes.update_from_config()
