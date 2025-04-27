#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/2 20:02
# @Author  : 兵
# @email    : 1747193328@qq.com

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen




def mkPen(*args, **kargs):
    """
    Convenience function for constructing QPen.

    Examples::

        mkPen(color)
        mkPen(color, width=2)
        mkPen(cosmetic=False, width=4.5, color='r')
        mkPen({'color': "#FF0", width: 2})
        mkPen(None)   # (no pen)

    In these examples, *color* may be replaced with any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`    """
    color = kargs.get('color', None)
    width = kargs.get('width', 1)
    style = kargs.get('style', None)
    dash = kargs.get('dash', None)
    cosmetic = kargs.get('cosmetic', True)
    hsv = kargs.get('hsv', None)

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

class ForcesMode(Enum):
    Raw="Raw"
    Norm="Norm"

class CanvasMode(Enum):
    vispy="vispy"
    pyqtgraph="pyqtgraph"



class Base:

    @classmethod
    def get(cls,name):
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            return getattr(cls,"Default")





class Pens(Base):
    Default=mkPen(color="blue", width=0.5)
    Energy = Default
    Force = Default
    Virial = Default
    Stress = Default
    Descriptor = Default
    Current=mkPen(color="red", width=1)
    Line = mkPen(color="red", width=2)
    def __getattr__(self, item):
        return getattr(self.Default, item)

class Brushes(Base):
    # 基本颜色刷子



    BlueBrush = QBrush(QColor(0, 0, 255))   # 蓝色
    YellowBrush = QBrush(QColor(255, 255, 0))  # 黄色
    Default = QBrush(QColor(255, 255, 255,0))  # 黄色
    Energy = Default
    Force =Default
    Virial =Default
    Stress = Default
    Descriptor = Default
    Show=QBrush(QColor(0, 255, 0))  # 绿色
    Selected=QBrush(QColor(255, 0, 0))
    Current=QBrush(QColor(255, 0,0 ))
    def __getattr__(self, item):
        return getattr(self.Default, item)