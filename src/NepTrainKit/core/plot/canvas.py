#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:03
# @Author  : 兵
# @email    : 1747193328@qq.com
import time
from abc import abstractmethod

import numpy as np
from PySide6.QtCore import Signal

from vispy import scene

from vispy.app import use_app
from vispy.app.backends import _pyside6
use_app('pyside6')


class ViewBoxWidget(scene.Widget):
    def __init__(self, title, *args, **kwargs):
        super(ViewBoxWidget, self).__init__(*args, **kwargs)
        self.unfreeze()
        self.grid = self.add_grid(margin=0)
        self._title=title
        self.grid.spacing = 0
        title = scene.Label(title, color='blue',font_size=8)
        title.height_max = 30
        self.grid.add_widget(title, row=0, col=0, col_span=2)

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

                                 # axis_label='X Axis',
                                 # axis_font_size=12,
                                 # axis_label_margin=10,
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


        self._scatter=None
        self.freeze()

    def scatter(self,x,y,**kwargs):
        if self._scatter is None:
            self._scatter = scene.visuals.Markers()
            self._view.add(self._scatter)
        self._scatter.set_data(np.vstack([x, y]).T, **kwargs)
        self._view.camera.set_range()

        return self._scatter

    def line(self,x,y,**kwargs):
        xy=np.vstack([x,y]).T

        line=scene.Line(xy , **kwargs)
        self.view.add(line)

    def diagonal(self,**kwargs):
        x_domain = self.xaxis.axis.domain
        line_data = np.linspace(*x_domain,num=100)
        return self.line(line_data,line_data,**kwargs)

    @property
    def view(self):
        return self._view

class PlotBase:

    currentPlotChanged=Signal()

    def __init__(self):
        self.current_plot=self

    @abstractmethod
    def select(self,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def delete(self,*args,**kwargs):
        raise NotImplementedError

    def select_point_from_polygon(self,polygon_x,polygon_y):
        pass
    @staticmethod
    def is_point_in_polygon(points, polygon):
        """
        判断多个点是否在多边形内
        :param points: (N, 2) 的数组，表示 N 个点的坐标
        :param polygon: (M, 2) 的数组，表示多边形的顶点坐标
        :return: (N,) 的布尔数组，表示每个点是否在多边形内
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



class LayoutPlotBase(PlotBase):
    def __init__(self,row=None,col=None,**kwargs):
        super().__init__()
        self.current_plot=None
        self.tool_bar=None

    def set_current_plot(self,plot):

        if self.current_plot != plot:
            self.current_plot=plot
            self.currentPlotChanged.emit()
            return True
        return False


class CustomGraphicsLayoutWidget(LayoutPlotBase,scene.SceneCanvas):

    def __init__(self,*args,**kwargs):
        super().__init__(self )

        scene.SceneCanvas.__init__(self,*args,**kwargs)


