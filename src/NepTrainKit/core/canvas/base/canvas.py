#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/31 20:24
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
实现canvas的一些基本函数功能 pyqtgraph vispy 均继承此类
"""
from abc import ABC, abstractmethod

import numpy as np
from PySide6.QtCore import Signal, QObject

from NepTrainKit.core import MessageManager
from NepTrainKit.core.types import Brushes


class CanvasBase(ABC):
    def __init__(self):
        self.current_axes = None

        self.tool_bar=None
    @abstractmethod
    def pan(self ,*args,**kwargs):
        """
        平移画布
        """
        pass


    def pen(self,*args,**kwargs):
        pass

    @abstractmethod
    def auto_range(self):
        """手动调整轴的显示范围 过滤掉无效数据 比如nep 1e6"""
        pass

    @abstractmethod
    def delete(self,*args,**kwargs):
        pass

    def select_point_from_polygon(self,*args,**kwargs):
        """
        根据鼠标绘制的多边形 选择结构
        """
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
class CombinedMeta(type(CanvasBase), type(QObject) ):
    pass
class CanvasLayoutBase(CanvasBase):
    CurrentAxesChanged=Signal()
    structureIndexChanged=Signal(int)

    def __init__(self):
        CanvasBase.__init__(self)
        self.draw_mode = False
        self.structure_index=0
        self.axes_list=[]
        self.CurrentAxesChanged.connect(self.set_view_layout)
    @abstractmethod
    def set_view_layout(self):

        """
        设置子图的排布
        """
        pass

    @abstractmethod

    def init_axes(self,*args,**kwargs):
        """
        初始化子图对象
        """
        pass
    def set_current_axes(self, axes):

        if self.current_axes != axes:

            self.current_axes=axes
            if self.tool_bar is not None:

                self.tool_bar.reset()
            self.CurrentAxesChanged.emit()
            return True
        return False
    def get_axes_dataset(self,axes):
        if axes is None or self.nep_result_data is None:
            return None
        axes_index = self.axes_list.index(axes)
        return self.nep_result_data.dataset[axes_index]
    def clear_axes(self):
        """
        清空逻辑
        """

        self.axes_list.clear()


    def delete(self):
        if self.nep_result_data is not None and self.nep_result_data.select_index:
            self.nep_result_data.delete_selected()
            self.plot_nep_result()
    def revoke(self):
        """
        如果有删除的结构  撤销上一次删除的
        :return:
        """

        if self.nep_result_data and  self.nep_result_data.is_revoke:
            self.nep_result_data.revoke()
            self.plot_nep_result()

        else:
            MessageManager.send_info_message("No undoable deletion!")
    def select_index(self,structure_index,reverse):
        if isinstance(structure_index,(int,np.int64,np.int32,np.uint32,np.uint64)):
            structure_index=[structure_index]
        elif isinstance(structure_index,np.ndarray):
            structure_index=structure_index.tolist()

        if not structure_index:
            return
        if reverse:
            self.nep_result_data.uncheck(structure_index)
            self.update_scatter_color(structure_index, Brushes.Default)

        else:

            self.nep_result_data.select(structure_index)

            self.update_scatter_color(structure_index, Brushes.Selected)

    def inverse_select(self):
        if self.nep_result_data is None:
            return

        active_indices = set(self.nep_result_data.structure.now_indices.tolist())
        selected = set(self.nep_result_data.select_index)

        self.select_index(list(selected), True)
        self.select_index(list(active_indices - selected), False)
class VispyCanvasLayoutBase(CanvasLayoutBase,QObject,metaclass=CombinedMeta):
    def __init__(self,*args,**kwargs):
        QObject.__init__(self)
        CanvasLayoutBase.__init__(self)