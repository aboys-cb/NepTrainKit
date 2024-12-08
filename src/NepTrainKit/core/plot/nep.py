#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 22:22
# @Author  : 兵
# @email    : 1747193328@qq.com

import numpy as np
from PySide6.QtCore import Signal

from pyqtgraph import mkPen, ScatterPlotItem, TextItem
from PySide6.QtWidgets import QWidget,QHBoxLayout
from vispy.scene import ViewBox

from .toolbar import NepDisplayGraphicsToolBar
from .canvas import CustomGraphicsLayoutWidget, ViewBoxWidget
from .. import MessageManager
from ..io import NepTrainResultData
from ..types import Brushes
from NepTrainKit import utils


class NepResultGraphicsLayoutWidget(QWidget):
    structureIndexChanged=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)

        self._parent=parent
        self.tool_bar:NepDisplayGraphicsToolBar=None
        self.dataset=None
        self.canvas=CustomGraphicsLayoutWidget(self,bgcolor='white')
        self.canvas.events.mouse_double_click.connect(self.switch_view_box)
        # 创建网格布局
        self.grid = self.canvas.central_widget.add_grid( margin=0,spacing=0)
        self.grid.spacing = 0


        self.view_box_widgets=[]
        self._layout=QHBoxLayout(self)
        self.setLayout(self._layout)

        self._layout.addWidget(self.canvas.native)
    def switch_view_box(self,event ):
        mouse_pos = event.pos
        view =self.canvas.visual_at(mouse_pos)

        if isinstance(view,ViewBox) and self.current_view!=view:
            self.current_view =view
            self.set_view_layout()
    def set_view_layout(self):
        i=0
        for widget in self.view_box_widgets:
            widget._stretch=(None,None)
            self.grid.remove_widget(widget)

            if widget.view ==self.current_view:
                self.grid.add_widget(widget, row=0, col=0, row_span=6, col_span=4)
            else:
                self.grid.add_widget(widget, row=6, col=i, row_span=2, col_span=1)

                i+=1
    def clear_all(self):
        for widget in self.view_box_widgets:
            self.grid.remove_widget(widget)
        self.view_box_widgets.clear()
        self.current_view=None
    def set_tool_bar(self,tool):
        self.tool_bar:NepDisplayGraphicsToolBar=tool
        self.tool_bar.panSignal.connect( self.pan)
        self.tool_bar.resetSignal.connect(lambda :self.current_view.camera.set_range())

    def pan(self,checked):

        self.current_view.camera.interactive = checked

    def set_dataset(self,dataset):
        self.clear_all()
        self.dataset:NepTrainResultData=dataset

        self.plot_all()

    def convert_pos(self,plot,pos):
        view_range = plot.viewRange()
        x_range = view_range[0]  # x轴范围 [xmin, xmax]
        y_range = view_range[1]  # y轴范围 [ymin, ymax]

        # 将百分比位置转换为坐标
        x_percent = pos[0] # 50% 对应 x 轴中间
        y_percent =  pos[1]  # 20% 对应 y 轴上的某个位置

        x_pos = x_range[0] + x_percent * (x_range[1] - x_range[0])  # 根据百分比计算实际位置
        y_pos = y_range[0] + y_percent * (y_range[1] - y_range[0])  # 根据百分比计算实际位置
        return x_pos,y_pos

    def get_current_dataset(self):
        if self.current_plot is None:
            return None
        plot_index = self.axes_list.index(self.current_plot)
        return self.dataset.dataset[plot_index]
    @utils.timeit
    def plot_all(self):
        self.dataset.select_index.clear()
        _pen = mkPen(None)
        # import time

        # start = time.time()
        for index,_dataset in enumerate(self.dataset.dataset):
            view_widget=ViewBoxWidget(_dataset.title)
            self.view_box_widgets.append(view_widget)

            view_widget.scatter(_dataset.x,_dataset.y,size=7)


            if _dataset.title not in ["descriptor"]:
                view_widget.diagonal( color="red",width=3,antialias=True)

        self.current_view = self.view_box_widgets[0].view
        self.set_view_layout()
        #5.67748498916626
    def item_clicked(self,scatter_item,items,event):

        if items.any():
            item=items[0]
            self.structureIndexChanged.emit(item.data())

    def delete(self):
        if self.dataset is not None and self.dataset.select_index:

            self.dataset.delete_selected()
            self.plot_all()


    def select_point_from_polygon(self,polygon_xy,reverse ):
        index=self.is_point_in_polygon(np.column_stack([self.current_plot.scatter.data["x"],self.current_plot.scatter.data["y"]]),polygon_xy)
        index = np.where(index)[0]
        select_index=self.current_plot.scatter.data[index]["data"].tolist()
        self.select_index(select_index,reverse)


    def select_point(self,pos,reverse):
        items=self.current_plot.scatter.pointsAt(pos)
        if len(items):

            item=items[0]

            index=item.index()
            structure_index =item.data()
            self.select_index(structure_index,reverse)

    def select_index(self,structure_index,reverse):
        if isinstance(structure_index,int):
            structure_index=[structure_index]
        elif isinstance(structure_index,np.ndarray):
            structure_index=structure_index.tolist()

        if not structure_index:
            return
        if reverse:
            self.dataset.uncheck(structure_index)
            self.update_axes_color(structure_index, Brushes.TransparentBrush)

        else:

            self.dataset.select(structure_index)

            self.update_axes_color(structure_index, Brushes.RedBrush)


    def update_axes_color(self,structure_index,color=Brushes.RedBrush):


        for i,plot in enumerate(self.axes_list):

            if not hasattr(plot,"scatter"):
                continue
            structure_index_set= set(structure_index)
            index_list = [i for i, val in enumerate(plot.scatter.data["data"]) if val in structure_index_set]

            plot.scatter.data["brush"][index_list]=   color
            plot.scatter.data['sourceRect'][index_list] = (0, 0, 0, 0)


            plot.scatter.updateSpots(  )





    def revoke(self):
        """
        如果有删除的结构  撤销上一次删除的
        :return:
        """
        if self.dataset.is_revoke:
            self.dataset.revoke()
            self.plot_all()

        else:
            MessageManager.send_info_message("No undoable deletion!")