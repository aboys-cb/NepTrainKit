#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 22:12
# @Author  : 兵
# @email    : 1747193328@qq.com
from enum import Enum


import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon, QPen
from PySide6.QtWidgets import QToolBar
from pyqtgraph import ViewBox, PlotCurveItem

from NepTrainKit import utils
from NepTrainKit.core import Config
from NepTrainKit.core.custom_widget.dialog import GetIntMessageBox, SparseMessageBox
from NepTrainKit.core.io.select import farthest_point_sampling


class _Mode(str, Enum):
    NONE = ""
    PAN = "pan"
    ZOOM = "zoom rect"
    PEN="pen"
    def __str__(self):
        return self.value

    @property
    def _navigate_mode(self):
        return self.name if self is not _Mode.NONE else None





class NepDisplayGraphicsToolBar(QToolBar):
    panSignal=Signal(bool)
    resetSignal=Signal()
    findMaxSignal=Signal()
    sparseSignal=Signal()
    PenSignal=Signal()
    undoSignal=Signal()
    deleteSignal=Signal()

    def __init__(self,graph_widget, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.graph_widget=graph_widget
        self.graph_widget.tool_bar=self
        # self.graph_widget.currentPlotChanged.connect(self.reset_connect)
        self._actions={}
        self.init_actions()
        self.mode=_Mode.NONE

    def init_actions(self):
        self.add_action("Reset View",QIcon(":/images/src/images/init.svg"),self.reset_view)
        pan_action=self.add_action("Pan View",QIcon(":/images/src/images/pan.svg"),self.pan)
        pan_action.setCheckable(True)
        find_max_action=self.add_action( "Find Max Error Point",
                                          QIcon(":/images/src/images/find_max.svg"),
                                          self.find_max_error_point)
        sparse_action=self.add_action( "Sparse samples",
                                          QIcon(":/images/src/images/sparse.svg"),
                                          self.sparse_point)


        pen_action=self.add_action("Mouse Selection",QIcon(":/images/src/images/pen.svg"),self.pen)
        pen_action.setCheckable(True)

        revoke_action=self.add_action("Undo",QIcon(":/images/src/images/revoke.svg"),self.revoke)

        delete_action=self.add_action("Delete Selected Items",QIcon(":/images/src/images/delete.svg"),self.delete)

    def add_action(self, name,icon,callback):
        action=QAction(QIcon(icon),name,self)
        action.triggered.connect(callback)
        self._actions[name]=action
        self.addAction(action)
        action.setToolTip(name)
        return action
    def reset_connect(self):
        """
        在选中工具栏的情况下  双击进行切换 会出现问题
        这里直接判断下
        :return:
        """



        if self.view_box is not None:

            del self.view_box.mousePressEvent
            del self.view_box.mouseMoveEvent
            del self.view_box.mouseReleaseEvent
            if self.mode==_Mode.PAN:
                self.pan(False)

            elif self.mode==_Mode.PEN:
                self.pen(False)
            self.mode = _Mode.NONE
            self._update_buttons_checked()
        self.current_plot=self.graph_widget.current_plot

        self.view_box=self.current_plot.getViewBox()

        self.view_box.mousePressEvent = self._mousePressEvent
        self.view_box.mouseMoveEvent = self._mouseMoveEvent
        self.view_box.mouseReleaseEvent = self._mouseReleaseEvent

    def reset_view(self):
        """重置视图"""
        self.resetSignal.emit()

    def delete(self):
        self.graph_widget.delete()

    def revoke(self):
        self.graph_widget.revoke()

    def find_max_error_point(self):
        dataset = self.graph_widget.get_current_dataset()
        if dataset is None:
            return
        box= GetIntMessageBox(self.graph_widget._parent,"Please enter an integer N, it will find the top N structures with the largest errors")
        n = Config.getint("widget","max_error_value",10)
        box.intSpinBox.setValue(n)

        if not box.exec():
            return
        nmax= box.intSpinBox.value()
        Config.set("widget","max_error_value",nmax)
        index= (dataset.get_max_error_index(nmax))

        self.graph_widget.select_index(index,False)

    def sparse_point(self):
        if  self.graph_widget.dataset is None:
            return
        box= SparseMessageBox(self.graph_widget._parent,"Please specify the maximum number of structures and minimum distance")
        n_samples = Config.getint("widget","sparse_num_value",10)
        distance = Config.getfloat("widget","sparse_distance_value",0.01)

        box.intSpinBox.setValue(n_samples)
        box.doubleSpinBox.setValue(distance)

        if not box.exec():
            return
        n_samples= box.intSpinBox.value()
        distance= box.doubleSpinBox.value()

        Config.set("widget","sparse_num_value",n_samples)
        Config.set("widget","sparse_distance_value",distance)

        dataset = self.graph_widget.dataset.descriptor
        indices_to_remove = farthest_point_sampling(dataset.now_data,n_samples=n_samples,min_dist=distance)

        # 获取所有索引（从 0 到 len(arr)-1）
        all_indices = np.arange(dataset.now_data.shape[0])

        # 使用 setdiff1d 获取不在 indices_to_remove 中的索引
        remaining_indices = np.setdiff1d(all_indices, indices_to_remove)
        structures = dataset.group_array[remaining_indices]
        self.graph_widget.select_index(structures.tolist(),False)

    def pan(self, checked):
        """切换平移模式"""


        if checked:
            self.mode = _Mode.PAN
            self.panSignal.emit(True)
            # self.view_box.setMouseEnabled(True, True)

            # self.view_box.setMouseMode(ViewBox.PanMode)  # 启用平移模式
        else:
            self.mode = _Mode.NONE
            self.panSignal.emit(False)

            # self.view_box.setMouseEnabled(False, False)
        self._update_buttons_checked()



    def pen(self, checked):
        if self.view_box is None:
            self._update_buttons_checked()

            return False

        if checked:
            self.mode = _Mode.PEN
            # 初始化鼠标状态和轨迹数据
            self.is_drawing = False
            self.x_data = []
            self.y_data = []

        else:
            self.mode = _Mode.NONE
        self._update_buttons_checked()
    def _update_buttons_checked(self,):

        # sync button checkstates to match active mode
        if 'Pan View' in self._actions:
            self._actions['Pan View'].setChecked(self.mode.name == 'PAN')

        if 'Mouse Selection' in self._actions:
            self._actions['Mouse Selection'].setChecked(self.mode.name == 'PEN')

    def _mousePressEvent(self, event):
        """鼠标按下时开始绘制"""
        if self.mode !=_Mode.PEN:
            return ViewBox.mousePressEvent(self.view_box,event)
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            self.is_drawing = True
            self.x_data.clear()  # 清空之前的轨迹数据
            self.y_data.clear()  # 清空之前的轨迹数据
            self.curve = self.current_plot.plot([], [], pen='r')

            self.curve.setData([], [])  # 清空绘制线条，避免对角线
            # 创建一个单独的 QGraphicsItem 用于绘制鼠标轨迹
            # self.curve = PlotCurveItem(pen=QPen(Qt.red,0.05))
            # self.current_plot.addItem(self.curve)



    def on_mouse_move(self,pos):
        if self.is_drawing:

            # 将场景坐标转换为视图坐标
            mouse_point =self.view_box.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            # 记录轨迹数据
            self.x_data.append(x)
            self.y_data.append(y)

            # 更新绘图
            self.curve.setData(self.x_data, self.y_data)


    def _mouseMoveEvent(self, event):
        """当鼠标移动时，如果按住左键，则绘制鼠标的轨迹"""
        if self.is_drawing:
            pos = event.scenePos()
            # 将场景坐标转换为视图坐标
            mouse_point =self.view_box.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            # 记录轨迹数据
            self.x_data.append(x)
            self.y_data.append(y)

            # 更新绘图
            self.curve.setData(self.x_data, self.y_data)
        else:
            return ViewBox.mousePressEvent(self.view_box,event)

    def _mouseReleaseEvent(self, event):
        """鼠标松开时停止绘制，并标记轨迹内的散点"""
        if self.mode !=_Mode.PEN:
            return ViewBox.mousePressEvent(self.view_box,event)



        if event.button() == Qt.MouseButton.LeftButton  or event.button() == Qt.MouseButton.RightButton:
            self.is_drawing = False
            reverse=event.button() == Qt.MouseButton.RightButton
            self.current_plot.removeItem(self.curve)
            # 创建鼠标轨迹的多边形
            if len(self.x_data)>2:

                self.graph_widget.select_point_from_polygon(np.column_stack((self.x_data, self.y_data)),reverse)
            else:


                # 右键的话  选中单个点
                pass
                pos = event.scenePos()
                mouse_point = self.view_box.mapSceneToView(pos)

                x = mouse_point.x()
                self.graph_widget.select_point(mouse_point,reverse)
            return

