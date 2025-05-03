#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/1 15:06
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
os.environ["VISPY_IGNORE_OLD_VERSION"] = "true"



import numpy as np
import vispy
from PySide6.QtGui import QBrush, QColor, QPen
from vispy import scene
# 不要去掉
from vispy.app.backends import _pyside6
from vispy.color import ColorArray
from vispy.visuals.filters import MarkerPickingFilter
from NepTrainKit import utils
from NepTrainKit.core.canvas.base.canvas import VispyCanvasLayoutBase
from NepTrainKit.core.io import NepTrainResultData
from NepTrainKit.core.types import Brushes, Pens
_pyside6
vispy.use("PySide6", "gl2")

class ViewBoxWidget(scene.Widget):

    def __init__(self, title, *args, **kwargs):
        super(ViewBoxWidget, self).__init__(*args, **kwargs)

        self.unfreeze()
        self.grid = self.add_grid(margin=10)

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
        self.picking_filter = MarkerPickingFilter()

        self._scatter=None
        self._diagonal=None
        self.current_point=None
        self.freeze()



    def convert_color(self, obj):
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

            self._view.camera.set_range( real_range,  real_range)
        else:
            self._view.camera.set_range( x_range,  y_range)

    def set_current_point(self, x,y):

        if np.array(x).size == 0:
            if self.current_point is not None:
                self.current_point.parent=None
                self.current_point=None

            return
        if self.current_point is None:
            self.current_point=scene.visuals.Markers()
            self.current_point.order=0

            self._view.add(self.current_point)

        z=np.full(x.shape,2)
        self.current_point.set_data(np.vstack([x, y,z]).T,face_color=self.convert_color(Brushes.Current) ,edge_color=self.convert_color(Pens.Current),
                                      symbol='star',size=20 )

    def scatter(self,x,y,data,brush=None,pen=None ,**kwargs):
        if self._scatter is None:
            self._scatter = scene.visuals.Markers()
            self._scatter.order=1
            self._scatter.attach(self.picking_filter)

            self._view.add(self._scatter)



        self.data=data

        if brush is not None:

            kwargs["face_color"]=self.convert_color(brush)
        if pen is not None:

            kwargs["edge_color"]=self.convert_color(pen)
        if x.size != 0:
            self._scatter.set_data(np.vstack([x, y]).T, **kwargs)
            self.auto_range()
        else:
            self._scatter.set_data(np.empty((0, 3)), **kwargs)
            self.auto_range()
        self.update_diagonal()
        return self._scatter

    def line(self,x,y,**kwargs):
        xy=np.vstack([x,y]).T

        line=scene.Line(xy , **kwargs)
        self.view.add(line)
        return line
    def add_diagonal(self,**kwargs):


        x_domain = self.xaxis.axis.domain
        line_data = np.linspace(*x_domain,num=100)
        self._diagonal=self.line(line_data,line_data,**kwargs)

        self._diagonal.order=3
    def update_diagonal(self):
        if self._diagonal is None:
            return None

        x_domain = self.xaxis.axis.domain

        line_data = np.linspace(*x_domain,num=100)
        xy = np.vstack([line_data, line_data]).T
        self._diagonal.set_data(xy)

    @property
    def view(self):
        return self._view

    @property
    def title(self):
        return self.title_label._text_visual.text

    @title.setter
    def title(self, t):

        if t==self.title:
            return
        self.title_label._text_visual.text = t
        if t != "descriptor":
            self.add_diagonal(color="red", width=3, antialias=True, method='gl')
class CombinedMeta(type(VispyCanvasLayoutBase), type(scene.SceneCanvas) ):
    pass


class VispyCanvas(VispyCanvasLayoutBase, scene.SceneCanvas, metaclass=CombinedMeta):

    def __init__(self, *args, **kwargs):

        VispyCanvasLayoutBase.__init__(self)

        scene.SceneCanvas.__init__(self, *args, fullscreen=True, **kwargs)

        self.unfreeze()
        self.nep_result_data = None


        self.grid = self.central_widget.add_grid(margin=10, spacing=0)
        self.grid.spacing = 0


        self.events.mouse_double_click.connect(self.switch_view_box)
        self.path_line = scene.visuals.Line(color='red', method='gl' )
        # Use filters to affect the rendering of the mesh.

    def clear_axes(self):
        for widget in self.axes_list:
            widget._stretch = (None, None)
            widget.parent=None
            self.grid.remove_widget(widget)

        super().clear_axes()


    def set_nep_result_data(self,dataset):
        self.nep_result_data:NepTrainResultData=dataset


    def point_at(self,pos):
        if self.nep_result_data is None:
            return None
        current_axes=self._get_clicked_axes(pos)
        if current_axes is None:
            return None
        # adjust the event position for hidpi screens
        render_size = tuple(d * self.pixel_scale for d in self.size)
        x_pos = pos[0] * self.pixel_scale
        y_pos = render_size[1] - (pos[1] * self.pixel_scale)
        # print(canvas.pixel_scale)
        # render a small patch around the mouse cursor
        restore_state = not current_axes.picking_filter.enabled
        current_axes.picking_filter.enabled = True
        current_axes._scatter.update_gl_state(blend=False)
        picking_render = self.render(
            crop=(x_pos - 2, y_pos - 2, 5, 5),
            bgcolor=(0, 0, 0, 0),
            alpha=True,
        )
        if restore_state:
            current_axes.picking_filter.enabled = False
        current_axes._scatter.update_gl_state(blend=not current_axes.picking_filter.enabled)

        # unpack the face index from the color in the center pixel
        marker_idx = (picking_render.view(np.uint32) - 1)[2, 2, 0]
        if current_axes.data.size!=0:

            if marker_idx>=current_axes.data.size:
                return None
        else:
            return None
        return marker_idx
    def on_mouse_press(self, event):
        """鼠标按下事件，开始记录轨迹"""

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
                self.mouse_path = [[x, y]]  # 记录初始点
                self.path_line.set_data(pos=np.array(self.mouse_path))  # 初始化轨迹线
                self.current_axes.view.add(self.path_line)


    def on_mouse_move(self, event):
        """鼠标移动事件，动态记录轨迹"""

        if not self.draw_mode:
            return
        if (event.button == 1 or event.button ==2) and len(self.mouse_path) > 0:
            # 记录当前鼠标位置
            tr = self.scene.node_transform(self.current_axes.view.scene)
            x, y, _, _ = tr.map(event.pos)

            self.mouse_path.append([x,y])

            # 更新轨迹线
            self.path_line.set_data(pos=np.array(self.mouse_path))

    def on_mouse_release(self, event):
        """鼠标释放事件，结束记录"""
        if not self.draw_mode:
            return
        if event.button == 1 or event.button ==2:
            # 停止记录
            reverse=event.button == 2

            self.path_line.parent=None

            if len(self.mouse_path)>2:

                self.select_point_from_polygon(np.array(self.mouse_path),reverse)
            else:
                # 右键的话  选中单个点
                index = self.point_at(event.pos)
                if index is not None:
                    structure_index = self.current_axes.data[index]

                    self.select_index(structure_index,reverse)

            self.mouse_path = []

    def _get_clicked_axes(self,pos):
        view = self.visual_at(pos)

        if isinstance(view, scene.ViewBox)  :
            for axes in self.axes_list:
                if axes.view == view:
                    return axes

        return None
    def switch_view_box(self,event ):
        mouse_pos = event.pos
        axes=self._get_clicked_axes(mouse_pos)
        if axes is None:
            return
        if self.current_axes != axes:
            self.set_current_axes(axes)
            self.set_view_layout()

    def init_axes(self,axes_num   ):
        self.clear_axes()
        for r in range(axes_num):
            plot = ViewBoxWidget(title="")
            self.axes_list.append(plot)
        self.set_view_layout()
        self.update()

    def set_view_layout(self):
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
                self.grid.add_widget(widget, row=0, col=0, row_span=6, col_span=row_0_col_span)
            else:
                self.grid.add_widget(widget, row=6, col=i, row_span=2, col_span=1)

                i += 1

    def auto_range(self):
        self.current_axes.auto_range()


    def pan(self ,checked):
        self.current_axes.view.camera.interactive = checked



    def pen(self, checked):
        if self.current_axes is None:
            return False

        if checked:
            self.draw_mode = True
            # 初始化鼠标状态和轨迹数据

        else:
            self.draw_mode = False
            pass

    @utils.timeit

    def plot_nep_result(self):
        self.nep_result_data.select_index.clear()

        for index,_dataset in enumerate(self.nep_result_data.dataset):

            plot=self.axes_list[index]


            plot.title= _dataset.title

            plot.scatter(_dataset.x,_dataset.y,data=_dataset.structure_index,
                                      brush=Brushes.get(_dataset.title.upper()) ,pen=Pens.get(_dataset.title.upper()),
                                      symbol='o',size=7,

                                      )

            # continue
            if _dataset.group_array.num !=0:
                #更新结构
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
        x_range = plot.xaxis.axis.domain  # x轴范围 [xmin, xmax]
        y_range = plot.yaxis.axis.domain # y轴范围 [ymin, ymax]

        # 将百分比位置转换为坐标
        x_percent = pos[0] # 50% 对应 x 轴中间
        y_percent =  pos[1]  # 20% 对应 y 轴上的某个位置

        x_pos = x_range[0] + x_percent * (x_range[1] - x_range[0])  # 根据百分比计算实际位置
        y_pos = y_range[0] + y_percent * (y_range[1] - y_range[0])  # 根据百分比计算实际位置
        return x_pos,y_pos
    def plot_current_point(self,structure_index):
        self.structure_index=structure_index
        for plot in  self.axes_list :
            dataset=self.get_axes_dataset(plot)
            array_index=dataset.convert_index(structure_index)
            if dataset.now_data.size!=0:
                data=dataset.now_data[array_index,: ]
                plot.set_current_point(data[:,dataset.cols:].flatten(),
                                       data[:, :dataset.cols].flatten(),
                                       )
    @utils.timeit
    def update_scatter_color(self,structure_index,color=Brushes.Selected):


        for i,plot in enumerate(self.axes_list):

            if not plot._scatter:
                continue
            structure_index_set=np.array( list(set(structure_index)))
            mask = np.isin(plot.data, structure_index_set)

            if mask.size == 0:
                continue
            # 使用 where 函数找到满足条件的索引，并转换为列表
            index_list = np.where(mask)[0].tolist()

            plot._scatter._data["a_bg_color"][index_list]=   ColorArray(plot.convert_color(color)).rgba
            # plot._scatter._data['sourceRect'][index_list] = (0, 0, 0, 0)
            plot._scatter.set_data(  pos=plot._scatter._data["a_position"]
                                   , size=plot._scatter._data["a_size"] ,
                                   edge_width=None, edge_width_rel=None,
                 edge_color=plot._scatter._data["a_fg_color"], face_color=plot._scatter._data["a_bg_color"],
                 symbol='o')


    def select_point_from_polygon(self,polygon_xy,reverse ):
        index=self.is_point_in_polygon(np.column_stack([self.current_axes._scatter._data["a_position"][:,0],self.current_axes._scatter._data["a_position"][:,1]]),polygon_xy)
        index = np.where(index)[0]
        select_index=self.current_axes.data[index].tolist()
        self.select_index(select_index,reverse)