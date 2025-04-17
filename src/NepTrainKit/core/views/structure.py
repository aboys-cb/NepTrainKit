#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/3/15 13:44
# @Author  : 兵
# @email    : 1747193328@qq.com
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *  # noqa
from PySide6.QtGui import QColor,QMatrix4x4
from PySide6.QtWidgets import QApplication

from NepTrainKit.core import Config
from NepTrainKit.core.structure import table_info, Structure
from NepTrainKit import utils

class StructurePlotWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        self.ortho=False

        super().__init__(*args, **kwargs)
        self.setBackgroundColor('w')
        # self.setCameraPosition(distance=30, elevation=30, azimuth=30)
        self.atom_items = []  # 存储所有原子的信息和对应的GLMeshItem

        self.structure=None
        self.show_bond_flag = None
    def set_projection(self,ortho=True):
        self.ortho=ortho
        self.setProjection()
        self.update()
    def set_show_bonds(self,show_bonds=True):
        self.show_bond_flag = show_bonds
        if self.structure is not None:
            if show_bonds:
                self.show_bond(self.structure)
            else:
                self.show_structure(self.structure)
    def setProjection(self, region=None ):
        m = self.projectionMatrix(region)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.array(m.data(), dtype=np.float32))
    def projectionMatrix(self, region=None ):
        # 如果已经缓存，直接返回
        if self.ortho:
            # 获取视口和参数
            x0, y0, w, h = self.getViewport()
            aspect = w / h if h != 0 else 1.0
            dist = max(self.opts['distance'], 1e-6)
            fov = self.opts['fov']
            nearClip = dist * 0.001  # 用于裁剪，不影响正交范围
            farClip = dist * 1000.0
            # 正交投影：基于 distance 计算视口范围
            r = dist * np.tan(np.radians(fov / 2))  # 在 distance 处的半宽度
            t = r * h / w  # 调整高度
            # 假设 region 为整个视口
            region = region or (x0, y0, w, h)
            left = r * ((region[0] - x0) * (2.0 / w) - 1)
            right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
            bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
            top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)
            mat = QMatrix4x4()
            mat.setToIdentity()

            mat.ortho(left, right, bottom, top, nearClip, farClip)

            return mat
        else:
            return super().projectionMatrix(region)
    def addItems(self, items):
        for item in items:

            self.items.append(item)

            if self.isValid():
                item.initialize()

            item._setView(self)
        self.update()
    def show_lattice(self, structure):
        origin = np.array([0.0, 0.0, 0.0])
        a1 = structure.cell[0]
        a2 = structure.cell[1]
        a3 = structure.cell[2]
        vertices = np.array([origin, a1, a2, a3, a1 + a2, a1 + a3, a2 + a3, a1 + a2 + a3])
        edges = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]
        lines = np.array([vertices[edge] for edge in edges]).reshape(-1, 3)
        lattice_lines = gl.GLLinePlotItem(
            pos=lines,
            color=(0, 0, 0, 1),
            width=1.5,
            mode='lines',
            glOptions="translucent",
            antialias=True
        )
        # center = structure.cell.sum(axis=0) / 2
        # self.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.addItem(lattice_lines)

    def show_bond(self,structure):
        if not self.show_bond_flag:
            return
        bond_pairs = structure.get_bond_pairs()
        bond_items = []
        for pair in bond_pairs:

            elem0_info = table_info[str(structure.numbers[pair[0]])]
            elem1_info = table_info[str(structure.numbers[pair[1]])]
            pos1 = structure.positions[pair[0]]
            pos2 = structure.positions[pair[1]]
            # bond_length = np.linalg.norm(pos1 - pos2)
            # if (elem0_info["radii"] + elem1_info["radii"]) * radius_coefficient_config > bond_length * 100:
            #     color1 = (1.0, 0.0, 0.0, 0.7)
            #     color2 = (1.0, 0.0, 0.0, 0.7)
            #     bond_radius = 0.3
            # else:
            color1 = QColor(elem0_info["color"]).getRgbF()
            color2 = QColor(elem1_info["color"]).getRgbF()
            bond_radius = 0.15
            radius1 = table_info[str(structure.numbers[pair[0]])]["radii"] / 150
            radius2 = table_info[str(structure.numbers[pair[1]])]["radii"] / 150
            bond1, bond2 = self.add_bond(pos1, pos2, color1, color2, radius1, radius2, bond_radius=bond_radius)
            bond_items.append(bond1)
            bond_items.append(bond2)
        self.addItems(bond_items)
    def add_bond(self, pos1, pos2, color1, color2, radius1, radius2, bond_radius=0.12):
        """使用圆柱体绘制两个原子之间的化学键，从球体表面开始"""
        bond_vector = pos2 - pos1
        full_length = np.linalg.norm(bond_vector)
        bond_dir = bond_vector / full_length
        start_point = pos1
        end_point = pos2
        mid_point = (start_point + end_point) / 2
        bond = full_length - radius1 - radius2
        bond1_length = radius1 + bond / 2
        bond2_length = radius2 + bond / 2
        mid_point = start_point + bond_dir * bond1_length

        cylinder1 = gl.MeshData.cylinder(rows=6, cols=12, radius=[bond_radius, bond_radius], length=bond1_length)
        bond1 = gl.GLMeshItem(meshdata=cylinder1, smooth=True, shader='shaded', color=color1)
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, bond_dir)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, bond_dir)) * 180 / np.pi
            bond1.rotate(angle, axis[0], axis[1], axis[2])
        bond1.translate(start_point[0], start_point[1], start_point[2])
        # self.addItem(bond1)

        cylinder2 = gl.MeshData.cylinder(rows=6, cols=12, radius=[bond_radius, bond_radius], length=bond2_length)
        bond2 = gl.GLMeshItem(meshdata=cylinder2, smooth=True, shader='shaded', color=color2)
        if np.linalg.norm(axis) > 0:
            bond2.rotate(angle, axis[0], axis[1], axis[2])
        bond2.translate(mid_point[0], mid_point[1], mid_point[2])
        # self.addItem(bond2)
        return bond1, bond2

    def show_elem(self, structure):
        self.atom_items = []  # 清空之前的原子信息
        atom_items = []
        for idx, (n, p) in enumerate(zip(structure.numbers, structure.positions)):
            color = QColor(table_info[str(n)]["color"]).getRgbF()
            size = table_info[str(n)]["radii"] / 150
            sphere = gl.MeshData.sphere(rows=10, cols=10, radius=size)
            m = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=color)
            m.translate(p[0], p[1], p[2])

            atom_items.append(m)

            # 存储原子的信息和对应的GLMeshItem
            self.atom_items.append({"mesh": m, "position": p, "original_color": color, "size": size, "halo": None})
        self.addItems(atom_items)
        radius_coefficient_config = Config.getfloat("widget", "radius_coefficient", 0.7)
        bond_pairs = structure.get_bad_bond_pairs( radius_coefficient_config)
        for pair in bond_pairs:

            self.highlight_atom(pair[0])
            self.highlight_atom(pair[1])

    #
    def highlight_atom(self, atom_index):
        """高亮指定的原子并添加光晕"""
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            # 高亮效果：增大尺寸并设置为亮红色
            # highlight_color = (1.0, 0.2, 0.2, 1.0)  # 浅红色
            highlight_size = atom["size"]
            # highlight_color =atom["original_color"]
            # 移除原来的原子
            # self.removeItem(atom["mesh"])

            # 创建新的高亮的原子
            # sphere = gl.MeshData.sphere(rows=10, cols=10, radius=highlight_size)
            # new_mesh = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=highlight_color)
            # new_mesh.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            # self.addItem(new_mesh)

            # 添加光晕效果
            halo_size = highlight_size * 1.2 # 光晕比高亮原子大2倍
            halo_color = ( 1, 1, 0, 0.6)  # 半透明的浅红色
            halo_sphere = gl.MeshData.sphere(rows=10, cols=10, radius=halo_size)
            halo = gl.GLMeshItem(meshdata=halo_sphere, smooth=True, shader='shaded', color=halo_color, glOptions='translucent')
            halo.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            self.addItem(halo)

            # 更新atom_items中的mesh和halo
            # self.atom_items[atom_index]["mesh"] = new_mesh
            self.atom_items[atom_index]["halo"] = halo
            self.update()  # 刷新视图

    def reset_atom(self, atom_index):
        """恢复原子的原始状态并移除光晕"""
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            # 移除当前原子和光晕
            # self.removeItem(atom["mesh"])
            if atom["halo"] is not None:
                self.atom_items[atom_index]["halo"] = None

                self.removeItem(atom["halo"])
            return
            # 恢复原始原子
            sphere = gl.MeshData.sphere(rows=10, cols=10, radius=atom["size"])
            new_mesh = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=atom["original_color"])
            new_mesh.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            self.addItem(new_mesh)

            # 更新atom_items
            self.atom_items[atom_index]["mesh"] = new_mesh
            # self.atom_items[atom_index]["halo"] = None
            self.update()

    def show_structure(self, structure):
        self.atom_items.clear()
        self.clear()
        self.structure = structure
        self.show_lattice(structure)
        self.show_elem(structure)
        self.show_bond(structure )
        # 计算边界框和相机参数
        coords=structure.positions
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dimension = np.max(size)

        # 设置相机参数
        fov = 60
        distance = max_dimension / (2 * np.tan(np.radians(fov / 2))) * 2.1
        self.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.opts['distance'] = distance
        aspect_ratio = size / np.max(size)  # 计算 x、y、z 的相对比例
        # print("aspect_ratio", aspect_ratio)
        flat_threshold=0.5
        # 根据扁平方向调整视角
        if aspect_ratio[0] < flat_threshold and aspect_ratio[1] >= flat_threshold and aspect_ratio[2] >= flat_threshold:
            # x 方向扁平，从 x 轴法线方向（侧面）看
            self.opts['elevation'] = 0
            self.opts['azimuth'] = 0
        elif aspect_ratio[1] < flat_threshold and aspect_ratio[0] >= flat_threshold and aspect_ratio[
            2] >= flat_threshold:
            # y 方向扁平，从 y 轴法线方向（侧面）看
            self.opts['elevation'] = 0
            self.opts['azimuth'] = 0
        elif aspect_ratio[2] < flat_threshold and aspect_ratio[0] >= flat_threshold and aspect_ratio[
            1] >= flat_threshold:
            # z 方向扁平，从 z 轴法线方向（顶部）看
            self.opts['elevation'] = 90
            self.opts['azimuth'] = 0
        else:
            # 没有明显扁平方向，使用默认斜视角
            self.opts['elevation'] = 30
            self.opts['azimuth'] = 45

        # 应用设置
        self.setCameraPosition( )


        # 示例：高亮第0个原子
        # self.highlight_atom(0)

if __name__ == '__main__':
    app = QApplication([])
    view = StructurePlotWidget()
    view.show()
    import time
    start = time.time()
    atoms = Structure.read_xyz("good.xyz")
    view.show_structure(atoms)  # 修改为show_structure，与代码一致
    print(time.time() - start)
    QApplication.instance().exec_()