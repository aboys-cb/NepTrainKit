#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/5/20 14:20
# @Author  : 兵
# @email    : 1747193328@qq.com
import pyqtgraph as pg
pg.setConfigOption('background', 'w')  # 设置背景为白色
pg.setConfigOption('foreground', 'k')  # 设置前景元素为黑色（如坐标轴）
pg.setConfigOptions(antialias=False,useOpenGL=False)
from .structure import StructurePlotWidget
from .canvas import  PyqtgraphCanvas