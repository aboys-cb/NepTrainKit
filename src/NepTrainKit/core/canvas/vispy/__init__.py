#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/5/20 14:18
# @Author  : 兵
# @email    : 1747193328@qq.com
from vispy import use
# 不要去掉
from vispy.app.backends import _pyside6
use("PySide6", "gl2")

from .structure import StructurePlotWidget
from .canvas import  VispyCanvas
