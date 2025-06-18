#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/7 14:06
# @Author  : 兵
# @email    : 1747193328@qq.com
from qfluentwidgets import  BodyLabel

from PySide6.QtGui import QPainter, QLinearGradient, QColor
from PySide6.QtCore import Qt, QRectF

class ProcessLabel(BodyLabel):
    def __init__(self, parent=None):
        super(ProcessLabel, self).__init__(parent)

        self._progress = 0  # 0~100
        self.set_colors( ["white" ])
    def set_progress(self, value):
        """设置进度(0-100)"""
        self._progress = max(0, min(100, value))
        self.update()  # 触发重绘
    def set_colors(self, colors):
        """设置渐变颜色列表(QColor对象或十六进制字符串)"""
        if len(colors)==1:
            colors=[colors[0] ]*2

        self._colors = [QColor(c) if isinstance(c, str) else c for c in colors]

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 计算进度对应的宽度
        progress_width = self.width() * (self._progress / 100)

        # 创建水平渐变
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for i, color in enumerate(self._colors):
            gradient.setColorAt(i / (len(self._colors) - 1), color)

        # 绘制背景（全宽度，半透明）
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(0, 0, self.width(), self.height()))

        # 绘制进度遮罩（右侧未完成部分用灰色覆盖）
        if self._progress < 100:
            mask_color = QColor("white")
            painter.setBrush(mask_color)
            painter.drawRect(QRectF(progress_width, 0,
                                    self.width() - progress_width, self.height()))

        # 可选：绘制进度文本
        painter.setPen(Qt.black)
        painter.drawText(self.rect(), Qt.AlignCenter,self.text())