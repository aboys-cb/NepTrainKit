#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/3/15 13:44
# @Author  : 兵
# @email    : 1747193328@qq.com

import numpy as np
from PySide6.QtCore import Qt

from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QSizePolicy, QLabel

from qfluentwidgets import BodyLabel



class StructureInfoWidget(QWidget):
    def __init__(self, parent=None):
        super(StructureInfoWidget, self).__init__(parent)
        self.init_ui()
    def init_ui(self):
        self._layout = QGridLayout(self)  # 创建布局
        self._layout.setContentsMargins(0, 0, 0, 0)  # 设置边距
        self._layout.setSpacing(0)  # 设置间距
        self.setLayout(self._layout)  # 设置布局


        self.atom_label = BodyLabel(self)
        self.atom_label.setText("Atoms:")
        self.atom_num_text = BodyLabel(self)

        self.formula_label = BodyLabel(self)
        self.formula_label.setText("Formula:")
        self.formula_text = BodyLabel(self)
        self.formula_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.formula_text.setWordWrap(True)

        self.lattice_label = BodyLabel(self)
        self.lattice_label.setText("Lattice:")
        self.lattice_text = BodyLabel(self)
        self.lattice_text.setWordWrap(True)

        self.length_label = BodyLabel(self)
        self.length_label.setText("a b c:")
        self.length_text = BodyLabel(self)

        self.angle_label = BodyLabel(self)
        self.angle_label.setText("Angles:")
        self.angle_text = BodyLabel(self)

        self.config_label = BodyLabel(self)
        self.config_label.setText("Config Type:")
        self.config_text = BodyLabel(self)
        self.config_text.setMaximumWidth(400)

        # self.config_text.setMinimumWidth(10)
        self.config_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.config_text.setWordWrap(True)

        self._layout.addWidget(self.atom_label, 0,0,1,1)
        self._layout.addWidget(self.atom_num_text, 0, 1,1,3)
        self._layout.addWidget(self.formula_label, 1,0,1,1)
        self._layout.addWidget(self.formula_text, 1, 1,1,3)


        self._layout.addWidget(self.config_label, 2, 0,1,1)
        self._layout.addWidget(self.config_text, 2, 1,1,3)

        self._layout.addWidget(self.lattice_label, 3, 0,1,1)
        self._layout.addWidget(self.lattice_text, 3, 1,1,3)
        self._layout.addWidget(self.length_label, 4, 0,1,1)
        self._layout.addWidget(self.length_text, 4, 1,1,3)
        self._layout.addWidget(self.angle_label, 5, 0,1,1)
        self._layout.addWidget(self.angle_text, 5, 1,1,3)

    def show_structure_info(self, structure):

        self.atom_num_text.setText(str(len(structure )))
        self.formula_text.setText(structure.html_formula)
        self.lattice_text.setText(str(np.round(structure.lattice,3)))
        self.length_text.setText(
            " ".join(f"{x:.3f}" for x in structure.abc)
        )
        self.angle_text.setText(
            " ".join(f"{x:.2f}°" for x in structure.angles)
        )
        self.config_text.setText('\n'.join(structure.tag[i:i+50] for i in range(0, len(structure.tag), 50)))


