#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com

from itertools import combinations
from typing import List, Tuple

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from ase.build import make_supercell
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox, RadioButton

from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class SuperCellCard(MakeDataCard):
    group = "Lattice"
    card_name= "Super Cell"
    menu_icon=r":/images/src/images/supercell.svg"
    separator = False
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Supercell")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("super_cell_card_widget")
        self.behavior_type_combo=ComboBox(self.setting_widget)
        self.behavior_type_combo.addItem("Maximum")
        self.behavior_type_combo.addItem("Iteration")

        self.combo_label=BodyLabel("Behavior:",self.setting_widget)
        self.combo_label.setToolTip("Select supercell generation method")
        self.combo_label.installEventFilter(ToolTipFilter(self.combo_label, 300, ToolTipPosition.TOP))

        self.super_scale_radio_button = RadioButton("Super scale",self.setting_widget)
        self.super_scale_radio_button.setChecked(True)
        self.super_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_scale_condition_frame.set_input("",3)
        self.super_scale_condition_frame.setRange(1,100)
        self.super_scale_condition_frame.set_input_value([3,3,3])
        self.super_scale_radio_button.setToolTip("Scale factors along axes")
        self.super_scale_radio_button.installEventFilter(ToolTipFilter(self.super_scale_radio_button, 300, ToolTipPosition.TOP))

        self.super_cell_radio_button = RadioButton("Super cell",self.setting_widget)
        self.super_cell_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_cell_condition_frame.set_input("Å",3)
        self.super_cell_condition_frame.setRange(1,100)
        self.super_cell_condition_frame.set_input_value([20,20,20])

        self.super_cell_radio_button.setToolTip("Target lattice constant in Å")
        self.super_cell_radio_button.installEventFilter(ToolTipFilter(self.super_cell_radio_button, 300, ToolTipPosition.TOP))


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)
        # self.max_atoms_condition_frame.setToolTip("Maximum allowed atoms")
        self.max_atoms_condition_frame.set_input_value([100])

        self.max_atoms_radio_button = RadioButton("Max atoms",self.setting_widget)
        self.max_atoms_radio_button.setToolTip("Limit cell size by atom count")
        self.max_atoms_radio_button.installEventFilter(ToolTipFilter(self.max_atoms_radio_button, 300, ToolTipPosition.TOP))


        self.settingLayout.addWidget(self.combo_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.behavior_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.super_scale_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.super_scale_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.super_cell_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.super_cell_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_radio_button, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 3, 1, 1, 2)

    def _get_scale_factors(self) -> List[Tuple[int, int, int]]:
        """从 super_scale_condition_frame 获取扩包比例"""
        na, nb, nc = self.super_scale_condition_frame.get_input_value()
        return [(na, nb, nc)]

    def _get_max_cell_factors(self, structure) -> List[Tuple[int, int, int]]:
        """根据最大晶格常数计算扩包比例"""
        max_a, max_b, max_c = self.super_cell_condition_frame.get_input_value()
        lattice = structure.cell.array

        # 计算晶格向量长度
        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])

        # 计算最大倍数并确保至少为 1
        na = max(int(max_a / a_len) if a_len > 0 else 0, 1)
        nb = max(int(max_b / b_len) if b_len > 0 else 0, 1)
        nc = max(int(max_c / c_len) if c_len > 0 else 0, 1)

        # 调整倍数以不超过最大值
        na = na - 1 if na * a_len > max_a else na
        nb = nb - 1 if nb * b_len > max_b else nb
        nc = nc - 1 if nc * c_len > max_c else nc

        # 确保最小值为 1
        return [(max(na, 1), max(nb, 1), max(nc, 1))]


    def _get_max_atoms_factors(self, structure) -> List[Tuple[int, int, int]]:
        """根据最大原子数计算所有可能的扩包比例"""
        max_atoms = self.max_atoms_condition_frame.get_input_value()[0]
        num_atoms_orig = len(structure)
        # 估算最大可能倍数
        max_n = int(max_atoms / num_atoms_orig)
        max_n_a = max_n_b = max_n_c = max(max_n, 1)

        # 枚举所有可能的扩包比例
        expansion_factors = []
        for na in range(1, max_n_a + 1):
            for nb in range(1, max_n_b + 1):
                for nc in range(1, max_n_c + 1):
                    total_atoms = num_atoms_orig * na * nb * nc
                    if total_atoms <= max_atoms:
                        expansion_factors.append((na, nb, nc))
                    else:
                        break

        # 按总原子数排序
        expansion_factors.sort(key=lambda x: num_atoms_orig * x[0] * x[1] * x[2])
        if len(expansion_factors)==0:
            return [(1, 1, 1)]

        return expansion_factors

    def _generate_structures(self, structure, expansion_factors, super_cell_type) :
        """根据超胞类型和扩包比例生成结构列表"""
        structure_list = []

        if super_cell_type == 0:  # 最大扩包
            na, nb, nc = expansion_factors[-1]  # 取最大的扩包比例

            if na == 1 and nb == 1 and nc == 1:  # 只有一个扩包


                return [structure.copy()]  # 直接返回原始结构

            supercell = make_supercell(structure,np.diag([na, nb, nc]),order="atom-major")
            supercell.info["Config_type"] = supercell.info.get("Config_type","") + f" supercell({na, nb, nc})"

            structure_list.append(supercell)

        elif super_cell_type == 1:  # 随机组合或所有组合
            if self.max_atoms_radio_button.isChecked():
                # 对于 max_atoms，返回所有可能的扩包
                for na, nb, nc in expansion_factors:


                    if na==1 and nb==1 and nc==1:  # 只有一个扩包
                        supercell=structure.copy()


                    else:
                        supercell = make_supercell(structure, np.diag([na, nb, nc]),order="atom-major")
                        supercell.info["Config_type"] = supercell.info.get("Config_type","") + f" supercell({na, nb, nc})"

                        # supercell = structure.supercell([na, nb, nc])
                    structure_list.append(supercell)
            else:
                # 对于 scale 或 max_cell，枚举所有子组合
                na, nb, nc = expansion_factors[0]
                for i in range(1, na + 1):
                    for j in range(1, nb + 1):
                        for k in range(1, nc + 1):

                            if na == 1 and nb == 1 and nc == 1:  # 只有一个扩包
                                supercell = structure.copy()
                            else:
                                supercell = make_supercell(structure, np.diag([i, j, k]),order="atom-major")
                                supercell.info["Config_type"]=supercell.info.get("Config_type","") +f" supercell({i,j,k})"
                                # supercell = structure.supercell((i, j, k))

                            structure_list.append(supercell)

        # super_cell_type == 2 的情况未实现，保持为空
        return structure_list

    def process_structure(self,structure):
        super_cell_type = self.behavior_type_combo.currentIndex()
        # 根据选择的扩包方式获取扩包参数
        if self.super_scale_radio_button.isChecked():
            expansion_factors = self._get_scale_factors()
        elif self.super_cell_radio_button.isChecked():
            expansion_factors = self._get_max_cell_factors(structure)
        elif self.max_atoms_radio_button.isChecked():
            expansion_factors = self._get_max_atoms_factors(structure)
        else:
            expansion_factors = [(1, 1, 1)]  # 默认情况

        # 根据超胞类型生成结构
        structure_list = self._generate_structures(structure, expansion_factors, super_cell_type)
        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()


        data_dict['super_cell_type'] = self.behavior_type_combo.currentIndex()
        data_dict['super_scale_radio_button'] = self.super_scale_radio_button.isChecked()
        data_dict['super_scale_condition'] = self.super_scale_condition_frame.get_input_value()
        data_dict['super_cell_radio_button'] = self.super_cell_radio_button.isChecked()
        data_dict['super_cell_condition'] = self.super_cell_condition_frame.get_input_value()
        data_dict['max_atoms_radio_button'] = self.max_atoms_radio_button.isChecked()
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.behavior_type_combo.setCurrentIndex(data_dict['super_cell_type'])
        self.super_scale_radio_button.setChecked(data_dict['super_scale_radio_button'])
        self.super_scale_condition_frame.set_input_value(data_dict['super_scale_condition'])
        self.super_cell_radio_button.setChecked(data_dict['super_cell_radio_button'])
        self.super_cell_condition_frame.set_input_value(data_dict['super_cell_condition'])
        self.max_atoms_radio_button.setChecked(data_dict['max_atoms_radio_button'])
        self.max_atoms_condition_frame.set_input_value(data_dict['max_atoms_condition'])
