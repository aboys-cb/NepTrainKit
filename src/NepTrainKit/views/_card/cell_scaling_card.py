#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard
from scipy.stats.qmc import Sobol


@CardManager.register_card
class CellScalingCard(MakeDataCard):
    group = "Lattice"
    card_name= "Lattice Perturb"
    menu_icon=r":/images/src/images/scaling.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Lattice Perturb")

        self.init_ui()

    def init_ui(self):
        self.setObjectName("cell_scaling_card_widget")


        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))


        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_condition_frame.set_input_value([0.04])

        self.scaling_radio_label=BodyLabel("Max Scaling:",self.setting_widget)
        self.scaling_radio_label.setToolTip("Maximum scaling factor")

        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.optional_frame=QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0,0,0,0)
        self.optional_frame_layout.setSpacing(2)
        self.perturb_angle_checkbox=CheckBox( self.setting_widget)
        self.perturb_angle_checkbox.setText("Perturb angle")
        self.perturb_angle_checkbox.setChecked(True)
        self.perturb_angle_checkbox.setToolTip("Also perturb lattice angles")
        self.perturb_angle_checkbox.installEventFilter(ToolTipFilter(self.perturb_angle_checkbox, 300, ToolTipPosition.TOP))


        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.organic_checkbox.setToolTip("Treat organic molecules as rigid units")
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))

        self.optional_frame_layout.addWidget(self.perturb_angle_checkbox,0,0,1,1)
        self.optional_frame_layout.addWidget(self.organic_checkbox,1,0,1,1)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_label=BodyLabel("Max num:",self.setting_widget)
        self.num_condition_frame.set_input_value([50])
        self.num_label.setToolTip("Number of structures to generate")
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 1, 1, 1,1)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)
        self.settingLayout.addWidget(self.num_label, 3, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame, 3, 1, 1,2)

    def process_structure(self, structure):
        structure_list=[]
        engine_type=self.engine_type_combo.currentIndex()
        max_scaling=self.scaling_condition_frame.get_input_value()[0]
        max_num=self.num_condition_frame.get_input_value()[0]
        identify_organic=self.organic_checkbox.isChecked()

        if self.perturb_angle_checkbox.isChecked():
            perturb_angles=True
            dim=6 #abc + angles
        else:
            dim=3 #abc
            perturb_angles=False
        if engine_type == 0:

            sobol_engine = Sobol(d=dim, scramble=True)
            sobol_seq = sobol_engine.random(max_num)  # 生成 [0, 1] 的序列
            perturbation_factors = 1 + (sobol_seq - 0.5) * 2 * max_scaling
        else:
            perturbation_factors = 1 + np.random.uniform(-max_scaling, max_scaling, (max_num, dim))

        orig_lattice = structure.cell.array
        orig_lengths = np.linalg.norm(orig_lattice, axis=1)
        unit_vectors = orig_lattice / orig_lengths[:, np.newaxis]  # 原始晶格的单位向量
        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)
        for i in range(max_num):
            # 提取微扰因子
            new_structure=structure.copy()
            length_factors = perturbation_factors[i, :3]
            new_lengths = orig_lengths * length_factors

            # 构造新晶格：仅缩放长度，保持方向
            new_lattice = unit_vectors * new_lengths[:, np.newaxis]

            # 可选：扰动角度
            if perturb_angles:
                angle_factors = perturbation_factors[i, 3:]
                angles = np.arccos([
                    np.dot(orig_lattice[1], orig_lattice[2]) / (orig_lengths[1] * orig_lengths[2]),
                    np.dot(orig_lattice[0], orig_lattice[2]) / (orig_lengths[0] * orig_lengths[2]),
                    np.dot(orig_lattice[0], orig_lattice[1]) / (orig_lengths[0] * orig_lengths[1])
                ])
                new_angles = angles * angle_factors
                # 重新构造晶格（保持角度扰动）
                new_lattice = np.zeros((3, 3), dtype=np.float32)
                new_lattice[0] = [new_lengths[0], 0, 0]
                new_lattice[1] = [new_lengths[1] * np.cos(new_angles[2]),
                                  new_lengths[1] * np.sin(new_angles[2]), 0]
                cx = new_lengths[2] * np.cos(new_angles[1])
                cy = new_lengths[2] * (np.cos(new_angles[0]) - np.cos(new_angles[1]) * np.cos(new_angles[2])) / np.sin(
                    new_angles[2])
                cz = np.sqrt(max(new_lengths[2] ** 2 - cx ** 2 - cy ** 2, 0))  # 防止负值
                new_lattice[2] = [cx, cy, cz]

            # 缩放原子位置

            new_structure.info["Config_type"] = new_structure.info.get("Config_type","") + f" Scaling(scaling={max_scaling},{'uniform' if engine_type==1 else 'Sobol'  })"

            new_structure.set_cell(new_lattice,  scale_atoms=True)
            if identify_organic:
                process_organic_clusters(structure, new_structure, clusters, is_organic_list )

            structure_list.append(new_structure)
        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()

        data_dict['engine_type'] = self.engine_type_combo.currentIndex()
        data_dict['perturb_angle'] = self.perturb_angle_checkbox.isChecked()
        data_dict['organic'] = self.organic_checkbox.isChecked()

        data_dict['scaling_condition'] = self.scaling_condition_frame.get_input_value()
        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.organic_checkbox.setChecked(data_dict.get("organic", False))
        self.engine_type_combo.setCurrentIndex(data_dict['engine_type'])
        self.perturb_angle_checkbox.setChecked(data_dict['perturb_angle'])
        self.scaling_condition_frame.set_input_value(data_dict['scaling_condition'])
        self.num_condition_frame.set_input_value(data_dict['num_condition'])
