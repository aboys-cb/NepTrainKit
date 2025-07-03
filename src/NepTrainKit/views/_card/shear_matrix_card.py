#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard

@CardManager.register_card
class ShearMatrixCard(MakeDataCard):
    group = "Lattice"
    card_name = "Shear Matrix Strain"
    menu_icon = r":/images/src/images/scaling.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Shear Matrix Strain")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("shear_strain_card_widget")
        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label = BodyLabel("Optional", self.setting_widget)
        self.organic_checkbox = CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.symmetric_checkbox = CheckBox("Symmetric shear", self.setting_widget)
        self.symmetric_checkbox.setChecked(True)
        self.symmetric_checkbox.setToolTip("Apply shear symmetrically")
        self.symmetric_checkbox.installEventFilter(ToolTipFilter(self.symmetric_checkbox, 300, ToolTipPosition.TOP))
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))
        self.optional_frame_layout.addWidget(self.organic_checkbox, 0, 0, 1, 1)
        self.optional_frame_layout.addWidget(self.symmetric_checkbox, 1, 0, 1, 1)

        self.xy_label = BodyLabel("XY:", self.setting_widget)
        self.xy_frame = SpinBoxUnitInputFrame(self)
        self.xy_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.xy_frame.setRange(-100, 100)
        self.xy_frame.set_input_value([-5, 5, 1])
        self.xy_label.setToolTip("XY shear strain range")
        self.xy_label.installEventFilter(ToolTipFilter(self.xy_label, 300, ToolTipPosition.TOP))

        self.yz_label = BodyLabel("YZ:", self.setting_widget)
        self.yz_frame = SpinBoxUnitInputFrame(self)
        self.yz_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.yz_frame.setRange(-100, 100)
        self.yz_frame.set_input_value([-5, 5, 1])
        self.yz_label.setToolTip("YZ shear strain range")
        self.yz_label.installEventFilter(ToolTipFilter(self.yz_label, 300, ToolTipPosition.TOP))

        self.xz_label = BodyLabel("XZ:", self.setting_widget)
        self.xz_frame = SpinBoxUnitInputFrame(self)
        self.xz_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.xz_frame.setRange(-100, 100)
        self.xz_frame.set_input_value([-5, 5, 1])
        self.xz_label.setToolTip("XZ shear strain range")
        self.xz_label.installEventFilter(ToolTipFilter(self.xz_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.optional_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.xy_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.xy_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.yz_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.yz_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.xz_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.xz_frame, 3, 1, 1, 2)

    def process_structure(self, structure):
        structure_list = []
        xy = self.xy_frame.get_input_value()
        yz = self.yz_frame.get_input_value()
        xz = self.xz_frame.get_input_value()
        identify_organic = self.organic_checkbox.isChecked()

        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        xy_range = np.arange(xy[0], xy[1] + 0.001, xy[2])
        yz_range = np.arange(yz[0], yz[1] + 0.001, yz[2])
        xz_range = np.arange(xz[0], xz[1] + 0.001, xz[2])
        cell = structure.get_cell()
        symmetric = self.symmetric_checkbox.isChecked()

        for sxy in xy_range:
            for syz in yz_range:
                for sxz in xz_range:
                    new_structure = structure.copy()
                    shear_matrix = np.eye(3)
                    shear_matrix[0, 1] += sxy / 100
                    shear_matrix[1, 2] += syz / 100
                    shear_matrix[0, 2] += sxz / 100
                    if symmetric:
                        shear_matrix[1, 0] += sxy / 100
                        shear_matrix[2, 1] += syz / 100
                        shear_matrix[2, 0] += sxz / 100

                    new_cell = np.matmul(cell, shear_matrix)
                    new_structure.set_cell(new_cell, scale_atoms=True)
                    if identify_organic:
                        process_organic_clusters(structure, new_structure, clusters, is_organic_list)
                    info_list = []
                    if abs(sxy) > 1e-8:
                        info_list.append(f"xy:{sxy}%")
                    if abs(syz) > 1e-8:
                        info_list.append(f"yz:{syz}%")
                    if abs(sxz) > 1e-8:
                        info_list.append(f"xz:{sxz}%")
                    info_str = "|".join(info_list)
                    new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Shear({info_str},symmetric={symmetric})"
                    structure_list.append(new_structure)
        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict["organic"] = self.organic_checkbox.isChecked()
        data_dict["symmetric"] = self.symmetric_checkbox.isChecked()
        data_dict["xy_range"] = self.xy_frame.get_input_value()
        data_dict["yz_range"] = self.yz_frame.get_input_value()
        data_dict["xz_range"] = self.xz_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.organic_checkbox.setChecked(data_dict.get("organic", False))
        self.symmetric_checkbox.setChecked(data_dict.get("symmetric", True))
        self.xy_frame.set_input_value(data_dict.get("xy_range", [-5, 5, 1]))
        self.yz_frame.set_input_value(data_dict.get("yz_range", [-5, 5, 1]))
        self.xz_frame.set_input_value(data_dict.get("xz_range", [-5, 5, 1]))
