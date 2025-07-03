#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit import utils
from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard
@CardManager.register_card

class CellStrainCard(MakeDataCard):
    group = "Lattice"

    card_name= "Lattice Strain"
    menu_icon=r":/images/src/images/scaling.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Cell Strain")

        self.init_ui()

    def init_ui(self):
        self.setObjectName("cell_strain_card_widget")


        self.engine_label=BodyLabel("Axes:",self.setting_widget)
        self.engine_type_combo=EditableComboBox(self.setting_widget)
        axes_type=["uniaxial","biaxial","triaxial","isotropic"]
        self.engine_type_combo.addItems(axes_type)
        self.engine_label.setToolTip('Pull down to select or enter a specific axis, such as X or XY')
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame=QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0,0,0,0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.organic_checkbox.setToolTip("Treat organic molecules as rigid units")
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))


        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.strain_x_label=BodyLabel("X:",self.setting_widget)
        self.strain_x_frame = SpinBoxUnitInputFrame(self)
        self.strain_x_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_x_frame.setRange(-100,100)

        self.strain_x_frame.set_input_value([-5,5,1])
        self.strain_x_label.setToolTip("X-axis strain range")
        self.strain_x_label.installEventFilter(ToolTipFilter(self.strain_x_label, 300, ToolTipPosition.TOP))

        self.strain_y_label=BodyLabel("Y:",self.setting_widget)
        self.strain_y_frame = SpinBoxUnitInputFrame(self)
        self.strain_y_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_y_frame.setRange(-100,100)
        self.strain_y_frame.set_input_value([-5,5,1])
        self.strain_y_label.setToolTip("Y-axis strain range")
        self.strain_y_label.installEventFilter(ToolTipFilter(self.strain_y_label, 300, ToolTipPosition.TOP))

        self.strain_z_label=BodyLabel("Z:",self.setting_widget)
        self.strain_z_frame = SpinBoxUnitInputFrame(self)
        self.strain_z_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_z_frame.setRange(-100,100)
        self.strain_z_frame.set_input_value([-5,5,1])
        self.strain_z_label.setToolTip("Z-axis strain range")
        self.strain_z_label.installEventFilter(ToolTipFilter(self.strain_z_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 1, 1, 1,1)
        self.settingLayout.addWidget(self.strain_x_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_x_frame, 2, 1, 1,1)
        self.settingLayout.addWidget(self.strain_y_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_y_frame, 3, 1, 1,1)
        self.settingLayout.addWidget(self.strain_z_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_z_frame, 4, 1, 1,1)

    def process_structure(self, structure):
        structure_list=[]
        axes=self.engine_type_combo.currentText()
        x=self.strain_x_frame.get_input_value()
        y=self.strain_y_frame.get_input_value()
        z=self.strain_z_frame.get_input_value()
        identify_organic=self.organic_checkbox.isChecked()


        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)
        strain_range=[
            np.arange(start=x[0],stop=x[1]+0.001,step=x[2]),
            np.arange(start=y[0], stop=y[1]+0.001, step=y[2]),
            np.arange(start=z[0], stop=z[1]+0.001, step=z[2]),
        ]
        cell = structure.get_cell()
        # Define possible axes (0: x, 1: y, 2: z)
        all_axes = [0, 1, 2]


        if axes == 'isotropic':
            for strain in strain_range[0]:
                new_structure = structure.copy()
                new_cell = cell.copy() * (1 + strain / 100)
                new_structure.set_cell(new_cell, scale_atoms=True)
                if identify_organic:
                    process_organic_clusters(structure, new_structure, clusters, is_organic_list )

                strain_info = [f"all:{strain}%"]
                new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Strain({'|'.join(strain_info)})"
                structure_list.append(new_structure)
        else:
            if axes == 'uniaxial':
                axes_combinations = [[i] for i in all_axes]
            elif axes == 'biaxial':
                axes_combinations = list(combinations(all_axes, 2))
            elif axes == 'triaxial':
                axes_combinations = [all_axes]
            else:
                axes_combinations = [["XYZ".index(i.upper()) for i in axes if i.upper() in "XYZ"]]
            for ax_comb in axes_combinations:
                if len(ax_comb) == 0:
                    continue
                strain_combinations = (np.array(np.meshgrid(*[strain_range[_] for _ in ax_comb])).T.reshape(-1, len(ax_comb)))
                for strain_vals in strain_combinations:
                    new_structure = structure.copy()
                    new_cell = cell.copy()
                    for ax_idx, strain in zip(ax_comb, strain_vals):
                        new_cell[ax_idx] *= (1 + strain / 100)
                    new_structure.set_cell(new_cell, scale_atoms=True)
                    if identify_organic:
                        process_organic_clusters(structure, new_structure, clusters, is_organic_list)

                    strain_info = ["XYZ"[ax] + ":" + str(s) + "%" for ax, s in zip(ax_comb, strain_vals)]
                    new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Strain({'|'.join(strain_info)})"
                    structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['organic'] = self.organic_checkbox.isChecked()

        data_dict['engine_type'] = self.engine_type_combo.currentText()
        data_dict['x_range'] = self.strain_x_frame.get_input_value()
        data_dict['y_range'] = self.strain_y_frame.get_input_value()
        data_dict['z_range'] = self.strain_z_frame.get_input_value()

        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.organic_checkbox.setChecked(data_dict.get("organic", False))

        self.engine_type_combo.setText(data_dict['engine_type'])

        self.strain_x_frame.set_input_value(data_dict['x_range'])
        self.strain_y_frame.set_input_value(data_dict['y_range'])
        self.strain_z_frame.set_input_value(data_dict['z_range'])



