# -*- coding: utf-8 -*-
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition
import numpy as np

from NepTrainKit.core import CardManager
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard

@CardManager.register_card
class StackingFaultCard(MakeDataCard):
    """Generate stacking fault or twin structures."""
    group = "Defect"
    card_name = "Stacking Fault"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Stacking Fault")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("stacking_fault_card_widget")



        self.hkl_label = BodyLabel("h k l", self.setting_widget)
        self.hkl_frame = SpinBoxUnitInputFrame(self)
        self.hkl_frame.set_input("", 3, "int")
        self.hkl_frame.setRange(-10, 10)
        self.hkl_frame.set_input_value([1, 1, 1])
        self.hkl_label.setToolTip("Enter Miller indices (h k l): Used to define the stacking fault plane.")
        self.hkl_label.installEventFilter(ToolTipFilter(self.hkl_label, 0, ToolTipPosition.TOP))

        self.step_label = BodyLabel("Step", self.setting_widget)
        self.step_frame = SpinBoxUnitInputFrame(self)
        self.step_frame.set_input(["-", "step", ""], 3, "float")
        self.step_frame.setRange(-10, 10)
        self.step_frame.set_input_value([0.0, 1.0, 0.5])
        self.step_label.setToolTip("Displacement start, end, and step: Controls the displacement during the fault. Units: multiples of lattice normal.")
        self.step_label.installEventFilter(ToolTipFilter(self.step_label, 0, ToolTipPosition.TOP))

        self.layer_label = BodyLabel("Layers", self.setting_widget)
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input("", 1, "int")
        self.layer_frame.setRange(1, 100)
        self.layer_frame.set_input_value([1])
        self.layer_label.setToolTip("Number of layers: Controls the number of layers involved in the stacking fault. For example, 1 means one layer, 2 means two layers, etc.")
        self.layer_label.installEventFilter(ToolTipFilter(self.layer_label, 0, ToolTipPosition.TOP))


        self.settingLayout.addWidget(self.hkl_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.hkl_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_frame,1, 1, 1, 2)
        self.settingLayout.addWidget(self.step_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.step_frame, 2, 1, 1, 2)

    def process_structure(self, structure):
        structure_list = []
        h, k, l = [int(v) for v in self.hkl_frame.get_input_value()]
        step_start, step_end, step_step = self.step_frame.get_input_value()
        num_layers = int(self.layer_frame.get_input_value()[0])
        # fault_type = self.type_combo.currentText()

        cell = structure.cell.array
        recip = np.linalg.inv(cell).T
        normal = h * recip[0] + k * recip[1] + l * recip[2]
        if np.linalg.norm(normal) < 1e-8:
            return [structure]
        normal = normal / np.linalg.norm(normal)

        positions = structure.get_positions()
        # 计算与 normal 垂直的向量
        non_parallel_vector = np.array([1, 0, 0]) if normal[0] != 1 else np.array([0, 1, 0])
        perpendicular_vector = np.cross(normal, non_parallel_vector)
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)  # 归一化

        # 使用垂直向量进行分层
        coord = positions @ perpendicular_vector  # 使用垂直向量进行投影
        unique_coords = np.unique(np.round(coord, 8))
        unique_coords.sort()
        # 按照分层数选择平面位置
        if num_layers >= len(unique_coords):
            plane_pos = unique_coords[len(unique_coords) // 2]
        else:
            plane_pos = unique_coords[num_layers - 1]
        mask = coord >= plane_pos

        step_values = np.arange(step_start, step_end + step_step / 2, step_step)
        for d in step_values:
            new_structure = structure.copy()
            pos = new_structure.positions.copy()

            pos[mask] += normal * d

            new_structure.set_positions(pos)
            new_structure.wrap()
            new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" StackingFault(hkl={h}{k}{l},step={d})"
            structure_list.append(new_structure)
        return structure_list

    def to_dict(self):
        data = super().to_dict()

        data['hkl'] = self.hkl_frame.get_input_value()
        data['step'] = self.step_frame.get_input_value()
        data['layers'] = self.layer_frame.get_input_value()
        return data

    def from_dict(self, data):
        super().from_dict(data)

        self.hkl_frame.set_input_value(data.get('hkl', [1, 1, 1]))
        self.step_frame.set_input_value(data.get('step', [0.0, 1.0, 0.5]))
        self.layer_frame.set_input_value(data.get('layers', [1]))

