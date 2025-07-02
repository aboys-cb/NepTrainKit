#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Card to generate stacking fault structures."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition
import numpy as np

from NepTrainKit.core import CardManager
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard

@CardManager.register_card
class StackingFaultCard(MakeDataCard):
    card_name = "Stacking Fault"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Stacking Fault")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("stacking_fault_card_widget")

        self.h_label = BodyLabel("h", self.setting_widget)
        self.h_label.setToolTip("Miller index h")
        self.h_label.installEventFilter(ToolTipFilter(self.h_label, 0, ToolTipPosition.TOP))
        self.h_frame = SpinBoxUnitInputFrame(self)
        self.h_frame.set_input("", 1, "int")
        self.h_frame.setRange(-5, 5)
        self.h_frame.set_input_value([1])

        self.k_label = BodyLabel("k", self.setting_widget)
        self.k_label.setToolTip("Miller index k")
        self.k_label.installEventFilter(ToolTipFilter(self.k_label, 0, ToolTipPosition.TOP))
        self.k_frame = SpinBoxUnitInputFrame(self)
        self.k_frame.set_input("", 1, "int")
        self.k_frame.setRange(-5, 5)
        self.k_frame.set_input_value([1])

        self.l_label = BodyLabel("l", self.setting_widget)
        self.l_label.setToolTip("Miller index l")
        self.l_label.installEventFilter(ToolTipFilter(self.l_label, 0, ToolTipPosition.TOP))
        self.l_frame = SpinBoxUnitInputFrame(self)
        self.l_frame.set_input("", 1, "int")
        self.l_frame.setRange(-5, 5)
        self.l_frame.set_input_value([1])

        self.step_label = BodyLabel("Shift(Å)", self.setting_widget)
        self.step_label.setToolTip("Displacement range")
        self.step_label.installEventFilter(ToolTipFilter(self.step_label, 0, ToolTipPosition.TOP))
        self.step_frame = SpinBoxUnitInputFrame(self)
        self.step_frame.set_input(["-", "step", "Å"], 3, "float")
        self.step_frame.setRange(-20, 20)
        self.step_frame.set_input_value([0.0, 1.0, 0.1])

        self.settingLayout.addWidget(self.h_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.h_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.k_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.k_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.l_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.l_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.step_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.step_frame, 3, 1, 1, 2)

    def process_structure(self, structure):
        h = int(self.h_frame.get_input_value()[0])
        k = int(self.k_frame.get_input_value()[0])
        l = int(self.l_frame.get_input_value()[0])
        shift_min, shift_max, shift_step = self.step_frame.get_input_value()
        shifts = np.arange(shift_min, shift_max + shift_step * 0.5, shift_step)

        cell = structure.cell
        rec = 2 * np.pi * np.linalg.inv(cell).T
        n = h * rec[0] + k * rec[1] + l * rec[2]
        if np.linalg.norm(n) == 0:
            return [structure]
        n = n / np.linalg.norm(n)

        slip_dir = np.cross(n, [0, 0, 1])
        if np.linalg.norm(slip_dir) < 1e-8:
            slip_dir = np.cross(n, [0, 1, 0])
        slip_dir = slip_dir / np.linalg.norm(slip_dir)

        proj = structure.positions @ n
        threshold = np.median(proj)
        mask = proj > threshold

        struct_list = []
        for s in shifts:
            new_struct = structure.copy()
            new_struct.structure_info['pos'][mask] += slip_dir * s
            info = f"SF(hkl={h}{k}{l},shift={s:.3f}Å)"
            new_struct.additional_fields['Config_type'] = new_struct.additional_fields.get('Config_type', '') + ' ' + info
            struct_list.append(new_struct)
        return struct_list

    def to_dict(self):
        data = super().to_dict()
        data['h'] = self.h_frame.get_input_value()
        data['k'] = self.k_frame.get_input_value()
        data['l'] = self.l_frame.get_input_value()
        data['shift_range'] = self.step_frame.get_input_value()
        return data

    def from_dict(self, data):
        super().from_dict(data)
        self.h_frame.set_input_value(data.get('h', [1]))
        self.k_frame.set_input_value(data.get('k', [1]))
        self.l_frame.set_input_value(data.get('l', [1]))
        self.step_frame.set_input_value(data.get('shift_range', [0.0, 1.0, 0.1]))

