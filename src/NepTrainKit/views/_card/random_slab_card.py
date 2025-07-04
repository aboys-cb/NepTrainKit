#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from ase.build import surface
from loguru import logger
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class RandomSlabCard(MakeDataCard):
    group = "Defect"

    card_name = "Random Slab"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Slab Generation")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_slab_card_widget")

        # Miller index ranges for h, k, l
        self.h_label = BodyLabel("h", self.setting_widget)
        self.h_label.setToolTip("h index range")
        self.h_label.installEventFilter(ToolTipFilter(self.h_label, 0, ToolTipPosition.TOP))
        self.h_frame = SpinBoxUnitInputFrame(self)
        self.h_frame.set_input(["-", "step", ""], 3, "int")
        self.h_frame.setRange(-10, 10)
        self.h_frame.set_input_value([0, 1, 1])

        self.k_label = BodyLabel("k", self.setting_widget)
        self.k_label.setToolTip("k index range")
        self.k_label.installEventFilter(ToolTipFilter(self.k_label, 0, ToolTipPosition.TOP))
        self.k_frame = SpinBoxUnitInputFrame(self)
        self.k_frame.set_input(["-", "step", ""], 3, "int")
        self.k_frame.setRange(-10, 10)
        self.k_frame.set_input_value([0, 1, 1])

        self.l_label = BodyLabel("l", self.setting_widget)
        self.l_label.setToolTip("l index range")
        self.l_label.installEventFilter(ToolTipFilter(self.l_label, 0, ToolTipPosition.TOP))
        self.l_frame = SpinBoxUnitInputFrame(self)
        self.l_frame.set_input(["-", "step", ""], 3, "int")
        self.l_frame.setRange(-10, 10)
        self.l_frame.set_input_value([1, 3, 1])

        # Layer number range
        self.layer_label = BodyLabel("Layers", self.setting_widget)
        self.layer_label.setToolTip("Layer range")
        self.layer_label.installEventFilter(ToolTipFilter(self.layer_label, 0, ToolTipPosition.TOP))
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input(["-", "step", ""], 3, "int")
        self.layer_frame.setRange(1, 50)
        self.layer_frame.set_input_value([3, 6, 1])

        # Vacuum thickness range
        self.vacuum_label = BodyLabel("Vacuum", self.setting_widget)
        self.vacuum_label.setToolTip("Vacuum thickness range in Å")
        self.vacuum_label.installEventFilter(ToolTipFilter(self.vacuum_label, 0, ToolTipPosition.TOP))
        self.vacuum_frame = SpinBoxUnitInputFrame(self)
        self.vacuum_frame.set_input(["-", "step", "Å"], 3, "int")
        self.vacuum_frame.setRange(0, 100)
        self.vacuum_frame.set_input_value([10, 10, 1])

        self.settingLayout.addWidget(self.h_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.h_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.k_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.k_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.l_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.l_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.vacuum_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.vacuum_frame, 4, 1, 1, 2)

    def process_structure(self, structure):
        structure_list = []

        h_min, h_max, h_step = self.h_frame.get_input_value()
        k_min, k_max, k_step = self.k_frame.get_input_value()
        l_min, l_max, l_step = self.l_frame.get_input_value()

        layer_min, layer_max, layer_step = self.layer_frame.get_input_value()
        vac_min, vac_max, vac_step = self.vacuum_frame.get_input_value()

        h_range = np.arange(h_min, h_max + 1, h_step)
        k_range = np.arange(k_min, k_max + 1, k_step)
        l_range = np.arange(l_min, l_max + 1, l_step)
        layer_range = np.arange(layer_min, layer_max + 1, layer_step)
        vac_range = np.arange(vac_min, vac_max + vac_step, vac_step)

        for h in h_range:
            for k in k_range:
                for l in l_range:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    for layers in layer_range:
                        for vac in vac_range:
                            try:
                                if vac==0:
                                    vac=None
                                slab = surface(structure, (int(h), int(k), int(l)), int(layers), vacuum=vac,periodic=True)
                                slab.wrap()
                                slab.info["Config_type"] = slab.info.get("Config_type", "") + f" Slab(hkl={int(h)}{int(k)}{int(l)},layers={int(layers)},vacuum={vac})"
                                structure_list.append(slab)
                            except Exception as e:
                                logger.error(f"Failed to build slab {(h, k, l)}: {e}")
        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['h_range'] = self.h_frame.get_input_value()
        data_dict['k_range'] = self.k_frame.get_input_value()
        data_dict['l_range'] = self.l_frame.get_input_value()
        data_dict['layer_range'] = self.layer_frame.get_input_value()
        data_dict['vacuum_range'] = self.vacuum_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.h_frame.set_input_value(data_dict.get('h_range', [0, 1, 1]))
        self.k_frame.set_input_value(data_dict.get('k_range', [0, 1, 1]))
        self.l_frame.set_input_value(data_dict.get('l_range', [1, 3, 1]))
        self.layer_frame.set_input_value(data_dict.get('layer_range', [3, 6, 1]))
        self.vacuum_frame.set_input_value(data_dict.get('vacuum_range', [10, 10, 1]))




#这里类名设计失误 但为了兼容以前的配置文件  不再修改类名了
