#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox, LineEdit

from NepTrainKit import module_path, utils
from NepTrainKit.core import CardManager, process_organic_clusters, get_clusters, MessageManager
from NepTrainKit.core.calculator import NEPProcess
from NepTrainKit.core.io.select import farthest_point_sampling
from NepTrainKit.custom_widget import SpinBoxUnitInputFrame
from NepTrainKit.custom_widget.card_widget import MakeDataCard, FilterDataCard


@CardManager.register_card

class FPSFilterDataCard(FilterDataCard):
    separator=True
    card_name= "FPS Filter"
    menu_icon=r":/images/src/images/fps.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Filter by FPS")
        self.init_ui()
    def init_ui(self):
        self.setObjectName("fps_filter_card_widget")
        self.nep_path_label = BodyLabel("NEP file path: ", self.setting_widget)

        self.nep_path_lineedit = LineEdit(self.setting_widget)
        self.nep_path_lineedit.setPlaceholderText("nep.txt path")
        self.nep_path_label.setToolTip("Path to NEP model")
        self.nep_path_label.installEventFilter(ToolTipFilter(self.nep_path_label, 300, ToolTipPosition.TOP))

        self.nep89_path = os.path.join(module_path, "Config","nep89.txt")
        self.nep_path_lineedit.setText(self.nep89_path )


        self.num_label = BodyLabel("Max selected", self.setting_widget)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([100])
        self.num_label.setToolTip("Number of structures to keep")
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.min_distance_condition_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_condition_frame.set_input("", 1,"float")
        self.min_distance_condition_frame.setRange(0, 100)
        self.min_distance_condition_frame.object_list[0].setDecimals(4)
        self.min_distance_condition_frame.set_input_value([0.01])

        self.min_distance_label = BodyLabel("Min distance", self.setting_widget)
        self.min_distance_label.setToolTip("Minimum distance between samples")

        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.num_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.min_distance_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_condition_frame, 1, 1, 1, 2)


        self.settingLayout.addWidget(self.nep_path_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.nep_path_lineedit, 2, 1, 1, 2)

    def process_structure(self,*args, **kwargs ):
        nep_path=self.nep_path_lineedit.text()
        n_samples=self.num_condition_frame.get_input_value()[0]
        distance=self.min_distance_condition_frame.get_input_value()[0]
        self.nep_thread = NEPProcess()
        self.nep_thread.run_nep3_calculator_process(nep_path, self.dataset, "descriptor",wait=True)
        desc_array=self.nep_thread.func_result
        remaining_indices = farthest_point_sampling(desc_array, n_samples=n_samples, min_dist=distance)

        self.result_dataset = [self.dataset[i] for i in remaining_indices]

    def stop(self):
        super().stop()
        if hasattr(self, "nep_thread"):
            self.nep_thread.stop()
            del self.nep_thread

    def run(self):
        # 创建并启动线程
        nep_path=self.nep_path_lineedit.text()

        if not os.path.exists(nep_path):
            MessageManager.send_warning_message(  "NEP file not exists!")
            self.runFinishedSignal.emit(self.index)

            return
        if self.check_state:
            self.worker_thread = utils.FilterProcessingThread(

                self.process_structure
            )
            self.status_label.set_colors(["#59745A"])

            # 连接信号
            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)

    def update_progress(self, progress):
        self.status_label.setText(f"generate descriptors ...")
        self.status_label.set_progress(progress)

    def to_dict(self):
        data_dict = super().to_dict()

        data_dict['nep_path']=self.nep_path_lineedit.text()
        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        data_dict['min_distance_condition'] = self.min_distance_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        try:
            super().from_dict(data_dict)

            if os.path.exists(data_dict['nep_path']):
                self.nep_path_lineedit.setText(data_dict['nep_path'])
            else:
                self.nep_path_lineedit.setText(self.nep89_path )
            self.num_condition_frame.set_input_value(data_dict['num_condition'])
            self.min_distance_condition_frame.set_input_value(data_dict['min_distance_condition'])
        except:
            pass

