#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
import importlib
import os
from itertools import combinations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QFrame, QWidget, QVBoxLayout

from qfluentwidgets import ComboBox, BodyLabel, RadioButton, RoundMenu, PrimaryDropDownPushButton, CommandBar, Action, CheckBox, LineEdit, EditableComboBox, \
    ToolTipFilter, ToolTipPosition
from NepTrainKit import utils, module_path,get_user_config_path

from NepTrainKit.core import MessageManager
from NepTrainKit.custom_widget import (
    SpinBoxUnitInputFrame,
    MakeDataCardWidget,
    ProcessLabel
)
from NepTrainKit.custom_widget import DopingRulesWidget, VacancyRulesWidget
from NepTrainKit.core.calculator import NEPProcess
from NepTrainKit.core.io.select import farthest_point_sampling
from scipy.sparse.csgraph import connected_components
from scipy.stats.qmc import Sobol
from ase import neighborlist

from ase.geometry import find_mic
from ase.io import write as ase_write
from ase.build import make_supercell,surface
from ase import Atoms

card_info_dict = {}
def register_card_info(card_class  ):
    card_info_dict[card_class.__name__] =card_class

    return card_class


def load_cards_from_directory(directory: str):
    """Load all card modules from a directory"""
    dir_path = Path(directory)

    if not dir_path.is_dir():
        return None
    #     raise ValueError(f"Directory not found: {directory}")

    for file_path in dir_path.glob("*.py"):

        if file_path.name.startswith("_"):
            continue  # Skip private/python module files

        module_name = file_path.stem
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # The module should register its cards automatically via decorators
            logger.success(f"Successfully loaded card module: {module_name}")

        except Exception as e:
            logger.error(f"Failed to load card module {file_path}: {str(e)}")


# 判断团簇是否为有机分子
def is_organic_cluster(symbols):
    has_carbon = 'C' in symbols
    organic_elements = {'H', 'O', 'N', 'S', 'P'}
    has_organic_elements = any(symbol in organic_elements for symbol in symbols)
    return has_carbon and has_organic_elements

# 识别结构中的团簇
def get_clusters(structure):
    cutoff = neighborlist.natural_cutoffs(structure)
    nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(structure)
    matrix = nl.get_connectivity_matrix()
    n_components, component_list = connected_components(matrix)

    clusters = []
    is_organic_list = []
    for i in range(n_components):
        cluster_indices = [j for j in range(len(structure)) if component_list[j] == i]
        cluster_symbols = [structure[j].symbol for j in cluster_indices]
        clusters.append(cluster_indices)
        is_organic_list.append(is_organic_cluster(cluster_symbols))
    return clusters, is_organic_list

# 解包跨越边界的分子
def unwrap_molecule(structure, cluster_indices):
    pos = structure.positions[cluster_indices]
    cell = structure.cell
    ref_pos = pos[0]
    unwrapped_pos = [ref_pos]

    for i in range(1, len(cluster_indices)):
        delta = pos[i] - ref_pos
        mic_delta, _ = find_mic(delta, cell, pbc=True)
        unwrapped_pos.append(ref_pos + mic_delta)
    return np.array(unwrapped_pos)

# 封装循环部分：处理有机分子团簇
def process_organic_clusters(structure, new_structure, clusters, is_organic_list):
    """处理有机分子团簇并更新原子位置"""

    for cluster_indices, is_organic in zip(clusters, is_organic_list):
        if is_organic:
            # 解包分子
            unwrapped_pos = unwrap_molecule(structure, cluster_indices)

            # 计算解包后质心
            center_unwrapped = np.mean(unwrapped_pos, axis=0)

            # 将质心转换到分数坐标并映射回晶胞内
            scaled_center = np.dot(center_unwrapped, np.linalg.inv(structure.cell)) % 1.0
            center_original = np.dot(scaled_center, structure.cell)

            # 计算原子相对于质心的位移
            delta_pos = unwrapped_pos - center_unwrapped

            # 在新晶胞中计算新质心
            center_new = np.dot(scaled_center, new_structure.cell)

            # 新位置 = 新质心 + 原始位移
            pos_new = center_new + delta_pos

            # 更新原子位置
            new_structure.positions[cluster_indices] = pos_new
    new_structure.wrap()
def sample_dopants(dopant_list, ratios, N, exact=False, seed=None):
    """
    采样 dopant 的函数。

    参数：
    - dopant_list: list，可选的 dopant 值列表，比如 [0,1,2]
    - ratios: list，与 dopant_list 对应的概率或比例列表，比如 [0.6,0.3,0.1]
    - N: int，要生成的样本总数
    - exact: bool，控制采样方式：
        - False（默认）：每次独立按概率 p=ratios 抽样，结果数量只在期望值附近波动
        - True：严格按 ratios*N 计算各值的个数（向下取整后把差值补给概率最高的那一项），然后打乱顺序
    - seed: int 或 None，用于设置随机种子，保证可复现

    返回：
    - list，长度为 N 的采样结果
    """
    if seed is not None:
        np.random.seed(seed)

    dopant_list = list(dopant_list)
    ratios = np.array(ratios, dtype=float)
    ratios = ratios / ratios.sum()  # 归一化，以防输入不规范

    if not exact:
        # 独立概率抽样
        return list(np.random.choice(dopant_list, size=N, p=ratios))
    else:
        # 严格按比例生成固定个数再打乱
        counts = (ratios * N).astype(int)
        diff = N - counts.sum()
        if diff != 0:
            # 差值补给比例最大的那一项
            max_idx = np.argmax(ratios)
            counts[max_idx] += diff

        arr = np.repeat(dopant_list, counts)
        np.random.shuffle(arr)
        return list(arr)



class MakeDataCard(MakeDataCardWidget):
    #通知下一个card执行
    separator=False
    card_name= "MakeDataCard"
    menu_icon=r":/images/src/images/logo.svg"
    runFinishedSignal=Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exportSignal.connect(self.export_data)
        self.dataset:list=None
        self.result_dataset=[]
        self.index=0
        # self.setFixedSize(400, 200)
        self.setting_widget = QWidget(self)
        self.viewLayout.setContentsMargins(3, 6, 3, 6)
        self.viewLayout.addWidget(self.setting_widget)
        self.settingLayout = QGridLayout(self.setting_widget)
        self.settingLayout.setContentsMargins(5, 0, 5,0)
        self.settingLayout.setSpacing(3)
        self.status_label = ProcessLabel(self)
        self.vBoxLayout.addWidget(self.status_label)
        self.windowStateChangedSignal.connect(self.show_setting)

    def show_setting(self ):
        if self.window_state == "expand":
            self.setting_widget.show( )

        else:
            self.setting_widget.hide( )

    def set_dataset(self,dataset):
        self.dataset = dataset
        self.result_dataset = []

        self.update_dataset_info()

    def write_result_dataset(self, file,**kwargs):
        ase_write(file,self.result_dataset,**kwargs)

    def export_data(self):

        if self.dataset is not None:

            path = utils.call_path_dialog(self, "Choose a file save location", "file",f"export_{self.getTitle().replace(' ', '_')}_structure.xyz")
            if not path:
                return
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.write_result_dataset, path)

    def process_structure(self, structure:Atoms) :
        """
        自定义对每个结构的处理 最后返回一个处理后的结构列表
        """
        raise NotImplementedError

    def closeEvent(self, event):

        if hasattr(self, "worker_thread"):

            if self.worker_thread.isRunning():

                self.worker_thread.terminate()
                self.runFinishedSignal.emit(self.index)

        self.deleteLater()
        super().closeEvent(event)

    def stop(self):
        if hasattr(self, "worker_thread"):
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.result_dataset = self.worker_thread.result_dataset
                self.update_dataset_info()
                del self.worker_thread

    def run(self):
        # 创建并启动线程

        if self.check_state:
            self.worker_thread = utils.DataProcessingThread(
                self.dataset,
                self.process_structure
            )
            self.status_label.set_colors(["#59745A" ])

            # 连接信号
            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)
        # self.worker_thread.wait()

    def update_progress(self, progress):
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):
        # self.status_label.setText("Processing finished")

        self.result_dataset = self.worker_thread.result_dataset
        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7" ])
        self.runFinishedSignal.emit(self.index)
        del self.worker_thread

    def on_processing_error(self, error):
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red" ])
        self.result_dataset = self.worker_thread.result_dataset
        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")



    def update_dataset_info(self ):
        text = f"Input structures: {len(self.dataset)} → Output: {len(self.result_dataset)}"
        self.status_label.setText(text)

class FilterDataCard(MakeDataCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Filter Data")

    def stop(self):

        if hasattr(self, "worker_thread"):

            if self.worker_thread.isRunning():
                self.worker_thread.terminate()


                self.result_dataset = []
                self.update_dataset_info()
                del self.worker_thread

    def update_progress(self, progress):
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):

        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7" ])
        self.runFinishedSignal.emit(self.index)
        if hasattr(self, "worker_thread"):
            del self.worker_thread

    def on_processing_error(self, error):
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red" ])

        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")

    def update_dataset_info(self ):
        text = f"Input structures: {len(self.dataset)} → Selected: {len(self.result_dataset)}"
        self.status_label.setText(text)

from ._card.super_cell_card import SuperCellCard
from ._card.vacancy_defect_card import VacancyDefectCard
from ._card.perturb_card import PerturbCard
from ._card.random_doping_card import RandomDopingCard
from ._card.random_vacancy_card import RandomVacancyCard
from ._card.random_slab_card import RandomSlabCard
from ._card.cell_scaling_card import CellScalingCard
from ._card.cell_strain_card import CellStrainCard
from ._card.fps_filter_card import FPSFilterDataCard
from ._card.card_group import CardGroup

user_config_path = get_user_config_path()
if os.path.exists(f"{user_config_path}/cards"):
    load_cards_from_directory(os.path.join(user_config_path, "cards"))

class ConsoleWidget(QWidget):
    """
控制台"""
    newCardSignal = Signal(str)  # 定义一个信号，用于通知上层组件新增卡片
    stopSignal = Signal()
    runSignal = Signal( )
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleWidget")
        self.setMinimumHeight(50)
        self.init_ui()

    def init_ui(self):
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("console_gridLayout")
        self.setting_command =CommandBar(self)
        self.new_card_button = PrimaryDropDownPushButton(QIcon(":/images/src/images/copy_figure.svg"),
                                                         "Add new card",self)
        self.new_card_button.setMaximumWidth(200 )
        self.new_card_button.setObjectName("new_card_button")

        self.new_card_button.setToolTip("Add a new card")
        self.new_card_button.installEventFilter(ToolTipFilter(self.new_card_button, 300, ToolTipPosition.TOP))

        self.menu = RoundMenu(parent=self)
        for class_name,card_class in card_info_dict.items():
            if card_class.separator:
                self.menu.addSeparator()
            action = QAction(QIcon(card_class.menu_icon),card_class.card_name)
            action.setObjectName(class_name)
            self.menu.addAction(action)


        self.menu.triggered.connect(self.menu_clicked)
        self.new_card_button.setMenu(self.menu)
        self.setting_command.addWidget(self.new_card_button)

        self.setting_command.addSeparator()
        run_action = Action(QIcon(r":/images/src/images/run.svg"), 'Run', triggered=self.run)
        run_action.setToolTip('Run selected cards')
        run_action.installEventFilter(ToolTipFilter(run_action, 300, ToolTipPosition.TOP))

        self.setting_command.addAction(run_action)
        stop_action = Action(QIcon(r":/images/src/images/stop.svg"), 'Stop', triggered=self.stop)
        stop_action.setToolTip('Stop running cards')
        stop_action.installEventFilter(ToolTipFilter(stop_action, 300, ToolTipPosition.TOP))

        self.setting_command.addAction(stop_action)



        self.gridLayout.addWidget(self.setting_command, 0, 0, 1, 1)

    def menu_clicked(self,action):


        self.newCardSignal.emit(action.objectName())

    def run(self,*args,**kwargs):
        self.runSignal.emit()
    def stop(self,*args,**kwargs):
        self.stopSignal.emit()
