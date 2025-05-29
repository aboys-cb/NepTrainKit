#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
import importlib
import os
import time
from itertools import combinations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QGridLayout, QFrame, QWidget, QVBoxLayout

from qfluentwidgets import ComboBox, BodyLabel, RadioButton, SplitToolButton, RoundMenu, PrimaryDropDownToolButton, \
    PrimaryDropDownPushButton, CommandBar, Action, CheckBox, LineEdit, EditableComboBox, PlainTextEdit, ToolTip, \
    ToolTipFilter, ToolTipPosition
from NepTrainKit import utils, module_path,get_user_config_path

from NepTrainKit.core import MessageManager
from NepTrainKit.core.custom_widget import (
    SpinBoxUnitInputFrame,
    MakeDataCardWidget,
    ProcessLabel
)
from NepTrainKit.core.custom_widget import DopingRulesWidget, VacancyRulesWidget
from NepTrainKit.core.calculator import NEPProcess
from NepTrainKit.core.io.select import farthest_point_sampling
from scipy.sparse.csgraph import connected_components
from scipy.stats.qmc import Sobol
from ase import neighborlist
from ase.io import extxyz,cif,vasp

from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.build import make_supercell,surface
from ase import Atoms

card_info_dict = {}
def register_card_info(card_class  ):
    card_info_dict[card_class.__name__] =card_class

    return card_class


def load_cards_from_directory(directory: str) -> None:
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



def is_organic_cluster(symbols):
    """
    判断一个团簇是否为有机分子。
    规则：必须含有碳（C），通常还含有氢（H）或其他有机元素（O, N, S, P）。

    参数:
        symbols (list): 团簇中所有原子的化学符号列表。

    返回:
        bool: 如果是有机分子，返回 True，否则返回 False。
    """
    has_carbon = 'C' in symbols
    if not has_carbon:
        return False
    # 可选：强制要求含氢（H）或其他有机元素
    organic_elements = {'H', 'O', 'N', 'S', 'P'}
    has_organic_elements = any(symbol in organic_elements for symbol in symbols)
    return has_carbon and has_organic_elements



def get_clusters(structure):
    """
    识别结构中的团簇（连通分量）。

    参数:
        structure (ase.Atoms): 输入的 ASE Atoms 对象。

    返回:
        list: 每个团簇的原子索引列表。
        list: 每个团簇是否为有机分子的布尔值列表。
    """
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

def unwrap_molecule(atoms, cluster_indices):
    # 获取分子中原子的分数坐标
    scaled_pos = atoms.get_scaled_positions(wrap=False)[cluster_indices]

    center =atoms[cluster_indices].get_center_of_mass(True)
    cell=atoms.cell


    # 计算所有原子相对于质心的位移（分数坐标）
    delta = scaled_pos - center
    # 转换为笛卡尔坐标的位移
    delta_cartesian = np.dot(delta, cell)

    # 使用find_mic计算最小镜像位移
    mic_vectors, _ = find_mic(delta_cartesian, cell, pbc=True)

    # 将最小镜像位移转换回分数坐标
    mic_delta = np.dot(mic_vectors, np.linalg.inv(cell))

    # 更新分数坐标
    scaled_pos = center + mic_delta

    unwrapped_pos = np.dot(scaled_pos, atoms.cell)

    return unwrapped_pos

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

#加载自定义卡片
user_config_path=get_user_config_path()

if os.path.exists(f"{user_config_path}/cards"):
    load_cards_from_directory(os.path.join(user_config_path,"cards"))




@register_card_info
class SuperCellCard(MakeDataCard):
    card_name= "Super Cell"
    menu_icon=r":/images/src/images/supercell.svg"
    separator = True
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Supercell")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("super_cell_card_widget")
        self.behavior_type_combo=ComboBox(self.setting_widget)
        self.behavior_type_combo.addItem("Maximum")
        self.behavior_type_combo.addItem("Iteration")

        self.combo_label=BodyLabel("behavior:",self.setting_widget)
        self.combo_label.setToolTip("Select supercell generation method")
        self.combo_label.installEventFilter(ToolTipFilter(self.combo_label, 300, ToolTipPosition.TOP))

        self.super_scale_radio_button = RadioButton("super scale",self.setting_widget)
        self.super_scale_radio_button.setChecked(True)
        self.super_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_scale_condition_frame.set_input("",3)
        self.super_scale_condition_frame.setRange(1,100)
        self.super_scale_radio_button.setToolTip("Scale factors along axes")
        self.super_scale_radio_button.installEventFilter(ToolTipFilter(self.super_scale_radio_button, 300, ToolTipPosition.TOP))

        self.super_cell_radio_button = RadioButton("super cell",self.setting_widget)
        self.super_cell_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_cell_condition_frame.set_input("Å",3)
        self.super_cell_condition_frame.setRange(1,100)
        self.super_cell_radio_button.setToolTip("Target lattice constant in Å")
        self.super_cell_radio_button.installEventFilter(ToolTipFilter(self.super_cell_radio_button, 300, ToolTipPosition.TOP))


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)
        # self.max_atoms_condition_frame.setToolTip("Maximum allowed atoms")

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

    def _get_max_cell_factors(self, structure:Atoms) -> List[Tuple[int, int, int]]:
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


@register_card_info
class VacancyDefectCard(MakeDataCard):
    card_name= "Vacancy Defect Generation"
    menu_icon=r":/images/src/images/defect.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Vacancy Defect")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("vacancy_defect_card_widget")

        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")

        self.num_radio_button = RadioButton("Vacancy num",self.setting_widget)
        self.num_radio_button.setChecked(True)
        self.num_radio_button.setToolTip("Set fixed number of vacancies")
        self.num_radio_button.installEventFilter(ToolTipFilter(self.num_radio_button, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1)
        self.num_condition_frame.setRange(1,10000)


        self.concentration_radio_button = RadioButton("Vacancy concentration",self.setting_widget)
        self.concentration_radio_button.setToolTip("Set vacancy concentration")
        self.concentration_radio_button.installEventFilter(ToolTipFilter(self.concentration_radio_button, 300, ToolTipPosition.TOP))


        self.concentration_condition_frame = SpinBoxUnitInputFrame(self)
        self.concentration_condition_frame.set_input("",1,"float")
        self.concentration_condition_frame.setRange(0,1)


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)

        self.max_atoms_label= BodyLabel("Max num",self.setting_widget)
        self.max_atoms_label.setToolTip("Number of structures to generate")

        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        #
        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.num_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.concentration_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.concentration_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 3, 1, 1, 2)

    def process_structure(self,structure):
        structure_list = []
        engine_type = self.engine_type_combo.currentIndex()
        concentration = self.concentration_condition_frame.get_input_value()[0]

        defect_num = self.num_condition_frame.get_input_value()[0]

        max_num = self.max_atoms_condition_frame.get_input_value()[0]

        n_atoms = len(structure)
        if self.concentration_radio_button.isChecked():
            max_defects = int(concentration * n_atoms)
        else:
            max_defects =  defect_num  # 固定数量
        if max_defects ==n_atoms:
            max_defects=max_defects-1

        if engine_type == 0:
            # 为数量和位置分配维度：1 维用于数量，n_atoms 维用于位置
            sobol_engine = Sobol(d=n_atoms + 1, scramble=True)
            sobol_seq = sobol_engine.random(max_num)  # 生成 [0, 1] 的序列
        else:
            # Uniform 模式下分开处理
            defect_counts = np.random.randint(1, max_defects + 1, max_num)

        for i in range(max_num):
            new_structure =structure.copy()

            # 确定当前结构的缺陷数量
            if engine_type == 0:

                target_defects = 1 + int(sobol_seq[i, 0] * max_defects)  # [0, 1] -> [1, max_defects]
                target_defects = min(target_defects, max_defects)  # 确保不超过 max_defects
                # 使用 Sobol 第 0 维控制数量
                # 使用剩余维度控制位置
                position_scores = sobol_seq[i, 1:]
            else:
                target_defects = defect_counts[i]

            if target_defects == 0:
                structure_list.append(new_structure)
                continue
            if engine_type == 0:
                sorted_indices = np.argsort(position_scores)
                defect_indices = sorted_indices[:target_defects]
            else:

                defect_indices = np.random.choice(n_atoms, target_defects, replace=False)
            # 创建空位
            mask = np.zeros(n_atoms, dtype=bool)
            mask[defect_indices] = True
            n_vacancies = np.sum(mask)
            del new_structure[mask]
            new_structure.info["Config_type"] = new_structure.info.get("Config_type","") + f" Vacancy(num={n_vacancies})"
            structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()

        data_dict['engine_type'] = self.engine_type_combo.currentIndex()
        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        data_dict["num_radio_button"]=self.num_radio_button.isChecked()
        data_dict["concentration_radio_button"]=self.concentration_radio_button.isChecked()
        data_dict['concentration_condition'] = self.concentration_condition_frame.get_input_value()
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()

        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.engine_type_combo.setCurrentIndex(data_dict['engine_type'])
        self.num_condition_frame.set_input_value(data_dict['num_condition'])
        self.concentration_condition_frame.set_input_value(data_dict['concentration_condition'])
        self.max_atoms_condition_frame.set_input_value(data_dict['max_atoms_condition'])
        self.concentration_radio_button.setChecked(data_dict['concentration_radio_button'])
        self.num_radio_button.setChecked(data_dict['num_radio_button'])

@register_card_info
class PerturbCard(MakeDataCard):
    card_name= "Atomic Perturb"
    menu_icon=r":/images/src/images/perturb.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Atomic Perturb")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("perturb_card_widget")
        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(True)
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))



        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("Å",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_radio_label=BodyLabel("Max distance:",self.setting_widget)
        self.scaling_condition_frame.set_input_value([0.3])
        self.scaling_radio_label.setToolTip("Maximum displacement distance")
        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_condition_frame.set_input_value([50])

        self.num_label=BodyLabel("Max num:",self.setting_widget)
        self.num_label.setToolTip("Number of structures to generate")

        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame,1, 1, 1, 2)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)

        self.settingLayout.addWidget(self.num_label,3, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame,3, 1, 1,2)

    def process_structure(self, structure):
        structure_list=[]
        engine_type=self.engine_type_combo.currentIndex()
        max_scaling=self.scaling_condition_frame.get_input_value()[0]
        max_num=self.num_condition_frame.get_input_value()[0]
        identify_organic=self.organic_checkbox.isChecked()
        n_atoms = len(structure)
        dim = n_atoms * 3  # 每个原子有 x, y, z 三个维度

        if engine_type == 0:

            sobol_engine = Sobol(d=dim, scramble=True)
            sobol_seq = sobol_engine.random(max_num)  # 生成 [0, 1] 的序列
            perturbation_factors = (sobol_seq - 0.5) * 2  # 转换为 [-1, 1]
        else:
            # 生成均匀分布的扰动因子，范围 [-1, 1]
            perturbation_factors = np.random.uniform(-1, 1, (max_num, dim))

            # 识别团簇和有机分子
        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        orig_positions = structure.positions
        for i in range(max_num):
            new_structure = structure.copy()

            # 提取当前结构的扰动因子并重塑为 (n_atoms, 3)
            delta = perturbation_factors[i].reshape(n_atoms, 3) * max_scaling
            if identify_organic:
                # 对每个团簇应用微扰
                new_positions = orig_positions.copy()

                for cluster_indices, is_organic in zip(clusters, is_organic_list):
                    if is_organic:
                        # 有机分子：整体平移，应用统一的偏移向量
                        # 从团簇的第一个原子的扰动因子中取偏移向量
                        cluster_delta = delta[cluster_indices[0]]
                        for idx in cluster_indices:
                            new_positions[idx] += cluster_delta
                    else:
                        # 非有机分子：逐原子微扰
                        for idx in cluster_indices:
                            new_positions[idx] += delta[idx]
            else:
                new_positions=orig_positions+delta

            # 更新新结构的坐标
            new_structure.set_positions(new_positions)
            new_structure.info["Config_type"] = new_structure.info.get("Config_type","") + f" Perturb(distance={max_scaling}, {'uniform' if engine_type == 1 else 'Sobol'})"
            structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()


        data_dict['engine_type'] = self.engine_type_combo.currentIndex()
        data_dict["organic"]=self.organic_checkbox.isChecked()
        data_dict['scaling_condition'] = self.scaling_condition_frame.get_input_value()

        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.engine_type_combo.setCurrentIndex(data_dict['engine_type'])

        self.scaling_condition_frame.set_input_value(data_dict['scaling_condition'])

        self.num_condition_frame.set_input_value(data_dict['num_condition'])
        self.organic_checkbox.setChecked(data_dict.get("organic", False))

@register_card_info
class RandomDopingCard(MakeDataCard):
    card_name = "Random Doping"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Doping Replacement")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_doping_card_widget")


        self.rules_label = BodyLabel("Rules", self.setting_widget)
        self.rules_widget = DopingRulesWidget(self.setting_widget)
        self.rules_label.setToolTip("doping rules")
        self.rules_label.installEventFilter(ToolTipFilter(self.rules_label, 300, ToolTipPosition.TOP))

        self.doping_label = BodyLabel("Doping", self.setting_widget)

        self.doping_type_combo=ComboBox(self.setting_widget)
        self.doping_type_combo.addItem("Random")
        self.doping_type_combo.addItem("Exact")
        self.doping_label.setToolTip("Select doping algorithm")
        self.doping_label.installEventFilter(ToolTipFilter(self.doping_label, 300, ToolTipPosition.TOP))

        self.max_atoms_label = BodyLabel("Max structures", self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_label.setToolTip("Number of structures to generate")
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.rules_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.rules_widget, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.doping_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.doping_type_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 2, 1, 1, 2)

    def process_structure(self, structure):
        structure_list = []

        rules = self.rules_widget.to_rules()
        if not isinstance(rules, list) or not rules:
            return [structure]

        max_num = self.max_atoms_condition_frame.get_input_value()[0]
        exact = self.doping_type_combo.currentText()=="Exact"
        for _ in range(max_num):
            new_structure = structure.copy()
            total_doping = 0
            for rule in rules:

                target = rule.get("target")
                dopants = rule.get("dopants", {})
                if not target or not dopants:
                    continue

                groups = rule.get("group")

                if groups and "group" in new_structure.arrays:

                    candidate_indices = [i for i,elem,g in zip(range(len(new_structure)), new_structure ,new_structure.arrays["group"]) if elem.symbol == target and g in groups]
                else:
                    candidate_indices = [i for i, a in enumerate(new_structure) if a.symbol == target]

                if not candidate_indices:
                    continue

                if "concentration" == rule["use"]:
                    conc_min, conc_max = rule.get("concentration", [0.0, 1.0])
                    conc = np.random.uniform(float(conc_min), float(conc_max))
                    doping_num = max(1, int(len(candidate_indices) * conc))
                elif "count" == rule["use"]:
                    count_min, count_max = rule.get("count", [1, 1])
                    doping_num = np.random.randint(int(count_min), int(count_max) + 1)
                else:
                    doping_num = len(candidate_indices)

                doping_num = min(doping_num, len(candidate_indices))

                idxs = np.random.choice(candidate_indices, doping_num, replace=False)



                dopant_list = list(dopants.keys())
                ratios = np.array(list(dopants.values()), dtype=float)
                ratios = ratios / ratios.sum()
                sample = sample_dopants(dopant_list,ratios,doping_num,exact )


                for idx,elem in zip(idxs,sample):
                    new_structure[idx].symbol = elem
                total_doping += doping_num
            if total_doping:
                new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Doping(num={total_doping})"

            structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()

        data_dict['rules'] = json.dumps(self.rules_widget.to_rules(), ensure_ascii=False)
        data_dict['doping_type'] = self.doping_type_combo.currentText()
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        rules = data_dict.get('rules', '')
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                rules = []
        self.rules_widget.from_rules(rules)
        self.doping_type_combo.setCurrentText(data_dict.get("doping_type","Exact"))
        self.max_atoms_condition_frame.set_input_value(data_dict.get('max_atoms_condition', [1]))


@register_card_info
class RandomVacancyCard(MakeDataCard):
    card_name = "Random Vacancy"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Vacancy Delete")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_vacancy_card_widget")

        self.rules_label = BodyLabel("Rules", self.setting_widget)
        self.rules_widget = VacancyRulesWidget(self.setting_widget)
        self.rules_label.setToolTip("vacancy rules")
        self.rules_label.installEventFilter(ToolTipFilter(self.rules_label, 300, ToolTipPosition.TOP))

        self.max_atoms_label = BodyLabel("Max structures", self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_label.setToolTip("Number of structures to generate")
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.rules_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.rules_widget, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 1, 1, 1, 2)

    def process_structure(self, structure):
        structure_list = []

        rules = self.rules_widget.to_rules()
        if not isinstance(rules, list) or not rules:
            return [structure]

        max_num = self.max_atoms_condition_frame.get_input_value()[0]
        for _ in range(max_num):
            new_structure = structure.copy()
            total_remove = 0
            for rule in rules:
                element = rule.get("element")
                count_min, count_max = rule.get("count", [0, 0])
                if not element or int(count_max) <= 0:
                    continue

                groups = rule.get("group")
                if groups and "group" in new_structure.arrays:
                    candidate_indices = [i for i, elem, g in zip(range(len(new_structure)), new_structure, new_structure.arrays["group"]) if elem.symbol == element and g in groups]
                else:
                    candidate_indices = [i for i, a in enumerate(new_structure) if a.symbol == element]

                if not candidate_indices:
                    continue

                remove_num = np.random.randint(int(count_min), int(count_max) + 1)
                remove_num = min(remove_num, len(candidate_indices))
                if remove_num <= 0:
                    continue

                idxs = np.random.choice(candidate_indices, remove_num, replace=False)
                for idx in sorted(idxs, reverse=True):
                    del new_structure[idx]
                total_remove += remove_num

            if total_remove:
                new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Vacancy(num={total_remove})"

            structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()

        data_dict['rules'] = json.dumps(self.rules_widget.to_rules(), ensure_ascii=False)
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        rules = data_dict.get('rules', '')
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                rules = []
        self.rules_widget.from_rules(rules)
        self.max_atoms_condition_frame.set_input_value(data_dict.get('max_atoms_condition', [1]))


@register_card_info
class RandomSlabCard(MakeDataCard):
    card_name = "Random Slab"
    menu_icon = r":/images/src/images/supercell.svg"

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
                                slab = surface(structure, (int(h), int(k), int(l)), int(layers), vacuum=float(vac))
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
@register_card_info
class CellScalingCard(MakeDataCard):
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
        self.organic_checkbox.setChecked(True)
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
                #判断下有没有有机分子  如果有 就将有机分子做整体的偏移 而不是拉伸
                for cluster_indices, is_organic in zip(clusters, is_organic_list):
                    if is_organic:

                        unwrap_old_pos = unwrap_molecule(structure,cluster_indices)
                        unwrap_new_pos = unwrap_molecule(new_structure,cluster_indices)
                        distance = unwrap_new_pos[0] - unwrap_old_pos[0]

                        pos = unwrap_old_pos + distance
                        new_structure.positions[cluster_indices] = pos
                        new_structure.wrap()

                    else:
                        pass

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
@register_card_info
class CellStrainCard(MakeDataCard):
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
        self.organic_checkbox.setChecked(True)
        self.organic_checkbox.setToolTip("Treat organic molecules as rigid units")
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))


        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.strain_x_label=BodyLabel("X:",self.setting_widget)
        self.strain_x_frame = SpinBoxUnitInputFrame(self)
        self.strain_x_frame.set_input(["-","% step:","%"],3,"int")
        self.strain_x_frame.setRange(-100,100)
        self.strain_x_frame.set_input_value([-5,5,1])
        self.strain_x_label.setToolTip("X-axis strain range")
        self.strain_x_label.installEventFilter(ToolTipFilter(self.strain_x_label, 300, ToolTipPosition.TOP))

        self.strain_y_label=BodyLabel("Y:",self.setting_widget)
        self.strain_y_frame = SpinBoxUnitInputFrame(self)
        self.strain_y_frame.set_input(["-","% step:","%"],3,"int")
        self.strain_y_frame.setRange(-100,100)
        self.strain_y_frame.set_input_value([-5,5,1])
        self.strain_y_label.setToolTip("Y-axis strain range")
        self.strain_y_label.installEventFilter(ToolTipFilter(self.strain_y_label, 300, ToolTipPosition.TOP))

        self.strain_z_label=BodyLabel("Z:",self.setting_widget)
        self.strain_z_frame = SpinBoxUnitInputFrame(self)
        self.strain_z_frame.set_input(["-","% step:","%"],3,"int")
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
            np.arange(start=x[0],stop=x[1]+1,step=x[2]),
            np.arange(start=y[0], stop=y[1]+1, step=y[2]),
            np.arange(start=z[0], stop=z[1]+1, step=z[2]),
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
                    for cluster_indices, is_organic in zip(clusters, is_organic_list):
                        if is_organic:
                            unwrap_old_pos = unwrap_molecule(structure, cluster_indices)
                            unwrap_new_pos = unwrap_molecule(new_structure, cluster_indices)
                            distance = unwrap_new_pos[0] - unwrap_old_pos[0]
                            pos = unwrap_old_pos + distance
                            new_structure.positions[cluster_indices] = pos
                            new_structure.wrap()
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
                        for cluster_indices, is_organic in zip(clusters, is_organic_list):
                            if is_organic:
                                unwrap_old_pos = unwrap_molecule(structure, cluster_indices)
                                unwrap_new_pos = unwrap_molecule(new_structure, cluster_indices)
                                distance = unwrap_new_pos[0] - unwrap_old_pos[0]
                                pos = unwrap_old_pos + distance
                                new_structure.positions[cluster_indices] = pos
                                new_structure.wrap()
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



@register_card_info
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

@register_card_info
class CardGroup(MakeDataCardWidget):
    separator=True
    card_name= "Card Group"
    menu_icon=r":/images/src/images/group.svg"
    #通知下一个card执行
    runFinishedSignal=Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Card Group")
        self.setAcceptDrops(True)
        self.index=0
        self.group_widget = QWidget(self)
        # self.setStyleSheet("CardGroup{boder: 2px solid #C0C0C0;}")
        self.viewLayout.addWidget(self.group_widget)
        self.group_layout = QVBoxLayout(self.group_widget)
        self.exportSignal.connect(self.export_data)
        self.windowStateChangedSignal.connect(self.show_card_setting)
        self.filter_widget = QWidget(self)
        self.filter_layout = QVBoxLayout(self.filter_widget)
        self.vBoxLayout.addWidget(self.filter_widget)

        self.filter_card=None
        self.dataset:list=None
        self.result_dataset=[]
        self.resize(400, 200)

    def set_filter_card(self,card):

        self.filter_card=card
        self.filter_layout.addWidget(card)

    def state_changed(self, state):
        super().state_changed(state)
        for card in self.card_list:
            card.state_checkbox.setChecked(state)

    @property
    def card_list(self)->["MakeDataCard"]:

        return [self.group_layout.itemAt(i).widget() for i in range(self.group_layout.count()) ]
    def show_card_setting(self):

        for card in self.card_list:
            card.window_state = self.window_state
            card.windowStateChangedSignal.emit()
    def set_dataset(self,dataset):
        self.dataset =dataset
        self.result_dataset=[]

    def add_card(self, card):
        self.group_layout.addWidget(card)

    def remove_card(self, card):
        self.group_layout.removeWidget(card)

    def clear_cards(self):
        for card in self.card_list:
            self.group_layout.removeWidget(card)

    def closeEvent(self, event):
        for card in self.card_list:
            card.close()
        self.deleteLater()
        super().closeEvent(event)

    def dragEnterEvent(self, event):

        widget = event.source()

        if widget == self:
            return
        if isinstance(widget, (MakeDataCard,CardGroup)):
            event.acceptProposedAction()
        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):

        widget = event.source()
        if widget == self:
            return
        if isinstance(widget, FilterDataCard):
            self.set_filter_card(widget)
        elif isinstance(widget, (MakeDataCard,CardGroup)):
            self.add_card(widget)
        event.acceptProposedAction()

    def on_card_finished(self, index):
        self.run_card_num-=1
        self.card_list[index].runFinishedSignal.disconnect(self.on_card_finished)
        self.result_dataset.extend(self.card_list[index].result_dataset)

        if self.run_card_num==0:
            self.runFinishedSignal.emit(self.index)
            if self.filter_card and self.filter_card.check_state:
                self.filter_card.set_dataset(self.result_dataset)
                self.filter_card.run()

    def stop(self):
        for card in self.card_list:
            card.stop()
        if self.filter_card:
            self.filter_card.stop()

    def run(self):
        # 创建并启动线程
        self.run_card_num = len(self.card_list)

        if self.check_state and self.run_card_num>0:
            self.result_dataset =[]
            for index,card in enumerate(self.card_list):
                if card.check_state:
                    card.set_dataset(self.dataset)
                    card.index=index
                    card.runFinishedSignal.connect(self.on_card_finished)
                    card.run()
                else:
                    self.run_card_num-=1
        else:
            self.result_dataset = self.dataset
            self.runFinishedSignal.emit(self.index)

    def write_result_dataset(self, file,**kwargs):
        if self.filter_card and self.filter_card.check_state:
            self.filter_card.write_result_dataset(file,**kwargs)
            return

        for index,card in enumerate(self.card_list):
            if index==0:
                if "append" not in kwargs:
                    kwargs["append"] = False
            else:
                kwargs["append"] = True
            if card.check_state:
                card.write_result_dataset(file,**kwargs)

    def export_data(self):
        if self.dataset is not None:
            path = utils.call_path_dialog(self, "Choose a file save location", "file",f"export_{self.getTitle()}_structure.xyz")
            if not path:
                return
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.write_result_dataset, path)
    def to_dict(self):
        data_dict = super().to_dict()

        data_dict["card_list"]=[]

        for card in self.card_list:
            data_dict["card_list"].append(card.to_dict())
        if self.filter_card:
            data_dict["filter_card"]=self.filter_card.to_dict()
        else:
            data_dict["filter_card"]=None

        return data_dict
    def from_dict(self,data_dict):
        self.state_checkbox.setChecked(data_dict['check_state'])
        for sub_card in data_dict.get("card_list",[]):
            card_name=sub_card["class"]
            card  = card_info_dict[card_name](self)
            self.add_card(card)
            card.from_dict(sub_card)

        if data_dict.get("filter_card"):
            card_name=data_dict["filter_card"]["class"]
            filter_card  = card_info_dict[card_name](self)
            filter_card.from_dict(data_dict["filter_card"])
            self.set_filter_card(filter_card)

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
