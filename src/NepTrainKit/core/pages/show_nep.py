#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:38
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path
import sys

from loguru import logger

import numpy as np
from PySide6.QtCore import QUrl, QTimer, Qt, Signal, QThread
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout,QSplitter
from qfluentwidgets import HyperlinkLabel, MessageBox, PlainTextEdit, CaptionLabel, SpinBox, \
    TransparentToolButton, StrongBodyLabel, getFont, ToolTipFilter, ToolTipPosition, TransparentToolButton, BodyLabel, \
    Action, StateToolTip

from NepTrainKit import utils
from NepTrainKit.core import MessageManager, Config
from NepTrainKit.core.custom_widget import ConfigTypeSearchLineEdit
from NepTrainKit.core.io import NepTrainResultData
from NepTrainKit.core.io.nep import NepPolarizabilityResultData, NepDipoleResultData
from NepTrainKit.core.io.utils import get_nep_type
from NepTrainKit.core.structure import table_info, atomic_numbers
from NepTrainKit.core.types import Brushes
from NepTrainKit.core.views import NepResultPlotWidget, NepDisplayGraphicsToolBar, StructurePlotWidget
from NepTrainKit.core.views.structure import StructureInfoWidget
from NepTrainKit.core.views.toolbar import StructureToolBar


class ShowNepWidget(QWidget):
    """
    针对NEP训练过程中 对预测结果的展示
    实现以下功能
        1.支持交互筛选训练集
        2.拖拽实现目录导入
        3.对于选定的结构 进行展示
    """
    updateBondInfoSignal=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("ShowNepWidget")
        self.setAcceptDrops(True)
        self.nep_result_data=None
        self.init_action()
        self.init_ui()
        self.updateBondInfoSignal.connect(self.bond_label.setText)

        self.first_show=False



    def showEvent(self, event):
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.addAction(self.export_selected_action)
        auto_load_config = Config.getboolean("widget","auto_load",False)
        if not auto_load_config:
            return
        if not self.first_show:
            self.first_show=True
            if os.path.exists("./train.xyz") and os.path.exists("./nep.txt"):
                self.set_work_path(os.path.join(os.getcwd(),"train.xyz"))

    def hideEvent(self, event):
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.removeAction(self.export_selected_action)

    def init_action(self):
        self.export_selected_action=Action(QIcon(":/images/src/images/export1.svg"),"Export Selected Structures")
        self.export_selected_action.triggered.connect(self.export_selected_structures)

    def init_ui(self):
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("show_nep_gridLayout")
        self.gridLayout.setContentsMargins(0,0,0,0)

        self.struct_widget = QWidget(self)
        self.struct_widget_layout = QGridLayout(self.struct_widget)

        self.show_struct_widget = StructurePlotWidget(self.struct_widget)
        # self.export_single_struct_button = TransparentToolButton(QIcon(':/images/src/images/export1.svg') ,self.struct_widget)
        # self.export_single_struct_button.clicked.connect(self.export_single_struct)
        self.structure_toolbar = StructureToolBar(self.struct_widget)
        self.structure_toolbar.showBondSignal.connect(self.show_struct_widget.set_show_bonds)
        self.structure_toolbar.orthoViewSignal.connect(self.show_struct_widget.set_projection)
        self.structure_toolbar.exportSignal.connect(self.export_single_struct)

        self.struct_info_widget = StructureInfoWidget(self.struct_widget)
        self.struct_index_widget = QWidget(self)
        self.struct_index_widget_layout = QHBoxLayout(self.struct_index_widget)
        self.struct_index_label = BodyLabel(self.struct_index_widget)
        self.struct_index_label.setText("Current structure (original file index):")

        self.struct_index_spinbox = SpinBox(self.struct_index_widget)

        self.struct_index_spinbox.upButton.clicked.disconnect(self.struct_index_spinbox.stepUp)
        self.struct_index_spinbox.downButton.clicked.disconnect(self.struct_index_spinbox.stepDown)
        self.struct_index_spinbox.downButton.clicked.connect(self.to_last_structure)
        self.struct_index_spinbox.upButton.clicked.connect(self.to_next_structure)
        self.struct_index_spinbox.setMinimum(0)
        self.struct_index_spinbox.setMaximum(0)
        self.play_timer=QTimer(self)
        self.play_timer.timeout.connect(self.play_show_structures)

        self.auto_switch_button = TransparentToolButton(QIcon(':/images/src/images/play.svg') ,self.struct_index_widget)
        self.auto_switch_button.clicked.connect(self.start_play)
        self.auto_switch_button.setCheckable(True)


        self.struct_index_widget_layout.addWidget(self.struct_index_label)
        self.struct_index_widget_layout.addWidget(self.struct_index_spinbox)

        self.struct_index_widget_layout.addWidget(self.auto_switch_button)
        self.struct_index_spinbox.valueChanged.connect(self.show_current_structure)

        self.bond_label=StrongBodyLabel(self.struct_widget)
        self.bond_label.setFont(getFont(20, QFont.DemiBold))
        # self.bond_label.setFixedHeight(30)  # 设置状态栏的高度
        self.bond_label.setWordWrap(True)
        # 添加到布局的底部
        # self.bond_label.setStyleSheet("QLabel { background-color: #f3f3f3; color: black; padding: 5px; }")
        self.bond_label.setToolTip('The Tip is the minimum distance between atoms in the current structure, in Å.')

        self.bond_label.installEventFilter(ToolTipFilter(self.bond_label, 300, ToolTipPosition.TOP))


        self.struct_widget_layout.addWidget(self.structure_toolbar, 0, 0, 1, 1)

        self.struct_widget_layout.addWidget(self.show_struct_widget, 1, 0, 1, 1)
        # self.struct_widget_layout.addWidget(self.export_single_struct_button, 1, 0, 1, 1, alignment=Qt.AlignRight)
        self.struct_widget_layout.addWidget(self.struct_info_widget, 2, 0, 1, 1)
        self.struct_widget_layout.addWidget(self.bond_label,3, 0, 1, 1)

        self.struct_widget_layout.addWidget(self.struct_index_widget, 4, 0, 1, 1)

        self.struct_widget_layout.setRowStretch(0, 3)
        self.struct_widget_layout.setRowStretch(1, 1)
        self.struct_widget_layout.setRowStretch(2, 0)
        self.struct_widget_layout.setSpacing(1)
        self.struct_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = QWidget(self)

        self.plot_widget_layout = QGridLayout(self.plot_widget)

        self.graph_widget = NepResultPlotWidget(self  )

        self.graph_widget.canvas.structureIndexChanged.connect(self.struct_index_spinbox.setValue)

        self.graph_toolbar = NepDisplayGraphicsToolBar(  self.plot_widget)
        self.graph_widget.set_tool_bar(self.graph_toolbar)

        self.search_lineEdit=ConfigTypeSearchLineEdit(self.plot_widget)
        self.search_lineEdit.searchSignal.connect(self.search_config_type)
        self.search_lineEdit.checkSignal.connect(self.checked_config_type)
        self.search_lineEdit.uncheckSignal.connect(self.uncheck_config_type)


        # 创建状态栏
        self.path_label = HyperlinkLabel(self.plot_widget)
        self.path_label.setFixedHeight(30)  # 设置状态栏的高度

        self.dataset_info_label = BodyLabel(self.plot_widget)
        self.dataset_info_label.setFixedHeight(30)  # 设置状态栏的高度


        self.plot_widget_layout.addWidget(self.graph_toolbar, 0, 0, 1, 2)

        self.plot_widget_layout.addWidget(self.search_lineEdit, 1, 0, 1, 2)
        self.plot_widget_layout.addWidget(self.graph_widget, 2, 0, 1, 2)
        self.plot_widget_layout.addWidget(self.path_label , 3, 0, 1, 1)
        self.plot_widget_layout.addWidget(self.dataset_info_label , 3, 1, 1, 1)
        self.plot_widget_layout.setContentsMargins(0,0,0,0)

        # 将状态栏添加到布局的底部
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.struct_widget)
        self.splitter.setSizes([400,200])
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)

    def dragEnterEvent(self, event):
        # 检查拖拽的内容是否包含文件
        if event.mimeData().hasUrls():
            event.acceptProposedAction()  # 接受拖拽事件
        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):
        # 获取拖拽的文件路径
        urls = event.mimeData().urls()
        if urls:
            # 获取第一个文件路径
            file_path = urls[0].toLocalFile()

            self.set_work_path(file_path)

    def open_file(self):
        path = utils.call_path_dialog(self,"Please choose the XYZ file","select",file_filter="XYZ files (*.xyz)")
        if path:
            self.set_work_path(path)

    def export_file(self):
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        path=utils.call_path_dialog(self,"Choose a file save location","directory")
        if path:
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.nep_result_data.export_model_xyz, path)

    def export_selected_structures(self):
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        if len(self.nep_result_data.select_index)==0:
            MessageManager.send_info_message("Please select some structures first！")
            return
        path = utils.call_path_dialog(self,"Please choose the XYZ file","file",file_filter="XYZ files (*.xyz)",default_filename="selected_structures.xyz")
        if path:
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.nep_result_data.export_selected_xyz, path)

    def set_work_path(self,path):
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path,"train.xyz")):
                path=os.path.join(path,"train.xyz")
            else:
                MessageManager.send_info_message("The directory does not contain a train.xyz file!")
                return
        if not path.endswith(".xyz"):
            MessageManager.send_info_message(f"Please choose a xyz file, not {path}!")
            return
        url=self.path_label.getUrl().toString()
        old_path=url.replace("file://","")
        if sys.platform == "win32":
            old_path=old_path[1:]
        else:
            pass
        if os.path.exists(old_path):
            box=MessageBox("Ask","A working directory already exists. Loading a new directory will erase the previous results.\nDo you want to load the new working path?",self)
            box.exec_()
            if box.result()==0:
                return

        self.check_nep_result(path)

    def set_dataset(self,*args):
        if self.nep_result_data is None:
            return
        if not self.nep_result_data.load_flag :
            self.nep_result_data=None
            return
        self.struct_index_spinbox.setMaximum(self.nep_result_data.num)
        self.graph_widget.set_dataset(self.nep_result_data)
        self.nep_result_data.updateInfoSignal.connect(self.update_dataset_info)
        self.nep_result_data.updateInfoSignal.emit()
        self.search_lineEdit.setCompleterKeyWord(self.nep_result_data.structure.get_all_config())
        self.struct_index_spinbox.valueChanged.emit(0)

    def check_nep_result(self,path):
        """
        检查输出文件都有什么
        然后设置窗口布局
        :return:
        """

        dir_path = os.path.dirname(path)
        file_name = os.path.basename(path)
        model_type=get_nep_type(os.path.join(dir_path,"nep.txt"))
        logger.info(f"NEP model type: {model_type}")
        if model_type==0:
            self.nep_result_data=NepTrainResultData.from_path(path)
        elif model_type==1:
            self.nep_result_data=NepDipoleResultData.from_path(path)
        elif model_type==2:
            self.nep_result_data=NepPolarizabilityResultData.from_path(path)
        else:
            self.nep_result_data=None

        if self.nep_result_data is None:
            return

        self.path_label.setText(f"Current file: {file_name}")
        self.path_label.setUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        # self.graph_widget.set_dataset(self.dataset)
        self.load_thread=QThread(self)
        tip = StateToolTip("Loading", 'Please wait patiently~~', self )
        tip.show()
        tip.closedSignal.connect(self.stop_loading)
        self.nep_result_data.moveToThread(self.load_thread)
        self.load_thread.finished.connect(self.set_dataset)
        self.load_thread.finished.connect(lambda :tip.setState(True))

        self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
        self.load_thread.started.connect(self.nep_result_data.load)
        self.load_thread.start()

        # self.nep_result_data.load()
    def stop_loading(self):

        self.load_thread.terminate()
        if self.nep_result_data is not None:
            self.nep_result_data.nep_calc_thread.stop()
    def to_last_structure(self):

        if self.nep_result_data is None:
            return None
        current_index = self.struct_index_spinbox.value()
        if self.nep_result_data.select_index:

            sort_index = np.sort(np.array(list(self.nep_result_data.select_index)) )
        else:
            sort_index = np.sort(self.nep_result_data.structure.group_array.now_data, axis=0)
        index = np.searchsorted(sort_index, current_index, side='left')

        self.struct_index_spinbox.setValue(int(sort_index[index-1 if index>0 else index]))

    # @utils.timeit
    def to_next_structure(self):
        if self.nep_result_data is None:
            return None
        current_index=self.struct_index_spinbox.value()
        if self.nep_result_data.select_index:
            sort_index = np.sort(np.array(list(self.nep_result_data.select_index)) )

        else:
            sort_index = np.sort(self.nep_result_data.structure.group_array.now_data, axis=0)
        index = np.searchsorted(sort_index, current_index, side='right')
        if index>=sort_index.shape[0]:
            return False
        self.struct_index_spinbox.setValue(int(sort_index[index]))

        if index==sort_index.shape[0]-1:
            return True
        else:
            return False

    def start_play(self):
        if self.auto_switch_button.isChecked():
            self.auto_switch_button.setIcon(QIcon(':/images/src/images/pause.svg'))
            self.play_timer.start(50)
        else:
            self.auto_switch_button.setIcon(QIcon(':/images/src/images/play.svg'))
            self.play_timer.stop()

    def play_show_structures(self):
        if self.to_next_structure():
            self.auto_switch_button.click()
    def export_single_struct(self):
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        index=self.struct_index_spinbox.value()
        atoms=self.nep_result_data.get_atoms(index)
        path=utils.call_path_dialog(self,"Choose a file save location","file",
                                    file_filter="XYZ files (*.xyz)",
                                    default_filename=f"structure_{index}.xyz")
        if path:
            with open(path,"w",encoding="utf-8") as f:
                atoms.write(f)


    # @utils.timeit
    def show_current_structure(self,current_index):

        try:
            atoms=self.nep_result_data.get_atoms(current_index)
        except:
            MessageManager.send_message_box("The index is invalid, perhaps the structure has been deleted")
            return

        self.graph_widget.canvas.plot_current_point(current_index)

        self.show_struct_widget.show_structure(atoms)
        self.update_structure_bond_info(atoms)
        self.struct_info_widget.show_structure_info(atoms)

    def update_structure_bond_info(self,atoms):
        self.calculate_bond_thread=utils.LoadingThread(self,show_tip=False )
        self.calculate_bond_thread.start_work(self.calculate_bond_info,atoms)

    def calculate_bond_info(self,atoms):
        distance_info = atoms.get_mini_distance_info()
        bond_text = ""
        radius_coefficient_config = Config.getfloat("widget","radius_coefficient",0.7)
        unreasonable = False
        for elems,bond_length in distance_info.items():
            elem0_info = table_info[str(atomic_numbers[elems[0]])]
            elem1_info = table_info[str(atomic_numbers[elems[1]])]
            #相邻原子距离小于共价半径之和×系数就标红
            if (elem0_info["radii"] + elem1_info["radii"]) * radius_coefficient_config > bond_length*100:
                bond_text += f"{elems[0]}-{elems[1]}:"

                bond_text += f'<font color="red">{bond_length:.2f}</font> Å | '
                unreasonable = True
            # else:
            #     bond_text+=f'<font color="green">{bond_length:.2f}</font> Å | '
        self.updateBondInfoSignal.emit( bond_text )
        if unreasonable:
            MessageManager.send_info_message("The distance between atoms is too small, and the structure may be unreasonable.")

    def search_config_type(self,config):

        indexes= self.nep_result_data.structure.search_config(config)

        self.graph_widget.canvas.update_scatter_color(indexes,Brushes.Show)

    def checked_config_type(self, config):

        indexes = self.nep_result_data.structure.search_config(config)
        self.graph_widget.canvas.select_index(indexes,  False)

    def uncheck_config_type(self, config):

        indexes = self.nep_result_data.structure.search_config(config)
        self.graph_widget.canvas.select_index(indexes,True )
    def update_dataset_info(self ):
        info=f"Data: Orig: {self.nep_result_data.atoms_num_list.shape[0]} Now: {self.nep_result_data.structure.now_data.shape[0]} "\
        f"Rm: {self.nep_result_data.structure.remove_data.shape[0]} Sel: {len(self.nep_result_data.select_index)} Unsel: {self.nep_result_data.structure.now_data.shape[0]-len(self.nep_result_data.select_index)}"
        self.dataset_info_label.setText(info)