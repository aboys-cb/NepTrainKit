#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 22:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import time

from NepTrainKit.core.calculator import NEPProcess

start=time.time()
import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QWidget, QProgressDialog


from NepTrainKit import utils
from NepTrainKit.core import MessageManager, Config
from NepTrainKit.custom_widget import (
    GetIntMessageBox,
    SparseMessageBox,
    IndexSelectMessageBox,
    RangeSelectMessageBox,
    ShiftEnergyMessageBox, DFTD3MessageBox,
)
from NepTrainKit.core.io.select import farthest_point_sampling
from NepTrainKit.views.toolbar import NepDisplayGraphicsToolBar
from NepTrainKit.core.energy_shift import shift_dataset_energy, suggest_group_patterns


class NepResultPlotWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self._parent=parent

        self.draw_mode=False
        # self.setRenderHint(QPainter.Antialiasing, False)
        self._layout = QHBoxLayout(self)
        self.setLayout(self._layout)
        canvas_type = Config.get("widget","canvas_type","pyqtgraph")
        self.last_figure_num=None
        self.swith_canvas(canvas_type)

    def swith_canvas(self,canvas_type="pyqtgraph"):

        if canvas_type == "pyqtgraph":
            from NepTrainKit.core.canvas.pyqtgraph import PyqtgraphCanvas
            self.canvas = PyqtgraphCanvas(self)
            self._layout.addWidget(self.canvas)

        elif canvas_type == "vispy":


            from NepTrainKit.core.canvas.vispy import VispyCanvas


            self.canvas = VispyCanvas(parent=self, bgcolor='white')
            self._layout.addWidget(self.canvas.native)
            # self.window().windowHandle().screenChanged.connect(self.canvas.native.screen_changed)



    # def clear(self):
    #     self.canvas.clear_axes()
        # self.last_figure_num=None

    def set_tool_bar(self, tool):
        self.tool_bar: NepDisplayGraphicsToolBar = tool
        self.tool_bar.panSignal.connect(self.canvas.pan)
        self.tool_bar.resetSignal.connect(self.canvas.auto_range)
        self.tool_bar.deleteSignal.connect(self.canvas.delete)
        self.tool_bar.revokeSignal.connect(self.canvas.revoke)
        self.tool_bar.penSignal.connect(self.canvas.pen)
        self.tool_bar.exportSignal.connect(self.export_descriptor_data)
        self.tool_bar.findMaxSignal.connect(self.find_max_error_point)
        self.tool_bar.discoverySignal.connect(self.find_non_physical_structures)
        self.tool_bar.sparseSignal.connect(self.sparse_point)
        self.tool_bar.shiftEnergySignal.connect(self.shift_energy_baseline)
        self.tool_bar.inverseSignal.connect(self.inverse_select)
        self.tool_bar.selectIndexSignal.connect(self.select_by_index)
        self.tool_bar.rangeSignal.connect(self.select_by_range)
        self.tool_bar.dftd3Signal.connect(self.calc_dft_d3)
        self.canvas.tool_bar=self.tool_bar


    def __find_non_physical_structures(self):
        """
        对每个结构进行非物理距离判断
        """
        structure_list = self.canvas.nep_result_data.structure.now_data
        group_array = self.canvas.nep_result_data.structure.group_array.now_data
        radius_coefficient_config = Config.getfloat("widget","radius_coefficient",0.7)
        unreasonable_index=[]
        for structure,index in zip(structure_list,group_array):

            if not structure.adjust_reasonable(radius_coefficient_config):

                unreasonable_index.append(index)
            yield 1
        self.canvas.select_index(unreasonable_index,False)


    def find_non_physical_structures(self):
        """
        非物理结构函数入口
        """
        if self.canvas.nep_result_data is None:
            return
        progress_diag = QProgressDialog(f"" ,"Cancel",0,self.canvas.nep_result_data.structure.num,self._parent)
        thread=utils.LoadingThread(self._parent,show_tip=False )
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Finding non-physical structures")
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)  # 用户取消时终止线程
        thread.start_work(self.__find_non_physical_structures)
        progress_diag.exec()

    def find_max_error_point(self):
        dataset = self.canvas.get_axes_dataset(self.canvas.current_axes)

        if dataset is None:
            return

        box= GetIntMessageBox(self._parent,"Please enter an integer N, it will find the top N structures with the largest errors")
        n = Config.getint("widget","max_error_value",10)
        box.intSpinBox.setValue(n)

        if not box.exec():
            return
        nmax= box.intSpinBox.value()
        Config.set("widget","max_error_value",nmax)
        index= (dataset.get_max_error_index(nmax))

        self.canvas.select_index(index,False)

    def sparse_point(self):
        if  self.canvas.nep_result_data is None:
            return
        box= SparseMessageBox(self._parent,"Please specify the maximum number of structures and minimum distance")
        n_samples = Config.getint("widget","sparse_num_value",10)
        distance = Config.getfloat("widget","sparse_distance_value",0.01)

        box.intSpinBox.setValue(n_samples)
        box.doubleSpinBox.setValue(distance)

        if not box.exec():
            return
        n_samples= box.intSpinBox.value()
        distance= box.doubleSpinBox.value()

        Config.set("widget","sparse_num_value",n_samples)
        Config.set("widget","sparse_distance_value",distance)

        dataset = self.canvas.nep_result_data.descriptor
        if dataset.now_data.size ==0:
            MessageManager.send_message_box("No descriptor data available","Error")
            return
        remaining_indices = farthest_point_sampling(dataset.now_data,n_samples=n_samples,min_dist=distance)

        # 获取所有索引（从 0 到 len(arr)-1）
        # all_indices = np.arange(dataset.now_data.shape[0])

        # 使用 setdiff1d 获取不在 indices_to_remove 中的索引
        # remove_indices = np.setdiff1d(all_indices, remaining_indices)
        structures = dataset.group_array[remaining_indices]
        self.canvas.select_index(structures.tolist(),False)

    def export_descriptor_data(self):
        if self.canvas.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        path = utils.call_path_dialog(self, "Choose a file save ", "file",default_filename="export_descriptor_data.out")
        if path:
            thread = utils.LoadingThread(self, show_tip=True, title="Exporting descriptor data")
            thread.start_work(self._export_descriptor_data, path)

    def _export_descriptor_data(self,path):

        if len(self.canvas.nep_result_data.select_index) == 0:
            MessageManager.send_info_message("No data selected!")
            return
        select_index=self.canvas.nep_result_data.descriptor.convert_index(list(self.canvas.nep_result_data.select_index))
        descriptor_data = self.canvas.nep_result_data.descriptor.now_data[select_index,:]
        if hasattr(self.canvas.nep_result_data,"energy") and self.canvas.nep_result_data.energy.num !=0:
            select_index = self.canvas.nep_result_data.energy.convert_index(
                list(self.canvas.nep_result_data.select_index))

            energy_data = self.canvas.nep_result_data.energy.now_data[select_index,1]
            descriptor_data = np.column_stack((descriptor_data,energy_data))

        with open(path, "w") as f:
            np.savetxt(f,descriptor_data,fmt='%.6g',delimiter='\t')


    def shift_energy_baseline(self):
        data = self.canvas.nep_result_data
        if data is None:
            return
        ref_index = list(data.select_index)
        # if len(ref_index) == 0:
        #     MessageManager.send_info_message("No data selected!")
        #     return

        max_generations = Config.getint("widget","max_generation_value",100000)
        population_size =  Config.getint("widget","population_size",40)
        convergence_tol = Config.getfloat("widget","convergence_tol", 1e-8)
        config_set = set(data.structure.get_all_config())
        suggested = suggest_group_patterns(list(config_set))
        box = ShiftEnergyMessageBox(
            self._parent,
            "Specify regex groups for Config_type (comma separated)"
        )
        box.groupEdit.setText(";".join(suggested))
        box.genSpinBox.setValue(max_generations)
        box.sizeSpinBox.setValue(population_size)
        box.tolSpinBox.setValue(convergence_tol)


        if not box.exec():
            return

        pattern_text = box.groupEdit.text().strip()
        group_patterns = [p.strip() for p in pattern_text.split(';') if p.strip()]

        alignment_mode = box.modeCombo.currentText()


        max_generations = box.genSpinBox.value()
        population_size = box.sizeSpinBox.value()
        convergence_tol = box.tolSpinBox.value()
        Config.set("widget","max_generation_value",max_generations)
        Config.set("widget","population_size",population_size)
        Config.set("widget","convergence_tol",convergence_tol)
        config_set = set(data.structure.get_all_config())
        progress_diag = QProgressDialog(f"", "Cancel", 0, len(config_set), self._parent)
        thread = utils.LoadingThread(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Shift energies")
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)  # 用户取消时终止线程
        thread.start_work(
            shift_dataset_energy,
            structures=data.structure.now_data,
            reference_structures=data.structure.all_data[ref_index],
            max_generations=max_generations,
            population_size=population_size,
            convergence_tol=convergence_tol,
            group_patterns=group_patterns,
            alignment_mode=alignment_mode,
            nep_energy_array=data.energy.y,
        )
        progress_diag.exec()
        if hasattr(data, "energy") and data.energy.num != 0:
            for i, s in enumerate(data.structure.all_data):
                # print(s.per_atom_energy)
                data.energy.data._data[i, data.energy.x_cols] = s.per_atom_energy
        self.canvas.plot_nep_result()
    def _calc_dft_d3(self,mode,functional,cutoff,cutoff_cn):
        nep_result_data = self.canvas.nep_result_data
        nep_txt_path = nep_result_data.nep_txt_path
        if mode == 0:
            calculate_type = "calculate"
            func_kwargs = {}
        elif mode == 2:
            calculate_type = "calculate_with_dftd3"
            func_kwargs = {
                "functional":functional,
                "cutoff": cutoff,
                "cutoff_cn": cutoff_cn,

            }
        else:
            calculate_type = "calculate_dftd3"
            func_kwargs = {
                "functional":functional,
                "cutoff": cutoff,
                "cutoff_cn": cutoff_cn,

            }


        nep_calc_thread = NEPProcess()
        nep_calc_thread.run_nep3_calculator_process(nep_txt_path.as_posix(),
                                                    nep_result_data.structure.now_data,
                                                    calculate_type, func_kwargs=func_kwargs, wait=True)
        nep_potentials_array, nep_forces_array, nep_virials_array = nep_calc_thread.func_result
        split_indices = np.cumsum(nep_result_data.atoms_num_list)[:-1]
        nep_forces_array = np.split(nep_forces_array, split_indices)
        nep_virials_array=nep_virials_array*nep_result_data.atoms_num_list[:, np.newaxis]

        if mode < 3:
            for index, structure in enumerate(nep_result_data.structure.now_data):
                structure.energy = nep_potentials_array[index]
                structure.forces = nep_forces_array[index]
                structure.virial = nep_virials_array[index]
        else:
            factor = 1 if mode == 3 else -1
            for index, structure in enumerate(nep_result_data.structure.now_data):
                structure.energy += nep_potentials_array[index] * factor
                structure.forces += nep_forces_array[index] * factor
                structure.virial += nep_virials_array[index] * factor

        now_indices = nep_result_data.structure.now_indices


        if hasattr(nep_result_data, "energy") and  nep_result_data.energy.num != 0:
            # print(s.per_atom_energy)
            ref_energies = np.array([s.per_atom_energy for s in nep_result_data.structure.now_data], dtype=np.float32).reshape(-1, 1)

            nep_result_data.energy.data._data[now_indices,  nep_result_data.energy.x_cols] = ref_energies
        if hasattr(nep_result_data, "force") and nep_result_data.force.num != 0:
            force_index=nep_result_data.force.convert_index(now_indices)
            ref_forces = np.vstack([s.forces for s in nep_result_data.structure.now_data], dtype=np.float32)

            nep_result_data.force.data._data[force_index,  nep_result_data.force.x_cols] = ref_forces
        if hasattr(nep_result_data, "virial") and nep_result_data.virial.num != 0:
            ref_virials = np.vstack([s.nep_virial for s in nep_result_data.structure.now_data], dtype=np.float32)
            # print(nep_result_data.structure.now_data[0].virial)
            # print(nep_result_data.structure.now_data[0].nep_virial)
            #
            # print(ref_virials[0])
            nep_result_data.virial.data._data[now_indices, nep_result_data.virial.x_cols] = ref_virials

            # print(nep_result_data.virial.data._data.tolist())
            if hasattr(nep_result_data, "stress") and nep_result_data.stress.num != 0:
                coefficient = (nep_result_data.atoms_num_list / np.array(
                    [s.volume for s in nep_result_data.structure.now_data]))[:, np.newaxis]
                stress_array = ref_virials * coefficient * 160.21766208  # 单位转换\
                stress_array = stress_array.astype(np.float32)

                nep_result_data.stress.data._data[now_indices, nep_result_data.stress.x_cols] = stress_array



    def calc_dft_d3(self):


        if  self.canvas.nep_result_data is None:
            return

        function = Config.getint("widget","functional","scan")
        cutoff = Config.getfloat("widget","cutoff",12)
        cutoff_cn = Config.getfloat("widget","cutoff_cn",6)
        mode = Config.getint("widget","d3_mode",0)

        box = DFTD3MessageBox(
            self._parent,
            "DFT D3"
        )
        box.functionEdit.setText(function)
        box.d1SpinBox.setValue(cutoff)
        box.d1cnSpinBox.setValue(cutoff_cn)
        box.modeCombo.setCurrentIndex(mode)
        if not box.exec():
            return

        mode = box.modeCombo.currentIndex()
        D3_cutoff = box.d1SpinBox.value()
        D3_cutoff_cn = box.d1cnSpinBox.value()
        functional=box.functionEdit.text().strip()
        Config.set("widget","cutoff",D3_cutoff)
        Config.set("widget","cutoff_cn",D3_cutoff_cn)
        Config.set("widget","functional",functional)
        Config.set("widget","d3_mode",mode)

        thread = utils.LoadingThread(self._parent, show_tip=True,title="calculating dftd3")

        thread.start_work(
            self._calc_dft_d3,
            mode,functional,cutoff,cutoff_cn
        )
        thread.finished.connect(self.canvas.plot_nep_result)

        # self.canvas.plot_nep_result()






    def inverse_select(self):
        self.canvas.inverse_select()

    def select_by_index(self):
        if self.canvas.nep_result_data is None:
            return
        box = IndexSelectMessageBox(self._parent, "Select structures by index")
        if not box.exec():
            return
        text = box.indexEdit.text().strip()
        use_origin = box.checkBox.isChecked()
        data = self.canvas.nep_result_data.structure
        total = data.all_data.shape[0] if use_origin else data.now_data.shape[0]
        indices = utils.parse_index_string(text, total)
        if not indices:
            return
        if not use_origin:
            indices = data.group_array.now_data[indices].tolist()
        self.canvas.select_index(indices, False)

    def select_by_range(self):
        if self.canvas.nep_result_data is None:
            return
        dataset = self.canvas.get_axes_dataset(self.canvas.current_axes)
        if dataset is None or dataset.now_data.size == 0:
            return
        box = RangeSelectMessageBox(self._parent, "Select structures by range")
        box.xMinSpin.setValue(float(np.min(dataset.x)))
        box.xMaxSpin.setValue(float(np.max(dataset.x)))
        box.yMinSpin.setValue(float(np.min(dataset.y)))
        box.yMaxSpin.setValue(float(np.max(dataset.y)))
        if not box.exec():
            return
        x_min, x_max = sorted([box.xMinSpin.value(), box.xMaxSpin.value()])
        y_min, y_max = sorted([box.yMinSpin.value(), box.yMaxSpin.value()])
        mask_x = (dataset.x >= x_min) & (dataset.x <= x_max)
        mask_y = (dataset.y >= y_min) & (dataset.y <= y_max)
        mask = mask_x & mask_y if box.logicCombo.currentText() == "AND" else mask_x | mask_y
        indices = np.unique(dataset.structure_index[mask]).tolist()
        self.canvas.select_index(indices, False)

    def set_dataset(self,dataset):

        if self.last_figure_num !=len(dataset.dataset):

            self.canvas.init_axes(len(dataset.dataset))
            self.last_figure_num = len(dataset.dataset)

        self.canvas.set_nep_result_data(dataset)
        self.canvas.plot_nep_result()















