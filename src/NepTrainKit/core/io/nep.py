#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 13:26
# @Author  : 兵
# @email    : 1747193328@qq.com

import os
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger
from NepTrainKit import module_path
from NepTrainKit.core import MessageManager, Structure, Config
from NepTrainKit.core.calculator import NEPProcess



from NepTrainKit.core.io.base import NepPlotData, StructureData


from NepTrainKit.core.io.utils import read_nep_out_file, check_fullbatch, read_nep_in, parse_array_by_atomnum

def pca(X, k):
    # 1. 标准化数据（去均值和方差标准化）

    mean = np.mean(X, axis=0)
    X_centered = X - mean


    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)

    # 3. 特征值分解协方差矩阵
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 对特征值进行排序，选择前k个特征值和对应的特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]  # 从大到小排序
    top_k_eigenvectors = eigenvectors[:, sorted_indices[:k]]

    # 5. 投影到前k个主成分
    X_pca = X_centered.dot(top_k_eigenvectors)

    return X_pca.astype(np.float32)


class ResultData(QObject):
    #通知界面更新训练集的数量情况
    updateInfoSignal = Signal( )
    loadFinishedSignal = Signal()


    def __init__(self,nep_txt_path,data_xyz_path,descriptor_path):
        super().__init__()
        self.load_flag=False

        self.descriptor_path=descriptor_path
        self.data_xyz_path=data_xyz_path
        self.nep_txt_path=nep_txt_path

        self.select_index=set()

        self.nep_calc_thread = NEPProcess()


    def load_structures(self):
        structures = Structure.read_multiple(self.data_xyz_path)
        self._atoms_dataset=StructureData(structures)
        self.atoms_num_list = np.array([len(struct) for struct in self.structure.now_data])


    def write_prediction(self):
        if self.atoms_num_list.shape[0] > 1000:
            #
            if not self.data_xyz_path.with_name("nep.in").exists():
                with open(self.data_xyz_path.with_name("nep.in"),
                          "w", encoding="utf8") as f:
                    f.write("prediction 1 ")

    def load(self ):

        self.load_structures()
        self._load_descriptors()
        self._load_dataset()
        self.load_flag=True
        self.loadFinishedSignal.emit()
    def _load_dataset(self):
        raise NotImplementedError()

    @property
    def dataset(self) -> ["NepPlotData"]:
        return []

    @property
    def descriptor(self):
        return self._descriptor_dataset

    @property
    def num(self):
        return self._atoms_dataset.num
    @property
    def structure(self):
        return self._atoms_dataset

    def is_select(self,i):
        return i in self.select_index

    def select(self,_list):
        """
        传入一个索引列表，将索引对应的结构标记为选中状态
        这个下标是结构在train.xyz中的索引
        """

        if isinstance(_list,(int,np.int_,np.int64, np.int32,np.uint32,np.uint64)):
            _list=[_list]

        for i in _list:
            self.select_index.add(i)
        self.updateInfoSignal.emit()

    def uncheck(self,_list):
        """
        check_list 传入一个索引列表，将索引对应的结构标记为未选中状态
        这个下标是结构在train.xyz中的索引
        """
        if isinstance(_list,int):
            _list=[_list]
        for i in _list:
            if i in self.select_index:
                self.select_index.remove(i)

        self.updateInfoSignal.emit()
    def export_selected_xyz(self,save_file_path):
        """
        导出当前选中的结构
        """
        index=list(self.select_index)
        try:

            with open(save_file_path,"w",encoding="utf8") as f:
                for structure in self._atoms_dataset.now_data[index]:
                    structure.write(f)

            MessageManager.send_info_message(f"File exported to: {save_file_path}")
        except:
            MessageManager.send_info_message(f"An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())

    def export_model_xyz(self,save_path):
        """
        导出当前结构
        :param save_path: 保存路径
        被删除的导出到export_remove_model.xyz
        被保留的导出到export_good_model.xyz
        """
        try:

            with open(Path(save_path).joinpath("export_good_model.xyz"),"w",encoding="utf8") as f:
                for structure in self._atoms_dataset.now_data:
                    structure.write(f)

            with open(Path(save_path).joinpath("export_remove_model.xyz"),"w",encoding="utf8") as f:
                for structure in self._atoms_dataset.remove_data:
                    structure.write(f)


            MessageManager.send_info_message(f"File exported to: {save_path}")
        except:
            MessageManager.send_info_message(f"An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())


    def get_atoms(self,index ):
        """根据原始索引获取原子结构对象"""
        index=self._atoms_dataset.convert_index(index)
        return self._atoms_dataset.now_data[index][0]



    def remove(self,i):

        """
        在所有的dataset中删除某个索引对应的结构
        """
        self._atoms_dataset.remove(i)
        for dataset in self.dataset:
            dataset.remove(i)
        self.updateInfoSignal.emit()

    @property
    def is_revoke(self):
        """
        判断是否有被删除的结构
        """
        return self._atoms_dataset.remove_data.size!=0
    def revoke(self):
        """
        撤销到上一次的删除
        """
        self._atoms_dataset.revoke()
        for dataset in self.dataset:
            dataset.revoke( )

    def delete_selected(self ):
        """
        删除所有selected的结构
        """
        self.remove(list(self.select_index))
        self.select_index.clear()
        self.updateInfoSignal.emit()


    def _load_descriptors(self):


        if os.path.exists(self.descriptor_path):
            desc_array = read_nep_out_file(self.descriptor_path,dtype=np.float32)

        else:
            desc_array = np.array([])

        if desc_array.size == 0:
            self.nep_calc_thread.run_nep3_calculator_process(self.nep_txt_path.as_posix(),
                self.structure.now_data,
                "descriptor" ,wait=True)
            desc_array=self.nep_calc_thread.func_result
            # desc_array = run_nep3_calculator_process(
            #     )

            if desc_array.size != 0:
                np.savetxt(self.descriptor_path, desc_array, fmt='%.6g')
        else:
            if desc_array.shape[0] != len(self.atoms_num_list):
                # 原子描述符 需要计算结构描述符
                desc_array = parse_array_by_atomnum(desc_array, self.atoms_num_list, map_func=np.mean, axis=0)

                pass
            else:
                # 结构描述符
                pass

        if desc_array.size != 0:
            if desc_array.shape[1] > 2:
                desc_array = pca(desc_array, 2)

        self._descriptor_dataset = NepPlotData(desc_array, title="descriptor")


class NepTrainResultData(ResultData):
    def __init__(self,
                 nep_txt_path,
                 data_xyz_path,
                 energy_out_path,
                 force_out_path,
                 stress_out_path,
                 virial_out_path,
                 descriptor_path

                 ):
        super().__init__(nep_txt_path,data_xyz_path,descriptor_path)
        self.energy_out_path = energy_out_path
        self.force_out_path = force_out_path
        self.stress_out_path = stress_out_path
        self.virial_out_path = virial_out_path

    @property
    def dataset(self):
        # return [self.energy, self.stress,self.virial, self.descriptor]
        return [self.energy,self.force,self.stress,self.virial, self.descriptor]

    @property
    def energy(self):
        return self._energy_dataset

    @property
    def force(self):
        return self._force_dataset

    @property
    def stress(self):
        return self._stress_dataset

    @property
    def virial(self):
        return self._virial_dataset

    @classmethod
    def from_path(cls, path ):
        dataset_path = Path(path)

        file_name=dataset_path.stem

        nep_txt_path = dataset_path.with_name(f"nep.txt")
        if not nep_txt_path.exists():
            nep89_path = os.path.join(module_path, "Config/nep89.txt")
            nep_txt_path=Path(nep89_path)
        energy_out_path = dataset_path.with_name(f"energy_{file_name}.out")
        force_out_path = dataset_path.with_name(f"force_{file_name}.out")
        stress_out_path = dataset_path.with_name(f"stress_{file_name}.out")
        virial_out_path = dataset_path.with_name(f"virial_{file_name}.out")
        if file_name=="train":

            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")
        return cls(nep_txt_path,dataset_path,energy_out_path,force_out_path,stress_out_path,virial_out_path,descriptor_path)

    def _load_dataset(self) -> None:
        """加载或计算 NEP 数据集，并更新内部数据集属性。"""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            energy_array, force_array, virial_array, stress_array = self._recalculate_and_save( )
        else:
            energy_array = read_nep_out_file(self.energy_out_path, dtype=np.float32)
            force_array = read_nep_out_file(self.force_out_path, dtype=np.float32)
            virial_array = read_nep_out_file(self.virial_out_path, dtype=np.float32)
            stress_array = read_nep_out_file(self.stress_out_path, dtype=np.float32)

        self._energy_dataset = NepPlotData(energy_array, title="energy")
        default_forces = Config.get("widget", "forces_data", "Row")
        if force_array.size != 0 and default_forces == "Norm":

            force_array = parse_array_by_atomnum(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)

            self._force_dataset = NepPlotData(force_array, title="force")
        else:
            self._force_dataset = NepPlotData(force_array, group_list=self.atoms_num_list, title="force")

        if float(nep_in.get("lambda_v", 1)) != 0:
            self._stress_dataset = NepPlotData(stress_array, title="stress")

            self._virial_dataset = NepPlotData(virial_array, title="virial")
        else:
            self._stress_dataset = NepPlotData([], title="stress")

            self._virial_dataset = NepPlotData([], title="virial")
    def _should_recalculate(self, nep_in: dict) -> bool:
        """判断是否需要重新计算 NEP 数据。"""
        output_files_exist = all([
            self.energy_out_path.exists(),
            self.force_out_path.exists(),
            self.stress_out_path.exists(),
            self.virial_out_path.exists()
        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist

    def _save_energy_data(self, potentials: np.ndarray)  :

        """保存能量数据到文件。"""

        try:
            ref_energies = np.array([s.per_atom_energy for s in self.structure.now_data], dtype=np.float32)

            if potentials.size  == 0:
                #计算失败 空数组
                energy_array = np.column_stack([ref_energies, ref_energies])
            else:
                energy_array = np.column_stack([potentials / self.atoms_num_list, ref_energies])
        except Exception:
            logger.debug(traceback.format_exc())
            if potentials.size == 0:
                # 计算失败 空数组
                energy_array = np.column_stack([potentials, potentials])
            else:
                energy_array = np.column_stack([potentials / self.atoms_num_list, potentials / self.atoms_num_list])
        energy_array = energy_array.astype(np.float32)
        if energy_array.size != 0:
            np.savetxt(self.energy_out_path, energy_array, fmt='%10.8f')
        return energy_array

    def _save_force_data(self, forces: np.ndarray)  :
        """保存力数据到文件。"""
        try:
            ref_forces = np.vstack([s.forces for s in self.structure.now_data], dtype=np.float32)

            if forces.size == 0:
                # 计算失败 空数组
                forces_array = np.column_stack([ref_forces, ref_forces])

            else:
                forces_array = np.column_stack([forces, ref_forces])
        except KeyError:
            MessageManager.send_warning_message("use nep3 calculator to calculate forces replace the original forces")
            forces_array = np.column_stack([forces, forces])

        except Exception:
            logger.debug(traceback.format_exc())
            forces_array = np.column_stack([forces, forces])
            MessageManager.send_error_message("an error occurred while calculating forces. Please check the input file.")
        if forces_array.size != 0:
            np.savetxt(self.force_out_path, forces_array, fmt='%10.8f')


        return forces_array



    def _save_virial_and_stress_data(self, virials: np.ndarray )    :
        """保存维里张量和应力数据到文件。"""
        coefficient = (self.atoms_num_list / np.array([s.volume for s in self.structure.now_data]))[:, np.newaxis]
        try:
            ref_virials = np.vstack([s.nep_virial for s in self.structure.now_data], dtype=np.float32)
            if virials.size == 0:
                # 计算失败 空数组
                virials_array = np.column_stack([ref_virials, ref_virials])
            else:
                virials_array = np.column_stack([virials, ref_virials])
        except AttributeError:
            MessageManager.send_warning_message("use nep3 calculator to calculate virial replace the original virial")
            virials_array = np.column_stack([virials, virials])

        except Exception:
            MessageManager.send_error_message(f"An error occurred while calculating virial and stress. Please check the input file.")
            logger.debug(traceback.format_exc())
            virials_array = np.column_stack([virials, virials])

        stress_array = virials_array * coefficient  * 160.21766208  # 单位转换\

        stress_array = stress_array.astype(np.float32)
        if virials_array.size != 0:
            np.savetxt(self.virial_out_path, virials_array, fmt='%10.8f')
        if stress_array.size != 0:
            np.savetxt(self.stress_out_path, stress_array, fmt='%10.8f')


        return virials_array, stress_array

    def _recalculate_and_save(self ):

        try:
            self.nep_calc_thread.run_nep3_calculator_process(self.nep_txt_path.as_posix(),
                self.structure.now_data,
                "calculate" ,wait=True)
            nep_potentials_array, nep_forces_array, nep_virials_array=self.nep_calc_thread.func_result
            # nep_potentials_array, nep_forces_array, nep_virials_array = run_nep3_calculator_process(
            #     self.nep_txt_path.as_posix(),
            #     self.structure.now_data,"calculate")
            if nep_potentials_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the potentials, use the original potentials instead.")


            energy_array = self._save_energy_data(nep_potentials_array)
            force_array = self._save_force_data(nep_forces_array)
            virial_array, stress_array = self._save_virial_and_stress_data(nep_virials_array)


            self.write_prediction()
            return energy_array,force_array,virial_array, stress_array
        except Exception as e:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])









class NepPolarizabilityResultData(ResultData):
    def __init__(self,
                 nep_txt_path,
                 data_xyz_path,
                 polarizability_out_path,

        descriptor_path
                 ):
        super().__init__(nep_txt_path,data_xyz_path,descriptor_path)
        self.polarizability_out_path = polarizability_out_path

    @property
    def dataset(self):

        return [self.polarizability_diagonal,self.polarizability_no_diagonal, self.descriptor]

    @property
    def num(self):
        return self._atoms_dataset.num
    @property
    def structure(self):
        return self._atoms_dataset

    @property
    def polarizability_diagonal(self):
        return self._polarizability_diagonal_dataset
    @property
    def polarizability_no_diagonal(self):
        return self._polarizability_no_diagonal_dataset

    @property
    def descriptor(self):
        return self._descriptor_dataset

    @classmethod
    def from_path(cls, path ):
        dataset_path = Path(path)
        file_name = dataset_path.stem
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        polarizability_out_path = dataset_path.with_name(f"polarizability_{file_name}.out")
        if file_name == "train":
            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")

        return cls(nep_txt_path, dataset_path, polarizability_out_path, descriptor_path)
    def _should_recalculate(self, nep_in: dict) -> bool:
        """判断是否需要重新计算 NEP 数据。"""
        output_files_exist = all([
            self.polarizability_out_path.exists(),

        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist

    def _recalculate_and_save(self ):

        try:
            # nep_polarizability_array = run_nep3_calculator_process(self.nep_txt_path.as_posix(),
            #                                                        self.structure.now_data, "polarizability")
            self.nep_calc_thread.run_nep3_calculator_process(self.nep_txt_path.as_posix(),
                self.structure.now_data,
                "polarizability" ,wait=True)
            nep_polarizability_array=self.nep_calc_thread.func_result
            if nep_polarizability_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the polarizability, use the original polarizability instead.")
            nep_polarizability_array = self._save_polarizability_data(  nep_polarizability_array)
            self.write_prediction()

        except Exception as e:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")

            nep_polarizability_array = np.array([])
        return nep_polarizability_array
    def _save_polarizability_data(self, polarizability: np.ndarray)  :
        """保存polarizability数据到文件。"""
        nep_polarizability_array = polarizability / (self.atoms_num_list[:, np.newaxis])

        try:
            ref_polarizability = np.vstack([s.nep_polarizability for s in self.structure.now_data], dtype=np.float32)
            if polarizability.size == 0:
                # 计算失败 空数组
                polarizability_array = np.column_stack([ref_polarizability, ref_polarizability])
            else:

                polarizability_array = np.column_stack([nep_polarizability_array,
                                                        ref_polarizability

                                                        ])

        except Exception:
            logger.debug(traceback.format_exc())
            polarizability_array = np.column_stack([polarizability, polarizability])
        polarizability_array = polarizability_array.astype(np.float32)
        if polarizability_array.size != 0:
            np.savetxt(self.polarizability_out_path, polarizability_array, fmt='%10.8f')

        return polarizability_array

    def _load_dataset(self) -> None:
        """加载或计算 NEP 数据集，并更新内部数据集属性。"""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            polarizability_array = self._recalculate_and_save( )
        else:
            polarizability_array= read_nep_out_file(self.polarizability_out_path, dtype=np.float32)
        self._polarizability_diagonal_dataset = NepPlotData(polarizability_array[:, [0,1,2,6,7,8]], title="Polar Diag")

        self._polarizability_no_diagonal_dataset = NepPlotData(polarizability_array[:, [3,4,5,9,10,11]], title="Polar NoDiag")


class NepDipoleResultData(ResultData):
    def __init__(self,
                 nep_txt_path,
                 data_xyz_path,
                 dipole_out_path,

                 descriptor_path
                 ):
        super().__init__(nep_txt_path, data_xyz_path, descriptor_path)

        self.dipole_out_path = dipole_out_path
    @property
    def dataset(self):
        return [self.dipole , self.descriptor]

    @property
    def num(self):
        return self._atoms_dataset.num

    @property
    def structure(self):
        return self._atoms_dataset

    @property
    def dipole(self):
        return self._dipole_dataset



    @property
    def descriptor(self):
        return self._descriptor_dataset

    @classmethod
    def from_path(cls, path, model="train"):
        dataset_path = Path(path)

        file_name = dataset_path.stem

        nep_txt_path = dataset_path.with_name(f"nep.txt")

        polarizability_out_path = dataset_path.with_name(f"dipole_{file_name}.out")

        if file_name == "train":

            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")

        return cls(nep_txt_path, dataset_path, polarizability_out_path, descriptor_path)


    def _should_recalculate(self, nep_in: dict) -> bool:
        """判断是否需要重新计算 NEP 数据。"""


        output_files_exist = all([
            self.dipole_out_path.exists(),

        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist

    def _recalculate_and_save(self ):

        try:
            # nep_dipole_array = run_nep3_calculator_process(self.nep_txt_path.as_posix(),
            #                                                self.structure.now_data, "dipole")
            self.nep_calc_thread.run_nep3_calculator_process(self.nep_txt_path.as_posix(),
                self.structure.now_data,
                "dipole" ,wait=True)
            nep_dipole_array=self.nep_calc_thread.func_result

            if nep_dipole_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the dipole, use the original dipole instead.")
            nep_dipole_array = self._save_dipole_data(  nep_dipole_array)
            self.write_prediction()

        except Exception as e:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")

            nep_dipole_array = np.array([])
        return nep_dipole_array
    def _save_dipole_data(self, dipole: np.ndarray)  :
        """保存dipole数据到文件。"""
        nep_dipole_array = dipole / (self.atoms_num_list[:, np.newaxis])

        try:
            ref_dipole = np.vstack([s.nep_dipole for s in self.structure.now_data], dtype=np.float32)
            if dipole.size == 0:
                # 计算失败 空数组
                dipole_array = np.column_stack([ref_dipole, ref_dipole])
            else:
                dipole_array = np.column_stack([nep_dipole_array,
                                            ref_dipole

                                                    ])

        except Exception:
            logger.debug(traceback.format_exc())
            dipole_array = np.column_stack([nep_dipole_array, nep_dipole_array])
        dipole_array = dipole_array.astype(np.float32)
        if dipole_array.size != 0:
            np.savetxt(self.dipole_out_path, dipole_array, fmt='%10.8f')

        return dipole_array

    def _load_dataset(self) -> None:
        """加载或计算 NEP 数据集，并更新内部数据集属性。"""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            dipole_array = self._recalculate_and_save( )
        else:
            dipole_array= read_nep_out_file(self.dipole_out_path, dtype=np.float32)
        self._dipole_dataset = NepPlotData(dipole_array, title="dipole")
