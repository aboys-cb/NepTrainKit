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
from NepTrainKit import module_path,utils
from NepTrainKit.core import MessageManager, Structure, Config
from NepTrainKit.core.calculator import NEPProcess



from NepTrainKit.core.io.base import NepPlotData, StructureData, ResultData

from NepTrainKit.core.io.utils import read_nep_out_file, check_fullbatch, read_nep_in, parse_array_by_atomnum




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
    def from_path(cls, path ,model_type=0):
        dataset_path = Path(path)

        file_name=dataset_path.stem

        nep_txt_path = dataset_path.with_name(f"nep.txt")
        if not nep_txt_path.exists() or model_type>2:
            nep89_path = os.path.join(module_path, "Config/nep89.txt")
            nep_txt_path=Path(nep89_path)
            MessageManager.send_warning_message(f"NEP_CPU currently does not support model_type={model_type}; the program will use nep89 instead.")
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
            energy_array = read_nep_out_file(self.energy_out_path, dtype=np.float32,ndmin=2)
            force_array = read_nep_out_file(self.force_out_path, dtype=np.float32,ndmin=2)
            virial_array = read_nep_out_file(self.virial_out_path, dtype=np.float32,ndmin=2)
            stress_array = read_nep_out_file(self.stress_out_path, dtype=np.float32,ndmin=2)

            if energy_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.energy_out_path.unlink(True)
                self.force_out_path.unlink(True)
                self.virial_out_path.unlink(True)
                self.stress_out_path.unlink(True)

                return self._load_dataset()


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
        except ValueError:
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
            virial_array, stress_array = self._save_virial_and_stress_data(nep_virials_array[:, [0, 4, 8, 1, 5, 6]])


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
            polarizability_array= read_nep_out_file(self.polarizability_out_path, dtype=np.float32,ndmin=2)
            if polarizability_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.polarizability_out_path.unlink()
                return self._load_dataset()
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
            dipole_array= read_nep_out_file(self.dipole_out_path, dtype=np.float32,ndmin=2)
            if dipole_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.dipole_out_path.unlink()
                return self._load_dataset()
        self._dipole_dataset = NepPlotData(dipole_array, title="dipole")
