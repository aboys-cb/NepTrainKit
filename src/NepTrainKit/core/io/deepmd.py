#!/usr/bin/env python
# -*- coding: utf-8 -*-
import traceback
from pathlib import Path
import os
import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger
import glob
from .base import NepPlotData, StructureData, ResultData,DPPlotData
from NepTrainKit.core.structure import Structure, load_npy_structure,save_npy_structure
from .utils import parse_array_by_atomnum,read_nep_out_file
from .. import Config, MessageManager
from ... import module_path
def is_deepmd_path(folder)-> bool:
    if os.path.exists(os.path.join(folder,"type.raw")):
        return True
    if glob.glob(os.path.join(folder, "**/type.raw"), recursive=True):
        return True
    return False
class DeepmdResultData(ResultData):


    def __init__(self, nep_txt_path,data_xyz_path: Path,
                 energy_out_path,
                 force_out_path,

                 virial_out_path,
                 descriptor_path,
                 spin_out_path=None
                 ):
        super().__init__(nep_txt_path, data_xyz_path,descriptor_path)
        self.energy_out_path = energy_out_path
        self.force_out_path = force_out_path
        self.spin_out_path = spin_out_path
        self.virial_out_path = virial_out_path



    @classmethod
    def from_path(cls, path):
        dataset_path = Path(path)

        file_name=dataset_path.name
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        if not nep_txt_path.exists():
            nep89_path = os.path.join(module_path, "Config/nep89.txt")
            nep_txt_path=Path(nep89_path)
        descriptor_path = dataset_path.with_name(f"descriptor.out")
        e_path = list(dataset_path.parent.glob("*.e_peratom.out") )
        if e_path:
            e_path = e_path[0]
            suffix = (e_path.name.replace(".e_peratom.out",""))

        else:
            suffix="detail"



        energy_out_path = dataset_path.with_name(f"{suffix}.e_peratom.out")
        force_out_path = dataset_path.with_name(f"{suffix}.fr.out")
        if  not force_out_path.exists():
            force_out_path = dataset_path.with_name(f"{suffix}.f.out")

        # stress_out_path = dataset_path.with_name(f"{suffix}.v.out")
        virial_out_path = dataset_path.with_name(f"{suffix}.v_peratom.out")
        spin_out_path=  dataset_path.with_name(f"{suffix}.fm.out")
        if not spin_out_path.exists():
            spin_out_path= None
        return cls(nep_txt_path,dataset_path,energy_out_path,force_out_path,virial_out_path,descriptor_path,spin_out_path=spin_out_path)

    def load_structures(self):

        structures = load_npy_structure(self.data_xyz_path)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(s) for s in structures])





    @property
    def dataset(self):
        if self.spin_out_path is None:
            return [self.energy, self.force,  self.virial, self.descriptor]
        else:
            return [self.energy, self.force,self.spin, self.virial, self.descriptor]
    @property
    def energy(self):
        return self._energy_dataset

    @property
    def force(self):
        return self._force_dataset

    @property
    def spin(self):
        return self._spin_dataset

    @property
    def virial(self):
        return self._virial_dataset


    def _load_dataset(self) -> None:

        if self._should_recalculate( ):
            energy_array, force_array, virial_array = self._recalculate_and_save( )
        else:
            energy_array=read_nep_out_file(self.energy_out_path,ndmin=2)
            force_array=read_nep_out_file(self.force_out_path,ndmin=2)
            virial_array=read_nep_out_file(self.virial_out_path,ndmin=2)
            if energy_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.energy_out_path.unlink(True)
                self.force_out_path.unlink(True)
                if self.spin_out_path is not None:
                    self.spin_out_path.unlink(True)

                self.virial_out_path.unlink(True)


                return self._load_dataset()

        self._energy_dataset = DPPlotData(energy_array, title="energy")
        default_forces = Config.get("widget", "forces_data", "Row")
        if force_array.size != 0 and default_forces == "Norm":

            force_array = parse_array_by_atomnum(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)

            self._force_dataset = DPPlotData(force_array, title="force")
        else:
            self._force_dataset = DPPlotData(force_array, group_list=self.atoms_num_list, title="force")

        if self.spin_out_path is not None:
            spin_array=read_nep_out_file(self.spin_out_path,ndmin=2)
            group_list=[s.spin_num for s in self.structure.now_data]
            if (np.sum(group_list))!=0:
                self._spin_dataset = DPPlotData(spin_array,  group_list=group_list, title="spin")
            else:
                self.spin_out_path = None

        self._virial_dataset = DPPlotData(virial_array, title="virial")


    def _should_recalculate(self  ) -> bool:
        """判断是否需要重新计算 NEP 数据。"""
        output_files_exist = any([
            self.energy_out_path.exists(),
            self.force_out_path.exists(),

            self.virial_out_path.exists()
        ])
        return   not output_files_exist

    def _save_energy_data(self, potentials: np.ndarray)  :

        """保存能量数据到文件。"""

        try:
            ref_energies = np.array([s.per_atom_energy for s in self.structure.now_data], dtype=np.float32)

            if potentials.size  == 0:
                #计算失败 空数组
                energy_array = np.column_stack([ref_energies, ref_energies])
            else:
                energy_array = np.column_stack([ref_energies,potentials / self.atoms_num_list  ])
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
                forces_array = np.column_stack([ref_forces,forces ])
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
                virials_array = np.column_stack([ref_virials,virials  ])
        except AttributeError:
            MessageManager.send_warning_message("use nep3 calculator to calculate virial replace the original virial")
            virials_array = np.column_stack([virials, virials])

        except Exception:
            MessageManager.send_error_message(f"An error occurred while calculating virial and stress. Please check the input file.")
            logger.debug(traceback.format_exc())
            virials_array = np.column_stack([virials, virials])




        if virials_array.size != 0:
            np.savetxt(self.virial_out_path, virials_array, fmt='%10.8f')



        return virials_array

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
            virial_array = self._save_virial_and_stress_data(nep_virials_array[:, [0, 4, 8, 1, 5, 6]])


            self.write_prediction()
            return energy_array,force_array,virial_array
        except Exception as e:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            return np.array([]), np.array([]), np.array([])

    def export_model_xyz(self,save_path):
        """
        导出当前结构
        :param save_path: 保存路径
        被删除的导出到export_remove_model
        被保留的导出到export_good_model
        """
        try:
            save_npy_structure(os.path.join(save_path, "export_good_model"),self.structure.now_data)
            save_npy_structure(os.path.join(save_path, "export_remove_model"),self.structure.remove_data)


            MessageManager.send_info_message(f"File exported to: {save_path}")
        except:
            MessageManager.send_info_message(f"An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())