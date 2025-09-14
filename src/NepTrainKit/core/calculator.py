#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 14:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import contextlib
import os
import traceback
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QThread, Signal, QObject
from loguru import logger
from multiprocessing import Process, JoinableQueue, Event

from NepTrainKit import utils
from NepTrainKit.config import Config
from NepTrainKit.core import Structure, MessageManager
from NepTrainKit.core.types import NepBackend

try:
    from NepTrainKit.nep_cpu import CpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_cpu")

    try:
        from nep_cpu import CpuNep
    except ImportError:
        logger.debug("no found nep_cpu")


        CpuNep=None
try:
    from NepTrainKit.nep_gpu import GpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_gpu")
    logger.debug(traceback.format_exc())
    try:
        from nep_gpu import GpuNep
    except ImportError:
        logger.debug(traceback.format_exc())

        logger.debug("no found nep_gpu")
        GpuNep=None
class NepCalculator():

    def __init__(self, model_file="nep.txt",backend:NepBackend=None,batch_size:int=None):
        super().__init__()
        if not isinstance(model_file, str):
            model_file = str(model_file )
        # print(model_file,backend,batch_size)
        self.initialized = False
        if backend is   None:
            self.backend=NepBackend.AUTO
        else:
            self.backend:NepBackend=backend
        if batch_size is None:
            self.batch_size= 1000
        else:

            self.batch_size:int = batch_size
        self.model_file=model_file
        if CpuNep is None and GpuNep is None:
            MessageManager.send_message_box("Failed to import NEP.\n To use the display functionality normally, please prepare the *.out and descriptor.out files.","Error")
            return

        if os.path.exists(model_file):

            self.load_nep()
            # Probe backend viability and fall back to CPU if GPU is not usable

            self.element_list = self.nep3.get_element_list()
            self.type_dict = {e: i for i, e in enumerate(self.element_list)}
            self.initialized=True
        else:
            self.initialized = False

    def cancel(self):
        self.nep3.cancel()

    def load_nep(self):
        if self.backend == NepBackend.AUTO:
            if not self._load_nep_backend(NepBackend.GPU):
                self._load_nep_backend(NepBackend.CPU)
        elif self.backend == NepBackend.GPU:
            if not self._load_nep_backend(NepBackend.GPU):
                MessageManager.send_warning_message("The NEP backend you selected is GPU, but it failed to load on your device; the program has switched to the CPU backend.")
                self._load_nep_backend(NepBackend.CPU)
        else:
            self._load_nep_backend(NepBackend.CPU)
    def _load_nep_backend(self,backend):
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    if backend==NepBackend.GPU:
                        if GpuNep is not None:
                            try:
                                self.nep3 = GpuNep(self.model_file)

                            except RuntimeError as e:
                                logger.error( e)
                                MessageManager.send_warning_message(str(e))
                                return False
                            self.nep3.set_batch_size(self.batch_size)
                        else:
                            return False
                    else:
                        # backend==NepBackend.CPU
                        if CpuNep is not None:

                            self.nep3 = CpuNep(self.model_file)
                        else:
                            return False
                    # track the active backend
                    self.backend = backend
                    return True
        except:
            logger.debug(traceback.format_exc())
            return False


    def compose_structures(self, structures:list[Structure]):
        group_size = []
        _types = []
        _boxs = []
        _positions = []
        if  not isinstance(structures,(list,np.ndarray)):
            structures = [structures]
        for structure in structures:
            symbols = structure.get_chemical_symbols()
            _type = [self.type_dict[k] for k in symbols]
            _box = structure.cell.transpose(1, 0).reshape(-1).tolist()
            _position = structure.positions.transpose(1, 0).reshape(-1).tolist()
            _types.append(_type)
            _boxs.append(_box)
            _positions.append(_position)
            group_size.append(len(_type))
        return  _types, _boxs, _positions,group_size

    @utils.timeit
    def calculate(self,
                  structures:list[Structure]
                  )->tuple[npt.NDArray[np.float32],npt.NDArray[np.float32],npt.NDArray[np.float32]]:
        if not self.initialized:
            return np.array([]),np.array([]),np.array([])
        _types, _boxs, _positions,group_size = self.compose_structures(structures)
        self.nep3.reset_cancel()
        try:
            potentials, forces, virials = self.nep3.calculate(_types, _boxs, _positions)
        except Exception:
            logger.debug(traceback.format_exc())
            # If GPU runtime fails, switch to CPU and retry once
            if GpuNep is not None and isinstance(self.nep3, GpuNep):
                MessageManager.send_warning_message("GPU calculation failed; switching to CPU backend.")
                if self._load_nep_backend(NepBackend.CPU):
                    try:
                        potentials, forces, virials = self.nep3.calculate(_types, _boxs, _positions)
                    except Exception:
                        logger.debug(traceback.format_exc())
                        return np.array([]),np.array([]),np.array([])
                else:
                    return np.array([]),np.array([]),np.array([])
            else:
                return np.array([]),np.array([]),np.array([])
        split_indices = np.cumsum(group_size)[:-1]
        #
        potentials=np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices)
        potentials_array = np.array(list(map(np.sum, split_potential_arrays)), dtype=np.float32)
        # print(potentials_array)
        # 处理每个force数组：reshape (3, -1) 和 transpose(1, 0)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]

        forces_array = np.vstack(reshaped_forces,dtype=np.float32)
        # print(forces_array)

        reshaped_virials = np.vstack([np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials],dtype=np.float32)

        # virials_array = reshaped_virials[:,[0,4,8,1,5,6]]

        return potentials_array,forces_array,reshaped_virials


    @utils.timeit
    def calculate_dftd3(self,
                        structures:list[Structure],
                        functional,
                        cutoff,
                        cutoff_cn)->tuple[npt.NDArray[np.float32],npt.NDArray[np.float32],npt.NDArray[np.float32]]:
        if not self.initialized:
            return np.array([]),np.array([]),np.array([])
        _types, _boxs, _positions,group_size = self.compose_structures(structures)
        self.nep3.reset_cancel()

        try:
            potentials, forces, virials = self.nep3.calculate_dftd3(functional,cutoff,cutoff_cn,_types, _boxs, _positions)
        except Exception:
            logger.debug(traceback.format_exc())
            if GpuNep is not None and isinstance(self.nep3, GpuNep):
                MessageManager.send_warning_message("GPU DFT-D3 failed; switching to CPU backend.")
                if self._load_nep_backend(NepBackend.CPU):
                    try:
                        potentials, forces, virials = self.nep3.calculate_dftd3(functional,cutoff,cutoff_cn,_types, _boxs, _positions)
                    except Exception:
                        logger.debug(traceback.format_exc())
                        return np.array([]),np.array([]),np.array([])
                else:
                    return np.array([]),np.array([]),np.array([])
            else:
                return np.array([]),np.array([]),np.array([])
        split_indices = np.cumsum(group_size)[:-1]
        #
        potentials=np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices)
        potentials_array = np.array(list(map(np.sum, split_potential_arrays)), dtype=np.float32)
        # print(potentials_array)
        # 处理每个force数组：reshape (3, -1) 和 transpose(1, 0)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]

        forces_array = np.vstack(reshaped_forces,dtype=np.float32)
        # print(forces_array)

        reshaped_virials = np.vstack([np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials],dtype=np.float32)

        # virials_array = reshaped_virials[:,[0,4,8,1,5,6]]

        return potentials_array,forces_array,reshaped_virials

    @utils.timeit
    def calculate_with_dftd3(self,
                             structures:list[Structure],
                             functional,
                             cutoff,
                             cutoff_cn)->tuple[npt.NDArray[np.float32],npt.NDArray[np.float32],npt.NDArray[np.float32]]:
        if not self.initialized:
            return np.array([]),np.array([]),np.array([])
        _types, _boxs, _positions,group_size = self.compose_structures(structures)
        self.nep3.reset_cancel()

        try:
            potentials, forces, virials = self.nep3.calculate_with_dftd3( functional,cutoff,cutoff_cn,_types, _boxs, _positions)
        except Exception:
            logger.debug(traceback.format_exc())
            if GpuNep is not None and isinstance(self.nep3, GpuNep):
                MessageManager.send_warning_message("GPU calculation failed; switching to CPU backend.")
                if self._load_nep_backend(NepBackend.CPU):
                    try:
                        potentials, forces, virials = self.nep3.calculate_with_dftd3( functional,cutoff,cutoff_cn,_types, _boxs, _positions)
                    except Exception:
                        logger.debug(traceback.format_exc())
                        return np.array([]),np.array([]),np.array([])
                else:
                    return np.array([]),np.array([]),np.array([])
            else:
                return np.array([]),np.array([]),np.array([])
        split_indices = np.cumsum(group_size)[:-1]
        #
        potentials=np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices)
        potentials_array = np.array(list(map(np.sum, split_potential_arrays)), dtype=np.float32)
        # print(potentials_array)
        # 处理每个force数组：reshape (3, -1) 和 transpose(1, 0)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]

        forces_array = np.vstack(reshaped_forces,dtype=np.float32)
        # print(forces_array)

        reshaped_virials = np.vstack([np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials],dtype=np.float32)

        # virials_array = reshaped_virials[:,[0,4,8,1,5,6]]

        return potentials_array,forces_array,reshaped_virials


    def get_descriptor(self,structure:Structure)->npt.NDArray[np.float32]:
        """
        获取单个结构的所有原子描述符
        """
        if not self.initialized:
            return np.array([])
        symbols = structure.get_chemical_symbols()
        _type = [self.type_dict[k] for k in symbols]
        _box = structure.cell.transpose(1, 0).reshape(-1).tolist()

        _position = structure.positions.transpose(1, 0).reshape(-1).tolist()
        self.nep3.reset_cancel()

        descriptor = self.nep3.get_descriptor(_type, _box, _position)

        descriptors_per_atom = np.array(descriptor,dtype=np.float32).reshape(-1, len(structure)).T

        return descriptors_per_atom
    @utils.timeit
    def get_structures_descriptor(self,
                                  structures:list[Structure]
                                  )->npt.NDArray[np.float32]:
        """
        获取结构描述符：原子平均
        返回的已经结构的描述符了 无需平均
        """
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)
        self.nep3.reset_cancel()

        descriptor = self.nep3.get_structures_descriptor(_types, _boxs, _positions)

        return np.array(descriptor,dtype=np.float32)

    @utils.timeit
    def get_structures_polarizability(self,
                                      structures:list[Structure]
                                      )->npt.NDArray[np.float32]:
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)
        self.nep3.reset_cancel()

        polarizability = self.nep3.get_structures_polarizability(_types, _boxs, _positions)

        return np.array(polarizability,dtype=np.float32)

    def get_structures_dipole(self,
                              structures:list[Structure]
                              )->npt.NDArray[np.float32]:
        if not self.initialized:
            return np.array([])
        self.nep3.reset_cancel()

        _types, _boxs, _positions, group_size = self.compose_structures(structures)

        dipole = self.nep3.get_structures_dipole(_types, _boxs, _positions)

        return np.array(dipole,dtype=np.float32)

Nep3Calculator = NepCalculator

def run_nep_calculator(nep_txt, structures, calculator_type, func_kwargs={},cls_kwargs={},queue=None):
    try:

        nep3 = NepCalculator(nep_txt,**cls_kwargs)
        if calculator_type == 'polarizability':
            result = nep3.get_structures_polarizability(structures)
        elif calculator_type == 'descriptor':
            result = nep3.get_structures_descriptor(structures)
        elif calculator_type == 'dipole':
            result = nep3.get_structures_dipole(structures)
        elif  calculator_type == 'calculate_with_dftd3':
            result = nep3.calculate_with_dftd3(structures,**func_kwargs)
        elif  calculator_type == 'calculate_dftd3':
            result = nep3.calculate_dftd3(structures,**func_kwargs)
        else:
            result = nep3.calculate(structures)
        if queue:
            queue.put(result)  # 将结果通过管道发送给主进程
    except Exception as e:
        logger.error(traceback.format_exc())
        result = np.array([])
        if queue:
            queue.put(result)
    if   queue is  None:
        return result

class NEPProcess(QObject):
    result_signal = Signal( )  # 信号，用于传递进程结果
    error_signal = Signal(str )  # 信号，用于传递错误信息

    def __init__(self ):
        super().__init__()

        self.process = None
        self.use_process = False
        self.func_result = None
    def run_nep3_calculator_process(self,nep_txt, structures, calculator_type="calculate",func_kwargs={},cls_kwargs={},wait=False):
        self.func=run_nep_calculator
        self.queue = JoinableQueue()

        self.input_kwargs={
            "nep_txt": nep_txt,
            "calculator_type": calculator_type,
            "structures": structures,
            "func_kwargs":func_kwargs,
            "cls_kwargs":cls_kwargs,

        }
        #好像在dll释放下gil就不会卡了  这里先统一不使用进程了
        # if len(structures) < 2000:
        #     self.use_process=False
        #     self.input_kwargs["queue"] =  None
        # else:
        #     self.input_kwargs["queue"] = self.queue
        #     self.use_process=True
        # self.use_process=False
        self.input_kwargs["queue"] =  None
        self.input_args=()
        self.func_result:tuple
        self.run()

    def run(self):
        try:
            if self.use_process:
                # 创建并启动进程
                self.process = Process(target=self.func, args= self.input_args ,kwargs = self.input_kwargs,daemon=True)
                self.process.start()
                # 获取结果
                self.func_result = self.queue.get()
            else:
                self.func_result=self.func(*self.input_args,**self.input_kwargs)
            if len(self.func_result) !=0:
                self.result_signal.emit( )
            else:
                self.error_signal.emit("No result returned from process")

        except Exception as e:
            logger.error(traceback.format_exc())
            self.error_signal.emit(f"Error: {str(e)}")
        finally:
            self.queue.close()  # 关闭队列
            # self.queue.join()  # 等待队列任务完成
            if self.process is not None :
                self.process.terminate()  # 确保进程被终止

                self.process.join()
                # self.process.close()

    def stop(self):
        """强制终止进程并清理资源"""
        return
        try:

            if self.process is not None:
                if self.process.is_alive():

                    self.process.terminate()

                    self.process.join()
            self.queue.cancel_join_thread()
        except Exception as e:
            logger.error(traceback.format_exc())

if __name__ == '__main__':
    structures = Structure.read_multiple(r"D:\Desktop\nep\nep-data-main\2023_Zhao_PdCuNiP\train.xyz")
    nep = NepCalculator(r"D:\Desktop\nep\nep-data-main\2023_Zhao_PdCuNiP\nep.txt")
    nep.calculate(structures)
