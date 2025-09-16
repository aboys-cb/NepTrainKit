#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 15:31
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import threading
import re
import traceback
from functools import cached_property
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger
from typing import Any, Callable, Optional
from numpy import bool_
import numpy.typing as npt
from NepTrainKit import utils
from NepTrainKit.config import Config
from NepTrainKit.core import Structure, MessageManager
from NepTrainKit.core.calculator import NEPProcess, run_nep_calculator, NepCalculator
from NepTrainKit.core.io.utils import read_nep_out_file, parse_array_by_atomnum,get_rmse
from NepTrainKit.core.types import Brushes, SearchType, NepBackend


def pca(X, n_components=None):
    """
    执行主成分分析 (PCA)，只返回降维后的数据
    """
    n_samples, n_features = X.shape

    # 1. 计算均值并中心化数据
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    #樊老师说不用处理 就不减去均值了
    # 但是我还不确定哪种好 还是保持现状把
    # X_centered = X


    # 3. 计算协方差矩阵
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

    # 4. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 5. 特征值和特征向量按降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 6. 确定要保留的主成分数量
    if n_components is None:
        n_components = n_features
    elif n_components > n_features:
        n_components = n_features

    # 7. 将数据投影到前n_components个主成分上 (降维)
    X_pca = np.dot(X_centered, eigenvectors[:, :n_components])

    return X_pca.astype(np.float32)



class DataBase:
    """
    优化后的 DataBase 类，对列表进行封装，支持根据索引删除结构和回退。
    使用布尔掩码管理活动/删除状态，减少列表操作开销。
    """
    def __init__(self, data_list:list[Any]|npt.NDArray[Any]):
        """Initialize with a NumPy array."""
        self._data = np.asarray(data_list)
        # 布尔掩码：True 表示活跃，False 表示已删除
        self._active_mask = np.ones(len(self._data), dtype=bool)
        # 历史记录栈，存储每次删除的掩码变化
        self._history = []
    @property
    def mask_array(self):
        return self._active_mask

    @property
    def num(self) -> int:
        """返回当前活跃数据的数量"""
        return np.sum(self._active_mask)
    @property
    def all_data(self) -> npt.NDArray[Any]:
        return self._data
    @property
    def now_data(self) -> npt.NDArray[Any]:
        """返回当前活跃数据"""
        return self._data[self._active_mask]

    @property
    def remove_data(self) -> npt.NDArray[Any]:
        """返回所有已删除的数据"""
        return self._data[~self._active_mask]

    @property
    def now_indices(self) -> npt.NDArray[np.int32]:
        """返回当前活跃数据的索引下标"""
        return np.where(self._active_mask)[0]

    @property
    def remove_indices(self) -> npt.NDArray[np.int32]:
        """返回已删除数据的索引下标"""
        return np.where(~self._active_mask)[0]

    def remove(self, indices)->None:
        idx = np.unique(np.asarray(indices, dtype=int) if not isinstance(indices, int) else [indices])
        idx = idx[(idx >= 0) & (idx < len(self._data))]
        if len(idx) == 0:
            return
        self._history.append(idx)  # 存储删除的索引
        self._active_mask[idx] = False

    def revoke(self)->None:
        if self._history:
            last_indices = self._history.pop()
            self._active_mask[last_indices] = True

    def __getitem__(self, item):
        """直接索引活跃数据集"""
        return self.now_data[item]


class NepData:
    """
    structure_data 结构性质数据点
    group_array 结构的组号 标记数据点对应结构在train.xyz中的下标
    title 能量 力 等 用于画图axes的标题

    """
    title:str
    def __init__(self,
                 data_list:list[Any]|npt.NDArray[Any],
                 group_list:int|list[int]=1,
                 **kwargs ):
        if isinstance(data_list,(list )):
            data_list=np.array(data_list)

        self.data = DataBase(data_list)
        if isinstance(group_list,int):
            group = np.arange(data_list.shape[0],dtype=np.uint32)
            self.group_array=DataBase(group)
        else:
            group = np.arange(len(group_list),dtype=np.uint32 )
            self.group_array=DataBase(group.repeat(group_list))

        for key,value in kwargs.items():
            setattr(self,key,value)
    @property
    def num(self)->int:
        return self.data.num
    @cached_property
    def cols(self)->int:
        """
        将列数除以2 前面是nep 后面是dft
        """
        if self.now_data.shape[0]==0:
            #数据为0
            return 0
        index = self.now_data.shape[1] // 2
        return index
    @property
    def now_data(self) -> npt.NDArray[Any]:
        """
        返回当前数据
        """
        return self.data.now_data
    @property
    def now_indices(self) -> npt.NDArray[np.int32]:
        return self.data.now_indices
    @property
    def all_data(self) -> npt.NDArray[Any]:
        return self.data.all_data
    def is_visible(self,index) -> bool:
        if self.data.all_data.size == 0:
            return False
        return self.data.mask_array[index].all()
    @property
    def remove_data(self) -> npt.NDArray[Any]:
        """返回删除的数据"""

        return self.data.remove_data

    def convert_index(self,index_list:list[int]|int) -> npt.NDArray[np.int32]:
        """
        传入结构的原始下标 然后转换成现在已有的
        这个主要是映射force等性质的下标 或者其他一对多的性质
        对于一对一的比如，转换后会将index_list按照原始下标排序
        输入[np.int64(98), np.int64(9), np.int64(42), np.int64(141), np.int64(79), np.int64(56)]
        输出[  9  42  56  79  98 141]
        """
        if isinstance(index_list,(int,np.number)):
            index_list=[index_list]
        return np.where(np.isin(self.group_array.all_data,index_list))[0]



    def remove(self,remove_index:list[int]|int):
        """
        根据index删除
        remove_index 结构的原始下标
        """
        remove_indices=self.convert_index(remove_index)

        self.data.remove(remove_indices)
        self.group_array.remove(remove_indices)

    def revoke(self):
        """将上一次删除的数据恢复"""
        self.data.revoke()
        self.group_array.revoke()

    def get_rmse(self)->float:
        if not self.cols:
            return 0
        return get_rmse(self.now_data[:, 0:self.cols],self.now_data[:, self.cols: ])
        # return np.sqrt(((self.now_data[:, 0:self.cols] - self.now_data[:, self.cols: ]) ** 2).mean( ))

    def get_formart_rmse(self)->str:
        rmse=self.get_rmse()
        if self.title =="energy":
            unit="meV/atom"
            rmse*=1000
        elif self.title =="force":
            unit="meV/Å"
            rmse*=1000
        elif self.title =="virial":
            unit="meV/atom"
            rmse*=1000
        elif self.title =="stress":
            unit="MPa"
            rmse*=1000
        elif "Polar" in self.title:
            unit="(m.a.u./atom)"
            rmse*=1000
        elif "dipole" == self.title:
            unit="(m.a.u./atom)"
            rmse*=1000
        elif "spin" ==self.title:
            unit = "meV/μB"
            rmse*=1000
        else:
            return ""
        return f"{rmse:.2f} {unit}"

    def get_max_error_index(self,nmax)->list[int]:
        """
        返回nmax个最大误差的下标
        这个下标是结构的原始下标
        """
        error = np.sum(np.abs(self.now_data[:, 0:self.cols] - self.now_data[:, self.cols: ]), axis=1)
        rmse_max_ids = np.argsort(-error)
        structure_index =self.group_array.now_data[rmse_max_ids]
        index,indices=np.unique(structure_index,return_index=True)

        return structure_index[np.sort(indices)][:nmax].tolist()




class NepPlotData(NepData):

    def __init__(self,data_list,**kwargs ):
        super().__init__(data_list,**kwargs )
        self.x_cols=slice(self.cols,None)
        self.y_cols=slice(None,self.cols )

    @property
    def x(self) -> npt.NDArray[Any]:
        """
        这里返回的是展平之后的数据 方便直接画图
        """
        if self.cols==0:
            return self.now_data
        return self.now_data[ : ,self.x_cols].ravel()
    @property
    def y(self) -> npt.NDArray[Any]:
        if self.cols==0:
            return self.now_data
        return self.now_data[ : , self.y_cols].ravel()


    @property
    def structure_index(self):
        """
        这里根据列数 将group做一个扩充 传入到画图的data里
        为了将散点将结构下标映射起来
        """
        return self.group_array[ : ].repeat(self.cols)
class DPPlotData(NepData):


    def __init__(self,data_list,**kwargs ):
        super().__init__(data_list,**kwargs )
        self.x_cols=slice(None,self.cols)
        self.y_cols=slice(self.cols,None)

    @property
    def x(self):
        if self.cols==0:
            return self.now_data
        return self.now_data[ : ,self.x_cols].ravel()
    @property
    def y(self):
        if self.cols==0:
            return self.now_data
        return self.now_data[ : , self.y_cols].ravel()



    # def all_x(self):
    #     if self.cols==0:
    #         return self.all_data
    #     return self.all_data[ : ,:self.cols].ravel()
    # @property
    # def all_y(self):
    #     if self.cols==0:
    #         return self.all_data
    #     return self.all_data[ : , self.cols:].ravel()
    @property
    def structure_index(self):
        return self.group_array[ : ].repeat(self.cols)

class StructureData(NepData):

    @utils.timeit
    def get_all_config(self,search_type:SearchType=None)->list[str]:
        """
        获取所有结构的某个属性 用于搜索
        :param search_type:SearchType
        :return:每个结构的属性值
        """
        if search_type is None:
            search_type=SearchType.TAG
        if search_type==SearchType.TAG:
            return [structure.tag for structure in self.now_data]
        elif search_type==SearchType.FORMULA:
            return [structure.formula for structure in self.now_data]
        else:
            MessageManager.send_warning_message(f"no such search_type:{search_type}")
            return []
    def search_config(self,config:str,search_type:SearchType):
        """
        根据传入的config 对结构的属性值进行匹配
        这里使用了正则 可以使用正则支持的语法
        :param config:
        :param search_type:
        :return:符合搜索的结构的下标
        """
        if search_type==SearchType.TAG:

            result_index=[i for i ,structure in enumerate(self.now_data) if re.search(config, structure.tag)]
        elif search_type==SearchType.FORMULA:
            result_index=[i for i ,structure in enumerate(self.now_data) if re.search(config, structure.formula)]
        else:
            MessageManager.send_warning_message(f"no such search_type:{search_type}")
            return []
        return self.group_array[result_index].tolist()


class ResultData(QObject):
    #通知界面更新训练集的数量情况
    updateInfoSignal = Signal( )
    #加载训练集结束后发出的信号
    loadFinishedSignal = Signal()
    atoms_num_list: npt.NDArray
    _atoms_dataset: StructureData

    def __init__(self,
                 nep_txt_path:Path|str,
                 data_xyz_path:Path|str,
                 descriptor_path:Path|str,
                 calculator_factory: Optional[Callable[[str], Any]] = None,
                 import_options: Optional[dict] = None):
        super().__init__()
        self.load_flag=False
        # cooperative cancel for long-running loads
        self.cancel_event = threading.Event()

        self.descriptor_path=Path(descriptor_path)
        self.data_xyz_path=Path(data_xyz_path)
        self.nep_txt_path=Path(nep_txt_path)
        #存储选中结构的真实下标
        self.select_index=set()
        # Optional pre-fetched structures to skip IO in load_structures
        self._prefetched_structures: Optional[list[Structure]] = None
        # Optional importer options forwarded to importers.import_structures
        self._import_options: dict = dict(import_options or {})
        self.calculator_factory=calculator_factory

    def request_cancel(self):
        """Request cooperative cancel during load. Also forward to calculator."""
        self.cancel_event.set()
        try:
            if hasattr(self, "nep_calc") and self.nep_calc is not None:
                self.nep_calc.cancel()
        except Exception:
            pass

    def reset_cancel(self):
        self.cancel_event.clear()
    @utils.timeit
    def load_structures(self):
        """
        加载结构xyz文件
        :return:
        """
        # If structures were provided upfront, use them; otherwise parse from file
        if self._prefetched_structures is not None:
            structures = self._prefetched_structures
        else:
            # Unified path: delegate to importers for all formats, including EXTXYZ.
            # ExtxyzImporter internally uses Structure.iter_read_multiple with cancel support.
            from NepTrainKit.core.io import importers as _imps
            opts = dict(self._import_options)
            opts.setdefault("cancel_event", self.cancel_event)
            structures = _imps.import_structures(self.data_xyz_path.as_posix(), **opts)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(struct) for struct in self.structure.now_data])

    def set_structures(self, structures: list[Structure]):
        """
        Provide pre-parsed structures so load_structures can skip file IO.
        """
        self._prefetched_structures = list(structures)

    def write_prediction(self):
        """
        程序将严格检测batch和结构数的关系，如果不是fullbatch，
        就会触发计算，通过写入一个nep.in文件 可以避免这种情况。
        在nep_cpu计算后，回调用该函数
        :return:
        """
        if self.atoms_num_list.shape[0] > 1000:
            #
            if not self.data_xyz_path.with_name("nep.in").exists():
                with open(self.data_xyz_path.with_name("nep.in"),
                          "w", encoding="utf8") as f:
                    f.write("prediction 1 ")

    def load(self ):
        """
        加载数据集的主函数
        :return:
        """
        try:
            # Calculator injection (default to NEP). Subclasses can pass in a factory for other ML potentials.
            if self.calculator_factory is None:
                self.nep_calc = NepCalculator(
                    model_file=self.nep_txt_path.as_posix(),
                    backend=NepBackend(Config.get("nep", "backend", "auto")),
                    batch_size=Config.getint("nep", "gpu_batch_size", 1000)
                )
            else:
                # Factory is responsible for creating a calculator compatible with this ResultData subclass
                try:
                    self.nep_calc = self.calculator_factory(self.nep_txt_path.as_posix())
                except Exception:
                    logger.debug(traceback.format_exc())
                    MessageManager.send_warning_message("Failed to create custom calculator; falling back to NEP.")
                    self.nep_calc = NepCalculator(
                        model_file=self.nep_txt_path.as_posix(),
                        backend=NepBackend(Config.get("nep", "backend", "auto")),
                        batch_size=Config.getint("nep", "gpu_batch_size", 1000)
                    )


            # If subclass overrides load_structures, defer to it; otherwise do cancel-aware read

            self.load_structures()
            if self._atoms_dataset.num!=0:



                if not self.cancel_event.is_set():
                    self._load_descriptors()
                if not self.cancel_event.is_set():
                    self._load_dataset()
                if not self.cancel_event.is_set():
                    self.load_flag=True
            else:
                MessageManager.send_warning_message("No structures were loaded.")

        except:
            logger.error(traceback.format_exc())

            MessageManager.send_error_message("load dataset error!")

        self.loadFinishedSignal.emit()
    def _load_dataset(self):
        """
        不同类型的势函数通过重写该函数实现不同的加载逻辑
        主要是加载结构性质
        :return:
        """
        raise NotImplementedError()

    @property
    def datasets(self) -> list["NepPlotData"]:
        raise NotImplementedError()

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

    def select(self,indices:list[int]|int):
        """
        传入一个索引列表，将索引对应的结构标记为选中状态
        这个下标是结构在train.xyz中的索引
        :param indices: 索引
        :return:
        """


        # 统一转换为 NumPy 数组
        idx = np.asarray(indices, dtype=int) if not isinstance(indices, int) else np.array([indices])
        # 去重并过滤有效索引（在数据范围内且为活跃数据）
        idx = np.unique(idx)
        idx = idx[(idx >= 0) & (idx < len(self.structure.all_data)) & (self.structure.data.mask_array[idx])]
        # 批量添加到选中集合
        self.select_index.update(idx)

        self.updateInfoSignal.emit()

    def uncheck(self,_list:list[int]|int):
        """
        check_list 传入一个索引列表，将索引对应的结构标记为未选中状态
        这个下标是结构在train.xyz中的索引
        :param _list:
        :return:
        """
        if isinstance(_list,(int,np.number)):
            _list=[_list]
        for i in _list:
            if i in self.select_index:
                self.select_index.remove(i)

        self.updateInfoSignal.emit()

    def inverse_select(self):
        """
        根据现在选择的结构对先有训练集进行一个反选的操作
        :return:
        """
        active_indices = set(self.structure.data.now_indices.tolist())
        selected_indices = set(self.select_index)
        unselect = list(selected_indices)
        select = list(active_indices - selected_indices)
        if unselect:
            self.uncheck(unselect)
        if select:
            self.select(select)
    def get_selected_structures(self)->list[Structure]:
        """
        获取选中结构的list
        :return: list[Structure]
        """
        index=list(self.select_index)
        index = self.structure.convert_index(index)

        return self.structure.all_data[index].tolist()

    def export_selected_xyz(self,save_file_path):
        """
        导出当前选中的结构
        :param save_file_path: 保存文件路径，具体到文件 例如~/select.xyz
        :return:
        """
        index=list(self.select_index)
        try:
            with open(save_file_path,"w",encoding="utf8") as f:

                index=self.structure.convert_index(index)
                for structure in self.structure.all_data[index]:
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
                for structure in self.structure.now_data:
                    structure.write(f)

            with open(Path(save_path).joinpath("export_remove_model.xyz"),"w",encoding="utf8") as f:
                for structure in self.structure.remove_data:
                    structure.write(f)


            MessageManager.send_info_message(f"File exported to: {save_path}")
        except:
            MessageManager.send_info_message(f"An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())


    def get_atoms(self,index ):
        """根据原始索引获取原子结构对象"""
        index=self.structure.convert_index(index)
        return self.structure.all_data[index][0]



    def remove(self,i):

        """
        在所有的dataset中删除某个索引对应的结构
        i是原始下标
        """
        self.structure.remove(i)
        for dataset in self.datasets:
            dataset.remove(i)
        self.updateInfoSignal.emit()

    @property
    def is_revoke(self):
        """
        判断是否有被删除的结构
        """
        return self.structure.remove_data.size!=0
    def revoke(self):
        """
        撤销到上一次的删除
        """
        self.structure.revoke()
        for dataset in self.datasets:
            dataset.revoke( )
        self.updateInfoSignal.emit()

    @utils.timeit
    def delete_selected(self ):
        """
        删除所有selected的结构
        """
        self.remove(list(self.select_index))
        self.select_index.clear()
        self.updateInfoSignal.emit()


    def _load_descriptors(self):
        """
        加载训练集的描述符
        :return:
        """

        if os.path.exists(self.descriptor_path):
            desc_array = read_nep_out_file(self.descriptor_path,dtype=np.float32,ndmin=2)

        else:
            desc_array = np.array([])

        if desc_array.size == 0:

            desc_array = self.nep_calc.get_structures_descriptor(self.structure.now_data.tolist())

            # desc_array=self.nep_calc_thread.func_result
            # desc_array = run_nep3_calculator_process(
            #     )

            if desc_array.size != 0:
                np.savetxt(self.descriptor_path, desc_array, fmt='%.6g')
        else:
            if desc_array.shape[0] == np.sum(self.atoms_num_list):
                # 原子描述符 需要计算结构描述符


                desc_array = parse_array_by_atomnum(desc_array, self.atoms_num_list, map_func=np.mean, axis=0)
            elif desc_array.shape[0] == self.atoms_num_list.shape[0]:
                # 结构描述符
                pass

            else:
                self.descriptor_path.unlink(True)
                return self._load_descriptors()

        if desc_array.size != 0:
            if desc_array.shape[1] > 2:
                try:
                    desc_array = pca(desc_array, 2)
                except:
                    MessageManager.send_error_message("PCA dimensionality reduction fails")
                    desc_array = np.array([])
        self._descriptor_dataset = NepPlotData(desc_array, title="descriptor")
