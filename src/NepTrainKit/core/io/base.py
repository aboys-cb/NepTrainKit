#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 15:31
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import re
import traceback
from functools import cached_property
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger
from numpy import bool_

from NepTrainKit import utils
from NepTrainKit.core import Structure, MessageManager
from NepTrainKit.core.calculator import NEPProcess
from NepTrainKit.core.io.utils import read_nep_out_file, parse_array_by_atomnum
from NepTrainKit.core.types import Brushes

import numpy as np
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
    def __init__(self, data_list):
        """Initialize with a NumPy array."""
        self._data = np.asarray(data_list)
        # 布尔掩码：True 表示活跃，False 表示已删除
        self._active_mask = np.ones(len(self._data), dtype=bool)
        # 历史记录栈，存储每次删除的掩码变化
        self._history = []

    @property
    def num(self) -> int:
        """返回当前活跃数据的数量"""
        return np.sum(self._active_mask)
    @property
    def all_data(self):
        return self._data
    @property
    def now_data(self):
        """返回当前活跃数据"""
        return self._data[self._active_mask]

    @property
    def remove_data(self):
        """返回所有已删除的数据"""
        return self._data[~self._active_mask]

    @property
    def now_indices(self):
        """返回当前活跃数据的索引下标"""
        return np.where(self._active_mask)[0]

    @property
    def remove_indices(self):
        """返回已删除数据的索引下标"""
        return np.where(~self._active_mask)[0]

    def remove(self, indices):
        idx = np.unique(np.asarray(indices, dtype=int) if not isinstance(indices, int) else [indices])
        idx = idx[(idx >= 0) & (idx < len(self._data))]
        if len(idx) == 0:
            return
        self._history.append(idx)  # 存储删除的索引
        self._active_mask[idx] = False

    def revoke(self):
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
    def __init__(self,data_list,group_list=1, **kwargs ):
        if isinstance(data_list,(list )):
            data_list=np.array(data_list)

        self.data = DataBase(data_list )
        if isinstance(group_list,int):
            group = np.arange(data_list.shape[0],dtype=np.uint32)
            self.group_array=DataBase(group)
        else:
            group = np.arange(len(group_list),dtype=np.uint32 )
            self.group_array=DataBase(group.repeat(group_list))

        for key,value in kwargs.items():
            setattr(self,key,value)
    @property
    def num(self):
        return self.data.num
    @cached_property
    def cols(self):
        """
        将列数除以2 前面是nep 后面是dft
        """
        if self.now_data.shape[0]==0:
            #数据为0
            return 0
        index = self.now_data.shape[1] // 2
        return index
    @property
    def now_data(self):
        """
        返回当前数据
        """
        return self.data.now_data
    @property
    def now_indices(self):
        return self.data.now_indices
    @property
    def all_data(self):
        return self.data.all_data
    def is_visible(self,index) -> bool_:
        if self.data.all_data.size == 0:
            return False
        return self.data._active_mask[index].all()
    @property
    def remove_data(self):
        """返回删除的数据"""

        return self.data.remove_data

    def convert_index(self,index_list):
        """
        传入结构的原始下标 然后转换成现在已有的
        """
        if isinstance(index_list,int):
            index_list=[index_list]
        return np.where(np.isin(self.group_array.all_data,index_list))[0]



    def remove(self,remove_index):
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

    def get_rmse(self):
        if not self.cols:
            return 0
        return np.sqrt(((self.now_data[:, 0:self.cols] - self.now_data[:, self.cols: ]) ** 2).mean( ))

    def get_formart_rmse(self):
        rmse=self.get_rmse()
        if self.title =="energy":
            unit="meV/atom"
            rmse*=1000
        elif self.title =="force":
            unit="meV/A"
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

    def get_max_error_index(self,nmax):
        """
        返回nmax个最大误差的下标
        这个下标是结构的原始下标
        """
        error = np.sum(np.abs(self.now_data[:, 0:self.cols] - self.now_data[:, self.cols: ]), axis=1)
        rmse_max_ids = np.argsort(-error)
        structure_index =self.group_array.now_data[rmse_max_ids]
        index,indices=np.unique(structure_index,return_index=True)

        return   structure_index[np.sort(indices)][:nmax].tolist()




class NepPlotData(NepData):

    def __init__(self,data_list,**kwargs ):
        super().__init__(data_list,**kwargs )
        self.x_cols=slice(self.cols,None)
        self.y_cols=slice(None,self.cols )
    @property
    def normal_color(self):
        return Brushes.TransparentBrush
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


    @property
    def structure_index(self):
        return self.group_array[ : ].repeat(self.cols)
class DPPlotData(NepData):
    def __init__(self,data_list,**kwargs ):
        super().__init__(data_list,**kwargs )
        self.x_cols=slice(None,self.cols)
        self.y_cols=slice(self.cols,None)
    @property
    def normal_color(self):
        return Brushes.TransparentBrush
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



    def all_x(self):
        if self.cols==0:
            return self.all_data
        return self.all_data[ : ,:self.cols].ravel()
    @property
    def all_y(self):
        if self.cols==0:
            return self.all_data
        return self.all_data[ : , self.cols:].ravel()
    @property
    def structure_index(self):
        return self.group_array[ : ].repeat(self.cols)

class StructureData(NepData):

    @utils.timeit
    def get_all_config(self):

        return [structure.tag for structure in self.now_data]

    def search_config(self,config):

        result_index=[i for i ,structure in enumerate(self.now_data) if re.search(config, structure.tag)]
        return self.group_array[result_index].tolist()


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
        try:
            self.load_structures()
            self._load_descriptors()
            self._load_dataset()
            self.load_flag=True
        except:
            logger.error(traceback.format_exc())

            MessageManager.send_error_message("load dataset error!")

        self.loadFinishedSignal.emit()
    def _load_dataset(self):
        raise NotImplementedError()

    @property
    def dataset(self) -> ["NepPlotData"]:
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

    def select(self,indices):
        """
        传入一个索引列表，将索引对应的结构标记为选中状态
        这个下标是结构在train.xyz中的索引
        """


        # 统一转换为 NumPy 数组
        idx = np.asarray(indices, dtype=int) if not isinstance(indices, int) else np.array([indices])
        # 去重并过滤有效索引（在数据范围内且为活跃数据）
        idx = np.unique(idx)
        idx = idx[(idx >= 0) & (idx < len(self.structure.all_data)) & (self.structure.data._active_mask[idx])]
        # 批量添加到选中集合
        self.select_index.update(idx)

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

    def inverse_select(self):
        """Invert the current selection state of all active structures"""
        active_indices = set(self.structure.data.now_indices.tolist())
        selected_indices = set(self.select_index)
        unselect = list(selected_indices)
        select = list(active_indices - selected_indices)
        if unselect:
            self.uncheck(unselect)
        if select:
            self.select(select)
    def export_selected_xyz(self,save_file_path):
        """
        导出当前选中的结构
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
        """
        self.structure.remove(i)
        for dataset in self.dataset:
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
        for dataset in self.dataset:
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


        if os.path.exists(self.descriptor_path):
            desc_array = read_nep_out_file(self.descriptor_path,dtype=np.float32,ndmin=2)

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
