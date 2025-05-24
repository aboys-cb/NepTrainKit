#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 15:31
# @Author  : 兵
# @email    : 1747193328@qq.com

import re
from functools import cached_property
import numpy as np
from NepTrainKit import utils
from NepTrainKit.core.types import Brushes


class DataBase:
    """
    对列表进行封装 比如结构训练集 能量 力
    以下功能要实现：
        1.根据索引删除指定结构
        2.支持回退（记录删除的data）

    """

    def __init__(self, data_list):
        """Initialize the database with a numpy array."""
        self._data = np.asarray(data_list)
        # active indices of ``_data``
        self._active_indices = list(range(len(self._data)))
        # indices that have been removed in order of deletion
        self._removed_indices = []
        # number of indices removed in each remove operation
        self._history = []

    @property
    def num(self) -> int:
        """
        返回当前数组的数量
        """
        return len(self._active_indices)

    @property
    def now_data(self):
        """Return the currently active data."""
        if len(self._active_indices) == 0:
            # Ensure correct shape when empty
            return self._data[:0]
        return self._data[self._active_indices]

    @property
    def remove_data(self):
        """Return all removed data."""
        if len(self._removed_indices) == 0:
            return self._data[:0]
        return self._data[self._removed_indices]
    def remove(self, i):
        """Remove items by indices and record their positions."""
        if self.num == 0:
            return
        if isinstance(i, int):
            idx_list = [i]
        else:
            idx_list = list(i)
        if not idx_list:
            return

        idx_list = sorted(idx_list)
        removed = [self._active_indices[idx] for idx in idx_list]
        for idx in reversed(idx_list):
            self._active_indices.pop(idx)
        self._removed_indices.extend(removed)
        self._history.append(len(removed))


    def __getitem__(self, item):
        """Direct indexing on the active dataset."""
        return self.now_data[item]

    def revoke(self):
        """Restore the last removed items."""
        if self._history:
            last_count = self._history.pop()
            restored = self._removed_indices[-last_count:]
            del self._removed_indices[-last_count:]
            self._active_indices.extend(restored)


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
    def remove_data(self):
        """返回删除的数据"""

        return self.data.remove_data

    def convert_index(self,index_list):
        """
        传入结构的原始下标 然后转换成现在已有的
        """
        if isinstance(index_list,int):
            index_list=[index_list]
        return np.where(np.isin(self.group_array.now_data,index_list))[0]



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
        else:
            return ""
        return f"{rmse:.2f}{unit}"

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

    @property
    def normal_color(self):
        return Brushes.TransparentBrush
    @property
    def x(self):
        if self.cols==0:
            return self.now_data
        return self.now_data[ : ,self.cols:].ravel()
    @property
    def y(self):
        if self.cols==0:
            return self.now_data
        return self.now_data[ : , :self.cols].ravel()
    @property
    def structure_index(self):
        return self.group_array[ : ].repeat(self.cols)


class StructureData(NepData):

    @utils.timeit
    def get_all_config(self):

        return [structure.additional_fields["Config_type"]  for structure in self.now_data]

    def search_config(self,config):

        result_index=[i for i ,structure in enumerate(self.now_data) if re.search(config, structure.additional_fields["Config_type"])]
        return self.group_array[result_index].tolist()