#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 17:14
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import re
from functools import partial

import numpy as np
from loguru import logger


def read_nep_in(  file_name):
    run_in={}

    if  not os.path.exists(file_name):
        return run_in
    with open(file_name, 'r', encoding="utf8") as f:

        groups = re.findall("^([A-Za-z_]+)\s+([^\#\n]*)", f.read(), re.MULTILINE)

        for group in groups:

            run_in[group[0].strip()] = group[1].strip()

    return run_in

def check_fullbatch(run_in,structure_num):

    if run_in.get("prediction")=="1":
        return True
    if int(run_in.get("batch",1000))>=structure_num:
        return True
    return False

def read_nep_out_file(file_path,**kwargs):

    logger.info("Reading file: {}".format(file_path))
    if os.path.exists(file_path):
        data = np.loadtxt(file_path,**kwargs)
        return data
    else:
        return np.array([])

def parse_array_by_atomnum(array,atoms_num_list,map_func=np.linalg.norm,axis=0):
    """
    根据一个映射列表，将原数组按照原子数列表拆分，
    这个主要是处理文件中原子数不一致的情况，比如力 描述符等文件是按照原子数的 把他们转换成结构的
    :param array: 原数组
    :param atoms_num_list: 原子数列表
    :param map_func: 需要对每个结构的数据进行处理的函数 比如求平均 求和等
    :param axis: 映射轴
    :return: 映射后的数组

    """
    if len(array)==0:
        return array
    # 使用 np.cumsum() 计算每个分组的结束索引
    split_indices = np.cumsum(atoms_num_list)[:-1]
    # 使用 np.split() 按照分组拆分数组
    split_arrays = np.split(array, split_indices)
    func = partial(map_func, axis=axis)

    # 对每个分组求和，使用 np.vectorize 进行向量化
    new_array = np.array(list(map(func, split_arrays)))
    return new_array

def get_nep_type(file_path):
    """
    根据nep.txt 判断势函数类别
    """
    nep_type_to_model_type = {
        "nep3": 0,
        "nep3_zbl": 0,
        "nep3_dipole": 1,
        "nep3_polarizability": 2,
        "nep4": 0,
        "nep4_zbl": 0,
        "nep4_dipole": 1,
        "nep4_polarizability": 2,
        "nep5": 0,
        "nep5_zbl": 0
    }
    model_type=0
    try:
        with open(file_path, 'r') as file:
            # 读取第一行
            first_line = file.readline().strip()
            parts = first_line.split()
            nep_type=parts[0]
            model_type = nep_type_to_model_type.get(nep_type )
    except FileNotFoundError:
        logger.warning(f"Error: File {file_path} not found. Default model_type is 0")
    except Exception as e:
        logger.warning(f"An error occurred while parsing the file: {e}")

    return model_type

