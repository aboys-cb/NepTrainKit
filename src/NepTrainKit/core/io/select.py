#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 00:24
# @Author  : 兵
# @email    : 1747193328@qq.com

import numpy as np

def numpy_cdist(X, Y):
    """
    使用 NumPy 计算两个数组之间的成对欧几里得距离

    参数:
        X (numpy.ndarray): 第一个数据集，形状为 (m, d)
        Y (numpy.ndarray): 第二个数据集，形状为 (n, d)

    返回:
        numpy.ndarray: 形状为 (m, n)，每个元素是 X 中的样本与 Y 中的样本之间的距离
    """
    # 计算每个点与每个点之间的差的平方
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]

    # 计算差的平方和
    squared_dist = np.sum(np.square(diff), axis=2)

    # 返回距离（平方根）
    return np.sqrt(squared_dist)

def farthest_point_sampling(points, n_samples, min_dist=0.1, selected_data=None):
    """
    最远点采样：支持已有样本扩展，并加入最小距离限制。

    参数:
        points (ndarray): 点集，形状为 (N, D)。
        n_samples (int): 最大采样点的数量。
        min_dist (float): 最小距离阈值。
        initial_indices (list or None): 已选择的样本索引列表，默认无。

    返回:
        sampled_indices (list): 采样点的索引。
    """
    n_points = points.shape[0]

    if isinstance(selected_data, np.ndarray) and selected_data.size == 0:
        selected_data=None
    # 初始化采样点列表
    sampled_indices = []

    # 如果已有采样点，则计算到所有点的最小距离
    if selected_data is not None :
        # 使用 cdist 计算已有点与所有点之间的距离，返回形状为 (n_points, len(sampled_indices)) 的矩阵
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)  # 每个点到现有采样点集的最小距离

    else:
        # 如果没有初始点，则随机选择一个作为第一个点
        first_index = 0
        sampled_indices.append(first_index)
        # 计算所有点到第一个点的距离
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

    # 进行最远点采样
    while len(sampled_indices) < n_samples:
        # 找到距离采样点集最远的点
        current_index = np.argmax(min_distances)

        # 如果没有点能满足最小距离要求，则提前终止
        if min_distances[current_index] < min_dist:
            break

        # 添加当前点到采样集
        sampled_indices.append(int(current_index))

        # 更新最小距离：仅计算当前点到新选择点的距离
        # 获取当前点到所有其他点的距离
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)

        # 更新每个点到现有样本点集的最小距离
        min_distances = np.minimum(min_distances, new_distances)
    return sampled_indices



