#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 12:23
# @Author  : 兵 + Grok
# @email    : 1747193328@qq.com
# @File    : https://github.com/aboys-cb/NepTrainKit/blob/master/tools/plot_descriptor.py
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
import time

# 选择计算方式：'dbscan' 或 'alphashape'
method = 'alphashape'

start_time = time.time()
config = [
    #(描述符路径, 图例标签, 标记符号)
    ("./cubic_descriptor.out", "cubic", "*"),
    ("./tetragonal_descriptor.out", "tetragonal", "s"),
    ("./orthorhombic_descriptor.out", "orthorhombic", "o")
]

fig, ax = plt.subplots()

all_weights = []
for file, _, _ in config:
    descriptor_array = np.loadtxt(file)
    all_weights.extend(descriptor_array[:, 2])

vmin, vmax = np.min(all_weights), np.max(all_weights)

colors = ['blue', 'orange', 'green']


# DBSCAN 方法
def compute_dbscan_boundary(points, color):
    from sklearn.cluster import DBSCAN # pyright:ignore
    from scipy.spatial import ConvexHull
    if len(points) > 1:
        db = DBSCAN(eps=0.1, min_samples=3).fit(points)  # 调整 eps 和 min_samples
        labels = db.labels_

        # 遍历每个簇
        for cluster in np.unique(labels):
            if cluster == -1:  # 跳过噪声点
                continue
            cluster_points = points[labels == cluster]

            # 计算簇的边界（使用凸包）
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                boundary_points = cluster_points[hull.vertices]

                # 样条插值平滑边界
                tck, u = splprep([boundary_points[:, 0], boundary_points[:, 1]], s=0, per=1)
                smooth_x, smooth_y = splev(np.linspace(0, 1, 100), tck)
                ax.plot(smooth_x, smooth_y, color=color, linestyle='-', linewidth=1)
            else:
                print(f"Skipping cluster in DBSCAN: insufficient points")


# Alpha Shape 方法
def compute_alphashape_boundary(points, color):
    import alphashape # pyright:ignore

    if len(points) >= 3:
        alpha = 1.0  # 调整 alpha 值
        hull = alphashape.alphashape(points, alpha)
        if hull.geom_type == 'Polygon':
            x, y = hull.exterior.xy
            # 样条插值平滑边界
            tck, u = splprep([x, y], s=0, per=1)
            smooth_x, smooth_y = splev(np.linspace(0, 1, 100), tck)
            ax.plot(smooth_x, smooth_y, color=color, linestyle='-', linewidth=1)
        else:
            print(f"Skipping boundary in Alpha Shape: complex geometry")


# 主绘图逻辑
scatter=None
for idx, (file, label, marker) in enumerate(config):
    descriptor_array = np.loadtxt(file)

    # 绘制散点图
    scatter = ax.scatter(descriptor_array[:, 0], descriptor_array[:, 1],
                         c=descriptor_array[:, 2],
                         cmap='viridis',
                         vmin=vmin, vmax=vmax,
                         label=label,s=5,
                         marker=marker)

    # 根据 method 选择计算方式
    points = descriptor_array[:, 0:2]
    if method == 'dbscan':
        compute_dbscan_boundary(points, colors[idx % len(colors)])
    elif method == 'alphashape':
        compute_alphashape_boundary(points, colors[idx % len(colors)])
    else:
        raise ValueError("Invalid method. Use 'dbscan' or 'alphashape'.")

ax.legend(frameon=False)
if scatter is not None:
    fig.colorbar(scatter, ax=ax, label='Energy Per Atom')

ax.set_xlabel('Descriptor 1')
ax.set_ylabel('Descriptor 2')
ax.set_title(f'Descriptor Scatter Plot with {method.capitalize()} Boundaries')
plt.tight_layout()
plt.savefig("descriptor_scatter_plot.png",dpi=300)

print(f"Execution time: {time.time() - start_time:.2f} seconds")