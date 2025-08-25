#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 21:08
# @Author  : 兵
# @email    : 1747193328@qq.com
# @File    : https://github.com/aboys-cb/NepTrainKit/blob/master/tools/dimensionality_reduction.py
import numpy as np
from sklearn.decomposition import PCA  # pyright:ignore
from sklearn.preprocessing import StandardScaler  # pyright:ignore
from umap import UMAP # pyright:ignore
import matplotlib.pyplot as plt # pyright:ignore
import argparse



def perform_pca(data, n_components=2 ):
    """执行PCA降维"""

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca


def perform_umap(data, n_components=2 ):
    """执行UMAP降维"""
    umap = UMAP(n_components=n_components )
    reduced_data = umap.fit_transform(data)
    return reduced_data, umap


def plot_results(reduced_data, output_file=None):
    """可视化降维结果"""
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Dimensionality Reduction Result')


    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")






def main():
    parser = argparse.ArgumentParser(description='Descriptor Data Dimensionality Reduction Tool')
    parser.add_argument( 'input', help='Input descriptor data file')
    parser.add_argument('-o', '--output',  default='reduced_descriptor.out', help='Output file for reduced data')
    parser.add_argument('-m', '--method', choices=['pca', 'umap'], default='pca',
                        help='Dimensionality reduction method')
    parser.add_argument('-n', '--n_components', type=int, default=2, help='Number of components to keep')

    args = parser.parse_args()

    # 加载数据
    print(f"Loading data from {args.input}...")
    descriptior_data = np.loadtxt(args.input)


    # 执行降维
    print(f"Performing {args.method.upper()} dimensionality reduction...")
    if args.method == 'pca':
        reduced_data, model = perform_pca(descriptior_data )
    elif args.method == 'umap':
        reduced_data, model = perform_umap(descriptior_data )
    else:
        return
    plot_results(reduced_data, "dimensionality_reduction.png")
    # 保存结果
    np.savetxt(args.output, reduced_data, fmt='%.10g')
    print(f"Reduced data saved to {args.output}")




if __name__ == '__main__':
    main()