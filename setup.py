#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:22
# @Author  : 兵
# @email    : 1747193328@qq.com
"""NepTrainKit package configuration."""

from __future__ import annotations
import tempfile

import platform
import sys

import os
import subprocess
import pybind11
from pybind11.setup_helpers import Pybind11Extension
from setuptools import  Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# 获取 pybind11 的 include 路径
pybind11_include = pybind11.get_include()



# 设定编译选项

extra_link_args = [ ]
extra_compile_args=[ ]
# 检查平台并设置相应的 OpenMP 编译标志

if sys.platform == "win32":
    # 对于 Windows 使用 MSVC 编译器时，需要使用 /openmp
    extra_compile_args.append('/openmp' )
    extra_compile_args.append('/O2' )
    extra_compile_args.append('/std:c++11' )



    extra_link_args.append('/openmp')
    extra_link_args.append('/O2' )
    extra_link_args.append('/std:c++11' )


elif sys.platform == "darwin":
    # 对于 macOS 和 Clang 使用 -fopenmp 编译标志
    # Clang 好像不支持openmp 先注释掉
    # extra_compile_args.append('-fopenmp' )
    #
    # extra_link_args.append('-fopenmp')
    # 通过环境变量获取目标架构，默认为 arm64（Apple Silicon）
    target_arch = os.environ.get('ARCHFLAGS', '-arch arm64').split()[-1]
    # extra_compile_args.append(f'-arch {target_arch}')
    # extra_link_args.append(f'-arch {target_arch}')

    extra_compile_args.append('-O3')
    extra_compile_args.append('-std=c++11')


    extra_link_args.append('-O3')
    extra_link_args.append('-std=c++11')

    omp_include = os.getenv("OMP_INCLUDE_PATH", "/opt/homebrew/opt/libomp/include")
    omp_lib = os.getenv("OMP_LIB_PATH", "/opt/homebrew/opt/libomp/lib")
    extra_compile_args.extend(["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"])
    extra_link_args.extend(["-lomp", f"-L{omp_lib}"])

    pass

else:
    # 对于 Linux 和 GCC 使用 -fopenmp 编译标志

    extra_compile_args.append('-fopenmp' )
    extra_compile_args.append('-O3')
    extra_compile_args.append('-std=c++11')


    extra_link_args.append('-fopenmp')
    extra_link_args.append('-O3')
    extra_link_args.append('-std=c++11')


# 定义扩展模块
ext_modules = [
    Extension(
        "NepTrainKit.nep_cpu",  # 模块名
        ["src/nep_cpu/nep_cpu.cpp"],  # 源文件
        include_dirs=[
            pybind11_include,
            # "src/nep_cpu"
        ],
        extra_compile_args=extra_compile_args,  # 编译选项
        extra_link_args=extra_link_args,
        language="c++",  # 指定语言为 C++
    )
]

# 自定义 build_ext 命令，确保兼容性
class BuildExt(build_ext):
    def build_extensions(self):
        # 设置编译器标准为 C++17
        ct = self.compiler.compiler_type
        opts = [ ]
        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
        try:
            # 尝试构建扩展模块
            build_ext.build_extensions(self)
        except Exception as e:
            # 捕获编译错误并打印警告
            print(f"WARNING: Failed to build extension module: {e}")
            print("WARNING: Skipping nep_cpu module build. The package will be installed without it.")
            # 清空 ext_modules，跳过扩展模块的构建
            self.ext_modules = []


setup(
    author="Chen Cheng bing",
cmdclass={'build_ext': BuildExt},
    # include_dirs=[np.get_include()],
ext_modules=ext_modules,
zip_safe=False,
)