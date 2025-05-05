#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:14
# @Author  : 兵
# @email    : 1747193328@qq.com
import platform
import subprocess
import time
import os
import traceback
from collections.abc import Iterable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QApplication
from loguru import logger
from qfluentwidgets import StateToolTip

from NepTrainKit.core import Config
from NepTrainKit.version import UPDATE_EXE, UPDATE_FILE, NepTrainKit_EXE


def timeit(func):
    """
    统计函数耗时
    使用用法：
        @timeit
        def demo():
            pass
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原始函数
        end_time = time.time()  # 记录结束时间
        logger.debug(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def check_path_type(path:str) -> str:
    """
    判断路径是文件夹还是文件，即使路径不存在。

    参数:
        path (str): 路径字符串。

    返回:
        str: "folder"（文件夹）、"file"（文件） 。
    """

    if os.path.isdir(path):
        return "folder"
    elif os.path.isfile(path):
        return "file"
    else:
        # 如果路径不存在，进一步检查是否有文件扩展名
        if os.path.splitext(path)[1]:  # 如果有扩展名，可能是文件
            return "file"
        else:  # 否则可能是文件夹
            return "folder"


def call_path_dialog(self,
                     title:str,
                     dialog_type:str = "file",
                     default_filename:str = "",
                     file_filter:str = "",
                     selected_filter:str = "") -> str|None:
    dialog_map = {
        "file": lambda: QFileDialog.getSaveFileName(self, title, os.path.join(Config.get_path(), default_filename), file_filter, selected_filter),
        "select": lambda: QFileDialog.getOpenFileName(self, title, Config.get_path(), file_filter),
        "selects": lambda: QFileDialog.getOpenFileNames(self, title, Config.get_path(), file_filter),
        "directory": lambda: QFileDialog.getExistingDirectory(self, title, Config.get_path())
    }

    dialog_func = dialog_map.get(dialog_type)
    if not dialog_func:
        return None

    select_path = dialog_func()

    if isinstance(select_path, tuple):
        select_path = select_path[0]  # 处理 `getSaveFileName` 和 `getOpenFileName` 返回的 tuple
    elif isinstance(select_path, list):
        if not select_path:
            return None
        select_path = select_path[0]  # `getOpenFileNames` 返回 list

    if not select_path:
        return None

    # 提取目录并保存到配置
    if isinstance(select_path,list):
        path = select_path[0]
    else:
        path = select_path
    if check_path_type(path)=="file":
        last_dir = os.path.dirname(path)
    else:
        last_dir = path

    Config.set("setting", "last_path", last_dir)
    return select_path

def unzip():

    cmd = f"ping -n 3 127.0.0.1&{UPDATE_EXE} {UPDATE_FILE}&ping -n 2 127.0.0.1&start {NepTrainKit_EXE}"

    subprocess.Popen(cmd, shell=True)
    if QApplication.instance():
        QApplication.instance().exit()
    else:
        quit()

class LoadingThread(QThread):
    progressSignal = Signal(int)
    def __init__(self, parent=None, show_tip=True, title='running'):
        super(LoadingThread,self).__init__(parent)
        self.show_tip=show_tip
        self.title=title
        self._parent=parent

    def run(self ):
        result =self._func(*self._args, **self._kwargs)
        if isinstance(result, Iterable):
            for i,_ in enumerate(result):
                self.progressSignal.emit(i)

    def start_work(self,func,*args,**kwargs):
        if self.show_tip:
            self.tip = StateToolTip(self.title, 'Please wait patiently~~', self._parent)
            self.tip.show()
            self.finished.connect(self.__finished_work)
            self.tip.closedSignal.connect(self.quit)
        else:
            self.tip=None
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self.start()

    def __finished_work(self ):
        if self.tip:
            self.tip.setContent('任务完成啦 😆')
            self.tip.setState(True)

    def stop_work(self ):
        self.terminate()




class DataProcessingThread(QThread):
    # 定义信号用于通信
    progressSignal = Signal(int)  # 进度更新信号
    finishSignal = Signal()  # 处理完成信号
    errorSignal = Signal(str)  # 错误信号

    def __init__(self, dataset, process_func):
        super().__init__()
        self.dataset = dataset
        self.process_func = process_func
        self.result_dataset = []

    def run(self):
        """线程主逻辑"""
        try:
            total = len(self.dataset)
            self.progressSignal.emit(0)
            for index, structure in enumerate(self.dataset):
                # 处理每个结构
                processed = self.process_func(structure)

                self.result_dataset.extend(processed)

                # 发射进度信号 (百分比)
                self.progressSignal.emit(int((index + 1) / total * 100))

            # 处理完成
            self.finishSignal.emit( )
        except Exception as e:
            logger.debug(traceback.format_exc())
            self.errorSignal.emit(str(e))


class FillterProcessingThread(QThread):
    # 定义信号用于通信
    progressSignal = Signal(int)  # 进度更新信号
    finishSignal = Signal()  # 处理完成信号
    errorSignal = Signal(str)  # 错误信号

    def __init__(self,  process_func):
        super().__init__()
        self.process_func = process_func

    def run(self):
        """线程主逻辑"""
        try:
            self.progressSignal.emit(0)
            # 处理每个结构
            self.process_func()
                # 发射进度信号 (百分比)
            self.progressSignal.emit(100)
            # 处理完成
            self.finishSignal.emit( )
        except Exception as e:
            logger.debug(traceback.format_exc())
            self.errorSignal.emit(str(e))