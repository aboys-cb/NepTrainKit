#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 17:17
# @Author  : 兵
# @email    : 1747193328@qq.com
import json
import os
import re
import sys
import traceback
from PySide6.QtCore import Signal, QObject
from loguru import logger
from qfluentwidgets import MessageBox
from NepTrainKit import utils,module_path,is_nuitka_compiled
from NepTrainKit.core import MessageManager
from NepTrainKit.version import RELEASES_API_URL, __version__, UPDATE_FILE


class UpdateWoker( QObject):
    version=Signal(dict)
    download_success=Signal( )
    def __init__(self,parent):
        self._parent=parent
        super().__init__(parent)
        self.func=self._check_update
        self.version.connect(self._check_update_call_back)
        self.download_success.connect(self._call_restart)
        self.update_thread=utils.LoadingThread(self._parent,show_tip=False)
        self.down_thread=utils.LoadingThread(self._parent,show_tip=True,title="Downloading")

    def download(self,url):
        # 为了做启动加速  在函数内导入requests
        import requests
        resp = requests.get(url, stream=True)
        count = 0
        with open(UPDATE_FILE, "wb") as f:
            for i in resp.iter_content(1024):
                if i:
                    f.write(i)
                    count += len(i)
        self.download_success.emit()

    def _call_restart(self):
        box = MessageBox("Do you want to restart？"  ,
                         "Update package downloaded successfully! Would you like to restart now?\nIf you cancel, the update will be applied automatically the next time you open the software.",
                         self._parent
                         )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        utils.unzip()

    def _check_update(self):
        import requests
        MessageManager.send_info_message("Checking for updates, please wait...")
        try:
            headers={
                "User-Agent": "Awesome-Octocat-App"
            }
            version_info = requests.get(RELEASES_API_URL,headers=headers).json()
            self.version.emit(version_info)
        except:
            logger.error(traceback.format_exc())
            MessageManager.send_error_message("Network error!")

    def _check_update_call_back(self,version_info):
        if "message" in version_info:
            MessageManager.send_warning_message(version_info['message'])
            return
        if version_info['tag_name'][1:] == __version__:
            MessageManager.send_success_message("You are already using the latest version!")
            return
        if not is_nuitka_compiled:
            MessageManager.send_info_message("You can update via pip install NepTrainKit -U")
            return
        box = MessageBox("New version detected:" + version_info["name"] + version_info["tag_name"],
                         version_info["body"],
                         self._parent
                         )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        for assets in version_info["assets"]:

            if sys.platform in assets["name"] and "NepTrainKit" in assets["name"]:
                self.down_thread.start_work(self.download,assets["browser_download_url"])
                return
        MessageManager.send_warning_message("No update package available for your system. Please download it manually!")

    def check_update(self):
        self.update_thread.start_work(self._check_update)


class UpdateNEP89Woker( QObject):
    version=Signal(int)
    download_success=Signal( )
    def __init__(self,parent):
        self._parent=parent
        super().__init__(parent)
        self.func=self._check_update
        self.version.connect(self._check_update_call_back)
        self.update_thread=utils.LoadingThread(self._parent,show_tip=False)
        self.down_thread=utils.LoadingThread(self._parent,show_tip=True,title="Downloading")

    def download(self,latest_date):

        import requests
        raw_url = (
            f"https://raw.githubusercontent.com/brucefan1983/GPUMD/master/"
            f"potentials/nep/nep89_{latest_date}/nep89_{latest_date}.txt"
        )
        resp = requests.get(raw_url, stream=True)
        count = 0
        with open(os.path.join(module_path,"Config/nep89.txt"), "wb") as f:
            for i in resp.iter_content(1024):
                if i:
                    f.write(i)
                    count += len(i)

        MessageManager.send_success_message("Update large model completed!")
        with open(  os.path.join(module_path, "Config/nep.json"), "r") as f:
            local_nep_info=json.load(f)
        local_nep_info["date"]=latest_date
        with open(os.path.join(module_path, "Config/nep.json"), "w") as f:
            json.dump(local_nep_info, f)

    def _check_update(self):
        import requests
        MessageManager.send_info_message("Checking for updates, please wait...")
        api_url = "https://api.github.com/repos/brucefan1983/GPUMD/contents/potentials/nep"
        response = requests.get(api_url)
        if response.status_code != 200:
            MessageManager.send_warning_message(f"Unable to access the warehouse directory, status code: {response.status_code}")
            return
        directories = []
        for item in response.json():
            if item['type'] == 'dir' and item['name'].startswith('nep89_'):
                directories.append(item['name'])

        # 提取日期并找出最新
        date_pattern = re.compile(r'nep89_(\d{8})')
        latest_date = None
        for dir_name in directories:
            match = date_pattern.match(dir_name)
            if match:
                current_date = int(match.group(1))
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date

        self.version.emit(latest_date)

    def _check_update_call_back(self,latest_date):
        with open(  os.path.join(module_path, "Config/nep.json"), "r") as f:
            local_nep_info=json.load(f)
        if local_nep_info["date"] >= latest_date:
            MessageManager.send_success_message("You are already using the latest version!")
            return
        box = MessageBox("New version",f"A new version of the large model has been detected:{latest_date}",
                         self._parent
                         )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        self.down_thread.start_work(self.download,latest_date)
        return

    def check_update(self):
        self.update_thread.start_work(self._check_update)
