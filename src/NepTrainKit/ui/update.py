#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 17:17
# @Author  : 
# @email    : 1747193328@qq.com
"""Qt helpers for checking NepTrainKit updates and downloading assets."""

import json
import re
import sys
import traceback

from PySide6.QtCore import Signal, QObject
from loguru import logger
from qfluentwidgets import MessageBox

from NepTrainKit import is_nuitka_compiled, module_path
from NepTrainKit.core import MessageManager
from NepTrainKit.paths import as_path
from NepTrainKit.version import RELEASES_API_URL, __version__, UPDATE_FILE

from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.ui.updater import unzip


class UpdateWoker(QObject):
    """Worker that checks for new application releases and applies updates."""

    version = Signal(dict)
    download_success = Signal()

    def __init__(self, parent):
        """Initialise the worker with the owning ``parent`` widget."""
        self._parent = parent
        super().__init__(parent)
        self.func = self._check_update
        self.version.connect(self._check_update_call_back)
        self.download_success.connect(self._call_restart)
        self.update_thread = LoadingThread(self._parent, show_tip=False)
        self.down_thread = LoadingThread(self._parent, show_tip=True, title="Downloading")

    def download(self, url: str) -> None:
        """Download the update archive from ``url`` and emit completion."""
        import requests  # Imported lazily to keep startup fast.

        response = requests.get(url, stream=True)
        update_path = as_path(UPDATE_FILE)
        with update_path.open("wb") as target:
            for chunk in response.iter_content(1024):
                if chunk:
                    target.write(chunk)
        self.download_success.emit()

    def _call_restart(self) -> None:
        """Prompt the user to relaunch once the update archive is ready."""
        box = MessageBox(
            "Do you want to restart?",
            (
                "Update package downloaded successfully! Would you like to restart now?\n"
                "If you cancel, the update will be applied automatically the next time you open the software."
            ),
            self._parent,
        )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        unzip()

    def _check_update(self) -> None:
        """Query GitHub releases and emit the response payload."""
        import requests

        MessageManager.send_info_message("Checking for updates, please wait...")
        try:
            headers = {"User-Agent": "Awesome-Octocat-App"}
            version_info = requests.get(RELEASES_API_URL, headers=headers).json()
            self.version.emit(version_info)
        except Exception:  # noqa: BLE001 - we log the traceback for user support.
            logger.error(traceback.format_exc())
            MessageManager.send_error_message("Network error!")

    def _check_update_call_back(self, version_info: dict) -> None:
        """Handle the GitHub API response and notify the user."""
        if "message" in version_info:
            MessageManager.send_warning_message(version_info["message"])
            return
        if version_info["tag_name"][1:] == __version__:
            MessageManager.send_success_message("You are already using the latest version!")
            return

        box = MessageBox(
            "New version detected:" + version_info["name"] + version_info["tag_name"],
            version_info["body"],
            self._parent,
        )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        for assets in version_info["assets"]:
            if sys.platform in assets["name"] and "NepTrainKit" in assets["name"]:
                self.down_thread.start_work(self.download, assets["browser_download_url"])
                return
        MessageManager.send_warning_message("No update package available for your system. Please download it manually!")

    def check_update(self) -> None:
        """Start the background task if running from a packaged build."""
        if not is_nuitka_compiled:
            MessageManager.send_info_message("You can update via pip install NepTrainKit -U --pre")
            return
        self.update_thread.start_work(self._check_update)


class UpdateNEP89Woker(QObject):
    """Worker that keeps the bundled ``nep89`` potential file up to date."""

    version = Signal(int)
    download_success = Signal()

    def __init__(self, parent):
        """Initialise the worker with the owning ``parent`` widget."""
        self._parent = parent
        super().__init__(parent)
        self.func = self._check_update
        self.version.connect(self._check_update_call_back)
        self.update_thread = LoadingThread(self._parent, show_tip=False)
        self.down_thread = LoadingThread(self._parent, show_tip=True, title="Downloading")

    def download(self, latest_date: int) -> None:
        """Download the latest ``nep89`` model and refresh metadata."""
        import requests

        raw_url = (
            "https://raw.githubusercontent.com/brucefan1983/GPUMD/master/"
            f"potentials/nep/nep89_{latest_date}/nep89_{latest_date}.txt"
        )
        response = requests.get(raw_url, stream=True)
        nep89_path = module_path/ "Config/nep89.txt"
        with nep89_path.open("wb") as target:
            for chunk in response.iter_content(1024):
                if chunk:
                    target.write(chunk)

        MessageManager.send_success_message("Update large model completed!")
        nep_json_path = module_path/ "Config/nep.json"
        with nep_json_path.open("r", encoding="utf-8") as config_file:
            local_nep_info = json.load(config_file)
        local_nep_info["date"] = latest_date
        with nep_json_path.open("w", encoding="utf-8") as config_file:
            json.dump(local_nep_info, config_file)

    def _check_update(self) -> None:
        """Check the remote repository for a newer ``nep89`` dataset."""
        import requests

        MessageManager.send_info_message("Checking for updates, please wait...")
        api_url = "https://api.github.com/repos/brucefan1983/GPUMD/contents/potentials/nep"
        response = requests.get(api_url)
        if response.status_code != 200:
            MessageManager.send_warning_message(
                f"Unable to access the warehouse directory, status code: {response.status_code}"
            )
            return
        directories = [item["name"] for item in response.json() if item["type"] == "dir" and item["name"].startswith("nep89_")]

        date_pattern = re.compile(r"nep89_(\d{8})")
        latest_date: int | None = None
        for dir_name in directories:
            match = date_pattern.match(dir_name)
            if match:
                current_date = int(match.group(1))
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date

        self.version.emit(latest_date)

    def _check_update_call_back(self, latest_date: int) -> None:
        """Prompt the user to download the updated ``nep89`` archive."""
        nep_json_path = module_path/ "Config/nep.json"
        with nep_json_path.open("r", encoding="utf-8") as config_file:
            local_nep_info = json.load(config_file)
        if local_nep_info["date"] >= latest_date:
            MessageManager.send_success_message("You are already using the latest version!")
            return
        box = MessageBox(
            "New version",
            f"A new version of the large model has been detected:{latest_date}",
            self._parent,
        )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        self.down_thread.start_work(self.download, latest_date)

    def check_update(self) -> None:
        """Start checking for ``nep89`` updates in the background."""
        self.update_thread.start_work(self._check_update)

