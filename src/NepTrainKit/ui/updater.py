#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess

from PySide6.QtWidgets import QApplication

from NepTrainKit.version import UPDATE_EXE, UPDATE_FILE, NepTrainKit_EXE


def unzip() -> None:
    """Run the platform-specific updater and relaunch the application."""
    cmd = f"ping -n 3 127.0.0.1&{UPDATE_EXE} {UPDATE_FILE}&ping -n 2 127.0.0.1&start {NepTrainKit_EXE}"
    subprocess.Popen(cmd, shell=True)
    if QApplication.instance():
        QApplication.instance().exit()  # pyright:ignore
    else:
        quit()


__all__ = ['unzip']

