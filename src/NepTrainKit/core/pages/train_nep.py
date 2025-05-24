#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/06/28
# @Author  : 兵
# @email    : 1747193328@qq.com

import os
from typing import Tuple, List

import paramiko
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox)


class TrainNepWidget(QWidget):
    """Widget for running NEP training with remote execution support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainNepWidget")
        self.ssh = None
        self.sftp = None
        self.channel = None
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.update_plot)
        self._init_ui()

    def _init_ui(self):
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.local_path_edit = QLineEdit(self)
        self.remote_path_edit = QLineEdit(self)
        self.host_edit = QLineEdit(self)
        self.port_spin = QSpinBox(self)
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.user_edit = QLineEdit(self)
        self.pass_edit = QLineEdit(self)
        self.pass_edit.setEchoMode(QLineEdit.Password)

        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        self.step_label = QLabel("Step: 0", self)
        self.progress_label = QLabel("", self)

        self.layout.addWidget(QLabel("Local Path:"), 0, 0)
        self.layout.addWidget(self.local_path_edit, 0, 1, 1, 3)
        self.layout.addWidget(QLabel("Remote Path:"), 1, 0)
        self.layout.addWidget(self.remote_path_edit, 1, 1, 1, 3)
        self.layout.addWidget(QLabel("Host:"), 2, 0)
        self.layout.addWidget(self.host_edit, 2, 1)
        self.layout.addWidget(QLabel("Port:"), 2, 2)
        self.layout.addWidget(self.port_spin, 2, 3)
        self.layout.addWidget(QLabel("Username:"), 3, 0)
        self.layout.addWidget(self.user_edit, 3, 1)
        self.layout.addWidget(QLabel("Password:"), 3, 2)
        self.layout.addWidget(self.pass_edit, 3, 3)
        self.layout.addWidget(self.start_button, 4, 0, 1, 2)
        self.layout.addWidget(self.stop_button, 4, 2, 1, 2)

        self.loss_plot = pg.PlotWidget(self)
        self.loss_plot.setLabel("left", "loss")
        self.loss_plot.setLabel("bottom", "step")
        self.diag_plot = pg.PlotWidget(self)
        self.diag_plot.setLabel("left", "diag diff")
        self.diag_plot.setLabel("bottom", "step")

        self.layout.addWidget(self.loss_plot, 5, 0, 1, 4)
        self.layout.addWidget(self.diag_plot, 6, 0, 1, 4)
        self.layout.addWidget(self.step_label, 7, 0, 1, 2)
        self.layout.addWidget(self.progress_label, 7, 2, 1, 2)

        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)

        self.loss_curve = self.loss_plot.plot([], [], pen="b")
        self.diag_curve = self.diag_plot.plot([], [], pen="r")

        self.setLayout(self.layout)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            hostname=self.host_edit.text(),
            port=self.port_spin.value(),
            username=self.user_edit.text(),
            password=self.pass_edit.text(),
        )
        self.sftp = self.ssh.open_sftp()

    def _start_remote_cmd(self):
        if not self.ssh:
            return
        cmd = f"cd {self.remote_path_edit.text()} && ./train.sh"
        transport = self.ssh.get_transport()
        self.channel = transport.open_session()
        self.channel.exec_command(cmd)

    def start_training(self):
        if self.host_edit.text():
            self._connect()
            self._start_remote_cmd()
        self.timer.start()

    def stop_training(self):
        self.timer.stop()
        if self.channel:
            self.channel.close()
            self.channel = None
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None

    # ------------------------------------------------------------------
    # Update plot
    # ------------------------------------------------------------------
    def update_plot(self):
        local_loss = os.path.join(self.local_path_edit.text(), "loss.out")
        remote_loss = os.path.join(self.remote_path_edit.text(), "loss.out")
        if self.sftp and self.remote_path_edit.text():
            try:
                self.sftp.get(remote_loss, local_loss)
            except Exception:
                pass
        if not os.path.exists(local_loss):
            return
        steps, loss, diag = self.parse_loss_file(local_loss)
        if not steps:
            return
        self.loss_curve.setData(steps, loss)
        if diag:
            self.diag_curve.setData(steps[: len(diag)], diag)
        self.step_label.setText(f"Step: {steps[-1]}")
        self.progress_label.setText(f"{len(steps)} steps")

    @staticmethod
    def parse_loss_file(path: str) -> Tuple[List[int], List[float], List[float]]:
        steps: List[int] = []
        loss: List[float] = []
        diag: List[float] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    try:
                        step = int(parts[0])
                        l = float(parts[1])
                    except ValueError:
                        continue
                    steps.append(step)
                    loss.append(l)
                    if len(parts) >= 4:
                        try:
                            d = float(parts[2]) - float(parts[3])
                            diag.append(d)
                        except ValueError:
                            pass
        except OSError:
            pass
        return steps, loss, diag

    def closeEvent(self, event):
        self.stop_training()
        super().closeEvent(event)
